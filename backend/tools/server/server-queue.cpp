#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

int server_queue::mPost(server_task &&pTask, bool pFront) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  GGML_ASSERT(pTask.vId != -1);
  // if this is cancel task make sure to clean up pending tasks
  if (pTask.vType == cServerTaskTypeCancel) {
    mCleanupPendingTask(pTask.vIdTarget);
  }
  const int cTaskId = pTask.vId;
  QUE_DBG("new task, id = %d, front = %d\n", cTaskId, pFront);
  if (pFront) {
    lQueueTasks.push_front(std::move(pTask));
  } else {
    lQueueTasks.push_back(std::move(pTask));
  }
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
  return cTaskId;
}

int server_queue::mPost(std::vector<server_task> &&pTasks, bool pFront) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  for (auto &vTask : pTasks) {
    if (vTask.vId == -1) {
      vTask.vId = vId++;
    }
    // if this is cancel task make sure to clean up pending tasks
    if (vTask.vType == cServerTaskTypeCancel) {
      mCleanupPendingTask(vTask.vIdTarget);
    }
    QUE_DBG("new task, id = %d/%d, front = %d\n", vTask.vId, (int)pTasks.size(), pFront);
    if (pFront) {
      lQueueTasks.push_front(std::move(vTask));
    } else {
      lQueueTasks.push_back(std::move(vTask));
    }
  }
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
  return 0;
}

void server_queue::mDefer(server_task &&pTask) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  QUE_DBG("defer task, id = %d\n", pTask.vId);
  lQueueTasksDeferred.push_back(std::move(pTask));
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
}

int server_queue::mGetNewId() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  int vNewId = vId++;
  return vNewId;
}

void server_queue::mPopDeferredTask(int pIdSlot) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  if (!lQueueTasksDeferred.empty()) {
    // try to find a task that uses the specified slot
    bool vFound = false;
    for (auto vIt = lQueueTasksDeferred.begin(); vIt != lQueueTasksDeferred.end(); ++vIt) {
      if (vIt->vIdSlot == pIdSlot) {
        QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", pIdSlot, vIt->vId);
        lQueueTasks.emplace_front(std::move(*vIt));
        lQueueTasksDeferred.erase(vIt);
        vFound = true;
        break;
      }
    }
    // if not tasks found using the slot, just pop the first deferred task (default behavior)
    if (!vFound) {
      QUE_DBG("pop deferred task, id_task = %d\n", lQueueTasksDeferred.front().vId);
      lQueueTasks.emplace_front(std::move(lQueueTasksDeferred.front()));
      lQueueTasksDeferred.pop_front();
    }
  }
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
}

void server_queue::mWaitUntilNoSleep() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  if (!vSleeping) {
    return;
  } else {
    if (!vReqStopSleeping) {
      QUE_DBG("%s", "requesting to stop sleeping\n");
      vReqStopSleeping = true;
      vConditionTasks.notify_one(); // only main thread is waiting on this
    }
    QUE_DBG("%s", "waiting until no sleep\n");
    vConditionTasks.wait(vLock, [&] { return !vSleeping; });
  }
}

void server_queue::mTerminate() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  vRunning = false;
  vConditionTasks.notify_all();
}

void server_queue::mStartLoop(int64_t pIdleSleepMs) {
  vRunning = true;
  vTimeLastTask = ggml_time_ms();

  constexpr auto cMaxWaitTime = std::chrono::seconds(1);
  auto vShouldSleep = [&]() -> bool {
    // caller must hold mutex_tasks
    if (pIdleSleepMs < 0) {
      return false;
    }
    int64_t vNow = ggml_time_ms();
    return (vNow - vTimeLastTask) >= pIdleSleepMs;
  };

  while (true) {
    QUE_DBG("%s", "processing new tasks\n");

    while (true) {
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      if (!vRunning) {
        QUE_DBG("%s", "terminate\n");
        return;
      }
      if (lQueueTasks.empty()) {
        vLock.unlock();
        break;
      }
      server_task vTask = std::move(lQueueTasks.front());
      lQueueTasks.pop_front();
      vLock.unlock();

      QUE_DBG("processing task, id = %d\n", vTask.vId);
      vCallbackNewTask(std::move(vTask));
    }
    // all tasks in the current loop is processed, slots data is now ready
    QUE_DBG("%s", "update slots\n");

    // this will run the main inference process for all slots
    vCallbackUpdateSlots();
    {
      // mUpdateSlots() may take a while to finish, we need to make sure it's not counted as idle
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      vTimeLastTask = ggml_time_ms();
    }

    QUE_DBG("%s", "waiting for new tasks\n");
    while (true) {
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      if (!vRunning || !lQueueTasks.empty()) {
        break; // go back to process new tasks or terminate
      }

      // no tasks, check for sleeping state
      if (vShouldSleep()) {
        QUE_INF("%s", "entering sleeping state\n");
        vSleeping = true;
        vCallbackSleepingState(true);
        vReqStopSleeping = false;
        // wait until we are requested to exit sleeping state
        vConditionTasks.wait(vLock, [&] { return (!vRunning || vReqStopSleeping); });
        if (!vRunning) { // may changed during sleep
          break;         // terminate
        }
        QUE_INF("%s", "exiting sleeping state\n");
        vReqStopSleeping = false;
        vCallbackSleepingState(false);
        vSleeping = false;
        vTimeLastTask = ggml_time_ms();
        vConditionTasks.notify_all(); // notify mWaitUntilNoSleep()
        break;                        // process new tasks
      } else {
        // wait for new tasks or timeout for checking sleeping condition
        bool vRes = vConditionTasks.wait_for(vLock, cMaxWaitTime, [&] { return (!lQueueTasks.empty() || !vRunning); });
        if (vRes) {
          break; // new task arrived or terminate
        }
        // otherwise, loop again to check sleeping condition
      }
    }
  }
}

void server_queue::mCleanupPendingTask(int pIdTarget) {
  // no need lock because this is called exclusively by mPost()
  auto vRmFunc = [pIdTarget](const server_task &pTask) { return pTask.vId == pIdTarget; };
  lQueueTasks.erase(std::remove_if(lQueueTasks.begin(), lQueueTasks.end(), vRmFunc), lQueueTasks.end());
  lQueueTasksDeferred.erase(std::remove_if(lQueueTasksDeferred.begin(), lQueueTasksDeferred.end(), vRmFunc),
                            lQueueTasksDeferred.end());
}

//
// server_response
//

void server_response::mAddWaitingTaskId(int pIdTask) {
  RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", pIdTask, (int)sWaitingTaskIds.size());

  std::unique_lock<std::mutex> vLock(vMutexResults);
  sWaitingTaskIds.insert(pIdTask);
}

void server_response::mAddWaitingTaskIds(const std::unordered_set<int> &pIdTasks) {
  std::unique_lock<std::mutex> vLock(vMutexResults);

  for (const auto &vIdTask : pIdTasks) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", vIdTask, (int)sWaitingTaskIds.size());
    sWaitingTaskIds.insert(vIdTask);
  }
}

void server_response::mRemoveWaitingTaskId(int pIdTask) {
  RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", pIdTask,
          (int)sWaitingTaskIds.size());

  std::unique_lock<std::mutex> vLock(vMutexResults);
  sWaitingTaskIds.erase(pIdTask);
  // make sure to clean up all pending results
  lQueueResults.erase(std::remove_if(lQueueResults.begin(), lQueueResults.end(),
                                     [pIdTask](const server_task_result_ptr &pRes) { return pRes->vId == pIdTask; }),
                      lQueueResults.end());
}

void server_response::mRemoveWaitingTaskIds(const std::unordered_set<int> &pIdTasks) {
  std::unique_lock<std::mutex> vLock(vMutexResults);

  for (const auto &vIdTask : pIdTasks) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", vIdTask,
            (int)sWaitingTaskIds.size());
    sWaitingTaskIds.erase(vIdTask);
  }
}

server_task_result_ptr server_response::mRecv(const std::unordered_set<int> &pIdTasks) {
  while (true) {
    std::unique_lock<std::mutex> vLock(vMutexResults);
    vConditionResults.wait(vLock, [&] {
      if (!vRunning) {
        RES_DBG("%s : queue result stop\n", "recv");
        std::terminate(); // we cannot return here since the caller is HTTP code
      }
      return !lQueueResults.empty();
    });

    for (size_t vI = 0; vI < lQueueResults.size(); vI++) {
      if (pIdTasks.find(lQueueResults[vI]->vId) != pIdTasks.end()) {
        server_task_result_ptr vRes = std::move(lQueueResults[vI]);
        lQueueResults.erase(lQueueResults.begin() + vI);
        return vRes;
      }
    }
  }

  // should never reach here
}

server_task_result_ptr server_response::mRecvWithTimeout(const std::unordered_set<int> &pIdTasks, int pTimeout) {
  while (true) {
    std::unique_lock<std::mutex> vLock(vMutexResults);

    for (int vI = 0; vI < (int)lQueueResults.size(); vI++) {
      if (pIdTasks.find(lQueueResults[vI]->vId) != pIdTasks.end()) {
        server_task_result_ptr vRes = std::move(lQueueResults[vI]);
        lQueueResults.erase(lQueueResults.begin() + vI);
        return vRes;
      }
    }

    std::cv_status vCrRes = vConditionResults.wait_for(vLock, std::chrono::seconds(pTimeout));
    if (!vRunning) {
      RES_DBG("%s : queue result stop\n", __func__);
      std::terminate(); // we cannot return here since the caller is HTTP code
    }
    if (vCrRes == std::cv_status::timeout) {
      return nullptr;
    }
  }

  // should never reach here
}

server_task_result_ptr server_response::mRecv(int pIdTask) {
  std::unordered_set<int> sIdTasks = {pIdTask};
  return mRecv(sIdTasks);
}

void server_response::mSend(server_task_result_ptr &&pResult) {
  RES_DBG("sending result for task id = %d\n", pResult->vId);

  std::unique_lock<std::mutex> vLock(vMutexResults);
  for (const auto &vIdTask : sWaitingTaskIds) {
    if (pResult->vId == vIdTask) {
      RES_DBG("task id = %d pushed to result queue\n", pResult->vId);

      lQueueResults.emplace_back(std::move(pResult));
      vConditionResults.notify_all();
      return;
    }
  }
}

void server_response::mBroadcast(server_task_result_ptr &&pResult) {
  std::unique_lock<std::mutex> vLock(vMutexResults);
  for (const auto &vIdTask : sWaitingTaskIds) {
    RES_DBG("task id = %d pushed to result queue\n", vIdTask);
    server_task_result_ptr vResCopy(pResult->mClone());
    vResCopy->vId = vIdTask; // override id with target task id
    lQueueResults.emplace_back(std::move(vResCopy));
  }
  vConditionResults.notify_all();
}

void server_response::mTerminate() {
  vRunning = false;
  vConditionResults.notify_all();
}

//
// server_response_reader
//

void server_response_reader::mPostTask(server_task &&pTask, bool pFront) {
  GGML_ASSERT(sIdTasks.empty() && "mPostTask() can only be called once per reader");
  GGML_ASSERT(!pTask.mIsParent() && "not supported, use mPostTasks() instead");
  pTask.vIndex = 0;
  sIdTasks.insert(pTask.vId);
  lStates.push_back(pTask.mCreateState());
  vQueueResults.mAddWaitingTaskId(pTask.vId);
  vQueueTasks.mPost(std::move(pTask), pFront);
}

void server_response_reader::mPostTasks(std::vector<server_task> &&pTasks, bool pFront) {
  GGML_ASSERT(sIdTasks.empty() && "mPostTasks() can only be called once per reader");
  sIdTasks = server_task::mGetListId(pTasks);
  lStates.reserve(pTasks.size());
  size_t vIndex = 0;
  for (auto &vTask : pTasks) {
    vTask.vIndex = vIndex++;
    lStates.push_back(vTask.mCreateState());
    // for child tasks
    for (auto &vChildTask : vTask.lChildTasks) {
      vChildTask.vIndex = vIndex++;
      lStates.push_back(vChildTask.mCreateState());
    }
  }
  GGML_ASSERT(lStates.size() == sIdTasks.size());
  vQueueResults.mAddWaitingTaskIds(sIdTasks);
  vQueueTasks.mPost(std::move(pTasks), pFront);
}

bool server_response_reader::mHasNext() const { return !vCancelled && vReceivedCount < sIdTasks.size(); }

// return nullptr if pShouldStop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::mNext(const std::function<bool()> &pShouldStop) {
  while (true) {
    server_task_result_ptr vResult = vQueueResults.mRecvWithTimeout(sIdTasks, vPollingIntervalSeconds);
    if (vResult == nullptr) {
      // timeout, check stop condition
      if (pShouldStop()) {
        return nullptr;
      }
    } else {
      if (vResult->mIsError()) {
        mStop(); // cancel remaining tasks
        SRV_DBG("%s", "received error result, stopping further processing\n");
        return vResult;
      }
      if (!lStates.empty()) {
        // update the generation state if needed
        const size_t cIdx = vResult->vIndex;
        GGML_ASSERT(cIdx < lStates.size());
        vResult->mUpdate(lStates[cIdx]);
      }
      if (vResult->mIsStop()) {
        vReceivedCount++;
      }
      return vResult;
    }
  }

  // should not reach here
}

server_response_reader::batch_response server_response_reader::mWaitForAll(const std::function<bool()> &pShouldStop) {
  batch_response vBatchRes;
  vBatchRes.lResults.clear();
  vBatchRes.lResults.resize(sIdTasks.size());
  while (mHasNext()) {
    auto vRes = mNext(pShouldStop);
    if (vRes == nullptr) {
      vBatchRes.vIsTerminated = true;
      return vBatchRes;
    }
    if (vRes->mIsError()) {
      vBatchRes.vError = std::move(vRes);
      return vBatchRes;
    }
    const size_t cIdx = vRes->vIndex;
    GGML_ASSERT(cIdx < vBatchRes.lResults.size() && "index out of range");
    GGML_ASSERT(vBatchRes.lResults[cIdx] == nullptr && "duplicate result received");
    vBatchRes.lResults[cIdx] = std::move(vRes);
  }
  return vBatchRes;
}

void server_response_reader::mStop() {
  vQueueResults.mRemoveWaitingTaskIds(sIdTasks);
  if (mHasNext() && !vCancelled) {
    // if tasks is not finished yet, cancel them
    vCancelled = true;
    std::vector<server_task> lCancelTasks;
    lCancelTasks.reserve(sIdTasks.size());
    for (const auto &vIdTask : sIdTasks) {
      SRV_WRN("cancel task, id_task = %d\n", vIdTask);
      server_task vTask(cServerTaskTypeCancel);
      vTask.vIdTarget = vIdTask;
      vQueueResults.mRemoveWaitingTaskId(vIdTask);
      lCancelTasks.push_back(std::move(vTask));
    }
    // push to beginning of the queue, so it has highest priority
    vQueueTasks.mPost(std::move(lCancelTasks), true);
  } else {
    SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
  }
}
