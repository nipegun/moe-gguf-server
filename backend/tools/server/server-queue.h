#pragma once

#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <unordered_set>

// a task that waits in the deferred queue, for the queue status notices
struct server_queue_waiting {
  int vId;
  size_t vIndex;
  bool vFromWebui;
  bool vQueueNotice;
};

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
  int vId = 0;
  bool vRunning = false;
  bool vSleeping = false;
  bool vReqStopSleeping = false;
  int64_t vTimeLastTask = 0;

  // queues
  std::deque<server_task> lQueueTasks;
  std::deque<server_task> lQueueTasksDeferred;

  std::mutex vMutexTasks;
  std::condition_variable vConditionTasks;

  // callback functions
  std::function<void(server_task &&)> vCallbackNewTask;
  std::function<void(void)> vCallbackUpdateSlots;
  std::function<void(bool)> vCallbackSleepingState;

public:
  // Add a new task to the end of the queue
  int mPost(server_task &&pTask, bool pFront = false);

  // multi-task version of mPost()
  int mPost(std::vector<server_task> &&pTasks, bool pFront = false);

  // Add a new task, but defer until one slot is available
  void mDefer(server_task &&pTask);

  // Get the next id for creating a new task
  int mGetNewId();

  // Call when the state of one slot is changed, it will move one task from deferred to main queue
  // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
  void mPopDeferredTask(int pIdSlot);

  // if sleeping, request exiting sleep state and wait until it is done
  // returns immediately if not sleeping
  void mWaitUntilNoSleep();

  bool mIsSleeping() {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    return vSleeping;
  }

  // end the start_loop routine
  void mTerminate();

  /**
   * Main loop consists of these steps:
   * - Wait until a new task arrives
   * - Process the task (i.e. maybe copy data into slot)
   * - Check if multitask is finished
   * - Update all slots
   *
   * Sleeping procedure (disabled if idle_sleep_ms < 0):
   * - If there is no task after idle_sleep_ms, enter sleeping state
   * - Call callback_sleeping_state(true)
   * - Wait until req_stop_sleeping is set to true
   * - Call callback_sleeping_state(false)
   * - Exit sleeping state
   */
  void mStartLoop(int64_t pIdleSleepMs = -1);

  // for metrics
  size_t mQueueTasksDeferredSize() {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    return lQueueTasksDeferred.size();
  }

  // tasks that wait for a free slot, in the order they will run
  std::vector<server_queue_waiting> mDeferredTasks() {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    std::vector<server_queue_waiting> lWaiting;
    lWaiting.reserve(lQueueTasksDeferred.size());
    for (const server_task &vTask : lQueueTasksDeferred) {
      lWaiting.push_back({vTask.vId, vTask.vIndex, vTask.vParams.vFromWebui, vTask.vParams.vQueueNotice});
    }
    return lWaiting;
  }

  //
  // Functions below are not thread-safe, must only be used before mStartLoop() is called
  //

  // Register function to process a new task
  void mOnNewTask(std::function<void(server_task &&)> pCallback) { vCallbackNewTask = std::move(pCallback); }

  // Register the function to be called when all slots data is ready to be processed
  void mOnUpdateSlots(std::function<void(void)> pCallback) { vCallbackUpdateSlots = std::move(pCallback); }

  // Register callback for sleeping state change; multiple callbacks are allowed
  // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
  //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
  void mOnSleepingState(std::function<void(bool)> pCallback) {
    if (vCallbackSleepingState) {
      auto vPrevCallback = std::move(vCallbackSleepingState);
      vCallbackSleepingState = [vPrevCallback, pCallback](bool pSleeping) {
        vPrevCallback(pSleeping);
        pCallback(pSleeping);
      };
    } else {
      vCallbackSleepingState = std::move(pCallback);
    }
  }

private:
  void mCleanupPendingTask(int pIdTarget);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
  bool vRunning = true;

  // for keeping track of all tasks waiting for the result
  std::unordered_set<int> sWaitingTaskIds;

  // the main result queue (using ptr for polymorphism)
  std::vector<server_task_result_ptr> lQueueResults;

  std::mutex vMutexResults;
  std::condition_variable vConditionResults;

public:
  // add the id_task to the list of tasks waiting for response
  void mAddWaitingTaskId(int pIdTask);

  void mAddWaitingTaskIds(const std::unordered_set<int> &pIdTasks);

  // when the request is finished, we can remove task associated with it
  void mRemoveWaitingTaskId(int pIdTask);

  // remove multiple tasks from waiting list
  void mRemoveWaitingTaskIds(const std::unordered_set<int> &pIdTasks);

  // This function blocks the thread until there is a response for one of the id_tasks
  server_task_result_ptr mRecv(const std::unordered_set<int> &pIdTasks);

  // same as recv(), but have timeout in seconds
  // if timeout is reached, nullptr is returned
  server_task_result_ptr mRecvWithTimeout(const std::unordered_set<int> &pIdTasks, int pTimeout);

  // single-task version of recv()
  server_task_result_ptr mRecv(int pIdTask);

  // Send a new result to a waiting id_task
  void mSend(server_task_result_ptr &&pResult);

  // broadcast a new result to all waiting tasks
  // (used by router mode)
  void mBroadcast(server_task_result_ptr &&pResult);

  // terminate the waiting loop
  void mTerminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
  std::unordered_set<int> sIdTasks;
  server_queue &vQueueTasks;
  server_response &vQueueResults;
  size_t vReceivedCount = 0;
  bool vCancelled = false;
  int vPollingIntervalSeconds;

  // tracking generation state and partial tool calls
  // only used by streaming completions
  std::vector<task_result_state> lStates;

  // should_stop function will be called each polling_interval_seconds
  server_response_reader(server_queue &pQueueTasks, server_response &pQueueResults, int pPollingIntervalSeconds)
      : vQueueTasks(pQueueTasks), vQueueResults(pQueueResults), vPollingIntervalSeconds(pPollingIntervalSeconds) {}
  ~server_response_reader() { mStop(); }

  int mGetNewId() { return vQueueTasks.mGetNewId(); }

  // if front = true, the task will be posted to the front of the queue (high priority)
  void mPostTask(server_task &&pTask, bool pFront = false);
  void mPostTasks(std::vector<server_task> &&pTasks, bool pFront = false);
  bool mHasNext() const;

  // return nullptr if pShouldStop() is true before receiving a result
  // note: if one error is received, it will stop further processing and return error result
  server_task_result_ptr mNext(const std::function<bool()> &pShouldStop);

  struct batch_response {
    bool vIsTerminated = false; // if true, indicates that processing was stopped before all results were received
    std::vector<server_task_result_ptr> lResults;
    server_task_result_ptr vError; // nullptr if no error
  };
  // aggregate multiple results
  batch_response mWaitForAll(const std::function<bool()> &pShouldStop);

  void mStop();
};
