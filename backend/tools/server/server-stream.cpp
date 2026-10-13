#include "server-stream.h"
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <chrono>
#include <memory>
#include <utility>
#include <shared_mutex>

enum class stream_read_status {
  cOk,
  cOffsetLost,
};

namespace {
constexpr int64_t cStreamSessionTtlSeconds = 300;
constexpr size_t cStreamSessionMaxBytes = 4 * 1024 * 1024;
constexpr int64_t cStreamSessionGcIntervalSeconds = 60;
constexpr int64_t cStreamReadWakeIntervalMs = 200;

int64_t fNowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
} // namespace

// owns all live sessions keyed by conversation_id, one conv = at most one live session.
// a periodic GC evicts expired ones
class stream_session_manager {
public:
  stream_session_manager();
  ~stream_session_manager();

  stream_session_manager(const stream_session_manager &) = delete;
  stream_session_manager &operator=(const stream_session_manager &) = delete;

  // install a new session, evicting and cancelling any previous one. conversation_id must be non empty
  stream_session_ptr mCreateOrReplace(const std::string &pConversationId);

  stream_session_ptr mGet(const std::string &pConversationId);

  std::vector<stream_session_ptr> mListAll() const;

  void mEvict(const std::string &pConversationId);

  void mEvictAndCancel(const std::string &pConversationId);

  void mStartGc();
  void mStopGc();

private:
  void mGcLoop();

  mutable std::shared_mutex vMapMu;
  std::unordered_map<std::string, stream_session_ptr> dSessions; // key: conversation_id
  std::thread vGcThread;
  bool vRunning;
  std::mutex vGcWakeMu;
  std::condition_variable vGcWakeCv;
};

// process wide manager, lifecycle controlled by mgs main() via start_gc/stop_gc
static stream_session_manager vGStreamSessions;

void fServerStreamSessionManagerStart() { vGStreamSessions.mStartGc(); }

void fServerStreamSessionManagerStop() { vGStreamSessions.mStopGc(); }

struct stream_session {
  std::string vConversationId;
  int64_t vStartedTs; // unix seconds at construction

  stream_session(std::string pConversationId, size_t pMaxBytes);
  stream_session(const stream_session &) = delete;
  stream_session &operator=(const stream_session &) = delete;

  bool mAppend(const char *pData, size_t pLen);

  void mFinalize();

  // drain from offset into sink, blocking for more bytes or finalize. OFFSET_LOST if offset
  // fell below the dropped prefix
  stream_read_status mReadFrom(size_t pOffset, const std::function<bool(const char *, size_t)> &pSink,
                               const std::function<bool()> &pShouldStop);

  bool mIsDone() const;
  bool mIsCancelled() const;
  size_t mTotalSize() const;     // bytes that ever entered the session
  size_t mDroppedPrefix() const; // bytes evicted from the front due to cap
  int64_t mCompletedAt() const;  // 0 while alive, unix seconds after finalize

  void mCancel();

private:
  mutable std::mutex vMu;
  std::condition_variable vCv;
  std::vector<char> lBuffer;
  size_t vPrefixDropped;
  size_t vCapBytes;
  bool vDone;
  std::atomic<bool> vCancelled; // polled lock-free by the should_stop closure, no mu
  int64_t vCompletedTs;
};
stream_session::stream_session(std::string pConversationId, size_t pMaxBytes)
    : vConversationId(std::move(pConversationId)), vStartedTs(fNowSeconds()), vPrefixDropped(0), vCapBytes(pMaxBytes),
      vDone(false), vCancelled(false), vCompletedTs(0) {
  lBuffer.reserve(64 * 1024);
}

bool stream_session::mAppend(const char *pData, size_t pLen) {
  if (pLen == 0) {
    return true;
  }
  {
    std::lock_guard<std::mutex> vLock(vMu);
    if (vDone) {
      return false;
    }
    if (pLen >= vCapBytes) {
      // single chunk bigger than the cap, keep only the tail that fits
      size_t vSkip = pLen - vCapBytes;
      vPrefixDropped += lBuffer.size() + vSkip;
      lBuffer.clear();
      lBuffer.insert(lBuffer.end(), pData + vSkip, pData + pLen);
    } else {
      size_t vNeeded = lBuffer.size() + pLen;
      if (vNeeded > vCapBytes) {
        size_t vToDrop = vNeeded - vCapBytes;
        lBuffer.erase(lBuffer.begin(), lBuffer.begin() + vToDrop);
        vPrefixDropped += vToDrop;
      }
      lBuffer.insert(lBuffer.end(), pData, pData + pLen);
    }
  }
  vCv.notify_all();
  return true;
}

void stream_session::mFinalize() {
  {
    std::lock_guard<std::mutex> vLock(vMu);
    if (vDone) {
      return;
    }
    vDone = true;
    vCompletedTs = fNowSeconds();
  }
  vCv.notify_all();
}

stream_read_status stream_session::mReadFrom(size_t pOffset, const std::function<bool(const char *, size_t)> &pSink,
                                             const std::function<bool()> &pShouldStop) {
  std::unique_lock<std::mutex> vLock(vMu);
  while (true) {
    if (pShouldStop && pShouldStop()) {
      return stream_read_status::cOk;
    }
    if (pOffset < vPrefixDropped) {
      return stream_read_status::cOffsetLost;
    }
    size_t vLogicalEnd = vPrefixDropped + lBuffer.size();
    if (pOffset < vLogicalEnd) {
      size_t vLocalOff = pOffset - vPrefixDropped;
      size_t vN = lBuffer.size() - vLocalOff;
      // copy the available chunk under the lock, release before calling the sink
      std::vector<char> lChunk(lBuffer.begin() + vLocalOff, lBuffer.begin() + vLocalOff + vN);
      pOffset += vN;
      vLock.unlock();
      bool vKeepGoing = pSink(lChunk.data(), lChunk.size());
      if (!vKeepGoing) {
        return stream_read_status::cOk;
      }
      vLock.lock();
      continue;
    }
    if (vDone) {
      return stream_read_status::cOk;
    }
    // wait for new bytes, finalize, or a periodic wake to re check should_stop
    vCv.wait_for(vLock, std::chrono::milliseconds(cStreamReadWakeIntervalMs));
  }
}

bool stream_session::mIsDone() const {
  std::lock_guard<std::mutex> vLock(vMu);
  return vDone;
}

size_t stream_session::mTotalSize() const {
  std::lock_guard<std::mutex> vLock(vMu);
  return vPrefixDropped + lBuffer.size();
}

size_t stream_session::mDroppedPrefix() const {
  std::lock_guard<std::mutex> vLock(vMu);
  return vPrefixDropped;
}

int64_t stream_session::mCompletedAt() const {
  std::lock_guard<std::mutex> vLock(vMu);
  return vCompletedTs;
}

void stream_session::mCancel() {
  // the vShouldStop closure on both the producer and any HTTP reader polls mIsCancelled()
  // so flipping this is the only signal needed to unwind both sides
  vCancelled.store(true, std::memory_order_release);
}

bool stream_session::mIsCancelled() const { return vCancelled.load(std::memory_order_acquire); }

stream_session_manager::stream_session_manager() : vRunning(false) {}

stream_session_manager::~stream_session_manager() { mStopGc(); }

stream_session_ptr stream_session_manager::mCreateOrReplace(const std::string &pConversationId) {
  // evict any previous session on the same conv, this guarantees the invariant
  // "one conv = at most one live session" and propagates cancel to its producer
  stream_session_ptr vPrevious;
  auto vFresh = std::make_shared<stream_session>(pConversationId, cStreamSessionMaxBytes);
  {
    std::unique_lock<std::shared_mutex> vLock(vMapMu);
    auto vIt = dSessions.find(pConversationId);
    if (vIt != dSessions.end()) {
      vPrevious = vIt->second;
      vIt->second = vFresh;
    } else {
      dSessions.emplace(pConversationId, vFresh);
    }
  }
  if (vPrevious) {
    vPrevious->mCancel();
    vPrevious->mFinalize();
  }
  return vFresh;
}

stream_session_ptr stream_session_manager::mGet(const std::string &pConversationId) {
  std::shared_lock<std::shared_mutex> vLock(vMapMu);
  auto vIt = dSessions.find(pConversationId);
  if (vIt == dSessions.end()) {
    return nullptr;
  }
  return vIt->second;
}

std::vector<stream_session_ptr> stream_session_manager::mListAll() const {
  std::vector<stream_session_ptr> lOut;
  std::shared_lock<std::shared_mutex> vLock(vMapMu);
  lOut.reserve(dSessions.size());
  for (auto &vKv : dSessions) {
    lOut.push_back(vKv.second);
  }
  return lOut;
}

void stream_session_manager::mEvict(const std::string &pConversationId) {
  stream_session_ptr vS;
  {
    std::unique_lock<std::shared_mutex> vLock(vMapMu);
    auto vIt = dSessions.find(pConversationId);
    if (vIt == dSessions.end()) {
      return;
    }
    vS = vIt->second;
    dSessions.erase(vIt);
  }
  // finalize outside the map lock so any pending readers wake up and exit
  vS->mFinalize();
}

void stream_session_manager::mEvictAndCancel(const std::string &pConversationId) {
  stream_session_ptr vS;
  {
    std::unique_lock<std::shared_mutex> vLock(vMapMu);
    auto vIt = dSessions.find(pConversationId);
    if (vIt == dSessions.end()) {
      std::string vLive;
      for (const auto &vKv : dSessions) {
        if (!vLive.empty())
          vLive += ", ";
        vLive += vKv.first;
      }
      SRV_WRN("stop on unknown stream session, conv_id=%s matched nothing, %zu live: [%s]\n", pConversationId.c_str(),
              dSessions.size(), vLive.c_str());
      return;
    }
    vS = vIt->second;
    dSessions.erase(vIt);
  }
  // cancel first so the producer's mOnComplete() drain loop and any pending HTTP reader
  // observe mIsCancelled() and stop pulling further output, then finalize to wake readers
  // blocked in mReadFrom(). note: this does not interrupt the underlying generation itself,
  // which keeps running to its own natural stop condition (EOS/max_tokens)
  vS->mCancel();
  vS->mFinalize();
}

void stream_session_manager::mStartGc() {
  {
    std::lock_guard<std::mutex> vLock(vGcWakeMu);
    if (vRunning) {
      return;
    }
    vRunning = true;
  }
  vGcThread = std::thread([this] { mGcLoop(); });
}

void stream_session_manager::mStopGc() {
  bool vWasRunning;
  {
    std::lock_guard<std::mutex> vLock(vGcWakeMu);
    vWasRunning = vRunning;
    vRunning = false;
  }
  if (vWasRunning) {
    vGcWakeCv.notify_all();
    if (vGcThread.joinable()) {
      vGcThread.join();
    }
  }
  // finalize all live sessions so no reader ever hangs
  std::vector<stream_session_ptr> lSnapshot;
  {
    std::unique_lock<std::shared_mutex> vLock(vMapMu);
    lSnapshot.reserve(dSessions.size());
    for (auto &vKv : dSessions) {
      lSnapshot.push_back(vKv.second);
    }
    dSessions.clear();
  }
  for (auto &vS : lSnapshot) {
    vS->mFinalize();
  }
}

void stream_session_manager::mGcLoop() {
  while (true) {
    {
      std::unique_lock<std::mutex> vLock(vGcWakeMu);
      vGcWakeCv.wait_for(vLock, std::chrono::seconds(cStreamSessionGcIntervalSeconds), [this] { return !vRunning; });
      if (!vRunning) {
        return;
      }
    }
    int64_t vCutoff = fNowSeconds() - cStreamSessionTtlSeconds;
    std::vector<stream_session_ptr> lToDrop;
    {
      std::unique_lock<std::shared_mutex> vLock(vMapMu);
      for (auto vIt = dSessions.begin(); vIt != dSessions.end();) {
        int64_t vCompleted = vIt->second->mCompletedAt();
        if (vCompleted != 0 && vCompleted <= vCutoff) {
          lToDrop.push_back(vIt->second);
          vIt = dSessions.erase(vIt);
        } else {
          ++vIt;
        }
      }
    }
    // finalize outside the map lock, idempotent if the session was already done
    for (auto &vS : lToDrop) {
      vS->mFinalize();
    }
  }
}

// stream_pipe

// consumer end: read-only replay of the ring buffer, the destructor does not finalize the session
struct stream_pipe_consumer : stream_pipe {
  stream_read_status mRead(size_t &pOffset, const std::function<bool(const char *, size_t)> &pSink,
                           const std::function<bool()> &pShouldStop);

  static std::shared_ptr<stream_pipe_consumer> mCreate(stream_session_ptr pSession);

private:
  explicit stream_pipe_consumer(stream_session_ptr pSession);
};

stream_pipe::stream_pipe(stream_session_ptr pSession) : vSession(std::move(pSession)) {}

bool stream_pipe::mIsCancelled() const { return vSession->mIsCancelled(); }

// stream_pipe_producer

stream_pipe_producer::stream_pipe_producer(stream_session_ptr pSession) : stream_pipe(std::move(pSession)) {}

stream_pipe_producer::~stream_pipe_producer() { vSession->mFinalize(); }

bool stream_pipe_producer::mWrite(const char *pData, size_t pLen) { return vSession->mAppend(pData, pLen); }

stream_pipe_producer *stream_pipe_producer::mCreate(stream_session_ptr pSession) {
  return new stream_pipe_producer(std::move(pSession));
}

// stream_pipe_consumer

stream_pipe_consumer::stream_pipe_consumer(stream_session_ptr pSession) : stream_pipe(std::move(pSession)) {}

stream_read_status stream_pipe_consumer::mRead(size_t &pOffset, const std::function<bool(const char *, size_t)> &pSink,
                                               const std::function<bool()> &pShouldStop) {
  return vSession->mReadFrom(pOffset, pSink, pShouldStop);
}

std::shared_ptr<stream_pipe_consumer> stream_pipe_consumer::mCreate(stream_session_ptr pSession) {
  return std::shared_ptr<stream_pipe_consumer>(new stream_pipe_consumer(std::move(pSession)));
}

// helper, builds the standard error response and assigns it to a brand new http_res
static server_http_res_ptr fMakeErrorResponse(int pStatus, const std::string &pMessage, error_type pType) {
  auto vRes = std::make_unique<server_http_res>();
  json vErr = fFormatErrorResponse(pMessage, pType);
  vRes->vStatus = fJsonValue(vErr, "code", pStatus);
  vRes->vContentType = "application/json; charset=utf-8";
  vRes->vData = fSafeJsonToStr({{"error", vErr}});
  return vRes;
}

server_http_context::handler_t fServerStreamMakeGetHandler() {
  return [](const server_http_req &pReq) -> server_http_res_ptr {
    // GET /v1/stream?conv_id=<id>&from=N replays buffered SSE bytes then blocks for live
    // bytes until the session finalizes, streamed as text/event-stream for EventSource
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      return fMakeErrorResponse(400, "Missing conversation id in path", cErrorTypeInvalidRequest);
    }
    auto vSession = vGStreamSessions.mGet(vConvId);
    if (!vSession) {
      return fMakeErrorResponse(404, "Stream not found or expired", cErrorTypeNotFound);
    }
    size_t vFrom = 0;
    std::string vFromStr = pReq.mGetParam("from");
    if (!vFromStr.empty()) {
      try {
        vFrom = static_cast<size_t>(std::stoull(vFromStr));
      } catch (const std::exception &) {
        return fMakeErrorResponse(400, "Invalid 'from' offset", cErrorTypeInvalidRequest);
      }
    }
    if (vFrom < vSession->mDroppedPrefix()) {
      return fMakeErrorResponse(400, "Stream offset lost, please restart", cErrorTypeInvalidRequest);
    }
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 200;
    vRes->vContentType = "text/event-stream";
    // the next closure reads from the ring buffer at the requested offset, blocks until
    // bytes arrive or the session finalizes. exit each call after draining the available
    // chunk so set_chunked_content_provider gets a chance to flush to the socket
    auto vOffsetPtr = std::make_shared<size_t>(vFrom);
    // consumer pipe: read-only, does not finalize the session on destruction
    auto vPipe = stream_pipe_consumer::mCreate(vSession);
    vRes->vNext = [vPipe, vOffsetPtr, &pReq](std::string &pOutput) -> bool {
      bool vGotAny = false;
      vPipe->mRead(
          *vOffsetPtr,
          [&](const char *pD, size_t pN) {
            pOutput.append(pD, pN);
            *vOffsetPtr += pN;
            vGotAny = true;
            return false;
          },
          pReq.vShouldStop);
      return vGotAny;
    };
    return vRes;
  };
}

server_http_context::handler_t fServerStreamMakeLookupHandler() {
  return [](const server_http_req &pReq) -> server_http_res_ptr {
    // POST /v1/streams/lookup returns the matching sessions, only for ids the caller already
    // knows. each id matches the exact key and any "<id>::<model>" per model variant
    std::vector<std::string> lRequested;
    try {
      json vBody = json::parse(pReq.vBody);
      if (vBody.contains("conversation_ids") && vBody["conversation_ids"].is_array()) {
        for (const auto &vV : vBody["conversation_ids"]) {
          if (vV.is_string()) {
            std::string vId = vV.get<std::string>();
            if (!vId.empty()) {
              lRequested.push_back(std::move(vId));
            }
          }
        }
      }
    } catch (const std::exception &vE) {
      auto vRes = std::make_unique<server_http_res>();
      vRes->vStatus = 400;
      vRes->vContentType = "application/json; charset=utf-8";
      vRes->vData = fSafeJsonToStr(
          {{"error", {{"message", std::string("invalid body: ") + vE.what()}, {"type", "invalid_request_error"}}}});
      return vRes;
    }

    std::vector<stream_session_ptr> lSessions;
    if (!lRequested.empty()) {
      auto vAll = vGStreamSessions.mListAll();
      for (const auto &vRid : lRequested) {
        const std::string cWithSep = vRid + "::";
        for (auto &vS : vAll) {
          if (vS->vConversationId == vRid || vS->vConversationId.compare(0, cWithSep.size(), cWithSep) == 0) {
            lSessions.push_back(vS);
          }
        }
      }
    }

    json vArr = json::array();
    for (auto &vS : lSessions) {
      vArr.push_back({
          {"conversation_id", vS->vConversationId},
          {"is_done", vS->mIsDone()},
          {"total_bytes", vS->mTotalSize()},
          {"started_at", vS->vStartedTs},
          {"completed_at", vS->mCompletedAt()},
      });
    }
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 200;
    vRes->vContentType = "application/json; charset=utf-8";
    vRes->vData = fSafeJsonToStr(vArr);
    return vRes;
  };
}

server_http_context::handler_t fServerStreamMakeDeleteHandler() {
  return [](const server_http_req &pReq) -> server_http_res_ptr {
    // DELETE /v1/stream?conv_id=<id> is the explicit user Stop, cancels the producer and evicts
    // the buffer. idempotent, returns 204 even if the session was already gone
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      return fMakeErrorResponse(400, "Missing conversation id in path", cErrorTypeInvalidRequest);
    }
    SRV_TRC("DELETE /v1/stream conv_id=%s -> evict_and_cancel\n", vConvId.c_str());
    vGStreamSessions.mEvictAndCancel(vConvId);
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 204;
    vRes->vContentType = "application/json";
    return vRes;
  };
}

std::string fServerStreamConvIdFromHeaders(const std::map<std::string, std::string> &pHeaders) {
  // case-insensitive scan for x-conversation-id
  static constexpr char aTarget[] = "x-conversation-id";
  static constexpr size_t cTargetLen = sizeof(aTarget) - 1;
  for (const auto &[vHk, vHv] : pHeaders) {
    if (vHk.size() != cTargetLen)
      continue;
    bool vMatch = true;
    for (size_t vI = 0; vI < cTargetLen; ++vI) {
      char vC = vHk[vI];
      if (vC >= 'A' && vC <= 'Z')
        vC = char(vC + 32);
      if (vC != aTarget[vI]) {
        vMatch = false;
        break;
      }
    }
    if (vMatch) {
      return vHv;
    }
  }
  return std::string();
}

static stream_pipe_producer *fServerStreamCreateSpipe(const std::map<std::string, std::string> &pHeaders) {
  std::string vConversationId = fServerStreamConvIdFromHeaders(pHeaders);
  SRV_TRC("conv_id=%s (empty=%d)\n", vConversationId.c_str(), vConversationId.empty() ? 1 : 0);
  if (vConversationId.empty()) {
    return nullptr;
  }
  auto vSession = vGStreamSessions.mCreateOrReplace(vConversationId);
  return stream_pipe_producer::mCreate(vSession);
}

//
// server_res_spipe
//

void server_res_spipe::mSetReq(const server_http_req *pReq) {
  this->vReq = pReq;
  // optionally attach spipe to the response when X-Conversation-Id is present
  vSpipe.reset(fServerStreamCreateSpipe(pReq->dHeaders));
}

bool server_res_spipe::mConnAlive() {
  GGML_ASSERT(vReq != nullptr);
  return !vReq->vShouldStop();
}

bool server_res_spipe::mShouldStop() {
  if (vSpipe) {
    // note: if DELETE /v1/stream is called for this conv, mIsCancelled() will be true
    return vSpipe->mIsCancelled();
  } else {
    return !mConnAlive();
  }
}

void server_res_spipe::mOnComplete() {
  if (!vSpipe || vNextFinished) {
    return;
  }
  // an empty vNextOrig means mSetNext() never ran: the request failed before streaming
  // started, typically a params validation throw. evict the session installed by mSetReq()
  // so the failed request leaves nothing behind for discovery or replay
  if (!vNextOrig) {
    vGStreamSessions.mEvict(fServerStreamConvIdFromHeaders(vReq->dHeaders));
    return;
  }
  std::string vChunk;
  while (!vSpipe->mIsCancelled()) {
    vChunk.clear();
    bool vHasNext = vNextOrig(vChunk);
    if (!vChunk.empty()) {
      vSpipe->mWrite(vChunk.data(), vChunk.size());
    }
    if (!vHasNext) {
      break;
    }
  }
}

void server_res_spipe::mSetNext(std::function<bool(std::string &)> pNextFn) {
  vNextOrig = std::move(pNextFn);
  vNext = [this](std::string &pOut) {
    bool vHasNext = vNextOrig(pOut);
    if (vSpipe) {
      // if spipe is set, tee-style pipe input to both HTTP and spipe
      vSpipe->mWrite(pOut.data(), pOut.size());
    }
    if (!vHasNext) {
      vNextFinished = true;
    }
    return vHasNext;
  };
}
