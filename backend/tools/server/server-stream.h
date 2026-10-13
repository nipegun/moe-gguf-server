#pragma once

#include "server-http.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

// streaming buffer for one generation, survives HTTP disconnect. the producer appends SSE bytes,
// readers drain from any offset via read_from. keyed by conversation_id, one conv = one live session

struct stream_session;

using stream_session_ptr = std::shared_ptr<stream_session>;

// base of the producer/consumer pipe ends. virtual dtor so each runs its own teardown:
// the producer finalizes the session, the consumer leaves it untouched
struct stream_pipe {
  virtual ~stream_pipe() = default;

  bool mIsCancelled() const;

protected:
  explicit stream_pipe(stream_session_ptr pSession);

  stream_session_ptr vSession;
};

// producer end: writes chunks into the ring buffer and owns the session lifetime, finalizing it
// on destruction.
struct stream_pipe_producer : stream_pipe {
  ~stream_pipe_producer() override;

  bool mWrite(const char *pData, size_t pLen);

  static stream_pipe_producer *mCreate(stream_session_ptr pSession);

private:
  explicit stream_pipe_producer(stream_session_ptr pSession);
};

void fServerStreamSessionManagerStart();
void fServerStreamSessionManagerStop();

// route handler factories wired under /v1/stream/* by server.cpp
// child-side handlers for the resumable stream routes. the conv id travels in the conv_id
// query string because it can embed a model name containing slashes (org/repo), which the
// decoded path would split before the param is captured
server_http_context::handler_t fServerStreamMakeGetHandler();
// POST /v1/streams/lookup with body {"conversation_ids": [...]}: only answers for ids the
// caller already owns (the WebUI passes the convs visible in its sidebar), the server never
// lists ids it has not been asked about, so a random caller cannot enumerate live sessions
server_http_context::handler_t fServerStreamMakeLookupHandler();
server_http_context::handler_t fServerStreamMakeDeleteHandler();

// extract the X-Conversation-Id header value (case-insensitive), empty when absent
std::string fServerStreamConvIdFromHeaders(const std::map<std::string, std::string> &pHeaders);

// implement tee-style pipe (spipe) for "stream replay" functionality
struct server_res_spipe : server_http_res {
private:
  // if set, the stream survives a client disconnect:
  // connection kept alive, output is forwarded to spipe and reuse later
  std::unique_ptr<stream_pipe_producer> vSpipe;
  // if spipe is set, use this next_orig to implement tee-style pipe
  std::function<bool(std::string &)> vNextOrig;
  const server_http_req *vReq = nullptr;
  // set once vNextOrig reports no more data, so mOnComplete() doesn't re-drain a finished stream
  bool vNextFinished = false;

public:
  void mSetReq(const server_http_req *pReq);
  bool mConnAlive();
  bool mShouldStop();
  void mOnComplete() override;
  void mSetNext(std::function<bool(std::string &)> pNextFn);
};
