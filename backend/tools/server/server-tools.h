#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-mcp.h"

#include <atomic>
#include <functional>
#include <memory>

struct server_tool {
  std::string vName;
  std::string vDisplayName;
  bool vPermissionWrite = false;
  bool vSupportStream = false; // if true, output can be streamed

  virtual ~server_tool() = default;
  virtual json mGetDefinition() const = 0;
  virtual std::string mType() const { return "builtin"; }

  struct stream {
    server_response &vQr;
    int vId;
    std::function<bool()> vAlive;
    void mPush(const std::string &pChunk);
  };
  virtual json mInvoke(json pParams, stream *pSt = nullptr) const = 0;

  json to_json() const;
};

struct lServerTools {
  std::vector<std::unique_ptr<server_tool>> lTools;

  // for streaming
  server_response vQueueRes;
  std::atomic<int> vResId{0};

  void mSetup(const std::vector<std::string> &pEnabledTools, server_mcp &pMcpMgr);

  server_http_context::handler_t vHandleGet;
  server_http_context::handler_t vHandlePost;
};
