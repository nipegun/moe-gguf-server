#pragma once

#include "server-common.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

//
// Configuration (Cursor-compatible "mcpServers" JSON)
//

struct server_mcp_server_config {
  std::string vName; // config key, e.g. "filesystem"
  std::string vCommand;
  std::vector<std::string> lArgs;
  std::map<std::string, std::string> dEnv; // merged over the parent env
  std::string vCwd;
  int vTimeoutMs = 30000; // per-tool-call timeout

  // throw on parse errors; missing "mcpServers" yields an empty list; entries without a "command" are skipped
  static std::vector<server_mcp_server_config> mParseFromJson(const std::string &pJsonStr);
  static std::vector<server_mcp_server_config> mParseCursorFormat(const json &pJ);
};

// a tool advertised by an MCP server
struct server_mcp_tool_def {
  std::string vServerName;
  std::string vName; // bare tool name, no "<server>_" prefix
  std::string vDescription;
  json vInputSchema; // JSON Schema for the arguments, or null
};

//
// server_mcp_transport: one MCP server session.
//
//   caller --send_rpc--> to_server   --[writer]--> framing --> server
//   caller <--send_rpc-- from_server <--[reader]-- framing <-- server
//
// each queue item is one complete serialized JSON message.
// subclass owns byte I/O and framing; base owns JSON and the JSON-RPC session (handshake, id correlation).
//

struct server_mcp_transport {
  std::string vName;
  int vTimeoutMs = 30000;

  server_pipe<std::string> vToServer;   // serialized messages we send to the server
  server_pipe<std::string> vFromServer; // serialized messages read from the server

  virtual ~server_mcp_transport() = default;

  virtual bool mStart() = 0;
  virtual void mClose() = 0;         // blocking and idempotent
  virtual bool mIsAlive() const = 0; // never blocks behind an in-flight mSendRpc()

  // human-readable diagnostics for logging when the transport fails/dies
  // (example: last RPC error, plus any transport-specific detail)
  // may run on a different thread than mSendRpc(), so vLastError is read under vRpcMutex
  virtual std::string mDiagnostics() {
    std::lock_guard<std::mutex> vLock(vRpcMutex);
    return vLastError;
  }

  std::vector<server_mcp_tool_def> mListTools(const std::function<bool()> &pShouldStop);

  json mCallTool(const std::string &pToolName, const json &pArguments, const std::function<bool()> &pShouldStop);

protected:
  // per-transport: mSendRpc() holds it across the reply wait, so sharing it would stall every server behind one slow
  // call. guards all members below.
  std::mutex vRpcMutex;
  uint64_t vNextId = 1; // reset to 1 per (re)spawn
  bool vInitialized = false;
  std::string vLastError;
  std::vector<server_mcp_tool_def> lTools;

  // both assume rpc_mutex is already held by the public caller
  bool mEnsureInit(const std::function<bool()> &pShouldStop); // initialize handshake, once
  json mSendRpc(const json &pRequest,
                const std::function<bool()> &pShouldStop); // returns the reply or an {"error": ...}
};

//
// server_mcp_stdio: child process, NDJSON JSON-RPC over stdio (stderr drained to the debug log)
//

struct server_mcp_stdio : server_mcp_transport {
  explicit server_mcp_stdio(const server_mcp_server_config &pConfig);
  ~server_mcp_stdio() override;

  bool mStart() override;
  void mClose() override;
  bool mIsAlive() const override;
  std::string mDiagnostics() override;

private:
  server_mcp_server_config vConfig;

  // defined in the .cpp so <windows.h> stays out of this header
  struct process_handle;
  std::unique_ptr<process_handle> vProc;

  std::thread vReader; // child stdout -> NDJSON de-framing -> from_server
  std::thread vWriter; // to_server -> NDJSON framing -> child stdin
  std::thread vErrlog; // child stderr -> debug log (must be drained or the child blocks)

  // cleared by close() or by the reader on stdout EOF; read without rpc_mutex
  std::atomic<bool> vRunning{false};

  // bounded tail of the child's stderr, for diagnostics when it dies
  std::mutex vErrMu;
  std::string vErrTail;

  void mReaderLoop();
  void mWriterLoop();
  void mErrlogLoop();
  void mJoinPumps();
};

//
// server_mcp
// declare before the HTTP context so it outlives every /tools handler.
//

class server_mcp {
public:
  server_mcp() = default;
  ~server_mcp();

  // parse the MCP config from params (file and/or inline JSON),
  // then spawn each server once,  list its tools, and shut it down
  // throws on config parse errors; spawn failures are logged.
  void mStart(const common_params &pParams);

  // true until mStart() has parsed at least one server from the config
  bool empty() const { return lConfigs.empty(); }

  std::vector<server_mcp_tool_def> mListTools() const;

  // lazily (re)spawns the transport. returns the MCP result or an {"error": ...}. should_stop is OR-ed with the
  // manager's cancel flag.
  json mCallTool(const std::string &pServerName, const std::string &pToolName, const json &pArguments,
                 const std::function<bool()> &pShouldStop = nullptr);

  // flip the cancel flag so in-flight calls return; blocking teardown is in the destructor. call before the HTTP server
  // drains. note: multiple calls are idempotent
  void mShutdown();

private:
  std::vector<server_mcp_server_config> lConfigs;

  mutable std::mutex vMutex; // guards transports, dead_servers, registry

  // shared_ptr: mCallTool() hands a transport to the caller and drops the lock for the blocking RPC, so a concurrent
  // evict/respawn must not destroy it mid-call
  std::map<std::string, std::shared_ptr<server_mcp_transport>> dTransports;
  std::map<std::string, std::chrono::steady_clock::time_point> dDeadServers; // spawn-failure cooldown
  std::vector<server_mcp_tool_def> lRegistry;

  std::atomic<bool> vStopping{false};

  const server_mcp_server_config *mFindConfig(const std::string &pName) const;

  // the only place that names a concrete transport
  std::shared_ptr<server_mcp_transport> mCreateTransport(const server_mcp_server_config &pCfg);

  // nullptr during cooldown or shutdown
  std::shared_ptr<server_mcp_transport> mGetOrCreate(const std::string &pName);
};
