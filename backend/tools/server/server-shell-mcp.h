#pragma once

#include "server-common.h"
#include "server-http.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

//
// Local shell MCP server, enabled with --local-shell-mcp-server.
//
// It answers MCP Streamable HTTP requests at <api prefix>/mcp/shell on the same listener as the API, so the Web UI
// reaches it through the same origin, TLS and API key. Responses are plain JSON (no SSE stream, no session). It
// exposes one tool per shell found on this machine: bash.run, zsh.run, sh.run and psh.run (PowerShell). Each call
// runs as the server's user, without a terminal and with empty input, in its own process group, so a timeout or a
// closed connection kills every process the command started.
//
// sudo password prompts (Linux): when the client names its call in the _meta of tools/call ("mgs/callId"), the
// command gets SUDO_ASKPASS pointing to this same executable. sudo starts it as its password helper, the helper
// forwards the prompt through a private socket, the Web UI finds it at <api prefix>/mcp/shell/askpass with the call
// id and sends the password, which only goes back to sudo. The model never sees it.
//

struct server_shell_mcp_shell {
  std::string vTool;                    // MCP tool name, e.g. "bash.run"
  std::string vTitle;                   // display name of the shell, e.g. "Bash"
  std::string vPath;                    // absolute path of the interpreter
  std::vector<std::string> lLArguments; // arguments placed between the interpreter and the command
};

class server_shell_mcp_askpass; // sudo password prompts, defined in server-shell-mcp.cpp

class server_shell_mcp {
public:
  server_shell_mcp();
  ~server_shell_mcp();

  // Detects the shells that are installed and executable; returns how many tools are exposed.
  size_t mStart();

  const std::vector<server_shell_mcp_shell> &mShells() const { return lLShells; }

  // POST: one JSON-RPC request or notification per call.
  server_http_res_ptr mHandlePost(const server_http_req &pRequest) const;

  // GET and DELETE: 405, the server offers neither an SSE stream nor sessions.
  server_http_res_ptr mHandleUnsupported(const server_http_req &pRequest) const;

  // GET ?call=<id>: the password prompt that sudo left pending in that call, if any.
  server_http_res_ptr mHandleAskpassGet(const server_http_req &pRequest) const;

  // POST {call, id, password | cancel}: answers that prompt; the password only goes back to sudo.
  server_http_res_ptr mHandleAskpassPost(const server_http_req &pRequest) const;

private:
  std::vector<server_shell_mcp_shell> lLShells;
  std::string vHomeDirectory; // default working directory
  std::string vUserName;
  std::string vHostName;
  std::unique_ptr<server_shell_mcp_askpass> vAskpass; // null when sudo prompts cannot be forwarded

  json mInitializeResult(const json &pParams) const;
  json mToolsListResult() const;
  json mToolsCallResult(const json &pParams, const std::function<bool()> &pShouldStop) const;
  const server_shell_mcp_shell *mFindShell(const std::string &pTool) const;
};

// Runs the password helper when sudo starts this executable as the SUDO_ASKPASS of a command of the local shell MCP
// server, and returns its exit status; returns -1 when the process was started for anything else.
int fServerShellMcpAskpassMain(int pArgumentCount, char **pArguments);
