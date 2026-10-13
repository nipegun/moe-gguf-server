#include "server-shell-mcp.h"

#include "build-info.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <thread>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
#endif

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#endif

namespace fs = std::filesystem;

// Protocol versions this server can answer; the first one is offered when the client asks for an unknown version.
static const std::vector<std::string> lLProtocolVersions = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"};

static constexpr int cDefaultTimeoutSeconds = 60;
static constexpr int cMaxTimeoutSeconds = 3600;
static constexpr size_t cStdoutHalfBytes = 8 * 1024; // kept from the beginning and from the end of stdout
static constexpr size_t cStderrHalfBytes = 4 * 1024; // kept from the beginning and from the end of stderr
static constexpr int cPollMilliseconds = 100;
static constexpr int cDrainMilliseconds = 250; // output still read after the shell exits
static constexpr int cKillWaitSeconds = 10;    // wait for a killed shell before answering without its status
static constexpr int cLogCommandBytes = 300;
static const char *cCallLogFolder = ".mgs/local-shell-mcp-server"; // one JSONL file per call, under the home

// sudo password prompts
static constexpr int cAskpassWaitSeconds = 300;                  // a prompt nobody answers is cancelled
static constexpr int cAskpassSocketSeconds = 5;                  // to send or receive one helper message
static constexpr size_t cAskpassMessageBytes = 16 * 1024;        // longest helper message
static constexpr size_t cAskpassPromptBytes = 512;               // longest prompt shown in the UI
static constexpr size_t cAskpassPasswordBytes = 4096;            // longest password accepted from the UI
static const char *cAskpassVariable = "MGS_LOCAL_SHELL_ASKPASS"; // "<socket name> <token>" for the helper
static const char *cAskpassDisplay = ":mgs-askpass"; // DISPLAY when the server has none: sudo only falls back to
                                                     // SUDO_ASKPASS without a terminal when DISPLAY is set, and X
                                                     // clients reject this name at once
static const char *cCallIdMeta = "mgs/callId";       // key of the call id in the _meta of tools/call

// JSON-RPC error codes
static constexpr int cParseError = -32700;
static constexpr int cInvalidRequest = -32600;
static constexpr int cMethodNotFound = -32601;
static constexpr int cInvalidParams = -32602;

// Terminal glyph drawn for light and for dark backgrounds.
static const char *cIconLight =
    "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' "
    "stroke='%231f2937' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect x='3' y='3' width='18' height='18' rx='2'/><path d='m7 9 3 3-3 3'/><path d='M12 15h5'/></svg>";
static const char *cIconDark =
    "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' "
    "stroke='%23e5e7eb' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect x='3' y='3' width='18' height='18' rx='2'/><path d='m7 9 3 3-3 3'/><path d='M12 15h5'/></svg>";

static std::string fLower(std::string pText) {
  std::transform(pText.begin(), pText.end(), pText.begin(),
                 [](unsigned char pVCharacter) { return static_cast<char>(std::tolower(pVCharacter)); });
  return pText;
}

// Header lookup without regard to case.
static std::string fHeader(const std::map<std::string, std::string> &pHeaders, const std::string &pName) {
  const std::string vName = fLower(pName);
  for (const auto &[vKey, vValue] : pHeaders) {
    if (fLower(vKey) == vName) {
      return vValue;
    }
  }
  return "";
}

// "https://Host:11443/path" -> "host:11443"; default ports are dropped so they compare equal to a Host header.
static std::string fAuthority(const std::string &pValue, const std::string &pScheme = "") {
  std::string vScheme = fLower(pScheme);
  std::string vRest = pValue;
  const size_t vSeparator = vRest.find("://");
  if (vSeparator != std::string::npos) {
    vScheme = fLower(vRest.substr(0, vSeparator));
    vRest = vRest.substr(vSeparator + 3);
  }
  vRest = fLower(vRest.substr(0, vRest.find_first_of("/?#,")));
  while (!vRest.empty() && std::isspace(static_cast<unsigned char>(vRest.back()))) {
    vRest.pop_back();
  }
  while (!vRest.empty() && std::isspace(static_cast<unsigned char>(vRest.front()))) {
    vRest.erase(0, 1);
  }
  if ((vScheme == "https" && vRest.size() > 4 && vRest.compare(vRest.size() - 4, 4, ":443") == 0) ||
      (vScheme == "http" && vRest.size() > 3 && vRest.compare(vRest.size() - 3, 3, ":80") == 0)) {
    vRest = vRest.substr(0, vRest.rfind(':'));
  }
  return vRest;
}

// "localhost:5173", "127.0.0.1:8080" or "[::1]:8080"
static bool fIsLoopback(const std::string &pAuthority) {
  std::string vHost = pAuthority;
  if (!vHost.empty() && vHost.front() == '[') {
    vHost = vHost.substr(1, vHost.find(']') - 1);
  } else {
    vHost = vHost.substr(0, vHost.find(':'));
  }
  return vHost == "localhost" || vHost == "::1" || vHost.rfind("127.", 0) == 0;
}

// Browsers always send Origin on cross-site POST requests: only the page served by this same server may call the
// endpoint (MCP requires Origin validation against DNS rebinding and cross-site requests). A loopback page talking to
// a loopback listener is also accepted, which covers the development proxy of the UI. Clients without Origin, such
// as curl, are only subject to the API key.
static bool fOriginAllowed(const server_http_req &pRequest) {
  const std::string vOrigin = fHeader(pRequest.dHeaders, "Origin");
  if (vOrigin.empty()) {
    return true;
  }
  if (fLower(vOrigin) == "null") {
    return false;
  }
  const std::string vScheme = vOrigin.substr(0, vOrigin.find("://"));
  const std::string vOriginAuthority = fAuthority(vOrigin);
  for (const char *vName : {"X-Forwarded-Host", "Host"}) {
    const std::string vHost = fHeader(pRequest.dHeaders, vName);
    if (vHost.empty()) {
      continue;
    }
    const std::string vHostAuthority = fAuthority(vHost, vScheme);
    if (vHostAuthority == vOriginAuthority || (fIsLoopback(vHostAuthority) && fIsLoopback(vOriginAuthority))) {
      return true;
    }
  }
  return false;
}

static std::string fDataUri(const std::string &pSvg) {
  std::string vUri = "data:image/svg+xml;utf8,";
  for (const char vCharacter : pSvg) {
    switch (vCharacter) {
    case '<':
      vUri += "%3C";
      break;
    case '>':
      vUri += "%3E";
      break;
    case ' ':
      vUri += "%20";
      break;
    default:
      vUri += vCharacter;
    }
  }
  return vUri;
}

static std::string fLogCommand(const std::string &pCommand) {
  std::string vText;
  for (const char vCharacter : pCommand) {
    if (vText.size() >= cLogCommandBytes) {
      vText += "...";
      break;
    }
    if (vCharacter == '\n') {
      vText += "\\n";
    } else if (vCharacter == '\r') {
      vText += "\\r";
    } else {
      vText += vCharacter;
    }
  }
  return vText;
}

static std::string fRandomHex(size_t pBytes) {
  static const char *cDigits = "0123456789abcdef";
  std::random_device vDevice;
  std::string vText;
  for (size_t vIndex = 0; vIndex < pBytes; vIndex++) {
    const unsigned int vByte = vDevice() & 0xffu;
    vText += cDigits[vByte >> 4];
    vText += cDigits[vByte & 0x0fu];
  }
  return vText;
}

// Overwrites a secret before its memory is released (best effort: copies made by libraries are not reached).
static void fWipe(std::string &pText) {
  volatile char *vData = pText.data();
  for (size_t vIndex = 0; vIndex < pText.size(); vIndex++) {
    vData[vIndex] = '\0';
  }
  pText.clear();
}

// Call ids come from the client: 8 to 128 letters, digits, '-' or '_'.
static bool fValidCallId(const std::string &pCallId) {
  if (pCallId.size() < 8 || pCallId.size() > 128) {
    return false;
  }
  return std::all_of(pCallId.begin(), pCallId.end(), [](unsigned char pVCharacter) {
    return std::isalnum(pVCharacter) || pVCharacter == '-' || pVCharacter == '_';
  });
}

// The prompt of sudo ("[sudo] password for user: ") as one line of text for the UI.
static std::string fCleanPrompt(const std::string &pPrompt) {
  std::string vText;
  for (const char vCharacter : pPrompt) {
    if (vText.size() >= cAskpassPromptBytes) {
      break;
    }
    vText += static_cast<unsigned char>(vCharacter) < 0x20 || vCharacter == 0x7f ? ' ' : vCharacter;
  }
  const size_t vEnd = vText.find_last_not_of(' ');
  return vEnd == std::string::npos ? std::string() : vText.substr(0, vEnd + 1);
}

//
// Output capture: keeps the beginning and the end of a stream and counts what is left out.
//

struct server_shell_mcp_capture {
  size_t vHalfBytes;
  std::string vHead;
  std::string vTail;
  size_t vTotal = 0;

  explicit server_shell_mcp_capture(size_t pHalfBytes) : vHalfBytes(pHalfBytes) {}

  void mAppend(const char *pData, size_t pSize) {
    vTotal += pSize;
    size_t vTaken = 0;
    if (vHead.size() < vHalfBytes) {
      vTaken = std::min(pSize, vHalfBytes - vHead.size());
      vHead.append(pData, vTaken);
    }
    if (vTaken < pSize) {
      vTail.append(pData + vTaken, pSize - vTaken);
      if (vTail.size() > 2 * vHalfBytes) {
        vTail.erase(0, vTail.size() - vHalfBytes);
      }
    }
  }

  std::string mText() const {
    const std::string vKept = vTail.size() > vHalfBytes ? vTail.substr(vTail.size() - vHalfBytes) : vTail;
    const size_t vOmitted = vTotal - vHead.size() - vKept.size();
    if (vOmitted == 0) {
      return vHead + vKept;
    }
    return vHead + "\n[... " + std::to_string(vOmitted) + " bytes omitted ...]\n" + vKept;
  }
};

struct server_shell_mcp_run {
  server_shell_mcp_capture vStdout{cStdoutHalfBytes};
  server_shell_mcp_capture vStderr{cStderrHalfBytes};
  int vExitCode = -1;
  int vSignal = 0;
  bool vTimedOut = false;
  bool vCancelled = false;
  std::string vError; // the process could not be started
  double vSeconds = 0.0;
};

//
// Call log: every command run leaves its own JSONL file in ~/.mgs/local-shell-mcp-server, named after the local
// time of the call down to the millisecond (y2026m10d12h23m59s59ms731.jsonl; -2, -3... for more calls in the same
// millisecond). The first line is the command, written before it runs; then every piece of stdout and stderr as it is
// read, complete and in the order it arrived (the tool result only keeps the beginning and the end of long outputs);
// and last the result, or the reason the command could not start. Nothing is kept in memory, and the file already
// holds the output if mgs stops while the command runs. The folder and the files are only readable by the user that
// runs mgs, because the output of a command may hold secrets.
//

// Length of the start of pText that does not end in the middle of a UTF-8 character, so a character split between
// two reads is written whole with the next one. Bytes that are not UTF-8 at all are left to the JSON writer.
static size_t fUtf8CompleteSize(const std::string &pText) {
  const size_t vSize = pText.size();
  size_t vStart = vSize;
  for (size_t vBack = 1; vBack <= 4 && vBack <= vSize; vBack++) {
    const unsigned char vByte = static_cast<unsigned char>(pText[vSize - vBack]);
    if ((vByte & 0xC0) != 0x80) {
      vStart = vSize - vBack;
      break;
    }
  }
  if (vStart == vSize) {
    return vSize;
  }
  const unsigned char vLead = static_cast<unsigned char>(pText[vStart]);
  const size_t vLength = (vLead & 0xE0) == 0xC0 ? 2 : (vLead & 0xF0) == 0xE0 ? 3 : (vLead & 0xF8) == 0xF0 ? 4 : 1;
  return vSize - vStart < vLength ? vStart : vSize;
}

// Local time with its offset, such as "2026-10-12T23:59:59+02:00".
static std::string fLogTime(std::time_t pTime) {
  std::tm vLocal = {};
#if defined(_WIN32)
  localtime_s(&vLocal, &pTime);
#else
  localtime_r(&pTime, &vLocal);
#endif
  char aText[40] = {};
  std::strftime(aText, sizeof(aText), "%Y-%m-%dT%H:%M:%S%z", &vLocal);
  std::string vText = aText;
  if (vText.size() > 5 && (vText[vText.size() - 5] == '+' || vText[vText.size() - 5] == '-')) {
    vText.insert(vText.size() - 2, ":");
  }
  return vText;
}

struct server_shell_mcp_call_log {
  int vFd = -1;
  std::string vPath;
  std::string aPending[2]; // end of a UTF-8 character split between two reads of stdout and of stderr
  bool vWarned = false;

  server_shell_mcp_call_log() = default;
  server_shell_mcp_call_log(const server_shell_mcp_call_log &) = delete;
  server_shell_mcp_call_log &operator=(const server_shell_mcp_call_log &) = delete;

  ~server_shell_mcp_call_log() {
#if !defined(_WIN32)
    if (vFd >= 0) {
      close(vFd);
    }
#endif
  }

  // Creates the file of a new call; false (with errno) when the folder or the file cannot be created.
  bool mOpen(const std::string &pHome) {
#if defined(_WIN32)
    (void)pHome;
    return false;
#else
    const fs::path vFolder = fs::path(pHome) / cCallLogFolder;
    for (const fs::path &vPart : {vFolder.parent_path(), vFolder}) {
      if (mkdir(vPart.c_str(), 0700) != 0 && errno != EEXIST) {
        return false;
      }
    }
    const auto vNow = std::chrono::system_clock::now();
    const auto vSecond = std::chrono::floor<std::chrono::seconds>(vNow);
    const std::time_t vTime = std::chrono::system_clock::to_time_t(vSecond);
    const long long vMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(vNow - vSecond).count();
    std::tm vLocal = {};
    localtime_r(&vTime, &vLocal);
    char aName[40] = {};
    const size_t vLength = std::strftime(aName, sizeof(aName), "y%Ym%md%dh%Hm%Ms%S", &vLocal);
    std::snprintf(aName + vLength, sizeof(aName) - vLength, "ms%03lld", vMilliseconds);
    for (int vNumber = 1; vNumber <= 1000; vNumber++) {
      const fs::path vFile =
          vFolder / (std::string(aName) + (vNumber > 1 ? "-" + std::to_string(vNumber) : std::string()) + ".jsonl");
      vFd = open(vFile.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
      if (vFd >= 0) {
        vPath = vFile.string();
        return true;
      }
      if (errno != EEXIST) {
        return false;
      }
    }
    return false;
#endif
  }

  // Appends one line with the time of the event; a failed write only leaves the log incomplete.
  bool mWrite(json pLine) {
#if defined(_WIN32)
    (void)pLine;
    return false;
#else
    if (vFd < 0) {
      return false;
    }
    json dLine = {{"time", fLogTime(std::time(nullptr))}};
    dLine.update(pLine);
    const std::string vText = fSafeJsonToStr(dLine) + "\n";
    size_t vWritten = 0;
    while (vWritten < vText.size()) {
      const ssize_t vCount = write(vFd, vText.data() + vWritten, vText.size() - vWritten);
      if (vCount < 0 && errno == EINTR) {
        continue;
      }
      if (vCount <= 0) {
        if (!vWarned) {
          vWarned = true;
          SRV_WRN("shell MCP: cannot write the call log %s: %s\n", vPath.c_str(), std::strerror(errno));
        }
        return false;
      }
      vWritten += (size_t)vCount;
    }
    return true;
#endif
  }

  // One line per read of the command output: {"event": "stdout" or "stderr", "data": "..."}.
  void mOutput(int pStream, const char *pData, size_t pSize) {
    if (vFd < 0 || pStream < 0 || pStream > 1) {
      return;
    }
    std::string &vPending = aPending[pStream];
    vPending.append(pData, pSize);
    const size_t vComplete = fUtf8CompleteSize(vPending);
    if (vComplete == 0) {
      return;
    }
    mWrite({{"event", pStream == 0 ? "stdout" : "stderr"}, {"data", vPending.substr(0, vComplete)}});
    vPending.erase(0, vComplete);
  }

  // Writes what is left of a character that the command never finished.
  void mFlushOutput() {
    for (int vStream = 0; vStream < 2; vStream++) {
      if (!aPending[vStream].empty()) {
        mWrite({{"event", vStream == 0 ? "stdout" : "stderr"}, {"data", aPending[vStream]}});
        aPending[vStream].clear();
      }
    }
  }
};

#if !defined(_WIN32)

static bool fCreatePipe(int pFds[2]) {
#if defined(__linux__)
  return pipe2(pFds, O_CLOEXEC) == 0;
#else
  if (pipe(pFds) != 0) {
    return false;
  }
  fcntl(pFds[0], F_SETFD, FD_CLOEXEC);
  fcntl(pFds[1], F_SETFD, FD_CLOEXEC);
  return true;
#endif
}

// Runs pCommand with the shell in a new session (its own process group, without the server's controlling terminal).
// stdin is /dev/null, stdout and stderr are captured separately. The deadline and pShouldStop (closed connection) kill
// the whole group; after a normal exit the remaining output is read for a short while, so background processes that
// keep the pipes open do not block.
// The command inherits the server's environment, with pEnvironment ("NAME=value") replacing the entries of the same
// name; pPaused returns the time spent waiting for the user (sudo password prompts), which moves the deadline.
// pOutput, when set, receives every piece of output as it is read (stream 0 is stdout and 1 is stderr), whole.
static void fRunCommand(const server_shell_mcp_shell &pShell, const std::string &pCommand,
                        const std::string &pDirectory, int pTimeoutSeconds, const std::function<bool()> &pShouldStop,
                        const std::vector<std::string> &pEnvironment,
                        const std::function<std::chrono::steady_clock::duration()> &pPaused,
                        const std::function<void(int, const char *, size_t)> &pOutput, server_shell_mcp_run &pRun) {
  const auto vStart = std::chrono::steady_clock::now();
  int aStdout[2] = {-1, -1};
  int aStderr[2] = {-1, -1};
  if (!fCreatePipe(aStdout)) {
    pRun.vError = std::string("cannot create a pipe: ") + std::strerror(errno);
    return;
  }
  if (!fCreatePipe(aStderr)) {
    pRun.vError = std::string("cannot create a pipe: ") + std::strerror(errno);
    close(aStdout[0]);
    close(aStdout[1]);
    return;
  }

  std::vector<std::string> lLArguments = {pShell.vPath};
  lLArguments.insert(lLArguments.end(), pShell.lLArguments.begin(), pShell.lLArguments.end());
  lLArguments.push_back(pCommand);
  std::vector<char *> lLArgv;
  for (auto &vArgument : lLArguments) {
    lLArgv.push_back(vArgument.data());
  }
  lLArgv.push_back(nullptr);

  std::vector<std::string> lLEnvironment;
  for (char **vEntry = environ; vEntry != nullptr && *vEntry != nullptr; vEntry++) {
    const std::string vText = *vEntry;
    const std::string vName = vText.substr(0, vText.find('=') + 1);
    const bool vReplaced = std::any_of(pEnvironment.begin(), pEnvironment.end(), [&vName](const std::string &pExtra) {
      return pExtra.compare(0, vName.size(), vName) == 0;
    });
    if (!vReplaced) {
      lLEnvironment.push_back(vText);
    }
  }
  lLEnvironment.insert(lLEnvironment.end(), pEnvironment.begin(), pEnvironment.end());
  std::vector<char *> lLEnvp;
  for (auto &vEntry : lLEnvironment) {
    lLEnvp.push_back(vEntry.data());
  }
  lLEnvp.push_back(nullptr);

  posix_spawn_file_actions_t vActions;
  posix_spawnattr_t vAttributes;
  posix_spawn_file_actions_init(&vActions);
  posix_spawnattr_init(&vAttributes);
  posix_spawn_file_actions_addopen(&vActions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&vActions, aStdout[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&vActions, aStderr[1], STDERR_FILENO);
  posix_spawn_file_actions_addchdir_np(&vActions, pDirectory.c_str());
  // The server ignores SIGPIPE and its threads may block signals; commands get the default behavior back.
  sigset_t vDefaultSignals;
  sigemptyset(&vDefaultSignals);
  for (const int vSignal :
       {SIGPIPE, SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGCHLD, SIGALRM, SIGUSR1, SIGUSR2, SIGTSTP, SIGTTIN, SIGTTOU}) {
    sigaddset(&vDefaultSignals, vSignal);
  }
  sigset_t vEmptyMask;
  sigemptyset(&vEmptyMask);
  posix_spawnattr_setsigdefault(&vAttributes, &vDefaultSignals);
  posix_spawnattr_setsigmask(&vAttributes, &vEmptyMask);
  // Without a session of its own the command would keep the terminal that started mgs: sudo (and ssh) would prompt
  // there, stopped by SIGTTIN as a background group, instead of using SUDO_ASKPASS. The session is also a new process
  // group, which the deadline kills as a whole.
#if defined(POSIX_SPAWN_SETSID)
  posix_spawnattr_setflags(&vAttributes, POSIX_SPAWN_SETSID | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
#else
  posix_spawnattr_setpgroup(&vAttributes, 0);
  posix_spawnattr_setflags(&vAttributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
#endif

  pid_t vPid = -1;
  const int vSpawnError =
      posix_spawn(&vPid, pShell.vPath.c_str(), &vActions, &vAttributes, lLArgv.data(), lLEnvp.data());
  posix_spawn_file_actions_destroy(&vActions);
  posix_spawnattr_destroy(&vAttributes);
  close(aStdout[1]);
  close(aStderr[1]);
  if (vSpawnError != 0) {
    pRun.vError = std::string("cannot start ") + pShell.vPath + ": " + std::strerror(vSpawnError);
    close(aStdout[0]);
    close(aStderr[0]);
    return;
  }

  int aFds[2] = {aStdout[0], aStderr[0]};
  server_shell_mcp_capture *aCaptures[2] = {&pRun.vStdout, &pRun.vStderr};
  const auto vDeadline = vStart + std::chrono::seconds(pTimeoutSeconds);
  std::optional<std::chrono::steady_clock::time_point> vDrainDeadline;
  std::chrono::steady_clock::time_point vKillTime;
  bool vReaped = false;
  bool vKilled = false;
  int vStatus = 0;
  char aBuffer[64 * 1024];

  while (true) {
    if (!vReaped) {
      const pid_t vWaited = waitpid(vPid, &vStatus, WNOHANG);
      if (vWaited == vPid || (vWaited < 0 && errno != EINTR)) {
        vReaped = true;
      }
    }
    const auto vNow = std::chrono::steady_clock::now();
    if (vReaped) {
      if (aFds[0] < 0 && aFds[1] < 0) {
        break;
      }
      if (!vDrainDeadline) {
        vDrainDeadline = vNow + std::chrono::milliseconds(cDrainMilliseconds);
      }
      if (vNow >= *vDrainDeadline) {
        break;
      }
    } else if (!vKilled) {
      if (pShouldStop && pShouldStop()) {
        pRun.vCancelled = true;
      } else if (vNow >= vDeadline + (pPaused ? pPaused() : std::chrono::steady_clock::duration::zero())) {
        pRun.vTimedOut = true;
      }
      if (pRun.vCancelled || pRun.vTimedOut) {
        kill(-vPid, SIGKILL);
        vKilled = true;
        vKillTime = vNow;
      }
    } else if (vNow - vKillTime >= std::chrono::seconds(cKillWaitSeconds)) {
      // a process stuck in the kernel ignores SIGKILL for a while: answer now and reap it later
      std::thread([vPid]() {
        int vLateStatus = 0;
        waitpid(vPid, &vLateStatus, 0);
      }).detach();
      break;
    }

    pollfd aPoll[2];
    int aOwner[2];
    nfds_t vCount = 0;
    for (int vIndex = 0; vIndex < 2; vIndex++) {
      if (aFds[vIndex] >= 0) {
        aPoll[vCount] = {aFds[vIndex], POLLIN, 0};
        aOwner[vCount] = vIndex;
        vCount++;
      }
    }
    if (vCount == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(cPollMilliseconds));
      continue;
    }
    const int vReady = poll(aPoll, vCount, cPollMilliseconds);
    if (vReady < 0) {
      if (errno != EINTR) {
        for (int &vFd : aFds) {
          if (vFd >= 0) {
            close(vFd);
            vFd = -1;
          }
        }
      }
      continue;
    }
    for (nfds_t vPosition = 0; vPosition < vCount; vPosition++) {
      if ((aPoll[vPosition].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) == 0) {
        continue;
      }
      const int vIndex = aOwner[vPosition];
      const ssize_t vRead = read(aFds[vIndex], aBuffer, sizeof(aBuffer));
      if (vRead > 0) {
        aCaptures[vIndex]->mAppend(aBuffer, static_cast<size_t>(vRead));
        if (pOutput) {
          pOutput(vIndex, aBuffer, static_cast<size_t>(vRead));
        }
      } else if (vRead == 0 || (errno != EINTR && errno != EAGAIN)) {
        close(aFds[vIndex]);
        aFds[vIndex] = -1;
      }
    }
  }

  for (const int vFd : aFds) {
    if (vFd >= 0) {
      close(vFd);
    }
  }
  pRun.vSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - vStart).count();
  if (WIFEXITED(vStatus)) {
    pRun.vExitCode = WEXITSTATUS(vStatus);
  } else if (WIFSIGNALED(vStatus)) {
    pRun.vSignal = WTERMSIG(vStatus);
    pRun.vExitCode = 128 + pRun.vSignal;
  }
}

static std::string fFindExecutable(const std::vector<std::string> &pNames) {
  std::vector<std::string> lLDirectories;
  const char *vPath = std::getenv("PATH");
  if (vPath != nullptr) {
    std::string vEntries = vPath;
    size_t vBegin = 0;
    while (vBegin <= vEntries.size()) {
      const size_t vEnd = std::min(vEntries.find(':', vBegin), vEntries.size());
      const std::string vDirectory = vEntries.substr(vBegin, vEnd - vBegin);
      // relative entries would depend on the working directory of each command
      if (!vDirectory.empty() && vDirectory.front() == '/') {
        lLDirectories.push_back(vDirectory);
      }
      vBegin = vEnd + 1;
    }
  }
  for (const char *vDirectory : {"/usr/local/bin", "/usr/bin", "/bin", "/opt/microsoft/powershell/7"}) {
    lLDirectories.push_back(vDirectory);
  }
  for (const auto &vName : pNames) {
    for (const auto &vDirectory : lLDirectories) {
      const std::string vCandidate = (fs::path(vDirectory) / vName).string();
      std::error_code vError;
      if (fs::is_regular_file(vCandidate, vError) && !vError && access(vCandidate.c_str(), X_OK) == 0) {
        return vCandidate;
      }
    }
  }
  return "";
}

#endif

//
// sudo password prompts. sudo runs this same executable as SUDO_ASKPASS (fServerShellMcpAskpassMain), which sends
// the token of its call and the prompt through an abstract Unix socket and prints the answer for sudo. The server
// only answers helpers of this user whose parent is sudo, so a command cannot run the helper itself, with a fake
// prompt, to read the password. The Web UI polls the prompt with the call id and posts the password or a cancellation.
//

struct server_shell_mcp_call {
  std::string vCallId;    // chosen by the client
  std::string vToken;     // secret given to the command in cAskpassVariable
  bool vFinished = false; // the command ended
  std::string vPromptId;  // pending prompt, empty when there is none
  std::string vPrompt;
  bool vAnswered = false;
  bool vCancelled = false;
  std::string vPassword;
  std::chrono::steady_clock::duration vPaused = std::chrono::steady_clock::duration::zero();
  std::optional<std::chrono::steady_clock::time_point> vPauseStart; // while a prompt waits for the user
};

#if defined(__linux__)

static std::string fReadProcFile(const std::string &pPath) {
  std::ifstream vFile(pPath);
  std::stringstream vText;
  vText << vFile.rdbuf();
  return vText.str();
}

// True when the parent of pPid is sudo authenticating this user: named "sudo" (or "sudoedit"), with our real uid and
// root as its effective or saved uid, which only a setuid-root program has. An unprivileged process cannot read the
// executable of a setuid process, so this is the closest check available; it stops a command from running the helper
// itself to read the password.
static bool fParentIsSudo(pid_t pPid) {
  const std::string vStat = fReadProcFile("/proc/" + std::to_string(pPid) + "/stat");
  const size_t vNameEnd = vStat.rfind(')');
  if (vNameEnd == std::string::npos) {
    return false;
  }
  std::istringstream vFields(vStat.substr(vNameEnd + 1));
  std::string vState;
  long vParent = 0;
  if (!(vFields >> vState >> vParent) || vParent <= 1) {
    return false;
  }
  std::istringstream vStatus(fReadProcFile("/proc/" + std::to_string(vParent) + "/status"));
  std::string vLine;
  std::string vName;
  long vReal = -1;
  long vEffective = -1;
  long vSaved = -1;
  while (std::getline(vStatus, vLine)) {
    if (vLine.rfind("Name:", 0) == 0) {
      std::istringstream(vLine.substr(5)) >> vName;
    } else if (vLine.rfind("Uid:", 0) == 0) {
      std::istringstream(vLine.substr(4)) >> vReal >> vEffective >> vSaved;
    }
  }
  return (vName == "sudo" || vName == "sudoedit") && vReal == static_cast<long>(getuid()) &&
         (vEffective == 0 || vSaved == 0);
}

// Abstract namespace: a leading NUL, so there is no file to create, protect or remove.
static sockaddr_un fSocketAddress(const std::string &pName, socklen_t &pLength) {
  sockaddr_un vAddress = {};
  vAddress.sun_family = AF_UNIX;
  const size_t vSize = std::min(pName.size(), sizeof(vAddress.sun_path) - 1);
  std::memcpy(vAddress.sun_path + 1, pName.data(), vSize);
  pLength = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + vSize);
  return vAddress;
}

static void fSetSocketTimeout(int pFd, int pSeconds) {
  timeval vTimeout = {};
  vTimeout.tv_sec = pSeconds;
  setsockopt(pFd, SOL_SOCKET, SO_RCVTIMEO, &vTimeout, sizeof(vTimeout));
  setsockopt(pFd, SOL_SOCKET, SO_SNDTIMEO, &vTimeout, sizeof(vTimeout));
}

static bool fSendAll(int pFd, const std::string &pText) {
  size_t vSent = 0;
  while (vSent < pText.size()) {
    const ssize_t vCount = send(pFd, pText.data() + vSent, pText.size() - vSent, MSG_NOSIGNAL);
    if (vCount < 0 && errno == EINTR) {
      continue;
    }
    if (vCount <= 0) {
      return false;
    }
    vSent += static_cast<size_t>(vCount);
  }
  return true;
}

// Reads one message, up to a newline (dropped) or the end of the stream; false on errors or oversized messages.
static bool fReceiveLine(int pFd, std::string &pText) {
  char aBuffer[1024];
  bool vComplete = false;
  while (pText.size() < cAskpassMessageBytes) {
    const ssize_t vCount = recv(pFd, aBuffer, sizeof(aBuffer), 0);
    if (vCount < 0 && errno == EINTR) {
      continue;
    }
    if (vCount <= 0) {
      vComplete = vCount == 0;
      break;
    }
    pText.append(aBuffer, static_cast<size_t>(vCount));
    const size_t vEnd = pText.find('\n');
    if (vEnd != std::string::npos) {
      pText.resize(vEnd);
      vComplete = true;
      break;
    }
  }
  volatile char *vBuffer = aBuffer;
  for (size_t vIndex = 0; vIndex < sizeof(aBuffer); vIndex++) {
    vBuffer[vIndex] = '\0';
  }
  return vComplete;
}

// Sends {pKey: pValue} and a newline, then overwrites the copies made here (pValue may be the password).
static void fReplyLine(int pFd, const std::string &pKey, std::string pValue) {
  json dAnswer = {{pKey, std::move(pValue)}};
  std::string vMessage = fSafeJsonToStr(dAnswer);
  fSendAll(pFd, vMessage);
  fSendAll(pFd, "\n");
  fWipe(vMessage);
  fWipe(dAnswer[pKey].get_ref<std::string &>());
}

#endif

class server_shell_mcp_askpass {
public:
  ~server_shell_mcp_askpass();

  // Listens on a new abstract socket; pExecutable is what sudo runs as SUDO_ASKPASS.
  bool mStart(const std::string &pExecutable);

  std::shared_ptr<server_shell_mcp_call> mRegister(const std::string &pCallId);
  void mUnregister(const std::shared_ptr<server_shell_mcp_call> &pCall);

  // Environment entries that send the password prompts of sudo to this server.
  std::vector<std::string> mEnvironment(const server_shell_mcp_call &pCall) const;

  // Time the call has spent waiting for the user, including a prompt still pending.
  std::chrono::steady_clock::duration mPaused(const server_shell_mcp_call &pCall);

  // {"id", "prompt"} of the prompt pending in the call, or null.
  json mPending(const std::string &pCallId);

  // Answers the prompt with the password, or cancels it when pPassword is null; false when it is not pending.
  bool mAnswer(const std::string &pCallId, const std::string &pPromptId, const std::string *pPassword);

private:
  std::string vExecutable;
  std::string vSocketName;
  int vListener = -1;
  std::atomic<bool> vStopping{false};
  std::thread vAcceptThread;
  std::mutex vMutex;
  std::condition_variable vCondition;
  std::vector<std::shared_ptr<server_shell_mcp_call>> lLCalls;
  size_t vActiveHelpers = 0;

  void mAcceptLoop();
  void mServe(int pClient);
};

server_shell_mcp_askpass::~server_shell_mcp_askpass() {
  vStopping = true;
  if (vAcceptThread.joinable()) {
    vAcceptThread.join();
  }
  std::unique_lock<std::mutex> vLock(vMutex);
  vCondition.notify_all();
  vCondition.wait(vLock, [this]() { return vActiveHelpers == 0; });
#if defined(__linux__)
  if (vListener >= 0) {
    close(vListener);
  }
#endif
}

bool server_shell_mcp_askpass::mStart(const std::string &pExecutable) {
#if defined(__linux__)
  vExecutable = pExecutable;
  vSocketName = "mgs-shell-askpass-" + std::to_string(getpid()) + "-" + fRandomHex(8);
  vListener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (vListener < 0) {
    SRV_WRN("shell MCP: cannot create the socket for sudo password prompts: %s\n", std::strerror(errno));
    return false;
  }
  socklen_t vLength = 0;
  const sockaddr_un vAddress = fSocketAddress(vSocketName, vLength);
  if (bind(vListener, reinterpret_cast<const sockaddr *>(&vAddress), vLength) != 0 || listen(vListener, 16) != 0) {
    SRV_WRN("shell MCP: cannot listen for sudo password prompts: %s\n", std::strerror(errno));
    close(vListener);
    vListener = -1;
    return false;
  }
  vAcceptThread = std::thread([this]() { mAcceptLoop(); });
  return true;
#else
  (void)pExecutable;
  return false;
#endif
}

std::shared_ptr<server_shell_mcp_call> server_shell_mcp_askpass::mRegister(const std::string &pCallId) {
  auto vCall = std::make_shared<server_shell_mcp_call>();
  vCall->vCallId = pCallId;
  vCall->vToken = fRandomHex(24);
  std::lock_guard<std::mutex> vLock(vMutex);
  lLCalls.push_back(vCall);
  return vCall;
}

void server_shell_mcp_askpass::mUnregister(const std::shared_ptr<server_shell_mcp_call> &pCall) {
  std::lock_guard<std::mutex> vLock(vMutex);
  pCall->vFinished = true;
  lLCalls.erase(std::remove(lLCalls.begin(), lLCalls.end(), pCall), lLCalls.end());
  vCondition.notify_all();
}

std::vector<std::string> server_shell_mcp_askpass::mEnvironment(const server_shell_mcp_call &pCall) const {
  std::vector<std::string> lLEntries = {"SUDO_ASKPASS=" + vExecutable,
                                        std::string(cAskpassVariable) + "=" + vSocketName + " " + pCall.vToken};
  const char *vDisplay = std::getenv("DISPLAY");
  if (vDisplay == nullptr || vDisplay[0] == '\0') {
    lLEntries.push_back(std::string("DISPLAY=") + cAskpassDisplay);
  }
  return lLEntries;
}

std::chrono::steady_clock::duration server_shell_mcp_askpass::mPaused(const server_shell_mcp_call &pCall) {
  std::lock_guard<std::mutex> vLock(vMutex);
  std::chrono::steady_clock::duration vPaused = pCall.vPaused;
  if (pCall.vPauseStart) {
    vPaused += std::chrono::steady_clock::now() - *pCall.vPauseStart;
  }
  return vPaused;
}

json server_shell_mcp_askpass::mPending(const std::string &pCallId) {
  std::lock_guard<std::mutex> vLock(vMutex);
  for (const auto &vCall : lLCalls) {
    if (vCall->vCallId == pCallId && !vCall->vPromptId.empty() && !vCall->vAnswered) {
      return {{"id", vCall->vPromptId}, {"prompt", vCall->vPrompt}};
    }
  }
  return nullptr;
}

bool server_shell_mcp_askpass::mAnswer(const std::string &pCallId, const std::string &pPromptId,
                                       const std::string *pPassword) {
  std::lock_guard<std::mutex> vLock(vMutex);
  for (const auto &vCall : lLCalls) {
    if (vCall->vCallId == pCallId && !vCall->vPromptId.empty() && vCall->vPromptId == pPromptId && !vCall->vAnswered) {
      vCall->vAnswered = true;
      vCall->vCancelled = pPassword == nullptr;
      if (pPassword != nullptr) {
        vCall->vPassword = *pPassword;
      }
      vCondition.notify_all();
      return true;
    }
  }
  return false;
}

void server_shell_mcp_askpass::mAcceptLoop() {
#if defined(__linux__)
  while (!vStopping) {
    pollfd vPoll = {vListener, POLLIN, 0};
    if (poll(&vPoll, 1, cPollMilliseconds) <= 0) {
      continue;
    }
    const int vClient = accept4(vListener, nullptr, nullptr, SOCK_CLOEXEC);
    if (vClient < 0) {
      continue;
    }
    {
      std::lock_guard<std::mutex> vLock(vMutex);
      vActiveHelpers++;
    }
    // one thread per helper: a prompt waits for the user while other commands keep asking
    std::thread([this, vClient]() {
      mServe(vClient);
      close(vClient);
      std::lock_guard<std::mutex> vLock(vMutex);
      vActiveHelpers--;
      vCondition.notify_all();
    }).detach();
  }
#endif
}

void server_shell_mcp_askpass::mServe(int pClient) {
#if defined(__linux__)
  fSetSocketTimeout(pClient, cAskpassSocketSeconds);
  ucred vPeer = {};
  socklen_t vPeerLength = sizeof(vPeer);
  if (getsockopt(pClient, SOL_SOCKET, SO_PEERCRED, &vPeer, &vPeerLength) != 0 || vPeer.uid != getuid()) {
    return;
  }
  std::string vRequest;
  if (!fReceiveLine(pClient, vRequest)) {
    return;
  }
  const json dRequest = json::parse(vRequest, nullptr, false);
  if (dRequest.is_discarded() || !dRequest.is_object()) {
    fReplyLine(pClient, "error", "invalid request");
    return;
  }
  const std::string vToken = fJsonValue(dRequest, "token", std::string());
  const std::string vPrompt = fCleanPrompt(fJsonValue(dRequest, "prompt", std::string()));
  if (!fParentIsSudo(vPeer.pid)) {
    SRV_WRN("%s", "shell MCP: refused a password request that does not come from sudo\n");
    fReplyLine(pClient, "error", "only sudo can ask for the password");
    return;
  }

  std::unique_lock<std::mutex> vLock(vMutex);
  std::shared_ptr<server_shell_mcp_call> vCall;
  for (const auto &vCandidate : lLCalls) {
    if (!vToken.empty() && vCandidate->vToken == vToken) {
      vCall = vCandidate;
      break;
    }
  }
  if (!vCall) {
    vLock.unlock();
    fReplyLine(pClient, "error",
               "the command of this prompt has ended or was not started by the local shell MCP server");
    return;
  }
  if (!vCall->vPromptId.empty()) {
    vLock.unlock();
    fReplyLine(pClient, "error", "another password prompt of this command is pending");
    return;
  }
  vCall->vPromptId = fRandomHex(8);
  vCall->vPrompt = vPrompt.empty() ? std::string("Password:") : vPrompt;
  vCall->vAnswered = false;
  vCall->vCancelled = false;
  vCall->vPauseStart = std::chrono::steady_clock::now();
  SRV_INF("shell MCP: sudo is asking for a password in call %s\n", vCall->vCallId.c_str());

  const auto vGiveUp = *vCall->vPauseStart + std::chrono::seconds(cAskpassWaitSeconds);
  bool vGone = false;
  while (!vCall->vAnswered && !vCall->vFinished && !vStopping && !vGone && std::chrono::steady_clock::now() < vGiveUp) {
    vCondition.wait_for(vLock, std::chrono::milliseconds(cPollMilliseconds));
    // the helper sends nothing more: readable means it closed (sudo was killed by a timeout or a cancellation)
    pollfd vPoll = {pClient, POLLIN, 0};
    vGone = poll(&vPoll, 1, 0) > 0;
  }
  const bool vGiven = vCall->vAnswered && !vCall->vCancelled;
  const char *vReason = "";
  if (vCall->vCancelled) {
    vReason = "the user cancelled the password prompt";
  } else if (!vGiven && (vCall->vFinished || vStopping || vGone)) {
    vReason = "the command ended";
  } else if (!vGiven) {
    vReason = "nobody answered the password prompt";
  }
  std::string vPassword;
  vPassword.swap(vCall->vPassword);
  vCall->vPaused += std::chrono::steady_clock::now() - *vCall->vPauseStart;
  vCall->vPauseStart.reset();
  vCall->vPromptId.clear();
  vCall->vPrompt.clear();
  vCall->vAnswered = false;
  vCall->vCancelled = false;
  const std::string vCallId = vCall->vCallId;
  vLock.unlock();

  if (vGiven) {
    fReplyLine(pClient, "password", std::move(vPassword));
    SRV_INF("shell MCP: the password of call %s was sent to sudo\n", vCallId.c_str());
  } else {
    fReplyLine(pClient, "error", vReason);
    SRV_INF("shell MCP: password prompt of call %s not answered: %s\n", vCallId.c_str(), vReason);
  }
  fWipe(vPassword);
#else
  (void)pClient;
#endif
}

int fServerShellMcpAskpassMain(int pArgumentCount, char **pArguments) {
#if defined(__linux__)
  const char *vVariable = std::getenv(cAskpassVariable);
  if (pArgumentCount != 2 || vVariable == nullptr || !fParentIsSudo(getpid())) {
    return -1;
  }
  const std::string vValue = vVariable;
  const size_t vSeparator = vValue.find(' ');
  const int vClient = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  socklen_t vLength = 0;
  const sockaddr_un vAddress = fSocketAddress(vValue.substr(0, vSeparator), vLength);
  if (vSeparator == std::string::npos || vClient < 0 ||
      connect(vClient, reinterpret_cast<const sockaddr *>(&vAddress), vLength) != 0) {
    fprintf(stderr, "mgs: the local shell MCP server that started this command cannot be reached\n");
    if (vClient >= 0) {
      close(vClient);
    }
    return 1;
  }
  fSetSocketTimeout(vClient, cAskpassSocketSeconds);
  const json dRequest = {{"token", vValue.substr(vSeparator + 1)}, {"prompt", std::string(pArguments[1])}};
  if (!fSendAll(vClient, fSafeJsonToStr(dRequest) + "\n")) {
    fprintf(stderr, "mgs: cannot send the password prompt to the local shell MCP server\n");
    close(vClient);
    return 1;
  }
  // the answer arrives when the user types the password: wait a little longer than the server does
  fSetSocketTimeout(vClient, cAskpassWaitSeconds + 30);
  std::string vReply;
  const bool vReceived = fReceiveLine(vClient, vReply);
  close(vClient);
  json dReply = vReceived ? json::parse(vReply, nullptr, false) : json();
  fWipe(vReply);
  if (dReply.is_object() && dReply.contains("password") && dReply.at("password").is_string()) {
    std::string &vPassword = dReply.at("password").get_ref<std::string &>();
    // straight to the pipe of sudo, without stdio buffers
    const bool vWritten =
        write(STDOUT_FILENO, vPassword.data(), vPassword.size()) == static_cast<ssize_t>(vPassword.size()) &&
        write(STDOUT_FILENO, "\n", 1) == 1;
    fWipe(vPassword);
    return vWritten ? 0 : 1;
  }
  const std::string vError = dReply.is_object() ? fJsonValue(dReply, "error", std::string("no answer"))
                                                : std::string("no answer from the local shell MCP server");
  fprintf(stderr, "mgs: %s\n", vError.c_str());
  return 1;
#else
  (void)pArgumentCount;
  (void)pArguments;
  return -1;
#endif
}

server_shell_mcp::server_shell_mcp() = default;

server_shell_mcp::~server_shell_mcp() = default;

size_t server_shell_mcp::mStart() {
  lLShells.clear();
#if defined(_WIN32)
  SRV_WRN("%s", "the shell MCP server needs a POSIX system; no shell tools are exposed\n");
#else
  struct candidate {
    const char *vTool;
    const char *vTitle;
    std::vector<std::string> lLNames;
    std::vector<std::string> lLArguments;
  };
  const std::vector<candidate> lLCandidates = {
      {"bash.run", "Bash", {"bash"}, {"-c"}},
      {"zsh.run", "Zsh", {"zsh"}, {"-c"}},
      {"sh.run", "sh", {"sh"}, {"-c"}},
      {"psh.run", "PowerShell", {"pwsh", "powershell"}, {"-NoLogo", "-NoProfile", "-NonInteractive", "-Command"}},
  };
  for (const auto &vCandidate : lLCandidates) {
    const std::string vPath = fFindExecutable(vCandidate.lLNames);
    if (vPath.empty()) {
      SRV_INF("shell MCP: %s not found, %s is not exposed\n", vCandidate.vTitle, vCandidate.vTool);
      continue;
    }
    lLShells.push_back({vCandidate.vTool, vCandidate.vTitle, vPath, vCandidate.lLArguments});
    SRV_INF("shell MCP: %s -> %s\n", vCandidate.vTool, vPath.c_str());
  }

  const passwd *vAccount = getpwuid(geteuid());
  vUserName = vAccount != nullptr && vAccount->pw_name != nullptr ? vAccount->pw_name : "";
  if (vUserName.empty() && std::getenv("USER") != nullptr) {
    vUserName = std::getenv("USER");
  }
  std::error_code vError;
  const char *vHome = std::getenv("HOME");
  if (vHome != nullptr && fs::is_directory(vHome, vError)) {
    vHomeDirectory = vHome;
  } else if (vAccount != nullptr && vAccount->pw_dir != nullptr && fs::is_directory(vAccount->pw_dir, vError)) {
    vHomeDirectory = vAccount->pw_dir;
  } else {
    vHomeDirectory = fs::current_path(vError).string();
  }
  char aHostName[256] = {};
  if (gethostname(aHostName, sizeof(aHostName) - 1) == 0) {
    vHostName = aHostName;
  }
  if (lLShells.empty()) {
    SRV_WRN("%s", "shell MCP: no shell was found; the server exposes no tools\n");
  }
#endif
#if defined(__linux__)
  // sudo runs this same executable as its password helper; the launcher of the self-contained bundle publishes its
  // own path, because the process image is the bundled dynamic loader
  vAskpass.reset();
  if (!lLShells.empty() && !fFindExecutable({"sudo"}).empty()) {
    const char *vLauncher = std::getenv("MGS_EXECUTABLE");
    std::error_code vLinkError;
    const std::string vExecutable = vLauncher != nullptr && vLauncher[0] != '\0'
                                        ? std::string(vLauncher)
                                        : fs::read_symlink("/proc/self/exe", vLinkError).string();
    auto vCandidate = std::make_unique<server_shell_mcp_askpass>();
    if (!vExecutable.empty() && vCandidate->mStart(vExecutable)) {
      vAskpass = std::move(vCandidate);
      SRV_INF("shell MCP: sudo password prompts are sent to the Web UI (SUDO_ASKPASS=%s)\n", vExecutable.c_str());
    }
  }
#endif
  return lLShells.size();
}

const server_shell_mcp_shell *server_shell_mcp::mFindShell(const std::string &pTool) const {
  for (const auto &vShell : lLShells) {
    if (vShell.vTool == pTool) {
      return &vShell;
    }
  }
  return nullptr;
}

json server_shell_mcp::mInitializeResult(const json &pParams) const {
  std::string vVersion = lLProtocolVersions.front();
  const std::string vRequested = fJsonValue(pParams, "protocolVersion", std::string());
  if (std::find(lLProtocolVersions.begin(), lLProtocolVersions.end(), vRequested) != lLProtocolVersions.end()) {
    vVersion = vRequested;
  }
  std::string vShellNames;
  for (const auto &vShell : lLShells) {
    vShellNames += (vShellNames.empty() ? "" : ", ") + vShell.vTool + " (" + vShell.vPath + ")";
  }
  const std::string vWhere = (vHostName.empty() ? std::string("the machine where mgs runs") : vHostName) +
                             (vUserName.empty() ? std::string() : " as user " + vUserName);
  return {
      {"protocolVersion", vVersion},
      {"capabilities", {{"tools", {{"listChanged", false}}}}},
      {"serverInfo",
       {{"name", "mgs-shell"},
        {"title", "local shell mcp server"},
        {"version", mgs_version()},
        {"icons", json::array({{{"src", fDataUri(cIconLight)}, {"mimeType", "image/svg+xml"}, {"theme", "light"}},
                               {{"src", fDataUri(cIconDark)}, {"mimeType", "image/svg+xml"}, {"theme", "dark"}}})}}},
      {"instructions",
       "Runs commands on " + vWhere +
           ", with the shells found when mgs started: " + (vShellNames.empty() ? std::string("none") : vShellNames) +
           ". Every call starts a new non-interactive process without a terminal and with empty input; the working "
           "directory and variables do not persist between calls, so pass cwd (default " +
           vHomeDirectory +
           ") or chain the commands. Each call answers a JSON object with exit_code, stdout and stderr. When the "
           "timeout expires every process started by the command is killed. Long outputs keep only their beginning "
           "and their end." +
           (vAskpass ? std::string(" sudo works without a terminal: when it needs a password, the mgs Web UI asks the "
                                   "user for it and the time spent waiting does not count against the timeout; you "
                                   "never see the password, and if the user cancels, sudo fails. Never pass "
                                   "passwords to sudo -S.")
                     : std::string())},
  };
}

json server_shell_mcp::mToolsListResult() const {
  json vLTools = json::array();
  for (const auto &vShell : lLShells) {
    const std::string vSyntax = vShell.vTool == "psh.run" ? "PowerShell" : vShell.vTitle;
    vLTools.push_back({
        {"name", vShell.vTool},
        {"title", "Run a " + vShell.vTitle + " command"},
        {"description",
         "Run a command or script with " + vShell.vTitle + " (" + vShell.vPath +
             ") on the machine where mgs runs and return a JSON object with its exit_code, stdout and stderr. Use " +
             vSyntax +
             " syntax. Each call is a new non-interactive process without a terminal: interactive programs cannot "
             "be used, input is empty, and the working directory and variables do not persist between calls. "
             "Background processes must redirect their output, because the call returns when the shell exits." +
             (vAskpass ? std::string(" When sudo needs a password, the user types it in the mgs Web UI.")
                       : std::string())},
        {"inputSchema",
         {{"type", "object"},
          {"properties",
           {{"command", {{"type", "string"}, {"description", "Command or script to run."}}},
            {"cwd", {{"type", "string"}, {"description", "Working directory (default: " + vHomeDirectory + ")."}}},
            {"timeout",
             {{"type", "integer"},
              {"minimum", 1},
              {"maximum", cMaxTimeoutSeconds},
              {"description", "Seconds before every process started by the command is killed (default " +
                                  std::to_string(cDefaultTimeoutSeconds) + ")."}}}}},
          {"required", json::array({"command"})},
          {"additionalProperties", false}}},
        {"annotations",
         {{"title", "Run a " + vShell.vTitle + " command"},
          {"readOnlyHint", false},
          {"destructiveHint", true},
          {"idempotentHint", false},
          {"openWorldHint", true}}},
    });
  }
  return {{"tools", vLTools}};
}

static json fToolText(const std::string &pText, bool pIsError) {
  return {{"content", json::array({{{"type", "text"}, {"text", pText}}})}, {"isError", pIsError}};
}

json server_shell_mcp::mToolsCallResult(const json &pParams, const std::function<bool()> &pShouldStop) const {
  const std::string vTool = fJsonValue(pParams, "name", std::string());
  const server_shell_mcp_shell *vShell = mFindShell(vTool);
  if (vShell == nullptr) {
    throw std::invalid_argument("unknown tool: " + vTool);
  }
  const json dArguments = pParams.contains("arguments") ? pParams.at("arguments") : json::object();
  const auto fArgument = [&dArguments](const char *pName) {
    return dArguments.is_object() && dArguments.contains(pName) ? dArguments.at(pName) : json(nullptr);
  };

  // Every call leaves its own file, also when its arguments are rejected: the command line holds what was asked
  // (cwd and timeout as the model sent them until they are checked) and the error line the reason.
  server_shell_mcp_call_log vLog;
  if (!vLog.mOpen(vHomeDirectory)) {
#if !defined(_WIN32)
    SRV_WRN("shell MCP: cannot create the call log in %s: %s\n", (fs::path(vHomeDirectory) / cCallLogFolder).c_str(),
            std::strerror(errno));
#endif
  }
  json dCommandLine = {
      {"event", "command"},
      {"tool", vTool},
      {"shell", vShell->vTitle},
      {"cwd", fArgument("cwd").is_null() ? json(vHomeDirectory) : fArgument("cwd")},
      {"timeout", fArgument("timeout").is_null() ? json(cDefaultTimeoutSeconds) : fArgument("timeout")},
      {"command", fArgument("command")}};
  const auto fRejected = [&vLog, &dCommandLine](const std::string &pReason) {
    vLog.mWrite(dCommandLine);
    vLog.mWrite({{"event", "error"}, {"error", pReason}});
    return fToolText(pReason, true);
  };

  if (!dArguments.is_object() || !dArguments.contains("command") || !dArguments.at("command").is_string()) {
    return fRejected("The argument \"command\" (string) is required.");
  }
  const std::string vCommand = dArguments.at("command").get<std::string>();
  if (vCommand.find('\0') != std::string::npos) {
    return fRejected("The command cannot contain NUL characters.");
  }
  std::string vDirectory = vHomeDirectory;
  if (dArguments.contains("cwd") && !dArguments.at("cwd").is_null()) {
    if (!dArguments.at("cwd").is_string()) {
      return fRejected("The argument \"cwd\" must be a string.");
    }
    const std::string vRequested = dArguments.at("cwd").get<std::string>();
    if (!vRequested.empty()) {
      vDirectory = fs::path(vRequested).is_absolute() ? vRequested : (fs::path(vHomeDirectory) / vRequested).string();
    }
  }
  dCommandLine["cwd"] = vDirectory;
  std::error_code vError;
  if (!fs::is_directory(vDirectory, vError)) {
    return fRejected("The working directory does not exist: " + vDirectory);
  }
  int vTimeout = cDefaultTimeoutSeconds;
  if (dArguments.contains("timeout") && !dArguments.at("timeout").is_null()) {
    if (!dArguments.at("timeout").is_number()) {
      return fRejected("The argument \"timeout\" must be a number of seconds.");
    }
    vTimeout = static_cast<int>(
        std::clamp(dArguments.at("timeout").get<double>(), 1.0, static_cast<double>(cMaxTimeoutSeconds)));
  }
  dCommandLine["timeout"] = vTimeout;

  // a client that names its call in _meta (the mgs Web UI) can answer the password prompts of sudo
  std::string vCallId;
  if (pParams.contains("_meta") && pParams.at("_meta").is_object()) {
    vCallId = fJsonValue(pParams.at("_meta"), cCallIdMeta, std::string());
  }
  std::shared_ptr<server_shell_mcp_call> vCall;
  std::vector<std::string> lLEnvironment;
  if (vAskpass && fValidCallId(vCallId)) {
    vCall = vAskpass->mRegister(vCallId);
    lLEnvironment = vAskpass->mEnvironment(*vCall);
  }
  const std::function<std::chrono::steady_clock::duration()> fPaused = [this, &vCall]() {
    return vCall ? vAskpass->mPaused(*vCall) : std::chrono::steady_clock::duration::zero();
  };

  // the command, before it runs, so the file holds it even if mgs stops meanwhile
  vLog.mWrite(dCommandLine);

  server_shell_mcp_run vRun;
#if defined(_WIN32)
  vRun.vError = "the shell MCP server needs a POSIX system";
#else
  try {
    fRunCommand(
        *vShell, vCommand, vDirectory, vTimeout, pShouldStop, lLEnvironment, fPaused,
        [&vLog](int pStream, const char *pData, size_t pSize) { vLog.mOutput(pStream, pData, pSize); }, vRun);
    vLog.mFlushOutput();
  } catch (...) {
    if (vCall) {
      vAskpass->mUnregister(vCall);
    }
    throw;
  }
#endif
  if (vCall) {
    vAskpass->mUnregister(vCall);
  }
  if (!vRun.vError.empty()) {
    SRV_WRN("shell MCP: %s failed in %s: %s: %s\n", vTool.c_str(), vDirectory.c_str(), vRun.vError.c_str(),
            fLogCommand(vCommand).c_str());
    vLog.mWrite({{"event", "error"}, {"error", vRun.vError}});
    return fToolText(vRun.vError, true);
  }
  SRV_INF("shell MCP: %s exit %d in %.2f s%s, cwd %s: %s\n", vTool.c_str(), vRun.vExitCode, vRun.vSeconds,
          vRun.vTimedOut ? " (timed out)" : (vRun.vCancelled ? " (cancelled)" : ""), vDirectory.c_str(),
          fLogCommand(vCommand).c_str());

  // The text block carries the same JSON object as structuredContent: the model reads it, and the Web UI splits it
  // into its stdin, stdout and stderr boxes without guessing where one stream ends and the next one begins.
  // exit_code is null when a killed shell could not be reaped in time.
  const json dResult = {
      {"exit_code", vRun.vExitCode >= 0 ? json(vRun.vExitCode) : json(nullptr)},
      {"stdout", vRun.vStdout.mText()},
      {"stderr", vRun.vStderr.mText()},
      {"signal", vRun.vSignal},
      {"timed_out", vRun.vTimedOut},
      {"cancelled", vRun.vCancelled},
      {"timeout", vTimeout},
      {"seconds", std::round(vRun.vSeconds * 100.0) / 100.0},
      {"shell", vShell->vTitle},
      {"cwd", vDirectory},
  };
  // the whole output is already in the stdout and stderr lines, so the result only counts its bytes
  vLog.mWrite({{"event", "result"},
               {"exit_code", dResult["exit_code"]},
               {"signal", vRun.vSignal},
               {"timed_out", vRun.vTimedOut},
               {"cancelled", vRun.vCancelled},
               {"seconds", dResult["seconds"]},
               {"stdout_bytes", vRun.vStdout.vTotal},
               {"stderr_bytes", vRun.vStderr.vTotal}});
  return {{"content", json::array({{{"type", "text"}, {"text", fSafeJsonToStr(dResult)}}})},
          {"structuredContent", dResult},
          {"isError", vRun.vTimedOut || vRun.vCancelled}};
}

static server_http_res_ptr fJsonResponse(int pStatus, const json &pBody) {
  auto vResponse = std::make_unique<server_http_res>();
  vResponse->vStatus = pStatus;
  vResponse->vContentType = "application/json; charset=utf-8";
  vResponse->vData = fSafeJsonToStr(pBody);
  return vResponse;
}

static json fRpcError(const json &pId, int pCode, const std::string &pMessage) {
  return {{"jsonrpc", "2.0"}, {"id", pId}, {"error", {{"code", pCode}, {"message", pMessage}}}};
}

server_http_res_ptr server_shell_mcp::mHandlePost(const server_http_req &pRequest) const {
  if (!fOriginAllowed(pRequest)) {
    SRV_WRN("shell MCP: rejected a request from origin %s\n", fHeader(pRequest.dHeaders, "Origin").c_str());
    return fJsonResponse(403, fRpcError(nullptr, cInvalidRequest, "Origin not allowed"));
  }
  if (fLower(fHeader(pRequest.dHeaders, "Content-Type")).rfind("application/json", 0) != 0) {
    return fJsonResponse(415, fRpcError(nullptr, cInvalidRequest, "Content-Type must be application/json"));
  }
  json dMessage;
  try {
    dMessage = json::parse(pRequest.vBody);
  } catch (const std::exception &vException) {
    return fJsonResponse(400, fRpcError(nullptr, cParseError, std::string("Parse error: ") + vException.what()));
  }
  if (dMessage.is_array()) {
    return fJsonResponse(400, fRpcError(nullptr, cInvalidRequest, "JSON-RPC batches are not supported"));
  }
  if (!dMessage.is_object()) {
    return fJsonResponse(400, fRpcError(nullptr, cInvalidRequest, "Invalid request"));
  }
  // notifications and responses to server requests are acknowledged without a body
  if (!dMessage.contains("id") || !dMessage.contains("method")) {
    auto vResponse = std::make_unique<server_http_res>();
    vResponse->vStatus = 202;
    vResponse->vContentType = "text/plain; charset=utf-8";
    return vResponse;
  }
  const json vId = dMessage.at("id");
  if (!dMessage.at("method").is_string()) {
    return fJsonResponse(400, fRpcError(vId, cInvalidRequest, "Invalid request"));
  }
  const std::string vMethod = dMessage.at("method").get<std::string>();
  const json dParams =
      dMessage.contains("params") && dMessage.at("params").is_object() ? dMessage.at("params") : json::object();
  try {
    json dResult;
    if (vMethod == "initialize") {
      dResult = mInitializeResult(dParams);
    } else if (vMethod == "ping") {
      dResult = json::object();
    } else if (vMethod == "tools/list") {
      dResult = mToolsListResult();
    } else if (vMethod == "tools/call") {
      dResult = mToolsCallResult(dParams, pRequest.vShouldStop);
    } else {
      return fJsonResponse(200, fRpcError(vId, cMethodNotFound, "Method not found: " + vMethod));
    }
    return fJsonResponse(200, {{"jsonrpc", "2.0"}, {"id", vId}, {"result", dResult}});
  } catch (const std::invalid_argument &vException) {
    return fJsonResponse(200, fRpcError(vId, cInvalidParams, vException.what()));
  } catch (const std::exception &vException) {
    return fJsonResponse(200, fRpcError(vId, cInvalidParams, std::string("Invalid params: ") + vException.what()));
  }
}

server_http_res_ptr server_shell_mcp::mHandleUnsupported(const server_http_req &) const {
  auto vResponse = fJsonResponse(405, fRpcError(nullptr, cInvalidRequest, "Method not allowed: use POST"));
  vResponse->dHeaders["Allow"] = "POST";
  return vResponse;
}

server_http_res_ptr server_shell_mcp::mHandleAskpassGet(const server_http_req &pRequest) const {
  if (!fOriginAllowed(pRequest)) {
    return fJsonResponse(403, {{"error", "Origin not allowed"}});
  }
  const std::string vCallId = pRequest.mGetParam("call");
  if (!fValidCallId(vCallId)) {
    return fJsonResponse(400, {{"error", "call must be the id sent in the _meta of tools/call"}});
  }
  return fJsonResponse(200, {{"pending", vAskpass ? vAskpass->mPending(vCallId) : json(nullptr)}});
}

server_http_res_ptr server_shell_mcp::mHandleAskpassPost(const server_http_req &pRequest) const {
  if (!fOriginAllowed(pRequest)) {
    SRV_WRN("shell MCP: rejected a password answer from origin %s\n", fHeader(pRequest.dHeaders, "Origin").c_str());
    return fJsonResponse(403, {{"error", "Origin not allowed"}});
  }
  if (fLower(fHeader(pRequest.dHeaders, "Content-Type")).rfind("application/json", 0) != 0) {
    return fJsonResponse(415, {{"error", "Content-Type must be application/json"}});
  }
  json dBody = json::parse(pRequest.vBody, nullptr, false);
  if (dBody.is_discarded() || !dBody.is_object()) {
    return fJsonResponse(400, {{"error", "the body must be a JSON object"}});
  }
  const std::string vCallId = fJsonValue(dBody, "call", std::string());
  const std::string vPromptId = fJsonValue(dBody, "id", std::string());
  const bool vCancel = fJsonValue(dBody, "cancel", false);
  const bool vHasPassword = dBody.contains("password") && dBody.at("password").is_string();
  if (!fValidCallId(vCallId) || vPromptId.empty() || vCancel == vHasPassword) {
    return fJsonResponse(400, {{"error", "send call, id and either password or cancel: true"}});
  }
  bool vAccepted = false;
  if (vHasPassword) {
    std::string &vPassword = dBody.at("password").get_ref<std::string &>();
    // sudo reads one line
    const bool vValid = vPassword.size() <= cAskpassPasswordBytes && vPassword.find('\n') == std::string::npos &&
                        vPassword.find('\r') == std::string::npos && vPassword.find('\0') == std::string::npos;
    if (vValid) {
      vAccepted = vAskpass && vAskpass->mAnswer(vCallId, vPromptId, &vPassword);
    }
    fWipe(vPassword);
    if (!vValid) {
      return fJsonResponse(400, {{"error", "the password must be one line of at most 4096 bytes"}});
    }
  } else {
    vAccepted = vAskpass && vAskpass->mAnswer(vCallId, vPromptId, nullptr);
  }
  return fJsonResponse(vAccepted ? 200 : 404, {{"accepted", vAccepted}});
}
