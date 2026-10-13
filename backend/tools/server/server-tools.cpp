#include "server-tools.h"

#include "subproc.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <memory>

namespace fs = std::filesystem;

//
// internal helpers
//

json server_tool::to_json() const {
  return {
      {"display_name", vDisplayName},
      {"tool", vName},
      {"type", mType()},
      {"permissions", json{{"write", vPermissionWrite}}},
      {"definition", mGetDefinition()},
  };
}

static constexpr size_t cServerToolGitLsFilesMaxOutput = 8 * 1024 * 1024; // 8 MB
static constexpr int cServerToolGitLsFilesTimeout = 15;                   // seconds

class tools_io {
public:
  struct exec_result {
    std::string vOutput;
    int vExitCode = -1;
    bool vTimedOut = false;
  };

  virtual ~tools_io() = default;

  virtual bool mIsDirectory(const std::string &pPath) const = 0;
  virtual bool mIsRegularFile(const std::string &pPath) const = 0;
  virtual bool mFileSize(const std::string &pPath, uintmax_t &pOutSize) const = 0;
  virtual bool mReadFile(const std::string &pPath, std::string &pOut) const = 0;
  virtual bool mWriteFile(const std::string &pPath, const std::string &pContent) const = 0;
  // paths relative to `pBase`, '/'-separated; sets `pErr` if `pBase` isn't a directory
  virtual std::vector<std::string> mListFiles(const std::string &pBase, std::string &pErr) const = 0;
  // pOnChunk, if set, is called with each chunk of output as it is read (before truncation cuts in);
  // returning false terminates the process early (e.g. the client disconnected)
  virtual exec_result mRun(const std::vector<std::string> &pArgs, size_t pMaxOutput, int pTimeoutSecs,
                           const std::function<bool(const std::string &)> &pOnChunk = nullptr) const = 0;
};

class tools_io_basic : public tools_io {
public:
  // pCwd, if non-empty, is used to resolve relative paths and as the working directory for mRun()
  explicit tools_io_basic(std::string pCwd = "") : vCwd(std::move(pCwd)) {}

  bool mIsDirectory(const std::string &pPath) const override {
    std::error_code vEc;
    return fs::is_directory(mResolve(pPath), vEc) && !vEc;
  }

  bool mIsRegularFile(const std::string &pPath) const override {
    std::error_code vEc;
    return fs::is_regular_file(mResolve(pPath), vEc) && !vEc;
  }

  bool mFileSize(const std::string &pPath, uintmax_t &pOutSize) const override {
    std::error_code vEc;
    pOutSize = fs::file_size(mResolve(pPath), vEc);
    return !vEc;
  }

  bool mReadFile(const std::string &pPath, std::string &pOut) const override {
    std::ifstream vF(mResolve(pPath), std::ios::binary);
    if (!vF)
      return false;
    std::ostringstream vSs;
    vSs << vF.rdbuf();
    pOut = vSs.str();
    return true;
  }

  bool mWriteFile(const std::string &pPath, const std::string &pContent) const override {
    std::error_code vEc;
    fs::path vFpath(mResolve(pPath));
    if (vFpath.has_parent_path()) {
      fs::create_directories(vFpath.parent_path(), vEc);
      if (vEc)
        return false;
    }
    std::ofstream vF(vFpath, std::ios::binary);
    if (!vF)
      return false;
    vF << pContent;
    return (bool)vF;
  }

  std::vector<std::string> mListFiles(const std::string &pBase, std::string &pErr) const override {
    pErr.clear();
    std::string vAbsBase = mResolve(pBase);
    if (!mIsDirectory(pBase)) {
      pErr = "path does not exist or is not a directory: " + pBase;
      return {};
    }

    auto vRes = mRun({"git", "-C", vAbsBase, "ls-files", "--cached", "--others", "--exclude-standard"},
                     cServerToolGitLsFilesMaxOutput, cServerToolGitLsFilesTimeout);

    if (vRes.vExitCode == 0 && !vRes.vTimedOut) {
      std::vector<std::string> lResult;
      std::istringstream vIss(vRes.vOutput);
      std::string vLine;
      while (std::getline(vIss, vLine)) {
        if (!vLine.empty() && vLine.back() == '\r')
          vLine.pop_back();
        if (vLine.empty())
          continue;
        std::replace(vLine.begin(), vLine.end(), '\\', '/');
        if (mIsRegularFile((fs::path(pBase) / vLine).string())) {
          lResult.push_back(vLine);
        }
      }
      return lResult;
    }

    return mListFilesFallback(vAbsBase);
  }

  exec_result mRun(const std::vector<std::string> &pArgs, size_t pMaxOutput, int pTimeoutSecs,
                   const std::function<bool(const std::string &)> &pOnChunk = nullptr) const override {
    exec_result vRes;

    common_subproc vProc;

    int vOptions = subprocess_option_no_window | subprocess_option_combined_stdout_stderr |
                   subprocess_option_inherit_environment | subprocess_option_search_user_path;

    if (!vProc.mCreate(pArgs, vOptions, {}, vCwd.empty() ? nullptr : vCwd.c_str())) {
      vRes.vOutput = "failed to spawn process";
      return vRes;
    }

    std::atomic<bool> vDone{false};
    std::atomic<bool> vTimedOut{false};

    std::thread vTimeoutThread([&]() {
      auto vDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(pTimeoutSecs);
      while (!vDone.load()) {
        if (std::chrono::steady_clock::now() >= vDeadline) {
          vTimedOut.store(true);
          vProc.mTerminate();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });

    FILE *vF = vProc.mStdoutFile();
    std::string vOutput;
    bool vTruncated = false;
    if (vF) {
      char aBuf[4096];
      while (fgets(aBuf, sizeof(aBuf), vF) != nullptr) {
        if (!vTruncated) {
          size_t vLen = strlen(aBuf);
          if (vOutput.size() + vLen <= pMaxOutput) {
            vOutput.append(aBuf, vLen);
            if (pOnChunk && !pOnChunk(std::string(aBuf, vLen))) {
              vProc.mTerminate();
              break;
            }
          } else {
            size_t vRemaining = pMaxOutput - vOutput.size();
            vOutput.append(aBuf, vRemaining);
            if (pOnChunk && vRemaining > 0)
              pOnChunk(std::string(aBuf, vRemaining));
            vTruncated = true;
          }
        }
      }
    }

    vDone.store(true);
    if (vTimeoutThread.joinable()) {
      vTimeoutThread.join();
    }

    vRes.vExitCode = vProc.mJoin();

    vRes.vOutput = vOutput;
    vRes.vTimedOut = vTimedOut.load();
    if (vTruncated) {
      vRes.vOutput += "\n[output truncated]";
    }
    return vRes;
  }

private:
  std::string vCwd;

  // resolves `pPath` against `vCwd` if `pPath` is relative and `vCwd` is set; otherwise returns `pPath` unchanged
  std::string mResolve(const std::string &pPath) const {
    if (vCwd.empty() || fs::path(pPath).is_absolute()) {
      return pPath;
    }
    return (fs::path(vCwd) / pPath).string();
  }

  static const std::unordered_set<std::string> &mJunkDirNames() {
    static const std::unordered_set<std::string> sNames = {
        ".git", ".svn",  ".hg",    "node_modules", "__pycache__", ".venv",   "venv",
        "dist", "build", "target", ".cache",       ".idea",       ".vscode",
    };
    return sNames;
  }

  std::vector<std::string> mListFilesFallback(const std::string &pBase) const {
    std::vector<std::string> lResult;
    std::error_code vEc;

    std::vector<std::pair<fs::path, fs::path>> lStack;
    lStack.emplace_back(fs::path(pBase), fs::path());

    while (!lStack.empty()) {
      auto [vDir, vRelDir] = lStack.back();
      lStack.pop_back();

      for (const auto &vEntry : fs::directory_iterator(vDir, fs::directory_options::skip_permission_denied, vEc)) {
        if (vEc)
          break;
        std::string vFname = vEntry.path().filename().string();
        std::error_code vTec;
        if (vEntry.is_directory(vTec)) {
          if (mJunkDirNames().count(vFname) > 0)
            continue;
          lStack.emplace_back(vEntry.path(), vRelDir / vFname);
        } else if (vEntry.is_regular_file(vTec)) {
          std::string vRel = (vRelDir / vFname).string();
          std::replace(vRel.begin(), vRel.end(), '\\', '/');
          lResult.push_back(vRel);
        }
      }
    }

    return lResult;
  }
};

static std::unique_ptr<tools_io> fMakeToolsIo(const json &pParams) {
  std::string vCwd = fJsonValue(pParams, "cwd", std::string());
  return std::make_unique<tools_io_basic>(vCwd);
}

// no '/' in pattern -> match basename at any depth; else match full relative path
static bool fPathGlobMatch(const std::string &pPattern, const std::string &pRelPath) {
  if (pPattern.find('/') == std::string::npos) {
    return fGlobMatch(pPattern, fs::path(pRelPath).filename().string());
  }
  if (pPattern == "**" || pPattern.rfind("**/", 0) == 0 || pPattern.rfind('/', 0) == 0) {
    return fGlobMatch(pPattern, pRelPath);
  }
  return fGlobMatch("**/" + pPattern, pRelPath);
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t cServerToolReadFileMaxSize = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
  server_tool_read_file() {
    vName = "read_file";
    vDisplayName = "Read file";
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description",
              "Read the contents of a file. Optionally specify a 1-based line range. "
              "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192...\")."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"path", {{"type", "string"}, {"description", "Path to the file"}}},
                       {"start_line",
                        {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                       {"end_line",
                        {{"type", "integer"},
                         {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                       {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                   }},
                  {"required", json::array({"path"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath = pParams.at("path").get<std::string>();
    int vStartLine = fJsonValue(pParams, "start_line", 1);
    int vEndLine = fJsonValue(pParams, "end_line", -1); // -1 = no limit
    bool vAppendLoc = fJsonValue(pParams, "append_loc", false);

    auto vIo = fMakeToolsIo(pParams);

    uintmax_t vFileSize = 0;
    if (!vIo->mFileSize(vPath, vFileSize)) {
      return {{"error", "cannot stat file: " + vPath}};
    }
    if (vFileSize > cServerToolReadFileMaxSize && vEndLine == -1) {
      return {{"error", fStringFormat("file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                                      (size_t)vFileSize, cServerToolReadFileMaxSize)}};
    }

    std::string vContent;
    if (!vIo->mReadFile(vPath, vContent)) {
      return {{"error", "failed to open file: " + vPath}};
    }

    std::istringstream vF(vContent);
    std::string vResult;
    std::string vLine;
    int vLineno = 0;

    while (std::getline(vF, vLine)) {
      vLineno++;
      if (vLineno < vStartLine)
        continue;
      if (vEndLine != -1 && vLineno > vEndLine)
        break;

      std::string vOutLine;
      if (vAppendLoc) {
        vOutLine = std::to_string(vLineno) + "\u2192" + vLine + "\n";
      } else {
        vOutLine = vLine + "\n";
      }

      if (vResult.size() + vOutLine.size() > cServerToolReadFileMaxSize) {
        vResult += "[output truncated]";
        break;
      }
      vResult += vOutLine;
    }

    return {{"plain_text_response", vResult}};
  }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t cServerToolFileSearchMaxResults = 100;

struct server_tool_file_glob_search : server_tool {
  server_tool_file_glob_search() {
    vName = "file_glob_search";
    vDisplayName = "File search";
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description",
              "Recursively search for files matching a glob pattern under a directory. "
              "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
              "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
              "A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. "
              "A pattern containing '/' matches the full relative path; unless already anchored with "
              "\"**/\" or a leading '/', it is automatically prefixed with \"**/\"."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"path", {{"type", "string"}, {"description", "Base directory to search in"}}},
                       {"include",
                        {{"type", "string"},
                         {"description",
                          "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"}}},
                       {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                   }},
                  {"required", json::array({"path"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vBase = pParams.at("path").get<std::string>();
    std::string vInclude = fJsonValue(pParams, "include", std::string("**"));
    std::string vExclude = fJsonValue(pParams, "exclude", std::string(""));

    auto vIo = fMakeToolsIo(pParams);
    std::string vErr;
    auto vFiles = vIo->mListFiles(vBase, vErr);
    if (!vErr.empty()) {
      return {{"error", vErr}};
    }

    std::vector<std::string> lMatches;
    for (const auto &vRel : vFiles) {
      if (!fPathGlobMatch(vInclude, vRel))
        continue;
      if (!vExclude.empty() && fPathGlobMatch(vExclude, vRel))
        continue;
      lMatches.push_back(vRel);
    }

    size_t vTotal = lMatches.size();
    size_t vShown = std::min(vTotal, cServerToolFileSearchMaxResults);

    std::ostringstream vOutputText;
    for (size_t vI = 0; vI < vShown; vI++) {
      vOutputText << lMatches[vI] << "\n";
    }

    vOutputText << "\n---\nTotal matches: " << vTotal << "\n";
    if (vTotal > vShown) {
      vOutputText << fStringFormat(
          "[%zu results limit reached (%zu total matches). Refine the glob pattern to narrow the search.]\n", vShown,
          vTotal);
    }

    return {{"plain_text_response", vOutputText.str()}};
  }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t cServerToolGrepSearchMaxResults = 100;

struct server_tool_grep_search : server_tool {
  server_tool_grep_search() {
    vName = "grep_search";
    vDisplayName = "Grep search";
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description",
              "Search for a pattern in files under a path. Returns matching lines with file paths "
              "(and, unless searching a single file, paths relative to the given directory). "
              "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
              "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
              "include/exclude: a pattern with no '/' matches the basename at any depth; a pattern "
              "containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored)."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"path", {{"type", "string"}, {"description", "File or directory to search in"}}},
                       {"pattern",
                        {{"type", "string"},
                         {"description", "Pattern to search for (regular expression unless literal is true)"}}},
                       {"include", {{"type", "string"}, {"description", "Glob pattern to filter files (default: **)"}}},
                       {"exclude", {{"type", "string"}, {"description", "Glob pattern to exclude files"}}},
                       {"return_line_numbers",
                        {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                       {"literal",
                        {{"type", "boolean"},
                         {"description",
                          "Treat pattern as a literal string instead of a regular expression (default: false)"}}},
                       {"ignore_case",
                        {{"type", "boolean"}, {"description", "Case-insensitive search (default: false)"}}},
                       {"context_lines",
                        {{"type", "integer"},
                         {"description",
                          "Number of lines of context to show before and after each match (default: 0)"}}},
                   }},
                  {"required", json::array({"path", "pattern"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath = pParams.at("path").get<std::string>();
    std::string vPatStr = pParams.at("pattern").get<std::string>();
    std::string vInclude = fJsonValue(pParams, "include", std::string("**"));
    std::string vExclude = fJsonValue(pParams, "exclude", std::string(""));
    bool vShowLineno = fJsonValue(pParams, "return_line_numbers", false);
    bool vLiteral = fJsonValue(pParams, "literal", false);
    bool vIgnoreCase = fJsonValue(pParams, "ignore_case", false);
    int vCtxLines = std::max(0, fJsonValue(pParams, "context_lines", 0));

    std::string vPatternSrc = vPatStr;
    if (vLiteral) {
      static const std::string cSpecials = "\\^$.|?*+()[]{}";
      std::string vEscaped;
      vEscaped.reserve(vPatStr.size() * 2);
      for (char vC : vPatStr) {
        if (cSpecials.find(vC) != std::string::npos)
          vEscaped += '\\';
        vEscaped += vC;
      }
      vPatternSrc = vEscaped;
    }

    std::regex vPattern;
    try {
      auto vFlags = std::regex::ECMAScript;
      if (vIgnoreCase)
        vFlags |= std::regex::icase;
      vPattern = std::regex(vPatternSrc, vFlags);
    } catch (const std::regex_error &vE) {
      return {{"error", std::string("invalid regex: ") + vE.what()}};
    }

    auto vIo = fMakeToolsIo(pParams);

    // collect (absolute_path, display_path) pairs to search
    std::vector<std::pair<std::string, std::string>> lFiles;

    if (vIo->mIsRegularFile(vPath)) {
      lFiles.emplace_back(vPath, vPath);
    } else if (vIo->mIsDirectory(vPath)) {
      std::string vErr;
      auto vCandidates = vIo->mListFiles(vPath, vErr);
      if (!vErr.empty()) {
        return {{"error", vErr}};
      }
      for (const auto &vRel : vCandidates) {
        if (!fPathGlobMatch(vInclude, vRel))
          continue;
        if (!vExclude.empty() && fPathGlobMatch(vExclude, vRel))
          continue;
        lFiles.emplace_back((fs::path(vPath) / vRel).string(), vRel);
      }
    } else {
      return {{"error", "path does not exist: " + vPath}};
    }

    std::ostringstream vOutputText;
    size_t vTotal = 0;
    bool vLimitReached = false;
    bool vShowNum = vShowLineno || vCtxLines > 0;

    for (const auto &vFileEntry : lFiles) {
      if (vLimitReached)
        break;
      const std::string &vFpath = vFileEntry.first;
      const std::string &vDisplayPath = vFileEntry.second;

      std::string vContent;
      if (!vIo->mReadFile(vFpath, vContent))
        continue;
      std::vector<std::string> lLines;
      {
        std::istringstream vF(vContent);
        std::string vLine;
        while (std::getline(vF, vLine))
          lLines.push_back(vLine);
      }

      for (size_t vI = 0; vI < lLines.size(); vI++) {
        if (vTotal >= cServerToolGrepSearchMaxResults) {
          vLimitReached = true;
          break;
        }
        if (!std::regex_search(lLines[vI], vPattern))
          continue;

        long vCtxStart = vCtxLines > 0 ? std::max<long>(0, (long)vI - vCtxLines) : (long)vI;
        long vCtxEnd = vCtxLines > 0 ? std::min<long>((long)lLines.size() - 1, (long)vI + vCtxLines) : (long)vI;

        for (long vJ = vCtxStart; vJ <= vCtxEnd; vJ++) {
          bool vIsMatch = (vJ == (long)vI);
          vOutputText << vDisplayPath << (vIsMatch ? ':' : '-');
          if (vShowNum) {
            vOutputText << (vJ + 1) << (vIsMatch ? ':' : '-');
          }
          vOutputText << lLines[vJ] << "\n";
        }
        if (vCtxLines > 0) {
          vOutputText << "--\n";
        }
        vTotal++;
      }
    }

    vOutputText << "\n---\nTotal matches: " << vTotal << "\n";
    if (vLimitReached) {
      vOutputText << fStringFormat("[%zu matches limit reached. Narrow the path/pattern/include to see more.]\n",
                                   cServerToolGrepSearchMaxResults);
    }

    return {{"plain_text_response", vOutputText.str()}};
  }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t cServerToolExecShellCommandMaxOutputSize = 16 * 1024; // 16 KB
static constexpr int cServerToolExecShellCommandMaxTimeout = 60;              // seconds

struct server_tool_exec_shell_command : server_tool {
  server_tool_exec_shell_command() {
    vName = "exec_shell_command";
    vDisplayName = "Execute shell command";
    vPermissionWrite = true;
    vSupportStream = true;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"command", {{"type", "string"}, {"description", "Shell command to execute"}}},
                       {"timeout",
                        {{"type", "integer"},
                         {"description", fStringFormat("Timeout in seconds (default 10, max %d)",
                                                       cServerToolExecShellCommandMaxTimeout)}}},
                       {"max_output_size",
                        {{"type", "integer"},
                         {"description", fStringFormat("Maximum output size in bytes (default %zu)",
                                                       cServerToolExecShellCommandMaxOutputSize)}}},
                   }},
                  {"required", json::array({"command"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *pSt) const override {
    std::string vCommand = pParams.at("command").get<std::string>();
    int vTimeout = fJsonValue(pParams, "timeout", 10);
    size_t vMaxOutput = (size_t)fJsonValue(pParams, "max_output_size", (int)cServerToolExecShellCommandMaxOutputSize);

    vTimeout = std::min(vTimeout, cServerToolExecShellCommandMaxTimeout);
    vMaxOutput = std::min(vMaxOutput, cServerToolExecShellCommandMaxOutputSize);

#ifdef _WIN32
    std::vector<std::string> lArgs = {"cmd", "/c", vCommand};
#else
    std::vector<std::string> lArgs = {"sh", "-c", vCommand};
#endif

    auto vIo = fMakeToolsIo(pParams);

    if (pSt) {
      auto vRes = vIo->mRun(lArgs, vMaxOutput, vTimeout, [pSt](const std::string &pChunk) {
        pSt->mPush(pChunk);
        return !pSt->vAlive || pSt->vAlive();
      });
      if (pSt->vAlive && !pSt->vAlive()) {
        return json();
      }
      std::string vTail = fStringFormat("\n[exit code: %d]", vRes.vExitCode);
      if (vRes.vTimedOut) {
        vTail += " [exit due to timed out]";
      }
      pSt->mPush(vTail);
      return json();
    }

    auto vRes = vIo->mRun(lArgs, vMaxOutput, vTimeout);

    std::string vTextOutput = vRes.vOutput;
    vTextOutput += fStringFormat("\n[exit code: %d]", vRes.vExitCode);
    if (vRes.vTimedOut) {
      vTextOutput += " [exit due to timed out]";
    }

    return {{"plain_text_response", vTextOutput}};
  }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
  server_tool_write_file() {
    vName = "write_file";
    vDisplayName = "Write file";
    vPermissionWrite = true;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description", "Write content to a file, creating it (including parent directories) if it does not "
                             "exist. May use with edit_file for more complex edits."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"path", {{"type", "string"}, {"description", "Path of the file to write"}}},
                       {"content", {{"type", "string"}, {"description", "Content to write"}}},
                   }},
                  {"required", json::array({"path", "content"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath = pParams.at("path").get<std::string>();
    std::string vContent = pParams.at("content").get<std::string>();

    auto vIo = fMakeToolsIo(pParams);
    if (!vIo->mWriteFile(vPath, vContent)) {
      return {{"error", "failed to write file: " + vPath}};
    }

    return {{"result", "file written successfully"}, {"path", vPath}, {"bytes", vContent.size()}};
  }
};

//
// edit_file: exact text replacement, one or more edits per call
//

struct server_tool_edit_file : server_tool {
  server_tool_edit_file() {
    vName = "edit_file";
    vDisplayName = "Edit file";
    vPermissionWrite = true;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description",
              "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file "
              "and is matched against the original content, not incrementally. Merge nearby changes into "
              "one edit instead of overlapping edits. Use write_file to replace the whole file."},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"path", {{"type", "string"}, {"description", "Path to the file to edit"}}},
                       {"edits",
                        {
                            {"type", "array"},
                            {"description", "One or more exact text replacements to apply"},
                            {"items",
                             {
                                 {"type", "object"},
                                 {"properties",
                                  {
                                      {"old_text",
                                       {{"type", "string"},
                                        {"description", "Exact text to find; must be unique in the file and must not "
                                                        "overlap with other edits"}}},
                                      {"new_text",
                                       {{"type", "string"}, {"description", "Text to replace old_text with"}}},
                                  }},
                                 {"required", json::array({"old_text", "new_text"})},
                             }},
                        }},
                   }},
                  {"required", json::array({"path", "edits"})},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath = pParams.at("path").get<std::string>();
    const json &vEditsJson = pParams.at("edits");

    if (!vEditsJson.is_array() || vEditsJson.empty()) {
      return {{"error", "\"edits\" must be a non-empty array"}};
    }

    struct edit_req {
      std::string vOldText;
      std::string vNewText;
    };
    std::vector<edit_req> lEdits;
    lEdits.reserve(vEditsJson.size());
    for (const auto &vE : vEditsJson) {
      edit_req vEr;
      vEr.vOldText = vE.at("old_text").get<std::string>();
      vEr.vNewText = vE.at("new_text").get<std::string>();
      if (vEr.vOldText.empty()) {
        return {{"error", fStringFormat("edits[%zu].old_text must not be empty", lEdits.size())}};
      }
      lEdits.push_back(std::move(vEr));
    }

    auto vIo = fMakeToolsIo(pParams);
    std::string vOriginalContent;
    if (!vIo->mReadFile(vPath, vOriginalContent)) {
      return {{"error", "failed to open file: " + vPath}};
    }

    // does any old_text need fuzzy matching (no exact match found)?
    bool vAnyFuzzy = false;
    for (size_t vI = 0; vI < lEdits.size(); vI++) {
      if (vOriginalContent.find(lEdits[vI].vOldText) != std::string::npos)
        continue;
      std::string vFuzzyContent = mNormalizeForFuzzyMatch(vOriginalContent);
      std::string vFuzzyOld = mNormalizeForFuzzyMatch(lEdits[vI].vOldText);
      if (vFuzzyContent.find(vFuzzyOld) == std::string::npos) {
        return {
            {"error",
             fStringFormat("could not find edits[%zu].old_text in %s, it must match the file's current content exactly",
                           vI, vPath.c_str())}};
      }
      vAnyFuzzy = true;
    }

    std::string vBaseContent = vAnyFuzzy ? mNormalizeForFuzzyMatch(vOriginalContent) : vOriginalContent;

    // uniqueness check always uses fuzzy-normalized text, so a whitespace-only duplicate still counts
    std::vector<matched_edit> lMatched;
    lMatched.reserve(lEdits.size());
    for (size_t vI = 0; vI < lEdits.size(); vI++) {
      std::string vNeedle = vAnyFuzzy ? mNormalizeForFuzzyMatch(lEdits[vI].vOldText) : lEdits[vI].vOldText;
      size_t vOccurrences =
          mCountOccurrences(mNormalizeForFuzzyMatch(vOriginalContent), mNormalizeForFuzzyMatch(lEdits[vI].vOldText));
      if (vOccurrences > 1) {
        return {{"error", fStringFormat("found %zu occurrences of edits[%zu].old_text in %s, it must be unique",
                                        vOccurrences, vI, vPath.c_str())}};
      }
      size_t vIdx = vBaseContent.find(vNeedle);
      lMatched.push_back({vI, vIdx, vNeedle.size(), lEdits[vI].vNewText});
    }

    std::sort(lMatched.begin(), lMatched.end(),
              [](const matched_edit &pA, const matched_edit &pB) { return pA.vMatchIndex < pB.vMatchIndex; });
    for (size_t vI = 1; vI < lMatched.size(); vI++) {
      if (lMatched[vI - 1].vMatchIndex + lMatched[vI - 1].vMatchLength > lMatched[vI].vMatchIndex) {
        return {{"error",
                 fStringFormat(
                     "edits[%zu] and edits[%zu] overlap in %s; merge them into one edit or target disjoint regions",
                     lMatched[vI - 1].vEditIndex, lMatched[vI].vEditIndex, vPath.c_str())}};
      }
    }

    std::string vNewContent = vAnyFuzzy
                                  ? mApplyReplacementsPreservingUnchangedLines(vOriginalContent, vBaseContent, lMatched)
                                  : mApplyReplacements(vBaseContent, lMatched, 0);

    if (vNewContent == vOriginalContent) {
      return {{"error", "no changes made: the replacement(s) produced identical content"}};
    }

    if (!vIo->mWriteFile(vPath, vNewContent)) {
      return {{"error", "failed to write file: " + vPath}};
    }

    return {{"result", "file edited successfully"}, {"path", vPath}, {"edits_applied", (int)lMatched.size()}};
  }

private:
  // strip trailing whitespace, normalize smart quotes/dashes/spaces to ASCII
  static std::string mNormalizeLineForFuzzyMatch(const std::string &pLine) {
    size_t vEnd = pLine.size();
    while (vEnd > 0 && (pLine[vEnd - 1] == ' ' || pLine[vEnd - 1] == '\t' || pLine[vEnd - 1] == '\r')) {
      vEnd--;
    }
    std::string vS = pLine.substr(0, vEnd);

    auto vReplaceAll = [](std::string &pStr, const std::string &pFrom, const std::string &pTo) {
      if (pFrom.empty())
        return;
      size_t vPos = 0;
      while ((vPos = pStr.find(pFrom, vPos)) != std::string::npos) {
        pStr.replace(vPos, pFrom.size(), pTo);
        vPos += pTo.size();
      }
    };

    // smart single quotes -> '
    for (unsigned char vB : {0x98, 0x99, 0x9A, 0x9B}) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char)vB, "'");
    }
    // smart double quotes -> "
    for (unsigned char vB : {0x9C, 0x9D, 0x9E, 0x9F}) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char)vB, "\"");
    }
    // various dashes -> -
    for (unsigned char vB = 0x90; vB <= 0x95; vB++) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char)vB, "-");
    }
    vReplaceAll(vS, "\xE2\x88\x92", "-"); // minus sign
    // special spaces -> ' '
    vReplaceAll(vS, "\xC2\xA0", " "); // no-break space
    for (unsigned char vB = 0x82; vB <= 0x8A; vB++) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char)vB, " ");
    }
    vReplaceAll(vS, "\xE2\x80\xAF", " "); // narrow no-break space
    vReplaceAll(vS, "\xE2\x81\x9F", " "); // medium mathematical space
    vReplaceAll(vS, "\xE3\x80\x80", " "); // ideographic space

    return vS;
  }

  // applies the per-line transform above to every line; preserves line count/positions
  static std::string mNormalizeForFuzzyMatch(const std::string &pContent) {
    std::string vResult;
    vResult.reserve(pContent.size());
    size_t vStart = 0;
    while (true) {
      size_t vNl = pContent.find('\n', vStart);
      bool vIsLast = vNl == std::string::npos;
      std::string vLine = vIsLast ? pContent.substr(vStart) : pContent.substr(vStart, vNl - vStart);
      vResult += mNormalizeLineForFuzzyMatch(vLine);
      if (vIsLast)
        break;
      vResult += '\n';
      vStart = vNl + 1;
    }
    return vResult;
  }

  // lines with trailing '\n' kept, so untouched ones can be reconstructed verbatim
  static std::vector<std::string> mSplitLinesWithEndings(const std::string &pContent) {
    std::vector<std::string> lLines;
    size_t vStart = 0;
    while (vStart < pContent.size()) {
      size_t vNl = pContent.find('\n', vStart);
      if (vNl == std::string::npos) {
        lLines.push_back(pContent.substr(vStart));
        break;
      }
      lLines.push_back(pContent.substr(vStart, vNl - vStart + 1));
      vStart = vNl + 1;
    }
    return lLines;
  }

  struct line_span {
    size_t vStart;
    size_t vEnd;
  };

  static std::vector<line_span> mGetLineSpans(const std::string &pContent) {
    std::vector<line_span> lSpans;
    size_t vOffset = 0;
    for (const auto &vLine : mSplitLinesWithEndings(pContent)) {
      lSpans.push_back({vOffset, vOffset + vLine.size()});
      vOffset += vLine.size();
    }
    return lSpans;
  }

  // count non-overlapping occurrences of `pNeedle` in `pContent`
  static size_t mCountOccurrences(const std::string &pContent, const std::string &pNeedle) {
    if (pNeedle.empty())
      return 0;
    size_t vCount = 0, vPos = 0;
    while ((vPos = pContent.find(pNeedle, vPos)) != std::string::npos) {
      vCount++;
      vPos += pNeedle.size();
    }
    return vCount;
  }

  struct matched_edit {
    size_t vEditIndex;
    size_t vMatchIndex; // offset into the "base content" (see below)
    size_t vMatchLength;
    std::string vNewText;
  };

  // replacements must be sorted ascending by match_index and non-overlapping
  static std::string mApplyReplacements(const std::string &pContent, const std::vector<matched_edit> &pReplacements,
                                        size_t pOffset) {
    std::string vResult = pContent;
    for (auto vIt = pReplacements.rbegin(); vIt != pReplacements.rend(); ++vIt) {
      size_t vLocalIndex = vIt->vMatchIndex - pOffset;
      vResult = vResult.substr(0, vLocalIndex) + vIt->vNewText + vResult.substr(vLocalIndex + vIt->vMatchLength);
    }
    return vResult;
  }

  // widen a replacement's byte range to the line(s) of `pLines` it touches
  static bool mGetReplacementLineRange(const std::vector<line_span> &pLines, size_t pMatchIndex, size_t pMatchLength,
                                       size_t &pOutStartLine, size_t &pOutEndLine /* exclusive */) {
    size_t vReplacementStart = pMatchIndex;
    size_t vReplacementEnd = pMatchIndex + pMatchLength;

    size_t vStartLine = (size_t)-1;
    for (size_t vI = 0; vI < pLines.size(); vI++) {
      if (vReplacementStart >= pLines[vI].vStart && vReplacementStart < pLines[vI].vEnd) {
        vStartLine = vI;
        break;
      }
    }
    if (vStartLine == (size_t)-1)
      return false;

    size_t vEndLine = vStartLine;
    while (vEndLine < pLines.size() && pLines[vEndLine].vEnd < vReplacementEnd) {
      vEndLine++;
    }
    if (vEndLine >= pLines.size())
      return false;

    pOutStartLine = vStartLine;
    pOutEndLine = vEndLine + 1;
    return true;
  }

  // like mApplyReplacements, but untouched lines come from `pOriginalContent`
  static std::string mApplyReplacementsPreservingUnchangedLines(
      const std::string &pOriginalContent, const std::string &pBaseContent,
      const std::vector<matched_edit> &pReplacements /* ascending, non-overlapping */) {
    auto vOriginalLines = mSplitLinesWithEndings(pOriginalContent);
    auto vBaseLines = mGetLineSpans(pBaseContent);

    struct group {
      size_t vStartLine;
      size_t vEndLine; // exclusive
      std::vector<matched_edit> lReps;
    };
    std::vector<group> lGroups;

    for (const auto &vRep : pReplacements) {
      size_t vStartLine = 0, vEndLine = 0;
      mGetReplacementLineRange(vBaseLines, vRep.vMatchIndex, vRep.vMatchLength, vStartLine, vEndLine);
      if (!lGroups.empty() && vStartLine < lGroups.back().vEndLine) {
        lGroups.back().vEndLine = std::max(lGroups.back().vEndLine, vEndLine);
        lGroups.back().lReps.push_back(vRep);
      } else {
        lGroups.push_back({vStartLine, vEndLine, {vRep}});
      }
    }

    size_t vOriginalLineIndex = 0;
    std::string vResult;
    for (auto &vG : lGroups) {
      for (size_t vI = vOriginalLineIndex; vI < vG.vStartLine; vI++) {
        vResult += vOriginalLines[vI];
      }

      size_t vGroupStartOffset = vBaseLines[vG.vStartLine].vStart;
      size_t vGroupEndOffset = vBaseLines[vG.vEndLine - 1].vEnd;
      std::string vSlice = pBaseContent.substr(vGroupStartOffset, vGroupEndOffset - vGroupStartOffset);
      vResult += mApplyReplacements(vSlice, vG.lReps, vGroupStartOffset);

      vOriginalLineIndex = vG.vEndLine;
    }
    for (size_t vI = vOriginalLineIndex; vI < vOriginalLines.size(); vI++) {
      vResult += vOriginalLines[vI];
    }

    return vResult;
  }
};

//
// get_datetime: returns the current date and time
//

struct server_tool_get_datetime : server_tool {
  server_tool_get_datetime() {
    vName = "get_datetime";
    vDisplayName = "Get Date & Time";
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description", "Returns the current date and time in UTC"},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties",
                   {
                       {"format",
                        {
                            {"type", "string"},
                            {"description",
                             "strftime()-style format string for the output (default: \"%Y-%m-%dT%H:%M:%SZ\", "
                             "e.g. ISO 8601). Choose your own format if you need something else, "
                             "e.g. \"%A, %B %d %Y\" for a human-readable date."},
                        }},
                   }},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vFormat = fJsonValue(pParams, "format", std::string("%Y-%m-%dT%H:%M:%SZ"));

    auto vNow = std::chrono::system_clock::now();
    auto vTime = std::chrono::system_clock::to_time_t(vNow);
    std::tm vTmUtc;
#ifdef _WIN32
    gmtime_s(&vTmUtc, &vTime);
#else
    gmtime_r(&vTime, &vTmUtc);
#endif

    char aBuf[256];
    size_t vLen = std::strftime(aBuf, sizeof(aBuf), vFormat.c_str(), &vTmUtc);
    if (vLen == 0) {
      return {{"error", "invalid format string"}};
    }

    return {{"result", std::string(aBuf, vLen)}};
  }
};

//
// get_info: returns runtime info (OS name/version and cwd)
//

struct server_tool_get_info : server_tool {
  server_tool_get_info() {
    vName = "get_info";
    vDisplayName = "Get Runtime Info";
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description", "Returns runtime info: the OS name/version and the current working directory"},
             {"parameters",
              {
                  {"type", "object"},
                  {"properties", json::object()},
              }},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    auto vIo = fMakeToolsIo(pParams);

#ifdef _WIN32
    auto vRes = vIo->mRun({"cmd", "/c", "ver"}, 4096, 5);
#else
    auto vRes = vIo->mRun({"uname", "-a"}, 4096, 5);
#endif
    // "ver" prints a blank line before the version, so the output is stripped on both ends;
    // a failed spawn or a timeout leaves a diagnostic in vRes.vOutput, which is not an OS name
    std::string vOsInfo = vRes.vExitCode == 0 && !vRes.vTimedOut ? fStringStrip(vRes.vOutput) : "unknown";

    std::string vCwd = fJsonValue(pParams, "cwd", std::string());
    if (vCwd.empty()) {
      std::error_code vEc;
      vCwd = fs::current_path(vEc).string();
    }

    return {
        {"os", vOsInfo},
        {"cwd", vCwd},
    };
  }
};

struct server_tool_stream_result : server_task_result {
  std::string vChunk;
  bool vDone = false;
  std::string vErrorMsg;

  json to_json() override {
    if (!vDone) {
      return {{"chunk", vChunk}};
    } else {
      json vResult = {{"done", true}};
      if (!vErrorMsg.empty()) {
        vResult["error"] = vErrorMsg;
      }
      return vResult;
    }
  }
};

void server_tool::stream::mPush(const std::string &pChunk) {
  if (pChunk.empty())
    return;
  auto vR = std::make_unique<server_tool_stream_result>();
  vR->vId = vId;
  vR->vChunk = pChunk;
  vQr.mSend(std::move(vR));
}

struct server_tools_res : server_http_res {
  std::thread vWorker;
  server_response *vQr = nullptr; // set only for streaming responses
  int vId = -1;

  ~server_tools_res() override {
    if (vWorker.joinable()) {
      vWorker.join();
    }
    if (vQr) {
      vQr->mRemoveWaitingTaskId(vId);
    }
  }
};

//
// server_mcp_tool: exposes one tool from a running MCP server as a server_tool.
//
struct server_mcp_tool : server_tool {
  std::string vServerName;
  std::string vToolName;
  server_mcp_tool_def vDef;
  server_mcp &vMcpMgr;

  server_mcp_tool(server_mcp_tool_def pD, server_mcp &pMgr)
      : vServerName(pD.vServerName), vToolName(pD.vName), vDef(std::move(pD)), vMcpMgr(pMgr) {
    vName = vServerName + "_" + vToolName;
    vDisplayName = vName;
    vPermissionWrite = false;
    vSupportStream = false;
  }

  std::string mType() const override { return "mcp"; }

  json mGetDefinition() const override {
    json vSchema = vDef.vInputSchema;
    if (vSchema.is_null() || !vSchema.is_object()) {
      vSchema = json::object();
    }
    return {
        {"type", "function"},
        {"function",
         {
             {"name", vName},
             {"description", vDef.vDescription},
             {"parameters", vSchema},
         }},
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    return vMcpMgr.mCallTool(vServerName, vToolName, pParams);
  }
};

static server_tool &fFindTool(std::vector<std::unique_ptr<server_tool>> &pTools, const std::string &pName,
                              bool pRequireStream) {
  for (auto &vT : pTools) {
    if (vT->vName == pName) {
      if (pRequireStream && !vT->vSupportStream) {
        throw std::invalid_argument(fStringFormat("tool \"%s\" does not support stream = true", pName.c_str()));
      }
      return *vT;
    }
  }
  throw std::invalid_argument(fStringFormat("unknown tool \"%s\"", pName.c_str()));
}

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> fBuildTools() {
  std::vector<std::unique_ptr<server_tool>> lTools;
  lTools.push_back(std::make_unique<server_tool_read_file>());
  lTools.push_back(std::make_unique<server_tool_file_glob_search>());
  lTools.push_back(std::make_unique<server_tool_grep_search>());
  lTools.push_back(std::make_unique<server_tool_exec_shell_command>());
  lTools.push_back(std::make_unique<server_tool_write_file>());
  lTools.push_back(std::make_unique<server_tool_edit_file>());
  lTools.push_back(std::make_unique<server_tool_get_datetime>());
  lTools.push_back(std::make_unique<server_tool_get_info>());
  return lTools;
}

static std::string fStrToLower(const std::string &pValue) {
  std::string vLowered(pValue.size(), '\0');
  std::transform(pValue.begin(), pValue.end(), vLowered.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return vLowered;
}

static std::string fGetHeader(const std::map<std::string, std::string> &pHeaders, const std::string &pKey,
                              std::string pDefaultValue = "") {
  const auto cLoweredKey = fStrToLower(pKey);
  for (const auto &vH : pHeaders) {
    if (fStrToLower(vH.first) == cLoweredKey) {
      return vH.second;
    }
  }
  return pDefaultValue;
}

void lServerTools::mSetup(const std::vector<std::string> &pEnabledTools, server_mcp &pMcpMgr) {
  if (!pEnabledTools.empty()) {
    if (!common_subproc::mIsSupported()) {
      throw std::runtime_error("subprocess is not enabled on this build");
    }

    std::unordered_set<std::string> sEnabledSet(pEnabledTools.begin(), pEnabledTools.end());
    auto vAllTools = fBuildTools();

    // collect all known tool names for validation
    std::vector<std::string> lKnownNames;
    lKnownNames.reserve(vAllTools.size());
    for (const auto &vT : vAllTools) {
      lKnownNames.push_back(vT->vName);
    }

    // validate that every requested tool is known
    for (const auto &vName : pEnabledTools) {
      if (vName == "all")
        continue;
      if (std::find(lKnownNames.begin(), lKnownNames.end(), vName) == lKnownNames.end()) {
        throw std::runtime_error(fStringFormat("unknown tool \"%s\". available tools: %s", vName.c_str(),
                                               fStringJoin(lKnownNames, ", ").c_str()));
      }
    }

    lTools.clear();
    for (auto &vT : vAllTools) {
      if (sEnabledSet.count(vT->vName) > 0 || sEnabledSet.count("all") > 0) {
        lTools.push_back(std::move(vT));
      }
    }
  }

  // append MCP tools, skipping any that collide with a built-in or another MCP tool of the same "<server>_<tool>" name
  if (!pMcpMgr.empty()) {
    std::unordered_set<std::string> sSeenNames;
    for (auto &vT : lTools) {
      sSeenNames.insert(vT->vName);
    }
    size_t vNAdded = 0;
    for (const auto &vDef : pMcpMgr.mListTools()) {
      std::string vMcpName = vDef.vServerName + "_" + vDef.vName;
      if (sSeenNames.count(vMcpName)) {
        SRV_WRN("MCP tool \"%s\" from server \"%s\" collides with an existing tool, skipping\n", vMcpName.c_str(),
                vDef.vServerName.c_str());
        continue;
      }
      sSeenNames.insert(vMcpName);
      lTools.push_back(std::make_unique<server_mcp_tool>(vDef, pMcpMgr));
      vNAdded++;
    }
    if (vNAdded > 0) {
      SRV_INF("Added %zu MCP tools\n", vNAdded);
    }
  }

  vHandleGet = [this](const server_http_req &) -> server_http_res_ptr {
    auto vRes = std::make_unique<server_http_res>();
    try {
      json vResult = json::array();
      for (const auto &vT : lTools) {
        vResult.push_back(vT->to_json());
      }
      vRes->vData = fSafeJsonToStr(vResult);
    } catch (const std::exception &vE) {
      SRV_ERR("got exception: %s\n", vE.what());
      vRes->vStatus = 500;
      vRes->vData = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeServer));
    }
    return vRes;
  };

  vHandlePost = [this](const server_http_req &pReq) -> server_http_res_ptr {
    auto vRes = std::make_unique<server_tools_res>();
    // a text/plain or form body is a CORS "simple request" that a cross-site page sends without a preflight;
    // requiring JSON forces the preflight, which CORS refuses to foreign origins
    if (fStrToLower(fGetHeader(pReq.dHeaders, "Content-Type")).rfind("application/json", 0) != 0) {
      vRes->vStatus = 415;
      vRes->vData =
          fSafeJsonToStr(fFormatErrorResponse("Content-Type must be application/json", cErrorTypeInvalidRequest));
      return vRes;
    }
    try {
      json vBody = json::parse(pReq.vBody);
      std::string vToolName = vBody.at("tool").get<std::string>();
      json vParams = vBody.value("params", json::object());
      bool vStream = vBody.value("stream", false);

      // accept x-tool-cwd header to override of the process
      auto vCwd = fGetHeader(pReq.dHeaders, "x-tool-cwd");
      if (!vCwd.empty()) {
        vParams["cwd"] = vCwd;
      }

      server_tool &vTool = fFindTool(lTools, vToolName, vStream);

      if (vStream) {
        int vId = vResId.fetch_add(1);
        vQueueRes.mAddWaitingTaskId(vId);
        vRes->vQr = &vQueueRes;
        vRes->vId = vId;

        vRes->vWorker = std::thread([this, vId, &pReq, &vTool, vParams]() mutable {
          server_tool::stream vSt{vQueueRes, vId, [&pReq]() { return !pReq.vShouldStop(); }};

          auto vDone = std::make_unique<server_tool_stream_result>();
          try {
            vTool.mInvoke(vParams, &vSt);
          } catch (const std::exception &vE) {
            vDone->vErrorMsg = vE.what();
          } catch (...) {
            vDone->vErrorMsg = "An unknown error occurred";
          }
          vDone->vId = vSt.vId;
          vDone->vDone = true;
          vSt.vQr.mSend(std::move(vDone));
        });

        vRes->vContentType = "text/event-stream";
        vRes->vStatus = 200;
        vRes->vNext = [this, vId](std::string &pOutput) -> bool {
          auto vResult = vQueueRes.mRecv(vId);
          auto *vR = dynamic_cast<server_tool_stream_result *>(vResult.get());
          GGML_ASSERT(vR != nullptr);
          pOutput = "data: " + fSafeJsonToStr(vR->to_json()) + "\n\n";
          if (vR->vDone) {
            vQueueRes.mRemoveWaitingTaskId(vId);
            return false;
          }
          return true;
        };
      } else {
        json vResult = vTool.mInvoke(vParams, nullptr);
        vRes->vStatus = 200;
        vRes->vData = fSafeJsonToStr(vResult);
      }
    } catch (const json::exception &vE) {
      vRes->vStatus = 400;
      vRes->vData = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    } catch (const std::invalid_argument &vE) {
      vRes->vStatus = 404;
      vRes->vData = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    } catch (const std::exception &vE) {
      SRV_ERR("got exception: %s\n", vE.what());
      vRes->vStatus = 500;
      vRes->vData = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeServer));
    }
    return vRes;
  };
}
