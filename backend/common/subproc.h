#pragma once

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#ifdef MGS_SUBPROCESS
#include <sheredom/subprocess.h>
#else
// dummy values to allow compilation when subprocess is disabled
struct subprocess_s {};
static constexpr int subprocess_option_no_window = 0;
static constexpr int subprocess_option_combined_stdout_stderr = 0;
static constexpr int subprocess_option_inherit_environment = 0;
static constexpr int subprocess_option_search_user_path = 0;
#endif

// RAII-style wrapper around https://github.com/sheredom/subprocess.h,
// exposing method calls instead of free functions operating on subprocess_s.
struct common_subproc {
  common_subproc() = default;
  ~common_subproc();

  common_subproc(const common_subproc &) = delete;
  common_subproc &operator=(const common_subproc &) = delete;

  // spawn a child process; if env is non-empty it replaces the child's environment
  // (do not combine with subprocess_option_inherit_environment)
  bool mCreate(const std::vector<std::string> &pArgs, int pOptions, const std::vector<std::string> &pEnv = {},
               const char *pCwd = nullptr);

  bool mAlive();

  // true if MGS_SUBPROCESS was enabled at build time; when false, mCreate() always fails
  static bool mIsSupported();

  FILE *mStdinFile();
  FILE *mStdoutFile();
  FILE *mStderrFile();

  // close stdin and detach it from the process, so a later mJoin()/destructor won't double-close it;
  // use this after writing all input to signal EOF to the child while it's still running
  void mCloseStdin();

  void mTerminate();

  // wait for the process to exit, release the underlying handle and return its exit code
  int mJoin();

private:
  subprocess_s vProc{};
  std::atomic<bool> vIsCreated{false};

  bool mHasHandle() const;
};
