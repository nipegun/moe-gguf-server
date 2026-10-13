#include "subproc.h"

bool common_subproc::mIsSupported() {
#ifdef MGS_SUBPROCESS
  return true;
#else
  return false;
#endif
}

#ifdef MGS_SUBPROCESS

static std::vector<char *> fToCstrVector(const std::vector<std::string> &pValues) {
  std::vector<char *> lResult;
  lResult.reserve(pValues.size() + 1);
  for (const auto &vValue : pValues) {
    lResult.push_back(const_cast<char *>(vValue.c_str()));
  }
  lResult.push_back(nullptr);
  return lResult;
}

common_subproc::~common_subproc() {
  if (vIsCreated) {
    subprocess_destroy(&vProc);
    vIsCreated = false;
  }
}

bool common_subproc::mCreate(const std::vector<std::string> &pArgs, int pOptions, const std::vector<std::string> &pEnv,
                            const char *pCwd) {
  auto lArgv = fToCstrVector(pArgs);

  int vResult;
  if (pEnv.empty() && pCwd == nullptr) {
    vResult = subprocess_create(lArgv.data(), pOptions, &vProc);
  } else {
    auto lEnvp = fToCstrVector(pEnv);
    vResult = subprocess_create_ex(lArgv.data(), pOptions, pEnv.empty() ? nullptr : lEnvp.data(), pCwd, &vProc);
  }

  vIsCreated = vResult == 0;
  return vIsCreated;
}

bool common_subproc::mHasHandle() const {
  if (!vIsCreated) {
    return false;
  }
#if defined(_WIN32)
  return vProc.hProcess != nullptr;
#else
  return vProc.child > 0;
#endif
}

bool common_subproc::mAlive() { return vIsCreated && subprocess_alive(&vProc); }

FILE *common_subproc::mStdinFile() { return vIsCreated ? subprocess_stdin(&vProc) : nullptr; }

FILE *common_subproc::mStdoutFile() { return vIsCreated ? subprocess_stdout(&vProc) : nullptr; }

FILE *common_subproc::mStderrFile() { return vIsCreated ? subprocess_stderr(&vProc) : nullptr; }

void common_subproc::mCloseStdin() {
  if (vIsCreated && vProc.stdin_file) {
    fclose(vProc.stdin_file);
    vProc.stdin_file = nullptr;
  }
}

void common_subproc::mTerminate() {
  if (mHasHandle()) {
    subprocess_terminate(&vProc);
  }
}

int common_subproc::mJoin() {
  int vExitCode = -1;
  if (vIsCreated) {
    subprocess_join(&vProc, &vExitCode);
    subprocess_destroy(&vProc);
    vIsCreated = false;
  }
  return vExitCode;
}

#else // !MGS_SUBPROCESS

common_subproc::~common_subproc() = default;

bool common_subproc::mCreate(const std::vector<std::string> &, int, const std::vector<std::string> &, const char *) {
  (void)(vProc);
  (void)(vIsCreated);
  return false;
}

bool common_subproc::mHasHandle() const { return false; }

bool common_subproc::mAlive() { return false; }

FILE *common_subproc::mStdinFile() { return nullptr; }

FILE *common_subproc::mStdoutFile() { return nullptr; }

FILE *common_subproc::mStderrFile() { return nullptr; }

void common_subproc::mCloseStdin() {}

void common_subproc::mTerminate() {}

int common_subproc::mJoin() { return -1; }

#endif // MGS_SUBPROCESS
