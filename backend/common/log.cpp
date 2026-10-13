#include "common.h"
#include "log.h"

#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif // defined(_WIN32)

int vCommonLogVerbosityThold = LOG_DEFAULT_MGS;

int fCommonLogGetVerbosityThold(void) { return vCommonLogVerbosityThold; }

void fCommonLogSetVerbosityThold(int pVerbosity) { vCommonLogVerbosityThold = pVerbosity; }

static int64_t fTUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// colors
enum common_log_col : int {
  cCommonLogColDefault = 0,
  cCommonLogColBold,
  cCommonLogColRed,
  cCommonLogColGreen,
  cCommonLogColYellow,
  cCommonLogColBlue,
  cCommonLogColMagenta,
  cCommonLogColCyan,
  cCommonLogColWhite,
};

// disable colors by default
static const char *aGCol[] = {
    "", "", "", "", "", "", "", "", "",
};

struct common_log_entry {
  enum ggml_log_level vLevel { GGML_LOG_LEVEL_INFO };

  std::vector<char> lMsg;

  int64_t vTimestamp{0};
  bool vIsEnd{false}; // signals the worker thread to stop
  bool vPrefix{false};

  common_log_entry(size_t pSize = 256) : lMsg(pSize) {}

  void mPrint(FILE *pFile = nullptr) const {
    FILE *vFcur = pFile;
    if (!vFcur) {
      // stderr displays DBG messages only when their verbosity level is not higher than the threshold
      // these messages will still be logged to a file
      if (vLevel == GGML_LOG_LEVEL_DEBUG && vCommonLogVerbosityThold < LOG_DEFAULT_DEBUG) {
        return;
      }

      vFcur = stdout;

      if (vLevel != GGML_LOG_LEVEL_NONE) {
        vFcur = stderr;
      }
    }

    if (vLevel != GGML_LOG_LEVEL_NONE && vLevel != GGML_LOG_LEVEL_CONT && vPrefix) {
      if (vTimestamp) {
        // [M.s.ms.us]
        fprintf(vFcur, "%s%d.%02d.%03d.%03d%s ", aGCol[cCommonLogColBlue], (int)(vTimestamp / 1000000 / 60),
                (int)(vTimestamp / 1000000 % 60), (int)(vTimestamp / 1000 % 1000), (int)(vTimestamp % 1000),
                aGCol[cCommonLogColDefault]);
      }

      switch (vLevel) {
      case GGML_LOG_LEVEL_INFO:
        fprintf(vFcur, "%sI %s", aGCol[cCommonLogColGreen], aGCol[cCommonLogColDefault]);
        break;
      case GGML_LOG_LEVEL_WARN:
        fprintf(vFcur, "%sW %s", aGCol[cCommonLogColMagenta], "");
        break;
      case GGML_LOG_LEVEL_ERROR:
        fprintf(vFcur, "%sE %s", aGCol[cCommonLogColRed], "");
        break;
      case GGML_LOG_LEVEL_DEBUG:
        fprintf(vFcur, "%sD %s", aGCol[cCommonLogColYellow], "");
        break;
      default:
        break;
      }
    }

    fprintf(vFcur, "%s", lMsg.data());

    if (vLevel == GGML_LOG_LEVEL_WARN || vLevel == GGML_LOG_LEVEL_ERROR || vLevel == GGML_LOG_LEVEL_DEBUG) {
      fprintf(vFcur, "%s", aGCol[cCommonLogColDefault]);
    }

    fflush(vFcur);
  }
};

struct common_log {
  // default capacity
  common_log(size_t pCapacity = 512) {
    vFile = nullptr;
    vPrefix = false;
    vTimestamps = false;
    vRunning = false;
    vTStart = fTUs();

    lQueue.resize(pCapacity, common_log_entry(256));
    vHead = 0;
    vTail = 0;

    mResume();
  }

  ~common_log() {
    mPause();
    if (vFile) {
      fclose(vFile);
    }
  }

private:
  std::mutex vMtx;
  std::thread vThrd;
  std::condition_variable vCvNew;  // new entry
  std::condition_variable vCvFull; // wait on full

  FILE *vFile;

  bool vPrefix;
  bool vTimestamps;
  bool vRunning;

  int64_t vTStart;

  // queue of entries
  std::vector<common_log_entry> lQueue;
  size_t vHead;
  size_t vTail;

  bool mPrintEntry(const common_log_entry &pE) const {
    if (pE.vIsEnd)
      return true;

    pE.mPrint();
    if (vFile) {
      pE.mPrint(vFile);
    }
    return false;
  }

  bool mFlushQueue(size_t pStartHead, size_t pEndTail, size_t &pOutHead) const {
    bool vStop = false;
    size_t vH = pStartHead;
    while (vH != pEndTail && !vStop) {
      vStop = mPrintEntry(lQueue[vH]);
      vH = (vH + 1) % lQueue.size();
    }
    pOutHead = vH;
    return vStop;
  }

public:
  bool mIsFull() const { return ((vTail + 1) % lQueue.size()) == vHead; }

  bool mIsEmpty() const { return vHead == vTail; }

  void mAdd(enum ggml_log_level pLevel, const char *pFmt, va_list pArgs) {
    std::unique_lock<std::mutex> vLock(vMtx);

    // block if the queue is full
    vCvFull.wait(vLock, [this]() { return !vRunning || !mIsFull(); });

    if (!vRunning) {
      // discard messages while the worker thread is paused
      return;
    }

    auto &vEntry = lQueue[vTail];

    {
      // cannot use args twice, so make a copy in case we need to expand the buffer
      va_list vArgsCopy;
      va_copy(vArgsCopy, pArgs);

#if 1
      const size_t cN = vsnprintf(vEntry.lMsg.data(), vEntry.lMsg.size(), pFmt, pArgs);
      if (cN >= vEntry.lMsg.size()) {
        vEntry.lMsg.resize(cN + 1);
        vsnprintf(vEntry.lMsg.data(), vEntry.lMsg.size(), pFmt, vArgsCopy);
      }
#else
      // hack for bolding arguments

      std::stringstream vSs;
      for (int vI = 0; pFmt[vI] != 0; vI++) {
        if (pFmt[vI] == '%') {
          vSs << LOG_COL_BOLD;
          while (pFmt[vI] != ' ' && pFmt[vI] != ')' && pFmt[vI] != ']' && pFmt[vI] != 0)
            vSs << pFmt[vI++];
          vSs << LOG_COL_DEFAULT;
          if (pFmt[vI] == 0)
            break;
        }
        vSs << pFmt[vI];
      }
      const size_t cN = vsnprintf(vEntry.lMsg.data(), vEntry.lMsg.size(), vSs.str().c_str(), pArgs);
      if (cN >= vEntry.lMsg.size()) {
        vEntry.lMsg.resize(cN + 1);
        vsnprintf(vEntry.lMsg.data(), vEntry.lMsg.size(), vSs.str().c_str(), vArgsCopy);
      }
#endif
      va_end(vArgsCopy);
    }

    vEntry.vIsEnd = false;
    vEntry.vLevel = pLevel;
    vEntry.vPrefix = vPrefix;
    vEntry.vTimestamp = 0;
    if (vTimestamps) {
      vEntry.vTimestamp = fTUs() - vTStart;
    }

    vTail = (vTail + 1) % lQueue.size();
    vCvNew.notify_one();
  }

  void mResume() {
    std::lock_guard<std::mutex> vLock(vMtx);

    if (vRunning) {
      return;
    }

    vRunning = true;

    vThrd = std::thread([this]() {
      while (true) {
        std::unique_lock<std::mutex> vLock(vMtx);
        vCvNew.wait(vLock, [this]() { return !mIsEmpty(); });

        size_t vCachedHead = vHead;
        size_t vCachedTail = vTail;

        vLock.unlock(); // drop the lock during flush

        size_t vNextHead;
        bool vStop = mFlushQueue(vCachedHead, vCachedTail, vNextHead);

        vLock.lock();
        vHead = vNextHead;
        vCvFull.notify_all();

        if (vStop) {
          break;
        }
      }
    });
  }

  void mPause() {
    {
      std::lock_guard<std::mutex> vLock(vMtx);

      if (!vRunning) {
        return;
      }

      vRunning = false;

      // push an entry to signal the worker thread to stop
      auto &vEntry = lQueue[vTail];
      vEntry.vIsEnd = true;
      vTail = (vTail + 1) % lQueue.size();

      // wakeup everyone
      vCvNew.notify_one();
      vCvFull.notify_all();
    }

    vThrd.join();
  }

  void mSetFile(const char *pPath) {
    mPause();

    if (vFile) {
      fclose(vFile);
    }

    if (pPath) {
      vFile = fopen(pPath, "w");
    } else {
      vFile = nullptr;
    }

    mResume();
  }

  void mSetColors(bool pColors) {
    mPause();

    if (pColors) {
      aGCol[cCommonLogColDefault] = LOG_COL_DEFAULT;
      aGCol[cCommonLogColBold] = LOG_COL_BOLD;
      aGCol[cCommonLogColRed] = LOG_COL_RED;
      aGCol[cCommonLogColGreen] = LOG_COL_GREEN;
      aGCol[cCommonLogColYellow] = LOG_COL_YELLOW;
      aGCol[cCommonLogColBlue] = LOG_COL_BLUE;
      aGCol[cCommonLogColMagenta] = LOG_COL_MAGENTA;
      aGCol[cCommonLogColCyan] = LOG_COL_CYAN;
      aGCol[cCommonLogColWhite] = LOG_COL_WHITE;
    } else {
      for (size_t vI = 0; vI < std::size(aGCol); vI++) {
        aGCol[vI] = "";
      }
    }

    mResume();
  }

  void mSetPrefix(bool pPrefix) {
    std::lock_guard<std::mutex> vLock(vMtx);

    this->vPrefix = pPrefix;
  }

  void mSetTimestamps(bool pTimestamps) {
    std::lock_guard<std::mutex> vLock(vMtx);

    this->vTimestamps = pTimestamps;
  }
};

//
// public API
//

struct common_log *fCommonLogInit() { return new common_log; }

struct common_log *fCommonLogMain() {
  // We intentionally leak (i.e. do not delete) the logger singleton because
  // common_log destructor called at DLL teardown phase will cause hanging on Windows.
  // OS will release resources anyway so it should not be a significant issue,
  // though this design may cause logs to be lost if not flushed before the program exits.
  static struct common_log *vLog;
  static std::once_flag vInitFlag;
  std::call_once(vInitFlag, [&]() {
    vLog = new common_log;
    // Set default to auto-detect colors
    vLog->mSetColors(fTtyCanUseColors());
  });

  return vLog;
}

void fCommonLogPause(struct common_log *pLog) { pLog->mPause(); }

void fCommonLogResume(struct common_log *pLog) { pLog->mResume(); }

void fCommonLogFree(struct common_log *pLog) { delete pLog; }

void fCommonLogAdd(struct common_log *pLog, enum ggml_log_level pLevel, const char *pFmt, ...) {
  va_list vArgs;
  va_start(vArgs, pFmt);
  pLog->mAdd(pLevel, pFmt, vArgs);
  va_end(vArgs);
}

void fCommonLogSetFile(struct common_log *pLog, const char *pFile) { pLog->mSetFile(pFile); }

void fCommonLogSetColors(struct common_log *pLog, log_colors pColors) {
  if (pColors == cLogColorsAuto) {
    pLog->mSetColors(fTtyCanUseColors());
    return;
  }

  if (pColors == cLogColorsDisabled) {
    pLog->mSetColors(false);
    return;
  }

  GGML_ASSERT(pColors == cLogColorsEnabled);
  pLog->mSetColors(true);
}

void fCommonLogSetPrefix(struct common_log *pLog, bool pPrefix) { pLog->mSetPrefix(pPrefix); }

void fCommonLogSetTimestamps(struct common_log *pLog, bool pTimestamps) { pLog->mSetTimestamps(pTimestamps); }

void fCommonLogFlush(struct common_log *pLog) {
  pLog->mPause();
  pLog->mResume();
}

static int fCommonGetVerbosity(enum ggml_log_level pLevel) {
  switch (pLevel) {
  case GGML_LOG_LEVEL_DEBUG:
    return LOG_LEVEL_DEBUG;
  case GGML_LOG_LEVEL_INFO:
    return LOG_LEVEL_TRACE;
  case GGML_LOG_LEVEL_WARN:
    return LOG_LEVEL_WARN;
  case GGML_LOG_LEVEL_ERROR:
    return LOG_LEVEL_ERROR;
  case GGML_LOG_LEVEL_CONT:
    return LOG_LEVEL_TRACE;
  case GGML_LOG_LEVEL_NONE:
  default:
    return LOG_LEVEL_OUTPUT;
  }
}

void fCommonLogDefaultCallback(enum ggml_log_level pLevel, const char *pText, void * /*user_data*/) {
  auto vVerbosity = fCommonGetVerbosity(pLevel);
  if (vVerbosity <= vCommonLogVerbosityThold) {
    fCommonLogAdd(fCommonLogMain(), pLevel, "%s", pText);
  }
}
