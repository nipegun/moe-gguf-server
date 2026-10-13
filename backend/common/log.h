#pragma once

#include "ggml.h" // for ggml_log_level

#define LOG_CLR_TO_EOL "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD "\033[1m"
#define LOG_COL_RED "\033[31m"
#define LOG_COL_GREEN "\033[32m"
#define LOG_COL_YELLOW "\033[33m"
#define LOG_COL_BLUE "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN "\033[36m"
#define LOG_COL_WHITE "\033[37m"

#ifndef __GNUC__
#define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define LOG_LEVEL_DEBUG 5
#define LOG_LEVEL_TRACE 4
#define LOG_LEVEL_INFO 3
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_OUTPUT 0 // output data from tools

#define LOG_DEFAULT_DEBUG LOG_LEVEL_DEBUG
#define LOG_DEFAULT_MGS LOG_LEVEL_INFO

enum log_colors {
  cLogColorsAuto = -1,
  cLogColorsDisabled = 0,
  cLogColorsEnabled = 1,
};

// needed by the LOG_TMPL macro to avoid computing log arguments if the verbosity lower
// set via common_log_set_verbosity()
int fCommonLogGetVerbosityThold(void);

void fCommonLogSetVerbosityThold(int pVerbosity); // not thread-safe

void fCommonLogDefaultCallback(enum ggml_log_level pLevel, const char *pText, void *pUserData);

// the common_log uses an internal worker thread to print/write log messages
// when the worker thread is paused, incoming log messages are discarded
struct common_log;

struct common_log *fCommonLogInit();

// Singleton, intentionally leaked to avoid Windows teardown hangs.
// Call fCommonLogFlush() before exit if you want to ensure all logs are flushed.
struct common_log *fCommonLogMain();

void fCommonLogPause(struct common_log *pLog);  // pause  the worker thread, not thread-safe
void fCommonLogResume(struct common_log *pLog); // resume the worker thread, not thread-safe
void fCommonLogFree(struct common_log *pLog);

LOG_ATTRIBUTE_FORMAT(3, 4)
void fCommonLogAdd(struct common_log *pLog, enum ggml_log_level pLevel, const char *pFmt, ...);

// defaults: file = NULL, colors = false, prefix = false, timestamps = false
//
// regular log output:
//
//   ggml_backend_metal_log_allocated_size: allocated buffer, size =  6695.84 MiB, ( 6695.91 / 21845.34)
//   llm_load_tensors: ggml ctx size =    0.27 MiB
//   llm_load_tensors: offloading 32 repeating layers to GPU
//   llm_load_tensors: offloading non-repeating layers to GPU
//
// with prefix = true, timestamps = true, the log output will look like this:
//
//   0.00.035.060 D ggml_backend_metal_log_allocated_size: allocated buffer, size =  6695.84 MiB, ( 6695.91 / 21845.34)
//   0.00.035.064 I llm_load_tensors: ggml ctx size =    0.27 MiB
//   0.00.090.578 I llm_load_tensors: offloading 32 repeating layers to GPU
//   0.00.090.579 I llm_load_tensors: offloading non-repeating layers to GPU
//
// D - debug   (stderr, V = LOG_DEFAULT_DEBUG)
// I - info    (stdout, V = LOG_DEFAULT_INFO)
// W - warning (stderr, V = LOG_DEFAULT_WARN)
// E - error   (stderr, V = LOG_DEFAULT_ERROR)
// O - output  (stdout, V = LOG_DEFAULT_OUTPUT)
//

void fCommonLogSetFile(struct common_log *pLog, const char *pFile);      // not thread-safe
void fCommonLogSetColors(struct common_log *pLog, log_colors pColors);   // not thread-safe
void fCommonLogSetPrefix(struct common_log *pLog, bool pPrefix);         // whether to output prefix to each log
void fCommonLogSetTimestamps(struct common_log *pLog, bool pTimestamps); // whether to output timestamps in the prefix
void fCommonLogFlush(struct common_log *pLog);                           // flush all pending log messages

// helper macros for logging
// use these to avoid computing log arguments if the verbosity of the log is higher than the threshold
//
// for example:
//
//   LOG_DBG("this is a debug message: %d\n", expensive_function());
//
// this will avoid calling expensive_function() if LOG_DEFAULT_DEBUG > vCommonLogVerbosityThold
//

#define LOG_TMPL(level, verbosity, ...)                                                                                \
  do {                                                                                                                 \
    if ((verbosity) <= fCommonLogGetVerbosityThold()) {                                                                \
      fCommonLogAdd(fCommonLogMain(), (level), __VA_ARGS__);                                                           \
    }                                                                                                                  \
  } while (0)

#define LOG(...) LOG_TMPL(GGML_LOG_LEVEL_NONE, LOG_LEVEL_OUTPUT, __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity, __VA_ARGS__)

#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_TRC(...) LOG_TMPL(GGML_LOG_LEVEL_INFO, LOG_LEVEL_TRACE, __VA_ARGS__)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO, LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN, LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT, LOG_LEVEL_INFO, __VA_ARGS__) // same as INFO

#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_TRCV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_TRACE, verbosity, __VA_ARGS__)
#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO, verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN, verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT, verbosity, __VA_ARGS__)
