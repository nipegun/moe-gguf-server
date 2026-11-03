#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES        // For M_PI on MSVC

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-threading.h"
#include "ggml-cpu.h"
#include "ggml.h"

// FIXME: required here for quantization functions
#include "ggml-quants.h"

#ifdef GGML_USE_CPU_HBM
#include <hbwmalloc.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define UNUSED GGML_UNUSED

uint64_t ggml_graph_next_uid(void) {
#ifdef _MSC_VER
#if defined(_WIN32)
  static volatile LONG vCounter = 1;
  return (uint64_t)InterlockedIncrement(&vCounter) - 1;
#else
  static volatile long long vCounter = 1;
  return (uint64_t)_InterlockedIncrement64(&vCounter) - 1;
#endif
#else
  static uint64_t vCounter = 1;
  return __atomic_fetch_add(&vCounter, 1, __ATOMIC_RELAXED);
#endif
}

// Needed for ggml_fp32_to_bf16_row()
#if defined(__AVX512BF16__)
#if defined(_MSC_VER)
#define m512i(p) p
#else
#include <immintrin.h>
#define m512i(p) (__m512i)(p)
#endif // defined(_MSC_VER)
#endif // defined(__AVX512BF16__)

#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) ||                       \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(__ANDROID__)
#include <unwind.h>
#include <dlfcn.h>
#include <stdio.h>

struct backtrace_state {
  void **current;
  void **end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context *context, void *arg) {
  struct backtrace_state *state = (struct backtrace_state *)arg;
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    if (state->current == state->end) {
      return _URC_END_OF_STACK;
    } else {
      *state->current++ = (void *)pc;
    }
  }
  return _URC_NO_REASON;
}

static void fGgmlPrintBacktraceSymbols(void) {
  const int max = 100;
  void *buffer[max];

  struct backtrace_state state = {buffer, buffer + max};
  _Unwind_Backtrace(unwind_callback, &state);

  int count = state.current - buffer;

  for (int idx = 0; idx < count; ++idx) {
    const void *addr = buffer[idx];
    const char *symbol = "";

    Dl_info info;
    if (dladdr(addr, &info) && info.dli_sname) {
      symbol = info.dli_sname;
    }

    fprintf(stderr, "%d: %p %s\n", idx, addr, symbol);
  }
}
#elif defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
static void fGgmlPrintBacktraceSymbols(void) {
  void *aTrace[100];
  int vNptrs = backtrace(aTrace, sizeof(aTrace) / sizeof(aTrace[0]));
  backtrace_symbols_fd(aTrace, vNptrs, STDERR_FILENO);
}
#elif defined(__APPLE__)
#include <execinfo.h>
static void fGgmlPrintBacktraceSymbols(void) {
  void *trace[100];
  int nptrs = backtrace(trace, sizeof(trace) / sizeof(trace[0]));
  backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#else
static void fGgmlPrintBacktraceSymbols(void) {
  // platform not supported
}
#endif

void ggml_print_backtrace(void) {
  const char *vGgmlNoBacktrace = getenv("GGML_NO_BACKTRACE");
  if (vGgmlNoBacktrace) {
    return;
  }
#if defined(__APPLE__)
  // On macOS, fork+debugger attachment is problematic due to:
  // 1. libdispatch "poisons" forked child processes
  // 2. lldb has issues attaching to parent from forked child
  // Use simple backtrace() instead to avoid Terminal.app crashes
  const char *GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
  if (!GGML_BACKTRACE_LLDB) {
    fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
    fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
    fGgmlPrintBacktraceSymbols();
    return;
  }
#endif
#if defined(__linux__)
  FILE *vF = fopen("/proc/self/status", "r");
  size_t vSize = 0;
  char *vLine = NULL;
  ssize_t vLength = 0;
  while ((vLength = getline(&vLine, &vSize, vF)) > 0) {
    if (!strncmp(vLine, "TracerPid:", sizeof("TracerPid:") - 1) &&
        (vLength != sizeof("TracerPid:\t0\n") - 1 || vLine[vLength - 2] != '0')) {
      // Already being debugged, and the breakpoint is the later abort()
      free(vLine);
      fclose(vF);
      return;
    }
  }
  free(vLine);
  fclose(vF);
  int aLock[2] = {-1, -1};
  (void)!pipe(aLock); // Don't start gdb until after PR_SET_PTRACER
#endif
  const int cParentPid = getpid();
  const int cChildPid = fork();
  if (cChildPid < 0) { // error
#if defined(__linux__)
    close(aLock[1]);
    close(aLock[0]);
#endif
    return;
  } else if (cChildPid == 0) { // child
    char aAttach[32];
    snprintf(aAttach, sizeof(aAttach), "attach %d", cParentPid);
#if defined(__linux__)
    close(aLock[1]);
    (void)!read(aLock[0], aLock, 1);
    close(aLock[0]);
#endif
    // try gdb
    execlp("gdb", "gdb", "--batch", "-ex", "set style enabled on", "-ex", aAttach, "-ex",
           "bt -frame-info source-and-location", "-ex", "detach", "-ex", "quit", (char *)NULL);
    // try lldb
    execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", &aAttach[sizeof("attach ") - 1], (char *)NULL);
    // gdb failed, fallback to backtrace_symbols
    fGgmlPrintBacktraceSymbols();
    _Exit(0);
  } else { // parent
#if defined(__linux__)
    prctl(PR_SET_PTRACER, cChildPid);
    close(aLock[1]);
    close(aLock[0]);
#endif
    waitpid(cChildPid, NULL, 0);
  }
}
#else
void ggml_print_backtrace(void) {
  // platform not supported
}
#endif

static ggml_abort_callback_t g_abort_callback = NULL;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t pCallback) {
  ggml_abort_callback_t vRetVal = g_abort_callback;
  g_abort_callback = pCallback;
  return vRetVal;
}

void ggml_abort(const char *pFile, int pLine, const char *pFmt, ...) {
  fflush(stdout);

  char aMessage[2048];
  int vOffset = snprintf(aMessage, sizeof(aMessage), "%s:%d: ", pFile, pLine);

  va_list vArgs;
  va_start(vArgs, pFmt);
  vsnprintf(aMessage + vOffset, sizeof(aMessage) - vOffset, pFmt, vArgs);
  va_end(vArgs);

  if (g_abort_callback) {
    g_abort_callback(aMessage);
  } else {
    // default: print error and backtrace to stderr
    fprintf(stderr, "%s\n", aMessage);
    ggml_print_backtrace();
  }

  abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

struct ggml_logger_state {
  ggml_log_callback vLogCallback;
  void *vLogCallbackUserData;
};
static struct ggml_logger_state g_logger_state = {ggml_log_callback_default, NULL};

static void fGgmlLogInternalV(enum ggml_log_level pLevel, const char *pFormat, va_list pArgs) {
  if (pFormat == NULL) {
    return;
  }
  va_list vArgsCopy;
  va_copy(vArgsCopy, pArgs);
  char aBuffer[128];
  int vLen = vsnprintf(aBuffer, 128, pFormat, pArgs);
  if (vLen < 128) {
    g_logger_state.vLogCallback(pLevel, aBuffer, g_logger_state.vLogCallbackUserData);
  } else {
    char *vBuffer2 = (char *)calloc(vLen + 1, sizeof(char));
    vsnprintf(vBuffer2, vLen + 1, pFormat, vArgsCopy);
    vBuffer2[vLen] = 0;
    g_logger_state.vLogCallback(pLevel, vBuffer2, g_logger_state.vLogCallbackUserData);
    free(vBuffer2);
  }
  va_end(vArgsCopy);
}

void ggml_log_internal(enum ggml_log_level pLevel, const char *pFormat, ...) {
  va_list vArgs;
  va_start(vArgs, pFormat);
  fGgmlLogInternalV(pLevel, pFormat, vArgs);
  va_end(vArgs);
}

void ggml_log_callback_default(enum ggml_log_level pLevel, const char *pText, void *pUserData) {
  (void)pLevel;
  (void)pUserData;
  fputs(pText, stderr);
  fflush(stderr);
}

//
// end of logging block
//

#ifdef GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
// #define GGML_SOFT_MAX_ACCELERATE
#endif

void *ggml_aligned_malloc(size_t size) {
#if defined(__s390x__)
  const int cAlignment = 256;
#else
  const int cAlignment = 64;
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
  return _aligned_malloc(size, cAlignment);
#else
  if (size == 0) {
    GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
    return NULL;
  }
  void *vAlignedMemory = NULL;
#ifdef GGML_USE_CPU_HBM
  int vResult = hbw_posix_memalign(&vAlignedMemory, cAlignment, size);
#elif TARGET_OS_OSX
  GGML_UNUSED(cAlignment);
  kern_return_t alloc_status =
      vm_allocate((vm_map_t)mach_task_self(), (vm_address_t *)&vAlignedMemory, size, VM_FLAGS_ANYWHERE);
  int vResult = EFAULT;
  switch (alloc_status) {
  case KERN_SUCCESS:
    vResult = 0;
    break;
  case KERN_INVALID_ADDRESS:
    vResult = EINVAL;
    break;
  case KERN_NO_SPACE:
    vResult = ENOMEM;
    break;
  default:
    vResult = EFAULT;
    break;
  }
#else
  int vResult = posix_memalign(&vAlignedMemory, cAlignment, size);
#endif
  if (vResult != 0) {
    // Handle allocation failure
    const char *vErrorDesc = "unknown allocation error";
    switch (vResult) {
    case EINVAL:
      vErrorDesc = "invalid alignment value";
      break;
    case ENOMEM:
      vErrorDesc = "insufficient memory";
      break;
    }
    GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, vErrorDesc, size / (1024.0 * 1024.0));
    return NULL;
  }
  return vAlignedMemory;
#endif
}

void ggml_aligned_free(void *pPtr, size_t size) {
  GGML_UNUSED(size);
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(pPtr);
#elif GGML_USE_CPU_HBM
  if (pPtr != NULL) {
    hbw_free(pPtr);
  }
#elif TARGET_OS_OSX
  if (pPtr != NULL) {
    vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)pPtr, size);
  }
#else
  free(pPtr);
#endif
}

inline static void *ggml_malloc(size_t pSize) {
  if (pSize == 0) {
    GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
    return NULL;
  }
  void *vResult = malloc(pSize);
  if (vResult == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, pSize / (1024.0 * 1024.0));
    GGML_ABORT("fatal error");
  }
  return vResult;
}

// calloc
inline static void *ggml_calloc(size_t pNum, size_t pSize) {
  if (pNum == 0 || pSize == 0) {
    GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
    return NULL;
  }
  void *vResult = calloc(pNum, pSize);
  if (vResult == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, pSize / (1024.0 * 1024.0));
    GGML_ABORT("fatal error");
  }
  return vResult;
}

#define GGML_MALLOC(size) ggml_malloc(size)
#define GGML_CALLOC(num, size) ggml_calloc(num, size)

#define GGML_FREE(ptr) free(ptr)

const char *ggml_status_to_string(enum ggml_status pStatus) {
  switch (pStatus) {
  case GGML_STATUS_ALLOC_FAILED:
    return "GGML status: error (failed to allocate memory)";
  case GGML_STATUS_FAILED:
    return "GGML status: error (operation failed)";
  case GGML_STATUS_SUCCESS:
    return "GGML status: success";
  case GGML_STATUS_ABORTED:
    return "GGML status: warning (operation aborted)";
  }

  return "GGML status: unknown";
}

float ggml_fp16_to_fp32(ggml_fp16_t pX) {
#define ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
  return GGML_FP16_TO_FP32(pX);
}

ggml_fp16_t ggml_fp32_to_fp16(float pX) {
#define ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
  return GGML_FP32_TO_FP16(pX);
}

float ggml_bf16_to_fp32(ggml_bf16_t pX) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
  return GGML_BF16_TO_FP32(pX); // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(float pX) {
#define ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
  return GGML_FP32_TO_BF16(pX);
}

void ggml_fp16_to_fp32_row(const ggml_fp16_t *pX, float *pY, int64_t pN) {
  for (int64_t vI = 0; vI < pN; vI++) {
    pY[vI] = GGML_FP16_TO_FP32(pX[vI]);
  }
}

void ggml_fp32_to_fp16_row(const float *pX, ggml_fp16_t *pY, int64_t pN) {
  int vI = 0;
  for (; vI < pN; ++vI) {
    pY[vI] = GGML_FP32_TO_FP16(pX[vI]);
  }
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t *pX, float *pY, int64_t pN) {
  int vI = 0;
  for (; vI < pN; ++vI) {
    pY[vI] = GGML_BF16_TO_FP32(pX[vI]);
  }
}

void ggml_fp32_to_bf16_row_ref(const float *pX, ggml_bf16_t *pY, int64_t pN) {
  for (int vI = 0; vI < pN; vI++) {
    pY[vI] = ggml_compute_fp32_to_bf16(pX[vI]);
  }
}

void ggml_fp32_to_bf16_row(const float *pX, ggml_bf16_t *pY, int64_t pN) {
  int vI = 0;
#if defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  for (; vI + 32 <= pN; vI += 32) {
    _mm512_storeu_si512((__m512i *)(pY + vI),
                        m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(pX + vI + 16), _mm512_loadu_ps(pX + vI))));
  }
#endif
  for (; vI < pN; vI++) {
    pY[vI] = GGML_FP32_TO_BF16(pX[vI]);
  }
}

bool ggml_guid_matches(ggml_guid_t pGuidA, ggml_guid_t pGuidB) {
  return memcmp(pGuidA, pGuidB, sizeof(ggml_guid)) == 0;
}

const char *ggml_version(void) { return GGML_VERSION; }

const char *ggml_commit(void) { return GGML_COMMIT; }

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
static BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
  UNUSED(once);
  UNUSED(param);
  UNUSED(ctx);

  LARGE_INTEGER t;
  QueryPerformanceFrequency(&t);
  timer_freq = t.QuadPart;

  // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
  // and the uptime is high enough.
  // We subtract the program start time to reduce the likelihood of that happening.
  QueryPerformanceCounter(&t);
  timer_start = t.QuadPart;

  return TRUE;
}
void ggml_time_init(void) {
  static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
  InitOnceExecuteOnce(&once, ggml_time_init_once, NULL, NULL);
}
int64_t ggml_time_ms(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
  struct timespec vTs;
  clock_gettime(CLOCK_MONOTONIC, &vTs);
  return (int64_t)vTs.tv_sec * 1000 + (int64_t)vTs.tv_nsec / 1000000;
}

int64_t ggml_time_us(void) {
  struct timespec vTs;
  clock_gettime(CLOCK_MONOTONIC, &vTs);
  return (int64_t)vTs.tv_sec * 1000000 + (int64_t)vTs.tv_nsec / 1000;
}
#endif

int64_t ggml_cycles(void) { return clock(); }

int64_t ggml_cycles_per_ms(void) { return CLOCKS_PER_SEC / 1000; }

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
static wchar_t *ggml_mbstowcs(const char *mbs) {
  int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
  if (!wlen) {
    errno = EINVAL;
    return NULL;
  }

  wchar_t *wbuf = GGML_MALLOC(wlen * sizeof(wchar_t));
  wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
  if (!wlen) {
    GGML_FREE(wbuf);
    errno = EINVAL;
    return NULL;
  }

  return wbuf;
}
#endif

FILE *ggml_fopen(const char *pFname, const char *pMode) {
#ifdef _WIN32
  FILE *file = NULL;

  // convert fname (UTF-8)
  wchar_t *wfname = ggml_mbstowcs(pFname);
  if (wfname) {
    // convert mode (UTF-8)
    wchar_t *wmode = ggml_mbstowcs(pMode);
    if (wmode) {
      // open file
      file = _wfopen(wfname, wmode);
      GGML_FREE(wmode);
    }

    GGML_FREE(wfname);
  }

  return file;
#else
  return fopen(pFname, pMode);
#endif
}

static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] =
    {
        [GGML_TYPE_I8] =
            {
                .type_name = "i8",
                .blck_size = 1,
                .type_size = sizeof(int8_t),
                .is_quantized = false,
            },
        [GGML_TYPE_I16] =
            {
                .type_name = "i16",
                .blck_size = 1,
                .type_size = sizeof(int16_t),
                .is_quantized = false,
            },
        [GGML_TYPE_I32] =
            {
                .type_name = "i32",
                .blck_size = 1,
                .type_size = sizeof(int32_t),
                .is_quantized = false,
            },
        [GGML_TYPE_I64] =
            {
                .type_name = "i64",
                .blck_size = 1,
                .type_size = sizeof(int64_t),
                .is_quantized = false,
            },
        [GGML_TYPE_F64] =
            {
                .type_name = "f64",
                .blck_size = 1,
                .type_size = sizeof(double),
                .is_quantized = false,
            },
        [GGML_TYPE_F32] =
            {
                .type_name = "f32",
                .blck_size = 1,
                .type_size = sizeof(float),
                .is_quantized = false,
            },
        [GGML_TYPE_F16] =
            {
                .type_name = "f16",
                .blck_size = 1,
                .type_size = sizeof(ggml_fp16_t),
                .is_quantized = false,
                .to_float = (ggml_to_float_t)ggml_fp16_to_fp32_row,
                .from_float_ref = (ggml_from_float_t)ggml_fp32_to_fp16_row,
            },
        [GGML_TYPE_Q1_0] =
            {
                .type_name = "q1_0",
                .blck_size = QK1_0,
                .type_size = sizeof(block_q1_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q1_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_q1_0_ref,
            },
        [GGML_TYPE_Q2_0] =
            {
                .type_name = "q2_0",
                .blck_size = QK2_0,
                .type_size = sizeof(block_q2_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q2_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_q2_0_ref,
            },
        [GGML_TYPE_Q4_0] =
            {
                .type_name = "q4_0",
                .blck_size = QK4_0,
                .type_size = sizeof(block_q4_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q4_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_q4_0_ref,
            },
        [GGML_TYPE_Q4_1] =
            {
                .type_name = "q4_1",
                .blck_size = QK4_1,
                .type_size = sizeof(block_q4_1),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q4_1,
                .from_float_ref = (ggml_from_float_t)quantize_row_q4_1_ref,
            },
        [4] =
            {
                // GGML_TYPE_Q4_2
                .type_name = "DEPRECATED",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [5] =
            {
                // GGML_TYPE_Q4_3
                .type_name = "DEPRECATED",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [GGML_TYPE_Q5_0] =
            {
                .type_name = "q5_0",
                .blck_size = QK5_0,
                .type_size = sizeof(block_q5_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q5_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_q5_0_ref,
            },
        [GGML_TYPE_Q5_1] =
            {
                .type_name = "q5_1",
                .blck_size = QK5_1,
                .type_size = sizeof(block_q5_1),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q5_1,
                .from_float_ref = (ggml_from_float_t)quantize_row_q5_1_ref,
            },
        [GGML_TYPE_Q8_0] =
            {
                .type_name = "q8_0",
                .blck_size = QK8_0,
                .type_size = sizeof(block_q8_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q8_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_q8_0_ref,
            },
        [GGML_TYPE_Q8_1] =
            {
                .type_name = "q8_1",
                .blck_size = QK8_1,
                .type_size = sizeof(block_q8_1),
                .is_quantized = true,
                .from_float_ref = (ggml_from_float_t)quantize_row_q8_1_ref,
            },
        [GGML_TYPE_MXFP4] =
            {
                .type_name = "mxfp4",
                .blck_size = QK_MXFP4,
                .type_size = sizeof(block_mxfp4),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_mxfp4,
                .from_float_ref = (ggml_from_float_t)quantize_row_mxfp4_ref,
            },
        [GGML_TYPE_NVFP4] =
            {
                .type_name = "nvfp4",
                .blck_size = QK_NVFP4,
                .type_size = sizeof(block_nvfp4),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_nvfp4,
                .from_float_ref = (ggml_from_float_t)quantize_row_nvfp4_ref,
            },
        [GGML_TYPE_Q2_K] =
            {
                .type_name = "q2_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q2_K),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q2_K,
                .from_float_ref = (ggml_from_float_t)quantize_row_q2_K_ref,
            },
        [GGML_TYPE_Q3_K] =
            {
                .type_name = "q3_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q3_K),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q3_K,
                .from_float_ref = (ggml_from_float_t)quantize_row_q3_K_ref,
            },
        [GGML_TYPE_Q4_K] =
            {
                .type_name = "q4_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q4_K),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q4_K,
                .from_float_ref = (ggml_from_float_t)quantize_row_q4_K_ref,
            },
        [GGML_TYPE_Q5_K] =
            {
                .type_name = "q5_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q5_K),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q5_K,
                .from_float_ref = (ggml_from_float_t)quantize_row_q5_K_ref,
            },
        [GGML_TYPE_Q6_K] =
            {
                .type_name = "q6_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q6_K),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_q6_K,
                .from_float_ref = (ggml_from_float_t)quantize_row_q6_K_ref,
            },
        [GGML_TYPE_IQ2_XXS] =
            {
                .type_name = "iq2_xxs",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq2_xxs),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq2_xxs,
                .from_float_ref = NULL,
            },
        [GGML_TYPE_IQ2_XS] =
            {
                .type_name = "iq2_xs",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq2_xs),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq2_xs,
                .from_float_ref = NULL,
            },
        [GGML_TYPE_IQ3_XXS] =
            {
                .type_name = "iq3_xxs",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq3_xxs),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq3_xxs,
                .from_float_ref = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
            },
        [GGML_TYPE_IQ3_S] =
            {
                .type_name = "iq3_s",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq3_s),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq3_s,
                .from_float_ref = (ggml_from_float_t)quantize_row_iq3_s_ref,
            },
        [GGML_TYPE_IQ2_S] =
            {
                .type_name = "iq2_s",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq2_s),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq2_s,
                .from_float_ref = (ggml_from_float_t)quantize_row_iq2_s_ref,
            },
        [GGML_TYPE_IQ1_S] =
            {
                .type_name = "iq1_s",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq1_s),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq1_s,
                .from_float_ref = NULL,
            },
        [GGML_TYPE_IQ1_M] =
            {
                .type_name = "iq1_m",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq1_m),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq1_m,
                .from_float_ref = NULL,
            },
        [GGML_TYPE_IQ4_NL] =
            {
                .type_name = "iq4_nl",
                .blck_size = QK4_NL,
                .type_size = sizeof(block_iq4_nl),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq4_nl,
                .from_float_ref = (ggml_from_float_t)quantize_row_iq4_nl_ref,
            },
        [GGML_TYPE_IQ4_XS] =
            {
                .type_name = "iq4_xs",
                .blck_size = QK_K,
                .type_size = sizeof(block_iq4_xs),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_iq4_xs,
                .from_float_ref = (ggml_from_float_t)quantize_row_iq4_xs_ref,
            },
        [GGML_TYPE_Q8_K] =
            {
                .type_name = "q8_K",
                .blck_size = QK_K,
                .type_size = sizeof(block_q8_K),
                .is_quantized = true,
            },
        [GGML_TYPE_BF16] =
            {
                .type_name = "bf16",
                .blck_size = 1,
                .type_size = sizeof(ggml_bf16_t),
                .is_quantized = false,
                .to_float = (ggml_to_float_t)ggml_bf16_to_fp32_row,
                .from_float_ref = (ggml_from_float_t)ggml_fp32_to_bf16_row_ref,
            },
        [31] =
            {
                // GGML_TYPE_Q4_0_4_4
                .type_name = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [32] =
            {
                // GGML_TYPE_Q4_0_4_8
                .type_name = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [33] =
            {
                // GGML_TYPE_Q4_0_8_8
                .type_name = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [GGML_TYPE_TQ1_0] =
            {
                .type_name = "tq1_0",
                .blck_size = QK_K,
                .type_size = sizeof(block_tq1_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_tq1_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_tq1_0_ref,
            },
        [GGML_TYPE_TQ2_0] =
            {
                .type_name = "tq2_0",
                .blck_size = QK_K,
                .type_size = sizeof(block_tq2_0),
                .is_quantized = true,
                .to_float = (ggml_to_float_t)dequantize_row_tq2_0,
                .from_float_ref = (ggml_from_float_t)quantize_row_tq2_0_ref,
            },
        [36] =
            {
                // GGML_TYPE_IQ4_NL_4_4
                .type_name = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [37] =
            {
                // GGML_TYPE_IQ4_NL_4_8
                .type_name = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
        [38] =
            {
                // GGML_TYPE_IQ4_NL_8_8
                .type_name = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
                .blck_size = 0,
                .type_size = 0,
                .is_quantized = false,
            },
};

const struct ggml_type_traits *ggml_get_type_traits(enum ggml_type pType) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  return &type_traits[pType];
}

//
// ggml object
//

struct ggml_object {
  size_t vOffs;
  size_t size;

  struct ggml_object *vNext;

  enum ggml_object_type vType;

  char aPadding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

//
// ggml context
//

struct ggml_context {
  size_t vMemSize;
  void *vMemBuffer;
  bool vMemBufferOwned;
  bool vNoAlloc;

  int vNObjects;

  struct ggml_object *vObjectsBegin;
  struct ggml_object *vObjectsEnd;
};

//
// data types
//

static const char *GGML_OP_NAME[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "CUMSUM",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",
    "L2_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "SET_ROWS",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "IM2COL_3D",
    "COL2IM_1D",
    "CONV_2D",
    "CONV_3D",
    "CONV_2D_DW",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "PAD_REFLECT_1D",
    "ROLL",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "TOP_K",
    "LEAKY_RELU",
    "TRI",
    "FILL",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",
    "GATED_LINEAR_ATTN",
    "RWKV_WKV7",
    "SOLVE_TRI",
    "GATED_DELTA_NET",
    "LIGHTNING_INDEXER",
    "DSV4_HC_COMB",
    "DSV4_HC_PRE",
    "DSV4_HC_POST",
    "AMAX_POW2",

    "UNARY",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CUSTOM",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",

    "GLU",
};

static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");

static const char *GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x[i]+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "sin(x)",
    "cos(x)",
    "Σx",
    "Σx_k",
    "cumsum(x)",
    "Σx/n",
    "argmax(x)",
    "count_equal(x)",
    "repeat(x)",
    "repeat_back(x)",
    "concat(x, y)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",
    "group_norm(x)",
    "l2_norm(x)",

    "X*Y",
    "X[i]*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "set_rows(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "clamp(x)",
    "conv_transpose_1d(x)",
    "im2col(x)",
    "im2col_back(x)",
    "im2col_3d(x)",
    "col2im_1d(x)",
    "conv_2d(x)",
    "conv_3d(x)",
    "conv_2d_dw(x)",
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
    "pool_2d_back(x)",
    "upscale(x)",
    "pad(x)",
    "pad_reflect_1d(x)",
    "roll(x)",
    "arange(start, stop, step)",
    "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",
    "top_k(x)",
    "leaky_relu(x)",
    "tri(x)",
    "fill(x, c)",

    "flash_attn_ext(x)",
    "flash_attn_back(x)",
    "ssm_conv(x)",
    "ssm_scan(x)",
    "win_part(x)",
    "win_unpart(x)",
    "get_rel_pos(x)",
    "add_rel_pos(x)",
    "rwkv_wkv6(k, v, r, tf, td, s)",
    "gated_linear_attn(k, v, q, gate, s)",
    "rwkv_wkv7(r, w, k, v, a, b, s)",
    "A X = B, A triangular, solve X",
    "gated_delta_net(q, k, v, g, beta, s)",
    "lightning_indexer(q, k, weights, mask)",
    "dsv4_hc_comb(mixes, scale, base)",
    "dsv4_hc_pre(x, weights)",
    "dsv4_hc_post(x, residual, post, comb)",
    "amax_pow2(x)",

    "unary(x)",

    "map_custom(x)",
    "map_custom(x,y)",
    "map_custom(x,y,z)",

    "custom(x)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
    "adamw(x)",
    "sgd(x)",

    "glu(x)",
};

static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_COUNT == 2, "GGML_OP_POOL_COUNT != 2");

static const char *GGML_UNARY_OP_NAME[GGML_UNARY_OP_COUNT] = {
    "ABS",      "SGN",        "NEG",   "STEP",      "TANH",        "ELU",   "RELU",  "SIGMOID",
    "GELU",     "GELU_QUICK", "SILU",  "HARDSWISH", "HARDSIGMOID", "EXP",   "EXPM1", "SOFTPLUS",
    "GELU_ERF", "XIELU",      "FLOOR", "CEIL",      "ROUND",       "TRUNC",
};

static_assert(GGML_UNARY_OP_COUNT == 22, "GGML_UNARY_OP_COUNT != 22");

static const char *GGML_GLU_OP_NAME[GGML_GLU_OP_COUNT] = {
    "REGLU", "GEGLU", "SWIGLU", "SWIGLU_OAI", "GEGLU_ERF", "GEGLU_QUICK", "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_COUNT == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(sizeof(struct ggml_object) % GGML_MEM_ALIGN == 0,
              "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(sizeof(struct ggml_tensor) % GGML_MEM_ALIGN == 0,
              "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");

////////////////////////////////////////////////////////////////////////////////

void ggml_print_object(const struct ggml_object *pObj) {
  GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n", pObj->vType, pObj->vOffs,
                pObj->size, (const void *)pObj->vNext);
}

void ggml_print_objects(const struct ggml_context *pCtx) {
  struct ggml_object *vObj = pCtx->vObjectsBegin;

  GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (const void *)pCtx);

  while (vObj != NULL) {
    ggml_print_object(vObj);
    vObj = vObj->vNext;
  }

  GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->ne[0] * pTensor->ne[1] * pTensor->ne[2] * pTensor->ne[3];
}

int64_t ggml_nrows(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->ne[1] * pTensor->ne[2] * pTensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor *pTensor) {
  for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
    if (pTensor->ne[vI] <= 0) {
      return 0;
    }
  }

  size_t vNbytes;
  const size_t cBlckSize = ggml_blck_size(pTensor->type);
  if (cBlckSize == 1) {
    vNbytes = ggml_type_size(pTensor->type);
    for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
      vNbytes += (pTensor->ne[vI] - 1) * pTensor->nb[vI];
    }
  } else {
    vNbytes = pTensor->ne[0] * pTensor->nb[0] / cBlckSize;
    for (int vI = 1; vI < GGML_MAX_DIMS; ++vI) {
      vNbytes += (pTensor->ne[vI] - 1) * pTensor->nb[vI];
    }
  }

  return vNbytes;
}

size_t ggml_nbytes_pad(const struct ggml_tensor *pTensor) { return GGML_PAD(ggml_nbytes(pTensor), GGML_MEM_ALIGN); }

int64_t ggml_blck_size(enum ggml_type pType) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  return type_traits[pType].blck_size;
}

size_t ggml_type_size(enum ggml_type pType) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  return type_traits[pType].type_size;
}

size_t ggml_row_size(enum ggml_type pType, int64_t pNe) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  assert(pNe % ggml_blck_size(pType) == 0);
  return ggml_type_size(pType) * pNe / ggml_blck_size(pType);
}

const char *ggml_type_name(enum ggml_type pType) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  return type_traits[pType].type_name;
}

bool ggml_is_quantized(enum ggml_type pType) {
  assert(pType >= 0);
  assert(pType < GGML_TYPE_COUNT);
  return type_traits[pType].is_quantized;
}

const char *ggml_op_name(enum ggml_op pOp) { return GGML_OP_NAME[pOp]; }

const char *ggml_op_symbol(enum ggml_op pOp) { return GGML_OP_SYMBOL[pOp]; }

const char *ggml_unary_op_name(enum ggml_unary_op pOp) { return GGML_UNARY_OP_NAME[pOp]; }

const char *ggml_glu_op_name(enum ggml_glu_op pOp) { return GGML_GLU_OP_NAME[pOp]; }

const char *ggml_op_desc(const struct ggml_tensor *pT) {
  if (pT->op == GGML_OP_UNARY) {
    enum ggml_unary_op vUop = ggml_get_unary_op(pT);
    return ggml_unary_op_name(vUop);
  }
  if (pT->op == GGML_OP_GLU) {
    enum ggml_glu_op vGop = ggml_get_glu_op(pT);
    return ggml_glu_op_name(vGop);
  }
  return ggml_op_name(pT->op);
}

size_t ggml_element_size(const struct ggml_tensor *pTensor) { return ggml_type_size(pTensor->type); }

bool ggml_is_scalar(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->ne[0] == 1 && pTensor->ne[1] == 1 && pTensor->ne[2] == 1 && pTensor->ne[3] == 1;
}

bool ggml_is_vector(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->ne[1] == 1 && pTensor->ne[2] == 1 && pTensor->ne[3] == 1;
}

bool ggml_is_matrix(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->ne[2] == 1 && pTensor->ne[3] == 1;
}

bool ggml_is_3d(const struct ggml_tensor *pTensor) { return pTensor->ne[3] == 1; }

int ggml_n_dims(const struct ggml_tensor *pTensor) {
  for (int vI = GGML_MAX_DIMS - 1; vI >= 1; --vI) {
    if (pTensor->ne[vI] > 1) {
      return vI + 1;
    }
  }
  return 1;
}

enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype pFtype) {
  enum ggml_type vWtype = GGML_TYPE_COUNT;

  switch (pFtype) {
  case GGML_FTYPE_ALL_F32:
    vWtype = GGML_TYPE_F32;
    break;
  case GGML_FTYPE_MOSTLY_F16:
    vWtype = GGML_TYPE_F16;
    break;
  case GGML_FTYPE_MOSTLY_BF16:
    vWtype = GGML_TYPE_BF16;
    break;
  case GGML_FTYPE_MOSTLY_Q4_0:
    vWtype = GGML_TYPE_Q4_0;
    break;
  case GGML_FTYPE_MOSTLY_Q4_1:
    vWtype = GGML_TYPE_Q4_1;
    break;
  case GGML_FTYPE_MOSTLY_Q1_0:
    vWtype = GGML_TYPE_Q1_0;
    break;
  case GGML_FTYPE_MOSTLY_Q2_0:
    vWtype = GGML_TYPE_Q2_0;
    break;
  case GGML_FTYPE_MOSTLY_Q5_0:
    vWtype = GGML_TYPE_Q5_0;
    break;
  case GGML_FTYPE_MOSTLY_Q5_1:
    vWtype = GGML_TYPE_Q5_1;
    break;
  case GGML_FTYPE_MOSTLY_Q8_0:
    vWtype = GGML_TYPE_Q8_0;
    break;
  case GGML_FTYPE_MOSTLY_MXFP4:
    vWtype = GGML_TYPE_MXFP4;
    break;
  case GGML_FTYPE_MOSTLY_NVFP4:
    vWtype = GGML_TYPE_NVFP4;
    break;
  case GGML_FTYPE_MOSTLY_Q2_K:
    vWtype = GGML_TYPE_Q2_K;
    break;
  case GGML_FTYPE_MOSTLY_Q3_K:
    vWtype = GGML_TYPE_Q3_K;
    break;
  case GGML_FTYPE_MOSTLY_Q4_K:
    vWtype = GGML_TYPE_Q4_K;
    break;
  case GGML_FTYPE_MOSTLY_Q5_K:
    vWtype = GGML_TYPE_Q5_K;
    break;
  case GGML_FTYPE_MOSTLY_Q6_K:
    vWtype = GGML_TYPE_Q6_K;
    break;
  case GGML_FTYPE_MOSTLY_IQ2_XXS:
    vWtype = GGML_TYPE_IQ2_XXS;
    break;
  case GGML_FTYPE_MOSTLY_IQ2_XS:
    vWtype = GGML_TYPE_IQ2_XS;
    break;
  case GGML_FTYPE_MOSTLY_IQ3_XXS:
    vWtype = GGML_TYPE_IQ3_XXS;
    break;
  case GGML_FTYPE_MOSTLY_IQ1_S:
    vWtype = GGML_TYPE_IQ1_S;
    break;
  case GGML_FTYPE_MOSTLY_IQ1_M:
    vWtype = GGML_TYPE_IQ1_M;
    break;
  case GGML_FTYPE_MOSTLY_IQ4_NL:
    vWtype = GGML_TYPE_IQ4_NL;
    break;
  case GGML_FTYPE_MOSTLY_IQ4_XS:
    vWtype = GGML_TYPE_IQ4_XS;
    break;
  case GGML_FTYPE_MOSTLY_IQ3_S:
    vWtype = GGML_TYPE_IQ3_S;
    break;
  case GGML_FTYPE_MOSTLY_IQ2_S:
    vWtype = GGML_TYPE_IQ2_S;
    break;
  case GGML_FTYPE_UNKNOWN:
    vWtype = GGML_TYPE_COUNT;
    break;
  case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
    vWtype = GGML_TYPE_COUNT;
    break;
  }

  GGML_ASSERT(vWtype != GGML_TYPE_COUNT);

  return vWtype;
}

size_t ggml_tensor_overhead(void) { return GGML_OBJECT_SIZE + GGML_TENSOR_SIZE; }

bool ggml_is_transposed(const struct ggml_tensor *pTensor) { return pTensor->nb[0] > pTensor->nb[1]; }

static bool fGgmlIsContiguousMN(const struct ggml_tensor *pTensor, int pM, int pN) {
  size_t vNextNb = ggml_type_size(pTensor->type);
  if (pTensor->ne[0] != ggml_blck_size(pTensor->type) && pTensor->nb[0] != vNextNb) {
    return false;
  }
  vNextNb *= pTensor->ne[0] / ggml_blck_size(pTensor->type);
  for (int vI = 1; vI < pN; vI++) {
    if (vI > pM) {
      if (pTensor->ne[vI] != 1 && pTensor->nb[vI] != vNextNb) {
        return false;
      }
      vNextNb *= pTensor->ne[vI];
    } else {
      // this dimension does not need to be contiguous
      vNextNb = pTensor->ne[vI] * pTensor->nb[vI];
    }
  }
  return true;
}

bool ggml_is_contiguous(const struct ggml_tensor *pTensor) { return ggml_is_contiguous_0(pTensor); }

bool ggml_is_contiguous_0(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 0, GGML_MAX_DIMS); }

bool ggml_is_contiguous_1(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 1, GGML_MAX_DIMS); }

bool ggml_is_contiguous_2(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 2, GGML_MAX_DIMS); }

bool ggml_is_contiguous_to_1(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 0, 1); }

bool ggml_is_contiguous_to_2(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 0, 2); }

bool ggml_is_contiguous_to_3(const struct ggml_tensor *pTensor) { return fGgmlIsContiguousMN(pTensor, 0, 3); }

bool ggml_is_contiguously_allocated(const struct ggml_tensor *pTensor) {
  return ggml_nbytes(pTensor) ==
         ggml_nelements(pTensor) * ggml_type_size(pTensor->type) / ggml_blck_size(pTensor->type);
}

bool ggml_is_permuted(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->nb[0] > pTensor->nb[1] || pTensor->nb[1] > pTensor->nb[2] || pTensor->nb[2] > pTensor->nb[3];
}

bool ggml_is_contiguous_channels(const struct ggml_tensor *pTensor) {
  return pTensor->nb[0] > pTensor->nb[2] && pTensor->nb[1] > pTensor->nb[0] &&
         pTensor->nb[2] == ggml_type_size(pTensor->type);
}

bool ggml_is_contiguous_rows(const struct ggml_tensor *pTensor) {
  return pTensor->ne[0] == ggml_blck_size(pTensor->type) || pTensor->nb[0] == ggml_type_size(pTensor->type);
}

static inline bool fGgmlIsPadded1d(const struct ggml_tensor *pTensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return pTensor->nb[0] == ggml_type_size(pTensor->type) && pTensor->nb[2] == pTensor->nb[1] * pTensor->ne[1] &&
         pTensor->nb[3] == pTensor->nb[2] * pTensor->ne[2];
}

bool ggml_is_empty(const struct ggml_tensor *pTensor) {
  for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
    if (pTensor->ne[vI] == 0) {
      // empty if any dimension has no elements
      return true;
    }
  }
  return false;
}

bool ggml_are_same_shape(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (pT0->ne[0] == pT1->ne[0]) && (pT0->ne[1] == pT1->ne[1]) && (pT0->ne[2] == pT1->ne[2]) &&
         (pT0->ne[3] == pT1->ne[3]);
}

bool ggml_are_same_stride(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (pT0->nb[0] == pT1->nb[0]) && (pT0->nb[1] == pT1->nb[1]) && (pT0->nb[2] == pT1->nb[2]) &&
         (pT0->nb[3] == pT1->nb[3]);
}

bool ggml_is_view(const struct ggml_tensor *pT) { return ggml_impl_is_view(pT); }

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return ggml_is_empty(pT0) ? ggml_is_empty(pT1)
                            : (pT1->ne[0] % pT0->ne[0] == 0) && (pT1->ne[1] % pT0->ne[1] == 0) &&
                                  (pT1->ne[2] % pT0->ne[2] == 0) && (pT1->ne[3] % pT0->ne[3] == 0);
}

static inline bool fGgmlCanRepeatRows(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (pT0->ne[0] == pT1->ne[0]) && ggml_can_repeat(pT0, pT1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define GGML_ASSERT_ALIGNED(ptr) GGML_ASSERT(((uintptr_t)(ptr)) % GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct ggml_context *ggml_init(struct ggml_init_params pParams) {
  static bool vIsFirstCall = true;

  ggml_critical_section_start();

  if (vIsFirstCall) {
    // initialize time system (required on Windows)
    ggml_time_init();

    vIsFirstCall = false;
  }

  ggml_critical_section_end();

  struct ggml_context *vCtx = GGML_MALLOC(sizeof(struct ggml_context));

  // allow to call ggml_init with 0 size
  if (pParams.mem_size == 0) {
    pParams.mem_size = GGML_MEM_ALIGN;
  }

  const size_t cMemSize = pParams.mem_buffer ? pParams.mem_size : GGML_PAD(pParams.mem_size, GGML_MEM_ALIGN);

  *vCtx = (struct ggml_context){
      /*.mem_size           =*/cMemSize,
      /*.mem_buffer         =*/pParams.mem_buffer ? pParams.mem_buffer : ggml_aligned_malloc(cMemSize),
      /*.mem_buffer_owned   =*/pParams.mem_buffer ? false : true,
      /*.no_alloc           =*/pParams.no_alloc,
      /*.n_objects          =*/0,
      /*.objects_begin      =*/NULL,
      /*.objects_end        =*/NULL,
  };

  GGML_ASSERT(vCtx->vMemBuffer != NULL);

  GGML_ASSERT_ALIGNED(vCtx->vMemBuffer);

  GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

  return vCtx;
}

void ggml_reset(struct ggml_context *pCtx) {
  if (pCtx == NULL) {
    return;
  }

  pCtx->vNObjects = 0;
  pCtx->vObjectsBegin = NULL;
  pCtx->vObjectsEnd = NULL;
}

void ggml_free(struct ggml_context *pCtx) {
  if (pCtx == NULL) {
    return;
  }

  if (pCtx->vMemBufferOwned) {
    ggml_aligned_free(pCtx->vMemBuffer, pCtx->vMemSize);
  }

  GGML_FREE(pCtx);
}

size_t ggml_used_mem(const struct ggml_context *pCtx) {
  return pCtx->vObjectsEnd == NULL ? 0 : pCtx->vObjectsEnd->vOffs + pCtx->vObjectsEnd->size;
}

bool ggml_get_no_alloc(struct ggml_context *pCtx) { return pCtx->vNoAlloc; }

void ggml_set_no_alloc(struct ggml_context *pCtx, bool pNoAlloc) { pCtx->vNoAlloc = pNoAlloc; }

void *ggml_get_mem_buffer(const struct ggml_context *pCtx) { return pCtx->vMemBuffer; }

size_t ggml_get_mem_size(const struct ggml_context *pCtx) { return pCtx->vMemSize; }

size_t ggml_get_max_tensor_size(const struct ggml_context *pCtx) {
  size_t vMaxSize = 0;

  for (struct ggml_tensor *vTensor = ggml_get_first_tensor(pCtx); vTensor != NULL;
       vTensor = ggml_get_next_tensor(pCtx, vTensor)) {
    size_t vBytes = ggml_nbytes(vTensor);
    vMaxSize = MAX(vMaxSize, vBytes);
  }

  return vMaxSize;
}

////////////////////////////////////////////////////////////////////////////////

static struct ggml_object *fGgmlNewObject(struct ggml_context *pCtx, enum ggml_object_type pType, size_t pSize) {
  // always insert objects at the end of the context's memory pool
  struct ggml_object *vObjCur = pCtx->vObjectsEnd;

  const size_t cCurOffs = vObjCur == NULL ? 0 : vObjCur->vOffs;
  const size_t cCurSize = vObjCur == NULL ? 0 : vObjCur->size;
  const size_t cCurEnd = cCurOffs + cCurSize;

  // align to GGML_MEM_ALIGN
  GGML_ASSERT(pSize <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
  size_t vSizeNeeded = GGML_PAD(pSize, GGML_MEM_ALIGN);

  char *const cMemBuffer = pCtx->vMemBuffer;
  struct ggml_object *const cObjNew = (struct ggml_object *)(cMemBuffer + cCurEnd);

  // integer overflow checks
  if (cCurEnd > SIZE_MAX - vSizeNeeded) {
    GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cCurEnd, vSizeNeeded);
    return NULL;
  }
  if (cCurEnd + vSizeNeeded > SIZE_MAX - GGML_OBJECT_SIZE) {
    GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                  cCurEnd, vSizeNeeded, (size_t)GGML_OBJECT_SIZE);
    return NULL;
  }

  if (cCurEnd + vSizeNeeded + GGML_OBJECT_SIZE > pCtx->vMemSize) {
    GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n", __func__,
                  cCurEnd + vSizeNeeded + GGML_OBJECT_SIZE, pCtx->vMemSize);
#ifndef NDEBUG
    GGML_ABORT("not enough space in the context's memory pool");
#endif
    return NULL;
  }

  *cObjNew = (struct ggml_object){
      .vOffs = cCurEnd + GGML_OBJECT_SIZE,
      .size = vSizeNeeded,
      .vNext = NULL,
      .vType = pType,
  };

  GGML_ASSERT_ALIGNED(cMemBuffer + cObjNew->vOffs);

  if (vObjCur != NULL) {
    vObjCur->vNext = cObjNew;
  } else {
    // this is the first object in this context
    pCtx->vObjectsBegin = cObjNew;
  }

  pCtx->vObjectsEnd = cObjNew;

  // printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

  return cObjNew;
}

static struct ggml_tensor *fGgmlNewTensorImpl(struct ggml_context *pCtx, enum ggml_type pType, int pNDims,
                                              const int64_t *pNe, struct ggml_tensor *pViewSrc, size_t pViewOffs) {

  GGML_ASSERT(pType >= 0 && pType < GGML_TYPE_COUNT);
  GGML_ASSERT(pNDims >= 1 && pNDims <= GGML_MAX_DIMS);

  // find the base tensor and absolute offset
  if (pViewSrc != NULL && pViewSrc->view_src != NULL) {
    pViewOffs += pViewSrc->view_offs;
    pViewSrc = pViewSrc->view_src;
  }

  size_t vDataSize = ggml_row_size(pType, pNe[0]);
  for (int vI = 1; vI < pNDims; vI++) {
    vDataSize *= pNe[vI];
  }

  GGML_ASSERT(pViewSrc == NULL || vDataSize == 0 || vDataSize + pViewOffs <= ggml_nbytes(pViewSrc));

  void *vData = pViewSrc != NULL ? pViewSrc->data : NULL;
  if (vData != NULL) {
    vData = (char *)vData + pViewOffs;
  }

  size_t vObjAllocSize = 0;

  if (pViewSrc == NULL && !pCtx->vNoAlloc) {
    // allocate tensor data in the context's memory pool
    vObjAllocSize = vDataSize;
  }

  GGML_ASSERT(GGML_TENSOR_SIZE <= SIZE_MAX - vObjAllocSize);

  struct ggml_object *const cObjNew = fGgmlNewObject(pCtx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + vObjAllocSize);
  GGML_ASSERT(cObjNew);

  struct ggml_tensor *const cResult = (struct ggml_tensor *)((char *)pCtx->vMemBuffer + cObjNew->vOffs);

  *cResult = (struct ggml_tensor){
      /*.type         =*/pType,
      /*.buffer       =*/NULL,
      /*.ne           =*/{1, 1, 1, 1},
      /*.nb           =*/{0, 0, 0, 0},
      /*.op           =*/GGML_OP_NONE,
      /*.op_params    =*/{0},
      /*.flags        =*/0,
      /*.src          =*/{NULL},
      /*.view_src     =*/pViewSrc,
      /*.view_offs    =*/pViewOffs,
      /*.data         =*/vObjAllocSize > 0 ? (void *)(cResult + 1) : vData,
      /*.name         =*/{0},
      /*.extra        =*/NULL,
      /*.padding      =*/{0},
  };

  // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
  // GGML_ASSERT_ALIGNED(result->data);

  for (int vI = 0; vI < pNDims; vI++) {
    cResult->ne[vI] = pNe[vI];
  }

  cResult->nb[0] = ggml_type_size(pType);
  cResult->nb[1] = cResult->nb[0] * (cResult->ne[0] / ggml_blck_size(pType));
  for (int vI = 2; vI < GGML_MAX_DIMS; vI++) {
    cResult->nb[vI] = cResult->nb[vI - 1] * cResult->ne[vI - 1];
  }

  pCtx->vNObjects++;

  return cResult;
}

struct ggml_tensor *ggml_new_tensor(struct ggml_context *pCtx, enum ggml_type pType, int pNDims, const int64_t *pNe) {
  return fGgmlNewTensorImpl(pCtx, pType, pNDims, pNe, NULL, 0);
}

struct ggml_tensor *ggml_new_tensor_1d(struct ggml_context *pCtx, enum ggml_type pType, int64_t pNe0) {
  return ggml_new_tensor(pCtx, pType, 1, &pNe0);
}

struct ggml_tensor *ggml_new_tensor_2d(struct ggml_context *pCtx, enum ggml_type pType, int64_t pNe0, int64_t pNe1) {
  const int64_t cNe[2] = {pNe0, pNe1};
  return ggml_new_tensor(pCtx, pType, 2, cNe);
}

struct ggml_tensor *ggml_new_tensor_3d(struct ggml_context *pCtx, enum ggml_type pType, int64_t pNe0, int64_t pNe1,
                                       int64_t pNe2) {
  const int64_t cNe[3] = {pNe0, pNe1, pNe2};
  return ggml_new_tensor(pCtx, pType, 3, cNe);
}

struct ggml_tensor *ggml_new_tensor_4d(struct ggml_context *pCtx, enum ggml_type pType, int64_t pNe0, int64_t pNe1,
                                       int64_t pNe2, int64_t pNe3) {
  const int64_t cNe[4] = {pNe0, pNe1, pNe2, pNe3};
  return ggml_new_tensor(pCtx, pType, 4, cNe);
}

void *ggml_new_buffer(struct ggml_context *pCtx, size_t pNbytes) {
  struct ggml_object *vObj = fGgmlNewObject(pCtx, GGML_OBJECT_TYPE_WORK_BUFFER, pNbytes);

  return (uint8_t *)pCtx->vMemBuffer + vObj->vOffs;
}

struct ggml_tensor *ggml_dup_tensor(struct ggml_context *pCtx, const struct ggml_tensor *pSrc) {
  return ggml_new_tensor(pCtx, pSrc->type, GGML_MAX_DIMS, pSrc->ne);
}

void ggml_unravel_index(const struct ggml_tensor *pTensor, int64_t pI, int64_t *pI0, int64_t *pI1, int64_t *pI2,
                        int64_t *pI3) {
  const int64_t cNe2 = pTensor->ne[2];
  const int64_t cNe1 = pTensor->ne[1];
  const int64_t cNe0 = pTensor->ne[0];

  const int64_t cI3 = (pI / (cNe2 * cNe1 * cNe0));
  const int64_t cI2 = (pI - cI3 * cNe2 * cNe1 * cNe0) / (cNe1 * cNe0);
  const int64_t cI1 = (pI - cI3 * cNe2 * cNe1 * cNe0 - cI2 * cNe1 * cNe0) / cNe0;
  const int64_t cI0 = (pI - cI3 * cNe2 * cNe1 * cNe0 - cI2 * cNe1 * cNe0 - cI1 * cNe0);

  if (pI0) {
    *pI0 = cI0;
  }
  if (pI1) {
    *pI1 = cI1;
  }
  if (pI2) {
    *pI2 = cI2;
  }
  if (pI3) {
    *pI3 = cI3;
  }
}

void *ggml_get_data(const struct ggml_tensor *pTensor) { return pTensor->data; }

float *ggml_get_data_f32(const struct ggml_tensor *pTensor) {
  assert(pTensor->type == GGML_TYPE_F32);
  return (float *)(pTensor->data);
}

enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor *pTensor) {
  GGML_ASSERT(pTensor->op == GGML_OP_UNARY);
  return (enum ggml_unary_op)ggml_get_op_params_i32(pTensor, 0);
}

enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor *pTensor) {
  GGML_ASSERT(pTensor->op == GGML_OP_GLU);
  return (enum ggml_glu_op)ggml_get_op_params_i32(pTensor, 0);
}

const char *ggml_get_name(const struct ggml_tensor *pTensor) { return pTensor->name; }

struct ggml_tensor *ggml_set_name(struct ggml_tensor *pTensor, const char *pName) {
  size_t vI;
  for (vI = 0; vI < sizeof(pTensor->name) - 1 && pName[vI] != '\0'; vI++) {
    pTensor->name[vI] = pName[vI];
  }
  pTensor->name[vI] = '\0';
  return pTensor;
}

struct ggml_tensor *ggml_format_name(struct ggml_tensor *pTensor, const char *pFmt, ...) {
  va_list vArgs;
  va_start(vArgs, pFmt);
  vsnprintf(pTensor->name, sizeof(pTensor->name), pFmt, vArgs);
  va_end(vArgs);
  return pTensor;
}

struct ggml_tensor *ggml_view_tensor(struct ggml_context *pCtx, struct ggml_tensor *pSrc) {
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pSrc->type, GGML_MAX_DIMS, pSrc->ne, pSrc, 0);
  ggml_format_name(vResult, "%s (view)", pSrc->name);

  for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
    vResult->nb[vI] = pSrc->nb[vI];
  }

  return vResult;
}

struct ggml_tensor *ggml_get_first_tensor(const struct ggml_context *pCtx) {
  struct ggml_object *vObj = pCtx->vObjectsBegin;

  char *const cMemBuffer = pCtx->vMemBuffer;

  while (vObj != NULL) {
    if (vObj->vType == GGML_OBJECT_TYPE_TENSOR) {
      return (struct ggml_tensor *)(cMemBuffer + vObj->vOffs);
    }

    vObj = vObj->vNext;
  }

  return NULL;
}

struct ggml_tensor *ggml_get_next_tensor(const struct ggml_context *pCtx, struct ggml_tensor *pTensor) {
  struct ggml_object *vObj = (struct ggml_object *)((char *)pTensor - GGML_OBJECT_SIZE);
  vObj = vObj->vNext;

  char *const cMemBuffer = pCtx->vMemBuffer;

  while (vObj != NULL) {
    if (vObj->vType == GGML_OBJECT_TYPE_TENSOR) {
      return (struct ggml_tensor *)(cMemBuffer + vObj->vOffs);
    }

    vObj = vObj->vNext;
  }

  return NULL;
}

struct ggml_tensor *ggml_get_tensor(struct ggml_context *pCtx, const char *pName) {
  struct ggml_object *vObj = pCtx->vObjectsBegin;

  char *const cMemBuffer = pCtx->vMemBuffer;

  while (vObj != NULL) {
    if (vObj->vType == GGML_OBJECT_TYPE_TENSOR) {
      struct ggml_tensor *vCur = (struct ggml_tensor *)(cMemBuffer + vObj->vOffs);
      if (strcmp(vCur->name, pName) == 0) {
        return vCur;
      }
    }

    vObj = vObj->vNext;
  }

  return NULL;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

static struct ggml_tensor *fGgmlDupImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_DUP;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_dup(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlDupImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_dup_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlDupImpl(pCtx, pA, true);
}

// ggml_add

static struct ggml_tensor *fGgmlAddImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        bool pInplace) {
  GGML_ASSERT(ggml_can_repeat(pB, pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_ADD;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_add(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlAddImpl(pCtx, pA, pB, false);
}

struct ggml_tensor *ggml_add_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlAddImpl(pCtx, pA, pB, true);
}

// ggml_add_cast

static struct ggml_tensor *fGgmlAddCastImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                            enum ggml_type pType) {
  // TODO: support less-strict constraint
  //       GGML_ASSERT(ggml_can_repeat(b, a));
  GGML_ASSERT(fGgmlCanRepeatRows(pB, pA));

  // currently only supported for quantized input and f16
  GGML_ASSERT(ggml_is_quantized(pA->type) || pA->type == GGML_TYPE_F16 || pA->type == GGML_TYPE_BF16);

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pType, GGML_MAX_DIMS, pA->ne);

  vResult->op = GGML_OP_ADD;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_add_cast(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                  enum ggml_type pType) {
  return fGgmlAddCastImpl(pCtx, pA, pB, pType);
}

struct ggml_tensor *ggml_add_id(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                struct ggml_tensor *pIds) {

  GGML_ASSERT(pA->ne[0] == pB->ne[0]);
  GGML_ASSERT(pA->ne[1] == pIds->ne[0]);
  GGML_ASSERT(pA->ne[2] == pIds->ne[1]);
  GGML_ASSERT(pIds->type == GGML_TYPE_I32);

  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_ADD_ID;
  vResult->src[0] = pA;
  vResult->src[1] = pB;
  vResult->src[2] = pIds;

  return vResult;
}

// ggml_add1

static struct ggml_tensor *fGgmlAdd1Impl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                         bool pInplace) {
  GGML_ASSERT(ggml_is_scalar(pB));
  GGML_ASSERT(fGgmlIsPadded1d(pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_ADD1;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_acc

static struct ggml_tensor *fGgmlAccImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        size_t pNb1, size_t pNb2, size_t pNb3, size_t pOffset, bool pInplace) {
  GGML_ASSERT(ggml_nelements(pB) <= ggml_nelements(pA));
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(pA->type == GGML_TYPE_F32);
  GGML_ASSERT(pB->type == GGML_TYPE_F32);

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  int32_t aParams[] = {pNb1, pNb2, pNb3, pOffset, pInplace ? 1 : 0};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_ACC;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_acc(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, size_t pNb1,
                             size_t pNb2, size_t pNb3, size_t pOffset) {
  return fGgmlAccImpl(pCtx, pA, pB, pNb1, pNb2, pNb3, pOffset, false);
}

struct ggml_tensor *ggml_acc_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                     size_t pNb1, size_t pNb2, size_t pNb3, size_t pOffset) {
  return fGgmlAccImpl(pCtx, pA, pB, pNb1, pNb2, pNb3, pOffset, true);
}

// ggml_sub

static struct ggml_tensor *fGgmlSubImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        bool pInplace) {
  GGML_ASSERT(ggml_can_repeat(pB, pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SUB;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_sub(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlSubImpl(pCtx, pA, pB, false);
}

struct ggml_tensor *ggml_sub_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlSubImpl(pCtx, pA, pB, true);
}

// ggml_mul

static struct ggml_tensor *fGgmlMulImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        bool pInplace) {
  GGML_ASSERT(ggml_can_repeat(pB, pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_MUL;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_mul(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlMulImpl(pCtx, pA, pB, false);
}

struct ggml_tensor *ggml_mul_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlMulImpl(pCtx, pA, pB, true);
}

// ggml_div

static struct ggml_tensor *fGgmlDivImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        bool pInplace) {
  GGML_ASSERT(ggml_can_repeat(pB, pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_DIV;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_div(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlDivImpl(pCtx, pA, pB, false);
}

struct ggml_tensor *ggml_div_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlDivImpl(pCtx, pA, pB, true);
}

// ggml_sqr

static struct ggml_tensor *fGgmlSqrImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SQR;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_sqr(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSqrImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_sqr_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSqrImpl(pCtx, pA, true);
}

// ggml_sqrt

static struct ggml_tensor *fGgmlSqrtImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SQRT;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_sqrt(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSqrtImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_sqrt_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSqrtImpl(pCtx, pA, true);
}

// ggml_log

static struct ggml_tensor *fGgmlLogImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_LOG;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_log(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlLogImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_log_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlLogImpl(pCtx, pA, true);
}

struct ggml_tensor *ggml_expm1(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_EXPM1);
}

struct ggml_tensor *ggml_expm1_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_EXPM1);
}

struct ggml_tensor *ggml_softplus(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_SOFTPLUS);
}

struct ggml_tensor *ggml_softplus_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

static struct ggml_tensor *fGgmlSinImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SIN;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_sin(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSinImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_sin_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSinImpl(pCtx, pA, true);
}

// ggml_cos

static struct ggml_tensor *fGgmlCosImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_COS;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_cos(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlCosImpl(pCtx, pA, false);
}

struct ggml_tensor *ggml_cos_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlCosImpl(pCtx, pA, true);
}

// ggml_sum

struct ggml_tensor *ggml_sum(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, pA->type, 1);

  vResult->op = GGML_OP_SUM;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_sum_rows

struct ggml_tensor *ggml_sum_rows(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  int64_t aNe[GGML_MAX_DIMS] = {1};
  for (int vI = 1; vI < GGML_MAX_DIMS; ++vI) {
    aNe[vI] = pA->ne[vI];
  }

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, GGML_MAX_DIMS, aNe);

  vResult->op = GGML_OP_SUM_ROWS;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_cumsum

struct ggml_tensor *ggml_cumsum(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_CUMSUM;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_mean

struct ggml_tensor *ggml_mean(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  int64_t aNe[4] = {1, pA->ne[1], pA->ne[2], pA->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, aNe);

  vResult->op = GGML_OP_MEAN;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_argmax

struct ggml_tensor *ggml_argmax(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  GGML_ASSERT(ggml_is_matrix(pA));
  GGML_ASSERT(pA->ne[0] <= INT32_MAX);

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, pA->ne[1]);

  vResult->op = GGML_OP_ARGMAX;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_amax_pow2

struct ggml_tensor *ggml_amax_pow2(struct ggml_context *pCtx, struct ggml_tensor *pA, float pThreshold) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32);
  GGML_ASSERT(pThreshold > 0.0f);

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, GGML_TYPE_F32, 1, pA->ne[1], pA->ne[2], pA->ne[3]);

  ggml_set_op_params_f32(vResult, 0, pThreshold);

  vResult->op = GGML_OP_AMAX_POW2;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_count_equal

struct ggml_tensor *ggml_count_equal(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_are_same_shape(pA, pB));

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, 1);

  vResult->op = GGML_OP_COUNT_EQUAL;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_repeat

struct ggml_tensor *ggml_repeat(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_can_repeat(pA, pB));

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, GGML_MAX_DIMS, pB->ne);

  vResult->op = GGML_OP_REPEAT;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_repeat_4d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                   int64_t pNe2, int64_t pNe3) {
  const bool cCanRepeat = ggml_is_empty(pA) || ((pNe0 % pA->ne[0] == 0) && (pNe1 % pA->ne[1] == 0) &&
                                                (pNe2 % pA->ne[2] == 0) && (pNe3 % pA->ne[3] == 0));
  GGML_ASSERT(cCanRepeat);

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, pA->type, pNe0, pNe1, pNe2, pNe3);

  vResult->op = GGML_OP_REPEAT;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_repeat_back

struct ggml_tensor *ggml_repeat_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_can_repeat(pB, pA));

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, GGML_MAX_DIMS, pB->ne);

  vResult->op = GGML_OP_REPEAT_BACK;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_concat

struct ggml_tensor *ggml_concat(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pDim) {
  GGML_ASSERT(pDim >= 0 && pDim < GGML_MAX_DIMS);
  GGML_ASSERT(pA->type == pB->type);

  int64_t aNe[GGML_MAX_DIMS];
  for (int vD = 0; vD < GGML_MAX_DIMS; ++vD) {
    if (vD == pDim) {
      aNe[vD] = pA->ne[vD] + pB->ne[vD];
      continue;
    }
    GGML_ASSERT(pA->ne[vD] == pB->ne[vD]);
    aNe[vD] = pA->ne[vD];
  }

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, GGML_MAX_DIMS, aNe);

  ggml_set_op_params_i32(vResult, 0, pDim);

  vResult->op = GGML_OP_CONCAT;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_abs

struct ggml_tensor *ggml_abs(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_ABS);
}

struct ggml_tensor *ggml_abs_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_ABS);
}

// ggml_sgn

struct ggml_tensor *ggml_sgn(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_SGN);
}

struct ggml_tensor *ggml_sgn_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_SGN);
}

// ggml_neg

struct ggml_tensor *ggml_neg(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_NEG);
}

struct ggml_tensor *ggml_neg_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_NEG);
}

// ggml_step

struct ggml_tensor *ggml_step(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_STEP);
}

struct ggml_tensor *ggml_step_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_STEP);
}

// ggml_tanh

struct ggml_tensor *ggml_tanh(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_TANH);
}

struct ggml_tensor *ggml_tanh_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_TANH);
}

// ggml_elu

struct ggml_tensor *ggml_elu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_ELU);
}

struct ggml_tensor *ggml_elu_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_ELU);
}

// ggml_relu

struct ggml_tensor *ggml_relu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_RELU);
}

struct ggml_tensor *ggml_relu_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

struct ggml_tensor *ggml_leaky_relu(struct ggml_context *pCtx, struct ggml_tensor *pA, float pNegativeSlope,
                                    bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params(vResult, &pNegativeSlope, sizeof(pNegativeSlope));

  vResult->op = GGML_OP_LEAKY_RELU;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_sigmoid

struct ggml_tensor *ggml_sigmoid(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_SIGMOID);
}

struct ggml_tensor *ggml_sigmoid_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

struct ggml_tensor *ggml_gelu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_GELU);
}

struct ggml_tensor *ggml_gelu_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

struct ggml_tensor *ggml_gelu_erf(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_GELU_ERF);
}

struct ggml_tensor *ggml_gelu_erf_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

struct ggml_tensor *ggml_gelu_quick(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_GELU_QUICK);
}

struct ggml_tensor *ggml_gelu_quick_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

struct ggml_tensor *ggml_silu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_SILU);
}

struct ggml_tensor *ggml_silu_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_SILU);
}

// ggml_xielu

struct ggml_tensor *ggml_xielu(struct ggml_context *pCtx, struct ggml_tensor *pA, float pAlphaN, float pAlphaP,
                               float pBeta, float pEps) {
  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_i32(vResult, 0, (int32_t)GGML_UNARY_OP_XIELU);
  ggml_set_op_params_f32(vResult, 1, pBeta + ggml_compute_softplus_f32(pAlphaN));
  ggml_set_op_params_f32(vResult, 2, ggml_compute_softplus_f32(pAlphaP));
  ggml_set_op_params_f32(vResult, 3, pBeta);
  ggml_set_op_params_f32(vResult, 4, pEps);

  vResult->op = GGML_OP_UNARY;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_silu_back

struct ggml_tensor *ggml_silu_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SILU_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml hardswish

struct ggml_tensor *ggml_hardswish(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

struct ggml_tensor *ggml_hardsigmoid(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

struct ggml_tensor *ggml_exp(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_EXP);
}

struct ggml_tensor *ggml_exp_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_EXP);
}

// ggml_glu

static struct ggml_tensor *fGgmlGluImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        enum ggml_glu_op pOp, bool pSwapped) {
  GGML_ASSERT(ggml_is_contiguous_1(pA));

  if (pB) {
    GGML_ASSERT(ggml_is_contiguous_1(pB));
    GGML_ASSERT(ggml_are_same_shape(pA, pB));
    GGML_ASSERT(pA->type == pB->type);
  }

  int64_t aNe[GGML_MAX_DIMS] = {pA->ne[0] / 2};
  for (int vI = 1; vI < GGML_MAX_DIMS; vI++)
    aNe[vI] = pA->ne[vI];
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, GGML_MAX_DIMS, pB ? pA->ne : aNe, NULL, 0);

  ggml_set_op_params_i32(vResult, 0, (int32_t)pOp);
  ggml_set_op_params_i32(vResult, 1, (int32_t)pSwapped);

  vResult->op = GGML_OP_GLU;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_floor

struct ggml_tensor *ggml_floor(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_FLOOR);
}

struct ggml_tensor *ggml_floor_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

struct ggml_tensor *ggml_ceil(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_CEIL);
}

struct ggml_tensor *ggml_ceil_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_CEIL);
}

// ggml_round

struct ggml_tensor *ggml_round(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_ROUND);
}

struct ggml_tensor *ggml_round_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_ROUND);
}

// ggml_trunc

struct ggml_tensor *ggml_trunc(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary(pCtx, pA, GGML_UNARY_OP_TRUNC);
}

struct ggml_tensor *ggml_trunc_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return ggml_unary_inplace(pCtx, pA, GGML_UNARY_OP_TRUNC);
}

struct ggml_tensor *ggml_glu(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_glu_op pOp, bool pSwapped) {
  return fGgmlGluImpl(pCtx, pA, NULL, pOp, pSwapped);
}

struct ggml_tensor *ggml_glu_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                   enum ggml_glu_op pOp) {
  return fGgmlGluImpl(pCtx, pA, pB, pOp, false);
}

// ggml_reglu

struct ggml_tensor *ggml_reglu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_REGLU, false);
}

struct ggml_tensor *ggml_reglu_swapped(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_REGLU, true);
}

struct ggml_tensor *ggml_reglu_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

struct ggml_tensor *ggml_geglu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU, false);
}

struct ggml_tensor *ggml_geglu_swapped(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU, true);
}

struct ggml_tensor *ggml_geglu_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

struct ggml_tensor *ggml_swiglu(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_SWIGLU, false);
}

struct ggml_tensor *ggml_swiglu_swapped(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_SWIGLU, true);
}

struct ggml_tensor *ggml_swiglu_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

struct ggml_tensor *ggml_geglu_erf(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU_ERF, false);
}

struct ggml_tensor *ggml_geglu_erf_swapped(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU_ERF, true);
}

struct ggml_tensor *ggml_geglu_erf_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

struct ggml_tensor *ggml_geglu_quick(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU_QUICK, false);
}

struct ggml_tensor *ggml_geglu_quick_swapped(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlGluImpl(pCtx, pA, NULL, GGML_GLU_OP_GEGLU_QUICK, true);
}

struct ggml_tensor *ggml_geglu_quick_split(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_GEGLU_QUICK, false);
}

struct ggml_tensor *ggml_swiglu_oai(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                    float pAlpha, float pLimit) {
  struct ggml_tensor *vResult = fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_SWIGLU_OAI, false);
  ggml_set_op_params_f32(vResult, 2, pAlpha);
  ggml_set_op_params_f32(vResult, 3, pLimit);

  return vResult;
}

struct ggml_tensor *ggml_swiglu_clamp(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                      float pLimit) {
  struct ggml_tensor *vResult = fGgmlGluImpl(pCtx, pA, pB, GGML_GLU_OP_SWIGLU_CLAMP, false);
  ggml_set_op_params_f32(vResult, 3, pLimit);

  return vResult;
}

// ggml_norm

static struct ggml_tensor *fGgmlNormImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params(vResult, &pEps, sizeof(pEps));

  vResult->op = GGML_OP_NORM;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_norm(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlNormImpl(pCtx, pA, pEps, false);
}

struct ggml_tensor *ggml_norm_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlNormImpl(pCtx, pA, pEps, true);
}

// ggml_rms_norm

static struct ggml_tensor *fGgmlRmsNormImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps,
                                            bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params(vResult, &pEps, sizeof(pEps));

  vResult->op = GGML_OP_RMS_NORM;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_rms_norm(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlRmsNormImpl(pCtx, pA, pEps, false);
}

struct ggml_tensor *ggml_rms_norm_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlRmsNormImpl(pCtx, pA, pEps, true);
}

// ggml_rms_norm_back

struct ggml_tensor *ggml_rms_norm_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                       float pEps) {
  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params(vResult, &pEps, sizeof(pEps));

  vResult->op = GGML_OP_RMS_NORM_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_group_norm

static struct ggml_tensor *fGgmlGroupNormImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNGroups,
                                              float pEps, bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_i32(vResult, 0, pNGroups);
  ggml_set_op_params_f32(vResult, 1, pEps);

  vResult->op = GGML_OP_GROUP_NORM;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_group_norm(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNGroups, float pEps) {
  return fGgmlGroupNormImpl(pCtx, pA, pNGroups, pEps, false);
}

struct ggml_tensor *ggml_group_norm_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNGroups,
                                            float pEps) {
  return fGgmlGroupNormImpl(pCtx, pA, pNGroups, pEps, true);
}

// ggml_l2_norm

static struct ggml_tensor *fGgmlL2NormImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps,
                                           bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_f32(vResult, 0, pEps);

  vResult->op = GGML_OP_L2_NORM;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_l2_norm(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlL2NormImpl(pCtx, pA, pEps, false);
}

struct ggml_tensor *ggml_l2_norm_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pEps) {
  return fGgmlL2NormImpl(pCtx, pA, pEps, true);
}

// ggml_mul_mat

static inline bool fGgmlCanMulMat(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (pT0->ne[0] == pT1->ne[0]) && (pT1->ne[2] % pT0->ne[2] == 0) && // verify t0 is broadcastable
         (pT1->ne[3] % pT0->ne[3] == 0);
}

struct ggml_tensor *ggml_mul_mat(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(fGgmlCanMulMat(pA, pB));
  GGML_ASSERT(!ggml_is_transposed(pA));

  const int64_t cNe[4] = {pA->ne[1], pB->ne[1], pB->ne[2], pB->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  vResult->op = GGML_OP_MUL_MAT;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

void ggml_mul_mat_set_prec(struct ggml_tensor *pA, enum ggml_prec pPrec) {
  GGML_ASSERT(pA->op == GGML_OP_MUL_MAT);

  const int32_t cPrecI32 = (int32_t)pPrec;

  ggml_set_op_params_i32(pA, 0, cPrecI32);
}

void ggml_mul_mat_set_hint(struct ggml_tensor *pA, enum ggml_op_hint pHint) {
  GGML_ASSERT(pA->op == GGML_OP_MUL_MAT);

  const int32_t cHintI32 = (int32_t)pHint;

  ggml_set_op_params_i32(pA, 1, cHintI32);
}

// ggml_mul_mat_id

/*
    c = ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    b   -> [cols, n_expert_used, n_tokens]
    ids -> [n_expert_used, n_tokens] (i32)
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_expert_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
struct ggml_tensor *ggml_mul_mat_id(struct ggml_context *pCtx, struct ggml_tensor *pAs, struct ggml_tensor *pB,
                                    struct ggml_tensor *pIds) {
  GGML_ASSERT(!ggml_is_transposed(pAs));
  GGML_ASSERT(pIds->type == GGML_TYPE_I32);

  GGML_ASSERT(pAs->ne[3] == 1);                      // as is 3d (one matrix per expert)
  GGML_ASSERT(pB->ne[3] == 1);                       // b is 3d
  GGML_ASSERT(pIds->ne[2] == 1 && pIds->ne[3] == 1); // ids is 2d
  GGML_ASSERT(pIds->ne[1] == pB->ne[2]);             // must have an expert list per b row
  GGML_ASSERT(pAs->ne[0] == pB->ne[0]);              // can_mul_mat
  GGML_ASSERT(pIds->ne[0] % pB->ne[1] == 0);         // can broadcast

  const int64_t cNe[4] = {pAs->ne[1], pIds->ne[0], pB->ne[2], 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  vResult->op = GGML_OP_MUL_MAT_ID;
  vResult->src[0] = pAs;
  vResult->src[1] = pB;
  vResult->src[2] = pIds;

  return vResult;
}

// ggml_out_prod

static inline bool fGgmlCanOutProd(const struct ggml_tensor *pT0, const struct ggml_tensor *pT1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (pT0->ne[1] == pT1->ne[1]) && (pT1->ne[2] % pT0->ne[2] == 0) && // verify t0 is broadcastable
         (pT1->ne[3] % pT0->ne[3] == 0);
}

struct ggml_tensor *ggml_out_prod(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(fGgmlCanOutProd(pA, pB));
  GGML_ASSERT(!ggml_is_transposed(pA));

  // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
  const int64_t cNe[4] = {pA->ne[0], pB->ne[0], pB->ne[2], pB->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  vResult->op = GGML_OP_OUT_PROD;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_scale

static struct ggml_tensor *fGgmlScaleImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, float pS, float pB,
                                          bool pInplace) {
  GGML_ASSERT(fGgmlIsPadded1d(pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  float aParams[2] = {pS, pB};
  ggml_set_op_params(vResult, &aParams, sizeof(aParams));

  vResult->op = GGML_OP_SCALE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_scale(struct ggml_context *pCtx, struct ggml_tensor *pA, float pS) {
  return fGgmlScaleImpl(pCtx, pA, pS, 0.0, false);
}

struct ggml_tensor *ggml_scale_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pS) {
  return fGgmlScaleImpl(pCtx, pA, pS, 0.0, true);
}

struct ggml_tensor *ggml_scale_bias(struct ggml_context *pCtx, struct ggml_tensor *pA, float pS, float pB) {
  return fGgmlScaleImpl(pCtx, pA, pS, pB, false);
}

struct ggml_tensor *ggml_scale_bias_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pS, float pB) {
  return fGgmlScaleImpl(pCtx, pA, pS, pB, true);
}

// ggml_set

static struct ggml_tensor *fGgmlSetImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        size_t pNb1, size_t pNb2, size_t pNb3, size_t pOffset, bool pInplace) {
  GGML_ASSERT(ggml_nelements(pA) >= ggml_nelements(pB));

  // make a view of the destination
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  GGML_ASSERT(pOffset < (size_t)(1 << 30));
  int32_t aParams[] = {pNb1, pNb2, pNb3, pOffset, pInplace ? 1 : 0};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_SET;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_set(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, size_t pNb1,
                             size_t pNb2, size_t pNb3, size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pNb1, pNb2, pNb3, pOffset, false);
}

struct ggml_tensor *ggml_set_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                     size_t pNb1, size_t pNb2, size_t pNb3, size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pNb1, pNb2, pNb3, pOffset, true);
}

struct ggml_tensor *ggml_set_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pA->nb[1], pA->nb[2], pA->nb[3], pOffset, false);
}

struct ggml_tensor *ggml_set_1d_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pA->nb[1], pA->nb[2], pA->nb[3], pOffset, true);
}

struct ggml_tensor *ggml_set_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, size_t pNb1,
                                size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pNb1, pA->nb[2], pA->nb[3], pOffset, false);
}

struct ggml_tensor *ggml_set_2d_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        size_t pNb1, size_t pOffset) {
  return fGgmlSetImpl(pCtx, pA, pB, pNb1, pA->nb[2], pA->nb[3], pOffset, true);
}

// ggml_cpy

static struct ggml_tensor *fGgmlCpyImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_nelements(pA) == ggml_nelements(pB));

  // make a view of the destination
  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pB);
  if (strlen(pB->name) > 0) {
    ggml_format_name(vResult, "%s (copy of %s)", pB->name, pA->name);
  } else {
    ggml_format_name(vResult, "%s (copy)", pA->name);
  }

  vResult->op = GGML_OP_CPY;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_cpy(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return fGgmlCpyImpl(pCtx, pA, pB);
}

struct ggml_tensor *ggml_cast(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_type pType) {
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pType, GGML_MAX_DIMS, pA->ne);
  ggml_format_name(vResult, "%s (copy)", pA->name);

  vResult->op = GGML_OP_CPY;
  vResult->src[0] = pA;
  vResult->src[1] = vResult; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

  return vResult;
}

// ggml_cont

static struct ggml_tensor *fGgmlContImpl(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);
  ggml_format_name(vResult, "%s (cont)", pA->name);

  vResult->op = GGML_OP_CONT;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_cont(struct ggml_context *pCtx, struct ggml_tensor *pA) { return fGgmlContImpl(pCtx, pA); }

// make contiguous, with new shape
GGML_API struct ggml_tensor *ggml_cont_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0) {
  return ggml_cont_4d(pCtx, pA, pNe0, 1, 1, 1);
}

GGML_API struct ggml_tensor *ggml_cont_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0,
                                          int64_t pNe1) {
  return ggml_cont_4d(pCtx, pA, pNe0, pNe1, 1, 1);
}

GGML_API struct ggml_tensor *ggml_cont_3d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                          int64_t pNe2) {
  return ggml_cont_4d(pCtx, pA, pNe0, pNe1, pNe2, 1);
}

struct ggml_tensor *ggml_cont_4d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                 int64_t pNe2, int64_t pNe3) {
  GGML_ASSERT(ggml_nelements(pA) == (pNe0 * pNe1 * pNe2 * pNe3));

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, pA->type, pNe0, pNe1, pNe2, pNe3);
  ggml_format_name(vResult, "%s (cont)", pA->name);

  vResult->op = GGML_OP_CONT;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_reshape

struct ggml_tensor *ggml_reshape(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_is_contiguous(pA));
  // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
  GGML_ASSERT(ggml_nelements(pA) == ggml_nelements(pB));

  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, GGML_MAX_DIMS, pB->ne, pA, 0);
  ggml_format_name(vResult, "%s (reshaped)", pA->name);

  vResult->op = GGML_OP_RESHAPE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_reshape_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0) {
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_nelements(pA) == pNe0);

  const int64_t cNe[1] = {pNe0};
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, 1, cNe, pA, 0);
  ggml_format_name(vResult, "%s (reshaped)", pA->name);

  vResult->op = GGML_OP_RESHAPE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_reshape_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1) {
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_nelements(pA) == pNe0 * pNe1);

  const int64_t cNe[2] = {pNe0, pNe1};
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, 2, cNe, pA, 0);
  ggml_format_name(vResult, "%s (reshaped)", pA->name);

  vResult->op = GGML_OP_RESHAPE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_reshape_3d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                    int64_t pNe2) {
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_nelements(pA) == pNe0 * pNe1 * pNe2);

  const int64_t cNe[3] = {pNe0, pNe1, pNe2};
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, 3, cNe, pA, 0);
  ggml_format_name(vResult, "%s (reshaped)", pA->name);

  vResult->op = GGML_OP_RESHAPE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_reshape_4d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                    int64_t pNe2, int64_t pNe3) {
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_nelements(pA) == pNe0 * pNe1 * pNe2 * pNe3);

  const int64_t cNe[4] = {pNe0, pNe1, pNe2, pNe3};
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, 4, cNe, pA, 0);
  ggml_format_name(vResult, "%s (reshaped)", pA->name);

  vResult->op = GGML_OP_RESHAPE;
  vResult->src[0] = pA;

  return vResult;
}

static struct ggml_tensor *fGgmlViewImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNDims,
                                         const int64_t *pNe, size_t pOffset) {
  struct ggml_tensor *vResult = fGgmlNewTensorImpl(pCtx, pA->type, pNDims, pNe, pA, pOffset);
  ggml_format_name(vResult, "%s (view)", pA->name);

  ggml_set_op_params(vResult, &pOffset, sizeof(pOffset));

  vResult->op = GGML_OP_VIEW;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_view_1d

struct ggml_tensor *ggml_view_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, size_t pOffset) {
  struct ggml_tensor *vResult = fGgmlViewImpl(pCtx, pA, 1, &pNe0, pOffset);

  return vResult;
}

// ggml_view_2d

struct ggml_tensor *ggml_view_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                 size_t pNb1, size_t pOffset) {
  const int64_t cNe[2] = {pNe0, pNe1};

  struct ggml_tensor *vResult = fGgmlViewImpl(pCtx, pA, 2, cNe, pOffset);

  vResult->nb[1] = pNb1;
  vResult->nb[2] = vResult->nb[1] * pNe1;
  vResult->nb[3] = vResult->nb[2];

  return vResult;
}

// ggml_view_3d

struct ggml_tensor *ggml_view_3d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                 int64_t pNe2, size_t pNb1, size_t pNb2, size_t pOffset) {
  const int64_t cNe[3] = {pNe0, pNe1, pNe2};

  struct ggml_tensor *vResult = fGgmlViewImpl(pCtx, pA, 3, cNe, pOffset);

  vResult->nb[1] = pNb1;
  vResult->nb[2] = pNb2;
  vResult->nb[3] = vResult->nb[2] * pNe2;

  return vResult;
}

// ggml_view_4d

struct ggml_tensor *ggml_view_4d(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                 int64_t pNe2, int64_t pNe3, size_t pNb1, size_t pNb2, size_t pNb3, size_t pOffset) {
  const int64_t cNe[4] = {pNe0, pNe1, pNe2, pNe3};

  struct ggml_tensor *vResult = fGgmlViewImpl(pCtx, pA, 4, cNe, pOffset);

  vResult->nb[1] = pNb1;
  vResult->nb[2] = pNb2;
  vResult->nb[3] = pNb3;

  return vResult;
}

// ggml_permute

struct ggml_tensor *ggml_permute(struct ggml_context *pCtx, struct ggml_tensor *pA, int pAxis0, int pAxis1, int pAxis2,
                                 int pAxis3) {
  GGML_ASSERT(pAxis0 >= 0 && pAxis0 < GGML_MAX_DIMS);
  GGML_ASSERT(pAxis1 >= 0 && pAxis1 < GGML_MAX_DIMS);
  GGML_ASSERT(pAxis2 >= 0 && pAxis2 < GGML_MAX_DIMS);
  GGML_ASSERT(pAxis3 >= 0 && pAxis3 < GGML_MAX_DIMS);

  GGML_ASSERT(pAxis0 != pAxis1);
  GGML_ASSERT(pAxis0 != pAxis2);
  GGML_ASSERT(pAxis0 != pAxis3);
  GGML_ASSERT(pAxis1 != pAxis2);
  GGML_ASSERT(pAxis1 != pAxis3);
  GGML_ASSERT(pAxis2 != pAxis3);

  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);
  ggml_format_name(vResult, "%s (permuted)", pA->name);

  int aNe[GGML_MAX_DIMS];
  int aNb[GGML_MAX_DIMS];

  aNe[pAxis0] = pA->ne[0];
  aNe[pAxis1] = pA->ne[1];
  aNe[pAxis2] = pA->ne[2];
  aNe[pAxis3] = pA->ne[3];

  aNb[pAxis0] = pA->nb[0];
  aNb[pAxis1] = pA->nb[1];
  aNb[pAxis2] = pA->nb[2];
  aNb[pAxis3] = pA->nb[3];

  vResult->ne[0] = aNe[0];
  vResult->ne[1] = aNe[1];
  vResult->ne[2] = aNe[2];
  vResult->ne[3] = aNe[3];

  vResult->nb[0] = aNb[0];
  vResult->nb[1] = aNb[1];
  vResult->nb[2] = aNb[2];
  vResult->nb[3] = aNb[3];

  vResult->op = GGML_OP_PERMUTE;
  vResult->src[0] = pA;

  int32_t aParams[] = {pAxis0, pAxis1, pAxis2, pAxis3};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  return vResult;
}

// ggml_transpose

struct ggml_tensor *ggml_transpose(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);
  ggml_format_name(vResult, "%s (transposed)", pA->name);

  vResult->ne[0] = pA->ne[1];
  vResult->ne[1] = pA->ne[0];

  vResult->nb[0] = pA->nb[1];
  vResult->nb[1] = pA->nb[0];

  vResult->op = GGML_OP_TRANSPOSE;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_get_rows

struct ggml_tensor *ggml_get_rows(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(pA->ne[2] == pB->ne[1]);
  GGML_ASSERT(pA->ne[3] == pB->ne[2]);
  GGML_ASSERT(pB->ne[3] == 1);
  GGML_ASSERT(pB->type == GGML_TYPE_I32);

  // TODO: implement non F32 return
  enum ggml_type vType = GGML_TYPE_F32;
  if (pA->type == GGML_TYPE_I32) {
    vType = pA->type;
  }
  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, vType, pA->ne[0], pB->ne[0], pB->ne[1], pB->ne[2]);

  vResult->op = GGML_OP_GET_ROWS;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_get_rows_back

struct ggml_tensor *ggml_get_rows_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                       struct ggml_tensor *pC) {
  GGML_ASSERT(ggml_is_matrix(pA) && ggml_is_vector(pB) && pB->type == GGML_TYPE_I32);
  GGML_ASSERT(ggml_is_matrix(pC) && (pA->ne[0] == pC->ne[0]));

  // TODO: implement non F32 return
  // struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ggml_tensor *vResult = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, pC->ne[0], pC->ne[1]);

  vResult->op = GGML_OP_GET_ROWS_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_set_rows

struct ggml_tensor *ggml_set_rows(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                  struct ggml_tensor *pC) {
  GGML_ASSERT(pA->ne[0] == pB->ne[0]);
  GGML_ASSERT(pA->ne[2] == pB->ne[2]);
  GGML_ASSERT(pA->ne[3] == pB->ne[3]);
  GGML_ASSERT(pB->ne[1] == pC->ne[0]);
  GGML_ASSERT(pB->ne[2] % pC->ne[1] == 0);
  GGML_ASSERT(pB->ne[3] % pC->ne[2] == 0);
  GGML_ASSERT(pC->ne[3] == 1);
  GGML_ASSERT(pB->type == GGML_TYPE_F32 || pB->type == GGML_TYPE_F16);
  GGML_ASSERT(pC->type == GGML_TYPE_I64 || pC->type == GGML_TYPE_I32);

  GGML_ASSERT(ggml_is_contiguous_rows(pA));
  GGML_ASSERT(ggml_is_contiguous_rows(pB));

  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);

  vResult->op = GGML_OP_SET_ROWS;
  vResult->src[0] = pB;
  vResult->src[1] = pC;
  vResult->src[2] = pA; // note: order is weird due to legacy reasons

  return vResult;
}

// ggml_diag

struct ggml_tensor *ggml_diag(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  GGML_ASSERT(pA->ne[1] == 1);

  const int64_t cNe[4] = {pA->ne[0], pA->ne[0], pA->ne[2], pA->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, 4, cNe);

  vResult->op = GGML_OP_DIAG;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_diag_mask_inf

static struct ggml_tensor *fGgmlDiagMaskInfImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast,
                                                bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  int32_t aParams[] = {pNPast};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_DIAG_MASK_INF;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_diag_mask_inf(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast) {
  return fGgmlDiagMaskInfImpl(pCtx, pA, pNPast, false);
}

struct ggml_tensor *ggml_diag_mask_inf_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast) {
  return fGgmlDiagMaskInfImpl(pCtx, pA, pNPast, true);
}

// ggml_diag_mask_zero

static struct ggml_tensor *fGgmlDiagMaskZeroImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast,
                                                 bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  int32_t aParams[] = {pNPast};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_DIAG_MASK_ZERO;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_diag_mask_zero(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast) {
  return fGgmlDiagMaskZeroImpl(pCtx, pA, pNPast, false);
}

struct ggml_tensor *ggml_diag_mask_zero_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, int pNPast) {
  return fGgmlDiagMaskZeroImpl(pCtx, pA, pNPast, true);
}

// ggml_soft_max

static struct ggml_tensor *fGgmlSoftMaxImpl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                            struct ggml_tensor *pMask, float pScale, float pMaxBias, bool pInplace) {
  GGML_ASSERT(ggml_is_contiguous(pA));

  if (pMask) {
    GGML_ASSERT(pMask->type == GGML_TYPE_F16 || pMask->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(pMask));
    GGML_ASSERT(pMask->ne[0] == pA->ne[0]);
    GGML_ASSERT(pMask->ne[1] >= pA->ne[1]);
    GGML_ASSERT(pA->ne[2] % pMask->ne[2] == 0);
    GGML_ASSERT(pA->ne[3] % pMask->ne[3] == 0);
  }

  if (pMaxBias > 0.0f) {
    GGML_ASSERT(pMask);
  }

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  float aParams[] = {pScale, pMaxBias};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_SOFT_MAX;
  vResult->src[0] = pA;
  vResult->src[1] = pMask;

  return vResult;
}

struct ggml_tensor *ggml_soft_max(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSoftMaxImpl(pCtx, pA, NULL, 1.0f, 0.0f, false);
}

struct ggml_tensor *ggml_soft_max_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA) {
  return fGgmlSoftMaxImpl(pCtx, pA, NULL, 1.0f, 0.0f, true);
}

struct ggml_tensor *ggml_soft_max_ext(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pMask,
                                      float pScale, float pMaxBias) {
  return fGgmlSoftMaxImpl(pCtx, pA, pMask, pScale, pMaxBias, false);
}

struct ggml_tensor *ggml_soft_max_ext_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                              struct ggml_tensor *pMask, float pScale, float pMaxBias) {
  return fGgmlSoftMaxImpl(pCtx, pA, pMask, pScale, pMaxBias, true);
}

void ggml_soft_max_add_sinks(struct ggml_tensor *pA, struct ggml_tensor *pSinks) {
  if (!pSinks) {
    pA->src[2] = NULL;
    return;
  }

  GGML_ASSERT(pA->op == GGML_OP_SOFT_MAX);
  GGML_ASSERT(pA->src[2] == NULL);
  GGML_ASSERT(pA->src[0]->ne[2] == pSinks->ne[0]);
  GGML_ASSERT(pSinks->type == GGML_TYPE_F32);

  pA->src[2] = pSinks;
}

// ggml_soft_max_ext_back

static struct ggml_tensor *fGgmlSoftMaxExtBackImpl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                                   struct ggml_tensor *pB, float pScale, float pMaxBias,
                                                   bool pInplace) {
  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  vResult->op = GGML_OP_SOFT_MAX_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  memcpy((float *)vResult->op_params + 0, &pScale, sizeof(float));
  memcpy((float *)vResult->op_params + 1, &pMaxBias, sizeof(float));

  return vResult;
}

struct ggml_tensor *ggml_soft_max_ext_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                           float pScale, float pMaxBias) {
  return fGgmlSoftMaxExtBackImpl(pCtx, pA, pB, pScale, pMaxBias, false);
}

struct ggml_tensor *ggml_soft_max_ext_back_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                                   struct ggml_tensor *pB, float pScale, float pMaxBias) {
  return fGgmlSoftMaxExtBackImpl(pCtx, pA, pB, pScale, pMaxBias, true);
}

// ggml_rope

static struct ggml_tensor *fGgmlRopeImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                         struct ggml_tensor *pC, int pNDims, int pSections[GGML_MROPE_SECTIONS],
                                         int pMode, int pNCtxOrig, float pFreqBase, float pFreqScale, float pExtFactor,
                                         float pAttnFactor, float pBetaFast, float pBetaSlow, bool pInplace) {
  GGML_ASSERT((pMode & 1) == 0 && "mode & 1 == 1 is no longer supported");

  GGML_ASSERT(ggml_is_vector(pB));
  GGML_ASSERT(pB->type == GGML_TYPE_I32);

  bool vMropeUsed = pMode & GGML_ROPE_TYPE_MROPE;
  if (vMropeUsed) {
    GGML_ASSERT(pA->ne[2] * 4 == pB->ne[0]); // mrope expecting 4 position ids per token
  } else {
    GGML_ASSERT(pA->ne[2] == pB->ne[0]);
  }

  if (pC) {
    GGML_ASSERT(pC->type == GGML_TYPE_F32);
    GGML_ASSERT(pC->ne[0] >= pNDims / 2);
  }

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  int32_t aParams[15] = {/*n_past*/ 0, pNDims, pMode, /*n_ctx*/ 0, pNCtxOrig};
  memcpy(aParams + 5, &pFreqBase, sizeof(float));
  memcpy(aParams + 6, &pFreqScale, sizeof(float));
  memcpy(aParams + 7, &pExtFactor, sizeof(float));
  memcpy(aParams + 8, &pAttnFactor, sizeof(float));
  memcpy(aParams + 9, &pBetaFast, sizeof(float));
  memcpy(aParams + 10, &pBetaSlow, sizeof(float));
  if (vMropeUsed && pSections) {
    memcpy(aParams + 11, pSections, sizeof(int32_t) * GGML_MROPE_SECTIONS);
  } else {
    memset(aParams + 11, 0, sizeof(int32_t) * GGML_MROPE_SECTIONS);
  }
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_ROPE;
  vResult->src[0] = pA;
  vResult->src[1] = pB;
  vResult->src[2] = pC;

  return vResult;
}

struct ggml_tensor *ggml_rope(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pNDims,
                              int pMode) {
  return fGgmlRopeImpl(pCtx, pA, pB, NULL, pNDims, NULL, pMode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false);
}

struct ggml_tensor *ggml_rope_multi(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                    struct ggml_tensor *pC, int pNDims, int pSections[GGML_MROPE_SECTIONS], int pMode,
                                    int pNCtxOrig, float pFreqBase, float pFreqScale, float pExtFactor,
                                    float pAttnFactor, float pBetaFast, float pBetaSlow) {
  return fGgmlRopeImpl(pCtx, pA, pB, pC, pNDims, pSections, pMode, pNCtxOrig, pFreqBase, pFreqScale, pExtFactor,
                       pAttnFactor, pBetaFast, pBetaSlow, false);
}

struct ggml_tensor *ggml_rope_multi_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                            struct ggml_tensor *pC, int pNDims, int pSections[GGML_MROPE_SECTIONS],
                                            int pMode, int pNCtxOrig, float pFreqBase, float pFreqScale,
                                            float pExtFactor, float pAttnFactor, float pBetaFast, float pBetaSlow) {
  return fGgmlRopeImpl(pCtx, pA, pB, pC, pNDims, pSections, pMode, pNCtxOrig, pFreqBase, pFreqScale, pExtFactor,
                       pAttnFactor, pBetaFast, pBetaSlow, true);
}

struct ggml_tensor *ggml_rope_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                      int pNDims, int pMode) {
  return fGgmlRopeImpl(pCtx, pA, pB, NULL, pNDims, NULL, pMode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true);
}

struct ggml_tensor *ggml_rope_ext(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                  struct ggml_tensor *pC, int pNDims, int pMode, int pNCtxOrig, float pFreqBase,
                                  float pFreqScale, float pExtFactor, float pAttnFactor, float pBetaFast,
                                  float pBetaSlow) {
  return fGgmlRopeImpl(pCtx, pA, pB, pC, pNDims, NULL, pMode, pNCtxOrig, pFreqBase, pFreqScale, pExtFactor, pAttnFactor,
                       pBetaFast, pBetaSlow, false);
}

struct ggml_tensor *ggml_rope_ext_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                          struct ggml_tensor *pC, int pNDims, int pMode, int pNCtxOrig, float pFreqBase,
                                          float pFreqScale, float pExtFactor, float pAttnFactor, float pBetaFast,
                                          float pBetaSlow) {
  return fGgmlRopeImpl(pCtx, pA, pB, pC, pNDims, NULL, pMode, pNCtxOrig, pFreqBase, pFreqScale, pExtFactor, pAttnFactor,
                       pBetaFast, pBetaSlow, true);
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float fGgmlRopeYarnCorrDim(int pNDims, int pNCtxOrig, float pNRot, float pBase) {
  return pNDims * logf(pNCtxOrig / (pNRot * 2 * (float)M_PI)) / (2 * logf(pBase));
}

void ggml_rope_yarn_corr_dims(int pNDims, int pNCtxOrig, float pFreqBase, float pBetaFast, float pBetaSlow,
                              float pDims[2]) {
  // start and end correction dims
  float vStart = floorf(fGgmlRopeYarnCorrDim(pNDims, pNCtxOrig, pBetaFast, pFreqBase));
  float vEnd = ceilf(fGgmlRopeYarnCorrDim(pNDims, pNCtxOrig, pBetaSlow, pFreqBase));
  pDims[0] = MAX(0, vStart);
  pDims[1] = MIN(pNDims - 1, vEnd);
}

// ggml_rope_back

struct ggml_tensor *ggml_rope_ext_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                       struct ggml_tensor *pC, int pNDims, int pMode, int pNCtxOrig, float pFreqBase,
                                       float pFreqScale, float pExtFactor, float pAttnFactor, float pBetaFast,
                                       float pBetaSlow) {
  struct ggml_tensor *vResult = ggml_rope_ext(pCtx, pA, pB, pC, pNDims, pMode, pNCtxOrig, pFreqBase, pFreqScale,
                                              pExtFactor, pAttnFactor, pBetaFast, pBetaSlow);
  vResult->op = GGML_OP_ROPE_BACK;
  return vResult;
}

struct ggml_tensor *ggml_rope_multi_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                         struct ggml_tensor *pC, int pNDims, int pSections[4], int pMode, int pNCtxOrig,
                                         float pFreqBase, float pFreqScale, float pExtFactor, float pAttnFactor,
                                         float pBetaFast, float pBetaSlow) {
  struct ggml_tensor *vResult = ggml_rope_multi(pCtx, pA, pB, pC, pNDims, pSections, pMode, pNCtxOrig, pFreqBase,
                                                pFreqScale, pExtFactor, pAttnFactor, pBetaFast, pBetaSlow);
  vResult->op = GGML_OP_ROPE_BACK;
  return vResult;
}
// ggml_clamp

struct ggml_tensor *ggml_clamp(struct ggml_context *pCtx, struct ggml_tensor *pA, float pMin, float pMax) {
  // TODO: when implement backward, fix this:
  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);

  float aParams[] = {pMin, pMax};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_CLAMP;
  vResult->src[0] = pA;

  return vResult;
}

static int64_t fGgmlCalcConvOutputSize(int64_t pIns, int64_t pKs, int pS, int p_, int pD) {
  return (pIns + 2 * p_ - pD * (pKs - 1) - 1) / pS + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct ggml_tensor *ggml_im2col(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS0,
                                int pS1, int p0, int p1, int pD0, int pD1, bool pIs2D, enum ggml_type pDstType) {
  if (pIs2D) {
    GGML_ASSERT(pA->ne[2] == pB->ne[2]);
  } else {
    // GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
    GGML_ASSERT(pB->ne[1] == pA->ne[1]);
    GGML_ASSERT(pB->ne[3] == 1);
  }

  const int64_t cOh = pIs2D ? fGgmlCalcConvOutputSize(pB->ne[1], pA->ne[1], pS1, p1, pD1) : 0;
  const int64_t cOw = fGgmlCalcConvOutputSize(pB->ne[0], pA->ne[0], pS0, p0, pD0);

  GGML_ASSERT((!pIs2D || cOh > 0) && "b too small compared to a");
  GGML_ASSERT((cOw > 0) && "b too small compared to a");

  const int64_t cNe[4] = {
      pIs2D ? (pA->ne[2] * pA->ne[1] * pA->ne[0]) : pA->ne[1] * pA->ne[0],
      cOw,
      pIs2D ? cOh : pB->ne[2],
      pIs2D ? pB->ne[3] : 1,
  };

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pDstType, 4, cNe);
  int32_t aParams[] = {pS0, pS1, p0, p1, pD0, pD1, (pIs2D ? 1 : 0)};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_IM2COL;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_im2col_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                     int64_t *pNe, int pS0, int pS1, int p0, int p1, int pD0, int pD1, bool pIs2D) {
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, pNe);
  int32_t aParams[] = {pS0, pS1, p0, p1, pD0, pD1, (pIs2D ? 1 : 0)};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_IM2COL_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_conv_1d

struct ggml_tensor *ggml_conv_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS0,
                                 int p0, int pD0) {
  struct ggml_tensor *vIm2col =
      ggml_im2col(pCtx, pA, pB, pS0, 0, p0, 0, pD0, 0, false,
                  pA->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

  struct ggml_tensor *vResult =
      ggml_mul_mat(pCtx,
                   ggml_reshape_2d(pCtx, vIm2col, vIm2col->ne[0],
                                   (vIm2col->ne[2] * vIm2col->ne[1])),             // [N, OL, IC * K] => [N*OL, IC * K]
                   ggml_reshape_2d(pCtx, pA, (pA->ne[0] * pA->ne[1]), pA->ne[2])); // [OC，IC, K] => [OC, IC * K]

  vResult = ggml_reshape_3d(pCtx, vResult, vIm2col->ne[1], pA->ne[2], vIm2col->ne[2]); // [N, OC, OL]

  return vResult;
}

// ggml_conv_1d_ph

struct ggml_tensor *ggml_conv_1d_ph(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS,
                                    int pD) {
  return ggml_conv_1d(pCtx, pA, pB, pS, pA->ne[0] / 2, pD);
}

// ggml_conv_1d_dw

struct ggml_tensor *ggml_conv_1d_dw(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS0,
                                    int p0, int pD0) {
  struct ggml_tensor *vNewB = ggml_reshape_4d(pCtx, pB, pB->ne[0], 1, pB->ne[1], pB->ne[2]);

  struct ggml_tensor *vIm2col = ggml_im2col(pCtx, pA, vNewB, pS0, 0, p0, 0, pD0, 0, false,
                                            pA->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

  struct ggml_tensor *vResult = ggml_mul_mat(pCtx, vIm2col, pA);

  vResult = ggml_reshape_3d(pCtx, vResult, vResult->ne[0], vResult->ne[2], 1);

  return vResult;
}

// ggml_conv_1d_dw_ph

struct ggml_tensor *ggml_conv_1d_dw_ph(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                       int pS0, int pD0) {
  return ggml_conv_1d_dw(pCtx, pA, pB, pS0, pA->ne[0] / 2, pD0);
}

// ggml_col2im_1d

struct ggml_tensor *ggml_col2im_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, int pS0, int pOc, int p0) {
  GGML_ASSERT(ggml_is_matrix(pA));
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(pA->type == GGML_TYPE_F32 || pA->type == GGML_TYPE_F16 || pA->type == GGML_TYPE_BF16);
  GGML_ASSERT(pS0 > 0);
  GGML_ASSERT(pOc > 0);
  GGML_ASSERT(p0 >= 0);

  const int64_t cKOc = pA->ne[0];
  const int64_t cTIn = pA->ne[1];
  const int64_t cK = cKOc / pOc;
  const int64_t cTOut = (cTIn - 1) * pS0 + cK - 2 * p0;

  GGML_ASSERT(cKOc == cK * pOc); // a->ne[0] must be a whole number of oc blocks
  GGML_ASSERT(cK > 0 && cTOut > 0);

  const int64_t cNe[4] = {cTOut, pOc, 1, 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pA->type, 2, cNe);

  int32_t aParams[] = {pS0, (int32_t)pOc, (int32_t)p0};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_COL2IM_1D;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_conv_transpose_1d

static int64_t fGgmlCalcConvTranspose1dOutputSize(int64_t pIns, int64_t pKs, int pS, int p_, int pD) {
  return (pIns - 1) * pS - 2 * p_ + pD * (pKs - 1) + 1;
}

GGML_API struct ggml_tensor *ggml_conv_transpose_1d(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                                    struct ggml_tensor *pB, int pS0, int p0, int pD0) {
  GGML_ASSERT(ggml_is_matrix(pB));
  GGML_ASSERT(pA->ne[2] == pB->ne[1]);
  GGML_ASSERT(pA->ne[3] == 1);

  GGML_ASSERT(p0 == 0);
  GGML_ASSERT(pD0 == 1);

  const int64_t cNe[4] = {
      fGgmlCalcConvTranspose1dOutputSize(pB->ne[0], pA->ne[0], pS0, 0 /*p0*/, 1 /*d0*/),
      pA->ne[1],
      pB->ne[2],
      1,
  };
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  int32_t aParams[] = {pS0, p0, pD0};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_CONV_TRANSPOSE_1D;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
struct ggml_tensor *ggml_conv_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS0,
                                 int pS1, int p0, int p1, int pD0, int pD1) {
  struct ggml_tensor *vIm2col =
      ggml_im2col(pCtx, pA, pB, pS0, pS1, p0, p1, pD0, pD1, true,
                  pA->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

  struct ggml_tensor *vResult =
      ggml_mul_mat(pCtx,
                   ggml_reshape_2d(pCtx, vIm2col, vIm2col->ne[0],
                                   vIm2col->ne[3] * vIm2col->ne[2] *
                                       vIm2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                   ggml_reshape_2d(pCtx, pA, (pA->ne[0] * pA->ne[1] * pA->ne[2]),
                                   pA->ne[3])); // [OC，IC, KH, KW] => [OC, IC * KH * KW]

  vResult =
      ggml_reshape_4d(pCtx, vResult, vIm2col->ne[1], vIm2col->ne[2], vIm2col->ne[3], pA->ne[3]); // [OC, N, OH, OW]
  vResult = ggml_cont(pCtx, ggml_permute(pCtx, vResult, 0, 1, 3, 2));                            // [N, OC, OH, OW]

  return vResult;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
struct ggml_tensor *ggml_im2col_3d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                   int64_t pIc,
                                   int pS0, // stride width
                                   int pS1, // stride height
                                   int pS2, // stride depth
                                   int p0,  // padding width
                                   int p1,  // padding height
                                   int p2,  // padding depth
                                   int pD0, // dilation width
                                   int pD1, // dilation height
                                   int pD2, // dilation depth
                                   enum ggml_type pDstType) {
  const int64_t cN = pB->ne[3] / pIc;
  const int64_t cId = pB->ne[2];
  const int64_t cIh = pB->ne[1];
  const int64_t cIw = pB->ne[0];

  const int64_t cOc = pA->ne[3] / pIc;
  UNUSED(cOc);
  const int64_t cKd = pA->ne[2];
  const int64_t cKh = pA->ne[1];
  const int64_t cKw = pA->ne[0];
  const int64_t cOd = fGgmlCalcConvOutputSize(cId, cKd, pS2, p2, pD2);
  const int64_t cOh = fGgmlCalcConvOutputSize(cIh, cKh, pS1, p1, pD1);
  const int64_t cOw = fGgmlCalcConvOutputSize(cIw, cKw, pS0, p0, pD0);

  GGML_ASSERT((cOd > 0) && "b too small compared to a");
  GGML_ASSERT((cOh > 0) && "b too small compared to a");
  GGML_ASSERT((cOw > 0) && "b too small compared to a");

  const int64_t cNe[4] = {cKw * cKh * cKd * pIc, cOw, cOh, cOd * cN};

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pDstType, 4, cNe);
  int32_t aParams[] = {pS0, pS1, pS2, p0, p1, p2, pD0, pD1, pD2, (int32_t)pIc};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_IM2COL_3D;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
struct ggml_tensor *ggml_conv_3d(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int64_t pIc,
                                 int pS0, // stride width
                                 int pS1, // stride height
                                 int pS2, // stride depth
                                 int p0,  // padding width
                                 int p1,  // padding height
                                 int p2,  // padding depth
                                 int pD0, // dilation width
                                 int pD1, // dilation height
                                 int pD2  // dilation depth
) {
  struct ggml_tensor *vIm2col =
      ggml_im2col_3d(pCtx, pA, pB, pIc, pS0, pS1, pS2, p0, p1, p2, pD0, pD1, pD2,
                     pA->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

  int64_t vOc = pA->ne[3] / pIc;
  int64_t vN = pB->ne[3] / pIc;
  struct ggml_tensor *vResult = ggml_mul_mat(
      pCtx,
      ggml_reshape_2d(pCtx, vIm2col, vIm2col->ne[0],
                      vIm2col->ne[3] * vIm2col->ne[2] *
                          vIm2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
      ggml_reshape_2d(pCtx, pA, (pA->ne[0] * pA->ne[1] * pA->ne[2] * pIc),
                      vOc)); // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

  int64_t vOd = vIm2col->ne[3] / vN;
  vResult = ggml_reshape_4d(pCtx, vResult, vIm2col->ne[1] * vIm2col->ne[2], vOd, vN,
                            vOc);                                     // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
  vResult = ggml_cont(pCtx, ggml_permute(pCtx, vResult, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
  vResult = ggml_reshape_4d(pCtx, vResult, vIm2col->ne[1], vIm2col->ne[2], vOd, vOc * vN); // [N*OC, OD, OH, OW]

  return vResult;
}

// ggml_conv_2d_sk_p0

struct ggml_tensor *ggml_conv_2d_sk_p0(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return ggml_conv_2d(pCtx, pA, pB, pA->ne[0], pA->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

struct ggml_tensor *ggml_conv_2d_s1_ph(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  return ggml_conv_2d(pCtx, pA, pB, 1, 1, pA->ne[0] / 2, pA->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

struct ggml_tensor *ggml_conv_2d_dw(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB, int pS0,
                                    int pS1, int p0, int p1, int pD0, int pD1) {
  struct ggml_tensor *vNewA = ggml_reshape_4d(pCtx, pA, pA->ne[0], pA->ne[1], 1, pA->ne[2] * pA->ne[3]);
  struct ggml_tensor *vIm2col = ggml_im2col(
      pCtx, vNewA, ggml_reshape_4d(pCtx, pB, pB->ne[0], pB->ne[1], 1, pB->ne[2] * pB->ne[3]), pS0, pS1, p0, p1, pD0,
      pD1, true, pA->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
  struct ggml_tensor *vNewB = ggml_reshape_4d(pCtx, vIm2col, vIm2col->ne[0], vIm2col->ne[2] * vIm2col->ne[1], pB->ne[2],
                                              pB->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

  vNewA = ggml_reshape_4d(pCtx, vNewA, (vNewA->ne[0] * vNewA->ne[1]), vNewA->ne[2], vNewA->ne[3],
                          1); // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
  struct ggml_tensor *vResult = ggml_mul_mat(pCtx, vNewA, vNewB);
  vResult = ggml_reshape_4d(pCtx, vResult, vIm2col->ne[1], vIm2col->ne[2], pB->ne[2], pB->ne[3]); // [N, OC, OH, OW]

  return vResult;
}

// ggml_conv_2d_dw_direct

struct ggml_tensor *ggml_conv_2d_dw_direct(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                           int pStride0, int pStride1, int pPad0, int pPad1, int pDilation0,
                                           int pDilation1) {
  GGML_ASSERT(pA->ne[2] == 1);
  GGML_ASSERT(pA->ne[3] == pB->ne[2]);
  int64_t aNe[4];
  aNe[0] = fGgmlCalcConvOutputSize(pB->ne[0], pA->ne[0], pStride0, pPad0, pDilation0);
  aNe[1] = fGgmlCalcConvOutputSize(pB->ne[1], pA->ne[1], pStride1, pPad1, pDilation1);
  aNe[2] = pB->ne[2];
  aNe[3] = pB->ne[3];

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pB->type, 4, aNe);

  if (ggml_is_contiguous_channels(pB)) {
    // Result will be permuted the same way as input (CWHN order)
    const int64_t cTypeSize = ggml_type_size(vResult->type);
    GGML_ASSERT(ggml_blck_size(vResult->type) == 1);
    vResult->nb[0] = vResult->ne[2] * cTypeSize;
    vResult->nb[1] = vResult->ne[0] * vResult->nb[0];
    vResult->nb[2] = cTypeSize;
  }

  int32_t aParams[] = {pStride0, pStride1, pPad0, pPad1, pDilation0, pDilation1};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_CONV_2D_DW;
  vResult->src[0] = pA;
  vResult->src[1] = pB;
  return vResult;
}

// ggml_conv_2d_direct

struct ggml_tensor *ggml_conv_2d_direct(struct ggml_context *pCtx,
                                        struct ggml_tensor *pA, // convolution kernel [KW, KH, IC, OC]
                                        struct ggml_tensor *pB, // input data [W, H, C, N]
                                        int pS0,                // stride dimension 0
                                        int pS1,                // stride dimension 1
                                        int p0,                 // padding dimension 0
                                        int p1,                 // padding dimension 1
                                        int pD0,                // dilation dimension 0
                                        int pD1) {              // dilation dimension 1

  GGML_ASSERT(pA->ne[2] == pB->ne[2]);
  // GGML_ASSERT(a->type == b->type);

  int64_t aNe[4];
  aNe[0] = fGgmlCalcConvOutputSize(pB->ne[0], pA->ne[0], pS0, p0, pD0);
  aNe[1] = fGgmlCalcConvOutputSize(pB->ne[1], pA->ne[1], pS1, p1, pD1);
  aNe[2] = pA->ne[3];
  aNe[3] = pB->ne[3];

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, pB->type, 4, aNe);

  ggml_set_op_params_i32(vResult, 0, pS0);
  ggml_set_op_params_i32(vResult, 1, pS1);
  ggml_set_op_params_i32(vResult, 2, p0);
  ggml_set_op_params_i32(vResult, 3, p1);
  ggml_set_op_params_i32(vResult, 4, pD0);
  ggml_set_op_params_i32(vResult, 5, pD1);

  vResult->op = GGML_OP_CONV_2D;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_conv_3d_direct

struct ggml_tensor *ggml_conv_3d_direct(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                        int pS0, int pS1, int pS2, int p0, int p1, int p2, int pD0, int pD1, int pD2,
                                        int pC, int pN, int pOc) {

  GGML_ASSERT(pA->ne[3] == (int64_t)pC * pOc);
  GGML_ASSERT(pB->ne[3] == (int64_t)pC * pN);

  int64_t aNe[4];
  aNe[0] = fGgmlCalcConvOutputSize(pB->ne[0], pA->ne[0], pS0, p0, pD0);
  aNe[1] = fGgmlCalcConvOutputSize(pB->ne[1], pA->ne[1], pS1, p1, pD1);
  aNe[2] = fGgmlCalcConvOutputSize(pB->ne[2], pA->ne[2], pS2, p2, pD2);
  aNe[3] = (int64_t)pOc * pN;

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, aNe);

  ggml_set_op_params_i32(vResult, 0, pS0);
  ggml_set_op_params_i32(vResult, 1, pS1);
  ggml_set_op_params_i32(vResult, 2, pS2);
  ggml_set_op_params_i32(vResult, 3, p0);
  ggml_set_op_params_i32(vResult, 4, p1);
  ggml_set_op_params_i32(vResult, 5, p2);
  ggml_set_op_params_i32(vResult, 6, pD0);
  ggml_set_op_params_i32(vResult, 7, pD1);
  ggml_set_op_params_i32(vResult, 8, pD2);
  ggml_set_op_params_i32(vResult, 9, pC);
  ggml_set_op_params_i32(vResult, 10, pN);
  ggml_set_op_params_i32(vResult, 11, pOc);

  vResult->op = GGML_OP_CONV_3D;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_conv_transpose_2d_p0

static int64_t fGgmlCalcConvTransposeOutputSize(int64_t pIns, int64_t pKs, int pS, int p_) {
  return (pIns - 1) * pS - 2 * p_ + pKs;
}

struct ggml_tensor *ggml_conv_transpose_2d_p0(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                              int pStride) {
  GGML_ASSERT(pA->ne[3] == pB->ne[2]);

  const int64_t cNe[4] = {
      fGgmlCalcConvTransposeOutputSize(pB->ne[0], pA->ne[0], pStride, 0 /*p0*/),
      fGgmlCalcConvTransposeOutputSize(pB->ne[1], pA->ne[1], pStride, 0 /*p1*/),
      pA->ne[2],
      pB->ne[3],
  };

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  ggml_set_op_params_i32(vResult, 0, pStride);

  vResult->op = GGML_OP_CONV_TRANSPOSE_2D;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_pool_*

static int64_t fGgmlCalcPoolOutputSize(int64_t pIns, int pKs, int pS, float p_) {
  return (pIns + 2 * p_ - pKs) / pS + 1;
}

// ggml_pool_1d

struct ggml_tensor *ggml_pool_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_op_pool pOp, int pK0,
                                 int pS0, int p0) {
  const int64_t cNe[4] = {
      fGgmlCalcPoolOutputSize(pA->ne[0], pK0, pS0, p0),
      pA->ne[1],
      pA->ne[2],
      pA->ne[3],
  };
  GGML_ASSERT(cNe[0] > 0);

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  int32_t aParams[] = {pOp, pK0, pS0, p0};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_POOL_1D;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_pool_2d

struct ggml_tensor *ggml_pool_2d(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_op_pool pOp, int pK0,
                                 int pK1, int pS0, int pS1, float p0, float p1) {
  struct ggml_tensor *vResult;
  const int64_t cNe[4] = {
      fGgmlCalcPoolOutputSize(pA->ne[0], pK0, pS0, p0),
      fGgmlCalcPoolOutputSize(pA->ne[1], pK1, pS1, p1),
      pA->ne[2],
      pA->ne[3],
  };
  GGML_ASSERT(cNe[0] > 0);
  GGML_ASSERT(cNe[1] > 0);

  vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  int32_t aParams[] = {pOp, pK0, pK1, pS0, pS1, p0, p1};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_POOL_2D;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_pool_2d_back(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pAf,
                                      enum ggml_op_pool pOp, int pK0, int pK1, int pS0, int pS1, float p0, float p1) {
  struct ggml_tensor *vResult;
  vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, pAf->ne);

  int32_t aParams[] = {pOp, pK0, pK1, pS0, pS1, p0, p1};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_POOL_2D_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pAf;

  return vResult;
}

// ggml_upscale / ggml_interpolate

static struct ggml_tensor *fGgmlInterpolateImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0,
                                                int64_t pNe1, int64_t pNe2, int64_t pNe3, uint32_t pMode) {
  GGML_ASSERT((pMode & 0xFF) < GGML_SCALE_MODE_COUNT);
  // TODO: implement antialias for modes other than bilinear
  GGML_ASSERT(!(pMode & GGML_SCALE_FLAG_ANTIALIAS) || (pMode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, pA->type, pNe0, pNe1, pNe2, pNe3);

  ggml_set_op_params_i32(vResult, 0, (int32_t)pMode);

  vResult->op = GGML_OP_UPSCALE;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_upscale(struct ggml_context *pCtx, struct ggml_tensor *pA, int pScaleFactor,
                                 enum ggml_scale_mode pMode) {
  GGML_ASSERT(pScaleFactor > 1);
  return fGgmlInterpolateImpl(pCtx, pA, pA->ne[0] * pScaleFactor, pA->ne[1] * pScaleFactor, pA->ne[2], pA->ne[3],
                              pMode);
}

struct ggml_tensor *ggml_interpolate(struct ggml_context *pCtx, struct ggml_tensor *pA, int64_t pNe0, int64_t pNe1,
                                     int64_t pNe2, int64_t pNe3, uint32_t pMode) {
  return fGgmlInterpolateImpl(pCtx, pA, pNe0, pNe1, pNe2, pNe3, pMode);
}

// ggml_pad

struct ggml_tensor *ggml_pad(struct ggml_context *pCtx, struct ggml_tensor *pA, int p0, int p1, int p2, int p3) {
  return ggml_pad_ext(pCtx, pA, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

struct ggml_tensor *ggml_pad_circular(struct ggml_context *pCtx, struct ggml_tensor *pA, int p0, int p1, int p2,
                                      int p3) {
  return ggml_pad_ext_circular(pCtx, pA, 0, p0, 0, p1, 0, p2, 0, p3);
}

struct ggml_tensor *ggml_pad_ext(struct ggml_context *pCtx, struct ggml_tensor *pA, int pLp0, int pRp0, int pLp1,
                                 int pRp1, int pLp2, int pRp2, int pLp3, int pRp3) {
  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, pA->type, pA->ne[0] + pLp0 + pRp0, pA->ne[1] + pLp1 + pRp1,
                                                   pA->ne[2] + pLp2 + pRp2, pA->ne[3] + pLp3 + pRp3);

  ggml_set_op_params_i32(vResult, 0, pLp0);
  ggml_set_op_params_i32(vResult, 1, pRp0);
  ggml_set_op_params_i32(vResult, 2, pLp1);
  ggml_set_op_params_i32(vResult, 3, pRp1);
  ggml_set_op_params_i32(vResult, 4, pLp2);
  ggml_set_op_params_i32(vResult, 5, pRp2);
  ggml_set_op_params_i32(vResult, 6, pLp3);
  ggml_set_op_params_i32(vResult, 7, pRp3);
  ggml_set_op_params_i32(vResult, 8, 0); // not circular by default

  vResult->op = GGML_OP_PAD;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_pad_ext_circular

struct ggml_tensor *ggml_pad_ext_circular(struct ggml_context *pCtx, struct ggml_tensor *pA, int pLp0, int pRp0,
                                          int pLp1, int pRp1, int pLp2, int pRp2, int pLp3, int pRp3) {
  struct ggml_tensor *vResult = ggml_pad_ext(pCtx, pA, pLp0, pRp0, pLp1, pRp1, pLp2, pRp2, pLp3, pRp3);
  ggml_set_op_params_i32(vResult, 8, 1); // circular
  return vResult;
}

// ggml_pad_reflect_1d

struct ggml_tensor *ggml_pad_reflect_1d(struct ggml_context *pCtx, struct ggml_tensor *pA, int p0, int p1) {
  GGML_ASSERT(p0 >= 0);
  GGML_ASSERT(p1 >= 0);

  GGML_ASSERT(p0 < pA->ne[0]); // padding length on each size must be less than the
  GGML_ASSERT(p1 < pA->ne[0]); // existing length of the dimension being padded

  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  struct ggml_tensor *vResult =
      ggml_new_tensor_4d(pCtx, pA->type, pA->ne[0] + p0 + p1, pA->ne[1], pA->ne[2], pA->ne[3]);

  int32_t aParams[] = {p0, p1};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_PAD_REFLECT_1D;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_roll

struct ggml_tensor *ggml_roll(struct ggml_context *pCtx, struct ggml_tensor *pA, int pShift0, int pShift1, int pShift2,
                              int pShift3) {
  GGML_ASSERT(pA->nb[0] == ggml_type_size(pA->type));
  GGML_ASSERT(abs(pShift0) < pA->ne[0]);
  GGML_ASSERT(abs(pShift1) < pA->ne[1]);
  GGML_ASSERT(abs(pShift2) < pA->ne[2]);
  GGML_ASSERT(abs(pShift3) < pA->ne[3]);

  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_i32(vResult, 0, pShift0);
  ggml_set_op_params_i32(vResult, 1, pShift1);
  ggml_set_op_params_i32(vResult, 2, pShift2);
  ggml_set_op_params_i32(vResult, 3, pShift3);

  vResult->op = GGML_OP_ROLL;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_timestep_embedding

struct ggml_tensor *ggml_timestep_embedding(struct ggml_context *pCtx, struct ggml_tensor *pTimesteps, int pDim,
                                            int pMaxPeriod) {

  struct ggml_tensor *vResult = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, pDim, pTimesteps->ne[0]);

  ggml_set_op_params_i32(vResult, 0, pDim);
  ggml_set_op_params_i32(vResult, 1, pMaxPeriod);

  vResult->op = GGML_OP_TIMESTEP_EMBEDDING;
  vResult->src[0] = pTimesteps;

  return vResult;
}

// ggml_tri

struct ggml_tensor *ggml_tri(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_tri_type pType) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(pA->ne[0] == pA->ne[1]);

  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_i32(vResult, 0, pType);

  vResult->op = GGML_OP_TRI;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_fill

static struct ggml_tensor *fGgmlFillImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, float pC, bool pInplace) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32 || pA->type == GGML_TYPE_F16);
  GGML_ASSERT(ggml_is_contiguous(pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_f32(vResult, 0, pC);

  vResult->op = GGML_OP_FILL;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_fill(struct ggml_context *pCtx, struct ggml_tensor *pA, float pC) {
  return fGgmlFillImpl(pCtx, pA, pC, false);
}

struct ggml_tensor *ggml_fill_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, float pC) {
  return fGgmlFillImpl(pCtx, pA, pC, true);
}

// ggml_argsort

struct ggml_tensor *ggml_argsort(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_sort_order pOrder) {
  GGML_ASSERT(pA->ne[0] <= INT32_MAX);

  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_I32, GGML_MAX_DIMS, pA->ne);

  ggml_set_op_params_i32(vResult, 0, (int32_t)pOrder);

  vResult->op = GGML_OP_ARGSORT;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_argsort_top_k

struct ggml_tensor *ggml_argsort_top_k(struct ggml_context *pCtx, struct ggml_tensor *pA, int pK) {
  GGML_ASSERT(pA->ne[0] >= pK);

  struct ggml_tensor *vResult = ggml_argsort(pCtx, pA, GGML_SORT_ORDER_DESC);

  vResult = ggml_view_4d(pCtx, vResult, pK, vResult->ne[1], vResult->ne[2], vResult->ne[3], vResult->nb[1],
                         vResult->nb[2], vResult->nb[3], 0);

  return vResult;
}

// ggml_top_k

struct ggml_tensor *ggml_top_k(struct ggml_context *pCtx, struct ggml_tensor *pA, int pK) {
  GGML_ASSERT(pA->ne[0] >= pK);

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, GGML_TYPE_I32, pK, pA->ne[1], pA->ne[2], pA->ne[3]);

  vResult->op = GGML_OP_TOP_K;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_arange

struct ggml_tensor *ggml_arange(struct ggml_context *pCtx, float pStart, float pStop, float pStep) {
  GGML_ASSERT(pStop > pStart);

  const int64_t cSteps = (int64_t)ceilf((pStop - pStart) / pStep);

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, cSteps);

  ggml_set_op_params_f32(vResult, 0, pStart);
  ggml_set_op_params_f32(vResult, 1, pStop);
  ggml_set_op_params_f32(vResult, 2, pStep);

  vResult->op = GGML_OP_ARANGE;

  return vResult;
}

// ggml_flash_attn_ext

struct ggml_tensor *ggml_flash_attn_ext(struct ggml_context *pCtx, struct ggml_tensor *pQ, struct ggml_tensor *pK,
                                        struct ggml_tensor *pV, struct ggml_tensor *pMask, float pScale, float pMaxBias,
                                        float pLogitSoftcap) {
  GGML_ASSERT(fGgmlCanMulMat(pK, pQ));
  // TODO: check if vT can be multiplied by (k*qT)

  GGML_ASSERT(pQ->ne[3] == pK->ne[3]);
  GGML_ASSERT(pQ->ne[3] == pV->ne[3]);

  if (pMask) {
    GGML_ASSERT(pMask->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(pMask));
    // GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

    GGML_ASSERT(pQ->ne[2] % pMask->ne[2] == 0);
    GGML_ASSERT(pQ->ne[3] % pMask->ne[3] == 0);
  }

  if (pMaxBias > 0.0f) {
    GGML_ASSERT(pMask);
  }

  // permute(0, 2, 1, 3)
  int64_t aNe[4] = {pV->ne[0], pQ->ne[2], pQ->ne[1], pQ->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, aNe);

  float aParams[] = {pScale, pMaxBias, pLogitSoftcap};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_FLASH_ATTN_EXT;
  vResult->src[0] = pQ;
  vResult->src[1] = pK;
  vResult->src[2] = pV;
  vResult->src[3] = pMask;

  return vResult;
}

void ggml_flash_attn_ext_set_prec(struct ggml_tensor *pA, enum ggml_prec pPrec) {
  GGML_ASSERT(pA->op == GGML_OP_FLASH_ATTN_EXT);

  const int32_t cPrecI32 = (int32_t)pPrec;

  ggml_set_op_params_i32(pA, 3, cPrecI32); // scale is on first pos, max_bias on second
}

enum ggml_prec ggml_flash_attn_ext_get_prec(const struct ggml_tensor *pA) {
  GGML_ASSERT(pA->op == GGML_OP_FLASH_ATTN_EXT);

  const int32_t cPrecI32 = ggml_get_op_params_i32(pA, 3);

  return (enum ggml_prec)cPrecI32;
}

void ggml_flash_attn_ext_add_sinks(struct ggml_tensor *pA, struct ggml_tensor *pSinks) {
  if (!pSinks) {
    pA->src[4] = NULL;
    return;
  }

  GGML_ASSERT(pA->op == GGML_OP_FLASH_ATTN_EXT);
  GGML_ASSERT(pA->src[4] == NULL);
  GGML_ASSERT(pA->src[0]->ne[2] == pSinks->ne[0]);
  GGML_ASSERT(pSinks->type == GGML_TYPE_F32);

  pA->src[4] = pSinks;
}

// ggml_flash_attn_back

struct ggml_tensor *ggml_flash_attn_back(struct ggml_context *pCtx, struct ggml_tensor *pQ, struct ggml_tensor *pK,
                                         struct ggml_tensor *pV, struct ggml_tensor *pD, bool pMasked) {
  GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

  GGML_ASSERT(fGgmlCanMulMat(pK, pQ));
  // TODO: check if vT can be multiplied by (k*qT)

  // d shape [D,N,ne2,ne3]
  // q shape [D,N,ne2,ne3]
  // k shape [D,M,kvne2,ne3]
  // v shape [M,D,kvne2,ne3]

  const int64_t cD = pQ->ne[0];
  const int64_t cN = pQ->ne[1];
  const int64_t cM = pK->ne[1];
  const int64_t cNe2 = pQ->ne[2];
  const int64_t cNe3 = pQ->ne[3];
  const int64_t cKvne2 = pK->ne[2];

  GGML_ASSERT(pK->ne[0] == cD);
  GGML_ASSERT(pV->ne[0] == cM);
  GGML_ASSERT(pV->ne[1] == cD);
  GGML_ASSERT(pD->ne[0] == cD);
  GGML_ASSERT(pD->ne[1] == cN);
  GGML_ASSERT(pK->ne[2] == cKvne2);
  GGML_ASSERT(pK->ne[3] == cNe3);
  GGML_ASSERT(pV->ne[2] == cKvne2);
  GGML_ASSERT(pV->ne[3] == cNe3);
  GGML_ASSERT(pD->ne[2] == cNe2);
  GGML_ASSERT(pD->ne[3] == cNe3);

  GGML_ASSERT(cNe2 % cKvne2 == 0);

  // store gradients of q, k and v as continuous tensors concatenated in result.
  // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
  const int64_t cElemQ = ggml_nelements(pQ);
  const int64_t cElemK = ggml_nelements(pK);
  const int64_t cElemV = ggml_nelements(pV);

  enum ggml_type vResultType = GGML_TYPE_F32;
  GGML_ASSERT(ggml_blck_size(vResultType) == 1);
  const size_t cTsize = ggml_type_size(vResultType);

  const size_t cOffsQ = 0;
  const size_t cOffsK = cOffsQ + GGML_PAD(cElemQ * cTsize, GGML_MEM_ALIGN);
  const size_t cOffsV = cOffsK + GGML_PAD(cElemK * cTsize, GGML_MEM_ALIGN);
  const size_t cEnd = cOffsV + GGML_PAD(cElemV * cTsize, GGML_MEM_ALIGN);

  const size_t cNelements = (cEnd + cTsize - 1) / cTsize;

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, cNelements);

  int32_t vMaskedI = pMasked ? 1 : 0;
  ggml_set_op_params(vResult, &vMaskedI, sizeof(vMaskedI));

  vResult->op = GGML_OP_FLASH_ATTN_BACK;
  vResult->src[0] = pQ;
  vResult->src[1] = pK;
  vResult->src[2] = pV;
  vResult->src[3] = pD;

  return vResult;
}

// ggml_ssm_conv

struct ggml_tensor *ggml_ssm_conv(struct ggml_context *pCtx, struct ggml_tensor *pSx, struct ggml_tensor *pC) {
  GGML_ASSERT(ggml_is_3d(pSx));
  GGML_ASSERT(ggml_is_matrix(pC));

  const int64_t cDConv = pC->ne[0];
  const int64_t cDInner = pC->ne[1];
  const int64_t cNT = pSx->ne[0] - cDConv + 1; // tokens per sequence
  const int64_t cNS = pSx->ne[2];

  // TODO: maybe support other strides than 1?
  GGML_ASSERT(pSx->ne[0] == cDConv - 1 + cNT);
  GGML_ASSERT(pSx->ne[1] == cDInner);
  GGML_ASSERT(cNT >= 0);

  struct ggml_tensor *vResult = ggml_new_tensor_3d(pCtx, GGML_TYPE_F32, cDInner, cNT, cNS);

  vResult->op = GGML_OP_SSM_CONV;
  vResult->src[0] = pSx;
  vResult->src[1] = pC;

  return vResult;
}

// ggml_ssm_scan

struct ggml_tensor *ggml_ssm_scan(struct ggml_context *pCtx, struct ggml_tensor *pS, struct ggml_tensor *pX,
                                  struct ggml_tensor *pDt, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                  struct ggml_tensor *pC, struct ggml_tensor *pIds) {
  GGML_ASSERT(ggml_is_contiguous(pS));
  GGML_ASSERT(ggml_is_contiguous(pDt));
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(pX->nb[0] == ggml_type_size(pX->type));
  GGML_ASSERT(pB->nb[0] == ggml_type_size(pB->type));
  GGML_ASSERT(pC->nb[0] == ggml_type_size(pC->type));
  GGML_ASSERT(pX->nb[1] == pX->ne[0] * pX->nb[0]);
  GGML_ASSERT(pB->nb[1] == pB->ne[0] * pB->nb[0]);
  GGML_ASSERT(pC->nb[1] == pC->ne[0] * pC->nb[0]);
  GGML_ASSERT(ggml_are_same_shape(pB, pC));
  GGML_ASSERT(pIds->type == GGML_TYPE_I32);

  {
    const int64_t cDState = pS->ne[0];
    const int64_t cHeadDim = pX->ne[0];
    const int64_t cNHead = pX->ne[1];
    const int64_t cNSeqTokens = pX->ne[2];
    const int64_t cNSeqs = pX->ne[3];

    GGML_ASSERT(pDt->ne[0] == cNHead);
    GGML_ASSERT(pDt->ne[1] == cNSeqTokens);
    GGML_ASSERT(pDt->ne[2] == cNSeqs);
    GGML_ASSERT(ggml_is_3d(pDt));
    GGML_ASSERT(pS->ne[1] == cHeadDim);
    GGML_ASSERT(pS->ne[2] == cNHead);
    GGML_ASSERT(pB->ne[0] == cDState);
    GGML_ASSERT(pB->ne[2] == cNSeqTokens);
    GGML_ASSERT(pB->ne[3] == cNSeqs);
    GGML_ASSERT(pIds->ne[0] == cNSeqs);
    GGML_ASSERT(ggml_is_vector(pIds));
    GGML_ASSERT(pA->ne[1] == cNHead);
    GGML_ASSERT(ggml_is_matrix(pA));

    if (pA->ne[0] != 1) {
      // Mamba-1 has more granular decay factors
      GGML_ASSERT(pA->ne[0] == cDState);
    }
  }

  // concatenated y + ssm_states
  struct ggml_tensor *vResult =
      ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, ggml_nelements(pX) + pS->ne[0] * pS->ne[1] * pS->ne[2] * pIds->ne[0]);

  vResult->op = GGML_OP_SSM_SCAN;
  vResult->src[0] = pS;
  vResult->src[1] = pX;
  vResult->src[2] = pDt;
  vResult->src[3] = pA;
  vResult->src[4] = pB;
  vResult->src[5] = pC;
  vResult->src[6] = pIds;

  return vResult;
}

// ggml_win_part

struct ggml_tensor *ggml_win_part(struct ggml_context *pCtx, struct ggml_tensor *pA, int pW) {
  GGML_ASSERT(pA->ne[3] == 1);
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  // padding
  const int cPx = (pW - pA->ne[1] % pW) % pW;
  const int cPy = (pW - pA->ne[2] % pW) % pW;

  const int cNpx = (cPx + pA->ne[1]) / pW;
  const int cNpy = (cPy + pA->ne[2]) / pW;
  const int cNp = cNpx * cNpy;

  const int64_t cNe[4] = {
      pA->ne[0],
      pW,
      pW,
      cNp,
  };
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  int32_t aParams[] = {cNpx, cNpy, pW};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_WIN_PART;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_win_unpart

struct ggml_tensor *ggml_win_unpart(struct ggml_context *pCtx, struct ggml_tensor *pA, int pW0, int pH0, int pW) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32);

  const int64_t cNe[4] = {
      pA->ne[0],
      pW0,
      pH0,
      1,
  };
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 3, cNe);

  int32_t aParams[] = {pW};
  ggml_set_op_params(vResult, aParams, sizeof(aParams));

  vResult->op = GGML_OP_WIN_UNPART;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_get_rel_pos

struct ggml_tensor *ggml_get_rel_pos(struct ggml_context *pCtx, struct ggml_tensor *pA, int pQh, int pKh) {
  GGML_ASSERT(pQh == pKh);
  GGML_ASSERT(2 * MAX(pQh, pKh) - 1 == pA->ne[1]);

  const int64_t cNe[4] = {
      pA->ne[0],
      pKh,
      pQh,
      1,
  };
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F16, 3, cNe);

  vResult->op = GGML_OP_GET_REL_POS;
  vResult->src[0] = pA;

  return vResult;
}

// ggml_add_rel_pos

static struct ggml_tensor *fGgmlAddRelPosImpl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                              struct ggml_tensor *pPw, struct ggml_tensor *pPh, bool pInplace) {
  GGML_ASSERT(ggml_are_same_shape(pPw, pPh));
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_is_contiguous(pPw));
  GGML_ASSERT(ggml_is_contiguous(pPh));
  GGML_ASSERT(pPh->type == GGML_TYPE_F32);
  GGML_ASSERT(pPw->type == GGML_TYPE_F32);
  GGML_ASSERT(pPw->ne[3] == pA->ne[2]);
  GGML_ASSERT(pPw->ne[0] * pPw->ne[0] == pA->ne[0]);
  GGML_ASSERT(pPw->ne[1] * pPw->ne[2] == pA->ne[1]);

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);
  ggml_set_op_params_i32(vResult, 0, pInplace ? 1 : 0);

  vResult->op = GGML_OP_ADD_REL_POS;
  vResult->src[0] = pA;
  vResult->src[1] = pPw;
  vResult->src[2] = pPh;

  return vResult;
}

struct ggml_tensor *ggml_add_rel_pos(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pPw,
                                     struct ggml_tensor *pPh) {
  return fGgmlAddRelPosImpl(pCtx, pA, pPw, pPh, false);
}

struct ggml_tensor *ggml_add_rel_pos_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pPw,
                                             struct ggml_tensor *pPh) {
  return fGgmlAddRelPosImpl(pCtx, pA, pPw, pPh, true);
}

// ggml_rwkv_wkv6

struct ggml_tensor *ggml_rwkv_wkv6(struct ggml_context *pCtx, struct ggml_tensor *pK, struct ggml_tensor *pV,
                                   struct ggml_tensor *pR, struct ggml_tensor *pTf, struct ggml_tensor *pTd,
                                   struct ggml_tensor *pState) {
  GGML_ASSERT(ggml_is_contiguous(pK));
  GGML_ASSERT(ggml_is_contiguous(pV));
  GGML_ASSERT(ggml_is_contiguous(pR));
  GGML_ASSERT(ggml_is_contiguous(pTf));
  GGML_ASSERT(ggml_is_contiguous(pTd));
  GGML_ASSERT(ggml_is_contiguous(pState));

  const int64_t cS = pK->ne[0];
  const int64_t cH = pK->ne[1];
  const int64_t cNTokens = pK->ne[2];
  const int64_t cNSeqs = pState->ne[1];
  {
    GGML_ASSERT(pV->ne[0] == cS && pV->ne[1] == cH && pV->ne[2] == cNTokens);
    GGML_ASSERT(pR->ne[0] == cS && pR->ne[1] == cH && pR->ne[2] == cNTokens);
    GGML_ASSERT(pTd->ne[0] == cS && pTd->ne[1] == cH && pTd->ne[2] == cNTokens);
    GGML_ASSERT(ggml_nelements(pState) == cS * cS * cH * cNSeqs);
  }

  // concat output and new_state
  const int64_t cNe[4] = {cS * cH, cNTokens + cS * cNSeqs, 1, 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  vResult->op = GGML_OP_RWKV_WKV6;
  vResult->src[0] = pK;
  vResult->src[1] = pV;
  vResult->src[2] = pR;
  vResult->src[3] = pTf;
  vResult->src[4] = pTd;
  vResult->src[5] = pState;

  return vResult;
}

// ggml_gated_linear_attn

struct ggml_tensor *ggml_gated_linear_attn(struct ggml_context *pCtx, struct ggml_tensor *pK, struct ggml_tensor *pV,
                                           struct ggml_tensor *pQ, struct ggml_tensor *pG, struct ggml_tensor *pState,
                                           float pScale) {
  GGML_ASSERT(ggml_is_contiguous(pK));
  GGML_ASSERT(ggml_is_contiguous(pV));
  GGML_ASSERT(ggml_is_contiguous(pQ));
  GGML_ASSERT(ggml_is_contiguous(pG));
  GGML_ASSERT(ggml_is_contiguous(pState));

  const int64_t cS = pK->ne[0];
  const int64_t cH = pK->ne[1];
  const int64_t cNTokens = pK->ne[2];
  const int64_t cNSeqs = pState->ne[1];
  {
    GGML_ASSERT(pV->ne[0] == cS && pV->ne[1] == cH && pV->ne[2] == cNTokens);
    GGML_ASSERT(pQ->ne[0] == cS && pQ->ne[1] == cH && pQ->ne[2] == cNTokens);
    GGML_ASSERT(pG->ne[0] == cS && pG->ne[1] == cH && pG->ne[2] == cNTokens);
    GGML_ASSERT(ggml_nelements(pState) == cS * cS * cH * cNSeqs);
  }

  // concat output and new_state
  const int64_t cNe[4] = {cS * cH, cNTokens + cS * cNSeqs, 1, 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  ggml_set_op_params_f32(vResult, 0, pScale);

  vResult->op = GGML_OP_GATED_LINEAR_ATTN;
  vResult->src[0] = pK;
  vResult->src[1] = pV;
  vResult->src[2] = pQ;
  vResult->src[3] = pG;
  vResult->src[4] = pState;

  return vResult;
}

// ggml_rwkv_wkv7

struct ggml_tensor *ggml_rwkv_wkv7(struct ggml_context *pCtx, struct ggml_tensor *pR, struct ggml_tensor *pW,
                                   struct ggml_tensor *pK, struct ggml_tensor *pV, struct ggml_tensor *pA,
                                   struct ggml_tensor *pB, struct ggml_tensor *pState) {
  GGML_ASSERT(ggml_is_contiguous(pR));
  GGML_ASSERT(ggml_is_contiguous(pW));
  GGML_ASSERT(ggml_is_contiguous(pK));
  GGML_ASSERT(ggml_is_contiguous(pV));
  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_is_contiguous(pB));
  GGML_ASSERT(ggml_is_contiguous(pState));

  const int64_t cS = pK->ne[0];
  const int64_t cH = pK->ne[1];
  const int64_t cNTokens = pK->ne[2];
  const int64_t cNSeqs = pState->ne[1];
  {
    GGML_ASSERT(pW->ne[0] == cS && pW->ne[1] == cH && pW->ne[2] == cNTokens);
    GGML_ASSERT(pK->ne[0] == cS && pK->ne[1] == cH && pK->ne[2] == cNTokens);
    GGML_ASSERT(pV->ne[0] == cS && pV->ne[1] == cH && pV->ne[2] == cNTokens);
    GGML_ASSERT(pA->ne[0] == cS && pA->ne[1] == cH && pA->ne[2] == cNTokens);
    GGML_ASSERT(pB->ne[0] == cS && pB->ne[1] == cH && pB->ne[2] == cNTokens);
    GGML_ASSERT(ggml_nelements(pState) == cS * cS * cH * cNSeqs);
  }

  // concat output and new_state
  const int64_t cNe[4] = {cS * cH, cNTokens + cS * cNSeqs, 1, 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  vResult->op = GGML_OP_RWKV_WKV7;
  vResult->src[0] = pR;
  vResult->src[1] = pW;
  vResult->src[2] = pK;
  vResult->src[3] = pV;
  vResult->src[4] = pA;
  vResult->src[5] = pB;
  vResult->src[6] = pState;

  return vResult;
}

// ggml_unary

static struct ggml_tensor *fGgmlUnaryImpl(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_unary_op pOp,
                                          bool pInplace) {
  GGML_ASSERT(ggml_is_contiguous_rows(pA));

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  ggml_set_op_params_i32(vResult, 0, (int32_t)pOp);

  vResult->op = GGML_OP_UNARY;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_unary(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_unary_op pOp) {
  return fGgmlUnaryImpl(pCtx, pA, pOp, false);
}

struct ggml_tensor *ggml_unary_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, enum ggml_unary_op pOp) {
  return fGgmlUnaryImpl(pCtx, pA, pOp, true);
}

// ggml_map_custom1

static struct ggml_tensor *fGgmlMapCustom1Impl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                               const ggml_custom1_op_t pFun, int pNTasks, void *pUserdata,
                                               bool pInplace) {
  GGML_ASSERT(pNTasks == GGML_N_TASKS_MAX || pNTasks > 0);

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  struct ggml_map_custom1_op_params vParams = {/*.fun      =*/pFun,
                                               /*.n_tasks  =*/pNTasks,
                                               /*.userdata =*/pUserdata};
  ggml_set_op_params(vResult, &vParams, sizeof(vParams));

  vResult->op = GGML_OP_MAP_CUSTOM1;
  vResult->src[0] = pA;

  return vResult;
}

struct ggml_tensor *ggml_map_custom1(struct ggml_context *pCtx, struct ggml_tensor *pA, const ggml_custom1_op_t pFun,
                                     int pNTasks, void *pUserdata) {
  return fGgmlMapCustom1Impl(pCtx, pA, pFun, pNTasks, pUserdata, false);
}

struct ggml_tensor *ggml_map_custom1_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                             const ggml_custom1_op_t pFun, int pNTasks, void *pUserdata) {
  return fGgmlMapCustom1Impl(pCtx, pA, pFun, pNTasks, pUserdata, true);
}

// ggml_map_custom2

static struct ggml_tensor *fGgmlMapCustom2Impl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                               struct ggml_tensor *pB, const ggml_custom2_op_t pFun, int pNTasks,
                                               void *pUserdata, bool pInplace) {
  GGML_ASSERT(pNTasks == GGML_N_TASKS_MAX || pNTasks > 0);

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  struct ggml_map_custom2_op_params vParams = {/*.fun      =*/pFun,
                                               /*.n_tasks  =*/pNTasks,
                                               /*.userdata =*/pUserdata};
  ggml_set_op_params(vResult, &vParams, sizeof(vParams));

  vResult->op = GGML_OP_MAP_CUSTOM2;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

struct ggml_tensor *ggml_map_custom2(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                     const ggml_custom2_op_t pFun, int pNTasks, void *pUserdata) {
  return fGgmlMapCustom2Impl(pCtx, pA, pB, pFun, pNTasks, pUserdata, false);
}

struct ggml_tensor *ggml_map_custom2_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                             const ggml_custom2_op_t pFun, int pNTasks, void *pUserdata) {
  return fGgmlMapCustom2Impl(pCtx, pA, pB, pFun, pNTasks, pUserdata, true);
}

// ggml_map_custom3

static struct ggml_tensor *fGgmlMapCustom3Impl(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                               struct ggml_tensor *pB, struct ggml_tensor *pC,
                                               const ggml_custom3_op_t pFun, int pNTasks, void *pUserdata,
                                               bool pInplace) {
  GGML_ASSERT(pNTasks == GGML_N_TASKS_MAX || pNTasks > 0);

  struct ggml_tensor *vResult = pInplace ? ggml_view_tensor(pCtx, pA) : ggml_dup_tensor(pCtx, pA);

  struct ggml_map_custom3_op_params vParams = {/*.fun      =*/pFun,
                                               /*.n_tasks  =*/pNTasks,
                                               /*.userdata =*/pUserdata};
  ggml_set_op_params(vResult, &vParams, sizeof(vParams));

  vResult->op = GGML_OP_MAP_CUSTOM3;
  vResult->src[0] = pA;
  vResult->src[1] = pB;
  vResult->src[2] = pC;

  return vResult;
}

struct ggml_tensor *ggml_map_custom3(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                     struct ggml_tensor *pC, const ggml_custom3_op_t pFun, int pNTasks,
                                     void *pUserdata) {
  return fGgmlMapCustom3Impl(pCtx, pA, pB, pC, pFun, pNTasks, pUserdata, false);
}

struct ggml_tensor *ggml_map_custom3_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                             struct ggml_tensor *pC, const ggml_custom3_op_t pFun, int pNTasks,
                                             void *pUserdata) {
  return fGgmlMapCustom3Impl(pCtx, pA, pB, pC, pFun, pNTasks, pUserdata, true);
}

struct ggml_tensor *ggml_custom_4d(struct ggml_context *pCtx, enum ggml_type pType, int64_t pNe0, int64_t pNe1,
                                   int64_t pNe2, int64_t pNe3, struct ggml_tensor **pArgs, int pNArgs,
                                   ggml_custom_op_t pFun, int pNTasks, void *pUserdata) {

  GGML_ASSERT(pNArgs < GGML_MAX_SRC);

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, pType, pNe0, pNe1, pNe2, pNe3);

  struct ggml_custom_op_params vParams = {/*.fun      =*/pFun,
                                          /*.n_tasks  =*/pNTasks,
                                          /*.userdata =*/pUserdata};
  ggml_set_op_params(vResult, &vParams, sizeof(vParams));

  vResult->op = GGML_OP_CUSTOM;
  for (int vI = 0; vI < pNArgs; vI++) {
    vResult->src[vI] = pArgs[vI];
  }

  return vResult;
}

struct ggml_tensor *ggml_custom_inplace(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor **pArgs,
                                        int pNArgs, ggml_custom_op_t pFun, int pNTasks, void *pUserdata) {

  GGML_ASSERT(pNArgs < GGML_MAX_SRC - 1);

  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);

  struct ggml_custom_op_params vParams = {/*.fun      =*/pFun,
                                          /*.n_tasks  =*/pNTasks,
                                          /*.userdata =*/pUserdata};
  ggml_set_op_params(vResult, &vParams, sizeof(vParams));

  vResult->op = GGML_OP_CUSTOM;
  vResult->src[0] = pA;
  for (int vI = 0; vI < pNArgs; vI++) {
    vResult->src[vI + 1] = pArgs[vI];
  }

  return vResult;
}
// ggml_cross_entropy_loss

struct ggml_tensor *ggml_cross_entropy_loss(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB) {
  GGML_ASSERT(ggml_are_same_shape(pA, pB));

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, pA->type, 1);

  vResult->op = GGML_OP_CROSS_ENTROPY_LOSS;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_cross_entropy_loss_back

struct ggml_tensor *ggml_cross_entropy_loss_back(struct ggml_context *pCtx, struct ggml_tensor *pA,
                                                 struct ggml_tensor *pB, struct ggml_tensor *pC) {
  GGML_ASSERT(ggml_is_scalar(pA));
  GGML_ASSERT(ggml_are_same_shape(pB, pC));

  struct ggml_tensor *vResult = ggml_dup_tensor(pCtx, pB);

  vResult->op = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
  vResult->src[0] = pA;
  vResult->src[1] = pB;
  vResult->src[2] = pC;

  return vResult;
}

// opt_step_adamw

struct ggml_tensor *ggml_opt_step_adamw(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pGrad,
                                        struct ggml_tensor *pM, struct ggml_tensor *pV,
                                        struct ggml_tensor *pAdamwParams) {
  GGML_ASSERT(pA->flags & GGML_TENSOR_FLAG_PARAM);
  GGML_ASSERT(ggml_are_same_shape(pA, pGrad));
  GGML_ASSERT(ggml_are_same_shape(pA, pM));
  GGML_ASSERT(ggml_are_same_shape(pA, pV));
  GGML_ASSERT(pAdamwParams->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_nelements(pAdamwParams) == 7);

  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);

  vResult->op = GGML_OP_OPT_STEP_ADAMW;
  vResult->src[0] = pA;
  vResult->src[1] = pGrad;
  vResult->src[2] = pM;
  vResult->src[3] = pV;
  vResult->src[4] = pAdamwParams;

  return vResult;
}

// opt_step_sgd

struct ggml_tensor *ggml_opt_step_sgd(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pGrad,
                                      struct ggml_tensor *pParams) {
  GGML_ASSERT(pA->flags & GGML_TENSOR_FLAG_PARAM);
  GGML_ASSERT(ggml_are_same_shape(pA, pGrad));
  GGML_ASSERT(pParams->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_nelements(pParams) == 2);

  struct ggml_tensor *vResult = ggml_view_tensor(pCtx, pA);

  vResult->op = GGML_OP_OPT_STEP_SGD;
  vResult->src[0] = pA;
  vResult->src[1] = pGrad;
  vResult->src[2] = pParams;

  return vResult;
}

// solve_tri

struct ggml_tensor *ggml_solve_tri(struct ggml_context *pCtx, struct ggml_tensor *pA, struct ggml_tensor *pB,
                                   bool pLeft, bool pLower, bool pUni) {
  GGML_ASSERT(pA->type == GGML_TYPE_F32);
  GGML_ASSERT(pB->type == GGML_TYPE_F32);

  // A must be square and lower diagonal
  GGML_ASSERT(pA->ne[0] == pA->ne[1]);
  // B must have same outer dimension as A
  GGML_ASSERT(pA->ne[1] == pB->ne[1]);

  // batch dimensions must be equal
  GGML_ASSERT(pA->ne[2] == pB->ne[2]);
  GGML_ASSERT(pA->ne[3] == pB->ne[3]);

  GGML_ASSERT(ggml_is_contiguous(pA));
  GGML_ASSERT(ggml_is_contiguous(pB));

  GGML_ASSERT(pLower && pLeft && !pUni); // TODO: support other variants

  struct ggml_tensor *vResult = ggml_new_tensor_4d(pCtx, GGML_TYPE_F32, pB->ne[0], pB->ne[1], pB->ne[2], pB->ne[3]);

  vResult->op = GGML_OP_SOLVE_TRI;
  vResult->src[0] = pA;
  vResult->src[1] = pB;

  return vResult;
}

// ggml_gated_delta_net

struct ggml_tensor *ggml_gated_delta_net(struct ggml_context *pCtx, struct ggml_tensor *pQ, struct ggml_tensor *k,
                                         struct ggml_tensor *pV, struct ggml_tensor *pG, struct ggml_tensor *pBeta,
                                         struct ggml_tensor *pState, int64_t K) {
  GGML_ASSERT(ggml_is_contiguous_rows(pQ));
  GGML_ASSERT(ggml_is_contiguous_rows(k));
  GGML_ASSERT(ggml_is_contiguous_rows(pV));
  GGML_ASSERT(ggml_is_contiguous(pG));
  GGML_ASSERT(ggml_is_contiguous(pBeta));
  GGML_ASSERT(ggml_is_contiguous(pState));

  GGML_ASSERT(pQ->type == GGML_TYPE_F32);
  GGML_ASSERT(k->type == GGML_TYPE_F32);
  GGML_ASSERT(pV->type == GGML_TYPE_F32);
  GGML_ASSERT(pG->type == GGML_TYPE_F32);
  GGML_ASSERT(pBeta->type == GGML_TYPE_F32);
  GGML_ASSERT(pState->type == GGML_TYPE_F32);

  const int64_t cSV = pV->ne[0];
  const int64_t cH = pV->ne[1];
  const int64_t cNTokens = pV->ne[2];
  const int64_t cNSeqs = pV->ne[3];

  // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
  GGML_ASSERT(pG->ne[0] == 1 || pG->ne[0] == cSV);
  GGML_ASSERT(pBeta->ne[0] == 1);

  // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
  GGML_ASSERT(pState->ne[0] == cSV);
  GGML_ASSERT(pState->ne[1] == cSV);
  GGML_ASSERT(pState->ne[2] == cH);
  GGML_ASSERT(pState->ne[3] == cNSeqs);
  GGML_ASSERT(K >= 1);
  const int64_t cStateRows = K * cSV * cNSeqs;
  const int64_t cNe[4] = {cSV * cH, cNTokens * cNSeqs + cStateRows, 1, 1};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, cNe);

  ggml_set_op_params_i32(vResult, 0, (int32_t)K);

  vResult->op = GGML_OP_GATED_DELTA_NET;
  vResult->src[0] = pQ;
  vResult->src[1] = k;
  vResult->src[2] = pV;
  vResult->src[3] = pG;
  vResult->src[4] = pBeta;
  vResult->src[5] = pState;

  return vResult;
}

// ggml_lightning_indexer

struct ggml_tensor *ggml_lightning_indexer(struct ggml_context *pCtx, struct ggml_tensor *pQ, struct ggml_tensor *pK,
                                           struct ggml_tensor *pWeights, struct ggml_tensor *pMask) {

  GGML_ASSERT(pQ->type == GGML_TYPE_F32);
  GGML_ASSERT(pWeights->type == GGML_TYPE_F32);
  GGML_ASSERT(pMask->type == GGML_TYPE_F16);
  GGML_ASSERT(pQ->ne[0] == pK->ne[0]);
  GGML_ASSERT(pMask->ne[0] == pK->ne[2]);
  GGML_ASSERT(pQ->ne[1] == pWeights->ne[0]);
  GGML_ASSERT(pK->ne[1] == 1);
  GGML_ASSERT(pMask->ne[1] == pQ->ne[2]);
  GGML_ASSERT(pQ->ne[2] == pWeights->ne[1]);
  GGML_ASSERT(pWeights->ne[2] == 1);
  GGML_ASSERT(pMask->ne[2] == 1);
  GGML_ASSERT(pQ->ne[3] == pK->ne[3]);
  GGML_ASSERT(pK->ne[3] == pWeights->ne[3]);
  GGML_ASSERT(pWeights->ne[3] % pMask->ne[3] == 0);

  int64_t aNe[4] = {pK->ne[2], pQ->ne[2], 1, pQ->ne[3]};
  struct ggml_tensor *vResult = ggml_new_tensor(pCtx, GGML_TYPE_F32, 4, aNe);

  vResult->op = GGML_OP_LIGHTNING_INDEXER;
  vResult->src[0] = pQ;
  vResult->src[1] = pK;
  vResult->src[2] = pWeights;
  vResult->src[3] = pMask;

  return vResult;
}

// ggml_dsv4_hc_comb

struct ggml_tensor *ggml_dsv4_hc_comb(struct ggml_context *pCtx, struct ggml_tensor *pMixes, struct ggml_tensor *pScale,
                                      struct ggml_tensor *pBase, float pEps, int32_t pNIter) {
  GGML_ASSERT(pMixes->type == GGML_TYPE_F32);
  GGML_ASSERT(pScale->type == GGML_TYPE_F32);
  GGML_ASSERT(pBase->type == GGML_TYPE_F32);
  GGML_ASSERT(pNIter > 0);

  const int64_t cHcMixDim = pMixes->ne[0];
  const int64_t cNTokens = pMixes->ne[1];

  int64_t vHc = 0;
  for (int64_t vI = 1; vI * vI + 2 * vI <= cHcMixDim; ++vI) {
    if ((2 + vI) * vI == cHcMixDim) {
      vHc = vI;
      break;
    }
  }

  GGML_ASSERT(vHc > 0);
  GGML_ASSERT(vHc == 4);
  GGML_ASSERT(pMixes->ne[2] == 1);
  GGML_ASSERT(pMixes->ne[3] == 1);
  GGML_ASSERT(pScale->ne[0] >= 3);
  GGML_ASSERT(pScale->ne[1] == 1);
  GGML_ASSERT(pScale->ne[2] == 1);
  GGML_ASSERT(pScale->ne[3] == 1);
  GGML_ASSERT(pBase->ne[0] == cHcMixDim);
  GGML_ASSERT(pBase->ne[1] == 1);
  GGML_ASSERT(pBase->ne[2] == 1);
  GGML_ASSERT(pBase->ne[3] == 1);

  struct ggml_tensor *vResult = ggml_new_tensor_3d(pCtx, GGML_TYPE_F32, vHc, vHc, cNTokens);

  ggml_set_op_params_f32(vResult, 0, pEps);
  ggml_set_op_params_i32(vResult, 1, pNIter);

  vResult->op = GGML_OP_DSV4_HC_COMB;
  vResult->src[0] = pMixes;
  vResult->src[1] = pScale;
  vResult->src[2] = pBase;

  return vResult;
}

// ggml_dsv4_hc_pre

static struct ggml_tensor *fGgmlDsv4HcPreImpl(struct ggml_context *pCtx, struct ggml_tensor *pX,
                                              struct ggml_tensor *pWeights, float pScale, bool pGated) {
  GGML_ASSERT(pX->type == GGML_TYPE_F32);
  GGML_ASSERT(pWeights->type == GGML_TYPE_F32);

  const int64_t cNEmbd = pX->ne[0];
  const int64_t cHc = pX->ne[1];
  const int64_t cNTokens = pX->ne[2];

  GGML_ASSERT(cHc > 0);
  GGML_ASSERT(pX->ne[3] == 1);
  if (pGated) {
    GGML_ASSERT(pWeights->ne[0] == cNEmbd);
    GGML_ASSERT(pWeights->ne[1] == cHc);
    GGML_ASSERT(pWeights->ne[2] == cNTokens);
  } else {
    GGML_ASSERT(pWeights->ne[0] == cHc);
    GGML_ASSERT(pWeights->ne[1] == cNTokens);
    GGML_ASSERT(pWeights->ne[2] == 1);
  }
  GGML_ASSERT(pWeights->ne[3] == 1);

  struct ggml_tensor *vResult = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, cNEmbd, cNTokens);

  ggml_set_op_params_f32(vResult, 0, pScale);
  ggml_set_op_params_i32(vResult, 1, pGated ? 1 : 0);

  vResult->op = GGML_OP_DSV4_HC_PRE;
  vResult->src[0] = pX;
  vResult->src[1] = pWeights;

  return vResult;
}

struct ggml_tensor *ggml_dsv4_hc_pre(struct ggml_context *pCtx, struct ggml_tensor *pX, struct ggml_tensor *pWeights) {
  return fGgmlDsv4HcPreImpl(pCtx, pX, pWeights, 1.0f, false);
}

struct ggml_tensor *ggml_dsv4_hc_pre_gated(struct ggml_context *pCtx, struct ggml_tensor *pX, struct ggml_tensor *pGate,
                                           float pScale) {
  return fGgmlDsv4HcPreImpl(pCtx, pX, pGate, pScale, true);
}

// ggml_dsv4_hc_post

struct ggml_tensor *ggml_dsv4_hc_post(struct ggml_context *pCtx, struct ggml_tensor *pX, struct ggml_tensor *pResidual,
                                      struct ggml_tensor *pPost, struct ggml_tensor *pComb) {
  GGML_ASSERT(pX->type == GGML_TYPE_F32);
  GGML_ASSERT(pResidual->type == GGML_TYPE_F32);
  GGML_ASSERT(pPost->type == GGML_TYPE_F32);

  const int64_t cNEmbd = pX->ne[0];
  const int64_t cNTokens = pX->ne[1];
  const int64_t cHc = pResidual->ne[1];

  GGML_ASSERT(cHc > 0);
  GGML_ASSERT(pX->ne[2] == 1);
  GGML_ASSERT(pX->ne[3] == 1);

  GGML_ASSERT(pResidual->ne[0] == cNEmbd);
  GGML_ASSERT(pResidual->ne[2] == cNTokens);
  GGML_ASSERT(pResidual->ne[3] == 1);

  GGML_ASSERT(pPost->ne[0] == cHc);
  GGML_ASSERT(pPost->ne[1] == cNTokens);
  GGML_ASSERT(pPost->ne[2] == 1);
  GGML_ASSERT(pPost->ne[3] == 1);

  if (pComb) {
    GGML_ASSERT(pComb->type == GGML_TYPE_F32);
    GGML_ASSERT(pComb->ne[0] == cHc);
    GGML_ASSERT(pComb->ne[1] == cHc);
    GGML_ASSERT(pComb->ne[2] == cNTokens);
    GGML_ASSERT(pComb->ne[3] == 1);
  }

  struct ggml_tensor *vResult = ggml_new_tensor_3d(pCtx, GGML_TYPE_F32, cNEmbd, cHc, cNTokens);

  vResult->op = GGML_OP_DSV4_HC_POST;
  vResult->src[0] = pX;
  vResult->src[1] = pResidual;
  vResult->src[2] = pPost;
  vResult->src[3] = pComb;

  return vResult;
}

////////////////////////////////////////////////////////////////////////////////

struct ggml_hash_set ggml_hash_set_new(size_t pSize) {
  pSize = ggml_hash_size(pSize);
  struct ggml_hash_set vResult;
  vResult.size = pSize;
  vResult.keys = GGML_MALLOC(sizeof(struct ggml_tensor *) * pSize);
  vResult.used = GGML_CALLOC(ggml_bitset_size(pSize), sizeof(ggml_bitset_t));
  return vResult;
}

void ggml_hash_set_reset(struct ggml_hash_set *pHashSet) {
  memset(pHashSet->used, 0, sizeof(ggml_bitset_t) * ggml_bitset_size(pHashSet->size));
}

void ggml_hash_set_free(struct ggml_hash_set *pHashSet) {
  GGML_FREE(pHashSet->used);
  GGML_FREE(pHashSet->keys);
}

size_t ggml_hash_size(size_t pMinSz) {
  // next primes after powers of two
  static const size_t aPrimes[] = {
      2,       3,       5,        11,       17,       37,        67,        131,       257,        521,       1031,
      2053,    4099,    8209,     16411,    32771,    65537,     131101,    262147,    524309,     1048583,   2097169,
      4194319, 8388617, 16777259, 33554467, 67108879, 134217757, 268435459, 536870923, 1073741827, 2147483659};
  static const size_t cNPrimes = sizeof(aPrimes) / sizeof(aPrimes[0]);

  // find the smallest prime that is larger or equal than min_sz
  size_t vL = 0;
  size_t vR = cNPrimes;
  while (vL < vR) {
    size_t vM = (vL + vR) / 2;
    if (aPrimes[vM] < pMinSz) {
      vL = vM + 1;
    } else {
      vR = vM;
    }
  }
  size_t vSz = vL < cNPrimes ? aPrimes[vL] : pMinSz | 1;
  return vSz;
}

struct hash_map {
  struct ggml_hash_set vSet;
  struct ggml_tensor **vVals;
};

static struct hash_map *fGgmlNewHashMap(size_t pSize) {
  struct hash_map *vResult = GGML_MALLOC(sizeof(struct hash_map));
  vResult->vSet = ggml_hash_set_new(pSize);
  vResult->vVals = GGML_CALLOC(vResult->vSet.size, sizeof(struct ggml_tensor *));
  return vResult;
}

static void fGgmlHashMapFree(struct hash_map *pMap) {
  ggml_hash_set_free(&pMap->vSet);
  GGML_FREE(pMap->vVals);
  GGML_FREE(pMap);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

static void fGgmlAddOrSet(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, size_t pIsrc,
                          struct ggml_tensor *pTensor) {
  struct ggml_tensor *vSrc = pCgraph->visited_hash_set.keys[pIsrc];
  GGML_ASSERT(vSrc);
  if (pCgraph->grads[pIsrc]) {
    pCgraph->grads[pIsrc] = fGgmlAddImpl(pCtx, pCgraph->grads[pIsrc], pTensor, /*inplace =*/pCgraph->grad_accs[pIsrc]);
  } else {
    pCgraph->grads[pIsrc] = pTensor;
  }
  ggml_format_name(pCgraph->grads[pIsrc], "grad for %s", vSrc->name);
  ggml_build_forward_expand(pCgraph, pCgraph->grads[pIsrc]);
}

static void fGgmlAccOrSet(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, size_t pIsrc,
                          struct ggml_tensor *pTensor, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                          const size_t pOffset) {
  struct ggml_tensor *vSrc = pCgraph->visited_hash_set.keys[pIsrc];
  GGML_ASSERT(vSrc);
  if (pCgraph->grads[pIsrc]) {
    pCgraph->grads[pIsrc] =
        fGgmlAccImpl(pCtx, pCgraph->grads[pIsrc], pTensor, pNb1, pNb2, pNb3, pOffset, pCgraph->grad_accs[pIsrc]);
  } else {
    struct ggml_tensor *vAZero =
        ggml_scale(pCtx, vSrc, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
    pCgraph->grads[pIsrc] = fGgmlAccImpl(pCtx, vAZero, pTensor, pNb1, pNb2, pNb3, pOffset, false);
  }
  ggml_format_name(pCgraph->grads[pIsrc], "grad for %s", pCgraph->visited_hash_set.keys[pIsrc]->name);
  ggml_build_forward_expand(pCgraph, pCgraph->grads[pIsrc]);
}

static void fGgmlAdd1OrSet(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, size_t pIsrc,
                           struct ggml_tensor *pTensor) {
  struct ggml_tensor *vSrc = pCgraph->visited_hash_set.keys[pIsrc];
  GGML_ASSERT(vSrc);
  if (pCgraph->grads[pIsrc]) {
    pCgraph->grads[pIsrc] = fGgmlAdd1Impl(pCtx, pCgraph->grads[pIsrc], pTensor, pCgraph->grad_accs[pIsrc]);
  } else {
    pCgraph->grads[pIsrc] = ggml_repeat(pCtx, pTensor, vSrc);
  }
  ggml_format_name(pCgraph->grads[pIsrc], "grad for %s", vSrc->name);
  ggml_build_forward_expand(pCgraph, pCgraph->grads[pIsrc]);
}

static void fGgmlSubOrSet(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, size_t pIsrc,
                          struct ggml_tensor *pTensor) {
  struct ggml_tensor *vSrc = pCgraph->visited_hash_set.keys[pIsrc];
  GGML_ASSERT(vSrc);
  if (pCgraph->grads[pIsrc]) {
    pCgraph->grads[pIsrc] = fGgmlSubImpl(pCtx, pCgraph->grads[pIsrc], pTensor, pCgraph->grad_accs[pIsrc]);
  } else {
    pCgraph->grads[pIsrc] = ggml_neg(pCtx, pTensor);
  }
  ggml_format_name(pCgraph->grads[pIsrc], "grad for %s", vSrc->name);
  ggml_build_forward_expand(pCgraph, pCgraph->grads[pIsrc]);
}

static void fGgmlComputeBackward(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, int pI,
                                 const bool *pGradsNeeded) {
  struct ggml_tensor *vTensor = pCgraph->nodes[pI];
  struct ggml_tensor *vGrad = ggml_graph_get_grad(pCgraph, vTensor);

  if (!vGrad) {
    return;
  }

  struct ggml_tensor *vSrc0 = vTensor->src[0];
  struct ggml_tensor *vSrc1 = vTensor->src[1];
  struct ggml_tensor *vSrc2 = vTensor->src[2];
  struct ggml_hash_set *vHashSet = &pCgraph->visited_hash_set;
  const size_t cIsrc0 = vSrc0 ? ggml_hash_find(vHashSet, vSrc0) : (size_t)-1;
  const size_t cIsrc1 = vSrc1 ? ggml_hash_find(vHashSet, vSrc1) : (size_t)-1;
  const size_t cIsrc2 = vSrc2 ? ggml_hash_find(vHashSet, vSrc2) : (size_t)-1;
  const bool cSrc0NeedsGrads =
      vSrc0 && cIsrc0 != GGML_HASHSET_FULL && ggml_bitset_get(vHashSet->used, cIsrc0) && pGradsNeeded[cIsrc0];
  const bool cSrc1NeedsGrads =
      vSrc1 && cIsrc1 != GGML_HASHSET_FULL && ggml_bitset_get(vHashSet->used, cIsrc1) && pGradsNeeded[cIsrc1];
  const bool cSrc2NeedsGrads =
      vSrc2 && cIsrc2 != GGML_HASHSET_FULL && ggml_bitset_get(vHashSet->used, cIsrc2) && pGradsNeeded[cIsrc2];

  switch (vTensor->op) {
  case GGML_OP_DUP: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
  } break;
  case GGML_OP_ADD: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
    if (cSrc1NeedsGrads) {
      struct ggml_tensor *vTmp = vGrad;
      if (!ggml_are_same_shape(vSrc0, vSrc1)) {
        vTmp = ggml_repeat_back(pCtx, vTmp, vSrc1);
      }
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, vTmp);
    }
  } break;
  case GGML_OP_ADD1: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
    if (cSrc1NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, ggml_mean(pCtx, vGrad)); // TODO: should probably be sum instead of mean
    }
  } break;
  case GGML_OP_ACC: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
    if (cSrc1NeedsGrads) {
      const size_t cNb1 = ((int32_t *)vTensor->op_params)[0];
      const size_t cNb2 = ((int32_t *)vTensor->op_params)[1];
      const size_t cNb3 = ((int32_t *)vTensor->op_params)[2];
      const size_t cOffset = ((int32_t *)vTensor->op_params)[3];

      struct ggml_tensor *vTensorGradView =
          ggml_view_4d(pCtx, vGrad, vSrc1->ne[0], vSrc1->ne[1], vSrc1->ne[2], vSrc1->ne[3], cNb1, cNb2, cNb3, cOffset);

      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, ggml_reshape(pCtx, ggml_cont(pCtx, vTensorGradView), vSrc1));
    }
  } break;
  case GGML_OP_SUB: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
    if (cSrc1NeedsGrads) {
      fGgmlSubOrSet(pCtx, pCgraph, cIsrc1, vGrad);
    }
  } break;
  case GGML_OP_MUL: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vGrad, vSrc1));
    }
    if (cSrc1NeedsGrads) {
      struct ggml_tensor *vTmp = ggml_mul(pCtx, vSrc0, vGrad);
      if (!ggml_are_same_shape(vSrc0, vSrc1)) {
        vTmp = ggml_repeat_back(pCtx, vTmp, vSrc1);
      }
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, vTmp);
    }
  } break;
  case GGML_OP_DIV: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_div(pCtx, vGrad, vSrc1));
    }
    if (cSrc1NeedsGrads) {
      fGgmlSubOrSet(pCtx, pCgraph, cIsrc1, ggml_mul(pCtx, vGrad, ggml_div(pCtx, vTensor, vSrc1)));
    }
  } break;
  case GGML_OP_SQR: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_scale(pCtx, ggml_mul(pCtx, vSrc0, vGrad), 2.0f));
    }
  } break;
  case GGML_OP_SQRT: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_scale(pCtx, ggml_div(pCtx, vGrad, vTensor), 0.5f));
    }
  } break;
  case GGML_OP_LOG: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_div(pCtx, vGrad, vSrc0));
    }
  } break;
  case GGML_OP_SIN: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vGrad, ggml_cos(pCtx, vSrc0)));
    }
  } break;
  case GGML_OP_COS: {
    if (cSrc0NeedsGrads) {
      fGgmlSubOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vGrad, ggml_sin(pCtx, vSrc0)));
    }
  } break;
  case GGML_OP_SUM: {
    if (cSrc0NeedsGrads) {
      fGgmlAdd1OrSet(pCtx, pCgraph, cIsrc0, vGrad);
    }
  } break;
  case GGML_OP_SUM_ROWS: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_repeat(pCtx, vGrad, vSrc0));
    }
  } break;
  case GGML_OP_MEAN: {
    if (cSrc0NeedsGrads) {
      fGgmlAdd1OrSet(pCtx, pCgraph, cIsrc0, fGgmlScaleImpl(pCtx, vGrad, 1.0f / vSrc0->ne[0], 0.0, false));
    }
  } break;
  case GGML_OP_REPEAT: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_repeat_back(pCtx, vGrad, vSrc0));
    }
  } break;
  case GGML_OP_REPEAT_BACK: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_repeat(pCtx, vGrad, vSrc0));
    }
  } break;
  case GGML_OP_RMS_NORM: {
    if (cSrc0NeedsGrads) {
      float vEps;
      memcpy(&vEps, vTensor->op_params, sizeof(float));
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_rms_norm_back(pCtx, vGrad, vSrc0, vEps));
    }
  } break;
  case GGML_OP_MUL_MAT: {
    // https://cs231n.github.io/optimization-2/#staged
    // # forward pass
    // s0 = np.random.randn(5, 10)
    // s1 = np.random.randn(10, 3)
    // t = s0.dot(s1)

    // # now suppose we had the gradient on t from above in the circuit
    // dt = np.random.randn(*t.shape) # same shape as t
    // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
    // ds1 = t.T.dot(dt)

    // tensor.shape [m,p,qq,rr]
    // src0.shape   [n,m,q1,r1]
    // src1.shape   [n,p,qq,rr]

    if (cSrc0NeedsGrads) {
      GGML_ASSERT(vGrad->ne[2] == vSrc1->ne[2]);
      GGML_ASSERT(vGrad->ne[3] == vSrc1->ne[3]);
      struct ggml_tensor *vTmp = ggml_out_prod(pCtx,   // [n,m,qq,rr]
                                               vSrc1,  // [n,p,qq,rr]
                                               vGrad); // [m,p,qq,rr]
      if (!ggml_are_same_shape(vTmp, vSrc0)) {
        GGML_ASSERT(vTmp->ne[0] == vSrc0->ne[0]);
        GGML_ASSERT(vTmp->ne[1] == vSrc0->ne[1]);
        GGML_ASSERT(vTmp->ne[3] == 1);

        const int64_t cNr2 = vTmp->ne[2] / vSrc0->ne[2];
        const size_t cNb2 = vTmp->nb[2] * cNr2;
        const size_t cNb3 = vTmp->nb[2];

        vTmp = ggml_view_4d(pCtx, vTmp, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2], cNr2, vTmp->nb[1], cNb2, cNb3, 0);
        vTmp = ggml_repeat_back(pCtx, vTmp, vSrc0);
      }
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vTmp);
    }
    if (cSrc1NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1,
                    // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                    //     ggml_cont(ctx,                  // [m,n,q1,r1]
                    //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                    //     grad),                          // [m,p,qq,rr]

                    // when src0 is bigger than tensor->grad (this is mostly the case in mgs),
                    // avoid transpose of src0, rather transpose smaller tensor->grad
                    // and then use ggml_out_prod
                    ggml_out_prod(pCtx,                    // [n,p,qq,rr]
                                  vSrc0,                   // [n,m,q1,r1]
                                  ggml_transpose(pCtx,     // [p,m,qq,rr]
                                                 vGrad))); // [m,p,qq,rr]
    }
  } break;
  case GGML_OP_SCALE: {
    if (cSrc0NeedsGrads) {
      float vS;
      memcpy(&vS, vTensor->op_params, sizeof(float));
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, fGgmlScaleImpl(pCtx, vGrad, vS, 0.0, false));
    }
  } break;
  case GGML_OP_SET: {
    const size_t cNb1 = ((const int32_t *)vTensor->op_params)[0];
    const size_t cNb2 = ((const int32_t *)vTensor->op_params)[1];
    const size_t cNb3 = ((const int32_t *)vTensor->op_params)[2];
    const size_t cOffset = ((const int32_t *)vTensor->op_params)[3];

    struct ggml_tensor *vTensorGradView = NULL;

    if (cSrc0NeedsGrads || cSrc1NeedsGrads) {
      GGML_ASSERT(vSrc0->type == vTensor->type);
      GGML_ASSERT(!pCgraph->grads[cIsrc0] || pCgraph->grads[cIsrc0]->type == vGrad->type);
      GGML_ASSERT(!pCgraph->grads[cIsrc1] || !cSrc1NeedsGrads || pCgraph->grads[cIsrc1]->type == vGrad->type);

      vTensorGradView =
          ggml_view_4d(pCtx, vGrad, vSrc1->ne[0], vSrc1->ne[1], vSrc1->ne[2], vSrc1->ne[3], cNb1, cNb2, cNb3, cOffset);
    }

    if (cSrc0NeedsGrads) {
      struct ggml_tensor *vTmp = ggml_neg(pCtx, vTensorGradView);
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, fGgmlAccImpl(pCtx, vGrad, vTmp, cNb1, cNb2, cNb3, cOffset, false));
    }

    if (cSrc1NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, ggml_reshape(pCtx, ggml_cont(pCtx, vTensorGradView), vSrc1));
    }
  } break;
  case GGML_OP_CPY: {
    // cpy overwrites value of src1 by src0 and returns view(src1)
    // the overwriting is mathematically equivalent to:
    // tensor = src0 * 1 + src1 * 0
    if (cSrc0NeedsGrads) {
      // dsrc0 = dtensor * 1
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_reshape(pCtx, vGrad, vSrc0));
    }
    if (cSrc1NeedsGrads) {
      // dsrc1 = dtensor * 0 -> noop
    }
  } break;
  case GGML_OP_CONT: {
    // same as cpy
    if (cSrc0NeedsGrads) {
      GGML_ASSERT(!pCgraph->grads[cIsrc0] || ggml_is_contiguous(pCgraph->grads[cIsrc0]));
      GGML_ASSERT(ggml_is_contiguous(vGrad));
      GGML_ASSERT(ggml_nelements(vTensor) == ggml_nelements(vSrc0));
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0,
                    ggml_are_same_shape(vTensor, vSrc0) ? vGrad : ggml_reshape(pCtx, vGrad, vSrc0));
    }
  } break;
  case GGML_OP_RESHAPE: {
    if (cSrc0NeedsGrads) {
      struct ggml_tensor *vGradCont = ggml_is_contiguous(vGrad) ? vGrad : ggml_cont(pCtx, vGrad);
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_reshape(pCtx, vGradCont, vSrc0));
    }
  } break;
  case GGML_OP_VIEW: {
    if (cSrc0NeedsGrads) {
      size_t vOffset;

      memcpy(&vOffset, vTensor->op_params, sizeof(vOffset));

      size_t vNb1 = vTensor->nb[1];
      size_t vNb2 = vTensor->nb[2];
      size_t vNb3 = vTensor->nb[3];

      if (pCgraph->grads[cIsrc0] && vSrc0->type != pCgraph->grads[cIsrc0]->type) {
        // gradient is typically F32, but src0 could be other type
        size_t vNg = ggml_element_size(pCgraph->grads[cIsrc0]);
        size_t vN0 = ggml_element_size(vSrc0);
        GGML_ASSERT(vOffset % vN0 == 0);
        GGML_ASSERT(vNb1 % vN0 == 0);
        GGML_ASSERT(vNb2 % vN0 == 0);
        GGML_ASSERT(vNb3 % vN0 == 0);
        vOffset = (vOffset / vN0) * vNg;
        vNb1 = (vNb1 / vN0) * vNg;
        vNb2 = (vNb2 / vN0) * vNg;
        vNb3 = (vNb3 / vN0) * vNg;
      }

      fGgmlAccOrSet(pCtx, pCgraph, cIsrc0, vGrad, vNb1, vNb2, vNb3, vOffset);
    }
  } break;
  case GGML_OP_PERMUTE: {
    if (cSrc0NeedsGrads) {
      const int32_t *vAxes = (const int32_t *)vTensor->op_params;
      const int cAxis0 = vAxes[0] & 0x3;
      const int cAxis1 = vAxes[1] & 0x3;
      const int cAxis2 = vAxes[2] & 0x3;
      const int cAxis3 = vAxes[3] & 0x3;
      int aAxb[4] = {0, 0, 0, 0}; // axes backward
      aAxb[cAxis0] = 0;
      aAxb[cAxis1] = 1;
      aAxb[cAxis2] = 2;
      aAxb[cAxis3] = 3;
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_permute(pCtx, vGrad, aAxb[0], aAxb[1], aAxb[2], aAxb[3]));
    }
  } break;
  case GGML_OP_TRANSPOSE: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_transpose(pCtx, vGrad));
    }
  } break;
  case GGML_OP_GET_ROWS: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_get_rows_back(pCtx, vGrad, vSrc1, vSrc0));
    }
    if (cSrc1NeedsGrads) {
      // noop
    }
  } break;
  case GGML_OP_DIAG_MASK_INF: {
    if (cSrc0NeedsGrads) {
      /* ggml_diag_mask_inf_impl() shouldn't be here */
      const int cNPast = ((const int32_t *)vTensor->op_params)[0];
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, fGgmlDiagMaskZeroImpl(pCtx, vGrad, cNPast, false));
    }
  } break;
  case GGML_OP_DIAG_MASK_ZERO: {
    if (cSrc0NeedsGrads) {
      const int cNPast = ((const int32_t *)vTensor->op_params)[0];
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, fGgmlDiagMaskZeroImpl(pCtx, vGrad, cNPast, false));
    }
  } break;
  case GGML_OP_SOFT_MAX: {
    if (cSrc0NeedsGrads) {
      float vScale = 1.0f;
      float vMaxBias = 0.0f;

      memcpy(&vScale, (const float *)vTensor->op_params + 0, sizeof(float));
      memcpy(&vMaxBias, (const float *)vTensor->op_params + 1, sizeof(float));

      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_soft_max_ext_back(pCtx, vGrad, vTensor, vScale, vMaxBias));
    }
    GGML_ASSERT((!vSrc1 || !cSrc1NeedsGrads) && "backward pass for softmax mask not implemented");
  } break;
  case GGML_OP_ROPE: {
    if (cSrc0NeedsGrads) {
      // const int n_past = ((int32_t *) tensor->op_params)[0];
      const int cNDims = ((const int32_t *)vTensor->op_params)[1];
      const int cMode = ((const int32_t *)vTensor->op_params)[2];
      // const int n_ctx      = ((int32_t *) tensor->op_params)[3];
      const int cNCtxOrig = ((const int32_t *)vTensor->op_params)[4];
      float vFreqBase, vFreqScale, vExtFactor, vAttnFactor, vBetaFast, vBetaSlow;
      int aSections[4] = {0, 0, 0, 0};

      memcpy(&vFreqBase, (const float *)vTensor->op_params + 5, sizeof(float));
      memcpy(&vFreqScale, (const float *)vTensor->op_params + 6, sizeof(float));
      memcpy(&vExtFactor, (const float *)vTensor->op_params + 7, sizeof(float));
      memcpy(&vAttnFactor, (const float *)vTensor->op_params + 8, sizeof(float));
      memcpy(&vBetaFast, (const float *)vTensor->op_params + 9, sizeof(float));
      memcpy(&vBetaSlow, (const float *)vTensor->op_params + 10, sizeof(float));
      memcpy(&aSections, vTensor->op_params + 11, sizeof(aSections));

      struct ggml_tensor *vRopeBack =
          vGrad->ne[2] == vSrc1->ne[0]
              ? ggml_rope_ext_back(pCtx, vGrad, vSrc1, vSrc2, cNDims, cMode, cNCtxOrig, vFreqBase, vFreqScale,
                                   vExtFactor, vAttnFactor, vBetaFast, vBetaSlow)
              : ggml_rope_multi_back(pCtx, vGrad, vSrc1, vSrc2, cNDims, aSections, cMode, cNCtxOrig, vFreqBase,
                                     vFreqScale, vExtFactor, vAttnFactor, vBetaFast, vBetaSlow);
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, vRopeBack);
    }
    GGML_ASSERT((!vSrc2 || !cSrc2NeedsGrads) && "gradients for freq factors not implemented");
  } break;
  case GGML_OP_IM2COL: {
    if (cSrc1NeedsGrads) {
      const int32_t cS0 = ggml_get_op_params_i32(vTensor, 0);
      const int32_t cS1 = ggml_get_op_params_i32(vTensor, 1);
      const int32_t cP0 = ggml_get_op_params_i32(vTensor, 2);
      const int32_t cP1 = ggml_get_op_params_i32(vTensor, 3);
      const int32_t cD0 = ggml_get_op_params_i32(vTensor, 4);
      const int32_t cD1 = ggml_get_op_params_i32(vTensor, 5);
      const bool cIs2D = ggml_get_op_params_i32(vTensor, 6) == 1;

      fGgmlAddOrSet(pCtx, pCgraph, cIsrc1,
                    ggml_im2col_back(pCtx, vGrad, vSrc0, vSrc1->ne, cS0, cS1, cP0, cP1, cD0, cD1, cIs2D));
    }
  } break;
  case GGML_OP_POOL_2D: {
    if (cSrc0NeedsGrads) {
      const enum ggml_op_pool cOp = ggml_get_op_params_i32(vTensor, 0);
      const int32_t cK0 = ggml_get_op_params_i32(vTensor, 1);
      const int32_t cK1 = ggml_get_op_params_i32(vTensor, 2);
      const int32_t cS0 = ggml_get_op_params_i32(vTensor, 3);
      const int32_t cS1 = ggml_get_op_params_i32(vTensor, 4);
      const int32_t cP0 = ggml_get_op_params_i32(vTensor, 5);
      const int32_t cP1 = ggml_get_op_params_i32(vTensor, 6);

      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_pool_2d_back(pCtx, vGrad, vSrc0, cOp, cK0, cK1, cS0, cS1, cP0, cP1));
    }
  } break;
  case GGML_OP_WIN_PART:
  case GGML_OP_WIN_UNPART:
  case GGML_OP_UNARY: {
    switch (ggml_get_unary_op(vTensor)) {
    case GGML_UNARY_OP_ABS: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, ggml_sgn(pCtx, vSrc0), vGrad));
      }
    } break;
    case GGML_UNARY_OP_SGN: {
      // noop
    } break;
    case GGML_UNARY_OP_NEG: {
      if (cSrc0NeedsGrads) {
        fGgmlSubOrSet(pCtx, pCgraph, cIsrc0, vGrad);
      }
    } break;
    case GGML_UNARY_OP_STEP: {
      // noop
    } break;
    case GGML_UNARY_OP_RELU: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, ggml_step(pCtx, vSrc0), vGrad));
      }
    } break;
    case GGML_UNARY_OP_SILU: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_silu_back(pCtx, vGrad, vSrc0));
      }
    } break;
    case GGML_UNARY_OP_EXP: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vTensor, vGrad));
      }
    } break;
    case GGML_UNARY_OP_EXPM1: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vGrad, ggml_exp(pCtx, vSrc0)));
      }
    } break;
    case GGML_UNARY_OP_SOFTPLUS: {
      if (cSrc0NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_mul(pCtx, vGrad, ggml_sigmoid(pCtx, vSrc0)));
      }
    } break;
    default: {
      fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n", __func__,
              ggml_unary_op_name(ggml_get_unary_op(vTensor)));
      GGML_ABORT("fatal error");
    } // break;
    }
  } break;
  case GGML_OP_CROSS_ENTROPY_LOSS: {
    if (cSrc0NeedsGrads) {
      fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_cross_entropy_loss_back(pCtx, vGrad, vSrc0, vSrc1));
    }
    GGML_ASSERT(!cSrc1NeedsGrads && "backward pass for labels not implemented");
  } break;
  case GGML_OP_GLU: {
    switch (ggml_get_glu_op(vTensor)) {
    case GGML_GLU_OP_SWIGLU: {
      if (cSrc0NeedsGrads) {
        GGML_ASSERT(vSrc1 && "backward pass only implemented for split swiglu");
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc0, ggml_silu_back(pCtx, ggml_mul(pCtx, vGrad, vSrc1), vSrc0));
      }
      if (cSrc1NeedsGrads) {
        fGgmlAddOrSet(pCtx, pCgraph, cIsrc1, ggml_mul(pCtx, ggml_silu(pCtx, vSrc0), vGrad));
      }
    } break;
    default: {
      GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(vTensor)));
    } // break;
    }
  } break;
  case GGML_OP_NONE: {
    // noop
  } break;
  case GGML_OP_COUNT:
  default: {
    GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(vTensor->op));
  } // break;
  }

  GGML_ASSERT(!cSrc0NeedsGrads || ggml_are_same_shape(vSrc0, pCgraph->grads[cIsrc0]));
  GGML_ASSERT(!cSrc1NeedsGrads || ggml_are_same_shape(vSrc1, pCgraph->grads[cIsrc1]));
  GGML_ASSERT(!cSrc2NeedsGrads || ggml_are_same_shape(vSrc2, pCgraph->grads[cIsrc2]));
}

static size_t fGgmlVisitParentsGraph(struct ggml_cgraph *pCgraph, struct ggml_tensor *pNode, bool pCompute) {
  if (pNode->op != GGML_OP_NONE && pCompute) {
    pNode->flags |= GGML_TENSOR_FLAG_COMPUTE;
  }

  const size_t cNodeHashPos = ggml_hash_find(&pCgraph->visited_hash_set, pNode);
  GGML_ASSERT(cNodeHashPos != GGML_HASHSET_FULL);

  if (ggml_bitset_get(pCgraph->visited_hash_set.used, cNodeHashPos)) {
    // already visited

    if (pCompute) {
      // update the compute flag regardless
      for (int vI = 0; vI < GGML_MAX_SRC; ++vI) {
        struct ggml_tensor *vSrc = pNode->src[vI];
        if (vSrc && ((vSrc->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
          fGgmlVisitParentsGraph(pCgraph, vSrc, true);
        }
      }
    }

    return cNodeHashPos;
  }

  // This is the first time we see this node in the current graph.
  pCgraph->visited_hash_set.keys[cNodeHashPos] = pNode;
  ggml_bitset_set(pCgraph->visited_hash_set.used, cNodeHashPos);
  pCgraph->use_counts[cNodeHashPos] = 0;

  for (int vI = 0; vI < GGML_MAX_SRC; ++vI) {
    const int cK = (pCgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? vI
                   : (pCgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT)
                       ? (GGML_MAX_SRC - 1 - vI)
                       :
                       /* unknown order, just fall back to using i */ vI;

    struct ggml_tensor *vSrc = pNode->src[cK];
    if (vSrc) {
      const size_t cSrcHashPos = fGgmlVisitParentsGraph(pCgraph, vSrc, pCompute);

      // Update the use count for this operand.
      pCgraph->use_counts[cSrcHashPos]++;
    }
  }

  if (pNode->op == GGML_OP_NONE && !(pNode->flags & GGML_TENSOR_FLAG_PARAM)) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)
    GGML_ASSERT(pCgraph->n_leafs < pCgraph->size);

    if (strlen(pNode->name) == 0) {
      ggml_format_name(pNode, "leaf_%d", pCgraph->n_leafs);
    }

    pCgraph->leafs[pCgraph->n_leafs] = pNode;
    pCgraph->n_leafs++;
  } else {
    GGML_ASSERT(pCgraph->n_nodes < pCgraph->size);

    if (strlen(pNode->name) == 0) {
      ggml_format_name(pNode, "node_%d", pCgraph->n_nodes);
    }

    pCgraph->nodes[pCgraph->n_nodes] = pNode;
    pCgraph->n_nodes++;
  }

  return cNodeHashPos;
}

static void fGgmlBuildForwardImpl(struct ggml_cgraph *pCgraph, struct ggml_tensor *pTensor, bool pExpand,
                                  bool pCompute) {
  if (!pExpand) {
    // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
    ggml_graph_clear(pCgraph);
  }

  const int cNOld = pCgraph->n_nodes;

  fGgmlVisitParentsGraph(pCgraph, pTensor, pCompute);

  const int cNNew = pCgraph->n_nodes - cNOld;
  GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, cNNew);

  if (cNNew > 0) {
    // the last added node should always be starting point
    GGML_ASSERT(pCgraph->nodes[pCgraph->n_nodes - 1] == pTensor);
  }
}

struct ggml_tensor *ggml_build_forward_select(struct ggml_cgraph *pCgraph, struct ggml_tensor **pTensors, int pNTensors,
                                              int pIdx) {
  GGML_ASSERT(pIdx >= 0 && pIdx < pNTensors);

  for (int vI = 0; vI < pNTensors; vI++) {
    fGgmlBuildForwardImpl(pCgraph, pTensors[vI], true, vI == pIdx ? true : false);
  }

  return pTensors[pIdx];
}

void ggml_build_forward_expand(struct ggml_cgraph *pCgraph, struct ggml_tensor *pTensor) {
  fGgmlBuildForwardImpl(pCgraph, pTensor, true, true);
}

void ggml_build_backward_expand(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph,
                                struct ggml_tensor **pGradAccs) {
  GGML_ASSERT(pCgraph->n_nodes > 0);
  GGML_ASSERT(pCgraph->grads);
  GGML_ASSERT(pCgraph->grad_accs);

  const int cNNodesF = pCgraph->n_nodes;

  memset(pCgraph->grads, 0, pCgraph->visited_hash_set.size * sizeof(struct ggml_tensor *));
  memset(pCgraph->grad_accs, 0, pCgraph->visited_hash_set.size * sizeof(struct ggml_tensor *));
  bool *vGradsNeeded = calloc(pCgraph->visited_hash_set.size, sizeof(bool));

  {
    bool vAnyParams = false;
    bool vAnyLoss = false;
    for (int vI = 0; vI < cNNodesF; ++vI) {
      struct ggml_tensor *vNode = pCgraph->nodes[vI];
      vAnyParams = vAnyParams || (vNode->flags & GGML_TENSOR_FLAG_PARAM);
      vAnyLoss = vAnyLoss || (vNode->flags & GGML_TENSOR_FLAG_LOSS);
    }
    GGML_ASSERT(vAnyParams && "no trainable parameters found, did you forget to call ggml_set_param?");
    GGML_ASSERT(vAnyLoss && "no training loss found, did you forget to call ggml_set_loss?");
  }

  for (int vI = 0; vI < cNNodesF; ++vI) {
    struct ggml_tensor *vNode = pCgraph->nodes[vI];

    if (vNode->type == GGML_TYPE_I32) {
      continue;
    }

    bool vNodeNeedsGrad = (vNode->flags & GGML_TENSOR_FLAG_PARAM) || (vNode->flags & GGML_TENSOR_FLAG_LOSS);
    bool vIgnoreSrc[GGML_MAX_SRC] = {false};
    switch (vNode->op) {
    // gradients in node->src[0] for one reason or another have no effect on output gradients
    case GGML_OP_IM2COL:      // only used for its shape
    case GGML_OP_IM2COL_BACK: // same as IM2COL
      vIgnoreSrc[0] = true;
      break;
    case GGML_OP_UNARY: {
      const enum ggml_unary_op cUop = ggml_get_unary_op(vNode);
      // SGN and STEP unary ops are piecewise constant
      if (cUop == GGML_UNARY_OP_SGN || cUop == GGML_UNARY_OP_STEP) {
        vIgnoreSrc[0] = true;
      }
    } break;

    // gradients in node->src[1] for one reason or another have no effect on output gradients
    case GGML_OP_CPY:           // gradients in CPY target are irrelevant
    case GGML_OP_GET_ROWS:      // row indices not differentiable
    case GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
    case GGML_OP_ROPE:          // positions not differentiable
      vIgnoreSrc[1] = true;
      break;

    default:
      break;
    }
    for (int vJ = 0; vJ < GGML_MAX_SRC; ++vJ) {
      if (!vNode->src[vJ] || vIgnoreSrc[vJ] ||
          !vGradsNeeded[ggml_hash_find(&pCgraph->visited_hash_set, vNode->src[vJ])]) {
        continue;
      }
      GGML_ASSERT(vNode->src[vJ]->type == GGML_TYPE_F32 || vNode->src[vJ]->type == GGML_TYPE_F16);
      vNodeNeedsGrad = true;
      break;
    }
    if (!vNodeNeedsGrad) {
      continue;
    }

    // inplace operations are currently not supported
    GGML_ASSERT(!vNode->view_src || vNode->op == GGML_OP_CPY || vNode->op == GGML_OP_VIEW ||
                vNode->op == GGML_OP_RESHAPE || vNode->op == GGML_OP_PERMUTE || vNode->op == GGML_OP_TRANSPOSE);

    const size_t cIhash = ggml_hash_find(&pCgraph->visited_hash_set, vNode);
    GGML_ASSERT(cIhash != GGML_HASHSET_FULL);
    GGML_ASSERT(ggml_bitset_get(pCgraph->visited_hash_set.used, cIhash));
    if (pGradAccs && pGradAccs[vI]) {
      pCgraph->grad_accs[cIhash] = pGradAccs[vI];
      pCgraph->grads[cIhash] = pCgraph->grad_accs[cIhash];
    } else if (vNode->flags & GGML_TENSOR_FLAG_LOSS) {
      // loss tensors always need a gradient accumulator
      pCgraph->grad_accs[cIhash] = ggml_new_tensor(pCtx, GGML_TYPE_F32, GGML_MAX_DIMS, vNode->ne);
      pCgraph->grads[cIhash] = pCgraph->grad_accs[cIhash];
    }
    vGradsNeeded[cIhash] = true;
  }

  for (int vI = cNNodesF - 1; vI >= 0; --vI) {
    // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
    // use allocator to automatically make inplace operations
    fGgmlComputeBackward(pCtx, pCgraph, vI, vGradsNeeded);
  }

  free(vGradsNeeded);
}

static void *fIncrPtrAligned(void **p_, size_t pSize, size_t pAlign) {
  void *vPtr = *p_;
  vPtr = (void *)GGML_PAD((uintptr_t)vPtr, pAlign);
  *p_ = (void *)((char *)vPtr + pSize);
  return vPtr;
}

static size_t fGgmlGraphNbytes(size_t pSize, bool pGrads) {
  size_t vHashSize = ggml_hash_size(pSize * 2);
  void *vP = 0;
  fIncrPtrAligned(&vP, sizeof(struct ggml_cgraph), 1);
  fIncrPtrAligned(&vP, pSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));     // nodes
  fIncrPtrAligned(&vP, pSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));     // leafs
  fIncrPtrAligned(&vP, vHashSize * sizeof(int32_t), sizeof(int32_t));                           // use_counts
  fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // hash keys
  if (pGrads) {
    fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grads
    fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grad_accs
  }
  fIncrPtrAligned(&vP, ggml_bitset_size(vHashSize) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));

  size_t vNbytes = (size_t)vP;
  return vNbytes;
}

size_t ggml_graph_overhead_custom(size_t pSize, bool pGrads) {
  return GGML_OBJECT_SIZE + GGML_PAD(fGgmlGraphNbytes(pSize, pGrads), GGML_MEM_ALIGN);
}

size_t ggml_graph_overhead(void) { return ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false); }

struct ggml_cgraph *ggml_new_graph_custom(struct ggml_context *pCtx, size_t pSize, bool pGrads) {
  const size_t cObjSize = fGgmlGraphNbytes(pSize, pGrads);
  struct ggml_object *vObj = fGgmlNewObject(pCtx, GGML_OBJECT_TYPE_GRAPH, cObjSize);
  struct ggml_cgraph *vCgraph = (struct ggml_cgraph *)((char *)pCtx->vMemBuffer + vObj->vOffs);

  // the size of the hash table is doubled since it needs to hold both nodes and leafs
  size_t vHashSize = ggml_hash_size(pSize * 2);

  void *vP = vCgraph + 1;

  struct ggml_tensor **vNodesPtr =
      fIncrPtrAligned(&vP, pSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
  struct ggml_tensor **vLeafsPtr =
      fIncrPtrAligned(&vP, pSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
  int32_t *vUseCountsPtr = fIncrPtrAligned(&vP, vHashSize * sizeof(int32_t), sizeof(int32_t));
  struct ggml_tensor **vHashKeysPtr =
      fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *));
  struct ggml_tensor **vGradsPtr =
      pGrads ? fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)) : NULL;
  struct ggml_tensor **vGradAccsPtr =
      pGrads ? fIncrPtrAligned(&vP, vHashSize * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)) : NULL;

  ggml_bitset_t *vHashUsed =
      fIncrPtrAligned(&vP, ggml_bitset_size(vHashSize) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));

  // check that we allocated the correct amount of memory
  assert(cObjSize == (size_t)((char *)vP - (char *)vCgraph));

  *vCgraph = (struct ggml_cgraph){
      /*.size         =*/pSize,
      /*.n_nodes      =*/0,
      /*.n_leafs      =*/0,
      /*.nodes        =*/vNodesPtr,
      /*.grads        =*/vGradsPtr,
      /*.grad_accs    =*/vGradAccsPtr,
      /*.leafs        =*/vLeafsPtr,
      /*.use_counts   =*/vUseCountsPtr,
      /*.hash_table   =*/{vHashSize, vHashUsed, vHashKeysPtr},
      /*.order        =*/GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
      /*.uid          =*/0,
  };

  ggml_hash_set_reset(&vCgraph->visited_hash_set);
  if (pGrads) {
    memset(vCgraph->grads, 0, vHashSize * sizeof(struct ggml_tensor *));
    memset(vCgraph->grad_accs, 0, vHashSize * sizeof(struct ggml_tensor *));
  }

  return vCgraph;
}

struct ggml_cgraph *ggml_new_graph(struct ggml_context *pCtx) {
  return ggml_new_graph_custom(pCtx, GGML_DEFAULT_GRAPH_SIZE, false);
}

struct ggml_cgraph ggml_graph_view(struct ggml_cgraph *pCgraph0, int pI0, int pI1) {
  struct ggml_cgraph vCgraph = {/*.size             =*/0,
                                /*.n_nodes          =*/pI1 - pI0,
                                /*.n_leafs          =*/0,
                                /*.nodes            =*/pCgraph0->nodes + pI0,
                                /*.grads            =*/NULL, // gradients would need visited_hash_set
                                /*.grad_accs        =*/NULL,
                                /*.leafs            =*/NULL,
                                /*.use_counts       =*/pCgraph0->use_counts,
                                /*.visited_hash_set =*/pCgraph0->visited_hash_set,
                                /*.order            =*/pCgraph0->order,
                                /*.uid              =*/0};

  return vCgraph;
}

void ggml_graph_cpy(struct ggml_cgraph *pSrc, struct ggml_cgraph *pDst) {
  GGML_ASSERT(pDst->size >= pSrc->n_leafs);
  GGML_ASSERT(pDst->size >= pSrc->n_nodes);
  GGML_ASSERT(pDst->visited_hash_set.size >= pSrc->visited_hash_set.size);

  pDst->n_leafs = pSrc->n_leafs;
  pDst->n_nodes = pSrc->n_nodes;
  pDst->order = pSrc->order;

  for (int vI = 0; vI < pSrc->n_leafs; ++vI) {
    pDst->leafs[vI] = pSrc->leafs[vI];
  }

  for (int vI = 0; vI < pSrc->n_nodes; ++vI) {
    pDst->nodes[vI] = pSrc->nodes[vI];
  }

  for (size_t vI = 0; vI < pSrc->visited_hash_set.size; ++vI) {
    // copy all hashset keys (tensors) that are in use
    if (ggml_bitset_get(pSrc->visited_hash_set.used, vI)) {
      size_t vNewHashPos = ggml_hash_insert(&pDst->visited_hash_set, pSrc->visited_hash_set.keys[vI]);
      pDst->use_counts[vNewHashPos] = pSrc->use_counts[vI];
    }
  }

  if (pDst->grads) {
    memset(pDst->grads, 0, pDst->visited_hash_set.size * sizeof(struct ggml_tensor *));
    memset(pDst->grad_accs, 0, pDst->visited_hash_set.size * sizeof(struct ggml_tensor *));
  }
  if (pSrc->grads) {
    GGML_ASSERT(pDst->grads != NULL);
    GGML_ASSERT(pDst->grad_accs != NULL);
    for (int vI = 0; vI < pSrc->n_nodes; ++vI) {
      const size_t cIgradSrc = ggml_hash_find(&pSrc->visited_hash_set, pSrc->nodes[vI]);
      const size_t cIgradDst = ggml_hash_find(&pDst->visited_hash_set, pDst->nodes[vI]);

      GGML_ASSERT(cIgradSrc != GGML_HASHSET_FULL);
      GGML_ASSERT(ggml_bitset_get(pSrc->visited_hash_set.used, cIgradSrc));
      GGML_ASSERT(cIgradDst != GGML_HASHSET_FULL);
      GGML_ASSERT(ggml_bitset_get(pDst->visited_hash_set.used, cIgradDst));

      pDst->grads[cIgradDst] = pSrc->grads[cIgradSrc];
      pDst->grad_accs[cIgradDst] = pSrc->grad_accs[cIgradSrc];
    }
  }
}

struct ggml_cgraph *ggml_graph_dup(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, bool pForceGrads) {
  struct ggml_cgraph *vResult = ggml_new_graph_custom(pCtx, pCgraph->size, pCgraph->grads || pForceGrads);
  ggml_graph_cpy(pCgraph, vResult);
  return vResult;
}

struct ggml_tensor *ggml_set_zero(struct ggml_tensor *pTensor) {
  if (ggml_is_empty(pTensor)) {
    return pTensor;
  }
  if (pTensor->buffer) {
    ggml_backend_tensor_memset(pTensor, 0, 0, ggml_nbytes(pTensor));
  } else {
    GGML_ASSERT(pTensor->data);
    memset(pTensor->data, 0, ggml_nbytes(pTensor));
  }
  return pTensor;
}

void ggml_graph_reset(struct ggml_cgraph *pCgraph) {
  if (!pCgraph) {
    return;
  }
  GGML_ASSERT(pCgraph->grads != NULL);

  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pCgraph->nodes[vI];
    struct ggml_tensor *vGradAcc = ggml_graph_get_grad_acc(pCgraph, vNode);

    if (vNode->op == GGML_OP_OPT_STEP_ADAMW) {
      // clear momenta
      ggml_set_zero(vNode->src[2]);
      ggml_set_zero(vNode->src[3]);
    }

    // initial gradients of loss should be 1, 0 otherwise
    if (vGradAcc) {
      if (vNode->flags & GGML_TENSOR_FLAG_LOSS) {
        GGML_ASSERT(vGradAcc->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_scalar(vGradAcc));

        const float cOnef = 1.0f;
        if (vGradAcc->buffer) {
          ggml_backend_tensor_set(vGradAcc, &cOnef, 0, sizeof(float));
        } else {
          GGML_ASSERT(vGradAcc->data);
          *((float *)vGradAcc->data) = cOnef;
        }
      } else {
        ggml_set_zero(vGradAcc);
      }
    }
  }
}

void ggml_graph_clear(struct ggml_cgraph *pCgraph) {
  pCgraph->n_leafs = 0;
  pCgraph->n_nodes = 0;
  ggml_hash_set_reset(&pCgraph->visited_hash_set);
}

int ggml_graph_size(struct ggml_cgraph *pCgraph) { return pCgraph->size; }

struct ggml_tensor *ggml_graph_node(struct ggml_cgraph *pCgraph, int pI) {
  if (pI < 0) {
    GGML_ASSERT(pCgraph->n_nodes + pI >= 0);
    return pCgraph->nodes[pCgraph->n_nodes + pI];
  }

  GGML_ASSERT(pI < pCgraph->n_nodes);
  return pCgraph->nodes[pI];
}

struct ggml_tensor **ggml_graph_nodes(struct ggml_cgraph *pCgraph) { return pCgraph->nodes; }

int ggml_graph_n_nodes(struct ggml_cgraph *pCgraph) { return pCgraph->n_nodes; }

void ggml_graph_add_node(struct ggml_cgraph *pCgraph, struct ggml_tensor *pTensor) {
  GGML_ASSERT(pCgraph->size > pCgraph->n_nodes);
  pCgraph->nodes[pCgraph->n_nodes] = pTensor;
  pCgraph->n_nodes++;
}

struct ggml_tensor *ggml_graph_get_tensor(const struct ggml_cgraph *pCgraph, const char *pName) {
  for (int vI = 0; vI < pCgraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pCgraph->leafs[vI];

    if (strcmp(vLeaf->name, pName) == 0) {
      return vLeaf;
    }
  }

  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pCgraph->nodes[vI];

    if (strcmp(vNode->name, pName) == 0) {
      return vNode;
    }
  }

  return NULL;
}

struct ggml_tensor *ggml_graph_get_grad(const struct ggml_cgraph *pCgraph, const struct ggml_tensor *pNode) {
  const size_t cIgrad = ggml_hash_find(&pCgraph->visited_hash_set, pNode);
  return cIgrad != GGML_HASHSET_FULL && ggml_bitset_get(pCgraph->visited_hash_set.used, cIgrad) && pCgraph->grads
             ? pCgraph->grads[cIgrad]
             : NULL;
}

struct ggml_tensor *ggml_graph_get_grad_acc(const struct ggml_cgraph *pCgraph, const struct ggml_tensor *pNode) {
  const size_t cIgrad = ggml_hash_find(&pCgraph->visited_hash_set, pNode);
  return cIgrad != GGML_HASHSET_FULL && ggml_bitset_get(pCgraph->visited_hash_set.used, cIgrad) && pCgraph->grad_accs
             ? pCgraph->grad_accs[cIgrad]
             : NULL;
}

void ggml_graph_print(const struct ggml_cgraph *pCgraph) {
  GGML_LOG_INFO("=== GRAPH ===\n");

  GGML_LOG_INFO("n_nodes = %d\n", pCgraph->n_nodes);
  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pCgraph->nodes[vI];

    GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n", vI, vNode->ne[0], vNode->ne[1],
                  vNode->ne[2], ggml_op_name(vNode->op),
                  (vNode->flags & GGML_TENSOR_FLAG_PARAM) ? "x"
                  : ggml_graph_get_grad(pCgraph, vNode)   ? "g"
                                                          : " ");
  }

  GGML_LOG_INFO("n_leafs = %d\n", pCgraph->n_leafs);
  for (int vI = 0; vI < pCgraph->n_leafs; vI++) {
    struct ggml_tensor *vNode = pCgraph->leafs[vI];

    GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n", vI, vNode->ne[0], vNode->ne[1],
                  ggml_op_name(vNode->op), ggml_get_name(vNode));
  }

  GGML_LOG_INFO("========================================\n");
}

static int fGgmlNodeListFindTensor(const struct ggml_cgraph *pCgraph, const int *pIdxs, int pCount,
                                   const struct ggml_tensor *pTensor) {
  GGML_ASSERT(pCgraph && pIdxs);
  for (int vI = 0; vI < pCount; ++vI) {
    const int cNodeIdx = pIdxs[vI];

    if (cNodeIdx >= pCgraph->n_nodes) {
      return -1;
    }
    if (pCgraph->nodes[cNodeIdx] == pTensor) {
      return vI;
    }
  }
  return -1;
}

static bool fGgmlIsConstant(const struct ggml_tensor *pTensor) {
  return pTensor->buffer != NULL &&
         ggml_backend_buffer_get_usage(pTensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
         (pTensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph *pCgraph, const int *pNodeIdxs, int pCount,
                                const enum ggml_op *pOps, const int *pOutputs, int pNumOutputs) {
  GGML_ASSERT(pOutputs && pNumOutputs > 0);

  for (int vI = 0; vI < pCount; ++vI) {
    if (pNodeIdxs[vI] >= pCgraph->n_nodes) {
      return false;
    }

    const struct ggml_tensor *vNode = pCgraph->nodes[pNodeIdxs[vI]];

    if (vNode->op != pOps[vI]) {
      return false;
    }

    if ((vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
      return false;
    }

    if (fGgmlNodeListFindTensor(pCgraph, pOutputs, pNumOutputs, vNode) != -1) {
      continue;
    }

    if (vNode->flags & GGML_TENSOR_FLAG_OUTPUT) {
      return false;
    }

    int vSubgraphUses = 0;
    for (int vJ = vI + 1; vJ < pCount; ++vJ) {
      const struct ggml_tensor *vOtherNode = pCgraph->nodes[pNodeIdxs[vJ]];
      for (int vSrcIdx = 0; vSrcIdx < GGML_MAX_SRC; vSrcIdx++) {
        if (vOtherNode->src[vSrcIdx] == vNode) {
          vSubgraphUses++;
        }
      }
    }

    if (vSubgraphUses != ggml_node_get_use_count(pCgraph, pNodeIdxs[vI])) {
      return false;
    }

    // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
    // external view sources are allowed only for weight tensors, which are constant for this graph execution.
    struct ggml_tensor *vViewSrc = vNode->view_src;
    while (vViewSrc) {
      if (fGgmlNodeListFindTensor(pCgraph, pNodeIdxs, pCount, vViewSrc) == -1 && !fGgmlIsConstant(vViewSrc)) {
        return false;
      }
      vViewSrc = vViewSrc->view_src;
    }
  }

  return true;
}

// check if node is part of the graph
static bool fGgmlGraphFind(const struct ggml_cgraph *pCgraph, const struct ggml_tensor *pNode) {
  if (pCgraph == NULL) {
    return true;
  }

  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    if (pCgraph->nodes[vI] == pNode) {
      return true;
    }
  }

  return false;
}

static struct ggml_tensor *fGgmlGraphGetParent(const struct ggml_cgraph *pCgraph, const struct ggml_tensor *pNode) {
  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    struct ggml_tensor *vParent = pCgraph->nodes[vI];
    struct ggml_tensor *vGrad = ggml_graph_get_grad(pCgraph, vParent);

    if (vGrad == pNode) {
      return vParent;
    }
  }

  return NULL;
}

static void fGgmlGraphDumpDotNodeEdge(FILE *pFp, const struct ggml_cgraph *pGb, struct ggml_tensor *pNode,
                                      struct ggml_tensor *pParent, const char *pLabel) {
  struct ggml_tensor *vGparent = fGgmlGraphGetParent(pGb, pNode);
  struct ggml_tensor *vGparent0 = fGgmlGraphGetParent(pGb, pParent);
  fprintf(pFp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
          vGparent0 ? (void *)vGparent0 : (void *)pParent, vGparent ? (void *)vGparent : (void *)pNode,
          vGparent ? "empty" : "vee", vGparent ? "dashed" : "solid", pLabel);
}

static void fGgmlGraphDumpDotLeafEdge(FILE *pFp, struct ggml_tensor *pNode, struct ggml_tensor *pParent,
                                      const char *pLabel) {
  fprintf(pFp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n", (void *)pParent, (void *)pNode, pLabel);
}

void ggml_graph_dump_dot(const struct ggml_cgraph *pGb, const struct ggml_cgraph *pCgraph, const char *pFilename) {
  char aColor[16];

  FILE *vFp = ggml_fopen(pFilename, "w");
  GGML_ASSERT(vFp);

  fprintf(vFp, "digraph G {\n");
  fprintf(vFp, "  newrank = true;\n");
  fprintf(vFp, "  rankdir = TB;\n");

  for (int vI = 0; vI < pGb->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGb->nodes[vI];
    struct ggml_tensor *vGrad = ggml_graph_get_grad(pGb, vNode);

    if (fGgmlGraphGetParent(pGb, vNode) != NULL) {
      continue;
    }

    if (vNode->flags & GGML_TENSOR_FLAG_PARAM) {
      snprintf(aColor, sizeof(aColor), "yellow");
    } else if (vGrad) {
      if (fGgmlGraphFind(pCgraph, vNode)) {
        snprintf(aColor, sizeof(aColor), "green");
      } else {
        snprintf(aColor, sizeof(aColor), "lightblue");
      }
    } else {
      snprintf(aColor, sizeof(aColor), "white");
    }

    fprintf(vFp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"",
            (void *)vNode, aColor);

    if (strlen(vNode->name) > 0) {
      fprintf(vFp, "%s (%s)|", vNode->name, ggml_type_name(vNode->type));
    } else {
      fprintf(vFp, "(%s)|", ggml_type_name(vNode->type));
    }

    if (ggml_is_matrix(vNode)) {
      fprintf(vFp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", vI, vNode->ne[0], vNode->ne[1], ggml_op_symbol(vNode->op));
    } else {
      fprintf(vFp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", vI, vNode->ne[0], vNode->ne[1], vNode->ne[2],
              ggml_op_symbol(vNode->op));
    }

    if (vGrad) {
      fprintf(vFp, " | <g>%s\"; ]\n", ggml_op_symbol(vGrad->op));
    } else {
      fprintf(vFp, "\"; ]\n");
    }
  }

  for (int vI = 0; vI < pGb->n_leafs; vI++) {
    struct ggml_tensor *vNode = pGb->leafs[vI];

    snprintf(aColor, sizeof(aColor), "pink");

    fprintf(vFp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"<x>",
            (void *)vNode, aColor);

    if (strlen(vNode->name) > 0) {
      fprintf(vFp, "%s (%s)|", vNode->name, ggml_type_name(vNode->type));
    } else {
      fprintf(vFp, "(%s)|", ggml_type_name(vNode->type));
    }

    fprintf(vFp, "CONST %d [%" PRId64 ", %" PRId64 "]", vI, vNode->ne[0], vNode->ne[1]);
    if (ggml_nelements(vNode) < 5 && vNode->data != NULL) {
      fprintf(vFp, " | (");
      for (int vJ = 0; vJ < ggml_nelements(vNode); vJ++) {
        // FIXME: use ggml-backend to obtain the tensor data
        // if (node->type == GGML_TYPE_I8 || node->type == GGML_TYPE_I16 || node->type == GGML_TYPE_I32) {
        //    fprintf(fp, "%d", ggml_get_i32_1d(node, j));
        //}
        // else if (node->type == GGML_TYPE_F32 ||
        //         node->type == GGML_TYPE_F16 ||
        //         node->type == GGML_TYPE_BF16) {
        //    fprintf(fp, "%.1e", (double)ggml_get_f32_1d(node, j));
        //}
        // else
        {
          fprintf(vFp, "#");
        }
        if (vJ < ggml_nelements(vNode) - 1) {
          fprintf(vFp, ", ");
        }
      }
      fprintf(vFp, ")");
    }
    fprintf(vFp, "\"; ]\n");
  }

  for (int vI = 0; vI < pGb->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGb->nodes[vI];

    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      if (vNode->src[vJ]) {
        char aLabel[16];
        snprintf(aLabel, sizeof(aLabel), "src %d", vJ);
        fGgmlGraphDumpDotNodeEdge(vFp, pGb, vNode, vNode->src[vJ], aLabel);
      }
    }
  }

  for (int vI = 0; vI < pGb->n_leafs; vI++) {
    struct ggml_tensor *vNode = pGb->leafs[vI];

    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      if (vNode->src[vJ]) {
        char aLabel[16];
        snprintf(aLabel, sizeof(aLabel), "src %d", vJ);
        fGgmlGraphDumpDotLeafEdge(vFp, vNode, vNode->src[vJ], aLabel);
      }
    }
  }

  fprintf(vFp, "}\n");

  fclose(vFp);

  GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, pFilename, pFilename, pFilename);
}

////////////////////////////////////////////////////////////////////////////////

void ggml_set_input(struct ggml_tensor *pTensor) { pTensor->flags |= GGML_TENSOR_FLAG_INPUT; }

void ggml_set_output(struct ggml_tensor *pTensor) {
  for (struct ggml_tensor *vCur = pTensor; vCur != NULL; vCur = vCur->view_src) {
    vCur->flags |= GGML_TENSOR_FLAG_OUTPUT;
  }
}

void ggml_set_param(struct ggml_tensor *pTensor) {
  GGML_ASSERT(pTensor->op == GGML_OP_NONE);
  pTensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

void ggml_set_loss(struct ggml_tensor *pTensor) {
  GGML_ASSERT(ggml_is_scalar(pTensor));
  GGML_ASSERT(pTensor->type == GGML_TYPE_F32);
  pTensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

void ggml_quantize_init(enum ggml_type pType) {
  ggml_critical_section_start();

  switch (pType) {
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
    iq2xs_init_impl(pType);
    break;
  case GGML_TYPE_IQ3_XXS:
    iq3xs_init_impl(256);
    break;
  case GGML_TYPE_IQ3_S:
    iq3xs_init_impl(512);
    break;
  default: // nothing
    break;
  }

  ggml_critical_section_end();
}

void ggml_quantize_free(void) {
  ggml_critical_section_start();

  iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
  iq2xs_free_impl(GGML_TYPE_IQ2_XS);
  iq2xs_free_impl(GGML_TYPE_IQ2_S);
  iq2xs_free_impl(GGML_TYPE_IQ1_S);
  iq2xs_free_impl(GGML_TYPE_IQ1_M);
  iq3xs_free_impl(256);
  iq3xs_free_impl(512);

  ggml_critical_section_end();
}

bool ggml_quantize_requires_imatrix(enum ggml_type pType) {
  return pType == GGML_TYPE_IQ2_XXS || pType == GGML_TYPE_IQ2_XS || pType == GGML_TYPE_IQ1_S; //   ||
  // type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(enum ggml_type pType, const float *pSrc, void *pDst, int64_t pStart, int64_t pNrows,
                           int64_t pNPerRow, const float *pImatrix) {
  const int64_t cN = pNrows * pNPerRow;

  if (ggml_quantize_requires_imatrix(pType)) {
    GGML_ASSERT(pImatrix != NULL);
  }

  GGML_ASSERT(pStart % type_traits[pType].blck_size == 0);
  GGML_ASSERT(pStart % pNPerRow == 0);

  ggml_quantize_init(pType); // this is noop if already initialized

  const size_t cStartRow = pStart / pNPerRow;
  const size_t cRowSize = ggml_row_size(pType, pNPerRow);

  size_t vResult = 0;

  switch (pType) {
  case GGML_TYPE_Q1_0:
    vResult = quantize_q1_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q2_0:
    vResult = quantize_q2_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q4_0:
    vResult = quantize_q4_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q4_1:
    vResult = quantize_q4_1(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q5_0:
    vResult = quantize_q5_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q5_1:
    vResult = quantize_q5_1(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q8_0:
    vResult = quantize_q8_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_MXFP4:
    vResult = quantize_mxfp4(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_NVFP4:
    vResult = quantize_nvfp4(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q2_K:
    vResult = quantize_q2_K(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q3_K:
    vResult = quantize_q3_K(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q4_K:
    vResult = quantize_q4_K(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q5_K:
    vResult = quantize_q5_K(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_Q6_K:
    vResult = quantize_q6_K(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_TQ1_0:
    vResult = quantize_tq1_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_TQ2_0:
    vResult = quantize_tq2_0(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ2_XXS:
    vResult = quantize_iq2_xxs(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ2_XS:
    vResult = quantize_iq2_xs(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ3_XXS:
    vResult = quantize_iq3_xxs(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ3_S:
    vResult = quantize_iq3_s(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ2_S:
    vResult = quantize_iq2_s(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ1_S:
    vResult = quantize_iq1_s(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ1_M:
    vResult = quantize_iq1_m(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ4_NL:
    vResult = quantize_iq4_nl(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_IQ4_XS:
    vResult = quantize_iq4_xs(pSrc + pStart, (char *)pDst + cStartRow * cRowSize, pNrows, pNPerRow, pImatrix);
    break;
  case GGML_TYPE_F16: {
    size_t vElemsize = sizeof(ggml_fp16_t);
    ggml_fp32_to_fp16_row(pSrc + pStart, (ggml_fp16_t *)pDst + pStart, cN);
    vResult = cN * vElemsize;
  } break;
  case GGML_TYPE_BF16: {
    size_t vElemsize = sizeof(ggml_bf16_t);
    ggml_fp32_to_bf16_row_ref(pSrc + pStart, (ggml_bf16_t *)pDst + pStart, cN);
    vResult = cN * vElemsize;
  } break;
  case GGML_TYPE_F32: {
    size_t vElemsize = sizeof(float);
    vResult = cN * vElemsize;
    memcpy((uint8_t *)pDst + pStart * vElemsize, pSrc + pStart, vResult);
  } break;
  default:
    assert(false);
  }

  GGML_ASSERT(vResult == pNrows * cRowSize);

  return vResult;
}

////////////////////////////////////////////////////////////////////////////////

void ggml_log_get(ggml_log_callback *pLogCallback, void **pUserData) {
  *pLogCallback = g_logger_state.vLogCallback;
  *pUserData = g_logger_state.vLogCallbackUserData;
}

void ggml_log_set(ggml_log_callback pLogCallback, void *pUserData) {
  g_logger_state.vLogCallback = pLogCallback ? pLogCallback : ggml_log_callback_default;
  g_logger_state.vLogCallbackUserData = pUserData;
}

void ggml_threadpool_params_init(struct ggml_threadpool_params *p_, int pNThreads) {
  p_->n_threads = pNThreads;
  p_->prio = 0;                               // default priority (usually means normal or inherited)
  p_->poll = 50;                              // hybrid-polling enabled
  p_->strict_cpu = false;                     // no strict placement (all threads share same cpumask)
  p_->paused = false;                         // threads are ready to go
  memset(p_->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

struct ggml_threadpool_params ggml_threadpool_params_default(int pNThreads) {
  struct ggml_threadpool_params vP;
  ggml_threadpool_params_init(&vP, pNThreads);
  return vP;
}

bool ggml_threadpool_params_match(const struct ggml_threadpool_params *p0, const struct ggml_threadpool_params *p1) {
  if (p0->n_threads != p1->n_threads)
    return false;
  if (p0->prio != p1->prio)
    return false;
  if (p0->poll != p1->poll)
    return false;
  if (p0->strict_cpu != p1->strict_cpu)
    return false;
  return memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
