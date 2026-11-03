#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES        // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"

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

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_TINYBLAS
#endif

#ifdef GGML_USE_TINYBLAS
#include "tinyblas/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE 64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T)                                                                                                  \
  do {                                                                                                                 \
    T SWAP = x;                                                                                                        \
    (x) = y;                                                                                                           \
    (y) = SWAP;                                                                                                        \
  } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
  int sve_cnt;
} ggml_arm_arch_features = {0};
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
  int rvv_vlen;
} ggml_riscv_arch_features = {0};
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
  memory_order_relaxed,
  memory_order_consume,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int *ptr, LONG val) { InterlockedExchange(ptr, val); }
static void atomic_store_explicit(atomic_int *ptr, LONG val, memory_order mo) {
  // TODO: add support for explicit memory order
  InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int *ptr) { return InterlockedCompareExchange(ptr, 0, 0); }
static LONG atomic_load_explicit(atomic_int *ptr, memory_order mo) {
  // TODO: add support for explicit memory order
  return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int *ptr, LONG inc) { return InterlockedExchangeAdd(ptr, inc); }
static LONG atomic_fetch_add_explicit(atomic_int *ptr, LONG inc, memory_order mo) {
  // TODO: add support for explicit memory order
  return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag *ptr) { return InterlockedExchange(ptr, 1); }
static void atomic_flag_clear(atomic_flag *ptr) { InterlockedExchange(ptr, 0); }
static void atomic_thread_fence(memory_order mo) { MemoryBarrier(); }
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t *out, void *unused, thread_ret_t (*func)(void *), void *arg) {
  (void)unused;
  HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
  if (handle == NULL) {
    return EAGAIN;
  }

  *out = handle;
  return 0;
}

static int pthread_join(pthread_t thread, void *unused) {
  (void)unused;
  int ret = (int)WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
  return ret;
}

static int sched_yield(void) {
  Sleep(0);
  return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void *thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] =
        {
            .from_float = (ggml_from_float_t)ggml_cpu_fp32_to_fp32,
            .vec_dot = (ggml_vec_dot_t)ggml_vec_dot_f32,
            .vec_dot_type = GGML_TYPE_F32,
            .nrows = 1,
        },
    [GGML_TYPE_F16] =
        {
            .from_float = (ggml_from_float_t)ggml_cpu_fp32_to_fp16,
            .vec_dot = (ggml_vec_dot_t)ggml_vec_dot_f16,
            .vec_dot_type = GGML_TYPE_F16,
            .nrows = 1,
        },
    [GGML_TYPE_Q1_0] =
        {
            .from_float = quantize_row_q1_0,
            .vec_dot = ggml_vec_dot_q1_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_Q2_0] =
        {
            .from_float = quantize_row_q2_0,
            .vec_dot = ggml_vec_dot_q2_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_Q4_0] =
        {
            .from_float = quantize_row_q4_0,
            .vec_dot = ggml_vec_dot_q4_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
#if defined(__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        },
    [GGML_TYPE_Q4_1] =
        {
            .from_float = quantize_row_q4_1,
            .vec_dot = ggml_vec_dot_q4_1_q8_1,
            .vec_dot_type = GGML_TYPE_Q8_1,
#if defined(__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        },
    [GGML_TYPE_Q5_0] =
        {
            .from_float = quantize_row_q5_0,
            .vec_dot = ggml_vec_dot_q5_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_Q5_1] =
        {
            .from_float = quantize_row_q5_1,
            .vec_dot = ggml_vec_dot_q5_1_q8_1,
            .vec_dot_type = GGML_TYPE_Q8_1,
            .nrows = 1,
        },
    [GGML_TYPE_Q8_0] =
        {
            .from_float = quantize_row_q8_0,
            .vec_dot = ggml_vec_dot_q8_0_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
#if defined(__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        },
    [GGML_TYPE_Q8_1] =
        {
            .from_float = quantize_row_q8_1,
            .vec_dot_type = GGML_TYPE_Q8_1,
            .nrows = 1,
        },
    [GGML_TYPE_MXFP4] =
        {
            .from_float = quantize_row_mxfp4,
            .vec_dot = ggml_vec_dot_mxfp4_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_NVFP4] =
        {
            .from_float = quantize_row_nvfp4,
            .vec_dot = ggml_vec_dot_nvfp4_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_Q2_K] =
        {
            .from_float = quantize_row_q2_K,
            .vec_dot = ggml_vec_dot_q2_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_Q3_K] =
        {
            .from_float = quantize_row_q3_K,
            .vec_dot = ggml_vec_dot_q3_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_Q4_K] =
        {
            .from_float = quantize_row_q4_K,
            .vec_dot = ggml_vec_dot_q4_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
#if defined(__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        },
    [GGML_TYPE_Q5_K] =
        {
            .from_float = quantize_row_q5_K,
            .vec_dot = ggml_vec_dot_q5_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_Q6_K] =
        {
            .from_float = quantize_row_q6_K,
            .vec_dot = ggml_vec_dot_q6_K_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
#if defined(__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        },
    [GGML_TYPE_IQ2_XXS] =
        {
            .from_float = NULL,
            .vec_dot = ggml_vec_dot_iq2_xxs_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ2_XS] =
        {
            .from_float = NULL,
            .vec_dot = ggml_vec_dot_iq2_xs_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ3_XXS] =
        {
            // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in
            // ggml_quantize_init
            //.from_float               = quantize_row_iq3_xxs,
            .vec_dot = ggml_vec_dot_iq3_xxs_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ3_S] =
        {
            //.from_float               = quantize_row_iq3_s,
            .vec_dot = ggml_vec_dot_iq3_s_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ2_S] =
        {
            //.from_float               = quantize_row_iq2_s,
            .vec_dot = ggml_vec_dot_iq2_s_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ1_S] =
        {
            .from_float = NULL,
            .vec_dot = ggml_vec_dot_iq1_s_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ1_M] =
        {
            .from_float = NULL,
            .vec_dot = ggml_vec_dot_iq1_m_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_IQ4_NL] =
        {
            .from_float = quantize_row_iq4_nl,
            .vec_dot = ggml_vec_dot_iq4_nl_q8_0,
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        },
    [GGML_TYPE_IQ4_XS] =
        {
            .from_float = quantize_row_iq4_xs,
            .vec_dot = ggml_vec_dot_iq4_xs_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_Q8_K] =
        {
            .from_float = quantize_row_q8_K,
        },
    [GGML_TYPE_BF16] =
        {
            .from_float = (ggml_from_float_t)ggml_cpu_fp32_to_bf16,
            .vec_dot = (ggml_vec_dot_t)ggml_vec_dot_bf16,
            .vec_dot_type = GGML_TYPE_BF16,
            .nrows = 1,
        },
    [GGML_TYPE_TQ1_0] =
        {
            .from_float = quantize_row_tq1_0,
            .vec_dot = ggml_vec_dot_tq1_0_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_TQ2_0] =
        {
            .from_float = quantize_row_tq2_0,
            .vec_dot = ggml_vec_dot_tq2_0_q8_K,
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        },
    [GGML_TYPE_I32] =
        {
            .from_float = (ggml_from_float_t)ggml_cpu_fp32_to_i32,
        },
};

const struct ggml_type_traits_cpu *ggml_get_type_traits_cpu(enum ggml_type pType) { return &type_traits_cpu[pType]; }

//
// Threading defs
//

typedef pthread_t ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK ggml_mutex_t;

#define ggml_mutex_init(m) InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m) AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m) AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c) InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join pthread_join

#else

typedef pthread_cond_t ggml_cond_t;
typedef pthread_mutex_t ggml_mutex_t;

#define ggml_mutex_init(m) pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m) pthread_mutex_destroy(m)
#define ggml_mutex_lock(m) pthread_mutex_lock(m)
#define ggml_mutex_unlock(m) pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m) pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x) UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x) _mm_pause()
#else
#define ggml_lock_lock(x) UNUSED(x)
#endif
#define ggml_lock_unlock(x) UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c) pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c) pthread_cond_destroy(c)
#define ggml_cond_wait(c, m) pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
  ggml_mutex_t vMutex; // mutex for cond.var
  ggml_cond_t vCond;   // cond.var for waiting for new work

  struct ggml_cgraph *vCgraph;
  struct ggml_cplan *cplan;

  // synchronization primitives
  atomic_int n_graph; // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
  atomic_int GGML_CACHE_ALIGN n_barrier;
  atomic_int GGML_CACHE_ALIGN vNBarrierPassed;
  atomic_int GGML_CACHE_ALIGN
      vCurrentChunk; // currently processing chunk during Mat_Mul, shared between all the threads.

  // these are atomic as an annotation for thread-sanitizer
  atomic_bool vStop;  // Used for stopping the threadpool altogether
  atomic_bool vPause; // Used for pausing the threadpool or individual threads
  atomic_int vAbort;  // Used for aborting processing of a graph

  struct ggml_compute_state *workers; // per thread state
  int n_threads;                      // Number of threads in the pool
  int32_t prio;                       // Scheduling priority
  uint32_t poll;                      // Polling level (0 - no polling)

  enum ggml_status vEc;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
  ggml_thread_t thrd;
  int last_graph;
  bool pending;
#endif
  bool cpumask[GGML_MAX_N_THREADS];
  struct ggml_threadpool *threadpool;
  int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
static inline void fGgmlThreadCpuRelax(void) { __asm__ volatile("yield" ::: "memory"); }
#elif defined(__x86_64__)
static inline void fGgmlThreadCpuRelax(void) { _mm_pause(); }
#elif defined(__riscv)
static inline void fGgmlThreadCpuRelax(void) {
#ifdef __riscv_zihintpause
  __asm__ __volatile__("pause");
#else
  /* Encoding of the pause instruction */
  __asm__ __volatile__(".4byte 0x100000F");
#endif
}
#else
static inline void fGgmlThreadCpuRelax(void) { ; }
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
  uint32_t vCpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
  uint32_t vNCpus;
};

struct ggml_numa_nodes {
  enum ggml_numa_strategy vNumaStrategy;
  struct ggml_numa_node aNodes[GGML_NUMA_MAX_NODES];
  uint32_t n_nodes;
  uint32_t vTotalCpus;   // hardware threads on system
  uint32_t vCurrentNode; // node on which main process is execting
#if defined(__gnu_linux__)
  cpu_set_t cpuset; // cpuset from numactl
#else
  uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
  struct ggml_numa_nodes vNuma;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool *pTp) {
  int n_threads = atomic_load_explicit(&pTp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
  if (n_threads == 1) {
    return;
  }

#ifdef GGML_USE_OPENMP
#pragma omp barrier
#else
  int n_passed = atomic_load_explicit(&pTp->vNBarrierPassed, memory_order_relaxed);

  // enter barrier (full seq-cst fence)
  int n_barrier = atomic_fetch_add_explicit(&pTp->n_barrier, 1, memory_order_seq_cst);

  if (n_barrier == (n_threads - 1)) {
    // last thread
    atomic_store_explicit(&pTp->n_barrier, 0, memory_order_relaxed);

    // exit barrier (full seq-cst fence)
    atomic_fetch_add_explicit(&pTp->vNBarrierPassed, 1, memory_order_seq_cst);
    return;
  }

  // wait for other threads
  while (atomic_load_explicit(&pTp->vNBarrierPassed, memory_order_relaxed) == n_passed) {
    fGgmlThreadCpuRelax();
  }

// exit barrier (full seq-cst fence)
// TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
#ifdef GGML_TSAN_ENABLED
  atomic_fetch_add_explicit(&pTp->vNBarrierPassed, 0, memory_order_seq_cst);
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool *pTp, int pValue) {
  atomic_store_explicit(&pTp->vCurrentChunk, pValue, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool *pTp, int pValue) {
  return atomic_fetch_add_explicit(&pTp->vCurrentChunk, pValue, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t fGgmlGetNumaAffinity(void) {
  cpu_set_t vCpuset;
  pthread_t vThread;
  vThread = pthread_self();
  CPU_ZERO(&vCpuset);
  pthread_getaffinity_np(vThread, sizeof(cpu_set_t), &vCpuset);
  return vCpuset;
}
#else
static uint32_t fGgmlGetNumaAffinity(void) {
  return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy pNumaFlag) {
  if (g_state.vNuma.n_nodes > 0) {
    fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

    return;
  }

#if defined(__gnu_linux__)
  struct stat vSt;
  char vPath[256];
  int vRv;

  // set numa scheme
  g_state.vNuma.vNumaStrategy = pNumaFlag;

  GGML_PRINT_DEBUG("numa strategy %u\n", g_state.vNuma.vNumaStrategy);

  g_state.vNuma.cpuset = fGgmlGetNumaAffinity();

  // enumerate nodes
  while (g_state.vNuma.n_nodes < GGML_NUMA_MAX_NODES) {
    vRv = snprintf(vPath, sizeof(vPath), "/sys/devices/system/node/node%u", g_state.vNuma.n_nodes);
    GGML_ASSERT(vRv > 0 && (unsigned)vRv < sizeof(vPath));
    if (stat(vPath, &vSt) != 0) {
      break;
    }
    ++g_state.vNuma.n_nodes;
  }

  // enumerate CPUs
  while (g_state.vNuma.vTotalCpus < GGML_NUMA_MAX_CPUS) {
    vRv = snprintf(vPath, sizeof(vPath), "/sys/devices/system/cpu/cpu%u", g_state.vNuma.vTotalCpus);
    GGML_ASSERT(vRv > 0 && (unsigned)vRv < sizeof(vPath));
    if (stat(vPath, &vSt) != 0) {
      break;
    }
    ++g_state.vNuma.vTotalCpus;
  }

  GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.vNuma.n_nodes, g_state.vNuma.vTotalCpus);

  // figure out which node we're on
  uint vCurrentCpu;
  int vGetcpuRet = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
  vGetcpuRet = getcpu(&vCurrentCpu, &g_state.vNuma.vCurrentNode);
#else
  // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#endif
  vGetcpuRet = syscall(SYS_getcpu, &vCurrentCpu, &g_state.vNuma.vCurrentNode);
#endif

  if (g_state.vNuma.n_nodes < 1 || g_state.vNuma.vTotalCpus < 1 || vGetcpuRet != 0) {
    g_state.vNuma.n_nodes = 0;
    return;
  }

  GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.vNuma.vCurrentNode, vCurrentCpu);

  for (uint32_t vN = 0; vN < g_state.vNuma.n_nodes; ++vN) {
    struct ggml_numa_node *vNode = &g_state.vNuma.aNodes[vN];
    GGML_PRINT_DEBUG("CPUs on node %u:", vN);
    vNode->vNCpus = 0;
    for (uint32_t vC = 0; vC < g_state.vNuma.vTotalCpus; ++vC) {
      vRv = snprintf(vPath, sizeof(vPath), "/sys/devices/system/node/node%u/cpu%u", vN, vC);
      GGML_ASSERT(vRv > 0 && (unsigned)vRv < sizeof(vPath));
      if (stat(vPath, &vSt) == 0) {
        vNode->vCpus[vNode->vNCpus++] = vC;
        GGML_PRINT_DEBUG(" %u", vC);
      }
    }
    GGML_PRINT_DEBUG("\n");
  }

  if (ggml_is_numa()) {
    FILE *vFptr = fopen("/proc/sys/kernel/numa_balancing", "r");
    if (vFptr != NULL) {
      char aBuf[42];
      if (fgets(aBuf, sizeof(aBuf), vFptr) && strncmp(aBuf, "0\n", sizeof(aBuf)) != 0) {
        GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
      }
      fclose(vFptr);
    }
  }
#else
  UNUSED(pNumaFlag);
  // TODO
#endif
}

bool ggml_is_numa(void) { return g_state.vNuma.n_nodes > 1; }

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) { ggml_arm_arch_features.sve_cnt = svcntb(); }
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void fGgmlInitRiscvArchFeatures(void) { ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb(); }
#else
static void fGgmlInitRiscvArchFeatures(void) {}
#endif

struct ggml_tensor *ggml_new_i32(struct ggml_context *pCtx, int32_t pValue) {
  GGML_ASSERT(!ggml_get_no_alloc(pCtx));

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, 1);

  ggml_set_i32(vResult, pValue);

  return vResult;
}

struct ggml_tensor *ggml_new_f32(struct ggml_context *pCtx, float pValue) {
  GGML_ASSERT(!ggml_get_no_alloc(pCtx));

  struct ggml_tensor *vResult = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, 1);

  ggml_set_f32(vResult, pValue);

  return vResult;
}

struct ggml_tensor *ggml_set_i32(struct ggml_tensor *pTensor, int32_t pValue) {
  const int cN = ggml_nrows(pTensor);
  const int cNc = pTensor->ne[0];
  const size_t cN1 = pTensor->nb[1];

  char *const cData = pTensor->data;

  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    assert(pTensor->nb[0] == sizeof(int8_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i8(cNc, (int8_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_I16: {
    assert(pTensor->nb[0] == sizeof(int16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i16(cNc, (int16_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_I32: {
    assert(pTensor->nb[0] == sizeof(int32_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i32(cNc, (int32_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_F16: {
    assert(pTensor->nb[0] == sizeof(ggml_fp16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_f16(cNc, (ggml_fp16_t *)(cData + vI * cN1), GGML_CPU_FP32_TO_FP16(pValue));
    }
  } break;
  case GGML_TYPE_BF16: {
    assert(pTensor->nb[0] == sizeof(ggml_fp16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_bf16(cNc, (ggml_bf16_t *)(cData + vI * cN1), GGML_FP32_TO_BF16(pValue));
    }
  } break;
  case GGML_TYPE_F32: {
    assert(pTensor->nb[0] == sizeof(float));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_f32(cNc, (float *)(cData + vI * cN1), pValue);
    }
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }

  return pTensor;
}

struct ggml_tensor *ggml_set_f32(struct ggml_tensor *pTensor, float pValue) {
  const int cN = ggml_nrows(pTensor);
  const int cNc = pTensor->ne[0];
  const size_t cN1 = pTensor->nb[1];

  char *const cData = pTensor->data;

  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    assert(pTensor->nb[0] == sizeof(int8_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i8(cNc, (int8_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_I16: {
    assert(pTensor->nb[0] == sizeof(int16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i16(cNc, (int16_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_I32: {
    assert(pTensor->nb[0] == sizeof(int32_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_i32(cNc, (int32_t *)(cData + vI * cN1), pValue);
    }
  } break;
  case GGML_TYPE_F16: {
    assert(pTensor->nb[0] == sizeof(ggml_fp16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_f16(cNc, (ggml_fp16_t *)(cData + vI * cN1), GGML_CPU_FP32_TO_FP16(pValue));
    }
  } break;
  case GGML_TYPE_BF16: {
    assert(pTensor->nb[0] == sizeof(ggml_bf16_t));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_bf16(cNc, (ggml_bf16_t *)(cData + vI * cN1), GGML_FP32_TO_BF16(pValue));
    }
  } break;
  case GGML_TYPE_F32: {
    assert(pTensor->nb[0] == sizeof(float));
    for (int vI = 0; vI < cN; vI++) {
      ggml_vec_set_f32(cNc, (float *)(cData + vI * cN1), pValue);
    }
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }

  return pTensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor *pTensor, int pI) {
  if (!ggml_is_contiguous(pTensor)) {
    int64_t vId[4] = {0, 0, 0, 0};
    ggml_unravel_index(pTensor, pI, &vId[0], &vId[1], &vId[2], &vId[3]);
    return ggml_get_i32_nd(pTensor, vId[0], vId[1], vId[2], vId[3]);
  }
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int8_t));
    return ((int8_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_I16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int16_t));
    return ((int16_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_I32: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int32_t));
    return ((int32_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_F16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(ggml_fp16_t));
    return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(pTensor->data))[pI]);
  }
  case GGML_TYPE_BF16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(ggml_bf16_t));
    return GGML_BF16_TO_FP32(((ggml_bf16_t *)(pTensor->data))[pI]);
  }
  case GGML_TYPE_F32: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(float));
    return ((float *)(pTensor->data))[pI];
  }
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

void ggml_set_i32_1d(const struct ggml_tensor *pTensor, int pI, int32_t pValue) {
  if (!ggml_is_contiguous(pTensor)) {
    int64_t vId[4] = {0, 0, 0, 0};
    ggml_unravel_index(pTensor, pI, &vId[0], &vId[1], &vId[2], &vId[3]);
    ggml_set_i32_nd(pTensor, vId[0], vId[1], vId[2], vId[3], pValue);
    return;
  }
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int8_t));
    ((int8_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_I16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int16_t));
    ((int16_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_I32: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(int32_t));
    ((int32_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_F16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(ggml_fp16_t));
    ((ggml_fp16_t *)(pTensor->data))[pI] = GGML_CPU_FP32_TO_FP16(pValue);
  } break;
  case GGML_TYPE_BF16: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(ggml_bf16_t));
    ((ggml_bf16_t *)(pTensor->data))[pI] = GGML_FP32_TO_BF16(pValue);
  } break;
  case GGML_TYPE_F32: {
    GGML_ASSERT(pTensor->nb[0] == sizeof(float));
    ((float *)(pTensor->data))[pI] = pValue;
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor *pTensor, int pI0, int pI1, int pI2, int pI3) {
  void *vData =
      (char *)pTensor->data + pI0 * pTensor->nb[0] + pI1 * pTensor->nb[1] + pI2 * pTensor->nb[2] + pI3 * pTensor->nb[3];
  switch (pTensor->type) {
  case GGML_TYPE_I8:
    return ((int8_t *)vData)[0];
  case GGML_TYPE_I16:
    return ((int16_t *)vData)[0];
  case GGML_TYPE_I32:
    return ((int32_t *)vData)[0];
  case GGML_TYPE_F16:
    return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)vData)[0]);
  case GGML_TYPE_BF16:
    return GGML_BF16_TO_FP32(((ggml_bf16_t *)vData)[0]);
  case GGML_TYPE_F32:
    return ((float *)vData)[0];
  default:
    GGML_ABORT("fatal error");
  }
}

void ggml_set_i32_nd(const struct ggml_tensor *pTensor, int pI0, int pI1, int pI2, int pI3, int32_t pValue) {
  void *vData =
      (char *)pTensor->data + pI0 * pTensor->nb[0] + pI1 * pTensor->nb[1] + pI2 * pTensor->nb[2] + pI3 * pTensor->nb[3];
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    ((int8_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_I16: {
    ((int16_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_I32: {
    ((int32_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_F16: {
    ((ggml_fp16_t *)(vData))[0] = GGML_CPU_FP32_TO_FP16(pValue);
  } break;
  case GGML_TYPE_BF16: {
    ((ggml_bf16_t *)(vData))[0] = GGML_FP32_TO_BF16(pValue);
  } break;
  case GGML_TYPE_F32: {
    ((float *)(vData))[0] = pValue;
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

float ggml_get_f32_1d(const struct ggml_tensor *pTensor, int pI) {
  if (!ggml_is_contiguous(pTensor)) {
    int64_t vId[4] = {0, 0, 0, 0};
    ggml_unravel_index(pTensor, pI, &vId[0], &vId[1], &vId[2], &vId[3]);
    return ggml_get_f32_nd(pTensor, vId[0], vId[1], vId[2], vId[3]);
  }
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    return ((int8_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_I16: {
    return ((int16_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_I32: {
    return ((int32_t *)(pTensor->data))[pI];
  }
  case GGML_TYPE_F16: {
    return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(pTensor->data))[pI]);
  }
  case GGML_TYPE_BF16: {
    return GGML_BF16_TO_FP32(((ggml_bf16_t *)(pTensor->data))[pI]);
  }
  case GGML_TYPE_F32: {
    return ((float *)(pTensor->data))[pI];
  }
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

void ggml_set_f32_1d(const struct ggml_tensor *pTensor, int pI, float pValue) {
  if (!ggml_is_contiguous(pTensor)) {
    int64_t vId[4] = {0, 0, 0, 0};
    ggml_unravel_index(pTensor, pI, &vId[0], &vId[1], &vId[2], &vId[3]);
    ggml_set_f32_nd(pTensor, vId[0], vId[1], vId[2], vId[3], pValue);
    return;
  }
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    ((int8_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_I16: {
    ((int16_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_I32: {
    ((int32_t *)(pTensor->data))[pI] = pValue;
  } break;
  case GGML_TYPE_F16: {
    ((ggml_fp16_t *)(pTensor->data))[pI] = GGML_CPU_FP32_TO_FP16(pValue);
  } break;
  case GGML_TYPE_BF16: {
    ((ggml_bf16_t *)(pTensor->data))[pI] = GGML_FP32_TO_BF16(pValue);
  } break;
  case GGML_TYPE_F32: {
    ((float *)(pTensor->data))[pI] = pValue;
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

float ggml_get_f32_nd(const struct ggml_tensor *pTensor, int pI0, int pI1, int pI2, int pI3) {
  void *vData =
      (char *)pTensor->data + pI0 * pTensor->nb[0] + pI1 * pTensor->nb[1] + pI2 * pTensor->nb[2] + pI3 * pTensor->nb[3];
  switch (pTensor->type) {
  case GGML_TYPE_I8:
    return ((int8_t *)vData)[0];
  case GGML_TYPE_I16:
    return ((int16_t *)vData)[0];
  case GGML_TYPE_I32:
    return ((int32_t *)vData)[0];
  case GGML_TYPE_F16:
    return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)vData)[0]);
  case GGML_TYPE_BF16:
    return GGML_BF16_TO_FP32(((ggml_bf16_t *)vData)[0]);
  case GGML_TYPE_F32:
    return ((float *)vData)[0];
  default:
    GGML_ABORT("fatal error");
  }
}

void ggml_set_f32_nd(const struct ggml_tensor *pTensor, int pI0, int pI1, int pI2, int pI3, float pValue) {
  void *vData =
      (char *)pTensor->data + pI0 * pTensor->nb[0] + pI1 * pTensor->nb[1] + pI2 * pTensor->nb[2] + pI3 * pTensor->nb[3];
  switch (pTensor->type) {
  case GGML_TYPE_I8: {
    ((int8_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_I16: {
    ((int16_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_I32: {
    ((int32_t *)(vData))[0] = pValue;
  } break;
  case GGML_TYPE_F16: {
    ((ggml_fp16_t *)(vData))[0] = GGML_CPU_FP32_TO_FP16(pValue);
  } break;
  case GGML_TYPE_BF16: {
    ((ggml_bf16_t *)(vData))[0] = GGML_FP32_TO_BF16(pValue);
  } break;
  case GGML_TYPE_F32: {
    ((float *)(vData))[0] = pValue;
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void fGgmlComputeForwardMulMatOneChunk(const struct ggml_compute_params *pParams, struct ggml_tensor *dst,
                                              const enum ggml_type pType, const int64_t pNumRowsPerVecDot,
                                              const int64_t pIr0Start, const int64_t pIr0End, const int64_t pIr1Start,
                                              const int64_t pIr1End) {

  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const bool cSrc1Cont = ggml_is_contiguous(src1);

  ggml_vec_dot_t const cVecDot = type_traits_cpu[pType].vec_dot;
  enum ggml_type const cVecDotType = type_traits_cpu[pType].vec_dot_type;

  // broadcast factors
  const int64_t cR2 = ne12 / ne02;
  const int64_t cR3 = ne13 / ne03;

  // printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start,
  // ir1_end);

  // threads with no work simply yield (not sure if it helps)
  if (pIr0Start >= pIr0End || pIr1Start >= pIr1End) {
    return;
  }

  const void *vWdata = (src1->type == cVecDotType) ? src1->data : pParams->wdata;
  const size_t cRowSize = ggml_row_size(cVecDotType, ne10);

  assert(ne12 % ne02 == 0);
  assert(ne13 % ne03 == 0);

  // block-tiling attempt
  const int64_t cBlck0 = 16;
  const int64_t cBlck1 = 16;

  const size_t cSrc1ColStride = cSrc1Cont || src1->type != cVecDotType ? cRowSize : nb11;

  // attempt to reduce false-sharing (does not seem to make a difference)
  // 16 * 2, accounting for mmla kernels
  float aTmp[32];

  for (int64_t vIir1 = pIr1Start; vIir1 < pIr1End; vIir1 += cBlck1) {
    for (int64_t vIir0 = pIr0Start; vIir0 < pIr0End; vIir0 += cBlck0) {
      for (int64_t vIr1 = vIir1; vIr1 < vIir1 + cBlck1 && vIr1 < pIr1End; vIr1 += pNumRowsPerVecDot) {
        const int64_t cI13 = (vIr1 / (ne12 * ne1));
        const int64_t cI12 = (vIr1 - cI13 * ne12 * ne1) / ne1;
        const int64_t cI11 = (vIr1 - cI13 * ne12 * ne1 - cI12 * ne1);

        // broadcast src0 into src1
        const int64_t cI03 = cI13 / cR3;
        const int64_t cI02 = cI12 / cR2;

        const int64_t cI1 = cI11;
        const int64_t cI2 = cI12;
        const int64_t cI3 = cI13;

        const char *vSrc0Row = (const char *)src0->data + (0 + cI02 * nb02 + cI03 * nb03);

        // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
        //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
        //       the original src1 data pointer, so we should index using the indices directly
        // TODO: this is a bit of a hack, we should probably have a better way to handle this
        const char *vSrc1Col = (const char *)vWdata + (cSrc1Cont || src1->type != cVecDotType
                                                           ? (cI11 + cI12 * ne11 + cI13 * ne12 * ne11) * cRowSize
                                                           : (cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
        float *vDstCol = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

        // for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
        //     vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
        // }

        for (int64_t vIr0 = vIir0; vIr0 < vIir0 + cBlck0 && vIr0 < pIr0End; vIr0 += pNumRowsPerVecDot) {
          cVecDot(ne00, &aTmp[vIr0 - vIir0], (pNumRowsPerVecDot > 1 ? 16 : 0), vSrc0Row + vIr0 * nb01,
                  (pNumRowsPerVecDot > 1 ? nb01 : 0), vSrc1Col, (pNumRowsPerVecDot > 1 ? cSrc1ColStride : 0),
                  pNumRowsPerVecDot);
        }

        for (int vCn = 0; vCn < pNumRowsPerVecDot; ++vCn) {
          memcpy(&vDstCol[vIir0 + vCn * nb1 / nb0], aTmp + (vCn * 16),
                 (MIN(vIir0 + cBlck0, pIr0End) - vIir0) * sizeof(float));
        }
      }
    }
  }
}

void ggml_compute_forward_mul_mat(const struct ggml_compute_params *pParams, struct ggml_tensor *dst) {

  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];

  const int32_t cHint = ggml_get_op_params_i32(dst, 1);
  if (cHint == GGML_HINT_SRC0_IS_HADAMARD && !pParams->use_ref) {
    ggml_compute_forward_fwht(pParams, dst);
    return;
  }

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = pParams->ith;
  const int cNth = pParams->nth;

  enum ggml_type const cVecDotType = type_traits_cpu[src0->type].vec_dot_type;
  ggml_from_float_t const cFromFloat = type_traits_cpu[cVecDotType].from_float;
  int64_t const cVecDotNumRows = type_traits_cpu[src0->type].nrows;

  GGML_ASSERT(ne0 == ne01);
  GGML_ASSERT(ne1 == ne11);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == ggml_type_size(src0->type));
  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  // TODO: extract to "extra_op"
#if GGML_USE_TINYBLAS
  // broadcast factors
  const int64_t cR2 = ne12 / ne02;
  const int64_t cR3 = ne13 / ne03;

  const bool cSrc1Cont = ggml_is_contiguous(src1);

  if (cSrc1Cont) {
    for (int64_t vI13 = 0; vI13 < ne13; vI13++)
      for (int64_t vI12 = 0; vI12 < ne12; vI12++)
        if (!tinyblas_sgemm(pParams, ne01, ne11, ne00 / ggml_blck_size(src0->type),
                            (const char *)src0->data + vI12 / cR2 * nb02 + vI13 / cR3 * nb03,
                            nb01 / ggml_type_size(src0->type), (const char *)src1->data + vI12 * nb12 + vI13 * nb13,
                            nb11 / ggml_type_size(src1->type), (char *)dst->data + vI12 * nb2 + vI13 * nb3,
                            nb1 / ggml_type_size(dst->type), src0->type, src1->type, dst->type))
          goto UseGgmlGemm1;
    return;
  }
UseGgmlGemm1:;
#endif

  if (src1->type != cVecDotType) {
    char *vWdata = pParams->wdata;

    const size_t cNbw0 = ggml_type_size(cVecDotType);
    const size_t cNbw1 = ggml_row_size(cVecDotType, ne10);
    const size_t cNbw2 = cNbw1 * ne11;
    const size_t cNbw3 = cNbw2 * ne12;

    assert(pParams->wsize >= ne13 * cNbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t vI13 = 0; vI13 < ne13; ++vI13) {
            for (int64_t vI12 = 0; vI12 < ne12; ++vI12) {
                for (int64_t vI11 = ith; vI11 < ne11; vI11 += cNth) {
                    cFromFloat((float *)((char *) src1->data + vI13*nb13 + vI12*nb12 + vI11*nb11),
                               (void *)               (vWdata + vI13*cNbw3 + vI12*cNbw2 + vI11*cNbw1),
                                ne10);
                }
            }
        }
#else
    for (int64_t vI13 = 0; vI13 < ne13; ++vI13) {
      for (int64_t vI12 = 0; vI12 < ne12; ++vI12) {
        for (int64_t vI11 = 0; vI11 < ne11; ++vI11) {
          size_t vBs = ggml_blck_size(cVecDotType);
          int64_t vNe10BlockStart = (ith * ne10 / vBs) / cNth;
          int64_t vNe10BlockEnd = ((ith + 1) * ne10 / vBs) / cNth;
          cFromFloat(
              (float *)((char *)src1->data + vI13 * nb13 + vI12 * nb12 + vI11 * nb11 + vNe10BlockStart * vBs * nb10),
              (void *)(vWdata + vI13 * cNbw3 + vI12 * cNbw2 + vI11 * cNbw1 + vNe10BlockStart * cNbw0),
              (vNe10BlockEnd - vNe10BlockStart) * vBs);
        }
      }
    }
#endif
  }

  if (ith == 0) {
    // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the
    // start.
    atomic_store_explicit(&pParams->threadpool->vCurrentChunk, cNth, memory_order_relaxed);
  }

  ggml_barrier(pParams->threadpool);

#if GGML_USE_TINYBLAS
  if (src1->type != cVecDotType) {
    const void *vWdata = (src1->type == cVecDotType) ? src1->data : pParams->wdata;
    const size_t cRowSize = ggml_row_size(cVecDotType, ne10);

    for (int64_t vI13 = 0; vI13 < ne13; vI13++)
      for (int64_t vI12 = 0; vI12 < ne12; vI12++)
        if (!tinyblas_sgemm(pParams, ne01, ne11, ne00 / ggml_blck_size(src0->type),
                            (const char *)src0->data + vI12 / cR2 * nb02 + vI13 / cR3 * nb03,
                            nb01 / ggml_type_size(src0->type),
                            (const char *)vWdata + (vI12 * ne11 + vI13 * ne12 * ne11) * cRowSize,
                            cRowSize / ggml_type_size(cVecDotType), (char *)dst->data + vI12 * nb2 + vI13 * nb3,
                            nb1 / ggml_type_size(dst->type), src0->type, cVecDotType, dst->type))
          goto UseGgmlGemm2;
    return;
  }
UseGgmlGemm2:;
#endif

  // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are
  // the same numbers)
  const int64_t cNr0 = ne0;

  // This is the size of the rest of the dimensions of the result
  const int64_t cNr1 = ne1 * ne2 * ne3;

  // Now select a reasonable chunk size.
  int vChunkSize = 16;

  // We need to step up the size if it's small
  if (cNr0 == 1 || cNr1 == 1) {
    vChunkSize = 64;
  }

  // distribute the work across the inner or outer loop based on which one is larger
  // The number of chunks in the 0/1 dim.
  // CEIL(nr0/chunk_size)
  int64_t vNchunk0 = (cNr0 + vChunkSize - 1) / vChunkSize;
  int64_t vNchunk1 = (cNr1 + vChunkSize - 1) / vChunkSize;

  // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
  //   Also, chunking by thread was measured to have perform better on NUMA systems.
  //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
  if (vNchunk0 * vNchunk1 < cNth * 4 || ggml_is_numa()) {
    // distribute the thread work across the inner or outer loop based on which one is larger
    vNchunk0 = cNr0 > cNr1 ? cNth : 1; // parallelize by src0 rows
    vNchunk1 = cNr0 > cNr1 ? 1 : cNth; // parallelize by src1 rows
  }

  // The number of elements in each chunk
  const int64_t cDr0 = (cNr0 + vNchunk0 - 1) / vNchunk0;
  const int64_t cDr1 = (cNr1 + vNchunk1 - 1) / vNchunk1;

  // The first chunk comes from our thread_id, the rest will get auto-assigned.
  int vCurrentChunk = ith;

  while (vCurrentChunk < vNchunk0 * vNchunk1) {
    const int64_t cIth0 = vCurrentChunk % vNchunk0;
    const int64_t cIth1 = vCurrentChunk / vNchunk0;

    const int64_t cIr0Start = cDr0 * cIth0;
    const int64_t cIr0End = MIN(cIr0Start + cDr0, cNr0);

    const int64_t cIr1Start = cDr1 * cIth1;
    const int64_t cIr1End = MIN(cIr1Start + cDr1, cNr1);

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t vNumRowsPerVecDot = cVecDotNumRows;

    // these checks are needed to avoid crossing dim1 boundaries
    // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
    if ((cNr0 % 2 != 0) || (ne11 % 2 != 0) || ((cIr0End - cIr0Start) % 2 != 0) || ((cIr1End - cIr1Start) % 2 != 0)) {
      vNumRowsPerVecDot = 1;
    }
    fGgmlComputeForwardMulMatOneChunk(pParams, dst, src0->type, vNumRowsPerVecDot, cIr0Start, cIr0End, cIr1Start,
                                      cIr1End);

    if (cNth >= vNchunk0 * vNchunk1) {
      break;
    }

    vCurrentChunk = atomic_fetch_add_explicit(&pParams->threadpool->vCurrentChunk, 1, memory_order_relaxed);
  }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ids->ne[0] * ids->ne[1] + (i1)]

struct mmid_row_mapping {
  int32_t i1;
  int32_t vI2;
};

static void fGgmlComputeForwardMulMatIdOneChunk(struct ggml_tensor *dst, const struct ggml_tensor *src0,
                                                const struct ggml_tensor *src1, const struct ggml_tensor *ids,
                                                const int64_t pCurA, const int64_t pIr0Start, const int64_t pIr0End,
                                                const int64_t pIr1Start, const int64_t pIr1End, const char *pSrc0Cur,
                                                const struct mmid_row_mapping *matrix_rows, const size_t pRowSize,
                                                const bool pSrc1Cont, const void *pWdata) {

  GGML_TENSOR_BINARY_OP_LOCALS

  const enum ggml_type cType = src0->type;

  ggml_vec_dot_t const cVecDot = type_traits_cpu[cType].vec_dot;
  enum ggml_type const cVecDotType = type_traits_cpu[cType].vec_dot_type;

  const int64_t cBlck0 = 16;
  const int64_t cBlck1 = 16;

  float aTmp[16];

  for (int64_t vIir1 = pIr1Start; vIir1 < pIr1End; vIir1 += cBlck1) {
    for (int64_t vIir0 = pIr0Start; vIir0 < pIr0End; vIir0 += cBlck0) {
      for (int64_t vIr1 = vIir1; vIr1 < vIir1 + cBlck1 && vIr1 < pIr1End; ++vIr1) {
        const int64_t _i12 = vIr1; // logical row index for this expert

        struct mmid_row_mapping vRowMapping = MMID_MATRIX_ROW(pCurA, _i12);
        const int cId = vRowMapping.i1; // selected expert index

        const int64_t cI11 = cId % ne11;
        const int64_t i12 = vRowMapping.vI2; // row index in src1

        const int64_t cI1 = cId; // selected expert index
        const int64_t cI2 = i12; // row

        // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
        //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
        //       the original src1 data pointer, so we should index using the indices directly
        // TODO: this is a bit of a hack, we should probably have a better way to handle this
        const char *vSrc1Col =
            (const char *)pWdata +
            (pSrc1Cont || src1->type != cVecDotType ? (cI11 + i12 * ne11) * pRowSize : (cI11 * nb11 + i12 * nb12));

        float *vDstCol = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2));

        for (int64_t vIr0 = vIir0; vIr0 < vIir0 + cBlck0 && vIr0 < pIr0End; ++vIr0) {
          cVecDot(ne00, &aTmp[vIr0 - vIir0], 0, pSrc0Cur + vIr0 * nb01, 0, vSrc1Col, 0, 1);
        }

        memcpy(&vDstCol[vIir0], aTmp, (MIN(vIir0 + cBlck0, pIr0End) - vIir0) * sizeof(float));
      }
    }
  }
}

static void *fIncrPtrAligned(void **p_, size_t pSize, size_t pAlign) {

  void *vPtr = *p_;
  vPtr = (void *)GGML_PAD((uintptr_t)vPtr, pAlign);
  *p_ = (void *)((char *)vPtr + pSize);
  return vPtr;
}

static void fGgmlComputeForwardMulMatId(const struct ggml_compute_params *pParams, struct ggml_tensor *dst) {

  const struct ggml_tensor *src0 = dst->src[0];
  const struct ggml_tensor *src1 = dst->src[1];
  const struct ggml_tensor *ids = dst->src[2];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = pParams->ith;
  const int cNth = pParams->nth;

  const enum ggml_type cType = src0->type;

  const bool cSrc1Cont = ggml_is_contiguous(src1);

  enum ggml_type const cVecDotType = type_traits_cpu[cType].vec_dot_type;
  ggml_from_float_t const cFromFloat = type_traits_cpu[cVecDotType].from_float;

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == ggml_type_size(cType));
  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  // row groups
  const int cNIds = ids->ne[0]; // n_expert_used
  const int cNAs = ne02;        // n_expert

  void *vWdataCur = pParams->wdata;

  if (src1->type != cVecDotType) {
    fIncrPtrAligned(&vWdataCur, ggml_row_size(cVecDotType, ggml_nelements(src1)), sizeof(int64_t));
  }

  int64_t *vMatrixRowCounts = // [n_as]
      fIncrPtrAligned(&vWdataCur, cNAs * sizeof(int64_t), sizeof(int64_t));

  struct mmid_row_mapping *matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
      fIncrPtrAligned(&vWdataCur, cNAs * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping), sizeof(int64_t));

  char(*vAtomicCurrentChunk)[CACHE_LINE_SIZE] = // [n_as]
      fIncrPtrAligned(&vWdataCur, CACHE_LINE_SIZE * cNAs, CACHE_LINE_SIZE);

  GGML_ASSERT(pParams->wsize >= (size_t)((char *)vWdataCur - (char *)pParams->wdata));

  if (src1->type != cVecDotType) {
    char *vWdata = pParams->wdata;

    const size_t cNbw0 = ggml_type_size(cVecDotType);
    const size_t cNbw1 = ggml_row_size(cVecDotType, ne10);
    const size_t cNbw2 = cNbw1 * ne11;
    const size_t cNbw3 = cNbw2 * ne12;

    assert(pParams->wsize >= ne13 * cNbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t vI13 = 0; vI13 < ne13; ++vI13) {
            for (int64_t vI12 = ith; vI12 < ne12; vI12 += cNth) {
                for (int64_t vI11 = 0; vI11 < ne11; ++vI11) {
                    cFromFloat((float *)((char *) src1->data + vI13*nb13 + vI12*nb12 + vI11*nb11),
                               (void *)               (vWdata + vI13*cNbw3 + vI12*cNbw2 + vI11*cNbw1),
                               ne10);
                }
            }
        }
#else
    for (int64_t vI13 = 0; vI13 < ne13; ++vI13) {
      for (int64_t vI12 = 0; vI12 < ne12; ++vI12) {
        for (int64_t vI11 = 0; vI11 < ne11; ++vI11) {
          size_t vBs = ggml_blck_size(cVecDotType);
          int64_t vNe10BlockStart = (ith * ne10 / vBs) / cNth;
          int64_t vNe10BlockEnd = ((ith + 1) * ne10 / vBs) / cNth;
          cFromFloat(
              (float *)((char *)src1->data + vI13 * nb13 + vI12 * nb12 + vI11 * nb11 + vNe10BlockStart * vBs * nb10),
              (void *)(vWdata + vI13 * cNbw3 + vI12 * cNbw2 + vI11 * cNbw1 + vNe10BlockStart * cNbw0),
              (vNe10BlockEnd - vNe10BlockStart) * vBs);
        }
      }
    }
#endif
  }

  if (ith == 0) {
    // initialize matrix_row_counts
    memset(vMatrixRowCounts, 0, cNAs * sizeof(int64_t));

    // group rows by src0 matrix
    for (int64_t vIid1 = 0; vIid1 < ids->ne[1]; ++vIid1) {
      for (int vId = 0; vId < cNIds; ++vId) {
        const int32_t cI02 = *(const int32_t *)((const char *)ids->data + vIid1 * ids->nb[1] + vId * ids->nb[0]);

        // A negative identifier means that another backend computes this expert slot
        // (hybrid MoE execution). Its output row is zeroed so the partial results add up.
        if (cI02 < 0) {
          memset((char *)dst->data + vId * nb1 + vIid1 * nb2, 0, ne0 * sizeof(float));
          continue;
        }

        assert(cI02 < cNAs);

        MMID_MATRIX_ROW(cI02, vMatrixRowCounts[cI02]) = (struct mmid_row_mapping){vId, vIid1};
        vMatrixRowCounts[cI02] += 1;
      }
    }
  }

  // reset current_chunk
  for (int vCurA = ith; vCurA < cNAs; vCurA += cNth) {
    atomic_int *vCurrentChunkCtr = (atomic_int *)(vAtomicCurrentChunk + vCurA);
    *vCurrentChunkCtr = cNth;
  }

  ggml_barrier(pParams->threadpool);

  for (int vCurA = 0; vCurA < cNAs; ++vCurA) {
    const int64_t cCne1 = vMatrixRowCounts[vCurA];

    if (cCne1 == 0) {
      continue;
    }

    const char *vSrc0Cur = (const char *)src0->data + vCurA * nb02;
    const void *vWdata = (src1->type == cVecDotType) ? src1->data : pParams->wdata;
    const size_t cRowSize = ggml_row_size(cVecDotType, ne10);

    const int64_t cNr0 = ne01;
    const int64_t cNr1 = cCne1;

    int vChunkSize = 16;
    if (cNr0 == 1 || cNr1 == 1) {
      vChunkSize = 64;
    }

    // disable for NUMA
    const bool cDisableChunking = ggml_is_numa();

    int64_t vNchunk0 = (cNr0 + vChunkSize - 1) / vChunkSize;
    int64_t vNchunk1 = (cNr1 + vChunkSize - 1) / vChunkSize;

    if (vNchunk0 * vNchunk1 < cNth * 4 || cDisableChunking) {
      vNchunk0 = cNr0 > cNr1 ? cNth : 1;
      vNchunk1 = cNr0 > cNr1 ? 1 : cNth;
    }

    const int64_t cDr0 = (cNr0 + vNchunk0 - 1) / vNchunk0;
    const int64_t cDr1 = (cNr1 + vNchunk1 - 1) / vNchunk1;

    int vCurrentChunk = ith;

    atomic_int *vCurrentChunkCtr = (atomic_int *)(vAtomicCurrentChunk + vCurA);

    while (vCurrentChunk < vNchunk0 * vNchunk1) {
      const int64_t cIth0 = vCurrentChunk % vNchunk0;
      const int64_t cIth1 = vCurrentChunk / vNchunk0;

      const int64_t cIr0Start = cDr0 * cIth0;
      const int64_t cIr0End = MIN(cIr0Start + cDr0, cNr0);

      const int64_t cIr1Start = cDr1 * cIth1;
      const int64_t cIr1End = MIN(cIr1Start + cDr1, cNr1);

      fGgmlComputeForwardMulMatIdOneChunk(dst, src0, src1, ids, vCurA, cIr0Start, cIr0End, cIr1Start, cIr1End, vSrc0Cur,
                                          matrix_rows, cRowSize, cSrc1Cont, vWdata);

      if (cNth >= vNchunk0 * vNchunk1) {
        break;
      }

      vCurrentChunk = atomic_fetch_add_explicit(vCurrentChunkCtr, 1, memory_order_relaxed);
    }
  }
}

/////////////////////////////////

static void fGgmlComputeForward(struct ggml_compute_params *pParams, struct ggml_tensor *pTensor) {
  GGML_ASSERT(pParams);

  if (pTensor->op == GGML_OP_NONE || ggml_is_empty(pTensor)) {
    return;
  }

  // extra_buffer op?
  if (ggml_cpu_extra_compute_forward(pParams, pTensor)) {
    return;
  }

  switch (pTensor->op) {
  case GGML_OP_DUP: {
    ggml_compute_forward_dup(pParams, pTensor);
  } break;
  case GGML_OP_ADD: {
    ggml_compute_forward_add(pParams, pTensor);
  } break;
  case GGML_OP_ADD_ID: {
    ggml_compute_forward_add_id(pParams, pTensor);
  } break;
  case GGML_OP_ADD1: {
    ggml_compute_forward_add1(pParams, pTensor);
  } break;
  case GGML_OP_ACC: {
    ggml_compute_forward_acc(pParams, pTensor);
  } break;
  case GGML_OP_SUB: {
    ggml_compute_forward_sub(pParams, pTensor);
  } break;
  case GGML_OP_MUL: {
    ggml_compute_forward_mul(pParams, pTensor);
  } break;
  case GGML_OP_DIV: {
    ggml_compute_forward_div(pParams, pTensor);
  } break;
  case GGML_OP_SQR: {
    ggml_compute_forward_sqr(pParams, pTensor);
  } break;
  case GGML_OP_SQRT: {
    ggml_compute_forward_sqrt(pParams, pTensor);
  } break;
  case GGML_OP_LOG: {
    ggml_compute_forward_log(pParams, pTensor);
  } break;
  case GGML_OP_SIN: {
    ggml_compute_forward_sin(pParams, pTensor);
  } break;
  case GGML_OP_COS: {
    ggml_compute_forward_cos(pParams, pTensor);
  } break;
  case GGML_OP_SUM: {
    ggml_compute_forward_sum(pParams, pTensor);
  } break;
  case GGML_OP_SUM_ROWS: {
    ggml_compute_forward_sum_rows(pParams, pTensor);
  } break;
  case GGML_OP_CUMSUM: {
    ggml_compute_forward_cumsum(pParams, pTensor);
  } break;
  case GGML_OP_MEAN: {
    ggml_compute_forward_mean(pParams, pTensor);
  } break;
  case GGML_OP_ARGMAX: {
    ggml_compute_forward_argmax(pParams, pTensor);
  } break;
  case GGML_OP_AMAX_POW2: {
    ggml_compute_forward_amax_pow2(pParams, pTensor);
  } break;
  case GGML_OP_COUNT_EQUAL: {
    ggml_compute_forward_count_equal(pParams, pTensor);
  } break;
  case GGML_OP_REPEAT: {
    ggml_compute_forward_repeat(pParams, pTensor);
  } break;
  case GGML_OP_REPEAT_BACK: {
    ggml_compute_forward_repeat_back(pParams, pTensor);
  } break;
  case GGML_OP_CONCAT: {
    ggml_compute_forward_concat(pParams, pTensor);
  } break;
  case GGML_OP_SILU_BACK: {
    ggml_compute_forward_silu_back(pParams, pTensor);
  } break;
  case GGML_OP_NORM: {
    ggml_compute_forward_norm(pParams, pTensor);
  } break;
  case GGML_OP_RMS_NORM: {
    ggml_compute_forward_rms_norm(pParams, pTensor);
  } break;
  case GGML_OP_RMS_NORM_BACK: {
    ggml_compute_forward_rms_norm_back(pParams, pTensor);
  } break;
  case GGML_OP_GROUP_NORM: {
    ggml_compute_forward_group_norm(pParams, pTensor);
  } break;
  case GGML_OP_L2_NORM: {
    ggml_compute_forward_l2_norm(pParams, pTensor);
  } break;
  case GGML_OP_MUL_MAT: {
    ggml_compute_forward_mul_mat(pParams, pTensor);
  } break;
  case GGML_OP_MUL_MAT_ID: {
    fGgmlComputeForwardMulMatId(pParams, pTensor);
  } break;
  case GGML_OP_OUT_PROD: {
    ggml_compute_forward_out_prod(pParams, pTensor);
  } break;
  case GGML_OP_SCALE: {
    ggml_compute_forward_scale(pParams, pTensor);
  } break;
  case GGML_OP_SET: {
    ggml_compute_forward_set(pParams, pTensor);
  } break;
  case GGML_OP_CPY: {
    ggml_compute_forward_cpy(pParams, pTensor);
  } break;
  case GGML_OP_CONT: {
    ggml_compute_forward_cont(pParams, pTensor);
  } break;
  case GGML_OP_GET_ROWS: {
    ggml_compute_forward_get_rows(pParams, pTensor);
  } break;
  case GGML_OP_GET_ROWS_BACK: {
    ggml_compute_forward_get_rows_back(pParams, pTensor);
  } break;
  case GGML_OP_SET_ROWS: {
    ggml_compute_forward_set_rows(pParams, pTensor);
  } break;
  case GGML_OP_DIAG: {
    ggml_compute_forward_diag(pParams, pTensor);
  } break;
  case GGML_OP_DIAG_MASK_INF: {
    ggml_compute_forward_diag_mask_inf(pParams, pTensor);
  } break;
  case GGML_OP_DIAG_MASK_ZERO: {
    ggml_compute_forward_diag_mask_zero(pParams, pTensor);
  } break;
  case GGML_OP_SOFT_MAX: {
    ggml_compute_forward_soft_max(pParams, pTensor);
  } break;
  case GGML_OP_SOFT_MAX_BACK: {
    ggml_compute_forward_soft_max_ext_back(pParams, pTensor);
  } break;
  case GGML_OP_ROPE: {
    ggml_compute_forward_rope(pParams, pTensor);
  } break;
  case GGML_OP_ROPE_BACK: {
    ggml_compute_forward_rope_back(pParams, pTensor);
  } break;
  case GGML_OP_CLAMP: {
    ggml_compute_forward_clamp(pParams, pTensor);
  } break;
  case GGML_OP_CONV_TRANSPOSE_1D: {
    ggml_compute_forward_conv_transpose_1d(pParams, pTensor);
  } break;
  case GGML_OP_IM2COL: {
    ggml_compute_forward_im2col(pParams, pTensor);
  } break;
  case GGML_OP_IM2COL_BACK: {
    ggml_compute_forward_im2col_back_f32(pParams, pTensor);
  } break;
  case GGML_OP_IM2COL_3D: {
    ggml_compute_forward_im2col_3d(pParams, pTensor);
  } break;
  case GGML_OP_COL2IM_1D: {
    ggml_compute_forward_col2im_1d(pParams, pTensor);
  } break;
  case GGML_OP_CONV_2D: {
    ggml_compute_forward_conv_2d(pParams, pTensor);
  } break;
  case GGML_OP_CONV_3D: {
    ggml_compute_forward_conv_3d(pParams, pTensor);
  } break;
  case GGML_OP_CONV_2D_DW: {
    ggml_compute_forward_conv_2d_dw(pParams, pTensor);
  } break;
  case GGML_OP_CONV_TRANSPOSE_2D: {
    ggml_compute_forward_conv_transpose_2d(pParams, pTensor);
  } break;
  case GGML_OP_POOL_1D: {
    ggml_compute_forward_pool_1d(pParams, pTensor);
  } break;
  case GGML_OP_POOL_2D: {
    ggml_compute_forward_pool_2d(pParams, pTensor);
  } break;
  case GGML_OP_POOL_2D_BACK: {
    ggml_compute_forward_pool_2d_back(pParams, pTensor);
  } break;
  case GGML_OP_UPSCALE: {
    ggml_compute_forward_upscale(pParams, pTensor);
  } break;
  case GGML_OP_PAD: {
    ggml_compute_forward_pad(pParams, pTensor);
  } break;
  case GGML_OP_PAD_REFLECT_1D: {
    ggml_compute_forward_pad_reflect_1d(pParams, pTensor);
  } break;
  case GGML_OP_ROLL: {
    ggml_compute_forward_roll(pParams, pTensor);
  } break;
  case GGML_OP_ARANGE: {
    ggml_compute_forward_arange(pParams, pTensor);
  } break;
  case GGML_OP_TIMESTEP_EMBEDDING: {
    ggml_compute_forward_timestep_embedding(pParams, pTensor);
  } break;
  case GGML_OP_ARGSORT: {
    ggml_compute_forward_argsort(pParams, pTensor);
  } break;
  case GGML_OP_TOP_K: {
    ggml_compute_forward_top_k(pParams, pTensor);
  } break;
  case GGML_OP_LEAKY_RELU: {
    ggml_compute_forward_leaky_relu(pParams, pTensor);
  } break;
  case GGML_OP_TRI: {
    ggml_compute_forward_tri(pParams, pTensor);
  } break;
  case GGML_OP_FILL: {
    ggml_compute_forward_fill(pParams, pTensor);
  } break;
  case GGML_OP_FLASH_ATTN_EXT: {
    ggml_compute_forward_flash_attn_ext(pParams, pTensor);
  } break;
  case GGML_OP_FLASH_ATTN_BACK: {
    int32_t vT = ggml_get_op_params_i32(pTensor, 0);
    GGML_ASSERT(vT == 0 || vT == 1);
    bool vMasked = vT != 0;
    ggml_compute_forward_flash_attn_back(pParams, vMasked, pTensor);
  } break;
  case GGML_OP_SSM_CONV: {
    ggml_compute_forward_ssm_conv(pParams, pTensor);
  } break;
  case GGML_OP_SSM_SCAN: {
    ggml_compute_forward_ssm_scan(pParams, pTensor);
  } break;
  case GGML_OP_WIN_PART: {
    ggml_compute_forward_win_part(pParams, pTensor);
  } break;
  case GGML_OP_WIN_UNPART: {
    ggml_compute_forward_win_unpart(pParams, pTensor);
  } break;
  case GGML_OP_UNARY: {
    ggml_compute_forward_unary(pParams, pTensor);
  } break;
  case GGML_OP_GLU: {
    ggml_compute_forward_glu(pParams, pTensor);
  } break;
  case GGML_OP_GET_REL_POS: {
    ggml_compute_forward_get_rel_pos(pParams, pTensor);
  } break;
  case GGML_OP_ADD_REL_POS: {
    ggml_compute_forward_add_rel_pos(pParams, pTensor);
  } break;
  case GGML_OP_RWKV_WKV6: {
    ggml_compute_forward_rwkv_wkv6(pParams, pTensor);
  } break;
  case GGML_OP_GATED_LINEAR_ATTN: {
    ggml_compute_forward_gla(pParams, pTensor);
  } break;
  case GGML_OP_RWKV_WKV7: {
    ggml_compute_forward_rwkv_wkv7(pParams, pTensor);
  } break;
  case GGML_OP_SOLVE_TRI: {
    ggml_compute_forward_solve_tri(pParams, pTensor);
  } break;
  case GGML_OP_GATED_DELTA_NET: {
    ggml_compute_forward_gated_delta_net(pParams, pTensor);
  } break;
  case GGML_OP_LIGHTNING_INDEXER: {
    ggml_compute_forward_lightning_indexer(pParams, pTensor);
  } break;
  case GGML_OP_DSV4_HC_COMB: {
    ggml_compute_forward_dsv4_hc_comb(pParams, pTensor);
  } break;
  case GGML_OP_DSV4_HC_PRE: {
    ggml_compute_forward_dsv4_hc_pre(pParams, pTensor);
  } break;
  case GGML_OP_DSV4_HC_POST: {
    ggml_compute_forward_dsv4_hc_post(pParams, pTensor);
  } break;
  case GGML_OP_MAP_CUSTOM1: {
    ggml_compute_forward_map_custom1(pParams, pTensor);
  } break;
  case GGML_OP_MAP_CUSTOM2: {
    ggml_compute_forward_map_custom2(pParams, pTensor);
  } break;
  case GGML_OP_MAP_CUSTOM3: {
    ggml_compute_forward_map_custom3(pParams, pTensor);
  } break;
  case GGML_OP_CUSTOM: {
    ggml_compute_forward_custom(pParams, pTensor);
  } break;
  case GGML_OP_CROSS_ENTROPY_LOSS: {
    ggml_compute_forward_cross_entropy_loss(pParams, pTensor);
  } break;
  case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
    ggml_compute_forward_cross_entropy_loss_back(pParams, pTensor);
  } break;
  case GGML_OP_OPT_STEP_ADAMW: {
    ggml_compute_forward_opt_step_adamw(pParams, pTensor);
  } break;
  case GGML_OP_OPT_STEP_SGD: {
    ggml_compute_forward_opt_step_sgd(pParams, pTensor);
  } break;
  case GGML_OP_NONE: {
    // nop
  } break;
  case GGML_OP_RESHAPE: {
    // nop
  } break;
  case GGML_OP_PERMUTE: {
    // nop
  } break;
  case GGML_OP_VIEW: {
    // nop
  } break;
  case GGML_OP_TRANSPOSE: {
    // nop
  } break;
  case GGML_OP_COUNT: {
    GGML_ABORT("fatal error");
  }
  }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void fSetNumaThreadAffinity(int pThreadN) {
  if (!ggml_is_numa()) {
    return;
  }

  int vNodeNum;
  int vRv;
  size_t vSetsize = CPU_ALLOC_SIZE(g_state.vNuma.vTotalCpus);

  switch (g_state.vNuma.vNumaStrategy) {
  case GGML_NUMA_STRATEGY_DISTRIBUTE:
    // run thread on node_num thread_n / (threads per node)
    vNodeNum = pThreadN % g_state.vNuma.n_nodes;
    break;
  case GGML_NUMA_STRATEGY_ISOLATE:
    // run thread on current_node
    vNodeNum = g_state.vNuma.vCurrentNode;
    break;
  case GGML_NUMA_STRATEGY_NUMACTL:
    // use the cpuset that numactl gave us
    vRv = pthread_setaffinity_np(pthread_self(), vSetsize, &g_state.vNuma.cpuset);
    if (vRv) {
      fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(vRv));
    }
    return;
  default:
    return;
  }

  struct ggml_numa_node *vNode = &g_state.vNuma.aNodes[vNodeNum];

  cpu_set_t *vCpus = CPU_ALLOC(g_state.vNuma.vTotalCpus);
  CPU_ZERO_S(vSetsize, vCpus);
  for (size_t vI = 0; vI < vNode->vNCpus; ++vI) {
    CPU_SET_S(vNode->vCpus[vI], vSetsize, vCpus);
  }

  vRv = pthread_setaffinity_np(pthread_self(), vSetsize, vCpus);
  if (vRv) {
    fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(vRv));
  }

  CPU_FREE(vCpus);
}

static void fClearNumaThreadAffinity(void) {
  if (!ggml_is_numa()) {
    return;
  }

  size_t vSetsize = CPU_ALLOC_SIZE(g_state.vNuma.vTotalCpus);

  cpu_set_t *vCpus = CPU_ALLOC(g_state.vNuma.vTotalCpus);
  CPU_ZERO_S(vSetsize, vCpus);
  for (unsigned vI = 0; vI < g_state.vNuma.vTotalCpus; ++vI) {
    CPU_SET_S(vI, vSetsize, vCpus);
  }

  int vRv = pthread_setaffinity_np(pthread_self(), vSetsize, vCpus);
  if (vRv) {
    fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(vRv));
  }

  CPU_FREE(vCpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void fSetNumaThreadAffinity(int thread_n) { UNUSED(thread_n); }
static void fClearNumaThreadAffinity(void) {}
#endif

static int fGgmlGetNTasks(struct ggml_tensor *pNode, int pNThreads) {
  int vNTasks = 0;

  if (ggml_is_empty(pNode)) {
    // no need to multi-thread a no-op
    vNTasks = 1;
    return vNTasks;
  }

  switch (pNode->op) {
  case GGML_OP_CPY:
  case GGML_OP_DUP:
  case GGML_OP_CONT:
  case GGML_OP_ADD:
  case GGML_OP_ADD_ID:
  case GGML_OP_ADD1:
  case GGML_OP_ACC:
  case GGML_OP_CUMSUM:
  case GGML_OP_TRI:
  case GGML_OP_FILL: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_SUB:
  case GGML_OP_SQR:
  case GGML_OP_SQRT:
  case GGML_OP_LOG:
  case GGML_OP_SIN:
  case GGML_OP_COS:
  case GGML_OP_SUM:
  case GGML_OP_SUM_ROWS:
  case GGML_OP_MEAN:
  case GGML_OP_ARGMAX: {
    vNTasks = 1;
  } break;
  case GGML_OP_COUNT_EQUAL:
  case GGML_OP_AMAX_POW2:
  case GGML_OP_SOLVE_TRI:
  case GGML_OP_GATED_DELTA_NET:
  case GGML_OP_DSV4_HC_COMB:
  case GGML_OP_DSV4_HC_PRE:
  case GGML_OP_DSV4_HC_POST: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_REPEAT:
  case GGML_OP_REPEAT_BACK:
  case GGML_OP_LEAKY_RELU: {
    vNTasks = 1;
  } break;
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(pNode)) {
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SGN:
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_STEP:
    case GGML_UNARY_OP_TANH:
    case GGML_UNARY_OP_ELU:
    case GGML_UNARY_OP_RELU:
    case GGML_UNARY_OP_SIGMOID:
    case GGML_UNARY_OP_HARDSWISH:
    case GGML_UNARY_OP_HARDSIGMOID:
    case GGML_UNARY_OP_EXP:
    case GGML_UNARY_OP_SOFTPLUS:
    case GGML_UNARY_OP_EXPM1:
    case GGML_UNARY_OP_FLOOR:
    case GGML_UNARY_OP_CEIL:
    case GGML_UNARY_OP_ROUND:
    case GGML_UNARY_OP_TRUNC: {
      vNTasks = 1;
    } break;

    case GGML_UNARY_OP_GELU:
    case GGML_UNARY_OP_GELU_ERF:
    case GGML_UNARY_OP_GELU_QUICK:
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_XIELU: {
      vNTasks = pNThreads;
    } break;
    default:
      GGML_ABORT("fatal error");
    }
    break;
  case GGML_OP_GLU:
    switch (ggml_get_glu_op(pNode)) {
    case GGML_GLU_OP_REGLU:
    case GGML_GLU_OP_GEGLU:
    case GGML_GLU_OP_SWIGLU:
    case GGML_GLU_OP_SWIGLU_OAI:
    case GGML_GLU_OP_GEGLU_ERF:
    case GGML_GLU_OP_GEGLU_QUICK:
    case GGML_GLU_OP_SWIGLU_CLAMP: {
      vNTasks = pNThreads;
    } break;
    default:
      GGML_ABORT("fatal error");
    }
    break;
  case GGML_OP_SILU_BACK:
  case GGML_OP_MUL:
  case GGML_OP_DIV:
  case GGML_OP_NORM:
  case GGML_OP_RMS_NORM:
  case GGML_OP_RMS_NORM_BACK:
  case GGML_OP_L2_NORM:
  case GGML_OP_GROUP_NORM:
  case GGML_OP_CONCAT:
  case GGML_OP_MUL_MAT:
  case GGML_OP_MUL_MAT_ID:
  case GGML_OP_OUT_PROD: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_GET_ROWS:
  case GGML_OP_SET_ROWS: {
    // FIXME: get_rows can use additional threads, but the cost of launching additional threads
    // decreases performance with GPU offloading
    // n_tasks = n_threads;
    vNTasks = 1;
  } break;
  case GGML_OP_SCALE:
  case GGML_OP_SET:
  case GGML_OP_RESHAPE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
  case GGML_OP_GET_ROWS_BACK:
  case GGML_OP_DIAG: {
    vNTasks = 1;
  } break;
  case GGML_OP_DIAG_MASK_ZERO:
  case GGML_OP_DIAG_MASK_INF:
  case GGML_OP_SOFT_MAX_BACK:
  case GGML_OP_ROPE:
  case GGML_OP_ROPE_BACK:
  case GGML_OP_ADD_REL_POS: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_CLAMP: {
    vNTasks = 1; // TODO
  } break;
  case GGML_OP_SOFT_MAX: {
    vNTasks = MIN(pNThreads, ggml_nrows(pNode->src[0]));
  } break;
  case GGML_OP_IM2COL:
  case GGML_OP_IM2COL_BACK:
  case GGML_OP_IM2COL_3D:
  case GGML_OP_CONV_2D:
  case GGML_OP_CONV_3D:
  case GGML_OP_CONV_2D_DW:
  case GGML_OP_COL2IM_1D:
  case GGML_OP_CONV_TRANSPOSE_1D:
  case GGML_OP_CONV_TRANSPOSE_2D: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_POOL_1D:
  case GGML_OP_POOL_2D:
  case GGML_OP_POOL_2D_BACK: {
    vNTasks = 1;
  } break;
  case GGML_OP_UPSCALE:
  case GGML_OP_PAD:
  case GGML_OP_PAD_REFLECT_1D:
  case GGML_OP_ROLL:
  case GGML_OP_ARANGE:
  case GGML_OP_TIMESTEP_EMBEDDING:
  case GGML_OP_ARGSORT:
  case GGML_OP_TOP_K:
  case GGML_OP_FLASH_ATTN_EXT:
  case GGML_OP_FLASH_ATTN_BACK:
  case GGML_OP_SSM_CONV:
  case GGML_OP_SSM_SCAN:
  case GGML_OP_LIGHTNING_INDEXER: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_RWKV_WKV6:
  case GGML_OP_GATED_LINEAR_ATTN:
  case GGML_OP_RWKV_WKV7: {
    const int64_t cNHeads = pNode->src[1]->ne[1];
    vNTasks = MIN(pNThreads, cNHeads);
  } break;
  case GGML_OP_WIN_PART:
  case GGML_OP_WIN_UNPART:
  case GGML_OP_GET_REL_POS: {
    vNTasks = 1;
  } break;
  case GGML_OP_MAP_CUSTOM1: {
    struct ggml_map_custom1_op_params vP;
    memcpy(&vP, pNode->op_params, sizeof(vP));
    if (vP.n_tasks == GGML_N_TASKS_MAX) {
      vNTasks = pNThreads;
    } else {
      vNTasks = MIN(vP.n_tasks, pNThreads);
    }
  } break;
  case GGML_OP_MAP_CUSTOM2: {
    struct ggml_map_custom2_op_params vP;
    memcpy(&vP, pNode->op_params, sizeof(vP));
    if (vP.n_tasks == GGML_N_TASKS_MAX) {
      vNTasks = pNThreads;
    } else {
      vNTasks = MIN(vP.n_tasks, pNThreads);
    }
  } break;
  case GGML_OP_MAP_CUSTOM3: {
    struct ggml_map_custom3_op_params vP;
    memcpy(&vP, pNode->op_params, sizeof(vP));
    if (vP.n_tasks == GGML_N_TASKS_MAX) {
      vNTasks = pNThreads;
    } else {
      vNTasks = MIN(vP.n_tasks, pNThreads);
    }
  } break;
  case GGML_OP_CUSTOM: {
    struct ggml_custom_op_params vP;
    memcpy(&vP, pNode->op_params, sizeof(vP));
    if (vP.n_tasks == GGML_N_TASKS_MAX) {
      vNTasks = pNThreads;
    } else {
      vNTasks = MIN(vP.n_tasks, pNThreads);
    }
  } break;
  case GGML_OP_CROSS_ENTROPY_LOSS:
  case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
  case GGML_OP_OPT_STEP_ADAMW:
  case GGML_OP_OPT_STEP_SGD: {
    vNTasks = pNThreads;
  } break;
  case GGML_OP_NONE: {
    vNTasks = 1;
  } break;
  case GGML_OP_COUNT: {
    GGML_ABORT("fatal error");
  }
  default: {
    fprintf(stderr, "%s: op not implemented: ", __func__);
    if (pNode->op < GGML_OP_COUNT) {
      fprintf(stderr, "%s\n", ggml_op_name(pNode->op));
    } else {
      fprintf(stderr, "%d\n", pNode->op);
    }
    GGML_ABORT("fatal error");
  }
  }

  assert(vNTasks > 0);

  return vNTasks;
}

static thread_ret_t fGgmlGraphComputeSecondaryThread(void *pData);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool fGgmlThreadApplyAffinity(bool *mask) {
  HANDLE h = GetCurrentThread();
  uint64_t bitmask = 0ULL;

  assert(GGML_MAX_N_THREADS >= 64);

  for (int32_t i = 0; i < 8; i++) {
    int32_t idx = i * 8;
    uint8_t val = 0;
    val |= mask[idx + 0] << 0;
    val |= mask[idx + 1] << 1;
    val |= mask[idx + 2] << 2;
    val |= mask[idx + 3] << 3;
    val |= mask[idx + 4] << 4;
    val |= mask[idx + 5] << 5;
    val |= mask[idx + 6] << 6;
    val |= mask[idx + 7] << 7;
    bitmask |= (uint64_t)val << idx;
  }

  for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
    if (mask[i]) {
      fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
      break;
    }
  }

  DWORD_PTR m = (DWORD_PTR)bitmask;

  m = SetThreadAffinityMask(h, m);

  return m != 0;
}

static bool fGgmlThreadApplyPriority(int32_t prio) {
  // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
  // This is up to the applications.
  DWORD p = THREAD_PRIORITY_NORMAL;
  switch (prio) {
  case GGML_SCHED_PRIO_LOW:
    p = THREAD_PRIORITY_BELOW_NORMAL;
    break;
  case GGML_SCHED_PRIO_NORMAL:
    p = THREAD_PRIORITY_NORMAL;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    p = THREAD_PRIORITY_ABOVE_NORMAL;
    break;
  case GGML_SCHED_PRIO_HIGH:
    p = THREAD_PRIORITY_HIGHEST;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    p = THREAD_PRIORITY_TIME_CRITICAL;
    break;
  }

  if (prio != GGML_SCHED_PRIO_LOW) {
// Tell Windows that this thread should not be throttled (needs its own CPU core).
// Newer Windows 11 versions aggressively park (offline) CPU cores and often place
// all our threads onto the first 4 cores which results in terrible performance with
// n_threads > 4
#if _WIN32_WINNT >= 0x0602
    THREAD_POWER_THROTTLING_STATE t;
    ZeroMemory(&t, sizeof(t));
    t.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    t.StateMask = 0;

    if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
      GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int)GetLastError());
      return false;
    }
#endif
  }

  if (prio == GGML_SCHED_PRIO_NORMAL) {
    // Keep inherited policy/priority
    return true;
  }

  if (!SetThreadPriority(GetCurrentThread(), p)) {
    fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int)GetLastError());
    return false;
  }

  return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool fGgmlThreadApplyAffinity(const bool *mask) {
  // Not supported on Apple platforms
  UNUSED(mask);
  return true;
}

static bool fGgmlThreadApplyPriority(int32_t prio) {
  struct sched_param p;
  int32_t policy = SCHED_OTHER;
  switch (prio) {
  // TODO: there seems to be no way to set lower prio on Apple platforms
  case GGML_SCHED_PRIO_LOW:
    policy = SCHED_OTHER;
    p.sched_priority = 0;
    break;
  case GGML_SCHED_PRIO_NORMAL:
    policy = SCHED_OTHER;
    p.sched_priority = 0;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    policy = SCHED_FIFO;
    p.sched_priority = 40;
    break;
  case GGML_SCHED_PRIO_HIGH:
    policy = SCHED_FIFO;
    p.sched_priority = 80;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    policy = SCHED_FIFO;
    p.sched_priority = 90;
    break;
  }

  if (prio == GGML_SCHED_PRIO_NORMAL) {
    // Keep inherited policy/priority
    return true;
  }

  int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
  if (err != 0) {
    fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
    return false;
  }

  return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool fGgmlThreadApplyAffinity(const bool *pMask) {
  cpu_set_t cpuset;
  int vErr;

  CPU_ZERO(&cpuset);

  for (uint32_t vI = 0; vI < GGML_MAX_N_THREADS; vI++) {
    if (pMask[vI]) {
      GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), vI);
      CPU_SET(vI, &cpuset);
    }
  }

#ifdef __ANDROID__
  vErr = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (vErr < 0) {
    vErr = errno;
  }
#else
  vErr = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
  if (vErr != 0) {
    fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)pMask, strerror(vErr),
            vErr);
    return false;
  }

  return true;
}

static bool fGgmlThreadApplyPriority(int32_t pPrio) {
  struct sched_param vP;
  int32_t vPolicy = SCHED_OTHER;
  switch (pPrio) {
  case GGML_SCHED_PRIO_LOW:
    vPolicy = SCHED_BATCH;
    vP.sched_priority = 0;
    break;
  case GGML_SCHED_PRIO_NORMAL:
    vPolicy = SCHED_OTHER;
    vP.sched_priority = 0;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    vPolicy = SCHED_FIFO;
    vP.sched_priority = 40;
    break;
  case GGML_SCHED_PRIO_HIGH:
    vPolicy = SCHED_FIFO;
    vP.sched_priority = 80;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    vPolicy = SCHED_FIFO;
    vP.sched_priority = 90;
    break;
  }

  if (pPrio == GGML_SCHED_PRIO_NORMAL) {
    // Keep inherited policy/priority
    return true;
  }

  int32_t vErr = pthread_setschedparam(pthread_self(), vPolicy, &vP);
  if (vErr != 0) {
    fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", pPrio, strerror(vErr), vErr);
    return false;
  }

  return true;
}

#else // unsupported platforms

static bool fGgmlThreadApplyAffinity(const bool *mask) {
  UNUSED(mask);
  return true;
}

static bool fGgmlThreadApplyPriority(int32_t prio) {
  UNUSED(prio);
  return true;
}

#endif

static bool fGgmlThreadCpumaskIsValid(const bool *pMask) {
  for (int vI = 0; vI < GGML_MAX_N_THREADS; vI++) {
    if (pMask[vI]) {
      return true;
    }
  }
  return false;
}

static void fGgmlThreadCpumaskNext(const bool *pGlobalMask, bool *pLocalMask, bool pStrict, int32_t *pIter) {
  if (!pStrict) {
    memcpy(pLocalMask, pGlobalMask, GGML_MAX_N_THREADS);
    return;
  } else {
    memset(pLocalMask, 0, GGML_MAX_N_THREADS);
    int32_t vBaseIdx = *pIter;
    for (int32_t vI = 0; vI < GGML_MAX_N_THREADS; vI++) {
      int32_t vIdx = vBaseIdx + vI;
      if (vIdx >= GGML_MAX_N_THREADS) {
        // Just a cheaper modulo
        vIdx -= GGML_MAX_N_THREADS;
      }
      if (pGlobalMask[vIdx]) {
        pLocalMask[vIdx] = 1;
        *pIter = vIdx + 1;
        return;
      }
    }
  }
}

void ggml_threadpool_free(struct ggml_threadpool *threadpool) {
  if (!threadpool)
    return;

  const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
  struct ggml_compute_state *workers = threadpool->workers;

  ggml_mutex_lock(&threadpool->vMutex);

  threadpool->vStop = true;
  threadpool->vPause = false;

  ggml_cond_broadcast(&threadpool->vCond);
  ggml_mutex_unlock(&threadpool->vMutex);

  for (int j = 1; j < n_threads; j++) {
    int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
    GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
    UNUSED(rc);
  }

  ggml_mutex_destroy(&threadpool->vMutex);
  ggml_cond_destroy(&threadpool->vCond);
#endif // GGML_USE_OPENMP

  const size_t cWorkersSize = sizeof(struct ggml_compute_state) * n_threads;
  ggml_aligned_free(threadpool->workers, cWorkersSize);
  ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool *threadpool) {
  GGML_PRINT_DEBUG("Pausing threadpool\n");
  threadpool->vPause = true;
  ggml_cond_broadcast(&threadpool->vCond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool *threadpool) {
  GGML_PRINT_DEBUG("Resuming threadpool\n");
  threadpool->vPause = false;
  ggml_cond_broadcast(&threadpool->vCond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool *threadpool) {
#ifndef GGML_USE_OPENMP
  ggml_mutex_lock(&threadpool->vMutex);
  if (!threadpool->vPause) {
    ggml_threadpool_pause_locked(threadpool);
  }
  ggml_mutex_unlock(&threadpool->vMutex);
#else
  UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool *threadpool) {
#ifndef GGML_USE_OPENMP
  ggml_mutex_lock(&threadpool->vMutex);
  if (threadpool->vPause) {
    ggml_threadpool_resume_locked(threadpool);
  }
  ggml_mutex_unlock(&threadpool->vMutex);
#else
  UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph *pCgraph, int n_threads,
                                  struct ggml_threadpool *pThreadpool) {

  if (pThreadpool == NULL) {
    // GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
  }
  if (n_threads <= 0) {
    n_threads = pThreadpool ? pThreadpool->n_threads : GGML_DEFAULT_N_THREADS;
  }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  // Emscripten without pthreads support can only use a single thread
  n_threads = 1;
#endif

  size_t vWorkSize = 0;

  struct ggml_cplan vCplan;
  memset(&vCplan, 0, sizeof(struct ggml_cplan));

  int vMaxTasks = 1;

  // thread scheduling for the different operations + work buffer size estimation
  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pCgraph->nodes[vI];

    const int cNTasks = fGgmlGetNTasks(vNode, n_threads);

    vMaxTasks = MAX(vMaxTasks, cNTasks);

    size_t vCur = 0;

    if (!ggml_cpu_extra_work_size(n_threads, vNode, &vCur)) {
      switch (vNode->op) {
      case GGML_OP_CPY:
      case GGML_OP_DUP: {
        if (ggml_is_quantized(vNode->type) ||
            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
            (vNode->src[0]->type == GGML_TYPE_F16 && vNode->src[1] && vNode->src[1]->type == GGML_TYPE_BF16) ||
            (vNode->src[0]->type == GGML_TYPE_BF16 && vNode->src[1] && vNode->src[1]->type == GGML_TYPE_F16) ||
            // conversion between F32 and I32
            (vNode->src[0]->type == GGML_TYPE_F32 && vNode->src[1] && vNode->src[1]->type == GGML_TYPE_I32) ||
            (vNode->src[0]->type == GGML_TYPE_I32 && vNode->src[1] && vNode->src[1]->type == GGML_TYPE_F32)) {
          vCur = ggml_type_size(GGML_TYPE_F32) * vNode->ne[0] * cNTasks;
        }
      } break;
      case GGML_OP_ADD:
      case GGML_OP_ADD_ID:
      case GGML_OP_ADD1: {
        if (ggml_is_quantized(vNode->src[0]->type)) {
          vCur = ggml_type_size(GGML_TYPE_F32) * vNode->src[0]->ne[0] * cNTasks;
        }
      } break;
      case GGML_OP_ACC: {
        if (ggml_is_quantized(vNode->src[0]->type)) {
          vCur = ggml_type_size(GGML_TYPE_F32) * vNode->src[1]->ne[0] * cNTasks;
        }
      } break;
      case GGML_OP_COUNT_EQUAL: {
        vCur = ggml_type_size(vNode->type) * cNTasks;
      } break;
      case GGML_OP_MUL_MAT: {
        const enum ggml_type cVecDotType = type_traits_cpu[vNode->src[0]->type].vec_dot_type;

        if (vNode->src[1]->type != cVecDotType) {
          vCur = ggml_row_size(cVecDotType, ggml_nelements(vNode->src[1]));
        }
      } break;
      case GGML_OP_MUL_MAT_ID: {
        vCur = 0;
        const struct ggml_tensor *vSrc0 = vNode->src[0];
        const struct ggml_tensor *vSrc1 = vNode->src[1];
        const struct ggml_tensor *vIds = vNode->src[2];
        const enum ggml_type cVecDotType = type_traits_cpu[vSrc0->type].vec_dot_type;
        const int cNAs = vSrc0->ne[2];
        // src1
        if (vSrc1->type != cVecDotType) {
          vCur += ggml_row_size(cVecDotType, ggml_nelements(vSrc1)) + sizeof(int64_t);
        }
        // matrix_row_counts
        vCur += cNAs * sizeof(int64_t) + sizeof(int64_t);
        // matrix_rows
        vCur += cNAs * vIds->ne[0] * vIds->ne[1] * sizeof(struct mmid_row_mapping) + sizeof(int64_t);
        // atomic_current_chunk
        vCur += CACHE_LINE_SIZE * cNAs + CACHE_LINE_SIZE;
      } break;
      case GGML_OP_OUT_PROD: {
        if (ggml_is_quantized(vNode->src[0]->type) || vNode->src[0]->type == GGML_TYPE_F16) {
          vCur = ggml_type_size(GGML_TYPE_F32) * vNode->src[0]->ne[0] * cNTasks;
        }
      } break;
      case GGML_OP_SET_ROWS: {
        if (vNode->src[0]->type == GGML_TYPE_F16 && vNode->type != GGML_TYPE_F16) {
          vCur = ggml_type_size(GGML_TYPE_F32) * vNode->src[0]->ne[0] * cNTasks;
        }
      } break;
      case GGML_OP_SOFT_MAX:
      case GGML_OP_ROPE:
      case GGML_OP_ROPE_BACK: {
        vCur = ggml_type_size(GGML_TYPE_F32) * vNode->ne[0] * cNTasks;
      } break;
      case GGML_OP_CONV_TRANSPOSE_1D: {
        GGML_ASSERT(vNode->src[0]->ne[3] == 1);
        GGML_ASSERT(vNode->src[1]->ne[2] == 1);
        GGML_ASSERT(vNode->src[1]->ne[3] == 1);

        const int64_t cNe00 = vNode->src[0]->ne[0]; // K
        const int64_t cNe01 = vNode->src[0]->ne[1]; // Cout
        const int64_t cNe02 = vNode->src[0]->ne[2]; // Cin
        const int64_t cNe10 = vNode->src[1]->ne[0]; // L
        const int64_t cNe11 = vNode->src[1]->ne[1]; // Cin

        if ((vNode->src[0]->type == GGML_TYPE_F16 || vNode->src[0]->type == GGML_TYPE_BF16) &&
            vNode->src[1]->type == GGML_TYPE_F32) {
          vCur += sizeof(ggml_fp16_t) * cNe00 * cNe01 * cNe02;
          vCur += sizeof(ggml_fp16_t) * cNe10 * cNe11;
        } else if (vNode->src[0]->type == GGML_TYPE_F32 && vNode->src[1]->type == GGML_TYPE_F32) {
          vCur += sizeof(float) * cNe00 * cNe01 * cNe02;
          vCur += sizeof(float) * cNe10 * cNe11;
        } else {
          GGML_ABORT("fatal error");
        }
      } break;
      case GGML_OP_CONV_2D:
      case GGML_OP_CONV_3D: {
        vCur = GGML_IM2COL_WORK_SIZE;
      } break;
      case GGML_OP_CONV_TRANSPOSE_2D: {
        const int64_t cNe00 = vNode->src[0]->ne[0]; // W
        const int64_t cNe01 = vNode->src[0]->ne[1]; // H
        const int64_t cNe02 = vNode->src[0]->ne[2]; // Channels Out
        const int64_t cNe03 = vNode->src[0]->ne[3]; // Channels In

        const int64_t cNe10 = vNode->src[1]->ne[0]; // W
        const int64_t cNe11 = vNode->src[1]->ne[1]; // H
        const int64_t cNe12 = vNode->src[1]->ne[2]; // Channels In

        GGML_ASSERT(vNode->src[0]->type == GGML_TYPE_F16 || vNode->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(vNode->src[1]->type == GGML_TYPE_F32);

        vCur += ggml_type_size(vNode->src[0]->type) * cNe00 * cNe01 * cNe02 * cNe03;
        vCur += ggml_type_size(vNode->src[0]->type) * cNe10 * cNe11 * cNe12;

      } break;
      case GGML_OP_TOP_K: {
        vCur += sizeof(int32_t) * vNode->src[0]->ne[0] * cNTasks;
      } break;
      case GGML_OP_FLASH_ATTN_EXT: {
        const int64_t cNeq2 = vNode->src[0]->ne[2]; // number of query heads
        const int64_t cDk = vNode->src[1]->ne[0];
        const int64_t cDv = vNode->src[2]->ne[0];

        // Tiled flash attention scratch (tile sizes defined in common.h)
        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
        size_t vPrefill = sizeof(float) *
                          (GGML_FA_TILE_Q * cDk + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * cDv +
                           GGML_FA_TILE_KV * cDv + GGML_FA_TILE_KV * cDk) *
                          cNTasks;

        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
        size_t vNChunks = cNTasks;
        size_t vDecode = sizeof(float) * (cNeq2 * vNChunks * (2 + cDv) + cNTasks * (cDk + 2 * cDv));

        vCur += MAX(vPrefill, vDecode);
      } break;
      case GGML_OP_FLASH_ATTN_BACK: {
        const int64_t cD = vNode->src[0]->ne[0];
        const int64_t cNe11 = ggml_up(vNode->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
        const int64_t cMxDn = MAX(cD, cNe11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
        if (vNode->src[1]->type == GGML_TYPE_F32) {
          vCur = sizeof(float) * cMxDn * cNTasks;  // TODO: this can become (n_tasks-1)
          vCur += sizeof(float) * cMxDn * cNTasks; // this is overestimated by x2
        } else if (vNode->src[1]->type == GGML_TYPE_F16) {
          vCur = sizeof(float) * cMxDn * cNTasks;  // TODO: this can become (n_tasks-1)
          vCur += sizeof(float) * cMxDn * cNTasks; // this is overestimated by x2
        } else if (vNode->src[1]->type == GGML_TYPE_BF16) {
          vCur = sizeof(float) * cMxDn * cNTasks;  // TODO: this can become (n_tasks-1)
          vCur += sizeof(float) * cMxDn * cNTasks; // this is overestimated by x2
        }
      } break;

      case GGML_OP_CROSS_ENTROPY_LOSS: {
        vCur = ggml_type_size(vNode->type) * (cNTasks + vNode->src[0]->ne[0] * cNTasks);
      } break;
      case GGML_OP_GATED_DELTA_NET: {
        const int64_t cSV = vNode->src[2]->ne[0];
        const int64_t cK = ggml_get_op_params_i32(vNode, 0);
        const int64_t cPerThread = cSV + (cK > 1 ? cSV * cSV : 0);
        vCur = cPerThread * sizeof(float) * cNTasks;
      } break;
      case GGML_OP_COUNT: {
        GGML_ABORT("fatal error");
      }
      case GGML_OP_LIGHTNING_INDEXER: {
        // temp buffer for dequantizing lightning indexer keys
        const int64_t cNe10 = vNode->src[1]->ne[0];
        vCur += sizeof(float) * cNe10 * cNTasks;
      } break;
      default:
        break;
      }
    }

    vWorkSize = MAX(vWorkSize, vCur);
  }

  if (vWorkSize > 0) {
    vWorkSize += CACHE_LINE_SIZE * (n_threads);
  }

  vCplan.threadpool = pThreadpool;
  vCplan.n_threads = MIN(vMaxTasks, n_threads);
  vCplan.work_size = vWorkSize;
  vCplan.work_data = NULL;

  return vCplan;
}

// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false; // initialized once in ggml_cpu_init(), read-only afterwards

static int fGgmlCpuTryFuseOps(const struct ggml_cgraph *pCgraph, const int pNodeN,
                              const struct ggml_compute_params *pParams, const struct ggml_cplan *pCplan) {

  if (ggml_cpu_disable_fusion || pCplan->use_ref) {
    return 0;
  }

  struct ggml_tensor *vNode = pCgraph->nodes[pNodeN];

  if (vNode->op == GGML_OP_RMS_NORM) {
    // RMS_NORM + MUL fusion
    const enum ggml_op aFuseOps[] = {GGML_OP_RMS_NORM, GGML_OP_MUL};
    if (ggml_can_fuse(pCgraph, pNodeN, aFuseOps, 2)) {
      struct ggml_tensor *vMulNode = pCgraph->nodes[pNodeN + 1];
      const struct ggml_tensor *vMulW = (vMulNode->src[0] == vNode) ? vMulNode->src[1] : vMulNode->src[0];
      if (vNode->src[0]->type == GGML_TYPE_F32 && vMulNode->type == GGML_TYPE_F32 && vMulW->type == GGML_TYPE_F32 &&
          vMulW->ne[0] == vNode->ne[0] && vMulW->nb[0] == sizeof(float)) {

        ggml_compute_forward_rms_norm_mul_fused(pParams, vNode, vMulNode);
        return 1;
      }
    }
  }

  return 0;
}

static thread_ret_t fGgmlGraphComputeThread(void *pData) {
  struct ggml_compute_state *vState = (struct ggml_compute_state *)pData;
  struct ggml_threadpool *vTp = vState->threadpool;

  const struct ggml_cgraph *vCgraph = vTp->vCgraph;
  const struct ggml_cplan *cplan = vTp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
  ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(vState->ith);
#else
  fSetNumaThreadAffinity(vState->ith);
#endif

  struct ggml_compute_params vParams = {
      /*.ith        =*/vState->ith,
      /*.nth        =*/atomic_load_explicit(&vTp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
      /*.wsize      =*/cplan->work_size,
      /*.wdata      =*/cplan->work_data,
      /*.threadpool =*/vTp,
      /*.use_ref    =*/cplan->use_ref,
  };

#ifdef GGML_USE_OPENMP
  GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", vState->ith, (const void *)cplan);
#else
  GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", vState->ith, (const void *)cplan,
                   vState->last_graph);
#endif

  for (int vNodeN = 0; vNodeN < vCgraph->n_nodes && atomic_load_explicit(&vTp->vAbort, memory_order_relaxed) != vNodeN;
       vNodeN++) {
    struct ggml_tensor *vNode = vCgraph->nodes[vNodeN];

    if (ggml_op_is_empty(vNode->op)) {
      // skip NOPs
      continue;
    }

    if ((vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
      continue;
    }

    // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
    // Try fused ops, fall back to normal compute
    const int cNFused = fGgmlCpuTryFuseOps(vCgraph, vNodeN, &vParams, cplan);
    if (cNFused > 0) {
      vNodeN += cNFused;
    } else {
      fGgmlComputeForward(&vParams, vNode);
    }

    if (vState->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
      atomic_store_explicit(&vTp->vAbort, vNodeN + 1, memory_order_relaxed);
      vTp->vEc = GGML_STATUS_ABORTED;
    }

    if (vNodeN + 1 < vCgraph->n_nodes) {
      ggml_barrier(vState->threadpool);
    }
  }

#ifdef GGML_USE_OPENMP
  GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", vState->ith, (const void *)cplan);
#else
  GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", vState->ith, (const void *)cplan,
                   vState->last_graph);
#endif

  ggml_barrier(vState->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
  ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(vState->ith);
#endif

  return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state *state) {
  struct ggml_threadpool *threadpool = state->threadpool;

  if (state->pending || threadpool->vStop || threadpool->vPause) {
    return true;
  }

  // check for new graph/work
  int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
  int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
  if (n_graph != state->last_graph) {
    state->pending = (state->ith < n_threads);
    state->last_graph = n_graph;
    return true;
  }

  return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state *state) {
// TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
#ifdef GGML_TSAN_ENABLED
  atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
  UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state *state) {
  struct ggml_threadpool *threadpool = state->threadpool;

  // This seems to make 0 ... 100 a decent range for polling level across modern processors.
  // Perhaps, we can adjust it dynamically based on load and things.
  const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

  for (uint64_t i = 0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
    // No new work. Keep polling.
    fGgmlThreadCpuRelax();
  }

  return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state *state) {
  struct ggml_threadpool *threadpool = state->threadpool;

  if (ggml_graph_compute_poll_for_work(state)) {
    ggml_graph_compute_thread_sync(state);
    return state->pending;
  }

  ggml_mutex_lock_shared(&threadpool->vMutex);
  while (!ggml_graph_compute_thread_ready(state)) {
    // No new work. Wait for the signal.
    GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
    ggml_cond_wait(&threadpool->vCond, &threadpool->vMutex);
  }
  ggml_mutex_unlock_shared(&threadpool->vMutex);

  return state->pending;
}

static thread_ret_t fGgmlGraphComputeSecondaryThread(void *data) {
  struct ggml_compute_state *state = (struct ggml_compute_state *)data;
  struct ggml_threadpool *threadpool = state->threadpool;

  fGgmlThreadApplyPriority(threadpool->prio);
  if (fGgmlThreadCpumaskIsValid(state->cpumask)) {
    fGgmlThreadApplyAffinity(state->cpumask);
  }

  while (true) {
    // Check if we need to sleep
    while (threadpool->vPause) {
      GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
      ggml_mutex_lock_shared(&threadpool->vMutex);
      if (threadpool->vPause) {
        ggml_cond_wait(&threadpool->vCond, &threadpool->vMutex);
      }
      GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
      ggml_mutex_unlock_shared(&threadpool->vMutex);
    }

    // This needs to be checked for after the cond_wait
    if (threadpool->vStop)
      break;

    // Check if there is new work
    // The main thread is the only one that can dispatch new work

    ggml_graph_compute_check_for_work(state);
    if (state->pending) {
      state->pending = false;
      fGgmlGraphComputeThread(state);
    }
  }

  return (thread_ret_t)0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool *threadpool, int n_threads) {
  // Always take the mutex here because the worker threads are doing hybrid poll/wait

  ggml_mutex_lock(&threadpool->vMutex);

  // Update the number of active threads and the graph count
  int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
  n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

  GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

  // Indicate the graph is ready to be processed
  // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
  atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

  if (threadpool->vPause) {
    // Update main thread prio and affinity to match the threadpool settings
    fGgmlThreadApplyPriority(threadpool->prio);
    if (fGgmlThreadCpumaskIsValid(threadpool->workers[0].cpumask)) {
      fGgmlThreadApplyAffinity(threadpool->workers[0].cpumask);
    }

    // resume does cond broadcast
    ggml_threadpool_resume_locked(threadpool);
  } else {
    ggml_cond_broadcast(&threadpool->vCond);
  }

  ggml_mutex_unlock(&threadpool->vMutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool *fGgmlThreadpoolNewImpl(struct ggml_threadpool_params *pTpp, struct ggml_cgraph *pCgraph,
                                                      struct ggml_cplan *pCplan) {

  struct ggml_threadpool *threadpool = ggml_aligned_malloc(sizeof(struct ggml_threadpool));
  {
    threadpool->vCgraph = pCgraph;
    threadpool->cplan = pCplan;
    threadpool->n_graph = 0;
    threadpool->n_barrier = 0;
    threadpool->vNBarrierPassed = 0;
    threadpool->vCurrentChunk = 0;
    threadpool->vStop = false;
    threadpool->vPause = pTpp->paused;
    threadpool->vAbort = -1;
    threadpool->workers = NULL;
    threadpool->n_threads = pTpp->n_threads;
    threadpool->poll = pTpp->poll;
    threadpool->prio = pTpp->prio;
    threadpool->vEc = GGML_STATUS_SUCCESS;
  }

  // Allocate and init workers state
  const size_t cWorkersSize = sizeof(struct ggml_compute_state) * pTpp->n_threads;
  struct ggml_compute_state *workers = ggml_aligned_malloc(cWorkersSize);

  memset(workers, 0, cWorkersSize);
  for (int vJ = 0; vJ < pTpp->n_threads; vJ++) {
    workers[vJ].threadpool = threadpool;
    workers[vJ].ith = vJ;
  }

  threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
  int32_t vCpumaskIter = 0;

  // Compute CPU masks for each thread
  for (int vJ = 0; vJ < pTpp->n_threads; vJ++) {
    fGgmlThreadCpumaskNext(pTpp->cpumask, workers[vJ].cpumask, pTpp->strict_cpu, &vCpumaskIter);
  }
#else  // GGML_USE_OPENMP
  ggml_mutex_init(&threadpool->vMutex);
  ggml_cond_init(&threadpool->vCond);

  // Spin the threads for all workers, and update CPU placements.
  // Place the main thread last (towards the higher numbered CPU cores).

  int32_t vCpumaskIter = 0;

  for (int vJ = 1; vJ < pTpp->n_threads; vJ++) {
    fGgmlThreadCpumaskNext(pTpp->cpumask, workers[vJ].cpumask, pTpp->strict_cpu, &vCpumaskIter);

    int32_t rc = ggml_thread_create(&workers[vJ].thrd, NULL, fGgmlGraphComputeSecondaryThread, &workers[vJ]);
    GGML_ASSERT(rc == 0);
  }

  fGgmlThreadCpumaskNext(pTpp->cpumask, workers[0].cpumask, pTpp->strict_cpu, &vCpumaskIter);

  if (!threadpool->vPause) {
    // Update main thread prio and affinity at the start, otherwise we'll do it in resume
    fGgmlThreadApplyPriority(threadpool->prio);
    if (fGgmlThreadCpumaskIsValid(threadpool->workers[0].cpumask)) {
      fGgmlThreadApplyAffinity(threadpool->workers[0].cpumask);
    }
  }
#endif // GGML_USE_OPENMP

  return threadpool;
}

struct ggml_threadpool *ggml_threadpool_new(struct ggml_threadpool_params *pTpp) {
  return fGgmlThreadpoolNewImpl(pTpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph *pCgraph, struct ggml_cplan *pCplan) {
  ggml_cpu_init();

  GGML_ASSERT(pCplan);
  GGML_ASSERT(pCplan->n_threads > 0);
  GGML_ASSERT(pCplan->work_size == 0 || pCplan->work_data != NULL);

  int n_threads = pCplan->n_threads;
  struct ggml_threadpool *threadpool = pCplan->threadpool;

  bool vDisposableThreadpool = false;

  if (threadpool == NULL) {
    // GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    vDisposableThreadpool = true;

    struct ggml_threadpool_params vTtp = ggml_threadpool_params_default(n_threads);
    threadpool = fGgmlThreadpoolNewImpl(&vTtp, pCgraph, pCplan);
  } else {
    // Reset some of the parameters that need resetting
    // No worker threads should be accessing the parameters below at this stage
    threadpool->vCgraph = pCgraph;
    threadpool->cplan = pCplan;
    threadpool->vCurrentChunk = 0;
    threadpool->vAbort = -1;
    threadpool->vEc = GGML_STATUS_SUCCESS;
  }

#ifdef GGML_USE_OPENMP
  if (n_threads > 1) {
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp single
      {
        // update the number of threads from the actual number of threads that we got from OpenMP
        n_threads = omp_get_num_threads();
        atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
      }

      // Apply thread CPU mask and priority
      int ith = omp_get_thread_num();

      fGgmlThreadApplyPriority(threadpool->prio);
      if (fGgmlThreadCpumaskIsValid(threadpool->workers[ith].cpumask)) {
        fGgmlThreadApplyAffinity(threadpool->workers[ith].cpumask);
      }
      fGgmlGraphComputeThread(&threadpool->workers[ith]);
    }
  } else {
    atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
    fGgmlGraphComputeThread(&threadpool->workers[0]);
  }
#else
  if (n_threads > threadpool->n_threads) {
    GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
    n_threads = threadpool->n_threads;
  }

  // Kick all threads to start the new graph
  ggml_graph_compute_kickoff(threadpool, n_threads);

  // This is a work thread too
  fGgmlGraphComputeThread(&threadpool->workers[0]);
#endif

  // don't leave affinity set on the main thread
  fClearNumaThreadAffinity();

  enum ggml_status vRet = threadpool->vEc;

  if (vDisposableThreadpool) {
    ggml_threadpool_free(threadpool);
  }

  return vRet;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context *pCtx, struct ggml_cgraph *pCgraph, int pNThreads) {
  struct ggml_cplan vCplan = ggml_graph_plan(pCgraph, pNThreads, NULL);

  vCplan.work_data = (uint8_t *)ggml_new_buffer(pCtx, vCplan.work_size);

  return ggml_graph_compute(pCgraph, &vCplan);
}

void ggml_cpu_fp32_to_fp32(const float *pX, float *pY, int64_t pN) { memcpy(pY, pX, pN * sizeof(float)); }

void ggml_cpu_fp32_to_fp16(const float *pX, ggml_fp16_t *pY, int64_t pN) {
  int64_t vI = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
  for (; vI + 15 < pN; vI += 16) {
    __m512 vXVec = _mm512_loadu_ps(pX + vI);
    __m256i vYVec = _mm512_cvtps_ph(vXVec, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256((__m256i *)(pY + vI), vYVec);
  }
#endif
  for (; vI + 7 < pN; vI += 8) {
    __m256 vXVec = _mm256_loadu_ps(pX + vI);
    __m128i vYVec = _mm256_cvtps_ph(vXVec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i *)(pY + vI), vYVec);
  }
  for (; vI + 3 < pN; vI += 4) {
    __m128 vXVec = _mm_loadu_ps(pX + vI);
    __m128i vYVec = _mm_cvtps_ph(vXVec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64((__m128i *)(pY + vI), vYVec);
  }
#elif defined(__riscv_zvfh)
  for (int vl; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e32m2(pN - vI);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(&pX[vI], vl);
    vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
    __riscv_vse16_v_f16m1((_Float16 *)&pY[vI], vy, vl);
  }
#endif
  for (; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(pX[vI]);
  }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *pX, float *pY, int64_t pN) {
  int64_t vI = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
  for (; vI + 15 < pN; vI += 16) {
    __m256i vXVec = _mm256_loadu_si256((const __m256i *)(pX + vI));
    __m512 vYVec = _mm512_cvtph_ps(vXVec);
    _mm512_storeu_ps(pY + vI, vYVec);
  }
#endif
  for (; vI + 7 < pN; vI += 8) {
    __m128i vXVec = _mm_loadu_si128((const __m128i *)(pX + vI));
    __m256 vYVec = _mm256_cvtph_ps(vXVec);
    _mm256_storeu_ps(pY + vI, vYVec);
  }
  for (; vI + 3 < pN; vI += 4) {
    __m128i vXVec = _mm_loadl_epi64((const __m128i *)(pX + vI));
    __m128 vYVec = _mm_cvtph_ps(vXVec);
    _mm_storeu_ps(pY + vI, vYVec);
  }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
  // calculate step size
  const int epr = __riscv_vsetvlmax_e16m2();
  const int step = epr * 2;
  const int np = (pN & ~(step - 1));

  // unroll by 2
  for (; vI < np; vI += step) {
    vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16 *)pX + vI, epr);
    vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
    __riscv_vse32_v_f32m4(pY + vI, ay0, epr);

    vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16 *)pX + vI + epr, epr);
    vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
    __riscv_vse32_v_f32m4(pY + vI + epr, ay1, epr);
  }

  // leftovers
  int vl;
  for (vI = np; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m2(pN - vI);
    vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16 *)pX + vI, vl);
    vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
    __riscv_vse32_v_f32m4(pY + vI, ay0, vl);
  }

#endif

  for (; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP16_TO_FP32(pX[vI]);
  }
}

void ggml_cpu_fp32_to_bf16(const float *pX, ggml_bf16_t *pY, int64_t pN) {
  int64_t vI = 0;
  for (; vI < pN; ++vI) {
    pY[vI] = GGML_FP32_TO_BF16(pX[vI]);
  }
}

void ggml_cpu_fp32_to_i32(const float *pX, int32_t *pY, int64_t pN) {
  int64_t vI = 0;
  for (; vI < pN; ++vI) {
    pY[vI] = pX[vI];
  }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *pX, float *pY, int64_t pN) {
  int64_t vI = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
  for (; vI + 15 < pN; vI += 16) {
    _mm512_storeu_ps(pY + vI, _mm512_castsi512_ps(_mm512_slli_epi32(
                                  _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(pX + vI))), 16)));
  }
#endif
  for (; vI + 7 < pN; vI += 8) {
    _mm256_storeu_ps(pY + vI, _mm256_castsi256_ps(_mm256_slli_epi32(
                                  _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(pX + vI))), 16)));
  }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
  // calculate step size
  const int epr = __riscv_vsetvlmax_e16m2();
  const int step = epr * 2;
  const int np = (pN & ~(step - 1));

  // unroll by 2
  for (; vI < np; vI += step) {
    vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)pX + vI, epr);
    vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
    __riscv_vse32_v_f32m4(pY + vI, ay0, epr);

    vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16 *)pX + vI + epr, epr);
    vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
    __riscv_vse32_v_f32m4(pY + vI + epr, ay1, epr);
  }

  // leftovers
  int vl;
  for (vI = np; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m2(pN - vI);
    vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)pX + vI, vl);
    vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
    __riscv_vse32_v_f32m4(pY + vI, ay0, vl);
  }
#endif
  for (; vI < pN; vI++) {
    pY[vI] = GGML_BF16_TO_FP32(pX[vI]);
  }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
  return ggml_riscv_arch_features.rvv_vlen;
#else
  return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_tinyblas(void) {
#if defined(GGML_USE_TINYBLAS)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
  return ggml_arm_arch_features.sve_cnt;
#else
  return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
  return 1;
#else
  return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
  return 1;
#else
  return 0;
#endif
}

void ggml_cpu_init(void) {
  // needed to initialize ggml_time
  {
    struct ggml_init_params vParams = {0, NULL, false};
    struct ggml_context *vCtx = ggml_init(vParams);
    ggml_free(vCtx);
  }

  ggml_critical_section_start();

  static bool vIsFirstCall = true;

  if (vIsFirstCall) {
    // initialize GELU, Quick GELU, SILU and EXP F32 tables
    {
      const uint64_t cTStart = ggml_time_us();
      UNUSED(cTStart);

      for (int vI = 0; vI < (1 << 16); ++vI) {
        union {
          uint16_t vU16;
          ggml_fp16_t vFp16;
        } vU = {vI};
        float vF = GGML_COMPUTE_FP16_TO_FP32(vU.vFp16);
        ggml_table_f32_f16[vI] = vF;
        ggml_table_gelu_f16[vI] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(vF));
        ggml_table_gelu_quick_f16[vI] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(vF));
      }

      // initialize E8M0 half table (256 entries)
      for (int vI = 0; vI < (1 << 8); ++vI) {
        ggml_table_f32_e8m0_half[vI] = GGML_E8M0_TO_FP32_HALF(vI);
      }

      // initialize UE4M3 table (256 entries)
      for (int vI = 0; vI < (1 << 8); ++vI) {
        ggml_table_f32_ue4m3[vI] = ggml_ue4m3_to_fp32(vI);
      }

      const uint64_t cTEnd = ggml_time_us();
      UNUSED(cTEnd);

      GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__,
                       (cTEnd - cTStart) / 1000.0);

#ifdef GGML_USE_OPENMP
      // if (!getenv("OMP_WAIT_POLICY")) {
      //     // set the wait policy to active, so that OpenMP threads don't sleep
      //     setenv("OMP_WAIT_POLICY", "active", 0)
      // }

      if (!getenv("KMP_BLOCKTIME")) {
        // set the time to wait before sleeping a thread
        // this is less aggressive than setting the wait policy to active, but should achieve similar results in most
        // cases
#ifdef _WIN32
        _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
        setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
      }
#endif
    }

#if defined(__ARM_ARCH)
    ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
    fGgmlInitRiscvArchFeatures();
#endif

    {
      const char *vEnv = getenv("GGML_CPU_DISABLE_FUSION");
      ggml_cpu_disable_fusion = (vEnv != NULL && atoi(vEnv) == 1);
    }

    vIsFirstCall = false;
  }

  ggml_critical_section_end();
}
