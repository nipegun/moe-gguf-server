#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>

#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX                                                                                                    \
  11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK 12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_PASCAL 600
#define GGML_CUDA_CC_DP4A 610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA 700
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_AMPERE 800
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_HOPPER 900
// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define GGML_CUDA_CC_BLACKWELL 1200
#define GGML_CUDA_CC_DGX_SPARK 1210
#define GGML_CUDA_CC_RUBIN 1300
#define GGML_CUDA_CC_OFFSET_AMD 0x1000000
#define GGML_CUDA_CC_IS_NVIDIA(cc) (cc < GGML_CUDA_CC_OFFSET_AMD)

// AMD
// GCN/CDNA, wave size is 64
#define GGML_CUDA_CC_GCN4 (GGML_CUDA_CC_OFFSET_AMD + 0x803)   // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA (GGML_CUDA_CC_OFFSET_AMD + 0x900)   // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20 (GGML_CUDA_CC_OFFSET_AMD + 0x906) // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA1 (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2 (GGML_CUDA_CC_OFFSET_AMD + 0x90a)  // MI210 (gfx90a), minimum acc register renaming
#define GGML_CUDA_CC_CDNA3 (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300
#define GGML_CUDA_CC_CDNA4 (GGML_CUDA_CC_OFFSET_AMD + 0x950)  // MI350X/MI355X

// RDNA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_CUDA_CC_RDNA1 (GGML_CUDA_CC_OFFSET_AMD + 0x1010)   // RX 5000
#define GGML_CUDA_CC_RDNA2 (GGML_CUDA_CC_OFFSET_AMD + 0x1030)   // RX 6000, minimum for dp4a
#define GGML_CUDA_CC_RDNA3 (GGML_CUDA_CC_OFFSET_AMD + 0x1100)   // RX 7000, minimum for WMMA
#define GGML_CUDA_CC_RDNA3_5 (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_RDNA4 (GGML_CUDA_CC_OFFSET_AMD + 0x1200)   // RX 9000

#define GGML_CUDA_CC_IS_AMD(cc) (cc >= GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_RDNA(cc) (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc) (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc) (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc) (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc) (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_GCN(cc) (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA1)
#define GGML_CUDA_CC_IS_CDNA(cc) (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA1(cc) (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_CDNA2)
#define GGML_CUDA_CC_IS_CDNA2(cc) (cc >= GGML_CUDA_CC_CDNA2 && cc < GGML_CUDA_CC_CDNA3)
#define GGML_CUDA_CC_IS_CDNA3(cc) (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_CDNA4)
#define GGML_CUDA_CC_IS_CDNA4(cc) (cc >= GGML_CUDA_CC_CDNA4 && cc < GGML_CUDA_CC_RDNA1)

#if !defined(GGML_USE_HIP) && CUDART_VERSION >= 11070
#define GGML_CUDA_USE_CUB
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION >= 11070

// PDL host-side support (cudaLaunchKernelEx) requires CUDART >= 11.8.
// However, this has been bugged in CTK < 12.3 for MSVC builds.
// __CUDA_ARCH__  is undefined in host passes; GPU arch check happens in device-side code.
#if !defined(GGML_USE_HIP) &&                                                                                          \
    (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))
#define GGML_CUDA_USE_PDL
#endif // !defined(GGML_USE_HIP) && (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) &&
       // CUDART_VERSION >= 11080))

static __device__ __forceinline__ void ggml_cuda_syncwarp() {
#ifndef GGML_USE_HIP
  __syncwarp();
#endif // GGML_USE_HIP
}

static __device__ __forceinline__ void ggml_cuda_pdl_sync() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
  cudaGridDependencySynchronize();
#endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

static __device__ __forceinline__ void ggml_cuda_pdl_lc() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
  cudaTriggerProgrammaticLaunchCompletion();
#endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) { return false; }

template <class... Archs> constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
  return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) { return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__); }

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
  if (cur == 0) {
    return -1;
  }
  return cur;
}

template <class... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
  if (first <= arch && first > cur) {
    return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
  } else {
    return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
  }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
  return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int pArch) { return pArch; }
#endif // __CUDA_ARCH_LIST__

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING                                                                                             \
  512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char *pStmt, const char *pFunc, const char *pFile, int pLine, const char *pMsg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                                                         \
  do {                                                                                                                 \
    auto err_ = (err);                                                                                                 \
    if (err_ != (success)) {                                                                                           \
      ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));                                             \
    }                                                                                                                  \
  } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000
static const char *cublas_get_error_str(const cublasStatus_t pErr) { return cublasGetStatusString(pErr); }
#else
static const char *cublas_get_error_str(const cublasStatus_t err) {
  switch (err) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    return "unknown error";
  }
}
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#ifdef GGML_USE_NCCL
#define NCCL_CHECK(err) CUDA_CHECK_GEN(err, ncclSuccess, ncclGetErrorString)
#endif // GGML_USE_NCCL

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
static const char *cu_get_error_str(CUresult pErr) {
  const char *vErrStr;
  cuGetErrorString(pErr, &vErrStr);
  return vErrStr;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if !defined(GGML_USE_HIP)
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                                   \
  do {                                                                                                                 \
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};                                           \
    const int id = ggml_cuda_get_device();                                                                             \
    if (!shared_memory_limit_raised[id]) {                                                                             \
      CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes));                   \
      shared_memory_limit_raised[id] = true;                                                                           \
    }                                                                                                                  \
  } while (0)
#else
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                                   \
  do {                                                                                                                 \
    GGML_UNUSED(nbytes);                                                                                               \
  } while (0)
#endif // !defined(GGML_USE_HIP)

#if CUDART_VERSION >= 11010
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11010

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
#define GGML_USE_VMM
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))

#if defined(GGML_USE_HIP) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // defined(GGML_USE_HIP) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#define BLACKWELL_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_CUDA_NO_FA)
#define FLASH_ATTN_AVAILABLE
#endif // !defined(GGML_CUDA_NO_FA)

static bool fp16_available(const int pCc) { return ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_PASCAL; }

static bool fast_fp16_available(const int pCc) {
  return GGML_CUDA_CC_IS_AMD(pCc) ||
         (GGML_CUDA_CC_IS_NVIDIA(pCc) && fp16_available(pCc) && ggml_cuda_highest_compiled_arch(pCc) != 610);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_fp16_hardware_available(const int pCc) {
  return (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc >= GGML_CUDA_CC_PASCAL && pCc != 610) || GGML_CUDA_CC_IS_AMD(pCc);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int pCc) {
  return (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc >= GGML_CUDA_CC_VOLTA) || GGML_CUDA_CC_IS_CDNA(pCc) ||
         GGML_CUDA_CC_IS_RDNA3(pCc) || GGML_CUDA_CC_IS_RDNA4(pCc);
}

static bool bf16_mma_hardware_available(const int pCc) {
  return (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc >= GGML_CUDA_CC_AMPERE) || GGML_CUDA_CC_IS_CDNA(pCc) ||
         pCc >= GGML_CUDA_CC_RDNA3;
}

static bool fp32_mma_hardware_available(const int pCc) { return GGML_CUDA_CC_IS_CDNA(pCc); }

static bool amd_mfma_available(const int pCc) {
#if !defined(GGML_HIP_NO_MMQ_MFMA)
  return GGML_CUDA_CC_IS_CDNA(pCc);
#else
  return false;
#endif //! defined(GGML_HIP_NO_MMQ_MFMA)
}

static bool amd_wmma_available(const int pCc) { return (GGML_CUDA_CC_IS_RDNA4(pCc) || GGML_CUDA_CC_IS_RDNA3(pCc)); }

static bool volta_mma_available(const int pCc) {
  return GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) == GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int pCc) {
  return GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_TURING;
}

static bool ampere_mma_available(const int pCc) {
  return GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_AMPERE;
}

static bool cp_async_available(const int pCc) {
  return GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_AMPERE;
}

static bool blackwell_mma_available(const int pCc) {
  return GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_BLACKWELL &&
         ggml_cuda_highest_compiled_arch(pCc) < GGML_CUDA_CC_RUBIN;
}

// Checks whether the tensor's base data pointer and higher-dimensional strides are byte-aligned to `alignment` bytes.
static bool ggml_cuda_is_aligned(const ggml_tensor *pTensor, const size_t pAlignment) {
  GGML_ASSERT(pTensor != nullptr);
  return (reinterpret_cast<uintptr_t>(pTensor->data) % pAlignment) == 0 && pTensor->nb[1] % pAlignment == 0 &&
         pTensor->nb[2] % pAlignment == 0 && pTensor->nb[3] % pAlignment == 0;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
  return 64;
#else
  return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
  return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
  return 16;
#else
  return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
}

[[noreturn]]
static __device__ void no_device_code(const char *pFileName, const int pLine, const char *pFunctionName,
                                      const int pArch, const char *pArchList) {

#if defined(GGML_USE_HIP)
  printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n", pFileName, pLine,
         pFunctionName, pArch);
  GGML_UNUSED(pArchList);
#else
  printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: "
         "%s\n",
         pFileName, pLine, pFunctionName, pArch, pArchList);
#endif // defined(GGML_USE_HIP)
  __trap();

  GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE // GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif                 // __CUDA_ARCH__

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n> struct ggml_cuda_unroll {
  template <typename Func, typename... Args> __device__ void operator()(const Func &pF, Args... pArgs) const {
    pF(n - 1, pArgs...);
    ggml_cuda_unroll<n - 1>{}(pF, pArgs...);
  }
};

template <> struct ggml_cuda_unroll<1> {
  template <typename Func, typename... Args> __device__ void operator()(const Func &pF, Args... pArgs) const {
    pF(0, pArgs...);
  }
};

template <int width = WARP_SIZE> static __device__ __forceinline__ int warp_reduce_sum(int pX) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
  return __reduce_add_sync(0xffffffff, pX);
#else
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pX += __shfl_xor_sync(0xffffffff, pX, vOffset, width);
  }
  return pX;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_sum(float pX) {
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pX += __shfl_xor_sync(0xffffffff, pX, vOffset, width);
  }
  return pX;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float2 warp_reduce_sum(float2 pA) {
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pA.x += __shfl_xor_sync(0xffffffff, pA.x, vOffset, width);
    pA.y += __shfl_xor_sync(0xffffffff, pA.y, vOffset, width);
  }
  return pA;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ half2 warp_reduce_sum(half2 pA) {
#ifdef FP16_AVAILABLE
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pA = __hadd2(pA, __shfl_xor_sync(0xffffffff, pA, vOffset, width));
  }
  return pA;

#else
  NO_DEVICE_CODE;
  return pA;
#endif // FP16_AVAILABLE
}

template <int width = WARP_SIZE> static __device__ __forceinline__ int warp_reduce_all(int pX) {
  if (width == ggml_cuda_get_physical_warp_size()) {
    return __all_sync(0xffffffff, pX);
  } else {
#pragma unroll
    for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
      pX = __shfl_xor_sync(0xffffffff, pX, vOffset, width) && pX;
    }
    return pX;
  }
}

template <int width = WARP_SIZE> static __device__ __forceinline__ int warp_reduce_any(int pX) {
  if (width == ggml_cuda_get_physical_warp_size()) {
    return __any_sync(0xffffffff, pX);
  } else {
#pragma unroll
    for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
      pX = __shfl_xor_sync(0xffffffff, pX, vOffset, width) || pX;
    }
    return pX;
  }
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_max(float pX) {
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pX = fmaxf(pX, __shfl_xor_sync(0xffffffff, pX, vOffset, width));
  }
  return pX;
}

template <typename T, int width = WARP_SIZE> static __device__ __forceinline__ T warp_prefix_inclusive_sum(T pX) {
  const int cLaneId = threadIdx.x % width;
#pragma unroll
  for (int vOffset = 1; vOffset < width; vOffset <<= 1) {
    const T cT = __shfl_up_sync(0xffffffff, pX, vOffset, width);
    if (cLaneId >= vOffset) {
      pX += cT;
    }
  }
  return pX;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 pA) {
  const int cLaneId = threadIdx.x % width;
#pragma unroll
  for (int vOffset = 1; vOffset < width; vOffset <<= 1) {
    const float cTX = __shfl_up_sync(0xffffffff, pA.x, vOffset, width);
    const float cTY = __shfl_up_sync(0xffffffff, pA.y, vOffset, width);
    if (cLaneId >= vOffset) {
      pA.x += cTX;
      pA.y += cTY;
    }
  }
  return pA;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 pA) {
#ifdef FP16_AVAILABLE
  const int cLaneId = threadIdx.x % width;
#pragma unroll
  for (int vOffset = 1; vOffset < width; vOffset <<= 1) {
    const half2 cT = __shfl_up_sync(0xffffffff, pA, vOffset, width);
    if (cLaneId >= vOffset) {
      pA = __hadd2(pA, cT);
    }
  }
  return pA;

#else
  NO_DEVICE_CODE;
  return pA;
#endif // FP16_AVAILABLE
}

enum class block_reduce_method {
  MAX,
  SUM,
};

template <block_reduce_method method_t, typename T> struct block_reduce_policy;

template <typename T, typename... Ts> inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template <typename...> inline constexpr bool ggml_cuda_dependent_false_v = false;

template <typename T> struct block_reduce_policy<block_reduce_method::SUM, T> {
  static __device__ T reduce(T pVal) {
    if constexpr (is_any<T, float, float2, half2, int>) {
      return warp_reduce_sum(pVal);
    } else {
      static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
    }
  }

  static __device__ T sentinel() {
    if constexpr (std::is_same_v<T, float>) {
      return 0.0f;
    } else if constexpr (std::is_same_v<T, float2>) {
      return make_float2(0.0f, 0.0f);
    } else if constexpr (std::is_same_v<T, half2>) {
      return make_half2(0.0f, 0.0f);
    } else if constexpr (std::is_same_v<T, int>) {
      return 0;
    } else {
      static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
    }
  }
};

template <typename T> struct block_reduce_policy<block_reduce_method::MAX, T> {
  static __device__ T reduce(T pVal) {
    if constexpr (is_any<T, float, half2>) {
      return warp_reduce_max(pVal);
    } else {
      static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
    }
  }

  static __device__ T sentinel() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else if constexpr (std::is_same_v<T, half2>) {
      return make_half2(-INFINITY, -INFINITY);
    } else {
      static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
    }
  }
};

template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T pVal, [[maybe_unused]] T *pSharedVals) {
  // for multi-warp reductions, callers must not reuse shared_vals until all reads from this invocation have completed
  pVal = block_reduce_policy<reduce_method_t, T>::reduce(pVal);
  const unsigned int cBlockSize = block_size_template == 0 ? blockDim.x : block_size_template;
  if (cBlockSize > WARP_SIZE) {
    assert((cBlockSize <= 1024) && (cBlockSize % WARP_SIZE) == 0);
    const int cWarpId = threadIdx.x / WARP_SIZE;
    const int cLaneId = threadIdx.x % WARP_SIZE;
    if (cLaneId == 0) {
      pSharedVals[cWarpId] = pVal;
    }
    __syncthreads();
    pVal = block_reduce_policy<reduce_method_t, T>::sentinel();
    if (cLaneId < (static_cast<int>(cBlockSize) / WARP_SIZE)) {
      pVal = pSharedVals[cLaneId];
    }
    return block_reduce_policy<reduce_method_t, T>::reduce(pVal);
  }

  return pVal;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half pA, const half pB) {
#ifdef FP16_AVAILABLE

#if !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX
  return __float2half(fmaxf(__half2float(pA), __half2float(pB)));
#else
  return __hmax(pA, pB);
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX

#else
  NO_DEVICE_CODE;
  GGML_UNUSED(pB);
  return pA;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 pA, const half2 pB) {
#if defined(GGML_USE_HIP)
  return half2(__hmax(pA.x, pB.x), __hmax(pA.y, pB.y));
#elif CUDART_VERSION >= CUDART_HMAX
  return __hmax2(pA, pB);
#else
  half2 ret;
  reinterpret_cast<half &>(ret.x) = __float2half(fmaxf(__low2float(pA), __low2float(pB)));
  reinterpret_cast<half &>(ret.y) = __float2half(fmaxf(__high2float(pA), __high2float(pB)));
  return ret;
#endif
}

template <int width = WARP_SIZE> static __device__ __forceinline__ half2 warp_reduce_max(half2 pX) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
#pragma unroll
  for (int vOffset = width / 2; vOffset > 0; vOffset >>= 1) {
    pX = ggml_cuda_hmax2(pX, __shfl_xor_sync(0xffffffff, pX, vOffset, width));
  }
  return pX;
#else
  GGML_UNUSED(pX);
  NO_DEVICE_CODE;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
}

#if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP)
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
  const uint32_t mask_low = 0x0000FFFF * (float(__low2half(a)) > float(__low2half(b)));
  const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
  return mask_low | mask_high;
}
#endif // (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int pA, const int pB, int pC) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
  pC = __builtin_amdgcn_sdot4(pA, pB, pC, false);
#elif defined(RDNA3) || defined(RDNA4)
  pC = __builtin_amdgcn_sudot4(true, pA, true, pB, pC, false);
#elif defined(RDNA1) || defined(__gfx900__)
  int tmp1;
  int tmp2;
  asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
      : "+v"(pC), "=&v"(tmp1), "=&v"(tmp2)
      : "v"(pA), "v"(pB));
#else
  const int8x4_t va = reinterpret_cast<const int8x4_t &>(pA);
  const int8x4_t vb = reinterpret_cast<const int8x4_t &>(pB);
  pC += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
  return pC;

#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
  return __dp4a(pA, pB, pC);
#else  // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
  const int8_t *vA8 = (const int8_t *)&pA;
  const int8_t *vB8 = (const int8_t *)&pB;
  return pC + vA8[0] * vB8[0] + vA8[1] * vB8[1] + vA8[2] * vB8[2] + vA8[3] * vB8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A

#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ void ggml_cuda_mad(float &pAcc, const float pV, const float pU) { pAcc += pV * pU; }

static __device__ __forceinline__ void ggml_cuda_mad(float &pAcc, const float2 pV, const float2 pU) {
  pAcc += pV.x * pU.x;
  pAcc += pV.y * pU.y;
}

#if defined(GGML_USE_HIP) &&                                                                                           \
    (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) ||
       // defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float &pAcc, const half2 pV, const half2 pU) {
#ifdef V_DOT2_F32_F16_AVAILABLE
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(pAcc) : "v"(pV), "v"(pU));
#else
#ifdef FAST_FP16_AVAILABLE
  const float2 cTmp = __half22float2(pV * pU);
  pAcc += cTmp.x + cTmp.y;
#else
  const float2 cTmpv = __half22float2(pV);
  const float2 cTmpu = __half22float2(pU);
  pAcc += cTmpv.x * cTmpu.x;
  pAcc += cTmpv.y * cTmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 &pAcc, const half2 pV, const half2 pU) {
#ifdef FAST_FP16_AVAILABLE
  pAcc += pV * pU;
#else
  const float2 cTmpv = __half22float2(pV);
  const float2 cTmpu = __half22float2(pU);
  float2 vTmpacc = __half22float2(pAcc);
  vTmpacc.x += cTmpv.x * cTmpu.x;
  vTmpacc.y += cTmpv.y * cTmpu.y;
  pAcc = make_half2(vTmpacc.x, vTmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right
//     instructions. If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void *__restrict__ pDst, const void *__restrict__ pSrc) {
  static_assert(
      nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
      "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
      "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
      "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be "
      "coalesced. "
      "Call ggml_cuda_memcpy_1 in a loop instead.");
  if constexpr (alignment != 0) {
    static_assert(nbytes % alignment == 0, "bad alignment");
  }
  constexpr int cNbPerCpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
  for (int vI = 0; vI < nbytes / cNbPerCpy; ++vI) {
    if constexpr (cNbPerCpy == 1) {
      ((char *)pDst)[vI] = ((const char *)pSrc)[vI];
    } else if constexpr (cNbPerCpy == 2) {
      ((short *)pDst)[vI] = ((const short *)pSrc)[vI];
    } else if constexpr (cNbPerCpy == 4) {
      ((int *)pDst)[vI] = ((const int *)pSrc)[vI];
    } else if constexpr (cNbPerCpy == 8) {
      ((int2 *)pDst)[vI] = ((const int2 *)pSrc)[vI];
    } else if constexpr (cNbPerCpy == 16) {
      ((int4 *)pDst)[vI] = ((const int4 *)pSrc)[vI];
    } else {
      static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
    }
  }
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t pX) {
#if CUDART_VERSION >= 12080
  const nv_bfloat16 cE = __nv_cvt_e8m0_to_bf16raw(pX);
  return (float)cE;
#else
  uint32_t bits;
  if (pX == 0) {
    bits = 0x00400000;
  } else {
    bits = (uint32_t)pX << 23;
  }

  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
#endif // CUDART_VERSION >= 12050
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t pX) {
#if defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
  // ROCm does not support fp8 in software on devices with fp8 hardware,
  // but CDNA3 supports only e4m3_fnuz (no inf).
  const uint32_t cBits = pX * (pX != 0x7F && pX != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
  const __hip_fp8_e4m3_fnuz cXf = *reinterpret_cast<const __hip_fp8_e4m3_fnuz *>(&cBits);
  return static_cast<float>(cXf) / 2;
#else
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
  const uint32_t cBits = pX * (pX != 0x7F && pX != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
  const __nv_fp8_e4m3 cXf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&cBits);
  return static_cast<float>(cXf) / 2;
#else
  if (pX == 0 || (pX == 0x7F && pX != 0xFF)) { // Convert NaN to 0.0f
    return 0.0f;
  }
  const int exp = (pX >> 3) & 0xF;
  const int man = pX & 0x7;
  float raw;
  if (exp == 0) {
    raw = ldexpf((float)man, -9);
  } else {
    raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  }
  return static_cast<float>(raw / 2);
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
#endif // defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
}

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float pX) {
#if defined(BLACKWELL_MMA_AVAILABLE) // This is used for NVFP4 subblock scale quantizations only
  if (!(pX > 0.0f)) {
    return 0;
  }
  const __nv_fp8_e4m3 xf(pX);
  return xf.__x;
#else
  NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

__device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float pX, float pE) {
  const uint8_t cSignBit = (pX < 0.0f) << 3;
  float vAx = fabsf(pX) * pE;

  // Positive LUT
  static constexpr float cPosLut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  int vBestI = 0;
  float vBestErr = fabsf(vAx - cPosLut[0]);

#pragma unroll
  for (int vI = 1; vI < 8; ++vI) {
    const float cErr = fabsf(vAx - cPosLut[vI]);
    if (cErr < vBestErr) {
      vBestErr = cErr;
      vBestI = vI;
    }
  }

  return static_cast<uint8_t>(vBestI | cSignBit);
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t pD64) {
  GGML_ASSERT(pD64 != 0);
  GGML_ASSERT(pD64 <= std::numeric_limits<uint32_t>::max());

  uint32_t vD = (uint32_t)pD64;

  // compute L = ceil(log2(d));
  uint32_t vL = 0;
  while (vL < 32 && (uint32_t{1} << vL) < vD) {
    vL++;
  }

  uint32_t vMp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << vL) - vD) / vD + 1);
  // pack divisor as well to reduce error surface
  return make_uint3(vMp, vL, vD);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t pN, const uint3 pFastdivValues) {
  // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
  // fastdiv_values.z is unused and optimized away by the compiler.
  // Compute high 32 bits of n * mp
  const uint32_t cHi = __umulhi(pN, pFastdivValues.x);
  // add n, apply bit shift
  return (cHi + pN) >> pFastdivValues.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t pN, const uint3 pFastdivValues) {
  // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
  return pN - fastdiv(pN, pFastdivValues) * pFastdivValues.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t pN, const uint3 pFastdivValues) {
  // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
  const uint32_t cDivVal = fastdiv(pN, pFastdivValues);
  const uint32_t cModVal = pN - cDivVal * pFastdivValues.z;
  return make_uint2(cDivVal, cModVal);
}

typedef void (*dequantize_kernel_t)(const void *vx, const int64_t ib, const int pIqs, float2 &pV);

template <typename dst_t> using dequantize_kq_t = void (*)(const void *vx, const int64_t ib, dst_t *pY, const int pTid);

static __device__ __forceinline__ float get_alibi_slope(const float pMaxBias, const uint32_t pH,
                                                        const uint32_t pNHeadLog2, const float pM0, const float pM1) {
  if (pMaxBias <= 0.0f) {
    return 1.0f;
  }
  const float cBase = pH < pNHeadLog2 ? pM0 : pM1;
  const int cExph = pH < pNHeadLog2 ? pH + 1 : 2 * (pH - pNHeadLog2) + 1;

  return powf(cBase, cExph);
}

template <ggml_type type> struct ggml_cuda_type_traits;

template <> struct ggml_cuda_type_traits<GGML_TYPE_F16> {
  static constexpr int qk = 1;
  static constexpr int qr = 1;
  static constexpr int bs = sizeof(ggml_half);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q1_0> {
  static constexpr int qk = QK1_0;
  static constexpr int qr = QR1_0;
  static constexpr int qi = QI1_0;
  static constexpr int bs = sizeof(block_q1_0);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q2_0> {
  static constexpr int qk = QK2_0;
  static constexpr int qr = QR2_0;
  static constexpr int qi = QI2_0;
  static constexpr int bs = sizeof(block_q2_0);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
  static constexpr int qk = QK4_0;
  static constexpr int qr = QR4_0;
  static constexpr int qi = QI4_0;
  static constexpr int bs = sizeof(block_q4_0);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
  static constexpr int qk = QK4_1;
  static constexpr int qr = QR4_1;
  static constexpr int qi = QI4_1;
  static constexpr int bs = sizeof(block_q4_1);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
  static constexpr int qk = QK5_0;
  static constexpr int qr = QR5_0;
  static constexpr int qi = QI5_0;
  static constexpr int bs = sizeof(block_q5_0);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
  static constexpr int qk = QK5_1;
  static constexpr int qr = QR5_1;
  static constexpr int qi = QI5_1;
  static constexpr int bs = sizeof(block_q5_1);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
  static constexpr int qk = QK8_0;
  static constexpr int qr = QR8_0;
  static constexpr int qi = QI8_0;
  static constexpr int bs = sizeof(block_q8_0);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
  static constexpr int qk = QK_MXFP4;
  static constexpr int qr = QR_MXFP4;
  static constexpr int qi = QI_MXFP4;
  static constexpr int bs = sizeof(block_mxfp4);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_NVFP4> {
  static constexpr int qk = QK_NVFP4;
  static constexpr int qr = QR_NVFP4;
  static constexpr int qi = QI_NVFP4;
  static constexpr int bs = sizeof(block_nvfp4);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR2_K;
  static constexpr int qi = QI2_K;
  static constexpr int bs = sizeof(block_q2_K);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR3_K;
  static constexpr int qi = QI3_K;
  static constexpr int bs = sizeof(block_q3_K);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR4_K;
  static constexpr int qi = QI4_K;
  static constexpr int bs = sizeof(block_q4_K);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR5_K;
  static constexpr int qi = QI5_K;
  static constexpr int bs = sizeof(block_q5_K);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR6_K;
  static constexpr int qi = QI6_K;
  static constexpr int bs = sizeof(block_q6_K);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR2_XXS;
  static constexpr int qi = QI2_XXS;
  static constexpr int bs = sizeof(block_iq2_xxs);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR2_XS;
  static constexpr int qi = QI2_XS;
  static constexpr int bs = sizeof(block_iq2_xs);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR2_S;
  static constexpr int qi = QI2_S;
  static constexpr int bs = sizeof(block_iq2_s);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR3_XXS;
  static constexpr int qi = QI3_XXS;
  static constexpr int bs = sizeof(block_iq3_xxs);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR1_S;
  static constexpr int qi = QI1_S;
  static constexpr int bs = sizeof(block_iq1_s);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR1_M;
  static constexpr int qi = QI1_M;
  static constexpr int bs = sizeof(block_iq1_m);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
  static constexpr int qk = QK4_NL;
  static constexpr int qr = QR4_NL;
  static constexpr int qi = QI4_NL;
  static constexpr int bs = sizeof(block_iq4_nl);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR4_XS;
  static constexpr int qi = QI4_XS;
  static constexpr int bs = sizeof(block_iq4_xs);
};

template <> struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
  static constexpr int qk = QK_K;
  static constexpr int qr = QR3_S;
  static constexpr int qi = QI3_S;
  static constexpr int bs = sizeof(block_iq3_s);
};

//////////////////////

struct ggml_cuda_device_info {
  int device_count;          // number of (possibly virtual) devices exposed to the rest of ggml
  int physical_device_count; // number of physical CUDA devices actually present

  struct cuda_device_info {
    int cc;                 // compute capability
    int nsm;                // number of streaming multiprocessors
    size_t smpb;            // max. shared memory per block
    size_t smpbo;           // max. shared memory per block (with opt-in)
    bool integrated;        // Device is integrated as opposed to discrete
    bool vmm;               // virtual memory support
    size_t vmm_granularity; // granularity of virtual memory
    size_t total_vram;
    int warp_size;                    // Number of threads in a dispatch
    bool supports_cooperative_launch; // whether cooperative launch is supported
    int physical_device;              // backing physical CUDA device for this (virtual) device
    int physical_share_count;         // number of (virtual) devices sharing this device's physical GPU
    int virtual_index;                // index of this (virtual) device among those sharing its physical GPU
  };

  cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

  std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info &ggml_cuda_info();

void ggml_cuda_set_device(int pDevice);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
  virtual ~ggml_cuda_pool() = default;

  virtual void *alloc(size_t pSize, size_t *pActualSize) = 0;
  virtual void free(void *pPtr, size_t pSize) = 0;
};

template <typename T> struct ggml_cuda_pool_alloc {
  ggml_cuda_pool *pool = nullptr;
  T *ptr = nullptr;
  size_t actual_size = 0;

  ggml_cuda_pool_alloc() = default;

  explicit ggml_cuda_pool_alloc(ggml_cuda_pool &pPool) : pool(&pPool) {}

  ggml_cuda_pool_alloc(ggml_cuda_pool &pPool, size_t pSize) : pool(&pPool) { alloc(pSize); }

  ~ggml_cuda_pool_alloc() {
    if (ptr != nullptr) {
      pool->free(ptr, actual_size);
    }
  }

  // size is in number of elements
  T *alloc(size_t pSize) {
    GGML_ASSERT(pool != nullptr);
    GGML_ASSERT(ptr == nullptr);
    ptr = (T *)pool->alloc(pSize * sizeof(T), &this->actual_size);
    return ptr;
  }

  T *alloc(ggml_cuda_pool &pPool, size_t pSize) {
    this->pool = &pPool;
    return alloc(pSize);
  }

  T *get() { return ptr; }

  ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
  ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
  ggml_cuda_pool_alloc &operator=(const ggml_cuda_pool_alloc &) = delete;
  ggml_cuda_pool_alloc &operator=(ggml_cuda_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void *data_device[GGML_CUDA_MAX_DEVICES];                         // 1 pointer for each device for split tensors
  cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};

#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS))
#define USE_CUDA_GRAPH
#endif

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
  ~ggml_cuda_graph() {
    if (instance != nullptr) {
      CUDA_CHECK(cudaGraphExecDestroy(instance));
    }
    if (graph != nullptr) {
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;
  size_t num_nodes = 0;
  std::vector<cudaGraphNode_t> nodes;
  bool disable_due_to_gpu_arch = false;
  bool warmup_complete = false;
  uint64_t uid = 0;
  int64_t last_used_time = 0;
  struct node_properties {
    ggml_tensor node;
    void *node_src_data_ptrs[GGML_MAX_SRC];
    int64_t node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
  };
  std::vector<node_properties> node_props;

  bool is_enabled() const {
    static const bool cDisableCudaGraphsDueToEnv = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
    return !(disable_due_to_gpu_arch || cDisableCudaGraphsDueToEnv);
  }
#endif
};

struct ggml_cuda_concurrent_event {
  std::vector<cudaEvent_t> join_events;
  cudaEvent_t fork_event = nullptr;

  int n_streams = 0;
  std::unordered_map<const ggml_tensor *, int> stream_mapping;

  // Original order of nodes in this concurrent region (before interleaving)
  // Used to restore grouping for fusion within streams
  std::vector<const ggml_tensor *> original_order;

  const ggml_tensor *join_node;

  ggml_cuda_concurrent_event() = default;

  ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event &) = delete;
  ggml_cuda_concurrent_event &operator=(const ggml_cuda_concurrent_event &) = delete;

  explicit ggml_cuda_concurrent_event(int pNStreams) : n_streams(pNStreams) {
    join_events.resize(pNStreams);

    for (size_t vI = 0; vI < join_events.size(); ++vI) {
      CUDA_CHECK(cudaEventCreateWithFlags(&join_events[vI], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
  }

  ggml_cuda_concurrent_event(ggml_cuda_concurrent_event &&pOther) noexcept
      : join_events(std::move(pOther.join_events)), fork_event(pOther.fork_event), n_streams(pOther.n_streams),
        stream_mapping(std::move(pOther.stream_mapping)), original_order(std::move(pOther.original_order)),
        join_node(pOther.join_node) {
    pOther.fork_event = nullptr;
  }

  // 1. check if any branches write to overlapping memory ranges (except the join node)
  // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
  // we assume all nodes have the same buffer
  bool is_valid() const {
    std::vector<std::vector<std::pair<int64_t, int64_t>>> vWriteRanges;
    vWriteRanges.resize(n_streams);

    // get join_node's memory range to exclude from overlap checking.
    // multiple nodes can use join_node's buffer; we synchronize on the join node.
    const ggml_tensor *vJoinT = join_node->view_src ? join_node->view_src : join_node;
    const int64_t cJoinStart = (int64_t)vJoinT->data;
    const int64_t cJoinEnd = cJoinStart + ggml_nbytes(vJoinT);

    for (const auto &[tensor, stream] : stream_mapping) {
      const ggml_tensor *vT = tensor->view_src ? tensor->view_src : tensor;
      const int64_t cTStart = (int64_t)vT->data;
      const int64_t cTEnd = cTStart + ggml_nbytes(vT);

      // skip tensors that overlap with join_node's buffer.
      if ((cTStart <= cJoinStart && cJoinStart < cTEnd) || (cJoinStart <= cTStart && cTStart < cJoinEnd)) {
        continue;
      }

      // concurrent streams begin from 1
      vWriteRanges[stream - 1].emplace_back(cTStart, cTEnd);
    }

    for (int vI = 0; vI < n_streams; ++vI) {
      // sorts first by start then by end of write range
      std::sort(vWriteRanges[vI].begin(), vWriteRanges[vI].end());
    }

    bool vWritesOverlap = false;
    bool vDependentSrcs = false;
    for (const auto &[tensor, stream] : stream_mapping) {
      const ggml_tensor *vT = tensor->view_src ? tensor->view_src : tensor;
      const int64_t cTStart = (int64_t)vT->data;
      const int64_t cTEnd = cTStart + ggml_nbytes(vT);

      // skip tensors that overlap with join_node's buffer
      if ((cTStart <= cJoinStart && cJoinStart < cTEnd) || (cJoinStart <= cTStart && cTStart < cJoinEnd)) {
        continue;
      }

      // check if this buffer's write data overlaps with another stream's
      std::pair<int64_t, int64_t> vDataRange = std::make_pair(cTStart, cTEnd);
      for (int vI = 0; vI < n_streams; ++vI) {
        if (vI == stream - 1) {
          continue;
        }
        auto vIt = std::lower_bound(vWriteRanges[vI].begin(), vWriteRanges[vI].end(), vDataRange);

        if (vIt != vWriteRanges[vI].end()) {
          const std::pair<int64_t, int64_t> &vOther = *vIt;

          // std::lower_bound returns the first element where other >= data_range (lexicographically).
          // This guarantees other.first >= data_range.first.
          // Therefore, overlap occurs iff other.first < data_range.second
          // (i.e., the other range starts before this range ends).
          if (vOther.first < vDataRange.second) {
            GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
            vWritesOverlap = true;
            break;
          }
        }
      }

      // check if all srcs are either in branch or don't have a branch
      for (int vI = 0; vI < GGML_MAX_SRC; ++vI) {
        if (!tensor->src[vI]) {
          continue;
        }

        auto vIt = stream_mapping.find(tensor->src[vI]);

        if (vIt == stream_mapping.end()) {
          continue;
        }

        if (vIt->second != stream) {
          vDependentSrcs = true;
          break;
        }
      }

      if (vDependentSrcs || vWritesOverlap) {
        break;
      }
    }

    return !vWritesOverlap && !vDependentSrcs;
  }

  ~ggml_cuda_concurrent_event() {
    if (fork_event != nullptr) {
      CUDA_CHECK(cudaEventDestroy(fork_event));
    }
    for (cudaEvent_t vE : join_events) {
      if (vE != nullptr) {
        CUDA_CHECK(cudaEventDestroy(vE));
      }
    }
  }
};

struct ggml_cuda_stream_context {
  std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;

  void reset() { concurrent_events.clear(); }
};

struct ggml_backend_cuda_context {
  int device;
  std::string name;
  cudaEvent_t copy_event = nullptr;

  cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {{nullptr}};
  cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

  int curr_stream_no = 0;

#ifdef USE_CUDA_GRAPH
  // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
  // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
  std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;

  int64_t last_graph_eviction_sweep = 0;

  ggml_cuda_graph *cuda_graph(const void *pFirstNodePtr) {
    const int64_t cTimeNow = ggml_time_us();

    // sweep every 5s, evicting cuda graphs unused for >=10s
    if (cTimeNow - last_graph_eviction_sweep >= 5'000'000) {
      last_graph_eviction_sweep = cTimeNow;
      for (auto vIt = cuda_graphs.begin(); vIt != cuda_graphs.end();) {
        if (cTimeNow - vIt->second->last_used_time >= 10'000'000) {
          vIt = cuda_graphs.erase(vIt);
        } else {
          ++vIt;
        }
      }
    }

    auto vIt = cuda_graphs.find(pFirstNodePtr);
    if (vIt == cuda_graphs.end()) {
      vIt = cuda_graphs.emplace(pFirstNodePtr, std::make_unique<ggml_cuda_graph>()).first;
    }
    vIt->second->last_used_time = cTimeNow;
    return vIt->second.get();
  }

  // Check if any CUDA graph is enabled for this context (used by kernels that need to know
  // if graphs are in use without having access to the specific graph key)
  bool any_cuda_graph_enabled() const {
    for (const auto &[key, graph] : cuda_graphs) {
      if (graph && graph->is_enabled()) {
        return true;
      }
    }
    return false;
  }

  // Check if any CUDA graph has an instance for this context
  bool any_cuda_graph_has_instance() const {
    for (const auto &[key, graph] : cuda_graphs) {
      if (graph && graph->instance != nullptr) {
        return true;
      }
    }
    return false;
  }
#endif // USE_CUDA_GRAPH

  explicit ggml_backend_cuda_context(int pDevice) : device(pDevice), name(GGML_CUDA_NAME + std::to_string(pDevice)) {}

  ggml_cuda_stream_context concurrent_stream_context;

  ~ggml_backend_cuda_context();

  cudaStream_t stream(int pDevice, int pStream) {
    if (streams[pDevice][pStream] == nullptr) {
      ggml_cuda_set_device(pDevice);
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[pDevice][pStream], cudaStreamNonBlocking));
    }
    return streams[pDevice][pStream];
  }

  cudaStream_t stream() { return stream(device, curr_stream_no); }

  ggml_cuda_stream_context &stream_context() { return concurrent_stream_context; }

  cublasHandle_t cublas_handle(int pDevice) {
    if (cublas_handles[pDevice] == nullptr) {
      ggml_cuda_set_device(pDevice);
      CUBLAS_CHECK(cublasCreate(&cublas_handles[pDevice]));
      CUBLAS_CHECK(cublasSetMathMode(cublas_handles[pDevice], CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return cublas_handles[pDevice];
  }

  cublasHandle_t cublas_handle() { return cublas_handle(device); }

  // pool
  std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];

  static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int pDevice, int pStreamNo);

  ggml_cuda_pool &pool(int pDevice) {
    if (pools[pDevice][curr_stream_no] == nullptr) {
      pools[pDevice][curr_stream_no] = new_pool_for_device(pDevice, curr_stream_no);
    }
    return *pools[pDevice][curr_stream_no];
  }

  ggml_cuda_pool &pool() { return pool(device); }
};

struct ggml_cuda_mm_fusion_args_host {
  const ggml_tensor *x_bias = nullptr;
  const ggml_tensor *gate = nullptr;
  const ggml_tensor *gate_bias = nullptr;
  const ggml_tensor *x_scale = nullptr;
  const ggml_tensor *gate_scale = nullptr;
  ggml_glu_op glu_op;
  float glu_limit = 0.0f;
};
struct ggml_cuda_mm_fusion_args_device {
  const void *x_bias = nullptr;
  const void *gate = nullptr;
  const void *gate_bias = nullptr;
  const void *x_scale = nullptr;
  const void *gate_scale = nullptr;
  ggml_glu_op glu_op;
  float glu_limit = 0.0f;
};

struct ggml_cuda_kernel_launch_params {
  dim3 block_nums;
  dim3 block_dims;
  size_t shmem;
  cudaStream_t stream;

  // size_t shmem
  ggml_cuda_kernel_launch_params(const dim3 &pBlockNums, const dim3 &pBlockDims, const size_t pShmem,
                                 const cudaStream_t pStream)
      : block_nums(pBlockNums), block_dims(pBlockDims), shmem(pShmem), stream(pStream) {}

  // Some call sites pass ints instead of the required size_t. This 2nd constructor casts int->size_t to avoid these
  // -Wnarrowing warnings.
  ggml_cuda_kernel_launch_params(const dim3 &pBlockNums, const dim3 &pBlockDims, const int pShmem,
                                 const cudaStream_t pStream)
      : block_nums(pBlockNums), block_dims(pBlockDims), shmem((size_t)pShmem), stream(pStream) {}
};

#if defined(GGML_CUDA_USE_PDL)
struct ggml_cuda_pdl_config {
  cudaLaunchAttribute attr;
  cudaLaunchConfig_t cfg;

  ggml_cuda_pdl_config(const ggml_cuda_kernel_launch_params &pParams) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;

    cfg = {};
    cfg.gridDim = pParams.block_nums;
    cfg.blockDim = pParams.block_dims;
    cfg.dynamicSmemBytes = pParams.shmem;
    cfg.stream = pParams.stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
  }

  // Delete due to &attr
  ggml_cuda_pdl_config(const ggml_cuda_pdl_config &) = delete;
  ggml_cuda_pdl_config &operator=(const ggml_cuda_pdl_config &) = delete;
  ggml_cuda_pdl_config &operator=(ggml_cuda_pdl_config &&) = delete;
};

static bool ggml_cuda_kernel_can_use_pdl(const void *pKernel) {
  const int cDevice = ggml_cuda_get_device();

  struct cache_key {
    int device;
    const void *kernel;

    bool operator==(const cache_key &pOther) const { return device == pOther.device && kernel == pOther.kernel; }
  };

  struct cache_key_hash {
    // MurmurHash3 mixing function for better hash distribution (vs. just std::hash which in some implementations simply
    // returns the identity)
    static size_t hash_mix(size_t pX) {
      std::uint64_t vY = pX;
      const std::uint64_t cM = 0xe9846af9b1a615d;

      vY ^= vY >> 32;
      vY *= cM;
      vY ^= vY >> 32;
      vY *= cM;
      vY ^= vY >> 28;

      return static_cast<size_t>(vY);
    }

    size_t operator()(const cache_key &pKey) const {
      // Use a nonzero seed to avoid mapping all-zero keys to zero
      size_t vH = 42;
      vH = hash_mix(vH + pKey.device);
      vH = hash_mix(vH + reinterpret_cast<size_t>(pKey.kernel));
      return vH;
    }
  };

  static std::mutex vCacheMutex;
  static std::unordered_map<cache_key, bool, cache_key_hash> vCache;

  const cache_key cKey = {cDevice, pKernel};
  std::lock_guard<std::mutex> vLock(vCacheMutex);
  const auto cIt = vCache.find(cKey);
  if (cIt != vCache.end()) {
    return cIt->second;
  }

  cudaFuncAttributes vAttr = {};
  CUDA_CHECK(cudaFuncGetAttributes(&vAttr, pKernel));

  // PDL device-side primitives are emitted only for PTX versions >= 90.
  // We have to guard on a loaded kernel's PTX version so a kernel forward-JIT'ed
  // from pre-Hopper PTX to a Hopper-or-newer GPU does not opt into PDL.
  const bool cCanUsePdl = vAttr.ptxVersion >= 90;
  vCache.emplace(cKey, cCanUsePdl);
  return cCanUsePdl;
}

#endif // defined(GGML_CUDA_USE_PDL)

// PDL and __restrict__ need to be mutually exclusive
#if (defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER)
#define GGML_CUDA_RESTRICT
#else
#define GGML_CUDA_RESTRICT __restrict__
#endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER

template <typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch(Kernel pKernel, const ggml_cuda_kernel_launch_params &pLaunchParams,
                                               Args &&...pArgs) {
#if defined(GGML_CUDA_USE_PDL)

  static const bool cEnvPdlEnabled = []() {
    const char *vEnv = getenv("GGML_CUDA_PDL");
    return vEnv == nullptr || std::atoi(vEnv) != 0;
  }();

  if (cEnvPdlEnabled && ggml_cuda_kernel_can_use_pdl(reinterpret_cast<const void *>(pKernel))) {
    auto vPdlCfg = ggml_cuda_pdl_config(pLaunchParams);

    CUDA_CHECK(cudaLaunchKernelEx(&vPdlCfg.cfg, pKernel, std::forward<Args>(pArgs)...));
    return;
  }
#endif // defined(GGML_CUDA_USE_PDL)

  pKernel<<<pLaunchParams.block_nums, pLaunchParams.block_dims, pLaunchParams.shmem, pLaunchParams.stream>>>(
      std::forward<Args>(pArgs)...);
  CUDA_CHECK(cudaGetLastError());
}
