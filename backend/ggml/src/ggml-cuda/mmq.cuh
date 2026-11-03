#pragma once

#include "common.cuh"

#include <climits>
#include <cstdint>

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256
#define MMQ_ITER_K_FP4 512
#define MMQ_NWARPS 8

typedef void (*ggml_cuda_mmq_load_tiles_t)(const char *__restrict__ x, int *pXTile, const int pKbx0, const int i_max,
                                           const int stride);
typedef void (*ggml_cuda_mmq_vec_dot_t)(const int *__restrict__ x, const int *__restrict__ pY, float *__restrict__ sum,
                                        const int pK00);
typedef void (*ggml_cuda_mmq_write_back_t)(const float *__restrict__ sum, const int32_t *__restrict__ pGetRowsToSorted,
                                           float *__restrict__ pDst, const float *__restrict__ pYScale,
                                           const int stride, const int i_max, const int pJMax);

enum mmq_q8_1_ds_layout {
  MMQ_Q8_1_DS_LAYOUT_D4,
  MMQ_Q8_1_DS_LAYOUT_DS4,
  MMQ_Q8_1_DS_LAYOUT_D2S6,
};

static constexpr int QK8_1_MMQ = 4 * QK8_1;
static constexpr int QK_FP4_MMQ = 2 * QK8_1_MMQ;

struct block_q8_1_mmq {
  // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
  // The y float data is first grouped as blocks of 128 values.
  // These blocks are then treated as individual data values and transposed.
  //
  // To avoid shared memory bank conflicts each block is padded with 16 bytes.
  // This padding is also used to store block scales/partial sums.
  // The scales multiplied with the quantized data are equal to the unquantized values.
  // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
  //     and are only needed for performance reasons.
  //
  // The exact data stored depends on the x data type.
  union {
    float d4[4];  // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
    half2 ds4[4]; // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
    half d2s6[8]; // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                  //     stored as d0,d1,s1,s2,s3,s4,s5
  };
  int8_t qs[QK8_1_MMQ];
};

// this struct is used for fp4 data types (currently only used for Blackwell)
// mxfp4 has block size 32, each int32 of d4 contains 2 e8m0 scales in the lower 16 bits
// nvfp4 has block size 16, each int32 of d4 contains 4 ue4m3 scales
struct block_fp4_mmq {
  uint32_t d4[4];
  int8_t qs[QK_FP4_MMQ / 2];
};

static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4 * sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq) == sizeof(block_q8_1_mmq), "Unexpected block_fp4_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type pTypeX) {
  switch (pTypeX) {
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_Q5_0:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_Q5_1:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_Q8_0:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_MXFP4:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_NVFP4:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_Q2_K:
    return MMQ_Q8_1_DS_LAYOUT_D2S6;
  case GGML_TYPE_Q3_K:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ3_S:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_IQ1_S:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ4_NL:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

struct tile_x_sizes {
  int qs;
  int dm;
  int sc;
};

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K

enum ggml_cuda_mmq_sram_layout {
  GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
  GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
  GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,
  GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
  GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
  GGML_CUDA_MMQ_SRAM_LAYOUT_FP4,   // MXFP4 and NVFP4 on Blackwell.
  GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, // Generic NVFP4
};

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout pSramLayout) {
  switch (pSramLayout) {
  case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
    return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
    return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_1 + 4;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
    return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K + 4;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
    return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
    return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_FP4:
    return 2 * MMQ_TILE_NE_K + 8 + 4;
  case GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4:
    return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4;
  default:
    return -1;
  }
}

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4) % 8 == 4, "Wrong padding.");

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) ==
                  ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1),
              "Wrong tile size for MXFP4");

// Config options for the MMQ kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct ggml_cuda_mmq_config {
  ggml_type type; // src0->type
  int nthreads;   // Number of threads per CUDA block.
  int occupancy;  // Targeted occupancy for the MMA kernel.
  int I;          // SRAM tile width in src0->ne[1]/dst->ne[0] direction.
  int J;          // SRAM tile width in src1->ne[1]/dst->ne[1] direction.
  ggml_cuda_mmq_sram_layout
      sram_layout; // SRAM tile length in src0->ne[0]/src1->ne[0] direction (physical 32 bit elements).
  int K_vram;      // VRAM tile length in src0->ne[0]/src1->ne[0] direction (logical elements).
  bool stream_k;   // Whether or not to use stream-k decomposition.
  bool fallback;   // Whether a fallback for out-of-bounds check in src0->ne[1] direction is needed.

  constexpr __host__ __device__ ggml_cuda_mmq_config(ggml_type pType, int pNthreads, int pOccupancy, int pI, int pJ,
                                                     ggml_cuda_mmq_sram_layout pSramLayout, int pKVram, bool pStreamK,
                                                     bool pFallback)
      : type(pType), nthreads(pNthreads), occupancy(pOccupancy), I(pI), J(pJ), sram_layout(pSramLayout), K_vram(pKVram),
        stream_k(pStreamK), fallback(pFallback) {}

  constexpr __device__ int rows_per_warp() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 16;
#else
    return J >= 48 && J % 16 == 0 ? 32 : 16;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  // TODO transition all combinations of GPUs and quantizations to the MMA data layout.
  __host__ int use_mma_data_layout(const int pCc) const {
    if (amd_mfma_available(pCc) || amd_wmma_available(pCc) || turing_mma_available(pCc)) {
      return true;
    }
    return false;
  }

  constexpr __device__ bool use_mma_data_layout() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    return true;
#else
    return false;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
  }
};

#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                        \
  if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                       \
    static_assert((nthreads_) % 32 == 0 && (nthreads_) <= 512, "bad nthreads");                                        \
    static_assert((occupancy_) <= 8, "bad occupancy");                                                                 \
    static_assert((I_) % 32 == 0, "bad I");                                                                            \
    static_assert((J_) % 8 == 0, "bad J");                                                                             \
    static_assert((K_vram_) % 256 == 0, "bad K_vram");                                                                 \
    return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_),             \
                                (stream_k_), (fallback_));                                                             \
  }

#include "mmq-config-pascal-older.cuh"
#include "mmq-config-pascal-dp4a.cuh"
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"

#include "mmq-config-cdna.cuh"
#include "mmq-config-rdna2.cuh"
#include "mmq-config-rdna3.cuh"
#include "mmq-config-rdna3-5.cuh"
#include "mmq-config-rdna4.cuh"

#undef CASE

static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type pType, const int pJ, const bool pFallback,
                                                              const int pCc) {
  if (GGML_CUDA_CC_IS_AMD(pCc)) {
    if (GGML_CUDA_CC_IS_CDNA(pCc)) {
      return ggml_cuda_mmq_get_config_cdna(pType, pJ, pFallback);
    }
    if (GGML_CUDA_CC_IS_RDNA4(pCc)) {
      return ggml_cuda_mmq_get_config_rdna4(pType, pJ, pFallback);
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(pCc)) {
      return ggml_cuda_mmq_get_config_rdna3_5(pType, pJ, pFallback);
    }
    if (GGML_CUDA_CC_IS_RDNA3(pCc)) { // covers RDNA 3.0
      return ggml_cuda_mmq_get_config_rdna3(pType, pJ, pFallback);
    }
    return ggml_cuda_mmq_get_config_rdna2(pType, pJ, pFallback);
  }
  if (blackwell_mma_available(pCc)) {
    return ggml_cuda_mmq_get_config_blackwell(pType, pJ, pFallback);
  }
  if (ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_VOLTA) {
    return ggml_cuda_mmq_get_config_ampere(pType, pJ, pFallback);
  }
  if (ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_DP4A) {
    return ggml_cuda_mmq_get_config_pascal_dp4a(pType, pJ, pFallback);
  }
  return ggml_cuda_mmq_get_config_pascal_older(pType, pJ, pFallback);
}

static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type pType, int pJ, bool pFallback) {
#ifdef GGML_USE_HIP
#ifdef CDNA
  return ggml_cuda_mmq_get_config_cdna(pType, pJ, pFallback);
#elif defined(RDNA4)
  return ggml_cuda_mmq_get_config_rdna4(pType, pJ, pFallback);
#elif defined(RDNA3_5)
  return ggml_cuda_mmq_get_config_rdna3_5(pType, pJ, pFallback);
#elif defined(RDNA3)
  return ggml_cuda_mmq_get_config_rdna3(pType, pJ, pFallback);
#else
  return ggml_cuda_mmq_get_config_rdna2(pType, pJ, pFallback);
#endif // CDNA
#else
#ifdef BLACKWELL_MMA_AVAILABLE
  return ggml_cuda_mmq_get_config_blackwell(pType, pJ, pFallback);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
  return ggml_cuda_mmq_get_config_ampere(pType, pJ, pFallback);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
  return ggml_cuda_mmq_get_config_pascal_dp4a(pType, pJ, pFallback);
#else
  return ggml_cuda_mmq_get_config_pascal_older(pType, pJ, pFallback);
#endif // BLACKWELL_MMA_AVAILABLE
#endif // GGML_USE_HIP
  GGML_UNUSED_VARS(pType, pJ, pFallback);
}

static __host__ int ggml_cuda_mmq_get_type(const ggml_type pType, const int pJ, const bool pFallback, const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).type;
}

static constexpr __device__ int ggml_cuda_mmq_get_type(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).type;
}

static __host__ int ggml_cuda_mmq_get_nthreads(const ggml_type pType, const int pJ, const bool pFallback,
                                               const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).nthreads;
}

static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).nthreads;
}

static __host__ int ggml_cuda_mmq_get_occupancy(const ggml_type pType, const int pJ, const bool pFallback,
                                                const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).occupancy;
}

static constexpr __device__ int ggml_cuda_mmq_get_occupancy(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).occupancy;
}

static __host__ int ggml_cuda_mmq_get_I(const ggml_type pType, const int pJ, const bool pFallback, const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).I;
}

static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).I;
}

static __host__ int ggml_cuda_mmq_get_J(const ggml_type pType, const int pJ, const bool pFallback, const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).J;
}

static constexpr __device__ int ggml_cuda_mmq_get_J(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).J;
}

static __host__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(const ggml_type pType, const int pJ,
                                                                        const bool pFallback, const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).sram_layout;
}

static constexpr __device__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(ggml_type pType, int pJ,
                                                                                    bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).sram_layout;
}

static __host__ int ggml_cuda_mmq_get_K_vram(const ggml_type pType, const int pJ, const bool pFallback, const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).K_vram;
}

static constexpr __device__ int ggml_cuda_mmq_get_K_vram(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).K_vram;
}

static __host__ bool ggml_cuda_mmq_get_stream_k(const ggml_type pType, const int pJ, const bool pFallback,
                                                const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).stream_k;
}

static constexpr __device__ bool ggml_cuda_mmq_get_stream_k(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).stream_k;
}

static __host__ int ggml_cuda_mmq_get_fallback(const ggml_type pType, const int pJ, const bool pFallback,
                                               const int pCc) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback, pCc).fallback;
}

static constexpr __device__ int ggml_cuda_mmq_get_fallback(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).fallback;
}

// ---------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_mmq_get_sram_stride(const ggml_type pType, const int pJ, const bool pFallback,
                                                  const int pCc) {
  return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(pType, pJ, pFallback, pCc));
}

static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(pType, pJ, pFallback));
}

static __host__ int ggml_cuda_mmq_get_J_max(const ggml_type pType, const bool pFallback, const int pCc,
                                            const int64_t pNe11) {
  int vRet = std::min(pNe11, int64_t(512));
  vRet -= vRet % 8;
  for (; vRet > 0; vRet -= 8) {
    if (ggml_cuda_mmq_get_config(pType, vRet, pFallback, pCc).type != GGML_TYPE_COUNT) {
      return vRet;
    }
  }
  return vRet;
}

static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type pType, int pJ, bool pFallback) {
  return ggml_cuda_mmq_get_config(pType, pJ, pFallback).rows_per_warp();
}

#define MMQ_DP4A_TXS_Q4_0 tile_x_sizes{I * MMQ_TILE_NE_K + I, I * MMQ_TILE_NE_K / QI4_0 + I / QI4_0, 0}
#define MMQ_DP4A_TXS_Q4_1 tile_x_sizes{I * MMQ_TILE_NE_K + I, I * MMQ_TILE_NE_K / QI4_1 + I / QI4_1, 0}
#define MMQ_DP4A_TXS_Q8_0 tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K * 2 / QI8_0 + I / (QI8_0 / 2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K * 4 / QI8_0 + I / (QI8_0 / 4), 0}
#define MMQ_DP4A_TXS_Q8_1 tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K * 2 / QI8_1 + I / (QI8_1 / 2), 0}
#define MMQ_DP4A_TXS_Q2_K tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K + I, 0}
#define MMQ_DP4A_TXS_Q3_K tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I, I * MMQ_TILE_NE_K / 8 + I / 8}
#define MMQ_DP4A_TXS_Q4_K tile_x_sizes{I * MMQ_TILE_NE_K + I, I * MMQ_TILE_NE_K / QI4_K, I * MMQ_TILE_NE_K / 8 + I / 8}
#define MMQ_DP4A_TXS_Q5_K                                                                                              \
  tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K / QI5_K + I / QI5_K, I * MMQ_TILE_NE_K / 8 + I / 8}
#define MMQ_DP4A_TXS_Q6_K                                                                                              \
  tile_x_sizes{I * MMQ_TILE_NE_K * 2 + I, I * MMQ_TILE_NE_K / QI6_K + I / QI6_K, I * MMQ_TILE_NE_K / 8 + I / 8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type pType, int I) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_Q2_0:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_Q4_0:
    return MMQ_DP4A_TXS_Q4_0;
  case GGML_TYPE_Q4_1:
    return MMQ_DP4A_TXS_Q4_1;
  case GGML_TYPE_Q5_0:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_Q5_1:
    return MMQ_DP4A_TXS_Q8_1;
  case GGML_TYPE_Q8_0:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_MXFP4:
    return MMQ_DP4A_TXS_Q8_1;
  case GGML_TYPE_NVFP4:
    return MMQ_DP4A_TXS_Q8_0_16;
  case GGML_TYPE_Q2_K:
    return MMQ_DP4A_TXS_Q2_K;
  case GGML_TYPE_Q3_K:
    return MMQ_DP4A_TXS_Q3_K;
  case GGML_TYPE_Q4_K:
    return MMQ_DP4A_TXS_Q4_K;
  case GGML_TYPE_Q5_K:
    return MMQ_DP4A_TXS_Q5_K;
  case GGML_TYPE_Q6_K:
    return MMQ_DP4A_TXS_Q6_K;
  case GGML_TYPE_IQ2_XXS:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_IQ2_XS:
    return MMQ_DP4A_TXS_Q8_0_16;
  case GGML_TYPE_IQ2_S:
    return MMQ_DP4A_TXS_Q8_0_16;
  case GGML_TYPE_IQ3_XXS:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_IQ3_S:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_IQ1_S:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_IQ4_XS:
    return MMQ_DP4A_TXS_Q8_0;
  case GGML_TYPE_IQ4_NL:
    return MMQ_DP4A_TXS_Q8_0;
  default:
    return tile_x_sizes{0, 0, 0};
  }
}

// FIXME temporary until all combinations of data types and GPUs can use the MMA data layout
static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config &pConfig, const int pCc) {
  if (pConfig.use_mma_data_layout(pCc)) {
    return pConfig.I * ggml_cuda_mmq_get_sram_stride(pConfig.sram_layout) * 4;
  }
  const tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(pConfig.type, pConfig.I);
  return (cTxs.qs + cTxs.dm + cTxs.sc) * 4;
}

// ------------------------------------------------------------

#include "mmq-load-tiles.cuh"
#include "mmq-vec-dot.cuh"

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void
ggml_cuda_mmq_write_back_dp4a(const float *__restrict__ pSum, const int32_t *__restrict__ pIdsDst,
                              float *__restrict__ pDst, const float *__restrict__ pYScale, const int pStride,
                              const int pIMax, const int pJMax) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  const bool cYScaleUsed = pYScale != nullptr;

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
    const int cJ = vJ0 + threadIdx.y;

    if (cJ > pJMax) {
      return;
    }

#pragma unroll
    for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
      const int i = vI0 + threadIdx.x;

      if (fallback && i > pIMax) {
        continue;
      }

      if constexpr (type == GGML_TYPE_NVFP4) {
        if (cYScaleUsed) {
          pDst[pIdsDst[cJ] * pStride + i] = pYScale[cJ] * pSum[(vJ0 / cNwarps) * (I / cWarpSize) + vI0 / cWarpSize];
        } else {
          pDst[pIdsDst[cJ] * pStride + i] = pSum[(vJ0 / cNwarps) * (I / cWarpSize) + vI0 / cWarpSize];
        }
      } else {
        pDst[pIdsDst[cJ] * pStride + i] = pSum[(vJ0 / cNwarps) * (I / cWarpSize) + vI0 / cWarpSize];
        GGML_UNUSED(cYScaleUsed);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void
ggml_cuda_mmq_write_back_mma(const float *__restrict__ pSum, const int *__restrict__ pIdsDst, float *__restrict__ pDst,
                             const float *__restrict__ pYScale, const int pStride, const int pIMax, const int pJMax) {

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
#else
  typedef tile<16, 8, int> tile_C;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  const int cI0 = (threadIdx.y / cNtx) * (cNtx * tile_C::I);

  const bool cYScaleUsed = pYScale != nullptr;

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
      for (int vL = 0; vL < tile_C::ne; ++vL) {
        const int cJ = vJ0 + (threadIdx.y % cNtx) * tile_C::J + tile_C::get_j(vL);

        if (cJ > pJMax) {
          continue;
        }

        const int i = cI0 + vN * tile_C::I + tile_C::get_i(vL);

        if (fallback && i > pIMax) {
          continue;
        }

        if constexpr (type == GGML_TYPE_NVFP4) {
          if (cYScaleUsed) {
            pDst[pIdsDst[cJ] * pStride + i] = pYScale[cJ] * pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL];
          } else {
            pDst[pIdsDst[cJ] * pStride + i] = pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL];
          }
        } else {
          pDst[pIdsDst[cJ] * pStride + i] = pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL];
          GGML_UNUSED(cYScaleUsed);
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------------------------------------------

// TODO remove this struct and use ggml_cuda_mmq_sram_layout instead.
struct ggml_cuda_mmq_util_funcs {
  int vdr;
  ggml_cuda_mmq_load_tiles_t load_tiles;
  ggml_cuda_mmq_vec_dot_t vec_dot;
  ggml_cuda_mmq_write_back_t write_back;

  constexpr __host__ __device__ ggml_cuda_mmq_util_funcs(int pVdr, ggml_cuda_mmq_load_tiles_t pLoadTiles,
                                                         ggml_cuda_mmq_vec_dot_t pVecDot,
                                                         ggml_cuda_mmq_write_back_t pWriteBack)
      : vdr(pVdr), load_tiles(pLoadTiles), vec_dot(pVecDot), write_back(pWriteBack) {}
};

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_funcs() {
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);

  if (!ggml_cuda_mmq_get_config(type, J, fallback).use_mma_data_layout()) {
    switch (type) {
    case GGML_TYPE_Q1_0:
      return ggml_cuda_mmq_util_funcs(VDR_Q1_0_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q2_0:
      return ggml_cuda_mmq_util_funcs(VDR_Q2_0_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q4_0:
      return ggml_cuda_mmq_util_funcs(VDR_Q4_0_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q4_1:
      return ggml_cuda_mmq_util_funcs(VDR_Q4_1_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q5_0:
      return ggml_cuda_mmq_util_funcs(VDR_Q5_0_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q5_1:
      return ggml_cuda_mmq_util_funcs(VDR_Q5_1_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q8_0:
      return ggml_cuda_mmq_util_funcs(VDR_Q8_0_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
      // ---------------------------------------------------------------------------------------------
    case GGML_TYPE_Q2_K:
      return ggml_cuda_mmq_util_funcs(VDR_Q2_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q3_K:
      return ggml_cuda_mmq_util_funcs(VDR_Q3_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q4_K:
      return ggml_cuda_mmq_util_funcs(VDR_Q4_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q5_K:
      return ggml_cuda_mmq_util_funcs(VDR_Q5_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_Q6_K:
      return ggml_cuda_mmq_util_funcs(VDR_Q6_K_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
      // ---------------------------------------------------------------------------------------------
    case GGML_TYPE_IQ1_S:
      return ggml_cuda_mmq_util_funcs(VDR_IQ1_S_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ2_XXS:
      return ggml_cuda_mmq_util_funcs(VDR_IQ2_XXS_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ2_XS:
      return ggml_cuda_mmq_util_funcs(VDR_IQ2_XS_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ2_S:
      return ggml_cuda_mmq_util_funcs(VDR_IQ2_S_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ3_XXS:
      return ggml_cuda_mmq_util_funcs(VDR_IQ3_XXS_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ3_S:
      return ggml_cuda_mmq_util_funcs(VDR_IQ3_S_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ4_XS:
      return ggml_cuda_mmq_util_funcs(VDR_IQ4_XS_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_IQ4_NL:
      return ggml_cuda_mmq_util_funcs(VDR_IQ4_NL_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
      // ---------------------------------------------------------------------------------------------
    case GGML_TYPE_MXFP4:
      return ggml_cuda_mmq_util_funcs(VDR_MXFP4_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    case GGML_TYPE_NVFP4:
      return ggml_cuda_mmq_util_funcs(VDR_NVFP4_Q8_1_MMQ, ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback>,
                                      ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                                      ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
    default:
      return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
    }
  }

  // ---------------------------------------------------------------------------------------------

#ifdef BLACKWELL_MMA_AVAILABLE
  switch (type) {
  case GGML_TYPE_MXFP4:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_mxfp4_fp4<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_NVFP4:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  default:
    break;
  }
#endif // BLACKWELL_MMA_AVAILABLE

  // ---------------------------------------------------------------------------------------------

  switch (type) {
  case GGML_TYPE_Q1_0:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q2_0:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q4_0:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_DS4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q4_1:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q5_0:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q5_1:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q8_0:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
    // ---------------------------------------------------------------------------------------------
  case GGML_TYPE_Q2_K:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q3_K:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q4_K:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q5_K:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_Q6_K:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
    // ---------------------------------------------------------------------------------------------
  case GGML_TYPE_IQ1_S:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ2_XXS:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ2_XS:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ2_S:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ3_XXS:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ3_S:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ4_XS:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_IQ4_NL:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
    // ---------------------------------------------------------------------------------------------
  case GGML_TYPE_MXFP4:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  case GGML_TYPE_NVFP4:
    return ggml_cuda_mmq_util_funcs(-1, ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback>,
                                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
  default:
    return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
  }
}

template <ggml_type type, int J, bool fallback> static constexpr __device__ int ggml_cuda_mmq_get_vdr() {
  return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vdr;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_load_tiles_t ggml_cuda_mmq_get_load_tiles() {
  return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().load_tiles;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_vec_dot_t ggml_cuda_mmq_get_vec_dot() {
  return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vec_dot;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_write_back_t ggml_cuda_mmq_get_write_back() {
  return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().write_back;
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void
mul_mat_q_process_tile(const char *__restrict__ pX, const int pOffsetX, const int *__restrict__ pY,
                       const int *__restrict__ pIdsDst, float *__restrict__ pDst, float *__restrict__ pTmpFixup,
                       const float *__restrict__ pYScale, const int pStrideRowX, const int pNcolsY,
                       const int pStrideColDst, const int pTileXMaxI, const int pTileYMaxJ, const int pKb0Start,
                       const int pKb0Stop) {

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr ggml_cuda_mmq_load_tiles_t cLoadTiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
  constexpr ggml_cuda_mmq_vec_dot_t cVecDot = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();
  constexpr ggml_cuda_mmq_write_back_t cWriteBack = ggml_cuda_mmq_get_write_back<type, J, fallback>();

  extern __shared__ int vDataMulMatQ[];
  int *vTileY = vDataMulMatQ + J;
  int *vTileX = vTileY + GGML_PAD(J * MMQ_TILE_Y_K, cNwarps * cWarpSize);

#if defined(BLACKWELL_MMA_AVAILABLE)
  // FP4 tile stores 8 blocks
  constexpr int cNeBlock = (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) ? QK_FP4_MMQ : QK8_1_MMQ;
#else
  constexpr int cNeBlock = QK8_1_MMQ;
#endif // defined(BLACKWELL_MMA_AVAILABLE)

  constexpr int cIterK = ggml_cuda_mmq_get_K_vram(type, J, fallback);
  constexpr int cBlocksPerIter = cIterK / qk;

  float vSum[J * cI / (cNwarps * cWarpSize)] = {0.0f};

  constexpr int cSz = sizeof(block_q8_1_mmq) / sizeof(int);

  for (int vKb0 = pKb0Start; vKb0 < pKb0Stop; vKb0 += cBlocksPerIter) {
    cLoadTiles(pX, vTileX, pOffsetX + vKb0, pTileXMaxI, pStrideRowX);
    {
      const int *vBy0 = pY + pNcolsY * (vKb0 * qk / cNeBlock) * cSz;
#pragma unroll
      for (int vL0 = 0; vL0 < J * MMQ_TILE_Y_K; vL0 += cNwarps * cWarpSize) {
        int vL = vL0 + threadIdx.y * cWarpSize + threadIdx.x;

        vTileY[vL] = vBy0[vL];
      }
    }

    __syncthreads();

    cVecDot(vTileX, vTileY, vSum, 0);

    __syncthreads();

    {
      const int *vBy0 = pY + pNcolsY * ((vKb0 * qk / cNeBlock) * cSz + cSz);
#pragma unroll
      for (int vL0 = 0; vL0 < J * MMQ_TILE_Y_K; vL0 += cNwarps * cWarpSize) {
        int vL = vL0 + threadIdx.y * cWarpSize + threadIdx.x;

        vTileY[vL] = vBy0[vL];
      }
    }

    __syncthreads();

    cVecDot(vTileX, vTileY, vSum, MMQ_TILE_NE_K);

    __syncthreads();
  }

  if (fixup) {
    cWriteBack(vSum, pIdsDst, pTmpFixup + blockIdx.x * (J * cI), pYScale, cI, cI, J);
  } else {
    cWriteBack(vSum, pIdsDst, pDst, pYScale, pStrideColDst, pTileXMaxI, pTileYMaxJ);
  }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback),
                  ggml_cuda_mmq_get_occupancy(type, J, fallback)) static __global__
    void mul_mat_q(const char *__restrict__ pX, const int *__restrict__ pY, const int32_t *__restrict__ pIdsDst,
                   const int32_t *__restrict__ pExpertBounds, float *__restrict__ pDst, float *__restrict__ pTmpFixup,
                   const float *__restrict__ pYScale, const uint3 pBlocksPerNe00, const int pNrowsX,
                   const int pNcolsDst, const int pStrideRowX, const int pNcolsY, const int pStrideColDst,
                   const uint3 pChannelRatio, const uint3 pNchannelsY, const int pStrideChannelX,
                   const int pStrideChannelY, const int pStrideChannelDst, const uint3 pSampleRatio,
                   const uint3 pNsamplesY, const int pStrideSampleX, const int pStrideSampleY,
                   const int pStrideSampleDst, const uint3 pNtx) {

  // Skip unused template specializations for faster compilation:
  if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
    NO_DEVICE_CODE;
    return;
  }

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);

  const uint32_t cNty = (pNrowsX + cI - 1) / cI; // Number of tiles y

  // Initialize the ids for writing back data with just the index.
  // For regular matrix multiplications this is never changed.
  // For MoE the correct indices are loaded from ids_dst.
  extern __shared__ int vIdsDstShared[]; // Stored at beginning of shared memory.
#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps * cWarpSize) {
    const int cJ = vJ0 + threadIdx.y * cWarpSize + threadIdx.x;

    if (vJ0 + cNwarps * cWarpSize > J && cJ >= J) {
      break;
    }

    vIdsDstShared[cJ] = cJ;
  }
  __syncthreads();

  if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
    const uint2 cTmp2 = fast_div_modulo(blockIdx.z, pNchannelsY);
    const int cWt = cTmp2.x;
    const int cZt = cTmp2.y;
    const int cJt = blockIdx.y;
    const int cIt = blockIdx.x;

    // Defaults for regular matrix multiplication:
    int vColLow = 0;
    int vColHigh = pNcolsDst;
    int vColDiff = pNcolsDst;
    int vOffsetY = cWt * pStrideSampleY + cZt * pStrideChannelY;
    int vOffsetDst = cWt * pStrideSampleDst + cZt * pStrideChannelDst + cJt * J * pStrideColDst;
    int vOffsetYScale;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vOffsetYScale = cWt * pNchannelsY.z * pNcolsY + cZt * pNcolsY;
    } else {
      GGML_UNUSED(vOffsetYScale);
    }

    if (pIdsDst) {
      vColLow = pExpertBounds[cZt + 0];
      vColHigh = pExpertBounds[cZt + 1];
      vColDiff = vColHigh - vColLow;

      vOffsetY = 0;
      vOffsetDst = 0;
      if constexpr (type == GGML_TYPE_NVFP4) {
        vOffsetYScale = 0;
      }

      if (cJt * J >= vColDiff) {
        return;
      }

      // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
      for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps * cWarpSize) {
        const int cJ = vJ0 + threadIdx.y * cWarpSize + threadIdx.x;

        if (vJ0 + cNwarps * cWarpSize > J && cJ >= J) {
          break;
        }

        vIdsDstShared[cJ] = pIdsDst[vColLow + cJt * J + cJ];
      }
      __syncthreads();
    }

    vOffsetY += (vColLow + cJt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    vOffsetDst += cIt * cI;
    const float *vYScaleTile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vOffsetYScale += vColLow + cJt * J;
      vYScaleTile = pYScale ? pYScale + vOffsetYScale : nullptr;
    }

    const int cTileXMaxI = pNrowsX - cIt * cI - 1;
    const int cTileYMaxJ = vColDiff - cJt * J - 1;

    const int cOffsetX = fastdiv(cWt, pSampleRatio) * pStrideSampleX + fastdiv(cZt, pChannelRatio) * pStrideChannelX +
                         cIt * cI * pStrideRowX;

    constexpr bool cFixup = false;
    mul_mat_q_process_tile<type, J, fallback, cFixup>(pX, cOffsetX, pY + vOffsetY, vIdsDstShared, pDst + vOffsetDst,
                                                      pTmpFixup, vYScaleTile, pStrideRowX, pNcolsY, pStrideColDst,
                                                      cTileXMaxI, cTileYMaxJ, 0, pBlocksPerNe00.z);
    return;
  }

  constexpr int cIterK = ggml_cuda_mmq_get_K_vram(type, J, fallback);
  constexpr int cBlocksPerIter = cIterK / qk;

  // kbc == k block continuous, current index in continuous ijk space.
  int vKbc = int64_t(blockIdx.x) * (pNsamplesY.z * pNchannelsY.z * pNtx.z * cNty * pBlocksPerNe00.z) / gridDim.x;
  int vKbcStop =
      int64_t(blockIdx.x + 1) * (pNsamplesY.z * pNchannelsY.z * pNtx.z * cNty * pBlocksPerNe00.z) / gridDim.x;

  vKbc -= fastmodulo(vKbc, pBlocksPerNe00) % cBlocksPerIter;
  vKbcStop -= fastmodulo(vKbcStop, pBlocksPerNe00) % cBlocksPerIter;

  // kb0 == k index when doing the matrix multiplication for an output tile.
  int vKb0Start = fastmodulo(vKbc, pBlocksPerNe00);
  int vKb0Stop = min(pBlocksPerNe00.z, uint32_t(vKb0Start + vKbcStop - vKbc));
  while (vKbc < vKbcStop && vKb0Stop == int(pBlocksPerNe00.z)) {
    int vTmp = fastdiv(vKbc, pBlocksPerNe00);
    uint2 vTmp2 = fast_div_modulo(vTmp, pNtx);
    const int cJt = vTmp2.y;
    vTmp = vTmp2.x;
    vTmp2 = fast_div_modulo(vTmp, pNchannelsY);
    const int cZt = vTmp2.y;
    vTmp = vTmp2.x;
    vTmp2 = fast_div_modulo(vTmp, pNsamplesY);
    const int cWt = vTmp2.y;
    const int cIt = vTmp2.x;

    // Defaults for regular matrix multiplication:
    int vColLow = 0;
    int vColHigh = pNcolsDst;
    int vColDiff = pNcolsDst;
    int vOffsetY = cWt * pStrideSampleY + cZt * pStrideChannelY;
    int vOffsetDst = cWt * pStrideSampleDst + cZt * pStrideChannelDst + cJt * J * pStrideColDst;
    int vOffsetYScale;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vOffsetYScale = cWt * pNchannelsY.z * pNcolsY + cZt * pNcolsY;
    } else {
      GGML_UNUSED(vOffsetYScale);
    }

    if (pIdsDst) {
      vColLow = pExpertBounds[cZt + 0];
      vColHigh = pExpertBounds[cZt + 1];
      vColDiff = vColHigh - vColLow;

      vOffsetY = 0;
      vOffsetDst = 0;
      if constexpr (type == GGML_TYPE_NVFP4) {
        vOffsetYScale = 0;
      }

      if (cJt * J >= vColDiff) {
        vKbc += pBlocksPerNe00.z;
        vKbc -= fastmodulo(vKbc, pBlocksPerNe00);

        vKb0Start = 0;
        vKb0Stop = min(pBlocksPerNe00.z, uint32_t(vKbcStop - vKbc));

        continue;
      }

      __syncthreads();
#pragma unroll
      for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps * cWarpSize) {
        const int cJ = vJ0 + threadIdx.y * cWarpSize + threadIdx.x;

        if (vJ0 + cNwarps * cWarpSize > J && cJ >= J) {
          break;
        }

        vIdsDstShared[cJ] = pIdsDst[vColLow + cJt * J + cJ];
      }
      __syncthreads();
    }

    vOffsetY += (vColLow + cJt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    vOffsetDst += cIt * cI;
    const float *vYScaleTile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vOffsetYScale += vColLow + cJt * J;
      vYScaleTile = pYScale ? pYScale + vOffsetYScale : nullptr;
    }

    const int cTileXMaxI = pNrowsX - cIt * cI - 1;
    const int cTileYMaxJ = vColDiff - cJt * J - 1;

    const int cOffsetX = fastdiv(cWt, pSampleRatio) * pStrideSampleX + fastdiv(cZt, pChannelRatio) * pStrideChannelX +
                         cIt * cI * pStrideRowX;

    constexpr bool cFixup =
        false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
    mul_mat_q_process_tile<type, J, fallback, cFixup>(pX, cOffsetX, pY + vOffsetY, vIdsDstShared, pDst + vOffsetDst,
                                                      pTmpFixup, vYScaleTile, pStrideRowX, pNcolsY, pStrideColDst,
                                                      cTileXMaxI, cTileYMaxJ, vKb0Start, vKb0Stop);

    vKbc += pBlocksPerNe00.z;
    vKbc -= fastmodulo(vKbc, pBlocksPerNe00);

    vKb0Start = 0;
    vKb0Stop = min(pBlocksPerNe00.z, uint32_t(vKbcStop - vKbc));
  }

  if (vKbc >= vKbcStop) {
    return;
  }

  int vTmp = fastdiv(vKbc, pBlocksPerNe00);
  uint2 vTmp2 = fast_div_modulo(vTmp, pNtx);
  const int cJt = vTmp2.y;
  vTmp = vTmp2.x;
  vTmp2 = fast_div_modulo(vTmp, pNchannelsY);
  const int cZt = vTmp2.y;
  vTmp = vTmp2.x;
  vTmp2 = fast_div_modulo(vTmp, pNsamplesY);
  const int cWt = vTmp2.y;
  const int cIt = vTmp2.x;

  // Defaults for regular matrix multiplication:
  int vColLow = 0;
  int vColHigh = pNcolsDst;
  int vColDiff = pNcolsDst;
  int vOffsetY = cWt * pStrideSampleY + cZt * pStrideChannelY;
  int vOffsetDst = cWt * pStrideSampleDst + cZt * pStrideChannelDst + cJt * J * pStrideColDst;
  int vOffsetYScale;
  if constexpr (type == GGML_TYPE_NVFP4) {
    vOffsetYScale = cWt * pNchannelsY.z * pNcolsY + cZt * pNcolsY;
  } else {
    GGML_UNUSED(vOffsetYScale);
  }

  if (pIdsDst) {
    vColLow = pExpertBounds[cZt + 0];
    vColHigh = pExpertBounds[cZt + 1];
    vColDiff = vColHigh - vColLow;

    vOffsetY = 0;
    vOffsetDst = 0;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vOffsetYScale = 0;
    }

    if (cJt * J >= vColDiff) {
      return;
    }

    // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
    __syncthreads();
#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps * cWarpSize) {
      const int cJ = vJ0 + threadIdx.y * cWarpSize + threadIdx.x;

      if (vJ0 + cNwarps * cWarpSize > J && cJ >= J) {
        break;
      }

      vIdsDstShared[cJ] = cJ;
    }
    __syncthreads();
  }

  vOffsetY += (vColLow + cJt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
  vOffsetDst += cIt * cI;
  const float *vYScaleTile = nullptr;
  if constexpr (type == GGML_TYPE_NVFP4) {
    vOffsetYScale += vColLow + cJt * J;
    vYScaleTile = pYScale ? pYScale + vOffsetYScale : nullptr;
  }

  const int cTileXMaxI = pNrowsX - cIt * cI - 1;
  const int cTileYMaxJ = vColDiff - cJt * J - 1;

  const int cOffsetX = fastdiv(cWt, pSampleRatio) * pStrideSampleX + fastdiv(cZt, pChannelRatio) * pStrideChannelX +
                       cIt * cI * pStrideRowX;

  constexpr bool cFixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
  mul_mat_q_process_tile<type, J, fallback, cFixup>(pX, cOffsetX, pY + vOffsetY, vIdsDstShared, pDst + vOffsetDst,
                                                    pTmpFixup, vYScaleTile, pStrideRowX, pNcolsY, pStrideColDst,
                                                    cTileXMaxI, cTileYMaxJ, vKb0Start, vKb0Stop);
}

template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2, 1) static __global__
    void mul_mat_q_stream_k_fixup(const int32_t *__restrict__ pIdsDst, const int32_t *__restrict__ pExpertBounds,
                                  float *__restrict__ pDst, float *__restrict__ pTmpLastTile,
                                  const uint3 pBlocksPerNe00, const int pNrowsX, const int pNcolsDst,
                                  const int pStrideColDst, const uint3 pNchannelsY, const int pStrideChannelDst,
                                  const uint3 pNsamplesY, const int pStrideSampleDst, const uint3 pNtx) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int cIterK = ggml_cuda_mmq_get_K_vram(type, J, fallback);
  constexpr int cBlocksPerIter = cIterK / qk;

  float vSum[J / cNwarps] = {0.0f};
  const int i = blockIdx.y * cWarpSize + threadIdx.x;

  const int cNty = (pNrowsX + I - 1) / I;

  const int cBidx0 = blockIdx.x;

  // kbc == k block continuous, current index in continuous ijk space.
  int vKbc0 = int64_t(blockIdx.x) * (pNsamplesY.z * pNchannelsY.z * pNtx.z * cNty * pBlocksPerNe00.z) / gridDim.x;
  int vKbc0Stop =
      int64_t(blockIdx.x + 1) * (pNsamplesY.z * pNchannelsY.z * pNtx.z * cNty * pBlocksPerNe00.z) / gridDim.x;

  vKbc0 -= fastmodulo(vKbc0, pBlocksPerNe00) % cBlocksPerIter;
  vKbc0Stop -= fastmodulo(vKbc0Stop, pBlocksPerNe00) % cBlocksPerIter;

  const bool cDidNotHaveAnyData = vKbc0 == vKbc0Stop;
  const bool cWroteBeginningOfTile = fastmodulo(vKbc0, pBlocksPerNe00) == 0;
  const bool cDidNotWriteLast = fastdiv(vKbc0, pBlocksPerNe00) == fastdiv(vKbc0Stop, pBlocksPerNe00) &&
                                fastmodulo(vKbc0Stop, pBlocksPerNe00) != 0;
  if (cDidNotHaveAnyData || cWroteBeginningOfTile || cDidNotWriteLast) {
    return;
  }

  bool vAnyFixup = false;

  // Iterate over previous blocks and sum up partial sums written to fixup buffer.
  // All CUDA blocks that get here must have a previous block that needs a fixup.
  int vBidx = cBidx0 - 1;
  int vKbcStop = vKbc0;
  while (true) {
    int vKbc = int64_t(vBidx) * (pNsamplesY.z * pNchannelsY.z * pNtx.z * cNty * pBlocksPerNe00.z) / gridDim.x;
    vKbc -= fastmodulo(vKbc, pBlocksPerNe00) % cBlocksPerIter;

    if (vKbc == vKbcStop) { // Did not have any data.
      vBidx--;
      vKbcStop = vKbc;
      continue;
    }

    vAnyFixup = true;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

      vSum[vJ0 / cNwarps] += pTmpLastTile[vBidx * (J * I) + cJ * I + i];
    }

    // If this block started in a previous tile we are done and don't need to combine additional partial results.
    if (fastmodulo(vKbc, pBlocksPerNe00) == 0 || fastdiv(vKbc, pBlocksPerNe00) < fastdiv(vKbc0, pBlocksPerNe00)) {
      break;
    }
    vBidx--;
    vKbcStop = vKbc;
  }

  if (!vAnyFixup) {
    return;
  }

  int vTmp = fastdiv(vKbc0, pBlocksPerNe00);
  uint2 vTmp2 = fast_div_modulo(vTmp, pNtx);
  const int cJt = vTmp2.y;
  vTmp = vTmp2.x;
  vTmp2 = fast_div_modulo(vTmp, pNchannelsY);
  const int cZt = vTmp2.y;
  vTmp = vTmp2.x;
  vTmp2 = fast_div_modulo(vTmp, pNsamplesY);
  const int cWt = vTmp2.y;
  const int cIt = vTmp2.x;

  if (!pIdsDst) {
    const int cOffsetDst = cWt * pStrideSampleDst + cZt * pStrideChannelDst + cJt * J * pStrideColDst + cIt * I;
    pDst += cOffsetDst;

    const int cIMax = pNrowsX - cIt * I - 1;
    const int cJMax = pNcolsDst - cJt * J - 1;
    if (fallback && i > cIMax) {
      return;
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

      if (cJ > cJMax) {
        return;
      }

      pDst[cJ * pStrideColDst + i] += vSum[vJ0 / cNwarps];
    }
    return;
  }

  __shared__ int vIdsDstShared[J];
  const int cColLow = pExpertBounds[cZt + 0];
  const int cColHigh = pExpertBounds[cZt + 1];
  const int cColDiff = cColHigh - cColLow;

  for (int vJ = threadIdx.y * cWarpSize + threadIdx.x; vJ < J; vJ += cNwarps * cWarpSize) {
    vIdsDstShared[vJ] = pIdsDst[cColLow + cJt * J + vJ];
  }
  __syncthreads();

  const int cOffsetDst = cIt * I;
  pDst += cOffsetDst;

  const int cIMax = pNrowsX - cIt * I - 1;
  const int cJMax = cColDiff - cJt * J - 1;
  if (fallback && i > cIMax) {
    return;
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
    const int cJ = vJ0 + threadIdx.y;

    if (cJ > cJMax) {
      return;
    }

    pDst[vIdsDstShared[cJ] * pStrideColDst + i] += vSum[vJ0 / cNwarps];
  }
}

struct mmq_args {
  const char *x;
  ggml_type type_x;
  const int *y;
  const int32_t *ids_dst;
  const int32_t *expert_bounds;
  float *dst;
  const float *y_scale;
  int64_t ncols_x;
  int64_t nrows_x;
  int64_t ncols_dst;
  int64_t stride_row_x;
  int64_t ncols_y;
  int64_t nrows_dst;
  int64_t nchannels_x;
  int64_t nchannels_y;
  int64_t stride_channel_x;
  int64_t stride_channel_y;
  int64_t stride_channel_dst;
  int64_t nsamples_x;
  int64_t nsamples_y;
  int64_t stride_sample_x;
  int64_t stride_sample_y;
  int64_t stride_sample_dst;
  int64_t ncols_max;
  int64_t ncols_opt; // value to optimize the tile size against, launch grid still uses ncols_max
};

static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config &pConfig, const int pCc) {
  const size_t cNbsIds = pConfig.J * sizeof(int);
  const size_t cNbsX = ggml_cuda_mmq_get_nbytes_shared_x(pConfig, pCc);
  const size_t cNbsY = pConfig.J * (sizeof(block_q8_1_mmq));
  return cNbsIds + cNbsX + GGML_PAD(cNbsY, pConfig.nthreads * sizeof(int));
}

template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q(ggml_backend_cuda_context &pCtx, const mmq_args &pArgs, cudaStream_t pStream) {
  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;
  const int cNsm = ggml_cuda_info().devices[cId].nsm;
  const int cWarpSize = ggml_cuda_info().devices[cId].warp_size;

  const ggml_cuda_mmq_config cConfig = ggml_cuda_mmq_get_config(type, J, fallback, cCc);
  GGML_ASSERT(cConfig.nthreads % cWarpSize == 0);
  const int cNwarps = cConfig.nthreads / cWarpSize;
  const int cNbytesShared = mmq_get_nbytes_shared(cConfig, cCc);

  const dim3 cBlockDims(cWarpSize, cNwarps, 1);

  CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false>), cNbytesShared);
  CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, true>), cNbytesShared);

  const int cNty = (pArgs.nrows_x + cConfig.I - 1) / cConfig.I;
  const int cNtx = (pArgs.ncols_max + cConfig.J - 1) / cConfig.J;
  const int cNtzw = pArgs.nchannels_y * pArgs.nsamples_y;
  const dim3 cBlockNumsXyTiling(cNty, cNtx, cNtzw);

  GGML_ASSERT(pArgs.nchannels_y % pArgs.nchannels_x == 0);
  GGML_ASSERT(pArgs.nsamples_y % pArgs.nsamples_x == 0);
  const int cChannelRatio = pArgs.nchannels_y / pArgs.nchannels_x;
  const int cSampleRatio = pArgs.nsamples_y / pArgs.nsamples_x;

  const uint3 cBlocksPerNe00Fd = init_fastdiv_values(pArgs.ncols_x / ggml_cuda_type_traits<type>::qk);
  const uint3 cNtxFd = init_fastdiv_values(cNtx);
  const uint3 cNchannelsYFd = init_fastdiv_values(pArgs.nchannels_y);
  const uint3 cNsamplesYFd = init_fastdiv_values(pArgs.nsamples_y);
  const uint3 cChannelRatioFd = init_fastdiv_values(cChannelRatio);
  const uint3 cSampleRatioFd = init_fastdiv_values(cSampleRatio);

  if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cCc)) {
    mul_mat_q<type, J, fallback><<<cBlockNumsXyTiling, cBlockDims, cNbytesShared, pStream>>>(
        pArgs.x, pArgs.y, pArgs.ids_dst, pArgs.expert_bounds, pArgs.dst, nullptr, pArgs.y_scale, cBlocksPerNe00Fd,
        pArgs.nrows_x, pArgs.ncols_dst, pArgs.stride_row_x, pArgs.ncols_y, pArgs.nrows_dst, cChannelRatioFd,
        cNchannelsYFd, pArgs.stride_channel_x, pArgs.stride_channel_y, pArgs.stride_channel_dst, cSampleRatioFd,
        cNsamplesYFd, pArgs.stride_sample_x, pArgs.stride_sample_y, pArgs.stride_sample_dst, cNtxFd);
    return;
  }

  // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the
  // number of tiles. This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more
  // important.
  const int cNtilesDst = cNtx * cNty * cNtzw;
  const int cTilesNwaves = (cNtilesDst + cNsm - 1) / cNsm;
  const int cTilesEfficiencyPercent = 100 * cNtilesDst / (cNsm * cTilesNwaves);
  const dim3 cBlockNumsStreamK(GGML_CUDA_CC_IS_NVIDIA(cCc) && cTilesEfficiencyPercent >= 90 ? cNtilesDst : cNsm, 1, 1);

  GGML_ASSERT(cNtilesDst * cBlocksPerNe00Fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

  const bool cFixupNeeded = cNtilesDst % cBlockNumsStreamK.x != 0;

  ggml_cuda_pool &vPool = pCtx.pool(cId);
  ggml_cuda_pool_alloc<float> vTmpFixup(vPool);
  if (cFixupNeeded) {
    vTmpFixup.alloc(cBlockNumsStreamK.x * cConfig.J * cConfig.I);
  }

  const dim3 cBlockNumsFixup(cBlockNumsStreamK.x, cConfig.I / cWarpSize, 1);
  const dim3 cBlockDimsFixup(cBlockDims.x, cBlockDims.y / 2, cBlockDims.z);

  mul_mat_q<type, J, fallback><<<cBlockNumsStreamK, cBlockDims, cNbytesShared, pStream>>>(
      pArgs.x, pArgs.y, pArgs.ids_dst, pArgs.expert_bounds, pArgs.dst, vTmpFixup.ptr, pArgs.y_scale, cBlocksPerNe00Fd,
      pArgs.nrows_x, pArgs.ncols_dst, pArgs.stride_row_x, pArgs.ncols_y, pArgs.nrows_dst, cChannelRatioFd,
      cNchannelsYFd, pArgs.stride_channel_x, pArgs.stride_channel_y, pArgs.stride_channel_dst, cSampleRatioFd,
      cNsamplesYFd, pArgs.stride_sample_x, pArgs.stride_sample_y, pArgs.stride_sample_dst, cNtxFd);

  if (!cFixupNeeded) {
    return;
  }

  CUDA_CHECK(cudaGetLastError());
  mul_mat_q_stream_k_fixup<type, J, fallback><<<cBlockNumsFixup, cBlockDimsFixup, 0, pStream>>>(
      pArgs.ids_dst, pArgs.expert_bounds, pArgs.dst, vTmpFixup.ptr, cBlocksPerNe00Fd, pArgs.nrows_x, pArgs.ncols_dst,
      pArgs.nrows_dst, cNchannelsYFd, pArgs.stride_channel_dst, cNsamplesYFd, pArgs.stride_sample_dst, cNtxFd);
}

template <ggml_type type, bool fallback>
void mul_mat_q_switch_J(ggml_backend_cuda_context &pCtx, const mmq_args &pArgs, cudaStream_t pStream) {
  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;

  int vJBest = 0;
  int vNtilesJBest = INT_MAX;

  for (int vJ = 8; vJ <= 128 && vNtilesJBest > 1; vJ += 8) {
    const ggml_cuda_mmq_config cConfig = ggml_cuda_mmq_get_config(type, vJ, fallback, cCc);
    if (cConfig.type == GGML_TYPE_COUNT) {
      continue;
    }

    if (mmq_get_nbytes_shared(cConfig, cCc) > cSmpbo) {
      continue;
    }

    const int cNtilesX = (pArgs.ncols_opt + cConfig.J - 1) / cConfig.J;

    if (cNtilesX < vNtilesJBest) {
      vJBest = vJ;
      vNtilesJBest = cNtilesX;
    }
  }

  switch (vJBest) {
  case 8:
    launch_mul_mat_q<type, 8, fallback>(pCtx, pArgs, pStream);
    break;
  case 16:
    launch_mul_mat_q<type, 16, fallback>(pCtx, pArgs, pStream);
    break;
  case 24:
    launch_mul_mat_q<type, 24, fallback>(pCtx, pArgs, pStream);
    break;
  case 32:
    launch_mul_mat_q<type, 32, fallback>(pCtx, pArgs, pStream);
    break;
  case 40:
    launch_mul_mat_q<type, 40, fallback>(pCtx, pArgs, pStream);
    break;
  case 48:
    launch_mul_mat_q<type, 48, fallback>(pCtx, pArgs, pStream);
    break;
  case 56:
    launch_mul_mat_q<type, 56, fallback>(pCtx, pArgs, pStream);
    break;
  case 64:
    launch_mul_mat_q<type, 64, fallback>(pCtx, pArgs, pStream);
    break;
  case 72:
    launch_mul_mat_q<type, 72, fallback>(pCtx, pArgs, pStream);
    break;
  case 80:
    launch_mul_mat_q<type, 80, fallback>(pCtx, pArgs, pStream);
    break;
  case 88:
    launch_mul_mat_q<type, 88, fallback>(pCtx, pArgs, pStream);
    break;
  case 96:
    launch_mul_mat_q<type, 96, fallback>(pCtx, pArgs, pStream);
    break;
  case 104:
    launch_mul_mat_q<type, 104, fallback>(pCtx, pArgs, pStream);
    break;
  case 112:
    launch_mul_mat_q<type, 112, fallback>(pCtx, pArgs, pStream);
    break;
  case 120:
    launch_mul_mat_q<type, 120, fallback>(pCtx, pArgs, pStream);
    break;
  case 128:
    launch_mul_mat_q<type, 128, fallback>(pCtx, pArgs, pStream);
    break;
  default:
    fprintf(stderr, "J_best=%d\n", vJBest);
    GGML_ABORT("fatal error");
    break;
  }
}

template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context &pCtx, const mmq_args &pArgs, cudaStream_t pStream) {
  if (pArgs.nrows_x % 128 == 0) {
    constexpr bool cFallback = false;
    mul_mat_q_switch_J<type, cFallback>(pCtx, pArgs, pStream);
  } else {
    constexpr bool cFallback = true;
    mul_mat_q_switch_J<type, cFallback>(pCtx, pArgs, pStream);
  }
}

#define DECL_MMQ_CASE(type)                                                                                            \
  template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args &args, cudaStream_t stream)

extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE(GGML_TYPE_NVFP4);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                         const ggml_tensor *pIds, ggml_tensor *pDst);

bool ggml_cuda_should_use_mmq(enum ggml_type pType, int pCc, int64_t pNe11, int64_t pNExperts);
