#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
  switch (type) {
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
    return true;
  default:
    return false;
  }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void *p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ pVbq, const block_q8_1 *__restrict__ pBq81, const int &kbx,
                                  const int &pIqs);

static constexpr __device__ vec_dot_q_cuda_t fGetVecDotQCuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return vec_dot_q1_0_q8_1;
  case GGML_TYPE_Q2_0:
    return vec_dot_q2_0_q8_1;
  case GGML_TYPE_Q4_0:
    return vec_dot_q4_0_q8_1;
  case GGML_TYPE_Q4_1:
    return vec_dot_q4_1_q8_1;
  case GGML_TYPE_Q5_0:
    return vec_dot_q5_0_q8_1;
  case GGML_TYPE_Q5_1:
    return vec_dot_q5_1_q8_1;
  case GGML_TYPE_Q8_0:
    return vec_dot_q8_0_q8_1;
  case GGML_TYPE_MXFP4:
    return vec_dot_mxfp4_q8_1;
  case GGML_TYPE_NVFP4:
    return vec_dot_nvfp4_q8_1;
  case GGML_TYPE_Q2_K:
    return vec_dot_q2_K_q8_1;
  case GGML_TYPE_Q3_K:
    return vec_dot_q3_K_q8_1;
  case GGML_TYPE_Q4_K:
    return vec_dot_q4_K_q8_1;
  case GGML_TYPE_Q5_K:
    return vec_dot_q5_K_q8_1;
  case GGML_TYPE_Q6_K:
    return vec_dot_q6_K_q8_1;
  case GGML_TYPE_IQ2_XXS:
    return vec_dot_iq2_xxs_q8_1;
  case GGML_TYPE_IQ2_XS:
    return vec_dot_iq2_xs_q8_1;
  case GGML_TYPE_IQ2_S:
    return vec_dot_iq2_s_q8_1;
  case GGML_TYPE_IQ3_XXS:
    return vec_dot_iq3_xxs_q8_1;
  case GGML_TYPE_IQ1_S:
    return vec_dot_iq1_s_q8_1;
  case GGML_TYPE_IQ1_M:
    return vec_dot_iq1_m_q8_1;
  case GGML_TYPE_IQ4_NL:
    return vec_dot_iq4_nl_q8_1;
  case GGML_TYPE_IQ4_XS:
    return vec_dot_iq4_xs_q8_1;
  case GGML_TYPE_IQ3_S:
    return vec_dot_iq3_s_q8_1;
  default:
    return nullptr;
  }
}

static constexpr __host__ __device__ int fGetVdrMmvq(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return VDR_Q1_0_Q8_1_MMVQ;
  case GGML_TYPE_Q2_0:
    return VDR_Q2_0_Q8_1_MMVQ;
  case GGML_TYPE_Q4_0:
    return VDR_Q4_0_Q8_1_MMVQ;
  case GGML_TYPE_Q4_1:
    return VDR_Q4_1_Q8_1_MMVQ;
  case GGML_TYPE_Q5_0:
    return VDR_Q5_0_Q8_1_MMVQ;
  case GGML_TYPE_Q5_1:
    return VDR_Q5_1_Q8_1_MMVQ;
  case GGML_TYPE_Q8_0:
    return VDR_Q8_0_Q8_1_MMVQ;
  case GGML_TYPE_MXFP4:
    return VDR_MXFP4_Q8_1_MMVQ;
  case GGML_TYPE_NVFP4:
    return VDR_NVFP4_Q8_1_MMVQ;
  case GGML_TYPE_Q2_K:
    return VDR_Q2_K_Q8_1_MMVQ;
  case GGML_TYPE_Q3_K:
    return VDR_Q3_K_Q8_1_MMVQ;
  case GGML_TYPE_Q4_K:
    return VDR_Q4_K_Q8_1_MMVQ;
  case GGML_TYPE_Q5_K:
    return VDR_Q5_K_Q8_1_MMVQ;
  case GGML_TYPE_Q6_K:
    return VDR_Q6_K_Q8_1_MMVQ;
  case GGML_TYPE_IQ2_XXS:
    return VDR_IQ2_XXS_Q8_1_MMVQ;
  case GGML_TYPE_IQ2_XS:
    return VDR_IQ2_XS_Q8_1_MMVQ;
  case GGML_TYPE_IQ2_S:
    return VDR_IQ2_S_Q8_1_MMVQ;
  case GGML_TYPE_IQ3_XXS:
    return VDR_IQ3_XXS_Q8_1_MMVQ;
  case GGML_TYPE_IQ3_S:
    return VDR_IQ3_S_Q8_1_MMVQ;
  case GGML_TYPE_IQ4_NL:
    return VDR_IQ4_NL_Q8_1_MMVQ;
  case GGML_TYPE_IQ4_XS:
    return VDR_IQ4_XS_Q8_1_MMVQ;
  default:
    return 1;
  }
}

enum mmvq_parameter_table_id {
  MMVQ_PARAMETERS_GENERIC = 0,
  MMVQ_PARAMETERS_TURING,
  MMVQ_PARAMETERS_GCN,
  MMVQ_PARAMETERS_RDNA2,
  MMVQ_PARAMETERS_RDNA3_0,
  MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id fGetDeviceTableId() {
#if defined(RDNA4)
  return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
  return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
  return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
  return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
  return MMVQ_PARAMETERS_TURING;
#else
  return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id fGetDeviceTableId(int pCc) {
  if (GGML_CUDA_CC_IS_RDNA4(pCc)) {
    return MMVQ_PARAMETERS_RDNA4;
  }
  if (GGML_CUDA_CC_IS_RDNA3_0(pCc)) {
    return MMVQ_PARAMETERS_RDNA3_0;
  }
  if (GGML_CUDA_CC_IS_RDNA2(pCc) || GGML_CUDA_CC_IS_RDNA3_5(pCc)) {
    return MMVQ_PARAMETERS_RDNA2;
  }
  if (GGML_CUDA_CC_IS_GCN(pCc) || GGML_CUDA_CC_IS_CDNA(pCc)) {
    return MMVQ_PARAMETERS_GCN;
  }
  if (GGML_CUDA_CC_IS_NVIDIA(pCc) && ggml_cuda_highest_compiled_arch(pCc) >= GGML_CUDA_CC_VOLTA &&
      ggml_cuda_highest_compiled_arch(pCc) < GGML_CUDA_CC_AMPERE) {
    return MMVQ_PARAMETERS_TURING;
  }
  return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchPascalOlder(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ1_S:
    return 6;
  case GGML_TYPE_IQ1_M:
    return 6;
  case GGML_TYPE_IQ2_S:
    return 4;
  case GGML_TYPE_IQ2_XS:
    return 5;
  case GGML_TYPE_IQ2_XXS:
    return 5;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 4;
  case GGML_TYPE_IQ4_NL:
    return 6;
  case GGML_TYPE_IQ4_XS:
    return 5;
  case GGML_TYPE_MXFP4:
    return 4;
  case GGML_TYPE_NVFP4:
    return 4;
  case GGML_TYPE_Q2_K:
    return 4;
  case GGML_TYPE_Q3_K:
    return 4;
  case GGML_TYPE_Q4_0:
    return 6;
  case GGML_TYPE_Q4_1:
    return 6;
  case GGML_TYPE_Q4_K:
    return 5;
  case GGML_TYPE_Q5_0:
    return 6;
  case GGML_TYPE_Q5_1:
    return 6;
  case GGML_TYPE_Q5_K:
    return 5;
  case GGML_TYPE_Q6_K:
    return 4;
  case GGML_TYPE_Q8_0:
    return 4;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchTuringPlus(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ2_S:
    return 7;
  case GGML_TYPE_IQ3_S:
    return 6;
  case GGML_TYPE_IQ3_XXS:
    return 7;
  case GGML_TYPE_MXFP4:
    return 7;
  case GGML_TYPE_NVFP4:
    return 8;
  case GGML_TYPE_Q2_K:
    return 7;
  case GGML_TYPE_Q3_K:
    return 5;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchGcn(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ1_S:
    return 5;
  case GGML_TYPE_IQ1_M:
    return 5;
  case GGML_TYPE_IQ2_S:
    return 4;
  case GGML_TYPE_IQ2_XS:
    return 4;
  case GGML_TYPE_IQ2_XXS:
    return 4;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 4;
  case GGML_TYPE_IQ4_NL:
    return 6;
  case GGML_TYPE_IQ4_XS:
    return 4;
  case GGML_TYPE_Q2_K:
    return 4;
  case GGML_TYPE_Q3_K:
    return 4;
  case GGML_TYPE_Q4_0:
    return 5;
  case GGML_TYPE_Q4_1:
    return 5;
  case GGML_TYPE_Q4_K:
    return 4;
  case GGML_TYPE_Q5_K:
    return 4;
  case GGML_TYPE_Q6_K:
    return 4;
  case GGML_TYPE_Q8_0:
    return 4;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchCdna(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ2_S:
    return 5;
  case GGML_TYPE_IQ2_XS:
    return 5;
  case GGML_TYPE_IQ2_XXS:
    return 5;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 5;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchRdna1Rdna2(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ2_S:
    return 4;
  case GGML_TYPE_IQ2_XS:
    return 4;
  case GGML_TYPE_IQ2_XXS:
    return 4;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 4;
  case GGML_TYPE_Q2_K:
    return 7;
  case GGML_TYPE_Q3_K:
    return 4;
  case GGML_TYPE_Q4_K:
    return 5;
  case GGML_TYPE_Q5_K:
    return 6;
  case GGML_TYPE_Q6_K:
    return 5;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchRdna3(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ1_S:
    return 6;
  case GGML_TYPE_IQ1_M:
    return 6;
  case GGML_TYPE_IQ2_S:
    return 4;
  case GGML_TYPE_IQ2_XS:
    return 4;
  case GGML_TYPE_IQ2_XXS:
    return 4;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 4;
  case GGML_TYPE_IQ4_NL:
    return 6;
  case GGML_TYPE_IQ4_XS:
    return 6;
  case GGML_TYPE_Q4_K:
    return 4;
  case GGML_TYPE_Q5_K:
    return 4;
  case GGML_TYPE_Q6_K:
    return 4;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

static constexpr __host__ __device__ int fGetMmvqMmidMaxBatchRdna4(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_IQ1_S:
    return 7;
  case GGML_TYPE_IQ1_M:
    return 7;
  case GGML_TYPE_IQ2_S:
    return 4;
  case GGML_TYPE_IQ2_XS:
    return 4;
  case GGML_TYPE_IQ2_XXS:
    return 4;
  case GGML_TYPE_IQ3_S:
    return 4;
  case GGML_TYPE_IQ3_XXS:
    return 4;
  case GGML_TYPE_IQ4_NL:
    return 7;
  case GGML_TYPE_IQ4_XS:
    return 5;
  case GGML_TYPE_MXFP4:
    return 5;
  case GGML_TYPE_NVFP4:
    return 5;
  case GGML_TYPE_Q3_K:
    return 4;
  case GGML_TYPE_Q4_0:
    return 7;
  case GGML_TYPE_Q4_1:
    return 7;
  case GGML_TYPE_Q4_K:
    return 4;
  case GGML_TYPE_Q5_0:
    return 7;
  case GGML_TYPE_Q5_1:
    return 7;
  case GGML_TYPE_Q5_K:
    return 5;
  case GGML_TYPE_Q6_K:
    return 5;
  case GGML_TYPE_Q8_0:
    return 7;
  default:
    return MMVQ_MAX_BATCH_SIZE;
  }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type pType, int pCc) {
  // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
  if (GGML_CUDA_CC_IS_NVIDIA(pCc)) {
    if (pCc == GGML_CUDA_CC_VOLTA || pCc >= GGML_CUDA_CC_ADA_LOVELACE) {
      return MMVQ_MAX_BATCH_SIZE;
    }
    if (pCc >= GGML_CUDA_CC_TURING) {
      return fGetMmvqMmidMaxBatchTuringPlus(pType);
    }
    return fGetMmvqMmidMaxBatchPascalOlder(pType);
  }

  // AMD
  if (GGML_CUDA_CC_IS_AMD(pCc)) {
    if (GGML_CUDA_CC_IS_RDNA4(pCc)) {
      return fGetMmvqMmidMaxBatchRdna4(pType);
    }
    if (GGML_CUDA_CC_IS_RDNA3(pCc)) {
      return fGetMmvqMmidMaxBatchRdna3(pType);
    }
    if (GGML_CUDA_CC_IS_RDNA1(pCc) || GGML_CUDA_CC_IS_RDNA2(pCc)) {
      return fGetMmvqMmidMaxBatchRdna1Rdna2(pType);
    }
    if (GGML_CUDA_CC_IS_CDNA(pCc)) {
      return fGetMmvqMmidMaxBatchCdna(pType);
    }
    if (GGML_CUDA_CC_IS_GCN(pCc)) {
      return fGetMmvqMmidMaxBatchGcn(pType);
    }
  }
  return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type pType, int pCc, int64_t pNe11) {
  if (!ggml_is_quantized(pType)) {
    return false;
  }
  // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
  // Only list quant-types MMQ supports, others would fall back to cuBLAS.
  if (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc == GGML_CUDA_CC_ADA_LOVELACE) {
    switch (pType) { // tuned on RTX 4090
    case GGML_TYPE_Q2_K:
      return pNe11 <= 4;
    case GGML_TYPE_Q3_K:
      return pNe11 <= 6;
    default:
      return pNe11 <= MMVQ_MAX_BATCH_SIZE;
    }
  }
  if (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc == GGML_CUDA_CC_BLACKWELL) {
    switch (pType) { // tuned on RTX 5090
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
      return pNe11 <= 5;
    case GGML_TYPE_Q5_K:
      return pNe11 <= 6;
    case GGML_TYPE_Q6_K:
      return pNe11 <= 7;
    default:
      return pNe11 <= MMVQ_MAX_BATCH_SIZE;
    }
  }
  if (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc == GGML_CUDA_CC_DGX_SPARK) {
    switch (pType) { // tuned on DGX Spark GB10
    case GGML_TYPE_Q2_K:
      return pNe11 <= 6;
    default:
      return pNe11 <= MMVQ_MAX_BATCH_SIZE;
    }
  }
  if (GGML_CUDA_CC_IS_NVIDIA(pCc) && pCc == GGML_CUDA_CC_VOLTA) {
    switch (pType) {
    case GGML_TYPE_Q2_K:
      return pNe11 <= 4;
    case GGML_TYPE_Q3_K:
      return pNe11 <= 6;
    case GGML_TYPE_Q4_K:
      return pNe11 <= 5;
    case GGML_TYPE_Q5_K:
      return pNe11 <= 6;
    case GGML_TYPE_Q6_K:
      return pNe11 <= 7;
    default:
      return pNe11 <= MMVQ_MAX_BATCH_SIZE;
    }
  }
  if (GGML_CUDA_CC_IS_CDNA(pCc)) {
    if (GGML_CUDA_CC_IS_CDNA1(pCc)) {
      switch (pType) {
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
        return pNe11 <= 7;
      case GGML_TYPE_Q5_1:
        return pNe11 <= 7;
      case GGML_TYPE_Q8_0:
        return pNe11 <= 6;
      case GGML_TYPE_Q2_K:
        return pNe11 <= 4;
      case GGML_TYPE_Q3_K:
        return pNe11 <= 3;
      case GGML_TYPE_Q4_K:
        return pNe11 <= 2;
      case GGML_TYPE_Q5_K:
        return pNe11 <= 3;
      case GGML_TYPE_Q6_K:
        return pNe11 <= 4;
      case GGML_TYPE_IQ1_S:
        return pNe11 <= 5;
      case GGML_TYPE_IQ2_XXS:
      case GGML_TYPE_IQ3_S:
      case GGML_TYPE_IQ4_XS:
        return pNe11 <= 6;
      default:
        return pNe11 <= MMVQ_MAX_BATCH_SIZE;
      }
    }
    switch (pType) { // tuned for CDNA2
    case GGML_TYPE_Q2_K:
      return pNe11 <= 5;
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
      return pNe11 <= 3;
    case GGML_TYPE_Q6_K:
      return pNe11 <= 5;
    default:
      return pNe11 <= MMVQ_MAX_BATCH_SIZE;
    }
  }
  return pNe11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type> static constexpr __device__ int fGetMmvqMmidMaxBatchForDevice() {
#if defined(RDNA4)
  return fGetMmvqMmidMaxBatchRdna4(type);
#elif defined(RDNA3)
  return fGetMmvqMmidMaxBatchRdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
  return fGetMmvqMmidMaxBatchRdna1Rdna2(type);
#elif defined(CDNA)
  return fGetMmvqMmidMaxBatchCdna(type);
#elif defined(GCN)
  return fGetMmvqMmidMaxBatchGcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
  return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  return fGetMmvqMmidMaxBatchTuringPlus(type);
#else
  return fGetMmvqMmidMaxBatchPascalOlder(type);
#endif
}

static constexpr __host__ __device__ int fCalcNwarps(ggml_type pType, int pNcolsDst, mmvq_parameter_table_id pTableId) {
  if (pTableId == MMVQ_PARAMETERS_GENERIC) {
    switch (pNcolsDst) {
    case 1:
    case 2:
    case 3:
    case 4:
      return 4;
    case 5:
    case 6:
    case 7:
    case 8:
      return 2;
    default:
      return 1;
    }
  } else if (pTableId == MMVQ_PARAMETERS_GCN) {
    switch (pNcolsDst) {
    case 1:
    case 2:
    case 3:
    case 4:
      return 2;
    case 5:
    case 6:
    case 7:
    case 8:
    default:
      return 1;
    }
  }
  if (pTableId == MMVQ_PARAMETERS_RDNA4) {
    // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
    // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
    // pressure and lookup table contention at higher thread counts.
    if (pNcolsDst == 1) {
      switch (pType) {
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q5_0:
      case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q8_0:
      case GGML_TYPE_Q2_K:
      case GGML_TYPE_Q4_K:
      case GGML_TYPE_Q5_K:
      case GGML_TYPE_Q6_K:
      case GGML_TYPE_IQ4_NL:
      case GGML_TYPE_IQ4_XS:
        return 8;
      default:
        return 1;
      }
    }
    return 1;
  }
  if (pTableId == MMVQ_PARAMETERS_RDNA3_0) {
    // RDNA3 (W7900): stricter whitelist than RDNA4.
    // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
    if (pNcolsDst == 1) {
      switch (pType) {
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q5_0:
      case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q8_0:
        return 8;
      case GGML_TYPE_Q6_K:
        return 2;
      case GGML_TYPE_IQ4_NL:
        return 8;
      default:
        return 1;
      }
    }
    return 1;
  }
  if (pTableId == MMVQ_PARAMETERS_TURING) {
    if (pNcolsDst == 1) {
      switch (pType) {
      case GGML_TYPE_Q2_K:
      case GGML_TYPE_Q3_K:
      case GGML_TYPE_Q4_K:
      case GGML_TYPE_Q5_K:
      case GGML_TYPE_Q6_K:
        return 2;
      default:
        return 4;
      }
    }
    switch (pNcolsDst) {
    case 2:
    case 3:
    case 4:
      return 4;
    case 5:
    case 6:
    case 7:
    case 8:
      return 2;
    default:
      return 1;
    }
  }
  return 1;
}

static constexpr __host__ __device__ int fCalcRowsPerBlock(int pNcolsDst, int pTableId, bool pSmallK = false,
                                                           int pNwarps = 1) {
  if (pTableId == MMVQ_PARAMETERS_GENERIC || pTableId == MMVQ_PARAMETERS_GCN || pTableId == MMVQ_PARAMETERS_TURING) {
    switch (pNcolsDst) {
    case 1:
      return pSmallK ? pNwarps : 1;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
      return 2;
    default:
      return 1;
    }
  }
  return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(fCalcNwarps(type, ncols_dst, fGetDeviceTableId()) * ggml_cuda_get_physical_warp_size(),
                  1) static __global__
    void fMulMatVecQ(const void *pVxPtr, const void *pVyPtr, const int32_t *pIdsPtr,
                     const ggml_cuda_mm_fusion_args_device pFusion, float *pDstPtr, const uint32_t pNcolsX,
                     const uint3 pNchannelsY, const uint32_t pStrideRowX, const uint32_t pStrideColY,
                     const uint32_t pStrideColDst, const uint3 pChannelRatio, const uint32_t pStrideChannelX,
                     const uint32_t pStrideChannelY, const uint32_t pStrideChannelDst, const uint3 pSampleRatio,
                     const uint32_t pStrideSampleX, const uint32_t pStrideSampleY, const uint32_t pStrideSampleDst,
                     const uint32_t pIdsStride) {
  const void *GGML_CUDA_RESTRICT vVx = pVxPtr;
  const void *GGML_CUDA_RESTRICT vVy = pVyPtr;
  const int32_t *GGML_CUDA_RESTRICT vIds = pIdsPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;

  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  constexpr int cVdr = fGetVdrMmvq(type);
  constexpr mmvq_parameter_table_id cTableId = fGetDeviceTableId();
  constexpr int cNwarps = fCalcNwarps(type, ncols_dst, cTableId);
  constexpr int cRowsPerCudaBlock = fCalcRowsPerBlock(ncols_dst, cTableId, small_k, cNwarps);
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();

  constexpr vec_dot_q_cuda_t cVecDotQCuda = fGetVecDotQCuda(type);

  const int cTid = cWarpSize * threadIdx.y + threadIdx.x;
  const int cRow0 = cRowsPerCudaBlock * blockIdx.x;
  const int cBlocksPerRowX = pNcolsX / qk;
  constexpr int cBlocksPerIter = cVdr * cNwarps * cWarpSize / qi;

  const uint32_t cChannelDst = blockIdx.y;

  uint32_t vChannelX;
  uint32_t vChannelY;
  uint32_t vSampleDst;

  ggml_cuda_pdl_sync();
  vChannelX = ncols_dst == 1 && vIds ? vIds[cChannelDst] : fastdiv(cChannelDst, pChannelRatio);
  vChannelY = ncols_dst == 1 && vIds ? fastmodulo(cChannelDst, pNchannelsY) : cChannelDst;
  vSampleDst = blockIdx.z;

  const uint32_t cSampleX = fastdiv(vSampleDst, pSampleRatio);
  const uint32_t cSampleY = vSampleDst;

  bool vUseGate = false;
  bool vUseBias = false;
  bool vUseGateBias = false;
  bool vUseScale = false;
  bool vUseGateScale = false;
  [[maybe_unused]] const void *vVgate = nullptr;
  const float *vXBias = nullptr;
  const float *vGateBias = nullptr;
  const float *vXScale = nullptr;
  const float *vGateScale = nullptr;
  ggml_glu_op vActiveGlu;
  float vGluLimit = 0.0f;

  if constexpr (has_fusion) {
    vUseGate = pFusion.gate != nullptr;
    vUseBias = pFusion.x_bias != nullptr;
    vUseGateBias = pFusion.gate_bias != nullptr && vUseGate;
    vVgate = pFusion.gate;
    vXBias = (const float *)pFusion.x_bias;
    vGateBias = (const float *)pFusion.gate_bias;
    vActiveGlu = pFusion.glu_op;
    vGluLimit = pFusion.glu_limit;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vUseScale = pFusion.x_scale != nullptr;
      vUseGateScale = pFusion.gate_scale != nullptr && vUseGate;
      vXScale = (const float *)pFusion.x_scale;
      vGateScale = (const float *)pFusion.gate_scale;
    }
  }

  [[maybe_unused]] float aXBiases[ncols_dst] = {0.0f};
  [[maybe_unused]] float aGateBiases[ncols_dst] = {0.0f};
  [[maybe_unused]] float vXScales = 1.0f;
  [[maybe_unused]] float vGateScales = 1.0f;
  if constexpr (has_fusion) {
    // 1. Hide latency by prefetching bias, gates and scales here
    // 2. load only on threads that won't die after partial sum calculation
    const uint32_t channel_bias = vIds ? vChannelX : cChannelDst;
    if (threadIdx.x < cRowsPerCudaBlock && threadIdx.y == 0 &&
        (cRowsPerCudaBlock == 1 || uint32_t(cRow0 + threadIdx.x) < pStrideColDst)) {
      if (vUseBias) {
        vXBias = vXBias + vSampleDst * pStrideSampleDst + channel_bias * pStrideChannelDst + cRow0;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
          aXBiases[j] = vXBias[j * pStrideColDst + threadIdx.x];
        }
      }
      if (vUseGateBias) {
        vGateBias = vGateBias + vSampleDst * pStrideSampleDst + channel_bias * pStrideChannelDst + cRow0;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
          aGateBiases[j] = vGateBias[j * pStrideColDst + threadIdx.x];
        }
      }
      if constexpr (type == GGML_TYPE_NVFP4) {
        if (vUseScale) {
          vXScales = vXScale[vIds ? vChannelX : 0];
        }
        if (vUseGateScale) {
          vGateScales = vGateScale[vIds ? vChannelX : 0];
        }
      }
    }
  }

  // partial sum for each thread
  float aTmp[ncols_dst][cRowsPerCudaBlock] = {{0.0f}};
  float vTmpGate[ncols_dst][cRowsPerCudaBlock] = {{0.0f}};

  const block_q8_1 *vY = ((const block_q8_1 *)vVy) + cSampleY * pStrideSampleY + vChannelY * pStrideChannelY;
  const int cKbxOffset = cSampleX * pStrideSampleX + vChannelX * pStrideChannelX + cRow0 * pStrideRowX;

  for (int kbx = cTid / (qi / cVdr); kbx < cBlocksPerRowX; kbx += cBlocksPerIter) {
    const int cKby = kbx * (qk / QK8_1); // y block index that aligns with kbx

    // x block quant index when casting the quants to int
    const int cKqs = cVdr * (cTid % (qi / cVdr));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    // start the next iterations' weight loads early
    if constexpr (mmvq_should_prefetch(type)) {
      constexpr int pf_dist = 2; // loop iterations, not blocks
      const int kbx_pf = kbx + pf_dist * cBlocksPerIter;
      if (kbx_pf < cBlocksPerRowX) {
#pragma unroll
        for (int vI = 0; vI < cRowsPerCudaBlock; ++vI) {
          const size_t off = (size_t)(cKbxOffset + vI * pStrideRowX + kbx_pf) * ggml_cuda_type_traits<type>::bs;
          mmvq_prefetch_l2((const char *)vVx + off);
          if constexpr (has_fusion) {
            if (vUseGate) {
              mmvq_prefetch_l2((const char *)vVgate + off);
            }
          }
        }
      }
    }
#endif

#pragma unroll
    for (int vJ = 0; vJ < ncols_dst; ++vJ) {
#pragma unroll
      for (int vI = 0; vI < cRowsPerCudaBlock; ++vI) {
        aTmp[vJ][vI] += cVecDotQCuda(vVx, &vY[vJ * pStrideColY + cKby], cKbxOffset + vI * pStrideRowX + kbx, cKqs);
        if constexpr (has_fusion) {
          if (vUseGate) {
            vTmpGate[vJ][vI] +=
                cVecDotQCuda(vVgate, &vY[vJ * pStrideColY + cKby], cKbxOffset + vI * pStrideRowX + kbx, cKqs);
          }
        }
      }
    }
  }

  __shared__ float vTmpShared[cNwarps - 1 > 0 ? cNwarps - 1 : 1][ncols_dst][cRowsPerCudaBlock][cWarpSize];
  [[maybe_unused]] __shared__ float aTmpSharedGate[(has_fusion && (cNwarps - 1 > 0)) ? cNwarps - 1 : 1][ncols_dst]
                                                  [cRowsPerCudaBlock][cWarpSize];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int vJ = 0; vJ < ncols_dst; ++vJ) {
#pragma unroll
      for (int vI = 0; vI < cRowsPerCudaBlock; ++vI) {
        vTmpShared[threadIdx.y - 1][vJ][vI][threadIdx.x] = aTmp[vJ][vI];
        if constexpr (has_fusion) {
          if (vUseGate) {
            aTmpSharedGate[threadIdx.y - 1][vJ][vI][threadIdx.x] = vTmpGate[vJ][vI];
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

  vDst += vSampleDst * pStrideSampleDst + cChannelDst * pStrideChannelDst + cRow0;

  // sum up partial sums and write back result
#pragma unroll
  for (int vJ = 0; vJ < ncols_dst; ++vJ) {
#pragma unroll
    for (int vI = 0; vI < cRowsPerCudaBlock; ++vI) {
#pragma unroll
      for (int vL = 0; vL < cNwarps - 1; ++vL) {
        aTmp[vJ][vI] += vTmpShared[vL][vJ][vI][threadIdx.x];
        if constexpr (has_fusion) {
          if (vUseGate) {
            vTmpGate[vJ][vI] += aTmpSharedGate[vL][vJ][vI][threadIdx.x];
          }
        }
      }
      aTmp[vJ][vI] = warp_reduce_sum<cWarpSize>(aTmp[vJ][vI]);
      if constexpr (has_fusion) {
        if (vUseGate) {
          vTmpGate[vJ][vI] = warp_reduce_sum<cWarpSize>(vTmpGate[vJ][vI]);
        }
      }

      if (threadIdx.x == vI && (cRowsPerCudaBlock == 1 || uint32_t(cRow0 + vI) < pStrideColDst)) {
        float vResult = aTmp[vJ][vI];
        if constexpr (has_fusion) {
          if constexpr (type == GGML_TYPE_NVFP4) {
            vResult *= vXScales;
          }
          vResult += aXBiases[vJ];
          if (vUseGate) {
            float gate_value = vTmpGate[vJ][vI];
            if constexpr (type == GGML_TYPE_NVFP4) {
              gate_value *= vGateScales;
            }
            gate_value += aGateBiases[vJ];
            switch (vActiveGlu) {
            case GGML_GLU_OP_SWIGLU:
              vResult *= ggml_cuda_op_silu_single(gate_value);
              break;
            case GGML_GLU_OP_GEGLU:
              vResult *= ggml_cuda_op_gelu_single(gate_value);
              break;
            case GGML_GLU_OP_SWIGLU_OAI:
              vResult = ggml_cuda_op_swiglu_oai_single(gate_value, vResult);
              break;
            case GGML_GLU_OP_SWIGLU_CLAMP:
              vResult = ggml_cuda_op_swiglu_clamp_single(gate_value, vResult, vGluLimit);
              break;
            default:
              vResult = vResult * gate_value;
              break;
            }
          }
        }
        vDst[vJ * pStrideColDst + vI] = vResult;
      }
    }
  }

  if constexpr (!has_fusion) {
    GGML_UNUSED_VARS(vUseGate, vUseBias, vUseGateBias, vUseScale, vUseGateScale, vActiveGlu, vGluLimit, vGateBias,
                     vXBias, vXScale, vGateScale, vTmpGate);
  }
  if constexpr (type != GGML_TYPE_NVFP4) {
    GGML_UNUSED_VARS(vUseScale, vUseGateScale, vXScale, vGateScale, vXScales, vGateScales);
  }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(fGetMmvqMmidMaxBatchForDevice<type>() * ggml_cuda_get_physical_warp_size(), 1) static __global__
    void fMulMatVecQMoe(const void *pVxPtr, const void *pVyPtr, const int32_t *pIdsPtr,
                        const ggml_cuda_mm_fusion_args_device pFusion, float *pDstPtr, const uint32_t pNcolsX,
                        const uint3 pNchannelsY, const uint32_t pNrowsX, const uint32_t pStrideRowX,
                        const uint32_t pStrideColY, const uint32_t pStrideColDst, const uint32_t pStrideChannelX,
                        const uint32_t pStrideChannelY, const uint32_t pStrideChannelDst, const uint32_t pNcolsDst,
                        const uint32_t pIdsStride) {
  const void *GGML_CUDA_RESTRICT vVx = pVxPtr;
  const void *GGML_CUDA_RESTRICT vVy = pVyPtr;
  const int32_t *GGML_CUDA_RESTRICT vIds = pIdsPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;

  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  constexpr int cVdr = fGetVdrMmvq(type);
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();

  constexpr vec_dot_q_cuda_t cVecDotQCuda = fGetVecDotQCuda(type);

  // fuse gate, bias, scales, and glu_op into the up projection
  bool vUseGate = false;
  const void *vVgate = nullptr;
  const float *vXBias = nullptr;
  const float *vGateBias = nullptr;
  const float *vXScale = nullptr;
  const float *vGateScale = nullptr;
  ggml_glu_op vActiveGlu = GGML_GLU_OP_SWIGLU;
  float vGluLimit = 0.0f;

  if constexpr (has_fusion) {
    vUseGate = pFusion.gate != nullptr;
    vVgate = pFusion.gate;
    vXBias = (const float *)pFusion.x_bias;
    vGateBias = (const float *)pFusion.gate_bias;
    vActiveGlu = pFusion.glu_op;
    vGluLimit = pFusion.glu_limit;
    if constexpr (type == GGML_TYPE_NVFP4) {
      vXScale = (const float *)pFusion.x_scale;
      vGateScale = (const float *)pFusion.gate_scale;
    }
  }

  const uint32_t cTokenIdx = threadIdx.y;
  const int cRow0 = c_rows_per_block * blockIdx.x;
  const int cBlocksPerRowX = pNcolsX / qk;
  constexpr int cBlocksPerIter = cVdr * cWarpSize / qi;

  const uint32_t cChannelDst = blockIdx.y;

  if (cTokenIdx >= pNcolsDst) {
    return;
  }

  ggml_cuda_pdl_sync();
  const uint32_t cChannelX = vIds[cChannelDst + cTokenIdx * pIdsStride];
  const uint32_t cChannelY = fastmodulo(cChannelDst, pNchannelsY);

  const block_q8_1 *vY = ((const block_q8_1 *)vVy) + cChannelY * pStrideChannelY + cTokenIdx * pStrideColY;
  const int cKbxOffset = cChannelX * pStrideChannelX + cRow0 * pStrideRowX;

  // partial sum for each thread
  float aTmp[c_rows_per_block] = {0.0f};
  float vTmpGate[c_rows_per_block] = {0.0f};

  for (int vKbx = threadIdx.x / (qi / cVdr); vKbx < cBlocksPerRowX; vKbx += cBlocksPerIter) {
    const int cKby = vKbx * (qk / QK8_1);
    const int cKqs = cVdr * (threadIdx.x % (qi / cVdr));

#pragma unroll
    for (int vI = 0; vI < c_rows_per_block; ++vI) {
      aTmp[vI] += cVecDotQCuda(vVx, &vY[cKby], cKbxOffset + vI * pStrideRowX + vKbx, cKqs);
      if constexpr (has_fusion) {
        if (vUseGate) {
          vTmpGate[vI] += cVecDotQCuda(vVgate, &vY[cKby], cKbxOffset + vI * pStrideRowX + vKbx, cKqs);
        }
      }
    }
  }

  ggml_cuda_pdl_lc();

  // Warp-level reduction only - no shared memory needed
#pragma unroll
  for (int vI = 0; vI < c_rows_per_block; ++vI) {
    aTmp[vI] = warp_reduce_sum<cWarpSize>(aTmp[vI]);
    if constexpr (has_fusion) {
      if (vUseGate) {
        vTmpGate[vI] = warp_reduce_sum<cWarpSize>(vTmpGate[vI]);
      }
    }
  }

  // Write results
  if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(cRow0 + threadIdx.x) < pNrowsX)) {
    float vResult = aTmp[threadIdx.x];
    if constexpr (has_fusion) {
      const uint32_t bias_idx = cChannelX * pStrideChannelDst + cRow0 + threadIdx.x;

      if constexpr (type == GGML_TYPE_NVFP4) {
        if (vXScale) {
          vResult *= vXScale[cChannelX];
        }
      }
      if (vXBias) {
        vResult += vXBias[bias_idx];
      }
      if (vUseGate) {
        float gate_value = vTmpGate[threadIdx.x];
        if constexpr (type == GGML_TYPE_NVFP4) {
          if (vGateScale) {
            gate_value *= vGateScale[cChannelX];
          }
        }
        if (vGateBias) {
          gate_value += vGateBias[bias_idx];
        }
        switch (vActiveGlu) {
        case GGML_GLU_OP_SWIGLU:
          vResult *= ggml_cuda_op_silu_single(gate_value);
          break;
        case GGML_GLU_OP_GEGLU:
          vResult *= ggml_cuda_op_gelu_single(gate_value);
          break;
        case GGML_GLU_OP_SWIGLU_OAI:
          vResult = ggml_cuda_op_swiglu_oai_single(gate_value, vResult);
          break;
        case GGML_GLU_OP_SWIGLU_CLAMP:
          vResult = ggml_cuda_op_swiglu_clamp_single(gate_value, vResult, vGluLimit);
          break;
        default:
          vResult = vResult * gate_value;
          break;
        }
      }
    }
    vDst[cChannelDst * pStrideChannelDst + cTokenIdx * pStrideColDst + cRow0 + threadIdx.x] = vResult;
  }

  if constexpr (!has_fusion) {
    GGML_UNUSED_VARS(vUseGate, vTmpGate, vVgate, vXBias, vGateBias, vActiveGlu, vGluLimit, vXScale, vGateScale);
  } else if constexpr (type != GGML_TYPE_NVFP4) {
    GGML_UNUSED_VARS(vXScale, vGateScale);
  }
}

template <ggml_type type>
static std::pair<dim3, dim3> fCalcLaunchParams(const int pNcolsDst, const int pNrowsX, const int pNchannelsDst,
                                               const int pNsamplesOrNtokens, const int pWarpSize,
                                               const mmvq_parameter_table_id pTableId, const bool pSmallK = false) {
  const int cNwarps = fCalcNwarps(type, pNcolsDst, pTableId);
  const int cRpb = fCalcRowsPerBlock(pNcolsDst, pTableId, pSmallK, cNwarps);
  const int64_t cNblocks = (pNrowsX + cRpb - 1) / cRpb;
  const dim3 cBlockNums(cNblocks, pNchannelsDst, pNsamplesOrNtokens);
  const dim3 cBlockDims(pWarpSize, cNwarps, 1);
  return {cBlockNums, cBlockDims};
}

template <ggml_type type, int c_ncols_dst, bool small_k = false>
static void fMulMatVecQSwitchFusion(const void *pVx, const void *pVy, const int32_t *pIds,
                                    const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const uint32_t pNcolsX,
                                    const uint3 pNchannelsY, const uint32_t pStrideRowX, const uint32_t pStrideColY,
                                    const uint32_t pStrideColDst, const uint3 pChannelRatio,
                                    const uint32_t pStrideChannelX, const uint32_t pStrideChannelY,
                                    const uint32_t pStrideChannelDst, const uint3 pSampleRatio,
                                    const uint32_t pStrideSampleX, const uint32_t pStrideSampleY,
                                    const uint32_t pStrideSampleDst, const dim3 &pBlockNums, const dim3 &pBlockDims,
                                    const int pNbytesShared, const uint32_t pIdsStride, cudaStream_t pStream) {

  const bool cHasFusion = pFusion.gate != nullptr || pFusion.x_bias != nullptr || pFusion.gate_bias != nullptr ||
                          pFusion.x_scale != nullptr || pFusion.gate_scale != nullptr;
  if constexpr (c_ncols_dst == 1) {
    if (cHasFusion) {
      const ggml_cuda_kernel_launch_params launch_params =
          ggml_cuda_kernel_launch_params(pBlockNums, pBlockDims, pNbytesShared, pStream);
      ggml_cuda_kernel_launch(fMulMatVecQ<type, c_ncols_dst, true, small_k>, launch_params, pVx, pVy, pIds, pFusion,
                              pDst, pNcolsX, pNchannelsY, pStrideRowX, pStrideColY, pStrideColDst, pChannelRatio,
                              pStrideChannelX, pStrideChannelY, pStrideChannelDst, pSampleRatio, pStrideSampleX,
                              pStrideSampleY, pStrideSampleDst, pIdsStride);
      return;
    }
  }

  GGML_ASSERT(!cHasFusion && "fusion only supported for ncols_dst=1");

  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(pBlockNums, pBlockDims, pNbytesShared, pStream);
  ggml_cuda_kernel_launch(fMulMatVecQ<type, c_ncols_dst, false, small_k>, cLaunchParams, pVx, pVy, pIds, pFusion, pDst,
                          pNcolsX, pNchannelsY, pStrideRowX, pStrideColY, pStrideColDst, pChannelRatio, pStrideChannelX,
                          pStrideChannelY, pStrideChannelDst, pSampleRatio, pStrideSampleX, pStrideSampleY,
                          pStrideSampleDst, pIdsStride);
}

template <ggml_type type>
static void fMulMatVecQMoeLaunch(const void *pVx, const void *pVy, const int32_t *pIds,
                                 const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const uint32_t pNcolsX,
                                 const uint3 pNchannelsY, const uint32_t pNrowsX, const uint32_t pStrideRowX,
                                 const uint32_t pStrideColY, const uint32_t pStrideColDst,
                                 const uint32_t pStrideChannelX, const uint32_t pStrideChannelY,
                                 const uint32_t pStrideChannelDst, const uint32_t pNcolsDst, const uint32_t pIdsStride,
                                 const int pWarpSize, const int pNchannelsDst, cudaStream_t pStream) {

  constexpr int cRowsPerBlock = 2; // 2 gives best perf based on tuning
  const int64_t cNblocksRows = (pNrowsX + cRowsPerBlock - 1) / cRowsPerBlock;
  const dim3 cBlockNums(cNblocksRows, pNchannelsDst);
  const dim3 cBlockDims(pWarpSize, pNcolsDst);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, pStream);

  const bool cHasFusion = pFusion.gate != nullptr || pFusion.x_bias != nullptr || pFusion.gate_bias != nullptr ||
                          pFusion.x_scale != nullptr || pFusion.gate_scale != nullptr;

  if (cHasFusion) {
    ggml_cuda_kernel_launch(fMulMatVecQMoe<type, cRowsPerBlock, true>, cLaunchParams, pVx, pVy, pIds, pFusion, pDst,
                            pNcolsX, pNchannelsY, pNrowsX, pStrideRowX, pStrideColY, pStrideColDst, pStrideChannelX,
                            pStrideChannelY, pStrideChannelDst, pNcolsDst, pIdsStride);
  } else {
    ggml_cuda_kernel_launch(fMulMatVecQMoe<type, cRowsPerBlock, false>, cLaunchParams, pVx, pVy, pIds, pFusion, pDst,
                            pNcolsX, pNchannelsY, pNrowsX, pStrideRowX, pStrideColY, pStrideColDst, pStrideChannelX,
                            pStrideChannelY, pStrideChannelDst, pNcolsDst, pIdsStride);
  }
}

template <ggml_type type>
static void fMulMatVecQSwitchNcolsDst(const void *pVx, const void *pVy, const int32_t *pIds,
                                      const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const int pNcolsX,
                                      const int pNrowsX, const int pNcolsDst, const int pStrideRowX,
                                      const int pStrideColY, const int pStrideColDst, const int pNchannelsX,
                                      const int pNchannelsY, const int pNchannelsDst, const int pStrideChannelX,
                                      const int pStrideChannelY, const int pStrideChannelDst, const int pNsamplesX,
                                      const int pNsamplesDst, const int pStrideSampleX, const int pStrideSampleY,
                                      const int pStrideSampleDst, const int pIdsStride, cudaStream_t pStream) {

  GGML_ASSERT(pNcolsX % ggml_blck_size(type) == 0);
  GGML_ASSERT(pNcolsDst <= MMVQ_MAX_BATCH_SIZE);

  const uint3 cNchannelsYFd = pIds ? init_fastdiv_values(pNchannelsY) : make_uint3(0, 0, 0);
  const uint3 cChannelRatioFd = pIds ? make_uint3(0, 0, 0) : init_fastdiv_values(pNchannelsDst / pNchannelsX);
  const uint3 cSampleRatioFd = init_fastdiv_values(pNsamplesDst / pNsamplesX);

  const int cDevice = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cDevice].cc;
  const int cWarpSize = ggml_cuda_info().devices[cDevice].warp_size;
  const mmvq_parameter_table_id cTableId = fGetDeviceTableId(cCc);

  const bool cHasIds = pIds != nullptr;

  const auto cShouldUseSmallK = [&](int pCNcolsDst) {
    // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
    // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int cVdr = fGetVdrMmvq(type);
    const int cBlocksPerRowX = pNcolsX / qk;
    const int cBlocksPerIter1warp = cVdr * cWarpSize / qi;
    const int cNwarps = fCalcNwarps(type, pCNcolsDst, cTableId);
    bool vUse = cNwarps > 1 && cBlocksPerRowX < cNwarps * cBlocksPerIter1warp;

    constexpr std::array<ggml_type, 2> aIqSlowTuring = {
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
    };
    constexpr std::array<ggml_type, 8> aIqSlowOther = {
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
    };
    constexpr std::array<ggml_type, 3> aSlowPascal = {
        GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
    };

    const bool cIsNvidiaTuringPlus = GGML_CUDA_CC_IS_NVIDIA(cCc) && cCc >= GGML_CUDA_CC_TURING;
    const bool cIsNvidiaPascalOlder = GGML_CUDA_CC_IS_NVIDIA(cCc) && cCc < GGML_CUDA_CC_VOLTA;

    if (cIsNvidiaTuringPlus) {
      if (pNcolsDst == 1 && std::find(aIqSlowTuring.begin(), aIqSlowTuring.end(), type) != aIqSlowTuring.end()) {
        vUse = false;
      }
    } else if ((pNcolsDst == 1 && std::find(aIqSlowOther.begin(), aIqSlowOther.end(), type) != aIqSlowOther.end()) ||
               (cIsNvidiaPascalOlder && std::find(aSlowPascal.begin(), aSlowPascal.end(), type) != aSlowPascal.end()) ||
               GGML_CUDA_CC_IS_RDNA(cCc)) {
      vUse = false;
    }

    return vUse;
  };

  if (cHasIds && pNcolsDst > 1) {
    // Multi-token MUL_MAT_ID path - dedicated MoE kernel
    fMulMatVecQMoeLaunch<type>(pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pNrowsX, pStrideRowX, pStrideColY,
                               pStrideColDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNcolsDst,
                               pIdsStride, cWarpSize, pNchannelsDst, pStream);
    return;
  }

  switch (pNcolsDst) {
  case 1: {
    constexpr int cCNcolsDst = 1;

    bool vUseSmallK = cShouldUseSmallK(cCNcolsDst);

    if (vUseSmallK) {
      std::pair<dim3, dim3> vDims =
          fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId, true);
      fMulMatVecQSwitchFusion<type, cCNcolsDst, true>(
          pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst,
          cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
          pStrideSampleY, pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
    } else {
      std::pair<dim3, dim3> vDims =
          fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
      fMulMatVecQSwitchFusion<type, cCNcolsDst>(
          pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst,
          cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
          pStrideSampleY, pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
    }
  } break;
  case 2: {
    constexpr int cCNcolsDst = 2;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 3: {
    constexpr int cCNcolsDst = 3;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 4: {
    constexpr int cCNcolsDst = 4;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 5: {
    constexpr int cCNcolsDst = 5;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 6: {
    constexpr int cCNcolsDst = 6;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 7: {
    constexpr int cCNcolsDst = 7;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  case 8: {
    constexpr int cCNcolsDst = 8;
    std::pair<dim3, dim3> vDims =
        fCalcLaunchParams<type>(cCNcolsDst, pNrowsX, pNchannelsDst, pNsamplesDst, cWarpSize, cTableId);
    fMulMatVecQSwitchFusion<type, cCNcolsDst>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, cNchannelsYFd, pStrideRowX, pStrideColY, pStrideColDst, cChannelRatioFd,
        pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX, pStrideSampleY,
        pStrideSampleDst, vDims.first, vDims.second, 0, pIdsStride, pStream);
  } break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}
static void fMulMatVecQSwitchType(const void *pVx, const ggml_type pTypeX, const void *pVy, const int32_t *pIds,
                                  const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const int pNcolsX,
                                  const int pNrowsX, const int pNcolsDst, const int pStrideRowX, const int pStrideColY,
                                  const int pStrideColDst, const int pNchannelsX, const int pNchannelsY,
                                  const int pNchannelsDst, const int pStrideChannelX, const int pStrideChannelY,
                                  const int pStrideChannelDst, const int pNsamplesX, const int pNsamplesDst,
                                  const int pStrideSampleX, const int pStrideSampleY, const int pStrideSampleDst,
                                  const int pIdsStride, cudaStream_t pStream) {
  switch (pTypeX) {
  case GGML_TYPE_Q1_0:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q1_0>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q2_0:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q2_0>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q4_0:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q4_0>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q4_1:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q4_1>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q5_0:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q5_0>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q5_1:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q5_1>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q8_0:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q8_0>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_MXFP4:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_MXFP4>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_NVFP4:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_NVFP4>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q2_K:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q2_K>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q3_K:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q3_K>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q4_K:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q4_K>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q5_K:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q5_K>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_Q6_K:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_Q6_K>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ2_XXS:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ2_XXS>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ2_XS:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ2_XS>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ2_S:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ2_S>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ3_XXS:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ3_XXS>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ1_S:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ1_S>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ1_M:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ1_M>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ4_NL:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ4_NL>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ4_XS:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ4_XS>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  case GGML_TYPE_IQ3_S:
    fMulMatVecQSwitchNcolsDst<GGML_TYPE_IQ3_S>(
        pVx, pVy, pIds, pFusion, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRowX, pStrideColY, pStrideColDst,
        pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX,
        pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
    break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                             const ggml_tensor *pIds, ggml_tensor *dst, const ggml_cuda_mm_fusion_args_host *pFusion) {
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(!pIds || pIds->type == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

  GGML_TENSOR_BINARY_OP_LOCALS;

  cudaStream_t vStream = pCtx.stream();

  const size_t cTsSrc0 = ggml_type_size(src0->type);
  const size_t cTsSrc1 = ggml_type_size(src1->type);
  const size_t cTsDst = ggml_type_size(dst->type);

  GGML_ASSERT(nb00 == cTsSrc0);
  GGML_ASSERT(nb10 == cTsSrc1);
  GGML_ASSERT(nb0 == cTsDst);
  GGML_ASSERT(!pIds || pIds->nb[0] == ggml_type_size(pIds->type));

  GGML_ASSERT(!pIds || ne12 <= MMVQ_MAX_BATCH_SIZE);

  const float *vSrc1D = (const float *)src1->data;
  const int32_t *vIdsD = pIds ? (const int32_t *)pIds->data : nullptr;
  float *vDstD = (float *)dst->data;

  ggml_cuda_mm_fusion_args_device vFusionLocal{};

  if (pFusion) {
    const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    GGML_ASSERT(!pIds || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cCc));
    GGML_ASSERT(pIds || dst->ne[1] == 1);
    // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
    // non-negligible for some models such as gpt-oss-20b
    GGML_ASSERT((pFusion->x_scale == nullptr && pFusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

    if (pFusion->x_bias) {
      GGML_ASSERT(pFusion->x_bias->type == GGML_TYPE_F32);
      GGML_ASSERT(pFusion->x_bias->ne[0] == dst->ne[0]);
      GGML_ASSERT(!pIds || pFusion->x_bias->ne[1] == src0->ne[2]);
      vFusionLocal.x_bias = pFusion->x_bias->data;
    }
    if (pFusion->gate) {
      GGML_ASSERT(pFusion->gate->type == src0->type && ggml_are_same_stride(pFusion->gate, src0));
      vFusionLocal.gate = pFusion->gate->data;
    }
    if (pFusion->gate_bias) {
      GGML_ASSERT(pFusion->gate_bias->type == GGML_TYPE_F32);
      GGML_ASSERT(pFusion->gate_bias->ne[0] == dst->ne[0]);
      GGML_ASSERT(!pIds || pFusion->gate_bias->ne[1] == src0->ne[2]);
      vFusionLocal.gate_bias = pFusion->gate_bias->data;
    }
    if (pFusion->x_scale) {
      GGML_ASSERT(pFusion->x_scale->type == GGML_TYPE_F32);
      GGML_ASSERT(ggml_is_contiguous(pFusion->x_scale));
      GGML_ASSERT(ggml_nelements(pFusion->x_scale) == (pIds ? src0->ne[2] : 1));
      vFusionLocal.x_scale = pFusion->x_scale->data;
    }
    if (pFusion->gate_scale) {
      GGML_ASSERT(pFusion->gate_scale->type == GGML_TYPE_F32);
      GGML_ASSERT(ggml_is_contiguous(pFusion->gate_scale));
      GGML_ASSERT(ggml_nelements(pFusion->gate_scale) == (pIds ? src0->ne[2] : 1));
      vFusionLocal.gate_scale = pFusion->gate_scale->data;
    }
    vFusionLocal.glu_op = pFusion->glu_op;
    vFusionLocal.glu_limit = pFusion->glu_limit;
  }

  // If src0 is a temporary compute buffer, clear any potential padding.
  if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
    const size_t cSizeData = ggml_nbytes(src0);
    const size_t cSizeAlloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
    if (cSizeAlloc > cSizeData) {
      GGML_ASSERT(ggml_is_contiguously_allocated(src0));
      GGML_ASSERT(!src0->view_src);
      CUDA_CHECK(cudaMemsetAsync((char *)src0->data + cSizeData, 0, cSizeAlloc - cSizeData, vStream));
    }
  }

  const int64_t cNe10Padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
  ggml_cuda_pool_alloc<char> vSrc1Q81(pCtx.pool(), ne13 * ne12 * ne11 * cNe10Padded * sizeof(block_q8_1) / QK8_1);
  {
    const int64_t cS11 = src1->nb[1] / cTsSrc1;
    const int64_t cS12 = src1->nb[2] / cTsSrc1;
    const int64_t cS13 = src1->nb[3] / cTsSrc1;
    quantize_row_q8_1_cuda(vSrc1D, nullptr, vSrc1Q81.get(), src0->type, ne10, cS11, cS12, cS13, cNe10Padded, ne11, ne12,
                           ne13, vStream);
  }

  const int64_t cS01 = src0->nb[1] / cTsSrc0;
  const int64_t cS11 = cNe10Padded / QK8_1;
  const int64_t cS1 = dst->nb[1] / cTsDst;
  const int64_t cS02 = src0->nb[2] / cTsSrc0;
  const int64_t cS2 = dst->nb[2] / cTsDst;
  const int64_t cS03 = src0->nb[3] / cTsSrc0;
  const int64_t cS3 = dst->nb[3] / cTsDst;

  const int64_t cS12 = ne11 * cS11;
  const int64_t cS13 = ne12 * cS12;

  // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
  const int64_t cNcolsDst = pIds ? ne2 : ne1;
  const int64_t cNchannelsY = pIds ? ne11 : ne12;
  const int64_t cNchannelsDst = pIds ? ne1 : ne2;
  const int64_t cStrideColDst = pIds ? cS2 : cS1;
  const int64_t cStrideColY = pIds ? cS12 : cS11;
  const int64_t cStrideChannelDst = pIds ? cS1 : cS2;
  const int64_t cStrideChannelY = pIds ? cS11 : cS12;

  const int64_t cIdsStride = pIds ? pIds->nb[1] / ggml_type_size(pIds->type) : 0;

  fMulMatVecQSwitchType(src0->data, src0->type, vSrc1Q81.get(), vIdsD, vFusionLocal, vDstD, ne00, ne01, cNcolsDst, cS01,
                        cStrideColY, cStrideColDst, ne02, cNchannelsY, cNchannelsDst, cS02, cStrideChannelY,
                        cStrideChannelDst, ne03, ne3, cS03, cS13, cS3, cIdsStride, vStream);
}

void ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                                ggml_tensor *pDst, const char *pSrc0DdI, const float *pSrc1DdfI, const char *pSrc1DdqI,
                                float *pDstDdI, const int64_t pRowLow, const int64_t pRowHigh, const int64_t pSrc1Ncols,
                                const int64_t pSrc1PaddedRowSize, cudaStream_t pStream) {

  const int64_t cNe00 = pSrc0->ne[0];
  const int64_t cRowDiff = pRowHigh - pRowLow;

  const int64_t cNe10 = pSrc1->ne[0];
  GGML_ASSERT(cNe10 % QK8_1 == 0);

  const int64_t cNe0 = pDst->ne[0];

  int vId = ggml_cuda_get_device();

  // the main device has a larger memory buffer to hold the results from all GPUs
  // nrows_dst == nrows of the matrix that the kernel writes into
  const int64_t cNrowsDst = vId == pCtx.device ? cNe0 : cRowDiff;

  const int cStrideRowX = cNe00 / ggml_blck_size(pSrc0->type);
  const int cStrideColY = pSrc1PaddedRowSize / QK8_1;

  ggml_cuda_mm_fusion_args_device vFusionLocal{};
  fMulMatVecQSwitchType(pSrc0DdI, pSrc0->type, pSrc1DdqI, nullptr, vFusionLocal, pDstDdI, cNe00, cRowDiff, pSrc1Ncols,
                        cStrideRowX, cStrideColY, cNrowsDst, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, pStream);

  GGML_UNUSED_VARS(pSrc1, pDst, pSrc1DdfI, pSrc1Ncols, pSrc1PaddedRowSize);
}
