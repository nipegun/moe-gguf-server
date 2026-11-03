#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

template <int DKQ, int DV, int ncols2>
static void fGgmlCudaFlashAttnExtMmaF16SwitchNcols1(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const ggml_tensor *vQ = pDst->src[0];

  if constexpr (ncols2 <= 8) {
    if (turing_mma_available(cCc) && vQ->ne[1] <= 8 / ncols2) {
      ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8 / ncols2, ncols2>(pCtx, pDst);
      return;
    }
  }

  if constexpr (ncols2 <= 16) {
    if (vQ->ne[1] <= 16 / ncols2) {
      ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16 / ncols2, ncols2>(pCtx, pDst);
      return;
    }
  }

  if (vQ->ne[1] <= 32 / ncols2 ||
      (GGML_CUDA_CC_IS_NVIDIA(cCc) && ggml_cuda_highest_compiled_arch(cCc) == GGML_CUDA_CC_TURING) ||
      (GGML_CUDA_CC_IS_AMD(cCc) && DKQ > 256)) {
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32 / ncols2, ncols2>(pCtx, pDst);
    return;
  }

  ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64 / ncols2, ncols2>(pCtx, pDst);
}

template <int DKQ, int DV>
static void fGgmlCudaFlashAttnExtMmaF16SwitchNcols2(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const ggml_tensor *vKqv = pDst;
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];

  float vMaxBias = 0.0f;
  memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));

  // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
  //     are put into the template specialization without GQA optimizations.
  bool vUseGqaOpt = vMask && vMaxBias == 0.0f && vK->ne[1] % FATTN_KQ_STRIDE == 0;
  for (const ggml_tensor *vT : {vQ, vK, vV, vMask}) {
    if (vT == nullptr || ggml_is_quantized(vT->type)) {
      continue;
    }
    for (size_t vI = 1; vI < GGML_MAX_DIMS; ++vI) {
      if (vT->nb[vI] % 16 != 0) {
        vUseGqaOpt = false;
        break;
      }
    }
  }

  GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);
  const int cGqaRatio = vQ->ne[2] / vK->ne[2];

  // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
  if (cCc == GGML_CUDA_CC_VOLTA) {
    if (vUseGqaOpt && cGqaRatio % 8 == 0) {
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 8>(pCtx, pDst);
      return;
    }

    if (vUseGqaOpt && cGqaRatio % 4 == 0) {
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 4>(pCtx, pDst);
      return;
    }

    if constexpr (DKQ <= 256) {
      if (vUseGqaOpt && cGqaRatio % 2 == 0) {
        fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 2>(pCtx, pDst);
        return;
      }

      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 1>(pCtx, pDst);
      return;
    } else {
      GGML_ABORT("fatal error");
    }
  }

  if (vUseGqaOpt && cGqaRatio > 4) {
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 8>(pCtx, pDst);
    return;
  }

  if (vUseGqaOpt && cGqaRatio > 2) {
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 4>(pCtx, pDst);
    return;
  }

  if (vUseGqaOpt && cGqaRatio > 1) {
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 2>(pCtx, pDst);
    return;
  }

  if constexpr (DKQ <= 256) {
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<DKQ, DV, 1>(pCtx, pDst);
  } else {
    GGML_ABORT("fatal error");
  }
}

static void fGgmlCudaFlashAttnExtMmaF16(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const ggml_tensor *vKqv = pDst;
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];

  switch (vQ->ne[0]) {
  case 64:
    GGML_ASSERT(vV->ne[0] == 64);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<64, 64>(pCtx, pDst);
    break;
  case 80:
    GGML_ASSERT(vV->ne[0] == 80);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<80, 80>(pCtx, pDst);
    break;
  case 96:
    GGML_ASSERT(vV->ne[0] == 96);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<96, 96>(pCtx, pDst);
    break;
  case 112:
    GGML_ASSERT(vV->ne[0] == 112);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<112, 112>(pCtx, pDst);
    break;
  case 128:
    GGML_ASSERT(vV->ne[0] == 128);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<128, 128>(pCtx, pDst);
    break;
  case 192: {
    // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
    GGML_ASSERT(vV->ne[0] == 128);
    float vMaxBias = 0.0f;
    memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));
    const bool cUseGqaOpt = vMask && vMaxBias == 0.0f;
    GGML_ASSERT(cUseGqaOpt);
    GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);
    const int cGqaRatio = vQ->ne[2] / vK->ne[2];
    if (cGqaRatio % 16 == 0) {
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<192, 128, 16>(pCtx, pDst);
    } else {
      GGML_ASSERT(cGqaRatio % 8 == 0);
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<192, 128, 8>(pCtx, pDst);
    }
  } break;
  case 256:
    GGML_ASSERT(vV->ne[0] == 256);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<256, 256>(pCtx, pDst);
    break;
  case 320:
    // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
    GGML_ASSERT(vV->ne[0] == 256);
    {
      float vMaxBias = 0.0f;
      memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));

      const bool cUseGqaOpt = vMask && vMaxBias == 0.0f;
      GGML_ASSERT(cUseGqaOpt);
      GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);
      const int cGqaRatio = vQ->ne[2] / vK->ne[2];
      GGML_ASSERT(cGqaRatio % 32 == 0);

      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<320, 256, 32>(pCtx, pDst);
    }
    break;
  case 512:
    GGML_ASSERT(vV->ne[0] == 512);
    fGgmlCudaFlashAttnExtMmaF16SwitchNcols2<512, 512>(pCtx, pDst);
    break;
  case 576: {
    // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
    GGML_ASSERT(vV->ne[0] == 512);
    float vMaxBias = 0.0f;
    memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));

    const bool cUseGqaOpt = vMask && vMaxBias == 0.0f;
    GGML_ASSERT(cUseGqaOpt);

    GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);
    const int cGqaRatio = vQ->ne[2] / vK->ne[2];
    if (cGqaRatio == 20) { // GLM 4.7 Flash
      if (cCc >= GGML_CUDA_CC_DGX_SPARK) {
        if (vQ->ne[1] <= 8) {
          fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 16>(pCtx, pDst);
          break;
        }
        fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
        break;
      }
      if (cCc >= GGML_CUDA_CC_BLACKWELL) {
        if (vQ->ne[1] <= 4 && vK->ne[1] >= 65536) {
          fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 16>(pCtx, pDst);
          break;
        }
        fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
        break;
      }
      if (cCc >= GGML_CUDA_CC_ADA_LOVELACE) {
        if (vQ->ne[1] <= 4) {
          fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 16>(pCtx, pDst);
          break;
        }
        fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
        break;
      }
      if (cCc >= GGML_CUDA_CC_TURING) {
        if (vQ->ne[1] <= 4) {
          if (vK->ne[1] <= 16384) {
            fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 16>(pCtx, pDst);
            break;
          }
          fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 32>(pCtx, pDst);
          break;
        }
        fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
        break;
      }
      // Volta:
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
    } else if (cGqaRatio % 16 == 0) {
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 16>(pCtx, pDst);
    } else {
      fGgmlCudaFlashAttnExtMmaF16SwitchNcols1<576, 512, 4>(pCtx, pDst);
    }
  } break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                    \
  if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                          \
    const bool type_K_okay =                                                                                           \
        type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16);    \
    const bool type_V_okay =                                                                                           \
        type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16);    \
    if (head_size == (D) && type_K_okay && type_V_okay) {                                                              \
      return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                   \
    }                                                                                                                  \
  }

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case)                                                                \
  FATTN_VEC_CASE(64, type_K_case, type_V_case)                                                                         \
  FATTN_VEC_CASE(128, type_K_case, type_V_case)                                                                        \
  FATTN_VEC_CASE(256, type_K_case, type_V_case)

typedef void (*fattn_vec_case_t)(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t fGgmlCudaGetFattnVecCase(const int64_t head_size, const ggml_type type_K,
                                                 const ggml_type type_V) {
  FATTN_VEC_CASES_ALL_D(F16, F16)
  FATTN_VEC_CASES_ALL_D(Q4_0, F16)
  FATTN_VEC_CASES_ALL_D(Q4_1, F16)
  FATTN_VEC_CASES_ALL_D(Q5_0, F16)
  FATTN_VEC_CASES_ALL_D(Q5_1, F16)
  FATTN_VEC_CASES_ALL_D(Q8_0, F16)
  FATTN_VEC_CASES_ALL_D(BF16, F16)

  FATTN_VEC_CASES_ALL_D(F16, Q4_0)
  FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
  FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
  FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
  FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
  FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
  FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

  FATTN_VEC_CASES_ALL_D(F16, Q4_1)
  FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
  FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
  FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
  FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
  FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
  FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

  FATTN_VEC_CASES_ALL_D(F16, Q5_0)
  FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
  FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
  FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
  FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
  FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
  FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

  FATTN_VEC_CASES_ALL_D(F16, Q5_1)
  FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
  FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
  FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
  FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
  FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
  FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

  FATTN_VEC_CASES_ALL_D(F16, Q8_0)
  FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
  FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
  FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
  FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
  FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
  FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

  FATTN_VEC_CASES_ALL_D(F16, BF16)
  FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
  FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
  FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
  FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
  FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
  FATTN_VEC_CASES_ALL_D(BF16, BF16)

  return nullptr;
}

static void fGgmlCudaFlashAttnExtVec(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];

  fattn_vec_case_t vVecCase = fGgmlCudaGetFattnVecCase(vQ->ne[0], vK->type, vV->type);
  if (vVecCase == nullptr) {
    static bool vWarned = false;
    if (!vWarned) {
      GGML_LOG_WARN(
          "%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
          "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
          __func__, ggml_type_name(vK->type), ggml_type_name(vV->type), ggml_type_name(vK->type),
          ggml_type_name(vV->type));
      vWarned = true;
    }
    vVecCase = fGgmlCudaGetFattnVecCase(vQ->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
  }
  GGML_ASSERT(vVecCase != nullptr);
  vVecCase(pCtx, pDst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
  BEST_FATTN_KERNEL_NONE = 0,
  BEST_FATTN_KERNEL_TILE = 200,
  BEST_FATTN_KERNEL_VEC = 100,
  BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool fGgmlCudaFattnKvTypeSupported(const ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
    return true;
  default:
    return false;
  }
}

static best_fattn_kernel fGgmlCudaGetBestFattnKernel(const int pDevice, const ggml_tensor *pDst) {
#ifndef FLASH_ATTN_AVAILABLE
  GGML_UNUSED(pDevice);
  GGML_UNUSED(pDst);
  return BEST_FATTN_KERNEL_NONE;
#endif // FLASH_ATTN_AVAILABLE

  const ggml_tensor *vKqv = pDst;
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];

  const int cGqaRatio = vQ->ne[2] / vK->ne[2];
  GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);

  float vMaxBias = 0.0f;
  memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));

  // The effective batch size for the kernel can be increased by gqa_ratio.
  // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is
  // not padded,
  bool vGqaOptApplies = cGqaRatio >= 2 && vMask && vMaxBias == 0.0f && vK->ne[1] % FATTN_KQ_STRIDE == 0;
  for (const ggml_tensor *vT : {vQ, vK, vV, vMask}) {
    if (vT == nullptr || ggml_is_quantized(vT->type)) {
      continue;
    }
    for (size_t vI = 1; vI < GGML_MAX_DIMS; ++vI) {
      if (vT->nb[vI] % 16 != 0) {
        vGqaOptApplies = false;
        break;
      }
    }
  }

  const int cCc = ggml_cuda_info().devices[pDevice].cc;

  switch (vK->ne[0]) {
  case 40:
  case 64:
  case 72:
  case 80:
  case 96:
  case 128:
  case 112:
  case 256:
    if (vV->ne[0] != vK->ne[0]) {
      return BEST_FATTN_KERNEL_NONE;
    }
    break;
  case 192:
    if (vV->ne[0] != 128 || !vGqaOptApplies) {
      return BEST_FATTN_KERNEL_NONE;
    }
    if (cGqaRatio % 8 != 0) {
      return BEST_FATTN_KERNEL_NONE;
    }
    break;
  case 320:
    if (vV->ne[0] != 256 || !vGqaOptApplies) {
      return BEST_FATTN_KERNEL_NONE;
    }
    if (cGqaRatio % 32 != 0) {
      return BEST_FATTN_KERNEL_NONE;
    }
    break;
  case 512:
    if (vV->ne[0] != vK->ne[0]) {
      return BEST_FATTN_KERNEL_NONE;
    }
    if (!vGqaOptApplies) {
      return BEST_FATTN_KERNEL_NONE;
    }
    break;
  case 576:
    if (vV->ne[0] != 512) {
      return BEST_FATTN_KERNEL_NONE;
    }
    if (!vGqaOptApplies) {
      return BEST_FATTN_KERNEL_NONE;
    }
    break;
  default:
    return BEST_FATTN_KERNEL_NONE;
  }

  if (!fGgmlCudaFattnKvTypeSupported(vK->type) || !fGgmlCudaFattnKvTypeSupported(vV->type)) {
    return BEST_FATTN_KERNEL_NONE;
  }

  if (vMask && vMask->ne[2] != 1) {
    return BEST_FATTN_KERNEL_NONE;
  }

  // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
  // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
  const bool cCanUseVectorKernel =
      vQ->ne[0] <= 256 && vQ->ne[0] % 64 == 0 && vQ->ne[0] != 192 && vK->ne[1] % FATTN_KQ_STRIDE == 0;

  // If Turing tensor cores are available, use them:
  if (turing_mma_available(cCc) && vQ->ne[0] != 40 && vQ->ne[0] != 72) {
    if (cCanUseVectorKernel) {
      if (!ggml_is_quantized(vK->type) && !ggml_is_quantized(vV->type)) {
        if (cCc >= GGML_CUDA_CC_ADA_LOVELACE && vQ->ne[1] == 1 && vQ->ne[3] == 1 &&
            !(cGqaRatio > 4 && vK->ne[1] >= 8192)) {
          return BEST_FATTN_KERNEL_VEC;
        }
      } else {
        if (cCc >= GGML_CUDA_CC_ADA_LOVELACE) {
          if (vQ->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
          }
        } else {
          if (vQ->ne[1] == 1) {
            return BEST_FATTN_KERNEL_VEC;
          }
        }
      }
      if (!vGqaOptApplies && vQ->ne[1] == 1) {
        return BEST_FATTN_KERNEL_VEC;
      }
    }
    return BEST_FATTN_KERNEL_MMA_F16;
  }

  const int cNcols2Max = vQ->ne[0] == 320 ? 32 : ((vQ->ne[0] == 576 || vQ->ne[0] == 192) ? 16 : 8);
  int vGqaRatioEff = 1;
  while (cGqaRatio % (2 * vGqaRatioEff) == 0 && vGqaRatioEff < cNcols2Max) {
    vGqaRatioEff *= 2;
  }

  if (volta_mma_available(cCc) && vQ->ne[0] != 40 && vQ->ne[0] != 72) {
    if (cCanUseVectorKernel && vQ->ne[1] * vGqaRatioEff <= 2) {
      return BEST_FATTN_KERNEL_VEC;
    }
    if (vQ->ne[1] * vGqaRatioEff <= 16) {
      return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
    }
    return BEST_FATTN_KERNEL_MMA_F16;
  }

  // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
  if ((amd_mfma_available(cCc) && vQ->ne[0] <= 256) && vQ->ne[0] != 40 && vQ->ne[0] != 72) {
    if ((vQ->ne[0] <= 64 && vQ->ne[1] * vGqaRatioEff > 8)) {
      return BEST_FATTN_KERNEL_MMA_F16;
    }
    if ((vQ->ne[0] <= 128 && vQ->ne[1] * vGqaRatioEff > 16)) {
      return BEST_FATTN_KERNEL_MMA_F16;
    }
    if ((vQ->ne[0] <= 256 && vQ->ne[1] * vGqaRatioEff > 64)) {
      return BEST_FATTN_KERNEL_MMA_F16;
    }
  }

  // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
  if ((amd_wmma_available(cCc) && vGqaOptApplies && vQ->ne[0] <= 128) && vQ->ne[0] != 40 && vQ->ne[0] != 72 &&
      vQ->ne[1] * vGqaRatioEff > 8) {
    return BEST_FATTN_KERNEL_MMA_F16;
  }

  // If there are no tensor cores available, use the generic tile kernel:
  if (cCanUseVectorKernel) {
    if (!ggml_is_quantized(vK->type) && !ggml_is_quantized(vV->type)) {
      if (vQ->ne[1] == 1) {
        if (!vGqaOptApplies) {
          return BEST_FATTN_KERNEL_VEC;
        }
      }
    } else {
      if (vQ->ne[1] <= 2) {
        return BEST_FATTN_KERNEL_VEC;
      }
    }
  }
  return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int pDevice, const ggml_tensor *pDst) {
  GGML_ASSERT(pDst->op == GGML_OP_FLASH_ATTN_EXT);

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];

  GGML_ASSERT(vK != nullptr);
  GGML_ASSERT(vV != nullptr);

  const best_fattn_kernel cKernel = fGgmlCudaGetBestFattnKernel(pDevice, pDst);

  bool vNeedF16K = false;
  bool vNeedF16V = false;

  switch (cKernel) {
  case BEST_FATTN_KERNEL_TILE:
  case BEST_FATTN_KERNEL_MMA_F16:
    vNeedF16K = true;
    vNeedF16V = true;
    break;
  case BEST_FATTN_KERNEL_VEC: {
    const bool cF16Fallback = fGgmlCudaGetFattnVecCase(vQ->ne[0], vK->type, vV->type) == nullptr;
    vNeedF16K = vK->type == GGML_TYPE_F32 || cF16Fallback;
    vNeedF16V = vV->type == GGML_TYPE_F32 || cF16Fallback;
  } break;
  case BEST_FATTN_KERNEL_NONE:
    break;
  }

  const ggml_cuda_flash_attn_ext_f16_extra_data cF16Extra =
      ggml_cuda_flash_attn_ext_get_f16_extra_data(pDst, vNeedF16K, vNeedF16V);

  return cF16Extra.end - (uintptr_t)pDst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  ggml_cuda_set_device(pCtx.device);
  switch (fGgmlCudaGetBestFattnKernel(ggml_cuda_get_device(), pDst)) {
  case BEST_FATTN_KERNEL_NONE:
    GGML_ABORT("fatal error");
  case BEST_FATTN_KERNEL_TILE:
    ggml_cuda_flash_attn_ext_tile(pCtx, pDst);
    break;
  case BEST_FATTN_KERNEL_VEC:
    fGgmlCudaFlashAttnExtVec(pCtx, pDst);
    break;
  case BEST_FATTN_KERNEL_MMA_F16:
    fGgmlCudaFlashAttnExtMmaF16(pCtx, pDst);
    break;
  }
}

bool ggml_cuda_flash_attn_ext_supported(int pDevice, const ggml_tensor *pDst) {
  return fGgmlCudaGetBestFattnKernel(pDevice, pDst) != BEST_FATTN_KERNEL_NONE;
}
