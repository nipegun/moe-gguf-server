#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
  bool vUseSigmoid;
  bool vUseSqrtSoftplus;
  bool vWithNorm;
  bool vDelayedSoftmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void fSoftmaxWarpInplace(float (&pVals)[experts_per_thread], const int pLimit, const int pLane) {
  float vMaxVal = -INFINITY;

#pragma unroll
  for (int vI = 0; vI < experts_per_thread; vI++) {
    const int cIdx = pLane + vI * WARP_SIZE;
    const bool cActive = !use_limit || (cIdx < pLimit);
    if (cActive) {
      vMaxVal = max(vMaxVal, pVals[vI]);
    }
  }

  vMaxVal = warp_reduce_max(vMaxVal);

  float vSum = 0.f;

#pragma unroll
  for (int vI = 0; vI < experts_per_thread; vI++) {
    const int cIdx = pLane + vI * WARP_SIZE;
    const bool cActive = !use_limit || (cIdx < pLimit);
    if (cActive) {
      const float cVal = expf(pVals[vI] - vMaxVal);
      pVals[vI] = cVal;
      vSum += cVal;
    } else {
      pVals[vI] = 0.f;
    }
  }

  vSum = warp_reduce_sum(vSum);

  const float cInvSum = 1.0f / vSum;

#pragma unroll
  for (int vI = 0; vI < experts_per_thread; vI++) {
    const int cIdx = pLane + vI * WARP_SIZE;
    const bool cActive = !use_limit || (cIdx < pLimit);
    if (cActive) {
      pVals[vI] *= cInvSum;
    }
  }
}

template <int experts_per_thread, bool use_limit>
__device__ void fSigmoidWarpInplace(float (&pVals)[experts_per_thread], const int pLimit, const int pLane) {
#pragma unroll
  for (int vI = 0; vI < experts_per_thread; vI++) {
    const int cIdx = pLane + vI * WARP_SIZE;
    const bool cActive = !use_limit || (cIdx < pLimit);
    pVals[vI] = cActive ? 1.f / (1.f + expf(-pVals[vI])) : -INFINITY;
  }
}

template <int experts_per_thread, bool use_limit>
__device__ void fSqrtSoftplusWarpInplace(float (&pVals)[experts_per_thread], const int pLimit, const int pLane) {
#pragma unroll
  for (int vI = 0; vI < experts_per_thread; vI++) {
    const int cIdx = pLane + vI * WARP_SIZE;
    const bool cActive = !use_limit || (cIdx < pLimit);
    pVals[vI] = cActive ? sqrtf(pVals[vI] > 20.0f ? pVals[vI] : logf(1.0f + expf(pVals[vI]))) : -INFINITY;
  }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK *WARP_SIZE, 1) __global__
    void fTopkMoeCuda(const float *pLogits, float *pWeights, int32_t *pIds, float *pBias, const int pNRows,
                      const int pNExpertUsed, const float pClampVal, const float pScaleVal,
                      const topk_moe_config pConfig) {
  const int cRow = blockIdx.x * blockDim.y + threadIdx.y;
  if (cRow >= pNRows) {
    return;
  }

  pLogits += n_experts * cRow;
  pWeights += pNExpertUsed * cRow;
  pIds += n_experts * cRow;

  constexpr int cExpertsPerThread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

  float aWt[cExpertsPerThread];

  // Initialize all slots to -INFINITY
#pragma unroll
  for (int vI = 0; vI < cExpertsPerThread; vI++) {
    aWt[vI] = -INFINITY;
  }

  ggml_cuda_pdl_sync();
#pragma unroll
  for (int vI = 0; vI < n_experts; vI += WARP_SIZE) {
    const int cExpert = vI + threadIdx.x;
    aWt[vI / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || cExpert < n_experts) ? pLogits[cExpert] : -INFINITY;
  }

  // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
  __syncthreads();

  if (!pConfig.vDelayedSoftmax) {
    if (pConfig.vUseSigmoid) {
      fSigmoidWarpInplace<cExpertsPerThread, false>(aWt, n_experts, threadIdx.x);
    } else if (pConfig.vUseSqrtSoftplus) {
      fSqrtSoftplusWarpInplace<cExpertsPerThread, false>(aWt, n_experts, threadIdx.x);
    } else {
      fSoftmaxWarpInplace<cExpertsPerThread, false>(aWt, n_experts, threadIdx.x);
    }
  }

  // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
  // NaN comparisons always return false, which would cause the same expert to be
  // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
  // -INFINITY sentinel used after each selection round.
  // More relevant for the cuBLAS path.
#pragma unroll
  for (int vI = 0; vI < cExpertsPerThread; vI++) {
    if (__isnanf(aWt[vI])) {
      aWt[vI] = -FLT_MAX;
    }
  }

  // selection_wt is only needed when bias is present (selection uses wt + bias)
  // when no bias, we use wt directly for both selection and weight values
  [[maybe_unused]] float aSelectionWt[has_bias ? cExpertsPerThread : 1];

  if constexpr (has_bias) {
#pragma unroll
    for (int i = 0; i < cExpertsPerThread; i++) {
      aSelectionWt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
      const int expert = i + threadIdx.x;
      aSelectionWt[i / WARP_SIZE] =
          (n_experts % WARP_SIZE == 0 || expert < n_experts) ? aWt[i / WARP_SIZE] + pBias[expert] : -INFINITY;
    }
  }

  // at this point, each thread holds either a portion of the softmax distribution
  // or the raw logits. We do the argmax reduce over n_expert_used, each time marking
  // the expert weight as -inf to exclude from the next iteration

  float vWtSum = 0.f;

  float aOutputWeights[cExpertsPerThread];

#pragma unroll
  for (int vI = 0; vI < cExpertsPerThread; vI++) {
    aOutputWeights[vI] = 0.f;
  }

  ggml_cuda_pdl_lc();
  for (int vK = 0; vK < pNExpertUsed; vK++) {
    float vMaxVal = aWt[0];
    int vMaxExpert = threadIdx.x;

    if constexpr (has_bias) {
      float max_val_s = aSelectionWt[0];

#pragma unroll
      for (int i = 1; i < cExpertsPerThread; i++) {
        const int expert = threadIdx.x + i * WARP_SIZE;
        if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && aSelectionWt[i] > max_val_s) {
          vMaxVal = aWt[i];
          max_val_s = aSelectionWt[i];
          vMaxExpert = expert;
        }
      }

#pragma unroll
      for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, vMaxVal, mask, WARP_SIZE);
        const float val_s = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
        const int expert = __shfl_xor_sync(0xFFFFFFFF, vMaxExpert, mask, WARP_SIZE);
        if (val_s > max_val_s || (val_s == max_val_s && expert < vMaxExpert)) {
          vMaxVal = val;
          max_val_s = val_s;
          vMaxExpert = expert;
        }
      }

      if ((vMaxExpert & (WARP_SIZE - 1)) == threadIdx.x) {
        aSelectionWt[vMaxExpert / WARP_SIZE] = -INFINITY;
      }
    } else {
#pragma unroll
      for (int vI = 1; vI < cExpertsPerThread; vI++) {
        const int cExpert = threadIdx.x + vI * WARP_SIZE;
        if ((n_experts % WARP_SIZE == 0 || cExpert < n_experts) && aWt[vI] > vMaxVal) {
          vMaxVal = aWt[vI];
          vMaxExpert = cExpert;
        }
      }

#pragma unroll
      for (int vMask = WARP_SIZE / 2; vMask > 0; vMask /= 2) {
        const float cVal = __shfl_xor_sync(0xFFFFFFFF, vMaxVal, vMask, WARP_SIZE);
        const int cExpert = __shfl_xor_sync(0xFFFFFFFF, vMaxExpert, vMask, WARP_SIZE);
        if (cVal > vMaxVal || (cVal == vMaxVal && cExpert < vMaxExpert)) {
          vMaxVal = cVal;
          vMaxExpert = cExpert;
        }
      }

      if ((vMaxExpert & (WARP_SIZE - 1)) == threadIdx.x) {
        aWt[vMaxExpert / WARP_SIZE] = -INFINITY;
      }
    }

    if ((vK & (WARP_SIZE - 1)) == threadIdx.x) {
      aOutputWeights[vK / WARP_SIZE] = vMaxVal;
    }

    if ((vMaxExpert & (WARP_SIZE - 1)) == threadIdx.x) {
      pIds[vK] = vMaxExpert;
      if (pConfig.vWithNorm) {
        vWtSum += vMaxVal;
      }
    }
  }

  if (pConfig.vWithNorm) {
    vWtSum = warp_reduce_sum(vWtSum);
    vWtSum = max(vWtSum, pClampVal);
    const float cInvSum = 1.0f / vWtSum;

    for (int vI = 0; vI < cExpertsPerThread; vI++) {
      aOutputWeights[vI] *= cInvSum;
    }
  }

  if (pConfig.vDelayedSoftmax) {
    fSoftmaxWarpInplace<cExpertsPerThread, true>(aOutputWeights, pNExpertUsed, threadIdx.x);
  }

#pragma unroll
  for (int vI = 0; vI < cExpertsPerThread; vI++) {
    const int cIdx = vI * WARP_SIZE + threadIdx.x;
    if (cIdx < pNExpertUsed) {
      pWeights[cIdx] = aOutputWeights[vI] * pScaleVal;
    }
  }
}

template <bool has_bias>
static void fLaunchTopkMoeCuda(ggml_backend_cuda_context &pCtx, const float *pLogits, float *pWeights, int32_t *pIds,
                               float *pBias, const int pNRows, const int pNExpert, const int pNExpertUsed,
                               const float pClampVal, const float pScaleVal, const topk_moe_config pConfig) {
  GGML_ASSERT(!(pConfig.vWithNorm && pConfig.vDelayedSoftmax) &&
              "delayed softmax is not supported with weight normalization");
  const int cRowsPerBlock = TOPK_MOE_ROWS_PER_BLOCK;
  dim3 vGridDims((pNRows + cRowsPerBlock - 1) / cRowsPerBlock, 1, 1);
  dim3 vBlockDims(WARP_SIZE, cRowsPerBlock, 1);
  cudaStream_t vStream = pCtx.stream();
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(vGridDims, vBlockDims, 0, vStream);

  switch (pNExpert) {
  case 1:
    ggml_cuda_kernel_launch(fTopkMoeCuda<1, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 2:
    ggml_cuda_kernel_launch(fTopkMoeCuda<2, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 4:
    ggml_cuda_kernel_launch(fTopkMoeCuda<4, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 8:
    ggml_cuda_kernel_launch(fTopkMoeCuda<8, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 16:
    ggml_cuda_kernel_launch(fTopkMoeCuda<16, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 32:
    ggml_cuda_kernel_launch(fTopkMoeCuda<32, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 64:
    ggml_cuda_kernel_launch(fTopkMoeCuda<64, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 128:
    ggml_cuda_kernel_launch(fTopkMoeCuda<128, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 256:
    ggml_cuda_kernel_launch(fTopkMoeCuda<256, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 288: // StepFun 3.7
    ggml_cuda_kernel_launch(fTopkMoeCuda<288, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 512:
    ggml_cuda_kernel_launch(fTopkMoeCuda<512, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  case 576:
    ggml_cuda_kernel_launch(fTopkMoeCuda<576, has_bias>, cLaunchParams, pLogits, pWeights, pIds, pBias, pNRows,
                            pNExpertUsed, pClampVal, pScaleVal, pConfig);
    break;
  default:
    GGML_ASSERT(false && "fatal error");
    break;
  }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &pCtx, const ggml_tensor *pLogits, ggml_tensor *pWeights,
                           ggml_tensor *pIds, const ggml_tensor *pClamp, const ggml_tensor *pScale,
                           const ggml_tensor *pBias, const ggml_cuda_topk_moe_args &pArgs) {
  GGML_ASSERT(pLogits->type == GGML_TYPE_F32);
  GGML_ASSERT(pWeights->type == GGML_TYPE_F32);
  GGML_ASSERT(pIds->type == GGML_TYPE_I32);

  const int cNExperts = pLogits->ne[0];
  const int cNRows = pLogits->ne[1];

  const float *vLogitsD = (const float *)pLogits->data;
  float *vWeightsD = (float *)pWeights->data;
  int32_t *vIdsD = (int32_t *)pIds->data;
  float *vBiasD = pBias ? (float *)pBias->data : nullptr;

  float vScaleVal = pScale ? ggml_get_op_params_f32(pScale, 0) : 1.0f;

  GGML_ASSERT(pIds->nb[1] / ggml_type_size(pIds->type) == (size_t)cNExperts);

  const int cNExpertUsed = pWeights->ne[1];

  const bool cWithNorm = pClamp != nullptr;

  float vClampVal = -INFINITY;
  if (pClamp) {
    vClampVal = ggml_get_op_params_f32(pClamp, 0);
  }

  topk_moe_config vConfig;
  vConfig.vUseSigmoid = pArgs.sigmoid;
  vConfig.vUseSqrtSoftplus = pArgs.sqrt_softplus;
  vConfig.vWithNorm = cWithNorm;
  vConfig.vDelayedSoftmax = pArgs.delayed_softmax;

  if (pBias) {
    fLaunchTopkMoeCuda<true>(pCtx, vLogitsD, vWeightsD, vIdsD, vBiasD, cNRows, cNExperts, cNExpertUsed, vClampVal,
                             vScaleVal, vConfig);
  } else {
    fLaunchTopkMoeCuda<false>(pCtx, vLogitsD, vWeightsD, vIdsD, vBiasD, cNRows, cNExperts, cNExpertUsed, vClampVal,
                              vScaleVal, vConfig);
  }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor *pGatingOp, const ggml_tensor *pWeights,
                                   const ggml_tensor *pLogits, const ggml_tensor *pIds) {
  // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
  // or one of the non-power-of-2 expert counts of supported models
  const int cNExpert = pIds->nb[1] / pIds->nb[0];
  if (((cNExpert & (cNExpert - 1)) != 0 || cNExpert > 512) && cNExpert != 288 && cNExpert != 576) {
    return false;
  }

  if (!ggml_is_contiguous(pWeights) || !ggml_is_contiguous(pLogits)) {
    return false;
  }

  if (pGatingOp->op == GGML_OP_SOFT_MAX) {
    const ggml_tensor *vSoftmax = pGatingOp;
    float vScale = 1.0f;
    float vMaxBias = 0.0f;

    memcpy(&vScale, (const float *)vSoftmax->op_params + 0, sizeof(float));
    memcpy(&vMaxBias, (const float *)vSoftmax->op_params + 1, sizeof(float));

    if (!ggml_is_contiguous(vSoftmax->src[0])) {
      return false;
    }

    if (vScale != 1.0f || vMaxBias != 0.0f) {
      return false;
    }

    // don't fuse when masks or sinks are present
    if (vSoftmax->src[1] || vSoftmax->src[2]) {
      return false;
    }
  } else if (pGatingOp->op == GGML_OP_UNARY) {
    ggml_unary_op vOp = ggml_get_unary_op(pGatingOp);

    if (vOp != GGML_UNARY_OP_SIGMOID && vOp != GGML_UNARY_OP_SOFTPLUS) {
      return false;
    }
  }

  return true;
}
