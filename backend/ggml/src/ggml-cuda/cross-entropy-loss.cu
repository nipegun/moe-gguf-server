#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

template <bool use_shared>
static __global__ void fCrossEntropyLossF32(const float *__restrict__ pLogits, const float *__restrict__ pLabels,
                                            float *__restrict__ pDst, const int pNclasses, const int pK) {
  extern __shared__ float aTmp[];

  pLogits += int64_t(blockIdx.x) * pNclasses;
  pLabels += int64_t(blockIdx.x) * pNclasses;

  // Find maximum for softmax:
  float vMaxLogit = -INFINITY;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cVal = pLogits[vI];
    vMaxLogit = fmaxf(vMaxLogit, cVal);

    if (use_shared) {
      aTmp[vI] = cVal;
    }
  }
  vMaxLogit = warp_reduce_max(vMaxLogit);

  // Calculate log(softmax(logits)) which is just logits - max:
  float vSum = 0.0f;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cLogitI = use_shared ? aTmp[vI] : pLogits[vI];
    vSum += expf(cLogitI - vMaxLogit);
  }
  vSum = warp_reduce_sum(vSum);
  vSum = logf(vSum);

  // log(exp(logits - max) / sum) = (logits - max) - log(sum)
  float vLoss = 0.0f;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cLogitI = use_shared ? aTmp[vI] : pLogits[vI];
    vLoss += (cLogitI - vMaxLogit - vSum) * pLabels[vI];
  }
  vLoss = -warp_reduce_sum(vLoss) / (float)pK;

  if (threadIdx.x != 0) {
    return;
  }

  pDst[blockIdx.x] = vLoss;
}

template <bool use_shared>
static __global__ void fCrossEntropyLossBackF32(const float *__restrict__ pGrad, const float *__restrict__ pLogits,
                                                const float *__restrict__ pLabels, float *__restrict__ pDst,
                                                const int pNclasses) {
  extern __shared__ float aTmp[];

  pLogits += int64_t(blockIdx.x) * pNclasses;
  pLabels += int64_t(blockIdx.x) * pNclasses;
  pDst += int64_t(blockIdx.x) * pNclasses;

  float vMaxval = -INFINITY;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cVal = pLogits[vI];
    vMaxval = fmaxf(vMaxval, cVal);

    if (use_shared) {
      aTmp[vI] = cVal;
    }
  }
  vMaxval = warp_reduce_max(vMaxval);

  float vSum = 0.0f;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cVal = expf((use_shared ? aTmp[vI] : pLogits[vI]) - vMaxval);
    vSum += cVal;

    if (use_shared) {
      aTmp[vI] = cVal;
    } else {
      pDst[vI] = cVal;
    }
  }
  vSum = warp_reduce_sum(vSum);
  const float cSmScale = 1.0f / vSum;

  const float cDByNrows = *pGrad / gridDim.x;
  for (int vI = threadIdx.x; vI < pNclasses; vI += WARP_SIZE) {
    const float cVal = use_shared ? aTmp[vI] : pDst[vI];
    pDst[vI] = (cVal * cSmScale - pLabels[vI]) * cDByNrows;
  }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc1));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  float *vDstD = (float *)pDst->data;

  ggml_cuda_pool &vPool = pCtx.pool();
  cudaStream_t vStream = pCtx.stream();

  const dim3 cBlocksDim(WARP_SIZE, 1, 1);
  const dim3 cBlocksNum(cNrows, 1, 1);
  const size_t cNbytesShared = cNe00 * sizeof(float);

  const int cId = ggml_cuda_get_device();
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;

  ggml_cuda_pool_alloc<float> vDstTmp(vPool, cBlocksNum.x);

  if (cNbytesShared <= cSmpbo) {
    CUDA_SET_SHARED_MEMORY_LIMIT((fCrossEntropyLossF32<true>), cSmpbo);
    fCrossEntropyLossF32<true>
        <<<cBlocksNum, cBlocksDim, cNbytesShared, vStream>>>(vSrc0D, vSrc1D, vDstTmp.ptr, cNe00, cNrows);
  } else {
    fCrossEntropyLossF32<false><<<cBlocksNum, cBlocksDim, 0, vStream>>>(vSrc0D, vSrc1D, vDstTmp.ptr, cNe00, cNrows);
  }
  CUDA_CHECK(cudaGetLastError());

  // Combine results from individual blocks:
  sum_f32_cuda(vPool, vDstTmp.ptr, vDstD, cBlocksNum.x, vStream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vGrad = pDst->src[0];
  const ggml_tensor *vSrc0f = pDst->src[1];
  const ggml_tensor *vSrc1f = pDst->src[2];

  GGML_ASSERT(vSrc0f->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc1f->type == GGML_TYPE_F32);
  GGML_ASSERT(vGrad->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_scalar(vGrad));
  GGML_ASSERT(ggml_is_contiguous(vSrc0f));
  GGML_ASSERT(ggml_is_contiguous(vSrc1f));
  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_are_same_shape(vSrc0f, vSrc1f));
  GGML_ASSERT(ggml_are_same_shape(vSrc0f, pDst));

  const int64_t cNe00 = vSrc0f->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0f);

  const float *vGradD = (const float *)vGrad->data;
  const float *vSrc0fD = (const float *)vSrc0f->data;
  const float *vSrc1fD = (const float *)vSrc1f->data;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  const dim3 cBlocksDim(WARP_SIZE, 1, 1);
  const dim3 cBlocksNum(cNrows, 1, 1);
  const size_t cNbytesShared = cNe00 * sizeof(float);

  const int cId = ggml_cuda_get_device();
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;

  if (cNbytesShared <= cSmpbo) {
    CUDA_SET_SHARED_MEMORY_LIMIT((fCrossEntropyLossBackF32<true>), cSmpbo);
    fCrossEntropyLossBackF32<true>
        <<<cBlocksNum, cBlocksDim, cNbytesShared, vStream>>>(vGradD, vSrc0fD, vSrc1fD, vDstD, cNe00);
  } else {
    fCrossEntropyLossBackF32<false><<<cBlocksNum, cBlocksDim, 0, vStream>>>(vGradD, vSrc0fD, vSrc1fD, vDstD, cNe00);
  }
}
