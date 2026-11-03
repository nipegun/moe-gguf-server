#include "ggml-impl.h"
#include "opt-step-sgd.cuh"

#include <cstdint>

static __global__ void fOptStepSgdF32(float *__restrict__ pX, const float *__restrict__ pG,
                                      const float *__restrict__ pPars, const int64_t pK) {

  const int64_t cI = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }
  pX[cI] = pX[cI] * (1.0f - pPars[0] * pPars[1]) - pPars[0] * pG[cI];
}

static void fOptStepSgdF32Cuda(float *pX, const float *pG, const float *__restrict__ pPars, const int64_t pK,
                               cudaStream_t pStream) {

  const dim3 cBlockDims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
  const dim3 cBlockNums((pK + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
  fOptStepSgdF32<<<cBlockNums, cBlockDims, 0, pStream>>>(pX, pG, pPars, pK);
}

void ggml_cuda_opt_step_sgd(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc0Grad = pDst->src[1];
  const ggml_tensor *vParams = pDst->src[2];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0Grad->type == GGML_TYPE_F32);
  GGML_ASSERT(vParams->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc0Grad));
  GGML_ASSERT(ggml_is_contiguous(vParams));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc0Grad));
  GGML_ASSERT(ggml_nelements(vParams) == 2);

  float *vSrc0D = (float *)vSrc0->data;
  const float *vSrc0GradD = (const float *)vSrc0Grad->data;
  const float *vParamsD = (const float *)vParams->data;

  cudaStream_t vStream = pCtx.stream();

  const int64_t cNe = ggml_nelements(vSrc0);

  fOptStepSgdF32Cuda(vSrc0D, vSrc0GradD, vParamsD, cNe, vStream);
}
