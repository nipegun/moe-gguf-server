#include "ggml-impl.h"
#include "opt-step-adamw.cuh"

#include <cstdint>

static __global__ void fOptStepAdamwF32(float *__restrict__ pX, const float *__restrict__ pG, float *__restrict__ pGM,
                                        float *__restrict__ pGV, const float *__restrict__ pPars, const int64_t pK) {

  const int64_t cI = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  const float cAlpha = pPars[0];
  const float cBeta1 = pPars[1];
  const float cBeta2 = pPars[2];
  const float cEps = pPars[3];
  const float cWd = pPars[4];
  const float cBeta1h = pPars[5];
  const float cBeta2h = pPars[6];

  const float cGi = pG[cI];
  const float cGmi = pGM[cI] * cBeta1 + cGi * (1.0f - cBeta1);
  const float cGvi = pGV[cI] * cBeta2 + cGi * cGi * (1.0f - cBeta2);

  pGM[cI] = cGmi;
  pGV[cI] = cGvi;

  const float cMh = cGmi * cBeta1h;
  const float cVh = sqrtf(cGvi * cBeta2h) + cEps;

  pX[cI] = pX[cI] * (1.0f - cAlpha * cWd) - cAlpha * cMh / cVh;
}

static void fOptStepAdamwF32Cuda(float *pX, const float *pG, float *pGM, float *pGV, const float *pPars,
                                 const int64_t pK, cudaStream_t pStream) {

  const dim3 cBlockDims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
  const dim3 cBlockNums((pK + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
  fOptStepAdamwF32<<<cBlockNums, cBlockDims, 0, pStream>>>(pX, pG, pGM, pGV, pPars, pK);
}

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc0Grad = pDst->src[1];
  const ggml_tensor *vSrc0GradM = pDst->src[2];
  const ggml_tensor *vSrc0GradV = pDst->src[3];
  const ggml_tensor *vAdamwParams = pDst->src[4];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0Grad->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0GradM->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0GradV->type == GGML_TYPE_F32);
  GGML_ASSERT(vAdamwParams->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc0Grad));
  GGML_ASSERT(ggml_is_contiguous(vSrc0GradM));
  GGML_ASSERT(ggml_is_contiguous(vSrc0GradV));
  GGML_ASSERT(ggml_is_contiguous(vAdamwParams));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc0Grad));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc0GradM));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc0GradV));
  GGML_ASSERT(ggml_nelements(vAdamwParams) == 7);

  float *vSrc0D = (float *)vSrc0->data;
  const float *vSrc0GradD = (const float *)vSrc0Grad->data;
  float *vSrc0GradMD = (float *)vSrc0GradM->data;
  float *vSrc0GradVD = (float *)vSrc0GradV->data;
  const float *vAdamwParamsD = (const float *)vAdamwParams->data;

  cudaStream_t vStream = pCtx.stream();

  const int64_t cNe = ggml_nelements(vSrc0);

  fOptStepAdamwF32Cuda(vSrc0D, vSrc0GradD, vSrc0GradMD, vSrc0GradVD, vAdamwParamsD, cNe, vStream);
}
