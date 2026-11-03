#include "acc.cuh"

static __global__ void fAccF32(const float *pX, const float *pY, float *pDst, const int64_t pNe, const int64_t pNe10,
                               const int64_t pNe11, const int64_t pNe12, const int64_t pNe13, const int64_t pS11,
                               const int64_t pS12, const int64_t pS13, const int64_t pOffset) {
  const int64_t cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pNe) {
    return;
  }

  int64_t vSrc1Idx = cI - pOffset;

  int64_t vTmp = vSrc1Idx;
  const int64_t cI13 = vTmp / pS13;
  vTmp -= cI13 * pS13;
  const int64_t cI12 = vTmp / pS12;
  vTmp -= cI12 * pS12;
  const int64_t cI11 = vTmp / pS11;
  vTmp -= cI11 * pS11;
  const int64_t cI10 = vTmp;

  float vVal = pX[cI];
  if (vSrc1Idx >= 0 && cI10 < pNe10 && cI11 < pNe11 && cI12 < pNe12 && cI13 < pNe13) {
    vVal += pY[((cI13 * pNe12 + cI12) * pNe11 + cI11) * pNe10 + cI10];
  }
  pDst[cI] = vVal;
}

static void fAccF32Cuda(const float *pX, const float *pY, float *pDst, const int64_t pNElements, const int64_t pNe10,
                        const int64_t pNe11, const int64_t pNe12, const int64_t pNe13, const int64_t pS1,
                        const int64_t pS2, const int64_t pS3, const int64_t pOffset, cudaStream_t pStream) {
  const int cNumBlocks = (pNElements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
  fAccF32<<<cNumBlocks, CUDA_ACC_BLOCK_SIZE, 0, pStream>>>(pX, pY, pDst, pNElements, pNe10, pNe11, pNe12, pNe13, pS1,
                                                           pS2, pS3, pOffset);
}

void ggml_cuda_op_acc(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_contiguous(vSrc1));
  GGML_ASSERT(pDst->nb[0] == ggml_element_size(pDst));
  GGML_ASSERT(ggml_is_contiguously_allocated(pDst));

  const int64_t cS1 = pDst->op_params[0] / sizeof(float);
  const int64_t cS2 = pDst->op_params[1] / sizeof(float);
  const int64_t cS3 = pDst->op_params[2] / sizeof(float);
  const int64_t cOffset = pDst->op_params[3] / sizeof(float);

  fAccF32Cuda(vSrc0D, vSrc1D, vDstD, ggml_nelements(pDst), vSrc1->ne[0], vSrc1->ne[1], vSrc1->ne[2], vSrc1->ne[3], cS1,
              cS2, cS3, cOffset, vStream);
}
