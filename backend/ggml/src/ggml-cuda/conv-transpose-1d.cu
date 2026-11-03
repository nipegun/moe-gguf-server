#include "conv-transpose-1d.cuh"

static __global__ void fConvTranspose1dKernel(const int pS0, const int p0, const int pD0, const int pOutputSize,
                                              const int pSrc0Ne0, const int pSrc0Ne1, const int pSrc0Ne2,
                                              const int pSrc0Ne3, const int pSrc1Ne0, const int pSrc1Ne1,
                                              const int pSrc1Ne2, const int pSrc1Ne3, const int pDstNe0,
                                              const int pDstNe1, const int pDstNe2, const int pDstNe3,
                                              const float *pSrc0, const float *pSrc1, float *pDst) {
  int vGlobalIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (vGlobalIndex >= pOutputSize) {
    return;
  }

  int vOutT = vGlobalIndex % pDstNe0;
  int vOutCh = (vGlobalIndex / pDstNe0) % pDstNe1;
  int vPlane = vGlobalIndex / (pDstNe0 * pDstNe1);

  float vAccumulator = 0;

  for (int vC = 0; vC < pSrc0Ne2; vC++) {
    int vKernelOffset = pSrc0Ne0 * (vOutCh + pSrc0Ne1 * vC);
    int vInputOffset = pSrc1Ne0 * (vC + pSrc1Ne1 * vPlane);

    for (int vK = 0; vK < pSrc0Ne0; vK++) {
      int vInputNumer = vOutT + p0 - vK * pD0;
      if (vInputNumer < 0 || vInputNumer % pS0 != 0) {
        continue;
      }

      int vInputT = vInputNumer / pS0;
      if (vInputT >= pSrc1Ne0) {
        continue;
      }

      vAccumulator += pSrc0[vKernelOffset + vK] * pSrc1[vInputOffset + vInputT];
    }
  }
  pDst[vGlobalIndex] = vAccumulator;
  GGML_UNUSED_VARS(pSrc0Ne3, pSrc1Ne2, pSrc1Ne3, pDstNe2, pDstNe3);
}

static void fConvTranspose1dF32F32Cuda(const int pS0, const int p0, const int pD0, const int pOutputSize,
                                       const int pSrc0Ne0, const int pSrc0Ne1, const int pSrc0Ne2, const int pSrc0Ne3,
                                       const int pSrc1Ne0, const int pSrc1Ne1, const int pSrc1Ne2, const int pSrc1Ne3,
                                       const int pDstNe0, const int pDstNe1, const int pDstNe2, const int pDstNe3,
                                       const float *pSrc0, const float *pSrc1, float *pDst, cudaStream_t pStream) {

  const int cNumBlocks = (pOutputSize + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
  fConvTranspose1dKernel<<<cNumBlocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, pStream>>>(
      pS0, p0, pD0, pOutputSize, pSrc0Ne0, pSrc0Ne1, pSrc0Ne2, pSrc0Ne3, pSrc1Ne0, pSrc1Ne1, pSrc1Ne2, pSrc1Ne3,
      pDstNe0, pDstNe1, pDstNe2, pDstNe3, pSrc0, pSrc1, pDst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;

  const ggml_tensor *vSrc1 = pDst->src[1];
  const float *vSrc1D = (const float *)vSrc1->data;

  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc1));

  const int32_t *vOpts = (const int32_t *)pDst->op_params;

  const int cS0 = vOpts[0];
  const int cP0 = 0; // opts[3];
  const int cD0 = 1; // opts[4];

  const int64_t cOutputSize = ggml_nelements(pDst);

  fConvTranspose1dF32F32Cuda(cS0, cP0, cD0, cOutputSize, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2], vSrc0->ne[3],
                             vSrc1->ne[0], vSrc1->ne[1], vSrc1->ne[2], vSrc1->ne[3], pDst->ne[0], pDst->ne[1],
                             pDst->ne[2], pDst->ne[3], vSrc0D, vSrc1D, vDstD, vStream);
}
