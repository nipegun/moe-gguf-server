#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void fScaleF32(const float *pX, float *pDst, const float pScale, const float pBias,
                                 const int64_t pNelements) {
  ggml_cuda_pdl_lc();
  int64_t vTid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t vStride = (int64_t)blockDim.x * (int64_t)gridDim.x;

  ggml_cuda_pdl_sync();
  for (int64_t vI = vTid; vI < pNelements; vI += vStride) {
    pDst[vI] = pScale * pX[vI] + pBias;
  }
}

static void fScaleF32Cuda(const float *pX, float *pDst, const float pScale, const float pBias, const int64_t pNelements,
                          cudaStream_t pStream) {
  const int64_t cNumBlocks = (pNelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, cNumBlocks), CUDA_SCALE_BLOCK_SIZE, 0, pStream);
  ggml_cuda_kernel_launch(fScaleF32, cLaunchParams, pX, pDst, pScale, pBias, pNelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  float vScale;
  float vBias;
  memcpy(&vScale, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vBias, (float *)pDst->op_params + 1, sizeof(float));

  fScaleF32Cuda(vSrc0D, vDstD, vScale, vBias, ggml_nelements(vSrc0), vStream);
}
