#include "softcap.cuh"

static __global__ void fSoftcapF32(const float *pX, float *pDst, const float pScale, const float pSoftcap,
                                   const int pK) {
  ggml_cuda_pdl_lc();
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  ggml_cuda_pdl_sync();
  pDst[cI] = tanhf(pScale * pX[cI]) * pSoftcap;
}

static void fSoftcapF32Cuda(const float *pX, float *pDst, const float pScale, const float pSoftcap, const int pK,
                            cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cNumBlocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, pStream);
  ggml_cuda_kernel_launch(fSoftcapF32, cLaunchParams, pX, pDst, pScale, pSoftcap, pK);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pSrc) {
  const ggml_tensor *vSrc0 = pSrc->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  float vScale;
  float vSoftcap;
  memcpy(&vScale, (float *)pSrc->op_params + 0, sizeof(float));
  memcpy(&vSoftcap, (float *)pDst->op_params + 0, sizeof(float));

  fSoftcapF32Cuda(vSrc0D, vDstD, vScale, vSoftcap, ggml_nelements(vSrc0), vStream);
}
