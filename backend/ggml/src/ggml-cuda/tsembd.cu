#include "tsembd.cuh"

static __global__ void fTimestepEmbeddingF32(const float *pTimesteps, float *pDst, const int pNb1, const int pDim,
                                             const int pMaxPeriod) {
  // blockIDx.y: idx of timesteps->ne[0]
  // blockIDx.x: idx of ((dim + 1) / 2) / BLOCK_SIZE
  int vI = blockIdx.y;
  int vJ = threadIdx.x + blockIdx.x * blockDim.x;
  float *vEmbedData = (float *)((char *)pDst + vI * pNb1);

  int vHalf = pDim / 2;
  if (pDim % 2 != 0 && vJ == vHalf) {
    vEmbedData[2 * vHalf] = 0.f;
  }

  if (vJ >= vHalf) {
    return;
  }

  float vTimestep = pTimesteps[vI];
  float vFreq = (float)expf(-logf(pMaxPeriod) * vJ / vHalf);
  float vArg = vTimestep * vFreq;
  vEmbedData[vJ] = cosf(vArg);
  vEmbedData[vJ + vHalf] = sinf(vArg);
}

static void fTimestepEmbeddingF32Cuda(const float *pX, float *pDst, const int pNe00, const int pNb1, const int pDim,
                                      const int pMaxPeriod, cudaStream_t pStream) {
  int vHalfCeil = (pDim + 1) / 2;
  int vNumBlocks = (vHalfCeil + CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE;
  dim3 vGridDim(vNumBlocks, pNe00, 1);
  fTimestepEmbeddingF32<<<vGridDim, CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE, 0, pStream>>>(pX, pDst, pNb1, pDim, pMaxPeriod);
}

void ggml_cuda_op_timestep_embedding(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int cDim = pDst->op_params[0];
  const int cMaxPeriod = pDst->op_params[1];

  fTimestepEmbeddingF32Cuda(vSrc0D, vDstD, vSrc0->ne[0], pDst->nb[1], cDim, cMaxPeriod, vStream);
}
