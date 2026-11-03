#include "arange.cuh"

static __global__ void fArangeF32(float *pDst, const int pNe0, const float pStart, const float pStep) {
  // blockIDx.x: idx of ne0 / BLOCK_SIZE
  int vNidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vNidx >= pNe0) {
    return;
  }
  pDst[vNidx] = pStart + pStep * vNidx;
}

static void fArangeF32Cuda(float *pDst, const int pNe0, const float pStart, const float pStep, cudaStream_t pStream) {
  int vNumBlocks = (pNe0 + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
  fArangeF32<<<vNumBlocks, CUDA_ARANGE_BLOCK_SIZE, 0, pStream>>>(pDst, pNe0, pStart, pStep);
}

void ggml_cuda_op_arange(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  float vStart;
  float vStop;
  float vStep;
  memcpy(&vStart, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vStop, (float *)pDst->op_params + 1, sizeof(float));
  memcpy(&vStep, (float *)pDst->op_params + 2, sizeof(float));

  int64_t vSteps = (int64_t)ceil((vStop - vStart) / vStep);
  GGML_ASSERT(ggml_nelements(pDst) == vSteps);

  fArangeF32Cuda(vDstD, pDst->ne[0], vStart, vStep, vStream);
}
