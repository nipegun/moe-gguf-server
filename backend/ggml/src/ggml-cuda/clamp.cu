#include "clamp.cuh"

static __device__ __forceinline__ float fOpClamp(float pX, float pMin, float pMax) {
  return fminf(fmaxf(pX, pMin), pMax);
}

template <class T>
static __global__ void fOpClampKernel(const T *pX, T *pDst, const T pMin, const T pMax, const int pK) {
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  pDst[cI] = (T)fOpClamp((float)pX[cI], (float)pMin, (float)pMax);
}

template <class T>
static void fClampCuda(const T *pX, T *pDst, const T pMin, const T pMax, const int pK, cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
  fOpClampKernel<<<cNumBlocks, CUDA_CLAMP_BLOCK_SIZE, 0, pStream>>>(pX, pDst, pMin, pMax, pK);
}

void ggml_cuda_op_clamp(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const void *vSrc0D = vSrc0->data;
  void *vDstD = pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);

  float vMin;
  float vMax;
  memcpy(&vMin, pDst->op_params, sizeof(float));
  memcpy(&vMax, (float *)pDst->op_params + 1, sizeof(float));

  if (vSrc0->type == GGML_TYPE_F16) {
    fClampCuda((const half *)vSrc0D, (half *)vDstD, (half)vMin, (half)vMax, ggml_nelements(vSrc0), vStream);
  } else {
    fClampCuda((const float *)vSrc0D, (float *)vDstD, (float)vMin, (float)vMax, ggml_nelements(vSrc0), vStream);
  }
}
