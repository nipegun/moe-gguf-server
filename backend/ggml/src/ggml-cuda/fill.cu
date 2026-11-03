#include "fill.cuh"
#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T> static __global__ void fFillKernel(T *pDst, const int64_t pK, const T pValue) {
  const int64_t cI = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (cI >= pK) {
    return;
  }
  pDst[cI] = pValue;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  void *vDstD = pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(pDst));

  float vValue;
  memcpy(&vValue, pDst->op_params, sizeof(float));

  const int64_t cK = ggml_nelements(pDst);
  const int64_t cNumBlocks = (cK + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

  switch (pDst->type) {
  case GGML_TYPE_F32:
    fFillKernel<<<cNumBlocks, CUDA_FILL_BLOCK_SIZE, 0, vStream>>>((float *)vDstD, cK, vValue);
    break;
  case GGML_TYPE_F16:
    fFillKernel<<<cNumBlocks, CUDA_FILL_BLOCK_SIZE, 0, vStream>>>((half *)vDstD, cK, ggml_cuda_cast<half>(vValue));
    break;
  default:
    GGML_ABORT("unsupported type");
  }
}
