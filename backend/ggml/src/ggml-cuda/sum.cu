#include "sum.cuh"
#include "sumrows.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // GGML_CUDA_USE_CUB

#include <cstdint>

void sum_f32_cuda(ggml_cuda_pool &pPool, const float *pX, float *pDst, const int64_t pNe, cudaStream_t pStream) {
#ifdef GGML_CUDA_USE_CUB
  size_t vTmpSize = 0;
  DeviceReduce::Sum(nullptr, vTmpSize, pX, pDst, pNe, pStream);
  ggml_cuda_pool_alloc<uint8_t> vTmpAlloc(pPool, vTmpSize);
  DeviceReduce::Sum(vTmpAlloc.ptr, vTmpSize, pX, pDst, pNe, pStream);
#else
  // Use (inefficient) sum_rows implementation as a fallback.
  // For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 ->
  // C++14.
  sum_rows_f32_cuda(pX, pDst, pNe, 1, pStream);
  GGML_UNUSED(pPool);
#endif // GGML_CUDA_USE_CUB
}

void ggml_cuda_op_sum(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguously_allocated(vSrc0));

  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;

  const int64_t cNe = ggml_nelements(vSrc0);

  ggml_cuda_pool &vPool = pCtx.pool();
  cudaStream_t vStream = pCtx.stream();

  sum_f32_cuda(vPool, vSrc0D, vDstD, cNe, vStream);
}
