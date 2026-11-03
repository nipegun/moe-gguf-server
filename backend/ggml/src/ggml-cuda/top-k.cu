#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
#if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#define CUB_TOP_K_AVAILABLE
#include <cuda/iterator>
using namespace cub;
#endif // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void fTopKCub(ggml_cuda_pool &pPool, const float *pSrc, int *pDst, const int pNcols, const int pK,
                     cudaStream_t pStream) {
  auto vRequirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                cuda::execution::output_ordering::unsorted);
  auto vStreamEnv = cuda::stream_ref{pStream};
  auto vEnv = cuda::std::execution::env{vStreamEnv, vRequirements};

  auto vIndexesIn = cuda::make_counting_iterator(0);

  size_t vTempStorageBytes = 0;
  CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, vTempStorageBytes, pSrc, cuda::discard_iterator(), vIndexesIn, pDst, pNcols,
                                  pK, vEnv));

  ggml_cuda_pool_alloc<uint8_t> vTempStorageAlloc(pPool, vTempStorageBytes);
  void *vDTempStorage = vTempStorageAlloc.get();

  CUDA_CHECK(DeviceTopK::MaxPairs(vDTempStorage, vTempStorageBytes, pSrc, cuda::discard_iterator(), vIndexesIn, pDst,
                                  pNcols, pK, vEnv));
}

#elif defined(GGML_CUDA_USE_CUB) // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
  int n = 1;
  while (n < x) {
    n *= 2;
  }
  return n;
}

#endif // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  int *vDstD = (int *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  // are these asserts truly necessary?
  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_I32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNcols = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);
  const int64_t cK = pDst->ne[0];
  ggml_cuda_pool &vPool = pCtx.pool();
#ifdef CUB_TOP_K_AVAILABLE
  // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
  // https://github.com/NVIDIA/cccl/issues/6391
  // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
  for (int vI = 0; vI < cNrows; vI++) {
    fTopKCub(vPool, vSrc0D + vI * cNcols, vDstD + vI * cK, cNcols, cK, vStream);
  }
#elif defined(GGML_CUDA_USE_CUB) // CUB_TOP_K_AVAILABLE
  // Fall back to argsort + copy
  const int ncols_pad = next_power_of_2(cNcols);
  const size_t shared_mem = ncols_pad * sizeof(int);
  const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
  const bool use_bitonic = shared_mem <= max_shared_mem && cNcols <= 1024;
  const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(vSrc0->nb[1], cNrows);

  ggml_cuda_pool_alloc<int> temp_dst_alloc(vPool, cNcols * chunk_nrows);
  int *tmp_dst = temp_dst_alloc.get();

  for (int64_t vI = 0; vI < cNrows; vI += chunk_nrows) {
    int iter_nrows = std::min((int64_t)chunk_nrows, cNrows - vI);

    if (use_bitonic) {
      argsort_f32_i32_cuda_bitonic(vSrc0D, tmp_dst, cNcols, iter_nrows, GGML_SORT_ORDER_DESC, vStream);
    } else {
      argsort_f32_i32_cuda_cub(vPool, vSrc0D, tmp_dst, cNcols, iter_nrows, GGML_SORT_ORDER_DESC, vStream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(vDstD, cK * sizeof(int), tmp_dst, cNcols * sizeof(int), cK * sizeof(int), iter_nrows,
                                 cudaMemcpyDeviceToDevice, vStream));

    vSrc0D += cNcols * iter_nrows;
    vDstD += cK * iter_nrows;
  }
#else                            // GGML_CUDA_USE_CUB
  ggml_cuda_pool_alloc<int> temp_dst_alloc(vPool, cNcols * cNrows);
  int *tmp_dst = temp_dst_alloc.get();
  argsort_f32_i32_cuda_bitonic(vSrc0D, tmp_dst, cNcols, cNrows, GGML_SORT_ORDER_DESC, vStream);
  CUDA_CHECK(cudaMemcpy2DAsync(vDstD, cK * sizeof(int), tmp_dst, cNcols * sizeof(int), cK * sizeof(int), cNrows,
                               cudaMemcpyDeviceToDevice, vStream));
#endif
}
