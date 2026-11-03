#include "mean.cuh"
#include "reduce_rows.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // GGML_CUDA_USE_CUB

template <typename T> __global__ void fDivideByCount(T *pResult, size_t pCount) { *pResult /= static_cast<T>(pCount); }

void ggml_cuda_op_mean(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNcols = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

// Special case for reducing vectors
#ifdef GGML_CUDA_USE_CUB
#ifdef USE_CUDA_GRAPH
  cudaStreamCaptureStatus vIscapturing;
  CUDA_CHECK(cudaStreamIsCapturing(vStream, &vIscapturing));
#endif // USE_CUDA_GRAPH
  if ((cNrows == 1) &&
#ifdef USE_CUDA_GRAPH
      // Determine if CUDA graphs are effectively disabled for this context
      // (no graph instance exists and we're not capturing, OR graphs are explicitly enabled)
      (((cNcols > 65536) && (((!pCtx.any_cuda_graph_has_instance()) && (vIscapturing == cudaStreamCaptureStatusNone)) ||
                             pCtx.any_cuda_graph_enabled())) ||
       // CUDA graphs are enabled - use lower threshold
       ((cNcols > 32768) &&
        !(((!pCtx.any_cuda_graph_has_instance()) && (vIscapturing == cudaStreamCaptureStatusNone)) ||
          pCtx.any_cuda_graph_enabled())))) {
#else
      (cNcols > 65536)) {
#endif // USE_CUDA_GRAPH
       // Single row - use device-wide reduction
    size_t vTmpSize = 0;
    ggml_cuda_pool &vPool = pCtx.pool();

    DeviceReduce::Sum(nullptr, vTmpSize, vSrc0D, vDstD, cNcols, vStream);

    ggml_cuda_pool_alloc<uint8_t> vTmpAlloc(vPool, vTmpSize);
    DeviceReduce::Sum(vTmpAlloc.ptr, vTmpSize, vSrc0D, vDstD, cNcols, vStream);

    // Divide by ncols
    fDivideByCount<float><<<1, 1, 0, vStream>>>(vDstD, cNcols);
    return;
  }
#endif // GGML_CUDA_USE_CUB

  const dim3 cBlockNums(cNrows, 1, 1);

  const int cId = ggml_cuda_get_device();
  const int cNsm = ggml_cuda_info().devices[cId].nsm;

  // Heuristic for block size selection to optimize occupancy.
  if ((cNrows / cNsm) < 2) {
    const dim3 cBlockDims(512, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, vStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/true>, cLaunchParams, vSrc0D, vDstD, cNcols);
  } else {
    const dim3 cBlockDims(cNcols < 1024 ? 32 : 128, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, vStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/true>, cLaunchParams, vSrc0D, vDstD, cNcols);
  }
}
