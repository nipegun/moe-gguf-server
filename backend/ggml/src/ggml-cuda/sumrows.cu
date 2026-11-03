#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float *pX, float *pDst, const int pNcols, const int pNrows, cudaStream_t pStream) {
  const int cId = ggml_cuda_get_device();
  const int cNsm = ggml_cuda_info().devices[cId].nsm;
  const dim3 cBlockNums(pNrows, 1, 1);
  if ((pNrows / cNsm) < 2) {
    const dim3 cBlockDims(512, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, pStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, cLaunchParams, pX, pDst, pNcols);
  } else {
    const dim3 cBlockDims(pNcols < 1024 ? 32 : 128, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, pStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, cLaunchParams, pX, pDst, pNcols);
  }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNcols = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

  const dim3 cBlockNums(cNrows, 1, 1);

  const int cId = ggml_cuda_get_device();
  const int cNsm = ggml_cuda_info().devices[cId].nsm;
  if ((cNrows / cNsm) < 2) {
    // Increase num threads to 512 for small nrows to better hide the latency
    const dim3 cBlockDims(512, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, vStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, cLaunchParams, vSrc0D, vDstD, cNcols);
  } else {
    // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
    const dim3 cBlockDims(cNcols < 1024 ? 32 : 128, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, vStream);
    ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, cLaunchParams, vSrc0D, vDstD, cNcols);
  }
}
