#include "diagmask.cuh"

static __global__ void fDiagMaskInfF32(const float *pX, float *pDst, const int pNcols, const int pRowsPerChannel,
                                       const int pNPast) {
  const int cCol = blockDim.y * blockIdx.y + threadIdx.y;
  const int cRow = blockDim.x * blockIdx.x + threadIdx.x;

  if (cCol >= pNcols) {
    return;
  }

  const int cI = cRow * pNcols + cCol;
  // dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
  // dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly
  // faster on GPU
  pDst[cI] = pX[cI] - (cCol > pNPast + cRow % pRowsPerChannel) * FLT_MAX;
}

static void fDiagMaskInfF32Cuda(const float *pX, float *pDst, const int pNcolsX, const int pNrowsX,
                                const int pRowsPerChannel, const int pNPast, cudaStream_t pStream) {
  const dim3 cBlockDims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
  const int cBlockNumX = (pNcolsX + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
  const dim3 cBlockNums(pNrowsX, cBlockNumX, 1);
  fDiagMaskInfF32<<<cBlockNums, cBlockDims, 0, pStream>>>(pX, pDst, pNcolsX, pRowsPerChannel, pNPast);
}

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNe01 = vSrc0->ne[1];
  const int cNrows0 = ggml_nrows(vSrc0);

  const int cNPast = ((int32_t *)pDst->op_params)[0];

  fDiagMaskInfF32Cuda(vSrc0D, vDstD, cNe00, cNrows0, cNe01, cNPast, vStream);
}
