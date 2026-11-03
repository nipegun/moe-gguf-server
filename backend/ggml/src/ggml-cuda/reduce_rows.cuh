#include "common.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm> static __global__ void reduce_rows_f32(const float *pXPtr, float *pDstPtr, const int pNcols) {
  const float *GGML_CUDA_RESTRICT vX = pXPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const int cRow = blockIdx.x;
  const int cCol = threadIdx.x;

  float vSum = 0.0f;
  const int cNumUnroll = 8;
  float vTemp[cNumUnroll];
  float vSumTemp[cNumUnroll] = {0.0f};

  ggml_cuda_pdl_sync();
  for (int vI = cCol; vI < pNcols;) {
    for (int vJ = 0; vJ < cNumUnroll; ++vJ) {
      if (vI < pNcols) {
        vTemp[vJ] = vX[cRow * pNcols + vI];
      } else {
        vTemp[vJ] = 0;
      }
      vI += blockDim.x;
    }
    for (int vJ = 0; vJ < cNumUnroll; ++vJ) {
      vSumTemp[vJ] += vTemp[vJ];
    }
  }
  for (int vJ = 0; vJ < cNumUnroll; ++vJ) {
    vSum += vSumTemp[vJ];
  }

  // sum up partial sums
  __shared__ float vSharedVals[32];
  vSum = block_reduce<block_reduce_method::SUM>(vSum, vSharedVals);

  if (cCol != 0) {
    return;
  }

  vDst[cRow] = norm ? vSum / pNcols : vSum;
}
