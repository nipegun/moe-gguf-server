#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void fArgmaxF32(const float *__restrict__ pX, int32_t *__restrict__ pDst, const int64_t pNcols) {
  const int64_t cRow = blockIdx.x;

  float vMaxval = -FLT_MAX;
  int vArgmax = -1;
  const float *vRowx = pX + cRow * pNcols;

  for (int32_t vCol = threadIdx.x; vCol < pNcols; vCol += blockDim.x) {
    const float cVal = vRowx[vCol];
    if (cVal > vMaxval) {
      vMaxval = cVal;
      vArgmax = vCol;
    }
  }

#pragma unroll
  for (int vOffset = WARP_SIZE / 2; vOffset > 0; vOffset >>= 1) {
    const float cVal = __shfl_xor_sync(0xFFFFFFFF, vMaxval, vOffset, WARP_SIZE);
    const int cCol = __shfl_xor_sync(0xFFFFFFFF, vArgmax, vOffset, WARP_SIZE);
    if (cVal > vMaxval) {
      vMaxval = cVal;
      vArgmax = cCol;
    }
  }

  const int cNWarps = blockDim.x / WARP_SIZE;
  const int cLaneId = threadIdx.x % WARP_SIZE;
  const int cWarpId = threadIdx.x / WARP_SIZE;
  if (cNWarps > 1) {
    constexpr int cMaxWarps = 1024 / WARP_SIZE;
    __shared__ float vSharedMaxval[cMaxWarps];
    __shared__ int vSharedArgmax[cMaxWarps];
    if (cLaneId == 0) {
      vSharedMaxval[cWarpId] = vMaxval;
      vSharedArgmax[cWarpId] = vArgmax;
    }

    __syncthreads();

    if (cWarpId == 0) {
      if (cLaneId < cNWarps) {
        vMaxval = vSharedMaxval[cLaneId];
        vArgmax = vSharedArgmax[cLaneId];
      }
#pragma unroll
      for (int vOffset = WARP_SIZE / 2; vOffset > 0; vOffset >>= 1) {
        const float cVal = __shfl_xor_sync(0xFFFFFFFF, vMaxval, vOffset, WARP_SIZE);
        const int cCol = __shfl_xor_sync(0xFFFFFFFF, vArgmax, vOffset, WARP_SIZE);
        if (cVal > vMaxval) {
          vMaxval = cVal;
          vArgmax = cCol;
        }
      }
    }
  }

  if (cWarpId == 0 && cLaneId == 0) {
    pDst[cRow] = vArgmax;
  }
}

void ggml_cuda_argmax(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_I32);

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

  const float *vSrc0D = (const float *)vSrc0->data;
  int32_t *vDstD = (int32_t *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  const int64_t cNumBlocks = cNrows;
  const int64_t cNumThreads = std::min<int64_t>(1024, (cNe00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
  const dim3 cBlocksDim(cNumThreads, 1, 1);
  const dim3 cBlocksNum(cNumBlocks, 1, 1);

  fArgmaxF32<<<cBlocksNum, cBlocksDim, 0, vStream>>>(vSrc0D, vDstD, cNe00);
}
