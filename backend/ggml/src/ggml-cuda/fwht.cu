#include "common.cuh"
#include "fwht.cuh"

template <int N>
__launch_bounds__(4 * ggml_cuda_get_physical_warp_size(), 1) __global__
    void fFwhtCuda(const float *pSrc, float *pDst, const int64_t pNRows, const float pScale) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();

  const int64_t cR = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;

  if (cR >= pNRows) {
    return;
  }

  pSrc += cR * N;
  pDst += cR * N;

  static constexpr int cElW = N / cWarpSize;
  float aReg[cElW];
  const int cLane = threadIdx.x;

  ggml_cuda_pdl_sync();
#pragma unroll
  for (int vI = 0; vI < cElW; ++vI) {
    aReg[vI] = pSrc[vI * cWarpSize + cLane] * pScale;
  }

#pragma unroll
  for (int vH = 1; vH < cWarpSize; vH *= 2) {
#pragma unroll
    for (int vJ = 0; vJ < cElW; vJ++) {
      const float cVal = aReg[vJ];
      const float cVal2 = __shfl_xor_sync(0xFFFFFFFF, cVal, vH, cWarpSize);

      aReg[vJ] = (cLane & vH) == 0 ? cVal + cVal2 : cVal2 - cVal;
    }
  }

#pragma unroll
  for (int vH = cWarpSize; vH < N; vH *= 2) {
    const int cStep = vH / cWarpSize;
#pragma unroll
    for (int vJ = 0; vJ < cElW; vJ += 2 * cStep) {
#pragma unroll
      for (int vK = 0; vK < cStep; vK++) {
        const float cX = aReg[vJ + vK];
        const float cY = aReg[vJ + vK + cStep];

        aReg[vJ + vK] = cX + cY;
        aReg[vJ + vK + cStep] = cX - cY;
      }
    }
  }

#pragma unroll
  for (int vI = 0; vI < cElW; ++vI) {
    pDst[vI * cWarpSize + cLane] = aReg[vI];
  }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc, ggml_tensor *pDst) {
  GGML_ASSERT(ggml_are_same_shape(pSrc, pDst));
  if (!ggml_is_contiguous(pSrc) || !ggml_is_contiguous(pDst)) {
    return false;
  }
  const int cN = pSrc->ne[0];
  const int64_t cRows = ggml_nrows(pSrc);

  const float *vSrcD = (const float *)pSrc->data;
  float *vDstD = (float *)pDst->data;

  const int cWarpSize = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
  const int cRowsPerBlock = 4;

  const int64_t cNumBlocks = (cRows + cRowsPerBlock - 1) / cRowsPerBlock;

  cudaStream_t vStream = pCtx.stream();
  dim3 vGridDims(cNumBlocks, 1, 1);
  dim3 vBlockDims(cWarpSize, cRowsPerBlock, 1);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(vGridDims, vBlockDims, 0, vStream);

  const float cScale = 1 / sqrtf(cN);

  switch (cN) {
  case 64:
    ggml_cuda_kernel_launch(fFwhtCuda<64>, cLaunchParams, vSrcD, vDstD, cRows, cScale);
    return true;
  case 128:
    ggml_cuda_kernel_launch(fFwhtCuda<128>, cLaunchParams, vSrcD, vDstD, cRows, cScale);
    return true;
  case 256:
    ggml_cuda_kernel_launch(fFwhtCuda<256>, cLaunchParams, vSrcD, vDstD, cRows, cScale);
    return true;
  case 512:
    ggml_cuda_kernel_launch(fFwhtCuda<512>, cLaunchParams, vSrcD, vDstD, cRows, cScale);
    return true;
  default:
    return false;
  }
}
