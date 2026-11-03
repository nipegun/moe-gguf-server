#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void fNormF32(const float *pX, float *pDst, const int pNcols, const int64_t pStrideRow,
                                const int64_t pStrideChannel, const int64_t pStrideSample, const float pEps) {
  const int cNrows = gridDim.x;
  const int cNchannels = gridDim.y;

  const int cRow = blockIdx.x;
  const int cChannel = blockIdx.y;
  const int cSample = blockIdx.z;
  const int cTid = threadIdx.x;

  pX += cSample * pStrideSample + cChannel * pStrideChannel + cRow * pStrideRow;
  pDst += ((cSample * cNchannels + cChannel) * cNrows + cRow) * pNcols;

  float2 vMeanVar = make_float2(0.0f, 0.0f);

  ggml_cuda_pdl_sync();
  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    const float cXi = pX[vCol];
    vMeanVar.x += cXi;
    vMeanVar.y += cXi * cXi;
  }

  // sum up partial sums
  extern __shared__ float2 aSSum2[];
  vMeanVar = block_reduce<block_reduce_method::SUM, block_size>(vMeanVar, aSSum2);

  const float cMean = vMeanVar.x / pNcols;
  const float cVar = vMeanVar.y / pNcols - cMean * cMean;
  const float cInvStd = rsqrtf(cVar + pEps);

  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    pDst[vCol] = (pX[vCol] - cMean) * cInvStd;
  }
}

template <int block_size>
static __global__ void fGroupNormF32(const float *pX, float *pDst, const int pGroupSize, const int pNeElements,
                                     const float pEps) {
  // blockIdx.x: num_groups idx
  // threadIdx.x: block_size idx
  const int cStart = blockIdx.x * pGroupSize + threadIdx.x;
  const int cEnd = min(blockIdx.x * pGroupSize + pGroupSize, pNeElements);

  float vTmp = 0.0f; // partial sum for thread in warp

  ggml_cuda_pdl_sync();
  for (int vJ = cStart; vJ < cEnd; vJ += block_size) {
    vTmp += pX[vJ];
  }

  extern __shared__ float aSSum[];
  vTmp = block_reduce<block_reduce_method::SUM, block_size>(vTmp, aSSum);

  const float cMean = vTmp / pGroupSize;
  vTmp = 0.0f;

  for (int vJ = cStart; vJ < cEnd; vJ += block_size) {
    const float cXi = pX[vJ] - cMean;
    pDst[vJ] = cXi;
    vTmp += cXi * cXi;
  }

  vTmp = block_reduce<block_reduce_method::SUM, block_size>(vTmp, aSSum + 32);

  const float cVariance = vTmp / pGroupSize;
  const float cScale = rsqrtf(cVariance + pEps);
  for (int vJ = cStart; vJ < cEnd; vJ += block_size) {
    pDst[vJ] *= cScale;
  }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void
fRmsNormF32(const float *pX, float *pDst, const int pNcols, const int64_t pStrideRow, const int64_t pStrideChannel,
            const int64_t pStrideSample, const float pEps, const float *pMul = nullptr, const int64_t pMulStrideRow = 0,
            const int64_t pMulStrideChannel = 0, const int64_t pMulStrideSample = 0,
            const uint3 pMulNcolsPacked = make_uint3(0, 0, 0), const uint3 pMulNrowsPacked = make_uint3(0, 0, 0),
            const uint3 pMulNchannelsPacked = make_uint3(0, 0, 0), const uint3 pMulNsamplesPacked = make_uint3(0, 0, 0),
            const float *pAdd = nullptr, const int64_t pAddStrideRow = 0, const int64_t pAddStrideChannel = 0,
            const int64_t pAddStrideSample = 0, const uint3 pAddNcolsPacked = make_uint3(0, 0, 0),
            const uint3 pAddNrowsPacked = make_uint3(0, 0, 0), const uint3 pAddNchannelsPacked = make_uint3(0, 0, 0),
            const uint3 pAddNsamplesPacked = make_uint3(0, 0, 0)) {
  ggml_cuda_pdl_lc();
  const int cNrows = gridDim.x;
  const int cNchannels = gridDim.y;

  const int cRow = blockIdx.x;
  const int cChannel = blockIdx.y;
  const int cSample = blockIdx.z;
  const int cTid = threadIdx.x;

  static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

  pX += cSample * pStrideSample + cChannel * pStrideChannel + cRow * pStrideRow;
  pDst += ((cSample * cNchannels + cChannel) * cNrows + cRow) * pNcols;

  if constexpr (do_multiply) {
    const uint32_t cMulRow = fastmodulo(cRow, pMulNrowsPacked);
    const uint32_t cMulChannel = fastmodulo(cChannel, pMulNchannelsPacked);
    const uint32_t cMulSample = fastmodulo(cSample, pMulNsamplesPacked);
    pMul += cMulSample * pMulStrideSample + cMulChannel * pMulStrideChannel + cMulRow * pMulStrideRow;
  }

  if constexpr (do_add) {
    const int cAddRow = fastmodulo(cRow, pAddNrowsPacked);
    const int cAddChannel = fastmodulo(cChannel, pAddNchannelsPacked);
    const int cAddSample = fastmodulo(cSample, pAddNsamplesPacked);
    pAdd += cAddSample * pAddStrideSample + cAddChannel * pAddStrideChannel + cAddRow * pAddStrideRow;
  }

  float vTmp = 0.0f; // partial sum for thread in warp

  ggml_cuda_pdl_sync();
  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    const float cXi = pX[vCol];
    vTmp += cXi * cXi;
  }

  // sum up partial sums
  extern __shared__ float aSSum[];
  vTmp = block_reduce<block_reduce_method::SUM, block_size>(vTmp, aSSum);

  const float cMean = vTmp / pNcols;
  const float cScale = rsqrtf(cMean + pEps);

  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    if constexpr (do_multiply && do_add) {
      const int cMulCol = fastmodulo(vCol, pMulNcolsPacked);
      const int cAddCol = fastmodulo(vCol, pAddNcolsPacked);
      pDst[vCol] = cScale * pX[vCol] * pMul[cMulCol] + pAdd[cAddCol];
    } else if constexpr (do_multiply) {
      const int mul_col = fastmodulo(vCol, pMulNcolsPacked);
      pDst[vCol] = cScale * pX[vCol] * pMul[mul_col];
    } else {
      pDst[vCol] = cScale * pX[vCol];
    }
  }
}

template <int block_size>
static __global__ void fRmsNormBackF32(const float *pGrad, const float *pXf, float *pDst, const int pNcols,
                                       const float pEps) {
  const int cRow = blockIdx.x * blockDim.y + threadIdx.y;
  const int cTid = threadIdx.x;

  pGrad += int64_t(cRow) * pNcols;
  pXf += int64_t(cRow) * pNcols;
  pDst += int64_t(cRow) * pNcols;

  float vSumXx = 0.0f; // sum for squares of x, equivalent to forward pass
  float vSumXg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

  ggml_cuda_pdl_sync();
  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    const float cXfi = pXf[vCol];
    vSumXx += cXfi * cXfi;
    vSumXg += cXfi * pGrad[vCol];
  }

  // sum up partial sums
  vSumXx = warp_reduce_sum(vSumXx);
  vSumXg = warp_reduce_sum(vSumXg);
  if constexpr (block_size > WARP_SIZE) {
    static_assert(block_size == 1024, "unexpected block_size");
    __shared__ float vSSumXx[32];
    __shared__ float vSSumXg[32];
    const int cWarpId = threadIdx.x / WARP_SIZE;
    const int cLaneId = threadIdx.x % WARP_SIZE;
    if (cLaneId == 0) {
      vSSumXx[cWarpId] = vSumXx;
      vSSumXg[cWarpId] = vSumXg;
    }
    __syncthreads();

    vSumXx = vSSumXx[cLaneId];
    vSumXx = warp_reduce_sum(vSumXx);

    vSumXg = vSSumXg[cLaneId];
    vSumXg = warp_reduce_sum(vSumXg);
  }

  const float cMeanEps = vSumXx / pNcols + pEps;
  const float cSumEps = vSumXx + pNcols * pEps;

  const float cScaleGrad = rsqrtf(cMeanEps);
  const float cScaleX = -cScaleGrad * vSumXg / cSumEps;

  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    pDst[vCol] = cScaleGrad * pGrad[vCol] + cScaleX * pXf[vCol];
  }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void fL2NormF32(const float *pX, float *pDst, const int pNcols, const int64_t pStrideRow,
                                  const int64_t pStrideChannel, const int64_t pStrideSample, const float pEps) {
  const int cNrows = gridDim.x;
  const int cNchannels = gridDim.y;

  const int cRow = blockIdx.x;
  const int cChannel = blockIdx.y;
  const int cSample = blockIdx.z;
  const int cTid = threadIdx.x;

  pX += cSample * pStrideSample + cChannel * pStrideChannel + cRow * pStrideRow;
  pDst += ((cSample * cNchannels + cChannel) * cNrows + cRow) * pNcols;

  float vTmp = 0.0f; // partial sum for thread in warp

  ggml_cuda_pdl_sync();
  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    const float cXi = pX[vCol];
    vTmp += cXi * cXi;
  }

  // sum up partial sums
  extern __shared__ float aSSum[];
  vTmp = block_reduce<block_reduce_method::SUM, block_size>(vTmp, aSSum);
  ggml_cuda_pdl_lc();

  // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
  const float cScale = rsqrtf(fmaxf(vTmp, pEps * pEps));

  for (int vCol = cTid; vCol < pNcols; vCol += block_size) {
    pDst[vCol] = cScale * pX[vCol];
  }
}

static void fNormF32Cuda(const float *pX, float *pDst, const int pNcols, const int pNrows, const int pNchannels,
                         const int pNsamples, const int64_t pStrideRow, const int64_t pStrideChannel,
                         const int64_t pStrideSample, const float pEps, cudaStream_t pStream) {
  const dim3 cBlocksNum(pNrows, pNchannels, pNsamples);
  if (pNcols < 1024) {
    const dim3 cBlockDims(WARP_SIZE, 1, 1);
    fNormF32<WARP_SIZE>
        <<<cBlocksNum, cBlockDims, 0, pStream>>>(pX, pDst, pNcols, pStrideRow, pStrideChannel, pStrideSample, pEps);
  } else {
    const dim3 cBlockDims(1024, 1, 1);
    fNormF32<1024><<<cBlocksNum, cBlockDims, (cBlockDims.x > WARP_SIZE ? 32 * sizeof(float2) : 0), pStream>>>(
        pX, pDst, pNcols, pStrideRow, pStrideChannel, pStrideSample, pEps);
  }
}

static void fGroupNormF32Cuda(const float *pX, float *pDst, const int pNumGroups, const float pEps,
                              const int pGroupSize, const int pNeElements, cudaStream_t pStream) {
  if (pGroupSize < 1024) {
    const dim3 cBlockDims(WARP_SIZE, 1, 1);
    fGroupNormF32<WARP_SIZE><<<pNumGroups, cBlockDims, 0, pStream>>>(pX, pDst, pGroupSize, pNeElements, pEps);
  } else {
    const dim3 cBlockDims(1024, 1, 1);
    fGroupNormF32<1024><<<pNumGroups, cBlockDims, (cBlockDims.x > WARP_SIZE ? 2 * 32 * sizeof(float) : 0), pStream>>>(
        pX, pDst, pGroupSize, pNeElements, pEps);
  }
}

static void fRmsNormF32Cuda(const float *pX, float *pDst, const int pNcols, const int pNrows, const int pNchannels,
                            const int pNsamples, const int64_t pStrideRow, const int64_t pStrideChannel,
                            const int64_t pStrideSample, const float pEps, cudaStream_t pStream) {
  const dim3 cBlocksNum(pNrows, pNchannels, pNsamples);
  if (pNcols < 1024) {
    const dim3 cBlockDims(256, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams = {cBlocksNum, cBlockDims,
                                                          cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
    ggml_cuda_kernel_launch(
        fRmsNormF32<256, false>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel, pStrideSample, pEps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), nullptr,
        0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
  } else {
    const dim3 cBlockDims(1024, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
        cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
    ggml_cuda_kernel_launch(
        fRmsNormF32<1024, false>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel, pStrideSample, pEps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), nullptr,
        0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
  }
}

static void fRmsNormMulF32Cuda(const float *pX, const float *pMul, const float *pAdd, float *pDst, const int pNcols,
                               const int pNrows, const int pNchannels, const int pNsamples, const int64_t pStrideRow,
                               const int64_t pStrideChannel, const int64_t pStrideSample, const int64_t pMulStrideRow,
                               const int64_t pMulStrideChannel, const int64_t pMulStrideSample,
                               const uint32_t pMulNcols, const uint32_t pMulNrows, const uint32_t pMulNchannels,
                               const uint32_t pMulNsamples, const int64_t pAddStrideRow,
                               const int64_t pAddStrideChannel, const int64_t pAddStrideSample,
                               const uint32_t pAddNcols, const uint32_t pAddNrows, const uint32_t pAddNchannels,
                               const uint32_t pAddNsamples, const float pEps, cudaStream_t pStream) {
  const dim3 cBlocksNum(pNrows, pNchannels, pNsamples);
  if (pMul == nullptr) {
    fRmsNormF32Cuda(pX, pDst, pNcols, pNrows, pNchannels, pNsamples, pStrideRow, pStrideChannel, pStrideSample, pEps,
                    pStream);
    return;
  }
  if (pAdd == nullptr) {
    const uint3 cMulNcolsPacked = init_fastdiv_values(pMulNcols);
    const uint3 cMulNrowsPacked = init_fastdiv_values(pMulNrows);
    const uint3 cMulNchannelsPacked = init_fastdiv_values(pMulNchannels);
    const uint3 cMulNsamplesPacked = init_fastdiv_values(pMulNsamples);
    if (pNcols < 1024) {
      const dim3 cBlockDims(256, 1, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
          cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
      ggml_cuda_kernel_launch(fRmsNormF32<256, true>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel,
                              pStrideSample, pEps, pMul, pMulStrideRow, pMulStrideChannel, pMulStrideSample,
                              cMulNcolsPacked, cMulNrowsPacked, cMulNchannelsPacked, cMulNsamplesPacked,
                              // underlying cudaLaunchKernelEx does not support default params
                              nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                              make_uint3(0, 0, 0));
    } else {
      const dim3 cBlockDims(1024, 1, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
          cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
      ggml_cuda_kernel_launch(fRmsNormF32<1024, true>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel,
                              pStrideSample, pEps, pMul, pMulStrideRow, pMulStrideChannel, pMulStrideSample,
                              cMulNcolsPacked, cMulNrowsPacked, cMulNchannelsPacked, cMulNsamplesPacked,
                              // underlying cudaLaunchKernelEx does not support default params
                              nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                              make_uint3(0, 0, 0));
    }
  } else {
    const uint3 cMulNcolsPacked = init_fastdiv_values(pMulNcols);
    const uint3 cMulNrowsPacked = init_fastdiv_values(pMulNrows);
    const uint3 cMulNchannelsPacked = init_fastdiv_values(pMulNchannels);
    const uint3 cMulNsamplesPacked = init_fastdiv_values(pMulNsamples);

    const uint3 cAddNcolsPacked = init_fastdiv_values(pAddNcols);
    const uint3 cAddNrowsPacked = init_fastdiv_values(pAddNrows);
    const uint3 cAddNchannelsPacked = init_fastdiv_values(pAddNchannels);
    const uint3 cAddNsamplesPacked = init_fastdiv_values(pAddNsamples);
    if (pNcols < 1024) {
      const dim3 cBlockDims(256, 1, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
          cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
      ggml_cuda_kernel_launch(fRmsNormF32<256, true, true>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel,
                              pStrideSample, pEps, pMul, pMulStrideRow, pMulStrideChannel, pMulStrideSample,
                              cMulNcolsPacked, cMulNrowsPacked, cMulNchannelsPacked, cMulNsamplesPacked, pAdd,
                              pAddStrideRow, pAddStrideChannel, pAddStrideSample, cAddNcolsPacked, cAddNrowsPacked,
                              cAddNchannelsPacked, cAddNsamplesPacked);
    } else {
      const dim3 cBlockDims(1024, 1, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
          cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
      ggml_cuda_kernel_launch(fRmsNormF32<1024, true, true>, cLaunchParams, pX, pDst, pNcols, pStrideRow,
                              pStrideChannel, pStrideSample, pEps, pMul, pMulStrideRow, pMulStrideChannel,
                              pMulStrideSample, cMulNcolsPacked, cMulNrowsPacked, cMulNchannelsPacked,
                              cMulNsamplesPacked, pAdd, pAddStrideRow, pAddStrideChannel, pAddStrideSample,
                              cAddNcolsPacked, cAddNrowsPacked, cAddNchannelsPacked, cAddNsamplesPacked);
    }
  }
}

static void fRmsNormBackF32Cuda(const float *pGrad, const float *pXf, float *pDst, const int pNcols, const int pNrows,
                                const float pEps, cudaStream_t pStream) {
  if (pNcols < 1024) {
    const dim3 cBlockDims(WARP_SIZE, 1, 1);
    fRmsNormBackF32<WARP_SIZE><<<pNrows, cBlockDims, 0, pStream>>>(pGrad, pXf, pDst, pNcols, pEps);
  } else {
    const dim3 cBlockDims(1024, 1, 1);
    fRmsNormBackF32<1024><<<pNrows, cBlockDims, 0, pStream>>>(pGrad, pXf, pDst, pNcols, pEps);
  }
}

static void fL2NormF32Cuda(const float *pX, float *pDst, const int pNcols, const int pNrows, const int pNchannels,
                           const int pNsamples, const int64_t pStrideRow, const int64_t pStrideChannel,
                           const int64_t pStrideSample, const float pEps, cudaStream_t pStream) {
  const dim3 cBlocksNum(pNrows, pNchannels, pNsamples);
  if (pNcols < 1024) {
    const dim3 cBlockDims(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params{cBlocksNum, cBlockDims, 0, pStream};
    ggml_cuda_kernel_launch(fL2NormF32<WARP_SIZE>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel,
                            pStrideSample, pEps);
  } else {
    const dim3 cBlockDims(1024, 1, 1);
    const ggml_cuda_kernel_launch_params cLaunchParams = ggml_cuda_kernel_launch_params{
        cBlocksNum, cBlockDims, cBlockDims.x > WARP_SIZE ? 32 * sizeof(float) : 0, pStream};
    ggml_cuda_kernel_launch(fL2NormF32<1024>, cLaunchParams, pX, pDst, pNcols, pStrideRow, pStrideChannel,
                            pStrideSample, pEps);
  }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const float *vSrc0D = (const float *)src0->data;
  float *vDstD = (float *)dst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_UNARY_OP_LOCALS;

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  const size_t cTs0 = ggml_type_size(src0->type);
  GGML_ASSERT(nb00 == cTs0);
  const int64_t cS01 = nb01 / cTs0;
  const int64_t cS02 = nb02 / cTs0;
  const int64_t cS03 = nb03 / cTs0;

  fNormF32Cuda(vSrc0D, vDstD, ne00, ne01, ne02, ne03, cS01, cS02, cS03, vEps, vStream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  int vNumGroups = pDst->op_params[0];

  float vEps;
  memcpy(&vEps, pDst->op_params + 1, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  int vGroupSize = vSrc0->ne[0] * vSrc0->ne[1] * ((vSrc0->ne[2] + vNumGroups - 1) / vNumGroups);
  fGroupNormF32Cuda(vSrc0D, vDstD, vNumGroups * vSrc0->ne[3], vEps, vGroupSize, ggml_nelements(vSrc0), vStream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const float *vSrc0D = (const float *)src0->data;
  float *vDstD = (float *)dst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_UNARY_OP_LOCALS;

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  const size_t cTs0 = ggml_type_size(src0->type);
  GGML_ASSERT(nb00 == cTs0);
  const int64_t cS01 = nb01 / cTs0;
  const int64_t cS02 = nb02 / cTs0;
  const int64_t cS03 = nb03 / cTs0;

  fRmsNormF32Cuda(vSrc0D, vDstD, ne00, ne01, ne02, ne03, cS01, cS02, cS03, vEps, vStream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pMulTensor) {
  const ggml_tensor *vRmsNormSrc = (ggml_tensor *)pDst->src[0];
  float vEps = 0.0f;

  memcpy(&vEps, pDst->op_params, sizeof(float));

  const float *vSrc0D = (const float *)vRmsNormSrc->data;
  const float *vMulD = nullptr;
  const ggml_tensor *vMulSrc = nullptr;

  if (pMulTensor->src[0] == pDst) {
    vMulD = (float *)pMulTensor->src[1]->data;
    vMulSrc = pMulTensor->src[1];
  } else if (pMulTensor->src[1] == pDst) {
    vMulD = (float *)pMulTensor->src[0]->data;
    vMulSrc = pMulTensor->src[0];
  } else {
    GGML_ASSERT(false);
  }

  float *vDstD = (float *)pMulTensor->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vRmsNormSrc->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(pMulTensor->type == GGML_TYPE_F32);
  GGML_ASSERT(vEps >= 0.0f);

  const int64_t cNe00 = vRmsNormSrc->ne[0];
  const int64_t cNe01 = vRmsNormSrc->ne[1];
  const int64_t cNe02 = vRmsNormSrc->ne[2];
  const int64_t cNe03 = vRmsNormSrc->ne[3];

  const size_t cTs0 = ggml_type_size(vRmsNormSrc->type);
  GGML_ASSERT(vRmsNormSrc->nb[0] == cTs0);
  const int64_t cS01 = vRmsNormSrc->nb[1] / cTs0;
  const int64_t cS02 = vRmsNormSrc->nb[2] / cTs0;
  const int64_t cS03 = vRmsNormSrc->nb[3] / cTs0;

  const size_t cTsMul = ggml_type_size(vMulSrc->type);
  GGML_ASSERT(vMulSrc->nb[0] == cTsMul);
  const int64_t cMulS01 = vMulSrc->nb[1] / cTsMul;
  const int64_t cMulS02 = vMulSrc->nb[2] / cTsMul;
  const int64_t cMulS03 = vMulSrc->nb[3] / cTsMul;

  const int cMulNcols = vMulSrc->ne[0];
  const int cMulNrows = vMulSrc->ne[1];
  const int cMulNchannels = vMulSrc->ne[2];
  const int cMulNsamples = vMulSrc->ne[3];

  fRmsNormMulF32Cuda(vSrc0D, vMulD, nullptr, vDstD, cNe00, cNe01, cNe02, cNe03,
                     /*s00*/ cS01, cS02, cS03,
                     /*mul_s00*/ cMulS01, cMulS02, cMulS03, cMulNcols, cMulNrows, cMulNchannels, cMulNsamples,
                     /*add_s00*/ 0, 0, 0, 0, 0, 0, 0, vEps, vStream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pMulTensor,
                                     ggml_tensor *pAddTensor) {
  const ggml_tensor *vRmsNormSrc = (ggml_tensor *)pDst->src[0];
  float vEps = 0.0f;

  memcpy(&vEps, pDst->op_params, sizeof(float));

  const float *vSrc0D = (const float *)vRmsNormSrc->data;
  const float *vMulD = nullptr;
  const ggml_tensor *vMulSrc = nullptr;

  if (pMulTensor->src[0] == pDst) {
    vMulD = (float *)pMulTensor->src[1]->data;
    vMulSrc = pMulTensor->src[1];
  } else if (pMulTensor->src[1] == pDst) {
    vMulD = (float *)pMulTensor->src[0]->data;
    vMulSrc = pMulTensor->src[0];
  } else {
    GGML_ASSERT(false);
  }

  const float *vAddD = nullptr;
  const ggml_tensor *vAddSrc = nullptr;

  if (pAddTensor->src[0] == pMulTensor) {
    vAddD = (float *)pAddTensor->src[1]->data;
    vAddSrc = pAddTensor->src[1];
  } else if (pAddTensor->src[1] == pMulTensor) {
    vAddD = (float *)pAddTensor->src[0]->data;
    vAddSrc = pAddTensor->src[0];
  } else {
    GGML_ASSERT(false);
  }

  float *vDstD = (float *)pAddTensor->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vRmsNormSrc->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(pMulTensor->type == GGML_TYPE_F32);
  GGML_ASSERT(pAddTensor->type == GGML_TYPE_F32);
  GGML_ASSERT(vEps >= 0.0f);

  const int64_t cNe00 = vRmsNormSrc->ne[0];
  const int64_t cNe01 = vRmsNormSrc->ne[1];
  const int64_t cNe02 = vRmsNormSrc->ne[2];
  const int64_t cNe03 = vRmsNormSrc->ne[3];

  const size_t cTs0 = ggml_type_size(vRmsNormSrc->type);
  GGML_ASSERT(vRmsNormSrc->nb[0] == cTs0);
  const int64_t cS01 = vRmsNormSrc->nb[1] / cTs0;
  const int64_t cS02 = vRmsNormSrc->nb[2] / cTs0;
  const int64_t cS03 = vRmsNormSrc->nb[3] / cTs0;

  const size_t cTsMul = ggml_type_size(vMulSrc->type);
  GGML_ASSERT(vMulSrc->nb[0] == cTsMul);
  const int64_t cMulS01 = vMulSrc->nb[1] / cTsMul;
  const int64_t cMulS02 = vMulSrc->nb[2] / cTsMul;
  const int64_t cMulS03 = vMulSrc->nb[3] / cTsMul;

  const int cMulNcols = vMulSrc->ne[0];
  const int cMulNrows = vMulSrc->ne[1];
  const int cMulNchannels = vMulSrc->ne[2];
  const int cMulNsamples = vMulSrc->ne[3];

  const size_t cTsAdd = ggml_type_size(vAddSrc->type);
  GGML_ASSERT(vAddSrc->nb[0] == cTsAdd);
  const int64_t cAddS01 = vAddSrc->nb[1] / cTsAdd;
  const int64_t cAddS02 = vAddSrc->nb[2] / cTsAdd;
  const int64_t cAddS03 = vAddSrc->nb[3] / cTsAdd;

  const int cAddNcols = vAddSrc->ne[0];
  const int cAddNrows = vAddSrc->ne[1];
  const int cAddNchannels = vAddSrc->ne[2];
  const int cAddNsamples = vAddSrc->ne[3];

  fRmsNormMulF32Cuda(vSrc0D, vMulD, vAddD, vDstD, cNe00, cNe01, cNe02, cNe03,
                     /*s00*/ cS01, cS02, cS03,
                     /*mul_s00*/ cMulS01, cMulS02, cMulS03, cMulNcols, cMulNrows, cMulNchannels, cMulNsamples,
                     /*add_s00*/ cAddS01, cAddS02, cAddS03, cAddNcols, cAddNrows, cAddNchannels, cAddNsamples, vEps,
                     vStream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vGrad = pDst->src[0];  // gradients
  const ggml_tensor *vSrc0f = pDst->src[1]; // src0 from forward pass

  const float *vGradD = (const float *)vGrad->data;
  const float *vSrc0fD = (const float *)vSrc0f->data;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vGrad));

  GGML_ASSERT(vGrad->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0f->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNe00 = vSrc0f->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0f);

  float vEps;
  memcpy(&vEps, pDst->op_params, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  fRmsNormBackF32Cuda(vGradD, vSrc0fD, vDstD, cNe00, cNrows, vEps, vStream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const float *vSrc0D = (const float *)src0->data;
  float *vDstD = (float *)dst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_UNARY_OP_LOCALS;

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  const size_t cTs0 = ggml_type_size(src0->type);
  GGML_ASSERT(nb00 == cTs0);
  const int64_t cS01 = nb01 / cTs0;
  const int64_t cS02 = nb02 / cTs0;
  const int64_t cS03 = nb03 / cTs0;

  fL2NormF32Cuda(vSrc0D, vDstD, ne00, ne01, ne02, ne03, cS01, cS02, cS03, vEps, vStream);
}
