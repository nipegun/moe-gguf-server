#include "common.cuh"
#include "ggml.h"
#include "softmax.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif // GGML_USE_HIP

#include <cstdint>
#include <utility>

template <typename T> static __device__ __forceinline__ float t2f32(T pVal) { return (float)pVal; }

template <> __device__ float __forceinline__ t2f32<half>(half pVal) { return __half2float(pVal); }

struct soft_max_params {

  int64_t vNheads;
  uint32_t vNHeadLog2;
  int64_t vNcols;
  int64_t vNrowsX;
  int64_t vNrowsY;
  int64_t vNe00;
  int64_t vNe01;
  int64_t vNe02;
  int64_t vNe03;
  int64_t vNb11;
  int64_t vNb12;
  int64_t vNb13;

  int64_t vNe12;
  int64_t vNe13;
  float vScale;
  float vMaxBias;
  float vM0;
  float vM1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(const float *pX, const T *pMask, const float *pSinks, float *pDst,
                                    const soft_max_params p_) {
  const int cNcols = ncols_template == 0 ? p_.vNcols : ncols_template;

  const int cTid = threadIdx.x;

  const int64_t cI03 = blockIdx.z;
  const int64_t cI02 = blockIdx.y;
  const int64_t cI01 = blockIdx.x;

  // TODO: noncontigous inputs/outputs
  const int cRowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

  const int64_t cI11 = cI01;
  const int64_t cI12 = cI02 % p_.vNe12;
  const int64_t cI13 = cI03 % p_.vNe13;

  pX += int64_t(cRowx) * cNcols;
  pMask += (cI11 * p_.vNb11 + cI12 * p_.vNb12 + cI13 * p_.vNb13) / sizeof(T) * (pMask != nullptr);
  pDst += int64_t(cRowx) * cNcols;

  const int cBlockSize = block_size_template == 0 ? blockDim.x : block_size_template;

  const float cSlope = get_alibi_slope(p_.vMaxBias, cI02, p_.vNHeadLog2, p_.vM0, p_.vM1);

  extern __shared__ float aDataSoftMaxF32[];
  float *vBufIw = aDataSoftMaxF32; // shared memory buffer for inter-warp communication
  // shared memory buffer to cache values between iterations:
  float *vVals = use_shared ? vBufIw + WARP_SIZE : pDst;

  float vMaxVal = pSinks ? pSinks[cI02] : -INFINITY;

#pragma unroll
  for (int vCol0 = 0; vCol0 < cNcols; vCol0 += cBlockSize) {
    const int cCol = vCol0 + cTid;

    if (ncols_template == 0 && cCol >= cNcols) {
      break;
    }

    const float cVal = pX[cCol] * p_.vScale + (pMask ? cSlope * t2f32(pMask[cCol]) : 0.0f);

    vVals[cCol] = cVal;
    vMaxVal = max(vMaxVal, cVal);
  }

  // find the max value in the block
  vMaxVal = block_reduce<block_reduce_method::MAX, block_size_template>(vMaxVal, vBufIw);

  float vTmp = 0.0f; // partial sum

#pragma unroll
  for (int vCol0 = 0; vCol0 < cNcols; vCol0 += cBlockSize) {
    const int cCol = vCol0 + cTid;

    if (ncols_template == 0 && cCol >= cNcols) {
      break;
    }

    const float cVal = expf(vVals[cCol] - vMaxVal);
    vTmp += cVal;
    vVals[cCol] = cVal;
  }

  if (cBlockSize > WARP_SIZE) {
    // sync is needed as we reuse buf_iw across block_reduce invocations, see #26385
    // for block_size <= WARP_SIZE, block_reduce does not access buf_iw
    __syncthreads();
  }
  // find the sum of exps in the block
  vTmp = block_reduce<block_reduce_method::SUM, block_size_template>(vTmp, vBufIw);

  if (pSinks) {
    vTmp += expf(pSinks[cI02] - vMaxVal);
  }

  const float cInvSum = 1.0f / vTmp;

#pragma unroll
  for (int vCol0 = 0; vCol0 < cNcols; vCol0 += cBlockSize) {
    const int cCol = vCol0 + cTid;

    if (ncols_template == 0 && cCol >= cNcols) {
      return;
    }

    pDst[cCol] = vVals[cCol] * cInvSum;
  }
}

// TODO: Template to allow keeping ncols in registers if they fit
static __device__ void fSoftMaxF32ParallelizeColsSingleRow(const float *__restrict__ pX, float *__restrict__ pDst,
                                                           float *__restrict__ pTmpMaxs, float *__restrict__ pTmpSums,
                                                           float *pSharedValsMax, float *pSharedValsSum,
                                                           const soft_max_params p_) {
  namespace cg = cooperative_groups;

  const cg::grid_group cG = cg::this_grid();

  const int cTid = threadIdx.x;
  const int cColStart = blockIdx.x * blockDim.x + cTid;
  const int cNElemPerThread = 4;

  float aLocalVals[cNElemPerThread] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float vLocalMax = -INFINITY;
  const int cStepSize = gridDim.x * blockDim.x;

  // Compute thread-local max
  for (int vCol = cColStart; vCol < p_.vNcols;) {
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      const int cIdx = vCol + vI * cStepSize;
      aLocalVals[vI] = cIdx < p_.vNcols ? pX[cIdx] : -INFINITY;
    }
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      vLocalMax = fmaxf(vLocalMax, aLocalVals[vI]);
    }
    vCol += cStepSize * cNElemPerThread;
  }

  // Compute CTA-level max
  vLocalMax = block_reduce<block_reduce_method::MAX>(vLocalMax, pSharedValsMax);

  // Store CTA-level max to GMEM
  if (cTid == 0) {
    pTmpMaxs[blockIdx.x] = vLocalMax;
  }
  cG.sync();

  // Compute compute global max from CTA-level maxs
  assert(gridDim.x < blockDim.x); // currently we only support this case
  if (cTid < gridDim.x) {
    vLocalMax = pTmpMaxs[cTid];
  } else {
    vLocalMax = -INFINITY;
  }
  vLocalMax = block_reduce<block_reduce_method::MAX>(vLocalMax, pSharedValsMax);

  // Compute softmax dividends, accumulate divisor
  float vTmpExpf = 0.0f;
  for (int vCol = cColStart; vCol < p_.vNcols;) {
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      const int cIdx = vCol + vI * cStepSize;
      aLocalVals[vI] = cIdx < p_.vNcols ? pX[cIdx] : -INFINITY;
    }
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      const int cIdx = vCol + vI * cStepSize;
      if (cIdx < p_.vNcols) {
        const float cTmp = expf(aLocalVals[vI] - vLocalMax);
        vTmpExpf += cTmp;
        pDst[cIdx] = cTmp;
      }
    }
    vCol += cStepSize * cNElemPerThread;
  }

  // Reduce divisor within CTA
  vTmpExpf = block_reduce<block_reduce_method::SUM>(vTmpExpf, pSharedValsSum);

  // Store CTA-level sum to GMEM
  if (cTid == 0) {
    pTmpSums[blockIdx.x] = vTmpExpf;
  }
  cG.sync();

  // Compute global sum from CTA-level sums
  if (cTid < gridDim.x) {
    vTmpExpf = pTmpSums[cTid];
  } else {
    vTmpExpf = 0.0f;
  }
  vTmpExpf = block_reduce<block_reduce_method::SUM>(vTmpExpf, pSharedValsSum);

  // Divide dividend by global sum + store data
  for (int vCol = cColStart; vCol < p_.vNcols;) {
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      const int cIdx = vCol + vI * cStepSize;
      aLocalVals[vI] = cIdx < p_.vNcols ? pDst[cIdx] : -INFINITY;
    }
#pragma unroll
    for (int vI = 0; vI < cNElemPerThread; vI++) {
      const int cIdx = vCol + vI * cStepSize;
      if (cIdx < p_.vNcols) {
        pDst[cIdx] = aLocalVals[vI] / vTmpExpf;
      }
    }
    vCol += cStepSize * cNElemPerThread;
  }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static __global__ void fSoftMaxBackF32(const float *pGrad, const float *pDstf, float *pDst, const int pNcols,
                                       const float pScale) {
  const int cTid = threadIdx.x;
  const int cRowx = blockIdx.x;

  pGrad += int64_t(cRowx) * pNcols;
  pDstf += int64_t(cRowx) * pNcols;
  pDst += int64_t(cRowx) * pNcols;

  float vDgfDot = 0.0f; // dot product of dst from forward pass and gradients

  for (int vCol = cTid; vCol < pNcols; vCol += WARP_SIZE) {
    vDgfDot += pDstf[vCol] * pGrad[vCol];
  }

  vDgfDot = warp_reduce_sum(vDgfDot);

  for (int vCol = cTid; vCol < pNcols; vCol += WARP_SIZE) {
    pDst[vCol] = pScale * (pGrad[vCol] - vDgfDot) * pDstf[vCol];
  }
}

template <int... Ns, typename T>
static void fLaunchSoftMaxKernels(const float *pX, const T *pMask, const float *pSinks, float *pDst,
                                  const soft_max_params &p_, cudaStream_t pStream, dim3 pBlockDims, dim3 pBlockNums,
                                  size_t pNbytesShared) {
  const int cId = ggml_cuda_get_device();
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;

  auto vLaunchKernel = [=](auto pI) -> bool {
    constexpr int cNcols = decltype(pI)::value;
    constexpr int cBlock = (cNcols > 1024 ? 1024 : cNcols);

    if (p_.vNcols == cNcols) {
      CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, cNcols, cBlock, T>), cSmpbo);
      soft_max_f32<true, cNcols, cBlock>
          <<<pBlockNums, pBlockDims, pNbytesShared, pStream>>>(pX, pMask, pSinks, pDst, p_);
      return true;
    }
    return false;
  };

  // unary fold over launch_kernel
  if ((vLaunchKernel(std::integral_constant<int, Ns>{}) || ...)) {
    return;
  }

  // default case
  CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), cSmpbo);
  soft_max_f32<true, 0, 0><<<pBlockNums, pBlockDims, pNbytesShared, pStream>>>(pX, pMask, pSinks, pDst, p_);
}

__launch_bounds__(8 * WARP_SIZE, 1) static __global__
    void fSoftMaxF32ParallelizeCols(const float *__restrict__ pX, float *__restrict__ pDst,
                                    float *__restrict__ pTmpMaxs, float *__restrict__ pTmpSums,
                                    const soft_max_params p_)
// We loop over all instead of parallelizing across gridDim.y as cooperative groups
// currently only support synchronizing the complete grid if not launched as a cluster group
// (which requires CC > 9.0)
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#grid-synchronization
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
{
  __shared__ float vSharedVals[2][32];

  for (int vRowx = 0; vRowx < p_.vNe01 * p_.vNe02 * p_.vNe03; vRowx++) {
    fSoftMaxF32ParallelizeColsSingleRow(pX + int64_t(vRowx) * p_.vNcols, pDst + int64_t(vRowx) * p_.vNcols, pTmpMaxs,
                                        pTmpSums, vSharedVals[0], vSharedVals[1], p_);
  }
}

template <typename T>
static void fSoftMaxF32Cuda(const float *pX, const T *pMask, const float *pSinks, float *pDst,
                            const soft_max_params &pParams, cudaStream_t pStream,
                            [[maybe_unused]] ggml_backend_cuda_context &pCtx) {
  int vNth = WARP_SIZE;
  const int64_t cNcolsX = pParams.vNcols;

  while (vNth < cNcolsX && vNth < CUDA_SOFT_MAX_BLOCK_SIZE)
    vNth *= 2;
  const dim3 cBlockDims(vNth, 1, 1);
  const dim3 cBlockNums(pParams.vNe01, pParams.vNe02, pParams.vNe03);
  const size_t cNbytesShared = (GGML_PAD(cNcolsX, WARP_SIZE) + WARP_SIZE) * sizeof(float);
  static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

  const int cId = ggml_cuda_get_device();
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;

  if (cNbytesShared <= cSmpbo) {
    fLaunchSoftMaxKernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(pX, pMask, pSinks, pDst, pParams, pStream,
                                                                   cBlockDims, cBlockNums, cNbytesShared);
  } else {
    // Parallelize across SMs for top-p/dist-sampling
    // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on
    // the basis of a B6000 GPU and Can be adapted further for lower-SM-count GPUs, though keeping data in registers
    // should be implemented first as that is the optimal solution.
    if (ggml_cuda_info().devices[cId].supports_cooperative_launch &&
        cNcolsX / (pParams.vNe01 * pParams.vNe02 * pParams.vNe03) > 8192 && pMask == nullptr && pSinks == nullptr &&
        pParams.vScale == 1.0f && pParams.vMaxBias == 0.0f) {
      ggml_cuda_pool_alloc<float> vTmpMaxsAlloc(pCtx.pool(), ggml_cuda_info().devices[cId].nsm * sizeof(float));
      ggml_cuda_pool_alloc<float> vTmpSumsAlloc(pCtx.pool(), ggml_cuda_info().devices[cId].nsm * sizeof(float));

      void *vKernelArgs[] = {(void *)&pX, (void *)&pDst, (void *)&vTmpMaxsAlloc.ptr, (void *)&vTmpSumsAlloc.ptr,
                             (void *)const_cast<soft_max_params *>(&pParams)};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void *)fSoftMaxF32ParallelizeCols,
                                             dim3(ggml_cuda_info().devices[cId].nsm, 1, 1), dim3(WARP_SIZE * 8, 1, 1),
                                             vKernelArgs, 0, pStream));
    } else {
      const size_t cNbytesSharedLow = WARP_SIZE * sizeof(float);
      soft_max_f32<false, 0, 0>
          <<<cBlockNums, cBlockDims, cNbytesSharedLow, pStream>>>(pX, pMask, pSinks, pDst, pParams);
    }
  }
}

static void fSoftMaxBackF32Cuda(const float *pGrad, const float *pDstf, float *pDst, const int pNcols, const int pNrows,
                                const float pScale, cudaStream_t pStream) {
  const dim3 cBlockDims(WARP_SIZE, 1, 1);
  const dim3 cBlockNums(pNrows, 1, 1);

  fSoftMaxBackF32<<<cBlockNums, cBlockDims, 0, pStream>>>(pGrad, pDstf, pDst, pNcols, pScale);
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  const ggml_tensor *vSrc2 = pDst->src[2];

  const float *vSrc0D = (const float *)vSrc0->data;
  const void *vSrc1D = vSrc1 ? (const void *)vSrc1->data : nullptr;
  const void *vSrc2D = vSrc2 ? (const void *)vSrc2->data : nullptr;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_ASSERT(!vSrc1 || vSrc1->type == GGML_TYPE_F16 ||
              vSrc1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

  const int64_t cNrowsX = ggml_nrows(vSrc0);
  const int64_t cNrowsY = vSrc0->ne[1];

  const int64_t cNe00 = vSrc0->ne[0];

  float vScale = 1.0f;
  float vMaxBias = 0.0f;

  memcpy(&vScale, (const float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (const float *)pDst->op_params + 1, sizeof(float));

  const bool cUseF16 = (vSrc1 && vSrc1->type == GGML_TYPE_F16);

  const int64_t cNb11 = vSrc1 ? vSrc1->nb[1] : 1;
  const int64_t cNb12 = vSrc1 ? vSrc1->nb[2] : 1;
  const int64_t cNb13 = vSrc1 ? vSrc1->nb[3] : 1;

  const int64_t cNe12 = vSrc1 ? vSrc1->ne[2] : 1;
  const int64_t cNe13 = vSrc1 ? vSrc1->ne[3] : 1;

  const uint32_t cNHead = vSrc0->ne[2];
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floorf(log2f((float)cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  soft_max_params vParams = {};
  vParams.vNheads = vSrc0->ne[2];
  vParams.vNHeadLog2 = cNHeadLog2;
  vParams.vNcols = cNe00;
  vParams.vNrowsX = cNrowsX;
  vParams.vNrowsY = cNrowsY;
  vParams.vNe00 = vSrc0->ne[0];
  vParams.vNe01 = vSrc0->ne[1];
  vParams.vNe02 = vSrc0->ne[2];
  vParams.vNe03 = vSrc0->ne[3];
  vParams.vNb11 = cNb11;
  vParams.vNb12 = cNb12;
  vParams.vNb13 = cNb13;
  vParams.vNe12 = cNe12;
  vParams.vNe13 = cNe13;
  vParams.vScale = vScale;
  vParams.vMaxBias = vMaxBias;
  vParams.vM0 = cM0;
  vParams.vM1 = cM1;

  if (cUseF16) {
    fSoftMaxF32Cuda(vSrc0D, (const half *)vSrc1D, (const float *)vSrc2D, vDstD, vParams, vStream, pCtx);
  } else {
    fSoftMaxF32Cuda(vSrc0D, (const float *)vSrc1D, (const float *)vSrc2D, vDstD, vParams, vStream, pCtx);
  }
}

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0]; // grad
  const ggml_tensor *vSrc1 = pDst->src[1]; // forward pass output

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNcols = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

  float vScale = 1.0f;
  float vMaxBias = 0.0f;

  memcpy(&vScale, (const float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (const float *)pDst->op_params + 1, sizeof(float));

  GGML_ASSERT(vMaxBias == 0.0f);

  fSoftMaxBackF32Cuda(vSrc0D, vSrc1D, vDstD, cNcols, cNrows, vScale, vStream);
}
