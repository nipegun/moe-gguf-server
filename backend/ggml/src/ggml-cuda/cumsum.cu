#include <algorithm>
#include "cumsum.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
#endif // GGML_CUDA_USE_CUB

template <typename T, int BLOCK_SIZE>
static __global__ void fCumsumCubKernel(const T *__restrict__ pSrc, T *__restrict__ pDst, const int64_t pNe00,
                                        const int64_t pNe01, const int64_t pNe02, const int64_t pNe03,
                                        const int64_t pS01, const int64_t pS02, const int64_t pS03, const int64_t pS1,
                                        const int64_t pS2, const int64_t pS3) {
#ifdef GGML_CUDA_USE_CUB
  using BlockScanT = cub::BlockScan<T, BLOCK_SIZE>;

  __shared__ typename BlockScanT::TempStorage vTempStorage;
  __shared__ T vBlockCarry;

  const int cTid = threadIdx.x;
  constexpr int cUnrollFactor = 4;
  constexpr int cTileSize = BLOCK_SIZE * cUnrollFactor;

  const int64_t cI1 = blockIdx.x;
  const int64_t cI2 = blockIdx.y;
  const int64_t cI3 = blockIdx.z;

  if (cI1 >= pNe01 || cI2 >= pNe02 || cI3 >= pNe03) {
    return;
  }

  const T *vSrcRow = pSrc + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;
  T *vDstRow = pDst + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;

  if (cTid == 0) {
    vBlockCarry = 0;
  }
  __syncthreads();

  for (int64_t vStart = 0; vStart < pNe00; vStart += cTileSize) {
    T aItems[cUnrollFactor];
    T vThreadSum = T(0);

#pragma unroll
    for (int vI = 0; vI < cUnrollFactor; vI++) {
      int64_t vIdx = vStart + cTid * cUnrollFactor + vI;
      T vVal = (vIdx < pNe00) ? vSrcRow[vIdx] : T(0);
      vThreadSum += vVal;
      aItems[vI] = vThreadSum;
    }

    // Block-wide scan on thread sums
    T vThreadPrefix;
    T vBlockTotal;
    BlockScanT(vTempStorage).InclusiveSum(vThreadSum, vThreadPrefix, vBlockTotal);
    __syncthreads();

    // Add offset to each item and store
    T vThreadOffset = vThreadPrefix - vThreadSum + vBlockCarry;
#pragma unroll
    for (int vI = 0; vI < cUnrollFactor; vI++) {
      int64_t vIdx = vStart + cTid * cUnrollFactor + vI;
      if (vIdx < pNe00) {
        vDstRow[vIdx] = aItems[vI] + vThreadOffset;
      }
    }

    __syncthreads();

    // Update carry for next tile
    if (cTid == 0) {
      vBlockCarry += vBlockTotal;
    }
  }
#else
  NO_DEVICE_CODE;
#endif // GGML_CUDA_USE_CUB
}

// Fallback kernel implementation
template <typename T>
static __global__ void fCumsumKernel(const T *pSrc, T *pDst, const int64_t pNe00, const int64_t pNe01,
                                     const int64_t pNe02, const int64_t pNe03, const int64_t pS00, const int64_t pS01,
                                     const int64_t pS02, const int64_t pS03, const int64_t pS0, const int64_t pS1,
                                     const int64_t pS2, const int64_t pS3) {

  GGML_UNUSED_VARS(pS00, pS0);

  const int cTid = threadIdx.x;
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  const int cLane = cTid % cWarpSize;
  const int cWarp = cTid / cWarpSize;
  const int cWarpsPerBlock = blockDim.x / cWarpSize;

  extern __shared__ float aSmem[];
  float *vSVals = aSmem;
  float *vSWarpSums = aSmem + blockDim.x;
  float *vSCarry = aSmem + blockDim.x + cWarpsPerBlock;
  float *vSChunkTotal = vSCarry + 1;

  // Initialize carry
  if (cTid == 0) {
    *vSCarry = 0.0f;
  }
  __syncthreads();

  const int64_t cI3 = blockIdx.z;
  const int64_t cI2 = blockIdx.y;
  const int64_t cI1 = blockIdx.x;
  if (cI3 >= pNe03 || cI2 >= pNe02 || cI1 >= pNe01) {
    return;
  }

  const T *vSrcRow = pSrc + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;
  T *vDstRow = pDst + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;

  // register blocking: process 4 elements per thread to hide latency
  // and reduce synchronization overhead
  constexpr int cNumUnroll = 4;
  T aTemp[cNumUnroll];

  for (int64_t vI = 0; vI < pNe00; vI += cNumUnroll * blockDim.x) {
    int64_t vIdx = vI + cTid * cNumUnroll;

    // thread local sequential scan
    aTemp[0] = (vIdx < pNe00 ? vSrcRow[vIdx] : T(0));
#pragma unroll
    for (int64_t vJ = 1; vJ < cNumUnroll; vJ++) {
      aTemp[vJ] = aTemp[vJ - 1];
      if (vIdx + vJ < pNe00) {
        aTemp[vJ] += vSrcRow[vIdx + vJ];
      } else {
        aTemp[vJ] += 0;
      }
    }

    // last emenent is sum of all values assigned to thread
    float vVal = (vIdx < pNe00) ? ggml_cuda_cast<float, T>(aTemp[cNumUnroll - 1]) : 0.0f;

    // Warp inclusive scan
    vVal = warp_prefix_inclusive_sum<T, cWarpSize>(vVal);
    vSVals[cTid] = vVal;

    if (cLane == cWarpSize - 1) {
      vSWarpSums[cWarp] = vVal;
    }
    __syncthreads();

    // Exclusive scan of warp sums (warp 0 only)
    if (cWarp == 0) {
      float vW = (cTid < cWarpsPerBlock) ? vSWarpSums[cTid] : 0.0f;
      float vInc = warp_prefix_inclusive_sum<T, cWarpSize>(vW);
      if (cTid < cWarpsPerBlock) {
        vSWarpSums[cTid] = vInc - vW; // exclusive sum
      }
      if (cTid == cWarpsPerBlock - 1) {
        *vSChunkTotal = vInc; // total sum of this chunk
      }
    }
    __syncthreads();

    // write back results
    float vCarry = *vSCarry;
    // calculate sum offset for this thread
    float vFinalValOffset = vSVals[cTid] + vSWarpSums[cWarp] + vCarry - aTemp[cNumUnroll - 1];

#pragma unroll
    for (int32_t vJ = 0; vJ < cNumUnroll; vJ++) {
      if (vIdx + vJ < pNe00) {
        vDstRow[vIdx + vJ] = aTemp[vJ] + ggml_cuda_cast<T, float>(vFinalValOffset);
      }
    }

    __syncthreads();

    // Update carry for next chunk
    if (cTid == 0) {
      *vSCarry += *vSChunkTotal;
    }
  }
}

#ifdef GGML_CUDA_USE_CUB
template <typename T>
static void fCumsumCub(ggml_cuda_pool &pPool, const T *pSrc, T *pDst, int64_t pNe, cudaStream_t pStream) {
  size_t vTmpSize = 0;

  // Query how much temp storage CUDA UnBound (CUB) needs
  cub::DeviceScan::InclusiveSum(nullptr,  // d_temp_storage (null = just query size)
                                vTmpSize, // reference to size (will be set by CUB)
                                pSrc,     // input pointer
                                pDst,     // output pointer
                                pNe,      // number of elements
                                pStream   // CUDA stream to use
  );

  ggml_cuda_pool_alloc<uint8_t> vTmpAlloc(pPool, vTmpSize);

  // Perform the inclusive scan
  cub::DeviceScan::InclusiveSum((void *)vTmpAlloc.get(), vTmpSize, pSrc, pDst, pNe, pStream);
}
#endif // GGML_CUDA_USE_CUB

template <typename T>
static void fCumsumCuda([[maybe_unused]] ggml_backend_cuda_context &pCtx, const T *pSrc, T *pDst, const int64_t pNe00,
                        const int64_t pNe01, const int64_t pNe02, const int64_t pNe03, const int64_t pNb00,
                        const int64_t pNb01, const int64_t pNb02, const int64_t pNb03, const int64_t pNb0,
                        const int64_t pNb1, const int64_t pNb2, const int64_t pNb3, cudaStream_t pStream) {

  const size_t cTypeSize = sizeof(T);
  bool vUseCub = false;
#ifdef GGML_CUDA_USE_CUB
  // Check if we can use CUB (data must be contiguous along innermost dimension)
  const bool cIsContiguous = (pNb00 == cTypeSize) && (pNb0 == cTypeSize);

  if (cIsContiguous) {
    vUseCub = true;
    const int64_t cNrows = pNe01 * pNe02 * pNe03;
    // TODO: Compare with DeviceSegmentedScan::InclusiveSegmentedSum for nrows > 1 once InclusiveSegmentedSum is
    // released Heuristics were determined empirically
    if (((cNrows == 1) && (pNe00 > 1024)) || (pNe00 / cNrows > 4096)) {
      for (int vI = 0; vI < cNrows; vI++) {
        fCumsumCub(pCtx.pool(), pSrc + vI * pNe00, pDst + vI * pNe00, pNe00, pStream);
      }
      return;
    }
  }
#endif // GGML_CUDA_USE_CUB
  dim3 vGridDims(pNe01, pNe02, pNe03);
  const auto &vInfo = ggml_cuda_info().devices[ggml_cuda_get_device()];
  const int cWarpSize = vInfo.warp_size;
  const int cNumWarps = (pNe00 + cWarpSize - 1) / cWarpSize;
  int vBlockSize = cNumWarps * cWarpSize;
  vBlockSize = std::min(vBlockSize, CUDA_CUMSUM_BLOCK_SIZE);
  dim3 vBlockDims(vBlockSize, 1, 1);
  const int cWarpsPerBlock = vBlockSize / cWarpSize;
  const size_t cShmemSize = (vBlockSize + cWarpsPerBlock + 2) * sizeof(float);

  if (vUseCub && pNe00 >= 1024) {
    fCumsumCubKernel<T, CUDA_CUMSUM_BLOCK_SIZE><<<vGridDims, CUDA_CUMSUM_BLOCK_SIZE, 0, pStream>>>(
        pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb01 / cTypeSize, pNb02 / cTypeSize, pNb03 / cTypeSize,
        pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
  } else {
    fCumsumKernel<<<vGridDims, vBlockDims, cShmemSize, pStream>>>(
        pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb00 / cTypeSize, pNb01 / cTypeSize, pNb02 / cTypeSize,
        pNb03 / cTypeSize, pNb0 / cTypeSize, pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
  }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == pDst->type);
  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fCumsumCuda(pCtx, (const float *)vSrc0->data, (float *)pDst->data, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2],
                vSrc0->ne[3], vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], pDst->nb[0], pDst->nb[1],
                pDst->nb[2], pDst->nb[3], vStream);
  } break;
  // We do not support those on CPU for now anyway, so comment them out because they cause errors on some CI platforms
  /*case GGML_TYPE_F16:
      {
          cumsum_cuda(
              (const half *)src0->data, (half *)dst->data,
              src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
              src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
              dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
              stream
          );
      } break;
  case GGML_TYPE_BF16:
      {
          cumsum_cuda(
              (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
              src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
              src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
              dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
              stream
          );
      } break;*/
  default:
    GGML_ABORT("fatal error");
  }
}
