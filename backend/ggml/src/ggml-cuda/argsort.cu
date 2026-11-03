#include "argsort.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
#if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#define STRIDED_ITERATOR_AVAILABLE
#include <cuda/iterator>
#endif
using namespace cub;
#endif // GGML_CUDA_USE_CUB

static __global__ void fInitIndices(int *pIndices, const int pNcols, const int pNrows) {
  const int cCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int cRow = blockIdx.y;

  if (cCol < pNcols && cRow < pNrows) {
    pIndices[cRow * pNcols + cCol] = cCol;
  }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int *offsets, const int ncols, const int nrows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= nrows) {
    offsets[idx] = idx * ncols;
  }
}
#endif // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB

// returns the suggested maximum number of rows to process during one argsort_f32_i32_cuda_cub() call
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t pNb01, const int64_t pNrows) {
  // perform argsort in chunks up to approximately this size (currently 64MB)
  // to avoid excessive temporary buffers memory usage
  const int cChunkBytes = 1 << 26;

  // calculate how many rows will fit in one chunk (must be at least one)
  const int cChunkNrows = std::max((int)(cChunkBytes / pNb01), 1);

  // limit the resulting amount to total nrows
  return std::min((int64_t)cChunkNrows, pNrows);
}

void argsort_f32_i32_cuda_cub(ggml_cuda_pool &pPool, const float *pX, int *pDst, const int pNcols, const int pNrows,
                              ggml_sort_order pOrder, cudaStream_t pStream) {
  ggml_cuda_pool_alloc<int> vTempIndicesAlloc(pPool, pNcols * pNrows);
  ggml_cuda_pool_alloc<float> vTempKeysAlloc(pPool, pNcols * pNrows);

  int *vTempIndices = vTempIndicesAlloc.get();
  float *vTempKeys = vTempKeysAlloc.get();

  static const int cBlockSize = 256;
  const dim3 cGridSize((pNcols + cBlockSize - 1) / cBlockSize, pNrows);
  fInitIndices<<<cGridSize, cBlockSize, 0, pStream>>>(vTempIndices, pNcols, pNrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
  auto vOffsetIterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), pNcols);
#else
  // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
  const int nrows_offset = pNrows + 1;
  ggml_cuda_pool_alloc<int> offsets_alloc(pPool, nrows_offset);
  int *vOffsetIterator = offsets_alloc.get();
  const dim3 offset_grid((nrows_offset + cBlockSize - 1) / cBlockSize);
  init_offsets<<<offset_grid, cBlockSize, 0, pStream>>>(vOffsetIterator, pNcols, pNrows);
#endif
  CUDA_CHECK(cudaMemcpyAsync(vTempKeys, pX, pNcols * pNrows * sizeof(float), cudaMemcpyDeviceToDevice, pStream));

  size_t vTempStorageBytes = 0;

  bool vIsCapturing = false;
#ifdef USE_CUDA_GRAPH
  // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while
  // DeviceSegmentedRadixSort does. See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
  // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
  cudaStreamCaptureStatus vCaptureStatus;
  CUDA_CHECK(cudaStreamIsCapturing(pStream, &vCaptureStatus));
  vIsCapturing = (vCaptureStatus != cudaStreamCaptureStatusNone);
#endif // USE_CUDA_GRAPH

  if (pOrder == GGML_SORT_ORDER_ASC) {
    if (pNrows == 1) {
      CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, vTempStorageBytes, vTempKeys, vTempKeys, // keys (in-place)
                                            vTempIndices, pDst,                               // values (indices)
                                            pNcols, 0, sizeof(float) * 8, pStream));
    } else if (vIsCapturing) {
      CUDA_CHECK(
          DeviceSegmentedRadixSort::SortPairs(nullptr, vTempStorageBytes, vTempKeys, vTempKeys, // keys (in-place)
                                              vTempIndices, pDst,                               // values (indices)
                                              pNcols * pNrows, pNrows, // num items, num segments
                                              vOffsetIterator, vOffsetIterator + 1, 0, sizeof(float) * 8, pStream));
    } else {
      CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, vTempStorageBytes, vTempKeys,
                                                vTempKeys,               // keys (in-place)
                                                vTempIndices, pDst,      // values (indices)
                                                pNcols * pNrows, pNrows, // num items, num segments
                                                vOffsetIterator, vOffsetIterator + 1, pStream));
    }
  } else {
    if (pNrows == 1) {
      CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, vTempStorageBytes, vTempKeys,
                                                      vTempKeys,          // keys (in-place)
                                                      vTempIndices, pDst, // values (indices)
                                                      pNcols, 0, sizeof(float) * 8, pStream));
    } else if (vIsCapturing) {
      CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
          nullptr, vTempStorageBytes, vTempKeys, vTempKeys, vTempIndices, pDst, pNcols * pNrows, pNrows,
          vOffsetIterator, vOffsetIterator + 1, 0, sizeof(float) * 8, pStream));
    } else {
      CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, vTempStorageBytes, vTempKeys, vTempKeys,
                                                          vTempIndices, pDst, pNcols * pNrows, pNrows, vOffsetIterator,
                                                          vOffsetIterator + 1, pStream));
    }
  }

  ggml_cuda_pool_alloc<uint8_t> vTempStorageAlloc(pPool, vTempStorageBytes);
  void *vDTempStorage = vTempStorageAlloc.get();

  if (pOrder == GGML_SORT_ORDER_ASC) {
    if (pNrows == 1) {
      CUDA_CHECK(DeviceRadixSort::SortPairs(vDTempStorage, vTempStorageBytes, vTempKeys,
                                            vTempKeys,          // keys (in-place)
                                            vTempIndices, pDst, // values (indices)
                                            pNcols, 0, sizeof(float) * 8, pStream));
    } else if (vIsCapturing) {
      CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(vDTempStorage, vTempStorageBytes, vTempKeys, vTempKeys,
                                                     vTempIndices, pDst, pNcols * pNrows, pNrows, vOffsetIterator,
                                                     vOffsetIterator + 1, 0, sizeof(float) * 8, pStream));
    } else {
      CUDA_CHECK(DeviceSegmentedSort::SortPairs(vDTempStorage, vTempStorageBytes, vTempKeys, vTempKeys, vTempIndices,
                                                pDst, pNcols * pNrows, pNrows, vOffsetIterator, vOffsetIterator + 1,
                                                pStream));
    }
  } else {
    if (pNrows == 1) {
      CUDA_CHECK(DeviceRadixSort::SortPairsDescending(vDTempStorage, vTempStorageBytes, vTempKeys,
                                                      vTempKeys,          // keys (in-place)
                                                      vTempIndices, pDst, // values (indices)
                                                      pNcols, 0, sizeof(float) * 8, pStream));
    } else if (vIsCapturing) {
      CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
          vDTempStorage, vTempStorageBytes, vTempKeys, vTempKeys, vTempIndices, pDst, pNcols * pNrows, pNrows,
          vOffsetIterator, vOffsetIterator + 1, 0, sizeof(float) * 8, pStream));
    } else {
      CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(vDTempStorage, vTempStorageBytes, vTempKeys, vTempKeys,
                                                          vTempIndices, pDst, pNcols * pNrows, pNrows, vOffsetIterator,
                                                          vOffsetIterator + 1, pStream));
    }
  }
}
#endif // GGML_CUDA_USE_CUB

// Bitonic sort implementation
template <typename T> static inline __device__ void fGgmlCudaSwap(T &pA, T &pB) {
  T vTmp = pA;
  pA = pB;
  pB = vTmp;
}

template <ggml_sort_order order>
static __global__ void fKArgsortF32I32(const float *pX, int *pDst, const int pNcols, int pNcolsPad) {
  // bitonic sort
  int vCol = threadIdx.x;
  int vRow = blockIdx.x;

  if (vCol >= pNcolsPad) {
    return;
  }

  const float *vXRow = pX + vRow * pNcols;
  extern __shared__ int aDstRow[];

  // initialize indices
  aDstRow[vCol] = vCol;

  __syncthreads();

  for (int vK = 2; vK <= pNcolsPad; vK *= 2) {
    for (int vJ = vK / 2; vJ > 0; vJ /= 2) {
      int vIxj = vCol ^ vJ;
      if (vIxj > vCol) {
        if ((vCol & vK) == 0) {
          if (aDstRow[vCol] >= pNcols ||
              (aDstRow[vIxj] < pNcols &&
               (order == GGML_SORT_ORDER_ASC ? vXRow[aDstRow[vCol]] > vXRow[aDstRow[vIxj]]
                                             : vXRow[aDstRow[vCol]] < vXRow[aDstRow[vIxj]]))) {
            fGgmlCudaSwap(aDstRow[vCol], aDstRow[vIxj]);
          }
        } else {
          if (aDstRow[vIxj] >= pNcols ||
              (aDstRow[vCol] < pNcols &&
               (order == GGML_SORT_ORDER_ASC ? vXRow[aDstRow[vCol]] < vXRow[aDstRow[vIxj]]
                                             : vXRow[aDstRow[vCol]] > vXRow[aDstRow[vIxj]]))) {
            fGgmlCudaSwap(aDstRow[vCol], aDstRow[vIxj]);
          }
        }
      }
      __syncthreads();
    }
  }

  // copy the result to dst without the padding
  if (vCol < pNcols) {
    pDst[vRow * pNcols + vCol] = aDstRow[vCol];
  }
}

static int fNextPowerOf2(int pX) {
  int vN = 1;
  while (vN < pX) {
    vN *= 2;
  }
  return vN;
}

void argsort_f32_i32_cuda_bitonic(const float *pX, int *pDst, const int pNcols, const int pNrows,
                                  ggml_sort_order pOrder, cudaStream_t pStream) {
  // bitonic sort requires ncols to be power of 2
  const int cNcolsPad = fNextPowerOf2(pNcols);

  const dim3 cBlockDims(cNcolsPad, 1, 1);
  const dim3 cBlockNums(pNrows, 1, 1);
  const size_t cSharedMem = cNcolsPad * sizeof(int);

  // FIXME: this limit could be raised by ~2-4x on Ampere or newer
  GGML_ASSERT(cSharedMem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

  if (pOrder == GGML_SORT_ORDER_ASC) {
    fKArgsortF32I32<GGML_SORT_ORDER_ASC><<<cBlockNums, cBlockDims, cSharedMem, pStream>>>(pX, pDst, pNcols, cNcolsPad);
  } else if (pOrder == GGML_SORT_ORDER_DESC) {
    fKArgsortF32I32<GGML_SORT_ORDER_DESC><<<cBlockNums, cBlockDims, cSharedMem, pStream>>>(pX, pDst, pNcols, cNcolsPad);
  } else {
    GGML_ABORT("fatal error");
  }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_I32);
  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNcols = vSrc0->ne[0];
  const int64_t cNrows = ggml_nrows(vSrc0);

  enum ggml_sort_order vOrder = (enum ggml_sort_order)pDst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
  const int cNcolsPad = fNextPowerOf2(cNcols);
  const size_t cSharedMem = cNcolsPad * sizeof(int);
  const size_t cMaxSharedMem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

  // early return if we can use bitonic argsort
  if (cSharedMem <= cMaxSharedMem && cNcols <= 1024) {
    argsort_f32_i32_cuda_bitonic(vSrc0D, (int *)vDstD, cNcols, cNrows, vOrder, vStream);
    return;
  }

  const int cChunkNrows = argsort_f32_i32_cuda_cub_chunk_nrows(vSrc0->nb[1], cNrows);

  ggml_cuda_pool &vPool = pCtx.pool();

  for (int64_t vI = 0; vI < cNrows; vI += cChunkNrows) {
    int vIterNrows = std::min((int64_t)cChunkNrows, cNrows - vI);

    argsort_f32_i32_cuda_cub(vPool, vSrc0D, (int *)vDstD, cNcols, vIterNrows, vOrder, vStream);

    vSrc0D += cNcols * vIterNrows;
    vDstD += cNcols * vIterNrows;
  }
#else
  argsort_f32_i32_cuda_bitonic(vSrc0D, (int *)vDstD, cNcols, cNrows, vOrder, vStream);
#endif
}
