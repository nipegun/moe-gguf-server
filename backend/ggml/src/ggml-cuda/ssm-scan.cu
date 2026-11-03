#if !defined(GGML_USE_HIP) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#include "ssm-scan.cuh"

// Minimum number of tokens to use SSD (State Space Duality) matmul path instead of scan path.
// For n_tok <= this threshold, the scan kernel is used (lower overhead for short sequences).
#define SSM_SSD_MIN_TOKENS 128

// prepare_dt kernel dimensions: one block per (head, seq), each block handles DT_MAX_ITEMS items.
#define SSM_SSD_DT_BLOCK 256
#define SSM_SSD_DT_MAX_ITEMS 32

// Maximum tokens the SSD path supports, derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)

// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.
#define SSM_SSD_CHUNK_SIZE 256

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    fSsmScanF32(const float *pSrc0Ptr, const float *pSrc1Ptr, const float *pSrc2Ptr, const float *pSrc3Ptr,
                const float *pSrc4Ptr, const float *pSrc5Ptr, const int32_t *pSrc6Ptr, float *pDstPtr,
                const int pSrc0Nb2, const int pSrc0Nb3, const int pSrc1Nb2, const int pSrc1Nb3, const int pSrc2Nb1,
                const int pSrc2Nb2, const int pSrc3Nb1, const int pSrc4Nb2, const int pSrc4Nb3, const int pSrc5Nb2,
                const int pSrc5Nb3, const int64_t pSOff, const int64_t pDInner, const int64_t pLParam) {
  const float *GGML_CUDA_RESTRICT vSrc0 = pSrc0Ptr;
  const float *GGML_CUDA_RESTRICT vSrc1 = pSrc1Ptr;
  const float *GGML_CUDA_RESTRICT vSrc2 = pSrc2Ptr;
  const float *GGML_CUDA_RESTRICT vSrc3 = pSrc3Ptr;
  const float *GGML_CUDA_RESTRICT vSrc4 = pSrc4Ptr;
  const float *GGML_CUDA_RESTRICT vSrc5 = pSrc5Ptr;
  const int32_t *GGML_CUDA_RESTRICT vSrc6 = pSrc6Ptr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const size_t cL = L_template == 0 ? pLParam : L_template;
  ggml_cuda_pdl_sync();
  const float *vS0Block =
      (const float *)((const char *)vSrc0 + vSrc6[blockIdx.x] * pSrc0Nb3 + blockIdx.y * splitD * pSrc0Nb2);
  const float *vXBlock =
      (const float *)((const char *)vSrc1 + (blockIdx.x * pSrc1Nb3) + blockIdx.y * splitD * sizeof(float));
  const float *vDtBlock =
      (const float *)((const char *)vSrc2 + (blockIdx.x * pSrc2Nb2) + blockIdx.y * splitD * sizeof(float));
  const float *vABlock = (const float *)((const char *)vSrc3 + blockIdx.y * splitD * pSrc3Nb1);
  const float *vBBlock = (const float *)((const char *)vSrc4 + (blockIdx.x * pSrc4Nb3));
  const float *vCBlock = (const float *)((const char *)vSrc5 + (blockIdx.x * pSrc5Nb3));
  float *vYBlock =
      (float *)((char *)vDst + (blockIdx.x * pDInner * cL * sizeof(float)) + blockIdx.y * splitD * sizeof(float));
  float *vSBlock = (float *)((char *)vDst + pSOff + blockIdx.x * pSrc0Nb3 + blockIdx.y * splitD * pSrc0Nb2);

  const int cStrideX = pSrc1Nb2 / sizeof(float);
  const int cStrideDt = pSrc2Nb1 / sizeof(float);
  const int cStrideB = pSrc4Nb2 / sizeof(float);
  const int cStrideC = pSrc5Nb2 / sizeof(float);
  const int cStrideY = pDInner;

  float vRegA[N];
  float vRegs0[N];

  __shared__ float vSmemB[N];
  __shared__ float vSmemC[N];

#ifdef USE_CUB
  using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

  union CubTempStorage {
    typename BlockLoad::TempStorage vLoadTemp;
    typename BlockStore::TempStorage vStoreTemp;
  };
  __shared__ CubTempStorage vCubTempStorage;

  BlockLoad(vCubTempStorage.vLoadTemp).Load(vABlock, vRegA);
  __syncthreads();
  BlockLoad(vCubTempStorage.vLoadTemp).Load(vS0Block, vRegs0);
#else
  const int stride_s0 = pSrc0Nb2 / sizeof(float);
  const int stride_A = pSrc3Nb1 / sizeof(float);
#pragma unroll
  for (size_t vN = 0; vN < N; ++vN) {
    vRegA[vN] = vABlock[threadIdx.x * stride_A + vN];
    vRegs0[vN] = vS0Block[threadIdx.x * stride_s0 + vN];
  }
#endif

#pragma unroll
  for (size_t vI = 0; vI < cL; vI++) {
    if (threadIdx.x < N) {
      vSmemB[threadIdx.x] = vBBlock[vI * cStrideB + threadIdx.x];
      vSmemC[threadIdx.x] = vCBlock[vI * cStrideC + threadIdx.x];
    }
    __syncthreads();

    float vDtSoftPlus = vDtBlock[vI * cStrideDt + threadIdx.x];
    if (vDtSoftPlus <= 20.0f) {
      vDtSoftPlus = log1pf(expf(vDtSoftPlus));
    }
    float vXDt = vXBlock[vI * cStrideX + threadIdx.x] * vDtSoftPlus;

    float vSumf = 0.0f;
#pragma unroll
    for (size_t vN = 0; vN < N; vN++) {
      float vState = vRegs0[vN] * expf(vDtSoftPlus * vRegA[vN]) + vSmemB[vN] * vXDt;
      vSumf += vState * vSmemC[vN];
      vRegs0[vN] = vState;
    }
    vYBlock[vI * cStrideY + threadIdx.x] = vSumf;
    __syncthreads();
  }

#ifdef USE_CUB
  BlockStore(vCubTempStorage.vStoreTemp).Store(vSBlock, vRegs0);
#else
  const int stride_s = stride_s0;
#pragma unroll
  for (size_t vN = 0; vN < N; ++vN) {
    vSBlock[threadIdx.x * stride_s + vN] = vRegs0[vN];
  }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// assumes as many threads as d_state
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    fSsmScanF32Group(const float *pSrc0Ptr, const float *pSrc1Ptr, const float *pSrc2Ptr, const float *pSrc3Ptr,
                     const float *pSrc4Ptr, const float *pSrc5Ptr, const int32_t *pSrc6Ptr, float *pDstPtr,
                     const int pSrc0Nb2, const int pSrc0Nb3, const int pSrc1Nb2, const int pSrc1Nb3, const int pSrc2Nb1,
                     const int pSrc2Nb2, const int pSrc3Nb1, const int pSrc4Nb2, const int pSrc4Nb3, const int pSrc5Nb2,
                     const int pSrc5Nb3, const int64_t pSOff, const int64_t pNHead, const int64_t pDHead,
                     const int64_t pNGroup, const int64_t pNTok) {
  const float *GGML_CUDA_RESTRICT vSrc0 = pSrc0Ptr;
  const float *GGML_CUDA_RESTRICT vSrc1 = pSrc1Ptr;
  const float *GGML_CUDA_RESTRICT vSrc2 = pSrc2Ptr;
  const float *GGML_CUDA_RESTRICT vSrc3 = pSrc3Ptr;
  const float *GGML_CUDA_RESTRICT vSrc4 = pSrc4Ptr;
  const float *GGML_CUDA_RESTRICT vSrc5 = pSrc5Ptr;
  const int32_t *GGML_CUDA_RESTRICT vSrc6 = pSrc6Ptr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;

  const int cWarp = threadIdx.x / WARP_SIZE;
  const int cLane = threadIdx.x % WARP_SIZE;
  const int cWarpIdx = blockIdx.x * c_factor + cWarp;

  const int cHeadIdx = cWarpIdx / pDHead;
  const int cHeadOff = (cWarpIdx % pDHead) * sizeof(float);
  const int cSeqIdx = blockIdx.y;

  const int cGroupOff = (cHeadIdx / (pNHead / pNGroup)) * d_state * sizeof(float);

  ggml_cuda_pdl_sync();
  // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the
  // codebase
  const float *vS0Warp =
      (const float *)((const char *)vSrc0 + vSrc6[cSeqIdx] * pSrc0Nb3 + cHeadIdx * pSrc0Nb2 + cHeadOff * d_state);
  const float *vXWarp = (const float *)((const char *)vSrc1 + (cSeqIdx * pSrc1Nb3) + (cWarpIdx * sizeof(float)));
  const float *vDtWarp = (const float *)((const char *)vSrc2 + (cSeqIdx * pSrc2Nb2) + cHeadIdx * sizeof(float));
  const float *vAWarp = (const float *)((const char *)vSrc3 + cHeadIdx * pSrc3Nb1);
  const float *vBWarp = (const float *)((const char *)vSrc4 + (cSeqIdx * pSrc4Nb3) + (cGroupOff));
  const float *vCWarp = (const float *)((const char *)vSrc5 + (cSeqIdx * pSrc5Nb3) + (cGroupOff));
  float *vYWarp = vDst + (cSeqIdx * pNTok * pNHead * pDHead) + cWarpIdx;
  float *vSWarp = (float *)((char *)vDst + pSOff + cSeqIdx * pSrc0Nb3 + cHeadIdx * pSrc0Nb2 + cHeadOff * d_state);

  // strides across n_seq_tokens
  const int cStrideX = pSrc1Nb2 / sizeof(float);
  const int cStrideDt = pSrc2Nb1 / sizeof(float);
  const int cStrideB = pSrc4Nb2 / sizeof(float);
  const int cStrideC = pSrc5Nb2 / sizeof(float);
  const int cStrideY = pNHead * pDHead;

  float aState[c_factor];
  float vStateSum = 0.0f;

#pragma unroll
  for (int vJ = 0; vJ < c_factor; vJ++) {
    aState[vJ] = vS0Warp[WARP_SIZE * vJ + cLane];
  }

  for (int64_t vI = 0; vI < pNTok; vI++) {
    // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
    // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
    const float cDtSoftPlus =
        (vDtWarp[vI * cStrideDt] <= 20.0f ? log1pf(expf(vDtWarp[vI * cStrideDt])) : vDtWarp[vI * cStrideDt]);

    vStateSum = 0.0f;
    const float cDA = expf(cDtSoftPlus * vAWarp[0]);
    const float cXDt = vXWarp[vI * cStrideX] * cDtSoftPlus;
#pragma unroll
    for (int vJ = 0; vJ < c_factor; vJ++) {
      const float cBVal = vBWarp[vI * cStrideB + WARP_SIZE * vJ + cLane];
      const float cCVal = vCWarp[vI * cStrideC + WARP_SIZE * vJ + cLane];
      aState[vJ] = (aState[vJ] * cDA) + (cBVal * cXDt);
      vStateSum += aState[vJ] * cCVal;
    }

    // parallel accumulation for output
    vStateSum = warp_reduce_sum(vStateSum);

    if (cLane == 0) {
      vYWarp[vI * cStrideY] = vStateSum;
    }
  }

  // write back the state
#pragma unroll
  for (int vJ = 0; vJ < c_factor; vJ++) {
    vSWarp[WARP_SIZE * vJ + cLane] = aState[vJ];
  }
}

static void fSsmScanF32Cuda(const float *pSrc0, const float *pSrc1, const float *pSrc2, const float *pSrc3,
                            const float *pSrc4, const float *pSrc5, const int32_t *pSrc6, float *pDst,
                            const int pSrc0Nb2, const int pSrc0Nb3, const int pSrc1Nb2, const int pSrc1Nb3,
                            const int pSrc2Nb1, const int pSrc2Nb2, const int pSrc3Nb1, const int pSrc4Nb2,
                            const int pSrc4Nb3, const int pSrc5Nb2, const int pSrc5Nb3, const int64_t pSOff,
                            const int64_t pDState, const int64_t pHeadDim, const int64_t pNHead, const int64_t pNGroup,
                            const int64_t pNTok, const int64_t pNSeq, cudaStream_t pStream) {
  // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
  if (pSrc3Nb1 == sizeof(float)) {
    // Mamba-2
    if (pDState == 128) {
      constexpr int cThreads = 128;
      constexpr int cNumWarps = cThreads / WARP_SIZE;

      const dim3 cBlocks((pNHead * pHeadDim + (cNumWarps - 1)) / cNumWarps, pNSeq, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocks, cThreads, 0, pStream);
      ggml_cuda_kernel_launch(fSsmScanF32Group<128 / WARP_SIZE, 128>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4,
                              pSrc5, pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                              pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pHeadDim, pNGroup, pNTok);
    } else if (pDState == 256) { // Falcon-H1
      constexpr int cThreads = 256;
      constexpr int cNumWarps = cThreads / WARP_SIZE;

      const dim3 cBlocks((pNHead * pHeadDim + (cNumWarps - 1)) / cNumWarps, pNSeq, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocks, cThreads, 0, pStream);
      ggml_cuda_kernel_launch(fSsmScanF32Group<256 / WARP_SIZE, 256>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4,
                              pSrc5, pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                              pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pHeadDim, pNGroup, pNTok);
    } else {
      GGML_ABORT("doesn't support d_state!=(128 or 256).");
    }
  } else {
    // Mamba-1
    constexpr int cThreads = 128;
    GGML_ASSERT(pNHead % cThreads == 0);
    GGML_ASSERT(pHeadDim == 1);
    GGML_ASSERT(pNGroup == 1);
    const dim3 cBlocks(pNSeq, (pNHead + cThreads - 1) / cThreads, 1);
    if (pDState == 16) {
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocks, cThreads, 0, pStream);
      switch (pNTok) {
      case 1:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 1>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 2:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 2>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 3:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 3>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 4:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 4>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 5:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 5>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 6:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 6>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 7:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 7>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      case 8:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 8>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      default:
        ggml_cuda_kernel_launch(fSsmScanF32<cThreads, 16, 0>, cLaunchParams, pSrc0, pSrc1, pSrc2, pSrc3, pSrc4, pSrc5,
                                pSrc6, pDst, pSrc0Nb2, pSrc0Nb3, pSrc1Nb2, pSrc1Nb3, pSrc2Nb1, pSrc2Nb2, pSrc3Nb1,
                                pSrc4Nb2, pSrc4Nb3, pSrc5Nb2, pSrc5Nb3, pSOff, pNHead, pNTok);
        break;
      }
    } else {
      GGML_ABORT("doesn't support d_state!=16.");
    }
  }
}

#if !defined(GGML_USE_HIP)
// ============================================================================
// SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS)
//
// Instead of a sequential scan, SSD reformulates the output as:
//   Y = (L (.) (C @ B^T)) @ (X * dt)  +  decay * C @ s_init
// where L is a causal decay mask derived from A and dt.
//
// This converts the O(T*N) sequential scan into parallel matmuls.
// ============================================================================
// Softplus(dt) and inclusive prefix sum per head using CUB BlockScan.
// Grid: (n_head, n_seqs)
template <int BLOCK_SIZE, int MAX_ITEMS>
__global__ void fSsmSsdPrepareDtKernel(const float *__restrict__ pDtRaw, float *__restrict__ pDtSpOut,
                                       float *__restrict__ pCsOut, const int pNHead, const int pNTok,
                                       const int pDtStrideTok,   // elements between tokens in dt
                                       const int pDtStrideSeq) { // elements between sequences in dt

  const int cH = blockIdx.x;
  const int cS = blockIdx.y;

  const float *vDtSeq = pDtRaw + cS * pDtStrideSeq;

  float *vDtSpSeq = pDtSpOut + cS * pNTok * pNHead;
  float *vCsSeq = pCsOut + cS * pNTok * pNHead;

  const int cItemsPerThread = (pNTok + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Phase 1: softplus with interleaved distribution (t = i*BLOCK_SIZE + threadIdx.x).
  // Each warp reads BLOCK_SIZE consecutive tokens, giving coalesced dt_raw loads
  // (stride n_head between threads vs. items_per_thread*n_head in blocked layout).
  float vLocalVals[MAX_ITEMS];
  for (int vI = 0; vI < cItemsPerThread; vI++) {
    const int cT = vI * BLOCK_SIZE + threadIdx.x;
    if (cT < pNTok) {
      float vVal = vDtSeq[cH + cT * pDtStrideTok];
      float vSp = (vVal <= 20.0f) ? log1pf(expf(vVal)) : vVal;
      vLocalVals[vI] = vSp;
      vDtSpSeq[cT * pNHead + cH] = vSp;
    } else {
      vLocalVals[vI] = 0.0f;
    }
  }

  // Phase 2+3: per-step inclusive scan to build cs[] in token order.
  // With interleaved distribution the per-thread total scan would not give token-order
  // prefix sums, so we scan one BLOCK_SIZE slab at a time and carry a running total.
#ifdef USE_CUB
  using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage vScanTemp;
  __shared__ float vStepTotal;

  float vRunning = 0.0f;
  for (int vI = 0; vI < cItemsPerThread; vI++) {
    float vInclusive;
    BlockScan(vScanTemp).InclusiveSum(vLocalVals[vI], vInclusive);
    const int cT = vI * BLOCK_SIZE + threadIdx.x;
    if (cT < pNTok) {
      vCsSeq[cT * pNHead + cH] = vRunning + vInclusive;
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
      vStepTotal = vInclusive;
    }
    __syncthreads();
    vRunning += vStepTotal;
  }
#else
  // Fallback: sequential prefix scan in shared memory, one slab at a time.
  __shared__ float sdata[BLOCK_SIZE];
  float vRunning = 0.0f;
  for (int vI = 0; vI < cItemsPerThread; vI++) {
    const int cT = vI * BLOCK_SIZE + threadIdx.x;
    sdata[threadIdx.x] = vLocalVals[vI];
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int j = 1; j < BLOCK_SIZE; j++) {
        sdata[j] += sdata[j - 1];
      }
    }
    __syncthreads();
    if (cT < pNTok) {
      vCsSeq[cT * pNHead + cH] = vRunning + sdata[threadIdx.x];
    }
    vRunning += sdata[BLOCK_SIZE - 1];
    __syncthreads();
  }
#endif
}

// Prepare SSD matmul inputs for one chunk: X_dt, B_weighted, C_scaled.
// T_matmul controls precision for X_dt, B_weighted (float or half).
// C_scaled is always float (pairs with float s_cur in step 3c).
// Computation is always FP32; only the final store converts to T_matmul.
// Also materializes the causal M matrix = exp(A*(cs_out - cs_in)) * CB (fused with prep to save a launch).
// Grid: (ceil(max(C*head_dim, d_state*C, chunk_len^2) / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE, typename T_matmul>
__global__ void
fSsmSsdPreMatmulKernel(const float *__restrict__ pCs,     // {n_tok, n_head} cumulative dt sums
                       const float *__restrict__ pDtSp,   // {n_tok, n_head} softplus(dt)
                       const float *__restrict__ pA,      // {1, n_head}
                       const float *__restrict__ pX,      // {head_dim, n_head, n_tok, n_seqs}
                       const float *__restrict__ pB,      // {d_state, n_group, n_tok, n_seqs}
                       const float *__restrict__ pCSrc,   // {d_state, n_group, n_tok, n_seqs}
                       T_matmul *__restrict__ pXDt,       // {head_dim, C, n_head} x * dt, d-fastest
                       T_matmul *__restrict__ pBWeighted, // {d_state, C, n_head} B * decay_from_end
                       float *__restrict__ pCScaled,      // {d_state, C, n_head} C * decay_to_pos (always float)
                       const float *__restrict__ pCb,     // {chunk_len, chunk_len, n_group, n_seqs}
                       half *__restrict__ pMOut,          // {chunk_len, chunk_len, n_head, n_seqs}
                       const int pChunkLen, const int pHeadDim, const int pNHead, const int pNGroup, const int pDState,
                       const int pAStride, const int pXStrideTok, const int pXStrideSeq, const int pBStrideTok,
                       const int pBStrideSeq, const int pCStrideTok, const int pCStrideSeq, const int pChunkOffset,
                       const int pNTokTotal) {

  const int cH = blockIdx.y;
  const int cS = blockIdx.z;
  const int cG = cH / (pNHead / pNGroup);

  const float cAH = pA[cH * pAStride];
  const int cIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  const int cCsSeqOff = cS * pNTokTotal * pNHead;
  const float cCsBase = (pChunkOffset > 0) ? pCs[cCsSeqOff + (pChunkOffset - 1) * pNHead + cH] : 0.0f;
  const float cCsLast = pCs[cCsSeqOff + (pChunkOffset + pChunkLen - 1) * pNHead + cH] - cCsBase;

  // Prepare X_dt = x * dt, stored d-fastest for coalesced reads and writes.
  const int cNXdt = pChunkLen * pHeadDim;
  if (cIdx < cNXdt) {
    const int cD = cIdx % pHeadDim;
    const int cT = cIdx / pHeadDim;

    const float cXVal = pX[cS * pXStrideSeq + (pChunkOffset + cT) * pXStrideTok + cD + cH * pHeadDim];
    const float cDtVal = pDtSp[cCsSeqOff + (pChunkOffset + cT) * pNHead + cH];

    pXDt[cD + cT * pHeadDim + cH * cNXdt + cS * cNXdt * pNHead] = (T_matmul)(cXVal * cDtVal);
  }

  // Prepare B_weighted and C_scaled together: both share the same index space (d_state * chunk_len)
  // and the same cs_t load, so merging halves the cs[] global memory traffic.
  const int cNBw = pDState * pChunkLen;
  if (cIdx < cNBw) {
    const int cN = cIdx % pDState;
    const int cT = cIdx / pDState;

    const float cCsT = pCs[cCsSeqOff + (pChunkOffset + cT) * pNHead + cH] - cCsBase;

    const float cBVal = pB[cS * pBStrideSeq + (pChunkOffset + cT) * pBStrideTok + cG * pDState + cN];
    pBWeighted[cN + cT * pDState + cH * cNBw + cS * cNBw * pNHead] = (T_matmul)(cBVal * __expf(cAH * (cCsLast - cCsT)));

    const float cCVal = pCSrc[cS * pCStrideSeq + (pChunkOffset + cT) * pCStrideTok + cG * pDState + cN];
    pCScaled[cN + cT * pDState + cH * cNBw + cS * cNBw * pNHead] = cCVal * __expf(cAH * cCsT);
  }

  // Materialize M = exp(A*(cs_out - cs_in)) * CB with causal mask.
  const int cNM = pChunkLen * pChunkLen;
  if (cIdx < cNM) {
    const int cTOut = cIdx % pChunkLen;
    const int cTIn = cIdx / pChunkLen;

    half vVal;
    if (cTIn <= cTOut) {
      const float cCsOut = pCs[cCsSeqOff + (pChunkOffset + cTOut) * pNHead + cH] - cCsBase;
      const float cCsIn = pCs[cCsSeqOff + (pChunkOffset + cTIn) * pNHead + cH] - cCsBase;
      const float cDecay = __expf(cAH * (cCsOut - cCsIn));
      const float *vCbG = pCb + (int64_t)cS * pChunkLen * pChunkLen * pNGroup + (int64_t)cG * pChunkLen * pChunkLen;
      const float cCbVal = vCbG[cTOut + cTIn * pChunkLen];
      vVal = __float2half(cDecay * cCbVal);
    } else {
      vVal = __float2half(0.0f);
    }

    pMOut[(int64_t)cS * cNM * pNHead + (int64_t)cH * cNM + cTIn * pChunkLen + cTOut] = vVal;
  }
}

// Scale running state in-place: s_cur *= decay_total(chunk).
// Called BEFORE cuBLAS state update (beta=1) to fuse inter-chunk decay.
// Eliminates the s_old buffer and D2D memcpy vs the old approach of:
//   memcpy(s_old, s_cur) -> cuBLAS(beta=0) -> s_cur += decay * s_old
// Grid: (ceil(d_state * head_dim / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE>
__global__ void fSsmSsdScaleStateKernel(float *__restrict__ pSCur,     // {d_state, head_dim, n_head, n_seqs}
                                        const float *__restrict__ pCs, // {n_tok, n_head} cumulative dt sums
                                        const float *__restrict__ pA,  // {1, n_head}
                                        const int pDState, const int pHeadDim, const int pNHead, const int pChunkOffset,
                                        const int pChunkLen, const int pNTokTotal, const int pAStride) {

  const int cH = blockIdx.y;
  const int cS = blockIdx.z;
  const int cIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int cStatePerHead = pDState * pHeadDim;
  if (cIdx >= cStatePerHead)
    return;

  const float cAH = pA[cH * pAStride];
  const int cCsSeqOff = cS * pNTokTotal * pNHead;
  const float cCsBase = (pChunkOffset > 0) ? pCs[cCsSeqOff + (pChunkOffset - 1) * pNHead + cH] : 0.0f;
  const float cCsLast = pCs[cCsSeqOff + (pChunkOffset + pChunkLen - 1) * pNHead + cH] - cCsBase;
  const float cDecayTotal = __expf(cAH * cCsLast);

  const int cOff = cS * cStatePerHead * pNHead + cH * cStatePerHead + cIdx;
  pSCur[cOff] *= cDecayTotal;
}

// Copy initial state from src0[ids[s]] into s_cur for each sequence.
// Grid: (ceil(d_state * head_dim * n_head / BLOCK), n_seqs)
template <int BLOCK_SIZE>
__global__ void fSsmSsdInitStateKernel(const float *__restrict__ pSrc0,  // {d_state, head_dim, n_head, n_rs}
                                       const int32_t *__restrict__ pIds, // {n_seqs}
                                       float *__restrict__ pSCur,        // {d_state, head_dim, n_head, n_seqs}
                                       const int pStateSize,             // d_state * head_dim * n_head
                                       const int64_t pS0StrideSeq) {     // elements between state rows
  const int cS = blockIdx.y;
  const int cIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (cIdx >= pStateSize)
    return;

  const float *vSSrc = pSrc0 + (int64_t)pIds[cS] * pS0StrideSeq;
  pSCur[cS * pStateSize + cIdx] = vSSrc[cIdx];
}

// SSD (State Space Duality) dispatch for Mamba-2 prefill.
// Chunked matmuls: CB, materialize M + cuBLAS Y, S@C, B@X_dt.
// All strides are in elements (floats), not bytes.
static void fSsmScanSsdF32Cuda(ggml_backend_cuda_context &pCtx, const float *pSrc0D, const float *pSrc1D,
                               const float *pSrc2D, const float *pSrc3D, const float *pSrc4D, const float *pSrc5D,
                               const int32_t *pSrc6D, float *pDstD,
                               const int64_t pS0StrideSeq,                     // state (src0) stride between seqs
                               const int pXStrideTok, const int pXStrideSeq,   // x (src1) strides
                               const int pDtStrideTok, const int pDtStrideSeq, // dt (src2) strides
                               const int pAStride,                             // A (src3) stride between heads
                               const int pBStrideTok, const int pBStrideSeq,   // B (src4) strides
                               const int pCStrideTok, const int pCStrideSeq,   // C (src5) strides
                               const int64_t pSOff, const int64_t pDState, const int64_t pHeadDim, const int64_t pNHead,
                               const int64_t pNGroup, const int64_t pNTok, const int64_t pNSeq) {

  cudaStream_t vStream = pCtx.stream();
  const int64_t cDInner = pHeadDim * pNHead;

  const int64_t cChunkSize = SSM_SSD_CHUNK_SIZE;
  const int64_t cNChunks = (pNTok + cChunkSize - 1) / cChunkSize;

  const int64_t cStatePerHead = pDState * pHeadDim;

  using matmul_t = half;
  static constexpr cudaDataType_t cMatmulDtype = CUDA_R_16F;

  ggml_cuda_pool_alloc<float> vDtSpBuf(pCtx.pool(), pNTok * pNHead * pNSeq);
  ggml_cuda_pool_alloc<float> vCsBuf(pCtx.pool(), pNTok * pNHead * pNSeq);
  ggml_cuda_pool_alloc<float> vCbBuf(pCtx.pool(), cChunkSize * cChunkSize * pNGroup * pNSeq);
  ggml_cuda_pool_alloc<matmul_t> vXDtBuf(pCtx.pool(), cChunkSize * pHeadDim * pNHead * pNSeq);
  ggml_cuda_pool_alloc<matmul_t> vBWBuf(pCtx.pool(), pDState * cChunkSize * pNHead * pNSeq);
  ggml_cuda_pool_alloc<float> vCSBuf(pCtx.pool(), pDState * cChunkSize * pNHead * pNSeq);
  float *vDtSp = vDtSpBuf.get();
  float *vCs = vCsBuf.get();
  float *vCb = vCbBuf.get();
  matmul_t *vXDt = vXDtBuf.get();
  matmul_t *vBWeighted = vBWBuf.get();
  float *vCScaled = vCSBuf.get();
  float *vSCur = (float *)((char *)pDstD + pSOff); // write state directly to dst

  // Step 1: softplus(dt) and parallel prefix sum over full sequence
  {
    dim3 vGrid(pNHead, pNSeq);
    fSsmSsdPrepareDtKernel<SSM_SSD_DT_BLOCK, SSM_SSD_DT_MAX_ITEMS>
        <<<vGrid, SSM_SSD_DT_BLOCK, 0, vStream>>>(pSrc2D, vDtSp, vCs, pNHead, pNTok, pDtStrideTok, pDtStrideSeq);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 2: initialize running state from src0[ids[s]]
  {
    constexpr int cBlock = 256;
    const int64_t cStateSize = pDState * pHeadDim * pNHead;
    dim3 vGrid((cStateSize + cBlock - 1) / cBlock, pNSeq);
    fSsmSsdInitStateKernel<cBlock><<<vGrid, cBlock, 0, vStream>>>(pSrc0D, pSrc6D, vSCur, cStateSize, pS0StrideSeq);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 3: chunked SSD loop
  // Per chunk: pre_matmul (incl. M) + 4 cuBLAS (CB, Y, S@C, state update) + scale_state
  cublasHandle_t vHandle = pCtx.cublas_handle();
  CUBLAS_CHECK(cublasSetStream(vHandle, vStream));
  const float cAlphaOne = 1.0f;
  const float cBetaZero = 0.0f;
  const float cBetaOne = 1.0f;
  const int cLdaCSrc = pCStrideTok; // leading dim for C in CB = C^T @ B
  const int cLdbBSrc = pBStrideTok; // leading dim for B in CB = C^T @ B

  // Scratch buffer for causal M matrix, reused across chunks (max size at chunk_size)
  const int64_t cNMMax = cChunkSize * cChunkSize;
  ggml_cuda_pool_alloc<half> vMBuf(pCtx.pool(), cNMMax * pNHead * pNSeq);
  half *vMMat = vMBuf.get();

  for (int64_t vK = 0; vK < cNChunks; vK++) {
    const int64_t cChunkOffset = vK * cChunkSize;
    const int64_t cChunkLen = (cChunkOffset + cChunkSize <= pNTok) ? cChunkSize : (pNTok - cChunkOffset);

    // 3a: CB = C^T @ B per group
    for (int64_t vS = 0; vS < pNSeq; vS++) {
      const float *vCS = pSrc5D + vS * pCStrideSeq + cChunkOffset * pCStrideTok;
      const float *vBS = pSrc4D + vS * pBStrideSeq + cChunkOffset * pBStrideTok;
      float *vCbS = vCb + vS * cChunkLen * cChunkLen * pNGroup;

      if (pNGroup == 1) {
        CUBLAS_CHECK(cublasSgemm(vHandle, CUBLAS_OP_T, CUBLAS_OP_N, cChunkLen, cChunkLen, pDState, &cAlphaOne, vCS,
                                 cLdaCSrc, vBS, cLdbBSrc, &cBetaZero, vCbS, (int)cChunkLen));
      } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            vHandle, CUBLAS_OP_T, CUBLAS_OP_N, cChunkLen, cChunkLen, pDState, &cAlphaOne, vCS, CUDA_R_32F, cLdaCSrc,
            pDState, vBS, CUDA_R_32F, cLdbBSrc, pDState, &cBetaZero, vCbS, CUDA_R_32F, (int)cChunkLen,
            (long long)(cChunkLen * cChunkLen), pNGroup, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }

    // 3b: prepare X_dt, B_weighted, C_scaled + materialize causal M matrix
    const int64_t cNM = cChunkLen * cChunkLen;
    {
      constexpr int cBlock = 256;
      const int64_t cNXdt = cChunkLen * pHeadDim;
      const int64_t cNBw = pDState * cChunkLen;
      int64_t vMaxWork = cNXdt;
      if (cNBw > vMaxWork)
        vMaxWork = cNBw;
      if (cNM > vMaxWork)
        vMaxWork = cNM;
      dim3 vGrid((vMaxWork + cBlock - 1) / cBlock, pNHead, pNSeq);
      fSsmSsdPreMatmulKernel<cBlock, matmul_t><<<vGrid, cBlock, 0, vStream>>>(
          vCs, vDtSp, pSrc3D, pSrc1D, pSrc4D, pSrc5D, vXDt, vBWeighted, vCScaled, vCb, vMMat, cChunkLen, pHeadDim,
          pNHead, pNGroup, pDState, pAStride, pXStrideTok, pXStrideSeq, pBStrideTok, pBStrideSeq, pCStrideTok,
          pCStrideSeq, cChunkOffset, pNTok);
      CUDA_CHECK(cudaGetLastError());
    }

    // 3c: dst = S_cur^T @ C_scaled (state contribution)
    {
      const int64_t cStrideS = cStatePerHead;
      const int64_t cStrideCs = pDState * cChunkLen;

      for (int64_t vS = 0; vS < pNSeq; vS++) {
        float *vDstChunk = pDstD + vS * cDInner * pNTok + cChunkOffset * cDInner;

        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            vHandle, CUBLAS_OP_T, CUBLAS_OP_N, pHeadDim, cChunkLen, pDState, &cAlphaOne, vSCur + vS * cStrideS * pNHead,
            CUDA_R_32F, pDState, cStrideS, vCScaled + vS * cStrideCs * pNHead, CUDA_R_32F, pDState, cStrideCs,
            &cBetaZero, vDstChunk, CUDA_R_32F, cDInner, pHeadDim, pNHead, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }

    // 3d: dst += X_dt @ M^T (intra-chunk contribution, adds to 3c result)
    // M is stored as M[t_out, t_in] (lower-triangular), transpose needed for Y = X @ M^T.
    {
      const int64_t cStrideM = cNM;
      const int64_t cStrideXH = (int64_t)cChunkLen * pHeadDim;

      for (int64_t vS = 0; vS < pNSeq; vS++) {
        float *vDstChunk = pDstD + vS * cDInner * pNTok + cChunkOffset * cDInner;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(vHandle, CUBLAS_OP_N, CUBLAS_OP_T, pHeadDim, cChunkLen, cChunkLen,
                                                &cAlphaOne, vXDt + vS * cStrideXH * pNHead, cMatmulDtype, pHeadDim,
                                                cStrideXH, vMMat + vS * cStrideM * pNHead, cMatmulDtype, cChunkLen,
                                                cStrideM, &cBetaOne, vDstChunk, CUDA_R_32F, cDInner, pHeadDim, pNHead,
                                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }

    // 3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)
    {
      // Scale s_cur in-place by per-head decay_total BEFORE cuBLAS overwrites it
      constexpr int cBlock = 256;
      dim3 vGrid((cStatePerHead + cBlock - 1) / cBlock, pNHead, pNSeq);
      fSsmSsdScaleStateKernel<cBlock><<<vGrid, cBlock, 0, vStream>>>(vSCur, vCs, pSrc3D, pDState, pHeadDim, pNHead,
                                                                     cChunkOffset, cChunkLen, pNTok, pAStride);
      CUDA_CHECK(cudaGetLastError());

      // cuBLAS with beta=1: s_cur = B_weighted @ X_dt^T + 1.0 * s_cur (already scaled)
      const int64_t cStrideBw = pDState * cChunkLen;
      const int64_t cStrideX = cChunkLen * pHeadDim;
      const int64_t cStrideS = cStatePerHead;

      for (int64_t vS = 0; vS < pNSeq; vS++) {
        // X_dt is d-fastest {hd, C}, read as OP_T to get {C, hd}
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(vHandle, CUBLAS_OP_N, CUBLAS_OP_T, pDState, pHeadDim, cChunkLen,
                                                &cAlphaOne, vBWeighted + vS * cStrideBw * pNHead, cMatmulDtype, pDState,
                                                cStrideBw, vXDt + vS * cStrideX * pNHead, cMatmulDtype, pHeadDim,
                                                cStrideX, &cBetaOne, vSCur + vS * cStrideS * pNHead, CUDA_R_32F,
                                                pDState, cStrideS, pNHead, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }
  }
}
#endif // !defined(GGML_USE_HIP)

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const struct ggml_tensor *vSrc0 = pDst->src[0]; // s
  const struct ggml_tensor *vSrc1 = pDst->src[1]; // x
  const struct ggml_tensor *vSrc2 = pDst->src[2]; // dt
  const struct ggml_tensor *vSrc3 = pDst->src[3]; // A
  const struct ggml_tensor *vSrc4 = pDst->src[4]; // B
  const struct ggml_tensor *vSrc5 = pDst->src[5]; // C
  const struct ggml_tensor *vSrc6 = pDst->src[6]; // ids

  const int64_t cNc = vSrc0->ne[0]; // d_state
  const int64_t cNr = vSrc0->ne[1]; // head_dim or 1
  const int64_t cNh = vSrc1->ne[1]; // n_head
  const int64_t cNg = vSrc4->ne[1]; // n_group
  const int64_t cNT = vSrc1->ne[2]; // number of tokens per sequence
  const int64_t cNS = vSrc1->ne[3]; // number of sequences in the batch

  const int64_t cSOff = ggml_nelements(vSrc1) * sizeof(float);

  GGML_ASSERT(ggml_nelements(vSrc1) + cNc * cNr * cNh * cNS == ggml_nelements(pDst));
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc1->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc2->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc3->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc4->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc5->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc6->nb[0] == sizeof(int32_t));

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  const float *vSrc2D = (const float *)vSrc2->data;
  const float *vSrc3D = (const float *)vSrc3->data;
  const float *vSrc4D = (const float *)vSrc4->data;
  const float *vSrc5D = (const float *)vSrc5->data;
  const int32_t *vSrc6D = (const int32_t *)vSrc6->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc6->type == GGML_TYPE_I32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  // Byte strides are narrowed to int for both scan and SSD paths.
  GGML_ASSERT(vSrc0->nb[2] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc0->nb[3] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc1->nb[2] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc1->nb[3] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc2->nb[1] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc2->nb[2] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc3->nb[1] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc4->nb[2] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc4->nb[3] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc5->nb[2] <= (size_t)INT_MAX);
  GGML_ASSERT(vSrc5->nb[3] <= (size_t)INT_MAX);

#if !defined(GGML_USE_HIP)
  // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
  // Requires NVIDIA Turing+ otherwise fallback to scan.
  const bool cIsMamba2 = (vSrc3->nb[1] == sizeof(float));
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const bool cUseSsd = cIsMamba2 && cNT > SSM_SSD_MIN_TOKENS && cNT <= SSM_SSD_MAX_TOKENS &&
                       GGML_CUDA_CC_IS_NVIDIA(cCc) && cCc >= GGML_CUDA_CC_TURING &&
                       cNr % 8 == 0; // cuBLAS requires 8-element (16-byte) alignment

  if (cUseSsd) {
    // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
    // so src0 must be fully contiguous across all inner dimensions.
    // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
    GGML_ASSERT(vSrc0->nb[1] == cNc * sizeof(float));
    GGML_ASSERT(vSrc0->nb[2] == cNc * cNr * sizeof(float));

    fSsmScanSsdF32Cuda(
        pCtx, vSrc0D, vSrc1D, vSrc2D, vSrc3D, vSrc4D, vSrc5D, vSrc6D, vDstD, (int64_t)(vSrc0->nb[3] / sizeof(float)),
        (int)(vSrc1->nb[2] / sizeof(float)), (int)(vSrc1->nb[3] / sizeof(float)), (int)(vSrc2->nb[1] / sizeof(float)),
        (int)(vSrc2->nb[2] / sizeof(float)), (int)(vSrc3->nb[1] / sizeof(float)), (int)(vSrc4->nb[2] / sizeof(float)),
        (int)(vSrc4->nb[3] / sizeof(float)), (int)(vSrc5->nb[2] / sizeof(float)), (int)(vSrc5->nb[3] / sizeof(float)),
        cSOff, cNc, cNr, cNh, cNg, cNT, cNS);
    return;
  }
#endif
  fSsmScanF32Cuda(vSrc0D, vSrc1D, vSrc2D, vSrc3D, vSrc4D, vSrc5D, vSrc6D, vDstD, vSrc0->nb[2], vSrc0->nb[3],
                  vSrc1->nb[2], vSrc1->nb[3], vSrc2->nb[1], vSrc2->nb[2], vSrc3->nb[1], vSrc4->nb[2], vSrc4->nb[3],
                  vSrc5->nb[2], vSrc5->nb[3], cSOff, cNc, cNr, cNh, cNg, cNT, cNS, vStream);
}
