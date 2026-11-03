#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
  int2 vI2;
  half2 aH2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void fLightningIndexerKernelWmma(const float *pQ, const char *pK, const float *pW, const half *pM,
                                                   float *pDst, int64_t pNStream, int64_t pNBatch, int64_t pNKv,
                                                   size_t pNb1, size_t pNb2, size_t pNb3, size_t pNbq1, size_t pNbq2,
                                                   size_t pNbq3, size_t pNbk1, size_t pNbk2, size_t pNbk3, size_t pNbw1,
                                                   size_t pNbw2, size_t pNbw3, size_t pNbm1, size_t pNbm2, size_t pNbm3,
                                                   int64_t pNem3) {

  constexpr int cThreadsPerBlock = WARPS_PER_BLOCK * WARP_SIZE;
  constexpr int cHeadsPerInnerLoop = 8;
  constexpr int cKEmbdPerInnerLoop = 16;
  constexpr int cNEmbdPadded = N_EMBD + 8;

  const int cIBatch = blockIdx.y;
  const int cIStream = blockIdx.z;
  const int cIWarp = threadIdx.y;
  const int cILane = threadIdx.x;
  const int cTid = cIWarp * WARP_SIZE + cILane;

  // each block processes K_VECS_PER_BLOCK K vectors
  const int cStartKv = blockIdx.x * K_VECS_PER_BLOCK;

  const char *vQBase = (const char *)pQ + cIBatch * pNbq2 + cIStream * pNbq3;
  const float *vWBase = (const float *)((const char *)pW + cIBatch * pNbw1 + cIStream * pNbw3);

  // phase 1 - load weights and first Q tile to shared memory

  __shared__ float vWShared[N_HEAD];
  __shared__ int2 vQSharedH[cHeadsPerInnerLoop][cNEmbdPadded / 4];

  if (cTid < N_HEAD) {
    vWShared[cTid] = vWBase[cTid];
  }

  // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
  constexpr int cNQTile = cHeadsPerInnerLoop * (N_EMBD / 4);
  // number of registers needed in each thread to store Q tile in thread block
  constexpr int cNQNext = (cNQTile + cThreadsPerBlock - 1) / cThreadsPerBlock;

#pragma unroll
  for (int vIQ = cTid; vIQ < cNQTile; vIQ += cThreadsPerBlock) {
    const int cIHead = vIQ / (N_EMBD / 4);
    const int cIEmbd = vIQ % (N_EMBD / 4);
    const float4 cQ = *(const float4 *)(vQBase + cIHead * pNbq1 + cIEmbd * sizeof(float4));
    half4 vQPacked;
    vQPacked.aH2[0] = __float22half2_rn(make_float2(cQ.x, cQ.y));
    vQPacked.aH2[1] = __float22half2_rn(make_float2(cQ.z, cQ.w));
    vQSharedH[cIHead][cIEmbd] = vQPacked.vI2;
  }

  // phase 2 - load (and dequantize if needed) K to shared mem

  __shared__ half2 vKSharedH[K_VECS_PER_BLOCK][cNEmbdPadded / 4][2];

  constexpr int cNK = K_VECS_PER_BLOCK * (N_EMBD / 4);

  if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
    for (int i_k = cTid; i_k < cNK; i_k += cThreadsPerBlock) {
      const int i_k_vec = i_k / (N_EMBD / 4);
      const int i_embd = i_k % (N_EMBD / 4);
      const int i_kv = cStartKv + i_k_vec;
      if (i_kv < pNKv) {
        const int2 *k_base = (const int2 *)((const char *)pK + i_kv * pNbk2 + cIStream * pNbk3);
        *(int2 *)&vKSharedH[i_k_vec][i_embd] = k_base[i_embd];
      } else {
        *(int2 *)&vKSharedH[i_k_vec][i_embd] = make_int2(0, 0);
      }
    }
  } else {
    constexpr dequantize_V_t cDequantizeK = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
    for (int vIK = cTid; vIK < cNK; vIK += cThreadsPerBlock) {
      const int cIKVec = vIK / (N_EMBD / 4);
      const int cIEmbd = vIK % (N_EMBD / 4);
      const int cIKv = cStartKv + cIKVec;
      if (cIKv < pNKv) {
        const void *vKBase = (const void *)((const char *)pK + cIKv * pNbk2 + cIStream * pNbk3);
        cDequantizeK(vKBase, &vKSharedH[cIKVec][cIEmbd][0], cIEmbd * 4);
      } else {
        *(int2 *)&vKSharedH[cIKVec][cIEmbd] = make_int2(0, 0);
      }
    }
  }

  __syncthreads();

  // phase 3 - calculate lightning indexer scores

  __shared__ float vQkShared[WARPS_PER_BLOCK][cHeadsPerInnerLoop][K_VECS_PER_BLOCK];

  // load K fragment
  wmma::fragment<wmma::matrix_b, cHeadsPerInnerLoop, K_VECS_PER_BLOCK, cKEmbdPerInnerLoop, half, wmma::col_major>
      vFragK;
  wmma::load_matrix_sync(vFragK, (half *)&vKSharedH[0][cIWarp * cKEmbdPerInnerLoop / 4], cNEmbdPadded);

  float vScoreK = 0.0f;

  for (int vIHead0 = 0; vIHead0 < N_HEAD; vIHead0 += cHeadsPerInnerLoop) {
    const int cIHeadNext = vIHead0 + cHeadsPerInnerLoop;

    // we don't use accumulator for anything, fill it with zeros
    wmma::fragment<wmma::accumulator, cHeadsPerInnerLoop, K_VECS_PER_BLOCK, cKEmbdPerInnerLoop, float> vFragAcc;
    wmma::fill_fragment(vFragAcc, 0.0f);

    // load Q fragment
    wmma::fragment<wmma::matrix_a, cHeadsPerInnerLoop, K_VECS_PER_BLOCK, cKEmbdPerInnerLoop, half, wmma::row_major>
        vFragQ;
    wmma::load_matrix_sync(vFragQ, (half *)&vQSharedH[0][cIWarp * cKEmbdPerInnerLoop / 4], cNEmbdPadded);

    // preload next Q tile to registers during matrix multiplication
    float4 aQNext[cNQNext];

    if (cIHeadNext < N_HEAD) {
#pragma unroll
      for (int vIQ = cTid, vIQNext = 0; vIQ < cNQTile; vIQ += cThreadsPerBlock) {
        const int cIHead = cIHeadNext + vIQ / (N_EMBD / 4);
        const int cIEmbd = vIQ % (N_EMBD / 4);
        aQNext[vIQNext++] = *(const float4 *)(vQBase + cIHead * pNbq1 + cIEmbd * sizeof(float4));
      }
    }

    // perform matrix multiplication
    wmma::mma_sync(vFragAcc, vFragQ, vFragK, vFragAcc);
    wmma::store_matrix_sync((float *)&vQkShared[cIWarp][0][0], vFragAcc, K_VECS_PER_BLOCK, wmma::mem_row_major);

    // make sure all threads finished using q_shared_h so we can store next tile
    __syncthreads();

    // write preloaded Q tile to shared memory
    if (cIHeadNext < N_HEAD) {
#pragma unroll
      for (int vIQ = cTid, vIQNext = 0; vIQ < cNQTile; vIQ += cThreadsPerBlock) {
        const int cIHead = vIQ / (N_EMBD / 4);
        const int cIEmbd = vIQ % (N_EMBD / 4);
        half4 vQPacked;
        vQPacked.aH2[0] = __float22half2_rn(make_float2(aQNext[vIQNext].x, aQNext[vIQNext].y));
        vQPacked.aH2[1] = __float22half2_rn(make_float2(aQNext[vIQNext].z, aQNext[vIQNext].w));
        vQSharedH[cIHead][cIEmbd] = vQPacked.vI2;
        ++vIQNext;
      }
    }

    // accumulate QK multiplication results from all block warps
    // (there are 256 threads in block and 256 matmul outputs)
    // TODO it will break if WARP_SIZE is not 32
    const int cH = cTid / K_VECS_PER_BLOCK;
    const int cK = cTid % K_VECS_PER_BLOCK;
    const float cWVal = vWShared[vIHead0 + cH];

    float vSum = 0.0f;
#pragma unroll
    for (int vW = 0; vW < WARPS_PER_BLOCK; ++vW) {
      vSum += vQkShared[vW][cH][cK];
    }

    // ReLU, weight
    vSum = vSum > 0.0f ? vSum : 0.0f;
    vSum *= cWVal;

    // wait until qk_shared[0] is no longer used
    __syncthreads();

    // reuse qk_shared[0] for storing partial results
    vQkShared[0][cH][cK] = vSum;

    // wait until all threads write their results
    __syncthreads();

    // accumulate result over heads
    if (cTid < K_VECS_PER_BLOCK) {
#pragma unroll
      for (int vIHead = 0; vIHead < cHeadsPerInnerLoop; ++vIHead) {
        vScoreK += vQkShared[0][vIHead][cTid];
      }
    }

    // make sure all threads finished using qk_shared
    __syncthreads();
  }

  // phase 4 - store output to VRAM

  if (cTid < K_VECS_PER_BLOCK) {
    const int cIKv = cStartKv + cTid;
    if (cIKv < pNKv) {
      const half *vMBase = (const half *)((const char *)pM + cIBatch * pNbm1 + (cIStream % pNem3) * pNbm3);
      float *vDstBase = (float *)((char *)pDst + cIBatch * pNb1 + cIStream * pNb3);
      vDstBase[cIKv] = vScoreK + __half2float(vMBase[cIKv]);
    }
  }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void
fLightningIndexerKernelWmma(const float *Q, const char *K, const float *W, const half *M, float *dst, int64_t n_stream,
                            int64_t n_batch, int64_t n_kv, size_t nb1, size_t nb2, size_t nb3, size_t nbq1, size_t nbq2,
                            size_t nbq3, size_t nbk1, size_t nbk2, size_t nbk3, size_t nbw1, size_t nbw2, size_t nbw3,
                            size_t nbm1, size_t nbm2, size_t nbm3, int64_t nem3) {
  GGML_UNUSED_VARS(Q, K, W, M, dst, n_stream, n_batch, n_kv, nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbw1,
                   nbw2, nbw3, nem3);
  NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void fLightningIndexerKernelVec(const float *pQ, const char *pK, const float *pW, const half *pM,
                                                  float *pDst, int64_t pNStream, int64_t pNBatch, int64_t pNKv,
                                                  size_t pNb1, size_t pNb2, size_t pNb3, size_t pNbq1, size_t pNbq2,
                                                  size_t pNbq3, size_t pNbk1, size_t pNbk2, size_t pNbk3, size_t pNbw1,
                                                  size_t pNbw2, size_t pNbw3, size_t pNbm1, size_t pNbm2, size_t pNbm3,
                                                  int64_t pNem3) {

  constexpr int cKVecsPerWarp = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
  constexpr int cThreadsPerBlock = WARPS_PER_BLOCK * WARP_SIZE;

  const int cIBatch = blockIdx.y;
  const int cIStream = blockIdx.z;
  const int cIWarp = threadIdx.y;
  const int cILane = threadIdx.x;
  const int cTid = cIWarp * WARP_SIZE + cILane;

  // each warp processes K_VECS_PER_WARP K vectors
  const int cStartKvBlock = blockIdx.x * K_VECS_PER_BLOCK;
  const int cStartKv = cStartKvBlock + cIWarp * cKVecsPerWarp;

  const char *vQBase = (const char *)pQ + cIBatch * pNbq2 + cIStream * pNbq3;
  const float *vWBase = (const float *)((const char *)pW + cIBatch * pNbw1 + cIStream * pNbw3);

  // phase 1 - load (and dequantize if needed) K to registers

  float4 aKRegF[cKVecsPerWarp];

  if constexpr (TYPE_K == GGML_TYPE_F32) {
    // direct copy of float4
#pragma unroll
    for (int vK = 0; vK < cKVecsPerWarp; ++vK) {
      int vIKv = cStartKv + vK;
      if (vIKv < pNKv) {
        const float4 *vKBase = (const float4 *)((const char *)pK + vIKv * pNbk2 + cIStream * pNbk3);
        aKRegF[vK] = vKBase[cILane];
      } else {
        aKRegF[vK] = make_float4(0, 0, 0, 0);
      }
    }
  } else {
    // dequantize remaining types to float
    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
    for (int k = 0; k < cKVecsPerWarp; ++k) {
      int i_kv = cStartKv + k;
      if (i_kv < pNKv) {
        const void *k_base = (const void *)((const char *)pK + i_kv * pNbk2 + cIStream * pNbk3);
        dequantize_k(k_base, &aKRegF[k], cILane * 4);
      } else {
        aKRegF[k] = make_float4(0, 0, 0, 0);
      }
    }
  }

  float aScoreK[cKVecsPerWarp] = {0.0f};

  // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
  constexpr int cNHeadInner = N_HEAD / 4;

  for (int vIHead0 = 0; vIHead0 < N_HEAD; vIHead0 += cNHeadInner) {
    // phase 2 - load weights and Q to shared memory

    __shared__ float vWShared[cNHeadInner];
    __shared__ float4 vQSharedF[cNHeadInner][N_EMBD / 4];

    if (cTid < cNHeadInner) {
      vWShared[cTid] = vWBase[vIHead0 + cTid];
    }

    constexpr int cNQ = cNHeadInner * (N_EMBD / 4);
#pragma unroll
    for (int vIQ = cTid; vIQ < cNQ; vIQ += cThreadsPerBlock) {
      const int cIHeadInner = vIQ / (N_EMBD / 4);
      const int cIHead = vIHead0 + cIHeadInner;
      const int cIEmbd = vIQ % (N_EMBD / 4);
      vQSharedF[cIHeadInner][cIEmbd] = *(const float4 *)(vQBase + cIHead * pNbq1 + cIEmbd * sizeof(float4));
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    for (int vIHeadInner = 0; vIHeadInner < cNHeadInner; ++vIHeadInner) {
      const float cWVal = vWShared[vIHeadInner];
      float aQk[cKVecsPerWarp] = {0.0f};

      // dot product of floats
      const float4 cQVec = vQSharedF[vIHeadInner][cILane];

#pragma unroll
      for (int vK = 0; vK < cKVecsPerWarp; ++vK) {
        ggml_cuda_mad(aQk[vK], cQVec.x, aKRegF[vK].x);
        ggml_cuda_mad(aQk[vK], cQVec.y, aKRegF[vK].y);
        ggml_cuda_mad(aQk[vK], cQVec.z, aKRegF[vK].z);
        ggml_cuda_mad(aQk[vK], cQVec.w, aKRegF[vK].w);
      }

#pragma unroll
      for (int vK = 0; vK < cKVecsPerWarp; ++vK) {
        float vSum = warp_reduce_sum(aQk[vK]);

        // ReLU, weight
        if (cILane == 0) {
          vSum = (vSum > 0.0f) ? vSum : 0.0f;
          aScoreK[vK] += vSum * cWVal;
        }
      }
    }

    __syncthreads();
  }

  // phase 4 - store outputs to shared memory

  __shared__ float vDstShared[K_VECS_PER_BLOCK];

  if (cILane == 0) {
#pragma unroll
    for (int vK = 0; vK < cKVecsPerWarp; ++vK) {
      vDstShared[cIWarp * cKVecsPerWarp + vK] = aScoreK[vK];
    }
  }

  __syncthreads();

  // phase 5 - write from shared memory to VRAM in coalesced manner

  if (cTid < K_VECS_PER_BLOCK) {
    int vIKv = cStartKvBlock + cTid;
    if (vIKv < pNKv) {
      const half *vMBase = (const half *)((const char *)pM + cIBatch * pNbm1 + (cIStream % pNem3) * pNbm3);
      float *vDstBase = (float *)((char *)pDst + cIBatch * pNb1 + cIStream * pNb3);
      vDstBase[vIKv] = vDstShared[cTid] + __half2float(vMBase[vIKv]);
    }
  }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)                                    \
  if (K->type == (type_K)) {                                                                                           \
    lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K>                                \
        <<<grid, block, 0, ctx.stream()>>>(q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb2, nb3, nbq1,    \
                                           nbq2, nbq3, nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3);    \
  } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context &ctx, ggml_tensor *pDst) {
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vW = pDst->src[2]; // weights
  const ggml_tensor *vM = pDst->src[3]; // mask

  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(vQ->type == GGML_TYPE_F32);
  GGML_ASSERT(vW->type == GGML_TYPE_F32);
  GGML_ASSERT(vM->type == GGML_TYPE_F16);

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, new, vW, ne)
  GGML_TENSOR_LOCALS(size_t, nbw, vW, nb)
  GGML_TENSOR_LOCALS(int64_t, nem, vM, ne)
  GGML_TENSOR_LOCALS(size_t, nbm, vM, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  // input tensor rows must be contiguous
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbw0 == ggml_type_size(vW->type));
  GGML_ASSERT(nbm0 == ggml_type_size(vM->type));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  const int cNEmbd = vQ->ne[0];
  const int cNHead = vQ->ne[1];
  const int n_batch = vQ->ne[2];
  const int n_stream = vQ->ne[3];
  const int n_kv = vK->ne[2];

  const float *q_d = (const float *)vQ->data;
  const char *k_d = (const char *)vK->data;
  const float *w_d = (const float *)vW->data;
  const half *m_d = (const half *)vM->data;
  float *dst_d = (float *)pDst->data;

  const int cDevice = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cDevice].cc;

  if (cNEmbd == 128 && cNHead == 64) {
#if !defined(GGML_USE_HIP)
    if (GGML_CUDA_CC_IS_NVIDIA(cCc) && turing_mma_available(cCc) && vK->type != GGML_TYPE_F32 &&
        vK->type != GGML_TYPE_BF16) {
      // use wmma kernel
      constexpr int K_VECS_PER_BLOCK = 32;
      constexpr int WARPS_PER_BLOCK = 8;

      dim3 block(32, WARPS_PER_BLOCK);
      int vNumKvBlocks = (n_kv + (K_VECS_PER_BLOCK)-1) / (K_VECS_PER_BLOCK);
      dim3 grid(vNumKvBlocks, n_batch, n_stream);

      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_F16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_Q4_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_Q4_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_Q5_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_Q5_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 64, vK, GGML_TYPE_Q8_0)
      GGML_ABORT("fatal error");
    } else {
#else  // !defined(GGML_USE_HIP)
    {
#endif // !defined(GGML_USE_HIP)
       // use vector kernel
      constexpr int cKVecsPerWarp = 8;
      constexpr int WARPS_PER_BLOCK = 8;
      constexpr int K_VECS_PER_BLOCK = cKVecsPerWarp * WARPS_PER_BLOCK;

      dim3 block(32, WARPS_PER_BLOCK);
      int vNumKvBlocks = (n_kv + (K_VECS_PER_BLOCK)-1) / (K_VECS_PER_BLOCK);
      dim3 grid(vNumKvBlocks, n_batch, n_stream);

      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_F16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_Q4_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_Q4_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_Q5_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_Q5_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_Q8_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_BF16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 64, vK, GGML_TYPE_F32)
      GGML_ABORT("fatal error");
    }
  } else if (cNEmbd == 128 && cNHead == 32) {
#if !defined(GGML_USE_HIP)
    if (GGML_CUDA_CC_IS_NVIDIA(cCc) && turing_mma_available(cCc) && vK->type != GGML_TYPE_F32 &&
        vK->type != GGML_TYPE_BF16) {
      // use wmma kernel
      constexpr int K_VECS_PER_BLOCK = 32;
      constexpr int WARPS_PER_BLOCK = 8;

      dim3 block(32, WARPS_PER_BLOCK);
      int vNumKvBlocks = (n_kv + (K_VECS_PER_BLOCK)-1) / (K_VECS_PER_BLOCK);
      dim3 grid(vNumKvBlocks, n_batch, n_stream);

      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_F16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_Q4_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_Q4_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_Q5_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_Q5_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelWmma, 128, 32, vK, GGML_TYPE_Q8_0)
      GGML_ABORT("fatal error");
    } else {
#else  // !defined(GGML_USE_HIP)
    {
#endif // !defined(GGML_USE_HIP)
       // use vector kernel
      constexpr int cKVecsPerWarp = 8;
      constexpr int WARPS_PER_BLOCK = 8;
      constexpr int K_VECS_PER_BLOCK = cKVecsPerWarp * WARPS_PER_BLOCK;

      dim3 block(32, WARPS_PER_BLOCK);
      int vNumKvBlocks = (n_kv + (K_VECS_PER_BLOCK)-1) / (K_VECS_PER_BLOCK);
      dim3 grid(vNumKvBlocks, n_batch, n_stream);

      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_F16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_Q4_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_Q4_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_Q5_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_Q5_1)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_Q8_0)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_BF16)
      LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 32, vK, GGML_TYPE_F32)
      GGML_ABORT("fatal error");
    }
  } else if (cNEmbd == 128 && cNHead == 4) {
    // too few heads for a wmma tile, use vector kernel
    constexpr int cKVecsPerWarp = 8;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int K_VECS_PER_BLOCK = cKVecsPerWarp * WARPS_PER_BLOCK;

    dim3 block(32, WARPS_PER_BLOCK);
    int vNumKvBlocks = (n_kv + (K_VECS_PER_BLOCK)-1) / (K_VECS_PER_BLOCK);
    dim3 grid(vNumKvBlocks, n_batch, n_stream);

    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_F16)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_Q4_0)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_Q4_1)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_Q5_0)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_Q5_1)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_Q8_0)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_BF16)
    LIGHTNING_INDEXER_CASE(fLightningIndexerKernelVec, 128, 4, vK, GGML_TYPE_F32)
    GGML_ABORT("fatal error");
  } else {
    GGML_ABORT("fatal error");
  }
}

bool ggml_cuda_lightning_indexer_supported(int pDevice, const ggml_tensor *pDst) {
  GGML_UNUSED(pDevice);

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vW = pDst->src[2]; // weights
  const ggml_tensor *vM = pDst->src[3]; // mask

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, new, vW, ne)
  GGML_TENSOR_LOCALS(size_t, nbw, vW, nb)
  GGML_TENSOR_LOCALS(int64_t, nem, vM, ne)
  GGML_TENSOR_LOCALS(size_t, nbm, vM, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  if (neq0 != 128) {
    return false;
  }

  if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
    return false;
  }

  // alignment checks
  for (const ggml_tensor *vT : {vQ, vK}) {
    if (ggml_is_quantized(vT->type)) {
      continue;
    }
    for (size_t vI = 1; vI < GGML_MAX_DIMS; ++vI) {
      if (vT->nb[vI] % 16 != 0) {
        return false;
      }
    }
  }

  switch (vK->type) {
  case GGML_TYPE_F32:
  case GGML_TYPE_BF16:
  case GGML_TYPE_F16:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q4_0:
    return true;
  default:
    return false;
  }
}
