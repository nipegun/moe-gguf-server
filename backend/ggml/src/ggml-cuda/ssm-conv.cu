#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float *pSrc0Ptr, const float *pSrc1Ptr, const float *pBiasPtr,
                                    const int pSrc0Nb0, const int pSrc0Nb1, const int pSrc0Nb2, const int pSrc1Nb1,
                                    float *pDstPtr, const int pDstNb0, const int pDstNb1, const int pDstNb2,
                                    const int64_t pNT) {
  ggml_cuda_pdl_lc();
  const float *GGML_CUDA_RESTRICT vSrc0 = pSrc0Ptr;
  const float *GGML_CUDA_RESTRICT vSrc1 = pSrc1Ptr;
  const float *GGML_CUDA_RESTRICT vBias = pBiasPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  GGML_UNUSED(pSrc0Nb0);
  const int cTid = threadIdx.x;
  const int cBidx = blockIdx.x;
  const int cBidy = blockIdx.y;

  const float *vXBlock = (const float *)((const char *)vSrc0 + cBidx * pSrc0Nb2 + cBidy * split_d_inner * pSrc0Nb1);
  const float *vWBlock = (const float *)((const char *)vSrc1 + cBidy * split_d_inner * pSrc1Nb1);
  float *vYBlock = (float *)((char *)vDst + cBidx * pDstNb2 + cBidy * split_d_inner * pDstNb0);

  const int cStrideX = pSrc0Nb1 / sizeof(float);
  const int cStrideW = pSrc1Nb1 / sizeof(float);
  const int cStrideY = pDstNb1 / sizeof(float);

  float aX[d_conv] = {0.0f};
  float aW[d_conv] = {0.0f};

  ggml_cuda_pdl_sync();
#pragma unroll
  for (size_t vJ = 0; vJ < d_conv; vJ++) {
    aW[vJ] = vWBlock[cTid * cStrideW + vJ];
  }

  float vB = vBias != nullptr ? vBias[cBidy * split_d_inner + cTid] : 0.0f;

  for (int64_t vI = 0; vI < pNT; vI++) {
    float vSumf = 0.0f;

    if (vI == 0) {
      for (size_t vJ = 0; vJ < d_conv; vJ++) {
        aX[vJ] = vXBlock[cTid * cStrideX + vJ];
      }
    } else {
      aX[(vI - 1) % d_conv] = vXBlock[cTid * cStrideX + vI + d_conv - 1];
    }

#pragma unroll
    for (size_t vJ = 0; vJ < d_conv; vJ++) {
      vSumf += aX[(vI + vJ) % d_conv] * aW[vJ];
    }
    vSumf += vB;
    vYBlock[vI * cStrideY + cTid] = apply_silu ? ggml_cuda_op_silu_single(vSumf) : vSumf;
  }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float *__restrict__ pSrc0, const float *__restrict__ pSrc1,
                                               const float *__restrict__ pBias, const int pSrc0Nb0, const int pSrc0Nb1,
                                               const int pSrc0Nb2, const int pSrc1Nb1, float *__restrict__ pDst,
                                               const int pDstNb0, const int pDstNb1, const int pDstNb2,
                                               const int64_t pNT) {
  const int cTid = threadIdx.x;
  const int cBidx = blockIdx.x;
  const int cBidy = blockIdx.y;
  const int cBidz = blockIdx.z;

  const float *vXBlock = (const float *)((const char *)pSrc0 + cBidx * pSrc0Nb2 + cBidy * split_d_inner * pSrc0Nb1 +
                                         cBidz * split_n_t * pSrc0Nb0);
  const float *vWBlock = (const float *)((const char *)pSrc1 + cBidy * split_d_inner * pSrc1Nb1);
  float *vYBlock =
      (float *)((char *)pDst + cBidx * pDstNb2 + cBidz * split_n_t * pDstNb1 + cBidy * split_d_inner * pDstNb0);

  const int cStrideX = pSrc0Nb1 / sizeof(float);
  const int cStrideW = pSrc1Nb1 / sizeof(float);
  const int cStrideY = pDstNb1 / sizeof(float);

  const int64_t cLocalNT = min(split_n_t, pNT - cBidz * split_n_t);
  const int cNCols = d_conv - 1 + split_n_t;

  extern __shared__ float aSmem[];

  constexpr int cLoadCols = d_conv - 1 + split_n_t;
  constexpr int cTotalElems = split_d_inner * cLoadCols;
  int vRow = cTid / cLoadCols;
  int vCol = cTid % cLoadCols;
#pragma unroll
  for (int vIdx = 0; vIdx < cTotalElems; vIdx += split_d_inner) {
    if (vRow < (int)split_d_inner) {
      aSmem[vRow * cNCols + vCol] = vXBlock[vRow * cStrideX + vCol];
    }

    vCol += split_d_inner;
    vRow += vCol / cLoadCols;
    vCol = vCol % cLoadCols;
    if (vIdx >= cTotalElems - cTid - split_d_inner) {
      break;
    }
  }
  __syncthreads();

  // Load weights into registers (done once, small)
  float aW[d_conv] = {0.0f};
#pragma unroll
  for (size_t vJ = 0; vJ < d_conv; vJ++) {
    aW[vJ] = vWBlock[cTid * cStrideW + vJ];
  }

  float vB = pBias != nullptr ? pBias[cBidy * split_d_inner + cTid] : 0.0f;

  // Compute from shared memory
  for (int64_t vI = 0; vI < cLocalNT; vI++) {
    float vSumf = 0.0f;
#pragma unroll
    for (size_t vJ = 0; vJ < d_conv; vJ++) {
      vSumf += aSmem[cTid * cNCols + vI + vJ] * aW[vJ];
    }
    vSumf += vB;
    vYBlock[vI * cStrideY + cTid] = apply_silu ? ggml_cuda_op_silu_single(vSumf) : vSumf;
  }
}

template <bool apply_silu>
static void fSsmConvF32Cuda(const float *pSrc0, const float *pSrc1, const float *pBias, const int pSrc0Nb0,
                            const int pSrc0Nb1, const int pSrc0Nb2, const int pSrc1Nb1, float *pDst, const int pDstNb0,
                            const int pDstNb1, const int pDstNb2, const int64_t nc, const int64_t pNr,
                            const int64_t pNT, const int64_t pNS, cudaStream_t pStream) {
  const int cThreads = 128;
  GGML_ASSERT(pNr % cThreads == 0);

  auto vLaunchKernel = [&](auto NC) {
    constexpr int cKNc = decltype(NC)::value;
    if (pNT <= 32) {
      const dim3 cBlocks(pNS, (pNr + cThreads - 1) / cThreads, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocks, cThreads, 0, pStream);
      ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, cThreads, cKNc>, cLaunchParams, pSrc0, pSrc1, pBias, pSrc0Nb0,
                              pSrc0Nb1, pSrc0Nb2, pSrc1Nb1, pDst, pDstNb0, pDstNb1, pDstNb2, pNT);
    } else {
      const int64_t cSplitNT = 32;
      dim3 vBlocks(pNS, (pNr + cThreads - 1) / cThreads, (pNT + cSplitNT - 1) / cSplitNT);
      const size_t cSmemSize = cThreads * (cKNc - 1 + cSplitNT) * sizeof(float);
      ssm_conv_long_token_f32<apply_silu, cThreads, cKNc, cSplitNT><<<vBlocks, cThreads, cSmemSize, pStream>>>(
          pSrc0, pSrc1, pBias, pSrc0Nb0, pSrc0Nb1, pSrc0Nb2, pSrc1Nb1, pDst, pDstNb0, pDstNb1, pDstNb2, pNT);
    }
  };

  switch (nc) {
  case 3:
    vLaunchKernel(std::integral_constant<int, 3>{});
    break;
  case 4:
    vLaunchKernel(std::integral_constant<int, 4>{});
    break;
  case 5:
    vLaunchKernel(std::integral_constant<int, 5>{});
    break;
  case 9:
    vLaunchKernel(std::integral_constant<int, 9>{});
    break;
  case 15:
    vLaunchKernel(std::integral_constant<int, 15>{});
    break;
  default:
    GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
  }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pBiasAddNode,
                           ggml_tensor *pSiluDst) {
  const struct ggml_tensor *vSrc0 = pDst->src[0]; // conv_x
  const struct ggml_tensor *vSrc1 = pDst->src[1]; // conv1d.weight
  const bool cFuseBias = pBiasAddNode != nullptr;
  const bool cFuseSilu = pSiluDst != nullptr;

  // bias always comes with silu.
  GGML_ASSERT(!cFuseBias || cFuseSilu);

  // The bias (when fused) is the non-conv operand of the ADD node.
  const struct ggml_tensor *vBias =
      cFuseBias ? (pBiasAddNode->src[0] == pDst ? pBiasAddNode->src[1] : pBiasAddNode->src[0]) : nullptr;

  // When fusing, write to silu_dst (the node downstream references).
  const struct ggml_tensor *vOut = cFuseSilu ? pSiluDst : pDst;

  const int64_t cNc = vSrc1->ne[0]; // d_conv
  const int64_t cNr = vSrc0->ne[1]; // d_inner
  const int64_t cNT = vOut->ne[1];  // tokens per sequence
  const int64_t cNS = vOut->ne[2];  // number of sequences in the batch

  GGML_ASSERT(vOut->ne[0] == cNr);
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc1->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc0->nb[1] == vSrc0->ne[0] * sizeof(float));

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  const float *vBiasD = cFuseBias ? (const float *)vBias->data : nullptr;
  float *vDstD = (float *)vOut->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vOut->type == GGML_TYPE_F32);
  if (cFuseBias) {
    GGML_ASSERT(vBias->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(vBias));
    GGML_ASSERT(ggml_nelements(vBias) == cNr);
  }

  if (cFuseSilu) {
    fSsmConvF32Cuda<true>(vSrc0D, vSrc1D, vBiasD, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc1->nb[1], vDstD,
                          vOut->nb[0], vOut->nb[1], vOut->nb[2], cNc, cNr, cNT, cNS, vStream);
  } else {
    fSsmConvF32Cuda<false>(vSrc0D, vSrc1D, vBiasD, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc1->nb[1], vDstD,
                           vOut->nb[0], vOut->nb[1], vOut->nb[2], cNc, cNr, cNT, cNS, vStream);
  }
}
