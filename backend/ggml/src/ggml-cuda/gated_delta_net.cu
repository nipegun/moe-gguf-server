#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void
__launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
    fGatedDeltaNetCuda(const float *pQ, const float *k, const float *pV, const float *pG, const float *pBeta,
                       const float *pCurrState, float *pDst, float *pState, int64_t pH, int64_t pNTokens,
                       int64_t pNSeqs, int64_t pSq1, int64_t pSq2, int64_t pSq3, int64_t pSv1, int64_t pSv2,
                       int64_t pSv3, int64_t pSb1, int64_t pSb2, int64_t pSb3, const uint3 pNeqk1Magic,
                       const uint3 pRq3Magic, float pScale, int64_t pStateSlotStride, int K) {
  const uint32_t cHIdx = blockIdx.x;
  const uint32_t cSequence = blockIdx.y;
  // each warp owns one column, using warp-level primitives to reduce across rows
  const int cLane = threadIdx.x;
  const int cCol = blockIdx.z * blockDim.y + threadIdx.y;

  const uint32_t cIq1 = fastmodulo(cHIdx, pNeqk1Magic);
  const uint32_t cIq3 = fastdiv(cSequence, pRq3Magic);

  float *vAttnData = pDst;

  // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
  // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
  const int64_t cStateInOffset = cSequence * pH * S_v * S_v + cHIdx * S_v * S_v;
  const int64_t cStateOutOffset = (cSequence * pH + cHIdx) * S_v * S_v;
  pState += cStateOutOffset;
  pCurrState += cStateInOffset + cCol * S_v;
  vAttnData += (cSequence * pNTokens * pH + cHIdx) * S_v;

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
  static_assert(S_v % cWarpSize == 0, "S_v must be a multiple of warp_size");
  constexpr int cRowsPerLane = (S_v + cWarpSize - 1) / cWarpSize;
  float aSShard[cRowsPerLane];
  // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

  ggml_cuda_pdl_sync();
#pragma unroll
  for (int vR = 0; vR < cRowsPerLane; vR++) {
    const int cI = vR * cWarpSize + cLane;
    aSShard[vR] = pCurrState[cI];
  }

  for (int vT = 0; vT < pNTokens; vT++) {
    const float *vQT = pQ + cIq3 * pSq3 + vT * pSq2 + cIq1 * pSq1;
    const float *vKT = k + cIq3 * pSq3 + vT * pSq2 + cIq1 * pSq1;
    const float *vVT = pV + cSequence * pSv3 + vT * pSv2 + cHIdx * pSv1;

    const int64_t cGbOffset = cSequence * pSb3 + vT * pSb2 + cHIdx * pSb1;
    const float *vBetaT = pBeta + cGbOffset;
    const float *vGT = pG + cGbOffset * (KDA ? S_v : 1);

    const float cBetaVal = *vBetaT;

    // Cache k and q in registers
    float aKReg[cRowsPerLane];
    float aQReg[cRowsPerLane];
#pragma unroll
    for (int vR = 0; vR < cRowsPerLane; vR++) {
      const int cI = vR * cWarpSize + cLane;
      aKReg[vR] = vKT[cI];
      aQReg[vR] = vQT[cI];
    }

    if constexpr (!KDA) {
      const float cGVal = expf(*vGT);

      // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
      float vKvShard = 0.0f;
#pragma unroll
      for (int vR = 0; vR < cRowsPerLane; vR++) {
        vKvShard += aSShard[vR] * aKReg[vR];
      }
      float vKvCol = warp_reduce_sum<cWarpSize>(vKvShard);

      // delta[col] = (v[col] - g * kv[col]) * beta
      float vDeltaCol = (vVT[cCol] - cGVal * vKvCol) * cBetaVal;

      // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
      // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
      float vAttnPartial = 0.0f;
#pragma unroll
      for (int vR = 0; vR < cRowsPerLane; vR++) {
        aSShard[vR] = cGVal * aSShard[vR] + aKReg[vR] * vDeltaCol;
        vAttnPartial += aSShard[vR] * aQReg[vR];
      }

      float vAttnCol = warp_reduce_sum<cWarpSize>(vAttnPartial);

      if (cLane == 0) {
        vAttnData[cCol] = vAttnCol * pScale;
      }
    } else {
      // kv[col] = sum_i g[i] * S[i][col] * k[i]
      float kv_shard = 0.0f;
#pragma unroll
      for (int r = 0; r < cRowsPerLane; r++) {
        const int i = r * cWarpSize + cLane;
        kv_shard += expf(vGT[i]) * aSShard[r] * aKReg[r];
      }

      float kv_col = warp_reduce_sum<cWarpSize>(kv_shard);

      // delta[col] = (v[col] - kv[col]) * beta
      float delta_col = (vVT[cCol] - kv_col) * cBetaVal;

      // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
      // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
      float attn_partial = 0.0f;
#pragma unroll
      for (int r = 0; r < cRowsPerLane; r++) {
        const int i = r * cWarpSize + cLane;
        aSShard[r] = expf(vGT[i]) * aSShard[r] + aKReg[r] * delta_col;
        attn_partial += aSShard[r] * aQReg[r];
      }

      float attn_col = warp_reduce_sum<cWarpSize>(attn_partial);

      if (cLane == 0) {
        vAttnData[cCol] = attn_col * pScale;
      }
    }

    vAttnData += S_v * pH;

    if constexpr (keep_rs_t) {
      // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
      // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
      const int target_slot = (int)pNTokens - 1 - vT;
      if (target_slot >= 0 && target_slot < K) {
        float *curr_state = pState + target_slot * pStateSlotStride;
#pragma unroll
        for (int r = 0; r < cRowsPerLane; r++) {
          const int i = r * cWarpSize + cLane;
          curr_state[cCol * S_v + i] = aSShard[r];
        }
      }
    }
  }

  if constexpr (!keep_rs_t) {
#pragma unroll
    for (int vR = 0; vR < cRowsPerLane; vR++) {
      const int cI = vR * cWarpSize + cLane;
      pState[cCol * S_v + cI] = aSShard[vR];
    }
  }
}

template <bool KDA, bool keep_rs_t>
static void fLaunchGatedDeltaNet(const float *pQD, const float *pKD, const float *pVD, const float *pGD,
                                 const float *pBD, const float *pSD, float *pDstD, float *pStateD, int64_t pSV,
                                 int64_t pH, int64_t pNTokens, int64_t pNSeqs, int64_t pSq1, int64_t pSq2, int64_t pSq3,
                                 int64_t pSv1, int64_t pSv2, int64_t pSv3, int64_t pSb1, int64_t pSb2, int64_t pSb3,
                                 int64_t pNeqk1, int64_t pRq3, float pScale, int64_t pStateSlotStride, int pK,
                                 cudaStream_t pStream) {
  // TODO: Add chunked kernel for even faster pre-fill
  const int cWarpSize = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
  const int cNumWarps = 4;
  dim3 vGridDims(pH, pNSeqs, (pSV + cNumWarps - 1) / cNumWarps);
  dim3 vBlockDims(cWarpSize <= pSV ? cWarpSize : pSV, cNumWarps, 1);

  const uint3 cNeqk1Magic = init_fastdiv_values(pNeqk1);
  const uint3 cRq3Magic = init_fastdiv_values(pRq3);

  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(vGridDims, vBlockDims, 0, pStream);
  switch (pSV) {
  case 16:
    ggml_cuda_kernel_launch(fGatedDeltaNetCuda<16, KDA, keep_rs_t>, cLaunchParams, pQD, pKD, pVD, pGD, pBD, pSD, pDstD,
                            pStateD, pH, pNTokens, pNSeqs, pSq1, pSq2, pSq3, pSv1, pSv2, pSv3, pSb1, pSb2, pSb3,
                            cNeqk1Magic, cRq3Magic, pScale, pStateSlotStride, pK);
    break;
  case 32:
    ggml_cuda_kernel_launch(fGatedDeltaNetCuda<32, KDA, keep_rs_t>, cLaunchParams, pQD, pKD, pVD, pGD, pBD, pSD, pDstD,
                            pStateD, pH, pNTokens, pNSeqs, pSq1, pSq2, pSq3, pSv1, pSv2, pSv3, pSb1, pSb2, pSb3,
                            cNeqk1Magic, cRq3Magic, pScale, pStateSlotStride, pK);
    break;
  case 64: {
    ggml_cuda_kernel_launch(fGatedDeltaNetCuda<64, KDA, keep_rs_t>, cLaunchParams, pQD, pKD, pVD, pGD, pBD, pSD, pDstD,
                            pStateD, pH, pNTokens, pNSeqs, pSq1, pSq2, pSq3, pSv1, pSv2, pSv3, pSb1, pSb2, pSb3,
                            cNeqk1Magic, cRq3Magic, pScale, pStateSlotStride, pK);
    break;
  }
  case 128: {
    ggml_cuda_kernel_launch(fGatedDeltaNetCuda<128, KDA, keep_rs_t>, cLaunchParams, pQD, pKD, pVD, pGD, pBD, pSD, pDstD,
                            pStateD, pH, pNTokens, pNSeqs, pSq1, pSq2, pSq3, pSv1, pSv2, pSv3, pSb1, pSb2, pSb3,
                            cNeqk1Magic, cRq3Magic, pScale, pStateSlotStride, pK);
    break;
  }
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

static void fGgmlCudaOpGatedDeltaNetImpl(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst,
                                         const ggml_cuda_gated_delta_net_fused_cache *pCache) {
  ggml_tensor *vSrcQ = pDst->src[0];
  ggml_tensor *vSrcK = pDst->src[1];
  ggml_tensor *vSrcV = pDst->src[2];
  ggml_tensor *vSrcG = pDst->src[3];
  ggml_tensor *vSrcBeta = pDst->src[4];
  ggml_tensor *vSrcState = pDst->src[5];

  GGML_TENSOR_LOCALS(int64_t, neq, vSrcQ, ne);
  GGML_TENSOR_LOCALS(size_t, nbq, vSrcQ, nb);
  GGML_TENSOR_LOCALS(int64_t, nek, vSrcK, ne);
  GGML_TENSOR_LOCALS(size_t, nbk, vSrcK, nb);
  GGML_TENSOR_LOCALS(int64_t, nev, vSrcV, ne);
  GGML_TENSOR_LOCALS(size_t, nbv, vSrcV, nb);
  GGML_TENSOR_LOCALS(size_t, nbb, vSrcBeta, nb);

  const int64_t cSV = nev0;
  const int64_t cH = nev1;
  const int64_t cNTokens = nev2;
  const int64_t cNSeqs = nev3;

  const bool cKda = (vSrcG->ne[0] == cSV);

  GGML_ASSERT(neq1 == nek1);
  const int64_t cNeqk1 = neq1;

  const int64_t cRq3 = nev3 / neq3;

  const float *vQD = (const float *)vSrcQ->data;
  const float *vKD = (const float *)vSrcK->data;
  const float *vVD = (const float *)vSrcV->data;
  const float *vGD = (const float *)vSrcG->data;
  const float *vBD = (const float *)vSrcBeta->data;

  const float *vSD = (const float *)vSrcState->data;
  float *vDstD = (float *)pDst->data;

  GGML_ASSERT(ggml_is_contiguous_rows(vSrcQ));
  GGML_ASSERT(ggml_is_contiguous_rows(vSrcK));
  GGML_ASSERT(ggml_is_contiguous_rows(vSrcV));
  GGML_ASSERT(ggml_are_same_stride(vSrcQ, vSrcK));
  GGML_ASSERT(vSrcG->ne[0] == 1 || cKda);
  GGML_ASSERT(ggml_is_contiguous(vSrcG));
  GGML_ASSERT(ggml_is_contiguous(vSrcBeta));
  GGML_ASSERT(ggml_is_contiguous(vSrcState));

  // strides in floats (beta strides used for both g and beta offset computation)
  const int64_t cSq1 = nbq1 / sizeof(float);
  const int64_t cSq2 = nbq2 / sizeof(float);
  const int64_t cSq3 = nbq3 / sizeof(float);
  const int64_t cSv1 = nbv1 / sizeof(float);
  const int64_t cSv2 = nbv2 / sizeof(float);
  const int64_t cSv3 = nbv3 / sizeof(float);
  const int64_t cSb1 = nbb1 / sizeof(float);
  const int64_t cSb2 = nbb2 / sizeof(float);
  const int64_t cSb3 = nbb3 / sizeof(float);

  const float cScale = 1.0f / sqrtf((float)cSV);

  cudaStream_t vStream = pCtx.stream();

  // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
  const int cK = ggml_get_op_params_i32(pDst, 0);
  const bool cKeepRs = cK > 1;

  // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
  float *vStateD = vDstD + cSV * cH * cNTokens * cNSeqs;
  int64_t vStateSlotStride = cSV * cSV * cH * cNSeqs;
  if (pCache != nullptr) {
    vStateD = pCache->data;
    vStateSlotStride = pCache->slot_stride;
  }

  if (cKda) {
    if (cKeepRs) {
      fLaunchGatedDeltaNet<true, true>(vQD, vKD, vVD, vGD, vBD, vSD, vDstD, vStateD, cSV, cH, cNTokens, cNSeqs, cSq1,
                                       cSq2, cSq3, cSv1, cSv2, cSv3, cSb1, cSb2, cSb3, cNeqk1, cRq3, cScale,
                                       vStateSlotStride, cK, vStream);
    } else {
      fLaunchGatedDeltaNet<true, false>(vQD, vKD, vVD, vGD, vBD, vSD, vDstD, vStateD, cSV, cH, cNTokens, cNSeqs, cSq1,
                                        cSq2, cSq3, cSv1, cSv2, cSv3, cSb1, cSb2, cSb3, cNeqk1, cRq3, cScale,
                                        vStateSlotStride, cK, vStream);
    }
  } else {
    if (cKeepRs) {
      fLaunchGatedDeltaNet<false, true>(vQD, vKD, vVD, vGD, vBD, vSD, vDstD, vStateD, cSV, cH, cNTokens, cNSeqs, cSq1,
                                        cSq2, cSq3, cSv1, cSv2, cSv3, cSb1, cSb2, cSb3, cNeqk1, cRq3, cScale,
                                        vStateSlotStride, cK, vStream);
    } else {
      fLaunchGatedDeltaNet<false, false>(vQD, vKD, vVD, vGD, vBD, vSD, vDstD, vStateD, cSV, cH, cNTokens, cNSeqs, cSq1,
                                         cSq2, cSq3, cSv1, cSv2, cSv3, cSb1, cSb2, cSb3, cNeqk1, cRq3, cScale,
                                         vStateSlotStride, cK, vStream);
    }
  }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpGatedDeltaNetImpl(pCtx, pDst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst,
                                              ggml_cuda_gated_delta_net_fused_cache pCache) {
  fGgmlCudaOpGatedDeltaNetImpl(pCtx, pDst, &pCache);
}
