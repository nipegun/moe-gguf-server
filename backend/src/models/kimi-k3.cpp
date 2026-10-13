#include "models.h"

#include <algorithm>
#include "mgs-memory-recurrent.h"

//
// Kimi-K3 text model: hybrid KDA (linear) + MLA (full) attention, as in kimi-linear.
// Parts that kimi-linear does not have:
//   1. cross-layer residual attention  (attn_res_block_size)
//   2. latent MoE                      (routed experts run at n_expert_latent)
//   3. situ activation                 (replaces SwiGLU everywhere)
//   4. MLA output gate                 (sigmoid gate before o_proj)
//   5. full-rank KDA gate              (single ssm_g instead of ssm_g_a/ssm_g_b)
//

void mgs_model_kimi_k3::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvAttentionKeyLengthMla, vHparams.vNEmbdHeadKMlaImpl);
  pMl.mGetKey(cLlmKvAttentionValueLengthMla, vHparams.vNEmbdHeadVMlaImpl);
  pMl.mGetKey(cLlmKvAttentionQLoraRank, vHparams.vNLoraQ, false);
  pMl.mGetKey(cLlmKvAttentionKvLoraRank, vHparams.vNLoraKv);
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvKdaHeadDim, vHparams.vNEmbdHeadKda);
  pMl.mGetKey(cLlmKvKdaGateLowerBound, vHparams.vKdaGateLowerBound, false);

  // the MLA cache holds the compressed latent; the published GGUFs store a value_length that is not the latent size
  vHparams.vNEmbdHeadVFull = vHparams.vNLoraKv;

  // n_head_kv == 0 marks a KDA (recurrent) layer, as in kimi-linear
  for (uint32_t vI = 0; vI < vHparams.mNLayer(); ++vI) {
    vHparams.aIsRecrImpl[vI] = vHparams.mNHeadKv(vI) == 0;
  }

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  pMl.mGetKey(cLlmKvExpertLatentLength, vHparams.vNExpertLatent, false);

  pMl.mGetKey(cLlmKvAttnResBlockSize, vHparams.vAttnResBlockSize);
  pMl.mGetKey(cLlmKvActivationSituBeta, vHparams.vSituBeta);
  pMl.mGetKey(cLlmKvActivationSituLinearBeta, vHparams.vSituLinearBeta);

  switch (vHparams.mNLayer()) {
  case 93:
    vType = cLlmType2Point8tA50b;
    break; // Kimi-K3
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_kimi_k3::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNEmbdLatent = vHparams.vNExpertLatent > 0 ? vHparams.vNExpertLatent : cNEmbd;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  if (vHparams.vAttnResBlockSize > 0) {
    vOutputResScore = mCreateTensor(cTn(cLlmTensorOutputResScore, "weight"), {cNEmbd}, 0);
  }

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (vHparams.vAttnResBlockSize > 0) {
      vLayer.vAttnResScore = mCreateTensor(cTn(cLlmTensorAttnResScore, "weight", vI), {cNEmbd}, 0);
      vLayer.vFfnResScore = mCreateTensor(cTn(cLlmTensorFfnResScore, "weight", vI), {cNEmbd}, 0);
    }

    const int64_t cHeadDim = vHparams.vNEmbdHeadKda;
    const int64_t cDConv = vHparams.vSsmDConv;
    const int64_t cDInner = cHeadDim * cNHead;

    if (vHparams.mIsRecr(vI)) {
      // conv1d is stored 4D [d_conv, 1, d_inner, 1] or 3D (quantization drops the trailing 1)
      auto fConv = [&](llm_tensor pTid) {
        ggml_tensor *vT = mCreateTensor(cTn(pTid, "weight", vI), {cDConv, 1, cDInner, 1}, cTensorNotRequired);
        return vT ? vT : mCreateTensor(cTn(pTid, "weight", vI), {cDConv, 1, cDInner}, 0);
      };
      vLayer.vSsmQConv = fConv(cLlmTensorSsmConv1dQ);
      vLayer.vSsmKConv = fConv(cLlmTensorSsmConv1dK);
      vLayer.vSsmVConv = fConv(cLlmTensorSsmConv1dV);

      vLayer.vWq = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", vI), {cNEmbd, cDInner}, 0);
      vLayer.vWk = mCreateTensor(cTn(cLlmTensorAttnK, "weight", vI), {cNEmbd, cDInner}, 0);
      vLayer.vWv = mCreateTensor(cTn(cLlmTensorAttnV, "weight", vI), {cNEmbd, cDInner}, 0);

      vLayer.vSsmFA = mCreateTensor(cTn(cLlmTensorSsmFA, "weight", vI), {cNEmbd, cHeadDim}, 0);
      vLayer.vSsmFB = mCreateTensor(cTn(cLlmTensorSsmFB, "weight", vI), {cHeadDim, cDInner}, 0);
      vLayer.vSsmBeta = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", vI), {cNEmbd, cNHead}, 0);

      // K3's A_log is a plain 1-D [n_head] tensor holding -exp(A_log)
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmANoscan, vI), {cNHead}, 0);
      vLayer.vSsmDtB = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", vI), {cDInner}, 0);

      // K3 uses a single full-rank gate instead of kimi-linear's g_a/g_b pair
      vLayer.vSsmG = mCreateTensor(cTn(cLlmTensorSsmG, "weight", vI), {cNEmbd, cDInner}, 0);
      vLayer.vSsmONorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", vI), {cHeadDim}, 0);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cDInner, cNEmbd}, 0);
    } else {
      const int64_t cQLoraRank = vHparams.vNLoraQ;
      const int64_t cKvLoraRank = vHparams.vNLoraKv;
      const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
      const int64_t cNEmbdHeadVMla = vHparams.mNEmbdHeadVMla();
      const int64_t cQkRopeHeadDim = vHparams.mNRot();
      const int64_t cQkNopeHeadDim = cNEmbdHeadKMla - cQkRopeHeadDim;

      vLayer.vAttnQANorm = mCreateTensor(cTn(cLlmTensorAttnQANorm, "weight", vI), {cQLoraRank}, cTensorNotRequired);
      vLayer.vAttnKvANorm = mCreateTensor(cTn(cLlmTensorAttnKvANorm, "weight", vI), {cKvLoraRank}, 0);

      if (vLayer.vAttnQANorm) {
        vLayer.vWqA = mCreateTensor(cTn(cLlmTensorAttnQA, "weight", vI), {cNEmbd, cQLoraRank}, 0);
        vLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQB, "weight", vI), {cQLoraRank, cNHead * cNEmbdHeadKMla}, 0);
      } else {
        vLayer.vWq = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", vI), {cNEmbd, cNHead * cNEmbdHeadKMla}, 0);
      }

      vLayer.vWkvAMqa =
          mCreateTensor(cTn(cLlmTensorAttnKvAMqa, "weight", vI), {cNEmbd, cKvLoraRank + cQkRopeHeadDim}, 0);
      vLayer.vWkB = mCreateTensor(cTn(cLlmTensorAttnKB, "weight", vI), {cQkNopeHeadDim, cKvLoraRank, cNHead}, 0);
      vLayer.vWvB = mCreateTensor(cTn(cLlmTensorAttnVB, "weight", vI), {cKvLoraRank, cNEmbdHeadVMla, cNHead}, 0);

      // K3: sigmoid output gate applied to the attention output before o_proj
      vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", vI), {cNEmbd, cNHead * cNEmbdHeadVMla},
                                       cTensorNotRequired);

      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNHead * cNEmbdHeadVMla, cNEmbd}, 0);
    }

    if (vI < (int)vHparams.vNLayerDenseLead) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    } else {
      const int64_t cNFfExp = vHparams.vNFfExp;

      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);

      // routed experts live in the latent space
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbdLatent, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbdLatent, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbdLatent, cNFfExp, cNExpert}, 0);

      if (vHparams.vNExpertLatent > 0) {
        vLayer.vFfnRoutedDown = mCreateTensor(cTn(cLlmTensorFfnRoutedDown, "weight", vI), {cNEmbd, cNEmbdLatent}, 0);
        vLayer.vFfnRoutedUp = mCreateTensor(cTn(cLlmTensorFfnRoutedUp, "weight", vI), {cNEmbdLatent, cNEmbd}, 0);
        vLayer.vFfnRoutedNorm =
            mCreateTensor(cTn(cLlmTensorFfnRoutedNorm, "weight", vI), {cNEmbdLatent}, cTensorNotRequired);
      }

      // shared experts stay at n_embd, width = moe_intermediate_size * n_expert_shared
      const int64_t cNFfShexp = cNFfExp * (vHparams.vNExpertShared > 0 ? vHparams.vNExpertShared : 1);
      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexp}, cTensorNotRequired);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, cTensorNotRequired);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, cTensorNotRequired);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_kimi_k3::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
// linear_beta <= 0 disables the transform on the up branch
static ggml_tensor *fKimiK3Situ(ggml_context *pCtx0, ggml_tensor *pGate, ggml_tensor *pUp, float pBeta,
                                float pLinearBeta) {
  ggml_tensor *vA = ggml_scale(pCtx0, ggml_tanh(pCtx0, ggml_scale(pCtx0, pGate, 1.0f / pBeta)), pBeta);
  vA = ggml_mul(pCtx0, vA, ggml_sigmoid(pCtx0, pGate));

  if (pLinearBeta > 0.0f) {
    pUp = ggml_scale(pCtx0, ggml_tanh(pCtx0, ggml_scale(pCtx0, pUp, 1.0f / pLinearBeta)), pLinearBeta);
  }
  return ggml_mul(pCtx0, vA, pUp);
}

//
// cross-layer residual attention
//

// layout is [n_embd, n_ckpt, n_tokens]: rms_norm reduces over ne0, dsv4_hc_pre over ne1
// append the new checkpoint, do not re-fold the whole chain
void mgs_model_kimi_k3::graph::mResPush(ggml_tensor *pCur, int64_t pNEmbd, int64_t pNTokens) {
  ggml_tensor *vCkpt = ggml_reshape_3d(vCtx0, pCur, pNEmbd, 1, pNTokens);

  vResiStack = vResiStack ? ggml_concat(vCtx0, vResiStack, vCkpt, 1) : vCkpt;
}

ggml_tensor *mgs_model_kimi_k3::graph::mResMix(ggml_tensor *pCur, ggml_tensor *pScoreW, int64_t pNTokens, int pIl) {
  if (!vResiStack) {
    return pCur; // layer 0: nothing banked yet
  }

  const int cNCkpt = (int)vResiStack->ne[1];
  const float cEps = vHparams.vFNormRmsEps;

  ggml_tensor *vSrc = vResiStack; // [n_embd, n_ckpt, n_tokens]

  // one rms_norm scores all checkpoints at once
  // note: the scores use the normalized values, but the sum below uses the raw ones
  ggml_tensor *vScSrc = ggml_rms_norm(vCtx0, vSrc, cEps);
  vScSrc = ggml_mul(vCtx0, vScSrc, pScoreW);
  vScSrc = ggml_sum_rows(vCtx0, vScSrc); // [1, n_ckpt, n_tokens]
  vScSrc = ggml_reshape_2d(vCtx0, vScSrc, cNCkpt, pNTokens);

  // the current residual stream is scored apart, so the stack stays append-only
  ggml_tensor *vScCur = ggml_rms_norm(vCtx0, pCur, cEps);
  vScCur = ggml_mul(vCtx0, vScCur, pScoreW);
  vScCur = ggml_sum_rows(vCtx0, vScCur); // [1, n_tokens]

  ggml_tensor *vScores = ggml_concat(vCtx0, vScSrc, vScCur, 0); // [n_ckpt+1, n_tokens]
  ggml_tensor *vProbs = ggml_soft_max(vCtx0, vScores);          // over ne0 = n_ckpt+1
  mCb(vProbs, "res_probs", pIl);

  // split the sum: hc_pre handles the stack, a broadcast-multiply the current stream
  ggml_tensor *vPSrc = ggml_cont(vCtx0, ggml_view_2d(vCtx0, vProbs, cNCkpt, pNTokens, vProbs->nb[1], 0));
  ggml_tensor *vPCur =
      ggml_cont(vCtx0, ggml_view_2d(vCtx0, vProbs, 1, pNTokens, vProbs->nb[1], vProbs->nb[0] * cNCkpt));

  ggml_tensor *vOut = ggml_dsv4_hc_pre(vCtx0, vSrc, vPSrc);
  vOut = ggml_add(vCtx0, vOut, ggml_mul(vCtx0, pCur, vPCur));

  return vOut;
}

mgs_model_kimi_k3::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_delta_net_base(pParams), vModel(pModel) {
  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  mCb(vInpL, "inp_embd", -1);

  // K3 MLA is nope-only, so there is no position input
  auto *vInpK = mBuildInpMemHybridK();
  auto *vInpRs = vInpK->mGetRecr();
  auto *vInpAttnK = vInpK->mGetAttn();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const int64_t cNHeadKda = vHparams.mNHead();
  const int64_t cHeadDim = vHparams.vNEmbdHeadKda;
  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = cNHeadKda * cHeadDim;
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

  const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
  const int64_t cKvLoraRank = vHparams.vNLoraKv;
  const int64_t cNEmbdHeadQkRope = vHparams.mNRot();
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadKMla - cNEmbdHeadQkRope;
  const float cKqScaleMla = 1.0f / sqrtf((float)cNEmbdHeadKMla);

  const uint32_t cResBs = vHparams.vAttnResBlockSize;
  const bool cUseAttnRes = cResBs > 0;

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = pModel.lLayers[vIl];

    // the residual stream, banked on checkpoint layers and then restarted
    // from the attention output alone
    ggml_tensor *vPrefixSum = vInpL;

    vCur = cUseAttnRes ? mResMix(vPrefixSum, vLayer.vAttnResScore, cNTokens, vIl) : vPrefixSum;

    bool vBanked = false;
    if (cUseAttnRes && (uint32_t)vIl % cResBs == 0) {
      mResPush(vPrefixSum, cNEmbd, cNTokens); // banks the raw layer input, not the mixed one
      vBanked = true;
    }

    vCur = mBuildNorm(vCur, vLayer.vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);
    ggml_build_forward_expand(vGf, vCur);

    if (vHparams.mIsRecr(vIl)) {
      vCur = mBuildKdaLayer(vCur, vLayer, vInpRs, cDConv, cHeadDim, cNHeadKda, cDInner, cNSeqTokens, cNSeqs, vIl);
    } else {
      vCur = mBuildMlaLayer(vCur, vLayer, vInpAttnK, cNEmbdHeadKMla, cKvLoraRank, cNEmbdHeadQkRope, cNEmbdHeadQkNope,
                            cKqScaleMla, vIl);
    }

    vPrefixSum = vBanked ? vCur : ggml_add(vCtx0, vPrefixSum, vCur);
    mCb(vPrefixSum, "prefix_sum_attn", vIl);

    vCur = cUseAttnRes ? mResMix(vPrefixSum, vLayer.vFfnResScore, cNTokens, vIl) : vPrefixSum;

    vCur = mBuildNorm(vCur, vLayer.vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      ggml_tensor *vG = ggml_mul_mat(vCtx0, vLayer.vFfnGate, vCur);
      ggml_tensor *vU = ggml_mul_mat(vCtx0, vLayer.vFfnUp, vCur);
      vCur = fKimiK3Situ(vCtx0, vG, vU, vHparams.vSituBeta, vHparams.vSituLinearBeta);
      vCur = ggml_mul_mat(vCtx0, vLayer.vFfnDown, vCur);
      mCb(vCur, "ffn_out", vIl);
    } else {
      vCur = mBuildLatentMoe(vCur, vLayer, vIl);
    }

    vPrefixSum = ggml_add(vCtx0, vPrefixSum, vCur);
    vPrefixSum = mBuildCvec(vPrefixSum, vIl);
    mCb(vPrefixSum, "l_out", vIl);

    vInpL = vPrefixSum;
  }

  vCur = vInpL;

  // final mix, then narrow to the output tokens
  if (cUseAttnRes) {
    vCur = mResMix(vCur, pModel.vOutputResScore, cNTokens, -1);
  }
  if (vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = ggml_mul_mat(vCtx0, pModel.vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

//
// KDA layer
//

// causal conv1d over one of Q/K/V. pQkv selects which third of the conv state to use
static ggml_tensor *fKimiK3Conv1d(ggml_cgraph *pGf, ggml_context *pCtx0, ggml_tensor *pConvStatesAll,
                                  ggml_tensor *pConvStateAll, int64_t pQkv, ggml_tensor *pX, ggml_tensor *pProjW,
                                  ggml_tensor *pConvW, int64_t pDConv, int64_t pHeadDim, int64_t pNHead,
                                  int64_t pNSeqTokens, int64_t pNSeqs, int64_t pNTokens, int64_t pKvHead,
                                  int64_t pMemSize, int64_t pKRs) {
  const int64_t cDInner = pHeadDim * pNHead;
  const int64_t cConvStateSize = (pDConv - 1) * cDInner;
  const int64_t cNEmbdRTotal = 3 * cConvStateSize;

  ggml_tensor *vConvStateX = ggml_view_3d(pCtx0, pConvStateAll, pDConv - 1, cDInner, pNSeqs,
                                          (pDConv - 1) * ggml_element_size(pConvStateAll),
                                          cNEmbdRTotal * ggml_element_size(pConvStateAll),
                                          pQkv * cConvStateSize * ggml_element_size(pConvStateAll));

  ggml_tensor *vXProj = ggml_mul_mat(pCtx0, pProjW, pX);
  ggml_tensor *vX3d = ggml_reshape_3d(pCtx0, vXProj, cDInner, pNSeqTokens, pNSeqs);
  ggml_tensor *vConvX = ggml_concat(pCtx0, vConvStateX, ggml_transpose(pCtx0, vX3d), 0);

  // group s holds the conv window s tokens back.
  // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
  for (int64_t vS = 0; vS < pKRs; ++vS) {
    const int64_t cSIdx = std::max<int64_t>(0, pNSeqTokens - vS);
    ggml_tensor *vConvXS = ggml_view_3d(pCtx0, vConvX, pDConv - 1, cDInner, pNSeqs, vConvX->nb[1], vConvX->nb[2],
                                        cSIdx * vConvX->nb[0]);
    ggml_build_forward_expand(
        pGf, ggml_cpy(pCtx0, vConvXS,
                      ggml_view_3d(pCtx0, pConvStatesAll, pDConv - 1, cDInner, pNSeqs,
                                   (pDConv - 1) * ggml_element_size(pConvStatesAll),
                                   cNEmbdRTotal * ggml_element_size(pConvStatesAll),
                                   ((vS * pMemSize + pKvHead) * cNEmbdRTotal + pQkv * cConvStateSize) *
                                       ggml_element_size(pConvStatesAll))));
  }

  ggml_tensor *vConvWeight = ggml_reshape_2d(pCtx0, pConvW, pDConv, cDInner);
  ggml_tensor *vXcur = ggml_ssm_conv(pCtx0, vConvX, vConvWeight);
  vXcur = ggml_reshape_2d(pCtx0, vXcur, cDInner, pNTokens);
  vXcur = ggml_silu(pCtx0, vXcur);

  return ggml_reshape_4d(pCtx0, vXcur, pHeadDim, pNHead, pNSeqTokens, pNSeqs);
}

ggml_tensor *mgs_model_kimi_k3::graph::mBuildKdaLayer(ggml_tensor *pCur, const mgs_layer &pLayer,
                                                      llm_graph_input_rs *pInpRs, int64_t pDConv, int64_t pHeadDim,
                                                      int64_t pNHeadKda, int64_t pDInner, int64_t pNSeqTokens,
                                                      int64_t pNSeqs, int pIl) {
  const auto *vMctxCur = pInpRs->vMctx;
  const auto cKvHead = vMctxCur->mGetHead();

  ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor *vConvStateAll = mBuildRs(pInpRs, vConvStatesAll, vHparams.mNEmbdR(), pNSeqs);

  const int64_t cMemSize = vMctxCur->mGetSize();
  const int64_t cKRs = (int64_t)vCparams.vNRsSeq + 1;

  ggml_tensor *vQcur = fKimiK3Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 0, pCur, pLayer.vWq, pLayer.vSsmQConv,
                                     pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize,
                                     cKRs);
  ggml_tensor *vKcur = fKimiK3Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 1, pCur, pLayer.vWk, pLayer.vSsmKConv,
                                     pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize,
                                     cKRs);
  ggml_tensor *vVcur = fKimiK3Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 2, pCur, pLayer.vWv, pLayer.vSsmVConv,
                                     pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize,
                                     cKRs);
  mCb(vQcur, "kda_q_conv", pIl);
  mCb(vKcur, "kda_k_conv", pIl);
  mCb(vVcur, "kda_v_conv", pIl);

  // gate_lower_bound is not a clamp - when set, it swaps the decay gate activation:
  //   unset (kimi-linear):  g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
  //   set   (K3, -5.0):     g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
  // ssm_a holds -exp(A_log) (folded at conversion time), so exp(A_log) == -ssm_a
  ggml_tensor *vFA = ggml_mul_mat(vCtx0, pLayer.vSsmFA, pCur);
  ggml_tensor *vG1 = ggml_mul_mat(vCtx0, pLayer.vSsmFB, vFA);
  vG1 = ggml_add(vCtx0, vG1, pLayer.vSsmDtB);

  ggml_tensor *vA = ggml_reshape_3d(vCtx0, pLayer.vSsmA, 1, pNHeadKda, 1);

  if (vHparams.vKdaGateLowerBound > -INFINITY) {
    vG1 = ggml_reshape_3d(vCtx0, vG1, pHeadDim, pNHeadKda, cNTokens);
    vG1 = ggml_mul(vCtx0, vG1, vA); // -exp(A_log) * (...)
    vG1 = ggml_sigmoid(vCtx0, ggml_scale(vCtx0, vG1, -1.0f));
    vG1 = ggml_scale(vCtx0, vG1, vHparams.vKdaGateLowerBound);
  } else {
    vG1 = ggml_softplus(vCtx0, vG1);
    vG1 = ggml_reshape_3d(vCtx0, vG1, pHeadDim, pNHeadKda, cNTokens);
    vG1 = ggml_mul(vCtx0, vG1, vA);
  }
  mCb(vG1, "kda_g1", pIl);

  vG1 = ggml_reshape_4d(vCtx0, vG1, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs);

  ggml_tensor *vBeta = ggml_mul_mat(vCtx0, pLayer.vSsmBeta, pCur);
  vBeta = ggml_reshape_4d(vCtx0, vBeta, 1, pNHeadKda, pNSeqTokens, pNSeqs);
  vBeta = ggml_sigmoid(vCtx0, vBeta);
  mCb(vBeta, "kda_beta", pIl);

  ggml_tensor *vCur3d = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], pNSeqTokens, pNSeqs);

  ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(pIl);
  ggml_tensor *vState = mBuildRs(pInpRs, vSsmStatesAll, vHparams.mNEmbdS(), pNSeqs);
  vState = ggml_reshape_4d(vCtx0, vState, pHeadDim, pHeadDim, pNHeadKda, pNSeqs);

  const float cEpsNorm = vHparams.vFNormRmsEps;
  vQcur = fBuildGdnL2Norm(vCtx0, vQcur, cEpsNorm);
  vKcur = fBuildGdnL2Norm(vCtx0, vKcur, cEpsNorm);

  ggml_tensor *vOutput =
      mBuildRecurrentAttn(pInpRs, vSsmStatesAll, vQcur, vKcur, vVcur, vG1, vBeta, vState, pIl);
  vOutput = ggml_cont(vCtx0, vOutput);
  mCb(vOutput, "kda_scan_out", pIl);

  // K3: single full-rank gate (kimi-linear factors this as g_b(g_a(x)))
  ggml_tensor *vCur2d = ggml_reshape_2d(vCtx0, vCur3d, vCur3d->ne[0], pNSeqTokens * pNSeqs);
  ggml_tensor *vG2 = ggml_mul_mat(vCtx0, pLayer.vSsmG, vCur2d);
  vG2 = ggml_reshape_3d(vCtx0, vG2, pHeadDim, pNHeadKda, pNSeqTokens * pNSeqs);

  ggml_tensor *vO = ggml_reshape_3d(vCtx0, vOutput, pHeadDim, pNHeadKda, pNSeqTokens * pNSeqs);
  ggml_tensor *vNormed = mBuildNorm(vO, pLayer.vSsmONorm, nullptr, cLlmNormRms, pIl);
  mCb(vG2, "kda_g2", pIl);
  mCb(vNormed, "kda_normed", pIl);
  ggml_tensor *vGated = ggml_mul(vCtx0, vNormed, ggml_sigmoid(vCtx0, vG2));

  vGated = ggml_cont_2d(vCtx0, vGated, pDInner, cNTokens);
  ggml_tensor *vOut = ggml_mul_mat(vCtx0, pLayer.vWo, vGated);
  mCb(vOut, "kda_out", pIl);

  return vOut;
}

//
// MLA layer (nope-only, with K3's sigmoid output gate)
//

ggml_tensor *mgs_model_kimi_k3::graph::mBuildMlaLayer(ggml_tensor *pCur, const mgs_layer &pLayer,
                                                      llm_graph_input_attn_k *pInpAttnK, int64_t pNEmbdHeadKMla,
                                                      int64_t pKvLoraRank, int64_t pNEmbdHeadQkRope,
                                                      int64_t pNEmbdHeadQkNope, float pKqScale, int pIl) {
  ggml_tensor *vInpGate = pCur; // the output gate reads the *normed* layer input

  ggml_tensor *vQcur;
  if (pLayer.vWqA) {
    vQcur = ggml_mul_mat(vCtx0, pLayer.vWqA, pCur);
    vQcur = mBuildNorm(vQcur, pLayer.vAttnQANorm, nullptr, cLlmNormRms, pIl);
    vQcur = ggml_mul_mat(vCtx0, pLayer.vWqB, vQcur);
  } else {
    vQcur = ggml_mul_mat(vCtx0, pLayer.vWq, pCur);
  }

  ggml_tensor *vKvCmprPe = ggml_mul_mat(vCtx0, pLayer.vWkvAMqa, pCur);

  ggml_tensor *vKvCmpr = ggml_view_2d(vCtx0, vKvCmprPe, pKvLoraRank, cNTokens,
                                      ggml_row_size(vKvCmprPe->type, pKvLoraRank + pNEmbdHeadQkRope), 0);
  ggml_tensor *vKPe = ggml_view_3d(vCtx0, vKvCmprPe, pNEmbdHeadQkRope, 1, cNTokens,
                                   ggml_row_size(vKvCmprPe->type, pKvLoraRank + pNEmbdHeadQkRope),
                                   ggml_row_size(vKvCmprPe->type, pKvLoraRank + pNEmbdHeadQkRope),
                                   ggml_row_size(vKvCmprPe->type, pKvLoraRank));

  // no RoPE: the converter asserts mla_use_nope
  vKvCmpr = mBuildNorm(vKvCmpr, pLayer.vAttnKvANorm, nullptr, cLlmNormRms, pIl);

  ggml_tensor *vQNope =
      ggml_view_3d(vCtx0, vQcur, pNEmbdHeadQkNope, cNHead, cNTokens, ggml_row_size(vQcur->type, pNEmbdHeadKMla),
                   ggml_row_size(vQcur->type, pNEmbdHeadKMla) * cNHead, 0);
  ggml_tensor *vQPe =
      ggml_view_3d(vCtx0, vQcur, pNEmbdHeadQkRope, cNHead, cNTokens, ggml_row_size(vQcur->type, pNEmbdHeadKMla),
                   ggml_row_size(vQcur->type, pNEmbdHeadKMla) * cNHead, ggml_row_size(vQcur->type, pNEmbdHeadQkNope));

  vQNope = ggml_permute(vCtx0, vQNope, 0, 2, 1, 3);
  ggml_tensor *vQNopeAbsorbed = ggml_mul_mat(vCtx0, pLayer.vWkB, vQNope);
  vQNopeAbsorbed = ggml_permute(vCtx0, vQNopeAbsorbed, 0, 2, 1, 3);

  ggml_tensor *vQ = ggml_concat(vCtx0, vQNopeAbsorbed, vQPe, 0);
  ggml_tensor *vKvCmpr3d = ggml_reshape_3d(vCtx0, vKvCmpr, pKvLoraRank, 1, cNTokens);
  ggml_tensor *vK = ggml_concat(vCtx0, vKvCmpr3d, vKPe, 0);
  ggml_tensor *vV = vKvCmpr3d;

  // wo == NULL: the output projection is applied after the gate below
  ggml_tensor *vOut =
      mBuildAttn(pInpAttnK, nullptr, NULL, nullptr, vQ, vK, vV, nullptr, nullptr, pLayer.vWvB, pKqScale, pIl);

  // K3: attn_output *= sigmoid(g_proj(x)), then o_proj
  if (pLayer.vWqkvGate) {
    ggml_tensor *vG = ggml_sigmoid(vCtx0, ggml_mul_mat(vCtx0, pLayer.vWqkvGate, vInpGate));
    vOut = ggml_mul(vCtx0, vOut, vG);
    mCb(vOut, "mla_gated", pIl);
  }

  vOut = ggml_mul_mat(vCtx0, pLayer.vWo, vOut);
  mCb(vOut, "mla_out", pIl);

  return vOut;
}

//
// latent MoE: down-project, run the routed experts in the latent space, norm, up-project;
// shared experts stay at n_embd and read the un-projected input.
//

ggml_tensor *mgs_model_kimi_k3::graph::mBuildLatentMoe(ggml_tensor *pCur, const mgs_layer &pLayer, int pIl) {
  ggml_tensor *vIdentity = pCur;

  ggml_tensor *vRoutedIn = pLayer.vFfnRoutedDown ? ggml_mul_mat(vCtx0, pLayer.vFfnRoutedDown, pCur) : pCur;

  // the router scores the full-width input while the experts take the latent one,
  // so the logits are computed here and passed to mBuildMoeFfn
  ggml_tensor *vLogits = ggml_mul_mat(vCtx0, pLayer.vFfnGateInp, vIdentity);
  mCb(vLogits, "ffn_moe_logits", pIl);

  ggml_tensor *vMoeOut =
      mBuildMoeFfn(vRoutedIn,
                   nullptr, // gate_inp unused: the logits above are passed instead
                   pLayer.vFfnUpExps, pLayer.vFfnGateExps, pLayer.vFfnDownExps, pLayer.vFfnExpProbsB,
                   vHparams.vNExpert, vHparams.vNExpertUsed, cLlmFfnSitu, vHparams.vExpertWeightsNorm,
                   vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, pIl, vLogits);
  mCb(vMoeOut, "ffn_moe_out", pIl);

  if (pLayer.vFfnRoutedNorm) {
    vMoeOut = mBuildNorm(vMoeOut, pLayer.vFfnRoutedNorm, NULL, cLlmNormRms, pIl);
  }
  if (pLayer.vFfnRoutedUp) {
    vMoeOut = ggml_mul_mat(vCtx0, pLayer.vFfnRoutedUp, vMoeOut);
  }

  if (pLayer.vFfnGateShexp) {
    ggml_tensor *vG = ggml_mul_mat(vCtx0, pLayer.vFfnGateShexp, vIdentity);
    ggml_tensor *vU = ggml_mul_mat(vCtx0, pLayer.vFfnUpShexp, vIdentity);
    ggml_tensor *vSh = fKimiK3Situ(vCtx0, vG, vU, vHparams.vSituBeta, vHparams.vSituLinearBeta);
    vSh = ggml_mul_mat(vCtx0, pLayer.vFfnDownShexp, vSh);
    mCb(vSh, "ffn_shexp", pIl);
    vMoeOut = ggml_add(vCtx0, vMoeOut, vSh);
  }

  mCb(vMoeOut, "ffn_out", pIl);
  return vMoeOut;
}
