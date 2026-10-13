// Laguna (poolside): sigmoid-routed MoE with a score-correction bias, one shared
// expert, a softplus attention output gate, QK-norm, and per-layer-type RoPE
// (YaRN on full-attention layers, plain RoPE on sliding-window layers). XS.2 is
// hybrid full/SWA with a per-head gate; M.1 is full-attention with a per-element
// gate. Shares the MoE/gate structure with afmoe.

#include "models.h"

void mgs_model_laguna::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);

  // Laguna ships one shared expert and stores its size directly (routed and
  // shared experts may differ), so read the size from expert_shared_feed_forward_length.
  // The count is not in the config; default to 1 but read the key if present.
  vHparams.vNExpertShared = 1;
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared, false);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  if (vHparams.vNFfShexp == 0) {
    // Weightless test fixtures omit this key; derive a nonzero
    // size so the shared expert is still built. Real GGUFs always carry the
    // exact value (routed and shared FF lengths may differ).
    vHparams.vNFfShexp = vHparams.vNFfExp * vHparams.vNExpertShared;
  }

  // Sliding-window attention is OPTIONAL. XS.2 is hybrid (full / SWA / SWA /
  // SWA repeating, period 4 starting with full); M.1 has no sliding window
  // (all layers full attention). When sliding_window is absent or zero we
  // leave swa_type = NONE and skip the SWA-specific per-layer-type RoPE.
  vHparams.vNSwa = 0;
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa, false);
  if (vHparams.vNSwa > 0) {
    vHparams.vSwaType = cMgsSwaTypeStandard;

    uint32_t vSwaPeriod = 4;
    pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
    vHparams.mSetSwaPattern(vSwaPeriod, /*dense_first=*/true); // XS.2: FULL at il%4==0

    // Per-layer-type RoPE: full layers use YaRN θ=500000 over 64 dims;
    // SWA layers use default RoPE θ=10000 over 128 dims. Base load_hparams
    // already reads ROPE_FREQ_BASE and ROPE_DIMENSION_COUNT into the
    // non-SWA fields; we explicitly pull the SWA mirrors here.
    vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
    vHparams.vRopeFreqScaleTrainSwa =
        1.0f; // SWA uses plain RoPE (no YaRN scaling); do NOT inherit full layers 1/factor
    pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);
    pMl.mGetKey(cLlmKvRopeDimensionCountSwa, vHparams.vNRotSwa, false);
  }

  switch (vHparams.mNLayer()) {
  case 40:
    vType = cLlmType30bA3b;
    break; // Laguna-XS.2
  case 48:
    vType = cLlmType118bA8b;
    break; // Laguna-S.2
  case 70:
    vType = cLlmType230bA10b;
    break; // Laguna-M.1
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_laguna::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  if (vOutput == NULL) {
    // tied embeddings fallback
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  const int64_t cNFfExp = vHparams.vNFfExp;
  const int64_t cNFfShexp = vHparams.vNFfShexp;

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    // Per-layer head count — Laguna varies n_head between full and SWA
    // layers (48 vs 64 in XS.2). KV head count is uniform.
    const int64_t cNHeadIl = vHparams.mNHead(vI);
    const int64_t cNHeadKvIl = vHparams.mNHeadKv(vI);
    const int64_t cNEmbdQIl = cNEmbdHeadK * cNHeadIl;
    const int64_t cNEmbdKIl = cNEmbdHeadK * cNHeadKvIl;
    const int64_t cNEmbdVIl = cNEmbdHeadV * cNHeadKvIl;

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdQIl, cNEmbdKIl, cNEmbdVIl, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdQIl, cNEmbd}, 0);

    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);

    // Attention output gate. XS.2 is per-head (g_proj -> n_head, one scalar
    // per head broadcast over head_dim at multiply time); M.1 is per-element
    // (g_proj -> n_head*head_dim, like afmoe). Detect from the stored tensor
    // shape so a single arch handles both; the graph mirrors this check.
    // Gate width selects per-head vs per-element. Real GGUFs always carry the
    // gate tensor, so read the width from it and require EXACTLY one of the two
    // valid widths -- never guess between them. Weightless test
    // fixtures have no gate tensor; fall back to the per-head layout so
    // the per-head reshape path is still exercised.
    const int64_t cNGatePerHead = cNHeadIl;
    const int64_t cNGatePerElem = cNEmbdHeadK * cNHeadIl;
    const ggml_tensor *vGateMeta = pMl.mGetTensorMeta(cTn(cLlmTensorAttnGate, "weight", vI).mStr().c_str());
    int64_t vNGateOut;
    if (vGateMeta != nullptr) {
      vNGateOut = vGateMeta->ne[1];
      if (vNGateOut != cNGatePerHead && vNGateOut != cNGatePerElem) {
        GGML_ABORT("Laguna: unexpected attention gate width %lld at layer %d "
                   "(expected %lld per-head or %lld per-element)",
                   (long long)vNGateOut, vI, (long long)cNGatePerHead, (long long)cNGatePerElem);
      }
    } else {
      vNGateOut = cNGatePerHead;
    }
    vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", vI), {cNEmbd, vNGateOut}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if ((uint32_t)vI >= vHparams.vNLayerDenseLead) {
      // MoE layer
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);

      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);

      // Always-on shared expert.
      vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexp}, 0);
      vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, 0);
      vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, 0);
    } else {
      // Dense layer (the leading n_layer_dense_lead layers)
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_laguna::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_laguna::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  // No MuP embedding scale (laguna omits this; afmoe scales by sqrt(hidden)).

  ggml_tensor *vInpPos = mBuildInpPos();
  // XS.2 is hybrid SWA -> interleaved-SWA KV input; M.1 is all-full -> plain
  // KV input. Pick the matching input (and build_attn overload) per swa_type.
  const bool cHasSwa = vHparams.vSwaType != cMgsSwaTypeNone;
  llm_graph_input_attn_kv *vInpAttnKv = cHasSwa ? nullptr : mBuildAttnInpKv();
  llm_graph_input_attn_kv_iswa *vInpAttnIswa = cHasSwa ? mBuildAttnInpKvIswa() : nullptr;
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const bool cIsSwaIl = vHparams.mIsSwa(vIl);
    const int64_t cNHeadIl = vHparams.mNHead(vIl);
    const int64_t cNHeadKvIl = vHparams.mNHeadKv(vIl);

    // Per-layer-type RoPE config. SWA layers run plain rope (no YaRN),
    // achieved by zeroing the YaRN ext/beta params for those layers.
    const int cNRotL = cIsSwaIl ? vHparams.vNRotSwa : cNRot;
    const float cFreqBaseL = cIsSwaIl ? vHparams.vRopeFreqBaseTrainSwa : cFreqBase;
    const float cFreqScaleL = cIsSwaIl ? vHparams.vRopeFreqScaleTrainSwa : cFreqScale;
    const float cExtFactorL = cIsSwaIl ? 0.0f : cExtFactor;
    // YaRN magnitude scaling (mscale) is already handled by the framework:
    // mgs_context pre-divides cparams.yarn_attn_factor by (1 + 0.1*ln(factor))
    // to cancel ggml rope_yarn's internal mscale *= 1 + 0.1*ln(1/freq_scale).
    // Pass attn_factor straight through (like every other arch); SWA layers run
    // plain RoPE (ext_factor 0, no mscale) so force 1.0 there.
    const float cAttnFactorL = cIsSwaIl ? 1.0f : cAttnFactor;
    const float cBetaFastL = cIsSwaIl ? 0.0f : cBetaFast;
    const float cBetaSlowL = cIsSwaIl ? 0.0f : cBetaSlow;
    const int cNCtxOrigL = cIsSwaIl ? vHparams.vNCtxTrain : cNCtxOrig;

    ggml_tensor *vInpSA = vInpL;

    // Pre-norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // Self-attention
    {
      ggml_tensor *vAttnInp = vCur; // saved for the gate projection

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHeadIl, cNHeadKvIl, vIl);

      // g_proj on the *pre-attention* hidden state (matches HF
      // reference: gate is computed from the same `hidden_states`
      // input as q/k/v, not from the attn output).
      ggml_tensor *vGate = mBuildLoraMm(pModel.lLayers[vIl].vWqkvGate, vAttnInp);
      mCb(vGate, "attn_gate_proj", vIl);

      // QK RMSNorm at head_dim level (Qwen3 style)
      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRotL, cRopeType, cNCtxOrigL, cFreqBaseL, cFreqScaleL,
                           cExtFactorL, cAttnFactorL, cBetaFastL, cBetaSlowL);
      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRotL, cRopeType, cNCtxOrigL, cFreqBaseL, cFreqScaleL,
                           cExtFactorL, cAttnFactorL, cBetaFastL, cBetaSlowL);
      mCb(vQcur, "Qcur_rope", vIl);
      mCb(vKcur, "Kcur_rope", vIl);

      vCur = cHasSwa
                 ? mBuildAttn(vInpAttnIswa, NULL, NULL, NULL, // o_proj deferred until after gating
                              vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, vIl)
                 : mBuildAttn(vInpAttnKv, NULL, NULL, NULL, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, vIl);
      mCb(vCur, "attn_out", vIl);

      // Softplus output gate (the unary kernel computes softplus in fp32
      // and casts back). Two shapes, distinguished by the g_proj output
      // dim (matching the load-time detection):
      //   XS.2 per-head     : gate [n_head_il, n_tokens] -> reshape to
      //                       [1, n_head_il, n_tokens] and broadcast over
      //                       head_dim against cur [head_dim, n_head, T].
      //   M.1  per-element  : gate [n_head_il*head_dim, n_tokens] spans the
      //                       full attention output -> direct ggml_mul.
      vGate = ggml_softplus(vCtx0, vGate);
      mCb(vGate, "attn_gate_softplus", vIl);

      const int64_t cNTokens = vCur->ne[1];
      if (pModel.lLayers[vIl].vWqkvGate->ne[1] == cNHeadIl) {
        vCur = ggml_reshape_3d(vCtx0, vCur, cNEmbdHead, cNHeadIl, cNTokens);
        vGate = ggml_reshape_3d(vCtx0, vGate, 1, cNHeadIl, cNTokens);
        vCur = ggml_mul(vCtx0, vCur, vGate);
        vCur = ggml_reshape_2d(vCtx0, vCur, cNEmbdHead * cNHeadIl, cNTokens);
      } else {
        vCur = ggml_mul(vCtx0, vCur, vGate);
      }
      mCb(vCur, "attn_gated", vIl);

      vCur = mBuildLoraMm(pModel.lLayers[vIl].vWo, vCur, pModel.lLayers[vIl].vWoS);
      mCb(vCur, "attn_o_proj", vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // Pre-norm only (no post-attn norm)
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl >= vHparams.vNLayerDenseLead) {
      // MoE: sigmoid routing + score-correction bias + sum-norm +
      // routed_scaling_factor (all handled by build_moe_ffn).
      ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
                                          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                                          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // Always-on shared expert, summed in parallel.
      ggml_tensor *vFfnShexp =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                    pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vFfnShexp, "ffn_shexp", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // Dense FFN for the leading n_layer_dense_lead layers (XS.2: 1, M.1: 3)
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    }

    // No post-ffn norm
    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    vInpL = vCur;
  }

  vCur = vInpL;
  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
