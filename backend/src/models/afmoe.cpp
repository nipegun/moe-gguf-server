#include "models.h"

void mgs_model_afmoe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa, false);

  // Set up interleaved sliding window attention (ISWA)
  // Pattern: 3 sliding - 1 full (global_attn_every_n_layers = 4)
  if (vHparams.vNSwa > 0) {
    vHparams.vSwaType = cMgsSwaTypeStandard;
    uint32_t vSwaPeriod = 4;
    pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
    vHparams.mSetSwaPattern(vSwaPeriod);

    vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
    vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;
    pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);
  } else {
    vHparams.vSwaType = cMgsSwaTypeNone;
  }

  // Default to sigmoid if not set
  if (vHparams.vExpertGatingFunc == cMgsExpertGatingFuncTypeNone) {
    vHparams.vExpertGatingFunc = cMgsExpertGatingFuncTypeSigmoid;
  }

  switch (vHparams.mNLayer()) {
  case 56:
    vType = cLlmType6b;
    break;
  case 32:
    vType = cLlmType26b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_afmoe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;
  const int64_t cNExpertShared = vHparams.vNExpertShared;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  const int64_t cNFfExp = vHparams.vNFfExp;

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    // dual attention normalization
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", vI), {cNEmbd}, 0);

    // attention projections
    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    // Q/K normalization
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);

    // attention gating
    vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", vI), {cNEmbd, cNEmbdHeadK * cNHead}, 0);

    // dual ffn normalization
    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vFfnPostNorm = mCreateTensor(cTn(cLlmTensorFfnPostNorm, "weight", vI), {cNEmbd}, 0);

    if (static_cast<uint32_t>(vI) >= vHparams.vNLayerDenseLead) {
      // MoE layers
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);

      // grouped expert weights
      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

      // shared expert
      if (cNExpertShared > 0) {
        const int64_t cNFfShexp = cNFfExp * cNExpertShared;
        vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexp}, 0);
        vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, 0);
      }
    } else {
      // Dense layers
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_afmoe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_afmoe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // MuP scaling: embeddings * sqrt(hidden_size)
  // mup_enabled = true, hidden_size = 1024, scale = 32.0
  vInpL = ggml_scale(vCtx0, vInpL, sqrtf(float(cNEmbd)));
  mCb(vInpL, "inp_embd_scaled", -1);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();
  auto *vInpAttn = mBuildAttnInpKvIswa();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    ggml_tensor *vInpSA = vInpL;

    // This overlaps with SWA layers in current models, so get_rope_freq_base/scale may be superfluous
    const bool cUseRope = vHparams.vNNoRopeLayerStep > 0 && (vIl + 1) % vHparams.vNNoRopeLayerStep != 0;

    // dual attention normalization (pre)
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      ggml_tensor *vAttnInp = vCur; // save input for gate computation

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      // compute gate from input
      ggml_tensor *vGate = mBuildLoraMm(pModel.lLayers[vIl].vWqkvGate, vAttnInp);
      mCb(vGate, "attn_gate_proj", vIl);

      // Q/K normalization
      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      if (cUseRope) {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        mCb(vQcur, "Qcur_rope", vIl);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        mCb(vKcur, "Kcur_rope", vIl);
      }

      vCur = mBuildAttn(vInpAttn, NULL, NULL, NULL, // wo will be applied after gating
                        vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, vIl);
      mCb(vCur, "attn_out", vIl);

      // attention gating: attn_out * sigmoid(gate) BEFORE o_proj
      vGate = ggml_sigmoid(vCtx0, vGate);
      mCb(vGate, "attn_gate_sig", vIl);
      vCur = ggml_mul(vCtx0, vCur, vGate);
      mCb(vCur, "attn_gated", vIl);

      // now apply output projection
      vCur = mBuildLoraMm(pModel.lLayers[vIl].vWo, vCur, pModel.lLayers[vIl].vWoS);
      mCb(vCur, "attn_o_proj", vIl);
    }

    // dual attention normalization (post)
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnPostNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_post_norm", vIl);

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // dual ffn normalization (pre)
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // MoE or dense FFN
    if ((uint32_t)vIl >= vHparams.vNLayerDenseLead) {
      // MoE layer with sigmoid routing, normalization, and scaling
      ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
                                          vHparams.vExpertWeightsNorm,  // norm_w (route_norm=True)
                                          vHparams.vExpertWeightsScale, // w_scale (route_scale=2.826)
                                          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // shared expert
      if (vHparams.vNExpertShared > 0) {
        ggml_tensor *vFfnShexp =
            mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                      pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
        mCb(vCur, "ffn_out", vIl);
      } else {
        vCur = vMoeOut;
      }
    } else {
      // dense layer
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    }

    // dual ffn normalization (post)
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vFfnPostNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_post_norm", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }

  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);

  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
