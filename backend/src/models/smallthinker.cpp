#include "models.h"

void mgs_model_smallthinker::mLoadArchHparams(mgs_model_loader &pMl) {
  const bool cFoundSwa = pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa, false);

  if (cFoundSwa && vHparams.vNSwa > 0) {
    vHparams.vSwaType = cMgsSwaTypeStandard;
    vHparams.vNSwa = 4096;
    uint32_t vSwaPeriod = 4;
    pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
    vHparams.mSetSwaPattern(vSwaPeriod, true);

    vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
    vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;
    pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);
  } else {
    vHparams.vSwaType = cMgsSwaTypeNone;
    vHparams.vNNoRopeLayerStep = vHparams.mNLayer();
  }

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);

  switch (vHparams.mNLayer()) {
  case 32:
    vType = cLlmType4b;
    break;
  case 52:
    vType = cLlmType20b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_smallthinker::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    GGML_ASSERT(cNExpert > 0 && "n_expert must be > 0 for SMALLTHINKER");
    GGML_ASSERT(cNExpertUsed > 0 && "n_expert_used must be > 0 for SMALLTHINKER");

    // MoE branch
    const int64_t cNFfExp = vHparams.vNFfExp;
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_smallthinker::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (vHparams.vSwaType == cMgsSwaTypeStandard) {
    return std::make_unique<graph<true>>(*this, pParams);
  } else {
    return std::make_unique<graph<false>>(*this, pParams);
  }
}

template <bool iswa>
mgs_model_smallthinker::graph<iswa>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
  inp_attn_type *vInpAttn = nullptr;

  if constexpr (iswa) {
    vInpAttn = mBuildAttnInpKvIswa();
  } else {
    vInpAttn = mBuildAttnInpKv();
  }
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    ggml_tensor *vInpSA = vInpL;

    // This overlaps with SWA layers in current models, so get_rope_freq_base/scale may be superfluous
    const bool cUseRope = vHparams.vNNoRopeLayerStep == cNLayer || vIl % vHparams.vNNoRopeLayerStep != 0;

    ggml_tensor *vProbs = mBuildLoraMm(pModel.lLayers[vIl].vFfnGateInp, vInpL); // [n_expert, n_tokens]
    mCb(vProbs, "ffn_moe_logits", vIl);

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self_attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      if (cUseRope) {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }
      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
      vProbs = ggml_get_rows(vCtx0, vProbs, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // MoE branch
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    ggml_tensor *vFfnOut =
        mBuildMoeFfn(vCur, nullptr, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
                     pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert, cNExpertUsed, cLlmFfnRelu, true,
                     vHparams.vExpertWeightsScale, static_cast<mgs_expert_gating_func_type>(vHparams.vExpertGatingFunc),
                     vIl, vProbs);

    mCb(vFfnOut, "ffn_out", vIl);
    vCur = vFfnOut;

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

// Explicit template instantiations
template struct mgs_model_smallthinker::graph<false>;
template struct mgs_model_smallthinker::graph<true>;
