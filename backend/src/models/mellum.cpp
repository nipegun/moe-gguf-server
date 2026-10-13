#include "models.h"

void mgs_model_mellum::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa, false);

  if (vHparams.vNSwa > 0) {
    vHparams.vSwaType = cMgsSwaTypeStandard;

    uint32_t vSwaPeriod = 4;
    const auto cRes = pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
    if (cRes) {
      vHparams.mSetSwaPattern(vSwaPeriod);
    } else {
      pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl, vHparams.mNLayer());
    }

    vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
    vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;

    pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);
  } else {
    vHparams.vSwaType = cMgsSwaTypeNone;
  }

  switch (vHparams.mNLayer()) {
  case 28:
    vType = cLlmType12bA2Point5B;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_mellum::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);

    if (cNExpert == 0) {
      throw std::runtime_error("n_expert must be > 0 for Mellum");
    }
    if (cNExpertUsed == 0) {
      throw std::runtime_error("n_expert_used must be > 0 for Mellum");
    }

    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;

    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_mellum::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (vHparams.vSwaType == cMgsSwaTypeStandard) {
    return std::make_unique<graph<true>>(*this, pParams);
  }
  return std::make_unique<graph<false>>(*this, pParams);
}

template <bool iswa>
mgs_model_mellum::graph<iswa>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
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
    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self_attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, nullptr, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);

      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, nullptr, cLlmNormRms, vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      const bool cIsSwa = vHparams.mIsSwa(vIl);

      if (cIsSwa) {
        // For sliding window layers, use regular rope with no yarn rope scaling.
        // This is achieved here by setting freq_scale and attn_factor to 1.
        // We also set ext_factor to 0 to avoid a few unnecessary computations.
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, 1.0, 0.0, 1.0,
                             cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, 1.0, 0.0, 1.0,
                             cBetaFast, cBetaSlow);
      } else {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // MoE
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    ggml_tensor *vMoeOut = mBuildMoeFfn(
        vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
        pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
        vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, vIl, nullptr, nullptr,
        pModel.lLayers[vIl].vFfnUpExpsS, pModel.lLayers[vIl].vFfnGateExpsS, pModel.lLayers[vIl].vFfnDownExpsS);
    mCb(vMoeOut, "ffn_moe_out", vIl);
    vCur = vMoeOut;

    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    mCb(vCur, "ffn_out", vIl);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

template struct mgs_model_mellum::graph<false>;
template struct mgs_model_mellum::graph<true>;
