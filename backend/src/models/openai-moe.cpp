#include "models.h"

void mgs_model_openai_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);

  vHparams.vSwaType = cMgsSwaTypeStandard;
  uint32_t vSwaPeriod = 2;
  pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
  vHparams.mSetSwaPattern(vSwaPeriod);

  vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
  vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;
  pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);

  switch (vHparams.mNLayer()) {
  case 24:
    vType = cLlmType20b;
    break;
  case 36:
    vType = cLlmType120b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_openai_moe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNFfExp = vHparams.vNFfExp;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNHead * cNRot, cNHeadKv * cNRot, cNHeadKv * cNRot, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNHead * cNRot, cNEmbd}, 0);

    vLayer.vAttnSinks = mCreateTensor(cTn(cLlmTensorAttnSinks, "weight", vI), {cNHead}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInpB = mCreateTensor(cTn(cLlmTensorFfnGateInp, "bias", vI), {cNExpert}, 0);
    vLayer.vFfnGateExpsB = mCreateTensor(cTn(cLlmTensorFfnGateExps, "bias", vI), {cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExpsB = mCreateTensor(cTn(cLlmTensorFfnDownExps, "bias", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExpsB = mCreateTensor(cTn(cLlmTensorFfnUpExps, "bias", vI), {cNFfExp, cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_openai_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_openai_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKvIswa();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vRes->lTLayerInp[vIl] = vInpL;

    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNRot, cNHead, cNHeadKv, vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, pModel.lLayers[vIl].vAttnSinks, nullptr, 1.0f / sqrtf(float(cNRot)), vIl);

      mCb(vCur, "attn_out", vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds && vCparams.vEmbeddingsNextnMasked) {
      // skip computing output for unused tokens
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = vFfnInp;
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnPostNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_post_norm", vIl);

    // MoE branch
    vCur = mBuildMoeFfn(
        vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnGateInpB, pModel.lLayers[vIl].vFfnUpExps,
        pModel.lLayers[vIl].vFfnUpExpsB, pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnGateExpsB,
        pModel.lLayers[vIl].vFfnDownExps, pModel.lLayers[vIl].vFfnDownExpsB, nullptr, cNExpert, cNExpertUsed,
        cLlmFfnSwigluOaiMoe, false, vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmaxWeight, vIl);
    mCb(vCur, "ffn_moe_out", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  vRes->vTHNextn = vCur;

  if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
