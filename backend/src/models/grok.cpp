#include "models.h"

void mgs_model_grok::mLoadArchHparams(mgs_model_loader &pMl) {
  // grok-1 conversions omit these keys
  vHparams.vYarnBetaFast = 8.0f;
  // no final_logit_softcapping in grok-1
  vHparams.vFFinalLogitSoftcapping = 0.0f;

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvLogitScale, vHparams.vFLogitScale);
  pMl.mGetKey(cLlmKvEmbeddingScale, vHparams.vFEmbeddingScale);
  pMl.mGetKey(cLlmKvAttentionOutputScale, vHparams.vFAttnOutScale);
  pMl.mGetKey(cLlmKvAttnLogitSoftcapping, vHparams.vFAttnLogitSoftcapping);
  pMl.mGetKey(cLlmKvRouterLogitSoftcapping, vHparams.vFRouterLogitSoftcapping, false);
  pMl.mGetKey(cLlmKvFinalLogitSoftcapping, vHparams.vFFinalLogitSoftcapping, false);

  pMl.mGetKey(cLlmKvAttentionTemperatureLength, vHparams.vAttnTempLength, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnExtFactor, vHparams.vYarnExtFactor, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnAttnFactor, vHparams.vYarnAttnFactor, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnBetaFast, vHparams.vYarnBetaFast, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnBetaSlow, vHparams.vYarnBetaSlow, false);

  switch (vHparams.mNLayer()) {
  case 64:
    vType = cLlmType314b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_grok::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  if (cNExpert == 0) {
    throw std::runtime_error(mArchName() + " model cannot have zero experts");
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf /* / n_expert_used*/; // grok-1 n_ff_exp == n_ff
  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);

    vLayer.vAttnOutNorm = mCreateTensor(cTn(cLlmTensorAttnOutNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, cTensorNotRequired);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, cTensorNotRequired);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, cTensorNotRequired);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnGateExps =
        mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cTensorNotRequired);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

    vLayer.vFfnPostNorm = mCreateTensor(cTn(cLlmTensorLayerOutNorm, "weight", vI), {cNEmbd}, cTensorNotRequired);
    if (!vLayer.vFfnPostNorm) {
      vLayer.vFfnPostNorm = mCreateTensor(cTn(cLlmTensorFfnPostNorm, "weight", vI), {cNEmbd}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_grok::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_grok::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, 1.0f, vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnOutNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_out_norm", vIl);

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // feed-forward network
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // MoE branch
    ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                        pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr,
                                        cNExpert, cNExpertUsed, cLlmFfnGelu, true, vHparams.vExpertWeightsScale,
                                        cMgsExpertGatingFuncTypeSoftmax, vIl);
    mCb(vMoeOut, "ffn_moe_out", vIl);

    if (pModel.lLayers[vIl].vFfnUp) {
      ggml_tensor *vFfnOut =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                    pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnGelu, cLlmFfnPar, vIl);
      mCb(vFfnOut, "ffn_out", vIl);

      vCur = ggml_scale(vCtx0, ggml_add(vCtx0, vFfnOut, vMoeOut), std::sqrt(2) / 2);
      mCb(vCur, "ffn_out", vIl);
    } else {
      vCur = vMoeOut;
    }
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vFfnPostNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_post_norm", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    mCb(vCur, "ffn_out", vIl);

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

  vCur = ggml_scale(vCtx0, vCur, vHparams.vFLogitScale);

  // final logit soft-capping
  if (vHparams.vFFinalLogitSoftcapping) {
    vCur = ggml_scale(vCtx0, vCur, 1.0f / vHparams.vFFinalLogitSoftcapping);
    vCur = ggml_tanh(vCtx0, vCur);
    vCur = ggml_scale(vCtx0, vCur, vHparams.vFFinalLogitSoftcapping);
  }
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
