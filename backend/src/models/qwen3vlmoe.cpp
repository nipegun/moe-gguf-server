#include "models.h"

void mgs_model_qwen3vlmoe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvNumDeepstackLayers, vHparams.vNDeepstackLayers, false);
  pMl.mGetKeyOrArr(cLlmKvRopeDimensionSections, vHparams.aRopeSections, 4, true);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  switch (vHparams.mNLayer()) {
  case 48:
    vType = cLlmType30bA3b;
    break;
  case 94:
    vType = cLlmType235bA22b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_qwen3vlmoe::mLoadArchTensors(mgs_model_loader &) {
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

    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);

    if (cNExpert == 0) {
      throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
    }
    if (cNExpertUsed == 0) {
      throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
    }

    // MoE branch
    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;

    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_qwen3vlmoe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_qwen3vlmoe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const size_t cNDeepstackLayers = vHparams.vNDeepstackLayers;

  const int64_t cNEmbd = vHparams.vNEmbd;
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self_attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);

      vQcur = ggml_rope_multi(vCtx0, vQcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase,
                             cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      vKcur = ggml_rope_multi(vCtx0, vKcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase,
                             cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

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

    // MoE branch
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                        pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr,
                                        cNExpert, cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale,
                                        cMgsExpertGatingFuncTypeSoftmax, vIl);
    mCb(vMoeOut, "ffn_moe_out", vIl);
    vCur = vMoeOut;

    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    if (vIl < (int)cNDeepstackLayers) {
      ggml_tensor *vDs = ggml_view_2d(vCtx0, vRes->vTInpEmbd, cNEmbd, cNTokens, vRes->vTInpEmbd->nb[1],
                                      (vIl + 1) * cNEmbd * sizeof(float));
      vCur = ggml_add(vCtx0, vCur, vDs);
      mCb(vCur, "deepstack_out", vIl);
    }

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
