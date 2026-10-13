#include "models.h"

void mgs_model_grovemoe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertChunkFeedForwardLength, vHparams.vNFfChexp, false);
  pMl.mGetKey(cLlmKvExpertGroupScale, vHparams.vExpertGroupScale);
  pMl.mGetKey(cLlmKvExpertsPerGroup, vHparams.vNGroupExperts);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  switch (vHparams.mNLayer()) {
  case 48:
    vType = cLlmType30bA3b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_grovemoe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  GGML_ASSERT(cNExpert > 0 && "n_expert must be > 0 for GROVEMOE");
  GGML_ASSERT(cNExpertUsed > 0 && "n_expert_used must be > 0 for GROVEMOE");
  GGML_ASSERT(vHparams.vNGroupExperts > 0 && "n_group_experts must be > 0 for GROVEMOE");

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);

    // MoE branch
    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;
    const int64_t cNFfChexp = vHparams.vNFfChexp ? vHparams.vNFfChexp : cNEmbdHeadK;
    const int64_t cNChunkExpert = cNExpert / vHparams.vNGroupExperts;

    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

    vLayer.vFfnGateChexps =
        mCreateTensor(cTn(cLlmTensorFfnGateChexps, "weight", vI), {cNEmbd, cNFfChexp, cNChunkExpert}, 0);
    vLayer.vFfnDownChexps =
        mCreateTensor(cTn(cLlmTensorFfnDownChexps, "weight", vI), {cNFfChexp, cNEmbd, cNChunkExpert}, 0);
    vLayer.vFfnUpChexps =
        mCreateTensor(cTn(cLlmTensorFfnUpChexps, "weight", vI), {cNEmbd, cNFfChexp, cNChunkExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_grovemoe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_grovemoe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  const int64_t cNChunkExpert = cNExpert / vHparams.vNGroupExperts;

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

    // self_attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

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

    ggml_tensor *vProbs = mBuildLoraMm(pModel.lLayers[vIl].vFfnGateInp, vCur); // [n_expert, n_tokens]
    mCb(vProbs, "ffn_moe_logits", vIl);

    ggml_tensor *vMoeOut =
        mBuildMoeFfn(vCur, nullptr, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
                     pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
                     vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, vIl, vProbs);
    mCb(vMoeOut, "ffn_moe_out", vIl);
    vCur = vMoeOut;

    // TODO: Only do the expert selection and weights once
    vMoeOut = mBuildMoeFfn(vCur, nullptr, pModel.lLayers[vIl].vFfnUpChexps, pModel.lLayers[vIl].vFfnGateChexps,
                           pModel.lLayers[vIl].vFfnDownChexps, nullptr, cNChunkExpert,
                           cNExpertUsed > cNChunkExpert ? cNChunkExpert : cNExpertUsed, cLlmFfnSilu, true,
                           vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, vIl, vProbs);
    mCb(vMoeOut, "ffn_adj_moe_out", vIl);

    vCur = ggml_add(vCtx0, vCur, ggml_scale(vCtx0, vMoeOut, vHparams.vExpertGroupScale));
    mCb(vCur, "ffn_final_moe_out", vIl);

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
