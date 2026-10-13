#include "models.h"

void mgs_model_minimax_m2::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);

  switch (vHparams.mNLayer()) {
  case 62:
    vType = cLlmType230bA10b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_minimax_m2::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK * cNHead}, 0);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdKGqa}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
    vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_minimax_m2::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_minimax_m2::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  // GGML_ASSERT(n_embd_head == n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  ggml_tensor *vInpPos = mBuildInpPos();
  auto vInpAttn = mBuildAttnInpKv();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vRes->lTLayerInp[vIl] = vInpL;

    ggml_tensor *vInpSA = vInpL;

    vCur = vInpL;

    // self_attention
    {
      vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "attn_norm", vIl);

      // compute Q and K and RoPE them
      ggml_tensor *vQcur = mBuildLoraMm(pModel.lLayers[vIl].vWq, vCur);
      mCb(vQcur, "Qcur", vIl);

      ggml_tensor *vKcur = mBuildLoraMm(pModel.lLayers[vIl].vWk, vCur);
      mCb(vKcur, "Kcur", vIl);

      ggml_tensor *vVcur = mBuildLoraMm(pModel.lLayers[vIl].vWv, vCur);
      mCb(vVcur, "Vcur", vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);

      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHead, cNHead, cNTokens);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                        nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
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

    vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                        pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                        pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
                        vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
    mCb(vCur, "ffn_moe_out", vIl);

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
