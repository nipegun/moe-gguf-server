#include "models.h"

void mgs_model_arctic::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  if (vHparams.vNExpert == 128) {
    switch (vHparams.mNLayer()) {
    case 35:
      vType = cLlmType10b128x3Point66B;
      break;
    default:
      vType = cLlmTypeUnknown;
    }
  } else {
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_arctic::mLoadArchTensors(mgs_model_loader &) {
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

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNEmbd}, 0);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNEmbd, cNEmbd}, 0);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnNormExps = mCreateTensor(cTn(cLlmTensorFfnNormExps, "weight", vI), {cNEmbd}, 0);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, false);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_arctic::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_arctic::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
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

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                        nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // feed-forward network
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                     pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
    mCb(vCur, "ffn_out", vIl);

    ggml_tensor *vFfnOut = ggml_add(vCtx0, vCur, vFfnInp);
    mCb(vFfnOut, "ffn_out", vIl);

    // MoE
    vCur = mBuildNorm(vInpSA, pModel.lLayers[vIl].vFfnNormExps, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm_exps", vIl);

    vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                        pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert,
                        cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax,
                        vIl);
    mCb(vCur, "ffn_moe_out", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnOut);
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

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
