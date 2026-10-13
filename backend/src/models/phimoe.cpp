#include "models.h"

void mgs_model_phimoe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  switch (vHparams.mNLayer()) {
  case 32:
    vType = cLlmType16x3Point8B;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_phimoe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNEmbdHead = cNEmbd / cNHead;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutputNormB = mCreateTensor(cTn(cLlmTensorOutputNorm, "bias"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);
  vOutputB = mCreateTensor(cTn(cLlmTensorOutput, "bias"), {cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnNormB = mCreateTensor(cTn(cLlmTensorAttnNorm, "bias", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vFfnNormB = mCreateTensor(cTn(cLlmTensorFfnNorm, "bias", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
    vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);

    vLayer.vRopeLong = mCreateTensor(cTn(cLlmTensorRopeFactorsLong, "weight", vI), {cNEmbdHead / 2},
                                     cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
    vLayer.vRopeShort = mCreateTensor(cTn(cLlmTensorRopeFactorsShort, "weight", vI), {cNEmbdHead / 2},
                                      cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
  }
}

std::unique_ptr<llm_graph_context> mgs_model_phimoe::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (vHparams.vSwaType != cMgsSwaTypeNone) {
    return std::make_unique<graph<true>>(*this, pParams);
  } else {
    return std::make_unique<graph<false>>(*this, pParams);
  }
}

template <bool iswa>
mgs_model_phimoe::graph<iswa>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

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
    auto *vResidual = vInpL;

    // self-attention
    {
      // rope freq factors for 128k context
      ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, vIl);

      ggml_tensor *vAttnNormOutput =
          mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, pModel.lLayers[vIl].vAttnNormB, cLlmNormRms, vIl);
      mCb(vAttnNormOutput, "attn_norm", vIl);

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vAttnNormOutput, cNEmbdHead, cNHead, cNHeadKv, vIl);
      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vQcur = ggml_scale(vCtx0, vQcur, 1.0f / sqrtf(float(cNEmbdHead)));
      mCb(vQcur, "Qcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, 1.0f, vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vResidual = ggml_get_rows(vCtx0, vResidual, vInpOutIds);
    }
    vCur = ggml_add(vCtx0, vCur, vResidual);
    vResidual = vCur;

    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vFfnNorm, pModel.lLayers[vIl].vFfnNormB, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // feed-forward network
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, NULL, NULL, NULL, pModel.lLayers[vIl].vFfnDown,
                       NULL, NULL, NULL, cLlmFfnSwiglu, cLlmFfnSeq, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert,
                          cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale,
                          cMgsExpertGatingFuncTypeSoftmax, vIl);
      mCb(vCur, "ffn_moe_out", vIl);
    }
    vCur = ggml_add(vCtx0, vResidual, vCur);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = mBuildNorm(vInpL, pModel.vOutputNorm, pModel.vOutputNormB, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  if (pModel.vOutputB != nullptr) {
    mCb(vCur, "result_output_no_bias", -1);
    vCur = ggml_add(vCtx0, vCur, pModel.vOutputB);
  }
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// Explicit template instantiations
template struct mgs_model_phimoe::graph<false>;
template struct mgs_model_phimoe::graph<true>;
