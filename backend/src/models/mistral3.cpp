#include "models.h"

void mgs_model_mistral3::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvAttentionTemperatureScale, vHparams.vFAttnTempScale, false);

  pMl.mGetKey(cLlmKvRopeScalingYarnBetaFast, vHparams.vYarnBetaFast, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnBetaSlow, vHparams.vYarnBetaSlow, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnLogMul, vHparams.vRopeYarnLogMul, false);

  vHparams.vFAttnTempOffset = 0.0f;

  // TODO: maybe add n_attn_temp_floor_scale as a separate KV?
  if (vHparams.vFAttnTempScale != 0.0f) {
    vHparams.vNAttnTempFloorScale = vHparams.vNCtxOrigYarn;
    if (vHparams.vNAttnTempFloorScale == 0) {
      throw std::runtime_error("invalid n_ctx_orig_yarn for attention temperature scaling");
    }
  }

  switch (vHparams.mNLayer()) {
  case 26:
    vType = cLlmType3b;
    break;
  case 34:
    vType = cLlmType8b;
    break;
  case 40:
    vType = cLlmType14b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_mistral3::mLoadArchTensors(mgs_model_loader &) {
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

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    // optional bias tensors
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (vHparams.vRopeScalingTypeTrain == MGS_ROPE_SCALING_TYPE_LONGROPE) {
      vLayer.vRopeLong = mCreateTensor(cTn(cLlmTensorRopeFactorsLong, "weight", vI), {cNRot / 2},
                                       cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
      vLayer.vRopeShort = mCreateTensor(cTn(cLlmTensorRopeFactorsShort, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
    } else {
      vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
    }

    if (cNExpert == 0) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);

      // optional MLP bias
      vLayer.vFfnGateB = mCreateTensor(cTn(cLlmTensorFfnGate, "bias", vI), {cNFf}, cTensorNotRequired);
      vLayer.vFfnDownB = mCreateTensor(cTn(cLlmTensorFfnDown, "bias", vI), {cNEmbd}, cTensorNotRequired);
      vLayer.vFfnUpB = mCreateTensor(cTn(cLlmTensorFfnUp, "bias", vI), {cNFf}, cTensorNotRequired);
    } else {
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, cTensorNotRequired);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);

      // For Granite MoE Shared
      if (vHparams.vNFfShexp > 0) {
        vLayer.vFfnGateShexp =
            mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnDownShexp =
            mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {vHparams.vNFfShexp, cNEmbd}, 0);
      }
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_mistral3::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_mistral3::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  // (optional) temperature tuning
  ggml_tensor *vInpAttnScale = nullptr;
  if (vHparams.vFAttnTempScale != 0.0f) {
    vInpAttnScale = mBuildInpAttnScale();
  }

  auto *vInpAttn = mBuildAttnInpKv();

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      // rope freq factors; may return nullptr for models without them
      ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, vIl);

      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      if (vInpAttnScale) {
        // apply attention temperature scaling
        vQcur = ggml_mul(vCtx0, vQcur, vInpAttnScale);
        mCb(vQcur, "Qcur_attn_temp_scaled", vIl);
      }

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // feed-forward network (non-MoE)
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {

      vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "ffn_norm", vIl);

      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, pModel.lLayers[vIl].vFfnUpS,
                       pModel.lLayers[vIl].vFfnGate, pModel.lLayers[vIl].vFfnGateB, pModel.lLayers[vIl].vFfnGateS,
                       pModel.lLayers[vIl].vFfnDown, pModel.lLayers[vIl].vFfnDownB, pModel.lLayers[vIl].vFfnDownS, NULL,
                       cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "ffn_norm", vIl);

      vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert,
                          cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale,
                          cMgsExpertGatingFuncTypeSoftmax, vIl, nullptr, nullptr, pModel.lLayers[vIl].vFfnUpExpsS,
                          pModel.lLayers[vIl].vFfnGateExpsS, pModel.lLayers[vIl].vFfnDownExpsS);
      mCb(vCur, "ffn_moe_out", vIl);
    }
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

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
