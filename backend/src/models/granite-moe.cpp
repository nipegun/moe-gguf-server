#include "models.h"

void mgs_model_granite_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLogitScale, vHparams.vFLogitScale);
  pMl.mGetKey(cLlmKvResidualScale, vHparams.vFResidualScale, false);
  pMl.mGetKey(cLlmKvEmbeddingScale, vHparams.vFEmbeddingScale, false);
  pMl.mGetKey(cLlmKvAttentionScale, vHparams.vFAttentionScale, false);

  // Granite uses rope_finetuned as a switch for rope, so default to true
  bool vRopeFinetuned = true;
  pMl.mGetKey(cLlmKvRopeScalingFinetuned, vRopeFinetuned, false);
  vHparams.vRopeFinetuned = vRopeFinetuned;

  switch (vHparams.mNLayer()) {
  case 32:
    vType = cLlmType3b;
    break;
  case 40:
    vType = cLlmType3b;
    break;
  // Add additional layer/vocab/etc checks here for other model sizes
  default:
    vType = cLlmTypeUnknown;
  }

  // For Granite MoE Shared
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, /* required */ false);
}

void mgs_model_granite_moe::mLoadArchTensors(mgs_model_loader &) {
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

std::unique_ptr<llm_graph_context> mgs_model_granite_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_granite_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - built only if rope enabled
  ggml_tensor *vInpPos = nullptr;
  if (vHparams.vRopeFinetuned) {
    vInpPos = mBuildInpPos();
  }
  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {

    // Granite Vision 4.1 deepstack: inject the projector stream that
    // targets decoder layer `il` before the decoder runs.
    // NOTE: skip the first deepstack layer since that's inpL
    const auto &vDeepstackEmbIdx = vHparams.aDeepstackMappingArr[vIl];
    if (vIl > 0 && vDeepstackEmbIdx >= 0) {
      ggml_tensor *vDs = ggml_view_2d(vCtx0, vRes->vTInpEmbd, cNEmbd, cNTokens, vRes->vTInpEmbd->nb[1],
                                      vDeepstackEmbIdx * cNEmbd * sizeof(float));
      vInpL = ggml_add(vCtx0, vInpL, vDs);
      mCb(vInpL, "deepstack_in", vIl);
    }

    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    vCur = mBuildAttentionLayer(vCur, vInpPos, vInpAttn, pModel, cNEmbdHead, vIl);

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    // ffn
    vCur = mBuildLayerFfn(vCur, vInpSA, pModel, vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  // For Granite architectures - scale logits
  vCur = ggml_scale(vCtx0, vCur, 1.0f / vHparams.vFLogitScale);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

ggml_tensor *mgs_model_granite_moe::graph::mBuildAttentionLayer(ggml_tensor *pCur, ggml_tensor *pInpPos,
                                                                llm_graph_input_attn_kv *pInpAttn,
                                                                const mgs_model &pModel, const int64_t pNEmbdHead,
                                                                const int pIl) {

  auto [vQcur, vKcur, vVcur] =
      mBuildQkv(pModel.lLayers[pIl], pCur, pNEmbdHead, vHparams.mNHead(pIl), vHparams.mNHeadKv(pIl), pIl);

  const bool cUseRope = vHparams.vRopeFinetuned;
  if (cUseRope) {
    ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, pIl);
    vQcur = ggml_rope_ext(vCtx0, vQcur, pInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

    vKcur = ggml_rope_ext(vCtx0, vKcur, pInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  }

  mCb(vQcur, "Qcur", pIl);
  mCb(vKcur, "Kcur", pIl);
  mCb(vVcur, "Vcur", pIl);

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(pNEmbdHead)) : vHparams.vFAttentionScale;
  pCur = mBuildAttn(pInpAttn, pModel.lLayers[pIl].vWo, pModel.lLayers[pIl].vWoB, pModel.lLayers[pIl].vWoS, vQcur, vKcur,
                    vVcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  mCb(pCur, "attn_out", pIl);
  return pCur;
}

ggml_tensor *mgs_model_granite_moe::graph::mBuildLayerFfn(ggml_tensor *pCur, ggml_tensor *pInpSA,
                                                          const mgs_model &pModel, const int pIl) {

  // For Granite architectures - scale residual
  if (vHparams.vFResidualScale) {
    pCur = ggml_scale(vCtx0, pCur, vHparams.vFResidualScale);
  }
  ggml_tensor *vFfnInp = ggml_add(vCtx0, pCur, pInpSA);
  mCb(vFfnInp, "ffn_inp", pIl);

  // feed-forward network (non-MoE)
  if (pModel.lLayers[pIl].vFfnGateInp == nullptr) {

    pCur = mBuildNorm(vFfnInp, pModel.lLayers[pIl].vFfnNorm, NULL, cLlmNormRms, pIl);
    mCb(pCur, "ffn_norm", pIl);

    pCur = mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUp, pModel.lLayers[pIl].vFfnUpB, NULL, pModel.lLayers[pIl].vFfnGate,
                     pModel.lLayers[pIl].vFfnGateB, NULL, pModel.lLayers[pIl].vFfnDown, pModel.lLayers[pIl].vFfnDownB,
                     NULL, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
    mCb(pCur, "ffn_out", pIl);

  } else {
    // MoE branch
    pCur = mBuildNorm(vFfnInp, pModel.lLayers[pIl].vFfnNorm, NULL, cLlmNormRms, pIl);
    mCb(pCur, "ffn_norm", pIl);

    ggml_tensor *vMoeOut = mBuildMoeFfn(pCur, pModel.lLayers[pIl].vFfnGateInp, pModel.lLayers[pIl].vFfnUpExps,
                                        pModel.lLayers[pIl].vFfnGateExps, pModel.lLayers[pIl].vFfnDownExps, nullptr,
                                        cNExpert, cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale,
                                        cMgsExpertGatingFuncTypeSoftmax, pIl);
    mCb(vMoeOut, "ffn_moe_out", pIl);

    // For Granite MoE Shared
    if (vHparams.vNFfShexp > 0) {
      ggml_tensor *vFfnShexp =
          mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[pIl].vFfnGateShexp, NULL, NULL,
                    pModel.lLayers[pIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
      mCb(vFfnShexp, "ffn_shexp", pIl);

      pCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(pCur, "ffn_out", pIl);
    } else {
      pCur = vMoeOut;
    }
  }

  // For Granite architectures - scale residual
  if (vHparams.vFResidualScale) {
    pCur = ggml_scale(vCtx0, pCur, vHparams.vFResidualScale);
  }
  pCur = ggml_add(vCtx0, pCur, vFfnInp);
  mCb(pCur, "ffn_out", pIl);

  pCur = mBuildCvec(pCur, pIl);
  mCb(pCur, "l_out", pIl);

  return pCur;
}
