#include "models.h"

void mgs_model_exaone_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  vHparams.vSwaType = cMgsSwaTypeStandard;
  vHparams.vNSwa = 128;
  uint32_t vSwaPeriod = 4;
  pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vSwaPeriod, false);
  vHparams.mSetSwaPattern(vSwaPeriod);
  vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
  vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;

  pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared, false);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);

  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_impl");

  switch (vHparams.mNLayer()) {
  case 32:
    vType = cLlmType30bA3b;
    break;
  case 48:
    vType = cLlmType235bA22b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_exaone_moe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNFfExp = vHparams.vNFfExp;
  const int64_t cNFfShexp = vHparams.vNFfShexp > 0 ? vHparams.vNFfShexp : cNFfExp;
  const int64_t cHeadDim = vHparams.mNEmbdHeadK();
  const int64_t cNQoDim = cNHead * cHeadDim;
  const int64_t cNKvDim = cNHeadKv * cHeadDim;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayerAll; ++vI) {
    int vFlags = 0;
    if (vI >= cNLayer) {
      // skip all tensors in the NextN layers
      vFlags |= cTensorSkip;
    }

    auto &vLayer = lLayers[vI];
    mCreateTensorQkv(vLayer, vI, cNEmbd, cNQoDim, cNKvDim, cNKvDim, vFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNQoDim, cNEmbd}, vFlags);

    vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", vI), {cNRot / 2},
                                      cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0) | vFlags);

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, vFlags);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, vFlags);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, vFlags);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, vFlags);

    // dense layers for first n_layer_dense_lead layers or nextn_predict_layers layers at the end
    if (vI < (int)vHparams.vNLayerDenseLead || (vI >= cNLayer)) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, vFlags);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, vFlags);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, vFlags);
    } else {
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, vFlags);
      vLayer.vFfnExpProbsB =
          mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cTensorNotRequired | vFlags);

      if (cNExpert == 0) {
        throw std::runtime_error("n_expert must be > 0");
      }
      if (cNExpertUsed == 0) {
        throw std::runtime_error("n_expert_used must be > 0");
      }

      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, vFlags);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, vFlags);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, vFlags);

      vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexp}, vFlags);
      vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, vFlags);
      vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, vFlags);
    }

    // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
    if (vI >= cNLayer) {
      vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, vFlags);
      vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, vFlags);
      vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, vFlags);

      vLayer.vNextn.vSharedHeadNorm =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, vFlags | cTensorNotRequired);
      vLayer.vNextn.vEmbedTokens =
          mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab}, vFlags | cTensorNotRequired);
      vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI), {cNEmbd, cNVocab},
                                                    vFlags | cTensorNotRequired);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_exaone_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_exaone_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadK();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadV());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttnIswa = mBuildAttnInpKvIswa();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // use RoPE for SWA layers
    const bool cIsLocalLayer = vHparams.mIsSwa(vIl);

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, vIl);

      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      if (cIsLocalLayer) {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }
      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttnIswa, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur,
                        nullptr, nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
      mCb(vCur, "attn_out", vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // norm
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // feed-forward network
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      // dense branch
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
                                          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                                          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // FFN shared expert
      {
        ggml_tensor *vFfnShexp =
            mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                      pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
        mCb(vCur, "ffn_out", vIl);
      }
    }

    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  // final norm
  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
