#include "models.h"

// Flint (FlintForCausalLM): leading dense layers, then MoE layers with sigmoid routing and always-active shared
// experts. Sliding-window layers use RoPE and the full-attention layers have no positional encoding (NoPE); every layer
// has attention sinks and Q/K RMS norms.

void mgs_model_flint::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);

  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp);
  pMl.mGetArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl);
  pMl.mGetArr(cLlmKvAttentionRopePattern, vHparams.aRopePattern);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  vHparams.vSwaType = cMgsSwaTypeStandard;

  // only the sliding-window layers use RoPE, with the frequencies of the model, so the graph and the K-shift take
  // them from the same place
  vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
  vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;

  vType = cLlmTypeUnknown;
}

void mgs_model_flint::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNFfExp = vHparams.vNFfExp;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vAttnSinks = mCreateTensor(cTn(cLlmTensorAttnSinks, "weight", vI), {cNHead}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (vI < (int)vHparams.vNLayerDenseLead) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    } else {
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB =
          mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cTensorNotRequired);

      if (cNExpert == 0) {
        throw std::runtime_error("n_expert must be > 0 for FLINT");
      }
      if (cNExpertUsed == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for FLINT");
      }

      // MoE branch
      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

      // Shared expert branch
      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, (int64_t)vHparams.vNFfShexp}, 0);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {(int64_t)vHparams.vNFfShexp, cNEmbd}, 0);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, (int64_t)vHparams.vNFfShexp}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_flint::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_flint::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKvIswa();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self_attention
    {
      // compute Q and K and RoPE them (only in the layers that use RoPE)
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, nullptr, cLlmNormRms, vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, nullptr, cLlmNormRms, vIl);
      if (vHparams.mHasRope(vIl)) {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                              cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                              cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }
      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, pModel.lLayers[vIl].vAttnSinks, nullptr,
                        1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, pModel.lLayers[vIl].vFfnUpS,
                       pModel.lLayers[vIl].vFfnGate, NULL, pModel.lLayers[vIl].vFfnGateS, pModel.lLayers[vIl].vFfnDown,
                       NULL, pModel.lLayers[vIl].vFfnDownS, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      ggml_tensor *vMoeOut = mBuildMoeFfn(
          vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
          pModel.lLayers[vIl].vFfnDownExps, pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl, nullptr, nullptr,
          pModel.lLayers[vIl].vFfnUpExpsS, pModel.lLayers[vIl].vFfnGateExpsS, pModel.lLayers[vIl].vFfnDownExpsS);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // Shared expert branch
      ggml_tensor *vFfnShexp =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, pModel.lLayers[vIl].vFfnUpShexpS,
                    pModel.lLayers[vIl].vFfnGateShexp, NULL, pModel.lLayers[vIl].vFfnGateShexpS,
                    pModel.lLayers[vIl].vFfnDownShexp, NULL, pModel.lLayers[vIl].vFfnDownShexpS, NULL, cLlmFfnSilu,
                    cLlmFfnPar, vIl);
      mCb(vFfnShexp, "ffn_shexp", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(vCur, "ffn_out", vIl);
    }
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
