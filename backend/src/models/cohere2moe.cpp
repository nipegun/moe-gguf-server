#include "models.h"

void mgs_model_cohere2moe::mLoadArchHparams(mgs_model_loader &pMl) {
  const bool cFoundNorm = pMl.mGetKey(cLlmKvAttentionLayernormEps, vHparams.vFNormEps, false);
  const bool cFoundNormRms = pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps, false);
  if (!cFoundNorm && !cFoundNormRms) {
    throw std::runtime_error("missing Cohere2 MoE norm epsilon");
  }
  if (!cFoundNormRms) {
    vHparams.vFNormRmsEps = 0.0f;
  }

  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  pMl.mGetKey(cLlmKvLogitScale, vHparams.vFLogitScale);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);

  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer");

  if (vHparams.vExpertGatingFunc == cMgsExpertGatingFuncTypeNone) {
    vHparams.vExpertGatingFunc = cMgsExpertGatingFuncTypeSigmoid;
  }

  // per-layer array; a scalar period is rejected as "array key not found"
  vHparams.vSwaType = cMgsSwaTypeStandard;
  pMl.mGetArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl);

  vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
  vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;
  pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);

  switch (vHparams.mNLayer()) {
  case 49:
    vType = cLlmType30bA3b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_cohere2moe::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  // Trunk-only: the GGUF declares MTP layers in metadata but the actual MTP
  // tensors live in a separate file. Mark MTP tensors NOT_REQUIRED so the
  // trunk loads cleanly.
  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  if (cNExpert == 0) {
    throw std::runtime_error("n_expert must be > 0 for Cohere2Moe");
  }
  if (cNExpertUsed == 0) {
    throw std::runtime_error("n_expert_used must be > 0 for Cohere2Moe");
  }

  auto vLoadBlockTrunk = [&](int pI, int pFlags) {
    auto &vLayer = lLayers[pI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pI), {cNEmbd}, pFlags);

    mCreateTensorQkv(vLayer, pI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, pFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pI), {cNEmbdHeadK * cNHead, cNEmbd}, pFlags);

    if (static_cast<uint32_t>(pI) < vHparams.vNLayerDenseLead) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pI), {cNEmbd, cNFf}, pFlags);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pI), {cNFf, cNEmbd}, pFlags);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pI), {cNEmbd, cNFf}, pFlags);
    } else {
      const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf;

      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pI), {cNEmbd, cNExpert}, pFlags);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pI), {cNFfExp, cNEmbd, cNExpert}, pFlags);
      mCreateTensorGateUpExps(vLayer, pI, cNEmbd, cNFfExp, cNExpert, pFlags);

      if (vHparams.vNExpertShared > 0) {
        const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFfExp * vHparams.vNExpertShared;
        vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pI), {cNEmbd, cNFfShexp}, pFlags);
        vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pI), {cNFfShexp, cNEmbd}, pFlags);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pI), {cNEmbd, cNFfShexp}, pFlags);
      }
    }
  };

  auto vLoadBlockMtp = [&](int pI, int pFlags) {
    auto &vLayer = lLayers[pI];

    // MTP block looks like a full-attention Cohere2 MoE decoder block.
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pI), {cNEmbd}, pFlags);

    mCreateTensorQkv(vLayer, pI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, pFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pI), {cNEmbdHeadK * cNHead, cNEmbd}, pFlags);

    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf;

    // Routed experts
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pI), {cNEmbd, cNExpert}, pFlags);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pI), {cNFfExp, cNEmbd, cNExpert}, pFlags);
    mCreateTensorGateUpExps(vLayer, pI, cNEmbd, cNFfExp, cNExpert, pFlags);

    if (vHparams.vNExpertShared > 0) {
      const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFfExp * vHparams.vNExpertShared;

      // Shared experts
      vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pI), {cNEmbd, cNFfShexp}, pFlags);
      vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pI), {cNFfShexp, cNEmbd}, pFlags);
      vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pI), {cNEmbd, cNFfShexp}, pFlags);
    }

    // NextN-specific tensors that define the MTP block.
    vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", pI), {2 * cNEmbd, cNEmbd}, pFlags);
    vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", pI), {cNEmbd}, pFlags);
    vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", pI), {cNEmbd}, pFlags);
    vLayer.vNextn.vEmbedTokens =
        mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", pI), {cNEmbd, cNVocab}, cTensorNotRequired);
    vLayer.vNextn.vSharedHeadHead =
        mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", pI), {cNEmbd, cNVocab}, cTensorNotRequired);
    vLayer.vNextn.vSharedHeadNorm =
        mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", pI), {cNEmbd}, cTensorNotRequired);
  };

  for (int vI = 0; vI < cNLayer; ++vI) {
    vLoadBlockTrunk(vI, cTrunkFlags);
  }
  // MTP/NextN layers are loaded as extra decoder blocks.
  for (int vI = cNLayer; vI < cNLayerAll; ++vI) {
    vLoadBlockMtp(vI, vMtpFlags);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_cohere2moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_cohere2moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  const llm_norm_type cCohere2moeNormType = vHparams.vFNormRmsEps == 0.0f ? cLlmNorm : cLlmNormRms;
  const float cFLogitScale = vHparams.vFLogitScale;
  ggml_tensor *vCur;
  ggml_tensor *vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKvIswa();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const bool cIsSwa = vHparams.mIsSwa(vIl);
    // Dense-prefix full-attention layers use RoPE; later layers follow the SWA pattern.
    const bool cForceRope = static_cast<uint32_t>(vIl) < vHparams.vNLayerDenseLead;

    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cCohere2moeNormType, vIl);
    mCb(vCur, "attn_norm", vIl);

    ggml_tensor *vFfnInp = vCur;

    {
      const auto &vLayer = pModel.lLayers[vIl];

      auto [vQcur, vKcur, vVcur] = mBuildQkv(vLayer, vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      if (cIsSwa || cForceRope) {
        ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, vIl);

        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, vLayer.vWo, vLayer.vWoB, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr,
                        1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds && vCparams.vEmbeddingsNextnMasked) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpL = ggml_get_rows(vCtx0, vInpL, vInpOutIds);
      vFfnInp = ggml_get_rows(vCtx0, vFfnInp, vInpOutIds);
    }

    ggml_tensor *vAttnOut = vCur;

    const auto &vLayer = pModel.lLayers[vIl];

    if (vLayer.vFfnGateInp == nullptr) {
      vCur = mBuildFfn(vFfnInp, vLayer.vFfnUp, nullptr, vLayer.vFfnUpS, vLayer.vFfnGate, nullptr, vLayer.vFfnGateS,
                       vLayer.vFfnDown, nullptr, vLayer.vFfnDownS, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      vCur =
          mBuildMoeFfn(vFfnInp, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                       nullptr, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                       vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl,
                       nullptr, vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
      mCb(vCur, "ffn_moe_out", vIl);

      if (vLayer.vFfnUpShexp) {
        ggml_tensor *vFfnShexp = mBuildFfn(vFfnInp, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS,
                                           vLayer.vFfnGateShexp, nullptr, vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp,
                                           nullptr, vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vCur, vFfnShexp);
        vCur = ggml_scale(vCtx0, vCur, 0.5f);
        mCb(vCur, "ffn_out", vIl);
      }
    }

    vCur = ggml_add(vCtx0, vCur, vInpL);
    vCur = ggml_add(vCtx0, vCur, vAttnOut);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    vInpL = vCur;
  }

  vCur = vInpL;
  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cCohere2moeNormType, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur);

  if (cFLogitScale) {
    vCur = ggml_scale(vCtx0, vCur, cFLogitScale);
  }

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

mgs_model_cohere2moe::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "COHERE2MOE MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "COHERE2MOE MTP currently only supports a single MTP block");

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  const int cIl = vHparams.mNLayer();
  const auto &vLayer = pModel.lLayers[cIl];
  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");
  GGML_ASSERT(vLayer.vFfnGateInp && "MTP block missing ffn_gate_inp");

  const llm_norm_type cCohere2moeNormType = vHparams.vFNormRmsEps == 0.0f ? cLlmNorm : cLlmNormRms;

  // TODO: extract in a common llm_graph_context::build_inp_embd_h()
  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.vNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInp(), cNTokens);
  ggml_set_input(vInp->vEmbd);

  // TODO: make static using `ggml_build_forward_select()`
  //       see llm_graph_context::mBuildInpEmbd() for reference
  ggml_tensor *vTokEmbd;
  if (vUbatch.vToken) {
    ggml_tensor *vTokEmbdW = vLayer.vNextn.vEmbedTokens ? vLayer.vNextn.vEmbedTokens : pModel.vTokEmbd;
    vTokEmbd = ggml_get_rows(vCtx0, vTokEmbdW, vInp->vTokens);
  } else {
    vTokEmbd = vInp->vEmbd;
  }
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  vInp->vH = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.vNEmbd, cNTokens);
  ggml_set_input(vInp->vH);
  ggml_set_name(vInp->vH, "mtp_h_input");

  ggml_tensor *vHEmbd = vInp->vH;

  vRes->mAddInput(std::move(vInp));

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();
  auto *vInpAttn = mBuildAttnInpKvIswa();

  ggml_tensor *vHNorm = mBuildNorm(vHEmbd, vLayer.vNextn.vHnorm, nullptr, cCohere2moeNormType, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cCohere2moeNormType, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat, vLayer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpL = vCur;

  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cCohere2moeNormType, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);
  ggml_tensor *vFfnInp = vCur;

  auto [vQcur, vKcur, vVcur] = mBuildQkv(vLayer, vCur, cNEmbdHead, cNHead, cNHeadKv, cIl);
  ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, cIl);
  vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                       cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                       cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "mtp_Qcur", cIl);
  mCb(vKcur, "mtp_Kcur", cIl);
  mCb(vVcur, "mtp_Vcur", cIl);

  vCur = mBuildAttn(vInpAttn, vLayer.vWo, vLayer.vWoB, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(cNEmbdHead)), cIl);
  mCb(vCur, "mtp_attn_out", cIl);

  ggml_tensor *vAttnOut = vCur;

  vCur = mBuildMoeFfn(vFfnInp, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps, nullptr,
                      cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                      (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl, nullptr, vLayer.vFfnGateUpExps,
                      vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
  mCb(vCur, "mtp_ffn_moe_out", cIl);

  if (vLayer.vFfnUpShexp) {
    ggml_tensor *vFfnShexp = mBuildFfn(vFfnInp, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp,
                                       nullptr, vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp, nullptr,
                                       vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
    mCb(vFfnShexp, "mtp_ffn_shexp", cIl);

    vCur = ggml_add(vCtx0, vCur, vFfnShexp);
    vCur = ggml_scale(vCtx0, vCur, 0.5f);
    mCb(vCur, "mtp_ffn_out", cIl);
  }

  vCur = ggml_add(vCtx0, vCur, vInpL);
  vCur = ggml_add(vCtx0, vCur, vAttnOut);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "COHERE2MOE MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cCohere2moeNormType, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  GGML_ASSERT(vHeadW && "COHERE2MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : nullptr);

  if (vHparams.vFLogitScale) {
    vCur = ggml_scale(vCtx0, vCur, vHparams.vFLogitScale);
  }

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
