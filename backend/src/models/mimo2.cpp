#include "models.h"

void mgs_model_mimo2::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  vHparams.vSwaType = cMgsSwaTypeStandard;

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);

  pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl, vHparams.mNLayer());

  float vValueScale = 0.0f;
  if (pMl.mGetKey(cLlmKvAttentionValueScale, vValueScale, false) && vValueScale != 1.0f) {
    vHparams.vFAttnValueScale = vValueScale;
  }

  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_impl");

  switch (vHparams.mNLayer()) {
  case 48:
    vType = cLlmType310bA15b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_mimo2::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayerAll; ++vI) {
    auto &vLayer = lLayers[vI];
    uint32_t vNEmbdKGqa = vHparams.mNEmbdKGqa(vI);
    uint32_t vNEmbdVGqa = vHparams.mNEmbdVGqa(vI);
    uint32_t vNHead = vHparams.mNHead(vI);

    const bool cIsNextn = vI >= cNLayer;
    const int cFlags = cIsNextn ? vMtpFlags : 0;

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * vNHead, vNEmbdKGqa, vNEmbdVGqa, cFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadV * vNHead, cNEmbd}, cFlags);

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, cFlags);
    vLayer.vAttnSinks = mCreateTensor(cTn(cLlmTensorAttnSinks, "weight", vI), {vNHead}, cTensorNotRequired | cFlags);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, cFlags);

    // non-MoE branch
    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, cTensorNotRequired | cFlags);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, cTensorNotRequired | cFlags);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, cTensorNotRequired | cFlags);

    // MoE branch
    int64_t vNFfExp = vHparams.vNFfExp;
    vLayer.vFfnGateInp =
        mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, cTensorNotRequired | cFlags);
    vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, vNFfExp, cNExpert},
                                        cTensorNotRequired | cFlags);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {vNFfExp, cNEmbd, cNExpert},
                                        cTensorNotRequired | cFlags);
    vLayer.vFfnUpExps =
        mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, vNFfExp, cNExpert}, cTensorNotRequired | cFlags);
    vLayer.vFfnExpProbsB =
        mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cTensorNotRequired | cFlags);

    if (cIsNextn) {
      vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, cFlags);
      vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vEmbedTokens =
          mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab}, cTensorNotRequired | cFlags);
      vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI), {cNEmbd, cNVocab},
                                                    cTensorNotRequired | cFlags);
      vLayer.vNextn.vSharedHeadNorm =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, cTensorNotRequired | cFlags);
      vLayer.vLayerOutNorm =
          mCreateTensor(cTn(cLlmTensorLayerOutNorm, "weight", vI), {cNEmbd}, cTensorNotRequired | cFlags);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_mimo2::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_mimo2::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  ggml_tensor *vInpPos = mBuildInpPos();
  auto *vInpAttn = mBuildAttnInpKvIswa();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const float cVScale = vHparams.vFAttnValueScale;
  const bool cEmitHNextn = vCparams.vEmbeddingsNextn;
  const bool cCropLastLayer = vInpOutIds && (!cEmitHNextn || vCparams.vEmbeddingsNextnMasked);

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    uint32_t vNHeadL = vHparams.mNHead(vIl);
    uint32_t vNHeadKvL = vHparams.mNHeadKv(vIl);
    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    vCur = vInpL;

    // self_attention
    {
      vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "attn_norm", vIl);

      ggml_tensor *vQcur;
      ggml_tensor *vKcur;
      ggml_tensor *vVcur;

      if (pModel.lLayers[vIl].vWqkv) {
        // Fused qkv_proj - Q/K share head_dim_k, V uses head_dim_v
        ggml_tensor *vQkv = mBuildLoraMm(pModel.lLayers[vIl].vWqkv, vCur);
        mCb(vQkv, "wqkv", vIl);

        const size_t cRowK = ggml_row_size(vQkv->type, cNEmbdHeadK);
        const size_t cRowV = ggml_row_size(vQkv->type, cNEmbdHeadV);
        const size_t cRowFull = vQkv->nb[1];
        const size_t cKOff = cRowK * vNHeadL;
        const size_t cVOff = cKOff + cRowK * vNHeadKvL;

        vQcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadK, vNHeadL, cNTokens, cRowK, cRowFull, 0);
        vKcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadK, vNHeadKvL, cNTokens, cRowK, cRowFull, cKOff);
        vVcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadV, vNHeadKvL, cNTokens, cRowV, cRowFull, cVOff);
      } else {
        // Split path
        vQcur = mBuildLoraMm(pModel.lLayers[vIl].vWq, vCur);
        mCb(vQcur, "Qcur", vIl);

        vKcur = mBuildLoraMm(pModel.lLayers[vIl].vWk, vCur);
        mCb(vKcur, "Kcur", vIl);

        vVcur = mBuildLoraMm(pModel.lLayers[vIl].vWv, vCur);
        mCb(vVcur, "Vcur", vIl);

        vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHeadK, vNHeadL, cNTokens);
        vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHeadK, vNHeadKvL, cNTokens);
        vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHeadV, vNHeadKvL, cNTokens);
      }

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      ggml_tensor *vSinks = pModel.lLayers[vIl].vAttnSinks;

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                        vSinks, nullptr, 1.0f / sqrtf(float(cNEmbdHeadK)), vIl);
      mCb(vCur, "attn_out", vIl);

      if (cVScale) {
        vCur = ggml_scale(vCtx0, vCur, cVScale);
        mCb(vCur, "attn_out_scaled", vIl);
      }
    }

    if (vIl == cNLayer - 1 && cCropLastLayer) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // feed-forward network
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      // dense branch
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, NULL,
                       pModel.lLayers[vIl].vFfnGate, pModel.lLayers[vIl].vFfnGateB, NULL, pModel.lLayers[vIl].vFfnDown,
                       pModel.lLayers[vIl].vFfnDownB, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
                          vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSigmoid, vIl);
      mCb(vCur, "ffn_moe_out", vIl);
    }

    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }

  vCur = vInpL;

  if (cEmitHNextn) {
    mCb(vCur, "h_nextn", -1);
    vRes->vTHNextn = vCur;

    if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
    }
  }

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// Mirrors MiMo's appended NextN block: normalize and fuse token and hidden inputs, run the decoder block,
// expose its pre-head-norm state to the next draft step, then apply the shared output norm and LM head.
// Converted checkpoints may store that shared norm as layer_out_norm, so it remains in the fallback chain.
mgs_model_mimo2::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "MIMO2 MTP requires n_layer_nextn > 0");

  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");

  const auto &vLayer = pModel.lLayers[cIl];
  GGML_ASSERT(vLayer.vNextn.vEhProj && "MIMO2 MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MIMO2 MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MIMO2 MTP block missing nextn.hnorm");
  GGML_ASSERT(vLayer.vWqkv && "MIMO2 MTP requires fused attn_qkv");

  const uint32_t cNHeadL = vHparams.mNHead(cIl);
  const uint32_t cNHeadKvL = vHparams.mNHeadKv(cIl);

  const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, cIl);
  const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, cIl);
  const float cVScale = vHparams.vFAttnValueScale;

  auto vInp = std::make_unique<llm_graph_input_embd>(vHparams.vNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.vNEmbd, cNTokens);
  ggml_set_input(vInp->vEmbd);
  ggml_set_name(vInp->vEmbd, "mtp_h_input");

  ggml_tensor *vTokEmbdW = vLayer.vNextn.vEmbedTokens ? vLayer.vNextn.vEmbedTokens : pModel.vTokEmbd;
  ggml_tensor *vHInput = vInp->vEmbd;
  ggml_tensor *vTokEmbd = ggml_get_rows(vCtx0, vTokEmbdW, vInp->vTokens);
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  vRes->mAddInput(std::move(vInp));

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();
  auto *vInpAttn = mBuildAttnInpKvIswa();

  ggml_tensor *vHNorm = mBuildNorm(vHInput, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat, vLayer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  ggml_tensor *vQkv = mBuildLoraMm(vLayer.vWqkv, vCur, vLayer.vWqkvS);
  mCb(vQkv, "mtp_wqkv", cIl);

  const size_t cRowK = ggml_row_size(vQkv->type, cNEmbdHeadK);
  const size_t cRowV = ggml_row_size(vQkv->type, cNEmbdHeadV);
  const size_t cRowFull = vQkv->nb[1];
  const size_t cKOff = cRowK * cNHeadL;
  const size_t cVOff = cKOff + cRowK * cNHeadKvL;

  ggml_tensor *vQcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadK, cNHeadL, cNTokens, cRowK, cRowFull, 0);
  ggml_tensor *vKcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadK, cNHeadKvL, cNTokens, cRowK, cRowFull, cKOff);
  ggml_tensor *vVcur = ggml_view_3d(vCtx0, vQkv, cNEmbdHeadV, cNHeadKvL, cNTokens, cRowV, cRowFull, cVOff);

  vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                        cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                        cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "mtp_Qcur", cIl);
  mCb(vKcur, "mtp_Kcur", cIl);
  mCb(vVcur, "mtp_Vcur", cIl);

  vCur = mBuildAttn(vInpAttn, vLayer.vWo, nullptr, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, vLayer.vAttnSinks,
                    nullptr, 1.0f / sqrtf(float(cNEmbdHeadK)), cIl);
  mCb(vCur, "mtp_attn_out", cIl);

  if (cVScale) {
    vCur = ggml_scale(vCtx0, vCur, cVScale);
    mCb(vCur, "mtp_attn_out_scaled", cIl);
  }

  ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vFfnInp, "mtp_ffn_inp", cIl);

  vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_ffn_norm", cIl);

  GGML_ASSERT(vLayer.vFfnGate && vLayer.vFfnDown && vLayer.vFfnUp && "MIMO2 MTP requires dense FFN tensors");
  vCur = mBuildFfn(vCur, vLayer.vFfnUp, vLayer.vFfnUpB, nullptr, vLayer.vFfnGate, vLayer.vFfnGateB, nullptr,
                   vLayer.vFfnDown, vLayer.vFfnDownB, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm
                                ? vLayer.vNextn.vSharedHeadNorm
                                : (vLayer.vLayerOutNorm ? vLayer.vLayerOutNorm : pModel.vOutputNorm);
  GGML_ASSERT(vHeadNormW && "MIMO2 MTP missing head norm fallback");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "MIMO2 MTP missing LM head fallback");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
