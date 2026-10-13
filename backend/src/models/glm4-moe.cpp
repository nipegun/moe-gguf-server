#include "models.h"

void mgs_model_glm4_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKeyOrArr(cLlmKvRopeDimensionSections, vHparams.aRopeSections, 4, false);

  // MoE parameters
  pMl.mGetKey(cLlmKvExpertCount, vHparams.vNExpert);
  pMl.mGetKey(cLlmKvExpertUsedCount, vHparams.vNExpertUsed);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);

  // Expert gating function (GLM-4.5 uses sigmoid)
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);

  // NextN/MTP parameters
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_impl");

  switch (vHparams.mNLayer()) {
  case 46:
    vType = cLlmType106bA12b;
    break; // GLM-4.5-Air
  case 48:
    vType = cLlmType102bA12b;
    break; // Solar Open
  case 92:
    vType = cLlmType355bA32b;
    break; // GLM-4.5
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_glm4_moe::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;
  const int64_t cNExpertShared = vHparams.vNExpertShared;

  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  GGML_ASSERT(vHparams.vNExpert > 0 && "n_expert must be > 0 for GLM4_MOE MoE layers");
  GGML_ASSERT(vHparams.vNExpertUsed > 0 && "n_expert_used must be > 0 for GLM4_MOE MoE layers");

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayerAll; ++vI) {
    auto &vLayer = lLayers[vI];
    const int cFlags = vI < cNLayer ? cTrunkFlags : vMtpFlags;

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, cFlags);

    // GLM-style attention with bias terms
    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, cFlags);

    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, cFlags);

    // K/Q norm tensors (optional for GLM-4.5 355B variant)
    vLayer.vAttnQNorm =
        mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, cTensorNotRequired | cFlags);
    vLayer.vAttnKNorm =
        mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, cTensorNotRequired | cFlags);

    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", vI), {cNEmbd}, cFlags);

    // Check if this layer uses MoE or dense FFN based on n_layer_dense_lead
    // GLM 4.5 uses hybrid architecture: layer 0 is dense, layers 1+ are MoE
    const bool cUseMoe = (static_cast<uint32_t>(vI) >= vHparams.vNLayerDenseLead);

    if (cUseMoe) {
      // MoE layers
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cFlags);

      // MoE branch
      const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;

      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);

      // Shared expert
      if (cNExpertShared > 0) {
        const int64_t cNFfShexp = cNFfExp * cNExpertShared;
        vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexp}, cFlags);
        vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, cFlags);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, cFlags);
      }
    } else {
      // Dense layers (first k layers) - GLM uses separate gate/up projections
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, cFlags);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, cFlags);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, cFlags);
    }

    // NextN/MTP tensors
    if (vI >= cNLayer) {
      vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, cFlags);
      vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, cFlags);

      // Optional tensors
      vLayer.vNextn.vEmbedTokens =
          mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab}, cTensorNotRequired | cFlags);
      vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI), {cNEmbd, cNVocab},
                                                    cTensorNotRequired | cFlags);
      vLayer.vNextn.vSharedHeadNorm =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, cTensorNotRequired | cFlags);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_glm4_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_glm4_moe::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "GLM4_MOE MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "GLM4_MOE MTP currently only supports a single MTP block");

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");

  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");
  GGML_ASSERT(vLayer.vFfnGateInp && "MTP block missing ffn_gate_inp");

  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.vNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInp(), cNTokens);
  ggml_set_input(vInp->vEmbd);

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

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vHNorm = mBuildNorm(vHEmbd, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, 0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat, vLayer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  auto [vQcur, vKcur, vVcur] = mBuildQkv(vLayer, vCur, cNEmbdHead, cNHead, cNHeadKv, cIl);

  if (vLayer.vAttnQNorm) {
    vQcur = mBuildNorm(vQcur, vLayer.vAttnQNorm, nullptr, cLlmNormRms, cIl);
    mCb(vQcur, "mtp_Qcur_normed", cIl);
  }
  if (vLayer.vAttnKNorm) {
    vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, nullptr, cLlmNormRms, cIl);
    mCb(vKcur, "mtp_Kcur_normed", cIl);
  }

  vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                       cAttnFactor, cBetaFast, cBetaSlow);

  vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                       cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "mtp_Qcur", cIl);
  mCb(vKcur, "mtp_Kcur", cIl);
  mCb(vVcur, "mtp_Vcur", cIl);

  vCur = mBuildAttn(vInpAttn, vLayer.vWo, nullptr, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(cNEmbdHead)), cIl);
  mCb(vCur, "mtp_attn_out", cIl);

  ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vFfnInp, "mtp_ffn_inp", cIl);

  vCur = mBuildNorm(vFfnInp, vLayer.vAttnPostNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_post_attn_norm", cIl);

  ggml_tensor *vRoutedOut =
      mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                   vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                   vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl);
  mCb(vRoutedOut, "mtp_ffn_moe_out", cIl);

  ggml_tensor *vSharedOut =
      mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, nullptr, vLayer.vFfnGateShexp, nullptr, nullptr,
                vLayer.vFfnDownShexp, nullptr, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
  mCb(vSharedOut, "mtp_ffn_shexp_out", cIl);

  vCur = ggml_add(vCtx0, vRoutedOut, vSharedOut);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "GLM4_MOE MTP: missing both nextn.shared_head_norm and output_norm");

  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "GLM4_MOE MTP: missing LM head (nextn.shared_head_head or model.output)");

  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}

mgs_model_glm4_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  bool vUseMrope = vHparams.mUseMrope();
  if (vUbatch.vEmbd && !vUseMrope) {
    // unfortunately, we need to forcefully stop here, to avoid users complaining about wrong results
    GGML_ABORT("This GGUF does not support multimodal. Please reconvert it.");
  }

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // NextN layers are processed by graph_mtp.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // Pre-attention norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      // Apply Q/K norm if available (GLM-4.5 355B variant)
      if (pModel.lLayers[vIl].vAttnQNorm) {
        vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
        mCb(vQcur, "Qcur_normed", vIl);
      }
      if (pModel.lLayers[vIl].vAttnKNorm) {
        vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
        mCb(vKcur, "Kcur_normed", vIl);
      }

      if (vUseMrope) {
        vQcur = ggml_rope_multi(vCtx0, vQcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase,
                               cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_multi(vCtx0, vKcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase,
                               cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      } else {
        // Normal RoPE
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                        nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds && (!vCparams.vEmbeddingsNextn || vCparams.vEmbeddingsNextnMasked)) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // Post-attention norm
    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vAttnPostNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "post_attn_norm", vIl);

    // Check if this is a dense layer (n_layer_dense_lead=1, so layer 0 is dense)
    if (static_cast<uint32_t>(vIl) < vHparams.vNLayerDenseLead) {
      // Dense FFN layer
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // Process routed experts using existing MoE infrastructure
      ggml_tensor *vRoutedOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                             pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                             pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
                                             vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                                             (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vRoutedOut, "ffn_moe_out", vIl);

      // Process shared expert on original input
      ggml_tensor *vSharedOut =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                    pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vSharedOut, "ffn_shexp_out", vIl);

      // Final output: routed_output + shared_output
      vCur = ggml_add(vCtx0, vRoutedOut, vSharedOut);
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

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (vCparams.vEmbeddingsNextn && !vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
