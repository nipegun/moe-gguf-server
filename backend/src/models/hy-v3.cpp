#include "models.h"

void mgs_model_hy_v3::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);

  // NextN/MTP (HY V3): extra decoder block(s) appended beyond the main stack
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_all");

  switch (vHparams.mNLayer()) {
  case 48:
    vType = cLlmType30bA3b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_hy_v3::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  // Trunk-only: the GGUF declares MTP layers in metadata but the actual MTP
  // tensors live in a separate file (e.g. user split target/draft). Mark
  // MTP tensors NOT_REQUIRED so the trunk loads cleanly.
  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  auto vLoadBlock = [&](int pI, int pFlags) {
    auto &vLayer = lLayers[pI];
    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / (cNExpertUsed > 0 ? cNExpertUsed : 1);
    const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFfExp;

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pI), {cNEmbd}, pFlags);

    mCreateTensorQkv(vLayer, pI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdKGqa, cNEmbdVGqa, pFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pI), {cNEmbdHeadK * cNHead, cNEmbd}, pFlags);

    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pI), {cNEmbdHeadK}, pFlags);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pI), {cNEmbdHeadK}, pFlags);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", pI), {cNEmbd}, pFlags);

    // dense FFN (leading dense blocks, first_k_dense_replace)
    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pI), {cNFf, cNEmbd}, cTensorNotRequired);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);

    // MoE routed experts (sigmoid router + expert selection bias)
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pI), {cNEmbd, cNExpert}, cTensorNotRequired);
    vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, pI), {cNExpert}, cTensorNotRequired);
    vLayer.vFfnDownExps =
        mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pI), {cNFfExp, cNEmbd, cNExpert}, cTensorNotRequired);
    mCreateTensorGateUpExps(vLayer, pI, cNEmbd, cNFfExp, cNExpert, cTensorNotRequired);

    // shared expert (always active, no gate)
    vLayer.vFfnGateShexp =
        mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pI), {cNEmbd, cNFfShexp}, cTensorNotRequired);
    vLayer.vFfnUpShexp =
        mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pI), {cNEmbd, cNFfShexp}, cTensorNotRequired);
    vLayer.vFfnDownShexp =
        mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pI), {cNFfShexp, cNEmbd}, cTensorNotRequired);
  };

  for (int vI = 0; vI < cNLayer; ++vI) {
    vLoadBlock(vI, cTrunkFlags);
  }

  // NextN/MTP block(s): a full hy_v3 decoder block plus the NextN projections.
  for (int vI = cNLayer; vI < cNLayerAll; ++vI) {
    auto &vLayer = lLayers[vI];

    vLoadBlock(vI, vMtpFlags);

    vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, vMtpFlags);
    vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, vMtpFlags);
    vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, vMtpFlags);
    vLayer.vNextn.vEmbedTokens =
        mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab}, cTensorNotRequired);
    vLayer.vNextn.vSharedHeadHead =
        mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI), {cNEmbd, cNVocab}, cTensorNotRequired);
    // hy_v3 stores the MTP block's trailing final_layernorm here (applied
    // after the decoder block, before the shared LM head).
    vLayer.vNextn.vSharedHeadNorm =
        mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, cTensorNotRequired);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_hy_v3::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_hy_v3::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  ggml_tensor *vInpPos = mBuildInpPos();
  auto *vInpAttn = mBuildAttnInpKv();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

  // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self-attention
    {
      ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, vIl);

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, nullptr, cLlmNormRms, vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, nullptr, cLlmNormRms, vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds && vCparams.vEmbeddingsNextnMasked) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      // dense FFN (leading dense blocks)
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, pModel.lLayers[vIl].vFfnUpS,
                       pModel.lLayers[vIl].vFfnGate, pModel.lLayers[vIl].vFfnGateB, pModel.lLayers[vIl].vFfnGateS,
                       pModel.lLayers[vIl].vFfnDown, pModel.lLayers[vIl].vFfnDownB, pModel.lLayers[vIl].vFfnDownS,
                       nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_dense_out", vIl);
    } else {
      // MoE routed experts (sigmoid gating + expert selection bias)
      ggml_tensor *vMoeOut = mBuildMoeFfn(
          vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
          pModel.lLayers[vIl].vFfnDownExps, pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl, nullptr, pModel.lLayers[vIl].vFfnGateUpExps,
          pModel.lLayers[vIl].vFfnUpExpsS, pModel.lLayers[vIl].vFfnGateExpsS, pModel.lLayers[vIl].vFfnDownExpsS);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // shared expert (always active, no gate)
      ggml_tensor *vShOut = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, nullptr, pModel.lLayers[vIl].vFfnUpShexpS,
                                      pModel.lLayers[vIl].vFfnGateShexp, nullptr, pModel.lLayers[vIl].vFfnGateShexpS,
                                      pModel.lLayers[vIl].vFfnDownShexp, nullptr, pModel.lLayers[vIl].vFfnDownShexpS,
                                      nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vShOut, "ffn_shared_out", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vShOut);
      mCb(vCur, "ffn_out", vIl);
    }

    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    vInpL = vCur;
  }

  vCur = mBuildNorm(vInpL, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);

  // Post-final-norm hidden state: what the MTP draft head's hnorm consumes.
  // vLLM feeds the target model's normed output states, and the MTP layer
  // itself returns final_layernorm(h), so the chained state is post-norm.
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for HY V3 (MoE).
// Semantics mirror vLLM's HYV3MultiTokenPredictorLayer (hy_v3_mtp.py):
//   enorm(embed) + hnorm(prev_hidden) -> concat(e, h) -> eh_proj ->
//   hy_v3 decoder block -> final_layernorm (stored as nextn.shared_head_norm) ->
//   shared LM head (the main model's lm_head; the checkpoint has no separate
//   MTP head or MTP embeddings).
mgs_model_hy_v3::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "HY_V3 MTP requires n_layer_nextn > 0");

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");

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
  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vHNorm = mBuildNorm(vHInput, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  // mtp_block: a full hy_v3 decoder layer (mirrors the trunk graph)
  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  {
    ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, cIl);

    auto [vQcur, vKcur, vVcur] = mBuildQkv(vLayer, vCur, cNEmbdHead, cNHead, cNHeadKv, cIl);

    vQcur = mBuildNorm(vQcur, vLayer.vAttnQNorm, nullptr, cLlmNormRms, cIl);
    vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, nullptr, cLlmNormRms, cIl);

    vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
    vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

    const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

    vCur = mBuildAttn(vInpAttn, vLayer.vWo, vLayer.vWoB, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr,
                      cKqScale, cIl);
    mCb(vCur, "mtp_attn_out", cIl);
  }

  ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vFfnInp, "mtp_ffn_inp", cIl);

  vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_ffn_norm", cIl);

  if (vLayer.vFfnGateInp == nullptr) {
    vCur = mBuildFfn(vCur, vLayer.vFfnUp, vLayer.vFfnUpB, vLayer.vFfnUpS, vLayer.vFfnGate, vLayer.vFfnGateB,
                     vLayer.vFfnGateS, vLayer.vFfnDown, vLayer.vFfnDownB, vLayer.vFfnDownS, nullptr, cLlmFfnSilu,
                     cLlmFfnPar, cIl);
    mCb(vCur, "mtp_ffn_dense_out", cIl);
  } else {
    ggml_tensor *vMoeOut =
        mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                     vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                     vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl,
                     nullptr, vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
    mCb(vMoeOut, "mtp_ffn_moe_out", cIl);

    ggml_tensor *vShOut = mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp,
                                    nullptr, vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp, nullptr,
                                    vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
    mCb(vShOut, "mtp_ffn_shared_out", cIl);

    vCur = ggml_add(vCtx0, vMoeOut, vShOut);
    mCb(vCur, "mtp_ffn_out", cIl);
  }

  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  // final_layernorm applied after the decoder block, before the shared head.
  // The post-norm hidden state seeds the next MTP step (matches vLLM, where
  // HYV3MultiTokenPredictorLayer returns final_layernorm(h)).
  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "HY_V3 MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "HY_V3 MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
