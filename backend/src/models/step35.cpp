#include "models.h"

void mgs_model_step35::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  vHparams.vSwaType = cMgsSwaTypeStandard;

  // full_attention layer only use half of the RoPE dimensions
  vHparams.vNRotFull = vHparams.vNRotFull / 2;

  // MoE + SWA parameters
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);

  // Step35 uses sigmoid gating by default (if not set in GGUF)
  if (vHparams.vExpertGatingFunc == cMgsExpertGatingFuncTypeNone) {
    vHparams.vExpertGatingFunc = cMgsExpertGatingFuncTypeSigmoid;
  }

  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  pMl.mGetKey(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa, false);

  pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl, vHparams.mNLayer());

  pMl.mGetKeyOrArr(cLlmKvSwigluClampExp, vHparams.aSwigluClampExp, vHparams.mNLayer(), false);
  pMl.mGetKeyOrArr(cLlmKvSwigluClampShexp, vHparams.aSwigluClampShexp, vHparams.mNLayer(), false);

  // NextN/MTP (Step3p5): extra decoder block appended beyond the main stack.
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_impl");

  switch (vHparams.mNLayer()) {
  case 45:
    vType = cLlmType196bA11b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_step35::mLoadArchTensors(mgs_model_loader &pMl) {
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

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTrunkFlags);

  // STEP35 supports per-layer partial RoPE dims; rope factors are stored as a single shared tensor
  // ("rope_freqs.weight") and ggml uses only the first (n_rot_l/2) entries per layer.
  uint32_t vNRotMax = 0;
  for (int vI = 0; vI < cNLayer; ++vI) {
    vNRotMax = std::max(vNRotMax, vHparams.mNRot(vI));
  }
  if (vNRotMax == 0) {
    vNRotMax = cNRot;
  }

  auto vLoadBlockTrunk = [&](int pI, int pFlags) {
    auto &vLayer = lLayers[pI];

    const uint32_t cNHeadL = vHparams.mNHead(pI);
    const uint32_t cNEmbdKGqa = vHparams.mNEmbdKGqa(pI);
    const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(pI);

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pI), {cNEmbd}, pFlags);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pI), {cNEmbdHeadK}, cTensorNotRequired);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pI), {cNEmbdHeadK}, cTensorNotRequired);

    // optional rope factors / longrope tensors
    if (vHparams.vRopeScalingTypeTrain == MGS_ROPE_SCALING_TYPE_LONGROPE) {
      vLayer.vRopeLong = mCreateTensor(cTn(cLlmTensorRopeFactorsLong, "weight", pI), {vNRotMax / 2},
                                       cTensorNotRequired | (pI != 0 ? cTensorDuplicated : 0));
      vLayer.vRopeShort = mCreateTensor(cTn(cLlmTensorRopeFactorsShort, "weight", pI), {vNRotMax / 2},
                                        cTensorNotRequired | (pI != 0 ? cTensorDuplicated : 0));
    } else {
      vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", pI), {vNRotMax / 2},
                                        cTensorNotRequired | (pI != 0 ? cTensorDuplicated : 0));
    }

    mCreateTensorQkv(vLayer, pI, cNEmbd, cNEmbdHeadK * cNHeadL, cNEmbdKGqa, cNEmbdVGqa, pFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pI), {cNEmbdHeadV * cNHeadL, cNEmbd}, pFlags);

    // head-wise attention gate (Step35 self_attn.g_proj)
    vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", pI), {cNEmbd, cNHeadL}, cTensorNotRequired);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", pI), {cNEmbd}, pFlags);

    // dense MLP (leading dense blocks)
    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pI), {cNFf, cNEmbd}, cTensorNotRequired);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);

    // MoE routed experts + selection bias (router_bias)
    const int64_t cNFfExp = vHparams.vNFfExp;
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pI), {cNEmbd, cNExpert}, cTensorNotRequired);
    vLayer.vFfnGateExps =
        mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pI), {cNEmbd, cNFfExp, cNExpert}, cTensorNotRequired);
    vLayer.vFfnDownExps =
        mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pI), {cNFfExp, cNEmbd, cNExpert}, cTensorNotRequired);
    vLayer.vFfnUpExps =
        mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pI), {cNEmbd, cNFfExp, cNExpert}, cTensorNotRequired);
    vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", pI), {cNExpert}, cTensorNotRequired);

    // shared expert MLP
    vLayer.vFfnGateShexp =
        mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pI), {cNEmbd, vHparams.vNFfShexp}, cTensorNotRequired);
    vLayer.vFfnUpShexp =
        mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pI), {cNEmbd, vHparams.vNFfShexp}, cTensorNotRequired);
    vLayer.vFfnDownShexp =
        mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pI), {vHparams.vNFfShexp, cNEmbd}, cTensorNotRequired);
  };

  auto vLoadBlockMtp = [&](int pI) {
    auto &vLayer = lLayers[pI];

    const uint32_t cNHeadL = vHparams.mNHead(pI);
    const uint32_t cNEmbdKGqa = vHparams.mNEmbdKGqa(pI);
    const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(pI);

    // The MTP block is a full Step3p5 decoder layer (mtp_block) plus the
    // NextN-specific wiring (enorm/hnorm/eh_proj + optional shared head).
    // Multi-block MTP: every declared MTP block is required (the draft chain
    // runs all n_layer_nextn heads), so each block uses the captured
    // `vMtpFlags` directly — already NOT_REQUIRED for a trunk-only GGUF,
    // which keeps that path correct.

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pI), {cNEmbd}, vMtpFlags);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pI), {cNEmbdHeadK}, cTensorNotRequired);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pI), {cNEmbdHeadK}, cTensorNotRequired);

    if (vHparams.vRopeScalingTypeTrain == MGS_ROPE_SCALING_TYPE_LONGROPE) {
      vLayer.vRopeLong = mCreateTensor(cTn(cLlmTensorRopeFactorsLong, "weight", pI), {vNRotMax / 2},
                                       cTensorNotRequired | cTensorDuplicated);
      vLayer.vRopeShort = mCreateTensor(cTn(cLlmTensorRopeFactorsShort, "weight", pI), {vNRotMax / 2},
                                        cTensorNotRequired | cTensorDuplicated);
    } else {
      vLayer.vRopeFreqs =
          mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", pI), {vNRotMax / 2}, cTensorNotRequired | cTensorDuplicated);
    }

    mCreateTensorQkv(vLayer, pI, cNEmbd, cNEmbdHeadK * cNHeadL, cNEmbdKGqa, cNEmbdVGqa, vMtpFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pI), {cNEmbdHeadV * cNHeadL, cNEmbd}, vMtpFlags);

    vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", pI), {cNEmbd, cNHeadL}, cTensorNotRequired);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", pI), {cNEmbd}, vMtpFlags);

    // dense MLP (leading dense blocks) — present if the MTP block isn't MoE
    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pI), {cNFf, cNEmbd}, cTensorNotRequired);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pI), {cNEmbd, cNFf}, cTensorNotRequired);

    // MoE routed experts + selection bias (router_bias)
    const int64_t cNFfExp = vHparams.vNFfExp;
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pI), {cNEmbd, cNExpert}, cTensorNotRequired);
    vLayer.vFfnGateExps =
        mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pI), {cNEmbd, cNFfExp, cNExpert}, cTensorNotRequired);
    vLayer.vFfnDownExps =
        mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pI), {cNFfExp, cNEmbd, cNExpert}, cTensorNotRequired);
    vLayer.vFfnUpExps =
        mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pI), {cNEmbd, cNFfExp, cNExpert}, cTensorNotRequired);
    vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", pI), {cNExpert}, cTensorNotRequired);

    vLayer.vFfnGateShexp =
        mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pI), {cNEmbd, vHparams.vNFfShexp}, cTensorNotRequired);
    vLayer.vFfnUpShexp =
        mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pI), {cNEmbd, vHparams.vNFfShexp}, cTensorNotRequired);
    vLayer.vFfnDownShexp =
        mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pI), {vHparams.vNFfShexp, cNEmbd}, cTensorNotRequired);

    // NextN-specific tensors that define the MTP block.
    vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", pI), {2 * cNEmbd, cNEmbd}, vMtpFlags);
    vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", pI), {cNEmbd}, vMtpFlags);
    vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", pI), {cNEmbd}, vMtpFlags);
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
  // All n_layer_nextn MTP blocks are required — the multi-block draft chain
  // runs every head (head k at offset k). The GGUF declares the count via
  // step35.nextn_predict_layers.
  for (int vI = cNLayer; vI < cNLayerAll; ++vI) {
    vLoadBlockMtp(vI);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_step35::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_step35::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  ggml_tensor *vInpPos = mBuildInpPos();
  auto *vInpAttn = mBuildAttnInpKvIswa();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    const uint32_t cNHeadL = vHparams.mNHead(vIl);
    const uint32_t cNHeadKvL = vHparams.mNHeadKv(vIl);

    const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, vIl);
    const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, vIl);

    vCur = vInpL;

    // dump pre-attn RMSNorm input to pinpoint layer boundary issues
    mCb(vCur, "attn_norm_in", vIl);

    // self-attention
    {
      vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
      mCb(vCur, "attn_norm", vIl);
      ggml_tensor *vQcur = mBuildLoraMm(pModel.lLayers[vIl].vWq, vCur);
      ggml_tensor *vKcur = mBuildLoraMm(pModel.lLayers[vIl].vWk, vCur);
      ggml_tensor *vVcur = mBuildLoraMm(pModel.lLayers[vIl].vWv, vCur);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHeadK, cNHeadL, cNTokens);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHeadK, cNHeadKvL, cNTokens);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHeadV, cNHeadKvL, cNTokens);

      // Q/K per-head RMSNorm (Step35 q_norm / k_norm)
      if (pModel.lLayers[vIl].vAttnQNorm) {
        vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, nullptr, cLlmNormRms, vIl);
        mCb(vQcur, "Qcur_normed", vIl);
      }
      if (pModel.lLayers[vIl].vAttnKNorm) {
        vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, nullptr, cLlmNormRms, vIl);
        mCb(vKcur, "Kcur_normed", vIl);
      }

      // RoPE (partial rotary factors per layer)
      const bool cIsSwa = vHparams.mIsSwa(vIl);
      ggml_tensor *vRopeFactors = cIsSwa ? nullptr : pModel.mGetRopeFactors(vCparams, vIl);
      const int64_t cNRotL = vHparams.mNRot(vIl);
      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRotL, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRotL, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      mCb(vQcur, "Qcur_pos", vIl);
      mCb(vKcur, "Kcur_pos", vIl);

      const float cKqScale = 1.0f / sqrtf(float(cNEmbdHeadK));
      ggml_tensor *vAttnOut = mBuildAttn(vInpAttn, nullptr, nullptr, nullptr, vQcur, vKcur, vVcur, nullptr, nullptr,
                                         nullptr, cKqScale, vIl);
      mCb(vAttnOut, "attn_out", vIl);
      // head-wise attention gate: sigmoid(g_proj(x)) in torch
      if (pModel.lLayers[vIl].vWqkvGate) {
        ggml_tensor *vGate = mBuildLoraMm(pModel.lLayers[vIl].vWqkvGate, vCur); // [n_head_l, n_tokens]
        mCb(vGate, "attn_gate", vIl);

        vGate = ggml_sigmoid(vCtx0, vGate);
        mCb(vGate, "attn_gate_sigmoid", vIl);

        // reshape + broadcast to [n_embd_head_v, n_head_l, n_tokens]
        ggml_tensor *vAttn3d = ggml_reshape_3d(vCtx0, vAttnOut, cNEmbdHeadV, cNHeadL, cNTokens);
        ggml_tensor *vGate3d = ggml_reshape_3d(vCtx0, vGate, 1, cNHeadL, cNTokens);
        mCb(vGate3d, "attn_gate_3d", vIl);

        vAttn3d = ggml_mul(vCtx0, vAttn3d, vGate3d);
        mCb(vAttn3d, "attn_gated_3d", vIl);

        vAttnOut = ggml_reshape_2d(vCtx0, vAttn3d, cNEmbdHeadV * cNHeadL, cNTokens);
        mCb(vAttnOut, "attn_gated", vIl);
      }

      // output projection
      vCur = mBuildLoraMm(pModel.lLayers[vIl].vWo, vAttnOut, pModel.lLayers[vIl].vWoS);
      mCb(vCur, "attn_proj", vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds && vCparams.vEmbeddingsNextnMasked) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // feed-forward
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      // dense MLP
      vCur =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, nullptr,
                    pModel.lLayers[vIl].vFfnGate, pModel.lLayers[vIl].vFfnGateB, nullptr, pModel.lLayers[vIl].vFfnDown,
                    pModel.lLayers[vIl].vFfnDownB, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE routed experts
      ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
                                          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                                          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // shared expert MLP (always added on MoE layers in Step35)
      ggml_tensor *vShOut = mBuildFfn(
          vCur, pModel.lLayers[vIl].vFfnUpShexp, nullptr, nullptr, pModel.lLayers[vIl].vFfnGateShexp, nullptr, nullptr,
          pModel.lLayers[vIl].vFfnDownShexp, nullptr, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vShOut, "ffn_shared_out", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vShOut);
      mCb(vCur, "ffn_out", vIl);
    }
    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }

  vCur = vInpL;

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Step3p5 (MoE)
mgs_model_step35::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "STEP35 MTP requires n_layer_nextn > 0");

  // Multi-block MTP: the DECODER_MTP graph runs the MTP head selected by
  // cparams.vNextnLayerOffset (0 = first trained head). The speculative driver
  // bumps the offset per draft step to chain heads 45->46->47. offset 0 keeps
  // single-block behavior identical to before.
  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");

  const uint32_t cNHeadL = vHparams.mNHead(cIl);
  const uint32_t cNHeadKvL = vHparams.mNHeadKv(cIl);

  const float cFreqBaseL = pModel.mGetRopeFreqBase(vCparams, cIl);
  const float cFreqScaleL = pModel.mGetRopeFreqScale(vCparams, cIl);

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
  auto *vInpAttn = mBuildAttnInpKvIswa();

  ggml_tensor *vHNorm = mBuildNorm(vHInput, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  // mtp_block: full Step3p5 decoder layer (attention with optional head-wise gate, then MoE/dense FFN)
  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  ggml_tensor *vQcur = mBuildLoraMm(vLayer.vWq, vCur, vLayer.vWqS);
  ggml_tensor *vKcur = mBuildLoraMm(vLayer.vWk, vCur, vLayer.vWkS);
  ggml_tensor *vVcur = mBuildLoraMm(vLayer.vWv, vCur, vLayer.vWvS);
  mCb(vQcur, "mtp_Qcur", cIl);
  mCb(vKcur, "mtp_Kcur", cIl);
  mCb(vVcur, "mtp_Vcur", cIl);

  vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHeadK, cNHeadL, cNTokens);
  vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHeadK, cNHeadKvL, cNTokens);
  vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHeadV, cNHeadKvL, cNTokens);

  if (vLayer.vAttnQNorm) {
    vQcur = mBuildNorm(vQcur, vLayer.vAttnQNorm, nullptr, cLlmNormRms, cIl);
    mCb(vQcur, "mtp_Qcur_normed", cIl);
  }
  if (vLayer.vAttnKNorm) {
    vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, nullptr, cLlmNormRms, cIl);
    mCb(vKcur, "mtp_Kcur_normed", cIl);
  }

  const bool cIsSwa = vHparams.mIsSwa(cIl);
  ggml_tensor *vRopeFactors = cIsSwa ? nullptr : pModel.mGetRopeFactors(vCparams, cIl);
  const int64_t cNRotL = vHparams.mNRot(cIl);

  vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, vRopeFactors, cNRotL, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                        cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, vRopeFactors, cNRotL, cRopeType, cNCtxOrig, cFreqBaseL, cFreqScaleL,
                        cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  mCb(vQcur, "mtp_Qcur_pos", cIl);
  mCb(vKcur, "mtp_Kcur_pos", cIl);

  const float cKqScale = 1.0f / sqrtf(float(cNEmbdHeadK));
  ggml_tensor *vAttnOut =
      mBuildAttn(vInpAttn, nullptr, nullptr, nullptr, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, cIl);
  mCb(vAttnOut, "mtp_attn_out", cIl);

  // head-wise attention gate: sigmoid(g_proj(x))
  if (vLayer.vWqkvGate) {
    ggml_tensor *vGate = mBuildLoraMm(vLayer.vWqkvGate, vCur); // [n_head_l, n_tokens]
    mCb(vGate, "mtp_attn_gate", cIl);

    vGate = ggml_sigmoid(vCtx0, vGate);
    mCb(vGate, "mtp_attn_gate_sigmoid", cIl);

    ggml_tensor *vAttn3d = ggml_reshape_3d(vCtx0, vAttnOut, cNEmbdHeadV, cNHeadL, cNTokens);
    ggml_tensor *vGate3d = ggml_reshape_3d(vCtx0, vGate, 1, cNHeadL, cNTokens);
    mCb(vGate3d, "mtp_attn_gate_3d", cIl);

    vAttn3d = ggml_mul(vCtx0, vAttn3d, vGate3d);
    mCb(vAttn3d, "mtp_attn_gated_3d", cIl);

    vAttnOut = ggml_reshape_2d(vCtx0, vAttn3d, cNEmbdHeadV * cNHeadL, cNTokens);
    mCb(vAttnOut, "mtp_attn_gated", cIl);
  }

  vCur = mBuildLoraMm(vLayer.vWo, vAttnOut, vLayer.vWoS);
  mCb(vCur, "mtp_attn_proj", cIl);

  vCur = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vCur, "mtp_attn_residual", cIl);

  ggml_tensor *vFfnInp = vCur;
  vCur = mBuildNorm(vCur, vLayer.vFfnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_ffn_norm", cIl);

  // FFN: dense MLP or MoE (mirrors trunk path)
  if (vLayer.vFfnGateInp == nullptr) {
    vCur = mBuildFfn(vCur, vLayer.vFfnUp, vLayer.vFfnUpB, nullptr, vLayer.vFfnGate, vLayer.vFfnGateB, nullptr,
                     vLayer.vFfnDown, vLayer.vFfnDownB, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
    mCb(vCur, "mtp_ffn_out", cIl);
  } else {
    ggml_tensor *vMoeOut =
        mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                     vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                     vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl);
    mCb(vMoeOut, "mtp_ffn_moe_out", cIl);

    ggml_tensor *vShOut = mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, nullptr, vLayer.vFfnGateShexp, nullptr, nullptr,
                                    vLayer.vFfnDownShexp, nullptr, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
    mCb(vShOut, "mtp_ffn_shared_out", cIl);

    vCur = ggml_add(vCtx0, vMoeOut, vShOut);
    mCb(vCur, "mtp_ffn_out", cIl);
  }
  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vInpOutIds = mBuildInpOutIds();
  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);

  // Pre-norm hidden state: used by the AR draft loop to seed the next MTP step.
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "STEP35 MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  GGML_ASSERT(vHeadW && "STEP35 MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
