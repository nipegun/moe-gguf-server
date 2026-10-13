#include "models.h"
#include "mgs-memory-recurrent.h"

void mgs_model_qwen35moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  pMl.mGetKeyOrArr(cLlmKvRopeDimensionSections, vHparams.aRopeSections, 4, true);

  // Load linear attention (gated delta net) parameters
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  pMl.mGetKey(cLlmKvSsmStateSize, vHparams.vSsmDState);
  pMl.mGetKey(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  pMl.mGetKey(cLlmKvSsmGroupCount, vHparams.vSsmNGroup);

  // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_impl");

  // Mark recurrent layers (linear attention layers). MTP layers are dense
  // attention-only and must be flagged non-recurrent.
  if (!pMl.mGetKeyOrArr(cLlmKvAttentionRecurrentLayers, vHparams.aIsRecrImpl, vHparams.vNLayerAll, false)) {
    uint32_t vFullAttnInterval = 4;
    pMl.mGetKey(cLlmKvFullAttentionInterval, vFullAttnInterval, false);
    for (uint32_t vI = 0; vI < vHparams.vNLayerAll; ++vI) {
      vHparams.aIsRecrImpl[vI] = (vI < vHparams.mNLayer()) && ((vI + 1) % vFullAttnInterval != 0);
    }
  }

  switch (vHparams.mNLayer()) {
  case 40:
    vType = cLlmType35bA3b;
    break;
  case 48:
    vType = cLlmType122bA10b;
    break;
  case 60:
    vType = cLlmType397bA17b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_qwen35moe::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = !pMl.vLoadMtp ? cTensorSkip : 0;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  auto vLoadBlockTrunk = [&](int pIl, int pFlags) {
    auto &vLayer = lLayers[pIl];

    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;
    const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFf;

    // Calculate dimensions from hyperparameters
    const int64_t cHeadKDim = vHparams.vSsmDState;
    const int64_t cHeadVDim = vHparams.vSsmDState;
    const int64_t cNKHeads = vHparams.vSsmNGroup;
    const int64_t cNVHeads = vHparams.vSsmDtRank;
    const int64_t cKeyDim = cHeadKDim * cNKHeads;
    const int64_t cValueDim = cHeadVDim * cNVHeads;
    const int64_t cConvDim = cKeyDim * 2 + cValueDim;

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pIl), {cNEmbd}, pFlags);
    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", pIl), {cNEmbd}, pFlags);

    if (!vHparams.mIsRecr(pIl)) {
      // Attention layers
      mCreateTensorQkv(vLayer, pIl, cNEmbd, cNEmbdHeadK * cNHead * 2, cNEmbdKGqa, cNEmbdVGqa, pFlags);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), {cNEmbdHeadK * cNHead, cNEmbd}, pFlags);

      // Q/K normalization for attention layers
      vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), {cNEmbdHeadK}, pFlags);
      vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), {cNEmbdHeadK}, pFlags);
    } else {
      // Linear attention (gated delta net) specific tensors
      // Create tensors with calculated dimensions
      vLayer.vWqkv =
          mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pIl), {cNEmbd, cKeyDim * 2 + cValueDim}, cTensorNotRequired);
      vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", pIl), {cNEmbd, cValueDim}, cTensorNotRequired);
      vLayer.vSsmConv1d =
          mCreateTensor(cTn(cLlmTensorSsmConv1d, "weight", pIl), {vHparams.vSsmDConv, cConvDim}, pFlags);
      vLayer.vSsmDt = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", pIl), {vHparams.vSsmDtRank}, pFlags);
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmANoscan, pIl), {vHparams.vSsmDtRank}, pFlags);
      vLayer.vSsmBeta = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", pIl), {cNEmbd, cNVHeads}, pFlags);
      vLayer.vSsmAlpha = mCreateTensor(cTn(cLlmTensorSsmAlpha, "weight", pIl), {cNEmbd, cNVHeads}, pFlags);
      vLayer.vSsmNorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", pIl), {cHeadVDim}, pFlags);
      vLayer.vSsmOut = mCreateTensor(cTn(cLlmTensorSsmOut, "weight", pIl), {cValueDim, cNEmbd}, pFlags);
    }

    // Routed experts
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pIl), {cNEmbd, cNExpert}, pFlags);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pIl), {cNFfExp, cNEmbd, cNExpert}, pFlags);
    mCreateTensorGateUpExps(vLayer, pIl, cNEmbd, cNFfExp, cNExpert, pFlags);

    // Shared experts
    vLayer.vFfnGateInpShexp = mCreateTensor(cTn(cLlmTensorFfnGateInpShexp, "weight", pIl), {cNEmbd}, pFlags);
    vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pIl), {cNEmbd, cNFfShexp}, pFlags);
    vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pIl), {cNEmbd, cNFfShexp}, pFlags);
    vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pIl), {cNFfShexp, cNEmbd}, pFlags);
  };

  auto vLoadBlockMtp = [&](int pIl) {
    auto &vLayer = lLayers[pIl];

    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;
    const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFf;

    // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pIl), {cNEmbd}, vMtpFlags);
    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", pIl), {cNEmbd}, vMtpFlags);

    mCreateTensorQkv(vLayer, pIl, cNEmbd, cNEmbdHeadK * cNHead * 2, cNEmbdKGqa, cNEmbdVGqa, vMtpFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), {cNEmbdHeadK * cNHead, cNEmbd}, vMtpFlags);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), {cNEmbdHeadK}, vMtpFlags);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), {cNEmbdHeadK}, vMtpFlags);

    // Routed experts
    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pIl), {cNEmbd, cNExpert}, vMtpFlags);
    vLayer.vFfnDownExps =
        mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pIl), {cNFfExp, cNEmbd, cNExpert}, vMtpFlags);
    mCreateTensorGateUpExps(vLayer, pIl, cNEmbd, cNFfExp, cNExpert, vMtpFlags);

    // Shared experts
    vLayer.vFfnGateInpShexp = mCreateTensor(cTn(cLlmTensorFfnGateInpShexp, "weight", pIl), {cNEmbd}, vMtpFlags);
    vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pIl), {cNEmbd, cNFfShexp}, vMtpFlags);
    vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pIl), {cNEmbd, cNFfShexp}, vMtpFlags);
    vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pIl), {cNFfShexp, cNEmbd}, vMtpFlags);

    // NextN-specific tensors that define the MTP block.
    vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", pIl), {2 * cNEmbd, cNEmbd}, vMtpFlags);
    vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", pIl), {cNEmbd}, vMtpFlags);
    vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", pIl), {cNEmbd}, vMtpFlags);
    vLayer.vNextn.vEmbedTokens = mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", pIl), {cNEmbd, cNVocab},
                                               vMtpFlags | cTensorNotRequired);
    vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", pIl), {cNEmbd, cNVocab},
                                                  vMtpFlags | cTensorNotRequired);
    vLayer.vNextn.vSharedHeadNorm =
        mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", pIl), {cNEmbd}, vMtpFlags | cTensorNotRequired);
  };

  for (int vI = 0; vI < cNLayer; ++vI) {
    vLoadBlockTrunk(vI, cTrunkFlags);
  }
  for (int vI = cNLayer; vI < cNLayerAll; ++vI) {
    vLoadBlockMtp(vI);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_qwen35moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_qwen35moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_delta_net_base(pParams), vModel(pModel) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  mCb(vInpL, "model.input_embed", -1);

  auto *vInp = mBuildInpMemHybrid();

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vRes->lTLayerInp[vIl] = vInpL;

    ggml_tensor *vInpSA = vInpL;

    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    ggml_build_forward_expand(vGf, vCur);

    // Determine layer type and build appropriate attention mechanism
    if (vHparams.mIsRecr(vIl)) {
      // Linear attention layer (gated delta net)
      vCur = mBuildLayerAttnLinear(vInp->mGetRecr(), vCur, vIl);
    } else {
      // Full attention layer
      vCur = mBuildLayerAttn(vInp->mGetAttn(), vCur, vInpPos, aSections, vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds && vCparams.vEmbeddingsNextnMasked) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    // Residual connection
    vCur = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vCur, "attn_residual", vIl);

    // Save the tensor before post-attention norm for residual connection
    ggml_tensor *vFfnResidual = vCur;

    // Post-attention norm
    ggml_tensor *vAttnPostNorm = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnPostNorm, nullptr, cLlmNormRms, vIl);
    mCb(vAttnPostNorm, "attn_post_norm", vIl);

    // MOE FFN layer
    vCur = mBuildLayerFfn(vAttnPostNorm, vIl);
    mCb(vCur, "ffn_out", vIl);

    // Residual connection for FFN - add to the tensor from before post_attention_layernorm
    vCur = ggml_add(vCtx0, vCur, vFfnResidual);
    mCb(vCur, "post_moe", vIl);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // Input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  // post-norm hidden state feeds both the LM head and the MTP seed below
  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (!vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // LM head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

std::pair<ggml_tensor *, ggml_tensor *> mgs_model_qwen35moe::graph::mBuildQkvz(ggml_tensor *pInput, int pIl) {
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  ggml_tensor *vQkvMixed = mBuildLoraMm(vModel.lLayers[pIl].vWqkv, pInput, vModel.lLayers[pIl].vWqkvS);
  vQkvMixed = ggml_reshape_3d(vCtx0, vQkvMixed, vQkvMixed->ne[0], cNSeqTokens, cNSeqs);
  mCb(vQkvMixed, "linear_attn_qkv_mixed", pIl);

  ggml_tensor *vZ = mBuildLoraMm(vModel.lLayers[pIl].vWqkvGate, pInput, vModel.lLayers[pIl].vWqkvGateS);
  mCb(vZ, "z", pIl);

  return {vQkvMixed, vZ};
}

ggml_tensor *mgs_model_qwen35moe::graph::mBuildNormGated(ggml_tensor *pInput, ggml_tensor *pWeights, ggml_tensor *pGate,
                                                         int pLayer) {
  ggml_tensor *vNormalized = mBuildNorm(pInput, pWeights, nullptr, cLlmNormRms, pLayer);
  ggml_tensor *vGatedSilu = ggml_silu(vCtx0, pGate);

  return ggml_mul(vCtx0, vNormalized, vGatedSilu);
}

ggml_tensor *mgs_model_qwen35moe::graph::mBuildLayerAttn(llm_graph_input_attn_kv *pInp, ggml_tensor *pCur,
                                                         ggml_tensor *pInpPos, int *pSections, int pIl) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

  // Qwen3Next uses a single Q projection that outputs query + gate
  ggml_tensor *vQcurFull =
      mBuildLoraMm(vModel.lLayers[pIl].vWq, pCur, vModel.lLayers[pIl].vWqS); // [ (n_embd_head * 2) * n_head, n_tokens ]
  mCb(vQcurFull, "Qcur_full", pIl);

  ggml_tensor *vQcur =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, 0);
  mCb(vQcur, "Qcur_reshaped", pIl);

  // Apply Q normalization
  vQcur = mBuildNorm(vQcur, vModel.lLayers[pIl].vAttnQNorm, nullptr, cLlmNormRms, pIl);
  mCb(vQcur, "Qcur_normed", pIl);

  ggml_tensor *vKcur = mBuildLoraMm(vModel.lLayers[pIl].vWk, pCur, vModel.lLayers[pIl].vWkS);
  mCb(vKcur, "Kcur", pIl);

  ggml_tensor *vVcur = mBuildLoraMm(vModel.lLayers[pIl].vWv, pCur, vModel.lLayers[pIl].vWvS);
  mCb(vVcur, "Vcur", pIl);

  // Apply K normalization
  vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
  vKcur = mBuildNorm(vKcur, vModel.lLayers[pIl].vAttnKNorm, nullptr, cLlmNormRms, pIl);
  mCb(vKcur, "Kcur_normed", pIl);

  ggml_tensor *vGate =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, ggml_element_size(vQcurFull) * cNEmbdHead);
  vGate = ggml_cont_2d(vCtx0, vGate, cNEmbdHead * cNHead, cNTokens);
  mCb(vGate, "gate_reshaped", pIl);

  vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);

  // Apply IMRoPE
  vQcur = ggml_rope_multi(vCtx0, vQcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  vKcur = ggml_rope_multi(vCtx0, vKcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "Qcur", pIl);
  mCb(vKcur, "Kcur", pIl);
  mCb(vVcur, "Vcur", pIl);

  // Attention computation
  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  pCur = mBuildAttn(pInp, nullptr, nullptr, nullptr, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  mCb(pCur, "attn_pregate", pIl);

  ggml_tensor *vGateSigmoid = ggml_sigmoid(vCtx0, vGate);
  mCb(vGateSigmoid, "gate_sigmoid", pIl);

  pCur = ggml_mul(vCtx0, pCur, vGateSigmoid);
  mCb(pCur, "attn_gated", pIl);

  pCur = mBuildLoraMm(vModel.lLayers[pIl].vWo, pCur, vModel.lLayers[pIl].vWoS);
  mCb(pCur, "attn_output", pIl);

  return pCur;
}

ggml_tensor *mgs_model_qwen35moe::graph::mBuildLayerAttnLinear(llm_graph_input_rs *pInp, ggml_tensor *pCur, int pIl) {
  const auto *vMctxCur = pInp->vMctx;

  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cHeadKDim = vHparams.vSsmDState;
  const int64_t cNumKHeads = vHparams.vSsmNGroup;
  const int64_t cNumVHeads = vHparams.vSsmDtRank;
  const int64_t cHeadVDim = cDInner / cNumVHeads;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

  // Input projections
  auto vQkvz = mBuildQkvz(pCur, pIl);
  ggml_tensor *vQkvMixed = vQkvz.first;
  ggml_tensor *vZ = vQkvz.second;

  ggml_tensor *vBeta = mBuildLoraMm(vModel.lLayers[pIl].vSsmBeta, pCur, vModel.lLayers[pIl].vSsmBetaS);
  vBeta = ggml_reshape_4d(vCtx0, vBeta, 1, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vBeta, "beta", pIl);

  vBeta = ggml_sigmoid(vCtx0, vBeta);
  mCb(vBeta, "beta_sigmoid", pIl);

  ggml_tensor *vAlpha = mBuildLoraMm(vModel.lLayers[pIl].vSsmAlpha, pCur, vModel.lLayers[pIl].vSsmAlphaS);
  vAlpha = ggml_reshape_3d(vCtx0, vAlpha, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vAlpha, "alpha", pIl);

  ggml_tensor *vAlphaBiased = ggml_add(vCtx0, vAlpha, vModel.lLayers[pIl].vSsmDt);
  ggml_tensor *vAlphaSoftplus = ggml_softplus(vCtx0, vAlphaBiased);
  mCb(vAlphaSoftplus, "a_softplus", pIl);

  ggml_tensor *vGate = ggml_mul(vCtx0, vAlphaSoftplus, vModel.lLayers[pIl].vSsmA); // -A_log.exp() * softplus
  mCb(vGate, "gate", pIl);

  vGate = ggml_reshape_4d(vCtx0, vGate, 1, cNumVHeads, cNSeqTokens, cNSeqs);

  ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(pIl);

  ggml_tensor *vConvKernel = vModel.lLayers[pIl].vSsmConv1d;
  const int64_t cConvKernelSize = vConvKernel->ne[0];
  const int64_t cConvChannels = cDInner + 2 * vHparams.vSsmNGroup * vHparams.vSsmDState;

  ggml_tensor *vConvInput = mBuildConvState(pInp, vConvStatesAll, vQkvMixed, cConvKernelSize, cConvChannels, pIl);

  ggml_tensor *vState = mBuildRs(pInp, vSsmStatesAll, vHparams.mNEmbdS(), cNSeqs);
  vState = ggml_reshape_4d(vCtx0, vState, cHeadVDim, cHeadVDim, cNumVHeads, cNSeqs);
  mCb(vState, "state_predelta", pIl);

  ggml_tensor *vConvOutputProper = ggml_ssm_conv(vCtx0, vConvInput, vConvKernel);
  mCb(vConvOutputProper, "conv_output_raw", pIl);

  ggml_tensor *vConvOutputSilu = ggml_silu(vCtx0, vConvOutputProper);
  mCb(vConvOutputSilu, "conv_output_silu", pIl);

  ggml_tensor *vConvQkvMix = vConvOutputSilu;

  // Calculate the total conv dimension
  int64_t vQkvDim = cHeadKDim * cNumKHeads * 2 + cHeadVDim * cNumVHeads;
  int64_t vNb1Qkv = ggml_row_size(vConvQkvMix->type, vQkvDim);

  // Extract the convolved Q, K, V from conv_output
  ggml_tensor *vQConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens, 0);

  ggml_tensor *vKConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                     cHeadKDim * cNumKHeads * ggml_element_size(vConvQkvMix));

  ggml_tensor *vVConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadVDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                     ggml_row_size(vConvQkvMix->type, 2 * cHeadKDim * cNumKHeads));

  mCb(vQConv, "q_conv", pIl);
  mCb(vKConv, "k_conv", pIl);
  mCb(vVConv, "v_conv", pIl);

  const float cEpsNorm = vHparams.vFNormRmsEps;

  vQConv = ggml_l2_norm(vCtx0, vQConv, cEpsNorm);
  vKConv = ggml_l2_norm(vCtx0, vKConv, cEpsNorm);

  // vQConv = ggml_cont_4d(vCtx0, vQConv, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs);
  // vKConv = ggml_cont_4d(vCtx0, vKConv, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs);
  // vVConv = ggml_cont_4d(vCtx0, vVConv, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs);

  // if head keys and value keys are different, repeat to force tensors into matching shapes
  // note: need explicit repeat only if we are not using the fused GDN.
  if (cNumKHeads != cNumVHeads && (!vCparams.vFusedGdnAr || !vCparams.vFusedGdnCh)) {
    GGML_ASSERT(cNumVHeads % cNumKHeads == 0);
    vQConv = ggml_repeat_4d(vCtx0, vQConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
    vKConv = ggml_repeat_4d(vCtx0, vKConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
  }

  mCb(vQConv, "q_conv_predelta", pIl);
  mCb(vKConv, "k_conv_predelta", pIl);
  mCb(vVConv, "v_conv_predelta", pIl);

  ggml_tensor *vOutput = mBuildRecurrentAttn(pInp, vSsmStatesAll, vQConv, vKConv, vVConv, vGate, vBeta, vState, pIl);

  // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
  ggml_tensor *vZ2d = ggml_reshape_4d(vCtx0, vZ, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs);

  // Apply gated normalization: self.norm(core_attn_out, z)
  ggml_tensor *vAttnOutNorm = mBuildNormGated(vOutput, vModel.lLayers[pIl].vSsmNorm, vZ2d, pIl);

  // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
  ggml_tensor *vFinalOutput = ggml_reshape_3d(vCtx0, vAttnOutNorm, cHeadVDim * cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vFinalOutput, "final_output", pIl);

  // Output projection
  pCur = mBuildLoraMm(vModel.lLayers[pIl].vSsmOut, vFinalOutput, vModel.lLayers[pIl].vSsmOutS);
  mCb(pCur, "linear_attn_out", pIl);

  // Reshape back to original dimensions
  pCur = ggml_reshape_2d(vCtx0, pCur, cNEmbd, cNSeqTokens * cNSeqs);

  return pCur;
}

ggml_tensor *mgs_model_qwen35moe::graph::mBuildLayerFfn(ggml_tensor *pCur, const int pIl) {
  // Check if this is an MoE layer
  GGML_ASSERT(vModel.lLayers[pIl].vFfnGateInp != nullptr);

  ggml_tensor *vMoeOut = mBuildMoeFfn(
      pCur, vModel.lLayers[pIl].vFfnGateInp, vModel.lLayers[pIl].vFfnUpExps, vModel.lLayers[pIl].vFfnGateExps,
      vModel.lLayers[pIl].vFfnDownExps, nullptr, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
      vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, pIl, nullptr, vModel.lLayers[pIl].vFfnGateUpExps,
      vModel.lLayers[pIl].vFfnUpExpsS, vModel.lLayers[pIl].vFfnGateExpsS, vModel.lLayers[pIl].vFfnDownExpsS);
  mCb(vMoeOut, "ffn_moe_out", pIl);

  // Add shared experts if present - following Qwen3Next reference implementation
  if (vModel.lLayers[pIl].vFfnUpShexp != nullptr) {
    ggml_tensor *vFfnShexp = mBuildFfn(pCur, vModel.lLayers[pIl].vFfnUpShexp, NULL, vModel.lLayers[pIl].vFfnUpShexpS,
                                       vModel.lLayers[pIl].vFfnGateShexp, NULL, vModel.lLayers[pIl].vFfnGateShexpS,
                                       vModel.lLayers[pIl].vFfnDownShexp, NULL, vModel.lLayers[pIl].vFfnDownShexpS,
                                       NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
    mCb(vFfnShexp, "ffn_shexp", pIl);

    // Apply shared expert gating as in the reference implementation
    // The shared expert has its own gate that is sigmoided
    // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
    ggml_tensor *vSharedGate = mBuildLoraMm(vModel.lLayers[pIl].vFfnGateInpShexp, pCur);
    mCb(vSharedGate, "shared_expert_gate", pIl);

    // Apply sigmoid to the gate
    vSharedGate = ggml_sigmoid(vCtx0, vSharedGate);
    mCb(vSharedGate, "shared_expert_gate_sigmoid", pIl);

    // Apply the gate to the shared expert output
    vFfnShexp = ggml_mul(vCtx0, vFfnShexp, vSharedGate);
    mCb(vFfnShexp, "ffn_shexp_gated", pIl);

    pCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
    mCb(pCur, "ffn_out", pIl);
  } else {
    pCur = vMoeOut;
  }

  return pCur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
mgs_model_qwen35moe::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "QWEN35MOE MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  const int cIl = vHparams.mNLayer();
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");
  GGML_ASSERT(vLayer.vFfnGateInp && "MTP block missing ffn_gate_inp");

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

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

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vHNorm = mBuildNorm(vHEmbd, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
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

  ggml_tensor *vQcurFull = mBuildLoraMm(vLayer.vWq, vCur, vLayer.vWqS);
  mCb(vQcurFull, "mtp_Qcur_full", cIl);

  ggml_tensor *vQcur =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, 0);
  vQcur = mBuildNorm(vQcur, vLayer.vAttnQNorm, nullptr, cLlmNormRms, cIl);
  mCb(vQcur, "mtp_Qcur_normed", cIl);

  ggml_tensor *vGate =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, ggml_element_size(vQcurFull) * cNEmbdHead);
  vGate = ggml_cont_2d(vCtx0, vGate, cNEmbdHead * cNHead, cNTokens);
  mCb(vGate, "mtp_gate", cIl);

  ggml_tensor *vKcur = mBuildLoraMm(vLayer.vWk, vCur, vLayer.vWkS);
  vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
  vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, nullptr, cLlmNormRms, cIl);
  mCb(vKcur, "mtp_Kcur_normed", cIl);

  ggml_tensor *vVcur = mBuildLoraMm(vLayer.vWv, vCur, vLayer.vWvS);
  vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);
  mCb(vVcur, "mtp_Vcur", cIl);

  vQcur = ggml_rope_multi(vCtx0, vQcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  vKcur = ggml_rope_multi(vCtx0, vKcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  vCur = mBuildAttn(vInpAttn, nullptr, nullptr, nullptr, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, cIl);
  mCb(vCur, "mtp_attn_pregate", cIl);

  vCur = ggml_mul(vCtx0, vCur, ggml_sigmoid(vCtx0, vGate));
  vCur = mBuildLoraMm(vLayer.vWo, vCur, vLayer.vWoS);
  mCb(vCur, "mtp_attn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vCur, "mtp_attn_residual", cIl);

  ggml_tensor *vFfnResidual = vCur;
  vCur = mBuildNorm(vCur, vLayer.vAttnPostNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_post_norm", cIl);

  // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
  ggml_tensor *vMoeOut = mBuildMoeFfn(
      vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps, nullptr, cNExpert,
      cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, cIl, nullptr,
      vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
  mCb(vMoeOut, "mtp_ffn_moe_out", cIl);

  if (vLayer.vFfnUpShexp != nullptr) {
    ggml_tensor *vFfnShexp = mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp,
                                       nullptr, vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp, nullptr,
                                       vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
    mCb(vFfnShexp, "mtp_ffn_shexp", cIl);

    ggml_tensor *vSharedGate = mBuildLoraMm(vLayer.vFfnGateInpShexp, vCur);
    vSharedGate = ggml_sigmoid(vCtx0, vSharedGate);
    mCb(vSharedGate, "mtp_shared_expert_gate_sigmoid", cIl);

    vFfnShexp = ggml_mul(vCtx0, vFfnShexp, vSharedGate);
    mCb(vFfnShexp, "mtp_ffn_shexp_gated", cIl);

    vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
  } else {
    vCur = vMoeOut;
  }
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnResidual);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
