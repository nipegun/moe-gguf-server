#include "models.h"
#include "mgs-memory-recurrent.h"

void mgs_model_qwen35::mLoadArchHparams(mgs_model_loader &pMl) {
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
  case 24:
    vType = vHparams.vNEmbd == 1024 ? cLlmType0Point8B : cLlmType2b;
    break;
  case 32:
    vType = vHparams.vNEmbd == 2560 ? cLlmType4b : cLlmType9b;
    break;
  case 64:
    vType = cLlmType27b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_qwen35::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  int vMtpFlags = !pMl.vLoadMtp ? cTensorSkip : 0;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  auto vLoadBlockMtp = [&](int pIl) {
    auto &vLayer = lLayers[pIl];

    // MTP block looks like a full-attention Qwen3.5 decoder block.
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pIl), {cNEmbd}, vMtpFlags);
    vLayer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", pIl), {cNEmbd}, vMtpFlags);

    mCreateTensorQkv(vLayer, pIl, cNEmbd, cNEmbdHeadK * cNHead * 2, cNEmbdKGqa, cNEmbdVGqa, vMtpFlags);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), {cNEmbdHeadK * cNHead, cNEmbd}, vMtpFlags);
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), {cNEmbdHeadK}, vMtpFlags);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), {cNEmbdHeadK}, vMtpFlags);

    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pIl), {cNEmbd, cNFf}, vMtpFlags);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pIl), {cNFf, cNEmbd}, vMtpFlags);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pIl), {cNEmbd, cNFf}, vMtpFlags);

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

  for (int vI = cNLayer; vI < cNLayerAll; ++vI) {
    vLoadBlockMtp(vI);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_qwen35::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  throw std::runtime_error("Qwen35 is available only as an external MTP head for a MoE target");
}

// External MTP head for a compatible MoE target.
mgs_model_qwen35::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  // hparams.mNLayer() includes both main model layers and MTP layers. The MTP
  // layer is stored immediately after the main layers in model.lLayers[].
  const int cIl = vHparams.mNLayer();
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");

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

  vCur = mBuildFfn(vCur, vLayer.vFfnUp, nullptr, vLayer.vFfnUpS, vLayer.vFfnGate, nullptr, vLayer.vFfnGateS,
                   vLayer.vFfnDown, nullptr, vLayer.vFfnDownS, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnResidual);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
