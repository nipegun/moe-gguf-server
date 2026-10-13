#include "models.h"

void mgs_model_ernie4_5_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvInterleaveMoeLayerStep, vHparams.vNMoeLayerStep);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);

  switch (vHparams.mNLayer()) {
  case 28:
    vType = cLlmType21bA3b;
    break;
  case 54:
    vType = cLlmType300bA47b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_ernie4_5_moe::mLoadArchTensors(mgs_model_loader &) {
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

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    // optional bias tensors
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (static_cast<uint32_t>(vI) >= vHparams.vNLayerDenseLead) { // MoE layers
      int vNFfExp = vHparams.vNFfExp;

      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cTensorNotRequired);
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, vNFfExp, cNExpert}, cTensorNotRequired);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {vNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, vNFfExp, cNExpert}, 0);

      // Shared expert (if present)
      if (vHparams.vNFfShexp > 0) {
        vLayer.vFfnGateShexp =
            mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnDownShexp =
            mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {vHparams.vNFfShexp, cNEmbd}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
      }
    } else { // Dense layers
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_ernie4_5_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_ernie4_5_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  GGML_ASSERT(cNEmbdHead == cNRot);

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  auto *vInpAttn = mBuildAttnInpKv();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  GGML_ASSERT(vHparams.vNMoeLayerStep > 0 && "Ernie 4.5 MoE requires n_moe_layer_step > 0");
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;
    // norm
    {
      vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "attn_norm", vIl);
    }
    // self-attention
    {
      // compute Q and K and RoPE them
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                        nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
      mCb(vCur, "attn_out", vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // feed-forward network
    bool vIsMoeLayer =
        static_cast<uint32_t>(vIl) >= vHparams.vNLayerDenseLead && (vIl + 1) % vHparams.vNMoeLayerStep == 0;

    if (!vIsMoeLayer) {
      vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "ffn_norm", vIl);

      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "ffn_norm", vIl);

      ggml_tensor *vMoeOut = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps,
                                          pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
                                          vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // Shared expert (if present)
      if (vHparams.vNFfShexp > 0) {
        ggml_tensor *vFfnShexp =
            mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                      pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      } else {
        vCur = vMoeOut;
      }
      mCb(vCur, "ffn_out", vIl);
    }
    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    mCb(vCur, "ffn_out", vIl);

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
