#include "models.h"

void mgs_model_granite_hybrid::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLogitScale, vHparams.vFLogitScale, /* required */ false);
  pMl.mGetKey(cLlmKvResidualScale, vHparams.vFResidualScale, /* required */ false);
  pMl.mGetKey(cLlmKvEmbeddingScale, vHparams.vFEmbeddingScale, /* required */ false);
  pMl.mGetKey(cLlmKvAttentionScale, vHparams.vFAttentionScale, /* required */ false);

  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  pMl.mGetKey(cLlmKvSsmStateSize, vHparams.vSsmDState);
  pMl.mGetKey(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  pMl.mGetKey(cLlmKvSsmGroupCount, vHparams.vSsmNGroup);

  // Granite uses rope_finetuned as a switch for rope
  bool vRopeFinetuned = false;
  pMl.mGetKey(cLlmKvRopeScalingFinetuned, vRopeFinetuned);
  vHparams.vRopeFinetuned = vRopeFinetuned;

  // A layer is recurrent IFF the n_head_kv value is set to 0
  for (uint32_t vI = 0; vI < vHparams.mNLayer(); ++vI) {
    vHparams.aIsRecrImpl[vI] = vHparams.mNHeadKv(vI) == 0;
  }

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  switch (vHparams.vNEmbd) {
  case 768:
    vType = cLlmType350m;
    break;
  case 1536:
    vType = (vHparams.mNFf() == 512 ? cLlmType7bA1b : cLlmType1b);
    break;
  case 2048:
  case 2560:
    vType = cLlmType3b;
    break;
  case 4096:
    vType = cLlmType32b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }

  // For Granite MoE Shared
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, /* required */ false);
}

void mgs_model_granite_hybrid::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  // mamba2 Mixer SSM params
  // NOTE: int64_t for tensor dimensions
  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cDState = vHparams.vSsmDState;
  const int64_t cNSsmHead = vHparams.vSsmDtRank;
  const int64_t cNGroup = vHparams.vSsmNGroup;
  const int64_t cDInProj = 2 * cDInner + 2 * cNGroup * cDState + cNSsmHead;

  // only an expansion factor of 2 is supported for now
  GGML_ASSERT(2 * cNEmbd == cDInner);

  // embeddings
  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  {
    vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
    vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
    // if output is NULL, init from the input tok embed, duplicated to allow offloading
    if (vOutput == NULL) {
      vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
    }
  }

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    // norm
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    if (vHparams.mIsRecr(vI)) {
      // ssm layers
      vLayer.vSsmIn = mCreateTensor(cTn(cLlmTensorSsmIn, "weight", vI), {cNEmbd, cDInProj}, 0);

      vLayer.vSsmConv1d =
          mCreateTensor(cTn(cLlmTensorSsmConv1d, "weight", vI), {cDConv, cDInner + 2 * cNGroup * cDState}, 0);
      vLayer.vSsmConv1dB =
          mCreateTensor(cTn(cLlmTensorSsmConv1d, "bias", vI), {cDInner + 2 * cNGroup * cDState}, cTensorNotRequired);

      vLayer.vSsmDtB = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", vI), {cNSsmHead}, 0);

      // no "weight" suffix for these
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmA, vI), {1, cNSsmHead}, 0);
      vLayer.vSsmD = mCreateTensor(cTn(cLlmTensorSsmD, vI), {1, cNSsmHead}, 0);

      vLayer.vSsmNorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", vI), {cDInner / cNGroup, cNGroup}, 0);

      // out_proj
      vLayer.vSsmOut = mCreateTensor(cTn(cLlmTensorSsmOut, "weight", vI), {cDInner, cNEmbd}, 0);
    } else {
      // attention layers (with optional bias)
      const int64_t cNHeadI = vHparams.mNHead(vI);
      const int64_t cNEmbdKGqaI = vHparams.mNEmbdKGqa(vI);
      const int64_t cNEmbdVGqaI = vHparams.mNEmbdVGqa(vI);
      mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHeadI, cNEmbdKGqaI, cNEmbdVGqaI, 0);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHeadI, cNEmbd}, 0);
      vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);
    }

    // feed forward (w/ optional biases)
    if (cNExpert > 0) {
      // MoE FFN
      vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
      vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, cTensorNotRequired);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);

      // For Granite MoE Shared
      if (vHparams.vNFfShexp > 0) {
        vLayer.vFfnGateShexp =
            mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, vHparams.vNFfShexp}, 0);
        vLayer.vFfnDownShexp =
            mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {vHparams.vNFfShexp, cNEmbd}, 0);
      }
    } else {
      vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
      vLayer.vRopeFreqs = mCreateTensor(cTn(cLlmTensorRopeFreqs, "weight", vI), {cNRot / 2},
                                        cTensorNotRequired | (vI != 0 ? cTensorDuplicated : 0));
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnGateB = mCreateTensor(cTn(cLlmTensorFfnGate, "bias", vI), {cNFf}, cTensorNotRequired);
      vLayer.vFfnDownB = mCreateTensor(cTn(cLlmTensorFfnDown, "bias", vI), {cNEmbd}, cTensorNotRequired);
      vLayer.vFfnUpB = mCreateTensor(cTn(cLlmTensorFfnUp, "bias", vI), {cNFf}, cTensorNotRequired);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_granite_hybrid::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_granite_hybrid::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_mamba_base(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  auto *vInp = mBuildInpMemHybrid();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // Positional embeddings populated if rope enabled
  ggml_tensor *vInpPos = nullptr;
  if (vHparams.vRopeFinetuned) {
    vInpPos = mBuildInpPos();
  }

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    struct ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    if (vHparams.mIsRecr(vIl)) {
      // ssm layer //
      vCur = mBuildMamba2Layer(vInp->mGetRecr(), vCur, pModel, vUbatch, vIl);
    } else {
      // attention layer //
      vCur = mBuildAttentionLayer(vCur, vInpPos, vInp->mGetAttn(), pModel, cNEmbdHead, vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    // ffn
    vCur = mBuildLayerFfn(vCur, vInpSA, pModel, vIl);

    // input for next layer
    vInpL = vCur;
  }

  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  // For Granite architectures - scale logits
  if (vHparams.vFLogitScale) {
    vCur = ggml_scale(vCtx0, vCur, 1.0f / vHparams.vFLogitScale);
  }
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

ggml_tensor *mgs_model_granite_hybrid::graph::mBuildAttentionLayer(ggml_tensor *pCur, ggml_tensor *pInpPos,
                                                                   llm_graph_input_attn_kv *pInpAttn,
                                                                   const mgs_model &pModel, const int64_t pNEmbdHead,
                                                                   const int pIl) {
  auto [vQcur, vKcur, vVcur] =
      mBuildQkv(pModel.lLayers[pIl], pCur, pNEmbdHead, vHparams.mNHead(pIl), vHparams.mNHeadKv(pIl), pIl);

  const bool cUseRope = vHparams.vRopeFinetuned;
  if (cUseRope) {
    ggml_tensor *vRopeFactors = pModel.mGetRopeFactors(vCparams, pIl);
    vQcur = ggml_rope_ext(vCtx0, vQcur, pInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

    vKcur = ggml_rope_ext(vCtx0, vKcur, pInpPos, vRopeFactors, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  }

  mCb(vQcur, "Qcur", pIl);
  mCb(vKcur, "Kcur", pIl);
  mCb(vVcur, "Vcur", pIl);

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(pNEmbdHead)) : vHparams.vFAttentionScale;
  pCur = mBuildAttn(pInpAttn, pModel.lLayers[pIl].vWo, pModel.lLayers[pIl].vWoB, pModel.lLayers[pIl].vWoS, vQcur, vKcur,
                    vVcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  mCb(pCur, "attn_out", pIl);
  return pCur;
}

ggml_tensor *mgs_model_granite_hybrid::graph::mBuildLayerFfn(ggml_tensor *pCur, ggml_tensor *pInpSA,
                                                             const mgs_model &pModel, const int pIl) {
  // For Granite architectures - scale residual
  if (vHparams.vFResidualScale) {
    pCur = ggml_scale(vCtx0, pCur, vHparams.vFResidualScale);
  }
  ggml_tensor *vFfnInp = ggml_add(vCtx0, pCur, pInpSA);
  mCb(vFfnInp, "ffn_inp", pIl);

  // feed-forward network (non-MoE)
  if (pModel.lLayers[pIl].vFfnGateInp == nullptr) {
    pCur = mBuildNorm(vFfnInp, pModel.lLayers[pIl].vFfnNorm, NULL, cLlmNormRms, pIl);
    mCb(pCur, "ffn_norm", pIl);

    pCur = mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUp, pModel.lLayers[pIl].vFfnUpB, NULL, pModel.lLayers[pIl].vFfnGate,
                     pModel.lLayers[pIl].vFfnGateB, NULL, pModel.lLayers[pIl].vFfnDown, pModel.lLayers[pIl].vFfnDownB,
                     NULL, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
    mCb(pCur, "ffn_out", pIl);

  } else {
    // MoE branch
    pCur = mBuildNorm(vFfnInp, pModel.lLayers[pIl].vFfnNorm, NULL, cLlmNormRms, pIl);
    mCb(pCur, "ffn_norm", pIl);

    ggml_tensor *vMoeOut = mBuildMoeFfn(pCur, pModel.lLayers[pIl].vFfnGateInp, pModel.lLayers[pIl].vFfnUpExps,
                                        pModel.lLayers[pIl].vFfnGateExps, pModel.lLayers[pIl].vFfnDownExps, nullptr,
                                        cNExpert, cNExpertUsed, cLlmFfnSilu, true, vHparams.vExpertWeightsScale,
                                        cMgsExpertGatingFuncTypeSoftmax, pIl);
    mCb(vMoeOut, "ffn_moe_out", pIl);

    // For Granite MoE Shared
    if (vHparams.vNFfShexp > 0) {
      ggml_tensor *vFfnShexp =
          mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[pIl].vFfnGateShexp, NULL, NULL,
                    pModel.lLayers[pIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
      mCb(vFfnShexp, "ffn_shexp", pIl);

      pCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(pCur, "ffn_out", pIl);
    } else {
      pCur = vMoeOut;
    }
  }

  // For Granite architectures - scale residual
  if (vHparams.vFResidualScale) {
    pCur = ggml_scale(vCtx0, pCur, vHparams.vFResidualScale);
  }
  pCur = ggml_add(vCtx0, pCur, vFfnInp);
  mCb(pCur, "ffn_out", pIl);

  pCur = mBuildCvec(pCur, pIl);
  mCb(pCur, "l_out", pIl);

  return pCur;
}
