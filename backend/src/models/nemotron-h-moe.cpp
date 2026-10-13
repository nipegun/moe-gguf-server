#include "models.h"

void mgs_model_nemotron_h_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  pMl.mGetKey(cLlmKvSsmStateSize, vHparams.vSsmDState);
  pMl.mGetKey(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  pMl.mGetKey(cLlmKvSsmGroupCount, vHparams.vSsmNGroup);

  // A layer is recurrent IFF the n_head_kv value is set to 0 and
  // the n_ff value is set to 0
  for (uint32_t vI = 0; vI < vHparams.mNLayer(); ++vI) {
    vHparams.aIsRecrImpl[vI] = (vHparams.mNHeadKv(vI) == 0 && vHparams.mNFf(vI) == 0);
  }

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvMoeLatentSize, vHparams.vMoeLatentSize, false);

  switch (vHparams.mNLayer()) {
  case 52:
    vType = cLlmType31bA3Point5B;
    break; // Nemotron-H_MOE 31B
  case 56:
    vType = cLlmType9b;
    break;
  case 88:
    vType = cLlmType120bA12b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_nemotron_h_moe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  // mamba2 Mixer SSM params
  // NOTE: int64_t for tensor dimensions
  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cDState = vHparams.vSsmDState;
  const int64_t cNSsmHead = vHparams.vSsmDtRank;
  const int64_t cNGroup = vHparams.vSsmNGroup;
  const int64_t cDInProj = 2 * cDInner + 2 * cNGroup * cDState + cNSsmHead;
  const int64_t cMoeNEmbd = vHparams.vMoeLatentSize > 0 ? vHparams.vMoeLatentSize : cNEmbd;

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

    // all blocks use the attn norm
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
    } else if (vHparams.mNFf(vI) == 0) {
      // attention layers (with optional bias)
      const int64_t cNHeadI = vHparams.mNHead(vI);
      const int64_t cNEmbdKGqaI = vHparams.mNEmbdKGqa(vI);
      const int64_t cNEmbdVGqaI = vHparams.mNEmbdVGqa(vI);
      mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHeadI, cNEmbdKGqaI, cNEmbdVGqaI, 0);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHeadI, cNEmbd}, 0);
      vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);
    } else {
      if (cNExpert != 0) {
        const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;
        const int64_t cNFfShexp = vHparams.vNFfShexp;

        vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
        vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);

        // MoE branch
        vLayer.vFfnLatentDown =
            mCreateTensor(cTn(cLlmTensorFfnLatentDown, "weight", vI), {cNEmbd, cMoeNEmbd}, cTensorNotRequired);
        vLayer.vFfnLatentUp =
            mCreateTensor(cTn(cLlmTensorFfnLatentUp, "weight", vI), {cMoeNEmbd, cNEmbd}, cTensorNotRequired);

        vLayer.vFfnDownExps =
            mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cMoeNEmbd, cNExpert}, 0);
        vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cMoeNEmbd, cNFfExp, cNExpert}, 0);

        // Shared expert branch
        vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexp, cNEmbd}, 0);
        vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexp}, 0);

      } else {
        // mlp layers
        vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {vHparams.mNFf(vI), cNEmbd}, 0);
        vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, vHparams.mNFf(vI)}, 0);
        vLayer.vFfnDownB = mCreateTensor(cTn(cLlmTensorFfnDown, "bias", vI), {cNEmbd}, cTensorNotRequired);
        vLayer.vFfnUpB = mCreateTensor(cTn(cLlmTensorFfnUp, "bias", vI), {vHparams.mNFf(vI)}, cTensorNotRequired);
      }
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_nemotron_h_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_nemotron_h_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_mamba_base(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  ggml_build_forward_expand(vGf, vInpL);

  auto *vInp = mBuildInpMemHybrid();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    struct ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    if (vHparams.mIsRecr(vIl)) {
      // ssm layer //
      vCur = mBuildMamba2Layer(vInp->mGetRecr(), vCur, pModel, vUbatch, vIl);
    } else if (vHparams.mNFf(vIl) == 0) {
      // attention layer //
      vCur = mBuildAttentionLayer(vCur, vInp->mGetAttn(), pModel, cNEmbdHead, vIl);
    } else {
      vCur = mBuildFfnLayer(vCur, pModel, vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    // add residual
    vCur = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vCur, "nemotron_h_block_out", vIl);

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

ggml_tensor *mgs_model_nemotron_h_moe::graph::mBuildAttentionLayer(ggml_tensor *pCur, llm_graph_input_attn_kv *pInpAttn,
                                                                   const mgs_model &pModel, int64_t pNEmbdHead,
                                                                   int pIl) {
  auto [vQcur, vKcur, vVcur] =
      mBuildQkv(pModel.lLayers[pIl], pCur, pNEmbdHead, vHparams.mNHead(pIl), vHparams.mNHeadKv(pIl), pIl);

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(pNEmbdHead)) : vHparams.vFAttentionScale;
  pCur = mBuildAttn(pInpAttn, pModel.lLayers[pIl].vWo, pModel.lLayers[pIl].vWoB, pModel.lLayers[pIl].vWoS, vQcur, vKcur,
                    vVcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  mCb(pCur, "attn_out", pIl);
  return pCur;
}

ggml_tensor *mgs_model_nemotron_h_moe::graph::mBuildFfnLayer(ggml_tensor *pCur, const mgs_model &pModel, int pIl) {
  if (pModel.lLayers[pIl].vFfnGateInp == nullptr) {
    pCur = mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUp, pModel.lLayers[pIl].vFfnUpB, pModel.lLayers[pIl].vFfnUpS, NULL,
                     NULL, NULL, pModel.lLayers[pIl].vFfnDown, pModel.lLayers[pIl].vFfnDownB,
                     pModel.lLayers[pIl].vFfnDownS, NULL, cLlmFfnReluSqr, cLlmFfnPar, pIl);
    mCb(pCur, "ffn_out", pIl);
  } else {
    ggml_tensor *vInpEmb = pCur;
    ggml_tensor *vInpLatent = pCur;

    if (pModel.lLayers[pIl].vFfnLatentDown) {
      vInpLatent = ggml_mul_mat(vCtx0, pModel.lLayers[pIl].vFfnLatentDown, pCur);
    }

    ggml_tensor *vRouterLogits = mBuildLoraMm(pModel.lLayers[pIl].vFfnGateInp, pCur);
    mCb(vRouterLogits, "ffn_moe_logits", pIl);

    ggml_tensor *vMoeOut =
        mBuildMoeFfn(vInpLatent, pModel.lLayers[pIl].vFfnGateInp, pModel.lLayers[pIl].vFfnUpExps,
                     nullptr, // no gate
                     pModel.lLayers[pIl].vFfnDownExps, pModel.lLayers[pIl].vFfnExpProbsB, cNExpert, cNExpertUsed,
                     cLlmFfnReluSqr, vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
                     cMgsExpertGatingFuncTypeSigmoid, pIl, vRouterLogits, nullptr, pModel.lLayers[pIl].vFfnUpExpsS,
                     nullptr, // no gate
                     pModel.lLayers[pIl].vFfnDownExpsS);
    mCb(vMoeOut, "ffn_moe_out", pIl);

    if (pModel.lLayers[pIl].vFfnLatentUp) {
      vMoeOut = ggml_mul_mat(vCtx0, pModel.lLayers[pIl].vFfnLatentUp, vMoeOut);
    }

    ggml_tensor *vFfnShexp = mBuildFfn(vInpEmb, pModel.lLayers[pIl].vFfnUpShexp, NULL, pModel.lLayers[pIl].vFfnUpShexpS,
                                       NULL /* no gate */, NULL, NULL, pModel.lLayers[pIl].vFfnDownShexp, NULL,
                                       pModel.lLayers[pIl].vFfnDownShexpS, NULL, cLlmFfnReluSqr, cLlmFfnPar, pIl);
    mCb(vFfnShexp, "ffn_shexp", pIl);

    pCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
    mCb(pCur, "ffn_out", pIl);
  }

  pCur = mBuildCvec(pCur, pIl);
  mCb(pCur, "l_out", pIl);

  return pCur;
}
