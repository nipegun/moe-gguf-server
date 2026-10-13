#include "models.h"

void mgs_model_jamba::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  pMl.mGetKey(cLlmKvSsmStateSize, vHparams.vSsmDState);
  pMl.mGetKey(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  for (uint32_t vI = 0; vI < vHparams.mNLayer(); ++vI) {
    vHparams.aIsRecrImpl[vI] = vHparams.mNHeadKv(vI) == 0;
  }

  switch (vHparams.mNLayer()) {
  // TODO: Jamba layers are a bit heterogeneous, so naming this is hard.
  case 12: // 900M  8x???M
  case 32: // 51B  16x?B
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_jamba::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cDState = vHparams.vSsmDState;
  const int64_t cDtRank = vHparams.vSsmDtRank;

  // only an expansion factor of 2 is supported for now
  GGML_ASSERT(2 * cNEmbd == cDInner);

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
    const int64_t cNHeadKv = vHparams.mNHeadKv(vI);
    const int64_t cNEmbdGqa = vHparams.mNEmbdVGqa(vI);

    auto &vLayer = lLayers[vI];

    // norm
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    if (cNHeadKv == 0) {
      // Mamba layer
      vLayer.vSsmIn = mCreateTensor(cTn(cLlmTensorSsmIn, "weight", vI), {cNEmbd, 2 * cDInner}, 0);

      vLayer.vSsmConv1d = mCreateTensor(cTn(cLlmTensorSsmConv1d, "weight", vI), {cDConv, cDInner}, 0);
      vLayer.vSsmConv1dB = mCreateTensor(cTn(cLlmTensorSsmConv1d, "bias", vI), {cDInner}, 0);

      vLayer.vSsmX = mCreateTensor(cTn(cLlmTensorSsmX, "weight", vI), {cDInner, cDtRank + 2 * cDState}, 0);

      vLayer.vSsmDtNorm = mCreateTensor(cTn(cLlmTensorSsmDtNorm, "weight", vI), {cDtRank}, 0);

      vLayer.vSsmDt = mCreateTensor(cTn(cLlmTensorSsmDt, "weight", vI), {cDtRank, cDInner}, 0);
      vLayer.vSsmDtB = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", vI), {cDInner}, 0);

      vLayer.vSsmBNorm = mCreateTensor(cTn(cLlmTensorSsmBNorm, "weight", vI), {cDState}, 0);
      vLayer.vSsmCNorm = mCreateTensor(cTn(cLlmTensorSsmCNorm, "weight", vI), {cDState}, 0);

      // no "weight" suffix for these
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmA, vI), {cDState, cDInner}, 0);
      vLayer.vSsmD = mCreateTensor(cTn(cLlmTensorSsmD, vI), {cDInner}, 0);

      // out_proj
      vLayer.vSsmOut = mCreateTensor(cTn(cLlmTensorSsmOut, "weight", vI), {cDInner, cNEmbd}, 0);
    } else {
      // Attention layers

      mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, cNEmbdGqa, cNEmbdGqa, 0);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);
    }

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, cTensorNotRequired);

    if (vLayer.vFfnGateInp) {
      // MoE
      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
    } else {
      // FFN (no MoE)
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_jamba::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_jamba::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_mamba_base(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  // {n_embd, n_tokens}
  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  auto *vInpHybrid = mBuildInpMemHybrid();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const int64_t cNHeadKv = vHparams.mNHeadKv(vIl);

    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    if (cNHeadKv == 0) {
      vCur = mBuildMambaLayer(vInpHybrid->mGetRecr(), vCur, pModel, vUbatch, vIl);
    } else {
      // Attention

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      // No RoPE :)
      vCur = mBuildAttn(vInpHybrid->mGetAttn(), pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur,
                        vVcur, NULL, NULL, NULL, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
    }
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpL = ggml_get_rows(vCtx0, vInpL, vInpOutIds);
    }
    // residual
    struct ggml_tensor *vFfnInp = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // feed-forward network
    if (pModel.lLayers[vIl].vFfnGateInp == nullptr) {
      // FFN
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps,
                          pModel.lLayers[vIl].vFfnGateExps, pModel.lLayers[vIl].vFfnDownExps, nullptr, cNExpert,
                          cNExpertUsed, cLlmFfnSilu, false, vHparams.vExpertWeightsScale,
                          cMgsExpertGatingFuncTypeSoftmax, vIl);
      mCb(vCur, "ffn_moe_out", vIl);
    }
    // residual
    vCur = ggml_add(vCtx0, vFfnInp, vCur);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  // final rmsnorm
  vCur = mBuildNorm(vInpL, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
