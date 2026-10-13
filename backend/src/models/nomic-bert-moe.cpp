#include "models.h"

void mgs_model_nomic_bert_moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormEps, vHparams.vFNormEps);
  pMl.mGetKey(cLlmKvMoeEveryNLayers, vHparams.vMoeEveryNLayers, 0);

  if (vHparams.mNLayer() == 12 && vHparams.vNEmbd == 768) {
    if (vArch == cLlmArchNomicBert) {
      vType = cLlmType137m;
    } else if (vArch == cLlmArchNomicBertMoe && vHparams.vMoeEveryNLayers == 2) {
      vType = cLlmType475m;
    }
  }
}

void mgs_model_nomic_bert_moe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  if (cNTokenTypes == 0) {
    throw std::runtime_error(mArchName() + " model needs to define token type count");
  }
  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);
  vTypeEmbd = mCreateTensor(cTn(cLlmTensorTokenTypes, "weight"), {cNEmbd, cNTokenTypes}, cTensorNotRequired);

  if (vArch == cLlmArchBert) {
    vPosEmbd = mCreateTensor(cTn(cLlmTensorPosEmbd, "weight"), {cNEmbd, cNCtxTrain}, 0);

    vCls = mCreateTensor(cTn(cLlmTensorCls, "weight"), {cNEmbd, cNEmbd}, cTensorNotRequired);
    vClsB = mCreateTensor(cTn(cLlmTensorCls, "bias"), {cNEmbd}, cTensorNotRequired);

    vClsOut = mCreateTensor(cTn(cLlmTensorClsOut, "weight"), {cNEmbd, vHparams.vNClsOut}, cTensorNotRequired);
    vClsOutB = mCreateTensor(cTn(cLlmTensorClsOut, "bias"), {vHparams.vNClsOut}, cTensorNotRequired);
  }

  vTokNorm = mCreateTensor(cTn(cLlmTensorTokenEmbdNorm, "weight", 0), {cNEmbd}, 0);
  vTokNormB = mCreateTensor(cTn(cLlmTensorTokenEmbdNorm, "bias", 0), {cNEmbd}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, cNEmbdGqa, cNEmbdGqa, 0);

    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOut, "bias", vI), {cNEmbd}, cTensorNotRequired);

    vLayer.vAttnOutNorm = mCreateTensor(cTn(cLlmTensorAttnOutNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnOutNormB = mCreateTensor(cTn(cLlmTensorAttnOutNorm, "bias", vI), {cNEmbd}, 0);

    if (vHparams.vMoeEveryNLayers > 0 && vI % vHparams.vMoeEveryNLayers == 1) {
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFf, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFf, cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    } else {
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnUpB = mCreateTensor(cTn(cLlmTensorFfnUp, "bias", vI), {cNFf}, cTensorNotRequired);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnDownB = mCreateTensor(cTn(cLlmTensorFfnDown, "bias", vI), {cNEmbd}, cTensorNotRequired);

      if (vArch == cLlmArchNomicBert) {
        vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      }
    }

    vLayer.vLayerOutNorm = mCreateTensor(cTn(cLlmTensorLayerOutNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vLayerOutNormB = mCreateTensor(cTn(cLlmTensorLayerOutNorm, "bias", vI), {cNEmbd}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_nomic_bert_moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_nomic_bert_moe::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vCur;
  ggml_tensor *vInpL;
  ggml_tensor *vInpPos = nullptr;

  if (pModel.vArch != cLlmArchJinaBertV2) {
    vInpPos = mBuildInpPos();
  }

  // construct input embeddings (token, type, position)
  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // token types are hardcoded to zero ("Sentence A")
  if (pModel.vTypeEmbd) {
    ggml_tensor *vTypeRow0 = ggml_view_1d(vCtx0, pModel.vTypeEmbd, cNEmbd, 0);
    vInpL = ggml_add(vCtx0, vInpL, vTypeRow0);
  }
  if (pModel.vArch == cLlmArchBert) {
    vInpL = ggml_add(vCtx0, ggml_get_rows(vCtx0, pModel.vPosEmbd, vInpPos), vInpL);
  }
  mCb(vInpL, "inp_embd", -1);

  // embed layer norm
  vInpL = mBuildNorm(vInpL, pModel.vTokNorm, pModel.vTokNormB, cLlmNorm, 0);
  mCb(vInpL, "inp_norm", 0);

  auto *vInpAttn = mBuildAttnInpNoCache();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vCur = vInpL;

    {
      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      if (pModel.lLayers[vIl].vAttnQNorm) {
        vQcur = ggml_reshape_2d(vCtx0, vQcur, cNEmbdHead * cNHead, cNTokens);

        vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, pModel.lLayers[vIl].vAttnQNormB, cLlmNorm, vIl);

        vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHead, cNHead, cNTokens);
      }

      if (pModel.lLayers[vIl].vAttnKNorm) {
        vKcur = ggml_reshape_2d(vCtx0, vKcur, cNEmbdHead * cNHeadKv, cNTokens);

        vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, pModel.lLayers[vIl].vAttnKNormB, cLlmNorm, vIl);

        vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
      }

      // RoPE
      if (pModel.vArch == cLlmArchNomicBert || pModel.vArch == cLlmArchNomicBertMoe ||
          pModel.vArch == cLlmArchJinaBertV3) {
        vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                             cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, pModel.lLayers[vIl].vWoB, pModel.lLayers[vIl].vWoS, vQcur,
                        vKcur, vVcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
      mCb(vCur, "kqv_out", vIl);
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpL = ggml_get_rows(vCtx0, vInpL, vInpOutIds);
    }

    // re-add the layer input
    vCur = ggml_add(vCtx0, vCur, vInpL);

    // attention layer norm
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnOutNorm, pModel.lLayers[vIl].vAttnOutNormB, cLlmNorm, vIl);

    if (pModel.lLayers[vIl].vAttnNorm2 != nullptr) {
      vCur = ggml_add(vCtx0, vCur, vInpL); // re-add the layer input
      vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnNorm2, pModel.lLayers[vIl].vAttnNorm2B, cLlmNorm, vIl);
    }

    ggml_tensor *vFfnInp = vCur;
    mCb(vFfnInp, "ffn_inp", vIl);

    // feed-forward network
    if (vHparams.vMoeEveryNLayers > 0 && vIl % vHparams.vMoeEveryNLayers == 1) {
      // MoE branch
      vCur = mBuildMoeFfn(vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, nullptr,
                          pModel.lLayers[vIl].vFfnDownExps, nullptr, vHparams.vNExpert, vHparams.vNExpertUsed,
                          cLlmFfnGelu, false, vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, vIl);
      mCb(vCur, "ffn_moe_out", vIl);
    } else if (pModel.vArch == cLlmArchBert || pModel.vArch == cLlmArchNomicBertMoe ||
               pModel.vArch == cLlmArchJinaBertV3) {
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, NULL, NULL, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, pModel.lLayers[vIl].vFfnDownB, NULL, NULL, cLlmFfnGelu, cLlmFfnSeq,
                       vIl);
      mCb(vCur, "ffn_out", vIl);
    } else if (pModel.vArch == cLlmArchJinaBertV2) {
      const bool cUpContainsGate =
          !pModel.lLayers[vIl].vFfnGate && pModel.lLayers[vIl].vFfnUp->ne[1] != vHparams.mNFf();
      auto vTypeOp = cUpContainsGate ? cLlmFfnGeglu : cLlmFfnGelu;
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, pModel.lLayers[vIl].vFfnUpB, NULL,
                       pModel.lLayers[vIl].vFfnGate, NULL, NULL, pModel.lLayers[vIl].vFfnDown,
                       pModel.lLayers[vIl].vFfnDownB, NULL, NULL, vTypeOp, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    }

    // attentions bypass the intermediate layer
    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    // output layer norm
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vLayerOutNorm, pModel.lLayers[vIl].vLayerOutNormB, cLlmNorm, vIl);

    // input for next layer
    vInpL = vCur;
  }

  vCur = vInpL;

  mCb(vCur, "result_embd", -1);
  vRes->vTEmbd = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
