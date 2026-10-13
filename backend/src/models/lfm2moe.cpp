#include "models.h"
#include "../mgs-memory-hybrid-iswa.h"
#include "../mgs-memory-hybrid.h"

void mgs_model_lfm2moe::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvShortconvLCache, vHparams.vNShortconvLCache);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);

  for (uint32_t vIl = 0; vIl < vHparams.mNLayer(); ++vIl) {
    vHparams.aIsRecrImpl[vIl] = vHparams.mNHeadKv(vIl) == 0;
  }

  switch (vHparams.mNLayer()) {
  case 24:
    vType = cLlmType8bA1b;
    break;
  case 40:
    vType = cLlmType24bA2b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_lfm2moe::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNormLfm2, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);

  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    const bool cIsMoeLayer = vI >= static_cast<int>(vHparams.vNLayerDenseLead);

    // ffn/moe is same for transformer and conv layers
    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
    if (cIsMoeLayer) {
      GGML_ASSERT(cNExpert && cNExpertUsed);
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, vHparams.vNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {vHparams.vNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps =
          mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, vHparams.vNFfExp, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);
    } else { // dense
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    }

    // for operator_norm
    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    if (!vHparams.mIsRecr(vI)) {
      vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);
      vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);
      GGML_ASSERT(cNEmbdVGqa == cNEmbdKGqa);

      mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbd, vHparams.mNEmbdKGqa(vI), vHparams.mNEmbdVGqa(vI), 0);

      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbd, cNEmbd}, 0);
    } else {
      vLayer.vShortconv.vConv =
          mCreateTensor(cTn(cLlmTensorShortconvConv, "weight", vI), {vHparams.vNShortconvLCache, cNEmbd}, 0);
      vLayer.vShortconv.vInProj = mCreateTensor(cTn(cLlmTensorShortconvInproj, "weight", vI), {cNEmbd, 3 * cNEmbd}, 0);
      vLayer.vShortconv.vOutProj = mCreateTensor(cTn(cLlmTensorShortconvOutproj, "weight", vI), {cNEmbd, cNEmbd}, 0);
    }
  }

  // for LFM2-ColBert-350M
  vDense2OutLayers =
      mCreateTensor(cTn(cLlmTensorDense2Out, "weight"), {cNEmbd, vHparams.mNEmbdOut()}, cTensorNotRequired);
  vDense2OutLayersB = mCreateTensor(cTn(cLlmTensorDense2Out, "bias"), {vHparams.mNEmbdOut()}, cTensorNotRequired);
}

std::unique_ptr<llm_graph_context> mgs_model_lfm2moe::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (vHparams.vSwaType == cMgsSwaTypeStandard) {
    return std::make_unique<graph<true>>(*this, pParams);
  } else {
    return std::make_unique<graph<false>>(*this, pParams);
  }
}

template <bool iswa>
mgs_model_lfm2moe::graph<iswa>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  using inp_hybrid_type = std::conditional_t<iswa, llm_graph_input_mem_hybrid_iswa, llm_graph_input_mem_hybrid>;
  using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
  using mem_hybrid_ctx = std::conditional_t<iswa, mgs_memory_hybrid_iswa_context, mgs_memory_hybrid_context>;

  // lambda helpers for readability
  auto vBuildDenseFeedForward = [&pModel, this](ggml_tensor *pCur, int pIl) -> ggml_tensor * {
    GGML_ASSERT(!pModel.lLayers[pIl].vFfnUpB);
    GGML_ASSERT(!pModel.lLayers[pIl].vFfnGateB);
    GGML_ASSERT(!pModel.lLayers[pIl].vFfnDownB);
    return mBuildFfn(pCur, pModel.lLayers[pIl].vFfnUp, NULL, NULL, pModel.lLayers[pIl].vFfnGate, NULL, NULL,
                     pModel.lLayers[pIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
  };
  auto vBuildMoeFeedForward = [&pModel, this](ggml_tensor *pCur, int pIl) -> ggml_tensor * {
    return mBuildMoeFfn(
        pCur, pModel.lLayers[pIl].vFfnGateInp, pModel.lLayers[pIl].vFfnUpExps, pModel.lLayers[pIl].vFfnGateExps,
        pModel.lLayers[pIl].vFfnDownExps, pModel.lLayers[pIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
        vHparams.vExpertWeightsScale, static_cast<mgs_expert_gating_func_type>(vHparams.vExpertGatingFunc), pIl);
  };
  auto vBuildAttnBlock = [&pModel, this](ggml_tensor *pCur, ggml_tensor *pInpPos, inp_attn_type *pInpAttn,
                                         int pIl) -> ggml_tensor * {
    GGML_ASSERT(vHparams.mNEmbdVGqa(pIl) == vHparams.mNEmbdKGqa(pIl));
    const auto cNEmbdHead = vHparams.mNEmbdHeadV();
    const auto cNHeadKv = vHparams.mNHeadKv(pIl);

    auto [vQ, vK, vV] = mBuildQkv(pModel.lLayers[pIl], pCur, cNEmbdHead, cNHead, cNHeadKv, pIl);

    // qk norm
    vQ = mBuildNorm(vQ, pModel.lLayers[pIl].vAttnQNorm, NULL, cLlmNormRms, pIl);
    mCb(vQ, "model.layers.{}.self_attn.q_layernorm", pIl);
    vK = mBuildNorm(vK, pModel.lLayers[pIl].vAttnKNorm, NULL, cLlmNormRms, pIl);
    mCb(vK, "model.layers.{}.self_attn.k_layernorm", pIl);

    // RoPE
    vQ = ggml_rope_ext(vCtx0, vQ, pInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                      cAttnFactor, cBetaFast, cBetaSlow);
    vK = ggml_rope_ext(vCtx0, vK, pInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                      cAttnFactor, cBetaFast, cBetaSlow);

    pCur = mBuildAttn(pInpAttn, pModel.lLayers[pIl].vWo, NULL, pModel.lLayers[pIl].vWoS, vQ, vK, vV, nullptr, nullptr,
                      nullptr, 1.0f / sqrtf(float(cNEmbdHead)), pIl);

    mCb(pCur, "model.layers.{}.self_attn.out_proj", pIl);

    return pCur;
  };
  auto vBuildShortconvBlock = [&pModel, this](ggml_tensor *pCur, llm_graph_input_rs *pInpRecr,
                                              int pIl) -> ggml_tensor * {
    const auto *vMctxCur = static_cast<const mem_hybrid_ctx *>(vMctx)->mGetRecr();
    const uint32_t cKvHead = vMctxCur->mGetHead();
    const int64_t cNSeqTokens = vUbatch.vNSeqTokens;
    const int64_t cNSeqs = vUbatch.vNSeqs;
    GGML_ASSERT(cNSeqs != 0);
    GGML_ASSERT(vUbatch.mEqualSeqs());
    GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

    GGML_ASSERT(vHparams.vNShortconvLCache > 1);
    const uint32_t cDConv = vHparams.vNShortconvLCache - 1;

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    pCur = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], cNSeqTokens, cNSeqs);

    auto *vBcx = mBuildLoraMm(pModel.lLayers[pIl].vShortconv.vInProj, pCur);
    mCb(vBcx, "model.layers.{}.conv.in_proj", pIl);

    constexpr auto cNChunks = 3;
    GGML_ASSERT(vBcx->ne[0] % cNChunks == 0);
    const auto cChunkSize = vBcx->ne[0] / cNChunks;
    auto *vB = ggml_view_3d(vCtx0, vBcx, cChunkSize, vBcx->ne[1], vBcx->ne[2], vBcx->nb[1], vBcx->nb[2],
                            0 * cChunkSize * ggml_element_size(vBcx));
    auto *vC = ggml_view_3d(vCtx0, vBcx, cChunkSize, vBcx->ne[1], vBcx->ne[2], vBcx->nb[1], vBcx->nb[2],
                            1 * cChunkSize * ggml_element_size(vBcx));
    auto *vX = ggml_view_3d(vCtx0, vBcx, cChunkSize, vBcx->ne[1], vBcx->ne[2], vBcx->nb[1], vBcx->nb[2],
                            2 * cChunkSize * ggml_element_size(vBcx));

    auto *vBx = ggml_transpose(vCtx0, ggml_mul(vCtx0, vB, vX));

    // read conv state
    auto *vConvState = vMctxCur->mGetRL(pIl);
    auto *vConvRs = mBuildRs(pInpRecr, vConvState, vHparams.mNEmbdR(), cNSeqs);
    auto *vConv = ggml_reshape_3d(vCtx0, vConvRs, cDConv, vHparams.vNEmbd, cNSeqs);

    // causal prepends the state, non-causal pads symmetrically for a centered window
    if (vHparams.vCausalAttn) {
      vBx = ggml_concat(vCtx0, vConv, vBx, 0);
    } else {
      const int64_t cPad = (vHparams.vNShortconvLCache - 1) / 2;
      auto *vLeft = ggml_cont(vCtx0, ggml_view_3d(vCtx0, vConv, cPad, vHparams.vNEmbd, cNSeqs, vConv->nb[1],
                                                  vConv->nb[2], (cDConv - cPad) * vConv->nb[0]));
      vBx = ggml_pad_ext(vCtx0, ggml_concat(vCtx0, vLeft, vBx, 0), 0, cPad, 0, 0, 0, 0, 0, 0);
    }
    GGML_ASSERT(vBx->ne[0] > vConv->ne[0]);

    // last d_conv columns is a new conv state
    auto *vNewConv = ggml_view_3d(vCtx0, vBx, vConv->ne[0], vBx->ne[1], vBx->ne[2], vBx->nb[1], vBx->nb[2],
                                  (vBx->ne[0] - vConv->ne[0]) * ggml_element_size(vBx));
    GGML_ASSERT(ggml_are_same_shape(vConv, vNewConv));

    // write new conv conv state
    ggml_build_forward_expand(vGf, ggml_cpy(vCtx0, vNewConv,
                                            ggml_view_1d(vCtx0, vConvState, ggml_nelements(vNewConv),
                                                         cKvHead * cDConv * cNEmbd * ggml_element_size(vNewConv))));

    auto *vConvKernel = pModel.lLayers[pIl].vShortconv.vConv;
    auto *vConvOut = ggml_ssm_conv(vCtx0, vBx, vConvKernel);
    mCb(vConvOut, "model.layers.{}.conv.conv", pIl);

    auto *vY = ggml_mul(vCtx0, vC, vConvOut);
    vY = mBuildLoraMm(pModel.lLayers[pIl].vShortconv.vOutProj, vY);
    mCb(vY, "model.layers.{}.conv.out_proj", pIl);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    vY = ggml_reshape_2d(vCtx0, vY, vY->ne[0], cNSeqTokens * cNSeqs);

    return vY;
  };

  // actual graph construction starts here
  ggml_tensor *vCur = mBuildInpEmbd(pModel.vTokEmbd);
  mCb(vCur, "model.embed_tokens", -1);

  ggml_build_forward_expand(vGf, vCur);

  inp_hybrid_type *vInpHybrid = nullptr;
  if constexpr (iswa) {
    vInpHybrid = mBuildInpMemHybridIswa();
  } else {
    vInpHybrid = mBuildInpMemHybrid();
  }

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const bool cIsMoeLayer = vIl >= static_cast<int>(vHparams.vNLayerDenseLead);

    auto *vPrevCur = vCur;
    vCur = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "model.layers.{}.operator_norm", vIl);

    vCur = vHparams.mIsRecr(vIl) ? vBuildShortconvBlock(vCur, vInpHybrid->mGetRecr(), vIl)
                                 : vBuildAttnBlock(vCur, vInpPos, vInpHybrid->mGetAttn(), vIl);

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vPrevCur = ggml_get_rows(vCtx0, vPrevCur, vInpOutIds);
    }

    vCur = ggml_add(vCtx0, vPrevCur, vCur);

    auto *vFfnNormOut = mBuildNorm(vCur, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vFfnNormOut, "model.layers.{}.ffn_norm", vIl);

    ggml_tensor *vFfnOut =
        cIsMoeLayer ? vBuildMoeFeedForward(vFfnNormOut, vIl) : vBuildDenseFeedForward(vFfnNormOut, vIl);
    mCb(vFfnNormOut, "model.layers.{}.ffn_out", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnOut);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);
  }

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  if (!vCparams.vEmbeddings) {
    vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
    mCb(vCur, "result_output", -1);

    vRes->vTLogits = vCur;
  }

  ggml_build_forward_expand(vGf, vCur);
}

// Explicit template instantiations
template struct mgs_model_lfm2moe::graph<true>;
template struct mgs_model_lfm2moe::graph<false>;
