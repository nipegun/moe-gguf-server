#include "models.h"

#include "mgs-impl.h"
#include "mgs-kv-cache.h"
#include "mgs-kv-cache-iswa.h"

void mgs_model_dflash::mLoadArchHparams(mgs_model_loader &pMl) {

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  if (!pMl.mGetArr(cLlmKvTargetLayers, lTargetLayerIds, false)) {
    throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
  }

  vHparams.vNEmbdInpEncImpl = (uint32_t)lTargetLayerIds.size() * vHparams.vNEmbd;

  MGS_LOG_INFO("%s: DFlash extract_layers = [", __func__);
  for (size_t vI = 0; vI < lTargetLayerIds.size(); ++vI) {
    MGS_LOG_INFO("%d%s", lTargetLayerIds[vI], vI + 1 < lTargetLayerIds.size() ? ", " : "");
  }
  MGS_LOG_INFO("]\n");

  // optional interleaved sliding-window attention with per-layer pattern array.
  // DFlash has a single rope, so the SWA rope == main rope.
  if (pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa, false) && vHparams.vNSwa > 0) {
    vHparams.vSwaType = cMgsSwaTypeStandard;
    pMl.mGetKeyOrArr(cLlmKvAttentionSlidingWindowPattern, vHparams.aIsSwaImpl, vHparams.mNLayer());
    vHparams.vRopeFreqBaseTrainSwa = vHparams.vRopeFreqBaseTrain;
    vHparams.vRopeFreqScaleTrainSwa = vHparams.vRopeFreqScaleTrain;
  }

  vType = cLlmTypeUnknown;
}

void mgs_model_dflash::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  const int64_t cNEmbdInp = vHparams.mNEmbdInpEnc();

  // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
  //
  // TODO: only Qwen3-style backbones are supported for now; other backbones need their own conversion path and
  //       graph tweaks
  const struct ggml_tensor *vMarkovMeta = vMl->mGetTensorMeta("markov_w1.weight");
  if (vMarkovMeta) {
    const int64_t cDsparkMarkovRank = vMarkovMeta->ne[0];

    vDsparkMarkovW1 = mCreateTensor(cTn(cLlmTensorDsparkMarkovW1, "weight"), {cDsparkMarkovRank, cNVocab}, 0);
    vDsparkMarkovW2 = mCreateTensor(cTn(cLlmTensorDsparkMarkovW2, "weight"), {cDsparkMarkovRank, cNVocab}, 0);

    vDsparkConfProj = mCreateTensor(cTn(cLlmTensorDsparkConfProj, "weight"), {cNEmbd + cDsparkMarkovRank, 1}, 0);
    vDsparkConfProjB = mCreateTensor(cTn(cLlmTensorDsparkConfProj, "bias"), {1}, cTensorNotRequired);

    MGS_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long)cDsparkMarkovRank);
  }

  vFc = mCreateTensor(cTn(cLlmTensorFc, "weight"), {cNEmbdInp, cNEmbd}, 0);
  vOutputNormEnc = mCreateTensor(cTn(cLlmTensorEncOutputNorm, "weight"), {cNEmbd}, 0); // encoder hidden_norm (after fc)
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);       // decoder final norm

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    vLayer.vWq = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", vI), {cNEmbd, cNEmbdHeadK * cNHead}, 0);
    vLayer.vWk = mCreateTensor(cTn(cLlmTensorAttnK, "weight", vI), {cNEmbd, cNEmbdKGqa}, 0);
    vLayer.vWv = mCreateTensor(cTn(cLlmTensorAttnV, "weight", vI), {cNEmbd, cNEmbdVGqa}, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
    vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
    vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_dflash::mBuildArchGraph(const llm_graph_params &pParams) const {
  switch (pParams.vGtype) {
  case cLlmGraphTypeEncoder:
    return std::make_unique<graph<true>>(*this, pParams);
  case cLlmGraphTypeDefault:
  case cLlmGraphTypeDecoder:
    return std::make_unique<graph<false>>(*this, pParams);
  default:
    GGML_ABORT("invalid graph type");
  };
}

template <> ggml_tensor *mgs_model_dflash::graph<true>::mBuildInpEmbdEnc() const {
  auto vInpTarget = std::make_unique<llm_graph_input_embd>(vHparams.mNEmbdInpEnc());

  vInpTarget->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInpEnc(), cNTokens);
  ggml_set_input(vInpTarget->vEmbd);

  ggml_tensor *vCur = vInpTarget->vEmbd;
  mCb(vCur, "inp_embd", -1);

  vRes->mAddInput(std::move(vInpTarget));

  return vCur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
mgs_model_dflash::graph<true>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  ggml_tensor *vCur = mBuildInpEmbdEnc();

  vCur = mBuildLoraMm(pModel.vFc, vCur);
  mCb(vCur, "fc_out", -1);

  vCur = mBuildNorm(vCur, pModel.vOutputNormEnc, NULL, cLlmNormRms, -1);
  mCb(vCur, "enc_norm_out", -1);

  ggml_set_output(vCur);
  vRes->vTHNextn = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
static void fBuildDsparkMarkovHead(llm_graph_context &pG, const mgs_model &pModel, ggml_tensor *pTokens) {
  ggml_context *vCtx0 = pG.vCtx0;
  auto &vRes = pG.vRes;

  ggml_tensor *vW1 = pModel.vDsparkMarkovW1;
  ggml_tensor *vW2 = pModel.vDsparkMarkovW2;
  GGML_ASSERT(vW1 && vW2 && pModel.vDsparkConfProj && "DSpark markov/confidence weights not loaded");

  ggml_tensor *vBase = vRes->vTLogits; // [n_vocab, n_tokens]
  const int64_t cNVocab = vBase->ne[0];
  const int64_t cNTok = vBase->ne[1];

  const auto cIt = pModel.dGgufKv.find("dflash.block_size");
  GGML_ASSERT(cIt != pModel.dGgufKv.end() && "DSpark draft requires 'dflash.block_size' in GGUF metadata");
  const int64_t cBlockSize = std::stoi(cIt->second);
  GGML_ASSERT(cBlockSize > 0);

  const int64_t cNBlocks = pG.vUbatch.vNSeqsUnq;
  GGML_ASSERT(cNBlocks > 0 && cNTok % cNBlocks == 0 && "DSpark markov head requires equal-size blocks");
  // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
  const int64_t cBlockDrafts = cNTok / cNBlocks;
  if (cBlockDrafts > cBlockSize) {
    return;
  }

  // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
  const size_t cTokenStride = (size_t)cBlockDrafts * pTokens->nb[0];
  const size_t cBaseStride = (size_t)cBlockDrafts * vBase->nb[1];

  ggml_tensor *vPrev = ggml_view_2d(vCtx0, pTokens, 1, cNBlocks, cTokenStride, 0);
  vPrev = ggml_cont_1d(vCtx0, vPrev, cNBlocks);

  // confidence head input: predicts per-position acceptance
  ggml_tensor *vConfInp = vRes->vTEmbd; // [n_embd, n_tok]

  ggml_tensor *vCat = nullptr;
  ggml_tensor *vCatConf = nullptr;

  // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
  //       token pick, not the Markov conditioning path
  for (int64_t vI = 0; vI < cBlockDrafts; ++vI) {
    ggml_tensor *vW1Prev = ggml_get_rows(vCtx0, vW1, vPrev); // [R, n_blocks]
    ggml_tensor *vBias = ggml_mul_mat(vCtx0, vW2, vW1Prev);  // [n_vocab, n_blocks]

    // position i of every block: strided view [n_vocab, n_blocks]
    ggml_tensor *vBaseI = ggml_view_2d(vCtx0, vBase, cNVocab, cNBlocks, cBaseStride, vI * vBase->nb[1]);
    ggml_tensor *vCol = ggml_add(vCtx0, vBaseI, vBias);

    vCat = vCat ? ggml_concat(vCtx0, vCat, vCol, 1) : vCol;

    // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
    ggml_tensor *vConfInpI = ggml_view_2d(vCtx0, vConfInp, vConfInp->ne[0], cNBlocks,
                                          (size_t)cBlockDrafts * vConfInp->nb[1], vI * vConfInp->nb[1]);
    ggml_tensor *vFeat = ggml_concat(vCtx0, ggml_cont(vCtx0, vConfInpI), vW1Prev, 0);
    ggml_tensor *vConf = ggml_mul_mat(vCtx0, pModel.vDsparkConfProj, vFeat);
    if (pModel.vDsparkConfProjB) {
      vConf = ggml_add(vCtx0, vConf, pModel.vDsparkConfProjB);
    }
    vConf = ggml_sigmoid(vCtx0, vConf);

    vCatConf = vCatConf ? ggml_concat(vCtx0, vCatConf, vConf, 1) : vConf;

    if (vI + 1 < cBlockDrafts) {
      vPrev = ggml_argmax(vCtx0, vCol);
    }
  }

  // cat is position-major; restore ubatch block-major order
  ggml_tensor *vOut = ggml_reshape_3d(vCtx0, vCat, cNVocab, cNBlocks, cBlockDrafts);
  vOut = ggml_cont(vCtx0, ggml_permute(vCtx0, vOut, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
  vOut = ggml_reshape_2d(vCtx0, vOut, cNVocab, cNTok);

  {
    ggml_tensor *vConf = ggml_reshape_3d(vCtx0, vCatConf, 1, cNBlocks, cBlockDrafts);
    vConf = ggml_cont(vCtx0, ggml_permute(vCtx0, vConf, 0, 2, 1, 3));
    vConf = ggml_reshape_2d(vCtx0, vConf, 1, cNTok);

    // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `mgs_get_embeddings_nextn`
    vConf = ggml_repeat(vCtx0, vConf, vRes->vTEmbd);
    vRes->vTHNextn = vConf;
    ggml_build_forward_expand(pG.vGf, vConf);
  }

  vRes->vTLogits = vOut;
  ggml_build_forward_expand(pG.vGf, vOut);
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
mgs_model_dflash::graph<false>::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  ggml_tensor *vInpPos = mBuildInpPos();

  // optional iSWA: pick the matching attention input
  const bool cUseIswa = vHparams.vSwaType != cMgsSwaTypeNone;

  llm_graph_input_attn_kv *vInpAttn = nullptr;
  llm_graph_input_attn_kv_iswa *vInpAttnIswa = nullptr;
  if (cUseIswa) {
    vInpAttnIswa = mBuildAttnInpKvIswa();
  } else {
    vInpAttn = mBuildAttnInpKv();
  }

  const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

  // KV cache injection
  if (vUbatch.vEmbd) {
    auto vInp = std::make_unique<llm_graph_input_embd>(cNEmbd);

    vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbd, cNTokens);
    ggml_set_input(vInp->vEmbd);

    ggml_tensor *vInpG = vInp->vEmbd;
    mCb(vInpG, "inp_g_embeddings", -1);

    vRes->mAddInput(std::move(vInp));

    for (int vIl = 0; vIl < cNLayer; ++vIl) {
      const auto &vLayer = pModel.lLayers[vIl];

      ggml_tensor *vKcur = mBuildLoraMm(vLayer.vWk, vInpG);
      ggml_tensor *vVcur = mBuildLoraMm(vLayer.vWv, vInpG);

      vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);

      vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, NULL, cLlmNormRms, vIl);
      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      mCb(vKcur, "Kcur_injected", vIl);
      mCb(vVcur, "Vcur_injected", vIl);

      if (cUseIswa) {
        // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
        const bool cIsSwa = vHparams.mIsSwa(vIl);
        const auto *vKv = cIsSwa ? vInpAttnIswa->vMctx->mGetSwa() : vInpAttnIswa->vMctx->mGetBase();
        ggml_tensor *vKIdxs = cIsSwa ? vInpAttnIswa->mGetKIdxsSwa() : vInpAttnIswa->mGetKIdxs();
        ggml_tensor *vVIdxs = cIsSwa ? vInpAttnIswa->mGetVIdxsSwa() : vInpAttnIswa->mGetVIdxs();
        // rotate K/V into the cache's rotated space
        ggml_tensor *vKRot = cIsSwa ? vInpAttnIswa->vSelfKRotSwa : vInpAttnIswa->vSelfKRot;
        ggml_tensor *vVRot = cIsSwa ? vInpAttnIswa->vSelfVRotSwa : vInpAttnIswa->vSelfVRot;
        if (vKRot) {
          vKcur = mgs_mul_mat_hadamard(vCtx0, vKcur, vKRot);
        }
        if (vVRot) {
          vVcur = mgs_mul_mat_hadamard(vCtx0, vVcur, vVRot);
        }
        ggml_build_forward_expand(vGf, vKv->mCpyK(vCtx0, vKcur, vKIdxs, vIl));
        ggml_build_forward_expand(vGf, vKv->mCpyV(vCtx0, vVcur, vVIdxs, vIl));
      } else {
        // rotate K/V into the cache's rotated space
        if (vInpAttn->vSelfKRot) {
          vKcur = mgs_mul_mat_hadamard(vCtx0, vKcur, vInpAttn->vSelfKRot);
        }
        if (vInpAttn->vSelfVRot) {
          vVcur = mgs_mul_mat_hadamard(vCtx0, vVcur, vInpAttn->vSelfVRot);
        }
        ggml_build_forward_expand(vGf, vInpAttn->vMctx->mCpyK(vCtx0, vKcur, vInpAttn->mGetKIdxs(), vIl));
        ggml_build_forward_expand(vGf, vInpAttn->vMctx->mCpyV(vCtx0, vVcur, vInpAttn->mGetVIdxs(), vIl));
      }
    }

    vRes->vTEmbd = vInpG;

    ggml_build_forward_expand(vGf, vInpG);
    return;
  }

  // tok_embd from the target model (shared via ctx_other)
  auto *vTokEmbd = pModel.vTokEmbd;
  if (vTokEmbd == nullptr) {
    GGML_ASSERT(vCparams.vCtxOther != nullptr);
    const auto *vModelOther = mgs_get_model(vCparams.vCtxOther);

    GGML_ASSERT(vModelOther->vTokEmbd != nullptr && "DFlash decoder requires the target model's token embeddings");
    vTokEmbd = vModelOther->vTokEmbd;
  }

  auto vInp = std::make_unique<llm_graph_input_embd>(cNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  ggml_tensor *vInpTokens = vInp->vTokens;

  ggml_tensor *vInpL = ggml_get_rows(vCtx0, vTokEmbd, vInp->vTokens);
  mCb(vInpL, "inp_noise_embd", -1);

  vRes->mAddInput(std::move(vInp));

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = pModel.lLayers[vIl];

    ggml_tensor *vNoiseNorm = mBuildNorm(vInpL, vLayer.vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vNoiseNorm, "noise_norm", vIl);

    ggml_tensor *vQcur = mBuildLoraMm(vLayer.vWq, vNoiseNorm);
    ggml_tensor *vKcur = mBuildLoraMm(vLayer.vWk, vNoiseNorm);
    ggml_tensor *vVcur = mBuildLoraMm(vLayer.vWv, vNoiseNorm);

    vQcur = ggml_reshape_3d(vCtx0, vQcur, cNEmbdHead, cNHead, cNTokens);
    vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
    vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);

    vQcur = mBuildNorm(vQcur, vLayer.vAttnQNorm, NULL, cLlmNormRms, vIl);
    vKcur = mBuildNorm(vKcur, vLayer.vAttnKNorm, NULL, cLlmNormRms, vIl);

    vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
    vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
    mCb(vQcur, "Qcur", vIl);
    mCb(vKcur, "Kcur", vIl);
    mCb(vVcur, "Vcur", vIl);

    // cache-aware, non-causal attention
    ggml_tensor *vCur = cUseIswa ? mBuildAttn(vInpAttnIswa, vLayer.vWo, NULL, NULL, vQcur, vKcur, vVcur, nullptr,
                                              nullptr, nullptr, cKqScale, vIl)
                                 : mBuildAttn(vInpAttn, vLayer.vWo, NULL, NULL, vQcur, vKcur, vVcur, nullptr, nullptr,
                                              nullptr, cKqScale, vIl);

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpL);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    vCur = mBuildFfn(vCur, vLayer.vFfnUp, NULL, NULL, vLayer.vFfnGate, NULL, NULL, vLayer.vFfnDown, NULL, NULL, NULL,
                     cLlmFfnSilu, cLlmFfnPar, vIl);
    mCb(vCur, "ffn_out", vIl);

    vCur = ggml_add(vCtx0, vCur, vFfnInp);
    mCb(vCur, "l_out", vIl);

    vInpL = vCur;
  }

  ggml_tensor *vCur = mBuildNorm(vInpL, pModel.vOutputNorm, NULL, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);

  vRes->vTEmbd = vCur;

  // lm_head from the target model (shared via ctx_other)
  auto *vOutput = pModel.vOutput;
  if (vOutput == nullptr) {
    GGML_ASSERT(vCparams.vCtxOther != nullptr);
    const auto *vModelOther = mgs_get_model(vCparams.vCtxOther);
    GGML_ASSERT(vModelOther->vOutput != nullptr && "DFlash decoder requires the target model's output projection");
    vOutput = vModelOther->vOutput;
  }

  vCur = mBuildLoraMm(vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);

  // DSpark: bias the draft logits with the Markov head
  if (pModel.vDsparkMarkovW1) {
    fBuildDsparkMarkovHead(*this, pModel, vInpTokens);
  }
}
