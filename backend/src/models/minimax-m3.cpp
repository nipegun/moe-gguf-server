#include "models.h"
#include "mgs-kv-cache-msa.h"
#include <cmath>
#include <vector>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling),
// swigluoai activation, and MiniMax Sparse Attention (MSA). MTP is not in released model weights.
// MSA blocks are defined over token positions. The graph translates between position space (block
// selection) and cell space (K/V/indexer storage) via per-ubatch pos<->cell maps populated from mgs_kv_cells

void mgs_model_minimax_m3::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  pMl.mGetKey(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  pMl.mGetKey(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  pMl.mGetKey(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);
  pMl.mGetKey(cLlmKvAttentionIndexerBlockSize, vHparams.vIndexerBlockSize);
  pMl.mGetKey(cLlmKvAttentionIndexerLocalBlocks, vHparams.vIndexerLocalBlocks);
  vMsaP = {(int)vHparams.vIndexerBlockSize, (int)vHparams.vIndexerTopK, (int)vHparams.vIndexerLocalBlocks};

  switch (vHparams.mNLayer()) {
  case 60:
    vType = cLlmType428bA23b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_minimax_m3::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;
  const int64_t cNExpertShared = vHparams.vNExpertShared;
  const int64_t cNFfExp = vHparams.vNFfExp;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadK * cNHead, cNEmbdGqa, cNEmbdGqa, 0);
    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadK * cNHead, cNEmbd}, 0);

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    // per-head QK-norm: a single head_dim vector applied to every head
    vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", vI), {cNEmbdHeadK}, 0);
    vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", vI), {cNEmbdHeadK}, 0);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    if (vI < (int)vHparams.vNLayerDenseLead) {
      // leading dense layers
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    } else {
      // routed experts
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);
      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

      // shared expert
      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, 0);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfExp * cNExpertShared, cNEmbd}, 0);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, 0);

      // indexer
      vLayer.vIndexQProj = mCreateTensor(cTn(cLlmTensorIndexerQProj, "weight", vI),
                                         {cNEmbd, vHparams.vIndexerNHead * vHparams.vIndexerHeadSize}, 0);
      vLayer.vIndexKProj =
          mCreateTensor(cTn(cLlmTensorIndexerKProj, "weight", vI), {cNEmbd, vHparams.vIndexerHeadSize}, 0);
      vLayer.vIndexQNorm = mCreateTensor(cTn(cLlmTensorIndexerQNorm, "weight", vI), {vHparams.vIndexerHeadSize}, 0);
      vLayer.vIndexKNorm = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "weight", vI), {vHparams.vIndexerHeadSize}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_minimax_m3::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

class llm_graph_input_msa : public llm_graph_input_i {
public:
  llm_graph_input_msa(const mgs_kv_cache_msa_context *pMctx, int pBlk, int pLocal)
      : vMctx(pMctx), vBlk(pBlk), vLocal(pLocal) {}

  void mSetInput(const mgs_ubatch *pUbatch) override {
    if (vPosSlotI) {
      vMctx->mSetInputPosSlot(vPosSlotI, pUbatch);
    }
    if (vPosSlotF) {
      vMctx->mSetInputPosSlot(vPosSlotF, pUbatch);
    }
    if (vCellBlk) {
      vMctx->mSetInputCellPos(vCellBlk, pUbatch, vBlk);
    }
    if (vPosMask) {
      vMctx->mSetInputPosMask(vPosMask, pUbatch);
    }

    // local-force bias over position blocks
    if (vBias && pUbatch->vPos) {
      const int64_t cNTokens = pUbatch->vNTokens;
      const int64_t cNblk = vBias->ne[0];
      std::vector<float> lData((size_t)cNblk * cNTokens, 0.0f);
      for (int64_t vI = 0; vI < cNTokens; ++vI) {
        const int64_t cL = pUbatch->vPos[vI] / vBlk;
        for (int vL = 0; vL < vLocal && cL - vL >= 0; ++vL) {
          if (cL - vL < cNblk) {
            lData[(size_t)vI * cNblk + (cL - vL)] = 1e30f;
          }
        }
      }
      ggml_backend_tensor_set(vBias, lData.data(), 0, lData.size() * sizeof(float));
    }
  }

  // valid as long as the tensor dims still match the new ubatch/cache window and the
  // ubatch is in the same regime (decode graphs have pos_slot_f, batch graphs cell_blk)
  bool mCanReuse(const llm_graph_params &pParams) override {
    const auto *vMctxNew = static_cast<const mgs_kv_cache_msa_context *>(pParams.vMctx);

    this->vMctx = vMctxNew;

    const int64_t cNPs = GGML_PAD((int64_t)vMctxNew->mGetNPos(), vBlk);
    const int64_t cNs = pParams.vCparams.vKvUnified ? 1 : pParams.vUbatch.vNSeqsUnq;

    const bool cDecode = pParams.vUbatch.vNTokens == cNs; // one token per stream

    bool vRes = true;

    vRes &= vBias->ne[0] * vBlk == cNPs;
    vRes &= vBias->ne[1] == pParams.vUbatch.vNTokens;

    vRes &= vPosMask->ne[0] == cNPs;
    vRes &= vPosMask->ne[1] == pParams.vUbatch.vNTokens;

    vRes &= vPosSlotI->ne[0] == cNPs;
    vRes &= vPosSlotI->ne[1] == cNs;

    vRes &= cDecode == (vPosSlotF != nullptr);
    vRes &= cDecode == (vCellBlk == nullptr);

    if (vPosSlotF) {
      vRes &= vPosSlotF->ne[0] == cNPs;
      vRes &= vPosSlotF->ne[1] == cNs;
    }

    if (vCellBlk) {
      vRes &= vCellBlk->ne[0] == (int64_t)vMctxNew->mGetBase()->mGetNKv();
      vRes &= vCellBlk->ne[1] == cNs;
    }

    return vRes;
  }

  ggml_tensor *vBias = nullptr;     // F32 [nblk, n_tokens] local-force bias (position blocks)
  ggml_tensor *vPosMask = nullptr;  // F32 [n_ps, n_tokens] 0/-inf visibility, by position
  ggml_tensor *vPosSlotI = nullptr; // I32 [n_ps, ns]       pos -> cell (get_rows index)
  ggml_tensor *vPosSlotF = nullptr; // F32 [n_ps, ns]       pos -> cell (gatherable values, decode)
  ggml_tensor *vCellBlk = nullptr;  // I32 [n_kv, ns]       cell -> position block (batch)

  const mgs_kv_cache_msa_context *vMctx;

  int vBlk;
  int vLocal;
};

// One FA call for all GQA groups (and at multi-stream decode, all streams) by mapping them onto the FA sequence dim
// (ne[3])
ggml_tensor *mgs_model_minimax_m3::graph::mBuildAttnMsaFa(ggml_tensor *pQCur, // [D, HQ, T]
                                                          ggml_tensor *pK,    // [D, n_keys, 1, C]
                                                          ggml_tensor *pV,    // [D, n_keys, 1, C]
                                                          ggml_tensor *pMask, // [n_keys, R, 1, C] f16, contiguous
                                                          int64_t pGp, float pKqScale, int pIl) const {

  const int64_t cD = pQCur->ne[0];
  const int64_t cHq = pQCur->ne[1];
  const int64_t cT = pQCur->ne[2];
  const int64_t cC = pK->ne[3];
  const int64_t cR = cHq * cT / (pGp * cC);
  GGML_ASSERT(pGp * cC * cR == cHq * cT);
  GGML_ASSERT(pMask->type == GGML_TYPE_F16);

  // [D, HQ, T] -> [D, Gp, C, R] -> [D, R, Gp, C]
  // batch  (C=HKV,   R=T): channel = group
  // decode (C=HKV*ns, R=1): channel = (group, stream), group innermost
  ggml_tensor *vQ = ggml_reshape_4d(vCtx0, pQCur, cD, pGp, cC, cR);
  vQ = ggml_permute(vCtx0, vQ, 0, 2, 3, 1);

  ggml_tensor *vO = ggml_flash_attn_ext(vCtx0, vQ, pK, pV, pMask, pKqScale, vHparams.vFMaxAlibiBias, 0.0f);
  ggml_flash_attn_ext_set_prec(vO, GGML_PREC_F32);
  mCb(vO, "msa_fattn", pIl);

  // [D, Gp, R, C] -> [D, Gp, C, R] -> [n_embd, T]
  vO = ggml_permute(vCtx0, vO, 0, 1, 3, 2);
  if (!ggml_is_contiguous(vO)) {
    vO = ggml_cont(vCtx0, vO); // no-op layout at decode (R == 1), copy at batch
  }
  return ggml_reshape_2d(vCtx0, vO, cD * cHq, cT);
}

mgs_model_minimax_m3::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  const auto &vMm = static_cast<const mgs_model_minimax_m3 &>(pModel);

  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());
  // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  ggml_tensor *vInpPos = mBuildInpPos();

  // ==========================================
  // TODO: avoid such kind of complexity in the model graphs

  // MSA calls ggml_flash_attn_ext directly and assumes the non-transposed V layout that
  // mgs only provides when flash attention is enabled. Block selection is anchored
  // to absolute KV cache slots, which equal positions only for append-only per-stream
  // caches either a single sequence, or multiple sequences with kv_unified == false (each
  // stream then has its own slot space). A unified cache with multiple sequences
  // interleaves slots and would silently break block anchoring so it falls back to dense.
  const bool cFaOn = vCparams.vFlashAttn;
  const bool cStreamsOk = vCparams.vNSeqMax == 1 || !vCparams.vKvUnified;
  const bool cMsaEnabled = cFaOn && cStreamsOk;

  auto *vInpAttn = mBuildAttnInpKvMsa(cMsaEnabled);

  static bool vWarnedNoFa = false;
  if (!cFaOn && !vWarnedNoFa) {
    MGS_LOG_WARN("%s: flash attention disabled; MSA requires it -> running DENSE attention "
                 "(output may be degraded). Enable flash attention for MSA.\n",
                 __func__);
    vWarnedNoFa = true;
  }
  static bool vWarnedUnified = false;
  if (cFaOn && !cStreamsOk && !vWarnedUnified) {
    MGS_LOG_WARN("%s: unified KV cache with n_seq_max > 1; MSA needs per-sequence streams "
                 "-> running DENSE attention. Output may be degraded. Drop --kv-unified to enable MSA.\n",
                 __func__);
    vWarnedUnified = true;
  }
  // ==========================================

  // hoisted per-graph MSA state (shared by every sparse layer)
  llm_graph_input_msa *vMsa = nullptr;
  ggml_tensor *vMsaKqm = nullptr;
  ggml_tensor *vMsaMf = nullptr; // F32 copy of the FA mask for the final mask add
  int64_t vNKv = 0, vNPs = 0, vNblk = 0, vNs = 1, vNTps = 0;
  bool vMsaDecode = false; // gather (1 token per stream) vs mask
  const int cBlk = vMm.vMsaP.vBlk;
  const int64_t cHd = vHparams.vIndexerNHead; // one indexer head per GQA group

  if (cMsaEnabled) {
    const auto *vMctxMsa = static_cast<const mgs_kv_cache_msa_context *>(vMctx);

    vMsaKqm = vInpAttn->mGetKqMask();
    vNKv = vMsaKqm->ne[0];
    vNTps = vMsaKqm->ne[1]; // tokens per stream
    vNs = vMsaKqm->ne[3];   // streams in this ubatch
    GGML_ASSERT(vMsaKqm->type == GGML_TYPE_F16 && "MSA requires the FA (f16) mask");
    GGML_ASSERT(vNTps * vNs == cNTokens);

    // the position axis covers every position currently in the cache and is padded to whole blocks
    vNPs = GGML_PAD((int64_t)vMctxMsa->mGetNPos(), cBlk);
    vNblk = vNPs / cBlk;
    vMsaDecode = vNTps == 1;

    auto vInp = std::make_unique<llm_graph_input_msa>(vMctxMsa, cBlk, vMm.vMsaP.vLocal);

    vInp->vBias = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vNblk, cNTokens); // stream-grouped tokens
    ggml_set_input(vInp->vBias);

    vInp->vPosMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vNPs, cNTokens);
    ggml_set_input(vInp->vPosMask);

    vInp->vPosSlotI = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, vNPs, vNs);
    ggml_set_input(vInp->vPosSlotI);

    if (vMsaDecode) {
      vInp->vPosSlotF = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vNPs, vNs);
      ggml_set_input(vInp->vPosSlotF);
    } else {
      vInp->vCellBlk = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, vNKv, vNs);
      ggml_set_input(vInp->vCellBlk);

      vMsaMf = ggml_cast(vCtx0, vMsaKqm, GGML_TYPE_F32);
    }

    vMsa = (llm_graph_input_msa *)vRes->mAddInput(std::move(vInp));
  }

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // self-attention
    {
      vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
      mCb(vCur, "attn_norm", vIl);

      auto [vQcur, vKcur, vVcur] = mBuildQkv(pModel.lLayers[vIl], vCur, cNEmbdHead, cNHead, cNHeadKv, vIl);

      // per-head QK RMSNorm (weights already include Gemma's +1)
      vQcur = mBuildNorm(vQcur, pModel.lLayers[vIl].vAttnQNorm, NULL, cLlmNormRms, vIl);
      mCb(vQcur, "Qcur_normed", vIl);
      vKcur = mBuildNorm(vKcur, pModel.lLayers[vIl].vAttnKNorm, NULL, cLlmNormRms, vIl);
      mCb(vKcur, "Kcur_normed", vIl);

      // partial rotary: only the first n_rot dims are rotated
      vQcur = ggml_rope_ext(vCtx0, vQcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      vKcur = ggml_rope_ext(vCtx0, vKcur, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      const bool cIsSparse = cMsaEnabled && vIl >= (int)vHparams.vNLayerDenseLead;

      if (!cIsSparse) {
        vCur = mBuildAttn(vInpAttn, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur, nullptr,
                          nullptr, nullptr, 1.0f / sqrtf(float(cNEmbdHead)), vIl);
      } else {
        const int64_t cNIdxDim = vHparams.vIndexerHeadSize; // 128

        // Index Branch, project, norm, partial RoPE, cache
        ggml_tensor *vIq = mBuildLoraMm(pModel.lLayers[vIl].vIndexQProj, vCur);
        ggml_tensor *vIk = mBuildLoraMm(pModel.lLayers[vIl].vIndexKProj, vCur);
        vIq = ggml_reshape_3d(vCtx0, vIq, cNIdxDim, cHd, cNTokens);
        vIk = ggml_reshape_3d(vCtx0, vIk, cNIdxDim, 1, cNTokens);
        vIq = mBuildNorm(vIq, pModel.lLayers[vIl].vIndexQNorm, NULL, cLlmNormRms, vIl); // +1 baked
        vIk = mBuildNorm(vIk, pModel.lLayers[vIl].vIndexKNorm, NULL, cLlmNormRms, vIl);
        vIq = ggml_rope_ext(vCtx0, vIq, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        vIk = ggml_rope_ext(vCtx0, vIk, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                            cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

        const auto *vMctxMsaL = static_cast<const mgs_kv_cache_msa_context *>(vMctx);
        const auto *vMctxCur = vMctxMsaL->mGetBase();
        const auto *vMctxIdx = vMctxMsaL->mGetIdx();
        ggml_build_forward_expand(vGf, vMctxIdx->mCpyK(vCtx0, vIk, vInpAttn->mGetKIdxsIdx(), vIl));
        ggml_tensor *vIkKv = vMctxIdx->mGetK(vCtx0, vIl);

        if (vInpAttn->vSelfKRot) {
          vQcur = mgs_mul_mat_hadamard(vCtx0, vQcur, vInpAttn->vSelfKRot);
          vKcur = mgs_mul_mat_hadamard(vCtx0, vKcur, vInpAttn->vSelfKRot);
        }
        if (vInpAttn->vSelfVRot) {
          vVcur = mgs_mul_mat_hadamard(vCtx0, vVcur, vInpAttn->vSelfVRot);
        }

        // Main branch: store K/V, take cache views
        ggml_build_forward_expand(vGf, vQcur);
        ggml_build_forward_expand(vGf, vKcur);
        ggml_build_forward_expand(vGf, vVcur);
        ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, vKcur, vInpAttn->mGetKIdxs(), vIl));
        ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, vVcur, vInpAttn->mGetVIdxs(), vIl));
        ggml_tensor *vK = vMctxCur->mGetK(vCtx0, vIl);
        ggml_tensor *vV = vMctxCur->mGetV(vCtx0, vIl);
        GGML_ASSERT(!(vV->nb[1] > vV->nb[2]) && "MSA assumes v_trans=false (FA on)");

        const int64_t cD = vK->ne[0];
        const int64_t cHkv = vK->ne[1];
        const int64_t cGp = cNHead / cHkv;
        GGML_ASSERT(cHkv == cHd && "MSA: one indexer head per GQA group");
        GGML_ASSERT(vK->ne[3] == vNs);
        const int cK = vMm.vMsaP.vTopkBlocks < (int)vNblk ? vMm.vMsaP.vTopkBlocks : (int)vNblk;

        const float cKqScale = 1.0f / sqrtf(float(cNEmbdHead));

        if (vMsaDecode) {
          // decode: batched over streams top-k + gather, one grouped FA
          // gather the indexer keys through the pos -> cell map
          ggml_tensor *vIk3 = ggml_view_3d(vCtx0, vIkKv, cNIdxDim, vNKv, vNs, vIkKv->nb[2], vIkKv->nb[3], 0);
          ggml_tensor *vIkp = ggml_get_rows(vCtx0, vIk3, vMsa->vPosSlotI); // [n_idx_dim, n_ps, ns]
          ggml_tensor *vIq4 = ggml_reshape_4d(vCtx0, vIq, cNIdxDim, cHd, 1, vNs);
          ggml_tensor *vSc = ggml_mul_mat(vCtx0, ggml_reshape_4d(vCtx0, vIkp, cNIdxDim, vNPs, 1, vNs), vIq4);
          ggml_mul_mat_set_prec(vSc, GGML_PREC_F32);
          // unmapped positions come out -inf, so they can never rank into the top-k
          vSc = ggml_add_inplace(vCtx0, vSc, ggml_reshape_4d(vCtx0, vMsa->vPosMask, vNPs, 1, 1, vNs));
          ggml_tensor *vBs = ggml_pool_2d(vCtx0, vSc, GGML_OP_POOL_MAX, cBlk, 1, cBlk, 1, 0, 0);
          mCb(vBs, "msa_bs", vIl);

          ggml_tensor *vBsf = ggml_add(vCtx0, vBs, ggml_reshape_4d(vCtx0, vMsa->vBias, vNblk, 1, 1, vNs));
          ggml_tensor *vIdx = ggml_top_k(vCtx0, vBsf, cK); // position blocks

          // pos idx:  tj[t,k,h,s] = blk*idx[k,h,s] + t   (positions - mask gather)
          // cell idx: cs[t,k,h,s] = pos_slot[tj]         (pos -> cell translation)
          // row idx:  tr[t,k,h,s] = cs*HKV + h           (per-stream K/V gather)
          ggml_tensor *vA = ggml_scale(vCtx0, ggml_cast(vCtx0, vIdx, GGML_TYPE_F32), (float)cBlk);
          vA = ggml_reshape_4d(vCtx0, vA, 1, cK, cHd, vNs);
          ggml_tensor *vTj = ggml_add(vCtx0, ggml_repeat_4d(vCtx0, vA, cBlk, cK, cHd, vNs),
                                      ggml_reshape_3d(vCtx0, ggml_arange(vCtx0, 0.0f, (float)cBlk, 1.0f), cBlk, 1, 1));

          ggml_tensor *vTokj =
              ggml_cast(vCtx0, ggml_reshape_2d(vCtx0, vTj, (int64_t)cBlk * cK * cHd, vNs), GGML_TYPE_I32);

          ggml_tensor *vCs =
              ggml_get_rows(vCtx0, ggml_reshape_3d(vCtx0, vMsa->vPosSlotF, 1, vNPs, vNs), vTokj); // [1, blk*K*Hd, ns]
          vCs = ggml_reshape_4d(vCtx0, vCs, cBlk, cK, cHd, vNs);

          ggml_tensor *vTr = ggml_add(vCtx0, ggml_scale(vCtx0, vCs, (float)cHkv),
                                      ggml_reshape_3d(vCtx0, ggml_arange(vCtx0, 0.0f, (float)cHkv, 1.0f), 1, 1, cHd));

          ggml_tensor *vTokr =
              ggml_cast(vCtx0, ggml_reshape_2d(vCtx0, vTr, (int64_t)cBlk * cK * cHd, vNs), GGML_TYPE_I32);

          ggml_tensor *vK3 = ggml_view_3d(vCtx0, vK, cD, cHkv * vNKv, vNs, vK->nb[1], vK->nb[3], 0);
          ggml_tensor *v3 = ggml_view_3d(vCtx0, vV, cD, cHkv * vNKv, vNs, vV->nb[1], vV->nb[3], 0);
          ggml_tensor *vMp = ggml_reshape_3d(vCtx0, vMsa->vPosMask, 1, vNPs, vNs);

          ggml_tensor *vKg = ggml_get_rows(vCtx0, vK3, vTokr);
          ggml_tensor *vVg = ggml_get_rows(vCtx0, v3, vTokr);
          ggml_tensor *vMg = ggml_get_rows(vCtx0, vMp, vTokj);

          // fold (group, stream) onto the FA channel dim
          const ggml_type cKt = ggml_is_quantized(vK->type) ? GGML_TYPE_F16 : vK->type;
          const ggml_type cVt = ggml_is_quantized(vV->type) ? GGML_TYPE_F16 : vV->type;
          ggml_tensor *vKfa = ggml_reshape_4d(vCtx0, vKg, cD, (int64_t)cBlk * cK, 1, cHd * vNs);
          ggml_tensor *vVfa = ggml_reshape_4d(vCtx0, vVg, cD, (int64_t)cBlk * cK, 1, cHd * vNs);
          if (vKfa->type != cKt) {
            vKfa = ggml_cast(vCtx0, vKfa, cKt);
          }
          if (vVfa->type != cVt) {
            vVfa = ggml_cast(vCtx0, vVfa, cVt);
          }
          // the FA mask must be F16
          ggml_tensor *vMfa =
              ggml_cast(vCtx0, ggml_reshape_4d(vCtx0, vMg, (int64_t)cBlk * cK, 1, 1, cHd * vNs), GGML_TYPE_F16);

          vCur = mBuildAttnMsaFa(vQcur, vKfa, vVfa, vMfa, cGp, cKqScale, vIl);
        } else {
          // batch: per-stream loop
          std::vector<ggml_tensor *> lOuts(vNs);
          for (int64_t vSt = 0; vSt < vNs; ++vSt) {
            ggml_tensor *vIqS =
                ggml_view_3d(vCtx0, vIq, cNIdxDim, cHd, vNTps, vIq->nb[1], vIq->nb[2], vSt * vNTps * vIq->nb[2]);
            ggml_tensor *vIkS = ggml_view_2d(vCtx0, vIkKv, cNIdxDim, vNKv, vIkKv->nb[2], vSt * vIkKv->nb[3]);
            ggml_tensor *vPslS = ggml_view_1d(vCtx0, vMsa->vPosSlotI, vNPs, vSt * vMsa->vPosSlotI->nb[1]);
            ggml_tensor *vPmS = ggml_view_3d(vCtx0, vMsa->vPosMask, vNPs, 1, vNTps, vMsa->vPosMask->nb[1],
                                             vMsa->vPosMask->nb[1], vSt * vNTps * vMsa->vPosMask->nb[1]);
            ggml_tensor *vCbS = ggml_view_1d(vCtx0, vMsa->vCellBlk, vNKv, vSt * vMsa->vCellBlk->nb[1]);
            ggml_tensor *vMfS =
                ggml_view_3d(vCtx0, vMsaMf, vNKv, vNTps, 1, vMsaMf->nb[1], vMsaMf->nb[3], vSt * vMsaMf->nb[3]);
            ggml_tensor *vBiasS = ggml_view_3d(vCtx0, vMsa->vBias, vNblk, 1, vNTps, vMsa->vBias->nb[1],
                                               vMsa->vBias->nb[1], vSt * vNTps * vMsa->vBias->nb[1]);
            ggml_tensor *vQS =
                ggml_view_3d(vCtx0, vQcur, cD, cNHead, vNTps, vQcur->nb[1], vQcur->nb[2], vSt * vNTps * vQcur->nb[2]);
            ggml_tensor *vKS =
                ggml_view_4d(vCtx0, vK, cD, cHkv, vNKv, 1, vK->nb[1], vK->nb[2], vK->nb[3], vSt * vK->nb[3]);
            ggml_tensor *vVS =
                ggml_view_4d(vCtx0, vV, cD, cHkv, vNKv, 1, vV->nb[1], vV->nb[2], vV->nb[3], vSt * vV->nb[3]);

            // block scores: the indexer keys are gathered through the pos -> cell map first
            // scores are unscaled, only the top-k ordering matters
            ggml_tensor *vIkp = ggml_get_rows(vCtx0, vIkS, vPslS); // [n_idx_dim, n_ps]
            ggml_tensor *vSc = ggml_mul_mat(vCtx0, vIkp, ggml_reshape_2d(vCtx0, vIqS, cNIdxDim, cHd * vNTps));
            // indexer scores run in F32
            ggml_mul_mat_set_prec(vSc, GGML_PREC_F32);
            vSc = ggml_reshape_3d(vCtx0, vSc, vNPs, cHd, vNTps);
            // unmapped positions (holes, padding, empty cells) come out -inf
            vSc = ggml_add_inplace(vCtx0, vSc, vPmS);
            ggml_tensor *vBs = ggml_pool_2d(vCtx0, vSc, GGML_OP_POOL_MAX, cBlk, 1, cBlk, 1, 0, 0);
            mCb(vBs, "msa_bs", vIl);

            // bias the scores so locally-forced blocks always rank first
            ggml_tensor *vBsf = ggml_add(vCtx0, vBs, vBiasS); // [nblk, Hd, n_tps]
            mCb(vBsf, "msa_bsf", vIl);

            ggml_tensor *vIdx = ggml_top_k(vCtx0, vBsf, cK); // [K, Hd, n_tps] i32

            ggml_tensor *vNinf = ggml_cast(vCtx0, ggml_scale_bias(vCtx0, vBiasS, 0.0f, -1e30f),
                                           GGML_TYPE_F16); // [nblk, 1, n_tps]
            vNinf = ggml_repeat_4d(vCtx0, vNinf, vNblk, cHd, vNTps, 1);
            ggml_tensor *vZero = ggml_scale(vCtx0, ggml_cast(vCtx0, vIdx, GGML_TYPE_F32), 0.0f);
            ggml_tensor *vBm = ggml_set_rows(vCtx0, ggml_reshape_3d(vCtx0, vNinf, 1, vNblk, cHd * vNTps),
                                             ggml_reshape_3d(vCtx0, vZero, 1, cK, cHd * vNTps),
                                             ggml_reshape_2d(vCtx0, vIdx, cK, cHd * vNTps));
            vBm = ggml_reshape_3d(vCtx0, vBm, vNblk, cHd, vNTps);
            vBm = ggml_cont(vCtx0, ggml_permute(vCtx0, vBm, 0, 2, 1, 3)); // [nblk, n_tps, Hd]
            mCb(vBm, "msa_block_mask", vIl);

            // expand block -> cell granularity through the cell -> position block
            // map, then combine with the causal mask. empty cells are masked by the causal mask.
            ggml_tensor *vBm2 = ggml_cont(
                vCtx0, ggml_transpose(vCtx0, ggml_reshape_2d(vCtx0, vBm, vNblk, vNTps * cHd))); // [n_tps*Hd, nblk]
            ggml_tensor *vBmc = ggml_get_rows(vCtx0, vBm2, vCbS);                               // [n_tps*Hd, n_kv] F32
            ggml_tensor *vBmx = ggml_cont(vCtx0, ggml_transpose(vCtx0, vBmc));
            vBmx = ggml_reshape_3d(vCtx0, vBmx, vNKv, vNTps, cHd);
            ggml_tensor *vMask4 = ggml_add_inplace(vCtx0, vBmx, vMfS);
            vMask4 = ggml_cast(vCtx0, ggml_reshape_4d(vCtx0, vMask4, vNKv, vNTps, 1, cHd), GGML_TYPE_F16);
            mCb(vMask4, "msa_mask4", vIl);

            // cache views with groups on ne[3];
            ggml_tensor *vKfa = ggml_permute(vCtx0, vKS, 0, 3, 1, 2);
            ggml_tensor *vVfa = ggml_permute(vCtx0, vVS, 0, 3, 1, 2);

            lOuts[vSt] = mBuildAttnMsaFa(vQS, vKfa, vVfa, vMask4, cGp, cKqScale, vIl);
          }
          vCur = lOuts[0];
          for (int64_t vSt = 1; vSt < vNs; ++vSt) {
            vCur = ggml_concat(vCtx0, vCur, lOuts[vSt], 1);
          }
        }
        if (vInpAttn->vSelfVRot) {
          vCur = mgs_mul_mat_hadamard(vCtx0, vCur, vInpAttn->vSelfVRot);
        }
        mCb(vCur, "kqv_out", vIl);
        if (pModel.lLayers[vIl].vWo) {
          vCur = mBuildLoraMm(pModel.lLayers[vIl].vWo, vCur, pModel.lLayers[vIl].vWoS);
        }
      }
    }

    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      // leading dense FFN (swigluoai)
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, NULL, pModel.lLayers[vIl].vFfnGate, NULL, NULL,
                       pModel.lLayers[vIl].vFfnDown, NULL, NULL, NULL, cLlmFfnSwigluOaiMoe, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // routed experts (swigluoai MoE)
      ggml_tensor *vMoeOut = mBuildMoeFfn(
          vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
          pModel.lLayers[vIl].vFfnDownExps, pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed,
          cLlmFfnSwigluOaiMoe, vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // shared expert (swigluoai)
      ggml_tensor *vFfnShexp =
          mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL, NULL, pModel.lLayers[vIl].vFfnGateShexp, NULL, NULL,
                    pModel.lLayers[vIl].vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSwigluOaiMoe, cLlmFfnPar, vIl);
      mCb(vFfnShexp, "ffn_shexp", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(vCur, "ffn_out", vIl);
    }

    vCur = ggml_add(vCtx0, vCur, vFfnInp);

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
