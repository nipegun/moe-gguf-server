#include "models.h"

#include "mgs-memory-hybrid-idx.h"

// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.

void mgs_model_glm5_next::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvAttentionLayernormEps, vHparams.vFNormEps, false);
  pMl.mGetKey(cLlmKvAttentionKeyLengthMla, vHparams.vNEmbdHeadKMlaImpl);
  pMl.mGetKey(cLlmKvAttentionValueLengthMla, vHparams.vNEmbdHeadVMlaImpl);
  pMl.mGetKey(cLlmKvAttentionQLoraRank, vHparams.vNLoraQ);
  pMl.mGetKey(cLlmKvAttentionKvLoraRank, vHparams.vNLoraKv);
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvKdaHeadDim, vHparams.vNEmbdHeadKda);
  pMl.mGetKey(cLlmKvKdaGateLowerBound, vHparams.vKdaGateLowerBound, false);

  // NextN/MTP parameters
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_all");

  // the MLA cache holds the compressed latent
  vHparams.vNEmbdHeadVFull = vHparams.vNLoraKv;

  for (uint32_t vI = 0; vI < vHparams.vNLayerAll; ++vI) {
    vHparams.aIsRecrImpl[vI] = vHparams.mNHeadKv(vI) == 0;
  }

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc, false);
  if (vHparams.vExpertGatingFunc == cMgsExpertGatingFuncTypeNone) {
    vHparams.vExpertGatingFunc = cMgsExpertGatingFuncTypeSigmoid;
  }
  pMl.mGetKeyOrArr(cLlmKvSwigluClampExp, vHparams.aSwigluClampExp, vHparams.vNLayerAll, false);
  if (!pMl.mGetKeyOrArr(cLlmKvSwigluClampShexp, vHparams.aSwigluClampShexp, vHparams.vNLayerAll, false)) {
    vHparams.aSwigluClampShexp = vHparams.aSwigluClampExp;
  }

  // DSA indexer with k-pool compression
  pMl.mGetKey(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  pMl.mGetKey(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  pMl.mGetKey(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);
  pMl.mGetKey(cLlmKvAttentionIndexerKpool, vHparams.vIndexerKpool);
  pMl.mGetKey(cLlmKvAttentionIndexerKpoolSelectTail, vHparams.vIndexerKpoolSelectTail, false);
  GGML_ASSERT(vHparams.vIndexerKpool > 1 && vHparams.vIndexerTopK % vHparams.vIndexerKpool == 0);
  std::fill(vHparams.aIsIndexerFullImpl.begin(), vHparams.aIsIndexerFullImpl.end(), 1);
  pMl.mGetKeyOrArr(cLlmKvAttentionIndexerTypes, vHparams.aIsIndexerFullImpl, vHparams.mNLayer(), false);

  // mHC
  pMl.mGetKey(cLlmKvHyperConnectionCount, vHparams.vDsv4HcMult);
  pMl.mGetKey(cLlmKvHyperConnectionSinkhornIterations, vHparams.vDsv4HcSinkhornIters);
  pMl.mGetKey(cLlmKvHyperConnectionEpsilon, vHparams.vDsv4HcEps);
  GGML_ASSERT(vHparams.vDsv4HcMult == 4 && "mHC with hc_mult != 4 is not supported");

  switch (vHparams.mNLayer()) {
  case 45:
    vType = cLlmType320bA18b;
    break; // GLM-5.3-Flash
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_glm5_next::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcMixDim = (2 + cHc) * cHc;

  // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayerAll; ++vI) {
    auto &vLayer = lLayers[vI];

    const bool cIsNextn = vI >= cNLayer;
    const int cFlags = cIsNextn ? vMtpFlags : cTrunkFlags;

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, cFlags);
    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, cFlags);

    if (vI < cNLayer) {
      vLayer.vHcAttnFn = mCreateTensor(cTn(cLlmTensorHcAttnFn, "weight", vI), {cHc * cNEmbd, cHcMixDim}, cFlags);
      vLayer.vHcAttnBase = mCreateTensor(cTn(cLlmTensorHcAttnBase, "weight", vI), {cHcMixDim}, cFlags);
      vLayer.vHcAttnScale = mCreateTensor(cTn(cLlmTensorHcAttnScale, "weight", vI), {3}, cFlags);
      vLayer.vHcFfnFn = mCreateTensor(cTn(cLlmTensorHcFfnFn, "weight", vI), {cHc * cNEmbd, cHcMixDim}, cFlags);
      vLayer.vHcFfnBase = mCreateTensor(cTn(cLlmTensorHcFfnBase, "weight", vI), {cHcMixDim}, cFlags);
      vLayer.vHcFfnScale = mCreateTensor(cTn(cLlmTensorHcFfnScale, "weight", vI), {3}, cFlags);
    }

    const int64_t cHeadDim = vHparams.vNEmbdHeadKda;
    const int64_t cDConv = vHparams.vSsmDConv;
    const int64_t cDInner = cHeadDim * cNHead;

    if (vHparams.mIsRecr(vI)) {
      // conv1d weights: 4D first, then 3D (quantization may remove the trailing 1)
      auto fConv = [&](llm_tensor pTid) {
        ggml_tensor *vT = mCreateTensor(cTn(pTid, "weight", vI), {cDConv, 1, cDInner, 1}, cTensorNotRequired);
        return vT ? vT : mCreateTensor(cTn(pTid, "weight", vI), {cDConv, 1, cDInner}, cFlags);
      };
      vLayer.vSsmQConv = fConv(cLlmTensorSsmConv1dQ);
      vLayer.vSsmKConv = fConv(cLlmTensorSsmConv1dK);
      vLayer.vSsmVConv = fConv(cLlmTensorSsmConv1dV);

      mCreateTensorQkv(vLayer, vI, cNEmbd, cDInner, cDInner, cDInner, cFlags);

      vLayer.vSsmFA = mCreateTensor(cTn(cLlmTensorSsmFA, "weight", vI), {cNEmbd, cHeadDim}, cFlags);
      vLayer.vSsmFB = mCreateTensor(cTn(cLlmTensorSsmFB, "weight", vI), {cHeadDim, cDInner}, cFlags);
      vLayer.vSsmBeta = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", vI), {cNEmbd, cNHead}, cFlags);

      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmANoscan, vI), {cNHead}, cFlags);
      vLayer.vSsmDtB = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", vI), {cDInner}, cFlags);

      vLayer.vSsmGA = mCreateTensor(cTn(cLlmTensorSsmGA, "weight", vI), {cNEmbd, cHeadDim}, cFlags);
      vLayer.vSsmGB = mCreateTensor(cTn(cLlmTensorSsmGB, "weight", vI), {cHeadDim, cDInner}, cFlags);
      vLayer.vSsmONorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", vI), {cHeadDim}, cFlags);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cDInner, cNEmbd}, cFlags);
    } else {
      const int64_t cQLoraRank = vHparams.vNLoraQ;
      const int64_t cKvLoraRank = vHparams.vNLoraKv;
      const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
      const int64_t cNEmbdHeadVMla = vHparams.mNEmbdHeadVMla();
      const int64_t cQkRopeHeadDim = vHparams.mNRot();
      const int64_t cQkNopeHeadDim = cNEmbdHeadKMla - cQkRopeHeadDim;

      vLayer.vAttnQANorm = mCreateTensor(cTn(cLlmTensorAttnQANorm, "weight", vI), {cQLoraRank}, cFlags);
      vLayer.vAttnKvANorm = mCreateTensor(cTn(cLlmTensorAttnKvANorm, "weight", vI), {cKvLoraRank}, cFlags);

      vLayer.vWqA = mCreateTensor(cTn(cLlmTensorAttnQA, "weight", vI), {cNEmbd, cQLoraRank}, cFlags);
      vLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQB, "weight", vI), {cQLoraRank, cNHead * cNEmbdHeadKMla}, cFlags);

      vLayer.vWkvAMqa =
          mCreateTensor(cTn(cLlmTensorAttnKvAMqa, "weight", vI), {cNEmbd, cKvLoraRank + cQkRopeHeadDim}, cFlags);
      vLayer.vWkB = mCreateTensor(cTn(cLlmTensorAttnKB, "weight", vI), {cQkNopeHeadDim, cKvLoraRank, cNHead}, cFlags);
      vLayer.vWvB = mCreateTensor(cTn(cLlmTensorAttnVB, "weight", vI), {cKvLoraRank, cNEmbdHeadVMla, cNHead}, cFlags);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNHead * cNEmbdHeadVMla, cNEmbd}, cFlags);

      const int64_t cNIndexerHead = vHparams.vIndexerNHead;
      const int64_t cNEmbdIndexer = vHparams.vIndexerHeadSize;
      const int64_t cKpool = vHparams.vIndexerKpool;

      const bool cFull = vI >= cNLayer || vHparams.mIsIndexerFull(vI);
      const int cIFlags = cFlags | (cFull ? 0 : cTensorNotRequired);

      vLayer.vIndexerKNorm = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "weight", vI), {cNEmbdIndexer}, cIFlags);
      vLayer.vIndexerKNormB = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "bias", vI), {cNEmbdIndexer}, cIFlags);
      vLayer.vIndexerProj = mCreateTensor(cTn(cLlmTensorIndexerProj, "weight", vI), {cNEmbd, cNIndexerHead}, cIFlags);
      vLayer.vIndexerAttnK =
          mCreateTensor(cTn(cLlmTensorIndexerAttnK, "weight", vI), {cNEmbd, cNEmbdIndexer}, cIFlags);
      vLayer.vIndexerAttnQB = mCreateTensor(cTn(cLlmTensorIndexerAttnQB, "weight", vI),
                                            {cQLoraRank, cNIndexerHead * cNEmbdIndexer}, cIFlags);
      vLayer.vIndexerKpoolGate =
          mCreateTensor(cTn(cLlmTensorIndexerKpoolGate, "weight", vI), {cNEmbd, cNEmbdIndexer}, cIFlags);
      vLayer.vIndexerKpoolApe =
          mCreateTensor(cTn(cLlmTensorIndexerKpoolApe, "weight", vI), {cNEmbdIndexer, cKpool}, cIFlags);
    }

    if (vI < (int)vHparams.vNLayerDenseLead) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, cFlags);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, cFlags);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, cFlags);
    } else {
      const int64_t cNFfExp = vHparams.vNFfExp;
      const int64_t cNExpertShared = vHparams.vNExpertShared;

      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cFlags);

      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);

      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, cFlags);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfExp * cNExpertShared, cNEmbd}, cFlags);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, cFlags);
    }

    if (vI >= cNLayer) {
      vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, cFlags);
      vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vEmbedTokens = mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab},
                                                 cFlags | cTensorNotRequired);
      vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI),
                                                    {cNEmbd, cNVocab}, cFlags | cTensorNotRequired);
      vLayer.vNextn.vSharedHeadNorm =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, cFlags | cTensorNotRequired);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_glm5_next::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

// causal conv1d over one of Q/K/V
static ggml_tensor *fGlm5Conv1d(ggml_cgraph *pGf, ggml_context *pCtx0, ggml_tensor *pConvStatesAll,
                                ggml_tensor *pConvStateAll, int64_t pQkv, ggml_tensor *pX, ggml_tensor *pProjW,
                                ggml_tensor *pConvW, int64_t pDConv, int64_t pHeadDim, int64_t pNHead,
                                int64_t pNSeqTokens, int64_t pNSeqs, int64_t pNTokens, int64_t pKvHead,
                                int64_t pMemSize, int64_t pKRs) {
  const int64_t cDInner = pHeadDim * pNHead;
  const int64_t cConvStateSize = (pDConv - 1) * cDInner;
  const int64_t cNEmbdRTotal = 3 * cConvStateSize;

  ggml_tensor *vConvStateX = ggml_view_3d(pCtx0, pConvStateAll, pDConv - 1, cDInner, pNSeqs,
                                          (pDConv - 1) * ggml_element_size(pConvStateAll),
                                          cNEmbdRTotal * ggml_element_size(pConvStateAll),
                                          pQkv * cConvStateSize * ggml_element_size(pConvStateAll));

  ggml_tensor *vXProj = ggml_mul_mat(pCtx0, pProjW, pX);
  ggml_tensor *vX3d = ggml_reshape_3d(pCtx0, vXProj, cDInner, pNSeqTokens, pNSeqs);
  ggml_tensor *vConvX = ggml_concat(pCtx0, vConvStateX, ggml_transpose(pCtx0, vX3d), 0);

  // group s holds the conv window s tokens back.
  // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
  for (int64_t vS = 0; vS < pKRs; ++vS) {
    const int64_t cSIdx = std::max<int64_t>(0, pNSeqTokens - vS);
    ggml_tensor *vConvXS =
        ggml_view_3d(pCtx0, vConvX, pDConv - 1, cDInner, pNSeqs, vConvX->nb[1], vConvX->nb[2], cSIdx * vConvX->nb[0]);
    ggml_build_forward_expand(
        pGf, ggml_cpy(pCtx0, vConvXS,
                      ggml_view_3d(pCtx0, pConvStatesAll, pDConv - 1, cDInner, pNSeqs,
                                   (pDConv - 1) * ggml_element_size(pConvStatesAll),
                                   cNEmbdRTotal * ggml_element_size(pConvStatesAll),
                                   ((vS * pMemSize + pKvHead) * cNEmbdRTotal + pQkv * cConvStateSize) *
                                       ggml_element_size(pConvStatesAll))));
  }

  ggml_tensor *vConvWeight = ggml_reshape_2d(pCtx0, pConvW, pDConv, cDInner);
  ggml_tensor *vXcur = ggml_ssm_conv(pCtx0, vConvX, vConvWeight);
  vXcur = ggml_reshape_2d(pCtx0, vXcur, cDInner, pNTokens);
  vXcur = ggml_silu(pCtx0, vXcur);

  return ggml_reshape_4d(pCtx0, vXcur, pHeadDim, pNHead, pNSeqTokens, pNSeqs);
}

// k-pool indexer inputs
class mgs_model_glm5_next::llm_graph_input_kpool : public llm_graph_input_i {
public:
  llm_graph_input_kpool(const mgs_memory_hybrid_idx_context *pMctx, uint32_t pKpool) : vMctx(pMctx), cKpool(pKpool) {}
  virtual ~llm_graph_input_kpool() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override {
    vMctx->mGetIdx()->mSetInputKIdxs(vKIdxs, pUbatch);
    vMctx->mSetInputKpool(vPoolCells, vPoolIdxs, vPoolMask, vTailIdxs, vSelMask, vNewPoolIdxs, vNewPoolRep, pUbatch);
  }

  bool mCanReuse(const llm_graph_params &pParams) override {
    vMctx = static_cast<const mgs_memory_hybrid_idx_context *>(pParams.vMctx);

    const auto *vIdx = vMctx->mGetIdx();
    if (vIdx == nullptr) {
      return false;
    }

    bool vRes = true;

    vRes &= vKIdxs->ne[0] == pParams.vUbatch.vNTokens;
    vRes &= vPoolCells->ne[0] == vMctx->mGetNKpool();
    vRes &= vPoolMask->ne[1] == pParams.vUbatch.vNTokens;
    vRes &= vTailIdxs->ne[1] == pParams.vUbatch.vNTokens;
    // the scatter mask shape follows n_kv
    vRes &= vNKv == vIdx->mGetNKv();
    vRes &= vNNew == std::max(vMctx->mGetNKpoolNew(), 1u);

    return vRes;
  }

  ggml_tensor *vKIdxs = nullptr;     // I64     [n_tokens]
  ggml_tensor *vPoolCells = nullptr; // I32     [n_pool]         cell caching each pool's pooled key
  ggml_tensor *vPoolIdxs = nullptr;  // I32     [kpool, n_pool]  member cells per pool, n_kv sentinel for padded pools
  ggml_tensor *vPoolMask = nullptr;  // F32/F16 [n_pool, n_tokens]
  ggml_tensor *vTailIdxs = nullptr;  // I32     [kpool - 1, n_tokens]
  ggml_tensor *vSelMask = nullptr;   // F32     [n_sel, 1, 1, n_tokens] 0 for live selection slots, -inf for dead ones
  // n_new is never below 1, see mBuildInpKpool
  ggml_tensor *vNewPoolIdxs = nullptr; // I32     [kpool, n_new]   members of the pools completed this ubatch
  ggml_tensor *vNewPoolRep = nullptr;  // I64     [n_new]          cell to write each new pooled key into

  const mgs_memory_hybrid_idx_context *vMctx;
  const uint32_t cKpool;
  uint32_t vNNew = 0;
  uint32_t vNSel = 0;
  uint32_t vNKv = 0;
};

mgs_model_glm5_next::llm_graph_input_kpool *
mgs_model_glm5_next::graph::mBuildInpKpool(const mgs_memory_hybrid_idx_context *pMctxHyb) {
  const auto *vMctxIdx = pMctxHyb->mGetIdx();
  GGML_ASSERT(vMctxIdx != nullptr);

  const uint32_t cKpool = vHparams.vIndexerKpool;
  const uint32_t cNPool = pMctxHyb->mGetNKpool();
  const uint32_t cNKv = vMctxIdx->mGetNKv();
  // a ubatch that completes no pool still builds one dummy entry, so the graph does not
  // change shape every kpool tokens
  const uint32_t cNNew = std::max(pMctxHyb->mGetNKpoolNew(), 1u);

  // the fused lightning indexer wants an f16 mask
  const auto cTypeMask = vCparams.vFusedLid ? GGML_TYPE_F16 : GGML_TYPE_F32;

  auto vInp = std::make_unique<llm_graph_input_kpool>(pMctxHyb, cKpool);

  vInp->vKIdxs = vMctxIdx->mBuildInputKIdxs(vCtx0, vUbatch);
  vInp->vPoolCells = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPool);
  vInp->vPoolIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool, cNPool);
  vInp->vPoolMask = ggml_new_tensor_2d(vCtx0, cTypeMask, cNPool, cNTokens);
  vInp->vTailIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool - 1, cNTokens);
  ggml_set_input(vInp->vPoolCells);
  ggml_set_input(vInp->vPoolIdxs);
  ggml_set_input(vInp->vPoolMask);
  ggml_set_input(vInp->vTailIdxs);

  ggml_build_forward_expand(vGf, vInp->vPoolCells);
  ggml_build_forward_expand(vGf, vInp->vPoolIdxs);
  ggml_build_forward_expand(vGf, vInp->vPoolMask);
  ggml_build_forward_expand(vGf, vInp->vTailIdxs);

  vInp->vNKv = cNKv;

  // selection width: the top pools plus the optional tail, also the sparse attention bound
  {
    const int64_t cNTopPool = std::min<int64_t>(cNPool, vHparams.vIndexerTopK / cKpool);
    const int64_t cNSel = cKpool * cNTopPool + (vHparams.vIndexerKpoolSelectTail ? cKpool - 1 : 0);
    vInp->vNSel = (uint32_t)cNSel;

    // the scatter maps the dead slots of this mask to dump rows
    vInp->vSelMask = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, cNSel, 1, 1, cNTokens);
    ggml_set_input(vInp->vSelMask);
    // mSetInputKpool always fills the mask, so it stays allocated in every graph
    ggml_build_forward_expand(vGf, vInp->vSelMask);
  }

  vInp->vNNew = cNNew;

  vInp->vNewPoolIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool, cNNew);
  ggml_set_input(vInp->vNewPoolIdxs);
  // one scatter row per new pool, each a distinct rep row (see mKpoolBuildState)
  vInp->vNewPoolRep = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I64, cNNew);
  ggml_set_input(vInp->vNewPoolRep);

  return (llm_graph_input_kpool *)vRes->mAddInput(std::move(vInp));
}

static size_t fGlm5ElemOffset(const ggml_tensor *pT, int64_t pI) { return ggml_row_size(pT->type, pI); }

static ggml_tensor *fGlm5View1d(ggml_context *pCtx, ggml_tensor *pT, int64_t pNe0, int64_t pI0) {
  return ggml_view_1d(pCtx, pT, pNe0, fGlm5ElemOffset(pT, pI0));
}

static ggml_tensor *fGlm5View2d(ggml_context *pCtx, ggml_tensor *pT, int64_t pNe0, int64_t pNe1, int64_t pI0) {
  return ggml_view_2d(pCtx, pT, pNe0, pNe1, pT->nb[1], fGlm5ElemOffset(pT, pI0));
}

ggml_tensor *mgs_model_glm5_next::graph::mBuildHcMean(ggml_tensor *pX) const {
  const int64_t cHc = pX->ne[1];

  ggml_tensor *vAcc = ggml_view_2d(vCtx0, pX, pX->ne[0], pX->ne[2], pX->nb[2], 0);
  for (int64_t vS = 1; vS < cHc; ++vS) {
    vAcc = ggml_add(vCtx0, vAcc, ggml_view_2d(vCtx0, pX, pX->ne[0], pX->ne[2], pX->nb[2], vS * pX->nb[1]));
  }
  return ggml_scale(vCtx0, vAcc, 1.0f / cHc);
}

static ggml_tensor *fGlm5HcAffine(ggml_context *pCtx, ggml_tensor *pX, ggml_tensor *pScale, ggml_tensor *pBase) {
  pX = ggml_mul(pCtx, pX, pScale);
  pX = ggml_add(pCtx, pX, pBase);
  return pX;
}

ggml_tensor *mgs_model_glm5_next::graph::mBuildHcPre(ggml_tensor *pX, ggml_tensor *pWeights, int pIl) const {
  GGML_ASSERT(pX->ne[0] == cNEmbd);
  GGML_ASSERT(pX->ne[1] == vHparams.vDsv4HcMult);

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cNt = pX->ne[2];

  if (vCparams.vFusedDsv4HcPre && pIl >= 0) {
    ggml_tensor *vResult = ggml_dsv4_hc_pre(vCtx0, pX, pWeights);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPre, vResult, pIl});
    return vResult;
  }

  ggml_tensor *vResult = nullptr;
  for (int64_t vIh = 0; vIh < cHc; ++vIh) {
    ggml_tensor *vXh = ggml_view_2d(vCtx0, pX, cNEmbd, cNt, pX->nb[2], vIh * pX->nb[1]);
    ggml_tensor *vWh = ggml_view_2d(vCtx0, pWeights, 1, cNt, pWeights->nb[1], vIh * pWeights->nb[0]);
    ggml_tensor *vCur = ggml_mul(vCtx0, vXh, vWh);
    vResult = vResult ? ggml_add(vCtx0, vResult, vCur) : vCur;
  }

  return vResult;
}

ggml_tensor *mgs_model_glm5_next::graph::mBuildHcSinkhorn(ggml_tensor *pComb, int pIl) const {
  GGML_UNUSED(pIl);

  // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
  // row softmax over dst, one column normalization, then repeated row/column normalization.
  pComb = ggml_soft_max(vCtx0, pComb);

  ggml_tensor *vEps = ggml_new_tensor_1d(vCtx0, GGML_TYPE_F32, 1);
  vEps = ggml_fill(vCtx0, vEps, vHparams.vDsv4HcEps);

  pComb = ggml_add(vCtx0, pComb, vEps);

  auto fNormCols = [&]() {
    ggml_tensor *vCombSrcDst = ggml_cont(vCtx0, ggml_permute(vCtx0, pComb, 1, 0, 2, 3));
    ggml_tensor *vColSum = ggml_sum_rows(vCtx0, vCombSrcDst);
    vColSum = ggml_add(vCtx0, vColSum, vEps);
    vColSum = ggml_permute(vCtx0, vColSum, 1, 0, 2, 3);
    pComb = ggml_div(vCtx0, pComb, vColSum);
  };

  auto fNormRows = [&]() {
    ggml_tensor *vRowSum = ggml_sum_rows(vCtx0, pComb);
    vRowSum = ggml_add(vCtx0, vRowSum, vEps);
    pComb = ggml_div(vCtx0, pComb, vRowSum);
  };

  fNormCols();
  for (uint32_t vI = 1; vI < vHparams.vDsv4HcSinkhornIters; ++vI) {
    fNormRows();
    fNormCols();
  }

  return pComb;
}

ggml_tensor *mgs_model_glm5_next::graph::mBuildHcPre(ggml_tensor *pX, ggml_tensor *pHcFn, ggml_tensor *pHcScale,
                                                     ggml_tensor *pHcBase, ggml_tensor **pPost, ggml_tensor **pComb,
                                                     int pIl) const {
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;
  const int64_t cHcMixDim = (2 + cHc) * cHc;
  const int64_t cNt = pX->ne[2];

  GGML_ASSERT(cHc == 4);
  GGML_ASSERT(pHcFn->ne[1] == cHcMixDim);

  ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, pX, cHcDim, cNt);
  ggml_tensor *vFlatNorm = ggml_rms_norm(vCtx0, vFlat, cNormRmsEps);
  ggml_tensor *vMixes = ggml_mul_mat(vCtx0, pHcFn, vFlatNorm);
  mCb(vMixes, "hc_mixes", pIl);

  ggml_tensor *vScalePre = fGlm5View1d(vCtx0, pHcScale, 1, 0);
  ggml_tensor *vScalePost = fGlm5View1d(vCtx0, pHcScale, 1, 1);

  ggml_tensor *vBasePre = fGlm5View1d(vCtx0, pHcBase, cHc, 0);
  ggml_tensor *vBasePost = fGlm5View1d(vCtx0, pHcBase, cHc, cHc);

  ggml_tensor *vPre = fGlm5View2d(vCtx0, vMixes, cHc, cNt, 0);
  vPre = fGlm5HcAffine(vCtx0, vPre, vScalePre, vBasePre);
  vPre = ggml_sigmoid(vCtx0, vPre);
  vPre = ggml_scale_bias(vCtx0, vPre, 1.0f, vHparams.vDsv4HcEps);
  mCb(vPre, "hc_pre", pIl);

  *pPost = fGlm5View2d(vCtx0, vMixes, cHc, cNt, cHc);
  *pPost = fGlm5HcAffine(vCtx0, *pPost, vScalePost, vBasePost);
  *pPost = ggml_sigmoid(vCtx0, *pPost);
  *pPost = ggml_scale(vCtx0, *pPost, 2.0f);
  mCb(*pPost, "hc_post", pIl);

  if (vCparams.vFusedDsv4HcComb) {
    *pComb = ggml_dsv4_hc_comb(vCtx0, vMixes, pHcScale, pHcBase, vHparams.vDsv4HcEps,
                               (int32_t)vHparams.vDsv4HcSinkhornIters);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcComb, *pComb, pIl});
  } else {
    ggml_tensor *vScaleComb = fGlm5View1d(vCtx0, pHcScale, 1, 2);
    ggml_tensor *vBaseComb = fGlm5View1d(vCtx0, pHcBase, cHc * cHc, 2 * cHc);

    *pComb = fGlm5View2d(vCtx0, vMixes, cHc * cHc, cNt, 2 * cHc);
    *pComb = fGlm5HcAffine(vCtx0, *pComb, vScaleComb, vBaseComb);
    *pComb = ggml_reshape_3d(vCtx0, *pComb, cHc, cHc, cNt);
    *pComb = mBuildHcSinkhorn(*pComb, pIl);
  }
  mCb(*pComb, "hc_comb", pIl);

  ggml_tensor *vResult = mBuildHcPre(pX, vPre, pIl);
  return vResult;
}

ggml_tensor *mgs_model_glm5_next::graph::mBuildHcPost(ggml_tensor *pX, ggml_tensor *pResidual, ggml_tensor *pPost,
                                                      ggml_tensor *pComb, int pIl) const {
  GGML_ASSERT(pX->ne[0] == cNEmbd);
  GGML_ASSERT(pResidual->ne[1] == vHparams.vDsv4HcMult);

  if (vCparams.vFusedDsv4HcPost) {
    ggml_tensor *vResult = ggml_dsv4_hc_post(vCtx0, pX, pResidual, pPost, pComb);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPost, vResult, pIl});
    return vResult;
  }

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cNt = pX->ne[1];

  ggml_tensor *vOut = nullptr;
  for (int64_t vDst = 0; vDst < cHc; ++vDst) {
    ggml_tensor *vPostDst = ggml_view_2d(vCtx0, pPost, 1, cNt, pPost->nb[1], vDst * pPost->nb[0]);
    ggml_tensor *vCur = ggml_mul(vCtx0, pX, vPostDst);

    for (int64_t vSrc = 0; vSrc < cHc; ++vSrc) {
      ggml_tensor *vResSrc = ggml_view_2d(vCtx0, pResidual, cNEmbd, cNt, pResidual->nb[2], vSrc * pResidual->nb[1]);
      ggml_tensor *vCombSrcDst =
          ggml_view_2d(vCtx0, pComb, 1, cNt, pComb->nb[2], vDst * pComb->nb[0] + vSrc * pComb->nb[1]);
      vCur = ggml_add(vCtx0, vCur, ggml_mul(vCtx0, vResSrc, vCombSrcDst));
    }

    vCur = ggml_reshape_3d(vCtx0, vCur, cNEmbd, 1, cNt);
    vOut = vOut ? ggml_concat(vCtx0, vOut, vCur, 1) : vCur;
  }

  return vOut;
}

// construct the graph helpers without building the trunk, so graph_mtp can reuse them
mgs_model_glm5_next::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams, no_build)
    : llm_build_delta_net_base(pParams), vModel(pModel) {}

mgs_model_glm5_next::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : graph(pModel, pParams, no_build{}) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "GLM5-Next MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "GLM5-Next MTP currently supports a single NextN block");

  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");

  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && vLayer.vNextn.vEnorm && vLayer.vNextn.vHnorm &&
              "GLM5-Next MTP block is missing - load the model with MTP enabled");

  GGML_ASSERT(vHparams.mNEmbdOut() == (uint32_t)cNEmbd && "GLM5-Next MTP hidden width mismatch");

  const auto *vMctxHyb = static_cast<const mgs_memory_hybrid_idx_context *>(vMctx);

  auto *vInpHyb = mBuildInpMemHybridK();
  auto *vInpAttn = vInpHyb->mGetAttn();
  auto *vInpKpool = mBuildInpKpool(vMctxHyb);

  ggml_build_forward_expand(vGf, vInpHyb->mGetRecr()->vSCopy);

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.vNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInp(), cNTokens);
  ggml_set_input(vInp->vEmbd);

  vInp->vH = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbd, cNTokens);
  ggml_set_input(vInp->vH);
  ggml_set_name(vInp->vH, "mtp_h_input");

  ggml_tensor *vTokEmbd;
  if (vUbatch.vToken) {
    vTokEmbd =
        ggml_get_rows(vCtx0, vLayer.vNextn.vEmbedTokens ? vLayer.vNextn.vEmbedTokens : pModel.vTokEmbd, vInp->vTokens);
  } else {
    vTokEmbd = vInp->vEmbd;
  }
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  ggml_tensor *vH = vInp->vH;

  vRes->mAddInput(std::move(vInp));

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vHNorm = mBuildNorm(vH, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vCur =
      mBuildLoraMm(vLayer.vNextn.vEhProj, ggml_concat(vCtx0, vENorm, vHNorm, 0), vLayer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  ggml_tensor *vPrevSel = nullptr;
  vCur = mBuildDsaLayer(vCur, vLayer, vMctxHyb, vInpAttn, vInpKpool, &vPrevSel, cIl);
  mCb(vCur, "mtp_attn_out", cIl);

  // narrow to the output tokens before the position-wise FFN; unmasked nextn embeddings need all rows
  if (vInpOutIds && (!vCparams.vEmbeddingsNextn || vCparams.vEmbeddingsNextnMasked)) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
    vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
  }

  ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vFfnInp, "mtp_ffn_inp", cIl);

  vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_ffn_norm", cIl);

  ggml_tensor *vMoeOut =
      mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                   vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                   vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl, nullptr,
                   vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
  mCb(vMoeOut, "mtp_ffn_moe_out", cIl);

  ggml_tensor *vFfnShexp =
      mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp, nullptr,
                vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp, nullptr, vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu,
                cLlmFfnPar, cIl);
  mCb(vFfnShexp, "mtp_ffn_shexp", cIl);

  vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor *vHeadNorm = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNorm && "GLM5-Next MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNorm, nullptr, cLlmNormRms, -1);
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (vInpOutIds && vCparams.vEmbeddingsNextn && !vCparams.vEmbeddingsNextnMasked) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "GLM5-Next MTP: missing both nextn.shared_head_head and output");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}

mgs_model_glm5_next::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_delta_net_base(pParams), vModel(pModel) {
  ggml_tensor *vCur;

  ggml_tensor *vInp = mBuildInpEmbd(pModel.vTokEmbd);
  mCb(vInp, "inp_embd", -1);

  // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
  const auto *vMctxHyb = static_cast<const mgs_memory_hybrid_idx_context *>(vMctx);

  auto *vInpHyb = mBuildInpMemHybridK();
  auto *vInpRs = vInpHyb->mGetRecr();
  auto *vInpAttn = vInpHyb->mGetAttn();
  auto *vInpKpool = mBuildInpKpool(vMctxHyb);

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  const int64_t cNHeadKda = vHparams.mNHead();
  const int64_t cHeadDim = vHparams.vNEmbdHeadKda;
  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = cNHeadKda * cHeadDim;
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

  const int64_t cHc = vHparams.vDsv4HcMult;
  ggml_tensor *vInpL = ggml_reshape_3d(vCtx0, vInp, cNEmbd, 1, cNTokens);
  vInpL = ggml_repeat_4d(vCtx0, vInpL, cNEmbd, cHc, cNTokens, 1);
  mCb(vInpL, "hc_init", -1);

  ggml_tensor *vPrevSel = nullptr;

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = pModel.lLayers[vIl];

    ggml_tensor *vResidual = vInpL;
    ggml_tensor *vPost = nullptr;
    ggml_tensor *vComb = nullptr;

    vCur = mBuildHcPre(vInpL, vLayer.vHcAttnFn, vLayer.vHcAttnScale, vLayer.vHcAttnBase, &vPost, &vComb, vIl);
    mCb(vCur, "hc_attn_pre", vIl);

    vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);
    ggml_build_forward_expand(vGf, vCur);

    if (vHparams.mIsRecr(vIl)) {
      vCur = mBuildKdaLayer(vCur, vLayer, vInpRs, cDConv, cHeadDim, cNHeadKda, cDInner, cNSeqTokens, cNSeqs, vIl);
    } else {
      vCur = mBuildDsaLayer(vCur, vLayer, vMctxHyb, vInpAttn, vInpKpool, &vPrevSel, vIl);
    }

    vInpL = mBuildHcPost(vCur, vResidual, vPost, vComb, vIl);
    mCb(vInpL, "hc_attn_post", vIl);

    vResidual = vInpL;
    vCur = mBuildHcPre(vInpL, vLayer.vHcFfnFn, vLayer.vHcFfnScale, vLayer.vHcFfnBase, &vPost, &vComb, vIl);
    mCb(vCur, "hc_ffn_pre", vIl);

    vCur = mBuildNorm(vCur, vLayer.vFfnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      vCur = mBuildFfn(vCur, vLayer.vFfnUp, nullptr, vLayer.vFfnUpS, vLayer.vFfnGate, nullptr, vLayer.vFfnGateS,
                       vLayer.vFfnDown, nullptr, vLayer.vFfnDownS, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      ggml_tensor *vMoeOut =
          mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                       vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                       vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl,
                       nullptr, vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS,
                       vLayer.vFfnDownExpsS);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      ggml_tensor *vFfnShexp =
          mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp, nullptr,
                    vLayer.vFfnGateShexpS, vLayer.vFfnDownShexp, nullptr, vLayer.vFfnDownShexpS, nullptr, cLlmFfnSilu,
                    cLlmFfnPar, vIl);
      mCb(vFfnShexp, "ffn_shexp", vIl);

      vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
      mCb(vCur, "ffn_out", vIl);
    }

    vInpL = mBuildHcPost(vCur, vResidual, vPost, vComb, vIl);
    vInpL = mBuildCvec(vInpL, vIl);
    mCb(vInpL, "l_out", vIl);
  }

  // narrow to the output tokens, then collapse the streams; unmasked nextn embeddings need all rows
  if (vInpOutIds && (!vCparams.vEmbeddingsNextn || vCparams.vEmbeddingsNextnMasked)) {
    ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, vInpL, cNEmbd * cHc, cNTokens);
    vFlat = ggml_get_rows(vCtx0, vFlat, vInpOutIds);
    vInpL = ggml_reshape_3d(vCtx0, vFlat, cNEmbd, cHc, cNOutputs);
  }

  vCur = mBuildHcMean(vInpL);
  mCb(vCur, "hc_head", -1);

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);

  // the post-norm hidden state feeds the draft head
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (vInpOutIds && vCparams.vEmbeddingsNextn && !vCparams.vEmbeddingsNextnMasked) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = ggml_mul_mat(vCtx0, pModel.vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// KDA layer, g_a/g_b output gate

ggml_tensor *mgs_model_glm5_next::graph::mBuildKdaLayer(ggml_tensor *pCur, const mgs_layer &pLayer,
                                                        llm_graph_input_rs *pInpRs, int64_t pDConv, int64_t pHeadDim,
                                                        int64_t pNHeadKda, int64_t pDInner, int64_t pNSeqTokens,
                                                        int64_t pNSeqs, int pIl) {
  const auto *vMctxCur = pInpRs->vMctx;
  const auto cKvHead = vMctxCur->mGetHead();

  ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor *vConvStateAll = mBuildRs(pInpRs, vConvStatesAll, vHparams.mNEmbdR(), pNSeqs);

  const int64_t cMemSize = vMctxCur->mGetSize();
  const int64_t cKRs = (int64_t)vCparams.vNRsSeq + 1;

  ggml_tensor *vQcur = fGlm5Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 0, pCur, pLayer.vWq, pLayer.vSsmQConv,
                                   pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize, cKRs);
  ggml_tensor *vKcur = fGlm5Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 1, pCur, pLayer.vWk, pLayer.vSsmKConv,
                                   pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize, cKRs);
  ggml_tensor *vVcur = fGlm5Conv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 2, pCur, pLayer.vWv, pLayer.vSsmVConv,
                                   pDConv, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs, cNTokens, cKvHead, cMemSize, cKRs);
  mCb(vQcur, "kda_q_conv", pIl);
  mCb(vKcur, "kda_k_conv", pIl);
  mCb(vVcur, "kda_v_conv", pIl);

  // decay gate, ssm_a holds -exp(A_log)
  ggml_tensor *vFA = ggml_mul_mat(vCtx0, pLayer.vSsmFA, pCur);
  ggml_tensor *vG1 = ggml_mul_mat(vCtx0, pLayer.vSsmFB, vFA);
  vG1 = ggml_add(vCtx0, vG1, pLayer.vSsmDtB);

  ggml_tensor *vA = ggml_reshape_3d(vCtx0, pLayer.vSsmA, 1, pNHeadKda, 1);

  if (vHparams.vKdaGateLowerBound > -INFINITY) {
    vG1 = ggml_reshape_3d(vCtx0, vG1, pHeadDim, pNHeadKda, cNTokens);
    vG1 = ggml_mul(vCtx0, vG1, vA);
    vG1 = ggml_sigmoid(vCtx0, ggml_scale(vCtx0, vG1, -1.0f));
    vG1 = ggml_scale(vCtx0, vG1, vHparams.vKdaGateLowerBound);
  } else {
    vG1 = ggml_softplus(vCtx0, vG1);
    vG1 = ggml_reshape_3d(vCtx0, vG1, pHeadDim, pNHeadKda, cNTokens);
    vG1 = ggml_mul(vCtx0, vG1, vA);
  }
  mCb(vG1, "kda_g1", pIl);

  vG1 = ggml_reshape_4d(vCtx0, vG1, pHeadDim, pNHeadKda, pNSeqTokens, pNSeqs);

  ggml_tensor *vBeta = ggml_mul_mat(vCtx0, pLayer.vSsmBeta, pCur);
  vBeta = ggml_reshape_4d(vCtx0, vBeta, 1, pNHeadKda, pNSeqTokens, pNSeqs);
  vBeta = ggml_sigmoid(vCtx0, vBeta);
  mCb(vBeta, "kda_beta", pIl);

  ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(pIl);
  ggml_tensor *vState = mBuildRs(pInpRs, vSsmStatesAll, vHparams.mNEmbdS(), pNSeqs);
  vState = ggml_reshape_4d(vCtx0, vState, pHeadDim, pHeadDim, pNHeadKda, pNSeqs);

  // match the FLA l2 norm
  constexpr float cL2Eps = 1e-6f;
  vQcur = fBuildGdnL2Norm(vCtx0, vQcur, cL2Eps);
  vKcur = fBuildGdnL2Norm(vCtx0, vKcur, cL2Eps);

  ggml_tensor *vOutput = mBuildRecurrentAttn(pInpRs, vSsmStatesAll, vQcur, vKcur, vVcur, vG1, vBeta, vState, pIl);
  vOutput = ggml_cont(vCtx0, vOutput);
  mCb(vOutput, "kda_scan_out", pIl);

  // output gate, then RMSNorm(o) * Sigmoid(g2)
  ggml_tensor *vGA = ggml_mul_mat(vCtx0, pLayer.vSsmGA, pCur);
  ggml_tensor *vG2 = ggml_mul_mat(vCtx0, pLayer.vSsmGB, vGA);
  vG2 = ggml_reshape_3d(vCtx0, vG2, pHeadDim, pNHeadKda, cNTokens);

  ggml_tensor *vO = ggml_reshape_3d(vCtx0, vOutput, pHeadDim, pNHeadKda, cNTokens);
  ggml_tensor *vNormed = mBuildNorm(vO, pLayer.vSsmONorm, nullptr, cLlmNormRms, pIl);
  mCb(vG2, "kda_g2", pIl);
  mCb(vNormed, "kda_normed", pIl);
  ggml_tensor *vGated = ggml_mul(vCtx0, vNormed, ggml_sigmoid(vCtx0, vG2));

  vGated = ggml_cont_2d(vCtx0, vGated, pDInner, cNTokens);
  pCur = ggml_mul_mat(vCtx0, pLayer.vWo, vGated);
  mCb(pCur, "kda_out", pIl);

  return pCur;
}

// scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an additive mask

ggml_tensor *mgs_model_glm5_next::graph::mBuildKpoolSelect(ggml_tensor *pCur, ggml_tensor *pQr, ggml_tensor *pKqMask,
                                                           const mgs_layer &pLayer,
                                                           const mgs_memory_hybrid_idx_context *pMctxHyb,
                                                           llm_graph_input_kpool *pInpKpool, int pIl) {
  const auto *vMctxLid = pMctxHyb->mGetIdx();

  const int64_t cNIndexerHead = vHparams.vIndexerNHead;
  const int64_t cNEmbdIndexer = vHparams.vIndexerHeadSize;
  const int64_t cKpool = vHparams.vIndexerKpool;
  const int64_t cNPool = pInpKpool->vPoolCells->ne[0];
  const int64_t cNNew = pInpKpool->vNNew;

  ggml_tensor *vIq = ggml_mul_mat(vCtx0, pLayer.vIndexerAttnQB, pQr);
  vIq = ggml_reshape_3d(vCtx0, vIq, cNEmbdIndexer, cNIndexerHead, cNTokens);
  mCb(vIq, "indexer_q", pIl);

  // per-token key and pool gate scores, cached together
  ggml_tensor *vIk = ggml_mul_mat(vCtx0, pLayer.vIndexerAttnK, pCur);
  vIk = mBuildNorm(vIk, pLayer.vIndexerKNorm, pLayer.vIndexerKNormB, cLlmNorm, pIl);
  mCb(vIk, "indexer_k", pIl);

  ggml_tensor *vIg = ggml_mul_mat(vCtx0, pLayer.vIndexerKpoolGate, pCur);
  mCb(vIg, "indexer_gate", pIl);

  // cache rows store key | gate | pooled
  ggml_tensor *vPzero = ggml_fill(vCtx0, ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbdIndexer, cNTokens), 0.0f);
  ggml_tensor *vPacked = ggml_concat(vCtx0, ggml_concat(vCtx0, vIk, vIg, 0), vPzero, 0);
  vPacked = ggml_reshape_3d(vCtx0, vPacked, 3 * cNEmbdIndexer, 1, cNTokens);
  ggml_build_forward_expand(vGf, vMctxLid->mCpyK(vCtx0, vPacked, pInpKpool->vKIdxs, pIl));

  auto vKpoolCache = pMctxHyb->mGetKpoolAccess(vCtx0, pIl, cNEmbdIndexer);
  const int64_t cNKv = vMctxLid->mGetNKv();

  // pool the entries completed by this ubatch; the last one is a dummy when the ubatch completes none
  ggml_tensor *vRows = vKpoolCache.mGatherKeyGate(ggml_reshape_1d(vCtx0, pInpKpool->vNewPoolIdxs, cKpool * cNNew));
  vRows = ggml_reshape_3d(vCtx0, vRows, 2 * cNEmbdIndexer, cKpool, cNNew);

  ggml_tensor *vPk = ggml_view_3d(vCtx0, vRows, cNEmbdIndexer, cKpool, cNNew, vRows->nb[1], vRows->nb[2], 0);
  ggml_tensor *vPg = ggml_view_3d(vCtx0, vRows, cNEmbdIndexer, cKpool, cNNew, vRows->nb[1], vRows->nb[2],
                                  ggml_row_size(vRows->type, cNEmbdIndexer));

  ggml_tensor *vLogits = ggml_add(vCtx0, vPg, pLayer.vIndexerKpoolApe);
  vLogits = ggml_cont(vCtx0, ggml_permute(vCtx0, vLogits, 1, 0, 2, 3)); // [kpool, head_dim, n_new]
  // fold the pool axis into rows, soft_max maps ne[2] to CUDA gridDim.y which caps at 65535
  ggml_tensor *vProbs = ggml_soft_max(vCtx0, ggml_reshape_2d(vCtx0, vLogits, cKpool, cNEmbdIndexer * cNNew));
  vProbs = ggml_reshape_3d(vCtx0, vProbs, cKpool, cNEmbdIndexer, cNNew);

  vPk = ggml_cont(vCtx0, ggml_permute(vCtx0, vPk, 1, 0, 2, 3));
  ggml_tensor *vPooledNew = ggml_sum_rows(vCtx0, ggml_mul(vCtx0, vProbs, vPk)); // [1, head_dim, n_new]
  vPooledNew = ggml_reshape_2d(vCtx0, vPooledNew, cNEmbdIndexer, cNNew);
  mCb(vPooledNew, "indexer_pool_k_new", pIl);

  // scatter the fresh pooled keys into their rep rows, then gather all n_pool of them by cell:
  // the older pools come from the rows earlier ubatches wrote
  ggml_build_forward_expand(vGf, vKpoolCache.mScatterPooled(vPooledNew, pInpKpool->vNewPoolRep));
  ggml_tensor *vPooled = vKpoolCache.mGatherPooled(pInpKpool->vPoolCells);
  vPooled = ggml_reshape_3d(vCtx0, vPooled, cNEmbdIndexer, 1, cNPool);
  mCb(vPooled, "indexer_pool_k", pIl);

  ggml_tensor *vSelIdx = nullptr;
  {
    ggml_tensor *vWeights = ggml_mul_mat(vCtx0, pLayer.vIndexerProj, pCur);
    vWeights = ggml_scale(vCtx0, vWeights, 1.0f / sqrtf(float(cNEmbdIndexer * cNIndexerHead)));
    mCb(vWeights, "indexer_weights", pIl);

    ggml_tensor *vScore = nullptr;
    if (vCparams.vFusedLid) {
      vScore = ggml_lightning_indexer(vCtx0, vIq, vPooled, vWeights, pInpKpool->vPoolMask);
      vRes->mAddFusedNode({cLlmFusedOpLightningIndexer, vScore, pIl});
    } else {
      ggml_tensor *vQP = ggml_permute(vCtx0, vIq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
      ggml_tensor *vKP = ggml_permute(vCtx0, vPooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

      ggml_tensor *vKq = ggml_mul_mat(vCtx0, vKP, vQP);              // [n_pool, n_tokens, n_head]
      vKq = ggml_cont(vCtx0, ggml_permute(vCtx0, vKq, 2, 1, 0, 3)); // [n_head, n_tokens, n_pool]
      vScore = ggml_relu(vCtx0, vKq);
      vScore = ggml_mul(vCtx0, vScore, vWeights);
      vScore = ggml_sum_rows(vCtx0, vScore);                             // [1, n_tokens, n_pool]
      vScore = ggml_cont(vCtx0, ggml_permute(vCtx0, vScore, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
      vScore = ggml_add(vCtx0, vScore, pInpKpool->vPoolMask);
    }
    mCb(vScore, "indexer_score", pIl);

    const int64_t cNTopPool = std::min<int64_t>(cNPool, vHparams.vIndexerTopK / cKpool);
    ggml_tensor *vTopK = ggml_top_k(vCtx0, vScore, cNTopPool); // [n_top_pool, n_tokens], UNORDERED

    // the selection mask marks the first min(nv, n_top_pool) slots as the visible pools, so order the set by
    // descending score
    ggml_tensor *vSelScore =
        ggml_get_rows(vCtx0, ggml_reshape_3d(vCtx0, vScore, 1, cNPool, cNTokens), vTopK); // [1, n_top_pool, n_tokens]
    ggml_tensor *vSelOrder =
        ggml_argsort(vCtx0, ggml_reshape_2d(vCtx0, vSelScore, cNTopPool, cNTokens), GGML_SORT_ORDER_DESC);
    vTopK = ggml_get_rows(vCtx0, ggml_reshape_3d(vCtx0, ggml_cast(vCtx0, vTopK, GGML_TYPE_F32), 1, cNTopPool, cNTokens),
                          vSelOrder);
    vTopK = ggml_cast(vCtx0, ggml_cont(vCtx0, ggml_reshape_2d(vCtx0, vTopK, cNTopPool, cNTokens)), GGML_TYPE_I32);
    mCb(vTopK, "indexer_top_k", pIl);

    vSelIdx = ggml_get_rows(vCtx0, pInpKpool->vPoolIdxs,
                            ggml_reshape_1d(vCtx0, vTopK, cNTopPool * cNTokens)); // [kpool, n_top_pool*n_tokens]
    vSelIdx = ggml_reshape_2d(vCtx0, vSelIdx, cKpool * cNTopPool, cNTokens);

    if (vHparams.vIndexerKpoolSelectTail) {
      // append the incomplete tail with n_kv for missing cells
      vSelIdx = ggml_concat(vCtx0, vSelIdx, pInpKpool->vTailIdxs, 0);
    }
  }
  const int64_t cNSel = vSelIdx->ne[0];

  ggml_build_forward_expand(vGf, vSelIdx);

  ggml_tensor *vMaskAll = ggml_new_tensor_4d(vCtx0, pKqMask->type, cNKv + cNSel, 1, 1, 1);
  vMaskAll = ggml_fill(vCtx0, vMaskAll, -INFINITY);
  vMaskAll = ggml_repeat_4d(vCtx0, vMaskAll, cNKv + cNSel, cNTokens, 1, 1);
  vMaskAll = ggml_reshape_3d(vCtx0, vMaskAll, 1, cNKv + cNSel, cNTokens);

  ggml_tensor *vZeros = ggml_new_tensor_4d(vCtx0, pKqMask->type, cNSel, 1, 1, 1);
  vZeros = ggml_fill(vCtx0, vZeros, 0.0f);
  vZeros = ggml_repeat_4d(vCtx0, vZeros, cNSel, cNTokens, 1, 1);
  vZeros = ggml_reshape_3d(vCtx0, vZeros, 1, cNSel, cNTokens);

  // live slots (visible pools, real tail cells) address disjoint cells. each dead slot writes its own dump row
  // n_kv + slot, so the scatter indices of a token are unique: idx = dump + live*(idx - dump), live = exp(mask).
  GGML_ASSERT(pInpKpool->vSelMask->ne[0] == cNSel && pInpKpool->vSelMask->ne[3] == cNTokens);
  ggml_tensor *vLive = ggml_exp(vCtx0, ggml_reshape_2d(vCtx0, pInpKpool->vSelMask, cNSel, cNTokens));
  ggml_tensor *vDump = ggml_arange(vCtx0, (float)cNKv, (float)(cNKv + cNSel), 1.0f);
  ggml_tensor *vIdxF = ggml_cast(vCtx0, vSelIdx, GGML_TYPE_F32);
  vIdxF = ggml_add(vCtx0, ggml_mul(vCtx0, ggml_sub(vCtx0, vIdxF, vDump), vLive), vDump);
  vSelIdx = ggml_cast(vCtx0, vIdxF, GGML_TYPE_I32);

  ggml_tensor *vSel = ggml_set_rows(vCtx0, vMaskAll, vZeros, ggml_reshape_3d(vCtx0, vSelIdx, cNSel, cNTokens, 1));
  vSel = ggml_view_2d(vCtx0, vSel, cNKv, cNTokens, vSel->nb[2], 0);

  // fold the causal visibility in before the shared-indexer reuse
  GGML_ASSERT(pKqMask->ne[0] == cNKv && pKqMask->ne[1] * pKqMask->ne[2] * pKqMask->ne[3] == cNTokens);
  vSel = ggml_add(vCtx0, vSel, ggml_reshape_2d(vCtx0, pKqMask, cNKv, cNTokens));
  mCb(vSel, "indexer_sel", pIl);

  return vSel;
}

// nope MLA layer with sparse attention over the indexer selection

ggml_tensor *mgs_model_glm5_next::graph::mBuildDsaLayer(ggml_tensor *pCur, const mgs_layer &pLayer,
                                                        const mgs_memory_hybrid_idx_context *pMctxHyb,
                                                        llm_graph_input_attn_k *pInpAttn,
                                                        llm_graph_input_kpool *pInpKpool, ggml_tensor **pPrevSel,
                                                        int pIl) {
  const auto *vMctxMla = pMctxHyb->mGetAttn();

  const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
  const int64_t cKvLoraRank = vHparams.vNLoraKv;
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadKMla - vHparams.mNRot();
  const float cKqScale = 1.0f / sqrtf((float)cNEmbdHeadKMla);

  GGML_ASSERT(vHparams.mNRot() == 0 && "GLM5-Next MLA is nope-only");

  ggml_tensor *vQr = ggml_mul_mat(vCtx0, pLayer.vWqA, pCur);
  vQr = mBuildNorm(vQr, pLayer.vAttnQANorm, nullptr, cLlmNormRms, pIl);
  mCb(vQr, "q_resid", pIl);

  ggml_tensor *vQ = ggml_mul_mat(vCtx0, pLayer.vWqB, vQr);
  vQ = ggml_reshape_3d(vCtx0, vQ, cNEmbdHeadQkNope, cNHead, cNTokens);

  ggml_tensor *vKvCmpr = ggml_mul_mat(vCtx0, pLayer.vWkvAMqa, pCur);
  vKvCmpr = mBuildNorm(vKvCmpr, pLayer.vAttnKvANorm, nullptr, cLlmNormRms, pIl);
  vKvCmpr = ggml_reshape_3d(vCtx0, vKvCmpr, cKvLoraRank, 1, cNTokens);
  mCb(vKvCmpr, "kv_cmpr", pIl);

  // absorb wk_b so the cache holds only the latent
  ggml_tensor *vQAbsorbed = ggml_permute(vCtx0, vQ, 0, 2, 1, 3);
  vQAbsorbed = ggml_mul_mat(vCtx0, pLayer.vWkB, vQAbsorbed);
  vQAbsorbed = ggml_permute(vCtx0, vQAbsorbed, 0, 2, 1, 3);
  mCb(vQAbsorbed, "q_absorbed", pIl);

  ggml_tensor *vKqMask = pInpAttn->mGetKqMask();

  ggml_tensor *vSel = nullptr;
  if (pIl >= (int)vHparams.mNLayer() || vHparams.mIsIndexerFull(pIl)) { // the NextN block always has a full indexer
    vSel = mBuildKpoolSelect(pCur, vQr, vKqMask, pLayer, pMctxHyb, pInpKpool, pIl);
    *pPrevSel = vSel;
  } else {
    GGML_ASSERT(*pPrevSel != nullptr && "shared indexer layer must follow a full indexer layer");
    vSel = *pPrevSel;
  }

  ggml_build_forward_expand(vGf, vQAbsorbed);
  ggml_build_forward_expand(vGf, vKvCmpr);
  ggml_build_forward_expand(vGf, vMctxMla->mCpyK(vCtx0, vKvCmpr, pInpAttn->mGetKIdxs(), pIl));

  // the scatter selection already includes the causal mask
  ggml_tensor *vMask = ggml_reshape_4d(vCtx0, vSel, vKqMask->ne[0], vKqMask->ne[1], vKqMask->ne[2], vKqMask->ne[3]);
  mCb(vMask, "kq_mask_dsa", pIl);

  ggml_tensor *vK = vMctxMla->mGetK(vCtx0, pIl);
  ggml_tensor *vV =
      ggml_view_4d(vCtx0, vK, cKvLoraRank, vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor *vOut = mBuildAttnMha(vQAbsorbed, vK, vV, nullptr, vMask, nullptr, pLayer.vWvB, cKqScale, pIl);
  mCb(vOut, "kqv_out", pIl);

  vOut = ggml_mul_mat(vCtx0, pLayer.vWo, vOut);
  mCb(vOut, "attn_out", pIl);

  return vOut;
}
