#include "models.h"
#include "mgs-impl.h"
#include "mgs-memory-hybrid-idx.h"
#include "mgs-memory-recurrent.h"

#include <algorithm>
#include <array>
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void fQwen4expRequireNonzero(const mgs_model_loader &pMl, llm_kv pKid, uint32_t pValue) {
  if (pValue == 0) {
    throw std::runtime_error(fFormat("%s must be greater than zero, got %u", pMl.vLlmKv(pKid).c_str(), pValue));
  }
}

// mGetArr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void fQwen4expRequireArrLen(mgs_model_loader &pMl, llm_kv pKid, uint32_t pNMin) {
  uint32_t vNArr = 0;
  pMl.mGetArrN(pKid, vNArr, true);
  if (vNArr < pNMin) {
    throw std::runtime_error(fFormat("%s has %u entries, but at least %u are required", pMl.vLlmKv(pKid).c_str(),
                                     vNArr, pNMin));
  }
}

void mgs_model_qwen4exp::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp, false);
  pMl.mGetKey(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp, false);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);

  pMl.mGetKeyOrArr(cLlmKvRopeDimensionSections, vHparams.aRopeSections, 4, true);

  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  pMl.mGetKey(cLlmKvSsmStateSize, vHparams.vSsmDState);
  pMl.mGetKey(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  pMl.mGetKey(cLlmKvSsmGroupCount, vHparams.vSsmNGroup);
  fQwen4expRequireNonzero(pMl, cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  fQwen4expRequireNonzero(pMl, cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  fQwen4expRequireNonzero(pMl, cLlmKvSsmStateSize, vHparams.vSsmDState);
  fQwen4expRequireNonzero(pMl, cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  fQwen4expRequireNonzero(pMl, cLlmKvSsmGroupCount, vHparams.vSsmNGroup);

  // NextN/MTP: one QSA block appended after the trunk
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_all");

  // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
  pMl.mGetKey(cLlmKvHyperConnectionCount, vHparams.vDsv4HcMult);
  pMl.mGetKey(cLlmKvHyperConnectionLowRank, vHparams.vHcLowRank);
  // a count of 1 has nothing to mix: the reference implementations all raise on hc_count <= 1
  if (vHparams.vDsv4HcMult <= 1) {
    throw std::runtime_error(fFormat("%s must be greater than one, got %u",
                                     pMl.vLlmKv(cLlmKvHyperConnectionCount).c_str(), vHparams.vDsv4HcMult));
  }
  fQwen4expRequireNonzero(pMl, cLlmKvHyperConnectionLowRank, vHparams.vHcLowRank);
  vHparams.vNEmbdOutImpl = vHparams.vDsv4HcMult * vHparams.vNEmbd;

  pMl.mGetKey(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  pMl.mGetKey(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  pMl.mGetKey(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);
  fQwen4expRequireNonzero(pMl, cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  fQwen4expRequireNonzero(pMl, cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  fQwen4expRequireNonzero(pMl, cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);
  pMl.mGetKeyOrArr(cLlmKvAttentionCompressRatios, vHparams.aDsv4CompressRatios, vHparams.vNLayerAll, false);

  // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
  vHparams.vIndexerKpool = 0;
  for (uint32_t vIl = 0; vIl < vHparams.vNLayerAll; ++vIl) {
    const uint32_t cR = vHparams.aDsv4CompressRatios[vIl];
    if (cR == 0) {
      continue;
    }
    if (vHparams.vIndexerKpool != 0 && cR != vHparams.vIndexerKpool) {
      throw std::runtime_error(
          fFormat("QSA layers must share one compress ratio, got %u and %u", vHparams.vIndexerKpool, cR));
    }
    vHparams.vIndexerKpool = cR;
  }
  if (vHparams.vIndexerKpool == 1 ||
      (vHparams.vIndexerKpool > 0 && vHparams.vIndexerTopK % vHparams.vIndexerKpool != 0)) {
    throw std::runtime_error(fFormat("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                     vHparams.vIndexerKpool, vHparams.vIndexerTopK));
  }
  // the reference groups the visible tokens in cache order and always keeps the tail
  vHparams.vIndexerKpoolRow = 2; // raw key | pooled key
  vHparams.vIndexerKpoolByOrder = true;
  vHparams.vIndexerKpoolSelectTail = true;

  // PLE n-gram hash embeddings; if the key group is absent every field stays zero
  vHparams.sIsPleImpl.reset();
  vHparams.vPleNHeads = 0;

  uint32_t vNPle = 0;
  pMl.mGetArrN(cLlmKvPleLayers, vNPle, false);
  if (vNPle > 0) {
    std::vector<uint32_t> lPleLayers;
    pMl.mGetArr(cLlmKvPleLayers, lPleLayers);
    if (vNPle != 1) {
      // hparams holds one set of hash constants, so several PLE modules cannot be represented
      throw std::runtime_error(fFormat("%s lists %u layers, but only one PLE layer is supported",
                                       pMl.vLlmKv(cLlmKvPleLayers).c_str(), vNPle));
    }
    for (uint32_t vIl : lPleLayers) {
      if (vIl >= vHparams.vNLayerAll) {
        throw std::runtime_error(fFormat("PLE layer %u is out of range", vIl));
      }
      vHparams.sIsPleImpl.set(vIl);
    }

    pMl.mGetKey(cLlmKvPleNgramSize, vHparams.vPleNgramSize);
    pMl.mGetKey(cLlmKvPleHeadsPerNgram, vHparams.vPleHeadsPerNgram);
    pMl.mGetKey(cLlmKvPleConvKernel, vHparams.vPleConvKernel);
    pMl.mGetKey(cLlmKvPleEosTokenId, vHparams.vPleEosTokenId);
    // optional: files written before this key fall back to the EOS token
    pMl.mGetKey(cLlmKvPleImageTokenId, vHparams.vPleImageTokenId, false);
    // the width of one gathered PLE row
    pMl.mGetKey(cLlmKvEmbeddingLengthPerLayer, vHparams.vPleHeadDim);
    fQwen4expRequireNonzero(pMl, cLlmKvPleConvKernel, vHparams.vPleConvKernel);
    fQwen4expRequireNonzero(pMl, cLlmKvEmbeddingLengthPerLayer, vHparams.vPleHeadDim);

    vHparams.vPleNHeads = (vHparams.vPleNgramSize - 1) * vHparams.vPleHeadsPerNgram;
    if (vHparams.vPleNgramSize < 2 || vHparams.vPleNgramSize > MGS_MAX_PLE_NGRAM) {
      throw std::runtime_error(fFormat("PLE n-gram size %u is out of range", vHparams.vPleNgramSize));
    }
    if (vHparams.vPleNHeads == 0 || vHparams.vPleNHeads > MGS_MAX_PLE_HEADS) {
      throw std::runtime_error(fFormat("PLE head count %u is out of range", vHparams.vPleNHeads));
    }

    fQwen4expRequireArrLen(pMl, cLlmKvPleLayerMultipliers, vHparams.vPleNgramSize);
    fQwen4expRequireArrLen(pMl, cLlmKvPleHeadOffsets, vHparams.vPleNHeads);
    fQwen4expRequireArrLen(pMl, cLlmKvPleHeadVocabSizes, vHparams.vPleNHeads);

    pMl.mGetArr(cLlmKvPleLayerMultipliers, vHparams.aPleLayerMultipliers);

    // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
    std::array<uint64_t, MGS_MAX_PLE_HEADS> aHeadOffsets = {};
    std::array<uint64_t, MGS_MAX_PLE_HEADS> aHeadVocabSizes = {};
    pMl.mGetArr(cLlmKvPleHeadOffsets, aHeadOffsets);
    pMl.mGetArr(cLlmKvPleHeadVocabSizes, aHeadVocabSizes);
    for (uint32_t vH = 0; vH < vHparams.vPleNHeads; ++vH) {
      if (aHeadVocabSizes[vH] == 0 || aHeadOffsets[vH] > INT32_MAX || aHeadVocabSizes[vH] > INT32_MAX ||
          aHeadOffsets[vH] + aHeadVocabSizes[vH] > INT32_MAX) {
        throw std::runtime_error(fFormat("PLE head %u range does not fit the int32 row index", vH));
      }
      vHparams.aPleHeadOffsets[vH] = (uint32_t)aHeadOffsets[vH];
      vHparams.aPleHeadVocabSizes[vH] = (uint32_t)aHeadVocabSizes[vH];
    }
  }

  // linear attention everywhere except every full_attention_interval-th layer
  if (!pMl.mGetKeyOrArr(cLlmKvAttentionRecurrentLayers, vHparams.aIsRecrImpl, vHparams.vNLayerAll, false)) {
    uint32_t vFullAttnInterval = 4;
    pMl.mGetKey(cLlmKvFullAttentionInterval, vFullAttnInterval, false);
    fQwen4expRequireNonzero(pMl, cLlmKvFullAttentionInterval, vFullAttnInterval);
    for (uint32_t vI = 0; vI < vHparams.vNLayerAll; ++vI) {
      vHparams.aIsRecrImpl[vI] = (vI < vHparams.mNLayer()) && ((vI + 1) % vFullAttnInterval != 0);
    }
  }

  // the PLE conv history is a row of the recurrent cache, which linear layers alone have
  for (uint32_t vI = 0; vI < vHparams.vNLayerAll; ++vI) {
    if (vHparams.mIsPle(vI) && !vHparams.mIsRecr(vI)) {
      throw std::runtime_error(fFormat("PLE layer %u is not a linear attention layer", vI));
    }
  }

  switch (vHparams.mNLayer()) {
  case 48:
    vType = cLlmTypeA3b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_qwen4exp::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;
  const int64_t cHcLr = vHparams.vHcLowRank;

  // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
  int vTrunkFlags = 0;
  int vMtpFlags = 0;
  if (vHparams.vNLayerNextn > 0) {
    if (pMl.mGetWeight(cTn(cLlmTensorHcAttnNorm, "weight", 0).mStr().c_str()) == nullptr) {
      vTrunkFlags = cTensorNotRequired;
    }
    if (pMl.mGetWeight(cTn(cLlmTensorNextnEhProj, "weight", cNLayer).mStr().c_str()) == nullptr) {
      vMtpFlags = cTensorNotRequired;
    }
  }
  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // there is no output_norm: the final hyper-connection mixer carries it
  // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
  vHcHeadNorm =
      mCreateTensor(cTn(cLlmTensorHcHeadNorm, "weight"), {cNEmbd, cHc}, vTrunkFlags | cTensorAllowReshape);
  vHcHeadDown = mCreateTensor(cTn(cLlmTensorHcHeadDown, "weight"), {cHcDim, cHcLr}, vTrunkFlags);
  vHcHeadUp = mCreateTensor(cTn(cLlmTensorHcHeadUp, "weight"), {cHcLr, cHcDim}, vTrunkFlags);

  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  // flat [ple_head_dim, n_rows] gather target
  if (vHparams.vPleNHeads > 0) {
    // the head ranges are what the gather indexes, so they set the minimum row count
    int64_t vPleRows = 0;
    for (uint32_t vH = 0; vH < vHparams.vPleNHeads; ++vH) {
      vPleRows = std::max(vPleRows, (int64_t)vHparams.aPleHeadOffsets[vH] + vHparams.aPleHeadVocabSizes[vH]);
    }

    // the converter pads the table
    const std::string cPleName = cTn(cLlmTensorPerLayerTokenEmbd, "weight").mStr();
    if (const auto *vPleW = pMl.mGetWeight(cPleName.c_str())) {
      if (vPleW->vTensor->ne[1] < vPleRows) {
        throw std::runtime_error(fFormat("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                         cPleName.c_str(), vPleW->vTensor->ne[1], vPleRows));
      }
      vPleRows = vPleW->vTensor->ne[1];
    }

    vPerLayerTokEmbd =
        mCreateTensor(cTn(cLlmTensorPerLayerTokenEmbd, "weight"), {(int64_t)vHparams.vPleHeadDim, vPleRows}, 0);
  }

  auto vLoadBlock = [&](int pIl, int pFlags) {
    auto &vLayer = lLayers[pIl];

    const int64_t cNFfExp = vHparams.vNFfExp ? vHparams.vNFfExp : cNFf / cNExpertUsed;
    const int64_t cNFfShexp = vHparams.vNFfShexp ? vHparams.vNFfShexp : cNFf;

    const int64_t cHeadKDim = vHparams.vSsmDState;
    const int64_t cHeadVDim = vHparams.vSsmDState;
    const int64_t cNKHeads = vHparams.vSsmNGroup;
    const int64_t cNVHeads = vHparams.vSsmDtRank;
    const int64_t cKeyDim = cHeadKDim * cNKHeads;
    const int64_t cValueDim = cHeadVDim * cNVHeads;
    const int64_t cConvDim = cKeyDim * 2 + cValueDim;

    // two HC modules per layer: before the token mixer, before the MoE
    vLayer.vHcAttnNorm =
        mCreateTensor(cTn(cLlmTensorHcAttnNorm, "weight", pIl), {cNEmbd, cHc}, pFlags | cTensorAllowReshape);
    vLayer.vHcAttnDown = mCreateTensor(cTn(cLlmTensorHcAttnDown, "weight", pIl), {cHcDim, cHcLr}, pFlags);
    vLayer.vHcAttnUp = mCreateTensor(cTn(cLlmTensorHcAttnUp, "weight", pIl), {cHcLr, cHcDim}, pFlags);
    vLayer.vHcAttnInject = mCreateTensor(cTn(cLlmTensorHcAttnInject, "weight", pIl), {cHcDim, cHc}, pFlags);
    vLayer.vHcFfnNorm =
        mCreateTensor(cTn(cLlmTensorHcFfnNorm, "weight", pIl), {cNEmbd, cHc}, pFlags | cTensorAllowReshape);
    vLayer.vHcFfnDown = mCreateTensor(cTn(cLlmTensorHcFfnDown, "weight", pIl), {cHcDim, cHcLr}, pFlags);
    vLayer.vHcFfnUp = mCreateTensor(cTn(cLlmTensorHcFfnUp, "weight", pIl), {cHcLr, cHcDim}, pFlags);
    vLayer.vHcFfnInject = mCreateTensor(cTn(cLlmTensorHcFfnInject, "weight", pIl), {cHcDim, cHc}, pFlags);

    if (!vHparams.mIsRecr(pIl)) {
      // full attention: wq holds [q|gate] interleaved per head
      mCreateTensorQkv(vLayer, pIl, cNEmbd, cNEmbdHeadK * cNHead * 2, cNEmbdKGqa, cNEmbdVGqa, pFlags);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), {cNEmbdHeadK * cNHead, cNEmbd}, pFlags);

      vLayer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), {cNEmbdHeadK}, pFlags);
      vLayer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), {cNEmbdHeadK}, pFlags);

      const int64_t cIdxDim = vHparams.vIndexerHeadSize;
      vLayer.vIndexQProj = mCreateTensor(cTn(cLlmTensorIndexerQProj, "weight", pIl),
                                         {cNEmbd, (int64_t)vHparams.vIndexerNHead * cIdxDim}, pFlags);
      vLayer.vIndexKProj = mCreateTensor(cTn(cLlmTensorIndexerKProj, "weight", pIl), {cNEmbd, cIdxDim}, pFlags);
      vLayer.vIndexQNorm = mCreateTensor(cTn(cLlmTensorIndexerQNorm, "weight", pIl), {cIdxDim}, pFlags);
      vLayer.vIndexKNorm = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "weight", pIl), {cIdxDim}, pFlags);
    } else {
      vLayer.vWqkv = mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pIl), {cNEmbd, cKeyDim * 2 + cValueDim}, pFlags);
      vLayer.vWqkvGate = mCreateTensor(cTn(cLlmTensorAttnGate, "weight", pIl), {cNEmbd, cValueDim}, pFlags);
      vLayer.vSsmConv1d =
          mCreateTensor(cTn(cLlmTensorSsmConv1d, "weight", pIl), {vHparams.vSsmDConv, cConvDim}, pFlags);
      vLayer.vSsmDt = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", pIl), {vHparams.vSsmDtRank}, pFlags);
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmANoscan, pIl), {vHparams.vSsmDtRank}, pFlags);
      vLayer.vSsmBeta = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", pIl), {cNEmbd, cNVHeads}, pFlags);
      vLayer.vSsmAlpha = mCreateTensor(cTn(cLlmTensorSsmAlpha, "weight", pIl), {cNEmbd, cNVHeads}, pFlags);
      vLayer.vSsmNorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", pIl), {cHeadVDim}, pFlags);
      vLayer.vSsmOut = mCreateTensor(cTn(cLlmTensorSsmOut, "weight", pIl), {cValueDim, cNEmbd}, pFlags);
    }

    if (vHparams.mIsPle(pIl)) {
      vLayer.vPleKey = mCreateTensor(cTn(cLlmTensorPleKey, "weight", pIl), {cNEmbd, cHcDim}, pFlags);
      vLayer.vPleValue = mCreateTensor(cTn(cLlmTensorPleValue, "weight", pIl), {cNEmbd, cNEmbd}, pFlags);
      vLayer.vPleNormKey =
          mCreateTensor(cTn(cLlmTensorPleNormKey, "weight", pIl), {cNEmbd, cHc}, pFlags | cTensorAllowReshape);
      vLayer.vPleNormQuery =
          mCreateTensor(cTn(cLlmTensorPleNormQuery, "weight", pIl), {cNEmbd, cHc}, pFlags | cTensorAllowReshape);
      vLayer.vPleNormConv =
          mCreateTensor(cTn(cLlmTensorPleNormConv, "weight", pIl), {cNEmbd, cHc}, pFlags | cTensorAllowReshape);
      vLayer.vPleConv1d = mCreateTensor(cTn(cLlmTensorPleConv1d, "weight", pIl),
                                        {(int64_t)vHparams.vPleConvKernel, cHcDim}, pFlags);
    }

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", pIl), {cNEmbd, cNExpert}, pFlags);
    vLayer.vFfnDownExps =
        mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", pIl), {cNFfExp, cNEmbd, cNExpert}, pFlags);
    mCreateTensorGateUpExps(vLayer, pIl, cNEmbd, cNFfExp, cNExpert, pFlags);

    vLayer.vFfnGateInpShexp = mCreateTensor(cTn(cLlmTensorFfnGateInpShexp, "weight", pIl), {cNEmbd}, pFlags);
    vLayer.vFfnGateShexp = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", pIl), {cNEmbd, cNFfShexp}, pFlags);
    vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", pIl), {cNEmbd, cNFfShexp}, pFlags);
    vLayer.vFfnDownShexp = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", pIl), {cNFfShexp, cNEmbd}, pFlags);
  };

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vLoadBlock(vIl, vTrunkFlags);
  }

  // the MTP block: one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream
  for (int vIl = cNLayer; vIl < cNLayerAll; ++vIl) {
    vLoadBlock(vIl, vMtpFlags);

    auto &vNextn = lLayers[vIl].vNextn;
    vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vIl), {2 * cNEmbd, cNEmbd}, vMtpFlags);
    vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vIl), {cNEmbd}, vMtpFlags);
    // RMS per hc stream of the trunk residual, so the gammas load as [n_embd, hc] like the mixer norms
    vNextn.vHnorm =
        mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vIl), {cNEmbd, cHc}, vMtpFlags | cTensorAllowReshape);
    vNextn.vHcHeadNorm = mCreateTensor(cTn(cLlmTensorNextnHcHeadNorm, "weight", vIl), {cNEmbd, cHc},
                                       vMtpFlags | cTensorAllowReshape);
    vNextn.vHcHeadDown = mCreateTensor(cTn(cLlmTensorNextnHcHeadDown, "weight", vIl), {cHcDim, cHcLr}, vMtpFlags);
    vNextn.vHcHeadUp = mCreateTensor(cTn(cLlmTensorNextnHcHeadUp, "weight", vIl), {cHcLr, cHcDim}, vMtpFlags);
  }
}

std::unique_ptr<llm_graph_context> mgs_model_qwen4exp::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

bool mgs_model_qwen4exp::mCanPrefetchPle() const {
  // with mmap the CPU copy of the table is the mapping itself, so only the gathered rows are ever read from disk
  return vPerLayerTokEmbd != nullptr && vParams.load_mode == MGS_LOAD_MODE_MMAP;
}

bool mgs_model_qwen4exp::graph::mCropBeforeNextn(const ggml_tensor *pInpOutIds) const {
  return pInpOutIds != nullptr && (!vCparams.vEmbeddingsNextn || vCparams.vEmbeddingsNextnMasked);
}

bool mgs_model_qwen4exp::graph::mCropAfterNextn(const ggml_tensor *pInpOutIds) const {
  return pInpOutIds != nullptr && vCparams.vEmbeddingsNextn && !vCparams.vEmbeddingsNextnMasked;
}

// hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms
// returns the mixed [n_embd, T] stream; pInject gets the [hc, T] scatter weights
ggml_tensor *mgs_model_qwen4exp::graph::mBuildHcMix(ggml_tensor *pX, ggml_tensor *pWNorm, ggml_tensor *pWDown,
                                                    ggml_tensor *pWUp, ggml_tensor *pWInject, ggml_tensor **pInject,
                                                    int pIl) {
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;
  const int64_t cNt = pX->ne[2];

  // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
  // the converter folded each gamma to (1 + w)
  ggml_tensor *vXn = ggml_mul(vCtx0, ggml_rms_norm(vCtx0, pX, vHparams.vFNormRmsEps), pWNorm);
  vXn = ggml_reshape_2d(vCtx0, vXn, cHcDim, cNt);
  mCb(vXn, "hc_norm", pIl);

  ggml_tensor *vLo = mBuildLoraMm(pWDown, vXn);
  vLo = ggml_silu(vCtx0, ggml_scale(vCtx0, vLo, 1.0f / (float)cHc));
  ggml_tensor *vGate = mBuildLoraMm(pWUp, vLo);
  mCb(vGate, "hc_gate", pIl);

  ggml_tensor *vMixed = nullptr;
  if (vCparams.vFusedDsv4HcPre && pIl >= 0) {
    // sigmoid gate and mean over the streams in one op
    vMixed = ggml_dsv4_hc_pre_gated(vCtx0, ggml_reshape_3d(vCtx0, vXn, cNEmbd, cHc, cNt),
                                    ggml_reshape_3d(vCtx0, vGate, cNEmbd, cHc, cNt), 1.0f / (float)cHc);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPre, vMixed, pIl});
  } else {
    ggml_tensor *vGated = ggml_mul(vCtx0, vXn, ggml_sigmoid(vCtx0, vGate));
    vGated = ggml_reshape_3d(vCtx0, vGated, cNEmbd, cHc, cNt);

    // collapse the streams by their mean
    vMixed = ggml_view_2d(vCtx0, vGated, cNEmbd, cNt, ggml_row_size(vGated->type, cNEmbd) * cHc, 0);
    vMixed = ggml_cont(vCtx0, vMixed);
    for (int64_t vC = 1; vC < cHc; ++vC) {
      ggml_tensor *vS = ggml_view_2d(vCtx0, vGated, cNEmbd, cNt, ggml_row_size(vGated->type, cNEmbd) * cHc,
                                     ggml_row_size(vGated->type, cNEmbd) * vC);
      vMixed = ggml_add(vCtx0, vMixed, vS);
    }
    vMixed = ggml_scale(vCtx0, vMixed, 1.0f / (float)cHc);
  }
  mCb(vMixed, "hc_mixed", pIl);

  if (pInject) {
    *pInject = mBuildLoraMm(pWInject, vXn);
    mCb(*pInject, "hc_inject", pIl);
  }

  return vMixed;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildHcCombine(ggml_tensor *pResidual, ggml_tensor *pBlockOut,
                                                        ggml_tensor *pInject, int pIl) {
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cNt = pResidual->ne[2];

  // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
  ggml_tensor *vW = ggml_sigmoid(vCtx0, ggml_scale(vCtx0, pInject, 1.0f / (float)cHc));
  vW = ggml_scale(vCtx0, vW, 2.0f);

  ggml_tensor *vCur = nullptr;
  if (vCparams.vFusedDsv4HcPost && pIl >= 0) {
    // identity comb: every stream adds the same block output, scaled by its own weight
    vCur = ggml_dsv4_hc_post(vCtx0, pBlockOut, pResidual, vW, nullptr);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPost, vCur, pIl});
  } else {
    vW = ggml_reshape_3d(vCtx0, vW, 1, cHc, cNt);

    ggml_tensor *vB = ggml_reshape_3d(vCtx0, pBlockOut, cNEmbd, 1, cNt);
    vB = ggml_repeat_4d(vCtx0, vB, cNEmbd, cHc, cNt, 1);

    vCur = ggml_add(vCtx0, pResidual, ggml_mul(vCtx0, vB, vW));
  }
  mCb(vCur, "hc_combine", pIl);

  return vCur;
}

mgs_model_qwen4exp::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_delta_net_base(pParams), vModel(pModel) {
  const int64_t cHc = vHparams.vDsv4HcMult;

  GGML_ASSERT(vHparams.mNEmbdHeadV() == vHparams.mNEmbdHeadK());

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  auto *vInp = mBuildInpMemHybrid();

  // qwen4exp always builds mgs_memory_hybrid_idx, so this downcast is safe
  // the indexer cache inside it is absent when the GGUF has no indexer tensors
  const auto *vMctxHyb = static_cast<const mgs_memory_hybrid_idx_context *>(vInp->vMctx);

  const mgs_kv_cache_context *vMctxIdx = vMctxHyb->mGetIdx();
  if (vMctxIdx) {
    GGML_ASSERT(vMctxIdx->mGetNKv() == vInp->vMctx->mGetAttn()->mGetNKv() &&
                "the indexer cache must track the attention cache cell for cell");
  }

  ggml_tensor *vPleEmb = nullptr;
  if (vHparams.vPleNHeads > 0) {
    vPleEmb = mBuildInpPle(vMctxHyb);
    // make sure vPleEmb and mBuildInpEmbd are in the same graph split
    ggml_build_forward_expand(vGf, vPleEmb);
  }

  ggml_tensor *vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  mCb(vInpL, "model.input_embed", -1);
  ggml_build_forward_expand(vGf, vInpL);

  // the QSA layers share one set of k-pool inputs
  llm_graph_input_kpool *vInpKpool = nullptr;
  if (vMctxIdx && vHparams.vIndexerKpool > 0) {
    vInpKpool = mBuildInpKpool(vMctxHyb);
  }

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // the wide residual starts as hc identical copies of the embedding
  ggml_tensor *vResHc = ggml_repeat_4d(vCtx0, ggml_reshape_3d(vCtx0, vInpL, cNEmbd, 1, cNTokens), cNEmbd, cHc,
                                       cNTokens, 1);
  mCb(vResHc, "hc_init", -1);
  // make sure hc_init is in the same graph split as the first layer
  ggml_build_forward_expand(vGf, vResHc);

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vRes->lTLayerInp[vIl] = vResHc;

    if (vHparams.mIsPle(vIl)) {
      vResHc = mBuildPle(vInp->mGetRecr(), vPleEmb, vResHc, vIl);
    }

    ggml_tensor *vInject = nullptr;
    ggml_tensor *vCur = mBuildHcMix(vResHc, pModel.lLayers[vIl].vHcAttnNorm, pModel.lLayers[vIl].vHcAttnDown,
                                    pModel.lLayers[vIl].vHcAttnUp, pModel.lLayers[vIl].vHcAttnInject, &vInject, vIl);

    ggml_build_forward_expand(vGf, vCur);

    if (vHparams.mIsRecr(vIl)) {
      vCur = mBuildLayerAttnLinear(vInp->mGetRecr(), vCur, vIl);
    } else {
      vCur = mBuildLayerAttn(vInp->mGetAttn(), vMctxHyb, vInpKpool, vCur, vInpPos, aSections, vIl);
    }

    if (vIl == cNLayer - 1 && mCropBeforeNextn(vInpOutIds)) {
      // everything below is per token, so drop the rows that produce no output
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInject = ggml_get_rows(vCtx0, vInject, vInpOutIds);

      vResHc = ggml_reshape_2d(vCtx0, vResHc, cNEmbd * cHc, vResHc->ne[2]);
      vResHc = ggml_get_rows(vCtx0, vResHc, vInpOutIds);
      vResHc = ggml_reshape_3d(vCtx0, vResHc, cNEmbd, cHc, vResHc->ne[1]);
    }

    vResHc = mBuildHcCombine(vResHc, vCur, vInject, vIl);

    vCur = mBuildHcMix(vResHc, pModel.lLayers[vIl].vHcFfnNorm, pModel.lLayers[vIl].vHcFfnDown,
                       pModel.lLayers[vIl].vHcFfnUp, pModel.lLayers[vIl].vHcFfnInject, &vInject, vIl);

    vCur = mBuildLayerFfn(vCur, vIl);
    mCb(vCur, "ffn_out", vIl);

    vResHc = mBuildHcCombine(vResHc, vCur, vInject, vIl);

    // "l_last" pins the last op of the layer to the layer's device
    mCb(vResHc, "l_last", vIl);
  }

  // the MTP head reads the hc-wide residual, before the final mixer
  vRes->vTHNextn = ggml_reshape_2d(vCtx0, vResHc, cNEmbd * cHc, vResHc->ne[2]);
  mCb(vRes->vTHNextn, "h_nextn", -1);
  ggml_build_forward_expand(vGf, vRes->vTHNextn);

  if (mCropAfterNextn(vInpOutIds)) {
    vResHc = ggml_reshape_2d(vCtx0, vResHc, cNEmbd * cHc, vResHc->ne[2]);
    vResHc = ggml_get_rows(vCtx0, vResHc, vInpOutIds);
    vResHc = ggml_reshape_3d(vCtx0, vResHc, cNEmbd, cHc, vResHc->ne[1]);
  }

  // the final mixer is the output norm: there is no separate one
  ggml_tensor *vCur = mBuildHcMix(vResHc, pModel.vHcHeadNorm, pModel.vHcHeadDown, pModel.vHcHeadUp, nullptr,
                                  nullptr, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

mgs_model_qwen4exp::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : graph(pModel, pParams, no_build{}) {
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "qwen4exp MTP has a single block");
  GGML_ASSERT(vUbatch.vToken && "qwen4exp MTP requires token input");

  const int64_t cHc = vHparams.vDsv4HcMult;
  GGML_ASSERT(vHparams.mNEmbdOut() == (uint32_t)(cNEmbd * cHc) && "qwen4exp MTP hidden width mismatch");

  const int cIl = vHparams.mNLayer();
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && vLayer.vNextn.vEnorm && vLayer.vNextn.vHnorm && vLayer.vNextn.vHcHeadNorm &&
              "MTP block missing, load the model with MTP enabled");

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.mNEmbdOut());

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdOut(), cNTokens);
  ggml_set_input(vInp->vEmbd);

  vInp->vH = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdOut(), cNTokens);
  ggml_set_input(vInp->vH);
  ggml_set_name(vInp->vH, "mtp_h_input");

  ggml_tensor *vTokEmbd = ggml_get_rows(vCtx0, pModel.vTokEmbd, vInp->vTokens);
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  ggml_tensor *vH = vInp->vH;

  vRes->mAddInput(std::move(vInp));

  auto *vInpHyb = mBuildInpMemHybrid();
  const auto *vMctxHyb = static_cast<const mgs_memory_hybrid_idx_context *>(vInpHyb->vMctx);

  // the draft memory has no recurrent layer, but its input still has to be allocated
  ggml_build_forward_expand(vGf, vInpHyb->mGetRecr()->vSCopy);

  llm_graph_input_kpool *vInpKpool = nullptr;
  if (vMctxHyb->mGetIdx() && vHparams.vIndexerKpool > 0) {
    GGML_ASSERT(vMctxHyb->mGetIdx()->mGetNKv() == vMctxHyb->mGetAttn()->mGetNKv() &&
                "the indexer cache must track the attention cache cell for cell");
    vInpKpool = mBuildInpKpool(vMctxHyb);
  }

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  ggml_tensor *vHNorm =
      mBuildNorm(ggml_reshape_3d(vCtx0, vH, cNEmbd, cHc, cNTokens), vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  vENorm = ggml_repeat_4d(vCtx0, ggml_reshape_3d(vCtx0, vENorm, cNEmbd, 1, cNTokens), cNEmbd, cHc, cNTokens, 1);
  mCb(vENorm, "mtp_enorm", cIl);

  // [n_embd, hc, n_tokens]
  ggml_tensor *vResHc =
      mBuildLoraMm(vLayer.vNextn.vEhProj, ggml_concat(vCtx0, vENorm, vHNorm, 0), vLayer.vNextn.vEhProjS);
  mCb(vResHc, "mtp_eh_proj", cIl);

  ggml_tensor *vInject = nullptr;
  ggml_tensor *vCur = mBuildHcMix(vResHc, vLayer.vHcAttnNorm, vLayer.vHcAttnDown, vLayer.vHcAttnUp,
                                  vLayer.vHcAttnInject, &vInject, cIl);
  vCur = mBuildLayerAttn(vInpHyb->mGetAttn(), vMctxHyb, vInpKpool, vCur, vInpPos, aSections, cIl);
  vResHc = mBuildHcCombine(vResHc, vCur, vInject, cIl);

  vCur = mBuildHcMix(vResHc, vLayer.vHcFfnNorm, vLayer.vHcFfnDown, vLayer.vHcFfnUp, vLayer.vHcFfnInject, &vInject,
                     cIl);
  vCur = mBuildLayerFfn(vCur, cIl);
  vResHc = mBuildHcCombine(vResHc, vCur, vInject, cIl);

  // the next draft step reads this residual as its h
  ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, vResHc, cNEmbd * cHc, cNTokens);
  ggml_tensor *vFlatOut = vInpOutIds ? ggml_get_rows(vCtx0, vFlat, vInpOutIds) : vFlat;
  vRes->vTHNextn = vCparams.vEmbeddingsNextnMasked ? vFlatOut : vFlat;
  mCb(vRes->vTHNextn, "h_nextn", cIl);
  ggml_build_forward_expand(vGf, vRes->vTHNextn);

  vCur = mBuildHcMix(ggml_reshape_3d(vCtx0, vFlatOut, cNEmbd, cHc, vFlatOut->ne[1]), vLayer.vNextn.vHcHeadNorm,
                     vLayer.vNextn.vHcHeadDown, vLayer.vNextn.vHcHeadUp, nullptr, nullptr, cIl);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

std::pair<ggml_tensor *, ggml_tensor *> mgs_model_qwen4exp::graph::mBuildQkvz(ggml_tensor *pInput, int pIl) {
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  ggml_tensor *vQkvMixed = mBuildLoraMm(vModel.lLayers[pIl].vWqkv, pInput, vModel.lLayers[pIl].vWqkvS);
  vQkvMixed = ggml_reshape_3d(vCtx0, vQkvMixed, vQkvMixed->ne[0], cNSeqTokens, cNSeqs);
  mCb(vQkvMixed, "linear_attn_qkv_mixed", pIl);

  ggml_tensor *vZ = mBuildLoraMm(vModel.lLayers[pIl].vWqkvGate, pInput, vModel.lLayers[pIl].vWqkvGateS);
  mCb(vZ, "z", pIl);

  return {vQkvMixed, vZ};
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildNormGated(ggml_tensor *pInput, ggml_tensor *pWeights,
                                                        ggml_tensor *pGate, int pLayer) {
  // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
  ggml_tensor *vNormalized = mBuildNorm(pInput, pWeights, nullptr, cLlmNormRms, pLayer);
  ggml_tensor *vGated = ggml_sigmoid(vCtx0, pGate);

  return ggml_mul(vCtx0, vNormalized, vGated);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order,
// see mgs_memory_hybrid_idx
class mgs_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
  llm_graph_input_kpool(const mgs_memory_hybrid_idx_context *pMctx, uint32_t pKpool) : vMctx(pMctx), cKpool(pKpool) {}
  virtual ~llm_graph_input_kpool() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override {
    vMctx->mGetIdx()->mSetInputKIdxs(vKIdxs, pUbatch);
    vMctx->mSetInputKpool(vPoolCells, vPoolIdxs, vPoolMask, vTailIdxs, nullptr, vNewPoolIdxs, vNewPoolRep, pUbatch,
                          vNewPoolPos);
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
    vRes &= vNNew == vMctx->mGetNKpoolNew();

    return vRes;
  }

  ggml_tensor *vKIdxs = nullptr;       // I64 [n_tokens]
  ggml_tensor *vPoolCells = nullptr;   // I32 [n_pool]         cell caching each block's pooled key
  ggml_tensor *vPoolIdxs = nullptr;    // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for padding
  ggml_tensor *vPoolMask = nullptr;    // F16 [n_pool, n_tokens]
  ggml_tensor *vTailIdxs = nullptr;    // I32 [kpool - 1, n_tokens]
  ggml_tensor *vNewPoolIdxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
  ggml_tensor *vNewPoolRep = nullptr;  // I64 [n_new]          cell to write each new pooled key into
  ggml_tensor *vNewPoolPos = nullptr;  // I32 [4*n_new]        M-RoPE position of each new block's first member

  const mgs_memory_hybrid_idx_context *vMctx;
  const uint32_t cKpool;
  uint32_t vNNew = 0; // padded to a stable bound, never below 1
  uint32_t vNSel = 0;
  uint32_t vNKv = 0;
};

mgs_model_qwen4exp::llm_graph_input_kpool *
mgs_model_qwen4exp::graph::mBuildInpKpool(const mgs_memory_hybrid_idx_context *pMctxHyb) {
  const auto *vMctxIdx = pMctxHyb->mGetIdx();
  GGML_ASSERT(vMctxIdx != nullptr);

  const uint32_t cKpool = vHparams.vIndexerKpool;
  const uint32_t cNPool = pMctxHyb->mGetNKpool();

  auto vInp = std::make_unique<llm_graph_input_kpool>(pMctxHyb, cKpool);

  vInp->vKIdxs = vMctxIdx->mBuildInputKIdxs(vCtx0, vUbatch);
  vInp->vPoolCells = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPool);
  vInp->vPoolIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool, cNPool);
  vInp->vPoolMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F16, cNPool, cNTokens);
  vInp->vTailIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool - 1, cNTokens);
  ggml_set_input(vInp->vPoolCells);
  ggml_set_input(vInp->vPoolIdxs);
  ggml_set_input(vInp->vPoolMask);
  ggml_set_input(vInp->vTailIdxs);

  // mSetInput fills them all, so keep them allocated even when no op reads them
  ggml_build_forward_expand(vGf, vInp->vPoolCells);
  ggml_build_forward_expand(vGf, vInp->vPoolIdxs);
  ggml_build_forward_expand(vGf, vInp->vPoolMask);
  ggml_build_forward_expand(vGf, vInp->vTailIdxs);

  vInp->vNKv = vMctxIdx->mGetNKv();
  vInp->vNNew = pMctxHyb->mGetNKpoolNew();
  // the top blocks plus the tail
  vInp->vNSel = cKpool * std::min<uint32_t>(cNPool, vHparams.vIndexerTopK / cKpool) + cKpool - 1;

  vInp->vNewPoolIdxs = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cKpool, vInp->vNNew);
  ggml_set_input(vInp->vNewPoolIdxs);
  // one scatter row per new pool, each a distinct rep row (see mKpoolBuildState)
  vInp->vNewPoolRep = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I64, vInp->vNNew);
  ggml_set_input(vInp->vNewPoolRep);
  vInp->vNewPoolPos = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, 4 * vInp->vNNew);
  ggml_set_input(vInp->vNewPoolPos);

  return (llm_graph_input_kpool *)vRes->mAddInput(std::move(vInp));
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
ggml_tensor *mgs_model_qwen4exp::graph::mBuildQsaSel(const mgs_memory_hybrid_idx_context *pMctxHyb,
                                                     llm_graph_input_kpool *pInpKpool, ggml_tensor *pCur,
                                                     ggml_tensor *pInpPos, ggml_tensor *pKqMask, int *pSections,
                                                     int pIl) {
  const mgs_kv_cache_context *vMctxIdx = pMctxHyb->mGetIdx();

  const int64_t cIdxDim = vHparams.vIndexerHeadSize;
  const int64_t cNIdxH = vHparams.vIndexerNHead;
  const int64_t cKpool = pInpKpool->cKpool;
  const int64_t cNPool = pInpKpool->vPoolCells->ne[0];
  const int64_t cNNew = pInpKpool->vNNew;

  GGML_ASSERT(vHparams.aDsv4CompressRatios[pIl] == cKpool);

  // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
  ggml_tensor *vKRaw = mBuildLoraMm(vModel.lLayers[pIl].vIndexKProj, pCur);
  mCb(vKRaw, "indexer_k_raw", pIl);

  ggml_tensor *vPzero = ggml_fill(vCtx0, ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cIdxDim, cNTokens), 0.0f);
  ggml_tensor *vPacked = ggml_reshape_3d(vCtx0, ggml_concat(vCtx0, vKRaw, vPzero, 0), 2 * cIdxDim, 1, cNTokens);
  ggml_build_forward_expand(vGf, vMctxIdx->mCpyK(vCtx0, vPacked, pInpKpool->vKIdxs, pIl));

  // the raw keys and the persistent pooled slots, see mgs_memory_hybrid_idx::aMemIdxStale
  auto vKpoolCache = pMctxHyb->mGetKpoolAccess(vCtx0, pIl, cIdxDim);

  // pool only the blocks this ubatch completes or regroups
  ggml_tensor *vRows =
      vKpoolCache.mGatherKeyGate(ggml_reshape_1d(vCtx0, pInpKpool->vNewPoolIdxs, cKpool * cNNew));
  vRows = ggml_reshape_3d(vCtx0, vRows, cIdxDim, cKpool, cNNew);

  // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
  ggml_tensor *vPooledNew = nullptr;
  for (int64_t vI = 0; vI < cKpool; ++vI) {
    ggml_tensor *vSlice = ggml_view_2d(vCtx0, vRows, cIdxDim, cNNew, vRows->nb[2], vI * vRows->nb[1]);
    vPooledNew = vPooledNew ? ggml_add(vCtx0, vPooledNew, vSlice) : ggml_cont(vCtx0, vSlice);
  }
  vPooledNew = ggml_scale(vCtx0, vPooledNew, 1.0f / (float)cKpool);
  vPooledNew = mBuildNorm(vPooledNew, vModel.lLayers[pIl].vIndexKNorm, nullptr, cLlmNormRms, pIl);

  vPooledNew = ggml_reshape_3d(vCtx0, vPooledNew, cIdxDim, 1, cNNew);
  vPooledNew = ggml_rope_multi(vCtx0, vPooledNew, pInpKpool->vNewPoolPos, nullptr, cNRot, pSections, cRopeType,
                               cNCtxOrig, cFreqBase, cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  vPooledNew = ggml_reshape_2d(vCtx0, vPooledNew, cIdxDim, cNNew);
  mCb(vPooledNew, "indexer_pool_k_new", pIl);

  // scatter the fresh pooled keys into their rep rows, then gather all n_pool of them by cell:
  // the older pools come from the rows earlier ubatches wrote
  ggml_build_forward_expand(vGf, vKpoolCache.mScatterPooled(vPooledNew, pInpKpool->vNewPoolRep));
  ggml_tensor *vPooled = vKpoolCache.mGatherPooled(pInpKpool->vPoolCells);
  vPooled = ggml_reshape_3d(vCtx0, vPooled, cIdxDim, 1, cNPool);
  mCb(vPooled, "indexer_k", pIl);

  ggml_tensor *vQ = mBuildLoraMm(vModel.lLayers[pIl].vIndexQProj, pCur);
  vQ = ggml_reshape_3d(vCtx0, vQ, cIdxDim, cNIdxH, cNTokens);
  vQ = mBuildNorm(vQ, vModel.lLayers[pIl].vIndexQNorm, nullptr, cLlmNormRms, pIl);
  vQ = ggml_rope_multi(vCtx0, vQ, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                       cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  mCb(vQ, "indexer_q", pIl);

  // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim),
  // which is the lightning indexer with every head weight set to that scale
  ggml_tensor *vWeights = ggml_fill(vCtx0, ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNIdxH, cNTokens),
                                    1.0f / sqrtf((float)cIdxDim));
  // [n_pool, n_tokens]
  ggml_tensor *vScore = ggml_lightning_indexer(vCtx0, vQ, vPooled, vWeights, pInpKpool->vPoolMask);
  vRes->mAddFusedNode({cLlmFusedOpLightningIndexer, vScore, pIl});
  mCb(vScore, "indexer_score", pIl);

  const int64_t cNTopPool = std::min<int64_t>(cNPool, vHparams.vIndexerTopK / cKpool);
  ggml_tensor *vTopK = ggml_top_k(vCtx0, vScore, cNTopPool); // [n_top_pool, n_tokens], unordered
  mCb(vTopK, "indexer_top_k", pIl);

  // the top blocks, then the incomplete tail with n_kv for missing cells
  ggml_tensor *vSelIdx = ggml_get_rows(vCtx0, pInpKpool->vPoolIdxs,
                                       ggml_reshape_1d(vCtx0, vTopK, cNTopPool * cNTokens)); // [kpool, n_top*n_tok]
  vSelIdx = ggml_reshape_2d(vCtx0, vSelIdx, cKpool * cNTopPool, cNTokens);
  vSelIdx = ggml_concat(vCtx0, vSelIdx, pInpKpool->vTailIdxs, 0);
  const int64_t cNSel = vSelIdx->ne[0];
  GGML_ASSERT(cNSel == pInpKpool->vNSel);

  ggml_build_forward_expand(vGf, vSelIdx);

  // scatter zeros for the selected cells into an all -inf row, each dead slot into its own dump row n_kv + slot
  // seeding from vSelIdx ties the scatter storage lifetime to this layer
  const int64_t cNKv = pInpKpool->vNKv;

  ggml_tensor *vMaskAll = ggml_new_tensor_4d(vCtx0, pKqMask->type, cNKv + cNSel, 1, 1, 1);
  vMaskAll = ggml_fill(vCtx0, vMaskAll, -INFINITY);
  vMaskAll = ggml_repeat_4d(vCtx0, vMaskAll, cNKv + cNSel, cNTokens, 1, 1);
  vMaskAll = ggml_reshape_3d(vCtx0, vMaskAll, 1, cNKv + cNSel, cNTokens);

  ggml_tensor *vZeros = ggml_new_tensor_4d(vCtx0, pKqMask->type, cNSel, 1, 1, 1);
  vZeros = ggml_fill(vCtx0, vZeros, 0.0f);
  vZeros = ggml_repeat_4d(vCtx0, vZeros, cNSel, cNTokens, 1, 1);
  vZeros = ggml_reshape_3d(vCtx0, vZeros, 1, cNSel, cNTokens);

  // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and
  // top_k fills a short selection with invisible pools that can overlap the tail, so the scatter would write
  // some cells from several threads: map every dead slot to its own dump row, idx = dump + live*(idx - dump)
  // a picked pool is live when visible: a visible score is a rectified sum >= 0, an invisible one is -inf
  ggml_tensor *vTopScore =
      ggml_get_rows(vCtx0, ggml_reshape_3d(vCtx0, vScore, 1, cNPool, cNTokens), vTopK); // [1, n_top_pool, n_tokens]
  ggml_tensor *vLivePool = ggml_clamp(vCtx0, ggml_scale_bias(vCtx0, vTopScore, 1.0f, 1.0f), 0.0f, 1.0f);
  vLivePool = ggml_reshape_2d(vCtx0, ggml_repeat_4d(vCtx0, vLivePool, cKpool, cNTopPool, cNTokens, 1),
                              cKpool * cNTopPool, cNTokens);
  // a tail cell is live unless it is the n_kv sentinel
  ggml_tensor *vLiveTail = ggml_cast(vCtx0, pInpKpool->vTailIdxs, GGML_TYPE_F32);
  vLiveTail = ggml_clamp(vCtx0, ggml_scale_bias(vCtx0, vLiveTail, -1.0f, (float)cNKv), 0.0f, 1.0f);
  ggml_tensor *vLive = ggml_concat(vCtx0, vLivePool, vLiveTail, 0); // [n_sel, n_tokens]

  // dump rows n_kv + slot as a cumulative sum: a tensor split cannot split an arange, which has no source
  ggml_tensor *vDump =
      ggml_scale_bias(vCtx0, ggml_cumsum(vCtx0, ggml_fill(vCtx0, vLive, 1.0f)), 1.0f, (float)(cNKv - 1));
  ggml_tensor *vIdxF = ggml_cast(vCtx0, vSelIdx, GGML_TYPE_F32);
  vIdxF = ggml_add(vCtx0, ggml_mul(vCtx0, ggml_sub(vCtx0, vIdxF, vDump), vLive), vDump);
  vSelIdx = ggml_cast(vCtx0, vIdxF, GGML_TYPE_I32);

  ggml_tensor *vSel = ggml_set_rows(vCtx0, vMaskAll, vZeros, ggml_reshape_3d(vCtx0, vSelIdx, cNSel, cNTokens, 1));

  GGML_ASSERT(pKqMask->ne[0] == cNKv && pKqMask->ne[1] * pKqMask->ne[2] * pKqMask->ne[3] == cNTokens);
  const size_t cRow = vSel->nb[2];
  vSel = ggml_view_4d(vCtx0, vSel, cNKv, pKqMask->ne[1], pKqMask->ne[2], pKqMask->ne[3], cRow, cRow * pKqMask->ne[1],
                      cRow * pKqMask->ne[1] * pKqMask->ne[2], 0);
  vSel = ggml_add(vCtx0, vSel, pKqMask);
  mCb(vSel, "indexer_sel", pIl);

  return vSel;
}

// dense GQA self-attention over the cells that the QSA mask keeps
ggml_tensor *mgs_model_qwen4exp::graph::mBuildAttnQsa(llm_graph_input_attn_kv *pInp, ggml_tensor *pQCur,
                                                      ggml_tensor *pKCur, ggml_tensor *pVCur, ggml_tensor *pSel,
                                                      int64_t pNSel, float pKqScale, int pIl) {
  // the dense attention reads every cell under the mask, so the selection size only bounds the live ones
  GGML_UNUSED(pNSel);

  // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
  // has already scored with its own query in mBuildQsaSel, so the selection is unaffected.
  if (pInp->vSelfKRot) {
    pQCur = mgs_mul_mat_hadamard(vCtx0, pQCur, pInp->vSelfKRot);
    pKCur = mgs_mul_mat_hadamard(vCtx0, pKCur, pInp->vSelfKRot);
  }

  if (pInp->vSelfVRot) {
    pVCur = mgs_mul_mat_hadamard(vCtx0, pVCur, pInp->vSelfVRot);
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto *vMctxCur = pInp->vMctx;

  // store to KV cache
  {
    const auto &vKIdxs = pInp->mGetKIdxs();
    const auto &vVIdxs = pInp->mGetVIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
    ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, pVCur, vVIdxs, pIl));
  }

  // the selection mask already carries the causal mask
  ggml_tensor *vKqMask = pInp->mGetKqMask();
  ggml_tensor *vMask = ggml_reshape_4d(vCtx0, pSel, vKqMask->ne[0], vKqMask->ne[1], vKqMask->ne[2], vKqMask->ne[3]);
  mCb(vMask, "kq_mask_qsa", pIl);

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV = vMctxCur->mGetV(vCtx0, pIl);

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, nullptr, vMask, nullptr, nullptr, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  // the rotation is its own inverse, so undo it on the value side of the output
  if (pInp->vSelfVRot) {
    vCur = mgs_mul_mat_hadamard(vCtx0, vCur, pInp->vSelfVRot);
  }

  return vCur;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildLayerAttn(llm_graph_input_attn_kv *pInp,
                                                        const mgs_memory_hybrid_idx_context *pMctxHyb,
                                                        llm_graph_input_kpool *pInpKpool, ggml_tensor *pCur,
                                                        ggml_tensor *pInpPos, int *pSections, int pIl) {
  const int64_t cNEmbdHead = vHparams.mNEmbdHeadV();
  GGML_ASSERT(cNEmbdHead == vHparams.mNEmbdHeadK());

  // indexer reads the same block input as q/k/v; no cache or no ratio means dense
  const bool cQsa = pInpKpool != nullptr && vHparams.aDsv4CompressRatios[pIl] > 0;

  ggml_tensor *vSel =
      cQsa ? mBuildQsaSel(pMctxHyb, pInpKpool, pCur, pInpPos, pInp->mGetKqMask(), pSections, pIl) : nullptr;

  // a single Q projection outputs query + gate, as in Qwen3Next
  ggml_tensor *vQcurFull = mBuildLoraMm(vModel.lLayers[pIl].vWq, pCur,
                                        vModel.lLayers[pIl].vWqS); // [ (n_embd_head * 2) * n_head, n_tokens ]
  mCb(vQcurFull, "Qcur_full", pIl);

  ggml_tensor *vQcur =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, 0);
  mCb(vQcur, "Qcur_reshaped", pIl);

  vQcur = mBuildNorm(vQcur, vModel.lLayers[pIl].vAttnQNorm, nullptr, cLlmNormRms, pIl);
  mCb(vQcur, "Qcur_normed", pIl);

  ggml_tensor *vKcur = mBuildLoraMm(vModel.lLayers[pIl].vWk, pCur, vModel.lLayers[pIl].vWkS);
  mCb(vKcur, "Kcur", pIl);

  ggml_tensor *vVcur = mBuildLoraMm(vModel.lLayers[pIl].vWv, pCur, vModel.lLayers[pIl].vWvS);
  mCb(vVcur, "Vcur", pIl);

  vKcur = ggml_reshape_3d(vCtx0, vKcur, cNEmbdHead, cNHeadKv, cNTokens);
  vKcur = mBuildNorm(vKcur, vModel.lLayers[pIl].vAttnKNorm, nullptr, cLlmNormRms, pIl);
  mCb(vKcur, "Kcur_normed", pIl);

  ggml_tensor *vGate =
      ggml_view_3d(vCtx0, vQcurFull, cNEmbdHead, cNHead, cNTokens, ggml_element_size(vQcurFull) * cNEmbdHead * 2,
                   ggml_element_size(vQcurFull) * cNEmbdHead * 2 * cNHead, ggml_element_size(vQcurFull) * cNEmbdHead);
  vGate = ggml_cont_2d(vCtx0, vGate, cNEmbdHead * cNHead, cNTokens);
  mCb(vGate, "gate_reshaped", pIl);

  vVcur = ggml_reshape_3d(vCtx0, vVcur, cNEmbdHead, cNHeadKv, cNTokens);

  // apply IMRoPE
  vQcur = ggml_rope_multi(vCtx0, vQcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  vKcur = ggml_rope_multi(vCtx0, vKcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "Qcur", pIl);
  mCb(vKcur, "Kcur", pIl);
  mCb(vVcur, "Vcur", pIl);

  const float cKqScale =
      vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  if (vSel) {
    pCur = mBuildAttnQsa(pInp, vQcur, vKcur, vVcur, vSel, pInpKpool->vNSel, cKqScale, pIl);
  } else {
    pCur = mBuildAttn(pInp, nullptr, nullptr, nullptr, vQcur, vKcur, vVcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  }
  mCb(pCur, "attn_pregate", pIl);

  ggml_tensor *vGateSigmoid = ggml_sigmoid(vCtx0, vGate);
  mCb(vGateSigmoid, "gate_sigmoid", pIl);

  pCur = ggml_mul(vCtx0, pCur, vGateSigmoid);
  mCb(pCur, "attn_gated", pIl);

  pCur = mBuildLoraMm(vModel.lLayers[pIl].vWo, pCur, vModel.lLayers[pIl].vWoS);
  mCb(pCur, "attn_output", pIl);

  return pCur;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildLayerAttnLinear(llm_graph_input_rs *pInp, ggml_tensor *pCur, int pIl) {
  const auto *vMctxCur = pInp->vMctx;

  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cHeadKDim = vHparams.vSsmDState;
  const int64_t cNumKHeads = vHparams.vSsmNGroup;
  const int64_t cNumVHeads = vHparams.vSsmDtRank;
  const int64_t cHeadVDim = vHparams.vSsmDState;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);
  GGML_ASSERT(cHeadVDim * cNumVHeads == cDInner);

  auto vQkvz = mBuildQkvz(pCur, pIl);
  ggml_tensor *vQkvMixed = vQkvz.first;
  ggml_tensor *vZ = vQkvz.second;

  ggml_tensor *vBeta = mBuildLoraMm(vModel.lLayers[pIl].vSsmBeta, pCur, vModel.lLayers[pIl].vSsmBetaS);
  vBeta = ggml_reshape_4d(vCtx0, vBeta, 1, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vBeta, "beta", pIl);

  vBeta = ggml_sigmoid(vCtx0, vBeta);
  mCb(vBeta, "beta_sigmoid", pIl);

  ggml_tensor *vAlpha = mBuildLoraMm(vModel.lLayers[pIl].vSsmAlpha, pCur, vModel.lLayers[pIl].vSsmAlphaS);
  vAlpha = ggml_reshape_3d(vCtx0, vAlpha, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vAlpha, "alpha", pIl);

  ggml_tensor *vAlphaBiased = ggml_add(vCtx0, vAlpha, vModel.lLayers[pIl].vSsmDt);
  ggml_tensor *vAlphaSoftplus = ggml_softplus(vCtx0, vAlphaBiased);
  mCb(vAlphaSoftplus, "a_softplus", pIl);

  ggml_tensor *vGate = ggml_mul(vCtx0, vAlphaSoftplus, vModel.lLayers[pIl].vSsmA); // -A_log.exp() * softplus
  mCb(vGate, "gate", pIl);

  vGate = ggml_reshape_4d(vCtx0, vGate, 1, cNumVHeads, cNSeqTokens, cNSeqs);

  ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(pIl);

  ggml_tensor *vConvKernel = vModel.lLayers[pIl].vSsmConv1d;
  const int64_t cConvKernelSize = vConvKernel->ne[0];

  // the channels must match how mLoadArchTensors sizes wqkv, not ssm_d_inner
  const int64_t cConvChannels = cHeadKDim * cNumKHeads * 2 + cHeadVDim * cNumVHeads;

  ggml_tensor *vConvInput =
      mBuildConvStateAt(pInp, vConvStatesAll, vQkvMixed, cConvKernelSize - 1, cConvChannels, pIl);

  ggml_tensor *vState = mBuildRs(pInp, vSsmStatesAll, vHparams.mNEmbdS(), cNSeqs);
  vState = ggml_reshape_4d(vCtx0, vState, cHeadVDim, cHeadVDim, cNumVHeads, cNSeqs);
  mCb(vState, "state_predelta", pIl);

  ggml_tensor *vConvOutputProper = ggml_ssm_conv(vCtx0, vConvInput, vConvKernel);
  mCb(vConvOutputProper, "conv_output_raw", pIl);

  ggml_tensor *vConvOutputSilu = ggml_silu(vCtx0, vConvOutputProper);
  mCb(vConvOutputSilu, "conv_output_silu", pIl);

  ggml_tensor *vConvQkvMix = vConvOutputSilu;

  int64_t vNb1Qkv = ggml_row_size(vConvQkvMix->type, cConvChannels);

  // extract the convolved Q, K, V from conv_output
  ggml_tensor *vQConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens, 0);

  ggml_tensor *vKConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                     cHeadKDim * cNumKHeads * ggml_element_size(vConvQkvMix));

  ggml_tensor *vVConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs,
                                     ggml_row_size(vConvQkvMix->type, cHeadVDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                     ggml_row_size(vConvQkvMix->type, 2 * cHeadKDim * cNumKHeads));

  mCb(vQConv, "q_conv", pIl);
  mCb(vKConv, "k_conv", pIl);
  mCb(vVConv, "v_conv", pIl);

  const float cEpsNorm = vHparams.vFNormRmsEps;

  vQConv = fBuildGdnL2Norm(vCtx0, vQConv, cEpsNorm);
  vKConv = fBuildGdnL2Norm(vCtx0, vKConv, cEpsNorm);

  // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
  if (cNumKHeads != cNumVHeads && (!vCparams.vFusedGdnAr || !vCparams.vFusedGdnCh)) {
    GGML_ASSERT(cNumVHeads % cNumKHeads == 0);
    vQConv = ggml_repeat_4d(vCtx0, vQConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
    vKConv = ggml_repeat_4d(vCtx0, vKConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
  }

  mCb(vQConv, "q_conv_predelta", pIl);
  mCb(vKConv, "k_conv_predelta", pIl);
  mCb(vVConv, "v_conv_predelta", pIl);

  ggml_tensor *vOutput = mBuildRecurrentAttn(pInp, vSsmStatesAll, vQConv, vKConv, vVConv, vGate, vBeta, vState, pIl);

  ggml_tensor *vZ2d = ggml_reshape_4d(vCtx0, vZ, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs);

  // gated normalization, as self.norm(core_attn_out, z) in the reference
  ggml_tensor *vAttnOutNorm = mBuildNormGated(vOutput, vModel.lLayers[pIl].vSsmNorm, vZ2d, pIl);

  ggml_tensor *vFinalOutput = ggml_reshape_3d(vCtx0, vAttnOutNorm, cHeadVDim * cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vFinalOutput, "final_output", pIl);

  pCur = mBuildLoraMm(vModel.lLayers[pIl].vSsmOut, vFinalOutput, vModel.lLayers[pIl].vSsmOutS);
  mCb(pCur, "linear_attn_out", pIl);

  pCur = ggml_reshape_2d(vCtx0, pCur, cNEmbd, cNSeqTokens * cNSeqs);

  return pCur;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildLayerFfn(ggml_tensor *pCur, const int pIl) {
  GGML_ASSERT(vModel.lLayers[pIl].vFfnGateInp != nullptr);

  ggml_tensor *vMoeOut = mBuildMoeFfn(
      pCur, vModel.lLayers[pIl].vFfnGateInp, vModel.lLayers[pIl].vFfnUpExps, vModel.lLayers[pIl].vFfnGateExps,
      vModel.lLayers[pIl].vFfnDownExps, nullptr, cNExpert, cNExpertUsed, cLlmFfnSilu, true,
      vHparams.vExpertWeightsScale, cMgsExpertGatingFuncTypeSoftmax, pIl, nullptr, vModel.lLayers[pIl].vFfnGateUpExps,
      vModel.lLayers[pIl].vFfnUpExpsS, vModel.lLayers[pIl].vFfnGateExpsS, vModel.lLayers[pIl].vFfnDownExpsS);
  mCb(vMoeOut, "ffn_moe_out", pIl);

  // shared experts, as in the Qwen3Next reference
  if (vModel.lLayers[pIl].vFfnUpShexp != nullptr) {
    ggml_tensor *vFfnShexp = mBuildFfn(pCur, vModel.lLayers[pIl].vFfnUpShexp, NULL, vModel.lLayers[pIl].vFfnUpShexpS,
                                       vModel.lLayers[pIl].vFfnGateShexp, NULL, vModel.lLayers[pIl].vFfnGateShexpS,
                                       vModel.lLayers[pIl].vFfnDownShexp, NULL, vModel.lLayers[pIl].vFfnDownShexpS,
                                       NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
    mCb(vFfnShexp, "ffn_shexp", pIl);

    // the shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
    ggml_tensor *vSharedGate = mBuildLoraMm(vModel.lLayers[pIl].vFfnGateInpShexp, pCur);
    mCb(vSharedGate, "shared_expert_gate", pIl);

    vSharedGate = ggml_sigmoid(vCtx0, vSharedGate);
    mCb(vSharedGate, "shared_expert_gate_sigmoid", pIl);

    vFfnShexp = ggml_mul(vCtx0, vFfnShexp, vSharedGate);
    mCb(vFfnShexp, "ffn_shexp_gated", pIl);

    pCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
    mCb(pCur, "ffn_out", pIl);
  } else {
    pCur = vMoeOut;
  }

  return pCur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
  llm_graph_input_qwen4exp_ple(const mgs_model &pModel, const mgs_kv_cache_context *pMctx, bool pPrefetch)
      : vModel(pModel), vMctx(pMctx), cPrefetch(pPrefetch) {}
  virtual ~llm_graph_input_qwen4exp_ple() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override {
    vMctx = static_cast<const mgs_memory_hybrid_idx_context *>(pParams.vMctx)->mGetAttn();
    return vRows->ne[0] == (int64_t)vModel.vHparams.vPleNHeads * pParams.vUbatch.vNTokens;
  }

  ggml_tensor *vRows = nullptr; // I32 [ple_n_heads * n_tokens]

  const mgs_model &vModel;

  // the predecessor tokens live in the attention KV cells (ext.vTok)
  const mgs_kv_cache_context *vMctx;

  // whether the gathered rows of the PLE table are prefetched
  const bool cPrefetch;

  // scratch, reused across mSetInput() calls
  std::vector<mgs_token> lPrev;
};

void llm_graph_input_qwen4exp_ple::mSetInput(const mgs_ubatch *pUbatch) {
  const auto &vHparams = vModel.vHparams;

  // an image arrives as an embd batch, so the token ids are null, but every position still needs a row for
  // ggml_get_rows: stand in the image token id that the reference hashes, or EOS if the file has no such key
  const mgs_token cImgTok =
      vHparams.vPleImageTokenId != 0 ? (mgs_token)vHparams.vPleImageTokenId : (mgs_token)vHparams.vPleEosTokenId;
  auto vTokOf = [&](int64_t pK) -> mgs_token { return pUbatch->vToken ? pUbatch->vToken[pK] : cImgTok; };

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNGram = vHparams.vPleNgramSize;
  const int64_t cNHeads = vHparams.vPleNHeads;
  const int64_t cPerGram = vHparams.vPleHeadsPerNgram;
  const int64_t cEos = vHparams.vPleEosTokenId;
  const int64_t cNPrev = cNGram - 1;

  std::vector<int32_t> lIdx(cNHeads * cNTokens);

  GGML_ASSERT(vMctx != nullptr);

  for (int64_t vI = 0; vI < cNTokens; ++vI) {
    // the preceding tokens would be ambiguous, see mGetPrevTokens()
    GGML_ASSERT(pUbatch->vNSeqId[vI] == 1 &&
                "PLE n-gram embeddings do not support tokens shared by multiple sequences");
  }

  // predecessors come from the KV cells (ext.vTok); mApplyUbatch() already stored this ubatch, so its own tokens
  // count too
  vMctx->mGetPrevTokens(*pUbatch, cNPrev, lPrev);

  std::vector<int64_t> lCtx(cNGram);
  for (int64_t vI = 0; vI < cNTokens; ++vI) {
    // an EOS in the window resets everything at or before it
    // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
    // the EOS of the token itself does not cut its own context, as in the reference
    lCtx[0] = vTokOf(vI);
    bool vCut = false;
    for (int64_t vS = 1; vS < cNGram; ++vS) {
      // predecessor s positions back; lPrev[] is oldest-first, missing entries are MGS_TOKEN_NULL
      const mgs_token cT = vCut ? MGS_TOKEN_NULL : lPrev[vI * cNPrev + (cNPrev - vS)];
      vCut = vCut || cT < 0 || cT == cEos;
      lCtx[vS] = vCut ? cEos : cT;
    }

    for (int64_t vN = 2; vN <= cNGram; ++vN) {
      uint64_t vMixed = (uint64_t)lCtx[0] * vHparams.aPleLayerMultipliers[0];
      for (int64_t vJ = 1; vJ < vN; ++vJ) {
        vMixed ^= (uint64_t)lCtx[vJ] * vHparams.aPleLayerMultipliers[vJ];
      }
      const int64_t cBase = (vN - 2) * cPerGram;
      for (int64_t vG = 0; vG < cPerGram; ++vG) {
        const int64_t cHI = cBase + vG;
        lIdx[vI * cNHeads + cHI] =
            (int32_t)(vMixed % vHparams.aPleHeadVocabSizes[cHI] + vHparams.aPleHeadOffsets[cHI]);
      }
    }
  }

  if (cPrefetch) {
    // the table stays memory-mapped on the host, so ask the kernel for the gathered rows ahead of the get_rows
    fMgsPrefetchRows(vModel.vPerLayerTokEmbd, lIdx.data(), lIdx.size());
  }

  ggml_backend_tensor_set(vRows, lIdx.data(), 0, lIdx.size() * ggml_element_size(vRows));
}

// read a conv history out of its own recurrent row and write the new tail back
// the shared mBuildConvState cannot do this: qwen4exp has two such rows per layer
ggml_tensor *mgs_model_qwen4exp::graph::mBuildConvStateAt(llm_graph_input_rs *pInp, ggml_tensor *pConvStatesAll,
                                                          ggml_tensor *pX, int64_t pStateCols, int64_t pChannels,
                                                          int pIl) {
  const auto *vMctxCur = pInp->vMctx;

  const auto cKvHead = vMctxCur->mGetHead();

  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cRowTotal = pConvStatesAll->ne[0];

  // the row is exactly this convolution's state, so the gather is reused as a whole
  GGML_ASSERT(pStateCols * pChannels == cRowTotal);

  auto vIt = dRsRows.find(pConvStatesAll);
  if (vIt == dRsRows.end()) {
    vIt = dRsRows.emplace(pConvStatesAll, mBuildRs(pInp, pConvStatesAll, cRowTotal, cNSeqs)).first;
  }
  ggml_tensor *vRows = vIt->second;

  ggml_tensor *vState = ggml_reshape_3d(vCtx0, vRows, pStateCols, pChannels, cNSeqs);
  mCb(vState, "conv_state_at", pIl);

  ggml_tensor *vConvInput = ggml_concat(vCtx0, vState, ggml_transpose(vCtx0, pX), 0);

  // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
  // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
  const size_t cRowSize = ggml_row_size(pConvStatesAll->type, cRowTotal);
  const uint32_t cMemSize = vMctxCur->mGetSize();

  const int64_t cNSlots = (int64_t)vCparams.vNRsSeq + 1;

  for (int64_t vSlot = 0; vSlot < cNSlots; ++vSlot) {
    const int64_t cSIdx = std::max<int64_t>(0, vConvInput->ne[0] - pStateCols - vSlot);

    ggml_tensor *vTail = ggml_view_3d(vCtx0, vConvInput, pStateCols, pChannels, cNSeqs, vConvInput->nb[1],
                                      vConvInput->nb[2], ggml_row_size(vConvInput->type, cSIdx));

    ggml_tensor *vDst = ggml_view_2d(vCtx0, pConvStatesAll, pStateCols * pChannels, cNSeqs, pConvStatesAll->nb[1],
                                     (vSlot * cMemSize + cKvHead) * cRowSize);

    ggml_build_forward_expand(vGf, ggml_cpy(vCtx0, ggml_cont(vCtx0, vTail), vDst));
  }

  return vConvInput;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildInpPle(const mgs_memory_hybrid_idx_context *pMctxHyb) {
  const int64_t cNHeads = vHparams.vPleNHeads;

  // the attention cells see every ubatch regardless of the layer types
  const bool cPrefetch = static_cast<const mgs_model_qwen4exp &>(vModel).mCanPrefetchPle();
  auto vPleInp = std::make_unique<llm_graph_input_qwen4exp_ple>(vModel, pMctxHyb->mGetAttn(), cPrefetch);

  vPleInp->vRows = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNHeads * cNTokens);
  ggml_set_input(vPleInp->vRows);
  ggml_tensor *vRows = vPleInp->vRows;
  vRes->mAddInput(std::move(vPleInp));

  // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
  ggml_tensor *vEmb = ggml_get_rows(vCtx0, vModel.vPerLayerTokEmbd, vRows);
  vEmb = ggml_reshape_2d(vCtx0, vEmb, vHparams.vPleHeadDim * cNHeads, cNTokens);
  mCb(vEmb, "ple_embd", -1);

  return vEmb;
}

ggml_tensor *mgs_model_qwen4exp::graph::mBuildPle(llm_graph_input_rs *pInp, ggml_tensor *pEmb, ggml_tensor *pHidden,
                                                  int pIl) {
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;

  ggml_tensor *vKey = mBuildLoraMm(vModel.lLayers[pIl].vPleKey, pEmb);
  ggml_tensor *vValue = mBuildLoraMm(vModel.lLayers[pIl].vPleValue, pEmb);

  // both norms group over one hc stream, with a [n_embd, hc] weight
  auto vGroupedNorm = [&](ggml_tensor *pX, ggml_tensor *pW) {
    ggml_tensor *vT = ggml_reshape_3d(vCtx0, pX, cNEmbd, cHc, cNTokens);
    return ggml_mul(vCtx0, ggml_rms_norm(vCtx0, vT, vHparams.vFNormRmsEps), pW);
  };

  vKey = vGroupedNorm(vKey, vModel.lLayers[pIl].vPleNormKey);
  ggml_tensor *vQuery = vGroupedNorm(pHidden, vModel.lLayers[pIl].vPleNormQuery);

  // per-stream dot product, then a signed square root before the sigmoid
  ggml_tensor *vS = ggml_sum_rows(vCtx0, ggml_mul(vCtx0, vKey, vQuery));
  vS = ggml_scale(vCtx0, vS, 1.0f / sqrtf((float)cNEmbd));

  ggml_tensor *vMag = ggml_sqrt(vCtx0, ggml_clamp(vCtx0, ggml_abs(vCtx0, vS), 1e-6f, INFINITY));
  ggml_tensor *vGate = ggml_sigmoid(vCtx0, ggml_mul(vCtx0, ggml_sgn(vCtx0, vS), vMag));
  mCb(vGate, "ple_gate", pIl);

  // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
  ggml_tensor *vV3 = ggml_reshape_3d(vCtx0, vValue, cNEmbd, 1, cNTokens);
  vV3 = ggml_repeat_4d(vCtx0, vV3, cNEmbd, cHc, cNTokens, 1);

  ggml_tensor *vGated = ggml_mul(vCtx0, vV3, vGate);
  mCb(vGated, "ple_gated_value", pIl);

  ggml_tensor *vNormalized =
      vGroupedNorm(ggml_reshape_2d(vCtx0, vGated, cHcDim, cNTokens), vModel.lLayers[pIl].vPleNormConv);
  vNormalized = ggml_reshape_2d(vCtx0, vNormalized, cHcDim, cNTokens);

  // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
  // (ggml_conv_1d_dw is not reliable for this):
  //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
  // the history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one
  const int64_t cKern = vHparams.vPleConvKernel;
  const int64_t cDil = vHparams.vPleNgramSize;
  const int64_t cHist = (cKern - 1) * cDil;

  // the conv history is per sequence, so the input carries the sequence axis too
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
  ggml_tensor *vPadded = mBuildConvStateAt(pInp, pInp->vMctx->mGetPL(pIl),
                                           ggml_reshape_3d(vCtx0, vNormalized, cHcDim, cNSeqTokens, cNSeqs), cHist,
                                           cHcDim, pIl);

  ggml_tensor *vConvOut = nullptr;
  for (int64_t vK = 0; vK < cKern; ++vK) {
    // tap k reads (kern-1-k)*dilation positions back
    const int64_t cStart = cHist - (cKern - 1 - vK) * cDil;

    ggml_tensor *vShifted =
        ggml_cont(vCtx0, ggml_transpose(vCtx0, ggml_view_3d(vCtx0, vPadded, cNSeqTokens, cHcDim, cNSeqs,
                                                            vPadded->nb[1], vPadded->nb[2],
                                                            ggml_row_size(vPadded->type, cStart))));

    // column k of the [kern, hc_dim] kernel is one weight per channel
    ggml_tensor *vWk = ggml_cont(vCtx0, ggml_view_2d(vCtx0, vModel.lLayers[pIl].vPleConv1d, 1, cHcDim,
                                                     vModel.lLayers[pIl].vPleConv1d->nb[1],
                                                     vK * vModel.lLayers[pIl].vPleConv1d->nb[0]));
    // this kernel keeps the file type, so cast it before it multiplies an f32 activation
    vWk = ggml_reshape_1d(vCtx0, vWk, cHcDim);
    if (vWk->type != GGML_TYPE_F32) {
      vWk = ggml_cast(vCtx0, vWk, GGML_TYPE_F32);
    }

    ggml_tensor *vTerm = ggml_mul(vCtx0, vShifted, vWk);
    vConvOut = vConvOut ? ggml_add(vCtx0, vConvOut, vTerm) : vTerm;
  }

  vConvOut = ggml_silu(vCtx0, vConvOut);
  vConvOut = ggml_reshape_3d(vCtx0, ggml_cont(vCtx0, vConvOut), cNEmbd, cHc, cNTokens);
  mCb(vConvOut, "ple_conv_out", pIl);

  return ggml_add(vCtx0, pHidden, ggml_add(vCtx0, vGated, vConvOut));
}
