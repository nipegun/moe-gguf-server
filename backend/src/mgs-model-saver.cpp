#include "mgs-model-saver.h"

#include "ggml.h"
#include "gguf.h"

#include "mgs-arch.h"
#include "mgs.h"
#include "mgs-hparams.h"
#include "mgs-model.h"
#include "mgs-vocab.h"

#include <cstdint>
#include <string>

bool mgs_model_saver_supports_arch(llm_arch pArch) {
  switch (pArch) {
  case cLlmArchPlamo3:
  case cLlmArchGemma3:
  case cLlmArchGemma3n:
  case cLlmArchCohere2:
  case cLlmArchCohere2moe:
  case cLlmArchOlmo2:
  case cLlmArchBitnet:
  case cLlmArchT5:
  case cLlmArchExaoneMoe:
  case cLlmArchAfmoe:
  case cLlmArchApertus:
  case cLlmArchMimo2:
  case cLlmArchStep35:
  case cLlmArchMellum:
  case cLlmArchLaguna:
    return false;
  default:
    return true;
  }
}

mgs_model_saver::mgs_model_saver(const struct mgs_model *pModel)
    : vGgufCtx(gguf_init_empty()), cGgufCtxOwned(true), vModel(pModel), cLlmKv(pModel->vArch) {
  GGML_ASSERT(mgs_model_saver_supports_arch(pModel->vArch));
}

mgs_model_saver::mgs_model_saver(enum llm_arch pArch, struct gguf_context *pGgufCtx)
    : vGgufCtx(pGgufCtx == nullptr ? gguf_init_empty() : pGgufCtx), cGgufCtxOwned(pGgufCtx == nullptr), vModel(nullptr),
      cLlmKv(pArch) {}

mgs_model_saver::~mgs_model_saver() {
  if (cGgufCtxOwned) {
    gguf_free(vGgufCtx);
  }
}

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const uint32_t pValue) {
  gguf_set_val_u32(vGgufCtx, cLlmKv(pKey).c_str(), pValue);
}

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const int32_t pValue) {
  gguf_set_val_i32(vGgufCtx, cLlmKv(pKey).c_str(), pValue);
}

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const float pValue) {
  gguf_set_val_f32(vGgufCtx, cLlmKv(pKey).c_str(), pValue);
}

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const bool pValue) {
  gguf_set_val_bool(vGgufCtx, cLlmKv(pKey).c_str(), pValue);
}

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const char *pValue) {
  gguf_set_val_str(vGgufCtx, cLlmKv(pKey).c_str(), pValue);
}

[[noreturn]]
void mgs_model_saver::mAddKv(const enum llm_kv pKey, const char pValue) {
  GGML_UNUSED(pKey);
  GGML_UNUSED(pValue);
  GGML_ABORT("fatal error"); // this should never be called, only needed to make the template below compile
}

template <typename Container>
void mgs_model_saver::mAddKv(const enum llm_kv pKey, const Container &pValue, const bool pPerLayer) {
  GGML_ASSERT(vModel != nullptr || !pPerLayer);
  const size_t cNValues = pPerLayer ? size_t(vModel->vHparams.mNLayer()) : pValue.size();
  GGML_ASSERT(cNValues <= pValue.size());

  if (cNValues == 0) {
    return;
  }

  if (pPerLayer) {
    bool vAllValuesTheSame = true;
    for (size_t vI = 1; vI < cNValues; ++vI) {
      if (pValue[vI] != pValue[0]) {
        vAllValuesTheSame = false;
        break;
      }
    }
    if (vAllValuesTheSame) {
      mAddKv(pKey, pValue[0]);
      return;
    }
  }

  if (std::is_same<typename Container::value_type, uint8_t>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_UINT8, pValue.data(), cNValues);
  } else if (std::is_same<typename Container::value_type, int8_t>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_INT8, pValue.data(), cNValues);
  } else if (std::is_same<typename Container::value_type, uint32_t>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_UINT32, pValue.data(), cNValues);
  } else if (std::is_same<typename Container::value_type, bool>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_BOOL, pValue.data(), cNValues);
  } else if (std::is_same<typename Container::value_type, int32_t>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_INT32, pValue.data(), cNValues);
  } else if (std::is_same<typename Container::value_type, float>::value) {
    gguf_set_arr_data(vGgufCtx, cLlmKv(pKey).c_str(), GGUF_TYPE_FLOAT32, pValue.data(), cNValues);
  } else if (std::is_same<Container, std::string>::value) {
    gguf_set_val_str(vGgufCtx, cLlmKv(pKey).c_str(), reinterpret_cast<const char *>(pValue.data()));
  } else {
    GGML_ABORT("fatal error");
  }
}
// instantiate for external usage:
template void mgs_model_saver::mAddKv<std::vector<uint32_t>>(const enum llm_kv, const std::vector<uint32_t> &,
                                                             const bool);

void mgs_model_saver::mAddKv(const enum llm_kv pKey, const std::vector<std::string> &pValue) {
  std::vector<const char *> lTmp(pValue.size());
  for (size_t vI = 0; vI < pValue.size(); ++vI) {
    lTmp[vI] = pValue[vI].c_str();
  }
  gguf_set_arr_str(vGgufCtx, cLlmKv(pKey).c_str(), lTmp.data(), lTmp.size());
}

void mgs_model_saver::mAddTensor(const struct ggml_tensor *pTensor) {
  if (!pTensor) {
    return;
  }
  if (gguf_find_tensor(vGgufCtx, pTensor->name) >= 0) {
    const std::string cTensorName = pTensor->name;
    GGML_ASSERT(cTensorName == "rope_freqs.weight" || cTensorName == "rope_factors_long.weight" ||
                cTensorName == "rope_factors_short.weight"); // FIXME
    return;
  }
  gguf_add_tensor(vGgufCtx, pTensor);
}

void mgs_model_saver::mAddKvFromModel() {
  const mgs_hparams &vHparams = vModel->vHparams;
  const mgs_vocab &vVocab = vModel->vVocab;

  const int32_t cNVocab = vVocab.mNTokens();
  std::vector<std::string> lTokens(cNVocab);
  std::vector<float> lScores(cNVocab);
  std::vector<int32_t> lTokenTypes(cNVocab);

  if (vVocab.mGetType() != MGS_VOCAB_TYPE_NONE) {
    for (int32_t vId = 0; vId < cNVocab; ++vId) {
      const mgs_vocab::vTokenData &vTokenData = vVocab.mGetTokenData(vId);

      lTokens[vId] = vTokenData.vText;
      lScores[vId] = vTokenData.vScore;

      // FIXME should this be treated as flags?
      switch (vTokenData.vAttr) {
      case MGS_TOKEN_ATTR_UNKNOWN:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_UNKNOWN;
        break;
      case MGS_TOKEN_ATTR_UNUSED:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_UNUSED;
        break;
      case MGS_TOKEN_ATTR_NORMAL:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_NORMAL;
        break;
      case MGS_TOKEN_ATTR_CONTROL:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_CONTROL;
        break;
      case MGS_TOKEN_ATTR_USER_DEFINED:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_USER_DEFINED;
        break;
      case MGS_TOKEN_ATTR_BYTE:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_BYTE;
        break;
      // case MGS_TOKEN_ATTR_NORMALIZED:   ???
      // case MGS_TOKEN_ATTR_LSTRIP:       ???
      // case MGS_TOKEN_ATTR_RSTRIP:       ???
      case MGS_TOKEN_ATTR_UNDEFINED:
      default:
        lTokenTypes[vId] = MGS_TOKEN_TYPE_UNDEFINED;
        break;
      }
    }
  }

  // mAddKv(cLlmKvGeneralType,                    ???);
  mAddKv(cLlmKvGeneralArchitecture, vModel->mArchName());
  // mAddKv(cLlmKvGeneralQuantizationVersion,     ???);
  // mAddKv(cLlmKvGeneralAlignment,               ???);
  // mAddKv(cLlmKvGeneralFileType,                ???);
  // mAddKv(cLlmKvGeneralSamplingSequence,        ???);
  // mAddKv(cLlmKvGeneralSamplingTopK,            ???);
  // mAddKv(cLlmKvGeneralSamplingTopP,            ???);
  // mAddKv(cLlmKvGeneralSamplingMinP,            ???);
  // mAddKv(cLlmKvGeneralSamplingXtcProbability,  ???);
  // mAddKv(cLlmKvGeneralSamplingXtcThreshold,    ???);
  // mAddKv(cLlmKvGeneralSamplingTemp,            ???);
  // mAddKv(cLlmKvGeneralSamplingPenaltyLastN,    ???);
  // mAddKv(cLlmKvGeneralSamplingPenaltyRepeat,   ???);
  // mAddKv(cLlmKvGeneralSamplingMirostat,        ???);
  // mAddKv(cLlmKvGeneralSamplingMirostatTau,     ???);
  // mAddKv(cLlmKvGeneralSamplingMirostatEta,     ???);
  mAddKv(cLlmKvGeneralName, vModel->vName);
  // mAddKv(cLlmKvGeneralAuthor,                  ???);
  // mAddKv(cLlmKvGeneralVersion,                 ???);
  // mAddKv(cLlmKvGeneralUrl,                     ???);
  // mAddKv(cLlmKvGeneralDescription,             ???);
  // mAddKv(cLlmKvGeneralLicense,                 ???);
  // mAddKv(cLlmKvGeneralSourceUrl,               ???);
  // mAddKv(cLlmKvGeneralSourceHfRepo,            ???);

  mAddKv(cLlmKvVocabSize, vVocab.mNTokens());
  mAddKv(cLlmKvContextLength, vHparams.vNCtxTrain);
  mAddKv(cLlmKvEmbeddingLength, vHparams.vNEmbd);
  if (vHparams.vNEmbdOutImpl > 0) {
    mAddKv(cLlmKvEmbeddingLengthOut, vHparams.vNEmbdOutImpl);
  }
  mAddKv(cLlmKvBlockCount, vHparams.vNLayerAll);
  mAddKv(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead);
  mAddKv(cLlmKvFeedForwardLength, vHparams.aNFfArr, true);
  mAddKv(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  mAddKv(cLlmKvExpertLatentLength, vHparams.vNExpertLatent);
  mAddKv(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfShexp);
  mAddKv(cLlmKvExpertSharedFeedForwardLength, vHparams.vNFfChexp);
  mAddKv(cLlmKvSwigluClampExp, vHparams.aSwigluClampExp);
  mAddKv(cLlmKvSwigluClampShexp, vHparams.aSwigluClampShexp);
  mAddKv(cLlmKvUseParallelResidual, vHparams.vUseParRes);
  // mAddKv(cLlmKvTensorDataLayout,               ???);
  mAddKv(cLlmKvExpertCount, vHparams.vNExpert);
  mAddKv(cLlmKvExpertUsedCount, vHparams.vNExpertUsed);
  mAddKv(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  mAddKv(cLlmKvExpertGroupCount, vHparams.vNExpertGroups);
  mAddKv(cLlmKvExpertGroupUsedCount, vHparams.vNGroupUsed);
  mAddKv(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale);
  mAddKv(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm);
  mAddKv(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  mAddKv(cLlmKvExpertGroupScale, vHparams.vExpertGroupScale);
  mAddKv(cLlmKvExpertsPerGroup, vHparams.vNGroupExperts);
  mAddKv(cLlmKvMoeEveryNLayers, vHparams.vMoeEveryNLayers);
  mAddKv(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn);
  mAddKv(cLlmKvNumDeepstackLayers, vHparams.vNDeepstackLayers);
  mAddKv(cLlmKvDeepstackMapping, vHparams.aDeepstackMappingArr);
  mAddKv(cLlmKvPoolingType, uint32_t(vHparams.vPoolingType));
  mAddKv(cLlmKvLogitScale, vHparams.vFLogitScale);
  mAddKv(cLlmKvDecoderStartTokenId, vHparams.vDecStartTokenId);
  mAddKv(cLlmKvDecoderBlockCount, vHparams.vDecNLayer);
  mAddKv(cLlmKvAttnLogitSoftcapping, vHparams.vFAttnLogitSoftcapping);
  mAddKv(cLlmKvRouterLogitSoftcapping, vHparams.vFRouterLogitSoftcapping);
  mAddKv(cLlmKvFinalLogitSoftcapping, vHparams.vFFinalLogitSoftcapping);
  mAddKv(cLlmKvSwinNorm, vHparams.vSwinNorm);
  mAddKv(cLlmKvRescaleEveryNLayers, vHparams.vRescaleEveryNLayers);
  mAddKv(cLlmKvTimeMixExtraDim, vHparams.vTimeMixExtraDim);
  mAddKv(cLlmKvTimeDecayExtraDim, vHparams.vTimeDecayExtraDim);
  mAddKv(cLlmKvResidualScale, vHparams.vFResidualScale);
  mAddKv(cLlmKvEmbeddingScale, vHparams.vFEmbeddingScale);
  mAddKv(cLlmKvTokenShiftCount, vHparams.vTokenShiftCount);
  mAddKv(cLlmKvInterleaveMoeLayerStep, vHparams.vNMoeLayerStep);
  // mAddKv(cLlmKvFullAttentionInterval,          ???); // saved as cLlmKvAttentionRecurrentLayers instead

  mAddKv(cLlmKvAttentionHeadCount, vHparams.aNHeadArr, true);
  mAddKv(cLlmKvAttentionHeadCountKv, vHparams.aNHeadKvArr, true);
  mAddKv(cLlmKvAttentionMaxAlibiBias, vHparams.vFMaxAlibiBias);
  mAddKv(cLlmKvAttentionClampKqv, vHparams.vFClampKqv);
  mAddKv(cLlmKvAttentionKeyLength, vHparams.vNEmbdHeadKFull);
  mAddKv(cLlmKvAttentionValueLength, vHparams.vNEmbdHeadVFull);
  mAddKv(cLlmKvAttentionLayernormEps, vHparams.vFNormEps);
  mAddKv(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  mAddKv(cLlmKvAttentionGroupnormEps, vHparams.vFNormGroupEps);
  mAddKv(cLlmKvAttentionGroupnormGroups, vHparams.vNNormGroups);
  mAddKv(cLlmKvAttentionCausal, vHparams.vCausalAttn);
  mAddKv(cLlmKvAttentionQLoraRank, vHparams.vNLoraQ);
  mAddKv(cLlmKvAttentionKvLoraRank, vHparams.vNLoraKv);
  mAddKv(cLlmKvAttentionDecayLoraRank, vHparams.vNLoraDecay);
  mAddKv(cLlmKvAttentionIclrLoraRank, vHparams.vNLoraIclr);
  mAddKv(cLlmKvAttentionValueResidualMixLoraRank, vHparams.vNLoraValueResMix);
  mAddKv(cLlmKvAttentionGateLoraRank, vHparams.vNLoraGate);
  mAddKv(cLlmKvAttentionRelativeBucketsCount, vHparams.vNRelAttnBkts);
  mAddKv(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);
  // mAddKv(cLlmKvAttentionSlidingWindowPattern,  ???);
  mAddKv(cLlmKvAttentionScale, vHparams.vFAttentionScale);
  mAddKv(cLlmKvAttentionOutputScale, vHparams.vFAttnOutScale);
  mAddKv(cLlmKvAttentionValueScale, vHparams.vFAttnValueScale);
  mAddKv(cLlmKvAttentionTemperatureLength, vHparams.vAttnTempLength);
  mAddKv(cLlmKvAttentionTemperatureScale, vHparams.vFAttnTempScale);
  mAddKv(cLlmKvAttentionKeyLengthMla, vHparams.vNEmbdHeadKMlaImpl);
  mAddKv(cLlmKvAttentionValueLengthMla, vHparams.vNEmbdHeadVMlaImpl);
  mAddKv(cLlmKvAttentionKeyLengthSwa, vHparams.vNEmbdHeadKSwa);
  mAddKv(cLlmKvAttentionValueLengthSwa, vHparams.vNEmbdHeadVSwa);
  mAddKv(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  mAddKv(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  mAddKv(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);
  mAddKv(cLlmKvAttentionIndexerBlockSize, vHparams.vIndexerBlockSize);
  mAddKv(cLlmKvAttentionIndexerKpool, vHparams.vIndexerKpool);
  mAddKv(cLlmKvAttentionIndexerKpoolSelectTail, vHparams.vIndexerKpoolSelectTail);
  mAddKv(cLlmKvAttentionIndexerLocalBlocks, vHparams.vIndexerLocalBlocks);
  mAddKv(cLlmKvAttentionIndexerTypes, vHparams.aIsIndexerFullImpl, true);
  mAddKv(cLlmKvAttentionRecurrentLayers, vHparams.aIsRecrImpl, true);

  const float cRopeScalingFactor = vHparams.vRopeFreqScaleTrain == 1.0f ? 0.0f : 1.0f / vHparams.vRopeFreqScaleTrain;

  mAddKv(cLlmKvRopeDimensionCount, vHparams.vNRotFull);
  mAddKv(cLlmKvRopeDimensionCountSwa, vHparams.vNRotSwa);
  mAddKv(cLlmKvRopeDimensionSections, vHparams.aRopeSections);
  mAddKv(cLlmKvRopeFreqBase, vHparams.vRopeFreqBaseTrain);
  mAddKv(cLlmKvRopeFreqBaseSwa, vHparams.vRopeFreqBaseTrainSwa);
  mAddKv(cLlmKvRopeScalingType, mgs_rope_scaling_type_name(vHparams.vRopeScalingTypeTrain));
  mAddKv(cLlmKvRopeScalingFactor, cRopeScalingFactor);
  mAddKv(cLlmKvRopeScalingAttnFactor, vHparams.vRopeAttnFactor);
  mAddKv(cLlmKvRopeScalingOrigCtxLen, vHparams.vNCtxOrigYarn);
  mAddKv(cLlmKvRopeScalingFinetuned, vHparams.vRopeFinetuned);
  mAddKv(cLlmKvRopeScalingYarnLogMul, vHparams.vRopeYarnLogMul);
  mAddKv(cLlmKvRopeScalingYarnExtFactor, vHparams.vYarnExtFactor);
  mAddKv(cLlmKvRopeScalingYarnAttnFactor, vHparams.vYarnAttnFactor);
  mAddKv(cLlmKvRopeScalingYarnBetaFast, vHparams.vYarnBetaFast);
  mAddKv(cLlmKvRopeScalingYarnBetaSlow, vHparams.vYarnBetaSlow);

  // TODO: implement split file support
  // mAddKv(cLlmKvSplitNo,                        ???);
  // mAddKv(cLlmKvSplitCount,                     ???);
  // mAddKv(cLlmKvSplitTensorsCount,              ???);

  mAddKv(cLlmKvSsmInnerSize, vHparams.vSsmDInner);
  mAddKv(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  mAddKv(cLlmKvSsmStateSize, vHparams.vSsmDState);
  mAddKv(cLlmKvSsmTimeStepRank, vHparams.vSsmDtRank);
  mAddKv(cLlmKvSsmGroupCount, vHparams.vSsmNGroup);
  mAddKv(cLlmKvSsmDtBCRms, vHparams.vSsmDtBCRms);

  mAddKv(cLlmKvKdaHeadDim, vHparams.vNEmbdHeadKda);
  mAddKv(cLlmKvKdaGateLowerBound, vHparams.vKdaGateLowerBound);

  mAddKv(cLlmKvWkvHeadSize, vHparams.vWkvHeadSize);

  mAddKv(cLlmKvTokenizerModel, vVocab.mGetTokenizerModel());
  mAddKv(cLlmKvTokenizerPre, vVocab.mGetTokenizerPre());
  mAddKv(cLlmKvTokenizerList, lTokens);
  mAddKv(cLlmKvTokenizerTokenType, lTokenTypes);
  mAddKv(cLlmKvTokenizerTokenTypeCount, vVocab.mNTokenTypes());
  mAddKv(cLlmKvTokenizerScores, lScores);
  mAddKv(cLlmKvTokenizerMerges, vVocab.mGetBpeMerges());
  // FIXME mgs_token is type i32 but when reading in a GGUF file u32 is expected, not an issue for writing though
  mAddKv(cLlmKvTokenizerBosId, uint32_t(vVocab.mTokenBos()));
  mAddKv(cLlmKvTokenizerEosId, uint32_t(vVocab.mTokenEos()));
  mAddKv(cLlmKvTokenizerEotId, uint32_t(vVocab.mTokenEot()));
  mAddKv(cLlmKvTokenizerEomId, uint32_t(vVocab.mTokenEom()));
  mAddKv(cLlmKvTokenizerUnkId, uint32_t(vVocab.mTokenUnk()));
  mAddKv(cLlmKvTokenizerSepId, uint32_t(vVocab.mTokenSep()));
  mAddKv(cLlmKvTokenizerPadId, uint32_t(vVocab.mTokenPad()));
  // mAddKv(cLlmKvTokenizerMaskId,                ???);
  mAddKv(cLlmKvTokenizerAddBos, vVocab.mGetAddBos());
  mAddKv(cLlmKvTokenizerAddEos, vVocab.mGetAddEos());
  mAddKv(cLlmKvTokenizerAddSep, vVocab.mGetAddSep());
  mAddKv(cLlmKvTokenizerAddPrefix, vVocab.mGetAddSpacePrefix());
  mAddKv(cLlmKvTokenizerRemoveExtraWs, vVocab.mGetRemoveExtraWhitespaces());
  mAddKv(cLlmKvTokenizerPrecompiledCharsmap, vVocab.mGetPrecompiledCharsmap());
  // mAddKv(cLlmKvTokenizerHfJson,                ???);
  // mAddKv(cLlmKvTokenizerRwkv,                  ???);
  mAddKv(cLlmKvTokenizerFimPreId, uint32_t(vVocab.mTokenFimPre()));
  mAddKv(cLlmKvTokenizerFimSufId, uint32_t(vVocab.mTokenFimSuf()));
  mAddKv(cLlmKvTokenizerFimMidId, uint32_t(vVocab.mTokenFimMid()));
  mAddKv(cLlmKvTokenizerFimPadId, uint32_t(vVocab.mTokenFimPad()));
  mAddKv(cLlmKvTokenizerFimRepId, uint32_t(vVocab.mTokenFimRep()));
  mAddKv(cLlmKvTokenizerFimSepId, uint32_t(vVocab.mTokenFimSep()));

  // TODO: implement LoRA support
  // mAddKv(cLlmKvAdapterType,                    ???);
  // mAddKv(cLlmKvAdapterLoraAlpha,               ???);
  // mAddKv(cLlmKvAdapterLoraTaskName,            ???);
  // mAddKv(cLlmKvAdapterLoraPromptPrefix,        ???);
  // mAddKv(cLlmKvAdapterAloraInvocationTokens,   ???);

  mAddKv(cLlmKvPosnetEmbeddingLength, vHparams.vPosnet.vNEmbd);
  mAddKv(cLlmKvPosnetBlockCount, vHparams.vPosnet.vNLayer);

  mAddKv(cLlmKvConvnextEmbeddingLength, vHparams.vConvnext.vNEmbd);
  mAddKv(cLlmKvConvnextBlockCount, vHparams.vConvnext.vNLayer);

  mAddKv(cLlmKvClassifierOutputLabels, vModel->lClassifierLabels);

  mAddKv(cLlmKvShortconvLCache, vHparams.vNShortconvLCache);

  mAddKv(cLlmKvXieluAlphaN, vHparams.aXieluAlphaN);
  mAddKv(cLlmKvXieluAlphaP, vHparams.aXieluAlphaP);
  mAddKv(cLlmKvXieluBeta, vHparams.aXieluBeta);
  mAddKv(cLlmKvXieluEps, vHparams.aXieluEps);

  mAddKv(cLlmKvAttnResBlockSize, vHparams.vAttnResBlockSize);
  mAddKv(cLlmKvActivationSituBeta, vHparams.vSituBeta);
  mAddKv(cLlmKvActivationSituLinearBeta, vHparams.vSituLinearBeta);

  mAddKv(cLlmKvDense2FeatIn, vHparams.vDense2FeatIn);
  mAddKv(cLlmKvDense2FeatOut, vHparams.vDense2FeatOut);
  mAddKv(cLlmKvDense3FeatIn, vHparams.vDense3FeatIn);
  mAddKv(cLlmKvDense3FeatOut, vHparams.vDense3FeatOut);
}

void mgs_model_saver::mAddTensorsFromModel() {
  if (vModel->vOutput != nullptr && std::string(vModel->vOutput->name) != std::string(vModel->vTokEmbd->name)) {
    mAddTensor(vModel->vTokEmbd); // some models use the same tensor for tok_embd and output
  }
  mAddTensor(vModel->vTypeEmbd);
  mAddTensor(vModel->vPosEmbd);
  mAddTensor(vModel->vTokNorm);
  mAddTensor(vModel->vTokNormB);
  mAddTensor(vModel->vOutputNorm);
  mAddTensor(vModel->vOutputNormB);
  mAddTensor(vModel->vOutput);
  mAddTensor(vModel->vOutputB);
  mAddTensor(vModel->vOutputNormEnc);
  mAddTensor(vModel->vOutputS);
  mAddTensor(vModel->vOutputInS);
  mAddTensor(vModel->vOutputResScore);
  mAddTensor(vModel->vCls);
  mAddTensor(vModel->vClsB);
  mAddTensor(vModel->vClsOut);
  mAddTensor(vModel->vClsOutB);
  mAddTensor(vModel->vClsNorm);

  for (const struct mgs_layer &vLayer : vModel->lLayers) {
    for (size_t vI = 0; vI < sizeof(vLayer) / sizeof(struct ggml_tensor *); ++vI) {
      mAddTensor(reinterpret_cast<const struct ggml_tensor *const *>(&vLayer)[vI]);
    }
  }
}

void mgs_model_saver::mSave(const std::string &pPathModel) { gguf_write_to_file(vGgufCtx, pPathModel.c_str(), false); }

void mgs_model_saver::mSave(FILE *pFile) { gguf_write_to_file_ptr(vGgufCtx, pFile, false); }
