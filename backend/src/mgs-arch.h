#pragma once

#include "ggml.h" // ggml_op

#include <string>
#include <set>
#include <vector>

//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
  cLlmArchFlint,
  cLlmArchQwen35,
  cLlmArchQwen35moe,
  cLlmArchQwen4exp,
  cLlmArchDeepseek41,
  cLlmArchGlmDsa,
  cLlmArchGlm5Next,
  cLlmArchNemotronHMoe,
  cLlmArchMimo2,
  cLlmArchKimiK3,
  cLlmArchMinimaxM3,
  cLlmArchUnknown,
};

enum llm_kv {
  cLlmKvGeneralType,
  cLlmKvGeneralArchitecture,
  cLlmKvGeneralQuantizationVersion,
  cLlmKvGeneralAlignment,
  cLlmKvGeneralFileType,
  cLlmKvGeneralSamplingSequence,
  cLlmKvGeneralSamplingTopK,
  cLlmKvGeneralSamplingTopP,
  cLlmKvGeneralSamplingMinP,
  cLlmKvGeneralSamplingXtcProbability,
  cLlmKvGeneralSamplingXtcThreshold,
  cLlmKvGeneralSamplingTemp,
  cLlmKvGeneralSamplingPenaltyLastN,
  cLlmKvGeneralSamplingPenaltyRepeat,
  cLlmKvGeneralSamplingMirostat,
  cLlmKvGeneralSamplingMirostatTau,
  cLlmKvGeneralSamplingMirostatEta,
  cLlmKvGeneralName,
  cLlmKvGeneralAuthor,
  cLlmKvGeneralVersion,
  cLlmKvGeneralUrl,
  cLlmKvGeneralDescription,
  cLlmKvGeneralLicense,
  cLlmKvGeneralSourceUrl,
  cLlmKvGeneralSourceHfRepo,

  cLlmKvVocabSize,
  cLlmKvContextLength,
  cLlmKvEmbeddingLength,
  cLlmKvEmbeddingLengthOut,
  cLlmKvEmbeddingLengthPerLayer,
  cLlmKvBlockCount,
  cLlmKvLeadingDenseBlockCount,
  cLlmKvAttnResBlockSize,
  cLlmKvActivationSituBeta,
  cLlmKvActivationSituLinearBeta,
  cLlmKvFeedForwardLength,
  cLlmKvExpertFeedForwardLength,
  cLlmKvExpertSharedFeedForwardLength,
  cLlmKvSwigluClampExp,
  cLlmKvSwigluClampShexp,
  cLlmKvTensorDataLayout,
  cLlmKvExpertCount,
  cLlmKvExpertUsedCount,
  cLlmKvExpertSharedCount,
  cLlmKvExpertGroupCount,
  cLlmKvExpertGroupUsedCount,
  cLlmKvExpertWeightsScale,
  cLlmKvExpertWeightsNorm,
  cLlmKvExpertLatentLength,
  cLlmKvExpertGatingFunc,
  cLlmKvMoeLatentSize,
  cLlmKvNextnPredictLayers,
  cLlmKvNumDeepstackLayers,
  cLlmKvDeepstackMapping,
  cLlmKvPoolingType,
  cLlmKvLogitScale,
  cLlmKvResidualScale,
  cLlmKvEmbeddingScale,
  cLlmKvFullAttentionInterval,

  cLlmKvAttentionHeadCount,
  cLlmKvAttentionHeadCountKv,
  cLlmKvAttentionMaxAlibiBias,
  cLlmKvAttentionClampKqv,
  cLlmKvAttentionKeyLength,
  cLlmKvAttentionValueLength,
  cLlmKvAttentionLayernormEps,
  cLlmKvAttentionLayernormRmsEps,
  cLlmKvAttentionCausal,
  cLlmKvAttentionQLoraRank,
  cLlmKvAttentionKvLoraRank,
  cLlmKvAttentionSlidingWindow,
  cLlmKvAttentionSlidingWindowPattern,
  cLlmKvAttentionRopePattern,
  cLlmKvAttentionScale,
  cLlmKvAttentionValueScale,
  cLlmKvAttentionKeyLengthMla,
  cLlmKvAttentionValueLengthMla,
  cLlmKvAttentionKeyLengthSwa,
  cLlmKvAttentionValueLengthSwa,
  cLlmKvAttentionIndexerHeadCount,
  cLlmKvAttentionIndexerKeyLength,
  cLlmKvAttentionIndexerTopK,
  cLlmKvAttentionIndexerBlockSize,
  cLlmKvAttentionIndexerLocalBlocks,
  cLlmKvAttentionIndexerTypes,
  cLlmKvAttentionIndexerKpool,
  cLlmKvAttentionIndexerKpoolSelectTail,
  cLlmKvAttentionOutputGroupCount,
  cLlmKvAttentionOutputLoraRank,
  cLlmKvAttentionCompressRopeFreqBase,
  cLlmKvAttentionCompressRatios,
  cLlmKvAttentionRecurrentLayers,

  cLlmKvHyperConnectionCount,
  cLlmKvHyperConnectionSinkhornIterations,
  cLlmKvHyperConnectionEpsilon,
  cLlmKvHyperConnectionLowRank,

  cLlmKvPleLayers,
  cLlmKvPleNgramSize,
  cLlmKvPleHeadsPerNgram,
  cLlmKvPleConvKernel,
  cLlmKvPleLayerMultipliers,
  cLlmKvPleHeadOffsets,
  cLlmKvPleHeadVocabSizes,
  cLlmKvPleEosTokenId,
  cLlmKvPleImageTokenId,

  cLlmKvRopeDimensionCount,
  cLlmKvRopeDimensionCountSwa,
  cLlmKvRopeDimensionSections,
  cLlmKvRopeFreqBase,
  cLlmKvRopeFreqBaseSwa,
  cLlmKvRopeScalingType,
  cLlmKvRopeScalingFactor,
  cLlmKvRopeScalingAttnFactor,
  cLlmKvRopeScalingOrigCtxLen,
  cLlmKvRopeScalingFinetuned,
  cLlmKvRopeScalingYarnLogMul,
  cLlmKvRopeScalingYarnExtFactor,
  cLlmKvRopeScalingYarnAttnFactor,
  cLlmKvRopeScalingYarnBetaFast,
  cLlmKvRopeScalingYarnBetaSlow,

  cLlmKvSplitNo,
  cLlmKvSplitCount,
  cLlmKvSplitTensorsCount,

  cLlmKvSsmInnerSize,
  cLlmKvSsmConvKernel,
  cLlmKvSsmStateSize,
  cLlmKvSsmTimeStepRank,
  cLlmKvSsmGroupCount,
  cLlmKvSsmDtBCRms,

  cLlmKvKdaHeadDim,
  cLlmKvKdaGateLowerBound,

  cLlmKvTokenizerModel,
  cLlmKvTokenizerPre,
  cLlmKvTokenizerList,
  cLlmKvTokenizerTokenType,
  cLlmKvTokenizerTokenTypeCount,
  cLlmKvTokenizerScores,
  cLlmKvTokenizerMerges,
  cLlmKvTokenizerBosId,
  cLlmKvTokenizerEosId,
  cLlmKvTokenizerEotId,
  cLlmKvTokenizerEomId,
  cLlmKvTokenizerUnkId,
  cLlmKvTokenizerSepId,
  cLlmKvTokenizerPadId,
  cLlmKvTokenizerMaskId,
  cLlmKvTokenizerAddBos,
  cLlmKvTokenizerAddEos,
  cLlmKvTokenizerAddSep,
  cLlmKvTokenizerAddPrefix,
  cLlmKvTokenizerRemoveExtraWs,
  cLlmKvTokenizerPrecompiledCharsmap,
  cLlmKvTokenizerHfJson,
  cLlmKvTokenizerRwkv,
  cLlmKvTokenizerChatTemplate,
  cLlmKvTokenizerNormalizerLowercase,
  cLlmKvTokenizerNormalizerStripAccents,
  cLlmKvTokenizerFimPreId,
  cLlmKvTokenizerFimSufId,
  cLlmKvTokenizerFimMidId,
  cLlmKvTokenizerFimPadId,
  cLlmKvTokenizerFimRepId,
  cLlmKvTokenizerFimSepId,
  cLlmKvTokenizerSuppressTokens,

  cLlmKvAdapterType,
  cLlmKvAdapterLoraAlpha,
  cLlmKvAdapterLoraTaskName,
  cLlmKvAdapterLoraPromptPrefix,
  cLlmKvAdapterAloraInvocationTokens,

  cLlmKvClassifierOutputLabels,

  cLlmKvShortconvLCache,
};

enum llm_tensor {
  cLlmTensorTokenEmbd,
  cLlmTensorTokenTypes,
  cLlmTensorPosEmbd,
  cLlmTensorOutput,
  cLlmTensorOutputNorm,
  cLlmTensorAttnQ,
  cLlmTensorAttnK,
  cLlmTensorAttnV,
  cLlmTensorAttnQkv,
  cLlmTensorAttnOut,
  cLlmTensorAttnNorm,
  cLlmTensorAttnPostNorm,
  cLlmTensorAttnSinks,
  cLlmTensorAttnGate,
  cLlmTensorFfnGateInp,
  cLlmTensorFfnGateInpShexp,
  cLlmTensorFfnNorm,
  cLlmTensorFfnGate,
  cLlmTensorFfnDown,
  cLlmTensorFfnUp,
  cLlmTensorFfnDownExps, // merged experts
  cLlmTensorFfnGateExps,
  cLlmTensorFfnUpExps,
  cLlmTensorFfnGateUpExps,
  cLlmTensorFfnDownShexp,
  cLlmTensorFfnGateShexp,
  cLlmTensorFfnUpShexp,
  cLlmTensorFfnExpProbsB,
  cLlmTensorFfnLatentDown,
  cLlmTensorFfnLatentUp,
  cLlmTensorAttnQNorm,
  cLlmTensorAttnKNorm,
  cLlmTensorLayerOutNorm,
  cLlmTensorPerLayerTokenEmbd, // qwen4exp
  cLlmTensorSsmIn,
  cLlmTensorSsmConv1d,
  cLlmTensorSsmDt,
  cLlmTensorSsmA,
  cLlmTensorSsmANoscan, // qwen3next special case with MUL instead of SSM_SCAN
  cLlmTensorSsmD,
  cLlmTensorSsmNorm,
  cLlmTensorSsmOut,
  cLlmTensorSsmAlpha,     // qwen3.5
  // Kimi Linear KDA (using SSM_ prefix for consistency)
  cLlmTensorSsmConv1dQ, // kimi: Q conv1d weight
  cLlmTensorSsmConv1dK, // kimi: K conv1d weight
  cLlmTensorSsmConv1dV, // kimi: V conv1d weight
  cLlmTensorSsmFA,      // kimi: forget gate projection A
  cLlmTensorSsmFB,      // kimi: forget gate projection B
  cLlmTensorSsmBeta,    // kimi: beta mixing coefficient and qwen3.5
  cLlmTensorSsmGA,      // kimi: output gate projection A
  cLlmTensorSsmGB,      // kimi: output gate projection B
  cLlmTensorSsmG,           // kimi-k3: full-rank KDA gate
  cLlmTensorAttnResScore,   // kimi-k3: fused res_norm*res_proj (pre-attention)
  cLlmTensorFfnResScore,    // kimi-k3: fused res_norm*res_proj (pre-ffn)
  cLlmTensorOutputResScore, // kimi-k3: fused res_norm*res_proj (final)
  cLlmTensorFfnRoutedDown,  // kimi-k3: latent MoE down
  cLlmTensorFfnRoutedUp,    // kimi-k3: latent MoE up
  cLlmTensorFfnRoutedNorm,  // kimi-k3: latent MoE norm
  cLlmTensorAttnQA,
  cLlmTensorAttnQB,
  cLlmTensorAttnKvAMqa,
  cLlmTensorAttnKv,
  cLlmTensorAttnKvNorm,
  cLlmTensorAttnOutA,
  cLlmTensorAttnOutB,
  cLlmTensorAttnKB,
  cLlmTensorAttnVB,
  cLlmTensorAttnQANorm,
  cLlmTensorAttnKvANorm,
  cLlmTensorHcHeadNorm, // qwen4exp
  cLlmTensorHcHeadDown, // qwen4exp
  cLlmTensorHcHeadUp,   // qwen4exp
  cLlmTensorHcAttnFn,
  cLlmTensorHcAttnBase,
  cLlmTensorHcAttnScale,
  cLlmTensorHcFfnFn,
  cLlmTensorHcFfnBase,
  cLlmTensorHcFfnScale,
  cLlmTensorHcAttnNorm,   // qwen4exp
  cLlmTensorHcAttnDown,   // qwen4exp
  cLlmTensorHcAttnUp,     // qwen4exp
  cLlmTensorHcAttnInject, // qwen4exp
  cLlmTensorHcFfnNorm,    // qwen4exp
  cLlmTensorHcFfnDown,    // qwen4exp
  cLlmTensorHcFfnUp,      // qwen4exp
  cLlmTensorHcFfnInject,  // qwen4exp
  cLlmTensorPleKey,       // qwen4exp
  cLlmTensorPleValue,     // qwen4exp
  cLlmTensorPleNormKey,   // qwen4exp
  cLlmTensorPleNormQuery, // qwen4exp
  cLlmTensorPleNormConv,  // qwen4exp
  cLlmTensorPleConv1d,    // qwen4exp
  cLlmTensorAttnCompressorWkv,
  cLlmTensorAttnCompressorWgate,
  cLlmTensorAttnCompressorNorm,
  cLlmTensorIndexerKNorm,
  cLlmTensorIndexerProj,
  cLlmTensorIndexerAttnK,
  cLlmTensorIndexerAttnQB,
  cLlmTensorIndexerKpoolGate, // glm5-next: k-pool gate scores
  cLlmTensorIndexerKpoolApe,  // glm5-next: k-pool position bias
  cLlmTensorIndexerQProj,
  cLlmTensorIndexerKProj,
  cLlmTensorIndexerQNorm,
  cLlmTensorFfnExpProbsBVl,
  cLlmTensorEngramEmbd,
  cLlmTensorEngramWkv,
  cLlmTensorEngramQ,
  cLlmTensorEngramK,
  cLlmTensorEngramKv,
  cLlmTensorEngramQNorm,
  cLlmTensorEngramKNorm,
  cLlmTensorNextnEhProj,
  cLlmTensorNextnEmbedTokens,
  cLlmTensorNextnEnorm,
  cLlmTensorNextnHnorm,
  cLlmTensorNextnSharedHeadHead,
  cLlmTensorNextnSharedHeadNorm,
  cLlmTensorNextnHcHeadNorm, // qwen4exp: the MTP block's own hyper-connection mixer
  cLlmTensorNextnHcHeadDown,
  cLlmTensorNextnHcHeadUp,
};

enum llm_tensor_layer {
  cLlmTensorLayerInput,
  cLlmTensorLayerRepeating,
  cLlmTensorLayerOutput,
};

struct LLM_KV {
  LLM_KV(llm_arch pArch, const char *pSuffix = nullptr);

  llm_arch vArch;
  const char *vSuffix;

  std::string operator()(llm_kv pKv) const;
};

// helper to handle gguf constants
// usage:
//
//   const auto cTn = LLM_TN(cLlmArchQwen35moe);
//
//   std::string vName = cTn(cLlmTensorOutput);                  -> "output"
//   std::string vName = cTn(cLlmTensorTokenEmbd, "bias");       -> "token_embd.bias"
//   std::string vName = cTn(cLlmTensorAttnNorm, "weight", 3);   -> "blk.3.attn_norm.weight"
//
struct LLM_TN_IMPL {
  const llm_arch cArch;
  const llm_tensor cTensor;
  const char *const cSuffix;
  const int cBid;
  const int cXid;

  LLM_TN_IMPL(llm_arch pArch, llm_tensor pTensor, const char *pSuffix, int pBid, int pXid);

  std::string mStr() const;

  operator std::string() const { return mStr(); }

  friend bool operator==(const std::string &pStr, const LLM_TN_IMPL &pTn) { return pStr == pTn.mStr(); }

  friend bool operator!=(const std::string &pStr, const LLM_TN_IMPL &pTn) { return pStr != pTn.mStr(); }
};

struct LLM_TN {
  LLM_TN(llm_arch pArch) : vArch(pArch) {}

  llm_arch vArch;

  LLM_TN_IMPL operator()(llm_tensor pTensor, const char *pSuffix, int pBid = -1, int pXid = -1) const {
    return LLM_TN_IMPL(vArch, pTensor, pSuffix, pBid, pXid);
  }

  LLM_TN_IMPL operator()(llm_tensor pTensor, int pBid = -1, int pXid = -1) const {
    return LLM_TN_IMPL(vArch, pTensor, nullptr, pBid, pXid);
  }
};

struct llm_tensor_info {
  llm_tensor_layer vLayer;
  ggml_op vOp;
};

std::vector<llm_arch> fLlmArchAll();

const char *fLlmArchName(llm_arch pArch);

llm_arch fLlmArchFromString(const std::string &pName);

// Name of the key pKv for the architecture called pArchName, also one that mgs does not know.
std::string fLlmKvNameForArch(llm_kv pKv, const std::string &pArchName);

const llm_tensor_info &fLlmTensorInfoFor(llm_tensor pTensor);

bool fLlmArchIsHybrid(const llm_arch &pArch);
bool fLlmArchSupportsSmTensor(const llm_arch &pArch);
bool fLlmArchSupportsRsRollback(const llm_arch &pArch);
