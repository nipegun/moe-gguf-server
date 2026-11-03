#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with mgs-impl.h
template <typename T> struct ring_buffer {
  ring_buffer(size_t pCap) : vCapacity(pCap), lData(pCap) {}

  T &mFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vFirst];
  }

  const T &mFront() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vFirst];
  }

  T &mBack() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vPos];
  }

  const T &mBack() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vPos];
  }

  void mPushBack(const T &pValue) {
    if (vSz == vCapacity) {
      // advance the start when buffer is full
      vFirst = (vFirst + 1) % vCapacity;
    } else {
      vSz++;
    }
    lData[vPos] = pValue;
    vPos = (vPos + 1) % vCapacity;
  }

  T mPopFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    T vValue = lData[vFirst];
    vFirst = (vFirst + 1) % vCapacity;
    vSz--;
    return vValue;
  }

  const T &mRat(size_t pI) const {
    if (pI >= vSz) {
      throw std::runtime_error("ring buffer: index out of bounds");
    }
    return lData[(vFirst + vSz - pI - 1) % vCapacity];
  }

  std::vector<T> mToVector() const {
    std::vector<T> lResult;
    lResult.reserve(vSz);
    for (size_t vI = 0; vI < vSz; vI++) {
      lResult.push_back(lData[(vFirst + vI) % vCapacity]);
    }
    return lResult;
  }

  void mClear() {
    // here only reset the status of the buffer
    vSz = 0;
    vFirst = 0;
    vPos = 0;
  }

  bool empty() const { return vSz == 0; }

  size_t size() const { return vSz; }

  size_t vCapacity = 0;
  size_t vSz = 0;
  size_t vFirst = 0;
  size_t vPos = 0;
  std::vector<T> lData;
};

struct common_sampler {
  common_params_sampling vParams;

  struct mgs_sampler *vGrmr;
  struct mgs_sampler *vRbudget;
  struct mgs_sampler *vChain;

  ring_buffer<mgs_token> vPrev;

  std::vector<mgs_token_data> lCur;

  mgs_token_data_array vCurP;

  void mReset() {
    vPrev.mClear();

    mgs_sampler_reset(vChain);
  }

  void mSetLogits(struct mgs_context *pCtx, int pIdx) {
    const float *vSampledProbs = mgs_get_sampled_probs_ith(pCtx, pIdx);
    const float *vSampledLogits = mgs_get_sampled_logits_ith(pCtx, pIdx);
    const mgs_token *vSampledIds = mgs_get_sampled_candidates_ith(pCtx, pIdx);

    const mgs_model *vModel = mgs_get_model(pCtx);
    const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);

    const int cNVocab = mgs_vocab_n_tokens(vVocab);

    if (vSampledProbs) {
      const uint32_t cSampledProbsCount = mgs_get_sampled_probs_count_ith(pCtx, pIdx);
      lCur.resize(cSampledProbsCount);
      for (uint32_t vI = 0; vI < cSampledProbsCount; ++vI) {
        lCur[vI] = mgs_token_data{vSampledIds[vI], vSampledLogits[vI], vSampledProbs[vI]};
      }
    } else if (vSampledLogits) {
      const uint32_t cSampledLogitsCount = mgs_get_sampled_logits_count_ith(pCtx, pIdx);
      lCur.resize(cSampledLogitsCount);
      for (uint32_t vI = 0; vI < cSampledLogitsCount; vI++) {
        lCur[vI] = mgs_token_data{vSampledIds[vI], vSampledLogits[vI], 0.0f};
      }
    } else {
      const auto *vLogits = mgs_get_logits_ith(pCtx, pIdx);
      GGML_ASSERT(vLogits != nullptr);
      lCur.resize(cNVocab);
      for (mgs_token vTokenId = 0; vTokenId < cNVocab; vTokenId++) {
        lCur[vTokenId] = mgs_token_data{vTokenId, vLogits[vTokenId], 0.0f};
      }
    }

    vCurP = {lCur.data(), lCur.size(), -1, false};
  }

  common_time_meas mTm() { return common_time_meas(vTTotalUs, vParams.vNoPerf); }

  mutable int64_t vTTotalUs = 0;
};

std::string common_params_sampling::mPrint() const {
  char aResult[1024];

  snprintf(aResult, sizeof(aResult),
           "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
           "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
           "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, "
           "top_n_sigma = %.3f, temp = %.3f\n"
           "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
           vPenaltyLastN, vPenaltyRepeat, vPenaltyFreq, vPenaltyPresent, vDryMultiplier, vDryBase, vDryAllowedLength,
           vDryPenaltyLastN, vTopK, vTopP, vMinP, vXtcProbability, vXtcThreshold, vTypP, vTopNSigma, vTemp, vMirostat,
           vMirostatEta, vMirostatTau, vAdaptiveTarget, vAdaptiveDecay);

  return std::string(aResult);
}

struct common_sampler *fCommonSamplerInit(const struct mgs_model *pModel, struct common_params_sampling &pParams,
                                          int32_t pNCtx) {
  if (!std::isfinite(pParams.vPenaltyRepeat) || pParams.vPenaltyRepeat <= 0.0f ||
      !std::isfinite(1.0f / pParams.vPenaltyRepeat)) {
    throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
  }
  if (!std::isfinite(pParams.vPenaltyFreq)) {
    throw std::invalid_argument("penalty_freq must be finite");
  }
  if (!std::isfinite(pParams.vPenaltyPresent)) {
    throw std::invalid_argument("penalty_present must be finite");
  }
  if (pParams.vPenaltyLastN == -1) {
    pParams.vPenaltyLastN = pNCtx > 0 ? pNCtx : mgs_model_n_ctx_train(pModel);
  }

  const mgs_vocab *vVocab = mgs_model_get_vocab(pModel);
  mgs_sampler_chain_params vLparams = mgs_sampler_chain_default_params();

  vLparams.no_perf = pParams.vNoPerf;

  mgs_sampler *vGrmr = nullptr;
  mgs_sampler *vRbudget = nullptr;
  mgs_sampler *vChain = mgs_sampler_chain_init(vLparams);

  std::vector<mgs_sampler *> lSamplers;

  const std::string &vGrammarStr = fCommonGrammarValue(pParams.vGrammar);
  if (vGrammarStr.compare(0, 11, "%llguidance") == 0) {
#ifdef MGS_USE_LLGUIDANCE
    vGrmr = mgs_sampler_init_llg(vVocab, "lark", vGrammarStr.c_str());
#else
    GGML_ABORT("llguidance (cmake -DMGS_LLGUIDANCE=ON) is not enabled");
#endif // MGS_USE_LLGUIDANCE
  } else {
    std::vector<std::string> lTriggerPatterns;
    std::vector<mgs_token> lTriggerTokens;
    for (const auto &vTrigger : pParams.lGrammarTriggers) {
      switch (vTrigger.vType) {
      case cCommonGrammarTriggerTypeWord: {
        const auto &vWord = vTrigger.vValue;
        lTriggerPatterns.push_back(fRegexEscape(vWord));
        break;
      }
      case cCommonGrammarTriggerTypePattern: {
        lTriggerPatterns.push_back(vTrigger.vValue);
        break;
      }
      case cCommonGrammarTriggerTypePatternFull: {
        const auto &vPattern = vTrigger.vValue;
        std::string vAnchored = "^$";
        if (!vPattern.empty()) {
          vAnchored = (vPattern.front() != '^' ? "^" : "") + vPattern + (vPattern.back() != '$' ? "$" : "");
        }
        lTriggerPatterns.push_back(vAnchored);
        break;
      }
      case cCommonGrammarTriggerTypeToken: {
        const auto cToken = vTrigger.vToken;
        lTriggerTokens.push_back(cToken);
        break;
      }
      default:
        GGML_ASSERT(false && "unknown trigger type");
      }
    }

    std::vector<const char *> lTriggerPatternsC;
    lTriggerPatternsC.reserve(lTriggerPatterns.size());
    for (const auto &vRegex : lTriggerPatterns) {
      lTriggerPatternsC.push_back(vRegex.c_str());
    }

    if (!vGrammarStr.empty()) {
      if (pParams.vGrammarLazy) {
        vGrmr = mgs_sampler_init_grammar_lazy_patterns(vVocab, vGrammarStr.c_str(), "root", lTriggerPatternsC.data(),
                                                       lTriggerPatternsC.size(), lTriggerTokens.data(),
                                                       lTriggerTokens.size());
      } else {
        vGrmr = mgs_sampler_init_grammar(vVocab, vGrammarStr.c_str(), "root");
      }
    }
  }
  if (!vGrmr && !vGrammarStr.empty()) {
    throw std::runtime_error("failed to parse grammar");
  }

  // Compute prefill tokens from the generation prompt
  std::vector<mgs_token> lPrefillTokens;
  if (!pParams.vGenerationPrompt.empty()) {
    GGML_ASSERT(vVocab != nullptr);
    auto vTokens = fCommonTokenize(vVocab, pParams.vGenerationPrompt, false, true);
    for (size_t vI = 0; vI < vTokens.size(); vI++) {
      std::string vPiece = fCommonTokenToPiece(vVocab, vTokens[vI], true);
      if (vI == 0 && std::isspace(vPiece[0]) && !std::isspace(pParams.vGenerationPrompt[0])) {
        // Some tokenizers will add a space before the first special token, need to exclude
        continue;
      }
      LOG_DBG("%s: prefill token: %d = %s\n", __func__, vTokens[vI], vPiece.c_str());
      lPrefillTokens.push_back(vTokens[vI]);
    }
  }

  // Feed generation prompt tokens to the grammar sampler so it advances past
  // tokens the template already placed in the prompt.
  // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
  if (vGrmr && !pParams.vGrammarLazy && fCommonGrammarNeedsPrefill(pParams.vGrammar)) {
    try {
      for (const auto &vToken : lPrefillTokens) {
        mgs_sampler_accept(vGrmr, vToken);
        LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, vToken);
      }
    } catch (std::exception &vE) {
      LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
              fCommonGrammarValue(pParams.vGrammar).c_str(), pParams.vGenerationPrompt.c_str());
      throw vE;
    }
  }

  // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for
  // thinking-block suppression)
  if (!pParams.lReasoningBudgetStart.empty() && !pParams.vReasoningBudgetEnd.empty() &&
      (pParams.vGrammarLazy || pParams.vReasoningBudgetTokens >= 0 || pParams.vReasoningControl)) {
    vRbudget = fCommonReasoningBudgetInit(
        vVocab, {pParams.lReasoningBudgetStart}, pParams.vReasoningBudgetEnd, pParams.lReasoningBudgetForced,
        pParams.vReasoningBudgetTokens < 0 ? INT_MAX : pParams.vReasoningBudgetTokens);

    for (const auto &vToken : lPrefillTokens) {
      mgs_sampler_accept(vRbudget, vToken);
      LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, vToken);
    }
  }

  // logit bias: user biases + model suppress tokens (-INFINITY)
  {
    std::vector<mgs_logit_bias> lMerged = pParams.lLogitBias;

    int32_t vNSuppress = 0;
    const mgs_token *vSuppress = mgs_vocab_get_suppress_tokens(vVocab, &vNSuppress);
    for (int32_t vI = 0; vI < vNSuppress; ++vI) {
      lMerged.push_back({vSuppress[vI], -INFINITY});
    }

    if (!lMerged.empty()) {
      lSamplers.push_back(mgs_sampler_init_logit_bias(mgs_vocab_n_tokens(vVocab), lMerged.size(), lMerged.data()));
    }
  }

  if (pParams.vMirostat == 0) {

    bool vUseAdaptiveP = false; // see below

    for (const auto &vCnstr : pParams.lSamplers) {
      switch (vCnstr) {
      case cCommonSamplerTypeDry: {
        std::vector<const char *> lCBreakers;
        lCBreakers.reserve(pParams.lDrySequenceBreakers.size());
        for (const auto &vStr : pParams.lDrySequenceBreakers) {
          lCBreakers.push_back(vStr.c_str());
        }
        lSamplers.push_back(mgs_sampler_init_dry(vVocab, mgs_model_n_ctx_train(pModel), pParams.vDryMultiplier,
                                                 pParams.vDryBase, pParams.vDryAllowedLength, pParams.vDryPenaltyLastN,
                                                 lCBreakers.data(), lCBreakers.size()));
      } break;
      case cCommonSamplerTypeTopK:
        lSamplers.push_back(mgs_sampler_init_top_k(pParams.vTopK));
        break;
      case cCommonSamplerTypeTopP:
        lSamplers.push_back(mgs_sampler_init_top_p(pParams.vTopP, pParams.vMinKeep));
        break;
      case cCommonSamplerTypeTopNSigma:
        lSamplers.push_back(mgs_sampler_init_top_n_sigma(pParams.vTopNSigma));
        break;
      case cCommonSamplerTypeMinP:
        lSamplers.push_back(mgs_sampler_init_min_p(pParams.vMinP, pParams.vMinKeep));
        break;
      case cCommonSamplerTypeXtc:
        lSamplers.push_back(
            mgs_sampler_init_xtc(pParams.vXtcProbability, pParams.vXtcThreshold, pParams.vMinKeep, pParams.vSeed));
        break;
      case cCommonSamplerTypeTypicalP:
        lSamplers.push_back(mgs_sampler_init_typical(pParams.vTypP, pParams.vMinKeep));
        break;
      case cCommonSamplerTypeTemperature:
        lSamplers.push_back(
            mgs_sampler_init_temp_ext(pParams.vTemp, pParams.vDynatempRange, pParams.vDynatempExponent));
        break;
      case cCommonSamplerTypeInfill:
        lSamplers.push_back(mgs_sampler_init_infill(vVocab));
        break;
      case cCommonSamplerTypePenalties:
        lSamplers.push_back(mgs_sampler_init_penalties(mgs_vocab_n_tokens(vVocab), pParams.vPenaltyLastN,
                                                       pParams.vPenaltyRepeat, pParams.vPenaltyFreq,
                                                       pParams.vPenaltyPresent));
        break;
      case cCommonSamplerTypeAdaptiveP:
        // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
        // a single token, so we will add `dist` at the end of the chain by default,
        // unless the user specifically included `adaptive-p`. we set this flag here
        // so we know to add the sampler at the very end.
        vUseAdaptiveP = true;
        break;
      default:
        GGML_ASSERT(false && "unknown sampler type");
      }
    }
    if (vUseAdaptiveP) {
      // only if user explicitly included adaptive-p sampler
      lSamplers.push_back(mgs_sampler_init_adaptive_p(pParams.vAdaptiveTarget, pParams.vAdaptiveDecay, pParams.vSeed));
    } else {
      // default: sample from distribution
      lSamplers.push_back(mgs_sampler_init_dist(pParams.vSeed));
    }
  } else if (pParams.vMirostat == 1) {
    lSamplers.push_back(mgs_sampler_init_temp(pParams.vTemp));
    lSamplers.push_back(mgs_sampler_init_mirostat(mgs_vocab_n_tokens(vVocab), pParams.vSeed, pParams.vMirostatTau,
                                                  pParams.vMirostatEta, 100));
  } else if (pParams.vMirostat == 2) {
    lSamplers.push_back(mgs_sampler_init_temp(pParams.vTemp));
    lSamplers.push_back(mgs_sampler_init_mirostat_v2(pParams.vSeed, pParams.vMirostatTau, pParams.vMirostatEta));
  } else {
    GGML_ASSERT(false && "unknown mirostat version");
  }

  for (auto *vSmpl : lSamplers) {
    mgs_sampler_chain_add(vChain, vSmpl);
  }

  if (vGrmr && pParams.vBackendSampling) {
    LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

    pParams.vBackendSampling = false;
  }

  if (vRbudget && pParams.vBackendSampling) {
    LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

    pParams.vBackendSampling = false;
  }

  auto *vResult = new common_sampler{
      /* .vParams  = */ pParams,
      /* .vGrmr    = */ vGrmr,
      /* .vRbudget = */ vRbudget,
      /* .vChain   = */ vChain,
      /* .vPrev    = */ ring_buffer<mgs_token>(std::max(32, pParams.vNPrev)),
      /* .lCur     = */ {},
      /* .vCurP    = */ {},
  };

  return vResult;
}

void fCommonSamplerFree(struct common_sampler *pGsmpl) {
  if (!pGsmpl) {
    return;
  }

  mgs_sampler_free(pGsmpl->vGrmr);
  mgs_sampler_free(pGsmpl->vRbudget);
  mgs_sampler_free(pGsmpl->vChain);

  delete pGsmpl;
}

static bool fGrammarShouldApply(struct common_sampler *pGsmpl) {
  if (!pGsmpl->vGrmr) {
    return false;
  }
  if (!pGsmpl->vRbudget) {
    return true;
  }
  if (pGsmpl->vParams.vGrammarLazy) {
    // if grammar is lazy, only apply when reasoning budget is not active
    const auto cState = fCommonReasoningBudgetGetState(pGsmpl->vRbudget);
    return cState == cReasoningBudgetIdle || cState == cReasoningBudgetDone;
  }
  return true;
}

void fCommonSamplerAccept(struct common_sampler *pGsmpl, mgs_token pToken, bool pIsGenerated) {
  if (!pGsmpl) {
    return;
  }

  const auto cTm = pGsmpl->mTm();

  // fGrammarShouldApply() checks the reasoning budget state, so calculate this before we accept
  const auto cAcceptGrammar = pIsGenerated && fGrammarShouldApply(pGsmpl);

  if (pGsmpl->vRbudget && pIsGenerated) {
    mgs_sampler_accept(pGsmpl->vRbudget, pToken);

    // if done, replay end sequence which may contain a grammar trigger
    const bool cIsDone = fCommonReasoningBudgetGetState(pGsmpl->vRbudget) == cReasoningBudgetDone;
    if (pGsmpl->vGrmr && !cAcceptGrammar && cIsDone) {
      const mgs_tokens *vEndSeq = fCommonReasoningBudgetGetEndMatch(pGsmpl->vRbudget);
      if (vEndSeq) {
        for (const mgs_token cEndToken : *vEndSeq) {
          mgs_sampler_accept(pGsmpl->vGrmr, cEndToken);
        }
      }
    }
  }

  if (pGsmpl->vGrmr && cAcceptGrammar) {
    mgs_sampler_accept(pGsmpl->vGrmr, pToken);
  }

  mgs_sampler_accept(pGsmpl->vChain, pToken);

  pGsmpl->vPrev.mPushBack(pToken);
}

void fCommonSamplerReset(struct common_sampler *pGsmpl) {
  if (!pGsmpl) {
    return;
  }

  pGsmpl->mReset();
}

struct common_sampler *fCommonSamplerClone(common_sampler *pGsmpl) {
  return new common_sampler{
      /* .vParams  = */ pGsmpl->vParams,
      /* .vGrmr    = */ mgs_sampler_clone(pGsmpl->vGrmr),
      /* .vRbudget = */ mgs_sampler_clone(pGsmpl->vRbudget),
      /* .vChain   = */ mgs_sampler_clone(pGsmpl->vChain),
      /* .vPrev    = */ pGsmpl->vPrev,
      /* .lCur     = */ pGsmpl->lCur,
      /* .vCurP    = */ pGsmpl->vCurP,
  };
}

void fCommonPerfPrint(const struct mgs_context *pCtx, const struct common_sampler *pGsmpl) {
  // TODO: measure grammar performance

  const double cTSamplingMs = pGsmpl ? 1e-3 * pGsmpl->vTTotalUs : 0;

  mgs_perf_sampler_data vDataSmpl;
  mgs_perf_context_data vDataCtx;

  memset(&vDataSmpl, 0, sizeof(vDataSmpl));
  memset(&vDataCtx, 0, sizeof(vDataCtx));

  if (pGsmpl) {
    auto &vData = vDataSmpl;

    vData = mgs_perf_sampler(pGsmpl->vChain);

    // note: the sampling time includes the samplers time + extra time spent in common/sampling
    LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, cTSamplingMs);
    LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, vData.t_sample_ms, vData.n_sample);
  }

  if (pCtx) {
    auto &vData = vDataCtx;

    vData = mgs_perf_context(pCtx);

    const double cTEndMs = 1e-3 * ggml_time_us();

    const double cTTotalMs = cTEndMs - vData.t_start_ms;
    const double cTUnaccMs = cTTotalMs - (cTSamplingMs + vData.t_p_eval_ms + vData.t_eval_ms);
    const double cTUnaccPc = 100.0 * cTUnaccMs / cTTotalMs;

    LOG_INF("%s:        load time = %10.2f ms\n", __func__, vData.t_load_ms);
    LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", __func__,
            vData.t_p_eval_ms, vData.n_p_eval, vData.t_p_eval_ms / vData.n_p_eval,
            1e3 / vData.t_p_eval_ms * vData.n_p_eval);
    LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n", __func__,
            vData.t_eval_ms, vData.n_eval, vData.t_eval_ms / vData.n_eval, 1e3 / vData.t_eval_ms * vData.n_eval);
    LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (cTEndMs - vData.t_start_ms),
            (vData.n_p_eval + vData.n_eval));
    LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n",
            __func__, cTUnaccMs, cTUnaccPc);
    LOG_INF("%s:    graphs reused = %10d\n", __func__, vData.n_reused);

    fCommonMemoryBreakdownPrint(pCtx);
  }
}

struct mgs_sampler *fCommonSamplerGet(const struct common_sampler *pGsmpl) {
  if (!pGsmpl) {
    return nullptr;
  }

  return pGsmpl->vChain;
}

mgs_token fCommonSamplerSample(struct common_sampler *pGsmpl, struct mgs_context *pCtx, int pIdx, bool pGrammarFirst) {
  mgs_synchronize(pCtx);

  // start measuring sampling time after the mgs_context synchronization in order to not measure any ongoing async
  // operations
  const auto cTm = pGsmpl->mTm();

  mgs_token vId = MGS_TOKEN_NULL;

  auto &vGrmr = pGsmpl->vGrmr;
  auto &vRbudget = pGsmpl->vRbudget;
  auto &vChain = pGsmpl->vChain;
  auto &vCurP = pGsmpl->vCurP; // initialized by set_logits

  pGsmpl->mSetLogits(pCtx, pIdx);

  // Check if a backend sampler has already sampled a token in which case we
  // return that token id directly.
  {
    vId = mgs_get_sampled_token_ith(pCtx, pIdx);

    if (vId != MGS_TOKEN_NULL) {
      LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, vId);

      GGML_ASSERT(!pGsmpl->vGrmr && "using grammar in combination with backend sampling is not supported");
      GGML_ASSERT(!pGsmpl->vRbudget && "using reasoning budget in combination with backend sampling is not supported");

      for (size_t vI = 0; vI < vCurP.size; ++vI) {
        if (vCurP.data[vI].id == vId) {
          vCurP.selected = vI;
          break;
        }
      }

      return vId;
    }
  }

  // apply reasoning budget first
  mgs_sampler_apply(vRbudget, &vCurP);

  if (pGrammarFirst && fGrammarShouldApply(pGsmpl)) {
    mgs_sampler_apply(vGrmr, &vCurP);
  }

  mgs_sampler_apply(vChain, &vCurP);

  vId = vCurP.data[vCurP.selected].id;

  if (pGrammarFirst || !fGrammarShouldApply(pGsmpl)) {
    return vId;
  }

  // check if it the sampled token fits the grammar (grammar-based rejection sampling)
  {
    mgs_token_data vSingleTokenData = {vId, 1.0f, 0.0f};
    mgs_token_data_array vSingleTokenDataArray = {&vSingleTokenData, 1, -1, false};

    mgs_sampler_apply(vGrmr, &vSingleTokenDataArray);

    const bool cIsValid = vSingleTokenDataArray.data[0].logit != -INFINITY;
    if (cIsValid) {
      return vId;
    }
  }

  // resampling:
  // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
  pGsmpl->mSetLogits(pCtx, pIdx);

  mgs_sampler_apply(vRbudget, &vCurP);

  if (fGrammarShouldApply(pGsmpl)) {
    mgs_sampler_apply(vGrmr, &vCurP);
  }

  mgs_sampler_apply(vChain, &vCurP);

  GGML_ASSERT(vCurP.selected != -1 && "no selected token during sampling - check your sampling configuration");

  vId = vCurP.data[vCurP.selected].id;

  return vId;
}

std::vector<mgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler *pGsmpl, struct mgs_context *pCtx,
                                                      const std::vector<int> &pIdxs, const mgs_tokens &pDraft,
                                                      bool pGrammarFirst) {
  GGML_ASSERT(pIdxs.size() == pDraft.size() + 1 && "idxs.size() must be draft.size() + 1");

  std::vector<mgs_token> lResult;
  lResult.reserve(pIdxs.size());

  const mgs_vocab *vVocab = mgs_model_get_vocab(mgs_get_model(pCtx));

  size_t vI = 0;
  for (; vI < pDraft.size(); vI++) {
    const mgs_token cId = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    fCommonSamplerAccept(pGsmpl, cId, true);

    lResult.push_back(cId);

    // do not accept draft tokens after an EOG - they are not output but would stay in the context
    // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
    if (pDraft[vI] != cId || (mgs_vocab_is_eog(vVocab, cId) && vI + 1 < pDraft.size())) {
      break;
    }
  }

  if (vI == pDraft.size()) {
    const mgs_token cId = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    fCommonSamplerAccept(pGsmpl, cId, true);

    lResult.push_back(cId);
  }

  return lResult;
}

std::vector<mgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler *pGsmpl, struct mgs_context *pCtx,
                                                      const mgs_tokens &pDraft, bool pGrammarFirst) {
  std::vector<int> lIdxs(pDraft.size() + 1);
  for (size_t vI = 0; vI < lIdxs.size(); ++vI) {
    lIdxs[vI] = vI;
  }

  return fCommonSamplerSampleAndAcceptN(pGsmpl, pCtx, lIdxs, pDraft, pGrammarFirst);
}

uint32_t fCommonSamplerGetSeed(const struct common_sampler *pGsmpl) { return mgs_sampler_get_seed(pGsmpl->vChain); }

bool fCommonSamplerReasoningBudgetForce(struct common_sampler *pGsmpl) {
  if (!pGsmpl) {
    return false;
  }

  return fCommonReasoningBudgetForce(pGsmpl->vRbudget);
}

// helpers

mgs_token_data_array *fCommonSamplerGetCandidates(struct common_sampler *pGsmpl, bool pDoSort) {
  const auto cTm = pGsmpl->mTm();

  auto *vRes = &pGsmpl->vCurP;

  if (pDoSort && !vRes->sorted) {
    // remember the selected token before sorting
    const mgs_token cId = vRes->data[vRes->selected].id;

    std::sort(vRes->data, vRes->data + vRes->size,
              [](const mgs_token_data &pA, const mgs_token_data &pB) { return pA.p > pB.p; });

    // restore the selected token after sorting
    for (size_t vI = 0; vI < vRes->size; ++vI) {
      if (vRes->data[vI].id == cId) {
        vRes->selected = vI;
        break;
      }
    }

    vRes->sorted = true;
  }

  return vRes;
}

mgs_token fCommonSamplerLast(const struct common_sampler *pGsmpl) { return pGsmpl->vPrev.mRat(0); }

std::string fCommonSamplerPrint(const struct common_sampler *pGsmpl) {
  std::string vResult = "logits ";

  for (int vI = 0; vI < mgs_sampler_chain_n(pGsmpl->vChain); vI++) {
    const auto *vSmpl = mgs_sampler_chain_get(pGsmpl->vChain, vI);
    vResult += std::string("-> ");
    vResult += std::string(mgs_sampler_name(vSmpl)) + " ";
  }

  return vResult;
}

std::string fCommonSamplerPrevStr(common_sampler *pGsmpl, mgs_context *pCtxMain, int pN) {
  pN = std::min(pN, (int)pGsmpl->vPrev.size());

  if (pN <= 0) {
    return "";
  }

  std::string vResult;
  vResult.reserve(8 * pN); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

  for (int vI = pN - 1; vI >= 0; vI--) {
    const mgs_token cId = pGsmpl->vPrev.mRat(vI);

    GGML_ASSERT(cId != MGS_TOKEN_NULL && "null token in the sampling history - should not happen");

    vResult += fCommonTokenToPiece(pCtxMain, cId);
  }

  return vResult;
}

char fCommonSamplerTypeToChr(enum common_sampler_type pCnstr) {
  switch (pCnstr) {
  case cCommonSamplerTypeDry:
    return 'd';
  case cCommonSamplerTypeTopK:
    return 'k';
  case cCommonSamplerTypeTypicalP:
    return 'y';
  case cCommonSamplerTypeTopP:
    return 'p';
  case cCommonSamplerTypeTopNSigma:
    return 's';
  case cCommonSamplerTypeMinP:
    return 'm';
  case cCommonSamplerTypeTemperature:
    return 't';
  case cCommonSamplerTypeXtc:
    return 'x';
  case cCommonSamplerTypeInfill:
    return 'i';
  case cCommonSamplerTypePenalties:
    return 'e';
  case cCommonSamplerTypeAdaptiveP:
    return 'a';
  default:
    return '?';
  }
}

std::string fCommonSamplerTypeToStr(enum common_sampler_type pCnstr) {
  switch (pCnstr) {
  case cCommonSamplerTypeDry:
    return "dry";
  case cCommonSamplerTypeTopK:
    return "top_k";
  case cCommonSamplerTypeTypicalP:
    return "typ_p";
  case cCommonSamplerTypeTopP:
    return "top_p";
  case cCommonSamplerTypeTopNSigma:
    return "top_n_sigma";
  case cCommonSamplerTypeMinP:
    return "min_p";
  case cCommonSamplerTypeTemperature:
    return "temperature";
  case cCommonSamplerTypeXtc:
    return "xtc";
  case cCommonSamplerTypeInfill:
    return "infill";
  case cCommonSamplerTypePenalties:
    return "penalties";
  case cCommonSamplerTypeAdaptiveP:
    return "adaptive_p";
  default:
    return "";
  }
}

std::vector<common_sampler_type> fCommonSamplerTypesFromNames(const std::vector<std::string> &pNames) {
  // sampler names can be written multiple ways; generate aliases from canonical names
  static const auto cSamplerNameMap = [] {
    // canonical sampler name mapping
    std::unordered_map<std::string, common_sampler_type> dCanonicalNameMap{
        {"dry", cCommonSamplerTypeDry},
        {"top_k", cCommonSamplerTypeTopK},
        {"top_p", cCommonSamplerTypeTopP},
        {"top_n_sigma", cCommonSamplerTypeTopNSigma},
        {"typ_p", cCommonSamplerTypeTypicalP},
        {"min_p", cCommonSamplerTypeMinP},
        {"temperature", cCommonSamplerTypeTemperature},
        {"xtc", cCommonSamplerTypeXtc},
        {"infill", cCommonSamplerTypeInfill},
        {"penalties", cCommonSamplerTypePenalties},
        {"adaptive_p", cCommonSamplerTypeAdaptiveP}};
    std::unordered_map<std::string, common_sampler_type> dAliasNameMap;
    for (const auto &vEntry : dCanonicalNameMap) {
      const std::string &vCanonical = vEntry.first;
      if (vCanonical.find('_') == std::string::npos) {
        continue;
      }
      // kebab-case: "top-k", "min-p", etc.
      {
        std::string vKebabCase = vCanonical;
        std::replace(vKebabCase.begin(), vKebabCase.end(), '_', '-');
        dAliasNameMap.insert({vKebabCase, vEntry.second});
      }
      // no dash: "topk", "minp", etc.
      {
        std::string vNoDash = vCanonical;
        vNoDash.erase(std::remove(vNoDash.begin(), vNoDash.end(), '_'), vNoDash.end());
        dAliasNameMap.insert({vNoDash, vEntry.second});
      }
    }
    // misc. aliases
    dAliasNameMap.insert({"nucleus", cCommonSamplerTypeTopP});
    dAliasNameMap.insert({"temp", cCommonSamplerTypeTemperature});
    dAliasNameMap.insert({"typ", cCommonSamplerTypeTypicalP});
    // include aliases + canonical names in the complete mapping
    dAliasNameMap.merge(dCanonicalNameMap);
    return dAliasNameMap;
  }();

  std::vector<common_sampler_type> lSamplers;
  lSamplers.reserve(pNames.size());

  for (const auto &vName : pNames) {
    std::string vNameLower = vName;
    std::transform(vNameLower.begin(), vNameLower.end(), vNameLower.begin(), ::tolower);
    auto vSampler = cSamplerNameMap.find(vNameLower);
    if (vSampler != cSamplerNameMap.end()) {
      lSamplers.push_back(vSampler->second);
      continue;
    }
    LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, vNameLower.c_str());
  }

  return lSamplers;
}

std::vector<common_sampler_type> fCommonSamplerTypesFromChars(const std::string &pChars) {
  std::unordered_map<char, common_sampler_type> dSamplerNameMap = {
      {fCommonSamplerTypeToChr(cCommonSamplerTypeDry), cCommonSamplerTypeDry},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeTopK), cCommonSamplerTypeTopK},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeTypicalP), cCommonSamplerTypeTypicalP},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeTopP), cCommonSamplerTypeTopP},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeTopNSigma), cCommonSamplerTypeTopNSigma},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeMinP), cCommonSamplerTypeMinP},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeTemperature), cCommonSamplerTypeTemperature},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeXtc), cCommonSamplerTypeXtc},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeInfill), cCommonSamplerTypeInfill},
      {fCommonSamplerTypeToChr(cCommonSamplerTypePenalties), cCommonSamplerTypePenalties},
      {fCommonSamplerTypeToChr(cCommonSamplerTypeAdaptiveP), cCommonSamplerTypeAdaptiveP},
  };

  std::vector<common_sampler_type> lSamplers;
  lSamplers.reserve(pChars.size());

  for (const auto &vC : pChars) {
    const auto cSampler = dSamplerNameMap.find(vC);
    if (cSampler != dSamplerNameMap.end()) {
      lSamplers.push_back(cSampler->second);
    } else {
      LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, vC);
    }
  }

  return lSamplers;
}
