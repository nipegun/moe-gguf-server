#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-monitor.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "mgs.h"
#include "../../src/mgs-ext.h" // staging API: MoE expert cache state and size, memory monitoring
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <mutex>
#include <filesystem>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int cHttpPollingSeconds = 1;

static uint32_t fServerNOutputsMax(const common_params &pParams) {
  const uint32_t cNBatch = pParams.vNBatch;

  if (pParams.vEmbedding ||
      (pParams.vPoolingType != MGS_POOLING_TYPE_UNSPECIFIED && pParams.vPoolingType != MGS_POOLING_TYPE_NONE)) {
    return cNBatch;
  }

  const uint32_t cNOutputsPerSeq = 1 + fCommonSpeculativeNMax(&pParams.vSpeculative);

  const uint64_t cNOutputs = (uint64_t)pParams.vNParallel * cNOutputsPerSeq;

  return std::max<uint32_t>(1, std::min<uint64_t>(cNBatch, cNOutputs));
}

enum slot_state {
  cSlotStateIdle,
  cSlotStateWaitOther, // after assigning a task, but waiting for parent slot to process prompt
  cSlotStateStarted,   // after assigning a task and about to process prompt
  cSlotStateProcessingPrompt,
  cSlotStateDonePrompt,
  cSlotStateGenerating,
};

struct server_slot; // forward declaration

struct server_batch {
  mgs_batch vBatch;
  bool vBatchRendered = false;

  struct token {
    int32_t vIdSlot;
    mgs_token vToken;
    mgs_pos vPos;
    bool vOutput;
  };
  std::vector<token> lTokens;
  int32_t vNTokensAlloc = 0;
  int32_t vNEmbd = 0;

  // track if given slot can be batched with slots already in the batch
  server_slot *vSlotBatched = nullptr;

  // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
  bool vHasEmbd = false;
  mgs_token *vTokensPtr = nullptr;
  std::vector<float> lEmbd;

  float vAloraScale = -1.0f;
  size_t vAloraDisabledId = 0;

  server_batch() {
    vBatch.pos = nullptr; // sentinel: uninitialized batch
  }

  ~server_batch() {
    if (vBatch.pos != nullptr) {
      mClear();
      mgs_batch_free(vBatch);
    }
  }

  void mInit(int32_t pNTokensAlloc, int32_t pNEmbd) {
    this->vNTokensAlloc = pNTokensAlloc;
    this->vNEmbd = pNEmbd;
    vBatch = mgs_batch_init(pNTokensAlloc, 0, 1);
    vTokensPtr = vBatch.token;
    lTokens.reserve(pNTokensAlloc);
  }

  bool mAdd(int32_t pIdSlot, mgs_token pToken, mgs_pos pPos, bool pOutput) {
    GGML_ASSERT(!vHasEmbd); // cannot mix tokens + embd in same batch
    GGML_ASSERT(vBatch.pos != nullptr);
    if ((int32_t)lTokens.size() >= vNTokensAlloc) {
      return false;
    }
    lTokens.push_back({pIdSlot, pToken, pPos, pOutput});
    return true;
  }

  bool mAdd(int32_t pIdSlot, const std::vector<float> &pEmbdIn, mgs_pos pPos, bool pOutput) {
    GGML_ASSERT(vBatch.pos != nullptr);
    if ((int32_t)lTokens.size() >= vNTokensAlloc) {
      return false;
    }
    lTokens.push_back({pIdSlot, MGS_TOKEN_NULL, pPos, pOutput});
    vHasEmbd = true;
    lEmbd.insert(lEmbd.end(), pEmbdIn.begin(), pEmbdIn.end());
    return true;
  }

  void mClear() {
    lTokens.clear();
    lEmbd.clear();
    fCommonBatchClear(vBatch);
    vSlotBatched = nullptr;
    vAloraScale = -1.0f;
    vAloraDisabledId = 0;
    vBatchRendered = false;
    vHasEmbd = false;
    if (vBatch.token == nullptr) {
      vBatch.token = vTokensPtr;
      vBatch.embd = nullptr;
    }
  }

  int32_t size() const { return (int32_t)lTokens.size(); }

  void mSetOutput(int32_t pIdx, bool pOutput) {
    GGML_ASSERT(pIdx >= 0 && pIdx < (int32_t)lTokens.size());
    lTokens[pIdx].vOutput = pOutput;
  }

  void mRender() {
    GGML_ASSERT(!vBatchRendered);
    GGML_ASSERT(vBatch.pos != nullptr);
    fCommonBatchClear(vBatch);
    for (int32_t vI = 0; vI < size(); vI++) {
      const auto &vT = lTokens[vI];
      fCommonBatchAdd(vBatch, vT.vToken, vT.vPos, {vT.vIdSlot}, vT.vOutput);
    }
    if (vHasEmbd) {
      vBatch.token = nullptr; // will be restored on clear()
      vBatch.embd = lEmbd.data();
    }
    vBatchRendered = true;
  }

  mgs_batch mGetView(int32_t pOff, int32_t pNTokens) const {
    GGML_ASSERT(vBatch.pos != nullptr);
    GGML_ASSERT(vBatchRendered);
    GGML_ASSERT(pOff >= 0 && pOff < size());
    GGML_ASSERT(pNTokens > 0 && pOff + pNTokens <= size());

    auto *vToken = vBatch.token ? vBatch.token + pOff : nullptr;
    auto *vEmbd = vBatch.embd ? vBatch.embd + pOff * vNEmbd : nullptr;

    mgs_batch vView = {
        pNTokens, vToken, vEmbd, vBatch.pos + pOff, vBatch.n_seq_id + pOff, vBatch.seq_id + pOff, vBatch.logits + pOff,
    };

    return vView;
  }
};

struct server_slot {
  int vId;

  mgs_context *vCtxTgt = nullptr;
  mgs_context *vCtxDft = nullptr;

  common_memory vMem;

  // multimodal
  mtmd_context *vMctx = nullptr;
  mtmd::batch_ptr vMbatch = nullptr;

  // speculative decoding
  common_speculative *vSpec;

  mgs_tokens vSpecDraft;
  mgs_tokens vSpecPrompt;
  std::vector<int32_t> lSpecIBatch;
  common_prompt_checkpoint vSpecCkpt;
  bool vSpecIsReplay = false;

  // TODO: move members that belong to the task (such as `vGeneratedText`, `vHasNewLine`) to task_results_state
  std::unique_ptr<const server_task> vTask;
  std::unique_ptr<const server_task> vTaskPrev; // used for debugging

  // used to determine the slot that has been used the longest
  int64_t vTLastUsed = -1;

  // generation props
  int32_t vNCtx = 0; // context size per slot
  int32_t vNKeep = 0;
  int32_t vNDecoded = 0;
  int32_t vNRemaining = -1;
  int32_t vIBatch = -1;

  int32_t vNPromptTokensCache = 0;
  int32_t vNPromptTokensProcessed = 0;

  size_t vLastNlPos = 0;

  std::string vGeneratedText;
  std::string vDebugGeneratedText;
  mgs_tokens vGeneratedTokens;

  std::vector<completion_token_output> lGeneratedTokenProbs;

  bool vHasNextToken = true;
  bool vHasNewLine = false;
  bool vTruncated = false;

  stop_type vStop;

  std::string vStoppingWord;

  // state
  slot_state vState = cSlotStateIdle;

  server_prompt vPrompt;

  bool mPromptSave(server_prompt_cache &pPromptCache) const {
    if (vPrompt.vTokens.size() == 0) {
      return false;
    }

    const size_t cCurSizeTgt = mgs_state_seq_get_size_ext(vCtxTgt, vId, MGS_STATE_SEQ_FLAGS_NONE);
    const size_t cCurSizeDft = vCtxDft ? mgs_state_seq_get_size_ext(vCtxDft, vId, MGS_STATE_SEQ_FLAGS_NONE) : 0;

    const size_t cCurSize = cCurSizeTgt + cCurSizeDft;

    SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
            (int)vPrompt.vTokens.size(), cCurSize / (1024.0 * 1024.0), cCurSizeDft / (1024.0 * 1024.0));

    auto *vCur = pPromptCache.mAlloc(vPrompt, cCurSizeTgt, cCurSizeDft);
    if (vCur == nullptr) {
      return false;
    }

    mgs_state_seq_get_data_ext(vCtxTgt, vCur->vData.lMain.data(), cCurSizeTgt, vId, MGS_STATE_SEQ_FLAGS_NONE);
    if (vCtxDft) {
      mgs_state_seq_get_data_ext(vCtxDft, vCur->vData.lDrft.data(), cCurSizeDft, vId, MGS_STATE_SEQ_FLAGS_NONE);
    }

    return true;
  }

  bool mPromptLoad(server_prompt_cache &pPromptCache, const server_tokens &pTokens) {
    bool vRes = pPromptCache.mLoad(vPrompt, pTokens, vCtxTgt, vCtxDft, vId);
    if (!vRes) {
      SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
    }

    return vRes;
  }

  void mPromptClear() {
    SLT_TRC(*this, "clearing prompt with %zu tokens\n", vPrompt.vTokens.size());

    vMem.mSeqRm(vId, -1, -1);

    vPrompt.mClear();
  }

  std::vector<common_adapter_lora_info> lLora;
  int32_t vAloraInvocationStart = -1;

  // sampling
  json vJsonSchema;

  common_sampler_ptr vSmpl;

  mgs_token vSampled; // in speculative mode, this is the last accepted token

  // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
  // corresponding to one token position (size = n_embd)
  std::vector<float> lInpEmbd;

  // stats
  size_t vNSentText = 0; // number of sent text character

  // TODO @ngxson : move all metrics to a sub-struct for clarity
  int64_t vTStartProcessPrompt;
  int64_t vTStartGeneration;
  int64_t vTPrintLast = 0;
  int32_t vNDecodedLast = 0;

  double vTPromptProcessing = 0.0; // ms
  double vTTokenGeneration = 0.0;  // ms

  std::function<void(int /* id_slot */)> vCallbackOnRelease;

  // Speculative decoding stats
  int32_t vNDraftTotal = 0;              // Total draft tokens generated
  int32_t vNDraftAccepted = 0;           // Draft tokens actually accepted
  int32_t vNDraftVerifSteps = 0;         // Total draft token verification steps by the target model
  std::vector<int32_t> lNAcceptedPerPos; // Accepted tokens per draft position

  void mReset() {
    SLT_DBG(*this, "%s", "\n");

    vSpecIsReplay = false;

    vNPromptTokensCache = 0;

    vLastNlPos = 0;
    vGeneratedText = "";
    vHasNewLine = false;
    vTruncated = false;
    vStop = cStopTypeNone;
    vStoppingWord = "";
    vNSentText = 0;

    if (mCanSpeculate()) {
      vSpecDraft.clear();
      lSpecIBatch.clear();
      vSpecCkpt.mClear();
    }
    vGeneratedTokens.clear();
    lGeneratedTokenProbs.clear();
    vJsonSchema = json();

    // clear speculative decoding stats
    vNDraftTotal = 0;
    vNDraftAccepted = 0;
    vNDraftVerifSteps = 0;
    lNAcceptedPerPos.clear();

    vTaskPrev = std::move(vTask);
    vTask.reset();

    mgs_set_sampler(vCtxTgt, vId, nullptr);

    // clear alora start
    vAloraInvocationStart = -1;

    // clear multimodal state
    vMbatch.reset();
  }

  void mInitSampler() const {
    fCommonSamplerReset(vSmpl.get());

    if (!vTask->mNeedSampling()) {
      return;
    }

    const int64_t cTStart = ggml_time_us();

    int vNText = 0;

    for (int vI = 0; vI < (int)vPrompt.vTokens.size(); vI++) {
      const mgs_token cId = vPrompt.vTokens[vI];

      if (cId != MGS_TOKEN_NULL) {
        fCommonSamplerAccept(vSmpl.get(), cId, false);
        vNText++;
      }
    }

    SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n", (ggml_time_us() - cTStart) / 1000.0,
            vNText, (int)vPrompt.vTokens.size());
  }

  bool mNeedEmbd() const {
    GGML_ASSERT(vTask);
    return vTask->mNeedEmbd() || (vSpec && fCommonSpeculativeNeedEmbd(vSpec));
  }

  bool mNeedEmbdNextn() const {
    GGML_ASSERT(vTask);
    return vSpec && fCommonSpeculativeNeedEmbdNextn(vSpec);
  }

  // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
  // also we cannot split if the pooling would require any past tokens
  // (MTP supports splitting — uses task->mNeedEmbd() not mNeedEmbd())
  bool mCanSplit() const {
    GGML_ASSERT(vTask);

    return !vTask->mNeedEmbd() || (mgs_get_memory(vCtxTgt) && mgs_pooling_type(vCtxTgt) == MGS_POOLING_TYPE_LAST);
  }

  bool mCanBatchWith(server_slot &pOtherSlot) const {
    GGML_ASSERT(vTask);

    return vTask->vType == pOtherSlot.vTask->vType && lInpEmbd.size() == pOtherSlot.lInpEmbd.size() &&
           fAreLoraEqual(lLora, pOtherSlot.lLora);
  }

  bool mHasBudget(const common_params &pGlobalParams) {
    GGML_ASSERT(vTask);

    if (vTask->vParams.vNPredict == -1 && pGlobalParams.vNPredict == -1) {
      return true; // limitless
    }

    vNRemaining = -1;

    if (vTask->vParams.vNPredict != -1) {
      vNRemaining = vTask->vParams.vNPredict - vNDecoded;
    } else if (pGlobalParams.vNPredict != -1) {
      vNRemaining = pGlobalParams.vNPredict - vNDecoded;
    }

    return vNRemaining > 0; // no budget
  }

  bool mIsProcessing() const { return vState != cSlotStateIdle; }

  bool mCanSpeculate() const { return !!vSpec; }

  void mAddToken(const completion_token_output &pToken) {
    if (!mIsProcessing()) {
      SLT_WRN(*this, "%s", "slot is not processing\n");
      return;
    }

    lGeneratedTokenProbs.push_back(pToken);
  }

  int mGetNDraftMax() const {
    GGML_ASSERT(vTask);

    if (!mCanSpeculate()) {
      return 0;
    }

    // determine the max draft that fits the current slot state
    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    int vNDraftMax = vNCtx - vPrompt.mNTokens() - 2;

    if (vNRemaining > 0) {
      vNDraftMax = std::min(vNDraftMax, vNRemaining - 1);
    }

    SLT_DBG(*this, "max possible draft: %d\n", vNDraftMax);

    return vNDraftMax;
  }

  // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
  void mHandleLastSampledToken(server_batch &pBatch) {
    bool vAddOk = true;
    if (vSpecDraft.empty()) {
      // no speculative decoding
      vIBatch = pBatch.size();

      if (!lInpEmbd.empty()) {
        vAddOk &= pBatch.mAdd(vId, lInpEmbd, vPrompt.vTokens.mPosNext(), true);
      } else {
        vAddOk &= pBatch.mAdd(vId, vSampled, vPrompt.vTokens.mPosNext(), true);
      }

      SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n", vSampled, vNCtx,
              vPrompt.mNTokens(), vTruncated);
    } else {
      SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n", vSampled, vPrompt.vTokens.size(),
              vSpecDraft.size(), vPrompt.vTokens.mPosNext());

      GGML_ASSERT(lSpecIBatch.empty());

      lSpecIBatch.push_back(pBatch.size());
      for (size_t vI = 0; vI < vSpecDraft.size(); vI++) {
        lSpecIBatch.push_back(pBatch.size() + vI + 1);
      }

      auto vPos0 = vPrompt.vTokens.mPosNext();

      vAddOk &= pBatch.mAdd(vId, vSampled, vPos0++, true);
      for (auto vToken : vSpecDraft) {
        vAddOk &= pBatch.mAdd(this->vId, vToken, vPos0++, true);
      }
    }

    GGML_ASSERT(vAddOk && "batch must be large enough to hold the sampled and draft tokens");

    vPrompt.vTokens.mPushBack(vSampled);
    vPrompt.vTokens.mInsert(vSpecDraft);
  }

  void mRelease() {
    if (mIsProcessing()) {
      GGML_ASSERT(vTask);

      SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", vPrompt.mNTokens(), vTruncated);

      vTLastUsed = ggml_time_us();
      vTTokenGeneration = (ggml_time_us() - vTStartGeneration) / 1e3;

      vState = cSlotStateIdle;

      // do not keep context of the child slots - the parent's context is enough
      if (vTask->mIsChild()) {
        mPromptClear();
      }

      mReset();

      vCallbackOnRelease(vId);
    }
  }

  result_timings mGetTimings() const {
    result_timings vTimings;
    vTimings.vCacheN = vNPromptTokensCache;

    vTimings.vPromptN = vNPromptTokensProcessed;
    vTimings.vPromptMs = vTPromptProcessing;
    vTimings.vPromptPerTokenMs = vTPromptProcessing / vNPromptTokensProcessed;
    vTimings.vPromptPerSecond = 1e3 / vTPromptProcessing * vNPromptTokensProcessed;

    vTimings.vPredictedN = vNDecoded;
    vTimings.vPredictedMs = vTTokenGeneration;
    vTimings.vPredictedPerTokenMs = vTTokenGeneration / vNDecoded;
    vTimings.vPredictedPerSecond = 1e3 / vTTokenGeneration * vNDecoded;

    // Add speculative metrics
    if (vNDraftTotal > 0) {
      vTimings.vDraftN = vNDraftTotal;
      vTimings.vDraftNAccepted = vNDraftAccepted;
    }

    return vTimings;
  }

  size_t mFindStoppingStrings(const std::string &pText, const size_t pLastTokenSize, bool pIsFullStop) {
    GGML_ASSERT(vTask);

    size_t vStopPos = std::string::npos;

    for (const std::string &vWord : vTask->vParams.lAntiprompt) {
      size_t vPos;

      if (pIsFullStop) {
        const size_t cTmp = vWord.size() + pLastTokenSize;
        const size_t cFromPos = pText.size() > cTmp ? pText.size() - cTmp : 0;

        vPos = pText.find(vWord, cFromPos);
      } else {
        // otherwise, partial stop
        vPos = fStringFindPartialStop(pText, vWord);
      }

      if (vPos != std::string::npos && (vStopPos == std::string::npos || vPos < vStopPos)) {
        if (pIsFullStop) {
          vStop = cStopTypeWord;
          vStoppingWord = vWord;
          vHasNextToken = false;
        }
        vStopPos = vPos;
      }
    }

    return vStopPos;
  }

  void mPrintTimingsTg() {
    if (vNDecoded < 100) {
      return;
    }

    const int64_t cTNow = ggml_time_us();

    if (cTNow - vTPrintLast < 3 * 1000 * 1000) {
      return;
    }

    const double cNGenSecond = 1e3 / (vTTokenGeneration) * (vNDecoded);
    const double cNGenSecondWin = 1e6 / (cTNow - vTPrintLast) * (vNDecoded - vNDecodedLast);

    vTPrintLast = cTNow;
    vNDecodedLast = vNDecoded;

    SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", vNDecoded, cNGenSecond, cNGenSecondWin);
  }

  void mPrintTimingsPp() const {
    const double cNPromptSecond = 1e3 / vTPromptProcessing * vNPromptTokensProcessed;
    const double cFProgress = (float)vPrompt.mNTokens() / vTask->mNTokens();

    if (vTPromptProcessing < 3000.0) {
      return;
    }

    SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
            vNPromptTokensProcessed, cFProgress, vTPromptProcessing / 1e3, cNPromptSecond);
  }

  void mPrintTimings() const {
    const double cTPrompt = vTPromptProcessing / vNPromptTokensProcessed;
    const double cNPromptSecond = 1e3 / vTPromptProcessing * vNPromptTokensProcessed;

    const double cTGen = vTTokenGeneration / vNDecoded;
    const double cNGenSecond = 1e3 / vTTokenGeneration * vNDecoded;

    SLT_INF(*this, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            vTPromptProcessing, vNPromptTokensProcessed, cTPrompt, cNPromptSecond);

    SLT_INF(*this, "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            vTTokenGeneration, vNDecoded, cTGen, cNGenSecond);

    SLT_INF(*this, "      total time = %10.2f ms / %5d tokens\n", vTPromptProcessing + vTTokenGeneration,
            vNPromptTokensProcessed + vNDecoded);

    SLT_INF(*this, "   graphs reused = %10d\n", mgs_perf_context(vCtxTgt).n_reused);

    if (vNDraftTotal > 0) {
      const float cDraftRatio = (float)vNDraftAccepted / vNDraftTotal;
      const double cMeanAccLen =
          vNDraftVerifSteps > 0 ? 1.0 + (double)vNDraftAccepted / (double)vNDraftVerifSteps : 1.0;

      std::string vAcceptanceRatesPerPos;
      if (vNDraftVerifSteps > 0) {
        for (size_t vI = 0; vI < lNAcceptedPerPos.size(); ++vI) {
          if (vI > 0) {
            vAcceptanceRatesPerPos += ", ";
          }
          vAcceptanceRatesPerPos += fStringFormat("%.3f", (double)lNAcceptedPerPos[vI] / (double)vNDraftVerifSteps);
        }
      }

      SLT_INF(*this, "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n", cDraftRatio,
              vNDraftAccepted, vNDraftTotal, cMeanAccLen);
      SLT_TRC(*this, "     acc per pos = (%s)\n", vAcceptanceRatesPerPos.c_str());
    }

    fCommonSpeculativePrintStats(vSpec);
  }

  json to_json(bool pOnlyMetrics = false) const {
    json vRes;

    vRes = {
        {"id", vId},
        {"n_ctx", vNCtx},
        {"speculative", mCanSpeculate()},
        {"is_processing", mIsProcessing()},
    };

    const auto &vPtask = vTask ? vTask : vTaskPrev;

    if (vPtask) {
      vRes["id_task"] = vPtask->vId;
      vRes["n_prompt_tokens"] = (int32_t)vPrompt.vTokens.size();
      vRes["n_prompt_tokens_processed"] = vNPromptTokensProcessed;
      vRes["n_prompt_tokens_cache"] = vNPromptTokensCache;
      vRes["params"] = vPtask->vParams.to_json(pOnlyMetrics);
      vRes["next_token"] = {{
          {"has_next_token", vHasNextToken},
          {"has_new_line", vHasNewLine},
          {"n_remain", vNRemaining},
          {"n_decoded", vNDecoded},
      }};

      if (!pOnlyMetrics) {
        vRes["prompt"] = vPtask->vTokens.mDetokenize(vCtxTgt, true);
        vRes["generated"] = vGeneratedText.empty() ? vDebugGeneratedText : vGeneratedText;
      }
    }

    return vRes;
  }

  void mCopyStateTo(server_slot &pOther) const {
    GGML_ASSERT(vState == cSlotStateDonePrompt);

    vMem.mSeqRm(pOther.vId, -1, -1);
    vMem.mSeqCp(vId, pOther.vId, -1, -1);

    pOther.vNDecoded = vNDecoded;
    pOther.vNRemaining = vNRemaining;
    pOther.vIBatch = vIBatch;

    pOther.vTStartProcessPrompt = vTStartProcessPrompt;
    pOther.vTPromptProcessing = vTPromptProcessing;
    pOther.vNPromptTokensCache = vNPromptTokensCache;
    pOther.vNPromptTokensProcessed = vNPromptTokensProcessed;

    pOther.vPrompt = vPrompt.mClone();
    pOther.mInitSampler();
  }

  // returns 0 on success
  // caller need to update prompt.tokens after a successful call to keep track of the processing progress
  int mProcessMtmdChunk(size_t pIdx, size_t &pNTokensOut) {
    GGML_ASSERT(vMctx);
    const auto &vInputTokens = vTask->vTokens;
    const auto &vChunk = vInputTokens.mFindChunk(pIdx);
    int32_t vRes = 0;

    auto vTryDecode = [&]() -> int32_t {
      if (vMbatch) {
        float *vEmbd = mtmd_batch_get_output_embd(vMbatch.get(), vChunk.get());
        if (vEmbd) {
          void *vCbData = vSpec;
          static auto vCb = [](mgs_batch pBatch, void *pUserData) {
            common_speculative *vSpec = static_cast<common_speculative *>(pUserData);
            if (!fCommonSpeculativeProcess(vSpec, pBatch)) {
              return 1;
            }
            return 0;
          };

          mgs_pos vNewNPast; // unused for now
          vRes = mtmd_helper_decode_image_chunk(vMctx, vCtxTgt, vChunk.get(), vEmbd, vPrompt.vTokens.mPosNext(), vId,
                                                mgs_n_batch(vCtxTgt), &vNewNPast, vCb, vCbData);
          if (vRes != 0) {
            SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", pIdx, vRes);
            return -1;
          }
          pNTokensOut = mtmd_input_chunk_get_n_tokens(vChunk.get());
          return 0; // success
        }
      }
      return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    vRes = vTryDecode();
    if (vRes == 0) {
      return 0;
    }
    if (vRes < 0) {
      // fatal error
      return vRes;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    vMbatch.reset(mtmd_batch_init(vMctx));
    vRes = mtmd_batch_add_chunk(vMbatch.get(), vChunk.get());
    GGML_ASSERT(vRes == 0); // we should never have an empty batch

    // try batching as much as possible
    int vNAdded = 1;
    size_t vIdxCur = pIdx;
    while (vRes == 0) {
      auto [vNextChunk, vNextIdx] = vInputTokens.mFindNextMediaChunk(vIdxCur);
      if (vNextChunk == nullptr) {
        break;
      }
      vRes = mtmd_batch_add_chunk(vMbatch.get(), vNextChunk->get());
      vNAdded += (vRes == 0 ? 1 : 0);
      vIdxCur = vNextIdx;
      SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", vNextIdx, vRes);
      // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", pIdx, vNAdded);

    vRes = mtmd_batch_encode(vMbatch.get());
    if (vRes != 0) {
      SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", pIdx, vRes);
      return -1;
    }

    return vTryDecode();
  }
};

//
// server_metrics
//

struct server_metrics {
  int64_t vTStart = 0;

  uint64_t vNPromptTokensProcessedTotal = 0;
  uint64_t vTPromptProcessingTotal = 0;
  uint64_t vNTokensPredictedTotal = 0;
  uint64_t vTTokensGenerationTotal = 0;

  uint64_t vNTokensMax = 0;

  uint64_t vNPromptTokensProcessed = 0;
  uint64_t vTPromptProcessing = 0;

  uint64_t vNTokensPredicted = 0;
  uint64_t vTTokensGeneration = 0;

  uint64_t vNDecodeTotal = 0;
  uint64_t vNBusySlotsTotal = 0;

  uint64_t vNDraftTokensTotal = 0;
  uint64_t vNDraftAcceptedTotal = 0;
  uint64_t vNDraftVerifStepsTotal = 0;
  std::vector<uint64_t> lNAcceptedPerPosTotal;

  void mInit() { vTStart = ggml_time_us(); }

  void mOnPromptEval(const server_slot &pSlot) {
    vNPromptTokensProcessedTotal += pSlot.vNPromptTokensProcessed;
    vNPromptTokensProcessed += pSlot.vNPromptTokensProcessed;
    vTPromptProcessing += pSlot.vTPromptProcessing;
    vTPromptProcessingTotal += pSlot.vTPromptProcessing;

    vNTokensMax = std::max(vNTokensMax, (uint64_t)pSlot.vPrompt.mNTokens());
  }

  void mOnPrediction(const server_slot &pSlot) {
    vNTokensPredictedTotal += pSlot.vNDecoded;
    vNTokensPredicted += pSlot.vNDecoded;
    vTTokensGeneration += pSlot.vTTokenGeneration;
    vTTokensGenerationTotal += pSlot.vTTokenGeneration;

    vNDraftTokensTotal += pSlot.vNDraftTotal;
    vNDraftAcceptedTotal += pSlot.vNDraftAccepted;
    vNDraftVerifStepsTotal += pSlot.vNDraftVerifSteps;

    if (lNAcceptedPerPosTotal.size() < pSlot.lNAcceptedPerPos.size()) {
      lNAcceptedPerPosTotal.resize(pSlot.lNAcceptedPerPos.size(), 0);
    }
    for (size_t vI = 0; vI < pSlot.lNAcceptedPerPos.size(); vI++) {
      lNAcceptedPerPosTotal[vI] += pSlot.lNAcceptedPerPos[vI];
    }
  }

  void mOnDecoded(const std::vector<server_slot> &pSlots) {
    vNDecodeTotal++;
    for (const auto &vSlot : pSlots) {
      if (vSlot.mIsProcessing()) {
        vNBusySlotsTotal++;
      }
      vNTokensMax = std::max(vNTokensMax, (uint64_t)vSlot.vPrompt.mNTokens());
    }
  }

  void mResetBucket() {
    vNPromptTokensProcessed = 0;
    vTPromptProcessing = 0;
    vNTokensPredicted = 0;
    vTTokensGeneration = 0;
  }
};

//
// server_context_impl (private implementation)
//

// MoE expert cache state reported by the engine, for /expert-cache and the metrics.
static json fExpertCacheJson(const mgs_context *pCtx) {
  const mgs_expert_cache_info vInfo = mgs_get_expert_cache_info(pCtx);
  const uint64_t vLookups = vInfo.n_hit + vInfo.n_miss;
  const uint64_t vPrefetchBytes = vInfo.prefetch_cache_bytes + vInfo.prefetch_host_bytes;
  return json{
      {"active", vInfo.active},
      {"size_bytes", vInfo.size},
      {"layers", vInfo.n_layer},
      {"slots", vInfo.n_slot},
      {"experts", vInfo.n_expert},
      {"steps", vInfo.n_step},
      {"hits", vInfo.n_hit},
      {"misses", vInfo.n_miss},
      {"hit_ratio", vLookups > 0 ? (double)vInfo.n_hit / (double)vLookups : 0.0},
      {"uploads", vInfo.n_upload},
      {"cpu_experts", vInfo.n_cpu_expert},
      {"prefetch_cache_bytes", vInfo.prefetch_cache_bytes},
      {"prefetch_host_bytes", vInfo.prefetch_host_bytes},
      {"prefetch_cache_ratio", vPrefetchBytes > 0 ? (double)vInfo.prefetch_cache_bytes / (double)vPrefetchBytes : 0.0},
      {"upload_bandwidth_bytes_per_second", vInfo.upload_bandwidth},
      {"cpu_bandwidth_bytes_per_second", vInfo.cpu_bandwidth},
  };
}
struct server_context_impl {
  friend struct server_context;

public:
  // only use these pointers outside of this class:
  //  - when not in sleeping state
  //  - and, with thread-safe APIs (e.g., tokenizer calls)
  mgs_model *vModelTgt = nullptr;

  mtmd_context *vMctx = nullptr;
  const mgs_vocab *vVocab = nullptr;

  server_queue vQueueTasks;
  server_response vQueueResults;

  // note: chat_params must not be refreshed upon existing sleeping state
  server_chat_params vChatParams;

  server_state_callback_t vCallbackState = [](server_state, json) -> void {};

  server_context_impl() { mtmd_helper_log_set(fCommonLogDefaultCallback, nullptr); }

  // Latest memory snapshot published by the main loop (see mRefreshMonitorSnapshot). Safe to call
  // from the HTTP threads: it never touches the contexts nor the task queue, so it neither wakes a
  // sleeping model nor delays its idle timer.
  json mGetMonitorSnapshot() const {
    std::lock_guard<std::mutex> vLock(vMonitorMutex);
    return dMonitorSnapshot;
  }

  ~server_context_impl() {
    if (!vSleeping) {
      // mDestroy() is already called when entering sleeping state
      // we don't call it again here to avoid double free
      mDestroy();
    }
  }

private:
  // note: accessing these fields outside of this class is not thread-safe
  // use server_context methods instead

  common_params vParamsBase;

  // note: keep these alive - they determine the lifetime of the model, context, etc.
  common_init_result_ptr vMgsInit;

  mgs_context *vCtxTgt = nullptr;

  server_batch vBatch;

  mgs_model *vModelDft = nullptr;
  mgs_context *vCtxDft = nullptr;

  common_speculative_init_result_ptr vSpecInit;

  common_context_seq_rm_type vCtxTgtSeqRmType = cCommonContextSeqRmTypeNo;
  common_context_seq_rm_type vCtxDftSeqRmType = cCommonContextSeqRmTypeNo;

  common_speculative_ptr vSpec;

  bool vAddBosToken = true;

  int32_t vNCtx; // total context for all clients / slots

  // set to mgs_model_n_swa(model)
  // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
  int32_t vNSwa;

  // slots / clients
  std::vector<server_slot> lSlots;

  int vTrace = 0;
  int vSlotsDebug = 0;
  int vNEmptyConsecutive = 0;

  std::unique_ptr<server_prompt_cache> vPromptCache;

  server_metrics vMetrics;

  json vJsonUiSettings = json::object();

  // Necessary similarity of prompt for slot selection
  float vSlotPromptSimilarity = 0.0f;

  std::string vModelName;              // name of the loaded model, to be used by API
  std::set<std::string> sModelAliases; // additional names for the model
  std::set<std::string> sModelTags;    // informational tags

  bool vSleeping = false;

  int64_t vTLastLoadProgressMs = 0;

  // memory monitoring: the snapshot is built on the main loop and copied by the HTTP threads
  mutable std::mutex vMonitorMutex;
  json dMonitorSnapshot = json{{"state", "loading"}};
  int64_t vMonitorRefreshMs = 0;
  mgs_model_weight_breakdown dMonitorWeights; // MTP and expert weights of model_tgt, fixed after loading
  bool vMonitorSpecMtp = false;               // ctx_dft is an MTP context (its own weights are MTP weights)

  void mPublishMonitorSnapshot(json pDSnapshot) {
    std::lock_guard<std::mutex> vLock(vMonitorMutex);
    dMonitorSnapshot = std::move(pDSnapshot);
  }

  // Memory of the loaded contexts by purpose, split between the GPUs and the RAM. Only the main
  // loop may call it: it reads the contexts and the slots.
  json mBuildMonitorSnapshot() const {
    std::map<ggml_backend_buffer_type_t, std::map<std::string, size_t>> dSegments;
    const auto fAdd = [&dSegments](ggml_backend_buffer_type_t pType, const char *pKey, size_t pBytes) {
      if (pType != nullptr && pBytes > 0) {
        dSegments[pType][pKey] += pBytes;
      }
    };

    for (const auto &[vType, vData] : mgs_get_memory_breakdown(vCtxTgt)) {
      size_t vMtp = 0;
      size_t vExperts = 0;
      const auto vSplit = dMonitorWeights.find(vType);
      if (vSplit != dMonitorWeights.end()) {
        vMtp = std::min(vSplit->second.mtp, vData.model);
        vExperts = std::min(vSplit->second.experts, vData.model - vMtp);
      }
      fAdd(vType, "mtp_weights", vMtp);
      fAdd(vType, "expert_weights", vExperts);
      fAdd(vType, "dense_weights", vData.model - vMtp - vExperts);
      fAdd(vType, "kv_cache", vData.context);
      fAdd(vType, "compute", vData.compute);
    }
    for (const auto &[vType, vData] : mgs_get_expert_memory_breakdown(vCtxTgt)) {
      fAdd(vType, "expert_cache", vData.cache);
      fAdd(vType, "prefetch", vData.prefetch);
    }
    if (vCtxDft != nullptr) {
      // an MTP context on the target model shares its weights; a separate GGUF brings its own
      const bool vOwnWeights = mgs_get_model(vCtxDft) != vModelTgt;
      for (const auto &[vType, vData] : mgs_get_memory_breakdown(vCtxDft)) {
        if (vOwnWeights) {
          fAdd(vType, vMonitorSpecMtp ? "mtp_weights" : "draft_weights", vData.model);
        }
        fAdd(vType, "draft_kv_cache", vData.context);
        fAdd(vType, "compute", vData.compute);
      }
      for (const auto &[vType, vData] : mgs_get_expert_memory_breakdown(vCtxDft)) {
        fAdd(vType, "expert_cache", vData.cache);
        fAdd(vType, "prefetch", vData.prefetch);
      }
    }

    // devices of the models first, in their order, then any other device that holds a buffer
    std::vector<ggml_backend_dev_t> lLDevices;
    const auto fAddDevice = [&lLDevices](ggml_backend_dev_t pDevice) {
      if (pDevice != nullptr && std::find(lLDevices.begin(), lLDevices.end(), pDevice) == lLDevices.end()) {
        lLDevices.push_back(pDevice);
      }
    };
    for (int vIndex = 0; vIndex < mgs_model_n_devices(vModelTgt); ++vIndex) {
      fAddDevice(mgs_model_get_device(vModelTgt, vIndex));
    }
    if (vModelDft != nullptr) {
      for (int vIndex = 0; vIndex < mgs_model_n_devices(vModelDft); ++vIndex) {
        fAddDevice(mgs_model_get_device(vModelDft, vIndex));
      }
    }

    json dHostSegments = json::object();
    std::map<ggml_backend_dev_t, json> dDeviceSegments;
    for (const auto &[vType, dKeys] : dSegments) {
      ggml_backend_dev_t vDevice = ggml_backend_buft_get_device(vType);
      const enum ggml_backend_dev_type vDeviceType =
          vDevice != nullptr ? ggml_backend_dev_type(vDevice) : GGML_BACKEND_DEVICE_TYPE_CPU;
      // pinned host buffers of a GPU are RAM too
      const bool vHost = ggml_backend_buft_is_host(vType) ||
                         (vDeviceType != GGML_BACKEND_DEVICE_TYPE_GPU && vDeviceType != GGML_BACKEND_DEVICE_TYPE_IGPU);
      if (!vHost) {
        fAddDevice(vDevice);
      }
      json &dTarget = vHost ? dHostSegments : dDeviceSegments[vDevice];
      if (!dTarget.is_object()) {
        dTarget = json::object();
      }
      for (const auto &[vKey, vBytes] : dKeys) {
        dTarget[vKey] = dTarget.value(vKey, (uint64_t)0) + (uint64_t)vBytes;
      }
    }

    // saved prompts (--cache-ram) and the context checkpoints of the slots live in RAM
    size_t vPromptBytes = vPromptCache ? vPromptCache->size() : 0;
    for (const server_slot &vSlot : lSlots) {
      for (const auto &vCheckpoint : vSlot.vPrompt.lCheckpoints) {
        vPromptBytes += vCheckpoint.size();
      }
    }
    if (vPromptBytes > 0) {
      dHostSegments["prompt_cache"] = dHostSegments.value("prompt_cache", (uint64_t)0) + (uint64_t)vPromptBytes;
    }

    // where the kernel driver really keeps this process's GPU memory (VRAM or system memory through GTT)
    const json dResidency = fMonitorDrmResidency();

    json aDevices = json::array();
    for (ggml_backend_dev_t vDevice : lLDevices) {
      const enum ggml_backend_dev_type vDeviceType = ggml_backend_dev_type(vDevice);
      if (vDeviceType != GGML_BACKEND_DEVICE_TYPE_GPU && vDeviceType != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        continue;
      }
      ggml_backend_dev_props vProps;
      ggml_backend_dev_get_props(vDevice, &vProps);
      const auto vSegments = dDeviceSegments.find(vDevice);
      const std::string vId = vProps.device_id != nullptr ? vProps.device_id : ggml_backend_dev_name(vDevice);
      json dDevice = {
          {"id", vId},
          {"name", ggml_backend_dev_name(vDevice)},
          {"description", ggml_backend_dev_description(vDevice)},
          {"total_bytes", (uint64_t)vProps.memory_total},
          {"free_bytes", (uint64_t)vProps.memory_free},
          {"segments", vSegments != dDeviceSegments.end() ? vSegments->second : json::object()},
      };
      const std::string vKey = fMonitorDeviceKey(vId);
      if (dResidency.contains(vKey)) {
        dDevice["residency"] = dResidency.at(vKey);
      }
      aDevices.push_back(std::move(dDevice));
    }

    return json{
        {"state", "ready"},
        {"model", vModelName},
        {"rss_bytes", (uint64_t)fMonitorProcessRss()},
        {"devices", aDevices},
        {"host", {{"segments", dHostSegments}}},
        {"expert_cache", fExpertCacheJson(vCtxTgt)},
    };
  }

  // Rebuilds the snapshot at most once per second while the slots work, and always when they are
  // idle (the end of a request saves its prompt and may leave the last second unpublished).
  void mRefreshMonitorSnapshot(bool pForce) {
    if (vCtxTgt == nullptr) {
      return;
    }
    const int64_t vNow = ggml_time_ms();
    if (!pForce && vNow - vMonitorRefreshMs < 1000) {
      return;
    }
    vMonitorRefreshMs = vNow;
    try {
      json dSnapshot = mBuildMonitorSnapshot();
      mCheckVramResidency(dSnapshot);
      mPublishMonitorSnapshot(std::move(dSnapshot));
    } catch (const std::exception &vE) {
      SRV_WRN("failed to update the memory monitor: %s\n", vE.what());
    }
  }

  // Warns once when the kernel driver keeps part of the GPU memory of this process outside VRAM (evicted
  // to system memory through GTT, usually because another program needs VRAM), and once more when it is
  // back. Only drivers that publish fdinfo residency (amdgpu, i915, xe) can be checked.
  bool vVramWarned = false;

  void mCheckVramResidency(const json &pDSnapshot) {
    for (const json &dDevice : pDSnapshot.value("devices", json::array())) {
      if (!dDevice.contains("residency")) {
        continue;
      }
      const json &dResidency = dDevice["residency"];
      uint64_t vAllocated = 0;
      const json dSegments = dDevice.value("segments", json::object());
      for (const auto &[vKey, vValue] : dSegments.items()) {
        vAllocated += vValue.get<uint64_t>();
      }
      const uint64_t vVram = dResidency.value("vram_bytes", (uint64_t)0);
      const uint64_t vShortfall = vAllocated > vVram ? vAllocated - vVram : 0;
      const uint64_t vEvicted = std::max(dResidency.value("evicted_vram_bytes", (uint64_t)0), vShortfall);
      const uint64_t vLimit = std::max<uint64_t>(64ull << 20, vAllocated / 50);
      if (vEvicted > vLimit && !vVramWarned) {
        vVramWarned = true;
        SRV_WRN("%s: %.0f MiB of the %.0f MiB this model allocated on the GPU are not in VRAM: the kernel driver "
                "moved them to system memory (GTT), usually because another program needs VRAM, and inference "
                "slows down until they return; close that program or lower the expert cache (--moe-cache-size)\n",
                dDevice.value("description", std::string()).c_str(), vEvicted / 1048576.0, vAllocated / 1048576.0);
      } else if (vEvicted <= vLimit && vVramWarned) {
        vVramWarned = false;
        SRV_INF("%s: the GPU memory of this model is back in VRAM\n",
                dDevice.value("description", std::string()).c_str());
      }
    }
  }

  void mDestroy() {
    vSpec.reset();
    vSpecInit.reset();

    vCtxDft = nullptr;
    vModelDft = nullptr;

    vMgsInit.reset();

    vCtxTgt = nullptr;
    vModelTgt = nullptr;

    mtmd_free(vMctx);
    vMctx = nullptr;
  }

  void mHandleSleepingState(bool pNewState) {
    GGML_ASSERT(vSleeping != pNewState);
    if (pNewState) {
      SRV_INF("%s", "server is entering sleeping state\n");
      mDestroy();
      mPublishMonitorSnapshot(json{{"state", "sleeping"}, {"model", vModelName}});
    } else {
      SRV_INF("%s", "server is exiting sleeping state\n");
      if (!mLoadModel(vParamsBase)) {
        GGML_ABORT("failed to reload model after sleeping");
      }
    }
    vSleeping = pNewState;
  }

  struct load_progress_data {
    server_context_impl *vCtx;
    std::string vStage;
    std::vector<std::string> lStages;
    int64_t vTLastLoadProgressMs = 0;
    load_progress_data(server_context_impl *pCtx, const std::string &pStage) : vCtx(pCtx), vStage(pStage) {}
  };
  static bool mLoadProgressCallback(float pProgress, void *pUserData) {
    auto *vD = static_cast<load_progress_data *>(pUserData);
    GGML_ASSERT(vD);
    // always emit the first and final sample; throttle the rest to one per 200ms
    {
      auto &vTLast = vD->vTLastLoadProgressMs;
      const int64_t cTNow = ggml_time_ms();
      const bool cFirst = vTLast == 0;
      const bool cDone = pProgress >= 1.0f;
      const bool cThrottled = !cFirst && !cDone && (cTNow - vTLast) < 200;
      if (cThrottled) {
        return true;
      }
      vTLast = cTNow;
    }
    if (vD->vCtx->vCallbackState) {
      vD->vCtx->vCallbackState(cServerStateLoading, {
                                                        {"stages", vD->lStages},
                                                        {"current", vD->vStage},
                                                        {"value", pProgress},
                                                    });
    }
    return true;
  }

  // load the model and initialize mgs_context
  // this may also be called to resume from sleeping state
  bool mLoadModel(common_params &pParams) {
    load_progress_data vLoadProgressText(this, "text_model");
    load_progress_data vLoadProgressMmproj(this, "mmproj_model");
    load_progress_data vLoadProgressSpec(this, "spec_model");

    const bool cIsResume = vSleeping;

    vParamsBase = pParams;
    fCommonMaybeEnableEmbeddedMtp(vParamsBase);
    vParamsBase.vNOutputsMax = fServerNOutputsMax(vParamsBase);

    const bool cHasMmproj = !pParams.vMmproj.vPath.empty();
    const bool cHasDraft = vParamsBase.vSpeculative.mHasDft();
    const bool cSpecMtp = std::find(vParamsBase.vSpeculative.lTypes.begin(), vParamsBase.vSpeculative.lTypes.end(),
                                    cCommonSpeculativeTypeDraftMtp) != vParamsBase.vSpeculative.lTypes.end();
    const bool cHasSpec = cHasDraft || cSpecMtp;

    if (vCallbackState) {
      std::vector<std::string> lStages = {"text_model"};
      if (cHasSpec) {
        lStages.push_back("spec_model");
      }
      if (cHasMmproj) {
        lStages.push_back("mmproj_model");
      }
      vLoadProgressText.lStages = lStages;
      vLoadProgressMmproj.lStages = lStages;
      vLoadProgressSpec.lStages = lStages;

      // trigger 0% progress
      mLoadProgressCallback(0.0f, &vLoadProgressText);
    }

    SRV_INF("loading model '%s'\n", pParams.vModel.mGetName().c_str());
    SRV_TRC("local path '%s'\n", pParams.vModel.vPath.c_str());

    std::string &vMmprojPath = vParamsBase.vMmproj.vPath;
    mtmd_context_params vMparams = mtmd_context_params_default();
    if (cHasMmproj) {
      vMparams.use_gpu = vParamsBase.vMmprojUseGpu;
      vMparams.print_timings = false;
      vMparams.n_threads = vParamsBase.vCpuparams.vNThreads;
      vMparams.flash_attn_type = vParamsBase.vFlashAttnType;
      vMparams.warmup = vParamsBase.vWarmup;
      vMparams.image_min_tokens = vParamsBase.vImageMinTokens;
      vMparams.image_max_tokens = vParamsBase.vImageMaxTokens;
      vMparams.batch_max_tokens = vParamsBase.vMtmdBatchMaxTokens;
      vMparams.media_marker = fGetMediaMarker();
      // progress callback
      vMparams.progress_callback = mLoadProgressCallback;
      vMparams.progress_callback_user_data = &vLoadProgressMmproj;
    }

    // optionally get the memory usage of mmproj
    if (cHasMmproj && vParamsBase.vFitParams) {
      int64_t vTStart = ggml_time_us();
      auto vMmprojMem = mtmd_get_memory_usage(vMmprojPath.c_str(), vMparams);
      int64_t vTElapsed = ggml_time_us() - vTStart;
      if (!vMmprojMem.empty()) {
        size_t vTotal = 0;
        for (auto &[vDev, vSize] : vMmprojMem) {
          vTotal += vSize;
        }
        SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n",
                vTotal / (1024.0 * 1024.0), vTElapsed / 1000.0);
        GGML_ASSERT(!vParamsBase.lFitParamsTarget.empty());
        for (auto &[vDev, vSize] : vMmprojMem) {
          for (size_t vI = 0; vI < ggml_backend_dev_count(); vI++) {
            if (ggml_backend_dev_get(vI) == vDev) {
              if (vI < vParamsBase.lFitParamsTarget.size()) {
                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", vSize / (1024.0 * 1024.0),
                        ggml_backend_dev_name(vDev));
                vParamsBase.lFitParamsTarget[vI] += vSize;
              }
              break;
            }
          }
        }
      } else {
        SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
      }
    }

    // optionally reserve VRAM for the draft / MTP context before fitting the target model
    if (vParamsBase.vFitParams) {
      if (cHasSpec) {
        // MTP draft context lives on the target model, only context+compute are new
        bool vMeasureModelBytes = cHasDraft && !cSpecMtp;

        common_params vParamsDft = fCommonBaseParamsToSpeculative(vParamsBase);

        auto vMparamsDft = fCommonModelParamsToMgs(vParamsDft);
        auto vCparamsDft = fCommonContextParamsToMgs(vParamsDft);
        if (cSpecMtp) {
          vCparamsDft.ctx_type = MGS_CONTEXT_TYPE_MTP;
        }
        vCparamsDft.n_rs_seq = 0;

        std::vector<ggml_backend_dev_t> lDevs;
        uint32_t vHpNgl = 0;
        uint32_t vHpNct = 0;
        uint32_t vHpNex = 0;
        try {
          auto vDmd = fCommonGetDeviceMemoryData(vParamsDft.vModel.vPath.c_str(), &vMparamsDft, &vCparamsDft, lDevs,
                                                 vHpNgl, vHpNct, vHpNex, GGML_LOG_LEVEL_ERROR);

          GGML_ASSERT(!vParamsBase.lFitParamsTarget.empty());
          size_t vTotal = 0;

          std::vector<ggml_backend_dev_t> lTgtDevices = pParams.lDevices;

          if (lTgtDevices.empty()) {
            for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
              lTgtDevices.push_back(ggml_backend_dev_get(vI));
            }
          }

          for (size_t vJ = 0; vJ < lDevs.size(); ++vJ) {
            const size_t cBytes = (vMeasureModelBytes ? vDmd[vJ].vModel : 0) + vDmd[vJ].vContext + vDmd[vJ].vCompute;
            vTotal += cBytes;
            for (size_t vI = 0; vI < lTgtDevices.size(); vI++) {
              if (lTgtDevices[vI] == lDevs[vJ]) {
                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n", cBytes / (1024.0 * 1024.0),
                        ggml_backend_dev_name(lDevs[vJ]));
                vParamsBase.lFitParamsTarget[vI] += cBytes;
                break;
              }
            }
          }
          SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n", cHasDraft ? "draft model" : "MTP context",
                  vTotal / (1024.0 * 1024.0));
        } catch (const std::exception &vE) {
          SRV_WRN("[spec] failed to measure %s memory: %s\n", cHasDraft ? "draft model" : "MTP context", vE.what());
        }
      }
    }

    // An external MTP GGUF gets its actual weight allocations before the target,
    // including with --fit off. Its context is created once the target exists.
    mgs_model_ptr vPreloadedMtp;
    if (cHasDraft && cSpecMtp) {
      common_params dDraftParams = fCommonBaseParamsToSpeculative(vParamsBase);
      auto dModelParams = fCommonModelParamsToMgs(dDraftParams);
      dModelParams.progress_callback = mLoadProgressCallback;
      dModelParams.progress_callback_user_data = &vLoadProgressSpec;
      vPreloadedMtp.reset(mgs_model_load_from_file(dDraftParams.vModel.vPath.c_str(), dModelParams));
      if (!vPreloadedMtp) {
        SRV_ERR("failed to preload MTP model '%s'\n", dDraftParams.vModel.vPath.c_str());
        return false;
      }
    }

    // attach a progress callback
    {
      vParamsBase.vLoadProgressCallback = mLoadProgressCallback;
      vParamsBase.vLoadProgressCallbackUserData = &vLoadProgressText;
    }

    vMgsInit = fCommonInitFromParams(vParamsBase);

    vModelTgt = vMgsInit->mModel();
    vCtxTgt = vMgsInit->mContext();

    // The MoE expert cache takes the device memory left by every other allocation, so it waits
    // until the draft context and the multimodal projector are loaded (see the end of load_model).
    if (vCtxTgt != nullptr) {
      mgs_set_expert_cache_size(vCtxTgt, 0);
    }

    if (vModelTgt == nullptr) {
      SRV_ERR("failed to load model, '%s'\n", vParamsBase.vModel.vPath.c_str());
      return false;
    }

    if (vCtxTgt == nullptr) {
      SRV_ERR("failed to create_context with model '%s'\n", vParamsBase.vModel.vPath.c_str());
      return false;
    }

    vVocab = mgs_model_get_vocab(vModelTgt);

    vNCtx = mgs_n_ctx(vCtxTgt);

    vAddBosToken = mgs_vocab_get_add_bos(vVocab);

    if (cHasSpec) {
      // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
      mLoadProgressCallback(0.0f, &vLoadProgressSpec);
      vLoadProgressSpec.vTLastLoadProgressMs = 0; // reset so internal cbs aren't delayed

      {
        common_params vParamsDft = fCommonBaseParamsToSpeculative(vParamsBase);
        // The draft never holds more positions than the target, whose context --fit may have changed;
        // without this a context size of 0 gave the draft KV cache the training context of the model.
        vParamsDft.vNCtx = vNCtx;

        // progress callback
        vParamsDft.vLoadProgressCallback = mLoadProgressCallback;
        vParamsDft.vLoadProgressCallbackUserData = &vLoadProgressSpec;

        vSpecInit = fCommonSpeculativeInitFromParams(vParamsDft, vModelTgt, vCtxTgt, std::move(vPreloadedMtp));
        vModelDft = vSpecInit->mModel();
        vCtxDft = vSpecInit->mContext();

        if (cHasDraft && vModelDft == nullptr) {
          SRV_ERR("failed to load draft model, '%s'\n", vParamsDft.vModel.vPath.c_str());
          return false;
        }

        if (vCtxDft == nullptr) {
          SRV_ERR("%s", "failed to create MTP context\n");
          return false;
        }

        vParamsBase.vSpeculative.vDraft.vCtxTgt = vCtxTgt;
        vParamsBase.vSpeculative.vDraft.vCtxDft = vCtxDft;
      }

      mLoadProgressCallback(1.0f, &vLoadProgressSpec);
    }

    if (cHasMmproj) {
      if (vCallbackState) {
        vCallbackState(cServerStateLoading, {{"stage", "mmproj_model"}});
      }

      if (!cIsResume) {
        mtmd_helper_log_set(fCommonLogDefaultCallback, nullptr);
      }

      vMctx = mtmd_init_from_file(vMmprojPath.c_str(), vModelTgt, vMparams);
      if (vMctx == nullptr) {
        SRV_ERR("failed to load multimodal model, '%s'\n", vMmprojPath.c_str());
        return false;
      }
      SRV_INF("loaded multimodal model, '%s'\n", vMmprojPath.c_str());

      if (vParamsBase.vCtxShift) {
        vParamsBase.vCtxShift = false;
        SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
      }

      if (vParamsBase.vNCacheReuse) {
        vParamsBase.vNCacheReuse = 0;
        SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
      }
    }

    if (!mgs_memory_can_shift(mgs_get_memory(vCtxTgt))) {
      if (vParamsBase.vCtxShift) {
        vParamsBase.vCtxShift = false;
        SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
      }

      if (vParamsBase.vNCacheReuse) {
        vParamsBase.vNCacheReuse = 0;
        SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
      }
    }

    if (mgs_model_n_swa(vModelTgt) == 0) {
      if (vParamsBase.vSwaFull) {
        vParamsBase.vSwaFull = false;
        SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
      }
    }

    vNSwa = vParamsBase.vSwaFull ? 0 : mgs_model_n_swa(vModelTgt);

    // Necessary similarity of prompt for slot selection
    vSlotPromptSimilarity = vParamsBase.vSlotPromptSimilarity;

    const int cNCtxTrain = mgs_model_n_ctx_train(vModelTgt);

    int vNCtxSlot = mgs_n_ctx_seq(vCtxTgt);
    if (vNCtxSlot > cNCtxTrain) {
      SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", vNCtxSlot,
              cNCtxTrain);
      vNCtxSlot = cNCtxTrain;
    }

    lSlots.clear();

    vCtxTgtSeqRmType = fCommonContextCanSeqRm(vCtxTgt);
    if (vCtxTgtSeqRmType == cCommonContextSeqRmTypeNo) {
      SRV_WRN("%s", "speculative decoding not supported by this context\n");
    }

    if (vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull) {
      SRV_TRC("%s", "speculative decoding will use checkpoints\n");
    }

    // setup slots
    SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n", vParamsBase.vNParallel, vNCtxSlot,
            vParamsBase.vKvUnified ? "true" : "false");

    // initialize slots
    for (int vI = 0; vI < vParamsBase.vNParallel; vI++) {
      lSlots.emplace_back();
    }

    // try speculative decoding
    if (vCtxTgtSeqRmType != cCommonContextSeqRmTypeNo) {
      try {
        vSpec.reset(fCommonSpeculativeInit(vParamsBase.vSpeculative, vParamsBase.vNParallel));
      } catch (const std::exception &vE) {
        SRV_ERR("failed to initialize speculative decoding context: %s\n", vE.what());
      }
    }

    if (vCtxDft) {
      vCtxDftSeqRmType = fCommonContextCanSeqRm(vCtxDft);
    }

    if (vSpec) {
      SRV_TRC("%s", "speculative decoding context initialized\n");
    } else {
      vSpecInit.reset();
      vCtxDft = nullptr;
      vModelDft = nullptr;
    }

    for (int vI = 0; vI < vParamsBase.vNParallel; vI++) {
      server_slot &vSlot = lSlots[vI];

      vSlot.vId = vI;
      vSlot.vCtxTgt = vCtxTgt;
      vSlot.vCtxDft = vCtxDft;
      vSlot.vMem.mInit(vCtxTgt, vCtxDft);
      vSlot.vSpec = vSpec.get();
      vSlot.vNCtx = vNCtxSlot;

      vSlot.vMctx = vMctx;
      vSlot.vPrompt.vTokens.vHasMtmd = vMctx != nullptr;

      SLT_TRC(vSlot, "new slot, n_ctx = %d\n", vSlot.vNCtx);

      vSlot.vCallbackOnRelease = [this](int pIdSlot) { vQueueTasks.mPopDeferredTask(pIdSlot); };

      vSlot.mReset();
    }

    {
      const char *vMgsTrace = getenv("MGS_TRACE");
      vTrace = vMgsTrace ? atoi(vMgsTrace) : 0;

      if (vTrace) {
        SRV_WRN("MGS_TRACE = %d\n", vTrace);
      }
    }

    {
      const char *vMgsServerSlotsDebug = getenv("MGS_SERVER_SLOTS_DEBUG");
      vSlotsDebug = vMgsServerSlotsDebug ? atoi(vMgsServerSlotsDebug) : 0;

      if (vSlotsDebug) {
        SRV_WRN("MGS_SERVER_SLOTS_DEBUG = %d\n", vSlotsDebug);
      }
    }

    // the mUpdateSlots() logic will always submit a maximum of n_batch or n_parallel tokens
    // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not
    // used)
    {
      const int32_t cNBatch = mgs_n_batch(vCtxTgt);
      const int32_t cNEmbd = mgs_model_n_embd_inp(vModelTgt);
      vBatch.mInit(std::max(cNBatch, vParamsBase.vNParallel), cNEmbd);
    }

    if (vParamsBase.vCacheRamMib != 0) {
      if (vParamsBase.vCacheRamMib < 0) {
        SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
      } else {
        SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", vParamsBase.vCacheRamMib);
      }
      SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

      vPromptCache = std::make_unique<server_prompt_cache>(vParamsBase.vCacheRamMib, vNCtx);
    } else {
      SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
    }

    if (vParamsBase.vNCtxCheckpoints > 0) {
      SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n", vParamsBase.vNCtxCheckpoints,
              vParamsBase.vCheckpointMinStep);
    } else {
      SRV_TRC("%s", "context checkpoints disabled\n");
    }

    if (!vParamsBase.model_alias.empty()) {
      // the first alias is the model name
      vModelName = *vParamsBase.model_alias.begin();
    } else if (!vParamsBase.vModel.mGetName().empty()) {
      vModelName = vParamsBase.vModel.mGetName();
    } else {
      // fallback: derive model name from file name
      auto vModelPath = std::filesystem::path(vParamsBase.vModel.vPath);
      vModelName = vModelPath.filename().string();
    }

    sModelAliases = vParamsBase.model_alias;
    sModelTags = vParamsBase.sModelTags;

    // Everything that needs device memory is loaded: let the next request create the MoE expert cache.
    mgs_set_expert_cache_size(vCtxTgt, fCommonMoeCacheSize());

    dMonitorWeights = mgs_model_get_weight_breakdown(vModelTgt);
    vMonitorSpecMtp = cSpecMtp && vCtxDft != nullptr;
    mRefreshMonitorSnapshot(true);

    // propagate new defaults back to caller
    pParams = vParamsBase;

    if (!cIsResume) {
      return mInit();
    }

    if (vCallbackState) {
      vCallbackState(cServerStateReady, {});
    }

    return true;
  }

  // unlike mLoadModel(), this is only called once during initialization
  bool mInit() {
    GGML_ASSERT(vCtxTgt != nullptr);
    GGML_ASSERT(vModelTgt != nullptr);

    GGML_ASSERT(!vSleeping);

    // wiring up server queues
    vQueueTasks.mOnNewTask([this](server_task &&pTask) { mProcessSingleTask(std::move(pTask)); });
    vQueueTasks.mOnUpdateSlots([this]() {
      mUpdateSlots();
      mRefreshMonitorSnapshot(
          std::none_of(lSlots.begin(), lSlots.end(), [](const server_slot &pSlot) { return pSlot.mIsProcessing(); }));
    });
    vQueueTasks.mOnSleepingState([this](bool pSleeping) { mHandleSleepingState(pSleeping); });

    vMetrics.mInit();

    if (vParamsBase.vCacheIdleSlots) {
      if (vParamsBase.vCacheRamMib == 0) {
        SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
        vParamsBase.vCacheIdleSlots = false;
      } else {
        if (vParamsBase.vKvUnified) {
          SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
        } else {
          // without a unified KV cache, clearing a slot frees no reusable room, so we only
          // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
          SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
        }
        SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
      }
    }

    {
      const std::string &vCfg = vParamsBase.vUiConfigJson;
      if (!vCfg.empty()) {
        try {
          json vJsonSettings = json::parse(vCfg);
          vJsonUiSettings = vJsonSettings;
        } catch (const std::exception &vE) {
          SRV_ERR("%s: failed to parse UI config: %s\n", __func__, vE.what());
          return false;
        }
      }
    }

    // populate chat template params
    {
      common_chat_templates_ptr vChatTemplates;
      bool vEnableThinking = false;

      try {
        vChatTemplates = fCommonChatTemplatesInit(vModelTgt, vParamsBase.chat_template);

        SRV_TRC(
            "%s: chat template, example_format: '%s'\n", __func__,
            fCommonChatFormatExample(vChatTemplates.get(), vParamsBase.dDefaultTemplateKwargs).c_str());

        // thinking is enabled if:
        // 1. It's not explicitly disabled via --reasoning off
        // 2. The chat template supports it
        const bool cTemplateSupportsThinking = fCommonChatTemplatesSupportEnableThinking(vChatTemplates.get());
        vEnableThinking = vParamsBase.vEnableReasoning != 0 && cTemplateSupportsThinking;
        SRV_TRC("%s: chat template, thinking = %d\n", __func__, vEnableThinking);
      } catch (const std::exception &vE) {
        SRV_ERR("%s: chat template parsing error: %s\n", __func__, vE.what());
        SRV_ERR("%s: please use a custom chat template via --chat-template or --chat-template-file\n", __func__);
        SRV_ERR("%s: for example: --chat-template chatml\n", __func__);
        return false;
      }

      // IMPORTANT: chat_params is reused across sleeping / resuming states,
      //            never store mgs_context/mgs_model pointers in chat_params,
      //            as they may be invalidated after sleeping
      vChatParams = {/* prefill_assistant     */ vParamsBase.vPrefillAssistant,
                     /* reasoning_format      */ vParamsBase.vReasoningFormat,
                     /* chat_template_kwargs  */ vParamsBase.dDefaultTemplateKwargs,
                     /* tmpls                 */ std::move(vChatTemplates),
                     /* allow_image           */ vMctx ? mtmd_support_vision(vMctx) : false,
                     /* allow_audio           */ vMctx ? mtmd_support_audio(vMctx) : false,
                     /* allow_video           */ vMctx ? mtmd_helper_support_video(vMctx) : false,
                     /* enable_thinking       */ vEnableThinking,
                     /* reasoning_budget      */ vParamsBase.vSampling.vReasoningBudgetTokens,
                     /* reasoning_budget_msg  */ vParamsBase.vSampling.vReasoningBudgetMessage,
                     /* media_path            */ vParamsBase.vMediaPath,
                     /* force_pure_content    */ vParamsBase.vForcePureContentParser,
                     /* media_allow_private   */ vParamsBase.vMediaAllowPrivateUrls};

      {
        auto vCaps = fCommonChatTemplatesGetCaps(vChatParams.vTmpls.get());
        auto vIt = vParamsBase.dDefaultTemplateKwargs.find("preserve_reasoning");
        bool vSupported = vCaps.at("supports_preserve_reasoning");
        bool vEnabled = vIt != vParamsBase.dDefaultTemplateKwargs.end();
        if (vSupported && !vEnabled) {
          SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
        }
        if (!vSupported && vEnabled) {
          SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
        }
      }
    }

    return true;
  }

  server_slot *mGetSlotById(int pIdSlot) {
    // note: allow id_slot to be out of bounds (wrap around)
    pIdSlot = pIdSlot % lSlots.size();

    for (server_slot &vSlot : lSlots) {
      if (vSlot.vId == pIdSlot) {
        return &vSlot;
      }
    }

    return nullptr;
  }

  server_slot *mGetSlotByCmplId(const std::string &pCmplId) {
    if (pCmplId.empty()) {
      return nullptr;
    }

    for (server_slot &vSlot : lSlots) {
      if (vSlot.mIsProcessing() && vSlot.vTask && vSlot.vTask->vParams.vOaicompatCmplId == pCmplId) {
        return &vSlot;
      }
    }

    return nullptr;
  }

  server_slot *mGetAvailableSlot(const server_task &pTask) {
    server_slot *vRet = nullptr;

    bool vUpdateCache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (pTask.vIdSlot != -1) {
      vRet = mGetSlotById(pTask.vIdSlot);
      if (vRet) {
        SLT_INF(*vRet, "selected slot by id (%d)\n", pTask.vIdSlot);
      }
    }

    // find the slot that has at least n% prompt similarity
    if (vSlotPromptSimilarity != 0.0f) {
      float vFSimBest = 0;

      for (server_slot &vSlot : lSlots) {
        if (pTask.vIdSlot != -1 && vSlot.vId != pTask.vIdSlot) {
          continue;
        }

        // skip the slot if it is not available
        if (vSlot.mIsProcessing()) {
          SLT_TRC(vSlot, " - skipping, is_processing = %d\n", vSlot.mIsProcessing());
          continue;
        }

        const auto &vTokens = vSlot.vPrompt.vTokens;

        // skip the slot if it does not contains cached tokens
        if (vTokens.empty()) {
          SLT_TRC(vSlot, "%s", " - skipping, slot is empty\n");
          continue;
        }

        // fraction of the Longest Common Prefix length with respect to the input prompt length
        const size_t cLcpLen = vTokens.mGetCommonPrefix(pTask.vTokens);
        const float cFSimCur = float(cLcpLen) / pTask.vTokens.size();

        SLT_TRC(vSlot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", cFSimCur, cLcpLen, pTask.vTokens.size(),
                vSlotPromptSimilarity);

        // select the current slot if the criteria match
        if (cFSimCur > vFSimBest && cFSimCur > vSlotPromptSimilarity) {
          vFSimBest = cFSimCur;

          vRet = &vSlot;
        }
      }

      if (vRet != nullptr) {
        const float cFKeep = (vFSimBest * pTask.vTokens.size()) / vRet->vPrompt.vTokens.size();

        if (pTask.vIdSlot == -1) {
          SLT_INF(*vRet, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                  vFSimBest, vSlotPromptSimilarity, cFKeep);
        }

        // if we are about to lose a large portion of the existing context - save it in the prompt cache
        if (cFKeep < 0.5f) {
          vUpdateCache = true;
        }
      }
    }

    // find the slot that has been least recently used
    if (vRet == nullptr) {
      int64_t vTLast = -1;

      for (server_slot &vSlot : lSlots) {
        // skip the slot if it is not available
        if (vSlot.mIsProcessing()) {
          continue;
        }

        // select the current slot if the criteria match
        if (!vRet || vSlot.vTLastUsed <= vTLast) {
          vTLast = vSlot.vTLastUsed;
          vRet = &vSlot;
        }
      }

      if (vRet != nullptr) {
        SLT_INF(*vRet, "selected slot by LRU, t_last = %" PRId64 "\n", vTLast);

        vUpdateCache = true;
      }
    }

    if (vRet) {
      vUpdateCache = vUpdateCache && vPromptCache;

      // cache prompts only for completion tasks
      vUpdateCache = vUpdateCache && pTask.vType == cServerTaskTypeCompletion;

      if (vUpdateCache) {
        SRV_TRC("%s", "updating prompt cache\n");

        const int64_t cTStart = ggml_time_us();

        vRet->mPromptSave(*vPromptCache);

        if (!vRet->mPromptLoad(*vPromptCache, pTask.vTokens)) {
          vRet->mPromptClear();
        }

        vPromptCache->mUpdate();

        SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - cTStart) / 1000.0);
      }
    }

    return vRet;
  }

  // return true if at least one slot has been cleared
  // TODO: improve logic
  //       - smarter decision which slot to clear (LRU or longest prompt?)
  //       - move slot to level 2 cache instead of removing?
  //       - instead of purging, try to store and resume later?
  bool mTryClearIdleSlots() {
    bool vRes = false;

    if (!vParamsBase.vKvUnified) {
      return vRes;
    }

    for (auto &vSlot : lSlots) {
      if (vSlot.mIsProcessing()) {
        continue;
      }

      if (vSlot.vPrompt.mNTokens() > 0) {
        SRV_WRN("purging slot %d with %zu tokens\n", vSlot.vId, vSlot.vPrompt.vTokens.size());

        vSlot.mPromptClear();

        vRes = true;

        // clear slots one by one
        break;
      }
    }

    return vRes;
  }

  std::vector<common_adapter_lora_info> mConstructLoraList(const std::map<int, float> &pConfig) const {
    std::vector<common_adapter_lora_info> lOutput = vParamsBase.lLoraAdapters; // copy
    for (size_t vI = 0; vI < lOutput.size(); ++vI) {
      auto vIt = pConfig.find(vI);
      if (vIt != pConfig.end()) {
        lOutput[vI].vScale = vIt->second;
      } else {
        lOutput[vI].vScale = 0.0f;
      }
    }
    return lOutput;
  }

  bool mLaunchSlotWithTask(server_slot &pSlot, server_task &&pTask) {
    // process per-request lora adapters
    if (!pTask.vParams.dLora.empty()) {
      auto vTaskLoras = mConstructLoraList(pTask.vParams.dLora);
      if (!fAreLoraEqual(vTaskLoras, pSlot.lLora)) {
        // if lora has changed, check to see if the cache should be cleared
        if (fLoraShouldClearCache(pSlot.lLora, vTaskLoras)) {
          SLT_TRC(pSlot, "clearing cache for lora change. %zu loras -> %zu loras\n", pSlot.lLora.size(),
                  pTask.vParams.dLora.size());
          pSlot.vPrompt.mClear();
        } else {
          SLT_TRC(pSlot, "keeping cache for alora. %zu target loras\n", vTaskLoras.size());
        }
        pSlot.lLora = vTaskLoras;
      }
    } else {
      pSlot.lLora = vParamsBase.lLoraAdapters;
    }

    // if using alora, make sure it's only a single one requested and active
    size_t vAloraInvocationStart = pTask.vTokens.size();
    if (fLoraAllAlora(pSlot.lLora)) {
      const auto &vEnabledIds = fLoraGetEnabledIds(pSlot.lLora);
      // TODO: This will error out if a user requests two aloras, but only
      // provides the activation string for one. We could, instead search
      // for all requested alora activation strings and then either keep
      // only the last one, or reject if multiple are found.
      if (vEnabledIds.size() != 1) {
        mSendError(pTask, "Cannot run multiple aLoRAs in a single request", cErrorTypeInvalidRequest);
        return false;
      }
      const auto &vLora = pSlot.lLora[vEnabledIds[0]].vPtr;

      // get the pointer and count for the invocation tokens
      const uint64_t cNInvocationTokens = mgs_adapter_get_alora_n_invocation_tokens(vLora);
      const mgs_token *vInvocationTokens = mgs_adapter_get_alora_invocation_tokens(vLora);

      // scan backwards through the prompt tokens to find the last
      // occurrence of the invocation sequence
      int vMatchIdx = static_cast<int>(cNInvocationTokens) - 1;
      for (int vI = pTask.vTokens.size() - 1; vI >= 0; --vI) {
        // the token in this position matches the next token to find in
        // the invocation sequence
        if (pTask.vTokens[vI] == vInvocationTokens[vMatchIdx]) {
          // if it's a full match, we've found the start
          if (vMatchIdx == 0) {
            vAloraInvocationStart = vI;
            break;
          }
          // otherwise, check the next token in the sequence
          --vMatchIdx;
        } else {
          // no match in this position, so start looking over again
          vMatchIdx = static_cast<int>(cNInvocationTokens) - 1;
        }
      }

      // if the activation string is not found, disable the alora
      if (vAloraInvocationStart == pTask.vTokens.size()) {
        SLT_DBG(pSlot, "alora %zu requested, but not found. deactivating\n", vEnabledIds[0]);
        pSlot.lLora[vEnabledIds[0]].vScale = 0.0f;
      } else {
        SLT_DBG(pSlot, "alora %zu activated starting at %zu\n", vEnabledIds[0], vAloraInvocationStart);
        pSlot.vAloraInvocationStart = vAloraInvocationStart;
      }
    }

    if (!pTask.vTokens.mValidate(vCtxTgt)) {
      mSendError(pTask, "Prompt contains invalid tokens", cErrorTypeInvalidRequest);
      return false;
    }

    SLT_DBG(pSlot, "launching slot : %s\n", fSafeJsonToStr(pSlot.to_json()).c_str());

    // initialize samplers
    if (pTask.mNeedSampling()) {
      try {
        pSlot.vSmpl.reset(fCommonSamplerInit(vModelTgt, pTask.vParams.vSampling, (int32_t)mgs_n_ctx(vCtxTgt)));
      } catch (std::exception &vE) {
        std::string vErrMsg = std::string("Failed to initialize samplers: ") + vE.what();
        mSendError(pTask, vErrMsg, cErrorTypeInvalidRequest);
        return false;
      }

      const bool cNeedPreSampleLogits = pTask.vParams.vSampling.vNProbs > 0 && !pTask.vParams.vPostSamplingProbs;

      bool vBackendSampling = true;

      vBackendSampling &= pTask.vParams.vSampling.vBackendSampling;

      // TODO: speculative decoding requires multiple samples per batch - not supported yet
      vBackendSampling &= !(pSlot.mCanSpeculate());

      // TODO: getting pre sampling logits is not yet supported with backend sampling
      vBackendSampling &= !cNeedPreSampleLogits;

      // TODO: tmp until backend sampling is fully implemented
      if (vBackendSampling) {
        mgs_set_sampler(vCtxTgt, pSlot.vId, fCommonSamplerGet(pSlot.vSmpl.get()));
      } else {
        mgs_set_sampler(vCtxTgt, pSlot.vId, nullptr);
      }

      SLT_TRC(pSlot, "sampler chain: %s\n", fCommonSamplerPrint(pSlot.vSmpl.get()).c_str());
      SLT_TRC(pSlot, "sampler params: \n%s\n", pTask.vParams.vSampling.mPrint().c_str());
    } else {
      pSlot.vSmpl.reset();
    }

    pSlot.vTask = std::make_unique<const server_task>(std::move(pTask));

    pSlot.vState = pSlot.vTask->mIsChild() ? cSlotStateWaitOther // wait for the parent to process prompt
                                           : cSlotStateStarted;

    // reset server kill-switch counter
    vNEmptyConsecutive = 0;

    SLT_INF(pSlot, "processing task, is_child = %d\n", pSlot.vTask->mIsChild());
    return true;
  }

  bool mProcessToken(completion_token_output &pResult, server_slot &pSlot) {
    // remember which tokens were sampled - used for repetition penalties during sampling
    const std::string cTokenStr = pResult.vTextToSend;
    pSlot.vSampled = pResult.vTok;

    pSlot.vGeneratedText += cTokenStr;
    if (pSlot.vTask->vParams.vReturnTokens) {
      pSlot.vGeneratedTokens.push_back(pResult.vTok);
    }
    pSlot.vHasNextToken = true;

    // check if there is incomplete UTF-8 character at the end
    bool vIncomplete = fValidateUtf8(pSlot.vGeneratedText) < pSlot.vGeneratedText.size();

    // search stop word and delete it
    if (!vIncomplete) {
      size_t vPos = std::min(pSlot.vNSentText, pSlot.vGeneratedText.size());

      const std::string cStrTest = pSlot.vGeneratedText.substr(vPos);
      bool vSendText = true;

      size_t vStopPos = pSlot.mFindStoppingStrings(cStrTest, cTokenStr.size(), true);
      if (vStopPos != std::string::npos) {
        pSlot.vGeneratedText.erase(pSlot.vGeneratedText.begin() + vPos + vStopPos, pSlot.vGeneratedText.end());
        vPos = std::min(pSlot.vNSentText, pSlot.vGeneratedText.size());
      } else if (pSlot.vHasNextToken && !mgs_vocab_is_eog(vVocab, pResult.vTok)) {
        vStopPos = pSlot.mFindStoppingStrings(cStrTest, cTokenStr.size(), false);
        vSendText = vStopPos == std::string::npos;
      }

      // check if there is any token to predict
      if (vSendText) {
        // no send the stop word in the response
        pResult.vTextToSend = pSlot.vGeneratedText.substr(vPos, std::string::npos);
        pSlot.vNSentText += pResult.vTextToSend.size();
        // add the token to slot queue and cache
      } else {
        pResult.vTextToSend = "";
      }

      pSlot.mAddToken(pResult);
      if (pSlot.vTask->vParams.vStream) {
        mSendPartialResponse(pSlot, pResult, false);
      }
    }

    if (vIncomplete) {
      pSlot.vHasNextToken = true;
    }

    // if context shifting is disabled, make sure that we don't run out of context
    if (!vParamsBase.vCtxShift && pSlot.vPrompt.mNTokens() + 1 >= pSlot.vNCtx) {
      pSlot.vTruncated = true;
      pSlot.vStop = cStopTypeLimit;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot,
              "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = "
              "%d, n_ctx = %d\n",
              pSlot.vPrompt.mNTokens(), pSlot.vTask->mNTokens(), pSlot.vNDecoded, pSlot.vNCtx);
    }

    // check the limits
    if (pSlot.vNDecoded > 0 && pSlot.vHasNextToken && !pSlot.mHasBudget(vParamsBase)) {
      pSlot.vStop = cStopTypeLimit;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot, "stopped by limit, n_decoded = %d, n_predict = %d\n", pSlot.vNDecoded,
              pSlot.vTask->vParams.vNPredict);
    }

    if (pSlot.vHasNewLine) {
      // require that each new line has a whitespace prefix (i.e. indentation) of at least pSlot.vTask->vParams.vNIndent
      if (pSlot.vTask->vParams.vNIndent > 0) {
        // check the current indentation
        // TODO: improve by not doing it more than once for each new line
        if (pSlot.vLastNlPos > 0) {
          size_t vPos = pSlot.vLastNlPos;

          int vNIndent = 0;
          while (vPos < pSlot.vGeneratedText.size() &&
                 (pSlot.vGeneratedText[vPos] == ' ' || pSlot.vGeneratedText[vPos] == '\t')) {
            vNIndent++;
            vPos++;
          }

          if (vPos < pSlot.vGeneratedText.size() && vNIndent < pSlot.vTask->vParams.vNIndent) {
            pSlot.vStop = cStopTypeLimit;
            pSlot.vHasNextToken = false;

            // cut the last line
            pSlot.vGeneratedText.erase(vPos, std::string::npos);

            SLT_DBG(pSlot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", pSlot.vNDecoded, vNIndent);
          }
        }

        // find the next new line
        {
          const size_t cPos = pSlot.vGeneratedText.find('\n', pSlot.vLastNlPos);

          if (cPos != std::string::npos) {
            pSlot.vLastNlPos = cPos + 1;
          }
        }
      }
    }

    // check if there is a new line in the generated text
    if (pResult.vTextToSend.find('\n') != std::string::npos) {
      pSlot.vHasNewLine = true;

      // if we have seen a new line, we stop after a certain time limit, but only upon another new line
      if (pSlot.vTask->vParams.vTMaxPredictMs > 0 &&
          (ggml_time_us() - pSlot.vTStartGeneration > 1000.0f * pSlot.vTask->vParams.vTMaxPredictMs)) {
        pSlot.vStop = cStopTypeLimit;
        pSlot.vHasNextToken = false;

        SLT_DBG(pSlot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", pSlot.vNDecoded,
                (int)pSlot.vTask->vParams.vTMaxPredictMs);
      }
    }

    if (mgs_vocab_is_eog(vVocab, pResult.vTok)) {
      pSlot.vStop = cStopTypeEos;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot, "%s", "stopped by EOS\n");
    }

    SLT_DBG(pSlot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", pSlot.vNDecoded, pSlot.vNRemaining,
            pResult.vTok, cTokenStr.c_str());

    return pSlot.vHasNextToken; // continue
  }

  void mPopulateTokenProbs(const server_slot &pSlot, completion_token_output &pResult, bool pPostSampling,
                           bool pSpecial, int pIdx) const {
    const size_t cNProbsRequest = pSlot.vTask->vParams.vSampling.vNProbs;

    if (pPostSampling) {
      const auto *vCurP = fCommonSamplerGetCandidates(pSlot.vSmpl.get(), true);
      const size_t cMaxProbs = vCurP->size;
      const size_t cNProbs = std::min(cMaxProbs, cNProbsRequest);

      // set probability for sampled token
      for (size_t vI = 0; vI < cMaxProbs; vI++) {
        if (vCurP->data[vI].id == pResult.vTok) {
          pResult.vProb = vCurP->data[vI].p;
          break;
        }
      }

      // set probability for top n_probs tokens
      pResult.lProbs.reserve(cNProbs);
      for (size_t vI = 0; vI < cNProbs; vI++) {
        // Some samplers do return 0.0 probabilities, others don't.
        // Filter 0.0 probailities, to ensure the behavior is consistent.
        if (vCurP->data[vI].p == 0.0) {
          break;
        }

        pResult.lProbs.push_back(
            {vCurP->data[vI].id, fCommonTokenToPiece(vCtxTgt, vCurP->data[vI].id, pSpecial), vCurP->data[vI].p});
      }
    } else {
      std::vector<mgs_token_data> lCur = fGetTokenProbabilities(vCtxTgt, pIdx, cNProbsRequest);
      const size_t cMaxProbs = lCur.size();
      const size_t cNProbs = std::min(cMaxProbs, cNProbsRequest);

      // set probability for sampled token
      for (size_t vI = 0; vI < cMaxProbs; vI++) {
        // set probability for sampled token
        if (lCur[vI].id == pResult.vTok) {
          pResult.vProb = lCur[vI].p;
          break;
        }
      }

      // set probability for top n_probs tokens
      pResult.lProbs.reserve(cNProbs);
      for (size_t vI = 0; vI < cNProbs; vI++) {
        pResult.lProbs.push_back({lCur[vI].id, fCommonTokenToPiece(vCtxTgt, lCur[vI].id, pSpecial), lCur[vI].p});
      }
    }
  }

  void mSendError(const server_task &pTask, const std::string &pError, const enum error_type pType = cErrorTypeServer) {
    mSendError(pTask.vId, pError, pType);
  }

  void mSendError(const server_slot &pSlot, const std::string &pError, const enum error_type pType = cErrorTypeServer) {
    mSendError(pSlot.vTask->vId, pError, pType, pSlot.vTask->mNTokens(), pSlot.vNCtx);
  }

  void mSendError(const int pIdTask, const std::string &pError, const enum error_type pType = cErrorTypeServer,
                  const int32_t pNPromptTokens = 0, const int32_t pNCtx = 0) {
    SRV_ERR("task id = %d, error: %s\n", pIdTask, pError.c_str());

    if (pType == cErrorTypeExceedContextSize) {
      GGML_ASSERT(pNCtx > 0 && pNPromptTokens > 0);
    }

    auto vRes = std::make_unique<server_task_result_error>();
    vRes->vId = pIdTask;
    vRes->vErrType = pType;
    vRes->vErrMsg = pError;
    vRes->vNPromptTokens = pNPromptTokens;
    vRes->vNCtx = pNCtx;

    vQueueResults.mSend(std::move(vRes));
  }

  // Gate slot save/restore/erase on slot content (does it hold media),
  // not model capability: a multimodal model may hold a pure-text slot.
  bool mCheckSlotNoMedia(const server_slot &pSlot, const int pIdTask) {
    if (pSlot.vPrompt.vTokens.mHasMedia()) {
      mSendError(
          pIdTask,
          "This operation is not supported while the slot holds image/audio tokens (a pure-text prefix is supported)",
          cErrorTypeNotSupported);
      return false;
    }
    return true;
  }

  void mSendPartialResponse(server_slot &pSlot, const completion_token_output &pTkn, bool pIsProgress,
                            bool pIsBegin = false) {
    auto vRes = std::make_unique<server_task_result_cmpl_partial>();

    vRes->vId = pSlot.vTask->vId;
    vRes->vIndex = pSlot.vTask->vIndex;

    if (pIsProgress) {
      vRes->vIsProgress = true;
      vRes->vProgress.vTotal = pSlot.vTask->mNTokens();
      vRes->vProgress.vCache = pSlot.vNPromptTokensCache;
      vRes->vProgress.vProcessed = pSlot.vPrompt.vTokens.size();
      vRes->vProgress.vTimeMs = (ggml_time_us() - pSlot.vTStartProcessPrompt) / 1000;
    }
    if (pIsBegin) {
      vRes->vIsBegin = true;
    } else {
      vRes->vContent = pTkn.vTextToSend;
      vRes->vTokens = {pTkn.vTok};
    }

    vRes->vNDecoded = pSlot.vNDecoded;
    vRes->vNPromptTokens = pSlot.vTask->mNTokens();
    vRes->vNPromptTokensCache = pSlot.vNPromptTokensCache;
    vRes->vPostSamplingProbs = pSlot.vTask->vParams.vPostSamplingProbs;

    vRes->vVerbose = pSlot.vTask->vParams.vVerbose;
    vRes->vResType = pSlot.vTask->vParams.vResType;
    vRes->vOaicompatModel = pSlot.vTask->vParams.vOaicompatModel;
    vRes->vOaicompatCmplId = pSlot.vTask->vParams.vOaicompatCmplId;

    // populate vRes->vProbOutput
    if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
      vRes->vProbOutput = pTkn; // copy the token probs
    }

    // populate timings if this is final response or timings_per_token is enabled
    if (pSlot.vStop != cStopTypeNone || pSlot.vTask->vParams.vTimingsPerToken) {
      vRes->vTimings = pSlot.mGetTimings();
    }

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendFinalResponse(server_slot &pSlot) {
    auto vRes = std::make_unique<server_task_result_cmpl_final>();

    vRes->vId = pSlot.vTask->vId;
    vRes->vIdSlot = pSlot.vId;

    vRes->vIndex = pSlot.vTask->vIndex;

    // keep copy of last generated text for debugging purposes
    if (vSlotsDebug) {
      pSlot.vDebugGeneratedText = pSlot.vGeneratedText;
    }

    // in stream mode, content and tokens are already in last partial chunk
    if (pSlot.vTask->vParams.vStream) {
      vRes->vContent = "";
      vRes->vTokens = mgs_tokens{};
    } else {
      vRes->vContent = std::move(pSlot.vGeneratedText);
      vRes->vTokens = std::move(pSlot.vGeneratedTokens);
    }
    vRes->vTimings = pSlot.mGetTimings();
    vRes->vPrompt = pSlot.vTask->vTokens.mDetokenize(vCtxTgt, true);
    vRes->lResponseFields = std::move(pSlot.vTask->vParams.lResponseFields);

    vRes->vTruncated = pSlot.vTruncated;
    vRes->vNDecoded = pSlot.vNDecoded;
    vRes->vNPromptTokens = pSlot.vTask->mNTokens();
    vRes->vNPromptTokensCache = pSlot.vNPromptTokensCache;
    vRes->vNTokensCached = pSlot.vPrompt.mNTokens();
    vRes->vHasNewLine = pSlot.vHasNewLine;
    vRes->vStoppingWord = pSlot.vStoppingWord;
    vRes->vStop = pSlot.vStop;
    vRes->vPostSamplingProbs = pSlot.vTask->vParams.vPostSamplingProbs;

    vRes->vVerbose = pSlot.vTask->vParams.vVerbose;
    vRes->vStream = pSlot.vTask->vParams.vStream;
    vRes->vIncludeUsage = pSlot.vTask->vParams.vIncludeUsage;
    vRes->vResType = pSlot.vTask->vParams.vResType;
    vRes->vOaicompatModel = pSlot.vTask->vParams.vOaicompatModel;
    vRes->vOaicompatCmplId = pSlot.vTask->vParams.vOaicompatCmplId;

    // populate vRes->lProbsOutput
    if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
      if (!pSlot.vTask->vParams.vStream && pSlot.vStop == cStopTypeWord) {
        const mgs_tokens cStopWordToks = fCommonTokenize(vCtxTgt, pSlot.vStoppingWord, false);

        size_t vSafeOffset = std::min(pSlot.lGeneratedTokenProbs.size(), cStopWordToks.size());
        vRes->lProbsOutput = std::vector<completion_token_output>(pSlot.lGeneratedTokenProbs.begin(),
                                                                  pSlot.lGeneratedTokenProbs.end() - vSafeOffset);
      } else {
        vRes->lProbsOutput =
            std::vector<completion_token_output>(pSlot.lGeneratedTokenProbs.begin(), pSlot.lGeneratedTokenProbs.end());
      }
    }

    vRes->vGenerationParams = pSlot.vTask->vParams; // copy the parameters

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendEmbedding(const server_slot &pSlot, const mgs_batch &pBatch) {
    auto vRes = std::make_unique<server_task_result_embd>();
    vRes->vId = pSlot.vTask->vId;
    vRes->vIndex = pSlot.vTask->vIndex;
    vRes->vNTokens = pSlot.vTask->mNTokens();
    vRes->vResType = pSlot.vTask->vParams.vResType;

    const int cNEmbdOut = mgs_model_n_embd_out(vModelTgt);

    std::vector<float> lEmbdRes(cNEmbdOut, 0.0f);

    for (int vI = 0; vI < pBatch.n_tokens; ++vI) {
      if (!pBatch.logits[vI] || pBatch.seq_id[vI][0] != pSlot.vId) {
        continue;
      }

      const float *vEmbd = nullptr;
      if (mgs_pooling_type(pSlot.vCtxTgt) == MGS_POOLING_TYPE_NONE) {
        vEmbd = mgs_get_embeddings_ith(pSlot.vCtxTgt, vI);
      } else {
        vEmbd = mgs_get_embeddings_seq(pSlot.vCtxTgt, pBatch.seq_id[vI][0]);
      }

      if (vEmbd == nullptr) {
        SLT_ERR(pSlot, "failed to get embeddings, token = %d, seq_id = %d\n", pBatch.token[vI], pBatch.seq_id[vI][0]);

        vRes->lEmbedding.push_back(std::vector<float>(cNEmbdOut, 0.0f));
        continue;
      }

      // normalize only when there is pooling
      if (mgs_pooling_type(pSlot.vCtxTgt) != MGS_POOLING_TYPE_NONE) {
        fCommonEmbdNormalize(vEmbd, lEmbdRes.data(), cNEmbdOut, pSlot.vTask->vParams.vEmbdNormalize);
        vRes->lEmbedding.push_back(lEmbdRes);
        break;
      }

      vRes->lEmbedding.emplace_back(vEmbd, vEmbd + cNEmbdOut);
    }

    SLT_DBG(pSlot, "%s", "sending embeddings\n");

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendRerank(const server_slot &pSlot, const mgs_batch &pBatch) {
    auto vRes = std::make_unique<server_task_result_rerank>();
    vRes->vId = pSlot.vTask->vId;
    vRes->vIndex = pSlot.vTask->vIndex;
    vRes->vNTokens = pSlot.vTask->mNTokens();

    for (int vI = 0; vI < pBatch.n_tokens; ++vI) {
      if (!pBatch.logits[vI] || pBatch.seq_id[vI][0] != pSlot.vId) {
        continue;
      }

      const float *vEmbd = mgs_get_embeddings_seq(vCtxTgt, pBatch.seq_id[vI][0]);
      if (vEmbd == NULL) {
        vEmbd = mgs_get_embeddings_ith(vCtxTgt, vI);
      }

      if (vEmbd == NULL) {
        SLT_ERR(pSlot, "failed to get embeddings, token = %d, seq_id = %d\n", pBatch.token[vI], pBatch.seq_id[vI][0]);

        vRes->vScore = -1e6;
        continue;
      }

      vRes->vScore = vEmbd[0];
    }

    SLT_DBG(pSlot, "sending rerank result, res.score = %f\n", vRes->vScore);

    vQueueResults.mSend(std::move(vRes));
  }

  //
  // Functions to process the task
  //

  // tokenize the input if it's set by CLI, return false on error
  bool mTokenizeCliInput(server_task &pTask) {
    try {
      auto &vPrompt = pTask.vCliPrompt;
      if (vMctx != nullptr) {
        pTask.vTokens = fProcessMtmdPrompt(vMctx, vPrompt, pTask.lCliFiles);
      } else {
        pTask.vTokens = std::move(fTokenizeInputPrompts(vVocab, vMctx, vPrompt, true, true)[0]);
      }
      pTask.vCliPrompt.clear();
      pTask.lCliFiles.clear();
    } catch (const std::exception &vE) {
      mSendError(pTask, std::string("Failed to format input: ") + vE.what(), cErrorTypeInvalidRequest);
      return false;
    }
    return true;
  }

  std::vector<server_slot *> mGetFreeSlots(size_t pNSlotsNeeded, int pExcludeIdSlot) {
    std::vector<server_slot *> lFreeSlots;
    for (auto &vSlot : lSlots) {
      if (!vSlot.mIsProcessing() && vSlot.vId != pExcludeIdSlot) {
        lFreeSlots.push_back(&vSlot);
      }
      if (lFreeSlots.size() >= pNSlotsNeeded) {
        break;
      }
    }
    return lFreeSlots;
  }

  // launch multiple slots for parent + child tasks
  bool mLaunchSlotsWithParentTask(server_slot &pParentSlot, std::vector<server_slot *> &pChildSlots,
                                  server_task &&pParentTask) {
    GGML_ASSERT(!pParentSlot.mIsProcessing());
    GGML_ASSERT(pParentTask.mIsParent());
    GGML_ASSERT(pChildSlots.size() == pParentTask.lChildTasks.size());

    int vIdParent = pParentTask.vId;

    SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", vIdParent,
            pParentTask.lChildTasks.size());

    // to be called in case of failure to release all launched slots
    auto vReleaseSlots = [this, vIdParent]() {
      for (auto &vSlot : lSlots) {
        if (vSlot.mIsProcessing() && (vSlot.vTask->vId == vIdParent || vSlot.vTask->vIdParent == vIdParent)) {
          vSlot.mRelease();
        }
      }
    };

    // launch all child tasks first
    size_t vIdx = 0;
    GGML_ASSERT(pChildSlots.size() == pParentTask.lChildTasks.size());
    for (auto *vSlot : pChildSlots) {
      int vIdChild = pParentTask.lChildTasks[vIdx].vId;
      if (!mLaunchSlotWithTask(*vSlot, std::move(pParentTask.lChildTasks[vIdx]))) {
        SRV_ERR("failed to launch slot with child task, id_task = %d\n", vIdChild);
        vReleaseSlots();
        return false;
      }
      vIdx++;
    }

    // finally, launch the parent task
    if (!mLaunchSlotWithTask(pParentSlot, std::move(pParentTask))) {
      SRV_ERR("failed to launch slot with task, id_task = %d\n", vIdParent);
      vReleaseSlots();
      return false;
    }

    return true;
  }

  // n_tokens_cur: the number of tokens added to the batch for the current slot
  void mCreateCheckpoint(server_slot &pSlot, const int64_t pNTokensCur, mgs_pos pPosMin, mgs_pos pPosMax) {
    const int cIdTask = pSlot.vTask->vId;

    // evict checkpoints within min-step of a previous checkpoint, unless they were
    // created by the current task
    int64_t vLast = -1;
    for (auto vIt = pSlot.vPrompt.lCheckpoints.begin(); vIt != pSlot.vPrompt.lCheckpoints.end();) {
      if (vIt->vIdTask != cIdTask && vLast >= 0 && vIt->vNTokens <= vLast + vParamsBase.vCheckpointMinStep) {
        SLT_TRC(
            pSlot,
            "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
            ", size = %.3f MiB)\n",
            vIt->vPosMin, vIt->vPosMax, vIt->vNTokens, (float)vIt->size() / 1024 / 1024);

        vIt = pSlot.vPrompt.lCheckpoints.erase(vIt);
        continue;
      }

      vLast = vIt->vNTokens;
      ++vIt;
    }

    while (pSlot.vPrompt.lCheckpoints.size() >= (size_t)vParamsBase.vNCtxCheckpoints) {
      // Make room for the new checkpoint by removing the one whose neighbours are closest,
      // so the remaining checkpoints stay spread over the whole context (agents usually edit
      // old parts of it) instead of dropping the oldest ones first. The newest one is kept.
      auto vVictim = pSlot.vPrompt.lCheckpoints.begin();
      int64_t vSmallestGap = INT64_MAX;
      int64_t vPrevious = 0;
      for (auto vIt = pSlot.vPrompt.lCheckpoints.begin(); vIt != pSlot.vPrompt.lCheckpoints.end(); ++vIt) {
        auto vNext = std::next(vIt);
        if (vNext == pSlot.vPrompt.lCheckpoints.end()) {
          break;
        }
        const int64_t vGap = vNext->vNTokens - vPrevious;
        if (vGap < vSmallestGap) {
          vSmallestGap = vGap;
          vVictim = vIt;
        }
        vPrevious = vIt->vNTokens;
      }

      SLT_WRN(pSlot,
              "erasing context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
              vVictim->vPosMin, vVictim->vPosMax, vVictim->vNTokens, (float)vVictim->size() / 1024 / 1024);

      pSlot.vPrompt.lCheckpoints.erase(vVictim);
    }

    auto &vCur = pSlot.vPrompt.lCheckpoints.emplace_back();

    vCur.vIdTask = cIdTask;

    // [TAG_CHECKPOINTS_FIX_POS_MIN]
    // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
    //       this is not true for SWA models
    vCur.mUpdatePos(pSlot.vPrompt.mNTokens() - pNTokensCur, pPosMin, pPosMax);

    vCur.mUpdateTgt(vCtxTgt, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    vCur.mUpdateDft(vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    // stash the draft's speculative state with the checkpoint
    fCommonSpeculativeGetState(vSpec.get(), pSlot.vId, vCur.lDataSpec);

    SLT_TRC(pSlot,
            "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
            ", size = %.3f MiB)\n",
            (int)pSlot.vPrompt.lCheckpoints.size(), vParamsBase.vNCtxCheckpoints, vCur.vPosMin, vCur.vPosMax,
            vCur.vNTokens, (float)vCur.size() / 1024 / 1024);
  }

  //
  // Queue status notices: a streamed Web UI request that waits for a free slot learns how many requests run or wait
  // ahead of it, and whether they were sent directly to the API or by the Web UI
  //

  void mSendQueueStatus(int pIdTask, size_t pIndex, int32_t pAheadApi, int32_t pAheadWebui) {
    auto vRes = std::make_unique<server_task_result_queue>();
    vRes->vId = pIdTask;
    vRes->vIndex = pIndex;
    vRes->vAheadApi = pAheadApi;
    vRes->vAheadWebui = pAheadWebui;
    vRes->vAhead = pAheadApi + pAheadWebui;
    vQueueResults.mSend(std::move(vRes));
  }

  // requests that run now, by origin
  void mCountRunning(int32_t &pApi, int32_t &pWebui) const {
    pApi = 0;
    pWebui = 0;
    for (const auto &vSlot : lSlots) {
      if (vSlot.mIsProcessing() && vSlot.vTask) {
        (vSlot.vTask->vParams.vFromWebui ? pWebui : pApi)++;
      }
    }
  }

  // tells every waiting task that asked for it how many requests run or wait ahead of it
  void mNotifyQueuePositions() {
    int32_t vAheadApi = 0;
    int32_t vAheadWebui = 0;
    mCountRunning(vAheadApi, vAheadWebui);
    for (const server_queue_waiting &vWaiting : vQueueTasks.mDeferredTasks()) {
      if (vWaiting.vQueueNotice) {
        mSendQueueStatus(vWaiting.vId, vWaiting.vIndex, vAheadApi, vAheadWebui);
      }
      (vWaiting.vFromWebui ? vAheadWebui : vAheadApi)++;
    }
  }

  // queues a task that found no free slot; the queue runs in arrival order (see server_queue::mPopDeferredTask)
  void mDeferTask(server_task &&pTask) {
    const bool cRequeued = pTask.vWasQueued;
    const bool cQueueNotice = pTask.vParams.vQueueNotice;
    const int cIdTask = pTask.vId;
    const size_t cIndex = pTask.vIndex;

    int32_t vAheadApi = 0;
    int32_t vAheadWebui = 0;
    mCountRunning(vAheadApi, vAheadWebui);
    for (const server_queue_waiting &vWaiting : vQueueTasks.mDeferredTasks()) {
      (vWaiting.vFromWebui ? vAheadWebui : vAheadApi)++;
    }

    pTask.vWasQueued = true;
    vQueueTasks.mDefer(std::move(pTask));

    if (cRequeued) {
      mNotifyQueuePositions();
    } else if (cQueueNotice) {
      mSendQueueStatus(cIdTask, cIndex, vAheadApi, vAheadWebui);
    }
  }

  void mProcessSingleTask(server_task &&pTask) {
    switch (pTask.vType) {
    case cServerTaskTypeCompletion:
    case cServerTaskTypeInfill:
    case cServerTaskTypeEmbedding:
    case cServerTaskTypeRerank: {
      // special case: if input is provided via CLI, tokenize it first
      // otherwise, no need to tokenize as it's already done inside the HTTP thread
      if (pTask.vCli) {
        if (!mTokenizeCliInput(pTask)) {
          break;
        }
      }

      const int cIdTask = pTask.vId;
      const size_t cIndex = pTask.vIndex;
      const bool cWasQueued = pTask.vWasQueued;
      const bool cQueueNotice = pTask.vParams.vQueueNotice;

      server_slot *vSlot = mGetAvailableSlot(pTask);

      //
      // slot scheduling logic
      //

      if (vSlot == nullptr) {
        // if no slot is available, we defer this task for processing later
        SRV_DBG("no slot is available, defer task, id_task = %d\n", cIdTask);
        mDeferTask(std::move(pTask));
        break;
      }

      if (vSlot->mIsProcessing()) {
        // if requested slot is unavailable, we defer this task for processing later
        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", cIdTask);
        mDeferTask(std::move(pTask));
        break;
      }

      if (pTask.mIsParent()) {
        // try getting free slots for all child tasks
        size_t vNChildTasks = pTask.lChildTasks.size();
        std::vector<server_slot *> lChildSlots = mGetFreeSlots(vNChildTasks, vSlot->vId);
        if (lChildSlots.size() < vNChildTasks) {
          SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n",
                  lChildSlots.size(), vNChildTasks, cIdTask);
          mDeferTask(std::move(pTask));
          break;
        }
        if (!mLaunchSlotsWithParentTask(*vSlot, lChildSlots, std::move(pTask))) {
          SRV_ERR("failed to launch slot with parent task, id_task = %d\n", cIdTask);
          break; // drop the task
        }
      } else if (!mLaunchSlotWithTask(*vSlot, std::move(pTask))) {
        SRV_ERR("failed to launch slot with task, id_task = %d\n", cIdTask);
        break; // drop the task
      }

      // a task that waited starts now: tell it so, and move up the ones still waiting
      if (cWasQueued) {
        if (cQueueNotice) {
          mSendQueueStatus(cIdTask, cIndex, 0, 0);
        }
        mNotifyQueuePositions();
      }

      if (vParamsBase.vCacheIdleSlots) {
        for (auto &vSlot : lSlots) {
          if (!vSlot.mIsProcessing()) {
            SLT_TRC(vSlot, "%s", "saving idle slot to prompt cache\n");

            if (vSlot.mPromptSave(*vPromptCache)) {
              SLT_DBG(vSlot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
              vPromptCache->mUpdate();
            }

            if (vParamsBase.vKvUnified) {
              // [TAG_IDLE_SLOT_CLEAR]
              vSlot.mPromptClear();
            }
          }
        }
      }
    } break;
    case cServerTaskTypeCancel: {
      // release slot linked with the task id
      for (auto &vSlot : lSlots) {
        if (vSlot.vTask && vSlot.vTask->vId == pTask.vIdTarget) {
          vSlot.mRelease();
          break;
        }
      }
    } break;
    case cServerTaskTypeControl: {
      auto vRes = std::make_unique<server_task_result_control>();
      vRes->vId = pTask.vId;

      server_slot *vSlot = mGetSlotByCmplId(pTask.vParams.vControlCmplId);
      if (vSlot == nullptr) {
        SRV_WRN("control %s on unknown completion id=%s, no live slot\n", pTask.vParams.vControlAction.c_str(),
                pTask.vParams.vControlCmplId.c_str());
        vRes->vSuccess = false;
        vRes->vMessage = "no active completion for this id";
        vQueueResults.mSend(std::move(vRes));
        break;
      }

      if (pTask.vParams.vControlAction == "reasoning_end") {
        // the budget sampler only exists when reasoning control was armed
        if (!vSlot->vTask->vParams.vSampling.vReasoningControl) {
          vRes->vSuccess = false;
          vRes->vMessage = "reasoning control not enabled for this completion";
          vQueueResults.mSend(std::move(vRes));
          break;
        }
        // act on the live slot mid generation, never defer
        fCommonSamplerReasoningBudgetForce(vSlot->vSmpl.get());
        vRes->vSuccess = true;
      } else {
        vRes->vSuccess = false;
        vRes->vMessage = "unknown control action";
      }

      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeNextResponse: {
      // do nothing
    } break;
    case cServerTaskTypeMetrics: {
      json vSlotsData = json::array();

      int vNIdleSlots = 0;
      int vNProcessingSlots = 0;

      for (server_slot &vSlot : lSlots) {
        json vSlotData = vSlot.to_json(vSlotsDebug == 0);

        if (vSlot.mIsProcessing()) {
          vNProcessingSlots++;
        } else {
          vNIdleSlots++;
        }

        vSlotsData.push_back(vSlotData);
      }
      SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", vNIdleSlots, vNProcessingSlots);

      auto vRes = std::make_unique<server_task_result_metrics>();
      vRes->vId = pTask.vId;
      vRes->vSlotsData = std::move(vSlotsData);
      vRes->vNIdleSlots = vNIdleSlots;
      vRes->vNProcessingSlots = vNProcessingSlots;
      vRes->vNTasksDeferred = vQueueTasks.mQueueTasksDeferredSize();
      vRes->vTStart = vMetrics.vTStart;

      vRes->vNPromptTokensProcessedTotal = vMetrics.vNPromptTokensProcessedTotal;
      vRes->vTPromptProcessingTotal = vMetrics.vTPromptProcessingTotal;
      vRes->vNTokensPredictedTotal = vMetrics.vNTokensPredictedTotal;
      vRes->vTTokensGenerationTotal = vMetrics.vTTokensGenerationTotal;

      vRes->vNTokensMax = vMetrics.vNTokensMax;

      vRes->vNPromptTokensProcessed = vMetrics.vNPromptTokensProcessed;
      vRes->vTPromptProcessing = vMetrics.vTPromptProcessing;
      vRes->vNTokensPredicted = vMetrics.vNTokensPredicted;
      vRes->vTTokensGeneration = vMetrics.vTTokensGeneration;

      vRes->vNDecodeTotal = vMetrics.vNDecodeTotal;
      vRes->vNBusySlotsTotal = vMetrics.vNBusySlotsTotal;

      vRes->vNDraftTokensTotal = vMetrics.vNDraftTokensTotal;
      vRes->vNDraftAcceptedTotal = vMetrics.vNDraftAcceptedTotal;
      vRes->vNDraftVerifStepsTotal = vMetrics.vNDraftVerifStepsTotal;
      vRes->lNAcceptedPerPosTotal = vMetrics.lNAcceptedPerPosTotal;

      vRes->vExpertCache = fExpertCacheJson(vCtxTgt);

      if (pTask.vMetricsResetBucket) {
        vMetrics.mResetBucket();
      }
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeExpertCache: {
      if (pTask.vExpertCacheSizeMib != -2) {
        mgs_set_expert_cache_size(vCtxTgt, pTask.vExpertCacheSizeMib);
        SRV_INF("MoE expert cache size requested: %" PRId64 " MiB, applied by the next request\n",
                pTask.vExpertCacheSizeMib);
      }
      auto vRes = std::make_unique<server_task_result_expert_cache>();
      vRes->vId = pTask.vId;
      vRes->vInfo = fExpertCacheJson(vCtxTgt);
      if (pTask.vExpertCacheSizeMib != -2) {
        vRes->vInfo["requested_size_mib"] = pTask.vExpertCacheSizeMib;
      }
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeSlotSave: {
      const int cIdSlot = pTask.vSlotAction.vIdSlot;
      server_slot *vSlot = mGetSlotById(cIdSlot);
      if (vSlot == nullptr) {
        mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
        break;
      }
      if (!mCheckSlotNoMedia(*vSlot, pTask.vId)) {
        break;
      }
      if (vSlot->mIsProcessing()) {
        // if requested slot is unavailable, we defer this task for processing later
        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
        vQueueTasks.mDefer(std::move(pTask));
        break;
      }

      const int64_t cTStart = ggml_time_us();

      std::string vFilename = pTask.vSlotAction.vFilename;
      std::string vFilepath = pTask.vSlotAction.vFilepath;

      const mgs_tokens cTokens = vSlot->vPrompt.vTokens.mGetTextTokens();
      const size_t cTokenCount = cTokens.size();
      const size_t cNwrite =
          mgs_state_seq_save_file(vCtxTgt, vFilepath.c_str(), vSlot->vId, cTokens.data(), cTokenCount);

      const int64_t cTEnd = ggml_time_us();
      const double cTSaveMs = (cTEnd - cTStart) / 1000.0;

      auto vRes = std::make_unique<server_task_result_slot_save_load>();
      vRes->vId = pTask.vId;
      vRes->vIdSlot = cIdSlot;
      vRes->vFilename = vFilename;
      vRes->vIsSave = true;
      vRes->vNTokens = cTokenCount;
      vRes->vNBytes = cNwrite;
      vRes->vTMs = cTSaveMs;
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeSlotRestore: {
      const int cIdSlot = pTask.vSlotAction.vIdSlot;
      server_slot *vSlot = mGetSlotById(cIdSlot);
      if (vSlot == nullptr) {
        mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
        break;
      }
      if (vSlot->mIsProcessing()) {
        // if requested slot is unavailable, we defer this task for processing later
        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
        vQueueTasks.mDefer(std::move(pTask));
        break;
      }

      const int64_t cTStart = ggml_time_us();

      std::string vFilename = pTask.vSlotAction.vFilename;
      std::string vFilepath = pTask.vSlotAction.vFilepath;

      mgs_tokens vTokens;
      vTokens.resize(vSlot->vNCtx);
      size_t vTokenCount = 0;
      size_t vNread =
          mgs_state_seq_load_file(vCtxTgt, vFilepath.c_str(), vSlot->vId, vTokens.data(), vTokens.size(), &vTokenCount);
      if (vNread == 0) {
        vSlot->vPrompt.mClear(); // KV may already been invalidated?
        mSendError(pTask, "Unable to restore slot, no available space in KV cache or invalid slot save file",
                   cErrorTypeInvalidRequest);
        break;
      }
      vTokens.resize(vTokenCount);
      vSlot->vPrompt.mClear();
      vSlot->vPrompt.vTokens.mInsert(vTokens);

      const int64_t cTEnd = ggml_time_us();
      const double cTRestoreMs = (cTEnd - cTStart) / 1000.0;

      auto vRes = std::make_unique<server_task_result_slot_save_load>();
      vRes->vId = pTask.vId;
      vRes->vIdSlot = cIdSlot;
      vRes->vFilename = vFilename;
      vRes->vIsSave = false;
      vRes->vNTokens = vTokenCount;
      vRes->vNBytes = vNread;
      vRes->vTMs = cTRestoreMs;
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeSlotErase: {
      const int cIdSlot = pTask.vSlotAction.vIdSlot;
      server_slot *vSlot = mGetSlotById(cIdSlot);
      if (vSlot == nullptr) {
        mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
        break;
      }
      // Gate on slot content, consistent with save/restore.
      if (!mCheckSlotNoMedia(*vSlot, pTask.vId)) {
        break;
      }
      if (vSlot->mIsProcessing()) {
        // if requested slot is unavailable, we defer this task for processing later
        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
        vQueueTasks.mDefer(std::move(pTask));
        break;
      }

      // Erase token cache
      const size_t cNErased = vSlot->vPrompt.vTokens.size();

      vSlot->mPromptClear();

      auto vRes = std::make_unique<server_task_result_slot_erase>();
      vRes->vId = pTask.vId;
      vRes->vIdSlot = cIdSlot;
      vRes->vNErased = cNErased;
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeGetLora: {
      // TODO @ngxson : make lora_adapters a dedicated member of server_context
      auto &vLoras = vParamsBase.lLoraAdapters;
      auto vRes = std::make_unique<server_task_result_get_lora>();
      vRes->vId = pTask.vId;
      for (size_t vI = 0; vI < vLoras.size(); ++vI) {
        auto &vLora = vLoras[vI];
        std::string vAloraInvocationString = "";
        const uint64_t cNAloraTokens = mgs_adapter_get_alora_n_invocation_tokens(vLora.vPtr);
        mgs_tokens vAloraInvocationTokens;
        if (cNAloraTokens) {
          const mgs_token *vAloraTokens = mgs_adapter_get_alora_invocation_tokens(vLora.vPtr);
          for (uint64_t vJ = 0; vJ < cNAloraTokens; ++vJ) {
            vAloraInvocationString += fCommonTokenToPiece(vVocab, vAloraTokens[vJ]);
            vAloraInvocationTokens.push_back(vAloraTokens[vJ]);
          }
        }
        vRes->lLoras.push_back(server_task_result_get_lora::lora{
            vLora,
            vAloraInvocationString,
            vAloraInvocationTokens,
        });
      }
      vQueueResults.mSend(std::move(vRes));
    } break;
    case cServerTaskTypeSetLora: {
      auto vNewLoras = mConstructLoraList(pTask.dSetLora);
      // logging
      for (size_t vI = 0; vI < vNewLoras.size(); ++vI) {
        SRV_TRC("set lora adapter idx=%zu scale=%f\n", vI, vNewLoras[vI].vScale);
      }
      // TODO @ngxson : make lora_adapters a dedicated member of server_context
      vParamsBase.lLoraAdapters = vNewLoras;
      auto vRes = std::make_unique<server_task_result_apply_lora>();
      vRes->vId = pTask.vId;
      vQueueResults.mSend(std::move(vRes));
    } break;
    }
  }

  void mIterate(std::vector<server_slot> &pSlots, std::function<void(server_slot &)> pCallback) {
    for (auto &vSlot : pSlots) {
      try {
        pCallback(vSlot);
      } catch (const std::exception &vE) {
        SLT_ERR(vSlot, "got exception: %s\n", vE.what());
        mSendError(vSlot, std::string("got exception: ") + vE.what(), cErrorTypeServer);
        vSlot.mRelease();
      }
    }
  }

  void mIterate(std::vector<server_slot *> &pSlots, std::function<void(server_slot &)> pCallback) {
    for (auto &vSlot : pSlots) {
      try {
        pCallback(*vSlot);
      } catch (const std::exception &vE) {
        SLT_ERR(*vSlot, "got exception: %s\n", vE.what());
        mSendError(*vSlot, std::string("got exception: ") + vE.what(), cErrorTypeServer);
        vSlot->mRelease();
      }
    }
  }

  void mAbortAllSlots(const std::string &pReason) {
    for (auto &vSlot : lSlots) {
      if (vSlot.mIsProcessing()) {
        mSendError(vSlot, pReason, cErrorTypeServer);
        vSlot.mRelease();
      }
    }
  }

  // @ngxson : for debugging only
  int64_t vTPreDecode = 0;
  int64_t vTDecode = 0;
  int64_t vTPostDecode = 0;
  int64_t vTSampl = 0;
  int64_t vNPreDecode = 0;
  int64_t vNDecode = 0;
  int64_t vNPostDecode = 0;
  int64_t vNSampl = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
  struct scoped_timer {
    int64_t &vT;
    int64_t &vN;
    int64_t vTStart;
    scoped_timer(int64_t &pT, int64_t &pN) : vT(pT), vN(pN) { vTStart = ggml_time_us(); }
    ~scoped_timer() {
      vT += ggml_time_us() - vTStart;
      vN++;
    }
  };
#else
  struct scoped_timer {
    scoped_timer(int64_t &, int64_t &) {}
    ~scoped_timer() {}
  };
#endif

  void mUpdateSlots() {
#ifdef DEBUG_TIMINGS
    static int64_t vTPrev = 0;
    int64_t vTStart = ggml_time_us();
    if (vTStart - vTPrev > 5 * 1000 * 1000) { // every 5 seconds
      vTPrev = vTStart;
      SRV_INF("n_pre_decode      = %" PRId64 "\n", vNPreDecode);
      SRV_INF("avg t_pre_decode  = %f ms\n", (double)vTPreDecode / vNPreDecode / 1000.0);
      SRV_INF("avg t_decode      = %f ms\n", (double)vTDecode / vNDecode / 1000.0);
      SRV_INF("avg t_post_decode = %f ms\n", (double)vTPostDecode / vNPostDecode / 1000.0);
      SRV_INF("avg t_sampl       = %f ms\n", (double)vTSampl / vNSampl / 1000.0);
    }
#endif

    // check if all slots are idle
    {
      bool vAllIdle = true;

      for (auto &vSlot : lSlots) {
        if (vSlot.mIsProcessing()) {
          vAllIdle = false;
          break;
        }
      }

      if (vAllIdle) {
        SRV_TRC("%s", "all slots are idle\n");
        return; // skip further processing

      } else {
        SRV_DBG("%s", "posting NEXT_RESPONSE\n");

        server_task vTask(cServerTaskTypeNextResponse);
        vTask.vId = vQueueTasks.mGetNewId();
        vQueueTasks.mPost(std::move(vTask));
      }
    }

    try {
      scoped_timer vT(vTPreDecode, vNPreDecode);
      mPreDecode();
      vBatch.mRender();
    } catch (const std::exception &vE) {
      SRV_ERR("mPreDecode() failed: %s\n", vE.what());
      mAbortAllSlots("mPreDecode() failed: " + std::string(vE.what()));
    }

    GGML_ASSERT(vBatch.vSlotBatched || vBatch.size() == 0);

    if (vBatch.vSlotBatched) {
      auto &vSlotBatched = vBatch.vSlotBatched;
      auto &vAloraScale = vBatch.vAloraScale;
      auto &vAloraDisabledId = vBatch.vAloraDisabledId;

      // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
      // apply lora, only need to do it once per batch
      fCommonSetAdapterLora(vCtxTgt, vSlotBatched->lLora);

      // if the lora is temporarily disabled for an alora, re-enable it
      // for next time
      if (vAloraScale > 0.0f) {
        SRV_DBG("re-enabling alora with scale %f\n", vAloraScale);
        vSlotBatched->lLora[vAloraDisabledId].vScale = vAloraScale;
      }

      mgs_set_embeddings(vCtxTgt, vSlotBatched->mNeedEmbd());
    }

    mgs_batch vBatchView;
    int32_t vOffNext = 0;
    int32_t vNBatch = mgs_n_batch(vCtxTgt);
    for (int32_t vOff = 0; vOff < vBatch.size(); vOff = vOffNext) {
      const int32_t cNTokens = std::min(vNBatch, vBatch.size() - vOff);
      try {
        scoped_timer vT(vTDecode, vNDecode);
        // TODO @ngxson : maybe handle n_batch == 1 here instead of inside mDecode()

        vBatchView = vBatch.mGetView(vOff, cNTokens);
        bool vOk = mDecode(vNBatch, vOff, vBatchView);
#ifdef DEBUG_TIMINGS
        mgs_synchronize(vCtxTgt);
#endif

        if (vOk) {
          // move the head of the batch forward with the number of tokens we just processed
          vOffNext = vOff + cNTokens;

          // on successful decode, restore the original batch size
          vNBatch = mgs_n_batch(vCtxTgt);
        } else {
          // try again with the updated n_batch
          continue;
        }
      } catch (const std::exception &vE) {
        SRV_ERR("mDecode() failed: %s\n", vE.what());
        mAbortAllSlots("mDecode() failed: " + std::string(vE.what()));
        break; // stop any further processing
      }

      try {
        scoped_timer vT(vTPostDecode, vNPostDecode);
        mPostDecode(cNTokens, vOff, vBatchView);
      } catch (const std::exception &vE) {
        SRV_ERR("mPostDecode() failed: %s\n", vE.what());
        mAbortAllSlots("mPostDecode() failed: " + std::string(vE.what()));
        break; // stop any further processing
      }
    }
  }

  void mPreDecode() {
    // apply context-shift if needed
    // TODO: simplify and improve
    mIterate(lSlots, [&](server_slot &pSlot) {
      if (pSlot.vState == cSlotStateGenerating && pSlot.vPrompt.mNTokens() + 1 >= pSlot.vNCtx) {
        if (!vParamsBase.vCtxShift) {
          // this check is redundant (for good)
          // we should never get here, because generation should already stopped in mProcessToken()
          mSendError(pSlot, "context shift is disabled", cErrorTypeServer);
          pSlot.mRelease();
          return;
        }

        if (vMctx) {
          // we should never reach this because vParamsBase.vCtxShift is automatically disabled if mmproj is loaded
          // we don't support ctx_shift because an image chunk may contains multiple tokens
          GGML_ABORT("not supported by multimodal");
        }

        if (pSlot.vTask->mIsParent() || pSlot.vTask->mIsChild()) {
          mSendError(pSlot, "context shift cannot be used for shared prompt", cErrorTypeServer);
          pSlot.mRelease();
          return;
        }

        // Shift context
        int vNKeep = pSlot.vTask->vParams.vNKeep < 0 ? pSlot.vTask->mNTokens() : pSlot.vTask->vParams.vNKeep;

        if (vAddBosToken) {
          vNKeep += 1;
        }

        vNKeep = std::min(pSlot.vNCtx - 4, vNKeep);

        const int cNLeft = pSlot.vPrompt.mNTokens() - vNKeep;
        int vNDiscard = pSlot.vTask->vParams.vNDiscard ? pSlot.vTask->vParams.vNDiscard : (cNLeft / 2);

        vNDiscard = std::clamp(vNDiscard, 0, std::max(0, cNLeft - 1));

        SLT_WRN(pSlot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", vNKeep, cNLeft, vNDiscard);

        pSlot.vMem.mSeqRm(pSlot.vId, vNKeep, vNKeep + vNDiscard);
        pSlot.vMem.mSeqAdd(pSlot.vId, vNKeep + vNDiscard, pSlot.vPrompt.vTokens.mPosNext(), -vNDiscard);

        // add generated tokens to cache
        {
          GGML_ASSERT(!pSlot.vPrompt.vTokens.vHasMtmd);

          mgs_tokens vNewTokens = pSlot.vPrompt.vTokens.mGetTokens(); // copy
          for (size_t vI = vNKeep + vNDiscard; vI < vNewTokens.size(); vI++) {
            vNewTokens[vI - vNDiscard] = vNewTokens[vI];
          }

          vNewTokens.resize(pSlot.vPrompt.vTokens.size() - vNDiscard);

          pSlot.vPrompt.mClear();
          pSlot.vPrompt.vTokens.mInsert(vNewTokens);
        }

        pSlot.vTruncated = true;
      }
    });

    // start populating the batch for this iteration
    vBatch.mClear();

    // track if given slot can be batched with slots already in the batch
    auto &vSlotBatched = vBatch.vSlotBatched;

    std::vector<server_slot *> lGenerating;
    std::vector<server_slot *> lDrafting;

    // determine which slots are generating and drafting
    mIterate(lSlots, [&](server_slot &pSlot) {
      if (pSlot.vState != cSlotStateGenerating) {
        return;
      }

      // check if we can batch this slot with the previous one
      if (!vSlotBatched) {
        vSlotBatched = &pSlot;
      } else if (!vSlotBatched->mCanBatchWith(pSlot)) {
        return;
      }

      lGenerating.push_back(&pSlot);

      if (vSpec) {
        fCommonSpeculativeGetDraftParams(vSpec.get(), pSlot.vId).vDrafting = false;

        const bool cUseCkptTgt = vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull;
        const bool cUseCkptDft = vCtxDftSeqRmType == cCommonContextSeqRmTypeFull;

        const int cNDraftMax = pSlot.mGetNDraftMax();

        if (cNDraftMax > 0) {
          GGML_ASSERT(pSlot.mCanSpeculate());

          if (!pSlot.vSpecDraft.empty()) {
            // we have a previous (partial) draft to reuse
            if (cUseCkptTgt) {
              GGML_ASSERT(!pSlot.vSpecCkpt.empty());
            }
          } else {
            GGML_ASSERT(pSlot.lSpecIBatch.empty());

            pSlot.vSpecCkpt.mUpdatePos(pSlot.vPrompt.mNTokens(),
                                       mgs_memory_seq_pos_min(mgs_get_memory(vCtxTgt), pSlot.vId),
                                       mgs_memory_seq_pos_max(mgs_get_memory(vCtxTgt), pSlot.vId));

            if (cUseCkptDft) {
              pSlot.vSpecCkpt.mUpdateDft(vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            pSlot.vSpecPrompt = pSlot.vPrompt.vTokens.mGetTextTokens();

            fCommonSpeculativeGetDraftParams(vSpec.get(), pSlot.vId) = {
                /* .vDrafting = */ true,
                /* .vNMax     = */ cNDraftMax,
                // Multimodal positions can differ from the number of prompt tokens.
                /* .vNPast    = */ pSlot.vPrompt.vTokens.mPosNext(),
                /* .vIdLast   = */ pSlot.vSampled,
                /* .vPrompt   = */ &pSlot.vSpecPrompt,
                /* .vResult   = */ &pSlot.vSpecDraft,
            };

            lDrafting.push_back(&pSlot);
          }
        }
      }
    });

    // generate the actual drafts (if any)
    {
      fCommonSpeculativeDraft(vSpec.get());
    }

    // make checkpoints if needed
    mIterate(lDrafting, [&](server_slot &pSlot) {
      auto &vDraft = pSlot.vSpecDraft;
      auto &vCkpt = pSlot.vSpecCkpt;

      pSlot.vNDraftTotal += vDraft.size();

      // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed
      // [TAG_SPEC_AVOID_DRAFT_REEVAL]
      const bool cUseCkptDft = vCtxDftSeqRmType == cCommonContextSeqRmTypeFull;

      if (vCtxDft) {
        if (cUseCkptDft) {
          vCkpt.mLoadDft(vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }

        if (!mgs_memory_seq_rm(mgs_get_memory(vCtxDft), pSlot.vId, vCkpt.vPosMax + 1, -1)) {
          GGML_ABORT("failed to remove sequence %d\n", pSlot.vId);
        }
      }

      if (!vDraft.empty()) {
        const bool cUseCkptTgt =
            vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
            (vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs && vDraft.size() > mgs_n_rs_seq(vCtxTgt));

        const bool cUseCkptDft =
            (vCtxDftSeqRmType == cCommonContextSeqRmTypeRs && vDraft.size() > mgs_n_rs_seq(vCtxDft));

        if (cUseCkptTgt) {
          // const int64_t cTStart = ggml_time_us();

          vCkpt.mUpdateTgt(vCtxTgt, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);

          // const int64_t cTTotal = ggml_time_us() - cTStart;
          // printf("checkpoint total: %f ms\n", cTTotal / 1000.0);

          SLT_DBG(pSlot,
                  "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = "
                  "%.3f MiB)\n",
                  vCkpt.vPosMin, vCkpt.vPosMax, pSlot.vPrompt.mNTokens(), (float)vCkpt.size() / 1024 / 1024,
                  (float)vCkpt.lDataDft.size() / 1024 / 1024);
        }

        if (cUseCkptDft) {
          vCkpt.mUpdateDft(vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
      }
    });

    // update the batch with the sampled/drafted tokens
    mIterate(lGenerating, [&](server_slot &pSlot) { pSlot.mHandleLastSampledToken(vBatch); });

    // process in chunks of params.n_batch
    int32_t vNBatch = mgs_n_batch(vCtxTgt);
    int32_t vNUbatch = mgs_n_ubatch(vCtxTgt);

    auto &vAloraScale = vBatch.vAloraScale;
    auto &vAloraDisabledId = vBatch.vAloraDisabledId;

    // next, batch any pending prompts without exceeding n_batch
    if (vParamsBase.vContBatching || vBatch.size() == 0) {
      bool vAddOk = true; // false means the batch is full, skip remaining slots

      mIterate(lSlots, [&](server_slot &pSlot) {
        if (!vAddOk || vBatch.size() >= vNBatch) {
          return; // batch is full, skip remaining slots
        }

        if (!pSlot.mIsProcessing()) {
          return;
        }

        // check if we can batch this slot with the previous one
        if (vSlotBatched && !vSlotBatched->mCanBatchWith(pSlot)) {
          return;
        }

        // check if this is a child slot
        if (pSlot.vState == cSlotStateWaitOther) {
          SLT_DBG(pSlot, "%s", "waiting for parent slot to complete\n");
          return;
        }

        // this slot still has a prompt to be processed
        if (pSlot.vState == cSlotStateProcessingPrompt || pSlot.vState == cSlotStateStarted) {
          const auto &vInputTokens = pSlot.vTask->vTokens;

          // used to determine the number of tokens added to the batch for the current slot
          const auto cNTokensPrev = vBatch.size();

          // TODO: maybe move branch to outside of this loop in the future
          if (pSlot.vState == cSlotStateStarted) {
            pSlot.vTStartProcessPrompt = ggml_time_us();
            pSlot.vTStartGeneration = 0;

            pSlot.vState = cSlotStateProcessingPrompt;

            SLT_TRC(pSlot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n", pSlot.vNCtx,
                    pSlot.vTask->vParams.vNKeep, pSlot.vTask->mNTokens());

            // print prompt tokens (for debugging)
            /*if (1) {
                // first 16 tokens (avoid flooding logs)
                for (int vI = 0; vI < std::min<int>(16, vInputTokens.size()); vI++) {
                    SLT_DBG(pSlot, "prompt token %3d: %6d '%s'\n", vI, vInputTokens[vI], fCommonTokenToPiece(vCtxTgt,
            vInputTokens[vI]).c_str());
                }
            } else {
                // all
                for (int vI = 0; vI < (int) vInputTokens.size(); vI++) {
                    SLT_DBG(pSlot, "prompt token %3d: %6d '%s'\n", vI, vInputTokens[vI], fCommonTokenToPiece(vCtxTgt,
            vInputTokens[vI]).c_str());
                }
            }*/

            // keep track how many tokens we can reuse from the previous state
            int vNPast = 0;

            // empty prompt passed -> release the slot and send empty response
            if (vInputTokens.empty()) {
              SLT_WRN(pSlot, "%s", "empty prompt - releasing slot\n");

              pSlot.mPrintTimings();
              mSendFinalResponse(pSlot);
              pSlot.mRelease();

              return;
            }

            // TODO: support memory-less logits computation
            if (pSlot.vTask->mNeedLogits() && !mgs_get_memory(vCtxTgt)) {
              mSendError(pSlot, "the current context does not logits computation. skipping", cErrorTypeServer);
              pSlot.mRelease();
              return;
            }

            if (!pSlot.mCanSplit()) {
              if (pSlot.vTask->mNTokens() > vNUbatch) {
                mSendError(pSlot,
                           fStringFormat("input (%d tokens) is too large to process. increase the physical batch "
                                         "size (current batch size: %d)",
                                         pSlot.vTask->mNTokens(), vNUbatch),
                           cErrorTypeServer);
                pSlot.mRelease();
                return;
              }

              if (pSlot.vTask->mNTokens() > pSlot.vNCtx) {
                mSendError(pSlot,
                           fStringFormat("input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                         pSlot.vTask->mNTokens(), pSlot.vNCtx),
                           cErrorTypeExceedContextSize);
                pSlot.mRelease();
                return;
              }
            } else {
              if (pSlot.vTask->mNTokens() >= pSlot.vNCtx) {
                mSendError(pSlot,
                           fStringFormat("request (%d tokens) exceeds the available context size (%d "
                                         "tokens), try increasing it",
                                         pSlot.vTask->mNTokens(), pSlot.vNCtx),
                           cErrorTypeExceedContextSize);
                pSlot.mRelease();
                return;
              }

              if (pSlot.vTask->vParams.vCachePrompt) {
                // reuse any previously computed tokens that are common with the new prompt
                vNPast = pSlot.vPrompt.vTokens.mGetCommonPrefix(vInputTokens);

                // if there is an alora invoked, don't cache after the invocation start
                if (pSlot.vAloraInvocationStart > 0) {
                  SLT_DBG(pSlot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n",
                          vNPast, pSlot.vAloraInvocationStart);
                  vNPast = std::min(vNPast, pSlot.vAloraInvocationStart - 1);
                }

                const auto cNCacheReuse = pSlot.vTask->vParams.vNCacheReuse;

                const bool cCanCacheReuse =
                    mgs_memory_can_shift(mgs_get_memory(vCtxTgt)) && !pSlot.vPrompt.vTokens.vHasMtmd;

                if (!cCanCacheReuse && cNCacheReuse > 0) {
                  SLT_WRN(pSlot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", cNCacheReuse);
                }

                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                if (cCanCacheReuse && cNCacheReuse > 0) {
                  GGML_ASSERT(!pSlot.vPrompt.vTokens.vHasMtmd);

                  size_t vHeadC = vNPast; // cache
                  size_t vHeadP = vNPast; // current prompt

                  if (vMctx) {
                    // we should never reach this
                    GGML_ABORT("not supported by multimodal");
                  }

                  SLT_DBG(pSlot, "trying to reuse chunks with size > %d, n_past = %d\n", cNCacheReuse, vNPast);

                  while (vHeadC < pSlot.vPrompt.vTokens.size() && vHeadP < vInputTokens.size()) {

                    size_t vNMatch = 0;
                    while (vHeadC + vNMatch < pSlot.vPrompt.vTokens.size() && vHeadP + vNMatch < vInputTokens.size() &&
                           pSlot.vPrompt.vTokens[vHeadC + vNMatch] == vInputTokens[vHeadP + vNMatch]) {
                      vNMatch++;
                    }

                    if (vNMatch >= (size_t)cNCacheReuse) {
                      SLT_TRC(pSlot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n",
                              vNMatch, vHeadC, vHeadC + vNMatch, vHeadP, vHeadP + vNMatch);
                      // for (size_t vI = vHeadP; vI < vHeadP + vNMatch; vI++) {
                      //     SLT_DBG(pSlot, "cache token %3zu: %6d '%s'\n", vI, vInputTokens[vI],
                      //     fCommonTokenToPiece(vCtxTgt, vInputTokens[vI]).c_str());
                      // }

                      const int64_t cKvShift = (int64_t)vHeadP - (int64_t)vHeadC;

                      pSlot.vMem.mSeqRm(pSlot.vId, vHeadP, vHeadC);
                      pSlot.vMem.mSeqAdd(pSlot.vId, vHeadC, vHeadC + vNMatch, cKvShift);

                      for (size_t vI = 0; vI < vNMatch; vI++) {
                        pSlot.vPrompt.vTokens.mSetToken(vHeadP + vI, pSlot.vPrompt.vTokens[vHeadC + vI]);
                        vNPast++;
                      }

                      vHeadC += vNMatch;
                      vHeadP += vNMatch;
                    } else {
                      vHeadC += 1;
                    }
                  }

                  SLT_DBG(pSlot, "after context reuse, new n_past = %d\n", vNPast);
                }
              } else {
                // if we don't cache the prompt, we have to remove all previous tokens
                vNPast = 0;
              }

              mgs_pos vPosNext = pSlot.vPrompt.vTokens.mPosNext(vNPast);

              const bool cHasNewTokens = (vNPast < pSlot.vTask->mNTokens());

              // the largest pos_min required for a checkpoint to be useful
              const auto cPosMinThold = std::max(0, vPosNext - vNSwa - (cHasNewTokens ? 0 : 1));

              if (vNPast > 0 && vNPast <= pSlot.vPrompt.mNTokens()) {
                const auto cPosMin = mgs_memory_seq_pos_min(mgs_get_memory(vCtxTgt), pSlot.vId);
                if (cPosMin == -1) {
                  SLT_ERR(pSlot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", vNPast,
                          (int)pSlot.vPrompt.vTokens.size(), pSlot.vId, cPosMin);
                  GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen");
                }

                // when the prompt prefix does not match, print the tokens around the mismatch
                // this is useful for debugging prompt caching
                if (vSlotsDebug) {
                  const int cNp0 = std::max<int>(vNPast - 4, 0);
                  const int cNp1 =
                      std::min<int>(vNPast + 6, std::min(pSlot.vPrompt.vTokens.size(), pSlot.vTask->vTokens.size()));

                  std::stringstream vSs0;
                  std::stringstream vSs1;

                  std::stringstream vSt0;
                  std::stringstream vSt1;

                  vSs0 << "old: ... ";
                  vSs1 << "new: ... ";

                  for (int vI = cNp0; vI < cNp1; vI++) {
                    if (vI == vNPast) {
                      vSs0 << " | ";
                      vSs1 << " | ";
                    }

                    {
                      const auto cToken = pSlot.vPrompt.vTokens[vI];
                      const auto cPiece = cToken != MGS_TOKEN_NULL ? fCommonTokenToPiece(vCtxTgt, cToken) : "[mtmd]";
                      vSs0 << cPiece;
                      vSt0 << std::setw(8) << cToken;
                    }

                    {
                      const auto cToken = pSlot.vTask->vTokens[vI];
                      const auto cPiece = cToken != MGS_TOKEN_NULL ? fCommonTokenToPiece(vCtxTgt, cToken) : "[mtmd]";
                      vSs1 << cPiece;
                      vSt1 << std::setw(8) << cToken;
                    }
                  }

                  SLT_WRN(pSlot, "%s\n", vSs0.str().c_str());
                  SLT_WRN(pSlot, "%s\n", vSs1.str().c_str());

                  SLT_WRN(pSlot, "%s\n", vSt0.str().c_str());
                  SLT_WRN(pSlot, "%s\n", vSt1.str().c_str());
                }

                if (cPosMin >= cPosMinThold) {
                  // search for a context checkpoint
                  const auto cIt = std::find_if(pSlot.vPrompt.lCheckpoints.rbegin(), pSlot.vPrompt.lCheckpoints.rend(),
                                                [&](const auto &pCur) {
                                                  // guarantee that a checkpoint will result in at least one token being
                                                  // processed [TAG_PROMPT_LOGITS]
                                                  SLT_TRC(pSlot, "checking checkpoint with [%d, %d] against %d...\n",
                                                          pCur.vPosMin, pCur.vPosMax, cPosMinThold);
                                                  // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                                  if (pCur.vPosMax > vPosNext) {
                                                    return false;
                                                  }
                                                  return pCur.vPosMin < cPosMinThold || pCur.vPosMin == 0;
                                                });

                  bool vDoReset = cIt == pSlot.vPrompt.lCheckpoints.rend();

                  if (!vDoReset) {
                    // restore the context checkpoint
                    cIt->mLoadTgt(vCtxTgt, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    cIt->mLoadDft(vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    // restore the draft's speculative state
                    fCommonSpeculativeSetState(vSpec.get(), pSlot.vId, cIt->lDataSpec);

                    vPosNext = std::min(vPosNext, std::max(cIt->vPosMin + 1, cIt->vPosMax));
                    vNPast = std::min(pSlot.vPrompt.vTokens.mSizeUpToPos(vPosNext), (size_t)cIt->vNTokens);
                    SLT_TRC(pSlot,
                            "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                            ", n_past = %d, size = %.3f MiB)\n",
                            cIt->vPosMin, cIt->vPosMax, cIt->vNTokens, vNPast, (float)cIt->size() / 1024 / 1024);
                  }

                  if (vDoReset) {
                    SLT_TRC(pSlot, "%s",
                            "forcing full prompt re-processing due to lack of cache data (likely due to SWA or "
                            "hybrid/recurrent memory)\n");
                    vPosNext = 0;
                    vNPast = 0;
                  }
                }
              }

              {
                // erase any checkpoints with pos_max > pos_next
                for (auto vIt = pSlot.vPrompt.lCheckpoints.begin(); vIt != pSlot.vPrompt.lCheckpoints.end();) {
                  const auto &vCur = *vIt;
                  if (vCur.vPosMax > vPosNext) {
                    SLT_TRC(pSlot,
                            "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                            ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n",
                            vCur.vPosMin, vCur.vPosMax, vCur.vNTokens, vNSwa, vPosNext,
                            (float)vCur.size() / 1024 / 1024);
                    vIt = pSlot.vPrompt.lCheckpoints.erase(vIt);
                  } else {
                    ++vIt;
                  }
                }
              }
            }

            // [TAG_PROMPT_LOGITS]
            if (vNPast == pSlot.vTask->mNTokens() && vNPast > 0) {
              SLT_WRN(pSlot,
                      "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n",
                      vNPast, pSlot.vTask->mNTokens());
              vNPast--;
              SLT_WRN(pSlot, "n_past was set to %d\n", vNPast);
            }

            pSlot.vNPromptTokensCache = vNPast;
            pSlot.vNPromptTokensProcessed = 0;

            pSlot.vPrompt.vTokens.mKeepFirst(vNPast);

            // this is to signal the client that the request has started processing
            if (pSlot.vTask->vParams.vStream) {
              if (pSlot.vTask->vParams.vReturnProgress) {
                // send initial 0% progress update if needed
                mSendPartialResponse(pSlot, {}, true);
              } else {
                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                mSendPartialResponse(pSlot, {}, false, true);
              }
            }
          } // end of SLOT_STATE_STARTED

          if (!pSlot.mCanSplit()) {
            // cannot fit the prompt in the current batch - will try next iter
            if (vBatch.size() + pSlot.vTask->mNTokens() > vNBatch) {
              return;
            }
          }

          const int64_t cTNow = ggml_time_us();
          pSlot.vTPromptProcessing = (cTNow - pSlot.vTStartProcessPrompt) / 1e3;
          pSlot.mPrintTimingsPp();

          // truncate any tokens that are beyond n_past for this slot
          const mgs_pos cP0 = pSlot.vPrompt.vTokens.mPosNext();

          SLT_TRC(pSlot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", pSlot.vPrompt.mNTokens(), cP0);

          pSlot.vMem.mSeqRm(pSlot.vId, cP0, -1);

          // If using an alora, there may be uncached tokens that come
          // before the invocation sequence. When this happens, the
          // tokens before the invocation sequence need to be
          // processed without the adapter in a separate batch, then
          // the adapter needs to be enabled for the remaining tokens.
          if (fLoraAllAlora(pSlot.lLora) && pSlot.vAloraInvocationStart - 1 > pSlot.vPrompt.mNTokens()) {
            SLT_DBG(pSlot,
                    "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n",
                    pSlot.vPrompt.mNTokens(), pSlot.vAloraInvocationStart);
            const auto &vEnabledLoras = fLoraGetEnabledIds(pSlot.lLora);
            GGML_ASSERT(vEnabledLoras.size() == 1);
            vAloraScale = pSlot.lLora[vEnabledLoras[0]].vScale;
            pSlot.lLora[vEnabledLoras[0]].vScale = 0.0f;
            vAloraDisabledId = vEnabledLoras[0];
          }

          bool vDoCheckpoint = vParamsBase.vNCtxCheckpoints > 0;

          // make checkpoints only for completion tasks
          vDoCheckpoint = vDoCheckpoint && pSlot.vTask->vType == cServerTaskTypeCompletion;

          // make a checkpoint of the parts of the memory that cannot be rolled back.
          // checkpoints are created only if:
          // - the model does not support partial sequence removal
          // - the model uses SWA (and we are not using `swa_full`)
          // - the model supports partial sequence removal but only up to a fixed bound
          vDoCheckpoint = vDoCheckpoint && (vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
                                            vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs || vNSwa > 0);

          bool vHasMtmd = false;

          // check if we should process the image
          while (true) {
            auto vCurTokenIdx = pSlot.vPrompt.mNTokens();
            if (vCurTokenIdx >= pSlot.vTask->mNTokens() ||
                vInputTokens[vCurTokenIdx] != MGS_TOKEN_NULL // encountered a text token
            ) {
              break;
            }

            // process the image
            size_t vNTokensOut = 0;
            int32_t vRes = pSlot.mProcessMtmdChunk(vCurTokenIdx, vNTokensOut);
            if (vRes != 0) {
              SLT_ERR(pSlot, "failed to process image, res = %d\n", vRes);
              mSendError(pSlot, "failed to process image", cErrorTypeServer);
              pSlot.mRelease();
              continue;
            }

            pSlot.vNPromptTokensProcessed += vNTokensOut;

            // add the image chunk to cache
            {
              const auto &vChunk = vInputTokens.mFindChunk(vCurTokenIdx);
              pSlot.vPrompt.vTokens.mPushBack(vChunk.get()); // copy
            }

            vHasMtmd = true;
          }

          const auto &vSpans = pSlot.vTask->vParams.vMessageSpans;
          const auto cLastUserPos = vSpans.mLastUserMessagePos();

          // add prompt tokens for processing in the current batch
          while (pSlot.vPrompt.mNTokens() < pSlot.vTask->mNTokens() && vBatch.size() < vNBatch) {
            // get next token to process
            mgs_token vCurTok = vInputTokens[pSlot.vPrompt.mNTokens()];
            if (vCurTok == MGS_TOKEN_NULL) {
              break; // end of text chunk
            }

            // if this is an alora request with pre-invocation
            // tokens that are not cached, we need to stop filling
            // this batch at those pre-invocation tokens.
            if (vAloraScale > 0 && pSlot.vPrompt.mNTokens() == pSlot.vAloraInvocationStart - 1) {
              SLT_DBG(pSlot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n",
                      pSlot.vPrompt.mNTokens(), pSlot.vAloraInvocationStart);
              break;
            }

            // embedding requires all tokens in the batch to be output;
            // MTP also wants logits at every prompt position so the
            // streaming hook can mirror t_h_nextn into ctx_dft.
            vAddOk &= vBatch.mAdd(pSlot.vId, vCurTok, pSlot.vPrompt.vTokens.mPosNext(), pSlot.mNeedEmbd());
            pSlot.vPrompt.vTokens.mPushBack(vCurTok);

            pSlot.vNPromptTokensProcessed++;

            // break at the last user message, or at user, assistant or tool messages at least min step past the
            // last checkpoint
            if (vDoCheckpoint && vSpans.mIsTurnStart(pSlot.vPrompt.mNTokens())) {
              const auto cPos = pSlot.vPrompt.mNTokens();
              const auto &vCheckpoints = pSlot.vPrompt.lCheckpoints;

              // Every extra micro-batch streams all offloaded expert weights again, so ordinary
              // turn checkpoints only cut a batch whose last micro-batch is at least 3/4 full.
              const int64_t vUbatchFill = (int64_t)vBatch.size() % vNUbatch;
              const bool vAlignedBreak = vUbatchFill == 0 || vUbatchFill >= (3 * (int64_t)vNUbatch) / 4;

              if (cPos == cLastUserPos || vCheckpoints.empty() ||
                  (cPos > vCheckpoints.back().vNTokens + vParamsBase.vCheckpointMinStep && vAlignedBreak)) {
                break;
              }
            }

            // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
            // create checkpoints that many tokens before the end of the prompt:
            //  - 4 + n_ubatch
            //  - 4
            if (vDoCheckpoint) {
              static const int aCheckpointOffsets[] = {4 + vNUbatch, 4};

              bool vShouldBreak = false;
              for (int vOffset : aCheckpointOffsets) {
                const int cNLast = std::min(vNBatch, vOffset);
                if (pSlot.vTask->mNTokens() == pSlot.vPrompt.mNTokens() + cNLast) {
                  vShouldBreak = true;
                  break;
                }
              }
              if (vShouldBreak) {
                break;
              }
            }
          }

          // the number of tokens added to the batch for the current slot
          const auto cNTokensCur = vBatch.size() - cNTokensPrev;

          const auto cNTokensStart = pSlot.vPrompt.mNTokens() - cNTokensCur;

          const bool cNearPromptEnd = pSlot.vTask->mNTokens() < pSlot.vPrompt.mNTokens() + vNUbatch;

          const bool cIsTurnStart = vSpans.mIsTurnStart(cNTokensStart);
          const bool cIsLastUserMessage = cNTokensStart == cLastUserPos;

          // entire prompt has been processed
          if (pSlot.vPrompt.mNTokens() == pSlot.vTask->mNTokens()) {
            pSlot.vState = cSlotStateDonePrompt;

            GGML_ASSERT(vBatch.size() > 0);

            // extract the logits only for the last token
            vBatch.mSetOutput(vBatch.size() - 1, true);

            pSlot.vNDecoded = 0;
            pSlot.vIBatch = vBatch.size() - 1;

            pSlot.mInitSampler();
          } else {
            // Mid-prompt checkpoints go at the start of a batch that starts a user, assistant or tool
            // message, near the end of the prompt, or at any batch once the previous checkpoint is
            // checkpoint_min_step tokens behind. The last case needs no extra cut, so it costs no
            // additional pass over the offloaded expert weights and bounds how far a context edit
            // has to go back.
            if (!cIsTurnStart && !cNearPromptEnd && !pSlot.vPrompt.lCheckpoints.empty() &&
                cNTokensStart <= pSlot.vPrompt.lCheckpoints.back().vNTokens + vParamsBase.vCheckpointMinStep) {
              vDoCheckpoint = false;
            }
          }

          const auto cPosMin = mgs_memory_seq_pos_min(mgs_get_memory(vCtxTgt), pSlot.vId);
          const auto cPosMax = mgs_memory_seq_pos_max(mgs_get_memory(vCtxTgt), pSlot.vId);

          // nothing to checkpoint yet
          // TODO: is this check needed?
          if (vDoCheckpoint && cPosMin < 0) {
            vDoCheckpoint = false;
          }

          // do not checkpoint after mtmd chunks
          vDoCheckpoint = vDoCheckpoint && !vHasMtmd;

          // no need to create checkpoints that are too close together, unless it's the last user message
          vDoCheckpoint = vDoCheckpoint &&
                          (pSlot.vPrompt.lCheckpoints.empty() || cIsLastUserMessage || cNearPromptEnd ||
                           cNTokensStart > pSlot.vPrompt.lCheckpoints.back().vNTokens + vParamsBase.vCheckpointMinStep);
          SLT_DBG(pSlot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", vDoCheckpoint ? "yes" : "no", cPosMin,
                  cPosMax);

          // note: we create the checkpoint before calling mgs_decode(), so the current batch is not
          //       yet processed and therefore it is not part of the checkpoint.
          if (vDoCheckpoint) {
            mCreateCheckpoint(pSlot, cNTokensCur, cPosMin, cPosMax);
          }
        }

        if (!vSlotBatched) {
          vSlotBatched = &pSlot;
        }
      });
    }
  }

  // returns true = success ; false = retry with smaller batch size
  // throw std::runtime_error on fatal error
  bool mDecode(int32_t &pNBatch, int32_t pOff, mgs_batch &pBatchView) {
    SRV_DBG("n_batch (effective) = %d, off = %d\n", pNBatch, pOff);

    if (vBatch.size() == 0) {
      SRV_WRN("%s", "no tokens to decode\n");

      if (++vNEmptyConsecutive > 3) {
        GGML_ABORT("fatal error - please provide logs and repro in %s\n",
                   "https://github.com/nipegun/moe-gguf-server/issues");
      }

      return true; // nothing to decode
    } else {
      vNEmptyConsecutive = 0;
    }

    // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
    // this case is not currently used by any models, but may need to be supported in the future
    if (vSpec && vBatch.vHasEmbd) {
      if (mgs_model_n_embd_inp(vModelDft) != mgs_model_n_embd_inp(vModelTgt)) {
        SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
        throw std::runtime_error("unsupported batch.has_embd + spec case");
      }
    }

    const int cRet = mgs_decode(vCtxTgt, pBatchView);

    vMetrics.mOnDecoded(lSlots);

    if (cRet != 0) {
      {
        std::string vErr;

        if (pNBatch == 1 && cRet == 1) {
          // TODO: try to terminate only the largest active slot/sequence and continue with the rest
          //       need to remove the tokens from the current batch too
          vErr = "Context size has been exceeded.";
        }

        if (cRet == -1) {
          vErr = "Invalid input batch.";
        }

        if (cRet < -1) {
          // TODO: update slot state based on mgs_memory_seq_pos_min() and mgs_memory_seq_pos_max()
          vErr = "Compute error.";
        }

        // TODO: handle ret == 2 (abort) when we start aborting

        if (!vErr.empty()) {
          SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", vErr.c_str(), pOff, pNBatch, cRet);

          for (auto &vSlot : lSlots) {
            if (vSlot.mIsProcessing()) {
              mSendError(vSlot, vErr);
              vSlot.mRelease();

              // note: it's complicated to keep track of how much of the current batch has been
              //       processed before the error occurred, so we simply clear the entire context
              vSlot.mPromptClear();
            }
          }

          // stop, do not retry with smaller batch size
          throw std::runtime_error(vErr);
        }
      }

      // retry with half the batch size to try to find a free slot in the KV cache
      if (!mTryClearIdleSlots()) {
        pNBatch /= 2;
      }

      SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, "
              "ret = %d\n",
              pOff, pNBatch, cRet);

      return false; // retry with the updated n_batch
    }

    // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed
    // [TAG_SPEC_AVOID_DRAFT_REEVAL]
    //       for now, always re-evaluate for simplicity
    if (!fCommonSpeculativeProcess(vSpec.get(), pBatchView)) {
      SRV_ERR("%s", "failed to process speculative batch\n");

      // TODO: handle error
      throw std::runtime_error("failed to process speculative batch");
    }

    // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
    for (auto &vSlot : lSlots) {
      if (vSlot.vState == cSlotStateDonePrompt && vSlot.vTask->mIsParent()) {
        std::vector<server_slot *> lChildren;
        for (auto &vOther : lSlots) {
          if (vOther.vState == cSlotStateWaitOther && vSlot.vTask->vId == vOther.vTask->vIdParent) {
            lChildren.push_back(&vOther);
          }
        }

        // all children slots should already launched by mLaunchSlotsWithParentTask()
        // copy state to the child slots
        for (auto &vChild : lChildren) {
          SLT_TRC(vSlot, " - copying state to child %d\n", vChild->vId);

          GGML_ASSERT(vChild->vState == cSlotStateWaitOther);

          vSlot.mCopyStateTo(*vChild);
          vChild->vState = cSlotStateDonePrompt;
        }
      }
    }

    return true;
  }

  void mPostDecode(int32_t pNBatchTokens, int32_t pOff, mgs_batch &pBatchView) {
    // for checking if a given batch index is inside batch_view
    auto vIsInsideView = [&](int32_t pIdx) { return pIdx >= pOff && pIdx < pOff + pNBatchTokens; };

    // TODO @ngxson : it's tricky to make sub-batch compatible with fCommonSamplerSampleAndAcceptN,
    // so for now we will throw an error in this case
    mIterate(lSlots, [&](server_slot &pSlot) {
      for (auto &vI : pSlot.lSpecIBatch) {
        if (!vIsInsideView(vI)) {
          throw std::runtime_error(
              fStringFormat("speculative batch index %d is not inside the current sub-batch [%d, %d)", vI, pOff,
                            pOff + pNBatchTokens));
        }
      }
    });

    auto vAcceptSpecialToken = [&](server_slot &pSlot, mgs_token pToken) {
      return vParamsBase.vSpecial || pSlot.vTask->vParams.vSampling.sPreservedTokens.find(pToken) !=
                                         pSlot.vTask->vParams.vSampling.sPreservedTokens.end();
    };

    mIterate(lSlots, [&](server_slot &pSlot) {
      // optionally send prompt processing progress
      if (pSlot.vState == cSlotStateProcessingPrompt || pSlot.vState == cSlotStateDonePrompt) {
        if (pSlot.vTask->vParams.vStream && pSlot.vTask->vParams.vReturnProgress) {
          mSendPartialResponse(pSlot, {}, true);
        }
      }

      if (!vIsInsideView(pSlot.vIBatch)) {
        // the required token not in this sub-batch, skip
        return;
      }

      if (pSlot.vState == cSlotStateDonePrompt) {
        if (pSlot.vTask->vType == cServerTaskTypeEmbedding) {
          // prompt evaluated for embedding
          mSendEmbedding(pSlot, pBatchView);
          pSlot.mRelease();
          pSlot.vIBatch = -1;
          return;
        }

        if (pSlot.vTask->vType == cServerTaskTypeRerank) {
          mSendRerank(pSlot, pBatchView);
          pSlot.mRelease();
          pSlot.vIBatch = -1;
          return;
        }

        GGML_ASSERT(pSlot.vTask->mNeedSampling());

        // prompt evaluated for next-token prediction
        pSlot.vState = cSlotStateGenerating;

        if (pSlot.mCanSpeculate()) {
          fCommonSpeculativeBegin(vSpec.get(), pSlot.vId, pSlot.vPrompt.vTokens.mGetTextTokens());
        }
      } else if (pSlot.vState != cSlotStateGenerating) {
        return;
      }

      if (pSlot.mCanSpeculate() && !pSlot.vSpecDraft.empty()) {
        return; // sample using speculative decoding
      }

      // shifted according to the current sub-batch
      const int cTokIdx = pSlot.vIBatch - pOff;

      mgs_token vId;
      {
        scoped_timer vTimer(vTSampl, vNSampl);
        vId = fCommonSamplerSample(pSlot.vSmpl.get(), pSlot.vCtxTgt, cTokIdx);
      }

      pSlot.vIBatch = -1;

      fCommonSamplerAccept(pSlot.vSmpl.get(), vId, true);

      // here we have synchronized the mgs_context (due to the sampling above), so we can do time measurement
      const int64_t cTNow = ggml_time_us();

      pSlot.vNDecoded += 1;

      if (pSlot.vNDecoded == 1) {
        pSlot.vTStartGeneration = cTNow;
        pSlot.vTPrintLast = cTNow;
        pSlot.vNDecodedLast = 0;
        pSlot.vTPromptProcessing = (pSlot.vTStartGeneration - pSlot.vTStartProcessPrompt) / 1e3;
        vMetrics.mOnPromptEval(pSlot);
      }

      pSlot.vTTokenGeneration = std::max<int64_t>(1, cTNow - pSlot.vTStartGeneration) / 1e3;

      completion_token_output vResult;
      vResult.vTok = vId;
      vResult.vTextToSend = fCommonTokenToPiece(pSlot.vCtxTgt, vResult.vTok, vAcceptSpecialToken(pSlot, vResult.vTok));
      vResult.vProb = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

      if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
        mPopulateTokenProbs(pSlot, vResult, pSlot.vTask->vParams.vPostSamplingProbs, vParamsBase.vSpecial, cTokIdx);
      }

      if (!mProcessToken(vResult, pSlot)) {
        // release slot because of stop condition
        pSlot.mPrintTimings();
        mSendFinalResponse(pSlot);
        vMetrics.mOnPrediction(pSlot);
        pSlot.mRelease();

        return;
      }

      pSlot.mPrintTimingsTg();
    });

    // speculative decoding - main model sample and accept
    mIterate(lSlots, [&](server_slot &pSlot) {
      if (pSlot.vState != cSlotStateGenerating || !pSlot.mCanSpeculate() || pSlot.vSpecDraft.empty()) {
        return;
      }

      // save the original draft size
      const size_t cNDraft = pSlot.vSpecDraft.size();

      GGML_ASSERT(cNDraft > 0);

      // verify and try to accept the draft
      {
        // save the sampler sampler state in case we need to restore it
        common_sampler_ptr vSmplSave(fCommonSamplerClone(pSlot.vSmpl.get()));

        GGML_ASSERT(pSlot.lSpecIBatch.size() == cNDraft + 1);
        auto vAccepted =
            fCommonSamplerSampleAndAcceptN(pSlot.vSmpl.get(), pSlot.vCtxTgt, pSlot.lSpecIBatch, pSlot.vSpecDraft);
        pSlot.lSpecIBatch.clear();

        GGML_ASSERT(vAccepted.size() >= 1);

        const uint32_t cNRollback = pSlot.vSpecDraft.size() + 1 - vAccepted.size();

        const bool cUseCkptTgt = vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
                                 (vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs && cNRollback > mgs_n_rs_seq(vCtxTgt));

        // check for partial draft acceptance
        if (cNRollback > 0) {
          if (cUseCkptTgt) {
            if (vTrace > 0) {
              SLT_INF(pSlot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", vAccepted.size() - 1,
                      pSlot.vSpecDraft.size());
            }

            // partial acceptance is not supported by the context -> truncate the draft and restore the state
            pSlot.vSpecIsReplay = true;
            pSlot.vSpecDraft = std::move(vAccepted);

            const auto &vCkpt = pSlot.vSpecCkpt;

            SLT_DBG(pSlot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", vCkpt.vPosMin,
                    vCkpt.vPosMax, vCkpt.size());

            vCkpt.mLoadTgt(pSlot.vCtxTgt, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            if (pSlot.vCtxDft) {
              vCkpt.mLoadDft(pSlot.vCtxDft, pSlot.vId, MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            pSlot.vMem.mSeqRm(pSlot.vId, vCkpt.vPosMax + 1, -1);

            pSlot.vPrompt.vTokens.mKeepFirst(vCkpt.vNTokens);
            pSlot.vSmpl = std::move(vSmplSave);

            return;
          }
        }

        if (vTrace > 0) {
          SLT_INF(pSlot, "accepted %2zu/%2zu draft tokens\n", vAccepted.size() - 1, cNDraft);
        }

        fCommonSpeculativeAccept(vSpec.get(), pSlot.vId, vAccepted.size() - 1);

        pSlot.vSpecDraft = std::move(vAccepted);
      }

      const int64_t cTNow = ggml_time_us();

      const auto cIds = std::move(pSlot.vSpecDraft);

      size_t vNAccepted = cIds.size() - 1;
      if (pSlot.vSpecIsReplay && vNAccepted > 0) {
        vNAccepted--;
      }
      pSlot.vSpecIsReplay = false;

      pSlot.vTTokenGeneration = std::max<int64_t>(1, cTNow - pSlot.vTStartGeneration) / 1e3;

      // update how many tokens out of those tested were accepted
      pSlot.vNDraftAccepted += vNAccepted;
      pSlot.vNDraftVerifSteps += 1;

      if (pSlot.lNAcceptedPerPos.empty()) {
        pSlot.lNAcceptedPerPos.resize(fCommonSpeculativeNMax(&vParamsBase.vSpeculative), 0);
      }
      for (size_t vI = 0; vI < vNAccepted && vI < pSlot.lNAcceptedPerPos.size(); ++vI) {
        pSlot.lNAcceptedPerPos[vI]++;
      }

      // add accepted tokens to the prompt
      pSlot.vPrompt.vTokens.mKeepFirst(pSlot.vPrompt.mNTokens() - cNDraft);
      pSlot.vPrompt.vTokens.mInsert({cIds.begin(), cIds.end() - 1});

      pSlot.vSampled = cIds.back(); // last accepted token
      SLT_DBG(pSlot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", pSlot.vSampled, cIds.size(),
              cNDraft);

      pSlot.vMem.mSeqRm(pSlot.vId, pSlot.vPrompt.vTokens.mPosNext(), -1);

      for (size_t vI = 0; vI < cIds.size(); ++vI) {
        completion_token_output vResult;

        vResult.vTok = cIds[vI];
        vResult.vTextToSend =
            fCommonTokenToPiece(pSlot.vCtxTgt, vResult.vTok, vAcceptSpecialToken(pSlot, vResult.vTok));
        vResult.vProb = 1.0f; // set later

        // TODO: set result.probs

        pSlot.vNDecoded += 1;

        if (!mProcessToken(vResult, pSlot)) {
          pSlot.mPrintTimings();
          mSendFinalResponse(pSlot);
          vMetrics.mOnPrediction(pSlot);
          pSlot.mRelease();

          return;
        }
      }

      pSlot.mPrintTimingsTg();

      SLT_DBG(pSlot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int)vNAccepted, (int)cNDraft,
              pSlot.vPrompt.mNTokens());
    });
  }

  int mGetSlotNCtx() { return lSlots.back().vNCtx; }

  server_response_reader mGetResponseReader() {
    return server_response_reader(vQueueTasks, vQueueResults, cHttpPollingSeconds);
  }
};

//
// server_context (public API)
//

server_context::server_context() : vImpl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::mLoadModel(common_params &pParams) { return vImpl->mLoadModel(pParams); }

void server_context::mStartLoop() {
  auto &vParams = vImpl->vParamsBase;
  vImpl->vQueueTasks.mStartLoop(vParams.vSleepIdleSeconds * 1000);
}

void server_context::mTerminate() { vImpl->vQueueTasks.mTerminate(); }

mgs_context *server_context::mGetMgsContext() const { return vImpl->vCtxTgt; }

server_response_reader server_context::mGetResponseReader() { return vImpl->mGetResponseReader(); }

server_context_meta server_context::mGetMeta() const {
  auto vBosId = mgs_vocab_bos(vImpl->vVocab);
  auto vEosId = mgs_vocab_eos(vImpl->vVocab);
  auto vBosTokenStr = vBosId != MGS_TOKEN_NULL ? fCommonTokenToPiece(vImpl->vCtxTgt, vBosId, true) : "";
  auto vEosTokenStr = vEosId != MGS_TOKEN_NULL ? fCommonTokenToPiece(vImpl->vCtxTgt, vEosId, true) : "";

  const char *vFtypeName = mgs_ftype_name(mgs_model_ftype(vImpl->vModelTgt));

  return server_context_meta{
      /* build_info             */ std::string(mgs_version()),
      /* model_name             */ vImpl->vModelName,
      /* model_aliases          */ vImpl->sModelAliases,
      /* model_tags             */ vImpl->sModelTags,
      /* model_path             */ vImpl->vParamsBase.vModel.vPath,
      /* has_mtmd               */ vImpl->vMctx != nullptr,
      /* has_inp_image          */ vImpl->vChatParams.vAllowImage,
      /* has_inp_audio          */ vImpl->vChatParams.vAllowAudio,
      /* has_inp_video          */ vImpl->vChatParams.vAllowVideo,
      /* json_ui_settings       */ vImpl->vJsonUiSettings,
      /* slot_n_ctx             */ vImpl->mGetSlotNCtx(),
      /* pooling_type           */ mgs_pooling_type(vImpl->vCtxTgt),

      /* chat_params            */ vImpl->vChatParams,
      /* chat_template_caps     */ fCommonChatTemplatesGetCaps(vImpl->vChatParams.vTmpls.get()),

      /* bos_token_str          */ vBosTokenStr,
      /* eos_token_str          */ vEosTokenStr,
      /* fim_pre_token          */ mgs_vocab_fim_pre(vImpl->vVocab),
      /* fim_sub_token          */ mgs_vocab_fim_suf(vImpl->vVocab),
      /* fim_mid_token          */ mgs_vocab_fim_mid(vImpl->vVocab),
      /* fim_pad_token          */ mgs_vocab_fim_pad(vImpl->vVocab),
      /* fim_rep_token          */ mgs_vocab_fim_rep(vImpl->vVocab),
      /* fim_sep_token          */ mgs_vocab_fim_sep(vImpl->vVocab),

      /* logit_bias_eog         */ vImpl->vParamsBase.vSampling.lLogitBiasEog,

      /* model_vocab_type       */ mgs_vocab_type(vImpl->vVocab),
      /* model_vocab_n_tokens   */ mgs_vocab_n_tokens(vImpl->vVocab),
      /* model_n_ctx_train      */ mgs_model_n_ctx_train(vImpl->vModelTgt),
      /* model_n_embd_inp       */ mgs_model_n_embd(vImpl->vModelTgt),
      /* model_n_params         */ mgs_model_n_params(vImpl->vModelTgt),
      /* model_size             */ mgs_model_size(vImpl->vModelTgt),
      /* model_ftype            */ vFtypeName,
  };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
  server_response_reader vRd;
  server_res_generator(server_queue &pQueueTasks, server_response &pQueueResults, int pSleepIdleSeconds,
                       bool pBypassSleep = false)
      : vRd(pQueueTasks, pQueueResults, cHttpPollingSeconds) {
    // fast path in case sleeping is disabled
    pBypassSleep |= pSleepIdleSeconds < 0;
    if (!pBypassSleep) {
      pQueueTasks.mWaitUntilNoSleep();
    }
  }
  void mOk(const json &pResponseData) {
    vStatus = 200;
    vData = fSafeJsonToStr(pResponseData);
  }
  void mError(const json &pErrorData) {
    vStatus = fJsonValue(pErrorData, "code", 500);
    vData = fSafeJsonToStr({{"error", pErrorData}});
  }
};

void server_context::mSetStateCallback(server_state_callback_t pCallback) {
  vImpl->vCallbackState = std::move(pCallback);
  vImpl->vQueueTasks.mOnSleepingState([this](bool pSleeping) {
    if (pSleeping) {
      vImpl->vCallbackState(cServerStateSleeping, {});
    }
    // for sleeping == false, event is emitted by mLoadModel()
  });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::mHandleCompletionsImpl(const server_http_req &pReq,
                                                                            server_task_type pType, const json &pData,
                                                                            const std::vector<vRawBuffer> &pFiles,
                                                                            task_response_type pResType) {
  GGML_ASSERT(pType == cServerTaskTypeCompletion || pType == cServerTaskTypeInfill);

  auto vRes = mCreateResponse();
  auto vCompletionId = fGenChatcmplid();
  auto &vRd = vRes->vRd;
  auto &vParams = this->vParams;

  vRes->mSetReq(&pReq); // will also set spipe if needed

  int32_t vSsePingInterval = vParams.vSsePingInterval;

  try {
    std::vector<server_task> lTasks;

    const auto &vPrompt = pData.at("prompt");
    // TODO: this log can become very long, put it behind a flag or think about a more compact format
    // SRV_DBG("Prompt: %s\n", vPrompt.is_string() ? vPrompt.get<std::string>().c_str() : vPrompt.dump(2).c_str());

    if (!vParams.path_prompts_log_dir.empty()) {
      const auto cFilePath =
          std::filesystem::path(vParams.path_prompts_log_dir) / fStringFormat("%012" PRId64 ".txt", ggml_time_ms());
      std::ofstream vF(cFilePath);
      if (vF) {
        vF << (vPrompt.is_string() ? vPrompt.get<std::string>().c_str() : vPrompt.dump(2).c_str());
      } else {
        SRV_ERR("failed to create %s\n", cFilePath.string().c_str());
      }
    }

    // process prompt
    std::vector<server_tokens> lInputs;

    if (pResType != cTaskResponseTypeNone && vCtxServer.vMctx != nullptr) {
      // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
      lInputs.push_back(fProcessMtmdPrompt(vCtxServer.vMctx, vPrompt.get<std::string>(), pFiles));
    } else {
      // Everything else, including multimodal completions.
      lInputs = fTokenizeInputPrompts(vCtxServer.vVocab, vCtxServer.vMctx, vPrompt, true, true);
    }

    // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

    // message delimiters for checkpointing
    auto vDelimiters = fCommonChatMsgDelimitersParse(fJsonValue(pData, "message_delimiters", json::array()));
    vDelimiters.mTokenize(vCtxServer.vVocab);

    for (size_t vI = 0; vI < lInputs.size(); vI++) {
      server_task vTask = server_task(pType);

      vTask.vId = vRd.mGetNewId();

      vTask.vTokens = std::move(lInputs[vI]);
      vTask.vParams =
          server_schema::fEvalMgsCmplSchema(vCtxServer.vVocab, vParams, vMeta->vSlotNCtx, vMeta->lLogitBiasEog, pData);

      vTask.vParams.vMessageSpans = vTask.vTokens.mFindMessageSpans(vDelimiters);

      vTask.vIdSlot = fJsonValue(pData, "id_slot", -1);
      vSsePingInterval = vTask.vParams.vSsePingInterval;

      // OAI-compat
      vTask.vParams.vResType = pResType;
      vTask.vParams.vOaicompatCmplId = vCompletionId;
      vTask.vParams.vOaicompatModel = vMeta->vModelName;

      // origin: the Web UI marks its requests (mgs_client); only its streamed OpenAI-style requests receive queue
      // notices
      vTask.vParams.vQueueNotice = vTask.vParams.vFromWebui && vTask.vParams.vStream &&
                                   pResType != cTaskResponseTypeAnthropic && pResType != cTaskResponseTypeOaiResp;

      // prepare child tasks
      if (vTask.vParams.vNCmpl > 1) {
        int vNChildren = vTask.vParams.vNCmpl - 1;
        for (int vJ = 0; vJ < vNChildren; vJ++) {
          vTask.mAddChild(vTask.vId, vRd.mGetNewId());
        }
      }

      lTasks.push_back(std::move(vTask));
    }

    vRd.mPostTasks(std::move(lTasks));
  } catch (const std::exception &vE) {
    vRes->mError(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    return vRes;
  }

  bool vStream = fJsonValue(pData, "stream", false);

  if (!vStream) {
    // non-stream, wait for the results
    auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);
    if (vAllResults.vIsTerminated) {
      return vRes; // connection is closed
    } else if (vAllResults.vError) {
      vRes->mError(vAllResults.vError->to_json());
      return vRes;
    } else {
      json vArr = json::array();
      for (auto &vRes : vAllResults.lResults) {
        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final *>(vRes.get()) != nullptr);
        vArr.push_back(vRes->to_json());
      }
      GGML_ASSERT(!vArr.empty() && "empty results");
      if (vArr.size() == 1) {
        // if single request, return single object instead of array
        vRes->mOk(vArr[0]);
      } else if (pResType == cTaskResponseTypeOaiChat || pResType == cTaskResponseTypeOaiCmpl) {
        // if multiple results in OAI format, we need to re-format them
        json &vChoices = vArr[0]["choices"];
        for (size_t vI = 1; vI < vArr.size(); vI++) {
          vChoices.push_back(std::move(vArr[vI]["choices"][0]));
        }
        vRes->mOk(vArr[0]);
      } else {
        // multi-results, non-OAI compat
        vRes->mOk(vArr);
      }
    }
  } else {
    // in streaming mode, the first error must be treated as non-stream response
    // this is to match the OAI API behavior
    auto vFirstResult = vRd.mNext(pReq.vShouldStop);
    if (vFirstResult == nullptr) {
      GGML_ASSERT(pReq.vShouldStop());
      return vRes; // connection is closed
    }

    if (vFirstResult->mIsError()) {
      vRes->mError(vFirstResult->to_json());
      return vRes;
    }

    // a queue status may come first: the stream starts with it while the task waits for a free slot
    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(vFirstResult.get()) != nullptr ||
                dynamic_cast<server_task_result_cmpl_final *>(vFirstResult.get()) != nullptr ||
                dynamic_cast<server_task_result_queue *>(vFirstResult.get()) != nullptr);

    // next responses are streamed
    // to be sent immediately
    json vFirstResultJson = vFirstResult->to_json();
    if (vFirstResultJson == nullptr) {
      vRes->vData = ""; // simply send HTTP headers and status code
    } else if (pResType == cTaskResponseTypeAnthropic) {
      vRes->vData = fFormatAnthropicSse(vFirstResultJson);
    } else if (pResType == cTaskResponseTypeOaiResp) {
      vRes->vData = fFormatOaiRespSse(vFirstResultJson);
    } else {
      vRes->vData = fFormatOaiSse(vFirstResultJson);
    }
    vRes->vStatus = 200;
    vRes->vContentType = "text/event-stream";
    vRes->mSetNext([vResThis = vRes.get(), pResType, vSsePingInterval](std::string &pOutput) -> bool {
      static auto vFormatError = [](task_response_type pResType, const json &pResJson) {
        if (pResType == cTaskResponseTypeAnthropic) {
          return fFormatAnthropicSse({
              {"event", "error"},
              {"data", pResJson},
          });
        } else {
          return fFormatOaiSse(json{{"error", pResJson}});
        }
      };

      auto vEffectiveShouldStop = [&vResThis]() { return vResThis->mShouldStop(); };

      try {
        if (vEffectiveShouldStop()) {
          SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
          return false; // should_stop condition met
        }

        if (!vResThis->vData.empty()) {
          // flush the first chunk
          pOutput = std::move(vResThis->vData);
          vResThis->vData.clear();
          return true;
        }

        server_response_reader &vRd = vResThis->vRd;

        // check if there is more data
        if (!vRd.mHasNext()) {
          switch (pResType) {
          case cTaskResponseTypeNone:
          case cTaskResponseTypeOaiResp:
          case cTaskResponseTypeAnthropic:
            pOutput = "";
            break;

          default:
            pOutput = "data: [DONE]\n\n";
            break;
          }
          SRV_DBG("%s", "all results received, terminating stream\n");
          return false; // no more data, terminate
        }

        // receive subsequent results
        bool vTimeout = false;
        int64_t vStartTime = ggml_time_ms();
        auto vResult = vRd.mNext([&vTimeout, &vStartTime, vSsePingInterval, &vEffectiveShouldStop]() {
          if (vEffectiveShouldStop()) {
            return true; // should_stop condition met
          } else if (vSsePingInterval > 0 && ggml_time_ms() - vStartTime > (int64_t)vSsePingInterval * 1000) {
            vTimeout = true;
            return true; // timeout
          }
          return false;
        });

        if (vTimeout) {
          // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to
          // send a ping to keep the connection alive
          SRV_DBG("%s", "sending SSE ping\n");
          pOutput = ":\n\n";
          return true;
        }

        if (vResult == nullptr) {
          SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
          GGML_ASSERT(vEffectiveShouldStop());
          return false; // should_stop condition met
        }

        // send the results
        if (vResult->mIsError()) {
          json vResJson = vResult->to_json();
          pOutput = vFormatError(pResType, vResJson);
          SRV_DBG("%s", "error received during streaming, terminating stream\n");
          return false; // terminate on error
        } else {
          GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(vResult.get()) != nullptr ||
                      dynamic_cast<server_task_result_cmpl_final *>(vResult.get()) != nullptr ||
                      dynamic_cast<server_task_result_queue *>(vResult.get()) != nullptr);
          json vResJson = vResult->to_json();
          if (pResType == cTaskResponseTypeAnthropic) {
            pOutput = fFormatAnthropicSse(vResJson);
          } else if (pResType == cTaskResponseTypeOaiResp) {
            pOutput = fFormatOaiRespSse(vResJson);
          } else {
            pOutput = fFormatOaiSse(vResJson);
          }
        }

        // has next data, continue
        return true;

      } catch (const std::exception &vE) {
        json vErrorJson = fFormatErrorResponse(vE.what(), cErrorTypeServer);
        pOutput = vFormatError(pResType, vErrorJson);

        // terminate on exception
        return false;
      }
    });
  }

  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mCreateResponse(bool pBypassSleep) {
  return std::make_unique<server_res_generator>(vQueueTasks, vQueueResults, vParams.vSleepIdleSeconds, pBypassSleep);
}

server_routes::server_routes(const common_params &pParams, server_context &pCtxServer)
    : vParams(pParams), vCtxServer(*pCtxServer.vImpl), vQueueTasks(pCtxServer.vImpl->vQueueTasks),
      vQueueResults(pCtxServer.vImpl->vQueueResults) {
  mInitRoutes();
}

void server_routes::mInitRoutes() {
  // IMPORTANT: all lambda functions must start with mCreateResponse()
  // this is to ensure that the server_res_generator can handle sleeping case correctly

  this->vGetHealth = [this](const server_http_req &) {
    // error and loading states are handled by middleware
    auto vRes = mCreateResponse(true);

    // this endpoint can be accessed during sleeping
    // the next LOC is to avoid someone accidentally use ctx_server
    bool vCtxServer; // do NOT delete this line
    GGML_UNUSED(vCtxServer);

    vRes->mOk({{"status", "ok"}});
    return vRes;
  };

  this->vGetMetrics = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    if (!vParams.vEndpointMetrics) {
      vRes->mError(fFormatErrorResponse("This server does not support metrics endpoint. Start it with `--metrics`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // request slots data using task queue
    {
      server_task vTask(cServerTaskTypeMetrics);
      vTask.vId = vRes->vRd.mGetNewId();
      vRes->vRd.mPostTask(std::move(vTask), true); // high-priority task
    }

    // get the result
    auto vResult = vRes->vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    // TODO: get rid of this dynamic_cast
    auto vResTask = dynamic_cast<server_task_result_metrics *>(vResult.get());
    GGML_ASSERT(vResTask != nullptr);

    // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
    json vAllMetricsDef =
        json{{"counter",
              {{{"name", "prompt_tokens_total"},
                {"help", "Number of prompt tokens processed."},
                {"value", (uint64_t)vResTask->vNPromptTokensProcessedTotal}},
               {{"name", "prompt_seconds_total"},
                {"help", "Prompt process time"},
                {"value", (uint64_t)vResTask->vTPromptProcessingTotal / 1.e3}},
               {{"name", "tokens_predicted_total"},
                {"help", "Number of generation tokens processed."},
                {"value", (uint64_t)vResTask->vNTokensPredictedTotal}},
               {{"name", "tokens_predicted_seconds_total"},
                {"help", "Predict process time"},
                {"value", (uint64_t)vResTask->vTTokensGenerationTotal / 1.e3}},
               {{"name", "n_decode_total"},
                {"help", "Total number of mgs_decode() calls"},
                {"value", vResTask->vNDecodeTotal}},
               {{"name", "n_tokens_max"}, {"help", "Largest observed n_tokens."}, {"value", vResTask->vNTokensMax}},
               {{"name", "spec_decode_num_draft_tokens_total"},
                {"help", "Total draft tokens generated"},
                {"value", vResTask->vNDraftTokensTotal}},
               {{"name", "spec_decode_num_accepted_tokens_total"},
                {"help", "Total draft tokens accepted by the target model"},
                {"value", vResTask->vNDraftAcceptedTotal}},
               {{"name", "spec_decode_num_drafts_total"},
                {"help", "Total speculative decoding verification steps"},
                {"value", vResTask->vNDraftVerifStepsTotal}}}},
             {"gauge",
              {{{"name", "prompt_tokens_seconds"},
                {"help", "Average prompt throughput in tokens/s."},
                {"value", vResTask->vNPromptTokensProcessed
                              ? 1.e3 / vResTask->vTPromptProcessing * vResTask->vNPromptTokensProcessed
                              : 0.}},
               {{"name", "predicted_tokens_seconds"},
                {"help", "Average generation throughput in tokens/s."},
                {"value",
                 vResTask->vNTokensPredicted ? 1.e3 / vResTask->vTTokensGeneration * vResTask->vNTokensPredicted : 0.}},
               {{"name", "requests_processing"},
                {"help", "Number of requests processing."},
                {"value", (uint64_t)vResTask->vNProcessingSlots}},
               {{"name", "requests_deferred"},
                {"help", "Number of requests deferred."},
                {"value", (uint64_t)vResTask->vNTasksDeferred}},
               {{"name", "n_busy_slots_per_decode"},
                {"help", "Average number of busy slots per mgs_decode() call"},
                {"value", (float)vResTask->vNBusySlotsTotal / std::max((float)vResTask->vNDecodeTotal, 1.f)}}}}};

    // MoE expert cache, when one is allocated
    if (fJsonValue(vResTask->vExpertCache, "active", false)) {
      const json &dCache = vResTask->vExpertCache;
      const std::vector<std::array<std::string, 3>> lLCounters = {
          {"moe_cache_steps_total", "Decoding steps executed with the MoE expert cache", "steps"},
          {"moe_cache_hits_total", "Selected experts found in the MoE expert cache", "hits"},
          {"moe_cache_misses_total", "Selected experts missing from the MoE expert cache", "misses"},
          {"moe_cache_uploads_total", "Missing experts uploaded into the MoE expert cache", "uploads"},
          {"moe_cache_cpu_experts_total", "Missing experts computed on the CPU", "cpu_experts"},
          {"moe_cache_prefetch_cache_bytes_total", "Expert prefetch bytes copied from the MoE expert cache",
           "prefetch_cache_bytes"},
          {"moe_cache_prefetch_host_bytes_total", "Expert prefetch bytes uploaded from host memory",
           "prefetch_host_bytes"}};
      const std::vector<std::array<std::string, 3>> lLGauges = {
          {"moe_cache_bytes", "Device memory of the MoE expert cache", "size_bytes"},
          {"moe_cache_slots", "Experts the MoE expert cache can hold", "slots"},
          {"moe_cache_experts", "Experts of the layers served by the MoE expert cache", "experts"},
          {"moe_cache_upload_bytes_per_second", "Measured expert upload bandwidth",
           "upload_bandwidth_bytes_per_second"},
          {"moe_cache_cpu_bytes_per_second", "Estimated CPU expert bandwidth", "cpu_bandwidth_bytes_per_second"}};
      for (const auto &aMetric : lLCounters) {
        vAllMetricsDef["counter"].push_back(
            {{"name", aMetric[0]}, {"help", aMetric[1]}, {"value", dCache.at(aMetric[2])}});
      }
      for (const auto &aMetric : lLGauges) {
        vAllMetricsDef["gauge"].push_back(
            {{"name", aMetric[0]}, {"help", aMetric[1]}, {"value", dCache.at(aMetric[2])}});
      }
    }

    std::stringstream vPrometheus;

    for (const auto &vEl : vAllMetricsDef.items()) {
      const auto &vType = vEl.key();
      const auto &vMetricsDef = vEl.value();

      for (const auto &vMetricDef : vMetricsDef) {
        const std::string cName = vMetricDef.at("name");
        const std::string cHelp = vMetricDef.at("help");

        auto vValue = fJsonValue(vMetricDef, "value", 0.);
        vPrometheus << "# HELP mgs:" << cName << " " << cHelp << "\n"
                    << "# TYPE mgs:" << cName << " " << vType << "\n"
                    << "mgs:" << cName << " " << vValue << "\n";
      }
    }

    // labeled counter: one time series per draft position
    if (!vResTask->lNAcceptedPerPosTotal.empty()) {
      vPrometheus << "# HELP mgs:spec_decode_num_accepted_tokens_per_pos_total"
                     " Accepted tokens per draft position\n"
                  << "# TYPE mgs:spec_decode_num_accepted_tokens_per_pos_total counter\n";
      for (size_t vI = 0; vI < vResTask->lNAcceptedPerPosTotal.size(); vI++) {
        vPrometheus << "mgs:spec_decode_num_accepted_tokens_per_pos_total{position=\"" << vI << "\"} "
                    << vResTask->lNAcceptedPerPosTotal[vI] << "\n";
      }
    }

    vRes->dHeaders["Process-Start-Time-Unix"] = std::to_string(vResTask->vTStart);
    vRes->vContentType = "text/plain; version=0.0.4";
    vRes->vStatus = 200;
    vRes->vData = vPrometheus.str();
    return vRes;
  };

  this->vGetSlots = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    if (!vParams.vEndpointSlots) {
      vRes->mError(fFormatErrorResponse("This server does not support slots endpoint. Start it with `--slots`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // request slots data using task queue
    {
      server_task vTask(cServerTaskTypeMetrics);
      vTask.vId = vRes->vRd.mGetNewId();
      vRes->vRd.mPostTask(std::move(vTask), true); // high-priority task
    }

    // get the result
    auto vResult = vRes->vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    // TODO: get rid of this dynamic_cast
    auto *vResTask = dynamic_cast<server_task_result_metrics *>(vResult.get());
    GGML_ASSERT(vResTask != nullptr);

    // optionally return "fail_on_no_slot" error
    if (!pReq.mGetParam("fail_on_no_slot").empty()) {
      if (vResTask->vNIdleSlots == 0) {
        vRes->mError(fFormatErrorResponse("no slot available", cErrorTypeUnavailable));
        return vRes;
      }
    }

    vRes->mOk(vResTask->vSlotsData);
    return vRes;
  };

  this->vPostSlots = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    if (vParams.vSlotSavePath.empty()) {
      vRes->mError(fFormatErrorResponse("This server does not support slots action. Start it with `--slot-save-path`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    std::string vIdSlotStr = pReq.mGetParam("id_slot");

    int vIdSlot;
    try {
      vIdSlot = std::stoi(vIdSlotStr);
    } catch (const std::exception &) {
      vRes->mError(fFormatErrorResponse("Invalid slot ID", cErrorTypeInvalidRequest));
      return vRes;
    }

    std::string vAction = pReq.mGetParam("action");

    if (vAction == "save") {
      return mHandleSlotsSave(pReq, vIdSlot);
    }
    if (vAction == "restore") {
      return mHandleSlotsRestore(pReq, vIdSlot);
    }
    if (vAction == "erase") {
      return mHandleSlotsErase(pReq, vIdSlot);
    }

    vRes->mError(fFormatErrorResponse("Invalid action", cErrorTypeInvalidRequest));
    return vRes;
  };

  this->vGetProps = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);

    // this endpoint can be accessed during sleeping
    // the next LOC is to avoid someone accidentally use ctx_server
    bool vCtxServer; // do NOT delete this line
    GGML_UNUSED(vCtxServer);

    task_params vTparams;
    vTparams.vSampling = vParams.vSampling;
    json vDefaultGenerationSettingsForProps = json{
        {"params", vTparams.to_json(true)},
        {"n_ctx", vMeta->vSlotNCtx},
    };

    std::string vTmplDefault = fCommonChatTemplatesSource(vMeta->vChatParams.vTmpls.get(), "");
    std::string vTmplTools = fCommonChatTemplatesSource(vMeta->vChatParams.vTmpls.get(), "tool_use");

    json vProps = {
        {"default_generation_settings", vDefaultGenerationSettingsForProps},
        {"total_slots", vParams.vNParallel},
        {"model_alias", vMeta->vModelName},
        {"model_ftype", vMeta->vModelFtype},
        {"model_path", vMeta->vModelPath},
        {"modalities",
         json{
             {"vision", vMeta->vHasInpImage},
             {"video", vMeta->vHasInpVideo},
             {"audio", vMeta->vHasInpAudio},
         }},
        {"media_marker", fGetMediaMarker()},
        {"endpoint_slots", vParams.vEndpointSlots},
        {"endpoint_props", vParams.vEndpointProps},
        {"endpoint_metrics", vParams.vEndpointMetrics},
        {"ui", vParams.vUi},
        {"ui_settings", vMeta->vJsonUiSettings},
        {"chat_template", vTmplDefault},
        {"chat_template_caps", vMeta->dChatTemplateCaps},
        {"bos_token", vMeta->vBosTokenStr},
        {"eos_token", vMeta->vEosTokenStr},
        {"build_info", vMeta->vBuildInfo},
        {"is_sleeping", vQueueTasks.mIsSleeping()},
        {"cors_proxy_enabled", vParams.vUiMcpProxy},
        {"shell_mcp_server", vParams.vShellMcpServer},
    };
    if (!vTmplTools.empty()) {
      vProps["chat_template_tool_use"] = vTmplTools;
    }
    vRes->mOk(vProps);
    return vRes;
  };

  this->vPostProps = [this](const server_http_req &) {
    auto vRes = mCreateResponse();
    if (!vParams.vEndpointProps) {
      vRes->mError(fFormatErrorResponse(
          "This server does not support changing global properties. Start it with `--props`", cErrorTypeNotSupported));
      return vRes;
    }
    // update any props here

    vRes->mOk({{"success", true}});
    return vRes;
  };

  this->vPostInfill = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    // check model compatibility
    std::string vErr;
    if (mgs_vocab_fim_pre(vCtxServer.vVocab) == MGS_TOKEN_NULL) {
      vErr += "prefix token is missing. ";
    }
    if (mgs_vocab_fim_suf(vCtxServer.vVocab) == MGS_TOKEN_NULL) {
      vErr += "suffix token is missing. ";
    }
    if (mgs_vocab_fim_mid(vCtxServer.vVocab) == MGS_TOKEN_NULL) {
      vErr += "middle token is missing. ";
    }
    if (!vErr.empty()) {
      vRes->mError(fFormatErrorResponse(fStringFormat("Infill is not supported by this model: %s", vErr.c_str()),
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // validate input
    json vData = json::parse(pReq.vBody);
    if (vData.contains("prompt") && !vData.at("prompt").is_string()) {
      // prompt is optional
      vRes->mError(fFormatErrorResponse("\"prompt\" must be a string", cErrorTypeInvalidRequest));
    }

    if (!vData.contains("input_prefix")) {
      vRes->mError(fFormatErrorResponse("\"input_prefix\" is required", cErrorTypeInvalidRequest));
    }

    if (!vData.contains("input_suffix")) {
      vRes->mError(fFormatErrorResponse("\"input_suffix\" is required", cErrorTypeInvalidRequest));
    }

    if (vData.contains("input_extra") && !vData.at("input_extra").is_array()) {
      // input_extra is optional
      vRes->mError(fFormatErrorResponse("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
                                        cErrorTypeInvalidRequest));
      return vRes;
    }

    json vInputExtra = fJsonValue(vData, "input_extra", json::array());
    for (const auto &vChunk : vInputExtra) {
      // { "text": string, "filename": string }
      if (!vChunk.contains("text") || !vChunk.at("text").is_string()) {
        vRes->mError(fFormatErrorResponse("extra_context chunk must contain a \"text\" field with a string value",
                                          cErrorTypeInvalidRequest));
        return vRes;
      }
      // filename is optional
      if (vChunk.contains("filename") && !vChunk.at("filename").is_string()) {
        vRes->mError(fFormatErrorResponse("extra_context chunk's \"filename\" field must be a string",
                                          cErrorTypeInvalidRequest));
        return vRes;
      }
    }
    vData["input_extra"] = vInputExtra; // default to empty array if it's not exist

    std::string vPrompt = fJsonValue(vData, "prompt", std::string());
    std::vector<server_tokens> lTokenizedPrompts =
        fTokenizeInputPrompts(vCtxServer.vVocab, vCtxServer.vMctx, vPrompt, false, true);
    SRV_DBG("creating infill tasks, n_prompts = %d\n", (int)lTokenizedPrompts.size());
    vData["prompt"] = fFormatPromptInfill(vCtxServer.vVocab, vData.at("input_prefix"), vData.at("input_suffix"),
                                          vData.at("input_extra"), vParams.vNBatch, vParams.vNPredict, vMeta->vSlotNCtx,
                                          vParams.vSpmInfill,
                                          lTokenizedPrompts[0].mGetTokens() // TODO: this could maybe be multimodal.
    );

    std::vector<vRawBuffer> lFiles; // dummy
    return mHandleCompletionsImpl(pReq, cServerTaskTypeInfill, vData, lFiles,
                                  cTaskResponseTypeNone); // infill is not OAI compatible
  };

  this->vPostCompletions = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles; // dummy
    const json cBody = json::parse(pReq.vBody);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, cBody, lFiles, cTaskResponseTypeNone);
  };

  this->vPostCompletionsOai = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles; // dummy
    const json cBody = json::parse(pReq.vBody);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, cBody, lFiles, cTaskResponseTypeOaiCmpl);
  };

  this->vPostChatCompletions = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles;
    json vBody = json::parse(pReq.vBody);
    json vBodyParsed = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiChat);
  };

  this->vPostChatCompletionsTok = [this](const server_http_req &pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeOaiChat);
  };

  this->vPostControl = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    const json cBody = json::parse(pReq.vBody);

    const std::string cCmplId = fJsonValue(cBody, "id", std::string());
    const std::string cAction = fJsonValue(cBody, "action", std::string());
    if (cCmplId.empty()) {
      vRes->mError(fFormatErrorResponse("missing completion id", cErrorTypeInvalidRequest));
      return vRes;
    }
    if (cAction != "reasoning_end") {
      vRes->mError(fFormatErrorResponse("unknown control action", cErrorTypeInvalidRequest));
      return vRes;
    }

    auto &vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeControl);
      vTask.vId = vRd.mGetNewId();
      vTask.vParams.vControlCmplId = cCmplId;
      vTask.vParams.vControlAction = cAction;
      vRd.mPostTask(std::move(vTask));
    }

    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }
    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }
    vRes->mOk(vResult->to_json());
    return vRes;
  };

  this->vPostResponsesOai = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles;
    json vBody = fServerChatConvertResponsesToChatcmpl(json::parse(pReq.vBody));
    SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.dump().c_str());
    json vBodyParsed = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiResp);
  };

  this->vPostResponsesTokOai = [this](const server_http_req &pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeOaiResp);
  };

  this->vPostTranscriptionsOai = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();

    if (!vMeta->vHasMtmd || !vMeta->vChatParams.vAllowAudio) {
      vRes->mError(fFormatErrorResponse("The current model does not support audio input.", cErrorTypeNotSupported));
      return vRes;
    }

    std::vector<vRawBuffer> lFiles;
    json vBody =
        fConvertTranscriptionsToChatcmpl(json::parse(pReq.vBody), vMeta->vChatParams.vTmpls.get(), pReq.dFiles, lFiles);
    SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.dump().c_str());
    json vBodyParsed = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiAsr);
  };

  this->vPostAnthropicMessages = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles;
    json vBody = fServerChatConvertAnthropicToOai(json::parse(pReq.vBody));
    SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.dump().c_str());
    json vBodyParsed = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeAnthropic);
  };

  this->vPostAnthropicCountTokens = [this](const server_http_req &pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeAnthropic);
  };

  // same with handle_chat_completions, but without inference part
  this->vPostApplyTemplate = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    std::vector<vRawBuffer> lFiles; // dummy, unused
    json vBody = json::parse(pReq.vBody);
    json vData = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
    vRes->mOk({{"prompt", std::move(vData.at("prompt"))}});
    return vRes;
  };

  this->vGetModels = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);

    // this endpoint can be accessed during sleeping
    // the next LOC is to avoid someone accidentally use ctx_server
    bool vCtxServer; // do NOT delete this line
    GGML_UNUSED(vCtxServer);

    json vModels = {{"models",
                     {{{"name", vMeta->vModelName},
                       {"model", vMeta->vModelName},
                       {"modified_at", ""},
                       {"size", ""},
                       {"digest", ""}, // dummy value, mgs does not support managing model file's hash
                       {"type", "model"},
                       {"description", ""},
                       {"tags", {""}},
                       {"capabilities", vMeta->vHasMtmd ? json({"completion", "multimodal"}) : json({"completion"})},
                       {"parameters", ""},
                       {"details",
                        {{"parent_model", ""},
                         {"format", "gguf"},
                         {"family", ""},
                         {"families", {""}},
                         {"parameter_size", ""},
                         {"quantization_level", ""}}}}}},
                    {"object", "list"},
                    {"data",
                     {
                         mGetModelInfo(),
                     }}};

    vRes->mOk(vModels);
    return vRes;
  };

  this->vPostTokenize = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    const json cBody = json::parse(pReq.vBody);
    json vTokensResponse = json::array();
    if (cBody.count("content") != 0) {
      const bool cAddSpecial = fJsonValue(cBody, "add_special", false);
      const bool cParseSpecial = fJsonValue(cBody, "parse_special", true);
      const bool cWithPieces = fJsonValue(cBody, "with_pieces", false);

      mgs_tokens vTokens = fTokenizeMixed(vCtxServer.vVocab, cBody.at("content"), cAddSpecial, cParseSpecial);

      if (cWithPieces) {
        for (const auto &vToken : vTokens) {
          std::string vPiece = fCommonTokenToPiece(vCtxServer.vVocab, vToken);
          json vPieceJson;

          // Check if the piece is valid UTF-8
          if (fIsValidUtf8(vPiece)) {
            vPieceJson = vPiece;
          } else {
            // If not valid UTF-8, store as array of byte values
            vPieceJson = json::array();
            for (unsigned char vC : vPiece) {
              vPieceJson.push_back(static_cast<int>(vC));
            }
          }

          vTokensResponse.push_back({{"id", vToken}, {"piece", vPieceJson}});
        }
      } else {
        vTokensResponse = vTokens;
      }
    }

    vRes->mOk(json{{"tokens", std::move(vTokensResponse)}});
    return vRes;
  };

  this->vPostDetokenize = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    const json cBody = json::parse(pReq.vBody);

    std::string vContent;
    if (cBody.count("tokens") != 0) {
      const mgs_tokens cTokens = cBody.at("tokens");
      vContent = fTokensToStr(vCtxServer.vVocab, cTokens);
    }

    vRes->mOk(json{{"content", std::move(vContent)}});
    return vRes;
  };

  this->vPostEmbeddings = [this](const server_http_req &pReq) {
    return mHandleEmbeddingsImpl(pReq, cTaskResponseTypeNone);
  };

  this->vPostEmbeddingsOai = [this](const server_http_req &pReq) {
    return mHandleEmbeddingsImpl(pReq, cTaskResponseTypeOaiEmbd);
  };

  this->vPostRerank = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    if (!vParams.vEmbedding || vParams.vPoolingType != MGS_POOLING_TYPE_RANK) {
      vRes->mError(fFormatErrorResponse("This server does not support reranking. Start it with `--rerank`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    const json cBody = json::parse(pReq.vBody);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool vIsTeiFormat = cBody.contains("texts");

    json vQuery;
    if (cBody.count("query") == 1) {
      vQuery = cBody.at("query");
      if (!vQuery.is_string()) {
        vRes->mError(fFormatErrorResponse("\"query\" must be a string", cErrorTypeInvalidRequest));
        return vRes;
      }
    } else {
      vRes->mError(fFormatErrorResponse("\"query\" must be provided", cErrorTypeInvalidRequest));
      return vRes;
    }

    std::vector<std::string> lDocuments =
        fJsonValue(cBody, "documents", fJsonValue(cBody, "texts", std::vector<std::string>()));
    if (lDocuments.empty()) {
      vRes->mError(fFormatErrorResponse("\"documents\" must be a non-empty string array", cErrorTypeInvalidRequest));
      return vRes;
    }

    int vTopN = fJsonValue(cBody, "top_n", (int)lDocuments.size());

    // create and queue the task
    json vResponses = json::array();
    auto &vRd = vRes->vRd;
    {
      std::vector<server_task> lTasks;
      lTasks.reserve(lDocuments.size());
      for (size_t vI = 0; vI < lDocuments.size(); vI++) {
        auto vTmp =
            fFormatPromptRerank(vCtxServer.vModelTgt, vCtxServer.vVocab, vCtxServer.vMctx, vQuery, lDocuments[vI]);
        server_task vTask = server_task(cServerTaskTypeRerank);
        vTask.vId = vRd.mGetNewId();
        vTask.vTokens = std::move(vTmp);
        lTasks.push_back(std::move(vTask));
      }
      vRd.mPostTasks(std::move(lTasks));
    }

    // wait for the results
    auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);

    // collect results
    if (vAllResults.vIsTerminated) {
      return vRes; // connection is closed
    } else if (vAllResults.vError) {
      vRes->mError(vAllResults.vError->to_json());
      return vRes;
    } else {
      for (auto &vRes : vAllResults.lResults) {
        GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(vRes.get()) != nullptr);
        vResponses.push_back(vRes->to_json());
      }
    }

    // write JSON response
    json vRoot = fFormatResponseRerank(cBody, vMeta->vModelName, vResponses, vIsTeiFormat, lDocuments, vTopN);

    vRes->mOk(vRoot);
    return vRes;
  };

  this->vGetLoraAdapters = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();

    auto &vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeGetLora);
      vTask.vId = vRd.mGetNewId();
      vRd.mPostTask(std::move(vTask));
    }

    // get the result
    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_get_lora *>(vResult.get()) != nullptr);
    vRes->mOk(vResult->to_json());
    return vRes;
  };

  // MoE expert cache: GET reports its state, POST {"size_mib": N} requests a new size
  // (-1 automatic, 0 disabled) that the next request applies.
  auto fExpertCacheHandler = [this](const server_http_req &pReq, int64_t pSizeMib) {
    auto vRes = mCreateResponse();
    auto &vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeExpertCache);
      vTask.vId = vRd.mGetNewId();
      vTask.vExpertCacheSizeMib = pSizeMib;
      vRd.mPostTask(std::move(vTask));
    }

    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }
    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_expert_cache *>(vResult.get()) != nullptr);
    vRes->mOk(vResult->to_json());
    return vRes;
  };

  this->vGetExpertCache = [fExpertCacheHandler](const server_http_req &pReq) { return fExpertCacheHandler(pReq, -2); };

  // Monitor: memory by purpose on every GPU and in RAM, plus CPU and GPU usage. The handlers read
  // the snapshot published by the main loop and the system counters, never the task queue.
  this->vGetMonitor = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);
    vRes->mOk(fMonitorReport({vCtxServer.mGetMonitorSnapshot()}));
    return vRes;
  };

  this->vGetMonitorMemory = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);
    vRes->mOk(json{{"models", json::array({vCtxServer.mGetMonitorSnapshot()})}});
    return vRes;
  };

  this->vGetMonitorUsage = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);
    vRes->mOk(fMonitorUsageJson());
    return vRes;
  };

  this->vPostExpertCache = [this, fExpertCacheHandler](const server_http_req &pReq) {
    const json vBody = json::parse(pReq.vBody);
    if (!vBody.is_object() || !vBody.contains("size_mib") || !vBody.at("size_mib").is_number_integer() ||
        vBody.at("size_mib").get<int64_t>() < -1) {
      auto vRes = mCreateResponse();
      vRes->mError(fFormatErrorResponse("size_mib must be an integer: MiB, -1 for automatic or 0 to disable",
                                        cErrorTypeInvalidRequest));
      return vRes;
    }
    return fExpertCacheHandler(pReq, vBody.at("size_mib").get<int64_t>());
  };

  this->vPostLoraAdapters = [this](const server_http_req &pReq) {
    auto vRes = mCreateResponse();
    const json cBody = json::parse(pReq.vBody);
    if (!cBody.is_array()) {
      vRes->mError(fFormatErrorResponse("Request body must be an array", cErrorTypeInvalidRequest));
      return vRes;
    }

    auto &vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeSetLora);
      vTask.vId = vRd.mGetNewId();
      vTask.dSetLora = fParseLoraRequest(cBody);
      vRd.mPostTask(std::move(vTask));
    }

    // get the result
    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_apply_lora *>(vResult.get()) != nullptr);
    vRes->mOk(vResult->to_json());
    return vRes;
  };
}

json server_routes::mGetModelInfo() const {
  return json{
      {"id", vMeta->vModelName},
      {"aliases", vMeta->sModelAliases},
      {"tags", vMeta->sModelTags},
      {"object", "model"},
      {"created", std::time(0)},
      {"owned_by", "mgs"},
      {"meta",
       {
           {"vocab_type", vMeta->vModelVocabType},
           {"n_vocab", vMeta->vModelVocabNTokens},
           {"n_ctx", vMeta->vSlotNCtx},
           {"n_ctx_train", vMeta->vModelNCtxTrain},
           {"n_embd", vMeta->vModelNEmbdInp},
           {"n_params", vMeta->vModelNParams},
           {"size", vMeta->vModelSize},
           {"ftype", vMeta->vModelFtype},
       }},
  };
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsSave(const server_http_req &pReq, int pIdSlot) {
  auto vRes = mCreateResponse();
  const json cRequestData = json::parse(pReq.vBody);
  std::string vFilename = cRequestData.at("filename");
  if (!fFsValidateFilename(vFilename)) {
    vRes->mError(fFormatErrorResponse("Invalid filename", cErrorTypeInvalidRequest));
    return vRes;
  }
  std::string vFilepath = vParams.vSlotSavePath + vFilename;

  auto &vRd = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotSave);
    vTask.vId = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot = pIdSlot;
    vTask.vSlotAction.vFilename = vFilename;
    vTask.vSlotAction.vFilepath = vFilepath;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsRestore(const server_http_req &pReq, int pIdSlot) {
  auto vRes = mCreateResponse();
  const json cRequestData = json::parse(pReq.vBody);
  std::string vFilename = cRequestData.at("filename");
  if (!fFsValidateFilename(vFilename)) {
    vRes->mError(fFormatErrorResponse("Invalid filename", cErrorTypeInvalidRequest));
    return vRes;
  }
  std::string vFilepath = vParams.vSlotSavePath + vFilename;

  auto &vRd = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotRestore);
    vTask.vId = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot = pIdSlot;
    vTask.vSlotAction.vFilename = vFilename;
    vTask.vSlotAction.vFilepath = vFilepath;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load *>(vResult.get()) != nullptr);
  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsErase(const server_http_req &pReq, int pIdSlot) {
  auto vRes = mCreateResponse();
  auto &vRd = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotErase);
    vTask.vId = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot = pIdSlot;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  GGML_ASSERT(dynamic_cast<server_task_result_slot_erase *>(vResult.get()) != nullptr);
  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleEmbeddingsImpl(const server_http_req &pReq,
                                                                           task_response_type pResType) {
  auto vRes = mCreateResponse();
  if (!vParams.vEmbedding) {
    vRes->mError(fFormatErrorResponse("This server does not support embeddings. Start it with `--embedding`",
                                      cErrorTypeNotSupported));
    return vRes;
  }

  if (pResType != cTaskResponseTypeNone && vMeta->vPoolingType == MGS_POOLING_TYPE_NONE) {
    vRes->mError(fFormatErrorResponse("Pooling type 'none' is not OAI compatible. Please use a different pooling type",
                                      cErrorTypeInvalidRequest));
    return vRes;
  }

  const json cBody = json::parse(pReq.vBody);

  // for the shape of input, see fTokenizeInputPrompts()
  if (!cBody.contains("input")) {
    vRes->mError(fFormatErrorResponse("\"input\" must be provided", cErrorTypeInvalidRequest));
    return vRes;
  }
  const json cPrompt = cBody.at("input");

  bool vUseBase64 = false;
  if (cBody.count("encoding_format") != 0) {
    const std::string &vFormat = cBody.at("encoding_format");
    if (vFormat == "base64") {
      vUseBase64 = true;
    } else if (vFormat != "float") {
      vRes->mError(fFormatErrorResponse("The format to return the embeddings in. Can be either float or base64",
                                        cErrorTypeInvalidRequest));
      return vRes;
    }
  }

  auto vTokenizedPrompts = fTokenizeInputPrompts(vCtxServer.vVocab, vCtxServer.vMctx, cPrompt, true, true);
  for (const auto &vTokens : vTokenizedPrompts) {
    // this check is necessary for models that do not add BOS token to the input
    if (vTokens.empty()) {
      vRes->mError(fFormatErrorResponse("Input content cannot be empty", cErrorTypeInvalidRequest));
      return vRes;
    }
  }

  int vEmbdNormalize = vParams.vEmbdNormalize;
  if (cBody.count("embd_normalize") != 0) {
    vEmbdNormalize = cBody.at("embd_normalize");
    if (vMeta->vPoolingType == MGS_POOLING_TYPE_NONE) {
      SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", vMeta->vPoolingType);
    }
  }

  // create and queue the task
  json vResponses = json::array();
  auto &vRd = vRes->vRd;
  {
    std::vector<server_task> lTasks;
    for (size_t vI = 0; vI < vTokenizedPrompts.size(); vI++) {
      server_task vTask = server_task(cServerTaskTypeEmbedding);

      vTask.vId = vRd.mGetNewId();
      vTask.vTokens = std::move(vTokenizedPrompts[vI]);

      // OAI-compat
      vTask.vParams.vResType = pResType;
      vTask.vParams.vEmbdNormalize = vEmbdNormalize;

      lTasks.push_back(std::move(vTask));
    }
    vRd.mPostTasks(std::move(lTasks));
  }

  // wait for the results
  auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);

  // collect results
  if (vAllResults.vIsTerminated) {
    return vRes; // connection is closed
  } else if (vAllResults.vError) {
    vRes->mError(vAllResults.vError->to_json());
    return vRes;
  } else {
    for (auto &vRes : vAllResults.lResults) {
      GGML_ASSERT(dynamic_cast<server_task_result_embd *>(vRes.get()) != nullptr);
      vResponses.push_back(vRes->to_json());
    }
  }

  // write JSON response
  json vRoot = pResType == cTaskResponseTypeOaiEmbd
                   ? fFormatEmbeddingsResponseOaicompat(cBody, vMeta->vModelName, vResponses, vUseBase64)
                   : json(vResponses);
  vRes->mOk(vRoot);
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleCountTokens(const server_http_req &pReq,
                                                                        task_response_type pResType) {
  auto vRes = mCreateResponse();
  std::vector<vRawBuffer> lFiles;
  json vBody = json::parse(pReq.vBody);
  bool vIsOai = false;

  switch (pResType) {
  case cTaskResponseTypeOaiChat: {
    vIsOai = true;
  } break;
  case cTaskResponseTypeOaiResp: {
    vIsOai = true;
    vBody = fServerChatConvertResponsesToChatcmpl(vBody);
  } break;
  case cTaskResponseTypeAnthropic: {
    vBody = fServerChatConvertAnthropicToOai(vBody);
  } break;
  default:
    vRes->mError(fFormatErrorResponse("invalid res_type", cErrorTypeInvalidRequest));
    return vRes;
  }

  json vBodyParsed = fOaicompatChatParamsParse(vBody, vMeta->vChatParams, lFiles);
  json vPrompt = vBodyParsed.at("prompt");
  // SRV_DBG("prompt = %s\n", vPrompt.dump().c_str());

  // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
  size_t vNTokens;
  if (vCtxServer.vMctx != nullptr) {
    if (!vPrompt.is_string()) {
      throw std::runtime_error("for mtmd, input prompt must be a string.");
    }
    vNTokens = fProcessMtmdPrompt(vCtxServer.vMctx, vPrompt.get<std::string>(), lFiles, true).size();
  } else {
    vNTokens = fTokenizeMixed(vCtxServer.vVocab, vPrompt, true, true).size();
  }

  json vResponse = {{"input_tokens", static_cast<int64_t>(vNTokens)}};
  if (vIsOai) {
    vResponse["object"] = "response.input_tokens";
  }
  vRes->mOk(vResponse);
  return vRes;
}
