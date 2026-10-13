#pragma once

#include "common.h"
#include "mgs.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum server_task_type {
  cServerTaskTypeCompletion,
  cServerTaskTypeEmbedding,
  cServerTaskTypeRerank,
  cServerTaskTypeInfill,
  cServerTaskTypeCancel,
  cServerTaskTypeControl,
  cServerTaskTypeNextResponse,
  cServerTaskTypeMetrics,
  cServerTaskTypeSlotSave,
  cServerTaskTypeSlotRestore,
  cServerTaskTypeSlotErase,
  cServerTaskTypeGetLora,
  cServerTaskTypeSetLora,
  cServerTaskTypeExpertCache,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
  cTaskResponseTypeNone, // mgs native format
  cTaskResponseTypeOaiChat,
  cTaskResponseTypeOaiCmpl,
  cTaskResponseTypeOaiResp,
  cTaskResponseTypeOaiAsr, // transcriptions API
  cTaskResponseTypeOaiEmbd,
  cTaskResponseTypeAnthropic,
};

enum stop_type {
  cStopTypeNone,
  cStopTypeEos,
  cStopTypeWord,
  cStopTypeLimit,
};

struct task_params {
  bool vStream = false;
  bool vIncludeUsage = false;

  // origin of the request: true when the Web UI sent it ("mgs_client": "webui"), false for API clients
  bool vFromWebui = false;
  // stream queue status events ({"mgs_queue": {...}}) while the task waits for a free slot; only streamed Web UI
  // requests ask for them, so API clients never receive anything new
  bool vQueueNotice = false;
  bool vCachePrompt = true; // remember the prompt to avoid reprocessing all prompt
  bool vReturnTokens = false;
  bool vReturnProgress = false;

  int32_t vSsePingInterval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

  int32_t vNKeep = 0; // number of tokens to keep from initial prompt
  int32_t vNDiscard =
      0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
  int32_t vNPredict = -1; // new tokens to predict
  int32_t vNIndent = 0;   // minimum line indentation for the generated text in number of whitespace characters
  int32_t vNCmpl = 1;     // number of completions to generate from this prompt

  int32_t vNCacheReuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

  int64_t vTMaxPromptMs = -1;  // TODO: implement
  int64_t vTMaxPredictMs = -1; // if positive, limit the generation phase to this time limit

  std::map<int, float> dLora; // mapping adapter ID -> scale

  std::vector<std::string> lAntiprompt;
  std::vector<std::string> lResponseFields;

  bool vTimingsPerToken = false;
  bool vPostSamplingProbs = false;

  struct common_params_sampling vSampling;
  struct common_params_speculative vSpeculative;

  // response formatting
  bool vVerbose = false;
  task_response_type vResType = cTaskResponseTypeNone;
  std::string vOaicompatModel;
  std::string vOaicompatCmplId;

  // realtime control (SERVER_TASK_TYPE_CONTROL)
  std::string vControlAction;
  std::string vControlCmplId;

  // per-request parameters for chat parsing
  common_chat_parser_params vChatParserParams;

  // message spans for checkpointing
  common_chat_msg_spans vMessageSpans;

  // Embeddings
  int32_t vEmbdNormalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

  json mFormatLogitBias(const std::vector<mgs_logit_bias> &pLogitBias) const;
  json to_json(bool pOnlyMetrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
  // tracking diffs for partial tool calls
  std::vector<common_chat_msg_diff> lDiffs;
  common_chat_parser_params vChatParserParams;
  common_chat_msg vChatMsg;
  std::string vGeneratedText; // append new chunks of generated text here
  std::vector<std::string> lGeneratedToolCallIds;
  std::unordered_set<size_t> sSentToolCallNames;

  // for OpenAI Responses and Anthropic streaming API:
  // track output item / content block state across chunks
  bool vThinkingBlockStarted = false;
  bool vTextBlockStarted = false;

  // for OpenAI Responses streaming API
  bool vOaiRespCreated = false;
  const std::string cOaiRespId;
  const std::string cOaiRespReasoningId;
  const std::string cOaiRespMessageId;
  std::string vOaiRespFcId; // function call ID for current args delta

  task_result_state(const common_chat_parser_params &pChatParserParams);

  // parse partial tool calls and update the internal state
  common_chat_msg mUpdateChatMsg(const std::string &pTextAdded, bool pIsPartial,
                                 std::vector<common_chat_msg_diff> &pDiffs, bool pFilterToolCalls = false);
};

struct server_task {
  int vId = -1; // to be filled by server_queue

  // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
  size_t vIndex = 0; // used when there are multiple prompts (batch request)

  // used by SERVER_TASK_TYPE_CANCEL
  int vIdTarget = -1;
  int vIdSlot = -1;

  // true once the task has waited in the queue for a free slot
  bool vWasQueued = false;

  // used by parallel sampling (multiple completions from same prompt)
  int vIdParent = -1;
  // temporary store of child tasks for scheduling
  // note: accessing to elements is invalid after the task is moved to server_slot
  std::vector<server_task> lChildTasks;

  // used by SERVER_TASK_TYPE_INFERENCE
  task_params vParams;
  server_tokens vTokens;

  // only used by CLI, this allow tokenizing CLI inputs on server side
  // we need this because mtmd_context and vocab are not accessible outside of server_context
  bool vCli = false;
  std::string vCliPrompt;
  std::vector<vRawBuffer> lCliFiles;

  server_task_type vType;

  // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
  struct vSlotAction {
    int vIdSlot;
    std::string vFilename;
    std::string vFilepath;
  };
  vSlotAction vSlotAction;

  // used by SERVER_TASK_TYPE_METRICS
  bool vMetricsResetBucket = false;

  // used by SERVER_TASK_TYPE_SET_LORA
  std::map<int, float> dSetLora; // mapping adapter ID -> scale

  // used by SERVER_TASK_TYPE_EXPERT_CACHE: requested size in MiB (-1 automatic, 0 disabled),
  // -2 only reports the current state
  int64_t vExpertCacheSizeMib = -2;

  server_task() = default;

  server_task(server_task_type pType) : vType(pType) {}

  int32_t mNTokens() const { return vTokens.size(); }

  bool mNeedEmbd() const {
    switch (vType) {
    case cServerTaskTypeEmbedding:
    case cServerTaskTypeRerank:
      return true;
    default:
      return false;
    }
  }

  bool mNeedLogits() const {
    switch (vType) {
    case cServerTaskTypeCompletion:
    case cServerTaskTypeInfill:
      return true;
    default:
      return false;
    }
  }

  bool mNeedSampling() const {
    switch (vType) {
    case cServerTaskTypeCompletion:
    case cServerTaskTypeInfill:
      return true;
    default:
      return false;
    }
  }

  // utility function
  static std::unordered_set<int> mGetListId(const std::vector<server_task> &pTasks) {
    std::unordered_set<int> sIds(pTasks.size());
    for (size_t vI = 0; vI < pTasks.size(); vI++) {
      sIds.insert(pTasks[vI].vId);
      for (auto &vChild : pTasks[vI].lChildTasks) {
        sIds.insert(vChild.vId);
      }
    }
    return sIds;
  }

  void mAddChild(int pIdParent, int pIdChild) {
    server_task vCopy;

    vCopy.vId = pIdChild;
    vCopy.vIdParent = pIdParent;
    vCopy.vParams = vParams;
    vCopy.vType = vType;
    vCopy.vTokens = vTokens.mClone();
    vCopy.vIdSlot = -1; // child tasks cannot specify slot

    // use different sampling seed for each child
    if (vCopy.vParams.vSampling.vSeed != MGS_DEFAULT_SEED) {
      vCopy.vParams.vSampling.vSeed += (uint32_t)lChildTasks.size() + 1;
    }

    lChildTasks.push_back(std::move(vCopy));
  }

  // the task will be moved into queue, then onto slots
  // however, the state must be kept by caller (e.g., HTTP thread)
  task_result_state mCreateState() const { return task_result_state(vParams.vChatParserParams); }

  bool mIsParent() const { return lChildTasks.size() > 0; }

  bool mIsChild() const { return vIdParent != -1; }
};

struct result_timings {
  int32_t vCacheN = -1;

  int32_t vPromptN = -1;
  double vPromptMs = 0.0;
  double vPromptPerTokenMs = 0.0;
  double vPromptPerSecond = 0.0;

  int32_t vPredictedN = -1;
  double vPredictedMs = 0.0;
  double vPredictedPerTokenMs = 0.0;
  double vPredictedPerSecond = 0.0;

  // Optional speculative metrics - only included when > 0
  int32_t vDraftN = 0;
  int32_t vDraftNAccepted = 0;

  json to_json() const;
};

struct result_prompt_progress {
  int32_t vTotal = 0;
  int32_t vCache = 0;
  int32_t vProcessed = 0;
  int64_t vTimeMs = 0;

  json to_json() const;
};

struct server_task_result {
  int vId = -1;
  int vIdSlot = -1;

  // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
  size_t vIndex = 0; // to be used for batched tasks

  virtual bool mIsError() {
    // only used by server_task_result_error
    return false;
  }
  virtual bool mIsStop() {
    // only used by server_task_result_cmpl_*
    return true;
  }
  virtual void mUpdate(task_result_state &) {
    // only used by server_task_result_cmpl_*
  }
  virtual json to_json() = 0;
  virtual ~server_task_result() = default;
  virtual server_task_result *mClone() const { GGML_ABORT("not implemented for this task type"); }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
  mgs_token vTok;
  float vProb;
  std::string vTextToSend;
  struct prob_info {
    mgs_token vTok;
    std::string vTxt;
    float vProb;
  };
  std::vector<prob_info> lProbs;

  json to_json(bool pPostSamplingProbs) const;

  static json mProbsVectorToJson(const std::vector<completion_token_output> &pProbs, bool pPostSamplingProbs);

  static float mLogarithm(float pX);

  static std::vector<unsigned char> mStrToBytes(const std::string &pStr);
};

struct server_task_result_cmpl_final : server_task_result {
  std::string vContent;
  mgs_tokens vTokens;

  bool vStream;
  bool vIncludeUsage;
  result_timings vTimings;
  std::string vPrompt;

  bool vTruncated;
  int32_t vNDecoded;
  int32_t vNPromptTokens;
  int32_t vNPromptTokensCache;
  int32_t vNTokensCached;
  bool vHasNewLine;
  std::string vStoppingWord;
  stop_type vStop = cStopTypeNone;

  bool vPostSamplingProbs;
  std::vector<completion_token_output> lProbsOutput;
  std::vector<std::string> lResponseFields;

  task_params vGenerationParams;

  // response formatting
  bool vVerbose = false;
  task_response_type vResType = cTaskResponseTypeNone;
  std::string vOaicompatModel;
  std::string vOaicompatCmplId;
  common_chat_msg vOaicompatMsg; // to be populated by update()

  std::vector<common_chat_msg_diff> lOaicompatMsgDiffs; // to be populated by update()
  bool vIsUpdated = false;

  // for OpenAI Responses API
  std::string vOaiRespId;
  std::string vOaiRespReasoningId;
  std::string vOaiRespMessageId;

  virtual bool mIsStop() override {
    return true; // in stream mode, final responses are considered stop
  }

  virtual json to_json() override;

  virtual void mUpdate(task_result_state &pState) override {
    vIsUpdated = true;
    vOaicompatMsg = pState.mUpdateChatMsg(vContent, false, lOaicompatMsgDiffs);

    vOaiRespId = pState.cOaiRespId;
    vOaiRespReasoningId = pState.cOaiRespReasoningId;
    vOaiRespMessageId = pState.cOaiRespMessageId;
  }

  json mToJsonNonOaicompat();

  json mUsageJsonOaicompat();

  json mToJsonOaicompat();

  json mToJsonOaicompatChat();

  json mToJsonOaicompatChatStream();

  json mToJsonOaicompatResp();

  json mToJsonOaicompatRespStream();

  json mToJsonOaicompatAsr();

  json mToJsonAnthropic();

  json mToJsonAnthropicStream();
};

struct server_task_result_cmpl_partial : server_task_result {
  std::string vContent;
  mgs_tokens vTokens;

  int32_t vNDecoded;
  int32_t vNPromptTokens;
  int32_t vNPromptTokensCache;

  bool vPostSamplingProbs;
  bool vIsProgress = false;
  bool vIsBegin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
  completion_token_output vProbOutput;
  result_timings vTimings;
  result_prompt_progress vProgress;

  // response formatting
  bool vVerbose = false;
  task_response_type vResType = cTaskResponseTypeNone;
  std::string vOaicompatModel;
  std::string vOaicompatCmplId;
  std::vector<common_chat_msg_diff> lOaicompatMsgDiffs; // to be populated by update()
  bool vIsUpdated = false;

  // Streaming state copied from task_result_state for this chunk
  bool vThinkingBlockStarted = false;
  bool vTextBlockStarted = false;

  // for OpenAI Responses API
  bool vOaiRespCreated = false;
  std::string vOaiRespId;
  std::string vOaiRespReasoningId;
  std::string vOaiRespMessageId;
  std::string vOaiRespFcId;

  // for Anthropic API: track if any reasoning content has been generated
  bool vAnthropicHasReasoning = false;

  virtual bool mIsStop() override {
    return false; // in stream mode, partial responses are not considered stop
  }

  virtual void mUpdate(task_result_state &pState) override;

  virtual json to_json() override;

  json mToJsonNonOaicompat();

  json mToJsonOaicompat();

  json mToJsonOaicompatChat();

  json mToJsonOaicompatResp();

  json mToJsonOaicompatAsr();

  json mToJsonAnthropic();
};

// Queue status of a task that waits for a free slot, streamed as {"mgs_queue": {...}} to the Web UI requests that ask
// for it: sent when the task is queued, when the requests ahead of it change and, with every count at 0, when it
// starts. API clients never receive it.
struct server_task_result_queue : server_task_result {
  int32_t vAhead = 0;      // requests resolved before this one, running or waiting
  int32_t vAheadApi = 0;   // of them, sent directly to the API
  int32_t vAheadWebui = 0; // of them, sent by the Web UI

  virtual bool mIsStop() override {
    return false; // the task keeps going
  }

  virtual json to_json() override;
};

struct server_task_result_embd : server_task_result {
  std::vector<std::vector<float>> lEmbedding;

  int32_t vNTokens;

  // response formatting
  task_response_type vResType = cTaskResponseTypeNone;

  virtual json to_json() override;

  json mToJsonNonOaicompat();

  json mToJsonOaicompat();
};

struct server_task_result_rerank : server_task_result {
  float vScore = -1e6;

  int32_t vNTokens;

  virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
  error_type vErrType = cErrorTypeServer;
  std::string vErrMsg;

  // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
  int32_t vNPromptTokens = 0;
  int32_t vNCtx = 0;

  virtual bool mIsError() override { return true; }

  virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
  int vNIdleSlots;
  int vNProcessingSlots;
  int vNTasksDeferred;
  int64_t vTStart;

  // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
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

  // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
  // therefore, we use json to temporarily store the slot.to_json() result
  json vSlotsData = json::array();

  // MoE expert cache state ("active" is false without a cache)
  json vExpertCache = json::object();

  virtual json to_json() override;
};

struct server_task_result_expert_cache : server_task_result {
  json vInfo = json::object();

  virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
  std::string vFilename;
  bool vIsSave; // true = save, false = load

  size_t vNTokens;
  size_t vNBytes;
  double vTMs;

  virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
  size_t vNErased;

  virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
  bool vSuccess = false;
  std::string vMessage; // optional detail when success is false

  virtual json to_json() override {
    json vOut = json{{"success", vSuccess}};
    if (!vMessage.empty()) {
      vOut["message"] = vMessage;
    }
    return vOut;
  }
};

struct server_task_result_get_lora : server_task_result {
  struct lora {
    common_adapter_lora_info vInfo;
    std::string vAloraInvocationString;
    mgs_tokens vAloraInvocationTokens;
  };
  std::vector<lora> lLoras;

  virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
  virtual json to_json() override;
};

struct server_prompt {
  server_tokens vTokens;

  std::list<common_prompt_checkpoint> lCheckpoints;

  void mClear() {
    vTokens.mClear();
    lCheckpoints.clear();
  }

  int mNTokens() const { return vTokens.size(); }

  server_prompt mClone() const {
    return server_prompt{
        vTokens.mClone(),
        lCheckpoints,
    };
  }
};

struct server_prompt_data {
  std::vector<uint8_t> lMain;
  std::vector<uint8_t> lDrft;

  size_t size() const { return lMain.size() + lDrft.size(); }
};

struct server_prompt_cache_state {
  server_prompt vPrompt;
  server_prompt_data vData;

  size_t size() const {
    size_t vRes = vData.size();

    for (const auto &vCkpt : vPrompt.lCheckpoints) {
      vRes += vCkpt.size();
    }

    return vRes;
  }
};

struct server_prompt_cache {
  server_prompt_cache(int32_t pLimitSizeMib, size_t pLimitTokens) {
    this->vLimitSize = 1024ull * 1024ull * (pLimitSizeMib < 0 ? 0 : pLimitSizeMib);
    this->vLimitTokens = pLimitTokens;
  }

  std::list<server_prompt_cache_state> lStates;

  // in bytes, 0 = no limit
  size_t vLimitSize = 0;

  // in tokens, 0 = no limit
  size_t vLimitTokens = 0;

  size_t size() const;

  size_t mNTokens() const;

  server_prompt_cache_state *mAlloc(const server_prompt &pPrompt, size_t pStateSizeMain, size_t pStateSizeDrft);

  bool mLoad(server_prompt &pPrompt, const server_tokens &pTokensNew, mgs_context *pCtxTgt, mgs_context *pCtxDft,
             int32_t pIdSlot);

  void mUpdate();
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
  json vData;
  virtual json to_json() override { return vData; }
  virtual server_task_result *mClone() const override { return new server_task_result_router(*this); }
};
