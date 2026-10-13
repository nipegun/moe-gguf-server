#pragma once

#include "common.h"
#include "log.h"
#include "mgs.h"
#include "chat.h"
#include "mtmd.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

#define SLT_DBG(slot, fmt, ...)                                                                                        \
  LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1),   \
          __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...)                                                                                        \
  LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1),   \
          __VA_ARGS__)
#define SLT_INF(slot, fmt, ...)                                                                                        \
  LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1),   \
          __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...)                                                                                        \
  LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1),   \
          __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...)                                                                                        \
  LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1),   \
          __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

using vRawBuffer = std::vector<uint8_t>;

template <typename T> static T fJsonValue(const json &pBody, const std::string &pKey, const T &pDefaultValue) {
  // Fallback null to default value
  if (pBody.contains(pKey) && !pBody.at(pKey).is_null()) {
    try {
      return pBody.at(pKey);
    } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &vErr) {
      LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", pKey.c_str(),
              json(pDefaultValue).type_name(), vErr.what());
      return pDefaultValue;
    }
  } else {
    return pDefaultValue;
  }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
  cErrorTypeInvalidRequest,
  cErrorTypeAuthentication,
  cErrorTypeServer,
  cErrorTypeNotFound,
  cErrorTypePermission,
  cErrorTypeUnavailable,       // custom error
  cErrorTypeNotSupported,      // custom error
  cErrorTypeExceedContextSize, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
  common_grammar_trigger vValue;

  server_grammar_trigger() = default;
  server_grammar_trigger(const common_grammar_trigger &pValue) : vValue(pValue) {}
  server_grammar_trigger(const json &pIn) {
    vValue.vType = (common_grammar_trigger_type)pIn.at("type").get<int>();
    vValue.vValue = pIn.at("value").get<std::string>();
    if (vValue.vType == cCommonGrammarTriggerTypeToken) {
      vValue.vToken = (mgs_token)pIn.at("token").get<int>();
    }
  }

  json to_json() const {
    json vOut{
        {"type", (int)vValue.vType},
        {"value", vValue.vValue},
    };
    if (vValue.vType == cCommonGrammarTriggerTypeToken) {
      vOut["token"] = (int)vValue.vToken;
    }
    return vOut;
  }
};

json fFormatErrorResponse(const std::string &pMessage, const enum error_type pType);

//
// random string / id
//

std::string fRandomString();
std::string fGenChatcmplid();
std::string fGenToolCallId();

// get a random marker; note: each time the server restarts, the marker will be different
const char *fGetMediaMarker();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool fLoraAllAlora(const std::vector<common_adapter_lora_info> &pLoras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool fLoraShouldClearCache(const std::vector<common_adapter_lora_info> &pCurrent,
                           const std::vector<common_adapter_lora_info> &pNext);

std::map<int, float> fParseLoraRequest(const json &pData);

bool fAreLoraEqual(const std::vector<common_adapter_lora_info> &pL1, const std::vector<common_adapter_lora_info> &pL2);

// get the ids of all enabled loras
std::vector<size_t> fLoraGetEnabledIds(const std::vector<common_adapter_lora_info> &pLoras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
  bool vHasMtmd = false;

private: // disallow accessing these members directly, risking out-of-sync
  // map a **start** index in tokens to the image chunk
  // note: the order need to be in-sync with tokens
  std::map<size_t, mtmd::input_chunk_ptr> dMapIdxToMedia;

  // list of tokens
  //   if the token is MGS_TOKEN_NULL, it indicates that this position is occupied by media chunk
  //   otherwise, it is a normal text token
  // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
  // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
  mgs_tokens vTokens;

  // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
  //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
  // idx  0   1   2   3   4   5      6      7      8      9      10
  // pos  0   1   2   3   4   5      5      5      7      7      7
  // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
  server_tokens() = default;
  ~server_tokens() = default;

  // Prevent copying
  // TODO: server_tokens should be copyable - remove this:
  server_tokens(const server_tokens &) = delete;
  server_tokens &operator=(const server_tokens &) = delete;

  // Allow moving (usually implicitly generated if members are movable)
  server_tokens(server_tokens &&) = default;
  server_tokens &operator=(server_tokens &&) = default;

  // Allow accessing elements using [] operator
  mgs_token operator[](size_t pIndex) { return vTokens[pIndex]; }
  const mgs_token &operator[](size_t pIndex) const { return vTokens[pIndex]; }

  server_tokens(mtmd::input_chunks &pMtmdChunks, bool pHasMtmd);
  server_tokens(const mgs_tokens &pTokens, bool pHasMtmd);

  // for debugging
  std::string mStr() const;

  // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
  mgs_pos mPosNext(int64_t pNTokens = -1) const;

  // number of tokens with position < max_pos
  size_t mSizeUpToPos(mgs_pos pMaxPos) const;

  const mtmd::input_chunk_ptr &mFindChunk(size_t pIdx) const;

  // find next media chunk after idx
  // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
  std::pair<const mtmd::input_chunk_ptr *, size_t> mFindNextMediaChunk(size_t pIdx) const;

  void mPushBack(mgs_token pTok);

  // will create a copy of the chunk if it contains non-text data
  void mPushBack(const mtmd_input_chunk *pChunk);

  // appends server tokens, updates the media map. copies media chunks.
  void mPushBack(server_tokens &pTokens);

  // for compatibility with context shift and prompt truncation
  void mInsert(const mgs_tokens &pInpTokens);

  // for compatibility with speculative decoding, ctx shift, slot save/load
  const mgs_tokens &mGetTokens() const;

  mgs_tokens mGetTextTokens() const;

  // for compatibility with speculative decoding
  void mSetToken(mgs_pos pPos, mgs_token pId);

  size_t size() const { return vTokens.size(); }

  bool empty() const { return vTokens.empty(); }

  // true if the sequence actually contains image/audio chunks.
  bool mHasMedia() const { return !dMapIdxToMedia.empty(); }

  void mClear() {
    dMapIdxToMedia.clear();
    vTokens.clear();
  }

  void mKeepFirst(size_t pN);

  std::string mDetokenize(const mgs_context *pCtx, bool pSpecial) const;

  size_t mGetCommonPrefix(const server_tokens &pB) const;

  // split the tokens into message spans, skipping over media chunks
  common_chat_msg_spans mFindMessageSpans(const common_chat_msg_delimiters &pDelims) const;

  // make sure all text tokens are within the vocab range
  bool mValidate(const struct mgs_context *pCtx) const;

  server_tokens mClone() const;
};

//
// tokenizer and input processing utils
//

bool fJsonIsArrayOfNumbers(const json &pData);

// is array having BOTH numbers & strings?
bool fJsonIsArrayOfMixedNumbersStrings(const json &pData);

// does array have any individual integers/tokens?
bool fJsonIsArrayAndContainsNumbers(const json &pData);

// get value by path(key1 / key2)
json fJsonGetNestedValues(const std::vector<std::string> &pPaths, const json &pJs);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
mgs_tokens fTokenizeMixed(const mgs_vocab *pVocab, const json &pJsonPrompt, bool pAddSpecial, bool pParseSpecial);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t fValidateUtf8(const std::string &pText);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
// if is_placeholder is true, the media chunk will be treated as placeholder for counting tokens; the output tokens are
// not usable for actual inference (e.g. for submitting a task to server_queue)
server_tokens fProcessMtmdPrompt(mtmd_context *pMctx, const std::string &pPrompt, const std::vector<vRawBuffer> &pFiles,
                                 bool pIsPlaceholder = false);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> fTokenizeInputPrompts(const mgs_vocab *pVocab, mtmd_context *pMctx, const json &pJsonPrompt,
                                                 bool pAddSpecial, bool pParseSpecial);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
  bool vPrefillAssistant;
  common_reasoning_format vReasoningFormat;
  std::map<std::string, std::string> dChatTemplateKwargs; // mapping key --> json value
  common_chat_templates_ptr vTmpls;
  bool vAllowImage;
  bool vAllowAudio;
  bool vAllowVideo;
  bool vEnableThinking = true;
  int vReasoningBudget = -1;
  std::string vReasoningBudgetMessage;
  std::string vMediaPath;
  bool vForcePureContent = false;
  bool vMediaAllowPrivateUrls = false; // remote media may reach non-public addresses (--media-allow-private-urls)
};

// used by /completions endpoint
json fOaicompatCompletionParamsParse(const json &pBody);

// used by /chat/completions endpoint
json fOaicompatChatParamsParse(json &pBody, /* openai api json semantics */
                               const server_chat_params &pOpt, std::vector<vRawBuffer> &pOutFiles);

// TODO: move it to server-task.cpp
json fFormatEmbeddingsResponseOaicompat(const json &pRequest, const std::string &pModelName, const json &pEmbeddings,
                                        bool pUseBase64 = false);

// TODO: move it to server-task.cpp
json fFormatResponseRerank(const json &pRequest, const std::string &pModelName, const json &pRanks, bool pIsTeiFormat,
                           std::vector<std::string> &pTexts, int pTopN);

//
// other utils
//

std::vector<mgs_token_data> fGetTokenProbabilities(mgs_context *pCtx, int pIdx, size_t pNTop);

std::string fSafeJsonToStr(const json &pData);

std::string fTokensToStr(mgs_context *pCtx, const mgs_tokens &pTokens);
std::string fTokensToStr(const mgs_vocab *pVocab, const mgs_tokens &pTokens);

// format incomplete utf-8 multibyte character for output
std::string fTokensToOutputFormattedString(const mgs_context *pCtx, const mgs_token pToken);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string fFormatOaiSse(const json &pData);

std::string fFormatOaiRespSse(const json &pData);

// format Anthropic-style SSE with event types
std::string fFormatAnthropicSse(const json &pData);

bool fIsValidUtf8(const std::string &pStr);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

mgs_tokens fFormatPromptInfill(const mgs_vocab *pVocab, const json &pInputPrefix, const json &pInputSuffix,
                               const json &pInputExtra, const int pNBatch, const int pNPredict, const int pNCtx,
                               const bool pSpmInfill, const mgs_tokens &pTokensPrompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens fFormatPromptRerank(const struct mgs_model *pModel, const struct mgs_vocab *pVocab, mtmd_context *pMctx,
                                  const std::string &pQuery, const std::string &pDoc);

// simple implementation of a pipe
// used for streaming data between threads
template <typename T> struct server_pipe {
  std::mutex vMutex;
  std::condition_variable vCv;
  std::queue<T> vQueue;
  std::atomic<bool> vWriterClosed{false};
  std::atomic<bool> vReaderClosed{false};

  // 0 = unbounded (default)
  // > 0, write() drops the oldest item once the queue is full
  size_t vMaxSize = 0;

  void mCloseWrite() {
    vWriterClosed.store(true, std::memory_order_relaxed);
    vCv.notify_all();
  }

  void mCloseRead() {
    vReaderClosed.store(true, std::memory_order_relaxed);
    vCv.notify_all();
  }

  // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
  // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
  bool mRead(T &pOutput, const std::function<bool()> &pShouldStop, bool pCloseOnStop = true) {
    std::unique_lock<std::mutex> vLk(vMutex);
    constexpr auto cPollInterval = std::chrono::milliseconds(500);
    while (true) {
      if (!vQueue.empty()) {
        pOutput = std::move(vQueue.front());
        vQueue.pop();
        return true;
      }
      if (vWriterClosed.load()) {
        return false; // clean EOF
      }
      if (pShouldStop && pShouldStop()) { // a null should_stop means "never stop"
        if (pCloseOnStop) {
          mCloseRead(); // signal broken pipe to writer
        }
        return false; // cancelled / deadline reached
      }
      vCv.wait_for(vLk, cPollInterval);
    }
  }

  bool mWrite(T &&pData) {
    std::lock_guard<std::mutex> vLk(vMutex);
    if (vReaderClosed.load()) {
      return false; // broken pipe
    }
    if (vMaxSize > 0) {
      while (vQueue.size() >= vMaxSize) {
        vQueue.pop(); // drop oldest to stay bounded
      }
    }
    vQueue.push(std::move(pData));
    vCv.notify_one();
    return true;
  }
};
