#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "mgs.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::mFormatLogitBias(const std::vector<mgs_logit_bias> &pLogitBias) const {
  json vData = json::array();
  for (const auto &vLb : pLogitBias) {
    vData.push_back(json{
        {"bias", vLb.bias},
        {"token", vLb.token},
    });
  }
  return vData;
}

json task_params::to_json(bool pOnlyMetrics) const {
  std::vector<std::string> lSamplers;
  lSamplers.reserve(vSampling.lSamplers.size());
  for (const auto &vSampler : vSampling.lSamplers) {
    lSamplers.emplace_back(fCommonSamplerTypeToStr(vSampler));
  }

  json vLora = json::array();
  for (auto &vIt : this->dLora) {
    vLora.push_back({{"id", vIt.first}, {"scale", vIt.second}});
  }

  if (pOnlyMetrics) {
    return json{
        {"seed", vSampling.vSeed},
        {"temperature", vSampling.vTemp},
        {"dynatemp_range", vSampling.vDynatempRange},
        {"dynatemp_exponent", vSampling.vDynatempExponent},
        {"top_k", vSampling.vTopK},
        {"top_p", vSampling.vTopP},
        {"min_p", vSampling.vMinP},
        {"top_n_sigma", vSampling.vTopNSigma},
        {"xtc_probability", vSampling.vXtcProbability},
        {"xtc_threshold", vSampling.vXtcThreshold},
        {"typical_p", vSampling.vTypP},
        {"repeat_last_n", vSampling.vPenaltyLastN},
        {"repeat_penalty", vSampling.vPenaltyRepeat},
        {"presence_penalty", vSampling.vPenaltyPresent},
        {"frequency_penalty", vSampling.vPenaltyFreq},
        {"dry_multiplier", vSampling.vDryMultiplier},
        {"dry_base", vSampling.vDryBase},
        {"dry_allowed_length", vSampling.vDryAllowedLength},
        {"dry_penalty_last_n", vSampling.vDryPenaltyLastN},
        {"mirostat", vSampling.vMirostat},
        {"mirostat_tau", vSampling.vMirostatTau},
        {"mirostat_eta", vSampling.vMirostatEta},
        {"adaptive_target", vSampling.vAdaptiveTarget},
        {"adaptive_decay", vSampling.vAdaptiveDecay},
        {"max_tokens", vNPredict},
        {"n_predict", vNPredict}, // TODO: deduplicate?
        {"n_keep", vNKeep},
        {"n_discard", vNDiscard},
        {"ignore_eos", vSampling.vIgnoreEos},
        {"stream", vStream},
        {"n_probs", vSampling.vNProbs},
        {"min_keep", vSampling.vMinKeep},
        {"chat_format", fCommonChatFormatName(vChatParserParams.vFormat)},
        {"reasoning_format", fCommonReasoningFormatName(vChatParserParams.vReasoningFormat)},
        {"generation_prompt", vChatParserParams.vGenerationPrompt},
        {"samplers", lSamplers},
        {"speculative.types", fCommonSpeculativeTypeNameStr(vSpeculative.lTypes)},
        {"timings_per_token", vTimingsPerToken},
        {"post_sampling_probs", vPostSamplingProbs},
        {"backend_sampling", vSampling.vBackendSampling},
        {"lora", vLora},
    };
  }

  auto vGrammarTriggers = json::array();
  for (const auto &vTrigger : vSampling.lGrammarTriggers) {
    server_grammar_trigger vCt(vTrigger);
    vGrammarTriggers.push_back(vCt.to_json());
  }

  return json{
      {"seed", vSampling.vSeed},
      {"temperature", vSampling.vTemp},
      {"dynatemp_range", vSampling.vDynatempRange},
      {"dynatemp_exponent", vSampling.vDynatempExponent},
      {"top_k", vSampling.vTopK},
      {"top_p", vSampling.vTopP},
      {"min_p", vSampling.vMinP},
      {"top_n_sigma", vSampling.vTopNSigma},
      {"xtc_probability", vSampling.vXtcProbability},
      {"xtc_threshold", vSampling.vXtcThreshold},
      {"typical_p", vSampling.vTypP},
      {"repeat_last_n", vSampling.vPenaltyLastN},
      {"repeat_penalty", vSampling.vPenaltyRepeat},
      {"presence_penalty", vSampling.vPenaltyPresent},
      {"frequency_penalty", vSampling.vPenaltyFreq},
      {"dry_multiplier", vSampling.vDryMultiplier},
      {"dry_base", vSampling.vDryBase},
      {"dry_allowed_length", vSampling.vDryAllowedLength},
      {"dry_penalty_last_n", vSampling.vDryPenaltyLastN},
      {"dry_sequence_breakers", vSampling.lDrySequenceBreakers},
      {"mirostat", vSampling.vMirostat},
      {"mirostat_tau", vSampling.vMirostatTau},
      {"mirostat_eta", vSampling.vMirostatEta},
      {"adaptive_target", vSampling.vAdaptiveTarget},
      {"adaptive_decay", vSampling.vAdaptiveDecay},
      {"stop", lAntiprompt},
      {"max_tokens", vNPredict},
      {"n_predict", vNPredict}, // TODO: deduplicate?
      {"n_keep", vNKeep},
      {"n_discard", vNDiscard},
      {"ignore_eos", vSampling.vIgnoreEos},
      {"stream", vStream},
      {"logit_bias", mFormatLogitBias(vSampling.lLogitBias)},
      {"n_probs", vSampling.vNProbs},
      {"min_keep", vSampling.vMinKeep},
      {"grammar", fCommonGrammarValue(vSampling.vGrammar)},
      {"grammar_lazy", vSampling.vGrammarLazy},
      {"grammar_triggers", vGrammarTriggers},
      {"preserved_tokens", vSampling.sPreservedTokens},
      {"chat_format", fCommonChatFormatName(vChatParserParams.vFormat)},
      {"reasoning_format", fCommonReasoningFormatName(vChatParserParams.vReasoningFormat)},
      {"generation_prompt", vChatParserParams.vGenerationPrompt},
      {"samplers", lSamplers},
      {"speculative.types", fCommonSpeculativeTypeNameStr(vSpeculative.lTypes)},
      {"timings_per_token", vTimingsPerToken},
      {"post_sampling_probs", vPostSamplingProbs},
      {"backend_sampling", vSampling.vBackendSampling},
      {"lora", vLora},
  };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params &pChatParserParams)
    : vChatParserParams(pChatParserParams), cOaiRespId("resp_" + fRandomString()),
      cOaiRespReasoningId("rs_" + fRandomString()), cOaiRespMessageId("msg_" + fRandomString()) {
  if (pChatParserParams.vIsContinuation && !pChatParserParams.vEcho) {
    // initialize chat_msg to avoid emitting a delta containing the assistant prefill
    vChatMsg = fCommonChatParse("", true, pChatParserParams);
  }
}

common_chat_msg task_result_state::mUpdateChatMsg(const std::string &pTextAdded, bool pIsPartial,
                                                  std::vector<common_chat_msg_diff> &pDiffs, bool pFilterToolCalls) {
  vGeneratedText += pTextAdded;
  auto vMsgPrvCopy = vChatMsg;
  // SRV_DBG("Parsing chat message: %s\n", vGeneratedText.c_str());
  auto vNewMsg = fCommonChatParse(vGeneratedText, pIsPartial, vChatParserParams);
  if (!vNewMsg.empty()) {
    vNewMsg.mSetToolCallIds(lGeneratedToolCallIds, fGenToolCallId);
    vChatMsg = vNewMsg;
    auto vAllDiffs = common_chat_msg_diff::mComputeDiffs(vMsgPrvCopy, vChatMsg);

    if (!pFilterToolCalls) {
      pDiffs = std::move(vAllDiffs);
    } else {
      for (auto &vD : vAllDiffs) {
        // If this is a new type of delta, flush all currently pending tool call names
        for (size_t vI = 0; vI < vChatMsg.lToolCalls.size(); ++vI) {
          if (sSentToolCallNames.count(vI) || vChatMsg.lToolCalls[vI].vName.empty()) {
            continue;
          }
          if (vD.vToolCallIndex != vI || !vD.vToolCallDelta.vArguments.empty()) {
            common_chat_msg_diff vHeader;
            vHeader.vToolCallIndex = vI;
            vHeader.vToolCallDelta.vId = vChatMsg.lToolCalls[vI].vId;
            vHeader.vToolCallDelta.vName = vChatMsg.lToolCalls[vI].vName;
            pDiffs.push_back(std::move(vHeader));
            sSentToolCallNames.insert(vI);
          }
        }

        if (vD.vToolCallIndex == std::string::npos) {
          pDiffs.push_back(std::move(vD));
        } else {
          size_t vI = vD.vToolCallIndex;
          if (sSentToolCallNames.count(vI)) {
            if (!vD.vToolCallDelta.vArguments.empty()) {
              vD.vToolCallDelta.vName = "";
              vD.vToolCallDelta.vId = "";
              pDiffs.push_back(std::move(vD));
            }
          } else {
            // Not sent yet.
            if (!vD.vToolCallDelta.vArguments.empty() || !pIsPartial) {
              vD.vToolCallDelta.vName = vChatMsg.lToolCalls[vI].vName;
              vD.vToolCallDelta.vId = vChatMsg.lToolCalls[vI].vId;
              pDiffs.push_back(std::move(vD));
              sSentToolCallNames.insert(vI);
            } else {
              // Suppress
            }
          }
        }
      }
      // Final check at EOF
      if (!pIsPartial) {
        for (size_t vI = 0; vI < vChatMsg.lToolCalls.size(); ++vI) {
          if (!sSentToolCallNames.count(vI) && !vChatMsg.lToolCalls[vI].vName.empty()) {
            common_chat_msg_diff vHeader;
            vHeader.vToolCallIndex = vI;
            vHeader.vToolCallDelta.vId = vChatMsg.lToolCalls[vI].vId;
            vHeader.vToolCallDelta.vName = vChatMsg.lToolCalls[vI].vName;
            pDiffs.push_back(std::move(vHeader));
            sSentToolCallNames.insert(vI);
          }
        }
      }
    }
  }
  return vChatMsg;
}

//

// result_timings
//

json result_timings::to_json() const {
  json vBase = {
      {"cache_n", vCacheN},

      {"prompt_n", vPromptN},
      {"prompt_ms", vPromptMs},
      {"prompt_per_token_ms", vPromptPerTokenMs},
      {"prompt_per_second", vPromptPerSecond},

      {"predicted_n", vPredictedN},
      {"predicted_ms", vPredictedMs},
      {"predicted_per_token_ms", vPredictedPerTokenMs},
      {"predicted_per_second", vPredictedPerSecond},
  };

  if (vDraftN > 0) {
    vBase["draft_n"] = vDraftN;
    vBase["draft_n_accepted"] = vDraftNAccepted;
  }

  return vBase;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
  return json{
      {"total", vTotal},
      {"cache", vCache},
      {"processed", vProcessed},
      {"time_ms", vTimeMs},
  };
}

static inline std::string fStopTypeToStr(stop_type pType) {
  switch (pType) {
  case cStopTypeEos:
    return "eos";
  case cStopTypeWord:
    return "word";
  case cStopTypeLimit:
    return "limit";
  default:
    return "none";
  }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool pPostSamplingProbs) const {
  json vProbsForToken = json::array();
  for (const auto &vP : lProbs) {
    std::string vTxt(vP.vTxt);
    vTxt.resize(fValidateUtf8(vTxt));
    vProbsForToken.push_back(json{
        {"id", vP.vTok},
        {"token", vTxt},
        {"bytes", mStrToBytes(vP.vTxt)},
        {pPostSamplingProbs ? "prob" : "logprob", pPostSamplingProbs ? vP.vProb : mLogarithm(vP.vProb)},
    });
  }
  return vProbsForToken;
}

json completion_token_output::mProbsVectorToJson(const std::vector<completion_token_output> &pProbs,
                                                 bool pPostSamplingProbs) {
  json vOut = json::array();
  for (const auto &vP : pProbs) {
    std::string vTxt(vP.vTextToSend);
    vTxt.resize(fValidateUtf8(vTxt));
    vOut.push_back(json{
        {"id", vP.vTok},
        {"token", vTxt},
        {"bytes", mStrToBytes(vP.vTextToSend)},
        {pPostSamplingProbs ? "prob" : "logprob", pPostSamplingProbs ? vP.vProb : mLogarithm(vP.vProb)},
        {pPostSamplingProbs ? "top_probs" : "top_logprobs", vP.to_json(pPostSamplingProbs)},
    });
  }
  return vOut;
}

float completion_token_output::mLogarithm(float pX) {
  // nlohmann::json converts -inf to null, so we need to prevent that
  return pX == 0.0f ? std::numeric_limits<float>::lowest() : std::log(pX);
}

std::vector<unsigned char> completion_token_output::mStrToBytes(const std::string &pStr) {
  std::vector<unsigned char> lBytes;
  for (unsigned char vC : pStr) {
    lBytes.push_back(vC);
  }
  return lBytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
  GGML_ASSERT(vIsUpdated && "update() must be called before to_json()");
  switch (vResType) {
  case cTaskResponseTypeNone:
    return mToJsonNonOaicompat();
  case cTaskResponseTypeOaiCmpl:
    return mToJsonOaicompat();
  case cTaskResponseTypeOaiChat:
    return vStream ? mToJsonOaicompatChatStream() : mToJsonOaicompatChat();
  case cTaskResponseTypeOaiResp:
    return vStream ? mToJsonOaicompatRespStream() : mToJsonOaicompatResp();
  case cTaskResponseTypeOaiAsr:
    return mToJsonOaicompatAsr();
  case cTaskResponseTypeAnthropic:
    return vStream ? mToJsonAnthropicStream() : mToJsonAnthropic();
  default:
    GGML_ASSERT(false && "Invalid task_response_type");
  }
}

json server_task_result_cmpl_final::mToJsonNonOaicompat() {
  json vRes = json{
      {"index", vIndex},
      {"content", vContent},
      {"tokens", vTokens},
      {"id_slot", vIdSlot},
      {"stop", true},
      {"model", vOaicompatModel},
      {"tokens_predicted", vNDecoded},
      {"tokens_evaluated", vNPromptTokens},
      {"generation_settings", vGenerationParams.to_json()},
      {"prompt", vPrompt},
      {"has_new_line", vHasNewLine},
      {"truncated", vTruncated},
      {"stop_type", fStopTypeToStr(vStop)},
      {"stopping_word", vStoppingWord},
      {"tokens_cached", vNTokensCached},
      {"timings", vTimings.to_json()},
  };
  if (!vStream && !lProbsOutput.empty()) {
    vRes["completion_probabilities"] = completion_token_output::mProbsVectorToJson(lProbsOutput, vPostSamplingProbs);
  }
  return lResponseFields.empty() ? vRes : fJsonGetNestedValues(lResponseFields, vRes);
}

json server_task_result_cmpl_final::mUsageJsonOaicompat() {
  return json{
      {"completion_tokens", vNDecoded},
      {"prompt_tokens", vNPromptTokens},
      {"total_tokens", vNDecoded + vNPromptTokens},
      {"prompt_tokens_details", json{{"cached_tokens", vNPromptTokensCache}}},
  };
}

json server_task_result_cmpl_final::mToJsonOaicompat() {
  std::time_t vT = std::time(0);
  json vLogprobs = json(nullptr); // OAI default to null
  if (!vStream && lProbsOutput.size() > 0) {
    vLogprobs = json{
        {"content", completion_token_output::mProbsVectorToJson(lProbsOutput, vPostSamplingProbs)},
    };
  }
  json vFinishReason = "length";
  if (vStop == cStopTypeWord || vStop == cStopTypeEos) {
    vFinishReason = "stop";
  }
  json vRes = json{{"choices", json::array({json{
                                   {"text", vContent},
                                   {"index", vIndex},
                                   {"logprobs", vLogprobs},
                                   {"finish_reason", vFinishReason},
                               }})},
                   {"created", vT},
                   {"model", vOaicompatModel},
                   {"system_fingerprint", std::string(mgs_version())},
                   {"object", "text_completion"},
                   {"usage", mUsageJsonOaicompat()},
                   {"id", vOaicompatCmplId}};

  // extra fields for debugging purposes
  if (vVerbose) {
    vRes["__verbose"] = mToJsonNonOaicompat();
  }
  if (vTimings.vPromptN >= 0) {
    vRes.push_back({"timings", vTimings.to_json()});
  }

  return vRes;
}

json server_task_result_cmpl_final::mToJsonOaicompatChat() {
  std::string vFinishReason = "length";
  common_chat_msg vMsg;
  if (!vOaicompatMsg.empty()) {
    vMsg = vOaicompatMsg;
  } else {
    vMsg.vRole = "assistant";
    vMsg.vContent = vContent;
  }
  if (vStop == cStopTypeWord || vStop == cStopTypeEos) {
    vFinishReason = vMsg.lToolCalls.empty() ? "stop" : "tool_calls";
  }

  json vChoice{
      {"finish_reason", vFinishReason},
      {"index", vIndex},
      {"message", vMsg.mToJsonOaicompat()},
  };

  if (!vStream && lProbsOutput.size() > 0) {
    vChoice["logprobs"] = json{
        {"content", completion_token_output::mProbsVectorToJson(lProbsOutput, vPostSamplingProbs)},
    };
  }

  std::time_t vT = std::time(0);

  json vRes = json{{"choices", json::array({vChoice})},
                   {"created", vT},
                   {"model", vOaicompatModel},
                   {"system_fingerprint", std::string(mgs_version())},
                   {"object", "chat.completion"},
                   {"usage", mUsageJsonOaicompat()},
                   {"id", vOaicompatCmplId}};

  // extra fields for debugging purposes
  if (vVerbose) {
    vRes["__verbose"] = mToJsonNonOaicompat();
  }
  if (vTimings.vPromptN >= 0) {
    vRes.push_back({"timings", vTimings.to_json()});
  }

  return vRes;
}

json server_task_result_cmpl_final::mToJsonOaicompatChatStream() {
  std::time_t vT = std::time(0);
  std::string vFinishReason = "length";
  if (vStop == cStopTypeWord || vStop == cStopTypeEos) {
    vFinishReason = vOaicompatMsg.lToolCalls.empty() ? "stop" : "tool_calls";
  }

  json vDeltas = json::array();
  for (const auto &vDiff : lOaicompatMsgDiffs) {
    vDeltas.push_back({
        {"choices", json::array({
                        json{
                            {"finish_reason", nullptr},
                            {"index", vIndex},
                            {"delta", fServerChatMsgDiffToJsonOaicompat(vDiff)},
                        },
                    })},
        {"created", vT},
        {"id", vOaicompatCmplId},
        {"model", vOaicompatModel},
        {"system_fingerprint", std::string(mgs_version())},
        {"object", "chat.completion.chunk"},
    });
  }

  vDeltas.push_back({
      {"choices", json::array({
                      json{
                          {"finish_reason", vFinishReason},
                          {"index", vIndex},
                          {"delta", json::object()},
                      },
                  })},
      {"created", vT},
      {"id", vOaicompatCmplId},
      {"model", vOaicompatModel},
      {"system_fingerprint", std::string(mgs_version())},
      {"object", "chat.completion.chunk"},
  });

  if (vIncludeUsage) {
    // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including
    // usage https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
    vDeltas.push_back({
        {"choices", json::array()},
        {"created", vT},
        {"id", vOaicompatCmplId},
        {"model", vOaicompatModel},
        {"system_fingerprint", std::string(mgs_version())},
        {"object", "chat.completion.chunk"},
        {"usage", mUsageJsonOaicompat()},
    });
  }

  if (vTimings.vPromptN >= 0) {
    vDeltas.back().push_back({"timings", vTimings.to_json()});
  }

  // extra fields for debugging purposes
  if (vVerbose && !vDeltas.empty()) {
    vDeltas.front()["__verbose"] = mToJsonNonOaicompat();
  }

  return vDeltas;
}

json server_task_result_cmpl_final::mToJsonOaicompatResp() {
  common_chat_msg vMsg;
  if (!vOaicompatMsg.empty()) {
    vMsg = vOaicompatMsg;
  } else {
    vMsg.vRole = "assistant";
    vMsg.vContent = vContent;
  }

  std::vector<json> lOutput;

  if (vMsg.vReasoningContent != "") {
    lOutput.push_back(json{
        {"id", "rs_" + fRandomString()},
        {"summary", json::array()},
        {"type", "reasoning"},
        {"content", json::array({json{
                        {"text", vMsg.vReasoningContent},
                        {"type", "reasoning_text"},
                    }})},
        {"encrypted_content", ""},
        {"status", "completed"},
    });
  }

  if (vMsg.vContent != "") {
    lOutput.push_back(json{
        {"content", json::array({json{
                        {"type", "output_text"},
                        {"annotations", json::array()},
                        {"logprobs", json::array()},
                        {"text", vMsg.vContent},
                    }})},
        {"id", "msg_" + fRandomString()},
        {"role", vMsg.vRole},
        {"status", "completed"},
        {"type", "message"},
    });
  }

  for (const common_chat_tool_call &vToolCall : vOaicompatMsg.lToolCalls) {
    lOutput.push_back(json{
        {"id", "fc_" + vToolCall.vId},
        {"type", "function_call"},
        {"status", "completed"},
        {"arguments", vToolCall.vArguments},
        {"call_id", "call_" + vToolCall.vId},
        {"name", vToolCall.vName},
    });
  }

  std::time_t vT = std::time(0);
  json vRes = {
      {"completed_at", vT},
      {"created_at", vT},
      {"id", vOaiRespId},
      {"model", vOaicompatModel},
      {"object", "response"},
      {"output", lOutput},
      {"status", "completed"},
      {"usage",
       json{
           {"input_tokens", vNPromptTokens},
           {"output_tokens", vNDecoded},
           {"total_tokens", vNDecoded + vNPromptTokens},
           {"input_tokens_details", json{{"cached_tokens", vNPromptTokensCache}}},
       }},
  };

  return vRes;
}

json server_task_result_cmpl_final::mToJsonOaicompatRespStream() {
  std::vector<json> lServerSentEvents;
  std::vector<json> lOutput;

  if (vOaicompatMsg.vReasoningContent != "") {
    const json cOutputItem = json{
        {"id", vOaiRespReasoningId},
        {"summary", json::array()},
        {"type", "reasoning"},
        {"content", json::array({json{
                        {"text", vOaicompatMsg.vReasoningContent},
                        {"type", "reasoning_text"},
                    }})},
        {"encrypted_content", ""},
    };

    lServerSentEvents.push_back(json{{"event", "response.output_item.done"},
                                     {"data", json{{"type", "response.output_item.done"}, {"item", cOutputItem}}}});
    lOutput.push_back(cOutputItem);
  }

  if (vOaicompatMsg.vContent != "") {
    lServerSentEvents.push_back(json{{"event", "response.output_text.done"},
                                     {"data", json{{"type", "response.output_text.done"},
                                                   {"item_id", vOaiRespMessageId},
                                                   {"text", vOaicompatMsg.vContent}}}});

    const json cContentPart = {{"type", "output_text"},
                               {"annotations", json::array()},
                               {"logprobs", json::array()},
                               {"text", vOaicompatMsg.vContent}};

    lServerSentEvents.push_back(
        json{{"event", "response.content_part.done"},
             {"data",
              json{{"type", "response.content_part.done"}, {"item_id", vOaiRespMessageId}, {"part", cContentPart}}}});
    const json cOutputItem = {{"type", "message"},
                              {"status", "completed"},
                              {"id", vOaiRespMessageId},
                              {"content", json::array({cContentPart})},
                              {"role", "assistant"}};

    lServerSentEvents.push_back(json{{"event", "response.output_item.done"},
                                     {"data", json{{"type", "response.output_item.done"}, {"item", cOutputItem}}}});
    lOutput.push_back(cOutputItem);
  }

  for (const common_chat_tool_call &vToolCall : vOaicompatMsg.lToolCalls) {
    const json cOutputItem = {{"id", "fc_" + vToolCall.vId},
                              {"type", "function_call"},
                              {"status", "completed"},
                              {"arguments", vToolCall.vArguments},
                              {"call_id", "call_" + vToolCall.vId},
                              {"name", vToolCall.vName}};
    lServerSentEvents.push_back(json{{"event", "response.output_item.done"},
                                     {"data", json{{"type", "response.output_item.done"}, {"item", cOutputItem}}}});
    lOutput.push_back(cOutputItem);
  }

  std::time_t vT = std::time(0);
  lServerSentEvents.push_back(
      json{{"event", "response.completed"},
           {"data", json{
                        {"type", "response.completed"},
                        {"response", json{{"id", vOaiRespId},
                                          {"object", "response"},
                                          {"created_at", vT},
                                          {"status", "completed"},
                                          {"model", vOaicompatModel},
                                          {"output", lOutput},
                                          {"usage",
                                           json{
                                               {"input_tokens", vNPromptTokens},
                                               {"output_tokens", vNDecoded},
                                               {"total_tokens", vNDecoded + vNPromptTokens},
                                               {"input_tokens_details", json{{"cached_tokens", vNPromptTokensCache}}},
                                           }}}},
                    }}});

  if (vTimings.vPromptN >= 0) {
    lServerSentEvents.back().at("data").push_back({"timings", vTimings.to_json()});
  }

  return lServerSentEvents;
}

json server_task_result_cmpl_final::mToJsonOaicompatAsr() {
  json vEvent = json{
      {"type", "transcript.text.done"},
      {"text", vOaicompatMsg.vContent},
      {"usage",
       json{
           {"type", "tokens"},
           {"input_tokens", vNPromptTokens},
           {"output_tokens", vNDecoded},
           {"total_tokens", vNDecoded + vNPromptTokens},
           {"input_tokens_details", json{{"cached_tokens", vNPromptTokensCache}}},
       }},
  };
  return vEvent;
}

json server_task_result_cmpl_final::mToJsonAnthropic() {
  std::string vStopReason = "max_tokens";
  if (vStop == cStopTypeWord || vStop == cStopTypeEos) {
    vStopReason = vOaicompatMsg.lToolCalls.empty() ? "end_turn" : "tool_use";
  }

  json vContentBlocks = json::array();

  common_chat_msg vMsg;
  if (!vOaicompatMsg.empty()) {
    vMsg = vOaicompatMsg;
  } else {
    vMsg.vRole = "assistant";
    vMsg.vContent = vContent;
  }

  // thinking block comes first (Anthropic extended thinking format)
  if (!vMsg.vReasoningContent.empty()) {
    vContentBlocks.push_back({
        {"type", "thinking"}, {"thinking", vMsg.vReasoningContent}, {"signature", ""}
        // empty signature for local models (no cryptographic verification)
    });
  }

  if (!vMsg.vContent.empty()) {
    vContentBlocks.push_back({{"type", "text"}, {"text", vMsg.vContent}});
  }

  for (const auto &vToolCall : vMsg.lToolCalls) {
    json vToolUseBlock = {{"type", "tool_use"}, {"id", vToolCall.vId}, {"name", vToolCall.vName}};

    try {
      vToolUseBlock["input"] = json::parse(vToolCall.vArguments);
    } catch (const std::exception &) {
      vToolUseBlock["input"] = json::object();
    }

    vContentBlocks.push_back(vToolUseBlock);
  }

  json vRes = {{"id", vOaicompatCmplId},
               {"type", "message"},
               {"role", "assistant"},
               {"content", vContentBlocks},
               {"model", vOaicompatModel},
               {"stop_reason", vStopReason},
               {"stop_sequence", vStoppingWord.empty() ? nullptr : json(vStoppingWord)},
               {"usage",
                {{"cache_read_input_tokens", vNPromptTokensCache},
                 {"input_tokens", vNPromptTokens - vNPromptTokensCache},
                 {"output_tokens", vNDecoded}}}};

  return vRes;
}

json server_task_result_cmpl_final::mToJsonAnthropicStream() {
  json vEvents = json::array();

  std::string vStopReason = "max_tokens";
  if (vStop == cStopTypeWord || vStop == cStopTypeEos) {
    vStopReason = vOaicompatMsg.lToolCalls.empty() ? "end_turn" : "tool_use";
  }

  bool vHasThinking = !vOaicompatMsg.vReasoningContent.empty();
  bool vHasText = !vOaicompatMsg.vContent.empty();
  size_t vNumToolCalls = vOaicompatMsg.lToolCalls.size();

  // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
  size_t vThinkingBlockIndex = 0;
  size_t vTextBlockIndex = vHasThinking ? 1 : 0;

  bool vThinkingBlockStarted = false;
  bool vTextBlockStarted = false;
  std::unordered_set<size_t> sToolCallsStarted;

  for (const auto &vDiff : lOaicompatMsgDiffs) {
    // handle thinking/reasoning content
    if (!vDiff.vReasoningContentDelta.empty()) {
      if (!vThinkingBlockStarted) {
        vEvents.push_back({{"event", "content_block_start"},
                           {"data",
                            {{"type", "content_block_start"},
                             {"index", vThinkingBlockIndex},
                             {"content_block", {{"type", "thinking"}, {"thinking", ""}}}}}});
        vThinkingBlockStarted = true;
      }

      vEvents.push_back({{"event", "content_block_delta"},
                         {"data",
                          {{"type", "content_block_delta"},
                           {"index", vThinkingBlockIndex},
                           {"delta", {{"type", "thinking_delta"}, {"thinking", vDiff.vReasoningContentDelta}}}}}});
    }

    // handle regular text content
    if (!vDiff.vContentDelta.empty()) {
      if (!vTextBlockStarted) {
        vEvents.push_back({{"event", "content_block_start"},
                           {"data",
                            {{"type", "content_block_start"},
                             {"index", vTextBlockIndex},
                             {"content_block", {{"type", "text"}, {"text", ""}}}}}});
        vTextBlockStarted = true;
      }

      vEvents.push_back({{"event", "content_block_delta"},
                         {"data",
                          {{"type", "content_block_delta"},
                           {"index", vTextBlockIndex},
                           {"delta", {{"type", "text_delta"}, {"text", vDiff.vContentDelta}}}}}});
    }

    // handle tool calls
    if (vDiff.vToolCallIndex != std::string::npos) {
      size_t vContentBlockIndex = (vHasThinking ? 1 : 0) + (vHasText ? 1 : 0) + vDiff.vToolCallIndex;

      if (sToolCallsStarted.find(vDiff.vToolCallIndex) == sToolCallsStarted.end()) {
        const auto &vFullToolCall = vOaicompatMsg.lToolCalls[vDiff.vToolCallIndex];

        vEvents.push_back(
            {{"event", "content_block_start"},
             {"data",
              {{"type", "content_block_start"},
               {"index", vContentBlockIndex},
               {"content_block", {{"type", "tool_use"}, {"id", vFullToolCall.vId}, {"name", vFullToolCall.vName}}}}}});
        sToolCallsStarted.insert(vDiff.vToolCallIndex);
      }

      if (!vDiff.vToolCallDelta.vArguments.empty()) {
        vEvents.push_back(
            {{"event", "content_block_delta"},
             {"data",
              {{"type", "content_block_delta"},
               {"index", vContentBlockIndex},
               {"delta", {{"type", "input_json_delta"}, {"partial_json", vDiff.vToolCallDelta.vArguments}}}}}});
      }
    }
  }

  // close content blocks in order
  if (vHasThinking) {
    // Anthropic API requires a signature_delta before closing thinking blocks
    // We use an empty signature since we can't generate a cryptographic signature for local models
    vEvents.push_back({{"event", "content_block_delta"},
                       {"data",
                        {{"type", "content_block_delta"},
                         {"index", vThinkingBlockIndex},
                         {"delta", {{"type", "signature_delta"}, {"signature", ""}}}}}});
    vEvents.push_back(
        {{"event", "content_block_stop"}, {"data", {{"type", "content_block_stop"}, {"index", vThinkingBlockIndex}}}});
  }

  if (vHasText) {
    vEvents.push_back(
        {{"event", "content_block_stop"}, {"data", {{"type", "content_block_stop"}, {"index", vTextBlockIndex}}}});
  }

  for (size_t vI = 0; vI < vNumToolCalls; vI++) {
    size_t vContentBlockIndex = (vHasThinking ? 1 : 0) + (vHasText ? 1 : 0) + vI;
    vEvents.push_back(
        {{"event", "content_block_stop"}, {"data", {{"type", "content_block_stop"}, {"index", vContentBlockIndex}}}});
  }

  vEvents.push_back(
      {{"event", "message_delta"},
       {"data",
        {{"type", "message_delta"},
         {"delta",
          {{"stop_reason", vStopReason}, {"stop_sequence", vStoppingWord.empty() ? nullptr : json(vStoppingWord)}}},
         {"usage", {{"output_tokens", vNDecoded}}}}}});

  vEvents.push_back({{"event", "message_stop"}, {"data", {{"type", "message_stop"}}}});

  return vEvents;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::mUpdate(task_result_state &pState) {
  vIsUpdated = true;
  if (vIsBegin) {
    return; // begin marker only flushes headers, skip parsing
  }
  pState.mUpdateChatMsg(vContent, true, lOaicompatMsgDiffs);

  // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
  vThinkingBlockStarted = pState.vThinkingBlockStarted;
  vTextBlockStarted = pState.vTextBlockStarted;

  vOaiRespCreated = pState.vOaiRespCreated;
  vOaiRespId = pState.cOaiRespId;
  vOaiRespReasoningId = pState.cOaiRespReasoningId;
  vOaiRespMessageId = pState.cOaiRespMessageId;
  vOaiRespFcId = pState.vOaiRespFcId;

  // track if the accumulated message has any reasoning content
  vAnthropicHasReasoning = !pState.vChatMsg.vReasoningContent.empty();

  if (vResType == cTaskResponseTypeOaiResp && !pState.vOaiRespCreated && (vIsProgress || vNDecoded == 1)) {
    pState.vOaiRespCreated = true;
  }

  // Pre-compute state updates based on diffs (for next chunk)
  for (const common_chat_msg_diff &vDiff : lOaicompatMsgDiffs) {
    if (!vDiff.vReasoningContentDelta.empty() && !pState.vThinkingBlockStarted) {
      pState.vThinkingBlockStarted = true;
    }
    if (!vDiff.vContentDelta.empty() && !pState.vTextBlockStarted) {
      pState.vTextBlockStarted = true;
    }
    if (!vDiff.vToolCallDelta.vName.empty()) {
      pState.vOaiRespFcId = vDiff.vToolCallDelta.vId;
    }
  }
}

json server_task_result_cmpl_partial::to_json() {
  GGML_ASSERT(vIsUpdated && "update() must be called before to_json()");
  if (vIsBegin) {
    return nullptr; // simply signal to HTTP handler to send the headers and status code
  }
  switch (vResType) {
  case cTaskResponseTypeNone:
    return mToJsonNonOaicompat();
  case cTaskResponseTypeOaiCmpl:
    return mToJsonOaicompat();
  case cTaskResponseTypeOaiChat:
    return mToJsonOaicompatChat();
  case cTaskResponseTypeOaiResp:
    return mToJsonOaicompatResp();
  case cTaskResponseTypeOaiAsr:
    return mToJsonOaicompatAsr();
  case cTaskResponseTypeAnthropic:
    return mToJsonAnthropic();
  default:
    GGML_ASSERT(false && "Invalid task_response_type");
  }
}

json server_task_result_cmpl_partial::mToJsonNonOaicompat() {
  // non-OAI-compat JSON
  json vRes = json{
      {"index", vIndex},
      {"content", vContent},
      {"tokens", vTokens},
      {"stop", false},
      {"id_slot", vIdSlot},
      {"tokens_predicted", vNDecoded},
      {"tokens_evaluated", vNPromptTokens},
  };
  // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
  if (vTimings.vPromptN > 0) {
    vRes.push_back({"timings", vTimings.to_json()});
  }
  if (vIsProgress) {
    vRes.push_back({"prompt_progress", vProgress.to_json()});
  }
  if (!vProbOutput.lProbs.empty()) {
    vRes["completion_probabilities"] = completion_token_output::mProbsVectorToJson({vProbOutput}, vPostSamplingProbs);
  }
  return vRes;
}

json server_task_result_cmpl_partial::mToJsonOaicompat() {
  std::time_t vT = std::time(0);
  json vLogprobs = json(nullptr); // OAI default to null
  if (vProbOutput.lProbs.size() > 0) {
    vLogprobs = json{
        {"content", completion_token_output::mProbsVectorToJson({vProbOutput}, vPostSamplingProbs)},
    };
  }
  json vRes = json{{"choices", json::array({json{
                                   {"text", vContent},
                                   {"index", vIndex},
                                   {"logprobs", vLogprobs},
                                   {"finish_reason", nullptr},
                               }})},
                   {"created", vT},
                   {"model", vOaicompatModel},
                   {"system_fingerprint", std::string(mgs_version())},
                   {"object", "text_completion"},
                   {"id", vOaicompatCmplId}};

  // extra fields for debugging purposes
  if (vVerbose) {
    vRes["__verbose"] = mToJsonNonOaicompat();
  }
  if (vTimings.vPromptN >= 0) {
    vRes.push_back({"timings", vTimings.to_json()});
  }
  if (vIsProgress) {
    vRes.push_back({"prompt_progress", vProgress.to_json()});
  }

  return vRes;
}

json server_task_result_cmpl_partial::mToJsonOaicompatChat() {
  bool vFirst = vNDecoded == 1;
  std::time_t vT = std::time(0);
  json vChoices;

  std::vector<json> lDeltas;
  auto vAddDelta = [&](const json &pDelta) {
    lDeltas.push_back({
        {"choices", json::array({
                        json{
                            {"finish_reason", nullptr},
                            {"index", vIndex},
                            {"delta", pDelta},
                        },
                    })},
        {"created", vT},
        {"id", vOaicompatCmplId},
        {"model", vOaicompatModel},
        {"system_fingerprint", std::string(mgs_version())},
        {"object", "chat.completion.chunk"},
    });
  };
  // We have to send an initial update to conform to openai behavior
  if (vFirst || vIsProgress) {
    vAddDelta({
        {"role", "assistant"},
        {"content", nullptr},
    });
  }

  for (const auto &vDiff : lOaicompatMsgDiffs) {
    vAddDelta(fServerChatMsgDiffToJsonOaicompat(vDiff));
  }

  if (!lDeltas.empty()) {
    auto &vLastJson = lDeltas[lDeltas.size() - 1];
    GGML_ASSERT(vLastJson.at("choices").size() >= 1);

    if (vProbOutput.lProbs.size() > 0) {
      vLastJson.at("choices").at(0)["logprobs"] = json{
          {"content", completion_token_output::mProbsVectorToJson({vProbOutput}, vPostSamplingProbs)},
      };
    }

    if (vTimings.vPromptN >= 0) {
      vLastJson.push_back({"timings", vTimings.to_json()});
    }
    if (vIsProgress) {
      vLastJson.push_back({"prompt_progress", vProgress.to_json()});
    }
  }

  return lDeltas;
}

json server_task_result_cmpl_partial::mToJsonOaicompatResp() {
  std::vector<json> lEvents;

  if (!vOaiRespCreated) {
    lEvents.push_back(json{
        {"event", "response.created"},
        {"data",
         json{
             {"type", "response.created"},
             {"response",
              json{
                  {"id", vOaiRespId},
                  {"object", "response"},
                  {"status", "in_progress"},
              }},
         }},
    });
    lEvents.push_back(json{
        {"event", "response.in_progress"},
        {"data",
         json{
             {"type", "response.in_progress"},
             {"response",
              json{
                  {"id", vOaiRespId},
                  {"object", "response"},
                  {"status", "in_progress"},
              }},
         }},
    });
  } else if (vIsProgress) {
    lEvents.push_back(json{
        {"event", "response.in_progress"},
        {"data",
         json{
             {"type", "response.in_progress"},
             {"response",
              json{
                  {"id", vOaiRespId},
                  {"object", "response"},
                  {"status", "in_progress"},
              }},
         }},
    });
  }

  for (const common_chat_msg_diff &vDiff : lOaicompatMsgDiffs) {
    if (!vDiff.vReasoningContentDelta.empty()) {
      if (!vThinkingBlockStarted) {
        lEvents.push_back(json{
            {"event", "response.output_item.added"},
            {"data",
             json{
                 {"type", "response.output_item.added"},
                 {"item",
                  json{
                      {"id", vOaiRespReasoningId},
                      {"summary", json::array()},
                      {"type", "reasoning"},
                      {"content", json::array()},
                      {"encrypted_content", ""},
                      {"status", "in_progress"},
                  }},
             }},
        });
        vThinkingBlockStarted = true;
      }
      lEvents.push_back(json{
          {"event", "response.reasoning_text.delta"},
          {"data",
           json{
               {"type", "response.reasoning_text.delta"},
               {"delta", vDiff.vReasoningContentDelta},
               {"item_id", vOaiRespReasoningId},
           }},
      });
    }

    if (!vDiff.vContentDelta.empty()) {
      if (!vTextBlockStarted) {
        lEvents.push_back(json{
            {"event", "response.output_item.added"},
            {"data",
             json{
                 {"type", "response.output_item.added"},
                 {"item",
                  json{
                      {"content", json::array()},
                      {"id", vOaiRespMessageId},
                      {"role", "assistant"},
                      {"status", "in_progress"},
                      {"type", "message"},
                  }},
             }},
        });
        lEvents.push_back(json{
            {"event", "response.content_part.added"},
            {"data",
             json{
                 {"type", "response.content_part.added"},
                 {"item_id", vOaiRespMessageId},
                 {"part",
                  json{
                      {"type", "output_text"},
                      {"text", ""},
                  }},
             }},
        });
        vTextBlockStarted = true;
      }
      lEvents.push_back(json{
          {"event", "response.output_text.delta"},
          {"data",
           json{
               {"type", "response.output_text.delta"},
               {"item_id", vOaiRespMessageId},
               {"delta", vDiff.vContentDelta},
           }},
      });
    }

    if (!vDiff.vToolCallDelta.vName.empty()) {
      lEvents.push_back(json{
          {"event", "response.output_item.added"},
          {"data",
           json{
               {"type", "response.output_item.added"},
               {"item",
                json{
                    {"id", "fc_" + vDiff.vToolCallDelta.vId},
                    {"arguments", ""},
                    {"call_id", "call_" + vDiff.vToolCallDelta.vId},
                    {"name", vDiff.vToolCallDelta.vName},
                    {"type", "function_call"},
                    {"status", "in_progress"},
                }},
           }},
      });
      vOaiRespFcId = vDiff.vToolCallDelta.vId;
    }

    if (!vDiff.vToolCallDelta.vArguments.empty()) {
      lEvents.push_back(json{
          {"event", "response.function_call_arguments.delta"},
          {"data",
           json{
               {"type", "response.function_call_arguments.delta"},
               {"delta", vDiff.vToolCallDelta.vArguments},
               {"item_id", "fc_" + vOaiRespFcId},
           }},
      });
    }
  }

  if (!lEvents.empty()) {
    json &vData = lEvents.back().at("data");
    if (vTimings.vPromptN >= 0) {
      vData.push_back({"timings", vTimings.to_json()});
    }
    if (vIsProgress) {
      vData.push_back({"prompt_progress", vProgress.to_json()});
    }
  }

  return lEvents;
}

json server_task_result_cmpl_partial::mToJsonOaicompatAsr() {
  json vEvent = json{
      {"type", "transcript.text.delta"},
      {"delta", vContent},
  };
  return vEvent;
}

json server_task_result_cmpl_partial::mToJsonAnthropic() {
  json vEvents = json::array();
  bool vFirst = (vNDecoded == 1);
  // use member variables to track block state across streaming calls
  // (anthropic_thinking_block_started, anthropic_text_block_started)

  if (vFirst) {
    vEvents.push_back({{"event", "message_start"},
                       {"data",
                        {{"type", "message_start"},
                         {"message",
                          {{"id", vOaicompatCmplId},
                           {"type", "message"},
                           {"role", "assistant"},
                           {"content", json::array()},
                           {"model", vOaicompatModel},
                           {"stop_reason", nullptr},
                           {"stop_sequence", nullptr},
                           {"usage",
                            {{"cache_read_input_tokens", vNPromptTokensCache},
                             {"input_tokens", vNPromptTokens - vNPromptTokensCache},
                             {"output_tokens", 0}}}}}}}});
  }

  // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
  size_t vThinkingBlockIndex = 0;
  // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
  size_t vTextBlockIndex = vAnthropicHasReasoning ? 1 : 0;

  // use local copies of streaming state (copied from task_result_state in update())
  // these reflect the state BEFORE this chunk was processed
  bool vThinkingStarted = vThinkingBlockStarted;
  bool vTextStarted = vTextBlockStarted;

  for (const auto &vDiff : lOaicompatMsgDiffs) {
    // handle thinking/reasoning content
    if (!vDiff.vReasoningContentDelta.empty()) {
      if (!vThinkingStarted) {
        vEvents.push_back({{"event", "content_block_start"},
                           {"data",
                            {{"type", "content_block_start"},
                             {"index", vThinkingBlockIndex},
                             {"content_block", {{"type", "thinking"}, {"thinking", ""}}}}}});
        vThinkingStarted = true;
      }

      vEvents.push_back({{"event", "content_block_delta"},
                         {"data",
                          {{"type", "content_block_delta"},
                           {"index", vThinkingBlockIndex},
                           {"delta", {{"type", "thinking_delta"}, {"thinking", vDiff.vReasoningContentDelta}}}}}});
    }

    // handle regular text content
    if (!vDiff.vContentDelta.empty()) {
      if (!vTextStarted) {
        vEvents.push_back({{"event", "content_block_start"},
                           {"data",
                            {{"type", "content_block_start"},
                             {"index", vTextBlockIndex},
                             {"content_block", {{"type", "text"}, {"text", ""}}}}}});
        vTextStarted = true;
      }

      vEvents.push_back({{"event", "content_block_delta"},
                         {"data",
                          {{"type", "content_block_delta"},
                           {"index", vTextBlockIndex},
                           {"delta", {{"type", "text_delta"}, {"text", vDiff.vContentDelta}}}}}});
    }

    // handle tool calls
    if (vDiff.vToolCallIndex != std::string::npos) {
      // use anthropic_has_reasoning for thinking block count (persists across calls)
      size_t vContentBlockIndex = (vAnthropicHasReasoning ? 1 : 0) + (vTextStarted ? 1 : 0) + vDiff.vToolCallIndex;

      if (!vDiff.vToolCallDelta.vName.empty()) {
        vEvents.push_back(
            {{"event", "content_block_start"},
             {"data",
              {{"type", "content_block_start"},
               {"index", vContentBlockIndex},
               {"content_block",
                {{"type", "tool_use"}, {"id", vDiff.vToolCallDelta.vId}, {"name", vDiff.vToolCallDelta.vName}}}}}});
      }

      if (!vDiff.vToolCallDelta.vArguments.empty()) {
        vEvents.push_back(
            {{"event", "content_block_delta"},
             {"data",
              {{"type", "content_block_delta"},
               {"index", vContentBlockIndex},
               {"delta", {{"type", "input_json_delta"}, {"partial_json", vDiff.vToolCallDelta.vArguments}}}}}});
      }
    }
  }

  return vEvents;
}

//
// server_task_result_queue
//
json server_task_result_queue::to_json() {
  return json{{"mgs_queue",
               {
                   {"ahead", vAhead},
                   {"ahead_api", vAheadApi},
                   {"ahead_webui", vAheadWebui},
               }}};
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
  return vResType == cTaskResponseTypeOaiEmbd ? mToJsonOaicompat() : mToJsonNonOaicompat();
}

json server_task_result_embd::mToJsonNonOaicompat() {
  return json{
      {"index", vIndex},
      {"embedding", lEmbedding},
  };
}

json server_task_result_embd::mToJsonOaicompat() {
  return json{
      {"index", vIndex},
      {"embedding", lEmbedding[0]},
      {"tokens_evaluated", vNTokens},
  };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
  return json{
      {"index", vIndex},
      {"score", vScore},
      {"tokens_evaluated", vNTokens},
  };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
  json vRes = fFormatErrorResponse(vErrMsg, vErrType);
  if (vErrType == cErrorTypeExceedContextSize) {
    vRes["n_prompt_tokens"] = vNPromptTokens;
    vRes["n_ctx"] = vNCtx;
  }
  return vRes;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
  return json{
      {"idle", vNIdleSlots},
      {"processing", vNProcessingSlots},
      {"deferred", vNTasksDeferred},
      {"t_start", vTStart},

      {"n_prompt_tokens_processed_total", vNPromptTokensProcessedTotal},
      {"t_tokens_generation_total", vTTokensGenerationTotal},
      {"n_tokens_predicted_total", vNTokensPredictedTotal},
      {"t_prompt_processing_total", vTPromptProcessingTotal},

      {"n_tokens_max", vNTokensMax},

      {"n_prompt_tokens_processed", vNPromptTokensProcessed},
      {"t_prompt_processing", vTPromptProcessing},
      {"n_tokens_predicted", vNTokensPredicted},
      {"t_tokens_generation", vTTokensGeneration},

      {"n_decode_total", vNDecodeTotal},
      {"n_busy_slots_total", vNBusySlotsTotal},

      {"n_draft_tokens_total", vNDraftTokensTotal},
      {"n_draft_accepted_total", vNDraftAcceptedTotal},
      {"n_draft_verif_steps_total", vNDraftVerifStepsTotal},
      {"n_accepted_per_pos_total", lNAcceptedPerPosTotal},

      {"slots", vSlotsData},
  };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
  if (vIsSave) {
    return json{
        {"id_slot", vIdSlot},   {"filename", vFilename},          {"n_saved", vNTokens},
        {"n_written", vNBytes}, {"timings", {{"save_ms", vTMs}}},
    };
  }

  return json{
      {"id_slot", vIdSlot},
      {"filename", vFilename},
      {"n_restored", vNTokens},
      {"n_read", vNBytes},
      {"timings", {{"restore_ms", vTMs}}},
  };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
  return json{
      {"id_slot", vIdSlot},
      {"n_erased", vNErased},
  };
}

//
// server_task_result_get_lora
//

json server_task_result_expert_cache::to_json() { return vInfo; }

json server_task_result_get_lora::to_json() {
  json vResult = json::array();
  for (size_t vI = 0; vI < lLoras.size(); ++vI) {
    auto &vLora = lLoras[vI];
    json vEntry = {
        {"id", vI},
        {"path", vLora.vInfo.vPath},
        {"scale", vLora.vInfo.vScale},
        {"task_name", vLora.vInfo.vTaskName},
        {"prompt_prefix", vLora.vInfo.vPromptPrefix},
    };
    if (!vLora.vAloraInvocationTokens.empty()) {
      vEntry["alora_invocation_string"] = vLora.vAloraInvocationString;
      vEntry["alora_invocation_tokens"] = vLora.vAloraInvocationTokens;
    }
    vResult.push_back(std::move(vEntry));
  }
  return vResult;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() { return json{{"success", true}}; }

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
  size_t vRes = 0;

  for (const auto &vState : lStates) {
    vRes += vState.size();
  }

  return vRes;
}

size_t server_prompt_cache::mNTokens() const {
  size_t vRes = 0;

  for (const auto &vState : lStates) {
    vRes += vState.vPrompt.mNTokens();
  }

  return vRes;
}

server_prompt_cache_state *server_prompt_cache::mAlloc(const server_prompt &pPrompt, size_t pStateSizeTgt,
                                                       size_t pStateSizeDft) {
  // first check if the current state is contained fully in the cache
  for (auto vIt = lStates.begin(); vIt != lStates.end(); ++vIt) {
    const int cCurLcpLen = vIt->vPrompt.vTokens.mGetCommonPrefix(pPrompt.vTokens);

    if (cCurLcpLen == (int)pPrompt.vTokens.size()) {
      SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
      return nullptr;
    }
  }

  // calculate checkpoints size to see if it will fit with the prompt
  size_t vCheckpointsSize = 0;
  for (const auto &vCkpt : pPrompt.lCheckpoints) {
    vCheckpointsSize += vCkpt.size();
  }

  const size_t cStateSizeNew = pStateSizeTgt + pStateSizeDft + vCheckpointsSize;

  // skip over-limit entries to avoid disturbing the cache
  if (vLimitSize > 0 && cStateSizeNew > vLimitSize) {
    SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
            cStateSizeNew / (1024.0 * 1024.0), vLimitSize / (1024.0 * 1024.0));
    return nullptr;
  }

  // remove any cached prompts that are fully contained in the current prompt
  for (auto vIt = lStates.begin(); vIt != lStates.end();) {
    const int cLen = vIt->vPrompt.vTokens.mGetCommonPrefix(pPrompt.vTokens);

    if (cLen == (int)vIt->vPrompt.vTokens.size()) {
      SRV_TRC(" - removing obsolete cached prompt with length %d\n", cLen);

      vIt = lStates.erase(vIt);
    } else {
      ++vIt;
    }
  }

  if (vLimitSize > 0) {
    // make room before allocating the new vectors to avoid breaching the limit
    while (!lStates.empty() && size() + cStateSizeNew > vLimitSize) {
      SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
              lStates.front().size() / (1024.0 * 1024.0));

      lStates.pop_front();
    }
  }

  std::vector<uint8_t> lStateDataTgt;
  std::vector<uint8_t> lStateDataDft;

  // check if we can allocate enough memory for the new state
  try {
    lStateDataTgt.resize(pStateSizeTgt);
    lStateDataDft.resize(pStateSizeDft);
  } catch (const std::bad_alloc &vE) {
    SRV_ERR("failed to allocate memory for prompt cache state: %s\n", vE.what());

    vLimitSize = std::max<size_t>(1, 0.4 * size());

    SRV_WRN(" - cache size limit reduced to %.3f MiB\n", vLimitSize / (1024.0 * 1024.0));

    mUpdate();

    return nullptr;
  }

  lStates.push_back({
      /*.vPrompt =*/{
          /*.vTokens      =*/pPrompt.vTokens.mClone(),
          /*.lCheckpoints =*/pPrompt.lCheckpoints,
      },
      /*.vData   =*/
      {
          /*.lMain =*/std::move(lStateDataTgt),
          /*.lDrft =*/std::move(lStateDataDft),
      },
  });

  return &lStates.back();
}

bool server_prompt_cache::mLoad(server_prompt &pPrompt, const server_tokens &pTokensNew, mgs_context *pCtxTgt,
                                mgs_context *pCtxDft, int32_t pIdSlot) {
  const int cLcpBest = pPrompt.vTokens.mGetCommonPrefix(pTokensNew);

  float vFKeepBest =
      pPrompt.vTokens.size() > 0 ? float(cLcpBest) / pPrompt.vTokens.size() : -1.0f; // empty slot: any cache entry wins
  float vFSimBest = float(cLcpBest) / pTokensNew.size();

  SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", vFKeepBest, vFSimBest);

  auto vItBest = lStates.end();

  // find the most similar cached prompt, that would also preserve the most context
  for (auto vIt = lStates.begin(); vIt != lStates.end(); ++vIt) {
    const int cLcpCur = vIt->vPrompt.vTokens.mGetCommonPrefix(pTokensNew);

    const float cFKeepCur = float(cLcpCur) / vIt->vPrompt.vTokens.size();
    const float cFSimCur = float(cLcpCur) / pTokensNew.size();

    SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", vIt->vPrompt.vTokens.size(),
            cLcpCur, cFKeepCur, cFSimCur);

    // don't trash large prompts
    if (cFKeepCur < 0.25f) {
      continue;
    }

    if (vFKeepBest < cFKeepCur && vFSimBest < cFSimCur) {
      vFKeepBest = cFKeepCur;
      vFSimBest = cFSimCur;

      vItBest = vIt;
    }
  }

  if (vItBest != lStates.end()) {
    SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", vFKeepBest, vFSimBest);

    {
      auto &vData = vItBest->vData.lMain;

      const size_t cSize = vData.size();
      const size_t cN = mgs_state_seq_set_data_ext(pCtxTgt, vData.data(), cSize, pIdSlot, 0);
      if (cN != cSize) {
        SRV_ERR("failed to restore state with size %zu\n", cSize);

        return false;
      }

      vData.clear();
      vData.shrink_to_fit();
    }

    {
      auto &vData = vItBest->vData.lDrft;

      if (!vData.empty()) {
        GGML_ASSERT(pCtxDft);

        const size_t cSize = vData.size();
        const size_t cN = mgs_state_seq_set_data_ext(pCtxDft, vData.data(), cSize, pIdSlot, 0);
        if (cN != cSize) {
          SRV_WRN("failed to restore state with size %zu\n", cSize);

          return false;
        }

        vData.clear();
        vData.shrink_to_fit();
      }
    }

    pPrompt = std::move(vItBest->vPrompt);

    lStates.erase(vItBest);
  }

  return true;
}

void server_prompt_cache::mUpdate() {
  if (vLimitSize > 0) {
    while (!lStates.empty() && size() > vLimitSize) {
      SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n",
              lStates.front().size() / (1024.0 * 1024.0));

      lStates.pop_front();
    }
  }

  // average size per token
  const float cSizePerToken = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, mNTokens())));

  // dynamically increase the token limit if it can fit in the memory limit
  const size_t cLimitTokensCur =
      vLimitSize > 0 ? std::max<size_t>(vLimitTokens, vLimitSize / cSizePerToken) : vLimitTokens;

  if (vLimitTokens > 0) {
    while (!lStates.empty() && mNTokens() > cLimitTokensCur) {
      SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n", vLimitTokens,
              cLimitTokensCur, lStates.front().size() / (1024.0 * 1024.0));

      lStates.pop_front();
    }
  }

  SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n", lStates.size(),
          size() / (1024.0 * 1024.0), vLimitSize / (1024.0 * 1024.0), vLimitTokens, cLimitTokensCur);

  for (const auto &vState : lStates) {
    SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n", (const void *)&vState,
            vState.vPrompt.mNTokens(), vState.vPrompt.lCheckpoints.size(), vState.size() / (1024.0 * 1024.0));
  }
}
