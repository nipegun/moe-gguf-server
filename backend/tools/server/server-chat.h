// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include <nlohmann/json_fwd.hpp>

using json = nlohmann::ordered_json;

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json fServerChatConvertResponsesToChatcmpl(const json &pBody);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json fServerChatConvertAnthropicToOai(const json &pBody);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json fConvertTranscriptionsToChatcmpl(const json &pBody, const common_chat_templates *pTmpls,
                                      const std::map<std::string, uploaded_file> &pInFiles,
                                      std::vector<vRawBuffer> &pOutFiles);

json fServerChatMsgDiffToJsonOaicompat(const common_chat_msg_diff &pDiff);
