#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "log.h"

#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "peg-parser.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <map>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static std::string fFormatTime(const std::chrono::system_clock::time_point &pNow, const std::string &pFormat) {
  auto vTime = std::chrono::system_clock::to_time_t(pNow);
  auto vLocalTime = *std::localtime(&vTime);
  std::ostringstream vSs;
  vSs << std::put_time(&vLocalTime, pFormat.c_str());
  auto vRes = vSs.str();
  return vRes;
}

static json fSafeArgsParse(const std::string &pToParse) {
  std::string vStripped = pToParse;
  if (pToParse.at(0) == '"' && pToParse.at(pToParse.length() - 1) == '"') {
    vStripped = pToParse.substr(1, pToParse.length() - 1);
  }
  try {
    return json::parse(vStripped);
  } catch (json::exception &vE) {
    return vStripped;
  }
}

static std::string fStringDiff(const std::string &pLast, const std::string &pCurrent) {
  if (pLast.empty()) {
    return pCurrent;
  }
  if (!fStringStartsWith(pCurrent, pLast)) {
    if (fStringStartsWith(pLast, pCurrent)) {
      // This happens if the last generation ended on a partial stop word (not erased),
      // and the current ended on a stop word (erased).
      return "";
    }
    throw std::runtime_error("Invalid diff: '" + pLast + "' not found at start of '" + pCurrent + "'");
  }
  return pCurrent.substr(pLast.size());
}

static bool fHasContentOrToolCalls(const common_chat_msg &pMsg) {
  return !pMsg.vContent.empty() || !pMsg.lToolCalls.empty();
}

std::string common_chat_msg::mRenderContent(const std::string &pDelimiter) const {
  if (!vContent.empty() && !lContentParts.empty()) {
    throw std::runtime_error("Cannot specify both content and content_parts");
  }
  if (!vContent.empty()) {
    return vContent;
  }

  std::string vText;
  for (const auto &vPart : lContentParts) {
    if (vPart.vType == "text") {
      if (!vText.empty()) {
        vText += pDelimiter;
      }
      vText += vPart.vText;
    }
  }
  return vText;
}

common_chat_role fCommonChatRoleFromString(const std::string &pRole) {
  if (pRole == "system") {
    return cCommonChatRoleSystem;
  }
  if (pRole == "assistant") {
    return cCommonChatRoleAssistant;
  }
  if (pRole == "user") {
    return cCommonChatRoleUser;
  }
  if (pRole == "tool") {
    return cCommonChatRoleTool;
  }
  return cCommonChatRoleUnknown;
}

const char *fCommonChatRoleToString(common_chat_role pRole) {
  switch (pRole) {
  case cCommonChatRoleSystem:
    return "system";
  case cCommonChatRoleAssistant:
    return "assistant";
  case cCommonChatRoleUser:
    return "user";
  case cCommonChatRoleTool:
    return "tool";
  case cCommonChatRoleUnknown:
    return "";
  }
  return "";
}

json common_chat_msg_delimiters::to_json() const {
  json vResult = json::array();
  for (const auto &vD : lDelimiters) {
    vResult.push_back({
        {"role", fCommonChatRoleToString(vD.vRole)},
        {"delimiter", vD.vDelimiter},
    });
  }
  return vResult;
}

common_chat_msg_delimiters fCommonChatMsgDelimitersParse(const json &pDelimiters) {
  common_chat_msg_delimiters vResult;

  if (!pDelimiters.is_array()) {
    return vResult;
  }

  vResult.lDelimiters.reserve(pDelimiters.size());
  for (const auto &vD : pDelimiters) {
    if (!vD.is_object()) {
      continue;
    }
    vResult.lDelimiters.push_back({
        fCommonChatRoleFromString(vD.value("role", std::string())),
        vD.value("delimiter", std::string()),
    });
  }

  return vResult;
}

void common_chat_msg_delimiters::mTokenize(const mgs_vocab *pVocab) {
  for (auto &vD : lDelimiters) {
    vD.vTokens = fCommonTokenize(pVocab, vD.vDelimiter, false, true);
  }
}

common_chat_msg_spans common_chat_msg_delimiters::mSplit(const mgs_tokens &pTokens,
                                                         const std::map<size_t, size_t> &pSkips) const {
  std::vector<std::pair<common_chat_role, size_t>> lMatches;

  auto vSkip = pSkips.begin();
  for (size_t vI = 0; vI < pTokens.size();) {
    if (vSkip != pSkips.end() && vI == vSkip->first) {
      vI += vSkip->second;
      ++vSkip;
      continue;
    }
    for (const auto &vD : lDelimiters) {
      if (vI + vD.vTokens.size() > pTokens.size()) {
        continue;
      }
      if (std::equal(vD.vTokens.begin(), vD.vTokens.end(), pTokens.begin() + vI)) {
        lMatches.emplace_back(vD.vRole, vI);
        break;
      }
    }
    vI++;
  }

  lMatches.emplace_back(cCommonChatRoleUnknown, pTokens.size());

  common_chat_msg_spans vSpans;
  for (size_t vI = 0; vI + 1 < lMatches.size(); vI++) {
    const auto &vCurr = lMatches[vI];
    const auto &vNext = lMatches[vI + 1];
    vSpans.mAdd(vCurr.first, vCurr.second, vNext.second - vCurr.second);
  }

  return vSpans;
}

json common_chat_msg::mToJsonOaicompat(bool pConcatTypedText) const {
  if (!vContent.empty() && !lContentParts.empty()) {
    throw std::runtime_error("Cannot specify both content and content_parts");
  }
  json vJmsg{
      {"role", vRole},
  };
  if (!vContent.empty()) {
    vJmsg["content"] = vContent;
  } else if (!lContentParts.empty()) {
    if (pConcatTypedText || mContainsMedia()) {
      std::string vText;
      bool vLastWasMediaMarker = false;
      // join parts with newline, do not add newline before or after media markers
      for (const auto &vPart : lContentParts) {
        bool vAddNewLine = true;
        if (vPart.vType == "text") {
          vAddNewLine = !vLastWasMediaMarker && !vText.empty();
          vLastWasMediaMarker = false;
        } else if (vPart.vType == "media_marker") {
          vAddNewLine = false;
          vLastWasMediaMarker = true;
        } else {
          LOG_WRN("Ignoring content part type: %s\n", vPart.vType.c_str());
          continue;
        }

        if (vAddNewLine) {
          vText += '\n';
        }

        vText += vPart.vText;
      }
      vJmsg["content"] = vText;
    } else {
      auto &vParts = vJmsg["content"] = json::array();
      for (const auto &vPart : lContentParts) {
        vParts.push_back({
            {"type", vPart.vType},
            {"text", vPart.vText},
        });
      }
    }
  } else {
    vJmsg["content"] = "";
  }
  if (!vReasoningContent.empty()) {
    vJmsg["reasoning_content"] = vReasoningContent;
  }
  if (!vToolName.empty()) {
    vJmsg["name"] = vToolName;
  }
  if (!vToolCallId.empty()) {
    vJmsg["tool_call_id"] = vToolCallId;
  }
  if (!lToolCalls.empty()) {
    vJmsg["tool_calls"] = json::array();
    auto &vJtoolCalls = vJmsg["tool_calls"];
    for (const auto &vToolCall : lToolCalls) {
      json vTc{
          {"type", "function"},
          {"function",
           {
               {"name", vToolCall.vName},
               {"arguments", json(vToolCall.vArguments)},
           }},
      };
      if (!vToolCall.vId.empty()) {
        vTc["id"] = vToolCall.vId;
      }
      // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
      // We only generate a random id for the ones that don't generate one by themselves
      // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
      // {"id", vToolCall.vId.empty() ? fGenToolCallId() : vToolCall.vId},
      vJtoolCalls.push_back(vTc);
    }
  }

  return vJmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::mComputeDiffs(const common_chat_msg &pMsgPrv,
                                                                      const common_chat_msg &pMsgNew) {
  std::vector<common_chat_msg_diff> lDiffs;
  if (pMsgNew.lToolCalls.size() > pMsgPrv.lToolCalls.size()) {
    lDiffs.reserve(pMsgNew.lToolCalls.size() - pMsgPrv.lToolCalls.size() + 3);
  } else {
    lDiffs.reserve(3);
  }

  // TODO: these can become expensive for long messages - how to optimize?
  if (pMsgPrv.vReasoningContent != pMsgNew.vReasoningContent) {
    auto &vDiff = lDiffs.emplace_back();
    vDiff.vReasoningContentDelta = fStringDiff(pMsgPrv.vReasoningContent, pMsgNew.vReasoningContent);
  }
  if (pMsgPrv.vContent != pMsgNew.vContent) {
    auto &vDiff = lDiffs.emplace_back();
    vDiff.vContentDelta = fStringDiff(pMsgPrv.vContent, pMsgNew.vContent);
  }

  if (pMsgNew.lToolCalls.size() < pMsgPrv.lToolCalls.size()) {
    std::string vErr = "Invalid diff: now finding less tool calls!\n";
    vErr += "  Previous (" + std::to_string(pMsgPrv.lToolCalls.size()) + "):\n";
    for (const auto &vTc : pMsgPrv.lToolCalls) {
      vErr += "    - name: '" + vTc.vName + "', args: '" + vTc.vArguments + "'\n";
    }
    vErr += "  Current (" + std::to_string(pMsgNew.lToolCalls.size()) + "):\n";
    for (const auto &vTc : pMsgNew.lToolCalls) {
      vErr += "    - name: '" + vTc.vName + "', args: '" + vTc.vArguments + "'\n";
    }
    vErr += "  Current msg text content:\n" + pMsgNew.vContent + "\n";
    throw std::runtime_error(vErr);
  }

  if (!pMsgPrv.lToolCalls.empty()) {
    const auto cIdx = pMsgPrv.lToolCalls.size() - 1;
    const auto &vPref = pMsgPrv.lToolCalls[cIdx];
    const auto &vNewf = pMsgNew.lToolCalls[cIdx];
    // Allow tool name to change during incremental parsing:
    // - empty -> non-empty (initial discovery)
    // - prefix -> longer string (name grows as more input is parsed)
    if (vPref.vName != vNewf.vName && !vPref.vName.empty() && !vNewf.vName.empty()) {
      // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
      bool vIsPrefix = (vNewf.vName.rfind(vPref.vName, 0) == 0);
      if (!vIsPrefix) {
        LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", vPref.vName.c_str(), vNewf.vName.c_str());
        throw std::runtime_error("Invalid diff: tool call mismatch!");
      }
    }
    const auto cArgsDiff = fStringDiff(vPref.vArguments, vNewf.vArguments);
    if (!cArgsDiff.empty() || vPref.vId != vNewf.vId || vPref.vName != vNewf.vName) {
      auto &vDiff = lDiffs.emplace_back();
      vDiff.vToolCallIndex = cIdx;
      if (vPref.vId != vNewf.vId || vPref.vName != vNewf.vName) {
        vDiff.vToolCallDelta.vId = vNewf.vId;
        vDiff.vToolCallDelta.vName = vNewf.vName;
      }
      vDiff.vToolCallDelta.vArguments = cArgsDiff;
    }
  }
  for (size_t vIdx = pMsgPrv.lToolCalls.size(); vIdx < pMsgNew.lToolCalls.size(); ++vIdx) {
    auto &vDiff = lDiffs.emplace_back();
    vDiff.vToolCallIndex = vIdx;
    vDiff.vToolCallDelta = pMsgNew.lToolCalls[vIdx];
  }

  return lDiffs;
}

using dChatTemplateCaps = jinja::caps;

struct common_chat_templates {
  bool vAddBos;
  bool vAddEos;
  bool vHasExplicitTemplate; // Model had builtin template or template overridden was specified.
  std::unique_ptr<common_chat_template> vTemplateDefault; // always set (defaults to chatml)
  std::unique_ptr<common_chat_template> vTemplateToolUse;
};

common_chat_tool_choice fCommonChatToolChoiceParseOaicompat(const std::string &pToolChoice) {
  if (pToolChoice == "auto") {
    return cCommonChatToolChoiceAuto;
  }
  if (pToolChoice == "none") {
    return cCommonChatToolChoiceNone;
  }
  if (pToolChoice == "required") {
    return cCommonChatToolChoiceRequired;
  }
  throw std::invalid_argument("Invalid tool_choice: " + pToolChoice);
}

bool fCommonChatTemplatesSupportEnableThinking(const common_chat_templates *pChatTemplates) {
  common_chat_templates_inputs vInputs;
  vInputs.vReasoningFormat = cCommonReasoningFormatDeepseek;
  common_chat_msg vMsg;
  vMsg.vRole = "user";
  vMsg.vContent = "test";
  vInputs.lMessages = {vMsg};
  vInputs.vEnableThinking = true;
  vInputs.vAddGenerationPrompt = true;
  vInputs.vReasoningFormat = cCommonReasoningFormatDeepseek;

  auto vParams = fCommonChatTemplatesApply(pChatTemplates, vInputs);
  return vParams.vSupportsThinking;
}

std::vector<common_chat_msg> fCommonChatMsgsParseOaicompat(const json &pMessages) {
  std::vector<common_chat_msg> lMsgs;

  try {
    if (!pMessages.is_array()) {
      throw std::invalid_argument("Expected 'messages' to be an array, got " + pMessages.dump());
    }

    for (const auto &vMessage : pMessages) {
      if (!vMessage.is_object()) {
        throw std::invalid_argument("Expected 'message' to be an object, got " + vMessage.dump());
      }

      common_chat_msg vMsg;
      if (!vMessage.contains("role")) {
        throw std::invalid_argument("Missing 'role' in message: " + vMessage.dump());
      }
      vMsg.vRole = vMessage.at("role");

      auto vHasContent = vMessage.contains("content");
      auto vHasToolCalls = vMessage.contains("tool_calls");
      if (vHasContent) {
        const auto &vContent = vMessage.at("content");
        if (vContent.is_string()) {
          vMsg.vContent = vContent;
        } else if (vContent.is_array()) {
          for (const auto &vPart : vContent) {
            if (!vPart.contains("type")) {
              throw std::invalid_argument("Missing content part type: " + vPart.dump());
            }
            const auto &vType = vPart.at("type");
            if (vType != "text" && vType != "media_marker") {
              throw std::invalid_argument("Unsupported content part type: " + vType.dump());
            }
            common_chat_msg_content_part vMsgPart;
            vMsgPart.vType = vType;
            vMsgPart.vText = vPart.at("text");
            vMsg.lContentParts.push_back(vMsgPart);
          }
        } else if (!vContent.is_null()) {
          throw std::invalid_argument("Invalid 'content' type: expected string or array, got " + vContent.dump());
        }
      }
      if (vHasToolCalls) {
        for (const auto &vToolCall : vMessage.at("tool_calls")) {
          common_chat_tool_call vTc;
          if (!vToolCall.contains("type")) {
            throw std::invalid_argument("Missing tool call type: " + vToolCall.dump());
          }
          const auto &vType = vToolCall.at("type");
          if (vType != "function") {
            throw std::invalid_argument("Unsupported tool call type: " + vToolCall.dump());
          }
          if (!vToolCall.contains("function")) {
            throw std::invalid_argument("Missing tool call function: " + vToolCall.dump());
          }
          const auto &vFc = vToolCall.at("function");
          if (!vFc.contains("name")) {
            throw std::invalid_argument("Missing tool call name: " + vToolCall.dump());
          }
          vTc.vName = vFc.at("name");
          const auto &vArgs = vFc.at("arguments");
          if (vArgs.is_string()) {
            vTc.vArguments = vArgs;
          } else {
            vTc.vArguments = vArgs.dump();
          }
          if (vToolCall.contains("id")) {
            vTc.vId = vToolCall.at("id");
          }
          vMsg.lToolCalls.push_back(vTc);
        }
      }
      if (!vHasContent && !vHasToolCalls) {
        throw std::invalid_argument("Expected 'content' or 'tool_calls'");
      }
      if (vMessage.contains("reasoning_content")) {
        vMsg.vReasoningContent = vMessage.at("reasoning_content");
      }
      if (vMessage.contains("name")) {
        vMsg.vToolName = vMessage.at("name");
      }
      if (vMessage.contains("tool_call_id")) {
        vMsg.vToolCallId = vMessage.at("tool_call_id");
      }

      lMsgs.push_back(vMsg);
    }
  } catch (const std::exception &vE) {
    // @ngxson : disable otherwise it's bloating the API response
    // printf("%s\n", std::string("; messages = ") + pMessages.dump(2));
    throw std::runtime_error("Failed to parse messages: " + std::string(vE.what()));
  }

  return lMsgs;
}

static json fRenderMessageToJson(const std::vector<common_chat_msg> &pMsgs, const jinja::caps &pC) {
  if (!pC.vSupportsStringContent && !pC.vSupportsTypedContent) {
    LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may "
            "lead to issues.\n",
            __func__);
  }

  bool vOnlyStringAccepted = pC.vSupportsStringContent && !pC.vSupportsTypedContent;
  bool vOnlyTypedAccepted = !pC.vSupportsStringContent && pC.vSupportsTypedContent;

  json vMessages = json::array();
  for (const auto &vMsg : pMsgs) {
    if (vOnlyStringAccepted) {
      json vJmsg = vMsg.mToJsonOaicompat(/* concat_typed_text= */ true);
      vMessages.push_back(vJmsg);
    } else if (vOnlyTypedAccepted) {
      json vJmsg = vMsg.mToJsonOaicompat(/* concat_typed_text= */ false);
      if (vJmsg.at("content").is_string()) {
        vJmsg["content"] = json::array({json{
            {"type", "text"},
            {"text", vJmsg.at("content").get<std::string>()},
        }});
      }
      vMessages.push_back(vJmsg);
    } else {
      json vJmsg = vMsg.mToJsonOaicompat(/* concat_typed_text= */ false);
      vMessages.push_back(vJmsg);
    }
  }
  return vMessages;
}

json fCommonChatMsgsToJsonOaicompat(const std::vector<common_chat_msg> &pMsgs, bool pConcatTypedText) {
  jinja::caps vC;
  vC.vSupportsStringContent = true;
  vC.vSupportsTypedContent = !pConcatTypedText;
  return fRenderMessageToJson(pMsgs, vC);
}

json fCommonChatToolsToJsonOaicompat(const std::vector<common_chat_tool> &pTools) {
  if (pTools.empty()) {
    return json();
  }

  auto vResult = json::array();
  for (const auto &vTool : pTools) {
    vResult.push_back({
        {"type", "function"},
        {"function",
         {
             {"name", vTool.vName},
             {"description", vTool.vDescription},
             {"parameters", json::parse(vTool.vParameters)},
         }},
    });
  }
  return vResult;
}

std::vector<common_chat_tool> fCommonChatToolsParseOaicompat(const json &pTools) {
  std::vector<common_chat_tool> lResult;

  try {
    if (!pTools.is_null()) {
      if (!pTools.is_array()) {
        throw std::invalid_argument("Expected 'tools' to be an array, got " + pTools.dump());
      }
      for (const auto &vTool : pTools) {
        if (!vTool.contains("type")) {
          throw std::invalid_argument("Missing tool type: " + vTool.dump());
        }
        const auto &vType = vTool.at("type");
        if (!vType.is_string() || vType != "function") {
          throw std::invalid_argument("Unsupported tool type: " + vTool.dump());
        }
        if (!vTool.contains("function")) {
          throw std::invalid_argument("Missing tool function: " + vTool.dump());
        }

        const auto &vFunction = vTool.at("function");
        lResult.push_back({
            /* .vName = */ vFunction.at("name"),
            /* .vDescription = */ vFunction.value("description", ""),
            /* .vParameters = */ vFunction.value("parameters", json::object()).dump(),
        });
      }
    }
  } catch (const std::exception &vE) {
    throw std::runtime_error("Failed to parse tools: " + std::string(vE.what()) + "; tools = " + pTools.dump(2));
  }

  return lResult;
}

common_chat_continuation fCommonChatContinuationParse(const nlohmann::ordered_json &pValue) {
  if (pValue.is_boolean() && pValue.get<bool>()) {
    return cCommonChatContinuationAuto;
  }
  if (pValue.is_string()) {
    auto vValueStr = pValue.get<std::string>();
    if (vValueStr == "reasoning_content") {
      return cCommonChatContinuationReasoning;
    }
    if (vValueStr == "content") {
      return cCommonChatContinuationContent;
    }
  }
  return cCommonChatContinuationNone;
}

bool fCommonChatVerifyTemplate(const std::string &pTmpl) {
  try {
    common_chat_msg vMsg;
    vMsg.vRole = "user";
    vMsg.vContent = "test";

    auto vTmpls = fCommonChatTemplatesInit(/* model= */ nullptr, pTmpl);

    common_chat_templates_inputs vInputs;
    vInputs.lMessages = {vMsg};

    fCommonChatTemplatesApply(vTmpls.get(), vInputs);
    return true;
  } catch (const std::exception &vE) {
    LOG_ERR("%s: failed to apply template: %s\n", __func__, vE.what());
    return false;
  }
}

std::string fCommonChatFormatExample(const struct common_chat_templates *pTmpls,
                                     const std::map<std::string, std::string> &pChatTemplateKwargs) {
  common_chat_templates_inputs vInputs;
  vInputs.vAddBos = pTmpls->vAddBos;
  vInputs.vAddEos = pTmpls->vAddEos;
  vInputs.dChatTemplateKwargs = pChatTemplateKwargs;
  auto vAddSimpleMsg = [&](auto pRole, auto pContent) {
    common_chat_msg vMsg;
    vMsg.vRole = pRole;
    vMsg.vContent = pContent;
    vInputs.lMessages.push_back(vMsg);
  };
  vAddSimpleMsg("system", "You are a helpful assistant");
  vAddSimpleMsg("user", "Hello");
  vAddSimpleMsg("assistant", "Hi there");
  vAddSimpleMsg("user", "How are you?");
  return fCommonChatTemplatesApply(pTmpls, vInputs).vPrompt;
}

#define CHATML_TEMPLATE_SRC                                                                                            \
  "{%- for message in messages -%}\n"                                                                                  \
  "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"                                \
  "{%- endfor -%}\n"                                                                                                   \
  "{%- if add_generation_prompt -%}\n"                                                                                 \
  "  {{- '<|im_start|>assistant\n' -}}\n"                                                                              \
  "{%- endif -%}"

// DeepSeek-V4.1 files written by the DwarfStar tools carry no chat template: use this one, which follows
// the encoding of the model release (DSML tool calls with a leading space, numeric reasoning effort)
#define DEEPSEEK41_TEMPLATE_SRC                                                                                        \
  "{#- DeepSeek-V4.1: the prompt format of encoding/encoding.py in the model release.\n"                               \
  "    Variables: enable_thinking (default true), drop_thinking (default true),\n"                                     \
  "    reasoning_effort (an integer in [1, 100] or low, medium, high, max; default high). -#}\n"                       \
  "{%- set dsml = '｜DSML｜' -%}\n"                                                                                  \
  "{%- if thinking is not defined -%}\n"                                                                               \
  "  {%- if enable_thinking is defined and enable_thinking is not none -%}\n"                                          \
  "    {%- set thinking = enable_thinking -%}\n"                                                                       \
  "  {%- else -%}\n"                                                                                                   \
  "    {%- set thinking = true -%}\n"                                                                                  \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endif -%}\n"                                                                                                    \
  "{%- if add_generation_prompt is not defined -%}\n"                                                                  \
  "  {%- set add_generation_prompt = false -%}\n"                                                                      \
  "{%- endif -%}\n"                                                                                                    \
  "{%- if drop_thinking is not defined -%}\n"                                                                          \
  "  {%- set drop_thinking = true -%}\n"                                                                               \
  "{%- endif -%}\n"                                                                                                    \
  "{%- if reasoning_effort is not defined or reasoning_effort is none -%}\n"                                           \
  "  {%- set reasoning_effort = 'high' -%}\n"                                                                          \
  "{%- endif -%}\n"                                                                                                    \
  "{%- set effort_map = {'low': 50, 'medium': 62, 'high': 75, 'max': 100} -%}\n"                                       \
  "{%- if reasoning_effort is string and reasoning_effort in effort_map -%}\n"                                         \
  "  {%- set budget = effort_map[reasoning_effort] -%}\n"                                                              \
  "{%- elif reasoning_effort is number and reasoning_effort >= 1 and reasoning_effort <= 100 -%}\n"                    \
  "  {%- set budget = reasoning_effort | int -%}\n"                                                                    \
  "{%- else -%}\n"                                                                                                     \
  "  {{- raise_exception('Invalid reasoning effort: use an integer in [1, 100] or low, medium, high, ma"               \
  "x') -}}\n"                                                                                                          \
  "{%- endif -%}\n"                                                                                                    \
  "{%- macro text_of(content) -%}\n"                                                                                   \
  "  {%- if content is string -%}\n"                                                                                   \
  "    {{- content -}}\n"                                                                                              \
  "  {%- elif content is mapping -%}\n"                                                                                \
  "    {{- content.text if content.text is defined else '' -}}\n"                                                      \
  "  {%- elif content is iterable -%}\n"                                                                               \
  "    {%- set parts = namespace(list=[]) -%}\n"                                                                       \
  "    {%- for b in content -%}\n"                                                                                     \
  "      {%- if b.type == 'text' -%}\n"                                                                                \
  "        {%- set parts.list = parts.list + [b.text] -%}\n"                                                           \
  "      {%- elif b.type == 'image' or b.type == 'image_url' -%}\n"                                                    \
  "        {%- set parts.list = parts.list + ['<｜deepseek_image｜>'] -%}\n"                                         \
  "      {%- endif -%}\n"                                                                                              \
  "    {%- endfor -%}\n"                                                                                               \
  "    {{- parts.list | join('\\n\\n') -}}\n"                                                                          \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endmacro -%}\n"                                                                                                 \
  "{#- merge tool results and consecutive user turns into one user turn -#}\n"                                         \
  "{%- set ns = namespace(msgs=[]) -%}\n"                                                                              \
  "{%- for m in messages -%}\n"                                                                                        \
  "  {%- set prev = ns.msgs[-1] if ns.msgs else none -%}\n"                                                            \
  "  {%- if m.role == 'tool' or m.role == 'user' -%}\n"                                                                \
  "    {%- if m.role == 'tool' -%}\n"                                                                                  \
  "      {%- set part = '<tool_result>' ~ text_of(m.content) ~ '</tool_result>' -%}\n"                                 \
  "    {%- else -%}\n"                                                                                                 \
  "      {%- set part = text_of(m.content) -%}\n"                                                                      \
  "    {%- endif -%}\n"                                                                                                \
  "    {%- if prev is not none and prev.role == 'user' -%}\n"                                                          \
  "      {%- set ns.msgs = ns.msgs[:-1] + [{'role': 'user', 'content': prev.content ~ '\\n\\n' ~ part}] -%}\n"         \
  "    {%- else -%}\n"                                                                                                 \
  "      {%- set ns.msgs = ns.msgs + [{'role': 'user', 'content': part}] -%}\n"                                        \
  "    {%- endif -%}\n"                                                                                                \
  "  {%- elif m.role == 'system' -%}\n"                                                                                \
  "    {%- set ns.msgs = ns.msgs + [{'role': 'system', 'content': text_of(m.content), 'tools': m.tools "               \
  "if m.tools is defined else none}] -%}\n"                                                                            \
  "  {%- elif m.role == 'latest_reminder' -%}\n"                                                                       \
  "    {%- set ns.msgs = ns.msgs + [{'role': 'latest_reminder', 'content': text_of(m.content)}] -%}\n"                 \
  "  {%- elif m.role == 'assistant' -%}\n"                                                                             \
  "    {%- set ns.msgs = ns.msgs + [m] -%}\n"                                                                          \
  "  {%- else -%}\n"                                                                                                   \
  "    {{- raise_exception('Unknown role: ' ~ m.role) -}}\n"                                                           \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endfor -%}\n"                                                                                                   \
  "{#- the tools attach to a leading system message, or to a new empty one -#}\n"                                      \
  "{%- if tools is defined and tools -%}\n"                                                                            \
  "  {%- if ns.msgs and ns.msgs[0].role == 'system' and ns.msgs[0].tools -%}\n"                                        \
  "  {%- elif ns.msgs and ns.msgs[0].role == 'system' -%}\n"                                                           \
  "    {%- set ns.msgs = [{'role': 'system', 'content': ns.msgs[0].content, 'tools': tools}] + ns.msgs["               \
  "1:] -%}\n"                                                                                                          \
  "  {%- else -%}\n"                                                                                                   \
  "    {%- set ns.msgs = [{'role': 'system', 'content': '', 'tools': tools}] + ns.msgs -%}\n"                          \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endif -%}\n"                                                                                                    \
  "{#- mid-conversation system messages count as user turns, tools disable drop_thinking -#}\n"                        \
  "{%- set st = namespace(last_user=-1, has_tools=false) -%}\n"                                                        \
  "{%- for m in ns.msgs -%}\n"                                                                                         \
  "  {%- if m.role == 'user' or (m.role == 'system' and loop.index0 > 0) -%}\n"                                        \
  "    {%- set st.last_user = loop.index0 -%}\n"                                                                       \
  "  {%- endif -%}\n"                                                                                                  \
  "  {%- if m.role == 'system' and m.tools -%}\n"                                                                      \
  "    {%- set st.has_tools = true -%}\n"                                                                              \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endfor -%}\n"                                                                                                   \
  "{%- set effective_drop = drop_thinking and not st.has_tools -%}\n"                                                  \
  "{{- bos_token -}}\n"                                                                                                \
  "{%- for m in ns.msgs -%}\n"                                                                                         \
  "  {%- set idx = loop.index0 -%}\n"                                                                                  \
  "  {%- if idx == 0 -%}\n"                                                                                            \
  "    {%- if thinking or m.role == 'system' -%}\n"                                                                    \
  "      {{- '<｜System｜>' -}}\n"                                                                                   \
  "    {%- endif -%}\n"                                                                                                \
  "    {%- if thinking -%}\n"                                                                                          \
  "      {{- 'Reasoning Effort: ' ~ budget ~ ' (range 1-100, the higher the value, the more thorough th"               \
  "e reasoning)\\n\\n' -}}\n"                                                                                          \
  "    {%- endif -%}\n"                                                                                                \
  "  {%- endif -%}\n"                                                                                                  \
  "  {%- if m.role == 'system' -%}\n"                                                                                  \
  "    {%- if idx > 0 -%}\n"                                                                                           \
  "      {{- '<｜System｜>' -}}\n"                                                                                   \
  "    {%- endif -%}\n"                                                                                                \
  "    {{- m.content -}}\n"                                                                                            \
  "    {%- if m.tools -%}\n"                                                                                           \
  "      {{- '\\n\\n## Tools\\n\\nYou have access to a set of tools to help answer the user\\'s questio"               \
  "n. ' -}}\n"                                                                                                         \
  "      {{- 'You can invoke tools by writing a \"<' ~ dsml ~ ' calls>\" block like the following:\\n\\n' -}}\n"       \
  "      {{- '<' ~ dsml ~ ' calls>\\n<' ~ dsml ~ ' invoke name=\"$TOOL_NAME\">\\n' -}}\n"                              \
  "      {{- '<' ~ dsml ~ ' parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</"               \
  "' ~ dsml -}}\n"                                                                                                     \
  "      {{- ' parameter>\\n...\\n</' ~ dsml ~ ' invoke>\\n<' ~ dsml ~ ' invoke name=\"$TOOL_NAME2\">"                 \
  "\\n...\\n</' ~ dsml -}}\n"                                                                                          \
  "      {{- ' invoke>\\n</' ~ dsml ~ ' calls>\\n\\n' -}}\n"                                                           \
  "      {{- 'String parameters should be specified as is and set `string=\"true\"`. For all other type"               \
  "s ' -}}\n"                                                                                                          \
  "      {{- '(numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"fal"               \
  "se\"`.\\n\\n' -}}\n"                                                                                                \
  "      {{- 'If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoni"               \
  "ng inside ' -}}\n"                                                                                                  \
  "      {{- '<think>...</think> BEFORE any tool calls or final response.\\n\\n' -}}\n"                                \
  "      {{- 'Otherwise, output directly after </think> with tool calls or final response.\\n\\n' -}}\n"               \
  "      {{- '### Available Tool Schemas\\n\\n' -}}\n"                                                                 \
  "      {%- for t in m.tools -%}\n"                                                                                   \
  "        {{- (t.function if t.function is defined else t) | tojson -}}\n"                                            \
  "        {%- if not loop.last -%}\n"                                                                                 \
  "          {{- '\\n' -}}\n"                                                                                          \
  "        {%- endif -%}\n"                                                                                            \
  "      {%- endfor -%}\n"                                                                                             \
  "      {{- '\\n\\nYou MUST strictly follow the above defined tool name and parameter schemas to invok"               \
  "e tool calls.\\n' -}}\n"                                                                                            \
  "    {%- endif -%}\n"                                                                                                \
  "    {%- if idx == 0 and response_format is defined and response_format -%}\n"                                       \
  "      {{- '\\n\\n## Response Format:\\n\\nYou MUST strictly adhere to the following schema to reply:"               \
  "\\n' -}}\n"                                                                                                         \
  "      {{- response_format | tojson -}}\n"                                                                           \
  "    {%- endif -%}\n"                                                                                                \
  "  {%- elif m.role == 'user' -%}\n"                                                                                  \
  "    {{- '<｜User｜>' ~ m.content -}}\n"                                                                           \
  "  {%- elif m.role == 'latest_reminder' -%}\n"                                                                       \
  "    {{- '<｜latest_reminder｜>' ~ m.content -}}\n"                                                                \
  "  {%- elif m.role == 'assistant' -%}\n"                                                                             \
  "    {%- if thinking and (not effective_drop or idx > st.last_user) -%}\n"                                           \
  "      {%- if m.reasoning_content is defined and m.reasoning_content is not none -%}\n"                              \
  "        {{- m.reasoning_content -}}\n"                                                                              \
  "      {%- endif -%}\n"                                                                                              \
  "      {{- '</think>' -}}\n"                                                                                         \
  "    {%- endif -%}\n"                                                                                                \
  "    {{- text_of(m.content) -}}\n"                                                                                   \
  "    {%- if m.tool_calls is defined and m.tool_calls -%}\n"                                                          \
  "      {%- set tc = namespace(lines=[]) -%}\n"                                                                       \
  "      {%- for call in m.tool_calls -%}\n"                                                                           \
  "        {%- set fn = call.function -%}\n"                                                                           \
  "        {%- set args = fn.arguments if fn.arguments is mapping else {'arguments': fn.arguments} -%}\n"              \
  "        {%- set pl = namespace(lines=[]) -%}\n"                                                                     \
  "        {%- for k, v in args.items() -%}\n"                                                                         \
  "          {%- set value = v if v is string else (v | tojson) -%}\n"                                                 \
  "          {%- set kind = 'true' if v is string else 'false' -%}\n"                                                  \
  "          {%- set pl.lines = pl.lines + ['<' ~ dsml ~ ' parameter name=\"' ~ k ~ '\" string=\"' ~ ki"               \
  "nd ~ '\">' ~ value ~ '</' ~ dsml ~ ' parameter>'] -%}\n"                                                            \
  "        {%- endfor -%}\n"                                                                                           \
  "        {%- set tc.lines = tc.lines + ['<' ~ dsml ~ ' invoke name=\"' ~ fn.name ~ '\">\\n' ~ (pl.lin"               \
  "es | join('\\n')) ~ '\\n</' ~ dsml ~ ' invoke>'] -%}\n"                                                             \
  "      {%- endfor -%}\n"                                                                                             \
  "      {{- '\\n\\n<' ~ dsml ~ ' calls>\\n' ~ (tc.lines | join('\\n')) ~ '\\n</' ~ dsml ~ ' calls>' -}}\n"            \
  "    {%- endif -%}\n"                                                                                                \
  "    {{- '<｜end▁of▁sentence｜>' -}}\n"                                                                        \
  "  {%- endif -%}\n"                                                                                                  \
  "  {%- if m.role == 'user' or (m.role == 'system' and idx > 0) -%}\n"                                                \
  "    {%- if (loop.last and add_generation_prompt) or (not loop.last and ns.msgs[idx + 1].role in ['as"               \
  "sistant', 'latest_reminder']) -%}\n"                                                                                \
  "      {%- if thinking and (not effective_drop or idx >= st.last_user) -%}\n"                                        \
  "        {{- '<｜Assistant｜><think>' -}}\n"                                                                       \
  "      {%- else -%}\n"                                                                                               \
  "        {{- '<｜Assistant｜></think>' -}}\n"                                                                      \
  "      {%- endif -%}\n"                                                                                              \
  "    {%- endif -%}\n"                                                                                                \
  "  {%- endif -%}\n"                                                                                                  \
  "{%- endfor -%}"

void fCommonChatTemplatesFree(struct common_chat_templates *pTmpls) { delete pTmpls; }

bool fCommonChatTemplatesWasExplicit(const struct common_chat_templates *pTmpls) {
  return pTmpls->vHasExplicitTemplate;
}

// LFM2 format detection: template uses <|tool_list_start|>[...]<|tool_list_end|> around the tool list
// and <|tool_call_start|>[...]<|tool_call_end|> around each tool call
static bool fIsLfm2Template(const std::string &pSrc) {
  return pSrc.find("<|tool_list_start|>") != std::string::npos && pSrc.find("<|tool_list_end|>") != std::string::npos;
}

common_chat_prompt_preset fCommonChatGetAsrPrompt(const common_chat_templates *pChatTemplates) {
  common_chat_prompt_preset vAsrPreset;
  vAsrPreset.vSystem = "";
  vAsrPreset.vUser = "Transcribe audio to text";

  if (pChatTemplates && pChatTemplates->vTemplateDefault &&
      fIsLfm2Template(pChatTemplates->vTemplateDefault->mSource())) {
    vAsrPreset.vSystem = "Perform ASR.";
    vAsrPreset.vUser = "";
  }

  return vAsrPreset;
}

std::string fCommonChatTemplatesSource(const struct common_chat_templates *pTmpls, const std::string &pVariant) {
  if (!pVariant.empty()) {
    if (pVariant == "tool_use") {
      if (pTmpls->vTemplateToolUse) {
        return pTmpls->vTemplateToolUse->mSource();
      }
      return "";
    }
    LOG_DBG("%s: unknown template variant: %s\n", __func__, pVariant.c_str());
  }
  return pTmpls->vTemplateDefault->mSource();
}

common_chat_templates_ptr fCommonChatTemplatesInit(const struct mgs_model *pModel,
                                                   const std::string &pChatTemplateOverride,
                                                   const std::string &pBosTokenOverride,
                                                   const std::string &pEosTokenOverride) {
  std::string vDefaultTemplateSrc;
  std::string vTemplateToolUseSrc;

  bool vHasExplicitTemplate = !pChatTemplateOverride.empty();
  if (pChatTemplateOverride.empty()) {
    GGML_ASSERT(pModel != nullptr);
    const auto *vStr = mgs_model_chat_template(pModel, /* name */ nullptr);
    if (vStr) {
      vDefaultTemplateSrc = vStr;
      vHasExplicitTemplate = true;
    }
    vStr = mgs_model_chat_template(pModel, /* name */ "tool_use");
    if (vStr) {
      vTemplateToolUseSrc = vStr;
      vHasExplicitTemplate = true;
    }
  } else {
    vDefaultTemplateSrc = pChatTemplateOverride;
  }
  if (vDefaultTemplateSrc.empty() && pChatTemplateOverride.empty() && pModel != nullptr) {
    char aArch[64] = {0};
    if (mgs_model_meta_val_str(pModel, "general.architecture", aArch, sizeof(aArch)) > 0 &&
        std::string(aArch) == "deepseek41") {
      vDefaultTemplateSrc = DEEPSEEK41_TEMPLATE_SRC;
      vHasExplicitTemplate = true;
    }
  }
  if (vDefaultTemplateSrc.empty() || vDefaultTemplateSrc == "chatml") {
    if (!vTemplateToolUseSrc.empty()) {
      vDefaultTemplateSrc = vTemplateToolUseSrc;
    } else {
      vDefaultTemplateSrc = CHATML_TEMPLATE_SRC;
    }
  }

  // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
  if (vDefaultTemplateSrc.find("<|channel|>") != std::string::npos
      // search for the error message and patch it
      && vDefaultTemplateSrc.find("in message.content or") != std::string::npos) {
    fStringReplaceAll(vDefaultTemplateSrc,
                      "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                      "\"<|channel|>final<|message|>\" in message.content %}",
                      "{%- if false %}");
  }

  // TODO @aldehir : this is a temporary fix, pending Minja changes
  if (vDefaultTemplateSrc.find("[TOOL_CALLS]") != std::string::npos
      // search for the error message and patch it
      && vDefaultTemplateSrc.find("if (message['content'] is none or") != std::string::npos) {
    fStringReplaceAll(vDefaultTemplateSrc,
                      "{%- if (message['content'] is none or message['content'] == '' or "
                      "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                      "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                      "{%- if false %}");
  }

  std::string vTokenBos = pBosTokenOverride;
  std::string vTokenEos = pEosTokenOverride;
  bool vAddBos = false;
  bool vAddEos = false;
  if (pModel) {
    const auto *vVocab = mgs_model_get_vocab(pModel);
    const auto cGetToken = [&](mgs_token pToken, const char *pName, const char *pJinjaVariableName) {
      if (pToken == MGS_TOKEN_NULL) {
        if (vDefaultTemplateSrc.find(pJinjaVariableName) != std::string::npos ||
            vTemplateToolUseSrc.find(pJinjaVariableName) != std::string::npos) {
          LOG_WRN("fCommonChatTemplatesInit: warning: vocab does not have a %s token, jinja template won't "
                  "work as intended.\n",
                  pName);
        }
        return std::string();
      }
      return fCommonTokenToPiece(vVocab, pToken, true);
    };
    vTokenBos = cGetToken(mgs_vocab_bos(vVocab), "BOS", "bos_token");
    vTokenEos = cGetToken(mgs_vocab_eos(vVocab), "EOS", "eos_token");
    vAddBos = mgs_vocab_get_add_bos(vVocab);
    vAddEos = mgs_vocab_get_add_eos(vVocab);
  }
  common_chat_templates_ptr vTmpls(new common_chat_templates());
  vTmpls->vHasExplicitTemplate = vHasExplicitTemplate;
  vTmpls->vAddBos = vAddBos;
  vTmpls->vAddEos = vAddEos;
  try {
    vTmpls->vTemplateDefault = std::make_unique<common_chat_template>(vDefaultTemplateSrc, vTokenBos, vTokenEos);
  } catch (const std::exception &vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    LOG_ERR("%s: failed to initialize chat template\n", __func__);
    LOG_ERR("%s: please use another chat template (--chat-template or --chat-template-file)\n", __func__);
    throw vE;
  }
  if (!vTemplateToolUseSrc.empty()) {
    try {
      vTmpls->vTemplateToolUse = std::make_unique<common_chat_template>(vTemplateToolUseSrc, vTokenBos, vTokenEos);
    } catch (const std::exception &vE) {
      LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, vE.what());
    }
  }
  return vTmpls;
}

const char *fCommonChatFormatName(common_chat_format pFormat) {
  switch (pFormat) {
  case cCommonChatFormatContentOnly:
    return "Content-only";
  case cCommonChatFormatPegSimple:
    return "peg-simple";
  case cCommonChatFormatPegNative:
    return "peg-native";
  case cCommonChatFormatPegMinimaxM3:
    return "peg-minimax-m3";
  default:
    throw std::runtime_error("Unknown chat format");
  }
}

const char *fCommonReasoningFormatName(common_reasoning_format pFormat) {
  switch (pFormat) {
  case cCommonReasoningFormatNone:
    return "none";
  case cCommonReasoningFormatAuto:
    return "auto";
  case cCommonReasoningFormatDeepseek:
    return "deepseek";
  default:
    throw std::runtime_error("Unknown reasoning format");
  }
}

common_reasoning_format fCommonReasoningFormatFromName(const std::string &pFormat) {
  if (pFormat == "none") {
    return cCommonReasoningFormatNone;
  }
  if (pFormat == "auto") {
    return cCommonReasoningFormatAuto;
  }
  if (pFormat == "deepseek") {
    return cCommonReasoningFormatDeepseek;
  }
  throw std::runtime_error("Unknown reasoning format: " + pFormat);
}

static void fForeachFunction(const json &pTools, const std::function<void(const json &)> &pFn) {
  for (const auto &vTool : pTools) {
    if (!vTool.contains("type") || vTool.at("type") != "function" || !vTool.contains("function")) {
      LOG_INF("Skipping tool without function: %s", vTool.dump(2).c_str());
      continue;
    }
    pFn(vTool);
  }
}

static void fForeachParameter(const json &pFunction,
                              const std::function<void(const std::string &, const json &, bool)> &pFn) {
  if (!pFunction.contains("parameters") || !pFunction.at("parameters").is_object()) {
    return;
  }
  const auto &vParams = pFunction.at("parameters");
  if (!vParams.contains("properties") || !vParams.at("properties").is_object()) {
    return;
  }
  const auto &vProps = vParams.at("properties");
  std::set<std::string> sRequired;
  if (vParams.contains("required") && vParams.at("required").is_array()) {
    vParams.at("required").get_to(sRequired);
  }
  for (const auto &[vName, vProp] : vProps.items()) {
    bool vIsRequired = (sRequired.find(vName) != sRequired.end());
    pFn(vName, vProp, vIsRequired);
  }
}

static std::string fCommonChatTemplateDirectApplyImpl(const common_chat_template &pTmpl,
                                                      const autoparser::vGenerationParams &pInputs,
                                                      const std::optional<json> &pMessagesOverride = std::nullopt,
                                                      const std::optional<json> &pToolsOverride = std::nullopt,
                                                      const std::optional<json> &pAdditionalContext = std::nullopt) {
  jinja::context vCtx(pTmpl.mSource());

  nlohmann::ordered_json vInp = nlohmann::ordered_json{
      {"messages", pMessagesOverride.has_value() ? *pMessagesOverride : pInputs.vMessages},
      {"bos_token", pTmpl.mBosToken()},
      {"eos_token", pTmpl.mEosToken()},
      {"enable_thinking", pInputs.vEnableThinking},
  };
  if (pToolsOverride.has_value() || !pInputs.vTools.empty()) {
    vInp["tools"] = pToolsOverride.has_value() ? *pToolsOverride : pInputs.vTools;
  }
  if (pInputs.vExtraContext.is_object()) {
    // TODO: do we need to merge, or replacing is fine?
    for (const auto &[vK, vV] : pInputs.vExtraContext.items()) {
      vInp[vK] = vV;
    }
  }
  if (pAdditionalContext.has_value()) {
    // TODO: merge properly instead of overwriting (matching old behavior)
    for (const auto &[vK, vV] : pAdditionalContext->items()) {
      vInp[vK] = vV;
    }
  }
  if (pInputs.vAddGenerationPrompt) {
    vInp["add_generation_prompt"] = true;
  }
  if (vInp.contains("preserve_reasoning") && vInp["preserve_reasoning"].is_boolean()) {
    bool vEnabled = vInp["preserve_reasoning"].get<bool>();
    jinja::fCapsApplyPreserveReasoning(vCtx, vEnabled);
  }

  jinja::fGlobalFromJson(vCtx, vInp, pInputs.vMarkInput);

  // render
  jinja::runtime vRuntime(vCtx);
  const jinja::value cResults = vRuntime.mExecute(pTmpl.vProg);
  auto vParts = jinja::runtime::mGatherStringParts(cResults);

  std::string vResult = vParts->mAsString().mStr();

  // TODO: improve this later
  if (pInputs.vAddBos && fStringStartsWith(vResult, pTmpl.mBosToken())) {
    vResult = vResult.substr(pTmpl.mBosToken().size());
  }
  if (pInputs.vAddEos && fStringEndsWith(vResult, pTmpl.mEosToken())) {
    vResult = vResult.substr(0, vResult.size() - pTmpl.mEosToken().size());
  }
  return vResult;
}

std::string fCommonChatTemplateDirectApply(const common_chat_template &pTmpl,
                                           const autoparser::vGenerationParams &pInputs) {
  return fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, std::nullopt, std::nullopt, std::nullopt);
}

static std::string
fCommonChatTemplateGenerationPromptImpl(const common_chat_template &pTmpl, const autoparser::vGenerationParams &pInputs,
                                        const std::optional<json> &pMessagesOverride = std::nullopt,
                                        const std::optional<json> &pToolsOverride = std::nullopt,
                                        const std::optional<json> &pAdditionalContext = std::nullopt) {

  auto vAdjustedMessages = pMessagesOverride ? *pMessagesOverride : pInputs.vMessages;

  autoparser::vGenerationParams vParams = pInputs;
  vParams.vAddGenerationPrompt = false;
  vParams.vContinueFinalMessage = cCommonChatContinuationNone;
  std::string vNoGenPrompt =
      fCommonChatTemplateDirectApplyImpl(pTmpl, vParams, vAdjustedMessages, pToolsOverride, pAdditionalContext);
  vParams.vAddGenerationPrompt = true;
  std::string vGenPrompt =
      fCommonChatTemplateDirectApplyImpl(pTmpl, vParams, vAdjustedMessages, pToolsOverride, pAdditionalContext);

  size_t vPrefixLen = 0;
  size_t vMinSize = std::min(vNoGenPrompt.size(), vGenPrompt.size());
  while (vPrefixLen < vMinSize && vNoGenPrompt[vPrefixLen] == vGenPrompt[vPrefixLen]) {
    vPrefixLen++;
  }
  return vGenPrompt.substr(vPrefixLen);
}

std::string fCommonChatTemplateGenerationPrompt(const common_chat_template &pTmpl,
                                                const autoparser::vGenerationParams &pInputs) {
  return fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, std::nullopt, std::nullopt, std::nullopt);
}

static common_chat_params fCommonChatParamsInitMinistral3(const common_chat_template &pTmpl,
                                                          const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  // Build up messages to follow the format:
  // https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
  auto vAdjustedMessages = json::array();
  for (const auto &vMsg : pInputs.vMessages) {
    auto vRole = vMsg.value("role", "");
    if (vRole != "system" && vRole != "assistant") {
      // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
      vAdjustedMessages.push_back(vMsg);
      continue;
    }

    auto vContent = json::array();

    // If message contains `reasoning_content`, add it as a block of type `thinking`
    if (vMsg.contains("reasoning_content") && vMsg.at("reasoning_content").is_string()) {
      vContent.push_back({
          {"type", "thinking"},
          {"thinking", vMsg.at("reasoning_content").get<std::string>()},
      });
    }

    // If message contains `content`, add it as a block of type `text`
    if (vMsg.contains("content")) {
      if (vMsg.at("content").is_string()) {
        vContent.push_back({
            {"type", "text"},
            {"text", vMsg.at("content").get<std::string>()},
        });
      } else if (vMsg.at("content").is_array()) {
        auto vBlocks = vMsg.at("content");
        vContent.insert(vContent.end(), vBlocks.begin(), vBlocks.end());
      }
    }

    auto vAdjusted = vMsg;
    vAdjusted["content"] = vContent;
    vAdjusted.erase("reasoning_content");
    vAdjustedMessages.push_back(vAdjusted);
  }

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = true;

  vData.vSupportsThinking = true;
  vData.vThinkingStartTag = "[THINK]";
  vData.lThinkingEndTags = {"[/THINK]"};
  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, /* messages_override = */ vAdjustedMessages);
  vData.vGenerationPrompt =
      fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, /* messages_override = */ vAdjustedMessages);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.lPreservedTokens = {
      "[THINK]",
      "[/THINK]",
      "[TOOL_CALLS]",
      "[ARGS]",
  };

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = "[THINK]" + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += "[/THINK]" + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mEps();
    auto vReasoning =
        vExtractReasoning ? pP.mOptional("[THINK]" + pP.mReasoning(pP.mUntil("[/THINK]")) + "[/THINK]") : pP.mEps();

    // Response format parser
    if (vHasResponseFormat) {
      // Ministral wants to emit json surrounded by code fences
      return vGenerationPrompt +
             (vReasoning << "```json" << pP.mContent(pP.mSchema(pP.mJson(), "response-format", pInputs.vJsonSchema))
                         << "```");
    }

    // Tool call parser
    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      auto vToolChoice = pP.mChoice();
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        std::string vName = vFunction.at("name");
        const auto &vSchema = vFunction.at("parameters");

        vToolChoice |=
            pP.mRule("tool-" + vName, pP.mToolOpen(pP.mToolName(pP.mLiteral(vName)) + "[ARGS]") +
                                          pP.mToolArgs(pP.mSchema(pP.mJson(), "tool-" + vName + "-schema", vSchema)));
      });

      auto vMinCalls = pInputs.vToolChoice == cCommonChatToolChoiceRequired ? 1 : 0;
      auto vMaxCalls = pInputs.vParallelToolCalls ? -1 : 1;
      auto vToolCalls = pP.mTriggerRule("tool-call", pP.mRepeat("[TOOL_CALLS]" + vToolChoice, vMinCalls, vMaxCalls));

      return vGenerationPrompt + (vReasoning << pP.mContent(pP.mUntil("[TOOL_CALLS]")) << vToolCalls);
    }

    // Content only parser
    vIncludeGrammar = false;
    return vGenerationPrompt + (vReasoning << pP.mContent(pP.mRest()));
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceAuto;

    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, "[TOOL_CALLS]"}};
  }

  return vData;
}

static common_chat_params fCommonChatParamsInitQwen3Coder(const common_chat_template &pTmpl,
                                                          const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  const std::string cGenPrefix = "<|im_start|>assistant\n";

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;

  auto vSupportsReasoning = pTmpl.mSource().find("<think>") != std::string::npos;

  vData.vSupportsThinking = vSupportsReasoning;
  vData.lPreservedTokens = {
      "<tool_call>",
      "</tool_call>",
  };

  if (vSupportsReasoning) {
    vData.vThinkingStartTag = "<think>";
    // Support both </think> and <tool_call> as reasoning end sequences.
    // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
    vData.lThinkingEndTags = {"</think>", "<tool_call>"};
    vData.lPreservedTokens.insert(vData.lPreservedTokens.end(), {"<think>", "</think>"});
  }

  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, "<|im_start|>assistant"},
      {cCommonChatRoleTool, "<|im_start|>user\n<tool_response>"}, // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
      {cCommonChatRoleTool, "<|im_start|>tool_response"},         // StepFun-3.5-Flash
      {cCommonChatRoleUser, "<|im_start|>user"},
      {cCommonChatRoleSystem, "<|im_start|>system"},
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cGenPrefix;
    if (vSupportsReasoning) {
      vData.vGenerationPrompt += "<think>\n" + vMsg.vReasoningContent;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        vData.vGenerationPrompt += "\n</think>\n\n";
      }
    }
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mLiteral(cGenPrefix);

    auto vReasoning = pP.mEps();
    if (vSupportsReasoning && vExtractReasoning) {
      vReasoning = pP.mOptional("<think>" + pP.mSpace() + pP.mReasoning(pP.mUntilOneOf({"</think>", "<tool_call>"})) +
                                (pP.mLiteral("</think>") | pP.mPeek(pP.mLiteral("<tool_call>"))));
    }

    // Response format parser
    if (vHasResponseFormat) {
      return vGenerationPrompt +
             (vReasoning << pP.mContent(pP.mSchema(pP.mJson(), "response-format", pInputs.vJsonSchema)));
    }

    // Tool call parser
    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      auto vArgClose = pP.mToolArgClose(pP.mLiteral("\n</parameter>\n"));
      auto vArgString =
          pP.mRule("xml-arg-string",
                   pP.mAc(pP.mToolArgStringValue(pP.mUntil("\n</parameter>\n")) + vArgClose, "\n</parameter>\n"));

      auto vToolChoice = pP.mChoice();
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        std::string vName = vFunction.at("name");
        auto vParameters = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();

        auto vSchemaInfo = common_schema_info();
        vSchemaInfo.mResolveRefs(vParameters);

        std::vector<common_peg_parser> lRequiredArgs;
        std::vector<common_peg_parser> lOptionalArgs;

        fForeachParameter(vFunction, [&](const std::string &pParamName, const json &pParamSchema, bool pIsRequired) {
          auto vRuleName = "tool-" + vName + "-arg-" + pParamName;

          auto vArgOpen = pP.mToolArgOpen("<parameter=" + pP.mToolArgName(pP.mLiteral(pParamName)) + ">\n");

          auto vArgValue =
              vSchemaInfo.mResolvesToString(pParamSchema)
                  ? vArgString
                  : pP.mToolArgJsonValue(pP.mSchema(pP.mJson(), vRuleName + "-schema", pParamSchema)) + vArgClose;

          auto vArgRule = pP.mRule(vRuleName, pP.mToolArg(vArgOpen + vArgValue));

          (pIsRequired ? lRequiredArgs : lOptionalArgs).push_back(vArgRule);
        });

        // Accept required arguments in any order, as Qwen does not always adhere to the
        // order provided.
        auto vArgs = pP.mPermute("tool-" + vName + "-args", lRequiredArgs);
        if (!lOptionalArgs.empty()) {
          vArgs = vArgs + pP.mZeroOrMore(pP.mChoice(lOptionalArgs));
        }

        auto vFunc = pP.mTool(pP.mToolOpen("<function=" + pP.mToolName(pP.mLiteral(vName)) + ">\n") +
                              pP.mToolArgs(vArgs) + pP.mToolClose(pP.mLiteral("</function>\n")));

        vToolChoice |= pP.mRule("tool-" + vName, vFunc);
      });

      auto vMinCalls = pInputs.vToolChoice == cCommonChatToolChoiceRequired ? 1 : 0;

      // Qwen3-Coder models may occasionally omit the <tool_call> token.
      auto vToolCallBody = vToolChoice + "</tool_call>" + pP.mSpace();
      auto vToolCallFirst = pP.mRule("tool-call-first", pP.mOptional(pP.mLiteral("<tool_call>\n")) + vToolCallBody);
      auto vToolCall = pP.mRule("tool-call", "<tool_call>\n" + vToolCallBody);

      auto vCalls = pInputs.vParallelToolCalls ? vToolCallFirst + pP.mZeroOrMore(vToolCall) : vToolCallFirst;
      auto vToolCalls = pP.mTriggerRule("tool-call-root", pP.mRepeat(vCalls, vMinCalls, 1));

      return vGenerationPrompt +
             (vReasoning << pP.mContent(pP.mUntilOneOf({"<tool_call>", "<function="})) << vToolCalls);
    }

    // Content only parser
    return vGenerationPrompt + (vReasoning << pP.mContent(pP.mRest()));
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceAuto;

    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    if (vData.vGrammarLazy) {
      vData.lGrammarTriggers = {
          {cCommonGrammarTriggerTypeWord, "<tool_call>"},
          // Trigger on "<function" and not "<function=" because the trailing "=" is part of
          // the token with the function name e.g. "=read"
          {cCommonGrammarTriggerTypeWord, "<function"},
      };
    }
  }

  return vData;
}

static common_chat_params fCommonChatParamsInitGptOss(const common_chat_template &pTmpl,
                                                      const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  // Copy reasoning to the "thinking" field as expected by the gpt-oss template
  auto vAdjustedMessages = json::array();
  for (auto vMsg : pInputs.vMessages) {
    if (vMsg.contains("reasoning_content") && vMsg.at("reasoning_content").is_string()) {
      vMsg["thinking"] = vMsg.at("reasoning_content");
      if (vMsg.contains("tool_calls") && vMsg.at("tool_calls").is_array() && !vMsg.at("tool_calls").empty()) {
        vMsg.erase("content");
      }
    }
    vAdjustedMessages.push_back(vMsg);
  }

  auto vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, /* messages_override= */ vAdjustedMessages);

  // Check if we need to replace the return token with end token during
  // inference and without generation prompt.
  if (pInputs.vIsInference && !pInputs.vAddGenerationPrompt) {
    static constexpr std::string_view cReturnToken = "<|return|>";
    static constexpr std::string_view cEndToken = "<|end|>";
    if (size_t vPos = vPrompt.rfind(cReturnToken); vPos != std::string::npos) {
      vPrompt.replace(vPos, cReturnToken.length(), cEndToken);
    }
  }

  vData.vPrompt = vPrompt;
  vData.vGenerationPrompt =
      fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, /* messages_override= */ vAdjustedMessages);
  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, "<|start|>assistant"}, {cCommonChatRoleUser, "<|start|>user"},
      {cCommonChatRoleSystem, "<|start|>developer"},    {cCommonChatRoleSystem, "<|start|>system"},
      {cCommonChatRoleTool, "<|start|>functions"},
  };

  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;

  vData.vThinkingStartTag = "<|channel|>analysis<|message|>";
  vData.lThinkingEndTags = {"<|end|>"};

  // These special tokens are required to parse properly, so we include them
  // even if parse_tool_calls is false.
  vData.lPreservedTokens = {
      "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
  };

  // Adjust prompt for continuation
  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = "<|start|>assistant<|channel|>analysis<|message|>" + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += "<|end|><|start|>assistant<|channel|>final<|message|>" + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = !pInputs.vJsonSchema.is_null() && pInputs.vJsonSchema.is_object();
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vStart = pP.mRule("start", pP.mLiteral("<|start|>assistant"));
    auto vEnd = pP.mRule("end", pP.mLiteral("<|end|>"));
    auto vContent = pP.mRule("message-content", pP.mUntil("<|end|>"));
    auto vChannel = pP.mLiteral("<|channel|>") + (pP.mLiteral("commentary") | pP.mLiteral("analysis"));
    auto vConstrainType = pP.mChars("[A-Za-z0-9_-]", 1, -1);

    // Occasionally, gpt-oss-20b will prefix channels with this commentary
    auto vStrayCommentary =
        pP.mOptional(pP.mLiteral("<|channel|>commentary") + pP.mOptional(pP.mLiteral(" to=assistant")));
    auto vStartAnalysis = vStrayCommentary + pP.mLiteral("<|channel|>analysis<|message|>");

    if (vExtractReasoning) {
      pP.mRule("analysis", vStartAnalysis + pP.mReasoning(vContent) + vEnd);
    } else {
      pP.mRule("analysis", pP.mContent(vStartAnalysis + vContent + vEnd));
    }

    auto vAnalysis = pP.mRef("analysis");
    auto vPreamble =
        pP.mRule("preamble", pP.mLiteral("<|channel|>commentary<|message|>") + pP.mContent(vContent) + vEnd);
    auto vFinalMsg =
        pP.mRule("final", vStrayCommentary + pP.mLiteral("<|channel|>final<|message|>") + pP.mContent(vContent));

    // Consume any unsolicited tool calls, e.g. builtin functions
    auto vUnsolicited =
        pP.mRule("unsolicited", pP.mAtomic(pP.mOptional(vChannel) + pP.mLiteral(" to=") + vContent + vEnd));

    auto vAny = pP.mRule("any", vPreamble | vAnalysis);

    if (vHasResponseFormat) {
      auto vConstraint = pP.mOptional(pP.mSpace() + pP.mOptional(pP.mLiteral("<|constrain|>")) + vConstrainType);
      auto vResponseFormat = pP.mRule(
          "response-format", pP.mLiteral("<|channel|>final") + vConstraint + pP.mLiteral("<|message|>") +
                                 pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)));

      return pP.mZeroOrMore(vStart + vAnalysis) + vStart + vResponseFormat;
    }

    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      auto vToolChoice = pP.mChoice();

      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        std::string vName = vFunction.at("name");
        const auto &vParams = vFunction.at("parameters");

        auto vFuncName = pP.mLiteral(" to=functions.") + pP.mToolName(pP.mLiteral(vName));
        auto vConstraint = pP.mOptional(pP.mSpace() + pP.mOptional(pP.mLiteral("<|constrain|>")) + vConstrainType);
        auto vArgs = pP.mToolArgs(pP.mSchema(pP.mJson(), "tool-" + vName + "-schema", vParams));

        // recipient in role header
        //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
        auto vToolInRole =
            pP.mTool(pP.mToolOpen(vFuncName + vChannel + vConstraint + pP.mLiteral("<|message|>")) + vArgs);

        // recipient in channel header
        //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
        auto vToolInChannel =
            pP.mTool(pP.mToolOpen(vChannel + vFuncName + vConstraint + pP.mLiteral("<|message|>")) + vArgs);

        vToolChoice |= pP.mRule("tool-" + vName, vToolInRole | vToolInChannel);
      });

      auto vToolCall = pP.mTriggerRule("tool-call", vToolChoice);

      if (pInputs.vToolChoice == cCommonChatToolChoiceRequired) {
        return pP.mZeroOrMore(vStart + vAny) + vStart + vToolCall;
      }

      return pP.mZeroOrMore(vStart + vAny) + vStart + (vToolCall | vFinalMsg);
    }

    return pP.mZeroOrMore(vStart + vAny) + vStart + (vFinalMsg | vUnsolicited);
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !(vHasResponseFormat || (vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceRequired));
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {
        {cCommonGrammarTriggerTypePattern, "^\\s+to$"},
        {cCommonGrammarTriggerTypePattern, "^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$"},
        {cCommonGrammarTriggerTypePattern, "<\\|start\\|>assistant(\\s+to)"},
        {cCommonGrammarTriggerTypePattern, "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)"}};
  }

  return vData;
}

// Functionary v3.2 - uses recipient-based format: >>>recipient\n{content}
static common_chat_params fCommonChatParamsInitFunctionaryV3Point2(const common_chat_template &pTmpl,
                                                                   const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.lPreservedTokens = {
      ">>>all",
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vIncludeGrammar = vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone;

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;
    vData.vGenerationPrompt = "<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n" + vMsg.mRenderContent();
    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    // Functionary v3.2 format:
    // - Normal content: >>>all\n{content}
    // - Tool calls: >>>function_name\n{json_args}
    // Generation prompt ends with ">>>" so model outputs recipient immediately

    // Build content parser for >>>all\n{content}
    // When tools are present, content stops before the next ">>>" (tool call)
    // When no tools, content goes until end
    auto vContentUntilTool = pP.mLiteral("all\n") + pP.mContent(pP.mUntil(">>>"));
    auto vContentUntilEnd = pP.mLiteral("all\n") + pP.mContent(pP.mRest());
    auto vGenerationPrompt = pP.mLiteral("<|start_header_id|>assistant<|end_header_id|>\n\n>>>");

    // If no tools or tool_choice is NONE, just parse content
    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      // When no tools, just match the prefix and capture everything after
      return vGenerationPrompt + vContentUntilEnd + pP.end();
    }

    // Build tool call parsers for each available function
    auto vToolChoice = pP.mChoice();
    fForeachFunction(pInputs.vTools, [&](const json &pTool) {
      const auto &vFunction = pTool.at("function");
      std::string vName = vFunction.at("name");
      const auto &vSchema = vFunction.at("parameters");

      // Tool format: >>>function_name\n{json_args}
      auto vToolParser = pP.mTool(pP.mToolOpen(pP.mToolName(pP.mLiteral(vName)) + pP.mLiteral("\n")) +
                                  pP.mToolArgs(pP.mSchema(pP.mJson(), "tool-" + vName + "-schema", vSchema)));

      vToolChoice |= pP.mRule("tool-" + vName, vToolParser);
    });

    auto vContentOnly = vContentUntilEnd;
    auto vToolsOnly = pP.mTriggerRule("tools", pP.mOneOrMore(vToolChoice));
    auto vContentAndTools = vContentUntilTool + vToolsOnly;

    auto vRet = pP.mEps();
    if (pInputs.vToolChoice == cCommonChatToolChoiceRequired) {
      if (pInputs.vParallelToolCalls) {
        vRet = pP.mChoice({vContentAndTools, vToolsOnly}) + pP.end();
      } else {
        vRet = pP.mChoice({vContentUntilTool + vToolChoice, vToolsOnly}) + pP.end();
      }
    } else if (pInputs.vParallelToolCalls) {
      vRet = pP.mChoice({vContentAndTools, vContentOnly, vToolsOnly}) + pP.end();
    } else {
      auto vContentAndTool = vContentUntilTool + vToolChoice;
      vRet = pP.mChoice({vContentAndTool, vContentOnly, vToolChoice}) + pP.end();
    }
    return vGenerationPrompt + vRet;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = pInputs.vToolChoice == cCommonChatToolChoiceAuto;

    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    // Grammar trigger for when the model starts outputting a tool call
    // (after the initial ">>>" in the generation prompt but recipient other than "all")
    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypePattern, ">>>(?!all)"}};
  }

  return vData;
}

// Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
// The ID contains both the function name and an incrementing counter
static common_chat_params fCommonChatParamsInitKimiK2(const common_chat_template &pTmpl,
                                                      const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;
  vData.lPreservedTokens = {
      "<|tool_calls_section_begin|>",
      "<|tool_calls_section_end|>",
      "<|tool_call_begin|>",
      "<|tool_call_argument_begin|>",
      "<|tool_call_end|>",
      "<think>",
      "</think>",
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone;

  const std::string cSectionBegin = "<|tool_calls_section_begin|>";
  const std::string cSectionEnd = "<|tool_calls_section_end|>";
  const std::string cCallBegin = "<|tool_call_begin|>";
  const std::string cArgsBegin = "<|tool_call_argument_begin|>";
  const std::string cCallEnd = "<|tool_call_end|>";

  const std::string cThinkStart = "<think>";
  const std::string cThinkEnd = "</think>";
  const std::string cGenPrompt = "<|im_assistant|>assistant<|im_middle|>";

  vData.vThinkingStartTag = cThinkStart;
  vData.lThinkingEndTags = {cThinkEnd};

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cGenPrompt + cThinkStart + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += cThinkEnd + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    // Kimi K2 Thinking format:
    // - Reasoning: <think>{reasoning}</think>
    // - Content: text after reasoning
    // - Tool calls section:
    //   <|tool_calls_section_begin|>
    //   <|tool_call_begin|>functions.<name>:<index><|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    //   ...
    //   <|tool_calls_section_end|>
    // The ID format is: functions.<function_name>:<counter> where counter is 0, 1, 2, ...

    // Tool call markers
    auto vEnd = pP.end();

    // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
    // For example, it can call tools at the end of reasoning without closing reasoning...
    auto vReasoning =
        vExtractReasoning
            ? pP.mOptional(
                  cThinkStart +
                  pP.mReasoning(pP.mUntilOneOf({cThinkEnd, "<|tool_calls_section_begin|>", "<|tool_call_begin|>"})) +
                  pP.mOptional(pP.mLiteral(cThinkEnd)))
            : pP.mEps();
    auto vGenerationPrompt = pP.mLiteral(cGenPrompt);

    // Content only parser (no tools)
    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      return vGenerationPrompt + vReasoning + pP.mContent(pP.mRest()) + vEnd;
    }

    // Build tool call parsers for each available function
    // The ID format is: functions.<name>:<index>
    // We need to match: functions.<name>:<digits>
    auto vToolChoice = pP.mChoice();
    fForeachFunction(pInputs.vTools, [&](const json &pTool) {
      const auto &vFunction = pTool.at("function");
      std::string vName = vFunction.at("name");
      const auto &vSchema = vFunction.at("parameters");

      // Match: functions.<name>:<digits>
      // Capture the full call id (functions.<name>:<digits>) using tool_id tag
      auto vToolId = pP.mToolId(pP.mLiteral("functions.") + pP.mToolName(pP.mLiteral(vName)) + pP.mLiteral(":") +
                                pP.mChars("[0-9]", 1, -1));
      auto vToolParser = pP.mTool(pP.mToolOpen(vToolId + pP.mLiteral(cArgsBegin)) +
                                  pP.mToolArgs(pP.mSchema(pP.mJson(), "tool-" + vName + "-schema", vSchema)) +
                                  pP.mToolClose(pP.mOptional((pP.mLiteral(cCallEnd)))));

      vToolChoice |= pP.mRule("tool-" + vName, vToolParser);
    });

    // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
    auto vMinCalls = pInputs.vToolChoice == cCommonChatToolChoiceRequired ? 1 : 0;
    auto vMaxCalls = pInputs.vParallelToolCalls ? -1 : 1;
    // Use trigger_rule so grammar generator knows where to start generating rules
    auto vToolCalls = pP.mRule(
        "tool-calls", pP.mOptional(pP.mLiteral(cSectionBegin)) +
                          pP.mTriggerRule("tool-call", pP.mRepeat(cCallBegin + vToolChoice, vMinCalls, vMaxCalls) +
                                                           pP.mOptional(pP.mLiteral(cSectionEnd))));

    auto vContentBeforeTools = pP.mContent(pP.mUntilOneOf({cSectionBegin, cCallBegin}));

    return vGenerationPrompt + vReasoning + vContentBeforeTools + vToolCalls + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = pInputs.vToolChoice == cCommonChatToolChoiceAuto;
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, "<|tool_call_begin|>"}};
  }

  return vData;
}

// Kimi K3 - XTML tagged format, built by open_tag/close_tag macros:
//   open_tag(t, attrs) = <|open|>t k="v"...<|sep|>   close_tag(t) = <|close|>t<|sep|>
//   assistant := [think] [response] [tools] close_tag(message) <|end_of_msg|>
// the generation prompt already opens the think (or response) section, so the
// section opener is optional here - same as Kimi K2 Thinking
static common_chat_params fCommonChatParamsInitKimiK3(const common_chat_template &pTmpl,
                                                      const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;

  const std::string cSep = "<|sep|>";
  const std::string cMsgStart = "<|open|>message role=\"assistant\"<|sep|>";
  const std::string cThinkStart = "<|open|>think<|sep|>";
  const std::string cThinkEnd = "<|close|>think<|sep|>";
  const std::string cRespStart = "<|open|>response<|sep|>";
  const std::string cRespEnd = "<|close|>response<|sep|>";
  const std::string cToolsStart = "<|open|>tools<|sep|>";
  const std::string cToolsEnd = "<|close|>tools<|sep|>";
  const std::string cCallStart = "<|open|>call tool=\"";
  const std::string cCallEnd = "<|close|>call<|sep|>";
  const std::string cArgStart = "<|open|>argument key=\"";
  const std::string cArgEnd = "<|close|>argument<|sep|>";
  const std::string cMsgEnd = "<|close|>message<|sep|>";
  const std::string cEomToken = "<|end_of_msg|>";

  // only the markers are special tokens. tag names ("think", "response", ...) are
  // normal tokens and must not be preserved, or prose with those words is broken
  vData.lPreservedTokens = {
      "<|open|>",
      "<|close|>",
      "<|sep|>",
      "<|end_of_msg|>",
  };

  vData.vThinkingStartTag = cThinkStart;
  vData.lThinkingEndTags = {cThinkEnd};

  // per-role message-start delimiters. user/assistant messages only have the role
  // attribute, so the full opener is used. system and tool messages have more
  // attributes, so those delimiters stop after the closing quote of the role
  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, "<|open|>message role=\"assistant\"<|sep|>"},
      {cCommonChatRoleUser, "<|open|>message role=\"user\"<|sep|>"},
      {cCommonChatRoleTool, "<|open|>message role=\"tool\""},
      {cCommonChatRoleSystem, "<|open|>message role=\"system\""},
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone;

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cMsgStart + cThinkStart + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += cThinkEnd + cRespStart + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vEnd = pP.end();

    auto vStart = pP.mOptional(pP.mLiteral(cMsgStart));

    // the think section is always consumed, even with reasoning extraction off:
    // the generation prompt ends with open_tag('think'), so it is always present.
    // reasoning stops at its own closer, or at the response opener if the model
    // skips the closer
    auto vThinkBody = vExtractReasoning ? pP.mReasoning(pP.mUntilOneOf({cThinkEnd, cRespStart}))
                                        : pP.mContent(pP.mUntilOneOf({cThinkEnd, cRespStart}));

    auto vReasoning =
        pP.mOptional(pP.mOptional(pP.mLiteral(cThinkStart)) + vThinkBody + pP.mOptional(pP.mLiteral(cThinkEnd)));

    // content runs to the response closer, or to the next section if truncated
    auto vResponse = pP.mOptional(pP.mLiteral(cRespStart)) +
                     pP.mContent(pP.mUntilOneOf({cRespEnd, cToolsStart, cMsgEnd})) +
                     pP.mOptional(pP.mLiteral(cRespEnd));

    // the EOG token after the message closer reaches the parser as text,
    // so it must be consumed or the parse stays incomplete
    auto vTrailer = pP.mOptional(pP.mLiteral(cMsgEnd)) + pP.mOptional(pP.mLiteral(cEomToken));

    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      return vStart + vReasoning + vResponse + vTrailer + vEnd;
    }

    auto vToolChoices = pP.mChoice();
    fForeachFunction(pInputs.vTools, [&](const json &pTool) {
      const auto &vFunction = pTool.at("function");
      std::string vName = vFunction.at("name");
      const json vSchema = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();

      // arguments come one tag per key, with the JSON type in a type="..."
      // attribute. the type is taken from the tool schema instead, as it tells
      // us if the value is JSON or a literal string
      auto vArgs = pP.mEps();
      if (vSchema.contains("properties") && !vSchema.at("properties").empty()) {
        auto vArgChoices = pP.mChoice();
        for (const auto &vProp : vSchema.at("properties").items()) {
          const std::string &vKey = vProp.key();

          std::string vType = "string";
          if (vProp.value().is_object() && vProp.value().contains("type") && vProp.value().at("type").is_string()) {
            vType = vProp.value().at("type").get<std::string>();
          }

          auto vValue = vType == "string" ? pP.mToolArgStringValue(pP.mUntil(cArgEnd))
                                          : pP.mToolArgValue(pP.mUntil(cArgEnd));

          // skip the trailing type="..." attribute: anything up to <|sep|>
          vArgChoices |= pP.mRule("kimi-k3-arg-" + vName + "-" + vKey,
                                  pP.mToolArg(pP.mToolArgOpen(pP.mLiteral(cArgStart)) +
                                              pP.mToolArgName(pP.mLiteral(vKey)) + pP.mLiteral("\"") +
                                              pP.mUntil(cSep) + pP.mLiteral(cSep) + vValue +
                                              pP.mToolArgClose(pP.mLiteral(cArgEnd))));
        }
        vArgs = pP.mZeroOrMore(vArgChoices);
      }

      // skip the trailing index="N" attribute the same way
      auto vCall = pP.mTool(pP.mToolOpen(pP.mLiteral(cCallStart) + pP.mToolName(pP.mLiteral(vName)) +
                                         pP.mLiteral("\"") + pP.mUntil(cSep) + pP.mLiteral(cSep)) +
                            pP.mToolArgs(vArgs) + pP.mToolClose(pP.mLiteral(cCallEnd)));

      vToolChoices |= pP.mRule("kimi-k3-tool-" + vName, vCall);
    });

    // all calls go inside one tools section, then the message is closed. the
    // message closer is part of the trigger rule, or else the lazy grammar
    // rejects it once tool calls have started
    auto vToolsSection = pP.mTriggerRule(
        "kimi-k3-tool-call", pP.mLiteral(cToolsStart) + pP.mOneOrMore(vToolChoices) + pP.mLiteral(cToolsEnd) +
                                 pP.mOptional(pP.mLiteral(cMsgEnd)) + pP.mOptional(pP.mLiteral(cEomToken)));

    auto vTools =
        pInputs.vToolChoice == cCommonChatToolChoiceRequired ? vToolsSection : pP.mOptional(vToolsSection);

    return vStart + vReasoning + vResponse + vTools + vTrailer + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = pInputs.vToolChoice != cCommonChatToolChoiceRequired;
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        if (vFunction.contains("parameters")) {
          auto vSchema = vFunction.at("parameters");
          pBuilder.vResolveRefs(vSchema);
        }
      });
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, cToolsStart}};
  }

  return vData;
}

// LFM2/LFM2.5 parser. Tool calls are almost Python-style and parallel-capable
// (except dotted names and JSON literals true/false/null).
// Always wrapped in <|tool_call_start|>[name(args)]<|tool_call_end|> with optional <think> reasoning.
// tool_list_tokens preserves LFM2 system tool-list markers.
static common_chat_params fCommonChatParamsInitLfm2(const common_chat_template &pTmpl,
                                                    const autoparser::vGenerationParams &pInputs,
                                                    bool pToolListTokens) {
  common_chat_params vData;

  const std::string cToolCallStart = "<|tool_call_start|>";
  const std::string cToolCallEnd = "<|tool_call_end|>";
  const std::string cToolListStart = "<|tool_list_start|>";
  const std::string cToolListEnd = "<|tool_list_end|>";
  const std::string cThinkStart = "<think>";
  const std::string cThinkEnd = "</think>";
  const std::string cGenPrompt = "<|im_start|>assistant\n";

  // Copy reasoning to the "thinking" field the template expects
  auto vAdjustedMessages = json::array();
  for (auto vMsg : pInputs.vMessages) {
    if (vMsg.contains("reasoning_content") && vMsg.at("reasoning_content").is_string()) {
      vMsg["thinking"] = vMsg.at("reasoning_content");
    }
    vAdjustedMessages.push_back(vMsg);
  }

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, vAdjustedMessages);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, vAdjustedMessages);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;
  vData.lPreservedTokens = {cToolCallStart, cToolCallEnd, cThinkStart, cThinkEnd};
  if (pToolListTokens) {
    vData.lPreservedTokens.push_back(cToolListStart);
    vData.lPreservedTokens.push_back(cToolListEnd);
  }

  vData.vThinkingStartTag = cThinkStart;
  vData.lThinkingEndTags = {cThinkEnd};

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = !pInputs.vJsonSchema.is_null() && pInputs.vJsonSchema.is_object();
  // Gate by reasoning format and whether the template supports <think>
  auto vExtractReasoning =
      pInputs.vReasoningFormat != cCommonReasoningFormatNone && pTmpl.mSource().find(cThinkStart) != std::string::npos;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cGenPrompt + cThinkStart + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += cThinkEnd + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mLiteral(cGenPrompt);
    auto vEnd = pP.end();

    auto vReasoning = pP.mEps();
    if (vExtractReasoning) {
      vReasoning = pP.mOptional(cThinkStart + pP.mReasoning(pP.mUntil(cThinkEnd)) + cThinkEnd);
    }

    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      if (vHasResponseFormat) {
        auto vResponseFormat = pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema));
        return vGenerationPrompt + vReasoning + vResponseFormat + vEnd;
      }
      return vGenerationPrompt + vReasoning + pP.mContent(pP.mRest()) + vEnd;
    }
    auto vToolCalls =
        pP.mRule("tool-calls",
                 pP.mTriggerRule("tool-call", pP.mLiteral(cToolCallStart) +
                                                  pP.mPythonStyleToolCalls(pInputs.vTools, pInputs.vParallelToolCalls,
                                                                           /* allow_json_literals = */ true) +
                                                  pP.mLiteral(cToolCallEnd)));

    auto vContent = pP.mContent(pP.mUntil(cToolCallStart));

    return vGenerationPrompt + vReasoning + vContent + vToolCalls + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !(vHasResponseFormat || (vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceRequired));
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, cToolCallStart}};
  }

  return vData;
}

static common_chat_params fCommonChatParamsInitGigachatV3(const common_chat_template &pTmpl,
                                                          const autoparser::vGenerationParams &pInputs) {

  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = false;
  vData.lPreservedTokens = {
      "<|message_sep|>\n\n",
      "<|role_sep|>\n",
  };

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;
    vData.vGenerationPrompt = "assistant<|role_sep|>\n" + vMsg.mRenderContent();
    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vIncludeGrammar = vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone;
  const auto *vToolCallStartPrefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vRet = pP.mEps();
    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      // Build a choice of all available tools
      auto vToolChoice = pP.mChoice();
      for (const auto &vTool : pInputs.vTools) {
        const auto &vFunction = vTool.at("function");
        std::string vName = vFunction.at("name");
        const auto &vSchema = vFunction.at("parameters");

        auto vToolName = pP.mJsonMember("name", "\"" + pP.mToolName(pP.mLiteral(vName)) + "\"");
        auto vToolArgs =
            pP.mJsonMember("arguments", pP.mToolArgs(pP.mSchema(pP.mJson(), "tool-" + vName + "-schema", vSchema)));

        auto vToolOpen = pP.mToolOpen(pP.mLiteral("{") << vToolName);

        vToolChoice |= pP.mRule("tool-" + vName, vToolOpen << "," << vToolArgs << "}");
      }

      // Define the tool call structure
      auto vMinCalls = pInputs.vToolChoice == cCommonChatToolChoiceRequired ? 1 : 0;
      auto vMaxCalls = 1; // parallel toolcalls are not supported
      auto vToolCall = pP.mRule("tool-call", pP.mLiteral(vToolCallStartPrefix) + vToolChoice);
      auto vToolCalls =
          pP.mTriggerRule("tool-call-root", pP.mRepeat(vToolCall, /* min = */ vMinCalls, /* max = */ vMaxCalls));

      vRet = pP.mContent(pP.mUntil("<|message_sep|>\n\n")) << vToolCalls;
    } else {
      // Content only parser
      vIncludeGrammar = false;
      vRet = pP.mContent(pP.mRest());
    }

    return pP.mLiteral("assistant<|role_sep|>\n") + vRet;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceAuto;

    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, vToolCallStartPrefix}};
  }
  return vData;
}

// The DeepSeek V4.1 reference implementation renders consecutive tool results into a single
// user block, ordered by the tool call order of the preceding assistant message (matched
// by tool call id) rather than by the order they appear in the conversation.
static json fDeepseekV41SortToolResults(const json &pMessages) {
  json vAdjusted = pMessages;
  std::map<std::string, size_t> dCallOrder;

  for (size_t vI = 0; vI < vAdjusted.size();) {
    const auto &vMsg = vAdjusted[vI];
    const auto cRole = vMsg.value("role", "");

    if (cRole == "assistant" && vMsg.contains("tool_calls") && vMsg.at("tool_calls").is_array() &&
        !vMsg.at("tool_calls").empty()) {
      dCallOrder.clear();
      const auto &vToolCalls = vMsg.at("tool_calls");
      for (size_t vIdx = 0; vIdx < vToolCalls.size(); vIdx++) {
        auto vId = vToolCalls[vIdx].value("id", "");
        if (!vId.empty()) {
          dCallOrder[vId] = vIdx;
        }
      }
      vI++;
      continue;
    }

    if (cRole != "user" && cRole != "tool") {
      vI++;
      continue;
    }

    // collect a maximal run of user/tool messages - they render into one user block
    std::vector<size_t> lToolPositions;
    size_t vRunEnd = vI;
    for (; vRunEnd < vAdjusted.size(); vRunEnd++) {
      const auto cR = vAdjusted[vRunEnd].value("role", "");
      if (cR == "tool") {
        lToolPositions.push_back(vRunEnd);
      } else if (cR != "user") {
        break;
      }
    }

    if (lToolPositions.size() > 1 && !dCallOrder.empty()) {
      std::vector<json> lResults;
      lResults.reserve(lToolPositions.size());
      for (auto vPos : lToolPositions) {
        lResults.push_back(vAdjusted[vPos]);
      }
      std::stable_sort(lResults.begin(), lResults.end(), [&](const json &pA, const json &pB) {
        const auto cOrder = [&](const json &pM) {
          auto vIt = dCallOrder.find(pM.value("tool_call_id", ""));
          return vIt == dCallOrder.end() ? (size_t)0 : vIt->second;
        };
        return cOrder(pA) < cOrder(pB);
      });
      for (size_t vK = 0; vK < lToolPositions.size(); vK++) {
        vAdjusted[lToolPositions[vK]] = std::move(lResults[vK]);
      }
    }

    vI = vRunEnd;
  }

  return vAdjusted;
}

static common_chat_params fCommonChatParamsInitDeepseekV3Point2(const common_chat_template &pTmpl,
                                                                const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  // V4.1 uses the same DSML markup as V3.2, but its tag names carry a leading space ("<｜DSML｜ calls>",
  // "<｜DSML｜ invoke" and "<｜DSML｜ parameter"), it renders tool results in tool call order and its
  // non-thinking generation prompt ends with a bare </think> instead of an empty <think></think> pair.
  const bool cIsV41 = pTmpl.mSource().find("function_calls") == std::string::npos;

  std::optional<json> vAdjustedMessages;
  if (cIsV41) {
    vAdjustedMessages = fDeepseekV41SortToolResults(pInputs.vMessages);
  }

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = !pInputs.vJsonSchema.is_null() && pInputs.vJsonSchema.is_object();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  std::optional<json> vAdditionalContext;
  if (cIsV41 && vHasResponseFormat) {
    vAdditionalContext = json{{"response_format", pInputs.vJsonSchema}};
  }

  const std::string cDsml = "｜DSML｜";
  const std::string cThinkStart = "<think>";
  const std::string cThinkEnd = "</think>";
  const std::string cTcBlock = cIsV41 ? " calls" : "function_calls";
  const std::string cInvokeTag = cIsV41 ? " invoke" : "invoke";
  const std::string cParamTag = cIsV41 ? " parameter" : "parameter";
  const std::string cFcStart = "<" + cDsml + cTcBlock + ">";
  const std::string cFcEnd = "</" + cDsml + cTcBlock + ">";
  const std::string cInvokeStart = "<" + cDsml + cInvokeTag;
  const std::string cInvokeEnd = "</" + cDsml + cInvokeTag + ">";
  const std::string cParamStart = "<" + cDsml + cParamTag;
  const std::string cParamEnd = "</" + cDsml + cParamTag + ">";
  const std::string cGenPrompt = "<｜Assistant｜>";
  const std::string cTcSeparator = "\n\n";

  vData.vPrompt =
      fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, vAdjustedMessages, std::nullopt, vAdditionalContext);
  vData.vGenerationPrompt =
      fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, vAdjustedMessages, std::nullopt, vAdditionalContext);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;
  vData.vThinkingStartTag = cThinkStart;
  vData.lThinkingEndTags = {cThinkEnd, cFcStart};
  vData.lPreservedTokens = {
      cDsml,
      cThinkStart,
      cThinkEnd,
  };

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    if (cIsV41 && vMsg.vReasoningContent.empty()) {
      vData.vGenerationPrompt = cGenPrompt + cThinkEnd;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        vData.vGenerationPrompt += vMsg.mRenderContent();
      }
    } else {
      vData.vGenerationPrompt = cGenPrompt + cThinkStart + vMsg.vReasoningContent;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        vData.vGenerationPrompt += cThinkEnd + vMsg.mRenderContent();
      }
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  bool vRequireTools = pInputs.vToolChoice == cCommonChatToolChoiceRequired;
  bool vHasToolCalls = vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone;

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mLiteral(cGenPrompt);
    auto vEnd = pP.end();

    // build tool call section first since we might need it in reasoning
    auto vToolChoice = pP.mChoice();
    if (vHasToolCalls) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        std::string vName = vFunction.at("name");
        auto vParams = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();
        const auto &vProps = vParams.contains("properties") ? vParams.at("properties") : json::object();

        std::set<std::string> sRequired;
        if (vParams.contains("required")) {
          vParams.at("required").get_to(sRequired);
        }

        auto vSchemaInfo = common_schema_info();
        vSchemaInfo.mResolveRefs(vParams);

        std::vector<common_peg_parser> lRequiredParsers;
        std::vector<common_peg_parser> lOptionalParsers;
        for (const auto &[vParamName, pParamSchema] : vProps.items()) {
          bool vIsRequired = sRequired.find(vParamName) != sRequired.end();
          bool vIsString = vSchemaInfo.mResolvesToString(pParamSchema);

          auto vArg = pP.mToolArg(
              pP.mToolArgOpen(pP.mLiteral(cParamStart + " name=\"") + pP.mToolArgName(pP.mLiteral(vParamName)) +
                              pP.mLiteral("\" string=\"" + std::string(vIsString ? "true" : "false") + "\">")) +
              (vIsString ? pP.mToolArgStringValue(pP.mUntil(cParamEnd))
                         : pP.mToolArgJsonValue(pP.mSchema(
                               pP.mJson(), "tool-" + vName + "-arg-" + vParamName + "-schema", pParamSchema, false))) +
              pP.mToolArgClose(pP.mLiteral(cParamEnd)));

          auto vNamedArg = pP.mRule("tool-" + vName + "-arg-" + vParamName, vArg);
          if (vIsRequired) {
            lRequiredParsers.push_back(vNamedArg);
          } else {
            lOptionalParsers.push_back(vNamedArg);
          }
        }

        common_peg_parser vArgsSeq = pP.mEps();
        for (size_t vI = 0; vI < lRequiredParsers.size(); vI++) {
          if (vI > 0) {
            vArgsSeq = vArgsSeq + pP.mSpace();
          }
          vArgsSeq = vArgsSeq + lRequiredParsers[vI];
        }

        if (!lOptionalParsers.empty()) {
          common_peg_parser vAnyOpt = pP.mChoice();
          for (const auto &vOpt : lOptionalParsers) {
            vAnyOpt |= vOpt;
          }
          vArgsSeq = vArgsSeq + pP.mRepeat(pP.mSpace() + vAnyOpt, 0, -1);
        }

        common_peg_parser vInvokeBody = vArgsSeq;
        auto vFuncParser = pP.mTool(pP.mToolOpen(pP.mLiteral(cInvokeStart + " name=\"") +
                                                 pP.mToolName(pP.mLiteral(vName)) + pP.mLiteral("\">\n")) +
                                    vInvokeBody + pP.mSpace() + pP.mToolClose(pP.mLiteral(cInvokeEnd)));

        vToolChoice |= pP.mRule("tool-" + vName, vFuncParser);
      });
    }

    common_peg_parser vToolCalls = pP.mEps();
    if (pInputs.vParallelToolCalls) {
      vToolCalls = pP.mTriggerRule("tool-call", pP.mLiteral(cFcStart) + pP.mSpace() + vToolChoice +
                                                    pP.mZeroOrMore(pP.mSpace() + vToolChoice) + pP.mSpace() +
                                                    pP.mLiteral(cFcEnd));
    } else {
      vToolCalls = pP.mTriggerRule("tool-call", pP.mLiteral(cFcStart) + pP.mSpace() + vToolChoice + pP.mSpace() +
                                                    pP.mLiteral(cFcEnd));
    }

    auto vReasoning = pP.mEps();
    auto vReasoningWithTc = pP.mEps();
    auto vObligatoryToolCalls = vToolCalls;
    bool vAllowReasoningWithTc = false;

    if (!vRequireTools) {
      vToolCalls = pP.mOptional(vToolCalls);
    }

    if (vExtractReasoning && pInputs.vEnableThinking) {
      vReasoning = pP.mOptional(cThinkStart + pP.mReasoning(pP.mUntil(cThinkEnd)) + cThinkEnd);
      vReasoningWithTc = cThinkStart + pP.mReasoning(pP.mUntilOneOf({cTcSeparator + cFcStart, cFcStart, cThinkEnd})) +
                         pP.mSpace() + vObligatoryToolCalls;
      vAllowReasoningWithTc = true;
    } else if (vExtractReasoning) {
      // Thinking disabled but reasoning extraction requested: the generation prompt
      // contains an empty <think></think> pair (V3.2) or a bare </think> (V4.1) that
      // must still be consumed.
      vReasoning = cIsV41 ? pP.mOptional(pP.mLiteral(cThinkEnd))
                         : pP.mOptional(pP.mLiteral(cThinkStart) + pP.mUntil(cThinkEnd) + pP.mLiteral(cThinkEnd));
    }

    if (vHasResponseFormat) {
      auto vResponseFormat = pP.mRule(
          "response-format", pP.mLiteral("```json") + pP.mSpace() +
                                 pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)) +
                                 pP.mSpace() + pP.mLiteral("```"));
      return vGenerationPrompt + vReasoning + vResponseFormat + vEnd;
    }

    if (!vHasToolCalls) {
      return vGenerationPrompt + vReasoning + pP.mContent(pP.mRest()) + vEnd;
    }

    auto vContentBeforeTools = pP.mNegate(pP.mLiteral(cThinkStart)) +
                               pP.mContent(pP.mUntilOneOf({cTcSeparator + cFcStart, cFcStart})) + pP.mSpace();
    return vAllowReasoningWithTc
               ? vGenerationPrompt + (vReasoningWithTc | (vReasoning + vContentBeforeTools + vToolCalls)) + vEnd
               : vGenerationPrompt + vReasoning + vContentBeforeTools + vToolCalls + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = vHasTools && !vRequireTools;
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {
        {cCommonGrammarTriggerTypeWord, cFcStart},
    };
  }

  return vData;
}

// Cohere2 MoE (a.k.a. "North Code") parser.
//
// The assistant turn is fully marker-wrapped:
//   <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
//     <|START_THINKING|>{reasoning}<|END_THINKING|>
//     then EITHER content:    <|START_TEXT|>{content}<|END_TEXT|>
//          OR     tool calls: <|START_ACTION|>[
//                                 {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ...
//                             ]<|END_ACTION|>
//   <|END_OF_TURN_TOKEN|>
//
// The generation prompt forces a leading <|START_THINKING|> (when reasoning is enabled, which is
// the template default), so the model's output continues from *inside* the thinking block. The
// parser literal therefore only covers the stable <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> prefix
// and the reasoning rule consumes the <|START_THINKING|> ... <|END_THINKING|> markers itself,
// regardless of whether they came from the generation prompt or the generated text.
static common_chat_params fCommonChatParamsInitCohere2moe(const common_chat_template &pTmpl,
                                                          const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  const std::string cTurnStart = "<|START_OF_TURN_TOKEN|>";
  const std::string cTurnEnd = "<|END_OF_TURN_TOKEN|>";
  const std::string cChatbot = "<|CHATBOT_TOKEN|>";
  const std::string cUser = "<|USER_TOKEN|>";
  const std::string cSystem = "<|SYSTEM_TOKEN|>";
  const std::string cThinkStart = "<|START_THINKING|>";
  const std::string cThinkEnd = "<|END_THINKING|>";
  const std::string cTextStart = "<|START_TEXT|>";
  const std::string cTextEnd = "<|END_TEXT|>";
  const std::string cActionStart = "<|START_ACTION|>";
  const std::string cActionEnd = "<|END_ACTION|>";
  const std::string cResultStart = "<|START_TOOL_RESULT|>";
  const std::string cResultEnd = "<|END_TOOL_RESULT|>";

  // Stable prefix of the generation prompt that precedes the (forced) <|START_THINKING|> marker.
  const std::string cGenPrefix = cTurnStart + cChatbot;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;
  vData.vThinkingStartTag = cThinkStart;
  vData.lThinkingEndTags = {cThinkEnd};
  vData.lPreservedTokens = {
      cTurnStart, cTurnEnd, cChatbot,     cUser,      cSystem,      cThinkStart, cThinkEnd,
      cTextStart, cTextEnd, cActionStart, cActionEnd, cResultStart, cResultEnd,
  };

  // Declare per-role message delimiters. Tool results are rendered with the
  // system token followed by <|START_TOOL_RESULT|>, so the "tool" delimiter must be listed before
  // the plain "system" one (it is a strict superset, and the role split tries delimiters in order).
  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, cGenPrefix},
      {cCommonChatRoleUser, cTurnStart + cUser},
      {cCommonChatRoleTool, cTurnStart + cSystem + cResultStart},
      {cCommonChatRoleSystem, cTurnStart + cSystem},
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cGenPrefix + cThinkStart + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += cThinkEnd + cTextStart + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mLiteral(cGenPrefix);
    auto vEnd = pP.end();

    // The thinking block is always present (the generation prompt forces <|START_THINKING|>).
    // When extracting reasoning, capture its body; otherwise keep the whole block (markers
    // included) inline as content, matching reasoning_format=NONE conventions.
    common_peg_parser vReasoning = pP.mEps();
    if (vExtractReasoning) {
      vReasoning =
          pP.mOptional(pP.mLiteral(cThinkStart) + pP.mReasoning(pP.mUntilOneOf({cThinkEnd, cTextStart, cActionStart})) +
                       pP.mOptional(pP.mLiteral(cThinkEnd)));
    } else {
      vReasoning =
          pP.mOptional(pP.mContent(pP.mLiteral(cThinkStart) + pP.mUntilOneOf({cThinkEnd, cTextStart, cActionStart}) +
                                   pP.mOptional(pP.mLiteral(cThinkEnd))));
    }

    auto vTextContent =
        vHasResponseFormat
            ? pP.mLiteral(cTextStart) +
                  pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)) +
                  pP.mOptional(pP.mLiteral(cTextEnd))
            : pP.mLiteral(cTextStart) + pP.mContent(pP.mUntil(cTextEnd)) + pP.mOptional(pP.mLiteral(cTextEnd));

    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      return vGenerationPrompt + vReasoning + vTextContent + pP.mOptional(pP.mLiteral(cTurnEnd)) + vEnd;
    }

    auto vRequireTools = pInputs.vToolChoice == cCommonChatToolChoiceRequired;

    // <|START_ACTION|>[ {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ... ]<|END_ACTION|>
    auto vToolCalls = pP.mStandardJsonTools(cActionStart, cActionEnd, pInputs.vTools, pInputs.vParallelToolCalls,
                                            /* force_tool_calls = */ true,
                                            /* name_key         = */ "tool_name",
                                            /* args_key         = */ "parameters",
                                            /* array_wrapped    = */ true,
                                            /* function_is_key  = */ false,
                                            /* call_id_key      = */ "",
                                            /* gen_call_id_key  = */ "tool_call_id",
                                            /* parameters_order = */ {"tool_call_id", "tool_name", "parameters"});

    // Content and tool calls are mutually exclusive in this format.
    common_peg_parser vBody = vRequireTools ? vToolCalls : pP.mChoice({vToolCalls, vTextContent});

    return vGenerationPrompt + vReasoning + vBody + pP.mOptional(pP.mLiteral(cTurnEnd)) + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !vHasResponseFormat && pInputs.vToolChoice == cCommonChatToolChoiceAuto;
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.at("parameters");
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, cActionStart}};
  }

  return vData;
}

static common_chat_params fCommonChatParamsInitMinimaxM3(const common_chat_template &pTmpl,
                                                         const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegMinimaxM3;
  vData.vSupportsThinking = true;
  vData.vThinkingStartTag = "<mm:think>";
  vData.lThinkingEndTags = {"</mm:think>"};

  // M3 prefixes every tool tag with the namespace token "]<]minimax[>[";
  // params use the parameter name as the tag (<file_path>...</file_path>).
  const std::string cNs = "]<]minimax[>[";
  const std::string cThinkStart = "<mm:think>";
  const std::string cThinkEnd = "</mm:think>";
  const std::string cFcStart = cNs + "<tool_call>";
  const std::string cFcEnd = cNs + "</tool_call>";
  const std::string cInvokeEnd = cNs + "</invoke>";

  vData.lPreservedTokens = {
      cNs, "<tool_call>", "</tool_call>", cThinkStart, cThinkEnd,
  };

  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, "]~b]ai"},  {cCommonChatRoleUser, "]~b]user"},
      {cCommonChatRoleTool, "]~b]tool"},     {cCommonChatRoleSystem, "]~b]developer"},
      {cCommonChatRoleSystem, "]~b]system"},
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = !pInputs.vJsonSchema.is_null() && pInputs.vJsonSchema.is_object();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  const std::string cGenPrompt = vData.vGenerationPrompt;

  using mm3 = common_chat_peg_minimax_m3_mapper;

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = cGenPrompt + cThinkStart + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += cThinkEnd + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mPrefix(cGenPrompt, cThinkStart);
    auto vEnd = pP.end();

    auto vReasoning = pP.mEps();
    if (vExtractReasoning) {
      auto vBlock = pInputs.vEnableThinking
                        ? pP.mLiteral(cThinkStart) + pP.mSpace() +
                              pP.mAc(pP.mReasoning(pP.mUntil(cThinkEnd)) + pP.mLiteral(cThinkEnd), cThinkEnd)
                        : pP.mLiteral(cThinkStart) + pP.mAc(pP.mUntil(cThinkEnd) + pP.mLiteral(cThinkEnd), cThinkEnd);

      // A turn without reasoning is prefixed with a bare </mm:think>, written either by the
      // generation prompt (thinking_mode = "disabled") or by the model itself.
      vReasoning = pP.mOptional(pP.mChoice({vBlock, pP.mLiteral(cThinkEnd)}));
    }

    if (vHasResponseFormat) {
      auto vResponseFormat = pP.mRule(
          "response-format", pP.mLiteral("```json") + pP.mSpace() +
                                 pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)) +
                                 pP.mSpace() + pP.mLiteral("```"));
      return vGenerationPrompt + vReasoning + vResponseFormat + vEnd;
    }

    if (!vHasTools || pInputs.vToolChoice == cCommonChatToolChoiceNone) {
      return vGenerationPrompt + vReasoning + pP.mContent(pP.mRest()) + vEnd;
    }

    auto vAlternativesOf = [](const json &pSchema) -> std::optional<json> {
      for (const auto *vKeyword : {"oneOf", "anyOf"}) {
        if (pSchema.contains(vKeyword) && pSchema.at(vKeyword).is_array() && !pSchema.at(vKeyword).empty()) {
          return pSchema.at(vKeyword);
        }
      }
      return std::nullopt;
    };

    auto vToolChoice = pP.mChoice();
    fForeachFunction(pInputs.vTools, [&](const json &pTool) {
      const auto &vFunction = pTool.at("function");
      std::string vName = vFunction.at("name");
      auto vParams = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();

      auto vSchemaInfo = common_schema_info();
      vSchemaInfo.mResolveRefs(vParams);

      // The template expands argument values recursively in XML (see the to_xml() macro)
      std::function<common_peg_parser(const json &, const std::string &, const std::string &)> vValueOf;
      std::function<common_peg_parser(const json &, const std::string &)> vMembersOf;

      auto vElementOf = [&](const std::string &pTag, const json &pSchema, const std::string &pRuleName) {
        const std::string cClose = cNs + "</" + pTag + ">";
        return pP.mRule(pRuleName, pP.mToolArg(pP.mToolArgOpen(pP.mLiteral(cNs + "<") +
                                                               pP.mToolArgName(pP.mLiteral(pTag)) + pP.mLiteral(">")) +
                                               vValueOf(pSchema, pRuleName, cClose)));
      };

      vValueOf = [&](const json &pSchema, const std::string &pRuleName,
                     const std::string &pClose) -> common_peg_parser {
        auto vCloseTag = pP.mToolArgClose(pP.mLiteral(pClose));

        // A string accepts anything, so a union with a string alternative is a string
        if (vSchemaInfo.mResolvesToString(pSchema)) {
          return pP.mAc(pP.mToolArgStringValue(pP.mUntil(pClose)) + vCloseTag, pClose);
        }

        if (auto vAlternatives = vAlternativesOf(pSchema)) {
          std::vector<common_peg_parser> lChoices;

          size_t vIndex = 0;
          for (const auto &vAlternative : *vAlternatives) {
            const std::string cAltName = pRuleName + "-" + std::to_string(vIndex++);

            // There is a risk that this breaks streaming deltas, but that's a risk we
            // assume to provide tool arg streaming.
            lChoices.push_back(vValueOf(vAlternative, cAltName, pClose));
          }

          return pP.mChoice(lChoices);
        }

        const std::string cType =
            pSchema.contains("type") && pSchema.at("type").is_string() ? pSchema.at("type").get<std::string>() : "";

        if (cType == "object" && pSchema.contains("properties")) {
          return pP.mTag(mm3::cToolArgObject, vMembersOf(pSchema, pRuleName)) + pP.mSpace() + vCloseTag;
        }

        if (cType == "array" && pSchema.contains("items")) {
          const std::string cItemClose = cNs + "</item>";
          auto vItem =
              pP.mRule(pRuleName + "-item",
                       pP.mTag(mm3::cToolArgItem, pP.mLiteral(cNs + "<item>") +
                                                      vValueOf(pSchema.at("items"), pRuleName + "-item", cItemClose)));
          return pP.mTag(mm3::cToolArgArray, pP.mRepeat(pP.mSpace() + vItem, 0, -1)) + pP.mSpace() + vCloseTag;
        }

        return pP.mToolArgJsonValue(pP.mSchema(pP.mJson(), pRuleName + "-schema", pSchema, false)) + vCloseTag;
      };

      // Required properties in schema order, then any number of optional ones in any order.
      vMembersOf = [&](const json &pSchema, const std::string &pRulePrefix) -> common_peg_parser {
        const auto &vProps = pSchema.at("properties");

        std::set<std::string> sRequired;
        if (pSchema.contains("required")) {
          pSchema.at("required").get_to(sRequired);
        }

        std::vector<common_peg_parser> lRequiredElements;
        std::vector<common_peg_parser> lOptionalElements;
        for (const auto &[vKey, vKeySchema] : vProps.items()) {
          auto vElement = vElementOf(vKey, vKeySchema, pRulePrefix + "-" + vKey);
          if (sRequired.find(vKey) != sRequired.end()) {
            lRequiredElements.push_back(vElement);
          } else {
            lOptionalElements.push_back(vElement);
          }
        }

        common_peg_parser vMembers = pP.mEps();
        for (size_t vI = 0; vI < lRequiredElements.size(); vI++) {
          if (vI > 0) {
            vMembers = vMembers + pP.mSpace();
          }
          vMembers = vMembers + lRequiredElements[vI];
        }

        if (!lOptionalElements.empty()) {
          common_peg_parser vAnyOptional = pP.mChoice();
          for (const auto &vElement : lOptionalElements) {
            vAnyOptional |= vElement;
          }
          vMembers = vMembers + pP.mRepeat(pP.mSpace() + vAnyOptional, 0, -1);
        }

        return vMembers;
      };

      common_peg_parser vInvokeBody =
          vParams.contains("properties") ? vMembersOf(vParams, "tool-" + vName + "-arg") : pP.mEps();

      auto vFuncParser = pP.mTool(
          pP.mToolOpen(pP.mLiteral(cNs + "<invoke name=\"") + pP.mToolName(pP.mLiteral(vName)) + pP.mLiteral("\">")) +
          pP.mSpace() + vInvokeBody + pP.mSpace() + pP.mToolClose(pP.mLiteral(cInvokeEnd)));

      vToolChoice |= pP.mRule("tool-" + vName, vFuncParser);
    });

    auto vRequireTools = pInputs.vToolChoice == cCommonChatToolChoiceRequired;

    common_peg_parser vToolCalls = pP.mEps();
    if (pInputs.vParallelToolCalls) {
      vToolCalls = pP.mTriggerRule("tool-call", pP.mLiteral(cFcStart) + pP.mSpace() + vToolChoice +
                                                    pP.mZeroOrMore(pP.mSpace() + vToolChoice) + pP.mSpace() +
                                                    pP.mLiteral(cFcEnd));
    } else {
      vToolCalls = pP.mTriggerRule("tool-call", pP.mLiteral(cFcStart) + pP.mSpace() + vToolChoice + pP.mSpace() +
                                                    pP.mLiteral(cFcEnd));
    }

    if (!vRequireTools) {
      vToolCalls = pP.mOptional(vToolCalls);
    }

    auto vContentBeforeTools = pP.mContent(pP.mUntil(cFcStart));
    return vGenerationPrompt + vReasoning + vContentBeforeTools + vToolCalls + vEnd;
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !(vHasResponseFormat || (vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceRequired));
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {
        {cCommonGrammarTriggerTypeWord, cFcStart},
    };
  }

  return vData;
}

namespace workaround {

static void fMapDeveloperRoleToSystem(json &pMessages) {
  for (auto &vMessage : pMessages) {
    if (vMessage.contains("role")) {
      if (vMessage["role"] == "developer") {
        vMessage["role"] = "system";
      }
    }
  }
}

// if first message is system and template does not support it, merge it with next message
static void fSystemMessageNotSupported(json &pMessages) {
  if (!pMessages.empty() && pMessages.front().at("role") == "system") {
    if (pMessages.size() > 1) {
      LOG_DBG("Merging system prompt into next message\n");
      auto &vFirstMsg = pMessages.front();
      auto &vSecondMsg = pMessages[1];
      vSecondMsg["content"] =
          vFirstMsg.at("content").get<std::string>() + "\n" + vSecondMsg.at("content").get<std::string>();
      pMessages.erase(pMessages.begin());
    } else {
      LOG_WRN("Removing system prompt due to template not supporting system role\n");
      pMessages.erase(pMessages.begin());
    }
  }
}

static void fRequiresNonNullContent(json &pMessages) {
  GGML_ASSERT(pMessages.is_array());
  for (auto &vMessage : pMessages) {
    if (vMessage.contains("tool_calls") && !vMessage.contains("content")) {
      vMessage["content"] = "";
    }
  }
}

static void fFuncArgsNotString(json &pMessages) {
  GGML_ASSERT(pMessages.is_array());
  for (auto &vMessage : pMessages) {
    if (vMessage.contains("tool_calls")) {
      for (auto &vToolCall : vMessage["tool_calls"]) {
        if (vToolCall.contains("function") && vToolCall["function"].contains("arguments")) {
          auto &vArgs = vToolCall["function"]["arguments"];
          if (vArgs.is_string()) {
            try {
              vArgs = json::parse(vArgs.get<std::string>());
            } catch (const std::exception &vE) {
              throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(vE.what()));
            }
          }
        }
      }
    }
  }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void fTrimAllContent(std::vector<common_chat_msg> &pMessages) {
  for (auto &vMessage : pMessages) {
    vMessage.vContent = fTrimWhitespace(vMessage.vContent);
    vMessage.vReasoningContent = fTrimWhitespace(vMessage.vReasoningContent);
    for (auto &vPart : vMessage.lContentParts) {
      if (vPart.vType == "text") {
        vPart.vText = fTrimWhitespace(vPart.vText);
      }
    }
  }
}

} // namespace workaround

// MiniCPM5 format:
// - Reasoning: <think>{reasoning}</think> (optional)
// - Tool calls: <function name="foo"><param name="bar">value</param></function>
static common_chat_params fCommonChatParamsInitMinicpm5(const common_chat_template &pTmpl,
                                                        const autoparser::vGenerationParams &pInputs) {
  common_chat_params vData;

  vData.vPrompt = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.vSupportsThinking = true;
  vData.lPreservedTokens = {
      "<function", "<param", "</function>", "</param>", "<think>", "</think>",
  };

  vData.vThinkingStartTag = "<think>";
  vData.lThinkingEndTags = {"</think>"};

  vData.vMessageDelimiters = {
      {cCommonChatRoleAssistant, "<|im_start|>assistant"},
      {cCommonChatRoleTool, "<|im_start|>user\n<tool_response>"},
      {cCommonChatRoleUser, "<|im_start|>user"},
      {cCommonChatRoleSystem, "<|im_start|>system"},
  };

  auto vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
  auto vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  if (pInputs.mHasContinuation()) {
    const auto &vMsg = pInputs.vContinueMsg;

    vData.vGenerationPrompt = "<|im_start|>assistant\n<think>\n" + vMsg.vReasoningContent;
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += "\n</think>\n\n" + vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    auto vGenerationPrompt = pP.mLiteral("<|im_start|>assistant\n");

    auto vReasoning = pP.mEps();
    if (vExtractReasoning) {
      vReasoning = ("<think>" << pP.mReasoning(pP.mUntil("</think>")) << "</think>") + pP.mSpace();
    }

    // Response format parser
    if (vHasResponseFormat) {
      return vGenerationPrompt + vReasoning +
             pP.mContent(pP.mSchema(pP.mJson(), "response-format", pInputs.vJsonSchema));
    }

    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      // CDATA lets a value carry characters that would otherwise close the tag (e.g.
      // </param>); capture the inner text only, excluding the CDATA markers.
      auto vStringValue = pP.mChoice(
          {pP.mLiteral("<![CDATA[") + pP.mAc(pP.mToolArgStringValue(pP.mUntil("]]>")) + pP.mLiteral("]]>"), "]]>") +
               pP.mToolArgClose(pP.mLiteral("</param>")),
           pP.mNegate(pP.mLiteral("<![CDATA[")) +
               pP.mAc(pP.mToolArgStringValue(pP.mUntil("</param>")) + pP.mToolArgClose(pP.mLiteral("</param>")),
                      "</param>")});

      auto vToolChoice = pP.mChoice();
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        const std::string cName = vFunction.at("name");
        auto vParams = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();

        auto vArgs = pP.mEps();
        if (vParams.contains("properties") && vParams.at("properties").is_object() &&
            !vParams.at("properties").empty()) {
          auto vSchemaInfo = common_schema_info();
          vSchemaInfo.mResolveRefs(vParams);

          auto vArgChoice = pP.mChoice();
          for (const auto &[vPropName, vPropSchema] : vParams.at("properties").items()) {
            auto vValueParser = pP.mEps();
            if (vSchemaInfo.mResolvesToString(vPropSchema)) {
              vValueParser = vStringValue;
            } else {
              vValueParser = pP.mToolArgJsonValue(pP.mSchema(
                                 pP.mJson(), "tool-" + cName + "-arg-" + vPropName + "-schema", vPropSchema, false)) +
                             pP.mToolArgClose(pP.mLiteral("</param>"));
            }

            auto vArgRule = pP.mToolArg(pP.mToolArgOpen(pP.mLiteral("<param name=\"") +
                                                        pP.mToolArgName(pP.mLiteral(vPropName)) + pP.mLiteral("\">")) +
                                        vValueParser);

            vArgChoice |= vArgRule;
          }
          vArgs = pP.mZeroOrMore(vArgChoice + pP.mSpace());
        }

        auto vToolParser = pP.mTool(
            pP.mToolOpen(pP.mLiteral("<function name=\"") + pP.mToolName(pP.mLiteral(cName)) + pP.mLiteral("\">"))
            << pP.mToolArgs(vArgs) << pP.mToolClose(pP.mLiteral("</function>")));

        vToolChoice |= pP.mRule("tool-" + cName, vToolParser);
      });

      auto vMaxCalls = pInputs.vParallelToolCalls ? -1 : 1;
      auto vToolCalls = pP.mTriggerRule("tool-call", pP.mRepeat(vToolChoice + pP.mSpace(), 1, vMaxCalls));

      auto vContent = pP.mContent(pP.mUntil("<function"));

      return vGenerationPrompt + vReasoning + vContent + vToolCalls + pP.end();
    }

    return vGenerationPrompt + vReasoning + pP.mContent(pP.mRest()) + pP.end();
  });

  vData.vParser = vParser.mSave();

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !(vHasResponseFormat || (vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceRequired));
    vData.vGrammar = fBuildGrammar([&](const common_grammar_builder &pBuilder) {
      fForeachFunction(pInputs.vTools, [&](const json &pTool) {
        const auto &vFunction = pTool.at("function");
        auto vSchema = vFunction.contains("parameters") ? vFunction.at("parameters") : json::object();
        pBuilder.vResolveRefs(vSchema);
      });
      if (vHasResponseFormat) {
        auto vSchema = pInputs.vJsonSchema;
        pBuilder.vResolveRefs(vSchema);
      }
      vParser.mBuildGrammar(pBuilder, vData.vGrammarLazy);
    });

    vData.lGrammarTriggers = {
        {cCommonGrammarTriggerTypeWord, "<function"},
    };
  }

  return vData;
}

static json fCommonChatExtraContext() {
  json vCtx = json::object();
  std::chrono::system_clock::time_point vNow = std::chrono::system_clock::now();
  std::string vDatetimeStr = fFormatTime(vNow, "%b %d %Y");
  std::string vDateStr = fFormatTime(vNow, "%d %b %Y");
  vCtx["datetime"] = vDatetimeStr;
  vCtx["date_string"] = vDateStr;
  return vCtx;
}

std::optional<common_chat_params> fCommonChatTrySpecializedTemplate(const common_chat_template &pTmpl,
                                                                    const std::string &pSrc,
                                                                    autoparser::vGenerationParams &pParams) {
  // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
  // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
  if (pSrc.find("[SYSTEM_PROMPT]") != std::string::npos && pSrc.find("[TOOL_CALLS]") != std::string::npos &&
      pSrc.find("[ARGS]") != std::string::npos && pSrc.find("[CALL_ID]") == std::string::npos) {
    LOG_DBG("Using specialized template: Ministral/Magistral Large 3\n");
    return fCommonChatParamsInitMinistral3(pTmpl, pParams);
  }

  // GPT-OSS - has unique channel-based structure that needs dedicated handler
  if (pSrc.find("<|channel|>") != std::string::npos) {
    LOG_DBG("Using specialized template: GPT-OSS\n");
    return fCommonChatParamsInitGptOss(pTmpl, pParams);
  }

  // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
  // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
  if (pSrc.find(">>>all") != std::string::npos && pSrc.find(">>>${recipient}") != std::string::npos) {
    LOG_DBG("Using specialized template: Functionary v3.2\n");
    return fCommonChatParamsInitFunctionaryV3Point2(pTmpl, pParams);
  }

  // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
  // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
  if (pSrc.find("<|tool_calls_section_begin|>") != std::string::npos &&
      pSrc.find("<|tool_call_begin|>") != std::string::npos) {
    LOG_DBG("Using specialized template: Kimi K2 Thinking\n");
    return fCommonChatParamsInitKimiK2(pTmpl, pParams);
  }

  // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
  if (pSrc.find("<|open|>") != std::string::npos && pSrc.find("<|close|>") != std::string::npos &&
      pSrc.find("<|end_of_msg|>") != std::string::npos) {
    LOG_DBG("Using specialized template: Kimi K3\n");
    return fCommonChatParamsInitKimiK3(pTmpl, pParams);
  }

  // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
  // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
  // Command-R templates use <|START_RESPONSE|>).
  if (pSrc.find("<|START_TEXT|>") != std::string::npos && pSrc.find("<|START_ACTION|>") != std::string::npos) {
    LOG_DBG("Using specialized template: Cohere2 MoE\n");
    return fCommonChatParamsInitCohere2moe(pTmpl, pParams);
  }

  if (fIsLfm2Template(pSrc)) {
    LOG_DBG("Using specialized template: LFM2\n");
    return fCommonChatParamsInitLfm2(pTmpl, pParams, /* tool_list_tokens = */ true);
  }

  // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
  if (pSrc.find("List of tools: [") != std::string::npos && pSrc.find("<|tool_list_start|>") == std::string::npos) {
    LOG_DBG("Using specialized template: LFM2.5\n");
    return fCommonChatParamsInitLfm2(pTmpl, pParams, /* tool_list_tokens = */ false);
  }

  // GigaChatV3 format detection
  if (pSrc.find("<|role_sep|>") != std::string::npos && pSrc.find("<|message_sep|>") != std::string::npos &&
      pSrc.find("<|function_call|>") == std::string::npos) {
    LOG_DBG("Using specialized template: GigaChatV3\n");
    return fCommonChatParamsInitGigachatV3(pTmpl, pParams);
  }

  // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
  // markup delimiters, so detect the template and use a dedicated parser.
  if (pSrc.find("]<]minimax[>[") != std::string::npos && pSrc.find("<tool_call>") != std::string::npos &&
      pSrc.find("<invoke name=") != std::string::npos) {
    LOG_DBG("Using specialized template: MiniMax-M3\n");
    return fCommonChatParamsInitMinimaxM3(pTmpl, pParams);
  }

  // DeepSeek V4.1 format detection: DSML tool calls whose tag names carry a leading space ("<｜DSML｜ calls>").
  if (pSrc.find("DSML") != std::string::npos && pSrc.find(" calls>") != std::string::npos &&
      pSrc.find("function_calls") == std::string::npos) {
    LOG_DBG("Using specialized template: DeepSeek V4.1\n");
    return fCommonChatParamsInitDeepseekV3Point2(pTmpl, pParams);
  }

  // DeepSeek V3.2 format detection: template defines dsml_token and uses it for tool calls.
  // The template source contains the token as a variable assignment, not as a literal in markup.
  // V3.2 names the tool call block "function_calls".
  if (pSrc.find("dsml_token") != std::string::npos && pSrc.find("DSML") != std::string::npos &&
      pSrc.find("function_calls") != std::string::npos) {
    LOG_DBG("Using specialized template: DeepSeek V3.2\n");
    return fCommonChatParamsInitDeepseekV3Point2(pTmpl, pParams);
  }

  // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
  if (pSrc.find("Tool usage guidelines:") != std::string::npos && pSrc.find("<function name=\"") != std::string::npos &&
      pSrc.find("<param name=\"") != std::string::npos) {
    LOG_DBG("Using specialized template: MiniCPM5\n");
    return fCommonChatParamsInitMinicpm5(pTmpl, pParams);
  }

  // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
  if (pSrc.find("<tool_call>") != std::string::npos && pSrc.find("<function=") != std::string::npos &&
      pSrc.find("<parameter=") != std::string::npos) {
    LOG_DBG("Using specialized template: Qwen3-Coder\n");
    return fCommonChatParamsInitQwen3Coder(pTmpl, pParams);
  }

  return std::nullopt;
}

static common_chat_params fCommonChatTemplatesApplyJinja(const struct common_chat_templates *pTmpls,
                                                         const struct common_chat_templates_inputs &pInputs) {
  autoparser::vGenerationParams vParams;
  vParams.vTools = fCommonChatToolsToJsonOaicompat(pInputs.lTools);
  const auto &vTmpl =
      vParams.vTools.is_array() && pTmpls->vTemplateToolUse ? *pTmpls->vTemplateToolUse : *pTmpls->vTemplateDefault;
  const auto &vSrc = vTmpl.mSource();
  const auto &vCaps = vTmpl.mOriginalCaps();
  std::vector<common_chat_msg> lTrimmedMessages;
  const std::vector<common_chat_msg> *lMessagesToRender = &pInputs.lMessages;
  if (vSrc.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
    // StepFun: trim message contents (including typed content parts) before rendering,
    // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
    lTrimmedMessages = pInputs.lMessages;
    workaround::fTrimAllContent(lTrimmedMessages);
    lMessagesToRender = &lTrimmedMessages;
  }
  vParams.vMessages = fRenderMessageToJson(*lMessagesToRender, vTmpl.mOriginalCaps());
  vParams.vToolChoice = pInputs.vToolChoice;
  vParams.vReasoningFormat = pInputs.vReasoningFormat;
  vParams.vEnableThinking = pInputs.vEnableThinking;
  vParams.vGrammar = pInputs.vGrammar;
  vParams.vNow = pInputs.vNow;
  vParams.vAddGenerationPrompt = pInputs.vAddGenerationPrompt;
  vParams.vAddBos = pTmpls->vAddBos;
  vParams.vAddEos = pTmpls->vAddEos;

  vParams.vContinueFinalMessage = pInputs.vContinueFinalMessage;
  if (vParams.vContinueFinalMessage != cCommonChatContinuationNone) {
    vParams.vAddGenerationPrompt = false;

    if (!pInputs.lMessages.empty()) {
      // Render messages[:-1] and store continuation message separately
      vParams.vContinueMsg = pInputs.lMessages.back();
      vParams.vMessages.erase(vParams.vMessages.size() - 1);
    }

    if (vParams.vContinueFinalMessage == cCommonChatContinuationAuto && !pInputs.lMessages.empty()) {
      // Resolve based on message content
      vParams.vContinueFinalMessage = cCommonChatContinuationContent;
      if (!vParams.vContinueMsg.vReasoningContent.empty() && vParams.vContinueMsg.vContent.empty() &&
          vParams.vContinueMsg.lContentParts.empty()) {
        vParams.vContinueFinalMessage = cCommonChatContinuationReasoning;
      }
    }
  }

  if (vSrc.find("<|channel|>") == std::string::npos) {
    // map developer to system for all models except for GPT-OSS
    workaround::fMapDeveloperRoleToSystem(vParams.vMessages);
  }

  if (!vTmpl.mOriginalCaps().vSupportsSystemRole) {
    workaround::fSystemMessageNotSupported(vParams.vMessages);
  }

  if (vTmpl.mOriginalCaps().vSupportsToolCalls) {
    // some templates will require the content field in tool call messages
    // to still be non-null, this puts an empty string everywhere where the
    // content field is null
    workaround::fRequiresNonNullContent(vParams.vMessages);
  }

  if (vTmpl.mOriginalCaps().vSupportsObjectArguments) {
    workaround::fFuncArgsNotString(vParams.vMessages);
  }

  vParams.vExtraContext = fCommonChatExtraContext();
  for (auto vEl : pInputs.dChatTemplateKwargs) {
    vParams.vExtraContext[vEl.first] = json::parse(vEl.second);
  }

  if (!pInputs.vJsonSchema.empty()) {
    vParams.vJsonSchema = json::parse(pInputs.vJsonSchema);
  }

  vParams.vParallelToolCalls = pInputs.vParallelToolCalls;

  if (vParams.vTools.is_array()) {
    if (vParams.vToolChoice != cCommonChatToolChoiceNone && !vParams.vGrammar.empty()) {
      throw std::runtime_error("Cannot specify grammar with tools");
    }
    if (vCaps.vSupportsToolCalls && !vCaps.vSupportsTools) {
      LOG_WRN("Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
              "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
    }
  }

  if (pInputs.vForcePureContent) {
    LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
    // Create the result structure
    common_chat_params vData;
    auto vParamsCopy = vParams;
    vParamsCopy.vReasoningFormat = cCommonReasoningFormatNone;
    vData.vPrompt = fCommonChatTemplateDirectApplyImpl(vTmpl, vParamsCopy);
    vData.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(vTmpl, vParams);
    vData.vFormat = cCommonChatFormatPegNative;
    auto vParser = fBuildChatPegParser([&vData](common_chat_peg_builder &pP) {
      return pP.mLiteral(vData.vGenerationPrompt) << pP.mContent(pP.mRest());
    });
    vData.vParser = vParser.mSave();
    return vData;
  }

  if (auto vResult = fCommonChatTrySpecializedTemplate(vTmpl, vSrc, vParams)) {
    return *vResult;
  }

  try {
    LOG_DBG("%s: using differential autoparser\n", __func__);
    struct autoparser::autoparser vAutoparser;
    vAutoparser.mAnalyzeTemplate(vTmpl);
    auto vAutoParams = autoparser::peg_generator::mGenerateParser(vTmpl, vParams, vAutoparser);

    common_chat_msg_delimiters vDelimiters;
    if (!vAutoparser.vAssistantStart.empty()) {
      vDelimiters.mAdd(cCommonChatRoleAssistant, vAutoparser.vAssistantStart);
    }
    if (!vAutoparser.vUserStart.empty()) {
      vDelimiters.mAdd(cCommonChatRoleUser, vAutoparser.vUserStart);
    }

    vAutoParams.vMessageDelimiters = std::move(vDelimiters);

    vAutoParams.vSupportsThinking = vAutoparser.vReasoning.vMode != autoparser::reasoning_mode::cNone;
    if (vAutoParams.vSupportsThinking) {
      vAutoParams.vThinkingStartTag = fTrimWhitespace(vAutoparser.vReasoning.vStart);
      auto vEndTag = fTrimWhitespace(vAutoparser.vReasoning.vEnd);
      if (!vEndTag.empty()) {
        vAutoParams.lThinkingEndTags = {std::move(vEndTag)};
      }
    }
    common_peg_arena vArena;
    vArena.mLoad(vAutoParams.vParser);
    LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__,
            vArena.mDump(vArena.mRoot()).c_str(), vAutoParams.vGenerationPrompt.c_str());
    return vAutoParams;
  } catch (const std::exception &vE) {
    throw std::invalid_argument(
        std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + vE.what());
  }
}

common_chat_params fCommonChatTemplatesApply(const struct common_chat_templates *pTmpls,
                                             const struct common_chat_templates_inputs &pInputs) {
  GGML_ASSERT(pTmpls != nullptr);
  return fCommonChatTemplatesApplyJinja(pTmpls, pInputs);
}

common_chat_msg fCommonChatParse(const std::string &pInput, bool pIsPartial, const common_chat_parser_params &pParams) {
  return fCommonChatPegParse(pParams.vParser, pInput, pIsPartial, pParams);
}

common_chat_msg fCommonChatPegParse(const common_peg_arena &pSrcParser, const std::string &pInput, bool pIsPartial,
                                    const common_chat_parser_params &pParams) {
  const common_peg_arena &vParser =
      pSrcParser.empty()
          ? fBuildChatPegParser([](common_chat_peg_builder &pP) { return pP.mContent(pP.mRest()) + pP.end(); })
          : pSrcParser;

  if (pSrcParser.empty()) {
    LOG_DBG("No parser definition detected, assuming pure content parser.");
  }

  const std::string cEffectiveInput = pParams.vGenerationPrompt.empty() ? pInput : pParams.vGenerationPrompt + pInput;

  // LOG_DBG("Parsing PEG input with format %s: %s\n", fCommonChatFormatName(pParams.vFormat), cEffectiveInput.c_str());

  common_peg_parse_flags vFlags = cCommonPegParseFlagLenient;
  if (pParams.vDebug) {
    vFlags |= cCommonPegParseFlagDebug;
  }

  common_peg_parse_context vCtx(cEffectiveInput, vFlags);
  auto vResult = vParser.mParse(vCtx);

  if (vResult.mFail()) {
    // During partial parsing, return partial results if any AST nodes were captured
    // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
    if (pIsPartial && vResult.vEnd > 0) {
      // Try to extract any partial results from what was successfully parsed
      common_chat_msg vMsg;
      vMsg.vRole = "assistant";
      std::unique_ptr<common_chat_peg_mapper> vMapper;
      if (pParams.vFormat == cCommonChatFormatPegMinimaxM3) {
        vMapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(vMsg);
      } else {
        vMapper = std::make_unique<common_chat_peg_mapper>(vMsg);
      }
      vMapper->mFromAst(vCtx.vAst, vResult);

      if (vCtx.mIsDebug()) {
        fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", vCtx.vAst.mDump().c_str());
        fflush(stderr);
      }
      return vMsg;
    }
    LOG_WRN("%s: unparsed %s output: %s\n", __func__, fCommonChatFormatName(pParams.vFormat),
            cEffectiveInput.substr(vResult.vEnd).c_str());
    LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__,
            fCommonChatFormatName(pParams.vFormat), cEffectiveInput.c_str());
    throw std::runtime_error(std::string("The model produced output that does not match the expected ") +
                             fCommonChatFormatName(pParams.vFormat) + " format");
  }

  common_chat_msg vMsg;
  vMsg.vRole = "assistant";

  std::unique_ptr<common_chat_peg_mapper> vMapper;
  if (pParams.vFormat == cCommonChatFormatPegMinimaxM3) {
    vMapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(vMsg);
  } else {
    vMapper = std::make_unique<common_chat_peg_mapper>(vMsg);
  }
  vMapper->mFromAst(vCtx.vAst, vResult);

  if (vCtx.mIsDebug()) {
    fprintf(stderr, "\nAST for %s parse:\n%s\n", pIsPartial ? "partial" : "full", vCtx.vAst.mDump().c_str());
    fflush(stderr);
  }

  if (!pIsPartial) {
    LOG_DBG("Parsed message: %s\n", fCommonChatMsgsToJsonOaicompat({vMsg}).at(0).dump().c_str());
  }
  return vMsg;
}

std::map<std::string, bool> fCommonChatTemplatesGetCaps(const common_chat_templates *pChatTemplates) {
  GGML_ASSERT(pChatTemplates != nullptr);
  GGML_ASSERT(pChatTemplates->vTemplateDefault != nullptr);
  if (pChatTemplates->vTemplateToolUse != nullptr) {
    // take the more expressive template when available
    return pChatTemplates->vTemplateToolUse->vCaps.mToMap();
  }
  return pChatTemplates->vTemplateDefault->vCaps.mToMap();
}
