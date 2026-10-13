// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "peg-parser.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include "nlohmann/json_fwd.hpp"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

using dChatTemplateCaps = jinja::caps;
using json = nlohmann::ordered_json;

struct common_chat_templates;

namespace autoparser {
struct vGenerationParams;
} // namespace autoparser

struct common_chat_tool_call {
  std::string vName;
  std::string vArguments;
  std::string vId;

  bool operator==(const common_chat_tool_call &pOther) const {
    return vName == pOther.vName && vArguments == pOther.vArguments && vId == pOther.vId;
  }
};

struct common_chat_msg_content_part {
  std::string vType;
  std::string vText;

  // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
  //                this can be useful for models with interleaved thinking (like Kimi-K2)
  //                if you see any templates explicitly support this, please ping me
  // std::string vReasoningContent;

  bool operator==(const common_chat_msg_content_part &pOther) const {
    return vType == pOther.vType && vText == pOther.vText;
  }
};

struct common_chat_template {
  jinja::program vProg;
  std::string vBosTok;
  std::string vEosTok;
  std::string vSrc;
  dChatTemplateCaps vCaps;

  common_chat_template(const std::string &pSrc, const std::string &pBosToken, const std::string &pEosToken) {
    jinja::lexer vLexer;
    auto vLexerRes = vLexer.mTokenize(pSrc);
    this->vProg = jinja::fParseFromTokens(vLexerRes);

    this->vSrc = vLexerRes.vSource;
    this->vBosTok = pBosToken;
    this->vEosTok = pEosToken;

    this->vCaps = jinja::fCapsGet(vProg);
    // LOG_INF("%s: caps:\n%s\n", __func__, this->vCaps.mToString().c_str());
  }

  const std::string &mSource() const { return vSrc; }
  const std::string &mBosToken() const { return vBosTok; }
  const std::string &mEosToken() const { return vEosTok; }

  dChatTemplateCaps mOriginalCaps() const { return vCaps; }
};

struct common_chat_msg {
  std::string vRole;
  std::string vContent;
  std::vector<common_chat_msg_content_part> lContentParts;
  std::vector<common_chat_tool_call> lToolCalls;
  std::string vReasoningContent;
  std::string vToolName;
  std::string vToolCallId;

  nlohmann::ordered_json mToJsonOaicompat(bool pConcatTypedText = false) const;

  std::string mRenderContent(const std::string &pDelimiter = "\n\n") const;

  bool empty() const {
    return vContent.empty() && lContentParts.empty() && lToolCalls.empty() && vReasoningContent.empty() &&
           vToolName.empty() && vToolCallId.empty();
  }

  bool mContainsMedia() const {
    for (const auto &vPart : lContentParts) {
      if (vPart.vType == "media_marker") {
        return true;
      }
    }
    return false;
  }

  void mSetToolCallIds(std::vector<std::string> &pIdsCache, const std::function<std::string()> &pGenToolCallId) {
    for (auto vI = 0u; vI < lToolCalls.size(); vI++) {
      if (pIdsCache.size() <= vI) {
        auto vId = lToolCalls[vI].vId;
        if (vId.empty()) {
          vId = pGenToolCallId();
        }
        pIdsCache.push_back(vId);
      }
      lToolCalls[vI].vId = pIdsCache[vI];
    }
  }

  bool operator==(const common_chat_msg &pOther) const {
    return vRole == pOther.vRole && vContent == pOther.vContent && lContentParts == pOther.lContentParts &&
           lToolCalls == pOther.lToolCalls && vReasoningContent == pOther.vReasoningContent &&
           vToolName == pOther.vToolName && vToolCallId == pOther.vToolCallId;
  }

  bool operator!=(const common_chat_msg &pOther) const { return !(*this == pOther); }
};

struct common_chat_msg_diff {
  std::string vReasoningContentDelta;
  std::string vContentDelta;
  size_t vToolCallIndex = std::string::npos;
  common_chat_tool_call vToolCallDelta;

  static std::vector<common_chat_msg_diff> mComputeDiffs(const common_chat_msg &pMsgPrv,
                                                         const common_chat_msg &pMsgNew);

  bool operator==(const common_chat_msg_diff &pOther) const {
    return vContentDelta == pOther.vContentDelta && vToolCallIndex == pOther.vToolCallIndex &&
           vToolCallDelta == pOther.vToolCallDelta;
  }
};

enum common_chat_role {
  cCommonChatRoleUnknown,
  cCommonChatRoleSystem,
  cCommonChatRoleAssistant,
  cCommonChatRoleUser,
  cCommonChatRoleTool
};

common_chat_role fCommonChatRoleFromString(const std::string &pRole);
const char *fCommonChatRoleToString(common_chat_role pRole);

struct common_chat_msg_span {
  common_chat_role vRole = cCommonChatRoleUnknown;
  std::size_t vPos = 0;
  std::size_t vLen = 0;

  bool mValid() const { return vRole != cCommonChatRoleUnknown; }
};

struct common_chat_msg_spans {
  std::vector<common_chat_msg_span> lSpans;

  void mAdd(common_chat_role pRole, size_t pPos, size_t pLen) { lSpans.push_back({pRole, pPos, pLen}); }

  bool mIsUserStart(int32_t pPos) const {
    for (auto vIt = lSpans.begin(); vIt != lSpans.end(); ++vIt) {
      if (vIt->vRole == cCommonChatRoleUser && pPos == (int32_t)vIt->vPos) {
        return true;
      }
    }
    return false;
  }

  // Start of a user, assistant or tool message: the boundaries where agents edit the context
  // (old reasoning removed, tool results trimmed or replaced), so prefix reuse resumes there.
  bool mIsTurnStart(int32_t pPos) const {
    for (auto vIt = lSpans.begin(); vIt != lSpans.end(); ++vIt) {
      if (pPos == (int32_t)vIt->vPos && (vIt->vRole == cCommonChatRoleUser || vIt->vRole == cCommonChatRoleAssistant ||
                                         vIt->vRole == cCommonChatRoleTool)) {
        return true;
      }
    }
    return false;
  }

  int32_t mLastUserMessagePos() const {
    for (auto vIt = lSpans.rbegin(); vIt != lSpans.rend(); ++vIt) {
      if (vIt->vRole == cCommonChatRoleUser) {
        return (int32_t)vIt->vPos;
      }
    }
    return -1;
  }
};

struct common_chat_msg_delimiter {
  common_chat_role vRole = cCommonChatRoleUnknown;
  std::string vDelimiter;
  mgs_tokens vTokens = {};
};

struct common_chat_msg_delimiters {
  std::vector<common_chat_msg_delimiter> lDelimiters;

  common_chat_msg_delimiters() = default;
  common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> pDelims) : lDelimiters(pDelims) {}

  void mAdd(common_chat_role pRole, const std::string &pDelimiter) { lDelimiters.push_back({pRole, pDelimiter}); }

  void mTokenize(const mgs_vocab *pVocab);

  // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
  common_chat_msg_spans mSplit(const mgs_tokens &pTokens, const std::map<size_t, size_t> &pSkips = {}) const;

  nlohmann::ordered_json to_json() const;
};

struct common_chat_tool {
  std::string vName;
  std::string vDescription;
  std::string vParameters;
};

enum common_chat_tool_choice {
  cCommonChatToolChoiceAuto,
  cCommonChatToolChoiceRequired,
  cCommonChatToolChoiceNone,
};

enum common_chat_format {
  cCommonChatFormatContentOnly,

  // These are intended to be parsed by the PEG parser
  cCommonChatFormatPegSimple,
  cCommonChatFormatPegNative,
  cCommonChatFormatPegMinimaxM3,

  cCommonChatFormatCount, // Not a format, just the # formats
};

// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
  cCommonChatContinuationNone,
  cCommonChatContinuationAuto,
  cCommonChatContinuationReasoning,
  cCommonChatContinuationContent,
};

struct common_chat_templates_inputs {
  std::vector<common_chat_msg> lMessages;
  std::string vGrammar;
  std::string vJsonSchema;
  bool vAddGenerationPrompt = true;
  common_chat_continuation vContinueFinalMessage = cCommonChatContinuationNone;
  std::vector<common_chat_tool> lTools;
  common_chat_tool_choice vToolChoice = cCommonChatToolChoiceAuto;
  bool vParallelToolCalls = false;
  common_reasoning_format vReasoningFormat =
      cCommonReasoningFormatNone; // TODO: refactor this to "bool enable_thinking"
  bool vEnableThinking = true;
  std::chrono::system_clock::time_point vNow = std::chrono::system_clock::now();
  std::map<std::string, std::string> dChatTemplateKwargs;
  bool vAddBos = false;
  bool vAddEos = false;
  bool vForcePureContent = false;
};

struct common_chat_params {
  common_chat_format vFormat = cCommonChatFormatContentOnly;
  std::string vPrompt;
  std::string vGrammar;
  bool vGrammarLazy = false;
  std::string vGenerationPrompt;
  bool vSupportsThinking = false;
  std::string vThinkingStartTag;             // e.g., "<think>"
  std::vector<std::string> lThinkingEndTags; // e.g., "</think>"
  std::vector<common_grammar_trigger> lGrammarTriggers;
  std::vector<std::string> lPreservedTokens;
  std::vector<std::string> lAdditionalStops;
  std::string vParser;
  common_chat_msg_delimiters vMessageDelimiters;
};

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
  common_chat_format vFormat = cCommonChatFormatContentOnly;
  common_reasoning_format vReasoningFormat =
      cCommonReasoningFormatNone; // TODO: refactor this to "bool parse_reasoning"
  std::string vGenerationPrompt;
  bool vParseToolCalls = true;
  bool vIsContinuation = false;
  bool vEcho = false;  // Include assistant prefilled msg in output
  bool vDebug = false; // Enable debug output for PEG parser
  common_peg_arena vParser = {};
  common_chat_parser_params() = default;
  common_chat_parser_params(const common_chat_params &pChatParams) {
    vFormat = pChatParams.vFormat;
    vGenerationPrompt = pChatParams.vGenerationPrompt;
  }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool fCommonChatVerifyTemplate(const std::string &pTmpl);

void fCommonChatTemplatesFree(struct common_chat_templates *pTmpls);

struct common_chat_templates_deleter {
  void operator()(common_chat_templates *pTmpls) { fCommonChatTemplatesFree(pTmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr fCommonChatTemplatesInit(const struct mgs_model *pModel,
                                                   const std::string &pChatTemplateOverride,
                                                   const std::string &pBosTokenOverride = "",
                                                   const std::string &pEosTokenOverride = "");

bool fCommonChatTemplatesWasExplicit(const struct common_chat_templates *pTmpls);
std::string fCommonChatTemplatesSource(const struct common_chat_templates *pTmpls, const std::string &pVariant = "");

struct common_chat_params fCommonChatTemplatesApply(const struct common_chat_templates *pTmpls,
                                                    const struct common_chat_templates_inputs &pInputs);

// Returns an example of formatted chat
std::string fCommonChatFormatExample(const struct common_chat_templates *pTmpls,
                                     const std::map<std::string, std::string> &pChatTemplateKwargs);

const char *fCommonChatFormatName(common_chat_format pFormat);
common_chat_msg fCommonChatParse(const std::string &pInput, bool pIsPartial, const common_chat_parser_params &pParams);
common_chat_msg fCommonChatPegParse(const common_peg_arena &pSrcParser, const std::string &pInput, bool pIsPartial,
                                    const common_chat_parser_params &pParams);

// used by arg and server
const char *fCommonReasoningFormatName(common_reasoning_format pFormat);
common_reasoning_format fCommonReasoningFormatFromName(const std::string &pFormat);

common_chat_tool_choice fCommonChatToolChoiceParseOaicompat(const std::string &pToolChoice);

bool fCommonChatTemplatesSupportEnableThinking(const common_chat_templates *pChatTemplates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> fCommonChatMsgsParseOaicompat(const nlohmann::ordered_json &pMessages);

std::vector<common_chat_tool> fCommonChatToolsParseOaicompat(const nlohmann::ordered_json &pTools);

common_chat_continuation fCommonChatContinuationParse(const nlohmann::ordered_json &pValue);

nlohmann::ordered_json fCommonChatMsgsToJsonOaicompat(const std::vector<common_chat_msg> &pMsgs,
                                                      bool pConcatTypedText = false);

nlohmann::ordered_json fCommonChatToolsToJsonOaicompat(const std::vector<common_chat_tool> &pTools);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> fCommonChatTemplatesGetCaps(const common_chat_templates *pChatTemplates);

std::string fCommonChatTemplateDirectApply(const common_chat_template &pTmpl,
                                           const autoparser::vGenerationParams &pInputs);

std::string fCommonChatTemplateGenerationPrompt(const common_chat_template &pTmpl,
                                                const autoparser::vGenerationParams &pInputs);

std::optional<common_chat_params> fCommonChatTrySpecializedTemplate(const common_chat_template &pTmpl,
                                                                    const std::string &pSrc,
                                                                    autoparser::vGenerationParams &pParams);

// specialized per-task preset
struct common_chat_prompt_preset {
  std::string vSystem;
  std::string vUser;
};

common_chat_prompt_preset fCommonChatGetAsrPrompt(const common_chat_templates *pChatTemplates);

common_chat_msg_delimiters fCommonChatMsgDelimitersParse(const nlohmann::ordered_json &pDelimiters);
