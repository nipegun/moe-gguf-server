#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

// Helper to iterate over tools/functions
static void fForeachFunction(const json &pTools, const std::function<void(const json &)> &pFn) {
  for (const auto &vTool : pTools) {
    if (!vTool.contains("type") || vTool.at("type") != "function" || !vTool.contains("function")) {
      continue;
    }
    pFn(vTool);
  }
}

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder &pP, const vGenerationParams &pInputs)
    : vP(pP), vInputs(pInputs), vReasoningParser(pP.mEps()) {}

common_chat_params peg_generator::mGenerateParser(const common_chat_template &pTmpl,
                                                  const struct vGenerationParams &pInputs) {
  // Run differential analysis to extract template structure
  struct autoparser vAutoparser;
  vAutoparser.mAnalyzeTemplate(pTmpl);
  return mGenerateParser(pTmpl, pInputs, vAutoparser);
}

common_chat_params peg_generator::mGenerateParser(const common_chat_template &pTmpl,
                                                  const struct vGenerationParams &pInputs,
                                                  const autoparser &pAutoparser) {
  // Create the result structure
  common_chat_params vData;
  vData.vPrompt = fCommonChatTemplateDirectApply(pTmpl, pInputs);
  vData.vGenerationPrompt = fCommonChatTemplateGenerationPrompt(pTmpl, pInputs);
  vData.vFormat = cCommonChatFormatPegNative;
  vData.lPreservedTokens = pAutoparser.lPreservedTokens;
  vData.lAdditionalStops.insert(vData.lAdditionalStops.end(), pAutoparser.lAdditionalStops.begin(),
                                pAutoparser.lAdditionalStops.end());

  std::string vParserGenerationPrompt = vData.vGenerationPrompt;

  if (pInputs.vContinueFinalMessage != cCommonChatContinuationNone && !pInputs.vContinueMsg.empty()) {
    // Build up generation prompt manually
    const auto &vMsg = pInputs.vContinueMsg;

    if (!pAutoparser.vReasoning.vStart.empty()) {
      vData.vGenerationPrompt =
          vData.vGenerationPrompt.substr(0, vData.vGenerationPrompt.find(pAutoparser.vReasoning.vStart));
      vData.vGenerationPrompt += pAutoparser.vReasoning.vStart + vMsg.vReasoningContent;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        vData.vGenerationPrompt += pAutoparser.vReasoning.vEnd;
      }
    }

    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      vData.vGenerationPrompt += vMsg.mRenderContent();
    }

    vData.vPrompt += vData.vGenerationPrompt;
  }

  auto vParser = pAutoparser.mBuildParser(pInputs, vParserGenerationPrompt);
  vData.vParser = vParser.mSave();

  // Build grammar if tools are present
  bool vHasTools =
      pAutoparser.vTools.vFormat.vMode != tool_format::cNone && pInputs.vTools.is_array() && !pInputs.vTools.empty();
  std::string vTriggerMarker = !pAutoparser.vTools.vFormat.vSectionStart.empty()
                                   ? pAutoparser.vTools.vFormat.vSectionStart
                                   : pAutoparser.vTools.vFormat.vPerCallStart;

  bool vHasResponseFormat = !pInputs.vJsonSchema.empty() && pInputs.vJsonSchema.is_object();
  bool vIncludeGrammar = vHasResponseFormat ||
                         (vHasTools && ((pInputs.vToolChoice == cCommonChatToolChoiceAuto && !vTriggerMarker.empty()) ||
                                        pInputs.vToolChoice == cCommonChatToolChoiceRequired));

  if (vIncludeGrammar) {
    vData.vGrammarLazy = !vHasResponseFormat && pInputs.vToolChoice == cCommonChatToolChoiceAuto;
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

    // Set grammar triggers based on tool section markers (fall back to per-call markers)
    if (vData.vGrammarLazy) {
      vData.lGrammarTriggers = {{cCommonGrammarTriggerTypeWord, vTriggerMarker}};
      if (pAutoparser.vTools.vFormat.vOpenaiWrapperTrigger) {
        // model emits the OpenAI function wrapper, trigger on it
        vData.lGrammarTriggers.push_back({cCommonGrammarTriggerTypeWord, "{\"type\": \"function\","});
      }
    }
  }

  return vData;
}

common_peg_arena autoparser::mBuildParser(const vGenerationParams &pInputs,
                                          const std::string &pGenerationPrompt) const {
  if (!vAnalysisComplete) {
    throw std::invalid_argument(
        "Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
  }
  return fBuildChatPegParser([&](common_chat_peg_builder &pP) {
    parser_build_context vCtx(pP, pInputs);
    bool vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;

    vCtx.vExtractingReasoning = vExtractReasoning && vReasoning.vMode != reasoning_mode::cNone;
    vCtx.vContent = &vContent;
    vCtx.vReasoning = &vReasoning;

    // Build reasoning parser
    vCtx.vReasoningParser = vReasoning.mBuildParser(vCtx);

    auto vParser = pP.mEps();

    bool vHasTools = pInputs.vTools.is_array() && !pInputs.vTools.empty();
    bool vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
    bool vPureContent = vReasoning.vMode == reasoning_mode::cNone;

    if (vHasResponseFormat) {
      auto vResponseFormat = pP.mRule(
          "response-format", pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)));
      vParser = vCtx.vReasoningParser + pP.mSpace() +
                pP.mChoice({pP.mLiteral("```json") + pP.mSpace() + vResponseFormat + pP.mSpace() + pP.mLiteral("```"),
                            pP.mSpace() + vResponseFormat + pP.mSpace()}) +
                pP.end();
      vPureContent = false;
    } else if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone && vJinjaCaps.vSupportsToolCalls) {
      vParser = vTools.mBuildParser(vCtx);
      vPureContent = false;
    } else {
      vParser = vContent.mBuildParser(vCtx);
    }
    const std::string cReasoningStart = fTrimWhitespace(vReasoning.vStart);
    return vPureContent ? pP.mPrefix(pGenerationPrompt, cReasoningStart) + vParser
                        : pP.mPrefix(pGenerationPrompt, cReasoningStart) << vParser;
  });
}

common_peg_parser analyze_reasoning::mBuildParser(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;

  if (!pCtx.vExtractingReasoning) {
    return vP.mEps();
  }

  if (vMode == reasoning_mode::cTagBased || vMode == reasoning_mode::cToolsOnly) {
    if (!vEnd.empty()) {
      if (!vStart.empty()) {
        // Standard tag-based: optional(<think>reasoning</think>)
        return vP.mOptional(vP.mOptspace(vStart) + vP.mReasoning(vP.mUntil(fTrimWhitespace(vEnd))) +
                            vP.mOptspace(vEnd));
      }
      // Delimiter-style (empty start)
      return vP.mOptional(vP.mReasoning(vP.mUntil(fTrimWhitespace(vEnd))) + vP.mOptspace(vEnd));
    }
  }

  return vP.mEps();
}

common_peg_parser analyze_content::mBuildParser(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;

  if (mIsAlwaysWrapped()) {
    if (pCtx.vExtractingReasoning) {
      return pCtx.vReasoningParser + vStart + vP.mContent(vP.mUntil(vEnd)) + vEnd + vP.end();
    }
    return vP.mContent(vP.mUntil(vStart)) + vStart + vP.mContent(vP.mUntil(vEnd)) + vEnd + vP.end();
  }
  return pCtx.vReasoningParser + vP.mContent(vP.mRest()) + vP.end();
}

common_peg_parser analyze_content::mBuildOptionalWrapped(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;

  if (mIsAlwaysWrapped()) {
    return vP.mOptional(vStart + vP.mContent(vP.mUntil(vEnd)) + vEnd);
  }
  return vP.mEps();
}

common_peg_parser analyze_tools::mBuildParser(parser_build_context &pCtx) const {
  switch (vFormat.vMode) {
  case tool_format::cJsonNative:
    return mBuildToolParserJsonNative(pCtx);
  case tool_format::cTagWithJson:
    return mBuildToolParserTagJson(pCtx);
  case tool_format::cTagWithTagged:
    return mBuildToolParserTagTagged(pCtx);
  default:
    LOG_ERR(
        "[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work "
        "properly. "
        "Check for a fixed template for your model (--chat-template-file) or "
        "report an issue at https://github.com/nipegun/moe-gguf-server/issues\n");
    return pCtx.vP.mEps();
  }
}

common_peg_parser analyze_tools::mBuildToolParserJsonNative(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;
  const auto &vInputs = pCtx.vInputs;

  // Build effective field names with dot notation if function_field is set
  std::string vNameField = vFormat.vNameField;
  std::string vArgsField = vFormat.vArgsField;

  if (!vFormat.vFunctionField.empty() && vFormat.vFunctionField != "function" &&
      vNameField.find('.') == std::string::npos) {
    vNameField = vFormat.vFunctionField + "." + vNameField;
    vArgsField = vFormat.vFunctionField + "." + vArgsField;
  }

  auto vToolsParser = vP.mEps();
  if (vFormat.vSectionStart.empty() && !vFormat.vPerCallStart.empty()) {
    auto vSingleToolParser =
        vP.mStandardJsonTools(vFormat.vPerCallStart, vFormat.vPerCallEnd, vInputs.vTools, vInputs.vParallelToolCalls,
                              vInputs.vToolChoice == cCommonChatToolChoiceRequired, vNameField, vArgsField,
                              vFormat.vToolsArrayWrapped, vFormat.vFunNameIsKey, vFormat.vIdField, vFormat.vGenIdField,
                              vFormat.lParameterOrder, vFormat.vOpenaiWrapperTrigger);
    vToolsParser = vP.mTriggerRule("tool-calls", vP.mOneOrMore(vSingleToolParser + vP.mSpace()));
  } else {
    vToolsParser =
        vP.mStandardJsonTools(vFormat.vSectionStart, vFormat.vSectionEnd, vInputs.vTools, vInputs.vParallelToolCalls,
                              vInputs.vToolChoice == cCommonChatToolChoiceRequired, vNameField, vArgsField,
                              vFormat.vToolsArrayWrapped, vFormat.vFunNameIsKey, vFormat.vIdField, vFormat.vGenIdField,
                              vFormat.lParameterOrder, vFormat.vOpenaiWrapperTrigger);
  }

  // Handle content wrappers if present
  if (pCtx.vContent && pCtx.vContent->mIsAlwaysWrapped()) {
    auto vWrappedContent = pCtx.vContent->mBuildOptionalWrapped(pCtx);
    return pCtx.vReasoningParser + vWrappedContent + vToolsParser + vP.end();
  }

  std::string vToolStart = "{";
  if (!vFormat.vSectionStart.empty()) {
    vToolStart = vFormat.vSectionStart;
  } else if (!vFormat.vPerCallStart.empty()) {
    vToolStart = vFormat.vPerCallStart;
  }

  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vP.mUntil(vToolStart))) + vToolsParser + vP.end();
}

common_peg_parser analyze_tools::mBuildFuncParser(common_chat_peg_builder &pP, const std::string &pName,
                                                  const common_peg_parser &pCallIdSection, bool pHaveCallId,
                                                  const common_peg_parser &pArgs,
                                                  std::optional<common_peg_parser> pAtomicPeek) const {
  auto vOpen = pP.mToolOpen(vFunction.vNamePrefix + pP.mToolName(pP.mLiteral(pName)) + vFunction.vNameSuffix);
  bool vMatchedAtomic = false;
  common_peg_parser vFuncParser = pP.mEps();

  if (!vFunction.vArgsSeparator.empty()) {
    vOpen = vOpen + pP.mSpace() + pP.mLiteral(vFunction.vArgsSeparator);
  }

  if (!vFunction.vNameSuffix.empty()) {
    vFuncParser = vOpen + pCallIdSection + pP.mSpace() + pArgs;
    vMatchedAtomic = true;
  } else if (pHaveCallId) {
    vFuncParser = pP.mAtomic(vOpen + pCallIdSection) + pP.mSpace() + pArgs;
    vMatchedAtomic = true;
  } else if (pAtomicPeek.has_value()) {
    vFuncParser = pP.mAtomic(vOpen + pCallIdSection + pP.mSpace() + *pAtomicPeek) + pArgs;
    vMatchedAtomic = true;
  } else {
    vFuncParser = vOpen + pCallIdSection + pP.mSpace() + pArgs;
  }

  if (!vFunction.vClose.empty()) {
    vFuncParser = vFuncParser + pP.mSpace() + pP.mToolClose(pP.mLiteral(vFunction.vClose));
  } else if (!vFormat.vPerCallEnd.empty()) {
    // When there's no func_close but there is a per_call_end marker, use mPeek() to ensure
    // we only emit tool_close when we can actually see the closing marker. This prevents
    // premature closing during partial parsing when we've seen e.g. "</" which could be
    // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
    // Laguna (v4): the model may emit whitespace between the last </arg_value> and
    // </tool_call> even though the template renders them tight. Tolerate optional
    // leading space in the close lookahead so the tool call still closes.
    auto vClosePeek = vArguments.vTolerateIntertagWhitespace ? pP.mPeek(pP.mSpace() + pP.mLiteral(vFormat.vPerCallEnd))
                                                             : pP.mPeek(pP.mLiteral(vFormat.vPerCallEnd));
    vFuncParser = vFuncParser + pP.mToolClose(vClosePeek);
  } else {
    vFuncParser = vFuncParser + pP.mToolClose(pP.mSpace()); // force this to process tool closing callbacks in mapper
  }
  if (!vMatchedAtomic) {
    vFuncParser = pP.mAtomic(vFuncParser);
  }
  return vFuncParser;
}

common_peg_parser analyze_tools::mBuildToolParserTagJson(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;
  const auto &vInputs = pCtx.vInputs;

  common_peg_parser vToolChoice = vP.mChoice();

  fForeachFunction(vInputs.vTools, [&](const json &pTool) {
    const auto &vFunc = pTool.at("function");
    std::string vName = vFunc.at("name");
    const auto &vSchema = vFunc.contains("parameters") ? vFunc.at("parameters") : json::object();

    // Build call_id parser based on position (if supported)
    bool vHaveCallId = false;
    common_peg_parser vCallIdSection = vP.mEps();
    if (vCallId.vPos == call_id_position::cBetweenFuncAndArgs && !vCallId.vPrefix.empty() &&
        (!vCallId.vSuffix.empty() || !vArguments.vStart.empty())) {
      if (!vCallId.vSuffix.empty()) {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vCallId.vSuffix))) + vCallId.vSuffix;
      } else {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vArguments.vStart)));
      }
      vHaveCallId = true;
    }
    auto vArgsParser = vP.mToolArgs(vP.mSchema(vP.mJson(), "tool-" + vName + "-schema", vSchema));
    if (!vArguments.vStart.empty()) {
      vArgsParser = vP.mLiteral(vArguments.vStart) + vArgsParser;
    }
    if (!vArguments.vEnd.empty()) {
      vArgsParser = vArgsParser + vP.mLiteral(vArguments.vEnd);
    }

    auto vAtomicPeek =
        !vArguments.vStart.empty() ? std::optional(vP.mPeek(vP.mLiteral(vArguments.vStart))) : std::nullopt;
    auto vFuncParser = mBuildFuncParser(vP, vName, vCallIdSection, vHaveCallId, vArgsParser, vAtomicPeek);
    vToolChoice |= vP.mRule("tool-" + vName, vFuncParser);
  });

  auto vRequireCalls = vInputs.vToolChoice == cCommonChatToolChoiceRequired;

  common_peg_parser vToolCalls = vP.mEps();

  if (!vFormat.vPerCallStart.empty()) {
    auto vWrappedCall = vFormat.vPerCallStart + vToolChoice + vFormat.vPerCallEnd;
    if (vInputs.vParallelToolCalls) {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall + vP.mZeroOrMore(vP.mSpace() + vWrappedCall));
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall);
    }
    if (!vFormat.vSectionStart.empty()) {
      vToolCalls = vP.mTriggerRule("tool-calls",
                                   vP.mLiteral(vFormat.vSectionStart) + vP.mSpace() + vToolCalls + vP.mSpace() +
                                       (vFormat.vSectionEnd.empty() ? vP.end() : vP.mLiteral(vFormat.vSectionEnd)));
    }
  } else {
    std::string vSeparator = ", "; // Default
    if (vInputs.vParallelToolCalls) {
      vToolCalls = vP.mTriggerRule("tool-call", vFormat.vSectionStart + vToolChoice +
                                                    vP.mZeroOrMore(vSeparator + vToolChoice) + vFormat.vSectionEnd);
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vFormat.vSectionStart + vToolChoice + vFormat.vSectionEnd);
    }
  }

  if (!vRequireCalls) {
    vToolCalls = vP.mOptional(vToolCalls);
  }

  std::string vTriggerMarker = !vFormat.vSectionStart.empty() ? vFormat.vSectionStart : vFormat.vPerCallStart;
  auto vContentBeforeTools = vTriggerMarker.empty() ? vP.mEps() : vP.mUntil(vTriggerMarker);
  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vContentBeforeTools)) + vToolCalls + vP.end();
}

common_peg_parser analyze_tools::mBuildToolParserTagTagged(parser_build_context &pCtx) const {
  auto &vP = pCtx.vP;
  const auto &vInputs = pCtx.vInputs;

  auto vUntilSuffix = vP.mRule("until-suffix", vP.mUntil(vArguments.vValueSuffix));

  common_peg_parser vToolChoice = vP.mChoice();

  fForeachFunction(vInputs.vTools, [&](const json &pTool) {
    const auto &vFunc = pTool.at("function");
    std::string vName = vFunc.at("name");
    auto vParams = vFunc.contains("parameters") ? vFunc.at("parameters") : json::object();
    const auto &vProperties = vParams.contains("properties") ? vParams.at("properties") : json::object();

    std::set<std::string> sRequired;
    if (vParams.contains("required")) {
      vParams.at("required").get_to(sRequired);
    }

    auto vSchemaInfo = common_schema_info();
    vSchemaInfo.mResolveRefs(vParams);

    // Build parser for each argument, separating required and optional
    std::vector<common_peg_parser> lRequiredParsers;
    std::vector<common_peg_parser> lOptionalParsers;
    for (const auto &[vParamName, pParamSchema] : vProperties.items()) {
      bool vIsRequired = sRequired.find(vParamName) != sRequired.end();

      auto vArg = vP.mToolArg(
          vP.mToolArgOpen(vArguments.vNamePrefix + vP.mToolArgName(vP.mLiteral(vParamName)) + vArguments.vNameSuffix) +
          vArguments.vValuePrefix +
          (vSchemaInfo.mResolvesToString(pParamSchema)
               ? vP.mAc(vP.mToolArgStringValue(vUntilSuffix) + vP.mToolArgClose(vP.mLiteral(vArguments.vValueSuffix)),
                        vArguments.vValueSuffix)
               : (vP.mToolArgJsonValue(
                      vP.mSchema(vP.mJson(), "tool-" + vName + "-arg-" + vParamName + "-schema", pParamSchema, false)) +
                  vP.mToolArgClose(vP.mLiteral(vArguments.vValueSuffix)))));

      auto vNamedArg = vP.mRule("tool-" + vName + "-arg-" + vParamName, vArg);
      if (vIsRequired) {
        lRequiredParsers.push_back(vNamedArg);
      } else {
        lOptionalParsers.push_back(vNamedArg);
      }
    }

    // Build required arg sequence in definition order
    common_peg_parser vArgsSeq = vP.mEps();
    for (size_t vI = 0; vI < lRequiredParsers.size(); vI++) {
      if (vI > 0) {
        vArgsSeq = vArgsSeq + vP.mSpace();
      }
      vArgsSeq = vArgsSeq + lRequiredParsers[vI];
    }

    // Build optional args with flexible ordering
    if (!lOptionalParsers.empty()) {
      common_peg_parser vAnyOpt = vP.mChoice();
      for (const auto &vOpt : lOptionalParsers) {
        vAnyOpt |= vOpt;
      }
      vArgsSeq = vArgsSeq + vP.mRepeat(vP.mSpace() + vAnyOpt, 0, -1);
    }

    if (!vArguments.vStart.empty()) {
      vArgsSeq = vP.mLiteral(vArguments.vStart) + vArgsSeq;
    }
    if (!vArguments.vEnd.empty()) {
      vArgsSeq = vArgsSeq + vP.mLiteral(vArguments.vEnd);
    }

    // Build call_id parser based on position (if supported)
    common_peg_parser vCallIdSection = vP.mEps();
    bool vHaveCallId = false;
    if (vCallId.vPos == call_id_position::cBetweenFuncAndArgs && !vCallId.vPrefix.empty() &&
        (!vCallId.vSuffix.empty() || !vArguments.vStart.empty())) {
      vHaveCallId = true;
      if (!vCallId.vSuffix.empty()) {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vCallId.vSuffix)) + vCallId.vSuffix);
      } else {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vArguments.vStart)));
      }
    }

    // Only peek for an arg tag when there are required args that must follow.
    // When all args are optional, the model may emit no arg tags at all (#20650).
    auto vAtomicPeek = (!vArguments.vNamePrefix.empty() && !lRequiredParsers.empty())
                           ? std::optional(vP.mPeek(vP.mLiteral(vArguments.vNamePrefix)))
                           : std::nullopt;
    auto vFuncParser = mBuildFuncParser(vP, vName, vCallIdSection, vHaveCallId, vArgsSeq, vAtomicPeek);
    vToolChoice |= vP.mRule("tool-" + vName, vFuncParser);
  });

  auto vRequireTools = vInputs.vToolChoice == cCommonChatToolChoiceRequired;

  common_peg_parser vToolCalls = vP.mEps();

  if (!vFormat.vPerCallStart.empty()) {
    auto vWrappedCall = vFormat.vPerCallStart + vP.mSpace() + vToolChoice + vP.mSpace() + vFormat.vPerCallEnd;
    if (vInputs.vParallelToolCalls) {
      vToolCalls =
          vP.mTriggerRule("tool-call", vWrappedCall + vP.mZeroOrMore(vP.mSpace() + vWrappedCall) + vP.mSpace());
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall + vP.mSpace());
    }
    if (!vFormat.vSectionStart.empty()) {
      vToolCalls = vP.mTriggerRule(
          "tool-calls", vP.mLiteral(vFormat.vSectionStart) + vP.mSpace() + vToolCalls + vP.mSpace() +
                            (vFormat.vSectionEnd.empty() ? vP.end() : vP.mLiteral(vFormat.vSectionEnd) + vP.mSpace()));
    }
  } else {
    std::string vSeparator = ", "; // Default

    if (vInputs.vParallelToolCalls) {
      vToolCalls = vP.mTriggerRule("tool-call", vFormat.vSectionStart + vP.mSpace() + vToolChoice +
                                                    vP.mZeroOrMore(vSeparator + vToolChoice) + vP.mSpace() +
                                                    vFormat.vSectionEnd);
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vFormat.vSectionStart + vP.mSpace() + vToolChoice + vP.mSpace() +
                                                    vFormat.vSectionEnd);
    }
  }

  if (!vRequireTools) {
    vToolCalls = vP.mOptional(vToolCalls);
  }

  std::string vTriggerMarker = !vFormat.vSectionStart.empty() ? vFormat.vSectionStart : vFormat.vPerCallStart;
  auto vContentBeforeTools = vTriggerMarker.empty() ? vP.mEps() : vP.mUntil(vTriggerMarker);
  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vContentBeforeTools)) + vToolCalls + vP.end();
}

} // namespace autoparser
