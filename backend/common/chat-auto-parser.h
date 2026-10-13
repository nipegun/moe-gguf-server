#pragma once

#include "chat.h"
#include "common.h"
#include "jinja/caps.h"
#include "peg-parser.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

class common_chat_peg_builder;

// ============================================================================
// Parameters for template application (low-level, used by diff analysis)
// ============================================================================
struct template_params {
  json vMessages;
  json vTools;
  bool vAddGenerationPrompt = false;
  bool vEnableThinking = true;
  std::optional<json> vExtraContext = std::nullopt;
};

struct diff_split {
  std::string vPrefix;
  std::string vSuffix;
  std::string vLeft;
  std::string vRight;

  bool operator==(struct diff_split &pOther) const {
    return vPrefix == pOther.vPrefix && vSuffix == pOther.vSuffix && vLeft == pOther.vLeft && vRight == pOther.vRight;
  }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
  diff_split vDiff;
  std::string vOutputA;
  std::string vOutputB;
};

namespace autoparser {

// ============================================================================
// High-level params for parser generation
// ============================================================================

struct vGenerationParams {
  json vMessages;
  json vTools;
  common_chat_tool_choice vToolChoice = cCommonChatToolChoiceAuto;
  json vJsonSchema;
  bool vParallelToolCalls = true;
  common_reasoning_format vReasoningFormat = cCommonReasoningFormatAuto;
  bool vStream = true;
  std::string vGrammar;
  bool vAddGenerationPrompt = false;
  common_chat_continuation vContinueFinalMessage = cCommonChatContinuationNone;
  common_chat_msg vContinueMsg;
  bool vEnableThinking = true;
  std::chrono::system_clock::time_point vNow = std::chrono::system_clock::now();
  json vExtraContext;
  bool vAddBos = false;
  bool vAddEos = false;
  bool vIsInference = true;
  bool vAddInference = false;
  bool vMarkInput = true; // whether to mark input strings in the jinja context

  bool mHasContinuation() const {
    return vContinueFinalMessage != cCommonChatContinuationNone && !vContinueMsg.empty();
  }
};

// ============================================================================
// Analysis Result Enums
// ============================================================================

// Reasoning handling mode (derived from R1-R3 comparisons)
enum class reasoning_mode {
  cNone,     // No reasoning markers detected
  cTagBased, // Tag-based: <think>...</think> (start can be empty for delimiter-style)
  cToolsOnly // Only reason on tool calls, not on normal content
};

inline std::ostream &operator<<(std::ostream &pOs, const reasoning_mode &pMode) {
  switch (pMode) {
  case reasoning_mode::cNone:
    return pOs << "NONE";
  case reasoning_mode::cTagBased:
    return pOs << "TAG_BASED";
  case reasoning_mode::cToolsOnly:
    return pOs << "TOOLS_ONLY";
  default:
    return pOs << "UNKNOWN";
  }
}

// Content wrapping mode (derived from C1 comparison)
enum class content_mode {
  cPlain,                // No content markers
  cAlwaysWrapped,        // Content always wrapped with markers
  cWrappedWithReasoning, // Content wrapped only when reasoning present
};

inline std::ostream &operator<<(std::ostream &pOs, const content_mode &pMode) {
  switch (pMode) {
  case content_mode::cPlain:
    return pOs << "PLAIN";
  case content_mode::cAlwaysWrapped:
    return pOs << "ALWAYS_WRAPPED";
  case content_mode::cWrappedWithReasoning:
    return pOs << "WRAPPED_WITH_REASONING";
  default:
    return pOs << "UNKNOWN";
  }
}

// Call ID position in tool calls (for non-JSON formats)
enum class call_id_position {
  cNone,               // No call ID support detected
  cPreFuncName,        // Call ID before function name: [CALL_ID]id[FUNC]name{args}
  cBetweenFuncAndArgs, // Call ID between function and args: [FUNC]name[CALL_ID]id{args}
  cPostArgs,           // Call ID after arguments: [FUNC]name{args}[CALL_ID]id
};

inline std::ostream &operator<<(std::ostream &pOs, const call_id_position &pPos) {
  switch (pPos) {
  case call_id_position::cNone:
    return pOs << "NONE";
  case call_id_position::cPreFuncName:
    return pOs << "PRE_FUNC_NAME";
  case call_id_position::cBetweenFuncAndArgs:
    return pOs << "BETWEEN_FUNC_AND_ARGS";
  case call_id_position::cPostArgs:
    return pOs << "POST_ARGS";
  default:
    return pOs << "UNKNOWN";
  }
}

// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
enum class tool_format {
  cNone,          // No tool support detected
  cJsonNative,    // Pure JSON: {"name": "X", "arguments": {...}}
  cTagWithJson,   // Tag-based with JSON args: <function=X>{...}</function>
  cTagWithTagged, // Tag-based with tagged args: <param=key>value</param>
};

inline std::ostream &operator<<(std::ostream &pOs, const tool_format &pFormat) {
  switch (pFormat) {
  case tool_format::cNone:
    return pOs << "NONE";
  case tool_format::cJsonNative:
    return pOs << "JSON_NATIVE";
  case tool_format::cTagWithJson:
    return pOs << "TAG_WITH_JSON";
  case tool_format::cTagWithTagged:
    return pOs << "TAG_WITH_TAGGED";
  default:
    return pOs << "UNKNOWN";
  }
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

struct tool_format_analysis {
  tool_format vMode = tool_format::cNone;

  std::string vSectionStart; // e.g., "<tool_call>", "[TOOL_CALLS]", ""
  std::string vSectionEnd;   // e.g., "</tool_call>", ""
  std::string vPerCallStart; // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
  std::string vPerCallEnd;   // e.g., "<|tool_call_end|>", ""

  bool vFunNameIsKey = false; // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
  bool vToolsArrayWrapped = false;    // Tool calls wrapped in JSON array [...]
  bool vOpenaiWrapperTrigger = false; // model emits the OpenAI function wrapper, trigger on it

  std::string vFunctionField = "function";
  std::string vNameField = "name";
  std::string vArgsField = "arguments";
  std::string vIdField;
  std::string vGenIdField;
  std::vector<std::string> lParameterOrder;
};

struct tool_function_analysis {
  std::string vNamePrefix;    // e.g., "<function=", "\"name\": \"", "functions."
  std::string vNameSuffix;    // e.g., ">", "\"", ":0"
  std::string vArgsSeparator; // e.g., "<tool_sep>" (marker between function name and arguments)
  std::string vClose;         // e.g., "</function>", "" (for tag-based)
};

struct tool_arguments_analysis {
  std::string vStart;                       // e.g., "<|tool_call_argument_begin|>", "<args>"
  std::string vEnd;                         // e.g., "<|tool_call_argument_end|>", "</args>"
  std::string vNamePrefix;                  // e.g., "<param=", "<arg_key>", "\""
  std::string vNameSuffix;                  // e.g., ">", "</arg_key>", "\":"
  std::string vValuePrefix;                 // e.g., "", "<arg_value>", ""
  std::string vValueSuffix;                 // e.g., "</param>", "</arg_value>", ""
  std::string vSeparator;                   // e.g., "", "\n", ","
  bool vTolerateIntertagWhitespace = false; // Laguna: accept optional whitespace between arg tags
};

struct tool_id_analysis {
  call_id_position vPos = call_id_position::cNone;

  std::string vPrefix; // e.g., "[CALL_ID]" (marker before call ID value)
  std::string vSuffix; // e.g., "" (marker after call ID value, before next section)
};

// ============================================================================
// Parser build context (shared interface for build_parser methods)
// ============================================================================

struct analyze_content;
struct analyze_reasoning;

struct parser_build_context {
  common_chat_peg_builder &vP;
  const vGenerationParams &vInputs;
  common_peg_parser vReasoningParser;
  bool vExtractingReasoning = false;
  const analyze_reasoning *vReasoning = nullptr;
  const analyze_content *vContent = nullptr;

  parser_build_context(common_chat_peg_builder &pP, const vGenerationParams &pInputs);
};

// ============================================================================
// Base class for analyzers with parser building
// ============================================================================

struct analyze_base {
  virtual ~analyze_base() = default;
  virtual common_peg_parser mBuildParser(parser_build_context &pCtx) const = 0;

protected:
  const common_chat_template *vTmpl = nullptr;

  analyze_base() = default;
  explicit analyze_base(const common_chat_template &pTmpl) : vTmpl(&pTmpl) {}
};

// ============================================================================
// Reasoning analyzer
// ============================================================================

struct analyze_reasoning : analyze_base {
  reasoning_mode vMode = reasoning_mode::cNone;

  std::string vStart; // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
  std::string vEnd;   // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

  analyze_reasoning() = default;
  analyze_reasoning(const common_chat_template &pTmpl, bool pSupportsTools);
  analyze_reasoning(std::string pStart, std::string pEnd) : vStart(std::move(pStart)), vEnd(std::move(pEnd)) {}

  common_peg_parser mBuildParser(parser_build_context &pCtx) const override;

private:
  // Look for reasoning markers in rendered content
  void mCompareReasoningPresence();

  // Compare generation prompt with enable_thinking=true vs false
  void mCompareThinkingEnabled();

  // Check if reasoning is always possible or only in tool calls
  void mCompareReasoningScope();
};

// ============================================================================
// Content analyzer
// ============================================================================

struct analyze_content : analyze_base {
  content_mode vMode = content_mode::cPlain;

  std::string vStart; // e.g., "<response>", ">>>all\n", ""
  std::string vEnd;   // e.g., "</response>", ""

  bool vRequiresNonnullContent = false;

  analyze_content() = default;
  analyze_content(const common_chat_template &pTmpl, const analyze_reasoning &pReasoning);

  common_peg_parser mBuildParser(parser_build_context &pCtx) const override;

  bool mIsAlwaysWrapped() const;
  common_peg_parser mBuildOptionalWrapped(parser_build_context &pCtx) const;
};

// ============================================================================
// Tool analyzer
// ============================================================================

struct analyze_tools : analyze_base {
  tool_format_analysis vFormat;
  tool_function_analysis vFunction;
  tool_arguments_analysis vArguments;
  tool_id_analysis vCallId;

  analyze_tools() = default;
  analyze_tools(const common_chat_template &pTmpl, const jinja::caps &pCaps, const analyze_reasoning &pReasoning);

  common_peg_parser mBuildParser(parser_build_context &pCtx) const override;

private:
  // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
  void mAnalyzeToolCalls(const analyze_reasoning &pReasoning, bool pSupportsParallelToolCalls);

  // Analyze format based on position of function and argument name in needle
  void mAnalyzeToolCallFormat(const std::string &pHaystack, const std::string &pFunNameNeedle,
                              const std::string &pArgNameNeedle, const analyze_reasoning &pReasoning,
                              bool pSupportsParallelToolCalls);

  // Analyze specifics of JSON native format (entire tool call is a JSON object)
  void mAnalyzeToolCallFormatJsonNative(const std::string &pCleanHaystack, const std::string &pFunNameNeedle,
                                        const std::string &pArgNameNeedle);

  // Check if parallel calls in JSON native format array wrapped or tag wrapped
  void mAnalyzeJsonNativeParallelCalls();

  // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
  void mAnalyzeToolCallFormatNonJson(const std::string &pCleanHaystack, const std::string &pFunNameNeedle);

  // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
  void mCheckPerCallMarkers();

  // Extract function name markers
  void mExtractFunctionMarkers();

  // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
  void mAnalyzeArguments();

  // Extract argument name markers
  void mExtractArgumentNameMarkers();

  // Extract argument value markers
  void mExtractArgumentValueMarkers();

  // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
  void mExtractArgumentSeparator();

  // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
  void mExtractArgsMarkers();

  // Extract call ID markers, if present
  void mExtractCallIdMarkers();

  // Per-format tool parser builders
  common_peg_parser mBuildToolParserJsonNative(parser_build_context &pCtx) const;
  common_peg_parser mBuildToolParserTagJson(parser_build_context &pCtx) const;
  common_peg_parser mBuildToolParserTagTagged(parser_build_context &pCtx) const;

  // Shared helper: builds func_parser from open+call_id+args, handling atomic wrapping and close.
  // atomic_peek: if present, used as the peek expression in the third atomicity branch.
  common_peg_parser mBuildFuncParser(common_chat_peg_builder &pP, const std::string &pName,
                                     const common_peg_parser &pCallIdSection, bool pHaveCallId,
                                     const common_peg_parser &pArgs,
                                     std::optional<common_peg_parser> pAtomicPeek) const;
};

// ============================================================================
// Main autoparser class
// ============================================================================

struct autoparser {
  jinja::caps vJinjaCaps;
  std::string vUserStart;
  std::string vAssistantStart;
  analyze_reasoning vReasoning;
  analyze_content vContent;
  analyze_tools vTools;
  bool vAnalysisComplete = false;

  // Preserved tokens for tokenizer (union of all non-empty markers)
  std::vector<std::string> lPreservedTokens;
  std::vector<std::string> lAdditionalStops; // literal stop strings (e.g. Laguna </assistant>) caught however tokenized

  autoparser() = default;

  // Find the starting marker for the user message and assistant message
  std::string mDetectUserStartMarker(const common_chat_template &pTmpl);
  std::string mDetectAssistantStartMarker(const common_chat_template &pTmpl);

  // Run full differential analysis on a template
  void mAnalyzeTemplate(const common_chat_template &pTmpl);

  // Build the PEG parser for this template
  common_peg_arena mBuildParser(const vGenerationParams &pInputs, const std::string &pGenerationPrompt) const;

private:
  // Collect tokens from entire analysis to preserve
  void mCollectPreservedTokens();
};

// ============================================================================
// Parser generator
// ============================================================================

class peg_generator {
public:
  static common_chat_params mGenerateParser(const common_chat_template &pTmpl, const struct vGenerationParams &pInputs);

  static common_chat_params mGenerateParser(const common_chat_template &pTmpl, const struct vGenerationParams &pInputs,
                                            const autoparser &pAutoparser);
};

} // namespace autoparser

enum segment_type { cText, cMarker };

inline std::ostream &operator<<(std::ostream &pOs, const segment_type &pType) {
  switch (pType) {
  case segment_type::cText:
    return pOs << "TEXT";
  case segment_type::cMarker:
    return pOs << "MARKER";
  default:
    return pOs << "UNKNOWN";
  }
}

struct segment {
  segment_type vType;
  std::string vValue;

  segment(segment_type pType, std::string pValue) : vType(pType), vValue(std::move(pValue)) {}

  bool operator==(const segment &pOther) const { return vType == pOther.vType && vValue == pOther.vValue; }

  bool operator!=(const segment &pOther) const { return !(*this == pOther); }
};
