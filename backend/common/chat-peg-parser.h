#pragma once

#include "chat.h"
#include "peg-parser.h"

#include <map>
#include <optional>
#include <vector>

class common_chat_peg_mapper {
public:
  common_chat_msg &vResult;

  common_chat_peg_mapper(common_chat_msg &pMsg) : vResult(pMsg) {}

  virtual ~common_chat_peg_mapper() = default;

  virtual void mFromAst(const common_peg_ast_arena &pArena, const common_peg_parse_result &pResult);
  virtual void mMap(const common_peg_ast_node &pNode);

protected:
  virtual std::string mNormalizeContainerValue(const std::string &pInput);

private:
  // Tool call handling state
  std::optional<common_chat_tool_call> vPendingToolCall; // Tool call waiting for name
  common_chat_tool_call *vCurrentTool = nullptr;
  int vArgCount = 0;
  bool vClosingQuotePending = false;
  std::string vArgsBuffer; // Buffer to delay arguments until tool name is known

  // Returns a reference to the active argument destination string.
  // Before tool_name is known, writes go to vArgsBuffer; after, to vCurrentTool->vArguments.
  std::string &mArgsTarget();
};

class common_chat_peg_minimax_m3_mapper : public common_chat_peg_mapper {
public:
  static constexpr const char *cToolArgObject = "tool-arg-object";
  static constexpr const char *cToolArgArray = "tool-arg-array";
  static constexpr const char *cToolArgItem = "tool-arg-item";

  common_chat_peg_minimax_m3_mapper(common_chat_msg &pMsg) : common_chat_peg_mapper(pMsg) {}
  virtual void mFromAst(const common_peg_ast_arena &pArena, const common_peg_parse_result &pResult);

private:
  void mVisit(const common_peg_ast_arena &pArena, common_peg_ast_id pId);
};

struct content_structure;
struct tool_call_structure;

constexpr size_t cCommonChatMaxPermute = 6;

class common_chat_peg_builder : public common_peg_parser_builder {
public:
  // Tag constants (from former common_chat_peg_base_builder)
  static constexpr const char *cReasoningBlock = "reasoning-block";
  static constexpr const char *cReasoning = "reasoning";
  static constexpr const char *cContent = "content";

  // Tag constants
  static constexpr const char *cTool = "tool";
  static constexpr const char *cToolOpen = "tool-open";
  static constexpr const char *cToolClose = "tool-close";
  static constexpr const char *cToolId = "tool-id";
  static constexpr const char *cToolName = "tool-name";
  static constexpr const char *cToolArgs = "tool-args";
  static constexpr const char *cToolArg = "tool-arg";
  static constexpr const char *cToolArgOpen = "tool-arg-open";
  static constexpr const char *cToolArgClose = "tool-arg-close";
  static constexpr const char *cToolArgName = "tool-arg-name";
  static constexpr const char *cToolArgValue = "tool-arg-value";
  static constexpr const char *cToolArgStringValue = "tool-arg-string-value"; // For schema-declared string types

  // Low-level tag methods (from former common_chat_peg_base_builder)
  common_peg_parser mReasoningBlock(const common_peg_parser &pP) { return mTag(cReasoningBlock, pP); }

  common_peg_parser mReasoning(const common_peg_parser &pP) { return mTag(cReasoning, pP); }

  common_peg_parser mContent(const common_peg_parser &pP) { return mTag(cContent, pP); }

  common_peg_parser mTagWithSafeContent(const std::string &pTagName, const std::string &pMarker,
                                        const common_peg_parser &pP);

  // Low-level tag methods
  common_peg_parser mTool(const common_peg_parser &pP) { return mTag(cTool, pP); }
  common_peg_parser mToolOpen(const common_peg_parser &pP) { return mAtomic(mTag(cToolOpen, pP)); }
  common_peg_parser mToolClose(const common_peg_parser &pP) { return mAtomic(mTag(cToolClose, pP)); }
  common_peg_parser mToolId(const common_peg_parser &pP) { return mAtomic(mTag(cToolId, pP)); }
  common_peg_parser mToolName(const common_peg_parser &pP) { return mAtomic(mTag(cToolName, pP)); }
  common_peg_parser mToolArgs(const common_peg_parser &pP) { return mTag(cToolArgs, pP); }
  common_peg_parser mToolArg(const common_peg_parser &pP) { return mTag(cToolArg, pP); }
  common_peg_parser mToolArgOpen(const common_peg_parser &pP) { return mAtomic(mTag(cToolArgOpen, pP)); }
  common_peg_parser mToolArgClose(const common_peg_parser &pP) { return mAtomic(mTag(cToolArgClose, pP)); }
  common_peg_parser mToolArgName(const common_peg_parser &pP) { return mAtomic(mTag(cToolArgName, pP)); }
  common_peg_parser mToolArgValue(const common_peg_parser &pP) { return mTag(cToolArgValue, pP); }

  // Use for schema-declared string types - won't be treated as potential JSON container
  common_peg_parser mToolArgStringValue(const common_peg_parser &pP) { return mTag(cToolArgStringValue, pP); }
  common_peg_parser mToolArgJsonValue(const common_peg_parser &pP) { return mTag(cToolArgValue, pP); }

  // Matches every parser exactly once, in any order.
  common_peg_parser mPermute(const std::string &pRulePrefix, const std::vector<common_peg_parser> &pParsers);

  // Return a parser that parses the prefix of a string, up to a given delimiter.
  common_peg_parser mPrefix(const std::string &pS, const std::string &pDelimiter = {});

  // Return a parser that parses all elements of tag, but leading and trailing spaces are optional
  common_peg_parser mOptspace(const std::string &pTag);

  // Helper for building standard JSON tool calls
  // Used by the automatic parser generator and manual parsers
  // name_key/args_key: JSON key names for function name and arguments
  //   Empty or "name"/"arguments" will accept both common variations
  //   Supports dot notation for nested objects (e.g., "function.name")
  // array_wrapped: if true, tool calls are wrapped in JSON array [...]
  // function_is_key: if true, function name is the JSON key (e.g., {"func_name": {...}})
  // call_id_key: JSON key for string call ID (e.g., "id")
  // gen_call_id_key: JSON key for generated integer call ID (e.g., "tool_call_id")
  // parameters_order: order in which JSON fields should be parsed
  common_peg_parser mStandardJsonTools(const std::string &pSectionStart, const std::string &pSectionEnd,
                                       const nlohmann::ordered_json &pTools, bool pParallelToolCalls,
                                       bool pForceToolCalls, const std::string &pNameKey = "",
                                       const std::string &pArgsKey = "", bool pArrayWrapped = false,
                                       bool pFunctionIsKey = false, const std::string &pCallIdKey = "",
                                       const std::string &pGenCallIdKey = "",
                                       const std::vector<std::string> &pParametersOrder = {},
                                       bool pAcceptOpenaiWrapper = false);

  // Helper for Python-style function call format: name(arg1="value1", arg2=123)
  // Used by LFM2 and similar templates
  common_peg_parser mPythonStyleToolCalls(const nlohmann::ordered_json &pTools, bool pParallelToolCalls,
                                          bool pAllowJsonLiterals);

private:
  // Python values plus JSON true/false/null.
  common_peg_parser mPythonOrJsonValue();

  // Implementation helpers for standard_json_tools — one per JSON tool call layout mode
  common_peg_parser mBuildJsonToolsFunctionIsKey(const nlohmann::ordered_json &pTools, const std::string &pArgsKey,
                                                 const std::string &pEffectiveArgsKey, const std::string &pCallIdKey,
                                                 const std::string &pGenCallIdKey);

  common_peg_parser mBuildJsonToolsNestedKeys(const nlohmann::ordered_json &pTools,
                                              const std::string &pEffectiveNameKey,
                                              const std::string &pEffectiveArgsKey, const std::string &pCallIdKey,
                                              const std::string &pGenCallIdKey);

  common_peg_parser mBuildJsonToolsFlatKeys(const nlohmann::ordered_json &pTools, const std::string &pEffectiveNameKey,
                                            const std::string &pEffectiveArgsKey, const std::string &pCallIdKey,
                                            const std::string &pGenCallIdKey,
                                            const std::vector<std::string> &pParametersOrder,
                                            bool pAcceptOpenaiWrapper);
};

inline common_peg_arena
fBuildChatPegParser(const std::function<common_peg_parser(common_chat_peg_builder &pBuilder)> &pFn) {
  common_chat_peg_builder vBuilder;
  vBuilder.mSetRoot(pFn(vBuilder));
  return vBuilder.mBuild();
}

class tag_based_peg_mapper {
public:
  std::map<std::string, std::string> dTags;

  void mFromAst(const common_peg_ast_arena &pArena, const common_peg_parse_result &pResult);
};

struct tagged_parse_result {
  common_peg_parse_result vResult;
  std::map<std::string, std::string> dTags;
};

struct tagged_peg_parser {
  common_peg_arena vArena;
  common_peg_parse_flags vFlags = cCommonPegParseFlagNone;

  tagged_peg_parser &mWithDebug() {
    vFlags |= cCommonPegParseFlagDebug;
    return *this;
  }

  tagged_peg_parser &mWithoutDebug() {
    vFlags = vFlags & ~cCommonPegParseFlagDebug;
    return *this;
  }

  tagged_parse_result mParseAndExtract(const std::string &pInput,
                                       common_peg_parse_flags pExtraFlags = cCommonPegParseFlagNone) const;
  tagged_parse_result mParseAnywhereAndExtract(const std::string &pInput) const;
};

tagged_peg_parser
fBuildTaggedPegParser(const std::function<common_peg_parser(common_peg_parser_builder &pBuilder)> &pFn);
