#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <set>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

struct common_grammar_builder;

class common_peg_parser_builder;

using common_peg_parser_id = size_t;
constexpr common_peg_parser_id cCommonPegInvalidParserId = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id cCommonPegInvalidAstId = static_cast<common_peg_ast_id>(-1);

// Lightweight wrapper around common_peg_parser_id for convenience
class common_peg_parser {
  common_peg_parser_id vId;
  common_peg_parser_builder &vBuilder;

public:
  common_peg_parser(const common_peg_parser &pOther) : vId(pOther.vId), vBuilder(pOther.vBuilder) {}
  common_peg_parser(common_peg_parser_id pId, common_peg_parser_builder &pBuilder) : vId(pId), vBuilder(pBuilder) {}

  common_peg_parser &operator=(const common_peg_parser &pOther);
  common_peg_parser &operator+=(const common_peg_parser &pOther);
  common_peg_parser &operator|=(const common_peg_parser &pOther);

  operator common_peg_parser_id() const { return vId; }
  common_peg_parser_id mId() const { return vId; }

  common_peg_parser_builder &mBuilder() const { return vBuilder; }

  // Creates a sequence
  common_peg_parser operator+(const common_peg_parser &pOther) const;

  // Creates a sequence separated by spaces.
  common_peg_parser operator<<(const common_peg_parser &pOther) const;

  // Creates a choice
  common_peg_parser operator|(const common_peg_parser &pOther) const;

  common_peg_parser operator+(const char *pStr) const;
  common_peg_parser operator+(const std::string &pStr) const;
  common_peg_parser operator<<(const char *pStr) const;
  common_peg_parser operator<<(const std::string &pStr) const;
  common_peg_parser operator|(const char *pStr) const;
  common_peg_parser operator|(const std::string &pStr) const;
};

common_peg_parser operator+(const char *pStr, const common_peg_parser &pP);
common_peg_parser operator+(const std::string &pStr, const common_peg_parser &pP);
common_peg_parser operator<<(const char *pStr, const common_peg_parser &pP);
common_peg_parser operator<<(const std::string &pStr, const common_peg_parser &pP);
common_peg_parser operator|(const char *pStr, const common_peg_parser &pP);
common_peg_parser operator|(const std::string &pStr, const common_peg_parser &pP);

enum common_peg_parse_result_type {
  cCommonPegParseResultFail = 0,
  cCommonPegParseResultSuccess = 1,
  cCommonPegParseResultNeedMoreInput = 2,
};

const char *fCommonPegParseResultTypeName(common_peg_parse_result_type pType);

struct common_peg_ast_node {
  common_peg_ast_id vId;
  std::string vRule;
  std::string vTag;
  size_t vStart;
  size_t vEnd;
  std::string_view vText;
  std::vector<common_peg_ast_id> lChildren;

  bool vIsPartial = false;
};

struct common_peg_parse_result;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node &pNode)>;

class common_peg_ast_arena {
  std::vector<common_peg_ast_node> lNodes;

public:
  common_peg_ast_id mAddNode(const std::string &pRule, const std::string &pTag, size_t pStart, size_t pEnd,
                             std::string_view pText, std::vector<common_peg_ast_id> pChildren,
                             bool pIsPartial = false) {
    common_peg_ast_id vId = lNodes.size();
    lNodes.push_back({vId, pRule, pTag, pStart, pEnd, pText, std::move(pChildren), pIsPartial});
    return vId;
  }

  const common_peg_ast_node &mGet(common_peg_ast_id pId) const { return lNodes.at(pId); }

  common_peg_ast_id mFindByTag(const common_peg_ast_node &pParent, const std::string &pTag, int pMaxDepth = 3) const;
  common_peg_ast_id mFindByRule(const common_peg_ast_node &pParent, const std::string &pTag, int pMaxDepth = 3) const;

  size_t size() const { return lNodes.size(); }

  void mClear() { lNodes.clear(); }

  void mVisit(common_peg_ast_id pId, const common_peg_ast_visitor &pVisitor) const;
  void mVisit(const common_peg_parse_result &pResult, const common_peg_ast_visitor &pVisitor) const;

  std::string mDump();
};

struct common_peg_parse_result {
  common_peg_parse_result_type vType = cCommonPegParseResultFail;
  size_t vStart = 0;
  size_t vEnd = 0;

  std::vector<common_peg_ast_id> lNodes;

  common_peg_parse_result() = default;

  common_peg_parse_result(common_peg_parse_result_type pType, size_t pStart)
      : vType(pType), vStart(pStart), vEnd(pStart) {}

  common_peg_parse_result(common_peg_parse_result_type pType, size_t pStart, size_t pEnd)
      : vType(pType), vStart(pStart), vEnd(pEnd) {}

  common_peg_parse_result(common_peg_parse_result_type pType, size_t pStart, size_t pEnd,
                          std::vector<common_peg_ast_id> pNodes)
      : vType(pType), vStart(pStart), vEnd(pEnd), lNodes(std::move(pNodes)) {}

  bool mFail() const { return vType == cCommonPegParseResultFail; }
  bool mNeedMoreInput() const { return vType == cCommonPegParseResultNeedMoreInput; }
  bool mSuccess() const { return vType == cCommonPegParseResultSuccess; }
};

enum common_peg_parse_flags {
  cCommonPegParseFlagNone = 0,
  cCommonPegParseFlagLenient = 1 << 0,
  cCommonPegParseFlagDebug = 1 << 1,
};

inline common_peg_parse_flags operator|(common_peg_parse_flags pA, common_peg_parse_flags pB) {
  return static_cast<common_peg_parse_flags>(int(pA) | int(pB));
}

inline common_peg_parse_flags &operator|=(common_peg_parse_flags &pA, common_peg_parse_flags pB) {
  return pA = pA | pB;
}

inline common_peg_parse_flags operator&(common_peg_parse_flags pA, common_peg_parse_flags pB) {
  return static_cast<common_peg_parse_flags>(int(pA) & int(pB));
}

inline common_peg_parse_flags operator~(common_peg_parse_flags pA) {
  return static_cast<common_peg_parse_flags>(~int(pA));
}

struct common_peg_parse_context {
  std::string vInput;
  common_peg_parse_flags vFlags;
  common_peg_ast_arena vAst;

  int vParseDepth;

  common_peg_parse_context(common_peg_parse_flags pFlags = cCommonPegParseFlagNone) : vFlags(pFlags), vParseDepth(0) {}

  common_peg_parse_context(const std::string &pInput, common_peg_parse_flags pFlags = cCommonPegParseFlagNone)
      : vInput(pInput), vFlags(pFlags), vParseDepth(0) {}

  bool mIsLenient() const { return vFlags & cCommonPegParseFlagLenient; }
  bool mIsDebug() const { return vFlags & cCommonPegParseFlagDebug; }
};

class common_peg_arena;

// Parser variants
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
  std::string vLiteral;
};

struct common_peg_sequence_parser {
  std::vector<common_peg_parser_id> lChildren;
};

struct common_peg_choice_parser {
  std::vector<common_peg_parser_id> lChildren;
};

struct common_peg_repetition_parser {
  common_peg_parser_id vChild;
  int vMinCount;
  int vMaxCount; // -1 for unbounded
};

struct common_peg_and_parser {
  common_peg_parser_id vChild;
};

struct common_peg_not_parser {
  common_peg_parser_id vChild;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
  struct char_range {
    uint32_t vStart;
    uint32_t vEnd;
    bool mContains(uint32_t pCodepoint) const { return pCodepoint >= vStart && pCodepoint <= vEnd; }
  };

  std::string vPattern;
  std::vector<char_range> lRanges;
  bool vNegated;
  int vMinCount;
  int vMaxCount; // -1 for unbounded
};

struct common_peg_string_parser {
  char vDelimiter;
};

struct common_peg_until_parser {
  std::vector<std::string> lDelimiters;
};

struct common_peg_schema_parser {
  common_peg_parser_id vChild;
  std::string vName;
  std::shared_ptr<nlohmann::ordered_json> vSchema;

  // Indicates if the GBNF should accept a raw string that matches the schema.
  bool vRaw;
};

struct common_peg_rule_parser {
  std::string vName;
  common_peg_parser_id vChild;
  bool vTrigger;
};

struct common_peg_ref_parser {
  std::string vName;
};

struct common_peg_atomic_parser {
  common_peg_parser_id vChild;
};

struct common_peg_tag_parser {
  common_peg_parser_id vChild;
  std::string vTag;
};

struct common_peg_gbnf_parser {
  common_peg_parser_id vChild;
  std::string vGrammar;
};

struct common_peg_ac_parser {
  common_peg_parser_id vChild;
  std::vector<std::string> lDelimiters;
};

// Variant holding all parser types
using common_peg_parser_variant =
    std::variant<common_peg_epsilon_parser, common_peg_start_parser, common_peg_end_parser, common_peg_literal_parser,
                 common_peg_sequence_parser, common_peg_choice_parser, common_peg_repetition_parser,
                 common_peg_and_parser, common_peg_not_parser, common_peg_any_parser, common_peg_space_parser,
                 common_peg_chars_parser, common_peg_string_parser, common_peg_until_parser, common_peg_schema_parser,
                 common_peg_rule_parser, common_peg_ref_parser, common_peg_atomic_parser, common_peg_tag_parser,
                 common_peg_gbnf_parser, common_peg_ac_parser>;

class common_peg_arena {
  std::vector<common_peg_parser_variant> lParsers;
  std::unordered_map<std::string, common_peg_parser_id> dRules;
  common_peg_parser_id vRoot = cCommonPegInvalidParserId;

public:
  const common_peg_parser_variant &mGet(common_peg_parser_id pId) const { return lParsers.at(pId); }
  common_peg_parser_variant &mGet(common_peg_parser_id pId) { return lParsers.at(pId); }

  size_t size() const { return lParsers.size(); }
  bool empty() const { return lParsers.empty(); }

  common_peg_parser_id mGetRule(const std::string &pName) const;
  bool mHasRule(const std::string &pName) const { return dRules.find(pName) != dRules.end(); }

  common_peg_parser_id mRoot() const { return vRoot; }
  void mSetRoot(common_peg_parser_id pId) { vRoot = pId; }

  common_peg_parse_result mParse(common_peg_parse_context &pCtx, size_t pStart = 0) const;
  common_peg_parse_result mParse(common_peg_parser_id pId, common_peg_parse_context &pCtx, size_t pStart) const;

  void mResolveRefs();

  void mBuildGrammar(const common_grammar_builder &pBuilder, bool pLazy = false) const;

  std::string mDump(common_peg_parser_id pId) const;

  nlohmann::json to_json() const;
  static common_peg_arena from_json(const nlohmann::json &pJ);

  std::string mSave() const;
  void mLoad(const std::string &pData);

  friend class common_peg_parser_builder;

private:
  std::string mDumpImpl(common_peg_parser_id pId, std::set<common_peg_parser_id> &pVisited) const;

  common_peg_parser_id mAddParser(common_peg_parser_variant pParser);
  void mAddRule(const std::string &pName, common_peg_parser_id pId);

  common_peg_parser_id mResolveRef(common_peg_parser_id pId);
};

class common_peg_parser_builder {
  common_peg_arena vArena;

  common_peg_parser mWrap(common_peg_parser_id pId) { return common_peg_parser(pId, *this); }
  common_peg_parser mAdd(const common_peg_parser_variant &pP) { return mWrap(vArena.mAddParser(pP)); }

public:
  common_peg_parser_builder();

  // Match nothing, always succeed.
  //   S -> ε
  common_peg_parser mEps() { return mAdd(common_peg_epsilon_parser{}); }

  // Matches the start of the input.
  //   S -> ^
  common_peg_parser mStart() { return mAdd(common_peg_start_parser{}); }

  // Matches the end of the input.
  //   S -> $
  common_peg_parser end() { return mAdd(common_peg_end_parser{}); }

  // Matches an exact literal string.
  //   S -> "hello"
  common_peg_parser mLiteral(const std::string &pLiteral) { return mAdd(common_peg_literal_parser{pLiteral}); }

  // Matches a sequence of parsers in order, all must succeed.
  //   S -> A B C
  common_peg_parser mSequence() { return mAdd(common_peg_sequence_parser{}); }
  common_peg_parser mSequence(const std::vector<common_peg_parser_id> &pParsers);
  common_peg_parser mSequence(const std::vector<common_peg_parser> &pParsers);
  common_peg_parser mSequence(std::initializer_list<common_peg_parser> pParsers);

  // Matches the first parser that succeeds from a list of alternatives.
  //   S -> A | B | C
  common_peg_parser mChoice() { return mAdd(common_peg_choice_parser{}); }
  common_peg_parser mChoice(const std::vector<common_peg_parser_id> &pParsers);
  common_peg_parser mChoice(const std::vector<common_peg_parser> &pParsers);
  common_peg_parser mChoice(std::initializer_list<common_peg_parser> pParsers);

  // Matches one or more repetitions of a parser.
  //   S -> A+
  common_peg_parser mOneOrMore(const common_peg_parser &pP) { return mRepeat(pP, 1, -1); }

  // Matches zero or more repetitions of a parser, always succeeds.
  //   S -> A*
  common_peg_parser mZeroOrMore(const common_peg_parser &pP) { return mRepeat(pP, 0, -1); }

  // Matches zero or one occurrence of a parser, always succeeds.
  //   S -> A?
  common_peg_parser mOptional(const common_peg_parser &pP) { return mRepeat(pP, 0, 1); }

  // Positive lookahead: succeeds if child parser succeeds, consumes no input.
  //   S -> &A
  common_peg_parser mPeek(const common_peg_parser &pP) { return mAdd(common_peg_and_parser{pP}); }

  // Negative lookahead: succeeds if child parser fails, consumes no input.
  //   S -> !A
  common_peg_parser mNegate(const common_peg_parser &pP) { return mAdd(common_peg_not_parser{pP}); }

  // Matches any single character.
  //   S -> .
  common_peg_parser mAny() { return mAdd(common_peg_any_parser{}); }

  // Matches between min and max repetitions of characters from a character class.
  //   S -> [a-z]{m,n}
  //
  // Use -1 for max to represent unbounded repetition (equivalent to {m,})
  common_peg_parser mChars(const std::string &pClasses, int pMin = 1, int pMax = -1);

  // Creates a lightweight reference to a named rule (resolved during mBuild()).
  // Use this for forward references in recursive grammars.
  //   expr_ref -> expr
  common_peg_parser mRef(const std::string &pName) { return mAdd(common_peg_ref_parser{pName}); }

  // Matches zero or more whitespace characters (space, tab, newline).
  //   S -> [ \t\n]*
  common_peg_parser mSpace() { return mAdd(common_peg_space_parser{}); }

  // Matches all characters until a delimiter is found (delimiter not consumed).
  //   S -> (!delim .)*
  common_peg_parser mUntil(const std::string &pDelimiter) { return mAdd(common_peg_until_parser{{pDelimiter}}); }

  // Matches all characters until one of the delimiters in the list is found (delimiter not consumed).
  //   S -> (!delim .)*
  common_peg_parser mUntilOneOf(const std::vector<std::string> &pDelimiters) {
    return mAdd(common_peg_until_parser{pDelimiters});
  }

  // Matches everything
  //   S -> .*
  common_peg_parser mRest() { return mUntilOneOf({}); }

  // Matches between min and max repetitions of a parser (inclusive).
  //   S -> A{m,n}
  // Use -1 for max to represent unbounded repetition (equivalent to {m,})
  common_peg_parser mRepeat(const common_peg_parser &pP, int pMin, int pMax) {
    return mAdd(common_peg_repetition_parser{pP, pMin, pMax});
  }

  // Matches exactly n repetitions of a parser.
  //   S -> A{n}
  common_peg_parser mRepeat(const common_peg_parser &pP, int pN) { return mRepeat(pP, pN, pN); }

  // Matches a double-quoted string: '"' content '"' space
  common_peg_parser mDoubleQuotedString();

  // Matches a single-quoted string: "'" content "'" space
  common_peg_parser mSingleQuotedString();

  // Matches a string that accepts both double-quoted and single-quoted styles.
  common_peg_parser mQuotedString();

  // Matches string content without the surrounding delimiter.
  common_peg_parser mStringContent(char pDelimiter);

  // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
  //   value -> object | array | string | number | true | false | null
  common_peg_parser mJson();
  common_peg_parser mJsonObject();
  common_peg_parser mJsonString();
  common_peg_parser mJsonArray();
  common_peg_parser mJsonNumber();
  common_peg_parser mJsonBool();
  common_peg_parser mJsonNull();

  // Matches a JSON object member with a key and associated parser as the
  // value.
  common_peg_parser mJsonMember(const std::string &pKey, const common_peg_parser &pP);

  // Creates a complete Python format parser supporting dicts, arrays, strings, numbers, booleans, and None.
  // Differs from JSON: uses True/False/None, accepts both single and double-quoted strings.
  //   value -> dict | array | string | number | True | False | None
  common_peg_parser mPythonValue();
  common_peg_parser mPythonDict();
  common_peg_parser mPythonString();
  common_peg_parser mPythonArray();
  common_peg_parser mPythonNumber();
  common_peg_parser mPythonBool();
  common_peg_parser mPythonNull();

  // A marker, i.e. text delimited by a pair of <> or []
  common_peg_parser mMarker();

  // Wraps a parser with JSON schema metadata for grammar generation.
  // Used internally to convert JSON schemas to GBNF grammar rules.
  common_peg_parser mSchema(const common_peg_parser &pP, const std::string &pName,
                            const nlohmann::ordered_json &pSchema, bool pRaw = false);

  // Creates a named rule, stores it in the grammar, and returns a ref.
  // If pTrigger=true, marks this rule as an entry point for lazy grammar generation.
  //   auto json = p.mRule("json", json_obj | json_arr | ...)
  common_peg_parser mRule(const std::string &pName, const common_peg_parser &pP, bool pTrigger = false);

  // Creates a named rule using a builder function, and returns a ref.
  // If pTrigger=true, marks this rule as an entry point for lazy grammar generation.
  //   auto json = p.mRule("json", [&]() { return mJsonObject() | mJsonArray() | ... })
  common_peg_parser mRule(const std::string &pName, const std::function<common_peg_parser()> &pBuilder,
                          bool pTrigger = false);

  // Creates a trigger rule. When generating a lazy grammar from the parser,
  // only trigger rules and descendents are emitted.
  common_peg_parser mTriggerRule(const std::string &pName, const common_peg_parser &pP) {
    return mRule(pName, pP, true);
  }
  common_peg_parser mTriggerRule(const std::string &pName, const std::function<common_peg_parser()> &pBuilder) {
    return mRule(pName, pBuilder, true);
  }

  // Creates an atomic parser. Atomic parsers do not create an AST node if
  // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
  // intended for situations where partial output is undesirable.
  common_peg_parser mAtomic(const common_peg_parser &pP) { return mAdd(common_peg_atomic_parser{pP}); }

  // Tags create nodes in the generated AST for semantic purposes.
  // Unlike rules, you can tag multiple nodes with the same tag.
  common_peg_parser mTag(const std::string &pTag, const common_peg_parser &pP) {
    return mAdd(common_peg_tag_parser{pP.mId(), pTag});
  }

  // Wraps a child parser but emits a custom GBNF grammar string instead of
  // the child's grammar. Parsing delegates entirely to the child.
  common_peg_parser mGbnf(const common_peg_parser &pP, const std::string &pGrammar) {
    return mAdd(common_peg_gbnf_parser{pP, pGrammar});
  }

  // Wraps a child parser but emits a GBNF grammar built from the Aho-Corasick
  // automaton of `pDelimiters`, matching everything up to and including the
  // first delimiter. Parsing delegates entirely to the child, which is
  // responsible for consuming the delimiter (e.g. mUntil(D) + mLiteral(D)).
  common_peg_parser mAc(const common_peg_parser &pP, const std::vector<std::string> &pDelimiters);
  common_peg_parser mAc(const common_peg_parser &pP, const std::string &pDelimiter) {
    return mAc(pP, std::vector<std::string>{pDelimiter});
  }

  void mSetRoot(const common_peg_parser &pP);

  common_peg_arena mBuild();
};

// Helper function for building parsers
common_peg_arena fBuildPegParser(const std::function<common_peg_parser(common_peg_parser_builder &pBuilder)> &pFn);
