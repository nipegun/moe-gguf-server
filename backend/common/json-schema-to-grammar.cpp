#include "json-schema-to-grammar.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static std::string fBuildRepetition(const std::string &pItemRule, int pMinItems, int pMaxItems,
                                    const std::string &pSeparatorRule = "") {
  auto vHasMax = pMaxItems != std::numeric_limits<int>::max();

  if (pMaxItems == 0) {
    return "";
  }
  if (pMinItems == 0 && pMaxItems == 1) {
    return pItemRule + "?";
  }

  if (pSeparatorRule.empty()) {
    if (pMinItems == 1 && !vHasMax) {
      return pItemRule + "+";
    }
    if (pMinItems == 0 && !vHasMax) {
      return pItemRule + "*";
    }
    return pItemRule + "{" + std::to_string(pMinItems) + "," + (vHasMax ? std::to_string(pMaxItems) : "") + "}";
  }

  auto vResult = pItemRule + " " +
                 fBuildRepetition("(" + pSeparatorRule + " " + pItemRule + ")", pMinItems == 0 ? 0 : pMinItems - 1,
                                  vHasMax ? pMaxItems - 1 : pMaxItems);
  if (pMinItems == 0) {
    vResult = "(" + vResult + ")?";
  }
  return vResult;
}

static void fBuildMinMaxInt(int64_t pMinValue, int64_t pMaxValue, std::stringstream &pOut, int pDecimalsLeft = 16,
                            bool pTopLevel = true) {
  auto vHasMin = pMinValue != std::numeric_limits<int64_t>::min();
  auto vHasMax = pMaxValue != std::numeric_limits<int64_t>::max();

  auto vDigitRange = [&](char pFrom, char pTo) {
    pOut << "[";
    if (pFrom == pTo) {
      pOut << pFrom;
    } else {
      pOut << pFrom << "-" << pTo;
    }
    pOut << "]";
  };
  auto vMoreDigits = [&](int pMinDigits, int pMaxDigits) {
    pOut << "[0-9]";
    if (pMinDigits == pMaxDigits && pMinDigits == 1) {
      return;
    }
    pOut << "{";
    pOut << pMinDigits;
    if (pMaxDigits != pMinDigits) {
      pOut << ",";
      if (pMaxDigits != std::numeric_limits<int>::max()) {
        pOut << pMaxDigits;
      }
    }
    pOut << "}";
  };
  std::function<void(const std::string_view &, const std::string_view &)> vUniformRange =
      [&](const std::string_view &pFrom, const std::string_view &pTo) {
        size_t vI = 0;
        while (vI < pFrom.length() && vI < pTo.length() && pFrom[vI] == pTo[vI]) {
          vI++;
        }
        if (vI > 0) {
          pOut << "\"" << pFrom.substr(0, vI) << "\"";
        }
        if (vI < pFrom.length() && vI < pTo.length()) {
          if (vI > 0) {
            pOut << " ";
          }
          auto vSubLen = pFrom.length() - vI - 1;
          if (vSubLen > 0) {
            auto vFromSub = pFrom.substr(vI + 1);
            auto vToSub = pTo.substr(vI + 1);
            auto vSubZeros = fStringRepeat("0", vSubLen);
            auto vSubNines = fStringRepeat("9", vSubLen);

            auto vToReached = false;
            pOut << "(";
            if (vFromSub == vSubZeros) {
              vDigitRange(pFrom[vI], pTo[vI] - 1);
              pOut << " ";
              vMoreDigits(vSubLen, vSubLen);
            } else {
              pOut << "[" << pFrom[vI] << "] ";
              pOut << "(";
              vUniformRange(vFromSub, vSubNines);
              pOut << ")";
              if (pFrom[vI] < pTo[vI] - 1) {
                pOut << " | ";
                if (vToSub == vSubNines) {
                  vDigitRange(pFrom[vI] + 1, pTo[vI]);
                  vToReached = true;
                } else {
                  vDigitRange(pFrom[vI] + 1, pTo[vI] - 1);
                }
                pOut << " ";
                vMoreDigits(vSubLen, vSubLen);
              }
            }
            if (!vToReached) {
              pOut << " | ";
              vDigitRange(pTo[vI], pTo[vI]);
              pOut << " ";
              vUniformRange(vSubZeros, vToSub);
            }
            pOut << ")";
          } else {
            pOut << "[" << pFrom[vI] << "-" << pTo[vI] << "]";
          }
        }
      };

  if (vHasMin && vHasMax) {
    if (pMinValue < 0 && pMaxValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(-pMaxValue, -pMinValue, pOut, pDecimalsLeft, /* top_level= */ true);
      pOut << ")";
      return;
    }

    if (pMinValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(0, -pMinValue, pOut, pDecimalsLeft, /* top_level= */ true);
      pOut << ") | ";
      pMinValue = 0;
    }

    auto vMinS = std::to_string(pMinValue);
    auto vMaxS = std::to_string(pMaxValue);
    auto vMinDigits = vMinS.length();
    auto vMaxDigits = vMaxS.length();

    for (auto vDigits = vMinDigits; vDigits < vMaxDigits; vDigits++) {
      vUniformRange(vMinS, fStringRepeat("9", vDigits));
      vMinS = "1" + fStringRepeat("0", vDigits);
      pOut << " | ";
    }
    vUniformRange(vMinS, vMaxS);
    return;
  }

  auto vLessDecimals = std::max(pDecimalsLeft - 1, 1);

  if (vHasMin) {
    if (pMinValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(std::numeric_limits<int64_t>::min(), -pMinValue, pOut, pDecimalsLeft, /* top_level= */ false);
      pOut << ") | [0] | [1-9] ";
      vMoreDigits(0, pDecimalsLeft - 1);
    } else if (pMinValue == 0) {
      if (pTopLevel) {
        pOut << "[0] | [1-9] ";
        vMoreDigits(0, vLessDecimals);
      } else {
        vMoreDigits(1, pDecimalsLeft);
      }
    } else if (pMinValue <= 9) {
      char vC = '0' + pMinValue;
      auto vRangeStart = pTopLevel ? '1' : '0';
      if (vC > vRangeStart) {
        vDigitRange(vRangeStart, vC - 1);
        pOut << " ";
        vMoreDigits(1, vLessDecimals);
        pOut << " | ";
      }
      vDigitRange(vC, '9');
      pOut << " ";
      vMoreDigits(0, vLessDecimals);
    } else {
      auto vMinS = std::to_string(pMinValue);
      auto vLen = vMinS.length();
      auto vC = vMinS[0];

      if (vC > '1') {
        vDigitRange(pTopLevel ? '1' : '0', vC - 1);
        pOut << " ";
        vMoreDigits(vLen, vLessDecimals);
        pOut << " | ";
      }
      vDigitRange(vC, vC);
      pOut << " (";
      fBuildMinMaxInt(std::stoll(vMinS.substr(1)), std::numeric_limits<int64_t>::max(), pOut, vLessDecimals,
                      /* top_level= */ false);
      pOut << ")";
      if (vC < '9') {
        pOut << " | ";
        vDigitRange(vC + 1, '9');
        pOut << " ";
        vMoreDigits(vLen - 1, vLessDecimals);
      }
    }
    return;
  }

  if (vHasMax) {
    if (pMaxValue >= 0) {
      if (pTopLevel) {
        pOut << "\"-\" [1-9] ";
        vMoreDigits(0, vLessDecimals);
        pOut << " | ";
      }
      fBuildMinMaxInt(0, pMaxValue, pOut, pDecimalsLeft, /* top_level= */ true);
    } else {
      pOut << "\"-\" (";
      fBuildMinMaxInt(-pMaxValue, std::numeric_limits<int64_t>::max(), pOut, pDecimalsLeft, /* top_level= */ false);
      pOut << ")";
    }
    return;
  }

  throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string cSpaceRule = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
  std::string vContent;
  std::vector<std::string> lDeps;
};

static std::unordered_map<std::string, BuiltinRule> dPrimitiveRules = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number",
     {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value",
     {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object",
     {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
      {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid",
     {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} "
      "\"\\\"\"",
      {}}},
    {"char", {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static std::unordered_map<std::string, BuiltinRule> dStringFormatRules = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time",
     {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) "
      "( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )",
      {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}}};

static bool fIsReservedName(const std::string &pName) {
  static const std::unordered_set<std::string> sReservedNames = [] {
    std::unordered_set<std::string> sS;
    sS.insert("root");
    for (const auto &vP : dPrimitiveRules) {
      sS.insert(vP.first);
    }
    for (const auto &vP : dStringFormatRules) {
      sS.insert(vP.first);
    }
    return sS;
  }();
  return sReservedNames.find(pName) != sReservedNames.end();
}

// Schemas come from requests and are untrusted: every recursive walk over them is bounded, so that excessive
// nesting ends in a conversion error instead of exhausting the native stack
static constexpr size_t cMaxSchemaDepth = 256;       // JSON nesting levels, nested visits and $ref chains
static constexpr size_t cMaxPatternDepth = 128;      // nested groups in a "pattern" regular expression
static constexpr size_t cMaxNotStringsLength = 1024; // bytes of each property name excluded from additionalProperties
static constexpr size_t cMaxSchemaTypes = 7;         // a "type" array lists unique names of the seven JSON types

// Counts one level of a recursive walk for as long as it is alive
struct schema_depth_guard {
  size_t &vDepth;
  explicit schema_depth_guard(size_t &pDepth) : vDepth(pDepth) { ++vDepth; }
  ~schema_depth_guard() { --vDepth; }
  schema_depth_guard(const schema_depth_guard &) = delete;
  schema_depth_guard &operator=(const schema_depth_guard &) = delete;
};

// Measures the nesting iteratively, so that an overly deep document never reaches a recursive walk or copy
static bool fJsonExceedsDepth(const json &pValue, size_t pMaxDepth) {
  std::vector<std::pair<const json *, size_t>> lPending{{&pValue, 1}};
  while (!lPending.empty()) {
    const auto [vNode, vDepth] = lPending.back();
    lPending.pop_back();
    if (vDepth > pMaxDepth) {
      return true;
    }
    if (vNode->is_structured()) {
      for (const auto &vChild : *vNode) {
        lPending.emplace_back(&vChild, vDepth + 1);
      }
    }
  }
  return false;
}

static std::string fSchemaDepthError() {
  return "JSON schema nesting exceeds the maximum depth of " + std::to_string(cMaxSchemaDepth);
}

// Collapses every run of characters outside [a-zA-Z0-9-] into a single '-', as the regular expression
// "[^a-zA-Z0-9-]+" would; a loop is used because std::regex recurses once per matched character, and long
// names taken from the request would exhaust the stack
static std::string fSanitizeRuleName(const std::string &pName) {
  std::string vOut;
  vOut.reserve(pName.size());
  bool vInRun = false;
  for (const char cC : pName) {
    const bool cValid = (cC >= 'a' && cC <= 'z') || (cC >= 'A' && cC <= 'Z') || (cC >= '0' && cC <= '9') || cC == '-';
    if (cValid) {
      vOut += cC;
      vInRun = false;
    } else if (!vInRun) {
      vOut += '-';
      vInRun = true;
    }
  }
  return vOut;
}

static std::regex vGrammarLiteralEscapeRe("[\r\n\"\\\\]");
static std::regex vGrammarRangeLiteralEscapeRe("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> dGrammarLiteralEscapes = {{'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""},
                                                                       {'-', "\\-"},  {']', "\\]"},  {'\\', "\\\\"}};

static std::unordered_set<char> sNonLiteralSet = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
static std::unordered_set<char> sEscapedInRegexpsButNotInLiterals = {'^', '$', '.', '[', ']', '(', ')',
                                                                     '|', '{', '}', '*', '+', '?'};

static std::string fReplacePattern(const std::string &pInput, const std::regex &pRegex,
                                   const std::function<std::string(const std::smatch &)> &pReplacement) {
  std::smatch vMatch;
  std::string vResult;

  std::string::const_iterator vSearchStart(pInput.cbegin());
  std::string::const_iterator vSearchEnd(pInput.cend());

  while (std::regex_search(vSearchStart, vSearchEnd, vMatch, pRegex)) {
    vResult.append(vSearchStart, vSearchStart + vMatch.position());
    vResult.append(pReplacement(vMatch));
    vSearchStart = vMatch.suffix().first;
  }

  vResult.append(vSearchStart, vSearchEnd);

  return vResult;
}

static std::string fFormatLiteral(const std::string &pLiteral) {
  std::string vEscaped = fReplacePattern(pLiteral, vGrammarLiteralEscapeRe, [&](const std::smatch &pMatch) {
    char vC = pMatch.str()[0];
    return dGrammarLiteralEscapes.at(vC);
  });
  return "\"" + vEscaped + "\"";
}

std::string fGbnfFormatLiteral(const std::string &pLiteral) { return fFormatLiteral(pLiteral); }

class common_schema_converter {
private:
  friend class common_schema_info;
  friend std::string fBuildGrammar(const std::function<void(const common_grammar_builder &)> &pCb,
                                   const common_grammar_options &pOptions);
  std::function<json(const std::string &)> vFetchJson;
  bool vDotall;
  std::map<std::string, std::string> dRules;
  std::unordered_map<std::string, json> dRefs;
  std::unordered_set<std::string> sRefsBeingResolved;
  std::vector<std::string> lErrors;
  std::vector<std::string> lWarnings;
  size_t vVisitDepth = 0;          // nested visits, including those that follow $ref chains
  bool vDepthErrorAdded = false;   // the depth error is reported once, however many branches reach it

  void mAddDepthError() {
    if (!vDepthErrorAdded) {
      lErrors.push_back(fSchemaDepthError());
      vDepthErrorAdded = true;
    }
  }

  std::string mAddRule(const std::string &pName, const std::string &pRule) {
    std::string vEscName = fSanitizeRuleName(pName);
    if (dRules.find(vEscName) == dRules.end() || dRules[vEscName] == pRule) {
      dRules[vEscName] = pRule;
      return vEscName;
    }
    int vI = 0;
    while (dRules.find(vEscName + std::to_string(vI)) != dRules.end() &&
           dRules[vEscName + std::to_string(vI)] != pRule) {
      vI++;
    }
    std::string vKey = vEscName + std::to_string(vI);
    dRules[vKey] = pRule;
    return vKey;
  }

  std::string mGenerateUnionRule(const std::string &pName, const std::vector<json> &pAltSchemas) {
    std::vector<std::string> lRules;
    lRules.reserve(pAltSchemas.size());
    for (size_t vI = 0; vI < pAltSchemas.size(); vI++) {
      lRules.push_back(mVisit(pAltSchemas[vI], pName + (pName.empty() ? "alternative-" : "-") + std::to_string(vI)));
    }
    return fStringJoin(lRules, " | ");
  }

  std::string mVisitPattern(const std::string &pPattern, const std::string &pName) {
    // an empty pattern (or a lone "^" or "$") has no anchors to check
    if (pPattern.size() < 2 || !(pPattern.front() == '^' && pPattern.back() == '$')) {
      lErrors.push_back("Pattern must start with '^' and end with '$'");
      return "";
    }
    std::string vSubPattern = pPattern.substr(1, pPattern.length() - 2);
    std::unordered_map<std::string, std::string> dSubRuleIds;

    size_t vI = 0;
    size_t vLength = vSubPattern.length();
    size_t vGroupDepth = 0;

    using literal_or_rule = std::pair<std::string, bool>;
    auto vToRule = [&](const literal_or_rule &pLs) {
      auto vIsLiteral = pLs.second;
      auto vS = pLs.first;
      return vIsLiteral ? "\"" + vS + "\"" : vS;
    };
    // A malformed pattern ends the conversion with an error, and every nested group stops with it
    auto vFail = [&](const std::string &pError) -> literal_or_rule {
      lErrors.push_back(pError);
      vI = vLength;
      return std::make_pair(std::string(), false);
    };
    // A quantifier needs an atom before it, not the start of the sequence or an alternation bar
    auto vNothingToRepeat = [](const std::vector<literal_or_rule> &pSeq) {
      return pSeq.empty() || (!pSeq.back().second && pSeq.back().first == "|");
    };
    std::function<literal_or_rule()> vTransform = [&]() -> literal_or_rule {
      size_t vStart = vI;
      std::vector<literal_or_rule> lSeq;

      auto vGetDot = [&]() {
        std::string vRule;
        if (vDotall) {
          vRule = "[\\U00000000-\\U0010FFFF]";
        } else {
          vRule = "[^\\x0A\\x0D]";
        }
        return mAddRule("dot", vRule);
      };

      // Joins the sequence, merging consecutive literals together.
      auto vJoinSeq = [&]() {
        std::vector<literal_or_rule> lRet;

        std::string vLiteral;
        auto vFlushLiteral = [&]() {
          if (vLiteral.empty()) {
            return false;
          }
          lRet.emplace_back(vLiteral, true);
          vLiteral.clear();
          return true;
        };

        for (const auto &vItem : lSeq) {
          auto vIsLiteral = vItem.second;
          if (vIsLiteral) {
            vLiteral += vItem.first;
          } else {
            vFlushLiteral();
            lRet.push_back(vItem);
          }
        }
        vFlushLiteral();

        std::vector<std::string> lResults;
        lResults.reserve(lRet.size());
        for (const auto &vItem : lRet) {
          lResults.push_back(vToRule(vItem));
        }
        return std::make_pair(fStringJoin(lResults, " "), false);
      };

      while (vI < vLength) {
        char vC = vSubPattern[vI];
        if (vC == '.') {
          lSeq.emplace_back(vGetDot(), false);
          vI++;
        } else if (vC == '(') {
          vI++;
          if (vI < vLength && vSubPattern[vI] == '?') {
            if (vI + 1 < vLength && vSubPattern[vI + 1] == ':') {
              vI += 2; // skip "?:" for non-capturing group, treat as regular group
            } else {
              // lookahead/lookbehind (?=, ?!, ?<=, ?<!) - not supported
              lWarnings.push_back("Unsupported pattern syntax");
              // skip to matching ')' to avoid UB on empty seq
              int vDepth = 1;
              while (vI < vLength && vDepth > 0) {
                if (vSubPattern[vI] == '\\' && vI + 1 < vLength) {
                  vI += 2; // skip escaped character
                } else {
                  if (vSubPattern[vI] == '(')
                    vDepth++;
                  else if (vSubPattern[vI] == ')')
                    vDepth--;
                  vI++;
                }
              }
              continue;
            }
          }
          if (vGroupDepth >= cMaxPatternDepth) {
            return vFail("Pattern nesting exceeds the maximum depth of " + std::to_string(cMaxPatternDepth));
          }
          schema_depth_guard vGroupGuard(vGroupDepth);
          lSeq.emplace_back("(" + vToRule(vTransform()) + ")", false);
        } else if (vC == ')') {
          vI++;
          if (vStart > 0 && vSubPattern[vStart - 1] != '(' &&
              (vStart < 2 || vSubPattern[vStart - 2] != '?' || vSubPattern[vStart - 1] != ':')) {
            lErrors.push_back("Unbalanced parentheses");
          }
          return vJoinSeq();
        } else if (vC == '[') {
          std::string vSquareBrackets = std::string(1, vC);
          vI++;
          while (vI < vLength && vSubPattern[vI] != ']') {
            if (vSubPattern[vI] == '\\') {
              vSquareBrackets += vSubPattern.substr(vI, 2);
              vI += 2;
            } else {
              vSquareBrackets += vSubPattern[vI];
              vI++;
            }
          }
          if (vI >= vLength) {
            lErrors.push_back("Unbalanced square brackets");
          }
          vSquareBrackets += ']';
          vI++;
          lSeq.emplace_back(vSquareBrackets, false);
        } else if (vC == '|') {
          lSeq.emplace_back("|", false);
          vI++;
        } else if (vC == '*' || vC == '+' || vC == '?') {
          if (vNothingToRepeat(lSeq)) {
            return vFail(std::string("Quantifier '") + vC + "' has nothing to repeat");
          }
          lSeq.back() = std::make_pair(vToRule(lSeq.back()) + vC, false);
          vI++;
        } else if (vC == '{') {
          if (vNothingToRepeat(lSeq)) {
            return vFail("Quantifier '{' has nothing to repeat");
          }
          std::string vCurlyBrackets = std::string(1, vC);
          vI++;
          while (vI < vLength && vSubPattern[vI] != '}') {
            vCurlyBrackets += vSubPattern[vI];
            vI++;
          }
          if (vI >= vLength) {
            lErrors.push_back("Unbalanced curly brackets");
          }
          vCurlyBrackets += '}';
          vI++;
          auto vNums = fStringSplit(vCurlyBrackets.substr(1, vCurlyBrackets.length() - 2), ",");
          int vMinTimes = 0;
          int vMaxTimes = std::numeric_limits<int>::max();
          try {
            if (vNums.size() == 1) {
              vMinTimes = vMaxTimes = std::stoi(vNums[0]);
            } else if (vNums.size() != 2) {
              lErrors.push_back("Wrong number of values in curly brackets");
            } else {
              if (!vNums[0].empty()) {
                vMinTimes = std::stoi(vNums[0]);
              }
              if (!vNums[1].empty()) {
                vMaxTimes = std::stoi(vNums[1]);
              }
            }
          } catch (const std::invalid_argument &vE) {
            lErrors.push_back("Invalid number in curly brackets");
            return std::make_pair("", false);
          } catch (const std::out_of_range &vE) {
            lErrors.push_back("Invalid number in curly brackets");
            return std::make_pair("", false);
          }
          auto &vLast = lSeq.back();
          auto &vSub = vLast.first;
          auto vSubIsLiteral = vLast.second;

          if (!vSubIsLiteral) {
            std::string &vSubId = dSubRuleIds[vSub];
            if (vSubId.empty()) {
              vSubId = mAddRule(pName + "-" + std::to_string(dSubRuleIds.size()), vSub);
            }
            vSub = vSubId;
          }
          lSeq.back().first = fBuildRepetition(vSubIsLiteral ? "\"" + vSub + "\"" : vSub, vMinTimes, vMaxTimes, "");
          lSeq.back().second = false;
        } else {
          std::string vLiteral;
          auto vIsNonLiteral = [&](char pC) { return sNonLiteralSet.find(pC) != sNonLiteralSet.end(); };
          while (vI < vLength) {
            if (vSubPattern[vI] == '\\' && vI < vLength - 1) {
              char vNext = vSubPattern[vI + 1];
              if (sEscapedInRegexpsButNotInLiterals.find(vNext) != sEscapedInRegexpsButNotInLiterals.end()) {
                vI++;
                vLiteral += vSubPattern[vI];
                vI++;
              } else {
                vLiteral += vSubPattern.substr(vI, 2);
                vI += 2;
              }
            } else if (vSubPattern[vI] == '"') {
              vLiteral += "\\\"";
              vI++;
            } else if (!vIsNonLiteral(vSubPattern[vI]) &&
                       (vI == vLength - 1 || vLiteral.empty() || vSubPattern[vI + 1] == '.' ||
                        !vIsNonLiteral(vSubPattern[vI + 1]))) {
              vLiteral += vSubPattern[vI];
              vI++;
            } else {
              break;
            }
          }
          if (!vLiteral.empty()) {
            lSeq.emplace_back(vLiteral, true);
          }
        }
      }
      return vJoinSeq();
    };
    return mAddRule(pName, "\"\\\"\" (" + vToRule(vTransform()) + ") \"\\\"\"");
  }

  /*
      Returns a rule that matches a JSON string that is none of the provided strings

      not_strings({"a"})
          -> ["] ( [a] char+ | [^"a] char* )? ["]
      not_strings({"and", "also"})
          -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln]
     char*) | [^"a] char* )? ["]
  */
  std::string mNotStrings(const std::vector<std::string> &pStrings) {
    // the trie below is visited recursively, one level (and one nested group) per byte of the longest name
    for (const auto &vS : pStrings) {
      if (vS.size() > cMaxNotStringsLength) {
        lErrors.push_back("Property name exceeds the maximum length of " + std::to_string(cMaxNotStringsLength) +
                          " bytes");
        return "";
      }
    }

    struct TrieNode {
      std::map<char, TrieNode> dChildren;
      bool vIsEndOfString;

      TrieNode() : vIsEndOfString(false) {}

      void mInsert(const std::string &pString) {
        auto *vNode = this;
        for (char vC : pString) {
          vNode = &vNode->dChildren[vC];
        }
        vNode->vIsEndOfString = true;
      }
    };

    TrieNode vTrie;
    for (const auto &vS : pStrings) {
      vTrie.mInsert(vS);
    }

    std::string vCharRule = mAddPrimitive("char", dPrimitiveRules.at("char"));
    std::ostringstream vOut;
    vOut << "[\"] ( ";
    std::function<void(const TrieNode &)> vVisit = [&](const TrieNode &pNode) {
      std::ostringstream vRejects;
      auto vFirst = true;
      for (const auto &vKv : pNode.dChildren) {
        vRejects << vKv.first;
        if (vFirst) {
          vFirst = false;
        } else {
          vOut << " | ";
        }
        vOut << "[" << vKv.first << "]";
        if (!vKv.second.dChildren.empty()) {
          vOut << " (";
          vVisit(vKv.second);
          vOut << ")";
        } else if (vKv.second.vIsEndOfString) {
          vOut << " " << vCharRule << "+";
        }
      }
      if (!pNode.dChildren.empty()) {
        if (!vFirst) {
          vOut << " | ";
        }
        vOut << "[^\"" << vRejects.str() << "] " << vCharRule << "*";
      }
    };
    vVisit(vTrie);

    vOut << " )";
    if (!vTrie.vIsEndOfString) {
      vOut << "?";
    }
    vOut << " [\"]";
    return vOut.str();
  }

  std::string mResolveRef(const std::string &pRef) {
    auto vIt = pRef.find('#');
    std::string vRefFragment = vIt != std::string::npos ? pRef.substr(vIt + 1) : pRef;
    std::string vRefName = "ref" + fSanitizeRuleName(vRefFragment);
    if (dRules.find(vRefName) == dRules.end() && sRefsBeingResolved.find(pRef) == sRefsBeingResolved.end()) {
      sRefsBeingResolved.insert(pRef);
      json vResolved = dRefs[pRef];
      vRefName = mVisit(vResolved, vRefName);
      sRefsBeingResolved.erase(pRef);
    }
    return vRefName;
  }

  std::string mBuildObjectRule(const std::vector<std::pair<std::string, json>> &pProperties,
                               const std::unordered_set<std::string> &pRequired, const std::string &pName,
                               const json &pAdditionalProperties) {
    std::vector<std::string> lRequiredProps;
    std::vector<std::string> lOptionalProps;
    std::unordered_map<std::string, std::string> dPropKvRuleNames;
    std::vector<std::string> lPropNames;
    for (const auto &vKv : pProperties) {
      const auto &vPropName = vKv.first;
      const auto &vPropSchema = vKv.second;

      std::string vPropRuleName = mVisit(vPropSchema, pName + (pName.empty() ? "" : "-") + vPropName);
      dPropKvRuleNames[vPropName] =
          mAddRule(pName + (pName.empty() ? "" : "-") + vPropName + "-kv",
                   fFormatLiteral(json(vPropName).dump()) + " space \":\" space " + vPropRuleName);
      if (pRequired.find(vPropName) != pRequired.end()) {
        lRequiredProps.push_back(vPropName);
      } else {
        lOptionalProps.push_back(vPropName);
      }
      lPropNames.push_back(vPropName);
    }
    if ((pAdditionalProperties.is_boolean() && pAdditionalProperties.get<bool>()) ||
        pAdditionalProperties.is_object()) {
      std::string vSubName = pName + (pName.empty() ? "" : "-") + "additional";
      std::string vValueRule = pAdditionalProperties.is_object() ? mVisit(pAdditionalProperties, vSubName + "-value")
                                                                 : mAddPrimitive("value", dPrimitiveRules.at("value"));

      auto vKeyRule = lPropNames.empty() ? mAddPrimitive("string", dPrimitiveRules.at("string"))
                                         : mAddRule(vSubName + "-k", mNotStrings(lPropNames));
      std::string vKvRule = mAddRule(vSubName + "-kv", vKeyRule + " \":\" space " + vValueRule);
      dPropKvRuleNames["*"] = vKvRule;
      lOptionalProps.push_back("*");
    }

    std::string vRule = "\"{\" space ";
    for (size_t vI = 0; vI < lRequiredProps.size(); vI++) {
      if (vI > 0) {
        vRule += " \",\" space ";
      }
      vRule += dPropKvRuleNames[lRequiredProps[vI]];
    }

    if (!lOptionalProps.empty()) {
      vRule += " (";
      if (!lRequiredProps.empty()) {
        vRule += " \",\" space ( ";
      }

      // Rule for the optional keys from position pI onwards: the key itself, then the "-rest" rule that matches
      // every key after it. The "-rest" rules are built once, from the last key backwards, in the same order and
      // with the same names and content as a recursive definition would add them, but without one stack frame
      // and one copy of the remaining keys per key, which a request with many optional properties could exhaust
      const size_t cNOptional = lOptionalProps.size();
      auto vOptionalHead = [&](size_t pI, bool pFirstIsOptional) {
        const std::string &vK = lOptionalProps[pI];
        std::string vKvRuleName = dPropKvRuleNames[vK];
        std::string vCommaRef = "( \",\" space " + vKvRuleName + " )";
        if (pFirstIsOptional) {
          return vCommaRef + (vK == "*" ? "*" : "?");
        }
        return vKvRuleName + (vK == "*" ? " " + vCommaRef + "*" : "");
      };
      std::vector<std::string> lRestRuleNames(cNOptional);
      auto vOptionalTail = [&](size_t pI) {
        return pI + 1 < cNOptional ? " " + lRestRuleNames[pI] : std::string();
      };
      for (size_t vI = cNOptional; vI-- > 1;) {
        lRestRuleNames[vI - 1] = mAddRule(pName + (pName.empty() ? "" : "-") + lOptionalProps[vI - 1] + "-rest",
                                          vOptionalHead(vI, true) + vOptionalTail(vI));
      }

      for (size_t vI = 0; vI < cNOptional; vI++) {
        if (vI > 0) {
          vRule += " | ";
        }
        vRule += vOptionalHead(vI, false) + vOptionalTail(vI);
      }
      if (!lRequiredProps.empty()) {
        vRule += " )";
      }
      vRule += " )?";
    }

    vRule += " space \"}\"";

    return vRule;
  }

  std::string mAddPrimitive(const std::string &pName, const BuiltinRule &pRule) {
    auto vN = mAddRule(pName, pRule.vContent);
    for (const auto &vDep : pRule.lDeps) {
      BuiltinRule vDepRule;
      auto vIt = dPrimitiveRules.find(vDep);
      if (vIt == dPrimitiveRules.end()) {
        vIt = dStringFormatRules.find(vDep);
        if (vIt == dStringFormatRules.end()) {
          lErrors.push_back("Rule " + vDep + " not known");
          continue;
        }
      }
      if (dRules.find(vDep) == dRules.end()) {
        mAddPrimitive(vDep, vIt->second);
      }
    }
    return vN;
  }

public:
  common_schema_converter(const std::function<json(const std::string &)> &pFetchJson, bool pDotall)
      : vFetchJson(pFetchJson), vDotall(pDotall) {
    dRules["space"] = cSpaceRule;
  }

  void mResolveRefs(json &pSchema, const std::string &pUrl) {
    /*
     * Resolves all $ref fields in the given schema, fetching any remote schemas,
     * replacing each $ref with absolute reference URL and populates _refs with the
     * respective referenced (sub)schema dictionaries.
     */
    // the walk below is recursive, so the nesting is measured iteratively first
    if (fJsonExceedsDepth(pSchema, cMaxSchemaDepth)) {
      mAddDepthError();
      return;
    }
    std::function<void(json &)> vVisitRefs = [&](json &pN) {
      if (pN.is_array()) {
        for (auto &vX : pN) {
          vVisitRefs(vX);
        }
      } else if (pN.is_object()) {
        if (pN.contains("$ref")) {
          std::string vRef = pN["$ref"];
          if (dRefs.find(vRef) == dRefs.end()) {
            // the target is followed through pointers, so that each reference copies only the subschema it
            // points to, not the whole document (a request with many references would copy it once per reference)
            static const json cNoTarget;
            const json *vTarget = &cNoTarget;
            if (vRef.find("https://") == 0) {
              std::string vBaseUrl = vRef.substr(0, vRef.find('#'));
              auto vIt = dRefs.find(vBaseUrl);
              if (vIt != dRefs.end()) {
                vTarget = &vIt->second;
              } else {
                // Fetch the referenced schema and resolve its refs
                auto vReferenced = vFetchJson(vRef);
                mResolveRefs(vReferenced, vBaseUrl);
                dRefs[vBaseUrl] = vReferenced;
              }
              if (vRef.find('#') == std::string::npos || vRef.substr(vRef.find('#') + 1).empty()) {
                return;
              }
            } else if (vRef.find("#/") == 0) {
              vTarget = &pSchema;
              pN["$ref"] = pUrl + vRef;
              vRef = pUrl + vRef;
            } else {
              lErrors.push_back("Unsupported ref: " + vRef);
              return;
            }
            std::string vPointer = vRef.substr(vRef.find('#') + 1);
            std::vector<std::string> lTokens = fStringSplit(vPointer, "/");
            for (size_t vI = 1; vI < lTokens.size(); ++vI) {
              const std::string &vSel = lTokens[vI];
              if (vTarget->is_object() && vTarget->contains(vSel)) {
                vTarget = &(*vTarget)[vSel];
              } else if (vTarget->is_array()) {
                size_t vSelIndex;
                try {
                  vSelIndex = std::stoull(vSel);
                } catch (const std::invalid_argument &vE) {
                  vSelIndex = vTarget->size();
                } catch (const std::out_of_range &vE) {
                  vSelIndex = vTarget->size();
                }
                if (vSelIndex >= vTarget->size()) {
                  lErrors.push_back("Error resolving ref " + vRef + ": " + vSel + " not in " + vTarget->dump());
                  return;
                }
                vTarget = &(*vTarget)[vSelIndex];
              } else {
                lErrors.push_back("Error resolving ref " + vRef + ": " + vSel + " not in " + vTarget->dump());
                return;
              }
            }
            dRefs[vRef] = *vTarget;
          }
        } else {
          for (const auto &vKv : pN.items()) {
            vVisitRefs(vKv.value());
          }
        }
      }
    };

    vVisitRefs(pSchema);
  }

  static std::string mGenerateConstantRule(const json &pValue) { return fFormatLiteral(pValue.dump()); }

  std::string mVisit(const json &pSchema, const std::string &pName) {
    // The whole document is measured iteratively before the first visit (copies and dumps of a subschema are
    // recursive as well); nested visits, including those that follow $ref chains, are counted
    if (vVisitDepth == 0 && fJsonExceedsDepth(pSchema, cMaxSchemaDepth)) {
      mAddDepthError();
      return "";
    }
    if (vVisitDepth >= cMaxSchemaDepth) {
      mAddDepthError();
      return "";
    }
    schema_depth_guard vDepthGuard(vVisitDepth);
    return mVisitSchema(pSchema, pName);
  }

  std::string mVisitSchema(const json &pSchema, const std::string &pName) {
    json vSchemaType = pSchema.contains("type") ? pSchema["type"] : json();
    std::string vSchemaFormat = pSchema.contains("format") ? pSchema["format"].get<std::string>() : "";
    std::string vRuleName = fIsReservedName(pName) ? pName + "-" : pName.empty() ? "root" : pName;

    if (pSchema.contains("$ref")) {
      return mAddRule(vRuleName, mResolveRef(pSchema["$ref"]));
    }
    if (pSchema.contains("oneOf") || pSchema.contains("anyOf")) {
      std::vector<json> lAltSchemas = pSchema.contains("oneOf") ? pSchema["oneOf"].get<std::vector<json>>()
                                                                : pSchema["anyOf"].get<std::vector<json>>();
      return mAddRule(vRuleName, mGenerateUnionRule(pName, lAltSchemas));
    }
    if (vSchemaType.is_array()) {
      // the schema is copied once per entry, so a long array would make the copies quadratic
      if (vSchemaType.size() > cMaxSchemaTypes) {
        lErrors.push_back("A \"type\" array cannot list more than " + std::to_string(cMaxSchemaTypes) + " types");
        return "";
      }
      std::vector<json> lSchemaTypes;
      for (const auto &vT : vSchemaType) {
        json vSchemaCopy(pSchema);
        vSchemaCopy["type"] = vT;
        lSchemaTypes.push_back(vSchemaCopy);
      }
      return mAddRule(vRuleName, mGenerateUnionRule(pName, lSchemaTypes));
    }
    if (pSchema.contains("const")) {
      return mAddRule(vRuleName, mGenerateConstantRule(pSchema["const"]));
    }
    if (pSchema.contains("enum")) {
      std::vector<std::string> lEnumValues;
      for (const auto &vV : pSchema["enum"]) {
        lEnumValues.push_back(mGenerateConstantRule(vV));
      }
      return mAddRule(vRuleName, "(" + fStringJoin(lEnumValues, " | ") + ")");
    }
    if ((vSchemaType.is_null() || vSchemaType == "object") &&
        (pSchema.contains("properties") ||
         (pSchema.contains("additionalProperties") && pSchema["additionalProperties"] != true))) {
      std::unordered_set<std::string> sRequired;
      if (pSchema.contains("required") && pSchema["required"].is_array()) {
        for (const auto &vItem : pSchema["required"]) {
          if (vItem.is_string()) {
            sRequired.insert(vItem.get<std::string>());
          }
        }
      }
      std::vector<std::pair<std::string, json>> lProperties;
      if (pSchema.contains("properties")) {
        for (const auto &vProp : pSchema["properties"].items()) {
          lProperties.emplace_back(vProp.key(), vProp.value());
        }
      }
      return mAddRule(
          vRuleName,
          mBuildObjectRule(lProperties, sRequired, pName,
                           pSchema.contains("additionalProperties") ? pSchema["additionalProperties"] : json()));
    }
    if ((vSchemaType.is_null() || vSchemaType == "object" || vSchemaType == "string") && pSchema.contains("allOf")) {
      std::unordered_set<std::string> sRequired;
      std::vector<std::pair<std::string, json>> lProperties;
      std::map<std::string, size_t> dEnumValues;
      const std::string &vHybridName = pName;
      std::function<void(const json &, bool)> vAddComponent = [&](const json &pCompSchema, bool pIsRequired) {
        if (pCompSchema.contains("$ref")) {
          // $ref chains, and cycles among them, are bounded like nested visits
          if (vVisitDepth >= cMaxSchemaDepth) {
            mAddDepthError();
            return;
          }
          schema_depth_guard vDepthGuard(vVisitDepth);
          vAddComponent(dRefs[pCompSchema["$ref"]], pIsRequired);
        } else if (pCompSchema.contains("properties")) {
          for (const auto &vProp : pCompSchema["properties"].items()) {
            lProperties.emplace_back(vProp.key(), vProp.value());
            if (pIsRequired) {
              sRequired.insert(vProp.key());
            }
          }
        } else if (pCompSchema.contains("enum")) {
          for (const auto &vV : pCompSchema["enum"]) {
            const auto cRule = mGenerateConstantRule(vV);
            if (dEnumValues.find(cRule) == dEnumValues.end()) {
              dEnumValues[cRule] = 0;
            }
            dEnumValues[cRule] += 1;
          }
        } else {
          // todo warning
        }
      };
      for (const auto &vT : pSchema["allOf"]) {
        if (vT.contains("anyOf")) {
          for (const auto &vTt : vT["anyOf"]) {
            vAddComponent(vTt, false);
          }
        } else {
          vAddComponent(vT, true);
        }
      }
      if (!dEnumValues.empty()) {
        std::vector<std::string> lEnumIntersection;
        for (const auto &vP : dEnumValues) {
          if (vP.second == pSchema["allOf"].size()) {
            lEnumIntersection.push_back(vP.first);
          }
        }
        if (!lEnumIntersection.empty()) {
          return mAddRule(vRuleName, "(" + fStringJoin(lEnumIntersection, " | ") + ")");
        }
      }
      return mAddRule(vRuleName, mBuildObjectRule(lProperties, sRequired, vHybridName, json()));
    }
    if ((vSchemaType.is_null() || vSchemaType == "array") &&
        (pSchema.contains("items") || pSchema.contains("prefixItems"))) {
      json vItems = pSchema.contains("items") ? pSchema["items"] : pSchema["prefixItems"];
      if (vItems.is_array()) {
        std::string vRule = "\"[\" space ";
        for (size_t vI = 0; vI < vItems.size(); vI++) {
          if (vI > 0) {
            vRule += " \",\" space ";
          }
          vRule += mVisit(vItems[vI], pName + (pName.empty() ? "" : "-") + "tuple-" + std::to_string(vI));
        }
        vRule += " space \"]\"";
        return mAddRule(vRuleName, vRule);
      }
      std::string vItemRuleName = mVisit(vItems, pName + (pName.empty() ? "" : "-") + "item");
      int vMinItems = pSchema.contains("minItems") ? pSchema["minItems"].get<int>() : 0;
      json vMaxItemsJson = pSchema.contains("maxItems") ? pSchema["maxItems"] : json();
      int vMaxItems = vMaxItemsJson.is_number_integer() ? vMaxItemsJson.get<int>() : std::numeric_limits<int>::max();

      return mAddRule(vRuleName, "\"[\" space " + fBuildRepetition(vItemRuleName, vMinItems, vMaxItems, "\",\" space") +
                                     " space \"]\"");
    }
    if ((vSchemaType.is_null() || vSchemaType == "string") && pSchema.contains("pattern")) {
      return mVisitPattern(pSchema["pattern"], vRuleName);
    }
    if ((vSchemaType.is_null() || vSchemaType == "string") &&
        std::regex_match(vSchemaFormat, std::regex("^uuid[1-5]?$"))) {
      return mAddPrimitive(vRuleName == "root" ? "root" : vSchemaFormat, dPrimitiveRules.at("uuid"));
    }
    if ((vSchemaType.is_null() || vSchemaType == "string") &&
        dStringFormatRules.find(vSchemaFormat + "-string") != dStringFormatRules.end()) {
      auto vPrimName = vSchemaFormat + "-string";
      return mAddRule(vRuleName, mAddPrimitive(vPrimName, dStringFormatRules.at(vPrimName)));
    }
    if (vSchemaType == "string" && (pSchema.contains("minLength") || pSchema.contains("maxLength"))) {
      std::string vCharRule = mAddPrimitive("char", dPrimitiveRules.at("char"));
      int vMinLen = pSchema.contains("minLength") ? pSchema["minLength"].get<int>() : 0;
      int vMaxLen = pSchema.contains("maxLength") ? pSchema["maxLength"].get<int>() : std::numeric_limits<int>::max();
      return mAddRule(vRuleName, "\"\\\"\" " + fBuildRepetition(vCharRule, vMinLen, vMaxLen) + " \"\\\"\"");
    }
    if (vSchemaType == "integer" && (pSchema.contains("minimum") || pSchema.contains("exclusiveMinimum") ||
                                     pSchema.contains("maximum") || pSchema.contains("exclusiveMaximum"))) {
      int64_t vMinValue = std::numeric_limits<int64_t>::min();
      int64_t vMaxValue = std::numeric_limits<int64_t>::max();
      if (pSchema.contains("minimum")) {
        vMinValue = pSchema["minimum"].get<int64_t>();
      } else if (pSchema.contains("exclusiveMinimum")) {
        vMinValue = pSchema["exclusiveMinimum"].get<int64_t>() + 1;
      }
      if (pSchema.contains("maximum")) {
        vMaxValue = pSchema["maximum"].get<int64_t>();
      } else if (pSchema.contains("exclusiveMaximum")) {
        vMaxValue = pSchema["exclusiveMaximum"].get<int64_t>() - 1;
      }
      std::stringstream vOut;
      vOut << "(";
      fBuildMinMaxInt(vMinValue, vMaxValue, vOut);
      vOut << ")";
      return mAddRule(vRuleName, vOut.str());
    }
    if (pSchema.empty() || vSchemaType == "object") {
      return mAddRule(vRuleName, mAddPrimitive("object", dPrimitiveRules.at("object")));
    }
    if (vSchemaType.is_null() && pSchema.is_object()) {
      // No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
      // Per JSON Schema semantics this is equivalent to {} and accepts any value.
      return mAddRule(vRuleName, mAddPrimitive("value", dPrimitiveRules.at("value")));
    }
    if (!vSchemaType.is_string() || dPrimitiveRules.find(vSchemaType.get<std::string>()) == dPrimitiveRules.end()) {
      lErrors.push_back("Unrecognized schema: " + pSchema.dump());
      return "";
    }
    // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
    return mAddPrimitive(vRuleName == "root" ? "root" : vSchemaType.get<std::string>(),
                         dPrimitiveRules.at(vSchemaType.get<std::string>()));
  }

  void mCheckErrors() {
    if (!lErrors.empty()) {
      throw std::invalid_argument("JSON schema conversion failed:\n" + fStringJoin(lErrors, "\n"));
    }
    if (!lWarnings.empty()) {
      fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", fStringJoin(lWarnings, "; ").c_str());
    }
  }

  std::string mFormatGrammar() {
    std::stringstream vSs;
    for (const auto &vKv : dRules) {
      vSs << vKv.first << " ::= " << vKv.second << '\n';
    }
    return vSs.str();
  }
};

// common_schema_info implementation (pimpl)

common_schema_info::common_schema_info()
    : vImpl(std::make_unique<common_schema_converter>([](const std::string &) { return json(); }, false)) {}

common_schema_info::~common_schema_info() = default;

common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info &common_schema_info::operator=(common_schema_info &&) noexcept = default;

void common_schema_info::mResolveRefs(nlohmann::ordered_json &pSchema) { vImpl->mResolveRefs(pSchema, ""); }

// Determines if a JSON schema can resolve to a string type through any path.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
// If any branch of the schema (via oneOf, anyOf, $ref, etc.) permits a string, this returns
// true, allowing callers to handle the value as a raw string for simplicity.
bool common_schema_info::mResolvesToString(const nlohmann::ordered_json &pSchema) {
  std::unordered_set<std::string> sVisitedRefs;

  // the check below is recursive: too deep a schema, or too long a $ref chain, is assumed not to be a string
  if (fJsonExceedsDepth(pSchema, cMaxSchemaDepth)) {
    return false;
  }
  size_t vDepth = 0;

  std::function<bool(const json &)> vCheck = [&](const json &pS) -> bool {
    if (!pS.is_object() || vDepth >= cMaxSchemaDepth) {
      return false;
    }
    schema_depth_guard vDepthGuard(vDepth);

    // Handle $ref
    if (pS.contains("$ref")) {
      const std::string &vRef = pS["$ref"];
      if (sVisitedRefs.find(vRef) != sVisitedRefs.end()) {
        // Circular reference, assume not a string to be safe
        return false;
      }
      sVisitedRefs.insert(vRef);
      auto vIt = vImpl->dRefs.find(vRef);
      if (vIt != vImpl->dRefs.end()) {
        return vCheck(vIt->second);
      }
      return false;
    }

    // Check type field
    if (pS.contains("type")) {
      const json &vSchemaType = pS["type"];
      if (vSchemaType.is_string()) {
        if (vSchemaType == "string") {
          return true;
        }
      } else if (vSchemaType.is_array()) {
        // Type can be an array like ["string", "null"]
        for (const auto &vT : vSchemaType) {
          if (vT == "string") {
            return true;
          }
        }
      }
    }

    // Check oneOf/anyOf - if any alternative can be a string
    if (pS.contains("oneOf")) {
      for (const auto &vAlt : pS["oneOf"]) {
        if (vCheck(vAlt)) {
          return true;
        }
      }
    }
    if (pS.contains("anyOf")) {
      for (const auto &vAlt : pS["anyOf"]) {
        if (vCheck(vAlt)) {
          return true;
        }
      }
    }

    // Check allOf - all components must be compatible with string type
    if (pS.contains("allOf")) {
      bool vAllString = true;
      for (const auto &vComponent : pS["allOf"]) {
        if (!vCheck(vComponent)) {
          vAllString = false;
          break;
        }
      }
      if (vAllString) {
        return true;
      }
    }

    // Check const - if the constant value is a string
    if (pS.contains("const")) {
      if (pS["const"].is_string()) {
        return true;
      }
    }

    // Check enum - if any enum value is a string
    if (pS.contains("enum")) {
      for (const auto &vVal : pS["enum"]) {
        if (vVal.is_string()) {
          return true;
        }
      }
    }

    // String-specific keywords imply string type
    if (pS.contains("pattern") || pS.contains("minLength") || pS.contains("maxLength")) {
      return true;
    }

    // Check format - many formats imply string
    if (pS.contains("format")) {
      const std::string &vFmt = pS["format"];
      if (vFmt == "date" || vFmt == "time" || vFmt == "date-time" || vFmt == "uri" || vFmt == "email" ||
          vFmt == "hostname" || vFmt == "ipv4" || vFmt == "ipv6" || vFmt == "uuid" || vFmt.find("uuid") == 0) {
        return true;
      }
    }

    return false;
  };

  return vCheck(pSchema);
}

std::string fJsonSchemaToGrammar(const json &pSchema, bool pForceGbnf) {
  // copying and dumping a JSON document are recursive, so the nesting is measured before either
  if (fJsonExceedsDepth(pSchema, cMaxSchemaDepth)) {
    throw std::invalid_argument("JSON schema conversion failed:\n" + fSchemaDepthError());
  }
#ifdef MGS_USE_LLGUIDANCE
  if (!pForceGbnf) {
    return "%llguidance {}\nstart: %json " + pSchema.dump();
  }
#else
  (void)pForceGbnf;
#endif // MGS_USE_LLGUIDANCE
  return fBuildGrammar([&](const common_grammar_builder &pCallbacks) {
    auto vCopy = pSchema;
    pCallbacks.vResolveRefs(vCopy);
    pCallbacks.vAddSchema("", vCopy);
  });
}

std::string fBuildGrammar(const std::function<void(const common_grammar_builder &)> &pCb,
                          const common_grammar_options &pOptions) {
  common_schema_converter vConverter([&](const std::string &) { return json(); }, pOptions.vDotall);
  common_grammar_builder vBuilder{
      /* .vAddRule = */ [&](const std::string &pName, const std::string &pRule) {
        return vConverter.mAddRule(pName, pRule);
      },
      /* .vAddSchema = */
      [&](const std::string &pName, const nlohmann::ordered_json &pSchema) {
        return vConverter.mVisit(pSchema, pName == "root" ? "" : pName);
      },
      /* .vResolveRefs = */ [&](nlohmann::ordered_json &pSchema) { vConverter.mResolveRefs(pSchema, ""); }};
  pCb(vBuilder);
  vConverter.mCheckErrors();
  return vConverter.mFormatGrammar();
}
