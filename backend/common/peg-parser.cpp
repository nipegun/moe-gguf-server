#include "peg-parser.h"

#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <stdexcept>

// Trick to catch missing branches
template <typename T> inline constexpr bool cIsAlwaysFalseV = false;

const char *fCommonPegParseResultTypeName(common_peg_parse_result_type pType) {
  switch (pType) {
  case cCommonPegParseResultFail:
    return "fail";
  case cCommonPegParseResultSuccess:
    return "success";
  case cCommonPegParseResultNeedMoreInput:
    return "need_more_input";
  default:
    return "unknown";
  }
}

static bool fIsHexDigit(const char pC) {
  return (pC >= '0' && pC <= '9') || (pC >= 'a' && pC <= 'f') || (pC >= 'A' && pC <= 'F');
}

static std::pair<uint32_t, size_t> fParseHexEscape(const std::string &pStr, size_t pPos, int pHexCount) {
  if (pPos + pHexCount > pStr.length()) {
    return {0, 0};
  }

  uint32_t vValue = 0;
  for (int vI = 0; vI < pHexCount; vI++) {
    char vC = pStr[pPos + vI];
    if (!fIsHexDigit(vC)) {
      return {0, 0};
    }
    vValue <<= 4;
    if ('a' <= vC && vC <= 'f') {
      vValue += vC - 'a' + 10;
    } else if ('A' <= vC && vC <= 'F') {
      vValue += vC - 'A' + 10;
    } else if ('0' <= vC && vC <= '9') {
      vValue += vC - '0';
    } else {
      break;
    }
  }
  return {vValue, static_cast<size_t>(pHexCount)};
}

static std::pair<uint32_t, size_t> fParseCharClassChar(const std::string &pContent, size_t pPos) {
  if (pContent[pPos] == '\\' && pPos + 1 < pContent.length()) {
    switch (pContent[pPos + 1]) {
    case 'x': {
      auto vResult = fParseHexEscape(pContent, pPos + 2, 2);
      if (vResult.second > 0) {
        return {vResult.first, 2 + vResult.second};
      }
      // Invalid escape, treat as literal 'x'
      return {static_cast<uint32_t>('x'), 2};
    }
    case 'u': {
      auto vResult = fParseHexEscape(pContent, pPos + 2, 4);
      if (vResult.second > 0) {
        return {vResult.first, 2 + vResult.second};
      }
      // Invalid escape, treat as literal 'u'
      return {static_cast<uint32_t>('u'), 2};
    }
    case 'U': {
      auto vResult = fParseHexEscape(pContent, pPos + 2, 8);
      if (vResult.second > 0) {
        return {vResult.first, 2 + vResult.second};
      }
      // Invalid escape, treat as literal 'U'
      return {static_cast<uint32_t>('U'), 2};
    }
    case 'n':
      return {'\n', 2};
    case 't':
      return {'\t', 2};
    case 'r':
      return {'\r', 2};
    case '\\':
      return {'\\', 2};
    case ']':
      return {']', 2};
    case '[':
      return {'[', 2};
    default:
      return {static_cast<uint32_t>(pContent[pPos + 1]), 2};
    }
  }

  // Regular character - return as codepoint
  return {static_cast<uint32_t>(static_cast<unsigned char>(pContent[pPos])), 1};
}

static std::pair<std::vector<common_peg_chars_parser::char_range>, bool>
fParseCharClasses(const std::string &pClasses) {
  std::vector<common_peg_chars_parser::char_range> lRanges;
  bool vNegated = false;

  std::string vContent = pClasses;
  if (vContent.front() == '[') {
    vContent = vContent.substr(1);
  }

  if (vContent.back() == ']') {
    vContent.pop_back();
  }

  // Check for negation
  if (!vContent.empty() && vContent.front() == '^') {
    vNegated = true;
    vContent = vContent.substr(1);
  }

  size_t vI = 0;
  while (vI < vContent.length()) {
    auto [vStart, vStartLen] = fParseCharClassChar(vContent, vI);
    vI += vStartLen;

    if (vI + 1 < vContent.length() && vContent[vI] == '-') {
      // Range detected
      auto [vEnd, vEndLen] = fParseCharClassChar(vContent, vI + 1);
      lRanges.push_back(common_peg_chars_parser::char_range{vStart, vEnd});
      vI += 1 + vEndLen;
    } else {
      lRanges.push_back(common_peg_chars_parser::char_range{vStart, vStart});
    }
  }

  return {lRanges, vNegated};
}

common_peg_ast_id common_peg_ast_arena::mFindByTag(const common_peg_ast_node &pParent, const std::string &pTag,
                                                   int pMaxDepth) const {
  for (auto vChildId : pParent.lChildren) {
    const auto &vChild = mGet(vChildId);
    if (vChild.vTag == pTag) {
      return vChildId;
    }
    if (pMaxDepth > 1) {
      auto vResult = mFindByTag(vChild, pTag, pMaxDepth - 1);
      if (vResult != cCommonPegInvalidAstId) {
        return vResult;
      }
    }
  }
  return cCommonPegInvalidAstId;
}

common_peg_ast_id common_peg_ast_arena::mFindByRule(const common_peg_ast_node &pParent, const std::string &pRule,
                                                    int pMaxDepth) const {
  for (auto vChildId : pParent.lChildren) {
    const auto &vChild = mGet(vChildId);
    if (vChild.vRule == pRule) {
      return vChildId;
    }
    if (pMaxDepth > 1) {
      auto vResult = mFindByRule(vChild, pRule, pMaxDepth - 1);
      if (vResult != cCommonPegInvalidAstId) {
        return vResult;
      }
    }
  }
  return cCommonPegInvalidAstId;
}

void common_peg_ast_arena::mVisit(common_peg_ast_id pId, const common_peg_ast_visitor &pVisitor) const {
  if (pId == cCommonPegInvalidAstId) {
    return;
  }
  const auto &vNode = mGet(pId);
  pVisitor(vNode);
  for (const auto &vChild : vNode.lChildren) {
    mVisit(vChild, pVisitor);
  }
}

void common_peg_ast_arena::mVisit(const common_peg_parse_result &pResult,
                                  const common_peg_ast_visitor &pVisitor) const {
  for (const auto &vNode : pResult.lNodes) {
    mVisit(vNode, pVisitor);
  }
}

struct parser_executor;

common_peg_parser_id common_peg_arena::mAddParser(common_peg_parser_variant pParser) {
  common_peg_parser_id vId = lParsers.size();
  lParsers.push_back(std::move(pParser));
  return vId;
}

void common_peg_arena::mAddRule(const std::string &pName, common_peg_parser_id pId) { dRules[pName] = pId; }

common_peg_parser_id common_peg_arena::mGetRule(const std::string &pName) const {
  auto vIt = dRules.find(pName);
  if (vIt == dRules.end()) {
    throw std::runtime_error("Rule not found: " + pName);
  }
  return vIt->second;
}

struct parser_executor {
  const common_peg_arena &vArena;
  common_peg_parse_context &vCtx;
  size_t vStartPos;

  parser_executor(const common_peg_arena &pArena, common_peg_parse_context &pCtx, size_t pStart)
      : vArena(pArena), vCtx(pCtx), vStartPos(pStart) {}

  std::string mDebugIndent() const { return std::string(vCtx.vParseDepth * 2, ' '); }

  std::string mDebugInputSnippet(size_t pPos, size_t pLen = 60) const {
    if (pPos >= vCtx.vInput.size()) {
      return "<EOF>";
    }
    auto vSnippet = vCtx.vInput.substr(pPos, pLen);
    // Escape newlines for display
    std::string vResult;
    for (char vC : vSnippet) {
      if (vC == '\n') {
        vResult += "\\n";
      } else if (vC == '\r') {
        vResult += "\\r";
      } else if (vC == '\t') {
        vResult += "\\t";
      } else {
        vResult += vC;
      }
    }
    if (pPos + pLen < vCtx.vInput.size()) {
      vResult += "...";
    }
    return vResult;
  }

  common_peg_parse_result operator()(const common_peg_epsilon_parser & /* p */) const {
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_start_parser & /* p */) const {
    return common_peg_parse_result(vStartPos == 0 ? cCommonPegParseResultSuccess : cCommonPegParseResultFail,
                                   vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_end_parser & /* p */) const {
    return common_peg_parse_result(
        vStartPos >= vCtx.vInput.size() ? cCommonPegParseResultSuccess : cCommonPegParseResultFail, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_literal_parser &pP) {
    auto vPos = vStartPos;
    for (auto vI = 0u; vI < pP.vLiteral.size(); ++vI) {
      if (vPos >= vCtx.vInput.size()) {
        if (!vCtx.mIsLenient()) {
          return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos);
      }
      if (vCtx.vInput[vPos] != pP.vLiteral[vI]) {
        return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
      }
      ++vPos;
    }

    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
  }

  common_peg_parse_result operator()(const common_peg_sequence_parser &pP) {
    if (vCtx.mIsDebug()) {
      LOG_DBG("%sSEQ start at %zu '%s' (%zu children)\n", mDebugIndent().c_str(), vStartPos,
              mDebugInputSnippet(vStartPos).c_str(), pP.lChildren.size());
    }
    vCtx.vParseDepth++;

    auto vPos = vStartPos;
    std::vector<common_peg_ast_id> lNodes;

    for (size_t vI = 0; vI < pP.lChildren.size(); vI++) {
      const auto &vChildId = pP.lChildren[vI];
      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sSEQ child %zu: %s\n", mDebugIndent().c_str(), vI, vArena.mDump(vChildId).c_str());
      }
      auto vResult = vArena.mParse(vChildId, vCtx, vPos);

      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sSEQ child %zu: %s at %zu->%zu\n", mDebugIndent().c_str(), vI,
                fCommonPegParseResultTypeName(vResult.vType), vResult.vStart, vResult.vEnd);
      }

      if (vResult.mFail()) {
        vCtx.vParseDepth--;
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sSEQ -> FAIL\n", mDebugIndent().c_str());
        }
        return common_peg_parse_result(cCommonPegParseResultFail, vStartPos, vResult.vEnd);
      }

      if (!vResult.lNodes.empty()) {
        lNodes.insert(lNodes.end(), vResult.lNodes.begin(), vResult.lNodes.end());
      }

      if (vResult.mNeedMoreInput()) {
        vCtx.vParseDepth--;
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sSEQ -> NEED_MORE\n", mDebugIndent().c_str());
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vResult.vEnd, std::move(lNodes));
      }

      vPos = vResult.vEnd;
    }

    vCtx.vParseDepth--;
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sSEQ -> SUCCESS at %zu->%zu\n", mDebugIndent().c_str(), vStartPos, vPos);
    }
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos, std::move(lNodes));
  }

  common_peg_parse_result operator()(const common_peg_choice_parser &pP) {
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sCHOICE start at %zu '%s' (%zu options)\n", mDebugIndent().c_str(), vStartPos,
              mDebugInputSnippet(vStartPos).c_str(), pP.lChildren.size());
    }
    vCtx.vParseDepth++;

    auto vPos = vStartPos;
    for (size_t vI = 0; vI < pP.lChildren.size(); vI++) {
      const auto &vChildId = pP.lChildren[vI];
      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sCHOICE option %zu: %s\n", mDebugIndent().c_str(), vI, vArena.mDump(vChildId).c_str());
      }
      auto vResult = vArena.mParse(vChildId, vCtx, vPos);
      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sCHOICE option %zu: %s\n", mDebugIndent().c_str(), vI,
                fCommonPegParseResultTypeName(vResult.vType));
      }
      if (!vResult.mFail()) {
        vCtx.vParseDepth--;
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sCHOICE -> %s (option %zu)\n", mDebugIndent().c_str(),
                  fCommonPegParseResultTypeName(vResult.vType), vI);
        }
        return vResult;
      }
    }

    vCtx.vParseDepth--;
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sCHOICE -> FAIL (no options matched)\n", mDebugIndent().c_str());
    }
    return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_repetition_parser &pP) {
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sREPEAT start at %zu '%s' (min=%d, max=%d)\n", mDebugIndent().c_str(), vStartPos,
              mDebugInputSnippet(vStartPos).c_str(), pP.vMinCount, pP.vMaxCount);
    }
    vCtx.vParseDepth++;

    auto vPos = vStartPos;
    int vMatchCount = 0;
    std::vector<common_peg_ast_id> lNodes;

    // Try to match up to max_count times (or unlimited if max_count is -1)
    while (pP.vMaxCount == -1 || vMatchCount < pP.vMaxCount) {
      if (vPos >= vCtx.vInput.size()) {
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sREPEAT: at end of input, count=%d\n", mDebugIndent().c_str(), vMatchCount);
        }
        break;
      }

      auto vResult = vArena.mParse(pP.vChild, vCtx, vPos);

      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sREPEAT iter %d: %s at %zu->%zu, nodes=%zu\n", mDebugIndent().c_str(), vMatchCount,
                fCommonPegParseResultTypeName(vResult.vType), vResult.vStart, vResult.vEnd, vResult.lNodes.size());
        fprintf(stderr, "%sREPEAT CHILD: %s\n", mDebugIndent().c_str(), vArena.mDump(pP.vChild).c_str());
      }

      if (vResult.mSuccess()) {
        // Prevent infinite loop on empty matches
        if (vResult.vEnd == vPos) {
          if (vCtx.mIsDebug()) {
            fprintf(stderr, "%s  REPEAT: empty match, stopping\n", mDebugIndent().c_str());
          }
          break;
        }

        if (!vResult.lNodes.empty()) {
          lNodes.insert(lNodes.end(), vResult.lNodes.begin(), vResult.lNodes.end());
        }

        vPos = vResult.vEnd;
        vMatchCount++;
        continue;
      }

      if (vResult.mNeedMoreInput()) {
        if (!vResult.lNodes.empty()) {
          lNodes.insert(lNodes.end(), vResult.lNodes.begin(), vResult.lNodes.end());
        }

        vCtx.vParseDepth--;
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sREPEAT -> NEED_MORE (count=%d, nodes=%zu)\n", mDebugIndent().c_str(), vMatchCount,
                  lNodes.size());
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vResult.vEnd, std::move(lNodes));
      }

      // Child failed - stop trying
      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sREPEAT: child failed, stopping\n", mDebugIndent().c_str());
      }
      break;
    }

    // Check if we got enough matches
    if (pP.vMinCount > 0 && vMatchCount < pP.vMinCount) {
      vCtx.vParseDepth--;
      if (vPos >= vCtx.vInput.size() && vCtx.mIsLenient()) {
        if (vCtx.mIsDebug()) {
          fprintf(stderr, "%sREPEAT -> NEED_MORE (not enough matches: %d < %d)\n", mDebugIndent().c_str(), vMatchCount,
                  pP.vMinCount);
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos, std::move(lNodes));
      }
      if (vCtx.mIsDebug()) {
        fprintf(stderr, "%sREPEAT -> FAIL (not enough matches: %d < %d)\n", mDebugIndent().c_str(), vMatchCount,
                pP.vMinCount);
      }
      return common_peg_parse_result(cCommonPegParseResultFail, vStartPos, vPos);
    }

    vCtx.vParseDepth--;
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sREPEAT -> SUCCESS (count=%d, nodes=%zu)\n", mDebugIndent().c_str(), vMatchCount,
              lNodes.size());
    }
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos, std::move(lNodes));
  }

  common_peg_parse_result operator()(const common_peg_and_parser &pP) {
    auto vResult = vArena.mParse(pP.vChild, vCtx, vStartPos);
    // Pass result but don't consume input
    return common_peg_parse_result(vResult.vType, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_not_parser &pP) {
    auto vResult = vArena.mParse(pP.vChild, vCtx, vStartPos);

    if (vResult.mSuccess()) {
      // Fail if the underlying parser matches
      return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
    }

    if (vResult.mNeedMoreInput()) {
      // Propagate - need to know what child would match before negating
      return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos);
    }

    // Child failed, so negation succeeds
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_any_parser & /* p */) const {
    // Parse a single UTF-8 codepoint (not just a single byte)
    auto vResult = fCommonParseUtf8Codepoint(vCtx.vInput, vStartPos);

    if (vResult.vStatus == utf8_parse_result::cIncomplete) {
      if (!vCtx.mIsLenient()) {
        return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
      }
      return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos);
    }
    if (vResult.vStatus == utf8_parse_result::cInvalid) {
      return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
    }
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vStartPos + vResult.vBytesConsumed);
  }

  common_peg_parse_result operator()(const common_peg_space_parser & /* p */) {
    auto vPos = vStartPos;
    while (vPos < vCtx.vInput.size()) {
      auto vC = static_cast<unsigned char>(vCtx.vInput[vPos]);
      if (std::isspace(vC)) {
        ++vPos;
      } else {
        break;
      }
    }

    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
  }

  common_peg_parse_result operator()(const common_peg_chars_parser &pP) const {
    auto vPos = vStartPos;
    int vMatchCount = 0;

    // Try to match up to max_count times (or unlimited if max_count is -1)
    while (pP.vMaxCount == -1 || vMatchCount < pP.vMaxCount) {
      auto vResult = fCommonParseUtf8Codepoint(vCtx.vInput, vPos);

      if (vResult.vStatus == utf8_parse_result::cIncomplete) {
        if (vMatchCount >= pP.vMinCount) {
          // We have enough matches, succeed with what we have
          return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
        }
        // Not enough matches yet
        if (!vCtx.mIsLenient()) {
          return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos);
      }

      if (vResult.vStatus == utf8_parse_result::cInvalid) {
        // Malformed UTF-8 in input
        if (vMatchCount >= pP.vMinCount) {
          // We have enough matches, succeed up to here
          return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
        }
        // Not enough matches, fail
        return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
      }

      // Check if this codepoint matches our character class
      bool vMatches = false;
      for (const auto &vRange : pP.lRanges) {
        if (vRange.mContains(vResult.vCodepoint)) {
          vMatches = true;
          break;
        }
      }

      // If negated, invert the match result
      if (pP.vNegated) {
        vMatches = !vMatches;
      }

      if (vMatches) {
        vPos += vResult.vBytesConsumed;
        ++vMatchCount;
      } else {
        // Character doesn't match, stop matching
        break;
      }
    }

    // Check if we got enough matches
    if (vMatchCount < pP.vMinCount) {
      if (vPos >= vCtx.vInput.size() && vCtx.mIsLenient()) {
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos);
      }
      return common_peg_parse_result(cCommonPegParseResultFail, vStartPos, vPos);
    }

    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
  }

  static common_peg_parse_result mHandleEscapeSequence(common_peg_parse_context &pCtx, size_t pStart, size_t &pPos,
                                                       const char pDelimiter) {
    ++pPos; // consume '\'
    if (pPos >= pCtx.vInput.size()) {
      if (!pCtx.mIsLenient()) {
        return common_peg_parse_result(cCommonPegParseResultFail, pStart);
      }
      return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, pStart, pPos);
    }

    char vC = pCtx.vInput[pPos];
    if (vC == pDelimiter || vC == '\\' || vC == '/' || vC == 'b' || vC == 'f' || vC == 'n' || vC == 'r' || vC == 't') {
      ++pPos;
      return common_peg_parse_result(cCommonPegParseResultSuccess, pStart, pPos);
    } else if (vC == 'u') {
      return mHandleUnicodeEscape(pCtx, pStart, pPos);
    } else {
      return common_peg_parse_result(cCommonPegParseResultFail, pStart);
    }
  }

  static common_peg_parse_result mHandleUnicodeEscape(common_peg_parse_context &pCtx, size_t pStart, size_t &pPos) {
    ++pPos; // consume 'u'
    for (int vI = 0; vI < 4; ++vI) {
      if (pPos >= pCtx.vInput.size()) {
        if (!pCtx.mIsLenient()) {
          return common_peg_parse_result(cCommonPegParseResultFail, pStart);
        }
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, pStart, pPos);
      }
      if (!fIsHexDigit(pCtx.vInput[pPos])) {
        return common_peg_parse_result(cCommonPegParseResultFail, pStart);
      }
      ++pPos;
    }
    return common_peg_parse_result(cCommonPegParseResultSuccess, pStart, pPos);
  }

  common_peg_parse_result operator()(const common_peg_string_parser &pP) {
    auto vPos = vStartPos;

    // Parse string content (without quotes)
    while (vPos < vCtx.vInput.size()) {
      char vC = vCtx.vInput[vPos];

      if (vC == pP.vDelimiter) {
        // Found closing delimiter - success (don't consume it)
        return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
      }

      if (vC == '\\') {
        auto vResult = mHandleEscapeSequence(vCtx, vStartPos, vPos, pP.vDelimiter);
        if (!vResult.mSuccess()) {
          return vResult;
        }
      } else {
        auto vUtf8Result = fCommonParseUtf8Codepoint(vCtx.vInput, vPos);

        if (vUtf8Result.vStatus == utf8_parse_result::cIncomplete) {
          if (!vCtx.mIsLenient()) {
            return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
          }
          return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos);
        }

        if (vUtf8Result.vStatus == utf8_parse_result::cInvalid) {
          return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
        }

        vPos += vUtf8Result.vBytesConsumed;
      }
    }

    // Reached end without finding closing quote
    if (!vCtx.mIsLenient()) {
      return common_peg_parse_result(cCommonPegParseResultFail, vStartPos, vPos);
    }
    return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vPos);
  }

  common_peg_parse_result operator()(const common_peg_until_parser &pP) const {
    common_trie vMatcher(pP.lDelimiters);

    // Scan input and check for delimiters
    size_t vPos = vStartPos;
    size_t vLastValidPos = vStartPos;

    while (vPos < vCtx.vInput.size()) {
      auto vUtf8Result = fCommonParseUtf8Codepoint(vCtx.vInput, vPos);

      if (vUtf8Result.vStatus == utf8_parse_result::cIncomplete) {
        // Incomplete UTF-8 sequence
        if (!vCtx.mIsLenient()) {
          // Input is complete but UTF-8 is incomplete = malformed
          return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
        }
        // Return what we have so far (before incomplete sequence)
        return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vLastValidPos);
      }

      if (vUtf8Result.vStatus == utf8_parse_result::cInvalid) {
        // Malformed UTF-8
        return common_peg_parse_result(cCommonPegParseResultFail, vStartPos);
      }

      // Check if a delimiter starts at this position
      auto vMatch = vMatcher.mCheckAt(vCtx.vInput, vPos);

      if (vMatch == common_trie::cCompleteMatch) {
        // Found a complete delimiter, return everything before it
        return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
      }

      if (vMatch == common_trie::cPartialMatch) {
        // Found a partial match extending to end of input, return everything before it
        return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vPos);
      }

      vPos += vUtf8Result.vBytesConsumed;
      vLastValidPos = vPos;
    }

    if (vLastValidPos == vCtx.vInput.size() && vCtx.mIsLenient()) {
      // Reached the end of a partial stream, there might still be more input that we need to consume.
      return common_peg_parse_result(cCommonPegParseResultNeedMoreInput, vStartPos, vLastValidPos);
    }
    return common_peg_parse_result(cCommonPegParseResultSuccess, vStartPos, vLastValidPos);
  }

  common_peg_parse_result operator()(const common_peg_schema_parser &pP) {
    return vArena.mParse(pP.vChild, vCtx, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_rule_parser &pP) {
    // Parse the child
    auto vResult = vArena.mParse(pP.vChild, vCtx, vStartPos);

    if (!vResult.mFail()) {
      std::string_view vText;
      if (vResult.vStart < vCtx.vInput.size()) {
        vText = std::string_view(vCtx.vInput).substr(vResult.vStart, vResult.vEnd - vResult.vStart);
      }

      auto vNodeId = vCtx.vAst.mAddNode(pP.vName, "", vResult.vStart, vResult.vEnd, vText, std::move(vResult.lNodes),
                                        vResult.mNeedMoreInput());

      return common_peg_parse_result(vResult.vType, vResult.vStart, vResult.vEnd, {vNodeId});
    }

    return vResult;
  }

  common_peg_parse_result operator()(const common_peg_tag_parser &pP) {
    // Parse the child
    if (vCtx.mIsDebug()) {
      fprintf(stderr, "%sTAG: %s\n", mDebugIndent().c_str(), pP.vTag.c_str());
    }
    auto vResult = vArena.mParse(pP.vChild, vCtx, vStartPos);

    if (!vResult.mFail()) {
      std::string_view vText;
      if (vResult.vStart < vCtx.vInput.size()) {
        vText = std::string_view(vCtx.vInput).substr(vResult.vStart, vResult.vEnd - vResult.vStart);
      }

      auto vNodeId = vCtx.vAst.mAddNode("", pP.vTag, vResult.vStart, vResult.vEnd, vText, std::move(vResult.lNodes),
                                        vResult.mNeedMoreInput());

      return common_peg_parse_result(vResult.vType, vResult.vStart, vResult.vEnd, {vNodeId});
    }

    return vResult;
  }

  common_peg_parse_result operator()(const common_peg_ref_parser &pP) {
    auto vRuleId = vArena.mGetRule(pP.vName);
    return vArena.mParse(vRuleId, vCtx, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_atomic_parser &pP) {
    auto vResult = vArena.mParse(pP.vChild, vCtx, vStartPos);
    if (vResult.mNeedMoreInput()) {
      // Clear nodes so they don't propagate up.
      vResult.lNodes.clear();
    }
    return vResult;
  }

  common_peg_parse_result operator()(const common_peg_gbnf_parser &pP) {
    return vArena.mParse(pP.vChild, vCtx, vStartPos);
  }

  common_peg_parse_result operator()(const common_peg_ac_parser &pP) {
    return vArena.mParse(pP.vChild, vCtx, vStartPos);
  }
};

common_peg_parse_result common_peg_arena::mParse(common_peg_parse_context &pCtx, size_t pStart) const {
  if (vRoot == cCommonPegInvalidParserId) {
    throw std::runtime_error("No root parser set");
  }
  return mParse(vRoot, pCtx, pStart);
}

common_peg_parse_result common_peg_arena::mParse(common_peg_parser_id pId, common_peg_parse_context &pCtx,
                                                 size_t pStart) const {
  // Execute parser
  const auto &vParser = lParsers.at(pId);
  parser_executor vExec(*this, pCtx, pStart);
  return std::visit(vExec, vParser);
}

common_peg_parser_id common_peg_arena::mResolveRef(common_peg_parser_id pId) {
  const auto &vParser = lParsers.at(pId);
  if (auto vRef = std::get_if<common_peg_ref_parser>(&vParser)) {
    return mGetRule(vRef->vName);
  }
  return pId;
}

static void fBfsNode(common_peg_ast_arena &pArena, std::ostringstream &pOss, const common_peg_ast_node &pNode,
                     int pIndent) {
  for (int vI = 0; vI < pIndent; vI++) {
    pOss << "  ";
  }
  pOss << "NODE " << pNode.vId;
  if (!pNode.vRule.empty()) {
    pOss << " (rule " << pNode.vRule << ")";
  }
  if (!pNode.vTag.empty()) {
    pOss << " (tag " << pNode.vTag << ")";
  }
  pOss << " ['" << pNode.vText << "']\n";
  for (const auto cChild : pNode.lChildren) {
    fBfsNode(pArena, pOss, pArena.mGet(cChild), pIndent + 1);
  }
}

std::string common_peg_ast_arena::mDump() {
  std::ostringstream vOss;
  for (auto &vNode : lNodes) {
    fBfsNode(*this, vOss, vNode, 0);
  }
  return vOss.str();
}

void common_peg_arena::mResolveRefs() {
  // Walk through all parsers and replace refs with their corresponding rule IDs
  for (auto &vParser : lParsers) {
    std::visit(
        [this](auto &pP) {
          using T = std::decay_t<decltype(pP)>;

          if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            for (auto &vChild : pP.lChildren) {
              vChild = mResolveRef(vChild);
            }
          } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            for (auto &vChild : pP.lChildren) {
              vChild = mResolveRef(vChild);
            }
          } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                               std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser> ||
                               std::is_same_v<T, common_peg_tag_parser> ||
                               std::is_same_v<T, common_peg_atomic_parser> ||
                               std::is_same_v<T, common_peg_gbnf_parser> || std::is_same_v<T, common_peg_ac_parser>) {
            pP.vChild = mResolveRef(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            pP.vChild = mResolveRef(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            pP.vChild = mResolveRef(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                               std::is_same_v<T, common_peg_start_parser> || std::is_same_v<T, common_peg_end_parser> ||
                               std::is_same_v<T, common_peg_ref_parser> || std::is_same_v<T, common_peg_until_parser> ||
                               std::is_same_v<T, common_peg_literal_parser> ||
                               std::is_same_v<T, common_peg_string_parser> ||
                               std::is_same_v<T, common_peg_chars_parser> || std::is_same_v<T, common_peg_any_parser> ||
                               std::is_same_v<T, common_peg_space_parser>) {
            // These rules do not have children
          } else {
            static_assert(cIsAlwaysFalseV<T>);
          }
        },
        vParser);
  }

  // Also flatten root if it's a ref
  if (vRoot != cCommonPegInvalidParserId) {
    vRoot = mResolveRef(vRoot);
  }
}

std::string common_peg_arena::mDump(common_peg_parser_id pId) const {
  std::set<common_peg_parser_id> sVisited;
  return mDumpImpl(pId, sVisited);
}

std::string common_peg_arena::mDumpImpl(common_peg_parser_id pId, std::set<common_peg_parser_id> &pVisited) const {
  // Check for cycles
  if (pVisited.count(pId)) {
    return "[cycle]";
  }
  pVisited.insert(pId);

  const auto &vParser = lParsers.at(pId);

  return std::visit(
      [this, &pVisited](const auto &pP) -> std::string {
        using T = std::decay_t<decltype(pP)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
          return "Epsilon";
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
          return "Start";
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
          return "End";
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
          return "Literal(" + pP.vLiteral + ")";
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
          std::vector<std::string> lParts;
          for (const auto &vChild : pP.lChildren) {
            lParts.push_back(mDumpImpl(vChild, pVisited));
          }
          return "Sequence(" + fStringJoin(lParts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
          std::vector<std::string> lParts;
          for (const auto &vChild : pP.lChildren) {
            lParts.push_back(mDumpImpl(vChild, pVisited));
          }
          return "Choice(" + fStringJoin(lParts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
          if (pP.vMaxCount == -1) {
            return "Repetition(" + mDumpImpl(pP.vChild, pVisited) + ", " + std::to_string(pP.vMinCount) +
                   ", unbounded)";
          }
          return "Repetition(" + mDumpImpl(pP.vChild, pVisited) + ", " + std::to_string(pP.vMinCount) + ", " +
                 std::to_string(pP.vMaxCount) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
          return "And(" + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
          return "Not(" + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
          return "Atomic(" + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
          return "Gbnf(" + pP.vGrammar + ", " + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_ac_parser>) {
          return "Ac(" + fStringJoin(pP.lDelimiters, " | ") + ", " + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
          return "Any";
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
          return "Space";
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
          if (pP.vMaxCount == -1) {
            return "CharRepeat(" + pP.vPattern + ", " + std::to_string(pP.vMinCount) + ", unbounded)";
          }
          return "CharRepeat(" + pP.vPattern + ", " + std::to_string(pP.vMinCount) + ", " +
                 std::to_string(pP.vMaxCount) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
          return "String(" + std::string(1, pP.vDelimiter) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
          return "Until(" + fStringJoin(pP.lDelimiters, " | ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
          return "Schema(" + mDumpImpl(pP.vChild, pVisited) + ", " + (pP.vSchema ? pP.vSchema->dump() : "null") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
          return "Rule(" + pP.vName + ", " + mDumpImpl(pP.vChild, pVisited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
          return "Ref(" + pP.vName + ")";
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
          return "Tag(" + pP.vTag + ", " + mDump(pP.vChild) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
          return "Atomic(" + mDump(pP.child) + ")";
        } else {
          return "Unknown";
        }
      },
      vParser);
}

common_peg_parser &common_peg_parser::operator=(const common_peg_parser &pOther) {
  vId = pOther.vId;
  return *this;
}

common_peg_parser &common_peg_parser::operator+=(const common_peg_parser &pOther) {
  vId = vBuilder.mSequence({vId, pOther.vId});
  return *this;
}

common_peg_parser &common_peg_parser::operator|=(const common_peg_parser &pOther) {
  vId = vBuilder.mChoice({vId, pOther.vId});
  return *this;
}

common_peg_parser common_peg_parser::operator+(const common_peg_parser &pOther) const {
  return vBuilder.mSequence({vId, pOther.vId});
}

common_peg_parser common_peg_parser::operator|(const common_peg_parser &pOther) const {
  return vBuilder.mChoice({vId, pOther.vId});
}

common_peg_parser common_peg_parser::operator<<(const common_peg_parser &pOther) const {
  return vBuilder.mSequence({vId, vBuilder.mSpace(), pOther.vId});
}

common_peg_parser common_peg_parser::operator+(const char *pStr) const { return *this + vBuilder.mLiteral(pStr); }

common_peg_parser common_peg_parser::operator+(const std::string &pStr) const {
  return *this + vBuilder.mLiteral(pStr);
}

common_peg_parser common_peg_parser::operator<<(const char *pStr) const { return *this << vBuilder.mLiteral(pStr); }

common_peg_parser common_peg_parser::operator<<(const std::string &pStr) const {
  return *this << vBuilder.mLiteral(pStr);
}

common_peg_parser common_peg_parser::operator|(const char *pStr) const { return *this | vBuilder.mLiteral(pStr); }

common_peg_parser common_peg_parser::operator|(const std::string &pStr) const {
  return *this | vBuilder.mLiteral(pStr);
}

common_peg_parser operator+(const char *pStr, const common_peg_parser &pP) { return pP.mBuilder().mLiteral(pStr) + pP; }

common_peg_parser operator+(const std::string &pStr, const common_peg_parser &pP) {
  return operator+(pStr.c_str(), pP);
}

common_peg_parser operator<<(const char *pStr, const common_peg_parser &pP) {
  return pP.mBuilder().mLiteral(pStr) << pP;
}

common_peg_parser operator<<(const std::string &pStr, const common_peg_parser &pP) {
  return operator<<(pStr.c_str(), pP);
}

common_peg_parser operator|(const char *pStr, const common_peg_parser &pP) { return pP.mBuilder().mLiteral(pStr) | pP; }

common_peg_parser operator|(const std::string &pStr, const common_peg_parser &pP) {
  return operator|(pStr.c_str(), pP);
}

static std::string fRuleName(const std::string &pName) {
  static const std::regex cInvalidRuleCharsRe("[^a-zA-Z0-9-]+");
  return std::regex_replace(pName, cInvalidRuleCharsRe, "-");
}

common_peg_parser_builder::common_peg_parser_builder() {}

common_peg_parser common_peg_parser_builder::mSequence(const std::vector<common_peg_parser_id> &pParsers) {
  // Flatten nested sequences
  std::vector<common_peg_parser_id> lFlattened;
  for (const auto &vP : pParsers) {
    const auto &vParser = vArena.mGet(vP);
    if (auto vSeq = std::get_if<common_peg_sequence_parser>(&vParser)) {
      lFlattened.insert(lFlattened.end(), vSeq->lChildren.begin(), vSeq->lChildren.end());
    } else {
      lFlattened.push_back(vP);
    }
  }
  return mWrap(vArena.mAddParser(common_peg_sequence_parser{lFlattened}));
}

common_peg_parser common_peg_parser_builder::mSequence(const std::vector<common_peg_parser> &pParsers) {
  std::vector<common_peg_parser_id> lIds;
  lIds.reserve(pParsers.size());
  for (const auto &vP : pParsers) {
    lIds.push_back(vP.mId());
  }
  return mSequence(lIds);
}

common_peg_parser common_peg_parser_builder::mSequence(std::initializer_list<common_peg_parser> pParsers) {
  std::vector<common_peg_parser_id> lIds;
  lIds.reserve(pParsers.size());
  for (const auto &vP : pParsers) {
    lIds.push_back(vP.mId());
  }
  return mSequence(lIds);
}

common_peg_parser common_peg_parser_builder::mChoice(const std::vector<common_peg_parser_id> &pParsers) {
  // Flatten nested choices
  std::vector<common_peg_parser_id> lFlattened;
  for (const auto &vP : pParsers) {
    const auto &vParser = vArena.mGet(vP);
    if (auto vChoice = std::get_if<common_peg_choice_parser>(&vParser)) {
      lFlattened.insert(lFlattened.end(), vChoice->lChildren.begin(), vChoice->lChildren.end());
    } else {
      lFlattened.push_back(vP);
    }
  }
  return mWrap(vArena.mAddParser(common_peg_choice_parser{lFlattened}));
}

common_peg_parser common_peg_parser_builder::mChoice(const std::vector<common_peg_parser> &pParsers) {
  std::vector<common_peg_parser_id> lIds;
  lIds.reserve(pParsers.size());
  for (const auto &vP : pParsers) {
    lIds.push_back(vP.mId());
  }
  return mChoice(lIds);
}

common_peg_parser common_peg_parser_builder::mChoice(std::initializer_list<common_peg_parser> pParsers) {
  std::vector<common_peg_parser_id> lIds;
  lIds.reserve(pParsers.size());
  for (const auto &vP : pParsers) {
    lIds.push_back(vP.mId());
  }
  return mChoice(lIds);
}

common_peg_parser common_peg_parser_builder::mChars(const std::string &pClasses, int pMin, int pMax) {
  auto [vRanges, vNegated] = fParseCharClasses(pClasses);
  return mWrap(vArena.mAddParser(common_peg_chars_parser{pClasses, vRanges, vNegated, pMin, pMax}));
}

common_peg_parser common_peg_parser_builder::mSchema(const common_peg_parser &pP, const std::string &pName,
                                                     const nlohmann::ordered_json &pSchema, bool pRaw) {
  return mWrap(vArena.mAddParser(
      common_peg_schema_parser{pP.mId(), pName, std::make_shared<nlohmann::ordered_json>(pSchema), pRaw}));
}

common_peg_parser common_peg_parser_builder::mRule(const std::string &pName, const common_peg_parser &pP,
                                                   bool pTrigger) {
  auto vCleanName = fRuleName(pName);
  auto vRuleId = vArena.mAddParser(common_peg_rule_parser{vCleanName, pP.mId(), pTrigger});
  vArena.mAddRule(vCleanName, vRuleId);
  return mRef(vCleanName);
}

common_peg_parser common_peg_parser_builder::mRule(const std::string &pName,
                                                   const std::function<common_peg_parser()> &pBuilderFn,
                                                   bool pTrigger) {
  auto vCleanName = fRuleName(pName);
  if (vArena.mHasRule(vCleanName)) {
    return mRef(vCleanName);
  }

  // Create placeholder rule to allow recursive references
  auto vPlaceholder = mAny(); // Temporary placeholder
  auto vPlaceholderRuleId = vArena.mAddParser(common_peg_rule_parser{vCleanName, vPlaceholder.mId(), pTrigger});
  vArena.mAddRule(vCleanName, vPlaceholderRuleId);

  // Build the actual parser
  auto vParser = pBuilderFn();

  // Replace placeholder with actual rule
  auto vRuleId = vArena.mAddParser(common_peg_rule_parser{vCleanName, vParser.mId(), pTrigger});
  vArena.dRules[vCleanName] = vRuleId;

  return mRef(vCleanName);
}

void common_peg_parser_builder::mSetRoot(const common_peg_parser &pP) { vArena.mSetRoot(pP.mId()); }

common_peg_arena common_peg_parser_builder::mBuild() {
  vArena.mResolveRefs();
  return std::move(vArena);
}

// String primitives

common_peg_parser common_peg_parser_builder::mStringContent(char pDelimiter) {
  return mWrap(vArena.mAddParser(common_peg_string_parser{pDelimiter}));
}

common_peg_parser common_peg_parser_builder::mDoubleQuotedString() {
  return mRule("double-quoted-string",
               [this]() { return mSequence({mLiteral("\""), mStringContent('"'), mLiteral("\"")}); });
}

common_peg_parser common_peg_parser_builder::mSingleQuotedString() {
  return mRule("single-quoted-string",
               [this]() { return mSequence({mLiteral("'"), mStringContent('\''), mLiteral("'")}); });
}

common_peg_parser common_peg_parser_builder::mQuotedString() {
  return mRule("quoted-string", [this]() { return mChoice({mDoubleQuotedString(), mSingleQuotedString()}); });
}

// JSON parsers

common_peg_parser common_peg_parser_builder::mJsonNumber() {
  return mRule("json-number", [this]() {
    auto vDigit1Point9 = mChars("[1-9]", 1, 1);
    auto vDigits = mChars("[0-9]");
    auto vIntPart = mChoice({mLiteral("0"), mSequence({vDigit1Point9, mChars("[0-9]", 0, -1)})});
    auto vFrac = mSequence({mLiteral("."), vDigits});
    auto vExp = mSequence({mChoice({mLiteral("e"), mLiteral("E")}), mOptional(mChars("[+-]", 1, 1)), vDigits});
    // Negative lookahead: only commit the number when the next character can't extend it.
    // At EOF in partial mode, chars returns NEED_MORE → negate propagates NEED_MORE → number not committed.
    // This prevents premature commits of partial numbers (e.g. "3" when "3.14" is incoming).
    auto vNotNumberContinuation = mNegate(mChars("[0-9.eE+-]", 1, 1));
    return mSequence({mOptional(mLiteral("-")), vIntPart, mOptional(vFrac), mOptional(vExp), vNotNumberContinuation});
  });
}

common_peg_parser common_peg_parser_builder::mJsonString() {
  return mRule("json-string", [this]() { return mSequence({mLiteral("\""), mStringContent('"'), mLiteral("\"")}); });
}

common_peg_parser common_peg_parser_builder::mJsonBool() {
  return mRule("json-bool", [this]() { return mChoice({mLiteral("true"), mLiteral("false")}); });
}

common_peg_parser common_peg_parser_builder::mJsonNull() {
  return mRule("json-null", [this]() { return mLiteral("null"); });
}

common_peg_parser common_peg_parser_builder::mJsonObject() {
  return mRule("json-object", [this]() {
    auto vWs = mSpace();
    auto vMember = mSequence({mJsonString(), vWs, mLiteral(":"), vWs, mJson()});
    auto vMembers = mSequence({vMember, mZeroOrMore(mSequence({vWs, mLiteral(","), vWs, vMember}))});
    return mSequence({mLiteral("{"), vWs, mChoice({mLiteral("}"), mSequence({vMembers, vWs, mLiteral("}")})})});
  });
}

common_peg_parser common_peg_parser_builder::mJsonArray() {
  return mRule("json-array", [this]() {
    auto vWs = mSpace();
    auto vElements = mSequence({mJson(), mZeroOrMore(mSequence({vWs, mLiteral(","), vWs, mJson()}))});
    return mSequence({mLiteral("["), vWs, mChoice({mLiteral("]"), mSequence({vElements, vWs, mLiteral("]")})})});
  });
}

common_peg_parser common_peg_parser_builder::mJson() {
  return mRule("json-value", [this]() {
    return mChoice({mJsonObject(), mJsonArray(), mJsonString(), mJsonNumber(), mJsonBool(), mJsonNull()});
  });
}

common_peg_parser common_peg_parser_builder::mPythonString() {
  return mRule("python-string", [this]() { return mChoice({mDoubleQuotedString(), mSingleQuotedString()}); });
}

common_peg_parser common_peg_parser_builder::mPythonNumber() { return mJsonNumber(); }

common_peg_parser common_peg_parser_builder::mPythonBool() {
  return mRule("python-bool", [this]() { return mChoice({mLiteral("True"), mLiteral("False")}); });
}

common_peg_parser common_peg_parser_builder::mPythonNull() {
  return mRule("python-none", [this]() { return mLiteral("None"); });
}

common_peg_parser common_peg_parser_builder::mPythonDict() {
  return mRule("python-dict", [this]() {
    auto vWs = mSpace();
    auto vMember = mSequence({mPythonString(), vWs, mLiteral(":"), vWs, mPythonValue()});
    auto vMembers = mSequence({vMember, mZeroOrMore(mSequence({vWs, mLiteral(","), vWs, vMember}))});
    return mSequence({mLiteral("{"), vWs, mChoice({mLiteral("}"), mSequence({vMembers, vWs, mLiteral("}")})}), vWs});
  });
}

common_peg_parser common_peg_parser_builder::mPythonArray() {
  return mRule("python-array", [this]() {
    auto vWs = mSpace();
    auto vElements = mSequence({mPythonValue(), mZeroOrMore(mSequence({mLiteral(","), vWs, mPythonValue()}))});
    return mSequence({mLiteral("["), vWs, mChoice({mLiteral("]"), mSequence({vElements, vWs, mLiteral("]")})}), vWs});
  });
}

common_peg_parser common_peg_parser_builder::mPythonValue() {
  return mRule("python-value", [this]() {
    return mChoice({mPythonDict(), mPythonArray(), mPythonString(), mPythonNumber(), mPythonBool(), mPythonNull()});
  });
}

common_peg_parser common_peg_parser_builder::mMarker() {
  auto vSharpBracketParser = mLiteral("<") + mUntil(">") + mLiteral(">");
  auto vSquareBracketParser = mLiteral("[") + mUntil("]") + mLiteral("]");
  return mChoice({vSharpBracketParser, vSquareBracketParser});
}

common_peg_parser common_peg_parser_builder::mJsonMember(const std::string &pKey, const common_peg_parser &pP) {
  auto vWs = mSpace();
  return mSequence({
      mLiteral("\"" + pKey + "\""),
      vWs,
      mLiteral(":"),
      vWs,
      pP,
  });
}

common_peg_parser common_peg_parser_builder::mAc(const common_peg_parser &pP,
                                                 const std::vector<std::string> &pDelimiters) {
  if (pDelimiters.empty()) {
    throw std::runtime_error("ac parser requires at least one delimiter");
  }
  return mAdd(common_peg_ac_parser{pP, pDelimiters});
}

static std::string fGbnfEscapeCharClass(uint32_t pC) {
  if (pC == '-' || pC == ']' || pC == '[' || pC == '\\') {
    return "\\" + std::string(1, (char)pC);
  }
  // Escape whitespace control characters
  if (pC == '\n') {
    return "\\n";
  }
  if (pC == '\t') {
    return "\\t";
  }
  if (pC == '\r') {
    return "\\r";
  }

  // Printable ASCII
  if (pC >= 0x20 && pC <= 0x7E) {
    return std::string(1, (char)pC);
  }

  // Hex escape
  char aBuf[16];
  const char *vHex = "0123456789ABCDEF";

  if (pC <= 0xFF) {
    aBuf[0] = '\\';
    aBuf[1] = 'x';
    aBuf[2] = vHex[(pC >> 4) & 0xF];
    aBuf[3] = vHex[pC & 0xF];
    aBuf[4] = '\0';
  } else if (pC <= 0xFFFF) {
    aBuf[0] = '\\';
    aBuf[1] = 'u';
    aBuf[2] = vHex[(pC >> 12) & 0xF];
    aBuf[3] = vHex[(pC >> 8) & 0xF];
    aBuf[4] = vHex[(pC >> 4) & 0xF];
    aBuf[5] = vHex[pC & 0xF];
    aBuf[6] = '\0';
  } else {
    aBuf[0] = '\\';
    aBuf[1] = 'U';
    for (int vI = 0; vI < 8; vI++) {
      aBuf[2 + vI] = vHex[(pC >> ((7 - vI) * 4)) & 0xF];
    }
    aBuf[10] = '\0';
  }

  return std::string(aBuf);
}

static std::string fGbnfCharClass(const std::vector<uint32_t> &pChars, bool pNegate) {
  std::string vS = pNegate ? "[^" : "[";
  for (uint32_t vCh : pChars) {
    vS += fGbnfEscapeCharClass(vCh);
  }
  return vS + "]";
}

static std::string fGbnfAcGrammar(
    const common_grammar_builder &pBuilder, const std::string &pPrefix, const std::vector<std::string> &pStrings,
    const std::function<std::string(const std::vector<uint32_t> &, const std::map<size_t, std::vector<uint32_t>> &,
                                    const std::vector<uint32_t> &, const std::function<std::string(size_t)> &)>
        &pBuildRule) {
  common_aho_corasick vAc(pStrings);

  auto vStateName = [&](size_t pS) -> std::string {
    if (pS == 0) {
      return pPrefix;
    }
    std::string vNum = std::to_string(pS);
    vNum = vNum.size() == 1 ? ("0" + vNum) : vNum;
    return pPrefix + "-" + vNum;
  };

  for (size_t vQ = 0; vQ < vAc.mNumStates(); vQ++) {
    if (vAc.mIsTerminal(vQ)) {
      continue; // match states
    }

    std::map<size_t, std::vector<uint32_t>> dBuckets;
    std::vector<uint32_t> lCompleting; // chars that complete a delimiter
    std::vector<uint32_t> lSpecific;   // chars with an explicit transition
    for (uint32_t vC : vAc.sAlphabet) {
      size_t vD = vAc.mNext(vQ, vC);
      if (vAc.mIsTerminal(vD)) {
        lCompleting.push_back(vC);
        lSpecific.push_back(vC);
      } else if (vD != 0) {
        dBuckets[vD].push_back(vC); // specific non-root destination
        lSpecific.push_back(vC);
      }
    }

    pBuilder.vAddRule(vStateName(vQ), pBuildRule(lCompleting, dBuckets, lSpecific, vStateName));
  }

  // An empty delimiter makes the start state terminal. Emit an entry rule
  // that matches the empty string so the returned reference stays valid.
  if (vAc.mIsTerminal(0)) {
    pBuilder.vAddRule(pPrefix, "|");
  }

  return vStateName(0);
}

// GBNF grammar matching strings that contain no string in `pStrings` as a
// substring. Emits the complement of an Aho-Corasick automaton DFA and returns
// the start state rule name.
static std::string fGbnfExcludingGrammar(const common_grammar_builder &pBuilder, const std::string &pPrefix,
                                         const std::vector<std::string> &pStrings) {
  return fGbnfAcGrammar(
      pBuilder, pPrefix, pStrings,
      [](const std::vector<uint32_t> & /*completing*/, const std::map<size_t, std::vector<uint32_t>> &pBuckets,
         const std::vector<uint32_t> &pSpecific, const std::function<std::string(size_t)> &pStateName) {
        // every state is accepting and completing chars get no
        // alternative, so a forbidden string can never be matched
        std::string vRhs = "|";
        for (const auto &[vD, vChars] : pBuckets) {
          vRhs += " " + fGbnfCharClass(vChars, false) + " " + pStateName(vD) + " |";
        }
        vRhs += " " + fGbnfCharClass(pSpecific, true) + " " + pStateName(0);
        return vRhs;
      });
}

// GBNF grammar matching everything up to and including the first occurrence of
// any string in `pStrings`. Emits the Aho-Corasick automaton DFA and returns
// the start state rule name.
static std::string fGbnfIncludingGrammar(const common_grammar_builder &pBuilder, const std::string &pPrefix,
                                         const std::vector<std::string> &pStrings) {
  return fGbnfAcGrammar(
      pBuilder, pPrefix, pStrings,
      [](const std::vector<uint32_t> &pCompleting, const std::map<size_t, std::vector<uint32_t>> &pBuckets,
         const std::vector<uint32_t> &pSpecific, const std::function<std::string(size_t)> &pStateName) {
        std::vector<std::string> lAlts;
        if (!pCompleting.empty()) {
          lAlts.push_back(fGbnfCharClass(pCompleting, false)); // terminate on match
        }
        for (const auto &[vD, vChars] : pBuckets) {
          lAlts.push_back(fGbnfCharClass(vChars, false) + " " + pStateName(vD));
        }
        // every other character keeps scanning from the start state
        lAlts.push_back(fGbnfCharClass(pSpecific, true) + " " + pStateName(0));
        return fStringJoin(lAlts, " | ");
      });
}

static std::set<std::string> fCollectReachableRules(const common_peg_arena &pArena, const common_peg_parser_id &pRule) {
  std::set<std::string> sReachable;
  std::set<std::string> sVisited;

  std::function<void(common_peg_parser_id)> vVisit = [&](common_peg_parser_id pId) {
    const auto &vParser = pArena.mGet(pId);

    std::visit(
        [&](const auto &pP) {
          using T = std::decay_t<decltype(pP)>;

          if constexpr (std::is_same_v<T, common_peg_epsilon_parser> || std::is_same_v<T, common_peg_start_parser> ||
                        std::is_same_v<T, common_peg_end_parser> || std::is_same_v<T, common_peg_until_parser> ||
                        std::is_same_v<T, common_peg_literal_parser> || std::is_same_v<T, common_peg_chars_parser> ||
                        std::is_same_v<T, common_peg_space_parser> || std::is_same_v<T, common_peg_any_parser> ||
                        std::is_same_v<T, common_peg_string_parser>) {
            // These parsers do not have any children
          } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            for (auto vChild : pP.lChildren) {
              vVisit(vChild);
            }
          } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            for (auto vChild : pP.lChildren) {
              vVisit(vChild);
            }
          } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                               std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser> ||
                               std::is_same_v<T, common_peg_tag_parser> ||
                               std::is_same_v<T, common_peg_atomic_parser> ||
                               std::is_same_v<T, common_peg_gbnf_parser> || std::is_same_v<T, common_peg_ac_parser> ||
                               std::is_same_v<T, common_peg_schema_parser>) {
            vVisit(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            if (sVisited.find(pP.vName) == sVisited.end()) {
              sVisited.insert(pP.vName);
              sReachable.insert(pP.vName);
              vVisit(pP.vChild);
            }
          } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            // Traverse rules so we pick up everything
            auto vReferencedRule = pArena.mGetRule(pP.vName);
            vVisit(vReferencedRule);
          } else {
            static_assert(cIsAlwaysFalseV<T>);
          }
        },
        vParser);
  };

  vVisit(pRule);
  return sReachable;
}

// GBNF generation implementation
void common_peg_arena::mBuildGrammar(const common_grammar_builder &pBuilder, bool pLazy) const {
  auto vSchemaDelegates = [](const common_peg_schema_parser &pS) -> bool {
    if (!pS.vSchema) {
      return true;
    }
    if (pS.vRaw && pS.vSchema->contains("type")) {
      const auto &vTypeVal = pS.vSchema->at("type");
      if (vTypeVal.is_string() && vTypeVal == "string") {
        return true;
      }
      // Handle nullable types like ["string", "null"] - delegate when the
      // non-null type is string, since the tagged format uses raw text
      if (vTypeVal.is_array()) {
        for (const auto &vT : vTypeVal) {
          if (vT.is_string() && vT.get<std::string>() != "null") {
            return vT.get<std::string>() == "string";
          }
        }
      }
    }
    // Delegate for enum schemas in raw mode - enum values are literal strings
    if (pS.vRaw && !pS.vSchema->contains("type") && pS.vSchema->contains("enum")) {
      return true;
    }
    return false;
  };

  // Unwrap the parser so we can properly check if it's a sequence or choice
  auto vEffectiveParser = [&](common_peg_parser_id pId) -> const common_peg_parser_variant & {
    while (true) {
      const auto &vP = lParsers.at(pId);
      if (const auto *vTag = std::get_if<common_peg_tag_parser>(&vP)) {
        pId = vTag->vChild;
      } else if (const auto *vAtomic = std::get_if<common_peg_atomic_parser>(&vP)) {
        pId = vAtomic->vChild;
      } else if (const auto *vSchema = std::get_if<common_peg_schema_parser>(&vP)) {
        if (vSchemaDelegates(*vSchema)) {
          pId = vSchema->vChild;
        } else {
          return vP;
        }
      } else {
        return vP;
      }
    }
  };

  // Generate GBNF for a parser
  std::function<std::string(common_peg_parser_id)> vToGbnf = [&](common_peg_parser_id pId) -> std::string {
    const auto &vParser = lParsers.at(pId);

    return std::visit(
        [&](const auto &pP) -> std::string {
          using T = std::decay_t<decltype(pP)>;

          if constexpr (std::is_same_v<T, common_peg_epsilon_parser> || std::is_same_v<T, common_peg_start_parser> ||
                        std::is_same_v<T, common_peg_end_parser>) {
            return "";
          } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return fGbnfFormatLiteral(pP.vLiteral);
          } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            std::string vS;
            for (const auto &vChild : pP.lChildren) {
              auto vChildGbnf = vToGbnf(vChild);
              if (vChildGbnf.empty()) {
                continue;
              }
              if (!vS.empty()) {
                vS += " ";
              }
              const auto &vChildParser = vEffectiveParser(vChild);
              if (std::holds_alternative<common_peg_choice_parser>(vChildParser) ||
                  std::holds_alternative<common_peg_sequence_parser>(vChildParser)) {
                vS += "(" + vChildGbnf + ")";
              } else {
                vS += vChildGbnf;
              }
            }
            return vS;
          } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            std::string vS;
            for (const auto &vChild : pP.lChildren) {
              if (!vS.empty()) {
                vS += " | ";
              }
              auto vChildGbnf = vToGbnf(vChild);
              const auto &vChildParser = vEffectiveParser(vChild);
              if (std::holds_alternative<common_peg_choice_parser>(vChildParser)) {
                vS += "(" + vChildGbnf + ")";
              } else {
                vS += vChildGbnf;
              }
            }
            return vS;
          } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            auto vChildGbnf = vToGbnf(pP.vChild);
            const auto &vChildParser = vEffectiveParser(pP.vChild);
            if (std::holds_alternative<common_peg_choice_parser>(vChildParser) ||
                std::holds_alternative<common_peg_sequence_parser>(vChildParser)) {
              vChildGbnf = "(" + vChildGbnf + ")";
            }
            if (pP.vMinCount == 0 && pP.vMaxCount == 1) {
              return vChildGbnf + "?";
            }
            if (pP.vMinCount == 0 && pP.vMaxCount == -1) {
              return vChildGbnf + "*";
            }
            if (pP.vMinCount == 1 && pP.vMaxCount == -1) {
              return vChildGbnf + "+";
            }
            if (pP.vMaxCount == -1) {
              return vChildGbnf + "{" + std::to_string(pP.vMinCount) + ",}";
            }
            if (pP.vMinCount == pP.vMaxCount) {
              if (pP.vMinCount == 1) {
                return vChildGbnf;
              }
              return vChildGbnf + "{" + std::to_string(pP.vMinCount) + "}";
            }
            return vChildGbnf + "{" + std::to_string(pP.vMinCount) + "," + std::to_string(pP.vMaxCount) + "}";
          } else if constexpr (std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser>) {
            return ""; // Lookahead not supported in GBNF
          } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return ".";
          } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return "space";
          } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            std::string vResult = pP.vPattern;
            if (pP.vMinCount == 0 && pP.vMaxCount == 1) {
              return vResult + "?";
            }
            if (pP.vMinCount == 0 && pP.vMaxCount == -1) {
              return vResult + "*";
            }
            if (pP.vMinCount == 1 && pP.vMaxCount == -1) {
              return vResult + "+";
            }
            if (pP.vMaxCount == -1) {
              return vResult + "{" + std::to_string(pP.vMinCount) + ",}";
            }
            if (pP.vMinCount == pP.vMaxCount) {
              if (pP.vMinCount == 1) {
                return vResult;
              }
              return vResult + "{" + std::to_string(pP.vMinCount) + "}";
            }
            return vResult + "{" + std::to_string(pP.vMinCount) + "," + std::to_string(pP.vMaxCount) + "}";
          } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
            const std::string cDelim(1, pP.vDelimiter);
            return R"(( [^)" + cDelim + R"(\\] | "\\" ( [)" + cDelim + R"(\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
          } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            if (pP.lDelimiters.empty()) {
              return ".*";
            }
            return fGbnfExcludingGrammar(pBuilder, "until-" + std::to_string(pId), pP.lDelimiters);
          } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            if (vSchemaDelegates(pP)) {
              return vToGbnf(pP.vChild);
            }
            return pBuilder.vAddSchema(pP.vName, *pP.vSchema);
          } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return pP.vName;
          } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            // Refs should not exist after flattening, but kept just in case
            return pP.vName;
          } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return vToGbnf(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return vToGbnf(pP.vChild);
          } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
            return pP.vGrammar;
          } else if constexpr (std::is_same_v<T, common_peg_ac_parser>) {
            return fGbnfIncludingGrammar(pBuilder, "ac-" + std::to_string(pId), pP.lDelimiters);
          } else {
            static_assert(cIsAlwaysFalseV<T>);
          }
        },
        vParser);
  };

  // Collect reachable rules
  std::set<std::string> sReachableRules;

  if (pLazy) {
    // Collect rules reachable from trigger rules
    for (const auto &[vName, vId] : dRules) {
      const auto &vParser = lParsers.at(vId);
      if (auto vRule = std::get_if<common_peg_rule_parser>(&vParser)) {
        if (vRule->vTrigger) {
          // Mark trigger as reachable and visit it
          sReachableRules.insert(vName);
          auto vAddRules = fCollectReachableRules(*this, vId);
          sReachableRules.insert(vAddRules.begin(), vAddRules.end());
        }
      }
    }
  } else {
    // Collect rules reachable from root
    sReachableRules = fCollectReachableRules(*this, vRoot);
  }

  // Create GBNF rules for all reachable rules
  for (const auto &[vName, vRuleId] : dRules) {
    if (sReachableRules.find(vName) == sReachableRules.end()) {
      continue;
    }

    const auto &vParser = lParsers.at(vRuleId);
    if (auto vRule = std::get_if<common_peg_rule_parser>(&vParser)) {
      pBuilder.vAddRule(vRule->vName, vToGbnf(vRule->vChild));
    }
  }

  if (pLazy) {
    // Generate root rule from trigger rules only
    std::vector<std::string> lTriggerNames;
    for (const auto &[vName, vRuleId] : dRules) {
      const auto &vParser = lParsers.at(vRuleId);
      if (auto vRule = std::get_if<common_peg_rule_parser>(&vParser)) {
        if (vRule->vTrigger) {
          lTriggerNames.push_back(vRule->vName);
        }
      }
    }

    // Sort for predictable order
    std::sort(lTriggerNames.begin(), lTriggerNames.end());
    pBuilder.vAddRule("root", fStringJoin(lTriggerNames, " | "));
  } else if (vRoot != cCommonPegInvalidParserId) {
    pBuilder.vAddRule("root", vToGbnf(vRoot));
  }
}

static nlohmann::json fSerializeParserVariant(const common_peg_parser_variant &pVariant) {
  using json = nlohmann::json;

  return std::visit(
      [](const auto &pP) -> json {
        using T = std::decay_t<decltype(pP)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
          return json{{"type", "epsilon"}};
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
          return json{{"type", "start"}};
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
          return json{{"type", "end"}};
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
          return json{{"type", "literal"}, {"literal", pP.vLiteral}};
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
          return json{{"type", "sequence"}, {"children", pP.lChildren}};
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
          return json{{"type", "choice"}, {"children", pP.lChildren}};
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
          return json{
              {"type", "repetition"}, {"child", pP.vChild}, {"min_count", pP.vMinCount}, {"max_count", pP.vMaxCount}};
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
          return json{{"type", "and"}, {"child", pP.vChild}};
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
          return json{{"type", "not"}, {"child", pP.vChild}};
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
          return json{{"type", "any"}};
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
          return json{{"type", "space"}};
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
          json vRanges = json::array();
          for (const auto &vRange : pP.lRanges) {
            vRanges.push_back({{"start", vRange.vStart}, {"end", vRange.vEnd}});
          }
          return json{{"type", "chars"},        {"pattern", pP.vPattern},    {"ranges", vRanges},
                      {"negated", pP.vNegated}, {"min_count", pP.vMinCount}, {"max_count", pP.vMaxCount}};
        } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
          return json{{"type", "string"}, {"delimiter", std::string(1, pP.vDelimiter)}};
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
          return json{{"type", "until"}, {"delimiters", pP.lDelimiters}};
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
          return json{{"type", "schema"},
                      {"child", pP.vChild},
                      {"name", pP.vName},
                      {"schema", pP.vSchema ? *pP.vSchema : nullptr},
                      {"raw", pP.vRaw}};
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
          return json{{"type", "rule"}, {"name", pP.vName}, {"child", pP.vChild}, {"trigger", pP.vTrigger}};
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
          return json{{"type", "ref"}, {"name", pP.vName}};
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
          return json{{"type", "atomic"}, {"child", pP.vChild}};
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
          return json{{"type", "tag"}, {"child", pP.vChild}, {"tag", pP.vTag}};
        } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
          return json{{"type", "gbnf"}, {"child", pP.vChild}, {"grammar", pP.vGrammar}};
        } else if constexpr (std::is_same_v<T, common_peg_ac_parser>) {
          return json{{"type", "ac"}, {"child", pP.vChild}, {"delimiters", pP.lDelimiters}};
        }
      },
      pVariant);
}

nlohmann::json common_peg_arena::to_json() const {
  auto vParsers = nlohmann::json::array();
  for (const auto &vParser : lParsers) {
    vParsers.push_back(fSerializeParserVariant(vParser));
  }
  return nlohmann::json{{"parsers", vParsers}, {"rules", dRules}, {"root", vRoot}};
}

static common_peg_parser_variant fDeserializeParserVariant(const nlohmann::json &pJ) {
  if (!pJ.contains("type") || !pJ["type"].is_string()) {
    throw std::runtime_error("Parser variant JSON missing or invalid 'type' field");
  }

  std::string vType = pJ["type"];

  if (vType == "epsilon") {
    return common_peg_epsilon_parser{};
  }
  if (vType == "start") {
    return common_peg_start_parser{};
  }
  if (vType == "end") {
    return common_peg_end_parser{};
  }
  if (vType == "literal") {
    if (!pJ.contains("literal") || !pJ["literal"].is_string()) {
      throw std::runtime_error("literal parser missing or invalid 'literal' field");
    }
    return common_peg_literal_parser{pJ["literal"]};
  }
  if (vType == "sequence") {
    if (!pJ.contains("children") || !pJ["children"].is_array()) {
      throw std::runtime_error("sequence parser missing or invalid 'children' field");
    }
    return common_peg_sequence_parser{pJ["children"].get<std::vector<common_peg_parser_id>>()};
  }
  if (vType == "choice") {
    if (!pJ.contains("children") || !pJ["children"].is_array()) {
      throw std::runtime_error("choice parser missing or invalid 'children' field");
    }
    return common_peg_choice_parser{pJ["children"].get<std::vector<common_peg_parser_id>>()};
  }
  if (vType == "repetition") {
    if (!pJ.contains("child") || !pJ.contains("min_count") || !pJ.contains("max_count")) {
      throw std::runtime_error("repetition parser missing required fields");
    }
    return common_peg_repetition_parser{pJ["child"].get<common_peg_parser_id>(), pJ["min_count"].get<int>(),
                                        pJ["max_count"].get<int>()};
  }
  if (vType == "and") {
    if (!pJ.contains("child")) {
      throw std::runtime_error("and parser missing 'child' field");
    }
    return common_peg_and_parser{pJ["child"].get<common_peg_parser_id>()};
  }
  if (vType == "not") {
    if (!pJ.contains("child")) {
      throw std::runtime_error("not parser missing 'child' field");
    }
    return common_peg_not_parser{pJ["child"].get<common_peg_parser_id>()};
  }
  if (vType == "any") {
    return common_peg_any_parser{};
  }
  if (vType == "space") {
    return common_peg_space_parser{};
  }
  if (vType == "chars") {
    if (!pJ.contains("pattern") || !pJ.contains("ranges") || !pJ.contains("negated") || !pJ.contains("min_count") ||
        !pJ.contains("max_count")) {
      throw std::runtime_error("chars parser missing required fields");
    }
    common_peg_chars_parser vParser;
    vParser.vPattern = pJ["pattern"];
    vParser.vNegated = pJ["negated"];
    vParser.vMinCount = pJ["min_count"];
    vParser.vMaxCount = pJ["max_count"];
    for (const auto &vRangeJson : pJ["ranges"]) {
      if (!vRangeJson.contains("start") || !vRangeJson.contains("end")) {
        throw std::runtime_error("char_range missing 'start' or 'end' field");
      }
      vParser.lRanges.push_back({vRangeJson["start"].get<uint32_t>(), vRangeJson["end"].get<uint32_t>()});
    }
    return vParser;
  }
  if (vType == "string") {
    if (!pJ.contains("delimiter")) {
      throw std::runtime_error("string parser missing delimiter field.");
    }
    std::string vDelimiter = pJ["delimiter"];
    if (vDelimiter.empty()) {
      throw std::runtime_error("string parser delimiter is empty.");
    }
    return common_peg_string_parser{vDelimiter[0]};
  }
  if (vType == "until") {
    if (!pJ.contains("delimiters") || !pJ["delimiters"].is_array()) {
      throw std::runtime_error("until parser missing or invalid 'delimiters' field");
    }
    return common_peg_until_parser{pJ["delimiters"].get<std::vector<std::string>>()};
  }
  if (vType == "schema") {
    if (!pJ.contains("child") || !pJ.contains("name") || !pJ.contains("schema") || !pJ.contains("raw")) {
      throw std::runtime_error("schema parser missing required fields");
    }
    common_peg_schema_parser vParser;
    vParser.vChild = pJ["child"].get<common_peg_parser_id>();
    vParser.vName = pJ["name"];
    if (!pJ["schema"].is_null()) {
      vParser.vSchema = std::make_shared<nlohmann::ordered_json>(pJ["schema"]);
    }
    vParser.vRaw = pJ["raw"].get<bool>();
    return vParser;
  }
  if (vType == "rule") {
    if (!pJ.contains("name") || !pJ.contains("child") || !pJ.contains("trigger")) {
      throw std::runtime_error("rule parser missing required fields");
    }
    return common_peg_rule_parser{pJ["name"].get<std::string>(), pJ["child"].get<common_peg_parser_id>(),
                                  pJ["trigger"].get<bool>()};
  }
  if (vType == "ref") {
    if (!pJ.contains("name") || !pJ["name"].is_string()) {
      throw std::runtime_error("ref parser missing or invalid 'name' field");
    }
    return common_peg_ref_parser{pJ["name"]};
  }
  if (vType == "atomic") {
    if (!pJ.contains("child")) {
      throw std::runtime_error("tag parser missing required fields");
    }
    return common_peg_atomic_parser{
        pJ["child"].get<common_peg_parser_id>(),
    };
  }
  if (vType == "tag") {
    if (!pJ.contains("child") || !pJ.contains("tag")) {
      throw std::runtime_error("tag parser missing required fields");
    }
    return common_peg_tag_parser{
        pJ["child"].get<common_peg_parser_id>(),
        pJ["tag"].get<std::string>(),
    };
  }

  if (vType == "gbnf") {
    if (!pJ.contains("child") || !pJ.contains("grammar")) {
      throw std::runtime_error("gbnf parser missing required fields");
    }
    return common_peg_gbnf_parser{
        pJ["child"].get<common_peg_parser_id>(),
        pJ["grammar"].get<std::string>(),
    };
  }

  if (vType == "ac") {
    if (!pJ.contains("child") || !pJ.contains("delimiters") || !pJ["delimiters"].is_array() ||
        pJ["delimiters"].empty()) {
      throw std::runtime_error("ac parser requires 'child' and a non-empty 'delimiters' array");
    }
    return common_peg_ac_parser{
        pJ["child"].get<common_peg_parser_id>(),
        pJ["delimiters"].get<std::vector<std::string>>(),
    };
  }

  throw std::runtime_error("Unknown parser type: " + vType);
}

common_peg_arena common_peg_arena::from_json(const nlohmann::json &pJ) {
  if (!pJ.contains("parsers") || !pJ["parsers"].is_array()) {
    throw std::runtime_error("JSON missing or invalid 'parsers' array");
  }
  if (!pJ.contains("rules") || !pJ["rules"].is_object()) {
    throw std::runtime_error("JSON missing or invalid 'rules' object");
  }
  if (!pJ.contains("root")) {
    throw std::runtime_error("JSON missing 'root' field");
  }

  common_peg_arena vArena;

  const auto &vParsersJson = pJ["parsers"];
  vArena.lParsers.reserve(vParsersJson.size());
  for (const auto &vParserJson : vParsersJson) {
    vArena.lParsers.push_back(fDeserializeParserVariant(vParserJson));
  }

  vArena.dRules = pJ["rules"].get<std::unordered_map<std::string, common_peg_parser_id>>();

  for (const auto &[vName, vId] : vArena.dRules) {
    if (vId >= vArena.lParsers.size()) {
      throw std::runtime_error("Rule '" + vName + "' references invalid parser ID: " + std::to_string(vId));
    }
  }

  vArena.vRoot = pJ["root"].get<common_peg_parser_id>();
  if (vArena.vRoot != cCommonPegInvalidParserId && vArena.vRoot >= vArena.lParsers.size()) {
    throw std::runtime_error("Root references invalid parser ID: " + std::to_string(vArena.vRoot));
  }

  return vArena;
}

std::string common_peg_arena::mSave() const { return to_json().dump(); }

void common_peg_arena::mLoad(const std::string &pData) { *this = from_json(nlohmann::json::parse(pData)); }

common_peg_arena fBuildPegParser(const std::function<common_peg_parser(common_peg_parser_builder &pBuilder)> &pFn) {
  common_peg_parser_builder vBuilder;
  vBuilder.mSetRoot(pFn(vBuilder));
  return vBuilder.mBuild();
}
