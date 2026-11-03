#include "mgs-grammar.h"

#include "mgs-impl.h"
#include "mgs-vocab.h"
#include "mgs-sampler.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>

#define MAX_REPETITION_THRESHOLD 2000
//
// helpers
//

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> fDecodeUtf8(const char *pSrc) {
  static const int aLookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t vFirstByte = static_cast<uint8_t>(*pSrc);
  uint8_t vHighbits = vFirstByte >> 4;
  int vLen = aLookup[vHighbits];
  uint8_t vMask = (1 << (8 - vLen)) - 1;
  uint32_t vValue = vFirstByte & vMask;
  const char *vEnd = pSrc + vLen; // may overrun!
  const char *vPos = pSrc + 1;
  for (; vPos < vEnd && *vPos; vPos++) {
    vValue = (vValue << 6) + (static_cast<uint8_t>(*vPos) & 0x3F);
  }
  return std::make_pair(vValue, vPos);
}

static std::pair<std::vector<uint32_t>, mgs_partial_utf8> fDecodeUtf8(const std::string &pSrc,
                                                                      mgs_partial_utf8 pPartialStart) {
  static const int aLookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
  const char *vPos = pSrc.c_str();
  std::vector<uint32_t> lCodePoints;

  // common english strings have the same number of codepoints and bytes. `+ 1` for the terminating 0.
  lCodePoints.reserve(pSrc.size() + 1);
  uint32_t vValue = pPartialStart.vValue;
  int vNRemain = pPartialStart.vNRemain;

  // continue previous decode, if applicable
  while (*vPos != 0 && vNRemain > 0) {
    uint8_t vNextByte = static_cast<uint8_t>(*vPos);
    if ((vNextByte >> 6) != 2) {
      // invalid sequence, abort
      lCodePoints.push_back(0);
      return std::make_pair(std::move(lCodePoints), mgs_partial_utf8{0, -1});
    }
    vValue = (vValue << 6) + (vNextByte & 0x3F);
    ++vPos;
    --vNRemain;
  }

  if (pPartialStart.vNRemain > 0 && vNRemain == 0) {
    lCodePoints.push_back(vValue);
  }

  // decode any subsequent utf-8 sequences, which may end in an incomplete one
  while (*vPos != 0) {
    uint8_t vFirstByte = static_cast<uint8_t>(*vPos);
    uint8_t vHighbits = vFirstByte >> 4;
    vNRemain = aLookup[vHighbits] - 1;

    if (vNRemain < 0) {
      // invalid sequence, abort
      lCodePoints.clear();
      lCodePoints.push_back(0);
      return std::make_pair(std::move(lCodePoints), mgs_partial_utf8{0, vNRemain});
    }

    uint8_t vMask = (1 << (7 - vNRemain)) - 1;
    vValue = vFirstByte & vMask;

    ++vPos;
    while (*vPos != 0 && vNRemain > 0) {
      vValue = (vValue << 6) + (static_cast<uint8_t>(*vPos) & 0x3F);
      ++vPos;
      --vNRemain;
    }
    if (vNRemain == 0) {
      lCodePoints.push_back(vValue);
    }
  }
  lCodePoints.push_back(0);

  return std::make_pair(std::move(lCodePoints), mgs_partial_utf8{vValue, vNRemain});
}

static bool fIsDigitChar(char pC) { return '0' <= pC && pC <= '9'; }

static bool fIsWordChar(char pC) {
  return ('a' <= pC && pC <= 'z') || ('A' <= pC && pC <= 'Z') || pC == '-' || fIsDigitChar(pC);
}

static std::pair<uint32_t, const char *> fParseHex(const char *pSrc, int pSize) {
  const char *vPos = pSrc;
  const char *vEnd = pSrc + pSize;
  uint32_t vValue = 0;
  for (; vPos < vEnd && *vPos; vPos++) {
    vValue <<= 4;
    char vC = *vPos;
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
  if (vPos != vEnd) {
    throw std::runtime_error("expecting " + std::to_string(pSize) + " hex chars at " + pSrc);
  }
  return std::make_pair(vValue, vPos);
}

static const char *fParseSpace(const char *pSrc, bool pNewlineOk) {
  const char *vPos = pSrc;
  while (*vPos == ' ' || *vPos == '\t' || *vPos == '#' || (pNewlineOk && (*vPos == '\r' || *vPos == '\n'))) {
    if (*vPos == '#') {
      while (*vPos && *vPos != '\r' && *vPos != '\n') {
        vPos++;
      }
    } else {
      vPos++;
    }
  }
  return vPos;
}

static const char *fParseName(const char *pSrc) {
  const char *vPos = pSrc;
  while (fIsWordChar(*vPos)) {
    vPos++;
  }
  if (vPos == pSrc) {
    throw std::runtime_error(std::string("expecting name at ") + pSrc);
  }
  return vPos;
}

static const char *fParseInt(const char *pSrc) {
  const char *vPos = pSrc;
  while (fIsDigitChar(*vPos)) {
    vPos++;
  }
  if (vPos == pSrc) {
    throw std::runtime_error(std::string("expecting integer at ") + pSrc);
  }
  return vPos;
}

static std::pair<uint32_t, const char *> fParseChar(const char *pSrc) {
  if (*pSrc == '\\') {
    switch (pSrc[1]) {
    case 'x':
      return fParseHex(pSrc + 2, 2);
    case 'u':
      return fParseHex(pSrc + 2, 4);
    case 'U':
      return fParseHex(pSrc + 2, 8);
    case 't':
      return std::make_pair('\t', pSrc + 2);
    case 'r':
      return std::make_pair('\r', pSrc + 2);
    case 'n':
      return std::make_pair('\n', pSrc + 2);
    case '\\':
    case '"':
    case '[':
    case ']':
      return std::make_pair(pSrc[1], pSrc + 2);
    default:
      throw std::runtime_error(std::string("unknown escape at ") + pSrc);
    }
  } else if (*pSrc) {
    return fDecodeUtf8(pSrc);
  }
  throw std::runtime_error("unexpected end of input");
}

static std::pair<uint32_t, const char *> fParseToken(const mgs_vocab *pVocab, const char *pSrc) {
  const char *vPos = pSrc;
  if (*vPos != '<') {
    throw std::runtime_error(std::string("expecting '<' at ") + vPos);
  }
  vPos++;

  // Parse <[id]>
  if (*vPos == '[') {
    vPos++;
    const char *vIntEnd = fParseInt(vPos);
    uint32_t vTokenId = std::stoul(std::string(vPos, vIntEnd - vPos));
    vPos = vIntEnd;
    if (*vPos != ']') {
      throw std::runtime_error(std::string("expecting ']' at ") + vPos);
    }
    vPos++;
    if (*vPos != '>') {
      throw std::runtime_error(std::string("expecting '>' at ") + vPos);
    }
    vPos++;
    return std::make_pair(vTokenId, vPos);
  }

  if (pVocab == nullptr) {
    throw std::runtime_error(std::string("no vocab to parse token at ") + pSrc);
  }

  // Parse <token> and tokenize to obtain the token id
  while (*vPos != 0 && *vPos != '>') {
    vPos++;
  }
  if (*vPos != '>') {
    throw std::runtime_error(std::string("expecting '>' at ") + vPos);
  }
  vPos++;

  mgs_token aTokens[2];
  int32_t vNTokens = pVocab->mTokenize(pSrc, static_cast<int32_t>(vPos - pSrc), aTokens, 2, false, true);
  if (vNTokens != 1) {
    // must tokenize to exactly 1 token
    throw std::runtime_error("invalid token '" + std::string(pSrc, vPos - pSrc) + "'");
  }
  return std::make_pair(aTokens[0], vPos);
}

static void fPrintGrammarChar(FILE *pFile, uint32_t pC) {
  if (0x20 <= pC && pC <= 0x7f) {
    fprintf(pFile, "%c", static_cast<char>(pC));
  } else {
    // cop out of encoding UTF-8
    fprintf(pFile, "<U+%04X>", pC);
  }
}

static bool fIsCharElement(mgs_grammar_element pElem) {
  switch (pElem.vType) {
  case cMgsGretypeChar:
    return true;
  case cMgsGretypeCharNot:
    return true;
  case cMgsGretypeCharAlt:
    return true;
  case cMgsGretypeCharRngUpper:
    return true;
  case cMgsGretypeCharAny:
    return true;
  default:
    return false;
  }
}

static void fPrintRuleBinary(FILE *pFile, const mgs_grammar_rule &pRule) {
  for (auto vElem : pRule) {
    switch (vElem.vType) {
    case cMgsGretypeEnd:
      fprintf(pFile, "END");
      break;
    case cMgsGretypeAlt:
      fprintf(pFile, "ALT");
      break;
    case cMgsGretypeRuleRef:
      fprintf(pFile, "RULE_REF");
      break;
    case cMgsGretypeChar:
      fprintf(pFile, "CHAR");
      break;
    case cMgsGretypeCharNot:
      fprintf(pFile, "CHAR_NOT");
      break;
    case cMgsGretypeCharRngUpper:
      fprintf(pFile, "CHAR_RNG_UPPER");
      break;
    case cMgsGretypeCharAlt:
      fprintf(pFile, "CHAR_ALT");
      break;
    case cMgsGretypeCharAny:
      fprintf(pFile, "CHAR_ANY");
      break;
    case cMgsGretypeToken:
      fprintf(pFile, "TOKEN");
      break;
    case cMgsGretypeTokenNot:
      fprintf(pFile, "TOKEN_NOT");
      break;
    }
    switch (vElem.vType) {
    case cMgsGretypeEnd:
    case cMgsGretypeAlt:
    case cMgsGretypeRuleRef:
      fprintf(pFile, "(%u) ", vElem.vValue);
      break;
    case cMgsGretypeChar:
    case cMgsGretypeCharNot:
    case cMgsGretypeCharRngUpper:
    case cMgsGretypeCharAlt:
    case cMgsGretypeCharAny:
      fprintf(pFile, "(\"");
      fPrintGrammarChar(pFile, vElem.vValue);
      fprintf(pFile, "\") ");
      break;
    case cMgsGretypeToken:
      fprintf(pFile, "<[");
      fprintf(pFile, "%u", vElem.vValue);
      fprintf(pFile, "]> ");
      break;
    case cMgsGretypeTokenNot:
      fprintf(pFile, "!");
      fprintf(pFile, "<[");
      fprintf(pFile, "%u", vElem.vValue);
      fprintf(pFile, "]> ");
      break;
    }
  }
  fprintf(pFile, "\n");
}

static void fPrintRule(FILE *pFile, uint32_t pRuleId, const mgs_grammar_rule &pRule,
                       const std::map<uint32_t, std::string> &pSymbolIdNames) {
  if (pRule.empty() || pRule.back().vType != cMgsGretypeEnd) {
    throw std::runtime_error("malformed rule, does not end with MGS_GRETYPE_END: " + std::to_string(pRuleId));
  }
  fprintf(pFile, "%s ::= ", pSymbolIdNames.at(pRuleId).c_str());
  for (size_t vI = 0, vEnd = pRule.size() - 1; vI < vEnd; vI++) {
    mgs_grammar_element vElem = pRule[vI];
    switch (vElem.vType) {
    case cMgsGretypeEnd:
      throw std::runtime_error("unexpected end of rule: " + std::to_string(pRuleId) + "," + std::to_string(vI));
    case cMgsGretypeAlt:
      fprintf(pFile, "| ");
      break;
    case cMgsGretypeRuleRef:
      fprintf(pFile, "%s ", pSymbolIdNames.at(vElem.vValue).c_str());
      break;
    case cMgsGretypeChar:
      fprintf(pFile, "[");
      fPrintGrammarChar(pFile, vElem.vValue);
      break;
    case cMgsGretypeCharNot:
      fprintf(pFile, "[^");
      fPrintGrammarChar(pFile, vElem.vValue);
      break;
    case cMgsGretypeCharRngUpper:
      if (vI == 0 || !fIsCharElement(pRule[vI - 1])) {
        throw std::runtime_error("MGS_GRETYPE_CHAR_RNG_UPPER without preceding char: " + std::to_string(pRuleId) + "," +
                                 std::to_string(vI));
      }
      fprintf(pFile, "-");
      fPrintGrammarChar(pFile, vElem.vValue);
      break;
    case cMgsGretypeCharAlt:
      if (vI == 0 || !fIsCharElement(pRule[vI - 1])) {
        throw std::runtime_error("MGS_GRETYPE_CHAR_ALT without preceding char: " + std::to_string(pRuleId) + "," +
                                 std::to_string(vI));
      }
      fPrintGrammarChar(pFile, vElem.vValue);
      break;
    case cMgsGretypeCharAny:
      fprintf(pFile, ".");
      break;
    case cMgsGretypeToken:
      fprintf(pFile, "<[");
      fprintf(pFile, "%u", vElem.vValue);
      fprintf(pFile, "]> ");
      break;
    case cMgsGretypeTokenNot:
      fprintf(pFile, "!");
      fprintf(pFile, "<[");
      fprintf(pFile, "%u", vElem.vValue);
      fprintf(pFile, "]> ");
      break;
    }
    if (fIsCharElement(vElem)) {
      switch (pRule[vI + 1].vType) {
      case cMgsGretypeCharAlt:
      case cMgsGretypeCharRngUpper:
      case cMgsGretypeCharAny:
        break;
      default:
        fprintf(pFile, "] ");
      }
    }
  }
  fprintf(pFile, "\n");
}

//
// Regex utilities
//

size_t mgs_grammar_trigger_pattern::mFind(const std::string &pInput) const {
  auto vFindStartPos = [](const std::smatch &pMatch) {
    // get from the first matched capturing group to the end of the string
    size_t vStart = std::string::npos;
    for (auto vI = 1u; vI < pMatch.size(); vI++) {
      if (pMatch.length(vI) > 0) {
        vStart = pMatch.position(vI);
        break;
      }
    }
    if (vStart == std::string::npos) {
      vStart = pMatch.position(0);
    }
    return vStart;
  };

  if (!vPattern.empty() && vPattern.front() == '^' && vPattern.back() == '$') {
    // match against the entire input
    std::smatch vMatch;
    if (std::regex_match(pInput, vMatch, vRegex)) {
      return vFindStartPos(vMatch);
    }
  }

  // search anywhere
  std::smatch vMatch;
  if (std::regex_search(pInput, vMatch, vRegex)) {
    return vFindStartPos(vMatch);
  }

  return std::string::npos;
}

//
// implementation
//

uint32_t mgs_grammar_parser::mGetSymbolId(const char *pSrc, size_t pLen) {
  uint32_t vNextId = static_cast<uint32_t>(dSymbolIds.size());
  auto vResult = dSymbolIds.emplace(std::string(pSrc, pLen), vNextId);
  return vResult.first->second;
}

uint32_t mgs_grammar_parser::mGenerateSymbolId(const std::string &pBaseName) {
  uint32_t vNextId = static_cast<uint32_t>(dSymbolIds.size());
  dSymbolIds[pBaseName + '_' + std::to_string(vNextId)] = vNextId;
  return vNextId;
}

void mgs_grammar_parser::mAddRule(uint32_t pRuleId, const mgs_grammar_rule &pRule) {
  if (vRules.size() <= pRuleId) {
    vRules.resize(pRuleId + 1);
  }
  vRules[pRuleId] = pRule;
}

const char *mgs_grammar_parser::mParseAlternates(const char *pSrc, const std::string &pRuleName, uint32_t pRuleId,
                                                 bool pIsNested) {
  mgs_grammar_rule vRule;
  const char *vPos = mParseSequence(pSrc, pRuleName, vRule, pIsNested);
  while (*vPos == '|') {
    vRule.push_back({cMgsGretypeAlt, 0});
    vPos = fParseSpace(vPos + 1, true);
    vPos = mParseSequence(vPos, pRuleName, vRule, pIsNested);
  }
  vRule.push_back({cMgsGretypeEnd, 0});
  mAddRule(pRuleId, vRule);
  return vPos;
}

const char *mgs_grammar_parser::mParseSequence(const char *pSrc, const std::string &pRuleName, mgs_grammar_rule &pRule,
                                               bool pIsNested) {
  size_t vLastSymStart = pRule.size();
  const char *vPos = pSrc;
  uint64_t vNPrevRules = 1;

  // use UINT64_MAX as the empty value because we aligned to the proper uint64_t type so -1 can't be used
  // (though it's technically the same as -1 now)
  auto vHandleRepetitions = [&](uint64_t pMinTimes, uint64_t pMaxTimes) {
    bool vNoMax = pMaxTimes == UINT64_MAX;
    if (vLastSymStart == pRule.size()) {
      throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + vPos);
    }

    // apply transformation to previous symbol (last_sym_start to end) according to
    // the following rewrite rules:
    // S{m,n} --> S S S (m times) S'(n-m)
    //            S'(x)   ::= S S'(x-1) |
    //            (... n-m definitions of these S' rules ...)
    //            S'(1)   ::= S |
    // S{m,} -->  S S S (m times) S'
    //            S'     ::= S S' |
    // S*     --> S{0,}
    //        --> S'     ::= S S' |
    // S+     --> S{1,}
    //        --> S S'
    //            S'     ::= S S' |
    // S?     --> S{0,1}
    //        --> S'
    //            S'     ::= S |

    mgs_grammar_rule vPrevRule(pRule.begin() + vLastSymStart, pRule.end());
    // Calculate the total number of rules that will be generated by this repetition
    uint64_t vTotalRules = 1; // Start with 1 for the original rule
    if (!vNoMax && pMaxTimes > 0) {
      vTotalRules = pMaxTimes;
    } else if (pMinTimes > 0) {
      vTotalRules = pMinTimes;
    }

    if (vNPrevRules * vTotalRules >= MAX_REPETITION_THRESHOLD) {
      throw std::runtime_error("number of rules that are going to be repeated multiplied by the new repetition exceeds "
                               "sane defaults, please reduce the number of repetitions or rule complexity");
    }

    if (pMinTimes == 0) {
      pRule.resize(vLastSymStart);
    } else {
      // Repeat the previous elements (min_times - 1) times
      for (uint64_t vI = 1; vI < pMinTimes; vI++) {
        pRule.insert(pRule.end(), vPrevRule.begin(), vPrevRule.end());
      }
    }

    uint32_t vLastRecRuleId = 0;
    auto vNOpt = vNoMax ? 1 : pMaxTimes - pMinTimes;

    mgs_grammar_rule vRecRule(vPrevRule);
    for (uint64_t vI = 0; vI < vNOpt; vI++) {
      vRecRule.resize(vPrevRule.size());
      uint32_t vRecRuleId = mGenerateSymbolId(pRuleName);
      if (vI > 0 || vNoMax) {
        vRecRule.push_back({cMgsGretypeRuleRef, vNoMax ? vRecRuleId : vLastRecRuleId});
      }
      vRecRule.push_back({cMgsGretypeAlt, 0});
      vRecRule.push_back({cMgsGretypeEnd, 0});
      mAddRule(vRecRuleId, vRecRule);
      vLastRecRuleId = vRecRuleId;
    }
    if (vNOpt > 0) {
      pRule.push_back({cMgsGretypeRuleRef, vLastRecRuleId});
    }
    vNPrevRules *= vTotalRules;
    GGML_ASSERT(vNPrevRules >= 1);
  };

  while (*vPos) {
    if (*vPos == '"') { // literal string
      vPos++;
      vLastSymStart = pRule.size();
      vNPrevRules = 1;
      while (*vPos != '"') {
        if (!*vPos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto vCharPair = fParseChar(vPos);
        vPos = vCharPair.second;
        pRule.push_back({cMgsGretypeChar, vCharPair.first});
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '[') { // char range(s)
      vPos++;
      enum mgs_gretype vStartType = cMgsGretypeChar;
      if (*vPos == '^') {
        vPos++;
        vStartType = cMgsGretypeCharNot;
      }
      vLastSymStart = pRule.size();
      vNPrevRules = 1;
      while (*vPos != ']') {
        if (!*vPos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto vCharPair = fParseChar(vPos);
        vPos = vCharPair.second;
        enum mgs_gretype vType = vLastSymStart < pRule.size() ? cMgsGretypeCharAlt : vStartType;

        pRule.push_back({vType, vCharPair.first});
        if (vPos[0] == '-' && vPos[1] != ']') {
          if (!vPos[1]) {
            throw std::runtime_error("unexpected end of input");
          }
          auto vEndcharPair = fParseChar(vPos + 1);
          vPos = vEndcharPair.second;
          pRule.push_back({cMgsGretypeCharRngUpper, vEndcharPair.first});
        }
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '<' || *vPos == '!') { // token
      auto vType = cMgsGretypeToken;
      if (*vPos == '!') { // token inverse
        vType = cMgsGretypeTokenNot;
        vPos++;
      }
      auto vTokenPair = fParseToken(vVocab, vPos);
      const char *vTokenEnd = vTokenPair.second;
      vLastSymStart = pRule.size();
      vNPrevRules = 1;
      pRule.push_back({vType, vTokenPair.first});
      vPos = fParseSpace(vTokenEnd, pIsNested);
    } else if (fIsWordChar(*vPos)) { // rule reference
      const char *vNameEnd = fParseName(vPos);
      uint32_t vRefRuleId = mGetSymbolId(vPos, vNameEnd - vPos);
      vPos = fParseSpace(vNameEnd, pIsNested);
      vLastSymStart = pRule.size();
      vNPrevRules = 1;
      pRule.push_back({cMgsGretypeRuleRef, vRefRuleId});
    } else if (*vPos == '(') { // grouping
      // parse nested alternates into synthesized rule
      vPos = fParseSpace(vPos + 1, true);
      uint32_t vNRulesBefore = dSymbolIds.size();
      uint32_t vSubRuleId = mGenerateSymbolId(pRuleName);
      vPos = mParseAlternates(vPos, pRuleName, vSubRuleId, true);
      vNPrevRules = std::max(1u, (uint32_t)dSymbolIds.size() - vNRulesBefore);
      vLastSymStart = pRule.size();
      // output reference to synthesized rule
      pRule.push_back({cMgsGretypeRuleRef, vSubRuleId});
      if (*vPos != ')') {
        throw std::runtime_error(std::string("expecting ')' at ") + vPos);
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '.') { // any char
      vLastSymStart = pRule.size();
      vNPrevRules = 1;
      pRule.push_back({cMgsGretypeCharAny, 0});
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '*') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(0, -1);
    } else if (*vPos == '+') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(1, -1);
    } else if (*vPos == '?') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(0, 1);
    } else if (*vPos == '{') {
      vPos = fParseSpace(vPos + 1, pIsNested);

      if (!fIsDigitChar(*vPos)) {
        throw std::runtime_error(std::string("expecting an int at ") + vPos);
      }
      const char *vIntEnd = fParseInt(vPos);
      uint64_t vMinTimes = std::stoull(std::string(vPos, vIntEnd - vPos));
      vPos = fParseSpace(vIntEnd, pIsNested);

      uint64_t vMaxTimes = UINT64_MAX; // default: no max limit

      if (*vPos == '}') {
        vMaxTimes = vMinTimes;
        vPos = fParseSpace(vPos + 1, pIsNested);
      } else if (*vPos == ',') {
        vPos = fParseSpace(vPos + 1, pIsNested);

        if (fIsDigitChar(*vPos)) {
          const char *vIntEnd = fParseInt(vPos);
          vMaxTimes = std::stoull(std::string(vPos, vIntEnd - vPos));
          vPos = fParseSpace(vIntEnd, pIsNested);
        }

        if (*vPos != '}') {
          throw std::runtime_error(std::string("expecting '}' at ") + vPos);
        }
        vPos = fParseSpace(vPos + 1, pIsNested);
      } else {
        throw std::runtime_error(std::string("expecting ',' at ") + vPos);
      }
      bool vHasMax = vMaxTimes != UINT64_MAX;
      if (vMinTimes > MAX_REPETITION_THRESHOLD || (vHasMax && vMaxTimes > MAX_REPETITION_THRESHOLD)) {
        throw std::runtime_error(
            std::string("number of repetitions exceeds sane defaults, please reduce the number of repetitions"));
      }
      vHandleRepetitions(vMinTimes, vMaxTimes);
    } else {
      break;
    }
  }
  return vPos;
}

const char *mgs_grammar_parser::mParseRule(const char *pSrc) {
  const char *vNameEnd = fParseName(pSrc);
  const char *vPos = fParseSpace(vNameEnd, false);
  size_t vNameLen = vNameEnd - pSrc;
  uint32_t vRuleId = mGetSymbolId(pSrc, vNameLen);
  const std::string cName(pSrc, vNameLen);

  if (!(vPos[0] == ':' && vPos[1] == ':' && vPos[2] == '=')) {
    throw std::runtime_error(std::string("expecting ::= at ") + vPos);
  }
  vPos = fParseSpace(vPos + 3, true);

  vPos = mParseAlternates(vPos, cName, vRuleId, false);

  if (*vPos == '\r') {
    vPos += vPos[1] == '\n' ? 2 : 1;
  } else if (*vPos == '\n') {
    vPos++;
  } else if (*vPos) {
    throw std::runtime_error(std::string("expecting newline or end at ") + vPos);
  }
  return fParseSpace(vPos, true);
}

bool mgs_grammar_parser::mParse(const char *pSrc) {
  try {
    const char *vPos = fParseSpace(pSrc, true);
    while (*vPos) {
      vPos = mParseRule(vPos);
    }
    // Validate the state to ensure that all rules are defined
    for (const auto &vRule : vRules) {
      if (vRule.empty()) {
        throw std::runtime_error("Undefined rule");
      }
      for (const auto &vElem : vRule) {
        if (vElem.vType == cMgsGretypeRuleRef) {
          // Ensure that the rule at that location exists
          if (vElem.vValue >= vRules.size() || vRules[vElem.vValue].empty()) {
            // Get the name of the rule that is missing
            for (const auto &vKv : dSymbolIds) {
              if (vKv.second == vElem.vValue) {
                throw std::runtime_error("Undefined rule identifier '" + vKv.first + "'");
              }
            }
          }
        }
      }
    }
  } catch (const std::exception &vErr) {
    fprintf(stderr, "%s: error parsing grammar: %s\n\n%s\n", __func__, vErr.what(), pSrc);
    vRules.clear();
    return false;
  }

  return true;
}

void mgs_grammar_parser::mPrint(FILE *pFile) {
  try {
    std::map<uint32_t, std::string> dSymbolIdNames;
    for (const auto &vKv : dSymbolIds) {
      dSymbolIdNames[vKv.second] = vKv.first;
    }
    for (size_t vI = 0, vEnd = vRules.size(); vI < vEnd; vI++) {
      // fprintf(pFile, "%zu: ", vI);
      // fPrintRuleBinary(pFile, vRules[vI]);
      fPrintRule(pFile, uint32_t(vI), vRules[vI], dSymbolIdNames);
      // fprintf(pFile, "\n");
    }
  } catch (const std::exception &vErr) {
    fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, vErr.what());
  }
}

mgs_grammar_stack mgs_grammar_parser::mCRules() const {
  mgs_grammar_stack vRet;
  vRet.reserve(vRules.size());
  for (const auto &vRule : vRules) {
    vRet.push_back(vRule.data());
  }
  return vRet;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool fMgsGrammarIsEndOfSequence(const mgs_grammar_element *pPos) {
  switch (pPos->vType) {
  case cMgsGretypeEnd:
    return true; // NOLINT
  case cMgsGretypeAlt:
    return true; // NOLINT
  default:
    return false;
  }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const mgs_grammar_element *> fMgsGrammarMatchChar(const mgs_grammar_element *pPos,
                                                                         const uint32_t pChr) {
  bool vFound = false;
  bool vIsPositiveChar = pPos->vType == cMgsGretypeChar || pPos->vType == cMgsGretypeCharAny;

  GGML_ASSERT(vIsPositiveChar || pPos->vType == cMgsGretypeCharNot); // NOLINT

  do {
    if (pPos[1].vType == cMgsGretypeCharRngUpper) {
      // inclusive range, e.g. [a-z]
      vFound = vFound || (pPos->vValue <= pChr && pChr <= pPos[1].vValue);
      pPos += 2;
    } else if (pPos->vType == cMgsGretypeCharAny) {
      // Any character matches "."
      vFound = true;
      pPos += 1;
    } else {
      // exact char match, e.g. [a] or "a"
      vFound = vFound || pPos->vValue == pChr;
      pPos += 1;
    }
  } while (pPos->vType == cMgsGretypeCharAlt);

  return std::make_pair(vFound == vIsPositiveChar, pPos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool fMgsGrammarMatchPartialChar(const mgs_grammar_element *pPos, const mgs_partial_utf8 pPartialUtf8) {
  bool vIsPositiveChar = pPos->vType == cMgsGretypeChar || pPos->vType == cMgsGretypeCharAny;
  GGML_ASSERT(vIsPositiveChar || pPos->vType == cMgsGretypeCharNot);

  uint32_t vPartialValue = pPartialUtf8.vValue;
  int vNRemain = pPartialUtf8.vNRemain;

  // invalid sequence or 7-bit char split across 2 bytes (overlong)
  if (vNRemain < 0 || (vNRemain == 1 && vPartialValue < 2)) {
    return false;
  }

  // range of possible code points this partial UTF-8 sequence could complete to
  uint32_t vLow = vPartialValue << (vNRemain * 6);
  uint32_t vHigh = vLow | ((1 << (vNRemain * 6)) - 1);

  if (vLow == 0) {
    if (vNRemain == 2) {
      vLow = 1 << 11;
    } else if (vNRemain == 3) {
      vLow = 1 << 16;
    }
  }

  do {
    if (pPos[1].vType == cMgsGretypeCharRngUpper) {
      // inclusive range, e.g. [a-z]
      if (pPos->vValue <= vHigh && vLow <= pPos[1].vValue) {
        return vIsPositiveChar;
      }
      pPos += 2;
    } else if (pPos->vType == cMgsGretypeCharAny) {
      // Any character matches "."
      return true;
    } else {
      // exact char match, e.g. [a] or "a"
      if (vLow <= pPos->vValue && pPos->vValue <= vHigh) {
        return vIsPositiveChar;
      }
      pPos += 1;
    }
  } while (pPos->vType == cMgsGretypeCharAlt);

  return !vIsPositiveChar;
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
static bool fMgsGrammarMatchToken(const mgs_grammar_element *pPos, const mgs_token pToken) {
  GGML_ASSERT(pPos->vType == cMgsGretypeToken || pPos->vType == cMgsGretypeTokenNot);
  if (pPos->vType == cMgsGretypeToken) {
    return pPos->vValue == static_cast<uint32_t>(pToken);
  }
  if (pPos->vType == cMgsGretypeTokenNot) {
    return pPos->vValue != static_cast<uint32_t>(pToken);
  }
  return false;
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void fMgsGrammarAdvanceStack(const mgs_grammar_rules &pRules, const mgs_grammar_stack &pStack,
                                    mgs_grammar_stacks &pNewStacks) {
  std::vector<mgs_grammar_stack> lTodo;
  lTodo.push_back(pStack);

  auto vStackCmp = [](const mgs_grammar_stack &pA, const mgs_grammar_stack &pB) {
    return std::lexicographical_compare(pA.begin(), pA.end(), pB.begin(), pB.end(),
                                        [](const mgs_grammar_element *pPa, const mgs_grammar_element *pPb) {
                                          return pPa < pPb; // Compare pointer addresses
                                        });
  };

  std::set<mgs_grammar_stack, decltype(vStackCmp)> sSeen(vStackCmp);

  while (!lTodo.empty()) {
    mgs_grammar_stack vCurrStack = std::move(lTodo.back());
    lTodo.pop_back();

    if (sSeen.find(vCurrStack) != sSeen.end()) {
      continue;
    }
    sSeen.insert(vCurrStack);

    if (vCurrStack.empty()) {
      if (std::find(pNewStacks.begin(), pNewStacks.end(), vCurrStack) == pNewStacks.end()) {
        pNewStacks.emplace_back(std::move(vCurrStack));
      }
      continue;
    }

    const mgs_grammar_element *vPos = vCurrStack.back();

    switch (vPos->vType) {
    case cMgsGretypeRuleRef: {
      const size_t cRuleId = static_cast<size_t>(vPos->vValue);
      const mgs_grammar_element *vSubpos = pRules[cRuleId].data();
      do {
        // init new stack without the top (pos)
        mgs_grammar_stack vNextStack(vCurrStack.begin(), vCurrStack.end() - 1);
        if (!fMgsGrammarIsEndOfSequence(vPos + 1)) {
          // if this rule ref is followed by another element, add that to stack
          vNextStack.push_back(vPos + 1);
        }
        if (!fMgsGrammarIsEndOfSequence(vSubpos)) {
          // if alternate is nonempty, add to stack
          vNextStack.push_back(vSubpos);
        }
        lTodo.push_back(std::move(vNextStack));
        while (!fMgsGrammarIsEndOfSequence(vSubpos)) {
          // scan to end of alternate def
          vSubpos++;
        }
        if (vSubpos->vType == cMgsGretypeAlt) {
          // there's another alternate def of this rule to process
          vSubpos++;
        } else {
          break;
        }
      } while (true);
      break;
    }
    case cMgsGretypeChar:
    case cMgsGretypeCharNot:
    case cMgsGretypeCharAny:
    case cMgsGretypeToken:
    case cMgsGretypeTokenNot:
      if (std::find(pNewStacks.begin(), pNewStacks.end(), vCurrStack) == pNewStacks.end()) {
        // only add the stack if it's not a duplicate of one we already have
        pNewStacks.emplace_back(std::move(vCurrStack));
      }
      break;
    default:
      // end of alternate (MGS_GRETYPE_END, MGS_GRETYPE_ALT) or middle of char range
      // (MGS_GRETYPE_CHAR_ALT, MGS_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
      // those
      GGML_ABORT("fatal error");
    }
  }
}

static mgs_grammar_candidates fMgsGrammarRejectCandidates(const mgs_grammar_rules &pRules,
                                                          const mgs_grammar_stacks &pStacks,
                                                          const mgs_grammar_candidates &pCandidates) {
  GGML_ASSERT(!pStacks.empty()); // REVIEW

  if (pCandidates.empty()) {
    return {};
  }

  auto vRejects = mgs_grammar_reject_candidates_for_stack(pRules, pStacks.front(), pCandidates);

  for (size_t vI = 1, vSize = pStacks.size(); vI < vSize; ++vI) {
    vRejects = mgs_grammar_reject_candidates_for_stack(pRules, pStacks[vI], vRejects);
  }

  return vRejects;
}

static bool fMgsGrammarDetectLeftRecursion(const mgs_grammar_rules &pRules, size_t pRuleIndex,
                                           std::vector<bool> *pRulesVisited, std::vector<bool> *pRulesInProgress,
                                           std::vector<bool> *pRulesMayBeEmpty) {
  if ((*pRulesInProgress)[pRuleIndex]) {
    return true;
  }

  (*pRulesInProgress)[pRuleIndex] = true;

  const mgs_grammar_rule &vRule = pRules[pRuleIndex];

  // First check if the rule might produce the empty string. This could be done combined with the second
  // step but it's more readable as two steps.
  bool vAtRuleStart = true;
  for (size_t vI = 0; vI < vRule.size(); vI++) {
    if (fMgsGrammarIsEndOfSequence(&vRule[vI])) {
      if (vAtRuleStart) {
        (*pRulesMayBeEmpty)[pRuleIndex] = true;
        break;
      }
      vAtRuleStart = true;
    } else {
      vAtRuleStart = false;
    }
  }

  // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
  // be empty)
  bool vRecurseIntoNonterminal = true;
  for (size_t vI = 0; vI < vRule.size(); vI++) {
    if (vRule[vI].vType == cMgsGretypeRuleRef && vRecurseIntoNonterminal) {
      if (fMgsGrammarDetectLeftRecursion(pRules, (size_t)vRule[vI].vValue, pRulesVisited, pRulesInProgress,
                                         pRulesMayBeEmpty)) {
        return true;
      }
      if (!((*pRulesMayBeEmpty)[(size_t)vRule[vI].vValue])) {
        vRecurseIntoNonterminal = false;
      }
    } else if (fMgsGrammarIsEndOfSequence(&vRule[vI])) {
      vRecurseIntoNonterminal = true;
    } else {
      vRecurseIntoNonterminal = false;
    }
  }

  (*pRulesInProgress)[pRuleIndex] = false;
  (*pRulesVisited)[pRuleIndex] = true;

  return false;
}

const mgs_grammar_rules &mgs_grammar_get_rules(const struct mgs_grammar *pGrammar) { return pGrammar->cRules; }

mgs_grammar_stacks &mgs_grammar_get_stacks(struct mgs_grammar *pGrammar) { return pGrammar->vStacks; }

static void fMgsGrammarAcceptChr(struct mgs_grammar &pGrammar, const mgs_grammar_stack &pStack, uint32_t pChr,
                                 mgs_grammar_stacks &pNewStacks) {
  if (pStack.empty()) {
    return;
  }

  const mgs_grammar_element *vPos = pStack.back();

  // ignore if this turns into a token
  if (vPos->vType == cMgsGretypeToken || vPos->vType == cMgsGretypeTokenNot) {
    return;
  }

  auto vMatch = fMgsGrammarMatchChar(vPos, pChr);
  if (vMatch.first) {
    mgs_grammar_stack vNewStack(pStack.begin(), pStack.end() - 1);
    if (!fMgsGrammarIsEndOfSequence(vMatch.second)) {
      vNewStack.push_back(vMatch.second);
    }
    fMgsGrammarAdvanceStack(pGrammar.cRules, vNewStack, pNewStacks);
  }
}

void mgs_grammar_accept(struct mgs_grammar *pGrammar, uint32_t pChr) {
  mgs_grammar_stacks vStacksNew;
  vStacksNew.reserve(pGrammar->vStacks.size());

  for (const auto &vStack : pGrammar->vStacks) {
    fMgsGrammarAcceptChr(*pGrammar, vStack, pChr, vStacksNew);
  }

  pGrammar->vStacks = std::move(vStacksNew);
}

mgs_grammar_candidates mgs_grammar_reject_candidates_for_stack(const mgs_grammar_rules &pRules,
                                                               const mgs_grammar_stack &pStack,
                                                               const mgs_grammar_candidates &pCandidates) {

  mgs_grammar_candidates vRejects;
  vRejects.reserve(pCandidates.size());

  if (pStack.empty()) {
    for (const auto &vTok : pCandidates) {
      if (*vTok.vCodePoints != 0 || vTok.vPartialUtf8.vNRemain != 0) {
        vRejects.push_back(vTok);
      }
    }
    return vRejects;
  }

  const mgs_grammar_element *vStackPos = pStack.back();

  // if the top of the stack is a token rule, then we only need to check the token id
  if (vStackPos->vType == cMgsGretypeToken || vStackPos->vType == cMgsGretypeTokenNot) {
    for (const auto &vTok : pCandidates) {
      if (*vTok.vCodePoints == 0) {
        // reached the end of a token consumed by char rules, reject iff it ended
        // in a partial response
        if (vTok.vPartialUtf8.vNRemain != 0) {
          vRejects.push_back(vTok);
        }
      } else if (!fMgsGrammarMatchToken(vStackPos, vTok.vId)) {
        vRejects.push_back(vTok);
      }
    }
    return vRejects;
  }

  mgs_grammar_candidates vNextCandidates;
  vNextCandidates.reserve(pCandidates.size());

  for (const auto &vTok : pCandidates) {
    if (*vTok.vCodePoints == 0) {
      // reached end of full codepoints in token, reject iff it ended in a partial sequence
      // that cannot satisfy this position in grammar
      if (vTok.vPartialUtf8.vNRemain != 0 && !fMgsGrammarMatchPartialChar(vStackPos, vTok.vPartialUtf8)) {
        vRejects.push_back(vTok);
      }
    } else if (fMgsGrammarMatchChar(vStackPos, *vTok.vCodePoints).first) {
      vNextCandidates.push_back({vTok.vIndex, vTok.vCodePoints + 1, vTok.vPartialUtf8, vTok.vId});
    } else {
      vRejects.push_back(vTok);
    }
  }

  const auto *vStackPosAfter = fMgsGrammarMatchChar(vStackPos, 0).second;

  // update top of stack to next element, if any
  mgs_grammar_stack vStackAfter(pStack.begin(), pStack.end() - 1);
  if (!fMgsGrammarIsEndOfSequence(vStackPosAfter)) {
    vStackAfter.push_back(vStackPosAfter);
  }
  mgs_grammar_stacks vNextStacks;
  fMgsGrammarAdvanceStack(pRules, vStackAfter, vNextStacks);

  auto vNextRejects = fMgsGrammarRejectCandidates(pRules, vNextStacks, vNextCandidates);
  for (const auto &vTok : vNextRejects) {
    vRejects.push_back({vTok.vIndex, vTok.vCodePoints - 1, vTok.vPartialUtf8, vTok.vId});
  }

  return vRejects;
}

////////////////////

struct mgs_grammar *mgs_grammar_init_impl(const struct mgs_vocab *pVocab, const mgs_grammar_element **pRules,
                                          size_t pNRules, size_t pStartRuleIndex) {
  const mgs_grammar_element *vPos;

  // copy rule definitions into vectors
  mgs_grammar_rules vVecRules(pNRules);
  for (size_t vI = 0; vI < pNRules; vI++) {
    for (vPos = pRules[vI]; vPos->vType != cMgsGretypeEnd; vPos++) {
      vVecRules[vI].push_back(*vPos);
    }
    vVecRules[vI].push_back({cMgsGretypeEnd, 0});
  }

  // Validate that all rule references point to valid rules
  for (size_t vI = 0; vI < pNRules; vI++) {
    for (const auto &vElem : vVecRules[vI]) {
      if (vElem.vType == cMgsGretypeRuleRef) {
        if (vElem.vValue >= pNRules || vVecRules[vElem.vValue].empty()) {
          MGS_LOG_ERROR("invalid grammar: rule %zu references undefined rule %u\n", vI, vElem.vValue);
          return nullptr;
        }
      }
    }
  }

  // Check for left recursion
  std::vector<bool> lRulesVisited(pNRules);
  std::vector<bool> lRulesInProgress(pNRules);
  std::vector<bool> lRulesMayBeEmpty(pNRules);
  for (size_t vI = 0; vI < pNRules; vI++) {
    if (lRulesVisited[vI]) {
      continue;
    }
    if (fMgsGrammarDetectLeftRecursion(vVecRules, vI, &lRulesVisited, &lRulesInProgress, &lRulesMayBeEmpty)) {
      MGS_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", vI);
      return nullptr;
    }
  }

  // loop over alternates of start rule to build initial stacks
  mgs_grammar_stacks vStacks;
  vPos = vVecRules[pStartRuleIndex].data();
  do {
    mgs_grammar_stack vStack;
    if (!fMgsGrammarIsEndOfSequence(vPos)) {
      // if alternate is nonempty, add to stack
      vStack.push_back(vPos);
    }
    fMgsGrammarAdvanceStack(vVecRules, vStack, vStacks);
    while (!fMgsGrammarIsEndOfSequence(vPos)) {
      // scan to end of alternate def
      vPos++;
    }
    if (vPos->vType == cMgsGretypeAlt) {
      // there's another alternate def of this rule to process
      vPos++;
    } else {
      break;
    }
  } while (true);

  // Important: vec_rules has to be moved here, not copied, because stacks contains
  // pointers to elements of vec_rules. If vec_rules were copied into mgs_grammar
  // then the pointers would be invalidated when the local vec_rules goes out of scope.
  return new mgs_grammar{
      pVocab,
      std::move(vVecRules),
      std::move(vStacks),
      /* .vPartialUtf8 = */ {},
      /* .vLazy = */ false,
      /* .vAwaitingTrigger = */ false,
      /* .vTriggerBuffer = */ "",
      /* .vTriggerBufferPositions = */ {},
      /* .vTriggerTokens = */ {},
      /* .vTriggerPatterns = */ {},
  };
}

struct mgs_grammar *mgs_grammar_init_impl(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                          const char *pGrammarRoot, bool pLazy, const char **pTriggerPatterns,
                                          size_t pNumTriggerPatterns, const mgs_token *pTriggerTokens,
                                          size_t pNumTriggerTokens) {
  mgs_grammar_parser vParser(pVocab);

  // if there is a grammar, parse it
  // rules will be empty (default) if there are parse errors
  if (!vParser.mParse(pGrammarStr) || vParser.vRules.empty()) {
    MGS_LOG_ERROR("failed to parse grammar\n");
    return nullptr;
  }

  // Ensure that the grammar contains the start symbol
  if (vParser.dSymbolIds.find(pGrammarRoot) == vParser.dSymbolIds.end()) {
    MGS_LOG_ERROR("grammar does not contain a '%s' symbol\n", pGrammarRoot);
    return nullptr;
  }

  std::vector<const mgs_grammar_element *> lGrammarRules(vParser.mCRules());

  const size_t cNRules = lGrammarRules.size();
  const size_t cStartRuleIndex = vParser.dSymbolIds.at(pGrammarRoot);

  const mgs_grammar_element *vPos;

  // copy rule definitions into vectors
  mgs_grammar_rules vVecRules(cNRules);
  for (size_t vI = 0; vI < cNRules; vI++) {
    for (vPos = lGrammarRules[vI]; vPos->vType != cMgsGretypeEnd; vPos++) {
      vVecRules[vI].push_back(*vPos);
    }
    vVecRules[vI].push_back({cMgsGretypeEnd, 0});
  }

  // Check for left recursion
  std::vector<bool> lRulesVisited(cNRules);
  std::vector<bool> lRulesInProgress(cNRules);
  std::vector<bool> lRulesMayBeEmpty(cNRules);
  for (size_t vI = 0; vI < cNRules; vI++) {
    if (lRulesVisited[vI]) {
      continue;
    }
    if (fMgsGrammarDetectLeftRecursion(vVecRules, vI, &lRulesVisited, &lRulesInProgress, &lRulesMayBeEmpty)) {
      MGS_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu\n", vI);
      return nullptr;
    }
  }

  // loop over alternates of start rule to build initial stacks
  mgs_grammar_stacks vStacks;
  vPos = vVecRules[cStartRuleIndex].data();
  do {
    mgs_grammar_stack vStack;
    if (!fMgsGrammarIsEndOfSequence(vPos)) {
      // if alternate is nonempty, add to stack
      vStack.push_back(vPos);
    }
    fMgsGrammarAdvanceStack(vVecRules, vStack, vStacks);
    while (!fMgsGrammarIsEndOfSequence(vPos)) {
      // scan to end of alternate def
      vPos++;
    }
    if (vPos->vType == cMgsGretypeAlt) {
      // there's another alternate def of this rule to process
      vPos++;
    } else {
      break;
    }
  } while (true);

  std::vector<mgs_token> lVecTriggerTokens;
  std::vector<mgs_grammar_trigger_pattern> lVecTriggerPatterns;
  for (size_t vI = 0; vI < pNumTriggerTokens; vI++) {
    GGML_ASSERT(pTriggerTokens != nullptr);
    lVecTriggerTokens.push_back(pTriggerTokens[vI]);
  }
  for (size_t vI = 0; vI < pNumTriggerPatterns; vI++) {
    GGML_ASSERT(pTriggerPatterns != nullptr);
    auto &vTrigger = lVecTriggerPatterns.emplace_back();
    vTrigger.vPattern = pTriggerPatterns[vI];
    vTrigger.vRegex = std::regex(vTrigger.vPattern);
  }

  // Important: vec_rules has to be moved here, not copied, because stacks contains
  // pointers to elements of vec_rules. If vec_rules were copied into mgs_grammar
  // then the pointers would be invalidated when the local vec_rules goes out of scope.
  return new mgs_grammar{
      pVocab,
      std::move(vVecRules),
      std::move(vStacks),
      /* .vPartialUtf8 = */ {},
      /* .vLazy = */ pLazy,
      /* .vAwaitingTrigger = */ pLazy,
      /* .vTriggerBuffer = */ "",
      /* .vTriggerBufferPositions = */ {},
      std::move(lVecTriggerTokens),
      std::move(lVecTriggerPatterns),
  };
}

void mgs_grammar_free_impl(struct mgs_grammar *pGrammar) {
  if (pGrammar == nullptr) {
    return;
  }

  delete pGrammar;
}

struct mgs_grammar *mgs_grammar_clone_impl(const struct mgs_grammar &pGrammar) {
  auto *vResult = new mgs_grammar{
      pGrammar.vVocab,         pGrammar.cRules,           pGrammar.vStacks,        pGrammar.vPartialUtf8,
      pGrammar.vLazy,          pGrammar.vAwaitingTrigger, pGrammar.vTriggerBuffer, pGrammar.vTriggerBufferPositions,
      pGrammar.vTriggerTokens, pGrammar.vTriggerPatterns,
  };

  // redirect elements in stacks to point to new rules
  for (size_t vIs = 0; vIs < vResult->vStacks.size(); vIs++) {
    for (size_t vIe = 0; vIe < vResult->vStacks[vIs].size(); vIe++) {
      for (size_t vIr0 = 0; vIr0 < pGrammar.cRules.size(); vIr0++) {
        for (size_t vIr1 = 0; vIr1 < pGrammar.cRules[vIr0].size(); vIr1++) {
          if (pGrammar.vStacks[vIs][vIe] == &pGrammar.cRules[vIr0][vIr1]) {
            vResult->vStacks[vIs][vIe] = &vResult->cRules[vIr0][vIr1];
          }
        }
      }
    }
  }

  return vResult;
}

void mgs_grammar_apply_impl(const struct mgs_grammar &pGrammar, mgs_token_data_array *pCurP) {
  GGML_ASSERT(pGrammar.vVocab != nullptr);

  if (pGrammar.vAwaitingTrigger) {
    return;
  }

  bool vAllowEog = false;
  for (const auto &vStack : pGrammar.vStacks) {
    if (vStack.empty()) {
      vAllowEog = true;
      break;
    }
  }

  std::vector<std::pair<std::vector<uint32_t>, mgs_partial_utf8>> lCandidatesDecoded;
  lCandidatesDecoded.reserve(pCurP->size);

  mgs_grammar_candidates vCandidatesGrammar;
  vCandidatesGrammar.reserve(pCurP->size);

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const mgs_token cId = pCurP->data[vI].id;
    const std::string &vPiece = pGrammar.vVocab->mTokenToPiece(cId);

    if (pGrammar.vVocab->mIsEog(cId)) {
      if (!vAllowEog) {
        pCurP->data[vI].logit = -INFINITY;
      }
    } else if (vPiece.empty() || vPiece[0] == 0) {
      pCurP->data[vI].logit = -INFINITY;
    } else {
      lCandidatesDecoded.push_back(fDecodeUtf8(vPiece, pGrammar.vPartialUtf8));
      vCandidatesGrammar.push_back({vI, lCandidatesDecoded.back().first.data(), lCandidatesDecoded.back().second, cId});
    }
  }

  const auto cRejects = fMgsGrammarRejectCandidates(pGrammar.cRules, pGrammar.vStacks, vCandidatesGrammar);
  for (const auto &vReject : cRejects) {
    pCurP->data[vReject.vIndex].logit = -INFINITY;
  }
}

void mgs_grammar_accept_impl(struct mgs_grammar &pGrammar, mgs_token pToken) {
  GGML_ASSERT(pGrammar.vVocab != nullptr);

  const auto &vPiece = pGrammar.vVocab->mTokenToPiece(pToken);

  if (pGrammar.vAwaitingTrigger) {
    if (std::find(pGrammar.vTriggerTokens.begin(), pGrammar.vTriggerTokens.end(), pToken) !=
        pGrammar.vTriggerTokens.end()) {
      pGrammar.vAwaitingTrigger = false;
      pGrammar.vTriggerBuffer.clear();
      mgs_grammar_accept_token(pGrammar, pToken, vPiece);
      MGS_LOG_DEBUG("Grammar triggered on token %u (`%s`)", pToken, vPiece.c_str());
      return;
    } else {
      auto vPosition = std::make_pair(pGrammar.vTriggerBuffer.size(), pGrammar.vTriggerBuffer.size() + vPiece.size());
      pGrammar.vTriggerBufferPositions.push_back(std::make_pair(pToken, vPosition));
      pGrammar.vTriggerBuffer += vPiece;

      for (const auto &vTriggerPattern : pGrammar.vTriggerPatterns) {
        auto vStart = vTriggerPattern.mFind(pGrammar.vTriggerBuffer);
        if (vStart != std::string::npos) {
          pGrammar.vAwaitingTrigger = false;

          // replay tokens that overlap with [start, end)
          for (const auto &[vTok, vTokPos] : pGrammar.vTriggerBufferPositions) {
            auto [vTokStart, vTokEnd] = vTokPos;
            if (vTokEnd <= vStart) {
              continue;
            }

            size_t vPieceStart = (vTokStart < vStart) ? vStart : vTokStart; // allow for partial token pieces
            size_t vPieceLen = vTokEnd - vPieceStart;
            auto vTokPiece = pGrammar.vTriggerBuffer.substr(vPieceStart, vPieceLen);
            mgs_grammar_accept_token(pGrammar, vTok, vTokPiece);
          }

          auto vConstrainedStr = pGrammar.vTriggerBuffer.substr(vStart);
          pGrammar.vTriggerBuffer.clear();
          pGrammar.vTriggerBufferPositions.clear();
          MGS_LOG_DEBUG("Grammar triggered on regex: '%s'\n", vConstrainedStr.c_str());
          return;
        }
      }
      MGS_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", pToken, vPiece.c_str());
      return;
    }
  }

  if (pGrammar.vVocab->mIsEog(pToken)) {
    for (const auto &vStack : pGrammar.vStacks) {
      if (vStack.empty()) {
        return;
      }
    }
    GGML_ABORT("fatal error");
  }

  mgs_grammar_accept_token(pGrammar, pToken, vPiece);
}

void mgs_grammar_accept_str(struct mgs_grammar &pGrammar, const std::string &pPiece) {
  // Note terminating 0 in decoded string
  const auto cDecoded = fDecodeUtf8(pPiece, pGrammar.vPartialUtf8);
  const auto &vCodePoints = cDecoded.first;

  for (auto vIt = vCodePoints.begin(), vEnd = vCodePoints.end() - 1; vIt != vEnd; ++vIt) {
    mgs_grammar_accept(&pGrammar, *vIt);
  }

  pGrammar.vPartialUtf8 = cDecoded.second;
  if (pGrammar.vStacks.empty()) {
    throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + pPiece);
  }
}

void mgs_grammar_accept_token(struct mgs_grammar &pGrammar, mgs_token pToken, const std::string &pPiece) {
  // Note terminating 0 in decoded string
  const auto cDecoded = fDecodeUtf8(pPiece, pGrammar.vPartialUtf8);
  const auto &vCodePoints = cDecoded.first;

  mgs_grammar_stacks vStacksNew;
  vStacksNew.reserve(pGrammar.vStacks.size());

  for (const auto &vStack : pGrammar.vStacks) {
    if (vStack.empty()) {
      continue;
    }

    const mgs_grammar_element *vPos = vStack.back();

    if (vPos->vType == cMgsGretypeToken || vPos->vType == cMgsGretypeTokenNot) {
      if (fMgsGrammarMatchToken(vPos, pToken)) {
        mgs_grammar_stack vNewStack(vStack.begin(), vStack.end() - 1);
        if (!fMgsGrammarIsEndOfSequence(vPos + 1)) {
          vNewStack.push_back(vPos + 1);
        }
        fMgsGrammarAdvanceStack(pGrammar.cRules, vNewStack, vStacksNew);
      }
    } else {
      mgs_grammar_stacks vCurrentStacks = {vStack};

      for (auto vIt = vCodePoints.begin(), vEnd = vCodePoints.end() - 1; vIt != vEnd; ++vIt) {
        mgs_grammar_stacks vNextStacks;

        for (const auto &vCurStack : vCurrentStacks) {
          fMgsGrammarAcceptChr(pGrammar, vCurStack, *vIt, vNextStacks);
        }

        vCurrentStacks = std::move(vNextStacks);
        if (vCurrentStacks.empty()) {
          break;
        }
      }

      for (auto &vSurvivingStack : vCurrentStacks) {
        if (std::find(vStacksNew.begin(), vStacksNew.end(), vSurvivingStack) == vStacksNew.end()) {
          vStacksNew.emplace_back(vSurvivingStack);
        }
      }
    }
  }

  pGrammar.vStacks = std::move(vStacksNew);
  pGrammar.vPartialUtf8 = cDecoded.second;

  if (pGrammar.vStacks.empty()) {
    throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + pPiece + " (" +
                             std::to_string(pToken) + ")");
  }
}
