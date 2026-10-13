#include "lexer.h"
#include "runtime.h"

#include <cctype>
#include <functional>
#include <map>
#include <string>
#include <vector>

#define FILENAME "jinja-lexer"

namespace jinja {

static void fStringLstrip(std::string &pS, const char *pChars) {
  size_t vStart = pS.find_first_not_of(pChars);
  if (vStart == std::string::npos) {
    pS.clear();
  } else {
    pS.erase(0, vStart);
  }
}

static void fStringRstrip(std::string &pS, const char *pChars) {
  size_t vEnd = pS.find_last_not_of(pChars);
  if (vEnd == std::string::npos) {
    pS.clear();
  } else {
    pS.erase(vEnd + 1);
  }
}

lexer_result lexer::mTokenize(const std::string &pSource) {
  std::vector<token> lTokens;

  // NOTE: do NOT transform the source string (i.e. preprocessing), as we need to keep
  //       the original character positions for error reporting etc.
  std::string vSrc = pSource;

  if (pSource.empty()) {
    return {lTokens, vSrc};
  }

  // Normalize \r\n or \r to \n
  for (std::string::size_type vPos = 0; (vPos = vSrc.find("\r\n", vPos)) != std::string::npos;) {
    vSrc.erase(vPos, 1);
    ++vPos;
  }
  for (std::string::size_type vPos = 0; (vPos = vSrc.find("\r", vPos)) != std::string::npos;) {
    vSrc.replace(vPos, 1, 1, '\n');
    ++vPos;
  }

  // In the default configuration:
  //  - a single trailing newline is stripped if present
  //  - other whitespace (spaces, tabs, newlines etc.) is returned unchanged
  if (pSource.back() == '\n') {
    vSrc.pop_back();
  }

  size_t vPos = 0;
  size_t vStartPos = 0;
  size_t vCurlyBracketDepth = 0;

  using pred = std::function<bool(char)>;
  auto vConsumeWhile = [&](const pred &pPredicate) -> std::string {
    std::string vStr;
    while (pPredicate(vSrc[vPos])) {
      // check for escape char
      if (vSrc[vPos] == '\\') {
        // consume backslash
        ++vPos;
        // check for end of input
        if (vPos >= vSrc.size()) {
          throw lexer_exception("unexpected end of input after escape character", pSource, vPos);
        }
        // add escaped char
        char vEscapedChar = vSrc[vPos++];
        if (dEscapeChars.find(vEscapedChar) == dEscapeChars.end()) {
          throw lexer_exception(std::string("unknown escape character \\") + vEscapedChar, pSource, vPos);
        }
        char vUnescapedChar = dEscapeChars.at(vEscapedChar);
        vStr += vUnescapedChar;
        continue;
      }

      vStr += vSrc[vPos++];
      if (vPos > vSrc.size()) {
        throw lexer_exception("unexpected end of input during consume_while", pSource, vPos);
      }
    }
    return vStr;
  };

  auto vConsumeNumeric = [&]() -> std::string {
    std::string vNum = vConsumeWhile(mIsInteger);
    if (vPos < vSrc.size() && vSrc[vPos] == '.' && vPos + 1 < vSrc.size() && mIsInteger(vSrc[vPos + 1])) {
      ++vPos; // Consume '.'
      std::string vFrac = vConsumeWhile(mIsInteger);
      vNum += "." + vFrac;
    }
    return vNum;
  };

  auto vNextPosIs = [&](std::initializer_list<char> pChars, size_t pN = 1) -> bool {
    if (vPos + pN >= vSrc.size())
      return false;
    for (char vC : pChars) {
      if (vSrc[vPos + pN] == vC)
        return true;
    }
    return false;
  };

  // note: default config for chat template: lstrip_blocks = true, trim_blocks = true

  // text\n[space]{block} --> text\n{block}
  bool vOptLstripBlocks = true;

  // {block}\n[space]text --> {block}[space]text
  bool vOptTrimBlocks = true;

  // options set dynamically based on current/last block
  bool vIsLstripBlock = false; // example: {%-
  bool vIsRstripBlock = false; // example: -%}

  while (vPos < vSrc.size()) {
    vStartPos = vPos;
    // JJ_DEBUG("lexer main loop at pos %zu: '%s...'", vPos, vSrc.substr(vPos, 10).c_str());

    // First, consume all text that is outside of a Jinja statement or expression
    token::type vLastTokenType = lTokens.empty() ? token::cCloseStatement // initial state
                                                 : lTokens.back().vT;
    if (vLastTokenType == token::cCloseStatement || vLastTokenType == token::cCloseExpression ||
        vLastTokenType == token::cComment) {

      bool vLastBlockCanRmNewline = false;
      vIsRstripBlock = false;
      if (vPos > 3) {
        char vC0 = vSrc[vPos - 3];
        char vC1 = vSrc[vPos - 2];
        char vC2 = vSrc[vPos - 1];
        // strip if: -[%}#]}text
        vIsRstripBlock = vC0 == '-' && (vC1 == '%' || vC1 == '}' || vC1 == '#') && vC2 == '}';
        // match behavior of hf.js: exclude {{ and }} cases, regex: ([#%-]})
        vLastBlockCanRmNewline = (vC1 == '#' || vC1 == '%' || vC1 == '-') && vC2 == '}';
      }

      size_t vStart = vPos;
      size_t vEnd = vStart;
      while (vPos < vSrc.size() &&
             // Keep going until we hit the next Jinja statement or expression
             !(vSrc[vPos] == '{' && vNextPosIs({'%', '{', '#'}))) {
        vEnd = ++vPos;
      }

      // equivalent to hf.js code: template.replace(/^[ \t]*({[#%-])/gm, "$1");
      if (vOptLstripBlocks && vSrc[vPos] == '{' && vNextPosIs({'%', '#', '-'})) {
        size_t vCurrent = vEnd;
        while (vCurrent > vStart) {
          char vC = vSrc[vCurrent - 1];
          if (vCurrent == 1) {
            vEnd = 0; // Trim from the start of the string
            break;
          }
          if (vC == '\n') {
            vEnd = vCurrent; // Trim from the start of the line
            break;
          }
          if (!std::isspace(static_cast<unsigned char>(vC))) {
            break; // Found non-whitespace before newline, keep
          }
          --vCurrent;
        }
      }

      std::string vText = vSrc.substr(vStart, vEnd - vStart);

      // equivalent to hf.js code: template.replace(/([#%-]})\n/g, "$1");
      if (vOptTrimBlocks && vLastBlockCanRmNewline) {
        if (!vText.empty() && vText.front() == '\n') {
          vText.erase(vText.begin());
        }
      }

      if (vIsRstripBlock) {
        // example: {last_block}[space]text
        // doing lstrip on text, effectively rstrip the LAST block
        // JJ_DEBUG("RSTRIP block detected, current text: '%s'", vText.c_str());
        fStringLstrip(vText, " \t\r\n");
      }

      vIsLstripBlock = vSrc[vPos] == '{' && vNextPosIs({'{', '%', '#'}) && vNextPosIs({'-'}, 2);
      if (vIsLstripBlock) {
        // example: text[space]{current_block}
        // doing rstrip on text, effectively lstrip the CURRENT block
        // JJ_DEBUG("LSTRIP block detected, current text: '%s'", vText.c_str());
        fStringRstrip(vText, " \t\r\n");
      }

      if (!vText.empty()) {
        // JJ_DEBUG("consumed text: '%s'", vText.c_str());
        lTokens.push_back({token::cText, vText, vStartPos});
        continue;
      }
    }

    // Possibly consume a comment
    // TODO: handle lstrip/rstrip for comments? (not important for now)
    if (vSrc[vPos] == '{' && vNextPosIs({'#'})) {
      vStartPos = vPos;
      vPos += 2; // Skip the opening {#
      std::string vComment;
      while (!(vSrc[vPos] == '#' && vNextPosIs({'}'}))) {
        if (vPos + 2 >= vSrc.size()) {
          throw lexer_exception("missing end of comment tag", pSource, vPos);
        }
        vComment += vSrc[vPos++];
      }
      JJ_DEBUG("consumed comment: '%s'", vComment.c_str());
      lTokens.push_back({token::cComment, vComment, vStartPos});
      vPos += 2; // Skip the closing #}
      continue;
    }

    if (vSrc[vPos] == '-' && (vLastTokenType == token::cOpenExpression || vLastTokenType == token::cOpenStatement)) {
      JJ_DEBUG("lexer main loop at pos %zu: '%s...'", vPos, vSrc.substr(vPos, 10).c_str());
      vPos++; // consume '-' in {%- or {{-
      if (vPos >= vSrc.size())
        break;
    }

    // Consume (and ignore) all whitespace inside Jinja statements or expressions
    vConsumeWhile([](char pC) { return std::isspace(static_cast<unsigned char>(pC)); });

    if (vPos >= vSrc.size())
      break;

    char vCh = vSrc[vPos];

    bool vIsClosingBlock = vCh == '-' && vNextPosIs({'%', '}'});

    // Check for unary operators
    if (!vIsClosingBlock && (vCh == '-' || vCh == '+')) {
      vStartPos = vPos;
      token::type vLastTokenType = lTokens.empty() ? token::cEof : lTokens.back().vT;
      if (vLastTokenType == token::cText || vLastTokenType == token::cEof) {
        throw lexer_exception(std::string("unexpected character: ") + vCh, pSource, vPos);
      }
      switch (vLastTokenType) {
      case token::cIdentifier:
      case token::cNumericLiteral:
      case token::cStringLiteral:
      case token::cCloseParen:
      case token::cCloseSquareBracket:
        // Part of a binary operator
        // a - 1, 1 - 1, true - 1, "apple" - 1, (1) - 1, a[1] - 1
        // Continue parsing normally
        break;
      default: {
        // Is part of a unary operator
        // (-1), [-1], (1 + -1), not -1, -apple
        ++vPos; // Consume the operator

        // Check for numbers following the unary operator
        std::string vNum = vConsumeNumeric();
        std::string vValue = std::string(1, vCh) + vNum;
        token::type vT = vNum.empty() ? token::cUnaryOperator : token::cNumericLiteral;
        // JJ_DEBUG("consumed unary operator or numeric literal: '%s'", vValue.c_str());
        lTokens.push_back({vT, vValue, vStartPos});
        continue;
      }
      }
    }

    // Try to match one of the tokens in the mapping table
    bool vMatched = false;
    for (const auto &[vSeq, vTyp] : lOrderedMappingTable) {
      vStartPos = vPos;
      // Inside an object literal, don't treat "}}" as expression-end
      if (vSeq == "}}" && vCurlyBracketDepth > 0) {
        continue;
      }
      if (vPos + vSeq.size() <= vSrc.size() && vSrc.substr(vPos, vSeq.size()) == vSeq) {
        lTokens.push_back({vTyp, vSeq, vStartPos});
        if (vTyp == token::cOpenExpression) {
          vCurlyBracketDepth = 0;
        } else if (vTyp == token::cOpenCurlyBracket) {
          ++vCurlyBracketDepth;
        } else if (vTyp == token::cCloseCurlyBracket) {
          --vCurlyBracketDepth;
        }

        vPos += vSeq.size();
        vMatched = true;
        break; // continue main loop
      }
    }
    if (vMatched)
      continue; // continue main loop

    // Strings
    if (vCh == '\'' || vCh == '"') {
      vStartPos = vPos;
      ++vPos; // Skip opening quote
      std::string vStr = vConsumeWhile([vCh](char pC) { return pC != vCh; });
      // JJ_DEBUG("consumed string literal: '%s'", vStr.c_str());
      lTokens.push_back({token::cStringLiteral, vStr, vStartPos});
      ++vPos; // Skip closing quote
      continue;
    }

    // Numbers
    if (mIsInteger(vCh)) {
      vStartPos = vPos;
      std::string vNum = vConsumeNumeric();
      // JJ_DEBUG("consumed numeric literal: '%s'", vNum.c_str());
      lTokens.push_back({token::cNumericLiteral, vNum, vStartPos});
      continue;
    }

    // Identifiers
    if (mIsWord(vCh)) {
      vStartPos = vPos;
      std::string vWord = vConsumeWhile(mIsWord);
      // JJ_DEBUG("consumed identifier: '%s'", vWord.c_str());
      lTokens.push_back({token::cIdentifier, vWord, vStartPos});
      continue;
    }

    throw lexer_exception(std::string("unexpected character: ") + vCh, pSource, vPos);
  }

  return {std::move(lTokens), vSrc};
}

} // namespace jinja
