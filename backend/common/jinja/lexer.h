#pragma once

#include "utils.h"

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace jinja {

struct token {
  enum type {
    cEof,  // end of source
    cText, // The text between Jinja statements or expressions

    cNumericLiteral,     // e.g., 123, 1.0
    cStringLiteral,      // 'string'
    cIdentifier,         // Variables, functions, statements, booleans, etc.
    cEquals,             // =
    cOpenParen,          // (
    cCloseParen,         // )
    cOpenStatement,      // {%
    cCloseStatement,     // %}
    cOpenExpression,     // {{
    cCloseExpression,    // }}
    cOpenSquareBracket,  // [
    cCloseSquareBracket, // ]
    cOpenCurlyBracket,   // {
    cCloseCurlyBracket,  // }
    cComma,              // ,
    cDot,                // .
    cColon,              // :
    cPipe,               // |

    cCallOperator,                 // ()
    cAdditiveBinaryOperator,       // + - ~
    cMultiplicativeBinaryOperator, // * / %
    cComparisonBinaryOperator,     // < > <= >= == !=
    cUnaryOperator,                // ! - +
    cComment,                      // {# ... #}
  };
  type vT;
  std::string vValue;
  size_t vPos;
};

static std::string fTypeToString(token::type pT) {
  switch (pT) {
  case token::cEof:
    return "eof";
  case token::cText:
    return "text";
  case token::cNumericLiteral:
    return "numeric_literal";
  case token::cStringLiteral:
    return "string_literal";
  case token::cIdentifier:
    return "identifier";
  case token::cEquals:
    return "equals";
  case token::cOpenParen:
    return "open_paren";
  case token::cCloseParen:
    return "close_paren";
  case token::cOpenStatement:
    return "open_statement";
  case token::cCloseStatement:
    return "close_statement";
  case token::cOpenExpression:
    return "open_expression";
  case token::cCloseExpression:
    return "close_expression";
  case token::cOpenSquareBracket:
    return "open_square_bracket";
  case token::cCloseSquareBracket:
    return "close_square_bracket";
  case token::cOpenCurlyBracket:
    return "open_curly_bracket";
  case token::cCloseCurlyBracket:
    return "close_curly_bracket";
  case token::cComma:
    return "comma";
  case token::cDot:
    return "dot";
  case token::cColon:
    return "colon";
  case token::cPipe:
    return "pipe";
  case token::cCallOperator:
    return "call_operator";
  case token::cAdditiveBinaryOperator:
    return "additive_binary_operator";
  case token::cMultiplicativeBinaryOperator:
    return "multiplicative_binary_operator";
  case token::cComparisonBinaryOperator:
    return "comparison_binary_operator";
  case token::cUnaryOperator:
    return "unary_operator";
  case token::cComment:
    return "comment";
  default:
    return "unknown";
  }
}

struct lexer_result {
  std::vector<token> lTokens;
  std::string vSource;
};

struct lexer {
  const std::map<char, char> dEscapeChars = {
      {'n', '\n'}, {'t', '\t'},  {'r', '\r'},  {'b', '\b'},  {'f', '\f'},
      {'v', '\v'}, {'\\', '\\'}, {'\'', '\''}, {'\"', '\"'},
  };

  static bool mIsWord(char pC) { return std::isalnum(static_cast<unsigned char>(pC)) || pC == '_'; }

  static bool mIsInteger(char pC) { return std::isdigit(static_cast<unsigned char>(pC)); }

  const std::vector<std::pair<std::string, token::type>> lOrderedMappingTable = {
      // Trimmed control sequences
      {"{%-", token::cOpenStatement},
      {"-%}", token::cCloseStatement},
      {"{{-", token::cOpenExpression},
      {"-}}", token::cCloseExpression},
      // Control sequences
      {"{%", token::cOpenStatement},
      {"%}", token::cCloseStatement},
      {"{{", token::cOpenExpression},
      {"}}", token::cCloseExpression},
      // Single character tokens
      {"(", token::cOpenParen},
      {")", token::cCloseParen},
      {"{", token::cOpenCurlyBracket},
      {"}", token::cCloseCurlyBracket},
      {"[", token::cOpenSquareBracket},
      {"]", token::cCloseSquareBracket},
      {",", token::cComma},
      {".", token::cDot},
      {":", token::cColon},
      {"|", token::cPipe},
      // Comparison operators
      {"<=", token::cComparisonBinaryOperator},
      {">=", token::cComparisonBinaryOperator},
      {"==", token::cComparisonBinaryOperator},
      {"!=", token::cComparisonBinaryOperator},
      {"<", token::cComparisonBinaryOperator},
      {">", token::cComparisonBinaryOperator},
      // Arithmetic operators
      {"+", token::cAdditiveBinaryOperator},
      {"-", token::cAdditiveBinaryOperator},
      {"~", token::cAdditiveBinaryOperator},
      {"*", token::cMultiplicativeBinaryOperator},
      {"/", token::cMultiplicativeBinaryOperator},
      {"%", token::cMultiplicativeBinaryOperator},
      // Assignment operator
      {"=", token::cEquals},
  };

  // tokenize the source string into a list of tokens
  // may throw lexer_exception on error
  lexer_result mTokenize(const std::string &pSource);
};

struct lexer_exception : public std::runtime_error {
  lexer_exception(const std::string &pMsg, const std::string &pSource, size_t pPos)
      : std::runtime_error(fFmtErrorWithSource("lexer", pMsg, pSource, pPos)) {}
};

} // namespace jinja
