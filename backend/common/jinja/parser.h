#pragma once

#include "lexer.h"
#include "runtime.h"
#include "utils.h"

#include <string>
#include <stdexcept>

namespace jinja {

// parse from a list of tokens into an AST (program)
// may throw parser_exception on error
program fParseFromTokens(const lexer_result &pLexerRes);

struct parser_exception : public std::runtime_error {
  parser_exception(const std::string &pMsg, const std::string &pSource, size_t pPos)
      : std::runtime_error(fFmtErrorWithSource("parser", pMsg, pSource, pPos)) {}
};

} // namespace jinja
