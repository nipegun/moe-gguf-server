#include "lexer.h"
#include "runtime.h"
#include "parser.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define FILENAME "jinja-parser"

namespace jinja {

// Helper to check type without asserting (useful for logic)
template <typename T> static bool fIsType(const statement_ptr &pPtr) {
  return dynamic_cast<const T *>(pPtr.get()) != nullptr;
}

class parser {
  const std::vector<token> &lTokens;
  size_t vCurrent = 0;

  std::string vSource; // for error reporting

  // Nesting of blocks, expressions and operator chains: every level becomes a nested node that the parser, the
  // runtime and the destructors walk recursively, so a template nested too deeply is rejected before it can
  // exhaust the native stack (templates come from model files and are untrusted)
  size_t vNesting = 0;

  // Restores the nesting when the node being parsed is complete, so that only the levels above it count
  struct nesting_scope {
    size_t &vNesting;
    const size_t cSaved;
    explicit nesting_scope(size_t &pNesting) : vNesting(pNesting), cSaved(pNesting) {}
    ~nesting_scope() { vNesting = cSaved; }
    nesting_scope(const nesting_scope &) = delete;
    nesting_scope &operator=(const nesting_scope &) = delete;
  };

public:
  parser(const std::vector<token> &pT, const std::string &pSrc) : lTokens(pT), vSource(pSrc) {}

  program mParse() {
    statements vBody;
    while (vCurrent < lTokens.size()) {
      vBody.push_back(mParseAny());
    }
    return program(std::move(vBody));
  }

  // NOTE: start_pos is the token index, used for error reporting
  template <typename T, typename... Args> std::unique_ptr<T> mMkStmt(size_t pStartPos, Args &&...pArgs) {
    auto vPtr = std::make_unique<T>(std::forward<Args>(pArgs)...);
    assert(pStartPos < lTokens.size());
    vPtr->vPos = lTokens[pStartPos].vPos;
    return vPtr;
  }

private:
  const token &mPeek(size_t pOffset = 0) const {
    if (vCurrent + pOffset >= lTokens.size()) {
      static const token cEndToken{token::cEof, "", 0};
      return cEndToken;
    }
    return lTokens[vCurrent + pOffset];
  }

  const token &mNext() {
    if (vCurrent >= lTokens.size()) {
      throw parser_exception("Parser Error: Unexpected EOF", vSource, lTokens.empty() ? 0 : lTokens.back().vPos);
    }
    return lTokens[vCurrent++];
  }

  token mExpect(token::type pType, const std::string &pError) {
    const auto &vT = mPeek();
    if (vT.vT != pType) {
      throw parser_exception("Parser Error: " + pError + " (Got " + vT.vValue + ")", vSource, vT.vPos);
    }
    vCurrent++;
    return vT;
  }

  void mExpectIdentifier(const std::string &pName) {
    const auto &vT = mPeek();
    if (vT.vT != token::cIdentifier || vT.vValue != pName) {
      throw parser_exception("Expected identifier: " + pName, vSource, vT.vPos);
    }
    vCurrent++;
  }

  // One more level of nesting for the node being built, until its nesting_scope ends
  void mEnterNesting() {
    if (++vNesting > cMaxParseDepth) {
      throw parser_exception("Parser Error: template nesting exceeds the maximum depth of " +
                                 std::to_string(cMaxParseDepth),
                             vSource, mPeek().vPos);
    }
  }

  bool mIs(token::type pType) const { return mPeek().vT == pType; }

  bool mIsIdentifier(const std::string &pName) const {
    return mPeek().vT == token::cIdentifier && mPeek().vValue == pName;
  }

  bool mIsStatement(const std::vector<std::string> &pNames) const {
    if (mPeek(0).vT != token::cOpenStatement || mPeek(1).vT != token::cIdentifier) {
      return false;
    }
    std::string vVal = mPeek(1).vValue;
    return std::find(pNames.begin(), pNames.end(), vVal) != pNames.end();
  }

  statement_ptr mParseAny() {
    nesting_scope vNestingScope(vNesting);
    mEnterNesting();
    size_t vStartPos = vCurrent;
    switch (mPeek().vT) {
    case token::cComment:
      return mMkStmt<comment_statement>(vStartPos, mNext().vValue);
    case token::cText:
      return mMkStmt<cStringLiteral>(vStartPos, mNext().vValue);
    case token::cOpenStatement:
      return mParseJinjaStatement();
    case token::cOpenExpression:
      return mParseJinjaExpression();
    default:
      throw std::runtime_error("Unexpected token type");
    }
  }

  statement_ptr mParseJinjaExpression() {
    // Consume {{ }} tokens
    mExpect(token::cOpenExpression, "Expected {{");
    auto vResult = mParseExpression();
    mExpect(token::cCloseExpression, "Expected }}");
    return vResult;
  }

  statement_ptr mParseJinjaStatement() {
    // Consume {% token
    mExpect(token::cOpenStatement, "Expected {%");

    if (mPeek().vT != token::cIdentifier) {
      throw std::runtime_error("Unknown statement");
    }

    size_t vStartPos = vCurrent;
    std::string vName = mNext().vValue;

    statement_ptr vResult;
    if (vName == "set") {
      vResult = mParseSetStatement(vStartPos);

    } else if (vName == "if") {
      vResult = mParseIfStatement(vStartPos);
      // expect {% endif %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endif");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (vName == "macro") {
      vResult = mParseMacroStatement(vStartPos);
      // expect {% endmacro %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endmacro");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (vName == "for") {
      vResult = mParseForStatement(vStartPos);
      // expect {% endfor %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endfor");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (vName == "break") {
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<break_statement>(vStartPos);

    } else if (vName == "continue") {
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<continue_statement>(vStartPos);

    } else if (vName == "call") {
      statements vCallerArgs;
      // bool vHasCallerArgs = false;
      if (mIs(token::cOpenParen)) {
        // Optional caller arguments, e.g. {% call(user) dump_users(...) %}
        vCallerArgs = mParseArgs();
        // vHasCallerArgs = true;
      }
      auto vCallee = mParsePrimaryExpression();
      if (!fIsType<identifier>(vCallee))
        throw std::runtime_error("Expected identifier");

      auto vCallArgs = mParseArgs();
      mExpect(token::cCloseStatement, "Expected %}");

      statements vBody;
      while (!mIsStatement({"endcall"})) {
        vBody.push_back(mParseAny());
      }

      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endcall");
      mExpect(token::cCloseStatement, "Expected %}");

      auto vCallExpr = mMkStmt<call_expression>(vStartPos, std::move(vCallee), std::move(vCallArgs));
      vResult = mMkStmt<call_statement>(vStartPos, std::move(vCallExpr), std::move(vCallerArgs), std::move(vBody));

    } else if (vName == "filter") {
      auto vFilterNode = mParsePrimaryExpression();
      if (fIsType<identifier>(vFilterNode) && mIs(token::cOpenParen)) {
        vFilterNode = mParseCallExpression(std::move(vFilterNode));
      }
      mExpect(token::cCloseStatement, "Expected %}");

      statements vBody;
      while (!mIsStatement({"endfilter"})) {
        vBody.push_back(mParseAny());
      }

      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endfilter");
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<filter_statement>(vStartPos, std::move(vFilterNode), std::move(vBody));

    } else if (vName == "generation" || vName == "endgeneration") {
      // Ignore generation blocks (transformers-specific)
      // See https://github.com/huggingface/transformers/pull/30650 for more information.
      vResult = mMkStmt<noop_statement>(vStartPos);
      ++vCurrent;

    } else {
      throw std::runtime_error("Unknown statement: " + vName);
    }
    return vResult;
  }

  statement_ptr mParseSetStatement(size_t pStartPos) {
    // NOTE: `set` acts as both declaration statement and assignment expression
    auto vLeft = mParseExpressionSequence();
    statement_ptr vValue = nullptr;
    statements vBody;

    if (mIs(token::cEquals)) {
      ++vCurrent;
      vValue = mParseExpressionSequence();
    } else {
      // parsing multiline set here
      mExpect(token::cCloseStatement, "Expected %}");
      while (!mIsStatement({"endset"})) {
        vBody.push_back(mParseAny());
      }
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endset");
    }
    mExpect(token::cCloseStatement, "Expected %}");
    return mMkStmt<set_statement>(pStartPos, std::move(vLeft), std::move(vValue), std::move(vBody));
  }

  statement_ptr mParseIfStatement(size_t pStartPos) {
    // every {% elif %} nests another if statement
    nesting_scope vNestingScope(vNesting);
    mEnterNesting();
    auto vTest = mParseExpression();
    mExpect(token::cCloseStatement, "Expected %}");

    statements vBody;
    statements vAlternate;

    // Keep parsing 'if' body until we reach the first {% elif %} or {% else %} or {% endif %}
    while (!mIsStatement({"elif", "else", "endif"})) {
      vBody.push_back(mParseAny());
    }

    if (mIsStatement({"elif"})) {
      size_t vPos0 = vCurrent;
      ++vCurrent;                                     // consume {%
      ++vCurrent;                                     // consume 'elif'
      vAlternate.push_back(mParseIfStatement(vPos0)); // nested If
    } else if (mIsStatement({"else"})) {
      ++vCurrent; // consume {%
      ++vCurrent; // consume 'else'
      mExpect(token::cCloseStatement, "Expected %}");

      // keep going until we hit {% endif %}
      while (!mIsStatement({"endif"})) {
        vAlternate.push_back(mParseAny());
      }
    }
    return mMkStmt<if_statement>(pStartPos, std::move(vTest), std::move(vBody), std::move(vAlternate));
  }

  statement_ptr mParseMacroStatement(size_t pStartPos) {
    auto vName = mParsePrimaryExpression();
    auto vArgs = mParseArgs();
    mExpect(token::cCloseStatement, "Expected %}");
    statements vBody;
    // Keep going until we hit {% endmacro
    while (!mIsStatement({"endmacro"})) {
      vBody.push_back(mParseAny());
    }
    return mMkStmt<macro_statement>(pStartPos, std::move(vName), std::move(vArgs), std::move(vBody));
  }

  statement_ptr mParseExpressionSequence(bool pPrimary = false) {
    size_t vStartPos = vCurrent;
    statements vExprs;
    vExprs.push_back(pPrimary ? mParsePrimaryExpression() : mParseExpression());
    bool vIsTuple = mIs(token::cComma);
    while (mIs(token::cComma)) {
      ++vCurrent; // consume comma
      vExprs.push_back(pPrimary ? mParsePrimaryExpression() : mParseExpression());
    }
    return vIsTuple ? mMkStmt<tuple_literal>(vStartPos, std::move(vExprs)) : std::move(vExprs[0]);
  }

  statement_ptr mParseForStatement(size_t pStartPos) {
    // e.g., `message` in `for message in messages`
    auto vLoopVar = mParseExpressionSequence(true); // should be an identifier/tuple
    if (!mIsIdentifier("in"))
      throw std::runtime_error("Expected 'in'");
    ++vCurrent; // consume 'in'

    // `messages` in `for message in messages`
    auto vIterable = mParseExpression();
    mExpect(token::cCloseStatement, "Expected %}");

    statements vBody;
    statements vAlternate;

    // Keep going until we hit {% endfor or {% else
    while (!mIsStatement({"endfor", "else"})) {
      vBody.push_back(mParseAny());
    }

    if (mIsStatement({"else"})) {
      ++vCurrent; // consume {%
      ++vCurrent; // consume 'else'
      mExpect(token::cCloseStatement, "Expected %}");
      while (!mIsStatement({"endfor"})) {
        vAlternate.push_back(mParseAny());
      }
    }
    return mMkStmt<for_statement>(pStartPos, std::move(vLoopVar), std::move(vIterable), std::move(vBody),
                                  std::move(vAlternate));
  }

  statement_ptr mParseExpression() {
    nesting_scope vNestingScope(vNesting);
    mEnterNesting();
    // Choose parse function with lowest precedence
    return mParseIfExpression();
  }

  statement_ptr mParseIfExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vA = mParseLogicalOrExpression();
    if (mIsIdentifier("if")) {
      // Ternary expression
      size_t vStartPos = vCurrent;
      ++vCurrent; // consume 'if'
      auto vTest = mParseLogicalOrExpression();
      if (mIsIdentifier("else")) {
        // Ternary expression with else
        size_t vPos0 = vCurrent;
        ++vCurrent; // consume 'else'
        mEnterNesting();
        auto vFalseExpr = mParseIfExpression(); // recurse to support chained ternaries
        return mMkStmt<ternary_expression>(vPos0, std::move(vTest), std::move(vA), std::move(vFalseExpr));
      } else {
        // Select expression on iterable
        return mMkStmt<select_expression>(vStartPos, std::move(vA), std::move(vTest));
      }
    }
    return vA;
  }

  statement_ptr mParseLogicalOrExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vLeft = mParseLogicalAndExpression();
    while (mIsIdentifier("or")) {
      mEnterNesting(); // every operator of the chain nests the previous result one level deeper
      size_t vStartPos = vCurrent;
      token vOp = mNext();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseLogicalAndExpression());
    }
    return vLeft;
  }

  statement_ptr mParseLogicalAndExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vLeft = mParseLogicalNegationExpression();
    while (mIsIdentifier("and")) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      auto vOp = mNext();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseLogicalNegationExpression());
    }
    return vLeft;
  }

  statement_ptr mParseLogicalNegationExpression() {
    nesting_scope vNestingScope(vNesting);
    // Try parse unary operators
    if (mIsIdentifier("not")) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      auto vOp = mNext();
      return mMkStmt<unary_expression>(vStartPos, vOp, mParseLogicalNegationExpression());
    }
    return mParseComparisonExpression();
  }

  statement_ptr mParseComparisonExpression() {
    // NOTE: membership has same precedence as comparison
    // e.g., ('a' in 'apple' == 'b' in 'banana') evaluates as ('a' in ('apple' == ('b' in 'banana')))
    nesting_scope vNestingScope(vNesting);
    auto vLeft = mParseAdditiveExpression();
    while (true) {
      token vOp;
      size_t vStartPos = vCurrent;
      if (mIsIdentifier("not") && mPeek(1).vT == token::cIdentifier && mPeek(1).vValue == "in") {
        vOp = {token::cIdentifier, "not in", lTokens[vCurrent].vPos};
        ++vCurrent; // consume 'not'
        ++vCurrent; // consume 'in'
      } else if (mIsIdentifier("in")) {
        vOp = mNext();
      } else if (mIs(token::cComparisonBinaryOperator)) {
        vOp = mNext();
      } else
        break;
      mEnterNesting();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseAdditiveExpression());
    }
    return vLeft;
  }

  statement_ptr mParseAdditiveExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vLeft = mParseMultiplicativeExpression();
    while (mIs(token::cAdditiveBinaryOperator)) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      auto vOp = mNext();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseMultiplicativeExpression());
    }
    return vLeft;
  }

  statement_ptr mParseMultiplicativeExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vLeft = mParseTestExpression();
    while (mIs(token::cMultiplicativeBinaryOperator)) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      auto vOp = mNext();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseTestExpression());
    }
    return vLeft;
  }

  statement_ptr mParseTestExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vOperand = mParseFilterExpression();
    while (mIsIdentifier("is")) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      ++vCurrent; // consume 'is'
      bool vNegate = false;
      if (mIsIdentifier("not")) {
        ++vCurrent;
        vNegate = true;
      }
      auto vTestId = mParsePrimaryExpression();
      // FIXME: tests can also be expressed like this: if x is eq 3
      if (mIs(token::cOpenParen))
        vTestId = mParseCallExpression(std::move(vTestId));
      vOperand = mMkStmt<test_expression>(vStartPos, std::move(vOperand), vNegate, std::move(vTestId));
    }
    return vOperand;
  }

  statement_ptr mParseFilterExpression() {
    nesting_scope vNestingScope(vNesting);
    auto vOperand = mParseCallMemberExpression();
    while (mIs(token::cPipe)) {
      mEnterNesting();
      size_t vStartPos = vCurrent;
      ++vCurrent; // consume pipe
      auto vFilter = mParsePrimaryExpression();
      if (mIs(token::cOpenParen))
        vFilter = mParseCallExpression(std::move(vFilter));
      vOperand = mMkStmt<filter_expression>(vStartPos, std::move(vOperand), std::move(vFilter));
    }
    return vOperand;
  }

  statement_ptr mParseCallMemberExpression() {
    // Handle member expressions recursively
    auto vMember = mParseMemberExpression(mParsePrimaryExpression());
    return mIs(token::cOpenParen) ? mParseCallExpression(std::move(vMember)) // foo.x()
                                  : std::move(vMember);
  }

  statement_ptr mParseCallExpression(statement_ptr pCallee) {
    // every call of a chain such as foo()()() nests the previous one
    nesting_scope vNestingScope(vNesting);
    mEnterNesting();
    size_t vStartPos = vCurrent;
    auto vExpr = mMkStmt<call_expression>(vStartPos, std::move(pCallee), mParseArgs());
    auto vMember = mParseMemberExpression(std::move(vExpr));                 // foo.x().y
    return mIs(token::cOpenParen) ? mParseCallExpression(std::move(vMember)) // foo.x()()
                                  : std::move(vMember);
  }

  statements mParseArgs() {
    // comma-separated arguments list
    mExpect(token::cOpenParen, "Expected (");
    statements vArgs;
    while (!mIs(token::cCloseParen)) {
      statement_ptr vArg;
      // unpacking: *expr
      if (mPeek().vT == token::cMultiplicativeBinaryOperator && mPeek().vValue == "*") {
        size_t vStartPos = vCurrent;
        ++vCurrent; // consume *
        vArg = mMkStmt<spread_expression>(vStartPos, mParseExpression());
      } else {
        vArg = mParseExpression();
        if (mIs(token::cEquals)) {
          // keyword argument
          // e.g., func(x = 5, y = a or b)
          size_t vStartPos = vCurrent;
          ++vCurrent; // consume equals
          vArg = mMkStmt<keyword_argument_expression>(vStartPos, std::move(vArg), mParseExpression());
        }
      }
      vArgs.push_back(std::move(vArg));
      if (mIs(token::cComma)) {
        ++vCurrent; // consume comma
      }
    }
    mExpect(token::cCloseParen, "Expected )");
    return vArgs;
  }

  statement_ptr mParseMemberExpression(statement_ptr pObject) {
    nesting_scope vNestingScope(vNesting);
    size_t vStartPos = vCurrent;
    while (mIs(token::cDot) || mIs(token::cOpenSquareBracket)) {
      mEnterNesting();
      auto vOp = mNext();
      bool vComputed = vOp.vT == token::cOpenSquareBracket;
      statement_ptr vProp;
      if (vComputed) {
        vProp = mParseMemberExpressionArguments();
        mExpect(token::cCloseSquareBracket, "Expected ]");
      } else {
        vProp = mParsePrimaryExpression();
      }
      pObject = mMkStmt<member_expression>(vStartPos, std::move(pObject), std::move(vProp), vComputed);
    }
    return pObject;
  }

  statement_ptr mParseMemberExpressionArguments() {
    // NOTE: This also handles slice expressions colon-separated arguments list
    // e.g., ['test'], [0], [:2], [1:], [1:2], [1:2:3]
    statements vSlices;
    bool vIsSlice = false;
    size_t vStartPos = vCurrent;
    while (!mIs(token::cCloseSquareBracket)) {
      if (mIs(token::cColon)) {
        // A case where a default is used
        // e.g., [:2] will be parsed as [undefined, 2]
        vSlices.push_back(nullptr);
        ++vCurrent; // consume colon
        vIsSlice = true;
      } else {
        vSlices.push_back(mParseExpression());
        if (mIs(token::cColon)) {
          ++vCurrent; // consume colon after expression, if it exists
          vIsSlice = true;
        }
      }
    }
    if (vIsSlice) {
      statement_ptr vStart = vSlices.size() > 0 ? std::move(vSlices[0]) : nullptr;
      statement_ptr vStop = vSlices.size() > 1 ? std::move(vSlices[1]) : nullptr;
      statement_ptr vStep = vSlices.size() > 2 ? std::move(vSlices[2]) : nullptr;
      return mMkStmt<slice_expression>(vStartPos, std::move(vStart), std::move(vStop), std::move(vStep));
    }
    if (vSlices.empty()) {
      return mMkStmt<blank_expression>(vStartPos);
    }
    return std::move(vSlices[0]);
  }

  statement_ptr mParsePrimaryExpression() {
    size_t vStartPos = vCurrent;
    auto vT = mNext();
    switch (vT.vT) {
    case token::cNumericLiteral:
      if (vT.vValue.find('.') != std::string::npos) {
        return mMkStmt<float_literal>(vStartPos, std::stod(vT.vValue));
      } else {
        return mMkStmt<integer_literal>(vStartPos, std::stoll(vT.vValue));
      }
    case token::cStringLiteral: {
      std::string vVal = vT.vValue;
      while (mIs(token::cStringLiteral)) {
        vVal += mNext().vValue;
      }
      return mMkStmt<cStringLiteral>(vStartPos, vVal);
    }
    case token::cIdentifier:
      return mMkStmt<identifier>(vStartPos, vT.vValue);
    case token::cOpenParen: {
      auto vExpr = mParseExpressionSequence();
      mExpect(token::cCloseParen, "Expected )");
      return vExpr;
    }
    case token::cOpenSquareBracket: {
      statements vVals;
      while (!mIs(token::cCloseSquareBracket)) {
        vVals.push_back(mParseExpression());
        if (mIs(token::cComma))
          ++vCurrent;
      }
      ++vCurrent;
      return mMkStmt<array_literal>(vStartPos, std::move(vVals));
    }
    case token::cOpenCurlyBracket: {
      std::vector<std::pair<statement_ptr, statement_ptr>> lPairs;
      while (!mIs(token::cCloseCurlyBracket)) {
        auto vKey = mParseExpression();
        mExpect(token::cColon, "Expected :");
        lPairs.push_back({std::move(vKey), mParseExpression()});
        if (mIs(token::cComma))
          ++vCurrent;
      }
      ++vCurrent;
      return mMkStmt<object_literal>(vStartPos, std::move(lPairs));
    }
    default:
      throw std::runtime_error("Unexpected token: " + vT.vValue + " of type " + std::to_string(vT.vT));
    }
  }
};

program fParseFromTokens(const lexer_result &pLexerRes) {
  return parser(pLexerRes.lTokens, pLexerRes.vSource).mParse();
}

} // namespace jinja
