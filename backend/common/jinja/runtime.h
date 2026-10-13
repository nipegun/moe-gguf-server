#pragma once

#include "lexer.h"
#include "value.h"

#include <cassert>
#include <ctime>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define JJ_DEBUG(msg, ...)                                                                                             \
  do {                                                                                                                 \
    if (vGJinjaDebug)                                                                                                  \
      printf("%s:%-3d : " msg "\n", FILENAME, __LINE__, __VA_ARGS__);                                                  \
  } while (0)

extern bool vGJinjaDebug;

namespace jinja {

struct statement;
using statement_ptr = std::unique_ptr<statement>;
using statements = std::vector<statement_ptr>;

// Helpers for dynamic casting and type checking
template <typename T> struct extract_pointee_unique {
  using type = T;
};
template <typename U> struct extract_pointee_unique<std::unique_ptr<U>> {
  using type = U;
};
template <typename T> bool fIsStmt(const statement_ptr &pPtr) { return dynamic_cast<const T *>(pPtr.get()) != nullptr; }
template <typename T> T *fCastStmt(statement_ptr &pPtr) { return dynamic_cast<T *>(pPtr.get()); }
template <typename T> const T *fCastStmt(const statement_ptr &pPtr) { return dynamic_cast<const T *>(pPtr.get()); }
// End Helpers

// not thread-safe
void fEnableDebug(bool pEnable);

// for visiting AST nodes
// function signature: void(bool is_leaf, statement * node, pair of <label, children>)
using visitor_pair = std::pair<std::string, std::vector<statement *>>;
using visitor_fn = std::function<void(bool, statement *, std::vector<visitor_pair>)>;

struct context {
  std::shared_ptr<std::string> vSrc; // for debugging; use shared_ptr to avoid copying on scope creation
  std::time_t vCurrentTime;          // for functions that need current time

  bool vIsGetStats = false; // whether to collect stats

  // nesting of statement executions, shared by every scope of one render (each scope is a copy of the context)
  std::shared_ptr<size_t> vExecutionDepth = std::make_shared<size_t>(0);

  visitor_fn vVisitor;

  // src is optional, used for error reporting
  context(std::string pSrc = "") : vSrc(std::make_shared<std::string>(std::move(pSrc))) {
    vEnv = fMkVal<value_object>();
    vEnv->vHasBuiltins = false; // context object has no builtins
    vEnv->insert("true", fMkVal<value_bool>(true));
    vEnv->insert("True", fMkVal<value_bool>(true));
    vEnv->insert("false", fMkVal<value_bool>(false));
    vEnv->insert("False", fMkVal<value_bool>(false));
    vEnv->insert("none", fMkVal<value_none>());
    vEnv->insert("None", fMkVal<value_none>());
    vCurrentTime = std::time(nullptr);
  }
  ~context() = default;

  context(const context &pParent) : context() {
    // inherit variables (for example, when entering a new scope)
    auto &vPvar = pParent.vEnv->mAsOrderedObject();
    for (const auto &vPair : vPvar) {
      mSetVal(vPair.first, vPair.second);
    }
    vCurrentTime = pParent.vCurrentTime;
    vIsGetStats = pParent.vIsGetStats;
    vSrc = pParent.vSrc;
    vExecutionDepth = pParent.vExecutionDepth;
  }

  value mGetVal(const std::string &pName) {
    value vDefaultVal = fMkVal<value_undefined>(pName);
    return vEnv->mAt(pName, vDefaultVal);
  }

  void mSetVal(const std::string &pName, const value &pVal) { vEnv->insert(pName, pVal); }

  void mSetVal(const value &pName, const value &pVal) { vEnv->mInsert(pName, pVal); }

  void mPrintVars() const { printf("Context Variables:\n%s\n", fValueToJson(vEnv, 2).c_str()); }

private:
  value_object vEnv;
};

// utils for visiting AST nodes
static std::vector<statement *> fStmtsToPtr(const statements &pStmts) {
  std::vector<statement *> lChildren;
  for (const auto &vStmt : pStmts) {
    lChildren.push_back(vStmt.get());
  }
  return lChildren;
}

/**
 * Base class for all nodes in the AST.
 */
struct statement {
  size_t vPos; // position in source, for debugging
  virtual ~statement() = default;
  virtual std::string mType() const { return "Statement"; }
  virtual void mVisit(context &pCtx) { pCtx.vVisitor(true, this, {}); }

  // execute_impl must be overridden by derived classes
  virtual value mExecuteImpl(context &) { mThrowExecError(); }
  // execute is the public method to execute a statement with error handling
  value mExecute(context &);

private:
  [[noreturn]] void mThrowExecError() const { throw std::runtime_error("cannot exec " + mType()); }
};

// Type Checking Utilities

template <typename T> static void fChkType(const statement_ptr &pPtr) {
  if (!pPtr)
    return; // Allow null for optional fields
  assert(dynamic_cast<T *>(pPtr.get()) != nullptr);
}

template <typename T, typename U> static void fChkType(const statement_ptr &pPtr) {
  if (!pPtr)
    return;
  assert(dynamic_cast<T *>(pPtr.get()) != nullptr || dynamic_cast<U *>(pPtr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
  std::string mType() const override { return "Expression"; }
};

// Statements

struct program : public statement {
  statements vBody;

  program() = default;
  explicit program(statements &&pBody) : vBody(std::move(pBody)) {}
  std::string mType() const override { return "Program"; }
  [[noreturn]] value mExecuteImpl(context &) override {
    throw std::runtime_error("Cannot execute program directly, use jinja::runtime instead");
  }
};

struct if_statement : public statement {
  statement_ptr vTest;
  statements vBody;
  statements vAlternate;

  if_statement(statement_ptr &&pTest, statements &&pBody, statements &&pAlternate)
      : vTest(std::move(pTest)), vBody(std::move(pBody)), vAlternate(std::move(pAlternate)) {
    fChkType<expression>(this->vTest);
  }

  std::string mType() const override { return "If"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this,
                  {{"test", {vTest.get()}}, {"body", fStmtsToPtr(vBody)}, {"alternate", fStmtsToPtr(vAlternate)}});
  }
};

struct identifier;
struct tuple_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
  statement_ptr vLoopvar; // Identifier | TupleLiteral
  statement_ptr vIterable;
  statements vBody;
  statements vDefaultBlock; // if no iteration took place

  for_statement(statement_ptr &&pLoopvar, statement_ptr &&pIterable, statements &&pBody, statements &&pDefaultBlock)
      : vLoopvar(std::move(pLoopvar)), vIterable(std::move(pIterable)), vBody(std::move(pBody)),
        vDefaultBlock(std::move(pDefaultBlock)) {
    fChkType<identifier, tuple_literal>(this->vLoopvar);
    fChkType<expression>(this->vIterable);
  }

  std::string mType() const override { return "For"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this,
                  {{"loopvar", {vLoopvar.get()}},
                   {"iterable", {vIterable.get()}},
                   {"body", fStmtsToPtr(vBody)},
                   {"default_block", fStmtsToPtr(vDefaultBlock)}});
  }
};

struct break_statement : public statement {
  std::string mType() const override { return "Break"; }

  struct signal : public std::exception {
    const char *what() const noexcept override { return "Break statement executed"; }
  };

  [[noreturn]] value mExecuteImpl(context &) override { throw break_statement::signal(); }
};

struct continue_statement : public statement {
  std::string mType() const override { return "Continue"; }

  struct signal : public std::exception {
    const char *what() const noexcept override { return "Continue statement executed"; }
  };

  [[noreturn]] value mExecuteImpl(context &) override { throw continue_statement::signal(); }
};

// do nothing
struct noop_statement : public statement {
  std::string mType() const override { return "Noop"; }
  value mExecuteImpl(context &) override { return fMkVal<value_undefined>(); }
};

struct set_statement : public statement {
  statement_ptr vAssignee;
  statement_ptr vVal;
  statements vBody;

  set_statement(statement_ptr &&pAssignee, statement_ptr &&pValue, statements &&pBody)
      : vAssignee(std::move(pAssignee)), vVal(std::move(pValue)), vBody(std::move(pBody)) {
    fChkType<expression>(this->vAssignee);
    fChkType<expression>(this->vVal);
  }

  std::string mType() const override { return "Set"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this,
                  {{"assignee", {vAssignee.get()}}, {"value", {vVal.get()}}, {"body", fStmtsToPtr(vBody)}});
  }
};

struct macro_statement : public statement {
  statement_ptr vName;
  statements vArgs;
  statements vBody;

  macro_statement(statement_ptr &&pName, statements &&pArgs, statements &&pBody)
      : vName(std::move(pName)), vArgs(std::move(pArgs)), vBody(std::move(pBody)) {
    fChkType<identifier>(this->vName);
    for (const auto &vArg : this->vArgs)
      fChkType<expression>(vArg);
  }

  std::string mType() const override { return "Macro"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"name", {vName.get()}}, {"args", fStmtsToPtr(vArgs)}, {"body", fStmtsToPtr(vBody)}});
  }
};

struct comment_statement : public statement {
  std::string vVal;
  explicit comment_statement(const std::string &pV) : vVal(pV) {}
  std::string mType() const override { return "Comment"; }
  value mExecuteImpl(context &) override { return fMkVal<value_undefined>(); }
};

// Expressions

// Represents an omitted expression in a computed member, e.g. `a[]`.
struct blank_expression : public expression {
  std::string mType() const override { return "BlankExpression"; }
  value mExecuteImpl(context &) override { return fMkVal<value_undefined>(); }
};

struct member_expression : public expression {
  statement_ptr vObject;
  statement_ptr vProperty;
  bool vComputed; // true if obj[expr] and false if obj.prop

  member_expression(statement_ptr &&pObject, statement_ptr &&pProperty, bool pComputed)
      : vObject(std::move(pObject)), vProperty(std::move(pProperty)), vComputed(pComputed) {
    fChkType<expression>(this->vObject);
    fChkType<expression>(this->vProperty);
  }
  std::string mType() const override { return "MemberExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"object", {vObject.get()}}, {"property", {vProperty.get()}}});
  }
};

struct call_expression : public expression {
  statement_ptr vCallee;
  statements vArgs;

  call_expression(statement_ptr &&pCallee, statements &&pArgs) : vCallee(std::move(pCallee)), vArgs(std::move(pArgs)) {
    fChkType<expression>(this->vCallee);
    for (const auto &vArg : this->vArgs)
      fChkType<expression>(vArg);
  }
  std::string mType() const override { return "CallExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"callee", {vCallee.get()}}, {"args", fStmtsToPtr(vArgs)}});
  }
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
  std::string vVal;
  explicit identifier(const std::string &pVal) : vVal(pVal) {}
  std::string mType() const override { return "Identifier"; }
  value mExecuteImpl(context &pCtx) override;
};

// Literals

struct integer_literal : public expression {
  int64_t vVal;
  explicit integer_literal(int64_t pVal) : vVal(pVal) {}
  std::string mType() const override { return "IntegerLiteral"; }
  value mExecuteImpl(context &) override { return fMkVal<value_int>(vVal); }
};

struct float_literal : public expression {
  double vVal;
  explicit float_literal(double pVal) : vVal(pVal) {}
  std::string mType() const override { return "FloatLiteral"; }
  value mExecuteImpl(context &) override { return fMkVal<value_float>(vVal); }
};

struct cStringLiteral : public expression {
  std::string vVal;
  explicit cStringLiteral(const std::string &pVal) : vVal(pVal) {}
  std::string mType() const override { return "StringLiteral"; }
  value mExecuteImpl(context &) override { return fMkVal<value_string>(vVal); }
};

struct array_literal : public expression {
  statements vVal;
  explicit array_literal(statements &&pVal) : vVal(std::move(pVal)) {
    for (const auto &vItem : this->vVal)
      fChkType<expression>(vItem);
  }
  std::string mType() const override { return "ArrayLiteral"; }
  value mExecuteImpl(context &pCtx) override {
    auto vArr = fMkVal<value_array>();
    for (const auto &vItemStmt : vVal) {
      vArr->mPushBack(vItemStmt->mExecute(pCtx));
    }
    return vArr;
  }
};

struct tuple_literal : public expression {
  statements vVal;
  explicit tuple_literal(statements &&pVal) : vVal(std::move(pVal)) {
    for (const auto &vItem : this->vVal)
      fChkType<expression>(vItem);
  }
  std::string mType() const override { return "TupleLiteral"; }
  value mExecuteImpl(context &pCtx) override {
    auto vArr = fMkVal<value_array>();
    for (const auto &vItemStmt : vVal) {
      vArr->mPushBack(vItemStmt->mExecute(pCtx));
    }
    return fMkVal<value_tuple>(std::move(vArr->mAsArray()));
  }
};

struct object_literal : public expression {
  std::vector<std::pair<statement_ptr, statement_ptr>> lVal;
  explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>> &&pVal) : lVal(std::move(pVal)) {
    for (const auto &vPair : this->lVal) {
      fChkType<expression>(vPair.first);
      fChkType<expression>(vPair.second);
    }
  }
  std::string mType() const override { return "ObjectLiteral"; }
  value mExecuteImpl(context &pCtx) override;
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
  token vOp;
  statement_ptr vLeft;
  statement_ptr vRight;

  binary_expression(token pOp, statement_ptr &&pLeft, statement_ptr &&pRight)
      : vOp(std::move(pOp)), vLeft(std::move(pLeft)), vRight(std::move(pRight)) {
    fChkType<expression>(this->vLeft);
    fChkType<expression>(this->vRight);
  }
  std::string mType() const override { return "BinaryExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"left", {vLeft.get()}}, {"right", {vRight.get()}}});
  }
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
  // either an expression or a value is allowed
  statement_ptr vOperand;
  value_string vVal; // will be set by filter_statement

  statement_ptr vFilter;

  filter_expression(statement_ptr &&pOperand, statement_ptr &&pFilter)
      : vOperand(std::move(pOperand)), vFilter(std::move(pFilter)) {
    fChkType<expression>(this->vOperand);
    fChkType<identifier, call_expression>(this->vFilter);
  }

  filter_expression(value_string &&pVal, statement_ptr &&pFilter) : vVal(std::move(pVal)), vFilter(std::move(pFilter)) {
    fChkType<identifier, call_expression>(this->vFilter);
  }

  std::string mType() const override { return "FilterExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"operand", {vOperand.get()}}, {"filter", {vFilter.get()}}});
  }
};

struct filter_statement : public statement {
  statement_ptr vFilter;
  statements vBody;

  filter_statement(statement_ptr &&pFilter, statements &&pBody) : vFilter(std::move(pFilter)), vBody(std::move(pBody)) {
    fChkType<identifier, call_expression>(this->vFilter);
  }
  std::string mType() const override { return "FilterStatement"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"filter", {vFilter.get()}}, {"body", fStmtsToPtr(vBody)}});
  }
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
  statement_ptr vLhs;
  statement_ptr vTest;

  select_expression(statement_ptr &&pLhs, statement_ptr &&pTest) : vLhs(std::move(pLhs)), vTest(std::move(pTest)) {
    fChkType<expression>(this->vLhs);
    fChkType<expression>(this->vTest);
  }
  std::string mType() const override { return "SelectExpression"; }
  value mExecuteImpl(context &pCtx) override {
    auto vPredicate = vTest->mExecuteImpl(pCtx);
    if (!vPredicate->mAsBool()) {
      return fMkVal<value_undefined>();
    }
    return vLhs->mExecuteImpl(pCtx);
  }
  void mVisit(context &pCtx) override { pCtx.vVisitor(false, this, {{"lhs", {vLhs.get()}}, {"test", {vTest.get()}}}); }
};

/**
 * An operation with two sides, separated by the "is" operator.
 * NOTE: "value is something" translates to function call "test_is_something(value)"
 */
struct test_expression : public expression {
  statement_ptr vOperand;
  bool vNegate;
  statement_ptr vTest;

  test_expression(statement_ptr &&pOperand, bool pNegate, statement_ptr &&pTest)
      : vOperand(std::move(pOperand)), vNegate(pNegate), vTest(std::move(pTest)) {
    fChkType<expression>(this->vOperand);
    fChkType<identifier, call_expression>(this->vTest);
  }
  std::string mType() const override { return "TestExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this, {{"operand", {vOperand.get()}}, {"test", {vTest.get()}}});
  }
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
  token vOp;
  statement_ptr vArgument;

  unary_expression(token pOp, statement_ptr &&pArgument) : vOp(std::move(pOp)), vArgument(std::move(pArgument)) {
    fChkType<expression>(this->vArgument);
  }
  std::string mType() const override { return "UnaryExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override { pCtx.vVisitor(false, this, {{"argument", {vArgument.get()}}}); }
};

struct slice_expression : public expression {
  statement_ptr vStartExpr;
  statement_ptr vStopExpr;
  statement_ptr vStepExpr;

  slice_expression(statement_ptr &&pStartExpr, statement_ptr &&pStopExpr, statement_ptr &&pStepExpr)
      : vStartExpr(std::move(pStartExpr)), vStopExpr(std::move(pStopExpr)), vStepExpr(std::move(pStepExpr)) {
    fChkType<expression>(this->vStartExpr);
    fChkType<expression>(this->vStopExpr);
    fChkType<expression>(this->vStepExpr);
  }
  std::string mType() const override { return "SliceExpression"; }
  [[noreturn]] value mExecuteImpl(context &) override {
    throw std::runtime_error("must be handled by MemberExpression");
  }
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(
        false, this,
        {{"start_expr", {vStartExpr.get()}}, {"stop_expr", {vStopExpr.get()}}, {"step_expr", {vStepExpr.get()}}});
  }
};

struct keyword_argument_expression : public expression {
  statement_ptr vKey;
  statement_ptr vVal;

  keyword_argument_expression(statement_ptr &&pKey, statement_ptr &&pVal)
      : vKey(std::move(pKey)), vVal(std::move(pVal)) {
    fChkType<identifier>(this->vKey);
    fChkType<expression>(this->vVal);
  }
  std::string mType() const override { return "KeywordArgumentExpression"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override { pCtx.vVisitor(false, this, {{"key", {vKey.get()}}, {"val", {vVal.get()}}}); }
};

struct spread_expression : public expression {
  statement_ptr vArgument;
  explicit spread_expression(statement_ptr &&pArgument) : vArgument(std::move(pArgument)) {
    fChkType<expression>(this->vArgument);
  }
  std::string mType() const override { return "SpreadExpression"; }
  void mVisit(context &pCtx) override { pCtx.vVisitor(false, this, {{"argument", {vArgument.get()}}}); }
};

struct call_statement : public statement {
  statement_ptr vCall;
  statements vCallerArgs;
  statements vBody;

  call_statement(statement_ptr &&pCall, statements &&pCallerArgs, statements &&pBody)
      : vCall(std::move(pCall)), vCallerArgs(std::move(pCallerArgs)), vBody(std::move(pBody)) {
    fChkType<call_expression>(this->vCall);
    for (const auto &vArg : this->vCallerArgs)
      fChkType<expression>(vArg);
  }
  std::string mType() const override { return "CallStatement"; }
  value mExecuteImpl(context &pCtx) override;
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(false, this,
                  {{"call", {vCall.get()}}, {"caller_args", fStmtsToPtr(vCallerArgs)}, {"body", fStmtsToPtr(vBody)}});
  }
};

struct ternary_expression : public expression {
  statement_ptr vCondition;
  statement_ptr vTrueExpr;
  statement_ptr vFalseExpr;

  ternary_expression(statement_ptr &&pCondition, statement_ptr &&pTrueExpr, statement_ptr &&pFalseExpr)
      : vCondition(std::move(pCondition)), vTrueExpr(std::move(pTrueExpr)), vFalseExpr(std::move(pFalseExpr)) {
    fChkType<expression>(this->vCondition);
    fChkType<expression>(this->vTrueExpr);
    fChkType<expression>(this->vFalseExpr);
  }
  std::string mType() const override { return "Ternary"; }
  value mExecuteImpl(context &pCtx) override {
    value vCondVal = vCondition->mExecute(pCtx);
    if (vCondVal->mAsBool()) {
      return vTrueExpr->mExecute(pCtx);
    } else {
      return vFalseExpr->mExecute(pCtx);
    }
  }
  void mVisit(context &pCtx) override {
    pCtx.vVisitor(
        false, this,
        {{"condition", {vCondition.get()}}, {"true_expr", {vTrueExpr.get()}}, {"false_expr", {vFalseExpr.get()}}});
  }
};

struct raised_exception : public std::exception {
  std::string vMessage;
  raised_exception(const std::string &pMsg) : vMessage(pMsg) {}
  const char *what() const noexcept override { return vMessage.c_str(); }
};

// Used to rethrow exceptions with modified messages
struct rethrown_exception : public std::exception {
  std::string vMessage;
  rethrown_exception(const std::string &pMsg) : vMessage(pMsg) {}
  const char *what() const noexcept override { return vMessage.c_str(); }
};

//////////////////////

static void fGatherStringPartsRecursive(const value &pVal, value_string &pParts) {
  // TODO: probably allow print value_none as "None" string? currently this breaks some templates
  if (fIsVal<value_string>(pVal)) {
    const auto &vStrVal = fCastVal<value_string>(pVal)->vValStr;
    pParts->vValStr.mAppend(vStrVal);
  } else if (fIsVal<value_int>(pVal) || fIsVal<value_float>(pVal) || fIsVal<value_bool>(pVal)) {
    std::string vStrVal = pVal->mAsString().mStr();
    pParts->vValStr.mAppend(vStrVal);
  } else if (fIsVal<value_array>(pVal)) {
    auto vItems = fCastVal<value_array>(pVal)->mAsArray();
    for (const auto &vItem : vItems) {
      fGatherStringPartsRecursive(vItem, pParts);
    }
  }
}

static std::string fRenderStringParts(const value_string &pParts) {
  std::ostringstream vOss;
  for (const auto &vPart : pParts->vValStr.lParts) {
    vOss << vPart.vVal;
  }
  return vOss.str();
}

struct runtime {
  context &vCtx;
  explicit runtime(context &pCtx) : vCtx(pCtx) {}

  value_array mExecute(const program &pProg) {
    value_array vResults = fMkVal<value_array>();
    for (const auto &vStmt : pProg.vBody) {
      value vRes = vStmt->mExecute(vCtx);
      vResults->mPushBack(std::move(vRes));
    }
    return vResults;
  }

  static value_string mGatherStringParts(const value &pVal) {
    value_string vParts = fMkVal<value_string>();
    fGatherStringPartsRecursive(pVal, vParts);
    // join consecutive parts with the same type
    auto &vP = vParts->vValStr.lParts;
    for (size_t vI = 1; vI < vP.size();) {
      if (vP[vI].vIsInput == vP[vI - 1].vIsInput) {
        vP[vI - 1].vVal += vP[vI].vVal;
        vP.erase(vP.begin() + vI);
      } else {
        vI++;
      }
    }
    return vParts;
  }

  static std::string mDebugDumpProgram(const program &pProg, const std::string &pSrc);
};

} // namespace jinja
