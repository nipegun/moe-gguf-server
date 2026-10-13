#include "lexer.h"
#include "runtime.h"
#include "value.h"
#include "utils.h"

#include <string>
#include <vector>
#include <memory>
#include <cmath>

#define FILENAME "jinja-runtime"

bool vGJinjaDebug = false;

namespace jinja {

void fEnableDebug(bool pEnable) { vGJinjaDebug = pEnable; }

static value_string fExecStatements(const statements &pStmts, context &pCtx) {
  auto vResult = fMkVal<value_array>();
  for (const auto &vStmt : pStmts) {
    JJ_DEBUG("Executing statement of type %s", vStmt->mType().c_str());
    vResult->mPushBack(vStmt->mExecute(pCtx));
  }
  // convert to string parts
  value_string vStr = fMkVal<value_string>();
  fGatherStringPartsRecursive(vResult, vStr);
  return vStr;
}

static std::string fGetLineCol(const std::string &pSource, size_t pPos) {
  size_t vLine = 1;
  size_t vCol = 1;
  for (size_t vI = 0; vI < pPos && vI < pSource.size(); vI++) {
    if (pSource[vI] == '\n') {
      vLine++;
      vCol = 1;
    } else {
      vCol++;
    }
  }
  return "line " + std::to_string(vLine) + ", column " + std::to_string(vCol);
}

static void fEnsureKeyTypeAllowed(const value &pVal) {
  if (!pVal->mIsHashable()) {
    throw std::runtime_error("Type: " + pVal->mType() + " is not allowed as object key");
  }
}

// execute with error handling
// Counts one nested statement execution for as long as it is alive
struct execution_depth_guard {
  size_t &vDepth;
  explicit execution_depth_guard(size_t &pDepth) : vDepth(pDepth) { ++vDepth; }
  ~execution_depth_guard() { --vDepth; }
  execution_depth_guard(const execution_depth_guard &) = delete;
  execution_depth_guard &operator=(const execution_depth_guard &) = delete;
};

value statement::mExecute(context &pCtx) {
  // deep expressions and recursive macros end with an error instead of exhausting the native stack
  size_t &vDepth = *pCtx.vExecutionDepth;
  if (vDepth >= cMaxExecutionDepth) {
    throw std::runtime_error("template execution exceeds the maximum nesting depth of " +
                             std::to_string(cMaxExecutionDepth));
  }
  execution_depth_guard vDepthGuard(vDepth);
  try {
    return mExecuteImpl(pCtx);
  } catch (const continue_statement::signal & /* ex */) {
    throw;
  } catch (const break_statement::signal & /* ex */) {
    throw;
  } catch (const rethrown_exception & /* ex */) {
    throw;
  } catch (const not_implemented_exception & /* ex */) {
    throw;
  } catch (const std::exception &vE) {
    const std::string &vSource = *pCtx.vSrc;
    if (vSource.empty()) {
      std::ostringstream vOss;
      vOss << "\nError executing " << mType() << " at position " << vPos << ": " << vE.what();
      throw rethrown_exception(vOss.str());
    } else {
      std::ostringstream vOss;
      vOss << "\n------------\n";
      vOss << "While executing " << mType() << " at " << fGetLineCol(vSource, vPos) << " in source:\n";
      vOss << fPeakSource(vSource, vPos) << "\n";
      vOss << "Error: " << vE.what();
      // throw as another exception to avoid repeated formatting
      throw rethrown_exception(vOss.str());
    }
  }
}

value identifier::mExecuteImpl(context &pCtx) {
  auto vIt = pCtx.mGetVal(vVal);
  auto vBuiltins = fGlobalBuiltins();
  if (!vIt->mIsUndefined()) {
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIt);
    }
    JJ_DEBUG("Identifier '%s' found, type = %s", vVal.c_str(), vIt->mType().c_str());
    return vIt;
  } else if (vBuiltins.find(vVal) != vBuiltins.end()) {
    JJ_DEBUG("Identifier '%s' found in builtins", vVal.c_str());
    return fMkVal<value_func>(vVal, vBuiltins.at(vVal));
  } else {
    JJ_DEBUG("Identifier '%s' not found, returning undefined", vVal.c_str());
    return fMkVal<value_undefined>(vVal);
  }
}

value object_literal::mExecuteImpl(context &pCtx) {
  auto vObj = fMkVal<value_object>();
  for (const auto &vPair : lVal) {
    value vKey = vPair.first->mExecute(pCtx);
    value vVal = vPair.second->mExecute(pCtx);
    JJ_DEBUG("Object literal: setting key '%s' with value type %s", vKey->mAsString().mStr().c_str(),
             vVal->mType().c_str());
    vObj->mInsert(vKey, vVal);
  }
  return vObj;
}

value binary_expression::mExecuteImpl(context &pCtx) {
  value vLeftVal = vLeft->mExecute(pCtx);

  // Logical operators
  if (vOp.vValue == "and") {
    JJ_DEBUG("Executing logical test: %s AND %s", vLeft->mType().c_str(), vRight->mType().c_str());
    return vLeftVal->mAsBool() ? vRight->mExecute(pCtx) : std::move(vLeftVal);
  } else if (vOp.vValue == "or") {
    JJ_DEBUG("Executing logical test: %s OR %s", vLeft->mType().c_str(), vRight->mType().c_str());
    return vLeftVal->mAsBool() ? std::move(vLeftVal) : vRight->mExecute(pCtx);
  }

  // Equality operators
  value vRightVal = vRight->mExecute(pCtx);
  JJ_DEBUG("Executing binary expression %s '%s' %s", vLeftVal->mType().c_str(), vOp.vValue.c_str(),
           vRightVal->mType().c_str());
  if (vOp.vValue == "==") {
    return fMkVal<value_bool>(*vLeftVal == *vRightVal);
  } else if (vOp.vValue == "!=") {
    return fMkVal<value_bool>(!(*vLeftVal == *vRightVal));
  }

  auto vWorkaroundConcatNullWithStr = [&](value &pRes) -> bool {
    bool vIsLeftNull = vLeftVal->mIsNone() || vLeftVal->mIsUndefined();
    bool vIsRightNull = vRightVal->mIsNone() || vRightVal->mIsUndefined();
    bool vIsLeftStr = fIsVal<value_string>(vLeftVal);
    bool vIsRightStr = fIsVal<value_string>(vRightVal);
    if ((vIsLeftNull && vIsRightStr) || (vIsRightNull && vIsLeftStr)) {
      JJ_DEBUG("%s", "Workaround: treating null/undefined as empty string for string concatenation");
      string vLeftStr = vIsLeftNull ? string() : vLeftVal->mAsString();
      string vRightStr = vIsRightNull ? string() : vRightVal->mAsString();
      auto vOutput = vLeftStr.mAppend(vRightStr);
      pRes = fMkVal<value_string>(std::move(vOutput));
      return true;
    }
    return false;
  };

  auto vTestIsIn = [&]() -> bool {
    func_args vArgs(pCtx);
    vArgs.mPushBack(vLeftVal);
    vArgs.mPushBack(vRightVal);
    return fGlobalBuiltins().at("test_is_in")(vArgs)->mAsBool();
  };

  // Handle undefined and null values
  if (fIsVal<value_undefined>(vLeftVal) || fIsVal<value_undefined>(vRightVal)) {
    if (fIsVal<value_undefined>(vRightVal) && (vOp.vValue == "in" || vOp.vValue == "not in")) {
      // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
      return fMkVal<value_bool>(vOp.vValue == "not in");
    }
    if (vOp.vValue == "+" || vOp.vValue == "~") {
      value vRes = fMkVal<value_undefined>();
      if (vWorkaroundConcatNullWithStr(vRes)) {
        return vRes;
      }
    }
    throw std::runtime_error("Cannot perform operation " + vOp.vValue + " on undefined values");
  } else if (fIsVal<value_none>(vLeftVal) || fIsVal<value_none>(vRightVal)) {
    if (vOp.vValue == "+" || vOp.vValue == "~") {
      value vRes = fMkVal<value_undefined>();
      if (vWorkaroundConcatNullWithStr(vRes)) {
        return vRes;
      }
    }
    throw std::runtime_error("Cannot perform operation on null values");
  }

  // Float operations
  if ((fIsVal<value_int>(vLeftVal) || fIsVal<value_float>(vLeftVal)) &&
      (fIsVal<value_int>(vRightVal) || fIsVal<value_float>(vRightVal))) {
    double vA = vLeftVal->mAsFloat();
    double vB = vRightVal->mAsFloat();
    if (vOp.vValue == "+" || vOp.vValue == "-" || vOp.vValue == "*") {
      double vRes = (vOp.vValue == "+") ? vA + vB : (vOp.vValue == "-") ? vA - vB : vA * vB;
      JJ_DEBUG("Arithmetic operation: %f %s %f = %f", vA, vOp.vValue.c_str(), vB, vRes);
      bool vIsFloat = fIsVal<value_float>(vLeftVal) || fIsVal<value_float>(vRightVal);
      if (vIsFloat) {
        return fMkVal<value_float>(vRes);
      } else {
        return fMkVal<value_int>(static_cast<int64_t>(vRes));
      }
    } else if (vOp.vValue == "/") {
      JJ_DEBUG("Division operation: %f / %f", vA, vB);
      return fMkVal<value_float>(vA / vB);
    } else if (vOp.vValue == "%") {
      double vRem = std::fmod(vA, vB);
      JJ_DEBUG("Modulo operation: %f %% %f = %f", vA, vB, vRem);
      bool vIsFloat = fIsVal<value_float>(vLeftVal) || fIsVal<value_float>(vRightVal);
      if (vIsFloat) {
        return fMkVal<value_float>(vRem);
      } else {
        return fMkVal<value_int>(static_cast<int64_t>(vRem));
      }
    } else if (vOp.vValue == "<") {
      JJ_DEBUG("Comparison operation: %f < %f is %d", vA, vB, vA < vB);
      return fMkVal<value_bool>(vA < vB);
    } else if (vOp.vValue == ">") {
      JJ_DEBUG("Comparison operation: %f > %f is %d", vA, vB, vA > vB);
      return fMkVal<value_bool>(vA > vB);
    } else if (vOp.vValue == ">=") {
      JJ_DEBUG("Comparison operation: %f >= %f is %d", vA, vB, vA >= vB);
      return fMkVal<value_bool>(vA >= vB);
    } else if (vOp.vValue == "<=") {
      JJ_DEBUG("Comparison operation: %f <= %f is %d", vA, vB, vA <= vB);
      return fMkVal<value_bool>(vA <= vB);
    }
  }

  // Array operations
  if (fIsVal<value_array>(vLeftVal) && fIsVal<value_array>(vRightVal)) {
    if (vOp.vValue == "+") {
      auto &vLeftArr = vLeftVal->mAsArray();
      auto &vRightArr = vRightVal->mAsArray();
      auto vResult = fMkVal<value_array>();
      for (const auto &vItem : vLeftArr) {
        vResult->mPushBack(vItem);
      }
      for (const auto &vItem : vRightArr) {
        vResult->mPushBack(vItem);
      }
      return vResult;
    }
  } else if (fIsVal<value_array>(vRightVal)) {
    // case: 1 in [0, 1, 2]
    bool vMember = vTestIsIn();
    if (vOp.vValue == "in") {
      return fMkVal<value_bool>(vMember);
    } else if (vOp.vValue == "not in") {
      return fMkVal<value_bool>(!vMember);
    }
  }

  // String concatenation with ~ and +
  if ((fIsVal<value_string>(vLeftVal) || fIsVal<value_string>(vRightVal)) && (vOp.vValue == "~" || vOp.vValue == "+")) {
    JJ_DEBUG("String concatenation with %s operator", vOp.vValue.c_str());
    auto vOutput = vLeftVal->mAsString().mAppend(vRightVal->mAsString());
    // concatenating a string with itself in a loop doubles it every time
    if (vOutput.lParts.size() > cMaxStringParts || vOutput.mLength() > cMaxStringBytes) {
      throw std::runtime_error("string concatenation exceeds the maximum of " + std::to_string(cMaxStringBytes) +
                               " bytes or " + std::to_string(cMaxStringParts) + " parts");
    }
    auto vRes = fMkVal<value_string>();
    vRes->vValStr = std::move(vOutput);
    return vRes;
  }

  // Python-style string repetition
  // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
  if (vOp.vValue == "*" && ((fIsVal<value_string>(vLeftVal) && fIsVal<value_int>(vRightVal)) ||
                            (fIsVal<value_int>(vLeftVal) && fIsVal<value_string>(vRightVal)))) {
    const auto &vStr = fIsVal<value_string>(vLeftVal) ? vLeftVal->mAsString() : vRightVal->mAsString();
    const int64_t cRepeat = fIsVal<value_int>(vRightVal) ? vRightVal->mAsInt() : vLeftVal->mAsInt();
    auto vRes = fMkVal<value_string>();
    if (cRepeat <= 0) {
      return vRes;
    }
    // The result is bounded in bytes and in parts before it is built, and it is built in one pass with the same
    // parts as appending the string once per repetition (appending to a copy of the result was quadratic)
    const uint64_t cNParts = vStr.lParts.size();
    if (cNParts == 0) {
      return vRes;
    }
    const uint64_t cRepeatCount = static_cast<uint64_t>(cRepeat);
    const uint64_t cLength = vStr.mLength();
    if (cRepeatCount > cMaxStringParts / cNParts || (cLength > 0 && cRepeatCount > cMaxStringBytes / cLength)) {
      throw std::runtime_error("string repetition exceeds the maximum of " + std::to_string(cMaxStringBytes) +
                               " bytes or " + std::to_string(cMaxStringParts) + " parts");
    }
    vRes->vValStr.lParts.reserve(cNParts * cRepeatCount);
    for (int64_t vI = 0; vI < cRepeat; ++vI) {
      vRes->vValStr.lParts.insert(vRes->vValStr.lParts.end(), vStr.lParts.begin(), vStr.lParts.end());
    }
    return vRes;
  }

  // String membership
  if (fIsVal<value_string>(vLeftVal) && fIsVal<value_string>(vRightVal)) {
    // case: "a" in "abc"
    bool vMember = vTestIsIn();
    if (vOp.vValue == "in") {
      return fMkVal<value_bool>(vMember);
    } else if (vOp.vValue == "not in") {
      return fMkVal<value_bool>(!vMember);
    }
  }

  // Value key in object
  if (fIsVal<value_object>(vRightVal)) {
    // case: key in {key: value}
    bool vMember = vTestIsIn();
    if (vOp.vValue == "in") {
      return fMkVal<value_bool>(vMember);
    } else if (vOp.vValue == "not in") {
      return fMkVal<value_bool>(!vMember);
    }
  }

  throw std::runtime_error("Unknown operator \"" + vOp.vValue + "\" between " + vLeftVal->mType() + " and " +
                           vRightVal->mType());
}

static value fTryBuiltinFunc(context &pCtx, const std::string &pName, value &pInput, bool pUndefOnMissing = false) {
  JJ_DEBUG("Trying built-in function '%s' for type %s", pName.c_str(), pInput->mType().c_str());
  if (pCtx.vIsGetStats) {
    value_t::stats_t::mMarkUsed(pInput);
    pInput->vStats.sOps.insert(pName);
  }
  auto vBuiltins = pInput->mGetBuiltins();
  auto vIt = vBuiltins.find(pName);
  if (vIt != vBuiltins.end()) {
    JJ_DEBUG("Binding built-in '%s'", pName.c_str());
    return fMkVal<value_func>(pName, vIt->second, pInput);
  }
  if (pUndefOnMissing) {
    return fMkVal<value_undefined>(pName);
  }
  throw std::runtime_error("Unknown (built-in) filter '" + pName + "' for type " + pInput->mType());
}

value filter_expression::mExecuteImpl(context &pCtx) {
  value vInput = vOperand ? vOperand->mExecute(pCtx) : vVal;

  JJ_DEBUG("Applying filter to %s", vInput->mType().c_str());

  auto vSetFilterAlias = [](auto &pFilterId) {
    if (pFilterId == "count") {
      pFilterId = "length";
    } else if (pFilterId == "d") {
      pFilterId = "default";
    } else if (pFilterId == "e") {
      pFilterId = "escape";
    } else if (pFilterId == "trim") {
      pFilterId = "strip";
    }
  };

  if (fIsStmt<identifier>(vFilter)) {
    auto vFilterId = fCastStmt<identifier>(vFilter)->vVal;

    vSetFilterAlias(vFilterId);
    JJ_DEBUG("Applying filter '%s' to %s", vFilterId.c_str(), vInput->mType().c_str());
    // TODO: Refactor filters so this coercion can be done automatically
    if (!vInput->mIsUndefined() && !fIsVal<value_string>(vInput) &&
        (vFilterId == "capitalize" || vFilterId == "lower" || vFilterId == "replace" || vFilterId == "strip" ||
         vFilterId == "title" || vFilterId == "upper" || vFilterId == "wordcount")) {
      JJ_DEBUG("Coercing %s to String for '%s' filter", vInput->mType().c_str(), vFilterId.c_str());
      vInput = fMkVal<value_string>(vInput->mAsString());
    }
    return fTryBuiltinFunc(pCtx, vFilterId, vInput)->mInvoke(func_args(pCtx));

  } else if (fIsStmt<call_expression>(vFilter)) {
    auto vCall = fCastStmt<call_expression>(vFilter);
    if (!fIsStmt<identifier>(vCall->vCallee)) {
      throw std::runtime_error("Filter callee must be an identifier");
    }
    auto vFilterId = fCastStmt<identifier>(vCall->vCallee)->vVal;

    vSetFilterAlias(vFilterId);
    JJ_DEBUG("Applying filter '%s' with arguments to %s", vFilterId.c_str(), vInput->mType().c_str());
    func_args vArgs(pCtx);
    for (const auto &vArgExpr : vCall->vArgs) {
      vArgs.mPushBack(vArgExpr->mExecute(pCtx));
    }

    return fTryBuiltinFunc(pCtx, vFilterId, vInput)->mInvoke(vArgs);

  } else {
    throw std::runtime_error("Invalid filter expression");
  }
}

value filter_statement::mExecuteImpl(context &pCtx) {
  // eval body as string, then apply filter
  auto vBodyVal = fExecStatements(vBody, pCtx);
  value_string vParts = fMkVal<value_string>();
  fGatherStringPartsRecursive(vBodyVal, vParts);

  JJ_DEBUG("FilterStatement: applying filter to body string of length %zu", vParts->vValStr.mLength());
  filter_expression vFilterExpr(std::move(vParts), std::move(vFilter));
  value vOut = vFilterExpr.mExecute(pCtx);

  // this node can be reused later, make sure filter is preserved
  this->vFilter = std::move(vFilterExpr.vFilter);
  return vOut;
}

value test_expression::mExecuteImpl(context &pCtx) {
  // NOTE: "value is something" translates to function call "test_is_something(value)"
  const auto &vBuiltins = fGlobalBuiltins();

  std::string vTestId;
  value vInput = vOperand->mExecute(pCtx);

  func_args vArgs(pCtx);
  vArgs.mPushBack(vInput);

  if (fIsStmt<identifier>(vTest)) {
    vTestId = fCastStmt<identifier>(vTest)->vVal;
  } else if (fIsStmt<call_expression>(vTest)) {
    auto vCall = fCastStmt<call_expression>(vTest);
    if (!fIsStmt<identifier>(vCall->vCallee)) {
      throw std::runtime_error("Test callee must be an identifier");
    }
    vTestId = fCastStmt<identifier>(vCall->vCallee)->vVal;

    JJ_DEBUG("Applying test '%s' with arguments to %s", vTestId.c_str(), vInput->mType().c_str());
    for (const auto &vArgExpr : vCall->vArgs) {
      vArgs.mPushBack(vArgExpr->mExecute(pCtx));
    }

  } else {
    throw std::runtime_error("Invalid test expression");
  }

  auto vIt = vBuiltins.find("test_is_" + vTestId);
  JJ_DEBUG("Test expression %s '%s' %s (using function 'test_is_%s')", vOperand->mType().c_str(), vTestId.c_str(),
           vNegate ? "(negate)" : "", vTestId.c_str());
  if (vIt == vBuiltins.end()) {
    throw std::runtime_error("Unknown test '" + vTestId + "'");
  }

  auto vRes = vIt->second(vArgs);

  if (vNegate) {
    return fMkVal<value_bool>(!vRes->mAsBool());
  } else {
    return vRes;
  }
}

value unary_expression::mExecuteImpl(context &pCtx) {
  value vOperandVal = vArgument->mExecute(pCtx);
  JJ_DEBUG("Executing unary expression with operator '%s'", vOp.vValue.c_str());

  if (vOp.vValue == "not") {
    return fMkVal<value_bool>(!vOperandVal->mAsBool());
  } else if (vOp.vValue == "-") {
    if (fIsVal<value_int>(vOperandVal)) {
      return fMkVal<value_int>(-vOperandVal->mAsInt());
    } else if (fIsVal<value_float>(vOperandVal)) {
      return fMkVal<value_float>(-vOperandVal->mAsFloat());
    } else {
      throw std::runtime_error("Unary - operator requires numeric operand");
    }
  }

  throw std::runtime_error("Unknown unary operator '" + vOp.vValue + "'");
}

value if_statement::mExecuteImpl(context &pCtx) {
  value vTestVal = vTest->mExecute(pCtx);

  auto vOut = fMkVal<value_array>();
  if (vTestVal->mAsBool()) {
    for (auto &vStmt : vBody) {
      JJ_DEBUG("IF --> Executing THEN body, current block: %s", vStmt->mType().c_str());
      vOut->mPushBack(vStmt->mExecute(pCtx));
    }
  } else {
    for (auto &vStmt : vAlternate) {
      JJ_DEBUG("IF --> Executing ELSE body, current block: %s", vStmt->mType().c_str());
      vOut->mPushBack(vStmt->mExecute(pCtx));
    }
  }
  // convert to string parts
  value_string vStr = fMkVal<value_string>();
  fGatherStringPartsRecursive(vOut, vStr);
  return vStr;
}

value for_statement::mExecuteImpl(context &pCtx) {
  context vScope(pCtx); // new scope for loop variables

  jinja::select_expression *vSelectExpr = fCastStmt<select_expression>(vIterable);
  statement_ptr vTestExprNullptr;

  statement_ptr &vIterExpr = [&]() -> statement_ptr & {
    auto vTmp = fCastStmt<select_expression>(vIterable);
    return vTmp ? vTmp->vLhs : vIterable;
  }();
  statement_ptr &vTestExpr = [&]() -> statement_ptr & {
    auto vTmp = fCastStmt<select_expression>(vIterable);
    return vTmp ? vTmp->vTest : vTestExprNullptr;
  }();

  JJ_DEBUG("Executing for statement, iterable type: %s", vIterExpr->mType().c_str());

  value vIterableVal = vIterExpr->mExecute(vScope);

  // mark the variable being iterated as used for stats
  if (pCtx.vIsGetStats) {
    value_t::stats_t::mMarkUsed(vIterableVal);
    vIterableVal->vStats.sOps.insert("array_access");
  }

  if (vIterableVal->mIsUndefined()) {
    JJ_DEBUG("%s", "For loop iterable is undefined, skipping loop");
    vIterableVal = fMkVal<value_array>();
  }

  if (!fIsVal<value_array>(vIterableVal) && !fIsVal<value_object>(vIterableVal)) {
    throw std::runtime_error("Expected iterable or object type in for loop: got " + vIterableVal->mType());
  }

  std::vector<value> lItems;
  if (fIsVal<value_object>(vIterableVal)) {
    JJ_DEBUG("%s", "For loop over object keys");
    auto &vObj = vIterableVal->mAsOrderedObject();
    for (auto &vP : vObj) {
      auto vTuple = fMkVal<value_tuple>(vP);
      lItems.push_back(std::move(vTuple));
    }
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIterableVal);
      vIterableVal->vStats.sOps.insert("object_access");
    }
  } else {
    JJ_DEBUG("%s", "For loop over array items");
    auto &vArr = vIterableVal->mAsArray();
    for (const auto &vItem : vArr) {
      lItems.push_back(vItem);
    }
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIterableVal);
      vIterableVal->vStats.sOps.insert("array_access");
    }
  }

  std::vector<std::function<void(context &)>> lScopeUpdateFns;

  std::vector<value> lFilteredItems;
  for (size_t vI = 0; vI < lItems.size(); ++vI) {
    context vLoopScope(vScope);

    value vCurrent = lItems[vI];

    std::function<void(context &)> vScopeUpdateFn = [](context &) { /* no-op */ };
    if (fIsStmt<identifier>(vLoopvar)) {
      auto vId = fCastStmt<identifier>(vLoopvar)->vVal;

      if (fIsVal<value_object>(vIterableVal)) {
        // case example: {% for key in dict %}
        vCurrent = lItems[vI]->mAsArray()[0];
        vScopeUpdateFn = [vId, &lItems, vI](context &pCtx) { pCtx.mSetVal(vId, lItems[vI]->mAsArray()[0]); };
      } else {
        // case example: {% for item in list %}
        vScopeUpdateFn = [vId, &lItems, vI](context &pCtx) { pCtx.mSetVal(vId, lItems[vI]); };
      }

    } else if (fIsStmt<tuple_literal>(vLoopvar)) {
      // case example: {% for key, value in dict %}
      auto vTuple = fCastStmt<tuple_literal>(vLoopvar);
      if (!fIsVal<value_array>(vCurrent)) {
        throw std::runtime_error("Cannot unpack non-iterable type: " + vCurrent->mType());
      }
      auto &vCArr = vCurrent->mAsArray();
      if (vTuple->vVal.size() != vCArr.size()) {
        throw std::runtime_error(std::string("Too ") + (vTuple->vVal.size() > vCArr.size() ? "few" : "many") +
                                 " items to unpack");
      }
      vScopeUpdateFn = [vTuple, &lItems, vI](context &pCtx) {
        auto &vCArr = lItems[vI]->mAsArray();
        for (size_t vJ = 0; vJ < vTuple->vVal.size(); ++vJ) {
          if (!fIsStmt<identifier>(vTuple->vVal[vJ])) {
            throw std::runtime_error("Cannot unpack non-identifier type: " + vTuple->vVal[vJ]->mType());
          }
          auto vId = fCastStmt<identifier>(vTuple->vVal[vJ])->vVal;
          pCtx.mSetVal(vId, vCArr[vJ]);
        }
      };

    } else {
      throw std::runtime_error("Invalid loop variable(s): " + vLoopvar->mType());
    }

    if (vSelectExpr && vTestExpr) {
      vScopeUpdateFn(vLoopScope);
      value vTestVal = vTestExpr->mExecute(vLoopScope);
      if (!vTestVal->mAsBool()) {
        continue;
      }
    }
    JJ_DEBUG("For loop: adding item type %s at index %zu", vCurrent->mType().c_str(), vI);
    lFilteredItems.push_back(vCurrent);
    lScopeUpdateFns.push_back(vScopeUpdateFn);
  }
  JJ_DEBUG("For loop: %zu items after filtering", lFilteredItems.size());

  auto vResult = fMkVal<value_array>();

  bool vNoIteration = true;
  for (size_t vI = 0; vI < lFilteredItems.size(); vI++) {
    JJ_DEBUG("For loop iteration %zu/%zu", vI + 1, lFilteredItems.size());
    value_object vLoopObj = fMkVal<value_object>();
    vLoopObj->vHasBuiltins = false; // loop object has no builtins
    vLoopObj->insert("index", fMkVal<value_int>(vI + 1));
    vLoopObj->insert("index0", fMkVal<value_int>(vI));
    vLoopObj->insert("revindex", fMkVal<value_int>(lFilteredItems.size() - vI));
    vLoopObj->insert("revindex0", fMkVal<value_int>(lFilteredItems.size() - vI - 1));
    vLoopObj->insert("first", fMkVal<value_bool>(vI == 0));
    vLoopObj->insert("last", fMkVal<value_bool>(vI == lFilteredItems.size() - 1));
    vLoopObj->insert("length", fMkVal<value_int>(lFilteredItems.size()));
    vLoopObj->insert("previtem", vI > 0 ? lFilteredItems[vI - 1] : fMkVal<value_undefined>("previtem"));
    vLoopObj->insert("nextitem",
                     vI < lFilteredItems.size() - 1 ? lFilteredItems[vI + 1] : fMkVal<value_undefined>("nextitem"));
    vScope.mSetVal("loop", vLoopObj);
    lScopeUpdateFns[vI](vScope);
    try {
      for (auto &vStmt : vBody) {
        value vVal = vStmt->mExecute(vScope);
        vResult->mPushBack(vVal);
      }
    } catch (const continue_statement::signal &) {
      continue;
    } catch (const break_statement::signal &) {
      break;
    }
    vNoIteration = false;
  }

  JJ_DEBUG("For loop complete, total iterations: %zu", lFilteredItems.size());
  if (vNoIteration) {
    for (auto &vStmt : vDefaultBlock) {
      value vVal = vStmt->mExecute(pCtx);
      vResult->mPushBack(vVal);
    }
  }

  // convert to string parts
  value_string vStr = fMkVal<value_string>();
  fGatherStringPartsRecursive(vResult, vStr);
  return vStr;
}

value set_statement::mExecuteImpl(context &pCtx) {
  auto vRhs = vVal ? vVal->mExecute(pCtx) : fExecStatements(vBody, pCtx);

  if (fIsStmt<identifier>(vAssignee)) {
    // case: {% set my_var = value %}
    auto vVarName = fCastStmt<identifier>(vAssignee)->vVal;
    JJ_DEBUG("Setting global variable '%s' with value type %s", vVarName.c_str(), vRhs->mType().c_str());
    pCtx.mSetVal(vVarName, vRhs);

  } else if (fIsStmt<tuple_literal>(vAssignee)) {
    // case: {% set a, b = value %}
    auto vTuple = fCastStmt<tuple_literal>(vAssignee);
    if (!fIsVal<value_array>(vRhs)) {
      throw std::runtime_error("Cannot unpack non-iterable type in set: " + vRhs->mType());
    }
    auto &vArr = vRhs->mAsArray();
    if (vArr.size() != vTuple->vVal.size()) {
      throw std::runtime_error(std::string("Too ") + (vTuple->vVal.size() > vArr.size() ? "few" : "many") +
                               " items to unpack in set");
    }
    for (size_t vI = 0; vI < vTuple->vVal.size(); ++vI) {
      auto &vElem = vTuple->vVal[vI];
      if (!fIsStmt<identifier>(vElem)) {
        throw std::runtime_error("Cannot unpack to non-identifier in set: " + vElem->mType());
      }
      auto vVarName = fCastStmt<identifier>(vElem)->vVal;
      pCtx.mSetVal(vVarName, vArr[vI]);
    }

  } else if (fIsStmt<member_expression>(vAssignee)) {
    // case: {% set ns.my_var = value %}
    auto vMember = fCastStmt<member_expression>(vAssignee);
    if (vMember->vComputed) {
      throw std::runtime_error("Cannot assign to computed member");
    }
    if (!fIsStmt<identifier>(vMember->vProperty)) {
      throw std::runtime_error("Cannot assign to member with non-identifier property");
    }
    auto vPropName = fCastStmt<identifier>(vMember->vProperty)->vVal;

    value vObject = vMember->vObject->mExecute(pCtx);
    if (!fIsVal<value_object>(vObject)) {
      throw std::runtime_error("Cannot assign to member of non-object");
    }
    auto vObjPtr = fCastVal<value_object>(vObject);
    JJ_DEBUG("Setting object property '%s' with value type %s", vPropName.c_str(), vRhs->mType().c_str());
    vObjPtr->insert(vPropName, vRhs);

  } else {
    throw std::runtime_error("Invalid LHS inside assignment expression: " + vAssignee->mType());
  }
  return fMkVal<value_undefined>();
}

static inline void fBindParameters(const std::string &pName, const statements &pThisArgs, const func_args &pArgs,
                                   context &pCtx) {
  const size_t cExpectedCount = pThisArgs.size();
  const size_t cInputCount = pArgs.mCount();

  JJ_DEBUG("Invoking '%s' with %zu input arguments (expected %zu)", pName.c_str(), cInputCount, cExpectedCount);
  for (size_t vI = 0; vI < cExpectedCount; ++vI) {
    if (vI < cInputCount) {
      if (fIsStmt<identifier>(pThisArgs[vI])) {
        // normal parameter
        std::string vParamName = fCastStmt<identifier>(pThisArgs[vI])->vVal;
        value vParamValue = pArgs.mGetKwargOrPos(vParamName, vI);
        JJ_DEBUG("  Binding parameter '%s' to argument of type %s", vParamName.c_str(), vParamValue->mType().c_str());
        pCtx.mSetVal(vParamName, vParamValue);
      } else if (fIsStmt<keyword_argument_expression>(pThisArgs[vI])) {
        // default argument used as normal parameter
        auto vKwarg = fCastStmt<keyword_argument_expression>(pThisArgs[vI]);
        if (!fIsStmt<identifier>(vKwarg->vKey)) {
          throw std::runtime_error("Keyword argument key must be an identifier in '" + pName + "'");
        }
        std::string vParamName = fCastStmt<identifier>(vKwarg->vKey)->vVal;
        value vParamValue = pArgs.mGetKwargOrPos(vParamName, vI);
        JJ_DEBUG("  Binding parameter '%s' to argument of type %s", vParamName.c_str(), vParamValue->mType().c_str());
        pCtx.mSetVal(vParamName, vParamValue);
      } else {
        throw std::runtime_error("Invalid parameter type in '" + pName + "'");
      }
    } else {
      auto &vDefaultArg = pThisArgs[vI];
      if (fIsStmt<keyword_argument_expression>(vDefaultArg)) {
        auto vKwarg = fCastStmt<keyword_argument_expression>(vDefaultArg);
        if (!fIsStmt<identifier>(vKwarg->vKey)) {
          throw std::runtime_error("Keyword argument key must be an identifier in '" + pName + "'");
        }
        std::string vParamName = fCastStmt<identifier>(vKwarg->vKey)->vVal;
        JJ_DEBUG("  Binding parameter '%s' to default argument of type %s", vParamName.c_str(),
                 vKwarg->vVal->mType().c_str());
        pCtx.mSetVal(vParamName, vKwarg->vVal->mExecute(pArgs.vCtx));
      } else {
        throw std::runtime_error("Not enough arguments provided to '" + pName + "'");
      }
      // std::string vParamName = fCastStmt<identifier>(pThisArgs[vI])->vVal;
      // JJ_DEBUG("  Binding parameter '%s' to default", vParamName.c_str());
      // pCtx.var[vParamName] = pThisArgs[vI]->mExecute(pCtx);
    }
  }
}

value macro_statement::mExecuteImpl(context &pCtx) {
  if (!fIsStmt<identifier>(this->vName)) {
    throw std::runtime_error("Macro name must be an identifier");
  }
  std::string vName = fCastStmt<identifier>(this->vName)->vVal;

  const func_handler cFunc = [this, vName](const func_args &pArgs) -> value {
    context vMacroCtx(pArgs.vCtx); // new scope for macro execution

    fBindParameters(vName, this->vArgs, pArgs, vMacroCtx);

    // execute macro body
    JJ_DEBUG("Executing macro '%s' body with %zu statements", vName.c_str(), this->vBody.size());
    auto vRes = fExecStatements(this->vBody, vMacroCtx);
    JJ_DEBUG("Macro '%s' execution complete, result: %s", vName.c_str(), vRes->vValStr.mStr().c_str());
    return vRes;
  };

  JJ_DEBUG("Defining macro '%s' with %zu parameters", vName.c_str(), vArgs.size());
  pCtx.mSetVal(vName, fMkVal<value_func>(vName, cFunc));
  return fMkVal<value_undefined>();
}

value call_statement::mExecuteImpl(context &pCtx) {
  auto vCallExpr = fCastStmt<call_expression>(this->vCall);
  if (!vCallExpr) {
    throw std::runtime_error("Call statement requires a valid call expression");
  }

  value vCalleeVal = vCallExpr->vCallee->mExecute(pCtx);
  if (!fIsVal<value_func>(vCalleeVal)) {
    throw std::runtime_error("Callee is not a function: got " + vCalleeVal->mType());
  }
  auto *vCalleeFunc = fCastVal<value_func>(vCalleeVal);

  context vCallerCtx(pCtx); // new scope for caller execution

  const func_handler cFunc = [this, vCallerCtx = std::move(vCallerCtx)](const func_args &pArgs) -> value {
    context vBlockCtx(vCallerCtx); // new scope for block execution

    fBindParameters("caller", this->vCallerArgs, pArgs, vBlockCtx);

    JJ_DEBUG("Executing call body with %zu statements", this->vBody.size());
    auto vRes = fExecStatements(this->vBody, vBlockCtx);
    JJ_DEBUG("Call body execution complete, result: %s", vRes->vValStr.mStr().c_str());
    return vRes;
  };

  context vCallCtx(pCtx);
  vCallCtx.mSetVal("caller", fMkVal<value_func>("caller", cFunc));

  func_args vArgs(vCallCtx);

  for (const auto &vArgExpr : vCallExpr->vArgs) {
    auto vArgVal = vArgExpr->mExecute(pCtx);
    JJ_DEBUG("  Argument type: %s", vArgVal->mType().c_str());
    vArgs.mPushBack(vArgVal);
  }

  JJ_DEBUG("Calling macro '%s' with %zu arguments", vCalleeFunc->vName.c_str(), vArgs.mCount());
  return vCalleeFunc->mInvoke(vArgs);
}

value member_expression::mExecuteImpl(context &pCtx) {
  value vObject = this->vObject->mExecute(pCtx);

  value vProperty;
  if (this->vComputed) {
    // syntax: obj[expr]
    JJ_DEBUG("Member expression, computing property type %s", this->vProperty->mType().c_str());

    int64_t vArrSize = 0;
    if (fIsVal<value_array>(vObject)) {
      vArrSize = vObject->mAsArray().size();
    } else if (fIsVal<value_string>(vObject)) {
      vArrSize = vObject->mAsString().mLength();
    }

    if (fIsStmt<slice_expression>(this->vProperty)) {
      auto vS = fCastStmt<slice_expression>(this->vProperty);
      value vStepVal = vS->vStepExpr ? vS->vStepExpr->mExecute(pCtx) : fMkVal<value_int>(1);
      value vStartVal = vS->vStartExpr
                            ? vS->vStartExpr->mExecute(pCtx)
                            : (vStepVal->mAsInt() < 0 ? fMkVal<value_int>(vArrSize - 1) : fMkVal<value_int>(0));
      value vStopVal = vS->vStopExpr ? vS->vStopExpr->mExecute(pCtx)
                                     : (vStepVal->mAsInt() < 0 ? fMkVal<value_int>(-1) : fMkVal<value_int>(vArrSize));

      // translate to function call: obj.slice(start, stop, step)
      JJ_DEBUG("Member expression is a slice: start %s, stop %s, step %s", vStartVal->mAsRepr().c_str(),
               vStopVal->mAsRepr().c_str(), vStepVal->mAsRepr().c_str());
      auto vSliceFunc = fTryBuiltinFunc(pCtx, "slice", vObject);
      func_args vArgs(pCtx);
      vArgs.mPushBack(vStartVal);
      vArgs.mPushBack(vStopVal);
      vArgs.mPushBack(vStepVal);
      return vSliceFunc->mInvoke(vArgs);
    } else {
      vProperty = this->vProperty->mExecute(pCtx);
    }
  } else {
    // syntax: obj.prop
    if (!fIsStmt<identifier>(this->vProperty)) {
      throw std::runtime_error("Static member property must be an identifier");
    }
    vProperty = fMkVal<value_string>(fCastStmt<identifier>(this->vProperty)->vVal);
    std::string vProp = vProperty->mAsString().mStr();
    JJ_DEBUG("Member expression, object type %s, static property '%s'", vObject->mType().c_str(), vProp.c_str());

    // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
    // then obj.prop returns the built-in function, not the property value.
    // while obj['prop'] returns the property value.
    // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is 123

    value vVal = fTryBuiltinFunc(pCtx, vProp, vObject, true);
    if (!fIsVal<value_undefined>(vVal)) {
      return vVal;
    }
    // else, fallthrough to normal property access below
  }

  JJ_DEBUG("Member expression on object type %s, property type %s", vObject->mType().c_str(),
           vProperty->mType().c_str());
  value vVal = fMkVal<value_undefined>("object_property");

  if (vProperty->mIsUndefined()) {
    JJ_DEBUG("%s", "Member expression property is undefined, returning undefined");
    return vVal;
  }

  fEnsureKeyTypeAllowed(vProperty);

  if (fIsVal<value_undefined>(vObject)) {
    JJ_DEBUG("%s", "Accessing property on undefined object, returning undefined");
    return vVal;

  } else if (fIsVal<value_object>(vObject)) {
    auto vKey = vProperty->mAsString().mStr();
    vVal = vObject->mAt(vProperty, vVal);
    if (fIsVal<value_undefined>(vVal)) {
      vVal = fTryBuiltinFunc(pCtx, vKey, vObject, true);
    }
    JJ_DEBUG("Accessed property '%s' value, got type: %s", vKey.c_str(), vVal->mType().c_str());

  } else if (fIsVal<value_array>(vObject) || fIsVal<value_string>(vObject)) {
    if (fIsVal<value_int>(vProperty)) {
      int64_t vIndex = vProperty->mAsInt();
      JJ_DEBUG("Accessing %s index %d", vObject->mType().c_str(), (int)vIndex);
      if (fIsVal<value_array>(vObject)) {
        auto &vArr = vObject->mAsArray();
        if (vIndex < 0) {
          vIndex += static_cast<int64_t>(vArr.size());
        }
        if (vIndex >= 0 && vIndex < static_cast<int64_t>(vArr.size())) {
          vVal = vArr[vIndex];
        }
      } else { // value_string
        auto vStr = vObject->mAsString().mStr();
        if (vIndex >= 0 && vIndex < static_cast<int64_t>(vStr.size())) {
          vVal = fMkVal<value_string>(std::string(1, vStr[vIndex]));
        }
      }

    } else if (fIsVal<value_string>(vProperty)) {
      auto vKey = vProperty->mAsString().mStr();
      JJ_DEBUG("Accessing %s built-in '%s'", fIsVal<value_array>(vObject) ? "array" : "string", vKey.c_str());
      vVal = fTryBuiltinFunc(pCtx, vKey, vObject, true);

    } else {
      throw std::runtime_error("Cannot access property with non-string/non-number: got " + vProperty->mType());
    }
  } else {
    if (!fIsVal<value_string>(vProperty)) {
      throw std::runtime_error("Cannot access property with non-string: got " + vProperty->mType());
    }
    auto vKey = vProperty->mAsString().mStr();
    vVal = fTryBuiltinFunc(pCtx, vKey, vObject, true);
  }

  if (pCtx.vIsGetStats && vVal && vObject && vProperty) {
    value_t::stats_t::mMarkUsed(vVal);
    value_t::stats_t::mMarkUsed(vObject);
    value_t::stats_t::mMarkUsed(vProperty);
    if (fIsVal<value_int>(vProperty)) {
      vObject->vStats.sOps.insert("array_access");
    } else if (fIsVal<value_string>(vProperty)) {
      vObject->vStats.sOps.insert("object_access");
    }
  }

  return vVal;
}

value call_expression::mExecuteImpl(context &pCtx) {
  // gather arguments
  func_args vArgs(pCtx);
  for (auto &vArgStmt : this->vArgs) {
    auto vArgVal = vArgStmt->mExecute(pCtx);
    JJ_DEBUG("  Argument type: %s", vArgVal->mType().c_str());
    vArgs.mPushBack(vArgVal);
  }
  // execute callee
  value vCalleeVal = vCallee->mExecute(pCtx);
  if (!fIsVal<value_func>(vCalleeVal)) {
    throw std::runtime_error("Callee is not a function: got " + vCalleeVal->mType());
  }
  auto *vCalleeFunc = fCastVal<value_func>(vCalleeVal);
  JJ_DEBUG("Calling function '%s' with %zu arguments", vCalleeFunc->vName.c_str(), vArgs.mCount());
  return vCalleeFunc->mInvoke(vArgs);
}

value keyword_argument_expression::mExecuteImpl(context &pCtx) {
  if (!fIsStmt<identifier>(vKey)) {
    throw std::runtime_error("Keyword argument key must be identifiers");
  }

  std::string vK = fCastStmt<identifier>(vKey)->vVal;
  JJ_DEBUG("Keyword argument expression key: %s, value: %s", vK.c_str(), vVal->mType().c_str());

  value vV = vVal->mExecute(pCtx);
  JJ_DEBUG("Keyword argument value executed, type: %s", vV->mType().c_str());

  return fMkVal<value_kwarg>(vK, vV);
}

std::string runtime::mDebugDumpProgram(const program &pProg, const std::string &pSrc) {
  std::ostringstream vOss;
  size_t vLvl = 0;
  context vCtx;
  vCtx.vSrc.reset(new std::string(pSrc));

  auto vIndent = [](size_t pLvl) -> std::string { return std::string(pLvl * 2, ' '); };

  vCtx.vVisitor = [&](bool pIsLeaf, statement *pNode, std::vector<visitor_pair> pChildren) {
    vOss << vIndent(vLvl) << pNode->mType() << ":\n";
    vLvl++;
    if (pIsLeaf) {
      const auto &vPos = pNode->vPos;
      vOss << vIndent(vLvl) << "(leaf) at " << fGetLineCol(pSrc, vPos) << " in source:\n";
      std::string vSnippet = fPeakSource(pSrc, vPos);
      fStringReplaceAll(vSnippet, "\n", "\n" + vIndent(vLvl));
      vOss << vIndent(vLvl) << vSnippet << "\n";
    } else {
      for (auto &[vLabel, vChildrenVec] : pChildren) {
        vOss << vIndent(vLvl) << vLabel << ":\n";
        vLvl++;
        if (vChildrenVec.empty()) {
          vOss << vIndent(vLvl) << "<empty>\n\n";
        } else {
          for (auto *vChild : vChildrenVec) {
            if (!vChild) {
              continue;
            }
            vChild->mVisit(vCtx);
          }
        }
        vLvl--;
      }
    }
    vLvl--;
  };

  for (const auto &vStmt : pProg.vBody) {
    vStmt->mVisit(vCtx);
  }

  return vOss.str();
}

} // namespace jinja
