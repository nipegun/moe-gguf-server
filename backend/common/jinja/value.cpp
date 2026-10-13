#include "runtime.h"
#include "unicode.h"
#include "value.h"

// for converting from JSON to jinja values
#include <nlohmann/json.hpp>

#include <sstream>
#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>

#define FILENAME "jinja-value"

namespace jinja {

// func_args method implementations

value func_args::mGetKwarg(const std::string &pKey, value pDefaultVal) const {
  for (const auto &vArg : lArgs) {
    if (fIsVal<value_kwarg>(vArg)) {
      auto *vKwarg = fCastVal<value_kwarg>(vArg);
      if (vKwarg->vKey == pKey) {
        return vKwarg->vVal;
      }
    }
  }
  return pDefaultVal;
}

value func_args::mGetKwargOrPos(const std::string &pKey, size_t pPos) const {
  value vVal = mGetKwarg(pKey, fMkVal<value_undefined>());

  if (vVal->mIsUndefined() && pPos < mCount() && !fIsVal<value_kwarg>(lArgs[pPos])) {
    return lArgs[pPos];
  }

  return vVal;
}

value func_args::mGetPos(size_t pPos) const {
  if (mCount() > pPos) {
    return lArgs[pPos];
  }
  throw raised_exception("Function '" + vFuncName + "' expected at least " + std::to_string(pPos + 1) +
                         " arguments, got " + std::to_string(mCount()));
}

value func_args::mGetPos(size_t pPos, value pDefaultVal) const {
  if (mCount() > pPos) {
    return lArgs[pPos];
  }
  return pDefaultVal;
}

void func_args::mPushBack(const value &pVal) { lArgs.push_back(pVal); }

void func_args::mPushFront(const value &pVal) { lArgs.insert(lArgs.begin(), pVal); }

const std::vector<value> &func_args::mGetArgs() const { return lArgs; }

/**
 * Function that mimics Python's array slicing.
 */
template <typename T> static T fSlice(const T &pArray, int64_t pStart, int64_t pStop, int64_t pStep = 1) {
  int64_t vLen = static_cast<int64_t>(pArray.size());
  int64_t vDirection = (pStep > 0) ? 1 : ((pStep < 0) ? -1 : 0);
  int64_t vStartVal = 0;
  int64_t vStopVal = 0;
  if (vDirection >= 0) {
    vStartVal = pStart;
    if (vStartVal < 0) {
      vStartVal = std::max(vLen + vStartVal, (int64_t)0);
    } else {
      vStartVal = std::min(vStartVal, vLen);
    }

    vStopVal = pStop;
    if (vStopVal < 0) {
      vStopVal = std::max(vLen + vStopVal, (int64_t)0);
    } else {
      vStopVal = std::min(vStopVal, vLen);
    }
  } else {
    vStartVal = pStart;
    if (vStartVal < 0) {
      vStartVal = std::max(vLen + vStartVal, (int64_t)0);
    } else {
      vStartVal = std::min(vStartVal, vLen - 1);
    }

    vStopVal = pStop;
    if (vStopVal < -1) {
      vStopVal = std::max(vLen + vStopVal, (int64_t)-1);
    } else {
      vStopVal = std::min(vStopVal, vLen - 1);
    }
  }
  T vResult;
  if (vDirection == 0) {
    return vResult;
  }
  for (int64_t vI = vStartVal; vDirection * vI < vDirection * vStopVal; vI += pStep) {
    if (vI >= 0 && vI < vLen) {
      vResult.push_back(pArray[static_cast<size_t>(vI)]);
    }
    // a step that would overflow the index lands past the stop anyway (start and stop are clamped to the length)
    if ((pStep > 0 && vI > INT64_MAX - pStep) || (pStep < 0 && vI <= INT64_MIN - pStep)) {
      break;
    }
  }
  return vResult;
}

// Throws when a string of pBase bytes that grows by pGrowth bytes pCount times would exceed cMaxStringBytes
static void fCheckGrownString(uint64_t pBase, uint64_t pGrowth, uint64_t pCount, const char *pWhat) {
  if (pBase > cMaxStringBytes || (pGrowth > 0 && pCount > (cMaxStringBytes - pBase) / pGrowth)) {
    throw raised_exception(std::string(pWhat) + " would produce a string longer than the maximum of " +
                           std::to_string(cMaxStringBytes) + " bytes");
  }
}

template <typename T> static value fEmptyValueFn(const func_args &) {
  if constexpr (std::is_same_v<T, value_int>) {
    return fMkVal<T>(0);
  } else if constexpr (std::is_same_v<T, value_float>) {
    return fMkVal<T>(0.0);
  } else if constexpr (std::is_same_v<T, value_bool>) {
    return fMkVal<T>(false);
  } else {
    return fMkVal<T>();
  }
}
template <typename T> static value fTestTypeFn(const func_args &pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = fIsVal<T>(pArgs.mGetPos(0));
  JJ_DEBUG("fTestTypeFn: type=%s result=%d", typeid(T).name(), vIsType ? 1 : 0);
  return fMkVal<value_bool>(vIsType);
}
template <typename T, typename U> static value fTestTypeFn(const func_args &pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = fIsVal<T>(pArgs.mGetPos(0)) || fIsVal<U>(pArgs.mGetPos(0));
  JJ_DEBUG("fTestTypeFn: type=%s or %s result=%d", typeid(T).name(), typeid(U).name(), vIsType ? 1 : 0);
  return fMkVal<value_bool>(vIsType);
}
template <typename T, typename U, typename V> static value fTestTypeFn(const func_args &pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = fIsVal<T>(pArgs.mGetPos(0)) || fIsVal<U>(pArgs.mGetPos(0)) || fIsVal<V>(pArgs.mGetPos(0));
  JJ_DEBUG("fTestTypeFn: type=%s, %s or %s result=%d", typeid(T).name(), typeid(U).name(), typeid(V).name(),
           vIsType ? 1 : 0);
  return fMkVal<value_bool>(vIsType);
}
template <value_compare_op op> static value fTestCompareFn(const func_args &pArgs) {
  pArgs.mEnsureCount(2, 2);
  return fMkVal<value_bool>(fValueCompare(pArgs.mGetPos(0), pArgs.mGetPos(1), op));
}

static void fAppendCodepointAsAsciiJsonEscape(std::string &pOut, uint32_t pCodepoint) {
  auto vAppendU16 = [&pOut](uint32_t pValue) {
    char aBuf[8];
    snprintf(aBuf, sizeof(aBuf), "\\u%04x", static_cast<unsigned int>(pValue));
    pOut += aBuf;
  };

  if (pCodepoint <= 0xFFFF) {
    vAppendU16(pCodepoint);
    return;
  }

  pCodepoint -= 0x10000;
  vAppendU16(0xD800 + ((pCodepoint >> 10) & 0x3FF));
  vAppendU16(0xDC00 + (pCodepoint & 0x3FF));
}

static std::string fJsonEnsureAsciiPreservingFormat(const std::string &pJsonStr) {
  std::string vOutput;
  vOutput.reserve(pJsonStr.size());

  bool vInString = false;
  bool vEscaped = false;

  for (size_t vPos = 0; vPos < pJsonStr.size();) {
    const char cCh = pJsonStr[vPos];
    if (!vInString) {
      vOutput.push_back(cCh);
      if (cCh == '"') {
        vInString = true;
      }
      ++vPos;
      continue;
    }

    if (vEscaped) {
      vOutput.push_back(cCh);
      vEscaped = false;
      ++vPos;
      continue;
    }

    if (cCh == '\\') {
      vOutput.push_back(cCh);
      vEscaped = true;
      ++vPos;
      continue;
    }

    if (cCh == '"') {
      vOutput.push_back(cCh);
      vInString = false;
      ++vPos;
      continue;
    }

    const unsigned char cUch = static_cast<unsigned char>(cCh);
    if (cUch < 0x80) {
      vOutput.push_back(cCh);
      ++vPos;
      continue;
    }

    auto vParsed = fCommonParseUtf8Codepoint(pJsonStr, vPos);
    if (vParsed.vStatus != utf8_parse_result::cSuccess) {
      vOutput += "\\ufffd";
      ++vPos;
      continue;
    }

    fAppendCodepointAsAsciiJsonEscape(vOutput, vParsed.vCodepoint);
    vPos += vParsed.vBytesConsumed;
  }

  return vOutput;
}

static value fTojson(const func_args &pArgs) {
  pArgs.mEnsureCount(1, 5);
  value vValAscii = pArgs.mGetKwargOrPos("ensure_ascii", 1);
  value vValIndent = pArgs.mGetKwargOrPos("indent", 2);
  value vValSeparators = pArgs.mGetKwargOrPos("separators", 3);
  value vValSort = pArgs.mGetKwargOrPos("sort_keys", 4);
  int vIndent = -1;
  if (pArgs.vCtx.vIsGetStats) {
    // mark as used (recursively) for stats
    auto vValInput = pArgs.mGetPos(0);
    value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
  }
  if (fIsVal<value_int>(vValIndent)) {
    // every line of the output allocates the indentation times its nesting level
    if (vValIndent->mAsInt() > cMaxIndentWidth) {
      throw raised_exception("tojson() indent exceeds the maximum of " + std::to_string(cMaxIndentWidth));
    }
    // every negative indent means the same (no newlines); -1 keeps the narrowing to int from changing its sign
    vIndent = vValIndent->mAsInt() < 0 ? -1 : static_cast<int>(vValIndent->mAsInt());
  }
  if (vValSort->mAsBool()) { // undefined == false
    throw not_implemented_exception("tojson sort_keys=true not implemented");
  }
  const bool cEnsureAscii = vValAscii->mAsBool(); // undefined == false
  auto vSeparators = (fIsVal<value_array>(vValSeparators) ? vValSeparators : fMkVal<value_array>())->mAsArray();
  std::string vItemSep = vSeparators.size() > 0 ? vSeparators[0]->mAsString().mStr() : (vIndent < 0 ? ", " : ",");
  std::string vKeySep = vSeparators.size() > 1 ? vSeparators[1]->mAsString().mStr() : ": ";
  std::string vJsonStr = fValueToJson(pArgs.mGetPos(0), vIndent, vItemSep, vKeySep);
  if (cEnsureAscii) {
    vJsonStr = fJsonEnsureAsciiPreservingFormat(vJsonStr);
  }
  return fMkVal<value_string>(vJsonStr);
}

template <bool is_reject> static value fSelectattr(const func_args &pArgs) {
  pArgs.mEnsureCount(2, 4);
  pArgs.mEnsureVals<value_array, value_string, value_string, value_string>(true, true, false, false);

  auto vArr = pArgs.mGetPos(0)->mAsArray();
  auto vAttribute = pArgs.mGetPos(1);
  auto vOut = fMkVal<value_array>();
  value vValDefault = fMkVal<value_undefined>();

  if (pArgs.mCount() == 2) {
    // example: array | selectattr("active")
    for (const auto &vItem : vArr) {
      if (!fIsVal<value_object>(vItem)) {
        throw raised_exception("selectattr: item is not an object");
      }
      value vAttrVal = vItem->mAt(vAttribute, vValDefault);
      bool vIsSelected = vAttrVal->mAsBool();
      if constexpr (is_reject)
        vIsSelected = !vIsSelected;
      if (vIsSelected)
        vOut->mPushBack(vItem);
    }
    return vOut;

  } else if (pArgs.mCount() == 3) {
    // example: array | selectattr("equalto", "text")
    // translated to: test_is_equalto(item, "text")
    std::string vTestName = pArgs.mGetPos(1)->mAsString().mStr();
    value vTestVal = pArgs.mGetPos(2);
    auto &vBuiltins = fGlobalBuiltins();
    auto vIt = vBuiltins.find("test_is_" + vTestName);
    if (vIt == vBuiltins.end()) {
      throw raised_exception("selectattr: unknown test '" + vTestName + "'");
    }
    auto vTestFn = vIt->second;
    for (const auto &vItem : vArr) {
      func_args vTestArgs(pArgs.vCtx);
      vTestArgs.mPushBack(vItem);    // current object
      vTestArgs.mPushBack(vTestVal); // extra argument
      value vTestResult = vTestFn(vTestArgs);
      bool vIsSelected = vTestResult->mAsBool();
      if constexpr (is_reject)
        vIsSelected = !vIsSelected;
      if (vIsSelected)
        vOut->mPushBack(vItem);
    }
    return vOut;

  } else if (pArgs.mCount() == 4) {
    // example: array | selectattr("status", "equalto", "active")
    // translated to: test_is_equalto(item.status, "active")
    std::string vTestName = pArgs.mGetPos(2)->mAsString().mStr();
    auto vExtraArg = pArgs.mGetPos(3);
    auto &vBuiltins = fGlobalBuiltins();
    auto vIt = vBuiltins.find("test_is_" + vTestName);
    if (vIt == vBuiltins.end()) {
      throw raised_exception("selectattr: unknown test '" + vTestName + "'");
    }
    auto vTestFn = vIt->second;
    for (const auto &vItem : vArr) {
      if (!fIsVal<value_object>(vItem)) {
        throw raised_exception("selectattr: item is not an object");
      }
      value vAttrVal = vItem->mAt(vAttribute, vValDefault);
      func_args vTestArgs(pArgs.vCtx);
      vTestArgs.mPushBack(vAttrVal);  // attribute value
      vTestArgs.mPushBack(vExtraArg); // extra argument
      value vTestResult = vTestFn(vTestArgs);
      bool vIsSelected = vTestResult->mAsBool();
      if constexpr (is_reject)
        vIsSelected = !vIsSelected;
      if (vIsSelected)
        vOut->mPushBack(vItem);
    }
    return vOut;
  } else {
    throw raised_exception("selectattr: invalid number of arguments");
  }

  return vOut;
}

static value fDefaultValue(const func_args &pArgs) {
  pArgs.mEnsureCount(2, 3);
  value vValCheck = pArgs.mGetKwargOrPos("boolean", 2);
  bool vCheckBool = vValCheck->mAsBool(); // undefined == false
  bool vNoValue =
      vCheckBool ? (!pArgs.mGetPos(0)->mAsBool()) : (pArgs.mGetPos(0)->mIsUndefined() || pArgs.mGetPos(0)->mIsNone());
  return vNoValue ? pArgs.mGetPos(1) : pArgs.mGetPos(0);
}

const func_builtins &fGlobalBuiltins() {
  static const func_builtins cBuiltins = {
      {"raise_exception",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         std::string vMsg = pArgs.mGetPos(0)->mAsString().mStr();
         throw raised_exception("Jinja Exception: " + vMsg);
       }},
      {"namespace",
       [](const func_args &pArgs) -> value {
         auto vOut = fMkVal<value_object>();
         for (const auto &vArg : pArgs.mGetArgs()) {
           if (!fIsVal<value_kwarg>(vArg)) {
             throw raised_exception("namespace() arguments must be kwargs");
           }
           auto vKwarg = fCastVal<value_kwarg>(vArg);
           JJ_DEBUG("namespace: adding key '%s'", vKwarg->vKey.c_str());
           vOut->insert(vKwarg->vKey, vKwarg->vVal);
         }
         return vOut;
       }},
      {"strftime_now",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         std::string vFormat = pArgs.mGetPos(0)->mAsString().mStr();
         // get current time
         // TODO: make sure this is the same behavior as Python's strftime
         char aBuf[100];
         if (std::strftime(aBuf, sizeof(aBuf), vFormat.c_str(), std::localtime(&pArgs.vCtx.vCurrentTime))) {
           return fMkVal<value_string>(std::string(aBuf));
         } else {
           throw raised_exception("strftime_now: failed to format time");
         }
       }},
      {"range",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 3);
         pArgs.mEnsureVals<value_int, value_int, value_int>(true, false, false);

         auto vArg0 = pArgs.mGetPos(0);
         auto vArg1 = pArgs.mGetPos(1, fMkVal<value_undefined>());
         auto vArg2 = pArgs.mGetPos(2, fMkVal<value_undefined>());

         int64_t vStart, vStop, vStep;
         if (pArgs.mCount() == 1) {
           vStart = 0;
           vStop = vArg0->mAsInt();
           vStep = 1;
         } else if (pArgs.mCount() == 2) {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = 1;
         } else {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = vArg2->mAsInt();
         }

         auto vOut = fMkVal<value_array>();
         if (vStep == 0) {
           throw raised_exception("range() step argument must not be zero");
         }
         // The number of elements is computed without overflow and bounded before any is produced, since the
         // template comes from the model file; the last element is never stepped past, so stepping cannot overflow
         uint64_t vCount = 0;
         if (vStep > 0 && vStart < vStop) {
           const uint64_t cSpan = static_cast<uint64_t>(vStop) - static_cast<uint64_t>(vStart);
           vCount = (cSpan - 1) / static_cast<uint64_t>(vStep) + 1;
         } else if (vStep < 0 && vStart > vStop) {
           const uint64_t cSpan = static_cast<uint64_t>(vStart) - static_cast<uint64_t>(vStop);
           const uint64_t cStepMagnitude = static_cast<uint64_t>(-(vStep + 1)) + 1;
           vCount = (cSpan - 1) / cStepMagnitude + 1;
         }
         if (vCount > cMaxRangeLength) {
           throw raised_exception("range() would produce " + std::to_string(vCount) +
                                  " elements, more than the maximum of " + std::to_string(cMaxRangeLength));
         }
         int64_t vI = vStart;
         for (uint64_t vN = 0; vN < vCount; ++vN) {
           vOut->mPushBack(fMkVal<value_int>(vI));
           if (vN + 1 < vCount) {
             vI += vStep;
           }
         }
         return vOut;
       }},
      {"tojson", fTojson},

      // tests
      {"test_is_boolean", fTestTypeFn<value_bool>},
      {"test_is_callable", fTestTypeFn<value_func>},
      {"test_is_odd",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int>();
         int64_t vVal = pArgs.mGetPos(0)->mAsInt();
         return fMkVal<value_bool>(vVal % 2 != 0);
       }},
      {"test_is_even",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int>();
         int64_t vVal = pArgs.mGetPos(0)->mAsInt();
         return fMkVal<value_bool>(vVal % 2 == 0);
       }},
      {"test_is_false",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1);
         bool vVal = fIsVal<value_bool>(pArgs.mGetPos(0)) && !pArgs.mGetPos(0)->mAsBool();
         return fMkVal<value_bool>(vVal);
       }},
      {"test_is_true",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1);
         bool vVal = fIsVal<value_bool>(pArgs.mGetPos(0)) && pArgs.mGetPos(0)->mAsBool();
         return fMkVal<value_bool>(vVal);
       }},
      {"test_is_divisibleby",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int, value_int>();
         bool vRes = pArgs.mGetPos(0)->vValInt % pArgs.mGetPos(1)->vValInt == 0;
         return fMkVal<value_bool>(vRes);
       }},
      {"test_is_string", fTestTypeFn<value_string>},
      {"test_is_integer", fTestTypeFn<value_int>},
      {"test_is_float", fTestTypeFn<value_float>},
      {"test_is_number", fTestTypeFn<value_int, value_float>},
      {"test_is_iterable", fTestTypeFn<value_array, value_string, value_undefined>},
      {"test_is_sequence", fTestTypeFn<value_array, value_string, value_undefined>},
      {"test_is_mapping", fTestTypeFn<value_object>},
      {"test_is_lower",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         return fMkVal<value_bool>(pArgs.mGetPos(0)->vValStr.mIsLowercase());
       }},
      {"test_is_upper",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         return fMkVal<value_bool>(pArgs.mGetPos(0)->vValStr.mIsUppercase());
       }},
      {"test_is_none", fTestTypeFn<value_none>},
      {"test_is_defined",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1);
         bool vRes = !pArgs.mGetPos(0)->mIsUndefined();
         JJ_DEBUG("test_is_defined: result=%d", vRes ? 1 : 0);
         return fMkVal<value_bool>(vRes);
       }},
      {"test_is_undefined", fTestTypeFn<value_undefined>},
      {"test_is_eq", fTestCompareFn<value_compare_op::cEq>},
      {"test_is_equalto", fTestCompareFn<value_compare_op::cEq>},
      {"test_is_ge", fTestCompareFn<value_compare_op::cGe>},
      {"test_is_gt", fTestCompareFn<value_compare_op::cGt>},
      {"test_is_greaterthan", fTestCompareFn<value_compare_op::cGt>},
      {"test_is_lt", fTestCompareFn<value_compare_op::cLt>},
      {"test_is_lessthan", fTestCompareFn<value_compare_op::cLt>},
      {"test_is_ne", fTestCompareFn<value_compare_op::cNe>},
      {"test_is_in",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(2);
         auto vNeedle = pArgs.mGetPos(0);
         auto vHaystack = pArgs.mGetPos(1);
         if (fIsVal<value_undefined>(vHaystack)) {
           return fMkVal<value_bool>(false);
         }
         if (fIsVal<value_array>(vHaystack)) {
           for (const auto &vItem : vHaystack->mAsArray()) {
             if (*vNeedle == *vItem) {
               return fMkVal<value_bool>(true);
             }
           }
           return fMkVal<value_bool>(false);
         }
         if (fIsVal<value_string>(vHaystack)) {
           if (!fIsVal<value_string>(vNeedle)) {
             throw raised_exception("'in' test expects args[1] as string when args[0] is string, got args[1] as " +
                                    vNeedle->mType());
           }
           return fMkVal<value_bool>(vHaystack->mAsString().mStr().find(vNeedle->mAsString().mStr()) !=
                                     std::string::npos);
         }
         if (fIsVal<value_object>(vHaystack)) {
           return fMkVal<value_bool>(vHaystack->mHasKey(vNeedle));
         }
         throw raised_exception("'in' test expects iterable as first argument, got " + vHaystack->mType());
       }},
      {"test_is_test",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         auto &vBuiltins = fGlobalBuiltins();
         std::string vTestName = pArgs.mGetPos(0)->vValStr.mStr();
         auto vIt = vBuiltins.find("test_is_" + vTestName);
         bool vRes = vIt != vBuiltins.end();
         return fMkVal<value_bool>(vRes);
       }},
      {"test_is_sameas",
       [](const func_args &pArgs) -> value {
         // Check if an object points to the same memory address as another object
         (void)pArgs;
         throw not_implemented_exception("sameas test not implemented");
       }},
      {"test_is_escaped",
       [](const func_args &pArgs) -> value {
         (void)pArgs;
         throw not_implemented_exception("escaped test not implemented");
       }},
      {"test_is_filter",
       [](const func_args &pArgs) -> value {
         (void)pArgs;
         throw not_implemented_exception("filter test not implemented");
       }},
  };
  return cBuiltins;
}

const func_builtins &value_int_t::mGetBuiltins() const {
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"abs",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int>();
         int64_t vVal = pArgs.mGetPos(0)->mAsInt();
         return fMkVal<value_int>(vVal < 0 ? -vVal : vVal);
       }},
      {"int",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int>();
         return fMkVal<value_int>(pArgs.mGetPos(0)->mAsInt());
       }},
      {"float",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_int>();
         double vVal = static_cast<double>(pArgs.mGetPos(0)->mAsInt());
         return fMkVal<value_float>(vVal);
       }},
      {"safe", fTojson},
      {"string", fTojson},
      {"tojson", fTojson},
  };
  return cBuiltins;
}

const func_builtins &value_float_t::mGetBuiltins() const {
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"abs",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_float>();
         double vVal = pArgs.mGetPos(0)->mAsFloat();
         return fMkVal<value_float>(vVal < 0.0 ? -vVal : vVal);
       }},
      {"int",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_float>();
         int64_t vVal = static_cast<int64_t>(pArgs.mGetPos(0)->mAsFloat());
         return fMkVal<value_int>(vVal);
       }},
      {"float",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_float>();
         return fMkVal<value_float>(pArgs.mGetPos(0)->mAsFloat());
       }},
      {"safe", fTojson},
      {"string", fTojson},
      {"tojson", fTojson},
  };
  return cBuiltins;
}

static bool fStringStartswith(const std::string &pStr, const std::string &pPrefix) {
  if (pStr.length() < pPrefix.length())
    return false;
  return pStr.compare(0, pPrefix.length(), pPrefix) == 0;
}

static bool fStringEndswith(const std::string &pStr, const std::string &pSuffix) {
  if (pStr.length() < pSuffix.length())
    return false;
  return pStr.compare(pStr.length() - pSuffix.length(), pSuffix.length(), pSuffix) == 0;
}

[[noreturn]] static value fStringJoinNotImplemented(const func_args &) {
  throw not_implemented_exception("String join builtin not implemented");
}

const func_builtins &value_string_t::mGetBuiltins() const {
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"upper",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         jinja::string vStr = pArgs.mGetPos(0)->mAsString().mUppercase();
         return fMkVal<value_string>(vStr);
       }},
      {"lower",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         jinja::string vStr = pArgs.mGetPos(0)->mAsString().mLowercase();
         return fMkVal<value_string>(vStr);
       }},
      {"strip",
       [](const func_args &pArgs) -> value {
         value vValInput = pArgs.mGetPos(0);
         if (!fIsVal<value_string>(vValInput)) {
           throw raised_exception("strip() first argument must be a string");
         }
         value vValChars = pArgs.mGetKwargOrPos("chars", 1);
         if (vValChars->mIsUndefined()) {
           return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, true));
         } else {
           return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, true, vValChars->mAsString().mStr()));
         }
       }},
      {"rstrip",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         value vValChars = pArgs.mGetKwargOrPos("chars", 1);
         if (vValChars->mIsUndefined()) {
           return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(false, true));
         } else {
           return fMkVal<value_string>(
               pArgs.mGetPos(0)->mAsString().mStrip(false, true, vValChars->mAsString().mStr()));
         }
       }},
      {"lstrip",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         value vValChars = pArgs.mGetKwargOrPos("chars", 1);
         if (vValChars->mIsUndefined()) {
           return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, false));
         } else {
           return fMkVal<value_string>(
               pArgs.mGetPos(0)->mAsString().mStrip(true, false, vValChars->mAsString().mStr()));
         }
       }},
      {"title",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         jinja::string vStr = pArgs.mGetPos(0)->mAsString().mTitlecase();
         return fMkVal<value_string>(vStr);
       }},
      {"capitalize",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         jinja::string vStr = pArgs.mGetPos(0)->mAsString().mCapitalize();
         return fMkVal<value_string>(vStr);
       }},
      {"length",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         jinja::string vStr = pArgs.mGetPos(0)->mAsString();
         return fMkVal<value_int>(vStr.mLength());
       }},
      {"startswith",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string, value_string>();
         std::string vStr = pArgs.mGetPos(0)->mAsString().mStr();
         std::string vPrefix = pArgs.mGetPos(1)->mAsString().mStr();
         return fMkVal<value_bool>(fStringStartswith(vStr, vPrefix));
       }},
      {"endswith",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string, value_string>();
         std::string vStr = pArgs.mGetPos(0)->mAsString().mStr();
         std::string vSuffix = pArgs.mGetPos(1)->mAsString().mStr();
         return fMkVal<value_bool>(fStringEndswith(vStr, vSuffix));
       }},
      {"split",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 3);
         value vValInput = pArgs.mGetPos(0);
         if (!fIsVal<value_string>(vValInput)) {
           throw raised_exception("split() first argument must be a string");
         }
         std::string vStr = vValInput->mAsString().mStr();
         // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
         std::string vDelim = (pArgs.mCount() > 1) ? pArgs.mGetPos(1)->mAsString().mStr() : " ";
         if (vDelim.empty()) {
           throw raised_exception("empty separator");
         }
         int64_t vMaxsplit = (pArgs.mCount() > 2) ? pArgs.mGetPos(2)->mAsInt() : -1;
         auto vResult = fMkVal<value_array>();
         size_t vPos = 0;
         std::string vToken;
         while ((vPos = vStr.find(vDelim)) != std::string::npos && vMaxsplit != 0) {
           vToken = vStr.substr(0, vPos);
           vResult->mPushBack(fMkVal<value_string>(vToken));
           vStr.erase(0, vPos + vDelim.length());
           --vMaxsplit;
         }
         auto vRes = fMkVal<value_string>(vStr);
         vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
         vResult->mPushBack(std::move(vRes));
         return vResult;
       }},
      {"rsplit",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 3);
         value vValInput = pArgs.mGetPos(0);
         if (!fIsVal<value_string>(vValInput)) {
           throw raised_exception("rsplit() first argument must be a string");
         }
         std::string vStr = vValInput->mAsString().mStr();
         // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
         std::string vDelim = (pArgs.mCount() > 1) ? pArgs.mGetPos(1)->mAsString().mStr() : " ";
         if (vDelim.empty()) {
           throw raised_exception("empty separator");
         }
         int64_t vMaxsplit = (pArgs.mCount() > 2) ? pArgs.mGetPos(2)->mAsInt() : -1;
         auto vResult = fMkVal<value_array>();
         size_t vPos = 0;
         std::string vToken;
         while ((vPos = vStr.rfind(vDelim)) != std::string::npos && vMaxsplit != 0) {
           vToken = vStr.substr(vPos + vDelim.length());
           vResult->mPushBack(fMkVal<value_string>(vToken));
           vStr.erase(vPos);
           --vMaxsplit;
         }
         auto vRes = fMkVal<value_string>(vStr);
         vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
         vResult->mPushBack(std::move(vRes));
         vResult->mReverse();
         return vResult;
       }},
      {"replace",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string, value_string, value_string, value_int>(true, true, true, false);
         std::string vStr = pArgs.mGetPos(0)->mAsString().mStr();
         std::string vOldStr = pArgs.mGetPos(1)->mAsString().mStr();
         std::string vNewStr = pArgs.mGetPos(2)->mAsString().mStr();
         int64_t vCount = pArgs.mCount() > 3 ? pArgs.mGetPos(3)->mAsInt() : -1;
         if (vCount > 0) {
           throw not_implemented_exception("String replace with count argument not implemented");
         }
         if (vOldStr != vNewStr) {
           if (vOldStr.empty()) {
             // the new string goes before every character and at the end
             fCheckGrownString(vStr.length(), vNewStr.length(), vStr.length() + 1, "replace()");
             std::string vNewRes;
             vNewRes.reserve(vStr.length() + vNewStr.length() * (vStr.length() + 1));
             vNewRes += vNewStr;
             for (const char cC : vStr) {
               vNewRes.push_back(cC);
               vNewRes += vNewStr;
             }
             vStr = vNewRes;
           } else {
             // the result size is checked before it is built, and it is built in one pass (replacing in place
             // moved the rest of the string once per occurrence)
             if (vNewStr.length() > vOldStr.length()) {
               uint64_t vOccurrences = 0;
               for (size_t vFound = vStr.find(vOldStr); vFound != std::string::npos;
                    vFound = vStr.find(vOldStr, vFound + vOldStr.length())) {
                 ++vOccurrences;
               }
               fCheckGrownString(vStr.length(), vNewStr.length() - vOldStr.length(), vOccurrences, "replace()");
             }
             fStringReplaceAll(vStr, vOldStr, vNewStr);
           }
         }
         auto vRes = fMkVal<value_string>(vStr);
         vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
         return vRes;
       }},
      {"format",
       [](const func_args &pArgs) -> value {
         value vValInput = pArgs.mGetPos(0);
         if (!fIsVal<value_string>(vValInput)) {
           throw raised_exception("format() first argument must be a string");
         }
         const jinja::string &vFmt = vValInput->mAsString();
         const bool cFmtIsInput = vFmt.mAllPartsAreInput();

         const std::string cStr = vFmt.mStr();
         jinja::string vResult;
         std::string vLiteral;
         auto vFlushLiteral = [&]() {
           if (!vLiteral.empty()) {
             vResult.lParts.push_back({cFmtIsInput, vLiteral});
             vLiteral.clear();
           }
         };

         size_t vArgIdx = 1; // positional args follow the format string
         for (size_t vI = 0; vI < cStr.size(); ++vI) {
           if (cStr[vI] != '{') {
             vLiteral += cStr[vI];
             continue;
           }
           if (vI + 1 >= cStr.size() || cStr[vI + 1] != '}') {
             throw not_implemented_exception("format() only supports simple '{}' placeholders");
           }
           ++vI;
           vFlushLiteral();
           const jinja::string cArgStr = pArgs.mGetPos(vArgIdx++)->mAsString();
           vResult.lParts.insert(vResult.lParts.end(), cArgStr.lParts.begin(), cArgStr.lParts.end());
         }
         vFlushLiteral();
         return fMkVal<value_string>(vResult);
       }},
      {"int",
       [](const func_args &pArgs) -> value {
         value vValInput = pArgs.mGetPos(0);
         value vValDefault = pArgs.mGetKwargOrPos("default", 1);
         value vValBase = pArgs.mGetKwargOrPos("base", 2);
         const int cBase = vValBase->mIsUndefined() ? 10 : vValBase->mAsInt();
         if (cBase != 0 && (cBase < 2 || cBase > 36)) {
           // an out-of-range base makes std::stoi fail fast on the MSVC CRT instead of throwing
           throw raised_exception("int() base must be 0 or between 2 and 36");
         }
         if (fIsVal<value_string>(vValInput) == false) {
           throw raised_exception("int() first argument must be a string");
         }
         std::string vStr = vValInput->mAsString().mStr();
         try {
           return fMkVal<value_int>(std::stoi(vStr, nullptr, cBase));
         } catch (...) {
           return fMkVal<value_int>(vValDefault->mIsUndefined() ? 0 : vValDefault->mAsInt());
         }
       }},
      {"float",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_string>();
         value vValDefault = pArgs.mGetKwargOrPos("default", 1);
         std::string vStr = pArgs.mGetPos(0)->mAsString().mStr();
         try {
           return fMkVal<value_float>(std::stod(vStr));
         } catch (...) {
           return fMkVal<value_float>(vValDefault->mIsUndefined() ? 0.0 : vValDefault->mAsFloat());
         }
       }},
      {"string",
       [](const func_args &pArgs) -> value {
         // no-op
         pArgs.mEnsureVals<value_string>();
         return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString());
       }},
      {"default",
       [](const func_args &pArgs) -> value {
         value vInput = pArgs.mGetPos(0);
         if (!fIsVal<value_string>(vInput)) {
           throw raised_exception("default() first argument must be a string");
         }
         value vDefaultVal = fMkVal<value_string>("");
         if (pArgs.mCount() > 1 && !pArgs.mGetPos(1)->mIsUndefined()) {
           vDefaultVal = pArgs.mGetPos(1);
         }
         value vBooleanVal = pArgs.mGetKwargOrPos("boolean", 2); // undefined == false
         if (vInput->mIsUndefined() || (vBooleanVal->mAsBool() && !vInput->mAsBool())) {
           return vDefaultVal;
         } else {
           return vInput;
         }
       }},
      {"slice",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         pArgs.mEnsureVals<value_string, value_int, value_int, value_int>(true, true, false, false);

         auto vArg0 = pArgs.mGetPos(1);
         auto vArg1 = pArgs.mGetPos(2, fMkVal<value_undefined>());
         auto vArg2 = pArgs.mGetPos(3, fMkVal<value_undefined>());

         int64_t vStart, vStop, vStep;
         if (pArgs.mCount() == 1) {
           vStart = 0;
           vStop = vArg0->mAsInt();
           vStep = 1;
         } else if (pArgs.mCount() == 2) {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = 1;
         } else {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = vArg2->mAsInt();
         }
         if (vStep == 0) {
           throw raised_exception("slice step cannot be zero");
         }
         auto vInput = pArgs.mGetPos(0);
         auto vSliced = fSlice(vInput->mAsString().mStr(), vStart, vStop, vStep);
         auto vRes = fMkVal<value_string>(vSliced);
         vRes->vValStr.mMarkInputBasedOn(vInput->mAsString());
         return vRes;
       }},
      {"safe",
       [](const func_args &pArgs) -> value {
         // no-op for now
         pArgs.mEnsureVals<value_string>();
         return pArgs.mGetPos(0);
       }},
      {"tojson", fTojson},
      {"indent",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         value vValInput = pArgs.mGetPos(0);
         value vValWidth = pArgs.mGetKwargOrPos("width", 1);
         const bool cFirst = pArgs.mGetKwargOrPos("first", 2)->mAsBool(); // undefined == false
         const bool cBlank = pArgs.mGetKwargOrPos("blank", 3)->mAsBool(); // undefined == false
         if (!fIsVal<value_string>(vValInput)) {
           throw raised_exception("indent() first argument must be a string");
         }
         std::string vIndent;
         if (fIsVal<value_int>(vValWidth)) {
           // the width allocates that many spaces for every indented line
           if (vValWidth->mAsInt() > cMaxIndentWidth) {
             throw raised_exception("indent() width exceeds the maximum of " + std::to_string(cMaxIndentWidth));
           }
           vIndent.assign(vValWidth->mAsInt(), ' ');
         } else if (fIsVal<value_string>(vValWidth)) {
           vIndent = vValWidth->mAsString().mStr();
         } else {
           vIndent = "    ";
         }
         std::string vIndented;
         std::string vInput = vValInput->mAsString().mStr();
         std::istringstream vIss = std::istringstream(vInput);
         std::string vLine;
         while (std::getline(vIss, vLine)) {
           if (!vIndented.empty()) {
             vIndented.push_back('\n');
           }
           if ((vIndented.empty() ? cFirst : (!vLine.empty() || cBlank))) {
             vIndented += vIndent;
           }
           vIndented += vLine;
         }
         if (!vInput.empty() && vInput.back() == '\n') {
           vIndented.push_back('\n');
           if (cBlank) {
             vIndented += vIndent;
           }
         }

         auto vRes = fMkVal<value_string>(vIndented);
         vRes->vValStr.mMarkInputBasedOn(vValInput->mAsString());
         return vRes;
       }},
      {"join", fStringJoinNotImplemented},
  };
  return cBuiltins;
}

const func_builtins &value_bool_t::mGetBuiltins() const {
  static const func_handler cTostring = [](const func_args &pArgs) -> value {
    pArgs.mEnsureVals<value_bool>();
    bool vVal = pArgs.mGetPos(0)->mAsBool();
    return fMkVal<value_string>(vVal ? "True" : "False");
  };
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"int",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_bool>();
         bool vVal = pArgs.mGetPos(0)->mAsBool();
         return fMkVal<value_int>(vVal ? 1 : 0);
       }},
      {"float",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_bool>();
         bool vVal = pArgs.mGetPos(0)->mAsBool();
         return fMkVal<value_float>(vVal ? 1.0 : 0.0);
       }},
      {"safe", cTostring},
      {"string", cTostring},
      {"tojson", fTojson},
  };
  return cBuiltins;
}

[[noreturn]] static value fArrayUniqueNotImplemented(const func_args &) {
  throw not_implemented_exception("Array unique builtin not implemented");
}

const func_builtins &value_array_t::mGetBuiltins() const {
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"list",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         auto vResult = fMkVal<value_array>();
         for (const auto &vV : vArr) {
           vResult->mPushBack(vV);
         }
         return vResult;
       }},
      {"first",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         if (vArr.empty()) {
           return fMkVal<value_undefined>();
         }
         return vArr[0];
       }},
      {"last",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         if (vArr.empty()) {
           return fMkVal<value_undefined>();
         }
         return vArr[vArr.size() - 1];
       }},
      {"length",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         return fMkVal<value_int>(static_cast<int64_t>(vArr.size()));
       }},
      {"slice",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         pArgs.mEnsureVals<value_array, value_int, value_int, value_int>(true, true, false, false);

         auto vVal = pArgs.mGetPos(0);
         auto vArg0 = pArgs.mGetPos(1);
         auto vArg1 = pArgs.mGetPos(2, fMkVal<value_undefined>());
         auto vArg2 = pArgs.mGetPos(3, fMkVal<value_undefined>());

         int64_t vStart, vStop, vStep;
         if (pArgs.mCount() == 1) {
           vStart = 0;
           vStop = vArg0->mAsInt();
           vStep = 1;
         } else if (pArgs.mCount() == 2) {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = 1;
         } else {
           vStart = vArg0->mAsInt();
           vStop = vArg1->mAsInt();
           vStep = vArg2->mAsInt();
         }
         if (vStep == 0) {
           throw raised_exception("slice step cannot be zero");
         }
         auto vArr = fSlice(vVal->mAsArray(), vStart, vStop, vStep);
         return fIsVal<value_tuple>(vVal) ? fMkVal<value_tuple>(std::move(vArr)) : fMkVal<value_array>(std::move(vArr));
       }},
      {"selectattr", fSelectattr<false>},
      {"select", fSelectattr<false>},
      {"rejectattr", fSelectattr<true>},
      {"reject", fSelectattr<true>},
      {"join",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 3);
         if (!fIsVal<value_array>(pArgs.mGetPos(0))) {
           throw raised_exception("join() first argument must be an array");
         }
         value vValDelim = pArgs.mGetKwargOrPos("d", 1);
         value vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         const bool cAttrIsInt = fIsVal<value_int>(vAttribute);
         if (!vAttribute->mIsUndefined() && !fIsVal<value_string>(vAttribute) && !cAttrIsInt) {
           throw raised_exception("join() attribute must be string or integer");
         }
         const int64_t cAttrInt = cAttrIsInt ? vAttribute->mAsInt() : 0;
         const std::string cDelim = vValDelim->mIsUndefined() ? "" : vValDelim->mAsString().mStr();
         std::string vResult;
         for (size_t vI = 0; vI < vArr.size(); ++vI) {
           value vValArr = vArr[vI];
           if (!vAttribute->mIsUndefined()) {
             if (cAttrIsInt && fIsVal<value_array>(vValArr)) {
               vValArr = vValArr->mAt(cAttrInt);
             } else if (!cAttrIsInt && fIsVal<value_object>(vValArr)) {
               vValArr = vValArr->mAt(vAttribute);
             }
           }
           if (!fIsVal<value_string>(vValArr) && !fIsVal<value_int>(vValArr) && !fIsVal<value_float>(vValArr)) {
             throw raised_exception("join() can only join arrays of strings or numerics");
           }
           vResult += vValArr->mAsString().mStr();
           if (vI < vArr.size() - 1) {
             vResult += cDelim;
           }
         }
         return fMkVal<value_string>(vResult);
       }},
      {"string",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         if (pArgs.vCtx.vIsGetStats) {
           // mark as used (recursively) for stats
           auto vValInput = pArgs.mGetPos(0);
           value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
         }
         return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString());
       }},
      {"tojson", fTojson},
      {"map",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(2);
         if (!fIsVal<value_array>(pArgs.mGetPos(0))) {
           throw raised_exception("map: first argument must be an array");
         }
         if (!fIsVal<value_kwarg>(pArgs.mGetArgs().at(1))) {
           throw not_implemented_exception("map: filter-mapping not implemented");
         }
         value vVal = pArgs.mGetPos(0);
         value vAttribute = pArgs.mGetKwargOrPos("attribute", 1);
         const bool cAttrIsInt = fIsVal<value_int>(vAttribute);
         if (!fIsVal<value_string>(vAttribute) && !cAttrIsInt) {
           throw raised_exception("map: attribute must be string or integer");
         }
         const int64_t cAttrInt = cAttrIsInt ? vAttribute->mAsInt() : 0;
         value vDefaultVal = pArgs.mGetKwarg("default", fMkVal<value_undefined>());
         auto vOut = fMkVal<value_array>();
         auto vArr = vVal->mAsArray();
         for (const auto &vItem : vArr) {
           value vAttrVal;
           if (cAttrIsInt) {
             vAttrVal = fIsVal<value_array>(vItem) ? vItem->mAt(cAttrInt, vDefaultVal) : vDefaultVal;
           } else {
             vAttrVal = fIsVal<value_object>(vItem) ? vItem->mAt(vAttribute, vDefaultVal) : vDefaultVal;
           }
           vOut->mPushBack(vAttrVal);
         }
         return fIsVal<value_tuple>(vVal) ? fMkVal<value_tuple>(std::move(vOut->mAsArray())) : vOut;
       }},
      {"append",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(2);
         if (!fIsVal<value_array>(pArgs.mGetPos(0))) {
           throw raised_exception("append: first argument must be an array");
         }
         const value_array_t *vArr = fCastVal<value_array>(pArgs.mGetPos(0));
         // need to use const_cast here to modify the array
         value_array_t *vArrEditable = const_cast<value_array_t *>(vArr);
         vArrEditable->mPushBack(pArgs.mGetPos(1));
         return pArgs.mGetPos(0);
       }},
      {"pop",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 2);
         pArgs.mEnsureVals<value_array, value_int>(true, false);
         int64_t vIndex = pArgs.mCount() == 2 ? pArgs.mGetPos(1)->mAsInt() : -1;
         const value_array_t *vArr = fCastVal<value_array>(pArgs.mGetPos(0));
         // need to use const_cast here to modify the array
         value_array_t *vArrEditable = const_cast<value_array_t *>(vArr);
         return vArrEditable->mPopAt(vIndex);
       }},
      {"sort",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         if (!fIsVal<value_array>(pArgs.mGetPos(0))) {
           throw raised_exception("sort: first argument must be an array");
         }
         value vVal = pArgs.mGetPos(0);
         value vValReverse = pArgs.mGetKwargOrPos("reverse", 1);
         value vValCase = pArgs.mGetKwargOrPos("case_sensitive", 2);
         value vAttribute = pArgs.mGetKwargOrPos("attribute", 3);
         // FIXME: sorting is currently always case sensitive
         // const bool cCaseSensitive = vValCase->mAsBool(); // undefined == false
         const bool cReverse = vValReverse->mAsBool(); // undefined == false
         const bool cAttrIsInt = fIsVal<value_int>(vAttribute);
         const int64_t cAttrInt = cAttrIsInt ? vAttribute->mAsInt() : 0;
         std::vector<value> lArr = vVal->mAsArray(); // copy
         std::sort(lArr.begin(), lArr.end(), [&](const value &pA, const value &pB) {
           value vValA = pA;
           value vValB = pB;
           if (!vAttribute->mIsUndefined()) {
             if (cAttrIsInt && fIsVal<value_array>(pA) && fIsVal<value_array>(pB)) {
               vValA = pA->mAt(cAttrInt);
               vValB = pB->mAt(cAttrInt);
             } else if (!cAttrIsInt && fIsVal<value_object>(pA) && fIsVal<value_object>(pB)) {
               vValA = pA->mAt(vAttribute);
               vValB = pB->mAt(vAttribute);
             } else {
               throw raised_exception("sort: unsupported object attribute comparison between " + pA->mType() + " and " +
                                      pB->mType());
             }
           }
           return fValueCompare(vValA, vValB, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
         });
         return fIsVal<value_tuple>(vVal) ? fMkVal<value_tuple>(std::move(lArr)) : fMkVal<value_array>(std::move(lArr));
       }},
      {"reverse",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_array>();
         value vVal = pArgs.mGetPos(0);
         std::vector<value> lArr = vVal->mAsArray(); // copy
         std::reverse(lArr.begin(), lArr.end());
         return fIsVal<value_tuple>(vVal) ? fMkVal<value_tuple>(std::move(lArr)) : fMkVal<value_array>(std::move(lArr));
       }},
      {"min",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         pArgs.mEnsureVals<value_array>();
         value vValCase = pArgs.mGetKwargOrPos("case_sensitive", 1);
         value vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
         if (!vAttribute->mIsUndefined()) {
           throw not_implemented_exception("min: attribute not implemented");
         }
         // FIXME: min is currently always case sensitive
         (void)vValCase;
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         if (vArr.empty()) {
           return fMkVal<value_undefined>();
         }
         value vResult = vArr[0];
         for (size_t vI = 1; vI < vArr.size(); ++vI) {
           if (fValueCompare(vArr[vI], vResult, value_compare_op::cLt)) {
             vResult = vArr[vI];
           }
         }
         return vResult;
       }},
      {"max",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(1, 4);
         pArgs.mEnsureVals<value_array>();
         value vValCase = pArgs.mGetKwargOrPos("case_sensitive", 1);
         value vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
         if (!vAttribute->mIsUndefined()) {
           throw not_implemented_exception("max: attribute not implemented");
         }
         // FIXME: max is currently always case sensitive
         (void)vValCase;
         const auto &vArr = pArgs.mGetPos(0)->mAsArray();
         if (vArr.empty()) {
           return fMkVal<value_undefined>();
         }
         value vResult = vArr[0];
         for (size_t vI = 1; vI < vArr.size(); ++vI) {
           if (fValueCompare(vArr[vI], vResult, value_compare_op::cGt)) {
             vResult = vArr[vI];
           }
         }
         return vResult;
       }},
      {"unique", fArrayUniqueNotImplemented},
  };
  return cBuiltins;
}

[[noreturn]] static value fObjectJoinNotImplemented(const func_args &) {
  throw not_implemented_exception("object join not implemented");
}

const func_builtins &value_object_t::mGetBuiltins() const {
  if (!vHasBuiltins) {
    static const func_builtins cNoBuiltins = {};
    return cNoBuiltins;
  }

  static const func_builtins cBuiltins = {
      // {"default", default_value}, // cause issue with gpt-oss
      {"get",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureCount(2, 3);
         if (!fIsVal<value_object>(pArgs.mGetPos(0))) {
           throw raised_exception("get: first argument must be an object");
         }
         if (!fIsVal<value_string>(pArgs.mGetPos(1))) {
           throw raised_exception("get: second argument must be a string (key)");
         }
         value vDefaultVal = fMkVal<value_none>();
         if (pArgs.mCount() == 3) {
           vDefaultVal = pArgs.mGetPos(2);
         }
         const value cObj = pArgs.mGetPos(0);
         const value cKey = pArgs.mGetPos(1);
         return cObj->mAt(cKey, vDefaultVal);
       }},
      {"keys",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         const auto &vObj = pArgs.mGetPos(0)->mAsOrderedObject();
         auto vResult = fMkVal<value_array>();
         for (const auto &vPair : vObj) {
           vResult->mPushBack(vPair.first);
         }
         return vResult;
       }},
      {"values",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         const auto &vObj = pArgs.mGetPos(0)->mAsOrderedObject();
         auto vResult = fMkVal<value_array>();
         for (const auto &vPair : vObj) {
           vResult->mPushBack(vPair.second);
         }
         return vResult;
       }},
      {"items",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         const auto &vObj = pArgs.mGetPos(0)->mAsOrderedObject();
         auto vResult = fMkVal<value_array>();
         for (const auto &vPair : vObj) {
           auto vItem = fMkVal<value_tuple>(vPair);
           vResult->mPushBack(std::move(vItem));
         }
         return vResult;
       }},
      {"tojson", fTojson},
      {"string",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         if (pArgs.vCtx.vIsGetStats) {
           // mark as used (recursively) for stats
           auto vValInput = pArgs.mGetPos(0);
           value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
         }
         return fMkVal<value_string>(pArgs.mGetPos(0)->mAsString());
       }},
      {"length",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         const auto &vObj = pArgs.mGetPos(0)->mAsOrderedObject();
         return fMkVal<value_int>(static_cast<int64_t>(vObj.size()));
       }},
      {"tojson",
       [](const func_args &pArgs) -> value {
         pArgs.mEnsureVals<value_object>();
         // use global to_json
         return fGlobalBuiltins().at("tojson")(pArgs);
       }},
      {"dictsort",
       [](const func_args &pArgs) -> value {
         value vValInput = pArgs.mGetPos(0);
         value vValCase = pArgs.mGetKwargOrPos("case_sensitive", 1);
         value vValBy = pArgs.mGetKwargOrPos("by", 2);
         value vValReverse = pArgs.mGetKwargOrPos("reverse", 3);
         // FIXME: sorting is currently always case sensitive
         // const bool cCaseSensitive = vValCase->mAsBool(); // undefined == false
         const bool cReverse = vValReverse->mAsBool(); // undefined == false
         const bool cByValue = fIsVal<value_string>(vValBy) && vValBy->mAsString().mStr() == "value" ? true : false;
         auto vResult = fMkVal<value_object>(vValInput); // copy
         std::sort(vResult->lValObj.begin(), vResult->lValObj.end(), [&](const auto &pA, const auto &pB) {
           if (cByValue) {
             return fValueCompare(pA.second, pB.second, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
           } else {
             return fValueCompare(pA.first, pB.first, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
           }
         });
         return vResult;
       }},
      {"join", fObjectJoinNotImplemented},
  };
  return cBuiltins;
}

const func_builtins &value_none_t::mGetBuiltins() const {
  static const func_handler cTostring = [](const func_args &) -> value { return fMkVal<value_string>("None"); };
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"tojson", fTojson},
      {"string", cTostring},
      {"safe", cTostring},
      {"items", fEmptyValueFn<value_array>},
      {"map", fEmptyValueFn<value_array>},
      {"reject", fEmptyValueFn<value_array>},
      {"rejectattr", fEmptyValueFn<value_array>},
      {"select", fEmptyValueFn<value_array>},
      {"selectattr", fEmptyValueFn<value_array>},
      {"unique", fEmptyValueFn<value_array>},
  };
  return cBuiltins;
}

const func_builtins &value_undefined_t::mGetBuiltins() const {
  static const func_builtins cBuiltins = {
      {"default", fDefaultValue},
      {"capitalize", fEmptyValueFn<value_string>},
      {"first", fEmptyValueFn<value_undefined>},
      {"items", fEmptyValueFn<value_array>},
      {"join", fEmptyValueFn<value_string>},
      {"last", fEmptyValueFn<value_undefined>},
      {"length", fEmptyValueFn<value_int>},
      {"list", fEmptyValueFn<value_array>},
      {"lower", fEmptyValueFn<value_string>},
      {"map", fEmptyValueFn<value_array>},
      {"max", fEmptyValueFn<value_undefined>},
      {"min", fEmptyValueFn<value_undefined>},
      {"reject", fEmptyValueFn<value_array>},
      {"rejectattr", fEmptyValueFn<value_array>},
      {"replace", fEmptyValueFn<value_string>},
      {"reverse", fEmptyValueFn<value_array>},
      {"safe", fEmptyValueFn<value_string>},
      {"select", fEmptyValueFn<value_array>},
      {"selectattr", fEmptyValueFn<value_array>},
      {"sort", fEmptyValueFn<value_array>},
      {"string", fEmptyValueFn<value_string>},
      {"strip", fEmptyValueFn<value_string>},
      {"sum", fEmptyValueFn<value_int>},
      {"title", fEmptyValueFn<value_string>},
      {"truncate", fEmptyValueFn<value_string>},
      {"unique", fEmptyValueFn<value_array>},
      {"upper", fEmptyValueFn<value_string>},
      {"wordcount", fEmptyValueFn<value_int>},
  };
  return cBuiltins;
}

//////////////////////////////////

static value from_json(const nlohmann::ordered_json &pJ, bool pMarkInput) {
  if (pJ.is_null()) {
    return fMkVal<value_none>();
  } else if (pJ.is_boolean()) {
    return fMkVal<value_bool>(pJ.get<bool>());
  } else if (pJ.is_number_integer()) {
    return fMkVal<value_int>(pJ.get<int64_t>());
  } else if (pJ.is_number_float()) {
    return fMkVal<value_float>(pJ.get<double>());
  } else if (pJ.is_string()) {
    auto vStr = fMkVal<value_string>(pJ.get<std::string>());
    if (pMarkInput) {
      vStr->mMarkInput();
    }
    return vStr;
  } else if (pJ.is_array()) {
    auto vArr = fMkVal<value_array>();
    for (const auto &vItem : pJ) {
      vArr->mPushBack(from_json(vItem, pMarkInput));
    }
    return vArr;
  } else if (pJ.is_object()) {
    auto vObj = fMkVal<value_object>();
    for (auto vIt = pJ.begin(); vIt != pJ.end(); ++vIt) {
      vObj->insert(vIt.key(), from_json(vIt.value(), pMarkInput));
    }
    return vObj;
  } else {
    throw std::runtime_error("Unsupported JSON value type");
  }
}

// compare operator for value_t
bool fValueCompare(const value &pA, const value &pB, value_compare_op pOp) {
  auto vCmp = [&]() {
    // compare numeric types
    if ((fIsVal<value_int>(pA) || fIsVal<value_float>(pA)) && (fIsVal<value_int>(pB) || fIsVal<value_float>(pB))) {
      try {
        if (pOp == value_compare_op::cEq) {
          return pA->mAsFloat() == pB->mAsFloat();
        } else if (pOp == value_compare_op::cGe) {
          return pA->mAsFloat() >= pB->mAsFloat();
        } else if (pOp == value_compare_op::cGt) {
          return pA->mAsFloat() > pB->mAsFloat();
        } else if (pOp == value_compare_op::cLt) {
          return pA->mAsFloat() < pB->mAsFloat();
        } else if (pOp == value_compare_op::cNe) {
          return pA->mAsFloat() != pB->mAsFloat();
        } else {
          throw std::runtime_error("Unsupported comparison operator for numeric types");
        }
      } catch (...) {
      }
    }
    // compare string and number
    // TODO: not sure if this is the right behavior
    if ((fIsVal<value_string>(pB) && (fIsVal<value_int>(pA) || fIsVal<value_float>(pA))) ||
        (fIsVal<value_string>(pA) && (fIsVal<value_int>(pB) || fIsVal<value_float>(pB))) ||
        (fIsVal<value_string>(pA) && fIsVal<value_string>(pB))) {
      try {
        if (pOp == value_compare_op::cEq) {
          return pA->mAsString().mStr() == pB->mAsString().mStr();
        } else if (pOp == value_compare_op::cGe) {
          return pA->mAsString().mStr() >= pB->mAsString().mStr();
        } else if (pOp == value_compare_op::cGt) {
          return pA->mAsString().mStr() > pB->mAsString().mStr();
        } else if (pOp == value_compare_op::cLt) {
          return pA->mAsString().mStr() < pB->mAsString().mStr();
        } else if (pOp == value_compare_op::cNe) {
          return pA->mAsString().mStr() != pB->mAsString().mStr();
        } else {
          throw std::runtime_error("Unsupported comparison operator for string/number types");
        }
      } catch (...) {
      }
    }
    // compare boolean simple
    if (fIsVal<value_bool>(pA) && fIsVal<value_bool>(pB)) {
      if (pOp == value_compare_op::cEq) {
        return pA->mAsBool() == pB->mAsBool();
      } else if (pOp == value_compare_op::cNe) {
        return pA->mAsBool() != pB->mAsBool();
      } else {
        throw std::runtime_error("Unsupported comparison operator for bool type");
      }
    }
    // compare by type
    if (pA->mType() != pB->mType()) {
      return false;
    }
    return false;
  };
  auto vResult = vCmp();
  JJ_DEBUG("Comparing types: %s and %s result=%d", pA->mType().c_str(), pB->mType().c_str(), vResult);
  return vResult;
}

template <> void fGlobalFromJson(context &pCtx, const nlohmann::ordered_json &pJsonObj, bool pMarkInput) {
  // printf("global_from_json: %s\n" , pJsonObj.dump(2).c_str());
  if (pJsonObj.is_null() || !pJsonObj.is_object()) {
    throw std::runtime_error("fGlobalFromJson: input JSON value must be an object");
  }
  for (auto vIt = pJsonObj.begin(); vIt != pJsonObj.end(); ++vIt) {
    JJ_DEBUG("fGlobalFromJson: setting key '%s'", vIt.key().c_str());
    pCtx.mSetVal(vIt.key(), from_json(vIt.value(), pMarkInput));
  }
}

// recursively convert value to JSON string
// TODO: avoid circular references
static void fValueToJsonInternal(std::ostringstream &pOss, const value &pVal, int pCurrLvl, int pIndent,
                                 const std::string_view pItemSep, const std::string_view pKeySep) {
  auto vIndentStr = [pIndent, pCurrLvl]() -> std::string {
    return (pIndent > 0) ? std::string(pCurrLvl * pIndent, ' ') : "";
  };
  auto vNewline = [pIndent]() -> std::string { return (pIndent >= 0) ? "\n" : ""; };

  if (fIsVal<value_none>(pVal) || pVal->mIsUndefined()) {
    pOss << "null";
  } else if (fIsVal<value_bool>(pVal)) {
    pOss << (pVal->mAsBool() ? "true" : "false");
  } else if (fIsVal<value_int>(pVal)) {
    pOss << pVal->mAsInt();
  } else if (fIsVal<value_float>(pVal)) {
    pOss << pVal->mAsFloat();
  } else if (fIsVal<value_string>(pVal)) {
    pOss << "\"";
    for (char vC : pVal->mAsString().mStr()) {
      switch (vC) {
      case '"':
        pOss << "\\\"";
        break;
      case '\\':
        pOss << "\\\\";
        break;
      case '\b':
        pOss << "\\b";
        break;
      case '\f':
        pOss << "\\f";
        break;
      case '\n':
        pOss << "\\n";
        break;
      case '\r':
        pOss << "\\r";
        break;
      case '\t':
        pOss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(vC) < 0x20) {
          char aBuf[7];
          snprintf(aBuf, sizeof(aBuf), "\\u%04x", static_cast<unsigned char>(vC));
          pOss << aBuf;
        } else {
          pOss << vC;
        }
      }
    }
    pOss << "\"";
  } else if (fIsVal<value_array>(pVal)) {
    const auto &vArr = pVal->mAsArray();
    pOss << "[";
    if (!vArr.empty()) {
      pOss << vNewline();
      for (size_t vI = 0; vI < vArr.size(); ++vI) {
        pOss << vIndentStr() << (pIndent > 0 ? std::string(pIndent, ' ') : "");
        fValueToJsonInternal(pOss, vArr[vI], pCurrLvl + 1, pIndent, pItemSep, pKeySep);
        if (vI < vArr.size() - 1) {
          pOss << pItemSep;
        }
        pOss << vNewline();
      }
      pOss << vIndentStr();
    }
    pOss << "]";
  } else if (fIsVal<value_object>(pVal)) {
    const auto &vObj = pVal->mAsOrderedObject(); // IMPORTANT: need to keep exact order
    pOss << "{";
    if (!vObj.empty()) {
      pOss << vNewline();
      size_t vI = 0;
      for (const auto &vPair : vObj) {
        pOss << vIndentStr() << (pIndent > 0 ? std::string(pIndent, ' ') : "");
        fValueToJsonInternal(pOss, fMkVal<value_string>(vPair.first->mAsString().mStr()), pCurrLvl + 1, pIndent,
                             pItemSep, pKeySep);
        pOss << pKeySep;
        fValueToJsonInternal(pOss, vPair.second, pCurrLvl + 1, pIndent, pItemSep, pKeySep);
        if (vI < vObj.size() - 1) {
          pOss << pItemSep;
        }
        pOss << vNewline();
        ++vI;
      }
      pOss << vIndentStr();
    }
    pOss << "}";
  } else {
    pOss << "null";
  }
}

std::string fValueToJson(const value &pVal, int pIndent, const std::string_view pItemSep,
                         const std::string_view pKeySep) {
  std::ostringstream vOss;
  fValueToJsonInternal(vOss, pVal, 0, pIndent, pItemSep, pKeySep);
  JJ_DEBUG("fValueToJson: result=%s", vOss.str().c_str());
  return vOss.str();
}

// TODO: avoid circular references
std::string fValueToStringRepr(const value &pVal) {
  if (fIsVal<value_string>(pVal)) {
    const std::string cValStr = pVal->mAsString().mStr();

    if (cValStr.find('\'') != std::string::npos) {
      return fValueToJson(pVal);
    } else {
      return "'" + cValStr + "'";
    }
  } else {
    return pVal->mAsRepr();
  }
}

// stats utility
void value_t::stats_t::mMarkUsed(value &pVal, bool pDeep) {
  pVal->vStats.vUsed = true;
  if (pDeep) {
    if (fIsVal<value_array>(pVal)) {
      for (auto &vItem : pVal->lValArr) {
        mMarkUsed(vItem, pDeep);
      }
    } else if (fIsVal<value_object>(pVal)) {
      for (auto &vPair : pVal->lValObj) {
        mMarkUsed(vPair.first, pDeep);
        mMarkUsed(vPair.second, pDeep);
      }
    }
  }
}

} // namespace jinja
