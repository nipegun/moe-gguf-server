#pragma once

#include "string.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace jinja {

// Chat templates come from model files and are untrusted: these limits keep a hostile template from exhausting
// memory, CPU or the native stack through builtins with unbounded arguments or through deep nesting
constexpr uint64_t cMaxRangeLength = 1000000;      // elements produced by one range()
constexpr uint64_t cMaxStringBytes = 256ull << 20; // bytes of a string built by concatenation or repetition
constexpr uint64_t cMaxStringParts = 1ull << 20;   // parts of a string built by concatenation or repetition
constexpr int64_t cMaxIndentWidth = 256;           // spaces of one indent() or tojson() indentation level
constexpr size_t cMaxExecutionDepth = 1024;        // nested statement executions, recursive macros included
constexpr size_t cMaxParseDepth = 256;             // nested blocks, expressions and operator chains

struct value_t;
using value = std::shared_ptr<value_t>;

// Helper to check the type of a value
template <typename T> struct extract_pointee {
  using type = T;
};
template <typename U> struct extract_pointee<std::shared_ptr<U>> {
  using type = U;
};
template <typename T> bool fIsVal(const value &pPtr) {
  using PointeeType = typename extract_pointee<T>::type;
  return dynamic_cast<const PointeeType *>(pPtr.get()) != nullptr;
}
template <typename T> bool fIsVal(const value_t *pPtr) {
  using PointeeType = typename extract_pointee<T>::type;
  return dynamic_cast<const PointeeType *>(pPtr) != nullptr;
}
template <typename T, typename... Args> std::shared_ptr<typename extract_pointee<T>::type> fMkVal(Args &&...pArgs) {
  using PointeeType = typename extract_pointee<T>::type;
  return std::make_shared<PointeeType>(std::forward<Args>(pArgs)...);
}
template <typename T> const typename extract_pointee<T>::type *fCastVal(const value &pPtr) {
  using PointeeType = typename extract_pointee<T>::type;
  return dynamic_cast<const PointeeType *>(pPtr.get());
}
template <typename T> typename extract_pointee<T>::type *fCastVal(value &pPtr) {
  using PointeeType = typename extract_pointee<T>::type;
  return dynamic_cast<PointeeType *>(pPtr.get());
}
// End Helper

struct context; // forward declaration

// for converting from JSON to jinja values
// example input JSON:
// {
//   "messages": [
//     {"role": "user", "content": "Hello!"},
//     {"role": "assistant", "content": "Hi there!"}
//   ],
//   "bos_token": "<s>",
//   "eos_token": "</s>",
// }
//
// to mark strings as user input, wrap them in a special object:
// {
//   "messages": [
//     {
//       "role": "user",
//       "content": {"__input__": "Hello!"}  // this string is user input
//     },
//     ...
//   ],
// }
//
// marking input can be useful for tracking data provenance
// and preventing template injection attacks
//
// Note: T_JSON can be nlohmann::ordered_json
template <typename T_JSON> void fGlobalFromJson(context &pCtx, const T_JSON &pJsonObj, bool pMarkInput);

//
// base value type
//

struct func_args; // function argument values

using func_hptr = value(const func_args &);
using func_handler = std::function<func_hptr>;
using func_builtins = std::map<std::string, func_handler>;

enum value_compare_op { cEq, cGe, cGt, cLt, cNe };
bool fValueCompare(const value &pA, const value &pB, value_compare_op pOp);

struct value_t {
  int64_t vValInt;
  double vValFlt;
  string vValStr;

  std::vector<value> lValArr;
  std::vector<std::pair<value, value>> lValObj;

  func_handler vValFunc;

  // only used if pCtx.vIsGetStats = true
  struct stats_t {
    bool vUsed = false;
    // ops can be builtin calls or operators: "array_access", "object_access"
    std::set<std::string> sOps;
    // utility to recursively mark value and its children as used
    static void mMarkUsed(value &pVal, bool pDeep = false);
  } vStats;

  value_t() = default;
  value_t(const value_t &) = default;
  virtual ~value_t() = default;

  // Note: only for debugging and error reporting purposes
  virtual std::string mType() const { return ""; }

  virtual int64_t mAsInt() const { mThrowTypeError("is not an int value"); }
  virtual double mAsFloat() const { mThrowTypeError("is not a float value"); }
  virtual string mAsString() const { mThrowTypeError("is not a string value"); }
  virtual bool mAsBool() const { mThrowTypeError("is not a bool value"); }
  virtual const std::vector<value> &mAsArray() const { mThrowTypeError("is not an array value"); }
  virtual const std::vector<std::pair<value, value>> &mAsOrderedObject() const {
    mThrowTypeError("is not an object value");
  }
  virtual value mInvoke(const func_args &) const { mThrowTypeError("is not a function value"); }
  virtual bool mIsNone() const { return false; }
  virtual bool mIsUndefined() const { return false; }
  virtual const func_builtins &mGetBuiltins() const { mThrowTypeError("has no builtins"); }

  virtual bool mHasKey(const value &) { mThrowTypeError("is not an object value"); }
  virtual void mInsert(const value & /* key */, const value & /* val */) { mThrowTypeError("is not an object value"); }
  virtual value &mAt(const value & /* key */, value & /* default_val */) { mThrowTypeError("is not an object value"); }
  virtual value &mAt(const value & /* key */) { mThrowTypeError("is not an object value"); }
  virtual value &mAt(const std::string & /* key */, value & /* default_val */) {
    mThrowTypeError("is not an object value");
  }
  virtual value &mAt(const std::string & /* key */) { mThrowTypeError("is not an object value"); }
  virtual value &mAt(int64_t /* idx */, value & /* default_val */) { mThrowTypeError("is not an array value"); }
  virtual value &mAt(int64_t /* idx */) { mThrowTypeError("is not an array value"); }

  virtual bool mIsNumeric() const { return false; }
  virtual bool mIsHashable() const { return false; }
  virtual bool mIsImmutable() const { return true; }
  virtual hasher mUniqueHash() const noexcept = 0;
  // TODO: C++20 <=> operator
  // NOTE: We are treating == as equivalent (for normal comparisons) and != as strict nonequal (for strict (is)
  // comparisons)
  virtual bool operator==(const value_t &pOther) const { return mEquivalent(pOther); }
  virtual bool operator!=(const value_t &pOther) const { return mNonequal(pOther); }

  // Note: only for debugging purposes
  virtual std::string mAsRepr() const { return mAsString().mStr(); }

private:
  [[noreturn]] void mThrowTypeError(const char *pExpected) const {
    throw std::runtime_error(mType() + " " + pExpected);
  }

protected:
  virtual bool mEquivalent(const value_t &) const = 0;
  virtual bool mNonequal(const value_t &pOther) const { return !mEquivalent(pOther); }
};

//
// utils
//

const func_builtins &fGlobalBuiltins();

std::string fValueToJson(const value &pVal, int pIndent = -1, const std::string_view pItemSep = ", ",
                         const std::string_view pKeySep = ": ");

// Note: only used for debugging purposes
std::string fValueToStringRepr(const value &pVal);

struct not_implemented_exception : public std::runtime_error {
  not_implemented_exception(const std::string &pMsg) : std::runtime_error("NotImplemented: " + pMsg) {}
};

struct value_hasher {
  size_t operator()(const value &pVal) const noexcept { return pVal->mUniqueHash().mDigest(); }
};

struct value_equivalence {
  bool operator()(const value &pLhs, const value &pRhs) const { return *pLhs == *pRhs; }
  bool operator()(const std::pair<value, value> &pLhs, const std::pair<value, value> &pRhs) const {
    return *(pLhs.first) == *(pRhs.first) && *(pLhs.second) == *(pRhs.second);
  }
};

struct value_equality {
  bool operator()(const value &pLhs, const value &pRhs) const { return !(*pLhs != *pRhs); }
};

//
// primitive value types
//

struct value_int_t : public value_t {
  value_int_t(int64_t pV) {
    vValInt = pV;
    vValFlt = static_cast<double>(pV);
    if (static_cast<int64_t>(vValFlt) != pV) {
      vValFlt = pV < 0 ? -INFINITY : INFINITY;
    }
  }
  virtual std::string mType() const override { return "Integer"; }
  virtual int64_t mAsInt() const override { return vValInt; }
  virtual double mAsFloat() const override { return vValFlt; }
  virtual string mAsString() const override { return std::to_string(vValInt); }
  virtual bool mAsBool() const override { return vValInt != 0; }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsNumeric() const override { return true; }
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override {
    return hasher(typeid(*this)).mUpdate(&vValInt, sizeof(vValInt)).mUpdate(&vValFlt, sizeof(vValFlt));
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return pOther.mIsNumeric() && vValInt == pOther.vValInt && vValFlt == pOther.vValFlt;
  }
  virtual bool mNonequal(const value_t &pOther) const override {
    return !(typeid(*this) == typeid(pOther) && vValInt == pOther.vValInt);
  }
};
using value_int = std::shared_ptr<value_int_t>;

struct value_float_t : public value_t {
  value vVal;
  value_float_t(double pV) {
    vValFlt = pV;
    vValInt = std::isfinite(pV) ? static_cast<int64_t>(pV) : 0;
    vVal = fMkVal<value_int>(vValInt);
  }
  virtual std::string mType() const override { return "Float"; }
  virtual double mAsFloat() const override { return vValFlt; }
  virtual int64_t mAsInt() const override { return vValInt; }
  virtual string mAsString() const override {
    std::string vOut = std::to_string(vValFlt);
    vOut.erase(vOut.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
    if (vOut.back() == '.')
      vOut.push_back('0'); // leave one zero if no decimals
    return vOut;
  }
  virtual bool mAsBool() const override { return vValFlt != 0.0; }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsNumeric() const override { return true; }
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override {
    if (static_cast<double>(vValInt) == vValFlt) {
      return vVal->mUniqueHash();
    } else {
      return hasher(typeid(*this)).mUpdate(&vValInt, sizeof(vValInt)).mUpdate(&vValFlt, sizeof(vValFlt));
    }
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return pOther.mIsNumeric() && vValInt == pOther.vValInt && vValFlt == pOther.vValFlt;
  }
  virtual bool mNonequal(const value_t &pOther) const override {
    return !(typeid(*this) == typeid(pOther) && vValFlt == pOther.vValFlt);
  }
};
using value_float = std::shared_ptr<value_float_t>;

struct value_string_t : public value_t {
  value_string_t() { vValStr = string(); }
  value_string_t(const std::string &pV) { vValStr = string(pV); }
  value_string_t(const string &pV) { vValStr = pV; }
  virtual std::string mType() const override { return "String"; }
  virtual string mAsString() const override { return vValStr; }
  virtual std::string mAsRepr() const override {
    std::ostringstream vSs;
    for (const auto &vPart : vValStr.lParts) {
      vSs << (vPart.vIsInput ? "INPUT: " : "TMPL:  ") << vPart.vVal << "\n";
    }
    return vSs.str();
  }
  virtual bool mAsBool() const override { return vValStr.mLength() > 0; }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override {
    const auto cTypeHash = typeid(*this).hash_code();
    auto vHash = hasher();
    vHash.mUpdate(&cTypeHash, sizeof(cTypeHash));
    vValStr.mHashUpdate(vHash);
    return vHash;
  }
  void mMarkInput() { vValStr.mMarkInput(); }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return typeid(*this) == typeid(pOther) && vValStr.mStr() == pOther.vValStr.mStr();
  }
};
using value_string = std::shared_ptr<value_string_t>;

struct value_bool_t : public value_t {
  value vVal;
  value_bool_t(bool pV) {
    vValInt = static_cast<int64_t>(pV);
    vValFlt = static_cast<double>(pV);
    vVal = fMkVal<value_int>(vValInt);
  }
  virtual std::string mType() const override { return "Boolean"; }
  virtual int64_t mAsInt() const override { return vValInt; }
  virtual bool mAsBool() const override { return vValInt; }
  virtual string mAsString() const override { return std::string(vValInt ? "True" : "False"); }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsNumeric() const override { return true; }
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override { return vVal->mUniqueHash(); }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return pOther.mIsNumeric() && vValInt == pOther.vValInt && vValFlt == pOther.vValFlt;
  }
  virtual bool mNonequal(const value_t &pOther) const override {
    return !(typeid(*this) == typeid(pOther) && vValInt == pOther.vValInt);
  }
};
using value_bool = std::shared_ptr<value_bool_t>;

struct value_array_t : public value_t {
  value_array_t() = default;
  value_array_t(value &pV) { lValArr = pV->lValArr; }
  value_array_t(std::vector<value> &&pArr) { lValArr = pArr; }
  value_array_t(const std::vector<value> &pArr) { lValArr = pArr; }
  void mReverse() {
    if (mIsImmutable()) {
      throw std::runtime_error("Attempting to modify immutable type");
    }
    std::reverse(lValArr.begin(), lValArr.end());
  }
  void mPushBack(const value &pVal) {
    if (mIsImmutable()) {
      throw std::runtime_error("Attempting to modify immutable type");
    }
    lValArr.push_back(pVal);
  }
  void mPushBack(value &&pVal) {
    if (mIsImmutable()) {
      throw std::runtime_error("Attempting to modify immutable type");
    }
    lValArr.push_back(std::move(pVal));
  }
  value mPopAt(int64_t pIndex) {
    if (mIsImmutable()) {
      throw std::runtime_error("Attempting to modify immutable type");
    }
    if (pIndex < 0) {
      pIndex = static_cast<int64_t>(lValArr.size()) + pIndex;
    }
    if (pIndex < 0 || pIndex >= static_cast<int64_t>(lValArr.size())) {
      throw std::runtime_error("Index " + std::to_string(pIndex) + " out of bounds for array of size " +
                               std::to_string(lValArr.size()));
    }
    value vVal = lValArr.at(static_cast<size_t>(pIndex));
    lValArr.erase(lValArr.begin() + pIndex);
    return vVal;
  }
  virtual std::string mType() const override { return "Array"; }
  virtual bool mIsImmutable() const override { return false; }
  virtual const std::vector<value> &mAsArray() const override { return lValArr; }
  virtual string mAsString() const override {
    const bool cImmutable = mIsImmutable();
    std::ostringstream vSs;
    vSs << (cImmutable ? "(" : "[");
    for (size_t vI = 0; vI < lValArr.size(); vI++) {
      if (vI > 0)
        vSs << ", ";
      value vVal = lValArr.at(vI);
      vSs << fValueToStringRepr(vVal);
    }
    if (cImmutable && lValArr.size() == 1) {
      vSs << ",";
    }
    vSs << (cImmutable ? ")" : "]");
    return vSs.str();
  }
  virtual bool mAsBool() const override { return !lValArr.empty(); }
  virtual value &mAt(int64_t pIndex, value &pDefaultVal) override {
    if (pIndex < 0) {
      pIndex += lValArr.size();
    }
    if (pIndex < 0 || static_cast<size_t>(pIndex) >= lValArr.size()) {
      return pDefaultVal;
    }
    return lValArr[pIndex];
  }
  virtual value &mAt(int64_t pIndex) override {
    if (pIndex < 0) {
      pIndex += lValArr.size();
    }
    if (pIndex < 0 || static_cast<size_t>(pIndex) >= lValArr.size()) {
      throw std::runtime_error("Index " + std::to_string(pIndex) + " out of bounds for array of size " +
                               std::to_string(lValArr.size()));
    }
    return lValArr[pIndex];
  }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsHashable() const override {
    if (std::all_of(lValArr.begin(), lValArr.end(),
                    [&](auto &pVal) -> bool { return pVal->mIsImmutable() && pVal->mIsHashable(); })) {
      return true;
    }
    return false;
  }
  virtual hasher mUniqueHash() const noexcept override {
    auto vHash = hasher(typeid(*this));
    for (const auto &vVal : lValArr) {
      // must use digest to prevent problems from "concatenation" property of hasher
      // for ex. hash of [ "ab", "c" ] should be different from [ "a", "bc" ]
      const size_t cValHash = vVal->mUniqueHash().mDigest();
      vHash.mUpdate(&cValHash, sizeof(size_t));
    }
    return vHash;
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return typeid(*this) == typeid(pOther) && mIsHashable() && pOther.mIsHashable() &&
           std::equal(lValArr.begin(), lValArr.end(), pOther.lValArr.begin(), pOther.lValArr.end(),
                      value_equivalence());
  }
};
using value_array = std::shared_ptr<value_array_t>;

struct value_tuple_t : public value_array_t {
  value_tuple_t(value &pV) { lValArr = pV->lValArr; }
  value_tuple_t(std::vector<value> &&pArr) { lValArr = pArr; }
  value_tuple_t(const std::vector<value> &pArr) { lValArr = pArr; }
  value_tuple_t(const std::pair<value, value> &pPair) {
    lValArr.push_back(pPair.first);
    lValArr.push_back(pPair.second);
  }
  virtual std::string mType() const override { return "Tuple"; }
  virtual bool mIsImmutable() const override { return true; }
};
using value_tuple = std::shared_ptr<value_tuple_t>;

struct value_object_t : public value_t {
  std::unordered_map<value, value, value_hasher, value_equivalence> dUnordered;
  bool vHasBuiltins = true; // context and loop objects do not have builtins
  value_object_t() = default;
  value_object_t(value &pV) {
    lValObj = pV->lValObj;
    for (const auto &vPair : lValObj) {
      dUnordered[vPair.first] = vPair.second;
    }
  }
  value_object_t(const std::map<value, value> &pObj) {
    for (const auto &vPair : pObj) {
      mInsert(vPair.first, vPair.second);
    }
  }
  value_object_t(const std::vector<std::pair<value, value>> &pObj) {
    for (const auto &vPair : pObj) {
      mInsert(vPair.first, vPair.second);
    }
  }
  void insert(const std::string &pKey, const value &pVal) { mInsert(fMkVal<value_string>(pKey), pVal); }
  virtual std::string mType() const override { return "Object"; }
  virtual bool mIsImmutable() const override { return false; }
  virtual const std::vector<std::pair<value, value>> &mAsOrderedObject() const override { return lValObj; }
  virtual string mAsString() const override {
    std::ostringstream vSs;
    vSs << "{";
    for (size_t vI = 0; vI < lValObj.size(); vI++) {
      if (vI > 0)
        vSs << ", ";
      auto &[vKey, vVal] = lValObj.at(vI);
      vSs << fValueToStringRepr(vKey) << ": " << fValueToStringRepr(vVal);
    }
    vSs << "}";
    return vSs.str();
  }
  virtual bool mAsBool() const override { return !dUnordered.empty(); }
  virtual bool mHasKey(const value &pKey) override {
    if (!pKey->mIsImmutable() || !pKey->mIsHashable()) {
      throw std::runtime_error("Object key of unhashable type: " + pKey->mType());
    }
    return dUnordered.find(pKey) != dUnordered.end();
  }
  virtual void mInsert(const value &pKey, const value &pVal) override {
    bool vReplaced = false;
    if (mIsImmutable()) {
      throw std::runtime_error("Attempting to modify immutable type");
    }
    if (mHasKey(pKey)) {
      // if key exists, replace value in ordered list instead of appending
      for (auto &vPair : lValObj) {
        if (*(vPair.first) == *pKey) {
          vPair.second = pVal;
          vReplaced = true;
          break;
        }
      }
    }
    dUnordered[pKey] = pVal;
    if (!vReplaced) {
      lValObj.push_back({pKey, pVal});
    }
  }
  virtual value &mAt(const value &pKey, value &pDefaultVal) override {
    if (!mHasKey(pKey)) {
      return pDefaultVal;
    }
    return dUnordered.at(pKey);
  }
  virtual value &mAt(const value &pKey) override {
    if (!mHasKey(pKey)) {
      throw std::runtime_error("Key '" + pKey->mAsString().mStr() + "' not found in value of type " + mType());
    }
    return dUnordered.at(pKey);
  }
  virtual value &mAt(const std::string &pKey, value &pDefaultVal) override {
    value vKeyVal = fMkVal<value_string>(pKey);
    return mAt(vKeyVal, pDefaultVal);
  }
  virtual value &mAt(const std::string &pKey) override {
    value vKeyVal = fMkVal<value_string>(pKey);
    return mAt(vKeyVal);
  }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsHashable() const override {
    if (std::all_of(lValObj.begin(), lValObj.end(), [&](auto &pPair) -> bool {
          const auto &vVal = pPair.second;
          return vVal->mIsImmutable() && vVal->mIsHashable();
        })) {
      return true;
    }
    return false;
  }
  virtual hasher mUniqueHash() const noexcept override {
    auto vHash = hasher(typeid(*this));
    for (const auto &[vKey, vVal] : lValObj) {
      // must use digest to prevent problems from "concatenation" property of hasher
      // for ex. hash of key="ab", value="c" should be different from key="a", value="bc"
      const size_t cKeyHash = vKey->mUniqueHash().mDigest();
      const size_t cValHash = vVal->mUniqueHash().mDigest();
      vHash.mUpdate(&cKeyHash, sizeof(cKeyHash));
      vHash.mUpdate(&cValHash, sizeof(cValHash));
    }
    return vHash;
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    return typeid(*this) == typeid(pOther) && mIsHashable() && pOther.mIsHashable() &&
           std::equal(lValObj.begin(), lValObj.end(), pOther.lValObj.begin(), pOther.lValObj.end(),
                      value_equivalence());
  }
};
using value_object = std::shared_ptr<value_object_t>;

//
// none and undefined types
//

struct value_none_t : public value_t {
  virtual std::string mType() const override { return "None"; }
  virtual bool mIsNone() const override { return true; }
  virtual bool mAsBool() const override { return false; }
  virtual string mAsString() const override { return string(mType()); }
  virtual std::string mAsRepr() const override { return mType(); }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override { return hasher(typeid(*this)); }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override { return typeid(*this) == typeid(pOther); }
};
using value_none = std::shared_ptr<value_none_t>;

struct value_undefined_t : public value_t {
  std::string vHint; // for debugging, to indicate where undefined came from
  value_undefined_t(const std::string &pH = "") : vHint(pH) {}
  virtual std::string mType() const override {
    return vHint.empty() ? "Undefined" : "Undefined (hint: '" + vHint + "')";
  }
  virtual bool mIsUndefined() const override { return true; }
  virtual bool mAsBool() const override { return false; }
  virtual std::string mAsRepr() const override { return mType(); }
  virtual const func_builtins &mGetBuiltins() const override;
  virtual hasher mUniqueHash() const noexcept override { return hasher(typeid(*this)); }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override { return mIsUndefined() == pOther.mIsUndefined(); }
};
using value_undefined = std::shared_ptr<value_undefined_t>;

//
// function type
//

struct func_args {
public:
  std::string vFuncName; // for error messages
  context &vCtx;
  func_args(context &pCtx) : vCtx(pCtx) {}
  value mGetKwarg(const std::string &pKey, value pDefaultVal) const;
  value mGetKwargOrPos(const std::string &pKey, size_t pPos) const;
  value mGetPos(size_t pPos) const;
  value mGetPos(size_t pPos, value pDefaultVal) const;
  const std::vector<value> &mGetArgs() const;
  size_t mCount() const { return lArgs.size(); }
  void mPushBack(const value &pVal);
  void mPushFront(const value &pVal);
  void mEnsureCount(size_t pMin, size_t pMax = 999) const {
    size_t vN = lArgs.size();
    if (vN < pMin || vN > pMax) {
      throw std::runtime_error("Function '" + vFuncName + "' expected between " + std::to_string(pMin) + " and " +
                               std::to_string(pMax) + " arguments, got " + std::to_string(vN));
    }
  }
  template <typename T> void mEnsureVal(const value &pPtr) const {
    if (!fIsVal<T>(pPtr)) {
      throw std::runtime_error("Function '" + vFuncName + "' expected value of type " + std::string(typeid(T).name()) +
                               ", got " + pPtr->mType());
    }
  }
  void mEnsureCount(bool pRequire0, bool pRequire1, bool pRequire2, bool pRequire3) const {
    static auto vBoolToInt = [](bool pB) { return pB ? 1 : 0; };
    size_t vRequired = vBoolToInt(pRequire0) + vBoolToInt(pRequire1) + vBoolToInt(pRequire2) + vBoolToInt(pRequire3);
    mEnsureCount(vRequired);
  }
  template <typename T0> void mEnsureVals(bool pRequired0 = true) const {
    mEnsureCount(pRequired0, false, false, false);
    if (pRequired0 && lArgs.size() > 0)
      mEnsureVal<T0>(lArgs[0]);
  }
  template <typename T0, typename T1> void mEnsureVals(bool pRequired0 = true, bool pRequired1 = true) const {
    mEnsureCount(pRequired0, pRequired1, false, false);
    if (pRequired0 && lArgs.size() > 0)
      mEnsureVal<T0>(lArgs[0]);
    if (pRequired1 && lArgs.size() > 1)
      mEnsureVal<T1>(lArgs[1]);
  }
  template <typename T0, typename T1, typename T2>
  void mEnsureVals(bool pRequired0 = true, bool pRequired1 = true, bool pRequired2 = true) const {
    mEnsureCount(pRequired0, pRequired1, pRequired2, false);
    if (pRequired0 && lArgs.size() > 0)
      mEnsureVal<T0>(lArgs[0]);
    if (pRequired1 && lArgs.size() > 1)
      mEnsureVal<T1>(lArgs[1]);
    if (pRequired2 && lArgs.size() > 2)
      mEnsureVal<T2>(lArgs[2]);
  }
  template <typename T0, typename T1, typename T2, typename T3>
  void mEnsureVals(bool pRequired0 = true, bool pRequired1 = true, bool pRequired2 = true,
                   bool pRequired3 = true) const {
    mEnsureCount(pRequired0, pRequired1, pRequired2, pRequired3);
    if (pRequired0 && lArgs.size() > 0)
      mEnsureVal<T0>(lArgs[0]);
    if (pRequired1 && lArgs.size() > 1)
      mEnsureVal<T1>(lArgs[1]);
    if (pRequired2 && lArgs.size() > 2)
      mEnsureVal<T2>(lArgs[2]);
    if (pRequired3 && lArgs.size() > 3)
      mEnsureVal<T3>(lArgs[3]);
  }

private:
  std::vector<value> lArgs;
};

struct value_func_t : public value_t {
  std::string vName;
  value vArg0; // bound "this" argument, if any
  value_func_t(const std::string &pName, const func_handler &pFunc) : vName(pName) { vValFunc = pFunc; }
  value_func_t(const std::string &pName, const func_handler &pFunc, const value &pArgThis)
      : vName(pName), vArg0(pArgThis) {
    vValFunc = pFunc;
  }
  virtual value mInvoke(const func_args &pArgs) const override {
    func_args vNewArgs(pArgs); // copy
    vNewArgs.vFuncName = vName;
    if (vArg0) {
      vNewArgs.mPushFront(vArg0);
    }
    return vValFunc(vNewArgs);
  }
  virtual std::string mType() const override { return "Function"; }
  virtual std::string mAsRepr() const override {
    return mType() + "<" + vName + ">(" + (vArg0 ? vArg0->mAsRepr() : "") + ")";
  }
  virtual bool mIsHashable() const override { return false; }
  virtual hasher mUniqueHash() const noexcept override {
    // Note: this is unused for now, we don't support function as object keys
    // use function pointer as unique identifier
    const auto cTarget = vValFunc.target<func_hptr>();
    return hasher(typeid(*this)).mUpdate(&cTarget, sizeof(cTarget));
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    // Note: this is unused for now, we don't support function as object keys
    // compare function pointers
    // (vValFunc == pOther.vValFunc does not work as std::function::operator== is only used for nullptr check)
    const auto cTargetThis = this->vValFunc.target<func_hptr>();
    const auto cTargetOther = pOther.vValFunc.target<func_hptr>();
    return typeid(*this) == typeid(pOther) && cTargetThis == cTargetOther;
  }
};
using value_func = std::shared_ptr<value_func_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
  std::string vKey;
  value vVal;
  value_kwarg_t(const std::string &pK, const value &pV) : vKey(pK), vVal(pV) {}
  virtual std::string mType() const override { return "KwArg"; }
  virtual std::string mAsRepr() const override { return mType(); }
  virtual bool mIsHashable() const override { return true; }
  virtual hasher mUniqueHash() const noexcept override {
    const auto cTypeHash = typeid(*this).hash_code();
    auto vHash = vVal->mUniqueHash();
    vHash.mUpdate(&cTypeHash, sizeof(cTypeHash)).mUpdate(vKey.data(), vKey.size());
    return vHash;
  }

protected:
  virtual bool mEquivalent(const value_t &pOther) const override {
    const value_kwarg_t &vOtherVal = static_cast<const value_kwarg_t &>(pOther);
    return typeid(*this) == typeid(pOther) && vKey == vOtherVal.vKey && vVal == vOtherVal.vVal;
  }
};
using value_kwarg = std::shared_ptr<value_kwarg_t>;

} // namespace jinja
