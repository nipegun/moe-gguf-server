#include "mgs-model-loader.h"

#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"
#include "mgs-hparams.h"
#include "mgs.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <future>
#include <regex>

static const size_t cKiB = 1024;
static const size_t cMiB = 1024 * cKiB;
static const size_t cGiB = 1024 * cMiB;

const char *mgs_file_version_name(mgs_fver pVersion) {
  switch (pVersion) {
  case cGgufFileVersionV3:
    return "GGUF V3 (latest)";
  }

  return "unknown";
}

#define MGS_FTYPE_PREFIX "(guessed) "

const char *mgs_ftype_name(mgs_ftype pFtype) {
  static constexpr size_t cGuessedPrefixLen = sizeof(MGS_FTYPE_PREFIX) - 1;
  const char *vName;
  switch ((enum mgs_ftype)(pFtype & ~MGS_FTYPE_GUESSED)) {
  case MGS_FTYPE_ALL_F32:
    vName = MGS_FTYPE_PREFIX "all F32";
    break;
  case MGS_FTYPE_MOSTLY_F16:
    vName = MGS_FTYPE_PREFIX "F16";
    break;
  case MGS_FTYPE_MOSTLY_BF16:
    vName = MGS_FTYPE_PREFIX "BF16";
    break;
  case MGS_FTYPE_MOSTLY_Q1_0:
    vName = MGS_FTYPE_PREFIX "Q1_0";
    break;
  case MGS_FTYPE_MOSTLY_Q2_0:
    vName = MGS_FTYPE_PREFIX "Q2_0";
    break;
  case MGS_FTYPE_MOSTLY_Q4_0:
    vName = MGS_FTYPE_PREFIX "Q4_0";
    break;
  case MGS_FTYPE_MOSTLY_Q4_1:
    vName = MGS_FTYPE_PREFIX "Q4_1";
    break;
  case MGS_FTYPE_MOSTLY_Q5_0:
    vName = MGS_FTYPE_PREFIX "Q5_0";
    break;
  case MGS_FTYPE_MOSTLY_Q5_1:
    vName = MGS_FTYPE_PREFIX "Q5_1";
    break;
  case MGS_FTYPE_MOSTLY_Q8_0:
    vName = MGS_FTYPE_PREFIX "Q8_0";
    break;
  case MGS_FTYPE_MOSTLY_MXFP4_MOE:
    vName = MGS_FTYPE_PREFIX "MXFP4 MoE";
    break;
  case MGS_FTYPE_MOSTLY_NVFP4:
    vName = MGS_FTYPE_PREFIX "NVFP4";
    break;
  case MGS_FTYPE_MOSTLY_Q2_K:
    vName = MGS_FTYPE_PREFIX "Q2_K - Medium";
    break;
  case MGS_FTYPE_MOSTLY_Q2_K_S:
    vName = MGS_FTYPE_PREFIX "Q2_K - Small";
    break;
  case MGS_FTYPE_MOSTLY_Q3_K_S:
    vName = MGS_FTYPE_PREFIX "Q3_K - Small";
    break;
  case MGS_FTYPE_MOSTLY_Q3_K_M:
    vName = MGS_FTYPE_PREFIX "Q3_K - Medium";
    break;
  case MGS_FTYPE_MOSTLY_Q3_K_L:
    vName = MGS_FTYPE_PREFIX "Q3_K - Large";
    break;
  case MGS_FTYPE_MOSTLY_Q4_K_S:
    vName = MGS_FTYPE_PREFIX "Q4_K - Small";
    break;
  case MGS_FTYPE_MOSTLY_Q4_K_M:
    vName = MGS_FTYPE_PREFIX "Q4_K - Medium";
    break;
  case MGS_FTYPE_MOSTLY_Q5_K_S:
    vName = MGS_FTYPE_PREFIX "Q5_K - Small";
    break;
  case MGS_FTYPE_MOSTLY_Q5_K_M:
    vName = MGS_FTYPE_PREFIX "Q5_K - Medium";
    break;
  case MGS_FTYPE_MOSTLY_Q6_K:
    vName = MGS_FTYPE_PREFIX "Q6_K";
    break;
  case MGS_FTYPE_MOSTLY_TQ1_0:
    vName = MGS_FTYPE_PREFIX "TQ1_0 - 1.69 bpw ternary";
    break;
  case MGS_FTYPE_MOSTLY_TQ2_0:
    vName = MGS_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary";
    break;
  case MGS_FTYPE_MOSTLY_IQ2_XXS:
    vName = MGS_FTYPE_PREFIX "IQ2_XXS - 2.0625 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ2_XS:
    vName = MGS_FTYPE_PREFIX "IQ2_XS - 2.3125 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ2_S:
    vName = MGS_FTYPE_PREFIX "IQ2_S - 2.5 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ2_M:
    vName = MGS_FTYPE_PREFIX "IQ2_M - 2.7 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ3_XS:
    vName = MGS_FTYPE_PREFIX "IQ3_XS - 3.3 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ3_XXS:
    vName = MGS_FTYPE_PREFIX "IQ3_XXS - 3.0625 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ1_S:
    vName = MGS_FTYPE_PREFIX "IQ1_S - 1.5625 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ1_M:
    vName = MGS_FTYPE_PREFIX "IQ1_M - 1.75 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ4_NL:
    vName = MGS_FTYPE_PREFIX "IQ4_NL - 4.5 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ4_XS:
    vName = MGS_FTYPE_PREFIX "IQ4_XS - 4.25 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ3_S:
    vName = MGS_FTYPE_PREFIX "IQ3_S - 3.4375 bpw";
    break;
  case MGS_FTYPE_MOSTLY_IQ3_M:
    vName = MGS_FTYPE_PREFIX "IQ3_S mix - 3.66 bpw";
    break;
  default:
    vName = MGS_FTYPE_PREFIX "unknown, may not work";
    break;
  }
  return (pFtype & MGS_FTYPE_GUESSED) ? vName : vName + cGuessedPrefixLen;
}

#undef MGS_FTYPE_PREFIX

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> fMgsGetListSplits(const std::string &pPath, const int pIdx, const int pNSplit) {
  std::vector<std::string> lPaths;
  std::string vSplitPrefix;
  std::vector<char> lBuf(mgs_path_max(), 0);

  {
    int vRet = mgs_split_prefix(lBuf.data(), lBuf.size(), pPath.c_str(), pIdx, pNSplit);
    if (!vRet) {
      throw std::runtime_error(fFormat("invalid split file name: %s", pPath.c_str()));
    }
    vSplitPrefix = std::string(lBuf.data(), vRet);
  }

  if (vSplitPrefix.empty()) {
    throw std::runtime_error(fFormat("invalid split file: %s", pPath.c_str()));
  }

  for (int vIdx = 0; vIdx < pNSplit; ++vIdx) {
    int vRet = mgs_split_path(lBuf.data(), lBuf.size(), vSplitPrefix.c_str(), vIdx, pNSplit);
    lPaths.push_back(std::string(lBuf.data(), vRet));
  }

  return lPaths;
}

namespace GGUFMeta {
template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)> struct GKV_Base_Type {
  static constexpr gguf_type cGt = gt_;

  static T mGetter(const gguf_context *pCtx, const int pKid) { return gfun(pCtx, pKid); }
};

template <typename T> struct GKV_Base;

template <> struct GKV_Base<bool> : GKV_Base_Type<bool, GGUF_TYPE_BOOL, gguf_get_val_bool> {};
template <> struct GKV_Base<uint8_t> : GKV_Base_Type<uint8_t, GGUF_TYPE_UINT8, gguf_get_val_u8> {};
template <> struct GKV_Base<uint16_t> : GKV_Base_Type<uint16_t, GGUF_TYPE_UINT16, gguf_get_val_u16> {};
template <> struct GKV_Base<uint32_t> : GKV_Base_Type<uint32_t, GGUF_TYPE_UINT32, gguf_get_val_u32> {};
template <> struct GKV_Base<uint64_t> : GKV_Base_Type<uint64_t, GGUF_TYPE_UINT64, gguf_get_val_u64> {};
template <> struct GKV_Base<int8_t> : GKV_Base_Type<int8_t, GGUF_TYPE_INT8, gguf_get_val_i8> {};
template <> struct GKV_Base<int16_t> : GKV_Base_Type<int16_t, GGUF_TYPE_INT16, gguf_get_val_i16> {};
template <> struct GKV_Base<int32_t> : GKV_Base_Type<int32_t, GGUF_TYPE_INT32, gguf_get_val_i32> {};
template <> struct GKV_Base<int64_t> : GKV_Base_Type<int64_t, GGUF_TYPE_INT64, gguf_get_val_i64> {};
template <> struct GKV_Base<float> : GKV_Base_Type<float, GGUF_TYPE_FLOAT32, gguf_get_val_f32> {};
template <> struct GKV_Base<double> : GKV_Base_Type<double, GGUF_TYPE_FLOAT64, gguf_get_val_f64> {};
template <> struct GKV_Base<const char *> : GKV_Base_Type<const char *, GGUF_TYPE_STRING, gguf_get_val_str> {};

template <> struct GKV_Base<std::string> {
  static constexpr gguf_type cGt = GGUF_TYPE_STRING;

  static std::string mGetter(const gguf_context *pCtx, const int pKid) { return gguf_get_val_str(pCtx, pKid); }
};

struct ArrayInfo {
  const gguf_type cGt;
  const size_t cLength;
  const void *vData;
};

template <> struct GKV_Base<ArrayInfo> {
public:
  static constexpr gguf_type cGt = GGUF_TYPE_ARRAY;
  static ArrayInfo mGetter(const gguf_context *pCtx, const int pK) {
    const enum gguf_type cArrType = gguf_get_arr_type(pCtx, pK);
    return ArrayInfo{
        cArrType,
        gguf_get_arr_n(pCtx, pK),
        cArrType == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(pCtx, pK),
    };
  }
};

template <typename T> class GKV : public GKV_Base<T> {
  GKV() = delete;

public:
  static T mGetKv(const gguf_context *pCtx, const int pK) {
    const enum gguf_type cKt = gguf_get_kv_type(pCtx, pK);

    if (cKt != GKV::cGt) {
      throw std::runtime_error(fFormat("key %s has wrong type %s but expected type %s", gguf_get_key(pCtx, pK),
                                       gguf_type_name(cKt), gguf_type_name(GKV::cGt)));
    }
    return GKV::mGetter(pCtx, pK);
  }

  static const char *mOverrideTypeToStr(const mgs_model_kv_override_type pTy) {
    switch (pTy) {
    case MGS_KV_OVERRIDE_TYPE_BOOL:
      return "bool";
    case MGS_KV_OVERRIDE_TYPE_INT:
      return "int";
    case MGS_KV_OVERRIDE_TYPE_FLOAT:
      return "float";
    case MGS_KV_OVERRIDE_TYPE_STR:
      return "str";
    }
    return "unknown";
  }

  static bool mValidateOverride(const mgs_model_kv_override_type pExpectedType,
                                const struct mgs_model_kv_override *pOvrd) {
    if (!pOvrd) {
      return false;
    }
    if (pOvrd->tag == pExpectedType) {
      MGS_LOG_INFO("%s: Using metadata override (%5s) '%s' = ", __func__, mOverrideTypeToStr(pOvrd->tag), pOvrd->key);
      switch (pOvrd->tag) {
      case MGS_KV_OVERRIDE_TYPE_BOOL: {
        MGS_LOG_INFO("%s\n", pOvrd->val_bool ? "true" : "false");
      } break;
      case MGS_KV_OVERRIDE_TYPE_INT: {
        MGS_LOG_INFO("%" PRId64 "\n", pOvrd->val_i64);
      } break;
      case MGS_KV_OVERRIDE_TYPE_FLOAT: {
        MGS_LOG_INFO("%.6f\n", pOvrd->val_f64);
      } break;
      case MGS_KV_OVERRIDE_TYPE_STR: {
        MGS_LOG_INFO("%s\n", pOvrd->val_str);
      } break;
      default:
        // Shouldn't be possible to end up here, but just in case...
        throw std::runtime_error(fFormat("Unsupported attempt to override %s type for metadata key %s\n",
                                         mOverrideTypeToStr(pOvrd->tag), pOvrd->key));
      }
      return true;
    }
    MGS_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n", __func__, pOvrd->key,
                 mOverrideTypeToStr(pExpectedType), mOverrideTypeToStr(pOvrd->tag));
    return false;
  }

  template <typename OT>
  static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
  try_override(OT &pTarget, const struct mgs_model_kv_override *pOvrd) {
    if (mValidateOverride(MGS_KV_OVERRIDE_TYPE_BOOL, pOvrd)) {
      pTarget = pOvrd->val_bool;
      return true;
    }
    return false;
  }

  template <typename OT>
  static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
  try_override(OT &pTarget, const struct mgs_model_kv_override *pOvrd) {
    if (mValidateOverride(MGS_KV_OVERRIDE_TYPE_INT, pOvrd)) {
      pTarget = pOvrd->val_i64;
      return true;
    }
    return false;
  }

  template <typename OT>
  static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
  try_override(T &pTarget, const struct mgs_model_kv_override *pOvrd) {
    if (mValidateOverride(MGS_KV_OVERRIDE_TYPE_FLOAT, pOvrd)) {
      pTarget = pOvrd->val_f64;
      return true;
    }
    return false;
  }

  template <typename OT>
  static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
  try_override(T &pTarget, const struct mgs_model_kv_override *pOvrd) {
    if (mValidateOverride(MGS_KV_OVERRIDE_TYPE_STR, pOvrd)) {
      pTarget = pOvrd->val_str;
      return true;
    }
    return false;
  }

  static bool mSet(const gguf_context *pCtx, const int pK, T &pTarget,
                   const struct mgs_model_kv_override *pOvrd = nullptr) {
    if (try_override<T>(pTarget, pOvrd)) {
      return true;
    }
    if (pK < 0) {
      return false;
    }
    pTarget = mGetKv(pCtx, pK);
    return true;
  }

  static bool mSet(const gguf_context *pCtx, const char *pKey, T &pTarget,
                   const struct mgs_model_kv_override *pOvrd = nullptr) {
    return mSet(pCtx, gguf_find_key(pCtx, pKey), pTarget, pOvrd);
  }

  static bool mSet(const gguf_context *pCtx, const std::string &pKey, T &pTarget,
                   const struct mgs_model_kv_override *pOvrd = nullptr) {
    return mSet(pCtx, pKey.c_str(), pTarget, pOvrd);
  }
};
} // namespace GGUFMeta

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type mgs_model_loader::mGetArrN(const std::string &pKey,
                                                                                           T &pResult, bool pRequired) {
  const int cKid = gguf_find_key(vMetadata, pKey.c_str());

  if (cKid < 0) {
    if (pRequired) {
      throw std::runtime_error(fFormat("key not found in model: %s", pKey.c_str()));
    }
    return false;
  }

  struct GGUFMeta::ArrayInfo vArrInfo = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::mGetKv(vMetadata, cKid);

  pResult = vArrInfo.cLength;
  return true;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type mgs_model_loader::mGetArrN(enum llm_kv pKid, T &pResult,
                                                                                           bool pRequired) {
  return mGetArrN(vLlmKv(pKid), pResult, pRequired);
}

template bool mgs_model_loader::mGetArrN(enum llm_kv kid, uint32_t &result, bool required);
template std::enable_if<std::is_integral<uint32_t>::value, bool>::type
mgs_model_loader::mGetArrN<uint32_t>(const std::string &key, uint32_t &result, bool required);

template <typename T> bool mgs_model_loader::mGetArr(const std::string &pKey, std::vector<T> &pResult, bool pRequired) {
  const gguf_context *vCtx = vMetadata;
  const int cKid = gguf_find_key(vCtx, pKey.c_str());

  if (cKid < 0 || gguf_get_kv_type(vCtx, cKid) != GGUF_TYPE_ARRAY) {
    if (pRequired) {
      throw std::runtime_error(fFormat("array key not found in model: %s", pKey.c_str()));
    }
    return false;
  }

  struct GGUFMeta::ArrayInfo vArrInfo = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::mGetKv(vCtx, cKid);

  switch (vArrInfo.cGt) {
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
    GGML_ASSERT((std::is_same<T, int32_t>::value) || (std::is_same<T, uint32_t>::value));
    break;
  case GGUF_TYPE_UINT64:
    GGML_ASSERT((std::is_same<T, uint64_t>::value));
    break;
  case GGUF_TYPE_FLOAT32:
    GGML_ASSERT((std::is_same<T, float>::value));
    break;
  case GGUF_TYPE_STRING:
    GGML_ASSERT((std::is_same<T, std::string>::value));
    break;
  default:
    throw std::runtime_error(fFormat("%s is not a string/float32/uint32/int32/uint64 array", pKey.c_str()));
  }

  if constexpr (std::is_same<T, std::string>::value) {
    const size_t cNItems = gguf_get_arr_n(vCtx, cKid);
    pResult.clear();

    for (size_t vI = 0; vI < cNItems; vI++) {
      const T cValue = gguf_get_arr_str(vCtx, cKid, vI);
      pResult.emplace_back(cValue);
    }
  } else {
    pResult.resize(vArrInfo.cLength);
    pResult.assign((const T *)vArrInfo.vData, (const T *)vArrInfo.vData + vArrInfo.cLength);
  }

  return true;
}

template <typename T, size_t N_MAX>
bool mgs_model_loader::mGetArr(const std::string &pKey, std::array<T, N_MAX> &pResult, bool pRequired) {
  const gguf_context *vCtx = vMetadata;
  const int cKid = gguf_find_key(vCtx, pKey.c_str());

  if (cKid < 0 || gguf_get_kv_type(vCtx, cKid) != GGUF_TYPE_ARRAY) {
    if (pRequired) {
      throw std::runtime_error(fFormat("array key not found in model: %s", pKey.c_str()));
    }
    return false;
  }

  struct GGUFMeta::ArrayInfo vArrInfo = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::mGetKv(vCtx, cKid);

  switch (vArrInfo.cGt) {
  case GGUF_TYPE_BOOL:
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
    GGML_ASSERT((std::is_same<T, int32_t>::value) || (std::is_same<T, uint32_t>::value));
    break;
  case GGUF_TYPE_UINT64:
    GGML_ASSERT((std::is_same<T, uint64_t>::value));
    break;
  case GGUF_TYPE_FLOAT32:
    GGML_ASSERT((std::is_same<T, float>::value));
    break;
  case GGUF_TYPE_STRING:
    GGML_ASSERT((std::is_same<T, std::string>::value));
    break;
  default:
    throw std::runtime_error(fFormat("%s is not a string/float32/uint32/int32/uint64 array", pKey.c_str()));
  }

  if (vArrInfo.cLength > N_MAX) {
    throw std::runtime_error(fFormat("array length %u for key %s exceeds max %u", (uint32_t)vArrInfo.cLength,
                                     pKey.c_str(), (uint32_t)N_MAX));
  }

  if constexpr (std::is_same<T, std::string>::value) {
    const size_t cNItems = gguf_get_arr_n(vCtx, cKid);

    for (size_t vI = 0; vI < cNItems; vI++) {
      const T cValue = gguf_get_arr_str(vCtx, cKid, vI);
      pResult[vI] = cValue;
    }
  } else {
    if (vArrInfo.cGt == GGUF_TYPE_BOOL) {
      const int8_t *vValues = (const int8_t *)vArrInfo.vData;
      std::transform(vValues, vValues + vArrInfo.cLength, pResult.begin(),
                     [](int8_t pX) { return static_cast<T>(pX != 0); });
    } else {
      std::copy((const T *)vArrInfo.vData, (const T *)vArrInfo.vData + vArrInfo.cLength, pResult.begin());
    }
  }

  return true;
}

template <typename T> bool mgs_model_loader::mGetArr(enum llm_kv pKid, T &pResult, bool pRequired) {
  return mGetArr(vLlmKv(pKid), pResult, pRequired);
}

template bool mgs_model_loader::mGetArr<std::vector<std::string>>(enum llm_kv kid, std::vector<std::string> &result,
                                                                  bool required);
template bool mgs_model_loader::mGetArr<std::array<int32_t, 512>>(enum llm_kv kid, std::array<int32_t, 512> &result,
                                                                  bool required);
template bool mgs_model_loader::mGetArr<std::vector<int32_t>>(enum llm_kv kid, std::vector<int32_t> &result,
                                                              bool required);
template bool mgs_model_loader::mGetArr<std::array<uint32_t, MGS_MAX_LAYERS>>(
    enum llm_kv kid, std::array<uint32_t, MGS_MAX_LAYERS> &result, bool required);
template bool mgs_model_loader::mGetArr<std::vector<uint32_t>>(enum llm_kv kid, std::vector<uint32_t> &result,
                                                               bool required);
template bool mgs_model_loader::mGetArr<std::array<uint64_t, MGS_MAX_PLE_NGRAM>>(
    enum llm_kv kid, std::array<uint64_t, MGS_MAX_PLE_NGRAM> &result, bool required);
template bool mgs_model_loader::mGetArr<std::array<uint64_t, MGS_MAX_PLE_HEADS>>(
    enum llm_kv kid, std::array<uint64_t, MGS_MAX_PLE_HEADS> &result, bool required);

template <typename T> bool mgs_model_loader::mGetKey(const std::string &pKey, T &pResult, bool pRequired) {
  auto vIt = dKvOverrides.find(pKey);

  const struct mgs_model_kv_override *vOverride = vIt != dKvOverrides.end() ? &vIt->second : nullptr;

  const bool cFound = GGUFMeta::GKV<T>::mSet(vMetadata, pKey, pResult, vOverride);

  if (pRequired && !cFound) {
    throw std::runtime_error(fFormat("key not found in model: %s", pKey.c_str()));
  }

  return cFound;
}

template <typename T> bool mgs_model_loader::mGetKey(enum llm_kv pKid, T &pResult, bool pRequired) {
  return mGetKey(vLlmKv(pKid), pResult, pRequired);
}

template bool mgs_model_loader::mGetKey<bool>(enum llm_kv kid, bool &result, bool required);
template bool mgs_model_loader::mGetKey<float>(enum llm_kv kid, float &result, bool required);
template bool mgs_model_loader::mGetKey<uint32_t>(enum llm_kv kid, uint32_t &result, bool required);
template bool mgs_model_loader::mGetKey<std::string>(enum llm_kv kid, std::string &result, bool required);

template <> bool mgs_model_loader::mGetKey(enum llm_kv pKid, enum mgs_pooling_type &pResult, bool pRequired) {
  uint32_t vTmp;
  const bool cFound = mGetKey(pKid, vTmp, pRequired);
  if (cFound) {
    pResult = (enum mgs_pooling_type)vTmp;
  } else {
    pResult = MGS_POOLING_TYPE_UNSPECIFIED;
  }
  return cFound;
}

// get array of n <= N_MAX elements, or a single element repeated n times
template <typename T, size_t N_MAX>
bool mgs_model_loader::mGetKeyOrArr(const std::string &pKey, std::array<T, N_MAX> &pResult, uint32_t pN,
                                    bool pRequired) {
  const int cKid = gguf_find_key(vMetadata, pKey.c_str());

  if (cKid < 0) {
    if (pRequired) {
      throw std::runtime_error(fFormat("key not found in model: %s", pKey.c_str()));
    }
    return false;
  }

  if (pN > N_MAX) {
    throw std::runtime_error(fFormat("n > N_MAX: %u > %u for key %s", pN, (uint32_t)N_MAX, pKey.c_str()));
  }

  if (gguf_get_kv_type(vMetadata, cKid) == GGUF_TYPE_ARRAY) {
    struct GGUFMeta::ArrayInfo vArrInfo = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::mGetKv(vMetadata, cKid);

    if (pN != vArrInfo.cLength) {
      throw std::runtime_error(
          fFormat("key %s has wrong array length; expected %u, got %u", pKey.c_str(), pN, (uint32_t)vArrInfo.cLength));
    }

    return mGetArr(pKey, pResult, pRequired);
  }

  T vValue;

  bool vOk = mGetKey(pKey, vValue, pRequired);
  if (!vOk) {
    return false;
  }

  for (uint32_t vI = 0; vI < pN; vI++) {
    pResult[vI] = vValue;
  }

  return true;
}

template <typename T> bool mgs_model_loader::mGetKeyOrArr(enum llm_kv pKid, T &pResult, uint32_t pN, bool pRequired) {
  return mGetKeyOrArr(vLlmKv(pKid), pResult, pN, pRequired);
}

bool mgs_model_loader::mGetKeyOrArr(enum llm_kv pKid, uint32_t &pResult, bool pRequired) {
  const std::string cKey = vLlmKv(pKid);

  const int cId = gguf_find_key(vMetadata, cKey.c_str());

  if (cId < 0) {
    if (pRequired) {
      throw std::runtime_error(fFormat("key not found in model: %s", cKey.c_str()));
    }
    return false;
  }

  // throw and error if type is an array
  if (gguf_get_kv_type(vMetadata, cId) == GGUF_TYPE_ARRAY) {
    if (pRequired) {
      throw std::runtime_error(fFormat("expected scalar, found array for key: %s", cKey.c_str()));
    }
    return false;
  }

  return mGetKey(cKey, pResult, pRequired);
}

// TODO: this is not very clever - figure out something better
template bool mgs_model_loader::mGetKeyOrArr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> &result,
                                                                 uint32_t n, bool required);
template bool mgs_model_loader::mGetKeyOrArr<std::array<uint32_t, 512>>(enum llm_kv kid,
                                                                        std::array<uint32_t, 512> &result, uint32_t n,
                                                                        bool required);
template bool mgs_model_loader::mGetKeyOrArr<std::array<float, 512>>(enum llm_kv kid, std::array<float, 512> &result,
                                                                     uint32_t n, bool required);

mgs_model_loader::mgs_model_loader(struct gguf_context *pMeta, mgs_model_set_tensor_data_t pSetTensorData,
                                   void *pSetTensorDataUd, const std::string &pFname, std::vector<std::string> &pSplits,
                                   FILE *pFile, mgs_load_mode pLoadMode, bool pCheckTensors, bool pNoAlloc,
                                   bool pLoadMtp, const mgs_model_kv_override *pParamOverridesP,
                                   const mgs_model_tensor_buft_override *pParamTensorBuftOverridesP)
    : vMetadata(pMeta), vSetTensorData(pSetTensorData), vSetTensorDataUd(pSetTensorDataUd) {
  int vTrace = 0;
  if (getenv("MGS_TRACE")) {
    vTrace = atoi(getenv("MGS_TRACE"));
  }

  if (pParamOverridesP != nullptr) {
    for (const struct mgs_model_kv_override *vP = pParamOverridesP; vP->key[0] != 0; vP++) {
      dKvOverrides.insert({std::string(vP->key), *vP});
    }
  }

  vTensorBuftOverrides = pParamTensorBuftOverridesP;

  this->vUseMmap = pLoadMode == MGS_LOAD_MODE_MMAP || pLoadMode == MGS_LOAD_MODE_MMAP_MLOCK;
  this->vUseDirectIo = pLoadMode == MGS_LOAD_MODE_DIRECT_IO;

  if (!pFname.empty()) {
    // Load the main GGUF
    struct ggml_context *vCtx = NULL;
    struct gguf_init_params vParams = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &vCtx,
    };

    vMetadataPtr.reset(gguf_init_from_file(pFname.c_str(), vParams));
    vMetadata = vMetadataPtr.get();
    if (vMetadata == nullptr) {
      throw std::runtime_error(fFormat("%s: failed to load model from %s", __func__, pFname.c_str()));
    }

    mGetKey(vLlmKv(cLlmKvGeneralArchitecture), vArchName, false);
    vLlmKv = LLM_KV(fLlmArchFromString(vArchName));

    vFiles.emplace_back(new mgs_file(pFname.c_str(), "rb", vUseDirectIo));
    lContexts.emplace_back(vCtx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `vMetadata` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor *vCur = ggml_get_first_tensor(vCtx); vCur; vCur = ggml_get_next_tensor(vCtx, vCur)) {
      std::string vTensorName = std::string(vCur->name);
      // make sure there is no duplicated tensor names
      if (dWeightsMap.find(vTensorName) != dWeightsMap.end()) {
        throw std::runtime_error(fFormat("invalid model: tensor '%s' is duplicated", ggml_get_name(vCur)));
      }
      vNElements += ggml_nelements(vCur);
      vNBytes += ggml_nbytes(vCur);
      dWeightsMap.emplace(vTensorName, mgs_tensor_weight(vFiles.back().get(), 0, vMetadata, vCur));
    }
    uint16_t vNSplit = 0;
    mGetKey(vLlmKv(cLlmKvSplitCount), vNSplit, false);

    // Load additional GGML contexts
    if (vNSplit > 1) {
      // make sure the main file is loaded first
      uint16_t vIdx = 0;
      const std::string cKvSplitNo = vLlmKv(cLlmKvSplitNo);
      mGetKey(cKvSplitNo, vIdx);
      if (vIdx != 0) {
        throw std::runtime_error(fFormat(
            "illegal split file idx: %d (file: %s), model must be loaded with the first split", vIdx, pFname.c_str()));
      }

      // generate list of splits if needed
      if (pSplits.empty()) {
        pSplits = fMgsGetListSplits(pFname, vIdx, vNSplit);
      }

      // in case user give a custom list of splits, check if it matches the expected number
      if (vNSplit != (uint16_t)pSplits.size()) {
        throw std::runtime_error(
            fFormat("invalid split count, given: %zu splits, but expected %d", pSplits.size(), vNSplit));
      }

      if (vTrace > 0) {
        MGS_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, vNSplit);
      }

      // load other splits
      for (vIdx = 1; vIdx < vNSplit; vIdx++) {
        const char *vFnameSplit = pSplits[vIdx].c_str();

        struct gguf_init_params vSplitParams = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &vCtx,
        };
        gguf_context_ptr vCtxGguf{gguf_init_from_file(vFnameSplit, vSplitParams)};
        if (!vCtxGguf) {
          throw std::runtime_error(fFormat("%s: failed to load GGUF split from %s", __func__, vFnameSplit));
        }

        // check idx
        {
          const int cKid = gguf_find_key(vCtxGguf.get(), cKvSplitNo.c_str());
          if (cKid < 0) {
            throw std::runtime_error(fFormat("missing key %s in GGUF split %s", cKvSplitNo.c_str(), vFnameSplit));
          }
          int vIdxGguf = gguf_get_val_u16(vCtxGguf.get(), cKid);
          if (vIdxGguf != vIdx) {
            throw std::runtime_error(
                fFormat("invalid split file idx: %d (file: %s), expected %d", vIdxGguf, vFnameSplit, vIdx));
          }
        }

        vFiles.emplace_back(new mgs_file(vFnameSplit, "rb", vUseDirectIo));
        lContexts.emplace_back(vCtx);

        // Save tensors data offset info of the shard.
        for (ggml_tensor *vCur = ggml_get_first_tensor(vCtx); vCur; vCur = ggml_get_next_tensor(vCtx, vCur)) {
          std::string vTensorName = std::string(vCur->name);
          // make sure there is no duplicated tensor names
          if (dWeightsMap.find(vTensorName) != dWeightsMap.end()) {
            throw std::runtime_error(fFormat("invalid model: tensor '%s' is duplicated", ggml_get_name(vCur)));
          }
          vNElements += ggml_nelements(vCur);
          vNBytes += ggml_nbytes(vCur);
          dWeightsMap.emplace(vTensorName, mgs_tensor_weight(vFiles.back().get(), vIdx, vCtxGguf.get(), vCur));
        }
      }

      mGetKey(vLlmKv(cLlmKvSplitTensorsCount), vNTensors);

      // sanity check
      {
        const int cNTensorsLoaded = (int)dWeightsMap.size();
        if (vNTensors != cNTensorsLoaded) {
          throw std::runtime_error(
              fFormat("corrupted model: %d tensors expected but %d found", vNTensors, cNTensorsLoaded));
        }
      }

      MGS_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n", __func__, vNSplit - 1);
    }
  } else if (pFile != nullptr) {
    struct ggml_context *vCtx = NULL;
    struct gguf_init_params vParams = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &vCtx,
    };

    vMetadataPtr.reset(gguf_init_from_file_ptr(pFile, vParams));
    vMetadata = vMetadataPtr.get();
    if (vMetadata == nullptr) {
      throw std::runtime_error(fFormat("%s: failed to load model from file pointer", __func__));
    }

    mGetKey(vLlmKv(cLlmKvGeneralArchitecture), vArchName, false);
    vLlmKv = LLM_KV(fLlmArchFromString(vArchName));

    vFiles.emplace_back(new mgs_file(pFile));
    lContexts.emplace_back(vCtx);

    // Save tensors data offset info of the main file.
    for (ggml_tensor *vCur = ggml_get_first_tensor(vCtx); vCur; vCur = ggml_get_next_tensor(vCtx, vCur)) {
      std::string vTensorName = std::string(vCur->name);
      // make sure there is no duplicated tensor names
      if (dWeightsMap.find(vTensorName) != dWeightsMap.end()) {
        throw std::runtime_error(fFormat("invalid model: tensor '%s' is duplicated", ggml_get_name(vCur)));
      }
      vNElements += ggml_nelements(vCur);
      vNBytes += ggml_nbytes(vCur);
      dWeightsMap.emplace(vTensorName, mgs_tensor_weight(vFiles.back().get(), 0, vMetadata, vCur));
    }
  } else {
    mGetKey(vLlmKv(cLlmKvGeneralArchitecture), vArchName, false);
    vLlmKv = LLM_KV(fLlmArchFromString(vArchName));
  }

  if (vMetadata != nullptr && fLlmArchFromString(vArchName) == cLlmArchDeepseek41) {
    fMgsDeepseek41AdaptMetadata(vMetadata);
  }

  vNKv = gguf_get_n_kv(vMetadata);
  vNTensors = dWeightsMap.size();

  vFver = (enum mgs_fver)gguf_get_version(vMetadata);

  MGS_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n", __func__, vNKv,
               vNTensors, pFname.empty() ? "(file*)" : pFname.c_str(), mgs_file_version_name(vFver));

  // determine file type based on the number of tensors for each quantization and print meta data
  // TODO: make optional
  {
    std::map<enum ggml_type, uint32_t> dNType;

    uint32_t vNTypeMax = 0;
    enum ggml_type vTypeMax = GGML_TYPE_F32;

    for (const auto &vIt : dWeightsMap) {
      const mgs_tensor_weight &vW = vIt.second;
      const ggml_tensor *vTensor = vW.vTensor;

      enum ggml_type vType = vTensor->type;

      dNType[vType]++;

      if (vNTypeMax < dNType[vType]) {
        vNTypeMax = dNType[vType];
        vTypeMax = vType;
      }

      if (vTrace > 0) {
        const uint16_t cSid = vW.vIdx;
        MGS_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__, cSid, ggml_get_name(vTensor),
                     ggml_type_name(vType), mgs_format_tensor_shape(vTensor).c_str(),
                     ggml_nbytes(vTensor) / 1024.0f / 1024.0f);
      }
    }

    switch (vTypeMax) {
    case GGML_TYPE_F32:
      vFtype = MGS_FTYPE_ALL_F32;
      break;
    case GGML_TYPE_F16:
      vFtype = MGS_FTYPE_MOSTLY_F16;
      break;
    case GGML_TYPE_BF16:
      vFtype = MGS_FTYPE_MOSTLY_BF16;
      break;
    case GGML_TYPE_Q4_0:
      vFtype = MGS_FTYPE_MOSTLY_Q4_0;
      break;
    case GGML_TYPE_Q4_1:
      vFtype = MGS_FTYPE_MOSTLY_Q4_1;
      break;
    case GGML_TYPE_Q5_0:
      vFtype = MGS_FTYPE_MOSTLY_Q5_0;
      break;
    case GGML_TYPE_Q5_1:
      vFtype = MGS_FTYPE_MOSTLY_Q5_1;
      break;
    case GGML_TYPE_Q8_0:
      vFtype = MGS_FTYPE_MOSTLY_Q8_0;
      break;
    case GGML_TYPE_Q2_K:
      vFtype = MGS_FTYPE_MOSTLY_Q2_K;
      break;
    case GGML_TYPE_Q3_K:
      vFtype = MGS_FTYPE_MOSTLY_Q3_K_M;
      break;
    case GGML_TYPE_Q4_K:
      vFtype = MGS_FTYPE_MOSTLY_Q4_K_M;
      break;
    case GGML_TYPE_Q5_K:
      vFtype = MGS_FTYPE_MOSTLY_Q5_K_M;
      break;
    case GGML_TYPE_Q6_K:
      vFtype = MGS_FTYPE_MOSTLY_Q6_K;
      break;
    case GGML_TYPE_TQ1_0:
      vFtype = MGS_FTYPE_MOSTLY_TQ1_0;
      break;
    case GGML_TYPE_TQ2_0:
      vFtype = MGS_FTYPE_MOSTLY_TQ2_0;
      break;
    case GGML_TYPE_IQ2_XXS:
      vFtype = MGS_FTYPE_MOSTLY_IQ2_XXS;
      break;
    case GGML_TYPE_IQ2_XS:
      vFtype = MGS_FTYPE_MOSTLY_IQ2_XS;
      break;
    case GGML_TYPE_IQ2_S:
      vFtype = MGS_FTYPE_MOSTLY_IQ2_S;
      break;
    case GGML_TYPE_IQ3_XXS:
      vFtype = MGS_FTYPE_MOSTLY_IQ3_XXS;
      break;
    case GGML_TYPE_IQ1_S:
      vFtype = MGS_FTYPE_MOSTLY_IQ1_S;
      break;
    case GGML_TYPE_IQ1_M:
      vFtype = MGS_FTYPE_MOSTLY_IQ1_M;
      break;
    case GGML_TYPE_IQ4_NL:
      vFtype = MGS_FTYPE_MOSTLY_IQ4_NL;
      break;
    case GGML_TYPE_IQ4_XS:
      vFtype = MGS_FTYPE_MOSTLY_IQ4_XS;
      break;
    case GGML_TYPE_IQ3_S:
      vFtype = MGS_FTYPE_MOSTLY_IQ3_S;
      break;
    case GGML_TYPE_NVFP4:
      vFtype = MGS_FTYPE_MOSTLY_NVFP4;
      break;
    case GGML_TYPE_Q1_0:
      vFtype = MGS_FTYPE_MOSTLY_Q1_0;
      break;
    case GGML_TYPE_Q2_0:
      vFtype = MGS_FTYPE_MOSTLY_Q2_0;
      break;
    default: {
      MGS_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(vTypeMax));
      vFtype = MGS_FTYPE_ALL_F32;
    } break;
    }

    // this is a way to mark that we have "guessed" the file type
    vFtype = (mgs_ftype)(vFtype | MGS_FTYPE_GUESSED);

    {
      uint32_t vFtypeVal = 0;
      if (mGetKey(cLlmKvGeneralFileType, vFtypeVal, false)) {
        vFtype = (mgs_ftype)vFtypeVal;
      }
    }

    MGS_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

    for (int vI = 0; vI < vNKv; vI++) {
      const char *vName = gguf_get_key(vMetadata, vI);
      const enum gguf_type cType = gguf_get_kv_type(vMetadata, vI);
      const std::string cTypeName = cType == GGUF_TYPE_ARRAY ? fFormat("%s[%s,%zu]", gguf_type_name(cType),
                                                                       gguf_type_name(gguf_get_arr_type(vMetadata, vI)),
                                                                       gguf_get_arr_n(vMetadata, vI))
                                                             : gguf_type_name(cType);

      std::string vValue = fGgufKvToStr(vMetadata, vI);
      const size_t cMaxValueLen = 40;
      if (vValue.size() > cMaxValueLen) {
        vValue = fFormat("%s...", vValue.substr(0, cMaxValueLen - 3).c_str());
      }
      fReplaceAll(vValue, "\n", "\\n");

      MGS_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, vI, vName, cTypeName.c_str(), vValue.c_str());
    }

    // print type counts
    for (auto &vKv : dNType) {
      if (vKv.second == 0) {
        continue;
      }

      MGS_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(vKv.first), vKv.second);
    }
  }

  if (this->vUseMmap && !mgs_mmap::cSupported) {
    MGS_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
    this->vUseMmap = false;
  }

  this->vCheckTensors = pCheckTensors;
  this->vNoAlloc = pNoAlloc;
  this->vLoadMtp = pLoadMtp;
}

std::string mgs_model_loader::mGetArchName() const { return vArchName; }

enum llm_arch mgs_model_loader::mGetArch() const { return vLlmKv.vArch; }

const mgs_model_loader::mgs_tensor_weight *mgs_model_loader::mGetWeight(const char *pName) const {
  auto vPos = dWeightsMap.find(pName);
  if (vPos != dWeightsMap.end()) {
    return &vPos->second;
  }

  return nullptr;
}

const mgs_model_loader::mgs_tensor_weight &mgs_model_loader::mRequireWeight(const char *pName) const {
  const mgs_tensor_weight *vWeight = mGetWeight(pName);
  if (!vWeight) {
    throw std::runtime_error(fFormat("%s: tensor '%s' not found", __func__, pName));
  }
  return *vWeight;
}

struct ggml_tensor *mgs_model_loader::mGetTensorMeta(const char *pName) const {
  const auto *vWeight = mGetWeight(pName);
  if (!vWeight) {
    return nullptr;
  }
  return vWeight->vTensor;
}

struct ggml_tensor *mgs_model_loader::mRequireTensorMeta(const std::string &pName) const {
  struct ggml_tensor *vTensor = mGetTensorMeta(pName.c_str());
  if (!vTensor) {
    throw std::runtime_error(fFormat("%s: tensor '%s' not found", __func__, pName.c_str()));
  }
  return vTensor;
}

const struct ggml_tensor *mgs_model_loader::mCheckTensorDims(const std::string &pName, const std::vector<int64_t> &pNe,
                                                             bool pRequired, bool pAllowReshape) const {
  const struct ggml_tensor *vCur = mGetTensorMeta(pName.c_str());

  if (vCur == NULL) {
    if (!pRequired) {
      return NULL;
    }
    throw std::runtime_error(fFormat("%s: tensor '%s' not found", __func__, pName.c_str()));
  }

  bool vIsOk = true;

  if (pAllowReshape) {
    // check total number of elements only
    const int64_t cNcur = ggml_nelements(vCur);
    int64_t vNexp = 1;
    for (size_t vI = 0; vI < pNe.size(); ++vI) {
      vNexp *= pNe[vI];
    }
    if (cNcur != vNexp) {
      vIsOk = false;
    }
  } else {
    for (size_t vI = 0; vI < GGML_MAX_DIMS; ++vI) {
      if ((vI < pNe.size() && pNe[vI] != vCur->ne[vI]) || (vI >= pNe.size() && vCur->ne[vI] != 1)) {
        vIsOk = false;
        break;
      }
    }
  }

  if (!vIsOk) {
    throw std::runtime_error(fFormat("%s: tensor '%s' has wrong shape; expected %s, got %s", __func__, pName.c_str(),
                                     mgs_format_tensor_shape(pNe).c_str(), mgs_format_tensor_shape(vCur).c_str()));
  }

  return vCur;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool fWeightBuftSupported(const mgs_hparams &pHparams, ggml_tensor *pW, ggml_op pOp,
                                 ggml_backend_buffer_type_t pBuft, ggml_backend_dev_t pDev) {
  GGML_ASSERT(pW != nullptr);

  if (pOp == GGML_OP_NONE) {
    return true;
  }

  ggml_init_params vParams = {
      /*.mem_size   =*/ggml_tensor_overhead() * 8,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,
  };
  ggml_context_ptr vCtxPtr{ggml_init(vParams)};
  if (!vCtxPtr) {
    throw std::runtime_error(fFormat("failed to create ggml context"));
  }
  ggml_context *vCtx = vCtxPtr.get();

  ggml_tensor *vOpTensor = nullptr;

  switch (pOp) {
  case GGML_OP_GET_ROWS: {
    ggml_tensor *vB = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, 512);
    vOpTensor = ggml_get_rows(vCtx, pW, vB);
  } break;
  case GGML_OP_MUL_MAT: {
    ggml_tensor *vB = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, pW->ne[0], 512, pW->ne[2], pW->ne[3]);
    vOpTensor = ggml_mul_mat(vCtx, pW, vB);
  } break;
  case GGML_OP_MUL_MAT_ID: {
    const int cNExpertUsed = pHparams.vNExpertUsed;
    GGML_ASSERT(cNExpertUsed > 0);
    ggml_tensor *vB = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pW->ne[0], cNExpertUsed, 512);
    ggml_tensor *vIds = ggml_new_tensor_2d(vCtx, GGML_TYPE_I32, cNExpertUsed, 512);
    vOpTensor = ggml_mul_mat_id(vCtx, pW, vB, vIds);
  } break;
  case GGML_OP_ADD: {
    ggml_tensor *vA = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, pW->ne[0], pW->ne[1], pW->ne[2], pW->ne[3]);
    vOpTensor = ggml_add(vCtx, vA, pW);
  } break;
  case GGML_OP_ADD_ID: {
    const int cNExpertUsed = pHparams.vNExpertUsed;
    GGML_ASSERT(cNExpertUsed > 0);
    ggml_tensor *vA = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pW->ne[0], cNExpertUsed, 512);
    ggml_tensor *vC = ggml_new_tensor_2d(vCtx, GGML_TYPE_I32, cNExpertUsed, 512);
    vOpTensor = ggml_add_id(vCtx, vA, pW, vC);
  } break;
  case GGML_OP_MUL: {
    ggml_tensor *vA = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, pW->ne[0], pW->ne[1], pW->ne[2], pW->ne[3]);
    vOpTensor = ggml_mul(vCtx, vA, pW);
  } break;
  case GGML_OP_DIV: {
    ggml_tensor *vA = ggml_new_tensor_1d(vCtx, GGML_TYPE_F32, pW->ne[0]);
    vOpTensor = ggml_div(vCtx, vA, pW);
  } break;
  case GGML_OP_ROPE: {
    const int cNEmbdHead = pHparams.mNEmbdHeadV();
    const int cNHead = pHparams.mNHead();
    ggml_tensor *vA = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cNEmbdHead, cNHead, 512);
    ggml_tensor *vB = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, 512);
    vOpTensor = ggml_rope_ext(vCtx, vA, vB, pW, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  } break;
  case GGML_OP_SSM_CONV: {
    const int64_t cNSeqTokens = 512;
    const int64_t cNSeqs = 3;
    ggml_tensor *vConvX = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pW->ne[0] - 1 + cNSeqTokens, pW->ne[1], cNSeqs);
    vOpTensor = ggml_ssm_conv(vCtx, vConvX, pW);
  } break;
  case GGML_OP_SSM_SCAN: {
    // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
    const int64_t cDState = pW->ne[0] == 1 ? pHparams.vSsmDState : pW->ne[0];
    const int64_t cNHead = pW->ne[1];
    const int64_t cHeadDim = pHparams.vSsmDInner / cNHead;
    const int64_t cNGroup = pHparams.vSsmNGroup ? pHparams.vSsmNGroup : 1;
    const int64_t cNSeqTokens = 512;
    const int64_t cNSeqs = 3;
    ggml_tensor *vS = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cDState, cHeadDim, cNHead, cNSeqs);
    ggml_tensor *vX = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cHeadDim, cNHead, cNSeqTokens, cNSeqs);
    ggml_tensor *vDt = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cNHead, cNSeqTokens, cNSeqs);
    ggml_tensor *vB = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cDState, cNGroup, cNSeqTokens, cNSeqs);
    ggml_tensor *vC = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cDState, cNGroup, cNSeqTokens, cNSeqs);
    ggml_tensor *vIds = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, cNSeqs);
    vOpTensor = ggml_ssm_scan(vCtx, vS, vX, vDt, pW, vB, vC, vIds);
  } break;
  case GGML_OP_RWKV_WKV6: {
    // FIXME
    const int64_t cS = 123;
    const int64_t cH = 123;
    const int64_t cNTokens = 123;
    const int64_t cNSeqs = 123;
    ggml_tensor *vK = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cS, cH, cNTokens);
    ggml_tensor *vV = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cS, cH, cNTokens);
    ggml_tensor *vR = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cS, cH, cNTokens);
    ggml_tensor *vTf = pW;
    ggml_tensor *vTd = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, cS, cH, cNTokens);
    ggml_tensor *vState = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cS, cNSeqs, cS, cH);
    vOpTensor = ggml_rwkv_wkv6(vCtx, vK, vV, vR, vTf, vTd, vState);
  } break;
  case GGML_OP_IM2COL: {
    const int cNEmbdInp = pHparams.mNEmbdInp();
    ggml_tensor *vB = ggml_new_tensor_4d(vCtx, GGML_TYPE_F32, cNEmbdInp, pW->ne[1], 1, 1);
    vOpTensor = ggml_im2col(vCtx, pW, vB, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
  } break;
  case GGML_OP_SCALE: {
    vOpTensor = ggml_scale(vCtx, pW, 1.0f);
  } break;
  default:
    GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(pOp), pW->name);
  }

  // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
  GGML_ASSERT(pW->buffer == nullptr);
  pW->buffer = ggml_backend_buft_alloc_buffer(pBuft, 0);
  bool vOpSupported = ggml_backend_dev_supports_op(pDev, vOpTensor);
  ggml_backend_buffer_free(pW->buffer);
  pW->buffer = nullptr;

  return vOpSupported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t fSelectWeightBuft(const mgs_hparams &pHparams, ggml_tensor *pTensor, ggml_op pOp,
                                                    const buft_list_t *pBuftList) {
  GGML_ASSERT(!pBuftList->empty());
  for (const auto &vCur : *pBuftList) {
    ggml_backend_dev_t vCurDev = vCur.first;
    ggml_backend_buffer_type_t vCurBuft = vCur.second;
    if (fWeightBuftSupported(pHparams, pTensor, pOp, vCurBuft, vCurDev)) {
      return vCurBuft;
    }
  }

  return nullptr;
}

struct ggml_tensor *mgs_model_loader::mCreateTensor(const mgs_hparams &pHparams, const buft_list_t *pBuftListCpu,
                                                    const buft_list_t *pBuftListInput,
                                                    const buft_list_t *pBuftListOutput,
                                                    const buft_list_t *pBuftListLayer, const LLM_TN_IMPL &pTn,
                                                    const std::initializer_list<int64_t> &pNe, int pFlags) {
  if (pFlags & cTensorExternal) {
    // the model reads this tensor from the file on demand: count it, but never allocate it
    if (gguf_find_tensor(vMetadata, pTn.mStr().c_str()) < 0) {
      if (pFlags & cTensorNotRequired) {
        return nullptr;
      }
      throw std::runtime_error(fFormat("missing tensor '%s'", pTn.mStr().c_str()));
    }
    if (!vFiles.empty()) {
      const struct ggml_tensor *vCur = mCheckTensorDims(pTn.mStr(), pNe, true, pFlags & cTensorAllowReshape);
      const mgs_tensor_weight &vWeight = mRequireWeight(pTn.mStr().c_str());
      const size_t cNbytes = ggml_nbytes(vCur);
      MGS_LOG_INFO("%s: tensor %s (%.2f GiB) is read from the model file on demand\n", __func__,
                   pTn.mStr().c_str(), cNbytes / 1024.0 / 1024.0 / 1024.0);

      vSizeData -= cNbytes;
      auto vIt = dExternalFirstOffs.find(vWeight.vIdx);
      if (vIt == dExternalFirstOffs.end() || vWeight.vOffs < vIt->second) {
        dExternalFirstOffs[vWeight.vIdx] = vWeight.vOffs;
      }
    }
    vNCreated++;
    return nullptr;
  }

  bool vMtpTensor = false;
  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto &dContexts = vMtpTensor ? dMtpContexts : dCtxMap;
    auto vIt = dContexts.find(pBuft);
    if (vIt == dContexts.end()) {
      // one ggml context per buffer type
      int vMaxNTensors = vNTensors;
      vMaxNTensors += 1;                      // duplicated output tensor
      vMaxNTensors += pHparams.mNLayer() * 2; // duplicated rope freq tensors
      if (vFiles.empty()) {
        vMaxNTensors += pHparams.mNLayer() * 256; // this should be well above what any model actually uses
      }
      const size_t cCtxSize = ggml_tensor_overhead() * vMaxNTensors;

      ggml_init_params vParams = {
          /*.mem_size   =*/cCtxSize,
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };

      ggml_context *vCtx = ggml_init(vParams);
      if (!vCtx) {
        throw std::runtime_error(fFormat("failed to create ggml context"));
      }

      dContexts.emplace(pBuft, vCtx);

      return vCtx;
    }
    return vIt->second.get();
  };

  auto vBuftForTensor = [&](ggml_tensor *pTMeta) -> ggml_backend_buffer_type_t {
    if (!pTMeta) {
      if (pFlags & cTensorNotRequired) {
        return nullptr;
      }
      throw std::runtime_error(fFormat("missing tensor '%s'", pTn.mStr().c_str()));
    }

    // some models use the token embedding tensor as the output, but since these are used in different layers and with
    // different ops the tensor is duplicated to handle this, we check if the tensor is duplicated, and if so, we assume
    // that it is being loaded as the output tensor
    llm_tensor vTnTensor = pTn.cTensor;
    if (pTn.cTensor == cLlmTensorTokenEmbd && (pFlags & cTensorDuplicated)) {
      vTnTensor = cLlmTensorOutput;
    }

    llm_tensor_info vInfo;
    try {
      vInfo = fLlmTensorInfoFor(vTnTensor);
    } catch (const std::out_of_range &vE) {
      throw std::runtime_error(fFormat("missing tensor info mapping for %s", pTn.mStr().c_str()));
    }

    // skip unused tensors
    if (vInfo.vOp == GGML_OP_NONE || (pFlags & cTensorSkip)) {
      const size_t cNbytes = ggml_nbytes(pTMeta);
      MGS_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", pTn.mStr().c_str(), cNbytes);

      vSizeData -= cNbytes;
      vNCreated++;

      return nullptr;
    }

    // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
    ggml_op vOp;
    bool vBias = pTn.cSuffix != nullptr && strcmp(pTn.cSuffix, "bias") == 0;
    if (vBias) {
      if (vInfo.vOp == GGML_OP_MUL_MAT_ID) {
        vOp = GGML_OP_ADD_ID;
      } else {
        vOp = GGML_OP_ADD;
      }
    } else {
      vOp = vInfo.vOp;
    }

    // sanity checks
    if (vInfo.vLayer == cLlmTensorLayerInput || vInfo.vLayer == cLlmTensorLayerOutput) {
      if (pTn.cBid != -1) {
        GGML_ABORT("input/output layer tensor %s used with a layer number", pTn.mStr().c_str());
      }
    } else {
      if (pTn.cBid == -1) {
        GGML_ABORT("repeating layer tensor %s used without a layer number", pTn.mStr().c_str());
      }
    }

    // select the buffer type for this tensor
    const buft_list_t *vBuftList;
    switch (vInfo.vLayer) {
    case cLlmTensorLayerInput:
      vBuftList = pBuftListInput;
      break;
    case cLlmTensorLayerOutput:
      vBuftList = pBuftListOutput;
      break;
    case cLlmTensorLayerRepeating:
      GGML_ASSERT(pBuftListLayer != nullptr);
      vBuftList = pBuftListLayer;
      break;
    default:
      GGML_ABORT("invalid layer %d for tensor %s", vInfo.vLayer, pTn.mStr().c_str());
    }

    ggml_backend_buffer_type_t vBuft = nullptr;

    // MTP placement already accounts for free memory on every selected GPU.
    // Neither host nor device overrides may replace that priority placement.
    vMtpTensor = vKeepMtpOnDevice &&
                 ((vInfo.vLayer == cLlmTensorLayerRepeating && pTn.cBid >= vMtpFirstLayer) ||
                  vInfo.vLayer == cLlmTensorLayerOutput || (vMtpHeadOnly && vInfo.vLayer == cLlmTensorLayerInput));

    // check overrides
    if (vTensorBuftOverrides && !vMtpTensor) {
      std::string vTensorName = pTn.mStr();
      for (const auto *vOverrides = vTensorBuftOverrides; vOverrides->pattern != nullptr; ++vOverrides) {
        std::regex vPattern(vOverrides->pattern);
        if (std::regex_search(vTensorName, vPattern)) {
          if (vOverrides->buft == ggml_backend_cpu_buffer_type()) {
            vHasCpuTensorOverrides = true;
            // when overriding to a CPU buffer, consider the extra buffer types
            vBuft = fSelectWeightBuft(pHparams, pTMeta, vOp, pBuftListCpu);
          } else {
            vBuft = vOverrides->buft;
          }

          MGS_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n", vTensorName.c_str(),
                        ggml_nbytes(pTMeta) / 1024 / 1024, ggml_type_name(pTMeta->type), ggml_backend_buft_name(vBuft));
          break;
        }
      }
    }

    if (!vBuft) {
      vBuft = fSelectWeightBuft(pHparams, pTMeta, vOp, vBuftList);
      if (!vBuft) {
        throw std::runtime_error(fFormat("failed to find a compatible buffer type for tensor %s", pTn.mStr().c_str()));
      }
    }

    // avoid using a host buffer when using mmap
    auto *vBuftDev = ggml_backend_buft_get_device(vBuft);
    if (vUseMmap && vBuftDev && vBuft == ggml_backend_dev_host_buffer_type(vBuftDev)) {
      auto *vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      if (!vCpuDev) {
        throw std::runtime_error("no CPU backend found");
      }
      vBuft = ggml_backend_dev_buffer_type(vCpuDev);
    }

    if (vBuft != vBuftList->front().second) {
      if (vNTensorsMoved == 0) {
        vFirstTensorMovedName = pTMeta->name;
        vFirstTensorMovedTypeName = ggml_type_name(pTMeta->type);
        vFirstMovedFromBuft = vBuftList->front().second;
        vFirstMovedToBuft = vBuft;
      }
      vNTensorsMoved++;
    }

    return vBuft;
  };

  if (vFiles.empty()) {
    if (pFlags & cTensorSkipIfVirtual) {
      return nullptr;
    }
    ggml_type vType = GGML_TYPE_F32;
    const int64_t cTid = gguf_find_tensor(vMetadata, pTn.mStr().c_str());
    if (cTid != -1) {
      vType = gguf_get_tensor_type(vMetadata, cTid);
    }

    // for tensors that are not required some of the dimensions can be invalid:
    if (pFlags & cTensorNotRequired) {
      for (size_t vDim = 0; vDim < pNe.size(); vDim++) {
        if (pNe.begin()[vDim] <= 0) {
          return nullptr;
        }
      }
    }

    ggml_tensor vTMeta;
    memset(&vTMeta, 0, sizeof(ggml_tensor));
    vTMeta.type = vType;
    for (size_t vDim = 0; vDim < GGML_MAX_DIMS; vDim++) {
      vTMeta.ne[vDim] = vDim < pNe.size() ? pNe.begin()[vDim] : 1;
      GGML_ASSERT(vTMeta.ne[vDim] >= 1);
      vTMeta.nb[vDim] = vDim == 0 ? ggml_type_size(vType) : vTMeta.ne[vDim - 1] * vTMeta.nb[vDim - 1];
      GGML_ASSERT(vTMeta.nb[vDim] >= 1);
    }
    ggml_set_name(&vTMeta, pTn.mStr().c_str());

    ggml_backend_buffer_type_t vBuft = vBuftForTensor(&vTMeta);
    GGML_ASSERT(vBuft != nullptr);
    ggml_context *vCtx = vCtxForBuft(vBuft);
    ggml_tensor *vRet = ggml_dup_tensor(vCtx, &vTMeta);
    ggml_set_name(vRet, pTn.mStr().c_str());
    return vRet;
  }

  MGS_LOG_DEBUG("%s: loading tensor %s\n", __func__, pTn.mStr().c_str());
  const struct ggml_tensor *vCur =
      mCheckTensorDims(pTn.mStr(), pNe, !(pFlags & cTensorNotRequired), pFlags & cTensorAllowReshape);
  if (vCur == NULL) {
    return NULL;
  }

  ggml_tensor vTMeta = *vCur;
  if (pFlags & cTensorAllowReshape) {
    for (size_t vDim = 0; vDim < GGML_MAX_DIMS; vDim++) {
      vTMeta.ne[vDim] = vDim < pNe.size() ? pNe.begin()[vDim] : 1;
      vTMeta.nb[vDim] = vDim == 0 ? ggml_type_size(vTMeta.type) : vTMeta.ne[vDim - 1] * vTMeta.nb[vDim - 1];
    }
  }

  ggml_backend_buffer_type_t vBuft = vBuftForTensor(&vTMeta);
  if (vBuft == nullptr) {
    return nullptr;
  }

  ggml_context *vCtx = vCtxForBuft(vBuft);

  // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new
  // one
  if (pFlags & cTensorDuplicated) {
    ggml_tensor *vT = ggml_get_tensor(vCtx, pTn.mStr().c_str());
    if (vT) {
      return vT;
    }
  }

  const bool cDuplicated = pFlags & cTensorDuplicated;

  struct ggml_tensor *vTensor = ggml_dup_tensor(vCtx, &vTMeta);
  ggml_set_name(vTensor, ggml_get_name(&vTMeta));

  if (cDuplicated) {
    vSizeData += ggml_nbytes(&vTMeta);
  } else {
    vNCreated++;
  }

  return vTensor;
}

void mgs_model_loader::mDoneGettingTensors(bool pPartial) const {
  if (vNCreated > vNTensors) {
    throw std::runtime_error(
        fFormat("%s: too many tensors created; expected %d, got %d", __func__, vNTensors, vNCreated));
  }
  if (vNCreated < vNTensors) {
    if (!pPartial) {
      throw std::runtime_error(
          fFormat("%s: wrong number of tensors; expected %d, got %d", __func__, vNTensors, vNCreated));
    }
    MGS_LOG_INFO(
        "%s: partial load — used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n",
        __func__, vNCreated, vNTensors);
  }
  if (vNTensorsMoved > 0) {
    MGS_LOG_DEBUG(
        "%s: tensor '%s' (%s) (and %zu others) cannot be used with preferred buffer type %s, using %s instead\n",
        __func__, vFirstTensorMovedName.c_str(), vFirstTensorMovedTypeName.c_str(), vNTensorsMoved - 1,
        ggml_backend_buft_name(vFirstMovedFromBuft), ggml_backend_buft_name(vFirstMovedToBuft));
  }
}

void mgs_model_loader::mInitMappings(bool pPrefetch, mgs_mlocks *pMlockMmaps) {
  if (vUseMmap) {
    vMappings.reserve(vFiles.size());
    lMmapsUsed.reserve(vFiles.size());
    for (const auto &vFile : vFiles) {
      bool vIsNuma = false;

      auto *vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      if (vDev) {
        auto *vReg = ggml_backend_dev_backend_reg(vDev);
        auto *vIsNumaFn = (decltype(ggml_is_numa) *)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_cpu_is_numa");
        if (vIsNumaFn) {
          vIsNuma = vIsNumaFn();
        }
      }

      // never prefetch the tensors that are read on demand (they can be hundreds of GiB): stop before the first
      size_t vPrefetch = pPrefetch ? (size_t)-1 : 0;
      const auto cItExternal = dExternalFirstOffs.find((uint16_t)vMappings.size());
      if (vPrefetch != 0 && cItExternal != dExternalFirstOffs.end()) {
        vPrefetch = cItExternal->second;
      }

      std::unique_ptr<mgs_mmap> vMapping = std::make_unique<mgs_mmap>(vFile.get(), vPrefetch, vIsNuma);
      lMmapsUsed.emplace_back(vMapping->size(), 0);
      if (pMlockMmaps) {
        std::unique_ptr<mgs_mlock> vMlockMmap(new mgs_mlock());
        vMlockMmap->mInit(vMapping->mAddr());
        pMlockMmaps->emplace_back(std::move(vMlockMmap));
      }
      vMappings.emplace_back(std::move(vMapping));
    }
  }

  // compute the total size of all tensors for progress reporting
  for (const auto &vIt : dWeightsMap) {
    vSizeData += ggml_nbytes(vIt.second.vTensor);
  }
}

void mgs_model_loader::mGetMappingRange(size_t *pFirst, size_t *pLast, void **pAddr, int pIdx,
                                        ggml_context *pCtx) const {
  GGML_ASSERT(!vMappings.empty());
  const auto &vMapping = vMappings.at(pIdx);

  *pFirst = vMapping->size();
  *pLast = 0;
  *pAddr = vMapping->mAddr();
  for (ggml_tensor *vTensor = ggml_get_first_tensor(pCtx); vTensor; vTensor = ggml_get_next_tensor(pCtx, vTensor)) {
    const auto *vWeight = mGetWeight(ggml_get_name(vTensor));
    if (!vWeight || vWeight->vIdx != pIdx) {
      continue;
    }
    *pFirst = std::min(*pFirst, vWeight->vOffs);
    *pLast = std::max(*pLast, vWeight->vOffs + ggml_nbytes(vTensor));
  }
}

void mgs_model_loader::mLoadDataFor(struct ggml_tensor *pCur) const {
  const auto &vW = mRequireWeight(ggml_get_name(pCur));

  if (vUseMmap) {
    const auto &vMapping = vMappings.at(vW.vIdx);
    if (pCur->data == nullptr) {
      pCur->data = (uint8_t *)vMapping->mAddr() + vW.vOffs;
    } else {
      memcpy(pCur->data, (uint8_t *)vMapping->mAddr() + vW.vOffs, ggml_nbytes(pCur));
    }
  } else {
    GGML_ASSERT(pCur->data != nullptr);
    GGML_ASSERT(vW.vIdx < vFiles.size());
    const auto &vFile = vFiles.at(vW.vIdx);
    vFile->mSeek(vW.vOffs, SEEK_SET);
    vFile->mReadRaw(pCur->data, ggml_nbytes(pCur));
  }

  if (vCheckTensors && !ggml_validate_row_data(pCur->type, pCur->data, ggml_nbytes(pCur))) {
    throw std::runtime_error(fFormat("tensor '%s' has invalid data", ggml_get_name(pCur)));
  }
}

bool mgs_model_loader::mLoadAllData(struct ggml_context *pCtx, mgs_buf_map &pBufs, mgs_mlocks *pLmlocks,
                                    mgs_progress_callback pProgressCallback, void *pProgressCallbackUserData) {
  if (vFiles.empty()) {
    for (ggml_tensor *vT = ggml_get_first_tensor(pCtx); vT != nullptr; vT = ggml_get_next_tensor(pCtx, vT)) {
      vSetTensorData(vT, vSetTensorDataUd);
    }
    return true;
  }
  GGML_ASSERT(vSizeData != 0 && "call mInitMappings() first");

  std::vector<no_init<uint8_t>> lReadBuf;
  std::vector<std::future<std::pair<ggml_tensor *, bool>>> lValidationResult;

  // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
  // NVMe raid configurations might require more / larger buffers.
  constexpr size_t cNBuffers = 4;

  size_t vAlignment = 1;
  for (const auto &vFile : vFiles) {
    vAlignment = std::max(vFile->mReadAlignment(), vAlignment);
  }

  // Buffer size: balance between memory usage and I/O efficiency
  const size_t cBufferSize = vAlignment != 1 ? MGS_DIRECT_IO_BUFFER_SIZE + 2 * vAlignment : 1 * 1024 * 1024;

  std::vector<ggml_backend_buffer_t> lHostBuffers;
  std::vector<ggml_backend_event_t> lEvents;
  std::vector<void *> lHostPtrs;
  size_t vBufferIdx = 0; // buffer to use for async loads
  ggml_backend_t vUploadBackend = [&](const char *pFunc) -> ggml_backend_t {
    if (vUseMmap || vCheckTensors) {
      return nullptr;
    }
    // When not using mmaped io use async uploads from pinned memory to GPU memory.
    // First determine if the backend supports the necessary features for async uploads.
    auto *vBuf = pBufs.count(0) ? pBufs.at(0) : nullptr;
    if (!vBuf) {
      MGS_LOG_DEBUG("%s: no buffer found for async uploads\n", pFunc);
      return nullptr;
    }

    auto *vBuft = ggml_backend_buffer_get_type(vBuf);
    auto *vDev = ggml_backend_buft_get_device(vBuft);
    if (!vDev) {
      MGS_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", pFunc, ggml_backend_buft_name(vBuft));
      return nullptr;
    }

    if (vBuft != ggml_backend_dev_buffer_type(vDev)) {
      MGS_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", pFunc,
                    ggml_backend_buft_name(vBuft), ggml_backend_dev_name(vDev));
      return nullptr;
    }

    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDev, &vProps);
    if (!vProps.caps.async || !vProps.caps.host_buffer || !vProps.caps.events) {
      MGS_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", pFunc,
                    ggml_backend_dev_name(vDev));
      return nullptr;
    }

    auto *vHostBuft = ggml_backend_dev_host_buffer_type(vDev);
    if (!vHostBuft) {
      MGS_LOG_DEBUG("%s: no host buffer type found for device %s\n", pFunc, ggml_backend_dev_name(vDev));
      return nullptr;
    }

    // If the backend is supported, create pinned memory buffers and events for synchronisation.
    for (size_t vIdx = 0; vIdx < cNBuffers; ++vIdx) {
      auto *vBuf = ggml_backend_buft_alloc_buffer(vHostBuft, cBufferSize);

      if (!vBuf) {
        MGS_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", pFunc,
                      ggml_backend_dev_name(vDev));
        return nullptr;
      }

      lHostBuffers.emplace_back(vBuf);
      lHostPtrs.emplace_back(ggml_backend_buffer_get_base(vBuf));

      auto *vEvent = ggml_backend_event_new(vDev);
      if (!vEvent) {
        MGS_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", pFunc,
                      ggml_backend_dev_name(vDev));
        return nullptr;
      }

      lEvents.emplace_back(vEvent);
    }

    ggml_backend_t vBackend = ggml_backend_dev_init(vDev, nullptr);
    if (!vBackend) {
      MGS_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", pFunc,
                    ggml_backend_dev_name(vDev));
      return nullptr;
    }

    return vBackend;
  }(__func__);

  if (vUploadBackend) {
    MGS_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
                  ggml_backend_dev_name(ggml_backend_get_device(vUploadBackend)),
                  ggml_backend_buft_name(ggml_backend_buffer_get_type(pBufs.at(0))), ggml_backend_name(vUploadBackend));
  }

  for (struct ggml_tensor *vCur = ggml_get_first_tensor(pCtx); vCur != NULL; vCur = ggml_get_next_tensor(pCtx, vCur)) {
    const auto *vWeight = mGetWeight(ggml_get_name(vCur));
    if (vWeight == nullptr) {
      // this can happen with split experts models
      continue;
    }

    if (pProgressCallback) {
      if (!pProgressCallback((float)vSizeDone / vSizeData, pProgressCallbackUserData)) {
        return false;
      }
    }

    size_t vNSize = ggml_nbytes(vCur);

    if (vUseMmap) {
      const auto &vMapping = vMappings.at(vWeight->vIdx);
      ggml_backend_buffer_t vBufMmap = nullptr;
      if (pBufs.count(vWeight->vIdx)) {
        vBufMmap = pBufs.at(vWeight->vIdx);
      }
      uint8_t *vData = (uint8_t *)vMapping->mAddr() + vWeight->vOffs;

      if (vCheckTensors) {
        lValidationResult.emplace_back(std::async(std::launch::async, [vCur, vData, vNSize] {
          return std::make_pair(vCur, ggml_validate_row_data(vCur->type, vData, vNSize));
        }));
      }

      GGML_ASSERT(vBufMmap ||
                  vCur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
      if (vBufMmap && vCur->data == nullptr) {
        ggml_backend_tensor_alloc(vBufMmap, vCur, vData);
        if (pLmlocks) {
          const auto &vLmlock = pLmlocks->at(vWeight->vIdx);
          vLmlock->mGrowTo(vWeight->vOffs + vNSize);
        }

        auto &vMmapUsed = lMmapsUsed[vWeight->vIdx];
        vMmapUsed.first = std::min(vMmapUsed.first, vWeight->vOffs);
        vMmapUsed.second = std::max(vMmapUsed.second, vWeight->vOffs + vNSize);
      } else {
        ggml_backend_tensor_set(vCur, vData, 0, vNSize);
      }
    } else {
      const auto &vFile = vFiles.at(vWeight->vIdx);

      if (ggml_backend_buffer_is_host(vCur->buffer)) {
        vFile->mSeek(vWeight->vOffs, SEEK_SET);
        vFile->mReadRaw(vCur->data, vNSize);
        if (vCheckTensors) {
          lValidationResult.emplace_back(std::async(std::launch::async, [vCur, vNSize] {
            return std::make_pair(vCur, ggml_validate_row_data(vCur->type, vCur->data, vNSize));
          }));
        }
      } else {
        // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously
        // to the GPU.
        if (vUploadBackend) {
          size_t vOffset = vWeight->vOffs;
          vAlignment = vFile->mReadAlignment();
          size_t vAlignedOffset = vOffset & ~(vAlignment - 1);
          size_t vOffsetFromAlignment = vOffset - vAlignedOffset;
          vFile->mSeek(vAlignedOffset, SEEK_SET);

          // Calculate aligned read boundaries
          size_t vReadStart = vAlignedOffset;
          size_t vReadEnd = (vOffset + vNSize + vAlignment - 1) & ~(vAlignment - 1);

          size_t vBytesRead = 0;
          size_t vDataRead = 0; // Actual tensor data copied (excluding padding)

          while (vBytesRead < vReadEnd - vReadStart) {
            size_t vReadSize = std::min<size_t>(cBufferSize, vReadEnd - vReadStart - vBytesRead);

            // Align the destination pointer within the pinned buffer
            uintptr_t vPtrDestAligned =
                (reinterpret_cast<uintptr_t>(lHostPtrs[vBufferIdx]) + vAlignment - 1) & ~(vAlignment - 1);

            // Wait for previous upload to complete before reusing buffer
            ggml_backend_event_synchronize(lEvents[vBufferIdx]);

            // Read aligned chunk from file
            vFile->mReadRawUnsafe(reinterpret_cast<void *>(vPtrDestAligned), vReadSize);

            // Calculate actual data portion (excluding alignment padding)
            uintptr_t vPtrData = vPtrDestAligned;
            size_t vDataToCopy = vReadSize;

            // Skip alignment padding at start of first chunk
            if (vBytesRead == 0) {
              vPtrData += vOffsetFromAlignment;
              vDataToCopy -= vOffsetFromAlignment;
            }

            // Trim alignment padding at end of last chunk
            if (vAlignedOffset + vBytesRead + vReadSize > vOffset + vNSize) {
              vDataToCopy -= (vReadEnd - (vOffset + vNSize));
            }

            // Async upload actual data to GPU
            ggml_backend_tensor_set_async(vUploadBackend, vCur, reinterpret_cast<void *>(vPtrData), vDataRead,
                                          vDataToCopy);
            ggml_backend_event_record(lEvents[vBufferIdx], vUploadBackend);

            vDataRead += vDataToCopy;
            vBytesRead += vReadSize;

            ++vBufferIdx;
            vBufferIdx %= cNBuffers;
          }
        } else {
          lReadBuf.resize(vNSize);
          vFile->mSeek(vWeight->vOffs, SEEK_SET);
          vFile->mReadRaw(lReadBuf.data(), vNSize);
          ggml_backend_tensor_set(vCur, lReadBuf.data(), 0, vNSize);
          if (vCheckTensors && !ggml_validate_row_data(vCur->type, lReadBuf.data(), vNSize)) {
            throw std::runtime_error(fFormat("tensor '%s' has invalid data", ggml_get_name(vCur)));
          }
        }
      }
    }

    vSizeDone += vNSize;
  }

  // free temporary resources used for async uploads
  for (auto *vEvent : lEvents) {
    ggml_backend_event_synchronize(vEvent);
    ggml_backend_event_free(vEvent);
  }
  for (auto *vBuf : lHostBuffers) {
    ggml_backend_buffer_free(vBuf);
  }
  ggml_backend_free(vUploadBackend);

  // check validation results
  bool vValidationFailed = false;
  for (auto &vFuture : lValidationResult) {
    auto vResult = vFuture.get();
    if (!vResult.second) {
      MGS_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(vResult.first));
      vValidationFailed = true;
    }
  }
  if (vValidationFailed) {
    throw std::runtime_error("found tensors with invalid data");
  }

  // check if this is the last call and do final cleanup
  if (vSizeDone >= vSizeData) {
    // unmap offloaded tensors and metadata
    if (vUseMmap) {
      // Pin weight pages retained in system memory so host-to-device transfers
      // do not need an intermediate pageable buffer.
      bool (*vRegFn)(void *, size_t) = nullptr;
      void (*vUnregFn)(void *) = nullptr;
      for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
        ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(ggml_backend_dev_get(vI));
        auto vCandidateReg =
            (bool (*)(void *, size_t))ggml_backend_reg_get_proc_address(vReg, "ggml_backend_register_host_buffer");
        auto vCandidateUnreg =
            (void (*)(void *))ggml_backend_reg_get_proc_address(vReg, "ggml_backend_unregister_host_buffer");
        if (vCandidateReg && vCandidateUnreg) {
          vRegFn = vCandidateReg;
          vUnregFn = vCandidateUnreg;
          break;
        }
      }

      size_t vNRegistrationRanges = 0;
      size_t vNRegisteredRanges = 0;
      size_t vNRegisteredBytes = 0;
      for (uint32_t vIdx = 0; vIdx < vMappings.size(); vIdx++) {
        const auto &vMmapUsed = lMmapsUsed.at(vIdx);
        auto &vMapping = vMappings.at(vIdx);
        vMapping->mUnmapFragment(0, vMmapUsed.first);
        if (vMmapUsed.second != 0) {
          vMapping->mUnmapFragment(vMmapUsed.second, vMapping->size());
        }
        if (vMmapUsed.second > vMmapUsed.first) {
          vNRegistrationRanges++;
          const size_t cNRegistered = vMapping->mRegisterHost(vMmapUsed.first, vMmapUsed.second, vRegFn, vUnregFn);
          if (cNRegistered > 0) {
            vNRegisteredRanges++;
            vNRegisteredBytes += cNRegistered;
          }
        }
      }

      if (vNRegisteredBytes > 0) {
        MGS_LOG_INFO("%s: registered %.2f MiB of mapped memory for H2D transfers; --load-mode none is not needed\n",
                     __func__, vNRegisteredBytes / 1024.0 / 1024.0);
      }
      if (vHasCpuTensorOverrides && vRegFn == nullptr && vNRegistrationRanges > 0) {
        // only the CUDA backend exports ggml_backend_register_host_buffer (Vulkan, for instance, does not)
        MGS_LOG_INFO("%s: no GPU backend can pin mapped memory for H2D transfers; --load-mode none may improve "
                     "transfer performance\n",
                     __func__);
      } else if (vHasCpuTensorOverrides && vNRegisteredRanges < vNRegistrationRanges) {
        MGS_LOG_WARN("%s: the GPU backend registered only %zu of %zu mmap ranges containing CPU weights; --load-mode "
                     "none may improve transfer performance\n",
                     __func__, vNRegisteredRanges, vNRegistrationRanges);
      } else if (vHasCpuTensorOverrides && vNRegistrationRanges == 0) {
        MGS_LOG_WARN("%s: CPU weights could not use registered mmap memory; --load-mode none may improve transfer "
                     "performance\n",
                     __func__);
      }
    }
    if (pProgressCallback) {
      // Even though the model is done loading, we still honor
      // cancellation since we need to free allocations.
      return pProgressCallback(1.0f, pProgressCallbackUserData);
    }
  }

  return true;
}

std::string mgs_model_loader::mFtypeName() const { return mgs_ftype_name(vFtype); }

void mgs_model_loader::mPrintInfo() const {
  MGS_LOG_INFO("%s: file format = %s\n", __func__, mgs_file_version_name(vFver));
  MGS_LOG_INFO("%s: file type   = %s\n", __func__, mgs_ftype_name(vFtype));
  if (vNBytes < cGiB) {
    MGS_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, vNBytes / 1024.0 / 1024.0,
                 vNBytes * 8.0 / vNElements);
  } else {
    MGS_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, vNBytes / 1024.0 / 1024.0 / 1024.0,
                 vNBytes * 8.0 / vNElements);
  }
}
