#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#if defined(__MINGW32__) && !defined(__clang__)
#define MGS_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define MGS_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define MGS_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

MGS_ATTRIBUTE_FORMAT(2, 3)
void mgs_log_internal(ggml_log_level pLevel, const char *pFormat, ...);
void mgs_log_callback_default(ggml_log_level pLevel, const char *pText, void *pUserData);

#define MGS_LOG(...) mgs_log_internal(GGML_LOG_LEVEL_NONE, __VA_ARGS__)
#define MGS_LOG_INFO(...) mgs_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define MGS_LOG_WARN(...) mgs_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define MGS_LOG_ERROR(...) mgs_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define MGS_LOG_DEBUG(...) mgs_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define MGS_LOG_CONT(...) mgs_log_internal(GGML_LOG_LEVEL_CONT, __VA_ARGS__)

//
// helpers
//

template <typename T> struct no_init {
  T vValue;
  no_init() = default;
};

template <typename dst_t, typename vSrcT> static inline dst_t mgs_cast(vSrcT pV) {
  if constexpr (std::is_same_v<vSrcT, dst_t>) {
    return pV;
  } else if constexpr (std::is_same_v<vSrcT, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
    return ggml_fp16_to_fp32(pV);
  } else if constexpr (std::is_same_v<vSrcT, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
    return ggml_fp32_to_fp16(pV);
  } else {
    static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
  }
}

static inline ggml_tensor *mgs_mul_mat_hadamard(ggml_context *pCtx, ggml_tensor *pCur, ggml_tensor *pRot) {
  const auto cN = pRot->ne[0];

  ggml_tensor *vRes;

  if (!ggml_is_contiguous(pCur)) {
    vRes = ggml_cont_2d(pCtx, pCur, cN, ggml_nelements(pCur) / cN);
  } else {
    vRes = ggml_reshape_2d(pCtx, pCur, cN, ggml_nelements(pCur) / cN);
  }
  vRes = ggml_mul_mat(pCtx, pRot, vRes);
  ggml_mul_mat_set_hint(vRes, GGML_HINT_SRC0_IS_HADAMARD);
  vRes = ggml_reshape_4d(pCtx, vRes, pCur->ne[0], pCur->ne[1], pCur->ne[2], pCur->ne[3]);

  return vRes;
}

// prefetch the host pages needed to gather these rows
void fMgsPrefetchRows(const ggml_tensor *pTensor, const int32_t *pRows, size_t pNRows);

struct time_meas {
  time_meas(int64_t &pTAcc, bool pDisable = false);
  ~time_meas();

  const int64_t cTStartUs;

  int64_t &vTAcc;
};

template <typename T> struct buffer_view {
  T *vData;
  size_t vSize = 0;

  bool mHasData() const { return vData && vSize > 0; }
};

void fReplaceAll(std::string &pS, const std::string &pSearch, const std::string &pReplace);

// TODO: rename to mgs_format ?
MGS_ATTRIBUTE_FORMAT(1, 2)
std::string fFormat(const char *pFmt, ...);

std::string mgs_format_tensor_shape(const std::vector<int64_t> &pNe);
std::string mgs_format_tensor_shape(const struct ggml_tensor *pT);

std::string fGgufKvToStr(const struct gguf_context *pCtxGguf, int pI);
