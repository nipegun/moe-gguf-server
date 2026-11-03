#include "imatrix-loader.h"
#include "common.h"
#include "log.h"
#include "gguf.h"

#include <cmath>
#include <cstring>

bool fCommonImatrixLoad(const std::string &pFname, common_imatrix &pImatrix) {
  struct ggml_context *vCtx = nullptr;
  struct gguf_init_params vMetaGgufParams = {
      /* .no_alloc = */ false,
      /* .ctx      = */ &vCtx,
  };
  struct gguf_context *vCtxGguf = gguf_init_from_file(pFname.c_str(), vMetaGgufParams);
  if (!vCtxGguf) {
    LOG_ERR("%s: failed to load GGUF imatrix from %s\n", __func__, pFname.c_str());
    return false;
  }

  const int32_t cNEntries = gguf_get_n_tensors(vCtxGguf);
  if (cNEntries < 1) {
    LOG_ERR("%s: no data in file %s\n", __func__, pFname.c_str());
    gguf_free(vCtxGguf);
    ggml_free(vCtx);
    return false;
  }

  const int64_t cDatasetsKey = gguf_find_key(vCtxGguf, cLlmKvImatrixDatasets);
  const int64_t cChunkCountKey = gguf_find_key(vCtxGguf, cLlmKvImatrixChunkCount);
  const int64_t cChunkSizeKey = gguf_find_key(vCtxGguf, cLlmKvImatrixChunkSize);

  if (cDatasetsKey != -1 && gguf_get_arr_type(vCtxGguf, cDatasetsKey) == GGUF_TYPE_STRING) {
    const int64_t cN = gguf_get_arr_n(vCtxGguf, cDatasetsKey);
    pImatrix.lDatasets.reserve(pImatrix.lDatasets.size() + cN);
    for (int64_t vI = 0; vI < cN; ++vI) {
      pImatrix.lDatasets.push_back(gguf_get_arr_str(vCtxGguf, cDatasetsKey, vI));
    }
  }

  pImatrix.vHasMetadata = (cDatasetsKey != -1 && cChunkCountKey != -1 && cChunkSizeKey != -1);
  pImatrix.vChunkCount = (cChunkCountKey != -1) ? gguf_get_val_u32(vCtxGguf, cChunkCountKey) : 0;
  pImatrix.vChunkSize = (cChunkSizeKey != -1) ? gguf_get_val_u32(vCtxGguf, cChunkSizeKey) : 0;

  const std::string cInSum2Suffix{".in_sum2"};
  const std::string cCountsSuffix{".counts"};

  std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> dSumsCountsFor;

  for (struct ggml_tensor *vCur = ggml_get_first_tensor(vCtx); vCur; vCur = ggml_get_next_tensor(vCtx, vCur)) {
    std::string vName = vCur->name;

    if (vName.empty()) {
      continue;
    }

    if (fStringRemoveSuffix(vName, cInSum2Suffix)) {
      dSumsCountsFor[std::move(vName)].first = vCur;
    } else if (fStringRemoveSuffix(vName, cCountsSuffix)) {
      dSumsCountsFor[std::move(vName)].second = vCur;
    }
  }

  for (const auto &vSc : dSumsCountsFor) {
    const std::string &vName = vSc.first;
    const struct ggml_tensor *vInSum2 = vSc.second.first;
    const struct ggml_tensor *vCounts = vSc.second.second;

    if (!vInSum2 || !vCounts) {
      LOG_ERR("%s: mismatched sums and counts for %s\n", __func__, vName.c_str());
      gguf_free(vCtxGguf);
      ggml_free(vCtx);
      return false;
    }

    auto &vE = pImatrix.dEntries[vName];

    const int64_t cNval = ggml_nelements(vInSum2);
    const int64_t cNcounts = ggml_nelements(vCounts);

    vE.lSums.resize(cNval);
    for (int64_t vJ = 0; vJ < cNval; ++vJ) {
      vE.lSums[vJ] = ((const float *)vInSum2->data)[vJ];
    }

    vE.lCounts.resize(cNcounts);
    for (int64_t vJ = 0; vJ < cNcounts; ++vJ) {
      vE.lCounts[vJ] = std::lround(((const float *)vCounts->data)[vJ]);
    }
  }

  gguf_free(vCtxGguf);
  ggml_free(vCtx);
  return true;
}
