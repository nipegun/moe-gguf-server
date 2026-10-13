#pragma once

#include "mgs.h"

#include "mgs-impl.h"
#include "mgs-arch.h"
#include "mgs-hparams.h"
#include "mgs-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

using mgs_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum mgs_fver {
  cGgufFileVersionV3 = 3,
};

const char *mgs_file_version_name(mgs_fver pVersion);

struct mgs_model_loader {
  // Holds information on a model weight
  struct mgs_tensor_weight {
    uint16_t vIdx; // source file index
    size_t vOffs;  // tensor data offset in the original file

    ggml_tensor *vTensor;

    mgs_tensor_weight(const mgs_file *pFile, uint16_t pIdx, const struct gguf_context *pGgufCtx, ggml_tensor *pTensor)
        : vIdx(pIdx), vTensor(pTensor) {
      const int cTensorIdx = gguf_find_tensor(pGgufCtx, ggml_get_name(pTensor));
      if (cTensorIdx < 0) {
        throw std::runtime_error(fFormat("tensor '%s' not found in the model", ggml_get_name(pTensor)));
      }

      vOffs = gguf_get_data_offset(pGgufCtx) + gguf_get_tensor_offset(pGgufCtx, cTensorIdx);
      if (vOffs + ggml_nbytes(pTensor) < vOffs || vOffs + ggml_nbytes(pTensor) > pFile->size()) {
        throw std::runtime_error(
            fFormat("tensor '%s' data is not within the file bounds, model is corrupted or incomplete",
                    ggml_get_name(pTensor)));
      }
    }
  };

  // custom comparator to sort weights more nicely by layer
  struct weight_name_comparer {
    bool operator()(const std::string &pA, const std::string &pB) const {
      int vALayer = -1;
      int vBLayer = -1;
      sscanf(pA.c_str(), "blk.%d.", &vALayer);
      sscanf(pB.c_str(), "blk.%d.", &vBLayer);
      if (vALayer != vBLayer) {
        return vALayer < vBLayer;
      }
      return pA < pB;
    }
  };

  static const int cTensorNotRequired = 1 << 0;
  static const int cTensorDuplicated = 1 << 1;
  static const int cTensorSkip = 1 << 2;
  static const int cTensorSkipIfVirtual = 1 << 3;
  static const int cTensorAllowReshape = 1 << 4;
  // the tensor stays in the model file: the model reads it on demand and it never gets a backend buffer
  static const int cTensorExternal = 1 << 5;

  int vNKv = 0;
  int vNTensors = 0;
  int vNCreated = 0;

  uint64_t vNElements = 0;
  size_t vNBytes = 0;

  bool vUseMmap = false;
  bool vUseDirectIo = false;
  bool vCheckTensors;
  bool vNoAlloc;
  bool vLoadMtp;

  mgs_files vFiles;
  mgs_ftype vFtype;
  mgs_fver vFver;

  mgs_mmaps vMappings;

  std::map<std::string, mgs_tensor_weight, weight_name_comparer> dWeightsMap;
  std::unordered_map<std::string, mgs_model_kv_override> dKvOverrides;
  const mgs_model_tensor_buft_override *vTensorBuftOverrides;
  bool vHasCpuTensorOverrides = false;

  // The MTP planner places heads and output before other weights, considering
  // every GPU. Overrides cannot replace that placement, including its explicit
  // CPU fallback when no selected GPU has enough memory for a group.
  bool vKeepMtpOnDevice = false;
  bool vMtpHeadOnly = false;
  int vMtpFirstLayer = -1;

  gguf_context_ptr vMetadataPtr;
  struct gguf_context *vMetadata; // either metadata_ptr.get() or externally set
  mgs_model_set_tensor_data_t vSetTensorData;
  void *vSetTensorDataUd;
  std::vector<ggml_context_ptr> lContexts;

  std::string vArchName;
  LLM_KV vLlmKv = LLM_KV(cLlmArchUnknown);

  size_t vSizeDone = 0;
  size_t vSizeData = 0;
  // per file index, the offset of the first tensor created with cTensorExternal (prefetching stops there)
  std::map<uint16_t, size_t> dExternalFirstOffs;
  std::vector<std::pair<size_t, size_t>> lMmapsUsed;

  // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t &pLhs, const ggml_backend_buffer_type_t &pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };

  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dCtxMap;
  // Separate contexts make MTP buffers allocate and load before trunk buffers.
  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dMtpContexts;

  // track tensors that had to be moved for debugging:
  size_t vNTensorsMoved = 0;
  std::string vFirstTensorMovedName;
  std::string vFirstTensorMovedTypeName;
  ggml_backend_buffer_type_t vFirstMovedFromBuft = nullptr;
  ggml_backend_buffer_type_t vFirstMovedToBuft = nullptr;

  mgs_model_loader(struct gguf_context *pMetadata, mgs_model_set_tensor_data_t pSetTensorData, void *pSetTensorDataUd,
                   const std::string &pFname,
                   std::vector<std::string> &pSplits, // optional, only need if the split does not follow naming scheme
                   FILE *pFile, mgs_load_mode pLoadMode, bool pCheckTensors, bool pNoAlloc, bool pLoadMtp,
                   const mgs_model_kv_override *pParamOverridesP,
                   const mgs_model_tensor_buft_override *pParamTensorBuftOverridesP);

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, bool>::type mGetArrN(const std::string &pKey, T &pResult,
                                                                           bool pRequired = true);

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, bool>::type mGetArrN(enum llm_kv pKid, T &pResult,
                                                                           bool pRequired = true);

  template <typename T> bool mGetArr(const std::string &pKey, std::vector<T> &pResult, bool pRequired = true);

  template <typename T, size_t N_MAX>
  bool mGetArr(const std::string &pKey, std::array<T, N_MAX> &pResult, bool pRequired = true);

  template <typename T> bool mGetArr(enum llm_kv pKid, T &pResult, bool pRequired = true);

  template <typename T> bool mGetKey(const std::string &pKey, T &pResult, bool pRequired = true);

  template <typename T> bool mGetKey(enum llm_kv pKid, T &pResult, bool pRequired = true);

  template <typename T, size_t N_MAX>
  bool mGetKeyOrArr(const std::string &pKey, std::array<T, N_MAX> &pResult, uint32_t pN, bool pRequired = true);

  template <typename T> bool mGetKeyOrArr(enum llm_kv pKid, T &pResult, uint32_t pN, bool pRequired = true);

  bool mGetKeyOrArr(enum llm_kv pKid, uint32_t &pResult, bool pRequired = true);

  std::string mGetArchName() const;

  enum llm_arch mGetArch() const;

  const mgs_tensor_weight *mGetWeight(const char *pName) const;

  const mgs_tensor_weight &mRequireWeight(const char *pName) const;

  struct ggml_tensor *mGetTensorMeta(const char *pName) const;

  struct ggml_tensor *mRequireTensorMeta(const std::string &pName) const;

  const struct ggml_tensor *mCheckTensorDims(const std::string &pName, const std::vector<int64_t> &pNe, bool pRequired,
                                             bool pAllowReshape) const;

  struct ggml_tensor *mCreateTensor(const mgs_hparams &pHparams, const buft_list_t *pBuftListCpu,
                                    const buft_list_t *pBuftListInput, const buft_list_t *pBuftListOutput,
                                    const buft_list_t *pBuftListLayer, const LLM_TN_IMPL &pTn,
                                    const std::initializer_list<int64_t> &pNe, int pFlags);

  void mDoneGettingTensors(bool pPartial = false) const;

  void mInitMappings(bool pPrefetch = true, mgs_mlocks *pMlockMmaps = nullptr);

  void mGetMappingRange(size_t *pFirst, size_t *pLast, void **pAddr, int pIdx, ggml_context *pCtx) const;

  // loads one tensor into host memory (used by quantization), does not support ggml-backend
  void mLoadDataFor(struct ggml_tensor *pCur) const;

  // Returns false if cancelled by progress_callback
  bool mLoadAllData(struct ggml_context *pCtx, mgs_buf_map &pBufs, mgs_mlocks *pLmlocks,
                    mgs_progress_callback pProgressCallback, void *pProgressCallbackUserData);

  std::string mFtypeName() const;

  void mPrintInfo() const;
};

// DeepSeek-V4.1 files written by the DwarfStar tools describe the model with the keys of the HF config
// (deepseek41.hidden_size, ...); add the standard keys the loaders read (models/deepseek41.cpp)
void fMgsDeepseek41AdaptMetadata(struct gguf_context *pMeta);
