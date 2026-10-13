#include "mgs.h"

#include "mgs-impl.h"

#include "mgs-context.h"
#include "mgs-mmap.h"
#include "mgs-vocab.h"
#include "mgs-model-loader.h"
#include "mgs-model-saver.h"
#include "mgs-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char *mgs_flash_attn_type_name(enum mgs_flash_attn_type pFlashAttnType) {
  switch (pFlashAttnType) {
  case MGS_FLASH_ATTN_TYPE_AUTO:
    return "auto";
  case MGS_FLASH_ATTN_TYPE_DISABLED:
    return "disabled";
  case MGS_FLASH_ATTN_TYPE_ENABLED:
    return "enabled";
  }
  GGML_ABORT("fatal error");
}

const char *mgs_load_mode_name(enum mgs_load_mode pLoadMode) {
  switch (pLoadMode) {
  case MGS_LOAD_MODE_NONE:
    return "none";
  case MGS_LOAD_MODE_MMAP:
    return "mmap";
  case MGS_LOAD_MODE_MLOCK:
    return "mlock";
  case MGS_LOAD_MODE_MMAP_MLOCK:
    return "mmap+mlock";
  case MGS_LOAD_MODE_DIRECT_IO:
    return "dio";
  }
  GGML_ABORT("fatal error");
}

enum mgs_load_mode mgs_load_mode_from_str(const char *pStr) {
  if (std::strcmp(pStr, "none") == 0) {
    return MGS_LOAD_MODE_NONE;
  }
  if (std::strcmp(pStr, "mmap") == 0) {
    return MGS_LOAD_MODE_MMAP;
  }
  if (std::strcmp(pStr, "mlock") == 0) {
    return MGS_LOAD_MODE_MLOCK;
  }
  if (std::strcmp(pStr, "mmap+mlock") == 0) {
    return MGS_LOAD_MODE_MMAP_MLOCK;
  }
  if (std::strcmp(pStr, "dio") == 0) {
    return MGS_LOAD_MODE_DIRECT_IO;
  }
  throw std::invalid_argument(std::string("unknown load mode: ") + pStr);
}

struct mgs_sampler_chain_params mgs_sampler_chain_default_params() {
  struct mgs_sampler_chain_params vResult = {
      /*.no_perf =*/true,
  };

  return vResult;
}

size_t mgs_max_devices(void) { return 16; }

size_t mgs_max_tensor_buft_overrides() { return 4096; }

bool mgs_supports_mmap(void) { return mgs_mmap::cSupported; }

bool mgs_supports_mlock(void) { return mgs_mlock::cSupported; }

bool mgs_supports_gpu_offload(void) {
  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
  return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
         ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr || mgs_supports_rpc();
}

bool mgs_supports_rpc(void) {
  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
  return ggml_backend_reg_by_name("RPC") != nullptr;
}

void mgs_backend_init(void) {
  ggml_time_init();

  // needed to initialize f16 tables
  {
    struct ggml_init_params vParams = {0, NULL, false};
    struct ggml_context *vCtx = ggml_init(vParams);
    ggml_free(vCtx);
  }

  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
}

void mgs_numa_init(enum ggml_numa_strategy pNuma) {
  if (pNuma != GGML_NUMA_STRATEGY_DISABLED) {
    auto *vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(vDev && "CPU backend is not loaded");
    auto *vReg = ggml_backend_dev_backend_reg(vDev);
    auto *vNumaInitFn =
        (decltype(ggml_numa_init) *)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_cpu_numa_init");
    if (vNumaInitFn) {
      vNumaInitFn(pNuma);
    }
  }
}

void mgs_backend_free(void) { ggml_quantize_free(); }

int64_t mgs_time_us(void) { return ggml_time_us(); }

// returns true on success
static bool fMgsPrepareModelDevices(const mgs_model_params &pParams, mgs_model *pModel) {
  // create list of devices to use with this model
  if (pParams.devices) {
    if (pParams.split_mode == MGS_SPLIT_MODE_TENSOR) {
      size_t vNDevs = 0;
      while (pParams.devices[vNDevs]) {
        vNDevs++;
      }
      if (vNDevs == 0) {
        MGS_LOG_ERROR("%s: MGS_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
        return false;
      }
      MGS_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, vNDevs);
      for (size_t vI = 0; vI < vNDevs; ++vI) {
        MGS_LOG_INFO("%s: - device %zu: %s\n", __func__, vI, ggml_backend_dev_name(pParams.devices[vI]));
      }
      pModel->vGetSplitStateUd.vNDevices = vNDevs;
      pModel->vGetSplitStateUd.vModel = pModel;
      pModel->lDevices.push_back(
          {true, ggml_backend_meta_device(pParams.devices, vNDevs, mgs_meta_device_get_split_state,
                                          &pModel->vGetSplitStateUd)});
    } else {
      for (ggml_backend_dev_t *vDev = pParams.devices; *vDev; ++vDev) {
        pModel->lDevices.push_back({false, *vDev});
      }
    }
  } else {
    // default device selection

    // build list of available devices
    std::vector<mgs_device> lGpus;
    std::vector<mgs_device> lIgpus;
    std::vector<mgs_device> lRpcServers;

    if (pParams.split_mode == MGS_SPLIT_MODE_TENSOR) {
      std::vector<ggml_backend_dev_t> lDevs;
      lDevs.reserve(ggml_backend_dev_count());
      for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
        auto *vDev = ggml_backend_dev_get(vI);
        if (ggml_backend_dev_buffer_type(vDev) == ggml_backend_cpu_buffer_type()) {
          MGS_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(vDev),
                       ggml_backend_dev_description(vDev));
          continue;
        }
        lDevs.push_back(vDev);
      }
      if (lDevs.empty()) {
        MGS_LOG_ERROR("%s: MGS_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
        return false;
      }

      MGS_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, lDevs.size());
      for (size_t vI = 0; vI < lDevs.size(); ++vI) {
        MGS_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, vI, ggml_backend_dev_name(lDevs[vI]),
                     ggml_backend_dev_description(lDevs[vI]));
      }

      GGML_ASSERT(!lDevs.empty());
      pModel->vGetSplitStateUd.vNDevices = lDevs.size();
      pModel->vGetSplitStateUd.vModel = pModel;
      lGpus.push_back({true, ggml_backend_meta_device(lDevs.data(), lDevs.size(), mgs_meta_device_get_split_state,
                                                      &pModel->vGetSplitStateUd)});
    } else {
      for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
        ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
        switch (ggml_backend_dev_type(vDev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
          // skip CPU backends since they are handled separately
          break;

        case GGML_BACKEND_DEVICE_TYPE_GPU: {
          ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(vDev);
          if (ggml_backend_reg_name(vReg) == std::string("RPC")) {
            lRpcServers.push_back({false, vDev});
          } else {
            // check if there is already a GPU with the same device id
            ggml_backend_dev_props vProps;
            ggml_backend_dev_get_props(vDev, &vProps);
            auto vIt = std::find_if(lGpus.begin(), lGpus.end(), [&vProps](const mgs_device &pD) {
              ggml_backend_dev_props vDProps;
              ggml_backend_dev_get_props(pD.vDev, &vDProps);
              if (vProps.device_id && vDProps.device_id) {
                return strcmp(vProps.device_id, vDProps.device_id) == 0;
              }
              return false;
            });

            if (vIt != lGpus.end()) {
              MGS_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                           __func__, ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
                           vProps.device_id ? vProps.device_id : "unknown id", ggml_backend_dev_name(vIt->vDev),
                           ggml_backend_dev_description(vIt->vDev));
            } else {
              lGpus.push_back({false, vDev});
            }
          }
          break;
        }

        case GGML_BACKEND_DEVICE_TYPE_IGPU:
          if (lIgpus.empty()) {
            lIgpus.push_back({false, vDev});
          }
          break;
        case GGML_BACKEND_DEVICE_TYPE_META:
          GGML_ABORT("fatal error");
        }
      }
    }

    // add RPC servers at the front of the list to minimize network transfers
    pModel->lDevices.insert(pModel->lDevices.begin(), lRpcServers.begin(), lRpcServers.end());

    // add GPUs
    pModel->lDevices.insert(pModel->lDevices.end(), lGpus.begin(), lGpus.end());

    // add integrated GPUs only if no discrete GPUs were found
    // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
    if (lGpus.empty()) {
      pModel->lDevices.insert(pModel->lDevices.end(), lIgpus.begin(), lIgpus.end());
    }
  }

  // if using single GPU mode, remove all except the main GPU
  if (pParams.split_mode == MGS_SPLIT_MODE_NONE && !pModel->lDevices.empty()) {
    if (pParams.main_gpu < 0) {
      pModel->lDevices.clear();
    } else {
      if (pParams.main_gpu >= (int)pModel->lDevices.size()) {
        MGS_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, pParams.main_gpu,
                      pModel->lDevices.size());
        return false;
      }
      mgs_device vMainGpu = pModel->lDevices[pParams.main_gpu];
      pModel->lDevices.clear();
      pModel->lDevices.push_back(vMainGpu);
    }
  }

  for (const auto &vDev : pModel->lDevices) {
    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDev.vDev, &vProps);
    MGS_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(vDev.vDev),
                 ggml_backend_dev_description(vDev.vDev), vProps.device_id ? vProps.device_id : "unknown id",
                 vProps.memory_free / 1024 / 1024);
  }

  return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via mgs_progress_callback
static std::pair<int, mgs_model *> fMgsModelLoad(struct gguf_context *pMetadata,
                                                 mgs_model_set_tensor_data_t pSetTensorData, void *pSetTensorDataUd,
                                                 const std::string &pFname, std::vector<std::string> &pSplits,
                                                 FILE *pFile, mgs_model_params &pParams) {
  try {
    mgs_model_loader vMl(pMetadata, pSetTensorData, pSetTensorDataUd, pFname, pSplits, pFile, pParams.load_mode,
                         pParams.check_tensors, pParams.no_alloc, pParams.load_mtp, pParams.kv_overrides,
                         pParams.tensor_buft_overrides);

    vMl.mPrintInfo();
    std::unique_ptr<mgs_model> vModelPtr(mgs_model_create(vMl, pParams));

    bool vOk = fMgsPrepareModelDevices(pParams, vModelPtr.get());
    if (!vOk) {
      return {-1, nullptr};
    }

    auto *vModel = dynamic_cast<mgs_model_base *>(vModelPtr.get());
    if (vModel == nullptr) {
      GGML_ABORT("fatal error: model does not implement mgs_model_base");
    }

    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    vModel->vTLoadUs = 0;
    time_meas vTm(vModel->vTLoadUs);

    vModel->vTStartUs = vTm.cTStartUs;

    vModel->vHparams.vVocabOnly = pParams.vocab_only;
    vModel->vHparams.vNoAlloc = pParams.no_alloc;

    try {
      vModel->mLoadHparams(vMl);
    } catch (const std::exception &vE) {
      throw std::runtime_error("error loading model hyperparameters: " + std::string(vE.what()));
    }
    if (vModel->vArch == cLlmArchClip) {
      throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
    }
    try {
      vModel->mLoadVocab(vMl);
    } catch (const std::exception &vE) {
      throw std::runtime_error("error loading model vocabulary: " + std::string(vE.what()));
    }

    vModel->mLoadStats(vMl);
    vModel->mPrintInfo();

    if (pParams.vocab_only) {
      MGS_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
      return {0, vModelPtr.release()};
    }

    if (!vModel->mLoadTensors(vMl)) {
      return {-2, nullptr};
    }

    return {0, vModelPtr.release()};
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error loading model: %s\n", __func__, vErr.what());
    return {-1, nullptr};
  }
}

static struct mgs_model *fMgsModelLoadFromFileImpl(struct gguf_context *pMetadata,
                                                   mgs_model_set_tensor_data_t pSetTensorData, void *pSetTensorDataUd,
                                                   const std::string &pPathModel, std::vector<std::string> &pSplits,
                                                   FILE *pFile, struct mgs_model_params pParams) {
  {
    int vNSourcesDefined = 0;
    if (pMetadata != nullptr) {
      vNSourcesDefined++;
    }
    if (!pPathModel.empty()) {
      vNSourcesDefined++;
    }
    if (pFile != nullptr) {
      vNSourcesDefined++;
    }
    if (vNSourcesDefined != 1) {
      MGS_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
      return nullptr;
    }
  }
  ggml_time_init();

  if (!pParams.vocab_only && ggml_backend_reg_count() == 0) {
    MGS_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a "
                  "backend before calling this function\n",
                  __func__);
    return nullptr;
  }

  unsigned vCurPercentage = 0;
  if (pParams.progress_callback == NULL) {
    pParams.progress_callback_user_data = &vCurPercentage;
    pParams.progress_callback = [](float pProgress, void *pCtx) {
      unsigned *vCurPercentageP = (unsigned *)pCtx;
      unsigned vPercentage = (unsigned)(100 * pProgress);
      while (vPercentage > *vCurPercentageP) {
        *vCurPercentageP = vPercentage;
        MGS_LOG_CONT(".");
        if (vPercentage >= 100) {
          MGS_LOG_CONT("\n");
        }
      }
      return true;
    };
  }

  const auto [vStatus, vModel] =
      fMgsModelLoad(pMetadata, pSetTensorData, pSetTensorDataUd, pPathModel, pSplits, pFile, pParams);
  GGML_ASSERT(vStatus <= 0);
  if (vStatus < 0) {
    if (vStatus == -1) {
      MGS_LOG_ERROR("%s: failed to load model\n", __func__);
    } else if (vStatus == -2) {
      MGS_LOG_INFO("%s: cancelled model load\n", __func__);
    }

    if (vModel) {
      mgs_model_free(vModel);
    }
    return nullptr;
  }

  return vModel;
}

struct mgs_model *mgs_model_init_from_user(struct gguf_context *pMetadata, mgs_model_set_tensor_data_t pSetTensorData,
                                           void *pSetTensorDataUd, struct mgs_model_params pParams) {
  GGML_ASSERT(pMetadata != nullptr);
  std::string vPathModel;
  std::vector<std::string> lSplits = {};
  pParams.load_mode = MGS_LOAD_MODE_NONE;
  pParams.use_extra_bufts = false;
  return fMgsModelLoadFromFileImpl(pMetadata, pSetTensorData, pSetTensorDataUd, vPathModel, lSplits,
                                   /*file*/ nullptr, pParams);
}
struct mgs_model *mgs_model_load_from_file(const char *pPathModel, struct mgs_model_params pParams) {
  std::vector<std::string> lSplits = {};
  return fMgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, pPathModel, lSplits, /*file*/ nullptr, pParams);
}

struct mgs_model *mgs_model_load_from_splits(const char **pPaths, size_t pNPaths, struct mgs_model_params pParams) {
  std::vector<std::string> lSplits;
  if (pNPaths == 0) {
    MGS_LOG_ERROR("%s: list of splits is empty\n", __func__);
    return nullptr;
  }
  lSplits.reserve(pNPaths);
  for (size_t vI = 0; vI < pNPaths; ++vI) {
    lSplits.push_back(pPaths[vI]);
  }
  return fMgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, lSplits.front(), lSplits, /*file*/ nullptr, pParams);
}

struct mgs_model *mgs_model_load_from_file_ptr(FILE *pFile, struct mgs_model_params pParams) {
  if (!pFile) {
    MGS_LOG_ERROR("%s: file is NULL\n", __func__);
    return nullptr;
  }
  std::string vPathModel;
  std::vector<std::string> lSplits = {};
  return fMgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, vPathModel, lSplits, pFile, pParams);
}

void mgs_model_save_to_file(const struct mgs_model *pModel, const char *pPathModel) {
  mgs_model_saver vMs(pModel);
  vMs.mAddKvFromModel();
  vMs.mAddTensorsFromModel();
  vMs.mSave(pPathModel);
}

//
// model split
//

int32_t mgs_split_path(char *pSplitPath, size_t pMaxlen, const char *pPathPrefix, int32_t pSplitNo,
                       int32_t pSplitCount) {

  static const char *const cSplitPathFormat = "%s-%05d-of-%05d.gguf";

  const int cWritten = snprintf(pSplitPath, pMaxlen, cSplitPathFormat, pPathPrefix, pSplitNo + 1, pSplitCount);

  if (cWritten < 0 || (size_t)cWritten >= pMaxlen) {
    return 0;
  }

  return (int32_t)cWritten;
}

int32_t mgs_split_prefix(char *pSplitPrefix, size_t pMaxlen, const char *pSplitPath, int32_t pSplitNo,
                         int32_t pSplitCount) {

  const std::string cStrSplitPath(pSplitPath);

  char aPostfix[32];
  snprintf(aPostfix, sizeof(aPostfix), "-%05d-of-%05d.gguf", pSplitNo + 1, pSplitCount);

  const std::string cStrPostfix(aPostfix);
  if (cStrSplitPath.size() <= cStrPostfix.size()) {
    return 0;
  }

  const size_t cSizePrefix = cStrSplitPath.size() - cStrPostfix.size();

  if (cStrSplitPath.compare(cSizePrefix, std::string::npos, cStrPostfix) == 0) {
    const size_t cCopyLen = std::min(cSizePrefix + 1, pMaxlen);
    snprintf(pSplitPrefix, cCopyLen, "%s", pSplitPath);

    return (int32_t)cSizePrefix;
  }

  return 0;
}

const char *mgs_print_system_info(void) {
  static std::string vS;
  vS.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

  for (size_t vI = 0; vI < ggml_backend_reg_count(); vI++) {
    auto *vReg = ggml_backend_reg_get(vI);
    auto *vGetFeaturesFn =
        (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_get_features");
    if (vGetFeaturesFn) {
      ggml_backend_feature *vFeatures = vGetFeaturesFn(vReg);
      vS += ggml_backend_reg_name(vReg);
      vS += " : ";
      for (; vFeatures->name; vFeatures++) {
        vS += vFeatures->name;
        vS += " = ";
        vS += vFeatures->value;
        vS += " | ";
      }
    }
  }

  return vS.c_str();
}
