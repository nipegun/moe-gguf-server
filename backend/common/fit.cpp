#include "fit.h"

#include "log.h"

#include "../src/mgs-ext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in mgs_params_fit_impl but needs to be defined outside of it to fix a Windows compilation
// issue enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
  cLayerFractionNone = 0, // nothing
  cLayerFractionAttn = 1, // attention
  cLayerFractionUp = 2,   // attention + up
  cLayerFractionGate = 3, // attention + up + gate
  cLayerFractionMoe = 4,  // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

static std::vector<mgs_device_memory_data>
fCommonGetDeviceMemoryDataImpl(const char *pPathModel, const mgs_model_params *pMparams,
                               const mgs_context_params *pCparams, std::vector<ggml_backend_dev_t> &pDevs,
                               uint32_t &pHpNgl, uint32_t &pHpNCtxTrain, uint32_t &pHpNExpert,
                               ggml_log_level pLogLevel) {
  struct user_data_t {
    struct {
      ggml_log_callback vCallback;
      void *vUserData;
    } vOriginalLogger;
    ggml_log_level vMinLevel; // prints below this log level go to debug log
  };
  user_data_t vUd;
  mgs_log_get(&vUd.vOriginalLogger.vCallback, &vUd.vOriginalLogger.vUserData);
  vUd.vMinLevel = pLogLevel;

  mgs_log_set(
      [](ggml_log_level pLevel, const char *pText, void *pUserData) {
        const user_data_t *vUd = (const user_data_t *)pUserData;
        const ggml_log_level cLevelEff = pLevel >= vUd->vMinLevel ? pLevel : GGML_LOG_LEVEL_DEBUG;
        vUd->vOriginalLogger.vCallback(cLevelEff, pText, vUd->vOriginalLogger.vUserData);
      },
      &vUd);

  mgs_model_params vMparamsCopy = *pMparams;
  vMparamsCopy.no_alloc = true;
  vMparamsCopy.load_mode = MGS_LOAD_MODE_NONE;

  mgs_model *vModel = mgs_model_load_from_file(pPathModel, vMparamsCopy);
  if (vModel == nullptr) {
    mgs_log_set(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);
    throw std::runtime_error("failed to load model");
  }

  mgs_context *vCtx = mgs_init_from_model(vModel, *pCparams);
  if (vCtx == nullptr) {
    mgs_model_free(vModel);
    mgs_log_set(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);
    throw std::runtime_error("failed to create mgs_context from model");
  }

  const size_t cNd = mgs_model_n_devices(vModel);
  std::vector<mgs_device_memory_data> lRet(cNd + 1);

  mgs_memory_breakdown vMemoryBreakdown = mgs_get_memory_breakdown(vCtx);

  for (const auto &[vBuft, vMb] : vMemoryBreakdown) {
    if (ggml_backend_buft_is_host(vBuft)) {
      lRet.back().mb.model += vMb.model;
      lRet.back().mb.context += vMb.context;
      lRet.back().mb.compute += vMb.compute;
      continue;
    }

    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft);
    if (!vDev) {
      continue;
    }
    for (size_t vI = 0; vI < cNd; vI++) {
      if (vDev == mgs_model_get_device(vModel, vI)) {
        lRet[vI].mb.model += vMb.model;
        lRet[vI].mb.context += vMb.context;
        lRet[vI].mb.compute += vMb.compute;
        break;
      }
    }
  }

  {
    ggml_backend_dev_t vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (vCpuDev == nullptr) {
      throw std::runtime_error("no CPU backend found");
    }
    size_t vFree;
    size_t vTotal;
    ggml_backend_dev_memory(vCpuDev, &vFree, &vTotal);
    lRet.back().free = vFree;
    lRet.back().total = vTotal;
  }
  for (size_t vI = 0; vI < cNd; vI++) {
    ggml_backend_dev_t vDev = mgs_model_get_device(vModel, vI);

    size_t vFree;
    size_t vTotal;
    ggml_backend_dev_memory(vDev, &vFree, &vTotal);

    // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
    // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
    // not assign anything to a device with an unknown memory budget.
    if (vFree == 0 && vTotal == 0) {
      const enum ggml_backend_dev_type cType = ggml_backend_dev_type(vDev);
      if (cType == GGML_BACKEND_DEVICE_TYPE_GPU || cType == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        LOG_WRN("%s: device %s did not report memory; --fit will not use it\n", __func__, ggml_backend_dev_name(vDev));
      } else {
        vFree = lRet.back().free;
        vTotal = lRet.back().total;
      }
    }
    lRet[vI].free = vFree;
    lRet[vI].total = vTotal;
  }

  pDevs.clear();
  for (int vI = 0; vI < mgs_model_n_devices(vModel); vI++) {
    pDevs.push_back(mgs_model_get_device(vModel, vI));
  }

  // The loader numbers the layers including the nextn (MTP) blocks and offloads the last
  // n_gpu_layers of them, so they count even when MTP is not loaded (their tensors are then
  // skipped); otherwise the first regular layer would be left on the CPU.
  pHpNgl = mgs_model_n_layer(vModel) + mgs_model_n_layer_nextn(vModel);
  pHpNCtxTrain = mgs_model_n_ctx_train(vModel);
  pHpNExpert = mgs_model_n_expert(vModel);

  fCommonMemoryBreakdownPrint(vCtx);

  mgs_free(vCtx);
  mgs_model_free(vModel);
  mgs_log_set(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);

  return lRet;
}

common_device_memory_data_vec fCommonGetDeviceMemoryData(const char *pPathModel, const mgs_model_params *pMparams,
                                                         const mgs_context_params *pCparams,
                                                         std::vector<ggml_backend_dev_t> &pDevs, uint32_t &pHpNgl,
                                                         uint32_t &pHpNCtxTrain, uint32_t &pHpNExpert,
                                                         ggml_log_level pLogLevel) {
  std::vector<mgs_device_memory_data> lImpl = fCommonGetDeviceMemoryDataImpl(
      pPathModel, pMparams, pCparams, pDevs, pHpNgl, pHpNCtxTrain, pHpNExpert, pLogLevel);

  common_device_memory_data_vec vRet(lImpl.size());
  for (size_t vI = 0; vI < lImpl.size(); vI++) {
    vRet[vI].vTotal = lImpl[vI].total;
    vRet[vI].vFree = lImpl[vI].free;
    vRet[vI].vModel = lImpl[vI].mb.model;
    vRet[vI].vContext = lImpl[vI].mb.context;
    vRet[vI].vCompute = lImpl[vI].mb.compute;
  }
  return vRet;
}

// The MoE expert cache is enabled unless MGS_MOE_CACHE (or --no-moe-cache) disables it.
static bool fMoeCacheEnabled() {
  const char *vValue = std::getenv("MGS_MOE_CACHE");
  if (vValue == nullptr) {
    return true;
  }
  const std::string vText = vValue;
  return !(vText == "0" || vText == "off" || vText == "false" || vText == "no");
}

// Memory the system can still give to new allocations, or -1 when it is unknown.
static int64_t fHostAvailableMemory() {
  std::ifstream vMeminfo("/proc/meminfo");
  std::string vKey;
  int64_t vValue = 0;
  std::string vUnit;
  while (vMeminfo >> vKey >> vValue >> vUnit) {
    if (vKey == "MemAvailable:") {
      return vValue * 1024;
    }
  }
  return -1;
}

static void fCommonParamsFitImpl(const char *pPathModel, struct mgs_model_params *pMparams,
                                 struct mgs_context_params *pCparams, float *pTensorSplit,
                                 struct mgs_model_tensor_buft_override *pTensorBuftOverrides, size_t *pMarginsS,
                                 uint32_t pNCtxMin, enum ggml_log_level pLogLevel, bool pAutoBatch) {
  if (pMparams->split_mode == MGS_SPLIT_MODE_TENSOR) {
    throw common_params_fit_exception("mgs_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
  }
  constexpr int64_t cMiB = 1024 * 1024;
  typedef std::vector<mgs_device_memory_data> dmds_t;
  const mgs_model_params cDefaultMparams = mgs_model_default_params();

  std::vector<ggml_backend_dev_t> lDevs;
  uint32_t vHpNgl = 0; // hparams.n_gpu_layers
  uint32_t vHpNct = 0; // hparams.n_ctx_train
  uint32_t vHpNex = 0; // hparams.n_expert

  // step 1: get data for default parameters and check whether any changes are necessary in the first place

  LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
  const dmds_t cDmdsFull =
      fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);
  const size_t cNd = lDevs.size(); // number of devices

  std::vector<int64_t>
      vMargins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
  vMargins.reserve(cNd);
  if (cNd == 0) {
    vMargins.push_back(pMarginsS[0]);
  } else {
    for (size_t vId = 0; vId < cNd; vId++) {
      vMargins.push_back(pMarginsS[vId]);
    }
  }

  std::vector<std::string> lDevNames;
  {
    lDevNames.reserve(cNd);
    size_t vMaxLength = 0;
    for (const auto &vDev : lDevs) {
      std::string vName = ggml_backend_dev_name(vDev);
      vName += " (";
      vName += ggml_backend_dev_description(vDev);
      vName += ")";
      lDevNames.push_back(vName);
      vMaxLength = std::max(vMaxLength, vName.length());
    }
    for (std::string &vDn : lDevNames) {
      vDn.insert(vDn.end(), vMaxLength - vDn.length(), ' ');
    }
  }

  int64_t vSumFree = 0;
  int64_t vSumProjectedFree = 0;
  int64_t vSumProjectedUsed = 0;
  int64_t vSumProjectedModel = 0;
  std::vector<int64_t> lProjectedFreePerDevice;
  lProjectedFreePerDevice.reserve(cNd);

  if (cNd == 0) {
    vSumProjectedUsed = cDmdsFull.back().mb.total();
    vSumFree = cDmdsFull.back().total;
    vSumProjectedFree = vSumFree - vSumProjectedUsed;
    LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n", __func__,
            vSumProjectedUsed / cMiB, vSumFree / cMiB);
    if (vSumProjectedFree >= vMargins[0]) {
      LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n", __func__,
              vSumProjectedFree / cMiB, vMargins[0] / cMiB);
      return;
    }
  } else {
    if (cNd > 1) {
      LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
    }
    for (size_t vId = 0; vId < cNd; vId++) {
      const mgs_device_memory_data &vDmd = cDmdsFull[vId];

      const int64_t cProjectedUsed = vDmd.mb.total();
      const int64_t cProjectedFree = vDmd.free - cProjectedUsed;
      lProjectedFreePerDevice.push_back(cProjectedFree);

      vSumFree += vDmd.free;
      vSumProjectedUsed += cProjectedUsed;
      vSumProjectedFree += cProjectedFree;
      vSumProjectedModel += vDmd.mb.model;

      if (cNd > 1) {
        LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                __func__, lDevNames[vId].c_str(), vDmd.total / cMiB, cProjectedUsed / cMiB, cProjectedFree / cMiB,
                vMargins[vId] / cMiB);
      }
    }
    assert(vSumFree >= 0 && vSumProjectedUsed >= 0);
    LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, vSumProjectedUsed / cMiB, vSumFree / cMiB);
    if (cNd == 1) {
      if (lProjectedFreePerDevice[0] >= vMargins[0]) {
        LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n", __func__,
                lProjectedFreePerDevice[0] / cMiB, vMargins[0] / cMiB);
        return;
      }
    } else {
      bool vChangesNeeded = false;
      for (size_t vId = 0; vId < cNd; vId++) {
        if (lProjectedFreePerDevice[vId] < vMargins[vId]) {
          vChangesNeeded = true;
          break;
        }
      }
      if (!vChangesNeeded) {
        LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
        return;
      }
    }
  }

  // step 2: try reducing memory use by reducing the context size

  bool vContextReduced = false; // the context size was not set by the user and was reduced here
  {
    int64_t vGlobalSurplus = vSumProjectedFree;
    if (cNd == 0) {
      vGlobalSurplus -= vMargins[0];
    } else {
      for (size_t vId = 0; vId < cNd; vId++) {
        vGlobalSurplus -= vMargins[vId];
      }
    }
    if (vGlobalSurplus < 0) {
      if (cNd <= 1) {
        LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64
                " MiB\n",
                __func__, vMargins[0] / cMiB, -vGlobalSurplus / cMiB);
      } else {
        LOG_TRC("%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                __func__, -vGlobalSurplus / cMiB);
      }
      if (pCparams->n_ctx == 0) {
        if (vHpNct > pNCtxMin) {
          int64_t vSumUsedTarget = vSumFree;
          if (cNd == 0) {
            vSumUsedTarget -= vMargins[0];
          } else {
            for (size_t vId = 0; vId < cNd; vId++) {
              vSumUsedTarget -= vMargins[vId];
            }
          }
          if (cNd > 1) {
            // for multiple devices we need to be more conservative in terms of how much context we think can fit:
            //   - for dense models only whole layers can be assigned to devices
            //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a
            //   layer
            //   - on average we expect a waste of 0.5 layers/tensors per device
            //   - use slightly more than the expected average for nd devices to be safe
            const int64_t cModelPerLayer = vSumProjectedModel / std::min(uint32_t(pMparams->n_gpu_layers), vHpNgl);
            vSumUsedTarget -= (cNd + 1) * cModelPerLayer / (vHpNex == 0 ? 2 : 6);
          }

          int64_t vSumProjectedUsedMinCtx = 0;
          pCparams->n_ctx = pNCtxMin;
          vContextReduced = true;
          const dmds_t cDmdsMinCtx =
              fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);
          if (cNd == 0) {
            vSumProjectedUsedMinCtx = cDmdsMinCtx.back().mb.total();
          } else {
            for (size_t vId = 0; vId < cNd; vId++) {
              vSumProjectedUsedMinCtx += cDmdsMinCtx[vId].mb.total();
            }
          }
          if (vSumUsedTarget > vSumProjectedUsedMinCtx) {
            // linear interpolation between minimum and maximum context size:
            pCparams->n_ctx += (vHpNct - pNCtxMin) * (vSumUsedTarget - vSumProjectedUsedMinCtx) /
                               (vSumProjectedUsed - vSumProjectedUsedMinCtx);
            pCparams->n_ctx =
                std::max(pCparams->n_ctx - pCparams->n_ctx % 256, pNCtxMin); // round down context for CUDA backend

            const int64_t cBytesPerCtx = (vSumProjectedUsed - vSumProjectedUsedMinCtx) / (vHpNct - pNCtxMin);
            const int64_t cMemoryReduction = (vHpNct - pCparams->n_ctx) * cBytesPerCtx;
            LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64
                    " MiB less memory in total\n",
                    __func__, vHpNct, pCparams->n_ctx, cMemoryReduction / cMiB);
            if (cNd <= 1) {
              LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
              return;
            }
            LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
          } else {
            const int64_t cMemoryReduction = vSumProjectedUsed - vSumProjectedUsedMinCtx;
            LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64
                    " MiB less memory in total\n",
                    __func__, vHpNct, pCparams->n_ctx, cMemoryReduction / cMiB);
          }
        } else {
          if (pNCtxMin == UINT32_MAX) {
            LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, vHpNct);
          } else {
            LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32
                    " -> no change\n",
                    __func__, vHpNct, pNCtxMin);
          }
        }
      } else {
        LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, pCparams->n_ctx);
      }
    }
  }
  if (cNd == 0) {
    throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
  }

  if (pMparams->n_gpu_layers != cDefaultMparams.n_gpu_layers) {
    throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(pMparams->n_gpu_layers) +
                                      ", abort");
  }
  if (cNd > 1) {
    if (!pTensorSplit) {
      throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
    }
    if (pMparams->tensor_split) {
      for (size_t vId = 0; vId < cNd; vId++) {
        if (pMparams->tensor_split[vId] != 0.0f) {
          throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
        }
      }
    }
    if (pMparams->split_mode == MGS_SPLIT_MODE_ROW) {
      throw common_params_fit_exception("changing weight allocation for MGS_SPLIT_MODE_ROW not implemented, abort");
    }
  }
  if (!pTensorBuftOverrides) {
    throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
  }
  if (pMparams->tensor_buft_overrides &&
      (pMparams->tensor_buft_overrides->pattern || pMparams->tensor_buft_overrides->buft)) {
    throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
  }

  // step 3: iteratively fill the back to front with "dense" layers
  //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
  //   - for a MoE model, same as dense model but with all MoE tensors in system memory

  // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
  auto vGetOverflowPattern = [&](const size_t pIl, const common_layer_fraction_t pLf) -> const char * {
    constexpr size_t cNStrings = 1000;
    if (pIl >= cNStrings) {
      throw std::runtime_error("at most " + std::to_string(cNStrings) + " model layers are supported");
    }
    switch (pLf) {
    case cLayerFractionAttn: {
      static std::array<std::string, cNStrings> aPatterns;
      if (aPatterns[pIl].empty()) {
        aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_(gate|up|gate_up|down).*";
      }
      return aPatterns[pIl].c_str();
    }
    case cLayerFractionUp: {
      static std::array<std::string, cNStrings> aPatterns;
      if (aPatterns[pIl].empty()) {
        aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_(gate|gate_up|down).*";
      }
      return aPatterns[pIl].c_str();
    }
    case cLayerFractionGate: {
      static std::array<std::string, cNStrings> aPatterns;
      if (aPatterns[pIl].empty()) {
        aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_down.*";
      }
      return aPatterns[pIl].c_str();
    }
    case cLayerFractionMoe: {
      static std::array<std::string, cNStrings> aPatterns;
      if (aPatterns[pIl].empty()) {
        aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
      }
      return aPatterns[pIl].c_str();
    }
    default:
      GGML_ABORT("fatal error");
    }
  };

  struct ngl_t {
    uint32_t vNLayer = 0; // number of total layers
    uint32_t vNPart = 0;  // number of partial layers, <= n_layer

    // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
    common_layer_fraction_t vOverflowType = cLayerFractionMoe;

    uint32_t mNFull() const {
      assert(vNLayer >= vNPart);
      return vNLayer - vNPart;
    }
  };

  const size_t cNtbo = mgs_max_tensor_buft_overrides();

  // utility function to set n_gpu_layers and tensor_split
  auto vSetNglTensorSplitTbo = [&](const std::vector<ngl_t> &pNglPerDevice,
                                   const std::vector<ggml_backend_buffer_type_t> &pOverflowBufts,
                                   mgs_model_params &pMparams) {
    pMparams.n_gpu_layers = 0;
    for (size_t vId = 0; vId < cNd; vId++) {
      pMparams.n_gpu_layers += pNglPerDevice[vId].vNLayer;
      if (cNd > 1) {
        pTensorSplit[vId] = pNglPerDevice[vId].vNLayer;
      }
    }
    assert(uint32_t(pMparams.n_gpu_layers) <= vHpNgl + 1);
    uint32_t vIl0 = vHpNgl + 1 - pMparams.n_gpu_layers; // start index for tensor buft overrides

    pMparams.tensor_split = pTensorSplit;

    size_t vItbo = 0;
    for (size_t vId = 0; vId < cNd; vId++) {
      vIl0 += pNglPerDevice[vId].mNFull();
      for (uint32_t vIl = vIl0; vIl < vIl0 + pNglPerDevice[vId].vNPart; vIl++) {
        if (vItbo + 1 >= cNtbo) {
          pTensorBuftOverrides[vItbo].pattern = nullptr;
          pTensorBuftOverrides[vItbo].buft = nullptr;
          vItbo++;
          pMparams.tensor_buft_overrides = pTensorBuftOverrides;
          throw common_params_fit_exception("mgs_max_tensor_buft_overrides() == " + std::to_string(cNtbo) +
                                            " is insufficient for model");
        }
        pTensorBuftOverrides[vItbo].pattern =
            vGetOverflowPattern(vIl, vIl == vIl0 ? pNglPerDevice[vId].vOverflowType : cLayerFractionMoe);
        pTensorBuftOverrides[vItbo].buft = vIl == vIl0 ? pOverflowBufts[vId] : ggml_backend_cpu_buffer_type();
        vItbo++;
      }
      vIl0 += pNglPerDevice[vId].vNPart;
    }
    pTensorBuftOverrides[vItbo].pattern = nullptr;
    pTensorBuftOverrides[vItbo].buft = nullptr;
    vItbo++;
    pMparams.tensor_buft_overrides = pTensorBuftOverrides;
  };

  // utility function that returns the memory use per device for given numbers of layers per device
  auto vGetMemoryForLayers =
      [&](const char *pFuncName, const std::vector<ngl_t> &pNglPerDevice,
          const std::vector<ggml_backend_buffer_type_t> &pOverflowBufts) -> std::vector<int64_t> {
    mgs_model_params vMparamsCopy = *pMparams;
    vSetNglTensorSplitTbo(pNglPerDevice, pOverflowBufts, vMparamsCopy);

    const dmds_t cDmdNl =
        fCommonGetDeviceMemoryDataImpl(pPathModel, &vMparamsCopy, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);

    LOG_TRC("%s: memory for test allocation by device:\n", pFuncName);
    for (size_t vId = 0; vId < cNd; vId++) {
      const ngl_t &vN = pNglPerDevice[vId];
      LOG_TRC("%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
              pFuncName, vId, vN.vNLayer, vN.vNPart, int(vN.vOverflowType), cDmdNl[vId].mb.total() / cMiB);
    }

    std::vector<int64_t> lRet;
    lRet.reserve(cNd);
    for (size_t vId = 0; vId < cNd; vId++) {
      lRet.push_back(cDmdNl[vId].mb.total());
    }
    return lRet;
  };

  int64_t vGlobalSurplusCpuMoe = 0;
  int64_t vHostBytesCpuMoe = -1; // host memory needed with every MoE tensor in system memory
  if (vHpNex > 0) {
    const static std::string cPatternMoeAll =
        "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
    ggml_backend_buffer_type_t vCpuBuft = ggml_backend_cpu_buffer_type();
    pTensorBuftOverrides[0] = {cPatternMoeAll.c_str(), vCpuBuft};
    pTensorBuftOverrides[1] = {nullptr, nullptr};
    pMparams->tensor_buft_overrides = pTensorBuftOverrides;

    LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
    const dmds_t cDmdsCpuMoe =
        fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);

    for (size_t vId = 0; vId < cNd; vId++) {
      vGlobalSurplusCpuMoe += cDmdsCpuMoe[vId].free;
      vGlobalSurplusCpuMoe -= int64_t(cDmdsCpuMoe[vId].mb.total()) + vMargins[vId];
    }
    if (cDmdsCpuMoe.size() > cNd) {
      vHostBytesCpuMoe = int64_t(cDmdsCpuMoe.back().mb.total());
    }

    if (vGlobalSurplusCpuMoe > 0) {
      LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n", __func__,
              vGlobalSurplusCpuMoe / cMiB);
    } else {
      LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
              __func__, -vGlobalSurplusCpuMoe / cMiB);
    }

    // reset
    pTensorBuftOverrides[0] = {nullptr, nullptr};
    pMparams->tensor_buft_overrides = pTensorBuftOverrides;
  }

  std::vector<int64_t> lTargets; // maximum acceptable memory use per device
  lTargets.reserve(cNd);
  for (size_t vId = 0; vId < cNd; vId++) {
    lTargets.push_back(cDmdsFull[vId].free - vMargins[vId]);
    LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, vId, lTargets[vId] / cMiB);
  }

  std::vector<ggml_backend_buffer_type_t>
      vOverflowBufts; // which bufts the first partial layer of a device overflows to:
  vOverflowBufts.reserve(cNd);
  for (size_t vId = 0; vId < cNd; vId++) {
    vOverflowBufts.push_back(ggml_backend_cpu_buffer_type());
  }

  std::vector<ngl_t> lNglPerDevice(cNd);
  std::vector<int64_t> lMem = vGetMemoryForLayers(__func__, lNglPerDevice, vOverflowBufts);

  // Step 2 reduces the context as if every weight had to fit in device memory, which leaves the
  // minimum context whenever the model is larger than the devices. Once the dense weights fit, the
  // experts that do not (all of them with the expert cache) stay in system memory anyway, so the
  // context is raised again until the KV cache and the buffers that grow with it take at most
  // 1/cContextShare of the device memory left after the dense weights; the experts (expert cache
  // or whole expert layers) keep the rest. pMem is the memory use of pNglPerDevice at the current
  // context and is updated when the context changes.
  auto fRaiseContext = [&](const char *pFuncName, const std::vector<ngl_t> &pNglPerDevice, std::vector<int64_t> &pMem) {
    constexpr int64_t cContextShare = 4;
    if (!vContextReduced || pCparams->n_ctx >= vHpNct) {
      return;
    }
    const uint32_t vCtxLow = pCparams->n_ctx;
    pCparams->n_ctx = vHpNct;
    const std::vector<int64_t> lLMemHigh = vGetMemoryForLayers(pFuncName, pNglPerDevice, vOverflowBufts);
    pCparams->n_ctx = vCtxLow;

    // The KV cache grows linearly with the context, so interpolate between both measurements.
    // Without any device memory growing with the context (KV cache in system memory) nothing changes.
    bool vGrows = false;
    double vExtra = double(vHpNct - vCtxLow);
    for (size_t vId = 0; vId < cNd; vId++) {
      const int64_t vBytesRange = lLMemHigh[vId] - pMem[vId];
      if (vBytesRange <= 0) {
        continue;
      }
      vGrows = true;
      const int64_t vBudget = std::max<int64_t>(lTargets[vId] - pMem[vId], 0) / cContextShare;
      vExtra = std::min(vExtra, double(vHpNct - vCtxLow) * double(vBudget) / double(vBytesRange));
    }
    if (!vGrows) {
      return;
    }

    for (int vAttempt = 0; vAttempt < 4; vAttempt++) {
      uint32_t vCtxNew = std::min(vCtxLow + uint32_t(vExtra), vHpNct);
      vCtxNew -= vCtxNew % 256; // round down context for CUDA backend
      if (vCtxNew <= vCtxLow) {
        return;
      }
      pCparams->n_ctx = vCtxNew;
      const std::vector<int64_t> lLMemNew = vGetMemoryForLayers(pFuncName, pNglPerDevice, vOverflowBufts);
      bool vFits = true;
      for (size_t vId = 0; vId < cNd; vId++) {
        vFits = vFits && lLMemNew[vId] <= lTargets[vId];
      }
      if (vFits) {
        pMem = lLMemNew;
        LOG_INF("%s: context size raised from %" PRIu32 " to %" PRIu32
                " tokens, the most that keeps the KV cache within 1/%" PRId64
                " of the device memory left after the dense weights (-c sets it explicitly)\n",
                pFuncName, vCtxLow, vCtxNew, cContextShare);
        return;
      }
      vExtra /= 2;
    }
    pCparams->n_ctx = vCtxLow;
  };

  // Every micro-batch of a prompt streams the experts that are not on the device from system memory, so
  // with experts in system memory the prompt speed grows almost linearly with the micro-batch size (a
  // 2514-token prompt on an RTX 4060 Ti: 215 tokens/s with 512, 408 with 2048 and 737 with 4096), while
  // generation does not depend on it. Unless the batch sizes were given, take the largest micro-batch
  // whose extra compute buffers stay within 1/cBatchShare of the device memory left after the dense
  // weights; the logical batch grows with it. pMem is updated like in fRaiseContext.
  auto fRaiseBatch = [&](const char *pFuncName, const std::vector<ngl_t> &pNglPerDevice, std::vector<int64_t> &pMem) {
    constexpr int64_t cBatchShare = 8;
    if (!pAutoBatch) {
      return;
    }
    const uint32_t vUbatchLow = pCparams->n_ubatch;
    const uint32_t vBatchLow = pCparams->n_batch;
    for (const uint32_t vUbatch : {4096u, 2048u, 1024u}) {
      if (vUbatch <= vUbatchLow) {
        break;
      }
      pCparams->n_ubatch = vUbatch;
      pCparams->n_batch = std::max(vBatchLow, vUbatch);
      const std::vector<int64_t> lLMemNew = vGetMemoryForLayers(pFuncName, pNglPerDevice, vOverflowBufts);
      bool vFits = true;
      for (size_t vId = 0; vId < cNd; vId++) {
        const int64_t vBudget = std::max<int64_t>(lTargets[vId] - pMem[vId], 0) / cBatchShare;
        vFits = vFits && lLMemNew[vId] - pMem[vId] <= vBudget && lLMemNew[vId] <= lTargets[vId];
      }
      if (vFits) {
        pMem = lLMemNew;
        LOG_INF("%s: micro-batch size raised from %" PRIu32 " to %" PRIu32 " tokens (batch %" PRIu32
                "), the largest whose compute buffers stay within 1/%" PRId64
                " of the device memory left after the dense weights (-ub and -b set them explicitly)\n",
                pFuncName, vUbatchLow, pCparams->n_ubatch, pCparams->n_batch, cBatchShare);
        return;
      }
    }
    pCparams->n_ubatch = vUbatchLow;
    pCparams->n_batch = vBatchLow;
  };

  // optimize the number of layers per device using the method of false position:
  //   - ngl_per_device has 0 layers for each device, lower bound
  //   - try a "high" configuration where a device is given all unassigned layers
  //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
  //   - check memory use of our guess, replace either the low or high bound
  //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
  //   - the last device has the output layer, which cannot be a partial layer
  if (vHpNex == 0) {
    LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
  } else {
    LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
  }
  for (int vId = cNd - 1; vId >= 0; vId--) {
    uint32_t vNUnassigned = vHpNgl + 1;
    for (size_t vJd = vId + 1; vJd < cNd; ++vJd) {
      assert(vNUnassigned >= lNglPerDevice[vJd].vNLayer);
      vNUnassigned -= lNglPerDevice[vJd].vNLayer;
    }

    std::vector<ngl_t> lNglPerDeviceHigh = lNglPerDevice;
    lNglPerDeviceHigh[vId].vNLayer = vNUnassigned;
    if (vHpNex > 0) {
      lNglPerDeviceHigh[vId].vNPart =
          size_t(vId) < cNd - 1 ? lNglPerDeviceHigh[vId].vNLayer : lNglPerDeviceHigh[vId].vNLayer - 1;
    }
    if (lNglPerDeviceHigh[vId].vNLayer > 0) {
      std::vector<int64_t> lMemHigh = vGetMemoryForLayers(__func__, lNglPerDeviceHigh, vOverflowBufts);
      if (lMemHigh[vId] > lTargets[vId]) {
        assert(lNglPerDeviceHigh[vId].vNLayer > lNglPerDevice[vId].vNLayer);
        uint32_t vDelta = lNglPerDeviceHigh[vId].vNLayer - lNglPerDevice[vId].vNLayer;
        LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, vId, vDelta);
        while (vDelta > 1) {
          uint32_t vStepSize = int64_t(vDelta) * (lTargets[vId] - lMem[vId]) / (lMemHigh[vId] - lMem[vId]);
          vStepSize = std::max(vStepSize, uint32_t(1));
          vStepSize = std::min(vStepSize, vDelta - 1);

          std::vector<ngl_t> lNglPerDeviceTest = lNglPerDevice;
          lNglPerDeviceTest[vId].vNLayer += vStepSize;
          if (vHpNex) {
            lNglPerDeviceTest[vId].vNPart +=
                size_t(vId) == cNd - 1 && lNglPerDeviceTest[vId].vNPart == 0
                    ? vStepSize - 1
                    : vStepSize; // the first layer is the output layer which must always be full
          }
          const std::vector<int64_t> lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, vOverflowBufts);

          if (lMemTest[vId] <= lTargets[vId]) {
            lNglPerDevice = lNglPerDeviceTest;
            lMem = lMemTest;
            LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, vId, lNglPerDevice[vId].vNLayer);
          } else {
            lNglPerDeviceHigh = lNglPerDeviceTest;
            lMemHigh = lMemTest;
            LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, vId,
                    lNglPerDeviceHigh[vId].vNLayer);
          }
          vDelta = lNglPerDeviceHigh[vId].vNLayer - lNglPerDevice[vId].vNLayer;
        }
      } else {
        assert(lNglPerDeviceHigh[vId].vNLayer == vNUnassigned);
        lNglPerDevice = lNglPerDeviceHigh;
        lMem = lMemHigh;
        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, vId, lNglPerDevice[vId].vNLayer);
      }
    }

    const int64_t cProjectedMargin = cDmdsFull[vId].free - lMem[vId];
    LOG_TRC("%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n", __func__,
            lDevNames[vId].c_str(), lNglPerDevice[vId].vNLayer, lMem[vId] / cMiB, cProjectedMargin / cMiB);
  }
  if (vHpNex == 0 || vGlobalSurplusCpuMoe <= 0) {
    vSetNglTensorSplitTbo(lNglPerDevice, vOverflowBufts, *pMparams);
    return;
  }

  // With the MoE expert cache, the device memory left after the dense weights holds the most
  // recently used experts of every layer, which serves decoding better than whole expert layers.
  // Keep every expert in system memory when it fits there with a margin; otherwise place whole
  // expert layers as usual and let the cache use what remains.
  if (fMoeCacheEnabled() && vHostBytesCpuMoe > 0) {
    const int64_t vAvailable = fHostAvailableMemory();
    const int64_t vHostMargin = std::max<int64_t>(4096 * cMiB, cDmdsFull.back().total / 10);
    if (vAvailable > 0 && vHostBytesCpuMoe + vHostMargin <= vAvailable) {
      LOG_INF("%s: keeping all MoE experts in system memory (%" PRId64 " of %" PRId64
              " MiB available) and leaving the free device memory to the MoE expert cache\n",
              __func__, vHostBytesCpuMoe / cMiB, vAvailable / cMiB);
      // Every GPU caches the experts of its own layers, so the dense-only layers are spread over the
      // GPUs by their memory target instead of being packed into the last ones; the packed placement
      // found above stays when the spread one does not fit.
      if (cNd > 1) {
        const uint32_t vLayers = vHpNgl + 1;
        int64_t vTargetSum = 0;
        for (size_t vId = 0; vId < cNd; vId++) {
          vTargetSum += std::max<int64_t>(lTargets[vId], 0);
        }
        std::vector<ngl_t> lNglSpread(cNd);
        if (vTargetSum > 0) {
          uint32_t vAssigned = 0;
          std::vector<std::pair<double, size_t>> lLRemainders;
          for (size_t vId = 0; vId < cNd; vId++) {
            const double vShare = double(vLayers) * double(std::max<int64_t>(lTargets[vId], 0)) / double(vTargetSum);
            lNglSpread[vId].vNLayer = uint32_t(vShare);
            vAssigned += lNglSpread[vId].vNLayer;
            lLRemainders.push_back({vShare - double(lNglSpread[vId].vNLayer), vId});
          }
          std::sort(lLRemainders.begin(), lLRemainders.end(),
                    [](const auto &pLeft, const auto &pRight) { return pLeft.first > pRight.first; });
          for (size_t vIndex = 0; vAssigned < vLayers; vIndex = (vIndex + 1) % cNd) {
            lNglSpread[lLRemainders[vIndex].second].vNLayer++;
            vAssigned++;
          }
          // The last GPU holds the output layer, which is never a partial layer.
          if (lNglSpread[cNd - 1].vNLayer == 0) {
            for (size_t vId = 0; vId + 1 < cNd; vId++) {
              if (lNglSpread[vId].vNLayer > 1) {
                lNglSpread[vId].vNLayer--;
                lNglSpread[cNd - 1].vNLayer++;
                break;
              }
            }
          }
          for (size_t vId = 0; vId < cNd; vId++) {
            lNglSpread[vId].vNPart = vId < cNd - 1 ? lNglSpread[vId].vNLayer : lNglSpread[vId].vNLayer - 1;
          }
          const std::vector<int64_t> lMemSpread = vGetMemoryForLayers(__func__, lNglSpread, vOverflowBufts);
          bool vFits = lNglSpread[cNd - 1].vNLayer > 0;
          for (size_t vId = 0; vId < cNd; vId++) {
            vFits = vFits && lMemSpread[vId] <= lTargets[vId];
          }
          if (vFits) {
            lNglPerDevice = lNglSpread;
            lMem = lMemSpread;
          } else {
            LOG_INF("%s: the layers do not fit spread over the GPUs by memory; keeping the packed placement\n",
                    __func__);
          }
        }
      }
      fRaiseBatch(__func__, lNglPerDevice, lMem);
      fRaiseContext(__func__, lNglPerDevice, lMem);
      if (cNd > 1) {
        for (size_t vId = 0; vId < cNd; vId++) {
          LOG_INF("%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64
                  " MiB left for the MoE expert cache\n",
                  __func__, lDevNames[vId].c_str(), lNglPerDevice[vId].vNLayer, lMem[vId] / cMiB,
                  (cDmdsFull[vId].free - lMem[vId]) / cMiB);
        }
      }
      vSetNglTensorSplitTbo(lNglPerDevice, vOverflowBufts, *pMparams);
      return;
    }
    LOG_INF("%s: the MoE experts need %" PRId64 " MiB of system memory but only %" PRId64
            " MiB are available; placing whole expert layers on the device instead\n",
            __func__, vHostBytesCpuMoe / cMiB, vAvailable / cMiB);
  }

  // The micro-batch and the context take their shares before the whole expert layers fill the rest of
  // the device memory.
  fRaiseBatch(__func__, lNglPerDevice, lMem);
  fRaiseContext(__func__, lNglPerDevice, lMem);

  // step 4: for a MoE model where all dense tensors fit,
  //     convert the dense-only layers in the back to full layers in the front until all devices are full
  // essentially the same procedure as for the dense-only layers except front-to-back
  // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

  size_t vIdDenseStart = cNd;
  for (int vId = cNd - 1; vId >= 0; vId--) {
    if (lNglPerDevice[vId].vNLayer > 0) {
      vIdDenseStart = vId;
      continue;
    }
    break;
  }
  assert(vIdDenseStart < cNd);

  LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next "
          "device/system memory:\n",
          __func__);
  for (size_t vId = 0; vId <= vIdDenseStart && vIdDenseStart < cNd; vId++) {
    std::vector<ngl_t> lNglPerDeviceHigh = lNglPerDevice;
    for (size_t vJd = vIdDenseStart; vJd < cNd; vJd++) {
      const uint32_t cNLayerMove = vJd < cNd - 1 ? lNglPerDeviceHigh[vJd].vNLayer : lNglPerDeviceHigh[vJd].vNLayer - 1;
      lNglPerDeviceHigh[vId].vNLayer += cNLayerMove;
      lNglPerDeviceHigh[vJd].vNLayer -= cNLayerMove;
      lNglPerDeviceHigh[vJd].vNPart = 0;
    }
    size_t vIdDenseStartHigh = cNd - 1;
    std::vector<int64_t> lMemHigh = vGetMemoryForLayers(__func__, lNglPerDeviceHigh, vOverflowBufts);

    if (lMemHigh[vId] > lTargets[vId]) {
      assert(lNglPerDeviceHigh[vId].mNFull() >= lNglPerDevice[vId].mNFull());
      uint32_t vDelta = lNglPerDeviceHigh[vId].mNFull() - lNglPerDevice[vId].mNFull();
      while (vDelta > 1) {
        uint32_t vStepSize = int64_t(vDelta) * (lTargets[vId] - lMem[vId]) / (lMemHigh[vId] - lMem[vId]);
        vStepSize = std::max(vStepSize, uint32_t(1));
        vStepSize = std::min(vStepSize, vDelta - 1);

        std::vector<ngl_t> lNglPerDeviceTest = lNglPerDevice;
        size_t vIdDenseStartTest = vIdDenseStart;
        uint32_t vNConvertedTest = 0;
        for (; vIdDenseStartTest < cNd; vIdDenseStartTest++) {
          const uint32_t cNConvertJd =
              std::min(vStepSize - vNConvertedTest, lNglPerDeviceTest[vIdDenseStartTest].vNPart);
          lNglPerDeviceTest[vIdDenseStartTest].vNLayer -= cNConvertJd;
          lNglPerDeviceTest[vIdDenseStartTest].vNPart -= cNConvertJd;
          lNglPerDeviceTest[vId].vNLayer += cNConvertJd;
          vNConvertedTest += cNConvertJd;

          if (lNglPerDeviceTest[vIdDenseStartTest].vNPart > 0) {
            break;
          }
        }
        const std::vector<int64_t> lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, vOverflowBufts);

        if (lMemTest[vId] <= lTargets[vId]) {
          lNglPerDevice = lNglPerDeviceTest;
          lMem = lMemTest;
          vIdDenseStart = vIdDenseStartTest;
          LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                  __func__, vId, lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, vIdDenseStart);
        } else {
          lNglPerDeviceHigh = lNglPerDeviceTest;
          lMemHigh = lMemTest;
          vIdDenseStartHigh = vIdDenseStartTest;
          LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32
                  "), id_dense_start_high=%zu\n",
                  __func__, vId, lNglPerDeviceHigh[vId].vNLayer, lNglPerDeviceHigh[vId].vNPart, vIdDenseStartHigh);
        }
        assert(lNglPerDeviceHigh[vId].mNFull() >= lNglPerDevice[vId].mNFull());
        vDelta = lNglPerDeviceHigh[vId].mNFull() - lNglPerDevice[vId].mNFull();
      }
    } else {
      lNglPerDevice = lNglPerDeviceHigh;
      lMem = lMemHigh;
      vIdDenseStart = vIdDenseStartHigh;
      LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
              __func__, vId, lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, vIdDenseStart);
    }

    // try to fit at least part of one more layer
    if (lNglPerDevice[vIdDenseStart].vNLayer > (vId < cNd - 1 ? 0 : 1)) {
      std::vector<ngl_t> lNglPerDeviceTest = lNglPerDevice;
      size_t vIdDenseStartTest = vIdDenseStart;
      lNglPerDeviceTest[vIdDenseStartTest].vNLayer--;
      lNglPerDeviceTest[vIdDenseStartTest].vNPart--;
      lNglPerDeviceTest[vId].vNLayer++;
      lNglPerDeviceTest[vId].vNPart++;
      if (lNglPerDeviceTest[vIdDenseStartTest].vNPart == 0) {
        vIdDenseStartTest++;
      }
      lNglPerDeviceTest[vId].vOverflowType = cLayerFractionUp;
      std::vector<ggml_backend_buffer_type_t> lOverflowBuftsTest = vOverflowBufts;
      if (vId < cNd - 1) {
        lOverflowBuftsTest[vId] = ggml_backend_dev_buffer_type(lDevs[vId + 1]);
      }
      LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
      std::vector<int64_t> lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, lOverflowBuftsTest);
      if (lMemTest[vId] < lTargets[vId] && (vId + 1 == cNd || lMemTest[vId + 1] < lTargets[vId + 1])) {
        lNglPerDevice = lNglPerDeviceTest;
        vOverflowBufts = lOverflowBuftsTest;
        lMem = lMemTest;
        vIdDenseStart = vIdDenseStartTest;
        LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32
                ", UP), id_dense_start=%zu\n",
                __func__, vId, lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, vIdDenseStart);

        lNglPerDeviceTest[vId].vOverflowType = cLayerFractionGate;
        LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
        lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, lOverflowBuftsTest);
        if (lMemTest[vId] < lTargets[vId] && (vId + 1 == cNd || lMemTest[vId + 1] < lTargets[vId + 1])) {
          lNglPerDevice = lNglPerDeviceTest;
          vOverflowBufts = lOverflowBuftsTest;
          lMem = lMemTest;
          vIdDenseStart = vIdDenseStartTest;
          LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32
                  ", GATE), id_dense_start=%zu\n",
                  __func__, vId, lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, vIdDenseStart);
        }
      } else {
        lNglPerDeviceTest[vId].vOverflowType = cLayerFractionAttn;
        LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
        lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, lOverflowBuftsTest);
        if (lMemTest[vId] < lTargets[vId] && (vId + 1 == cNd || lMemTest[vId + 1] < lTargets[vId + 1])) {
          lNglPerDevice = lNglPerDeviceTest;
          vOverflowBufts = lOverflowBuftsTest;
          lMem = lMemTest;
          vIdDenseStart = vIdDenseStartTest;
          LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32
                  ", ATTN), id_dense_start=%zu\n",
                  __func__, vId, lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, vIdDenseStart);
        }
      }
    }

    const int64_t cProjectedMargin = cDmdsFull[vId].free - lMem[vId];
    LOG_TRC("%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64
            " MiB free\n",
            __func__, lDevNames[vId].c_str(), lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, lMem[vId] / cMiB,
            cProjectedMargin / cMiB);
  }

  // print info for devices that were not changed during the conversion from dense only to full layers:
  for (size_t vId = vIdDenseStart + 1; vId < cNd; vId++) {
    const int64_t cProjectedMargin = cDmdsFull[vId].free - lMem[vId];
    LOG_TRC("%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64
            " MiB free\n",
            __func__, lDevNames[vId].c_str(), lNglPerDevice[vId].vNLayer, lNglPerDevice[vId].vNPart, lMem[vId] / cMiB,
            cProjectedMargin / cMiB);
  }

  vSetNglTensorSplitTbo(lNglPerDevice, vOverflowBufts, *pMparams);
}

enum common_params_fit_status fCommonFitParams(const char *pPathModel, mgs_model_params *pMparams,
                                               mgs_context_params *pCparams, float *pTensorSplit,
                                               mgs_model_tensor_buft_override *pTensorBuftOverrides, size_t *pMargins,
                                               uint32_t pNCtxMin, ggml_log_level pLogLevel, bool pAutoBatch) {
  const int64_t cT0Us = mgs_time_us();
  common_params_fit_status vStatus = cCommonParamsFitStatusSuccess;
  try {
    fCommonParamsFitImpl(pPathModel, pMparams, pCparams, pTensorSplit, pTensorBuftOverrides, pMargins, pNCtxMin,
                         pLogLevel, pAutoBatch);
    LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
  } catch (const common_params_fit_exception &vE) {
    LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, vE.what());
    vStatus = cCommonParamsFitStatusFailure;
  } catch (const std::runtime_error &vE) {
    LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, vE.what());
    vStatus = cCommonParamsFitStatusError;
  }
  const int64_t cT1Us = mgs_time_us();
  LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (cT1Us - cT0Us) * 1e-6);
  return vStatus;
}

void fCommonMemoryBreakdownPrint(const struct mgs_context *pCtx) {
  // const auto &lDevices = pCtx->mGetModel().lDevices;
  const auto *vModel = mgs_get_model(pCtx);

  std::vector<ggml_backend_dev_t> lDevices;
  for (int vI = 0; vI < mgs_model_n_devices(vModel); vI++) {
    lDevices.push_back(mgs_model_get_device(vModel, vI));
  }

  mgs_memory_breakdown vMemoryBreakdown = mgs_get_memory_breakdown(pCtx);

  std::vector<std::array<std::string, 9>> lTableData;
  lTableData.reserve(lDevices.size());
  const std::string cTemplateHeader = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
  const std::string cTemplateGpu = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
  const std::string cTemplateOther = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

  lTableData.push_back({cTemplateHeader, "memory breakdown [MiB]", "total", "free", "self", "model", "context",
                        "compute", "unaccounted"});

  constexpr size_t cMiB = 1024 * 1024;
  const std::vector<std::string> lDescPrefixesStrip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

  // track seen buffer types to avoid double counting:
  std::set<ggml_backend_buffer_type_t> sSeenBufferTypes;

  // accumulative memory breakdown for each device and for host:
  std::vector<mgs_memory_breakdown_data> lMbDev(lDevices.size());
  mgs_memory_breakdown_data vMbHost;

  for (const auto &vBuftMb : vMemoryBreakdown) {
    ggml_backend_buffer_type_t vBuft = vBuftMb.first;
    const mgs_memory_breakdown_data &vMb = vBuftMb.second;
    if (ggml_backend_buft_is_host(vBuft)) {
      vMbHost.model += vMb.model;
      vMbHost.context += vMb.context;
      vMbHost.compute += vMb.compute;
      sSeenBufferTypes.insert(vBuft);
      continue;
    }
    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft);
    if (vDev) {
      int vIDev = -1;
      for (size_t vI = 0; vI < lDevices.size(); vI++) {
        if (lDevices[vI] == vDev) {
          vIDev = vI;
          break;
        }
      }
      if (vIDev != -1) {
        lMbDev[vIDev].model += vMb.model;
        lMbDev[vIDev].context += vMb.context;
        lMbDev[vIDev].compute += vMb.compute;
        sSeenBufferTypes.insert(vBuft);
        continue;
      }
    }
  }

  // print memory breakdown for each device:
  for (size_t vI = 0; vI < lDevices.size(); vI++) {
    ggml_backend_dev_t vDev = lDevices[vI];
    mgs_memory_breakdown_data vMb = lMbDev[vI];

    const std::string cName = ggml_backend_dev_name(vDev);
    std::string vDesc = ggml_backend_dev_description(vDev);
    for (const std::string &vPrefix : lDescPrefixesStrip) {
      if (vDesc.length() >= vPrefix.length() && vDesc.substr(0, vPrefix.length()) == vPrefix) {
        vDesc = vDesc.substr(vPrefix.length());
      }
    }

    size_t vFree, vTotal;
    ggml_backend_dev_memory(vDev, &vFree, &vTotal);

    const size_t cSelf = vMb.model + vMb.context + vMb.compute;
    const int64_t cUnaccounted =
        static_cast<int64_t>(vTotal) - static_cast<int64_t>(vFree) - static_cast<int64_t>(cSelf);

    lTableData.push_back({cTemplateGpu, "  - " + cName + " (" + vDesc + ")", std::to_string(vTotal / cMiB),
                          std::to_string(vFree / cMiB), std::to_string(cSelf / cMiB), std::to_string(vMb.model / cMiB),
                          std::to_string(vMb.context / cMiB), std::to_string(vMb.compute / cMiB),
                          std::to_string(cUnaccounted / static_cast<int64_t>(cMiB))});
  }

  // print memory breakdown for host:
  {
    const size_t cSelf = vMbHost.model + vMbHost.context + vMbHost.compute;
    lTableData.push_back({cTemplateOther, "  - Host",
                          "", // total
                          "", // free
                          std::to_string(cSelf / cMiB), std::to_string(vMbHost.model / cMiB),
                          std::to_string(vMbHost.context / cMiB), std::to_string(vMbHost.compute / cMiB),
                          ""}); // unaccounted
  }

  // print memory breakdown for all remaining buffer types:
  for (const auto &vBuftMb : vMemoryBreakdown) {
    ggml_backend_buffer_type_t vBuft = vBuftMb.first;
    const mgs_memory_breakdown_data &vMb = vBuftMb.second;
    if (sSeenBufferTypes.count(vBuft) == 1) {
      continue;
    }
    const std::string cName = ggml_backend_buft_name(vBuft);
    const size_t cSelf = vMb.model + vMb.context + vMb.compute;
    lTableData.push_back({cTemplateOther, "  - " + cName,
                          "", // total
                          "", // free
                          std::to_string(cSelf / cMiB), std::to_string(vMb.model / cMiB),
                          std::to_string(vMb.context / cMiB), std::to_string(vMb.compute / cMiB), ""}); // unaccounted
    sSeenBufferTypes.insert(vBuft);
  }

  for (size_t vJ = 1; vJ < lTableData[0].size(); vJ++) {
    size_t vMaxLen = 0;
    for (const auto &vTd : lTableData) {
      vMaxLen = std::max(vMaxLen, vTd[vJ].length());
    }
    for (auto &vTd : lTableData) {
      vTd[vJ].insert(vJ == 1 ? vTd[vJ].length() : 0, vMaxLen - vTd[vJ].length(), ' ');
    }
  }
  for (const auto &vTd : lTableData) {
    LOG_TRC(vTd[0].c_str(), __func__, vTd[1].c_str(), vTd[2].c_str(), vTd[3].c_str(), vTd[4].c_str(), vTd[5].c_str(),
            vTd[6].c_str(), vTd[7].c_str(), vTd[8].c_str());
  }
}

void fCommonFitPrint(const char *pPathModel, mgs_model_params *pMparams, mgs_context_params *pCparams) {
  std::vector<ggml_backend_dev_t> lDevs;
  uint32_t vHpNgl = 0; // hparams.n_gpu_layers
  uint32_t vHpNct = 0; // hparams.n_ctx_train
  uint32_t vHpNex = 0; // hparams.n_expert

  auto vDmd = fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex,
                                             GGML_LOG_LEVEL_ERROR);
  GGML_ASSERT(vDmd.size() == lDevs.size() + 1);

  for (size_t vId = 0; vId < lDevs.size(); vId++) {
    printf("%s ", ggml_backend_dev_name(lDevs[vId]));
    printf("%zu ", vDmd[vId].mb.model / 1024 / 1024);
    printf("%zu ", vDmd[vId].mb.context / 1024 / 1024);
    printf("%zu ", vDmd[vId].mb.compute / 1024 / 1024);
    printf("\n");
  }

  printf("Host ");
  printf("%zu ", vDmd.back().mb.model / 1024 / 1024);
  printf("%zu ", vDmd.back().mb.context / 1024 / 1024);
  printf("%zu ", vDmd.back().mb.compute / 1024 / 1024);
  printf("\n");
}
