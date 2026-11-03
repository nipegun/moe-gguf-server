#include "mgs-expert-cache.h"

#include "mgs-ext.h"
#include "mgs-impl.h"
#include "mgs-model.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// A GPU that runs layers with host experts: it holds the pools of its groups.
struct mgs_expert_cache_device {
  ggml_backend_t vBackend = nullptr;
  ggml_backend_dev_t vDevice = nullptr;
  size_t vHostBytes = 0; // host experts of the layers it caches
  size_t vPoolBytes = 0;
  double vPcieBandwidth = 0.0; // uploads to this device, in bytes per second
};

// State of one cache group: layers of one device whose per-expert tensors share types and shapes.
struct mgs_expert_cache_group {
  int vDevice = -1;
  std::vector<int> lLayers;
  mgs_expert_tensors vPool;
  size_t vSlotBytes = 0;
  int32_t vSlotCount = 0;
  // The first vEmptySlots slots stay zeroed: position k of a token whose expert runs on the CPU
  // points the device branch to empty slot k, so no slot repeats within a token (device kernels
  // for expert products assume distinct experts per token).
  int32_t vEmptySlots = 0;

  // Links of the lists of the real slots (vEmptySlots .. vSlotCount - 1); every slot belongs to the lists of the
  // layer that owns it (see mgs_expert_cache_layer).
  std::vector<int32_t> lPrev;
  std::vector<int32_t> lNext;
  std::vector<int8_t> lSegment; // -1 unlinked, 0 probation, 1 protected

  std::vector<int32_t> lOwnerLayer;
  std::vector<int32_t> lOwnerExpert;
  std::vector<uint64_t> lStamp;
};

// State of one cached layer.
struct mgs_expert_cache_layer {
  mgs_expert_cache *vCache = nullptr;
  int vLayer = -1;
  int vGroup = -1;
  mgs_expert_tensors vHost;
  int64_t vExpertCount = 0;
  int64_t vExpertsUsed = 0;

  std::vector<int32_t> lSlotOfExpert; // 0 when the expert is not cached
  // Uses of every expert, halved every cHeatDecayTokens tokens: when the upload quota covers only part of the
  // misses, the ones used most recently and often go first.
  std::vector<float> lHeat;

  // The layer owns vSlotShare slots of the pool of its group, kept in a segmented least-recently-used order: a new
  // expert enters the probation list (vHead .. vTail) and moves to the protected list (vProtectedHead ..
  // vProtectedTail) when it is used again; the protected list holds at most vProtectedMax slots and returns its
  // oldest one to the head of the probation list. Victims come from the probation list first, so experts used once
  // cannot evict the ones in regular use, and every layer evicts only its own experts, so a burst of misses in some
  // layers cannot empty the cache of the others. With vProtectedMax 0 both lists behave as one least-recently-used
  // list.
  int32_t vSlotShare = 0;
  int32_t vHead = -1;
  int32_t vTail = -1;
  int32_t vProtectedHead = -1;
  int32_t vProtectedTail = -1;
  int32_t vProtectedCount = 0;
  int32_t vProtectedMax = 0;

  ggml_tensor *vDeviceIds = nullptr; // I32 [vExpertsUsed * max tokens] in device memory
  int32_t *vStagingIds = nullptr;    // host copy uploaded before the expert fills

  // Scratch space reused by every partition.
  std::vector<int32_t> lCount;
  std::vector<int32_t> lPlan;
  std::vector<int32_t> lUnique;
  std::vector<int32_t> lMisses;

  // CPU branch measurement.
  int64_t vPartitionEndUs = 0;
  size_t vCpuBytes = 0;
  double vCpuOverheadUs = 0.0;
};

bool mgs_expert_tensors::mEquals(const mgs_expert_tensors &pOther) const {
  for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
    if (aRoles[vRole] != pOther.aRoles[vRole]) {
      return false;
    }
  }
  return true;
}

static mgs_expert_tensors fLayerExpertTensors(const mgs_layer &pLayer) {
  mgs_expert_tensors vTensors;
  vTensors.aRoles[cMgsExpertRoleUp] = pLayer.vFfnUpExps;
  vTensors.aRoles[cMgsExpertRoleGate] = pLayer.vFfnGateExps;
  vTensors.aRoles[cMgsExpertRoleGateUp] = pLayer.vFfnGateUpExps;
  vTensors.aRoles[cMgsExpertRoleDown] = pLayer.vFfnDownExps;
  vTensors.aRoles[cMgsExpertRoleUpB] = pLayer.vFfnUpExpsB;
  vTensors.aRoles[cMgsExpertRoleGateB] = pLayer.vFfnGateExpsB;
  vTensors.aRoles[cMgsExpertRoleGateUpB] = pLayer.vFfnGateUpExpsB;
  vTensors.aRoles[cMgsExpertRoleDownB] = pLayer.vFfnDownExpsB;
  vTensors.aRoles[cMgsExpertRoleUpS] = pLayer.vFfnUpExpsS;
  vTensors.aRoles[cMgsExpertRoleGateS] = pLayer.vFfnGateExpsS;
  vTensors.aRoles[cMgsExpertRoleDownS] = pLayer.vFfnDownExpsS;
  return vTensors;
}

static bool fIsWeightRole(int pRole) {
  return pRole == cMgsExpertRoleUp || pRole == cMgsExpertRoleGate || pRole == cMgsExpertRoleGateUp ||
         pRole == cMgsExpertRoleDown;
}

// The expert index is the outermost dimension of every per-expert tensor.
static int fExpertDim(const ggml_tensor *pTensor) { return ggml_n_dims(pTensor) - 1; }

static size_t fExpertBytes(const ggml_tensor *pTensor) { return pTensor->nb[fExpertDim(pTensor)]; }

static int64_t fEnvInt64(const char *pName, int64_t pDefault) {
  const char *vValue = getenv(pName);
  if (vValue == nullptr || vValue[0] == '\0') {
    return pDefault;
  }
  return std::strtoll(vValue, nullptr, 10);
}

static double fEnvDouble(const char *pName, double pDefault) {
  const char *vValue = getenv(pName);
  if (vValue == nullptr || vValue[0] == '\0') {
    return pDefault;
  }
  return std::strtod(vValue, nullptr);
}

// Device memory for the pools. A backend may offer a buffer type with a lower eviction priority for
// caches (Vulkan does): when another program needs VRAM, the kernel driver then moves the pool to system
// memory before the weights, the KV cache or the MTP layers. Other backends use their default type.
static ggml_backend_buffer_type_t fPoolBufferType(ggml_backend_dev_t pDevice) {
  ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(pDevice);
  auto fGetCacheType = (ggml_backend_buffer_type_t (*)(ggml_backend_dev_t))ggml_backend_reg_get_proc_address(
      vReg, "ggml_backend_dev_get_cache_buffer_type");
  return fGetCacheType != nullptr ? fGetCacheType(pDevice) : ggml_backend_dev_buffer_type(pDevice);
}

// The device must evaluate the expert matrix product for this weight type.
static bool fDeviceSupportsWeights(ggml_backend_dev_t pDevice, const ggml_tensor *pWeights, int64_t pExpertsUsed) {
  ggml_init_params vParams = {
      /*.mem_size   =*/ggml_tensor_overhead() * 8,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  ggml_context *vCtx = ggml_init(vParams);
  if (vCtx == nullptr) {
    return false;
  }
  ggml_tensor *vWeights = ggml_new_tensor_3d(vCtx, pWeights->type, pWeights->ne[0], pWeights->ne[1], 2);
  ggml_tensor *vInput = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pWeights->ne[0], 1, 1);
  ggml_tensor *vIds = ggml_new_tensor_2d(vCtx, GGML_TYPE_I32, pExpertsUsed, 1);
  ggml_tensor *vProduct = ggml_mul_mat_id(vCtx, vWeights, vInput, vIds);
  const bool vSupported = ggml_backend_dev_supports_op(pDevice, vProduct);
  ggml_free(vCtx);
  return vSupported;
}

static void fPartitionOp(ggml_tensor *pDst, int pIth, int pNth, void *pUserData) {
  GGML_UNUSED(pNth);
  if (pIth != 0) {
    return;
  }
  auto *vLayer = static_cast<mgs_expert_cache_layer *>(pUserData);
  vLayer->vCache->mPartition(*vLayer, pDst);
}

static void fCpuTimerOp(ggml_tensor *pDst, int pIth, int pNth, void *pUserData) {
  GGML_UNUSED(pNth);
  if (pIth != 0) {
    return;
  }
  *(float *)pDst->data = 0.0f;
  auto *vLayer = static_cast<mgs_expert_cache_layer *>(pUserData);
  vLayer->vCache->mMeasureCpu(*vLayer);
}

std::unique_ptr<mgs_expert_cache> mgs_expert_cache::mFCreate(const mgs_model &pModel,
                                                             const std::vector<ggml_backend_t> &pBackends,
                                                             int64_t pRequestedMib, int64_t pReserveMib) {
  const int64_t vRequestedMib = std::max<int64_t>(-1, pRequestedMib);
  if (vRequestedMib == 0) {
    MGS_LOG_DEBUG("%s: MoE expert cache disabled (size 0)\n", __func__);
    return nullptr;
  }

  const auto &vHparams = pModel.vHparams;
  const int64_t vExpertsUsed = vHparams.vNExpertUsed;
  if (vHparams.vNExpert == 0 || vExpertsUsed == 0) {
    return nullptr;
  }

  // Hybrid execution covers the micro-batches whose host weights are not offloaded.
  const int64_t vMaxTokens = fEnvInt64("GGML_OP_OFFLOAD_MIN_BATCH", 32);
  if (vMaxTokens <= 1) {
    return nullptr;
  }

  std::unique_ptr<mgs_expert_cache> vCache(new mgs_expert_cache());
  vCache->vMaxTokens = vMaxTokens;
  vCache->lLayers.resize(pModel.lLayers.size());

  // Select the layers that keep every per-expert tensor in host memory while the
  // layer itself runs on a GPU, and group them by tensor signature.
  std::vector<std::vector<int64_t>> lSignatures;
  size_t vLargestHostExpertTensor = 0;
  int vHostLayers = 0;
  int vOtherDeviceLayers = 0;
  int vUmaLayers = 0;
  int vUnsupportedLayers = 0;

  for (size_t vIndex = 0; vIndex < pModel.lLayers.size(); ++vIndex) {
    const int vLayerIndex = (int)vIndex;
    const mgs_expert_tensors vTensors = fLayerExpertTensors(pModel.lLayers[vIndex]);
    const ggml_tensor *vDown = vTensors.aRoles[cMgsExpertRoleDown];
    if (vDown == nullptr) {
      continue;
    }

    bool vEligible = true;
    for (int vRole = 0; vRole < cMgsExpertRoleCount && vEligible; ++vRole) {
      const ggml_tensor *vTensor = vTensors.aRoles[vRole];
      if (vTensor == nullptr) {
        continue;
      }
      vEligible = vTensor->buffer != nullptr && vTensor->data != nullptr &&
                  ggml_backend_buffer_is_host(vTensor->buffer) && ggml_is_contiguous(vTensor) &&
                  vTensor->ne[fExpertDim(vTensor)] == vDown->ne[2];
      if (vEligible && fIsWeightRole(vRole)) {
        vLargestHostExpertTensor = std::max(vLargestHostExpertTensor, ggml_nbytes(vTensor));
      }
    }
    if (!vEligible) {
      continue;
    }
    ++vHostLayers;

    ggml_backend_dev_t vDevice = pModel.mDevLayer(vLayerIndex);
    const auto vDeviceType = ggml_backend_dev_type(vDevice);
    // An integrated GPU shares the system memory where the host experts already are: a pool there
    // would copy them from RAM to RAM and hold them twice, so its layers keep the regular execution.
    if (vDeviceType == GGML_BACKEND_DEVICE_TYPE_IGPU) {
      ++vUmaLayers;
      continue;
    }
    if (vDeviceType != GGML_BACKEND_DEVICE_TYPE_GPU) {
      ++vOtherDeviceLayers;
      continue;
    }
    // Every GPU caches the layers assigned to it in its own pool.
    int vDeviceIndex = -1;
    for (size_t vCandidate = 0; vCandidate < vCache->lDevices.size(); ++vCandidate) {
      if (vCache->lDevices[vCandidate]->vDevice == vDevice) {
        vDeviceIndex = (int)vCandidate;
        break;
      }
    }
    if (vDeviceIndex < 0) {
      ggml_backend_t vDeviceBackend = nullptr;
      for (ggml_backend_t vBackend : pBackends) {
        if (ggml_backend_get_device(vBackend) == vDevice) {
          vDeviceBackend = vBackend;
          break;
        }
      }
      if (vDeviceBackend == nullptr) {
        ++vOtherDeviceLayers;
        continue;
      }
      auto vEntry = std::make_unique<mgs_expert_cache_device>();
      vEntry->vBackend = vDeviceBackend;
      vEntry->vDevice = vDevice;
      vDeviceIndex = (int)vCache->lDevices.size();
      vCache->lDevices.push_back(std::move(vEntry));
    }

    bool vSupported = true;
    for (int vRole = 0; vRole < cMgsExpertRoleCount && vSupported; ++vRole) {
      if (vTensors.aRoles[vRole] != nullptr && fIsWeightRole(vRole)) {
        vSupported = fDeviceSupportsWeights(vDevice, vTensors.aRoles[vRole], vExpertsUsed);
      }
    }
    if (!vSupported) {
      ++vUnsupportedLayers;
      continue;
    }

    std::vector<int64_t> lSignature = {vDeviceIndex};
    for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
      const ggml_tensor *vTensor = vTensors.aRoles[vRole];
      if (vTensor == nullptr) {
        lSignature.push_back(-1);
        continue;
      }
      lSignature.push_back(vTensor->type);
      for (int vDim = 0; vDim < GGML_MAX_DIMS; ++vDim) {
        lSignature.push_back(vDim == fExpertDim(vTensor) ? 0 : vTensor->ne[vDim]);
      }
    }

    int vGroupIndex = -1;
    for (size_t vCandidate = 0; vCandidate < lSignatures.size(); ++vCandidate) {
      if (lSignatures[vCandidate] == lSignature) {
        vGroupIndex = (int)vCandidate;
        break;
      }
    }
    if (vGroupIndex < 0) {
      vGroupIndex = (int)lSignatures.size();
      lSignatures.push_back(lSignature);
      auto vGroup = std::make_unique<mgs_expert_cache_group>();
      vGroup->vDevice = vDeviceIndex;
      for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
        if (vTensors.aRoles[vRole] != nullptr) {
          vGroup->vSlotBytes += fExpertBytes(vTensors.aRoles[vRole]);
        }
      }
      vCache->lGroups.push_back(std::move(vGroup));
    }
    vCache->lGroups[vGroupIndex]->lLayers.push_back(vLayerIndex);
    vCache->lDevices[vDeviceIndex]->vHostBytes += (size_t)vDown->ne[2] * vCache->lGroups[vGroupIndex]->vSlotBytes;

    auto vLayer = std::make_unique<mgs_expert_cache_layer>();
    vLayer->vCache = vCache.get();
    vLayer->vLayer = vLayerIndex;
    vLayer->vGroup = vGroupIndex;
    vLayer->vHost = vTensors;
    vLayer->vExpertCount = vDown->ne[2];
    vLayer->vExpertsUsed = vExpertsUsed;
    vLayer->lSlotOfExpert.assign(vLayer->vExpertCount, 0);
    vLayer->lHeat.assign(vLayer->vExpertCount, 0.0f);
    vLayer->lCount.assign(vLayer->vExpertCount, 0);
    vLayer->lPlan.assign(vLayer->vExpertCount, 0);
    vCache->lLayers[vIndex] = std::move(vLayer);
  }

  if (vHostLayers > 0) {
    MGS_LOG_INFO("%s: MoE expert cache: %d layers keep their experts in host memory (%d not on a GPU layer, %d on "
                 "an integrated GPU that shares the system memory, %d with unsupported types)\n",
                 __func__, vHostLayers, vOtherDeviceLayers, vUmaLayers, vUnsupportedLayers);
  }
  if (vCache->lGroups.empty()) {
    return nullptr;
  }

  // Device memory budget of every GPU: what is left after weights, KV cache and compute
  // buffers, minus a safety margin and, on the GPU of the first layer with host experts
  // (the one whose expert prefetch for large batches allocates its slots), those slots.
  // A requested size is split among the GPUs by the size of the host experts they cache,
  // so every GPU holds the same share of its experts.
  const int64_t vPrefetchSetting = fEnvInt64("GGML_SCHED_PREFETCH_EXPERTS", 3);
  const size_t vPrefetchSlots = vPrefetchSetting <= 0 ? 0 : (size_t)std::clamp<int64_t>(vPrefetchSetting, 2, 8);
  const size_t vReserveBase = (size_t)std::max<int64_t>(0, pReserveMib) * 1024 * 1024;
  int vPrefetchDevice = -1;
  for (const auto &vLayer : vCache->lLayers) {
    if (vLayer) {
      vPrefetchDevice = vCache->lGroups[vLayer->vGroup]->vDevice;
      break;
    }
  }
  size_t vHostBytesTotal = 0;
  for (const auto &vDevice : vCache->lDevices) {
    vHostBytesTotal += vDevice->vHostBytes;
  }

  // Allocate the pools of every GPU, shrinking them when the device refuses the allocation.
  for (size_t vDeviceIndex = 0; vDeviceIndex < vCache->lDevices.size(); ++vDeviceIndex) {
    mgs_expert_cache_device &vDevice = *vCache->lDevices[vDeviceIndex];
    if (vDevice.vHostBytes == 0) {
      continue;
    }
    size_t vFree = 0;
    size_t vTotal = 0;
    ggml_backend_dev_memory(vDevice.vDevice, &vFree, &vTotal);
    const size_t vReserve =
        vReserveBase + ((int)vDeviceIndex == vPrefetchDevice ? vPrefetchSlots * vLargestHostExpertTensor : 0);

    size_t vBudget = 0;
    if (vRequestedMib > 0) {
      vBudget = (size_t)((double)vRequestedMib * 1024 * 1024 * (double)vDevice.vHostBytes / (double)vHostBytesTotal);
    } else if (vFree > vReserve) {
      vBudget = vFree - vReserve;
    }

    double vWeightTotal = 0.0;
    for (const auto &vGroup : vCache->lGroups) {
      if (vGroup->vDevice == (int)vDeviceIndex) {
        const int64_t vExperts = vCache->lLayers[vGroup->lLayers[0]]->vExpertCount;
        vWeightTotal += (double)vGroup->lLayers.size() * (double)vExperts * (double)vGroup->vSlotBytes;
      }
    }

    bool vAllocated = false;
    bool vTooSmall = false;
    double vScale = 1.0;
    for (int vAttempt = 0; vAttempt < 8 && !vAllocated; ++vAttempt) {
      size_t vPoolBytes = 0;
      bool vAnyGroup = false;
      for (auto &vGroup : vCache->lGroups) {
        if (vGroup->vDevice != (int)vDeviceIndex) {
          continue;
        }
        const int64_t vExperts = vCache->lLayers[vGroup->lLayers[0]]->vExpertCount;
        const double vWeight = (double)vGroup->lLayers.size() * (double)vExperts * (double)vGroup->vSlotBytes;
        const double vShare = vWeightTotal > 0.0 ? (double)vBudget * vScale * vWeight / vWeightTotal : 0.0;
        const int64_t vCapacity = (int64_t)vGroup->lLayers.size() * vExperts;
        const int64_t vSlots = std::min<int64_t>((int64_t)(vShare / (double)vGroup->vSlotBytes), vCapacity);
        // Keep a cache only when it can hold at least the experts of one token.
        vGroup->vEmptySlots = (int32_t)vExpertsUsed;
        vGroup->vSlotCount = vSlots >= vExpertsUsed ? (int32_t)(vSlots + vExpertsUsed) : 0;
        vAnyGroup = vAnyGroup || vGroup->vSlotCount > 0;
        vPoolBytes += (size_t)vGroup->vSlotCount * vGroup->vSlotBytes;
      }
      if (!vAnyGroup) {
        MGS_LOG_INFO("%s: MoE expert cache skipped in %s: %.2f MiB of free device memory is not enough\n", __func__,
                     ggml_backend_dev_name(vDevice.vDevice), vFree / 1024.0 / 1024.0);
        vTooSmall = true;
        break;
      }

      ggml_init_params vParams = {
          /*.mem_size   =*/ggml_tensor_overhead() * (vCache->lGroups.size() * cMgsExpertRoleCount + 1),
          /*.mem_buffer =*/nullptr,
          /*.no_alloc   =*/true,
      };
      ggml_context *vCtx = ggml_init(vParams);
      if (vCtx == nullptr) {
        return nullptr;
      }
      for (auto &vGroup : vCache->lGroups) {
        if (vGroup->vDevice != (int)vDeviceIndex) {
          continue;
        }
        vGroup->vPool = mgs_expert_tensors();
        if (vGroup->vSlotCount == 0) {
          continue;
        }
        const mgs_expert_tensors &vHost = vCache->lLayers[vGroup->lLayers[0]]->vHost;
        for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
          const ggml_tensor *vTensor = vHost.aRoles[vRole];
          if (vTensor == nullptr) {
            continue;
          }
          int64_t aShape[GGML_MAX_DIMS];
          for (int vDim = 0; vDim < GGML_MAX_DIMS; ++vDim) {
            aShape[vDim] = vTensor->ne[vDim];
          }
          aShape[fExpertDim(vTensor)] = vGroup->vSlotCount;
          ggml_tensor *vPool = ggml_new_tensor(vCtx, vTensor->type, ggml_n_dims(vTensor), aShape);
          ggml_format_name(vPool, "expert_cache.%d.%d", (int)(&vGroup - &vCache->lGroups[0]), vRole);
          vGroup->vPool.aRoles[vRole] = vPool;
        }
      }
      ggml_backend_buffer_t vBuffer = ggml_backend_alloc_ctx_tensors_from_buft(vCtx, fPoolBufferType(vDevice.vDevice));
      if (vBuffer == nullptr) {
        ggml_free(vCtx);
        vScale *= 0.85;
        continue;
      }
      ggml_backend_buffer_clear(vBuffer, 0);
      ggml_backend_buffer_set_usage(vBuffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      vCache->lContexts.push_back(vCtx);
      vCache->lBuffers.push_back(vBuffer);
      vDevice.vPoolBytes = ggml_backend_buffer_get_size(vBuffer);
      vAllocated = true;
      vCache->vPrefetchSetAside =
          vCache->vPrefetchSetAside || (vRequestedMib <= 0 && (int)vDeviceIndex == vPrefetchDevice);
      MGS_LOG_INFO("%s: MoE expert cache: %.2f MiB in %s for %zu group(s), %.2f MiB free before allocation\n", __func__,
                   vPoolBytes / 1024.0 / 1024.0, ggml_backend_dev_name(vDevice.vDevice),
                   (size_t)std::count_if(vCache->lGroups.begin(), vCache->lGroups.end(),
                                         [&](const std::unique_ptr<mgs_expert_cache_group> &pGroup) {
                                           return pGroup->vDevice == (int)vDeviceIndex && pGroup->vSlotCount > 0;
                                         }),
                   vFree / 1024.0 / 1024.0);
    }
    if (!vAllocated) {
      // The layers of this GPU keep the regular execution.
      for (auto &vGroup : vCache->lGroups) {
        if (vGroup->vDevice == (int)vDeviceIndex) {
          vGroup->vSlotCount = 0;
          vGroup->vPool = mgs_expert_tensors();
        }
      }
      if (!vTooSmall) {
        MGS_LOG_WARN("%s: MoE expert cache skipped in %s: device allocation failed\n", __func__,
                     ggml_backend_dev_name(vDevice.vDevice));
      }
    }
  }
  if (vCache->lBuffers.empty()) {
    return nullptr;
  }

  // Layers of groups without slots keep the regular execution.
  size_t vCachedLayers = 0;
  for (auto &vLayer : vCache->lLayers) {
    if (vLayer && vCache->lGroups[vLayer->vGroup]->vSlotCount == 0) {
      vLayer.reset();
    }
    vCachedLayers += vLayer ? 1 : 0;
  }

  // Host tensors served from the cache during the expert prefetch of large micro-batches.
  int64_t vMaxExperts = 0;
  for (const auto &vLayer : vCache->lLayers) {
    if (!vLayer) {
      continue;
    }
    vMaxExperts = std::max(vMaxExperts, vLayer->vExpertCount);
    for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
      if (vLayer->vHost.aRoles[vRole] != nullptr) {
        vCache->dHostTensors[vLayer->vHost.aRoles[vRole]] = {vLayer->vLayer, vRole};
      }
    }
  }
  {
    ggml_init_params vParams = {
        /*.mem_size   =*/ggml_tensor_overhead() * (size_t)(2 * vMaxExperts + 4),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    vCache->vViewCtx = ggml_init(vParams);
  }

  // Device identifiers of every cached layer in the memory of its GPU, plus their host staging
  // copies (pinned memory of that GPU when available).
  std::vector<int32_t *> lDeviceStaging(vCache->lDevices.size(), nullptr);
  size_t vFallbackInts = 0;
  for (size_t vDeviceIndex = 0; vDeviceIndex < vCache->lDevices.size(); ++vDeviceIndex) {
    const mgs_expert_cache_device &vDevice = *vCache->lDevices[vDeviceIndex];
    size_t vDeviceLayers = 0;
    for (const auto &vLayer : vCache->lLayers) {
      vDeviceLayers += vLayer && vCache->lGroups[vLayer->vGroup]->vDevice == (int)vDeviceIndex ? 1 : 0;
    }
    if (vDeviceLayers == 0) {
      continue;
    }
    ggml_init_params vParams = {
        /*.mem_size   =*/ggml_tensor_overhead() * (vDeviceLayers + 1),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context *vCtx = ggml_init(vParams);
    if (vCtx == nullptr) {
      return nullptr;
    }
    for (auto &vLayer : vCache->lLayers) {
      if (vLayer && vCache->lGroups[vLayer->vGroup]->vDevice == (int)vDeviceIndex) {
        vLayer->vDeviceIds = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, vLayer->vExpertsUsed * vMaxTokens);
        ggml_format_name(vLayer->vDeviceIds, "expert_cache_ids.%d", vLayer->vLayer);
      }
    }
    ggml_backend_buffer_t vBuffer =
        ggml_backend_alloc_ctx_tensors_from_buft(vCtx, ggml_backend_dev_buffer_type(vDevice.vDevice));
    if (vBuffer == nullptr) {
      ggml_free(vCtx);
      return nullptr;
    }
    ggml_backend_buffer_clear(vBuffer, 0);
    vCache->lContexts.push_back(vCtx);
    vCache->lBuffers.push_back(vBuffer);

    const size_t vStagingInts = vDeviceLayers * (size_t)vExpertsUsed * (size_t)vMaxTokens;
    ggml_backend_buffer_type_t vHostType = ggml_backend_dev_host_buffer_type(vDevice.vDevice);
    if (vHostType != nullptr) {
      ggml_backend_buffer_t vHostBuffer = ggml_backend_buft_alloc_buffer(vHostType, vStagingInts * sizeof(int32_t));
      if (vHostBuffer != nullptr) {
        vCache->lBuffers.push_back(vHostBuffer);
        lDeviceStaging[vDeviceIndex] = (int32_t *)ggml_backend_buffer_get_base(vHostBuffer);
      }
    }
    if (lDeviceStaging[vDeviceIndex] == nullptr) {
      vFallbackInts += vStagingInts;
    }
  }
  vCache->lHostStaging.assign(vFallbackInts, 0);
  int32_t *vFallback = vCache->lHostStaging.data();
  for (size_t vDeviceIndex = 0; vDeviceIndex < vCache->lDevices.size(); ++vDeviceIndex) {
    int32_t *vStaging = lDeviceStaging[vDeviceIndex];
    const bool vUsesFallback = vStaging == nullptr;
    if (vUsesFallback) {
      vStaging = vFallback;
    }
    for (auto &vLayer : vCache->lLayers) {
      if (vLayer && vCache->lGroups[vLayer->vGroup]->vDevice == (int)vDeviceIndex) {
        vLayer->vStagingIds = vStaging;
        vStaging += vLayer->vExpertsUsed * vMaxTokens;
      }
    }
    if (vUsesFallback) {
      vFallback = vStaging;
    }
  }

  // Share of the slots of every layer that its protected list may hold (MGS_MOE_CACHE_PROTECTED; 0 gives a plain
  // least-recently-used list per layer).
  double vProtectedShare = fEnvDouble("MGS_MOE_CACHE_PROTECTED", 0.5);
  vProtectedShare = std::isfinite(vProtectedShare) ? std::clamp(vProtectedShare, 0.0, 1.0) : 0.5;
  for (auto &vGroup : vCache->lGroups) {
    const int32_t vSlots = vGroup->vSlotCount;
    vGroup->lPrev.assign(vSlots, -1);
    vGroup->lNext.assign(vSlots, -1);
    vGroup->lSegment.assign(vSlots, -1);
    vGroup->lOwnerLayer.assign(vSlots, -1);
    vGroup->lOwnerExpert.assign(vSlots, -1);
    vGroup->lStamp.assign(vSlots, 0);
    if (vSlots == 0) {
      continue;
    }
    // Every layer of the group owns an equal share of the real slots.
    const int32_t vLayers = (int32_t)vGroup->lLayers.size();
    const int32_t vRealSlots = vSlots - vGroup->vEmptySlots;
    int32_t vSlot = vGroup->vEmptySlots;
    for (int32_t vIndex = 0; vIndex < vLayers; ++vIndex) {
      mgs_expert_cache_layer &vLayer = *vCache->lLayers[vGroup->lLayers[vIndex]];
      vLayer.vSlotShare = vRealSlots / vLayers + (vIndex < vRealSlots % vLayers ? 1 : 0);
      vLayer.vProtectedMax = (int32_t)((double)vLayer.vSlotShare * vProtectedShare);
      for (const int32_t vEnd = vSlot + vLayer.vSlotShare; vSlot < vEnd; ++vSlot) {
        vCache->mInsert(*vGroup, vLayer, vSlot);
      }
    }
  }

  for (auto &vLayer : vCache->lLayers) {
    if (vLayer) {
      vCache->vFirstLayer = vLayer->vLayer;
      break;
    }
  }

  // Measure the upload bandwidth of every GPU with real experts of its first cached layer;
  // they stay in the cache as its initial contents.
  for (size_t vDeviceIndex = 0; vDeviceIndex < vCache->lDevices.size(); ++vDeviceIndex) {
    mgs_expert_cache_device &vDevice = *vCache->lDevices[vDeviceIndex];
    mgs_expert_cache_layer *vProbeLayer = nullptr;
    for (auto &vLayer : vCache->lLayers) {
      if (vLayer && vCache->lGroups[vLayer->vGroup]->vDevice == (int)vDeviceIndex) {
        vProbeLayer = vLayer.get();
        break;
      }
    }
    if (vProbeLayer == nullptr) {
      continue;
    }
    mgs_expert_cache_layer &vLayer = *vProbeLayer;
    mgs_expert_cache_group &vGroup = *vCache->lGroups[vLayer.vGroup];
    // The probe experts stay within the slots of their layer, so no probe expert evicts another one; when the layer
    // has fewer slots than the probe needs, its experts are uploaded again until the probe reaches its size.
    const int64_t vProbeUploads =
        std::max<int64_t>(1, std::min<int64_t>({(int64_t)vGroup.vSlotCount - vGroup.vEmptySlots, vLayer.vExpertCount,
                                                (int64_t)((256u << 20) / std::max<size_t>(1, vGroup.vSlotBytes))}));
    const int64_t vProbeExperts = std::min<int64_t>(vProbeUploads, vLayer.vSlotShare);
    ggml_backend_synchronize(vDevice.vBackend);
    const int64_t vStart = ggml_time_us();
    int64_t vProbed = 0;
    for (int32_t vExpert = 0; vExpert < (int32_t)vProbeExperts; ++vExpert) {
      const int32_t vSlot = vCache->mFindVictim(vGroup, vLayer, 0);
      if (vSlot <= 0) {
        break;
      }
      vGroup.lOwnerLayer[vSlot] = vLayer.vLayer;
      vGroup.lOwnerExpert[vSlot] = vExpert;
      vLayer.lSlotOfExpert[vExpert] = vSlot;
      vCache->mInsert(vGroup, vLayer, vSlot);
      vCache->mUploadExpert(vLayer, vExpert, vSlot);
      ++vProbed;
    }
    int64_t vUploads = vProbed;
    for (; vProbed > 0 && vUploads < vProbeUploads; ++vUploads) {
      const int32_t vExpert = (int32_t)(vUploads % vProbed);
      vCache->mUploadExpert(vLayer, vExpert, vLayer.lSlotOfExpert[vExpert]);
    }
    ggml_backend_synchronize(vDevice.vBackend);
    const int64_t vElapsed = std::max<int64_t>(1, ggml_time_us() - vStart);
    vDevice.vPcieBandwidth =
        std::clamp((double)vUploads * (double)vGroup.vSlotBytes * 1e6 / (double)vElapsed, 0.5e9, 128e9);
    // The CPU estimate starts from the first measured upload bandwidth and adapts while decoding.
    if (vCache->vHostBandwidth == 0.0) {
      vCache->vHostBandwidth = vDevice.vPcieBandwidth;
    }
  }

  vCache->vFillRatioOverride = fEnvDouble("MGS_MOE_CACHE_FILL_RATIO", -1.0);
  vCache->vStatsInterval = fEnvInt64("MGS_MOE_CACHE_STATS", 0);

  for (const auto &vGroup : vCache->lGroups) {
    if (vGroup->vSlotCount == 0) {
      continue;
    }
    const int64_t vExperts = vCache->lLayers[vGroup->lLayers[0]]->vExpertCount;
    MGS_LOG_INFO("%s: MoE expert cache group in %s: %zu layers, %d of %" PRId64
                 " experts (%.1f%%), %.2f MiB per slot\n",
                 __func__, ggml_backend_dev_name(vCache->lDevices[vGroup->vDevice]->vDevice), vGroup->lLayers.size(),
                 vGroup->vSlotCount - vGroup->vEmptySlots, (int64_t)vGroup->lLayers.size() * vExperts,
                 100.0 * (vGroup->vSlotCount - vGroup->vEmptySlots) / ((double)vGroup->lLayers.size() * vExperts),
                 vGroup->vSlotBytes / 1024.0 / 1024.0);
  }
  for (const auto &vDevice : vCache->lDevices) {
    if (vDevice->vPoolBytes > 0) {
      MGS_LOG_INFO("%s: MoE expert cache: upload bandwidth to %s %.2f GB/s\n", __func__,
                   ggml_backend_dev_name(vDevice->vDevice), vDevice->vPcieBandwidth / 1e9);
    }
  }
  MGS_LOG_INFO("%s: MoE expert cache: %zu cached layers, micro-batches below %" PRId64 " tokens\n", __func__,
               vCachedLayers, vMaxTokens);

  return vCache;
}

mgs_expert_cache::~mgs_expert_cache() {
  for (const auto &vDevice : lDevices) {
    ggml_backend_synchronize(vDevice->vBackend);
  }
  if (vSteps > 0) {
    mPrintStats("final");
  }
  for (ggml_backend_buffer_t vBuffer : lBuffers) {
    ggml_backend_buffer_free(vBuffer);
  }
  for (ggml_context *vCtx : lContexts) {
    ggml_free(vCtx);
  }
  if (vViewCtx != nullptr) {
    ggml_free(vViewCtx);
  }
}

bool mgs_expert_cache::mFUploadWeights(void *pUserData, ggml_backend_t pBackend, const ggml_tensor *pSrc,
                                       ggml_tensor *pDst) {
  return static_cast<mgs_expert_cache *>(pUserData)->mUploadWeights(pBackend, pSrc, pDst);
}

bool mgs_expert_cache::mUploadWeights(ggml_backend_t pBackend, const ggml_tensor *pSrc, ggml_tensor *pDst) {
  const auto vFound = dHostTensors.find(pSrc);
  if (vFound == dHostTensors.end() || vViewCtx == nullptr || pSrc->type != pDst->type ||
      ggml_nbytes(pSrc) != ggml_nbytes(pDst) || !ggml_is_contiguous(pDst)) {
    return false;
  }
  const mgs_expert_cache_layer &vLayer = *lLayers[vFound->second.first];
  // Only the GPU that holds the pool of the layer can copy from it.
  if (ggml_backend_get_device(pBackend) != lDevices[lGroups[vLayer.vGroup]->vDevice]->vDevice) {
    return false;
  }
  const ggml_tensor *vPool = lGroups[vLayer.vGroup]->vPool.aRoles[vFound->second.second];
  const size_t vBytes = fExpertBytes(pSrc);
  const int64_t vElements = ggml_nelements(pSrc) / vLayer.vExpertCount;

  ggml_reset(vViewCtx);
  int64_t vRunStart = -1;
  auto fFlushHostRun = [&](int64_t pEnd) {
    if (vRunStart >= 0) {
      ggml_backend_tensor_set_async(pBackend, pDst, (const char *)pSrc->data + vRunStart * vBytes, vRunStart * vBytes,
                                    (pEnd - vRunStart) * vBytes);
      vUploadHostBytes += (pEnd - vRunStart) * vBytes;
      vRunStart = -1;
    }
  };
  for (int64_t vExpert = 0; vExpert < vLayer.vExpertCount; ++vExpert) {
    const int32_t vSlot = vLayer.lSlotOfExpert[vExpert];
    if (vSlot <= 0) {
      if (vRunStart < 0) {
        vRunStart = vExpert;
      }
      continue;
    }
    fFlushHostRun(vExpert);
    ggml_tensor *vSrcView = ggml_view_1d(vViewCtx, const_cast<ggml_tensor *>(vPool), vElements, vSlot * vBytes);
    ggml_tensor *vDstView = ggml_view_1d(vViewCtx, pDst, vElements, vExpert * vBytes);
    ggml_backend_view_init(vSrcView);
    ggml_backend_view_init(vDstView);
    ggml_backend_tensor_copy_async(pBackend, pBackend, vSrcView, vDstView);
    vUploadDeviceBytes += vBytes;
  }
  fFlushHostRun(vLayer.vExpertCount);
  return true;
}

mgs_expert_cache_layer *mgs_expert_cache::mFindLayer(int pLayer, const mgs_expert_tensors &pTensors,
                                                     int64_t pExpertsUsed) const {
  if (pLayer < 0 || pLayer >= (int)lLayers.size() || !lLayers[pLayer]) {
    return nullptr;
  }
  mgs_expert_cache_layer *vLayer = lLayers[pLayer].get();
  if (!vLayer->vHost.mEquals(pTensors) || pExpertsUsed <= 0 || pExpertsUsed > vLayer->vExpertsUsed) {
    return nullptr;
  }
  return vLayer;
}

const mgs_expert_tensors &mgs_expert_cache::mPoolTensors(const mgs_expert_cache_layer *pLayer) const {
  return lGroups[pLayer->vGroup]->vPool;
}

ggml_backend_t mgs_expert_cache::mBackend(const mgs_expert_cache_layer *pLayer) const {
  return lDevices[lGroups[pLayer->vGroup]->vDevice]->vBackend;
}

ggml_tensor *mgs_expert_cache::mBuildPartition(ggml_context *pCtx, mgs_expert_cache_layer *pLayer,
                                               ggml_tensor *pSelected, ggml_tensor *pInput) const {
  ggml_tensor *aArgs[2] = {pSelected, pInput};
  return ggml_custom_4d(pCtx, GGML_TYPE_I32, pSelected->ne[0], pSelected->ne[1], 2, 1, aArgs, 2, fPartitionOp, 1,
                        pLayer);
}

ggml_tensor *mgs_expert_cache::mBuildDeviceIds(ggml_context *pCtx, mgs_expert_cache_layer *pLayer, int64_t pExpertsUsed,
                                               int64_t pTokens) const {
  // The partition writes the identifiers densely, token by token.
  GGML_ASSERT(pExpertsUsed <= pLayer->vExpertsUsed && pTokens <= vMaxTokens);
  return ggml_view_2d(pCtx, pLayer->vDeviceIds, pExpertsUsed, pTokens, pExpertsUsed * sizeof(int32_t), 0);
}

ggml_tensor *mgs_expert_cache::mBuildCpuTimer(ggml_context *pCtx, mgs_expert_cache_layer *pLayer,
                                              ggml_tensor *pCpuExperts) const {
  ggml_tensor *aArgs[1] = {pCpuExperts};
  return ggml_custom_4d(pCtx, GGML_TYPE_F32, 1, 1, 1, 1, aArgs, 1, fCpuTimerOp, 1, pLayer);
}

// Takes a slot out of the list of its segment among the lists of its layer.
void mgs_expert_cache::mUnlink(mgs_expert_cache_group &pGroup, mgs_expert_cache_layer &pLayer, int32_t pSlot) {
  const int8_t vSegment = pGroup.lSegment[pSlot];
  if (vSegment < 0) {
    return;
  }
  int32_t &vHead = vSegment == 1 ? pLayer.vProtectedHead : pLayer.vHead;
  int32_t &vTail = vSegment == 1 ? pLayer.vProtectedTail : pLayer.vTail;
  const int32_t vPrev = pGroup.lPrev[pSlot];
  const int32_t vNext = pGroup.lNext[pSlot];
  if (vPrev != -1) {
    pGroup.lNext[vPrev] = vNext;
  } else {
    vHead = vNext;
  }
  if (vNext != -1) {
    pGroup.lPrev[vNext] = vPrev;
  } else {
    vTail = vPrev;
  }
  pGroup.lPrev[pSlot] = -1;
  pGroup.lNext[pSlot] = -1;
  pGroup.lSegment[pSlot] = -1;
  if (vSegment == 1) {
    --pLayer.vProtectedCount;
  }
}

// Puts an unlinked slot at the head (the most recently used end) of the list of a segment.
void mgs_expert_cache::mPushFront(mgs_expert_cache_group &pGroup, mgs_expert_cache_layer &pLayer, int32_t pSlot,
                                  int8_t pSegment) {
  int32_t &vHead = pSegment == 1 ? pLayer.vProtectedHead : pLayer.vHead;
  int32_t &vTail = pSegment == 1 ? pLayer.vProtectedTail : pLayer.vTail;
  pGroup.lPrev[pSlot] = -1;
  pGroup.lNext[pSlot] = vHead;
  if (vHead != -1) {
    pGroup.lPrev[vHead] = pSlot;
  }
  vHead = pSlot;
  if (vTail == -1) {
    vTail = pSlot;
  }
  pGroup.lSegment[pSlot] = pSegment;
  if (pSegment == 1) {
    ++pLayer.vProtectedCount;
  }
}

// A slot whose expert is used again moves to the head of the protected list; the oldest protected slots beyond
// vProtectedMax go back to the head of the probation list.
void mgs_expert_cache::mTouch(mgs_expert_cache_group &pGroup, mgs_expert_cache_layer &pLayer, int32_t pSlot) {
  if (pGroup.lSegment[pSlot] == 1 && pLayer.vProtectedHead == pSlot) {
    return;
  }
  mUnlink(pGroup, pLayer, pSlot);
  mPushFront(pGroup, pLayer, pSlot, 1);
  while (pLayer.vProtectedCount > pLayer.vProtectedMax) {
    const int32_t vOldest = pLayer.vProtectedTail;
    mUnlink(pGroup, pLayer, vOldest);
    mPushFront(pGroup, pLayer, vOldest, 0);
  }
}

// A slot that receives a new expert (or starts empty) goes to the head of the probation list.
void mgs_expert_cache::mInsert(mgs_expert_cache_group &pGroup, mgs_expert_cache_layer &pLayer, int32_t pSlot) {
  mUnlink(pGroup, pLayer, pSlot);
  mPushFront(pGroup, pLayer, pSlot, 0);
}

// Least recently used slot of the layer that the current partition does not use, from the probation list first and
// then from the protected list.
int32_t mgs_expert_cache::mFindVictim(const mgs_expert_cache_group &pGroup, const mgs_expert_cache_layer &pLayer,
                                      uint64_t pStamp) const {
  for (const int32_t vTail : {pLayer.vTail, pLayer.vProtectedTail}) {
    int32_t vSlot = vTail;
    while (vSlot != -1 && pStamp != 0 && pGroup.lStamp[vSlot] == pStamp) {
      vSlot = pGroup.lPrev[vSlot];
    }
    if (vSlot != -1) {
      return vSlot;
    }
  }
  return -1;
}

void mgs_expert_cache::mUploadExpert(const mgs_expert_cache_layer &pLayer, int32_t pExpert, int32_t pSlot) {
  const mgs_expert_cache_group &vGroup = *lGroups[pLayer.vGroup];
  ggml_backend_t vBackend = lDevices[vGroup.vDevice]->vBackend;
  for (int vRole = 0; vRole < cMgsExpertRoleCount; ++vRole) {
    const ggml_tensor *vHost = pLayer.vHost.aRoles[vRole];
    if (vHost == nullptr) {
      continue;
    }
    const size_t vBytes = fExpertBytes(vHost);
    ggml_backend_tensor_set_async(vBackend, vGroup.vPool.aRoles[vRole], (const char *)vHost->data + pExpert * vBytes,
                                  pSlot * vBytes, vBytes);
  }
}

// Tokens between two halvings of the expert heats.
static constexpr uint64_t cHeatDecayTokens = 1024;

void mgs_expert_cache::mPartition(mgs_expert_cache_layer &pLayer, ggml_tensor *pDst) {
  const ggml_tensor *vSelected = pDst->src[0];
  const int64_t vExpertsUsed = vSelected->ne[0];
  const int64_t vTokens = vSelected->ne[1];
  mgs_expert_cache_group &vGroup = *lGroups[pLayer.vGroup];
  const mgs_expert_cache_device &vDevice = *lDevices[vGroup.vDevice];
  const uint64_t vCurrentStamp = ++vStamp;

  auto fSelectedExpert = [&](int64_t pSlot, int64_t pToken) {
    return *(const int32_t *)((const char *)vSelected->data + pSlot * vSelected->nb[0] + pToken * vSelected->nb[1]);
  };

  // Count every distinct expert of the micro-batch.
  pLayer.lUnique.clear();
  for (int64_t vToken = 0; vToken < vTokens; ++vToken) {
    for (int64_t vSlot = 0; vSlot < vExpertsUsed; ++vSlot) {
      const int32_t vExpert = fSelectedExpert(vSlot, vToken);
      GGML_ASSERT(vExpert >= 0 && vExpert < pLayer.vExpertCount);
      if (pLayer.lCount[vExpert]++ == 0) {
        pLayer.lUnique.push_back(vExpert);
      }
    }
  }

  // Recent uses of every expert; the heats of every cached layer halve every cHeatDecayTokens tokens.
  if (pLayer.vLayer == vFirstLayer) {
    vHeatTokens += (uint64_t)vTokens;
    if (vHeatTokens >= cHeatDecayTokens) {
      vHeatTokens = 0;
      for (auto &vCachedLayer : lLayers) {
        if (vCachedLayer) {
          for (float &vHeat : vCachedLayer->lHeat) {
            vHeat *= 0.5f;
          }
        }
      }
    }
  }
  for (const int32_t vExpert : pLayer.lUnique) {
    pLayer.lHeat[vExpert] += (float)pLayer.lCount[vExpert];
  }

  // Hits run on the device; misses are uploaded or computed on the CPU.
  pLayer.lMisses.clear();
  for (const int32_t vExpert : pLayer.lUnique) {
    const int32_t vSlot = pLayer.lSlotOfExpert[vExpert];
    if (vSlot > 0) {
      pLayer.lPlan[vExpert] = vSlot;
      vGroup.lStamp[vSlot] = vCurrentStamp;
      mTouch(vGroup, pLayer, vSlot);
    } else {
      pLayer.lPlan[vExpert] = 0;
      pLayer.lMisses.push_back(vExpert);
    }
  }
  vHits += pLayer.lUnique.size() - pLayer.lMisses.size();
  vMisses += pLayer.lMisses.size();

  // Upload as many misses as keep the transfer time balanced with the CPU work for
  // the rest, and at least one so the cache keeps adapting to the routing.
  const int64_t vMissCount = (int64_t)pLayer.lMisses.size();
  int64_t vFillCount = 0;
  if (vMissCount > 0) {
    if (vFillRatioOverride >= 0.0) {
      vFillCount = std::llround((double)vMissCount * std::min(1.0, vFillRatioOverride));
    } else {
      const double vRatio = vDevice.vPcieBandwidth / (vDevice.vPcieBandwidth + vHostBandwidth);
      vFillCount = std::clamp<int64_t>(std::llround((double)vMissCount * vRatio), 1, vMissCount);
    }
  }
  if (vFillCount > 0 && vFillCount < vMissCount) {
    // Prefer the experts selected by more tokens, then the ones used most recently and often.
    std::stable_sort(pLayer.lMisses.begin(), pLayer.lMisses.end(), [&](int32_t pLeft, int32_t pRight) {
      if (pLayer.lCount[pLeft] != pLayer.lCount[pRight]) {
        return pLayer.lCount[pLeft] > pLayer.lCount[pRight];
      }
      return pLayer.lHeat[pLeft] > pLayer.lHeat[pRight];
    });
  }

  int64_t vFilled = 0;
  for (int64_t vIndex = 0; vIndex < vFillCount; ++vIndex) {
    const int32_t vSlot = mFindVictim(vGroup, pLayer, vCurrentStamp);
    if (vSlot <= 0) {
      break;
    }
    const int32_t vOwnerLayer = vGroup.lOwnerLayer[vSlot];
    if (vOwnerLayer >= 0) {
      lLayers[vOwnerLayer]->lSlotOfExpert[vGroup.lOwnerExpert[vSlot]] = 0;
    }
    const int32_t vExpert = pLayer.lMisses[vIndex];
    vGroup.lOwnerLayer[vSlot] = pLayer.vLayer;
    vGroup.lOwnerExpert[vSlot] = vExpert;
    vGroup.lStamp[vSlot] = vCurrentStamp;
    pLayer.lSlotOfExpert[vExpert] = vSlot;
    pLayer.lPlan[vExpert] = vSlot;
    mInsert(vGroup, pLayer, vSlot);
    ++vFilled;
  }

  // Device identifiers (empty slot k for a CPU expert at position k) and CPU identifiers
  // (-1 for device experts).
  int32_t *vCpuIds = (int32_t *)pDst->data;
  int32_t *vScaleIds = vCpuIds + vExpertsUsed * vTokens;
  size_t vCpuCount = 0;
  for (int64_t vToken = 0; vToken < vTokens; ++vToken) {
    for (int64_t vSlot = 0; vSlot < vExpertsUsed; ++vSlot) {
      const int32_t vExpert = fSelectedExpert(vSlot, vToken);
      const int32_t vPlan = pLayer.lPlan[vExpert];
      const int64_t vIndex = vToken * vExpertsUsed + vSlot;
      pLayer.vStagingIds[vIndex] = vPlan > 0 ? vPlan : (int32_t)vSlot;
      vCpuIds[vIndex] = vPlan > 0 ? -1 : vExpert;
      vScaleIds[vIndex] = vExpert;
    }
  }
  for (const int32_t vExpert : pLayer.lUnique) {
    vCpuCount += pLayer.lPlan[vExpert] == 0 ? 1 : 0;
    pLayer.lCount[vExpert] = 0;
  }

  // The identifiers go first so that the upload queue does not delay them behind the fills.
  ggml_backend_tensor_set_async(vDevice.vBackend, pLayer.vDeviceIds, pLayer.vStagingIds, 0,
                                vExpertsUsed * vTokens * sizeof(int32_t));
  for (int64_t vIndex = 0; vIndex < vFilled; ++vIndex) {
    const int32_t vExpert = pLayer.lMisses[vIndex];
    mUploadExpert(pLayer, vExpert, pLayer.lSlotOfExpert[vExpert]);
  }

  vFills += vFilled;
  vCpuExperts += vCpuCount;
  pLayer.vCpuBytes = vCpuCount * vGroup.vSlotBytes;
  pLayer.vPartitionEndUs = ggml_time_us();

  if (pLayer.vLayer == vFirstLayer) {
    ++vSteps;
    if (vStatsInterval > 0 && vSteps % (uint64_t)vStatsInterval == 0) {
      mPrintStats("periodic");
    }
  }
}

void mgs_expert_cache::mMeasureCpu(mgs_expert_cache_layer &pLayer) {
  const double vElapsedUs = (double)(ggml_time_us() - pLayer.vPartitionEndUs);
  if (vElapsedUs <= 0.0) {
    return;
  }
  if (pLayer.vCpuBytes == 0) {
    // Fixed cost of the CPU branch when it has no expert to evaluate.
    pLayer.vCpuOverheadUs = pLayer.vCpuOverheadUs == 0.0 ? vElapsedUs : 0.9 * pLayer.vCpuOverheadUs + 0.1 * vElapsedUs;
    return;
  }
  const double vWorkUs = std::max(vElapsedUs - pLayer.vCpuOverheadUs, 0.25 * vElapsedUs);
  const double vSample = (double)pLayer.vCpuBytes * 1e6 / vWorkUs;
  vHostBandwidth = 0.9 * vHostBandwidth + 0.1 * std::clamp(vSample, 0.1e9, 512e9);
}

void mgs_expert_cache::mFillInfo(mgs_expert_cache_info &pInfo) const {
  pInfo = mgs_expert_cache_info();
  pInfo.active = true;
  pInfo.size = mPoolBytes();
  for (const auto &vGroup : lGroups) {
    if (vGroup->vSlotCount == 0) {
      continue;
    }
    pInfo.n_layer += (int32_t)vGroup->lLayers.size();
    pInfo.n_slot += vGroup->vSlotCount - vGroup->vEmptySlots;
    pInfo.n_expert += (int64_t)vGroup->lLayers.size() * lLayers[vGroup->lLayers[0]]->vExpertCount;
  }
  pInfo.n_step = vSteps;
  pInfo.n_hit = vHits;
  pInfo.n_miss = vMisses;
  pInfo.n_upload = vFills;
  pInfo.n_cpu_expert = vCpuExperts;
  pInfo.prefetch_cache_bytes = vUploadDeviceBytes;
  pInfo.prefetch_host_bytes = vUploadHostBytes;
  pInfo.upload_bandwidth = mUploadBandwidth();
  pInfo.cpu_bandwidth = vHostBandwidth;
}

void mgs_expert_cache::mAddMemory(std::map<ggml_backend_buffer_type_t, size_t> &pMemory) const {
  for (ggml_backend_buffer_t vBuffer : lBuffers) {
    pMemory[ggml_backend_buffer_get_type(vBuffer)] += ggml_backend_buffer_get_size(vBuffer);
  }
}

size_t mgs_expert_cache::mPoolBytes() const {
  size_t vBytes = 0;
  for (const auto &vDevice : lDevices) {
    vBytes += vDevice->vPoolBytes;
  }
  return vBytes;
}

// Mean upload bandwidth of the GPUs that hold a pool.
double mgs_expert_cache::mUploadBandwidth() const {
  double vSum = 0.0;
  int vCount = 0;
  for (const auto &vDevice : lDevices) {
    if (vDevice->vPoolBytes > 0) {
      vSum += vDevice->vPcieBandwidth;
      ++vCount;
    }
  }
  return vCount > 0 ? vSum / vCount : 0.0;
}

void mgs_expert_cache::mPrintStats(const char *pReason) const {
  const uint64_t vLookups = vHits + vMisses;
  const uint64_t vPrefetchBytes = vUploadDeviceBytes + vUploadHostBytes;
  MGS_LOG_INFO("%s: MoE expert cache (%s): %" PRIu64 " steps, hit rate %.1f%%, %" PRIu64 " uploads, %" PRIu64
               " CPU experts, upload %.2f GB/s, CPU %.2f GB/s, prefetch served from the cache %.1f%%\n",
               __func__, pReason, vSteps, vLookups > 0 ? 100.0 * (double)vHits / (double)vLookups : 0.0, vFills,
               vCpuExperts, mUploadBandwidth() / 1e9, vHostBandwidth / 1e9,
               vPrefetchBytes > 0 ? 100.0 * (double)vUploadDeviceBytes / (double)vPrefetchBytes : 0.0);
}
