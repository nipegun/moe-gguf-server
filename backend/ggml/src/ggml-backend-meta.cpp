#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char *ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis pSplitAxis) {
  switch (pSplitAxis) {
  case GGML_BACKEND_SPLIT_AXIS_0:
    return "0";
  case GGML_BACKEND_SPLIT_AXIS_1:
    return "1";
  case GGML_BACKEND_SPLIT_AXIS_2:
    return "2";
  case GGML_BACKEND_SPLIT_AXIS_3:
    return "3";
  case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
    return "MIRRORED";
  case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
    return "PARTIAL";
  case GGML_BACKEND_SPLIT_AXIS_NONE:
    return "NONE";
  case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
    return "UNKNOWN";
  default:
    GGML_ABORT("fatal error");
  }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
  std::vector<ggml_backend_dev_t> vSimpleDevs;
  ggml_backend_meta_get_split_state_t vGetSplitState;
  void *vGetSplitStateUd;

  std::string vName;
  std::string vDescription;

  ggml_backend_meta_device_context(std::vector<ggml_backend_dev_t> pSimpleDevs,
                                   ggml_backend_meta_get_split_state_t pGetSplitState, void *pGetSplitStateUd)
      : vSimpleDevs(std::move(pSimpleDevs)), vGetSplitState(pGetSplitState), vGetSplitStateUd(pGetSplitStateUd) {
    vName = std::string("Meta(");
    vDescription = std::string("Meta(");
    for (size_t vI = 0; vI < pSimpleDevs.size(); vI++) {
      if (vI > 0) {
        vName += ",";
        vDescription += ",";
      }
      vName += ggml_backend_dev_name(pSimpleDevs[vI]);
      vDescription += ggml_backend_dev_description(pSimpleDevs[vI]);
    }
    vName += ")";
    vDescription += ")";
  }

  bool operator<(const ggml_backend_meta_device_context &pOther) const {
    return std::tie(vSimpleDevs, vGetSplitState, vGetSplitStateUd) <
           std::tie(pOther.vSimpleDevs, pOther.vGetSplitState, pOther.vGetSplitStateUd);
  }
};

static bool fGgmlBackendDevIsMeta(ggml_backend_dev_t pDev);

static const char *fGgmlBackendMetaDeviceGetName(ggml_backend_dev_t pDev) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;
  return vMetaDevCtx->vName.c_str();
}

static const char *fGgmlBackendMetaDeviceGetDescription(ggml_backend_dev_t pDev) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;
  return vMetaDevCtx->vDescription.c_str();
}

static void fGgmlBackendMetaDeviceGetMemory(ggml_backend_dev_t pDev, size_t *pFree, size_t *pTotal) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;
  *pFree = 0;
  *pTotal = 0;
  for (ggml_backend_dev_t vDev : vMetaDevCtx->vSimpleDevs) {
    size_t vTmpFree, vTmpTotal;
    ggml_backend_dev_memory(vDev, &vTmpFree, &vTmpTotal);
    *pFree += vTmpFree;
    *pTotal += vTmpTotal;
  }
}

static enum ggml_backend_dev_type fGgmlBackendMetaDeviceGetType(ggml_backend_dev_t pDev) {
  return GGML_BACKEND_DEVICE_TYPE_META;

  GGML_UNUSED(pDev);
}

static void fGgmlBackendMetaDeviceGetProps(ggml_backend_dev_t pDev, ggml_backend_dev_props *pProps) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;

  // TODO replace placeholders
  pProps->name = fGgmlBackendMetaDeviceGetName(pDev);
  pProps->description = fGgmlBackendMetaDeviceGetDescription(pDev);
  pProps->type = fGgmlBackendMetaDeviceGetType(pDev);
  pProps->device_id = 0;

  fGgmlBackendMetaDeviceGetMemory(pDev, &pProps->memory_free, &pProps->memory_total);

  pProps->caps = {
      /* .async                 = */ true,
      /* .host_buffer           = */ false, // Not implemented.
      /* .buffer_from_host_ptr  = */ false, // Not implemented.
      /* .events                = */ false, // Not implemented.
  };
  for (ggml_backend_dev_t vSimpleDev : vMetaDevCtx->vSimpleDevs) {
    ggml_backend_dev_props vTmpProps;
    ggml_backend_dev_get_props(vSimpleDev, &vTmpProps);
    pProps->caps.async = pProps->caps.async && vTmpProps.caps.async;
    pProps->caps.host_buffer = pProps->caps.host_buffer && vTmpProps.caps.host_buffer;
    pProps->caps.buffer_from_host_ptr = pProps->caps.buffer_from_host_ptr && vTmpProps.caps.buffer_from_host_ptr;
    pProps->caps.events = pProps->caps.events && vTmpProps.caps.events;
  }
}

static ggml_backend_t fGgmlBackendMetaDeviceInitBackend(ggml_backend_dev_t pDev, const char *pParams);

static ggml_backend_buffer_type_t fGgmlBackendMetaDeviceGetBufferType(ggml_backend_dev_t pDev);

static ggml_backend_buffer_type_t fGgmlBackendMetaDeviceGetHostBufferType(ggml_backend_dev_t pDev);

static bool fGgmlBackendMetaDeviceSupportsOp(ggml_backend_dev_t pDev, const ggml_tensor *pOp) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;
  return std::all_of(vMetaDevCtx->vSimpleDevs.begin(), vMetaDevCtx->vSimpleDevs.end(),
                     [pOp](ggml_backend_dev_t pSimpleDev) { return ggml_backend_dev_supports_op(pSimpleDev, pOp); });
}

static bool fGgmlBackendMetaDeviceSupportsBuft(ggml_backend_dev_t pDev, ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  ggml_backend_dev_t vDevBuft = ggml_backend_buft_get_device(pBuft);
  if (!fGgmlBackendDevIsMeta(vDevBuft)) {
    return false;
  }
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;
  const ggml_backend_meta_device_context *vMetaBuftDevCtx = (const ggml_backend_meta_device_context *)vDevBuft->context;
  if (vMetaDevCtx->vSimpleDevs.size() != vMetaBuftDevCtx->vSimpleDevs.size()) {
    return false;
  }
  for (size_t vI = 0; vI < vMetaDevCtx->vSimpleDevs.size(); vI++) {
    if (vMetaDevCtx->vSimpleDevs[vI] != vMetaBuftDevCtx->vSimpleDevs[vI]) {
      return false;
    }
  }
  return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ fGgmlBackendMetaDeviceGetName,
    /* .get_description      = */ fGgmlBackendMetaDeviceGetDescription,
    /* .get_memory           = */ fGgmlBackendMetaDeviceGetMemory,
    /* .get_type             = */ fGgmlBackendMetaDeviceGetType,
    /* .get_props            = */ fGgmlBackendMetaDeviceGetProps,
    /* .init_backend         = */ fGgmlBackendMetaDeviceInitBackend,
    /* .get_buffer_type      = */ fGgmlBackendMetaDeviceGetBufferType,
    /* .get_host_buffer_type = */ fGgmlBackendMetaDeviceGetHostBufferType,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ fGgmlBackendMetaDeviceSupportsOp,
    /* .supports_buft        = */ fGgmlBackendMetaDeviceSupportsBuft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool fGgmlBackendDevIsMeta(ggml_backend_dev_t pDev) {
  return pDev != nullptr && pDev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t fGgmlBackendMetaDevNDevs(ggml_backend_dev_t pMetaDev) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pMetaDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pMetaDev->context;
  return vMetaDevCtx->vSimpleDevs.size();
}

static ggml_backend_dev_t fGgmlBackendMetaDevSimpleDev(ggml_backend_dev_t pMetaDev, size_t pIndex) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pMetaDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pMetaDev->context;
  GGML_ASSERT(pIndex < vMetaDevCtx->vSimpleDevs.size());
  return vMetaDevCtx->vSimpleDevs[pIndex];
}

ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t *pDevs, size_t pNDevs,
                                            ggml_backend_meta_get_split_state_t pGetSplitState,
                                            void *pGetSplitStateUd) {
  GGML_ASSERT(pNDevs <= GGML_BACKEND_META_MAX_DEVICES);
  // TODO: this is not thread-safe - needs to be fixed
  static std::vector<std::unique_ptr<ggml_backend_meta_device_context>> lCtxs;
  static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> dMetaDevs;

  std::vector<ggml_backend_dev_t> vSimpleDevs;
  vSimpleDevs.reserve(pNDevs);
  for (size_t vI = 0; vI < pNDevs; vI++) {
    vSimpleDevs.push_back(pDevs[vI]);
  }
  ggml_backend_meta_device_context vCtx(vSimpleDevs, pGetSplitState, pGetSplitStateUd);

  {
    auto vIt = dMetaDevs.find(vCtx);
    if (vIt != dMetaDevs.end()) {
      return &vIt->second;
    }
  }
  lCtxs.push_back(std::make_unique<ggml_backend_meta_device_context>(vCtx));

  struct ggml_backend_device vMetaDev = {
      /*iface  =*/ggml_backend_meta_device_iface,
      /*reg    =*/nullptr,
      /*ctx    =*/lCtxs.back().get(),
  };

  auto vResult = dMetaDevs.emplace(*lCtxs.back(), vMetaDev);
  return &vResult.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
  std::vector<ggml_backend_buffer_type_t> vSimpleBufts;

  std::string vName;

  ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> pSimpleBufts)
      : vSimpleBufts(std::move(pSimpleBufts)) {
    vName = "Meta(";
    for (size_t vI = 0; vI < pSimpleBufts.size(); vI++) {
      if (vI > 0) {
        vName += ",";
      }
      vName += ggml_backend_buft_name(pSimpleBufts[vI]);
    }
    vName += ")";
  }

  bool operator<(const ggml_backend_meta_buffer_type_context &pOther) const {
    return vSimpleBufts < pOther.vSimpleBufts;
  }
};

static size_t fGgmlBackendMetaBuftNBufts(ggml_backend_buffer_type_t pMetaBuft) {
  GGML_ASSERT(ggml_backend_buft_is_meta(pMetaBuft));
  const ggml_backend_meta_buffer_type_context *vMetaBuftCtx =
      (const ggml_backend_meta_buffer_type_context *)pMetaBuft->context;
  return vMetaBuftCtx->vSimpleBufts.size();
}

static const char *fGgmlBackendMetaBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(ggml_backend_buft_is_meta(pBuft));
  const ggml_backend_meta_buffer_type_context *vMetaBuftCtx =
      (const ggml_backend_meta_buffer_type_context *)pBuft->context;
  return vMetaBuftCtx->vName.c_str();
}

static ggml_backend_buffer_type_t fGgmlBackendMetaBuftSimpleBuft(ggml_backend_buffer_type_t pMetaBuft, size_t pIndex) {
  GGML_ASSERT(ggml_backend_buft_is_meta(pMetaBuft));
  const ggml_backend_meta_buffer_type_context *vMetaBuftCtx =
      (const ggml_backend_meta_buffer_type_context *)pMetaBuft->context;
  GGML_ASSERT(pIndex < vMetaBuftCtx->vSimpleBufts.size());
  return vMetaBuftCtx->vSimpleBufts[pIndex];
}

static ggml_backend_buffer_t fGgmlBackendMetaBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize);

static size_t fGgmlBackendMetaBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);
  size_t vMaxAlignment = 1;
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    const size_t cAlignment = ggml_backend_buft_get_alignment(fGgmlBackendMetaBuftSimpleBuft(pBuft, vI));
    vMaxAlignment = std::max(vMaxAlignment, cAlignment);
    GGML_ASSERT(vMaxAlignment % cAlignment == 0);
  }
  return vMaxAlignment;
}

static size_t fGgmlBackendMetaBufferTypeGetMaxSize(ggml_backend_buffer_type_t pBuft) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);
  size_t vMaxSize = SIZE_MAX;
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    vMaxSize = std::min(vMaxSize, ggml_backend_buft_get_max_size(fGgmlBackendMetaBuftSimpleBuft(pBuft, vI)));
  }
  return vMaxSize;
}

static size_t fGgmlBackendMetaBufferTypeGetAllocSize(ggml_backend_buffer_type_t pBuft, const ggml_tensor *pTensor) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);
  size_t vMaxAllocSize = 0;
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    const size_t cAllocSize = ggml_backend_buft_get_alloc_size(fGgmlBackendMetaBuftSimpleBuft(pBuft, vI), pTensor);
    vMaxAllocSize = std::max(vMaxAllocSize, cAllocSize);
  }
  return vMaxAllocSize;
}

static bool fGgmlBackendMetaBufferTypeIsHost(ggml_backend_buffer_type_t pBuft) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    if (!ggml_backend_buft_is_host(fGgmlBackendMetaBuftSimpleBuft(pBuft, vI))) {
      return false;
    }
  }
  return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ fGgmlBackendMetaBufferTypeGetName,
    /* .alloc_buffer     = */ fGgmlBackendMetaBufferTypeAllocBuffer,
    /* .get_alignment    = */ fGgmlBackendMetaBufferTypeGetAlignment,
    /* .get_max_size     = */ fGgmlBackendMetaBufferTypeGetMaxSize,
    /* .get_alloc_size   = */ fGgmlBackendMetaBufferTypeGetAllocSize,
    /* .is_host          = */ fGgmlBackendMetaBufferTypeIsHost,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t pBuft) {
  return pBuft != nullptr && pBuft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t fGgmlBackendMetaDeviceGetBufferType(ggml_backend_dev_t pDev) {
  static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> dMetaBufts;
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  {
    auto vIt = dMetaBufts.find(pDev);
    if (vIt != dMetaBufts.end()) {
      return &vIt->second;
    }
  }

  const size_t cNDevs = fGgmlBackendMetaDevNDevs(pDev);
  std::vector<ggml_backend_buffer_type_t> vSimpleBufts;
  vSimpleBufts.reserve(cNDevs);
  for (size_t vI = 0; vI < cNDevs; vI++) {
    vSimpleBufts.push_back(ggml_backend_dev_buffer_type(fGgmlBackendMetaDevSimpleDev(pDev, vI)));
  }
  ggml_backend_meta_buffer_type_context *vBuftCtx = new ggml_backend_meta_buffer_type_context(vSimpleBufts);

  struct ggml_backend_buffer_type vMetaBuft = {
      /*iface  =*/ggml_backend_meta_buffer_type_iface,
      /*device =*/pDev,
      /*ctx    =*/vBuftCtx,
  };
  auto vResult = dMetaBufts.emplace(pDev, vMetaBuft);
  return &vResult.first->second;
}

static ggml_backend_buffer_type_t fGgmlBackendMetaDeviceGetHostBufferType(ggml_backend_dev_t pDev) {
  GGML_ASSERT(fGgmlBackendDevIsMeta(pDev));
  const ggml_backend_meta_device_context *vMetaDevCtx = (const ggml_backend_meta_device_context *)pDev->context;

  ggml_backend_buffer_type_t vHostBuft = nullptr;
  for (ggml_backend_dev_t vSimpleDev : vMetaDevCtx->vSimpleDevs) {
    ggml_backend_buffer_type_t vSimpleHostBuft = ggml_backend_dev_host_buffer_type(vSimpleDev);
    if (vSimpleHostBuft == nullptr) {
      return nullptr;
    }
    if (vHostBuft == nullptr) {
      vHostBuft = vSimpleHostBuft;
    } else if (vHostBuft != vSimpleHostBuft) {
      // if different simple devices have different host buffer types,
      // we cannot provide a single host buffer type for the meta device
      return nullptr;
    }
  }
  return vHostBuft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
  std::vector<ggml_context_ptr> lCtxs;
  std::map<const ggml_tensor *, std::vector<ggml_tensor *>> dSimpleTensors;

  ggml_backend_meta_simple_tensor_container(const ggml_init_params &pParams, const int pNSimple) {
    lCtxs.reserve(pNSimple);
    for (int vI = 0; vI < pNSimple; vI++) {
      lCtxs.emplace_back(ggml_init(pParams));
    }
  }
  ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
  // FIXME
  // Most tensors can simply be stored statically in their own buffer.
  // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
  // If external views are simply using that buffer they will slowly deplete its memory.
  // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for mgs.
  // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
  //     currently not possible due to graph-external operations in the backend scheduler.
  ggml_backend_meta_simple_tensor_container vStcStatic;
  ggml_backend_meta_simple_tensor_container aStcCompute[2];
  int vStcComputeIndex = 0;
  int vStcComputeIndexNext = 0;
  std::vector<ggml_backend_buffer_ptr> vBufs;

  // FIXME
  // The size of the split state cache is unbounded and can theoretically grow infinitely large.
  // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too
  // expensive.
  static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
  std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> vSplitStateCache;

  int vDebug;

  ggml_backend_meta_buffer_context(ggml_backend_meta_simple_tensor_container &pStcStatic,
                                   ggml_backend_meta_simple_tensor_container &pStcCompute0,
                                   ggml_backend_meta_simple_tensor_container &pStcCompute1,
                                   const std::vector<ggml_backend_buffer_t> &pBufs)
      : vStcStatic(std::move(pStcStatic)), aStcCompute{std::move(pStcCompute0), std::move(pStcCompute1)} {
    this->vBufs.reserve(pBufs.size());
    for (ggml_backend_buffer_t vBuf : pBufs) {
      this->vBufs.emplace_back(vBuf);
    }
    const char *vGgmlMetaDebug = getenv("GGML_META_DEBUG");
    vDebug = vGgmlMetaDebug ? atoi(vGgmlMetaDebug) : 0;
  }

  ggml_backend_meta_simple_tensor_container &mGetSimpleTensorContainer(const ggml_tensor *pTensor) {
    if (vStcStatic.dSimpleTensors.find(pTensor) != vStcStatic.dSimpleTensors.end()) {
      return vStcStatic;
    }
    return aStcCompute[vStcComputeIndex];
  }
};

static void fGgmlBackendMetaBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pBuffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pBuffer->context;
  delete vBufCtx;
}

static size_t fGgmlBackendMetaBufferNBufs(ggml_backend_buffer_t pMetaBuf) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pMetaBuf));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pMetaBuf->context;
  return vBufCtx->vBufs.size();
}

static ggml_backend_buffer_t fGgmlBackendMetaBufferSimpleBuffer(ggml_backend_buffer_t pMetaBuf, size_t pIndex) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pMetaBuf));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pMetaBuf->context;
  GGML_ASSERT(pIndex < vBufCtx->vBufs.size());
  return vBufCtx->vBufs[pIndex].get();
}

static struct ggml_tensor *fGgmlBackendMetaBufferSimpleTensor(const struct ggml_tensor *pTensor, size_t pIndex) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pTensor->buffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pTensor->buffer->context;
  GGML_ASSERT(pIndex < vBufCtx->vBufs.size());

  ggml_backend_meta_simple_tensor_container &vStc = vBufCtx->mGetSimpleTensorContainer(pTensor);
  auto vIt = vStc.dSimpleTensors.find(pTensor);
  if (vIt == vStc.dSimpleTensors.end()) {
    return nullptr;
  }
  return vIt->second[pIndex];
}

static struct ggml_backend_meta_split_state fGgmlBackendMetaGetSplitState(const struct ggml_tensor *pTensor,
                                                                          bool pAssumeSync);

static struct ggml_backend_meta_split_state
fGgmlBackendMetaGetSplitState(ggml_backend_meta_simple_tensor_container &pStc, const struct ggml_tensor *pTensor,
                              bool pAssumeSync) {
  // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
  // Since the operations in question are developed specifically for mgs this currently does not manifest as a bug
  // there. However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
  const size_t cNBufs = fGgmlBackendMetaBufferNBufs(pTensor->buffer);
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pTensor->buffer->context;

  auto vSplitStatesEqual = [&](const ggml_backend_meta_split_state &pA,
                               const ggml_backend_meta_split_state &pB) -> bool {
    if (pA.axis != pB.axis) {
      return false;
    }
    for (size_t vJ = 0; vJ < cNBufs; vJ++) {
      int64_t vSumA = 0;
      for (size_t vS = 0; vS < pA.n_segments; vS++) {
        vSumA += pA.ne[vS * cNBufs + vJ] * pA.nr[vS];
      }
      int64_t vSumB = 0;
      for (size_t vS = 0; vS < pB.n_segments; vS++) {
        vSumB += pB.ne[vS * cNBufs + vJ] * pB.nr[vS];
      }
      if (vSumA != vSumB) {
        return false;
      }
    }
    return true;
  };

  auto fHandleGeneric = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs,
                            bool pScalarOnly) -> ggml_backend_meta_split_state {
    ggml_backend_meta_split_state vRet = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
    for (size_t vI = 0; vI < GGML_MAX_SRC; vI++) {
      if (pTensor->src[vI] == nullptr || pTensor->src[vI] == pTensor) {
        continue;
      }
      if (vRet.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
        vRet = pSrcSs[vI];
      } else if (!vSplitStatesEqual(pSrcSs[vI], vRet)) {
        vRet = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        break;
      }
    }
    if (vRet.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
      vRet = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    if (pScalarOnly && vRet.axis >= 0 && vRet.axis < GGML_MAX_DIMS) {
      vRet = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    GGML_ASSERT(vRet.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    return vRet;
  };

  // Some ops process data on a per-row bases:
  auto fHandlePerRow = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    GGML_ASSERT(pSrcSs[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
    return pSrcSs[0];
  };

  // Some ops broadcast the src1 data across src0:
  auto fHandleBinBcast =
      [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis >= 0 && pSrcSs[0].axis < GGML_MAX_DIMS && pTensor->src[1]->ne[pSrcSs[0].axis] == 1 &&
        pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      return pSrcSs[0];
    }
    if (pSrcSs[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
        (pSrcSs[0].axis == pSrcSs[1].axis ||
         (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
      return pSrcSs[0]; // GGML_OP_ADD_ID
    }
    GGML_ASSERT(pTensor->src[2] == nullptr || pSrcSs[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    return fHandleGeneric(pSrcSs, /*scalar_only =*/false);
  };

  auto fHandleConcat = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    const ggml_backend_meta_split_axis cConcatAxis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(pTensor, 0));
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[1].axis >= 0 && pSrcSs[1].axis < GGML_MAX_DIMS) {
      GGML_ASSERT(cConcatAxis != pSrcSs[1].axis);
      return pSrcSs[1];
    }
    if (pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[0].axis >= 0 && pSrcSs[0].axis < GGML_MAX_DIMS) {
      GGML_ASSERT(cConcatAxis != pSrcSs[0].axis);
      return pSrcSs[0];
    }
    if (pSrcSs[0].axis == pSrcSs[1].axis && pSrcSs[0].axis != cConcatAxis) {
      return pSrcSs[0];
    }
    return fHandleGeneric(pSrcSs, /*scalar_only =*/true);
  };

  auto fHandleMulMat = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      ggml_backend_meta_split_state vRet = pSrcSs[0];
      vRet.axis = GGML_BACKEND_SPLIT_AXIS_0;
      vRet.nr[0] = 1;
      vRet.n_segments = 1;
      return vRet;
    }
    if (pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      return pSrcSs[1];
    }
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
      GGML_ASSERT(vSplitStatesEqual(pSrcSs[0], pSrcSs[1]));
      return {pAssumeSync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
    }
    GGML_ABORT("fatal error");
    // return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
  };

  auto fHandleReshape = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    switch (pSrcSs[0].axis) {
    case GGML_BACKEND_SPLIT_AXIS_0:
    case GGML_BACKEND_SPLIT_AXIS_1:
    case GGML_BACKEND_SPLIT_AXIS_2:
    case GGML_BACKEND_SPLIT_AXIS_3: {
      GGML_ASSERT(pSrcSs[0].n_segments == 1);
      if (pSrcSs[0].axis == ggml_n_dims(pTensor->src[0]) - 1 && pSrcSs[0].nr[0] == 1) {
        return {ggml_backend_meta_split_axis(ggml_n_dims(pTensor) - 1), {0}, {1}, 1};
      }
      int64_t vBaseNeIn = pTensor->src[0]->ne[0];
      for (int vDim = 1; vDim <= pSrcSs[0].axis; vDim++) {
        vBaseNeIn *= pTensor->src[0]->ne[vDim];
      }
      vBaseNeIn /= pSrcSs[0].nr[0];
      int64_t vBaseNeOut = 1;
      for (int vDim = 0; vDim < GGML_MAX_DIMS; vDim++) {
        const int64_t cBaseNeOutNext = vBaseNeOut *= pTensor->ne[vDim];
        if (cBaseNeOutNext % vBaseNeIn == 0) {
          return {ggml_backend_meta_split_axis(vDim), {0}, {uint32_t(cBaseNeOutNext / vBaseNeIn)}, 1};
        }
        if (cBaseNeOutNext > vBaseNeIn) {
          GGML_ASSERT(pSrcSs[0].n_segments == 1);
          GGML_ASSERT(pSrcSs[0].nr[0] == 1);
          return {ggml_backend_meta_split_axis(vDim), {0}, {1}, 1};
        }
        vBaseNeOut = cBaseNeOutNext;
      }
      GGML_ABORT("shape mismatch for %s", ggml_op_name(pTensor->op));
    }
    case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
    case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
      return pSrcSs[0];
    }
    default: {
      GGML_ABORT("fatal error");
      // return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    }
  };

  auto fHandleCpy = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis >= 0 && pSrcSs[0].axis < GGML_MAX_DIMS) {
      return fHandleReshape(pSrcSs);
    }
    return fHandleGeneric(pSrcSs, /*scalar_only =*/false);
  };

  auto fHandleView = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (ggml_is_contiguous(pTensor) && ggml_is_contiguous(pTensor->src[0])) {
      return fHandleReshape(pSrcSs);
    }
    const int cAxis = pSrcSs[0].axis;
    {
      bool vAllStridesTheSame = true;
      for (int vDim = 0; vDim < GGML_MAX_DIMS; vDim++) {
        if (pTensor->ne[vDim] == 1 && pTensor->src[0]->ne[vDim] == 1) {
          continue;
        }
        if (pTensor->nb[vDim] != pTensor->src[0]->nb[vDim]) {
          vAllStridesTheSame = false;
          break;
        }
      }
      if (vAllStridesTheSame) {
        return pSrcSs[0];
      }
    }
    if (!ggml_is_permuted(pTensor) && !ggml_is_permuted(pTensor->src[0]) && cAxis >= 0 && cAxis < GGML_MAX_DIMS - 1) {
      for (int vDim = 0; vDim < GGML_MAX_DIMS - 1; vDim++) {
        if (pTensor->nb[vDim + 1] == pTensor->src[0]->nb[cAxis + 1]) {
          return {ggml_backend_meta_split_axis(vDim), {0}, {1}, 1};
        }
      }
      GGML_ABORT("fatal error");
    }
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
      return pSrcSs[0];
    }
    GGML_ABORT("view of permuted tensor not implemented");
    // return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
  };

  auto fHandlePermute = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    switch (pSrcSs[0].axis) {
    case GGML_BACKEND_SPLIT_AXIS_0:
    case GGML_BACKEND_SPLIT_AXIS_1:
    case GGML_BACKEND_SPLIT_AXIS_2:
    case GGML_BACKEND_SPLIT_AXIS_3: {
      GGML_ASSERT(pSrcSs[0].n_segments == 1 || pSrcSs[0].nr[0] == 1);
      return {ggml_backend_meta_split_axis(pTensor->op_params[pSrcSs[0].axis]), {0}, {pSrcSs[0].nr[0]}, 1};
    }
    case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
    case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
      return pSrcSs[0];
    }
    default: {
      GGML_ABORT("fatal error");
      // return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    }
  };

  auto fHandleTranspose =
      [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    switch (pSrcSs[0].axis) {
    case GGML_BACKEND_SPLIT_AXIS_0:
    case GGML_BACKEND_SPLIT_AXIS_1: {
      GGML_ASSERT(pSrcSs[0].n_segments == 1 || pSrcSs[0].nr[0] == 1);
      return {ggml_backend_meta_split_axis(int(pSrcSs[0].axis) ^ 1), {0}, {pSrcSs[0].nr[0]}, 1};
    }
    case GGML_BACKEND_SPLIT_AXIS_2:
    case GGML_BACKEND_SPLIT_AXIS_3:
    case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
    case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
      return pSrcSs[0];
    }
    default: {
      GGML_ABORT("fatal error");
      // return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    }
  };

  auto fHandleGetRows = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      return pSrcSs[0];
    }
    return fHandleGeneric(pSrcSs, /*scalar_only =*/true);
  };

  auto fHandleSetRows = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    GGML_ASSERT(pSrcSs[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
    GGML_ASSERT(pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    GGML_ASSERT(vSplitStatesEqual(pSrcSs[0], pSrcSs[2]));
    return pSrcSs[0];
  };

  auto fHandleRope = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    GGML_ASSERT(pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    return pSrcSs[0];
  };

  auto fHandlePad = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis >= 0 && pSrcSs[0].axis < GGML_MAX_DIMS) {
      GGML_ASSERT(pTensor->op_params[2 * pSrcSs[0].axis + 0] == 0);
      GGML_ASSERT(pTensor->op_params[2 * pSrcSs[0].axis + 1] == 0);
    }
    return pSrcSs[0];
  };

  auto fHandleFlashAttnExt =
      [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    GGML_ASSERT(pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
    GGML_ASSERT(pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
    GGML_ASSERT(pSrcSs[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
    GGML_ASSERT(pTensor->src[4] == nullptr || pSrcSs[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    GGML_ASSERT(pTensor->src[4] == nullptr || pSrcSs[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
    return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
  };

  auto fHandleSsmConv = [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis == pSrcSs[1].axis) {
      if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
      }
      if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
      }
    }
    return fHandleGeneric(pSrcSs, /*scalar_only =*/false);
  };

  auto fHandleGatedDeltaNet =
      [&](const std::vector<ggml_backend_meta_split_state> &pSrcSs) -> ggml_backend_meta_split_state {
    if (pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
        pSrcSs[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
        pSrcSs[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && pSrcSs[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
      return pSrcSs[0];
    }
    GGML_ASSERT(pSrcSs[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
    GGML_ASSERT(pSrcSs[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
    GGML_ASSERT(pSrcSs[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
    GGML_ASSERT(pSrcSs[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
    GGML_ASSERT(pSrcSs[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
    // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
    // so a head-aligned split on the input cache lands on axis 2 here.
    GGML_ASSERT(pSrcSs[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || pSrcSs[5].axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                pSrcSs[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
    return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
  };

  auto fCalculateSplitState = [&]() -> ggml_backend_meta_split_state {
    if (ggml_nelements(pTensor) == 0) {
      return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    }
    if (ggml_backend_buffer_get_usage(pTensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        pTensor->view_src == nullptr) {
      ggml_backend_dev_t vDev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pTensor->buffer));
      const ggml_backend_meta_device_context *vDevCtx = (const ggml_backend_meta_device_context *)vDev->context;
      ggml_backend_meta_split_state vRet = vDevCtx->vGetSplitState(pTensor, vDevCtx->vGetSplitStateUd);
      if (vRet.axis >= 0 && vRet.axis <= GGML_MAX_DIMS) {
        const int64_t cGranularity = vRet.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(pTensor->type) : 1;
        int64_t vNeSum = 0;
        for (size_t vS = 0; vS < vRet.n_segments; vS++) {
          for (size_t vJ = 0; vJ < cNBufs; vJ++) {
            GGML_ASSERT(vRet.ne[vS * cNBufs + vJ] % cGranularity == 0);
            vNeSum += vRet.ne[vS * cNBufs + vJ] * vRet.nr[vS];
          }
        }
        GGML_ASSERT(vNeSum == pTensor->ne[vRet.axis]);
      }
      return vRet;
    }

    std::vector<ggml_backend_meta_split_state> vSrcSs(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
    for (size_t vI = 0; vI < GGML_MAX_SRC; vI++) {
      if (pTensor->src[vI] == nullptr || pTensor->src[vI] == pTensor) {
        vSrcSs[vI] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        continue;
      }
      vSrcSs[vI] = fGgmlBackendMetaGetSplitState(pStc, pTensor->src[vI], /*assume_sync =*/true);
      GGML_ASSERT(vSrcSs[vI].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    }

    ggml_backend_meta_split_state vSplitState;
    switch (pTensor->op) {
    case GGML_OP_NONE: {
      vSplitState = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    } break;
    case GGML_OP_DUP: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_ADD:
    case GGML_OP_ADD_ID: {
      vSplitState = fHandleBinBcast(vSrcSs);
    } break;
    case GGML_OP_ADD1:
    case GGML_OP_ACC: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV: {
      vSplitState = fHandleBinBcast(vSrcSs);
    } break;
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_LOG:
    case GGML_OP_SIN:
    case GGML_OP_COS: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_SUM: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SUM_ROWS:
    case GGML_OP_CUMSUM:
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX:
    case GGML_OP_AMAX_POW2:
    case GGML_OP_COUNT_EQUAL: {
      vSplitState = fHandlePerRow(vSrcSs);
    } break;
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_CONCAT: {
      vSplitState = fHandleConcat(vSrcSs);
    } break;
    case GGML_OP_SILU_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_L2_NORM: {
      vSplitState = fHandlePerRow(vSrcSs);
    } break;
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID: {
      vSplitState = fHandleMulMat(vSrcSs);
    } break;
    case GGML_OP_OUT_PROD: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SCALE: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_SET: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_CPY: {
      vSplitState = fHandleCpy(vSrcSs);
    } break;
    case GGML_OP_CONT:
    case GGML_OP_RESHAPE: {
      vSplitState = fHandleReshape(vSrcSs);
    } break;
    case GGML_OP_VIEW: {
      vSplitState = fHandleView(vSrcSs);
    } break;
    case GGML_OP_PERMUTE: {
      vSplitState = fHandlePermute(vSrcSs);
    } break;
    case GGML_OP_TRANSPOSE: {
      vSplitState = fHandleTranspose(vSrcSs);
    } break;
    case GGML_OP_GET_ROWS: {
      vSplitState = fHandleGetRows(vSrcSs);
    } break;
    case GGML_OP_GET_ROWS_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SET_ROWS: {
      vSplitState = fHandleSetRows(vSrcSs);
    } break;
    case GGML_OP_DIAG:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_DIAG_MASK_ZERO: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_ROPE: {
      vSplitState = fHandleRope(vSrcSs);
    } break;
    case GGML_OP_ROPE_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_CLAMP: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_CONV_TRANSPOSE_1D:
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_BACK:
    case GGML_OP_IM2COL_3D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_3D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_CONV_TRANSPOSE_2D:
    case GGML_OP_POOL_1D:
    case GGML_OP_POOL_2D:
    case GGML_OP_POOL_2D_BACK:
    case GGML_OP_UPSCALE: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_PAD: {
      vSplitState = fHandlePad(vSrcSs);
    } break;
    case GGML_OP_PAD_REFLECT_1D:
    case GGML_OP_ROLL:
    case GGML_OP_ARANGE:
    case GGML_OP_TIMESTEP_EMBEDDING: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_ARGSORT:
    case GGML_OP_TOP_K: {
      vSplitState = fHandlePerRow(vSrcSs);
    } break;
    case GGML_OP_LEAKY_RELU: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_TRI: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_FILL: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_FLASH_ATTN_EXT: {
      vSplitState = fHandleFlashAttnExt(vSrcSs);
    } break;
    case GGML_OP_FLASH_ATTN_BACK: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_SSM_CONV: {
      vSplitState = fHandleSsmConv(vSrcSs);
    } break;
    case GGML_OP_SSM_SCAN:
    case GGML_OP_WIN_PART:
    case GGML_OP_WIN_UNPART:
    case GGML_OP_GET_REL_POS:
    case GGML_OP_ADD_REL_POS:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_GATED_LINEAR_ATTN:
    case GGML_OP_RWKV_WKV7:
    case GGML_OP_SOLVE_TRI: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_GATED_DELTA_NET: {
      vSplitState = fHandleGatedDeltaNet(vSrcSs);
    } break;
    case GGML_OP_DSV4_HC_COMB:
    case GGML_OP_DSV4_HC_PRE:
    case GGML_OP_DSV4_HC_POST: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_UNARY: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    case GGML_OP_MAP_CUSTOM1:
    case GGML_OP_MAP_CUSTOM2:
    case GGML_OP_MAP_CUSTOM3:
    case GGML_OP_CUSTOM: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/true);
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS:
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
      vSplitState = fHandlePerRow(vSrcSs);
    } break;
    case GGML_OP_OPT_STEP_ADAMW:
    case GGML_OP_OPT_STEP_SGD:
    case GGML_OP_GLU: {
      vSplitState = fHandleGeneric(vSrcSs, /*scalar_only =*/false);
    } break;
    default: {
      GGML_ABORT("ggml op not implemented: %s", ggml_op_name(pTensor->op));
      vSplitState = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    } break;
    }
    if (vSplitState.axis >= 0 && vSplitState.axis < GGML_MAX_DIMS) {
      bool vFirstSrcSplitByAxis = true;
      const size_t cNBufs = fGgmlBackendMetaBufferNBufs(pTensor->buffer);

      for (size_t vI = 0; vI < GGML_MAX_SRC; vI++) {
        if (pTensor->src[vI] == nullptr || vSrcSs[vI].axis < 0 || vSrcSs[vI].axis >= GGML_MAX_DIMS) {
          continue;
        }
        if (vFirstSrcSplitByAxis) {
          for (size_t vJ = 0; vJ < cNBufs; vJ++) {
            // Take over ratio from src:
            for (size_t vS = 0; vS < vSrcSs[vI].n_segments; vS++) {
              vSplitState.ne[vS * cNBufs + vJ] = 0;
            }
            for (size_t vS = 0; vS < vSrcSs[vI].n_segments; vS++) {
              vSplitState.ne[vJ] += vSrcSs[vI].ne[vS * cNBufs + vJ] * vSrcSs[vI].nr[vS];
            }
            vSplitState.ne[vJ] *= pTensor->ne[vSplitState.axis];
            if (vSplitState.ne[vJ] != 0 || pTensor->src[vI]->ne[vSrcSs[vI].axis] != 0) {
              const int64_t cDiv = pTensor->src[vI]->ne[vSrcSs[vI].axis] * vSplitState.nr[0];
              GGML_ASSERT(vSplitState.ne[vJ] % cDiv == 0);
              vSplitState.ne[vJ] /= cDiv;
            }
          }
        } else {
          GGML_ASSERT(vSplitState.n_segments == 1);
          for (size_t vJ = 0; vJ < cNBufs; vJ++) {
            // Assert that ratio is consistent:
            int64_t vSum = 0;
            for (size_t vS = 0; vS < vSrcSs[vI].n_segments; vS++) {
              vSum += vSrcSs[vI].ne[vS * cNBufs + vJ] * vSrcSs[vI].nr[vS];
            }
            GGML_ASSERT(vSplitState.ne[vJ] * vSplitState.nr[0] * pTensor->src[vI]->ne[vSrcSs[vI].axis] ==
                        vSum * pTensor->ne[vSplitState.axis]);
          }
        }
        vFirstSrcSplitByAxis = false;
      }
      GGML_ASSERT(!vFirstSrcSplitByAxis);
    }
    return vSplitState;
  };

  const std::pair cKey = std::make_pair(pTensor, pAssumeSync);
  auto vIt = vBufCtx->vSplitStateCache.find(cKey);
  if (vIt != vBufCtx->vSplitStateCache.end() &&
      memcmp(vIt->second.second, (const char *)pTensor, sizeof(vIt->second.second)) != 0) {
    vBufCtx->vSplitStateCache.clear();
    vIt = vBufCtx->vSplitStateCache.end();
  }

  if (vIt == vBufCtx->vSplitStateCache.end()) {
    vBufCtx->vSplitStateCache[cKey].first = fCalculateSplitState();
    memcpy(vBufCtx->vSplitStateCache[cKey].second, pTensor, sizeof(vBufCtx->vSplitStateCache[cKey].second));
    if (vBufCtx->vDebug > 0) {
      std::string vSrcsInfo;
      for (size_t vI = 0; vI < GGML_MAX_SRC; vI++) {
        if (pTensor->src[vI] == nullptr) {
          continue;
        }
        if (!vSrcsInfo.empty()) {
          vSrcsInfo += ", ";
        }
        const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pTensor->src[0], true);
        GGML_ASSERT(cSplitState.n_segments == 1);
        const char *vAxisName = ggml_backend_meta_split_axis_name(cSplitState.axis);
        std::string vNeInfo;
        for (size_t vJ = 0; vJ < cNBufs; vJ++) {
          if (!vNeInfo.empty()) {
            vNeInfo += ", ";
          }
          vNeInfo += std::to_string(cSplitState.ne[vJ]) + "x" + std::to_string(cSplitState.nr[0]);
        }
        vSrcsInfo += std::string(pTensor->src[vI]->name) + "[" + ggml_op_name(pTensor->src[vI]->op) + ", " + vAxisName +
                     ", {" + vNeInfo + "}]";
      }
      std::string vNeInfo;
      for (size_t vJ = 0; vJ < cNBufs; vJ++) {
        if (!vNeInfo.empty()) {
          vNeInfo += ", ";
        }
        const ggml_backend_meta_split_state &vSs = vBufCtx->vSplitStateCache[cKey].first;
        vNeInfo += std::to_string(vSs.ne[vJ]) + "x" + std::to_string(vSs.nr[0]);
      }
      GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", vSrcsInfo.c_str(), pTensor->name,
                     ggml_op_name(pTensor->op),
                     ggml_backend_meta_split_axis_name(vBufCtx->vSplitStateCache[cKey].first.axis), vNeInfo.c_str());
    }
  }

  ggml_backend_meta_split_state vRet = vBufCtx->vSplitStateCache[cKey].first;
  GGML_ASSERT(vRet.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
  if (vRet.axis >= 0 && vRet.axis < GGML_MAX_DIMS) {
    int64_t vNeRet = 0;
    for (size_t vS = 0; vS < vRet.n_segments; vS++) {
      for (size_t vJ = 0; vJ < cNBufs; vJ++) {
        vNeRet += vRet.ne[vS * cNBufs + vJ] * vRet.nr[vS];
      }
    }
    assert(vNeRet == pTensor->ne[int(vRet.axis)]);
  }
#endif // NDEBUG
  return vRet;
}

static struct ggml_backend_meta_split_state fGgmlBackendMetaGetSplitState(const struct ggml_tensor *pTensor,
                                                                          bool pAssumeSync) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pTensor->buffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pTensor->buffer->context;
  return fGgmlBackendMetaGetSplitState(vBufCtx->mGetSimpleTensorContainer(pTensor), pTensor, pAssumeSync);
}

static void *fGgmlBackendMetaBufferGetBase(ggml_backend_buffer_t pBuffer) {
  GGML_UNUSED(pBuffer);
  return (void *)0x1000000000000000; // FIXME
}

static enum ggml_status fGgmlBackendMetaBufferInitTensorImpl(ggml_backend_meta_simple_tensor_container &pStc,
                                                             ggml_tensor *pTensor) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pTensor->buffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pTensor->buffer->context;
  const size_t cNSimpleBufs = fGgmlBackendMetaBufferNBufs(pTensor->buffer);

  const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pStc, pTensor, /*assume_sync =*/true);
  GGML_ASSERT(ggml_nelements(pTensor) == 0 || cSplitState.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
  GGML_ASSERT(cSplitState.n_segments <= 16);

  int vSplitDim = cSplitState.axis;
  int64_t aNe[GGML_MAX_DIMS];
  size_t aNb[GGML_MAX_DIMS];
  for (size_t vK = 0; vK < GGML_MAX_DIMS; vK++) {
    aNe[vK] = pTensor->ne[vK];
    aNb[vK] = pTensor->nb[vK];
  }

  std::vector<ggml_tensor *> lSimpleTensors;
  lSimpleTensors.reserve(cNSimpleBufs);
  for (size_t vJ = 0; vJ < cNSimpleBufs; vJ++) {
    ggml_context *vSimpleCtx = pStc.lCtxs[vJ].get();
    ggml_backend_buffer_t vSimpleBuf = vBufCtx->vBufs[vJ].get();

    if ((vSimpleBuf != nullptr) && ggml_backend_buffer_is_multi_buffer(vSimpleBuf)) {
      GGML_ABORT("multi buffers are not supported by the meta backend");
    }

    if (vSplitDim >= 0 && vSplitDim < GGML_MAX_DIMS) {
      // TODO: the following assert fails for parallel decoding even though the results are correct:
      // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
      aNe[vSplitDim] = 0;
      for (size_t vS = 0; vS < cSplitState.n_segments; vS++) {
        aNe[vSplitDim] += cSplitState.ne[vS * cNSimpleBufs + vJ] * cSplitState.nr[vS];
      }
      for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
        if (pTensor->nb[vI] > pTensor->nb[vSplitDim]) {
          aNb[vI] = pTensor->nb[vI] * aNe[vSplitDim] / pTensor->ne[vSplitDim];
        }
      }
    }

    ggml_tensor *vTIj = ggml_new_tensor(vSimpleCtx, pTensor->type, GGML_MAX_DIMS, aNe);
    vTIj->op = pTensor->op;
    for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
      vTIj->nb[vI] = aNb[vI];
    }
    vTIj->flags = pTensor->flags;
    memcpy(vTIj->op_params, pTensor->op_params, sizeof(pTensor->op_params));
    ggml_set_name(vTIj, pTensor->name);
    vTIj->buffer = vSimpleBuf;
    vTIj->view_src = pTensor->view_src;
    vTIj->view_offs = pTensor->view_offs;
    if (vTIj->view_src != nullptr && ggml_backend_buffer_is_meta(vTIj->view_src->buffer)) {
      vTIj->view_src = fGgmlBackendMetaBufferSimpleTensor(pTensor->view_src, vJ);
      if (vTIj->view_offs > 0 && vSplitDim >= 0 && vSplitDim < GGML_MAX_DIMS) {
        GGML_ASSERT(pTensor->ne[vSplitDim] != 0);
        const int cSplitDimViewSrc = fGgmlBackendMetaGetSplitState(pTensor->view_src, /*assume_sync =*/true).axis;
        GGML_ASSERT(cSplitDimViewSrc >= 0 && cSplitDimViewSrc < GGML_MAX_DIMS);

        // The offset can be internal to the data split, in those cases the view offset should not be scaled.
        // If however, the offset is larger than the data split then it needs to be scaled proportionally.
        bool vSplitInternalOffset = vTIj->view_offs <= pTensor->view_src->nb[cSplitDimViewSrc];
        for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
          const size_t cDimSize = pTensor->ne[vI] * pTensor->nb[vI];
          if (pTensor->view_offs <= cDimSize && cDimSize < pTensor->nb[vSplitDim]) {
            vSplitInternalOffset = true;
            break;
          }
        }
        if (!vSplitInternalOffset) {
          vTIj->view_offs = vTIj->view_offs * aNe[vSplitDim] / pTensor->ne[vSplitDim];
        }
      }
    }
    if (vTIj->view_src != nullptr) {
      vTIj->data = (char *)vTIj->view_src->data + vTIj->view_offs;
    } else if (vSimpleBuf != nullptr) {
      vTIj->data = (char *)ggml_backend_buffer_get_base(vSimpleBuf) + size_t(pTensor->data) -
                   size_t(ggml_backend_buffer_get_base(pTensor->buffer));
    }
    vTIj->extra = pTensor->extra;
    for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
      vTIj->src[vI] = pTensor->src[vI];
      if (pTensor->src[vI] == pTensor) {
        vTIj->src[vI] = vTIj;
      } else if (vTIj->src[vI] != nullptr && ggml_backend_buffer_is_meta(vTIj->src[vI]->buffer)) {
        vTIj->src[vI] = fGgmlBackendMetaBufferSimpleTensor(pTensor->src[vI], vJ);
      }
    }

    lSimpleTensors.push_back(vTIj);
  }

  // If one of the sources has a zero-sized slice, disable the computation:
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    if (pTensor->src[vI] == nullptr || !ggml_backend_buffer_is_meta(pTensor->src[vI]->buffer)) {
      continue;
    }

    const ggml_backend_meta_split_state cSplitStateSrc =
        fGgmlBackendMetaGetSplitState(pTensor->src[vI], /*assume_sync =*/true);
    if (cSplitStateSrc.axis < 0 || cSplitStateSrc.axis >= GGML_MAX_DIMS) {
      continue;
    }
    for (size_t vJ = 0; vJ < cNSimpleBufs; vJ++) {
      int64_t vNeSum = 0;
      for (size_t vS = 0; vS < cSplitStateSrc.n_segments; vS++) {
        vNeSum += cSplitStateSrc.ne[vS * cNSimpleBufs + vJ] * cSplitStateSrc.nr[vS];
      }
      if (vNeSum == 0) {
        lSimpleTensors[vJ]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
      }
    }
  }

  pStc.dSimpleTensors[pTensor] = lSimpleTensors;

  return GGML_STATUS_SUCCESS;
}

static enum ggml_status fGgmlBackendMetaBufferInitTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pBuffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pBuffer->context;
  vBufCtx->vStcComputeIndex = vBufCtx->vStcComputeIndexNext;
  return fGgmlBackendMetaBufferInitTensorImpl(vBufCtx->mGetSimpleTensorContainer(pTensor), pTensor);
}

static void fGgmlBackendMetaBufferSetTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor, const void *pData,
                                            size_t pOffset, size_t pSize) {
  const size_t cNBufs = fGgmlBackendMetaBufferNBufs(pBuffer);
  const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pTensor, /*assume_sync =*/false);
  GGML_ASSERT(ggml_is_contiguous(pTensor) || cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

  if (cSplitState.n_segments != 1 || cSplitState.nr[0] != 1) {
    GGML_ASSERT(cSplitState.axis >= 0 && cSplitState.axis < GGML_MAX_DIMS);
    GGML_ASSERT(cSplitState.nr[0] != 0);
    GGML_ASSERT(pTensor->ne[3] == 1);

    size_t vOffsetData = 0;
    std::vector<size_t> lSimpleOffsets(cNBufs, 0);
    if (cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_0) {
      GGML_ASSERT(pTensor->ne[2] == 1);

      const size_t cRowStride = pTensor->nb[1];
      GGML_ASSERT(pOffset % cRowStride == 0);
      GGML_ASSERT(pSize % cRowStride == 0);
      const int64_t cRowStart = pOffset / cRowStride;
      const int64_t cRowCount = pSize / cRowStride;
      GGML_ASSERT(cRowStart + cRowCount <= pTensor->ne[1]);

      const int64_t cBlckSize = ggml_blck_size(pTensor->type);
      for (size_t vS = 0; vS < cSplitState.n_segments; vS++) {
        for (size_t vR = 0; vR < cSplitState.nr[vS]; vR++) {
          for (size_t vJ = 0; vJ < cNBufs; vJ++) {
            ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
            GGML_ASSERT(cSplitState.ne[vS * cNBufs + vJ] % cBlckSize == 0);
            const size_t cNbytes = cSplitState.ne[vS * cNBufs + vJ] / cBlckSize * pTensor->nb[0];
            ggml_backend_tensor_set_2d(vSimpleTensor, (const char *)pData + vOffsetData,
                                       lSimpleOffsets[vJ] + cRowStart * vSimpleTensor->nb[1], cNbytes, cRowCount,
                                       vSimpleTensor->nb[1], pTensor->nb[1]);
            vOffsetData += cNbytes;
            lSimpleOffsets[vJ] += cNbytes;
          }
        }
      }
      GGML_ASSERT(vOffsetData * cRowCount == pSize);
      return;
    }
    GGML_ASSERT(cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_1);

    const size_t cRowStride = pTensor->nb[2];
    GGML_ASSERT(pOffset % cRowStride == 0);
    GGML_ASSERT(pSize % cRowStride == 0);
    const int64_t cRowStart = pOffset / cRowStride;
    const int64_t cRowCount = pSize / cRowStride;
    GGML_ASSERT(cRowStart + cRowCount <= pTensor->ne[2]);

    for (size_t vS = 0; vS < cSplitState.n_segments; vS++) {
      for (size_t vR = 0; vR < cSplitState.nr[vS]; vR++) {
        for (size_t vJ = 0; vJ < cNBufs; vJ++) {
          ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
          const size_t cNbytes = cSplitState.ne[vS * cNBufs + vJ] * pTensor->nb[1];
          ggml_backend_tensor_set_2d(vSimpleTensor, (const char *)pData + vOffsetData,
                                     lSimpleOffsets[vJ] + cRowStart * vSimpleTensor->nb[2], cNbytes, cRowCount,
                                     vSimpleTensor->nb[2], pTensor->nb[2]);
          vOffsetData += cNbytes;
          lSimpleOffsets[vJ] += cNbytes;
        }
      }
    }
    GGML_ASSERT(vOffsetData * cRowCount == pSize);
    return;
  }

  switch (cSplitState.axis) {
  case GGML_BACKEND_SPLIT_AXIS_0:
  case GGML_BACKEND_SPLIT_AXIS_1:
  case GGML_BACKEND_SPLIT_AXIS_2: {
    // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
    const size_t cChunkSizeFull = pTensor->nb[cSplitState.axis + 1];
    GGML_ASSERT(pOffset % cChunkSizeFull == 0);
    GGML_ASSERT(pSize % cChunkSizeFull == 0);
    const int64_t cIStart = pOffset / cChunkSizeFull;
    const int64_t cIStop = (pOffset + pSize) / cChunkSizeFull;
    size_t vOffsetJ = 0;
    for (size_t vJ = 0; vJ < cNBufs; vJ++) {
      ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      const size_t cChunkSizeJ = vSimpleTensor->nb[cSplitState.axis + 1];
      if (cChunkSizeJ == 0) {
        continue;
      }
      const size_t cSimpleOffset = cIStart * cChunkSizeJ;
      ggml_backend_tensor_set_2d(vSimpleTensor, (const char *)pData + vOffsetJ, cSimpleOffset, cChunkSizeJ,
                                 cIStop - cIStart, cChunkSizeJ, cChunkSizeFull);
      vOffsetJ += cChunkSizeJ;
    }
    GGML_ASSERT(vOffsetJ == cChunkSizeFull);
  } break;
  case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    for (size_t vJ = 0; vJ < cNBufs; vJ++) {
      ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      ggml_backend_tensor_set(vSimpleTensor, pData, pOffset, pSize);
    }
  } break;
  case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
    GGML_ASSERT(pTensor->type == GGML_TYPE_F32);
    const int64_t cNe = ggml_nelements(pTensor);
    std::vector<float> lTmp;
    lTmp.reserve(cNe);
    for (int64_t vI = 0; vI < cNe; vI++) {
      lTmp.push_back(((const float *)pData)[vI] / cNBufs);
    }
    for (size_t vJ = 0; vJ < cNBufs; vJ++) {
      ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      ggml_backend_tensor_set(vSimpleTensor, lTmp.data(), pOffset, pSize);
    }
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlBackendMetaBufferGetTensor(ggml_backend_buffer_t pBuffer, const ggml_tensor *pTensor, void *pData,
                                            size_t pOffset, size_t pSize) {
  const size_t cNBufs = fGgmlBackendMetaBufferNBufs(pBuffer);
  const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pTensor, /*assume_sync =*/false);
  GGML_ASSERT(ggml_is_contiguous(pTensor) || cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

  if (cSplitState.n_segments != 1 || cSplitState.nr[0] != 1) {
    GGML_ASSERT(cSplitState.axis >= 0 && cSplitState.axis < GGML_MAX_DIMS);
    GGML_ASSERT(cSplitState.nr[0] != 0);
    GGML_ASSERT(pTensor->ne[3] == 1);

    size_t vOffsetData = 0;
    std::vector<size_t> lSimpleOffsets(cNBufs, 0);
    if (cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_0) {
      GGML_ASSERT(pTensor->ne[2] == 1);

      const size_t cRowStride = pTensor->nb[1];
      GGML_ASSERT(pOffset % cRowStride == 0);
      GGML_ASSERT(pSize % cRowStride == 0);
      const int64_t cRowStart = pOffset / cRowStride;
      const int64_t cRowCount = pSize / cRowStride;
      GGML_ASSERT(cRowStart + cRowCount <= pTensor->ne[1]);

      const int64_t cBlckSize = ggml_blck_size(pTensor->type);
      for (size_t vS = 0; vS < cSplitState.n_segments; vS++) {
        for (size_t vR = 0; vR < cSplitState.nr[vS]; vR++) {
          for (size_t vJ = 0; vJ < cNBufs; vJ++) {
            const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
            GGML_ASSERT(cSplitState.ne[vS * cNBufs + vJ] % cBlckSize == 0);
            const size_t cNbytes = cSplitState.ne[vS * cNBufs + vJ] / cBlckSize * pTensor->nb[0];
            ggml_backend_tensor_get_2d(vSimpleTensor, (char *)pData + vOffsetData,
                                       lSimpleOffsets[vJ] + cRowStart * vSimpleTensor->nb[1], cNbytes, cRowCount,
                                       vSimpleTensor->nb[1], pTensor->nb[1]);
            vOffsetData += cNbytes;
            lSimpleOffsets[vJ] += cNbytes;
          }
        }
      }
      GGML_ASSERT(vOffsetData * cRowCount == pSize);
      return;
    }
    GGML_ASSERT(cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_1);

    const size_t cRowStride = pTensor->nb[2];
    GGML_ASSERT(pOffset % cRowStride == 0);
    GGML_ASSERT(pSize % cRowStride == 0);
    const int64_t cRowStart = pOffset / cRowStride;
    const int64_t cRowCount = pSize / cRowStride;
    GGML_ASSERT(cRowStart + cRowCount <= pTensor->ne[2]);

    for (size_t vS = 0; vS < cSplitState.n_segments; vS++) {
      for (size_t vR = 0; vR < cSplitState.nr[vS]; vR++) {
        for (size_t vJ = 0; vJ < cNBufs; vJ++) {
          const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
          const size_t cNbytes = cSplitState.ne[vS * cNBufs + vJ] * pTensor->nb[1];
          ggml_backend_tensor_get_2d(vSimpleTensor, (char *)pData + vOffsetData,
                                     lSimpleOffsets[vJ] + cRowStart * vSimpleTensor->nb[2], cNbytes, cRowCount,
                                     vSimpleTensor->nb[2], pTensor->nb[2]);
          vOffsetData += cNbytes;
          lSimpleOffsets[vJ] += cNbytes;
        }
      }
    }
    GGML_ASSERT(vOffsetData * cRowCount == pSize);
    return;
  }

  switch (cSplitState.axis) {
  case GGML_BACKEND_SPLIT_AXIS_0:
  case GGML_BACKEND_SPLIT_AXIS_1:
  case GGML_BACKEND_SPLIT_AXIS_2: {
    // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
    const size_t cChunkSizeFull = pTensor->nb[cSplitState.axis + 1];
    GGML_ASSERT(pOffset % cChunkSizeFull == 0);
    GGML_ASSERT(pSize % cChunkSizeFull == 0);
    const int64_t cIStart = pOffset / cChunkSizeFull;
    const int64_t cIStop = (pOffset + pSize) / cChunkSizeFull;
    size_t vOffsetJ = 0;
    for (size_t vJ = 0; vJ < cNBufs; vJ++) {
      const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      const size_t cChunkSizeJ = vSimpleTensor->nb[cSplitState.axis + 1];
      if (cChunkSizeJ == 0) {
        continue;
      }
      const size_t cSimpleOffset = cIStart * cChunkSizeJ;
      ggml_backend_tensor_get_2d(vSimpleTensor, (char *)pData + vOffsetJ, cSimpleOffset, cChunkSizeJ, cIStop - cIStart,
                                 cChunkSizeJ, cChunkSizeFull);
      vOffsetJ += cChunkSizeJ;
    }
    GGML_ASSERT(vOffsetJ == cChunkSizeFull);
  } break;
  case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    // TODO other simple backend may be better
    const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, 0);
    ggml_backend_tensor_get(vSimpleTensor, pData, pOffset, pSize);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlBackendMetaBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  const size_t cNBuffers = fGgmlBackendMetaBufferNBufs(pBuffer);
  for (size_t vI = 0; vI < cNBuffers; vI++) {
    ggml_backend_buffer_clear(fGgmlBackendMetaBufferSimpleBuffer(pBuffer, vI), pValue);
  }
}

static void fGgmlBackendMetaBufferReset(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(ggml_backend_buffer_is_meta(pBuffer));
  ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)pBuffer->context;
  for (size_t vI = 0; vI < vBufCtx->vBufs.size(); vI++) {
    ggml_backend_buffer_reset(fGgmlBackendMetaBufferSimpleBuffer(pBuffer, vI));
  }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ fGgmlBackendMetaBufferFreeBuffer,
    /* .get_base        = */ fGgmlBackendMetaBufferGetBase,
    /* .init_tensor     = */ fGgmlBackendMetaBufferInitTensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ fGgmlBackendMetaBufferSetTensor,
    /* .get_tensor      = */ fGgmlBackendMetaBufferGetTensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ fGgmlBackendMetaBufferClear,
    /* .reset           = */ fGgmlBackendMetaBufferReset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t pBuf) {
  return pBuf != nullptr && pBuf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t fGgmlBackendMetaBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);

  const ggml_init_params cParams = {
      /*.mem_size   =*/1024 * 1024 * ggml_tensor_overhead(), // FIXME
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  ggml_backend_meta_simple_tensor_container vStcStatic;
  ggml_backend_meta_simple_tensor_container vStcCompute0(cParams, cNSimpleBufts);
  ggml_backend_meta_simple_tensor_container vStcCompute1(cParams, cNSimpleBufts);

  size_t vMaxSize = 0;
  std::vector<ggml_backend_buffer_t> vBufs;
  vBufs.reserve(cNSimpleBufts);
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    vBufs.push_back(ggml_backend_buft_alloc_buffer(fGgmlBackendMetaBuftSimpleBuft(pBuft, vI), pSize));
    GGML_ASSERT(vBufs.back() != nullptr);
    vMaxSize = std::max(vMaxSize, ggml_backend_buffer_get_size(vBufs.back()));
  }
  ggml_backend_meta_buffer_context *vBufCtx =
      new ggml_backend_meta_buffer_context(vStcStatic, vStcCompute0, vStcCompute1, vBufs);

  return ggml_backend_buffer_init(pBuft, ggml_backend_meta_buffer_iface, vBufCtx, vMaxSize);
}

struct ggml_backend_buffer *ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context *pCtx,
                                                                          ggml_backend_buffer_type_t pBuft) {
  const size_t cNSimpleBufts = fGgmlBackendMetaBuftNBufts(pBuft);

  constexpr size_t cComputeHeadroom =
      16; // Maximum number of views per statically allocated tensor that can be created between evals.
  const ggml_init_params cParamsStatic = {
      /*.mem_size   =*/ggml_get_mem_size(pCtx),
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  const ggml_init_params cParamsCompute = {
      /*.mem_size   =*/cComputeHeadroom * ggml_get_mem_size(pCtx),
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  ggml_backend_meta_simple_tensor_container vStcStatic(cParamsStatic, cNSimpleBufts);
  ggml_backend_meta_simple_tensor_container vStcCompute0(cParamsCompute, cNSimpleBufts);
  ggml_backend_meta_simple_tensor_container vStcCompute1(cParamsCompute, cNSimpleBufts);

  std::vector<ggml_backend_buffer_t> vBufs(cNSimpleBufts, nullptr);
  ggml_backend_meta_buffer_context *vMetaBufCtx =
      new ggml_backend_meta_buffer_context(vStcStatic, vStcCompute0, vStcCompute1, vBufs);

  ggml_backend_buffer_t vMetaBuf = ggml_backend_buffer_init(pBuft, ggml_backend_meta_buffer_iface, vMetaBufCtx, 0);
  for (ggml_tensor *vT = ggml_get_first_tensor(pCtx); vT != nullptr; vT = ggml_get_next_tensor(pCtx, vT)) {
    vT->buffer = vMetaBuf;
    fGgmlBackendMetaBufferInitTensorImpl(vMetaBufCtx->vStcStatic, vT);
    vT->data = (void *)0x2000000000000000; // FIXME
  }
  for (size_t vI = 0; vI < cNSimpleBufts; vI++) {
    ggml_context *vCtx = vMetaBufCtx->vStcStatic.lCtxs[vI].get();
    ggml_backend_buffer_type_t vSimpleBuft = fGgmlBackendMetaBuftSimpleBuft(pBuft, vI);

    // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
    // For those edge cases, allocate a dummy buffer instead.
    bool vAnyNonzeroSlice = false;
    for (ggml_tensor *vT = ggml_get_first_tensor(vCtx); vT != nullptr; vT = ggml_get_next_tensor(vCtx, vT)) {
      if (ggml_nelements(vT) != 0) {
        vAnyNonzeroSlice = true;
        break;
      }
    }
    if (vAnyNonzeroSlice) {
      vMetaBufCtx->vBufs[vI].reset(ggml_backend_alloc_ctx_tensors_from_buft(vCtx, vSimpleBuft));
    } else {
      vMetaBufCtx->vBufs[vI].reset(ggml_backend_buft_alloc_buffer(vSimpleBuft, 0));
      for (ggml_tensor *vT = ggml_get_first_tensor(vCtx); vT != nullptr; vT = ggml_get_next_tensor(vCtx, vT)) {
        vT->buffer = vMetaBufCtx->vBufs[vI].get();
      }
    }
    GGML_ASSERT(vMetaBufCtx->vBufs[vI]);
    vMetaBuf->size = std::max(vMetaBuf->size, ggml_backend_buffer_get_size(vMetaBufCtx->vBufs[vI].get()));
  }
  return vMetaBuf;
}

//
// meta backend
//

static ggml_guid_t fGgmlBackendMetaGuid() {
  static ggml_guid aGuid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb,
                            0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
  return &aGuid;
}

struct ggml_backend_meta_context {
  struct cgraph_config {
    ggml_cgraph *vCgraphMain = nullptr;
    int vOffset = 0; // Node offset vs. original graph

    std::vector<ggml_cgraph *> lCgraphsAux;
  };
  struct backend_config {
    ggml_backend_t vBackend;

    std::vector<cgraph_config> lCgraphs;
    std::vector<ggml_tensor *> vNodes;
    std::vector<ggml_backend_buffer_ptr> vBufs;

    backend_config(ggml_backend_t pBackend, const size_t pNReduceSteps) : vBackend(pBackend) {
      vBufs.resize(pNReduceSteps);
    }
  };
  std::string vName;
  std::vector<backend_config> lBackendConfigs;
  ggml_context_ptr vCtx;
  std::vector<ggml_cgraph *> lCgraphsAux;
  std::vector<ggml_tensor *> lNodesAux;
  size_t vNReduceSteps;
  int vMaxNnodes = 0;
  size_t vMaxTmpSize = 0;
  size_t vMaxSubgraphs = 0;
  size_t vNSubgraphs = 0;
  uint64_t vUid = 0;

  void *vCommCtx = nullptr;
  ggml_backend_comm_allreduce_tensor_t vCommAllreduce = nullptr;

  ggml_backend_meta_context(ggml_backend_dev_t pMetaDev, const char *pParams) {
    const size_t cNDevs = fGgmlBackendMetaDevNDevs(pMetaDev);
    vNReduceSteps = std::ceil(std::log2(cNDevs));
    vName = "Meta(";
    std::vector<ggml_backend_t> lSimpleBackends;
    lBackendConfigs.reserve(cNDevs);
    lSimpleBackends.reserve(cNDevs);
    for (size_t vI = 0; vI < cNDevs; vI++) {
      ggml_backend_dev_t vSimpleDev = fGgmlBackendMetaDevSimpleDev(pMetaDev, vI);
      if (vI > 0) {
        vName += ",";
      }
      vName += ggml_backend_dev_name(vSimpleDev);
      lSimpleBackends.push_back(ggml_backend_dev_init(vSimpleDev, pParams));
      lBackendConfigs.emplace_back(lSimpleBackends.back(), vNReduceSteps);
    }
    vName += ")";

    if (cNDevs > 1) {
      ggml_backend_comm_init_t vCommInit = (ggml_backend_comm_init_t)ggml_backend_reg_get_proc_address(
          ggml_backend_dev_backend_reg(ggml_backend_get_device(lSimpleBackends[0])), "ggml_backend_comm_init");
      if (vCommInit != nullptr) {
        vCommCtx = vCommInit(lSimpleBackends.data(), lSimpleBackends.size());
      }
    }
    if (vCommCtx != nullptr) {
      vCommAllreduce = (ggml_backend_comm_allreduce_tensor_t)ggml_backend_reg_get_proc_address(
          ggml_backend_dev_backend_reg(ggml_backend_get_device(lSimpleBackends[0])),
          "ggml_backend_comm_allreduce_tensor");
      GGML_ASSERT(vCommAllreduce != nullptr);
    }
  }

  ~ggml_backend_meta_context() {
    if (vCommCtx != nullptr) {
      ggml_backend_comm_free_t vCommFree = (ggml_backend_comm_free_t)ggml_backend_reg_get_proc_address(
          ggml_backend_dev_backend_reg(ggml_backend_get_device(lBackendConfigs[0].vBackend)), "ggml_backend_comm_free");
      GGML_ASSERT(vCommFree != nullptr);
      vCommFree(vCommCtx);
    }
    for (auto &vBc : lBackendConfigs) {
      ggml_backend_free(vBc.vBackend);
    }
  }
};

static const char *fGgmlBackendMetaGetName(ggml_backend_t pBackend) {
  GGML_ASSERT(ggml_backend_is_meta(pBackend));
  const ggml_backend_meta_context *vBackendCtx = (const ggml_backend_meta_context *)pBackend->context;
  return vBackendCtx->vName.c_str();
}

static void fGgmlBackendMetaFree(ggml_backend_t pBackend) {
  GGML_ASSERT(ggml_backend_is_meta(pBackend));
  ggml_backend_meta_context *vBackendCtx = (ggml_backend_meta_context *)pBackend->context;
  delete vBackendCtx;
  delete pBackend;
}

static void fGgmlBackendMetaSetTensorAsync(ggml_backend_t pBackend, ggml_tensor *pTensor, const void *pData,
                                           size_t pOffset, size_t pSize) {
  const size_t cNBackends = ggml_backend_meta_n_backends(pBackend);
  GGML_ASSERT(pOffset == 0);
  GGML_ASSERT(ggml_is_contiguous(pTensor));

  const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pTensor, /*assume_sync =*/false);
  GGML_ASSERT(cSplitState.n_segments == 1);
  GGML_ASSERT(cSplitState.nr[0] == 1);

  switch (cSplitState.axis) {
  case GGML_BACKEND_SPLIT_AXIS_0:
  case GGML_BACKEND_SPLIT_AXIS_1:
  case GGML_BACKEND_SPLIT_AXIS_2: {
    // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
    const size_t cChunkSizeFull = pTensor->nb[cSplitState.axis + 1];
    GGML_ASSERT(pOffset % cChunkSizeFull == 0);
    GGML_ASSERT(pSize % cChunkSizeFull == 0);
    const int64_t cIStart = pOffset / cChunkSizeFull;
    const int64_t cIStop = (pOffset + pSize) / cChunkSizeFull;
    size_t vOffsetJ = 0;
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      ggml_backend_t vSimpleBackend = ggml_backend_meta_simple_backend(pBackend, vJ);
      ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      const size_t cChunkSizeJ = vSimpleTensor->nb[cSplitState.axis + 1];
      if (cChunkSizeJ == 0) {
        continue;
      }
      ggml_backend_tensor_set_2d_async(vSimpleBackend, vSimpleTensor, (const char *)pData + vOffsetJ, pOffset,
                                       cChunkSizeJ, cIStop - cIStart, cChunkSizeJ, cChunkSizeFull);
      vOffsetJ += cChunkSizeJ;
    }
    GGML_ASSERT(vOffsetJ == cChunkSizeFull);
  } break;
  case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      ggml_backend_tensor_set_async(ggml_backend_meta_simple_backend(pBackend, vJ),
                                    fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ), pData, pOffset, pSize);
    }
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlBackendMetaGetTensorAsync(ggml_backend_t pBackend, const ggml_tensor *pTensor, void *pData,
                                           size_t pOffset, size_t pSize) {
  const size_t cNBackends = ggml_backend_meta_n_backends(pBackend);
  GGML_ASSERT(pOffset == 0);
  GGML_ASSERT(ggml_is_contiguous(pTensor));

  const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(pTensor, /*assume_sync =*/false);
  GGML_ASSERT(cSplitState.n_segments == 1);
  GGML_ASSERT(cSplitState.nr[0] == 1);

  switch (cSplitState.axis) {
  case GGML_BACKEND_SPLIT_AXIS_0:
  case GGML_BACKEND_SPLIT_AXIS_1:
  case GGML_BACKEND_SPLIT_AXIS_2: {
    // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
    const size_t cChunkSizeFull = pTensor->nb[cSplitState.axis + 1];
    GGML_ASSERT(pOffset % cChunkSizeFull == 0);
    GGML_ASSERT(pSize % cChunkSizeFull == 0);
    const int64_t cIStart = pOffset / cChunkSizeFull;
    const int64_t cIStop = (pOffset + pSize) / cChunkSizeFull;
    size_t vOffsetJ = 0;
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      ggml_backend_t vSimpleBackend = ggml_backend_meta_simple_backend(pBackend, vJ);
      const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, vJ);
      const size_t cChunkSizeJ = vSimpleTensor->nb[cSplitState.axis + 1];
      if (cChunkSizeJ == 0) {
        continue;
      }
      ggml_backend_tensor_get_2d_async(vSimpleBackend, vSimpleTensor, (char *)pData + vOffsetJ, pOffset, cChunkSizeJ,
                                       cIStop - cIStart, cChunkSizeJ, cChunkSizeFull);
      vOffsetJ += cChunkSizeJ;
    }
    GGML_ASSERT(vOffsetJ == cChunkSizeFull);
  } break;
  case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    // TODO other simple backend may be better
    ggml_backend_t vSimpleBackend = ggml_backend_meta_simple_backend(pBackend, 0);
    const ggml_tensor *vSimpleTensor = fGgmlBackendMetaBufferSimpleTensor(pTensor, 0);
    ggml_backend_tensor_get_async(vSimpleBackend, vSimpleTensor, pData, pOffset, pSize);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlBackendMetaSynchronize(ggml_backend_t pBackend) {
  const size_t cNBackends = ggml_backend_meta_n_backends(pBackend);
  for (size_t vI = 0; vI < cNBackends; vI++) {
    ggml_backend_synchronize(ggml_backend_meta_simple_backend(pBackend, vI));
  }
}

static enum ggml_status fGgmlBackendMetaGraphCompute(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph) {
  GGML_ASSERT(pCgraph->grads == nullptr);
  const size_t cNBackends = ggml_backend_meta_n_backends(pBackend);
  ggml_backend_meta_context *vBackendCtx = (ggml_backend_meta_context *)pBackend->context;

  // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
  const bool cNeedsRebuild = (pCgraph->uid == 0) || (pCgraph->uid != vBackendCtx->vUid);

  bool vMaxNnodesRaised = false;
  if (pCgraph->n_nodes > vBackendCtx->vMaxNnodes) {
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
      vBcj.vNodes.resize(pCgraph->n_nodes);
      vBcj.lCgraphs.resize(pCgraph->n_nodes);
    }
    vBackendCtx->vMaxNnodes = pCgraph->n_nodes;
    vMaxNnodesRaised = true;
    assert(cNeedsRebuild);
  }

  if (cNeedsRebuild) {
    std::set<ggml_backend_buffer_t> sUsedBuffers;
    for (int vI = 0; vI < pCgraph->n_leafs; vI++) {
      if (ggml_backend_buffer_is_meta(pCgraph->leafs[vI]->buffer)) {
        sUsedBuffers.emplace(pCgraph->leafs[vI]->buffer);
      }
    }
    for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
      if (ggml_backend_buffer_is_meta(pCgraph->nodes[vI]->buffer)) {
        sUsedBuffers.emplace(pCgraph->nodes[vI]->buffer);
      }
    }
    for (ggml_backend_buffer_t vBuf : sUsedBuffers) {
      ggml_backend_meta_buffer_context *vBufCtx = (ggml_backend_meta_buffer_context *)vBuf->context;
      vBufCtx->vStcComputeIndexNext = vBufCtx->vStcComputeIndex ^ 1;
      ggml_backend_meta_simple_tensor_container &vStc = vBufCtx->aStcCompute[vBufCtx->vStcComputeIndexNext];
      for (ggml_context_ptr &vCtx : vStc.lCtxs) {
        ggml_reset(vCtx.get());
      }
      vStc.dSimpleTensors.clear();
    }
    size_t vNSubgraphs = 0;
    size_t vMaxTmpSize = 0;

    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      auto &vBcj = vBackendCtx->lBackendConfigs[vJ];

      for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
        ggml_tensor *vNode = pCgraph->nodes[vI];
        if (vNode->view_src != nullptr && vNode->view_src->op == GGML_OP_NONE &&
            ggml_backend_buffer_is_host(vNode->view_src->buffer)) {
          // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
          // For regular usage this doesn't matter since it's a noop but trying to call
          // ggml_backend_meta_buffer_simple_tensor results in a crash.
          vBcj.vNodes[vI] = vNode;
          continue;
        }
        vBcj.vNodes[vI] = fGgmlBackendMetaBufferSimpleTensor(vNode, vJ);
        GGML_ASSERT(vBcj.vNodes[vI]);
      }
    }

    {
      // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
      auto fGetIDelayed = [&](const int pI) -> int {
        int vId = pI;  // i_delayed
        int vIdr = pI; // i_delayed return, last safe return value

        ggml_tensor *vNode = pCgraph->nodes[vId];
        int32_t vNUsed = ggml_node_get_use_count(pCgraph, vId);

        // Skip MIRRORED nodes that don't consume node
        auto fSkipUnrelated = [&]() {
          while (vId + 1 < pCgraph->n_nodes) {
            ggml_tensor *vNext = pCgraph->nodes[vId + 1];
            if (fGgmlBackendMetaGetSplitState(vNext, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
              break;
            }
            bool vSafe = true;
            for (int vS = 0; vS < GGML_MAX_SRC; vS++) {
              if (vNext->src[vS] == nullptr) {
                continue;
              }
              if (vNext->src[vS] == vNode) {
                vSafe = false;
                break;
              }
              if (fGgmlBackendMetaGetSplitState(vNext->src[vS], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                vSafe = false;
                break;
              }
            }
            if (!vSafe) {
              break;
            }
            vId++;
          }
        };

        fSkipUnrelated();
        if (vId + 1 >= pCgraph->n_nodes) {
          return vIdr;
        }
        {
          ggml_tensor *vNext = pCgraph->nodes[vId + 1];
          if (vNext->op == GGML_OP_ADD_ID && vNext->src[0] == vNode &&
              fGgmlBackendMetaGetSplitState(vNext->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
              fGgmlBackendMetaGetSplitState(vNext->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            vNode = vNext;
            vId++;
            vIdr = vId;
            vNUsed = ggml_node_get_use_count(pCgraph, vId);
          }
        }
        // Chain of MULs with MIRRORED src[1]
        while (true) {
          fSkipUnrelated();
          if (vId + 1 >= pCgraph->n_nodes) {
            return vIdr;
          }
          ggml_tensor *vNext = pCgraph->nodes[vId + 1];
          if (vNext->op == GGML_OP_MUL && vNext->src[0] == vNode &&
              fGgmlBackendMetaGetSplitState(vNext->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            vNode = vNext;
            vId++;
            vIdr = vId;
            vNUsed = ggml_node_get_use_count(pCgraph, vId);
          } else {
            break;
          }
        }

        if (vNUsed != vNode->ne[1] || vId + 2 * vNUsed - 1 >= pCgraph->n_nodes) {
          return vIdr;
        }
        for (int32_t vK = 0; vK < vNUsed; vK++) {
          ggml_tensor *vNext = pCgraph->nodes[vId + 1];
          if (vNext->op != GGML_OP_VIEW || vNext->view_src != vNode || vNext->view_offs != vK * vNode->nb[1] ||
              vNext->ne[0] != vNode->ne[0] || vNext->ne[1] != vNode->ne[2] || vNext->nb[1] != vNode->nb[2] ||
              ggml_node_get_use_count(pCgraph, vId + 1) != 1) {
            return vIdr;
          }
          vId++;
        }
        {
          ggml_tensor *vNext = pCgraph->nodes[vId + 1];
          if (vNext->op != GGML_OP_ADD || vNext->src[0] != pCgraph->nodes[vId - (vNUsed - 1)] ||
              vNext->src[1] != pCgraph->nodes[vId - (vNUsed - 2)] || ggml_node_get_use_count(pCgraph, vId + 1) != 1) {
            return vIdr;
          }
          vId++;
        }
        for (int32_t vK = 0; vK < vNUsed - 2; vK++) {
          ggml_tensor *vNext = pCgraph->nodes[vId + 1];
          if (vNext->op != GGML_OP_ADD || vNext->src[0] != pCgraph->nodes[vId] ||
              vNext->src[1] != pCgraph->nodes[vId - (vNUsed - 2)] || ggml_node_get_use_count(pCgraph, vId + 1) != 1) {
            return vIdr;
          }
          vId++;
        }
        vIdr = vId;
        return vIdr;
      };

      int vIStart = 0;
      for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
        ggml_tensor *vNode = pCgraph->nodes[vI];
        if (vNode->view_src != nullptr && vNode->view_src->op == GGML_OP_NONE &&
            ggml_backend_buffer_is_host(vNode->view_src->buffer)) {
          continue;
        }
        const ggml_backend_meta_split_state cSplitState = fGgmlBackendMetaGetSplitState(vNode, /*assume_sync =*/false);
        if (cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
          vMaxTmpSize = std::max(vMaxTmpSize, ggml_nbytes(vNode));
        }
        const bool cNewSubgraph = vI + 1 == pCgraph->n_nodes || cSplitState.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
        if (!cNewSubgraph) {
          continue;
        }

        const int cIDelayed = fGetIDelayed(vI);

        // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
        // A backend with such a slice would normally have valid data after participating in the AllReduce with a node
        // that has
        //     its compute flag disabled and thus gets its data zeroed out.
        // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
        if (cIDelayed > vI) {
          for (size_t vJ = 0; vJ < cNBackends; vJ++) {
            auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
            if ((vBcj.vNodes[vI]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
              for (int vIi = vI + 1; vIi <= cIDelayed; vIi++) {
                vBcj.vNodes[vIi]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
              }
            }
          }
        }

        vI = cIDelayed;

        for (size_t vJ = 0; vJ < cNBackends; vJ++) {
          auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
          vBcj.lCgraphs[vNSubgraphs].vOffset = vIStart;
        }
        vNSubgraphs++;
        vIStart = vI + 1;
      }
      GGML_ASSERT(vIStart == pCgraph->n_nodes);
    }

    vBackendCtx->vUid = pCgraph->uid;
    vBackendCtx->vNSubgraphs = vNSubgraphs;

    if (vMaxTmpSize > vBackendCtx->vMaxTmpSize) {
      for (size_t vJ = 0; vJ < cNBackends; vJ++) {
        auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
        for (size_t vI = 0; vI < vBackendCtx->vNReduceSteps; vI++) {
          vBcj.vBufs[vI].reset(ggml_backend_alloc_buffer(vBcj.vBackend, vMaxTmpSize));
        }
      }
      vBackendCtx->vMaxTmpSize = vMaxTmpSize;
    }

    if (vMaxNnodesRaised || vNSubgraphs > vBackendCtx->vMaxSubgraphs) {
      vBackendCtx->vMaxSubgraphs = std::max(vBackendCtx->vMaxSubgraphs, vNSubgraphs);
      const size_t cNNodesPerDevice = 3 * vBackendCtx->vNReduceSteps; // tmp + ADD (+zeroing) graph per step and device
      const size_t cNCgraphsPerDevice = 2 * vBackendCtx->vNReduceSteps; // ADD ( + zeroing) graph per step and device
      const size_t cMemPerDeviceGraphsMain =
          vBackendCtx->vMaxSubgraphs * ggml_graph_overhead_custom(vBackendCtx->vMaxNnodes, pCgraph->grads);
      const size_t cMemPerDeviceGraphsAux =
          cNCgraphsPerDevice * vBackendCtx->vMaxSubgraphs * ggml_graph_overhead_custom(1, pCgraph->grads);
      const size_t cMemPerDeviceNodesAux = cNNodesPerDevice * vBackendCtx->vMaxSubgraphs * ggml_tensor_overhead();
      const ggml_init_params cParams = {
          /*.mem_size   =*/cNBackends * (cMemPerDeviceGraphsMain + cMemPerDeviceGraphsAux + cMemPerDeviceNodesAux),
          /*.mem_buffer =*/nullptr,
          /*.no_alloc   =*/true,
      };
      vBackendCtx->vCtx.reset(ggml_init(cParams));
      for (size_t vJ = 0; vJ < cNBackends; vJ++) {
        auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
        for (size_t vI = 0; vI < vNSubgraphs; vI++) {
          vBcj.lCgraphs[vI].vCgraphMain =
              ggml_new_graph_custom(vBackendCtx->vCtx.get(), pCgraph->n_nodes, /*grads =*/false);
        }
      }
      vBackendCtx->lCgraphsAux.resize(cNBackends * cNCgraphsPerDevice * vBackendCtx->vMaxSubgraphs);
      for (size_t vK = 0; vK < vBackendCtx->lCgraphsAux.size(); vK++) {
        vBackendCtx->lCgraphsAux[vK] = ggml_new_graph_custom(vBackendCtx->vCtx.get(), 1, pCgraph->grads);
      }
      vBackendCtx->lNodesAux.resize(cNBackends * cNNodesPerDevice * vBackendCtx->vMaxSubgraphs);
      for (size_t vK = 0; vK < vBackendCtx->lNodesAux.size(); vK++) {
        vBackendCtx->lNodesAux[vK] = ggml_new_tensor_1d(vBackendCtx->vCtx.get(), GGML_TYPE_F32, 1);
      }
    }

    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
      for (size_t vIGraph = 0; vIGraph < vNSubgraphs; vIGraph++) {
        ggml_cgraph *vCgraphIj = vBcj.lCgraphs[vIGraph].vCgraphMain;
        const size_t cINodeStart = vBcj.lCgraphs[vIGraph].vOffset;
        const size_t cINodeStop = vIGraph + 1 < vNSubgraphs ? vBcj.lCgraphs[vIGraph + 1].vOffset : pCgraph->n_nodes;
        vCgraphIj->n_nodes = cINodeStop - cINodeStart;
        ggml_hash_set_reset(&vCgraphIj->visited_hash_set);
        for (size_t vINode = cINodeStart; vINode < cINodeStop; vINode++) {
          ggml_tensor *vNodeIj = vBcj.vNodes[vINode];
          vCgraphIj->nodes[vINode - cINodeStart] = vNodeIj;
          const size_t cHashPosOrig = ggml_hash_find(&pCgraph->visited_hash_set, pCgraph->nodes[vINode]);
          const size_t cHashPosIj = ggml_hash_insert(&vCgraphIj->visited_hash_set, vNodeIj);
          vCgraphIj->use_counts[cHashPosIj] = pCgraph->use_counts[cHashPosOrig];
        }
        vCgraphIj->uid = ggml_graph_next_uid();
      }
    }
  }

  size_t vIga = 0; // i graph aux
  size_t vIna = 0; // i node aux

  auto fGetNodeAux = [&](ggml_tensor *pT) -> ggml_tensor * {
    ggml_tensor *vRet = vBackendCtx->lNodesAux[vIna++];
    memset(vRet, 0, sizeof(ggml_tensor));
    vRet->op = GGML_OP_NONE;
    vRet->type = pT->type;
    for (size_t vK = 0; vK < GGML_MAX_DIMS; vK++) {
      vRet->ne[vK] = pT->ne[vK];
      vRet->nb[vK] = pT->nb[vK];
    }
    return vRet;
  };
  auto fSetTmpData = [&](ggml_tensor *pTensor, const size_t pJ, const size_t pIBuf) {
    auto &vBcj = vBackendCtx->lBackendConfigs[pJ];
    ggml_backend_buffer_ptr &vBufPtr = vBcj.vBufs[pIBuf];
    if (!vBufPtr || ggml_backend_buffer_get_size(vBufPtr.get()) < vBackendCtx->vMaxTmpSize) {
      vBufPtr.reset(ggml_backend_alloc_buffer(vBcj.vBackend, vBackendCtx->vMaxTmpSize));
    }
    pTensor->buffer = vBufPtr.get();
    pTensor->data = ggml_backend_buffer_get_base(vBufPtr.get());
  };
  // FIXME usage_counts
  auto fGetCgraphAux = [&]() -> ggml_cgraph * {
    ggml_cgraph *vRet = vBackendCtx->lCgraphsAux[vIga++];
    return vRet;
  };

  // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if
  // unavailable:
  auto fAllreduceFallback = [&](size_t pI) -> ggml_status {
    std::vector<ggml_cgraph *> vStepCgraphs(cNBackends, nullptr);

    // Zero out nodes that were disabled due to having a zero-sized slice:
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
      ggml_tensor *vNode = vBcj.lCgraphs[pI].vCgraphMain->nodes[vBcj.lCgraphs[pI].vCgraphMain->n_nodes - 1];
      if (vNode->flags & GGML_TENSOR_FLAG_COMPUTE) {
        continue;
      }
      ggml_tensor *vNodeZero = fGetNodeAux(vNode);
      vNodeZero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
      vNodeZero->src[0] = vNode;
      ggml_set_op_params_f32(vNodeZero, 0, 0.0f);
      vNodeZero->data = vNode->data;
      vNodeZero->buffer = vNode->buffer;
      vNodeZero->flags |= GGML_TENSOR_FLAG_COMPUTE;

      vStepCgraphs[vJ] = fGetCgraphAux();
      vStepCgraphs[vJ]->nodes[0] = vNodeZero;
      vStepCgraphs[vJ]->n_nodes = 1;
      const ggml_status cStatus = ggml_backend_graph_compute_async(vBcj.vBackend, vStepCgraphs[vJ]);
      if (cStatus != GGML_STATUS_SUCCESS) {
        return cStatus;
      }
    }
    std::fill(vStepCgraphs.begin(), vStepCgraphs.end(), nullptr);

    auto fPushData = [&](const size_t pJSrc, const size_t pJDst, const size_t pIBuf) {
      assert(vStepCgraphs[pJDst] == nullptr);
      auto &vBcjSrc = vBackendCtx->lBackendConfigs[pJSrc];
      auto &vBcjDst = vBackendCtx->lBackendConfigs[pJDst];

      ggml_tensor *vNodeSrc = vBcjSrc.lCgraphs[pI].vCgraphMain->nodes[vBcjSrc.lCgraphs[pI].vCgraphMain->n_nodes - 1];
      ggml_tensor *vNodeDst = vBcjDst.lCgraphs[pI].vCgraphMain->nodes[vBcjDst.lCgraphs[pI].vCgraphMain->n_nodes - 1];
      GGML_ASSERT(ggml_is_contiguous(vNodeSrc));
      GGML_ASSERT(ggml_is_contiguous(vNodeDst));

      ggml_tensor *vNodeTmp = fGetNodeAux(vNodeDst);
      fSetTmpData(vNodeTmp, pJDst, pIBuf);

      ggml_backend_tensor_copy_async(vBcjSrc.vBackend, vBcjDst.vBackend, vNodeSrc, vNodeTmp);

      ggml_tensor *vNodeRed = fGetNodeAux(vNodeDst);
      vNodeRed->view_src = vNodeDst->view_src == nullptr ? vNodeDst : vNodeDst->view_src;
      vNodeRed->view_offs = vNodeDst->view_offs;
      vNodeRed->op = GGML_OP_ADD;
      vNodeRed->src[0] = vNodeDst;
      vNodeRed->src[1] = vNodeTmp;
      vNodeRed->flags |= GGML_TENSOR_FLAG_COMPUTE;
      ggml_backend_view_init(vNodeRed);

      ggml_cgraph *vCgraphAux = fGetCgraphAux();
      vCgraphAux->nodes[0] = vNodeRed;
      vCgraphAux->n_nodes = 1;
      vStepCgraphs[pJDst] = vCgraphAux;
    };

    size_t vOffsetJ = cNBackends / 2;
    while ((vOffsetJ & (vOffsetJ - 1)) != 0) {
      vOffsetJ--;
    }
    const size_t cOffsetJMax = vOffsetJ;
    size_t vIBuf = 0;

    // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
    for (size_t vJSrc = 2 * cOffsetJMax; vJSrc < cNBackends; vJSrc++) {
      const size_t cJDst = vJSrc - 2 * cOffsetJMax;
      fPushData(vJSrc, cJDst, vIBuf);
      const ggml_status cStatus =
          ggml_backend_graph_compute_async(vBackendCtx->lBackendConfigs[cJDst].vBackend, vStepCgraphs[cJDst]);
      if (cStatus != GGML_STATUS_SUCCESS) {
        return cStatus;
      }
      vIBuf = 1;
    }

    // Butterfly reduction:
    for (; vOffsetJ >= 1; vOffsetJ /= 2) {
      std::fill(vStepCgraphs.begin(), vStepCgraphs.end(), nullptr);

      for (size_t vJ = 0; vJ < 2 * cOffsetJMax; vJ++) {
        const size_t cJOther = vJ ^ vOffsetJ;
        if (cJOther >= cNBackends) {
          continue;
        }
        fPushData(vJ, cJOther, vIBuf);
      }

      for (size_t vJ = 0; vJ < 2 * cOffsetJMax; vJ++) {
        if (vStepCgraphs[vJ] == nullptr) {
          continue;
        }
        auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
        const ggml_status cStatus = ggml_backend_graph_compute_async(vBcj.vBackend, vStepCgraphs[vJ]);
        if (cStatus != GGML_STATUS_SUCCESS) {
          return cStatus;
        }
      }
      vIBuf++;
    }
    assert(vIBuf == vBackendCtx->vNReduceSteps);

    // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
    for (size_t vJ = 2 * cOffsetJMax; vJ < cNBackends; vJ++) {
      auto &vBcjSrc = vBackendCtx->lBackendConfigs[vJ - 2 * cOffsetJMax];
      auto &vBcjDst = vBackendCtx->lBackendConfigs[vJ];

      ggml_tensor *vNodeSrc = vBcjSrc.lCgraphs[pI].vCgraphMain->nodes[vBcjSrc.lCgraphs[pI].vCgraphMain->n_nodes - 1];
      ggml_tensor *vNodeDst = vBcjDst.lCgraphs[pI].vCgraphMain->nodes[vBcjDst.lCgraphs[pI].vCgraphMain->n_nodes - 1];
      ggml_backend_tensor_copy_async(vBcjSrc.vBackend, vBcjDst.vBackend, vNodeSrc, vNodeDst);
    }

    return GGML_STATUS_SUCCESS;
  };

  for (size_t vI = 0; vI < vBackendCtx->vNSubgraphs; vI++) {
    for (size_t vJ = 0; vJ < cNBackends; vJ++) {
      auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
      const ggml_status cStatus = ggml_backend_graph_compute_async(vBcj.vBackend, vBcj.lCgraphs[vI].vCgraphMain);
      if (cStatus != GGML_STATUS_SUCCESS) {
        return cStatus;
      }
    }

    if (cNBackends > 1 && vI < vBackendCtx->vNSubgraphs - 1) {
      bool vBackendAllreduceSuccess = false;
      if (vBackendCtx->vCommCtx) {
        std::vector<ggml_tensor *> vNodes;
        vNodes.reserve(cNBackends);
        for (size_t vJ = 0; vJ < cNBackends; vJ++) {
          auto &vBcj = vBackendCtx->lBackendConfigs[vJ];
          ggml_cgraph *vCgraphIj = vBcj.lCgraphs[vI].vCgraphMain;
          vNodes.push_back(vCgraphIj->nodes[vCgraphIj->n_nodes - 1]);
        }
        vBackendAllreduceSuccess = vBackendCtx->vCommAllreduce(vBackendCtx->vCommCtx, vNodes.data());
      }

      if (!vBackendAllreduceSuccess) {
        const ggml_status cStatus = fAllreduceFallback(vI);
        if (cStatus != GGML_STATUS_SUCCESS) {
          return cStatus;
        }
      }
    }
  }
  return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ fGgmlBackendMetaGetName,
    /* .free                    = */ fGgmlBackendMetaFree,
    /* .set_tensor_async        = */ fGgmlBackendMetaSetTensorAsync,
    /* .get_tensor_async        = */ fGgmlBackendMetaGetTensorAsync,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ fGgmlBackendMetaSynchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ fGgmlBackendMetaGraphCompute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t pBackend) {
  return pBackend != nullptr && pBackend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t fGgmlBackendMetaDeviceInitBackend(ggml_backend_dev_t pDev, const char *pParams) {
  ggml_backend_meta_context *vBackendCtx = new ggml_backend_meta_context(pDev, pParams);

  ggml_backend_t vBackend = new struct ggml_backend;
  vBackend->guid = fGgmlBackendMetaGuid();
  vBackend->iface = ggml_backend_meta_i;
  vBackend->device = pDev;
  vBackend->context = vBackendCtx;
  return vBackend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t pMetaBackend) {
  GGML_ASSERT(ggml_backend_is_meta(pMetaBackend));
  const ggml_backend_meta_context *vBackendCtx = (const ggml_backend_meta_context *)pMetaBackend->context;
  return vBackendCtx->lBackendConfigs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t pMetaBackend, size_t pIndex) {
  GGML_ASSERT(ggml_backend_is_meta(pMetaBackend));
  const ggml_backend_meta_context *vBackendCtx = (const ggml_backend_meta_context *)pMetaBackend->context;
  return vBackendCtx->lBackendConfigs[pIndex].vBackend;
}
