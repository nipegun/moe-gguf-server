// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

// backend buffer type

const char *ggml_backend_buft_name(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBuft);
  return pBuft->iface.get_name(pBuft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  GGML_ASSERT(pBuft);
  if (pSize == 0) {
    // return a dummy buffer for zero-sized allocations
    return ggml_backend_buffer_init(pBuft, {}, NULL, 0);
  }
  return pBuft->iface.alloc_buffer(pBuft, pSize);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBuft);
  return pBuft->iface.get_alignment(pBuft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBuft);
  // get_max_size is optional, defaults to SIZE_MAX
  if (pBuft->iface.get_max_size) {
    return pBuft->iface.get_max_size(pBuft);
  }
  return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t pBuft, const struct ggml_tensor *pTensor) {
  GGML_ASSERT(pBuft);
  // get_alloc_size is optional, defaults to ggml_nbytes
  if (pBuft->iface.get_alloc_size) {
    size_t vSize = pBuft->iface.get_alloc_size(pBuft, pTensor);
    assert(vSize >= ggml_nbytes(pTensor));
    return vSize;
  }
  return ggml_nbytes(pTensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBuft);
  if (pBuft->iface.is_host) {
    return pBuft->iface.is_host(pBuft);
  }
  return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBuft);
  return pBuft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(ggml_backend_buffer_type_t pBuft, struct ggml_backend_buffer_i pIface,
                                               void *pContext, size_t pSize) {
  ggml_backend_buffer_t vBuffer = new ggml_backend_buffer{/* .interface = */ pIface,
                                                          /* .buft      = */ pBuft,
                                                          /* .context   = */ pContext,
                                                          /* .size      = */ pSize,
                                                          /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY};

  return vBuffer;
}

const char *ggml_backend_buffer_name(ggml_backend_buffer_t pBuffer) {
  return ggml_backend_buft_name(ggml_backend_buffer_get_type(pBuffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t pBuffer) {
  if (pBuffer == NULL) {
    return;
  }

  if (pBuffer->iface.free_buffer != NULL) {
    pBuffer->iface.free_buffer(pBuffer);
  }
  delete pBuffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  return pBuffer->size;
}

void *ggml_backend_buffer_get_base(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  // get_base is optional if the buffer is zero-sized
  if (!ggml_backend_buffer_is_meta(pBuffer) && pBuffer->size == 0) {
    return NULL;
  }

  // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
  //     I don't know whether the above comment is correct
  if (!pBuffer->iface.get_base) {
    return NULL;
  }

  void *vBase = pBuffer->iface.get_base(pBuffer);

  GGML_ASSERT(vBase != NULL && "backend buffer base cannot be NULL");

  return vBase;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor) {
  GGML_ASSERT(pBuffer);
  // init_tensor is optional
  if (pBuffer->iface.init_tensor) {
    return pBuffer->iface.init_tensor(pBuffer, pTensor);
  }
  return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  GGML_ASSERT(pBuffer);
  // clear is optional if the buffer is zero-sized
  if (pBuffer->size == 0) {
    return;
  }

  pBuffer->iface.clear(pBuffer, pValue);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t pBuffer) {
  return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(pBuffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t pBuffer) {
  return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(pBuffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t pBuffer, const struct ggml_tensor *pTensor) {
  return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(pBuffer), pTensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t pBuffer) {
  return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(pBuffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t pBuffer, enum ggml_backend_buffer_usage pUsage) {
  GGML_ASSERT(pBuffer);
  pBuffer->usage = pUsage;

  // FIXME: add a generic callback to the buffer interface
  if (ggml_backend_buffer_is_multi_buffer(pBuffer)) {
    ggml_backend_multi_buffer_set_usage(pBuffer, pUsage);
  }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  return pBuffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  return pBuffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  if (pBuffer->iface.reset) {
    pBuffer->iface.reset(pBuffer);
  }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor *pSrc, struct ggml_tensor *pDst) {
  ggml_backend_buffer_t vDstBuf = pDst->view_src ? pDst->view_src->buffer : pDst->buffer;
  if (vDstBuf->iface.cpy_tensor) {
    return vDstBuf->iface.cpy_tensor(vDstBuf, pSrc, pDst);
  }
  return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t pBackend) {
  if (pBackend == NULL) {
    return NULL;
  }
  return pBackend->guid;
}

const char *ggml_backend_name(ggml_backend_t pBackend) {
  if (pBackend == NULL) {
    return "NULL";
  }
  return pBackend->iface.get_name(pBackend);
}

void ggml_backend_free(ggml_backend_t pBackend) {
  if (pBackend == NULL) {
    return;
  }

  pBackend->iface.free(pBackend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t pBackend) {
  GGML_ASSERT(pBackend);
  return ggml_backend_dev_buffer_type(pBackend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t pBackend, size_t pSize) {
  return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(pBackend), pSize);
}

size_t ggml_backend_get_alignment(ggml_backend_t pBackend) {
  return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(pBackend));
}

size_t ggml_backend_get_max_size(ggml_backend_t pBackend) {
  return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(pBackend));
}

void ggml_backend_tensor_set_async(ggml_backend_t pBackend, struct ggml_tensor *pTensor, const void *pData,
                                   size_t pOffset, size_t pSize) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + pSize <= ggml_nbytes(pTensor) && "tensor write out of bounds");

  if (pBackend->iface.set_tensor_async == NULL) {
    ggml_backend_synchronize(pBackend);
    ggml_backend_tensor_set(pTensor, pData, pOffset, pSize);
  } else {
    pBackend->iface.set_tensor_async(pBackend, pTensor, pData, pOffset, pSize);
  }
}

void ggml_backend_tensor_get_async(ggml_backend_t pBackend, const struct ggml_tensor *pTensor, void *pData,
                                   size_t pOffset, size_t pSize) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + pSize <= ggml_nbytes(pTensor) && "tensor read out of bounds");

  if (pBackend->iface.get_tensor_async == NULL) {
    ggml_backend_synchronize(pBackend);
    ggml_backend_tensor_get(pTensor, pData, pOffset, pSize);
  } else {
    pBackend->iface.get_tensor_async(pBackend, pTensor, pData, pOffset, pSize);
  }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t pBackend, struct ggml_tensor *pTensor, const void *pData,
                                      size_t pOffset, size_t pSize, size_t pNCopies, size_t pStrideTensor,
                                      size_t pStrideData) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");

  if (pNCopies <= 1 || pBackend->iface.set_tensor_2d_async == NULL) {
    for (size_t vI = 0; vI < pNCopies; vI++) {
      ggml_backend_tensor_set_async(pBackend, pTensor, (const char *)pData + vI * pStrideData,
                                    pOffset + vI * pStrideTensor, pSize);
    }
    return;
  }
  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + (pNCopies - 1) * pStrideTensor + pSize <= ggml_nbytes(pTensor) && "tensor write out of bounds");
  pBackend->iface.set_tensor_2d_async(pBackend, pTensor, pData, pOffset, pSize, pNCopies, pStrideTensor, pStrideData);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t pBackend, const struct ggml_tensor *pTensor, void *pData,
                                      size_t pOffset, size_t pSize, size_t pNCopies, size_t pStrideTensor,
                                      size_t pStrideData) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");

  if (pNCopies <= 1 || pBackend->iface.get_tensor_2d_async == NULL) {
    for (size_t vI = 0; vI < pNCopies; vI++) {
      ggml_backend_tensor_get_async(pBackend, pTensor, (char *)pData + vI * pStrideData, pOffset + vI * pStrideTensor,
                                    pSize);
    }
    return;
  }
  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + (pNCopies - 1) * pStrideTensor + pSize <= ggml_nbytes(pTensor) && "tensor read out of bounds");
  pBackend->iface.get_tensor_2d_async(pBackend, pTensor, pData, pOffset, pSize, pNCopies, pStrideTensor, pStrideData);
}

void ggml_backend_tensor_set(struct ggml_tensor *pTensor, const void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
  GGML_ASSERT(vBuf != NULL && "tensor buffer not set");

  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + pSize <= ggml_nbytes(pTensor) && "tensor write out of bounds");

  vBuf->iface.set_tensor(vBuf, pTensor, pData, pOffset, pSize);
}

void ggml_backend_tensor_get(const struct ggml_tensor *pTensor, void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
  GGML_ASSERT(vBuf != NULL && "tensor buffer not set");

  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + pSize <= ggml_nbytes(pTensor) && "tensor read out of bounds");

  vBuf->iface.get_tensor(vBuf, pTensor, pData, pOffset, pSize);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor *pTensor, const void *pData, size_t pOffset, size_t pSize,
                                size_t pNCopies, size_t pStrideTensor, size_t pStrideData) {
  GGML_ASSERT(pTensor);
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
  GGML_ASSERT(vBuf != NULL && "tensor buffer not set");

  if (pNCopies <= 1 || vBuf->iface.set_tensor_2d == NULL) {
    for (size_t vI = 0; vI < pNCopies; vI++) {
      ggml_backend_tensor_set(pTensor, (const char *)pData + vI * pStrideData, pOffset + vI * pStrideTensor, pSize);
    }
    return;
  }
  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + (pNCopies - 1) * pStrideTensor + pSize <= ggml_nbytes(pTensor) && "tensor write out of bounds");

  vBuf->iface.set_tensor_2d(vBuf, pTensor, pData, pOffset, pSize, pNCopies, pStrideTensor, pStrideData);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor *pTensor, void *pData, size_t pOffset, size_t pSize,
                                size_t pNCopies, size_t pStrideTensor, size_t pStrideData) {
  GGML_ASSERT(pTensor);
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
  GGML_ASSERT(vBuf != NULL && "tensor buffer not set");

  if (pNCopies <= 1 || vBuf->iface.get_tensor_2d == NULL) {
    for (size_t vI = 0; vI < pNCopies; vI++) {
      ggml_backend_tensor_get(pTensor, (char *)pData + vI * pStrideData, pOffset + vI * pStrideTensor, pSize);
    }
    return;
  }
  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + (pNCopies - 1) * pStrideTensor + pSize <= ggml_nbytes(pTensor) && "tensor read out of bounds");

  vBuf->iface.get_tensor_2d(vBuf, pTensor, pData, pOffset, pSize, pNCopies, pStrideTensor, pStrideData);
}

void ggml_backend_tensor_memset(struct ggml_tensor *pTensor, uint8_t pValue, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;

  if (pSize == 0) {
    return;
  }

  GGML_ASSERT(vBuf != NULL && "tensor buffer not set");
  GGML_ASSERT(pTensor->data != NULL && "tensor not allocated");
  GGML_ASSERT(pOffset + pSize <= ggml_nbytes(pTensor) && "tensor write out of bounds");
  GGML_ASSERT(vBuf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

  vBuf->iface.memset_tensor(vBuf, pTensor, pValue, pOffset, pSize);
}

void ggml_backend_synchronize(ggml_backend_t pBackend) {
  GGML_ASSERT(pBackend);
  if (pBackend->iface.synchronize == NULL) {
    return;
  }

  pBackend->iface.synchronize(pBackend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pBackend->iface.graph_plan_create != NULL);

  return pBackend->iface.graph_plan_create(pBackend, pCgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t pBackend, ggml_backend_graph_plan_t pPlan) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pBackend->iface.graph_plan_free != NULL);

  pBackend->iface.graph_plan_free(pBackend, pPlan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t pBackend, ggml_backend_graph_plan_t pPlan) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pBackend->iface.graph_plan_compute != NULL);

  return pBackend->iface.graph_plan_compute(pBackend, pPlan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph) {
  enum ggml_status vErr = ggml_backend_graph_compute_async(pBackend, pCgraph);
  ggml_backend_synchronize(pBackend);
  return vErr;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph) {
  GGML_ASSERT(pBackend);
  return pBackend->iface.graph_compute(pBackend, pCgraph);
}

bool ggml_backend_supports_op(ggml_backend_t pBackend, const struct ggml_tensor *pOp) {
  GGML_ASSERT(pBackend);
  return ggml_backend_dev_supports_op(pBackend->device, pOp);
}

bool ggml_backend_supports_buft(ggml_backend_t pBackend, ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pBackend);
  return ggml_backend_dev_supports_buft(pBackend->device, pBuft);
}

bool ggml_backend_offload_op(ggml_backend_t pBackend, const struct ggml_tensor *pOp) {
  GGML_ASSERT(pBackend);
  return ggml_backend_dev_offload_op(pBackend->device, pOp);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t pBackend) {
  GGML_ASSERT(pBackend);
  return pBackend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor *pSrc, struct ggml_tensor *pDst) {
  GGML_ASSERT(ggml_are_same_layout(pSrc, pDst) && "cannot copy tensors with different layouts");

  if (pSrc == pDst) {
    return;
  }

  if (ggml_backend_buffer_is_host(pSrc->buffer)) {
    ggml_backend_tensor_set(pDst, pSrc->data, 0, ggml_nbytes(pSrc));
  } else if (ggml_backend_buffer_is_host(pDst->buffer)) {
    ggml_backend_tensor_get(pSrc, pDst->data, 0, ggml_nbytes(pSrc));
  } else if (!ggml_backend_buffer_copy_tensor(pSrc, pDst)) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(pSrc->buffer),
                   ggml_backend_buffer_name(pDst->buffer));
#endif // NDEBUG
    size_t vNbytes = ggml_nbytes(pSrc);
    void *vData = malloc(vNbytes);
    ggml_backend_tensor_get(pSrc, vData, 0, vNbytes);
    ggml_backend_tensor_set(pDst, vData, 0, vNbytes);
    free(vData);
  }
}

void ggml_backend_tensor_copy_async(ggml_backend_t pBackendSrc, ggml_backend_t pBackendDst,
                                    const struct ggml_tensor *pSrc, struct ggml_tensor *pDst) {
  GGML_ASSERT(ggml_are_same_layout(pSrc, pDst) && "cannot copy tensors with different layouts");

  if (pSrc == pDst) {
    return;
  }

  GGML_ASSERT(pBackendDst);
  if (pBackendDst->iface.cpy_tensor_async != NULL) {
    if (pBackendDst->iface.cpy_tensor_async(pBackendSrc, pBackendDst, pSrc, pDst)) {
      return;
    }
  }

  // an async copy would normally happen after all the queued operations on both backends are completed
  // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
  ggml_backend_synchronize(pBackendSrc);
  ggml_backend_synchronize(pBackendDst);
  ggml_backend_tensor_copy(pSrc, pDst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t pDevice) {
  // null device is allowed for the transition period to the device interface
  if (pDevice == NULL || pDevice->iface.event_new == NULL) {
    return NULL;
  }
  return pDevice->iface.event_new(pDevice);
}

void ggml_backend_event_free(ggml_backend_event_t pEvent) {
  if (pEvent == NULL) {
    return;
  }
  pEvent->device->iface.event_free(pEvent->device, pEvent);
}

void ggml_backend_event_record(ggml_backend_event_t pEvent, ggml_backend_t pBackend) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pBackend->iface.event_record != NULL);

  pBackend->iface.event_record(pBackend, pEvent);
}

void ggml_backend_event_synchronize(ggml_backend_event_t pEvent) {
  GGML_ASSERT(pEvent);
  GGML_ASSERT(pEvent->device->iface.event_synchronize);

  pEvent->device->iface.event_synchronize(pEvent->device, pEvent);
}

void ggml_backend_event_wait(ggml_backend_t pBackend, ggml_backend_event_t pEvent) {
  GGML_ASSERT(pBackend);
  GGML_ASSERT(pBackend->iface.event_wait != NULL);

  pBackend->iface.event_wait(pBackend, pEvent);
}

static void fGgmlBackendGraphOptimize(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph,
                                      struct ggml_backend_graph_optimize_params *pParams) {
  GGML_ASSERT(pBackend);
  if (pBackend->iface.graph_optimize != NULL) {
    pBackend->iface.graph_optimize(pBackend, pCgraph, pParams);
  }
}

// Backend device

const char *ggml_backend_dev_name(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.get_name(pDevice);
}

const char *ggml_backend_dev_description(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.get_description(pDevice);
}

void ggml_backend_dev_memory(ggml_backend_dev_t pDevice, size_t *pFree, size_t *pTotal) {
  GGML_ASSERT(pDevice);
  pDevice->iface.get_memory(pDevice, pFree, pTotal);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.get_type(pDevice);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t pDevice, struct ggml_backend_dev_props *pProps) {
  GGML_ASSERT(pDevice);
  memset(pProps, 0, sizeof(*pProps));
  pDevice->iface.get_props(pDevice, pProps);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  return pDevice->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t pDevice, const char *pParams) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.init_backend(pDevice, pParams);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.get_buffer_type(pDevice);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t pDevice) {
  GGML_ASSERT(pDevice);
  if (pDevice->iface.get_host_buffer_type == NULL) {
    return NULL;
  }

  return pDevice->iface.get_host_buffer_type(pDevice);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t pDevice, void *pPtr, size_t pSize,
                                                            size_t pMaxTensorSize) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.buffer_from_host_ptr(pDevice, pPtr, pSize, pMaxTensorSize);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t pDevice, const struct ggml_tensor *pOp) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.supports_op(pDevice, pOp);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t pDevice, ggml_backend_buffer_type_t pBuft) {
  GGML_ASSERT(pDevice);
  return pDevice->iface.supports_buft(pDevice, pBuft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t pDevice, const struct ggml_tensor *pOp) {
  GGML_ASSERT(pDevice);
  if (pDevice->iface.offload_op != NULL) {
    return pDevice->iface.offload_op(pDevice, pOp);
  }

  return false;
}

// Backend (reg)

const char *ggml_backend_reg_name(ggml_backend_reg_t pReg) {
  GGML_ASSERT(pReg);
  return pReg->iface.get_name(pReg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t pReg) {
  GGML_ASSERT(pReg);
  return pReg->iface.get_device_count(pReg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t pReg, size_t pIndex) {
  GGML_ASSERT(pReg);
  return pReg->iface.get_device(pReg, pIndex);
}

void *ggml_backend_reg_get_proc_address(ggml_backend_reg_t pReg, const char *pName) {
  GGML_ASSERT(pReg);
  if (!pReg->iface.get_proc_address) {
    return NULL;
  }
  return pReg->iface.get_proc_address(pReg, pName);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
  ggml_backend_buffer_t *vBuffers;
  size_t vNBuffers;
};

static void fGgmlBackendMultiBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  ggml_backend_multi_buffer_context *vCtx = (ggml_backend_multi_buffer_context *)pBuffer->context;
  for (size_t vI = 0; vI < vCtx->vNBuffers; vI++) {
    ggml_backend_buffer_free(vCtx->vBuffers[vI]);
  }

  free(vCtx->vBuffers);
  free(vCtx);
}

static void fGgmlBackendMultiBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  GGML_ASSERT(pBuffer);
  ggml_backend_multi_buffer_context *vCtx = (ggml_backend_multi_buffer_context *)pBuffer->context;
  for (size_t vI = 0; vI < vCtx->vNBuffers; vI++) {
    ggml_backend_buffer_clear(vCtx->vBuffers[vI], pValue);
  }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ fGgmlBackendMultiBufferFreeBuffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ fGgmlBackendMultiBufferClear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t *pBuffers, size_t pNBuffers) {
  ggml_backend_multi_buffer_context *vCtx =
      (ggml_backend_multi_buffer_context *)malloc(sizeof(struct ggml_backend_multi_buffer_context));
  vCtx->vNBuffers = pNBuffers;
  vCtx->vBuffers = (ggml_backend_buffer_t *)malloc(pNBuffers * sizeof(ggml_backend_buffer_t));

  GGML_ASSERT(vCtx->vBuffers != NULL);

  size_t vTotalSize = 0;
  for (size_t vI = 0; vI < pNBuffers; vI++) {
    vCtx->vBuffers[vI] = pBuffers[vI];
    vTotalSize += ggml_backend_buffer_get_size(pBuffers[vI]);
  }

  return ggml_backend_buffer_init(pBuffers[0]->buft, ggml_backend_multi_buffer_i, vCtx, vTotalSize);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  return pBuffer->iface.free_buffer == fGgmlBackendMultiBufferFreeBuffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t pBuffer, enum ggml_backend_buffer_usage pUsage) {
  GGML_ASSERT(pBuffer);
  GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(pBuffer));
  ggml_backend_multi_buffer_context *vCtx = (ggml_backend_multi_buffer_context *)pBuffer->context;
  for (size_t vI = 0; vI < vCtx->vNBuffers; vI++) {
    ggml_backend_buffer_set_usage(vCtx->vBuffers[vI], pUsage);
  }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor *fGgmlDupTensorLayout(struct ggml_context *pCtx, const struct ggml_tensor *pTensor) {
  struct ggml_tensor *vDup = ggml_dup_tensor(pCtx, pTensor);
  for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
    vDup->nb[vI] = pTensor->nb[vI];
  }
  return vDup;
}

static bool fGgmlIsViewOp(enum ggml_op pOp) {
  return pOp == GGML_OP_VIEW || pOp == GGML_OP_RESHAPE || pOp == GGML_OP_PERMUTE || pOp == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS
#define GGML_SCHED_MAX_PREFETCH_SLOTS 8
#endif

struct ggml_backend_sched_split {
  int backend_id;
  int vIStart;
  int vIEnd;
  struct ggml_tensor **vInputs;
  int vNInputs;
  int vInputsCapacity;
  // graph view of this split
  struct ggml_cgraph vGraph;
};

struct ggml_backend_sched {
  bool vIsReset; // true if the scheduler has been reset since the last graph split
  bool vIsAlloc;

  int n_backends;

  ggml_backend_t aBackends[GGML_SCHED_MAX_BACKENDS];
  ggml_backend_buffer_type_t vBufts[GGML_SCHED_MAX_BACKENDS];
  ggml_gallocr_t vGalloc;

  // hash map of the nodes in the graph
  struct ggml_hash_set hash_set;
  int *hv_tensor_backend_ids;            // [hash_set.size]
  struct ggml_tensor **hv_tensor_copies; // [hash_set.size][n_backends][n_copies]

  int *vNodeBackendIds; // [graph_size]
  int *vLeafBackendIds; // [graph_size]

  int *vPrevNodeBackendIds; // [graph_size]
  int *vPrevLeafBackendIds; // [graph_size]

  // copy of the graph with modified inputs
  struct ggml_cgraph vGraph;

  // graph splits
  struct ggml_backend_sched_split *vSplits;
  int vNSplits;
  int vSplitsCapacity;

  // pipeline parallelism support
  int n_copies;
  int vCurCopy;
  int vNextCopy;
  ggml_backend_event_t aEvents[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
  struct ggml_tensor **vGraphInputs;
  int vNGraphInputs;
  int vGraphInputsCapacity;

  struct ggml_context *vCtx;

  ggml_backend_sched_eval_callback vCallbackEval;
  void *vCallbackEvalUserData;

  // checked before every split (ggml_backend_sched_set_abort_callback)
  ggml_abort_callback vAbortCallback;
  void *vAbortCallbackData;

  char *vContextBuffer;
  size_t vContextBufferSize;

  bool vOpOffload;

  // Prefetch entire offloaded expert tensors. Use a separate backend instance
  // to overlap transfers with computation and rotate through multiple slots.
  bool vPrefetchExperts;
  ggml_backend_t vPrefetchBackend;
  ggml_backend_dev_t vPrefetchDevice;
  int vPrefetchSlotCount;
  ggml_backend_buffer_t aAPrefetchSlots[GGML_SCHED_MAX_PREFETCH_SLOTS];
  ggml_backend_event_t aAPrefetchReady[GGML_SCHED_MAX_PREFETCH_SLOTS];
  ggml_backend_event_t aAPrefetchFree[GGML_SCHED_MAX_PREFETCH_SLOTS];
  bool aAPrefetchUsed[GGML_SCHED_MAX_PREFETCH_SLOTS];
  int vPrefetchCurrent;
  // Device memory the slots leave free (ggml_backend_sched_set_prefetch_reserve).
  size_t vPrefetchReserve;

  // Optional provider for the prefetch uploads (device-resident copies of host weights).
  ggml_backend_sched_weight_upload_t vWeightUpload;
  void *vWeightUploadData;

  int vDebug;

  // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
  int vDebugRealloc;
  int vDebugGraphSize;
  int vDebugPrevGraphSize;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id)                                                                        \
  sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void fGgmlBackendSchedSplitInputsGrow(struct ggml_backend_sched_split *pSplit) {
  int vNewCap = GGML_SCHED_MAX_SPLIT_INPUTS;
  if (pSplit->vInputsCapacity > 0) {
    vNewCap = 2 * pSplit->vInputsCapacity;
    GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, pSplit->vInputsCapacity, vNewCap);
  }
  auto *vPnew = (struct ggml_tensor **)realloc((void *)pSplit->vInputs, vNewCap * sizeof(struct ggml_tensor *));
  if (vPnew == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, vNewCap * sizeof(struct ggml_tensor *));
    GGML_ABORT("failed to grow split inputs container");
  }
  pSplit->vInputs = vPnew;
  pSplit->vInputsCapacity = vNewCap;
}

static void fGgmlBackendSchedGraphInputsGrow(ggml_backend_sched_t pSched) {
  int vNewCap = GGML_SCHED_MAX_SPLIT_INPUTS;
  if (pSched->vGraphInputsCapacity > 0) {
    vNewCap = 2 * pSched->vGraphInputsCapacity;
    GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, pSched->vGraphInputsCapacity,
                   vNewCap);
  }
  auto *vPnew = (struct ggml_tensor **)realloc((void *)pSched->vGraphInputs, vNewCap * sizeof(struct ggml_tensor *));
  if (vPnew == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, vNewCap * sizeof(struct ggml_tensor *));
    GGML_ABORT("failed to grow graph inputs container");
  }
  pSched->vGraphInputs = vPnew;
  pSched->vGraphInputsCapacity = vNewCap;
}

// returns the priority of the backend, lower id is higher priority
static int fGgmlBackendSchedBackendId(ggml_backend_sched_t pSched, ggml_backend_t pBackend) {
  for (int vI = 0; vI < pSched->n_backends; vI++) {
    if (pSched->aBackends[vI] == pBackend) {
      return vI;
    }
  }
  return -1;
}

static int fGgmlBackendSchedBackendFromBuffer(ggml_backend_sched_t pSched, const struct ggml_tensor *pTensor,
                                              const struct ggml_tensor *pOp) {
  ggml_backend_buffer_t vBuffer = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
  if (vBuffer == NULL) {
    return -1;
  }

  // find highest prio backend that supports the buffer type and the op
  for (int vI = 0; vI < pSched->n_backends; vI++) {
    if (ggml_backend_supports_buft(pSched->aBackends[vI], vBuffer->buft) &&
        ggml_backend_supports_op(pSched->aBackends[vI], pOp)) {
      return vI;
    }
  }

#ifndef NDEBUG
  GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the "
                 "weight will need to be copied\n",
                 __func__, ggml_op_desc(pTensor), ggml_backend_buffer_name(vBuffer), pTensor->name);
#endif

  return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int fGgmlBackendSchedBackendIdFromCur(ggml_backend_sched_t pSched, struct ggml_tensor *pTensor) {
  // assign pre-allocated nodes to their backend
  int vCurBackendId = fGgmlBackendSchedBackendFromBuffer(pSched, pTensor, pTensor);
  if (vCurBackendId != -1) {
    SET_CAUSE(pTensor, "1.dst");
    return vCurBackendId;
  }

  // view_src
  if (pTensor->view_src != NULL) {
    vCurBackendId = fGgmlBackendSchedBackendFromBuffer(pSched, pTensor->view_src, pTensor);
    if (vCurBackendId != -1) {
      SET_CAUSE(pTensor, "1.vsrc");
      return vCurBackendId;
    }
  }

  if (pTensor->buffer || (pTensor->view_src && pTensor->view_src->buffer)) {
    // since the tensor is pre-allocated, it cannot be moved to another backend
    ggml_backend_buffer_t vBuffer = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;
    GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", pTensor->name,
               ggml_backend_buffer_name(vBuffer), ggml_op_name(pTensor->op));
  }

  // graph input
  if (pTensor->flags & GGML_TENSOR_FLAG_INPUT) {
    vCurBackendId = pSched->n_backends - 1; // last backend (assumed CPU)
    SET_CAUSE(pTensor, "1.inp");
    return vCurBackendId;
  }

  // operations with weights are preferably run on the same backend as the weights
  // TODO: there are exceptions (see below) - not an ideal solution
  bool vAllow = true;

  // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
  vAllow = vAllow && pTensor->op != GGML_OP_ROPE;

  // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
  vAllow = vAllow && pTensor->op != GGML_OP_FLASH_ATTN_EXT;

  if (vAllow) {
    for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
      const struct ggml_tensor *vSrc = pTensor->src[vI];
      if (vSrc == NULL) {
        continue;
      }
      if (vSrc->buffer != NULL && vSrc->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        int vSrcBackendId = fGgmlBackendSchedBackendFromBuffer(pSched, vSrc, pTensor);
        // check if a backend with higher prio wants to offload the op
        if (pSched->vOpOffload && vSrcBackendId == pSched->n_backends - 1 &&
            ggml_backend_buffer_is_host(vSrc->buffer)) {
          for (int vB = 0; vB < vSrcBackendId; vB++) {
            if (ggml_backend_supports_op(pSched->aBackends[vB], pTensor) &&
                ggml_backend_offload_op(pSched->aBackends[vB], pTensor)) {
              SET_CAUSE(pTensor, "1.off");
              return vB;
            }
          }
        }
        SET_CAUSE(pTensor, "1.wgt%d", vI);
        return vSrcBackendId;
      }
    }
  }

  return -1;
}

static char *fFmtSize(size_t pSize) {
  static char vBuffer[128];
  if (pSize >= 1024 * 1024) {
    snprintf(vBuffer, sizeof(vBuffer), "%zuM", pSize / 1024 / 1024);
  } else {
    snprintf(vBuffer, sizeof(vBuffer), "%zuK", pSize / 1024);
  }
  return vBuffer;
}

static void fGgmlBackendSchedPrintAssignments(ggml_backend_sched_t pSched, struct ggml_cgraph *pGraph) {
  int vCurSplit = 0;
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    if (vCurSplit < pSched->vNSplits && vI == pSched->vSplits[vCurSplit].vIStart) {
      ggml_backend_t vSplitBackend = pSched->aBackends[pSched->vSplits[vCurSplit].backend_id];
      GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", vCurSplit, ggml_backend_name(vSplitBackend),
                     pSched->vSplits[vCurSplit].vNInputs);
      for (int vJ = 0; vJ < pSched->vSplits[vCurSplit].vNInputs; vJ++) {
        if (vJ == 0) {
          GGML_LOG_DEBUG(": ");
        }
        GGML_LOG_DEBUG("[%s (%5.5s)] ", pSched->vSplits[vCurSplit].vInputs[vJ]->name,
                       fFmtSize(ggml_nbytes(pSched->vSplits[vCurSplit].vInputs[vJ])));
      }
      GGML_LOG_DEBUG("\n");
      vCurSplit++;
    }
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    if (fGgmlIsViewOp(vNode->op)) {
      continue;
    }
    if (pSched->vDebug > 1) {
      ggml_backend_t vTensorBackend = ggml_backend_sched_get_tensor_backend(pSched, vNode);
      GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", vI, ggml_op_desc(vNode),
                     vNode->name, fFmtSize(ggml_nbytes(vNode)),
                     vTensorBackend ? ggml_backend_name(vTensorBackend) : "NULL", GET_CAUSE(vNode),
                     pGraph->use_counts[ggml_hash_find(&pGraph->visited_hash_set, vNode)],
                     vNode->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
      for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
        struct ggml_tensor *vSrc = vNode->src[vJ];
        if (vSrc == NULL) {
          continue;
        }
        ggml_backend_t vSrcBackend = ggml_backend_sched_get_tensor_backend(pSched, vSrc);
        GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", vSrc->name, fFmtSize(ggml_nbytes(vSrc)),
                       vSrcBackend ? ggml_backend_name(vSrcBackend) : "NULL", GET_CAUSE(vSrc));
      }
      GGML_LOG_DEBUG("\n");
    }
  }
}

static bool fGgmlBackendSchedBufferSupported(ggml_backend_sched_t sched, struct ggml_tensor *pT, int pBackendId) {
  ggml_backend_buffer_t vBuf = pT->view_src ? pT->view_src->buffer : pT->buffer;
  ggml_backend_buffer_type_t vBuft = NULL;

  if (vBuf) {
    // the tensor is already allocated
    vBuft = vBuf->buft;
  } else {
    // see if the tensor already has a backend assigned, and use the buffer type of that backend
    int vTensorBackendId = tensor_backend_id(pT);
    if (vTensorBackendId == -1 && pT->view_src) {
      vTensorBackendId = tensor_backend_id(pT->view_src);
    }
    if (vTensorBackendId != -1) {
      vBuft = sched->vBufts[vTensorBackendId];
    }
  }

  return vBuft != NULL && ggml_backend_supports_buft(sched->aBackends[pBackendId], vBuft);
}

static void fGgmlBackendSchedSetIfSupported(ggml_backend_sched_t pSched, struct ggml_tensor *pNode, int pCurBackendId,
                                            int *pNodeBackendId) {
  if (ggml_backend_supports_op(pSched->aBackends[pCurBackendId], pNode)) {
    *pNodeBackendId = pCurBackendId;
    SET_CAUSE(pNode, "2.sup");
  }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph *pGraph) {
  // reset splits
  sched->vNSplits = 0;
  sched->vNGraphInputs = 0;
  sched->vIsReset = false;

  struct ggml_init_params vParams = {/* .mem_size =   */ sched->vContextBufferSize,
                                     /* .mem_buffer = */ sched->vContextBuffer,
                                     /* .no_alloc =   */ true};

  ggml_free(sched->vCtx);

  sched->vCtx = ggml_init(vParams);
  if (sched->vCtx == NULL) {
    GGML_ABORT("%s: failed to initialize context\n", __func__);
  }

  pGraph->uid = ggml_graph_next_uid();

  // pass 1: assign backends to ops with pre-allocated inputs
  for (int vI = 0; vI < pGraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pGraph->leafs[vI];
    int *vLeafBackendId = &tensor_backend_id(vLeaf);
    // do not overwrite user assignments
    if (*vLeafBackendId == -1) {
      *vLeafBackendId = fGgmlBackendSchedBackendIdFromCur(sched, vLeaf);
    }
  }

  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    int *vNodeBackendId = &tensor_backend_id(vNode);
    // do not overwrite user assignments
    if (*vNodeBackendId == -1) {
      *vNodeBackendId = fGgmlBackendSchedBackendIdFromCur(sched, vNode);

#if 0
            // src
            if (vNode->op == GGML_OP_NONE) {
                continue;
            }

            for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
                struct ggml_tensor * vSrc = vNode->src[vJ];
                if (vSrc == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(vSrc);
                if (*src_backend_id == -1) {
                    *src_backend_id = fGgmlBackendSchedBackendIdFromCur(sched, vSrc);
                }
            }
#endif
    }
  }

  // pass 2: expand current backend assignments
  // assign the same backend to adjacent nodes
  // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
  // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
  // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the
  // locations of its inputs are known expand gpu down
  {
    int vCurBackendId = -1;
    for (int vI = 0; vI < pGraph->n_nodes; vI++) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];
      if (fGgmlIsViewOp(vNode->op)) {
        continue;
      }
      int *vNodeBackendId = &tensor_backend_id(vNode);
      if (*vNodeBackendId != -1) {
        if (*vNodeBackendId == sched->n_backends - 1) {
          // skip cpu (lowest prio backend)
          vCurBackendId = -1;
        } else {
          vCurBackendId = *vNodeBackendId;
        }
      } else if (vCurBackendId != -1) {
        fGgmlBackendSchedSetIfSupported(sched, vNode, vCurBackendId, vNodeBackendId);
      }
    }
  }
  // expand gpu up
  {
    int vCurBackendId = -1;
    for (int vI = pGraph->n_nodes - 1; vI >= 0; vI--) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];
      if (fGgmlIsViewOp(vNode->op)) {
        continue;
      }
      int *vNodeBackendId = &tensor_backend_id(vNode);
      if (*vNodeBackendId != -1) {
        if (*vNodeBackendId == sched->n_backends - 1) {
          // skip cpu (lowest prio backend)
          vCurBackendId = -1;
        } else {
          vCurBackendId = *vNodeBackendId;
        }
      } else if (vCurBackendId != -1) {
        fGgmlBackendSchedSetIfSupported(sched, vNode, vCurBackendId, vNodeBackendId);
      }
    }
  }
  // expand rest down
  {
    int vCurBackendId = -1;
    for (int vI = 0; vI < pGraph->n_nodes; vI++) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];
      if (fGgmlIsViewOp(vNode->op)) {
        continue;
      }
      int *vNodeBackendId = &tensor_backend_id(vNode);
      if (*vNodeBackendId != -1) {
        vCurBackendId = *vNodeBackendId;
      } else if (vCurBackendId != -1) {
        fGgmlBackendSchedSetIfSupported(sched, vNode, vCurBackendId, vNodeBackendId);
      }
    }
  }
  // expand rest up
  {
    int vCurBackendId = -1;
    for (int vI = pGraph->n_nodes - 1; vI >= 0; vI--) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];
      if (fGgmlIsViewOp(vNode->op)) {
        continue;
      }
      int *vNodeBackendId = &tensor_backend_id(vNode);
      if (*vNodeBackendId != -1) {
        vCurBackendId = *vNodeBackendId;
      } else if (vCurBackendId != -1) {
        fGgmlBackendSchedSetIfSupported(sched, vNode, vCurBackendId, vNodeBackendId);
      }
    }
  }

  // pass 3: upgrade nodes to higher prio backends with compatible buffer types
  // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
  // however, we also need to verify that the sources are in compatible buffer types
  // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of
  // this tensor further down the graph however, this is slow to verify, so we have a more strict requirement that the
  // buffer type is the same this is not uncommon since multiple backends can use host memory, with the same buffer type
  // (eg. BLAS and CPU) additionally, set remaining unassigned nodes to the backend with the most supported inputs only
  // nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at
  // this point
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    if (fGgmlIsViewOp(vNode->op)) {
      continue;
    }
    int *vNodeBackendId = &tensor_backend_id(vNode);
    if (*vNodeBackendId == -1) {
      // unassigned node: find the backend with the most supported inputs
      int vNSupportedBest = -1;
      for (int vB = 0; vB < sched->n_backends; vB++) {
        if (ggml_backend_supports_op(sched->aBackends[vB], vNode)) {
          int vNSupported = 0;
          for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
            struct ggml_tensor *vSrc = vNode->src[vJ];
            if (vSrc == NULL) {
              continue;
            }
            if ((tensor_backend_id(vSrc) != -1 || tensor_backend_id(vSrc->view_src) != -1) &&
                fGgmlBackendSchedBufferSupported(sched, vSrc, vB)) {
              vNSupported++;
            }
          }
          if (vNSupported > vNSupportedBest) {
            vNSupportedBest = vNSupported;
            *vNodeBackendId = vB;
            SET_CAUSE(vNode, "3.best");
          }
        }
      }
    } else {
      // assigned node: upgrade to higher prio backend if possible
      for (int vB = 0; vB < *vNodeBackendId; vB++) {
        if (sched->vBufts[vB] == sched->vBufts[*vNodeBackendId] &&
            ggml_backend_supports_op(sched->aBackends[vB], vNode)) {
          bool vSupported = true;
          for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
            struct ggml_tensor *vSrc = vNode->src[vJ];
            if (vSrc == NULL) {
              continue;
            }
            if (!fGgmlBackendSchedBufferSupported(sched, vSrc, vB)) {
              vSupported = false;
              break;
            }
          }
          if (vSupported) {
            *vNodeBackendId = vB;
            SET_CAUSE(vNode, "3.upg");
            break;
          }
        }
      }
    }
  }

  // pass 4: assign backends to remaining src from dst and view_src
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    int *vCurBackendId = &tensor_backend_id(vNode);
    if (vNode->view_src != NULL && *vCurBackendId == -1) {
      *vCurBackendId = tensor_backend_id(vNode->view_src);
      SET_CAUSE(vNode, "4.vsrc");
    }
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vSrc = vNode->src[vJ];
      if (vSrc == NULL) {
        continue;
      }
      int *src_backend_id = &tensor_backend_id(vSrc);
      if (*src_backend_id == -1) {
        if (vSrc->view_src != NULL) {
          // views are always on the same backend as the source
          *src_backend_id = tensor_backend_id(vSrc->view_src);
          SET_CAUSE(vSrc, "4.vsrc");
        } else {
          *src_backend_id = *vCurBackendId;
          SET_CAUSE(vSrc, "4.cur");
        }
      }
    }
    // if the node is still unassigned, assign it to the first backend that supports it
    for (int vB = 0; vB < sched->n_backends && *vCurBackendId == -1; vB++) {
      fGgmlBackendSchedSetIfSupported(sched, vNode, vB, vCurBackendId);
    }
    GGML_ASSERT(*vCurBackendId != -1);
  }

  // pass 5: split graph, find tensors that need to be copied
  {
    int vISplit = 0;
    struct ggml_backend_sched_split *vSplit = &sched->vSplits[0];
    // find the backend of the first split, skipping view ops
    int vI = 0;
    for (; vI < pGraph->n_nodes; vI++) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];
      if (!fGgmlIsViewOp(vNode->op)) {
        vSplit->backend_id = tensor_backend_id(vNode);
        break;
      }
    }
    vSplit->vIStart = 0;
    vSplit->vNInputs = 0;
    int vCurBackendId = vSplit->backend_id;
    for (; vI < pGraph->n_nodes; vI++) {
      struct ggml_tensor *vNode = pGraph->nodes[vI];

      if (fGgmlIsViewOp(vNode->op)) {
        continue;
      }

      const int cNodeBackendId = tensor_backend_id(vNode);

      GGML_ASSERT(cNodeBackendId !=
                  -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

      // check if we should start a new split based on the sources of the current node
      bool vNeedNewSplit = false;
      if (cNodeBackendId == vCurBackendId && vSplit->vNInputs > 0) {
        for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
          struct ggml_tensor *vSrc = vNode->src[vJ];
          if (vSrc == NULL) {
            continue;
          }
          // check if a weight is on a different and incompatible backend
          // by starting a new split, the memory of the previously offloaded weights can be reused
          if (vSrc->buffer != NULL && vSrc->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = tensor_backend_id(vSrc);
            if (src_backend_id != vCurBackendId && !fGgmlBackendSchedBufferSupported(sched, vSrc, vCurBackendId)) {
              vNeedNewSplit = true;
              break;
            }
          }
        }
      }

      if (cNodeBackendId != vCurBackendId || vNeedNewSplit) {
        vSplit->vIEnd = vI;
        vISplit++;
        if (vISplit >= sched->vSplitsCapacity) {
          int vOldCap = sched->vSplitsCapacity;
          sched->vSplitsCapacity *= 2;
          sched->vSplits = (ggml_backend_sched_split *)realloc(
              sched->vSplits, sched->vSplitsCapacity * sizeof(struct ggml_backend_sched_split));
          GGML_ASSERT(sched->vSplits != NULL);
          for (int vK = vOldCap; vK < sched->vSplitsCapacity; vK++) {
            memset(&sched->vSplits[vK], 0, sizeof(struct ggml_backend_sched_split));
          }
        }
        vSplit = &sched->vSplits[vISplit];
        vSplit->backend_id = cNodeBackendId;
        vSplit->vIStart = vI;
        vSplit->vNInputs = 0;
        vCurBackendId = cNodeBackendId;
      }

      // find inputs that are not on the same backend
      for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
        struct ggml_tensor *vSrc = vNode->src[vJ];
        if (vSrc == NULL) {
          continue;
        }

        size_t vSrcId = hash_id(vSrc);
        const int src_backend_id = sched->hv_tensor_backend_ids[vSrcId];
        GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

        if (vSrc->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
          if (tensor_id_copy(vSrcId, src_backend_id, 0) == NULL) {
            ggml_backend_t vBackend = sched->aBackends[src_backend_id];
            for (int vC = 0; vC < sched->n_copies; vC++) {
              struct ggml_tensor *vTensorCopy;
              if (vC == sched->vCurCopy) {
                vTensorCopy = vSrc; // use the original tensor as the current copy
              } else {
                vTensorCopy = fGgmlDupTensorLayout(sched->vCtx, vSrc);
                ggml_format_name(vTensorCopy, "%s#%s#%d", ggml_backend_name(vBackend), vSrc->name, vC);
              }
              ggml_set_input(vTensorCopy);
              ggml_set_output(vTensorCopy); // prevent ggml-alloc from overwriting the tensor
              tensor_id_copy(vSrcId, src_backend_id, vC) = vTensorCopy;
              SET_CAUSE(vTensorCopy, "4.cpy");
            }
            int vNGraphInputs = sched->vNGraphInputs++;
            if (vNGraphInputs >= sched->vGraphInputsCapacity) {
              fGgmlBackendSchedGraphInputsGrow(sched);
            }
            sched->vGraphInputs[vNGraphInputs] = vSrc;
          }
        }

        if (src_backend_id != vCurBackendId && !fGgmlBackendSchedBufferSupported(sched, vSrc, vCurBackendId)) {
          // create a copy of the input in the split's backend
          if (tensor_id_copy(vSrcId, vCurBackendId, 0) == NULL) {
            ggml_backend_t vBackend = sched->aBackends[vCurBackendId];
            for (int vC = 0; vC < sched->n_copies; vC++) {
              struct ggml_tensor *vTensorCopy = fGgmlDupTensorLayout(sched->vCtx, vSrc);
              ggml_format_name(vTensorCopy, "%s#%s#%d", ggml_backend_name(vBackend), vSrc->name, vC);
              if (sched->n_copies > 1) {
                ggml_set_input(vTensorCopy);
                ggml_set_output(vTensorCopy); // prevent ggml-alloc from overwriting the tensor
              }
              tensor_id_copy(vSrcId, vCurBackendId, vC) = vTensorCopy;
              SET_CAUSE(vTensorCopy, "4.cpy");
            }
            int vNInputs = vSplit->vNInputs++;
            if (vNInputs >= vSplit->vInputsCapacity) {
              fGgmlBackendSchedSplitInputsGrow(vSplit);
            }
            vSplit->vInputs[vNInputs] = vSrc;
          }
          vNode->src[vJ] = tensor_id_copy(vSrcId, vCurBackendId, sched->vCurCopy);
        }
      }
    }
    vSplit->vIEnd = pGraph->n_nodes;
    sched->vNSplits = vISplit + 1;
  }

  if (sched->vDebug) {
    fGgmlBackendSchedPrintAssignments(sched, pGraph);
  }

  // swap node_backend_ids and leaf _backend_ids with prevs
  {
    int *vTmp = sched->vNodeBackendIds;
    sched->vNodeBackendIds = sched->vPrevNodeBackendIds;
    sched->vPrevNodeBackendIds = vTmp;

    vTmp = sched->vLeafBackendIds;
    sched->vLeafBackendIds = sched->vPrevLeafBackendIds;
    sched->vPrevLeafBackendIds = vTmp;
  }

  // Optimize the split graphs and collect the allocation dependencies that the backends add. This needs to happen
  // before we make graph_copy, so they are in sync.
  std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> dAllocDeps;

  struct ggml_backend_graph_optimize_params vOptParams = {
      /* .add_alloc_dep = */ [](void *pUserData, ggml_tensor *pTensor, ggml_tensor *pUntil) {
        auto &dDeps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *)pUserData;
        std::vector<ggml_tensor *> &lKeep = dDeps[pUntil];
        if (std::find(lKeep.begin(), lKeep.end(), pTensor) == lKeep.end()) {
          lKeep.push_back(pTensor);
        }
      },
      /* .user_data     = */ &dAllocDeps,
  };

  for (int vI = 0; vI < sched->vNSplits; vI++) {
    struct ggml_backend_sched_split *vSplit = &sched->vSplits[vI];
    vSplit->vGraph = ggml_graph_view(pGraph, vSplit->vIStart, vSplit->vIEnd);

    fGgmlBackendGraphOptimize(sched->aBackends[vSplit->backend_id], &vSplit->vGraph, &vOptParams);
  }

  // each dependency is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
  int vNDepNodes = 0;
  for (const auto &vIt : dAllocDeps) {
    vNDepNodes += (vIt.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
  }

  int vTotalInputs = sched->vNGraphInputs;
  for (int vI = 0; vI < sched->vNSplits; vI++) {
    vTotalInputs += sched->vSplits[vI].vNInputs;
  }
  int vGraphSize = std::max(pGraph->n_nodes, pGraph->n_leafs) + vTotalInputs * 2 * sched->n_copies + vNDepNodes;

  // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
  sched->vDebugPrevGraphSize = sched->vDebugGraphSize;
  sched->vDebugGraphSize = vGraphSize;

  if (sched->vGraph.size < vGraphSize) {
    sched->vGraph.size = vGraphSize;
    sched->vGraph.nodes = (ggml_tensor **)realloc(sched->vGraph.nodes, vGraphSize * sizeof(struct ggml_tensor *));
    sched->vGraph.leafs = (ggml_tensor **)realloc(sched->vGraph.leafs, vGraphSize * sizeof(struct ggml_tensor *));
    GGML_ASSERT(sched->vGraph.nodes != NULL);
    GGML_ASSERT(sched->vGraph.leafs != NULL);
  }
  sched->vGraph.n_nodes = 0;
  sched->vGraph.n_leafs = 0;

  struct ggml_cgraph *vGraphCopy = &sched->vGraph;

  int vNDepNodesAdded = 0;

  for (int vI = 0; vI < sched->vNSplits; vI++) {
    struct ggml_backend_sched_split *vSplit = &sched->vSplits[vI];

    // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
    for (int vJ = 0; vJ < vSplit->vNInputs; vJ++) {
      assert(vGraphCopy->size > (vGraphCopy->n_nodes + 1));

      struct ggml_tensor *vInput = vSplit->vInputs[vJ];
      const size_t cInputId = hash_id(vInput);
      struct ggml_tensor *vInputCpy = tensor_id_copy(cInputId, vSplit->backend_id, sched->vCurCopy);

      // add a dependency to the input source so that it is not freed before the copy is done
      struct ggml_tensor *vInputDep = ggml_view_tensor(sched->vCtx, vInput);
      vInputDep->src[0] = vInput;
      sched->vNodeBackendIds[vGraphCopy->n_nodes] = sched->hv_tensor_backend_ids[cInputId];
      vGraphCopy->nodes[vGraphCopy->n_nodes++] = vInputDep;

      // add a dependency to the input copy so that it is allocated at the start of the split
      sched->vNodeBackendIds[vGraphCopy->n_nodes] = vSplit->backend_id;
      vGraphCopy->nodes[vGraphCopy->n_nodes++] = vInputCpy;
    }

    for (int vJ = vSplit->vIStart; vJ < vSplit->vIEnd; vJ++) {
      assert(vGraphCopy->size > vGraphCopy->n_nodes);
      sched->vNodeBackendIds[vGraphCopy->n_nodes] = tensor_backend_id(pGraph->nodes[vJ]);
      vGraphCopy->nodes[vGraphCopy->n_nodes++] = pGraph->nodes[vJ];

      if (dAllocDeps.empty()) {
        continue;
      }

      // add a dependency node so that the kept tensors are not freed before this node is computed
      auto vIt = dAllocDeps.find(pGraph->nodes[vJ]);
      if (vIt != dAllocDeps.end()) {
        const std::vector<ggml_tensor *> &lKeep = vIt->second;
        for (size_t vK = 0; vK < lKeep.size(); vK += GGML_MAX_SRC) {
          struct ggml_tensor *vDep = ggml_view_tensor(sched->vCtx, lKeep[vK]);
          for (size_t vS = 0; vS < GGML_MAX_SRC && vK + vS < lKeep.size(); vS++) {
            vDep->src[vS] = lKeep[vK + vS];
          }
          assert(vGraphCopy->size > vGraphCopy->n_nodes);
          sched->vNodeBackendIds[vGraphCopy->n_nodes] = vSplit->backend_id;
          vGraphCopy->nodes[vGraphCopy->n_nodes++] = vDep;
          vNDepNodesAdded++;
        }
      }
    }
  }

  // a mismatch means a backend added a dependency with an `until` tensor that is not a node of the optimized graph
  GGML_ASSERT(vNDepNodesAdded == vNDepNodes);

  if (sched->n_copies > 1) {
    // add input copies as leafs so that they are allocated first
    for (int vI = 0; vI < sched->vNGraphInputs; vI++) {
      struct ggml_tensor *vInput = sched->vGraphInputs[vI];
      size_t vId = hash_id(vInput);
      int vBackendId = tensor_backend_id(vInput);
      for (int vC = 0; vC < sched->n_copies; vC++) {
        struct ggml_tensor *vInputCpy = tensor_id_copy(vId, vBackendId, vC);
        sched->vLeafBackendIds[vGraphCopy->n_leafs] = vBackendId;
        assert(vGraphCopy->size > vGraphCopy->n_leafs);
        vGraphCopy->leafs[vGraphCopy->n_leafs++] = vInputCpy;
      }
    }

    for (int vI = 0; vI < sched->vNSplits; vI++) {
      struct ggml_backend_sched_split *vSplit = &sched->vSplits[vI];
      int vBackendId = vSplit->backend_id;
      for (int vJ = 0; vJ < vSplit->vNInputs; vJ++) {
        struct ggml_tensor *vInput = vSplit->vInputs[vJ];
        size_t vId = hash_id(vInput);
        for (int vC = 0; vC < sched->n_copies; vC++) {
          struct ggml_tensor *vInputCpy = tensor_id_copy(vId, vBackendId, vC);
          sched->vLeafBackendIds[vGraphCopy->n_leafs] = vBackendId;
          assert(vGraphCopy->size > vGraphCopy->n_leafs);
          vGraphCopy->leafs[vGraphCopy->n_leafs++] = vInputCpy;
        }
      }
    }
  }

  // add leafs from the original graph
  for (int vI = 0; vI < pGraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pGraph->leafs[vI];
    sched->vLeafBackendIds[vGraphCopy->n_leafs] = tensor_backend_id(vLeaf);
    assert(vGraphCopy->size > vGraphCopy->n_leafs);
    vGraphCopy->leafs[vGraphCopy->n_leafs++] = vLeaf;
  }

  // set ids for all splits
  for (int vI = 0; vI < sched->vNSplits; ++vI) {
    sched->vSplits[vI].vGraph.uid = ggml_graph_next_uid();
  }
}

static bool fGgmlBackendSchedAllocSplits(ggml_backend_sched_t pSched) {
  bool vBackendIdsChanged = false;
  for (int vI = 0; vI < pSched->vGraph.n_nodes; vI++) {
    if (pSched->vNodeBackendIds[vI] != pSched->vPrevNodeBackendIds[vI] &&
        pSched->vBufts[pSched->vNodeBackendIds[vI]] != pSched->vBufts[pSched->vPrevNodeBackendIds[vI]]) {
      vBackendIdsChanged = true;
      break;
    }
  }
  if (!vBackendIdsChanged) {
    for (int vI = 0; vI < pSched->vGraph.n_leafs; vI++) {
      if (pSched->vLeafBackendIds[vI] != pSched->vPrevLeafBackendIds[vI] &&
          pSched->vBufts[pSched->vLeafBackendIds[vI]] != pSched->vBufts[pSched->vPrevLeafBackendIds[vI]]) {
        vBackendIdsChanged = true;
        break;
      }
    }
  }

  // allocate graph
  if (vBackendIdsChanged || !ggml_gallocr_alloc_graph(pSched->vGalloc, &pSched->vGraph)) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__,
                   vBackendIdsChanged);
#endif

    if (pSched->vDebugRealloc > 0) {
      // we are interested only in situations where the graph was reallocated even though its size remained the same
      // [GGML_SCHED_DEBUG_REALLOC]
      const bool cUnexpected = !vBackendIdsChanged && pSched->vDebugPrevGraphSize == pSched->vDebugGraphSize;

      if (cUnexpected || pSched->vDebugRealloc > 1) {
        GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n",
                   __func__, pSched->vDebugGraphSize, pSched->vGraph.n_nodes, pSched->vGraph.n_leafs,
                   pSched->vDebugRealloc);
      }
    }

    // the re-allocation may cause the split inputs to be moved to a different address
    // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
    for (int vI = 0; vI < pSched->n_backends; vI++) {
      ggml_backend_synchronize(pSched->aBackends[vI]);
    }

    ggml_gallocr_reserve_n(pSched->vGalloc, &pSched->vGraph, pSched->vNodeBackendIds, pSched->vLeafBackendIds);
    if (!ggml_gallocr_alloc_graph(pSched->vGalloc, &pSched->vGraph)) {
      GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
      return false;
    }
  }

  return true;
}

static void fDisableExpertPrefetch(ggml_backend_sched_t pScheduler, ggml_backend_t pSplitBackend) {
  pScheduler->vPrefetchExperts = false;
  if (pScheduler->vPrefetchBackend) {
    ggml_backend_synchronize(pSplitBackend);
    ggml_backend_synchronize(pScheduler->vPrefetchBackend);
  }
  for (int vIndex = 0; vIndex < pScheduler->vPrefetchSlotCount; ++vIndex) {
    ggml_backend_buffer_free(pScheduler->aAPrefetchSlots[vIndex]);
    pScheduler->aAPrefetchSlots[vIndex] = NULL;
    pScheduler->aAPrefetchUsed[vIndex] = false;
  }
}

// Find the largest offloaded expert tensor to size all slots before use
// and avoid references to released memory during evaluation. The size is the one the slot buffer type allocates for
// the tensor, with the padding that its kernels read past the last row (CUDA pads quantized rows to 512 elements).
static size_t fLargestExpertTensorSize(ggml_backend_sched_t pScheduler, ggml_backend_dev_t pDevice,
                                       ggml_backend_buffer_type_t pBufferType) {
  size_t vMaximumSize = 0;
  for (int vSplitId = 0; vSplitId < pScheduler->vNSplits; ++vSplitId) {
    struct ggml_backend_sched_split *vSplit = &pScheduler->vSplits[vSplitId];
    if (pScheduler->aBackends[vSplit->backend_id]->device != pDevice || vSplit->vGraph.n_nodes == 0 ||
        vSplit->vGraph.nodes[0]->op != GGML_OP_MUL_MAT_ID) {
      continue;
    }
    for (int vInputId = 0; vInputId < vSplit->vNInputs; ++vInputId) {
      const ggml_tensor *vInput = vSplit->vInputs[vInputId];
      if (vInput->buffer && ggml_backend_buffer_get_usage(vInput->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
          ggml_backend_buffer_is_host(vInput->buffer)) {
        vMaximumSize = std::max(vMaximumSize, ggml_backend_buft_get_alloc_size(pBufferType, vInput));
      }
    }
  }
  return vMaximumSize;
}

// Writes zeros over the padding that follows a tensor transferred into a prefetch slot (pBytes to pPaddedBytes), on
// the prefetch stream after the transfer. A weight buffer zeroes that padding once; a slot holds a different tensor
// every time, and a smaller one leaves part of the previous one there, whose bytes can read as NaN scales.
static void fClearPrefetchPadding(ggml_backend_sched_t pScheduler, const ggml_tensor *pSlotTensor, size_t pBytes,
                                  size_t pPaddedBytes) {
  thread_local std::vector<uint8_t> lZeros;
  const size_t cSize = pPaddedBytes - pBytes;
  if (lZeros.size() < cSize) {
    lZeros.resize(cSize, 0);
  }
  ggml_tensor vPadding = *pSlotTensor;
  vPadding.type = GGML_TYPE_I8;
  vPadding.view_src = NULL;
  vPadding.view_offs = 0;
  vPadding.data = (char *)pSlotTensor->data + pBytes;
  vPadding.ne[0] = (int64_t)cSize;
  vPadding.nb[0] = 1;
  for (int vDim = 1; vDim < GGML_MAX_DIMS; ++vDim) {
    vPadding.ne[vDim] = 1;
    vPadding.nb[vDim] = cSize;
  }
  ggml_backend_tensor_set_async(pScheduler->vPrefetchBackend, &vPadding, lZeros.data(), 0, cSize);
}

static bool fInitializeExpertPrefetch(ggml_backend_sched_t pScheduler, ggml_backend_t pSplitBackend,
                                      const ggml_tensor *pTensor) {
  ggml_backend_dev_t vDevice = pSplitBackend->device;
  if (pScheduler->vPrefetchDevice != NULL && pScheduler->vPrefetchDevice != vDevice) {
    // Events and slots belong to the first device that enabled prefetch.
    // Other devices keep the regular selective copy path.
    return false;
  }

  if (pScheduler->vPrefetchBackend == NULL) {
    // An integrated GPU shares the system memory: filling a slot is a copy from RAM to RAM by the CPU, with no bus
    // transfer for the separate stream to overlap, and the slots pin memory that the page cache of the experts left
    // in system memory needs. The selective copy below is faster there (Qwen3.8-Flash-Next on a Radeon 860M with
    // 42 expert layers in system memory: prompt 15 -> 33 tokens/s, generation after it 4.3 -> 5.0 tokens/s).
    if (ggml_backend_dev_type(vDevice) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
      pScheduler->vPrefetchExperts = false;
      return false;
    }

    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDevice, &vProps);
    if (!vProps.caps.async || !vProps.caps.events) {
      pScheduler->vPrefetchExperts = false;
      return false;
    }

    pScheduler->vPrefetchBackend = ggml_backend_dev_init(vDevice, NULL);
    if (pScheduler->vPrefetchBackend == NULL) {
      pScheduler->vPrefetchExperts = false;
      return false;
    }
    pScheduler->vPrefetchDevice = vDevice;

    for (int vIndex = 0; vIndex < pScheduler->vPrefetchSlotCount; ++vIndex) {
      pScheduler->aAPrefetchReady[vIndex] = ggml_backend_event_new(vDevice);
      pScheduler->aAPrefetchFree[vIndex] = ggml_backend_event_new(vDevice);
      if (pScheduler->aAPrefetchReady[vIndex] == NULL || pScheduler->aAPrefetchFree[vIndex] == NULL) {
        pScheduler->vPrefetchExperts = false;
        return false;
      }
    }
  }

  ggml_backend_buffer_type_t vBufferType = ggml_backend_get_default_buffer_type(pSplitBackend);
  const size_t cSlotSize = std::max(ggml_backend_buft_get_alloc_size(vBufferType, pTensor),
                                    fLargestExpertTensorSize(pScheduler, vDevice, vBufferType));

  for (int vIndex = 0; vIndex < pScheduler->vPrefetchSlotCount; ++vIndex) {
    if (pScheduler->aAPrefetchSlots[vIndex] == NULL ||
        ggml_backend_buffer_get_size(pScheduler->aAPrefetchSlots[vIndex]) < cSlotSize) {
      // The slots are allocated during the first large micro-batch, while the computation still allocates memory on
      // demand (temporary buffers of the backend, cuBLAS workspaces), so a slot that leaves less than the reserve free
      // counts as one the device refused; taking every slot the device accepted left Qwen3.8-Flash-Next with MTP on an
      // RTX 4060 Ti about 340 MiB.
      ggml_backend_buffer_t vNewBuffer = NULL;
      size_t vFree = 0;
      size_t vTotal = 0;
      if (pScheduler->vPrefetchReserve > 0) {
        ggml_backend_dev_memory(vDevice, &vFree, &vTotal);
      }
      if (vTotal > 0 && vFree < cSlotSize + pScheduler->vPrefetchReserve) {
        GGML_LOG_INFO("%s: expert prefetch slot %d of %.2f MiB not allocated: %s has %.2f MiB free and keeps a "
                      "reserve of %.2f MiB\n",
                      __func__, vIndex + 1, cSlotSize / 1048576.0, ggml_backend_dev_name(vDevice), vFree / 1048576.0,
                      pScheduler->vPrefetchReserve / 1048576.0);
      } else {
        // Allocate first so a failure preserves the previous slot.
        vNewBuffer = ggml_backend_buft_alloc_buffer(vBufferType, cSlotSize);
      }
      if (vNewBuffer == NULL) {
        // Overlap requires at least two slots; continue with those that fit.
        if (vIndex >= 2 && pScheduler->aAPrefetchSlots[0] != NULL &&
            ggml_backend_buffer_get_size(pScheduler->aAPrefetchSlots[0]) >= cSlotSize) {
          pScheduler->vPrefetchSlotCount = vIndex;
          pScheduler->vPrefetchCurrent = 0;
          return true;
        }
        fDisableExpertPrefetch(pScheduler, pSplitBackend);
        return false;
      }
      if (pScheduler->aAPrefetchSlots[vIndex] != NULL) {
        ggml_backend_synchronize(pSplitBackend);
        ggml_backend_synchronize(pScheduler->vPrefetchBackend);
        ggml_backend_buffer_free(pScheduler->aAPrefetchSlots[vIndex]);
      }
      pScheduler->aAPrefetchSlots[vIndex] = vNewBuffer;
      pScheduler->aAPrefetchUsed[vIndex] = false;
    }
  }
  return true;
}

static enum ggml_status fGgmlBackendSchedComputeSplits(ggml_backend_sched_t sched) {
  GGML_ASSERT(sched);
  struct ggml_backend_sched_split *vSplits = sched->vSplits;

  ggml_tensor *vPrevIdsTensor = nullptr;
  std::vector<int32_t> lIds;
  std::vector<ggml_bitset_t> lUsedIds;

  int vPreviousBackendId = -1;
  bool vPreviousSplitHadInputs = false;

  for (int vSplitId = 0; vSplitId < sched->vNSplits; vSplitId++) {
    // The boundary between splits is where a graph computed by backends without an abort callback of their own (the
    // GPUs) can stop, so that a cancelled decode ends after the split in progress instead of after the whole graph.
    if (sched->vAbortCallback != NULL && sched->vAbortCallback(sched->vAbortCallbackData)) {
      return GGML_STATUS_ABORTED;
    }

    struct ggml_backend_sched_split *vSplit = &vSplits[vSplitId];
    int vSplitBackendId = vSplit->backend_id;
    ggml_backend_t vSplitBackend = sched->aBackends[vSplitBackendId];
    int vSplitPrefetchSlot = -1;
    ggml_tensor *vPrefetchInputCopy = NULL;
    ggml_backend_buffer_t vPrefetchSavedBuffer = NULL;
    void *vPrefetchSavedData = NULL;

    // The allocator can reuse memory across splits even when there are no inputs to copy:
    // an input-free split waits for the previous split while that split may still read
    // memory of this backend, either through copied inputs or through direct access to
    // its buffer type. A previous split without inputs on a backend that cannot access
    // this buffer type reads none of it, so both can run at the same time (hybrid MoE
    // execution relies on this to overlap the CPU and device expert branches).
    if (vSplit->vNInputs == 0 && vPreviousBackendId >= 0 && vPreviousBackendId != vSplitBackendId &&
        (vPreviousSplitHadInputs ||
         ggml_backend_supports_buft(sched->aBackends[vPreviousBackendId], sched->vBufts[vSplitBackendId]))) {
      if (sched->aEvents[vPreviousBackendId][sched->vCurCopy] != NULL) {
        ggml_backend_event_synchronize(sched->aEvents[vPreviousBackendId][sched->vCurCopy]);
      } else {
        ggml_backend_synchronize(sched->aBackends[vPreviousBackendId]);
      }
    }

    // copy the input tensors to the split backend
    for (int vInputId = 0; vInputId < vSplit->vNInputs; vInputId++) {
      ggml_backend_t vInputBackend = ggml_backend_sched_get_tensor_backend(sched, vSplit->vInputs[vInputId]);
      struct ggml_tensor *vInput = vSplit->vInputs[vInputId];
      struct ggml_tensor *vInputCpy = tensor_copy(vInput, vSplitBackendId, sched->vCurCopy);

      if (vInput->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is
        // done
        if (sched->aEvents[vSplitBackendId][sched->vCurCopy] != NULL) {
          ggml_backend_event_synchronize(sched->aEvents[vSplitBackendId][sched->vCurCopy]);
        } else {
          ggml_backend_synchronize(vSplitBackend);
        }
        ggml_backend_tensor_copy(vInput, vInputCpy);
      } else {
        // wait for the split backend to finish using the input before overwriting it
        if (sched->aEvents[vSplitBackendId][sched->vCurCopy] != NULL) {
          ggml_backend_event_wait(vSplitBackend, sched->aEvents[vSplitBackendId][sched->vCurCopy]);
        } else {
          ggml_backend_synchronize(vSplitBackend);
        }

        ggml_tensor *vNode = vSplit->vGraph.n_nodes > 0 ? vSplit->vGraph.nodes[0] : NULL;

        // Large batches use almost every expert. Transfer the entire tensor
        // on a separate stream without waiting for the expert identifiers.
        if (sched->vPrefetchExperts && !sched->vCallbackEval && vSplitPrefetchSlot == -1 && vNode != NULL &&
            ggml_backend_buffer_get_usage(vInput->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(vInput->buffer) && vNode->op == GGML_OP_MUL_MAT_ID &&
            vNode->src[0] == vInputCpy) {
          const ggml_tensor *vIds = vNode->src[2];
          const int64_t cNExpert = vInput->ne[2];
          if (vIds->ne[0] * vIds->ne[1] >= 2 * cNExpert && fInitializeExpertPrefetch(sched, vSplitBackend, vInput)) {
            const int cSlot = sched->vPrefetchCurrent;
            sched->vPrefetchCurrent = (sched->vPrefetchCurrent + 1) % sched->vPrefetchSlotCount;

            // Wait until the preceding computation has finished using this slot.
            if (sched->aAPrefetchUsed[cSlot]) {
              ggml_backend_event_wait(sched->vPrefetchBackend, sched->aAPrefetchFree[cSlot]);
            }

            // Redirect the copy for this split only. It is restored after the kernel is launched, so that a later
            // fallback never sees a stale pointer.
            vPrefetchInputCopy = vInputCpy;
            vPrefetchSavedBuffer = vInputCpy->buffer;
            vPrefetchSavedData = vInputCpy->data;
            vInputCpy->buffer = sched->aAPrefetchSlots[cSlot];
            vInputCpy->data = ggml_backend_buffer_get_base(sched->aAPrefetchSlots[cSlot]);

            if (sched->vWeightUpload == NULL ||
                !sched->vWeightUpload(sched->vWeightUploadData, sched->vPrefetchBackend, vInput, vInputCpy)) {
              ggml_backend_tensor_set_async(sched->vPrefetchBackend, vInputCpy, vInput->data, 0, ggml_nbytes(vInput));
            }
            // Without the padding in the slot, the product read past its end: Qwen3.8-Flash-Next (Q8_0 experts with
            // 640-element rows, which CUDA pads to 1024) stopped with an illegal memory access on an RTX 4060 Ti.
            const size_t cPaddedBytes = ggml_backend_buft_get_alloc_size(
                ggml_backend_buffer_get_type(sched->aAPrefetchSlots[cSlot]), vInput);
            if (cPaddedBytes > ggml_nbytes(vInput)) {
              fClearPrefetchPadding(sched, vInputCpy, ggml_nbytes(vInput), cPaddedBytes);
            }
            ggml_backend_event_record(sched->aAPrefetchReady[cSlot], sched->vPrefetchBackend);
            ggml_backend_event_wait(vSplitBackend, sched->aAPrefetchReady[cSlot]);
            vSplitPrefetchSlot = cSlot;
            continue;
          }
        }

        // When MoE weights are uploaded, only the used experts are copied to reduce the transfer.
        if (vNode != NULL && ggml_backend_buffer_get_usage(vInput->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(vInput->buffer) &&
            ((vNode->src[0] == vInputCpy && vNode->op == GGML_OP_MUL_MAT_ID)
             //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not
             // worth splitting */
             )) {

          const int64_t cNExpert = vNode->op == GGML_OP_MUL_MAT_ID ? vInput->ne[2] : vInput->ne[1];
          const size_t cExpertSize = vNode->op == GGML_OP_MUL_MAT_ID ? vInput->nb[2] : vInput->nb[1];

          ggml_backend_synchronize(vInputBackend);

          // get the ids
          ggml_tensor *vIdsTensor = vNode->src[2];
          ggml_backend_t vIdsBackend = vSplitBackend;

          if (ggml_nelements(vIdsTensor) == 0) {
            continue;
          }

          // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
          // in that case, we use the original ids tensor
          for (int vI = vInputId + 1; vI < vSplit->vNInputs; vI++) {
            if (vIdsTensor == tensor_copy(vSplit->vInputs[vI], vSplitBackendId, sched->vCurCopy)) {
              vIdsTensor = vSplit->vInputs[vI];
              vIdsBackend = ggml_backend_sched_get_tensor_backend(sched, vSplit->vInputs[vI]);
              break;
            }
          }

          if (vIdsTensor != vPrevIdsTensor) {
            lIds.resize(ggml_nbytes(vIdsTensor) / sizeof(int32_t));
            ggml_backend_tensor_get_async(vIdsBackend, vIdsTensor, lIds.data(), 0, ggml_nbytes(vIdsTensor));
            ggml_backend_synchronize(vIdsBackend);

            // find the used experts
            lUsedIds.clear();
            lUsedIds.resize(ggml_bitset_size(cNExpert));
            for (int64_t vI1 = 0; vI1 < vIdsTensor->ne[1]; vI1++) {
              for (int64_t vI0 = 0; vI0 < vIdsTensor->ne[0]; vI0++) {
                int32_t vId =
                    lIds[vI1 * vIdsTensor->nb[1] / sizeof(int32_t) + vI0 * vIdsTensor->nb[0] / sizeof(int32_t)];
                GGML_ASSERT(vId >= 0 && vId < cNExpert);
                ggml_bitset_set(lUsedIds.data(), vId);
              }
            }

            vPrevIdsTensor = vIdsTensor;
          }

          // group consecutive experts and copy them together
          auto fCopyExperts = [&](int32_t pFirstId, int32_t pLastId) {
            const size_t cExpertOffset = pFirstId * cExpertSize;
            const size_t cExpertSizeCopy = (pLastId - pFirstId + 1) * cExpertSize;
            const size_t cPadding = std::min<size_t>(cExpertSize, 512);
            const size_t cPaddingEnd = pLastId < cNExpert - 1 ? cPadding : 0;

            ggml_backend_tensor_set_async(vSplitBackend, vInputCpy, (const uint8_t *)vInput->data + cExpertOffset,
                                          cExpertOffset,
                                          // copy a bit extra at the to ensure there are no NaNs in the padding of the
                                          // last expert this is necessary for MMQ in the CUDA backend
                                          cExpertSizeCopy + cPaddingEnd);
          };

          int vId = 0;
          while (!ggml_bitset_get(lUsedIds.data(), vId)) {
            vId++;
          }
          int32_t vFirstId = vId;
          int32_t vLastId = vFirstId;

          for (++vId; vId < cNExpert; ++vId) {
            if (!ggml_bitset_get(lUsedIds.data(), vId)) {
              continue;
            }

            if (vId == vLastId + 1) {
              vLastId = vId;
              continue;
            }

            fCopyExperts(vFirstId, vLastId);

            vFirstId = vId;
            vLastId = vId;
          }
          fCopyExperts(vFirstId, vLastId);
        } else {
          // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend,
          // since we handle the synchronization here with multiple copies and events
          // TODO: add public function to facilitate this, since applications do not have direct access to the backend
          // interface
          if (!vSplitBackend->iface.cpy_tensor_async ||
              !vSplitBackend->iface.cpy_tensor_async(vInputBackend, vSplitBackend, vInput, vInputCpy)) {
            ggml_backend_synchronize(vInputBackend);
            if (sched->aEvents[vSplitBackendId][sched->vCurCopy] != NULL) {
              ggml_backend_event_synchronize(sched->aEvents[vSplitBackendId][sched->vCurCopy]);
            } else {
              ggml_backend_synchronize(vSplitBackend);
            }
            ggml_backend_tensor_copy(vInput, vInputCpy);
          }
        }
      }
    }

    if (!sched->vCallbackEval) {
      enum ggml_status vEc = ggml_backend_graph_compute_async(vSplitBackend, &vSplit->vGraph);
      if (vSplitPrefetchSlot != -1) {
        // The kernel captured the slot address when it was launched.
        ggml_backend_event_record(sched->aAPrefetchFree[vSplitPrefetchSlot], vSplitBackend);
        sched->aAPrefetchUsed[vSplitPrefetchSlot] = true;
        vPrefetchInputCopy->buffer = vPrefetchSavedBuffer;
        vPrefetchInputCopy->data = vPrefetchSavedData;
      }
      if (vEc != GGML_STATUS_SUCCESS) {
        return vEc;
      }
    } else {
      // similar to ggml_backend_compare_graph_backend
      for (int vJ0 = 0; vJ0 < vSplit->vGraph.n_nodes; vJ0++) {
        struct ggml_tensor *vT = vSplit->vGraph.nodes[vJ0];

        // check if the user needs data from this node
        bool vNeed = sched->vCallbackEval(vT, true, sched->vCallbackEvalUserData);

        int vJ1 = vJ0;

        // determine the range [j0, j1] of nodes that can be computed together
        while (!vNeed && vJ1 < vSplit->vGraph.n_nodes - 1) {
          vT = vSplit->vGraph.nodes[++vJ1];
          vNeed = sched->vCallbackEval(vT, true, sched->vCallbackEvalUserData);
        }

        struct ggml_cgraph vGv = ggml_graph_view(&vSplit->vGraph, vJ0, vJ1 + 1);

        enum ggml_status vEc = ggml_backend_graph_compute_async(vSplitBackend, &vGv);
        if (vEc != GGML_STATUS_SUCCESS) {
          return vEc;
        }

        // TODO: pass backend to the callback, then the user can decide if they want to synchronize
        ggml_backend_synchronize(vSplitBackend);

        if (vNeed && !sched->vCallbackEval(vT, false, sched->vCallbackEvalUserData)) {
          break;
        }

        vJ0 = vJ1;
      }
    }

    // Record every split, including those without copied inputs.
    if (sched->aEvents[vSplitBackendId][sched->vCurCopy] != NULL) {
      ggml_backend_event_record(sched->aEvents[vSplitBackendId][sched->vCurCopy], vSplitBackend);
    }
    vPreviousBackendId = vSplitBackendId;
    vPreviousSplitHadInputs = vSplit->vNInputs > 0;
  }

  return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t *pBackends, ggml_backend_buffer_type_t *pBufts,
                                            int pNBackends, size_t pGraphSize, bool pParallel, bool pOpOffload) {
  GGML_ASSERT(pNBackends > 0);
  GGML_ASSERT(pNBackends <= GGML_SCHED_MAX_BACKENDS);
  GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(pBackends[pNBackends - 1])) ==
              GGML_BACKEND_DEVICE_TYPE_CPU);

  struct ggml_backend_sched *vSched = (ggml_backend_sched *)calloc(1, sizeof(struct ggml_backend_sched));

  const char *vGgmlSchedDebug = getenv("GGML_SCHED_DEBUG");
  vSched->vDebug = vGgmlSchedDebug ? atoi(vGgmlSchedDebug) : 0;

  vSched->vDebugRealloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
  vSched->vDebugRealloc = 1;
#endif
  const char *vGgmlSchedDebugRealloc = getenv("GGML_SCHED_DEBUG_REALLOC");
  vSched->vDebugRealloc = vGgmlSchedDebugRealloc ? atoi(vGgmlSchedDebugRealloc) : vSched->vDebugRealloc;

  vSched->n_backends = pNBackends;
  vSched->n_copies = pParallel ? GGML_SCHED_MAX_COPIES : 1;

  // initialize hash table
  // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
  vSched->hash_set = ggml_hash_set_new(pGraphSize);
  vSched->hv_tensor_backend_ids = (int *)malloc(vSched->hash_set.size * sizeof(vSched->hv_tensor_backend_ids[0]));
  vSched->hv_tensor_copies = (ggml_tensor **)malloc(vSched->hash_set.size * vSched->n_backends * vSched->n_copies *
                                                    sizeof(struct ggml_tensor *));

  const size_t cGgmlSchedMaxSplits = pGraphSize; // at most there is one split for each node in the graph
  const size_t cNodesSize = pGraphSize + cGgmlSchedMaxSplits * GGML_SCHED_MAX_SPLIT_INPUTS * 2;
  vSched->vNodeBackendIds = (int *)calloc(cNodesSize, sizeof(vSched->vNodeBackendIds[0]));
  vSched->vLeafBackendIds = (int *)calloc(cNodesSize, sizeof(vSched->vLeafBackendIds[0]));
  vSched->vPrevNodeBackendIds = (int *)calloc(cNodesSize, sizeof(vSched->vPrevNodeBackendIds[0]));
  vSched->vPrevLeafBackendIds = (int *)calloc(cNodesSize, sizeof(vSched->vPrevLeafBackendIds[0]));

  vSched->vDebugGraphSize = 0;
  vSched->vDebugPrevGraphSize = 0;

  vSched->vContextBufferSize = cGgmlSchedMaxSplits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct ggml_tensor) +
                               ggml_graph_overhead_custom(pGraphSize, false);
  vSched->vContextBuffer = (char *)malloc(vSched->vContextBufferSize);

  const int cInitialSplitsCapacity = 16;
  vSched->vSplits = (ggml_backend_sched_split *)calloc(cInitialSplitsCapacity, sizeof(vSched->vSplits[0]));
  vSched->vSplitsCapacity = cInitialSplitsCapacity;

  vSched->vGraphInputsCapacity = GGML_SCHED_MAX_SPLIT_INPUTS;
  vSched->vGraphInputs = (struct ggml_tensor **)calloc(vSched->vGraphInputsCapacity, sizeof(struct ggml_tensor *));

  for (int vB = 0; vB < pNBackends; vB++) {
    vSched->aBackends[vB] = pBackends[vB];
    vSched->vBufts[vB] = pBufts ? pBufts[vB] : ggml_backend_get_default_buffer_type(pBackends[vB]);
    GGML_ASSERT(ggml_backend_supports_buft(pBackends[vB], vSched->vBufts[vB]));

    if (vSched->n_copies > 1) {
      for (int vC = 0; vC < vSched->n_copies; vC++) {
        vSched->aEvents[vB][vC] = ggml_backend_event_new(pBackends[vB]->device);
      }
    }
  }

  vSched->vGalloc = ggml_gallocr_new_n(vSched->vBufts, pNBackends);
  vSched->vOpOffload = pOpOffload;

  // MoE prefetch defaults to three slots
  // (gate/up/down for one layer). Set 0 to disable it for diagnostics;
  // values from 2 to the maximum explicitly select the slot count.
  const char *vGgmlSchedPrefetchExperts = getenv("GGML_SCHED_PREFETCH_EXPERTS");
  const int cVPrefetchSlotCount = vGgmlSchedPrefetchExperts ? atoi(vGgmlSchedPrefetchExperts) : 1;
  vSched->vPrefetchExperts = pOpOffload && cVPrefetchSlotCount > 0;
  vSched->vPrefetchSlotCount =
      cVPrefetchSlotCount <= 1 ? 3 : std::min(cVPrefetchSlotCount, GGML_SCHED_MAX_PREFETCH_SLOTS);

  ggml_backend_sched_reset(vSched);

  return vSched;
}

void ggml_backend_sched_free(ggml_backend_sched_t pSched) {
  if (pSched == NULL) {
    return;
  }
  for (int vB = 0; vB < pSched->n_backends; vB++) {
    for (int vC = 0; vC < pSched->n_copies; vC++) {
      ggml_backend_event_free(pSched->aEvents[vB][vC]);
    }
  }
  if (pSched->vPrefetchBackend) {
    ggml_backend_synchronize(pSched->vPrefetchBackend);
    // The active count may have decreased after an allocation failure.
    for (int vI = 0; vI < GGML_SCHED_MAX_PREFETCH_SLOTS; ++vI) {
      ggml_backend_event_free(pSched->aAPrefetchReady[vI]);
      ggml_backend_event_free(pSched->aAPrefetchFree[vI]);
      ggml_backend_buffer_free(pSched->aAPrefetchSlots[vI]);
    }
    ggml_backend_free(pSched->vPrefetchBackend);
  }
  ggml_gallocr_free(pSched->vGalloc);
  ggml_free(pSched->vCtx);
  ggml_hash_set_free(&pSched->hash_set);
  for (int vI = 0; vI < pSched->vSplitsCapacity; vI++) {
    free(pSched->vSplits[vI].vInputs);
  }
  free(pSched->vSplits);
  free(pSched->vGraphInputs);
  free(pSched->hv_tensor_backend_ids);
  free(pSched->hv_tensor_copies);
  free(pSched->vNodeBackendIds);
  free(pSched->vLeafBackendIds);
  free(pSched->vPrevNodeBackendIds);
  free(pSched->vPrevLeafBackendIds);
  free(pSched->vContextBuffer);
  free(pSched->vGraph.nodes);
  free(pSched->vGraph.leafs);
  free(pSched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t pSched) {
  GGML_ASSERT(pSched);
  // reset state for the next run
  if (!pSched->vIsReset) {
    ggml_hash_set_reset(&pSched->hash_set);
    memset(pSched->hv_tensor_backend_ids, -1, pSched->hash_set.size * sizeof(pSched->hv_tensor_backend_ids[0]));
    memset(pSched->hv_tensor_copies, 0,
           pSched->hash_set.size * pSched->n_backends * pSched->n_copies * sizeof(struct ggml_tensor *));
    pSched->vIsReset = true;
  }
  pSched->vIsAlloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t pSched, struct ggml_cgraph *pMeasureGraph, size_t *pSizes) {
  GGML_ASSERT(pSched);
  GGML_ASSERT((int)pSched->hash_set.size >= pMeasureGraph->n_nodes + pMeasureGraph->n_leafs);
  GGML_ASSERT(pSizes);

  ggml_backend_sched_reset(pSched);

  ggml_backend_sched_synchronize(pSched);

  ggml_backend_sched_split_graph(pSched, pMeasureGraph);

  ggml_gallocr_reserve_n_size(pSched->vGalloc, &pSched->vGraph, pSched->vNodeBackendIds, pSched->vLeafBackendIds,
                              pSizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t pSched, struct ggml_cgraph *pMeasureGraph) {
  GGML_ASSERT(pSched);
  GGML_ASSERT((int)pSched->hash_set.size >= pMeasureGraph->n_nodes + pMeasureGraph->n_leafs);

  ggml_backend_sched_synchronize(pSched);

  ggml_backend_sched_split_graph(pSched, pMeasureGraph);

  if (!ggml_gallocr_reserve_n(pSched->vGalloc, &pSched->vGraph, pSched->vNodeBackendIds, pSched->vLeafBackendIds)) {
    return false;
  }

  ggml_backend_sched_reset(pSched);

  return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t pSched, struct ggml_cgraph *pGraph) {
  GGML_ASSERT(pSched);
  GGML_ASSERT((int)pSched->hash_set.size >= pGraph->n_nodes + pGraph->n_leafs);
  GGML_ASSERT(!pSched->vIsAlloc);

  pSched->vCurCopy = pSched->vNextCopy;
  pSched->vNextCopy = (pSched->vNextCopy + 1) % pSched->n_copies;

  ggml_backend_sched_split_graph(pSched, pGraph);

  if (!fGgmlBackendSchedAllocSplits(pSched)) {
    return false;
  }

  pSched->vIsAlloc = true;

  return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t pSched, struct ggml_cgraph *pGraph) {
  enum ggml_status vErr = ggml_backend_sched_graph_compute_async(pSched, pGraph);
  ggml_backend_sched_synchronize(pSched);
  return vErr;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t pSched, struct ggml_cgraph *pGraph) {
  GGML_ASSERT(pSched);
  if (!pSched->vIsReset && !pSched->vIsAlloc) {
    ggml_backend_sched_reset(pSched);
  }

  if (!pSched->vIsAlloc) {
    if (!ggml_backend_sched_alloc_graph(pSched, pGraph)) {
      return GGML_STATUS_ALLOC_FAILED;
    }
  }

  return fGgmlBackendSchedComputeSplits(pSched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t pSched) {
  GGML_ASSERT(pSched);
  for (int vI = 0; vI < pSched->n_backends; vI++) {
    ggml_backend_synchronize(pSched->aBackends[vI]);
  }
  if (pSched->vPrefetchBackend) {
    ggml_backend_synchronize(pSched->vPrefetchBackend);
  }
  if (!pSched->vIsAlloc) {
    // if the graph is not already allocated, always use copy 0 after a synchronization
    // this ensures that during generation the same copy is used every time,
    // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
    pSched->vNextCopy = 0;
  }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t pSched, ggml_backend_sched_eval_callback pCallback,
                                          void *pUserData) {
  GGML_ASSERT(pSched);
  pSched->vCallbackEval = pCallback;
  pSched->vCallbackEvalUserData = pUserData;
}

void ggml_backend_sched_set_abort_callback(ggml_backend_sched_t pSched, ggml_abort_callback pCallback, void *pData) {
  GGML_ASSERT(pSched);
  pSched->vAbortCallback = pCallback;
  pSched->vAbortCallbackData = pData;
}

void ggml_backend_sched_set_weight_upload(ggml_backend_sched_t pSched, ggml_backend_sched_weight_upload_t pCallback,
                                          void *pUserData) {
  GGML_ASSERT(pSched);
  pSched->vWeightUpload = pCallback;
  pSched->vWeightUploadData = pUserData;
}

void ggml_backend_sched_set_prefetch_reserve(ggml_backend_sched_t pSched, size_t pReserve) {
  GGML_ASSERT(pSched);
  pSched->vPrefetchReserve = pReserve;
}

size_t ggml_backend_sched_get_prefetch_size(ggml_backend_sched_t pSched, ggml_backend_buffer_type_t *pBufferType) {
  GGML_ASSERT(pSched);
  size_t vBytes = 0;
  ggml_backend_buffer_type_t vBufferType = NULL;
  // A failed reallocation can shorten vPrefetchSlotCount while later slots stay allocated.
  for (int vIndex = 0; vIndex < GGML_SCHED_MAX_PREFETCH_SLOTS; ++vIndex) {
    if (pSched->aAPrefetchSlots[vIndex] != NULL) {
      vBytes += ggml_backend_buffer_get_size(pSched->aAPrefetchSlots[vIndex]);
      vBufferType = ggml_backend_buffer_get_type(pSched->aAPrefetchSlots[vIndex]);
    }
  }
  if (pBufferType != NULL) {
    *pBufferType = vBufferType;
  }
  return vBytes;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t pSched) {
  GGML_ASSERT(pSched);
  return pSched->vNSplits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t pSched) {
  GGML_ASSERT(pSched);
  return pSched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t pSched) {
  GGML_ASSERT(pSched);
  return pSched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t pSched, int pI) {
  GGML_ASSERT(pSched);
  GGML_ASSERT(pI >= 0 && pI < pSched->n_backends);
  return pSched->aBackends[pI];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t pSched, ggml_backend_t pBackend) {
  GGML_ASSERT(pSched);
  int vBackendIndex = fGgmlBackendSchedBackendId(pSched, pBackend);
  GGML_ASSERT(vBackendIndex >= 0 && vBackendIndex < pSched->n_backends);

  return pSched->vBufts[vBackendIndex];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t pSched, ggml_backend_t pBackend) {
  GGML_ASSERT(pSched);
  int vBackendIndex = fGgmlBackendSchedBackendId(pSched, pBackend);
  GGML_ASSERT(vBackendIndex >= 0 && vBackendIndex < pSched->n_backends);

  return ggml_gallocr_get_buffer_size(pSched->vGalloc, vBackendIndex);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor *pNode,
                                           ggml_backend_t pBackend) {
  GGML_ASSERT(sched);
  int vBackendIndex = fGgmlBackendSchedBackendId(sched, pBackend);
  GGML_ASSERT(vBackendIndex >= 0 && vBackendIndex < sched->n_backends);
  tensor_backend_id(pNode) = vBackendIndex;
  SET_CAUSE(pNode, "usr");
  sched->vIsReset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor *pNode) {
  GGML_ASSERT(sched);
  int vBackendIndex = tensor_backend_id(pNode);
  if (vBackendIndex == -1) {
    return NULL;
  }
  return sched->aBackends[vBackendIndex];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor *pTensor) {
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->buffer == NULL);
  GGML_ASSERT(pTensor->view_src != NULL);
  GGML_ASSERT(pTensor->view_src->buffer != NULL);
  GGML_ASSERT(pTensor->view_src->data != NULL);

  pTensor->buffer = pTensor->view_src->buffer;
  pTensor->data = (char *)pTensor->view_src->data + pTensor->view_offs;
  return ggml_backend_buffer_init_tensor(pTensor->buffer, pTensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor, void *pAddr) {
  GGML_ASSERT(pTensor);
  GGML_ASSERT(pTensor->buffer == NULL);
  GGML_ASSERT(pTensor->data == NULL);
  GGML_ASSERT(pTensor->view_src == NULL);
  GGML_ASSERT(pAddr >= ggml_backend_buffer_get_base(pBuffer));
  GGML_ASSERT(ggml_backend_buffer_is_meta(pBuffer) ||
              (char *)pAddr + ggml_backend_buffer_get_alloc_size(pBuffer, pTensor) <=
                  (char *)ggml_backend_buffer_get_base(pBuffer) + ggml_backend_buffer_get_size(pBuffer));

  pTensor->buffer = pBuffer;
  pTensor->data = pAddr;
  return ggml_backend_buffer_init_tensor(pBuffer, pTensor);
}

static struct ggml_tensor *fGraphCopyDupTensor(struct ggml_hash_set pHashSet, struct ggml_tensor **pNodeCopies,
                                               struct ggml_context *pCtxAllocated, struct ggml_context *pCtxUnallocated,
                                               struct ggml_tensor *pSrc) {

  GGML_ASSERT(pSrc != NULL);
  GGML_ASSERT(pSrc->data && "graph must be allocated");

  size_t vId = ggml_hash_insert(&pHashSet, pSrc);
  if (vId == GGML_HASHSET_ALREADY_EXISTS) {
    return pNodeCopies[ggml_hash_find(&pHashSet, pSrc)];
  }

  struct ggml_tensor *vDst =
      fGgmlDupTensorLayout(pSrc->data && !pSrc->view_src ? pCtxAllocated : pCtxUnallocated, pSrc);
  if (pSrc->view_src != NULL) {
    vDst->view_src = fGraphCopyDupTensor(pHashSet, pNodeCopies, pCtxAllocated, pCtxUnallocated, pSrc->view_src);
    vDst->view_offs = pSrc->view_offs;
  }
  vDst->op = pSrc->op;
  vDst->flags = pSrc->flags;
  memcpy(vDst->op_params, pSrc->op_params, sizeof(vDst->op_params));
  ggml_set_name(vDst, pSrc->name);

  // copy src
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    struct ggml_tensor *vS = pSrc->src[vI];
    if (vS == NULL) {
      continue;
    }
    vDst->src[vI] = fGraphCopyDupTensor(pHashSet, pNodeCopies, pCtxAllocated, pCtxUnallocated, vS);
  }

  pNodeCopies[vId] = vDst;
  return vDst;
}

static void fGraphCopyInitTensor(struct ggml_hash_set *pHashSet, struct ggml_tensor **pNodeCopies, bool *pNodeInit,
                                 struct ggml_tensor *pSrc) {
  size_t vId = ggml_hash_find(pHashSet, pSrc);
  if (pNodeInit[vId]) {
    return;
  }
  pNodeInit[vId] = true;

  struct ggml_tensor *vDst = pNodeCopies[vId];
  if (vDst->view_src != NULL) {
    fGraphCopyInitTensor(pHashSet, pNodeCopies, pNodeInit, pSrc->view_src);
    enum ggml_status vStatus = ggml_backend_view_init(vDst);
    GGML_ASSERT(vStatus == GGML_STATUS_SUCCESS);
  } else {
    ggml_backend_tensor_copy(pSrc, vDst);
  }

  // init src
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    struct ggml_tensor *vS = pSrc->src[vI];
    if (vS == NULL) {
      continue;
    }
    fGraphCopyInitTensor(pHashSet, pNodeCopies, pNodeInit, vS);
  }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t pBackend, struct ggml_cgraph *pGraph) {
  GGML_ASSERT(pGraph);
  struct ggml_hash_set vHashSet = ggml_hash_set_new(pGraph->visited_hash_set.size);
  struct ggml_tensor **node_copies = (ggml_tensor **)calloc(vHashSet.size, sizeof(node_copies[0])); // NOLINT
  bool *vNodeInit = (bool *)calloc(vHashSet.size, sizeof(vNodeInit[0]));

  struct ggml_init_params vParams = {/* .mem_size   = */ ggml_tensor_overhead() * vHashSet.size +
                                         ggml_graph_overhead_custom(pGraph->size, false),
                                     /* .mem_buffer = */ NULL,
                                     /* .no_alloc   = */ true};

  struct ggml_context *vCtxAllocated = ggml_init(vParams);
  struct ggml_context *vCtxUnallocated = ggml_init(vParams);

  if (vCtxAllocated == NULL || vCtxUnallocated == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
    ggml_hash_set_free(&vHashSet);
    free(node_copies);
    free(vNodeInit);
    ggml_free(vCtxAllocated);
    ggml_free(vCtxUnallocated);
    return {
        /* .buffer           = */ NULL,
        /* .ctx_allocated    = */ NULL,
        /* .ctx_unallocated  = */ NULL,
        /* .graph            = */ NULL,
    };
  }

  // dup nodes
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    fGraphCopyDupTensor(vHashSet, node_copies, vCtxAllocated, vCtxUnallocated, vNode);
  }

  // allocate nodes
  ggml_backend_buffer_t vBuffer = ggml_backend_alloc_ctx_tensors(vCtxAllocated, pBackend);
  if (vBuffer == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
    ggml_hash_set_free(&vHashSet);
    free(node_copies);
    free(vNodeInit);
    ggml_free(vCtxAllocated);
    ggml_free(vCtxUnallocated);
    return {
        /* .buffer           = */ NULL,
        /* .ctx_allocated    = */ NULL,
        /* .ctx_unallocated  = */ NULL,
        /* .graph            = */ NULL,
    };
  }

  // printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

  // copy data and init views
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    fGraphCopyInitTensor(&vHashSet, node_copies, vNodeInit, vNode);
  }

  // build graph copy
  struct ggml_cgraph *vGraphCopy = ggml_new_graph_custom(vCtxAllocated, pGraph->size, false);
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    struct ggml_tensor *vNodeCopy = node_copies[ggml_hash_find(&vHashSet, vNode)];
    vGraphCopy->nodes[vI] = vNodeCopy;
  }
  vGraphCopy->n_nodes = pGraph->n_nodes;

  ggml_hash_set_free(&vHashSet);
  free(node_copies);
  free(vNodeInit);

  return {
      /* .buffer           = */ vBuffer,
      /* .ctx_allocated    = */ vCtxAllocated,
      /* .ctx_unallocated  = */ vCtxUnallocated,
      /* .graph            = */ vGraphCopy,
  };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy pCopy) {
  ggml_backend_buffer_free(pCopy.buffer);
  ggml_free(pCopy.ctx_allocated);
  ggml_free(pCopy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t pBackend1, ggml_backend_t pBackend2, struct ggml_cgraph *pGraph,
                                        ggml_backend_eval_callback pCallback, void *pUserData,
                                        struct ggml_tensor const *const *pTestNodes, size_t pNumTestNodes) {
  struct ggml_backend_graph_copy vCopy = ggml_backend_graph_copy(pBackend2, pGraph);
  if (vCopy.buffer == NULL) {
    return false;
  }

  struct ggml_cgraph *vG1 = pGraph;
  struct ggml_cgraph *vG2 = vCopy.graph;

  assert(vG1->n_nodes == vG2->n_nodes);

  if (pNumTestNodes != 0) {
    GGML_ASSERT(pTestNodes);
    // Compute the whole graph and only test the output for specific tensors
    ggml_backend_graph_compute(pBackend1, vG1);
    ggml_backend_graph_compute(pBackend2, vG2);

    bool vVerified = false;
    for (int vI = 0; vI < vG1->n_nodes; vI++) {
      for (size_t vJ = 0; vJ < pNumTestNodes; ++vJ) {
        if (vG1->nodes[vI] == pTestNodes[vJ]) {
          pCallback(vI, vG1->nodes[vI], vG2->nodes[vI], pUserData);
          vVerified = true;
        }
      }
    }
    GGML_ASSERT(vVerified);
  } else {
    for (int vI = 0; vI < vG1->n_nodes; vI++) {
      struct ggml_tensor *vT1 = vG1->nodes[vI];
      struct ggml_tensor *vT2 = vG2->nodes[vI];

      assert(vT1->op == vT2->op && ggml_are_same_layout(vT1, vT2));

      struct ggml_cgraph vG1v = ggml_graph_view(vG1, vI, vI + 1);
      struct ggml_cgraph vG2v = ggml_graph_view(vG2, vI, vI + 1);

      ggml_backend_graph_compute(pBackend1, &vG1v);
      ggml_backend_graph_compute(pBackend2, &vG2v);

      if (fGgmlIsViewOp(vT1->op)) {
        continue;
      }

      // compare results, calculate rms etc
      if (!pCallback(vI, vT1, vT2, pUserData)) {
        break;
      }
    }
  }
  ggml_backend_graph_copy_free(vCopy);

  return true;
}

// CPU backend - buffer

static void *fGgmlBackendCpuBufferGetBase(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  uintptr_t vData = (uintptr_t)pBuffer->context;

  // align the buffer
  if (vData % TENSOR_ALIGNMENT != 0) {
    vData = GGML_PAD(vData, TENSOR_ALIGNMENT);
  }

  return (void *)vData;
}

static void fGgmlBackendCpuBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);
  ggml_aligned_free(pBuffer->context, pBuffer->size);
}

static void fGgmlBackendCpuBufferMemsetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                              uint8_t pValue, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  memset((char *)pTensor->data + pOffset, pValue, pSize);

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendCpuBufferSetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                           const void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  memcpy((char *)pTensor->data + pOffset, pData, pSize);

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendCpuBufferGetTensor(ggml_backend_buffer_t pBuffer, const struct ggml_tensor *pTensor,
                                           void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  memcpy(pData, (const char *)pTensor->data + pOffset, pSize);

  GGML_UNUSED(pBuffer);
}

static bool fGgmlBackendCpuBufferCpyTensor(ggml_backend_buffer_t pBuffer, const struct ggml_tensor *pSrc,
                                           struct ggml_tensor *pDst) {
  GGML_ASSERT(pSrc);
  if (ggml_backend_buffer_is_host(pSrc->buffer)) {
    memcpy(pDst->data, pSrc->data, ggml_nbytes(pSrc));
    return true;
  }
  return false;

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendCpuBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  GGML_ASSERT(pBuffer);
  memset(pBuffer->context, pValue, pBuffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ fGgmlBackendCpuBufferFreeBuffer,
    /* .get_base        = */ fGgmlBackendCpuBufferGetBase,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ fGgmlBackendCpuBufferMemsetTensor,
    /* .set_tensor      = */ fGgmlBackendCpuBufferSetTensor,
    /* .get_tensor      = */ fGgmlBackendCpuBufferGetTensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ fGgmlBackendCpuBufferCpyTensor,
    /* .clear           = */ fGgmlBackendCpuBufferClear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ fGgmlBackendCpuBufferGetBase,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ fGgmlBackendCpuBufferMemsetTensor,
    /* .set_tensor      = */ fGgmlBackendCpuBufferSetTensor,
    /* .get_tensor      = */ fGgmlBackendCpuBufferGetTensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ fGgmlBackendCpuBufferCpyTensor,
    /* .clear           = */ fGgmlBackendCpuBufferClear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char *fGgmlBackendCpuBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  return "CPU";

  GGML_UNUSED(pBuft);
}

static ggml_backend_buffer_t fGgmlBackendCpuBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  void *vData = ggml_aligned_malloc(pSize);

  if (vData == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, pSize);
    return NULL;
  }

  return ggml_backend_buffer_init(pBuft, ggml_backend_cpu_buffer_i, vData, pSize);
}

static size_t fGgmlBackendCpuBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  return TENSOR_ALIGNMENT;

  GGML_UNUSED(pBuft);
}

static bool fGgmlBackendCpuBufferTypeIsHost(ggml_backend_buffer_type_t pBuft) {
  return true;

  GGML_UNUSED(pBuft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
  static struct ggml_backend_buffer_type vGgmlBackendCpuBufferType = {
      /* .iface   = */ {
          /* .get_name         = */ fGgmlBackendCpuBufferTypeGetName,
          /* .alloc_buffer     = */ fGgmlBackendCpuBufferTypeAllocBuffer,
          /* .get_alignment    = */ fGgmlBackendCpuBufferTypeGetAlignment,
          /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
          /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
          /* .is_host          = */ fGgmlBackendCpuBufferTypeIsHost,
      },
      /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */ NULL,
  };

  return &vGgmlBackendCpuBufferType;
}

static const char *fGgmlBackendCpuBufferFromPtrTypeGetName(ggml_backend_buffer_type_t pBuft) {
  return "CPU_Mapped";

  GGML_UNUSED(pBuft);
}

static ggml_backend_buffer_type_t fGgmlBackendCpuBufferFromPtrType(void) {
  static struct ggml_backend_buffer_type vGgmlBackendCpuBufferType = {
      /* .iface   = */ {
          /* .get_name         = */ fGgmlBackendCpuBufferFromPtrTypeGetName,
          /* .alloc_buffer     = */ fGgmlBackendCpuBufferTypeAllocBuffer,
          /* .get_alignment    = */ fGgmlBackendCpuBufferTypeGetAlignment,
          /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
          /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
          /* .is_host          = */ fGgmlBackendCpuBufferTypeIsHost,
      },
      /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */ NULL,
  };

  return &vGgmlBackendCpuBufferType;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void *pPtr, size_t pSize) {
  GGML_ASSERT((uintptr_t)pPtr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
  return ggml_backend_buffer_init(fGgmlBackendCpuBufferFromPtrType(), ggml_backend_cpu_buffer_from_ptr_i, pPtr, pSize);
}
