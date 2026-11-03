#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
  bool work_size(int /* n_threads */, const struct ggml_tensor *pOp, size_t &pSize) override {
    pSize = ggml_backend_amx_desired_wsize(pOp);
    return true;
  }

  bool compute_forward(struct ggml_compute_params *pParams, struct ggml_tensor *pOp) override {
    if (pOp->op == GGML_OP_MUL_MAT) {
      ggml_backend_amx_mul_mat(pParams, pOp);
      return true;
    }
    return false;
  }
};

static ggml::cpu::tensor_traits *fGetTensorTraits(ggml_backend_buffer_t, struct ggml_tensor *) {
  static tensor_traits vTraits;
  return &vTraits;
}
} // namespace ggml::cpu::amx

// AMX buffer interface
static void fGgmlBackendAmxBufferFreeBuffer(ggml_backend_buffer_t pBuffer) { free(pBuffer->context); }

static void *fGgmlBackendAmxBufferGetBase(ggml_backend_buffer_t pBuffer) { return (void *)(pBuffer->context); }

static enum ggml_status fGgmlBackendAmxBufferInitTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor) {
  pTensor->extra = (void *)ggml::cpu::amx::fGetTensorTraits(pBuffer, pTensor);

  GGML_UNUSED(pBuffer);
  return GGML_STATUS_SUCCESS;
}

static void fGgmlBackendAmxBufferMemsetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                              uint8_t pValue, size_t pOffset, size_t pSize) {
  memset((char *)pTensor->data + pOffset, pValue, pSize);

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendAmxBufferSetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                           const void *pData, size_t pOffset, size_t pSize) {
  if (qtype_has_amx_kernels(pTensor->type)) {
    GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, pTensor->name, ggml_type_name(pTensor->type));
    ggml_backend_amx_convert_weight(pTensor, pData, pOffset, pSize);
  } else {
    memcpy((char *)pTensor->data + pOffset, pData, pSize);
  }

  GGML_UNUSED(pBuffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void *
data, size_t offset, size_t size) { GGML_ASSERT(!qtype_has_amx_kernels(tensor->type)); memcpy(data, (const char
*)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct
ggml_tensor * dst) { if (ggml_backend_buffer_is_host(src->buffer)) { if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

static void fGgmlBackendAmxBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  memset(pBuffer->context, pValue, pBuffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ fGgmlBackendAmxBufferFreeBuffer,
    /* .get_base        = */ fGgmlBackendAmxBufferGetBase,
    /* .init_tensor     = */ fGgmlBackendAmxBufferInitTensor,
    /* .memset_tensor   = */ fGgmlBackendAmxBufferMemsetTensor,
    /* .set_tensor      = */ fGgmlBackendAmxBufferSetTensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ fGgmlBackendAmxBufferClear,
    /* .reset           = */ nullptr,
};

static const char *fGgmlBackendAmxBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  return "AMX";

  GGML_UNUSED(pBuft);
}

static ggml_backend_buffer_t fGgmlBackendAmxBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  void *vData = ggml_aligned_malloc(pSize);
  if (vData == NULL) {
    fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, pSize);
    return NULL;
  }

  return ggml_backend_buffer_init(pBuft, ggml_backend_amx_buffer_interface, vData, pSize);
}

static size_t fGgmlBackendAmxBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  return TENSOR_ALIGNMENT;

  GGML_UNUSED(pBuft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
  bool supports_op(ggml_backend_dev_t, const struct ggml_tensor *pOp) override {
    if (pOp->op != GGML_OP_MUL_MAT) {
      return false;
    }
    auto *vSrc0 = pOp->src[0];
    auto *vSrc1 = pOp->src[1];

    if (!ggml_is_contiguous(vSrc0) || !ggml_is_contiguous(vSrc1)) {
      return false;
    }
    if (!vSrc0->buffer || vSrc0->buffer->buft != ggml_backend_amx_buffer_type()) {
      return false;
    }
    if (vSrc1->buffer && !ggml_backend_buft_is_host(vSrc1->buffer->buft)) {
      return false;
    }
    if (pOp->ne[0] % (TILE_N * 2)) {
      return false;
    }
    int vAlignment;
    switch (vSrc0->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q8_0:
      vAlignment = TILE_K;
      break;
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ4_XS:
      vAlignment = 256; // QK_K
      break;
    case GGML_TYPE_F16:
      vAlignment = 16;
      break;
    default:
      return false;
    }
    if (vSrc0->ne[0] % vAlignment) {
      return false;
    }
    if (vSrc1->type != GGML_TYPE_F32) {
      return false;
    }
    return true;
  }

  ggml::cpu::tensor_traits *get_tensor_traits(const struct ggml_tensor *pOp) override {
    if (pOp->op == GGML_OP_MUL_MAT && pOp->src[0]->buffer &&
        pOp->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
      return (ggml::cpu::tensor_traits *)pOp->src[0]->extra;
    }

    return nullptr;
  }
};
} // namespace ggml::cpu::amx

static size_t fGgmlBackendAmxBufferTypeGetAllocSize(ggml_backend_buffer_type_t pBuft, const ggml_tensor *pTensor) {
  return ggml_backend_amx_get_alloc_size(pTensor);

  GGML_UNUSED(pBuft);
}

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

static bool fGgmlAmxInit() {
#if defined(__linux__)
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    fprintf(stderr, "AMX is not ready to be used!\n");
    return false;
  }
  return true;
#elif defined(_WIN32)
  return true;
#else
  return false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
  static struct ggml_backend_buffer_type vGgmlBackendBufferTypeAmx = {
      /* .iface = */ {
          /* .get_name         = */ fGgmlBackendAmxBufferTypeGetName,
          /* .alloc_buffer     = */ fGgmlBackendAmxBufferTypeAllocBuffer,
          /* .get_alignment    = */ fGgmlBackendAmxBufferTypeGetAlignment,
          /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
          /* .get_alloc_size   = */ fGgmlBackendAmxBufferTypeGetAllocSize,
          /* .is_host          = */ nullptr,
      },
      /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
  };

  if (!fGgmlAmxInit()) {
    return nullptr;
  }

  return &vGgmlBackendBufferTypeAmx;
}

#endif // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
