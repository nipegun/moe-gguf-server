#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
} // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params *pParams, struct ggml_tensor *pOp) {
  for (auto vExtra : ggml_backend_cpu_get_extra_buffer_types()) {
    if (vExtra && vExtra->context) {
      auto vBufExtra = (ggml::cpu::extra_buffer_type *)vExtra->context;
      auto vTensorTraits = vBufExtra->get_tensor_traits(pOp);
      if (vTensorTraits && vTensorTraits->compute_forward(pParams, pOp)) {
        return true;
      }
    }
  }
  return false;
}

bool ggml_cpu_extra_work_size(int pNThreads, const struct ggml_tensor *pOp, size_t *pSize) {
  for (auto vExtra : ggml_backend_cpu_get_extra_buffer_types()) {
    if (vExtra && vExtra->context) {
      auto vBufExtra = (ggml::cpu::extra_buffer_type *)vExtra->context;
      auto vTensorTraits = vBufExtra->get_tensor_traits(pOp);
      if (vTensorTraits && vTensorTraits->work_size(pNThreads, pOp, *pSize)) {
        return true;
      }
    }
  }
  return false;
}
