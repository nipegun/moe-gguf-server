#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
  float *data;         // rollback slot 0
  int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst,
                                              ggml_cuda_gated_delta_net_fused_cache pCache);
