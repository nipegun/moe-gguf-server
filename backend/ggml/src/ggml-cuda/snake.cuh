#include "common.cuh"

// Fusion entry point. Caller supplies x/a/inv_b explicitly.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context &pCtx, const ggml_tensor *pX, const ggml_tensor *pA,
                              const ggml_tensor *pInvB, ggml_tensor *pDst);
