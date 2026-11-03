#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pSetRows);
