#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
