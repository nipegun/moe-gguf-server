#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, ggml_tensor *pSrc1);

void ggml_cuda_dup(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
