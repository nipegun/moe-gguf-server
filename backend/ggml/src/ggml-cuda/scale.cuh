#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
