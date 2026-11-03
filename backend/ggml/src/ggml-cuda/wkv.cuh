#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
