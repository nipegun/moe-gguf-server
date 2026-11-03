#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pSrc);
