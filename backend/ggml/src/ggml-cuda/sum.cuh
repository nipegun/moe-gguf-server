#include "common.cuh"

void sum_f32_cuda(ggml_cuda_pool &pPool, const float *pX, float *pDst, const int64_t pNe, cudaStream_t pStream);

void ggml_cuda_op_sum(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
