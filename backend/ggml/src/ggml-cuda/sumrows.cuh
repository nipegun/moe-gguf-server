#include "common.cuh"

void sum_rows_f32_cuda(const float *pX, float *pDst, const int pNcols, const int pNrows, cudaStream_t pStream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
