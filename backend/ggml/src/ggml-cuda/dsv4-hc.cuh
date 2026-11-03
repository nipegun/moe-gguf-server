#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
