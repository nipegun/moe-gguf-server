#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, ggml_tensor *pBiasAddNode = nullptr,
                           ggml_tensor *pSiluDst = nullptr);
