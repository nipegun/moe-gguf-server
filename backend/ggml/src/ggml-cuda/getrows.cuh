#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(const void *pSrc0D, ggml_type pSrc0Type, const int32_t *pSrc1D, void *pDstD, ggml_type pDstType,
                   int64_t pNe00, size_t pNb01, size_t pNb02, size_t pNb03, int64_t pNe10, int64_t pNe11, int64_t pNe12,
                   size_t pNb10, size_t pNb11, size_t pNb12, size_t pNb1, size_t pNb2, size_t pNb3,
                   cudaStream_t pStream);

void ggml_cuda_op_get_rows(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
