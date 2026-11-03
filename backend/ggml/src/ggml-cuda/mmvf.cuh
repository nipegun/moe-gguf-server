#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                             const ggml_tensor *pIds, ggml_tensor *pDst,
                             const ggml_cuda_mm_fusion_args_host *pFusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                                ggml_tensor *pDst, const char *pSrc0DdI, const float *pSrc1DdfI, const char *pSrc1DdqI,
                                float *pDstDdI, const int64_t pRowLow, const int64_t pRowHigh, const int64_t pSrc1Ncols,
                                const int64_t pSrc1PaddedRowSize, cudaStream_t pStream);

bool ggml_cuda_should_use_mmvf(enum ggml_type pType, int pCc, const int64_t *pSrc0Ne, const size_t *pSrc0Nb,
                               int64_t pNe11);
