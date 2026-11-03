#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type pType, int pCc, int64_t pNe11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type pType, int pCc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                             const ggml_tensor *pIds, ggml_tensor *pDst,
                             const ggml_cuda_mm_fusion_args_host *pFusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                                ggml_tensor *pDst, const char *pSrc0DdI, const float *pSrc1DdfI, const char *pSrc1DdqI,
                                float *pDstDdI, const int64_t pRowLow, const int64_t pRowHigh, const int64_t pSrc1Ncols,
                                const int64_t pSrc1PaddedRowSize, cudaStream_t pStream);
