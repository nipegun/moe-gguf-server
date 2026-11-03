#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING % CUDA_QUANTIZE_BLOCK_SIZE == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(const float *pX, const int32_t *pIds, void *pVy, ggml_type pTypeSrc0, int64_t pNe00,
                                int64_t pS01, int64_t pS02, int64_t pS03, int64_t pNe0, int64_t pNe1, int64_t pNe2,
                                int64_t pNe3, cudaStream_t pStream);

void quantize_row_q8_1_cuda(const float *pX, const int32_t *pIds, void *pVy, ggml_type pTypeSrc0, int64_t pNe00,
                            int64_t pS01, int64_t pS02, int64_t pS03, int64_t pNe0, int64_t pNe1, int64_t pNe2,
                            int64_t pNe3, cudaStream_t pStream);

void quantize_mmq_q8_1_cuda(const float *pX, const int32_t *pIds, void *pVy, ggml_type pTypeSrc0, int64_t pNe00,
                            int64_t pS01, int64_t pS02, int64_t pS03, int64_t pNe0, int64_t pNe1, int64_t pNe2,
                            int64_t pNe3, cudaStream_t pStream);

void quantize_mmq_fp4_cuda(const float *pX, const int32_t *pIds, void *pVy, float *pScale, ggml_type pTypeSrc0,
                           bool pUseAlignedFloat8, int64_t pNe00, int64_t pS01, int64_t pS02, int64_t pS03,
                           int64_t pNe0, int64_t pNe1, int64_t pNe2, int64_t pNe3, cudaStream_t pStream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *pX, const int32_t *pIdsSrc1Inv, void *pVy, float *pScale,
                                   ggml_type pTypeSrc0, bool pUseAlignedFloat8, int64_t pNe00, int64_t pStrideToken,
                                   int64_t pNe0, int64_t pNTokens, int64_t pNrowsDst, int pNExpertUsed,
                                   cudaStream_t pStream);

void quantize_scatter_mmq_q8_1_cuda(const float *pX, const int32_t *pIdsSrc1Inv, void *pVy, ggml_type pTypeSrc0,
                                    int64_t pNe00, int64_t pStrideToken, int64_t pNe0, int64_t pNTokens,
                                    int64_t pNrowsDst, int pNExpertUsed, cudaStream_t pStream);
