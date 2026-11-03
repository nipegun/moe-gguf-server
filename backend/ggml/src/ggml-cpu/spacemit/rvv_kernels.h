#pragma once

#include "ggml-cpu-impl.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace spacemit_kernels {

constexpr auto div_round_up(auto pUp, auto pDown) { return (pUp + pDown - 1) / pDown; }

// Q8 Blk [f32] [s16] [int8 * blk_len]
// Q8 Blk N [f32 * N] [s16 * N] [int8 * blk_len * N]
constexpr size_t q8_blk_size(size_t pBlkLen, bool pWithBlkSum = false) {
  const size_t cBlkSize = sizeof(float) + pBlkLen * sizeof(int8_t) + (pWithBlkSum ? sizeof(int16_t) : 0);
  return cBlkSize;
}

// Q8 HP row block: K is split into K32 subblocks.
// Each subblock stores [f32 scale] [int8 * 32], with an optional fp16 sum trailer per subblock.
constexpr size_t q8_hp_blk_size(size_t pBlkLen, bool pWithBlkSum = false, bool pWithBlkScale = false) {
  const size_t cSubblkCount = div_round_up(pBlkLen, size_t(32));
  const size_t cBlkSize = pBlkLen * sizeof(int8_t) + cSubblkCount * sizeof(_Float16) +
                          (pWithBlkSum ? cSubblkCount * sizeof(_Float16) : 0) + (pWithBlkScale ? sizeof(_Float16) : 0);
  return cBlkSize;
}

// Q8K Blk [f32] [s16 * (blk_len / 16)] [int8 * blk_len]
// Q8K Blk N [f32 * N] [s16 * (blk_len / 16) * N] [int8 * blk_len * N]
constexpr size_t q8k_blk_size(size_t pBlkLen) {
  const size_t cBlkSize = sizeof(float) + pBlkLen * sizeof(int8_t) + sizeof(int16_t) * pBlkLen / 16;
  return cBlkSize;
}

using quantize_a_row_def = std::function<void(size_t, const float *, size_t, uint8_t *)>;

namespace rvv {
void memcpy1d(void *pDst, const void *pSrc, int64_t pSize);

void memcpy2d(void *pDst, int64_t pDstStride, const void *pSrc, int64_t pSrcStride, int64_t pTileRows, int64_t pSize);

void forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                        int pIr1, void *pTcmBuffer, size_t pTcmBufferSize);

void forward_flash_attn_ext_f16_tiled_vlen1024_vf16(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                    int pIr1, void *pTcmBuffer, size_t pTcmBufferSize);

void forward_rms_norm_f32(ggml_compute_params *pParams, ggml_tensor *pOp);

void forward_norm_f32(ggml_compute_params *pParams, ggml_tensor *pOp);

void forward_cont_with_permute(ggml_compute_params *pParams, ggml_tensor *pOp);

void forward_cpy_with_permute(ggml_compute_params *pParams, ggml_tensor *pOp);

template <typename T> void forward_get_rows(ggml_compute_params *pParams, ggml_tensor *pOp);

template <typename T> void forward_concat(ggml_compute_params *pParams, ggml_tensor *pOp);

template <ggml_op op_type, typename T> void forward_binary(ggml_compute_params *pParams, ggml_tensor *pOp);

template <typename T> void forward_sum_rows(const ggml_compute_params *pParams, ggml_tensor *pOp);

template <typename T> void forward_repeat_nrows(ggml_compute_params *pParams, ggml_tensor *pOp);

template <typename T> void forward_repeat_dim1(ggml_compute_params *pParams, ggml_tensor *pOp);

void quantize_a_row_i8(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

void quantize_a_4row_i8(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

void quantize_a_row_i8_hp(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

void quantize_a_4row_i8_hp(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

void quantize_a_row_i8k(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

void quantize_a_4row_i8k(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr);

} // namespace rvv

} // namespace spacemit_kernels
