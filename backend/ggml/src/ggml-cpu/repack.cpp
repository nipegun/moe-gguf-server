#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include "arch-fallback.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio> // for GGML_ASSERT

#include "repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

static inline int fNearestInt(float pFval) {
  assert(fabsf(pFval) <= 4194303.f);
  float vVal = pFval + 12582912.f;
  int vI;
  memcpy(&vI, &vVal, sizeof(int));
  return (vI & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

extern "C" {

#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float *GGML_RESTRICT x, void *GGML_RESTRICT vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0x4 *GGML_RESTRICT y = (block_q8_0x4 *)vy;

  // scalar
  const int blck_size_interleave = 1;
  float srcv[4][QK8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < QK8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
    }

    for (int j = 0; j < QK8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
}

void ggml_quantize_mat_q8_K_4x1_generic(const float *GGML_RESTRICT x, void *GGML_RESTRICT vy, int64_t k) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  block_q8_Kx4 *GGML_RESTRICT y = (block_q8_Kx4 *)vy;

  const int blck_size_interleave = 1;
  float srcv[4][QK_K];
  float iscale[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max
      float max = 0;

      for (int j = 0; j < QK_K; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
        // Update the maximum value of the corresponding super block
        if (amax < fabsf(srcv[row_iter][j])) {
          amax = fabsf(srcv[row_iter][j]);
          max = srcv[row_iter][j];
        }
      }

      iscale[row_iter] = amax ? -127.f / max : 0;
      y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
    }

    for (int j = 0; j < QK_K / 4; j++) {
      y[i].bsums[j] = 0;
    }
    for (int j = 0; j < QK_K * 4; j++) {
      int src_id = j % 4;
      int src_offset = j / 4;
      int index = ((j >> 6) << 2) + (j & 3);

      float x0 = srcv[src_id][src_offset] * iscale[src_id];
      y[i].qs[j] = fNearestInt(x0);
      y[i].bsums[index] += y[i].qs[j];
    }
  }
}
#endif

void ggml_quantize_mat_q8_0_4x4_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;

  // scalar
  const int cBlckSizeInterleave = 4;
  float vSrcv[4][QK8_0];
  float vId[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      float vAmax = 0.0f; // absolute max

      for (int vJ = 0; vJ < QK8_0; vJ++) {
        vSrcv[vRowIter][vJ] = pX[vRowIter * pK + vI * QK8_0 + vJ];
        vAmax = MAX(vAmax, fabsf(vSrcv[vRowIter][vJ]));
      }

      const float cD = vAmax / ((1 << 7) - 1);
      vId[vRowIter] = cD ? 1.0f / cD : 0.0f;

      vY[vI].d[vRowIter] = GGML_CPU_FP32_TO_FP16(cD);
    }

    for (int vJ = 0; vJ < QK8_0 * 4; vJ++) {
      int vSrcOffset = (vJ / (4 * cBlckSizeInterleave)) * cBlckSizeInterleave;
      int vSrcId = (vJ % (4 * cBlckSizeInterleave)) / cBlckSizeInterleave;
      vSrcOffset += (vJ % cBlckSizeInterleave);

      float vX0 = vSrcv[vSrcId][vSrcOffset] * vId[vSrcId];
      vY[vI].qs[vJ] = roundf(vX0);
    }
  }
}

void ggml_quantize_mat_q8_0_4x8_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;

  // scalar
  const int cBlckSizeInterleave = 8;
  float vSrcv[4][QK8_0];
  float vId[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      float vAmax = 0.0f; // absolute max

      for (int vJ = 0; vJ < QK8_0; vJ++) {
        vSrcv[vRowIter][vJ] = pX[vRowIter * pK + vI * QK8_0 + vJ];
        vAmax = MAX(vAmax, fabsf(vSrcv[vRowIter][vJ]));
      }

      const float cD = vAmax / ((1 << 7) - 1);
      vId[vRowIter] = cD ? 1.0f / cD : 0.0f;

      vY[vI].d[vRowIter] = GGML_CPU_FP32_TO_FP16(cD);
    }

    for (int vJ = 0; vJ < QK8_0 * 4; vJ++) {
      int vSrcOffset = (vJ / (4 * cBlckSizeInterleave)) * cBlckSizeInterleave;
      int vSrcId = (vJ % (4 * cBlckSizeInterleave)) / cBlckSizeInterleave;
      vSrcOffset += (vJ % cBlckSizeInterleave);

      float vX0 = vSrcv[vSrcId][vSrcOffset] * vId[vSrcId];
      vY[vI].qs[vJ] = roundf(vX0);
    }
  }
}

void ggml_quantize_mat_q8_K_4x4_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK_K == 256);
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  block_q8_Kx4 *GGML_RESTRICT vY = (block_q8_Kx4 *)pVy;

  // scalar
  const int cBlckSizeInterleave = 4;
  float vSrcv[4][QK_K];
  float vIscale[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      float vAmax = 0.0f; // absolute max
      float vMax = 0;

      for (int vJ = 0; vJ < QK_K; vJ++) {
        vSrcv[vRowIter][vJ] = pX[vRowIter * pK + vI * QK_K + vJ];
        // Update the maximum value of the corresponding super block
        if (vAmax < fabsf(vSrcv[vRowIter][vJ])) {
          vAmax = fabsf(vSrcv[vRowIter][vJ]);
          vMax = vSrcv[vRowIter][vJ];
        }
      }

      vIscale[vRowIter] = vAmax ? -127.f / vMax : 0;

      vY[vI].d[vRowIter] = vAmax ? 1 / vIscale[vRowIter] : 0;
    }

    for (int vJ = 0; vJ < QK_K / 4; vJ++) {
      vY[vI].bsums[vJ] = 0;
    }

    // Quants values are interleaved in sequence of four bytes from corresponding super blocks
    // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
    // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
    for (int vJ = 0; vJ < QK_K * 4; vJ++) {
      int vSrcOffset = (vJ / (4 * cBlckSizeInterleave)) * cBlckSizeInterleave;
      int vSrcId = (vJ % (4 * cBlckSizeInterleave)) / cBlckSizeInterleave;
      vSrcOffset += (vJ % cBlckSizeInterleave);
      int vIndex = (((vJ & 15) >> 2) << 2) + ((vJ >> 8) << 4) + ((vJ >> 6) & 3);

      float vX0 = vSrcv[vSrcId][vSrcOffset] * vIscale[vSrcId];
      vY[vI].qs[vJ] = fNearestInt(vX0);
      vY[vI].bsums[vIndex] += vY[vI].qs[vJ];
    }
  }
}

void ggml_quantize_mat_q8_K_4x8_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK_K == 256);
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  block_q8_Kx4 *GGML_RESTRICT vY = (block_q8_Kx4 *)pVy;

  // scalar
  const int cBlckSizeInterleave = 8;
  float vSrcv[4][QK_K];
  float vIscale[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      float vAmax = 0.0f; // absolute max
      float vMax = 0;

      for (int vJ = 0; vJ < QK_K; vJ++) {
        vSrcv[vRowIter][vJ] = pX[vRowIter * pK + vI * QK_K + vJ];
        // Update the maximum value of the corresponding super block
        if (vAmax < fabsf(vSrcv[vRowIter][vJ])) {
          vAmax = fabsf(vSrcv[vRowIter][vJ]);
          vMax = vSrcv[vRowIter][vJ];
        }
      }

      vIscale[vRowIter] = vAmax ? -127.f / vMax : 0;

      vY[vI].d[vRowIter] = vAmax ? 1 / vIscale[vRowIter] : 0;
    }

    for (int vJ = 0; vJ < QK_K / 4; vJ++) {
      vY[vI].bsums[vJ] = 0;
    }

    // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
    // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
    // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
    for (int vJ = 0; vJ < QK_K * 4; vJ++) {
      int vSrcOffset = (vJ / (4 * cBlckSizeInterleave)) * cBlckSizeInterleave;
      int vSrcId = (vJ % (4 * cBlckSizeInterleave)) / cBlckSizeInterleave;
      vSrcOffset += (vJ % cBlckSizeInterleave);
      int vIndex = (((vJ & 31) >> 3) << 2) + ((vJ >> 8) << 4) + ((vJ >> 6) & 3);

      float vX0 = vSrcv[vSrcId][vSrcOffset] * vIscale[vSrcId];
      vY[vI].qs[vJ] = fNearestInt(vX0);
      vY[vI].bsums[vIndex] += vY[vI].qs[vJ];
    }
  }
}

} // extern "C"

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
void ggml_quantize_mat_t(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pNrow, int64_t pNPerRow);

template <>
void ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pNrow,
                                            int64_t pNPerRow) {
  assert(pNrow == 4);
  UNUSED(pNrow);
  ggml_quantize_mat_q8_0_4x4(pX, pVy, pNPerRow);
}

template <>
void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pNrow,
                                            int64_t pNPerRow) {
  assert(pNrow == 4);
  UNUSED(pNrow);
  ggml_quantize_mat_q8_0_4x8(pX, pVy, pNPerRow);
}

template <>
void ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pNrow,
                                            int64_t pNPerRow) {
  assert(pNrow == 4);
  UNUSED(pNrow);
  ggml_quantize_mat_q8_K_4x4(pX, pVy, pNPerRow);
}

template <>
void ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pNrow,
                                            int64_t pNPerRow) {
  assert(pNrow == 4);
  UNUSED(pNrow);
  ggml_quantize_mat_q8_K_4x8(pX, pVy, pNPerRow);
}

#if defined __riscv_zvfh
template <>
void ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(const float *GGML_RESTRICT x, void *GGML_RESTRICT vy, int64_t nrow,
                                            int64_t n_per_row) {
  assert(nrow == 4);
  UNUSED(nrow);
  ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <>
void ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(const float *GGML_RESTRICT x, void *GGML_RESTRICT vy, int64_t nrow,
                                            int64_t n_per_row) {
  assert(nrow == 4);
  UNUSED(nrow);
  ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <int M, int N>
static void fGgmlGemvQ6KNxMQ8KGenericImpl(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                          const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cBlocklen = M;
  constexpr int cNcolsInterleaved = N;
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cBlocksPerHalf = 64 / cBlocklen;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[8];

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q6_Kx8 *vBPtr = (const block_q6_Kx8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0f;
    }

    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        const int cBaseL = (vK / cBlocksPerHalf) * 128 + (vK % cBlocksPerHalf) * cBlocklen;
        const int cBaseH = cBaseL + 64;

        const int cScaleIdxL = cBaseL / 16;
        const int cScaleIdxH = cBaseH / 16;

        const int cQhShiftL = ((cBaseL % 128) / 32) * 2;
        const int cQhShiftH = ((cBaseH % 128) / 32) * 2;

        const int cQhHalfL = (cBaseL / 128) * 32;
        const int cQhHalfH = (cBaseH / 128) * 32;

        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          const int8_t cScaleL = vBPtr[vL].scales[cScaleIdxL * cNcolsInterleaved + vJ];
          const int8_t cScaleH = vBPtr[vL].scales[cScaleIdxH * cNcolsInterleaved + vJ];

          int vSumiL = 0;
          int vSumiH = 0;

          for (int vI = 0; vI < cBlocklen; vI++) {
            const int cQlPos = vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI;
            const int cL4 = vBPtr[vL].ql[cQlPos] & 0xF;
            const int cHi4 = (vBPtr[vL].ql[cQlPos] >> 4) & 0xF;

            const int cQhIdxL = cQhHalfL + ((cBaseL + vI) % 32);
            const int cQhChunkL = cQhIdxL / cBlocklen;
            const int cQhPosL = cQhIdxL % cBlocklen;
            const int cQhOffsetL = cQhChunkL * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPosL;
            const int cHi2L = (vBPtr[vL].qh[cQhOffsetL] >> cQhShiftL) & 0x3;

            const int cQhIdxH = cQhHalfH + ((cBaseH + vI) % 32);
            const int cQhChunkH = cQhIdxH / cBlocklen;
            const int cQhPosH = cQhIdxH % cBlocklen;
            const int cQhOffsetH = cQhChunkH * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPosH;
            const int cHi2H = (vBPtr[vL].qh[cQhOffsetH] >> cQhShiftH) & 0x3;

            const int cQL = ((cHi2L << 4) | cL4) - 32;
            const int cQH = ((cHi2H << 4) | cHi4) - 32;

            const int8_t cAL = vAPtr[vL].qs[cBaseL + vI];
            const int8_t cAH = vAPtr[vL].qs[cBaseH + vI];

            vSumiL += cQL * cAL;
            vSumiH += cQH * cAH;
          }

          vSumf[vJ] += (vSumiL * cScaleL + vSumiH * cScaleH) * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d;
        }
      }
    }

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
    }
  }
}

template <int M, int N>
static void fGgmlGemmQ6KNxMQ8KGenericImpl(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                          const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cBlocklen = M;
  constexpr int cNcolsInterleaved = N;
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cBlocksPerHalf = 64 / cBlocklen;
  const int cQ8HalfStride = 512;
  const int cQ8LowHighStep = 256;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);

  float vSumf[4][8];

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q6_Kx8 *vBPtr = (const block_q6_Kx8 *)pVx + (vX * cNb);

      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0f;
        }
      }

      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          const int cBaseL = (vK / cBlocksPerHalf) * 128 + (vK % cBlocksPerHalf) * cBlocklen;
          const int cBaseH = cBaseL + 64;

          const int cScaleIdxL = cBaseL / 16;
          const int cScaleIdxH = cBaseH / 16;

          const int cQhShiftL = ((cBaseL % 128) / 32) * 2;
          const int cQhShiftH = ((cBaseH % 128) / 32) * 2;

          const int cQhHalfL = (cBaseL / 128) * 32;
          const int cQhHalfH = (cBaseH / 128) * 32;

          const int cQ8Base = (vK / cBlocksPerHalf) * cQ8HalfStride + (vK % cBlocksPerHalf) * (cBlocklen * 4);

          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              const int8_t cScaleL = vBPtr[vL].scales[cScaleIdxL * cNcolsInterleaved + vJ];
              const int8_t cScaleH = vBPtr[vL].scales[cScaleIdxH * cNcolsInterleaved + vJ];

              int vSumiL = 0;
              int vSumiH = 0;

              for (int vI = 0; vI < cBlocklen; vI++) {
                const int cQlPos = vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI;
                const int cL4 = vBPtr[vL].ql[cQlPos] & 0xF;
                const int cHi4 = (vBPtr[vL].ql[cQlPos] >> 4) & 0xF;

                const int cQhIdxL = cQhHalfL + ((cBaseL + vI) % 32);
                const int cQhChunkL = cQhIdxL / cBlocklen;
                const int cQhPosL = cQhIdxL % cBlocklen;
                const int cQhOffsetL = cQhChunkL * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPosL;
                const int cHi2L = (vBPtr[vL].qh[cQhOffsetL] >> cQhShiftL) & 0x3;

                const int cQhIdxH = cQhHalfH + ((cBaseH + vI) % 32);
                const int cQhChunkH = cQhIdxH / cBlocklen;
                const int cQhPosH = cQhIdxH % cBlocklen;
                const int cQhOffsetH = cQhChunkH * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPosH;
                const int cHi2H = (vBPtr[vL].qh[cQhOffsetH] >> cQhShiftH) & 0x3;

                const int cQL = ((cHi2L << 4) | cL4) - 32;
                const int cQH = ((cHi2H << 4) | cHi4) - 32;

                const int8_t cQ8L = vAPtr[vL].qs[cQ8Base + vM * cBlocklen + vI];
                const int8_t cQ8H = vAPtr[vL].qs[cQ8Base + vM * cBlocklen + vI + cQ8LowHighStep];

                vSumiL += cQL * cQ8L;
                vSumiH += cQH * cQ8H;
              }

              vSumf[vM][vJ] +=
                  (vSumiL * cScaleL + vSumiH * cScaleH) * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
      }

      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
        }
      }
    }
  }
}

template <int M, int N>
static void fGgmlGemvQ5KNxMQ8KGenericImpl(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                          const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cBlocklen = M;
  constexpr int cNcolsInterleaved = N;
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[cNcolsInterleaved];
  float vSumMinf[cNcolsInterleaved];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q5_Kx8 *vBPtr = (const block_q5_Kx8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
      vSumMinf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vSb = 0; vSb < 8; vSb++) {
        memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * K_SCALE_SIZE, K_SCALE_SIZE);
        vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
        const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
        vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
        vUtmp[vSb * 4 + 2] = cUaux0;
        vUtmp[vSb * 4 + 0] &= cKmask1;
      }
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        constexpr int cScaleStride = 32;
        uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / (32 / cBlocklen)) * cScaleStride;
        uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / (32 / cBlocklen)) * cScaleStride + 16;

        const int cQhShift = (vK / (32 / cBlocklen)) * 2;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi1 = 0;
          vSumi2 = 0;
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cBQsOffset = vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI;

            const int cQhIdx = (vK * cBlocklen + vI) % 32;
            const int cQhChunk = cQhIdx / cBlocklen;
            const int cQhPos = cQhIdx % cBlocklen;
            const int cBQhOffset = cQhChunk * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPos;

            const uint8_t cQhVal = vBPtr[vL].qh[cBQhOffset];
            const uint8_t cH0 = (cQhVal >> cQhShift) & 1;
            const uint8_t cH1 = (cQhVal >> (cQhShift + 1)) & 1;

            const int cV0 = (int8_t)((vBPtr[vL].qs[cBQsOffset] & 0xF) | (cH0 << 4));
            const int cV1 = (int8_t)((vBPtr[vL].qs[cBQsOffset] >> 4) | (cH1 << 4));

            const int cQ8Offset = (vK / (32 / cBlocklen)) * 64 + (vK % (32 / cBlocklen)) * cBlocklen + vI;

            vSumi1 = (cV0 * vAPtr[vL].qs[cQ8Offset]);
            vSumi2 = (cV1 * vAPtr[vL].qs[cQ8Offset + 32]);
            vSumi1 = vSumi1 * vScales0[vJ];
            vSumi2 = vSumi2 * vScales1[vJ];
            vSumi += vSumi1 + vSumi2;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d;
        }
      }
      for (int vSb = 0; vSb < 8; vSb++) {
        uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumMinf[vJ] += vMins[vJ] * (vAPtr[vL].bsums[vSb * 2] + vAPtr[vL].bsums[vSb * 2 + 1]) *
                          GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d;
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ] - vSumMinf[vJ];
    }
  }
}

template <int M, int N>
static void fGgmlGemmQ5KNxMQ8KGenericImpl(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                          const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cBlocklen = M;
  constexpr int cNcolsInterleaved = N;
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][cNcolsInterleaved];
  float vSumMinf[4][cNcolsInterleaved];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q5_Kx8 *vBPtr = (const block_q5_Kx8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
          vSumMinf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vSb = 0; vSb < 8; vSb++) {
          memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * K_SCALE_SIZE, K_SCALE_SIZE);
          vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
          vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
          vUtmp[vSb * 4 + 2] = cUaux0;
          vUtmp[vSb * 4 + 0] &= cKmask1;
        }
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          constexpr int cScaleStride = 32;
          uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / (32 / cBlocklen)) * cScaleStride;
          uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / (32 / cBlocklen)) * cScaleStride + 16;

          const int cQhShift = (vK / (32 / cBlocklen)) * 2;
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi1 = 0;
              vSumi2 = 0;
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cBQsOffset = vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI;

                const int cQhIdx = (vK * cBlocklen + vI) % 32;
                const int cQhChunk = cQhIdx / cBlocklen;
                const int cQhPos = cQhIdx % cBlocklen;
                const int cBQhOffset = cQhChunk * (cBlocklen * cNcolsInterleaved) + vJ * cBlocklen + cQhPos;

                const uint8_t cQhVal = vBPtr[vL].qh[cBQhOffset];
                const uint8_t cH0 = (cQhVal >> cQhShift) & 1;
                const uint8_t cH1 = (cQhVal >> (cQhShift + 1)) & 1;

                const int cV0 = (int8_t)((vBPtr[vL].qs[cBQsOffset] & 0xF) | (cH0 << 4));
                const int cV1 = (int8_t)((vBPtr[vL].qs[cBQsOffset] >> 4) | (cH1 << 4));

                const int cQ8Offset =
                    (vK / (32 / cBlocklen)) * 256 + (vK % (32 / cBlocklen)) * 4 * cBlocklen + vM * cBlocklen + vI;

                vSumi1 = (cV0 * vAPtr[vL].qs[cQ8Offset]);
                vSumi2 = (cV1 * vAPtr[vL].qs[cQ8Offset + 128]);
                vSumi1 = vSumi1 * vScales0[vJ];
                vSumi2 = vSumi2 * vScales1[vJ];
                vSumi += vSumi1 + vSumi2;
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
        for (int vSb = 0; vSb < 8; vSb++) {
          uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
          for (int vM = 0; vM < 4; vM++) {
            const int16_t *vBsums = vAPtr[vL].bsums + (vSb * 8) + (vM * 4) - ((vSb % 2) * 6);
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumMinf[vM][vJ] +=
                  vMins[vJ] * (vBsums[0] + vBsums[1]) * GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ] - vSumMinf[vM][vJ];
        }
      }
    }
  }
}

extern "C" {

void ggml_gemv_q4_0_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
            const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
            vSumi +=
                ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2])) >> 4;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_q4_0_4x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
            const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
            vSumi +=
                ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2])) >> 4;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_q4_0_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[8];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_0x8 *vBPtr = (const block_q4_0x8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
            const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
            vSumi +=
                ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2])) >> 4;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_q4_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 4;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[8];
  float vSumMinf[8];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_Kx8 *vBPtr = (const block_q4_Kx8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
      vSumMinf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vSb = 0; vSb < 8; vSb++) {
        memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * 12, 12);
        vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
        const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
        vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
        vUtmp[vSb * 4 + 2] = cUaux0;
        vUtmp[vSb * 4 + 0] &= cKmask1;
      }
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / 8) * 32;
        uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / 8) * 32 + 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi1 = 0;
          vSumi2 = 0;
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF);
            const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4);
            vSumi1 = (cV0 * vAPtr[vL].qs[(vK / 8) * 64 + (vK % 8) * cBlocklen + vI]);
            vSumi2 = (cV1 * vAPtr[vL].qs[(vK / 8) * 64 + (vK % 8) * cBlocklen + vI + 32]);
            vSumi1 = vSumi1 * vScales0[vJ];
            vSumi2 = vSumi2 * vScales1[vJ];
            vSumi += vSumi1 + vSumi2;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d;
        }
      }
      for (int vSb = 0; vSb < 8; vSb++) {
        uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumMinf[vJ] += vMins[vJ] * (vAPtr[vL].bsums[vSb * 2] + vAPtr[vL].bsums[vSb * 2 + 1]) *
                          GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d;
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ] - vSumMinf[vJ];
    }
  }
}

void ggml_gemv_q4_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[8];
  float vSumMinf[8];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_Kx8 *vBPtr = (const block_q4_Kx8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
      vSumMinf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vSb = 0; vSb < 8; vSb++) {
        memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * 12, 12);
        vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
        const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
        vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
        vUtmp[vSb * 4 + 2] = cUaux0;
        vUtmp[vSb * 4 + 0] &= cKmask1;
      }
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / 4) * 32;
        uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / 4) * 32 + 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi1 = 0;
          vSumi2 = 0;
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF);
            const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4);
            vSumi1 = (cV0 * vAPtr[vL].qs[(vK >> 2) * 64 + (vK % 4) * cBlocklen + vI]);
            vSumi2 = (cV1 * vAPtr[vL].qs[(vK >> 2) * 64 + (vK % 4) * cBlocklen + vI + 32]);
            vSumi1 = vSumi1 * vScales0[vJ];
            vSumi2 = vSumi2 * vScales1[vJ];
            vSumi += vSumi1 + vSumi2;
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d;
        }
      }
      for (int vSb = 0; vSb < 8; vSb++) {
        uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumMinf[vJ] += vMins[vJ] * (vAPtr[vL].bsums[vSb * 2] + vAPtr[vL].bsums[vSb * 2 + 1]) *
                          GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d;
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ] - vSumMinf[vJ];
    }
  }
}

void ggml_gemv_q2_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[8];
  float vSumMinf[8];
  int vSumi1, vSumi2, vSumi3, vSumi4;
  int vSumi;

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q2_Kx8 *vBPtr = (const block_q2_Kx8 *)pVx + (vX * cNb);
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
      vSumMinf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (4 * cBlocklen)); vK++) {
        const uint8_t *vScales0 = vBPtr[vL].scales + (vK / 4) * 64;
        const uint8_t *vScales1 = vBPtr[vL].scales + (vK / 4) * 64 + 16;
        const uint8_t *vScales2 = vBPtr[vL].scales + (vK / 4) * 64 + 32;
        const uint8_t *vScales3 = vBPtr[vL].scales + (vK / 4) * 64 + 48;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi1 = 0;
          vSumi2 = 0;
          vSumi3 = 0;
          vSumi4 = 0;
          vSumi = 0;
          int vOffset = ((vK / 2) % 2) + vJ * 2;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 3);
            const int cV1 = (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 2) & 3);
            const int cV2 = (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4) & 3);
            const int cV3 = (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 6) & 3);
            vSumi1 = (cV0 * vAPtr[vL].qs[(vK >> 2) * 128 + (vK % 4) * cBlocklen + vI]);
            vSumi2 = (cV1 * vAPtr[vL].qs[(vK >> 2) * 128 + (vK % 4) * cBlocklen + vI + 32]);
            vSumi3 = (cV2 * vAPtr[vL].qs[(vK >> 2) * 128 + (vK % 4) * cBlocklen + vI + 64]);
            vSumi4 = (cV3 * vAPtr[vL].qs[(vK >> 2) * 128 + (vK % 4) * cBlocklen + vI + 96]);

            vSumi1 = vSumi1 * (vScales0[vOffset] & 0xF);
            vSumi2 = vSumi2 * (vScales1[vOffset] & 0xF);
            vSumi3 = vSumi3 * (vScales2[vOffset] & 0xF);
            vSumi4 = vSumi4 * (vScales3[vOffset] & 0xF);
            vSumi += vSumi1 + vSumi2 + vSumi3 + vSumi4;
          }
          vSumf[vJ] += vSumi * GGML_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d;
        }
      }
      for (int vSb = 0; vSb < 8; vSb++) {
        const uint8_t *vMins = vBPtr[vL].scales + vSb * 16;
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumMinf[vJ] += ((vMins[vJ * 2] >> 4) * vAPtr[vL].bsums[vSb * 2] +
                           (vMins[(vJ * 2) + 1] >> 4) * vAPtr[vL].bsums[vSb * 2 + 1]) *
                          GGML_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d;
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ] - vSumMinf[vJ];
    }
  }
}

void ggml_gemv_q5_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemvQ5KNxMQ8KGenericImpl<4, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q5_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemvQ5KNxMQ8KGenericImpl<8, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q6_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemvQ6KNxMQ8KGenericImpl<4, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q6_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemvQ6KNxMQ8KGenericImpl<8, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_iq4_nl_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_iq4_nlx4 *vBPtr = (const block_iq4_nlx4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 =
                kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
            const int cV1 = kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
            vSumi += ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2]));
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_iq4_nl_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[8];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_iq4_nlx8 *vBPtr = (const block_iq4_nlx8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 =
                kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
            const int cV1 = kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
            vSumi += ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2]));
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_mxfp4_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_mxfp4x4 *vBPtr = (const block_mxfp4x4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 =
                kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
            const int cV1 = kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
            vSumi += ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2]));
          }
          vSumf[vJ] += vSumi * GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_mxfp4_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[8];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_mxfp4x8 *vBPtr = (const block_mxfp4x8 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      vSumf[vJ] = 0.0;
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 =
                kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
            const int cV1 = kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
            vSumi += ((cV0 * vAPtr[vL].qs[vK * cBlocklen + vI]) + (cV1 * vAPtr[vL].qs[vK * cBlocklen + vI + cQk / 2]));
          }
          vSumf[vJ] += vSumi * GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
  }
}

void ggml_gemv_q8_0_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / cBlocklen); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI];
            vSumi += cV0 * vAPtr[vL].qs[vK * cBlocklen + vI];
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
    }
  }
}

void ggml_gemv_q8_0_4x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pNr == 1);
  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);
  UNUSED(pNr);

  float vSumf[4];
  int vSumi;

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx + (vX * cNb);

    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      vSumf[vJ] = 0.0;
    }
    for (int vL = 0; vL < cNb; vL++) {
      for (int vK = 0; vK < (cQk / cBlocklen); vK++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumi = 0;
          for (int vI = 0; vI < cBlocklen; ++vI) {
            const int cV0 = vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI];
            vSumi += cV0 * vAPtr[vL].qs[vK * cBlocklen + vI];
          }
          vSumf[vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        }
      }
    }
    for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
      pS[vX * cNcolsInterleaved + vJ] = vSumf[vJ];
    }
  }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);

  float sumf[16];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x16 *b_ptr = (const block_q4_0x16 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
            const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
          }
          sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
}

void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;
  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);
  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);
  float sumf[16];
  float sum_minf[16];
  uint8_t scales[128];
  uint8_t mins[128];
  int sumi1;
  int sumi2;
  int sumi;
  const block_q8_K *a_ptr = (const block_q8_K *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_Kx16 *b_ptr = (const block_q4_Kx16 *)vx + (x * nb);
    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0f;
      sum_minf[j] = 0.0f;
    }
    for (int l = 0; l < nb; l++) {
      for (int i = 0; i < 128; i++) {
        scales[i] = b_ptr[l].scales[i] & 0x0F;
        mins[i] = b_ptr[l].scales[i] >> 4;
      }
      for (int i = 0; i < 64; i++) {
        scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
        mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
        scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
        mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
      }
      for (int sb = 0; sb < 8; sb++) {
        uint8_t *min = &mins[sb * 16];
        for (int j = 0; j < ncols_interleaved; j++) {
          sum_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                         GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
        }
      }
      for (int sb = 0; sb < 8; sb += 2) {
        uint8_t *scales_0 = &scales[sb * 16];
        uint8_t *scales_1 = &scales[(sb + 1) * 16];
        for (int i = 0; i < QK4_0; i++) {
          for (int j = 0; j < ncols_interleaved; j++) {
            sumi1 = 0;
            sumi2 = 0;
            sumi = 0;
            const int v0 = (int8_t)(b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
            const int v1 = (int8_t)(b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
            sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
            sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
            sumi1 = sumi1 * scales_0[j];
            sumi2 = sumi2 * scales_1[j];
            sumi += sumi1 + sumi2;
            sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
          }
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
    }
  }
}

void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                        const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(nr == 1);
  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(bs);
  UNUSED(nr);

  float sumf[16];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_iq4_nlx16 *b_ptr = (const block_iq4_nlx16 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
            const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
          }
          sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
}

void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(nr == 1);
  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(bs);
  UNUSED(nr);

  float sumf[16];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q8_0x16 *b_ptr = (const block_q8_0x16 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0;
    }
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / blocklen); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
            sumi += v0 * a_ptr[l].qs[k * blocklen + i];
          }
          sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j];
    }
  }
}

void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  assert(n % QK_K == 0);
  assert(nr == 1);
  assert(nc % 16 == 0);

  UNUSED(bs);
  UNUSED(nr);

  const int nb = n / QK_K;
  const block_q2_Kx16 *x = (const block_q2_Kx16 *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;

  // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
  const int sb_perm[16] = {
      0, 4,  1, 5,  2,  6,  3,  7, // 0-7
      8, 12, 9, 13, 10, 14, 11, 15 // 8-15
  };

  for (int col_tile = 0; col_tile < nc; col_tile += 16) {
    const block_q2_Kx16 *x_ptr = x + (col_tile / 16) * nb;
    const block_q8_K *y_ptr = y;

    float sumf[16] = {0};

    // Loop over K-blocks
    for (int k_block = 0; k_block < nb; ++k_block) {
      int32_t isum[16] = {0};
      int32_t summs[16] = {0};

      const uint8_t *qs_rhs = x_ptr[k_block].qs;
      const uint8_t *sc_rhs = x_ptr[k_block].scales;
      const int8_t *qs_lhs = y_ptr[k_block].qs;
      const int16_t *bs_lhs = y_ptr[k_block].bsums;

      // Iterate over sub-blocks 0..15
      for (int sb = 0; sb < 16; ++sb) {
        // Correction Term
        int16_t bsum = bs_lhs[sb];
        int scale_offset = sb_perm[sb] * 16;

        for (int col = 0; col < 16; ++col) {
          uint8_t sc_val = sc_rhs[scale_offset + col];
          summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
        }

        // Main Dot Product
        // Calculate base offsets for Q2 unpacking based on SB
        int byte_base;
        if (sb < 8)
          byte_base = (sb % 2 == 0) ? 0 : 16;
        else
          byte_base = (sb % 2 == 0) ? 32 : 48;

        int shift = ((sb / 2) % 4) * 2;

        for (int col = 0; col < 16; ++col) {
          uint8_t sc_val = sc_rhs[scale_offset + col];
          int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

          // Process 16 elements (l=0..15)
          for (int l = 0; l < 16; ++l) {
            // Q2: Interleaved by column. Byte `l` contains 4 k-values.
            int qs_idx = (byte_base + l) * 16 + col;
            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

            // Q8: Linear access
            int k = sb * 16 + l;
            int8_t q8_val = qs_lhs[k];

            isum[col] += q8_val * q2_val * d_sb;
          }
        }
      }

      // Finalize K-Block
      for (int col = 0; col < 16; ++col) {
        float d_lhs = y_ptr[k_block].d;
        float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
        float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

        float d_all = d_lhs * d_rhs;
        float d_min = d_lhs * dm_rhs;

        sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
      }
    }

    for (int col = 0; col < 16; ++col) {
      s[col_tile + col] = sumf[col];
    }
  }
}
#endif

void ggml_gemm_q4_0_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  {
    float vSumf[4][4];
    int vSumi;

    for (int vY = 0; vY < pNr / 4; vY++) {
      const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
      for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
        const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx + (vX * cNb);
        for (int vM = 0; vM < 4; vM++) {
          for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
            vSumf[vM][vJ] = 0.0;
        }
        for (int vL = 0; vL < cNb; vL++) {
          for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
            for (int vM = 0; vM < 4; vM++) {
              for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
                vSumi = 0;
                for (int vI = 0; vI < cBlocklen; ++vI) {
                  const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
                  const int cV1 =
                      (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
                  vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                            (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4])) >>
                           4;
                }
                vSumf[vM][vJ] +=
                    vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
              }
            }
          }
        }
        for (int vM = 0; vM < 4; vM++) {
          for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
            pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_q4_0_4x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4][4];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          vSumf[vM][vJ] = 0.0;
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
                const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
                vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                          (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4])) >>
                         4;
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
      }
    }
  }
}

void ggml_gemm_q4_0_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4][8];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_0x8 *vBPtr = (const block_q4_0x8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          vSumf[vM][vJ] = 0.0;
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] << 4);
                const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF0);
                vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                          (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4])) >>
                         4;
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
      }
    }
  }
}

void ggml_gemm_q4_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 4;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4][8];
  float vSumMinf[4][8];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_Kx8 *vBPtr = (const block_q4_Kx8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
          vSumMinf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vSb = 0; vSb < 8; vSb++) {
          memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * 12, 12);
          vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
          vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
          vUtmp[vSb * 4 + 2] = cUaux0;
          vUtmp[vSb * 4 + 0] &= cKmask1;
        }
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / 8) * 32;
          uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / 8) * 32 + 16;
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi1 = 0;
              vSumi2 = 0;
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF);
                const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4);
                vSumi1 = (cV0 * vAPtr[vL].qs[(vK / 8) * 256 + (vK % 8) * 4 * cBlocklen + vM * cBlocklen + vI]);
                vSumi2 = (cV1 * vAPtr[vL].qs[(vK / 8) * 256 + (vK % 8) * 4 * cBlocklen + vM * cBlocklen + vI + 128]);
                vSumi1 = vSumi1 * vScales0[vJ];
                vSumi2 = vSumi2 * vScales1[vJ];
                vSumi += vSumi1 + vSumi2;
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
        for (int vSb = 0; vSb < 8; vSb++) {
          uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
          for (int vM = 0; vM < 4; vM++) {
            const int16_t *vBsums = vAPtr[vL].bsums + (vSb * 8) + (vM * 4) - ((vSb % 2) * 6);
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumMinf[vM][vJ] +=
                  vMins[vJ] * (vBsums[0] + vBsums[1]) * GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ] - vSumMinf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_q4_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;
  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pBs);

  float vSumf[4][8];
  float vSumMinf[4][8];
  uint32_t vUtmp[32];
  int vSumi1;
  int vSumi2;
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_Kx8 *vBPtr = (const block_q4_Kx8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
          vSumMinf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vSb = 0; vSb < 8; vSb++) {
          memcpy(vUtmp + vSb * 4, vBPtr[vL].scales + vSb * 12, 12);
          vUtmp[vSb * 4 + 3] = ((vUtmp[vSb * 4 + 2] >> 4) & cKmask2) | (((vUtmp[vSb * 4 + 1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = vUtmp[vSb * 4 + 1] & cKmask1;
          vUtmp[vSb * 4 + 1] = (vUtmp[vSb * 4 + 2] & cKmask2) | (((vUtmp[vSb * 4 + 0] >> 6) & cKmask3) << 4);
          vUtmp[vSb * 4 + 2] = cUaux0;
          vUtmp[vSb * 4 + 0] &= cKmask1;
        }
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          uint8_t *vScales0 = (uint8_t *)vUtmp + (vK / 4) * 32;
          uint8_t *vScales1 = (uint8_t *)vUtmp + (vK / 4) * 32 + 16;
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi1 = 0;
              vSumi2 = 0;
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0xF);
                const int cV1 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4);
                vSumi1 = (cV0 * vAPtr[vL].qs[(vK >> 2) * 256 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI]);
                vSumi2 = (cV1 * vAPtr[vL].qs[(vK >> 2) * 256 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI + 128]);
                vSumi1 = vSumi1 * vScales0[vJ];
                vSumi2 = vSumi2 * vScales1[vJ];
                vSumi += vSumi1 + vSumi2;
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
        for (int vSb = 0; vSb < 8; vSb++) {
          uint8_t *vMins = (uint8_t *)vUtmp + 8 + vSb * 16;
          for (int vM = 0; vM < 4; vM++) {
            const int16_t *vBsums = vAPtr[vL].bsums + (vSb * 8) + (vM * 4) - ((vSb % 2) * 6);
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumMinf[vM][vJ] +=
                  vMins[vJ] * (vBsums[0] + vBsums[1]) * GGML_CPU_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ] - vSumMinf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_q2_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  float vSumf[4][8];
  float vSumMinf[4][8];
  int vSumi1, vSumi2, vSumi3, vSumi4;
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q2_Kx8 *vBPtr = (const block_q2_Kx8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
          vSumMinf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (4 * cBlocklen)); vK++) {

          const uint8_t *vScales0 = vBPtr[vL].scales + (vK / 4) * 64;
          const uint8_t *vScales1 = vBPtr[vL].scales + (vK / 4) * 64 + 16;
          const uint8_t *vScales2 = vBPtr[vL].scales + (vK / 4) * 64 + 32;
          const uint8_t *vScales3 = vBPtr[vL].scales + (vK / 4) * 64 + 48;
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi1 = 0;
              vSumi2 = 0;
              vSumi3 = 0;
              vSumi4 = 0;
              vSumi = 0;
              int vOffset = ((vK / 2) % 2) + vJ * 2;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = (int8_t)(vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 3);
                const int cV1 =
                    (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 2) & 3);
                const int cV2 =
                    (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4) & 3);
                const int cV3 =
                    (int8_t)((vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 6) & 3);
                vSumi1 = (cV0 * vAPtr[vL].qs[(vK >> 2) * 512 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI]);
                vSumi2 = (cV1 * vAPtr[vL].qs[(vK >> 2) * 512 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI + 128]);
                vSumi3 = (cV2 * vAPtr[vL].qs[(vK >> 2) * 512 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI + 256]);
                vSumi4 = (cV3 * vAPtr[vL].qs[(vK >> 2) * 512 + (vK % 4) * 4 * cBlocklen + vM * cBlocklen + vI + 384]);
                vSumi1 = vSumi1 * (vScales0[vOffset] & 0xF);
                vSumi2 = vSumi2 * (vScales1[vOffset] & 0xF);
                vSumi3 = vSumi3 * (vScales2[vOffset] & 0xF);
                vSumi4 = vSumi4 * (vScales3[vOffset] & 0xF);
                vSumi += vSumi1 + vSumi2 + vSumi3 + vSumi4;
              }
              vSumf[vM][vJ] += vSumi * GGML_FP16_TO_FP32(vBPtr[vL].d[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
        for (int vSb = 0; vSb < 8; vSb++) {
          const uint8_t *vMins = vBPtr[vL].scales + vSb * 16;
          for (int vM = 0; vM < 4; vM++) {
            const int16_t *vBsums = vAPtr[vL].bsums + (vSb * 8) + (vM * 4) - ((vSb % 2) * 6);
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              int vMinsProd = ((vMins[vJ * 2] >> 4) * vBsums[0] + (vMins[(vJ * 2) + 1] >> 4) * vBsums[1]);
              vSumMinf[vM][vJ] += (vMinsProd)*GGML_FP16_TO_FP32(vBPtr[vL].dmin[vJ]) * vAPtr[vL].d[vM];
            }
          }
        }
      }

      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ] - vSumMinf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_q5_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemmQ5KNxMQ8KGenericImpl<4, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q5_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemmQ5KNxMQ8KGenericImpl<8, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q6_K_8x4_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemmQ6KNxMQ8KGenericImpl<4, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q6_K_8x8_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  fGgmlGemmQ6KNxMQ8KGenericImpl<8, 8>(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_iq4_nl_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

  {
    float vSumf[4][4];
    int vSumi;

    for (int vY = 0; vY < pNr / 4; vY++) {
      const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
      for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
        const block_iq4_nlx4 *vBPtr = (const block_iq4_nlx4 *)pVx + (vX * cNb);
        for (int vM = 0; vM < 4; vM++) {
          for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
            vSumf[vM][vJ] = 0.0;
        }
        for (int vL = 0; vL < cNb; vL++) {
          for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
            for (int vM = 0; vM < 4; vM++) {
              for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
                vSumi = 0;
                for (int vI = 0; vI < cBlocklen; ++vI) {
                  const int cV0 =
                      kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
                  const int cV1 =
                      kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
                  vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                            (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4]));
                }
                vSumf[vM][vJ] +=
                    vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
              }
            }
          }
        }
        for (int vM = 0; vM < 4; vM++) {
          for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
            pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_iq4_nl_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][8];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_iq4_nlx8 *vBPtr = (const block_iq4_nlx8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          vSumf[vM][vJ] = 0.0;
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 =
                    kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
                const int cV1 =
                    kvalues_iq4nl[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
                vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                          (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4]));
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
      }
    }
  }
}

void ggml_gemm_mxfp4_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][4];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_mxfp4x4 *vBPtr = (const block_mxfp4x4 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          vSumf[vM][vJ] = 0.0;
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 =
                    kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
                const int cV1 =
                    kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
                vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                          (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4]));
              }
              vSumf[vM][vJ] +=
                  vSumi * GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
      }
    }
  }
}

void ggml_gemm_mxfp4_8x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][8];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_mxfp4x8 *vBPtr = (const block_mxfp4x8 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          vSumf[vM][vJ] = 0.0;
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / (2 * cBlocklen)); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 =
                    kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] & 0x0F];
                const int cV1 =
                    kvalues_mxfp4[vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI] >> 4];
                vSumi += ((cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI]) +
                          (cV1 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI + cQk / 2 * 4]));
              }
              vSumf[vM][vJ] +=
                  vSumi * GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++)
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
      }
    }
  }
}

void ggml_gemm_q8_0_4x4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][4];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / cBlocklen); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI];
                vSumi += cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI];
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
        }
      }
    }
  }
}

void ggml_gemm_q8_0_4x8_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  float vSumf[4][4];
  int vSumi;

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx + (vX * cNb);
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          vSumf[vM][vJ] = 0.0;
        }
      }
      for (int vL = 0; vL < cNb; vL++) {
        for (int vK = 0; vK < (cQk / cBlocklen); vK++) {
          for (int vM = 0; vM < 4; vM++) {
            for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
              vSumi = 0;
              for (int vI = 0; vI < cBlocklen; ++vI) {
                const int cV0 = vBPtr[vL].qs[vK * cNcolsInterleaved * cBlocklen + vJ * cBlocklen + vI];
                vSumi += cV0 * vAPtr[vL].qs[vK * 4 * cBlocklen + vM * cBlocklen + vI];
              }
              vSumf[vM][vJ] += vSumi * GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[vJ]) * GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[vM]);
            }
          }
        }
      }
      for (int vM = 0; vM < 4; vM++) {
        for (int vJ = 0; vJ < cNcolsInterleaved; vJ++) {
          pS[(vY * 4 + vM) * pBs + vX * cNcolsInterleaved + vJ] = vSumf[vM][vJ];
        }
      }
    }
  }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);

  float sumf[4][16];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x16 *b_ptr = (const block_q4_0x16 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >>
                        4;
              }
              sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
}

void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);

  float sumf[4][16];
  float sum_minf[4][16];
  uint8_t scales[128];
  uint8_t mins[128];
  int sumi1;
  int sumi2;
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_Kx4 *a_ptr = (const block_q8_Kx4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_Kx16 *b_ptr = (const block_q4_Kx16 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
          sum_minf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int i = 0; i < 128; i++) {
          scales[i] = b_ptr[l].scales[i] & 0x0F;
          mins[i] = b_ptr[l].scales[i] >> 4;
        }
        for (int i = 0; i < 64; i++) {
          scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
          mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
          scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
          mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
        }

        for (int sb = 0; sb < 8; sb++) {
          uint8_t *min = &mins[sb * 16];
          for (int m = 0; m < 4; m++) {
            const int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
            for (int j = 0; j < ncols_interleaved; j++) {
              sum_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
            }
          }
        }

        for (int sb = 0; sb < 8; sb += 2) {
          uint8_t *scales_0 = &scales[sb * 16];
          uint8_t *scales_1 = &scales[(sb + 1) * 16];

          for (int i = 0; i < QK4_0; i++) {
            for (int m = 0; m < 4; m++) {
              for (int j = 0; j < ncols_interleaved; j++) {
                sumi1 = 0;
                sumi2 = 0;
                sumi = 0;

                const int v0 = (int8_t)(b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                const int v1 = (int8_t)(b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                sumi1 = sumi1 * scales_0[j];
                sumi2 = sumi2 * scales_1[j];
                sumi += sumi1 + sumi2;

                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
              }
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
        }
      }
    }
  }
}

void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                        const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][16];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_iq4_nlx16 *b_ptr = (const block_iq4_nlx16 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
              }
              sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
}

void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 16;
  const int blocklen = 1;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][16];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q8_0x16 *b_ptr = (const block_q8_0x16 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / blocklen); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
              }
              sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
        }
      }
    }
  }
}

void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                      const void *GGML_RESTRICT vy, int nr, int nc) {
  assert(n % QK_K == 0);
  assert(nr % 4 == 0);
  assert(nc % 16 == 0);
  const int nb = n / QK_K;
  const block_q2_Kx16 *x = (const block_q2_Kx16 *)vx;
  const block_q8_Kx4 *y = (const block_q8_Kx4 *)vy;

  const int sb_perm[16] = {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};

  // Iterate Rows in tiles of 4
  for (int row_tile = 0; row_tile < nr; row_tile += 4) {
    // Iterate Columns in tiles of 16
    for (int col_tile = 0; col_tile < nc; col_tile += 16) {

      const block_q2_Kx16 *x_ptr = x + (col_tile / 16) * nb;
      const block_q8_Kx4 *y_ptr = y + (row_tile / 4) * nb;

      float sumf[4][16];
      memset(sumf, 0, sizeof(sumf));

      for (int k_block = 0; k_block < nb; ++k_block) {
        int32_t isum[4][16];
        int32_t summs[4][16];
        memset(isum, 0, sizeof(isum));
        memset(summs, 0, sizeof(summs));

        const uint8_t *qs_rhs = x_ptr[k_block].qs;
        const uint8_t *sc_rhs = x_ptr[k_block].scales;
        const int8_t *qs_lhs = y_ptr[k_block].qs;
        const int16_t *bs_lhs = y_ptr[k_block].bsums;

        for (int sb = 0; sb < 16; ++sb) {
          int scale_offset = sb_perm[sb] * 16;

          int byte_base;
          if (sb < 8)
            byte_base = (sb % 2 == 0) ? 0 : 16;
          else
            byte_base = (sb % 2 == 0) ? 32 : 48;
          int shift = ((sb / 2) % 4) * 2;

          for (int col = 0; col < 16; ++col) {
            uint8_t sc_val = sc_rhs[scale_offset + col];
            int32_t d_sb = sc_val & 0xF;
            int32_t m_sb = sc_val >> 4;

            // Correction Term
            for (int r = 0; r < 4; ++r) {
              int bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
              summs[r][col] += bs_lhs[bsum_idx] * m_sb;
            }

            // Main Dot Product
            for (int l = 0; l < 16; ++l) {
              int qs_idx = (byte_base + l) * 16 + col;
              uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

              // Calculate Q8 index for this specific k and row
              int k = sb * 16 + l;
              int q8_idx = (k / 4) * 16 + (k % 4);

              for (int r = 0; r < 4; ++r) {
                // Add r*4 to jump to the correct row within the 4x4 chunk
                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                isum[r][col] += q8_val * q2_val * d_sb;
              }
            }
          }
        }

        // Finalize K-Block
        for (int col = 0; col < 16; ++col) {
          float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
          float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

          for (int r = 0; r < 4; ++r) {
            float d_lhs = y_ptr[k_block].d[r];
            float d_all = d_lhs * d_rhs;
            float d_min = d_lhs * dm_rhs;
            sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
          }
        }
      }

      for (int r = 0; r < 4; ++r) {
        for (int col = 0; col < 16; ++col) {
          s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
        }
      }
    }
  }
}
#endif

} // extern "C"

static block_q8_0x4 fMakeBlockQ80x4(block_q8_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q8_0x4 vOut;

  for (int vI = 0; vI < 4; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK8_0 * 4 / pBlckSizeInterleave;
  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcId = vI % 4;
    int vSrcOffset = (vI / 4) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;
    memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], pBlckSizeInterleave);
  }
  return vOut;
}

static block_q4_0x4 fMakeBlockQ40x4(block_q4_0 *pIn, int pBlckSizeInterleave) {
  block_q4_0x4 vOut;

  for (int vI = 0; vI < 4; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_0 * 2 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 8) {
    const uint64_t cXorMask = 0x8888888888888888ULL;
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 4;
      int vSrcOffset = (vI / 4) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      uint64_t vElems;
      // Using memcpy to avoid unaligned memory accesses
      memcpy(&vElems, &pIn[vSrcId].qs[vSrcOffset], sizeof(uint64_t));
      vElems ^= cXorMask;
      memcpy(&vOut.qs[vDstOffset], &vElems, sizeof(uint64_t));
    }
  } else if (pBlckSizeInterleave == 4) {
    const uint32_t cXorMask = 0x88888888;
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 4;
      int vSrcOffset = (vI / 4) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      uint32_t vElems;
      memcpy(&vElems, &pIn[vSrcId].qs[vSrcOffset], sizeof(uint32_t));
      vElems ^= cXorMask;
      memcpy(&vOut.qs[vDstOffset], &vElems, sizeof(uint32_t));
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 fMakeBlockQ40x8(block_q4_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_0x8 vOut;

  for (int vI = 0; vI < 8; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_0 * 4 / pBlckSizeInterleave;
  const uint64_t cXorMask = 0x8888888888888888ULL;

  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcId = vI % 8;
    int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    uint64_t vElems;
    memcpy(&vElems, &pIn[vSrcId].qs[vSrcOffset], sizeof(uint64_t));
    vElems ^= cXorMask;
    memcpy(&vOut.qs[vDstOffset], &vElems, sizeof(uint64_t));
  }

  return vOut;
}

static block_q4_0x16 fMakeBlockQ40x16(block_q4_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_0x16 vOut;

  for (int vI = 0; vI < 16; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_0 * 8 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 1) {
    const uint8_t cXorMask = 0x88;
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 16;
      int vSrcOffset = vI / 16;
      int vDstOffset = vI;

      vOut.qs[vDstOffset] = pIn[vSrcId].qs[vSrcOffset] ^ cXorMask;
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static block_q4_Kx8 fMakeBlockQ4Kx8(block_q4_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_Kx8 vOut;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
  for (int vI = 0; vI < 8; vI++) {
    vOut.d[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
  }

  for (int vI = 0; vI < 8; vI++) {
    vOut.dmin[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  const int cEnd = QK_K * 4 / pBlckSizeInterleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcId = vI % 8;
    int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    // buffer large enough for the max interleave block size (8 bytes)
    uint64_t vElems;
    memcpy(&vElems, &pIn[vSrcId].qs[vSrcOffset], pBlckSizeInterleave);
    memcpy(&vOut.qs[vDstOffset], &vElems, pBlckSizeInterleave);
  }

  // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
  // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
  // The output Q4_Kx8 structure has 96 bytes
  // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
  // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
  uint8_t aS[8], vM[8];

  for (int vI = 0; vI < 4; vI++) {
    for (int vJ = 0; vJ < 8; vJ++) {
      aS[vJ] = pIn[vJ].scales[vI] & 63;
      vM[vJ] = pIn[vJ].scales[vI + 4] & 63;
    }

    vOut.scales[vI * 12] = (aS[0] & 63) + ((aS[4] & 48) << 2);
    vOut.scales[vI * 12 + 1] = (aS[1] & 63) + ((aS[5] & 48) << 2);
    vOut.scales[vI * 12 + 2] = (aS[2] & 63) + ((aS[6] & 48) << 2);
    vOut.scales[vI * 12 + 3] = (aS[3] & 63) + ((aS[7] & 48) << 2);
    vOut.scales[vI * 12 + 4] = (vM[0] & 63) + ((vM[4] & 48) << 2);
    vOut.scales[vI * 12 + 5] = (vM[1] & 63) + ((vM[5] & 48) << 2);
    vOut.scales[vI * 12 + 6] = (vM[2] & 63) + ((vM[6] & 48) << 2);
    vOut.scales[vI * 12 + 7] = (vM[3] & 63) + ((vM[7] & 48) << 2);
    vOut.scales[vI * 12 + 8] = (aS[4] & 15) + ((vM[4] & 15) << 4);
    vOut.scales[vI * 12 + 9] = (aS[5] & 15) + ((vM[5] & 15) << 4);
    vOut.scales[vI * 12 + 10] = (aS[6] & 15) + ((vM[6] & 15) << 4);
    vOut.scales[vI * 12 + 11] = (aS[7] & 15) + ((vM[7] & 15) << 4);
  }

  for (int vI = 0; vI < 4; vI++) {
    for (int vJ = 0; vJ < 8; vJ++) {
      aS[vJ] = ((pIn[vJ].scales[vI] & 192) >> 2) | (pIn[vJ].scales[vI + 8] & 15);
      vM[vJ] = ((pIn[vJ].scales[vI + 4] & 192) >> 2) | ((pIn[vJ].scales[vI + 8] & 240) >> 4);
    }

    vOut.scales[vI * 12 + 48] = (aS[0] & 63) + ((aS[4] & 48) << 2);
    vOut.scales[vI * 12 + 49] = (aS[1] & 63) + ((aS[5] & 48) << 2);
    vOut.scales[vI * 12 + 50] = (aS[2] & 63) + ((aS[6] & 48) << 2);
    vOut.scales[vI * 12 + 51] = (aS[3] & 63) + ((aS[7] & 48) << 2);
    vOut.scales[vI * 12 + 52] = (vM[0] & 63) + ((vM[4] & 48) << 2);
    vOut.scales[vI * 12 + 53] = (vM[1] & 63) + ((vM[5] & 48) << 2);
    vOut.scales[vI * 12 + 54] = (vM[2] & 63) + ((vM[6] & 48) << 2);
    vOut.scales[vI * 12 + 55] = (vM[3] & 63) + ((vM[7] & 48) << 2);
    vOut.scales[vI * 12 + 56] = (aS[4] & 15) + ((vM[4] & 15) << 4);
    vOut.scales[vI * 12 + 57] = (aS[5] & 15) + ((vM[5] & 15) << 4);
    vOut.scales[vI * 12 + 58] = (aS[6] & 15) + ((vM[6] & 15) << 4);
    vOut.scales[vI * 12 + 59] = (aS[7] & 15) + ((vM[7] & 15) << 4);
  }

  return vOut;
}

static block_q4_Kx16 fMakeBlockQ4Kx16(block_q4_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_Kx16 vOut;
  // Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
  for (int vI = 0; vI < 16; vI++) {
    vOut.d[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
  }

  for (int vI = 0; vI < 16; vI++) {
    vOut.dmin[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  const int cEnd = QK_K * 8 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 1) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 16;
      int vSrcOffset = vI / 16;
      int vDstOffset = vI;

      vOut.qs[vDstOffset] = pIn[vSrcId].qs[vSrcOffset];
    }

    // RVV repacking.
    //
    // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
    uint8_t aS[128], vM[128];
    for (int vI = 0; vI < 4; vI++) {
      for (int vJ = 0; vJ < 16; vJ++) {
        aS[vI * 16 + vJ] = pIn[vJ].scales[vI] & 63;
        vM[vI * 16 + vJ] = pIn[vJ].scales[vI + 4] & 63;
      }
    }
    for (int vI = 0; vI < 4; vI++) {
      for (int vJ = 0; vJ < 16; vJ++) {
        aS[64 + vI * 16 + vJ] = ((pIn[vJ].scales[vI] & 192) >> 2) | (pIn[vJ].scales[vI + 8] & 15);
        vM[64 + vI * 16 + vJ] = ((pIn[vJ].scales[vI + 4] & 192) >> 2) | ((pIn[vJ].scales[vI + 8] & 240) >> 4);
      }
    }

    for (int vI = 0; vI < 128; vI++) {
      vOut.scales[vI] = (aS[vI] & 15) | ((vM[vI] & 15) << 4);
    }
    for (int vI = 0; vI < 64; vI++) {
      vOut.scales[128 + vI] =
          ((aS[vI] & 48) >> 4) | ((vM[vI] & 48) >> 2) | (aS[64 + vI] & 48) | ((vM[64 + vI] & 48) << 2);
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static block_q2_Kx8 fMakeBlockQ2Kx8(block_q2_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q2_Kx8 vOut;

  // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
  for (int vI = 0; vI < 8; vI++) {
    vOut.d[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
  }

  for (int vI = 0; vI < 8; vI++) {
    vOut.dmin[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  const int cEnd = QK_K * 2 / pBlckSizeInterleave;

  // Interleave Q2_K quants by taking 8 bytes at a time
  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcId = vI % 8;
    int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    uint64_t vElems;
    memcpy(&vElems, &pIn[vSrcId].qs[vSrcOffset], sizeof(uint64_t));
    memcpy(&vOut.qs[vDstOffset], &vElems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
  // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
  // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
  // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
  // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K
  // structures

  for (int vI = 0; vI < 128; vI++) {
    // Index for selecting which q2k super block
    int vSrc1 = (vI % 16) / 2;
    // Index for selecting scale
    int vSrc2 = ((vI / 16) * 2) + (vI % 2);

    vOut.scales[vI] = pIn[vSrc1].scales[vSrc2];
  }
  return vOut;
}

static block_q5_Kx8 fMakeBlockQ5Kx8(block_q5_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q5_Kx8 vOut;
  // Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
  for (int vI = 0; vI < 8; vI++) {
    vOut.d[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
  }

  for (int vI = 0; vI < 8; vI++) {
    vOut.dmin[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  const int cEnd = QK_K * 4 / pBlckSizeInterleave;

  // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcId = vI % 8;
    int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], pBlckSizeInterleave);
  }

  // Repeat for high bits with the same chunk size, since
  // the high bits are interleaved in Q5_K and the index is
  // qh_idx = (qs_idx % 32);
  // qh_val = qh[qh_idx] >> (qs_idx / 32);
  for (int vI = 0; vI < cEnd / 4; ++vI) {
    int vSrcId = vI % 8;
    int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    memcpy(&vOut.qh[vDstOffset], &pIn[vSrcId].qh[vSrcOffset], pBlckSizeInterleave);
  }

  // The below logic is copied over from Q4_K
  // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
  // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
  // The output Q5_Kx8 structure has 96 bytes
  // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
  // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
  uint8_t aS[8], vM[8];

  for (int vI = 0; vI < 4; vI++) {
    for (int vJ = 0; vJ < 8; vJ++) {
      aS[vJ] = pIn[vJ].scales[vI] & 63;
      vM[vJ] = pIn[vJ].scales[vI + 4] & 63;
    }

    vOut.scales[vI * 12] = (aS[0] & 63) + ((aS[4] & 48) << 2);
    vOut.scales[vI * 12 + 1] = (aS[1] & 63) + ((aS[5] & 48) << 2);
    vOut.scales[vI * 12 + 2] = (aS[2] & 63) + ((aS[6] & 48) << 2);
    vOut.scales[vI * 12 + 3] = (aS[3] & 63) + ((aS[7] & 48) << 2);
    vOut.scales[vI * 12 + 4] = (vM[0] & 63) + ((vM[4] & 48) << 2);
    vOut.scales[vI * 12 + 5] = (vM[1] & 63) + ((vM[5] & 48) << 2);
    vOut.scales[vI * 12 + 6] = (vM[2] & 63) + ((vM[6] & 48) << 2);
    vOut.scales[vI * 12 + 7] = (vM[3] & 63) + ((vM[7] & 48) << 2);
    vOut.scales[vI * 12 + 8] = (aS[4] & 15) + ((vM[4] & 15) << 4);
    vOut.scales[vI * 12 + 9] = (aS[5] & 15) + ((vM[5] & 15) << 4);
    vOut.scales[vI * 12 + 10] = (aS[6] & 15) + ((vM[6] & 15) << 4);
    vOut.scales[vI * 12 + 11] = (aS[7] & 15) + ((vM[7] & 15) << 4);
  }

  for (int vI = 0; vI < 4; vI++) {
    for (int vJ = 0; vJ < 8; vJ++) {
      aS[vJ] = ((pIn[vJ].scales[vI] & 192) >> 2) | (pIn[vJ].scales[vI + 8] & 15);
      vM[vJ] = ((pIn[vJ].scales[vI + 4] & 192) >> 2) | ((pIn[vJ].scales[vI + 8] & 240) >> 4);
    }

    vOut.scales[vI * 12 + 48] = (aS[0] & 63) + ((aS[4] & 48) << 2);
    vOut.scales[vI * 12 + 49] = (aS[1] & 63) + ((aS[5] & 48) << 2);
    vOut.scales[vI * 12 + 50] = (aS[2] & 63) + ((aS[6] & 48) << 2);
    vOut.scales[vI * 12 + 51] = (aS[3] & 63) + ((aS[7] & 48) << 2);
    vOut.scales[vI * 12 + 52] = (vM[0] & 63) + ((vM[4] & 48) << 2);
    vOut.scales[vI * 12 + 53] = (vM[1] & 63) + ((vM[5] & 48) << 2);
    vOut.scales[vI * 12 + 54] = (vM[2] & 63) + ((vM[6] & 48) << 2);
    vOut.scales[vI * 12 + 55] = (vM[3] & 63) + ((vM[7] & 48) << 2);
    vOut.scales[vI * 12 + 56] = (aS[4] & 15) + ((vM[4] & 15) << 4);
    vOut.scales[vI * 12 + 57] = (aS[5] & 15) + ((vM[5] & 15) << 4);
    vOut.scales[vI * 12 + 58] = (aS[6] & 15) + ((vM[6] & 15) << 4);
    vOut.scales[vI * 12 + 59] = (aS[7] & 15) + ((vM[7] & 15) << 4);
  }

  return vOut;
}

static block_q6_Kx8 fMakeBlockQ6Kx8(block_q6_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q6_Kx8 vOut;
  constexpr int cNBlocks = 8; // Kx8
  for (int vI = 0; vI < cNBlocks; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEndLs = QK_K * 4 / pBlckSizeInterleave;
  // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
  for (int vI = 0; vI < cEndLs; ++vI) {
    int vSrcId = vI % cNBlocks;
    int vSrcOffset = (vI / cNBlocks) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    uint64_t vElemLs;
    memcpy(&vElemLs, &pIn[vSrcId].ql[vSrcOffset], pBlckSizeInterleave);
    memcpy(&vOut.ql[vDstOffset], &vElemLs, pBlckSizeInterleave);
  }

  // Interleave high bits using same chunk size as low bits
  const int cEndHs = cEndLs / 2;
  for (int vI = 0; vI < cEndHs; ++vI) {
    int vSrcId = vI % cNBlocks;
    int vSrcOffset = (vI / cNBlocks) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;

    uint64_t vElemHs;
    memcpy(&vElemHs, &pIn[vSrcId].qh[vSrcOffset], pBlckSizeInterleave);
    memcpy(&vOut.qh[vDstOffset], &vElemHs, pBlckSizeInterleave);
  }

  // The below logic is designed so as to unpack and rearrange scales in Q6_K
  // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
  // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
  // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
  constexpr int cNScales = QK_K / 16;

  for (int vI = 0; vI < cNBlocks; vI++) {
    for (int vJ = 0; vJ < cNScales; vJ++) {
      vOut.scales[vJ * cNBlocks + vI] = pIn[vI].scales[vJ];
    }
  }

  return vOut;
}

static block_q2_Kx16 fMakeBlockQ2Kx16(const block_q2_K *pIn, unsigned int pBlckSizeInterleave) {
  block_q2_Kx16 vOut;
  constexpr int cNCols = 16;

  // 1. Copy Super-Scales (d) and Super-Mins (dmin)
  for (int vI = 0; vI < cNCols; vI++) {
    vOut.d[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    vOut.dmin[vI] = pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  // 2. Interleave Q2_K Data
  const int cBytesPerCol = 64;
  const int cTotalBytes = cNCols * cBytesPerCol;
  const int cEnd = cTotalBytes / pBlckSizeInterleave;

  for (int vI = 0; vI < cEnd; ++vI) {
    int vSrcColId = vI % cNCols;
    int vSrcOffset = (vI / cNCols) * pBlckSizeInterleave;
    int vDstOffset = vI * pBlckSizeInterleave;
    memcpy(&vOut.qs[vDstOffset], &pIn[vSrcColId].qs[vSrcOffset], pBlckSizeInterleave);
  }

  // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
  int vOutIdx = 0;

  // Arrays define the sub-block order for each group
  const int aEvenLowSbs[] = {0, 2, 4, 6};
  const int aOddLowSbs[] = {1, 3, 5, 7};
  const int aEvenHighSbs[] = {8, 10, 12, 14};
  const int aOddHighSbs[] = {9, 11, 13, 15};

  // Pack Group 1: Even-Low
  for (int vSb : aEvenLowSbs) {
    for (int vCol = 0; vCol < cNCols; vCol++) {
      vOut.scales[vOutIdx++] = pIn[vCol].scales[vSb];
    }
  }

  // Pack Group 2: Odd-Low
  for (int vSb : aOddLowSbs) {
    for (int vCol = 0; vCol < cNCols; vCol++) {
      vOut.scales[vOutIdx++] = pIn[vCol].scales[vSb];
    }
  }

  // Pack Group 3: Even-High
  for (int vSb : aEvenHighSbs) {
    for (int vCol = 0; vCol < cNCols; vCol++) {
      vOut.scales[vOutIdx++] = pIn[vCol].scales[vSb];
    }
  }

  // Pack Group 4: Odd-High
  for (int vSb : aOddHighSbs) {
    for (int vCol = 0; vCol < cNCols; vCol++) {
      vOut.scales[vOutIdx++] = pIn[vCol].scales[vSb];
    }
  }

  return vOut;
}

static int fRepackQ40ToQ404Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 4 || pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 4;

  block_q4_0x4 *vDst = (block_q4_0x4 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 aDstTmp[4];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ40x4(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ4KToQ4K8Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_K);
  GGML_ASSERT(pInterleaveBlock == 8 || pInterleaveBlock == 4);
  constexpr int cNrowsInterleaved = 8;

  block_q4_Kx8 *vDst = (block_q4_Kx8 *)pT->data;
  const block_q4_K *vSrc = (const block_q4_K *)pData;
  block_q4_K aDstTmp[8];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ4Kx8(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ4KToQ4K16Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_K);
  constexpr int cNrowsInterleaved = 16;

  block_q4_Kx16 *vDst = (block_q4_Kx16 *)pT->data;
  const block_q4_K *vSrc = (const block_q4_K *)pData;
  block_q4_K aDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ4Kx16(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ2KToQ2K8Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q2_K);
  GGML_ASSERT(pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 8;

  block_q2_Kx8 *vDst = (block_q2_Kx8 *)pT->data;
  const block_q2_K *vSrc = (const block_q2_K *)pData;
  block_q2_K aDstTmp[8];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q2_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ2Kx8(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ2KToQ2K16Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q2_K);
  constexpr int cNrowsInterleaved = 16;

  block_q2_Kx16 *vDst = (block_q2_Kx16 *)pT->data;
  const block_q2_K *vSrc = (const block_q2_K *)pData;

  block_q2_K aDstTmp[cNrowsInterleaved];

  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q2_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      // This loop gathers 16 separate blocks (one from each column)
      // that correspond to the same K-dimension chunk.
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }

      *vDst++ = fMakeBlockQ2Kx16(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ40ToQ4016Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  constexpr int cNrowsInterleaved = 16;

  block_q4_0x16 *vDst = (block_q4_0x16 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 aDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ40x16(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ5KToQ5K8Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q5_K);
  GGML_ASSERT(pInterleaveBlock == 4 || pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 8;

  block_q5_Kx8 *vDst = (block_q5_Kx8 *)pT->data;
  const block_q5_K *vSrc = (const block_q5_K *)pData;
  block_q5_K aDstTmp[8];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q5_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ5Kx8(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static int fRepackQ6KToQ6K8Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q6_K);
  GGML_ASSERT(pInterleaveBlock == 4 || pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 8;

  block_q6_Kx8 *vDst = (block_q6_Kx8 *)pT->data;
  const block_q6_K *vSrc = (const block_q6_K *)pData;
  block_q6_K aDstTmp[8];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q6_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ6Kx8(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static int fRepackQ40ToQ408Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 8;

  block_q4_0x8 *vDst = (block_q4_0x8 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 aDstTmp[8];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ40x8(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ80ToQ804Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                              size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q8_0);
  GGML_ASSERT(pInterleaveBlock == 4 || pInterleaveBlock == 8);
  constexpr int cNrowsInterleaved = 4;

  block_q8_0x4 *vDst = (block_q8_0x4 *)pT->data;
  const block_q8_0 *vSrc = (const block_q8_0 *)pData;
  block_q8_0 aDstTmp[4];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK8_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q8_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ80x4(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static block_q8_0x16 fMakeBlockQ80x16(block_q8_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q8_0x16 vOut;

  for (int vI = 0; vI < 16; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK8_0 * 16 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 1) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 16;
      int vSrcOffset = vI / 16;
      int vDstOffset = vI;
      vOut.qs[vDstOffset] = pIn[vSrcId].qs[vSrcOffset];
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackQ80ToQ8016Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q8_0);
  constexpr int cNrowsInterleaved = 16;

  block_q8_0x16 *vDst = (block_q8_0x16 *)pT->data;
  const block_q8_0 *vSrc = (const block_q8_0 *)pData;
  block_q8_0 aDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK8_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q8_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ80x16(aDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static block_iq4_nlx4 fMakeBlockIq4Nlx4(block_iq4_nl *pIn, unsigned int pBlckSizeInterleave) {
  block_iq4_nlx4 vOut;

  for (int vI = 0; vI < 4; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_NL * 2 / pBlckSizeInterleave;

  // TODO: this branch seems wrong
  // if (blck_size_interleave == 8) {
  //    for (int i = 0; i < end; ++i) {
  //        int src_id = i % 4;
  //        int src_offset = (i / 4) * blck_size_interleave;
  //        int dst_offset = i * blck_size_interleave;

  //        // Using memcpy to avoid unaligned memory accesses
  //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
  //    }
  //} else
  if (pBlckSizeInterleave == 4) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 4;
      int vSrcOffset = (vI / 4) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], sizeof(uint32_t));
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackIq4NlToIq4Nl4Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_IQ4_NL);
  GGML_ASSERT(pInterleaveBlock == 4);

  const block_iq4_nl *vSrc = (const block_iq4_nl *)pData;
  block_iq4_nlx4 *vDst = (block_iq4_nlx4 *)pT->data;

  block_iq4_nl aDstTmp[4];

  int vNrow = ggml_nrows(pT);
  int vNrowsInterleaved = 4;
  int vNblocks = pT->ne[0] / QK4_NL;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_iq4_nl));

  if (pT->ne[1] % vNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += vNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < vNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockIq4Nlx4(aDstTmp, pInterleaveBlock);
    }
    vSrc += vNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static block_iq4_nlx8 fMakeBlockIq4Nlx8(block_iq4_nl *pIn, unsigned int pBlckSizeInterleave) {
  block_iq4_nlx8 vOut;

  for (int vI = 0; vI < 8; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_NL * 4 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 8) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 8;
      int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], sizeof(uint64_t));
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackIq4NlToIq4Nl8Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_IQ4_NL);
  GGML_ASSERT(pInterleaveBlock == 8);

  const block_iq4_nl *vSrc = (const block_iq4_nl *)pData;
  block_iq4_nlx8 *vDst = (block_iq4_nlx8 *)pT->data;

  block_iq4_nl aDstTmp[8];

  int vNrow = ggml_nrows(pT);
  int vNrowsInterleaved = 8;
  int vNblocks = pT->ne[0] / QK4_NL;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_iq4_nl));

  if (pT->ne[1] % vNrowsInterleaved != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += vNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < vNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockIq4Nlx8(aDstTmp, pInterleaveBlock);
    }
    vSrc += vNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static block_iq4_nlx16 fMakeBlockIq4Nlx16(block_iq4_nl *pIn, unsigned int pBlckSizeInterleave) {
  block_iq4_nlx16 vOut;

  for (int vI = 0; vI < 16; vI++) {
    vOut.d[vI] = pIn[vI].d;
  }

  const int cEnd = QK4_NL * 8 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 1) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 16;
      int vSrcOffset = vI / 16;
      int vDstOffset = vI;

      vOut.qs[vDstOffset] = pIn[vSrcId].qs[vSrcOffset];
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackIq4NlToIq4Nl16Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                   size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_IQ4_NL);
  GGML_ASSERT(pInterleaveBlock == 1);

  const block_iq4_nl *vSrc = (const block_iq4_nl *)pData;
  block_iq4_nlx16 *vDst = (block_iq4_nlx16 *)pT->data;

  block_iq4_nl aDstTmp[16];

  int vNrow = ggml_nrows(pT);
  int vNrowsInterleaved = 16;
  int vNblocks = pT->ne[0] / QK4_NL;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_iq4_nl));

  if (pT->ne[1] % vNrowsInterleaved != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += vNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < vNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockIq4Nlx16(aDstTmp, pInterleaveBlock);
    }
    vSrc += vNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static block_mxfp4x4 fMakeBlockMxfp4x4(block_mxfp4 *pIn, unsigned int pBlckSizeInterleave) {
  block_mxfp4x4 vOut;

  for (int vI = 0; vI < 4; vI++) {
    vOut.e[vI] = pIn[vI].e;
  }

  const int cEnd = QK_MXFP4 * 2 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 4) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 4;
      int vSrcOffset = (vI / 4) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], sizeof(uint32_t));
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackMxfp4ToMxfp44Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_MXFP4);
  GGML_ASSERT(pInterleaveBlock == 4);

  const block_mxfp4 *vSrc = (const block_mxfp4 *)pData;
  block_mxfp4x4 *vDst = (block_mxfp4x4 *)pT->data;

  block_mxfp4 aDstTmp[4];

  int vNrow = ggml_nrows(pT);
  int vNrowsInterleaved = 4;
  int vNblocks = pT->ne[0] / QK_MXFP4;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_mxfp4));

  if (pT->ne[1] % vNrowsInterleaved != 0 || pT->ne[0] % 8 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += vNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < vNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockMxfp4x4(aDstTmp, pInterleaveBlock);
    }
    vSrc += vNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static block_mxfp4x8 fMakeBlockMxfp4x8(block_mxfp4 *pIn, unsigned int pBlckSizeInterleave) {
  block_mxfp4x8 vOut;

  for (int vI = 0; vI < 8; vI++) {
    vOut.e[vI] = pIn[vI].e;
  }

  const int cEnd = QK_MXFP4 * 4 / pBlckSizeInterleave;

  if (pBlckSizeInterleave == 8) {
    for (int vI = 0; vI < cEnd; ++vI) {
      int vSrcId = vI % 8;
      int vSrcOffset = (vI / 8) * pBlckSizeInterleave;
      int vDstOffset = vI * pBlckSizeInterleave;

      memcpy(&vOut.qs[vDstOffset], &pIn[vSrcId].qs[vSrcOffset], sizeof(uint64_t));
    }
  } else {
    GGML_ASSERT(false);
  }

  return vOut;
}

static int fRepackMxfp4ToMxfp48Bl(struct ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_MXFP4);
  GGML_ASSERT(pInterleaveBlock == 8);

  const block_mxfp4 *vSrc = (const block_mxfp4 *)pData;
  block_mxfp4x8 *vDst = (block_mxfp4x8 *)pT->data;

  block_mxfp4 aDstTmp[8];

  int vNrow = ggml_nrows(pT);
  int vNrowsInterleaved = 8;
  int vNblocks = pT->ne[0] / QK_MXFP4;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_mxfp4));

  if (pT->ne[1] % vNrowsInterleaved != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += vNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < vNrowsInterleaved; vI++) {
        aDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockMxfp4x8(aDstTmp, pInterleaveBlock);
    }
    vSrc += vNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

namespace ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

// TODO: generalise.
template <> int repack<block_q4_0, 4, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ40ToQ404Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_q4_0, 8, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ40ToQ404Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ40ToQ408Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q4_K, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ4KToQ4K8Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q4_K, 4, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ4KToQ4K8Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_q2_K, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ2KToQ2K8Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q5_K, 4, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ5KToQ5K8Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_q5_K, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ5KToQ5K8Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q6_K, 4, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ6KToQ6K8Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_q6_K, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ6KToQ6K8Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_iq4_nl, 4, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackIq4NlToIq4Nl4Bl(pT, 4, pData, pDataSize);
}

// TODO: needs to be revisited
// template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> int repack<block_iq4_nl, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackIq4NlToIq4Nl8Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_mxfp4, 4, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackMxfp4ToMxfp44Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_mxfp4, 8, 8>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackMxfp4ToMxfp48Bl(pT, 8, pData, pDataSize);
}

template <> int repack<block_q8_0, 4, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ80ToQ804Bl(pT, 4, pData, pDataSize);
}

template <> int repack<block_q8_0, 8, 4>(struct ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ80ToQ804Bl(pT, 8, pData, pDataSize);
}

#if defined __riscv_zvfh
template <> int repack<block_q4_0, 1, 16>(struct ggml_tensor *t, const void *data, size_t data_size) {
  return fRepackQ40ToQ4016Bl(t, 1, data, data_size);
}

template <> int repack<block_q4_K, 1, 16>(struct ggml_tensor *t, const void *data, size_t data_size) {
  return fRepackQ4KToQ4K16Bl(t, 1, data, data_size);
}

template <> int repack<block_iq4_nl, 1, 16>(struct ggml_tensor *t, const void *data, size_t data_size) {
  return fRepackIq4NlToIq4Nl16Bl(t, 1, data, data_size);
}

template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor *t, const void *data, size_t data_size) {
  return fRepackQ80ToQ8016Bl(t, 1, data, data_size);
}

template <> int repack<block_q2_K, 1, 16>(struct ggml_tensor *t, const void *data, size_t data_size) {
  return fRepackQ2KToQ2K16Bl(t, 1, data, data_size);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <>
void gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q4_0_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q4_0_4x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q4_0_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q2_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q4_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q4_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q5_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q5_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q6_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q6_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                              int pNc) {
  ggml_gemv_iq4_nl_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                              int pNc) {
  ggml_gemv_iq4_nl_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                             int pNc) {
  ggml_gemv_mxfp4_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                             int pNc) {
  ggml_gemv_mxfp4_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q8_0_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemv_q8_0_4x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

#if defined __riscv_zvfh
template <>
void gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                               int nc) {
  ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <>
void gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q4_0_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q4_0_4x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q4_0_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q2_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q4_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q4_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q5_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q5_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q6_K_8x4_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q6_K_8x8_q8_K(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                              int pNc) {
  ggml_gemm_iq4_nl_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                              int pNc) {
  ggml_gemm_iq4_nl_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                             int pNc) {
  ggml_gemm_mxfp4_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                             int pNc) {
  ggml_gemm_mxfp4_8x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q8_0_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

template <>
void gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int pN, float *pS, size_t pBs, const void *pVx, const void *pVy, int pNr,
                                            int pNc) {
  ggml_gemm_q8_0_4x8_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

#if defined __riscv_zvfh
template <>
void gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                               int nc) {
  ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                                             int nc) {
  ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

class tensor_traits_base : public ggml::cpu::tensor_traits {
public:
  virtual int repack(struct ggml_tensor *pT, const void *pData, size_t pDataSize) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
class tensor_traits : public tensor_traits_base {

  bool work_size(int /* n_threads */, const struct ggml_tensor *pOp, size_t &pSize) override {
    // not realy a GGML_TYPE_Q8_0 but same size.
    switch (pOp->op) {
    case GGML_OP_MUL_MAT: {
      pSize = ggml_row_size(PARAM_TYPE, ggml_nelements(pOp->src[1]));
      return true;
    }
    case GGML_OP_MUL_MAT_ID: {
      pSize = ggml_row_size(PARAM_TYPE, ggml_nelements(pOp->src[1]));
      pSize = GGML_PAD(pSize, sizeof(int64_t)); // + padding for next block.

      const int64_t cNe02 = pOp->src[0]->ne[2]; // n_as, n_expert
      const int64_t cNe12 = pOp->src[1]->ne[2]; // n_tokens

      const size_t cSizeofMmidRowMapping = sizeof(int64_t);

      pSize += cSizeofMmidRowMapping * cNe02 * (cNe12 + 1);

      return true;
    }
    default:
      // GGML_ABORT("fatal error");
      break;
    }
    return false;
  }

  bool compute_forward(struct ggml_compute_params *pParams, struct ggml_tensor *pOp) override {
    switch (pOp->op) {
    case GGML_OP_MUL_MAT:
      mForwardMulMat(pParams, pOp);
      return true;
    case GGML_OP_MUL_MAT_ID:
      mForwardMulMatId(pParams, pOp);
      return true;
    default:
      // GGML_ABORT("fatal error");
      break;
    }
    return false;
  }

  void mForwardMulMatOneChunk(ggml_compute_params *pParams, ggml_tensor *pOp, int64_t pSrc0Start, int64_t pSrc0End,
                              int64_t pSrc1Start, int64_t pSrc1End) {
    const ggml_tensor *src0 = pOp->src[0];
    const ggml_tensor *src1 = pOp->src[1];
    ggml_tensor *dst = pOp;

    GGML_TENSOR_BINARY_OP_LOCALS

    const size_t cSrc1ColStride = ggml_row_size(PARAM_TYPE, ne10);

    GGML_ASSERT(ne03 == 1 && ne13 == 1);
    GGML_ASSERT(ne12 % ne02 == 0);
    const int64_t cR2 = ne12 / ne02;

    const int64_t cI12 = pSrc1Start / ne1;
    const int64_t cI11 = pSrc1Start - cI12 * ne1;

    // Determine batch index
    const int64_t cI02 = cI12 / cR2;

    const int64_t cI1 = cI11;
    const int64_t cI2 = cI12;

    const char *vSrc0Ptr = (const char *)src0->data + cI02 * nb02;
    const char *vSrc1Ptr = (const char *)pParams->wdata + (cI11 + cI12 * ne11) * cSrc1ColStride;
    char *vDstPtr = ((char *)dst->data + (cI1 * nb1 + cI2 * nb2));

    const int64_t cNrows = pSrc1End - pSrc1Start;
    const int64_t cNcols = pSrc0End - pSrc0Start;

    GGML_ASSERT(vSrc1Ptr + cSrc1ColStride * cNrows <= (const char *)pParams->wdata + pParams->wsize);

    // If there are more than three rows in src1, use gemm; otherwise, use gemv.
    if (cNrows > 3) {
      gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *)(vDstPtr) + pSrc0Start, nb1 / nb0,
                                                       vSrc0Ptr + pSrc0Start * nb01, vSrc1Ptr, cNrows - (cNrows % 4),
                                                       cNcols);
    }
    for (int vIter = cNrows - (cNrows % 4); vIter < cNrows; vIter++) {
      gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *)(vDstPtr + (vIter * nb1)) + pSrc0Start, ne01,
                                                       vSrc0Ptr + pSrc0Start * nb01,
                                                       vSrc1Ptr + (cSrc1ColStride * vIter), 1 /* nrows */, cNcols);
    }
  }

  void mForwardMulMat(ggml_compute_params *pParams, ggml_tensor *pOp) {
    const ggml_tensor *src0 = pOp->src[0];
    const ggml_tensor *src1 = pOp->src[1];
    ggml_tensor *dst = pOp;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cIth = pParams->ith;
    const int cNth = pParams->nth;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // TODO: General batched mul mat for 4D tensors
    // Currently only supports 3D tensors
    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3 == 1);

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_n_dims(pOp->src[0]) == 2);
    // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

    char *vWdata = static_cast<char *>(pParams->wdata);
    const size_t cNbw1 = ggml_row_size(PARAM_TYPE, ne10);
    const size_t cNbw2 = cNbw1 * ne11;

    assert(pParams->wsize >= cNbw2 * ne12);

    const ggml_from_float_t cFromFloat = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

    // INFO: Quantization is done in planes to avoid extra complexity in chunking.
    // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
    // the planes are broadcast.
    for (int64_t vI12 = 0; vI12 < ne12; vI12++) {
      char *vDataPtr = (char *)src1->data + vI12 * nb12;
      char *vWdataPtr = vWdata + vI12 * cNbw2;

      for (int64_t vI11 = cIth * 4; vI11 < ne11 - ne11 % 4; vI11 += cNth * 4) {
        ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((float *)(vDataPtr + vI11 * nb11),
                                                    (void *)(vWdataPtr + vI11 * cNbw1), 4, ne10);
      }

      const int64_t cI11Processed = ne11 - ne11 % 4;
      for (int64_t vI11 = cI11Processed + cIth; vI11 < ne11; vI11 += cNth) {
        cFromFloat((float *)(vDataPtr + vI11 * nb11), (void *)(vWdataPtr + vI11 * cNbw1), ne10);
      }
    }

    // disable for NUMA
    const bool cDisableChunking = ggml_is_numa();

    // 4x chunks per thread
    const int64_t cNr0 = ggml_nrows(pOp->src[0]);

    int vNthScaled = cNth * 4;
    int64_t vChunkSize0 = (cNr0 + vNthScaled - 1) / vNthScaled;
    int64_t vNchunk0 = (cNr0 + vChunkSize0 - 1) / vChunkSize0;

    // src1 is chunked only by full planes.
    // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
    // to route them thorugh GEMV.
    // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
    // to avoid affecting their performance
    int64_t vNchunk1 = ne12;

    // Ensure minimum chunk size to avoid alignment issues with high thread counts
    // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
    const int64_t cMinChunkSize = NB_COLS;
    if (vNchunk0 > 0 && (cNr0 / vNchunk0) < cMinChunkSize && cNr0 >= cMinChunkSize) {
      vNchunk0 = (cNr0 + cMinChunkSize - 1) / cMinChunkSize;
    }

    int64_t vDr0 = (cNr0 + vNchunk0 - 1) / vNchunk0;
    // Only increase nchunk0 to nth if it won't make chunks too small
    if (cNth == 1 || ((vNchunk0 < cNth || cDisableChunking) && (cNr0 + cNth - 1) / cNth >= cMinChunkSize)) {
      vNchunk0 = cNth;
      vDr0 = (cNr0 + vNchunk0 - 1) / vNchunk0;
    }

    // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
    // This prevents creating too many tiny chunks that could overlap after alignment
    const int64_t cMaxNchunk = (cNr0 + cMinChunkSize - 1) / cMinChunkSize;
    vNchunk0 = MIN(vNchunk0, cMaxNchunk);

    if (cIth == 0) {
      // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at
      // the start.
      ggml_threadpool_chunk_set(pParams->threadpool, cNth);
    }

    ggml_barrier(pParams->threadpool);

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int vCurrentChunk = cIth;

    while (vCurrentChunk < vNchunk0 * vNchunk1) {
      const int64_t cIth0 = vCurrentChunk % vNchunk0;
      const int64_t cIth1 = vCurrentChunk / vNchunk0;

      int64_t vSrc0Start = vDr0 * cIth0;
      int64_t vSrc0End = MIN(vSrc0Start + vDr0, cNr0);

      // full-plane range for src1
      int64_t vSrc1Start = cIth1 * ne11;
      int64_t vSrc1End = (cIth1 + 1) * ne11;

      // Align boundaries to NB_COLS - round up to ensure all data is included
      // The chunk size limiting above ensures chunks are large enough to prevent overlaps
      vSrc0Start = (vSrc0Start % NB_COLS) ? vSrc0Start + NB_COLS - (vSrc0Start % NB_COLS) : vSrc0Start;
      vSrc0End = (vSrc0End % NB_COLS) ? vSrc0End + NB_COLS - (vSrc0End % NB_COLS) : vSrc0End;
      vSrc0End = MIN(vSrc0End, ne01);

      // Make sure current plane is the last one before exiting
      if (vSrc0Start >= vSrc0End) {
        vCurrentChunk = ggml_threadpool_chunk_add(pParams->threadpool, 1);
        continue;
      }

      mForwardMulMatOneChunk(pParams, dst, vSrc0Start, vSrc0End, vSrc1Start, vSrc1End);

      vCurrentChunk = ggml_threadpool_chunk_add(pParams->threadpool, 1);
    }
  }

  void mForwardMulMatId(ggml_compute_params *pParams, ggml_tensor *pOp) {
    const ggml_tensor *src0 = pOp->src[0];
    const ggml_tensor *src1 = pOp->src[1];
    const ggml_tensor *vIds = pOp->src[2];
    ggml_tensor *dst = pOp;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cIth = pParams->ith;
    const int cNth = pParams->nth;

    const ggml_from_float_t cFromFloat = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3 == 1);

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // row groups
    const int cNIds = vIds->ne[0]; // n_expert_used
    const int cNAs = ne02;         // n_expert

    const size_t cNbw1 = ggml_row_size(PARAM_TYPE, ne10);
    const size_t cNbw2 = cNbw1 * ne11;
    const size_t cNbw3 = cNbw2 * ne12;

    struct mmid_row_mapping {
      int32_t i1;
      int32_t vI2;
    };

    GGML_ASSERT(pParams->wsize >= (GGML_PAD(cNbw3, sizeof(int64_t)) + cNAs * (ne12 + 1) * sizeof(mmid_row_mapping)));

    auto *vWdata = (char *)pParams->wdata;
    auto *vWdataSrc1End = (char *)vWdata + GGML_PAD(cNbw3, sizeof(int64_t));

    // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
    auto *vMatrixRowCounts = (int64_t *)(vWdataSrc1End);                                         // [n_as]
    struct mmid_row_mapping *matrix_rows = (struct mmid_row_mapping *)(vMatrixRowCounts + cNAs); // [n_as][ne12]

    // src1: float32 => param type
    for (int64_t vI12 = 0; vI12 < ne12; ++vI12) {
      for (int64_t vI11 = cIth; vI11 < ne11; vI11 += cNth) {
        cFromFloat((float *)((char *)src1->data + vI12 * nb12 + vI11 * nb11),
                   (void *)(vWdata + vI12 * cNbw2 + vI11 * cNbw1), ne10);
      }
    }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

    if (cIth == 0) {
      // initialize matrix_row_counts
      memset(vMatrixRowCounts, 0, cNAs * sizeof(int64_t));

      // group rows by src0 matrix
      for (int32_t vIid1 = 0; vIid1 < vIds->ne[1]; ++vIid1) {
        for (int32_t vId = 0; vId < cNIds; ++vId) {
          const int32_t cI02 = *(const int32_t *)((const char *)vIds->data + vIid1 * vIds->nb[1] + vId * vIds->nb[0]);

          GGML_ASSERT(cI02 >= 0 && cI02 < cNAs);

          MMID_MATRIX_ROW(cI02, vMatrixRowCounts[cI02]) = {vId, vIid1};
          vMatrixRowCounts[cI02] += 1;
        }
      }
    }

    ggml_barrier(pParams->threadpool);

    // compute each matrix multiplication in sequence
    for (int vCurA = 0; vCurA < cNAs; ++vCurA) {
      const int64_t cCne1 = vMatrixRowCounts[vCurA];

      if (cCne1 == 0) {
        continue;
      }

      const auto *vSrc0Cur = (const char *)src0->data + vCurA * nb02;

      // const int64_t nr0 = ne01; // src0 rows
      const int64_t cNr1 = cCne1; // src1 rows

      int64_t vSrc0CurStart = (cIth * ne01) / cNth;
      int64_t vSrc0CurEnd = ((cIth + 1) * ne01) / cNth;

      // Align boundaries to NB_COLS - round up to ensure all data is included
      vSrc0CurStart = (vSrc0CurStart % NB_COLS) ? vSrc0CurStart + NB_COLS - (vSrc0CurStart % NB_COLS) : vSrc0CurStart;
      vSrc0CurEnd = (vSrc0CurEnd % NB_COLS) ? vSrc0CurEnd + NB_COLS - (vSrc0CurEnd % NB_COLS) : vSrc0CurEnd;
      if (vSrc0CurEnd > ne01) {
        vSrc0CurEnd = ne01;
      }

      if (vSrc0CurStart >= vSrc0CurEnd) {
        return;
      }

      for (int vIr1 = 0; vIr1 < cNr1; vIr1++) {
        struct mmid_row_mapping vRowMapping = MMID_MATRIX_ROW(vCurA, vIr1);

        const int cId = vRowMapping.i1; // selected expert index

        const int64_t cI11 = cId % ne11;
        const int64_t cI12 = vRowMapping.vI2; // row index in src1

        const int64_t i1 = cId;   // selected expert index
        const int64_t cI2 = cI12; // row

        const auto *vSrc1Col = (const char *)vWdata + (cI11 * cNbw1 + cI12 * cNbw2);

        gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
            ne00, (float *)((char *)dst->data + (i1 * nb1 + cI2 * nb2)) + vSrc0CurStart, ne01,
            vSrc0Cur + vSrc0CurStart * nb01, vSrc1Col, 1, vSrc0CurEnd - vSrc0CurStart);
      }
    }
#undef MMID_MATRIX_ROW
  }

  int repack(struct ggml_tensor *pT, const void *pData, size_t pDataSize) override {
    GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, pT->name, ggml_type_name(pT->type), (int)NB_COLS,
                   (int)INTER_SIZE);
    return ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(pT, pData, pDataSize);
  }
};

} // namespace ggml::cpu::repack

static const ggml::cpu::tensor_traits *fGgmlRepackGetOptimalRepackType(const struct ggml_tensor *pCur) {
  // instance for Q4
  static const ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> cQ404x4Q80;
  static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> cQ404x8Q80;
  static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> cQ408x8Q80;

  // instance for Q4_K
  static const ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> cQ4K8x4Q8K;
  static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> cQ4K8x8Q8K;

  // instance for Q5_K
  static const ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> cQ5K8x4Q8K;
  static const ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> cQ5K8x8Q8K;

  // instance for Q6_K
  static const ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> cQ6K8x4Q8K;
  static const ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> cQ6K8x8Q8K;

  // instance for Q2
  static const ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> cQ2K8x8Q8K;

  // instance for IQ4
  static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> cIq4Nl4x4Q80;
  static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> cIq4Nl8x8Q80;

  // instance for MXFP4
  static const ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> cMxfp44x4Q80;
  static const ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> cMxfp48x8Q80;

  // instance for Q8_0
  static const ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> cQ804x4Q80;
  static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> cQ804x8Q80;

  // instances for RISC-V
  //
  // These implement outer-product style matrix multiplication kernels with
  // an interleave of 1.
#if defined __riscv_zvfh
  static const ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
  static const ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
  static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
  static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
  static const ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

  if (pCur->type == GGML_TYPE_Q4_0) {
    if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ408x8Q80;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cQ404x8Q80;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cQ404x4Q80;
      }
    }
    if (ggml_cpu_has_riscv_v()) {
#if defined __riscv_zvfh
      switch (__riscv_vlenb() * 8) {
      case 128: {
        break;
      } // TODO
      case 256: {
        if (pCur->ne[1] % 16 == 0) {
          return &q4_0_16x1_q8_0;
        }
        break;
      }
      case 512: {
        break;
      } // TODO
      case 1024: {
        break;
      } // TODO
      default: {
        return nullptr;
      }
      }
#endif
    }
  } else if (pCur->type == GGML_TYPE_Q4_K) {
    if (ggml_cpu_has_avx2()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ4K8x8Q8K;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ4K8x8Q8K;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ4K8x4Q8K;
      }
    }
    if (ggml_cpu_has_riscv_v()) {
#if defined __riscv_zvfh
      switch (__riscv_vlenb() * 8) {
      case 128: {
        break;
      } // TODO
      case 256: {
        if (pCur->ne[1] % 16 == 0) {
          return &q4_K_16x1_q8_K;
        }
        break;
      }
      case 512: {
        break;
      } // TODO
      case 1024: {
        break;
      } // TODO
      default: {
        return nullptr;
      }
      }
#endif
    }
  } else if (pCur->type == GGML_TYPE_Q2_K) {
    if (ggml_cpu_has_avx512()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ2K8x8Q8K;
      }
    }
    if (ggml_cpu_has_riscv_v()) {
#if defined __riscv_zvfh
      switch (__riscv_vlenb() * 8) {
      case 128: {
        break;
      } // TODO
      case 256: {
        if (pCur->ne[1] % 16 == 0) {
          return &q2_K_16x1_q8_K;
        }
        break;
      }
      case 512: {
        break;
      } // TODO
      case 1024: {
        break;
      } // TODO
      default: {
        return nullptr;
      }
      }
#endif
    }
  } else if (pCur->type == GGML_TYPE_Q5_K) {
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ5K8x8Q8K;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ5K8x4Q8K;
      }
    }
  } else if (pCur->type == GGML_TYPE_Q6_K) {
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ6K8x8Q8K;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cQ6K8x4Q8K;
      }
    }
  } else if (pCur->type == GGML_TYPE_IQ4_NL) {
    if (ggml_cpu_has_avx2()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cIq4Nl8x8Q80;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cIq4Nl4x4Q80;
      }
    }
    if (ggml_cpu_has_riscv_v()) {
#if defined __riscv_zvfh
      switch (__riscv_vlenb() * 8) {
      case 128: {
        break;
      } // TODO
      case 256: {
        if (pCur->ne[1] % 16 == 0) {
          return &iq4_nl_16x1_q8_0;
        }
        break;
      }
      case 512: {
        break;
      } // TODO
      case 1024: {
        break;
      } // TODO
      default: {
        return nullptr;
      }
      }
#endif
    }
  } else if (pCur->type == GGML_TYPE_MXFP4) {
    if (ggml_cpu_has_avx2()) {
      if (pCur->ne[1] % 8 == 0) {
        return &cMxfp48x8Q80;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cMxfp44x4Q80;
      }
    }
  } else if (pCur->type == GGML_TYPE_Q8_0) {
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cQ804x8Q80;
      }
    }
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
      if (pCur->ne[1] % 4 == 0) {
        return &cQ804x4Q80;
      }
    }
    if (ggml_cpu_has_riscv_v()) {
#if defined __riscv_zvfh
      switch (__riscv_vlenb() * 8) {
      case 128: {
        break;
      } // TODO
      case 256: {
        if (pCur->ne[1] % 16 == 0) {
          return &q8_0_16x1_q8_0;
        }
        break;
      }
      case 512: {
        break;
      } // TODO
      case 1024: {
        break;
      } // TODO
      default: {
        return nullptr;
      }
      }
#endif
    }
  }

  return nullptr;
}

static enum ggml_status fGgmlBackendCpuRepackBufferInitTensor(ggml_backend_buffer_t pBuffer,
                                                              struct ggml_tensor *pTensor) {
  pTensor->extra = (void *)const_cast<ggml::cpu::tensor_traits *>(fGgmlRepackGetOptimalRepackType(pTensor));

  GGML_UNUSED(pBuffer);
  return GGML_STATUS_SUCCESS;
}

static void fGgmlBackendCpuRepackBufferSetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                                 const void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pOffset == 0);
  GGML_ASSERT(pSize == ggml_nbytes(pTensor));

  auto vTensorTraits = (ggml::cpu::repack::tensor_traits_base *)pTensor->extra;
  auto vOk = vTensorTraits->repack(pTensor, pData, pSize);

  GGML_ASSERT(vOk == 0);
  GGML_UNUSED(pBuffer);
}

static const char *fGgmlBackendCpuRepackBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  return "CPU_REPACK";

  GGML_UNUSED(pBuft);
}

static ggml_backend_buffer_t fGgmlBackendCpuRepackBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft,
                                                                        size_t pSize) {
  ggml_backend_buffer_t vBuffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), pSize);

  if (vBuffer == nullptr) {
    return nullptr;
  }

  vBuffer->buft = pBuft;
  vBuffer->iface.init_tensor = fGgmlBackendCpuRepackBufferInitTensor;
  vBuffer->iface.set_tensor = fGgmlBackendCpuRepackBufferSetTensor;
  vBuffer->iface.get_tensor = nullptr;
  vBuffer->iface.cpy_tensor = nullptr;
  return vBuffer;
}

static size_t fGgmlBackendCpuRepackBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  return TENSOR_ALIGNMENT;

  GGML_UNUSED(pBuft);
}

namespace ggml::cpu::repack {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
  bool supports_op(ggml_backend_dev_t, const struct ggml_tensor *pOp) override {
    if (pOp->op == GGML_OP_MUL_MAT && pOp->src[0]->buffer && (ggml_n_dims(pOp->src[0]) == 2) &&
        pOp->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
        fGgmlRepackGetOptimalRepackType(pOp->src[0])) {
      if (pOp->src[1]->buffer && !ggml_backend_buft_is_host(pOp->src[1]->buffer->buft)) {
        return false;
      }
      if (pOp->src[1]->type == GGML_TYPE_F32) {
        return true;
      }
      // if (op->src[1]->type == GGML_TYPE_Q8_0) {
      //     return true;
      // }
      //  may be possible if Q8_0 packed...
    } else if (pOp->op == GGML_OP_MUL_MAT_ID && pOp->src[0]->buffer && (ggml_n_dims(pOp->src[0]) == 3) &&
               pOp->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
               fGgmlRepackGetOptimalRepackType(pOp->src[0])) {
      if (pOp->src[1]->buffer && !ggml_backend_buft_is_host(pOp->src[1]->buffer->buft)) {
        return false;
      }
      if (pOp->src[1]->type == GGML_TYPE_F32) {
        return true;
      }
      // if (op->src[1]->type == GGML_TYPE_Q8_0) {
      //     return true;
      // }
    }
    return false;
  }

  ggml::cpu::tensor_traits *get_tensor_traits(const struct ggml_tensor *pOp) override {
    if (pOp->op == GGML_OP_MUL_MAT || pOp->op == GGML_OP_MUL_MAT_ID) {
      if (pOp->src[0]->buffer && pOp->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()) {
        return (ggml::cpu::tensor_traits *)pOp->src[0]->extra;
      }
    }
    return nullptr;
  }
};
} // namespace ggml::cpu::repack

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
  static struct ggml_backend_buffer_type vGgmlBackendCpuBufferTypeRepack = {
      /* .iface    = */ {
          /* .get_name         = */ fGgmlBackendCpuRepackBufferTypeGetName,
          /* .alloc_buffer     = */ fGgmlBackendCpuRepackBufferTypeAllocBuffer,
          /* .get_alignment    = */ fGgmlBackendCpuRepackBufferTypeGetAlignment,
          /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
          /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
          /* .is_host          = */ nullptr,
      },
      /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */ new ggml::cpu::repack::extra_buffer_type(),
  };

  return &vGgmlBackendCpuBufferTypeRepack;
}
