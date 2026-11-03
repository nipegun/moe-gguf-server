#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)                                                                                       \
  _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)), _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x) _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)                                                                           \
  _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const *)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)                                                                     \
  _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(ggml_fp16_t *x, ggml_fp16_t *y) {
  float tmp[16];

  for (int i = 0; i < 8; i++) {
    tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
  }

  for (int i = 0; i < 8; i++) {
    tmp[i + 8] = GGML_CPU_FP16_TO_FP32(y[i]);
  }

  return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
  float tmp[16];
  uint16_t tmphalf[8];
  _mm_storeu_si128((__m128i *)tmphalf, x);

  for (int i = 0; i < 4; i++) {
    tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    tmp[i + 4] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    tmp[i + 8] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    tmp[i + 12] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
  }

  return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
  float tmp[8];

  for (int i = 0; i < 8; i++) {
    tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
  }

  return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
  float tmp[8];

  for (int i = 0; i < 4; i++) {
    tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    tmp[i + 4] = GGML_CPU_FP16_TO_FP32(x[i]);
  }

  return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
  uint16_t tmphalf[8];
  float tmp[8];

  _mm_storeu_si128((__m128i *)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)x), arrangeMask));
  for (int i = 0; i < 8; i++) {
    tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
  }

  return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x) __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask) __avx_repeat_f32cx8_load(x)
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask) __avx_rearranged_f32cx8_load(x, arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y) __avx512_f32cx8x2_load(x, y)
#define GGML_F32Cx16_REPEAT_LOAD(x) __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif

static inline int fNearestInt(float pFval) {
  assert(fabsf(pFval) <= 4194303.f);
  float vVal = pFval + 12582912.f;
  int vI;
  memcpy(&vI, &vVal, sizeof(int));
  return (vI & 0x007fffff) - 0x00400000;
}

#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector, then add the accumulator
static inline __m512i fSumI16PairsAccInt32x16(const __m512i pAcc, const __m512i pX) {
  const __m512i cOnes = _mm512_set1_epi16(1);
  return _mm512_add_epi32(pAcc, _mm512_madd_epi16(cOnes, pX));
}

static inline __m512i fMulSumUs8PairsAccInt32x16(const __m512i pAcc, const __m512i pAx, const __m512i pSy) {
#if defined(__AVX512VNNI__)
  return _mm512_dpbusd_epi32(pAcc, pAx, pSy);
#else
  // Perform multiplication and create 16-bit values
  const __m512i dot = _mm512_maddubs_epi16(pAx, pSy);
  return fSumI16PairsAccInt32x16(pAcc, dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector，then add the accumulator
static inline __m512i fMulSumI8PairsAccInt32x16(const __m512i pAcc, const __m512i pX, const __m512i pY) {
  const __m512i cZero = _mm512_setzero_si512();
  // Get absolute values of x vectors
  const __m512i cAx = _mm512_abs_epi8(pX);
  // Sign the values of the y vectors
  __mmask64 vBlt0 = _mm512_movepi8_mask(pX);
  const __m512i cSy = _mm512_mask_sub_epi8(pY, vBlt0, cZero, pY);
  return fMulSumUs8PairsAccInt32x16(pAcc, cAx, cSy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector, then add the accumulator
static inline __m256i fSumI16PairsAccInt32x8(const __m256i pAcc, const __m256i pX) {
  const __m256i cOnes = _mm256_set1_epi16(1);
  return _mm256_add_epi32(pAcc, _mm256_madd_epi16(cOnes, pX));
}

static inline __m256i fMulSumUs8PairsAccInt32x8(const __m256i pAcc, const __m256i pAx, const __m256i pSy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  return _mm256_dpbusd_epi32(pAcc, pAx, pSy);
#elif defined(__AVXVNNI__)
  return _mm256_dpbusd_avx_epi32(pAcc, pAx, pSy);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(pAx, pSy);
  return fSumI16PairsAccInt32x8(pAcc, dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector, then add the accumulator
static inline __m256i fMulSumI8PairsAccInt32x8(const __m256i pAcc, const __m256i pX, const __m256i pY) {
#if defined(__AVXVNNIINT8__)
  return _mm256_dpbssd_epi32(pAcc, pX, pY);
#else
  // Get absolute values of x vectors
  const __m256i cAx = _mm256_sign_epi8(pX, pX);
  // Sign the values of the y vectors
  const __m256i cSy = _mm256_sign_epi8(pY, pX);
  return fMulSumUs8PairsAccInt32x8(pAcc, cAx, cSy);
#endif
}
#endif

void ggml_quantize_mat_q8_0_4x8(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;

#if defined(__AVX2__) || defined(__AVX__)
  float aId[4];
  __m256 aSrcv[4][4];
  __m256 aIdvec[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(pX + vRowIter * pK + vI * 32);
      __m256 v1 = _mm256_loadu_ps(pX + vRowIter * pK + vI * 32 + 8);
      __m256 v2 = _mm256_loadu_ps(pX + vRowIter * pK + vI * 32 + 16);
      __m256 v3 = _mm256_loadu_ps(pX + vRowIter * pK + vI * 32 + 24);

      // Compute max(abs(e)) for the block
      const __m256 cSignBit = _mm256_set1_ps(-0.0f);
      __m256 vMaxAbs = _mm256_andnot_ps(cSignBit, v0);
      vMaxAbs = _mm256_max_ps(vMaxAbs, _mm256_andnot_ps(cSignBit, v1));
      vMaxAbs = _mm256_max_ps(vMaxAbs, _mm256_andnot_ps(cSignBit, v2));
      vMaxAbs = _mm256_max_ps(vMaxAbs, _mm256_andnot_ps(cSignBit, v3));

      __m128 vMax4 = _mm_max_ps(_mm256_extractf128_ps(vMaxAbs, 1), _mm256_castps256_ps128(vMaxAbs));
      vMax4 = _mm_max_ps(vMax4, _mm_movehl_ps(vMax4, vMax4));
      vMax4 = _mm_max_ss(vMax4, _mm_movehdup_ps(vMax4));
      const float cMaxScalar = _mm_cvtss_f32(vMax4);

      // Divided by 127.f to mirror results in quantize_row_q8_0
      const float cD = cMaxScalar / 127.f;
      aId[vRowIter] = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f; // d ? 1.0f / d : 0.0f;

      // Store the scale for the individual block
      vY[vI].d[vRowIter] = GGML_CPU_FP32_TO_FP16(cD);

      // Store the values in blocks of eight values - Aim is to use these later for block interleaving
      aSrcv[vRowIter][0] = v0;
      aSrcv[vRowIter][1] = v1;
      aSrcv[vRowIter][2] = v2;
      aSrcv[vRowIter][3] = v3;
      aIdvec[vRowIter] = _mm256_set1_ps(aId[vRowIter]);
    }

    // The loop iterates four times - The aim is to get 4 corresponding chunks of eight bytes from the original weight
    // blocks that are interleaved
    for (int vJ = 0; vJ < 4; vJ++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(aSrcv[0][vJ], aIdvec[0]);
      __m256 v1 = _mm256_mul_ps(aSrcv[1][vJ], aIdvec[1]);
      __m256 v2 = _mm256_mul_ps(aSrcv[2][vJ], aIdvec[2]);
      __m256 v3 = _mm256_mul_ps(aSrcv[3][vJ], aIdvec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i vI0 = _mm256_cvtps_epi32(v0);
      __m256i vI1 = _mm256_cvtps_epi32(v1);
      __m256i vI2 = _mm256_cvtps_epi32(v2);
      __m256i vI3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
      // Convert int32 to int16
      vI0 = _mm256_packs_epi32(vI0, vI1);
      vI2 = _mm256_packs_epi32(vI2, vI3);
      // Convert int16 to int8
      vI0 = _mm256_packs_epi16(vI0, vI2);

      //  Permute and store the quantized weights in the required order after the pack instruction
      const __m256i cPerm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      vI0 = _mm256_permutevar8x32_epi32(vI0, cPerm);

      _mm256_storeu_si256((__m256i *)(vY[vI].qs + 32 * vJ), vI0);
#else
      // Since we don't have in AVX some necessary functions,
      // we split the registers in half and call AVX2 analogs from SSE
      __m128i ni0 = _mm256_castsi256_si128(vI0);
      __m128i ni1 = _mm256_extractf128_si256(vI0, 1);
      __m128i ni2 = _mm256_castsi256_si128(vI1);
      __m128i ni3 = _mm256_extractf128_si256(vI1, 1);
      __m128i ni4 = _mm256_castsi256_si128(vI2);
      __m128i ni5 = _mm256_extractf128_si256(vI2, 1);
      __m128i ni6 = _mm256_castsi256_si128(vI3);
      __m128i ni7 = _mm256_extractf128_si256(vI3, 1);

      // Convert int32 to int16
      ni0 = _mm_packs_epi32(ni0, ni1);
      ni2 = _mm_packs_epi32(ni2, ni3);
      ni4 = _mm_packs_epi32(ni4, ni5);
      ni6 = _mm_packs_epi32(ni6, ni7);
      // Convert int16 to int8
      ni0 = _mm_packs_epi16(ni0, ni2);
      ni4 = _mm_packs_epi16(ni4, ni6);
      _mm_storeu_si128((__m128i *)(vY[vI].qs + 32 * vJ), ni0);
      _mm_storeu_si128((__m128i *)(vY[vI].qs + 32 * vJ + 16), ni4);
#endif
    }
  }

#else
  UNUSED(cNb);
  UNUSED(vY);
  ggml_quantize_mat_q8_0_4x8_generic(pX, pVy, pK);
#endif
}

void ggml_quantize_mat_q8_K_4x8(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t k) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  const int cNb = k / QK_K;

  block_q8_Kx4 *GGML_RESTRICT vY = (block_q8_Kx4 *)pVy;

#if defined(__AVX2__)
  float aIscale[4];
  __m256 aSrcv[4][32];
  __m256 aIscaleVec[4];

  for (int vI = 0; vI < cNb; vI++) {
    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256);
      __m256 v1 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + 8);
      __m256 v2 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + 16);
      __m256 v3 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + 24);

      // Compute max(abs(e)) for the block
      const __m256 cSignBit = _mm256_set1_ps(-0.0f);
      __m256 vAbs0 = _mm256_andnot_ps(cSignBit, v0);
      __m256 vAbs1 = _mm256_andnot_ps(cSignBit, v1);
      __m256 vAbs2 = _mm256_andnot_ps(cSignBit, v2);
      __m256 vAbs3 = _mm256_andnot_ps(cSignBit, v3);

      __m256 vMaxAbs = _mm256_max_ps(vAbs0, vAbs1);
      vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs2);
      vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs3);

      __m256 vMask0 = _mm256_cmp_ps(vMaxAbs, v0, _CMP_EQ_OQ);
      __m256 vMask1 = _mm256_cmp_ps(vMaxAbs, v1, _CMP_EQ_OQ);
      __m256 vMask2 = _mm256_cmp_ps(vMaxAbs, v2, _CMP_EQ_OQ);
      __m256 vMask3 = _mm256_cmp_ps(vMaxAbs, v3, _CMP_EQ_OQ);

      __m256 vMaskAbs = _mm256_or_ps(_mm256_or_ps(vMask0, vMask1), _mm256_or_ps(vMask2, vMask3));

      aSrcv[vRowIter][0] = v0;
      aSrcv[vRowIter][1] = v1;
      aSrcv[vRowIter][2] = v2;
      aSrcv[vRowIter][3] = v3;

      for (int vSb = 1; vSb < 8; vSb++) {
        // Temporarily stores absolute quant values
        __m256 vTempAbs = vMaxAbs;

        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + vSb * 32);
        __m256 v1 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + vSb * 32 + 8);
        __m256 v2 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + vSb * 32 + 16);
        __m256 v3 = _mm256_loadu_ps(pX + vRowIter * k + vI * 256 + vSb * 32 + 24);

        // Compute max(abs(e)) for the block
        __m256 vAbs0 = _mm256_andnot_ps(cSignBit, v0);
        __m256 vAbs1 = _mm256_andnot_ps(cSignBit, v1);
        __m256 vAbs2 = _mm256_andnot_ps(cSignBit, v2);
        __m256 vAbs3 = _mm256_andnot_ps(cSignBit, v3);

        vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs0);
        vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs1);
        vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs2);
        vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs3);

        __m256 vMaskPrev = _mm256_cmp_ps(vTempAbs, vMaxAbs, _CMP_EQ_OQ);
        vMaskAbs = _mm256_and_ps(vMaskAbs, vMaskPrev);

        vMask0 = _mm256_cmp_ps(vMaxAbs, v0, _CMP_EQ_OQ);
        vMask1 = _mm256_cmp_ps(vMaxAbs, v1, _CMP_EQ_OQ);
        vMask2 = _mm256_cmp_ps(vMaxAbs, v2, _CMP_EQ_OQ);
        vMask3 = _mm256_cmp_ps(vMaxAbs, v3, _CMP_EQ_OQ);

        __m256 vMaskCurr = _mm256_or_ps(_mm256_or_ps(vMask0, vMask1), _mm256_or_ps(vMask2, vMask3));
        vMaskAbs = _mm256_or_ps(vMaskAbs, vMaskCurr);

        aSrcv[vRowIter][vSb * 4] = v0;
        aSrcv[vRowIter][vSb * 4 + 1] = v1;
        aSrcv[vRowIter][vSb * 4 + 2] = v2;
        aSrcv[vRowIter][vSb * 4 + 3] = v3;
      }

      __m128 vMax4 = _mm_max_ps(_mm256_extractf128_ps(vMaxAbs, 1), _mm256_castps256_ps128(vMaxAbs));
      vMax4 = _mm_max_ps(vMax4, _mm_movehl_ps(vMax4, vMax4));
      vMax4 = _mm_max_ss(vMax4, _mm_movehdup_ps(vMax4));
      const float cMaxScalar = _mm_cvtss_f32(vMax4);

      __m256 vMaxScalarVec = _mm256_set1_ps(cMaxScalar);

      __m256 vMaskNext = _mm256_cmp_ps(vMaxScalarVec, vMaxAbs, _CMP_EQ_OQ);
      __m256 vFinalMask = _mm256_and_ps(vMaskAbs, vMaskNext);

      const int cMask = _mm256_movemask_ps(vFinalMask);
      aIscale[vRowIter] = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f;

      if (cMask) {
        aIscale[vRowIter] = (cMaxScalar != 0.0f) ? -127.f / cMaxScalar : 0.0f;
      }

      vY[vI].d[vRowIter] = cMaxScalar ? 1 / aIscale[vRowIter] : 0;
      aIscaleVec[vRowIter] = _mm256_set1_ps(aIscale[vRowIter]);
    }

    __m256i aQuantsInterleaved[32];
    for (int vJ = 0; vJ < 32; vJ++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(aSrcv[0][vJ], aIscaleVec[0]);
      __m256 v1 = _mm256_mul_ps(aSrcv[1][vJ], aIscaleVec[1]);
      __m256 v2 = _mm256_mul_ps(aSrcv[2][vJ], aIscaleVec[2]);
      __m256 v3 = _mm256_mul_ps(aSrcv[3][vJ], aIscaleVec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i vI0 = _mm256_cvtps_epi32(v0);
      __m256i vI1 = _mm256_cvtps_epi32(v1);
      __m256i vI2 = _mm256_cvtps_epi32(v2);
      __m256i vI3 = _mm256_cvtps_epi32(v3);

      // Convert int32 to int16
      vI0 = _mm256_packs_epi32(vI0, vI1);
      vI2 = _mm256_packs_epi32(vI2, vI3);
      // Convert int16 to int8
      vI0 = _mm256_packs_epi16(vI0, vI2);

      //  Permute and store the quantized weights in the required order after the pack instruction
      const __m256i cPerm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      vI0 = _mm256_permutevar8x32_epi32(vI0, cPerm);

      _mm256_storeu_si256((__m256i *)(vY[vI].qs + 32 * vJ), vI0);
      aQuantsInterleaved[vJ] = vI0;
    }

    // Masks to shuffle the quants of corresponding sub blocks for rearranging quants for vectorized bsums computation
    __m256i vShuffleMaskSb2 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
    vShuffleMaskSb2 = _mm256_permute2f128_si256(vShuffleMaskSb2, vShuffleMaskSb2, 0);
    __m256i vShuffleMaskSb3 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
    vShuffleMaskSb3 = _mm256_permute2f128_si256(vShuffleMaskSb3, vShuffleMaskSb3, 0);
    __m256i vShuffleMaskSb4 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
    vShuffleMaskSb4 = _mm256_permute2f128_si256(vShuffleMaskSb4, vShuffleMaskSb4, 0);

    for (int k = 0; k < 4; k++) {
      // Quants from four different sub blocks are taken
      __m256i vQ0 = aQuantsInterleaved[k * 8 + 0];
      __m256i vQ1 = aQuantsInterleaved[k * 8 + 1];
      __m256i vQ2 = aQuantsInterleaved[k * 8 + 2];
      __m256i vQ3 = aQuantsInterleaved[k * 8 + 3];
      __m256i vQ4 = aQuantsInterleaved[k * 8 + 4];
      __m256i vQ5 = aQuantsInterleaved[k * 8 + 5];
      __m256i vQ6 = aQuantsInterleaved[k * 8 + 6];
      __m256i vQ7 = aQuantsInterleaved[k * 8 + 7];

      // The below code block has the first half of different sub blocks shuffled and blended so as to process 2 values
      // from each sub block at a time
      __m256i vSb2H1Shuffled = _mm256_shuffle_epi8(vQ2, vShuffleMaskSb2);
      __m256i vSbH1Interleaved = _mm256_blend_epi16(vQ0, vSb2H1Shuffled, 34);
      __m256i vSb3H1Shuffled = _mm256_shuffle_epi8(vQ4, vShuffleMaskSb3);
      vSbH1Interleaved = _mm256_blend_epi16(vSbH1Interleaved, vSb3H1Shuffled, 68);
      __m256i vSb4H1Shuffled = _mm256_shuffle_epi8(vQ6, vShuffleMaskSb4);
      vSbH1Interleaved = _mm256_blend_epi16(vSbH1Interleaved, vSb4H1Shuffled, 136);

      __m256i vOne = _mm256_set1_epi8(1);
      __m256i vBsumsR1 = _mm256_maddubs_epi16(vOne, vSbH1Interleaved);

      for (int vL = 0; vL < 3; vL++) {
        // Quants value shifted to process next two values from each sub block
        vQ0 = _mm256_srli_epi64(vQ0, 16);
        vQ2 = _mm256_srli_epi64(vQ2, 16);
        vQ4 = _mm256_srli_epi64(vQ4, 16);
        vQ6 = _mm256_srli_epi64(vQ6, 16);

        vSb2H1Shuffled = _mm256_shuffle_epi8(vQ2, vShuffleMaskSb2);
        vSbH1Interleaved = _mm256_blend_epi16(vQ0, vSb2H1Shuffled, 34);
        vSb3H1Shuffled = _mm256_shuffle_epi8(vQ4, vShuffleMaskSb3);
        vSbH1Interleaved = _mm256_blend_epi16(vSbH1Interleaved, vSb3H1Shuffled, 68);
        vSb4H1Shuffled = _mm256_shuffle_epi8(vQ6, vShuffleMaskSb4);
        vSbH1Interleaved = _mm256_blend_epi16(vSbH1Interleaved, vSb4H1Shuffled, 136);

        vBsumsR1 = _mm256_add_epi16(vBsumsR1, _mm256_maddubs_epi16(vOne, vSbH1Interleaved));
      }

      // The below code block has the second half of different sub blocks shuffled and blended so as to process 2 values
      // from each sub block at a time
      __m256i vSb2H2Shuffled = _mm256_shuffle_epi8(vQ3, vShuffleMaskSb2);
      __m256i vSbH2Interleaved = _mm256_blend_epi16(vQ1, vSb2H2Shuffled, 34);
      __m256i vSb3H2Shuffled = _mm256_shuffle_epi8(vQ5, vShuffleMaskSb3);
      vSbH2Interleaved = _mm256_blend_epi16(vSbH2Interleaved, vSb3H2Shuffled, 68);
      __m256i vSb4H2Shuffled = _mm256_shuffle_epi8(vQ7, vShuffleMaskSb4);
      vSbH2Interleaved = _mm256_blend_epi16(vSbH2Interleaved, vSb4H2Shuffled, 136);

      __m256i vBsumsR2 = _mm256_maddubs_epi16(vOne, vSbH2Interleaved);

      for (int vL = 0; vL < 3; vL++) {
        // Quants value shifted to process next two values from each sub block
        vQ1 = _mm256_srli_epi64(vQ1, 16);
        vQ3 = _mm256_srli_epi64(vQ3, 16);
        vQ5 = _mm256_srli_epi64(vQ5, 16);
        vQ7 = _mm256_srli_epi64(vQ7, 16);

        vSb2H2Shuffled = _mm256_shuffle_epi8(vQ3, vShuffleMaskSb2);
        vSbH2Interleaved = _mm256_blend_epi16(vQ1, vSb2H2Shuffled, 34);
        vSb3H2Shuffled = _mm256_shuffle_epi8(vQ5, vShuffleMaskSb3);
        vSbH2Interleaved = _mm256_blend_epi16(vSbH2Interleaved, vSb3H2Shuffled, 68);
        vSb4H2Shuffled = _mm256_shuffle_epi8(vQ7, vShuffleMaskSb4);
        vSbH2Interleaved = _mm256_blend_epi16(vSbH2Interleaved, vSb4H2Shuffled, 136);

        vBsumsR2 = _mm256_add_epi16(vBsumsR2, _mm256_maddubs_epi16(vOne, vSbH2Interleaved));
      }

      // Overall bsums in interleaved fashion computed by adding results of both halves
      __m256i vBsumsR = _mm256_add_epi16(vBsumsR1, vBsumsR2);
      _mm256_storeu_si256((__m256i *)(vY[vI].bsums + 16 * k), vBsumsR);
    }
  }

#else
  UNUSED(cNb);
  UNUSED(vY);
  ggml_quantize_mat_q8_K_4x8_generic(pX, pVy, k);
#endif
}

//
// GEMV/GEMM templates
//

#if defined(__AVX2__) || defined(__AVX512F__)

// GEMV for 8x blocks of 32 4-bit quants with a single scale factor per block
template <typename block_tx8>
static void fGemvQ4B328x8Q80LutAvx(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                   const void *GGML_RESTRICT pVy, int pNr, int pNc, __m256i pSignextendlut) {
  static_assert(std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8> ||
                    std::is_same_v<block_tx8, block_mxfp4x8>,
                "Unsupported block type");

  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  UNUSED(pBs);

  __m256i vFinalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Permute mask used for easier vector processing at later stages
  const __m256i cM4b = _mm256_set1_epi8(0x0F);

  int64_t vBNb = pN / 32;

  const block_tx8 *vBPtrStart = (const block_tx8 *)pVx;
  const block_q8_0 *vAPtrStart = (const block_q8_0 *)pVy;

  // Process Q8_0 blocks one by one
  for (int64_t vY = 0; vY < pNr; vY++) {

    // Pointers to LHS blocks of block_q8_0 format
    const block_q8_0 *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight blocks at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < pNc / 8; vX++) {

      // Pointers to RHS blocks
      const block_tx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulator
      __m256 vAccRow = _mm256_setzero_ps();

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Load 8 blocks of 32 interleaved as 8 bytes (B0 - B7)
        const __m256i cRhsRawVec01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs));
        const __m256i cRhsRawVec45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs) + 1);
        const __m256i cRhsRawVec01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs) + 2);
        const __m256i cRhsRawVec45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs) + 3);

        // 4-bit -> 8-bit - Sign is maintained
        const __m256i cRhsVec01230 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawVec01230, cM4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
        const __m256i cRhsVec45670 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawVec45670, cM4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
        const __m256i cRhsVec01231 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawVec01231, cM4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
        const __m256i cRhsVec45671 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawVec45671, cM4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

        const __m256i cRhsVec01232 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01230, 4),
                                                                 cM4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
        const __m256i cRhsVec45672 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45670, 4),
                                                                 cM4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
        const __m256i cRhsVec01233 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01231, 4),
                                                                 cM4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
        const __m256i cRhsVec45673 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45671, 4),
                                                                 cM4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

        // Load the scale values for the 8 blocks interleaved in block_tx8
        __m256 vColScaleF32;
        if constexpr (std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8>) {
          const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
          vColScaleF32 = GGML_F32Cx8_REARRANGE_LOAD(vBPtr[vB].d, changemask);
        } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
          // Load 8 E8M0 exponents and convert to float via LUT
          // Rearranged to match changemask order: 0,4,1,5,2,6,3,7
          vColScaleF32 =
              _mm256_set_ps(GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[3]),
                            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[6]), GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[1]),
                            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[4]), GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vB].e[0]));
        }

        // Load and convert to FP32 scale from block_q8_0
        const __m256 cRowScaleF32 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vAPtr[vB].d));

        // Load the block values in block_q8_0 in batches of 16 bytes and replicate the same across 256 bit vector
        __m256i vLhsVec0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)vAPtr[vB].qs));
        __m256i vLhsVec1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 16)));

        vLhsVec0 = _mm256_permute2f128_si256(vLhsVec0, vLhsVec0, 0); // A0 (0-15) A0(0-15)
        vLhsVec1 = _mm256_permute2f128_si256(vLhsVec1, vLhsVec1, 0); // A0 (16-31) A0(16-31))

        __m256i vIacc = _mm256_setzero_si256();

        // Dot product done within 32 bit lanes and accumulated in the same vector
        // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
        // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
        // ...........................................................................
        // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(cRhsVec01230, _mm256_shuffle_epi32(cRhsVec45670, 177), 170),
                                         _mm256_shuffle_epi32(vLhsVec0, 0));
        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec01230, 177), cRhsVec45670, 170),
                                         _mm256_shuffle_epi32(vLhsVec0, 85));

        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(cRhsVec01231, _mm256_shuffle_epi32(cRhsVec45671, 177), 170),
                                         _mm256_shuffle_epi32(vLhsVec0, 170));
        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec01231, 177), cRhsVec45671, 170),
                                         _mm256_shuffle_epi32(vLhsVec0, 255));

        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(cRhsVec01232, _mm256_shuffle_epi32(cRhsVec45672, 177), 170),
                                         _mm256_shuffle_epi32(vLhsVec1, 0));
        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec01232, 177), cRhsVec45672, 170),
                                         _mm256_shuffle_epi32(vLhsVec1, 85));

        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(cRhsVec01233, _mm256_shuffle_epi32(cRhsVec45673, 177), 170),
                                         _mm256_shuffle_epi32(vLhsVec1, 170));
        vIacc = fMulSumI8PairsAccInt32x8(vIacc,
                                         _mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec01233, 177), cRhsVec45673, 170),
                                         _mm256_shuffle_epi32(vLhsVec1, 255));

        // Accumulated values multiplied with appropriate scales
        vAccRow = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIacc), _mm256_mul_ps(vColScaleF32, cRowScaleF32), vAccRow);
      }

      // Accumulated output values permuted so as to be stored in appropriate order post accumulation
      vAccRow = _mm256_permutevar8x32_ps(vAccRow, vFinalpermutemask);
      _mm256_storeu_ps(pS + (vY * pNr + vX * 8), vAccRow);
    }
  }
}

// GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block
template <typename block_tx8>
static void fGemmQ4B328x8Q80LutAvx(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                   const void *GGML_RESTRICT pVy, int pNr, int pNc, __m256i pSignextendlut) {
  static_assert(std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8> ||
                    std::is_same_v<block_tx8, block_mxfp4x8>,
                "Unsupported block type");

  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  const block_tx8 *vBPtrStart = (const block_tx8 *)pVx;
  const block_q8_0x4 *vAPtrStart = (const block_q8_0x4 *)pVy;

  int64_t vBNb = pN / 32;
  int64_t vY = 0;
  // Mask to mask out nibbles from packed bytes
  const __m256i cM4b = _mm256_set1_epi8(0x0F);
  const __m128i cLoadMask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
  // Permute mask used for easier vector processing at later stages
  __m256i vRequiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  int64_t vXstart = 0;
  int vAnr = pNr - pNr % 16; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
  int vAnc = pNc - pNc % 16; // Used to align nc with boundary of 16
                             // Mask to mask out nibbles from packed bytes expanded to 512 bit length
  const __m512i cM4bexpanded = _mm512_set1_epi8(0x0F);
  // Lookup table to convert signed nibbles to signed bytes expanded to 512 bit length
  __m512i vSignextendlutexpanded = _mm512_inserti32x8(_mm512_castsi256_si512(pSignextendlut), pSignextendlut, 1);

  // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation
  for (; vY < vAnr / 4; vY += 4) {

    const block_q8_0x4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_tx8 *b_ptr_0 = vBPtrStart + ((vX)*vBNb);
      const block_tx8 *b_ptr_1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
        const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs));
        const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 32));
        const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 64));
        const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 96));

        const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs));
        const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 32));
        const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 64));
        const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 96));

        // Save the values in the following vectors in the formats B0B1B4B5B8B9BCBD, B2B3B6B7BABBBEBF for further
        // processing and storing of values
        const __m256i cRhsRawMat01450 =
            _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
        const __m256i cRhsRawMat23670 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
        const __m256i cRhsRawMat01451 =
            _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
        const __m256i cRhsRawMat23671 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

        const __m256i cRhsRawMat89Cd0 =
            _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
        const __m256i cRhsRawMatAbef0 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
        const __m256i cRhsRawMat89Cd1 =
            _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
        const __m256i cRhsRawMatAbef1 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);

        const __m512i cRhsRawMat014589Cd0 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
        const __m512i cRhsRawMat2367Abef0 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
        const __m512i cRhsRawMat014589Cd1 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
        const __m512i cRhsRawMat2367Abef1 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

        // 4-bit -> 8-bit - Sign is maintained
        const __m512i cRhsMat014589Cd0 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat014589Cd0,
                             cM4bexpanded)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
        const __m512i cRhsMat2367Abef0 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat2367Abef0,
                             cM4bexpanded)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

        const __m512i cRhsMat014589Cd1 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat014589Cd1,
                             cM4bexpanded)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
        const __m512i cRhsMat2367Abef1 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat2367Abef1,
                             cM4bexpanded)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

        const __m512i cRhsMat014589Cd2 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                cM4bexpanded)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
        const __m512i cRhsMat2367Abef2 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                cM4bexpanded)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

        const __m512i cRhsMat014589Cd3 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
                cM4bexpanded)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
        const __m512i cRhsMat2367Abef3 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
                cM4bexpanded)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

        // Shuffle pattern one - right side input
        const __m512i cRhsMat014589Cd0Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd0, (_MM_PERM_ENUM)136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                                                   // B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
        const __m512i cRhsMat2367Abef0Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef0, (_MM_PERM_ENUM)136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)
                                                   // BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

        const __m512i cRhsMat014589Cd1Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd1,
            (_MM_PERM_ENUM)136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11)
                                 // B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
        const __m512i cRhsMat2367Abef1Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef1,
            (_MM_PERM_ENUM)136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11)
                                 // BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

        const __m512i cRhsMat014589Cd2Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd2,
            (_MM_PERM_ENUM)136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                                 // B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
        const __m512i cRhsMat2367Abef2Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef2,
            (_MM_PERM_ENUM)136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)
                                 // BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

        const __m512i cRhsMat014589Cd3Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd3,
            (_MM_PERM_ENUM)136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                                 // B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
        const __m512i cRhsMat2367Abef3Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef3,
            (_MM_PERM_ENUM)136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)
                                 // BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

        // Shuffle pattern two - right side input

        const __m512i cRhsMat014589Cd0Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd0, (_MM_PERM_ENUM)221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                                                   // B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
        const __m512i cRhsMat2367Abef0Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef0, (_MM_PERM_ENUM)221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)
                                                   // BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

        const __m512i cRhsMat014589Cd1Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd1,
            (_MM_PERM_ENUM)221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                                 // B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
        const __m512i cRhsMat2367Abef1Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef1,
            (_MM_PERM_ENUM)221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)
                                 // BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

        const __m512i cRhsMat014589Cd2Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd2,
            (_MM_PERM_ENUM)221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                                 // B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
        const __m512i cRhsMat2367Abef2Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef2,
            (_MM_PERM_ENUM)221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)
                                 // BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

        const __m512i cRhsMat014589Cd3Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd3,
            (_MM_PERM_ENUM)221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                                 // B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
        const __m512i cRhsMat2367Abef3Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef3,
            (_MM_PERM_ENUM)221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)
                                 // BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)

        // Scale values - Load the weight scale values of two block_tx8
        __m512 vColScaleF32;
        if constexpr (std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8>) {
          vColScaleF32 = GGML_F32Cx8x2_LOAD(b_ptr_0[vB].d, b_ptr_1[vB].d);
        } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
          // TODO: simd-ify
          vColScaleF32 =
              _mm512_set_ps(GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[0]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[0]));
        }

        // Process LHS in pairs of rows
        for (int vRp = 0; vRp < 4; vRp++) {

          // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
          // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into
          // 512 bit vector
          __m256i vLhsMatYmm01230 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs)));
          __m256i vLhsMatYmm010 = _mm256_permute2f128_si256(vLhsMatYmm01230, vLhsMatYmm01230, 0);
          __m256i vLhsMatYmm230 = _mm256_permute2f128_si256(vLhsMatYmm01230, vLhsMatYmm01230, 17);
          __m256i vLhsMatYmm01231 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32)));
          __m256i vLhsMatYmm011 = _mm256_permute2f128_si256(vLhsMatYmm01231, vLhsMatYmm01231, 0);
          __m256i vLhsMatYmm231 = _mm256_permute2f128_si256(vLhsMatYmm01231, vLhsMatYmm01231, 17);
          __m256i vLhsMatYmm01232 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64)));
          __m256i vLhsMatYmm012 = _mm256_permute2f128_si256(vLhsMatYmm01232, vLhsMatYmm01232, 0);
          __m256i vLhsMatYmm232 = _mm256_permute2f128_si256(vLhsMatYmm01232, vLhsMatYmm01232, 17);
          __m256i vLhsMatYmm01233 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96)));
          __m256i vLhsMatYmm013 = _mm256_permute2f128_si256(vLhsMatYmm01233, vLhsMatYmm01233, 0);
          __m256i vLhsMatYmm233 = _mm256_permute2f128_si256(vLhsMatYmm01233, vLhsMatYmm01233, 17);

          __m512i vLhsMat010 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm010), vLhsMatYmm010, 1);
          __m512i vLhsMat230 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm230), vLhsMatYmm230, 1);
          __m512i vLhsMat011 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm011), vLhsMatYmm011, 1);
          __m512i vLhsMat231 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm231), vLhsMatYmm231, 1);
          __m512i vLhsMat012 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm012), vLhsMatYmm012, 1);
          __m512i vLhsMat232 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm232), vLhsMatYmm232, 1);
          __m512i vLhsMat013 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm013), vLhsMatYmm013, 1);
          __m512i vLhsMat233 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm233), vLhsMatYmm233, 1);

          // Shuffle pattern one - left side input

          const __m512i cLhsMat010Sp1 = _mm512_shuffle_epi32(
              vLhsMat010, (_MM_PERM_ENUM)160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                                               // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
          const __m512i cLhsMat230Sp1 = _mm512_shuffle_epi32(
              vLhsMat230, (_MM_PERM_ENUM)160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)
                                               // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

          const __m512i cLhsMat011Sp1 = _mm512_shuffle_epi32(
              vLhsMat011,
              (_MM_PERM_ENUM)160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11)
                                   // A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
          const __m512i cLhsMat231Sp1 = _mm512_shuffle_epi32(
              vLhsMat231,
              (_MM_PERM_ENUM)160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11)
                                   // A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

          const __m512i cLhsMat012Sp1 = _mm512_shuffle_epi32(
              vLhsMat012,
              (_MM_PERM_ENUM)160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                   // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
          const __m512i cLhsMat232Sp1 = _mm512_shuffle_epi32(
              vLhsMat232,
              (_MM_PERM_ENUM)160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                   // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

          const __m512i cLhsMat013Sp1 = _mm512_shuffle_epi32(
              vLhsMat013,
              (_MM_PERM_ENUM)160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                   // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
          const __m512i cLhsMat233Sp1 = _mm512_shuffle_epi32(
              vLhsMat233,
              (_MM_PERM_ENUM)160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                   // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

          // Shuffle pattern two - left side input

          const __m512i cLhsMat010Sp2 = _mm512_shuffle_epi32(
              vLhsMat010, (_MM_PERM_ENUM)245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                                               // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
          const __m512i cLhsMat230Sp2 = _mm512_shuffle_epi32(
              vLhsMat230, (_MM_PERM_ENUM)245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)
                                               // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

          const __m512i cLhsMat011Sp2 = _mm512_shuffle_epi32(
              vLhsMat011,
              (_MM_PERM_ENUM)245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                   // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
          const __m512i cLhsMat231Sp2 = _mm512_shuffle_epi32(
              vLhsMat231,
              (_MM_PERM_ENUM)245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                   // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

          const __m512i cLhsMat012Sp2 = _mm512_shuffle_epi32(
              vLhsMat012,
              (_MM_PERM_ENUM)245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                   // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
          const __m512i cLhsMat232Sp2 = _mm512_shuffle_epi32(
              vLhsMat232,
              (_MM_PERM_ENUM)245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                   // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

          const __m512i cLhsMat013Sp2 = _mm512_shuffle_epi32(
              vLhsMat013,
              (_MM_PERM_ENUM)245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                   // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
          const __m512i cLhsMat233Sp2 = _mm512_shuffle_epi32(
              vLhsMat233,
              (_MM_PERM_ENUM)245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                   // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane Resembles MMLAs into
          // 2x2 matrices in ARM Version
          const __m512i cZero = _mm512_setzero_epi32();
          __m512i vIaccMat00Sp1 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp1, cRhsMat014589Cd3Sp1),
                                            cLhsMat012Sp1, cRhsMat014589Cd2Sp1),
                  cLhsMat011Sp1, cRhsMat014589Cd1Sp1),
              cLhsMat010Sp1, cRhsMat014589Cd0Sp1);
          __m512i vIaccMat01Sp1 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp1, cRhsMat2367Abef3Sp1),
                                            cLhsMat012Sp1, cRhsMat2367Abef2Sp1),
                  cLhsMat011Sp1, cRhsMat2367Abef1Sp1),
              cLhsMat010Sp1, cRhsMat2367Abef0Sp1);
          __m512i vIaccMat10Sp1 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp1, cRhsMat014589Cd3Sp1),
                                            cLhsMat232Sp1, cRhsMat014589Cd2Sp1),
                  cLhsMat231Sp1, cRhsMat014589Cd1Sp1),
              cLhsMat230Sp1, cRhsMat014589Cd0Sp1);
          __m512i vIaccMat11Sp1 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp1, cRhsMat2367Abef3Sp1),
                                            cLhsMat232Sp1, cRhsMat2367Abef2Sp1),
                  cLhsMat231Sp1, cRhsMat2367Abef1Sp1),
              cLhsMat230Sp1, cRhsMat2367Abef0Sp1);
          __m512i vIaccMat00Sp2 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp2, cRhsMat014589Cd3Sp2),
                                            cLhsMat012Sp2, cRhsMat014589Cd2Sp2),
                  cLhsMat011Sp2, cRhsMat014589Cd1Sp2),
              cLhsMat010Sp2, cRhsMat014589Cd0Sp2);
          __m512i vIaccMat01Sp2 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp2, cRhsMat2367Abef3Sp2),
                                            cLhsMat012Sp2, cRhsMat2367Abef2Sp2),
                  cLhsMat011Sp2, cRhsMat2367Abef1Sp2),
              cLhsMat010Sp2, cRhsMat2367Abef0Sp2);
          __m512i vIaccMat10Sp2 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp2, cRhsMat014589Cd3Sp2),
                                            cLhsMat232Sp2, cRhsMat014589Cd2Sp2),
                  cLhsMat231Sp2, cRhsMat014589Cd1Sp2),
              cLhsMat230Sp2, cRhsMat014589Cd0Sp2);
          __m512i vIaccMat11Sp2 = fMulSumI8PairsAccInt32x16(
              fMulSumI8PairsAccInt32x16(
                  fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp2, cRhsMat2367Abef3Sp2),
                                            cLhsMat232Sp2, cRhsMat2367Abef2Sp2),
                  cLhsMat231Sp2, cRhsMat2367Abef1Sp2),
              cLhsMat230Sp2, cRhsMat2367Abef0Sp2);

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          __m512i vIaccMat00 = _mm512_add_epi32(vIaccMat00Sp1, vIaccMat00Sp2);
          __m512i vIaccMat01 = _mm512_add_epi32(vIaccMat01Sp1, vIaccMat01Sp2);
          __m512i vIaccMat10 = _mm512_add_epi32(vIaccMat10Sp1, vIaccMat10Sp2);
          __m512i vIaccMat11 = _mm512_add_epi32(vIaccMat11Sp1, vIaccMat11Sp2);

          // Straighten out to make 4 row vectors
          __m512i vIaccRow0 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat00, _mm512_shuffle_epi32(vIaccMat01, (_MM_PERM_ENUM)78));
          __m512i vIaccRow1 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat00, (_MM_PERM_ENUM)78), vIaccMat01);
          __m512i vIaccRow2 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat10, _mm512_shuffle_epi32(vIaccMat11, (_MM_PERM_ENUM)78));
          __m512i vIaccRow3 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat10, (_MM_PERM_ENUM)78), vIaccMat11);

          // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
          const __m128i cRowScaleF16 =
              _mm_shuffle_epi32(_mm_maskload_epi32((int const *)(vAPtrs[vRp][vB].d), cLoadMask), 68);
          const __m512 cRowScaleF32 = GGML_F32Cx16_REPEAT_LOAD(cRowScaleF16);

          // Multiply with appropriate scales and accumulate
          aAccRows[vRp * 4] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRow0),
              _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
          aAccRows[vRp * 4 + 1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRow1),
              _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
          aAccRows[vRp * 4 + 2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRow2),
              _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
          aAccRows[vRp * 4 + 3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRow3),
              _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);
        }
      }

      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), aAccRows[vI]);
      }
    }
  }

  // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
  for (; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_tx8 *b_ptr_0 = vBPtrStart + ((vX)*vBNb);
      const block_tx8 *b_ptr_1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
        const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs));
        const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 32));
        const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 64));
        const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[vB].qs + 96));

        const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs));
        const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 32));
        const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 64));
        const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[vB].qs + 96));

        // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing
        // of values
        const __m256i cRhsRawMat01450 =
            _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
        const __m256i cRhsRawMat23670 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
        const __m256i cRhsRawMat01451 =
            _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
        const __m256i cRhsRawMat23671 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

        const __m256i cRhsRawMat89Cd0 =
            _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
        const __m256i cRhsRawMatAbef0 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
        const __m256i cRhsRawMat89Cd1 =
            _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
        const __m256i cRhsRawMatAbef1 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);

        const __m512i cRhsRawMat014589Cd0 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
        const __m512i cRhsRawMat2367Abef0 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
        const __m512i cRhsRawMat014589Cd1 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
        const __m512i cRhsRawMat2367Abef1 =
            _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

        // 4-bit -> 8-bit - Sign is maintained
        const __m512i cRhsMat014589Cd0 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat014589Cd0,
                             cM4bexpanded)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
        const __m512i cRhsMat2367Abef0 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat2367Abef0,
                             cM4bexpanded)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

        const __m512i cRhsMat014589Cd1 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat014589Cd1,
                             cM4bexpanded)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
        const __m512i cRhsMat2367Abef1 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(cRhsRawMat2367Abef1,
                             cM4bexpanded)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

        const __m512i cRhsMat014589Cd2 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                cM4bexpanded)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
        const __m512i cRhsMat2367Abef2 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                cM4bexpanded)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

        const __m512i cRhsMat014589Cd3 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
                cM4bexpanded)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
        const __m512i cRhsMat2367Abef3 = _mm512_shuffle_epi8(
            vSignextendlutexpanded,
            _mm512_and_si512(
                _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
                cM4bexpanded)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

        // Shuffle pattern one - right side input
        const __m512i cRhsMat014589Cd0Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd0, (_MM_PERM_ENUM)136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                                                   // B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
        const __m512i cRhsMat2367Abef0Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef0, (_MM_PERM_ENUM)136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)
                                                   // BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

        const __m512i cRhsMat014589Cd1Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd1,
            (_MM_PERM_ENUM)136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11)
                                 // B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
        const __m512i cRhsMat2367Abef1Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef1,
            (_MM_PERM_ENUM)136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11)
                                 // BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

        const __m512i cRhsMat014589Cd2Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd2,
            (_MM_PERM_ENUM)136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                                 // B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
        const __m512i cRhsMat2367Abef2Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef2,
            (_MM_PERM_ENUM)136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)
                                 // BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

        const __m512i cRhsMat014589Cd3Sp1 = _mm512_shuffle_epi32(
            cRhsMat014589Cd3,
            (_MM_PERM_ENUM)136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                                 // B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
        const __m512i cRhsMat2367Abef3Sp1 = _mm512_shuffle_epi32(
            cRhsMat2367Abef3,
            (_MM_PERM_ENUM)136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)
                                 // BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

        // Shuffle pattern two - right side input

        const __m512i cRhsMat014589Cd0Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd0, (_MM_PERM_ENUM)221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                                                   // B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
        const __m512i cRhsMat2367Abef0Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef0, (_MM_PERM_ENUM)221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)
                                                   // BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

        const __m512i cRhsMat014589Cd1Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd1,
            (_MM_PERM_ENUM)221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                                 // B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
        const __m512i cRhsMat2367Abef1Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef1,
            (_MM_PERM_ENUM)221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)
                                 // BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

        const __m512i cRhsMat014589Cd2Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd2,
            (_MM_PERM_ENUM)221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                                 // B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
        const __m512i cRhsMat2367Abef2Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef2,
            (_MM_PERM_ENUM)221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)
                                 // BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

        const __m512i cRhsMat014589Cd3Sp2 = _mm512_shuffle_epi32(
            cRhsMat014589Cd3,
            (_MM_PERM_ENUM)221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                                 // B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
        const __m512i cRhsMat2367Abef3Sp2 = _mm512_shuffle_epi32(
            cRhsMat2367Abef3,
            (_MM_PERM_ENUM)221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)
                                 // BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)

        // Scale values - Load the weight scale values of two block_tx8
        __m512 vColScaleF32;
        if constexpr (std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8>) {
          vColScaleF32 = GGML_F32Cx8x2_LOAD(b_ptr_0[vB].d, b_ptr_1[vB].d);
        } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
          // TODO: simd-ify
          vColScaleF32 =
              _mm512_set_ps(GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[vB].e[0]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[vB].e[0]));
        }

        // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into
        // 512 bit vector
        __m256i vLhsMatYmm01230 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs)));
        __m256i vLhsMatYmm010 = _mm256_permute2f128_si256(vLhsMatYmm01230, vLhsMatYmm01230, 0);
        __m256i vLhsMatYmm230 = _mm256_permute2f128_si256(vLhsMatYmm01230, vLhsMatYmm01230, 17);
        __m256i vLhsMatYmm01231 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32)));
        __m256i vLhsMatYmm011 = _mm256_permute2f128_si256(vLhsMatYmm01231, vLhsMatYmm01231, 0);
        __m256i vLhsMatYmm231 = _mm256_permute2f128_si256(vLhsMatYmm01231, vLhsMatYmm01231, 17);
        __m256i vLhsMatYmm01232 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64)));
        __m256i vLhsMatYmm012 = _mm256_permute2f128_si256(vLhsMatYmm01232, vLhsMatYmm01232, 0);
        __m256i vLhsMatYmm232 = _mm256_permute2f128_si256(vLhsMatYmm01232, vLhsMatYmm01232, 17);
        __m256i vLhsMatYmm01233 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96)));
        __m256i vLhsMatYmm013 = _mm256_permute2f128_si256(vLhsMatYmm01233, vLhsMatYmm01233, 0);
        __m256i vLhsMatYmm233 = _mm256_permute2f128_si256(vLhsMatYmm01233, vLhsMatYmm01233, 17);

        __m512i vLhsMat010 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm010), vLhsMatYmm010, 1);
        __m512i vLhsMat230 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm230), vLhsMatYmm230, 1);
        __m512i vLhsMat011 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm011), vLhsMatYmm011, 1);
        __m512i vLhsMat231 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm231), vLhsMatYmm231, 1);
        __m512i vLhsMat012 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm012), vLhsMatYmm012, 1);
        __m512i vLhsMat232 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm232), vLhsMatYmm232, 1);
        __m512i vLhsMat013 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm013), vLhsMatYmm013, 1);
        __m512i vLhsMat233 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm233), vLhsMatYmm233, 1);

        // Shuffle pattern one - left side input

        const __m512i cLhsMat010Sp1 = _mm512_shuffle_epi32(
            vLhsMat010, (_MM_PERM_ENUM)160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                             // A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
        const __m512i cLhsMat230Sp1 = _mm512_shuffle_epi32(
            vLhsMat230, (_MM_PERM_ENUM)160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                             // A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

        const __m512i cLhsMat011Sp1 = _mm512_shuffle_epi32(
            vLhsMat011, (_MM_PERM_ENUM)160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                             // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
        const __m512i cLhsMat231Sp1 = _mm512_shuffle_epi32(
            vLhsMat231, (_MM_PERM_ENUM)160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                             // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

        const __m512i cLhsMat012Sp1 = _mm512_shuffle_epi32(
            vLhsMat012,
            (_MM_PERM_ENUM)160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                 // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
        const __m512i cLhsMat232Sp1 = _mm512_shuffle_epi32(
            vLhsMat232,
            (_MM_PERM_ENUM)160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                 // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

        const __m512i cLhsMat013Sp1 = _mm512_shuffle_epi32(
            vLhsMat013,
            (_MM_PERM_ENUM)160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                 // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
        const __m512i cLhsMat233Sp1 = _mm512_shuffle_epi32(
            vLhsMat233,
            (_MM_PERM_ENUM)160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                 // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

        // Shuffle pattern two - left side input

        const __m512i cLhsMat010Sp2 = _mm512_shuffle_epi32(
            vLhsMat010, (_MM_PERM_ENUM)245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                             // A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
        const __m512i cLhsMat230Sp2 = _mm512_shuffle_epi32(
            vLhsMat230, (_MM_PERM_ENUM)245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                             // A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

        const __m512i cLhsMat011Sp2 = _mm512_shuffle_epi32(
            vLhsMat011,
            (_MM_PERM_ENUM)245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                 // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
        const __m512i cLhsMat231Sp2 = _mm512_shuffle_epi32(
            vLhsMat231,
            (_MM_PERM_ENUM)245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                 // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

        const __m512i cLhsMat012Sp2 = _mm512_shuffle_epi32(
            vLhsMat012,
            (_MM_PERM_ENUM)245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                 // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
        const __m512i cLhsMat232Sp2 = _mm512_shuffle_epi32(
            vLhsMat232,
            (_MM_PERM_ENUM)245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                 // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

        const __m512i cLhsMat013Sp2 = _mm512_shuffle_epi32(
            vLhsMat013,
            (_MM_PERM_ENUM)245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                 // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
        const __m512i cLhsMat233Sp2 = _mm512_shuffle_epi32(
            vLhsMat233,
            (_MM_PERM_ENUM)245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                 // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
        // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane Resembles MMLAs into 2x2
        // matrices in ARM Version
        const __m512i cZero = _mm512_setzero_epi32();
        __m512i vIaccMat00Sp1 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp1, cRhsMat014589Cd3Sp1),
                                          cLhsMat012Sp1, cRhsMat014589Cd2Sp1),
                cLhsMat011Sp1, cRhsMat014589Cd1Sp1),
            cLhsMat010Sp1, cRhsMat014589Cd0Sp1);
        __m512i vIaccMat01Sp1 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp1, cRhsMat2367Abef3Sp1),
                                          cLhsMat012Sp1, cRhsMat2367Abef2Sp1),
                cLhsMat011Sp1, cRhsMat2367Abef1Sp1),
            cLhsMat010Sp1, cRhsMat2367Abef0Sp1);
        __m512i vIaccMat10Sp1 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp1, cRhsMat014589Cd3Sp1),
                                          cLhsMat232Sp1, cRhsMat014589Cd2Sp1),
                cLhsMat231Sp1, cRhsMat014589Cd1Sp1),
            cLhsMat230Sp1, cRhsMat014589Cd0Sp1);
        __m512i vIaccMat11Sp1 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp1, cRhsMat2367Abef3Sp1),
                                          cLhsMat232Sp1, cRhsMat2367Abef2Sp1),
                cLhsMat231Sp1, cRhsMat2367Abef1Sp1),
            cLhsMat230Sp1, cRhsMat2367Abef0Sp1);
        __m512i vIaccMat00Sp2 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp2, cRhsMat014589Cd3Sp2),
                                          cLhsMat012Sp2, cRhsMat014589Cd2Sp2),
                cLhsMat011Sp2, cRhsMat014589Cd1Sp2),
            cLhsMat010Sp2, cRhsMat014589Cd0Sp2);
        __m512i vIaccMat01Sp2 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat013Sp2, cRhsMat2367Abef3Sp2),
                                          cLhsMat012Sp2, cRhsMat2367Abef2Sp2),
                cLhsMat011Sp2, cRhsMat2367Abef1Sp2),
            cLhsMat010Sp2, cRhsMat2367Abef0Sp2);
        __m512i vIaccMat10Sp2 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp2, cRhsMat014589Cd3Sp2),
                                          cLhsMat232Sp2, cRhsMat014589Cd2Sp2),
                cLhsMat231Sp2, cRhsMat014589Cd1Sp2),
            cLhsMat230Sp2, cRhsMat014589Cd0Sp2);
        __m512i vIaccMat11Sp2 = fMulSumI8PairsAccInt32x16(
            fMulSumI8PairsAccInt32x16(
                fMulSumI8PairsAccInt32x16(fMulSumI8PairsAccInt32x16(cZero, cLhsMat233Sp2, cRhsMat2367Abef3Sp2),
                                          cLhsMat232Sp2, cRhsMat2367Abef2Sp2),
                cLhsMat231Sp2, cRhsMat2367Abef1Sp2),
            cLhsMat230Sp2, cRhsMat2367Abef0Sp2);

        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
        __m512i vIaccMat00 = _mm512_add_epi32(vIaccMat00Sp1, vIaccMat00Sp2);
        __m512i vIaccMat01 = _mm512_add_epi32(vIaccMat01Sp1, vIaccMat01Sp2);
        __m512i vIaccMat10 = _mm512_add_epi32(vIaccMat10Sp1, vIaccMat10Sp2);
        __m512i vIaccMat11 = _mm512_add_epi32(vIaccMat11Sp1, vIaccMat11Sp2);

        // Straighten out to make 4 row vectors
        __m512i vIaccRow0 =
            _mm512_mask_blend_epi32(0xCCCC, vIaccMat00, _mm512_shuffle_epi32(vIaccMat01, (_MM_PERM_ENUM)78));
        __m512i vIaccRow1 =
            _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat00, (_MM_PERM_ENUM)78), vIaccMat01);
        __m512i vIaccRow2 =
            _mm512_mask_blend_epi32(0xCCCC, vIaccMat10, _mm512_shuffle_epi32(vIaccMat11, (_MM_PERM_ENUM)78));
        __m512i vIaccRow3 =
            _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat10, (_MM_PERM_ENUM)78), vIaccMat11);

        // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
        const __m128i cRowScaleF16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const *)(vAPtr[vB].d), cLoadMask), 68);
        const __m512 cRowScaleF32 = GGML_F32Cx16_REPEAT_LOAD(cRowScaleF16);

        // Multiply with appropriate scales and accumulate
        aAccRows[0] =
            _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow0),
                            _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[0]);
        aAccRows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow1),
                                      _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                      aAccRows[1]);
        aAccRows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow2),
                                      _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                      aAccRows[2]);
        aAccRows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow3),
                                      _mm512_mul_ps(vColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                      aAccRows[3]);
      }

      // Store the accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), aAccRows[vI]);
      }
    }
  }
  if (vAnc != pNc) {
    vXstart = vAnc / 8;
    vY = 0;
  }
#endif // __AVX512BW__ && __AVX512DQ__

  // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation

  for (; vY < vAnr / 4; vY += 4) {
    const block_q8_0x4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of eight block_tx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {

      const block_tx8 *b_ptr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
        const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs));
        const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 32));
        const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 64));
        const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 96));

        // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing
        // of values
        const __m256i cRhsRawMat01450 =
            _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
        const __m256i cRhsRawMat23670 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
        const __m256i cRhsRawMat01451 =
            _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
        const __m256i cRhsRawMat23671 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

        // 4-bit -> 8-bit - Sign is maintained
        const __m256i cRhsMat01450 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat01450, cM4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
        const __m256i cRhsMat23670 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat23670, cM4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

        const __m256i cRhsMat01451 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat01451, cM4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
        const __m256i cRhsMat23671 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat23671, cM4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

        const __m256i cRhsMat01452 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4),
                                                                 cM4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
        const __m256i cRhsMat23672 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4),
                                                                 cM4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

        const __m256i cRhsMat01453 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4),
                                                                 cM4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
        const __m256i cRhsMat23673 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4),
                                                                 cM4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

        // Shuffle pattern one - right side input
        const __m256i cRhsMat01450Sp1 =
            _mm256_shuffle_epi32(cRhsMat01450, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
        const __m256i cRhsMat23670Sp1 =
            _mm256_shuffle_epi32(cRhsMat23670, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

        const __m256i cRhsMat01451Sp1 = _mm256_shuffle_epi32(
            cRhsMat01451, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
        const __m256i cRhsMat23671Sp1 = _mm256_shuffle_epi32(
            cRhsMat23671, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

        const __m256i cRhsMat01452Sp1 = _mm256_shuffle_epi32(
            cRhsMat01452, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
        const __m256i cRhsMat23672Sp1 = _mm256_shuffle_epi32(
            cRhsMat23672, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

        const __m256i cRhsMat01453Sp1 = _mm256_shuffle_epi32(
            cRhsMat01453, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
        const __m256i cRhsMat23673Sp1 = _mm256_shuffle_epi32(
            cRhsMat23673, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

        // Shuffle pattern two - right side input

        const __m256i cRhsMat01450Sp2 =
            _mm256_shuffle_epi32(cRhsMat01450, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
        const __m256i cRhsMat23670Sp2 =
            _mm256_shuffle_epi32(cRhsMat23670, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

        const __m256i cRhsMat01451Sp2 = _mm256_shuffle_epi32(
            cRhsMat01451, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
        const __m256i cRhsMat23671Sp2 = _mm256_shuffle_epi32(
            cRhsMat23671, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

        const __m256i cRhsMat01452Sp2 = _mm256_shuffle_epi32(
            cRhsMat01452, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
        const __m256i cRhsMat23672Sp2 = _mm256_shuffle_epi32(
            cRhsMat23672, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

        const __m256i cRhsMat01453Sp2 = _mm256_shuffle_epi32(
            cRhsMat01453, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
        const __m256i cRhsMat23673Sp2 = _mm256_shuffle_epi32(
            cRhsMat23673, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

        // Scale values - Load the wight scale values of block_tx8
        __m256 vColScaleF32;
        if constexpr (std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8>) {
          vColScaleF32 = GGML_F32Cx8_LOAD(b_ptr[vB].d);
        } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
          vColScaleF32 =
              _mm256_set_ps(GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[0]));
        }

        // Process LHS in groups of four
        for (int vRp = 0; vRp < 4; vRp++) {
          // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
          // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
          __m256i vLhsMat01230 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs)));
          __m256i vLhsMat010 = _mm256_permute2f128_si256(vLhsMat01230, vLhsMat01230, 0);
          __m256i vLhsMat230 = _mm256_permute2f128_si256(vLhsMat01230, vLhsMat01230, 17);
          __m256i vLhsMat01231 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32)));
          __m256i vLhsMat011 = _mm256_permute2f128_si256(vLhsMat01231, vLhsMat01231, 0);
          __m256i vLhsMat231 = _mm256_permute2f128_si256(vLhsMat01231, vLhsMat01231, 17);
          __m256i vLhsMat01232 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64)));
          __m256i vLhsMat012 = _mm256_permute2f128_si256(vLhsMat01232, vLhsMat01232, 0);
          __m256i vLhsMat232 = _mm256_permute2f128_si256(vLhsMat01232, vLhsMat01232, 17);
          __m256i vLhsMat01233 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96)));
          __m256i vLhsMat013 = _mm256_permute2f128_si256(vLhsMat01233, vLhsMat01233, 0);
          __m256i vLhsMat233 = _mm256_permute2f128_si256(vLhsMat01233, vLhsMat01233, 17);

          // Shuffle pattern one - left side input
          const __m256i cLhsMat010Sp1 =
              _mm256_shuffle_epi32(vLhsMat010, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
          const __m256i cLhsMat230Sp1 =
              _mm256_shuffle_epi32(vLhsMat230, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

          const __m256i cLhsMat011Sp1 = _mm256_shuffle_epi32(
              vLhsMat011, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
          const __m256i cLhsMat231Sp1 = _mm256_shuffle_epi32(
              vLhsMat231, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

          const __m256i cLhsMat012Sp1 = _mm256_shuffle_epi32(
              vLhsMat012, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
          const __m256i cLhsMat232Sp1 = _mm256_shuffle_epi32(
              vLhsMat232, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

          const __m256i cLhsMat013Sp1 = _mm256_shuffle_epi32(
              vLhsMat013, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
          const __m256i cLhsMat233Sp1 = _mm256_shuffle_epi32(
              vLhsMat233, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

          // Shuffle pattern two - left side input
          const __m256i cLhsMat010Sp2 =
              _mm256_shuffle_epi32(vLhsMat010, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
          const __m256i cLhsMat230Sp2 =
              _mm256_shuffle_epi32(vLhsMat230, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

          const __m256i cLhsMat011Sp2 = _mm256_shuffle_epi32(
              vLhsMat011, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
          const __m256i cLhsMat231Sp2 = _mm256_shuffle_epi32(
              vLhsMat231, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

          const __m256i cLhsMat012Sp2 = _mm256_shuffle_epi32(
              vLhsMat012, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
          const __m256i cLhsMat232Sp2 = _mm256_shuffle_epi32(
              vLhsMat232, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

          const __m256i cLhsMat013Sp2 = _mm256_shuffle_epi32(
              vLhsMat013, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
          const __m256i cLhsMat233Sp2 = _mm256_shuffle_epi32(
              vLhsMat233, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane Resembles MMLAs into
          // 2x2 matrices in ARM Version
          const __m256i cZero = _mm256_setzero_si256();
          __m256i vIaccMat00Sp1 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp1, cRhsMat01453Sp1),
                                           cLhsMat012Sp1, cRhsMat01452Sp1),
                  cLhsMat011Sp1, cRhsMat01451Sp1),
              cLhsMat010Sp1, cRhsMat01450Sp1);
          __m256i vIaccMat01Sp1 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp1, cRhsMat23673Sp1),
                                           cLhsMat012Sp1, cRhsMat23672Sp1),
                  cLhsMat011Sp1, cRhsMat23671Sp1),
              cLhsMat010Sp1, cRhsMat23670Sp1);
          __m256i vIaccMat10Sp1 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp1, cRhsMat01453Sp1),
                                           cLhsMat232Sp1, cRhsMat01452Sp1),
                  cLhsMat231Sp1, cRhsMat01451Sp1),
              cLhsMat230Sp1, cRhsMat01450Sp1);
          __m256i vIaccMat11Sp1 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp1, cRhsMat23673Sp1),
                                           cLhsMat232Sp1, cRhsMat23672Sp1),
                  cLhsMat231Sp1, cRhsMat23671Sp1),
              cLhsMat230Sp1, cRhsMat23670Sp1);
          __m256i vIaccMat00Sp2 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp2, cRhsMat01453Sp2),
                                           cLhsMat012Sp2, cRhsMat01452Sp2),
                  cLhsMat011Sp2, cRhsMat01451Sp2),
              cLhsMat010Sp2, cRhsMat01450Sp2);
          __m256i vIaccMat01Sp2 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp2, cRhsMat23673Sp2),
                                           cLhsMat012Sp2, cRhsMat23672Sp2),
                  cLhsMat011Sp2, cRhsMat23671Sp2),
              cLhsMat010Sp2, cRhsMat23670Sp2);
          __m256i vIaccMat10Sp2 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp2, cRhsMat01453Sp2),
                                           cLhsMat232Sp2, cRhsMat01452Sp2),
                  cLhsMat231Sp2, cRhsMat01451Sp2),
              cLhsMat230Sp2, cRhsMat01450Sp2);
          __m256i vIaccMat11Sp2 = fMulSumI8PairsAccInt32x8(
              fMulSumI8PairsAccInt32x8(
                  fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp2, cRhsMat23673Sp2),
                                           cLhsMat232Sp2, cRhsMat23672Sp2),
                  cLhsMat231Sp2, cRhsMat23671Sp2),
              cLhsMat230Sp2, cRhsMat23670Sp2);

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          __m256i vIaccMat00 = _mm256_add_epi32(vIaccMat00Sp1, vIaccMat00Sp2);
          __m256i vIaccMat01 = _mm256_add_epi32(vIaccMat01Sp1, vIaccMat01Sp2);
          __m256i vIaccMat10 = _mm256_add_epi32(vIaccMat10Sp1, vIaccMat10Sp2);
          __m256i vIaccMat11 = _mm256_add_epi32(vIaccMat11Sp1, vIaccMat11Sp2);

          // Straighten out to make 4 row vectors
          __m256i vIaccRow0 = _mm256_blend_epi32(vIaccMat00, _mm256_shuffle_epi32(vIaccMat01, 78), 204);
          __m256i vIaccRow1 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat00, 78), vIaccMat01, 204);
          __m256i vIaccRow2 = _mm256_blend_epi32(vIaccMat10, _mm256_shuffle_epi32(vIaccMat11, 78), 204);
          __m256i vIaccRow3 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat10, 78), vIaccMat11, 204);

          // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
          const __m256 cRowScaleF32 = GGML_F32Cx8_REPEAT_LOAD(vAPtrs[vRp][vB].d, cLoadMask);

          // Multiply with appropriate scales and accumulate
          aAccRows[vRp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRow0),
              _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
          aAccRows[vRp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRow1),
              _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
          aAccRows[vRp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRow2),
              _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
          aAccRows[vRp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRow3),
              _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);
        }
      }

      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), aAccRows[vI]);
      }
    }
  }

  // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
  for (; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = vAPtrStart + (vY * cNb);

    // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {
      const block_tx8 *b_ptr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Load the eight block_q8_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
        const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs));
        const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 32));
        const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 64));
        const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(b_ptr[vB].qs + 96));

        // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing
        // of values
        const __m256i cRhsRawMat01450 =
            _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
        const __m256i cRhsRawMat23670 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
        const __m256i cRhsRawMat01451 =
            _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
        const __m256i cRhsRawMat23671 =
            _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

        // 4-bit -> 8-bit - Sign is maintained
        const __m256i cRhsMat01450 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat01450, cM4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
        const __m256i cRhsMat23670 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat23670, cM4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

        const __m256i cRhsMat01451 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat01451, cM4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
        const __m256i cRhsMat23671 = _mm256_shuffle_epi8(
            pSignextendlut, _mm256_and_si256(cRhsRawMat23671, cM4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

        const __m256i cRhsMat01452 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4),
                                                                 cM4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
        const __m256i cRhsMat23672 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4),
                                                                 cM4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

        const __m256i cRhsMat01453 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4),
                                                                 cM4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
        const __m256i cRhsMat23673 =
            _mm256_shuffle_epi8(pSignextendlut, _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4),
                                                                 cM4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

        // Shuffle pattern one - right side input
        const __m256i cRhsMat01450Sp1 =
            _mm256_shuffle_epi32(cRhsMat01450, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
        const __m256i cRhsMat23670Sp1 =
            _mm256_shuffle_epi32(cRhsMat23670, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

        const __m256i cRhsMat01451Sp1 = _mm256_shuffle_epi32(
            cRhsMat01451, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
        const __m256i cRhsMat23671Sp1 = _mm256_shuffle_epi32(
            cRhsMat23671, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

        const __m256i cRhsMat01452Sp1 = _mm256_shuffle_epi32(
            cRhsMat01452, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
        const __m256i cRhsMat23672Sp1 = _mm256_shuffle_epi32(
            cRhsMat23672, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

        const __m256i cRhsMat01453Sp1 = _mm256_shuffle_epi32(
            cRhsMat01453, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
        const __m256i cRhsMat23673Sp1 = _mm256_shuffle_epi32(
            cRhsMat23673, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

        // Shuffle pattern two - right side input

        const __m256i cRhsMat01450Sp2 =
            _mm256_shuffle_epi32(cRhsMat01450, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
        const __m256i cRhsMat23670Sp2 =
            _mm256_shuffle_epi32(cRhsMat23670, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

        const __m256i cRhsMat01451Sp2 = _mm256_shuffle_epi32(
            cRhsMat01451, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
        const __m256i cRhsMat23671Sp2 = _mm256_shuffle_epi32(
            cRhsMat23671, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

        const __m256i cRhsMat01452Sp2 = _mm256_shuffle_epi32(
            cRhsMat01452, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
        const __m256i cRhsMat23672Sp2 = _mm256_shuffle_epi32(
            cRhsMat23672, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

        const __m256i cRhsMat01453Sp2 = _mm256_shuffle_epi32(
            cRhsMat01453, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
        const __m256i cRhsMat23673Sp2 = _mm256_shuffle_epi32(
            cRhsMat23673, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

        // Scale values - Load the wight scale values of block_tx8
        __m256 vColScaleF32;
        if constexpr (std::is_same_v<block_tx8, block_q4_0x8> || std::is_same_v<block_tx8, block_iq4_nlx8>) {
          vColScaleF32 = GGML_F32Cx8_LOAD(b_ptr[vB].d);
        } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
          vColScaleF32 =
              _mm256_set_ps(GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[7]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[6]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[5]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[4]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[3]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[2]),
                            GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[1]), GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[vB].e[0]));
        }

        // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
        __m256i vLhsMat01230 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs)));
        __m256i vLhsMat010 = _mm256_permute2f128_si256(vLhsMat01230, vLhsMat01230, 0);
        __m256i vLhsMat230 = _mm256_permute2f128_si256(vLhsMat01230, vLhsMat01230, 17);
        __m256i vLhsMat01231 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32)));
        __m256i vLhsMat011 = _mm256_permute2f128_si256(vLhsMat01231, vLhsMat01231, 0);
        __m256i vLhsMat231 = _mm256_permute2f128_si256(vLhsMat01231, vLhsMat01231, 17);
        __m256i vLhsMat01232 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64)));
        __m256i vLhsMat012 = _mm256_permute2f128_si256(vLhsMat01232, vLhsMat01232, 0);
        __m256i vLhsMat232 = _mm256_permute2f128_si256(vLhsMat01232, vLhsMat01232, 17);
        __m256i vLhsMat01233 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96)));
        __m256i vLhsMat013 = _mm256_permute2f128_si256(vLhsMat01233, vLhsMat01233, 0);
        __m256i vLhsMat233 = _mm256_permute2f128_si256(vLhsMat01233, vLhsMat01233, 17);

        // Shuffle pattern one - left side input

        const __m256i cLhsMat010Sp1 =
            _mm256_shuffle_epi32(vLhsMat010, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
        const __m256i cLhsMat230Sp1 =
            _mm256_shuffle_epi32(vLhsMat230, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

        const __m256i cLhsMat011Sp1 = _mm256_shuffle_epi32(
            vLhsMat011, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
        const __m256i cLhsMat231Sp1 = _mm256_shuffle_epi32(
            vLhsMat231, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

        const __m256i cLhsMat012Sp1 = _mm256_shuffle_epi32(
            vLhsMat012, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
        const __m256i cLhsMat232Sp1 = _mm256_shuffle_epi32(
            vLhsMat232, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

        const __m256i cLhsMat013Sp1 = _mm256_shuffle_epi32(
            vLhsMat013, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
        const __m256i cLhsMat233Sp1 = _mm256_shuffle_epi32(
            vLhsMat233, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

        // Shuffle pattern two - left side input

        const __m256i cLhsMat010Sp2 =
            _mm256_shuffle_epi32(vLhsMat010, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
        const __m256i cLhsMat230Sp2 =
            _mm256_shuffle_epi32(vLhsMat230, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

        const __m256i cLhsMat011Sp2 = _mm256_shuffle_epi32(
            vLhsMat011, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
        const __m256i cLhsMat231Sp2 = _mm256_shuffle_epi32(
            vLhsMat231, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

        const __m256i cLhsMat012Sp2 = _mm256_shuffle_epi32(
            vLhsMat012, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
        const __m256i cLhsMat232Sp2 = _mm256_shuffle_epi32(
            vLhsMat232, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

        const __m256i cLhsMat013Sp2 = _mm256_shuffle_epi32(
            vLhsMat013, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
        const __m256i cLhsMat233Sp2 = _mm256_shuffle_epi32(
            vLhsMat233, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
        // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane Resembles MMLAs into 2x2
        // matrices in ARM Version
        const __m256i cZero = _mm256_setzero_si256();
        __m256i vIaccMat00Sp1 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp1, cRhsMat01453Sp1), cLhsMat012Sp1,
                                         cRhsMat01452Sp1),
                cLhsMat011Sp1, cRhsMat01451Sp1),
            cLhsMat010Sp1, cRhsMat01450Sp1);
        __m256i vIaccMat01Sp1 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp1, cRhsMat23673Sp1), cLhsMat012Sp1,
                                         cRhsMat23672Sp1),
                cLhsMat011Sp1, cRhsMat23671Sp1),
            cLhsMat010Sp1, cRhsMat23670Sp1);
        __m256i vIaccMat10Sp1 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp1, cRhsMat01453Sp1), cLhsMat232Sp1,
                                         cRhsMat01452Sp1),
                cLhsMat231Sp1, cRhsMat01451Sp1),
            cLhsMat230Sp1, cRhsMat01450Sp1);
        __m256i vIaccMat11Sp1 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp1, cRhsMat23673Sp1), cLhsMat232Sp1,
                                         cRhsMat23672Sp1),
                cLhsMat231Sp1, cRhsMat23671Sp1),
            cLhsMat230Sp1, cRhsMat23670Sp1);
        __m256i vIaccMat00Sp2 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp2, cRhsMat01453Sp2), cLhsMat012Sp2,
                                         cRhsMat01452Sp2),
                cLhsMat011Sp2, cRhsMat01451Sp2),
            cLhsMat010Sp2, cRhsMat01450Sp2);
        __m256i vIaccMat01Sp2 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat013Sp2, cRhsMat23673Sp2), cLhsMat012Sp2,
                                         cRhsMat23672Sp2),
                cLhsMat011Sp2, cRhsMat23671Sp2),
            cLhsMat010Sp2, cRhsMat23670Sp2);
        __m256i vIaccMat10Sp2 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp2, cRhsMat01453Sp2), cLhsMat232Sp2,
                                         cRhsMat01452Sp2),
                cLhsMat231Sp2, cRhsMat01451Sp2),
            cLhsMat230Sp2, cRhsMat01450Sp2);
        __m256i vIaccMat11Sp2 = fMulSumI8PairsAccInt32x8(
            fMulSumI8PairsAccInt32x8(
                fMulSumI8PairsAccInt32x8(fMulSumI8PairsAccInt32x8(cZero, cLhsMat233Sp2, cRhsMat23673Sp2), cLhsMat232Sp2,
                                         cRhsMat23672Sp2),
                cLhsMat231Sp2, cRhsMat23671Sp2),
            cLhsMat230Sp2, cRhsMat23670Sp2);

        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
        __m256i vIaccMat00 = _mm256_add_epi32(vIaccMat00Sp1, vIaccMat00Sp2);
        __m256i vIaccMat01 = _mm256_add_epi32(vIaccMat01Sp1, vIaccMat01Sp2);
        __m256i vIaccMat10 = _mm256_add_epi32(vIaccMat10Sp1, vIaccMat10Sp2);
        __m256i vIaccMat11 = _mm256_add_epi32(vIaccMat11Sp1, vIaccMat11Sp2);

        // Straighten out to make 4 row vectors
        __m256i vIaccRow0 = _mm256_blend_epi32(vIaccMat00, _mm256_shuffle_epi32(vIaccMat01, 78), 204);
        __m256i vIaccRow1 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat00, 78), vIaccMat01, 204);
        __m256i vIaccRow2 = _mm256_blend_epi32(vIaccMat10, _mm256_shuffle_epi32(vIaccMat11, 78), 204);
        __m256i vIaccRow3 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat10, 78), vIaccMat11, 204);

        // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
        const __m256 cRowScaleF32 = GGML_F32Cx8_REPEAT_LOAD(vAPtr[vB].d, cLoadMask);

        // Multiply with appropriate scales and accumulate
        aAccRows[0] =
            _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow0),
                            _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[0]);
        aAccRows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow1),
                                      _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                      aAccRows[1]);
        aAccRows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow2),
                                      _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                      aAccRows[2]);
        aAccRows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow3),
                                      _mm256_mul_ps(vColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                      aAccRows[3]);
      }

      // Store the accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), aAccRows[vI]);
      }
    }
  }
}

#endif // defined(__AVX2__) || defined(__AVX512F__)

void ggml_gemv_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__) || defined(__AVX512F__)
  {
    // Lookup table to convert signed nibbles to signed bytes
    __m256i vSignextendlut =
        _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

    fGemvQ4B328x8Q80LutAvx<block_q4_0x8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

    return;
  }
#endif

  ggml_gemv_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q4_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__AVX2__)
  // Lookup table to convert signed nibbles to signed bytes
  __m256i vSignextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);
  // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
  __m128i vDeltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m128i vScalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
  // Permute mask used for easier vector processing at later stages
  __m256i vFinalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Mask to extract nibbles from bytes
  const __m256i cM4b = _mm256_set1_epi8(0x0F);

  int64_t vBNb = pN / QK_K;

  const block_q4_Kx8 *vBPtrStart = (const block_q4_Kx8 *)pVx;
  const block_q8_K *vAPtrStart = (const block_q8_K *)pVy;

  // Process Q8_K blocks one by one
  for (int64_t vY = 0; vY < pNr; vY++) {

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < pNc / 8; vX++) {

      // Pointers to RHS blocks
      const block_q4_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 vAccRow = _mm256_setzero_ps();
      __m256 vAccMinRows = _mm256_setzero_ps();

      for (int64_t vB = 0; vB < cNb; vB++) {

        // Load and convert to FP32 scale from block_q8_K
        const __m256 cRowScaleF32 = _mm256_set1_ps((vAPtr[vB].d));

        // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
        // col_scale_f32 rearranged so as to multiply with appropriate quants
        const __m256 cColScaleF32 = GGML_F32Cx8_REARRANGE_LOAD(vBPtr[vB].d, vDeltamask);
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        __m256i vIaccB = _mm256_setzero_si256();
        __m256i vIaccMinB = _mm256_setzero_si256();

        const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)(vAPtr[vB].bsums));
        __m256i vQ8s = _mm256_castsi128_si256(
            _mm_hadd_epi16(_mm256_castsi256_si128(cQ8sums), _mm256_extracti128_si256(cQ8sums, 1)));
        vQ8s = _mm256_permute2f128_si256(vQ8s, vQ8s, 0);

        // Processes two sub blocks from each Q4_K in each iteration
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {

          // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of
          // eight - B0,B1 ....B6,B7
          const __m256i cRhsRawVec01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawVec45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawVec01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawVec45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawVec01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawVec45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawVec01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawVec45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // 4-bit -> 8-bit
          // Values of the first sub block of eight block_q4_K structures for the sb loop
          const __m256i cRhsVec012300 = _mm256_and_si256(cRhsRawVec01230, cM4b);
          const __m256i cRhsVec456700 = _mm256_and_si256(cRhsRawVec45670, cM4b);
          const __m256i cRhsVec012301 = _mm256_and_si256(cRhsRawVec01231, cM4b);
          const __m256i cRhsVec456701 = _mm256_and_si256(cRhsRawVec45671, cM4b);
          const __m256i cRhsVec012302 = _mm256_and_si256(cRhsRawVec01232, cM4b);
          const __m256i cRhsVec456702 = _mm256_and_si256(cRhsRawVec45672, cM4b);
          const __m256i cRhsVec012303 = _mm256_and_si256(cRhsRawVec01233, cM4b);
          const __m256i cRhsVec456703 = _mm256_and_si256(cRhsRawVec45673, cM4b);

          // Values of the second sub block of eight block_q4_K structures when sb = 1
          const __m256i cRhsVec012310 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01230, 4), cM4b);
          const __m256i cRhsVec456710 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45670, 4), cM4b);
          const __m256i cRhsVec012311 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01231, 4), cM4b);
          const __m256i cRhsVec456711 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45671, 4), cM4b);
          const __m256i cRhsVec012312 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01232, 4), cM4b);
          const __m256i cRhsVec456712 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45672, 4), cM4b);
          const __m256i cRhsVec012313 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01233, 4), cM4b);
          const __m256i cRhsVec456713 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45673, 4), cM4b);

          uint32_t aUtmp0[4], aUtmp1[4];

          // Scales and Mins of corresponding sub blocks from different Q8_K structures are stored together
          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp0, vBPtr[vB].scales + 24 * vSb, 12);
          aUtmp0[3] = ((aUtmp0[2] >> 4) & cKmask2) | (((aUtmp0[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = aUtmp0[1] & cKmask1;
          aUtmp0[1] = (aUtmp0[2] & cKmask2) | (((aUtmp0[0] >> 6) & cKmask3) << 4);
          aUtmp0[2] = cUaux0;
          aUtmp0[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp1, vBPtr[vB].scales + 12 + vSb * 24, 12);
          aUtmp1[3] = ((aUtmp1[2] >> 4) & cKmask2) | (((aUtmp1[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux1 = aUtmp1[1] & cKmask1;
          aUtmp1[1] = (aUtmp1[2] & cKmask2) | (((aUtmp1[0] >> 6) & cKmask3) << 4);
          aUtmp1[2] = cUaux1;
          aUtmp1[0] &= cKmask1;

          // Scales of first sub block in the sb loop
          const __m128i cMinsAndScales0 = _mm_set_epi32(aUtmp0[3], aUtmp0[2], aUtmp0[1], aUtmp0[0]);
          __m128i vScalesRearrange0 = _mm_shuffle_epi8(cMinsAndScales0, vScalemask);
          __m256i vScales0 = _mm256_cvtepu8_epi16(vScalesRearrange0);

          // Scales of second sub block in the sb loop
          __m128i vMinsAndScales1 = _mm_set_epi32(aUtmp1[3], aUtmp1[2], aUtmp1[1], aUtmp1[0]);
          __m128i vScalesRearrange1 = _mm_shuffle_epi8(vMinsAndScales1, vScalemask);
          __m256i vScales1 = _mm256_cvtepu8_epi16(vScalesRearrange1);

          // Mins of first and second sub block of Q4_K block are arranged side by side
          __m256i vMins01 = _mm256_cvtepu8_epi16(
              _mm_unpacklo_epi8(_mm_shuffle_epi32(cMinsAndScales0, 78), _mm_shuffle_epi32(vMinsAndScales1, 78)));

          // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the
          // same across 256 bit vector
          __m256i vLhsVec00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + vSb * 64)));
          __m256i vLhsVec01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 16 + vSb * 64)));
          __m256i vLhsVec10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 32 + vSb * 64)));
          __m256i vLhsVec11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 48 + vSb * 64)));

          vLhsVec00 = _mm256_permute2f128_si256(vLhsVec00, vLhsVec00, 0);
          vLhsVec01 = _mm256_permute2f128_si256(vLhsVec01, vLhsVec01, 0);
          vLhsVec10 = _mm256_permute2f128_si256(vLhsVec10, vLhsVec10, 0);
          vLhsVec11 = _mm256_permute2f128_si256(vLhsVec11, vLhsVec11, 0);

          // Dot product done within 32 bit lanes and accumulated in the same vector
          // First done for first sub block and then for second sub block in each sb
          // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
          // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
          // ...........................................................................
          // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

          __m256i vIacc0 = _mm256_setzero_si256();
          __m256i vIacc1 = _mm256_setzero_si256();

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012300, _mm256_shuffle_epi32(cRhsVec456700, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec00, 0)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012300, 177), cRhsVec456700, 170),
                                   _mm256_shuffle_epi32(vLhsVec00, 85)));

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012301, _mm256_shuffle_epi32(cRhsVec456701, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec00, 170)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012301, 177), cRhsVec456701, 170),
                                   _mm256_shuffle_epi32(vLhsVec00, 255)));

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012302, _mm256_shuffle_epi32(cRhsVec456702, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec01, 0)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012302, 177), cRhsVec456702, 170),
                                   _mm256_shuffle_epi32(vLhsVec01, 85)));

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012303, _mm256_shuffle_epi32(cRhsVec456703, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec01, 170)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012303, 177), cRhsVec456703, 170),
                                   _mm256_shuffle_epi32(vLhsVec01, 255)));

          vIacc0 = _mm256_madd_epi16(vIacc0, vScales0);

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012310, _mm256_shuffle_epi32(cRhsVec456710, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec10, 0)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012310, 177), cRhsVec456710, 170),
                                   _mm256_shuffle_epi32(vLhsVec10, 85)));

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012311, _mm256_shuffle_epi32(cRhsVec456711, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec10, 170)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012311, 177), cRhsVec456711, 170),
                                   _mm256_shuffle_epi32(vLhsVec10, 255)));

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012312, _mm256_shuffle_epi32(cRhsVec456712, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec11, 0)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012312, 177), cRhsVec456712, 170),
                                   _mm256_shuffle_epi32(vLhsVec11, 85)));

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012313, _mm256_shuffle_epi32(cRhsVec456713, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec11, 170)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012313, 177), cRhsVec456713, 170),
                                   _mm256_shuffle_epi32(vLhsVec11, 255)));

          vIacc1 = _mm256_madd_epi16(vIacc1, vScales1);

          // Accumulate the iacc value for one sb
          __m256i vIaccSb = _mm256_add_epi32(vIacc0, vIacc1);

          // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
          // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
          __m256i vQ8sSb = _mm256_shuffle_epi32(vQ8s, 0);
          __m256i vIaccMinSb = _mm256_madd_epi16(vQ8sSb, vMins01);
          vQ8s = _mm256_bsrli_epi128(vQ8s, 4);

          // Accumulate for the complete block
          vIaccB = _mm256_add_epi32(vIaccB, vIaccSb);
          vIaccMinB = _mm256_add_epi32(vIaccMinB, vIaccMinSb);
        }

        // Multiply-Add with scale values for the complete super block
        vAccRow = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccB), _mm256_mul_ps(cColScaleF32, cRowScaleF32), vAccRow);
        vAccMinRows =
            _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccMinB), _mm256_mul_ps(cColDminF32, cRowScaleF32), vAccMinRows);
      }

      // Accumulated output values permuted so as to be stored in appropriate order post accumulation
      vAccRow = _mm256_permutevar8x32_ps(vAccRow, vFinalpermutemask);
      _mm256_storeu_ps(pS + (vY * pNr + vX * 8), _mm256_sub_ps(vAccRow, vAccMinRows));
    }
  }

#else
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  ggml_gemv_q4_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
#endif
}

void ggml_gemv_iq4_nl_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                               const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__)
  __m256i vSignextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
  vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

  fGemvQ4B328x8Q80LutAvx<block_iq4_nlx8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

  return;
#endif

  ggml_gemv_iq4_nl_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_mxfp4_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__)
  __m256i vSignextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_mxfp4));
  vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

  fGemvQ4B328x8Q80LutAvx<block_mxfp4x8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

  return;
#endif

  ggml_gemv_mxfp4_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q2_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

#if defined(__AVX2__)
  // Lookup table to convert signed nibbles to signed bytes
  __m256i vSignextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);
  // Shuffle masks to rearrange delta values to multiply with appropriate scales
  __m128i vDeltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  // Permute mask used for easier vector processing at later stages
  __m256i vFinalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  const __m256i cM3b = _mm256_set1_epi8(3);
  const __m128i cM4bSse = _mm_set1_epi8(0xF);

  // Mask to get appropriate scales
  __m128i vScalemask1 = _mm_set_epi8(14, 14, 6, 6, 12, 12, 4, 4, 10, 10, 2, 2, 8, 8, 0, 0);
  __m128i vScalemask2 = _mm_set_epi8(15, 15, 7, 7, 13, 13, 5, 5, 11, 11, 3, 3, 9, 9, 1, 1);

  int64_t vBNb = pN / QK_K;

  const block_q2_Kx8 *vBPtrStart = (const block_q2_Kx8 *)pVx;
  const block_q8_K *vAPtrStart = (const block_q8_K *)pVy;

  // Process Q8_K blocks one by one
  for (int64_t vY = 0; vY < pNr; vY++) {

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight interleaved block_q2_K structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < pNc / 8; vX++) {

      // Pointers to RHS blocks
      const block_q2_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 vAccRow = _mm256_setzero_ps();
      __m256 vAccMinRows = _mm256_setzero_ps();

      for (int64_t vB = 0; vB < cNb; vB++) {

        // Load and convert to FP32 delta from block_q8_K
        const __m256 cRowScaleF32 = _mm256_set1_ps((vAPtr[vB].d));

        // Load the delta values for the 8 blocks interleaved in block_q2_Kx8
        // col_scale_f32 rearranged so as to multiply with appropriate quants
        const __m256 cColScaleF32 = GGML_F32Cx8_REARRANGE_LOAD(vBPtr[vB].d, vDeltamask);
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        __m256i vIaccB = _mm256_setzero_si256();
        __m256i vIaccMinB = _mm256_setzero_si256();

        // Processes eight sub blocks from each Q2_K in each iteration
        for (int vSb = 0; vSb < QK_K / 128; vSb++) {

          // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of
          // eight - B0,B1 ....B6,B7
          const __m256i cRhsRawVec01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawVec45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawVec01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawVec45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawVec01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawVec45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawVec01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawVec45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // 2-bit -> 8-bit
          // Values of the 0th,2nd,4th,6th sub blocks of eight block_q2_K structures for the sb loop
          const __m256i cRhsVec012300 = _mm256_and_si256(cRhsRawVec01230, cM3b); // B00(0-7) B01(0-7) B02(0-7) B03(0-7)
          const __m256i cRhsVec012320 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01230, 2), cM3b); // B20(0-7) B21(0-7) B22(0-7) B23(0-7)
          const __m256i cRhsVec012340 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01230, 4), cM3b); // B40(0-7) B41(0-7) B42(0-7) B43(0-7)
          const __m256i cRhsVec012360 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01230, 6), cM3b); // B60(0-7) B61(0-7) B62(0-7) B63(0-7)

          const __m256i cRhsVec456700 = _mm256_and_si256(cRhsRawVec45670, cM3b); // B04(0-7) B05(0-7) B06(0-7) B07(0-7)
          const __m256i cRhsVec456720 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45670, 2), cM3b); // B24(0-7) B25(0-7) B26(0-7) B27(0-7)
          const __m256i cRhsVec456740 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45670, 4), cM3b); // B44(0-7) B45(0-7) B46(0-7) B47(0-7)
          const __m256i cRhsVec456760 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45670, 6), cM3b); // B64(0-7) B65(0-7) B66(0-7) B67(0-7)

          const __m256i cRhsVec012301 =
              _mm256_and_si256(cRhsRawVec01231, cM3b); // B00(8-15) B01(8-15) B02(8-15) B03(8-15)
          const __m256i cRhsVec012321 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01231, 2), cM3b); // B20(8-15) B21(8-15) B22(8-15) B23(8-15)
          const __m256i cRhsVec012341 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01231, 4), cM3b); // B40(8-15) B41(8-15) B42(8-15) B43(8-15)
          const __m256i cRhsVec012361 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01231, 6), cM3b); // B60(8-15) B61(8-15) B62(8-15) B63(8-15)

          const __m256i cRhsVec456701 =
              _mm256_and_si256(cRhsRawVec45671, cM3b); // B04(8-15) B05(8-15) B06(8-15) B07(8-15)
          const __m256i cRhsVec456721 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45671, 2), cM3b); // B24(8-15) B25(8-15) B26(8-15) B27(8-15)
          const __m256i cRhsVec456741 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45671, 4), cM3b); // B44(8-15) B45(8-15) B46(8-15) B47(8-15)
          const __m256i cRhsVec456761 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45671, 6), cM3b); // B64(8-15) B65(8-15) B66(8-15) B67(8-15)

          // Values of the 1st,3rd,5th,7th sub blocks of eight block_q2_K structures for the sb loop
          const __m256i cRhsVec012310 = _mm256_and_si256(cRhsRawVec01232, cM3b); // B10(0-7) B11(0-7) B12(0-7) B13(0-7)
          const __m256i cRhsVec012330 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01232, 2), cM3b); // B30(0-7) B31(0-7) B32(0-7) B33(0-7)
          const __m256i cRhsVec012350 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01232, 4), cM3b); // B50(0-7) B51(0-7) B52(0-7) B53(0-7)
          const __m256i cRhsVec012370 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01232, 6), cM3b); // B70(0-7) B71(0-7) B72(0-7) B73(0-7)

          const __m256i cRhsVec456710 = _mm256_and_si256(cRhsRawVec45672, cM3b); // B14(0-7) B15(0-7) B16(0-7) B17(0-7)
          const __m256i cRhsVec456730 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45672, 2), cM3b); // B34(0-7) B35(0-7) B36(0-7) B37(0-7)
          const __m256i cRhsVec456750 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45672, 4), cM3b); // B54(0-7) B55(0-7) B56(0-7) B57(0-7)
          const __m256i cRhsVec456770 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45672, 6), cM3b); // B74(0-7) B75(0-7) B76(0-7) B77(0-7)

          const __m256i cRhsVec012311 =
              _mm256_and_si256(cRhsRawVec01233, cM3b); // B10(8-15) B11(8-15) B12(8-15) B13(8-15)
          const __m256i cRhsVec012331 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01233, 2), cM3b); // B30(8-15) B31(8-15) B32(8-15) B33(8-15)
          const __m256i cRhsVec012351 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01233, 4), cM3b); // B50(8-15) B51(8-15) B52(8-15) B53(8-15)
          const __m256i cRhsVec012371 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec01233, 6), cM3b); // B70(8-15) B71(8-15) B72(8-15) B73(8-15)

          const __m256i cRhsVec456711 =
              _mm256_and_si256(cRhsRawVec45673, cM3b); // B14(8-15) B15(8-15) B16(8-15) B17(8-15)
          const __m256i cRhsVec456731 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45673, 2), cM3b); // B34(8-15) B35(8-15) B36(8-15) B37(8-15)
          const __m256i cRhsVec456751 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45673, 4), cM3b); // B54(8-15) B55(8-15) B56(8-15) B57(8-15)
          const __m256i cRhsVec456771 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawVec45673, 6), cM3b); // B74(8-15) B75(8-15) B76(8-15) B77(8-15)

          // Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
          // s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51
          // m51  s60 m60  s61 m61   s70 m70  s71 m71

          const __m128i cMinsAndScales01 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + vSb * 64));
          const __m128i cMinsAndScales23 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales45 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales67 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 48 + vSb * 64));

          // Extract scales which is lower half from mins_and_scales
          const __m128i cScales01 = _mm_and_si128(cMinsAndScales01, cM4bSse);
          const __m128i cScales23 = _mm_and_si128(cMinsAndScales23, cM4bSse);
          const __m128i cScales45 = _mm_and_si128(cMinsAndScales45, cM4bSse);
          const __m128i cScales67 = _mm_and_si128(cMinsAndScales67, cM4bSse);

          // Extract mins which is upper half from mins_and_scales
          const __m256i cMins01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales01, 4), cM4bSse));
          const __m256i cMins23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales23, 4), cM4bSse));
          const __m256i cMins45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales45, 4), cM4bSse));
          const __m256i cMins67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales67, 4), cM4bSse));

          // Scales of sub blocks in the sb loop
          // Scales of the 0th sub block from each super block
          __m128i vScalesRearrange0 = _mm_shuffle_epi8(cScales01, vScalemask1);
          __m256i vScales0 = _mm256_cvtepu8_epi16(vScalesRearrange0);

          // Scales of the 1st sub block from each super block
          __m128i vScalesRearrange1 = _mm_shuffle_epi8(cScales01, vScalemask2);
          __m256i vScales1 = _mm256_cvtepu8_epi16(vScalesRearrange1);

          // Scales of the 2nd sub block from each super block
          __m128i vScalesRearrange2 = _mm_shuffle_epi8(cScales23, vScalemask1);
          __m256i vScales2 = _mm256_cvtepu8_epi16(vScalesRearrange2);

          // Scales of the 3rd sub block from each super block
          __m128i vScalesRearrange3 = _mm_shuffle_epi8(cScales23, vScalemask2);
          __m256i vScales3 = _mm256_cvtepu8_epi16(vScalesRearrange3);

          // Scales of the 4th sub block from each super block
          __m128i vScalesRearrange4 = _mm_shuffle_epi8(cScales45, vScalemask1);
          __m256i vScales4 = _mm256_cvtepu8_epi16(vScalesRearrange4);

          // Scales of the 5th sub block from each super block
          __m128i vScalesRearrange5 = _mm_shuffle_epi8(cScales45, vScalemask2);
          __m256i vScales5 = _mm256_cvtepu8_epi16(vScalesRearrange5);

          // Scales of the 6th sub block from each super block
          __m128i vScalesRearrange6 = _mm_shuffle_epi8(cScales67, vScalemask1);
          __m256i vScales6 = _mm256_cvtepu8_epi16(vScalesRearrange6);

          // Scales of the 7th sub block from each super block
          __m128i vScalesRearrange7 = _mm_shuffle_epi8(cScales67, vScalemask2);
          __m256i vScales7 = _mm256_cvtepu8_epi16(vScalesRearrange7);

          // Load the sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same
          // across 256 bit vector
          __m256i vLhsVec0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + vSb * 128)));
          __m256i vLhsVec1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 16 + vSb * 128)));
          __m256i vLhsVec2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 32 + vSb * 128)));
          __m256i vLhsVec3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 48 + vSb * 128)));
          __m256i vLhsVec4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 64 + vSb * 128)));
          __m256i vLhsVec5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 80 + vSb * 128)));
          __m256i vLhsVec6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 96 + vSb * 128)));
          __m256i vLhsVec7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(vAPtr[vB].qs + 112 + vSb * 128)));

          vLhsVec0 = _mm256_permute2f128_si256(vLhsVec0, vLhsVec0, 0);
          vLhsVec1 = _mm256_permute2f128_si256(vLhsVec1, vLhsVec1, 0);
          vLhsVec2 = _mm256_permute2f128_si256(vLhsVec2, vLhsVec2, 0);
          vLhsVec3 = _mm256_permute2f128_si256(vLhsVec3, vLhsVec3, 0);
          vLhsVec4 = _mm256_permute2f128_si256(vLhsVec4, vLhsVec4, 0);
          vLhsVec5 = _mm256_permute2f128_si256(vLhsVec5, vLhsVec5, 0);
          vLhsVec6 = _mm256_permute2f128_si256(vLhsVec6, vLhsVec6, 0);
          vLhsVec7 = _mm256_permute2f128_si256(vLhsVec7, vLhsVec7, 0);

          __m256i vIacc0 = _mm256_setzero_si256();
          __m256i vIacc1 = _mm256_setzero_si256();
          __m256i vIacc2 = _mm256_setzero_si256();
          __m256i vIacc3 = _mm256_setzero_si256();
          __m256i vIacc4 = _mm256_setzero_si256();
          __m256i vIacc5 = _mm256_setzero_si256();
          __m256i vIacc6 = _mm256_setzero_si256();
          __m256i vIacc7 = _mm256_setzero_si256();

          // Dot product done within 32 bit lanes and accumulated in the same vector
          // First done for 0th sub block and then for seven (1st - 7th) other sub blocks processed for each sb (sb <
          // QK_K/128 loop)                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with
          // A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7) B0(8-11) B4(8-11)
          // B1(8-11) B5(8-11) B2(8-11) B6(8-11) B3(8-11) B7(8-11) with A0(8-11) B0(12-15) B4(12-15) B1(12-15) B5(12-15)
          // B2(12-15) B6(12-15) B3(12-15) B7(12-15) with A0(12-15)

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012300, _mm256_shuffle_epi32(cRhsVec456700, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec0, 0)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012300, 177), cRhsVec456700, 170),
                                   _mm256_shuffle_epi32(vLhsVec0, 85)));

          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012301, _mm256_shuffle_epi32(cRhsVec456701, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec0, 170)));
          vIacc0 = _mm256_add_epi16(
              vIacc0,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012301, 177), cRhsVec456701, 170),
                                   _mm256_shuffle_epi32(vLhsVec0, 255)));

          vIacc0 = _mm256_madd_epi16(vIacc0, vScales0);

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012310, _mm256_shuffle_epi32(cRhsVec456710, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec1, 0)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012310, 177), cRhsVec456710, 170),
                                   _mm256_shuffle_epi32(vLhsVec1, 85)));

          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012311, _mm256_shuffle_epi32(cRhsVec456711, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec1, 170)));
          vIacc1 = _mm256_add_epi16(
              vIacc1,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012311, 177), cRhsVec456711, 170),
                                   _mm256_shuffle_epi32(vLhsVec1, 255)));

          vIacc1 = _mm256_madd_epi16(vIacc1, vScales1);

          vIacc2 = _mm256_add_epi16(
              vIacc2,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012320, _mm256_shuffle_epi32(cRhsVec456720, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec2, 0)));
          vIacc2 = _mm256_add_epi16(
              vIacc2,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012320, 177), cRhsVec456720, 170),
                                   _mm256_shuffle_epi32(vLhsVec2, 85)));

          vIacc2 = _mm256_add_epi16(
              vIacc2,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012321, _mm256_shuffle_epi32(cRhsVec456721, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec2, 170)));
          vIacc2 = _mm256_add_epi16(
              vIacc2,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012321, 177), cRhsVec456721, 170),
                                   _mm256_shuffle_epi32(vLhsVec2, 255)));

          vIacc2 = _mm256_madd_epi16(vIacc2, vScales2);

          vIacc3 = _mm256_add_epi16(
              vIacc3,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012330, _mm256_shuffle_epi32(cRhsVec456730, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec3, 0)));
          vIacc3 = _mm256_add_epi16(
              vIacc3,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012330, 177), cRhsVec456730, 170),
                                   _mm256_shuffle_epi32(vLhsVec3, 85)));

          vIacc3 = _mm256_add_epi16(
              vIacc3,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012331, _mm256_shuffle_epi32(cRhsVec456731, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec3, 170)));
          vIacc3 = _mm256_add_epi16(
              vIacc3,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012331, 177), cRhsVec456731, 170),
                                   _mm256_shuffle_epi32(vLhsVec3, 255)));

          vIacc3 = _mm256_madd_epi16(vIacc3, vScales3);

          vIacc4 = _mm256_add_epi16(
              vIacc4,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012340, _mm256_shuffle_epi32(cRhsVec456740, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec4, 0)));
          vIacc4 = _mm256_add_epi16(
              vIacc4,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012340, 177), cRhsVec456740, 170),
                                   _mm256_shuffle_epi32(vLhsVec4, 85)));

          vIacc4 = _mm256_add_epi16(
              vIacc4,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012341, _mm256_shuffle_epi32(cRhsVec456741, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec4, 170)));
          vIacc4 = _mm256_add_epi16(
              vIacc4,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012341, 177), cRhsVec456741, 170),
                                   _mm256_shuffle_epi32(vLhsVec4, 255)));

          vIacc4 = _mm256_madd_epi16(vIacc4, vScales4);

          vIacc5 = _mm256_add_epi16(
              vIacc5,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012350, _mm256_shuffle_epi32(cRhsVec456750, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec5, 0)));
          vIacc5 = _mm256_add_epi16(
              vIacc5,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012350, 177), cRhsVec456750, 170),
                                   _mm256_shuffle_epi32(vLhsVec5, 85)));

          vIacc5 = _mm256_add_epi16(
              vIacc5,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012351, _mm256_shuffle_epi32(cRhsVec456751, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec5, 170)));
          vIacc5 = _mm256_add_epi16(
              vIacc5,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012351, 177), cRhsVec456751, 170),
                                   _mm256_shuffle_epi32(vLhsVec5, 255)));

          vIacc5 = _mm256_madd_epi16(vIacc5, vScales5);

          vIacc6 = _mm256_add_epi16(
              vIacc6,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012360, _mm256_shuffle_epi32(cRhsVec456760, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec6, 0)));
          vIacc6 = _mm256_add_epi16(
              vIacc6,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012360, 177), cRhsVec456760, 170),
                                   _mm256_shuffle_epi32(vLhsVec6, 85)));

          vIacc6 = _mm256_add_epi16(
              vIacc6,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012361, _mm256_shuffle_epi32(cRhsVec456761, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec6, 170)));
          vIacc6 = _mm256_add_epi16(
              vIacc6,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012361, 177), cRhsVec456761, 170),
                                   _mm256_shuffle_epi32(vLhsVec6, 255)));

          vIacc6 = _mm256_madd_epi16(vIacc6, vScales6);

          vIacc7 = _mm256_add_epi16(
              vIacc7,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012370, _mm256_shuffle_epi32(cRhsVec456770, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec7, 0)));
          vIacc7 = _mm256_add_epi16(
              vIacc7,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012370, 177), cRhsVec456770, 170),
                                   _mm256_shuffle_epi32(vLhsVec7, 85)));

          vIacc7 = _mm256_add_epi16(
              vIacc7,
              _mm256_maddubs_epi16(_mm256_blend_epi32(cRhsVec012371, _mm256_shuffle_epi32(cRhsVec456771, 177), 170),
                                   _mm256_shuffle_epi32(vLhsVec7, 170)));
          vIacc7 = _mm256_add_epi16(
              vIacc7,
              _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(cRhsVec012371, 177), cRhsVec456771, 170),
                                   _mm256_shuffle_epi32(vLhsVec7, 255)));

          vIacc7 = _mm256_madd_epi16(vIacc7, vScales7);

          // Accumulate the iacc value for one sb
          __m256i vIaccSb =
              _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(vIacc0, vIacc1), _mm256_add_epi32(vIacc2, vIacc3)),
                               _mm256_add_epi32(_mm256_add_epi32(vIacc4, vIacc5), _mm256_add_epi32(vIacc6, vIacc7)));

          __m128i vQ8sums = _mm_loadu_si128((const __m128i *)(vAPtr[vB].bsums + vSb * 8));
          __m256i vQ8s = _mm256_castsi128_si256(vQ8sums);
          vQ8s = _mm256_permute2f128_si256(vQ8s, vQ8s, 0);

          // Broadcast the bsums of the two corresponding subblocks of q8_k
          // Multiply-Add with corresponding mins of Q2_Kx8 with bsums
          __m256i vIaccMinSb01 = _mm256_madd_epi16(_mm256_shuffle_epi32(vQ8s, 0), cMins01);
          __m256i vIaccMinSb23 = _mm256_madd_epi16(_mm256_shuffle_epi32(vQ8s, 85), cMins23);
          __m256i vIaccMinSb45 = _mm256_madd_epi16(_mm256_shuffle_epi32(vQ8s, 170), cMins45);
          __m256i vIaccMinSb67 = _mm256_madd_epi16(_mm256_shuffle_epi32(vQ8s, 255), cMins67);

          __m256i vIaccMinSb = _mm256_add_epi32(_mm256_add_epi32(vIaccMinSb01, vIaccMinSb23),
                                                _mm256_add_epi32(vIaccMinSb45, vIaccMinSb67));

          // Accumulate for the complete block
          vIaccB = _mm256_add_epi32(vIaccB, vIaccSb);
          vIaccMinB = _mm256_add_epi32(vIaccMinB, vIaccMinSb);
        }

        // Multiply-Add with scale values for complete super block
        vAccRow = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccB), _mm256_mul_ps(cColScaleF32, cRowScaleF32), vAccRow);
        vAccMinRows =
            _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccMinB), _mm256_mul_ps(cColDminF32, cRowScaleF32), vAccMinRows);
      }
      // Accumulated output values permuted so as to be stored in appropriate order post accumulation
      vAccRow = _mm256_permutevar8x32_ps(vAccRow, vFinalpermutemask);
      _mm256_storeu_ps(pS + (vY * pNr + vX * 8), _mm256_sub_ps(vAccRow, vAccMinRows));
    }
  }
#else

  ggml_gemv_q2_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);

#endif
}

void ggml_gemm_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__) || defined(__AVX512F__)
  {
    // Lookup table to convert signed nibbles to signed bytes
    __m256i vSignextendlut =
        _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

    fGemmQ4B328x8Q80LutAvx<block_q4_0x8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

    return;
  }
#endif // defined(__AVX2__) || defined(__AVX512F__)

  ggml_gemm_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q4_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(pNc);
  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__AVX2__) || defined(__AVX512F__)
  const block_q4_Kx8 *vBPtrStart = (const block_q4_Kx8 *)pVx;
  const block_q8_Kx4 *vAPtrStart = (const block_q8_Kx4 *)pVy;
  int64_t vBNb = pN / QK_K;
  int64_t vY = 0;

  // Mask to mask out nibbles from packed bytes
  const __m256i cM4b = _mm256_set1_epi8(0x0F);
  // Permute mask used for easier vector processing at later stages
  __m256i vRequiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  int64_t vXstart = 0;
  int vAnr = pNr - pNr % 16;
  ; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
  int vAnc = pNc - pNc % 16; // Used to align nc with boundary of 16
  // Mask to mask out nibbles from packed bytes expanded to 512 bit length
  const __m512i cM4bexpanded = _mm512_set1_epi8(0x0F);
  // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
  for (; vY < vAnr / 4; vY += 4) {

    const block_q8_Kx4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_q4_Kx8 *vBPtr0 = vBPtrStart + ((vX)*vBNb);
      const block_q4_Kx8 *vBPtr1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      __m512 aAccMinRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccMinRows[vI] = _mm512_setzero_ps();
      }

      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {
        // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
        const __m512 cColScaleF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].d, vBPtr1[vB].d);

        // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
        const __m512 cColDminF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].dmin, vBPtr1[vB].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {

          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + vSb * 256));
          const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat89Ab2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMatCdef2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat89Ab3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMatCdef3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          const __m256i cRhsRawMat89Cd0 =
              _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef0 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
          const __m256i cRhsRawMat89Cd1 =
              _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef1 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);
          const __m256i cRhsRawMat89Cd2 =
              _mm256_blend_epi32(cRhsRawMat89Ab2, _mm256_permutevar8x32_epi32(cRhsRawMatCdef2, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef2 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab2, vRequiredOrder), cRhsRawMatCdef2, 240);
          const __m256i cRhsRawMat89Cd3 =
              _mm256_blend_epi32(cRhsRawMat89Ab3, _mm256_permutevar8x32_epi32(cRhsRawMatCdef3, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef3 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab3, vRequiredOrder), cRhsRawMatCdef3, 240);

          const __m512i cRhsRawMat014589Cd0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
          const __m512i cRhsRawMat2367Abef0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
          const __m512i cRhsRawMat014589Cd1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
          const __m512i cRhsRawMat2367Abef1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

          const __m512i cRhsRawMat014589Cd2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01452), cRhsRawMat89Cd2, 1);
          const __m512i cRhsRawMat2367Abef2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23672), cRhsRawMatAbef2, 1);
          const __m512i cRhsRawMat014589Cd3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01453), cRhsRawMat89Cd3, 1);
          const __m512i cRhsRawMat2367Abef3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23673), cRhsRawMatAbef3, 1);

          // 4-bit -> 8-bit
          const __m512i cRhsMat014589Cd00 =
              _mm512_and_si512(cRhsRawMat014589Cd0,
                               cM4bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i cRhsMat2367Abef00 =
              _mm512_and_si512(cRhsRawMat2367Abef0,
                               cM4bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i cRhsMat014589Cd01 = _mm512_and_si512(
              cRhsRawMat014589Cd1,
              cM4bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i cRhsMat2367Abef01 = _mm512_and_si512(
              cRhsRawMat2367Abef1,
              cM4bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

          const __m512i cRhsMat014589Cd02 = _mm512_and_si512(
              cRhsRawMat014589Cd2,
              cM4bexpanded); // B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
          const __m512i cRhsMat2367Abef02 = _mm512_and_si512(
              cRhsRawMat2367Abef2,
              cM4bexpanded); // B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
          const __m512i cRhsMat014589Cd03 = _mm512_and_si512(
              cRhsRawMat014589Cd3,
              cM4bexpanded); // B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
          const __m512i cRhsMat2367Abef03 = _mm512_and_si512(
              cRhsRawMat2367Abef3,
              cM4bexpanded); // B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

          const __m512i cRhsMat014589Cd10 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                               cM4bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i cRhsMat2367Abef10 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                               cM4bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i cRhsMat014589Cd11 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
              cM4bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i cRhsMat2367Abef11 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
              cM4bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i cRhsMat014589Cd12 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd2, 4),
              cM4bexpanded); // B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
          const __m512i cRhsMat2367Abef12 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef2, 4),
              cM4bexpanded); // B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
          const __m512i cRhsMat014589Cd13 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 4),
              cM4bexpanded); // B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
          const __m512i cRhsMat2367Abef13 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 4),
              cM4bexpanded); // B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

          // Shuffle pattern one - right side input
          const __m512i cRhsMat014589Cd00Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3)
                                   // B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
          const __m512i cRhsMat2367Abef00Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3)
                                   // B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
          const __m512i cRhsMat014589Cd01Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef01Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                   // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
          const __m512i cRhsMat014589Cd02Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd02,
              (_MM_PERM_ENUM)136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19)
                                   // B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19)
                                   // B0C(16-19) B0D(16-19)
          const __m512i cRhsMat2367Abef02Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef02,
              (_MM_PERM_ENUM)136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19)
                                   // B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19)
                                   // B0E(16-19) B0F(16-19)
          const __m512i cRhsMat014589Cd03Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd03,
              (_MM_PERM_ENUM)136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27)
                                   // B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27)
                                   // B0C(24-27) B0D(24-27)
          const __m512i cRhsMat2367Abef03Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef03,
              (_MM_PERM_ENUM)136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27)
                                   // B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27)
                                   // B0E(24-27) B0F(24-27)

          const __m512i cRhsMat014589Cd10Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3)
                                   // B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
          const __m512i cRhsMat2367Abef10Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3)
                                   // B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
          const __m512i cRhsMat014589Cd11Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                   // B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i cRhsMat2367Abef11Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                   // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
          const __m512i cRhsMat014589Cd12Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd12,
              (_MM_PERM_ENUM)136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19)
                                   // B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19)
                                   // B1C(16-19) B1D(16-19)
          const __m512i cRhsMat2367Abef12Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef12,
              (_MM_PERM_ENUM)136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19)
                                   // B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19)
                                   // B1E(16-19) B1F(16-19)
          const __m512i cRhsMat014589Cd13Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd13,
              (_MM_PERM_ENUM)136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27)
                                   // B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27)
                                   // B1C(24-27) B1D(24-27)
          const __m512i cRhsMat2367Abef13Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef13,
              (_MM_PERM_ENUM)136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27)
                                   // B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27)
                                   // B1E(24-27) B1F(24-27)

          // Shuffle pattern two - right side input
          const __m512i cRhsMat014589Cd00Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7)
                                   // B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
          const __m512i cRhsMat2367Abef00Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7)
                                   // B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
          const __m512i cRhsMat014589Cd01Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15)
                                   // B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15)
                                   // B0C(12-15) B0D(12-15)
          const __m512i cRhsMat2367Abef01Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15)
                                   // B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15)
                                   // B0E(12-15) B0F(12-15)
          const __m512i cRhsMat014589Cd02Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd02,
              (_MM_PERM_ENUM)221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23)
                                   // B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23)
                                   // B0C(20-23) B0D(20-23)
          const __m512i cRhsMat2367Abef02Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef02,
              (_MM_PERM_ENUM)221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23)
                                   // B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23)
                                   // B0E(20-23) B0F(20-23)
          const __m512i cRhsMat014589Cd03Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd03,
              (_MM_PERM_ENUM)221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31)
                                   // B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31)
                                   // B0C(28-31) 0BD(28-31)
          const __m512i cRhsMat2367Abef03Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef03,
              (_MM_PERM_ENUM)221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31)
                                   // B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31)
                                   // B0E(28-31) B0F(28-31)

          const __m512i cRhsMat014589Cd10Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7)
                                   // B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
          const __m512i cRhsMat2367Abef10Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7)
                                   // B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
          const __m512i cRhsMat014589Cd11Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15)
                                   // B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15)
                                   // B1C(12-15) B1D(12-15)
          const __m512i cRhsMat2367Abef11Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15)
                                   // B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15)
                                   // B1E(12-15) B1F(12-15)
          const __m512i cRhsMat014589Cd12Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd12,
              (_MM_PERM_ENUM)221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23)
                                   // B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23)
                                   // B1C(20-23) B1D(20-23)
          const __m512i cRhsMat2367Abef12Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef12,
              (_MM_PERM_ENUM)221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23)
                                   // B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23)
                                   // B1E(20-23) B1F(20-23)
          const __m512i cRhsMat014589Cd13Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd13,
              (_MM_PERM_ENUM)221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31)
                                   // B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31)
                                   // B1C(28-31) B1D(28-31)
          const __m512i cRhsMat2367Abef13Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef13,
              (_MM_PERM_ENUM)221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31)
                                   // B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31)
                                   // B1E(28-31) B1F(28-31)

          uint32_t aUtmp00[4], aUtmp01[4], aUtmp10[4], aUtmp11[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp00, vBPtr0[vB].scales + 24 * vSb, 12);
          aUtmp00[3] = ((aUtmp00[2] >> 4) & cKmask2) | (((aUtmp00[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux00 = aUtmp00[1] & cKmask1;
          aUtmp00[1] = (aUtmp00[2] & cKmask2) | (((aUtmp00[0] >> 6) & cKmask3) << 4);
          aUtmp00[2] = cUaux00;
          aUtmp00[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp01, vBPtr0[vB].scales + 12 + vSb * 24, 12);
          aUtmp01[3] = ((aUtmp01[2] >> 4) & cKmask2) | (((aUtmp01[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux01 = aUtmp01[1] & cKmask1;
          aUtmp01[1] = (aUtmp01[2] & cKmask2) | (((aUtmp01[0] >> 6) & cKmask3) << 4);
          aUtmp01[2] = cUaux01;
          aUtmp01[0] &= cKmask1;

          memcpy(aUtmp10, vBPtr1[vB].scales + vSb * 24, 12);
          aUtmp10[3] = ((aUtmp10[2] >> 4) & cKmask2) | (((aUtmp10[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux10 = aUtmp10[1] & cKmask1;
          aUtmp10[1] = (aUtmp10[2] & cKmask2) | (((aUtmp10[0] >> 6) & cKmask3) << 4);
          aUtmp10[2] = cUaux10;
          aUtmp10[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp11, vBPtr1[vB].scales + 12 + vSb * 24, 12);
          aUtmp11[3] = ((aUtmp11[2] >> 4) & cKmask2) | (((aUtmp11[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux11 = aUtmp11[1] & cKmask1;
          aUtmp11[1] = (aUtmp11[2] & cKmask2) | (((aUtmp11[0] >> 6) & cKmask3) << 4);
          aUtmp11[2] = cUaux11;
          aUtmp11[0] &= cKmask1;

          // Scales of first sub block in the sb loop
          const __m256i cMinsAndScales0 = _mm256_set_epi32(aUtmp10[3], aUtmp10[2], aUtmp10[1], aUtmp10[0], aUtmp00[3],
                                                           aUtmp00[2], aUtmp00[1], aUtmp00[0]);
          const __m512i cScales0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(cMinsAndScales0, cMinsAndScales0));

          // Scales of second sub block in the sb loop
          const __m256i cMinsAndScales1 = _mm256_set_epi32(aUtmp11[3], aUtmp11[2], aUtmp11[1], aUtmp11[0], aUtmp01[3],
                                                           aUtmp01[2], aUtmp01[1], aUtmp01[0]);
          const __m512i cScales1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(cMinsAndScales1, cMinsAndScales1));

          // Mins of first and second sub block of Q4_K block are arranged side by side
          const __m512i cMins01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(cMinsAndScales0, 78),
                                                                            _mm256_shuffle_epi32(cMinsAndScales1, 78)));

          const __m512i cScale014589Cd0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)238);

          for (int vRp = 0; vRp < 4; vRp++) {

            // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
            // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again
            // repeating into 512 bit vector
            __m256i vLhsMatYmm012300 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 256 * vSb)));
            __m256i vLhsMatYmm0100 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 0);
            __m256i vLhsMatYmm2300 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 17);
            __m256i vLhsMatYmm012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32 + 256 * vSb)));
            __m256i vLhsMatYmm0101 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 0);
            __m256i vLhsMatYmm2301 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 17);
            __m256i vLhsMatYmm012302 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64 + 256 * vSb)));
            __m256i vLhsMatYmm0102 = _mm256_permute2f128_si256(vLhsMatYmm012302, vLhsMatYmm012302, 0);
            __m256i vLhsMatYmm2302 = _mm256_permute2f128_si256(vLhsMatYmm012302, vLhsMatYmm012302, 17);
            __m256i vLhsMatYmm012303 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96 + 256 * vSb)));
            __m256i vLhsMatYmm0103 = _mm256_permute2f128_si256(vLhsMatYmm012303, vLhsMatYmm012303, 0);
            __m256i vLhsMatYmm2303 = _mm256_permute2f128_si256(vLhsMatYmm012303, vLhsMatYmm012303, 17);
            __m256i vLhsMatYmm012310 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 128 + 256 * vSb)));
            __m256i vLhsMatYmm0110 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 0);
            __m256i vLhsMatYmm2310 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 17);
            __m256i vLhsMatYmm012311 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 160 + 256 * vSb)));
            __m256i vLhsMatYmm0111 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 0);
            __m256i vLhsMatYmm2311 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 17);
            __m256i vLhsMatYmm012312 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 192 + 256 * vSb)));
            __m256i vLhsMatYmm0112 = _mm256_permute2f128_si256(vLhsMatYmm012312, vLhsMatYmm012312, 0);
            __m256i vLhsMatYmm2312 = _mm256_permute2f128_si256(vLhsMatYmm012312, vLhsMatYmm012312, 17);
            __m256i vLhsMatYmm012313 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 224 + 256 * vSb)));
            __m256i vLhsMatYmm0113 = _mm256_permute2f128_si256(vLhsMatYmm012313, vLhsMatYmm012313, 0);
            __m256i vLhsMatYmm2313 = _mm256_permute2f128_si256(vLhsMatYmm012313, vLhsMatYmm012313, 17);

            __m512i vLhsMat0100 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0100), vLhsMatYmm0100, 1);
            __m512i vLhsMat2300 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2300), vLhsMatYmm2300, 1);
            __m512i vLhsMat0101 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0101), vLhsMatYmm0101, 1);
            __m512i vLhsMat2301 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2301), vLhsMatYmm2301, 1);
            __m512i vLhsMat0102 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0102), vLhsMatYmm0102, 1);
            __m512i vLhsMat2302 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2302), vLhsMatYmm2302, 1);
            __m512i vLhsMat0103 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0103), vLhsMatYmm0103, 1);
            __m512i vLhsMat2303 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2303), vLhsMatYmm2303, 1);

            __m512i vLhsMat0110 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0110), vLhsMatYmm0110, 1);
            __m512i vLhsMat2310 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2310), vLhsMatYmm2310, 1);
            __m512i vLhsMat0111 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0111), vLhsMatYmm0111, 1);
            __m512i vLhsMat2311 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2311), vLhsMatYmm2311, 1);
            __m512i vLhsMat0112 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0112), vLhsMatYmm0112, 1);
            __m512i vLhsMat2312 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2312), vLhsMatYmm2312, 1);
            __m512i vLhsMat0113 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0113), vLhsMatYmm0113, 1);
            __m512i vLhsMat2313 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2313), vLhsMatYmm2313, 1);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
            __m256i vLhsBsums012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].bsums + 16 * vSb)));
            __m256i vLhsBsumsHsumYmm012301 = _mm256_castsi128_si256(
                _mm_hadd_epi16(_mm256_castsi256_si128(vLhsBsums012301), _mm256_extractf128_si256(vLhsBsums012301, 1)));
            vLhsBsumsHsumYmm012301 = _mm256_permute2x128_si256(vLhsBsumsHsumYmm012301, vLhsBsumsHsumYmm012301, 0);
            __m512i vLhsBsumsHsum012301 =
                _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsHsumYmm012301), vLhsBsumsHsumYmm012301, 1);

            // Shuffle pattern one - left side input
            const __m512i cLhsMat0100Sp1 = _mm512_shuffle_epi32(
                vLhsMat0100,
                (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                     // A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m512i cLhsMat2300Sp1 = _mm512_shuffle_epi32(
                vLhsMat2300,
                (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3)
                                     // A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
            const __m512i cLhsMat0101Sp1 = _mm512_shuffle_epi32(
                vLhsMat0101,
                (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                     // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m512i cLhsMat2301Sp1 = _mm512_shuffle_epi32(
                vLhsMat2301,
                (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                     // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
            const __m512i cLhsMat0102Sp1 = _mm512_shuffle_epi32(
                vLhsMat0102, (_MM_PERM_ENUM)160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19)
                                                  // A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                                  // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
            const __m512i cLhsMat2302Sp1 = _mm512_shuffle_epi32(
                vLhsMat2302, (_MM_PERM_ENUM)160); // A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19)
                                                  // A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                                                  // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
            const __m512i cLhsMat0103Sp1 = _mm512_shuffle_epi32(
                vLhsMat0103, (_MM_PERM_ENUM)160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27)
                                                  // A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                                  // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
            const __m512i cLhsMat2303Sp1 = _mm512_shuffle_epi32(
                vLhsMat2303, (_MM_PERM_ENUM)160); // A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27)
                                                  // A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)
                                                  // A02(24-27) A02(24-27) A03(24-27) A03(24-27)

            const __m512i cLhsMat0110Sp1 = _mm512_shuffle_epi32(
                vLhsMat0110,
                (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                     // A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m512i cLhsMat2310Sp1 = _mm512_shuffle_epi32(
                vLhsMat2310,
                (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3)
                                     // A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
            const __m512i cLhsMat0111Sp1 = _mm512_shuffle_epi32(
                vLhsMat0111,
                (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                     // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m512i cLhsMat2311Sp1 = _mm512_shuffle_epi32(
                vLhsMat2311,
                (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                     // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
            const __m512i cLhsMat0112Sp1 = _mm512_shuffle_epi32(
                vLhsMat0112, (_MM_PERM_ENUM)160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19)
                                                  // A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                                  // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
            const __m512i cLhsMat2312Sp1 = _mm512_shuffle_epi32(
                vLhsMat2312, (_MM_PERM_ENUM)160); // A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19)
                                                  // A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                                                  // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
            const __m512i cLhsMat0113Sp1 = _mm512_shuffle_epi32(
                vLhsMat0113, (_MM_PERM_ENUM)160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27)
                                                  // A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                                  // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
            const __m512i cLhsMat2313Sp1 = _mm512_shuffle_epi32(
                vLhsMat2313, (_MM_PERM_ENUM)160); // A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27)
                                                  // A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)
                                                  // A12(24-27) A12(24-27) A13(24-27) A13(24-27)

            const __m512i cLhsMat0100Sp2 = _mm512_shuffle_epi32(
                vLhsMat0100,
                (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                     // A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m512i cLhsMat2300Sp2 = _mm512_shuffle_epi32(
                vLhsMat2300,
                (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7)
                                     // A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
            const __m512i cLhsMat0101Sp2 = _mm512_shuffle_epi32(
                vLhsMat0101, (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15)
                                                  // A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                                  // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
            const __m512i cLhsMat2301Sp2 = _mm512_shuffle_epi32(
                vLhsMat2301, (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15)
                                                  // A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                                  // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
            const __m512i cLhsMat0102Sp2 = _mm512_shuffle_epi32(
                vLhsMat0102, (_MM_PERM_ENUM)245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23)
                                                  // A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                                  // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
            const __m512i cLhsMat2302Sp2 = _mm512_shuffle_epi32(
                vLhsMat2302, (_MM_PERM_ENUM)245); // A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23)
                                                  // A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                                                  // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
            const __m512i cLhsMat0103Sp2 = _mm512_shuffle_epi32(
                vLhsMat0103, (_MM_PERM_ENUM)245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31)
                                                  // A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                                  // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
            const __m512i cLhsMat2303Sp2 = _mm512_shuffle_epi32(
                vLhsMat2303, (_MM_PERM_ENUM)245); // A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31)
                                                  // A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)
                                                  // A02(28-31) A02(28-31) A03(28-31) A03(28-31)

            const __m512i cLhsMat0110Sp2 = _mm512_shuffle_epi32(
                vLhsMat0110,
                (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                     // A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m512i cLhsMat2310Sp2 = _mm512_shuffle_epi32(
                vLhsMat2310,
                (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7)
                                     // A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
            const __m512i cLhsMat0111Sp2 = _mm512_shuffle_epi32(
                vLhsMat0111, (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15)
                                                  // A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                                  // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
            const __m512i cLhsMat2311Sp2 = _mm512_shuffle_epi32(
                vLhsMat2311, (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15)
                                                  // A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                                  // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
            const __m512i cLhsMat0112Sp2 = _mm512_shuffle_epi32(
                vLhsMat0112, (_MM_PERM_ENUM)245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23)
                                                  // A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                                  // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
            const __m512i cLhsMat2312Sp2 = _mm512_shuffle_epi32(
                vLhsMat2312, (_MM_PERM_ENUM)245); // A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23)
                                                  // A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                                                  // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
            const __m512i cLhsMat0113Sp2 = _mm512_shuffle_epi32(
                vLhsMat0113, (_MM_PERM_ENUM)245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31)
                                                  // A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                                  // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
            const __m512i cLhsMat2313Sp2 = _mm512_shuffle_epi32(
                vLhsMat2313, (_MM_PERM_ENUM)245); // A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31)
                                                  // A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)
                                                  // A12(28-31) A12(28-31) A13(28-31) A13(28-31)

            // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
            // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
            __m512i vIaccMat000Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp1, cLhsMat0103Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd02Sp1, cLhsMat0102Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat0101Sp1)),
                _mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat0100Sp1));
            __m512i vIaccMat010Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp1, cLhsMat0103Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef02Sp1, cLhsMat0102Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat0101Sp1)),
                _mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat0100Sp1));
            __m512i vIaccMat100Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp1, cLhsMat2303Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd02Sp1, cLhsMat2302Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat2301Sp1)),
                _mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat2300Sp1));
            __m512i vIaccMat110Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp1, cLhsMat2303Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef02Sp1, cLhsMat2302Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat2301Sp1)),
                _mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat2300Sp1));
            __m512i vIaccMat001Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp1, cLhsMat0113Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd12Sp1, cLhsMat0112Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat0111Sp1)),
                _mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat0110Sp1));
            __m512i vIaccMat011Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp1, cLhsMat0113Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef12Sp1, cLhsMat0112Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat0111Sp1)),
                _mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat0110Sp1));
            __m512i vIaccMat101Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp1, cLhsMat2313Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd12Sp1, cLhsMat2312Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat2311Sp1)),
                _mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat2310Sp1));
            __m512i vIaccMat111Sp1 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp1, cLhsMat2313Sp1),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef12Sp1, cLhsMat2312Sp1)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat2311Sp1)),
                _mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat2310Sp1));

            __m512i vIaccMat000Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp2, cLhsMat0103Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd02Sp2, cLhsMat0102Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat0101Sp2)),
                _mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat0100Sp2));
            __m512i vIaccMat010Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp2, cLhsMat0103Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef02Sp2, cLhsMat0102Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat0101Sp2)),
                _mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat0100Sp2));
            __m512i vIaccMat100Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp2, cLhsMat2303Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd02Sp2, cLhsMat2302Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat2301Sp2)),
                _mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat2300Sp2));
            __m512i vIaccMat110Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp2, cLhsMat2303Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef02Sp2, cLhsMat2302Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat2301Sp2)),
                _mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat2300Sp2));
            __m512i vIaccMat001Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp2, cLhsMat0113Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd12Sp2, cLhsMat0112Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat0111Sp2)),
                _mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat0110Sp2));
            __m512i vIaccMat011Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp2, cLhsMat0113Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef12Sp2, cLhsMat0112Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat0111Sp2)),
                _mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat0110Sp2));
            __m512i vIaccMat101Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp2, cLhsMat2313Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat014589Cd12Sp2, cLhsMat2312Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat2311Sp2)),
                _mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat2310Sp2));
            __m512i vIaccMat111Sp2 = _mm512_add_epi16(
                _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp2, cLhsMat2313Sp2),
                                                  _mm512_maddubs_epi16(cRhsMat2367Abef12Sp2, cLhsMat2312Sp2)),
                                 _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat2311Sp2)),
                _mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat2310Sp2));

            // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
            __m512i vIaccMat000 = _mm512_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
            __m512i vIaccMat010 = _mm512_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
            __m512i vIaccMat100 = _mm512_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
            __m512i vIaccMat110 = _mm512_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

            __m512i vIaccMat001 = _mm512_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
            __m512i vIaccMat011 = _mm512_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
            __m512i vIaccMat101 = _mm512_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
            __m512i vIaccMat111 = _mm512_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

            vIaccMat000 = _mm512_madd_epi16(vIaccMat000, cScale014589Cd0);
            vIaccMat010 = _mm512_madd_epi16(vIaccMat010, cScale2367Abef0);
            vIaccMat100 = _mm512_madd_epi16(vIaccMat100, cScale014589Cd0);
            vIaccMat110 = _mm512_madd_epi16(vIaccMat110, cScale2367Abef0);

            vIaccMat001 = _mm512_madd_epi16(vIaccMat001, cScale014589Cd1);
            vIaccMat011 = _mm512_madd_epi16(vIaccMat011, cScale2367Abef1);
            vIaccMat101 = _mm512_madd_epi16(vIaccMat101, cScale014589Cd1);
            vIaccMat111 = _mm512_madd_epi16(vIaccMat111, cScale2367Abef1);

            // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next
            // step)
            __m512i vIaccRow00 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat000, _mm512_shuffle_epi32(vIaccMat010, (_MM_PERM_ENUM)78));
            __m512i vIaccRow10 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat000, (_MM_PERM_ENUM)78), vIaccMat010);
            __m512i vIaccRow20 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat100, _mm512_shuffle_epi32(vIaccMat110, (_MM_PERM_ENUM)78));
            __m512i vIaccRow30 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat100, (_MM_PERM_ENUM)78), vIaccMat110);
            __m512i vIaccRow01 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat001, _mm512_shuffle_epi32(vIaccMat011, (_MM_PERM_ENUM)78));
            __m512i vIaccRow11 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat001, (_MM_PERM_ENUM)78), vIaccMat011);
            __m512i vIaccRow21 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat101, _mm512_shuffle_epi32(vIaccMat111, (_MM_PERM_ENUM)78));
            __m512i vIaccRow31 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat101, (_MM_PERM_ENUM)78), vIaccMat111);

            __m512i vIaccRow0 = _mm512_add_epi32(vIaccRow00, vIaccRow01);
            __m512i vIaccRow1 = _mm512_add_epi32(vIaccRow10, vIaccRow11);
            __m512i vIaccRow2 = _mm512_add_epi32(vIaccRow20, vIaccRow21);
            __m512i vIaccRow3 = _mm512_add_epi32(vIaccRow30, vIaccRow31);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
            const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtrs[vRp][vB].d);
            const __m256 cRowScaleF32Ymm = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);
            const __m512 cRowScaleF32 = _mm512_insertf32x8(_mm512_castps256_ps512(cRowScaleF32Ymm), cRowScaleF32Ymm, 1);

            // Multiply with appropriate scales and accumulate (for both d and dmin) below
            aAccRows[vRp * 4] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow0),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
            aAccRows[vRp * 4 + 1] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow1),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
            aAccRows[vRp * 4 + 2] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow2),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
            aAccRows[vRp * 4 + 3] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow3),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);

            __m512i vIaccRowMin0 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)0), cMins01);
            __m512i vIaccRowMin1 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)85), cMins01);
            __m512i vIaccRowMin2 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)170), cMins01);
            __m512i vIaccRowMin3 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)255), cMins01);

            aAccMinRows[vRp * 4] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRowMin0),
                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccMinRows[vRp * 4]);
            aAccMinRows[vRp * 4 + 1] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin1),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                aAccMinRows[vRp * 4 + 1]);
            aAccMinRows[vRp * 4 + 2] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin2),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                aAccMinRows[vRp * 4 + 2]);
            aAccMinRows[vRp * 4 + 3] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin3),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                aAccMinRows[vRp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm512_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }

  for (; vY < pNr / 4; vY++) {

    const block_q8_Kx4 *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_q4_Kx8 *vBPtr0 = vBPtrStart + ((vX)*vBNb);
      const block_q4_Kx8 *vBPtr1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      __m512 aAccMinRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccMinRows[vI] = _mm512_setzero_ps();
      }

      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {
        // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
        const __m512 cColScaleF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].d, vBPtr1[vB].d);

        // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
        const __m512 cColDminF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].dmin, vBPtr1[vB].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {

          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + vSb * 256));
          const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat89Ab2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMatCdef2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat89Ab3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMatCdef3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          const __m256i cRhsRawMat89Cd0 =
              _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef0 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
          const __m256i cRhsRawMat89Cd1 =
              _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef1 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);
          const __m256i cRhsRawMat89Cd2 =
              _mm256_blend_epi32(cRhsRawMat89Ab2, _mm256_permutevar8x32_epi32(cRhsRawMatCdef2, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef2 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab2, vRequiredOrder), cRhsRawMatCdef2, 240);
          const __m256i cRhsRawMat89Cd3 =
              _mm256_blend_epi32(cRhsRawMat89Ab3, _mm256_permutevar8x32_epi32(cRhsRawMatCdef3, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef3 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab3, vRequiredOrder), cRhsRawMatCdef3, 240);

          const __m512i cRhsRawMat014589Cd0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
          const __m512i cRhsRawMat2367Abef0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
          const __m512i cRhsRawMat014589Cd1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
          const __m512i cRhsRawMat2367Abef1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

          const __m512i cRhsRawMat014589Cd2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01452), cRhsRawMat89Cd2, 1);
          const __m512i cRhsRawMat2367Abef2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23672), cRhsRawMatAbef2, 1);
          const __m512i cRhsRawMat014589Cd3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01453), cRhsRawMat89Cd3, 1);
          const __m512i cRhsRawMat2367Abef3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23673), cRhsRawMatAbef3, 1);

          // 4-bit -> 8-bit
          const __m512i cRhsMat014589Cd00 =
              _mm512_and_si512(cRhsRawMat014589Cd0,
                               cM4bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i cRhsMat2367Abef00 =
              _mm512_and_si512(cRhsRawMat2367Abef0,
                               cM4bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i cRhsMat014589Cd01 = _mm512_and_si512(
              cRhsRawMat014589Cd1,
              cM4bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i cRhsMat2367Abef01 = _mm512_and_si512(
              cRhsRawMat2367Abef1,
              cM4bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

          const __m512i cRhsMat014589Cd02 = _mm512_and_si512(
              cRhsRawMat014589Cd2,
              cM4bexpanded); // B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
          const __m512i cRhsMat2367Abef02 = _mm512_and_si512(
              cRhsRawMat2367Abef2,
              cM4bexpanded); // B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
          const __m512i cRhsMat014589Cd03 = _mm512_and_si512(
              cRhsRawMat014589Cd3,
              cM4bexpanded); // B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
          const __m512i cRhsMat2367Abef03 = _mm512_and_si512(
              cRhsRawMat2367Abef3,
              cM4bexpanded); // B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

          const __m512i cRhsMat014589Cd10 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                               cM4bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i cRhsMat2367Abef10 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                               cM4bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i cRhsMat014589Cd11 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
              cM4bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i cRhsMat2367Abef11 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
              cM4bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i cRhsMat014589Cd12 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd2, 4),
              cM4bexpanded); // B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
          const __m512i cRhsMat2367Abef12 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef2, 4),
              cM4bexpanded); // B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
          const __m512i cRhsMat014589Cd13 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 4),
              cM4bexpanded); // B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
          const __m512i cRhsMat2367Abef13 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 4),
              cM4bexpanded); // B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

          // Shuffle pattern one - right side input
          const __m512i cRhsMat014589Cd00Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3)
                                   // B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
          const __m512i cRhsMat2367Abef00Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3)
                                   // B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
          const __m512i cRhsMat014589Cd01Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef01Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                   // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
          const __m512i cRhsMat014589Cd02Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd02,
              (_MM_PERM_ENUM)136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19)
                                   // B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19)
                                   // B0C(16-19) B0D(16-19)
          const __m512i cRhsMat2367Abef02Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef02,
              (_MM_PERM_ENUM)136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19)
                                   // B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19)
                                   // B0E(16-19) B0F(16-19)
          const __m512i cRhsMat014589Cd03Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd03,
              (_MM_PERM_ENUM)136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27)
                                   // B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27)
                                   // B0C(24-27) B0D(24-27)
          const __m512i cRhsMat2367Abef03Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef03,
              (_MM_PERM_ENUM)136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27)
                                   // B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27)
                                   // B0E(24-27) B0F(24-27)

          const __m512i cRhsMat014589Cd10Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3)
                                   // B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
          const __m512i cRhsMat2367Abef10Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3)
                                   // B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
          const __m512i cRhsMat014589Cd11Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                   // B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i cRhsMat2367Abef11Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                   // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
          const __m512i cRhsMat014589Cd12Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd12,
              (_MM_PERM_ENUM)136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19)
                                   // B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19)
                                   // B1C(16-19) B1D(16-19)
          const __m512i cRhsMat2367Abef12Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef12,
              (_MM_PERM_ENUM)136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19)
                                   // B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19)
                                   // B1E(16-19) B1F(16-19)
          const __m512i cRhsMat014589Cd13Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd13,
              (_MM_PERM_ENUM)136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27)
                                   // B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27)
                                   // B1C(24-27) B1D(24-27)
          const __m512i cRhsMat2367Abef13Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef13,
              (_MM_PERM_ENUM)136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27)
                                   // B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27)
                                   // B1E(24-27) B1F(24-27)

          // Shuffle pattern two - right side input
          const __m512i cRhsMat014589Cd00Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7)
                                   // B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
          const __m512i cRhsMat2367Abef00Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7)
                                   // B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
          const __m512i cRhsMat014589Cd01Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15)
                                   // B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15)
                                   // B0C(12-15) B0D(12-15)
          const __m512i cRhsMat2367Abef01Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15)
                                   // B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15)
                                   // B0E(12-15) B0F(12-15)
          const __m512i cRhsMat014589Cd02Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd02,
              (_MM_PERM_ENUM)221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23)
                                   // B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23)
                                   // B0C(20-23) B0D(20-23)
          const __m512i cRhsMat2367Abef02Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef02,
              (_MM_PERM_ENUM)221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23)
                                   // B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23)
                                   // B0E(20-23) B0F(20-23)
          const __m512i cRhsMat014589Cd03Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd03,
              (_MM_PERM_ENUM)221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31)
                                   // B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31)
                                   // B0C(28-31) 0BD(28-31)
          const __m512i cRhsMat2367Abef03Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef03,
              (_MM_PERM_ENUM)221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31)
                                   // B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31)
                                   // B0E(28-31) B0F(28-31)

          const __m512i cRhsMat014589Cd10Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7)
                                   // B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
          const __m512i cRhsMat2367Abef10Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7)
                                   // B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
          const __m512i cRhsMat014589Cd11Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15)
                                   // B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15)
                                   // B1C(12-15) B1D(12-15)
          const __m512i cRhsMat2367Abef11Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15)
                                   // B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15)
                                   // B1E(12-15) B1F(12-15)
          const __m512i cRhsMat014589Cd12Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd12,
              (_MM_PERM_ENUM)221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23)
                                   // B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23)
                                   // B1C(20-23) B1D(20-23)
          const __m512i cRhsMat2367Abef12Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef12,
              (_MM_PERM_ENUM)221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23)
                                   // B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23)
                                   // B1E(20-23) B1F(20-23)
          const __m512i cRhsMat014589Cd13Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd13,
              (_MM_PERM_ENUM)221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31)
                                   // B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31)
                                   // B1C(28-31) B1D(28-31)
          const __m512i cRhsMat2367Abef13Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef13,
              (_MM_PERM_ENUM)221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31)
                                   // B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31)
                                   // B1E(28-31) B1F(28-31)

          uint32_t aUtmp00[4], aUtmp01[4], aUtmp10[4], aUtmp11[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp00, vBPtr0[vB].scales + 24 * vSb, 12);
          aUtmp00[3] = ((aUtmp00[2] >> 4) & cKmask2) | (((aUtmp00[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux00 = aUtmp00[1] & cKmask1;
          aUtmp00[1] = (aUtmp00[2] & cKmask2) | (((aUtmp00[0] >> 6) & cKmask3) << 4);
          aUtmp00[2] = cUaux00;
          aUtmp00[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp01, vBPtr0[vB].scales + 12 + vSb * 24, 12);
          aUtmp01[3] = ((aUtmp01[2] >> 4) & cKmask2) | (((aUtmp01[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux01 = aUtmp01[1] & cKmask1;
          aUtmp01[1] = (aUtmp01[2] & cKmask2) | (((aUtmp01[0] >> 6) & cKmask3) << 4);
          aUtmp01[2] = cUaux01;
          aUtmp01[0] &= cKmask1;

          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp10, vBPtr1[vB].scales + vSb * 24, 12);
          aUtmp10[3] = ((aUtmp10[2] >> 4) & cKmask2) | (((aUtmp10[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux10 = aUtmp10[1] & cKmask1;
          aUtmp10[1] = (aUtmp10[2] & cKmask2) | (((aUtmp10[0] >> 6) & cKmask3) << 4);
          aUtmp10[2] = cUaux10;
          aUtmp10[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp11, vBPtr1[vB].scales + 12 + vSb * 24, 12);
          aUtmp11[3] = ((aUtmp11[2] >> 4) & cKmask2) | (((aUtmp11[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux11 = aUtmp11[1] & cKmask1;
          aUtmp11[1] = (aUtmp11[2] & cKmask2) | (((aUtmp11[0] >> 6) & cKmask3) << 4);
          aUtmp11[2] = cUaux11;
          aUtmp11[0] &= cKmask1;

          // Scales of first sub block in the sb loop
          const __m256i cMinsAndScales0 = _mm256_set_epi32(aUtmp10[3], aUtmp10[2], aUtmp10[1], aUtmp10[0], aUtmp00[3],
                                                           aUtmp00[2], aUtmp00[1], aUtmp00[0]);
          const __m512i cScales0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(cMinsAndScales0, cMinsAndScales0));

          // Scales of second sub block in the sb loop
          const __m256i cMinsAndScales1 = _mm256_set_epi32(aUtmp11[3], aUtmp11[2], aUtmp11[1], aUtmp11[0], aUtmp01[3],
                                                           aUtmp01[2], aUtmp01[1], aUtmp01[0]);
          const __m512i cScales1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(cMinsAndScales1, cMinsAndScales1));

          // Mins of first and second sub block of Q4_K block are arranged side by side
          const __m512i cMins01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(cMinsAndScales0, 78),
                                                                            _mm256_shuffle_epi32(cMinsAndScales1, 78)));

          const __m512i cScale014589Cd0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)238);

          // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
          // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
          __m256i vLhsMatYmm012300 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 256 * vSb)));
          __m256i vLhsMatYmm0100 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 0);
          __m256i vLhsMatYmm2300 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 17);
          __m256i vLhsMatYmm012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32 + 256 * vSb)));
          __m256i vLhsMatYmm0101 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 0);
          __m256i vLhsMatYmm2301 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 17);
          __m256i vLhsMatYmm012302 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64 + 256 * vSb)));
          __m256i vLhsMatYmm0102 = _mm256_permute2f128_si256(vLhsMatYmm012302, vLhsMatYmm012302, 0);
          __m256i vLhsMatYmm2302 = _mm256_permute2f128_si256(vLhsMatYmm012302, vLhsMatYmm012302, 17);
          __m256i vLhsMatYmm012303 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96 + 256 * vSb)));
          __m256i vLhsMatYmm0103 = _mm256_permute2f128_si256(vLhsMatYmm012303, vLhsMatYmm012303, 0);
          __m256i vLhsMatYmm2303 = _mm256_permute2f128_si256(vLhsMatYmm012303, vLhsMatYmm012303, 17);
          __m256i vLhsMatYmm012310 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 128 + 256 * vSb)));
          __m256i vLhsMatYmm0110 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 0);
          __m256i vLhsMatYmm2310 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 17);
          __m256i vLhsMatYmm012311 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 160 + 256 * vSb)));
          __m256i vLhsMatYmm0111 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 0);
          __m256i vLhsMatYmm2311 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 17);
          __m256i vLhsMatYmm012312 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 192 + 256 * vSb)));
          __m256i vLhsMatYmm0112 = _mm256_permute2f128_si256(vLhsMatYmm012312, vLhsMatYmm012312, 0);
          __m256i vLhsMatYmm2312 = _mm256_permute2f128_si256(vLhsMatYmm012312, vLhsMatYmm012312, 17);
          __m256i vLhsMatYmm012313 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 224 + 256 * vSb)));
          __m256i vLhsMatYmm0113 = _mm256_permute2f128_si256(vLhsMatYmm012313, vLhsMatYmm012313, 0);
          __m256i vLhsMatYmm2313 = _mm256_permute2f128_si256(vLhsMatYmm012313, vLhsMatYmm012313, 17);

          // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into
          // a 512 bit vector
          __m512i vLhsMat0100 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0100), vLhsMatYmm0100, 1);
          __m512i vLhsMat2300 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2300), vLhsMatYmm2300, 1);
          __m512i vLhsMat0101 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0101), vLhsMatYmm0101, 1);
          __m512i vLhsMat2301 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2301), vLhsMatYmm2301, 1);
          __m512i vLhsMat0102 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0102), vLhsMatYmm0102, 1);
          __m512i vLhsMat2302 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2302), vLhsMatYmm2302, 1);
          __m512i vLhsMat0103 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0103), vLhsMatYmm0103, 1);
          __m512i vLhsMat2303 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2303), vLhsMatYmm2303, 1);

          __m512i vLhsMat0110 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0110), vLhsMatYmm0110, 1);
          __m512i vLhsMat2310 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2310), vLhsMatYmm2310, 1);
          __m512i vLhsMat0111 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0111), vLhsMatYmm0111, 1);
          __m512i vLhsMat2311 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2311), vLhsMatYmm2311, 1);
          __m512i vLhsMat0112 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0112), vLhsMatYmm0112, 1);
          __m512i vLhsMat2312 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2312), vLhsMatYmm2312, 1);
          __m512i vLhsMat0113 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0113), vLhsMatYmm0113, 1);
          __m512i vLhsMat2313 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2313), vLhsMatYmm2313, 1);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
          __m256i vLhsBsums012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].bsums + 16 * vSb)));
          __m256i vLhsBsumsHsumYmm012301 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(vLhsBsums012301), _mm256_extractf128_si256(vLhsBsums012301, 1)));
          vLhsBsumsHsumYmm012301 = _mm256_permute2x128_si256(vLhsBsumsHsumYmm012301, vLhsBsumsHsumYmm012301, 0);
          __m512i vLhsBsumsHsum012301 =
              _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsHsumYmm012301), vLhsBsumsHsumYmm012301, 1);

          // Shuffle pattern one - left side input
          const __m512i cLhsMat0100Sp1 = _mm512_shuffle_epi32(
              vLhsMat0100,
              (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                   // A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
          const __m512i cLhsMat2300Sp1 = _mm512_shuffle_epi32(
              vLhsMat2300,
              (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3)
                                   // A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
          const __m512i cLhsMat0101Sp1 = _mm512_shuffle_epi32(
              vLhsMat0101,
              (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m512i cLhsMat2301Sp1 = _mm512_shuffle_epi32(
              vLhsMat2301,
              (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                   // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
          const __m512i cLhsMat0102Sp1 = _mm512_shuffle_epi32(
              vLhsMat0102, (_MM_PERM_ENUM)160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19)
                                                // A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                                // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m512i cLhsMat2302Sp1 = _mm512_shuffle_epi32(
              vLhsMat2302, (_MM_PERM_ENUM)160); // A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19)
                                                // A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                                                // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
          const __m512i cLhsMat0103Sp1 = _mm512_shuffle_epi32(
              vLhsMat0103, (_MM_PERM_ENUM)160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27)
                                                // A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                                // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m512i cLhsMat2303Sp1 = _mm512_shuffle_epi32(
              vLhsMat2303, (_MM_PERM_ENUM)160); // A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27)
                                                // A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)
                                                // A02(24-27) A02(24-27) A03(24-27) A03(24-27)

          const __m512i cLhsMat0110Sp1 = _mm512_shuffle_epi32(
              vLhsMat0110,
              (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                   // A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
          const __m512i cLhsMat2310Sp1 = _mm512_shuffle_epi32(
              vLhsMat2310,
              (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3)
                                   // A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
          const __m512i cLhsMat0111Sp1 = _mm512_shuffle_epi32(
              vLhsMat0111,
              (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m512i cLhsMat2311Sp1 = _mm512_shuffle_epi32(
              vLhsMat2311,
              (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                   // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
          const __m512i cLhsMat0112Sp1 = _mm512_shuffle_epi32(
              vLhsMat0112, (_MM_PERM_ENUM)160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19)
                                                // A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                                // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m512i cLhsMat2312Sp1 = _mm512_shuffle_epi32(
              vLhsMat2312, (_MM_PERM_ENUM)160); // A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19)
                                                // A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                                                // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
          const __m512i cLhsMat0113Sp1 = _mm512_shuffle_epi32(
              vLhsMat0113, (_MM_PERM_ENUM)160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27)
                                                // A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                                // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m512i cLhsMat2313Sp1 = _mm512_shuffle_epi32(
              vLhsMat2313, (_MM_PERM_ENUM)160); // A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27)
                                                // A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)
                                                // A12(24-27) A12(24-27) A13(24-27) A13(24-27)

          const __m512i cLhsMat0100Sp2 = _mm512_shuffle_epi32(
              vLhsMat0100,
              (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                   // A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
          const __m512i cLhsMat2300Sp2 = _mm512_shuffle_epi32(
              vLhsMat2300,
              (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7)
                                   // A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
          const __m512i cLhsMat0101Sp2 = _mm512_shuffle_epi32(
              vLhsMat0101, (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15)
                                                // A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                                // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m512i cLhsMat2301Sp2 = _mm512_shuffle_epi32(
              vLhsMat2301, (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15)
                                                // A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                                // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
          const __m512i cLhsMat0102Sp2 = _mm512_shuffle_epi32(
              vLhsMat0102, (_MM_PERM_ENUM)245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23)
                                                // A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                                // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m512i cLhsMat2302Sp2 = _mm512_shuffle_epi32(
              vLhsMat2302, (_MM_PERM_ENUM)245); // A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23)
                                                // A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                                                // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
          const __m512i cLhsMat0103Sp2 = _mm512_shuffle_epi32(
              vLhsMat0103, (_MM_PERM_ENUM)245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31)
                                                // A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                                // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m512i cLhsMat2303Sp2 = _mm512_shuffle_epi32(
              vLhsMat2303, (_MM_PERM_ENUM)245); // A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31)
                                                // A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)
                                                // A02(28-31) A02(28-31) A03(28-31) A03(28-31)

          const __m512i cLhsMat0110Sp2 = _mm512_shuffle_epi32(
              vLhsMat0110,
              (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                   // A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
          const __m512i cLhsMat2310Sp2 = _mm512_shuffle_epi32(
              vLhsMat2310,
              (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7)
                                   // A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
          const __m512i cLhsMat0111Sp2 = _mm512_shuffle_epi32(
              vLhsMat0111, (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15)
                                                // A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                                // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m512i cLhsMat2311Sp2 = _mm512_shuffle_epi32(
              vLhsMat2311, (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15)
                                                // A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                                // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
          const __m512i cLhsMat0112Sp2 = _mm512_shuffle_epi32(
              vLhsMat0112, (_MM_PERM_ENUM)245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23)
                                                // A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                                // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m512i cLhsMat2312Sp2 = _mm512_shuffle_epi32(
              vLhsMat2312, (_MM_PERM_ENUM)245); // A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23)
                                                // A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                                                // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
          const __m512i cLhsMat0113Sp2 = _mm512_shuffle_epi32(
              vLhsMat0113, (_MM_PERM_ENUM)245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31)
                                                // A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                                // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m512i cLhsMat2313Sp2 = _mm512_shuffle_epi32(
              vLhsMat2313, (_MM_PERM_ENUM)245); // A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31)
                                                // A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)
                                                // A12(28-31) A12(28-31) A13(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
          __m512i vIaccMat000Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp1, cLhsMat0103Sp1),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd02Sp1, cLhsMat0102Sp1)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat0101Sp1)),
              _mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat0100Sp1));
          __m512i vIaccMat010Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp1, cLhsMat0103Sp1),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef02Sp1, cLhsMat0102Sp1)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat0101Sp1)),
              _mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat0100Sp1));
          __m512i vIaccMat100Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp1, cLhsMat2303Sp1),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd02Sp1, cLhsMat2302Sp1)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat2301Sp1)),
              _mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat2300Sp1));
          __m512i vIaccMat110Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp1, cLhsMat2303Sp1),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef02Sp1, cLhsMat2302Sp1)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat2301Sp1)),
              _mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat2300Sp1));
          __m512i vIaccMat001Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp1, cLhsMat0113Sp1),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd12Sp1, cLhsMat0112Sp1)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat0111Sp1)),
              _mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat0110Sp1));
          __m512i vIaccMat011Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp1, cLhsMat0113Sp1),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef12Sp1, cLhsMat0112Sp1)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat0111Sp1)),
              _mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat0110Sp1));
          __m512i vIaccMat101Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp1, cLhsMat2313Sp1),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd12Sp1, cLhsMat2312Sp1)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat2311Sp1)),
              _mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat2310Sp1));
          __m512i vIaccMat111Sp1 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp1, cLhsMat2313Sp1),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef12Sp1, cLhsMat2312Sp1)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat2311Sp1)),
              _mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat2310Sp1));

          __m512i vIaccMat000Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp2, cLhsMat0103Sp2),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd02Sp2, cLhsMat0102Sp2)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat0101Sp2)),
              _mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat0100Sp2));
          __m512i vIaccMat010Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp2, cLhsMat0103Sp2),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef02Sp2, cLhsMat0102Sp2)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat0101Sp2)),
              _mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat0100Sp2));
          __m512i vIaccMat100Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd03Sp2, cLhsMat2303Sp2),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd02Sp2, cLhsMat2302Sp2)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat2301Sp2)),
              _mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat2300Sp2));
          __m512i vIaccMat110Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef03Sp2, cLhsMat2303Sp2),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef02Sp2, cLhsMat2302Sp2)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat2301Sp2)),
              _mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat2300Sp2));
          __m512i vIaccMat001Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp2, cLhsMat0113Sp2),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd12Sp2, cLhsMat0112Sp2)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat0111Sp2)),
              _mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat0110Sp2));
          __m512i vIaccMat011Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp2, cLhsMat0113Sp2),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef12Sp2, cLhsMat0112Sp2)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat0111Sp2)),
              _mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat0110Sp2));
          __m512i vIaccMat101Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd13Sp2, cLhsMat2313Sp2),
                                                _mm512_maddubs_epi16(cRhsMat014589Cd12Sp2, cLhsMat2312Sp2)),
                               _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat2311Sp2)),
              _mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat2310Sp2));
          __m512i vIaccMat111Sp2 = _mm512_add_epi16(
              _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef13Sp2, cLhsMat2313Sp2),
                                                _mm512_maddubs_epi16(cRhsMat2367Abef12Sp2, cLhsMat2312Sp2)),
                               _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat2311Sp2)),
              _mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat2310Sp2));

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          __m512i vIaccMat000 = _mm512_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
          __m512i vIaccMat010 = _mm512_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
          __m512i vIaccMat100 = _mm512_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
          __m512i vIaccMat110 = _mm512_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

          __m512i vIaccMat001 = _mm512_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
          __m512i vIaccMat011 = _mm512_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
          __m512i vIaccMat101 = _mm512_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
          __m512i vIaccMat111 = _mm512_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

          vIaccMat000 = _mm512_madd_epi16(vIaccMat000, cScale014589Cd0);
          vIaccMat010 = _mm512_madd_epi16(vIaccMat010, cScale2367Abef0);
          vIaccMat100 = _mm512_madd_epi16(vIaccMat100, cScale014589Cd0);
          vIaccMat110 = _mm512_madd_epi16(vIaccMat110, cScale2367Abef0);

          vIaccMat001 = _mm512_madd_epi16(vIaccMat001, cScale014589Cd1);
          vIaccMat011 = _mm512_madd_epi16(vIaccMat011, cScale2367Abef1);
          vIaccMat101 = _mm512_madd_epi16(vIaccMat101, cScale014589Cd1);
          vIaccMat111 = _mm512_madd_epi16(vIaccMat111, cScale2367Abef1);

          // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
          __m512i vIaccRow00 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat000, _mm512_shuffle_epi32(vIaccMat010, (_MM_PERM_ENUM)78));
          __m512i vIaccRow10 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat000, (_MM_PERM_ENUM)78), vIaccMat010);
          __m512i vIaccRow20 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat100, _mm512_shuffle_epi32(vIaccMat110, (_MM_PERM_ENUM)78));
          __m512i vIaccRow30 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat100, (_MM_PERM_ENUM)78), vIaccMat110);
          __m512i vIaccRow01 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat001, _mm512_shuffle_epi32(vIaccMat011, (_MM_PERM_ENUM)78));
          __m512i vIaccRow11 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat001, (_MM_PERM_ENUM)78), vIaccMat011);
          __m512i vIaccRow21 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat101, _mm512_shuffle_epi32(vIaccMat111, (_MM_PERM_ENUM)78));
          __m512i vIaccRow31 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat101, (_MM_PERM_ENUM)78), vIaccMat111);

          __m512i vIaccRow0 = _mm512_add_epi32(vIaccRow00, vIaccRow01);
          __m512i vIaccRow1 = _mm512_add_epi32(vIaccRow10, vIaccRow11);
          __m512i vIaccRow2 = _mm512_add_epi32(vIaccRow20, vIaccRow21);
          __m512i vIaccRow3 = _mm512_add_epi32(vIaccRow30, vIaccRow31);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
          const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtr[vB].d);
          const __m256 cRowScaleF32Ymm = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);
          const __m512 cRowScaleF32 = _mm512_insertf32x8(_mm512_castps256_ps512(cRowScaleF32Ymm), cRowScaleF32Ymm, 1);

          // Multiply with appropriate scales and accumulate (for both d and dmin) below
          aAccRows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow0),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                        aAccRows[0]);
          aAccRows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow1),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                        aAccRows[1]);
          aAccRows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow2),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                        aAccRows[2]);
          aAccRows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow3),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                        aAccRows[3]);

          __m512i vIaccRowMin0 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)0), cMins01);
          __m512i vIaccRowMin1 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)85), cMins01);
          __m512i vIaccRowMin2 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)170), cMins01);
          __m512i vIaccRowMin3 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsumsHsum012301, (_MM_PERM_ENUM)255), cMins01);

          aAccMinRows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin0),
                                           _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                           aAccMinRows[0]);
          aAccMinRows[1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin1),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccMinRows[1]);
          aAccMinRows[2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin2),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccMinRows[2]);
          aAccMinRows[3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin3),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccMinRows[3]);
        }
      }
      // Store accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm512_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }
  if (vAnc != pNc) {
    vXstart = vAnc / 8;
    vY = 0;
  }
#endif // __AVX512BW__ && __AVX512DQ__

  // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
  for (; vY < vAnr / 4; vY += 4) {

    const block_q8_Kx4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {

      const block_q4_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      __m256 aAccMinRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccMinRows[vI] = _mm256_setzero_ps();
      }

      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {

        // Scale values - Load the eight scale values of block_q4_kx8
        const __m256 cColScaleF32 = GGML_F32Cx8_LOAD(vBPtr[vB].d);

        // dmin values - Load the eight dmin values of block_q4_kx8
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {

          // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and
          // storing of values
          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i cRhsMat014500 = _mm256_and_si256(cRhsRawMat01450, cM4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i cRhsMat236700 = _mm256_and_si256(cRhsRawMat23670, cM4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i cRhsMat014501 =
              _mm256_and_si256(cRhsRawMat01451, cM4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i cRhsMat236701 =
              _mm256_and_si256(cRhsRawMat23671, cM4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i cRhsMat014502 =
              _mm256_and_si256(cRhsRawMat01452, cM4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i cRhsMat236702 =
              _mm256_and_si256(cRhsRawMat23672, cM4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i cRhsMat014503 =
              _mm256_and_si256(cRhsRawMat01453, cM4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i cRhsMat236703 =
              _mm256_and_si256(cRhsRawMat23673, cM4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i cRhsMat014510 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4), cM4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i cRhsMat236710 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4), cM4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i cRhsMat014511 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4), cM4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i cRhsMat236711 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4), cM4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i cRhsMat014512 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 4),
                                                         cM4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i cRhsMat236712 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 4),
                                                         cM4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i cRhsMat014513 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 4),
                                                         cM4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i cRhsMat236713 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 4),
                                                         cM4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i cRhsMat014500Sp1 = _mm256_shuffle_epi32(
              cRhsMat014500, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i cRhsMat236700Sp1 = _mm256_shuffle_epi32(
              cRhsMat236700, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i cRhsMat014501Sp1 = _mm256_shuffle_epi32(
              cRhsMat014501, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i cRhsMat236701Sp1 = _mm256_shuffle_epi32(
              cRhsMat236701, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i cRhsMat014502Sp1 = _mm256_shuffle_epi32(
              cRhsMat014502,
              136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
          const __m256i cRhsMat236702Sp1 = _mm256_shuffle_epi32(
              cRhsMat236702,
              136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

          const __m256i cRhsMat014503Sp1 = _mm256_shuffle_epi32(
              cRhsMat014503,
              136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
          const __m256i cRhsMat236703Sp1 = _mm256_shuffle_epi32(
              cRhsMat236703,
              136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

          const __m256i cRhsMat014510Sp1 = _mm256_shuffle_epi32(
              cRhsMat014510, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i cRhsMat236710Sp1 = _mm256_shuffle_epi32(
              cRhsMat236710, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i cRhsMat014511Sp1 = _mm256_shuffle_epi32(
              cRhsMat014511, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i cRhsMat236711Sp1 = _mm256_shuffle_epi32(
              cRhsMat236711, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i cRhsMat014512Sp1 = _mm256_shuffle_epi32(
              cRhsMat014512,
              136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
          const __m256i cRhsMat236712Sp1 = _mm256_shuffle_epi32(
              cRhsMat236712,
              136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

          const __m256i cRhsMat014513Sp1 = _mm256_shuffle_epi32(
              cRhsMat014513,
              136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
          const __m256i cRhsMat236713Sp1 = _mm256_shuffle_epi32(
              cRhsMat236713,
              136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i cRhsMat014500Sp2 = _mm256_shuffle_epi32(
              cRhsMat014500, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i cRhsMat236700Sp2 = _mm256_shuffle_epi32(
              cRhsMat236700, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i cRhsMat014501Sp2 = _mm256_shuffle_epi32(
              cRhsMat014501,
              221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
          const __m256i cRhsMat236701Sp2 = _mm256_shuffle_epi32(
              cRhsMat236701,
              221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

          const __m256i cRhsMat014502Sp2 = _mm256_shuffle_epi32(
              cRhsMat014502,
              221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
          const __m256i cRhsMat236702Sp2 = _mm256_shuffle_epi32(
              cRhsMat236702,
              221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

          const __m256i cRhsMat014503Sp2 = _mm256_shuffle_epi32(
              cRhsMat014503,
              221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
          const __m256i cRhsMat236703Sp2 = _mm256_shuffle_epi32(
              cRhsMat236703,
              221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

          const __m256i cRhsMat014510Sp2 = _mm256_shuffle_epi32(
              cRhsMat014510, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i cRhsMat236710Sp2 = _mm256_shuffle_epi32(
              cRhsMat236710, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i cRhsMat014511Sp2 = _mm256_shuffle_epi32(
              cRhsMat014511,
              221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
          const __m256i cRhsMat236711Sp2 = _mm256_shuffle_epi32(
              cRhsMat236711,
              221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

          const __m256i cRhsMat014512Sp2 = _mm256_shuffle_epi32(
              cRhsMat014512,
              221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
          const __m256i cRhsMat236712Sp2 = _mm256_shuffle_epi32(
              cRhsMat236712,
              221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

          const __m256i cRhsMat014513Sp2 = _mm256_shuffle_epi32(
              cRhsMat014513,
              221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
          const __m256i cRhsMat236713Sp2 = _mm256_shuffle_epi32(
              cRhsMat236713,
              221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

          uint32_t aUtmp0[4], aUtmp1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp0, vBPtr[vB].scales + 24 * vSb, 12);
          aUtmp0[3] = ((aUtmp0[2] >> 4) & cKmask2) | (((aUtmp0[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = aUtmp0[1] & cKmask1;
          aUtmp0[1] = (aUtmp0[2] & cKmask2) | (((aUtmp0[0] >> 6) & cKmask3) << 4);
          aUtmp0[2] = cUaux0;
          aUtmp0[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp1, vBPtr[vB].scales + 12 + vSb * 24, 12);
          aUtmp1[3] = ((aUtmp1[2] >> 4) & cKmask2) | (((aUtmp1[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux1 = aUtmp1[1] & cKmask1;
          aUtmp1[1] = (aUtmp1[2] & cKmask2) | (((aUtmp1[0] >> 6) & cKmask3) << 4);
          aUtmp1[2] = cUaux1;
          aUtmp1[0] &= cKmask1;

          // Scales of first sub block in the sb loop
          const __m128i cMinsAndScales0 = _mm_set_epi32(aUtmp0[3], aUtmp0[2], aUtmp0[1], aUtmp0[0]);
          const __m256i cScales0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(cMinsAndScales0, cMinsAndScales0));

          // Scales of second sub block in the sb loop
          const __m128i cMinsAndScales1 = _mm_set_epi32(aUtmp1[3], aUtmp1[2], aUtmp1[1], aUtmp1[0]);
          const __m256i cScales1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(cMinsAndScales1, cMinsAndScales1));

          // Mins of first and second sub block of Q4_K block are arranged side by side
          const __m256i cMins01 = _mm256_cvtepu8_epi16(
              _mm_unpacklo_epi8(_mm_shuffle_epi32(cMinsAndScales0, 78), _mm_shuffle_epi32(cMinsAndScales1, 78)));

          const __m256i cScale01450 = _mm256_shuffle_epi32(cScales0, 68);
          const __m256i cScale23670 = _mm256_shuffle_epi32(cScales0, 238);

          const __m256i cScale01451 = _mm256_shuffle_epi32(cScales1, 68);
          const __m256i cScale23671 = _mm256_shuffle_epi32(cScales1, 238);

          for (int vRp = 0; vRp < 4; vRp++) {

            // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
            // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
            __m256i vLhsMat012300 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 256 * vSb)));
            __m256i vLhsMat0100 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 0);
            __m256i vLhsMat2300 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 17);
            __m256i vLhsMat012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32 + 256 * vSb)));
            __m256i vLhsMat0101 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 0);
            __m256i vLhsMat2301 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 17);
            __m256i vLhsMat012302 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64 + 256 * vSb)));
            __m256i vLhsMat0102 = _mm256_permute2f128_si256(vLhsMat012302, vLhsMat012302, 0);
            __m256i vLhsMat2302 = _mm256_permute2f128_si256(vLhsMat012302, vLhsMat012302, 17);
            __m256i vLhsMat012303 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96 + 256 * vSb)));
            __m256i vLhsMat0103 = _mm256_permute2f128_si256(vLhsMat012303, vLhsMat012303, 0);
            __m256i vLhsMat2303 = _mm256_permute2f128_si256(vLhsMat012303, vLhsMat012303, 17);
            __m256i vLhsMat012310 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 128 + 256 * vSb)));
            __m256i vLhsMat0110 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 0);
            __m256i vLhsMat2310 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 17);
            __m256i vLhsMat012311 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 160 + 256 * vSb)));
            __m256i vLhsMat0111 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 0);
            __m256i vLhsMat2311 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 17);
            __m256i vLhsMat012312 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 192 + 256 * vSb)));
            __m256i vLhsMat0112 = _mm256_permute2f128_si256(vLhsMat012312, vLhsMat012312, 0);
            __m256i vLhsMat2312 = _mm256_permute2f128_si256(vLhsMat012312, vLhsMat012312, 17);
            __m256i vLhsMat012313 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 224 + 256 * vSb)));
            __m256i vLhsMat0113 = _mm256_permute2f128_si256(vLhsMat012313, vLhsMat012313, 0);
            __m256i vLhsMat2313 = _mm256_permute2f128_si256(vLhsMat012313, vLhsMat012313, 17);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
            __m256i vLhsBsums012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].bsums + 16 * vSb)));
            __m256i vLhsBsumsHsum012301 = _mm256_castsi128_si256(
                _mm_hadd_epi16(_mm256_castsi256_si128(vLhsBsums012301), _mm256_extractf128_si256(vLhsBsums012301, 1)));
            vLhsBsumsHsum012301 = _mm256_permute2x128_si256(vLhsBsumsHsum012301, vLhsBsumsHsum012301, 0);

            // Shuffle pattern one - left side input
            const __m256i cLhsMat0100Sp1 = _mm256_shuffle_epi32(
                vLhsMat0100, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m256i cLhsMat2300Sp1 = _mm256_shuffle_epi32(
                vLhsMat2300, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

            const __m256i cLhsMat0101Sp1 = _mm256_shuffle_epi32(
                vLhsMat0101, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m256i cLhsMat2301Sp1 = _mm256_shuffle_epi32(
                vLhsMat2301, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

            const __m256i cLhsMat0102Sp1 = _mm256_shuffle_epi32(
                vLhsMat0102,
                160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
            const __m256i cLhsMat2302Sp1 = _mm256_shuffle_epi32(
                vLhsMat2302,
                160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

            const __m256i cLhsMat0103Sp1 = _mm256_shuffle_epi32(
                vLhsMat0103,
                160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
            const __m256i cLhsMat2303Sp1 = _mm256_shuffle_epi32(
                vLhsMat2303,
                160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

            const __m256i cLhsMat0110Sp1 = _mm256_shuffle_epi32(
                vLhsMat0110, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m256i cLhsMat2310Sp1 = _mm256_shuffle_epi32(
                vLhsMat2310, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

            const __m256i cLhsMat0111Sp1 = _mm256_shuffle_epi32(
                vLhsMat0111, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m256i cLhsMat2311Sp1 = _mm256_shuffle_epi32(
                vLhsMat2311, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

            const __m256i cLhsMat0112Sp1 = _mm256_shuffle_epi32(
                vLhsMat0112,
                160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
            const __m256i cLhsMat2312Sp1 = _mm256_shuffle_epi32(
                vLhsMat2312,
                160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

            const __m256i cLhsMat0113Sp1 = _mm256_shuffle_epi32(
                vLhsMat0113,
                160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
            const __m256i cLhsMat2313Sp1 = _mm256_shuffle_epi32(
                vLhsMat2313,
                160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

            // Shuffle pattern two- left side input
            const __m256i cLhsMat0100Sp2 = _mm256_shuffle_epi32(
                vLhsMat0100, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m256i cLhsMat2300Sp2 = _mm256_shuffle_epi32(
                vLhsMat2300, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

            const __m256i cLhsMat0101Sp2 = _mm256_shuffle_epi32(
                vLhsMat0101,
                245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
            const __m256i cLhsMat2301Sp2 = _mm256_shuffle_epi32(
                vLhsMat2301,
                245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

            const __m256i cLhsMat0102Sp2 = _mm256_shuffle_epi32(
                vLhsMat0102,
                245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
            const __m256i cLhsMat2302Sp2 = _mm256_shuffle_epi32(
                vLhsMat2302,
                245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

            const __m256i cLhsMat0103Sp2 = _mm256_shuffle_epi32(
                vLhsMat0103,
                245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
            const __m256i cLhsMat2303Sp2 = _mm256_shuffle_epi32(
                vLhsMat2303,
                245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

            const __m256i cLhsMat0110Sp2 = _mm256_shuffle_epi32(
                vLhsMat0110, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m256i cLhsMat2310Sp2 = _mm256_shuffle_epi32(
                vLhsMat2310, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

            const __m256i cLhsMat0111Sp2 = _mm256_shuffle_epi32(
                vLhsMat0111,
                245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
            const __m256i cLhsMat2311Sp2 = _mm256_shuffle_epi32(
                vLhsMat2311,
                245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

            const __m256i cLhsMat0112Sp2 = _mm256_shuffle_epi32(
                vLhsMat0112,
                245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
            const __m256i cLhsMat2312Sp2 = _mm256_shuffle_epi32(
                vLhsMat2312,
                245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

            const __m256i cLhsMat0113Sp2 = _mm256_shuffle_epi32(
                vLhsMat0113,
                245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
            const __m256i cLhsMat2313Sp2 = _mm256_shuffle_epi32(
                vLhsMat2313,
                245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

            // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
            // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
            __m256i vIaccMat000Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp1, cLhsMat0103Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat014502Sp1, cLhsMat0102Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat0101Sp1)),
                _mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat0100Sp1));
            __m256i vIaccMat010Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp1, cLhsMat0103Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat236702Sp1, cLhsMat0102Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat0101Sp1)),
                _mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat0100Sp1));
            __m256i vIaccMat100Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp1, cLhsMat2303Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat014502Sp1, cLhsMat2302Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat2301Sp1)),
                _mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat2300Sp1));
            __m256i vIaccMat110Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp1, cLhsMat2303Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat236702Sp1, cLhsMat2302Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat2301Sp1)),
                _mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat2300Sp1));
            __m256i vIaccMat001Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp1, cLhsMat0113Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat014512Sp1, cLhsMat0112Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat0111Sp1)),
                _mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat0110Sp1));
            __m256i vIaccMat011Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp1, cLhsMat0113Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat236712Sp1, cLhsMat0112Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat0111Sp1)),
                _mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat0110Sp1));
            __m256i vIaccMat101Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp1, cLhsMat2313Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat014512Sp1, cLhsMat2312Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat2311Sp1)),
                _mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat2310Sp1));
            __m256i vIaccMat111Sp1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp1, cLhsMat2313Sp1),
                                                  _mm256_maddubs_epi16(cRhsMat236712Sp1, cLhsMat2312Sp1)),
                                 _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat2311Sp1)),
                _mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat2310Sp1));

            __m256i vIaccMat000Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp2, cLhsMat0103Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat014502Sp2, cLhsMat0102Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat0101Sp2)),
                _mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat0100Sp2));
            __m256i vIaccMat010Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp2, cLhsMat0103Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat236702Sp2, cLhsMat0102Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat0101Sp2)),
                _mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat0100Sp2));
            __m256i vIaccMat100Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp2, cLhsMat2303Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat014502Sp2, cLhsMat2302Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat2301Sp2)),
                _mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat2300Sp2));
            __m256i vIaccMat110Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp2, cLhsMat2303Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat236702Sp2, cLhsMat2302Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat2301Sp2)),
                _mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat2300Sp2));
            __m256i vIaccMat001Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp2, cLhsMat0113Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat014512Sp2, cLhsMat0112Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat0111Sp2)),
                _mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat0110Sp2));
            __m256i vIaccMat011Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp2, cLhsMat0113Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat236712Sp2, cLhsMat0112Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat0111Sp2)),
                _mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat0110Sp2));
            __m256i vIaccMat101Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp2, cLhsMat2313Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat014512Sp2, cLhsMat2312Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat2311Sp2)),
                _mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat2310Sp2));
            __m256i vIaccMat111Sp2 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp2, cLhsMat2313Sp2),
                                                  _mm256_maddubs_epi16(cRhsMat236712Sp2, cLhsMat2312Sp2)),
                                 _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat2311Sp2)),
                _mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat2310Sp2));

            // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
            __m256i vIaccMat000 = _mm256_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
            __m256i vIaccMat010 = _mm256_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
            __m256i vIaccMat100 = _mm256_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
            __m256i vIaccMat110 = _mm256_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

            __m256i vIaccMat001 = _mm256_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
            __m256i vIaccMat011 = _mm256_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
            __m256i vIaccMat101 = _mm256_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
            __m256i vIaccMat111 = _mm256_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

            // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
            vIaccMat000 = _mm256_madd_epi16(vIaccMat000, cScale01450);
            vIaccMat010 = _mm256_madd_epi16(vIaccMat010, cScale23670);
            vIaccMat100 = _mm256_madd_epi16(vIaccMat100, cScale01450);
            vIaccMat110 = _mm256_madd_epi16(vIaccMat110, cScale23670);

            vIaccMat001 = _mm256_madd_epi16(vIaccMat001, cScale01451);
            vIaccMat011 = _mm256_madd_epi16(vIaccMat011, cScale23671);
            vIaccMat101 = _mm256_madd_epi16(vIaccMat101, cScale01451);
            vIaccMat111 = _mm256_madd_epi16(vIaccMat111, cScale23671);

            // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next
            // step)
            __m256i vIaccRow00 = _mm256_blend_epi32(vIaccMat000, _mm256_shuffle_epi32(vIaccMat010, 78), 204);
            __m256i vIaccRow10 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat000, 78), vIaccMat010, 204);
            __m256i vIaccRow20 = _mm256_blend_epi32(vIaccMat100, _mm256_shuffle_epi32(vIaccMat110, 78), 204);
            __m256i vIaccRow30 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat100, 78), vIaccMat110, 204);
            __m256i vIaccRow01 = _mm256_blend_epi32(vIaccMat001, _mm256_shuffle_epi32(vIaccMat011, 78), 204);
            __m256i vIaccRow11 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat001, 78), vIaccMat011, 204);
            __m256i vIaccRow21 = _mm256_blend_epi32(vIaccMat101, _mm256_shuffle_epi32(vIaccMat111, 78), 204);
            __m256i vIaccRow31 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat101, 78), vIaccMat111, 204);

            __m256i vIaccRow0 = _mm256_add_epi32(vIaccRow00, vIaccRow01);
            __m256i vIaccRow1 = _mm256_add_epi32(vIaccRow10, vIaccRow11);
            __m256i vIaccRow2 = _mm256_add_epi32(vIaccRow20, vIaccRow21);
            __m256i vIaccRow3 = _mm256_add_epi32(vIaccRow30, vIaccRow31);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
            const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtrs[vRp][vB].d);
            const __m256 cRowScaleF32 = _mm256_set_m128(
                cRowScaleF32Sse, cRowScaleF32Sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

            // Multiply with appropriate scales and accumulate (for both d and dmin) below
            aAccRows[vRp * 4] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow0),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
            aAccRows[vRp * 4 + 1] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow1),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
            aAccRows[vRp * 4 + 2] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow2),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
            aAccRows[vRp * 4 + 3] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow3),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);

            __m256i vIaccRowMin0 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 0), cMins01);
            __m256i vIaccRowMin1 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 85), cMins01);
            __m256i vIaccRowMin2 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 170), cMins01);
            __m256i vIaccRowMin3 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 255), cMins01);

            aAccMinRows[vRp * 4] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRowMin0),
                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccMinRows[vRp * 4]);
            aAccMinRows[vRp * 4 + 1] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin1),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                aAccMinRows[vRp * 4 + 1]);
            aAccMinRows[vRp * 4 + 2] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin2),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                aAccMinRows[vRp * 4 + 2]);
            aAccMinRows[vRp * 4 + 3] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin3),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                aAccMinRows[vRp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm256_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }
  for (; vY < pNr / 4; vY++) {

    const block_q8_Kx4 *vAPtr = vAPtrStart + (vY * cNb);

    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {

      const block_q4_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      __m256 aAccMinRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccMinRows[vI] = _mm256_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {

        // Scale values - Load the eight scale values of block_q4_Kx8
        const __m256 cColScaleF32 = GGML_F32Cx8_LOAD(vBPtr[vB].d);

        // dmin values - Load the eight dmin values of block_q4_Kx8
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {

          // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and
          // storing of values
          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i cRhsMat014500 = _mm256_and_si256(cRhsRawMat01450, cM4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i cRhsMat236700 = _mm256_and_si256(cRhsRawMat23670, cM4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i cRhsMat014501 =
              _mm256_and_si256(cRhsRawMat01451, cM4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i cRhsMat236701 =
              _mm256_and_si256(cRhsRawMat23671, cM4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i cRhsMat014502 =
              _mm256_and_si256(cRhsRawMat01452, cM4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i cRhsMat236702 =
              _mm256_and_si256(cRhsRawMat23672, cM4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i cRhsMat014503 =
              _mm256_and_si256(cRhsRawMat01453, cM4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i cRhsMat236703 =
              _mm256_and_si256(cRhsRawMat23673, cM4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i cRhsMat014510 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4), cM4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i cRhsMat236710 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4), cM4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i cRhsMat014511 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4), cM4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i cRhsMat236711 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4), cM4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i cRhsMat014512 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 4),
                                                         cM4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i cRhsMat236712 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 4),
                                                         cM4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i cRhsMat014513 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 4),
                                                         cM4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i cRhsMat236713 = _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 4),
                                                         cM4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i cRhsMat014500Sp1 = _mm256_shuffle_epi32(
              cRhsMat014500, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i cRhsMat236700Sp1 = _mm256_shuffle_epi32(
              cRhsMat236700, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i cRhsMat014501Sp1 = _mm256_shuffle_epi32(
              cRhsMat014501, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i cRhsMat236701Sp1 = _mm256_shuffle_epi32(
              cRhsMat236701, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i cRhsMat014502Sp1 = _mm256_shuffle_epi32(
              cRhsMat014502,
              136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
          const __m256i cRhsMat236702Sp1 = _mm256_shuffle_epi32(
              cRhsMat236702,
              136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

          const __m256i cRhsMat014503Sp1 = _mm256_shuffle_epi32(
              cRhsMat014503,
              136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
          const __m256i cRhsMat236703Sp1 = _mm256_shuffle_epi32(
              cRhsMat236703,
              136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

          const __m256i cRhsMat014510Sp1 = _mm256_shuffle_epi32(
              cRhsMat014510, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i cRhsMat236710Sp1 = _mm256_shuffle_epi32(
              cRhsMat236710, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i cRhsMat014511Sp1 = _mm256_shuffle_epi32(
              cRhsMat014511, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i cRhsMat236711Sp1 = _mm256_shuffle_epi32(
              cRhsMat236711, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i cRhsMat014512Sp1 = _mm256_shuffle_epi32(
              cRhsMat014512,
              136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
          const __m256i cRhsMat236712Sp1 = _mm256_shuffle_epi32(
              cRhsMat236712,
              136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

          const __m256i cRhsMat014513Sp1 = _mm256_shuffle_epi32(
              cRhsMat014513,
              136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
          const __m256i cRhsMat236713Sp1 = _mm256_shuffle_epi32(
              cRhsMat236713,
              136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i cRhsMat014500Sp2 = _mm256_shuffle_epi32(
              cRhsMat014500, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i cRhsMat236700Sp2 = _mm256_shuffle_epi32(
              cRhsMat236700, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i cRhsMat014501Sp2 = _mm256_shuffle_epi32(
              cRhsMat014501,
              221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
          const __m256i cRhsMat236701Sp2 = _mm256_shuffle_epi32(
              cRhsMat236701,
              221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

          const __m256i cRhsMat014502Sp2 = _mm256_shuffle_epi32(
              cRhsMat014502,
              221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
          const __m256i cRhsMat236702Sp2 = _mm256_shuffle_epi32(
              cRhsMat236702,
              221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

          const __m256i cRhsMat014503Sp2 = _mm256_shuffle_epi32(
              cRhsMat014503,
              221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
          const __m256i cRhsMat236703Sp2 = _mm256_shuffle_epi32(
              cRhsMat236703,
              221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

          const __m256i cRhsMat014510Sp2 = _mm256_shuffle_epi32(
              cRhsMat014510, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i cRhsMat236710Sp2 = _mm256_shuffle_epi32(
              cRhsMat236710, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i cRhsMat014511Sp2 = _mm256_shuffle_epi32(
              cRhsMat014511,
              221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
          const __m256i cRhsMat236711Sp2 = _mm256_shuffle_epi32(
              cRhsMat236711,
              221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

          const __m256i cRhsMat014512Sp2 = _mm256_shuffle_epi32(
              cRhsMat014512,
              221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
          const __m256i cRhsMat236712Sp2 = _mm256_shuffle_epi32(
              cRhsMat236712,
              221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

          const __m256i cRhsMat014513Sp2 = _mm256_shuffle_epi32(
              cRhsMat014513,
              221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
          const __m256i cRhsMat236713Sp2 = _mm256_shuffle_epi32(
              cRhsMat236713,
              221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

          uint32_t aUtmp0[4], aUtmp1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
          // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for
          // the sb loop
          memcpy(aUtmp0, vBPtr[vB].scales + 24 * vSb, 12);
          aUtmp0[3] = ((aUtmp0[2] >> 4) & cKmask2) | (((aUtmp0[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux0 = aUtmp0[1] & cKmask1;
          aUtmp0[1] = (aUtmp0[2] & cKmask2) | (((aUtmp0[0] >> 6) & cKmask3) << 4);
          aUtmp0[2] = cUaux0;
          aUtmp0[0] &= cKmask1;

          // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when
          // sb = 1
          memcpy(aUtmp1, vBPtr[vB].scales + 12 + vSb * 24, 12);
          aUtmp1[3] = ((aUtmp1[2] >> 4) & cKmask2) | (((aUtmp1[1] >> 6) & cKmask3) << 4);
          const uint32_t cUaux1 = aUtmp1[1] & cKmask1;
          aUtmp1[1] = (aUtmp1[2] & cKmask2) | (((aUtmp1[0] >> 6) & cKmask3) << 4);
          aUtmp1[2] = cUaux1;
          aUtmp1[0] &= cKmask1;

          // Scales of first sub block in the sb loop
          const __m128i cMinsAndScales0 = _mm_set_epi32(aUtmp0[3], aUtmp0[2], aUtmp0[1], aUtmp0[0]);
          const __m256i cScales0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(cMinsAndScales0, cMinsAndScales0));

          // Scales of second sub block in the sb loop
          const __m128i cMinsAndScales1 = _mm_set_epi32(aUtmp1[3], aUtmp1[2], aUtmp1[1], aUtmp1[0]);
          const __m256i cScales1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(cMinsAndScales1, cMinsAndScales1));

          // Mins of first and second sub block of Q4_K block are arranged side by side
          const __m256i cMins01 = _mm256_cvtepu8_epi16(
              _mm_unpacklo_epi8(_mm_shuffle_epi32(cMinsAndScales0, 78), _mm_shuffle_epi32(cMinsAndScales1, 78)));

          const __m256i cScale01450 = _mm256_shuffle_epi32(cScales0, 68);
          const __m256i cScale23670 = _mm256_shuffle_epi32(cScales0, 238);

          const __m256i cScale01451 = _mm256_shuffle_epi32(cScales1, 68);
          const __m256i cScale23671 = _mm256_shuffle_epi32(cScales1, 238);

          // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
          // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
          __m256i vLhsMat012300 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 256 * vSb)));
          __m256i vLhsMat0100 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 0);
          __m256i vLhsMat2300 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 17);
          __m256i vLhsMat012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32 + 256 * vSb)));
          __m256i vLhsMat0101 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 0);
          __m256i vLhsMat2301 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 17);
          __m256i vLhsMat012302 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64 + 256 * vSb)));
          __m256i vLhsMat0102 = _mm256_permute2f128_si256(vLhsMat012302, vLhsMat012302, 0);
          __m256i vLhsMat2302 = _mm256_permute2f128_si256(vLhsMat012302, vLhsMat012302, 17);
          __m256i vLhsMat012303 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96 + 256 * vSb)));
          __m256i vLhsMat0103 = _mm256_permute2f128_si256(vLhsMat012303, vLhsMat012303, 0);
          __m256i vLhsMat2303 = _mm256_permute2f128_si256(vLhsMat012303, vLhsMat012303, 17);
          __m256i vLhsMat012310 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 128 + 256 * vSb)));
          __m256i vLhsMat0110 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 0);
          __m256i vLhsMat2310 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 17);
          __m256i vLhsMat012311 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 160 + 256 * vSb)));
          __m256i vLhsMat0111 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 0);
          __m256i vLhsMat2311 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 17);
          __m256i vLhsMat012312 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 192 + 256 * vSb)));
          __m256i vLhsMat0112 = _mm256_permute2f128_si256(vLhsMat012312, vLhsMat012312, 0);
          __m256i vLhsMat2312 = _mm256_permute2f128_si256(vLhsMat012312, vLhsMat012312, 17);
          __m256i vLhsMat012313 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 224 + 256 * vSb)));
          __m256i vLhsMat0113 = _mm256_permute2f128_si256(vLhsMat012313, vLhsMat012313, 0);
          __m256i vLhsMat2313 = _mm256_permute2f128_si256(vLhsMat012313, vLhsMat012313, 17);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
          __m256i vLhsBsums012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].bsums + 16 * vSb)));
          __m256i vLhsBsumsHsum012301 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(vLhsBsums012301), _mm256_extractf128_si256(vLhsBsums012301, 1)));
          vLhsBsumsHsum012301 = _mm256_permute2x128_si256(vLhsBsumsHsum012301, vLhsBsumsHsum012301, 0);

          // Shuffle pattern one - left side input
          const __m256i cLhsMat0100Sp1 = _mm256_shuffle_epi32(
              vLhsMat0100, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
          const __m256i cLhsMat2300Sp1 = _mm256_shuffle_epi32(
              vLhsMat2300, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

          const __m256i cLhsMat0101Sp1 = _mm256_shuffle_epi32(
              vLhsMat0101, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m256i cLhsMat2301Sp1 = _mm256_shuffle_epi32(
              vLhsMat2301, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

          const __m256i cLhsMat0102Sp1 = _mm256_shuffle_epi32(
              vLhsMat0102,
              160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m256i cLhsMat2302Sp1 = _mm256_shuffle_epi32(
              vLhsMat2302,
              160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

          const __m256i cLhsMat0103Sp1 = _mm256_shuffle_epi32(
              vLhsMat0103,
              160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m256i cLhsMat2303Sp1 = _mm256_shuffle_epi32(
              vLhsMat2303,
              160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

          const __m256i cLhsMat0110Sp1 = _mm256_shuffle_epi32(
              vLhsMat0110, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
          const __m256i cLhsMat2310Sp1 = _mm256_shuffle_epi32(
              vLhsMat2310, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

          const __m256i cLhsMat0111Sp1 = _mm256_shuffle_epi32(
              vLhsMat0111, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m256i cLhsMat2311Sp1 = _mm256_shuffle_epi32(
              vLhsMat2311, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

          const __m256i cLhsMat0112Sp1 = _mm256_shuffle_epi32(
              vLhsMat0112,
              160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m256i cLhsMat2312Sp1 = _mm256_shuffle_epi32(
              vLhsMat2312,
              160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

          const __m256i cLhsMat0113Sp1 = _mm256_shuffle_epi32(
              vLhsMat0113,
              160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m256i cLhsMat2313Sp1 = _mm256_shuffle_epi32(
              vLhsMat2313,
              160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

          // Shuffle pattern two- left side input
          const __m256i cLhsMat0100Sp2 = _mm256_shuffle_epi32(
              vLhsMat0100, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
          const __m256i cLhsMat2300Sp2 = _mm256_shuffle_epi32(
              vLhsMat2300, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

          const __m256i cLhsMat0101Sp2 = _mm256_shuffle_epi32(
              vLhsMat0101,
              245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m256i cLhsMat2301Sp2 = _mm256_shuffle_epi32(
              vLhsMat2301,
              245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

          const __m256i cLhsMat0102Sp2 = _mm256_shuffle_epi32(
              vLhsMat0102,
              245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m256i cLhsMat2302Sp2 = _mm256_shuffle_epi32(
              vLhsMat2302,
              245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

          const __m256i cLhsMat0103Sp2 = _mm256_shuffle_epi32(
              vLhsMat0103,
              245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m256i cLhsMat2303Sp2 = _mm256_shuffle_epi32(
              vLhsMat2303,
              245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

          const __m256i cLhsMat0110Sp2 = _mm256_shuffle_epi32(
              vLhsMat0110, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
          const __m256i cLhsMat2310Sp2 = _mm256_shuffle_epi32(
              vLhsMat2310, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

          const __m256i cLhsMat0111Sp2 = _mm256_shuffle_epi32(
              vLhsMat0111,
              245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m256i cLhsMat2311Sp2 = _mm256_shuffle_epi32(
              vLhsMat2311,
              245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

          const __m256i cLhsMat0112Sp2 = _mm256_shuffle_epi32(
              vLhsMat0112,
              245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m256i cLhsMat2312Sp2 = _mm256_shuffle_epi32(
              vLhsMat2312,
              245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

          const __m256i cLhsMat0113Sp2 = _mm256_shuffle_epi32(
              vLhsMat0113,
              245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m256i cLhsMat2313Sp2 = _mm256_shuffle_epi32(
              vLhsMat2313,
              245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
          __m256i vIaccMat000Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp1, cLhsMat0103Sp1),
                                                _mm256_maddubs_epi16(cRhsMat014502Sp1, cLhsMat0102Sp1)),
                               _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat0101Sp1)),
              _mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat0100Sp1));
          __m256i vIaccMat010Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp1, cLhsMat0103Sp1),
                                                _mm256_maddubs_epi16(cRhsMat236702Sp1, cLhsMat0102Sp1)),
                               _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat0101Sp1)),
              _mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat0100Sp1));
          __m256i vIaccMat100Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp1, cLhsMat2303Sp1),
                                                _mm256_maddubs_epi16(cRhsMat014502Sp1, cLhsMat2302Sp1)),
                               _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat2301Sp1)),
              _mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat2300Sp1));
          __m256i vIaccMat110Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp1, cLhsMat2303Sp1),
                                                _mm256_maddubs_epi16(cRhsMat236702Sp1, cLhsMat2302Sp1)),
                               _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat2301Sp1)),
              _mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat2300Sp1));
          __m256i vIaccMat001Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp1, cLhsMat0113Sp1),
                                                _mm256_maddubs_epi16(cRhsMat014512Sp1, cLhsMat0112Sp1)),
                               _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat0111Sp1)),
              _mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat0110Sp1));
          __m256i vIaccMat011Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp1, cLhsMat0113Sp1),
                                                _mm256_maddubs_epi16(cRhsMat236712Sp1, cLhsMat0112Sp1)),
                               _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat0111Sp1)),
              _mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat0110Sp1));
          __m256i vIaccMat101Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp1, cLhsMat2313Sp1),
                                                _mm256_maddubs_epi16(cRhsMat014512Sp1, cLhsMat2312Sp1)),
                               _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat2311Sp1)),
              _mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat2310Sp1));
          __m256i vIaccMat111Sp1 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp1, cLhsMat2313Sp1),
                                                _mm256_maddubs_epi16(cRhsMat236712Sp1, cLhsMat2312Sp1)),
                               _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat2311Sp1)),
              _mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat2310Sp1));

          __m256i vIaccMat000Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp2, cLhsMat0103Sp2),
                                                _mm256_maddubs_epi16(cRhsMat014502Sp2, cLhsMat0102Sp2)),
                               _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat0101Sp2)),
              _mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat0100Sp2));
          __m256i vIaccMat010Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp2, cLhsMat0103Sp2),
                                                _mm256_maddubs_epi16(cRhsMat236702Sp2, cLhsMat0102Sp2)),
                               _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat0101Sp2)),
              _mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat0100Sp2));
          __m256i vIaccMat100Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014503Sp2, cLhsMat2303Sp2),
                                                _mm256_maddubs_epi16(cRhsMat014502Sp2, cLhsMat2302Sp2)),
                               _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat2301Sp2)),
              _mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat2300Sp2));
          __m256i vIaccMat110Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236703Sp2, cLhsMat2303Sp2),
                                                _mm256_maddubs_epi16(cRhsMat236702Sp2, cLhsMat2302Sp2)),
                               _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat2301Sp2)),
              _mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat2300Sp2));
          __m256i vIaccMat001Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp2, cLhsMat0113Sp2),
                                                _mm256_maddubs_epi16(cRhsMat014512Sp2, cLhsMat0112Sp2)),
                               _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat0111Sp2)),
              _mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat0110Sp2));
          __m256i vIaccMat011Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp2, cLhsMat0113Sp2),
                                                _mm256_maddubs_epi16(cRhsMat236712Sp2, cLhsMat0112Sp2)),
                               _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat0111Sp2)),
              _mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat0110Sp2));
          __m256i vIaccMat101Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014513Sp2, cLhsMat2313Sp2),
                                                _mm256_maddubs_epi16(cRhsMat014512Sp2, cLhsMat2312Sp2)),
                               _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat2311Sp2)),
              _mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat2310Sp2));
          __m256i vIaccMat111Sp2 = _mm256_add_epi16(
              _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236713Sp2, cLhsMat2313Sp2),
                                                _mm256_maddubs_epi16(cRhsMat236712Sp2, cLhsMat2312Sp2)),
                               _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat2311Sp2)),
              _mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat2310Sp2));

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          __m256i vIaccMat000 = _mm256_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
          __m256i vIaccMat010 = _mm256_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
          __m256i vIaccMat100 = _mm256_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
          __m256i vIaccMat110 = _mm256_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

          __m256i vIaccMat001 = _mm256_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
          __m256i vIaccMat011 = _mm256_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
          __m256i vIaccMat101 = _mm256_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
          __m256i vIaccMat111 = _mm256_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          vIaccMat000 = _mm256_madd_epi16(vIaccMat000, cScale01450);
          vIaccMat010 = _mm256_madd_epi16(vIaccMat010, cScale23670);
          vIaccMat100 = _mm256_madd_epi16(vIaccMat100, cScale01450);
          vIaccMat110 = _mm256_madd_epi16(vIaccMat110, cScale23670);

          vIaccMat001 = _mm256_madd_epi16(vIaccMat001, cScale01451);
          vIaccMat011 = _mm256_madd_epi16(vIaccMat011, cScale23671);
          vIaccMat101 = _mm256_madd_epi16(vIaccMat101, cScale01451);
          vIaccMat111 = _mm256_madd_epi16(vIaccMat111, cScale23671);

          // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
          __m256i vIaccRow00 = _mm256_blend_epi32(vIaccMat000, _mm256_shuffle_epi32(vIaccMat010, 78), 204);
          __m256i vIaccRow10 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat000, 78), vIaccMat010, 204);
          __m256i vIaccRow20 = _mm256_blend_epi32(vIaccMat100, _mm256_shuffle_epi32(vIaccMat110, 78), 204);
          __m256i vIaccRow30 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat100, 78), vIaccMat110, 204);
          __m256i vIaccRow01 = _mm256_blend_epi32(vIaccMat001, _mm256_shuffle_epi32(vIaccMat011, 78), 204);
          __m256i vIaccRow11 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat001, 78), vIaccMat011, 204);
          __m256i vIaccRow21 = _mm256_blend_epi32(vIaccMat101, _mm256_shuffle_epi32(vIaccMat111, 78), 204);
          __m256i vIaccRow31 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat101, 78), vIaccMat111, 204);

          __m256i vIaccRow0 = _mm256_add_epi32(vIaccRow00, vIaccRow01);
          __m256i vIaccRow1 = _mm256_add_epi32(vIaccRow10, vIaccRow11);
          __m256i vIaccRow2 = _mm256_add_epi32(vIaccRow20, vIaccRow21);
          __m256i vIaccRow3 = _mm256_add_epi32(vIaccRow30, vIaccRow31);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
          const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtr[vB].d);
          const __m256 cRowScaleF32 =
              _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

          // Multiply with appropriate scales and accumulate (for both d and dmin) below
          aAccRows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow0),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                        aAccRows[0]);
          aAccRows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow1),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                        aAccRows[1]);
          aAccRows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow2),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                        aAccRows[2]);
          aAccRows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow3),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                        aAccRows[3]);

          __m256i vIaccRowMin0 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 0), cMins01);
          __m256i vIaccRowMin1 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 85), cMins01);
          __m256i vIaccRowMin2 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 170), cMins01);
          __m256i vIaccRowMin3 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsumsHsum012301, 255), cMins01);

          aAccMinRows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin0),
                                           _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                           aAccMinRows[0]);
          aAccMinRows[1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin1),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccMinRows[1]);
          aAccMinRows[2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin2),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccMinRows[2]);
          aAccMinRows[3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin3),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccMinRows[3]);
        }
      }

      // Store the accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm256_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }

#else
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  ggml_gemm_q4_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
#endif
}

void ggml_gemm_iq4_nl_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                               const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__) || defined(__AVX512F__)
  {
    __m256i vSignextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
    vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

    fGemmQ4B328x8Q80LutAvx<block_iq4_nlx8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

    return;
  }
#endif // defined(__AVX2__) || defined(__AVX512F__)

  ggml_gemm_iq4_nl_4x4_q8_0(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_mxfp4_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
#if defined(__AVX2__) || defined(__AVX512F__)
  {
    __m256i vSignextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_mxfp4));
    vSignextendlut = _mm256_permute2f128_si256(vSignextendlut, vSignextendlut, 0);

    fGemmQ4B328x8Q80LutAvx<block_mxfp4x8>(pN, pS, pBs, pVx, pVy, pNr, pNc, vSignextendlut);

    return;
  }
#endif // defined(__AVX2__) || defined(__AVX512F__)

  ggml_gemm_mxfp4_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q2_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

#if defined(__AVX2__) || defined(__AVX512F__)
  const block_q2_Kx8 *vBPtrStart = (const block_q2_Kx8 *)pVx;
  const block_q8_Kx4 *vAPtrStart = (const block_q8_Kx4 *)pVy;
  int64_t vBNb = pN / QK_K;
  int64_t vY = 0;

  // Permute mask used for easier vector processing at later stages
  __m256i vRequiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  int64_t vXstart = 0;
  int vAnr = pNr - pNr % 16; // Used to align nr with boundary of 16

  // Mask to convert 2 bit and 4 bit values into a bytes
  const __m256i cM3b = _mm256_set1_epi8(3);
  const __m128i cM4bSse = _mm_set1_epi8(0xF);

  // Mask to get appropriate scales
  __m128i vScalesmask1Sse = _mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  __m128i vScalesmask2Sse = _mm_set_epi8(15, 15, 13, 13, 11, 11, 9, 9, 7, 7, 5, 5, 3, 3, 1, 1);

  __m256i vScalesmask1 = _mm256_castsi128_si256(vScalesmask1Sse);
  vScalesmask1 = _mm256_permute2f128_si256(vScalesmask1, vScalesmask1, 0);
  __m256i vScalesmask2 = _mm256_castsi128_si256(vScalesmask2Sse);
  vScalesmask2 = _mm256_permute2f128_si256(vScalesmask2, vScalesmask2, 0);

#if defined(__AVX512BW__) && defined(__AVX512DQ__)

  int vAnc = pNc - pNc % 16; // Used to align nc with boundary of 16

  // Mask to mask out nibbles from packed bytes
  const __m256i cM4b = _mm256_set1_epi8(0x0F);
  // Mask to mask out nibbles from packed bytes expanded to 512 bit length
  const __m512i cM3bexpanded = _mm512_set1_epi8(3);
  // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
  for (; vY < vAnr / 4; vY += 4) {

    const block_q8_Kx4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_q2_Kx8 *vBPtr0 = vBPtrStart + ((vX)*vBNb);
      const block_q2_Kx8 *vBPtr1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      __m512 aAccMinRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccMinRows[vI] = _mm512_setzero_ps();
      }
      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {
        // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
        const __m512 cColScaleF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].d, vBPtr1[vB].d);

        // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
        const __m512 cColDminF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].dmin, vBPtr1[vB].dmin);

        // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 128; vSb++) {

          // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + vSb * 256));
          const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat89Ab2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMatCdef2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat89Ab3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMatCdef3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          const __m256i cRhsRawMat89Cd0 =
              _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef0 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
          const __m256i cRhsRawMat89Cd1 =
              _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef1 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);
          const __m256i cRhsRawMat89Cd2 =
              _mm256_blend_epi32(cRhsRawMat89Ab2, _mm256_permutevar8x32_epi32(cRhsRawMatCdef2, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef2 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab2, vRequiredOrder), cRhsRawMatCdef2, 240);
          const __m256i cRhsRawMat89Cd3 =
              _mm256_blend_epi32(cRhsRawMat89Ab3, _mm256_permutevar8x32_epi32(cRhsRawMatCdef3, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef3 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab3, vRequiredOrder), cRhsRawMatCdef3, 240);

          const __m512i cRhsRawMat014589Cd0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
          const __m512i cRhsRawMat2367Abef0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
          const __m512i cRhsRawMat014589Cd1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
          const __m512i cRhsRawMat2367Abef1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

          const __m512i cRhsRawMat014589Cd2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01452), cRhsRawMat89Cd2, 1);
          const __m512i cRhsRawMat2367Abef2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23672), cRhsRawMatAbef2, 1);
          const __m512i cRhsRawMat014589Cd3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01453), cRhsRawMat89Cd3, 1);
          const __m512i cRhsRawMat2367Abef3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23673), cRhsRawMatAbef3, 1);

          // 2-bit -> 8-bit
          const __m512i cRhsMat014589Cd00 =
              _mm512_and_si512(cRhsRawMat014589Cd0,
                               cM3bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i cRhsMat2367Abef00 =
              _mm512_and_si512(cRhsRawMat2367Abef0,
                               cM3bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i cRhsMat014589Cd01 = _mm512_and_si512(
              cRhsRawMat014589Cd1,
              cM3bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i cRhsMat2367Abef01 = _mm512_and_si512(
              cRhsRawMat2367Abef1,
              cM3bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
          const __m512i cRhsMat014589Cd10 =
              _mm512_and_si512(cRhsRawMat014589Cd2,
                               cM3bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i cRhsMat2367Abef10 =
              _mm512_and_si512(cRhsRawMat2367Abef2,
                               cM3bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i cRhsMat014589Cd11 = _mm512_and_si512(
              cRhsRawMat014589Cd3,
              cM3bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i cRhsMat2367Abef11 = _mm512_and_si512(
              cRhsRawMat2367Abef3,
              cM3bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i cRhsMat014589Cd20 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 2),
                               cM3bexpanded); // B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
          const __m512i cRhsMat2367Abef20 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 2),
                               cM3bexpanded); // B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

          const __m512i cRhsMat014589Cd21 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 2),
              cM3bexpanded); // B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
          const __m512i cRhsMat2367Abef21 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 2),
              cM3bexpanded); // B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

          const __m512i cRhsMat014589Cd30 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 2),
                               cM3bexpanded); // B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
          const __m512i cRhsMat2367Abef30 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 2),
                               cM3bexpanded); // B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

          const __m512i cRhsMat014589Cd31 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 2),
              cM3bexpanded); // B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
          const __m512i cRhsMat2367Abef31 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 2),
              cM3bexpanded); // B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

          const __m512i cRhsMat014589Cd40 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                               cM3bexpanded); // B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
          const __m512i cRhsMat2367Abef40 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                               cM3bexpanded); // B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

          const __m512i cRhsMat014589Cd41 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
              cM3bexpanded); // B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
          const __m512i cRhsMat2367Abef41 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
              cM3bexpanded); // B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

          const __m512i cRhsMat014589Cd50 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 4),
                               cM3bexpanded); // B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
          const __m512i cRhsMat2367Abef50 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 4),
                               cM3bexpanded); // B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

          const __m512i cRhsMat014589Cd51 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 4),
              cM3bexpanded); // B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
          const __m512i cRhsMat2367Abef51 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 4),
              cM3bexpanded); // B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

          const __m512i cRhsMat014589Cd60 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 6),
                               cM3bexpanded); // B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
          const __m512i cRhsMat2367Abef60 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 6),
                               cM3bexpanded); // B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

          const __m512i cRhsMat014589Cd61 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 6),
              cM3bexpanded); // B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
          const __m512i cRhsMat2367Abef61 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 6),
              cM3bexpanded); // B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

          const __m512i cRhsMat014589Cd70 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 6),
                               cM3bexpanded); // B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
          const __m512i cRhsMat2367Abef70 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 6),
                               cM3bexpanded); // B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

          const __m512i cRhsMat014589Cd71 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 6),
              cM3bexpanded); // B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
          const __m512i cRhsMat2367Abef71 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 6),
              cM3bexpanded); // B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

          const __m512i cRhsMat014589Cd00Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3)
                                   // B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
          const __m512i cRhsMat2367Abef00Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3)
                                   // B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

          const __m512i cRhsMat014589Cd01Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef01Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                   // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

          const __m512i cRhsMat014589Cd10Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3)
                                   // B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
          const __m512i cRhsMat2367Abef10Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3)
                                   // B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

          const __m512i cRhsMat014589Cd11Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                   // B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i cRhsMat2367Abef11Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                   // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

          const __m512i cRhsMat014589Cd20Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd20,
              (_MM_PERM_ENUM)136); // B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3)
                                   // B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
          const __m512i cRhsMat2367Abef20Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef20,
              (_MM_PERM_ENUM)136); // B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3)
                                   // B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

          const __m512i cRhsMat014589Cd21Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd21,
              (_MM_PERM_ENUM)136); // B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                                   // B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
          const __m512i cRhsMat2367Abef21Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef21,
              (_MM_PERM_ENUM)136); // B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)
                                   // B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)

          const __m512i cRhsMat014589Cd30Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd30,
              (_MM_PERM_ENUM)136); /// B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3)
                                   /// B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
          const __m512i cRhsMat2367Abef30Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef30,
              (_MM_PERM_ENUM)136); // B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3)
                                   // B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

          const __m512i cRhsMat014589Cd31Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd31,
              (_MM_PERM_ENUM)136); // B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11)
                                   // B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
          const __m512i cRhsMat2367Abef31Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef31,
              (_MM_PERM_ENUM)136); // B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)
                                   // B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

          const __m512i cRhsMat014589Cd40Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd40,
              (_MM_PERM_ENUM)136); // B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3)
                                   // B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
          const __m512i cRhsMat2367Abef40Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef40,
              (_MM_PERM_ENUM)136); // B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3)
                                   // B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

          const __m512i cRhsMat014589Cd41Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd41,
              (_MM_PERM_ENUM)136); // B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                                   // B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
          const __m512i cRhsMat2367Abef41Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef41,
              (_MM_PERM_ENUM)136); // B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)
                                   // B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

          const __m512i cRhsMat014589Cd50Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd50,
              (_MM_PERM_ENUM)136); // B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3)
                                   // B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
          const __m512i cRhsMat2367Abef50Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef50,
              (_MM_PERM_ENUM)136); // B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3)
                                   // B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

          const __m512i cRhsMat014589Cd51Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd51,
              (_MM_PERM_ENUM)136); // B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                                   // B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
          const __m512i cRhsMat2367Abef51Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef51,
              (_MM_PERM_ENUM)136); // B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)
                                   // B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

          const __m512i cRhsMat014589Cd60Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd60,
              (_MM_PERM_ENUM)136); // B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3)
                                   // B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
          const __m512i cRhsMat2367Abef60Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef60,
              (_MM_PERM_ENUM)136); // B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3)
                                   // B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

          const __m512i cRhsMat014589Cd61Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd61,
              (_MM_PERM_ENUM)136); // B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                                   // B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
          const __m512i cRhsMat2367Abef61Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef61,
              (_MM_PERM_ENUM)136); // B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)
                                   // B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

          const __m512i cRhsMat014589Cd70Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd70,
              (_MM_PERM_ENUM)136); // B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3)
                                   // B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
          const __m512i cRhsMat2367Abef70Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef70,
              (_MM_PERM_ENUM)136); // B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3)
                                   // B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

          const __m512i cRhsMat014589Cd71Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd71,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef71Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef71,
              (_MM_PERM_ENUM)136); // B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)
                                   // B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

          const __m512i cRhsMat014589Cd00Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7)
                                   // B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
          const __m512i cRhsMat2367Abef00Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7)
                                   // B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

          const __m512i cRhsMat014589Cd01Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15)
                                   // B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15)
                                   // B0C(12-15) B0D(12-15)
          const __m512i cRhsMat2367Abef01Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15)
                                   // B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15)
                                   // B0E(12-15) B0F(12-15)

          const __m512i cRhsMat014589Cd10Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7)
                                   // B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
          const __m512i cRhsMat2367Abef10Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7)
                                   // B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

          const __m512i cRhsMat014589Cd11Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15)
                                   // B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15)
                                   // B1C(12-15) B1D(12-15)
          const __m512i cRhsMat2367Abef11Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15)
                                   // B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15)
                                   // B1E(12-15) B1F(12-15)

          const __m512i cRhsMat014589Cd20Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd20,
              (_MM_PERM_ENUM)221); // B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7)
                                   // B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
          const __m512i cRhsMat2367Abef20Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef20,
              (_MM_PERM_ENUM)221); // B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7)
                                   // B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

          const __m512i cRhsMat014589Cd21Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd21,
              (_MM_PERM_ENUM)221); // B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15)
                                   // B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15)
                                   // B2C(12-15) B2D(12-15)
          const __m512i cRhsMat2367Abef21Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef21,
              (_MM_PERM_ENUM)221); // B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15)
                                   // B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15)
                                   // B2E(12-15) B2F(12-15)

          const __m512i cRhsMat014589Cd30Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd30,
              (_MM_PERM_ENUM)221); // B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7)
                                   // B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
          const __m512i cRhsMat2367Abef30Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef30,
              (_MM_PERM_ENUM)221); // B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7)
                                   // B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

          const __m512i cRhsMat014589Cd31Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd31,
              (_MM_PERM_ENUM)221); // B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15)
                                   // B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15)
                                   // B3C(12-15) B3D(12-15)
          const __m512i cRhsMat2367Abef31Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef31,
              (_MM_PERM_ENUM)221); // B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15)
                                   // B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15)
                                   // B3E(12-15) B3F(12-15)

          const __m512i cRhsMat014589Cd40Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd40,
              (_MM_PERM_ENUM)221); // B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7)
                                   // B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
          const __m512i cRhsMat2367Abef40Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef40,
              (_MM_PERM_ENUM)221); // B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7)
                                   // B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

          const __m512i cRhsMat014589Cd41Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd41,
              (_MM_PERM_ENUM)221); // B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15)
                                   // B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15)
                                   // B4C(12-15) B4D(12-15)
          const __m512i cRhsMat2367Abef41Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef41,
              (_MM_PERM_ENUM)221); // B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15)
                                   // B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15)
                                   // B4E(12-15) B4F(12-15)

          const __m512i cRhsMat014589Cd50Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd50,
              (_MM_PERM_ENUM)221); // B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7)
                                   // B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
          const __m512i cRhsMat2367Abef50Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef50,
              (_MM_PERM_ENUM)221); // B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7)
                                   // B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

          const __m512i cRhsMat014589Cd51Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd51,
              (_MM_PERM_ENUM)221); // B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15)
                                   // B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15)
                                   // B5C(12-15) B5D(12-15)
          const __m512i cRhsMat2367Abef51Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef51,
              (_MM_PERM_ENUM)221); // B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15)
                                   // B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15)
                                   // B5E(12-15) B5F(12-15)

          const __m512i cRhsMat014589Cd60Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd60,
              (_MM_PERM_ENUM)221); // B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7)
                                   // B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
          const __m512i cRhsMat2367Abef60Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef60,
              (_MM_PERM_ENUM)221); // B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7)
                                   // B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

          const __m512i cRhsMat014589Cd61Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd61,
              (_MM_PERM_ENUM)221); // B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15)
                                   // B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15)
                                   // B6C(12-15) B6D(12-15)
          const __m512i cRhsMat2367Abef61Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef61,
              (_MM_PERM_ENUM)221); // B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15)
                                   // B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15)
                                   // B6E(12-15) B6F(12-15)

          const __m512i cRhsMat014589Cd70Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd70,
              (_MM_PERM_ENUM)221); // B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7)
                                   // B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
          const __m512i cRhsMat2367Abef70Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef70,
              (_MM_PERM_ENUM)221); // B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7)
                                   // B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

          const __m512i cRhsMat014589Cd71Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd71,
              (_MM_PERM_ENUM)221); // B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15)
                                   // B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15)
                                   // B7C(12-15) B7D(12-15)
          const __m512i cRhsMat2367Abef71Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef71,
              (_MM_PERM_ENUM)221); // B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15)
                                   // B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15)
                                   // B7E(12-15) B7F(12-15)

          // notation:superblock subblock
          // s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51
          // m51  s60 m60  s61 m61   s70 m70  s71 m71

          const __m128i cMinsAndScales010 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + vSb * 64));
          const __m128i cMinsAndScales230 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales450 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales670 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 48 + vSb * 64));

          const __m128i cMinsAndScales011 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + vSb * 64));
          const __m128i cMinsAndScales231 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales451 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales671 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 48 + vSb * 64));

          // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
          const __m256i cMinsAndScales01 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales010), cMinsAndScales011, 1);
          const __m256i cMinsAndScales23 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales230), cMinsAndScales231, 1);
          const __m256i cMinsAndScales45 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales450), cMinsAndScales451, 1);
          const __m256i cMinsAndScales67 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales670), cMinsAndScales671, 1);

          // Extract scales which is lower half from mins_and_scales
          const __m256i cScales01 = _mm256_and_si256(cMinsAndScales01, cM4b);
          const __m256i cScales23 = _mm256_and_si256(cMinsAndScales23, cM4b);
          const __m256i cScales45 = _mm256_and_si256(cMinsAndScales45, cM4b);
          const __m256i cScales67 = _mm256_and_si256(cMinsAndScales67, cM4b);

          // Extract mins which is upper half from mins_and_scales
          const __m512i cMins01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales01, 4), cM4b));
          const __m512i cMins23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales23, 4), cM4b));
          const __m512i cMins45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales45, 4), cM4b));
          const __m512i cMins67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales67, 4), cM4b));

          const __m512i cScales0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales01, vScalesmask1));
          const __m512i cScales1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales01, vScalesmask2));
          const __m512i cScales2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales23, vScalesmask1));
          const __m512i cScales3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales23, vScalesmask2));
          const __m512i cScales4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales45, vScalesmask1));
          const __m512i cScales5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales45, vScalesmask2));
          const __m512i cScales6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales67, vScalesmask1));
          const __m512i cScales7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales67, vScalesmask2));

          const __m512i cScale014589Cd0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd2 = _mm512_shuffle_epi32(cScales2, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef2 = _mm512_shuffle_epi32(cScales2, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd3 = _mm512_shuffle_epi32(cScales3, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef3 = _mm512_shuffle_epi32(cScales3, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd4 = _mm512_shuffle_epi32(cScales4, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef4 = _mm512_shuffle_epi32(cScales4, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd5 = _mm512_shuffle_epi32(cScales5, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef5 = _mm512_shuffle_epi32(cScales5, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd6 = _mm512_shuffle_epi32(cScales6, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef6 = _mm512_shuffle_epi32(cScales6, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd7 = _mm512_shuffle_epi32(cScales7, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef7 = _mm512_shuffle_epi32(cScales7, (_MM_PERM_ENUM)238);

          for (int vRp = 0; vRp < 4; vRp++) {

            // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
            // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again
            // repeating into 512 bit vector
            __m256i vLhsMatYmm012300 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 512 * vSb)));
            __m256i vLhsMatYmm0100 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 0);
            __m256i vLhsMatYmm2300 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 17);
            __m256i vLhsMatYmm012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32 + 512 * vSb)));
            __m256i vLhsMatYmm0101 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 0);
            __m256i vLhsMatYmm2301 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 17);
            __m256i vLhsMatYmm012310 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64 + 512 * vSb)));
            __m256i vLhsMatYmm0110 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 0);
            __m256i vLhsMatYmm2310 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 17);
            __m256i vLhsMatYmm012311 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96 + 512 * vSb)));
            __m256i vLhsMatYmm0111 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 0);
            __m256i vLhsMatYmm2311 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 17);
            __m256i vLhsMatYmm012320 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 128 + 512 * vSb)));
            __m256i vLhsMatYmm0120 = _mm256_permute2f128_si256(vLhsMatYmm012320, vLhsMatYmm012320, 0);
            __m256i vLhsMatYmm2320 = _mm256_permute2f128_si256(vLhsMatYmm012320, vLhsMatYmm012320, 17);
            __m256i vLhsMatYmm012321 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 160 + 512 * vSb)));
            __m256i vLhsMatYmm0121 = _mm256_permute2f128_si256(vLhsMatYmm012321, vLhsMatYmm012321, 0);
            __m256i vLhsMatYmm2321 = _mm256_permute2f128_si256(vLhsMatYmm012321, vLhsMatYmm012321, 17);
            __m256i vLhsMatYmm012330 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 192 + 512 * vSb)));
            __m256i vLhsMatYmm0130 = _mm256_permute2f128_si256(vLhsMatYmm012330, vLhsMatYmm012330, 0);
            __m256i vLhsMatYmm2330 = _mm256_permute2f128_si256(vLhsMatYmm012330, vLhsMatYmm012330, 17);
            __m256i vLhsMatYmm012331 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 224 + 512 * vSb)));
            __m256i vLhsMatYmm0131 = _mm256_permute2f128_si256(vLhsMatYmm012331, vLhsMatYmm012331, 0);
            __m256i vLhsMatYmm2331 = _mm256_permute2f128_si256(vLhsMatYmm012331, vLhsMatYmm012331, 17);

            __m256i vLhsMatYmm012340 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 256 + 512 * vSb)));
            __m256i vLhsMatYmm0140 = _mm256_permute2f128_si256(vLhsMatYmm012340, vLhsMatYmm012340, 0);
            __m256i vLhsMatYmm2340 = _mm256_permute2f128_si256(vLhsMatYmm012340, vLhsMatYmm012340, 17);
            __m256i vLhsMatYmm012341 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 288 + 512 * vSb)));
            __m256i vLhsMatYmm0141 = _mm256_permute2f128_si256(vLhsMatYmm012341, vLhsMatYmm012341, 0);
            __m256i vLhsMatYmm2341 = _mm256_permute2f128_si256(vLhsMatYmm012341, vLhsMatYmm012341, 17);
            __m256i vLhsMatYmm012350 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 320 + 512 * vSb)));
            __m256i vLhsMatYmm0150 = _mm256_permute2f128_si256(vLhsMatYmm012350, vLhsMatYmm012350, 0);
            __m256i vLhsMatYmm2350 = _mm256_permute2f128_si256(vLhsMatYmm012350, vLhsMatYmm012350, 17);
            __m256i vLhsMatYmm012351 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 352 + 512 * vSb)));
            __m256i vLhsMatYmm0151 = _mm256_permute2f128_si256(vLhsMatYmm012351, vLhsMatYmm012351, 0);
            __m256i vLhsMatYmm2351 = _mm256_permute2f128_si256(vLhsMatYmm012351, vLhsMatYmm012351, 17);
            __m256i vLhsMatYmm012360 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 384 + 512 * vSb)));
            __m256i vLhsMatYmm0160 = _mm256_permute2f128_si256(vLhsMatYmm012360, vLhsMatYmm012360, 0);
            __m256i vLhsMatYmm2360 = _mm256_permute2f128_si256(vLhsMatYmm012360, vLhsMatYmm012360, 17);
            __m256i vLhsMatYmm012361 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 416 + 512 * vSb)));
            __m256i vLhsMatYmm0161 = _mm256_permute2f128_si256(vLhsMatYmm012361, vLhsMatYmm012361, 0);
            __m256i vLhsMatYmm2361 = _mm256_permute2f128_si256(vLhsMatYmm012361, vLhsMatYmm012361, 17);
            __m256i vLhsMatYmm012370 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 448 + 512 * vSb)));
            __m256i vLhsMatYmm0170 = _mm256_permute2f128_si256(vLhsMatYmm012370, vLhsMatYmm012370, 0);
            __m256i vLhsMatYmm2370 = _mm256_permute2f128_si256(vLhsMatYmm012370, vLhsMatYmm012370, 17);
            __m256i vLhsMatYmm012371 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 480 + 512 * vSb)));
            __m256i vLhsMatYmm0171 = _mm256_permute2f128_si256(vLhsMatYmm012371, vLhsMatYmm012371, 0);
            __m256i vLhsMatYmm2371 = _mm256_permute2f128_si256(vLhsMatYmm012371, vLhsMatYmm012371, 17);

            __m512i vLhsMat0100 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0100), vLhsMatYmm0100, 1);
            __m512i vLhsMat2300 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2300), vLhsMatYmm2300, 1);
            __m512i vLhsMat0101 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0101), vLhsMatYmm0101, 1);
            __m512i vLhsMat2301 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2301), vLhsMatYmm2301, 1);

            __m512i vLhsMat0110 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0110), vLhsMatYmm0110, 1);
            __m512i vLhsMat2310 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2310), vLhsMatYmm2310, 1);
            __m512i vLhsMat0111 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0111), vLhsMatYmm0111, 1);
            __m512i vLhsMat2311 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2311), vLhsMatYmm2311, 1);

            __m512i vLhsMat0120 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0120), vLhsMatYmm0120, 1);
            __m512i vLhsMat2320 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2320), vLhsMatYmm2320, 1);
            __m512i vLhsMat0121 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0121), vLhsMatYmm0121, 1);
            __m512i vLhsMat2321 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2321), vLhsMatYmm2321, 1);

            __m512i vLhsMat0130 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0130), vLhsMatYmm0130, 1);
            __m512i vLhsMat2330 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2330), vLhsMatYmm2330, 1);
            __m512i vLhsMat0131 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0131), vLhsMatYmm0131, 1);
            __m512i vLhsMat2331 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2331), vLhsMatYmm2331, 1);

            __m512i vLhsMat0140 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0140), vLhsMatYmm0140, 1);
            __m512i vLhsMat2340 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2340), vLhsMatYmm2340, 1);
            __m512i vLhsMat0141 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0141), vLhsMatYmm0141, 1);
            __m512i vLhsMat2341 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2341), vLhsMatYmm2341, 1);

            __m512i vLhsMat0150 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0150), vLhsMatYmm0150, 1);
            __m512i vLhsMat2350 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2350), vLhsMatYmm2350, 1);
            __m512i vLhsMat0151 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0151), vLhsMatYmm0151, 1);
            __m512i vLhsMat2351 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2351), vLhsMatYmm2351, 1);

            __m512i vLhsMat0160 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0160), vLhsMatYmm0160, 1);
            __m512i vLhsMat2360 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2360), vLhsMatYmm2360, 1);
            __m512i vLhsMat0161 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0161), vLhsMatYmm0161, 1);
            __m512i vLhsMat2361 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2361), vLhsMatYmm2361, 1);

            __m512i vLhsMat0170 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0170), vLhsMatYmm0170, 1);
            __m512i vLhsMat2370 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2370), vLhsMatYmm2370, 1);
            __m512i vLhsMat0171 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0171), vLhsMatYmm0171, 1);
            __m512i vLhsMat2371 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2371), vLhsMatYmm2371, 1);

            // Bsums are loaded for the different Q8_K blocks
            __m128i vLhsRawBsums010123 = _mm_loadu_si128((const __m128i *)((vAPtrs[vRp][vB].bsums + 32 * vSb)));
            __m128i vLhsRawBsums230123 = _mm_loadu_si128((const __m128i *)(vAPtrs[vRp][vB].bsums + 8 + 32 * vSb));
            __m128i vLhsRawBsums014567 = _mm_loadu_si128((const __m128i *)((vAPtrs[vRp][vB].bsums + 16 + 32 * vSb)));
            __m128i vLhsRawBsums234567 = _mm_loadu_si128((const __m128i *)(vAPtrs[vRp][vB].bsums + 24 + 32 * vSb));

            __m256i vLhsBsumsYmm010123 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums010123), vLhsRawBsums010123, 1);
            __m512i vLhsBsums010123 =
                _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm010123), vLhsBsumsYmm010123, 1);
            __m256i vLhsBsumsYmm230123 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums230123), vLhsRawBsums230123, 1);
            __m512i vLhsBsums230123 =
                _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm230123), vLhsBsumsYmm230123, 1);
            __m256i vLhsBsumsYmm014567 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums014567), vLhsRawBsums014567, 1);
            __m512i vLhsBsums014567 =
                _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm014567), vLhsBsumsYmm014567, 1);
            __m256i vLhsBsumsYmm234567 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums234567), vLhsRawBsums234567, 1);
            __m512i vLhsBsums234567 =
                _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm234567), vLhsBsumsYmm234567, 1);

            // Shuffle pattern one - left side input
            const __m512i cLhsMat0100Sp1 = _mm512_shuffle_epi32(
                vLhsMat0100,
                (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                     // A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m512i cLhsMat2300Sp1 = _mm512_shuffle_epi32(
                vLhsMat2300,
                (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3)
                                     // A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

            const __m512i cLhsMat0101Sp1 = _mm512_shuffle_epi32(
                vLhsMat0101,
                (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                     // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m512i cLhsMat2301Sp1 = _mm512_shuffle_epi32(
                vLhsMat2301,
                (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                     // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

            const __m512i cLhsMat0110Sp1 = _mm512_shuffle_epi32(
                vLhsMat0110,
                (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                     // A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m512i cLhsMat2310Sp1 = _mm512_shuffle_epi32(
                vLhsMat2310,
                (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3)
                                     // A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

            const __m512i cLhsMat0111Sp1 = _mm512_shuffle_epi32(
                vLhsMat0111,
                (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                     // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m512i cLhsMat2311Sp1 = _mm512_shuffle_epi32(
                vLhsMat2311,
                (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                     // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

            const __m512i cLhsMat0120Sp1 = _mm512_shuffle_epi32(
                vLhsMat0120,
                (_MM_PERM_ENUM)160); // A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3)
                                     // A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
            const __m512i cLhsMat2320Sp1 = _mm512_shuffle_epi32(
                vLhsMat2320,
                (_MM_PERM_ENUM)160); // A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3)
                                     // A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

            const __m512i cLhsMat0121Sp1 = _mm512_shuffle_epi32(
                vLhsMat0121,
                (_MM_PERM_ENUM)160); // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                                     // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
            const __m512i cLhsMat2321Sp1 = _mm512_shuffle_epi32(
                vLhsMat2321,
                (_MM_PERM_ENUM)160); // A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)
                                     // A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

            const __m512i cLhsMat0130Sp1 = _mm512_shuffle_epi32(
                vLhsMat0130,
                (_MM_PERM_ENUM)160); // A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3)
                                     // A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
            const __m512i cLhsMat2330Sp1 = _mm512_shuffle_epi32(
                vLhsMat2330,
                (_MM_PERM_ENUM)160); // A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3)
                                     // A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

            const __m512i cLhsMat0131Sp1 = _mm512_shuffle_epi32(
                vLhsMat0131,
                (_MM_PERM_ENUM)160); // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                                     // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
            const __m512i cLhsMat2331Sp1 = _mm512_shuffle_epi32(
                vLhsMat2331,
                (_MM_PERM_ENUM)160); // A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)
                                     // A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

            const __m512i cLhsMat0140Sp1 = _mm512_shuffle_epi32(
                vLhsMat0140,
                (_MM_PERM_ENUM)160); // A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3)
                                     // A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
            const __m512i cLhsMat2340Sp1 = _mm512_shuffle_epi32(
                vLhsMat2340,
                (_MM_PERM_ENUM)160); // A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3)
                                     // A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

            const __m512i cLhsMat0141Sp1 = _mm512_shuffle_epi32(
                vLhsMat0141,
                (_MM_PERM_ENUM)160); // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                                     // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
            const __m512i cLhsMat2341Sp1 = _mm512_shuffle_epi32(
                vLhsMat2341,
                (_MM_PERM_ENUM)160); // A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)
                                     // A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

            const __m512i cLhsMat0150Sp1 = _mm512_shuffle_epi32(
                vLhsMat0150,
                (_MM_PERM_ENUM)160); // A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3)
                                     // A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
            const __m512i cLhsMat2350Sp1 = _mm512_shuffle_epi32(
                vLhsMat2350,
                (_MM_PERM_ENUM)160); // A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3)
                                     // A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

            const __m512i cLhsMat0151Sp1 = _mm512_shuffle_epi32(
                vLhsMat0151,
                (_MM_PERM_ENUM)160); // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                                     // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
            const __m512i cLhsMat2351Sp1 = _mm512_shuffle_epi32(
                vLhsMat2351,
                (_MM_PERM_ENUM)160); // A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)
                                     // A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

            const __m512i cLhsMat0160Sp1 = _mm512_shuffle_epi32(
                vLhsMat0160,
                (_MM_PERM_ENUM)160); // A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3)
                                     // A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
            const __m512i cLhsMat2360Sp1 = _mm512_shuffle_epi32(
                vLhsMat2360,
                (_MM_PERM_ENUM)160); // A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3)
                                     // A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

            const __m512i cLhsMat0161Sp1 = _mm512_shuffle_epi32(
                vLhsMat0161,
                (_MM_PERM_ENUM)160); // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                                     // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
            const __m512i cLhsMat2361Sp1 = _mm512_shuffle_epi32(
                vLhsMat2361,
                (_MM_PERM_ENUM)160); // A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)
                                     // A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

            const __m512i cLhsMat0170Sp1 = _mm512_shuffle_epi32(
                vLhsMat0170,
                (_MM_PERM_ENUM)160); // A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3)
                                     // A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
            const __m512i cLhsMat2370Sp1 = _mm512_shuffle_epi32(
                vLhsMat2370,
                (_MM_PERM_ENUM)160); // A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3)
                                     // A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

            const __m512i cLhsMat0171Sp1 = _mm512_shuffle_epi32(
                vLhsMat0171,
                (_MM_PERM_ENUM)160); // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                                     // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
            const __m512i cLhsMat2371Sp1 = _mm512_shuffle_epi32(
                vLhsMat2371,
                (_MM_PERM_ENUM)160); // A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)
                                     // A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

            const __m512i cLhsMat0100Sp2 = _mm512_shuffle_epi32(
                vLhsMat0100,
                (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                     // A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m512i cLhsMat2300Sp2 = _mm512_shuffle_epi32(
                vLhsMat2300,
                (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7)
                                     // A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

            const __m512i cLhsMat0101Sp2 = _mm512_shuffle_epi32(
                vLhsMat0101, (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15)
                                                  // A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                                  // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
            const __m512i cLhsMat2301Sp2 = _mm512_shuffle_epi32(
                vLhsMat2301, (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15)
                                                  // A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                                  // A02(12-15) A02(12-15) A03(12-15) A03(12-15)

            const __m512i cLhsMat0110Sp2 = _mm512_shuffle_epi32(
                vLhsMat0110,
                (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                     // A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m512i cLhsMat2310Sp2 = _mm512_shuffle_epi32(
                vLhsMat2310,
                (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7)
                                     // A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

            const __m512i cLhsMat0111Sp2 = _mm512_shuffle_epi32(
                vLhsMat0111, (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15)
                                                  // A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                                  // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
            const __m512i cLhsMat2311Sp2 = _mm512_shuffle_epi32(
                vLhsMat2311, (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15)
                                                  // A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                                  // A12(12-15) A12(12-15) A13(12-15) A13(12-15)

            const __m512i cLhsMat0120Sp2 = _mm512_shuffle_epi32(
                vLhsMat0120,
                (_MM_PERM_ENUM)245); // A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7)
                                     // A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
            const __m512i cLhsMat2320Sp2 = _mm512_shuffle_epi32(
                vLhsMat2320,
                (_MM_PERM_ENUM)245); // A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7)
                                     // A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

            const __m512i cLhsMat0121Sp2 = _mm512_shuffle_epi32(
                vLhsMat0121, (_MM_PERM_ENUM)245); // A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15)
                                                  // A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                                                  // A20(12-15) A20(12-15) A21(12-15) A21(12-15)
            const __m512i cLhsMat2321Sp2 = _mm512_shuffle_epi32(
                vLhsMat2321, (_MM_PERM_ENUM)245); // A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15)
                                                  // A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)
                                                  // A22(12-15) A22(12-15) A23(12-15) A23(12-15)

            const __m512i cLhsMat0130Sp2 = _mm512_shuffle_epi32(
                vLhsMat0130,
                (_MM_PERM_ENUM)245); // A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7)
                                     // A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
            const __m512i cLhsMat2330Sp2 = _mm512_shuffle_epi32(
                vLhsMat2330,
                (_MM_PERM_ENUM)245); // A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7)
                                     // A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

            const __m512i cLhsMat0131Sp2 = _mm512_shuffle_epi32(
                vLhsMat0131, (_MM_PERM_ENUM)245); // A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15)
                                                  // A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                                                  // A30(12-15) A30(12-15) A31(12-15) A31(12-15)
            const __m512i cLhsMat2331Sp2 = _mm512_shuffle_epi32(
                vLhsMat2331, (_MM_PERM_ENUM)245); // A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15)
                                                  // A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)
                                                  // A32(12-15) A32(12-15) A33(12-15) A33(12-15)

            const __m512i cLhsMat0140Sp2 = _mm512_shuffle_epi32(
                vLhsMat0140,
                (_MM_PERM_ENUM)245); // A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7)
                                     // A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
            const __m512i cLhsMat2340Sp2 = _mm512_shuffle_epi32(
                vLhsMat2340,
                (_MM_PERM_ENUM)245); // A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7)
                                     // A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

            const __m512i cLhsMat0141Sp2 = _mm512_shuffle_epi32(
                vLhsMat0141, (_MM_PERM_ENUM)245); // A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15)
                                                  // A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                                                  // A40(12-15) A40(12-15) A41(12-15) A41(12-15)
            const __m512i cLhsMat2341Sp2 = _mm512_shuffle_epi32(
                vLhsMat2341, (_MM_PERM_ENUM)245); // A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15)
                                                  // A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)
                                                  // A42(12-15) A42(12-15) A43(12-15) A43(12-15)

            const __m512i cLhsMat0150Sp2 = _mm512_shuffle_epi32(
                vLhsMat0150,
                (_MM_PERM_ENUM)245); // A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7)
                                     // A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
            const __m512i cLhsMat2350Sp2 = _mm512_shuffle_epi32(
                vLhsMat2350,
                (_MM_PERM_ENUM)245); // A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7)
                                     // A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

            const __m512i cLhsMat0151Sp2 = _mm512_shuffle_epi32(
                vLhsMat0151, (_MM_PERM_ENUM)245); // A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15)
                                                  // A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                                                  // A50(12-15) A50(12-15) A51(12-15) A51(12-15)
            const __m512i cLhsMat2351Sp2 = _mm512_shuffle_epi32(
                vLhsMat2351, (_MM_PERM_ENUM)245); // A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15)
                                                  // A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)
                                                  // A52(12-15) A52(12-15) A53(12-15) A53(12-15)

            const __m512i cLhsMat0160Sp2 = _mm512_shuffle_epi32(
                vLhsMat0160,
                (_MM_PERM_ENUM)245); // A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7)
                                     // A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
            const __m512i cLhsMat2360Sp2 = _mm512_shuffle_epi32(
                vLhsMat2360,
                (_MM_PERM_ENUM)245); // A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7)
                                     // A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

            const __m512i cLhsMat0161Sp2 = _mm512_shuffle_epi32(
                vLhsMat0161, (_MM_PERM_ENUM)245); // A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15)
                                                  // A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                                                  // A60(12-15) A60(12-15) A61(12-15) A61(12-15)
            const __m512i cLhsMat2361Sp2 = _mm512_shuffle_epi32(
                vLhsMat2361, (_MM_PERM_ENUM)245); // A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15)
                                                  // A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)
                                                  // A62(12-15) A62(12-15) A63(12-15) A63(12-15)

            const __m512i cLhsMat0170Sp2 = _mm512_shuffle_epi32(
                vLhsMat0170,
                (_MM_PERM_ENUM)245); // A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7)
                                     // A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
            const __m512i cLhsMat2370Sp2 = _mm512_shuffle_epi32(
                vLhsMat2370,
                (_MM_PERM_ENUM)245); // A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7)
                                     // A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

            const __m512i cLhsMat0171Sp2 = _mm512_shuffle_epi32(
                vLhsMat0171, (_MM_PERM_ENUM)245); // A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15)
                                                  // A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                                                  // A70(12-15) A70(12-15) A71(12-15) A71(12-15)
            const __m512i cLhsMat2371Sp2 = _mm512_shuffle_epi32(
                vLhsMat2371, (_MM_PERM_ENUM)245); // A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15)
                                                  // A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)
                                                  // A72(12-15) A72(12-15) A73(12-15) A73(12-15)

            // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
            // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
            __m512i vIaccMat000Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat0100Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat0101Sp1));
            __m512i vIaccMat010Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat0100Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat0101Sp1));

            __m512i vIaccMat100Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat2300Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat2301Sp1));
            __m512i vIaccMat110Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat2300Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat2301Sp1));

            __m512i vIaccMat001Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat0110Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat0111Sp1));
            __m512i vIaccMat011Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat0110Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat0111Sp1));

            __m512i vIaccMat101Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat2310Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat2311Sp1));
            __m512i vIaccMat111Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat2310Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat2311Sp1));

            __m512i vIaccMat002Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp1, cLhsMat0120Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd21Sp1, cLhsMat0121Sp1));
            __m512i vIaccMat012Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp1, cLhsMat0120Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef21Sp1, cLhsMat0121Sp1));

            __m512i vIaccMat102Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp1, cLhsMat2320Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd21Sp1, cLhsMat2321Sp1));
            __m512i vIaccMat112Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp1, cLhsMat2320Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef21Sp1, cLhsMat2321Sp1));

            __m512i vIaccMat003Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp1, cLhsMat0130Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd31Sp1, cLhsMat0131Sp1));
            __m512i vIaccMat013Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp1, cLhsMat0130Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef31Sp1, cLhsMat0131Sp1));

            __m512i vIaccMat103Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp1, cLhsMat2330Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd31Sp1, cLhsMat2331Sp1));
            __m512i vIaccMat113Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp1, cLhsMat2330Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef31Sp1, cLhsMat2331Sp1));

            __m512i vIaccMat004Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp1, cLhsMat0140Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd41Sp1, cLhsMat0141Sp1));
            __m512i vIaccMat014Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp1, cLhsMat0140Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef41Sp1, cLhsMat0141Sp1));

            __m512i vIaccMat104Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp1, cLhsMat2340Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd41Sp1, cLhsMat2341Sp1));
            __m512i vIaccMat114Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp1, cLhsMat2340Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef41Sp1, cLhsMat2341Sp1));

            __m512i vIaccMat005Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp1, cLhsMat0150Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd51Sp1, cLhsMat0151Sp1));
            __m512i vIaccMat015Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp1, cLhsMat0150Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef51Sp1, cLhsMat0151Sp1));

            __m512i vIaccMat105Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp1, cLhsMat2350Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd51Sp1, cLhsMat2351Sp1));
            __m512i vIaccMat115Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp1, cLhsMat2350Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef51Sp1, cLhsMat2351Sp1));

            __m512i vIaccMat006Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp1, cLhsMat0160Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd61Sp1, cLhsMat0161Sp1));
            __m512i vIaccMat016Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp1, cLhsMat0160Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef61Sp1, cLhsMat0161Sp1));

            __m512i vIaccMat106Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp1, cLhsMat2360Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd61Sp1, cLhsMat2361Sp1));
            __m512i vIaccMat116Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp1, cLhsMat2360Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef61Sp1, cLhsMat2361Sp1));

            __m512i vIaccMat007Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp1, cLhsMat0170Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd71Sp1, cLhsMat0171Sp1));
            __m512i vIaccMat017Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp1, cLhsMat0170Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef71Sp1, cLhsMat0171Sp1));

            __m512i vIaccMat107Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp1, cLhsMat2370Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd71Sp1, cLhsMat2371Sp1));
            __m512i vIaccMat117Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp1, cLhsMat2370Sp1),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef71Sp1, cLhsMat2371Sp1));

            __m512i vIaccMat000Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat0100Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat0101Sp2));
            __m512i vIaccMat010Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat0100Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat0101Sp2));

            __m512i vIaccMat100Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat2300Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat2301Sp2));
            __m512i vIaccMat110Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat2300Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat2301Sp2));

            __m512i vIaccMat001Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat0110Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat0111Sp2));
            __m512i vIaccMat011Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat0110Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat0111Sp2));

            __m512i vIaccMat101Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat2310Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat2311Sp2));
            __m512i vIaccMat111Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat2310Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat2311Sp2));

            __m512i vIaccMat002Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp2, cLhsMat0120Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd21Sp2, cLhsMat0121Sp2));
            __m512i vIaccMat012Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp2, cLhsMat0120Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef21Sp2, cLhsMat0121Sp2));

            __m512i vIaccMat102Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp2, cLhsMat2320Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd21Sp2, cLhsMat2321Sp2));
            __m512i vIaccMat112Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp2, cLhsMat2320Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef21Sp2, cLhsMat2321Sp2));

            __m512i vIaccMat003Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp2, cLhsMat0130Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd31Sp2, cLhsMat0131Sp2));
            __m512i vIaccMat013Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp2, cLhsMat0130Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef31Sp2, cLhsMat0131Sp2));

            __m512i vIaccMat103Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp2, cLhsMat2330Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd31Sp2, cLhsMat2331Sp2));
            __m512i vIaccMat113Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp2, cLhsMat2330Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef31Sp2, cLhsMat2331Sp2));

            __m512i vIaccMat004Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp2, cLhsMat0140Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd41Sp2, cLhsMat0141Sp2));
            __m512i vIaccMat014Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp2, cLhsMat0140Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef41Sp2, cLhsMat0141Sp2));

            __m512i vIaccMat104Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp2, cLhsMat2340Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd41Sp2, cLhsMat2341Sp2));
            __m512i vIaccMat114Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp2, cLhsMat2340Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef41Sp2, cLhsMat2341Sp2));

            __m512i vIaccMat005Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp2, cLhsMat0150Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd51Sp2, cLhsMat0151Sp2));
            __m512i vIaccMat015Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp2, cLhsMat0150Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef51Sp2, cLhsMat0151Sp2));

            __m512i vIaccMat105Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp2, cLhsMat2350Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd51Sp2, cLhsMat2351Sp2));
            __m512i vIaccMat115Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp2, cLhsMat2350Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef51Sp2, cLhsMat2351Sp2));

            __m512i vIaccMat006Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp2, cLhsMat0160Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd61Sp2, cLhsMat0161Sp2));
            __m512i vIaccMat016Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp2, cLhsMat0160Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef61Sp2, cLhsMat0161Sp2));

            __m512i vIaccMat106Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp2, cLhsMat2360Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd61Sp2, cLhsMat2361Sp2));
            __m512i vIaccMat116Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp2, cLhsMat2360Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef61Sp2, cLhsMat2361Sp2));

            __m512i vIaccMat007Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp2, cLhsMat0170Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd71Sp2, cLhsMat0171Sp2));
            __m512i vIaccMat017Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp2, cLhsMat0170Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef71Sp2, cLhsMat0171Sp2));

            __m512i vIaccMat107Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp2, cLhsMat2370Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat014589Cd71Sp2, cLhsMat2371Sp2));
            __m512i vIaccMat117Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp2, cLhsMat2370Sp2),
                                                      _mm512_maddubs_epi16(cRhsMat2367Abef71Sp2, cLhsMat2371Sp2));

            // Combine results from both shuffle patterns for each output block
            __m512i vIaccMat000 = _mm512_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
            __m512i vIaccMat010 = _mm512_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
            __m512i vIaccMat100 = _mm512_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
            __m512i vIaccMat110 = _mm512_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

            __m512i vIaccMat001 = _mm512_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
            __m512i vIaccMat011 = _mm512_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
            __m512i vIaccMat101 = _mm512_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
            __m512i vIaccMat111 = _mm512_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

            __m512i vIaccMat002 = _mm512_add_epi16(vIaccMat002Sp1, vIaccMat002Sp2);
            __m512i vIaccMat012 = _mm512_add_epi16(vIaccMat012Sp1, vIaccMat012Sp2);
            __m512i vIaccMat102 = _mm512_add_epi16(vIaccMat102Sp1, vIaccMat102Sp2);
            __m512i vIaccMat112 = _mm512_add_epi16(vIaccMat112Sp1, vIaccMat112Sp2);

            __m512i vIaccMat003 = _mm512_add_epi16(vIaccMat003Sp1, vIaccMat003Sp2);
            __m512i vIaccMat013 = _mm512_add_epi16(vIaccMat013Sp1, vIaccMat013Sp2);
            __m512i vIaccMat103 = _mm512_add_epi16(vIaccMat103Sp1, vIaccMat103Sp2);
            __m512i vIaccMat113 = _mm512_add_epi16(vIaccMat113Sp1, vIaccMat113Sp2);

            __m512i vIaccMat004 = _mm512_add_epi16(vIaccMat004Sp1, vIaccMat004Sp2);
            __m512i vIaccMat014 = _mm512_add_epi16(vIaccMat014Sp1, vIaccMat014Sp2);
            __m512i vIaccMat104 = _mm512_add_epi16(vIaccMat104Sp1, vIaccMat104Sp2);
            __m512i vIaccMat114 = _mm512_add_epi16(vIaccMat114Sp1, vIaccMat114Sp2);

            __m512i vIaccMat005 = _mm512_add_epi16(vIaccMat005Sp1, vIaccMat005Sp2);
            __m512i vIaccMat015 = _mm512_add_epi16(vIaccMat015Sp1, vIaccMat015Sp2);
            __m512i vIaccMat105 = _mm512_add_epi16(vIaccMat105Sp1, vIaccMat105Sp2);
            __m512i vIaccMat115 = _mm512_add_epi16(vIaccMat115Sp1, vIaccMat115Sp2);

            __m512i vIaccMat006 = _mm512_add_epi16(vIaccMat006Sp1, vIaccMat006Sp2);
            __m512i vIaccMat016 = _mm512_add_epi16(vIaccMat016Sp1, vIaccMat016Sp2);
            __m512i vIaccMat106 = _mm512_add_epi16(vIaccMat106Sp1, vIaccMat106Sp2);
            __m512i vIaccMat116 = _mm512_add_epi16(vIaccMat116Sp1, vIaccMat116Sp2);

            __m512i vIaccMat007 = _mm512_add_epi16(vIaccMat007Sp1, vIaccMat007Sp2);
            __m512i vIaccMat017 = _mm512_add_epi16(vIaccMat017Sp1, vIaccMat017Sp2);
            __m512i vIaccMat107 = _mm512_add_epi16(vIaccMat107Sp1, vIaccMat107Sp2);
            __m512i vIaccMat117 = _mm512_add_epi16(vIaccMat117Sp1, vIaccMat117Sp2);

            // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
            vIaccMat000 = _mm512_madd_epi16(vIaccMat000, cScale014589Cd0);
            vIaccMat010 = _mm512_madd_epi16(vIaccMat010, cScale2367Abef0);
            vIaccMat100 = _mm512_madd_epi16(vIaccMat100, cScale014589Cd0);
            vIaccMat110 = _mm512_madd_epi16(vIaccMat110, cScale2367Abef0);

            vIaccMat001 = _mm512_madd_epi16(vIaccMat001, cScale014589Cd1);
            vIaccMat011 = _mm512_madd_epi16(vIaccMat011, cScale2367Abef1);
            vIaccMat101 = _mm512_madd_epi16(vIaccMat101, cScale014589Cd1);
            vIaccMat111 = _mm512_madd_epi16(vIaccMat111, cScale2367Abef1);

            vIaccMat002 = _mm512_madd_epi16(vIaccMat002, cScale014589Cd2);
            vIaccMat012 = _mm512_madd_epi16(vIaccMat012, cScale2367Abef2);
            vIaccMat102 = _mm512_madd_epi16(vIaccMat102, cScale014589Cd2);
            vIaccMat112 = _mm512_madd_epi16(vIaccMat112, cScale2367Abef2);

            vIaccMat003 = _mm512_madd_epi16(vIaccMat003, cScale014589Cd3);
            vIaccMat013 = _mm512_madd_epi16(vIaccMat013, cScale2367Abef3);
            vIaccMat103 = _mm512_madd_epi16(vIaccMat103, cScale014589Cd3);
            vIaccMat113 = _mm512_madd_epi16(vIaccMat113, cScale2367Abef3);

            vIaccMat004 = _mm512_madd_epi16(vIaccMat004, cScale014589Cd4);
            vIaccMat014 = _mm512_madd_epi16(vIaccMat014, cScale2367Abef4);
            vIaccMat104 = _mm512_madd_epi16(vIaccMat104, cScale014589Cd4);
            vIaccMat114 = _mm512_madd_epi16(vIaccMat114, cScale2367Abef4);

            vIaccMat005 = _mm512_madd_epi16(vIaccMat005, cScale014589Cd5);
            vIaccMat015 = _mm512_madd_epi16(vIaccMat015, cScale2367Abef5);
            vIaccMat105 = _mm512_madd_epi16(vIaccMat105, cScale014589Cd5);
            vIaccMat115 = _mm512_madd_epi16(vIaccMat115, cScale2367Abef5);

            vIaccMat006 = _mm512_madd_epi16(vIaccMat006, cScale014589Cd6);
            vIaccMat016 = _mm512_madd_epi16(vIaccMat016, cScale2367Abef6);
            vIaccMat106 = _mm512_madd_epi16(vIaccMat106, cScale014589Cd6);
            vIaccMat116 = _mm512_madd_epi16(vIaccMat116, cScale2367Abef6);

            vIaccMat007 = _mm512_madd_epi16(vIaccMat007, cScale014589Cd7);
            vIaccMat017 = _mm512_madd_epi16(vIaccMat017, cScale2367Abef7);
            vIaccMat107 = _mm512_madd_epi16(vIaccMat107, cScale014589Cd7);
            vIaccMat117 = _mm512_madd_epi16(vIaccMat117, cScale2367Abef7);

            __m512i vIaccMat00 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(vIaccMat000, vIaccMat001),
                                                                   _mm512_add_epi32(vIaccMat002, vIaccMat003)),
                                                  _mm512_add_epi32(_mm512_add_epi32(vIaccMat004, vIaccMat005),
                                                                   _mm512_add_epi32(vIaccMat006, vIaccMat007)));
            __m512i vIaccMat01 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(vIaccMat010, vIaccMat011),
                                                                   _mm512_add_epi32(vIaccMat012, vIaccMat013)),
                                                  _mm512_add_epi32(_mm512_add_epi32(vIaccMat014, vIaccMat015),
                                                                   _mm512_add_epi32(vIaccMat016, vIaccMat017)));
            __m512i vIaccMat10 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(vIaccMat100, vIaccMat101),
                                                                   _mm512_add_epi32(vIaccMat102, vIaccMat103)),
                                                  _mm512_add_epi32(_mm512_add_epi32(vIaccMat104, vIaccMat105),
                                                                   _mm512_add_epi32(vIaccMat106, vIaccMat107)));
            __m512i vIaccMat11 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(vIaccMat110, vIaccMat111),
                                                                   _mm512_add_epi32(vIaccMat112, vIaccMat113)),
                                                  _mm512_add_epi32(_mm512_add_epi32(vIaccMat114, vIaccMat115),
                                                                   _mm512_add_epi32(vIaccMat116, vIaccMat117)));

            // Straighten out to make 4 row vectors
            __m512i vIaccRow0 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat00, _mm512_shuffle_epi32(vIaccMat01, (_MM_PERM_ENUM)78));
            __m512i vIaccRow1 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat00, (_MM_PERM_ENUM)78), vIaccMat01);
            __m512i vIaccRow2 =
                _mm512_mask_blend_epi32(0xCCCC, vIaccMat10, _mm512_shuffle_epi32(vIaccMat11, (_MM_PERM_ENUM)78));
            __m512i vIaccRow3 =
                _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat10, (_MM_PERM_ENUM)78), vIaccMat11);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
            const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtrs[vRp][vB].d);
            const __m256 cRowScaleF32Ymm = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);
            const __m512 cRowScaleF32 = _mm512_insertf32x8(_mm512_castps256_ps512(cRowScaleF32Ymm), cRowScaleF32Ymm, 1);

            // Multiply with appropriate scales and accumulate (for both d and dmin) below
            aAccRows[vRp * 4] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow0),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
            aAccRows[vRp * 4 + 1] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow1),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
            aAccRows[vRp * 4 + 2] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow2),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
            aAccRows[vRp * 4 + 3] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRow3),
                _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);

            // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
            __m512i vIaccRowMin001 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)0), cMins01);
            __m512i vIaccRowMin101 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)170), cMins01);
            __m512i vIaccRowMin201 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)0), cMins01);
            __m512i vIaccRowMin301 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)170), cMins01);

            __m512i vIaccRowMin023 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)85), cMins23);
            __m512i vIaccRowMin123 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)255), cMins23);
            __m512i vIaccRowMin223 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)85), cMins23);
            __m512i vIaccRowMin323 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)255), cMins23);

            __m512i vIaccRowMin045 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)0), cMins45);
            __m512i vIaccRowMin145 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)170), cMins45);
            __m512i vIaccRowMin245 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)0), cMins45);
            __m512i vIaccRowMin345 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)170), cMins45);

            __m512i vIaccRowMin067 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)85), cMins67);
            __m512i vIaccRowMin167 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)255), cMins67);
            __m512i vIaccRowMin267 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)85), cMins67);
            __m512i vIaccRowMin367 =
                _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)255), cMins67);

            __m512i vIaccRowMin0 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin001, vIaccRowMin023),
                                                    _mm512_add_epi32(vIaccRowMin045, vIaccRowMin067));
            __m512i vIaccRowMin1 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin101, vIaccRowMin123),
                                                    _mm512_add_epi32(vIaccRowMin145, vIaccRowMin167));
            __m512i vIaccRowMin2 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin201, vIaccRowMin223),
                                                    _mm512_add_epi32(vIaccRowMin245, vIaccRowMin267));
            __m512i vIaccRowMin3 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin301, vIaccRowMin323),
                                                    _mm512_add_epi32(vIaccRowMin345, vIaccRowMin367));

            aAccMinRows[vRp * 4] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(vIaccRowMin0),
                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccMinRows[vRp * 4]);
            aAccMinRows[vRp * 4 + 1] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin1),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                aAccMinRows[vRp * 4 + 1]);
            aAccMinRows[vRp * 4 + 2] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin2),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                aAccMinRows[vRp * 4 + 2]);
            aAccMinRows[vRp * 4 + 3] =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin3),
                                _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                aAccMinRows[vRp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm512_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }

  for (; vY < pNr / 4; vY++) {

    const block_q8_Kx4 *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = 0; vX < vAnc / 8; vX += 2) {

      const block_q2_Kx8 *vBPtr0 = vBPtrStart + ((vX)*vBNb);
      const block_q2_Kx8 *vBPtr1 = vBPtrStart + ((vX + 1) * vBNb);

      // Master FP accumulators
      __m512 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm512_setzero_ps();
      }

      __m512 aAccMinRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccMinRows[vI] = _mm512_setzero_ps();
      }
      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {
        // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
        const __m512 cColScaleF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].d, vBPtr1[vB].d);

        // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
        const __m512 cColDminF32 = GGML_F32Cx8x2_LOAD(vBPtr0[vB].dmin, vBPtr1[vB].dmin);

        // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 128; vSb++) {

          // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr0[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat89Ab0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + vSb * 256));
          const __m256i cRhsRawMatCdef0 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat89Ab1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMatCdef1 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat89Ab2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMatCdef2 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat89Ab3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMatCdef3 = _mm256_loadu_si256((const __m256i *)(vBPtr1[vB].qs + 224 + vSb * 256));

          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);
          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);
          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);
          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          const __m256i cRhsRawMat89Cd0 =
              _mm256_blend_epi32(cRhsRawMat89Ab0, _mm256_permutevar8x32_epi32(cRhsRawMatCdef0, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef0 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab0, vRequiredOrder), cRhsRawMatCdef0, 240);
          const __m256i cRhsRawMat89Cd1 =
              _mm256_blend_epi32(cRhsRawMat89Ab1, _mm256_permutevar8x32_epi32(cRhsRawMatCdef1, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef1 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab1, vRequiredOrder), cRhsRawMatCdef1, 240);
          const __m256i cRhsRawMat89Cd2 =
              _mm256_blend_epi32(cRhsRawMat89Ab2, _mm256_permutevar8x32_epi32(cRhsRawMatCdef2, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef2 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab2, vRequiredOrder), cRhsRawMatCdef2, 240);
          const __m256i cRhsRawMat89Cd3 =
              _mm256_blend_epi32(cRhsRawMat89Ab3, _mm256_permutevar8x32_epi32(cRhsRawMatCdef3, vRequiredOrder), 240);
          const __m256i cRhsRawMatAbef3 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat89Ab3, vRequiredOrder), cRhsRawMatCdef3, 240);

          const __m512i cRhsRawMat014589Cd0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01450), cRhsRawMat89Cd0, 1);
          const __m512i cRhsRawMat2367Abef0 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23670), cRhsRawMatAbef0, 1);
          const __m512i cRhsRawMat014589Cd1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01451), cRhsRawMat89Cd1, 1);
          const __m512i cRhsRawMat2367Abef1 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23671), cRhsRawMatAbef1, 1);

          const __m512i cRhsRawMat014589Cd2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01452), cRhsRawMat89Cd2, 1);
          const __m512i cRhsRawMat2367Abef2 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23672), cRhsRawMatAbef2, 1);
          const __m512i cRhsRawMat014589Cd3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat01453), cRhsRawMat89Cd3, 1);
          const __m512i cRhsRawMat2367Abef3 =
              _mm512_inserti32x8(_mm512_castsi256_si512(cRhsRawMat23673), cRhsRawMatAbef3, 1);

          // 2-bit -> 8-bit
          const __m512i cRhsMat014589Cd00 =
              _mm512_and_si512(cRhsRawMat014589Cd0,
                               cM3bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i cRhsMat2367Abef00 =
              _mm512_and_si512(cRhsRawMat2367Abef0,
                               cM3bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i cRhsMat014589Cd01 = _mm512_and_si512(
              cRhsRawMat014589Cd1,
              cM3bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i cRhsMat2367Abef01 = _mm512_and_si512(
              cRhsRawMat2367Abef1,
              cM3bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
          const __m512i cRhsMat014589Cd10 =
              _mm512_and_si512(cRhsRawMat014589Cd2,
                               cM3bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i cRhsMat2367Abef10 =
              _mm512_and_si512(cRhsRawMat2367Abef2,
                               cM3bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i cRhsMat014589Cd11 = _mm512_and_si512(
              cRhsRawMat014589Cd3,
              cM3bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i cRhsMat2367Abef11 = _mm512_and_si512(
              cRhsRawMat2367Abef3,
              cM3bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i cRhsMat014589Cd20 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 2),
                               cM3bexpanded); // B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
          const __m512i cRhsMat2367Abef20 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 2),
                               cM3bexpanded); // B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

          const __m512i cRhsMat014589Cd21 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 2),
              cM3bexpanded); // B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
          const __m512i cRhsMat2367Abef21 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 2),
              cM3bexpanded); // B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

          const __m512i cRhsMat014589Cd30 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 2),
                               cM3bexpanded); // B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
          const __m512i cRhsMat2367Abef30 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 2),
                               cM3bexpanded); // B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

          const __m512i cRhsMat014589Cd31 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 2),
              cM3bexpanded); // B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
          const __m512i cRhsMat2367Abef31 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 2),
              cM3bexpanded); // B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

          const __m512i cRhsMat014589Cd40 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 4),
                               cM3bexpanded); // B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
          const __m512i cRhsMat2367Abef40 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 4),
                               cM3bexpanded); // B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

          const __m512i cRhsMat014589Cd41 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 4),
              cM3bexpanded); // B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
          const __m512i cRhsMat2367Abef41 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 4),
              cM3bexpanded); // B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

          const __m512i cRhsMat014589Cd50 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 4),
                               cM3bexpanded); // B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
          const __m512i cRhsMat2367Abef50 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 4),
                               cM3bexpanded); // B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

          const __m512i cRhsMat014589Cd51 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 4),
              cM3bexpanded); // B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
          const __m512i cRhsMat2367Abef51 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 4),
              cM3bexpanded); // B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

          const __m512i cRhsMat014589Cd60 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd0, 6),
                               cM3bexpanded); // B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
          const __m512i cRhsMat2367Abef60 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef0, 6),
                               cM3bexpanded); // B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

          const __m512i cRhsMat014589Cd61 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd1, 6),
              cM3bexpanded); // B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
          const __m512i cRhsMat2367Abef61 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef1, 6),
              cM3bexpanded); // B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

          const __m512i cRhsMat014589Cd70 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat014589Cd2, 6),
                               cM3bexpanded); // B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
          const __m512i cRhsMat2367Abef70 =
              _mm512_and_si512(_mm512_srli_epi16(cRhsRawMat2367Abef2, 6),
                               cM3bexpanded); // B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

          const __m512i cRhsMat014589Cd71 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat014589Cd3, 6),
              cM3bexpanded); // B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
          const __m512i cRhsMat2367Abef71 = _mm512_and_si512(
              _mm512_srli_epi16(cRhsRawMat2367Abef3, 6),
              cM3bexpanded); // B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

          const __m512i cRhsMat014589Cd00Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3)
                                   // B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
          const __m512i cRhsMat2367Abef00Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3)
                                   // B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

          const __m512i cRhsMat014589Cd01Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef01Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                   // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

          const __m512i cRhsMat014589Cd10Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3)
                                   // B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
          const __m512i cRhsMat2367Abef10Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3)
                                   // B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

          const __m512i cRhsMat014589Cd11Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                   // B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i cRhsMat2367Abef11Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                   // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

          const __m512i cRhsMat014589Cd20Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd20,
              (_MM_PERM_ENUM)136); // B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3)
                                   // B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
          const __m512i cRhsMat2367Abef20Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef20,
              (_MM_PERM_ENUM)136); // B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3)
                                   // B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

          const __m512i cRhsMat014589Cd21Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd21,
              (_MM_PERM_ENUM)136); // B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                                   // B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
          const __m512i cRhsMat2367Abef21Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef21,
              (_MM_PERM_ENUM)136); // B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)
                                   // B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)
          const __m512i cRhsMat014589Cd30Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd30,
              (_MM_PERM_ENUM)136); /// B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3)
                                   /// B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
          const __m512i cRhsMat2367Abef30Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef30,
              (_MM_PERM_ENUM)136); // B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3)
                                   // B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

          const __m512i cRhsMat014589Cd31Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd31,
              (_MM_PERM_ENUM)136); // B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11)
                                   // B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
          const __m512i cRhsMat2367Abef31Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef31,
              (_MM_PERM_ENUM)136); // B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)
                                   // B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

          const __m512i cRhsMat014589Cd40Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd40,
              (_MM_PERM_ENUM)136); // B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3)
                                   // B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
          const __m512i cRhsMat2367Abef40Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef40,
              (_MM_PERM_ENUM)136); // B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3)
                                   // B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

          const __m512i cRhsMat014589Cd41Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd41,
              (_MM_PERM_ENUM)136); // B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                                   // B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
          const __m512i cRhsMat2367Abef41Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef41,
              (_MM_PERM_ENUM)136); // B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)
                                   // B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

          const __m512i cRhsMat014589Cd50Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd50,
              (_MM_PERM_ENUM)136); // B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3)
                                   // B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
          const __m512i cRhsMat2367Abef50Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef50,
              (_MM_PERM_ENUM)136); // B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3)
                                   // B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

          const __m512i cRhsMat014589Cd51Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd51,
              (_MM_PERM_ENUM)136); // B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                                   // B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
          const __m512i cRhsMat2367Abef51Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef51,
              (_MM_PERM_ENUM)136); // B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)
                                   // B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

          const __m512i cRhsMat014589Cd60Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd60,
              (_MM_PERM_ENUM)136); // B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3)
                                   // B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
          const __m512i cRhsMat2367Abef60Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef60,
              (_MM_PERM_ENUM)136); // B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3)
                                   // B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

          const __m512i cRhsMat014589Cd61Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd61,
              (_MM_PERM_ENUM)136); // B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                                   // B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
          const __m512i cRhsMat2367Abef61Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef61,
              (_MM_PERM_ENUM)136); // B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)
                                   // B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

          const __m512i cRhsMat014589Cd70Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd70,
              (_MM_PERM_ENUM)136); // B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3)
                                   // B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
          const __m512i cRhsMat2367Abef70Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef70,
              (_MM_PERM_ENUM)136); // B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3)
                                   // B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

          const __m512i cRhsMat014589Cd71Sp1 = _mm512_shuffle_epi32(
              cRhsMat014589Cd71,
              (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                   // B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i cRhsMat2367Abef71Sp1 = _mm512_shuffle_epi32(
              cRhsMat2367Abef71,
              (_MM_PERM_ENUM)136); // B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)
                                   // B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

          const __m512i cRhsMat014589Cd00Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd00,
              (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7)
                                   // B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
          const __m512i cRhsMat2367Abef00Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef00,
              (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7)
                                   // B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

          const __m512i cRhsMat014589Cd01Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd01,
              (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15)
                                   // B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15)
                                   // B0C(12-15) B0D(12-15)
          const __m512i cRhsMat2367Abef01Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef01,
              (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15)
                                   // B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15)
                                   // B0E(12-15) B0F(12-15)

          const __m512i cRhsMat014589Cd10Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd10,
              (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7)
                                   // B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
          const __m512i cRhsMat2367Abef10Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef10,
              (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7)
                                   // B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

          const __m512i cRhsMat014589Cd11Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd11,
              (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15)
                                   // B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15)
                                   // B1C(12-15) B1D(12-15)
          const __m512i cRhsMat2367Abef11Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef11,
              (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15)
                                   // B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15)
                                   // B1E(12-15) B1F(12-15)

          const __m512i cRhsMat014589Cd20Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd20,
              (_MM_PERM_ENUM)221); // B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7)
                                   // B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
          const __m512i cRhsMat2367Abef20Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef20,
              (_MM_PERM_ENUM)221); // B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7)
                                   // B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

          const __m512i cRhsMat014589Cd21Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd21,
              (_MM_PERM_ENUM)221); // B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15)
                                   // B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15)
                                   // B2C(12-15) B2D(12-15)
          const __m512i cRhsMat2367Abef21Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef21,
              (_MM_PERM_ENUM)221); // B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15)
                                   // B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15)
                                   // B2E(12-15) B2F(12-15)

          const __m512i cRhsMat014589Cd30Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd30,
              (_MM_PERM_ENUM)221); // B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7)
                                   // B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
          const __m512i cRhsMat2367Abef30Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef30,
              (_MM_PERM_ENUM)221); // B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7)
                                   // B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

          const __m512i cRhsMat014589Cd31Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd31,
              (_MM_PERM_ENUM)221); // B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15)
                                   // B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15)
                                   // B3C(12-15) B3D(12-15)
          const __m512i cRhsMat2367Abef31Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef31,
              (_MM_PERM_ENUM)221); // B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15)
                                   // B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15)
                                   // B3E(12-15) B3F(12-15)

          const __m512i cRhsMat014589Cd40Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd40,
              (_MM_PERM_ENUM)221); // B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7)
                                   // B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
          const __m512i cRhsMat2367Abef40Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef40,
              (_MM_PERM_ENUM)221); // B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7)
                                   // B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

          const __m512i cRhsMat014589Cd41Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd41,
              (_MM_PERM_ENUM)221); // B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15)
                                   // B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15)
                                   // B4C(12-15) B4D(12-15)
          const __m512i cRhsMat2367Abef41Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef41,
              (_MM_PERM_ENUM)221); // B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15)
                                   // B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15)
                                   // B4E(12-15) B4F(12-15)

          const __m512i cRhsMat014589Cd50Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd50,
              (_MM_PERM_ENUM)221); // B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7)
                                   // B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
          const __m512i cRhsMat2367Abef50Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef50,
              (_MM_PERM_ENUM)221); // B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7)
                                   // B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

          const __m512i cRhsMat014589Cd51Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd51,
              (_MM_PERM_ENUM)221); // B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15)
                                   // B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15)
                                   // B5C(12-15) B5D(12-15)
          const __m512i cRhsMat2367Abef51Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef51,
              (_MM_PERM_ENUM)221); // B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15)
                                   // B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15)
                                   // B5E(12-15) B5F(12-15)

          const __m512i cRhsMat014589Cd60Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd60,
              (_MM_PERM_ENUM)221); // B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7)
                                   // B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
          const __m512i cRhsMat2367Abef60Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef60,
              (_MM_PERM_ENUM)221); // B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7)
                                   // B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

          const __m512i cRhsMat014589Cd61Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd61,
              (_MM_PERM_ENUM)221); // B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15)
                                   // B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15)
                                   // B6C(12-15) B6D(12-15)
          const __m512i cRhsMat2367Abef61Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef61,
              (_MM_PERM_ENUM)221); // B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15)
                                   // B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15)
                                   // B6E(12-15) B6F(12-15)

          const __m512i cRhsMat014589Cd70Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd70,
              (_MM_PERM_ENUM)221); // B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7)
                                   // B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
          const __m512i cRhsMat2367Abef70Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef70,
              (_MM_PERM_ENUM)221); // B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7)
                                   // B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

          const __m512i cRhsMat014589Cd71Sp2 = _mm512_shuffle_epi32(
              cRhsMat014589Cd71,
              (_MM_PERM_ENUM)221); // B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15)
                                   // B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15)
                                   // B7C(12-15) B7D(12-15)
          const __m512i cRhsMat2367Abef71Sp2 = _mm512_shuffle_epi32(
              cRhsMat2367Abef71,
              (_MM_PERM_ENUM)221); // B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15)
                                   // B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15)
                                   // B7E(12-15) B7F(12-15)

          // notation:superblock subblock
          // s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51
          // m51  s60 m60  s61 m61   s70 m70  s71 m71

          const __m128i cMinsAndScales010 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + vSb * 64));
          const __m128i cMinsAndScales230 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales450 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales670 = _mm_loadu_si128((const __m128i *)(vBPtr0[vB].scales + 48 + vSb * 64));

          const __m128i cMinsAndScales011 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + vSb * 64));
          const __m128i cMinsAndScales231 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales451 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales671 = _mm_loadu_si128((const __m128i *)(vBPtr1[vB].scales + 48 + vSb * 64));

          // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
          const __m256i cMinsAndScales01 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales010), cMinsAndScales011, 1);
          const __m256i cMinsAndScales23 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales230), cMinsAndScales231, 1);
          const __m256i cMinsAndScales45 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales450), cMinsAndScales451, 1);
          const __m256i cMinsAndScales67 =
              _mm256_insertf128_si256(_mm256_castsi128_si256(cMinsAndScales670), cMinsAndScales671, 1);

          // Extract scales which is lower half from mins_and_scales
          const __m256i cScales01 = _mm256_and_si256(cMinsAndScales01, cM4b);
          const __m256i cScales23 = _mm256_and_si256(cMinsAndScales23, cM4b);
          const __m256i cScales45 = _mm256_and_si256(cMinsAndScales45, cM4b);
          const __m256i cScales67 = _mm256_and_si256(cMinsAndScales67, cM4b);

          // Extract mins which is upper half from mins_and_scales
          const __m512i cMins01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales01, 4), cM4b));
          const __m512i cMins23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales23, 4), cM4b));
          const __m512i cMins45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales45, 4), cM4b));
          const __m512i cMins67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cMinsAndScales67, 4), cM4b));

          const __m512i cScales0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales01, vScalesmask1));
          const __m512i cScales1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales01, vScalesmask2));
          const __m512i cScales2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales23, vScalesmask1));
          const __m512i cScales3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales23, vScalesmask2));
          const __m512i cScales4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales45, vScalesmask1));
          const __m512i cScales5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales45, vScalesmask2));
          const __m512i cScales6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales67, vScalesmask1));
          const __m512i cScales7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(cScales67, vScalesmask2));

          const __m512i cScale014589Cd0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef0 = _mm512_shuffle_epi32(cScales0, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef1 = _mm512_shuffle_epi32(cScales1, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd2 = _mm512_shuffle_epi32(cScales2, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef2 = _mm512_shuffle_epi32(cScales2, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd3 = _mm512_shuffle_epi32(cScales3, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef3 = _mm512_shuffle_epi32(cScales3, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd4 = _mm512_shuffle_epi32(cScales4, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef4 = _mm512_shuffle_epi32(cScales4, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd5 = _mm512_shuffle_epi32(cScales5, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef5 = _mm512_shuffle_epi32(cScales5, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd6 = _mm512_shuffle_epi32(cScales6, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef6 = _mm512_shuffle_epi32(cScales6, (_MM_PERM_ENUM)238);

          const __m512i cScale014589Cd7 = _mm512_shuffle_epi32(cScales7, (_MM_PERM_ENUM)68);
          const __m512i cScale2367Abef7 = _mm512_shuffle_epi32(cScales7, (_MM_PERM_ENUM)238);

          // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
          // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
          __m256i vLhsMatYmm012300 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 512 * vSb)));
          __m256i vLhsMatYmm0100 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 0);
          __m256i vLhsMatYmm2300 = _mm256_permute2f128_si256(vLhsMatYmm012300, vLhsMatYmm012300, 17);
          __m256i vLhsMatYmm012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32 + 512 * vSb)));
          __m256i vLhsMatYmm0101 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 0);
          __m256i vLhsMatYmm2301 = _mm256_permute2f128_si256(vLhsMatYmm012301, vLhsMatYmm012301, 17);
          __m256i vLhsMatYmm012310 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64 + 512 * vSb)));
          __m256i vLhsMatYmm0110 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 0);
          __m256i vLhsMatYmm2310 = _mm256_permute2f128_si256(vLhsMatYmm012310, vLhsMatYmm012310, 17);
          __m256i vLhsMatYmm012311 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96 + 512 * vSb)));
          __m256i vLhsMatYmm0111 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 0);
          __m256i vLhsMatYmm2311 = _mm256_permute2f128_si256(vLhsMatYmm012311, vLhsMatYmm012311, 17);
          __m256i vLhsMatYmm012320 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 128 + 512 * vSb)));
          __m256i vLhsMatYmm0120 = _mm256_permute2f128_si256(vLhsMatYmm012320, vLhsMatYmm012320, 0);
          __m256i vLhsMatYmm2320 = _mm256_permute2f128_si256(vLhsMatYmm012320, vLhsMatYmm012320, 17);
          __m256i vLhsMatYmm012321 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 160 + 512 * vSb)));
          __m256i vLhsMatYmm0121 = _mm256_permute2f128_si256(vLhsMatYmm012321, vLhsMatYmm012321, 0);
          __m256i vLhsMatYmm2321 = _mm256_permute2f128_si256(vLhsMatYmm012321, vLhsMatYmm012321, 17);
          __m256i vLhsMatYmm012330 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 192 + 512 * vSb)));
          __m256i vLhsMatYmm0130 = _mm256_permute2f128_si256(vLhsMatYmm012330, vLhsMatYmm012330, 0);
          __m256i vLhsMatYmm2330 = _mm256_permute2f128_si256(vLhsMatYmm012330, vLhsMatYmm012330, 17);
          __m256i vLhsMatYmm012331 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 224 + 512 * vSb)));
          __m256i vLhsMatYmm0131 = _mm256_permute2f128_si256(vLhsMatYmm012331, vLhsMatYmm012331, 0);
          __m256i vLhsMatYmm2331 = _mm256_permute2f128_si256(vLhsMatYmm012331, vLhsMatYmm012331, 17);

          __m256i vLhsMatYmm012340 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 256 + 512 * vSb)));
          __m256i vLhsMatYmm0140 = _mm256_permute2f128_si256(vLhsMatYmm012340, vLhsMatYmm012340, 0);
          __m256i vLhsMatYmm2340 = _mm256_permute2f128_si256(vLhsMatYmm012340, vLhsMatYmm012340, 17);
          __m256i vLhsMatYmm012341 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 288 + 512 * vSb)));
          __m256i vLhsMatYmm0141 = _mm256_permute2f128_si256(vLhsMatYmm012341, vLhsMatYmm012341, 0);
          __m256i vLhsMatYmm2341 = _mm256_permute2f128_si256(vLhsMatYmm012341, vLhsMatYmm012341, 17);
          __m256i vLhsMatYmm012350 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 320 + 512 * vSb)));
          __m256i vLhsMatYmm0150 = _mm256_permute2f128_si256(vLhsMatYmm012350, vLhsMatYmm012350, 0);
          __m256i vLhsMatYmm2350 = _mm256_permute2f128_si256(vLhsMatYmm012350, vLhsMatYmm012350, 17);
          __m256i vLhsMatYmm012351 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 352 + 512 * vSb)));
          __m256i vLhsMatYmm0151 = _mm256_permute2f128_si256(vLhsMatYmm012351, vLhsMatYmm012351, 0);
          __m256i vLhsMatYmm2351 = _mm256_permute2f128_si256(vLhsMatYmm012351, vLhsMatYmm012351, 17);
          __m256i vLhsMatYmm012360 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 384 + 512 * vSb)));
          __m256i vLhsMatYmm0160 = _mm256_permute2f128_si256(vLhsMatYmm012360, vLhsMatYmm012360, 0);
          __m256i vLhsMatYmm2360 = _mm256_permute2f128_si256(vLhsMatYmm012360, vLhsMatYmm012360, 17);
          __m256i vLhsMatYmm012361 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 416 + 512 * vSb)));
          __m256i vLhsMatYmm0161 = _mm256_permute2f128_si256(vLhsMatYmm012361, vLhsMatYmm012361, 0);
          __m256i vLhsMatYmm2361 = _mm256_permute2f128_si256(vLhsMatYmm012361, vLhsMatYmm012361, 17);
          __m256i vLhsMatYmm012370 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 448 + 512 * vSb)));
          __m256i vLhsMatYmm0170 = _mm256_permute2f128_si256(vLhsMatYmm012370, vLhsMatYmm012370, 0);
          __m256i vLhsMatYmm2370 = _mm256_permute2f128_si256(vLhsMatYmm012370, vLhsMatYmm012370, 17);
          __m256i vLhsMatYmm012371 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 480 + 512 * vSb)));
          __m256i vLhsMatYmm0171 = _mm256_permute2f128_si256(vLhsMatYmm012371, vLhsMatYmm012371, 0);
          __m256i vLhsMatYmm2371 = _mm256_permute2f128_si256(vLhsMatYmm012371, vLhsMatYmm012371, 17);

          __m512i vLhsMat0100 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0100), vLhsMatYmm0100, 1);
          __m512i vLhsMat2300 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2300), vLhsMatYmm2300, 1);
          __m512i vLhsMat0101 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0101), vLhsMatYmm0101, 1);
          __m512i vLhsMat2301 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2301), vLhsMatYmm2301, 1);

          __m512i vLhsMat0110 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0110), vLhsMatYmm0110, 1);
          __m512i vLhsMat2310 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2310), vLhsMatYmm2310, 1);
          __m512i vLhsMat0111 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0111), vLhsMatYmm0111, 1);
          __m512i vLhsMat2311 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2311), vLhsMatYmm2311, 1);

          __m512i vLhsMat0120 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0120), vLhsMatYmm0120, 1);
          __m512i vLhsMat2320 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2320), vLhsMatYmm2320, 1);
          __m512i vLhsMat0121 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0121), vLhsMatYmm0121, 1);
          __m512i vLhsMat2321 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2321), vLhsMatYmm2321, 1);

          __m512i vLhsMat0130 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0130), vLhsMatYmm0130, 1);
          __m512i vLhsMat2330 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2330), vLhsMatYmm2330, 1);
          __m512i vLhsMat0131 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0131), vLhsMatYmm0131, 1);
          __m512i vLhsMat2331 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2331), vLhsMatYmm2331, 1);

          __m512i vLhsMat0140 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0140), vLhsMatYmm0140, 1);
          __m512i vLhsMat2340 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2340), vLhsMatYmm2340, 1);
          __m512i vLhsMat0141 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0141), vLhsMatYmm0141, 1);
          __m512i vLhsMat2341 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2341), vLhsMatYmm2341, 1);

          __m512i vLhsMat0150 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0150), vLhsMatYmm0150, 1);
          __m512i vLhsMat2350 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2350), vLhsMatYmm2350, 1);
          __m512i vLhsMat0151 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0151), vLhsMatYmm0151, 1);
          __m512i vLhsMat2351 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2351), vLhsMatYmm2351, 1);

          __m512i vLhsMat0160 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0160), vLhsMatYmm0160, 1);
          __m512i vLhsMat2360 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2360), vLhsMatYmm2360, 1);
          __m512i vLhsMat0161 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0161), vLhsMatYmm0161, 1);
          __m512i vLhsMat2361 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2361), vLhsMatYmm2361, 1);

          __m512i vLhsMat0170 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0170), vLhsMatYmm0170, 1);
          __m512i vLhsMat2370 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2370), vLhsMatYmm2370, 1);
          __m512i vLhsMat0171 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm0171), vLhsMatYmm0171, 1);
          __m512i vLhsMat2371 = _mm512_inserti32x8(_mm512_castsi256_si512(vLhsMatYmm2371), vLhsMatYmm2371, 1);

          // Bsums are loaded for the different Q8_K blocks
          __m128i vLhsRawBsums010123 = _mm_loadu_si128((const __m128i *)((vAPtr[vB].bsums + 32 * vSb)));
          __m128i vLhsRawBsums230123 = _mm_loadu_si128((const __m128i *)(vAPtr[vB].bsums + 8 + 32 * vSb));
          __m128i vLhsRawBsums014567 = _mm_loadu_si128((const __m128i *)((vAPtr[vB].bsums + 16 + 32 * vSb)));
          __m128i vLhsRawBsums234567 = _mm_loadu_si128((const __m128i *)(vAPtr[vB].bsums + 24 + 32 * vSb));

          __m256i vLhsBsumsYmm010123 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums010123), vLhsRawBsums010123, 1);
          __m512i vLhsBsums010123 =
              _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm010123), vLhsBsumsYmm010123, 1);
          __m256i vLhsBsumsYmm230123 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums230123), vLhsRawBsums230123, 1);
          __m512i vLhsBsums230123 =
              _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm230123), vLhsBsumsYmm230123, 1);
          __m256i vLhsBsumsYmm014567 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums014567), vLhsRawBsums014567, 1);
          __m512i vLhsBsums014567 =
              _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm014567), vLhsBsumsYmm014567, 1);
          __m256i vLhsBsumsYmm234567 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums234567), vLhsRawBsums234567, 1);
          __m512i vLhsBsums234567 =
              _mm512_inserti32x8(_mm512_castsi256_si512(vLhsBsumsYmm234567), vLhsBsumsYmm234567, 1);

          // Shuffle pattern one - left side input
          const __m512i cLhsMat0100Sp1 = _mm512_shuffle_epi32(
              vLhsMat0100,
              (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                   // A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
          const __m512i cLhsMat2300Sp1 = _mm512_shuffle_epi32(
              vLhsMat2300,
              (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3)
                                   // A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

          const __m512i cLhsMat0101Sp1 = _mm512_shuffle_epi32(
              vLhsMat0101,
              (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m512i cLhsMat2301Sp1 = _mm512_shuffle_epi32(
              vLhsMat2301,
              (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                   // A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

          const __m512i cLhsMat0110Sp1 = _mm512_shuffle_epi32(
              vLhsMat0110,
              (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                   // A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
          const __m512i cLhsMat2310Sp1 = _mm512_shuffle_epi32(
              vLhsMat2310,
              (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3)
                                   // A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

          const __m512i cLhsMat0111Sp1 = _mm512_shuffle_epi32(
              vLhsMat0111,
              (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m512i cLhsMat2311Sp1 = _mm512_shuffle_epi32(
              vLhsMat2311,
              (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                   // A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

          const __m512i cLhsMat0120Sp1 = _mm512_shuffle_epi32(
              vLhsMat0120,
              (_MM_PERM_ENUM)160); // A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3)
                                   // A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
          const __m512i cLhsMat2320Sp1 = _mm512_shuffle_epi32(
              vLhsMat2320,
              (_MM_PERM_ENUM)160); // A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3)
                                   // A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

          const __m512i cLhsMat0121Sp1 = _mm512_shuffle_epi32(
              vLhsMat0121,
              (_MM_PERM_ENUM)160); // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                                   // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
          const __m512i cLhsMat2321Sp1 = _mm512_shuffle_epi32(
              vLhsMat2321,
              (_MM_PERM_ENUM)160); // A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)
                                   // A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

          const __m512i cLhsMat0130Sp1 = _mm512_shuffle_epi32(
              vLhsMat0130,
              (_MM_PERM_ENUM)160); // A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3)
                                   // A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
          const __m512i cLhsMat2330Sp1 = _mm512_shuffle_epi32(
              vLhsMat2330,
              (_MM_PERM_ENUM)160); // A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3)
                                   // A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

          const __m512i cLhsMat0131Sp1 = _mm512_shuffle_epi32(
              vLhsMat0131,
              (_MM_PERM_ENUM)160); // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                                   // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
          const __m512i cLhsMat2331Sp1 = _mm512_shuffle_epi32(
              vLhsMat2331,
              (_MM_PERM_ENUM)160); // A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)
                                   // A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

          const __m512i cLhsMat0140Sp1 = _mm512_shuffle_epi32(
              vLhsMat0140,
              (_MM_PERM_ENUM)160); // A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3)
                                   // A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
          const __m512i cLhsMat2340Sp1 = _mm512_shuffle_epi32(
              vLhsMat2340,
              (_MM_PERM_ENUM)160); // A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3)
                                   // A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

          const __m512i cLhsMat0141Sp1 = _mm512_shuffle_epi32(
              vLhsMat0141,
              (_MM_PERM_ENUM)160); // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                                   // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
          const __m512i cLhsMat2341Sp1 = _mm512_shuffle_epi32(
              vLhsMat2341,
              (_MM_PERM_ENUM)160); // A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)
                                   // A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

          const __m512i cLhsMat0150Sp1 = _mm512_shuffle_epi32(
              vLhsMat0150,
              (_MM_PERM_ENUM)160); // A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3)
                                   // A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
          const __m512i cLhsMat2350Sp1 = _mm512_shuffle_epi32(
              vLhsMat2350,
              (_MM_PERM_ENUM)160); // A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3)
                                   // A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

          const __m512i cLhsMat0151Sp1 = _mm512_shuffle_epi32(
              vLhsMat0151,
              (_MM_PERM_ENUM)160); // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                                   // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
          const __m512i cLhsMat2351Sp1 = _mm512_shuffle_epi32(
              vLhsMat2351,
              (_MM_PERM_ENUM)160); // A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)
                                   // A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

          const __m512i cLhsMat0160Sp1 = _mm512_shuffle_epi32(
              vLhsMat0160,
              (_MM_PERM_ENUM)160); // A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3)
                                   // A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
          const __m512i cLhsMat2360Sp1 = _mm512_shuffle_epi32(
              vLhsMat2360,
              (_MM_PERM_ENUM)160); // A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3)
                                   // A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

          const __m512i cLhsMat0161Sp1 = _mm512_shuffle_epi32(
              vLhsMat0161,
              (_MM_PERM_ENUM)160); // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                                   // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
          const __m512i cLhsMat2361Sp1 = _mm512_shuffle_epi32(
              vLhsMat2361,
              (_MM_PERM_ENUM)160); // A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)
                                   // A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

          const __m512i cLhsMat0170Sp1 = _mm512_shuffle_epi32(
              vLhsMat0170,
              (_MM_PERM_ENUM)160); // A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3)
                                   // A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
          const __m512i cLhsMat2370Sp1 = _mm512_shuffle_epi32(
              vLhsMat2370,
              (_MM_PERM_ENUM)160); // A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3)
                                   // A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

          const __m512i cLhsMat0171Sp1 = _mm512_shuffle_epi32(
              vLhsMat0171,
              (_MM_PERM_ENUM)160); // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                                   // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
          const __m512i cLhsMat2371Sp1 = _mm512_shuffle_epi32(
              vLhsMat2371,
              (_MM_PERM_ENUM)160); // A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)
                                   // A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

          const __m512i cLhsMat0100Sp2 = _mm512_shuffle_epi32(
              vLhsMat0100,
              (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                   // A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
          const __m512i cLhsMat2300Sp2 = _mm512_shuffle_epi32(
              vLhsMat2300,
              (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7)
                                   // A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

          const __m512i cLhsMat0101Sp2 = _mm512_shuffle_epi32(
              vLhsMat0101, (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15)
                                                // A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                                // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m512i cLhsMat2301Sp2 = _mm512_shuffle_epi32(
              vLhsMat2301, (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15)
                                                // A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                                // A02(12-15) A02(12-15) A03(12-15) A03(12-15)

          const __m512i cLhsMat0110Sp2 = _mm512_shuffle_epi32(
              vLhsMat0110,
              (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                   // A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
          const __m512i cLhsMat2310Sp2 = _mm512_shuffle_epi32(
              vLhsMat2310,
              (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7)
                                   // A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

          const __m512i cLhsMat0111Sp2 = _mm512_shuffle_epi32(
              vLhsMat0111, (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15)
                                                // A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                                // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m512i cLhsMat2311Sp2 = _mm512_shuffle_epi32(
              vLhsMat2311, (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15)
                                                // A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                                // A12(12-15) A12(12-15) A13(12-15) A13(12-15)

          const __m512i cLhsMat0120Sp2 = _mm512_shuffle_epi32(
              vLhsMat0120,
              (_MM_PERM_ENUM)245); // A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7)
                                   // A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
          const __m512i cLhsMat2320Sp2 = _mm512_shuffle_epi32(
              vLhsMat2320,
              (_MM_PERM_ENUM)245); // A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7)
                                   // A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

          const __m512i cLhsMat0121Sp2 = _mm512_shuffle_epi32(
              vLhsMat0121, (_MM_PERM_ENUM)245); // A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15)
                                                // A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                                                // A20(12-15) A20(12-15) A21(12-15) A21(12-15)
          const __m512i cLhsMat2321Sp2 = _mm512_shuffle_epi32(
              vLhsMat2321, (_MM_PERM_ENUM)245); // A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15)
                                                // A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)
                                                // A22(12-15) A22(12-15) A23(12-15) A23(12-15)

          const __m512i cLhsMat0130Sp2 = _mm512_shuffle_epi32(
              vLhsMat0130,
              (_MM_PERM_ENUM)245); // A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7)
                                   // A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
          const __m512i cLhsMat2330Sp2 = _mm512_shuffle_epi32(
              vLhsMat2330,
              (_MM_PERM_ENUM)245); // A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7)
                                   // A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

          const __m512i cLhsMat0131Sp2 = _mm512_shuffle_epi32(
              vLhsMat0131, (_MM_PERM_ENUM)245); // A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15)
                                                // A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                                                // A30(12-15) A30(12-15) A31(12-15) A31(12-15)
          const __m512i cLhsMat2331Sp2 = _mm512_shuffle_epi32(
              vLhsMat2331, (_MM_PERM_ENUM)245); // A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15)
                                                // A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)
                                                // A32(12-15) A32(12-15) A33(12-15) A33(12-15)

          const __m512i cLhsMat0140Sp2 = _mm512_shuffle_epi32(
              vLhsMat0140,
              (_MM_PERM_ENUM)245); // A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7)
                                   // A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
          const __m512i cLhsMat2340Sp2 = _mm512_shuffle_epi32(
              vLhsMat2340,
              (_MM_PERM_ENUM)245); // A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7)
                                   // A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

          const __m512i cLhsMat0141Sp2 = _mm512_shuffle_epi32(
              vLhsMat0141, (_MM_PERM_ENUM)245); // A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15)
                                                // A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                                                // A40(12-15) A40(12-15) A41(12-15) A41(12-15)
          const __m512i cLhsMat2341Sp2 = _mm512_shuffle_epi32(
              vLhsMat2341, (_MM_PERM_ENUM)245); // A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15)
                                                // A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)
                                                // A42(12-15) A42(12-15) A43(12-15) A43(12-15)

          const __m512i cLhsMat0150Sp2 = _mm512_shuffle_epi32(
              vLhsMat0150,
              (_MM_PERM_ENUM)245); // A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7)
                                   // A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
          const __m512i cLhsMat2350Sp2 = _mm512_shuffle_epi32(
              vLhsMat2350,
              (_MM_PERM_ENUM)245); // A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7)
                                   // A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

          const __m512i cLhsMat0151Sp2 = _mm512_shuffle_epi32(
              vLhsMat0151, (_MM_PERM_ENUM)245); // A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15)
                                                // A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                                                // A50(12-15) A50(12-15) A51(12-15) A51(12-15)
          const __m512i cLhsMat2351Sp2 = _mm512_shuffle_epi32(
              vLhsMat2351, (_MM_PERM_ENUM)245); // A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15)
                                                // A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)
                                                // A52(12-15) A52(12-15) A53(12-15) A53(12-15)

          const __m512i cLhsMat0160Sp2 = _mm512_shuffle_epi32(
              vLhsMat0160,
              (_MM_PERM_ENUM)245); // A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7)
                                   // A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
          const __m512i cLhsMat2360Sp2 = _mm512_shuffle_epi32(
              vLhsMat2360,
              (_MM_PERM_ENUM)245); // A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7)
                                   // A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

          const __m512i cLhsMat0161Sp2 = _mm512_shuffle_epi32(
              vLhsMat0161, (_MM_PERM_ENUM)245); // A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15)
                                                // A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                                                // A60(12-15) A60(12-15) A61(12-15) A61(12-15)
          const __m512i cLhsMat2361Sp2 = _mm512_shuffle_epi32(
              vLhsMat2361, (_MM_PERM_ENUM)245); // A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15)
                                                // A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)
                                                // A62(12-15) A62(12-15) A63(12-15) A63(12-15)

          const __m512i cLhsMat0170Sp2 = _mm512_shuffle_epi32(
              vLhsMat0170,
              (_MM_PERM_ENUM)245); // A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7)
                                   // A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
          const __m512i cLhsMat2370Sp2 = _mm512_shuffle_epi32(
              vLhsMat2370,
              (_MM_PERM_ENUM)245); // A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7)
                                   // A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

          const __m512i cLhsMat0171Sp2 = _mm512_shuffle_epi32(
              vLhsMat0171, (_MM_PERM_ENUM)245); // A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15)
                                                // A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                                                // A70(12-15) A70(12-15) A71(12-15) A71(12-15)
          const __m512i cLhsMat2371Sp2 = _mm512_shuffle_epi32(
              vLhsMat2371, (_MM_PERM_ENUM)245); // A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15)
                                                // A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)
                                                // A72(12-15) A72(12-15) A73(12-15) A73(12-15)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
          __m512i vIaccMat000Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat0100Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat0101Sp1));
          __m512i vIaccMat010Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat0100Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat0101Sp1));

          __m512i vIaccMat100Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp1, cLhsMat2300Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd01Sp1, cLhsMat2301Sp1));
          __m512i vIaccMat110Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp1, cLhsMat2300Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef01Sp1, cLhsMat2301Sp1));

          __m512i vIaccMat001Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat0110Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat0111Sp1));
          __m512i vIaccMat011Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat0110Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat0111Sp1));

          __m512i vIaccMat101Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp1, cLhsMat2310Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd11Sp1, cLhsMat2311Sp1));
          __m512i vIaccMat111Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp1, cLhsMat2310Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef11Sp1, cLhsMat2311Sp1));

          __m512i vIaccMat002Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp1, cLhsMat0120Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd21Sp1, cLhsMat0121Sp1));
          __m512i vIaccMat012Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp1, cLhsMat0120Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef21Sp1, cLhsMat0121Sp1));

          __m512i vIaccMat102Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp1, cLhsMat2320Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd21Sp1, cLhsMat2321Sp1));
          __m512i vIaccMat112Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp1, cLhsMat2320Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef21Sp1, cLhsMat2321Sp1));

          __m512i vIaccMat003Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp1, cLhsMat0130Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd31Sp1, cLhsMat0131Sp1));
          __m512i vIaccMat013Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp1, cLhsMat0130Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef31Sp1, cLhsMat0131Sp1));

          __m512i vIaccMat103Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp1, cLhsMat2330Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd31Sp1, cLhsMat2331Sp1));
          __m512i vIaccMat113Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp1, cLhsMat2330Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef31Sp1, cLhsMat2331Sp1));

          __m512i vIaccMat004Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp1, cLhsMat0140Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd41Sp1, cLhsMat0141Sp1));
          __m512i vIaccMat014Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp1, cLhsMat0140Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef41Sp1, cLhsMat0141Sp1));

          __m512i vIaccMat104Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp1, cLhsMat2340Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd41Sp1, cLhsMat2341Sp1));
          __m512i vIaccMat114Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp1, cLhsMat2340Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef41Sp1, cLhsMat2341Sp1));

          __m512i vIaccMat005Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp1, cLhsMat0150Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd51Sp1, cLhsMat0151Sp1));
          __m512i vIaccMat015Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp1, cLhsMat0150Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef51Sp1, cLhsMat0151Sp1));

          __m512i vIaccMat105Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp1, cLhsMat2350Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd51Sp1, cLhsMat2351Sp1));
          __m512i vIaccMat115Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp1, cLhsMat2350Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef51Sp1, cLhsMat2351Sp1));

          __m512i vIaccMat006Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp1, cLhsMat0160Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd61Sp1, cLhsMat0161Sp1));
          __m512i vIaccMat016Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp1, cLhsMat0160Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef61Sp1, cLhsMat0161Sp1));

          __m512i vIaccMat106Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp1, cLhsMat2360Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd61Sp1, cLhsMat2361Sp1));
          __m512i vIaccMat116Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp1, cLhsMat2360Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef61Sp1, cLhsMat2361Sp1));

          __m512i vIaccMat007Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp1, cLhsMat0170Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd71Sp1, cLhsMat0171Sp1));
          __m512i vIaccMat017Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp1, cLhsMat0170Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef71Sp1, cLhsMat0171Sp1));

          __m512i vIaccMat107Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp1, cLhsMat2370Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd71Sp1, cLhsMat2371Sp1));
          __m512i vIaccMat117Sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp1, cLhsMat2370Sp1),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef71Sp1, cLhsMat2371Sp1));

          __m512i vIaccMat000Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat0100Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat0101Sp2));
          __m512i vIaccMat010Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat0100Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat0101Sp2));

          __m512i vIaccMat100Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd00Sp2, cLhsMat2300Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd01Sp2, cLhsMat2301Sp2));
          __m512i vIaccMat110Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef00Sp2, cLhsMat2300Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef01Sp2, cLhsMat2301Sp2));

          __m512i vIaccMat001Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat0110Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat0111Sp2));
          __m512i vIaccMat011Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat0110Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat0111Sp2));

          __m512i vIaccMat101Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd10Sp2, cLhsMat2310Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd11Sp2, cLhsMat2311Sp2));
          __m512i vIaccMat111Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef10Sp2, cLhsMat2310Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef11Sp2, cLhsMat2311Sp2));

          __m512i vIaccMat002Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp2, cLhsMat0120Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd21Sp2, cLhsMat0121Sp2));
          __m512i vIaccMat012Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp2, cLhsMat0120Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef21Sp2, cLhsMat0121Sp2));

          __m512i vIaccMat102Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd20Sp2, cLhsMat2320Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd21Sp2, cLhsMat2321Sp2));
          __m512i vIaccMat112Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef20Sp2, cLhsMat2320Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef21Sp2, cLhsMat2321Sp2));

          __m512i vIaccMat003Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp2, cLhsMat0130Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd31Sp2, cLhsMat0131Sp2));
          __m512i vIaccMat013Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp2, cLhsMat0130Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef31Sp2, cLhsMat0131Sp2));

          __m512i vIaccMat103Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd30Sp2, cLhsMat2330Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd31Sp2, cLhsMat2331Sp2));
          __m512i vIaccMat113Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef30Sp2, cLhsMat2330Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef31Sp2, cLhsMat2331Sp2));

          __m512i vIaccMat004Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp2, cLhsMat0140Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd41Sp2, cLhsMat0141Sp2));
          __m512i vIaccMat014Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp2, cLhsMat0140Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef41Sp2, cLhsMat0141Sp2));

          __m512i vIaccMat104Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd40Sp2, cLhsMat2340Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd41Sp2, cLhsMat2341Sp2));
          __m512i vIaccMat114Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef40Sp2, cLhsMat2340Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef41Sp2, cLhsMat2341Sp2));

          __m512i vIaccMat005Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp2, cLhsMat0150Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd51Sp2, cLhsMat0151Sp2));
          __m512i vIaccMat015Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp2, cLhsMat0150Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef51Sp2, cLhsMat0151Sp2));

          __m512i vIaccMat105Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd50Sp2, cLhsMat2350Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd51Sp2, cLhsMat2351Sp2));
          __m512i vIaccMat115Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef50Sp2, cLhsMat2350Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef51Sp2, cLhsMat2351Sp2));

          __m512i vIaccMat006Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp2, cLhsMat0160Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd61Sp2, cLhsMat0161Sp2));
          __m512i vIaccMat016Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp2, cLhsMat0160Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef61Sp2, cLhsMat0161Sp2));

          __m512i vIaccMat106Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd60Sp2, cLhsMat2360Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd61Sp2, cLhsMat2361Sp2));
          __m512i vIaccMat116Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef60Sp2, cLhsMat2360Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef61Sp2, cLhsMat2361Sp2));

          __m512i vIaccMat007Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp2, cLhsMat0170Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd71Sp2, cLhsMat0171Sp2));
          __m512i vIaccMat017Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp2, cLhsMat0170Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef71Sp2, cLhsMat0171Sp2));

          __m512i vIaccMat107Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat014589Cd70Sp2, cLhsMat2370Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat014589Cd71Sp2, cLhsMat2371Sp2));
          __m512i vIaccMat117Sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(cRhsMat2367Abef70Sp2, cLhsMat2370Sp2),
                                                    _mm512_maddubs_epi16(cRhsMat2367Abef71Sp2, cLhsMat2371Sp2));

          // Combine results from both shuffle patterns for each output block
          __m512i vIaccMat000 = _mm512_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
          __m512i vIaccMat010 = _mm512_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
          __m512i vIaccMat100 = _mm512_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
          __m512i vIaccMat110 = _mm512_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

          __m512i vIaccMat001 = _mm512_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
          __m512i vIaccMat011 = _mm512_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
          __m512i vIaccMat101 = _mm512_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
          __m512i vIaccMat111 = _mm512_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

          __m512i vIaccMat002 = _mm512_add_epi16(vIaccMat002Sp1, vIaccMat002Sp2);
          __m512i vIaccMat012 = _mm512_add_epi16(vIaccMat012Sp1, vIaccMat012Sp2);
          __m512i vIaccMat102 = _mm512_add_epi16(vIaccMat102Sp1, vIaccMat102Sp2);
          __m512i vIaccMat112 = _mm512_add_epi16(vIaccMat112Sp1, vIaccMat112Sp2);

          __m512i vIaccMat003 = _mm512_add_epi16(vIaccMat003Sp1, vIaccMat003Sp2);
          __m512i vIaccMat013 = _mm512_add_epi16(vIaccMat013Sp1, vIaccMat013Sp2);
          __m512i vIaccMat103 = _mm512_add_epi16(vIaccMat103Sp1, vIaccMat103Sp2);
          __m512i vIaccMat113 = _mm512_add_epi16(vIaccMat113Sp1, vIaccMat113Sp2);

          __m512i vIaccMat004 = _mm512_add_epi16(vIaccMat004Sp1, vIaccMat004Sp2);
          __m512i vIaccMat014 = _mm512_add_epi16(vIaccMat014Sp1, vIaccMat014Sp2);
          __m512i vIaccMat104 = _mm512_add_epi16(vIaccMat104Sp1, vIaccMat104Sp2);
          __m512i vIaccMat114 = _mm512_add_epi16(vIaccMat114Sp1, vIaccMat114Sp2);

          __m512i vIaccMat005 = _mm512_add_epi16(vIaccMat005Sp1, vIaccMat005Sp2);
          __m512i vIaccMat015 = _mm512_add_epi16(vIaccMat015Sp1, vIaccMat015Sp2);
          __m512i vIaccMat105 = _mm512_add_epi16(vIaccMat105Sp1, vIaccMat105Sp2);
          __m512i vIaccMat115 = _mm512_add_epi16(vIaccMat115Sp1, vIaccMat115Sp2);

          __m512i vIaccMat006 = _mm512_add_epi16(vIaccMat006Sp1, vIaccMat006Sp2);
          __m512i vIaccMat016 = _mm512_add_epi16(vIaccMat016Sp1, vIaccMat016Sp2);
          __m512i vIaccMat106 = _mm512_add_epi16(vIaccMat106Sp1, vIaccMat106Sp2);
          __m512i vIaccMat116 = _mm512_add_epi16(vIaccMat116Sp1, vIaccMat116Sp2);

          __m512i vIaccMat007 = _mm512_add_epi16(vIaccMat007Sp1, vIaccMat007Sp2);
          __m512i vIaccMat017 = _mm512_add_epi16(vIaccMat017Sp1, vIaccMat017Sp2);
          __m512i vIaccMat107 = _mm512_add_epi16(vIaccMat107Sp1, vIaccMat107Sp2);
          __m512i vIaccMat117 = _mm512_add_epi16(vIaccMat117Sp1, vIaccMat117Sp2);

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          vIaccMat000 = _mm512_madd_epi16(vIaccMat000, cScale014589Cd0);
          vIaccMat010 = _mm512_madd_epi16(vIaccMat010, cScale2367Abef0);
          vIaccMat100 = _mm512_madd_epi16(vIaccMat100, cScale014589Cd0);
          vIaccMat110 = _mm512_madd_epi16(vIaccMat110, cScale2367Abef0);

          vIaccMat001 = _mm512_madd_epi16(vIaccMat001, cScale014589Cd1);
          vIaccMat011 = _mm512_madd_epi16(vIaccMat011, cScale2367Abef1);
          vIaccMat101 = _mm512_madd_epi16(vIaccMat101, cScale014589Cd1);
          vIaccMat111 = _mm512_madd_epi16(vIaccMat111, cScale2367Abef1);

          vIaccMat002 = _mm512_madd_epi16(vIaccMat002, cScale014589Cd2);
          vIaccMat012 = _mm512_madd_epi16(vIaccMat012, cScale2367Abef2);
          vIaccMat102 = _mm512_madd_epi16(vIaccMat102, cScale014589Cd2);
          vIaccMat112 = _mm512_madd_epi16(vIaccMat112, cScale2367Abef2);

          vIaccMat003 = _mm512_madd_epi16(vIaccMat003, cScale014589Cd3);
          vIaccMat013 = _mm512_madd_epi16(vIaccMat013, cScale2367Abef3);
          vIaccMat103 = _mm512_madd_epi16(vIaccMat103, cScale014589Cd3);
          vIaccMat113 = _mm512_madd_epi16(vIaccMat113, cScale2367Abef3);

          vIaccMat004 = _mm512_madd_epi16(vIaccMat004, cScale014589Cd4);
          vIaccMat014 = _mm512_madd_epi16(vIaccMat014, cScale2367Abef4);
          vIaccMat104 = _mm512_madd_epi16(vIaccMat104, cScale014589Cd4);
          vIaccMat114 = _mm512_madd_epi16(vIaccMat114, cScale2367Abef4);

          vIaccMat005 = _mm512_madd_epi16(vIaccMat005, cScale014589Cd5);
          vIaccMat015 = _mm512_madd_epi16(vIaccMat015, cScale2367Abef5);
          vIaccMat105 = _mm512_madd_epi16(vIaccMat105, cScale014589Cd5);
          vIaccMat115 = _mm512_madd_epi16(vIaccMat115, cScale2367Abef5);

          vIaccMat006 = _mm512_madd_epi16(vIaccMat006, cScale014589Cd6);
          vIaccMat016 = _mm512_madd_epi16(vIaccMat016, cScale2367Abef6);
          vIaccMat106 = _mm512_madd_epi16(vIaccMat106, cScale014589Cd6);
          vIaccMat116 = _mm512_madd_epi16(vIaccMat116, cScale2367Abef6);

          vIaccMat007 = _mm512_madd_epi16(vIaccMat007, cScale014589Cd7);
          vIaccMat017 = _mm512_madd_epi16(vIaccMat017, cScale2367Abef7);
          vIaccMat107 = _mm512_madd_epi16(vIaccMat107, cScale014589Cd7);
          vIaccMat117 = _mm512_madd_epi16(vIaccMat117, cScale2367Abef7);

          __m512i vIaccMat00 = _mm512_add_epi32(
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat000, vIaccMat001), _mm512_add_epi32(vIaccMat002, vIaccMat003)),
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat004, vIaccMat005), _mm512_add_epi32(vIaccMat006, vIaccMat007)));
          __m512i vIaccMat01 = _mm512_add_epi32(
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat010, vIaccMat011), _mm512_add_epi32(vIaccMat012, vIaccMat013)),
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat014, vIaccMat015), _mm512_add_epi32(vIaccMat016, vIaccMat017)));
          __m512i vIaccMat10 = _mm512_add_epi32(
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat100, vIaccMat101), _mm512_add_epi32(vIaccMat102, vIaccMat103)),
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat104, vIaccMat105), _mm512_add_epi32(vIaccMat106, vIaccMat107)));
          __m512i vIaccMat11 = _mm512_add_epi32(
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat110, vIaccMat111), _mm512_add_epi32(vIaccMat112, vIaccMat113)),
              _mm512_add_epi32(_mm512_add_epi32(vIaccMat114, vIaccMat115), _mm512_add_epi32(vIaccMat116, vIaccMat117)));

          // Straighten out to make 4 row vectors
          __m512i vIaccRow0 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat00, _mm512_shuffle_epi32(vIaccMat01, (_MM_PERM_ENUM)78));
          __m512i vIaccRow1 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat00, (_MM_PERM_ENUM)78), vIaccMat01);
          __m512i vIaccRow2 =
              _mm512_mask_blend_epi32(0xCCCC, vIaccMat10, _mm512_shuffle_epi32(vIaccMat11, (_MM_PERM_ENUM)78));
          __m512i vIaccRow3 =
              _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(vIaccMat10, (_MM_PERM_ENUM)78), vIaccMat11);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
          const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtr[vB].d);
          const __m256 cRowScaleF32Ymm = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);
          const __m512 cRowScaleF32 = _mm512_insertf32x8(_mm512_castps256_ps512(cRowScaleF32Ymm), cRowScaleF32Ymm, 1);

          // Multiply with appropiate scales and accumulate (for both d and dmin) below
          aAccRows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow0),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                        aAccRows[0]);
          aAccRows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow1),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                        aAccRows[1]);
          aAccRows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow2),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                        aAccRows[2]);
          aAccRows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRow3),
                                        _mm512_mul_ps(cColScaleF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                        aAccRows[3]);

          // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
          __m512i vIaccRowMin001 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)0), cMins01);
          __m512i vIaccRowMin101 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)170), cMins01);
          __m512i vIaccRowMin201 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)0), cMins01);
          __m512i vIaccRowMin301 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)170), cMins01);

          __m512i vIaccRowMin023 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)85), cMins23);
          __m512i vIaccRowMin123 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums010123, (_MM_PERM_ENUM)255), cMins23);
          __m512i vIaccRowMin223 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)85), cMins23);
          __m512i vIaccRowMin323 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums230123, (_MM_PERM_ENUM)255), cMins23);

          __m512i vIaccRowMin045 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)0), cMins45);
          __m512i vIaccRowMin145 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)170), cMins45);
          __m512i vIaccRowMin245 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)0), cMins45);
          __m512i vIaccRowMin345 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)170), cMins45);

          __m512i vIaccRowMin067 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)85), cMins67);
          __m512i vIaccRowMin167 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums014567, (_MM_PERM_ENUM)255), cMins67);
          __m512i vIaccRowMin267 = _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)85), cMins67);
          __m512i vIaccRowMin367 =
              _mm512_madd_epi16(_mm512_shuffle_epi32(vLhsBsums234567, (_MM_PERM_ENUM)255), cMins67);

          __m512i vIaccRowMin0 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin001, vIaccRowMin023),
                                                  _mm512_add_epi32(vIaccRowMin045, vIaccRowMin067));
          __m512i vIaccRowMin1 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin101, vIaccRowMin123),
                                                  _mm512_add_epi32(vIaccRowMin145, vIaccRowMin167));
          __m512i vIaccRowMin2 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin201, vIaccRowMin223),
                                                  _mm512_add_epi32(vIaccRowMin245, vIaccRowMin267));
          __m512i vIaccRowMin3 = _mm512_add_epi32(_mm512_add_epi32(vIaccRowMin301, vIaccRowMin323),
                                                  _mm512_add_epi32(vIaccRowMin345, vIaccRowMin367));

          aAccMinRows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vIaccRowMin0),
                                           _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                           aAccMinRows[0]);
          aAccMinRows[1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin1),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccMinRows[1]);
          aAccMinRows[2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin2),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccMinRows[2]);
          aAccMinRows[3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(vIaccRowMin3),
              _mm512_mul_ps(cColDminF32, _mm512_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccMinRows[3]);
        }
      }
      // Store accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm512_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm512_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }

  if (vAnc != pNc) {
    vXstart = vAnc / 8;
    vY = 0;
  }

#endif // __AVX512BW__ && __AVX512DQ__

  // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
  for (; vY < vAnr / 4; vY += 4) {

    const block_q8_Kx4 *vAPtrs[4];

    vAPtrs[0] = vAPtrStart + (vY * cNb);
    for (int vI = 0; vI < 3; ++vI) {
      vAPtrs[vI + 1] = vAPtrs[vI] + cNb;
    }

    // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {

      const block_q2_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      __m256 aAccMinRows[16];
      for (int vI = 0; vI < 16; vI++) {
        aAccMinRows[vI] = _mm256_setzero_ps();
      }

      // For super block
      for (int64_t vB = 0; vB < cNb; vB++) {
        // Delta values - Load the eight scale values of block_q2_kx8
        const __m256 cColScaleF32 = GGML_F32Cx8_LOAD(vBPtr[vB].d);

        // dmin values - Load the eight dmin values of block_q2_kx8
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 128; vSb++) {

          // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and
          // storing of values
          // superblock    sub block   which part of sub block
          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);

          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);

          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          // 2-bit -> 8-bit
          // First sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014500 = _mm256_and_si256(cRhsRawMat01450, cM3b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i cRhsMat236700 = _mm256_and_si256(cRhsRawMat23670, cM3b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i cRhsMat014501 =
              _mm256_and_si256(cRhsRawMat01451, cM3b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i cRhsMat236701 =
              _mm256_and_si256(cRhsRawMat23671, cM3b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          // Second sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014510 = _mm256_and_si256(cRhsRawMat01452, cM3b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i cRhsMat236710 = _mm256_and_si256(cRhsRawMat23672, cM3b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i cRhsMat014511 =
              _mm256_and_si256(cRhsRawMat01453, cM3b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i cRhsMat236711 =
              _mm256_and_si256(cRhsRawMat23673, cM3b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          // Third sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014520 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 2), cM3b); // B20(0-7) B21(0-7) B24(0-7) B25(0-7)
          const __m256i cRhsMat236720 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 2), cM3b); // B22(0-7) B23(0-7) B26(0-7) B27(0-7)

          const __m256i cRhsMat014521 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 2), cM3b); // B20(8-15) B21(8-15) B24(8-15) B25(8-15)
          const __m256i cRhsMat236721 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 2), cM3b); // B22(8-15) B23(8-15) B26(8-15) B27(8-15)

          // Fourth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014530 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 2), cM3b); // B30(0-7) B31(0-7) B34(0-7) B35(0-7)
          const __m256i cRhsMat236730 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 2), cM3b); // B32(0-7) B33(0-7) B36(0-7) B37(0-7)

          const __m256i cRhsMat014531 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 2), cM3b); // B30(8-15) B31(8-15) B34(8-15) B35(8-15)
          const __m256i cRhsMat236731 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 2), cM3b); // B32(8-15) B33(8-15) B36(8-15) B37(8-15)

          // Fifth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014540 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4), cM3b); // B40(0-7) B41(0-7) B44(0-7) B45(0-7)
          const __m256i cRhsMat236740 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4), cM3b); // B42(0-7) B43(0-7) B46(0-7) B47(0-7)

          const __m256i cRhsMat014541 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4), cM3b); // B40(8-15) B41(8-15) B44(8-15) B45(8-15)
          const __m256i cRhsMat236741 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4), cM3b); // B42(8-15) B43(8-15) B46(8-15) B47(8-15)

          // Sixth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014550 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 4), cM3b); // B50(0-7) B51(0-7) B54(0-7) B55(0-7)
          const __m256i cRhsMat236750 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 4), cM3b); // B52(0-7) B53(0-7) B56(0-7) B57(0-7)

          const __m256i cRhsMat014551 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 4), cM3b); // B50(8-15) B51(8-15) B54(8-15) B55(8-15)
          const __m256i cRhsMat236751 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 4), cM3b); // B52(8-15) B53(8-15) B56(8-15) B57(8-15)

          // Seventh sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014560 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 6), cM3b); // B60(0-7) B61(0-7) B64(0-7) B65(0-7)
          const __m256i cRhsMat236760 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 6), cM3b); // B62(0-7) B63(0-7) B66(0-7) B67(0-7)

          const __m256i cRhsMat014561 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 6), cM3b); // B60(8-15) B61(8-15) B64(8-15) B65(8-15)
          const __m256i cRhsMat236761 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 6), cM3b); // B62(8-15) B63(8-15) B66(8-15) B67(8-15)

          // Eighth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014570 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 6), cM3b); // B70(0-7) B71(0-7) B74(0-7) B75(0-7)
          const __m256i cRhsMat236770 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 6), cM3b); // B72(0-7) B73(0-7) B76(0-7) B77(0-7)

          const __m256i cRhsMat014571 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 6), cM3b); // B70(8-15) B71(8-15) B74(8-15) B75(8-15)
          const __m256i cRhsMat236771 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 6), cM3b); // B72(8-15) B73(8-15) B76(8-15) B77(8-15)

          // Shuffle pattern one - right side input
          const __m256i cRhsMat014500Sp1 = _mm256_shuffle_epi32(
              cRhsMat014500, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i cRhsMat236700Sp1 = _mm256_shuffle_epi32(
              cRhsMat236700, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i cRhsMat014501Sp1 = _mm256_shuffle_epi32(
              cRhsMat014501, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i cRhsMat236701Sp1 = _mm256_shuffle_epi32(
              cRhsMat236701, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i cRhsMat014510Sp1 = _mm256_shuffle_epi32(
              cRhsMat014510, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i cRhsMat236710Sp1 = _mm256_shuffle_epi32(
              cRhsMat236710, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i cRhsMat014511Sp1 = _mm256_shuffle_epi32(
              cRhsMat014511, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i cRhsMat236711Sp1 = _mm256_shuffle_epi32(
              cRhsMat236711, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i cRhsMat014520Sp1 = _mm256_shuffle_epi32(
              cRhsMat014520, 136); // B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
          const __m256i cRhsMat236720Sp1 = _mm256_shuffle_epi32(
              cRhsMat236720, 136); // B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

          const __m256i cRhsMat014521Sp1 = _mm256_shuffle_epi32(
              cRhsMat014521, 136); // B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
          const __m256i cRhsMat236721Sp1 = _mm256_shuffle_epi32(
              cRhsMat236721, 136); // B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

          const __m256i cRhsMat014530Sp1 = _mm256_shuffle_epi32(
              cRhsMat014530, 136); // B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
          const __m256i cRhsMat236730Sp1 = _mm256_shuffle_epi32(
              cRhsMat236730, 136); // B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

          const __m256i cRhsMat014531Sp1 = _mm256_shuffle_epi32(
              cRhsMat014531, 136); // B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
          const __m256i cRhsMat236731Sp1 = _mm256_shuffle_epi32(
              cRhsMat236731, 136); // B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

          const __m256i cRhsMat014540Sp1 = _mm256_shuffle_epi32(
              cRhsMat014540, 136); // B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
          const __m256i cRhsMat236740Sp1 = _mm256_shuffle_epi32(
              cRhsMat236740, 136); // B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

          const __m256i cRhsMat014541Sp1 = _mm256_shuffle_epi32(
              cRhsMat014541, 136); // B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
          const __m256i cRhsMat236741Sp1 = _mm256_shuffle_epi32(
              cRhsMat236741, 136); // B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

          const __m256i cRhsMat014550Sp1 = _mm256_shuffle_epi32(
              cRhsMat014550, 136); // B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
          const __m256i cRhsMat236750Sp1 = _mm256_shuffle_epi32(
              cRhsMat236750, 136); // B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

          const __m256i cRhsMat014551Sp1 = _mm256_shuffle_epi32(
              cRhsMat014551, 136); // B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
          const __m256i cRhsMat236751Sp1 = _mm256_shuffle_epi32(
              cRhsMat236751, 136); // B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

          const __m256i cRhsMat014560Sp1 = _mm256_shuffle_epi32(
              cRhsMat014560, 136); // B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
          const __m256i cRhsMat236760Sp1 = _mm256_shuffle_epi32(
              cRhsMat236760, 136); // B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

          const __m256i cRhsMat014561Sp1 = _mm256_shuffle_epi32(
              cRhsMat014561, 136); // B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
          const __m256i cRhsMat236761Sp1 = _mm256_shuffle_epi32(
              cRhsMat236761, 136); // B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

          const __m256i cRhsMat014570Sp1 = _mm256_shuffle_epi32(
              cRhsMat014570, 136); // B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
          const __m256i cRhsMat236770Sp1 = _mm256_shuffle_epi32(
              cRhsMat236770, 136); // B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

          const __m256i cRhsMat014571Sp1 = _mm256_shuffle_epi32(
              cRhsMat014571, 136); // B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
          const __m256i cRhsMat236771Sp1 = _mm256_shuffle_epi32(
              cRhsMat236771, 136); // B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)

          // Shuffle pattern two - right side input
          const __m256i cRhsMat014500Sp2 = _mm256_shuffle_epi32(
              cRhsMat014500, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i cRhsMat236700Sp2 = _mm256_shuffle_epi32(
              cRhsMat236700, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i cRhsMat014501Sp2 = _mm256_shuffle_epi32(
              cRhsMat014501,
              221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
          const __m256i cRhsMat236701Sp2 = _mm256_shuffle_epi32(
              cRhsMat236701,
              221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

          const __m256i cRhsMat014510Sp2 = _mm256_shuffle_epi32(
              cRhsMat014510, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i cRhsMat236710Sp2 = _mm256_shuffle_epi32(
              cRhsMat236710, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i cRhsMat014511Sp2 = _mm256_shuffle_epi32(
              cRhsMat014511,
              221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
          const __m256i cRhsMat236711Sp2 = _mm256_shuffle_epi32(
              cRhsMat236711,
              221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

          const __m256i cRhsMat014520Sp2 = _mm256_shuffle_epi32(
              cRhsMat014520, 221); // B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
          const __m256i cRhsMat236720Sp2 = _mm256_shuffle_epi32(
              cRhsMat236720, 221); // B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

          const __m256i cRhsMat014521Sp2 = _mm256_shuffle_epi32(
              cRhsMat014521,
              221); // B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
          const __m256i cRhsMat236721Sp2 = _mm256_shuffle_epi32(
              cRhsMat236721,
              221); // B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

          const __m256i cRhsMat014530Sp2 = _mm256_shuffle_epi32(
              cRhsMat014530, 221); // B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
          const __m256i cRhsMat236730Sp2 = _mm256_shuffle_epi32(
              cRhsMat236730, 221); // B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

          const __m256i cRhsMat014531Sp2 = _mm256_shuffle_epi32(
              cRhsMat014531,
              221); // B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
          const __m256i cRhsMat236731Sp2 = _mm256_shuffle_epi32(
              cRhsMat236731,
              221); // B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

          const __m256i cRhsMat014540Sp2 = _mm256_shuffle_epi32(
              cRhsMat014540, 221); // B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
          const __m256i cRhsMat236740Sp2 = _mm256_shuffle_epi32(
              cRhsMat236740, 221); // B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

          const __m256i cRhsMat014541Sp2 = _mm256_shuffle_epi32(
              cRhsMat014541,
              221); // B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
          const __m256i cRhsMat236741Sp2 = _mm256_shuffle_epi32(
              cRhsMat236741,
              221); // B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

          const __m256i cRhsMat014550Sp2 = _mm256_shuffle_epi32(
              cRhsMat014550, 221); // B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
          const __m256i cRhsMat236750Sp2 = _mm256_shuffle_epi32(
              cRhsMat236750, 221); // B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

          const __m256i cRhsMat014551Sp2 = _mm256_shuffle_epi32(
              cRhsMat014551,
              221); // B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
          const __m256i cRhsMat236751Sp2 = _mm256_shuffle_epi32(
              cRhsMat236751,
              221); // B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

          const __m256i cRhsMat014560Sp2 = _mm256_shuffle_epi32(
              cRhsMat014560, 221); // B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
          const __m256i cRhsMat236760Sp2 = _mm256_shuffle_epi32(
              cRhsMat236760, 221); // B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

          const __m256i cRhsMat014561Sp2 = _mm256_shuffle_epi32(
              cRhsMat014561,
              221); // B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
          const __m256i cRhsMat236761Sp2 = _mm256_shuffle_epi32(
              cRhsMat236761,
              221); // B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

          const __m256i cRhsMat014570Sp2 = _mm256_shuffle_epi32(
              cRhsMat014570, 221); // B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
          const __m256i cRhsMat236770Sp2 = _mm256_shuffle_epi32(
              cRhsMat236770, 221); // B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

          const __m256i cRhsMat014571Sp2 = _mm256_shuffle_epi32(
              cRhsMat014571,
              221); // B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
          const __m256i cRhsMat236771Sp2 = _mm256_shuffle_epi32(
              cRhsMat236771,
              221); // B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)

          // Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
          // s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51
          // m51  s60 m60  s61 m61   s70 m70  s71 m71

          // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
          const __m128i cMinsAndScales01 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + vSb * 64));
          const __m128i cMinsAndScales23 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales45 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales67 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 48 + vSb * 64));

          // Extract scales which is lower half from mins_and_scales
          const __m128i cScales01 = _mm_and_si128(cMinsAndScales01, cM4bSse);
          const __m128i cScales23 = _mm_and_si128(cMinsAndScales23, cM4bSse);
          const __m128i cScales45 = _mm_and_si128(cMinsAndScales45, cM4bSse);
          const __m128i cScales67 = _mm_and_si128(cMinsAndScales67, cM4bSse);

          // Extract mins which is upper half from mins_and_scales
          const __m256i cMins01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales01, 4), cM4bSse));
          const __m256i cMins23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales23, 4), cM4bSse));
          const __m256i cMins45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales45, 4), cM4bSse));
          const __m256i cMins67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales67, 4), cM4bSse));

          const __m256i cScales0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales01, vScalesmask1Sse));
          const __m256i cScales1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales01, vScalesmask2Sse));

          const __m256i cScales2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales23, vScalesmask1Sse));
          const __m256i cScales3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales23, vScalesmask2Sse));

          const __m256i cScales4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales45, vScalesmask1Sse));
          const __m256i cScales5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales45, vScalesmask2Sse));

          const __m256i cScales6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales67, vScalesmask1Sse));
          const __m256i cScales7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales67, vScalesmask2Sse));

          const __m256i cScale01450 = _mm256_shuffle_epi32(cScales0, 68);
          const __m256i cScale23670 = _mm256_shuffle_epi32(cScales0, 238);

          const __m256i cScale01451 = _mm256_shuffle_epi32(cScales1, 68);
          const __m256i cScale23671 = _mm256_shuffle_epi32(cScales1, 238);

          const __m256i cScale01452 = _mm256_shuffle_epi32(cScales2, 68);
          const __m256i cScale23672 = _mm256_shuffle_epi32(cScales2, 238);

          const __m256i cScale01453 = _mm256_shuffle_epi32(cScales3, 68);
          const __m256i cScale23673 = _mm256_shuffle_epi32(cScales3, 238);

          const __m256i cScale01454 = _mm256_shuffle_epi32(cScales4, 68);
          const __m256i cScale23674 = _mm256_shuffle_epi32(cScales4, 238);

          const __m256i cScale01455 = _mm256_shuffle_epi32(cScales5, 68);
          const __m256i cScale23675 = _mm256_shuffle_epi32(cScales5, 238);

          const __m256i cScale01456 = _mm256_shuffle_epi32(cScales6, 68);
          const __m256i cScale23676 = _mm256_shuffle_epi32(cScales6, 238);

          const __m256i cScale01457 = _mm256_shuffle_epi32(cScales7, 68);
          const __m256i cScale23677 = _mm256_shuffle_epi32(cScales7, 238);

          for (int vRp = 0; vRp < 4; vRp++) {

            // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
            // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
            __m256i vLhsMat012300 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 512 * vSb)));
            __m256i vLhsMat0100 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 0);
            __m256i vLhsMat2300 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 17);
            __m256i vLhsMat012301 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 32 + 512 * vSb)));
            __m256i vLhsMat0101 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 0);
            __m256i vLhsMat2301 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 17);
            __m256i vLhsMat012310 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 64 + 512 * vSb)));
            __m256i vLhsMat0110 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 0);
            __m256i vLhsMat2310 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 17);
            __m256i vLhsMat012311 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 96 + 512 * vSb)));
            __m256i vLhsMat0111 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 0);
            __m256i vLhsMat2311 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 17);
            __m256i vLhsMat012320 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 128 + 512 * vSb)));
            __m256i vLhsMat0120 = _mm256_permute2f128_si256(vLhsMat012320, vLhsMat012320, 0);
            __m256i vLhsMat2320 = _mm256_permute2f128_si256(vLhsMat012320, vLhsMat012320, 17);
            __m256i vLhsMat012321 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 160 + 512 * vSb)));
            __m256i vLhsMat0121 = _mm256_permute2f128_si256(vLhsMat012321, vLhsMat012321, 0);
            __m256i vLhsMat2321 = _mm256_permute2f128_si256(vLhsMat012321, vLhsMat012321, 17);
            __m256i vLhsMat012330 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 192 + 512 * vSb)));
            __m256i vLhsMat0130 = _mm256_permute2f128_si256(vLhsMat012330, vLhsMat012330, 0);
            __m256i vLhsMat2330 = _mm256_permute2f128_si256(vLhsMat012330, vLhsMat012330, 17);
            __m256i vLhsMat012331 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 224 + 512 * vSb)));
            __m256i vLhsMat0131 = _mm256_permute2f128_si256(vLhsMat012331, vLhsMat012331, 0);
            __m256i vLhsMat2331 = _mm256_permute2f128_si256(vLhsMat012331, vLhsMat012331, 17);

            __m256i vLhsMat012340 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 256 + 512 * vSb)));
            __m256i vLhsMat0140 = _mm256_permute2f128_si256(vLhsMat012340, vLhsMat012340, 0);
            __m256i vLhsMat2340 = _mm256_permute2f128_si256(vLhsMat012340, vLhsMat012340, 17);
            __m256i vLhsMat012341 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 288 + 512 * vSb)));
            __m256i vLhsMat0141 = _mm256_permute2f128_si256(vLhsMat012341, vLhsMat012341, 0);
            __m256i vLhsMat2341 = _mm256_permute2f128_si256(vLhsMat012341, vLhsMat012341, 17);
            __m256i vLhsMat012350 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 320 + 512 * vSb)));
            __m256i vLhsMat0150 = _mm256_permute2f128_si256(vLhsMat012350, vLhsMat012350, 0);
            __m256i vLhsMat2350 = _mm256_permute2f128_si256(vLhsMat012350, vLhsMat012350, 17);
            __m256i vLhsMat012351 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 352 + 512 * vSb)));
            __m256i vLhsMat0151 = _mm256_permute2f128_si256(vLhsMat012351, vLhsMat012351, 0);
            __m256i vLhsMat2351 = _mm256_permute2f128_si256(vLhsMat012351, vLhsMat012351, 17);
            __m256i vLhsMat012360 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 384 + 512 * vSb)));
            __m256i vLhsMat0160 = _mm256_permute2f128_si256(vLhsMat012360, vLhsMat012360, 0);
            __m256i vLhsMat2360 = _mm256_permute2f128_si256(vLhsMat012360, vLhsMat012360, 17);
            __m256i vLhsMat012361 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 416 + 512 * vSb)));
            __m256i vLhsMat0161 = _mm256_permute2f128_si256(vLhsMat012361, vLhsMat012361, 0);
            __m256i vLhsMat2361 = _mm256_permute2f128_si256(vLhsMat012361, vLhsMat012361, 17);
            __m256i vLhsMat012370 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 448 + 512 * vSb)));
            __m256i vLhsMat0170 = _mm256_permute2f128_si256(vLhsMat012370, vLhsMat012370, 0);
            __m256i vLhsMat2370 = _mm256_permute2f128_si256(vLhsMat012370, vLhsMat012370, 17);
            __m256i vLhsMat012371 = _mm256_loadu_si256((const __m256i *)((vAPtrs[vRp][vB].qs + 480 + 512 * vSb)));
            __m256i vLhsMat0171 = _mm256_permute2f128_si256(vLhsMat012371, vLhsMat012371, 0);
            __m256i vLhsMat2371 = _mm256_permute2f128_si256(vLhsMat012371, vLhsMat012371, 17);

            // Bsums are loaded for the different Q8_K blocks
            __m128i vLhsRawBsums010123 = _mm_loadu_si128((const __m128i *)((vAPtrs[vRp][vB].bsums + 32 * vSb)));
            __m128i vLhsRawBsums230123 = _mm_loadu_si128((const __m128i *)(vAPtrs[vRp][vB].bsums + 8 + 32 * vSb));
            __m128i vLhsRawBsums014567 = _mm_loadu_si128((const __m128i *)((vAPtrs[vRp][vB].bsums + 16 + 32 * vSb)));
            __m128i vLhsRawBsums234567 = _mm_loadu_si128((const __m128i *)(vAPtrs[vRp][vB].bsums + 24 + 32 * vSb));

            // Shuffle pattern one - left side input
            const __m256i cLhsMat0100Sp1 = _mm256_shuffle_epi32(
                vLhsMat0100, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m256i cLhsMat2300Sp1 = _mm256_shuffle_epi32(
                vLhsMat2300, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

            const __m256i cLhsMat0101Sp1 = _mm256_shuffle_epi32(
                vLhsMat0101, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m256i cLhsMat2301Sp1 = _mm256_shuffle_epi32(
                vLhsMat2301, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

            const __m256i cLhsMat0110Sp1 = _mm256_shuffle_epi32(
                vLhsMat0110, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m256i cLhsMat2310Sp1 = _mm256_shuffle_epi32(
                vLhsMat2310, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

            const __m256i cLhsMat0111Sp1 = _mm256_shuffle_epi32(
                vLhsMat0111, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m256i cLhsMat2311Sp1 = _mm256_shuffle_epi32(
                vLhsMat2311, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

            const __m256i cLhsMat0120Sp1 = _mm256_shuffle_epi32(
                vLhsMat0120, 160); // A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
            const __m256i cLhsMat2320Sp1 = _mm256_shuffle_epi32(
                vLhsMat2320, 160); // A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

            const __m256i cLhsMat0121Sp1 = _mm256_shuffle_epi32(
                vLhsMat0121, 160); // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
            const __m256i cLhsMat2321Sp1 = _mm256_shuffle_epi32(
                vLhsMat2321, 160); // A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

            const __m256i cLhsMat0130Sp1 = _mm256_shuffle_epi32(
                vLhsMat0130, 160); // A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
            const __m256i cLhsMat2330Sp1 = _mm256_shuffle_epi32(
                vLhsMat2330, 160); // A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

            const __m256i cLhsMat0131Sp1 = _mm256_shuffle_epi32(
                vLhsMat0131, 160); // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
            const __m256i cLhsMat2331Sp1 = _mm256_shuffle_epi32(
                vLhsMat2331, 160); // A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

            const __m256i cLhsMat0140Sp1 = _mm256_shuffle_epi32(
                vLhsMat0140, 160); // A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
            const __m256i cLhsMat2340Sp1 = _mm256_shuffle_epi32(
                vLhsMat2340, 160); // A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

            const __m256i cLhsMat0141Sp1 = _mm256_shuffle_epi32(
                vLhsMat0141, 160); // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
            const __m256i cLhsMat2341Sp1 = _mm256_shuffle_epi32(
                vLhsMat2341, 160); // A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

            const __m256i cLhsMat0150Sp1 = _mm256_shuffle_epi32(
                vLhsMat0150, 160); // A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
            const __m256i cLhsMat2350Sp1 = _mm256_shuffle_epi32(
                vLhsMat2350, 160); // A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

            const __m256i cLhsMat0151Sp1 = _mm256_shuffle_epi32(
                vLhsMat0151, 160); // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
            const __m256i cLhsMat2351Sp1 = _mm256_shuffle_epi32(
                vLhsMat2351, 160); // A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

            const __m256i cLhsMat0160Sp1 = _mm256_shuffle_epi32(
                vLhsMat0160, 160); // A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
            const __m256i cLhsMat2360Sp1 = _mm256_shuffle_epi32(
                vLhsMat2360, 160); // A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

            const __m256i cLhsMat0161Sp1 = _mm256_shuffle_epi32(
                vLhsMat0161, 160); // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
            const __m256i cLhsMat2361Sp1 = _mm256_shuffle_epi32(
                vLhsMat2361, 160); // A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

            const __m256i cLhsMat0170Sp1 = _mm256_shuffle_epi32(
                vLhsMat0170, 160); // A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
            const __m256i cLhsMat2370Sp1 = _mm256_shuffle_epi32(
                vLhsMat2370, 160); // A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

            const __m256i cLhsMat0171Sp1 = _mm256_shuffle_epi32(
                vLhsMat0171, 160); // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
            const __m256i cLhsMat2371Sp1 = _mm256_shuffle_epi32(
                vLhsMat2371, 160); // A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

            // Shuffle pattern two- left side input
            const __m256i cLhsMat0100Sp2 = _mm256_shuffle_epi32(
                vLhsMat0100, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m256i cLhsMat2300Sp2 = _mm256_shuffle_epi32(
                vLhsMat2300, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

            const __m256i cLhsMat0101Sp2 = _mm256_shuffle_epi32(
                vLhsMat0101,
                245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
            const __m256i cLhsMat2301Sp2 = _mm256_shuffle_epi32(
                vLhsMat2301,
                245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

            const __m256i cLhsMat0110Sp2 = _mm256_shuffle_epi32(
                vLhsMat0110, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m256i cLhsMat2310Sp2 = _mm256_shuffle_epi32(
                vLhsMat2310, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

            const __m256i cLhsMat0111Sp2 = _mm256_shuffle_epi32(
                vLhsMat0111,
                245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
            const __m256i cLhsMat2311Sp2 = _mm256_shuffle_epi32(
                vLhsMat2311,
                245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

            const __m256i cLhsMat0120Sp2 = _mm256_shuffle_epi32(
                vLhsMat0120, 245); // A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
            const __m256i cLhsMat2320Sp2 = _mm256_shuffle_epi32(
                vLhsMat2320, 245); // A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

            const __m256i cLhsMat0121Sp2 = _mm256_shuffle_epi32(
                vLhsMat0121,
                245); // A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
            const __m256i cLhsMat2321Sp2 = _mm256_shuffle_epi32(
                vLhsMat2321,
                245); // A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

            const __m256i cLhsMat0130Sp2 = _mm256_shuffle_epi32(
                vLhsMat0130, 245); // A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
            const __m256i cLhsMat2330Sp2 = _mm256_shuffle_epi32(
                vLhsMat2330, 245); // A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

            const __m256i cLhsMat0131Sp2 = _mm256_shuffle_epi32(
                vLhsMat0131,
                245); // A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
            const __m256i cLhsMat2331Sp2 = _mm256_shuffle_epi32(
                vLhsMat2331,
                245); // A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

            const __m256i cLhsMat0140Sp2 = _mm256_shuffle_epi32(
                vLhsMat0140, 245); // A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
            const __m256i cLhsMat2340Sp2 = _mm256_shuffle_epi32(
                vLhsMat2340, 245); // A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

            const __m256i cLhsMat0141Sp2 = _mm256_shuffle_epi32(
                vLhsMat0141,
                245); // A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
            const __m256i cLhsMat2341Sp2 = _mm256_shuffle_epi32(
                vLhsMat2341,
                245); // A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

            const __m256i cLhsMat0150Sp2 = _mm256_shuffle_epi32(
                vLhsMat0150, 245); // A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
            const __m256i cLhsMat2350Sp2 = _mm256_shuffle_epi32(
                vLhsMat2350, 245); // A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

            const __m256i cLhsMat0151Sp2 = _mm256_shuffle_epi32(
                vLhsMat0151,
                245); // A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
            const __m256i cLhsMat2351Sp2 = _mm256_shuffle_epi32(
                vLhsMat2351,
                245); // A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

            const __m256i cLhsMat0160Sp2 = _mm256_shuffle_epi32(
                vLhsMat0160, 245); // A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
            const __m256i cLhsMat2360Sp2 = _mm256_shuffle_epi32(
                vLhsMat2360, 245); // A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

            const __m256i cLhsMat0161Sp2 = _mm256_shuffle_epi32(
                vLhsMat0161,
                245); // A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
            const __m256i cLhsMat2361Sp2 = _mm256_shuffle_epi32(
                vLhsMat2361,
                245); // A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

            const __m256i cLhsMat0170Sp2 = _mm256_shuffle_epi32(
                vLhsMat0170, 245); // A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
            const __m256i cLhsMat2370Sp2 = _mm256_shuffle_epi32(
                vLhsMat2370, 245); // A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

            const __m256i cLhsMat0171Sp2 = _mm256_shuffle_epi32(
                vLhsMat0171,
                245); // A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
            const __m256i cLhsMat2371Sp2 = _mm256_shuffle_epi32(
                vLhsMat2371,
                245); // A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

            // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
            // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
            __m256i vIaccMat000Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat0100Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat0101Sp1));
            __m256i vIaccMat010Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat0100Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat0101Sp1));

            __m256i vIaccMat100Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat2300Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat2301Sp1));
            __m256i vIaccMat110Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat2300Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat2301Sp1));

            __m256i vIaccMat001Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat0110Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat0111Sp1));
            __m256i vIaccMat011Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat0110Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat0111Sp1));

            __m256i vIaccMat101Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat2310Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat2311Sp1));
            __m256i vIaccMat111Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat2310Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat2311Sp1));

            __m256i vIaccMat002Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp1, cLhsMat0120Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014521Sp1, cLhsMat0121Sp1));
            __m256i vIaccMat012Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp1, cLhsMat0120Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236721Sp1, cLhsMat0121Sp1));

            __m256i vIaccMat102Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp1, cLhsMat2320Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014521Sp1, cLhsMat2321Sp1));
            __m256i vIaccMat112Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp1, cLhsMat2320Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236721Sp1, cLhsMat2321Sp1));

            __m256i vIaccMat003Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp1, cLhsMat0130Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014531Sp1, cLhsMat0131Sp1));
            __m256i vIaccMat013Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp1, cLhsMat0130Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236731Sp1, cLhsMat0131Sp1));

            __m256i vIaccMat103Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp1, cLhsMat2330Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014531Sp1, cLhsMat2331Sp1));
            __m256i vIaccMat113Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp1, cLhsMat2330Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236731Sp1, cLhsMat2331Sp1));

            __m256i vIaccMat004Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp1, cLhsMat0140Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014541Sp1, cLhsMat0141Sp1));
            __m256i vIaccMat014Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp1, cLhsMat0140Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236741Sp1, cLhsMat0141Sp1));

            __m256i vIaccMat104Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp1, cLhsMat2340Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014541Sp1, cLhsMat2341Sp1));
            __m256i vIaccMat114Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp1, cLhsMat2340Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236741Sp1, cLhsMat2341Sp1));

            __m256i vIaccMat005Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp1, cLhsMat0150Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014551Sp1, cLhsMat0151Sp1));
            __m256i vIaccMat015Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp1, cLhsMat0150Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236751Sp1, cLhsMat0151Sp1));

            __m256i vIaccMat105Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp1, cLhsMat2350Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014551Sp1, cLhsMat2351Sp1));
            __m256i vIaccMat115Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp1, cLhsMat2350Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236751Sp1, cLhsMat2351Sp1));

            __m256i vIaccMat006Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp1, cLhsMat0160Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014561Sp1, cLhsMat0161Sp1));
            __m256i vIaccMat016Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp1, cLhsMat0160Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236761Sp1, cLhsMat0161Sp1));

            __m256i vIaccMat106Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp1, cLhsMat2360Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014561Sp1, cLhsMat2361Sp1));
            __m256i vIaccMat116Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp1, cLhsMat2360Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236761Sp1, cLhsMat2361Sp1));

            __m256i vIaccMat007Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp1, cLhsMat0170Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014571Sp1, cLhsMat0171Sp1));
            __m256i vIaccMat017Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp1, cLhsMat0170Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236771Sp1, cLhsMat0171Sp1));

            __m256i vIaccMat107Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp1, cLhsMat2370Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat014571Sp1, cLhsMat2371Sp1));
            __m256i vIaccMat117Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp1, cLhsMat2370Sp1),
                                                      _mm256_maddubs_epi16(cRhsMat236771Sp1, cLhsMat2371Sp1));

            __m256i vIaccMat000Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat0100Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat0101Sp2));
            __m256i vIaccMat010Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat0100Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat0101Sp2));

            __m256i vIaccMat100Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat2300Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat2301Sp2));
            __m256i vIaccMat110Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat2300Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat2301Sp2));

            __m256i vIaccMat001Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat0110Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat0111Sp2));
            __m256i vIaccMat011Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat0110Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat0111Sp2));

            __m256i vIaccMat101Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat2310Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat2311Sp2));
            __m256i vIaccMat111Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat2310Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat2311Sp2));

            __m256i vIaccMat002Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp2, cLhsMat0120Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014521Sp2, cLhsMat0121Sp2));
            __m256i vIaccMat012Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp2, cLhsMat0120Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236721Sp2, cLhsMat0121Sp2));

            __m256i vIaccMat102Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp2, cLhsMat2320Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014521Sp2, cLhsMat2321Sp2));
            __m256i vIaccMat112Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp2, cLhsMat2320Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236721Sp2, cLhsMat2321Sp2));

            __m256i vIaccMat003Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp2, cLhsMat0130Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014531Sp2, cLhsMat0131Sp2));
            __m256i vIaccMat013Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp2, cLhsMat0130Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236731Sp2, cLhsMat0131Sp2));

            __m256i vIaccMat103Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp2, cLhsMat2330Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014531Sp2, cLhsMat2331Sp2));
            __m256i vIaccMat113Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp2, cLhsMat2330Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236731Sp2, cLhsMat2331Sp2));

            __m256i vIaccMat004Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp2, cLhsMat0140Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014541Sp2, cLhsMat0141Sp2));
            __m256i vIaccMat014Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp2, cLhsMat0140Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236741Sp2, cLhsMat0141Sp2));

            __m256i vIaccMat104Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp2, cLhsMat2340Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014541Sp2, cLhsMat2341Sp2));
            __m256i vIaccMat114Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp2, cLhsMat2340Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236741Sp2, cLhsMat2341Sp2));

            __m256i vIaccMat005Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp2, cLhsMat0150Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014551Sp2, cLhsMat0151Sp2));
            __m256i vIaccMat015Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp2, cLhsMat0150Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236751Sp2, cLhsMat0151Sp2));

            __m256i vIaccMat105Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp2, cLhsMat2350Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014551Sp2, cLhsMat2351Sp2));
            __m256i vIaccMat115Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp2, cLhsMat2350Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236751Sp2, cLhsMat2351Sp2));

            __m256i vIaccMat006Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp2, cLhsMat0160Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014561Sp2, cLhsMat0161Sp2));
            __m256i vIaccMat016Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp2, cLhsMat0160Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236761Sp2, cLhsMat0161Sp2));

            __m256i vIaccMat106Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp2, cLhsMat2360Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014561Sp2, cLhsMat2361Sp2));
            __m256i vIaccMat116Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp2, cLhsMat2360Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236761Sp2, cLhsMat2361Sp2));

            __m256i vIaccMat007Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp2, cLhsMat0170Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014571Sp2, cLhsMat0171Sp2));
            __m256i vIaccMat017Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp2, cLhsMat0170Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236771Sp2, cLhsMat0171Sp2));

            __m256i vIaccMat107Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp2, cLhsMat2370Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat014571Sp2, cLhsMat2371Sp2));
            __m256i vIaccMat117Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp2, cLhsMat2370Sp2),
                                                      _mm256_maddubs_epi16(cRhsMat236771Sp2, cLhsMat2371Sp2));

            // Combine results from both shuffle patterns for each output block
            __m256i vIaccMat000 = _mm256_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
            __m256i vIaccMat010 = _mm256_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
            __m256i vIaccMat100 = _mm256_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
            __m256i vIaccMat110 = _mm256_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

            __m256i vIaccMat001 = _mm256_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
            __m256i vIaccMat011 = _mm256_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
            __m256i vIaccMat101 = _mm256_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
            __m256i vIaccMat111 = _mm256_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

            __m256i vIaccMat002 = _mm256_add_epi16(vIaccMat002Sp1, vIaccMat002Sp2);
            __m256i vIaccMat012 = _mm256_add_epi16(vIaccMat012Sp1, vIaccMat012Sp2);
            __m256i vIaccMat102 = _mm256_add_epi16(vIaccMat102Sp1, vIaccMat102Sp2);
            __m256i vIaccMat112 = _mm256_add_epi16(vIaccMat112Sp1, vIaccMat112Sp2);

            __m256i vIaccMat003 = _mm256_add_epi16(vIaccMat003Sp1, vIaccMat003Sp2);
            __m256i vIaccMat013 = _mm256_add_epi16(vIaccMat013Sp1, vIaccMat013Sp2);
            __m256i vIaccMat103 = _mm256_add_epi16(vIaccMat103Sp1, vIaccMat103Sp2);
            __m256i vIaccMat113 = _mm256_add_epi16(vIaccMat113Sp1, vIaccMat113Sp2);

            __m256i vIaccMat004 = _mm256_add_epi16(vIaccMat004Sp1, vIaccMat004Sp2);
            __m256i vIaccMat014 = _mm256_add_epi16(vIaccMat014Sp1, vIaccMat014Sp2);
            __m256i vIaccMat104 = _mm256_add_epi16(vIaccMat104Sp1, vIaccMat104Sp2);
            __m256i vIaccMat114 = _mm256_add_epi16(vIaccMat114Sp1, vIaccMat114Sp2);

            __m256i vIaccMat005 = _mm256_add_epi16(vIaccMat005Sp1, vIaccMat005Sp2);
            __m256i vIaccMat015 = _mm256_add_epi16(vIaccMat015Sp1, vIaccMat015Sp2);
            __m256i vIaccMat105 = _mm256_add_epi16(vIaccMat105Sp1, vIaccMat105Sp2);
            __m256i vIaccMat115 = _mm256_add_epi16(vIaccMat115Sp1, vIaccMat115Sp2);

            __m256i vIaccMat006 = _mm256_add_epi16(vIaccMat006Sp1, vIaccMat006Sp2);
            __m256i vIaccMat016 = _mm256_add_epi16(vIaccMat016Sp1, vIaccMat016Sp2);
            __m256i vIaccMat106 = _mm256_add_epi16(vIaccMat106Sp1, vIaccMat106Sp2);
            __m256i vIaccMat116 = _mm256_add_epi16(vIaccMat116Sp1, vIaccMat116Sp2);

            __m256i vIaccMat007 = _mm256_add_epi16(vIaccMat007Sp1, vIaccMat007Sp2);
            __m256i vIaccMat017 = _mm256_add_epi16(vIaccMat017Sp1, vIaccMat017Sp2);
            __m256i vIaccMat107 = _mm256_add_epi16(vIaccMat107Sp1, vIaccMat107Sp2);
            __m256i vIaccMat117 = _mm256_add_epi16(vIaccMat117Sp1, vIaccMat117Sp2);

            // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
            vIaccMat000 = _mm256_madd_epi16(vIaccMat000, cScale01450);
            vIaccMat010 = _mm256_madd_epi16(vIaccMat010, cScale23670);
            vIaccMat100 = _mm256_madd_epi16(vIaccMat100, cScale01450);
            vIaccMat110 = _mm256_madd_epi16(vIaccMat110, cScale23670);

            vIaccMat001 = _mm256_madd_epi16(vIaccMat001, cScale01451);
            vIaccMat011 = _mm256_madd_epi16(vIaccMat011, cScale23671);
            vIaccMat101 = _mm256_madd_epi16(vIaccMat101, cScale01451);
            vIaccMat111 = _mm256_madd_epi16(vIaccMat111, cScale23671);

            vIaccMat002 = _mm256_madd_epi16(vIaccMat002, cScale01452);
            vIaccMat012 = _mm256_madd_epi16(vIaccMat012, cScale23672);
            vIaccMat102 = _mm256_madd_epi16(vIaccMat102, cScale01452);
            vIaccMat112 = _mm256_madd_epi16(vIaccMat112, cScale23672);

            vIaccMat003 = _mm256_madd_epi16(vIaccMat003, cScale01453);
            vIaccMat013 = _mm256_madd_epi16(vIaccMat013, cScale23673);
            vIaccMat103 = _mm256_madd_epi16(vIaccMat103, cScale01453);
            vIaccMat113 = _mm256_madd_epi16(vIaccMat113, cScale23673);

            vIaccMat004 = _mm256_madd_epi16(vIaccMat004, cScale01454);
            vIaccMat014 = _mm256_madd_epi16(vIaccMat014, cScale23674);
            vIaccMat104 = _mm256_madd_epi16(vIaccMat104, cScale01454);
            vIaccMat114 = _mm256_madd_epi16(vIaccMat114, cScale23674);

            vIaccMat005 = _mm256_madd_epi16(vIaccMat005, cScale01455);
            vIaccMat015 = _mm256_madd_epi16(vIaccMat015, cScale23675);
            vIaccMat105 = _mm256_madd_epi16(vIaccMat105, cScale01455);
            vIaccMat115 = _mm256_madd_epi16(vIaccMat115, cScale23675);

            vIaccMat006 = _mm256_madd_epi16(vIaccMat006, cScale01456);
            vIaccMat016 = _mm256_madd_epi16(vIaccMat016, cScale23676);
            vIaccMat106 = _mm256_madd_epi16(vIaccMat106, cScale01456);
            vIaccMat116 = _mm256_madd_epi16(vIaccMat116, cScale23676);

            vIaccMat007 = _mm256_madd_epi16(vIaccMat007, cScale01457);
            vIaccMat017 = _mm256_madd_epi16(vIaccMat017, cScale23677);
            vIaccMat107 = _mm256_madd_epi16(vIaccMat107, cScale01457);
            vIaccMat117 = _mm256_madd_epi16(vIaccMat117, cScale23677);

            __m256i vIaccMat00 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(vIaccMat000, vIaccMat001),
                                                                   _mm256_add_epi32(vIaccMat002, vIaccMat003)),
                                                  _mm256_add_epi32(_mm256_add_epi32(vIaccMat004, vIaccMat005),
                                                                   _mm256_add_epi32(vIaccMat006, vIaccMat007)));
            __m256i vIaccMat01 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(vIaccMat010, vIaccMat011),
                                                                   _mm256_add_epi32(vIaccMat012, vIaccMat013)),
                                                  _mm256_add_epi32(_mm256_add_epi32(vIaccMat014, vIaccMat015),
                                                                   _mm256_add_epi32(vIaccMat016, vIaccMat017)));
            __m256i vIaccMat10 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(vIaccMat100, vIaccMat101),
                                                                   _mm256_add_epi32(vIaccMat102, vIaccMat103)),
                                                  _mm256_add_epi32(_mm256_add_epi32(vIaccMat104, vIaccMat105),
                                                                   _mm256_add_epi32(vIaccMat106, vIaccMat107)));
            __m256i vIaccMat11 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(vIaccMat110, vIaccMat111),
                                                                   _mm256_add_epi32(vIaccMat112, vIaccMat113)),
                                                  _mm256_add_epi32(_mm256_add_epi32(vIaccMat114, vIaccMat115),
                                                                   _mm256_add_epi32(vIaccMat116, vIaccMat117)));

            // Straighten out to make 4 row vectors
            __m256i vIaccRow0 = _mm256_blend_epi32(vIaccMat00, _mm256_shuffle_epi32(vIaccMat01, 78), 204);
            __m256i vIaccRow1 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat00, 78), vIaccMat01, 204);
            __m256i vIaccRow2 = _mm256_blend_epi32(vIaccMat10, _mm256_shuffle_epi32(vIaccMat11, 78), 204);
            __m256i vIaccRow3 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat10, 78), vIaccMat11, 204);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
            const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtrs[vRp][vB].d);
            const __m256 cRowScaleF32 = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);

            // Multiply with appropriate scales and accumulate (for both d and dmin) below
            aAccRows[vRp * 4] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow0),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccRows[vRp * 4]);
            aAccRows[vRp * 4 + 1] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow1),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccRows[vRp * 4 + 1]);
            aAccRows[vRp * 4 + 2] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow2),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccRows[vRp * 4 + 2]);
            aAccRows[vRp * 4 + 3] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRow3),
                _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccRows[vRp * 4 + 3]);

            __m256i vLhsBsums010123 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums010123), vLhsRawBsums010123, 1);
            __m256i vLhsBsums230123 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums230123), vLhsRawBsums230123, 1);
            __m256i vLhsBsums014567 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums014567), vLhsRawBsums014567, 1);
            __m256i vLhsBsums234567 =
                _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums234567), vLhsRawBsums234567, 1);

            // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
            __m256i vIaccRowMin001 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 0), cMins01);
            __m256i vIaccRowMin101 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 170), cMins01);
            __m256i vIaccRowMin201 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 0), cMins01);
            __m256i vIaccRowMin301 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 170), cMins01);

            __m256i vIaccRowMin023 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 85), cMins23);
            __m256i vIaccRowMin123 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 255), cMins23);
            __m256i vIaccRowMin223 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 85), cMins23);
            __m256i vIaccRowMin323 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 255), cMins23);

            __m256i vIaccRowMin045 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 0), cMins45);
            __m256i vIaccRowMin145 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 170), cMins45);
            __m256i vIaccRowMin245 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 0), cMins45);
            __m256i vIaccRowMin345 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 170), cMins45);

            __m256i vIaccRowMin067 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 85), cMins67);
            __m256i vIaccRowMin167 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 255), cMins67);
            __m256i vIaccRowMin267 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 85), cMins67);
            __m256i vIaccRowMin367 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 255), cMins67);

            __m256i vIaccRowMin0 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin001, vIaccRowMin023),
                                                    _mm256_add_epi32(vIaccRowMin045, vIaccRowMin067));
            __m256i vIaccRowMin1 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin101, vIaccRowMin123),
                                                    _mm256_add_epi32(vIaccRowMin145, vIaccRowMin167));
            __m256i vIaccRowMin2 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin201, vIaccRowMin223),
                                                    _mm256_add_epi32(vIaccRowMin245, vIaccRowMin267));
            __m256i vIaccRowMin3 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin301, vIaccRowMin323),
                                                    _mm256_add_epi32(vIaccRowMin345, vIaccRowMin367));

            aAccMinRows[vRp * 4] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(vIaccRowMin0),
                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)), aAccMinRows[vRp * 4]);
            aAccMinRows[vRp * 4 + 1] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin1),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                aAccMinRows[vRp * 4 + 1]);
            aAccMinRows[vRp * 4 + 2] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin2),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                aAccMinRows[vRp * 4 + 2]);
            aAccMinRows[vRp * 4 + 3] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin3),
                                _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                aAccMinRows[vRp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int vI = 0; vI < 16; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm256_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }

  for (; vY < pNr / 4; vY++) {

    const block_q8_Kx4 *vAPtr = vAPtrStart + (vY * cNb);

    // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
    for (int64_t vX = vXstart; vX < pNc / 8; vX++) {

      const block_q2_Kx8 *vBPtr = vBPtrStart + (vX * vBNb);

      // Master FP accumulators
      __m256 aAccRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccRows[vI] = _mm256_setzero_ps();
      }

      __m256 aAccMinRows[4];
      for (int vI = 0; vI < 4; vI++) {
        aAccMinRows[vI] = _mm256_setzero_ps();
      }

      for (int64_t vB = 0; vB < cNb; vB++) {
        // Delta values - Load the eight scale values of block_q2_kx8
        const __m256 cColScaleF32 = GGML_F32Cx8_LOAD(vBPtr[vB].d);

        // dmin values - Load the eight dmin values of block_q2_kx8
        const __m256 cColDminF32 = GGML_F32Cx8_LOAD(vBPtr[vB].dmin);

        // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
        for (int vSb = 0; vSb < QK_K / 128; vSb++) {

          // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of
          // eight bytes - B0,B1 ....B6,B7
          const __m256i cRhsRawMat01230 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + vSb * 256));
          const __m256i cRhsRawMat45670 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 32 + vSb * 256));
          const __m256i cRhsRawMat01231 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 64 + vSb * 256));
          const __m256i cRhsRawMat45671 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 96 + vSb * 256));
          const __m256i cRhsRawMat01232 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 128 + vSb * 256));
          const __m256i cRhsRawMat45672 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 160 + vSb * 256));
          const __m256i cRhsRawMat01233 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 192 + vSb * 256));
          const __m256i cRhsRawMat45673 = _mm256_loadu_si256((const __m256i *)(vBPtr[vB].qs + 224 + vSb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and
          // storing of values
          // superblock    sub block   which part of sub block
          const __m256i cRhsRawMat01450 =
              _mm256_blend_epi32(cRhsRawMat01230, _mm256_permutevar8x32_epi32(cRhsRawMat45670, vRequiredOrder), 240);
          const __m256i cRhsRawMat23670 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01230, vRequiredOrder), cRhsRawMat45670, 240);

          const __m256i cRhsRawMat01451 =
              _mm256_blend_epi32(cRhsRawMat01231, _mm256_permutevar8x32_epi32(cRhsRawMat45671, vRequiredOrder), 240);
          const __m256i cRhsRawMat23671 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01231, vRequiredOrder), cRhsRawMat45671, 240);

          const __m256i cRhsRawMat01452 =
              _mm256_blend_epi32(cRhsRawMat01232, _mm256_permutevar8x32_epi32(cRhsRawMat45672, vRequiredOrder), 240);
          const __m256i cRhsRawMat23672 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01232, vRequiredOrder), cRhsRawMat45672, 240);

          const __m256i cRhsRawMat01453 =
              _mm256_blend_epi32(cRhsRawMat01233, _mm256_permutevar8x32_epi32(cRhsRawMat45673, vRequiredOrder), 240);
          const __m256i cRhsRawMat23673 =
              _mm256_blend_epi32(_mm256_permutevar8x32_epi32(cRhsRawMat01233, vRequiredOrder), cRhsRawMat45673, 240);

          // 2-bit -> 8-bit
          // First sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014500 = _mm256_and_si256(cRhsRawMat01450, cM3b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i cRhsMat236700 = _mm256_and_si256(cRhsRawMat23670, cM3b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i cRhsMat014501 =
              _mm256_and_si256(cRhsRawMat01451, cM3b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i cRhsMat236701 =
              _mm256_and_si256(cRhsRawMat23671, cM3b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          // Second sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014510 = _mm256_and_si256(cRhsRawMat01452, cM3b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i cRhsMat236710 = _mm256_and_si256(cRhsRawMat23672, cM3b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i cRhsMat014511 =
              _mm256_and_si256(cRhsRawMat01453, cM3b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i cRhsMat236711 =
              _mm256_and_si256(cRhsRawMat23673, cM3b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          // Third sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014520 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 2), cM3b); // B20(0-7) B21(0-7) B24(0-7) B25(0-7)
          const __m256i cRhsMat236720 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 2), cM3b); // B22(0-7) B23(0-7) B26(0-7) B27(0-7)

          const __m256i cRhsMat014521 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 2), cM3b); // B20(8-15) B21(8-15) B24(8-15) B25(8-15)
          const __m256i cRhsMat236721 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 2), cM3b); // B22(8-15) B23(8-15) B26(8-15) B27(8-15)

          // Fourth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014530 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 2), cM3b); // B30(0-7) B31(0-7) B34(0-7) B35(0-7)
          const __m256i cRhsMat236730 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 2), cM3b); // B32(0-7) B33(0-7) B36(0-7) B37(0-7)

          const __m256i cRhsMat014531 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 2), cM3b); // B30(8-15) B31(8-15) B34(8-15) B35(8-15)
          const __m256i cRhsMat236731 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 2), cM3b); // B32(8-15) B33(8-15) B36(8-15) B37(8-15)

          // Fifth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014540 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 4), cM3b); // B40(0-7) B41(0-7) B44(0-7) B45(0-7)
          const __m256i cRhsMat236740 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 4), cM3b); // B42(0-7) B43(0-7) B46(0-7) B47(0-7)

          const __m256i cRhsMat014541 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 4), cM3b); // B40(8-15) B41(8-15) B44(8-15) B45(8-15)
          const __m256i cRhsMat236741 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 4), cM3b); // B42(8-15) B43(8-15) B46(8-15) B47(8-15)

          // Sixth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014550 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 4), cM3b); // B50(0-7) B51(0-7) B54(0-7) B55(0-7)
          const __m256i cRhsMat236750 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 4), cM3b); // B52(0-7) B53(0-7) B56(0-7) B57(0-7)

          const __m256i cRhsMat014551 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 4), cM3b); // B50(8-15) B51(8-15) B54(8-15) B55(8-15)
          const __m256i cRhsMat236751 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 4), cM3b); // B52(8-15) B53(8-15) B56(8-15) B57(8-15)

          // Seventh sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014560 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01450, 6), cM3b); // B60(0-7) B61(0-7) B64(0-7) B65(0-7)
          const __m256i cRhsMat236760 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23670, 6), cM3b); // B62(0-7) B63(0-7) B66(0-7) B67(0-7)

          const __m256i cRhsMat014561 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01451, 6), cM3b); // B60(8-15) B61(8-15) B64(8-15) B65(8-15)
          const __m256i cRhsMat236761 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23671, 6), cM3b); // B62(8-15) B63(8-15) B66(8-15) B67(8-15)

          // Eighth sub block of the eight sub blocks processed in the iteration
          const __m256i cRhsMat014570 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01452, 6), cM3b); // B70(0-7) B71(0-7) B74(0-7) B75(0-7)
          const __m256i cRhsMat236770 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23672, 6), cM3b); // B72(0-7) B73(0-7) B76(0-7) B77(0-7)

          const __m256i cRhsMat014571 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat01453, 6), cM3b); // B70(8-15) B71(8-15) B74(8-15) B75(8-15)
          const __m256i cRhsMat236771 =
              _mm256_and_si256(_mm256_srli_epi16(cRhsRawMat23673, 6), cM3b); // B72(8-15) B73(8-15) B76(8-15) B77(8-15)

          // Shuffle pattern one - right side input
          const __m256i cRhsMat014500Sp1 = _mm256_shuffle_epi32(
              cRhsMat014500, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i cRhsMat236700Sp1 = _mm256_shuffle_epi32(
              cRhsMat236700, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i cRhsMat014501Sp1 = _mm256_shuffle_epi32(
              cRhsMat014501, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i cRhsMat236701Sp1 = _mm256_shuffle_epi32(
              cRhsMat236701, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i cRhsMat014510Sp1 = _mm256_shuffle_epi32(
              cRhsMat014510, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i cRhsMat236710Sp1 = _mm256_shuffle_epi32(
              cRhsMat236710, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i cRhsMat014511Sp1 = _mm256_shuffle_epi32(
              cRhsMat014511, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i cRhsMat236711Sp1 = _mm256_shuffle_epi32(
              cRhsMat236711, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i cRhsMat014520Sp1 = _mm256_shuffle_epi32(
              cRhsMat014520, 136); // B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
          const __m256i cRhsMat236720Sp1 = _mm256_shuffle_epi32(
              cRhsMat236720, 136); // B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

          const __m256i cRhsMat014521Sp1 = _mm256_shuffle_epi32(
              cRhsMat014521, 136); // B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
          const __m256i cRhsMat236721Sp1 = _mm256_shuffle_epi32(
              cRhsMat236721, 136); // B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

          const __m256i cRhsMat014530Sp1 = _mm256_shuffle_epi32(
              cRhsMat014530, 136); // B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
          const __m256i cRhsMat236730Sp1 = _mm256_shuffle_epi32(
              cRhsMat236730, 136); // B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

          const __m256i cRhsMat014531Sp1 = _mm256_shuffle_epi32(
              cRhsMat014531, 136); // B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
          const __m256i cRhsMat236731Sp1 = _mm256_shuffle_epi32(
              cRhsMat236731, 136); // B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

          const __m256i cRhsMat014540Sp1 = _mm256_shuffle_epi32(
              cRhsMat014540, 136); // B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
          const __m256i cRhsMat236740Sp1 = _mm256_shuffle_epi32(
              cRhsMat236740, 136); // B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

          const __m256i cRhsMat014541Sp1 = _mm256_shuffle_epi32(
              cRhsMat014541, 136); // B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
          const __m256i cRhsMat236741Sp1 = _mm256_shuffle_epi32(
              cRhsMat236741, 136); // B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

          const __m256i cRhsMat014550Sp1 = _mm256_shuffle_epi32(
              cRhsMat014550, 136); // B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
          const __m256i cRhsMat236750Sp1 = _mm256_shuffle_epi32(
              cRhsMat236750, 136); // B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

          const __m256i cRhsMat014551Sp1 = _mm256_shuffle_epi32(
              cRhsMat014551, 136); // B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
          const __m256i cRhsMat236751Sp1 = _mm256_shuffle_epi32(
              cRhsMat236751, 136); // B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

          const __m256i cRhsMat014560Sp1 = _mm256_shuffle_epi32(
              cRhsMat014560, 136); // B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
          const __m256i cRhsMat236760Sp1 = _mm256_shuffle_epi32(
              cRhsMat236760, 136); // B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

          const __m256i cRhsMat014561Sp1 = _mm256_shuffle_epi32(
              cRhsMat014561, 136); // B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
          const __m256i cRhsMat236761Sp1 = _mm256_shuffle_epi32(
              cRhsMat236761, 136); // B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

          const __m256i cRhsMat014570Sp1 = _mm256_shuffle_epi32(
              cRhsMat014570, 136); // B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
          const __m256i cRhsMat236770Sp1 = _mm256_shuffle_epi32(
              cRhsMat236770, 136); // B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

          const __m256i cRhsMat014571Sp1 = _mm256_shuffle_epi32(
              cRhsMat014571, 136); // B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
          const __m256i cRhsMat236771Sp1 = _mm256_shuffle_epi32(
              cRhsMat236771, 136); // B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)

          // Shuffle pattern two - right side input
          const __m256i cRhsMat014500Sp2 = _mm256_shuffle_epi32(
              cRhsMat014500, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i cRhsMat236700Sp2 = _mm256_shuffle_epi32(
              cRhsMat236700, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i cRhsMat014501Sp2 = _mm256_shuffle_epi32(
              cRhsMat014501,
              221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
          const __m256i cRhsMat236701Sp2 = _mm256_shuffle_epi32(
              cRhsMat236701,
              221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

          const __m256i cRhsMat014510Sp2 = _mm256_shuffle_epi32(
              cRhsMat014510, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i cRhsMat236710Sp2 = _mm256_shuffle_epi32(
              cRhsMat236710, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i cRhsMat014511Sp2 = _mm256_shuffle_epi32(
              cRhsMat014511,
              221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
          const __m256i cRhsMat236711Sp2 = _mm256_shuffle_epi32(
              cRhsMat236711,
              221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

          const __m256i cRhsMat014520Sp2 = _mm256_shuffle_epi32(
              cRhsMat014520, 221); // B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
          const __m256i cRhsMat236720Sp2 = _mm256_shuffle_epi32(
              cRhsMat236720, 221); // B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

          const __m256i cRhsMat014521Sp2 = _mm256_shuffle_epi32(
              cRhsMat014521,
              221); // B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
          const __m256i cRhsMat236721Sp2 = _mm256_shuffle_epi32(
              cRhsMat236721,
              221); // B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

          const __m256i cRhsMat014530Sp2 = _mm256_shuffle_epi32(
              cRhsMat014530, 221); // B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
          const __m256i cRhsMat236730Sp2 = _mm256_shuffle_epi32(
              cRhsMat236730, 221); // B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

          const __m256i cRhsMat014531Sp2 = _mm256_shuffle_epi32(
              cRhsMat014531,
              221); // B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
          const __m256i cRhsMat236731Sp2 = _mm256_shuffle_epi32(
              cRhsMat236731,
              221); // B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

          const __m256i cRhsMat014540Sp2 = _mm256_shuffle_epi32(
              cRhsMat014540, 221); // B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
          const __m256i cRhsMat236740Sp2 = _mm256_shuffle_epi32(
              cRhsMat236740, 221); // B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

          const __m256i cRhsMat014541Sp2 = _mm256_shuffle_epi32(
              cRhsMat014541,
              221); // B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
          const __m256i cRhsMat236741Sp2 = _mm256_shuffle_epi32(
              cRhsMat236741,
              221); // B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

          const __m256i cRhsMat014550Sp2 = _mm256_shuffle_epi32(
              cRhsMat014550, 221); // B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
          const __m256i cRhsMat236750Sp2 = _mm256_shuffle_epi32(
              cRhsMat236750, 221); // B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

          const __m256i cRhsMat014551Sp2 = _mm256_shuffle_epi32(
              cRhsMat014551,
              221); // B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
          const __m256i cRhsMat236751Sp2 = _mm256_shuffle_epi32(
              cRhsMat236751,
              221); // B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

          const __m256i cRhsMat014560Sp2 = _mm256_shuffle_epi32(
              cRhsMat014560, 221); // B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
          const __m256i cRhsMat236760Sp2 = _mm256_shuffle_epi32(
              cRhsMat236760, 221); // B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

          const __m256i cRhsMat014561Sp2 = _mm256_shuffle_epi32(
              cRhsMat014561,
              221); // B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
          const __m256i cRhsMat236761Sp2 = _mm256_shuffle_epi32(
              cRhsMat236761,
              221); // B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

          const __m256i cRhsMat014570Sp2 = _mm256_shuffle_epi32(
              cRhsMat014570, 221); // B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
          const __m256i cRhsMat236770Sp2 = _mm256_shuffle_epi32(
              cRhsMat236770, 221); // B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

          const __m256i cRhsMat014571Sp2 = _mm256_shuffle_epi32(
              cRhsMat014571,
              221); // B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
          const __m256i cRhsMat236771Sp2 = _mm256_shuffle_epi32(
              cRhsMat236771,
              221); // B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)

          // Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
          // s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51
          // m51  s60 m60  s61 m61   s70 m70  s71 m71

          // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
          const __m128i cMinsAndScales01 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + vSb * 64));
          const __m128i cMinsAndScales23 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 16 + vSb * 64));
          const __m128i cMinsAndScales45 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 32 + vSb * 64));
          const __m128i cMinsAndScales67 = _mm_loadu_si128((const __m128i *)(vBPtr[vB].scales + 48 + vSb * 64));

          // Extract scales which is lower half from mins_and_scales
          const __m128i cScales01 = _mm_and_si128(cMinsAndScales01, cM4bSse);
          const __m128i cScales23 = _mm_and_si128(cMinsAndScales23, cM4bSse);
          const __m128i cScales45 = _mm_and_si128(cMinsAndScales45, cM4bSse);
          const __m128i cScales67 = _mm_and_si128(cMinsAndScales67, cM4bSse);

          // Extract mins which is upper half from mins_and_scales
          const __m256i cMins01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales01, 4), cM4bSse));
          const __m256i cMins23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales23, 4), cM4bSse));
          const __m256i cMins45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales45, 4), cM4bSse));
          const __m256i cMins67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(cMinsAndScales67, 4), cM4bSse));

          const __m256i cScales0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales01, vScalesmask1Sse));
          const __m256i cScales1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales01, vScalesmask2Sse));

          const __m256i cScales2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales23, vScalesmask1Sse));
          const __m256i cScales3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales23, vScalesmask2Sse));

          const __m256i cScales4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales45, vScalesmask1Sse));
          const __m256i cScales5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales45, vScalesmask2Sse));

          const __m256i cScales6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales67, vScalesmask1Sse));
          const __m256i cScales7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(cScales67, vScalesmask2Sse));

          const __m256i cScale01450 = _mm256_shuffle_epi32(cScales0, 68);
          const __m256i cScale23670 = _mm256_shuffle_epi32(cScales0, 238);

          const __m256i cScale01451 = _mm256_shuffle_epi32(cScales1, 68);
          const __m256i cScale23671 = _mm256_shuffle_epi32(cScales1, 238);

          const __m256i cScale01452 = _mm256_shuffle_epi32(cScales2, 68);
          const __m256i cScale23672 = _mm256_shuffle_epi32(cScales2, 238);

          const __m256i cScale01453 = _mm256_shuffle_epi32(cScales3, 68);
          const __m256i cScale23673 = _mm256_shuffle_epi32(cScales3, 238);

          const __m256i cScale01454 = _mm256_shuffle_epi32(cScales4, 68);
          const __m256i cScale23674 = _mm256_shuffle_epi32(cScales4, 238);

          const __m256i cScale01455 = _mm256_shuffle_epi32(cScales5, 68);
          const __m256i cScale23675 = _mm256_shuffle_epi32(cScales5, 238);

          const __m256i cScale01456 = _mm256_shuffle_epi32(cScales6, 68);
          const __m256i cScale23676 = _mm256_shuffle_epi32(cScales6, 238);

          const __m256i cScale01457 = _mm256_shuffle_epi32(cScales7, 68);
          const __m256i cScale23677 = _mm256_shuffle_epi32(cScales7, 238);

          // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes -
          // A0,A1,A2,A3 Loaded as set of 128 bit vectors and repeated into a 256 bit vector
          __m256i vLhsMat012300 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 512 * vSb)));
          __m256i vLhsMat0100 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 0);
          __m256i vLhsMat2300 = _mm256_permute2f128_si256(vLhsMat012300, vLhsMat012300, 17);
          __m256i vLhsMat012301 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 32 + 512 * vSb)));
          __m256i vLhsMat0101 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 0);
          __m256i vLhsMat2301 = _mm256_permute2f128_si256(vLhsMat012301, vLhsMat012301, 17);
          __m256i vLhsMat012310 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 64 + 512 * vSb)));
          __m256i vLhsMat0110 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 0);
          __m256i vLhsMat2310 = _mm256_permute2f128_si256(vLhsMat012310, vLhsMat012310, 17);
          __m256i vLhsMat012311 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 96 + 512 * vSb)));
          __m256i vLhsMat0111 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 0);
          __m256i vLhsMat2311 = _mm256_permute2f128_si256(vLhsMat012311, vLhsMat012311, 17);
          __m256i vLhsMat012320 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 128 + 512 * vSb)));
          __m256i vLhsMat0120 = _mm256_permute2f128_si256(vLhsMat012320, vLhsMat012320, 0);
          __m256i vLhsMat2320 = _mm256_permute2f128_si256(vLhsMat012320, vLhsMat012320, 17);
          __m256i vLhsMat012321 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 160 + 512 * vSb)));
          __m256i vLhsMat0121 = _mm256_permute2f128_si256(vLhsMat012321, vLhsMat012321, 0);
          __m256i vLhsMat2321 = _mm256_permute2f128_si256(vLhsMat012321, vLhsMat012321, 17);
          __m256i vLhsMat012330 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 192 + 512 * vSb)));
          __m256i vLhsMat0130 = _mm256_permute2f128_si256(vLhsMat012330, vLhsMat012330, 0);
          __m256i vLhsMat2330 = _mm256_permute2f128_si256(vLhsMat012330, vLhsMat012330, 17);
          __m256i vLhsMat012331 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 224 + 512 * vSb)));
          __m256i vLhsMat0131 = _mm256_permute2f128_si256(vLhsMat012331, vLhsMat012331, 0);
          __m256i vLhsMat2331 = _mm256_permute2f128_si256(vLhsMat012331, vLhsMat012331, 17);

          __m256i vLhsMat012340 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 256 + 512 * vSb)));
          __m256i vLhsMat0140 = _mm256_permute2f128_si256(vLhsMat012340, vLhsMat012340, 0);
          __m256i vLhsMat2340 = _mm256_permute2f128_si256(vLhsMat012340, vLhsMat012340, 17);
          __m256i vLhsMat012341 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 288 + 512 * vSb)));
          __m256i vLhsMat0141 = _mm256_permute2f128_si256(vLhsMat012341, vLhsMat012341, 0);
          __m256i vLhsMat2341 = _mm256_permute2f128_si256(vLhsMat012341, vLhsMat012341, 17);
          __m256i vLhsMat012350 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 320 + 512 * vSb)));
          __m256i vLhsMat0150 = _mm256_permute2f128_si256(vLhsMat012350, vLhsMat012350, 0);
          __m256i vLhsMat2350 = _mm256_permute2f128_si256(vLhsMat012350, vLhsMat012350, 17);
          __m256i vLhsMat012351 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 352 + 512 * vSb)));
          __m256i vLhsMat0151 = _mm256_permute2f128_si256(vLhsMat012351, vLhsMat012351, 0);
          __m256i vLhsMat2351 = _mm256_permute2f128_si256(vLhsMat012351, vLhsMat012351, 17);
          __m256i vLhsMat012360 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 384 + 512 * vSb)));
          __m256i vLhsMat0160 = _mm256_permute2f128_si256(vLhsMat012360, vLhsMat012360, 0);
          __m256i vLhsMat2360 = _mm256_permute2f128_si256(vLhsMat012360, vLhsMat012360, 17);
          __m256i vLhsMat012361 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 416 + 512 * vSb)));
          __m256i vLhsMat0161 = _mm256_permute2f128_si256(vLhsMat012361, vLhsMat012361, 0);
          __m256i vLhsMat2361 = _mm256_permute2f128_si256(vLhsMat012361, vLhsMat012361, 17);
          __m256i vLhsMat012370 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 448 + 512 * vSb)));
          __m256i vLhsMat0170 = _mm256_permute2f128_si256(vLhsMat012370, vLhsMat012370, 0);
          __m256i vLhsMat2370 = _mm256_permute2f128_si256(vLhsMat012370, vLhsMat012370, 17);
          __m256i vLhsMat012371 = _mm256_loadu_si256((const __m256i *)((vAPtr[vB].qs + 480 + 512 * vSb)));
          __m256i vLhsMat0171 = _mm256_permute2f128_si256(vLhsMat012371, vLhsMat012371, 0);
          __m256i vLhsMat2371 = _mm256_permute2f128_si256(vLhsMat012371, vLhsMat012371, 17);

          // Bsums are loaded for the different Q8_K blocks
          __m128i vLhsRawBsums010123 = _mm_loadu_si128((const __m128i *)((vAPtr[vB].bsums + 32 * vSb)));
          __m128i vLhsRawBsums230123 = _mm_loadu_si128((const __m128i *)(vAPtr[vB].bsums + 8 + 32 * vSb));
          __m128i vLhsRawBsums014567 = _mm_loadu_si128((const __m128i *)((vAPtr[vB].bsums + 16 + 32 * vSb)));
          __m128i vLhsRawBsums234567 = _mm_loadu_si128((const __m128i *)(vAPtr[vB].bsums + 24 + 32 * vSb));

          // Shuffle pattern one - left side input
          const __m256i cLhsMat0100Sp1 = _mm256_shuffle_epi32(
              vLhsMat0100, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
          const __m256i cLhsMat2300Sp1 = _mm256_shuffle_epi32(
              vLhsMat2300, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

          const __m256i cLhsMat0101Sp1 = _mm256_shuffle_epi32(
              vLhsMat0101, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m256i cLhsMat2301Sp1 = _mm256_shuffle_epi32(
              vLhsMat2301, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

          const __m256i cLhsMat0110Sp1 = _mm256_shuffle_epi32(
              vLhsMat0110, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
          const __m256i cLhsMat2310Sp1 = _mm256_shuffle_epi32(
              vLhsMat2310, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

          const __m256i cLhsMat0111Sp1 = _mm256_shuffle_epi32(
              vLhsMat0111, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m256i cLhsMat2311Sp1 = _mm256_shuffle_epi32(
              vLhsMat2311, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

          const __m256i cLhsMat0120Sp1 = _mm256_shuffle_epi32(
              vLhsMat0120, 160); // A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
          const __m256i cLhsMat2320Sp1 = _mm256_shuffle_epi32(
              vLhsMat2320, 160); // A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

          const __m256i cLhsMat0121Sp1 = _mm256_shuffle_epi32(
              vLhsMat0121, 160); // A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
          const __m256i cLhsMat2321Sp1 = _mm256_shuffle_epi32(
              vLhsMat2321, 160); // A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

          const __m256i cLhsMat0130Sp1 = _mm256_shuffle_epi32(
              vLhsMat0130, 160); // A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
          const __m256i cLhsMat2330Sp1 = _mm256_shuffle_epi32(
              vLhsMat2330, 160); // A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

          const __m256i cLhsMat0131Sp1 = _mm256_shuffle_epi32(
              vLhsMat0131, 160); // A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
          const __m256i cLhsMat2331Sp1 = _mm256_shuffle_epi32(
              vLhsMat2331, 160); // A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

          const __m256i cLhsMat0140Sp1 = _mm256_shuffle_epi32(
              vLhsMat0140, 160); // A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
          const __m256i cLhsMat2340Sp1 = _mm256_shuffle_epi32(
              vLhsMat2340, 160); // A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

          const __m256i cLhsMat0141Sp1 = _mm256_shuffle_epi32(
              vLhsMat0141, 160); // A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
          const __m256i cLhsMat2341Sp1 = _mm256_shuffle_epi32(
              vLhsMat2341, 160); // A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

          const __m256i cLhsMat0150Sp1 = _mm256_shuffle_epi32(
              vLhsMat0150, 160); // A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
          const __m256i cLhsMat2350Sp1 = _mm256_shuffle_epi32(
              vLhsMat2350, 160); // A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

          const __m256i cLhsMat0151Sp1 = _mm256_shuffle_epi32(
              vLhsMat0151, 160); // A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
          const __m256i cLhsMat2351Sp1 = _mm256_shuffle_epi32(
              vLhsMat2351, 160); // A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

          const __m256i cLhsMat0160Sp1 = _mm256_shuffle_epi32(
              vLhsMat0160, 160); // A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
          const __m256i cLhsMat2360Sp1 = _mm256_shuffle_epi32(
              vLhsMat2360, 160); // A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

          const __m256i cLhsMat0161Sp1 = _mm256_shuffle_epi32(
              vLhsMat0161, 160); // A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
          const __m256i cLhsMat2361Sp1 = _mm256_shuffle_epi32(
              vLhsMat2361, 160); // A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

          const __m256i cLhsMat0170Sp1 = _mm256_shuffle_epi32(
              vLhsMat0170, 160); // A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
          const __m256i cLhsMat2370Sp1 = _mm256_shuffle_epi32(
              vLhsMat2370, 160); // A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

          const __m256i cLhsMat0171Sp1 = _mm256_shuffle_epi32(
              vLhsMat0171, 160); // A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
          const __m256i cLhsMat2371Sp1 = _mm256_shuffle_epi32(
              vLhsMat2371, 160); // A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

          // Shuffle pattern two- left side input
          const __m256i cLhsMat0100Sp2 = _mm256_shuffle_epi32(
              vLhsMat0100, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
          const __m256i cLhsMat2300Sp2 = _mm256_shuffle_epi32(
              vLhsMat2300, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

          const __m256i cLhsMat0101Sp2 = _mm256_shuffle_epi32(
              vLhsMat0101,
              245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m256i cLhsMat2301Sp2 = _mm256_shuffle_epi32(
              vLhsMat2301,
              245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

          const __m256i cLhsMat0110Sp2 = _mm256_shuffle_epi32(
              vLhsMat0110, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
          const __m256i cLhsMat2310Sp2 = _mm256_shuffle_epi32(
              vLhsMat2310, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

          const __m256i cLhsMat0111Sp2 = _mm256_shuffle_epi32(
              vLhsMat0111,
              245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m256i cLhsMat2311Sp2 = _mm256_shuffle_epi32(
              vLhsMat2311,
              245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

          const __m256i cLhsMat0120Sp2 = _mm256_shuffle_epi32(
              vLhsMat0120, 245); // A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
          const __m256i cLhsMat2320Sp2 = _mm256_shuffle_epi32(
              vLhsMat2320, 245); // A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

          const __m256i cLhsMat0121Sp2 = _mm256_shuffle_epi32(
              vLhsMat0121,
              245); // A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
          const __m256i cLhsMat2321Sp2 = _mm256_shuffle_epi32(
              vLhsMat2321,
              245); // A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

          const __m256i cLhsMat0130Sp2 = _mm256_shuffle_epi32(
              vLhsMat0130, 245); // A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
          const __m256i cLhsMat2330Sp2 = _mm256_shuffle_epi32(
              vLhsMat2330, 245); // A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

          const __m256i cLhsMat0131Sp2 = _mm256_shuffle_epi32(
              vLhsMat0131,
              245); // A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
          const __m256i cLhsMat2331Sp2 = _mm256_shuffle_epi32(
              vLhsMat2331,
              245); // A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

          const __m256i cLhsMat0140Sp2 = _mm256_shuffle_epi32(
              vLhsMat0140, 245); // A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
          const __m256i cLhsMat2340Sp2 = _mm256_shuffle_epi32(
              vLhsMat2340, 245); // A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

          const __m256i cLhsMat0141Sp2 = _mm256_shuffle_epi32(
              vLhsMat0141,
              245); // A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
          const __m256i cLhsMat2341Sp2 = _mm256_shuffle_epi32(
              vLhsMat2341,
              245); // A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

          const __m256i cLhsMat0150Sp2 = _mm256_shuffle_epi32(
              vLhsMat0150, 245); // A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
          const __m256i cLhsMat2350Sp2 = _mm256_shuffle_epi32(
              vLhsMat2350, 245); // A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

          const __m256i cLhsMat0151Sp2 = _mm256_shuffle_epi32(
              vLhsMat0151,
              245); // A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
          const __m256i cLhsMat2351Sp2 = _mm256_shuffle_epi32(
              vLhsMat2351,
              245); // A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

          const __m256i cLhsMat0160Sp2 = _mm256_shuffle_epi32(
              vLhsMat0160, 245); // A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
          const __m256i cLhsMat2360Sp2 = _mm256_shuffle_epi32(
              vLhsMat2360, 245); // A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

          const __m256i cLhsMat0161Sp2 = _mm256_shuffle_epi32(
              vLhsMat0161,
              245); // A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
          const __m256i cLhsMat2361Sp2 = _mm256_shuffle_epi32(
              vLhsMat2361,
              245); // A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

          const __m256i cLhsMat0170Sp2 = _mm256_shuffle_epi32(
              vLhsMat0170, 245); // A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
          const __m256i cLhsMat2370Sp2 = _mm256_shuffle_epi32(
              vLhsMat2370, 245); // A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

          const __m256i cLhsMat0171Sp2 = _mm256_shuffle_epi32(
              vLhsMat0171,
              245); // A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
          const __m256i cLhsMat2371Sp2 = _mm256_shuffle_epi32(
              vLhsMat2371,
              245); // A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

          // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e
          // corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
          __m256i vIaccMat000Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat0100Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat0101Sp1));
          __m256i vIaccMat010Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat0100Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat0101Sp1));

          __m256i vIaccMat100Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp1, cLhsMat2300Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014501Sp1, cLhsMat2301Sp1));
          __m256i vIaccMat110Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp1, cLhsMat2300Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236701Sp1, cLhsMat2301Sp1));

          __m256i vIaccMat001Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat0110Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat0111Sp1));
          __m256i vIaccMat011Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat0110Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat0111Sp1));

          __m256i vIaccMat101Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp1, cLhsMat2310Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014511Sp1, cLhsMat2311Sp1));
          __m256i vIaccMat111Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp1, cLhsMat2310Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236711Sp1, cLhsMat2311Sp1));

          __m256i vIaccMat002Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp1, cLhsMat0120Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014521Sp1, cLhsMat0121Sp1));
          __m256i vIaccMat012Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp1, cLhsMat0120Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236721Sp1, cLhsMat0121Sp1));

          __m256i vIaccMat102Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp1, cLhsMat2320Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014521Sp1, cLhsMat2321Sp1));
          __m256i vIaccMat112Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp1, cLhsMat2320Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236721Sp1, cLhsMat2321Sp1));

          __m256i vIaccMat003Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp1, cLhsMat0130Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014531Sp1, cLhsMat0131Sp1));
          __m256i vIaccMat013Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp1, cLhsMat0130Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236731Sp1, cLhsMat0131Sp1));

          __m256i vIaccMat103Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp1, cLhsMat2330Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014531Sp1, cLhsMat2331Sp1));
          __m256i vIaccMat113Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp1, cLhsMat2330Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236731Sp1, cLhsMat2331Sp1));

          __m256i vIaccMat004Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp1, cLhsMat0140Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014541Sp1, cLhsMat0141Sp1));
          __m256i vIaccMat014Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp1, cLhsMat0140Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236741Sp1, cLhsMat0141Sp1));

          __m256i vIaccMat104Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp1, cLhsMat2340Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014541Sp1, cLhsMat2341Sp1));
          __m256i vIaccMat114Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp1, cLhsMat2340Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236741Sp1, cLhsMat2341Sp1));

          __m256i vIaccMat005Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp1, cLhsMat0150Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014551Sp1, cLhsMat0151Sp1));
          __m256i vIaccMat015Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp1, cLhsMat0150Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236751Sp1, cLhsMat0151Sp1));

          __m256i vIaccMat105Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp1, cLhsMat2350Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014551Sp1, cLhsMat2351Sp1));
          __m256i vIaccMat115Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp1, cLhsMat2350Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236751Sp1, cLhsMat2351Sp1));

          __m256i vIaccMat006Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp1, cLhsMat0160Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014561Sp1, cLhsMat0161Sp1));
          __m256i vIaccMat016Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp1, cLhsMat0160Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236761Sp1, cLhsMat0161Sp1));

          __m256i vIaccMat106Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp1, cLhsMat2360Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014561Sp1, cLhsMat2361Sp1));
          __m256i vIaccMat116Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp1, cLhsMat2360Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236761Sp1, cLhsMat2361Sp1));

          __m256i vIaccMat007Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp1, cLhsMat0170Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014571Sp1, cLhsMat0171Sp1));
          __m256i vIaccMat017Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp1, cLhsMat0170Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236771Sp1, cLhsMat0171Sp1));

          __m256i vIaccMat107Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp1, cLhsMat2370Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat014571Sp1, cLhsMat2371Sp1));
          __m256i vIaccMat117Sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp1, cLhsMat2370Sp1),
                                                    _mm256_maddubs_epi16(cRhsMat236771Sp1, cLhsMat2371Sp1));

          __m256i vIaccMat000Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat0100Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat0101Sp2));
          __m256i vIaccMat010Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat0100Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat0101Sp2));

          __m256i vIaccMat100Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014500Sp2, cLhsMat2300Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014501Sp2, cLhsMat2301Sp2));
          __m256i vIaccMat110Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236700Sp2, cLhsMat2300Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236701Sp2, cLhsMat2301Sp2));

          __m256i vIaccMat001Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat0110Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat0111Sp2));
          __m256i vIaccMat011Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat0110Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat0111Sp2));

          __m256i vIaccMat101Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014510Sp2, cLhsMat2310Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014511Sp2, cLhsMat2311Sp2));
          __m256i vIaccMat111Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236710Sp2, cLhsMat2310Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236711Sp2, cLhsMat2311Sp2));

          __m256i vIaccMat002Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp2, cLhsMat0120Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014521Sp2, cLhsMat0121Sp2));
          __m256i vIaccMat012Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp2, cLhsMat0120Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236721Sp2, cLhsMat0121Sp2));

          __m256i vIaccMat102Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014520Sp2, cLhsMat2320Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014521Sp2, cLhsMat2321Sp2));
          __m256i vIaccMat112Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236720Sp2, cLhsMat2320Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236721Sp2, cLhsMat2321Sp2));

          __m256i vIaccMat003Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp2, cLhsMat0130Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014531Sp2, cLhsMat0131Sp2));
          __m256i vIaccMat013Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp2, cLhsMat0130Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236731Sp2, cLhsMat0131Sp2));

          __m256i vIaccMat103Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014530Sp2, cLhsMat2330Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014531Sp2, cLhsMat2331Sp2));
          __m256i vIaccMat113Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236730Sp2, cLhsMat2330Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236731Sp2, cLhsMat2331Sp2));

          __m256i vIaccMat004Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp2, cLhsMat0140Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014541Sp2, cLhsMat0141Sp2));
          __m256i vIaccMat014Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp2, cLhsMat0140Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236741Sp2, cLhsMat0141Sp2));

          __m256i vIaccMat104Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014540Sp2, cLhsMat2340Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014541Sp2, cLhsMat2341Sp2));
          __m256i vIaccMat114Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236740Sp2, cLhsMat2340Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236741Sp2, cLhsMat2341Sp2));

          __m256i vIaccMat005Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp2, cLhsMat0150Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014551Sp2, cLhsMat0151Sp2));
          __m256i vIaccMat015Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp2, cLhsMat0150Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236751Sp2, cLhsMat0151Sp2));

          __m256i vIaccMat105Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014550Sp2, cLhsMat2350Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014551Sp2, cLhsMat2351Sp2));
          __m256i vIaccMat115Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236750Sp2, cLhsMat2350Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236751Sp2, cLhsMat2351Sp2));

          __m256i vIaccMat006Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp2, cLhsMat0160Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014561Sp2, cLhsMat0161Sp2));
          __m256i vIaccMat016Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp2, cLhsMat0160Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236761Sp2, cLhsMat0161Sp2));

          __m256i vIaccMat106Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014560Sp2, cLhsMat2360Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014561Sp2, cLhsMat2361Sp2));
          __m256i vIaccMat116Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236760Sp2, cLhsMat2360Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236761Sp2, cLhsMat2361Sp2));

          __m256i vIaccMat007Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp2, cLhsMat0170Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014571Sp2, cLhsMat0171Sp2));
          __m256i vIaccMat017Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp2, cLhsMat0170Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236771Sp2, cLhsMat0171Sp2));

          __m256i vIaccMat107Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat014570Sp2, cLhsMat2370Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat014571Sp2, cLhsMat2371Sp2));
          __m256i vIaccMat117Sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(cRhsMat236770Sp2, cLhsMat2370Sp2),
                                                    _mm256_maddubs_epi16(cRhsMat236771Sp2, cLhsMat2371Sp2));

          // Combine results from both shuffle patterns for each output block.
          __m256i vIaccMat000 = _mm256_add_epi16(vIaccMat000Sp1, vIaccMat000Sp2);
          __m256i vIaccMat010 = _mm256_add_epi16(vIaccMat010Sp1, vIaccMat010Sp2);
          __m256i vIaccMat100 = _mm256_add_epi16(vIaccMat100Sp1, vIaccMat100Sp2);
          __m256i vIaccMat110 = _mm256_add_epi16(vIaccMat110Sp1, vIaccMat110Sp2);

          __m256i vIaccMat001 = _mm256_add_epi16(vIaccMat001Sp1, vIaccMat001Sp2);
          __m256i vIaccMat011 = _mm256_add_epi16(vIaccMat011Sp1, vIaccMat011Sp2);
          __m256i vIaccMat101 = _mm256_add_epi16(vIaccMat101Sp1, vIaccMat101Sp2);
          __m256i vIaccMat111 = _mm256_add_epi16(vIaccMat111Sp1, vIaccMat111Sp2);

          __m256i vIaccMat002 = _mm256_add_epi16(vIaccMat002Sp1, vIaccMat002Sp2);
          __m256i vIaccMat012 = _mm256_add_epi16(vIaccMat012Sp1, vIaccMat012Sp2);
          __m256i vIaccMat102 = _mm256_add_epi16(vIaccMat102Sp1, vIaccMat102Sp2);
          __m256i vIaccMat112 = _mm256_add_epi16(vIaccMat112Sp1, vIaccMat112Sp2);

          __m256i vIaccMat003 = _mm256_add_epi16(vIaccMat003Sp1, vIaccMat003Sp2);
          __m256i vIaccMat013 = _mm256_add_epi16(vIaccMat013Sp1, vIaccMat013Sp2);
          __m256i vIaccMat103 = _mm256_add_epi16(vIaccMat103Sp1, vIaccMat103Sp2);
          __m256i vIaccMat113 = _mm256_add_epi16(vIaccMat113Sp1, vIaccMat113Sp2);

          __m256i vIaccMat004 = _mm256_add_epi16(vIaccMat004Sp1, vIaccMat004Sp2);
          __m256i vIaccMat014 = _mm256_add_epi16(vIaccMat014Sp1, vIaccMat014Sp2);
          __m256i vIaccMat104 = _mm256_add_epi16(vIaccMat104Sp1, vIaccMat104Sp2);
          __m256i vIaccMat114 = _mm256_add_epi16(vIaccMat114Sp1, vIaccMat114Sp2);

          __m256i vIaccMat005 = _mm256_add_epi16(vIaccMat005Sp1, vIaccMat005Sp2);
          __m256i vIaccMat015 = _mm256_add_epi16(vIaccMat015Sp1, vIaccMat015Sp2);
          __m256i vIaccMat105 = _mm256_add_epi16(vIaccMat105Sp1, vIaccMat105Sp2);
          __m256i vIaccMat115 = _mm256_add_epi16(vIaccMat115Sp1, vIaccMat115Sp2);

          __m256i vIaccMat006 = _mm256_add_epi16(vIaccMat006Sp1, vIaccMat006Sp2);
          __m256i vIaccMat016 = _mm256_add_epi16(vIaccMat016Sp1, vIaccMat016Sp2);
          __m256i vIaccMat106 = _mm256_add_epi16(vIaccMat106Sp1, vIaccMat106Sp2);
          __m256i vIaccMat116 = _mm256_add_epi16(vIaccMat116Sp1, vIaccMat116Sp2);

          __m256i vIaccMat007 = _mm256_add_epi16(vIaccMat007Sp1, vIaccMat007Sp2);
          __m256i vIaccMat017 = _mm256_add_epi16(vIaccMat017Sp1, vIaccMat017Sp2);
          __m256i vIaccMat107 = _mm256_add_epi16(vIaccMat107Sp1, vIaccMat107Sp2);
          __m256i vIaccMat117 = _mm256_add_epi16(vIaccMat117Sp1, vIaccMat117Sp2);

          // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
          vIaccMat000 = _mm256_madd_epi16(vIaccMat000, cScale01450);
          vIaccMat010 = _mm256_madd_epi16(vIaccMat010, cScale23670);
          vIaccMat100 = _mm256_madd_epi16(vIaccMat100, cScale01450);
          vIaccMat110 = _mm256_madd_epi16(vIaccMat110, cScale23670);

          vIaccMat001 = _mm256_madd_epi16(vIaccMat001, cScale01451);
          vIaccMat011 = _mm256_madd_epi16(vIaccMat011, cScale23671);
          vIaccMat101 = _mm256_madd_epi16(vIaccMat101, cScale01451);
          vIaccMat111 = _mm256_madd_epi16(vIaccMat111, cScale23671);

          vIaccMat002 = _mm256_madd_epi16(vIaccMat002, cScale01452);
          vIaccMat012 = _mm256_madd_epi16(vIaccMat012, cScale23672);
          vIaccMat102 = _mm256_madd_epi16(vIaccMat102, cScale01452);
          vIaccMat112 = _mm256_madd_epi16(vIaccMat112, cScale23672);

          vIaccMat003 = _mm256_madd_epi16(vIaccMat003, cScale01453);
          vIaccMat013 = _mm256_madd_epi16(vIaccMat013, cScale23673);
          vIaccMat103 = _mm256_madd_epi16(vIaccMat103, cScale01453);
          vIaccMat113 = _mm256_madd_epi16(vIaccMat113, cScale23673);

          vIaccMat004 = _mm256_madd_epi16(vIaccMat004, cScale01454);
          vIaccMat014 = _mm256_madd_epi16(vIaccMat014, cScale23674);
          vIaccMat104 = _mm256_madd_epi16(vIaccMat104, cScale01454);
          vIaccMat114 = _mm256_madd_epi16(vIaccMat114, cScale23674);

          vIaccMat005 = _mm256_madd_epi16(vIaccMat005, cScale01455);
          vIaccMat015 = _mm256_madd_epi16(vIaccMat015, cScale23675);
          vIaccMat105 = _mm256_madd_epi16(vIaccMat105, cScale01455);
          vIaccMat115 = _mm256_madd_epi16(vIaccMat115, cScale23675);

          vIaccMat006 = _mm256_madd_epi16(vIaccMat006, cScale01456);
          vIaccMat016 = _mm256_madd_epi16(vIaccMat016, cScale23676);
          vIaccMat106 = _mm256_madd_epi16(vIaccMat106, cScale01456);
          vIaccMat116 = _mm256_madd_epi16(vIaccMat116, cScale23676);

          vIaccMat007 = _mm256_madd_epi16(vIaccMat007, cScale01457);
          vIaccMat017 = _mm256_madd_epi16(vIaccMat017, cScale23677);
          vIaccMat107 = _mm256_madd_epi16(vIaccMat107, cScale01457);
          vIaccMat117 = _mm256_madd_epi16(vIaccMat117, cScale23677);

          __m256i vIaccMat00 = _mm256_add_epi32(
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat000, vIaccMat001), _mm256_add_epi32(vIaccMat002, vIaccMat003)),
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat004, vIaccMat005), _mm256_add_epi32(vIaccMat006, vIaccMat007)));
          __m256i vIaccMat01 = _mm256_add_epi32(
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat010, vIaccMat011), _mm256_add_epi32(vIaccMat012, vIaccMat013)),
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat014, vIaccMat015), _mm256_add_epi32(vIaccMat016, vIaccMat017)));
          __m256i vIaccMat10 = _mm256_add_epi32(
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat100, vIaccMat101), _mm256_add_epi32(vIaccMat102, vIaccMat103)),
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat104, vIaccMat105), _mm256_add_epi32(vIaccMat106, vIaccMat107)));
          __m256i vIaccMat11 = _mm256_add_epi32(
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat110, vIaccMat111), _mm256_add_epi32(vIaccMat112, vIaccMat113)),
              _mm256_add_epi32(_mm256_add_epi32(vIaccMat114, vIaccMat115), _mm256_add_epi32(vIaccMat116, vIaccMat117)));

          // Straighten out to make 4 row vectors
          __m256i vIaccRow0 = _mm256_blend_epi32(vIaccMat00, _mm256_shuffle_epi32(vIaccMat01, 78), 204);
          __m256i vIaccRow1 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat00, 78), vIaccMat01, 204);
          __m256i vIaccRow2 = _mm256_blend_epi32(vIaccMat10, _mm256_shuffle_epi32(vIaccMat11, 78), 204);
          __m256i vIaccRow3 = _mm256_blend_epi32(_mm256_shuffle_epi32(vIaccMat10, 78), vIaccMat11, 204);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
          const __m128 cRowScaleF32Sse = _mm_load_ps(vAPtr[vB].d);
          const __m256 cRowScaleF32 = _mm256_set_m128(cRowScaleF32Sse, cRowScaleF32Sse);

          // Multiply with appropriate scales and accumulate (for both d and dmin) below
          aAccRows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow0),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                        aAccRows[0]);
          aAccRows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow1),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)),
                                        aAccRows[1]);
          aAccRows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow2),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)),
                                        aAccRows[2]);
          aAccRows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRow3),
                                        _mm256_mul_ps(cColScaleF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)),
                                        aAccRows[3]);

          __m256i vLhsBsums010123 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums010123), vLhsRawBsums010123, 1);
          __m256i vLhsBsums230123 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums230123), vLhsRawBsums230123, 1);
          __m256i vLhsBsums014567 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums014567), vLhsRawBsums014567, 1);
          __m256i vLhsBsums234567 =
              _mm256_inserti128_si256(_mm256_castsi128_si256(vLhsRawBsums234567), vLhsRawBsums234567, 1);

          // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
          __m256i vIaccRowMin001 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 0), cMins01);
          __m256i vIaccRowMin101 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 170), cMins01);
          __m256i vIaccRowMin201 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 0), cMins01);
          __m256i vIaccRowMin301 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 170), cMins01);

          __m256i vIaccRowMin023 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 85), cMins23);
          __m256i vIaccRowMin123 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums010123, 255), cMins23);
          __m256i vIaccRowMin223 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 85), cMins23);
          __m256i vIaccRowMin323 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums230123, 255), cMins23);

          __m256i vIaccRowMin045 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 0), cMins45);
          __m256i vIaccRowMin145 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 170), cMins45);
          __m256i vIaccRowMin245 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 0), cMins45);
          __m256i vIaccRowMin345 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 170), cMins45);

          __m256i vIaccRowMin067 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 85), cMins67);
          __m256i vIaccRowMin167 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums014567, 255), cMins67);
          __m256i vIaccRowMin267 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 85), cMins67);
          __m256i vIaccRowMin367 = _mm256_madd_epi16(_mm256_shuffle_epi32(vLhsBsums234567, 255), cMins67);

          __m256i vIaccRowMin0 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin001, vIaccRowMin023),
                                                  _mm256_add_epi32(vIaccRowMin045, vIaccRowMin067));
          __m256i vIaccRowMin1 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin101, vIaccRowMin123),
                                                  _mm256_add_epi32(vIaccRowMin145, vIaccRowMin167));
          __m256i vIaccRowMin2 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin201, vIaccRowMin223),
                                                  _mm256_add_epi32(vIaccRowMin245, vIaccRowMin267));
          __m256i vIaccRowMin3 = _mm256_add_epi32(_mm256_add_epi32(vIaccRowMin301, vIaccRowMin323),
                                                  _mm256_add_epi32(vIaccRowMin345, vIaccRowMin367));

          aAccMinRows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vIaccRowMin0),
                                           _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 0)),
                                           aAccMinRows[0]);
          aAccMinRows[1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin1),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 85)), aAccMinRows[1]);
          aAccMinRows[2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin2),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 170)), aAccMinRows[2]);
          aAccMinRows[3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(vIaccRowMin3),
              _mm256_mul_ps(cColDminF32, _mm256_shuffle_ps(cRowScaleF32, cRowScaleF32, 255)), aAccMinRows[3]);
        }
      }
      // Store the accumulated values
      for (int vI = 0; vI < 4; vI++) {
        _mm256_storeu_ps((float *)(pS + ((vY * 4 + vI) * pBs + vX * 8)), _mm256_sub_ps(aAccRows[vI], aAccMinRows[vI]));
      }
    }
  }
#else

  ggml_gemm_q2_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);

#endif
}
