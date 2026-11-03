#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i fMulSumI8Pairs(const __m128i pX, const __m128i pY) {
  // Get absolute values of x vectors
  const __m128i cAx = _mm_sign_epi8(pX, pX);
  // Sign the values of the y vectors
  const __m128i cSy = _mm_sign_epi8(pY, pX);
  // Perform multiplication and create 16-bit values
  const __m128i cDot = _mm_maddubs_epi16(cAx, cSy);
  const __m128i cOnes = _mm_set1_epi16(1);
  return _mm_madd_epi16(cOnes, cDot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float fHsumFloat8(const __m256 pX) {
  __m128 vRes = _mm256_extractf128_ps(pX, 1);
  vRes = _mm_add_ps(vRes, _mm256_castps256_ps128(pX));
  vRes = _mm_add_ps(vRes, _mm_movehl_ps(vRes, vRes));
  vRes = _mm_add_ss(vRes, _mm_movehdup_ps(vRes));
  return _mm_cvtss_f32(vRes);
}

// horizontally add 8 int32_t
static inline int fHsumI328(const __m256i pA) {
  const __m128i cSum128 = _mm_add_epi32(_mm256_castsi256_si128(pA), _mm256_extractf128_si256(pA, 1));
  const __m128i cHi64 = _mm_unpackhi_epi64(cSum128, cSum128);
  const __m128i cSum64 = _mm_add_epi32(cHi64, cSum128);
  const __m128i cHi32 = _mm_shuffle_epi32(cSum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(cSum64, cHi32));
}

// horizontally add 4 int32_t
static inline int fHsumI324(const __m128i pA) {
  const __m128i cHi64 = _mm_unpackhi_epi64(pA, pA);
  const __m128i cSum64 = _mm_add_epi32(cHi64, pA);
  const __m128i cHi32 = _mm_shuffle_epi32(cSum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(cSum64, cHi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline __m256i fMulAddEpi8(const __m256i pX, const __m256i pY) {
  const __m256i cAx = _mm256_sign_epi8(pX, pX);
  const __m256i cSy = _mm256_sign_epi8(pY, pX);
  return _mm256_maddubs_epi16(cAx, cSy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i fBytesFromBits32(const uint8_t *pX) {
  uint32_t vX32;
  memcpy(&vX32, pX, sizeof(uint32_t));
  const __m256i cShufMask =
      _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
  __m256i vBytes = _mm256_shuffle_epi8(_mm256_set1_epi32(vX32), cShufMask);
  const __m256i cBitMask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  vBytes = _mm256_or_si256(vBytes, cBitMask);
  return _mm256_cmpeq_epi8(vBytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i fBytesFromNibbles32(const uint8_t *pRsi) {
  const __m128i cTmp = _mm_loadu_si128((const __m128i *)pRsi);
  const __m256i cBytes = MM256_SET_M128I(_mm_srli_epi16(cTmp, 4), cTmp);
  const __m256i cLowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(cLowMask, cBytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 fSumI16PairsFloat(const __m256i pX) {
  const __m256i cOnes = _mm256_set1_epi16(1);
  const __m256i cSummedPairs = _mm256_madd_epi16(cOnes, pX);
  return _mm256_cvtepi32_ps(cSummedPairs);
}

static inline __m256 fMulSumUs8PairsFloat(const __m256i pAx, const __m256i pSy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  const __m256i cZero = _mm256_setzero_si256();
  const __m256i cSummedPairs = _mm256_dpbusd_epi32(cZero, pAx, pSy);
  return _mm256_cvtepi32_ps(cSummedPairs);
#elif defined(__AVXVNNI__)
  const __m256i cZero = _mm256_setzero_si256();
  const __m256i cSummedPairs = _mm256_dpbusd_avx_epi32(cZero, pAx, pSy);
  return _mm256_cvtepi32_ps(cSummedPairs);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(pAx, pSy);
  return fSumI16PairsFloat(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 fMulSumI8PairsFloat(const __m256i pX, const __m256i pY) {
#if __AVXVNNIINT8__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, pX, pY);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Get absolute values of x vectors
  const __m256i cAx = _mm256_sign_epi8(pX, pX);
  // Sign the values of the y vectors
  const __m256i cSy = _mm256_sign_epi8(pY, pX);
  return fMulSumUs8PairsFloat(cAx, cSy);
#endif
}

static inline __m128i fPackNibbles(__m256i pBytes) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
  const __m256i cBytesSrli4 = _mm256_srli_epi16(pBytes, 4); // 0000_0000_abcd_0000
  pBytes = _mm256_or_si256(pBytes, cBytesSrli4);            // 0000_abcd_abcd_efgh
  return _mm256_cvtepi16_epi8(pBytes);                      // abcd_efgh
#else
  const __m256i lowByte = _mm256_set1_epi16(0xFF);
  __m256i high = _mm256_andnot_si256(lowByte, pBytes);
  __m256i low = _mm256_and_si256(lowByte, pBytes);
  high = _mm256_srli_epi16(high, 4);
  pBytes = _mm256_or_si256(low, high);

  // Compress uint16_t lanes into bytes
  __m128i r0 = _mm256_castsi256_si128(pBytes);
  __m128i r1 = _mm256_extracti128_si256(pBytes, 1);
  return _mm_packus_epi16(r0, r1);
#endif
}
#elif defined(__AVX__)
static inline __m128i fPackNibbles(__m128i bytes1, __m128i bytes2) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
  const __m128i lowByte = _mm_set1_epi16(0xFF);
  __m128i high = _mm_andnot_si128(lowByte, bytes1);
  __m128i low = _mm_and_si128(lowByte, bytes1);
  high = _mm_srli_epi16(high, 4);
  bytes1 = _mm_or_si128(low, high);
  high = _mm_andnot_si128(lowByte, bytes2);
  low = _mm_and_si128(lowByte, bytes2);
  high = _mm_srli_epi16(high, 4);
  bytes2 = _mm_or_si128(low, high);

  return _mm_packus_epi16(bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
  const __m128i ax = _mm_sign_epi8(x, x);
  const __m128i sy = _mm_sign_epi8(y, x);
  return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i fBytesFromBits32(const uint8_t *x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
  const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
  __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
  __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
  const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytesl = _mm_or_si128(bytesl, bit_mask);
  bytesh = _mm_or_si128(bytesh, bit_mask);
  bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
  bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
  return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i fBytesFromNibbles32(const uint8_t *rsi) {
  // Load 16 bytes from memory
  __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
  __m128i tmph = _mm_srli_epi16(tmpl, 4);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  tmpl = _mm_and_si128(lowMask, tmpl);
  tmph = _mm_and_si128(lowMask, tmph);
  return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 fSumI16PairsFloat(const __m128i xh, const __m128i xl) {
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
  const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
  const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 fMulSumUs8PairsFloat(const __m256i ax, const __m256i sy) {
  const __m128i axl = _mm256_castsi256_si128(ax);
  const __m128i axh = _mm256_extractf128_si256(ax, 1);
  const __m128i syl = _mm256_castsi256_si128(sy);
  const __m128i syh = _mm256_extractf128_si256(sy, 1);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return fSumI16PairsFloat(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 fMulSumI8PairsFloat(const __m256i x, const __m256i y) {
  const __m128i xl = _mm256_castsi256_si128(x);
  const __m128i xh = _mm256_extractf128_si256(x, 1);
  const __m128i yl = _mm256_castsi256_si128(y);
  const __m128i yh = _mm256_extractf128_si256(y, 1);
  // Get absolute values of x vectors
  const __m128i axl = _mm_sign_epi8(xl, xl);
  const __m128i axh = _mm_sign_epi8(xh, xh);
  // Sign the values of the y vectors
  const __m128i syl = _mm_sign_epi8(yl, xl);
  const __m128i syh = _mm_sign_epi8(yh, xh);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return fSumI16PairsFloat(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0,
                                           const __m128i x_2_1, const __m128i y_1_0, const __m128i y_1_1,
                                           const __m128i y_2_0, const __m128i y_2_1) {
  const __m128i mone = _mm_set1_epi16(1);

  const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
  const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
  const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
  const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
  const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
  const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
  const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
  const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
  const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
  const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
  return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
  // GGML_CPU_FP16_TO_FP32 is faster than Intel F16C
  return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                         _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 quad_mx_delta_float(const uint8_t x0, const float y0, const uint8_t x1, const float y1) {
  return _mm256_set_m128(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                         _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
static inline __m128i bytes_from_bits_16(const uint8_t *x) {
  uint16_t x16;
  memcpy(&x16, x, sizeof(uint16_t));

  const __m128i shuf_mask = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
  __m128i bytes = _mm_shuffle_epi8(_mm_set1_epi16((short)x16), shuf_mask);
  const __m128i bit_mask = _mm_set_epi64x(0x7fbfdfeff7fbfdfe, 0x7fbfdfeff7fbfdfe);
  bytes = _mm_or_si128(bytes, bit_mask);

  return _mm_cmpeq_epi8(bytes, _mm_set1_epi64x(-1));
}

// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
  __m128 res_0 = _mm_hadd_ps(a, b);
  __m128 res_1 = _mm_hadd_ps(c, d);
  __m128 res = _mm_hadd_ps(res_0, res_1);
  res = _mm_hadd_ps(res, res);
  res = _mm_hadd_ps(res, res);

  return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

void quantize_row_q8_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__AVX2__) || defined(__AVX__)
  for (int vI = 0; vI < cNb; vI++) {
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps(pX);
    __m256 v1 = _mm256_loadu_ps(pX + 8);
    __m256 v2 = _mm256_loadu_ps(pX + 16);
    __m256 v3 = _mm256_loadu_ps(pX + 24);
    pX += 32;

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

    // Quantize these floats
    const float cD = cMaxScalar / 127.f;
    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);
    const float cId = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f;
    const __m256 cMul = _mm256_set1_ps(cId);

    // Apply the multiplier
    v0 = _mm256_mul_ps(v0, cMul);
    v1 = _mm256_mul_ps(v1, cMul);
    v2 = _mm256_mul_ps(v2, cMul);
    v3 = _mm256_mul_ps(v3, cMul);

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
    vI0 = _mm256_packs_epi32(vI0, vI1); // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
    vI2 = _mm256_packs_epi32(vI2, vI3); // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                        // Convert int16 to int8
    vI0 = _mm256_packs_epi16(vI0, vI2); // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12,
                                        // 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

    // We got our precious signed bytes, but the order is now wrong
    // These AVX2 pack instructions process 16-byte pieces independently
    // The following instruction is fixing the order
    const __m256i cPerm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    vI0 = _mm256_permutevar8x32_epi32(vI0, cPerm);

    _mm256_storeu_si256((__m256i *)vY[vI].qs, vI0);
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

    _mm_storeu_si128((__m128i *)(vY[vI].qs + 0), ni0);
    _mm_storeu_si128((__m128i *)(vY[vI].qs + 16), ni4);
#endif
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_0_ref(pX, vY, pK);
#endif
}

void quantize_row_q8_1(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK8_1 == 0);
  const int cNb = pK / QK8_1;

  block_q8_1 *GGML_RESTRICT vY = pVy;
#if defined(__AVX2__) || defined(__AVX__)
  for (int vI = 0; vI < cNb; vI++) {
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps(pX);
    __m256 v1 = _mm256_loadu_ps(pX + 8);
    __m256 v2 = _mm256_loadu_ps(pX + 16);
    __m256 v3 = _mm256_loadu_ps(pX + 24);
    pX += 32;

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

    // Quantize these floats
    const float cD = cMaxScalar / 127.f;
    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);
    const float cId = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f;
    const __m256 cMul = _mm256_set1_ps(cId);

    // Apply the multiplier
    v0 = _mm256_mul_ps(v0, cMul);
    v1 = _mm256_mul_ps(v1, cMul);
    v2 = _mm256_mul_ps(v2, cMul);
    v3 = _mm256_mul_ps(v3, cMul);

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
    // Compute the sum of the quants and set y[i].s
    vY[vI].s =
        GGML_CPU_FP32_TO_FP16(cD * fHsumI328(_mm256_add_epi32(_mm256_add_epi32(vI0, vI1), _mm256_add_epi32(vI2, vI3))));

    // Convert int32 to int16
    vI0 = _mm256_packs_epi32(vI0, vI1); // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
    vI2 = _mm256_packs_epi32(vI2, vI3); // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                        // Convert int16 to int8
    vI0 = _mm256_packs_epi16(vI0, vI2); // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12,
                                        // 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

    // We got our precious signed bytes, but the order is now wrong
    // These AVX2 pack instructions process 16-byte pieces independently
    // The following instruction is fixing the order
    const __m256i cPerm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    vI0 = _mm256_permutevar8x32_epi32(vI0, cPerm);

    _mm256_storeu_si256((__m256i *)vY[vI].qs, vI0);
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

    // Compute the sum of the quants and set y[i].s
    const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
    const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * fHsumI324(_mm_add_epi32(s0, s1)));

    // Convert int32 to int16
    ni0 = _mm_packs_epi32(ni0, ni1);
    ni2 = _mm_packs_epi32(ni2, ni3);
    ni4 = _mm_packs_epi32(ni4, ni5);
    ni6 = _mm_packs_epi32(ni6, ni7);
    // Convert int16 to int8
    ni0 = _mm_packs_epi16(ni0, ni2);
    ni4 = _mm_packs_epi16(ni4, ni6);

    _mm_storeu_si128((__m128i *)(vY[vI].qs + 0), ni0);
    _mm_storeu_si128((__m128i *)(vY[vI].qs + 16), ni4);
#endif
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_1_ref(pX, vY, pK);
#endif
}

// placeholder implementation for Apple targets
void quantize_row_q8_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q8_K_ref(pX, pY, pK);
}

//===================================== Dot products =================================

//
// Helper functions
//

#if __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
static inline __m256i fGetScaleShuffleQ3k(int pI) {
  static const uint8_t aKShuffle[128] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };
  return _mm256_loadu_si256((const __m256i *)aKShuffle + pI);
}
static inline __m256i fGetScaleShuffleK4(int pI) {
  static const uint8_t aKShuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_loadu_si256((const __m256i *)aKShuffle + pI);
}
static inline __m128i fGetScaleShuffle(int pI) {
  static const uint8_t aKShuffle[128] = {
      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
      3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
      6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
      9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
      13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128((const __m128i *)aKShuffle + pI);
}
#endif

void ggml_vec_dot_q1_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK1_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__AVX2__)
  const __m256i cOnes8 = _mm256_set1_epi8(1);
  const __m256i cOnes16 = _mm256_set1_epi16(1);
  const __m256i cByteShuf =
      _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
  const __m256i cBitMasks = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)-128, 1, 2, 4, 8, 16, 32, 64, (char)-128, 1,
                                             2, 4, 8, 16, 32, 64, (char)-128, 1, 2, 4, 8, 16, 32, 64, (char)-128);
  const __m256i cZero = _mm256_setzero_si256();
  __m256 vAcc = _mm256_setzero_ps();

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    const uint32_t *GGML_RESTRICT vQs32 = (const uint32_t *)vX[vIb].qs;
    const block_q8_0 *GGML_RESTRICT vYPtr = &vY[vIb * 4];

    __m256 vAccBlock;
    {
      const __m256i cQy = _mm256_loadu_si256((const __m256i *)vYPtr[0].qs);
      const __m256i cSm = _mm256_cmpeq_epi8(
          _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)vQs32[0]), cByteShuf), cBitMasks), cZero);
      const __m256i cSy = _mm256_sub_epi8(_mm256_xor_si256(cQy, cSm), cSm);
      const __m256i cS32 = _mm256_madd_epi16(_mm256_maddubs_epi16(cOnes8, cSy), cOnes16);
      vAccBlock = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vYPtr[0].d)), _mm256_cvtepi32_ps(cS32));
    }
    for (int vK = 1; vK < 4; ++vK) {
      const __m256i cQy = _mm256_loadu_si256((const __m256i *)vYPtr[vK].qs);
      const __m256i cSm = _mm256_cmpeq_epi8(
          _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)vQs32[vK]), cByteShuf), cBitMasks), cZero);
      const __m256i cSy = _mm256_sub_epi8(_mm256_xor_si256(cQy, cSm), cSm);
      const __m256i cS32 = _mm256_madd_epi16(_mm256_maddubs_epi16(cOnes8, cSy), cOnes16);
      vAccBlock =
          _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vYPtr[vK].d)), _mm256_cvtepi32_ps(cS32), vAccBlock);
    }
    vAcc = _mm256_fmadd_ps(_mm256_set1_ps(cD0), vAccBlock, vAcc);
  }

  *pS = fHsumFloat8(vAcc);
#elif defined(__AVX__)
  const __m128i cOnes8 = _mm_set1_epi8(1);
  const __m128i cOnes16 = _mm_set1_epi16(1);
  const __m128i cZero = _mm_setzero_si128();
  __m256 vAcc = _mm256_setzero_ps();

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    const block_q8_0 *GGML_RESTRICT vYPtr = &vY[vIb * 4];
    __m256 vAccBlock;
    {
      const __m256i bit_mask = fBytesFromBits32(&vX[vIb].qs[0]);
      const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
      const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
      const __m128i qy_0 = _mm_loadu_si128((const __m128i *)&vYPtr[0].qs[0]);
      const __m128i qy_1 = _mm_loadu_si128((const __m128i *)&vYPtr[0].qs[16]);
      const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, cZero);
      const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, cZero);
      const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
      const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
      const __m128i sum16_0 = _mm_maddubs_epi16(cOnes8, sy_0);
      const __m128i sum16_1 = _mm_maddubs_epi16(cOnes8, sy_1);
      const __m128i sum32_0 = _mm_madd_epi16(sum16_0, cOnes16);
      const __m128i sum32_1 = _mm_madd_epi16(sum16_1, cOnes16);
      const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
      vAccBlock = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vYPtr[0].d)), q);
    }
    for (int vK = 1; vK < 4; ++vK) {
      const __m256i bit_mask = fBytesFromBits32(&vX[vIb].qs[(vK) * 4]);
      const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
      const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
      const __m128i qy_0 = _mm_loadu_si128((const __m128i *)&vYPtr[(vK)].qs[0]);
      const __m128i qy_1 = _mm_loadu_si128((const __m128i *)&vYPtr[(vK)].qs[16]);
      const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, cZero);
      const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, cZero);
      const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
      const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
      const __m128i sum16_0 = _mm_maddubs_epi16(cOnes8, sy_0);
      const __m128i sum16_1 = _mm_maddubs_epi16(cOnes8, sy_1);
      const __m128i sum32_0 = _mm_madd_epi16(sum16_0, cOnes16);
      const __m128i sum32_1 = _mm_madd_epi16(sum16_1, cOnes16);
      const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
      vAccBlock = _mm256_add_ps(vAccBlock, _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vYPtr[(vK)].d)), q));
    }
#undef Q1_AVX_BLOCK

    vAcc = _mm256_add_ps(vAcc, _mm256_mul_ps(_mm256_set1_ps(cD0), vAccBlock));
  }

  *pS = fHsumFloat8(vAcc);
#elif defined(__SSSE3__)
  const __m128i cOnes8 = _mm_set1_epi8(1);
  const __m128i cOnes16 = _mm_set1_epi16(1);
  const __m128i cZero = _mm_setzero_si128();
  __m128 acc_0 = _mm_setzero_ps();
  __m128 acc_1 = _mm_setzero_ps();
  __m128 acc_2 = _mm_setzero_ps();
  __m128 acc_3 = _mm_setzero_ps();

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const __m128 cD0 = _mm_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    const block_q8_0 *GGML_RESTRICT vYPtr = &vY[vIb * 4];

#define Q1_SSSE3_BLOCK(QS_OFF, Y_IDX, ACC)                                                                             \
  {                                                                                                                    \
    const __m128i bit_mask_0 = bytes_from_bits_16(&vX[vIb].qs[(QS_OFF) + 0]);                                          \
    const __m128i bit_mask_1 = bytes_from_bits_16(&vX[vIb].qs[(QS_OFF) + 2]);                                          \
    const __m128i qy_0 = _mm_loadu_si128((const __m128i *)&vYPtr[(Y_IDX)].qs[0]);                                      \
    const __m128i qy_1 = _mm_loadu_si128((const __m128i *)&vYPtr[(Y_IDX)].qs[16]);                                     \
    const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, cZero);                                                     \
    const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, cZero);                                                     \
    const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);                                  \
    const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);                                  \
    const __m128i sum_0 = _mm_madd_epi16(_mm_maddubs_epi16(cOnes8, sy_0), cOnes16);                                    \
    const __m128i sum_1 = _mm_madd_epi16(_mm_maddubs_epi16(cOnes8, sy_1), cOnes16);                                    \
    const __m128 q = _mm_cvtepi32_ps(_mm_add_epi32(sum_0, sum_1));                                                     \
    (ACC) = _mm_add_ps((ACC), _mm_mul_ps(_mm_mul_ps(cD0, _mm_set1_ps(GGML_CPU_FP16_TO_FP32(vYPtr[(Y_IDX)].d))), q));   \
  }
    Q1_SSSE3_BLOCK(0, 0, acc_0)
    Q1_SSSE3_BLOCK(4, 1, acc_1)
    Q1_SSSE3_BLOCK(8, 2, acc_2)
    Q1_SSSE3_BLOCK(12, 3, acc_3)
#undef Q1_SSSE3_BLOCK
  }

  *pS = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q1_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q4_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();

  // Main loop
  for (; vIb < cNb; ++vIb) {
    /* Compute combined scale for the block */
    const __m256 cD = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i cOff = _mm256_set1_epi8(8);
    vQx = _mm256_sub_epi8(vQx, cOff);

    __m256i vQy = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    /* Multiply q with scale and accumulate */
    vAcc = _mm256_fmadd_ps(cD, cQ, vAcc);
  }

  vSumf = fHsumFloat8(vAcc);
#elif defined(__AVX__)
  __m256 accum = _mm256_setzero_ps();
  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)vX[vIb + 0].qs);
    const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs);
    const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs + 1);
    const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs);
    const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs + 1);

    const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
    const __m128i q4b_1_1 =
        _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
    const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
    const __m128i q4b_2_1 =
        _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

    const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
    const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
    const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
    const __m256 p = fSumI16PairsFloat(p_2, p_1);

    const __m256 deltas = quad_fp16_delta_float(vX[vIb].d, vY[vIb].d, vX[vIb + 1].d, vY[vIb + 1].d);
    accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
  }

  vSumf = fHsumFloat8(accum);
#elif defined(__SSSE3__)
  // set constants
  const __m128i lowMask = _mm_set1_epi8(0xF);
  const __m128i cOff = _mm_set1_epi8(8);

  // Initialize accumulator with zeros
  __m128 acc_0 = _mm_setzero_ps();
  __m128 acc_1 = _mm_setzero_ps();
  __m128 acc_2 = _mm_setzero_ps();
  __m128 acc_3 = _mm_setzero_ps();

  for (; vIb + 1 < cNb; vIb += 2) {
    _mm_prefetch(&vX[vIb] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&vY[vIb] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)vX[vIb].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i *)vY[vIb].qs);
    bx_0 = _mm_sub_epi8(bx_0, cOff);
    const __m128i i32_0 = fMulSumI8Pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i *)(vY[vIb].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, cOff);
    const __m128i i32_1 = fMulSumI8Pairs(bx_1, by_1);

    _mm_prefetch(&vX[vIb] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&vY[vIb] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs);
    bx_2 = _mm_sub_epi8(bx_2, cOff);
    const __m128i i32_2 = fMulSumI8Pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i *)(vY[vIb + 1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, cOff);
    const __m128i i32_3 = fMulSumI8Pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    __m128 p0_d = _mm_mul_ps(d_0_1, p0);
    __m128 p1_d = _mm_mul_ps(d_0_1, p1);
    __m128 p2_d = _mm_mul_ps(d_2_3, p2);
    __m128 p3_d = _mm_mul_ps(d_2_3, p3);

    // Acummulate
    acc_0 = _mm_add_ps(p0_d, acc_0);
    acc_1 = _mm_add_ps(p1_d, acc_1);
    acc_2 = _mm_add_ps(p2_d, acc_2);
    acc_3 = _mm_add_ps(p3_d, acc_3);
  }

  vSumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

#endif
  for (; vIb < cNb; ++vIb) {
    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cV0 = (vX[vIb].qs[vJ] & 0x0F) - 8;
      const int cV1 = (vX[vIb].qs[vJ] >> 4) - 8;

      vSumi0 += (cV0 * vY[vIb].qs[vJ]);
      vSumi1 += (cV1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += vSumi * GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d);
  }

  *pS = vSumf;
}

void ggml_vec_dot_q4_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

  int vIb = 0;

#if defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();

  float vSumms = 0;

  // Main loop
  for (; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    const float cD1 = GGML_CPU_FP16_TO_FP32(vY[vIb].d);

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    const __m256 cD0v = _mm256_set1_ps(cD0);
    const __m256 cD1v = _mm256_set1_ps(cD1);

    // Compute combined scales
    const __m256 cD0d1 = _mm256_mul_ps(cD0v, cD1v);

    // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
    const __m256i cQx = fBytesFromNibbles32(vX[vIb].qs);
    const __m256i cQy = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cXy = fMulSumUs8PairsFloat(cQx, cQy);

    // Accumulate d0*d1*x*y
#if defined(__AVX2__)
    vAcc = _mm256_fmadd_ps(cD0d1, cXy, vAcc);
#else
    vAcc = _mm256_add_ps(_mm256_mul_ps(cD0d1, cXy), vAcc);
#endif
  }

  *pS = fHsumFloat8(vAcc) + vSumms;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  ggml_vec_dot_q4_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_mxfp4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_MXFP4 == 0);
  static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

  const block_mxfp4 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_MXFP4;

  int vIb = 0;
  float vSumf = 0;

#if defined __AVX2__

  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_fp4);
  const __m128i cM4b = _mm_set1_epi8(0x0f);
  const __m256i cMone = _mm256_set1_epi16(1);

  __m256 vAccum1 = _mm256_setzero_ps();
  __m256 vAccum2 = _mm256_setzero_ps();

  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vX[vIb + 0].qs);
    const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m256i cQ8b1 = _mm256_loadu_si256((const __m256i *)vY[vIb + 0].qs);
    const __m256i cQ8b2 = _mm256_loadu_si256((const __m256i *)vY[vIb + 1].qs);
    const __m256i cQ4b1 =
        MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b)),
                        _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b)));
    const __m256i cQ4b2 =
        MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b)),
                        _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b)));
    const __m256i cP161 = fMulAddEpi8(cQ4b1, cQ8b1);
    const __m256i cP162 = fMulAddEpi8(cQ4b2, cQ8b2);
    const __m256i cP1 = _mm256_madd_epi16(cP161, cMone);
    const __m256i cP2 = _mm256_madd_epi16(cP162, cMone);
    const __m256 cScale0 =
        _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * GGML_CPU_E8M0_TO_FP32_HALF(vX[vIb + 0].e));
    const __m256 cScale1 =
        _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * GGML_CPU_E8M0_TO_FP32_HALF(vX[vIb + 1].e));
    vAccum1 = _mm256_fmadd_ps(cScale0, _mm256_cvtepi32_ps(cP1), vAccum1);
    vAccum2 = _mm256_fmadd_ps(cScale1, _mm256_cvtepi32_ps(cP2), vAccum2);
  }

  vSumf = fHsumFloat8(_mm256_add_ps(vAccum1, vAccum2));

#elif defined __AVX__
  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_fp4);
  const __m128i cM4b = _mm_set1_epi8(0x0f);

  __m256 accum = _mm256_setzero_ps();
  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vX[vIb + 0].qs);
    const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs);
    const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs + 1);
    const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs);
    const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs + 1);

    const __m128i q4b_1_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b));
    const __m128i q4b_1_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b));
    const __m128i q4b_2_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b));
    const __m128i q4b_2_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b));

    const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
    const __m256 deltas = quad_mx_delta_float(vX[vIb].e, vY[vIb].d, vX[vIb + 1].e, vY[vIb + 1].d);
    accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
  }

  vSumf = fHsumFloat8(accum);

#endif
  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_CPU_E8M0_TO_FP32_HALF(vX[vIb].e);
    int vSumi1 = 0;
    int vSumi2 = 0;
    for (int vJ = 0; vJ < QK_MXFP4 / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_fp4[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK_MXFP4 / 2] * kvalues_fp4[vX[vIb].qs[vJ] >> 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;
}

void ggml_vec_dot_nvfp4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_NVFP4 == 0);

  const block_nvfp4 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_NVFP4;
  int vIb = 0;
  float vSumf = 0;

#if defined(__AVX2__)

  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_fp4);
  const __m128i cM4b = _mm_set1_epi8(0x0f);
  const __m256i cMone = _mm256_set1_epi16(1);

  __m256 vAccum = _mm256_setzero_ps();
  for (; vIb < cNb; vIb++) {

    const __m128i cQ4bits01 = _mm_loadu_si128((const __m128i *)(vX[vIb].qs + 0));
    const __m128i cQ4bits23 = _mm_loadu_si128((const __m128i *)(vX[vIb].qs + 16));

    const __m256i cQ801 = _mm256_loadu_si256((const __m256i *)vY[2 * vIb + 0].qs);
    const __m256i cQ823 = _mm256_loadu_si256((const __m256i *)vY[2 * vIb + 1].qs);

    const __m128i cQ401Lo = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits01, cM4b));
    const __m128i cQ401Hi = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits01, 4), cM4b));
    const __m128i cQ423Lo = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits23, cM4b));
    const __m128i cQ423Hi = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits23, 4), cM4b));

    // reordering
    const __m256i cQ401 = MM256_SET_M128I(_mm_unpackhi_epi64(cQ401Lo, cQ401Hi), _mm_unpacklo_epi64(cQ401Lo, cQ401Hi));
    const __m256i cQ423 = MM256_SET_M128I(_mm_unpackhi_epi64(cQ423Lo, cQ423Hi), _mm_unpacklo_epi64(cQ423Lo, cQ423Hi));

    const __m256i cP01 = fMulAddEpi8(cQ401, cQ801);
    const __m256i cP1 = _mm256_madd_epi16(cP01, cMone);

    const __m256i cP23 = fMulAddEpi8(cQ423, cQ823);
    const __m256i cP2 = _mm256_madd_epi16(cP23, cMone);

    const float cDy0 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb].d);
    const float cDy1 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + 1].d);

    const float cS0 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[0]) * cDy0;
    const float cS1 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[1]) * cDy0;
    const float cS2 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[2]) * cDy1;
    const float cS3 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[3]) * cDy1;

    const __m256 cScales01 = _mm256_set_m128(_mm_set1_ps(cS1), _mm_set1_ps(cS0));
    const __m256 cScales23 = _mm256_set_m128(_mm_set1_ps(cS3), _mm_set1_ps(cS2));

    vAccum = _mm256_fmadd_ps(cScales01, _mm256_cvtepi32_ps(cP1), vAccum);
    vAccum = _mm256_fmadd_ps(cScales23, _mm256_cvtepi32_ps(cP2), vAccum);
  }
  vSumf = fHsumFloat8(vAccum);

#elif defined(__AVX__)

  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_fp4);
  const __m128i cM4b = _mm_set1_epi8(0x0f);

  __m256 vAccum = _mm256_setzero_ps();
  for (; vIb < cNb; vIb++) {

    const __m128i cQ4bits01 = _mm_loadu_si128((const __m128i *)(vX[vIb].qs + 0));
    const __m128i cQ4bits23 = _mm_loadu_si128((const __m128i *)(vX[vIb].qs + 16));

    const __m128i q8_0 = _mm_loadu_si128((const __m128i *)(vY[2 * vIb + 0].qs + 0));
    const __m128i q8_1 = _mm_loadu_si128((const __m128i *)(vY[2 * vIb + 0].qs + 16));
    const __m128i q8_2 = _mm_loadu_si128((const __m128i *)(vY[2 * vIb + 1].qs + 0));
    const __m128i q8_3 = _mm_loadu_si128((const __m128i *)(vY[2 * vIb + 1].qs + 16));

    const __m128i cQ401Lo = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits01, cM4b));
    const __m128i cQ401Hi = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits01, 4), cM4b));
    const __m128i cQ423Lo = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits23, cM4b));
    const __m128i cQ423Hi = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits23, 4), cM4b));

    const __m128i q4_0 = _mm_unpacklo_epi64(cQ401Lo, cQ401Hi);
    const __m128i q4_1 = _mm_unpackhi_epi64(cQ401Lo, cQ401Hi);
    const __m128i q4_2 = _mm_unpacklo_epi64(cQ423Lo, cQ423Hi);
    const __m128i q4_3 = _mm_unpackhi_epi64(cQ423Lo, cQ423Hi);

    const __m128i p0_i32 = fMulSumI8Pairs(q4_0, q8_0);
    const __m128i p1_i32 = fMulSumI8Pairs(q4_1, q8_1);
    const __m128i p2_i32 = fMulSumI8Pairs(q4_2, q8_2);
    const __m128i p3_i32 = fMulSumI8Pairs(q4_3, q8_3);

    const __m128 p0 = _mm_cvtepi32_ps(p0_i32);
    const __m128 p1 = _mm_cvtepi32_ps(p1_i32);
    const __m128 p2 = _mm_cvtepi32_ps(p2_i32);
    const __m128 p3 = _mm_cvtepi32_ps(p3_i32);

    const __m256 cP01 = _mm256_set_m128(p1, p0);
    const __m256 cP23 = _mm256_set_m128(p3, p2);

    const float cDy0 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb].d);
    const float cDy1 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + 1].d);

    const float cS0 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[0]) * cDy0;
    const float cS1 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[1]) * cDy0;
    const float cS2 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[2]) * cDy1;
    const float cS3 = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[3]) * cDy1;

    const __m256 cScales01 = _mm256_set_m128(_mm_set1_ps(cS1), _mm_set1_ps(cS0));
    const __m256 cScales23 = _mm256_set_m128(_mm_set1_ps(cS3), _mm_set1_ps(cS2));

    vAccum = _mm256_add_ps(vAccum, _mm256_mul_ps(cP01, cScales01));
    vAccum = _mm256_add_ps(vAccum, _mm256_mul_ps(cP23, cScales23));
  }
  vSumf = fHsumFloat8(vAccum);

#endif

  for (; vIb < cNb; ++vIb) {
    for (int vSIdx = 0; vSIdx < 4; ++vSIdx) {
      const float cD = GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[vSIdx]);
      const int cQ8Block = vSIdx / 2;
      const int cQ8Off = (vSIdx % 2) * QK_NVFP4_SUB;
      const float cDy = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + cQ8Block].d);

      int vSumiLo = 0, vSumiHi = 0;
      for (int vJ = 0; vJ < QK_NVFP4_SUB / 2; ++vJ) {
        const uint8_t cQv = vX[vIb].qs[vSIdx * (QK_NVFP4_SUB / 2) + vJ];
        vSumiLo += vY[2 * vIb + cQ8Block].qs[cQ8Off + vJ + 0] * kvalues_fp4[cQv & 0xf];
        vSumiHi += vY[2 * vIb + cQ8Block].qs[cQ8Off + vJ + QK_NVFP4_SUB / 2] * kvalues_fp4[cQv >> 4];
      }

      vSumf += cDy * cD * (vSumiLo + vSumiHi);
    }
  }
  *pS = vSumf;
}

void ggml_vec_dot_q5_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  int vIb = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();

  // Main loop
  for (; vIb < cNb; ++vIb) {
    /* Compute combined scale for the block */
    const __m256 cD = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);
    __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    vBxhi = _mm256_andnot_si256(vBxhi, _mm256_set1_epi8((char)0xF0));
    vQx = _mm256_or_si256(vQx, vBxhi);

    __m256i vQy = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    /* Multiply q with scale and accumulate */
    vAcc = _mm256_fmadd_ps(cD, cQ, vAcc);
  }

  *pS = fHsumFloat8(vAcc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8((char)0xF0);

  // Main loop
  for (; vIb < cNb; ++vIb) {
    /* Compute combined scale for the block */
    const __m256 cD = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    __m256i bx_0 = fBytesFromNibbles32(vX[vIb].qs);
    const __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    __m128i bxhil = _mm256_castsi256_si128(vBxhi);
    __m128i bxhih = _mm256_extractf128_si256(vBxhi, 1);
    bxhil = _mm_andnot_si128(bxhil, mask);
    bxhih = _mm_andnot_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx_0);
    __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx_0 = MM256_SET_M128I(bxh, bxl);

    const __m256i by_0 = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumI8PairsFloat(bx_0, by_0);

    /* Multiply q with scale and accumulate */
    vAcc = _mm256_add_ps(_mm256_mul_ps(cD, cQ), vAcc);
  }

  *pS = fHsumFloat8(vAcc);
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q5_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  int vIb = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_1);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();

  float vSumms = 0.0f;

  // Main loop
  for (; vIb < cNb; ++vIb) {
    const __m256 cDx = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d));

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);
    __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    vBxhi = _mm256_and_si256(vBxhi, _mm256_set1_epi8(0x10));
    vQx = _mm256_or_si256(vQx, vBxhi);

    const __m256 cDy = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    const __m256i cQy = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumUs8PairsFloat(vQx, cQy);

    vAcc = _mm256_fmadd_ps(cQ, _mm256_mul_ps(cDx, cDy), vAcc);
  }

  *pS = fHsumFloat8(vAcc) + vSumms;
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8(0x10);

  float vSumms = 0.0f;

  // Main loop
  for (; vIb < cNb; ++vIb) {
    const __m256 cDx = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d));

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    __m256i bx_0 = fBytesFromNibbles32(vX[vIb].qs);
    const __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    __m128i bxhil = _mm256_castsi256_si128(vBxhi);
    __m128i bxhih = _mm256_extractf128_si256(vBxhi, 1);
    bxhil = _mm_and_si128(bxhil, mask);
    bxhih = _mm_and_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx_0);
    __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx_0 = MM256_SET_M128I(bxh, bxl);

    const __m256 cDy = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    const __m256i by_0 = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumUs8PairsFloat(bx_0, by_0);

    vAcc = _mm256_add_ps(_mm256_mul_ps(cQ, _mm256_mul_ps(cDx, cDy)), vAcc);
  }

  *pS = fHsumFloat8(vAcc) + vSumms;
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q5_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q8_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q8_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 vAcc = _mm256_setzero_ps();

  // Main loop
  for (; vIb < cNb; ++vIb) {
    // Compute combined scale for the block
    const __m256 cD = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    __m256i vQx = _mm256_loadu_si256((const __m256i *)vX[vIb].qs);
    __m256i vQy = _mm256_loadu_si256((const __m256i *)vY[vIb].qs);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    // Multiply q with scale and accumulate
    vAcc = _mm256_fmadd_ps(cD, cQ, vAcc);
  }

  vSumf = fHsumFloat8(vAcc);
#elif defined(__AVX__)
  __m256 accum = _mm256_setzero_ps();

  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i qx_1_0 = _mm_loadu_si128((const __m128i *)vX[vIb].qs);
    const __m128i qx_1_1 = _mm_loadu_si128((const __m128i *)vX[vIb].qs + 1);
    const __m128i qx_2_0 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m128i qx_2_1 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs + 1);
    const __m128i qy_1_0 = _mm_loadu_si128((const __m128i *)vY[vIb].qs);
    const __m128i qy_1_1 = _mm_loadu_si128((const __m128i *)vY[vIb].qs + 1);
    const __m128i qy_2_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs);
    const __m128i qy_2_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs + 1);

    const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
    const __m256 deltas = quad_fp16_delta_float(vX[vIb].d, vY[vIb].d, vX[vIb + 1].d, vY[vIb + 1].d);
    accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
  }

  vSumf = fHsumFloat8(accum);
#endif
  for (; vIb < cNb; ++vIb) {
    int vSumi = 0;

    for (int vJ = 0; vJ < cQk; vJ++) {
      vSumi += vX[vIb].qs[vJ] * vY[vIb].qs[vJ];
    }

    vSumf += vSumi * (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));
  }

  *pS = vSumf;
}

void ggml_vec_dot_tq1_0_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)
  __m256 vSumf = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {
    // 16-bit sums
    __m256i vSumi0 = _mm256_setzero_si256();
    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();

    // first 32 bytes of 5 elements
    {
      __m256i vQx0 = _mm256_loadu_si256((const __m256i *)(vX[vI].qs));
      // 8-bit multiplies with shifts, masks and adds
      __m256i vQx1 = _mm256_add_epi8(vQx0, _mm256_add_epi8(vQx0, vQx0));                                        // 1 * 3
      __m256i vQx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(vQx0, 3), _mm256_set1_epi8(-8)), vQx0); // 1 * 9
      __m256i vQx3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(vQx1, 3), _mm256_set1_epi8(-8)), vQx1); // 3 * 9
      __m256i vQx4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(vQx2, 3), _mm256_set1_epi8(-8)), vQx2); // 9 * 9

      // TODO: can _mm256_mulhi_epu16 be faster even if 16-bits?

      // Cancel the +1 from avg so that it behaves like a halving add
      vQx0 = _mm256_subs_epu8(vQx0, _mm256_set1_epi8(1));
      vQx1 = _mm256_subs_epu8(vQx1, _mm256_set1_epi8(1));
      vQx2 = _mm256_subs_epu8(vQx2, _mm256_set1_epi8(1));
      vQx3 = _mm256_subs_epu8(vQx3, _mm256_set1_epi8(1));
      vQx4 = _mm256_subs_epu8(vQx4, _mm256_set1_epi8(1));
      // Multiply by 3 and get the top 2 bits
      vQx0 = _mm256_avg_epu8(vQx0, _mm256_avg_epu8(vQx0, _mm256_setzero_si256()));
      vQx1 = _mm256_avg_epu8(vQx1, _mm256_avg_epu8(vQx1, _mm256_setzero_si256()));
      vQx2 = _mm256_avg_epu8(vQx2, _mm256_avg_epu8(vQx2, _mm256_setzero_si256()));
      vQx3 = _mm256_avg_epu8(vQx3, _mm256_avg_epu8(vQx3, _mm256_setzero_si256()));
      vQx4 = _mm256_avg_epu8(vQx4, _mm256_avg_epu8(vQx4, _mm256_setzero_si256()));
      vQx0 = _mm256_and_si256(_mm256_srli_epi16(vQx0, 6), _mm256_set1_epi8(3));
      vQx1 = _mm256_and_si256(_mm256_srli_epi16(vQx1, 6), _mm256_set1_epi8(3));
      vQx2 = _mm256_and_si256(_mm256_srli_epi16(vQx2, 6), _mm256_set1_epi8(3));
      vQx3 = _mm256_and_si256(_mm256_srli_epi16(vQx3, 6), _mm256_set1_epi8(3));
      vQx4 = _mm256_and_si256(_mm256_srli_epi16(vQx4, 6), _mm256_set1_epi8(3));

      const __m256i cQy0 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 0));
      const __m256i cQy1 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 32));
      const __m256i cQy2 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 64));
      const __m256i cQy3 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 96));
      const __m256i cQy4 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 128));

      vQx0 = _mm256_maddubs_epi16(vQx0, cQy0);
      vQx1 = _mm256_maddubs_epi16(vQx1, cQy1);
      vQx2 = _mm256_maddubs_epi16(vQx2, cQy2);
      vQx3 = _mm256_maddubs_epi16(vQx3, cQy3);
      vQx4 = _mm256_maddubs_epi16(vQx4, cQy4);

      vSumi0 = _mm256_add_epi16(vSumi0, _mm256_add_epi16(vQx0, vQx1));
      vSumi1 = _mm256_add_epi16(vSumi1, _mm256_add_epi16(vQx2, vQx3));
      vSumi2 = _mm256_add_epi16(vSumi2, vQx4);
    }

    // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
    {
      __m128i vQx0 = _mm_loadu_si128((const __m128i *)(vX[vI].qs + 32));
      uint32_t vQh;
      memcpy(&vQh, vX[vI].qh, sizeof(vQh)); // potentially unaligned
      __m256i vQx5L = _mm256_cvtepu8_epi16(_mm_set1_epi32(vQh));
      __m128i vQx1 = _mm_add_epi8(vQx0, _mm_add_epi8(vQx0, vQx0));                                  // 1 * 3
      __m128i vQx2 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(vQx0, 3), _mm_set1_epi8(-8)), vQx0); // 1 * 9
      __m128i vQx3 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(vQx1, 3), _mm_set1_epi8(-8)), vQx1); // 3 * 9
      __m128i vQx4 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(vQx2, 3), _mm_set1_epi8(-8)), vQx2); // 9 * 9
      __m256i vQx01 = MM256_SET_M128I(vQx1, vQx0);
      __m256i vQx23 = MM256_SET_M128I(vQx3, vQx2);

      // avx2 does not have 8-bit multiplies, so 16-bit it is.
      vQx5L = _mm256_mullo_epi16(vQx5L, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
      vQx5L = _mm256_and_si256(vQx5L, _mm256_set1_epi16(0xFF));
      __m128i vQx5 = _mm_packus_epi16(_mm256_castsi256_si128(vQx5L), _mm256_extracti128_si256(vQx5L, 1));

      __m256i vQx45 = MM256_SET_M128I(vQx5, vQx4);

      // Cancel the +1 from avg so that it behaves like a halving add
      vQx01 = _mm256_subs_epu8(vQx01, _mm256_set1_epi8(1));
      vQx23 = _mm256_subs_epu8(vQx23, _mm256_set1_epi8(1));
      vQx45 = _mm256_subs_epu8(vQx45, _mm256_set1_epi8(1));
      // Multiply by 3 and get the top 2 bits
      vQx01 = _mm256_avg_epu8(vQx01, _mm256_avg_epu8(vQx01, _mm256_setzero_si256()));
      vQx23 = _mm256_avg_epu8(vQx23, _mm256_avg_epu8(vQx23, _mm256_setzero_si256()));
      vQx45 = _mm256_avg_epu8(vQx45, _mm256_avg_epu8(vQx45, _mm256_setzero_si256()));
      vQx01 = _mm256_and_si256(_mm256_srli_epi16(vQx01, 6), _mm256_set1_epi8(3));
      vQx23 = _mm256_and_si256(_mm256_srli_epi16(vQx23, 6), _mm256_set1_epi8(3));
      vQx45 = _mm256_and_si256(_mm256_srli_epi16(vQx45, 6), _mm256_set1_epi8(3));

      const __m256i cQy01 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 160));
      const __m256i cQy23 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 192));
      const __m256i cQy45 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + 224));

      vQx01 = _mm256_maddubs_epi16(vQx01, cQy01);
      vQx23 = _mm256_maddubs_epi16(vQx23, cQy23);
      vQx45 = _mm256_maddubs_epi16(vQx45, cQy45);

      vSumi0 = _mm256_add_epi16(vSumi0, vQx01);
      vSumi1 = _mm256_add_epi16(vSumi1, vQx23);
      vSumi2 = _mm256_add_epi16(vSumi2, vQx45);
    }

    const __m256i cYsum = _mm256_loadu_si256((const __m256i *)vY[vI].bsums);
    const __m256 cD = _mm256_set1_ps(vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d));

    vSumi0 = _mm256_sub_epi16(vSumi0, cYsum);
    vSumi0 = _mm256_add_epi16(vSumi0, _mm256_add_epi16(vSumi1, vSumi2));
    vSumi0 = _mm256_madd_epi16(vSumi0, _mm256_set1_epi16(1));

    vSumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vSumi0), cD), vSumf);
  }

  *pS = fHsumFloat8(vSumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_tq1_0_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_tq2_0_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq2_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)
  __m256 vSumf = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {
    // 16-bit sums, because 256*127 still fits
    __m256i vSumi0 = _mm256_setzero_si256();
    __m256i vSumi1 = _mm256_setzero_si256();

    for (size_t vJ = 0; vJ < sizeof(vX->qs); vJ += 32) {
      __m256i vQx0 = _mm256_loadu_si256((const __m256i *)(vX[vI].qs + vJ));
      __m256i vQx1 = _mm256_srli_epi16(vQx0, 2);
      __m256i vQx2 = _mm256_srli_epi16(vQx0, 4);
      __m256i vQx3 = _mm256_srli_epi16(vQx0, 6);

      // 0, 1, 2 (should not be 3)
      vQx0 = _mm256_and_si256(vQx0, _mm256_set1_epi8(3));
      vQx1 = _mm256_and_si256(vQx1, _mm256_set1_epi8(3));
      vQx2 = _mm256_and_si256(vQx2, _mm256_set1_epi8(3));
      vQx3 = _mm256_and_si256(vQx3, _mm256_set1_epi8(3));

      const __m256i cQy0 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + vJ * 4 + 0));
      const __m256i cQy1 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + vJ * 4 + 32));
      const __m256i cQy2 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + vJ * 4 + 64));
      const __m256i cQy3 = _mm256_loadu_si256((const __m256i *)(vY[vI].qs + vJ * 4 + 96));

      vQx0 = _mm256_maddubs_epi16(vQx0, cQy0);
      vQx1 = _mm256_maddubs_epi16(vQx1, cQy1);
      vQx2 = _mm256_maddubs_epi16(vQx2, cQy2);
      vQx3 = _mm256_maddubs_epi16(vQx3, cQy3);

      vSumi0 = _mm256_add_epi16(vSumi0, _mm256_add_epi16(vQx0, vQx1));
      vSumi1 = _mm256_add_epi16(vSumi1, _mm256_add_epi16(vQx2, vQx3));
    }

    const __m256i cYsum = _mm256_loadu_si256((const __m256i *)vY[vI].bsums);
    const __m256 cD = _mm256_set1_ps(vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d));

    vSumi0 = _mm256_add_epi16(vSumi0, vSumi1);
    vSumi0 = _mm256_sub_epi16(vSumi0, cYsum);
    vSumi0 = _mm256_madd_epi16(vSumi0, _mm256_set1_epi16(1));

    vSumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vSumi0), cD), vSumf);
  }

  *pS = fHsumFloat8(vSumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_tq2_0_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __AVX2__

  const __m256i cM3 = _mm256_set1_epi8(3);
  const __m128i cM4 = _mm_set1_epi8(0xF);

  __m256 vAcc = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m128i cMinsAndScales = _mm_loadu_si128((const __m128i *)vX[vI].scales);
    const __m128i cScales8 = _mm_and_si128(cMinsAndScales, cM4);
    const __m128i cMins8 = _mm_and_si128(_mm_srli_epi16(cMinsAndScales, 4), cM4);
    const __m256i cMins = _mm256_cvtepi8_epi16(cMins8);
    const __m256i cProd = _mm256_madd_epi16(cMins, _mm256_loadu_si256((const __m256i *)vY[vI].bsums));

    vAcc = _mm256_fmadd_ps(_mm256_broadcast_ss(&cDmin), _mm256_cvtepi32_ps(cProd), vAcc);

    const __m256i cAllScales = _mm256_cvtepi8_epi16(cScales8);
    const __m128i cLScales = _mm256_extracti128_si256(cAllScales, 0);
    const __m128i cHScales = _mm256_extracti128_si256(cAllScales, 1);
    const __m256i cScales[2] = {MM256_SET_M128I(cLScales, cLScales), MM256_SET_M128I(cHScales, cHScales)};

    __m256i vSumi = _mm256_setzero_si256();

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const __m256i cQ2bits = _mm256_loadu_si256((const __m256i *)vQ2);
      vQ2 += 32;

      const __m256i cQ80 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ83 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      const __m256i cQ20 = _mm256_and_si256(cQ2bits, cM3);
      const __m256i cQ21 = _mm256_and_si256(_mm256_srli_epi16(cQ2bits, 2), cM3);
      const __m256i cQ22 = _mm256_and_si256(_mm256_srli_epi16(cQ2bits, 4), cM3);
      const __m256i cQ23 = _mm256_and_si256(_mm256_srli_epi16(cQ2bits, 6), cM3);

      __m256i vP0 = _mm256_maddubs_epi16(cQ20, cQ80);
      __m256i vP1 = _mm256_maddubs_epi16(cQ21, cQ81);
      __m256i vP2 = _mm256_maddubs_epi16(cQ22, cQ82);
      __m256i vP3 = _mm256_maddubs_epi16(cQ23, cQ83);

      vP0 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(0)), vP0);
      vP1 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(1)), vP1);
      vP2 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(2)), vP2);
      vP3 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(3)), vP3);

      vP0 = _mm256_add_epi32(vP0, vP1);
      vP2 = _mm256_add_epi32(vP2, vP3);

      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(vP0, vP2));
    }

    vAcc = _mm256_fmadd_ps(_mm256_broadcast_ss(&cD), _mm256_cvtepi32_ps(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#elif defined __AVX__

  const __m128i cM3 = _mm_set1_epi8(0x3);
  const __m128i cM4 = _mm_set1_epi8(0xF);
  const __m128i m2 = _mm_set1_epi8(0x2);

  __m256 vAcc = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float dall = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // load mins and scales from block_q2_K.scales[QK_K/16]
    const __m128i cMinsAndScales = _mm_loadu_si128((const __m128i *)vX[vI].scales);
    const __m128i scales16 = _mm_and_si128(cMinsAndScales, cM4);
    const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(cMinsAndScales, 4), cM4);
    const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
    const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

    // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
    const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i *)&vY[vI].bsums[0]));
    const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i *)&vY[vI].bsums[8]));

    // sumf += -dmin * summs in 32bits*8
    vAcc = _mm256_add_ps(
        _mm256_mul_ps(_mm256_broadcast_ss(&cDmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), vAcc);

    const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
    const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
    const __m128i cScales[2] = {scales_0, scales_1};

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
      const __m128i cQ80 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ81 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ82 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ83 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
      __m128i cQ2bits = _mm_loadu_si128((const __m128i *)vQ2);
      vQ2 += 16;
      const __m128i cQ20 = _mm_and_si128(cQ2bits, cM3);
      const __m128i cQ22 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 2), cM3);
      const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 4), cM3);
      const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 6), cM3);
      cQ2bits = _mm_loadu_si128((const __m128i *)vQ2);
      vQ2 += 16;
      const __m128i cQ21 = _mm_and_si128(cQ2bits, cM3);
      const __m128i cQ23 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 2), cM3);
      const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 4), cM3);
      const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(cQ2bits, 6), cM3);

      // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
      __m128i vP0 = _mm_maddubs_epi16(cQ20, cQ80);
      __m128i vP1 = _mm_maddubs_epi16(cQ21, cQ81);
      __m128i vP2 = _mm_maddubs_epi16(cQ22, cQ82);
      __m128i vP3 = _mm_maddubs_epi16(cQ23, cQ83);
      __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
      __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
      __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
      __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

      // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
      __m128i shuffle = _mm_set1_epi16(0x0100);
      vP0 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP0);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP1 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP1);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP2 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP2);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP3 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP3);
      shuffle = _mm_add_epi16(shuffle, m2);
      p4 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p4);
      shuffle = _mm_add_epi16(shuffle, m2);
      p5 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p5);
      shuffle = _mm_add_epi16(shuffle, m2);
      p6 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p6);
      shuffle = _mm_add_epi16(shuffle, m2);
      p7 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p7);

      vP0 = _mm_add_epi32(vP0, vP1);
      vP2 = _mm_add_epi32(vP2, vP3);
      p4 = _mm_add_epi32(p4, p5);
      p6 = _mm_add_epi32(p6, p7);

      // isum in 32bits*4*2
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(vP0, vP2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
    }

    // sumf += dall * isum - dmin * summs in 32bits
    __m256i vSumi = MM256_SET_M128I(sumi_1, sumi_0);
    vAcc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(vSumi)), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q2_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q3_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const uint32_t cKmask1 = 0x03030303;
  const uint32_t cKmask2 = 0x0f0f0f0f;

  const block_q3_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __AVX2__

  const __m256i cM3 = _mm256_set1_epi8(3);
  const __m256i cMone = _mm256_set1_epi8(1);
  const __m128i cM32 = _mm_set1_epi8(32);

  __m256 vAcc = _mm256_setzero_ps();

  uint32_t vAux[3];

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // Set up scales
    memcpy(vAux, vX[vI].scales, 12);
    __m128i vScales128 = _mm_set_epi32(((vAux[1] >> 4) & cKmask2) | (((vAux[2] >> 6) & cKmask1) << 4),
                                       ((vAux[0] >> 4) & cKmask2) | (((vAux[2] >> 4) & cKmask1) << 4),
                                       (vAux[1] & cKmask2) | (((vAux[2] >> 2) & cKmask1) << 4),
                                       (vAux[0] & cKmask2) | (((vAux[2] >> 0) & cKmask1) << 4));
    vScales128 = _mm_sub_epi8(vScales128, cM32);
    const __m256i cAllScales = _mm256_cvtepi8_epi16(vScales128);
    const __m128i cLScales = _mm256_extracti128_si256(cAllScales, 0);
    const __m128i cHScales = _mm256_extracti128_si256(cAllScales, 1);
    const __m256i cScales[2] = {MM256_SET_M128I(cLScales, cLScales), MM256_SET_M128I(cHScales, cHScales)};

    // high bit
    const __m256i cHbits = _mm256_loadu_si256((const __m256i *)vX[vI].hmask);

    // integer accumulator
    __m256i vSumi = _mm256_setzero_si256();

    int vBit = 0;
    int vIs = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load low 2 bits
      const __m256i cQ3bits = _mm256_loadu_si256((const __m256i *)vQ3);
      vQ3 += 32;

      // prepare low and high bits
      const __m256i cQ3l0 = _mm256_and_si256(cQ3bits, cM3);
      const __m256i cQ3h0 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(cHbits, _mm256_slli_epi16(cMone, vBit)), vBit), 2);
      ++vBit;

      const __m256i cQ3l1 = _mm256_and_si256(_mm256_srli_epi16(cQ3bits, 2), cM3);
      const __m256i cQ3h1 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(cHbits, _mm256_slli_epi16(cMone, vBit)), vBit), 2);
      ++vBit;

      const __m256i cQ3l2 = _mm256_and_si256(_mm256_srli_epi16(cQ3bits, 4), cM3);
      const __m256i cQ3h2 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(cHbits, _mm256_slli_epi16(cMone, vBit)), vBit), 2);
      ++vBit;

      const __m256i cQ3l3 = _mm256_and_si256(_mm256_srli_epi16(cQ3bits, 6), cM3);
      const __m256i cQ3h3 =
          _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(cHbits, _mm256_slli_epi16(cMone, vBit)), vBit), 2);
      ++vBit;

      // load Q8 quants
      const __m256i cQ80 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ83 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
      // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not
      // set, and 2 if the high bit was set)
      __m256i vQ8s0 = _mm256_maddubs_epi16(cQ3h0, cQ80);
      __m256i vQ8s1 = _mm256_maddubs_epi16(cQ3h1, cQ81);
      __m256i vQ8s2 = _mm256_maddubs_epi16(cQ3h2, cQ82);
      __m256i vQ8s3 = _mm256_maddubs_epi16(cQ3h3, cQ83);

      __m256i vP160 = _mm256_maddubs_epi16(cQ3l0, cQ80);
      __m256i vP161 = _mm256_maddubs_epi16(cQ3l1, cQ81);
      __m256i vP162 = _mm256_maddubs_epi16(cQ3l2, cQ82);
      __m256i vP163 = _mm256_maddubs_epi16(cQ3l3, cQ83);

      vP160 = _mm256_sub_epi16(vP160, vQ8s0);
      vP161 = _mm256_sub_epi16(vP161, vQ8s1);
      vP162 = _mm256_sub_epi16(vP162, vQ8s2);
      vP163 = _mm256_sub_epi16(vP163, vQ8s3);

      // multiply with scales
      vP160 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(vIs + 0)), vP160);
      vP161 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(vIs + 1)), vP161);
      vP162 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(vIs + 2)), vP162);
      vP163 = _mm256_madd_epi16(_mm256_shuffle_epi8(cScales[vJ], fGetScaleShuffleQ3k(vIs + 3)), vP163);

      // accumulate
      vP160 = _mm256_add_epi32(vP160, vP161);
      vP162 = _mm256_add_epi32(vP162, vP163);
      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(vP160, vP162));
    }

    // multiply with block scale and accumulate
    vAcc = _mm256_fmadd_ps(_mm256_broadcast_ss(&cD), _mm256_cvtepi32_ps(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#elif defined __AVX__

  const __m128i cM3 = _mm_set1_epi8(3);
  const __m128i cMone = _mm_set1_epi8(1);
  const __m128i cM32 = _mm_set1_epi8(32);
  const __m128i m2 = _mm_set1_epi8(2);

  __m256 vAcc = _mm256_setzero_ps();

  const uint32_t *vAux;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // Set up scales
    vAux = (const uint32_t *)vX[vI].scales;
    __m128i vScales128 = _mm_set_epi32(((vAux[1] >> 4) & cKmask2) | (((vAux[2] >> 6) & cKmask1) << 4),
                                       ((vAux[0] >> 4) & cKmask2) | (((vAux[2] >> 4) & cKmask1) << 4),
                                       (vAux[1] & cKmask2) | (((vAux[2] >> 2) & cKmask1) << 4),
                                       (vAux[0] & cKmask2) | (((vAux[2] >> 0) & cKmask1) << 4));
    vScales128 = _mm_sub_epi8(vScales128, cM32);
    const __m128i scales_0 = _mm_cvtepi8_epi16(vScales128);
    const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(vScales128, vScales128));
    const __m128i cScales[2] = {scales_0, scales_1};

    // high bit *128*2 from block_q3_K.hmask[QK_K/8]
    const __m128i hbits_0 = _mm_loadu_si128((const __m128i *)&vX[vI].hmask[0]);
    const __m128i hbits_1 = _mm_loadu_si128((const __m128i *)&vX[vI].hmask[16]);

    // integer accumulator
    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
      const __m128i q3bits_0 = _mm_loadu_si128((const __m128i *)vQ3);
      vQ3 += 16;
      const __m128i q3bits_1 = _mm_loadu_si128((const __m128i *)vQ3);
      vQ3 += 16;

      // prepare low and high bits
      const int vBit = vJ << 2;

      const __m128i cQ3l0 = _mm_and_si128(q3bits_0, cM3);
      const __m128i cQ3l1 = _mm_and_si128(q3bits_1, cM3);
      const __m128i cQ3h0 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(cMone, vBit)), vBit), 2);
      const __m128i cQ3h1 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(cMone, vBit)), vBit), 2);

      const __m128i cQ3l2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), cM3);
      const __m128i cQ3l3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), cM3);
      const __m128i cQ3h2 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(cMone, vBit + 1)), vBit + 1), 2);
      const __m128i cQ3h3 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(cMone, vBit + 1)), vBit + 1), 2);

      const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), cM3);
      const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), cM3);
      const __m128i q3h_4 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(cMone, vBit + 2)), vBit + 2), 2);
      const __m128i q3h_5 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(cMone, vBit + 2)), vBit + 2), 2);

      const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), cM3);
      const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), cM3);
      const __m128i q3h_6 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(cMone, vBit + 3)), vBit + 3), 2);
      const __m128i q3h_7 =
          _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(cMone, vBit + 3)), vBit + 3), 2);

      // load Q8 quants from block_q8_K.qs[QK_K]
      const __m128i cQ80 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ81 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ82 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ83 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
      // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not
      // set, and 2 if the high bit was set)
      __m128i vQ8s0 = _mm_maddubs_epi16(cQ3h0, cQ80);
      __m128i vQ8s1 = _mm_maddubs_epi16(cQ3h1, cQ81);
      __m128i vQ8s2 = _mm_maddubs_epi16(cQ3h2, cQ82);
      __m128i vQ8s3 = _mm_maddubs_epi16(cQ3h3, cQ83);
      __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
      __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
      __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
      __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

      __m128i vP160 = _mm_maddubs_epi16(cQ3l0, cQ80);
      __m128i vP161 = _mm_maddubs_epi16(cQ3l1, cQ81);
      __m128i vP162 = _mm_maddubs_epi16(cQ3l2, cQ82);
      __m128i vP163 = _mm_maddubs_epi16(cQ3l3, cQ83);
      __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

      vP160 = _mm_sub_epi16(vP160, vQ8s0);
      vP161 = _mm_sub_epi16(vP161, vQ8s1);
      vP162 = _mm_sub_epi16(vP162, vQ8s2);
      vP163 = _mm_sub_epi16(vP163, vQ8s3);
      p16_4 = _mm_sub_epi16(p16_4, q8s_4);
      p16_5 = _mm_sub_epi16(p16_5, q8s_5);
      p16_6 = _mm_sub_epi16(p16_6, q8s_6);
      p16_7 = _mm_sub_epi16(p16_7, q8s_7);

      // multiply with scales
      __m128i shuffle = _mm_set1_epi16(0x0100);
      vP160 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP160);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP161 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP161);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP162 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP162);
      shuffle = _mm_add_epi16(shuffle, m2);
      vP163 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), vP163);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p16_4);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p16_5);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p16_6);
      shuffle = _mm_add_epi16(shuffle, m2);
      p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(cScales[vJ], shuffle), p16_7);

      // accumulate
      vP160 = _mm_add_epi32(vP160, vP161);
      vP162 = _mm_add_epi32(vP162, vP163);
      p16_4 = _mm_add_epi32(p16_4, p16_5);
      p16_6 = _mm_add_epi32(p16_6, p16_7);
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(vP160, vP162));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));
    }

    // multiply with block scale and accumulate
    __m256i vSumi = MM256_SET_M128I(sumi_1, sumi_0);
    vAcc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&cD), _mm256_cvtepi32_ps(vSumi)), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#else
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q3_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q4_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  uint32_t vUtmp[4];

#if defined __AVX2__

  const __m256i cM4 = _mm256_set1_epi8(0xF);

  __m256 vAcc = _mm256_setzero_ps();
  __m128 vAccM = _mm_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m256i cMinsAndScales = _mm256_cvtepu8_epi16(_mm_set_epi32(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]));

    const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vY[vI].bsums);
    const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));
    const __m128i cProd = _mm_madd_epi16(_mm256_extracti128_si256(cMinsAndScales, 1), cQ8s);
    vAccM = _mm_fmadd_ps(_mm_set1_ps(cDmin), _mm_cvtepi32_ps(cProd), vAccM);

    const __m128i cSc128 = _mm256_extracti128_si256(cMinsAndScales, 0);
    const __m256i cScales = MM256_SET_M128I(cSc128, cSc128);

    __m256i vSumi = _mm256_setzero_si256();

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m256i cScaleL = _mm256_shuffle_epi8(cScales, fGetScaleShuffleK4(2 * vJ + 0));
      const __m256i cScaleH = _mm256_shuffle_epi8(cScales, fGetScaleShuffleK4(2 * vJ + 1));

      const __m256i cQ4bits = _mm256_loadu_si256((const __m256i *)vQ4);
      vQ4 += 32;
      const __m256i cQ4l = _mm256_and_si256(cQ4bits, cM4);
      const __m256i cQ4h = _mm256_and_si256(_mm256_srli_epi16(cQ4bits, 4), cM4);

      const __m256i cQ8l = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      __m256i vP16l = _mm256_maddubs_epi16(cQ4l, cQ8l);
      vP16l = _mm256_madd_epi16(cScaleL, vP16l);

      const __m256i cQ8h = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      __m256i vP16h = _mm256_maddubs_epi16(cQ4h, cQ8h);
      vP16h = _mm256_madd_epi16(cScaleH, vP16h);
      const __m256i cSumj = _mm256_add_epi32(vP16l, vP16h);

      vSumi = _mm256_add_epi32(vSumi, cSumj);
    }

    __m256 vVd = _mm256_set1_ps(cD);
    vAcc = _mm256_fmadd_ps(vVd, _mm256_cvtepi32_ps(vSumi), vAcc);
  }

  vAccM = _mm_add_ps(vAccM, _mm_movehl_ps(vAccM, vAccM));
  vAccM = _mm_add_ss(vAccM, _mm_movehdup_ps(vAccM));

  *pS = fHsumFloat8(vAcc) + _mm_cvtss_f32(vAccM);

#elif defined __AVX__

  const __m128i cM4 = _mm_set1_epi8(0xF);
  const __m128i m2 = _mm_set1_epi8(0x2);

  __m256 vAcc = _mm256_setzero_ps();
  __m128 vAccM = _mm_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const __m128i utmps = _mm_set_epi32(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]);
    const __m128i cScales = _mm_cvtepu8_epi16(utmps);
    const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

    const __m128i q8sums_0 = _mm_loadu_si128((const __m128i *)&vY[vI].bsums[0]);
    const __m128i q8sums_1 = _mm_loadu_si128((const __m128i *)&vY[vI].bsums[8]);
    const __m128i cQ8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
    const __m128i cProd = _mm_madd_epi16(mins, cQ8s);
    vAccM = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cDmin), _mm_cvtepi32_ps(cProd)), vAccM);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    __m128i shuffle = _mm_set1_epi16(0x0100);
    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m128i cScaleL = _mm_shuffle_epi8(cScales, shuffle);
      shuffle = _mm_add_epi16(shuffle, m2);
      const __m128i cScaleH = _mm_shuffle_epi8(cScales, shuffle);
      shuffle = _mm_add_epi16(shuffle, m2);

      __m128i cQ4bits = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;
      const __m128i q4l_0 = _mm_and_si128(cQ4bits, cM4);
      const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(cQ4bits, 4), cM4);
      cQ4bits = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;
      const __m128i q4l_1 = _mm_and_si128(cQ4bits, cM4);
      const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(cQ4bits, 4), cM4);

      const __m128i q8l_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      __m128i vP16l = _mm_maddubs_epi16(q4l_0, q8l_0);
      vP16l = _mm_madd_epi16(cScaleL, vP16l);
      sumi_0 = _mm_add_epi32(sumi_0, vP16l);
      const __m128i q8l_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      vP16l = _mm_maddubs_epi16(q4l_1, q8l_1);
      vP16l = _mm_madd_epi16(cScaleL, vP16l);
      sumi_1 = _mm_add_epi32(sumi_1, vP16l);

      const __m128i q8h_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      __m128i vP16h = _mm_maddubs_epi16(q4h_0, q8h_0);
      vP16h = _mm_madd_epi16(cScaleH, vP16h);
      sumi_0 = _mm_add_epi32(sumi_0, vP16h);
      const __m128i q8h_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      vP16h = _mm_maddubs_epi16(q4h_1, q8h_1);
      vP16h = _mm_madd_epi16(cScaleH, vP16h);
      sumi_1 = _mm_add_epi32(sumi_1, vP16h);
    }

    __m256 vVd = _mm256_set1_ps(cD);
    __m256i vSumi = MM256_SET_M128I(sumi_1, sumi_0);
    vAcc = _mm256_add_ps(_mm256_mul_ps(vVd, _mm256_cvtepi32_ps(vSumi)), vAcc);
  }

  vAccM = _mm_add_ps(vAccM, _mm_movehl_ps(vAccM, vAccM));
  vAccM = _mm_add_ss(vAccM, _mm_movehdup_ps(vAccM));

  *pS = fHsumFloat8(vAcc) + _mm_cvtss_f32(vAccM);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  UNUSED(vUtmp);
  ggml_vec_dot_q4_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  uint32_t vUtmp[4];

#if defined __AVX2__

  const __m256i cM4 = _mm256_set1_epi8(0xF);
  const __m128i cMzero = _mm_setzero_si128();
  const __m256i cMone = _mm256_set1_epi8(1);

  __m256 vAcc = _mm256_setzero_ps();

  float vSumms = 0.f;

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const __m256i cMinsAndScales = _mm256_cvtepu8_epi16(_mm_set_epi32(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]));

    const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vY[vI].bsums);
    const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));
    const __m128i cProd = _mm_madd_epi16(_mm256_extracti128_si256(cMinsAndScales, 1), cQ8s);
    const __m128i cHsum = _mm_hadd_epi32(_mm_hadd_epi32(cProd, cMzero), cMzero);
    vSumms += cDmin * _mm_extract_epi32(cHsum, 0);

    const __m128i cSc128 = _mm256_extracti128_si256(cMinsAndScales, 0);
    const __m256i cScales = MM256_SET_M128I(cSc128, cSc128);

    const __m256i cHbits = _mm256_loadu_si256((const __m256i *)vX[vI].qh);
    __m256i vHmask = cMone;

    __m256i vSumi = _mm256_setzero_si256();

    int vBit = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m256i cScale0 = _mm256_shuffle_epi8(cScales, fGetScaleShuffleK4(2 * vJ + 0));
      const __m256i cScale1 = _mm256_shuffle_epi8(cScales, fGetScaleShuffleK4(2 * vJ + 1));

      const __m256i cQ5bits = _mm256_loadu_si256((const __m256i *)vQ5);
      vQ5 += 32;

      const __m256i cQ5l0 = _mm256_and_si256(cQ5bits, cM4);
      const __m256i cQ5h0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(cHbits, vHmask), vBit++), 4);
      const __m256i cQ50 = _mm256_add_epi8(cQ5l0, cQ5h0);
      vHmask = _mm256_slli_epi16(vHmask, 1);

      const __m256i cQ5l1 = _mm256_and_si256(_mm256_srli_epi16(cQ5bits, 4), cM4);
      const __m256i cQ5h1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(cHbits, vHmask), vBit++), 4);
      const __m256i cQ51 = _mm256_add_epi8(cQ5l1, cQ5h1);
      vHmask = _mm256_slli_epi16(vHmask, 1);

      const __m256i cQ80 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      __m256i vP160 = _mm256_maddubs_epi16(cQ50, cQ80);
      __m256i vP161 = _mm256_maddubs_epi16(cQ51, cQ81);

      vP160 = _mm256_madd_epi16(cScale0, vP160);
      vP161 = _mm256_madd_epi16(cScale1, vP161);

      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(vP160, vP161));
    }

    __m256 vVd = _mm256_set1_ps(cD);
    vAcc = _mm256_fmadd_ps(vVd, _mm256_cvtepi32_ps(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc) + vSumms;

#elif defined __AVX__

  const __m128i cM4 = _mm_set1_epi8(0xF);
  const __m128i cMzero = _mm_setzero_si128();
  const __m128i cMone = _mm_set1_epi8(1);
  const __m128i m2 = _mm_set1_epi8(2);

  __m256 vAcc = _mm256_setzero_ps();

  float vSumms = 0.f;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const __m128i utmps = _mm_set_epi32(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]);
    const __m128i cScales = _mm_cvtepu8_epi16(utmps);
    const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

    const __m128i q8sums_0 = _mm_loadu_si128((const __m128i *)&vY[vI].bsums[0]);
    const __m128i q8sums_1 = _mm_loadu_si128((const __m128i *)&vY[vI].bsums[8]);
    const __m128i cQ8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
    const __m128i cProd = _mm_madd_epi16(mins, cQ8s);
    const __m128i cHsum = _mm_hadd_epi32(_mm_hadd_epi32(cProd, cMzero), cMzero);
    vSumms += cDmin * _mm_extract_epi32(cHsum, 0);

    const __m128i hbits_0 = _mm_loadu_si128((const __m128i *)&vX[vI].qh[0]);
    const __m128i hbits_1 = _mm_loadu_si128((const __m128i *)&vX[vI].qh[16]);
    __m128i vHmask = cMone;

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    int vBit = 0;

    __m128i shuffle = _mm_set1_epi16(0x0100);
    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m128i cScale0 = _mm_shuffle_epi8(cScales, shuffle);
      shuffle = _mm_add_epi16(shuffle, m2);
      const __m128i cScale1 = _mm_shuffle_epi8(cScales, shuffle);
      shuffle = _mm_add_epi16(shuffle, m2);

      const __m128i q5bits_0 = _mm_loadu_si128((const __m128i *)vQ5);
      vQ5 += 16;
      const __m128i q5bits_1 = _mm_loadu_si128((const __m128i *)vQ5);
      vQ5 += 16;

      __m128i cQ5l0 = _mm_and_si128(q5bits_0, cM4);
      __m128i cQ5l1 = _mm_and_si128(q5bits_1, cM4);
      __m128i cQ5h0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, vHmask), vBit), 4);
      __m128i cQ5h1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, vHmask), vBit++), 4);
      __m128i cQ50 = _mm_add_epi8(cQ5l0, cQ5h0);
      __m128i cQ51 = _mm_add_epi8(cQ5l1, cQ5h1);
      vHmask = _mm_slli_epi16(vHmask, 1);

      __m128i cQ80 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      __m128i cQ81 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      __m128i vP160 = _mm_maddubs_epi16(cQ50, cQ80);
      __m128i vP161 = _mm_maddubs_epi16(cQ51, cQ81);
      vP160 = _mm_madd_epi16(cScale0, vP160);
      vP161 = _mm_madd_epi16(cScale0, vP161);

      cQ5l0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), cM4);
      cQ5l1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), cM4);
      cQ5h0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, vHmask), vBit), 4);
      cQ5h1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, vHmask), vBit++), 4);
      cQ50 = _mm_add_epi8(cQ5l0, cQ5h0);
      cQ51 = _mm_add_epi8(cQ5l1, cQ5h1);
      vHmask = _mm_slli_epi16(vHmask, 1);

      cQ80 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      cQ81 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      __m128i p16_2 = _mm_maddubs_epi16(cQ50, cQ80);
      __m128i p16_3 = _mm_maddubs_epi16(cQ51, cQ81);
      p16_2 = _mm_madd_epi16(cScale1, p16_2);
      p16_3 = _mm_madd_epi16(cScale1, p16_3);

      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(vP160, p16_2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(vP161, p16_3));
    }

    __m256 vVd = _mm256_set1_ps(cD);
    __m256i vSumi = MM256_SET_M128I(sumi_1, sumi_0);
    vAcc = _mm256_add_ps(_mm256_mul_ps(vVd, _mm256_cvtepi32_ps(vSumi)), vAcc);
  }

  *pS = fHsumFloat8(vAcc) + vSumms;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  UNUSED(vUtmp);
  ggml_vec_dot_q5_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q6_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __AVX2__

  const __m256i cM3 = _mm256_set1_epi8(3);
  const __m256i cM15 = _mm256_set1_epi8(15);

  __m256 vAcc = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vY[vI].bsums);
    const __m128i cScales = _mm_loadu_si128((const __m128i *)vX[vI].scales);
    const __m256i cScales16 = _mm256_cvtepi8_epi16(cScales);
    const __m256i cQ8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(cQ8sums, cScales16), 5);

    __m256i vSumi = _mm256_setzero_si256();

    int vIs = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      const __m256i cQ4bits1 = _mm256_loadu_si256((const __m256i *)vQ4);
      vQ4 += 32;
      const __m256i cQ4bits2 = _mm256_loadu_si256((const __m256i *)vQ4);
      vQ4 += 32;
      const __m256i cQ4bitsH = _mm256_loadu_si256((const __m256i *)vQh);
      vQh += 32;

      const __m256i cQ4h0 = _mm256_slli_epi16(_mm256_and_si256(cQ4bitsH, cM3), 4);
      const __m256i cQ4h1 = _mm256_slli_epi16(_mm256_and_si256(cQ4bitsH, _mm256_set1_epi8(12)), 2);
      const __m256i cQ4h2 = _mm256_and_si256(cQ4bitsH, _mm256_set1_epi8(48));
      const __m256i cQ4h3 = _mm256_srli_epi16(_mm256_and_si256(cQ4bitsH, _mm256_set1_epi8(-64)), 2);

      const __m256i cQ40 = _mm256_or_si256(_mm256_and_si256(cQ4bits1, cM15), cQ4h0);
      const __m256i cQ41 = _mm256_or_si256(_mm256_and_si256(cQ4bits2, cM15), cQ4h1);
      const __m256i cQ42 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(cQ4bits1, 4), cM15), cQ4h2);
      const __m256i cQ43 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(cQ4bits2, 4), cM15), cQ4h3);

      const __m256i cQ80 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ83 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      __m256i vP160 = _mm256_maddubs_epi16(cQ40, cQ80);
      __m256i vP161 = _mm256_maddubs_epi16(cQ41, cQ81);
      __m256i vP162 = _mm256_maddubs_epi16(cQ42, cQ82);
      __m256i vP163 = _mm256_maddubs_epi16(cQ43, cQ83);

      const __m128i cScale0 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 0));
      const __m128i cScale1 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 1));
      const __m128i cScale2 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 2));
      const __m128i cScale3 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 3));
      vIs += 4;

      vP160 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(cScale0), vP160);
      vP161 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(cScale1), vP161);
      vP162 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(cScale2), vP162);
      vP163 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(cScale3), vP163);

      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(vP160, vP161));
      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(vP162, vP163));
    }

    vSumi = _mm256_sub_epi32(vSumi, cQ8sclsub);
    vAcc = _mm256_fmadd_ps(_mm256_broadcast_ss(&cD), _mm256_cvtepi32_ps(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#elif defined __AVX__

  const __m128i cM3 = _mm_set1_epi8(3);
  const __m128i cM15 = _mm_set1_epi8(15);

  __m256 vAcc = _mm256_setzero_ps();

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // handle the q6_k -32 offset separately using bsums
    const __m128i q8sums_0 = _mm_loadu_si128((const __m128i *)vY[vI].bsums);
    const __m128i q8sums_1 = _mm_loadu_si128((const __m128i *)vY[vI].bsums + 1);
    const __m128i cScales = _mm_loadu_si128((const __m128i *)vX[vI].scales);
    const __m128i scales_16_0 = _mm_cvtepi8_epi16(cScales);
    const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(cScales, 8));
    const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
    const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    int vIs = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i *)vQh);
      vQh += 16;
      const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i *)vQh);
      vQh += 16;

      const __m128i cQ4h0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, cM3), 4);
      const __m128i cQ4h1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, cM3), 4);
      const __m128i cQ4h2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
      const __m128i cQ4h3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
      const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
      const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
      const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
      const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

      const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;
      const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;
      const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;
      const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i *)vQ4);
      vQ4 += 16;

      const __m128i cQ40 = _mm_or_si128(_mm_and_si128(q4bits1_0, cM15), cQ4h0);
      const __m128i cQ41 = _mm_or_si128(_mm_and_si128(q4bits1_1, cM15), cQ4h1);
      const __m128i cQ42 = _mm_or_si128(_mm_and_si128(q4bits2_0, cM15), cQ4h2);
      const __m128i cQ43 = _mm_or_si128(_mm_and_si128(q4bits2_1, cM15), cQ4h3);
      const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), cM15), q4h_4);
      const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), cM15), q4h_5);
      const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), cM15), q4h_6);
      const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), cM15), q4h_7);

      const __m128i cQ80 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ81 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ82 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i cQ83 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      __m128i vP160 = _mm_maddubs_epi16(cQ40, cQ80);
      __m128i vP161 = _mm_maddubs_epi16(cQ41, cQ81);
      __m128i vP162 = _mm_maddubs_epi16(cQ42, cQ82);
      __m128i vP163 = _mm_maddubs_epi16(cQ43, cQ83);
      __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

      const __m128i cScale0 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 0));
      const __m128i cScale1 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 1));
      const __m128i cScale2 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 2));
      const __m128i cScale3 = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIs + 3));
      vIs += 4;

      vP160 = _mm_madd_epi16(_mm_cvtepi8_epi16(cScale0), vP160);
      vP161 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(cScale0, 8)), vP161);
      vP162 = _mm_madd_epi16(_mm_cvtepi8_epi16(cScale1), vP162);
      vP163 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(cScale1, 8)), vP163);
      p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(cScale2), p16_4);
      p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(cScale2, 8)), p16_5);
      p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(cScale3), p16_6);
      p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(cScale3, 8)), p16_7);

      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(vP160, vP162));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(vP161, vP163));
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
    }

    sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
    sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
    const __m256i vSumi = MM256_SET_M128I(sumi_1, sumi_0);
    vAcc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(vSumi)), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined(__AVX__) || defined(__AVX2__)
static const int8_t keven_signs_q2xs[1024] = {
    1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,  1,  1,
    1,  1,  1,  1,  1,  -1, 1,  1,  1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1,
    -1, 1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  -1, 1,  -1, 1,  1,  1,
    1,  -1, -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1,
    1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  1,  1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,
    1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,  1,  -1,
    -1, 1,  -1, -1, -1, 1,  1,  1,  1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,
    -1, 1,  -1, -1, 1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,
    -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  1,  -1, 1,
    1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1,
    1,  -1, 1,  1,  1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  -1, -1, -1, 1,  -1, 1,  1,  -1,
    -1, -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1,
    1,  -1, -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  -1, -1,
    1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  1,
    1,  -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1, -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  -1, -1, -1,
    -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, -1, -1, 1,
    1,  1,  1,  1,  -1, 1,  1,  -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  1,  1,  1,  -1,
    1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  -1, 1,  1,
    1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  -1, -1, -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  1,
    -1, -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,
    -1, -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  -1, -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1,
    1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, -1, 1,  -1,
    -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1,
    -1, 1,  -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,
    1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, -1, -1, 1,  1,  -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1,
    -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, 1,  1,  1,  1,
    1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, 1,  1,  -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, -1,
    1,  1,  -1, 1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1,
    -1, -1, 1,  1,  1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, -1, 1,  -1, 1,  -1, -1, -1, -1, -1, -1, -1,
    1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, 1,  1,  -1, -1, -1, -1, -1, -1,
    1,  -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[4];
  const uint8_t *vAux8 = (const uint8_t *)vAux32;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      memcpy(vAux32, vQ2, 4 * sizeof(uint32_t));
      vQ2 += 8;
      const __m256i cQ21 =
          _mm256_set_epi64x(iq2xxs_grid[vAux8[3]], iq2xxs_grid[vAux8[2]], iq2xxs_grid[vAux8[1]], iq2xxs_grid[vAux8[0]]);
      const __m256i cQ22 = _mm256_set_epi64x(iq2xxs_grid[vAux8[11]], iq2xxs_grid[vAux8[10]], iq2xxs_grid[vAux8[9]],
                                             iq2xxs_grid[vAux8[8]]);
      const __m256i cS21 = _mm256_set_epi64x(vSigns64[(vAux32[1] >> 21) & 127], vSigns64[(vAux32[1] >> 14) & 127],
                                             vSigns64[(vAux32[1] >> 7) & 127], vSigns64[(vAux32[1] >> 0) & 127]);
      const __m256i cS22 = _mm256_set_epi64x(vSigns64[(vAux32[3] >> 21) & 127], vSigns64[(vAux32[3] >> 14) & 127],
                                             vSigns64[(vAux32[3] >> 7) & 127], vSigns64[(vAux32[3] >> 0) & 127]);
      const __m256i cQ8s1 = _mm256_sign_epi8(cQ81, cS21);
      const __m256i cQ8s2 = _mm256_sign_epi8(cQ82, cS22);
      const __m256i cDot1 = _mm256_maddubs_epi16(cQ21, cQ8s1);
      const __m256i cDot2 = _mm256_maddubs_epi16(cQ22, cQ8s2);
      const uint16_t cLs1 = vAux32[1] >> 28;
      const uint16_t cLs2 = vAux32[3] >> 28;
      const __m256i cP1 = _mm256_madd_epi16(cDot1, _mm256_set1_epi16(2 * cLs1 + 1));
      const __m256i cP2 = _mm256_madd_epi16(cDot2, _mm256_set1_epi16(2 * cLs2 + 1));
      vSumi1 = _mm256_add_epi32(vSumi1, cP1);
      vSumi2 = _mm256_add_epi32(vSumi2, cP2);
    }

    vAccumf = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#elif defined(__AVX__)
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[4];
  const uint8_t *vAux8 = (const uint8_t *)vAux32;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      memcpy(vAux32, vQ2, 4 * sizeof(uint32_t));
      vQ2 += 8;
      const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[vAux8[1]], iq2xxs_grid[vAux8[0]]);
      const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[vAux8[3]], iq2xxs_grid[vAux8[2]]);
      const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[vAux8[9]], iq2xxs_grid[vAux8[8]]);
      const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[vAux8[11]], iq2xxs_grid[vAux8[10]]);
      const __m128i s2_1_0 = _mm_set_epi64x(vSigns64[(vAux32[1] >> 7) & 127], vSigns64[(vAux32[1] >> 0) & 127]);
      const __m128i s2_1_1 = _mm_set_epi64x(vSigns64[(vAux32[1] >> 21) & 127], vSigns64[(vAux32[1] >> 14) & 127]);
      const __m128i s2_2_0 = _mm_set_epi64x(vSigns64[(vAux32[3] >> 7) & 127], vSigns64[(vAux32[3] >> 0) & 127]);
      const __m128i s2_2_1 = _mm_set_epi64x(vSigns64[(vAux32[3] >> 21) & 127], vSigns64[(vAux32[3] >> 14) & 127]);
      const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
      const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
      const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
      const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
      const uint16_t cLs1 = vAux32[1] >> 28;
      const uint16_t cLs2 = vAux32[3] >> 28;
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2 * cLs2 + 1));
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2 * cLs2 + 1));
      sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
      sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
      sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
      sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
    }

    vAccumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1),
                                                                             _mm_add_epi32(sumi1_0, sumi2_0)))),
        vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq2_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)

  const __m256i cMone = _mm256_set1_epi8(1);
  static const char cBlockSignShuffleMask1[32] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
      0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
  };
  static const char cBlockSignShuffleMask2[32] = {
      0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
      0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
  };
  static const uint8_t cBitSelectorMaskBytes[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m256i cBitSelectorMask = _mm256_loadu_si256((const __m256i *)cBitSelectorMaskBytes);
  const __m256i cBlockSignShuffle1 = _mm256_loadu_si256((const __m256i *)cBlockSignShuffleMask1);
  const __m256i cBlockSignShuffle2 = _mm256_loadu_si256((const __m256i *)cBlockSignShuffleMask2);

  static const uint8_t cKBitHelper[32] = {
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
  };
  const __m256i cBitHelper = _mm256_loadu_si256((const __m256i *)cKBitHelper);
  const __m256i cM511 = _mm256_set1_epi16(511);
  const __m128i cM4 = _mm_set1_epi8(0xf);
  const __m128i cM1 = _mm_set1_epi8(1);

  uint64_t vAux64;

  // somewhat hacky, but gives a significant boost in performance
  __m256i vAuxGindex;
  const uint16_t *vGindex = (const uint16_t *)&vAuxGindex;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(&vAux64, vX[vI].scales, 8);
    __m128i vStmp = _mm_set1_epi64x(vAux64);
    vStmp = _mm_unpacklo_epi8(_mm_and_si128(vStmp, cM4), _mm_and_si128(_mm_srli_epi16(vStmp, 4), cM4));
    const __m128i cScales = _mm_add_epi8(_mm_slli_epi16(vStmp, 1), cM1);

    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 4) {

      const __m256i cQ2Data = _mm256_loadu_si256((const __m256i *)vQ2);
      vQ2 += 16;
      vAuxGindex = _mm256_and_si256(cQ2Data, cM511);

      const __m256i cPartialSignBits = _mm256_srli_epi16(cQ2Data, 9);
      const __m256i cPartialSignBitsUpper = _mm256_srli_epi16(cQ2Data, 13);
      const __m256i cPartialSignBitsForCounting = _mm256_xor_si256(cPartialSignBits, cPartialSignBitsUpper);

      const __m256i cOddBits = _mm256_shuffle_epi8(cBitHelper, cPartialSignBitsForCounting);
      const __m256i cFullSignBits = _mm256_or_si256(cPartialSignBits, cOddBits);

      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ83 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ84 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      const __m256i cQ21 = _mm256_set_epi64x(iq2xs_grid[vGindex[3]], iq2xs_grid[vGindex[2]], iq2xs_grid[vGindex[1]],
                                             iq2xs_grid[vGindex[0]]);
      const __m256i cQ22 = _mm256_set_epi64x(iq2xs_grid[vGindex[7]], iq2xs_grid[vGindex[6]], iq2xs_grid[vGindex[5]],
                                             iq2xs_grid[vGindex[4]]);
      const __m256i cQ23 = _mm256_set_epi64x(iq2xs_grid[vGindex[11]], iq2xs_grid[vGindex[10]], iq2xs_grid[vGindex[9]],
                                             iq2xs_grid[vGindex[8]]);
      const __m256i cQ24 = _mm256_set_epi64x(iq2xs_grid[vGindex[15]], iq2xs_grid[vGindex[14]], iq2xs_grid[vGindex[13]],
                                             iq2xs_grid[vGindex[12]]);

      const __m128i cFullSignsL = _mm256_castsi256_si128(cFullSignBits);
      const __m128i cFullSignsH = _mm256_extractf128_si256(cFullSignBits, 1);
      const __m256i cFullSigns1 = MM256_SET_M128I(cFullSignsL, cFullSignsL);
      const __m256i cFullSigns2 = MM256_SET_M128I(cFullSignsH, cFullSignsH);

      __m256i vSigns;
      vSigns = _mm256_shuffle_epi8(cFullSigns1, cBlockSignShuffle1);
      vSigns = _mm256_cmpeq_epi8(_mm256_and_si256(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s1 = _mm256_sign_epi8(cQ81, _mm256_or_si256(vSigns, cMone));

      vSigns = _mm256_shuffle_epi8(cFullSigns1, cBlockSignShuffle2);
      vSigns = _mm256_cmpeq_epi8(_mm256_and_si256(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s2 = _mm256_sign_epi8(cQ82, _mm256_or_si256(vSigns, cMone));

      vSigns = _mm256_shuffle_epi8(cFullSigns2, cBlockSignShuffle1);
      vSigns = _mm256_cmpeq_epi8(_mm256_and_si256(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s3 = _mm256_sign_epi8(cQ83, _mm256_or_si256(vSigns, cMone));

      vSigns = _mm256_shuffle_epi8(cFullSigns2, cBlockSignShuffle2);
      vSigns = _mm256_cmpeq_epi8(_mm256_and_si256(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s4 = _mm256_sign_epi8(cQ84, _mm256_or_si256(vSigns, cMone));

      const __m256i cDot1 = _mm256_maddubs_epi16(cQ21, cQ8s1);
      const __m256i cDot2 = _mm256_maddubs_epi16(cQ22, cQ8s2);
      const __m256i cDot3 = _mm256_maddubs_epi16(cQ23, cQ8s3);
      const __m256i cDot4 = _mm256_maddubs_epi16(cQ24, cQ8s4);

      const __m256i cSc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 0)));
      const __m256i cSc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 1)));
      const __m256i cSc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 2)));
      const __m256i cSc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 3)));

      vSumi1 = _mm256_add_epi32(vSumi1, _mm256_madd_epi16(cDot1, cSc1));
      vSumi2 = _mm256_add_epi32(vSumi2, _mm256_madd_epi16(cDot2, cSc2));
      vSumi1 = _mm256_add_epi32(vSumi1, _mm256_madd_epi16(cDot3, cSc3));
      vSumi2 = _mm256_add_epi32(vSumi2, _mm256_madd_epi16(cDot4, cSc4));
    }

    vAccumf = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#elif defined(__AVX__)
  const __m128i cMone = _mm_set1_epi8(1);
  static const char cBlockSignShuffleMask1[32] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
      0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
  };
  static const char cBlockSignShuffleMask2[32] = {
      0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
      0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
  };
  static const uint8_t cBitSelectorMaskBytes[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m128i bit_selector_mask_0 = _mm_loadu_si128((const __m128i *)cBitSelectorMaskBytes);
  const __m128i bit_selector_mask_1 = _mm_loadu_si128((const __m128i *)cBitSelectorMaskBytes + 1);
  const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i *)cBlockSignShuffleMask1);
  const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i *)cBlockSignShuffleMask1 + 1);
  const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i *)cBlockSignShuffleMask2);
  const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i *)cBlockSignShuffleMask2 + 1);

  static const uint8_t cKBitHelper[32] = {
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
  };
  const __m128i bit_helper_0 = _mm_loadu_si128((const __m128i *)cKBitHelper);
  const __m128i bit_helper_1 = _mm_loadu_si128((const __m128i *)cKBitHelper + 1);
  const __m128i cM511 = _mm_set1_epi16(511);
  const __m128i cM4 = _mm_set1_epi8(0xf);
  const __m128i cM1 = _mm_set1_epi8(1);

  uint64_t vAux64;

  // somewhat hacky, but gives a significant boost in performance
  __m256i vAuxGindex;
  const uint16_t *vGindex = (const uint16_t *)&vAuxGindex;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(&vAux64, vX[vI].scales, 8);
    __m128i vStmp = _mm_set1_epi64x(vAux64);
    vStmp = _mm_unpacklo_epi8(_mm_and_si128(vStmp, cM4), _mm_and_si128(_mm_srli_epi16(vStmp, 4), cM4));
    const __m128i cScales = _mm_add_epi8(_mm_slli_epi16(vStmp, 1), cM1);

    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 4) {

      const __m128i q2_data_0 = _mm_loadu_si128((const __m128i *)vQ2);
      const __m128i q2_data_1 = _mm_loadu_si128((const __m128i *)vQ2 + 1);
      vQ2 += 16;
      vAuxGindex = MM256_SET_M128I(_mm_and_si128(q2_data_1, cM511), _mm_and_si128(q2_data_0, cM511));

      const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
      const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
      const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
      const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
      const __m128i partial_sign_bits_for_counting_0 = _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0);
      const __m128i partial_sign_bits_for_counting_1 = _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1);

      const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper_0, partial_sign_bits_for_counting_0);
      const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper_1, partial_sign_bits_for_counting_1);
      const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
      const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[vGindex[1]], iq2xs_grid[vGindex[0]]);
      const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[vGindex[3]], iq2xs_grid[vGindex[2]]);
      const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[vGindex[5]], iq2xs_grid[vGindex[4]]);
      const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[vGindex[7]], iq2xs_grid[vGindex[6]]);
      const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[vGindex[9]], iq2xs_grid[vGindex[8]]);
      const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[vGindex[11]], iq2xs_grid[vGindex[10]]);
      const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[vGindex[13]], iq2xs_grid[vGindex[12]]);
      const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[vGindex[15]], iq2xs_grid[vGindex[14]]);

      // AVX2 full_signs_1 is full_sign_bits_0 here
      // AVX2 full_signs_2 is full_sign_bits_1 here
      __m128i signs_0, signs_1;
      signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
      signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
      signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
      signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
      const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, cMone));
      const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, cMone));

      signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
      signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
      signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
      signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
      const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, cMone));
      const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, cMone));

      signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
      signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
      signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
      signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
      const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, cMone));
      const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, cMone));

      signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
      signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
      signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
      signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
      const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, cMone));
      const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, cMone));

      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
      const __m128i dot3_0 = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
      const __m128i dot3_1 = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
      const __m128i dot4_0 = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
      const __m128i dot4_1 = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

      __m128i sc_tmp = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 0));
      const __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
      const __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
      sc_tmp = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 1));
      const __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
      const __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
      sc_tmp = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 2));
      const __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
      const __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
      sc_tmp = _mm_shuffle_epi8(cScales, fGetScaleShuffle(vIb32 + 3));
      const __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
      const __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

      sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
      sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
      sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
      sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
      sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
      sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
      sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
      sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
    }

    vAccumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1),
                                                                             _mm_add_epi32(sumi1_0, sumi2_0)))),
        vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m128i cM4 = _mm_set1_epi8(0xf);
  const __m128i cM1 = _mm_set1_epi8(1);

  const __m256i cMask1 = _mm256_loadu_si256((const __m256i *)cKMask1);
  const __m256i cMask2 = _mm256_loadu_si256((const __m256i *)cKMask2);

  uint64_t vAux64;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].qs + QK_K / 8);
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(&vAux64, vX[vI].scales, 8);
    const __m128i cScales8 =
        _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(vAux64 >> 4, vAux64), cM4), 1), cM1);
    const __m256i cScales16 = _mm256_cvtepi8_epi16(cScales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ21 = _mm256_set_epi64x(
          iq2s_grid[vQs[3] | ((vQh[vIb32 + 0] << 2) & 0x300)], iq2s_grid[vQs[2] | ((vQh[vIb32 + 0] << 4) & 0x300)],
          iq2s_grid[vQs[1] | ((vQh[vIb32 + 0] << 6) & 0x300)], iq2s_grid[vQs[0] | ((vQh[vIb32 + 0] << 8) & 0x300)]);
      const __m256i cQ22 = _mm256_set_epi64x(
          iq2s_grid[vQs[7] | ((vQh[vIb32 + 1] << 2) & 0x300)], iq2s_grid[vQs[6] | ((vQh[vIb32 + 1] << 4) & 0x300)],
          iq2s_grid[vQs[5] | ((vQh[vIb32 + 1] << 6) & 0x300)], iq2s_grid[vQs[4] | ((vQh[vIb32 + 1] << 8) & 0x300)]);
      vQs += 8;

      __m256i vAux256 = _mm256_set1_epi32(vSigns[0] | ((uint32_t)vSigns[1] << 16));
      vAux256 = _mm256_and_si256(_mm256_shuffle_epi8(vAux256, cMask1), cMask2);
      const __m256i cS21 = _mm256_cmpeq_epi8(vAux256, cMask2);
      const __m256i cQ8s1 = _mm256_sub_epi8(_mm256_xor_si256(cS21, cQ81), cS21);

      vAux256 = _mm256_set1_epi32(vSigns[2] | ((uint32_t)vSigns[3] << 16));
      vAux256 = _mm256_and_si256(_mm256_shuffle_epi8(vAux256, cMask1), cMask2);
      const __m256i cS22 = _mm256_cmpeq_epi8(vAux256, cMask2);
      const __m256i cQ8s2 = _mm256_sub_epi8(_mm256_xor_si256(cS22, cQ82), cS22);

      vSigns += 4;

      const __m256i cDot1 = _mm256_maddubs_epi16(cQ21, cQ8s1); // blocks 2*ib32+0, 2*ib32+1
      const __m256i cDot2 = _mm256_maddubs_epi16(cQ22, cQ8s2); // blocks 2*ib32+2, 2*ib32+3

      const __m256i cP1 = _mm256_madd_epi16(cDot1, _mm256_shuffle_epi8(cScales16, fGetScaleShuffleK4(vIb32 + 0)));
      const __m256i cP2 = _mm256_madd_epi16(cDot2, _mm256_shuffle_epi8(cScales16, fGetScaleShuffleK4(vIb32 + 1)));
      vSumi1 = _mm256_add_epi32(vSumi1, cP1);
      vSumi2 = _mm256_add_epi32(vSumi2, cP2);
    }

    vAccumf = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#elif defined(__AVX__)
  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m128i cM4 = _mm_set1_epi8(0xf);
  const __m128i cM1 = _mm_set1_epi8(1);

  const __m128i mask1_0 = _mm_loadu_si128((const __m128i *)cKMask1);
  const __m128i mask1_1 = _mm_loadu_si128((const __m128i *)cKMask1 + 1);
  const __m128i mask2_0 = _mm_loadu_si128((const __m128i *)cKMask2);
  const __m128i mask2_1 = _mm_loadu_si128((const __m128i *)cKMask2 + 1);

  uint64_t vAux64;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].qs + QK_K / 8);
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(&vAux64, vX[vI].scales, 8);
    const __m128i cScales8 =
        _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(vAux64 >> 4, vAux64), cM4), 1), cM1);
    const __m128i scales16_0 = _mm_cvtepi8_epi16(cScales8);
    const __m128i scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(cScales8, 8));

    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q2_1_0 = _mm_set_epi64x(iq2s_grid[vQs[1] | ((vQh[vIb32 + 0] << 6) & 0x300)],
                                            iq2s_grid[vQs[0] | ((vQh[vIb32 + 0] << 8) & 0x300)]);
      const __m128i q2_1_1 = _mm_set_epi64x(iq2s_grid[vQs[3] | ((vQh[vIb32 + 0] << 2) & 0x300)],
                                            iq2s_grid[vQs[2] | ((vQh[vIb32 + 0] << 4) & 0x300)]);
      const __m128i q2_2_0 = _mm_set_epi64x(iq2s_grid[vQs[5] | ((vQh[vIb32 + 1] << 6) & 0x300)],
                                            iq2s_grid[vQs[4] | ((vQh[vIb32 + 1] << 8) & 0x300)]);
      const __m128i q2_2_1 = _mm_set_epi64x(iq2s_grid[vQs[7] | ((vQh[vIb32 + 1] << 2) & 0x300)],
                                            iq2s_grid[vQs[6] | ((vQh[vIb32 + 1] << 4) & 0x300)]);
      vQs += 8;

      __m128i aux128_0 = _mm_set1_epi32(vSigns[0] | ((uint32_t)vSigns[1] << 16));
      __m128i aux128_1 = aux128_0;
      aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2_0);
      aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2_1);
      const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
      const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
      const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
      const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

      aux128_0 = _mm_set1_epi32(vSigns[2] | ((uint32_t)vSigns[3] << 16));
      aux128_1 = aux128_0;
      aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2_0);
      aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2_1);
      const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
      const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
      const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
      const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

      vSigns += 4;

      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

      const __m128i p1_0 = _mm_madd_epi16(
          dot1_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(fGetScaleShuffleK4(vIb32 + 0), 0)));
      const __m128i p1_1 = _mm_madd_epi16(
          dot1_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(fGetScaleShuffleK4(vIb32 + 0), 1)));
      const __m128i p2_0 = _mm_madd_epi16(
          dot2_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(fGetScaleShuffleK4(vIb32 + 1), 0)));
      const __m128i p2_1 = _mm_madd_epi16(
          dot2_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(fGetScaleShuffleK4(vIb32 + 1), 1)));
      sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
      sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
      sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
      sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
    }

    vAccumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1),
                                                                             _mm_add_epi32(sumi1_0, sumi2_0)))),
        vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq3_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[2];

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vGas = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ21 =
          _mm256_set_epi32(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]],
                           iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      vQ3 += 8;
      const __m256i cQ22 =
          _mm256_set_epi32(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]],
                           iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      vQ3 += 8;
      memcpy(vAux32, vGas, 8);
      vGas += 8;
      const __m256i cS21 = _mm256_set_epi64x(vSigns64[(vAux32[0] >> 21) & 127], vSigns64[(vAux32[0] >> 14) & 127],
                                             vSigns64[(vAux32[0] >> 7) & 127], vSigns64[(vAux32[0] >> 0) & 127]);
      const __m256i cS22 = _mm256_set_epi64x(vSigns64[(vAux32[1] >> 21) & 127], vSigns64[(vAux32[1] >> 14) & 127],
                                             vSigns64[(vAux32[1] >> 7) & 127], vSigns64[(vAux32[1] >> 0) & 127]);
      const __m256i cQ8s1 = _mm256_sign_epi8(cQ81, cS21);
      const __m256i cQ8s2 = _mm256_sign_epi8(cQ82, cS22);
      const __m256i cDot1 = _mm256_maddubs_epi16(cQ21, cQ8s1);
      const __m256i cDot2 = _mm256_maddubs_epi16(cQ22, cQ8s2);
      const uint16_t cLs1 = vAux32[0] >> 28;
      const uint16_t cLs2 = vAux32[1] >> 28;
      const __m256i cP1 = _mm256_madd_epi16(cDot1, _mm256_set1_epi16(2 * cLs1 + 1));
      const __m256i cP2 = _mm256_madd_epi16(cDot2, _mm256_set1_epi16(2 * cLs2 + 1));
      vSumi1 = _mm256_add_epi32(vSumi1, cP1);
      vSumi2 = _mm256_add_epi32(vSumi2, cP2);
    }

    vAccumf = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.25f * fHsumFloat8(vAccumf);

#elif defined(__AVX__)
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[2];

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vGas = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q2_1_0 =
          _mm_set_epi32(iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      const __m128i q2_1_1 =
          _mm_set_epi32(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]]);
      vQ3 += 8;
      const __m128i q2_2_0 =
          _mm_set_epi32(iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      const __m128i q2_2_1 =
          _mm_set_epi32(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]]);
      vQ3 += 8;
      memcpy(vAux32, vGas, 8);
      vGas += 8;
      const __m128i s2_1_0 = _mm_set_epi64x(vSigns64[(vAux32[0] >> 7) & 127], vSigns64[(vAux32[0] >> 0) & 127]);
      const __m128i s2_1_1 = _mm_set_epi64x(vSigns64[(vAux32[0] >> 21) & 127], vSigns64[(vAux32[0] >> 14) & 127]);
      const __m128i s2_2_0 = _mm_set_epi64x(vSigns64[(vAux32[1] >> 7) & 127], vSigns64[(vAux32[1] >> 0) & 127]);
      const __m128i s2_2_1 = _mm_set_epi64x(vSigns64[(vAux32[1] >> 21) & 127], vSigns64[(vAux32[1] >> 14) & 127]);
      const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
      const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
      const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
      const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
      const uint16_t cLs1 = vAux32[0] >> 28;
      const uint16_t cLs2 = vAux32[1] >> 28;
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2 * cLs2 + 1));
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2 * cLs2 + 1));
      sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
      sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
      sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
      sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
    }

    vAccumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1),
                                                                             _mm_add_epi32(sumi1_0, sumi2_0)))),
        vAccumf);
  }

  *pS = 0.25f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq3_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq3_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__AVX2__)

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m256i cMask1 = _mm256_loadu_si256((const __m256i *)cKMask1);
  const __m256i cMask2 = _mm256_loadu_si256((const __m256i *)cKMask2);

  const __m256i cIdxShift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  const __m256i cIdxMask = _mm256_set1_epi32(256);

  typedef union {
    __m256i vec[2];
    uint32_t index[16];
  } index_t;

  index_t vIdx;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ82 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cIdxL = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)vQs));
      vQs += 16;
      vIdx.vec[0] = _mm256_set1_epi32(vQh[vIb32 + 0]);
      vIdx.vec[1] = _mm256_set1_epi32(vQh[vIb32 + 1]);
      vIdx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(vIdx.vec[0], cIdxShift), cIdxMask);
      vIdx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(vIdx.vec[1], cIdxShift), cIdxMask);
      vIdx.vec[0] = _mm256_or_si256(vIdx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(cIdxL)));
      vIdx.vec[1] = _mm256_or_si256(vIdx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(cIdxL, 1)));

      // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
      // const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
      // const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
      const __m256i cQ21 = _mm256_set_epi32(
          iq3s_grid[vIdx.index[7]], iq3s_grid[vIdx.index[6]], iq3s_grid[vIdx.index[5]], iq3s_grid[vIdx.index[4]],
          iq3s_grid[vIdx.index[3]], iq3s_grid[vIdx.index[2]], iq3s_grid[vIdx.index[1]], iq3s_grid[vIdx.index[0]]);
      const __m256i cQ22 = _mm256_set_epi32(
          iq3s_grid[vIdx.index[15]], iq3s_grid[vIdx.index[14]], iq3s_grid[vIdx.index[13]], iq3s_grid[vIdx.index[12]],
          iq3s_grid[vIdx.index[11]], iq3s_grid[vIdx.index[10]], iq3s_grid[vIdx.index[9]], iq3s_grid[vIdx.index[8]]);

      __m256i vAux256 = _mm256_set1_epi32(vSigns[0] | (vSigns[1] << 16));
      vAux256 = _mm256_and_si256(_mm256_shuffle_epi8(vAux256, cMask1), cMask2);
      const __m256i cS21 = _mm256_cmpeq_epi8(vAux256, cMask2);
      const __m256i cQ8s1 = _mm256_sub_epi8(_mm256_xor_si256(cS21, cQ81), cS21);

      vAux256 = _mm256_set1_epi32(vSigns[2] | (vSigns[3] << 16));
      vAux256 = _mm256_and_si256(_mm256_shuffle_epi8(vAux256, cMask1), cMask2);
      const __m256i cS22 = _mm256_cmpeq_epi8(vAux256, cMask2);
      const __m256i cQ8s2 = _mm256_sub_epi8(_mm256_xor_si256(cS22, cQ82), cS22);

      vSigns += 4;

      const __m256i cDot1 = _mm256_maddubs_epi16(cQ21, cQ8s1);
      const __m256i cDot2 = _mm256_maddubs_epi16(cQ22, cQ8s2);
      const uint16_t cLs1 = vX[vI].scales[vIb32 / 2] & 0xf;
      const uint16_t cLs2 = vX[vI].scales[vIb32 / 2] >> 4;
      const __m256i cP1 = _mm256_madd_epi16(cDot1, _mm256_set1_epi16(2 * cLs1 + 1));
      const __m256i cP2 = _mm256_madd_epi16(cDot2, _mm256_set1_epi16(2 * cLs2 + 1));
      vSumi1 = _mm256_add_epi32(vSumi1, cP1);
      vSumi2 = _mm256_add_epi32(vSumi2, cP2);
    }

    vAccumf = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccumf);
  }

  *pS = fHsumFloat8(vAccumf);

#elif defined(__AVX__)
  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m128i mask1_0 = _mm_loadu_si128((const __m128i *)cKMask1);
  const __m128i mask1_1 = _mm_loadu_si128((const __m128i *)cKMask1 + 1);
  const __m128i mask2_0 = _mm_loadu_si128((const __m128i *)cKMask2);
  const __m128i mask2_1 = _mm_loadu_si128((const __m128i *)cKMask2 + 1);

  const __m128i idx_mul_0 = _mm_set_epi32(32, 64, 128, 256);
  const __m128i idx_mul_1 = _mm_set_epi32(2, 4, 8, 16);
  const __m128i cIdxMask = _mm_set1_epi32(256);

  typedef union {
    __m128i vec[4];
    uint32_t index[16];
  } index_t;

  index_t vIdx;

  __m256 vAccumf = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i qs_tmp = _mm_loadu_si128((const __m128i *)vQs);
      const __m128i idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
      const __m128i idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8));
      vQs += 16;
      vIdx.vec[0] = _mm_set1_epi32(vQh[vIb32 + 0]);
      vIdx.vec[1] = vIdx.vec[0];
      vIdx.vec[2] = _mm_set1_epi32(vQh[vIb32 + 1]);
      vIdx.vec[3] = vIdx.vec[2];

      vIdx.vec[0] = _mm_and_si128(_mm_mullo_epi32(vIdx.vec[0], idx_mul_0), cIdxMask);
      vIdx.vec[1] = _mm_and_si128(_mm_mullo_epi32(vIdx.vec[1], idx_mul_1), cIdxMask);
      vIdx.vec[2] = _mm_and_si128(_mm_mullo_epi32(vIdx.vec[2], idx_mul_0), cIdxMask);
      vIdx.vec[3] = _mm_and_si128(_mm_mullo_epi32(vIdx.vec[3], idx_mul_1), cIdxMask);

      vIdx.vec[0] = _mm_or_si128(vIdx.vec[0], _mm_cvtepi16_epi32(idx_l_0));
      vIdx.vec[1] = _mm_or_si128(vIdx.vec[1], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
      vIdx.vec[2] = _mm_or_si128(vIdx.vec[2], _mm_cvtepi16_epi32(idx_l_1));
      vIdx.vec[3] = _mm_or_si128(vIdx.vec[3], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));

      const __m128i q2_1_0 = _mm_set_epi32(iq3s_grid[vIdx.index[3]], iq3s_grid[vIdx.index[2]], iq3s_grid[vIdx.index[1]],
                                           iq3s_grid[vIdx.index[0]]);
      const __m128i q2_1_1 = _mm_set_epi32(iq3s_grid[vIdx.index[7]], iq3s_grid[vIdx.index[6]], iq3s_grid[vIdx.index[5]],
                                           iq3s_grid[vIdx.index[4]]);
      const __m128i q2_2_0 = _mm_set_epi32(iq3s_grid[vIdx.index[11]], iq3s_grid[vIdx.index[10]],
                                           iq3s_grid[vIdx.index[9]], iq3s_grid[vIdx.index[8]]);
      const __m128i q2_2_1 = _mm_set_epi32(iq3s_grid[vIdx.index[15]], iq3s_grid[vIdx.index[14]],
                                           iq3s_grid[vIdx.index[13]], iq3s_grid[vIdx.index[12]]);

      __m128i aux128_0 = _mm_set1_epi32(vSigns[0] | (vSigns[1] << 16));
      __m128i aux128_1 = aux128_0;
      aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2_0);
      aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2_1);
      const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
      const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
      const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
      const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

      aux128_0 = _mm_set1_epi32(vSigns[2] | (vSigns[3] << 16));
      aux128_1 = aux128_0;
      aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0, mask1_0), mask2_0);
      aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1, mask1_1), mask2_1);
      const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
      const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
      const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
      const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

      vSigns += 4;

      const __m128i dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
      const __m128i dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
      const __m128i dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
      const __m128i dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
      const uint16_t cLs1 = vX[vI].scales[vIb32 / 2] & 0xf;
      const uint16_t cLs2 = vX[vI].scales[vIb32 / 2] >> 4;
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2 * cLs1 + 1));
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2 * cLs2 + 1));
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2 * cLs2 + 1));
      sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
      sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
      sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
      sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
    }

    vAccumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1),
                                                                             _mm_add_epi32(sumi1_0, sumi2_0)))),
        vAccumf);
  }

  *pS = fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq3_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq1_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __AVX2__

  __m256 vAccum = _mm256_setzero_ps();
  float vAccum1 = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint16_t *vQh = vX[vI].qh;

    __m256i vSumi = _mm256_setzero_si256();
    int vSumi1 = 0;
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
#ifdef __BMI2__
      const uint64_t cPackedIdx1 =
          _pdep_u64(*(const uint32_t *)vQs, 0x00ff00ff00ff00ffULL) | _pdep_u64(vQh[vIb], 0x700070007000700ULL);
      const uint64_t cPackedIdx2 = _pdep_u64(*(const uint32_t *)(vQs + 4), 0x00ff00ff00ff00ffULL) |
                                   _pdep_u64(vQh[vIb + 1], 0x700070007000700ULL);
      const uint16_t *vIdx1 = (const uint16_t *)(&cPackedIdx1);
      const uint16_t *vIdx2 = (const uint16_t *)(&cPackedIdx2);
      const __m256i cQ1b1 =
          _mm256_set_epi64x(iq1s_grid[vIdx1[3]], iq1s_grid[vIdx1[2]], iq1s_grid[vIdx1[1]], iq1s_grid[vIdx1[0]]);
      const __m256i cQ1b2 =
          _mm256_set_epi64x(iq1s_grid[vIdx2[3]], iq1s_grid[vIdx2[2]], iq1s_grid[vIdx2[1]], iq1s_grid[vIdx2[0]]);
#else
      const __m256i cQ1b1 = _mm256_set_epi64x(
          iq1s_grid[vQs[3] | ((vQh[vIb + 0] >> 1) & 0x700)], iq1s_grid[vQs[2] | ((vQh[vIb + 0] << 2) & 0x700)],
          iq1s_grid[vQs[1] | ((vQh[vIb + 0] << 5) & 0x700)], iq1s_grid[vQs[0] | ((vQh[vIb + 0] << 8) & 0x700)]);
      const __m256i cQ1b2 = _mm256_set_epi64x(
          iq1s_grid[vQs[7] | ((vQh[vIb + 1] >> 1) & 0x700)], iq1s_grid[vQs[6] | ((vQh[vIb + 1] << 2) & 0x700)],
          iq1s_grid[vQs[5] | ((vQh[vIb + 1] << 5) & 0x700)], iq1s_grid[vQs[4] | ((vQh[vIb + 1] << 8) & 0x700)]);
#endif
      vQs += 8;
      const __m256i cQ8b1 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ8b2 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      const __m256i cDot1 = fMulAddEpi8(cQ1b1, cQ8b1);
      const __m256i cDot2 = fMulAddEpi8(cQ1b2, cQ8b2);
      const int16_t cLs1 = 2 * ((vQh[vIb + 0] >> 12) & 7) + 1;
      const int16_t cLs2 = 2 * ((vQh[vIb + 1] >> 12) & 7) + 1;
      const __m256i cP1 = _mm256_madd_epi16(cDot1, _mm256_set1_epi16(cLs1));
      const __m256i cP2 = _mm256_madd_epi16(cDot2, _mm256_set1_epi16(cLs2));

      vSumi = _mm256_add_epi32(vSumi, _mm256_add_epi32(cP1, cP2));
      vSumi1 += (vY[vI].bsums[2 * vIb + 0] + vY[vI].bsums[2 * vIb + 1]) * (vQh[vIb + 0] & 0x8000 ? -1 : 1) * cLs1 +
                (vY[vI].bsums[2 * vIb + 2] + vY[vI].bsums[2 * vIb + 3]) * (vQh[vIb + 1] & 0x8000 ? -1 : 1) * cLs2;
    }

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    vAccum = _mm256_fmadd_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(vSumi), vAccum);
    vAccum1 += cD * vSumi1;
  }

  *pS = fHsumFloat8(vAccum) + IQ1S_DELTA * vAccum1;

#elif defined __AVX__
  __m256 vAccum = _mm256_setzero_ps();
  float vAccum1 = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint16_t *vQh = vX[vI].qh;

    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    int vSumi1 = 0;
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const __m128i q1b_1_0 = _mm_set_epi64x(iq1s_grid[vQs[1] | ((vQh[vIb + 0] << 5) & 0x700)],
                                             iq1s_grid[vQs[0] | ((vQh[vIb + 0] << 8) & 0x700)]);
      const __m128i q1b_1_1 = _mm_set_epi64x(iq1s_grid[vQs[3] | ((vQh[vIb + 0] >> 1) & 0x700)],
                                             iq1s_grid[vQs[2] | ((vQh[vIb + 0] << 2) & 0x700)]);
      const __m128i q1b_2_0 = _mm_set_epi64x(iq1s_grid[vQs[5] | ((vQh[vIb + 1] << 5) & 0x700)],
                                             iq1s_grid[vQs[4] | ((vQh[vIb + 1] << 8) & 0x700)]);
      const __m128i q1b_2_1 = _mm_set_epi64x(iq1s_grid[vQs[7] | ((vQh[vIb + 1] >> 1) & 0x700)],
                                             iq1s_grid[vQs[6] | ((vQh[vIb + 1] << 2) & 0x700)]);
      vQs += 8;
      const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
      const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
      const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
      const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);
      const int16_t cLs1 = 2 * ((vQh[vIb + 0] >> 12) & 7) + 1;
      const int16_t cLs2 = 2 * ((vQh[vIb + 1] >> 12) & 7) + 1;
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(cLs1));
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(cLs1));
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(cLs2));
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(cLs2));

      sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
      sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
      vSumi1 += (vY[vI].bsums[2 * vIb + 0] + vY[vI].bsums[2 * vIb + 1]) * (vQh[vIb + 0] & 0x8000 ? -1 : 1) * cLs1 +
                (vY[vI].bsums[2 * vIb + 2] + vY[vI].bsums[2 * vIb + 3]) * (vQh[vIb + 1] & 0x8000 ? -1 : 1) * cLs2;
    }

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    vAccum =
        _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(cD), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), vAccum);
    vAccum1 += cD * vSumi1;
  }

  *pS = fHsumFloat8(vAccum) + IQ1S_DELTA * vAccum1;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq1_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq1_m_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_m *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  iq1m_scale_t vScale;

#if defined __AVX2__

  const __m256i cMask = _mm256_set1_epi16(0x7);
  const __m256i cMone = _mm256_set1_epi16(1);
  const __m256i cMone8 = _mm256_set1_epi8(1);
  const __m256i cMtwo8 = _mm256_set1_epi8(2);
  // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
  const __m256i cScalesShift = _mm256_set_epi64x(9, 3, 6, 0);

  __m256 vAccum1 = _mm256_setzero_ps();
  __m256 vAccum2 = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);
    // Extract 3-bit scales (16 values)
    __m256i vScales = _mm256_set1_epi64x(*(const uint64_t *)vSc);
    vScales = _mm256_srlv_epi64(vScales, cScalesShift);
    vScales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(vScales, cMask), 1), cMone);

    // Indices to repeat each scale 8 times.
    __m256i vScalesIdx1 = _mm256_set1_epi16(0x0100);
    __m256i vScalesIdx2 = _mm256_add_epi8(vScalesIdx1, _mm256_set1_epi8(8));

    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
#ifdef __BMI2__
      const uint64_t cPackedIdx1 = _pdep_u64(*(const uint32_t *)vQs, 0x00ff00ff00ff00ffULL) |
                                   _pdep_u64(*(const uint16_t *)(vQh) & 0x7777, 0xf000f000f000f00ULL);
      const uint64_t cPackedIdx2 = _pdep_u64(*(const uint32_t *)(vQs + 4), 0x00ff00ff00ff00ffULL) |
                                   _pdep_u64(*(const uint16_t *)(vQh + 2) & 0x7777, 0xf000f000f000f00ULL);
      const uint16_t *vIdx1 = (const uint16_t *)(&cPackedIdx1);
      const uint16_t *vIdx2 = (const uint16_t *)(&cPackedIdx2);
      const __m256i cQ1b1 =
          _mm256_set_epi64x(iq1s_grid[vIdx1[3]], iq1s_grid[vIdx1[2]], iq1s_grid[vIdx1[1]], iq1s_grid[vIdx1[0]]);
      const __m256i cQ1b2 =
          _mm256_set_epi64x(iq1s_grid[vIdx2[3]], iq1s_grid[vIdx2[2]], iq1s_grid[vIdx2[1]], iq1s_grid[vIdx2[0]]);

      // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
      const uint64_t cDeltaSign = _pdep_u64(*(const uint32_t *)(vQh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
      const __m256i cDelta1 = _mm256_or_si256(cMone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(cDeltaSign)));
      const __m256i cDelta2 = _mm256_or_si256(cMone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(cDeltaSign >> 32)));
#else
      const __m256i cQ1b1 = _mm256_set_epi64x(
          iq1s_grid[vQs[3] | (((uint16_t)vQh[1] << 4) & 0x700)], iq1s_grid[vQs[2] | (((uint16_t)vQh[1] << 8) & 0x700)],
          iq1s_grid[vQs[1] | (((uint16_t)vQh[0] << 4) & 0x700)], iq1s_grid[vQs[0] | (((uint16_t)vQh[0] << 8) & 0x700)]);
      const __m256i cQ1b2 = _mm256_set_epi64x(
          iq1s_grid[vQs[7] | (((uint16_t)vQh[3] << 4) & 0x700)], iq1s_grid[vQs[6] | (((uint16_t)vQh[3] << 8) & 0x700)],
          iq1s_grid[vQs[5] | (((uint16_t)vQh[2] << 4) & 0x700)], iq1s_grid[vQs[4] | (((uint16_t)vQh[2] << 8) & 0x700)]);

      const __m256i cDelta1 = _mm256_set_epi64x(vQh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m256i cDelta2 = _mm256_set_epi64x(vQh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                vQh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
#endif
      const __m256i cQ8b1 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ8b2 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;

      const __m256i cDot1 = fMulAddEpi8(cQ1b1, cQ8b1);
      const __m256i cDot2 = fMulAddEpi8(cQ1b2, cQ8b2);
      const __m256i cDot3 = _mm256_maddubs_epi16(cMone8, _mm256_sign_epi8(cQ8b1, cDelta1));
      const __m256i cDot4 = _mm256_maddubs_epi16(cMone8, _mm256_sign_epi8(cQ8b2, cDelta2));

      __m256i vScale1 = _mm256_shuffle_epi8(vScales, vScalesIdx1);
      __m256i vScale2 = _mm256_shuffle_epi8(vScales, vScalesIdx2);

      vScalesIdx1 = _mm256_add_epi8(vScalesIdx1, cMtwo8);
      vScalesIdx2 = _mm256_add_epi8(vScalesIdx2, cMtwo8);

      const __m256i cP1 = _mm256_madd_epi16(cDot1, vScale1);
      const __m256i cP2 = _mm256_madd_epi16(cDot2, vScale2);
      const __m256i cP3 = _mm256_madd_epi16(cDot3, vScale1);
      const __m256i cP4 = _mm256_madd_epi16(cDot4, vScale2);

      vSumi1 = _mm256_add_epi32(vSumi1, _mm256_add_epi32(cP1, cP2));
      vSumi2 = _mm256_add_epi32(vSumi2, _mm256_add_epi32(cP3, cP4));

      vQs += 8;
      vQh += 4;
    }

    const __m256 cD = _mm256_set1_ps(vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16));

    vAccum1 = _mm256_fmadd_ps(cD, _mm256_cvtepi32_ps(vSumi1), vAccum1);
    vAccum2 = _mm256_fmadd_ps(cD, _mm256_cvtepi32_ps(vSumi2), vAccum2);
  }

  *pS = fHsumFloat8(vAccum1) + IQ1M_DELTA * fHsumFloat8(vAccum2);

#elif defined __AVX__
  const __m128i cMask = _mm_set1_epi16(0x7);
  const __m128i cMone = _mm_set1_epi16(1);

  __m256 vAccum1 = _mm256_setzero_ps();
  __m256 vAccum2 = _mm256_setzero_ps();
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const __m128i q1b_1_0 = _mm_set_epi64x(iq1s_grid[vQs[1] | (((uint16_t)vQh[0] << 4) & 0x700)],
                                             iq1s_grid[vQs[0] | (((uint16_t)vQh[0] << 8) & 0x700)]);
      const __m128i q1b_1_1 = _mm_set_epi64x(iq1s_grid[vQs[3] | (((uint16_t)vQh[1] << 4) & 0x700)],
                                             iq1s_grid[vQs[2] | (((uint16_t)vQh[1] << 8) & 0x700)]);
      const __m128i q1b_2_0 = _mm_set_epi64x(iq1s_grid[vQs[5] | (((uint16_t)vQh[2] << 4) & 0x700)],
                                             iq1s_grid[vQs[4] | (((uint16_t)vQh[2] << 8) & 0x700)]);
      const __m128i q1b_2_1 = _mm_set_epi64x(iq1s_grid[vQs[7] | (((uint16_t)vQh[3] << 4) & 0x700)],
                                             iq1s_grid[vQs[6] | (((uint16_t)vQh[3] << 8) & 0x700)]);
      const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;

      const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
      const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
      const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
      const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);

      const __m128i delta1_0 = _mm_set_epi64x(vQh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                              vQh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m128i delta1_1 = _mm_set_epi64x(vQh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                              vQh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m128i delta2_0 = _mm_set_epi64x(vQh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                              vQh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m128i delta2_1 = _mm_set_epi64x(vQh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                              vQh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

      const __m128i dot3_0 = mul_add_epi8_sse(delta1_0, q8b_1_0);
      const __m128i dot3_1 = mul_add_epi8_sse(delta1_1, q8b_1_1);
      const __m128i dot4_0 = mul_add_epi8_sse(delta2_0, q8b_2_0);
      const __m128i dot4_1 = mul_add_epi8_sse(delta2_1, q8b_2_1);

      __m128i scale1_0 = _mm_set1_epi16(vSc[vIb / 2] >> 0);
      __m128i scale1_1 = _mm_set1_epi16(vSc[vIb / 2] >> 3);
      __m128i scale2_0 = _mm_set1_epi16(vSc[vIb / 2] >> 6);
      __m128i scale2_1 = _mm_set1_epi16(vSc[vIb / 2] >> 9);

      scale1_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_0, cMask), 1), cMone);
      scale1_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_1, cMask), 1), cMone);
      scale2_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_0, cMask), 1), cMone);
      scale2_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_1, cMask), 1), cMone);
      const __m128i p1_0 = _mm_madd_epi16(dot1_0, scale1_0);
      const __m128i p1_1 = _mm_madd_epi16(dot1_1, scale1_1);
      const __m128i p2_0 = _mm_madd_epi16(dot2_0, scale2_0);
      const __m128i p2_1 = _mm_madd_epi16(dot2_1, scale2_1);
      const __m128i p3_0 = _mm_madd_epi16(dot3_0, scale1_0);
      const __m128i p3_1 = _mm_madd_epi16(dot3_1, scale1_1);
      const __m128i p4_0 = _mm_madd_epi16(dot4_0, scale2_0);
      const __m128i p4_1 = _mm_madd_epi16(dot4_1, scale2_1);

      sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
      sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
      sumi2_0 = _mm_add_epi32(sumi2_0, _mm_add_epi32(p3_0, p4_0));
      sumi2_1 = _mm_add_epi32(sumi2_1, _mm_add_epi32(p3_1, p4_1));

      vQs += 8;
      vQh += 4;
    }

    const __m256 cD = _mm256_set1_ps(vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16));

    vAccum1 = _mm256_add_ps(_mm256_mul_ps(cD, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), vAccum1);
    vAccum2 = _mm256_add_ps(_mm256_mul_ps(cD, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi2_1, sumi2_0))), vAccum2);
  }

  *pS = fHsumFloat8(vAccum1) + IQ1M_DELTA * fHsumFloat8(vAccum2);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(vScale);
  ggml_vec_dot_iq1_m_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq4_nl_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK4_NL == 0);
  static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

  const block_iq4_nl *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK4_NL;

  int vIb = 0;
  float vSumf = 0;

#if defined __AVX2__

  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
  const __m128i cM4b = _mm_set1_epi8(0x0f);
  const __m256i cMone = _mm256_set1_epi16(1);

  __m256 vAccum1 = _mm256_setzero_ps();
  __m256 vAccum2 = _mm256_setzero_ps();
  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vX[vIb + 0].qs);
    const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m256i cQ8b1 = _mm256_loadu_si256((const __m256i *)vY[vIb + 0].qs);
    const __m256i cQ8b2 = _mm256_loadu_si256((const __m256i *)vY[vIb + 1].qs);
    const __m256i cQ4b1 =
        MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b)),
                        _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b)));
    const __m256i cQ4b2 =
        MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b)),
                        _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b)));
    const __m256i cP161 = fMulAddEpi8(cQ4b1, cQ8b1);
    const __m256i cP162 = fMulAddEpi8(cQ4b2, cQ8b2);
    const __m256i cP1 = _mm256_madd_epi16(cP161, cMone);
    const __m256i cP2 = _mm256_madd_epi16(cP162, cMone);
    vAccum1 =
        _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * GGML_CPU_FP16_TO_FP32(vX[vIb + 0].d)),
                        _mm256_cvtepi32_ps(cP1), vAccum1);
    vAccum2 =
        _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d)),
                        _mm256_cvtepi32_ps(cP2), vAccum2);
  }

  vSumf = fHsumFloat8(_mm256_add_ps(vAccum1, vAccum2));

#elif defined __AVX__
  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
  const __m128i cM4b = _mm_set1_epi8(0x0f);

  __m256 accum = _mm256_setzero_ps();
  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vX[vIb + 0].qs);
    const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vX[vIb + 1].qs);
    const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs);
    const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 0].qs + 1);
    const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs);
    const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vY[vIb + 1].qs + 1);

    const __m128i q4b_1_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b));
    const __m128i q4b_1_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b));
    const __m128i q4b_2_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b));
    const __m128i q4b_2_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b));

    const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
    const __m256 deltas = quad_fp16_delta_float(vX[vIb].d, vY[vIb].d, vX[vIb + 1].d, vY[vIb + 1].d);
    accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
  }

  vSumf = fHsumFloat8(accum);

#endif
  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    int vSumi1 = 0, vSumi2 = 0;
    for (int vJ = 0; vJ < QK4_NL / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_iq4nl[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK4_NL / 2] * kvalues_iq4nl[vX[vIb].qs[vJ] >> 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;
}

void ggml_vec_dot_iq4_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __AVX2__

  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
  const __m128i cM4b = _mm_set1_epi8(0x0f);

  __m256 vAccum = _mm256_setzero_ps();
  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const uint8_t *vQs = vX[vIbl].qs;
    const int8_t *vQ8 = vY[vIbl].qs;
    uint16_t vSh = vX[vIbl].scales_h;
    __m256i vSumi1 = _mm256_setzero_si256();
    __m256i vSumi2 = _mm256_setzero_si256();
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vQs);
      vQs += 16;
      const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vQs);
      vQs += 16;
      const __m256i cQ8b1 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ8b2 = _mm256_loadu_si256((const __m256i *)vQ8);
      vQ8 += 32;
      const __m256i cQ4b1 =
          MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b)),
                          _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b)));
      const __m256i cQ4b2 =
          MM256_SET_M128I(_mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b)),
                          _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b)));
      const __m256i cP161 = fMulAddEpi8(cQ4b1, cQ8b1);
      const __m256i cP162 = fMulAddEpi8(cQ4b2, cQ8b2);
      const int16_t cLs1 = ((vX[vIbl].scales_l[vIb / 2] & 0xf) | ((vSh << 4) & 0x30)) - 32;
      const int16_t cLs2 = ((vX[vIbl].scales_l[vIb / 2] >> 4) | ((vSh << 2) & 0x30)) - 32;
      vSh >>= 4;
      const __m256i cP1 = _mm256_madd_epi16(cP161, _mm256_set1_epi16(cLs1));
      const __m256i cP2 = _mm256_madd_epi16(cP162, _mm256_set1_epi16(cLs2));
      vSumi1 = _mm256_add_epi32(cP1, vSumi1);
      vSumi2 = _mm256_add_epi32(cP2, vSumi2);
    }
    vAccum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d),
                             _mm256_cvtepi32_ps(_mm256_add_epi32(vSumi1, vSumi2)), vAccum);
  }

  *pS = fHsumFloat8(vAccum);

#elif defined __AVX__
  const __m128i cValues128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
  const __m128i cM4b = _mm_set1_epi8(0x0f);

  __m256 vAccum = _mm256_setzero_ps();
  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const uint8_t *vQs = vX[vIbl].qs;
    const int8_t *vQ8 = vY[vIbl].qs;
    uint16_t vSh = vX[vIbl].scales_h;
    __m128i sumi1_0 = _mm_setzero_si128();
    __m128i sumi1_1 = _mm_setzero_si128();
    __m128i sumi2_0 = _mm_setzero_si128();
    __m128i sumi2_1 = _mm_setzero_si128();
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const __m128i cQ4bits1 = _mm_loadu_si128((const __m128i *)vQs);
      vQs += 16;
      const __m128i cQ4bits2 = _mm_loadu_si128((const __m128i *)vQs);
      vQs += 16;
      const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)vQ8);
      vQ8 += 16;
      const __m128i q4b_1_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits1, cM4b));
      const __m128i q4b_1_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits1, 4), cM4b));
      const __m128i q4b_2_0 = _mm_shuffle_epi8(cValues128, _mm_and_si128(cQ4bits2, cM4b));
      const __m128i q4b_2_1 = _mm_shuffle_epi8(cValues128, _mm_and_si128(_mm_srli_epi16(cQ4bits2, 4), cM4b));
      const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
      const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
      const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
      const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
      const int16_t cLs1 = ((vX[vIbl].scales_l[vIb / 2] & 0xf) | ((vSh << 4) & 0x30)) - 32;
      const int16_t cLs2 = ((vX[vIbl].scales_l[vIb / 2] >> 4) | ((vSh << 2) & 0x30)) - 32;
      vSh >>= 4;
      const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(cLs1));
      const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(cLs1));
      const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(cLs2));
      const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(cLs2));
      sumi1_0 = _mm_add_epi32(p_1_0, sumi1_0);
      sumi1_1 = _mm_add_epi32(p_1_1, sumi1_1);
      sumi2_0 = _mm_add_epi32(p_2_0, sumi2_0);
      sumi2_1 = _mm_add_epi32(p_2_1, sumi2_1);
    }
    __m128i sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
    __m128i sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
    vAccum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d),
                                         _mm256_cvtepi32_ps(MM256_SET_M128I(sumi12_1, sumi12_0))),
                           vAccum);
  }

  *pS = fHsumFloat8(vAccum);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
