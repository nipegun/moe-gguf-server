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
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__ARM_NEON)
#define B1(c, s, n) 0x##n##c, 0x##n##s
#define B2(c, s, n) B1(c, s, n##c), B1(c, s, n##s)
#define B3(c, s, n) B2(c, s, n##c), B2(c, s, n##s)
#define B4(c, s, n) B3(c, s, n##c), B3(c, s, n##s)
#define B5(c, s, n) B4(c, s, n##c), B4(c, s, n##s)
#define B6(c, s, n) B5(c, s, n##c), B5(c, s, n##s)
#define B7(c, s, n) B6(c, s, n##c), B6(c, s, n##s)
#define B8(c, s) B7(c, s, c), B7(c, s, s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = {B8(00, 10)}; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = {B8(10, 00)}; // (!b) << 4
#endif

void quantize_row_q8_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_NEON)
  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t aSrcv[8];
    float32x4_t aAsrcv[8];
    float32x4_t aAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = vld1q_f32(pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vabsq_f32(aSrcv[vJ]);

    for (int vJ = 0; vJ < 4; vJ++)
      aAmaxv[2 * vJ] = vmaxq_f32(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      aAmaxv[4 * vJ] = vmaxq_f32(aAmaxv[4 * vJ], aAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      aAmaxv[8 * vJ] = vmaxq_f32(aAmaxv[8 * vJ], aAmaxv[8 * vJ + 4]);

    const float cAmax = vmaxvq_f32(aAmaxv[0]);

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < 8; vJ++) {
      const float32x4_t cV = vmulq_n_f32(aSrcv[vJ], cId);
      const int32x4_t cVi = vcvtnq_s32_f32(cV);

      vY[vI].qs[4 * vJ + 0] = vgetq_lane_s32(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = vgetq_lane_s32(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = vgetq_lane_s32(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = vgetq_lane_s32(cVi, 3);
    }
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
#if defined(__ARM_NEON)
  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t aSrcv[8];
    float32x4_t aAsrcv[8];
    float32x4_t aAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = vld1q_f32(pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vabsq_f32(aSrcv[vJ]);

    for (int vJ = 0; vJ < 4; vJ++)
      aAmaxv[2 * vJ] = vmaxq_f32(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      aAmaxv[4 * vJ] = vmaxq_f32(aAmaxv[4 * vJ], aAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      aAmaxv[8 * vJ] = vmaxq_f32(aAmaxv[8 * vJ], aAmaxv[8 * vJ + 4]);

    const float cAmax = vmaxvq_f32(aAmaxv[0]);

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    int32x4_t vAccv = vdupq_n_s32(0);

    for (int vJ = 0; vJ < 8; vJ++) {
      const float32x4_t cV = vmulq_n_f32(aSrcv[vJ], cId);
      const int32x4_t cVi = vcvtnq_s32_f32(cV);

      vY[vI].qs[4 * vJ + 0] = vgetq_lane_s32(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = vgetq_lane_s32(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = vgetq_lane_s32(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = vgetq_lane_s32(cVi, 3);

      vAccv = vaddq_s32(vAccv, cVi);
    }

    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * vaddvq_s32(vAccv));
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

void ggml_vec_dot_q1_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK1_0; // 128
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_NEON)
  float32x4_t vSumv = vdupq_n_f32(0.0f);

  for (int vI = 0; vI < cNb; vI++) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    // Process 4 Q8_0 blocks (each has 32 elements)
    for (int vK = 0; vK < 4; vK++) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vI * 4 + vK];
      const float cD1 = GGML_CPU_FP16_TO_FP32(vYb->d);

      // Get the 4 bytes of bits for this Q8_0 block (32 bits = 4 bytes)
      // Bits are at offset k*4 bytes in x[i].qs
      const uint8_t *vBits = &vX[vI].qs[vK * 4];

      // Load 32 int8 values from y
      const int8x16_t cY0 = vld1q_s8(vYb->qs);
      const int8x16_t cY1 = vld1q_s8(vYb->qs + 16);

      // Byte 0-1: bits for y0[0..15]
      const uint64_t cExpand0 = table_b2b_0[vBits[0]];
      const uint64_t cExpand1 = table_b2b_0[vBits[1]];
      // Byte 2-3: bits for y1[0..15]
      const uint64_t cExpand2 = table_b2b_0[vBits[2]];
      const uint64_t cExpand3 = table_b2b_0[vBits[3]];

      // Build the sign vectors by reinterpreting the table values
      uint8x8_t vE0 = vcreate_u8(cExpand0);
      uint8x8_t vE1 = vcreate_u8(cExpand1);
      uint8x8_t vE2 = vcreate_u8(cExpand2);
      uint8x8_t vE3 = vcreate_u8(cExpand3);

      // Shift right by 4 to get 0 or 1
      int8x8_t vS0 = vreinterpret_s8_u8(vshr_n_u8(vE0, 4));
      int8x8_t vS1 = vreinterpret_s8_u8(vshr_n_u8(vE1, 4));
      int8x8_t vS2 = vreinterpret_s8_u8(vshr_n_u8(vE2, 4));
      int8x8_t vS3 = vreinterpret_s8_u8(vshr_n_u8(vE3, 4));

      // Convert 0/1 to -1/+1: sign = 2*val - 1
      int8x8_t vOne = vdup_n_s8(1);
      vS0 = vsub_s8(vadd_s8(vS0, vS0), vOne); // 2*s0 - 1
      vS1 = vsub_s8(vadd_s8(vS1, vS1), vOne);
      vS2 = vsub_s8(vadd_s8(vS2, vS2), vOne);
      vS3 = vsub_s8(vadd_s8(vS3, vS3), vOne);

      // Combine into 16-element vectors
      int8x16_t vSigns0 = vcombine_s8(vS0, vS1);
      int8x16_t vSigns1 = vcombine_s8(vS2, vS3);

      // Multiply signs with y values and accumulate
      // dot(signs, y) where signs are +1/-1
      int32x4_t vP0 = ggml_vdotq_s32(vdupq_n_s32(0), vSigns0, cY0);
      int32x4_t vP1 = ggml_vdotq_s32(vP0, vSigns1, cY1);

      // Scale by d1 and accumulate
      vSumv = vmlaq_n_f32(vSumv, vcvtq_f32_s32(vP1), cD0 * cD1);
    }
  }

  *pS = vaddvq_f32(vSumv);
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q1_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q2_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK2_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  float vSumf = 0.0f;

#if defined(__ARM_NEON)
  // Replicate pattern: each byte repeated 4 times
  static const uint8_t cTblIdxLo[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  static const uint8_t cTblIdxHi[16] = {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
  // Right-shift amounts: 0,2,4,6 repeated for each group of 4
  static const int8_t cShiftVals[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};

  const uint8x16_t cIdxLo = vld1q_u8(cTblIdxLo);
  const uint8x16_t cIdxHi = vld1q_u8(cTblIdxHi);
  const int8x16_t cShifts = vld1q_s8(cShiftVals);
  const uint8x16_t cMask2 = vdupq_n_u8(0x03);
  const int8x16_t cOne = vdupq_n_s8(1);

  float32x4_t vSumv = vdupq_n_f32(0.0f);

  for (int vI = 0; vI < cNb; vI++) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
    for (int vK = 0; vK < 2; vK++) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vI * 2 + vK];
      const float cD1 = GGML_CPU_FP16_TO_FP32(vYb->d);

      // Load 8 bytes of packed 2-bit values
      const uint8x8_t cRaw = vld1_u8(&vX[vI].qs[vK * 8]);
      const uint8x16_t cRaw16 = vcombine_u8(cRaw, cRaw);

      // First 16 elements: replicate bytes 0-3, shift, mask, subtract 1
      uint8x16_t vBytes0 = ggml_vqtbl1q_u8(cRaw16, cIdxLo);
      int8x16_t vQv0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vBytes0, cShifts), cMask2)), cOne);

      // Second 16 elements: replicate bytes 4-7, shift, mask, subtract 1
      uint8x16_t vBytes1 = ggml_vqtbl1q_u8(cRaw16, cIdxHi);
      int8x16_t vQv1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vBytes1, cShifts), cMask2)), cOne);

      // Load Q8_0 values and dot product
      const int8x16_t cY0 = vld1q_s8(vYb->qs);
      const int8x16_t cY1 = vld1q_s8(vYb->qs + 16);

      int32x4_t vP0 = ggml_vdotq_s32(vdupq_n_s32(0), vQv0, cY0);
      int32x4_t vP1 = ggml_vdotq_s32(vP0, vQv1, cY1);

      vSumv = vmlaq_n_f32(vSumv, vcvtq_f32_s32(vP1), cD0 * cD1);
    }
  }

  vSumf = vaddvq_f32(vSumv);
#else
  ggml_vec_dot_q2_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
  return;
#endif

  *pS = vSumf;
}

void ggml_vec_dot_q4_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  assert((pNrc == 2) || (pNrc == 1));
#else
  assert(pNrc == 1);
#endif
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const block_q4_0 *GGML_RESTRICT vVx0 = pVx;
    const block_q4_0 *GGML_RESTRICT vVx1 = (const block_q4_0 *)((const uint8_t *)pVx + pBx);
    const block_q8_0 *GGML_RESTRICT vVy0 = pVy;
    const block_q8_0 *GGML_RESTRICT vVy1 = (const block_q8_0 *)((const uint8_t *)pVy + pBy);

    float32x4_t vSumv0 = vdupq_n_f32(0.0f);

    for (int vI = 0; vI < cNb; vI++) {
      const block_q4_0 *GGML_RESTRICT vBX0 = &vVx0[vI];
      const block_q4_0 *GGML_RESTRICT vBX1 = &vVx1[vI];
      const block_q8_0 *GGML_RESTRICT vBY0 = &vVy0[vI];
      const block_q8_0 *GGML_RESTRICT vBY1 = &vVy1[vI];

      const uint8x16_t cM4b = vdupq_n_u8(0x0F);
      const int8x16_t cS8b = vdupq_n_s8(0x8);

      const uint8x16_t cV00 = vld1q_u8(vBX0->qs);
      const uint8x16_t cV01 = vld1q_u8(vBX1->qs);

      // 4-bit -> 8-bit
      const int8x16_t cV00l = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
      const int8x16_t cV00h = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
      const int8x16_t cV01l = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
      const int8x16_t cV01h = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

      // sub 8
      const int8x16_t cX0L = vsubq_s8(cV00l, cS8b);
      const int8x16_t cX0H = vsubq_s8(cV00h, cS8b);
      const int8x16_t cX1L = vsubq_s8(cV01l, cS8b);
      const int8x16_t cX1H = vsubq_s8(cV01h, cS8b);

      // load y
      const int8x16_t cY0L = vld1q_s8(vBY0->qs);
      const int8x16_t cY0H = vld1q_s8(vBY0->qs + 16);
      const int8x16_t cY1L = vld1q_s8(vBY1->qs);
      const int8x16_t cY1H = vld1q_s8(vBY1->qs + 16);

      float32_t _scale[4] = {GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY1->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY1->d)};
      float32x4_t scale = vld1q_f32(_scale);

      int8x16_t vL0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));
      int8x16_t vL1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));

      int8x16_t vL2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));
      int8x16_t vL3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));

      int8x16_t vR0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));
      int8x16_t vR1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));

      int8x16_t vR2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));
      int8x16_t vR3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));

      vSumv0 = vmlaq_f32(
          vSumv0,
          (vcvtq_f32_s32(vmmlaq_s32(
              (vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), vL0, vR0)), vL1, vR1)), vL2, vR2)), vL3, vR3))),
          scale);
    }

    float32x4_t vSumv1 = vextq_f32(vSumv0, vSumv0, 2);
    float32x4_t vSumv2 = vzip1q_f32(vSumv0, vSumv1);

    vst1_f32(pS, vget_low_f32(vSumv2));
    vst1_f32(pS + pBs, vget_high_f32(vSumv2));

    return;
  }
#endif

  int vIb = 0;
  float vSumf = 0;

#if defined(__ARM_FEATURE_SVE)
  svfloat32_t vSumv0 = svdup_n_f32(0.0f);
  svfloat32_t vSumv1 = svdup_n_f32(0.0f);

  const int cVectorLength = ggml_cpu_get_sve_cnt() * 8;

  // VLA Implementation using switch case
  switch (cVectorLength) {
  case 128: {
    // predicate for activating higher lanes for 4 float32 elements
    const svbool_t cPh4 = svptrue_pat_b32(SV_VL4);

    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q4_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q4_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load x
      const svuint8_t cQx0r = svld1rq_u8(svptrue_b8(), vX0->qs);
      const svuint8_t cQx1r = svld1rq_u8(svptrue_b8(), vX1->qs);

      // 4-bit -> 8-bit
      const svint8_t cQx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), cQx0r, 0x0F));
      const svint8_t cQx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), cQx0r, 0x04));
      const svint8_t cQx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), cQx1r, 0x0F));
      const svint8_t cQx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), cQx1r, 0x04));

      // sub 8
      const svint8_t cQx0ls = svsub_n_s8_x(svptrue_b8(), cQx0h, 8);
      const svint8_t cQx0hs = svsub_n_s8_x(svptrue_b8(), cQx0l, 8);
      const svint8_t cQx1ls = svsub_n_s8_x(svptrue_b8(), cQx1h, 8);
      const svint8_t cQx1hs = svsub_n_s8_x(svptrue_b8(), cQx1l, 8);

      // load y
      const svint8_t cQy0h = svld1_s8(svptrue_b8(), vY0->qs);
      const svint8_t cQy0l = svld1_s8(svptrue_b8(), vY0->qs + 16);
      const svint8_t cQy1h = svld1_s8(svptrue_b8(), vY1->qs);
      const svint8_t cQy1l = svld1_s8(svptrue_b8(), vY1->qs + 16);

      // dot product
      vSumv0 = svmla_n_f32_x(cPh4, vSumv0,
                             svcvt_f32_s32_x(cPh4, svadd_x(cPh4, svdot_s32(svdup_n_s32(0), cQx0ls, cQy0l),
                                                           svdot_s32(svdup_n_s32(0), cQx0hs, cQy0h))),
                             GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
      vSumv1 = svmla_n_f32_x(cPh4, vSumv1,
                             svcvt_f32_s32_x(cPh4, svadd_x(cPh4, svdot_s32(svdup_n_s32(0), cQx1ls, cQy1l),
                                                           svdot_s32(svdup_n_s32(0), cQx1hs, cQy1h))),
                             GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
    }

    vSumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), vSumv0, vSumv1));
  } break;
  case 256: {
    // predicate for activating higher lanes for 16 int8 elements
    const svbool_t cPh16 = svptrue_pat_b8(SV_VL16);
    // predicate for activating lower lanes for  16 int8 elements
    const svbool_t cPl16 = svnot_b_z(svptrue_b8(), cPh16);

    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q4_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q4_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load x
      const svuint8_t cQx0r = svld1rq_u8(svptrue_b8(), vX0->qs);
      const svuint8_t cQx1r = svld1rq_u8(svptrue_b8(), vX1->qs);

      // 4-bit -> 8-bit
      const svint8_t cQx0 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_n_u8_m(cPh16, cQx0r, 0x0F), 0x04));
      const svint8_t cQx1 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_n_u8_m(cPh16, cQx1r, 0x0F), 0x04));

      // sub 8
      const svint8_t cQx0s = svsub_n_s8_x(svptrue_b8(), cQx0, 8);
      const svint8_t cQx1s = svsub_n_s8_x(svptrue_b8(), cQx1, 8);

      // load y
      const svint8_t cQy0 = svld1_s8(svptrue_b8(), vY0->qs);
      const svint8_t cQy1 = svld1_s8(svptrue_b8(), vY1->qs);

      // dot product
      vSumv0 =
          svmla_n_f32_x(svptrue_b32(), vSumv0, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), cQx0s, cQy0)),
                        GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
      vSumv1 =
          svmla_n_f32_x(svptrue_b32(), vSumv1, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), cQx1s, cQy1)),
                        GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
    }

    vSumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), vSumv0, vSumv1));
  } break;
  case 512: {
    // predicate for activating higher lanes for 32 int8 elements
    const svbool_t cPh32 = svptrue_pat_b8(SV_VL32);

    // predicate for activating higher lanes for 16 int8 elements
    const svbool_t cPh16 = svptrue_pat_b8(SV_VL16);
    // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
    const svbool_t cPl16 = svnot_b_z(cPh32, cPh16);

    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q4_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q4_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load x
      const svuint8_t cQx0r = svld1rq_u8(cPh32, vX0->qs);
      const svuint8_t cQx1r = svld1rq_u8(cPh32, vX1->qs);

      // 4-bit -> 8-bit
      const svint8_t cQx0 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_n_u8_m(cPh16, cQx0r, 0x0F), 0x04));
      const svint8_t cQx1 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_n_u8_m(cPh16, cQx1r, 0x0F), 0x04));

      // sub 8
      const svint8_t cQx0s = svsub_n_s8_x(cPh32, cQx0, 8);
      const svint8_t cQx1s = svsub_n_s8_x(cPh32, cQx1, 8);

      // load y
      const svint8_t cQy0 = svld1_s8(cPh32, vY0->qs);
      const svint8_t cQy1 = svld1_s8(cPh32, vY1->qs);

      // dot product
      vSumv0 = svmla_n_f32_x(cPh32, vSumv0, svcvt_f32_s32_x(cPh32, svdot_s32(svdup_n_s32(0), cQx0s, cQy0)),
                             GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
      vSumv1 = svmla_n_f32_x(cPh32, vSumv1, svcvt_f32_s32_x(cPh32, svdot_s32(svdup_n_s32(0), cQx1s, cQy1)),
                             GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
    }

    vSumf = svaddv_f32(cPh32, svadd_f32_x(cPh32, vSumv0, vSumv1));
  } break;
  default:
    assert(false && "Unsupported vector length");
    break;
  }

#elif defined(__ARM_NEON)
  float32x4_t vSumv0 = vdupq_n_f32(0.0f);
  float32x4_t vSumv1 = vdupq_n_f32(0.0f);

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q4_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q4_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    const uint8x16_t cM4b = vdupq_n_u8(0x0F);
    const int8x16_t cS8b = vdupq_n_s8(0x8);

    const uint8x16_t cV00 = vld1q_u8(vX0->qs);
    const uint8x16_t cV01 = vld1q_u8(vX1->qs);

    // 4-bit -> 8-bit
    const int8x16_t cV00l = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
    const int8x16_t cV00h = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
    const int8x16_t cV01l = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
    const int8x16_t cV01h = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

    // sub 8
    const int8x16_t v0_0ls = vsubq_s8(cV00l, cS8b);
    const int8x16_t v0_0hs = vsubq_s8(cV00h, cS8b);
    const int8x16_t v0_1ls = vsubq_s8(cV01l, cS8b);
    const int8x16_t v0_1hs = vsubq_s8(cV01h, cS8b);

    // load y
    const int8x16_t v1_0l = vld1q_s8(vY0->qs);
    const int8x16_t v1_0h = vld1q_s8(vY0->qs + 16);
    const int8x16_t v1_1l = vld1q_s8(vY1->qs);
    const int8x16_t v1_1h = vld1q_s8(vY1->qs + 16);

    // dot product into int32x4_t
    const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
    const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

    vSumv0 = vmlaq_n_f32(vSumv0, vcvtq_f32_s32(p_0), GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    vSumv1 = vmlaq_n_f32(vSumv1, vcvtq_f32_s32(p_1), GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
  }

  vSumf = vaddvq_f32(vSumv0) + vaddvq_f32(vSumv1);
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
#if defined(__ARM_FEATURE_MATMUL_INT8)
  assert((pNrc == 2) || (pNrc == 1));
#else
  assert(pNrc == 1);
#endif
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const block_q4_1 *GGML_RESTRICT vVx0 = pVx;
    const block_q4_1 *GGML_RESTRICT vVx1 = (const block_q4_1 *)((const uint8_t *)pVx + pBx);
    const block_q8_1 *GGML_RESTRICT vVy0 = pVy;
    const block_q8_1 *GGML_RESTRICT vVy1 = (const block_q8_1 *)((const uint8_t *)pVy + pBy);

    float32x4_t vSumv0 = vdupq_n_f32(0.0f);
    float32x4_t vSumms0 = vdupq_n_f32(0.0f);

    for (int vI = 0; vI < cNb; vI++) {
      const block_q4_1 *GGML_RESTRICT vBX0 = &vVx0[vI];
      const block_q4_1 *GGML_RESTRICT vBX1 = &vVx1[vI];
      const block_q8_1 *GGML_RESTRICT vBY0 = &vVy0[vI];
      const block_q8_1 *GGML_RESTRICT vBY1 = &vVy1[vI];

      float32_t vSummsT[4] = {GGML_CPU_FP16_TO_FP32(vBX0->m) * GGML_CPU_FP16_TO_FP32(vBY0->s),
                              GGML_CPU_FP16_TO_FP32(vBX1->m) * GGML_CPU_FP16_TO_FP32(vBY0->s),
                              GGML_CPU_FP16_TO_FP32(vBX0->m) * GGML_CPU_FP16_TO_FP32(vBY1->s),
                              GGML_CPU_FP16_TO_FP32(vBX1->m) * GGML_CPU_FP16_TO_FP32(vBY1->s)};
      vSumms0 = vaddq_f32(vSumms0, vld1q_f32(vSummsT));

      const uint8x16_t cM4b = vdupq_n_u8(0x0F);

      const uint8x16_t cV00 = vld1q_u8(vBX0->qs);
      const uint8x16_t cV01 = vld1q_u8(vBX1->qs);

      // 4-bit -> 8-bit
      const int8x16_t cX0L = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
      const int8x16_t cX0H = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
      const int8x16_t cX1L = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
      const int8x16_t cX1H = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

      // load y
      const int8x16_t cY0L = vld1q_s8(vBY0->qs);
      const int8x16_t cY0H = vld1q_s8(vBY0->qs + 16);
      const int8x16_t cY1L = vld1q_s8(vBY1->qs);
      const int8x16_t cY1H = vld1q_s8(vBY1->qs + 16);

      // mmla into int32x4_t
      float32_t _scale[4] = {GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY1->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY1->d)};
      float32x4_t scale = vld1q_f32(_scale);

      int8x16_t vL0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));
      int8x16_t vL1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));

      int8x16_t vL2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));
      int8x16_t vL3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));

      int8x16_t vR0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));
      int8x16_t vR1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));

      int8x16_t vR2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));
      int8x16_t vR3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));
      vSumv0 = vmlaq_f32(
          vSumv0,
          (vcvtq_f32_s32(vmmlaq_s32(
              (vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), vL0, vR0)), vL1, vR1)), vL2, vR2)), vL3, vR3))),
          scale);
    }

    float32x4_t vSumv1 = vextq_f32(vSumv0, vSumv0, 2);
    float32x4_t vSumv2 = vzip1q_f32(vSumv0, vSumv1);

    vSumv2 = vaddq_f32(vSumv2, vSumms0);

    vst1_f32(pS, vget_low_f32(vSumv2));
    vst1_f32(pS + pBs, vget_high_f32(vSumv2));

    return;
  }
#endif

  int vIb = 0;
  float vSumf = 0;

#if defined(__ARM_NEON)
  float32x4_t vSumv0 = vdupq_n_f32(0.0f);
  float32x4_t vSumv1 = vdupq_n_f32(0.0f);

  float vSumms = 0;

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q4_1 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q4_1 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_1 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    vSumms += GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s) +
              GGML_CPU_FP16_TO_FP32(vX1->m) * GGML_CPU_FP16_TO_FP32(vY1->s);

    const uint8x16_t cM4b = vdupq_n_u8(0x0F);

    const uint8x16_t cV00 = vld1q_u8(vX0->qs);
    const uint8x16_t cV01 = vld1q_u8(vX1->qs);

    // 4-bit -> 8-bit
    const int8x16_t cV00l = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
    const int8x16_t cV00h = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
    const int8x16_t cV01l = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
    const int8x16_t cV01h = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

    // load y
    const int8x16_t cV10l = vld1q_s8(vY0->qs);
    const int8x16_t cV10h = vld1q_s8(vY0->qs + 16);
    const int8x16_t cV11l = vld1q_s8(vY1->qs);
    const int8x16_t cV11h = vld1q_s8(vY1->qs + 16);

    // dot product into int32x4_t
    const int32x4_t cP0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV00l, cV10l), cV00h, cV10h);
    const int32x4_t cP1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV01l, cV11l), cV01h, cV11h);

    vSumv0 = vmlaq_n_f32(vSumv0, vcvtq_f32_s32(cP0), GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    vSumv1 = vmlaq_n_f32(vSumv1, vcvtq_f32_s32(cP1), GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
  }

  vSumf = vaddvq_f32(vSumv0) + vaddvq_f32(vSumv1) + vSumms;

#endif
  for (; vIb < cNb; ++vIb) {
    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cV0 = (vX[vIb].qs[vJ] & 0x0F);
      const int cV1 = (vX[vIb].qs[vJ] >> 4);

      vSumi0 += (cV0 * vY[vIb].qs[vJ]);
      vSumi1 += (cV1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
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

#if defined __ARM_NEON
  const int8x16_t cValues = vld1q_s8(kvalues_mxfp4);
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  uint8x16x2_t vQ4bits;
  int8x16x4_t vQ4b;
  int8x16x4_t vQ8b;
  int32x4_t vProd1;
  int32x4_t vProd2;

  for (; vIb + 1 < cNb; vIb += 2) {
    vQ4bits.val[0] = vld1q_u8(vX[vIb + 0].qs);
    vQ4bits.val[1] = vld1q_u8(vX[vIb + 1].qs);
    vQ8b.val[0] = vld1q_s8(vY[vIb + 0].qs);
    vQ8b.val[1] = vld1q_s8(vY[vIb + 0].qs + 16);
    vQ8b.val[2] = vld1q_s8(vY[vIb + 1].qs);
    vQ8b.val[3] = vld1q_s8(vY[vIb + 1].qs + 16);

    vQ4b.val[0] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[0], cM4b));
    vQ4b.val[1] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[0], 4));
    vQ4b.val[2] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[1], cM4b));
    vQ4b.val[3] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[1], 4));

    vProd1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[0], vQ8b.val[0]), vQ4b.val[1], vQ8b.val[1]);
    vProd2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[2], vQ8b.val[2]), vQ4b.val[3], vQ8b.val[3]);

    vSumf += GGML_E8M0_TO_FP32_HALF(vX[vIb + 0].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vaddvq_s32(vProd1) +
             GGML_E8M0_TO_FP32_HALF(vX[vIb + 1].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vaddvq_s32(vProd2);
  }

#endif
  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_E8M0_TO_FP32_HALF(vX[vIb].e);
    int vSumi1 = 0;
    int vSumi2 = 0;
    for (int vJ = 0; vJ < QK_MXFP4 / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_mxfp4[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK_MXFP4 / 2] * kvalues_mxfp4[vX[vIb].qs[vJ] >> 4];
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

  // Each NVFP4 super-block (64 elements) spans 2 q8_0 blocks
  const int cNb = pN / QK_NVFP4;

  float vSumf = 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
  const int8x16_t cValues = vld1q_s8(kvalues_mxfp4);
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  float32x4_t vAcc = vdupq_n_f32(0.0f);

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const uint8x16_t cQ4bits0 = vld1q_u8(vX[vIb].qs);
    const uint8x16_t cQ4bits1 = vld1q_u8(vX[vIb].qs + 16);

    const int8x16_t cQ4Lo0 = ggml_vqtbl1q_s8(cValues, vandq_u8(cQ4bits0, cM4b));
    const int8x16_t cQ4Hi0 = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(cQ4bits0, 4));
    const int8x16_t cQ4Lo1 = ggml_vqtbl1q_s8(cValues, vandq_u8(cQ4bits1, cM4b));
    const int8x16_t cQ4Hi1 = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(cQ4bits1, 4));

#if defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t cQ80a = vld1q_s8(vY[2 * vIb].qs);
    const int8x16_t cQ80b = vld1q_s8(vY[2 * vIb].qs + 16);
    const int8x16_t cQ8Lo0 = vcombine_s8(vget_low_s8(cQ80a), vget_low_s8(cQ80b));
    const int8x16_t cQ8Hi0 = vcombine_s8(vget_high_s8(cQ80a), vget_high_s8(cQ80b));

    const int8x16_t cQ81a = vld1q_s8(vY[2 * vIb + 1].qs);
    const int8x16_t cQ81b = vld1q_s8(vY[2 * vIb + 1].qs + 16);
    const int8x16_t cQ8Lo1 = vcombine_s8(vget_low_s8(cQ81a), vget_low_s8(cQ81b));
    const int8x16_t cQ8Hi1 = vcombine_s8(vget_high_s8(cQ81a), vget_high_s8(cQ81b));

    const int32x4_t cP0 =
        vaddq_s32(vdotq_s32(vdupq_n_s32(0), cQ4Lo0, cQ8Lo0), vdotq_s32(vdupq_n_s32(0), cQ4Hi0, cQ8Hi0));
    const int32x4_t cP1 =
        vaddq_s32(vdotq_s32(vdupq_n_s32(0), cQ4Lo1, cQ8Lo1), vdotq_s32(vdupq_n_s32(0), cQ4Hi1, cQ8Hi1));

    const int32x4_t cSumi = vpaddq_s32(cP0, cP1);
#else
    const int8x8_t q4_0_lo = vget_low_s8(cQ4Lo0);
    const int8x8_t q4_0_hi = vget_low_s8(cQ4Hi0);
    const int8x8_t q4_1_lo = vget_high_s8(cQ4Lo0);
    const int8x8_t q4_1_hi = vget_high_s8(cQ4Hi0);
    const int8x8_t q4_2_lo = vget_low_s8(cQ4Lo1);
    const int8x8_t q4_2_hi = vget_low_s8(cQ4Hi1);
    const int8x8_t q4_3_lo = vget_high_s8(cQ4Lo1);
    const int8x8_t q4_3_hi = vget_high_s8(cQ4Hi1);

    const int8x8_t q8_0_lo = vld1_s8(vY[2 * vIb].qs);
    const int8x8_t q8_0_hi = vld1_s8(vY[2 * vIb].qs + 8);
    const int8x8_t q8_1_lo = vld1_s8(vY[2 * vIb].qs + 16);
    const int8x8_t q8_1_hi = vld1_s8(vY[2 * vIb].qs + 24);
    const int8x8_t q8_2_lo = vld1_s8(vY[2 * vIb + 1].qs);
    const int8x8_t q8_2_hi = vld1_s8(vY[2 * vIb + 1].qs + 8);
    const int8x8_t q8_3_lo = vld1_s8(vY[2 * vIb + 1].qs + 16);
    const int8x8_t q8_3_hi = vld1_s8(vY[2 * vIb + 1].qs + 24);

    const int32x4_t cSumi = (int32x4_t){
        vaddvq_s32(ggml_nvfp4_dot8(q4_0_lo, q8_0_lo, q4_0_hi, q8_0_hi)),
        vaddvq_s32(ggml_nvfp4_dot8(q4_1_lo, q8_1_lo, q4_1_hi, q8_1_hi)),
        vaddvq_s32(ggml_nvfp4_dot8(q4_2_lo, q8_2_lo, q4_2_hi, q8_2_hi)),
        vaddvq_s32(ggml_nvfp4_dot8(q4_3_lo, q8_3_lo, q4_3_hi, q8_3_hi)),
    };
#endif

    const float cDy0 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb].d);
    const float cDy1 = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + 1].d);
    const float32x4_t cNvsc = {GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[0]), GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[1]),
                               GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[2]), GGML_CPU_UE4M3_TO_FP32(vX[vIb].d[3])};
    const float32x4_t cScales = vmulq_f32(cNvsc, (float32x4_t){cDy0, cDy0, cDy1, cDy1});

    vAcc = vfmaq_f32(vAcc, vcvtq_f32_s32(cSumi), cScales);
  }
  vSumf = vaddvq_f32(vAcc);
#else
  for (int vIb = 0; vIb < cNb; ++vIb) {
    for (int si = 0; si < 4; ++si) {
      const float d = ggml_ue4m3_to_fp32(vX[vIb].d[si]);
      const int q8b = si / 2;
      const int q8o = (si % 2) * QK_NVFP4_SUB;
      const float dy = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + q8b].d);

      int sumi_lo = 0, sumi_hi = 0;
      for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
        const uint8_t qv = vX[vIb].qs[si * (QK_NVFP4_SUB / 2) + j];
        sumi_lo += vY[2 * vIb + q8b].qs[q8o + j + 0] * kvalues_mxfp4[qv & 0xf];
        sumi_hi += vY[2 * vIb + q8b].qs[q8o + j + QK_NVFP4_SUB / 2] * kvalues_mxfp4[qv >> 4];
      }
      vSumf += dy * d * (sumi_lo + sumi_hi);
    }
  }
#endif
  *pS = vSumf;
}

void ggml_vec_dot_q5_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  int vIb = 0;
  float vSumf = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_NEON)
  float32x4_t vSumv0 = vdupq_n_f32(0.0f);
  float32x4_t vSumv1 = vdupq_n_f32(0.0f);

  uint32_t vQh0;
  uint32_t vQh1;

  uint64_t vTmp0[4];
  uint64_t vTmp1[4];

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q5_0 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q5_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    const uint8x16_t cM4b = vdupq_n_u8(0x0F);

    // extract the 5th bit via lookup table ((!b) << 4)
    memcpy(&vQh0, vX0->qh, sizeof(vQh0));
    memcpy(&vQh1, vX1->qh, sizeof(vQh1));

    vTmp0[0] = table_b2b_1[(vQh0 >> 0) & 0xFF];
    vTmp0[1] = table_b2b_1[(vQh0 >> 8) & 0xFF];
    vTmp0[2] = table_b2b_1[(vQh0 >> 16) & 0xFF];
    vTmp0[3] = table_b2b_1[(vQh0 >> 24)];

    vTmp1[0] = table_b2b_1[(vQh1 >> 0) & 0xFF];
    vTmp1[1] = table_b2b_1[(vQh1 >> 8) & 0xFF];
    vTmp1[2] = table_b2b_1[(vQh1 >> 16) & 0xFF];
    vTmp1[3] = table_b2b_1[(vQh1 >> 24)];

    const int8x16_t cQhl0 = vld1q_s8((const int8_t *)(vTmp0 + 0));
    const int8x16_t cQhh0 = vld1q_s8((const int8_t *)(vTmp0 + 2));
    const int8x16_t cQhl1 = vld1q_s8((const int8_t *)(vTmp1 + 0));
    const int8x16_t cQhh1 = vld1q_s8((const int8_t *)(vTmp1 + 2));

    const uint8x16_t cV00 = vld1q_u8(vX0->qs);
    const uint8x16_t cV01 = vld1q_u8(vX1->qs);

    // 4-bit -> 8-bit
    int8x16_t vV00l = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
    int8x16_t vV00h = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
    int8x16_t vV01l = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
    int8x16_t vV01h = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

    // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
    const int8x16_t cV00lf = vsubq_s8(vV00l, cQhl0);
    const int8x16_t cV00hf = vsubq_s8(vV00h, cQhh0);
    const int8x16_t cV01lf = vsubq_s8(vV01l, cQhl1);
    const int8x16_t cV01hf = vsubq_s8(vV01h, cQhh1);

    // load y
    const int8x16_t cV10l = vld1q_s8(vY0->qs);
    const int8x16_t cV10h = vld1q_s8(vY0->qs + 16);
    const int8x16_t cV11l = vld1q_s8(vY1->qs);
    const int8x16_t cV11h = vld1q_s8(vY1->qs + 16);

    vSumv0 = vmlaq_n_f32(vSumv0,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV00lf, cV10l),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), cV00hf, cV10h))),
                         GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    vSumv1 = vmlaq_n_f32(vSumv1,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV01lf, cV11l),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), cV01hf, cV11h))),
                         GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
  }

  vSumf = vaddvq_f32(vSumv0) + vaddvq_f32(vSumv1);

#endif
  for (; vIb < cNb; ++vIb) {
    uint32_t vQh;
    memcpy(&vQh, vX[vIb].qh, sizeof(vQh));

    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh & (1u << (vJ + 0))) >> (vJ + 0)) << 4;
      const uint8_t cXh1 = ((vQh & (1u << (vJ + 16))) >> (vJ + 12));

      const int32_t cX0 = (int8_t)(((vX[vIb].qs[vJ] & 0x0F) | cXh0) - 16);
      const int32_t cX1 = (int8_t)(((vX[vIb].qs[vJ] >> 4) | cXh1) - 16);

      vSumi0 += (cX0 * vY[vIb].qs[vJ]);
      vSumi1 += (cX1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi;
  }

  *pS = vSumf;
}

void ggml_vec_dot_q5_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  int vIb = 0;
  float vSumf = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_1);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_NEON)
  float32x4_t vSumv0 = vdupq_n_f32(0.0f);
  float32x4_t vSumv1 = vdupq_n_f32(0.0f);

  float vSumms0 = 0.0f;
  float vSumms1 = 0.0f;

  uint32_t vQh0;
  uint32_t vQh1;

  uint64_t vTmp0[4];
  uint64_t vTmp1[4];

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q5_1 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q5_1 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb];
    const block_q8_1 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    const uint8x16_t cM4b = vdupq_n_u8(0x0F);

    vSumms0 += GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s);
    vSumms1 += GGML_CPU_FP16_TO_FP32(vX1->m) * GGML_CPU_FP16_TO_FP32(vY1->s);

    // extract the 5th bit via lookup table ((b) << 4)
    memcpy(&vQh0, vX0->qh, sizeof(vQh0));
    memcpy(&vQh1, vX1->qh, sizeof(vQh1));

    vTmp0[0] = table_b2b_0[(vQh0 >> 0) & 0xFF];
    vTmp0[1] = table_b2b_0[(vQh0 >> 8) & 0xFF];
    vTmp0[2] = table_b2b_0[(vQh0 >> 16) & 0xFF];
    vTmp0[3] = table_b2b_0[(vQh0 >> 24)];

    vTmp1[0] = table_b2b_0[(vQh1 >> 0) & 0xFF];
    vTmp1[1] = table_b2b_0[(vQh1 >> 8) & 0xFF];
    vTmp1[2] = table_b2b_0[(vQh1 >> 16) & 0xFF];
    vTmp1[3] = table_b2b_0[(vQh1 >> 24)];

    const int8x16_t cQhl0 = vld1q_s8((const int8_t *)(vTmp0 + 0));
    const int8x16_t cQhh0 = vld1q_s8((const int8_t *)(vTmp0 + 2));
    const int8x16_t cQhl1 = vld1q_s8((const int8_t *)(vTmp1 + 0));
    const int8x16_t cQhh1 = vld1q_s8((const int8_t *)(vTmp1 + 2));

    const uint8x16_t cV00 = vld1q_u8(vX0->qs);
    const uint8x16_t cV01 = vld1q_u8(vX1->qs);

    // 4-bit -> 8-bit
    const int8x16_t cV00l = vreinterpretq_s8_u8(vandq_u8(cV00, cM4b));
    const int8x16_t cV00h = vreinterpretq_s8_u8(vshrq_n_u8(cV00, 4));
    const int8x16_t cV01l = vreinterpretq_s8_u8(vandq_u8(cV01, cM4b));
    const int8x16_t cV01h = vreinterpretq_s8_u8(vshrq_n_u8(cV01, 4));

    // add high bit
    const int8x16_t cV00lf = vorrq_s8(cV00l, cQhl0);
    const int8x16_t cV00hf = vorrq_s8(cV00h, cQhh0);
    const int8x16_t cV01lf = vorrq_s8(cV01l, cQhl1);
    const int8x16_t cV01hf = vorrq_s8(cV01h, cQhh1);

    // load y
    const int8x16_t cV10l = vld1q_s8(vY0->qs);
    const int8x16_t cV10h = vld1q_s8(vY0->qs + 16);
    const int8x16_t cV11l = vld1q_s8(vY1->qs);
    const int8x16_t cV11h = vld1q_s8(vY1->qs + 16);

    vSumv0 = vmlaq_n_f32(vSumv0,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV00lf, cV10l),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), cV00hf, cV10h))),
                         GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    vSumv1 = vmlaq_n_f32(vSumv1,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), cV01lf, cV11l),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), cV01hf, cV11h))),
                         GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
  }

  vSumf = vaddvq_f32(vSumv0) + vaddvq_f32(vSumv1) + vSumms0 + vSumms1;

#endif
  for (; vIb < cNb; ++vIb) {
    uint32_t vQh;
    memcpy(&vQh, vX[vIb].qh, sizeof(vQh));

    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh >> (vJ + 0)) << 4) & 0x10;
      const uint8_t cXh1 = ((vQh >> (vJ + 12))) & 0x10;

      const int32_t cX0 = (vX[vIb].qs[vJ] & 0xF) | cXh0;
      const int32_t cX1 = (vX[vIb].qs[vJ] >> 4) | cXh1;

      vSumi0 += (cX0 * vY[vIb].qs[vJ]);
      vSumi1 += (cX1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
}

void ggml_vec_dot_q8_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  assert((pNrc == 2) || (pNrc == 1));
#else
  assert(pNrc == 1);
#endif
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q8_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const block_q8_0 *GGML_RESTRICT vVx0 = pVx;
    const block_q8_0 *GGML_RESTRICT vVx1 = (const block_q8_0 *)((const uint8_t *)pVx + pBx);
    const block_q8_0 *GGML_RESTRICT vVy0 = pVy;
    const block_q8_0 *GGML_RESTRICT vVy1 = (const block_q8_0 *)((const uint8_t *)pVy + pBy);

    float32x4_t vSumv0 = vdupq_n_f32(0.0f);

    for (int vI = 0; vI < cNb; vI++) {
      const block_q8_0 *GGML_RESTRICT vBX0 = &vVx0[vI];
      const block_q8_0 *GGML_RESTRICT vBY0 = &vVy0[vI];

      const block_q8_0 *GGML_RESTRICT vBX1 = &vVx1[vI];
      const block_q8_0 *GGML_RESTRICT vBY1 = &vVy1[vI];

      const int8x16_t cX0L = vld1q_s8(vBX0->qs);
      const int8x16_t cX0H = vld1q_s8(vBX0->qs + 16);
      const int8x16_t cX1L = vld1q_s8(vBX1->qs);
      const int8x16_t cX1H = vld1q_s8(vBX1->qs + 16);

      // load y
      const int8x16_t cY0L = vld1q_s8(vBY0->qs);
      const int8x16_t cY0H = vld1q_s8(vBY0->qs + 16);
      const int8x16_t cY1L = vld1q_s8(vBY1->qs);
      const int8x16_t cY1H = vld1q_s8(vBY1->qs + 16);

      float32_t _scale[4] = {GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX0->d) * GGML_CPU_FP16_TO_FP32(vBY1->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY0->d),
                             GGML_CPU_FP16_TO_FP32(vBX1->d) * GGML_CPU_FP16_TO_FP32(vBY1->d)};
      float32x4_t scale = vld1q_f32(_scale);

      int8x16_t vL0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));
      int8x16_t vL1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0L), vreinterpretq_s64_s8(cX1L)));

      int8x16_t vL2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));
      int8x16_t vL3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cX0H), vreinterpretq_s64_s8(cX1H)));

      int8x16_t vR0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));
      int8x16_t vR1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0L), vreinterpretq_s64_s8(cY1L)));

      int8x16_t vR2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));
      int8x16_t vR3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(cY0H), vreinterpretq_s64_s8(cY1H)));

      vSumv0 = vmlaq_f32(
          vSumv0,
          (vcvtq_f32_s32(vmmlaq_s32(
              (vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), vL0, vR0)), vL1, vR1)), vL2, vR2)), vL3, vR3))),
          scale);
    }

    float32x4_t vSumv1 = vextq_f32(vSumv0, vSumv0, 2);
    float32x4_t vSumv2 = vzip1q_f32(vSumv0, vSumv1);

    vst1_f32(pS, vget_low_f32(vSumv2));
    vst1_f32(pS + pBs, vget_high_f32(vSumv2));

    return;
  }
#endif

  int vIb = 0;
  float vSumf = 0;

#if defined(__ARM_FEATURE_SVE)
  svfloat32_t vSumv0 = svdup_n_f32(0.0f);
  svfloat32_t vSumv1 = svdup_n_f32(0.0f);

  const int cVectorLength = ggml_cpu_get_sve_cnt() * 8;

  // VLA Implementation for SVE
  switch (cVectorLength) {
  case 128: {
    // predicate for activating lanes for 16 Int8 elements
    const svbool_t cPh16 = svptrue_pat_b8(SV_VL16);
    const svbool_t cPl16 = svptrue_pat_b32(SV_VL4);

    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q8_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load x
      const svint8_t cQx00 = svld1_s8(cPh16, vX0->qs);
      const svint8_t cQx01 = svld1_s8(cPh16, vX0->qs + 16);
      const svint8_t cQx10 = svld1_s8(cPh16, vX1->qs);
      const svint8_t cQx11 = svld1_s8(cPh16, vX1->qs + 16);

      // load y
      const svint8_t cQy00 = svld1_s8(cPh16, vY0->qs);
      const svint8_t cQy01 = svld1_s8(cPh16, vY0->qs + 16);
      const svint8_t cQy10 = svld1_s8(cPh16, vY1->qs);
      const svint8_t cQy11 = svld1_s8(cPh16, vY1->qs + 16);

      vSumv0 = svmla_n_f32_x(cPl16, vSumv0,
                             svcvt_f32_s32_x(cPl16, svadd_x(cPl16, svdot_s32(svdup_n_s32(0), cQx00, cQy00),
                                                            svdot_s32(svdup_n_s32(0), cQx01, cQy01))),
                             GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
      vSumv1 = svmla_n_f32_x(cPl16, vSumv1,
                             svcvt_f32_s32_x(cPl16, svadd_x(cPl16, svdot_s32(svdup_n_s32(0), cQx10, cQy10),
                                                            svdot_s32(svdup_n_s32(0), cQx11, cQy11))),
                             GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
    }

    vSumf = svaddv_f32(cPl16, svadd_f32_x(cPl16, vSumv0, vSumv1));
  } break;
  case 256: {
    // printf("sve256");
    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q8_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load x
      const svint8_t cQx0 = svld1_s8(svptrue_b8(), vX0->qs);
      const svint8_t cQx1 = svld1_s8(svptrue_b8(), vX1->qs);

      // load y
      const svint8_t cQy0 = svld1_s8(svptrue_b8(), vY0->qs);
      const svint8_t cQy1 = svld1_s8(svptrue_b8(), vY1->qs);

      vSumv0 =
          svmla_n_f32_x(svptrue_b32(), vSumv0, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), cQx0, cQy0)),
                        GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
      vSumv1 =
          svmla_n_f32_x(svptrue_b32(), vSumv1, svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), cQx1, cQy1)),
                        GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
    }

    vSumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), vSumv0, vSumv1));
  } break;
  case 512: {
    // predicate for activating high 256 bit
    const svbool_t cPh32 = svptrue_pat_b8(SV_VL32);
    // predicate for activating low 256 bit
    const svbool_t cPl32 = svnot_b_z(svptrue_b8(), cPh32);

    // predicate for activating high lanes for 8 float32 elements
    const svbool_t cPh8 = svptrue_pat_b32(SV_VL8);
    // predicate for activating low lanes for 8 float32 elements
    const svbool_t cPl8 = svnot_b_z(svptrue_b32(), cPh8);

    svfloat32_t vSumv00 = svdup_n_f32(0.0f);

    for (; vIb + 1 < cNb; vIb += 2) {
      const block_q8_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
      const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
      const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

      // load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
      //  and add them to make one 64 element vector
      //  load x
      const svint8_t cQx32 = svld1_s8(cPh32, vX0->qs);
      svint8_t vQx64 = svld1_s8(cPl32, vX0->qs + 2);

      vQx64 = svadd_s8_x(svptrue_b8(), cQx32, vQx64);

      // load y
      const svint8_t cQy32 = svld1_s8(cPh32, vY0->qs);
      svint8_t vQy64 = svld1_s8(cPl32, vY0->qs + 2);

      vQy64 = svadd_s8_x(svptrue_b8(), cQy32, vQy64);

      // scale creation
      const float32_t cDeq1 = GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d);
      const float32_t cDeq2 = GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d);

      // duplicate deq1 in first half of vector and deq2 in second half of vector
      const svfloat32_t cTemp = svdup_f32_m(svdup_f32_z(cPh8, cDeq1), cPl8, cDeq2);

      const svfloat32_t cSumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), vQx64, vQy64));

      vSumv00 = svmla_f32_m(svptrue_b32(), vSumv00, cSumvt, cTemp);
    }

    vSumf = svaddv_f32(svptrue_b32(), vSumv00);
    break;
  }
  default:
    assert(false && "Unsupported vector length");
    break;
  }
#elif defined(__ARM_NEON)
  float32x4_t vSumv0 = vdupq_n_f32(0.0f);
  float32x4_t vSumv1 = vdupq_n_f32(0.0f);

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q8_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    const int8x16_t x0_0 = vld1q_s8(vX0->qs);
    const int8x16_t x0_1 = vld1q_s8(vX0->qs + 16);
    const int8x16_t x1_0 = vld1q_s8(vX1->qs);
    const int8x16_t x1_1 = vld1q_s8(vX1->qs + 16);

    // load y
    const int8x16_t y0_0 = vld1q_s8(vY0->qs);
    const int8x16_t y0_1 = vld1q_s8(vY0->qs + 16);
    const int8x16_t y1_0 = vld1q_s8(vY1->qs);
    const int8x16_t y1_1 = vld1q_s8(vY1->qs + 16);

    vSumv0 = vmlaq_n_f32(vSumv0,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
                         GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));

    vSumv1 = vmlaq_n_f32(vSumv1,
                         vcvtq_f32_s32(vaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                                                 ggml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
                         GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));
  }

  vSumf = vaddvq_f32(vSumv0) + vaddvq_f32(vSumv1);
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

#if defined(__ARM_NEON)
  float vSumf = 0.0f;

  uint8_t vKShift[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  const uint8x16_t cShift = vld1q_u8(vKShift);

  for (int vI = 0; vI < cNb; ++vI) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t vSumi0 = vdupq_n_s32(0);
    int32x4_t vSumi1 = vdupq_n_s32(0);
#else
    int16x8_t vSumi0 = vdupq_n_s16(0);
    int16x8_t vSumi1 = vdupq_n_s16(0);
#endif

    // first 32 bytes of 5 elements
    {
      uint8x16_t vQx0 = vld1q_u8(vX[vI].qs + 0);
      uint8x16_t vQx1 = vld1q_u8(vX[vI].qs + 16);
      uint8x16_t vQx2 = vmulq_u8(vQx0, vdupq_n_u8(3));
      uint8x16_t vQx3 = vmulq_u8(vQx1, vdupq_n_u8(3));
      uint8x16_t vQx4 = vmulq_u8(vQx0, vdupq_n_u8(9));
      uint8x16_t vQx5 = vmulq_u8(vQx1, vdupq_n_u8(9));
      uint8x16_t vQx6 = vmulq_u8(vQx0, vdupq_n_u8(27));
      uint8x16_t vQx7 = vmulq_u8(vQx1, vdupq_n_u8(27));
      uint8x16_t vQx8 = vmulq_u8(vQx0, vdupq_n_u8(81));
      uint8x16_t vQx9 = vmulq_u8(vQx1, vdupq_n_u8(81));

      // multiply by 3 and keep the 2 bits above 8 bits
      int8x16_t vSqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx0, vshrq_n_u8(vQx0, 1)), 6));
      int8x16_t vSqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx1, vshrq_n_u8(vQx1, 1)), 6));
      int8x16_t vSqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx2, vshrq_n_u8(vQx2, 1)), 6));
      int8x16_t vSqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx3, vshrq_n_u8(vQx3, 1)), 6));
      int8x16_t vSqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx4, vshrq_n_u8(vQx4, 1)), 6));
      int8x16_t vSqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx5, vshrq_n_u8(vQx5, 1)), 6));
      int8x16_t vSqx6 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx6, vshrq_n_u8(vQx6, 1)), 6));
      int8x16_t vSqx7 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx7, vshrq_n_u8(vQx7, 1)), 6));
      int8x16_t vSqx8 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx8, vshrq_n_u8(vQx8, 1)), 6));
      int8x16_t vSqx9 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx9, vshrq_n_u8(vQx9, 1)), 6));

      const int8x16_t cQy0 = vld1q_s8(vY[vI].qs + 0);
      const int8x16_t cQy1 = vld1q_s8(vY[vI].qs + 16);
      const int8x16_t cQy2 = vld1q_s8(vY[vI].qs + 32);
      const int8x16_t cQy3 = vld1q_s8(vY[vI].qs + 48);
      const int8x16_t cQy4 = vld1q_s8(vY[vI].qs + 64);
      const int8x16_t cQy5 = vld1q_s8(vY[vI].qs + 80);
      const int8x16_t cQy6 = vld1q_s8(vY[vI].qs + 96);
      const int8x16_t cQy7 = vld1q_s8(vY[vI].qs + 112);
      const int8x16_t cQy8 = vld1q_s8(vY[vI].qs + 128);
      const int8x16_t cQy9 = vld1q_s8(vY[vI].qs + 144);

#if defined(__ARM_FEATURE_DOTPROD)
      vSumi0 = vdotq_s32(vSumi0, vSqx0, cQy0);
      vSumi1 = vdotq_s32(vSumi1, vSqx1, cQy1);
      vSumi0 = vdotq_s32(vSumi0, vSqx2, cQy2);
      vSumi1 = vdotq_s32(vSumi1, vSqx3, cQy3);
      vSumi0 = vdotq_s32(vSumi0, vSqx4, cQy4);
      vSumi1 = vdotq_s32(vSumi1, vSqx5, cQy5);
      vSumi0 = vdotq_s32(vSumi0, vSqx6, cQy6);
      vSumi1 = vdotq_s32(vSumi1, vSqx7, cQy7);
      vSumi0 = vdotq_s32(vSumi0, vSqx8, cQy8);
      vSumi1 = vdotq_s32(vSumi1, vSqx9, cQy9);
#else
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx0), vget_low_s8(cQy0));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx0), vget_high_s8(cQy0));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx1), vget_low_s8(cQy1));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx1), vget_high_s8(cQy1));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx2), vget_low_s8(cQy2));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx2), vget_high_s8(cQy2));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx3), vget_low_s8(cQy3));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx3), vget_high_s8(cQy3));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx4), vget_low_s8(cQy4));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx4), vget_high_s8(cQy4));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx5), vget_low_s8(cQy5));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx5), vget_high_s8(cQy5));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx6), vget_low_s8(cQy6));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx6), vget_high_s8(cQy6));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx7), vget_low_s8(cQy7));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx7), vget_high_s8(cQy7));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx8), vget_low_s8(cQy8));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx8), vget_high_s8(cQy8));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx9), vget_low_s8(cQy9));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx9), vget_high_s8(cQy9));
#endif
    }

    // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
    {
      uint8x16_t vQx0 = vld1q_u8(vX[vI].qs + 32);
      uint8x16_t vQx1 = vmulq_u8(vQx0, vdupq_n_u8(3));
      uint8x16_t vQx2 = vmulq_u8(vQx0, vdupq_n_u8(9));
      uint8x16_t vQx3 = vmulq_u8(vQx0, vdupq_n_u8(27));
      uint8x16_t vQx4 = vmulq_u8(vQx0, vdupq_n_u8(81));
      uint32_t vQh;
      memcpy(&vQh, vX[vI].qh, sizeof(vQh)); // potentially unaligned
      uint8x16_t vQx5 = vreinterpretq_u8_u32(vdupq_n_u32(vQh));
      vQx5 = vmulq_u8(vQx5, cShift);

      // multiply by 3 and keep the 2 bits above 8 bits
      int8x16_t vSqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx0, vshrq_n_u8(vQx0, 1)), 6));
      int8x16_t vSqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx1, vshrq_n_u8(vQx1, 1)), 6));
      int8x16_t vSqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx2, vshrq_n_u8(vQx2, 1)), 6));
      int8x16_t vSqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx3, vshrq_n_u8(vQx3, 1)), 6));
      int8x16_t vSqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx4, vshrq_n_u8(vQx4, 1)), 6));
      int8x16_t vSqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(vQx5, vshrq_n_u8(vQx5, 1)), 6));

      const int8x16_t cQy0 = vld1q_s8(vY[vI].qs + 160);
      const int8x16_t cQy1 = vld1q_s8(vY[vI].qs + 176);
      const int8x16_t cQy2 = vld1q_s8(vY[vI].qs + 192);
      const int8x16_t cQy3 = vld1q_s8(vY[vI].qs + 208);
      const int8x16_t cQy4 = vld1q_s8(vY[vI].qs + 224);
      const int8x16_t cQy5 = vld1q_s8(vY[vI].qs + 240);

#if defined(__ARM_FEATURE_DOTPROD)
      vSumi0 = vdotq_s32(vSumi0, vSqx0, cQy0);
      vSumi1 = vdotq_s32(vSumi1, vSqx1, cQy1);
      vSumi0 = vdotq_s32(vSumi0, vSqx2, cQy2);
      vSumi1 = vdotq_s32(vSumi1, vSqx3, cQy3);
      vSumi0 = vdotq_s32(vSumi0, vSqx4, cQy4);
      vSumi1 = vdotq_s32(vSumi1, vSqx5, cQy5);
#else
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx0), vget_low_s8(cQy0));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx0), vget_high_s8(cQy0));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx1), vget_low_s8(cQy1));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx1), vget_high_s8(cQy1));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx2), vget_low_s8(cQy2));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx2), vget_high_s8(cQy2));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx3), vget_low_s8(cQy3));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx3), vget_high_s8(cQy3));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx4), vget_low_s8(cQy4));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx4), vget_high_s8(cQy4));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx5), vget_low_s8(cQy5));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx5), vget_high_s8(cQy5));
#endif
    }

    const int16x8_t cYsum0 = vld1q_s16(vY[vI].bsums);
    const int16x8_t cYsum1 = vld1q_s16(vY[vI].bsums + 8);

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

#if defined(__ARM_FEATURE_DOTPROD)
    vSumi0 = vaddq_s32(vSumi0, vSumi1);
    vSumi0 = vsubq_s32(vSumi0, vpaddlq_s16(vaddq_s16(cYsum0, cYsum1)));

    vSumf += cD * (float)vaddvq_s32(vSumi0);
#else
    vSumi0 = vaddq_s16(vSumi0, vSumi1);
    vSumi0 = vsubq_s16(vSumi0, vaddq_s16(cYsum0, cYsum1));

    vSumf += cD * (float)vaddlvq_s16(vSumi0);
#endif
  }

  *pS = vSumf;

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

#if defined(__ARM_NEON)
  float vSumf = 0.0f;

  const uint8x16_t cM3 = vdupq_n_u8(3);

  for (int vI = 0; vI < cNb; ++vI) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t vSumi0 = vdupq_n_s32(0);
    int32x4_t vSumi1 = vdupq_n_s32(0);
#else
    int16x8_t vSumi0 = vdupq_n_s16(0);
    int16x8_t vSumi1 = vdupq_n_s16(0);
#endif

    for (size_t vJ = 0; vJ < sizeof(vX->qs); vJ += 32) {
      uint8x16_t vQx0 = vld1q_u8(vX[vI].qs + vJ);
      uint8x16_t vQx1 = vld1q_u8(vX[vI].qs + vJ + 16);
      uint8x16_t vQx2 = vshrq_n_u8(vQx0, 2);
      uint8x16_t vQx3 = vshrq_n_u8(vQx1, 2);
      uint8x16_t vQx4 = vshrq_n_u8(vQx0, 4);
      uint8x16_t vQx5 = vshrq_n_u8(vQx1, 4);
      uint8x16_t vQx6 = vshrq_n_u8(vQx0, 6);
      uint8x16_t vQx7 = vshrq_n_u8(vQx1, 6);

      int8x16_t vSqx0 = vreinterpretq_s8_u8(vandq_u8(vQx0, cM3));
      int8x16_t vSqx1 = vreinterpretq_s8_u8(vandq_u8(vQx1, cM3));
      int8x16_t vSqx2 = vreinterpretq_s8_u8(vandq_u8(vQx2, cM3));
      int8x16_t vSqx3 = vreinterpretq_s8_u8(vandq_u8(vQx3, cM3));
      int8x16_t vSqx4 = vreinterpretq_s8_u8(vandq_u8(vQx4, cM3));
      int8x16_t vSqx5 = vreinterpretq_s8_u8(vandq_u8(vQx5, cM3));
      int8x16_t vSqx6 = vreinterpretq_s8_u8(vandq_u8(vQx6, cM3));
      int8x16_t vSqx7 = vreinterpretq_s8_u8(vandq_u8(vQx7, cM3));

      const int8x16_t cQy0 = vld1q_s8(vY[vI].qs + vJ * 4 + 0);
      const int8x16_t cQy1 = vld1q_s8(vY[vI].qs + vJ * 4 + 16);
      const int8x16_t cQy2 = vld1q_s8(vY[vI].qs + vJ * 4 + 32);
      const int8x16_t cQy3 = vld1q_s8(vY[vI].qs + vJ * 4 + 48);
      const int8x16_t cQy4 = vld1q_s8(vY[vI].qs + vJ * 4 + 64);
      const int8x16_t cQy5 = vld1q_s8(vY[vI].qs + vJ * 4 + 80);
      const int8x16_t cQy6 = vld1q_s8(vY[vI].qs + vJ * 4 + 96);
      const int8x16_t cQy7 = vld1q_s8(vY[vI].qs + vJ * 4 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
      vSumi0 = vdotq_s32(vSumi0, vSqx0, cQy0);
      vSumi1 = vdotq_s32(vSumi1, vSqx1, cQy1);
      vSumi0 = vdotq_s32(vSumi0, vSqx2, cQy2);
      vSumi1 = vdotq_s32(vSumi1, vSqx3, cQy3);
      vSumi0 = vdotq_s32(vSumi0, vSqx4, cQy4);
      vSumi1 = vdotq_s32(vSumi1, vSqx5, cQy5);
      vSumi0 = vdotq_s32(vSumi0, vSqx6, cQy6);
      vSumi1 = vdotq_s32(vSumi1, vSqx7, cQy7);
#else
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx0), vget_low_s8(cQy0));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx0), vget_high_s8(cQy0));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx1), vget_low_s8(cQy1));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx1), vget_high_s8(cQy1));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx2), vget_low_s8(cQy2));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx2), vget_high_s8(cQy2));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx3), vget_low_s8(cQy3));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx3), vget_high_s8(cQy3));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx4), vget_low_s8(cQy4));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx4), vget_high_s8(cQy4));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx5), vget_low_s8(cQy5));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx5), vget_high_s8(cQy5));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx6), vget_low_s8(cQy6));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx6), vget_high_s8(cQy6));
      vSumi0 = vmlal_s8(vSumi0, vget_low_s8(vSqx7), vget_low_s8(cQy7));
      vSumi1 = vmlal_s8(vSumi1, vget_high_s8(vSqx7), vget_high_s8(cQy7));
#endif
    }

    const int16x8_t cYsum0 = vld1q_s16(vY[vI].bsums);
    const int16x8_t cYsum1 = vld1q_s16(vY[vI].bsums + 8);

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

#if defined(__ARM_FEATURE_DOTPROD)
    vSumi0 = vaddq_s32(vSumi0, vSumi1);
    vSumi0 = vsubq_s32(vSumi0, vpaddlq_s16(vaddq_s16(cYsum0, cYsum1)));

    vSumf += cD * (float)vaddvq_s32(vSumi0);
#else
    vSumi0 = vaddq_s16(vSumi0, vSumi1);
    vSumi0 = vsubq_s16(vSumi0, vaddq_s16(cYsum0, cYsum1));

    vSumf += cD * (float)vaddlvq_s16(vSumi0);
#endif
  }

  *pS = vSumf;

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

#ifdef __ARM_FEATURE_SVE
  const int cVectorLength = svcntb() * 8;
  const svuint8_t cM3s = svdup_n_u8(0x3);
  const svuint32_t cM4s = svdup_n_u32(0xF);
  const svint32_t cVzeroSv = svdup_n_s32(0);
  svfloat32_t vAccSum = svdup_n_f32(0);
  svbool_t vPredS32 = svptrue_pat_b32(SV_VL4);

  switch (cVectorLength) {
  case 128:
    for (int vI = 0; vI < cNb; ++vI) {
      const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
      svfloat32_t vDBroad = svdup_n_f32((float32_t)cD);
      const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);
      svfloat32_t vDminBroad = svdup_n_f32((float32_t)cDmin);

      const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
      const int8_t *GGML_RESTRICT vQ8Sv = vY[vI].qs;
      const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;

      svuint32_t vMinsAndScalesSve = svld1ub_u32(svptrue_b32(), vSc);
      const svint32_t cMinsSv1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), vMinsAndScalesSve, 4));

      vMinsAndScalesSve = svld1ub_u32(svptrue_b32(), vSc + 4);
      const svint32_t cMinsSv2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), vMinsAndScalesSve, 4));

      svint32_t vQ8sumsSv1 = svld1sh_s32(svptrue_b32(), vY[vI].bsums);
      svint32_t vQ8sumsSv2 = svld1sh_s32(svptrue_b32(), vY[vI].bsums + 4);

      const svint32_t cS0 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), cMinsSv1, vQ8sumsSv1),
                                        svmul_s32_x(svptrue_b32(), cMinsSv2, vQ8sumsSv2));

      vMinsAndScalesSve = svld1ub_u32(svptrue_b32(), vSc + 8);
      const svint32_t cMinsSv3 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), vMinsAndScalesSve, 4));

      vMinsAndScalesSve = svld1ub_u32(svptrue_b32(), vSc + 12);
      const svint32_t cMinsSv4 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), vMinsAndScalesSve, 4));

      vQ8sumsSv1 = svld1sh_s32(svptrue_b32(), vY[vI].bsums + 8);
      vQ8sumsSv2 = svld1sh_s32(svptrue_b32(), vY[vI].bsums + 12);

      svint32_t vS1 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), cMinsSv3, vQ8sumsSv1),
                                  svmul_s32_x(svptrue_b32(), cMinsSv4, vQ8sumsSv2));

      svfloat32_t vTemp = svcvt_f32_s32_x(svptrue_b32(), svadd_s32_x(svptrue_b32(), cS0, vS1));

      vAccSum = svmla_f32_m(svptrue_b32(), vAccSum, vTemp, vDminBroad);

      svint32_t vSumi1 = svdup_n_s32(0);

      {
        const svuint8_t cQ2bits1 = svld1_u8(svptrue_b8(), vQ2);
        svint8_t vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), cQ2bits1, cM3s));
        svint8_t vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;
        const svint32_t cScalesSv =
            svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), vSc), cM4s));

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv, 0));

        const svuint8_t cQ2bits3 = svld1_u8(svptrue_b8(), vQ2 + 16);
        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), cQ2bits3, cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv, 1));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits1, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv, 2));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits3, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv, 3));

        const svint32_t cScalesSv1 =
            svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), vSc + 4), cM4s));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits1, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv1, 0));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits3, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv1, 1));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits1, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv1, 2));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits3, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv1, 3));

        //-------------------------------

        vQ2 += 32;
        const svint32_t cScalesSv2 =
            svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), vSc + 8), cM4s));
        const svuint8_t cQ2bits2 = svld1_u8(svptrue_b8(), vQ2);

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), cQ2bits2, cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv2, 0));

        const svuint8_t cQ2bits4 = svld1_u8(svptrue_b8(), vQ2 + 16);
        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), cQ2bits4, cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv2, 1));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits2, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv2, 2));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits4, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv2, 3));

        const svint32_t cScalesSv3 =
            svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), vSc + 12), cM4s));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits2, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv3, 0));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits4, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv3, 1));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits2, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv3, 2));

        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ2bits4, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vSumi1 = svmla_s32_m(svptrue_b32(), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv),
                             svdup_lane_s32(cScalesSv3, 3));
      }
      vAccSum = svmla_f32_m(svptrue_b32(), vAccSum, svcvt_f32_s32_x(svptrue_b32(), vSumi1), vDBroad);
    }
    *pS = svaddv_f32(svptrue_b32(), vAccSum);
    break;

  case 256:
  case 512:
    for (int vI = 0; vI < cNb; ++vI) {
      const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
      svfloat32_t vDBroad = svdup_n_f32((float32_t)cD);
      const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);
      svfloat32_t vDminBroad = svdup_n_f32((float32_t)cDmin);

      const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
      const int8_t *GGML_RESTRICT vQ8Sv = vY[vI].qs;
      const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;

      const svuint32_t cMinsAndScalesSve = svld1ub_u32(svptrue_pat_b32(SV_VL8), vSc);
      vSc += 8;
      const svint32_t cScalesSv = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), cMinsAndScalesSve, cM4s));
      const svint32_t cMinsSv1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), cMinsAndScalesSve, 4));
      svint32_t vQ8sumsSv1 = svld1sh_s32(svptrue_pat_b32(SV_VL8), vY[vI].bsums);

      const svuint32_t cMinsAndScalesSve1 = svld1ub_u32(svptrue_pat_b32(SV_VL8), vSc);
      const svint32_t cScalesSv1 =
          svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), cMinsAndScalesSve1, cM4s));
      const svint32_t cMinsSv2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), cMinsAndScalesSve1, 4));

      svint32_t vQ8sumsSv2 = svld1sh_s32(svptrue_pat_b32(SV_VL8), vY[vI].bsums + 8);

      svfloat32_t vTemp = svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8),
                                          svadd_s32_x(svptrue_pat_b32(SV_VL8),
                                                      svmul_s32_x(svptrue_pat_b32(SV_VL8), cMinsSv1, vQ8sumsSv1),
                                                      svmul_s32_x(svptrue_pat_b32(SV_VL8), cMinsSv2, vQ8sumsSv2)));

      vAccSum = svmla_f32_m(svptrue_pat_b32(SV_VL8), vAccSum, vTemp, vDminBroad);

      svint32_t vSumi1 = svdup_n_s32(0);

      {
        const svuint8_t cQ2bits1 = svld1_u8(svptrue_pat_b8(SV_VL32), vQ2);
        svint8_t vQ2bytesSv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), cQ2bits1, cM3s));
        svint8_t vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        svint32_t vScale1 = svsel(vPredS32, svdup_lane_s32(cScalesSv, 0), svdup_lane_s32(cScalesSv, 1));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale1);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits1, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        svint32_t vScale2 = svsel(vPredS32, svdup_lane_s32(cScalesSv, 2), svdup_lane_s32(cScalesSv, 3));
        vSumi1 =
            svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(svdup_n_s32(0), vQ2bytesSv, vQ8bytesSv), vScale2);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits1, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale1 = svsel(vPredS32, svdup_lane_s32(cScalesSv, 4), svdup_lane_s32(cScalesSv, 5));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale1);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits1, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale2 = svsel(vPredS32, svdup_lane_s32(cScalesSv, 6), svdup_lane_s32(cScalesSv, 7));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale2);

        vQ2 += 32;

        const svuint8_t cQ2bits2 = svld1_u8(svptrue_pat_b8(SV_VL32), vQ2);
        vQ2bytesSv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), cQ2bits2, cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale1 = svsel(vPredS32, svdup_lane_s32(cScalesSv1, 0), svdup_lane_s32(cScalesSv1, 1));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale1);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits2, 2), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale2 = svsel(vPredS32, svdup_lane_s32(cScalesSv1, 2), svdup_lane_s32(cScalesSv1, 3));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale2);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits2, 4), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale1 = svsel(vPredS32, svdup_lane_s32(cScalesSv1, 4), svdup_lane_s32(cScalesSv1, 5));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale1);

        vQ2bytesSv = svreinterpret_s8_u8(
            svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ2bits2, 6), cM3s));
        vQ8bytesSv = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vScale2 = svsel(vPredS32, svdup_lane_s32(cScalesSv1, 6), svdup_lane_s32(cScalesSv1, 7));
        vSumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cVzeroSv, vQ2bytesSv, vQ8bytesSv), vScale2);
      }
      vAccSum =
          svmla_f32_m(svptrue_pat_b32(SV_VL8), vAccSum, svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), vSumi1), vDBroad);
    }
    *pS = svaddv_f32(svptrue_pat_b32(SV_VL8), vAccSum);
    break;

  default:
    assert(false && "Unsupported vector length");
    break;
  }

#elif __ARM_NEON
  const uint8x16_t m3 = vdupq_n_u8(0x3);
  const uint8x16_t m4 = vdupq_n_u8(0xF);

  const int32x4_t vzero = vdupq_n_s32(0);

  ggml_int8x16x2_t q2bytes;
  uint8_t aux[16];

  float sum = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT q8 = vY[vI].qs;
    const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;

    const uint8x16_t mins_and_scales = vld1q_u8(vSc);
    const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
    vst1q_u8(aux, scales);

    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(vY[vI].bsums);
    const ggml_int16x8x2_t mins16 = {
        {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
    const int32x4_t cS0 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
                                    vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
    const int32x4_t vS1 = vaddq_s32(vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
                                    vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
    sum += cDmin * vaddvq_s32(vaddq_s32(cS0, vS1));

    int isum = 0;
    int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)                                                                               \
  isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is + (index)];                       \
  isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is + 1 + (index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)                                                                  \
  q8bytes = ggml_vld1q_s8_x2(q8);                                                                                      \
  q8 += 32;                                                                                                            \
  q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));                              \
  q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));                              \
  MULTIPLY_ACCUM_WITH_SCALE((index));

    for (int j = 0; j < QK_K / 128; ++j) {
      const ggml_uint8x16x2_t q2bits = ggml_vld1q_u8_x2(vQ2);
      vQ2 += 32;

      ggml_int8x16x2_t q8bytes = ggml_vld1q_s8_x2(q8);
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

      MULTIPLY_ACCUM_WITH_SCALE(0);

      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
      SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

      is += 8;
    }

    sum += cD * isum;
  }

  *pS = sum;

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

#if defined(__ARM_FEATURE_SVE)

  uint32_t vAux[3];
  uint32_t vUtmp[4];

  const int8_t cM32 = 32;
  const int cVectorLength = svcntb() * 8;
  const svuint8_t cM3bSv = svdup_n_u8(0x3);
  const svint32_t cVzeroSv = svdup_n_s32(0);

  const svuint8_t cM0Sv = svdup_n_u8(1);
  const svuint8_t cM1Sv = svlsl_n_u8_x(svptrue_b8(), cM0Sv, 1);
  const svuint8_t cM2Sv = svlsl_n_u8_x(svptrue_b8(), cM0Sv, 2);
  const svuint8_t cM3Sv = svlsl_n_u8_x(svptrue_b8(), cM0Sv, 3);

  float vSum = 0;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ3Sv = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQhSv = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8Sv = vY[vI].qs;

    // Set up scales
    memcpy(vAux, vX[vI].scales, 12);
    vUtmp[3] = ((vAux[1] >> 4) & cKmask2) | (((vAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((vAux[0] >> 4) & cKmask2) | (((vAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (vAux[1] & cKmask2) | (((vAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (vAux[0] & cKmask2) | (((vAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;

    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= cM32;

    switch (cVectorLength) {
    case 128: {
      svuint8_t vQhbitsSv1 = svld1_u8(svptrue_b8(), vQhSv);
      svuint8_t vQhbitsSv2 = svld1_u8(svptrue_b8(), vQhSv + 16);
      svuint8_t vQ3hSv;

      svint32_t vSumi11 = svdup_n_s32(0);
      svint8_t vQ3bytesSv;

      for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

        const svuint8_t cQ3bitsSv = svld1_u8(svptrue_b8(), vQ3Sv);
        vQ3Sv += 16;
        const svuint8_t cQ3bitsSv1 = svld1_u8(svptrue_b8(), vQ3Sv);
        vQ3Sv += 16;
        svint8_t vQ8bytes1Sv1 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;
        svint8_t vQ8bytes1Sv2 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vQ3hSv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM0Sv, vQhbitsSv1), 2);
        vQ3bytesSv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), cQ3bitsSv, cM3bSv)),
                                svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1),
                              svdup_n_s32((int32_t)vScale[0]));

        vQ3hSv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM0Sv, vQhbitsSv2), 2);
        vQ3bytesSv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), cQ3bitsSv1, cM3bSv)),
                                svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2),
                              svdup_n_s32((int32_t)vScale[1]));

        vQ8bytes1Sv1 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;
        vQ8bytes1Sv2 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vQ3hSv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM1Sv, vQhbitsSv1), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv, 2), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1),
                              svdup_n_s32((int32_t)vScale[2]));

        vQ3hSv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM1Sv, vQhbitsSv2), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv1, 2), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2),
                              svdup_n_s32((int32_t)vScale[3]));

        vScale += 4;
        vQ8bytes1Sv1 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;
        vQ8bytes1Sv2 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vQ3hSv = svbic_u8_x(svptrue_b8(), cM2Sv, vQhbitsSv1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv, 4), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1),
                              svdup_n_s32((int32_t)vScale[0]));

        vQ3hSv = svbic_u8_x(svptrue_b8(), cM2Sv, vQhbitsSv2);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv1, 4), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2),
                              svdup_n_s32((int32_t)vScale[1]));

        vQ8bytes1Sv1 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;
        vQ8bytes1Sv2 = svld1_s8(svptrue_b8(), vQ8Sv);
        vQ8Sv += 16;

        vQ3hSv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM3Sv, vQhbitsSv1), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv, 6), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1),
                              svdup_n_s32((int32_t)vScale[2]));

        vQ3hSv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), cM3Sv, vQhbitsSv2), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_b8(),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), cQ3bitsSv1, 6), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vSumi11 = svmla_s32_m(svptrue_b32(), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2),
                              svdup_n_s32((int32_t)vScale[3]));

        if (vJ == 0) {
          vQhbitsSv1 = svlsr_n_u8_x(svptrue_b8(), vQhbitsSv1, 4);
          vQhbitsSv2 = svlsr_n_u8_x(svptrue_b8(), vQhbitsSv2, 4);
        }

        vScale += 4;
      }

      vSum += cD * (svaddv_s32(svptrue_b32(), vSumi11));
    } break;
    case 256:
    case 512: {
      svuint8_t vQhbitsSv = svld1_u8(svptrue_pat_b8(SV_VL32), vQhSv);
      svuint8_t vQ3hSv;

      svint32_t vSumi11 = svdup_n_s32(0);
      svint8_t vQ3bytesSv;

      for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

        const svuint8_t cQ3bitsSv = svld1_u8(svptrue_pat_b8(SV_VL32), vQ3Sv);
        vQ3Sv += 32;
        svint8_t vQ8bytes1Sv1 = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;
        svint8_t vQ8bytes1Sv2 = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vQ3hSv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), cM0Sv, vQhbitsSv), 2);
        vQ3bytesSv = svsub_s8_x(svptrue_pat_b8(SV_VL32),
                                svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), cQ3bitsSv, cM3bSv)),
                                svreinterpret_s8_u8(vQ3hSv));

        svint32_t vScale1 =
            svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)vScale[0]), svdup_n_s32((int32_t)vScale[1]));
        vSumi11 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1), vScale1);

        vQ3hSv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), cM1Sv, vQhbitsSv), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_pat_b8(SV_VL32),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                      svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ3bitsSv, 2), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vScale1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)vScale[2]), svdup_n_s32((int32_t)vScale[3]));
        vSumi11 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2), vScale1);

        vScale += 4;
        vQ8bytes1Sv1 = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;
        vQ8bytes1Sv2 = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8Sv);
        vQ8Sv += 32;

        vQ3hSv = svbic_u8_x(svptrue_pat_b8(SV_VL32), cM2Sv, vQhbitsSv);
        vQ3bytesSv =
            svsub_s8_x(svptrue_pat_b8(SV_VL32),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                      svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ3bitsSv, 4), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vScale1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)vScale[0]), svdup_n_s32((int32_t)vScale[1]));
        vSumi11 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv1), vScale1);

        vQ3hSv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), cM3Sv, vQhbitsSv), 1);
        vQ3bytesSv =
            svsub_s8_x(svptrue_pat_b8(SV_VL32),
                       svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32),
                                                      svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ3bitsSv, 6), cM3bSv)),
                       svreinterpret_s8_u8(vQ3hSv));

        vScale1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)vScale[2]), svdup_n_s32((int32_t)vScale[3]));
        vSumi11 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vSumi11, svdot_s32(cVzeroSv, vQ3bytesSv, vQ8bytes1Sv2), vScale1);

        if (vJ == 0) {
          vQhbitsSv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), vQhbitsSv, 4);
        }

        vScale += 4;
      }

      vSum += cD * (svaddv_s32(svptrue_pat_b32(SV_VL8), vSumi11));
    } break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }
  }
  *pS = vSum;

#elif __ARM_NEON

  uint32_t vAux[3];
  uint32_t vUtmp[4];

  const uint8x16_t m3b = vdupq_n_u8(0x3);
  const int32x4_t vzero = vdupq_n_s32(0);

  const uint8x16_t m0 = vdupq_n_u8(1);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);
  const int8_t cM32 = 32;

  ggml_int8x16x4_t q3bytes;

  float vSum = 0;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT q3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT qh = vX[vI].hmask;
    const int8_t *GGML_RESTRICT q8 = vY[vI].qs;

    ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);

    ggml_uint8x16x4_t q3h;

    int32_t isum = 0;

    // Set up scales
    memcpy(vAux, vX[vI].scales, 12);
    vUtmp[3] = ((vAux[1] >> 4) & cKmask2) | (((vAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((vAux[0] >> 4) & cKmask2) | (((vAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (vAux[1] & cKmask2) | (((vAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (vAux[0] & cKmask2) | (((vAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;
    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= cM32;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const ggml_uint8x16x2_t q3bits = ggml_vld1q_u8_x2(q3);
      q3 += 32;
      const ggml_int8x16x4_t q8bytes_1 = ggml_vld1q_s8_x4(q8);
      q8 += 64;
      const ggml_int8x16x4_t q8bytes_2 = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
      q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
      q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
      q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

      q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * vScale[0];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * vScale[1];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * vScale[2];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * vScale[3];

      vScale += 4;

      q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
      q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
      q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
      q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

      q3bytes.val[0] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * vScale[0];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * vScale[1];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * vScale[2];
      isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * vScale[3];

      vScale += 4;

      if (vJ == 0) {
        qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
        qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
      }
    }
    vSum += cD * isum;
  }

  *pS = vSum;

#else
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q3_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#ifdef __ARM_FEATURE_SVE
static inline svuint32_t fGgmlDecodeQ4scalesAndMinsForMmla(const uint32_t *pVxScales) {
  const svbool_t cPgAll = svptrue_pat_b32(SV_VL4);
  const svbool_t cPgFalse = svpfalse_b();         // 0x0000
  const svbool_t cPgLo8 = svwhilelt_b8_s32(0, 8); // 0x00ff
  const svbool_t cPgOdd = svzip1_b32(cPgFalse, cPgLo8);

  svuint32_t vVutmpHi, vVutmpLo;
  svuint32_t vVx01 = svld1_u32(cPgLo8, pVxScales);
  vVutmpHi = svzip1_u32(vVx01, vVx01);
  vVutmpHi = svlsr_n_u32_m(cPgOdd, vVutmpHi, 2);
  vVutmpHi =
      svreinterpret_u32_u64(svand_n_u64_x(cPgAll, svreinterpret_u64_u32(vVutmpHi), UINT64_C(0x303030303f3f3f3f)));
  const svuint32_t cVx2 = svdup_u32(pVxScales[2]);
  vVutmpLo = svlsr_u32_x(cPgAll, cVx2, svreinterpret_u32_s32(svindex_s32(-2, 2)));
  vVutmpLo = svand_n_u32_z(cPgOdd, vVutmpLo, UINT32_C(0x0f0f0f0f));
  svuint32_t vVutmp = svorr_u32_z(cPgAll, vVutmpHi, vVutmpLo);
  return vVutmp;
}
#endif

void ggml_vec_dot_q4_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
#ifdef __ARM_FEATURE_MATMUL_INT8
  assert((pNrc == 2) || (pNrc == 1));
#else
  assert(pNrc == 1);
#endif
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
#ifdef __ARM_FEATURE_SVE
  const int cVectorLength = ggml_cpu_get_sve_cnt() * 8;
#endif

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    svbool_t vPg322 = svptrue_pat_b32(SV_VL2);

    const block_q4_K *GGML_RESTRICT vVx0 = pVx;
    const block_q8_K *GGML_RESTRICT vVy0 = pVy;
    const block_q4_K *GGML_RESTRICT vVx1 = (const block_q4_K *)((const uint8_t *)pVx + pBx);
    const block_q8_K *GGML_RESTRICT vVy1 = (const block_q8_K *)((const uint8_t *)pVy + pBy);

    union {
      uint32_t aU32[8];
      uint64_t aU64[4];
    } vNewUtmp;

    svfloat32_t vSumf1 = svdup_n_f32(0);

    switch (cVectorLength) {
    case 128: {
      svbool_t vPgFalse = svpfalse_b();
      svbool_t vPgLo8 = svwhilelt_b8_s32(0, 8);
      svbool_t vVminsMask1 = svzip1_b32(vPgLo8, vPgFalse);
      svbool_t vVminsMask2 = svzip1_b32(vPgFalse, vPgLo8);
      svbool_t vPg128All = svptrue_pat_b8(SV_VL16);
      for (int vI = 0; vI < cNb; ++vI) {
        svfloat32_t vVyD = svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d));
        svfloat32_t vVxD =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].d)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].d)));
        svfloat32_t vSvsuperBlockScales = svmul_f32_x(vPg128All, vVyD, vVxD);
        svfloat32_t vVxDmins =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].dmin)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].dmin)));
        svfloat32_t vVyDmins = svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d));
        svfloat32_t vSvdmins = svmul_n_f32_x(vPg128All, svmul_f32_x(vPg128All, vVyDmins, vVxDmins), -1);
        const uint8_t *GGML_RESTRICT vQ40 = vVx0[vI].qs;
        const int8_t *GGML_RESTRICT vQ80 = vVy0[vI].qs;
        const uint8_t *GGML_RESTRICT vQ41 = vVx1[vI].qs;
        const int8_t *GGML_RESTRICT vQ81 = vVy1[vI].qs;
        svint16_t vLo = svld1_s16(vPg128All, vVy0[vI].bsums + 0);
        svint16_t vHi = svld1_s16(vPg128All, vVy0[vI].bsums + 8);
        svint16_t vSumTmp1 = svuzp1_s16(vLo, vHi);
        svint16_t vSumTmp2 = svuzp2_s16(vLo, vHi);
        svint16_t vSvq8sums0 = svadd_s16_x(vPg128All, vSumTmp1, vSumTmp2);
        vLo = svld1_s16(vPg128All, vVy1[vI].bsums + 0);
        vHi = svld1_s16(vPg128All, vVy1[vI].bsums + 8);
        vSumTmp1 = svuzp1(vLo, vHi);
        vSumTmp2 = svuzp2(vLo, vHi);
        svint16_t vSvq8sums1 = svadd_s16_x(vPg128All, vSumTmp1, vSumTmp2);
        svuint32_t vDecodedScales0 = fGgmlDecodeQ4scalesAndMinsForMmla((const uint32_t *)vVx0[vI].scales);
        svuint32_t vDecodedScales1 = fGgmlDecodeQ4scalesAndMinsForMmla((const uint32_t *)vVx1[vI].scales);
        svuint32x2_t vDecodedScales = svcreate2_u32(vDecodedScales0, vDecodedScales1);
        svst2_u32(vPg128All, vNewUtmp.aU32, vDecodedScales);
        svint16_t vSvmins80 = svreinterpret_s16_u16(
            svunpklo_u16(svreinterpret_u8_u32(svuzp1_u32(svld1_u32(vVminsMask1, vNewUtmp.aU32 + 4), svdup_n_u32(0)))));
        svint16_t vSvmins81 = svreinterpret_s16_u16(
            svunpklo_u16(svreinterpret_u8_u32(svuzp2_u32(svld1_u32(vVminsMask2, vNewUtmp.aU32 + 4), svdup_n_u32(0)))));
        svint32_t vSvsumfsTmp1 = svreinterpret_s32_s64(svdot_s64(svdup_n_s64(0), vSvq8sums0, vSvmins80));
        svint32_t vSvsumfsTmp2 = svreinterpret_s32_s64(svdot_s64(svdup_n_s64(0), vSvq8sums0, vSvmins81));
        svint32_t vSvsumfsTmp3 = svtrn1_s32(vSvsumfsTmp1, vSvsumfsTmp2);
        svint32_t vSvsumfsTmp4 = svreinterpret_s32_s64(svdot_s64(svdup_n_s64(0), vSvq8sums1, vSvmins80));
        svint32_t vSvsumfsTmp5 = svreinterpret_s32_s64(svdot_s64(svdup_n_s64(0), vSvq8sums1, vSvmins81));
        svint32_t vSvsumfsTmp6 = svtrn1_s32(vSvsumfsTmp4, vSvsumfsTmp5);
        svint32_t vSvsumfsTmp7 =
            svreinterpret_s32_s64(svtrn2_s64(svreinterpret_s64_s32(vSvsumfsTmp3), svreinterpret_s64_s32(vSvsumfsTmp6)));
        svint32_t vSvsumfsTmp8 =
            svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(vSvsumfsTmp3), svreinterpret_s64_s32(vSvsumfsTmp6)));
        svint32_t vSvsumfsTmp = svadd_s32_x(vPg128All, vSvsumfsTmp7, vSvsumfsTmp8);
        svint32_t vSvscales, vSumi1, vSumi2;
        svint32_t vAccSumif1 = svdup_n_s32(0);
        svint32_t vAccSumif2 = svdup_n_s32(0);
        svint8_t vQ4bytes0L, vQ4bytes0H, vQ4bytes1L, vQ4bytes1H, vL0, vL1, vL2, vL3, vQ8bytes0H, vQ8bytes0L, vQ8bytes1H,
            vQ8bytes1L, vR0, vR1, vR2, vR3;
#pragma GCC unroll 1
        for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
          vQ4bytes0L = svreinterpret_s8_u8(svand_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ40), 0xf));
          vQ4bytes1L = svreinterpret_s8_u8(svand_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ41), 0xf));
          vQ4bytes0H = svreinterpret_s8_u8(svand_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ40 + 16), 0xf));
          vQ4bytes1H = svreinterpret_s8_u8(svand_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ41 + 16), 0xf));
          vL0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ4bytes0L), svreinterpret_s64_s8(vQ4bytes1L)));
          vL1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ4bytes0L), svreinterpret_s64_s8(vQ4bytes1L)));
          vL2 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ4bytes0H), svreinterpret_s64_s8(vQ4bytes1H)));
          vL3 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ4bytes0H), svreinterpret_s64_s8(vQ4bytes1H)));
          vQ8bytes0H = svld1_s8(vPg128All, vQ80);
          vQ8bytes1H = svld1_s8(vPg128All, vQ81);
          vQ8bytes0L = svld1_s8(vPg128All, vQ80 + 16);
          vQ8bytes1L = svld1_s8(vPg128All, vQ81 + 16);
          vR0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0H), svreinterpret_s64_s8(vQ8bytes1H)));
          vR1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0H), svreinterpret_s64_s8(vQ8bytes1H)));
          vR2 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0L), svreinterpret_s64_s8(vQ8bytes1L)));
          vR3 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0L), svreinterpret_s64_s8(vQ8bytes1L)));
          vSumi1 =
              svmmla_s32(svmmla_s32(svmmla_s32(svmmla_s32(svdup_n_s32(0), vR0, vL0), vR1, vL1), vR2, vL2), vR3, vL3);
          vSvscales = svreinterpret_s32_u32(
              svlsr_n_u32_x(vPg128All,
                            svlsl_n_u32_x(vPg128All, svreinterpret_u32_u64(svdup_n_u64(vNewUtmp.aU64[vJ / 2])),
                                          8 * (4 - 2 * (vJ % 2) - 1)),
                            24));
          vAccSumif1 = svmla_s32_x(vPg128All, vAccSumif1, vSvscales, vSumi1);

          vQ4bytes0L = svreinterpret_s8_u8(svlsr_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ40), 4));
          vQ4bytes1L = svreinterpret_s8_u8(svlsr_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ41), 4));
          vQ4bytes0H = svreinterpret_s8_u8(svlsr_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ40 + 16), 4));
          vQ4bytes1H = svreinterpret_s8_u8(svlsr_n_u8_x(vPg128All, svld1_u8(vPg128All, vQ41 + 16), 4));
          vL0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ4bytes0L), svreinterpret_s64_s8(vQ4bytes1L)));
          vL1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ4bytes0L), svreinterpret_s64_s8(vQ4bytes1L)));
          vL2 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ4bytes0H), svreinterpret_s64_s8(vQ4bytes1H)));
          vL3 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ4bytes0H), svreinterpret_s64_s8(vQ4bytes1H)));
          vQ8bytes0H = svld1_s8(vPg128All, vQ80 + 32);
          vQ8bytes1H = svld1_s8(vPg128All, vQ81 + 32);
          vQ8bytes0L = svld1_s8(vPg128All, vQ80 + 48);
          vQ8bytes1L = svld1_s8(vPg128All, vQ81 + 48);
          vR0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0H), svreinterpret_s64_s8(vQ8bytes1H)));
          vR1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0H), svreinterpret_s64_s8(vQ8bytes1H)));
          vR2 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0L), svreinterpret_s64_s8(vQ8bytes1L)));
          vR3 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0L), svreinterpret_s64_s8(vQ8bytes1L)));
          vSumi2 =
              svmmla_s32(svmmla_s32(svmmla_s32(svmmla_s32(svdup_n_s32(0), vR0, vL0), vR1, vL1), vR2, vL2), vR3, vL3);
          vSvscales = svreinterpret_s32_u32(
              svlsr_n_u32_x(vPg128All,
                            svlsl_n_u32_x(vPg128All, svreinterpret_u32_u64(svdup_n_u64(vNewUtmp.aU64[vJ / 2])),
                                          8 * (4 - 2 * (vJ % 2) - 2)),
                            24));
          vAccSumif2 = svmla_s32_x(vPg128All, vAccSumif2, vSvscales, vSumi2);
          vQ40 += 32;
          vQ41 += 32;
          vQ80 += 64;
          vQ81 += 64;
        }
        vSumf1 = svmla_f32_x(vPg128All,
                             svmla_f32_x(vPg128All, vSumf1,
                                         svcvt_f32_x(vPg128All, svadd_s32_x(vPg128All, vAccSumif1, vAccSumif2)),
                                         vSvsuperBlockScales),
                             vSvdmins, svcvt_f32_s32_x(vPg128All, vSvsumfsTmp));
      } // end of for nb
    } // end of case 128
    break;
    case 256:
    case 512: {
      const svbool_t cPg324 = svptrue_pat_b32(SV_VL4);
      const svbool_t cPg816 = svptrue_pat_b8(SV_VL16);
      const svbool_t cPg256All = svptrue_pat_b8(SV_ALL);
      for (int vI = 0; vI < cNb; ++vI) {
        const uint8_t *GGML_RESTRICT vQ40 = vVx0[vI].qs;
        const int8_t *GGML_RESTRICT vQ80 = vVy0[vI].qs;
        const uint8_t *GGML_RESTRICT vQ41 = vVx1[vI].qs;
        const int8_t *GGML_RESTRICT vQ81 = vVy1[vI].qs;
        svint32_t vSvscales, vSumi1, vSumi2;
        svint32_t vAccSumif1 = svdup_n_s32(0);
        svint32_t vAccSumif2 = svdup_n_s32(0);
        svint8_t vL0, vL1, vL2, vL3, vR0, vR1, vR2, vR3;
        svfloat32_t vVxD =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].d)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].d)));
        svfloat64_t vVyDTmp = svreinterpret_f64_f32(svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d)));
        svfloat32_t vVyD = svreinterpret_f32_f64(svuzp1_f64(vVyDTmp, vVyDTmp));
        svfloat32_t vSvsuperBlockScales = svmul_f32_z(cPg324, vVyD, vVxD);
        svfloat32_t vVxDmins =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].dmin)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].dmin)));
        svfloat64_t vVyDminsTmp = svreinterpret_f64_f32(svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d)));
        svfloat32_t vVyDmins = svreinterpret_f32_f64(svuzp1_f64(vVyDminsTmp, vVyDminsTmp));
        svfloat32_t vSvdmins = svmul_n_f32_x(cPg324, svmul_f32_x(cPg324, vVxDmins, vVyDmins), -1);
        svint16_t vRc1 = svuzp1_s16(svld1_s16(cPg256All, vVy0[vI].bsums), svld1_s16(cPg256All, vVy1[vI].bsums));
        svint16_t vRc2 = svuzp2_s16(svld1_s16(cPg256All, vVy0[vI].bsums), svld1_s16(cPg256All, vVy1[vI].bsums));
        svint16_t vSvq8sums = svadd_s16_x(cPg256All, vRc1, vRc2);
        svuint32_t vDecodedScales0 = fGgmlDecodeQ4scalesAndMinsForMmla((const uint32_t *)vVx0[vI].scales);
        svuint32_t vDecodedScales1 = fGgmlDecodeQ4scalesAndMinsForMmla((const uint32_t *)vVx1[vI].scales);
        svuint32x2_t vDecodedScales = svcreate2_u32(vDecodedScales0, vDecodedScales1);
        svst2_u32(cPg816, vNewUtmp.aU32, vDecodedScales);
        svint16_t vNewSvq8sums0 =
            svreinterpret_s16_u64(svtrn1_u64(svreinterpret_u64_s16(vSvq8sums), svreinterpret_u64_s16(vSvq8sums)));
        svint16_t vNewSvq8sums1 =
            svreinterpret_s16_u64(svtrn2_u64(svreinterpret_u64_s16(vSvq8sums), svreinterpret_u64_s16(vSvq8sums)));
        svuint64_t vNewMins0 = svdup_u64(vNewUtmp.aU64[2]);
        svuint64_t vNewMins1 = svdup_u64(vNewUtmp.aU64[3]);
        svint16_t vNewSvmins80 = svreinterpret_s16_u16(svunpklo_u16(svreinterpret_u8_u64(vNewMins0)));
        svint16_t vNewSvmins81 = svreinterpret_s16_u16(svunpklo_u16(svreinterpret_u8_u64(vNewMins1)));
        svint64_t vDotProd0 = svdot_s64(svdup_s64(0), vNewSvmins80, vNewSvq8sums0);
        svint64_t vDotProd1 = svdot_s64(vDotProd0, vNewSvmins81, vNewSvq8sums1);
        svfloat32_t vConvertedDotProd1 = svcvt_f32_s64_x(cPg256All, vDotProd1);
        svfloat32_t vSvsumfsTmp = svuzp1_f32(vConvertedDotProd1, vConvertedDotProd1);

#pragma GCC unroll 1
        for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
          svuint8_t vQ4bytes0 = svand_n_u8_x(cPg256All, svld1_u8(cPg256All, vQ40), 0xf);
          svuint8_t vQ4bytes1 = svand_n_u8_x(cPg256All, svld1_u8(cPg256All, vQ41), 0xf);
          svuint8_t vQ4bytes2 = svlsr_n_u8_x(cPg256All, svld1_u8(cPg256All, vQ40), 4);
          svuint8_t vQ4bytes3 = svlsr_n_u8_x(cPg256All, svld1_u8(cPg256All, vQ41), 4);
          vL0 = svreinterpret_s8_u64(svzip1_u64(svreinterpret_u64_u8(vQ4bytes0), svreinterpret_u64_u8(vQ4bytes1)));
          vL1 = svreinterpret_s8_u64(svzip2_u64(svreinterpret_u64_u8(vQ4bytes0), svreinterpret_u64_u8(vQ4bytes1)));
          vL2 = svreinterpret_s8_u64(svzip1_u64(svreinterpret_u64_u8(vQ4bytes2), svreinterpret_u64_u8(vQ4bytes3)));
          vL3 = svreinterpret_s8_u64(svzip2_u64(svreinterpret_u64_u8(vQ4bytes2), svreinterpret_u64_u8(vQ4bytes3)));
          svint8_t vQ8bytes0 = svld1_s8(cPg256All, vQ80);
          svint8_t vQ8bytes1 = svld1_s8(cPg256All, vQ81);
          svint8_t vQ8bytes2 = svld1_s8(cPg256All, vQ80 + 32);
          svint8_t vQ8bytes3 = svld1_s8(cPg256All, vQ81 + 32);
          vR0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
          vR1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
          vR2 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes2), svreinterpret_s64_s8(vQ8bytes3)));
          vR3 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes2), svreinterpret_s64_s8(vQ8bytes3)));
          vSumi1 = svmmla(svmmla(svdup_n_s32(0), vR0, vL0), vR1, vL1);
          vSvscales = svreinterpret_s32_u32(
              svlsr_n_u32_x(cPg256All,
                            svlsl_n_u32_x(cPg256All, svreinterpret_u32_u64(svdup_n_u64(vNewUtmp.aU64[vJ / 2])),
                                          8 * (4 - 2 * (vJ % 2) - 1)),
                            24));
          vAccSumif1 = svmla_s32_x(cPg256All, vAccSumif1, vSvscales, vSumi1);
          vSumi2 = svmmla(svmmla(svdup_n_s32(0), vR2, vL2), vR3, vL3);
          vSvscales = svreinterpret_s32_u32(
              svlsr_n_u32_x(cPg256All,
                            svlsl_n_u32_x(cPg256All, svreinterpret_u32_u64(svdup_n_u64(vNewUtmp.aU64[vJ / 2])),
                                          8 * (4 - 2 * (vJ % 2) - 2)),
                            24));
          vAccSumif2 = svmla_s32_x(cPg256All, vAccSumif2, vSvscales, vSumi2);
          vQ40 += 32;
          vQ41 += 32;
          vQ80 += 64;
          vQ81 += 64;
        }
        svint32_t vAccSumif = svadd_s32_x(cPg256All, vAccSumif1, vAccSumif2);
        svint32_t vSwapAccSumif = svext_s32(vAccSumif, vAccSumif, 4);
        vAccSumif = svadd_s32_x(cPg324, vAccSumif, vSwapAccSumif);
        vSumf1 = svmla_f32_x(cPg324, svmla_f32_x(cPg324, vSumf1, svcvt_f32_x(cPg324, vAccSumif), vSvsuperBlockScales),
                             vSvdmins, vSvsumfsTmp);
      } // end of for nb
    } // end of case 256-512
    break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }

    svst1_f32(vPg322, pS, vSumf1);
    svst1_f32(vPg322, pS + pBs, svreinterpret_f32_u8(svext_u8(svreinterpret_u8_f32(vSumf1), svdup_n_u8(0), 8)));

    return;
  }
#elif defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const block_q4_K *GGML_RESTRICT x0 = vX;
    const block_q4_K *GGML_RESTRICT x1 = (const block_q4_K *)((const uint8_t *)pVx + pBx);
    const block_q8_K *GGML_RESTRICT y0 = vY;
    const block_q8_K *GGML_RESTRICT y1 = (const block_q8_K *)((const uint8_t *)pVy + pBy);

    const uint8x16_t cM4b = vdupq_n_u8(0x0f);

    float32x4_t vfsum = vdupq_n_f32(0.0f);

    for (int vI = 0; vI < cNb; ++vI, ++x0, ++x1, ++y0, ++y1) {
      const uint8_t *GGML_RESTRICT qx0 = x0->qs;
      const uint8_t *GGML_RESTRICT qx1 = x1->qs;
      const int8_t *GGML_RESTRICT qy0 = y0->qs;
      const int8_t *GGML_RESTRICT qy1 = y1->qs;

      // decode scales and mins
      int8_t x0_scales[8], x1_scales[8];
      int16x8_t x0_mins, x1_mins;
      {
        uint32_t scales_mins[3];
        memcpy(scales_mins, x0->scales, 12);
        const uint32_t mins_0_3 = scales_mins[1] & cKmask1;
        const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & cKmask2) | (((scales_mins[1] >> 6) & cKmask3) << 4);
        const uint32x2_t cMins = {mins_0_3, mins_4_7};
        x0_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(cMins)));
        uint32_t vScales[2];
        vScales[0] = scales_mins[0] & cKmask1;                                              // scales 0~3
        vScales[1] = (scales_mins[2] & cKmask2) | (((scales_mins[0] >> 6) & cKmask3) << 4); // scales 4~7
        memcpy(x0_scales, vScales, 8);
      }
      {
        uint32_t scales_mins[3];
        memcpy(scales_mins, x1->scales, 12);
        const uint32_t mins_0_3 = scales_mins[1] & cKmask1;
        const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & cKmask2) | (((scales_mins[1] >> 6) & cKmask3) << 4);
        const uint32x2_t cMins = {mins_0_3, mins_4_7};
        x1_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(cMins)));
        uint32_t vScales[2];
        vScales[0] = scales_mins[0] & cKmask1;                                              // scales 0~3
        vScales[1] = (scales_mins[2] & cKmask2) | (((scales_mins[0] >> 6) & cKmask3) << 4); // scales 4~7
        memcpy(x1_scales, vScales, 8);
      }

      int32x4_t visum = {0};

      // process 64 data points per iteration, totally 256 data points
      for (int vJ = 0; vJ < QK_K / 64; ++vJ, qx0 += 32, qx1 += 32, qy0 += 64, qy1 += 64) {
        const int8x16x4_t vVy0 = vld1q_s8_x4(qy0);
        const int8x16x4_t vVy1 = vld1q_s8_x4(qy1);

        int8x16_t vVx0[4], vVx1[4];
        {
          const uint8x16x2_t vv = vld1q_u8_x2(qx0);
          vVx0[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], cM4b));
          vVx0[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], cM4b));
          vVx0[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
          vVx0[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
        }
        {
          const uint8x16x2_t vv = vld1q_u8_x2(qx1);
          vVx1[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], cM4b));
          vVx1[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], cM4b));
          vVx1[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
          vVx1[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
        }

        // process 32 data points (share same block scale) per iteration
        for (int k = 0; k < 2; ++k) {
          const int blk = vJ * 2 + k;
          const int32x4_t block_scale = {
              x0_scales[blk],
              x0_scales[blk],
              x1_scales[blk],
              x1_scales[blk],
          };

          int32x4_t vr = {0};
          for (int l = 0; l < 2; ++l) {
            const int idx = k * 2 + l;
            const int64x2_t vx0_s64 = vreinterpretq_s64_s8(vVx0[idx]);
            const int64x2_t vx1_s64 = vreinterpretq_s64_s8(vVx1[idx]);
            const int64x2_t vy0_s64 = vreinterpretq_s64_s8(vVy0.val[idx]);
            const int64x2_t vy1_s64 = vreinterpretq_s64_s8(vVy1.val[idx]);
            const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vx0_s64, vx1_s64));
            const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vx0_s64, vx1_s64));
            const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vy0_s64, vy1_s64));
            const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vy0_s64, vy1_s64));
            vr = vmmlaq_s32(vr, vx_l, vy_l);
            vr = vmmlaq_s32(vr, vx_h, vy_h);
          }
          // apply block scale, will NOT overflow
          // block_scale * sum_256(int4*int8) <= 2^(8+8+4+8) = 28 bits
          visum = vmlaq_s32(visum, vr, block_scale);
        }
      }

      // adjust bias, apply superblock scale
      {
        int32_t bias[4];
        // no obvious uplift from sve sdot-16, just use neon mul add
        const int16x8_t y0_sums = vpaddq_s16(vld1q_s16(y0->bsums), vld1q_s16(y0->bsums + 8));
        const int16x8_t y1_sums = vpaddq_s16(vld1q_s16(y1->bsums), vld1q_s16(y1->bsums + 8));
        bias[0] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x0_mins)),
                                       vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x0_mins))));
        bias[1] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x0_mins)),
                                       vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x0_mins))));
        bias[2] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x1_mins)),
                                       vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x1_mins))));
        bias[3] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x1_mins)),
                                       vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x1_mins))));
        const float32x4_t dmins = {
            GGML_CPU_FP16_TO_FP32(x0->dmin) * y0->d,
            GGML_CPU_FP16_TO_FP32(x0->dmin) * y1->d,
            GGML_CPU_FP16_TO_FP32(x1->dmin) * y0->d,
            GGML_CPU_FP16_TO_FP32(x1->dmin) * y1->d,
        };
        vfsum = vmlsq_f32(vfsum, vcvtq_f32_s32(vld1q_s32(bias)), dmins);

        const float32x4_t superblock_scale = {
            GGML_CPU_FP16_TO_FP32(x0->d) * y0->d,
            GGML_CPU_FP16_TO_FP32(x0->d) * y1->d,
            GGML_CPU_FP16_TO_FP32(x1->d) * y0->d,
            GGML_CPU_FP16_TO_FP32(x1->d) * y1->d,
        };
        vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
      }
    }

    // vfsum = ABCD -> ACBD
    // AC -> s, BD -> (s+bs)
    vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
    vst1_f32(pS, vget_low_f32(vfsum));
    vst1_f32(pS + pBs, vget_high_f32(vfsum));

    return;
  }
#endif

#ifdef __ARM_FEATURE_SVE
  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const int16x8_t cQ8sums = vpaddq_s16(vld1q_s16(vY[vI].bsums), vld1q_s16(vY[vI].bsums + 8));

    memcpy(vUtmp, vX[vI].scales, K_SCALE_SIZE);

    uint32x2_t vMins8 = {0};
    vMins8 = vset_lane_u32(vUtmp[1] & cKmask1, vMins8, 0);
    vMins8 = vset_lane_u32(((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4), vMins8, 1);

    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[0] &= cKmask1;

    const int16x8_t cMins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vMins8)));
    const int32x4_t cProd = vaddq_s32(vmull_s16(vget_low_s16(cQ8sums), vget_low_s16(cMins)),
                                      vmull_s16(vget_high_s16(cQ8sums), vget_high_s16(cMins)));
    vSumf -= cDmin * vaddvq_s32(cProd);

    const uint8_t *vScales = (const uint8_t *)vUtmp;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const svuint8_t cM4b = svdup_n_u8(0xf);
    const svint32_t cMzero = svdup_n_s32(0);
    svint32_t vSumi1 = svdup_n_s32(0);
    svint32_t vSumi11 = svdup_n_s32(0);
    svint32_t vSumi12 = svdup_n_s32(0);
    svint32_t vSumi2 = svdup_n_s32(0);
    svint32_t vSumi21 = svdup_n_s32(0);
    svint32_t vSumi22 = svdup_n_s32(0);
    switch (cVectorLength) {
    case 128: {
      for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
        svint8_t vQ4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), vQ4), cM4b));
        svint8_t vQ8bytes = svld1_s8(svptrue_b8(), vQ8);
        vQ8 += 16;
        vSumi11 = svmla_n_s32_x(svptrue_b32(), vSumi11, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 0]);
        vQ4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), vQ4 + 16), cM4b));
        vQ8bytes = svld1_s8(svptrue_b8(), vQ8);
        vQ8 += 16;
        vSumi12 = svmla_n_s32_x(svptrue_b32(), vSumi12, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 0]);

        vQ4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), vQ4), 4));
        vQ8bytes = svld1_s8(svptrue_b8(), vQ8);
        vQ8 += 16;
        vSumi21 = svmla_n_s32_x(svptrue_b32(), vSumi21, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 1]);
        vQ4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), vQ4 + 16), 4));
        vQ8bytes = svld1_s8(svptrue_b8(), vQ8);
        vQ8 += 16;
        vSumi22 = svmla_n_s32_x(svptrue_b32(), vSumi22, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 1]);
        vQ4 += 32;
      }
      vSumi1 = svadd_s32_x(svptrue_b32(), vSumi11, vSumi12);
      vSumi2 = svadd_s32_x(svptrue_b32(), vSumi21, vSumi22);
      vSumf += cD * (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), vSumi1, vSumi2)));
    } break;
    case 256:
    case 512: {
      for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
        const svuint8_t cQ4bits = svld1_u8(svptrue_pat_b8(SV_VL32), vQ4);
        vQ4 += 32;
        svint8_t vQ4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_pat_b8(SV_VL32), cQ4bits, cM4b));
        svint8_t vQ8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8);
        vQ8 += 32;
        vSumi1 =
            svmla_n_s32_x(svptrue_pat_b32(SV_VL8), vSumi1, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 0]);

        vQ4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), cQ4bits, 4));
        vQ8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), vQ8);
        vQ8 += 32;
        vSumi2 =
            svmla_n_s32_x(svptrue_pat_b32(SV_VL8), vSumi2, svdot_s32(cMzero, vQ4bytes, vQ8bytes), vScales[2 * vJ + 1]);
      }
      vSumf += cD * (svaddv_s32(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), vSumi1, vSumi2)));
    } break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }
  }
  *pS = vSumf;
#elif defined __ARM_NEON
  const uint8x16_t cM4b = vdupq_n_u8(0xf);
  const int32x4_t cMzero = vdupq_n_s32(0);

  ggml_int8x16x2_t vQ4bytes;
  ggml_int8x16x2_t vQ8bytes;

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const int16x8_t cQ8sums = vpaddq_s16(vld1q_s16(vY[vI].bsums), vld1q_s16(vY[vI].bsums + 8));

    memcpy(vUtmp, vX[vI].scales, 12);

    uint32x2_t vMins8 = {0};
    vMins8 = vset_lane_u32(vUtmp[1] & cKmask1, vMins8, 0);
    vMins8 = vset_lane_u32(((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4), vMins8, 1);

    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[0] &= cKmask1;

    const int16x8_t cMins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vMins8)));
    const int32x4_t cProd = vaddq_s32(vmull_s16(vget_low_s16(cQ8sums), vget_low_s16(cMins)),
                                      vmull_s16(vget_high_s16(cQ8sums), vget_high_s16(cMins)));
    vSumf -= cDmin * vaddvq_s32(cProd);

    const uint8_t *vScales = (const uint8_t *)vUtmp;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    int32_t vSumi1 = 0;
    int32_t vSumi2 = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      const ggml_uint8x16x2_t cQ4bits = ggml_vld1q_u8_x2(vQ4);
      vQ4 += 32;

      vQ8bytes = ggml_vld1q_s8_x2(vQ8);
      vQ8 += 32;
      vQ4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(cQ4bits.val[0], cM4b));
      vQ4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(cQ4bits.val[1], cM4b));

      const int32x4_t p1 =
          ggml_vdotq_s32(ggml_vdotq_s32(cMzero, vQ4bytes.val[0], vQ8bytes.val[0]), vQ4bytes.val[1], vQ8bytes.val[1]);
      vSumi1 += vaddvq_s32(p1) * vScales[2 * vJ + 0];

      vQ8bytes = ggml_vld1q_s8_x2(vQ8);
      vQ8 += 32;
      vQ4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(cQ4bits.val[0], 4));
      vQ4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(cQ4bits.val[1], 4));

      const int32x4_t p2 =
          ggml_vdotq_s32(ggml_vdotq_s32(cMzero, vQ4bytes.val[0], vQ8bytes.val[0]), vQ4bytes.val[1], vQ8bytes.val[1]);

      vSumi2 += vaddvq_s32(p2) * vScales[2 * vJ + 1];
    }

    vSumf += cD * (vSumi1 + vSumi2);
  }

  *pS = vSumf;

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

#ifdef __ARM_NEON
  const uint8x16_t cM4b = vdupq_n_u8(0xf);
  const uint8x16_t cMone = vdupq_n_u8(1);
  const uint8x16_t cMtwo = vdupq_n_u8(2);
  const int32x4_t cMzero = vdupq_n_s32(0);

  ggml_int8x16x4_t vQ5bytes;

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const int16x8_t cQ8sums = vpaddq_s16(vld1q_s16(vY[vI].bsums), vld1q_s16(vY[vI].bsums + 8));

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const uint8x8_t cMins8 = vld1_u8((const uint8_t *)vUtmp + 8);
    const int16x8_t cMins = vreinterpretq_s16_u16(vmovl_u8(cMins8));
    const int32x4_t cProd = vaddq_s32(vmull_s16(vget_low_s16(cQ8sums), vget_low_s16(cMins)),
                                      vmull_s16(vget_high_s16(cQ8sums), vget_high_s16(cMins)));
    int32_t vSumiMins = vaddvq_s32(cProd);

    const uint8_t *vScales = (const uint8_t *)vUtmp;

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    ggml_uint8x16x2_t vQhbits = ggml_vld1q_u8_x2(vQh);

    ggml_uint8x16x4_t vQ5h;

    int32_t vSumi = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const ggml_uint8x16x2_t cQ5bits = ggml_vld1q_u8_x2(vQ5);
      vQ5 += 32;
      const ggml_int8x16x4_t cQ8bytes = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      vQ5h.val[0] = vshlq_n_u8(vandq_u8(cMone, vQhbits.val[0]), 4);
      vQ5h.val[1] = vshlq_n_u8(vandq_u8(cMone, vQhbits.val[1]), 4);
      vQ5h.val[2] = vshlq_n_u8(vandq_u8(cMtwo, vQhbits.val[0]), 3);
      vQ5h.val[3] = vshlq_n_u8(vandq_u8(cMtwo, vQhbits.val[1]), 3);
      vQhbits.val[0] = vshrq_n_u8(vQhbits.val[0], 2);
      vQhbits.val[1] = vshrq_n_u8(vQhbits.val[1], 2);

      vQ5bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(cQ5bits.val[0], cM4b), vQ5h.val[0]));
      vQ5bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(cQ5bits.val[1], cM4b), vQ5h.val[1]));
      vQ5bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ5bits.val[0], 4), vQ5h.val[2]));
      vQ5bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ5bits.val[1], 4), vQ5h.val[3]));

      vSumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(cMzero, vQ5bytes.val[0], cQ8bytes.val[0]), vQ5bytes.val[1],
                                         cQ8bytes.val[1])) *
               *vScales++;
      vSumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(cMzero, vQ5bytes.val[2], cQ8bytes.val[2]), vQ5bytes.val[3],
                                         cQ8bytes.val[3])) *
               *vScales++;
    }

    vSumf += cD * vSumi - cDmin * vSumiMins;
  }

  *pS = vSumf;

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
#ifdef __ARM_FEATURE_MATMUL_INT8
  assert((pNrc == 2) || (pNrc == 1));
#else
  assert(pNrc == 1);
#endif
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#ifdef __ARM_FEATURE_SVE
  const int cVectorLength = ggml_cpu_get_sve_cnt() * 8;
#endif
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const svbool_t cPg322 = svptrue_pat_b32(SV_VL2);

    svfloat32_t vSum = svdup_n_f32(0);

    const block_q6_K *GGML_RESTRICT vVx0 = pVx;
    const block_q8_K *GGML_RESTRICT vVy0 = pVy;
    const block_q6_K *GGML_RESTRICT vVx1 = (const block_q6_K *)((const uint8_t *)pVx + pBx);
    const block_q8_K *GGML_RESTRICT vVy1 = (const block_q8_K *)((const uint8_t *)pVy + pBy);

    switch (cVectorLength) {
    case 128: {
      const svbool_t cPg128All = svptrue_pat_b8(SV_ALL);
      for (int vI = 0; vI < cNb; ++vI) {
        const uint8_t *GGML_RESTRICT vQl0 = vVx0[vI].ql;
        const uint8_t *GGML_RESTRICT vQh0 = vVx0[vI].qh;
        const uint8_t *GGML_RESTRICT vQl1 = vVx1[vI].ql;
        const uint8_t *GGML_RESTRICT vQh1 = vVx1[vI].qh;
        const int8_t *GGML_RESTRICT vQ80 = vVy0[vI].qs;
        const int8_t *GGML_RESTRICT vQ81 = vVy1[vI].qs;

        const int8_t *GGML_RESTRICT vScale0 = vVx0[vI].scales;
        const int8_t *GGML_RESTRICT vScale1 = vVx1[vI].scales;

        svfloat32_t vVyD = svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d));
        svfloat32_t vVxD =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].d)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].d)));
        svfloat32_t vSvsuperBlockScales = svmul_f32_x(cPg128All, vVyD, vVxD);
        // process q8sum summation 128 bit route
        const svint16_t cQ8sums01 = svld1_s16(cPg128All, vVy0[vI].bsums);
        const svint16_t cQ8sums02 = svld1_s16(cPg128All, vVy0[vI].bsums + 8);
        const svint16_t cQ8sums11 = svld1_s16(cPg128All, vVy1[vI].bsums);
        const svint16_t cQ8sums12 = svld1_s16(cPg128All, vVy1[vI].bsums + 8);
        const svint64x2_t cQ6scales0Tmp = svld2_s64(cPg128All, (const int64_t *)vScale0);
        const svint16_t cQ6scales01 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(cQ6scales0Tmp, 0)));
        const svint16_t cQ6scales02 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(cQ6scales0Tmp, 1)));
        const svint64x2_t cQ6scales1Tmp = svld2_s64(cPg128All, (const int64_t *)vScale1);
        const svint16_t cQ6scales11 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(cQ6scales1Tmp, 0)));
        const svint16_t cQ6scales12 = svunpklo_s16(svreinterpret_s8_s64(svget2_s64(cQ6scales1Tmp, 1)));
        const svint64_t cProd = svdup_n_s64(0);

        svint32_t vIsumTmp1 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(cProd, cQ8sums01, cQ6scales01), cQ8sums02, cQ6scales02));
        svint32_t vIsumTmp2 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(cProd, cQ8sums01, cQ6scales11), cQ8sums02, cQ6scales12));
        svint32_t vIsumTmp3 = svtrn1_s32(vIsumTmp1, vIsumTmp2);
        svint32_t vIsumTmp4 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(cProd, cQ8sums11, cQ6scales01), cQ8sums12, cQ6scales02));
        svint32_t vIsumTmp5 =
            svreinterpret_s32_s64(svdot_s64(svdot_s64(cProd, cQ8sums11, cQ6scales11), cQ8sums12, cQ6scales12));
        svint32_t vIsumTmp6 = svtrn1_s32(vIsumTmp4, vIsumTmp5);
        svint32_t vIsumTmp7 =
            svreinterpret_s32_s64(svtrn2_s64(svreinterpret_s64_s32(vIsumTmp3), svreinterpret_s64_s32(vIsumTmp6)));
        svint32_t vIsumTmp8 =
            svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(vIsumTmp3), svreinterpret_s64_s32(vIsumTmp6)));
        svint32_t vSvisumMins = svadd_s32_x(cPg128All, vIsumTmp7, vIsumTmp8);

        // process mmla
        svint8_t vL0, vL1, vR0, vR1;
        svint32_t vIsumTmp = svdup_n_s32(0);
        for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
          for (int vK = 0; vK < 8; ++vK) {
            svuint8_t vQhbits0 = svld1_u8(cPg128All, vQh0 + 16 * (vK % 2));
            svuint8_t vQhbits1 = svld1_u8(cPg128All, vQh1 + 16 * (vK % 2));
            svuint8_t vQ6bits0 = svld1_u8(cPg128All, vQl0 + 16 * (vK % 4));
            svuint8_t vQ6bits1 = svld1_u8(cPg128All, vQl1 + 16 * (vK % 4));
            const int cQlPos = (vK / 4) * 4;
            svuint8_t vQ6bytes0Lo =
                (cQlPos < 4) ? svand_n_u8_x(cPg128All, vQ6bits0, 0xf) : svlsr_n_u8_x(cPg128All, vQ6bits0, 4);
            svuint8_t vQ6bytes1Lo =
                (cQlPos < 4) ? svand_n_u8_x(cPg128All, vQ6bits1, 0xf) : svlsr_n_u8_x(cPg128All, vQ6bits1, 4);
            const int cQhPos = (vK / 2) * 2;
            svuint8_t vQ6bytes0Hi = svand_n_u8_x(cPg128All, vQhbits0, 0x3 << cQhPos);
            svuint8_t vQ6bytes1Hi = svand_n_u8_x(cPg128All, vQhbits1, 0x3 << cQhPos);
            svint8_t vQ6bytes0, vQ6bytes1;
            if (cQhPos <= 4) {
              vQ6bytes0 = svreinterpret_s8_u8(svmla_n_u8_x(cPg128All, vQ6bytes0Lo, vQ6bytes0Hi, 1 << (4 - cQhPos)));
              vQ6bytes1 = svreinterpret_s8_u8(svmla_n_u8_x(cPg128All, vQ6bytes1Lo, vQ6bytes1Hi, 1 << (4 - cQhPos)));
            } else {
              vQ6bytes0 = svreinterpret_s8_u8(
                  svorr_u8_x(cPg128All, vQ6bytes0Lo, svlsr_n_u8_x(cPg128All, vQ6bytes0Hi, (cQhPos - 4))));
              vQ6bytes1 = svreinterpret_s8_u8(
                  svorr_u8_x(cPg128All, vQ6bytes1Lo, svlsr_n_u8_x(cPg128All, vQ6bytes1Hi, (cQhPos - 4))));
            }
            svint8_t vQ8bytes0 = svld1_s8(cPg128All, vQ80 + 16 * (vK % 8));
            svint8_t vQ8bytes1 = svld1_s8(cPg128All, vQ81 + 16 * (vK % 8));
            vL0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ6bytes0), svreinterpret_s64_s8(vQ6bytes1)));
            vL1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ6bytes0), svreinterpret_s64_s8(vQ6bytes1)));
            vR0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
            vR1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
            svint32_t vSvscale = svzip1_s32(svdup_n_s32(vScale0[vK]), svdup_n_s32(vScale1[vK]));
            vIsumTmp =
                svmla_s32_x(cPg128All, vIsumTmp, svmmla_s32(svmmla_s32(svdup_n_s32(0), vR0, vL0), vR1, vL1), vSvscale);
          }
          vQh0 += 32;
          vQh1 += 32;
          vQl0 += 64;
          vQl1 += 64;
          vQ80 += 128;
          vQ81 += 128;
          vScale0 += 8;
          vScale1 += 8;
        }
        vSum = svmla_f32_x(cPg128All, vSum,
                           svcvt_f32_x(cPg128All, svmla_s32_x(cPg128All, vIsumTmp, vSvisumMins, svdup_n_s32(-32))),
                           vSvsuperBlockScales);
      }
    } // end of case 128
    break;
    case 256:
    case 512: {
      const svbool_t cPg256All = svptrue_pat_b8(SV_ALL);
      const svbool_t cPg324 = svptrue_pat_b32(SV_VL4);
      for (int vI = 0; vI < cNb; ++vI) {
        const uint8_t *GGML_RESTRICT vQl0 = vVx0[vI].ql;
        const uint8_t *GGML_RESTRICT vQh0 = vVx0[vI].qh;
        const uint8_t *GGML_RESTRICT vQl1 = vVx1[vI].ql;
        const uint8_t *GGML_RESTRICT vQh1 = vVx1[vI].qh;
        const int8_t *GGML_RESTRICT vQ80 = vVy0[vI].qs;
        const int8_t *GGML_RESTRICT vQ81 = vVy1[vI].qs;

        const int8_t *GGML_RESTRICT vScale0 = vVx0[vI].scales;
        const int8_t *GGML_RESTRICT vScale1 = vVx1[vI].scales;
        svfloat32_t vVxD =
            svzip1_f32(svdup_n_f32(GGML_FP16_TO_FP32(vVx0[vI].d)), svdup_n_f32(GGML_FP16_TO_FP32(vVx1[vI].d)));
        svfloat64_t vVyDTmp = svreinterpret_f64_f32(svuzp1_f32(svdup_n_f32(vVy0[vI].d), svdup_n_f32(vVy1[vI].d)));
        svfloat32_t vVyD = svreinterpret_f32_f64(svuzp1_f64(vVyDTmp, vVyDTmp));
        svfloat32_t vSvsuperBlockScales = svmul_f32_x(cPg324, vVyD, vVxD);
        // process q8sum summation 256 bit route
        const svint16_t cQ8sums0 = svld1_s16(cPg256All, vVy0[vI].bsums);
        const svint16_t cQ8sums1 = svld1_s16(cPg256All, vVy1[vI].bsums);
        const svint16_t cQ6scales0 = svunpklo_s16(svld1_s8(cPg256All, vScale0));
        const svint16_t cQ6scales1 = svunpklo_s16(svld1_s8(cPg256All, vScale1));
        const svint64_t cProd = svdup_n_s64(0);
        svint32_t vIsumTmp1 = svreinterpret_s32_s64(svdot_s64(cProd, cQ8sums0, cQ6scales0));
        svint32_t vIsumTmp2 = svreinterpret_s32_s64(svdot_s64(cProd, cQ8sums0, cQ6scales1));
        svint32_t vIsumTmp3 = svreinterpret_s32_s64(svdot_s64(cProd, cQ8sums1, cQ6scales0));
        svint32_t vIsumTmp4 = svreinterpret_s32_s64(svdot_s64(cProd, cQ8sums1, cQ6scales1));
        svint32_t vIsumTmp5 = svtrn1_s32(vIsumTmp1, vIsumTmp2);
        svint32_t vIsumTmp6 = svtrn1_s32(vIsumTmp3, vIsumTmp4);
        svint32_t vIsumTmp7 =
            svreinterpret_s32_s64(svtrn2_s64(svreinterpret_s64_s32(vIsumTmp5), svreinterpret_s64_s32(vIsumTmp6)));
        svint32_t vIsumTmp8 =
            svreinterpret_s32_s64(svtrn1_s64(svreinterpret_s64_s32(vIsumTmp5), svreinterpret_s64_s32(vIsumTmp6)));
        svint32_t vIsumTmp9 = svadd_s32_x(cPg256All, vIsumTmp7, vIsumTmp8);
        svint32_t vIsumTmp10 =
            svreinterpret_s32_u8(svext_u8(svreinterpret_u8_s32(vIsumTmp9), svreinterpret_u8_s32(vIsumTmp9), 16));
        svint32_t vSvisumMins = svadd_s32_z(cPg324, vIsumTmp9, vIsumTmp10);

        // process mmla
        svint8_t vL0, vL1, vR0, vR1;
        svint32_t vIsumTmp = svdup_n_s32(0);
        for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
          for (int vK = 0; vK < 8; vK += 2) { // process 2 block
            svuint8_t vQhbits0 = svld1_u8(cPg256All, vQh0);
            svuint8_t vQhbits1 = svld1_u8(cPg256All, vQh1);
            svuint8_t vQ6bits0 = svld1_u8(cPg256All, vQl0 + 32 * ((vK % 4) / 2));
            svuint8_t vQ6bits1 = svld1_u8(cPg256All, vQl1 + 32 * ((vK % 4) / 2));
            const int cQlPos = (vK / 4) * 4;
            svuint8_t vQ6bytes0Lo =
                (cQlPos < 4) ? svand_n_u8_x(cPg256All, vQ6bits0, 0xf) : svlsr_n_u8_x(cPg256All, vQ6bits0, 4);
            svuint8_t vQ6bytes1Lo =
                (cQlPos < 4) ? svand_n_u8_x(cPg256All, vQ6bits1, 0xf) : svlsr_n_u8_x(cPg256All, vQ6bits1, 4);
            const int cQhPos = (vK / 2) * 2;
            svuint8_t vQ6bytes0Hi = svand_n_u8_x(cPg256All, vQhbits0, 0x3 << cQhPos);
            svuint8_t vQ6bytes1Hi = svand_n_u8_x(cPg256All, vQhbits1, 0x3 << cQhPos);
            svint8_t vQ6bytes0, vQ6bytes1;
            if (cQhPos <= 4) {
              vQ6bytes0 = svreinterpret_s8_u8(svmla_n_u8_x(cPg256All, vQ6bytes0Lo, vQ6bytes0Hi, 1 << (4 - cQhPos)));
              vQ6bytes1 = svreinterpret_s8_u8(svmla_n_u8_x(cPg256All, vQ6bytes1Lo, vQ6bytes1Hi, 1 << (4 - cQhPos)));
            } else {
              vQ6bytes0 = svreinterpret_s8_u8(
                  svorr_u8_x(cPg256All, vQ6bytes0Lo, svlsr_n_u8_x(cPg256All, vQ6bytes0Hi, (cQhPos - 4))));
              vQ6bytes1 = svreinterpret_s8_u8(
                  svorr_u8_x(cPg256All, vQ6bytes1Lo, svlsr_n_u8_x(cPg256All, vQ6bytes1Hi, (cQhPos - 4))));
            }
            svint8_t vQ8bytes0 = svld1_s8(cPg256All, vQ80 + 32 * (vK / 2));
            svint8_t vQ8bytes1 = svld1_s8(cPg256All, vQ81 + 32 * (vK / 2));
            vL0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ6bytes0), svreinterpret_s64_s8(vQ6bytes1)));
            vL1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ6bytes0), svreinterpret_s64_s8(vQ6bytes1)));
            vR0 = svreinterpret_s8_s64(svzip1_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
            vR1 = svreinterpret_s8_s64(svzip2_s64(svreinterpret_s64_s8(vQ8bytes0), svreinterpret_s64_s8(vQ8bytes1)));
            svint32_t vSvscale0 = svzip1_s32(svdup_n_s32(vScale0[vK]), svdup_n_s32(vScale1[vK]));
            svint32_t vSvscale1 = svzip1_s32(svdup_n_s32(vScale0[vK + 1]), svdup_n_s32(vScale1[vK + 1]));
            vIsumTmp = svmla_s32_x(cPg256All, vIsumTmp, svmmla_s32(svdup_n_s32(0), vR0, vL0), vSvscale0);
            vIsumTmp = svmla_s32_x(cPg256All, vIsumTmp, svmmla_s32(svdup_n_s32(0), vR1, vL1), vSvscale1);
          }
          vQh0 += 32;
          vQh1 += 32;
          vQl0 += 64;
          vQl1 += 64;
          vQ80 += 128;
          vQ81 += 128;
          vScale0 += 8;
          vScale1 += 8;
        } // end of for
        svint32_t vSwapIsumTmp = svext_s32(vIsumTmp, vIsumTmp, 4);
        vIsumTmp = svadd_s32_x(cPg324, vIsumTmp, vSwapIsumTmp);
        vSum =
            svmla_f32_x(cPg324, vSum, svcvt_f32_x(cPg324, svmla_s32_x(cPg324, vIsumTmp, vSvisumMins, svdup_n_s32(-32))),
                        vSvsuperBlockScales);
      }
    } // end of case 256
    break;
    default:
      assert(false && "Unsupported vector length");
      break;
    } // end of switch

    svst1_f32(cPg322, pS, vSum);
    svst1_f32(cPg322, pS + pBs, svreinterpret_f32_u8(svext_u8(svreinterpret_u8_f32(vSum), svdup_n_u8(0), 8)));

    return;
  }
#elif defined(__ARM_FEATURE_MATMUL_INT8)
  if (pNrc == 2) {
    const block_q6_K *GGML_RESTRICT x0 = vX;
    const block_q6_K *GGML_RESTRICT x1 = (const block_q6_K *)((const uint8_t *)pVx + pBx);
    const block_q8_K *GGML_RESTRICT y0 = vY;
    const block_q8_K *GGML_RESTRICT y1 = (const block_q8_K *)((const uint8_t *)pVy + pBy);

    float32x4_t vfsum = vdupq_n_f32(0.0f);

    for (int vI = 0; vI < cNb; ++vI, ++x0, ++x1, ++y0, ++y1) {
      const uint8_t *GGML_RESTRICT vQl0 = x0->ql;
      const uint8_t *GGML_RESTRICT vQl1 = x1->ql;
      const uint8_t *GGML_RESTRICT vQh0 = x0->qh;
      const uint8_t *GGML_RESTRICT vQh1 = x1->qh;
      const int8_t *GGML_RESTRICT qy0 = y0->qs;
      const int8_t *GGML_RESTRICT qy1 = y1->qs;

      const uint8x16_t vMone = vdupq_n_u8(0x30);
      const uint8x16_t vM4b = vdupq_n_u8(0x0f);

      int32x4_t visum = vdupq_n_s32(0);

      // process 8 blocks per iteration, totally 16 blocks
      for (int vJ = 0; vJ < 2; ++vJ, vQh0 += 32, vQl0 += 64, vQh1 += 32, vQl1 += 64) {
        int8x16_t vVx0[8], vVx1[8];

        // de-quantize vx0[8]
        {
          const uint8x16x2_t qh_bits = vld1q_u8_x2(vQh0);
          const uint8x16x4_t ql_bits = vld1q_u8_x4(vQl0);

          uint8x16_t q6h_0 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[0], 4));
          uint8x16_t vQ6h1 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[1], 4));
          uint8x16_t vQ6h2 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[0], 2));
          uint8x16_t vQ6h3 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[1], 2));

          vVx0[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], vM4b), q6h_0));
          vVx0[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], vM4b), vQ6h1));
          vVx0[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], vM4b), vQ6h2));
          vVx0[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], vM4b), vQ6h3));

          q6h_0 = vandq_u8(vMone, qh_bits.val[0]);
          vQ6h1 = vandq_u8(vMone, qh_bits.val[1]);
          vQ6h2 = vandq_u8(vMone, vshrq_n_u8(qh_bits.val[0], 2));
          vQ6h3 = vandq_u8(vMone, vshrq_n_u8(qh_bits.val[1], 2));

          vVx0[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
          vVx0[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), vQ6h1));
          vVx0[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), vQ6h2));
          vVx0[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), vQ6h3));
        }

        // de-quantize vx1[8]
        {
          const uint8x16x2_t qh_bits = vld1q_u8_x2(vQh1);
          const uint8x16x4_t ql_bits = vld1q_u8_x4(vQl1);

          uint8x16_t q6h_0 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[0], 4));
          uint8x16_t vQ6h1 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[1], 4));
          uint8x16_t vQ6h2 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[0], 2));
          uint8x16_t vQ6h3 = vandq_u8(vMone, vshlq_n_u8(qh_bits.val[1], 2));

          vVx1[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], vM4b), q6h_0));
          vVx1[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], vM4b), vQ6h1));
          vVx1[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], vM4b), vQ6h2));
          vVx1[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], vM4b), vQ6h3));

          q6h_0 = vandq_u8(vMone, qh_bits.val[0]);
          vQ6h1 = vandq_u8(vMone, qh_bits.val[1]);
          vQ6h2 = vandq_u8(vMone, vshrq_n_u8(qh_bits.val[0], 2));
          vQ6h3 = vandq_u8(vMone, vshrq_n_u8(qh_bits.val[1], 2));

          vVx1[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
          vVx1[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), vQ6h1));
          vVx1[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), vQ6h2));
          vVx1[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), vQ6h3));
        }

        // process 16 elements (one block with same scale) per iteration
        // - vx = concat(ql, qh) - 32
        // - r1,r2,r3,r4 = smmla(vx, vy)
        for (int vK = 0; vK < 8; ++vK) {
          const int blk = vJ * 8 + vK;

          const int8x16_t vVy0 = vld1q_s8(qy0);
          const int8x16_t vVy1 = vld1q_s8(qy1);
          qy0 += 16;
          qy1 += 16;

          const int32x4_t block_scale = {
              x0->scales[blk],
              x0->scales[blk],
              x1->scales[blk],
              x1->scales[blk],
          };

          // calculate four results at once with outer product
          const int8x16_t vx_l =
              vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vVx0[vK]), vreinterpretq_s64_s8(vVx1[vK])));
          const int8x16_t vx_h =
              vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vVx0[vK]), vreinterpretq_s64_s8(vVx1[vK])));
          const int8x16_t vy_l =
              vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vVy0), vreinterpretq_s64_s8(vVy1)));
          const int8x16_t vy_h =
              vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vVy0), vreinterpretq_s64_s8(vVy1)));
          int32x4_t vr = vdupq_n_s32(0);
          vr = vmmlaq_s32(vr, vx_l, vy_l);
          vr = vmmlaq_s32(vr, vx_h, vy_h);

          // apply block scale, will NOT overflow
          // block_scale * sum_256(int6*int8) <= 2^(8+8+6+8) = 30 bits
          visum = vmlaq_s32(visum, vr, block_scale);
        }
      }

      // adjust bias, apply superblock scale
      {
        int32_t bias[4];
        // NEON doesn't support int16 dot product, fallback to separated mul and add
        const int16x8x2_t q8sums0 = vld1q_s16_x2(y0->bsums);
        const int16x8x2_t q8sums1 = vld1q_s16_x2(y1->bsums);

        int8x16_t scales_s8 = vld1q_s8(x0->scales);
        const int16x8x2_t q6scales0 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};
        scales_s8 = vld1q_s8(x1->scales);
        const int16x8x2_t q6scales1 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};

        int32x4_t cProd;
        cProd = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[0]), vget_low_s16(q6scales0.val[0])),
                                    vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales0.val[0]))),
                          vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[1]), vget_low_s16(q6scales0.val[1])),
                                    vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales0.val[1]))));
        bias[0] = vaddvq_s32(cProd);
        cProd = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[0]), vget_low_s16(q6scales0.val[0])),
                                    vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales0.val[0]))),
                          vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[1]), vget_low_s16(q6scales0.val[1])),
                                    vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales0.val[1]))));
        bias[1] = vaddvq_s32(cProd);
        cProd = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[0]), vget_low_s16(q6scales1.val[0])),
                                    vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales1.val[0]))),
                          vaddq_s32(vmull_s16(vget_low_s16(q8sums0.val[1]), vget_low_s16(q6scales1.val[1])),
                                    vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales1.val[1]))));
        bias[2] = vaddvq_s32(cProd);
        cProd = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[0]), vget_low_s16(q6scales1.val[0])),
                                    vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales1.val[0]))),
                          vaddq_s32(vmull_s16(vget_low_s16(q8sums1.val[1]), vget_low_s16(q6scales1.val[1])),
                                    vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales1.val[1]))));
        bias[3] = vaddvq_s32(cProd);

        const int32x4_t vibias = vmulq_n_s32(vld1q_s32(bias), 32);

        const float32x4_t superblock_scale = {
            GGML_CPU_FP16_TO_FP32(x0->d) * y0->d,
            GGML_CPU_FP16_TO_FP32(x0->d) * y1->d,
            GGML_CPU_FP16_TO_FP32(x1->d) * y0->d,
            GGML_CPU_FP16_TO_FP32(x1->d) * y1->d,
        };

        visum = vsubq_s32(visum, vibias);
        vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
      }
    }

    // vfsum = ABCD -> ACBD
    // AC -> s, BD -> (s+bs)
    vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
    vst1_f32(pS, vget_low_f32(vfsum));
    vst1_f32(pS + pBs, vget_high_f32(vfsum));

    return;
  }
#endif

#ifdef __ARM_FEATURE_SVE
  float vSum = 0;
  svuint8_t vM4b = svdup_n_u8(0xf);
  svint32_t vVzero = svdup_n_s32(0);
  svuint8_t vMone = svdup_n_u8(0x30);
  svint8_t vQ6bytes1, vQ6bytes2, vQ6bytes3, vQ6bytes4;
  svuint8_t vQ6h1, vQ6h2, vQ6h3, vQ6h4;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cDAll = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    const svbool_t cPg168 = svptrue_pat_b16(SV_VL8);
    const svint16_t cQ8sums1 = svld1_s16(cPg168, vY[vI].bsums);
    const svint16_t cQ8sums2 = svld1_s16(cPg168, vY[vI].bsums + 8);
    const svint16_t cQ6scales1 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), vScale));
    const svint16_t cQ6scales2 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), vScale + 8));
    const svint64_t cProd = svdup_n_s64(0);
    int32_t vIsumMins = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(cProd, cQ8sums1, cQ6scales1),
                                                              svdot_s64(cProd, cQ8sums2, cQ6scales2)));
    int32_t vIsum = 0;

    switch (cVectorLength) {
    case 128: {
      const svbool_t cPg324 = svptrue_pat_b32(SV_VL4);
      const svbool_t cPg816 = svptrue_pat_b8(SV_VL16);
      svint32_t vIsumTmp = svdup_n_s32(0);
      for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
        svuint8_t vQhbits1 = svld1_u8(cPg816, vQh);
        svuint8_t vQhbits2 = svld1_u8(cPg816, vQh + 16);
        vQh += 32;
        svuint8_t vQ6bits1 = svld1_u8(cPg816, vQ6);
        svuint8_t vQ6bits2 = svld1_u8(cPg816, vQ6 + 16);
        svuint8_t vQ6bits3 = svld1_u8(cPg816, vQ6 + 32);
        svuint8_t vQ6bits4 = svld1_u8(cPg816, vQ6 + 48);
        vQ6 += 64;
        svint8_t vQ8bytes1 = svld1_s8(cPg816, vQ8);
        svint8_t vQ8bytes2 = svld1_s8(cPg816, vQ8 + 16);
        svint8_t vQ8bytes3 = svld1_s8(cPg816, vQ8 + 32);
        svint8_t vQ8bytes4 = svld1_s8(cPg816, vQ8 + 48);
        vQ8 += 64;

        vQ6h1 = svand_u8_x(cPg168, vMone, svlsl_n_u8_x(cPg168, vQhbits1, 4));
        vQ6h2 = svand_u8_x(cPg168, vMone, svlsl_n_u8_x(cPg168, vQhbits2, 4));
        vQ6h3 = svand_u8_x(cPg168, vMone, svlsl_n_u8_x(cPg168, vQhbits1, 2));
        vQ6h4 = svand_u8_x(cPg168, vMone, svlsl_n_u8_x(cPg168, vQhbits2, 2));
        vQ6bytes1 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svand_u8_x(cPg816, vQ6bits1, vM4b), vQ6h1));
        vQ6bytes2 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svand_u8_x(cPg816, vQ6bits2, vM4b), vQ6h2));
        vQ6bytes3 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svand_u8_x(cPg816, vQ6bits3, vM4b), vQ6h3));
        vQ6bytes4 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svand_u8_x(cPg816, vQ6bits4, vM4b), vQ6h4));
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes1, vQ8bytes1), vScale[0]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes2, vQ8bytes2), vScale[1]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes3, vQ8bytes3), vScale[2]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes4, vQ8bytes4), vScale[3]);

        vScale += 4;
        vQ8bytes1 = svld1_s8(cPg816, vQ8);
        vQ8bytes2 = svld1_s8(cPg816, vQ8 + 16);
        vQ8bytes3 = svld1_s8(cPg816, vQ8 + 32);
        vQ8bytes4 = svld1_s8(cPg816, vQ8 + 48);
        vQ8 += 64;

        vQ6h1 = svand_u8_x(cPg168, vMone, vQhbits1);
        vQ6h2 = svand_u8_x(cPg168, vMone, vQhbits2);
        vQ6h3 = svand_u8_x(cPg168, vMone, svlsr_n_u8_x(cPg168, vQhbits1, 2));
        vQ6h4 = svand_u8_x(cPg168, vMone, svlsr_n_u8_x(cPg168, vQhbits2, 2));
        vQ6bytes1 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svlsr_n_u8_x(cPg816, vQ6bits1, 4), vQ6h1));
        vQ6bytes2 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svlsr_n_u8_x(cPg816, vQ6bits2, 4), vQ6h2));
        vQ6bytes3 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svlsr_n_u8_x(cPg816, vQ6bits3, 4), vQ6h3));
        vQ6bytes4 = svreinterpret_s8_u8(svorr_u8_x(cPg816, svlsr_n_u8_x(cPg816, vQ6bits4, 4), vQ6h4));
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes1, vQ8bytes1), vScale[0]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes2, vQ8bytes2), vScale[1]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes3, vQ8bytes3), vScale[2]);
        vIsumTmp = svmla_n_s32_x(cPg324, vIsumTmp, svdot_s32(vVzero, vQ6bytes4, vQ8bytes4), vScale[3]);
        vScale += 4;
      }
      vIsum += svaddv_s32(cPg324, vIsumTmp);
      vSum += cDAll * vY[vI].d * (vIsum - 32 * vIsumMins);
    } break;
    case 256:
    case 512: {
      const svbool_t cPg82 = svptrue_pat_b8(SV_VL2);
      const svbool_t cPg328 = svptrue_pat_b32(SV_VL8);
      const svbool_t cPg832 = svptrue_pat_b8(SV_VL32);
      svint32_t vIsumTmp = svdup_n_s32(0);
      for (int vJ = 0; vJ < QK_K / 128; vJ++) {
        svuint8_t vQhbits1 = svld1_u8(cPg832, vQh);
        vQh += 32;
        svuint8_t vQ6bits1 = svld1_u8(cPg832, vQ6);
        svuint8_t vQ6bits2 = svld1_u8(cPg832, vQ6 + 32);
        vQ6 += 64;
        svint8_t vQ8bytes1 = svld1_s8(cPg832, vQ8);
        svint8_t vQ8bytes2 = svld1_s8(cPg832, vQ8 + 32);
        svint8_t vQ8bytes3 = svld1_s8(cPg832, vQ8 + 64);
        svint8_t vQ8bytes4 = svld1_s8(cPg832, vQ8 + 96);
        vQ8 += 128;
        vQ6h1 = svand_u8_x(cPg832, vMone, svlsl_n_u8_x(cPg832, vQhbits1, 4));
        vQ6h2 = svand_u8_x(cPg832, vMone, svlsl_n_u8_x(cPg832, vQhbits1, 2));
        vQ6h3 = svand_u8_x(cPg832, vMone, vQhbits1);
        vQ6h4 = svand_u8_x(cPg832, vMone, svlsr_n_u8_x(cPg832, vQhbits1, 2));
        vQ6bytes1 = svreinterpret_s8_u8(svorr_u8_x(cPg832, svand_u8_x(cPg832, vQ6bits1, vM4b), vQ6h1));
        vQ6bytes2 = svreinterpret_s8_u8(svorr_u8_x(cPg832, svand_u8_x(cPg832, vQ6bits2, vM4b), vQ6h2));
        vQ6bytes3 = svreinterpret_s8_u8(svorr_u8_x(cPg832, svlsr_n_u8_x(cPg832, vQ6bits1, 4), vQ6h3));
        vQ6bytes4 = svreinterpret_s8_u8(svorr_u8_x(cPg832, svlsr_n_u8_x(cPg832, vQ6bits2, 4), vQ6h4));

        svint8_t vScaleLane1Tmp = svld1_s8(cPg82, vScale);
        vScaleLane1Tmp = svzip1_s8(vScaleLane1Tmp, vScaleLane1Tmp);
        vScaleLane1Tmp = svzip1_s8(vScaleLane1Tmp, vScaleLane1Tmp);
        svint8_t vScaleLane2Tmp = svld1_s8(cPg82, vScale + 2);
        vScaleLane2Tmp = svzip1_s8(vScaleLane2Tmp, vScaleLane2Tmp);
        vScaleLane2Tmp = svzip1_s8(vScaleLane2Tmp, vScaleLane2Tmp);
        svint8_t vScaleLane3Tmp = svld1_s8(cPg82, vScale + 4);
        vScaleLane3Tmp = svzip1_s8(vScaleLane3Tmp, vScaleLane3Tmp);
        vScaleLane3Tmp = svzip1_s8(vScaleLane3Tmp, vScaleLane3Tmp);
        svint8_t vScaleLane4Tmp = svld1_s8(cPg82, vScale + 6);
        vScaleLane4Tmp = svzip1_s8(vScaleLane4Tmp, vScaleLane4Tmp);
        vScaleLane4Tmp = svzip1_s8(vScaleLane4Tmp, vScaleLane4Tmp);
        svint32_t vScaleLane1 = svunpklo_s32(svunpklo_s16(vScaleLane1Tmp));
        svint32_t vScaleLane2 = svunpklo_s32(svunpklo_s16(vScaleLane2Tmp));
        svint32_t vScaleLane3 = svunpklo_s32(svunpklo_s16(vScaleLane3Tmp));
        svint32_t vScaleLane4 = svunpklo_s32(svunpklo_s16(vScaleLane4Tmp));

        vIsumTmp = svmla_s32_x(cPg328, vIsumTmp, svdot_s32(vVzero, vQ6bytes1, vQ8bytes1), vScaleLane1);
        vIsumTmp = svmla_s32_x(cPg328, vIsumTmp, svdot_s32(vVzero, vQ6bytes2, vQ8bytes2), vScaleLane2);
        vIsumTmp = svmla_s32_x(cPg328, vIsumTmp, svdot_s32(vVzero, vQ6bytes3, vQ8bytes3), vScaleLane3);
        vIsumTmp = svmla_s32_x(cPg328, vIsumTmp, svdot_s32(vVzero, vQ6bytes4, vQ8bytes4), vScaleLane4);
        vScale += 8;
      }
      vIsum += svaddv_s32(cPg328, vIsumTmp);
      vSum += cDAll * vY[vI].d * (vIsum - 32 * vIsumMins);
    } break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }
  }

  *pS = vSum;

#elif __ARM_NEON
  float vSum = 0;

  const uint8x16_t vM4b = vdupq_n_u8(0xF);
  const int32x4_t vVzero = vdupq_n_s32(0);
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t vMone = vdupq_n_u8(3);

  ggml_int8x16x4_t q6bytes;
  ggml_uint8x16x4_t q6h;

  for (int vI = 0; vI < cNb; ++vI) {

    const float cDAll = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(vY[vI].bsums);
    const int8x16_t scales = vld1q_s8(vScale);
    const ggml_int16x8x2_t q6scales = {{vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

    const int32x4_t cProd =
        vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]), vget_low_s16(q6scales.val[0])),
                            vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                  vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]), vget_low_s16(q6scales.val[1])),
                            vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
    int32_t vIsumMins = vaddvq_s32(cProd);

    int32_t vIsum = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(vQh);
      vQh += 32;
      ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(vQ6);
      vQ6 += 64;
      ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(vMone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(vMone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], vM4b), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], vM4b), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], vM4b), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], vM4b), q6h.val[3]));

      vIsum += vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[0], q8bytes.val[0])) * vScale[0] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[1], q8bytes.val[1])) * vScale[1] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[2], q8bytes.val[2])) * vScale[2] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[3], q8bytes.val[3])) * vScale[3];

      vScale += 4;

      q8bytes = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(vMone, shifted), 4);

      // q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
      // q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
      // q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
      // q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
      q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

      vIsum += vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[0], q8bytes.val[0])) * vScale[0] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[1], q8bytes.val[1])) * vScale[1] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[2], q8bytes.val[2])) * vScale[2] +
               vaddvq_s32(ggml_vdotq_s32(vVzero, q6bytes.val[3], q8bytes.val[3])) * vScale[3];
      vScale += 4;
    }
    // sum += isum * d_all * y[i].d;
    vSum += cDAll * vY[vI].d * (vIsum - 32 * vIsumMins);
  }
  *pS = vSum;
#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined(__ARM_NEON)
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

#if defined(__ARM_NEON)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[4];
  const uint8_t *vAux8 = (const uint8_t *)vAux32;

  ggml_int8x16x4_t vQ2u;
  ggml_int8x16x4_t vQ2s;
  ggml_int8x16x4_t vQ8b;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    float vSumf1 = 0, vSumf2 = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;
      memcpy(vAux32, vQ2, 4 * sizeof(uint32_t));
      vQ2 += 8;
      vQ2u.val[0] =
          vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + vAux8[0])), vld1_s8((const void *)(iq2xxs_grid + vAux8[1])));
      vQ2u.val[1] =
          vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + vAux8[2])), vld1_s8((const void *)(iq2xxs_grid + vAux8[3])));
      vQ2u.val[2] =
          vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + vAux8[8])), vld1_s8((const void *)(iq2xxs_grid + vAux8[9])));
      vQ2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + vAux8[10])),
                                vld1_s8((const void *)(iq2xxs_grid + vAux8[11])));
      vQ2s.val[0] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 0) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 7) & 127))));
      vQ2s.val[1] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 14) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 21) & 127))));
      vQ2s.val[2] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[3] >> 0) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[3] >> 7) & 127))));
      vQ2s.val[3] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[3] >> 14) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[3] >> 21) & 127))));
      vQ2u.val[0] = vmulq_s8(vQ2u.val[0], vQ2s.val[0]);
      vQ2u.val[1] = vmulq_s8(vQ2u.val[1], vQ2s.val[1]);
      vQ2u.val[2] = vmulq_s8(vQ2u.val[2], vQ2s.val[2]);
      vQ2u.val[3] = vmulq_s8(vQ2u.val[3], vQ2s.val[3]);
      const int32x4_t cP1 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[0], vQ8b.val[0]), vQ2u.val[1], vQ8b.val[1]);
      const int32x4_t cP2 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[2], vQ8b.val[2]), vQ2u.val[3], vQ8b.val[3]);
      vSumf1 += vaddvq_s32(cP1) * (0.5f + (vAux32[1] >> 28));
      vSumf2 += vaddvq_s32(cP2) * (0.5f + (vAux32[3] >> 28));
    }
    vSumf += cD * (vSumf1 + vSumf2);
  }
  *pS = 0.25f * vSumf;

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

#if defined(__ARM_NEON)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  ggml_int8x16x4_t vQ2u;
  ggml_int8x16x4_t vQ2s;
  ggml_int8x16x4_t vQ8b;

  int32x4x4_t vScales32;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    const uint8x8_t cScales8 = vld1_u8(vX[vI].scales);
    const uint8x8_t cScalesL = vand_u8(cScales8, vdup_n_u8(0xf));
    const uint8x8_t cScalesH = vshr_n_u8(cScales8, 4);
    uint8x16_t vScales = vcombine_u8(vzip1_u8(cScalesL, cScalesH), vzip2_u8(cScalesL, cScalesH));
    vScales = vaddq_u8(vshlq_n_u8(vScales, 1), vdupq_n_u8(1));
    const uint16x8_t cScales1 = vmovl_u8(vget_low_u8(vScales));
    const uint16x8_t cScales2 = vmovl_u8(vget_high_u8(vScales));
    vScales32.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(cScales1)));
    vScales32.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(cScales1)));
    vScales32.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(cScales2)));
    vScales32.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(cScales2)));
    int32x4_t vSumi = vdupq_n_s32(0);
    for (int vIb64 = 0; vIb64 < QK_K / 64; ++vIb64) {
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;
      vQ2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (vQ2[0] & 511))),
                                vld1_s8((const void *)(iq2xs_grid + (vQ2[1] & 511))));
      vQ2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (vQ2[2] & 511))),
                                vld1_s8((const void *)(iq2xs_grid + (vQ2[3] & 511))));
      vQ2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (vQ2[4] & 511))),
                                vld1_s8((const void *)(iq2xs_grid + (vQ2[5] & 511))));
      vQ2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (vQ2[6] & 511))),
                                vld1_s8((const void *)(iq2xs_grid + (vQ2[7] & 511))));
      vQ2s.val[0] = vcombine_s8(vld1_s8((const void *)(vSigns64 + (vQ2[0] >> 9))),
                                vld1_s8((const void *)(vSigns64 + (vQ2[1] >> 9))));
      vQ2s.val[1] = vcombine_s8(vld1_s8((const void *)(vSigns64 + (vQ2[2] >> 9))),
                                vld1_s8((const void *)(vSigns64 + (vQ2[3] >> 9))));
      vQ2s.val[2] = vcombine_s8(vld1_s8((const void *)(vSigns64 + (vQ2[4] >> 9))),
                                vld1_s8((const void *)(vSigns64 + (vQ2[5] >> 9))));
      vQ2s.val[3] = vcombine_s8(vld1_s8((const void *)(vSigns64 + (vQ2[6] >> 9))),
                                vld1_s8((const void *)(vSigns64 + (vQ2[7] >> 9))));
      vQ2u.val[0] = vmulq_s8(vQ2u.val[0], vQ2s.val[0]);
      vQ2u.val[1] = vmulq_s8(vQ2u.val[1], vQ2s.val[1]);
      vQ2u.val[2] = vmulq_s8(vQ2u.val[2], vQ2s.val[2]);
      vQ2u.val[3] = vmulq_s8(vQ2u.val[3], vQ2s.val[3]);
      const int32x4_t cP1 = ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[0], vQ8b.val[0]);
      const int32x4_t cP2 = ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[1], vQ8b.val[1]);
      const int32x4_t cP3 = ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[2], vQ8b.val[2]);
      const int32x4_t cP4 = ggml_vdotq_s32(vdupq_n_s32(0), vQ2u.val[3], vQ8b.val[3]);
      const int32x4_t cP = vpaddq_s32(vpaddq_s32(cP1, cP2), vpaddq_s32(cP3, cP4));
      vSumi = vmlaq_s32(vSumi, cP, vScales32.val[vIb64]);
      vQ2 += 8;
    }
    vSumf += cD * vaddvq_s32(vSumi);
  }
  *pS = 0.125f * vSumf;

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

#if defined(__ARM_NEON)

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[16] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const ggml_uint8x16x2_t cMask1 = ggml_vld1q_u8_x2(cKMask1);
  const uint8x16_t cMask2 = vld1q_u8(cKMask2);
  const uint8x16_t cM1 = vdupq_n_u8(1);
  const int32x4_t cVzero = vdupq_n_s32(0);

  uint8x16x2_t vVs;
  ggml_int8x16x4_t vQ2s;
  ggml_int8x16x4_t vQ8b;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].qs + QK_K / 8);
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    int vSumi1 = 0, vSumi2 = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;
      vQ2s.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (vQs[0] | ((vQh[vIb32 + 0] << 8) & 0x300)))),
                                vld1_s8((const int8_t *)(iq2s_grid + (vQs[1] | ((vQh[vIb32 + 0] << 6) & 0x300)))));
      vQ2s.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (vQs[2] | ((vQh[vIb32 + 0] << 4) & 0x300)))),
                                vld1_s8((const int8_t *)(iq2s_grid + (vQs[3] | ((vQh[vIb32 + 0] << 2) & 0x300)))));
      vQ2s.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (vQs[4] | ((vQh[vIb32 + 1] << 8) & 0x300)))),
                                vld1_s8((const int8_t *)(iq2s_grid + (vQs[5] | ((vQh[vIb32 + 1] << 6) & 0x300)))));
      vQ2s.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (vQs[6] | ((vQh[vIb32 + 1] << 4) & 0x300)))),
                                vld1_s8((const int8_t *)(iq2s_grid + (vQs[7] | ((vQh[vIb32 + 1] << 2) & 0x300)))));
      vQs += 8;

      vVs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(vSigns[0] | ((uint32_t)vSigns[1] << 16)));
      vVs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[1]), cMask2);
      vVs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[0]), cMask2);
      vVs.val[0] = vceqq_u8(vVs.val[0], cMask2);
      vVs.val[1] = vceqq_u8(vVs.val[1], cMask2);

      vQ2s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vVs.val[0], cM1)), vQ2s.val[0]);
      vQ2s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vVs.val[1], cM1)), vQ2s.val[1]);

      vVs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(vSigns[2] | ((uint32_t)vSigns[3] << 16)));
      vVs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[1]), cMask2);
      vVs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[0]), cMask2);
      vVs.val[0] = vceqq_u8(vVs.val[0], cMask2);
      vVs.val[1] = vceqq_u8(vVs.val[1], cMask2);

      vSigns += 4;

      vQ2s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vVs.val[0], cM1)), vQ2s.val[2]);
      vQ2s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vVs.val[1], cM1)), vQ2s.val[3]);

      const int32x4_t cP1 = ggml_vdotq_s32(cVzero, vQ2s.val[0], vQ8b.val[0]);
      const int32x4_t cP2 = ggml_vdotq_s32(cVzero, vQ2s.val[1], vQ8b.val[1]);
      const int32x4_t cP3 = ggml_vdotq_s32(cVzero, vQ2s.val[2], vQ8b.val[2]);
      const int32x4_t cP4 = ggml_vdotq_s32(cVzero, vQ2s.val[3], vQ8b.val[3]);

      vSumi1 += vaddvq_s32(cP1) * (1 + 2 * (vX[vI].scales[vIb32 + 0] & 0xf));
      vSumi2 += vaddvq_s32(cP2) * (1 + 2 * (vX[vI].scales[vIb32 + 0] >> 4));
      vSumi1 += vaddvq_s32(cP3) * (1 + 2 * (vX[vI].scales[vIb32 + 1] & 0xf));
      vSumi2 += vaddvq_s32(cP4) * (1 + 2 * (vX[vI].scales[vIb32 + 1] >> 4));
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }

  *pS = 0.125f * vSumf;

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

#if defined(__ARM_NEON)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t vAux32[2];

  ggml_int8x16x4_t vQ3s;
  ggml_int8x16x4_t vQ8b;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vGas = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    float vSumf1 = 0, vSumf2 = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;
      memcpy(vAux32, vGas, 2 * sizeof(uint32_t));
      vGas += 2 * sizeof(uint32_t);
      const uint32x4_t cAux32x40 =
          ggml_vld1q_u32(iq3xxs_grid[vQ3[0]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[3]]);
      const uint32x4_t cAux32x41 =
          ggml_vld1q_u32(iq3xxs_grid[vQ3[4]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[7]]);
      const uint32x4_t cAux32x42 =
          ggml_vld1q_u32(iq3xxs_grid[vQ3[8]], iq3xxs_grid[vQ3[9]], iq3xxs_grid[vQ3[10]], iq3xxs_grid[vQ3[11]]);
      const uint32x4_t cAux32x43 =
          ggml_vld1q_u32(iq3xxs_grid[vQ3[12]], iq3xxs_grid[vQ3[13]], iq3xxs_grid[vQ3[14]], iq3xxs_grid[vQ3[15]]);
      vQ3 += 16;
      vQ3s.val[0] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[0] >> 0) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[0] >> 7) & 127))));
      vQ3s.val[1] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[0] >> 14) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[0] >> 21) & 127))));
      vQ3s.val[2] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 0) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 7) & 127))));
      vQ3s.val[3] = vcombine_s8(vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 14) & 127))),
                                vld1_s8((const void *)(vSigns64 + ((vAux32[1] >> 21) & 127))));
      vQ3s.val[0] = vmulq_s8(vQ3s.val[0], vreinterpretq_s8_u32(cAux32x40));
      vQ3s.val[1] = vmulq_s8(vQ3s.val[1], vreinterpretq_s8_u32(cAux32x41));
      vQ3s.val[2] = vmulq_s8(vQ3s.val[2], vreinterpretq_s8_u32(cAux32x42));
      vQ3s.val[3] = vmulq_s8(vQ3s.val[3], vreinterpretq_s8_u32(cAux32x43));
      const int32x4_t cP1 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ3s.val[0], vQ8b.val[0]), vQ3s.val[1], vQ8b.val[1]);
      const int32x4_t cP2 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ3s.val[2], vQ8b.val[2]), vQ3s.val[3], vQ8b.val[3]);
      vSumf1 += vaddvq_s32(cP1) * (0.5f + (vAux32[0] >> 28));
      vSumf2 += vaddvq_s32(cP2) * (0.5f + (vAux32[1] >> 28));
    }
    vSumf += cD * (vSumf1 + vSumf2);
  }
  *pS = 0.5f * vSumf;

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

#if defined(__ARM_NEON)

  typedef union {
    uint16x8_t vVecIndex;
    uint16_t index[8];
  } vec_index_t;

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[16] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  static const int16_t cKShift[8] = {8, 7, 6, 5, 4, 3, 2, 1};

  const ggml_uint8x16x2_t cMask1 = ggml_vld1q_u8_x2(cKMask1);
  const uint8x16_t cMask2 = vld1q_u8(cKMask2);

  const int16x8_t cHshift = vld1q_s16(cKShift);
  const uint16x8_t cM256 = vdupq_n_u16(256);
  const uint8x16_t cM1 = vdupq_n_u8(1);

  uint8x16x2_t vVs;
  ggml_int8x16x4_t vQ3s;
  ggml_int8x16x4_t vQ8b;
  vec_index_t vIdx;

  uint32_t aScales32[2];
  const uint8_t *vScales8 = (const uint8_t *)aScales32;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(aScales32, vX[vI].scales, 4);
    aScales32[1] = (((aScales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
    aScales32[0] = ((aScales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;

    int vSumi1 = 0, vSumi2 = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      const uint8x16_t cIdxL = vld1q_u8(vQs);
      vQs += 16;
      vIdx.vVecIndex =
          vorrq_u16(vmovl_u8(vget_low_u8(cIdxL)), vandq_u16(vshlq_u16(vdupq_n_u16(vQh[vIb32 + 0]), cHshift), cM256));
      const uint32x4_t cAux32x40 = ggml_vld1q_u32(iq3s_grid[vIdx.index[0]], iq3s_grid[vIdx.index[1]],
                                                  iq3s_grid[vIdx.index[2]], iq3s_grid[vIdx.index[3]]);
      const uint32x4_t cAux32x41 = ggml_vld1q_u32(iq3s_grid[vIdx.index[4]], iq3s_grid[vIdx.index[5]],
                                                  iq3s_grid[vIdx.index[6]], iq3s_grid[vIdx.index[7]]);
      vIdx.vVecIndex =
          vorrq_u16(vmovl_u8(vget_high_u8(cIdxL)), vandq_u16(vshlq_u16(vdupq_n_u16(vQh[vIb32 + 1]), cHshift), cM256));
      const uint32x4_t cAux32x42 = ggml_vld1q_u32(iq3s_grid[vIdx.index[0]], iq3s_grid[vIdx.index[1]],
                                                  iq3s_grid[vIdx.index[2]], iq3s_grid[vIdx.index[3]]);
      const uint32x4_t cAux32x43 = ggml_vld1q_u32(iq3s_grid[vIdx.index[4]], iq3s_grid[vIdx.index[5]],
                                                  iq3s_grid[vIdx.index[6]], iq3s_grid[vIdx.index[7]]);

      vVs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(vSigns[0] | ((uint32_t)vSigns[1] << 16)));
      vVs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[1]), cMask2);
      vVs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[0]), cMask2);
      vVs.val[0] = vorrq_u8(vceqq_u8(vVs.val[0], cMask2), cM1);
      vVs.val[1] = vorrq_u8(vceqq_u8(vVs.val[1], cMask2), cM1);

      vQ3s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vVs.val[0]), vreinterpretq_s8_u32(cAux32x40));
      vQ3s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vVs.val[1]), vreinterpretq_s8_u32(cAux32x41));

      vVs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(vSigns[2] | ((uint32_t)vSigns[3] << 16)));
      vVs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[1]), cMask2);
      vVs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vVs.val[0], cMask1.val[0]), cMask2);
      vVs.val[0] = vorrq_u8(vceqq_u8(vVs.val[0], cMask2), cM1);
      vVs.val[1] = vorrq_u8(vceqq_u8(vVs.val[1], cMask2), cM1);

      vSigns += 4;

      vQ3s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vVs.val[0]), vreinterpretq_s8_u32(cAux32x42));
      vQ3s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vVs.val[1]), vreinterpretq_s8_u32(cAux32x43));

      const int32x4_t cP1 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ3s.val[0], vQ8b.val[0]), vQ3s.val[1], vQ8b.val[1]);
      const int32x4_t cP2 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ3s.val[2], vQ8b.val[2]), vQ3s.val[3], vQ8b.val[3]);

      vSumi1 += vaddvq_s32(cP1) * vScales8[vIb32 / 2 + 0];
      vSumi2 += vaddvq_s32(cP2) * vScales8[vIb32 / 2 + 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;

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

#if defined __ARM_NEON

  ggml_int8x16x4_t vQ1b;
  ggml_int8x16x4_t vQ8b;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint16_t *vQh = vX[vI].qh;

    int vSumi1 = 0, vSumi2 = 0, vSumi3 = 0;

    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {

      vQ1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[0] | ((vQh[vIb + 0] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[1] | ((vQh[vIb + 0] << 5) & 0x700)))));
      vQ1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[2] | ((vQh[vIb + 0] << 2) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[3] | ((vQh[vIb + 0] >> 1) & 0x700)))));
      vQ1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[4] | ((vQh[vIb + 1] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[5] | ((vQh[vIb + 1] << 5) & 0x700)))));
      vQ1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[6] | ((vQh[vIb + 1] << 2) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[7] | ((vQh[vIb + 1] >> 1) & 0x700)))));
      vQs += 8;

      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      const int32x4_t cP1 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ1b.val[0], vQ8b.val[0]), vQ1b.val[1], vQ8b.val[1]);
      const int32x4_t cP2 =
          ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ1b.val[2], vQ8b.val[2]), vQ1b.val[3], vQ8b.val[3]);

      const int cLs1 = 2 * ((vQh[vIb + 0] >> 12) & 7) + 1;
      const int cLs2 = 2 * ((vQh[vIb + 1] >> 12) & 7) + 1;
      vSumi1 += vaddvq_s32(cP1) * cLs1;
      vSumi2 += vaddvq_s32(cP2) * cLs2;
      vSumi3 += (vY[vI].bsums[2 * vIb + 0] + vY[vI].bsums[2 * vIb + 1]) * cLs1 * (vQh[vIb + 0] & 0x8000 ? -1 : 1) +
                (vY[vI].bsums[2 * vIb + 2] + vY[vI].bsums[2 * vIb + 3]) * cLs2 * (vQh[vIb + 1] & 0x8000 ? -1 : 1);
    }

    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d) * (vSumi1 + vSumi2 + IQ1S_DELTA * vSumi3);
  }

  *pS = vSumf;

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

#if defined __ARM_NEON
  const int32x4_t cMask = vdupq_n_s32(0x7);
  const int32x4_t cMone = vdupq_n_s32(1);
  const int32x4_t cMzero = vdupq_n_s32(0);

  ggml_int8x16x4_t vDeltas;
  vDeltas.val[0] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(+1));
  vDeltas.val[1] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(+1));
  vDeltas.val[2] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(-1));
  vDeltas.val[3] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1));

  ggml_int8x16x4_t vQ1b;
  ggml_int8x16x4_t vQ8b;

  uint32_t vAux32;
  const uint8_t *vAux8 = (const uint8_t *)&vAux32;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    int32x4_t vSumi1 = cMzero;
    int32x4_t vSumi2 = cMzero;

    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {

      vQ1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[0] | ((vQh[0] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[1] | ((vQh[0] << 4) & 0x700)))));
      vQ1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[2] | ((vQh[1] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[3] | ((vQh[1] << 4) & 0x700)))));
      vQ1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[4] | ((vQh[2] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[5] | ((vQh[2] << 4) & 0x700)))));
      vQ1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (vQs[6] | ((vQh[3] << 8) & 0x700)))),
                                vld1_s8((const int8_t *)(iq1s_grid + (vQs[7] | ((vQh[3] << 4) & 0x700)))));

      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      const int32x4_t cP1 = vpaddq_s32(ggml_vdotq_s32(cMzero, vQ1b.val[0], vQ8b.val[0]),
                                       ggml_vdotq_s32(cMzero, vQ1b.val[1], vQ8b.val[1]));
      const int32x4_t cP2 = vpaddq_s32(ggml_vdotq_s32(cMzero, vQ1b.val[2], vQ8b.val[2]),
                                       ggml_vdotq_s32(cMzero, vQ1b.val[3], vQ8b.val[3]));
      const int32x4_t cP12 = vpaddq_s32(cP1, cP2);

      const uint32_t *vQh32 = (const uint32_t *)vQh; // we are 4-byte aligned, so we can do that
      vAux32 = ((vQh32[0] >> 3) & 0x01010101) | ((vQh32[0] >> 6) & 0x02020202);

      const int32x4_t cP3 = vpaddq_s32(ggml_vdotq_s32(cMzero, vDeltas.val[vAux8[0]], vQ8b.val[0]),
                                       ggml_vdotq_s32(cMzero, vDeltas.val[vAux8[1]], vQ8b.val[1]));
      const int32x4_t cP4 = vpaddq_s32(ggml_vdotq_s32(cMzero, vDeltas.val[vAux8[2]], vQ8b.val[2]),
                                       ggml_vdotq_s32(cMzero, vDeltas.val[vAux8[3]], vQ8b.val[3]));
      const int32x4_t cP34 = vpaddq_s32(cP3, cP4);

      int32x4_t vScales4 = ggml_vld1q_u32(vSc[vIb / 2] >> 0, vSc[vIb / 2] >> 3, vSc[vIb / 2] >> 6, vSc[vIb / 2] >> 9);

      vScales4 = vaddq_s32(vshlq_n_s32(vandq_s32(vScales4, cMask), 1), cMone);

      vSumi1 = vmlaq_s32(vSumi1, vScales4, cP12);
      vSumi2 = vmlaq_s32(vSumi2, vScales4, cP34);

      vQs += 8;
      vQh += 4;
    }

    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16) * (vaddvq_s32(vSumi1) + IQ1M_DELTA * vaddvq_s32(vSumi2));
  }

  *pS = vSumf;

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

#if defined __ARM_NEON
  const int8x16_t cValues = vld1q_s8(kvalues_iq4nl);
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  uint8x16x2_t vQ4bits;
  int8x16x4_t vQ4b;
  int8x16x4_t vQ8b;
  int32x4_t vProd1, vProd2;

  for (; vIb + 1 < cNb; vIb += 2) {

    vQ4bits.val[0] = vld1q_u8(vX[vIb + 0].qs);
    vQ4bits.val[1] = vld1q_u8(vX[vIb + 1].qs);
    vQ8b.val[0] = vld1q_s8(vY[vIb + 0].qs);
    vQ8b.val[1] = vld1q_s8(vY[vIb + 0].qs + 16);
    vQ8b.val[2] = vld1q_s8(vY[vIb + 1].qs);
    vQ8b.val[3] = vld1q_s8(vY[vIb + 1].qs + 16);

    vQ4b.val[0] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[0], cM4b));
    vQ4b.val[1] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[0], 4));
    vQ4b.val[2] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[1], cM4b));
    vQ4b.val[3] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[1], 4));

    vProd1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[0], vQ8b.val[0]), vQ4b.val[1], vQ8b.val[1]);
    vProd2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[2], vQ8b.val[2]), vQ4b.val[3], vQ8b.val[3]);

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIb + 0].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vaddvq_s32(vProd1) +
             GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vaddvq_s32(vProd2);
  }

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

#if defined __ARM_NEON
  const int8x16_t cValues = vld1q_s8(kvalues_iq4nl);
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  ggml_uint8x16x2_t vQ4bits;
  ggml_int8x16x4_t vQ4b;
  ggml_int8x16x4_t vQ8b;
  int32x4_t vProd1, vProd2;

  float vSumf = 0;

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {

    const int8_t *vQ8 = vY[vIbl].qs;
    const uint8_t *vQ4 = vX[vIbl].qs;
    uint16_t vH = vX[vIbl].scales_h;

    int vSumi1 = 0, vSumi2 = 0;
    for (int vIb = 0; vIb < QK_K / 64; ++vIb) {

      vQ4bits = ggml_vld1q_u8_x2(vQ4);
      vQ4 += 32;
      vQ8b = ggml_vld1q_s8_x4(vQ8);
      vQ8 += 64;

      vQ4b.val[0] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[0], cM4b));
      vQ4b.val[1] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[0], 4));
      vQ4b.val[2] = ggml_vqtbl1q_s8(cValues, vandq_u8(vQ4bits.val[1], cM4b));
      vQ4b.val[3] = ggml_vqtbl1q_s8(cValues, vshrq_n_u8(vQ4bits.val[1], 4));

      vProd1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[0], vQ8b.val[0]), vQ4b.val[1], vQ8b.val[1]);
      vProd2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), vQ4b.val[2], vQ8b.val[2]), vQ4b.val[3], vQ8b.val[3]);

      int vLs1 = ((vX[vIbl].scales_l[vIb] & 0xf) | ((vH << 4) & 0x30)) - 32;
      int vLs2 = ((vX[vIbl].scales_l[vIb] >> 4) | ((vH << 2) & 0x30)) - 32;
      vH >>= 4;
      vSumi1 += vaddvq_s32(vProd1) * vLs1;
      vSumi2 += vaddvq_s32(vProd2) * vLs2;
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d * (vSumi1 + vSumi2);
  }

  *pS = vSumf;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
