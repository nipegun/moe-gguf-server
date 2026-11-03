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

#if defined(__aarch64__) && defined(__ARM_NEON) &&                                                                     \
    (defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_DOTPROD))
// Helper for decoding scales and mins of Q4_K and Q5_K block formats
static inline void fDecodeQKx86bitScales(const uint8_t *pScalesIn, int16x8_t *pOutMins, int8_t *pOutScales) {
  constexpr uint32_t cKmask1 = 0x3f3f3f3f;
  constexpr uint32_t cKmask2 = 0x0f0f0f0f;
  constexpr uint32_t cKmask3 = 0x03030303;
  constexpr uint8_t cScalesSize = 12;

  uint32_t aSm[3];
  memcpy(aSm, pScalesIn, cScalesSize);

  const uint32_t cMins03 = aSm[1] & cKmask1;
  const uint32_t cMins47 = ((aSm[2] >> 4) & cKmask2) | (((aSm[1] >> 6) & cKmask3) << 4);
  const uint32x2_t cMinsU32 = {cMins03, cMins47};

  *pOutMins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(cMinsU32)));

  uint32_t aScalesU32[2];
  aScalesU32[0] = aSm[0] & cKmask1;
  aScalesU32[1] = (aSm[2] & cKmask2) | (((aSm[0] >> 6) & cKmask3) << 4);
  memcpy(pOutScales, aScalesU32, 8);
}
#endif

void ggml_quantize_mat_q8_0_4x4(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;

#if defined(__ARM_NEON)
  float32x4_t aSrcv[4][8];
  float aId[4];

  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t aAsrcv[8];
    float32x4_t aAmaxv[8];

    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      for (int vJ = 0; vJ < 8; vJ++)
        aSrcv[vRowIter][vJ] = vld1q_f32(pX + vRowIter * pK + vI * 32 + 4 * vJ);
      for (int vJ = 0; vJ < 8; vJ++)
        aAsrcv[vJ] = vabsq_f32(aSrcv[vRowIter][vJ]);

      for (int vJ = 0; vJ < 4; vJ++)
        aAmaxv[2 * vJ] = vmaxq_f32(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
      for (int vJ = 0; vJ < 2; vJ++)
        aAmaxv[4 * vJ] = vmaxq_f32(aAmaxv[4 * vJ], aAmaxv[4 * vJ + 2]);
      for (int vJ = 0; vJ < 1; vJ++)
        aAmaxv[8 * vJ] = vmaxq_f32(aAmaxv[8 * vJ], aAmaxv[8 * vJ + 4]);

      const float cAmax = vmaxvq_f32(aAmaxv[0]);

      const float cD = cAmax / ((1 << 7) - 1);
      aId[vRowIter] = cD ? 1.0f / cD : 0.0f;

      vY[vI].d[vRowIter] = GGML_CPU_FP32_TO_FP16(cD);
    }

    for (int vJ = 0; vJ < 8; vJ++) {
      float32x4_t v_ = vmulq_n_f32(aSrcv[0][vJ], aId[0]);
      int32x4_t vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[16 * vJ + 0] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[16 * vJ + 1] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[16 * vJ + 2] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[16 * vJ + 3] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[1][vJ], aId[1]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[16 * vJ + 4] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[16 * vJ + 5] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[16 * vJ + 6] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[16 * vJ + 7] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[2][vJ], aId[2]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[16 * vJ + 8] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[16 * vJ + 9] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[16 * vJ + 10] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[16 * vJ + 11] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[3][vJ], aId[3]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[16 * vJ + 12] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[16 * vJ + 13] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[16 * vJ + 14] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[16 * vJ + 15] = vgetq_lane_s32(vVi, 3);
    }
  }
#else
  UNUSED(cNb);
  UNUSED(vY);
  ggml_quantize_mat_q8_0_4x4_generic(pX, pVy, pK);
#endif
}

void ggml_quantize_mat_q8_0_4x8(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;

#if defined(__ARM_NEON)
  float32x4_t aSrcv[4][8];
  float aId[4];

  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t aAsrcv[8];
    float32x4_t aAmaxv[8];

    for (int vRowIter = 0; vRowIter < 4; vRowIter++) {
      for (int vJ = 0; vJ < 8; vJ++)
        aSrcv[vRowIter][vJ] = vld1q_f32(pX + vRowIter * pK + vI * 32 + 4 * vJ);
      for (int vJ = 0; vJ < 8; vJ++)
        aAsrcv[vJ] = vabsq_f32(aSrcv[vRowIter][vJ]);

      for (int vJ = 0; vJ < 4; vJ++)
        aAmaxv[2 * vJ] = vmaxq_f32(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
      for (int vJ = 0; vJ < 2; vJ++)
        aAmaxv[4 * vJ] = vmaxq_f32(aAmaxv[4 * vJ], aAmaxv[4 * vJ + 2]);
      for (int vJ = 0; vJ < 1; vJ++)
        aAmaxv[8 * vJ] = vmaxq_f32(aAmaxv[8 * vJ], aAmaxv[8 * vJ + 4]);

      const float cAmax = vmaxvq_f32(aAmaxv[0]);

      const float cD = cAmax / ((1 << 7) - 1);
      aId[vRowIter] = cD ? 1.0f / cD : 0.0f;

      vY[vI].d[vRowIter] = GGML_CPU_FP32_TO_FP16(cD);
    }

    for (int vJ = 0; vJ < 4; vJ++) {
      float32x4_t v_ = vmulq_n_f32(aSrcv[0][2 * vJ], aId[0]);
      int32x4_t vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 0] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 1] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 2] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 3] = vgetq_lane_s32(vVi, 3);
      v_ = vmulq_n_f32(aSrcv[0][2 * vJ + 1], aId[0]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 4] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 5] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 6] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 7] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[1][2 * vJ], aId[1]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 8] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 9] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 10] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 11] = vgetq_lane_s32(vVi, 3);
      v_ = vmulq_n_f32(aSrcv[1][2 * vJ + 1], aId[1]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 12] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 13] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 14] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 15] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[2][2 * vJ], aId[2]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 16] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 17] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 18] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 19] = vgetq_lane_s32(vVi, 3);
      v_ = vmulq_n_f32(aSrcv[2][2 * vJ + 1], aId[2]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 20] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 21] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 22] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 23] = vgetq_lane_s32(vVi, 3);

      v_ = vmulq_n_f32(aSrcv[3][2 * vJ], aId[3]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 24] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 25] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 26] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 27] = vgetq_lane_s32(vVi, 3);
      v_ = vmulq_n_f32(aSrcv[3][2 * vJ + 1], aId[3]);
      vVi = vcvtnq_s32_f32(v_);
      vY[vI].qs[32 * vJ + 28] = vgetq_lane_s32(vVi, 0);
      vY[vI].qs[32 * vJ + 29] = vgetq_lane_s32(vVi, 1);
      vY[vI].qs[32 * vJ + 30] = vgetq_lane_s32(vVi, 2);
      vY[vI].qs[32 * vJ + 31] = vgetq_lane_s32(vVi, 3);
    }
  }

#else
  UNUSED(cNb);
  UNUSED(vY);
  ggml_quantize_mat_q8_0_4x8_generic(pX, pVy, pK);
#endif
}

void ggml_gemv_q4_0_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx;

  for (int vC = 0; vC < pNc; vC += cNcolsInterleaved) {
    const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
    float32x4_t vAcc = vdupq_n_f32(0);
    for (int vB = 0; vB < cNb; vB++) {
      int8x16_t vB0 = vld1q_s8((const int8_t *)vBPtr->qs);
      int8x16_t vB1 = vld1q_s8((const int8_t *)vBPtr->qs + 16);
      int8x16_t vB2 = vld1q_s8((const int8_t *)vBPtr->qs + 32);
      int8x16_t vB3 = vld1q_s8((const int8_t *)vBPtr->qs + 48);
      float16x4_t vBd = vld1_f16((const __fp16 *)vBPtr->d);

      int8x16_t vA0 = vld1q_s8(vAPtr->qs);
      int8x16_t vA1 = vld1q_s8(vAPtr->qs + cQk / 2);
      float16x4_t vAd = vld1_dup_f16((const __fp16 *)&vAPtr->d);

      int32x4_t vRet = vdupq_n_s32(0);

      vRet = vdotq_laneq_s32(vRet, vB0 << 4, vA0, 0);
      vRet = vdotq_laneq_s32(vRet, vB1 << 4, vA0, 1);
      vRet = vdotq_laneq_s32(vRet, vB2 << 4, vA0, 2);
      vRet = vdotq_laneq_s32(vRet, vB3 << 4, vA0, 3);

      vRet = vdotq_laneq_s32(vRet, vB0 & 0xf0U, vA1, 0);
      vRet = vdotq_laneq_s32(vRet, vB1 & 0xf0U, vA1, 1);
      vRet = vdotq_laneq_s32(vRet, vB2 & 0xf0U, vA1, 2);
      vRet = vdotq_laneq_s32(vRet, vB3 & 0xf0U, vA1, 3);

      vAcc = vfmaq_f32(vAcc, vcvtq_n_f32_s32(vRet, 4), vmulq_f32(vcvt_f32_f16(vAd), vcvt_f32_f16(vBd)));
      vAPtr++;
      vBPtr++;
    }
    vst1q_f32(pS, vAcc);
    pS += cNcolsInterleaved;
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&
       // defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q4_0_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q4_0_4x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const block_q4_0x4 *vBPtr = (const block_q4_0x4 *)pVx;

  for (int vC = 0; vC < pNc; vC += cNcolsInterleaved) {
    const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
    float32x4_t vAcc = vdupq_n_f32(0);
    for (int vB = 0; vB < cNb; vB++) {
      int8x16_t vB0 = vld1q_s8((const int8_t *)vBPtr->qs);
      int8x16_t vB1 = vld1q_s8((const int8_t *)vBPtr->qs + 16);
      int8x16_t vB2 = vld1q_s8((const int8_t *)vBPtr->qs + 32);
      int8x16_t vB3 = vld1q_s8((const int8_t *)vBPtr->qs + 48);
      float16x4_t vBd = vld1_f16((const __fp16 *)vBPtr->d);

      int8x16_t vA0 = (int8x16_t)vld1q_dup_s64((const int64_t *)vAPtr->qs);
      int8x16_t vA1 = (int8x16_t)vld1q_dup_s64((const int64_t *)vAPtr->qs + 1);
      int8x16_t vA2 = (int8x16_t)vld1q_dup_s64((const int64_t *)vAPtr->qs + 2);
      int8x16_t vA3 = (int8x16_t)vld1q_dup_s64((const int64_t *)vAPtr->qs + 3);
      float16x4_t vAd = vld1_dup_f16((const __fp16 *)&vAPtr->d);

      int32x4_t vRet0 = vdupq_n_s32(0);
      int32x4_t vRet1 = vdupq_n_s32(0);

      vRet0 = vdotq_s32(vRet0, vB0 << 4, vA0);
      vRet1 = vdotq_s32(vRet1, vB1 << 4, vA0);
      vRet0 = vdotq_s32(vRet0, vB2 << 4, vA1);
      vRet1 = vdotq_s32(vRet1, vB3 << 4, vA1);

      vRet0 = vdotq_s32(vRet0, vB0 & 0xf0U, vA2);
      vRet1 = vdotq_s32(vRet1, vB1 & 0xf0U, vA2);
      vRet0 = vdotq_s32(vRet0, vB2 & 0xf0U, vA3);
      vRet1 = vdotq_s32(vRet1, vB3 & 0xf0U, vA3);

      int32x4_t vRet = vpaddq_s32(vRet0, vRet1);

      vAcc = vfmaq_f32(vAcc, vcvtq_n_f32_s32(vRet, 4), vmulq_f32(vcvt_f32_f16(vAd), vcvt_f32_f16(vBd)));
      vAPtr++;
      vBPtr++;
    }
    vst1q_f32(pS, vAcc);
    pS += cNcolsInterleaved;
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&
       // defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q4_0_4x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int nc) {
  const int cQk = QK8_0;
  const int nb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(nc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(pNr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE)
  if (ggml_cpu_get_sve_cnt() == QK8_0) {
    const void *b_ptr = pVx;
    const void *a_ptr = pVy;
    float *res_ptr = pS;

    __asm__ __volatile__("ptrue p0.b\n"
                         "add %x[b_ptr], %x[b_ptr], #0x10\n"
                         "1:" // Column loop
                         "add x22, %x[a_ptr], #0x2\n"
                         "mov z31.b, #0x0\n"
                         "mov x21, %x[nb]\n"
                         "2:" // Block loop
                         "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
                         "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
                         "mov z28.s, #0x0\n"
                         "mov z27.s, #0x0\n"
                         "ld1rd { z26.d }, p0/Z, [x22]\n"
                         "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
                         "sub x20, x22, #0x2\n"
                         "sub x21, x21, #0x1\n"
                         "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
                         "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
                         "lsl z22.b, z30.b, #0x4\n"
                         "lsl z16.b, z29.b, #0x4\n"
                         "and z30.b, z30.b, #0xf0\n"
                         "and z29.b, z29.b, #0xf0\n"
                         "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
                         "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
                         "lsl z19.b, z25.b, #0x4\n"
                         "and z25.b, z25.b, #0xf0\n"
                         "ld1rh { z17.h }, p0/Z, [x20]\n"
                         "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
                         "sdot z28.s, z22.b, z26.b\n"
                         "sdot z27.s, z16.b, z26.b\n"
                         "lsl z16.b, z24.b, #0x4\n"
                         "add x22, x22, #0x22\n"
                         "and z24.b, z24.b, #0xf0\n"
                         "add %x[b_ptr], %x[b_ptr], #0x90\n"
                         "fcvt z17.s, p0/m, z17.h\n"
                         "fcvt z18.s, p0/m, z18.h\n"
                         "sdot z28.s, z19.b, z23.b\n"
                         "sdot z27.s, z16.b, z23.b\n"
                         "fmul z18.s, z18.s, z17.s\n"
                         "sdot z28.s, z30.b, z21.b\n"
                         "sdot z27.s, z29.b, z21.b\n"
                         "sdot z28.s, z25.b, z20.b\n"
                         "sdot z27.s, z24.b, z20.b\n"
                         "uzp1 z17.s, z28.s, z27.s\n"
                         "uzp2 z16.s, z28.s, z27.s\n"
                         "add z17.s, z17.s, z16.s\n"
                         "asr z17.s, z17.s, #0x4\n"
                         "scvtf z17.s, p0/m, z17.s\n"
                         "fmla z31.s, p0/M, z17.s, z18.s\n"
                         "cbnz x21, 2b\n"
                         "sub %x[nc], %x[nc], #0x8\n"
                         "st1w { z31.s }, p0, [%x[res_ptr]]\n"
                         "add %x[res_ptr], %x[res_ptr], #0x20\n"
                         "cbnz %x[nc], 1b\n"
                         : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
                         : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
                         : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
                           "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
    return;
  }
#endif // #if defined(__ARM_FEATURE_SVE)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
  ggml_gemv_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, nc);
}

void ggml_gemv_iq4_nl_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                               const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const int8x16_t cKvalues = vld1q_s8(kvalues_iq4nl);
  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  float *vResPtr = pS;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_iq4_nlx4 *vBPtr = (const block_iq4_nlx4 *)pVx + (vX * cNb);

    float32x4_t vSumf = vdupq_n_f32(0);
    for (int vL = 0; vL < cNb; vL++) {
      uint8x16_t vB0 = vld1q_u8(vBPtr[vL].qs + 0);
      uint8x16_t vB1 = vld1q_u8(vBPtr[vL].qs + 16);
      uint8x16_t vB2 = vld1q_u8(vBPtr[vL].qs + 32);
      uint8x16_t vB3 = vld1q_u8(vBPtr[vL].qs + 48);

      int8x16_t vB0Hi = vqtbl1q_s8(cKvalues, vB0 >> 4);
      int8x16_t vB0Lo = vqtbl1q_s8(cKvalues, vB0 & 0x0F);
      int8x16_t vB1Hi = vqtbl1q_s8(cKvalues, vB1 >> 4);
      int8x16_t vB1Lo = vqtbl1q_s8(cKvalues, vB1 & 0x0F);
      int8x16_t vB2Hi = vqtbl1q_s8(cKvalues, vB2 >> 4);
      int8x16_t vB2Lo = vqtbl1q_s8(cKvalues, vB2 & 0x0F);
      int8x16_t vB3Hi = vqtbl1q_s8(cKvalues, vB3 >> 4);
      int8x16_t vB3Lo = vqtbl1q_s8(cKvalues, vB3 & 0x0F);

      int8x16_t vA0 = vld1q_s8(vAPtr[vL].qs + 0);
      int8x16_t vA1 = vld1q_s8(vAPtr[vL].qs + 16);

      int32x4_t vSumi = vdupq_n_s32(0);
      vSumi = vdotq_laneq_s32(vSumi, vB0Lo, vA0, 0);
      vSumi = vdotq_laneq_s32(vSumi, vB0Hi, vA1, 0);
      vSumi = vdotq_laneq_s32(vSumi, vB1Lo, vA0, 1);
      vSumi = vdotq_laneq_s32(vSumi, vB1Hi, vA1, 1);
      vSumi = vdotq_laneq_s32(vSumi, vB2Lo, vA0, 2);
      vSumi = vdotq_laneq_s32(vSumi, vB2Hi, vA1, 2);
      vSumi = vdotq_laneq_s32(vSumi, vB3Lo, vA0, 3);
      vSumi = vdotq_laneq_s32(vSumi, vB3Hi, vA1, 3);

      float32x4_t vAD = vcvt_f32_f16(vld1_dup_f16((const float16_t *)&vAPtr[vL].d));
      float32x4_t vBD = vcvt_f32_f16(vld1_f16((const float16_t *)vBPtr[vL].d));
      float32x4_t vD = vAD * vBD;

      vSumf = vmlaq_f32(vSumf, vD, vcvtq_f32_s32(vSumi));
    }

    vst1q_f32(vResPtr + vX * 4, vSumf);
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
  ggml_gemv_iq4_nl_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_mxfp4_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const int8x16_t cKvalues = vld1q_s8(kvalues_mxfp4);
  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  float *vResPtr = pS;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_mxfp4x4 *vBPtr = (const block_mxfp4x4 *)pVx + (vX * cNb);

    float32x4_t vSumf = vdupq_n_f32(0);
    for (int vL = 0; vL < cNb; vL++) {
      uint8x16_t vB0 = vld1q_u8(vBPtr[vL].qs + 0);
      uint8x16_t vB1 = vld1q_u8(vBPtr[vL].qs + 16);
      uint8x16_t vB2 = vld1q_u8(vBPtr[vL].qs + 32);
      uint8x16_t vB3 = vld1q_u8(vBPtr[vL].qs + 48);

      int8x16_t vB0Hi = vqtbl1q_s8(cKvalues, vB0 >> 4);
      int8x16_t vB0Lo = vqtbl1q_s8(cKvalues, vB0 & 0x0F);
      int8x16_t vB1Hi = vqtbl1q_s8(cKvalues, vB1 >> 4);
      int8x16_t vB1Lo = vqtbl1q_s8(cKvalues, vB1 & 0x0F);
      int8x16_t vB2Hi = vqtbl1q_s8(cKvalues, vB2 >> 4);
      int8x16_t vB2Lo = vqtbl1q_s8(cKvalues, vB2 & 0x0F);
      int8x16_t vB3Hi = vqtbl1q_s8(cKvalues, vB3 >> 4);
      int8x16_t vB3Lo = vqtbl1q_s8(cKvalues, vB3 & 0x0F);

      int8x16_t vA0 = vld1q_s8(vAPtr[vL].qs + 0);
      int8x16_t vA1 = vld1q_s8(vAPtr[vL].qs + 16);

      int32x4_t vSumi = vdupq_n_s32(0);
      vSumi = vdotq_laneq_s32(vSumi, vB0Lo, vA0, 0);
      vSumi = vdotq_laneq_s32(vSumi, vB0Hi, vA1, 0);
      vSumi = vdotq_laneq_s32(vSumi, vB1Lo, vA0, 1);
      vSumi = vdotq_laneq_s32(vSumi, vB1Hi, vA1, 1);
      vSumi = vdotq_laneq_s32(vSumi, vB2Lo, vA0, 2);
      vSumi = vdotq_laneq_s32(vSumi, vB2Hi, vA1, 2);
      vSumi = vdotq_laneq_s32(vSumi, vB3Lo, vA0, 3);
      vSumi = vdotq_laneq_s32(vSumi, vB3Hi, vA1, 3);

      float32x4_t vAD = vcvt_f32_f16(vld1_dup_f16((const float16_t *)&vAPtr[vL].d));
      float32x4_t vBD = {
          GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[0]),
          GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[1]),
          GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[2]),
          GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[3]),
      };
      float32x4_t vD = vAD * vBD;

      vSumf = vmlaq_f32(vSumf, vD, vcvtq_f32_s32(vSumi));
    }

    vst1q_f32(vResPtr + vX * 4, vSumf);
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
  ggml_gemv_mxfp4_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q4_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColGroups = cNcolsInterleaved / 4; // 0123 and 4567
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[cColGroups];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_Kx8 *GGML_RESTRICT vQ4Ptr = (const block_q4_Kx8 *)pVx + (vX * cNb);

    for (int vI = 0; vI < cColGroups; vI++) {
      vAccF32[vI] = vdupq_n_f32(0);
    }

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ4D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ4D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0123 = vmulq_f32(vQ4D0, vQ8D);
      float32x4_t vSbScale4567 = vmulq_f32(vQ4D1, vQ8D);
      float32x4_t vQ4Dmin0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin));     // dmin 0..3
      float32x4_t vQ4Dmin1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin + 4)); // dmin 4..7
      float32x4_t vSbMin0123 = vmulq_f32(vQ4Dmin0, vQ8D);
      float32x4_t vSbMin4567 = vmulq_f32(vQ4Dmin1, vQ8D);

      // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
      int32x4_t aBiasAcc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
      int32x4_t vAccLo[cColGroups];
      int32x4_t vAccHi[cColGroups];

      // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
      const int16x8_t cBsums = vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums), vld1q_s16(vQ8Ptr[vB].bsums + 8));
      int16_t vBsumsArr[8];
      vst1q_s16(vBsumsArr, cBsums);
      for (int vSb = 0; vSb < QK_K / 64; vSb++) {
        for (int vI = 0; vI < cColGroups; vI++) {
          vAccLo[vI] = vdupq_n_s32(0);
          vAccHi[vI] = vdupq_n_s32(0);
        }
        // Need scales for the low and high nibbles
        // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
        int16x8_t aQ4sbMins[2];
        int16x8_t aQ4sbScales[2];
        for (int vI = 0; vI < 2; vI++) {
          int8_t vAuxQ4sb[8];
          const int cOffset = vSb * 24 + vI * 12;
          fDecodeQKx86bitScales(&vQ4Ptr[vB].scales[cOffset], &aQ4sbMins[vI], vAuxQ4sb);
          aQ4sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ4sb));
        }

        int8x16_t vQ8Qs[64 / 16];
        for (int vI = 0; vI < 64 / 16; vI++) {
          vQ8Qs[vI] = vld1q_s8(vQ8Ptr[vB].qs + vSb * 64 + vI * 16);
        }

        for (int vC = 0; vC < cColGroups; vC++) {
          uint8x16_t vQ4Cols[8];
          for (int vI = 0; vI < 8; vI++) {
            vQ4Cols[vI] = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + vI * 32 + 16 * vC);
          }

          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[0], cM4b)), vQ8Qs[0], 0);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[1], cM4b)), vQ8Qs[0], 1);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[2], cM4b)), vQ8Qs[0], 2);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[3], cM4b)), vQ8Qs[0], 3);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[4], cM4b)), vQ8Qs[1], 0);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[5], cM4b)), vQ8Qs[1], 1);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[6], cM4b)), vQ8Qs[1], 2);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vreinterpretq_s8_u8(vandq_u8(vQ4Cols[7], cM4b)), vQ8Qs[1], 3);

          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[0], 4)), vQ8Qs[2], 0);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[1], 4)), vQ8Qs[2], 1);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[2], 4)), vQ8Qs[2], 2);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[3], 4)), vQ8Qs[2], 3);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[4], 4)), vQ8Qs[3], 0);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[5], 4)), vQ8Qs[3], 1);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[6], 4)), vQ8Qs[3], 2);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vreinterpretq_s8_u8(vshrq_n_u8(vQ4Cols[7], 4)), vQ8Qs[3], 3);
        }

        // Scales
        // row c0123 blk0 and blk1
        const int16x4_t cSc0123Lo = vget_low_s16(aQ4sbScales[0]);
        const int16x4_t cSc0123Hi = vget_low_s16(aQ4sbScales[1]);
        const float32x4_t cSumf0123 = vcvtq_f32_s32(
            vaddq_s32(vmulq_s32(vmovl_s16(cSc0123Lo), vAccLo[0]), vmulq_s32(vmovl_s16(cSc0123Hi), vAccHi[0])));
        vAccF32[0] = vfmaq_f32(vAccF32[0], vSbScale0123, cSumf0123);
        // row c4567 blk0 and blk1
        const int16x4_t cSc4567Lo = vget_high_s16(aQ4sbScales[0]);
        const int16x4_t cSc4567Hi = vget_high_s16(aQ4sbScales[1]);
        const float32x4_t cSumf4567 = vcvtq_f32_s32(
            vaddq_s32(vmulq_s32(vmovl_s16(cSc4567Lo), vAccLo[1]), vmulq_s32(vmovl_s16(cSc4567Hi), vAccHi[1])));
        vAccF32[1] = vfmaq_f32(vAccF32[1], vSbScale4567, cSumf4567);

        // Bias Correction
        const int16x4_t cBsumsVecLo = vdup_n_s16(vBsumsArr[2 * vSb + 0]);
        const int16x4_t cBsumsVecHi = vdup_n_s16(vBsumsArr[2 * vSb + 1]);

        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], cBsumsVecLo, vget_low_s16(aQ4sbMins[0]));
        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], cBsumsVecHi, vget_low_s16(aQ4sbMins[1]));
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], cBsumsVecLo, vget_high_s16(aQ4sbMins[0]));
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], cBsumsVecHi, vget_high_s16(aQ4sbMins[1]));
      } // for sb

      vAccF32[0] = vmlsq_f32(vAccF32[0], vcvtq_f32_s32(aBiasAcc[0]), vSbMin0123);
      vAccF32[1] = vmlsq_f32(vAccF32[1], vcvtq_f32_s32(aBiasAcc[1]), vSbMin4567);
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // #if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q4_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q4_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColPairs = cNcolsInterleaved / 2;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[cNcolsInterleaved / 4];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_Kx8 *GGML_RESTRICT vQ4Ptr = (const block_q4_Kx8 *)pVx + (vX * cNb);

    for (int vI = 0; vI < cNcolsInterleaved / 4; vI++) {
      vAccF32[vI] = vdupq_n_f32(0);
    }

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ4D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ4D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0 = vmulq_f32(vQ4D0, vQ8D);
      float32x4_t vSbScale1 = vmulq_f32(vQ4D1, vQ8D);
      float32x4_t vQ4Dmin0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin));     // dmin 0..3
      float32x4_t vQ4Dmin1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin + 4)); // dmin 4..7
      float32x4_t vSbMin0 = vmulq_f32(vQ4Dmin0, vQ8D);
      float32x4_t vSbMin1 = vmulq_f32(vQ4Dmin1, vQ8D);

      // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
      int32x4_t aBiasAcc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
      // 2 sb each iteration
      int32x4_t vAccLo[cColPairs];
      int32x4_t vAccHi[cColPairs];

      // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
      const int16x8_t cBsums = vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums), vld1q_s16(vQ8Ptr[vB].bsums + 8));
      int16_t vBsumsArr[8];
      vst1q_s16(vBsumsArr, cBsums);
      for (int vSb = 0; vSb < QK_K / 64; vSb++) {
        for (int vI = 0; vI < cColPairs; vI++) {
          vAccLo[vI] = vdupq_n_s32(0);
          vAccHi[vI] = vdupq_n_s32(0);
        }
        // Need scales for the low and high nibbles
        // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
        int16x8_t aQ4sbMins[2]; // int16 as its needed for bias_acc later
        int16x8_t aQ4sbScales[2];
        for (int vI = 0; vI < 2; vI++) {
          int8_t vAuxQ4sb[8];
          const int cOffset = vSb * 24 + vI * 12;
          fDecodeQKx86bitScales(&vQ4Ptr[vB].scales[cOffset], &aQ4sbMins[vI], vAuxQ4sb);
          aQ4sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ4sb));
        }

        const uint8_t *vQ4Base = vQ4Ptr[vB].qs + vSb * QK_K;

        // Load the 64 quants from q8K duplicated to use vecdots with the interleaved columns
        // but still need the qs to use the low and hi bits from q4
        const int8_t *vQ8Base = vQ8Ptr[vB].qs + vSb * 64;
        int8x16_t vQ8Qs[8];
        for (int vI = 0; vI < 8; vI++) {
          vQ8Qs[vI] = (int8x16_t)vld1q_dup_s64((const int64_t *)(vQ8Base + vI * 8));
        }

        // Q4s columns iterated in pairs (01, 23, 45, 67)
        for (int vCp = 0; vCp < cColPairs; vCp++) {
          uint8x16_t vQ4QsCp0 = vld1q_u8(vQ4Base + 16 * vCp);
          uint8x16_t vQ4QsCp1 = vld1q_u8(vQ4Base + 16 * vCp + 64);
          uint8x16_t vQ4QsCp2 = vld1q_u8(vQ4Base + 16 * vCp + 128);
          uint8x16_t vQ4QsCp3 = vld1q_u8(vQ4Base + 16 * vCp + 192);

          vAccLo[vCp] = ggml_vdotq_s32(vAccLo[vCp], vreinterpretq_s8_u8(vandq_u8(vQ4QsCp0, cM4b)), vQ8Qs[0]); // 0 .. 7
          vAccLo[vCp] = ggml_vdotq_s32(vAccLo[vCp], vreinterpretq_s8_u8(vandq_u8(vQ4QsCp1, cM4b)), vQ8Qs[1]); // 8 ..15
          vAccLo[vCp] = ggml_vdotq_s32(vAccLo[vCp], vreinterpretq_s8_u8(vandq_u8(vQ4QsCp2, cM4b)), vQ8Qs[2]); // 16..23
          vAccLo[vCp] = ggml_vdotq_s32(vAccLo[vCp], vreinterpretq_s8_u8(vandq_u8(vQ4QsCp3, cM4b)), vQ8Qs[3]); // 24..31

          vAccHi[vCp] = ggml_vdotq_s32(vAccHi[vCp], vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp0, 4)), vQ8Qs[4]); // 32..39
          vAccHi[vCp] = ggml_vdotq_s32(vAccHi[vCp], vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp1, 4)), vQ8Qs[5]); // 40..47
          vAccHi[vCp] = ggml_vdotq_s32(vAccHi[vCp], vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp2, 4)), vQ8Qs[6]); // 48..55
          vAccHi[vCp] = ggml_vdotq_s32(vAccHi[vCp], vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp3, 4)), vQ8Qs[7]); // 56..63
        }

        // Iterates over a pair of column pairs (4 columns) to use a single 128 register
        // p = 0 -> 0123  p2 -> 4567
        for (int vI = 0, vP = 0; vP < cColPairs; vI++, vP += 2) {
          int16x4_t vGroupScalesLo = vP == 0 ? vget_low_s16(aQ4sbScales[0]) : vget_high_s16(aQ4sbScales[0]);
          int16x4_t vGroupScalesHi = vP == 0 ? vget_low_s16(aQ4sbScales[1]) : vget_high_s16(aQ4sbScales[1]);
          float32x4_t vSbScale = vP == 0 ? vSbScale0 : vSbScale1;

          // 0123 or 4567
          float32x4_t vSumf0 =
              vcvtq_f32_s32(vmulq_s32(vmovl_s16(vGroupScalesLo), vpaddq_s32(vAccLo[vP], vAccLo[vP + 1])));
          vAccF32[vI] = vfmaq_f32(vAccF32[vI], vSbScale, vSumf0);

          float32x4_t vSumf1 =
              vcvtq_f32_s32(vmulq_s32(vmovl_s16(vGroupScalesHi), vpaddq_s32(vAccHi[vP], vAccHi[vP + 1])));
          vAccF32[vI] = vfmaq_f32(vAccF32[vI], vSbScale, vSumf1);
        }

        // Multiply Acc bsum + mins
        // Each pair of subblocks share the same bsums
        // Load scalar bsum → broadcast to a vector (vdupq_n_s16(s)).
        int16x4_t vBsumsVecLo = vdup_n_s16(vBsumsArr[2 * vSb + 0]);
        int16x4_t vBsumsVecHi = vdup_n_s16(vBsumsArr[2 * vSb + 1]);

        // cols 0-3 bias
        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], vBsumsVecLo, vget_low_s16(aQ4sbMins[0]));
        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], vBsumsVecHi, vget_low_s16(aQ4sbMins[1]));

        // cols 4-7 bias
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], vBsumsVecLo, vget_high_s16(aQ4sbMins[0]));
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], vBsumsVecHi, vget_high_s16(aQ4sbMins[1]));
      } // for sb

      vAccF32[0] = vmlsq_f32(vAccF32[0], vcvtq_f32_s32(aBiasAcc[0]), vSbMin0);
      vAccF32[1] = vmlsq_f32(vAccF32[1], vcvtq_f32_s32(aBiasAcc[1]), vSbMin1);
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q4_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q5_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColGroups = cNcolsInterleaved / 4; // 0123 and 4567
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMone = vdupq_n_u8(1);
  const uint8x16_t cMtwo = vdupq_n_u8(2);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[cColGroups];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q5_Kx8 *GGML_RESTRICT vQ5Ptr = (const block_q5_Kx8 *)pVx + (vX * cNb);

    for (int vI = 0; vI < cColGroups; vI++) {
      vAccF32[vI] = vdupq_n_f32(0);
    }

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ5D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ5D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0123 = vmulq_f32(vQ5D0, vQ8D);
      float32x4_t vSbScale4567 = vmulq_f32(vQ5D1, vQ8D);
      float32x4_t vQ5Dmin0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin));     // dmin 0..3
      float32x4_t vQ5Dmin1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin + 4)); // dmin 4..7
      float32x4_t vSbMin0123 = vmulq_f32(vQ5Dmin0, vQ8D);
      float32x4_t vSbMin4567 = vmulq_f32(vQ5Dmin1, vQ8D);

      // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
      int32x4_t aBiasAcc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
      int32x4_t vAccLo[cColGroups];
      int32x4_t vAccHi[cColGroups];

      // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
      const int16x8_t cBsums = vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums), vld1q_s16(vQ8Ptr[vB].bsums + 8));
      int16_t vBsumsArr[8];
      vst1q_s16(vBsumsArr, cBsums);

      uint8x16_t vQh[cColGroups][8];
      for (int vC = 0; vC < cColGroups; vC++) {
        for (int vI = 0; vI < 8; vI++) {
          vQh[vC][vI] = vld1q_u8(vQ5Ptr[vB].qh + vI * 32 + 16 * vC);
        }
      }

      for (int vSb = 0; vSb < QK_K / 64; vSb++) {
        for (int vI = 0; vI < cColGroups; vI++) {
          vAccLo[vI] = vdupq_n_s32(0);
          vAccHi[vI] = vdupq_n_s32(0);
        }
        // Need scales for the low and high nibbles
        // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
        int16x8_t aQ5sbMins[2];
        int16x8_t aQ5sbScales[2];
        for (int vI = 0; vI < 2; vI++) {
          int8_t vAuxQ5sb[8];
          const int cOffset = vSb * 24 + vI * 12;
          fDecodeQKx86bitScales(&vQ5Ptr[vB].scales[cOffset], &aQ5sbMins[vI], vAuxQ5sb);
          aQ5sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ5sb));
        }

        int8x16_t vQ8Qs[4];
        for (int vI = 0; vI < 4; vI++) {
          vQ8Qs[vI] = vld1q_s8(vQ8Ptr[vB].qs + vSb * 64 + vI * 16);
        }

        for (int vC = 0; vC < cColGroups; vC++) {
          uint8x16_t vQ5Cols[8];
          uint8x16_t vHbitLo[8];
          uint8x16_t aHbitHi[8];
          int8x16_t vQ5Lo[8];
          int8x16_t vQ5Hi[8];

          for (int vI = 0; vI < 8; vI++) {
            vQ5Cols[vI] = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + vI * 32 + 16 * vC);
            vHbitLo[vI] = vandq_u8(vQh[vC][vI], cMone);
            aHbitHi[vI] = vshlq_n_u8(vandq_u8(vQh[vC][vI], cMtwo), 3);
            vQh[vC][vI] = vshrq_n_u8(vQh[vC][vI], 2);
            vQ5Lo[vI] = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQ5Cols[vI], cM4b), vHbitLo[vI], 4));
            vQ5Hi[vI] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQ5Cols[vI], 4), aHbitHi[vI]));
          }

          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[0], vQ8Qs[0], 0);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[1], vQ8Qs[0], 1);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[2], vQ8Qs[0], 2);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[3], vQ8Qs[0], 3);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[4], vQ8Qs[1], 0);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[5], vQ8Qs[1], 1);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[6], vQ8Qs[1], 2);
          vAccLo[vC] = vdotq_laneq_s32(vAccLo[vC], vQ5Lo[7], vQ8Qs[1], 3);

          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[0], vQ8Qs[2], 0);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[1], vQ8Qs[2], 1);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[2], vQ8Qs[2], 2);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[3], vQ8Qs[2], 3);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[4], vQ8Qs[3], 0);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[5], vQ8Qs[3], 1);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[6], vQ8Qs[3], 2);
          vAccHi[vC] = vdotq_laneq_s32(vAccHi[vC], vQ5Hi[7], vQ8Qs[3], 3);
        }

        // Scales
        // row c0123 blk0 and blk1
        const int16x4_t cSc0123Lo = vget_low_s16(aQ5sbScales[0]);
        const int16x4_t cSc0123Hi = vget_low_s16(aQ5sbScales[1]);
        const float32x4_t cSumf0123 = vcvtq_f32_s32(
            vaddq_s32(vmulq_s32(vmovl_s16(cSc0123Lo), vAccLo[0]), vmulq_s32(vmovl_s16(cSc0123Hi), vAccHi[0])));
        vAccF32[0] = vfmaq_f32(vAccF32[0], vSbScale0123, cSumf0123);
        // row c4567 blk0 and blk1
        const int16x4_t cSc4567Lo = vget_high_s16(aQ5sbScales[0]);
        const int16x4_t cSc4567Hi = vget_high_s16(aQ5sbScales[1]);
        const float32x4_t cSumf4567 = vcvtq_f32_s32(
            vaddq_s32(vmulq_s32(vmovl_s16(cSc4567Lo), vAccLo[1]), vmulq_s32(vmovl_s16(cSc4567Hi), vAccHi[1])));
        vAccF32[1] = vfmaq_f32(vAccF32[1], vSbScale4567, cSumf4567);

        // Bias Correction
        const int16x4_t cBsumsVecLo = vdup_n_s16(vBsumsArr[2 * vSb + 0]);
        const int16x4_t cBsumsVecHi = vdup_n_s16(vBsumsArr[2 * vSb + 1]);

        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], cBsumsVecLo, vget_low_s16(aQ5sbMins[0]));
        aBiasAcc[0] = vmlal_s16(aBiasAcc[0], cBsumsVecHi, vget_low_s16(aQ5sbMins[1]));
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], cBsumsVecLo, vget_high_s16(aQ5sbMins[0]));
        aBiasAcc[1] = vmlal_s16(aBiasAcc[1], cBsumsVecHi, vget_high_s16(aQ5sbMins[1]));
      } // for sb

      vAccF32[0] = vmlsq_f32(vAccF32[0], vcvtq_f32_s32(aBiasAcc[0]), vSbMin0123);
      vAccF32[1] = vmlsq_f32(vAccF32[1], vcvtq_f32_s32(aBiasAcc[1]), vSbMin4567);
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q5_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q5_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColPairs = cNcolsInterleaved / 2;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMone = vdupq_n_u8(1);
  const uint8x16_t cMtwo = vdupq_n_u8(2);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[cNcolsInterleaved / 4];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q5_Kx8 *GGML_RESTRICT vQ5Ptr = (const block_q5_Kx8 *)pVx + (vX * cNb);

    for (int vI = 0; vI < cNcolsInterleaved / 4; vI++) {
      vAccF32[vI] = vdupq_n_f32(0);
    }

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ5D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ5D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0 = vmulq_f32(vQ5D0, vQ8D);
      float32x4_t vSbScale1 = vmulq_f32(vQ5D1, vQ8D);
      float32x4_t vQ5Dmin0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin));     // dmin 0..3
      float32x4_t vQ5Dmin1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin + 4)); // dmin 4..7
      float32x4_t vSbMin0 = vmulq_f32(vQ5Dmin0, vQ8D);
      float32x4_t vSbMin1 = vmulq_f32(vQ5Dmin1, vQ8D);

      // 2 sb each iteration
      int32x4_t vAccLo[cColPairs];
      int32x4_t vAccHi[cColPairs];

      // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
      const int16x8_t cBsums = vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums), vld1q_s16(vQ8Ptr[vB].bsums + 8));
      int16_t vBsumsArr[8];
      vst1q_s16(vBsumsArr, cBsums);

      // Load qh once per block and shift after each subblock
      const uint8_t *vQhBase = vQ5Ptr[vB].qh;
      uint8x16_t vQh[cColPairs][4];
      for (int vCp = 0; vCp < cColPairs; vCp++) {
        vQh[vCp][0] = vld1q_u8(vQhBase + 16 * vCp);
        vQh[vCp][1] = vld1q_u8(vQhBase + 16 * vCp + 64);
        vQh[vCp][2] = vld1q_u8(vQhBase + 16 * vCp + 128);
        vQh[vCp][3] = vld1q_u8(vQhBase + 16 * vCp + 192);
      }

      for (int vSb = 0; vSb < QK_K / 64; vSb++) {
        for (int vI = 0; vI < cColPairs; vI++) {
          vAccLo[vI] = vdupq_n_s32(0);
          vAccHi[vI] = vdupq_n_s32(0);
        }
        // Need scales for the low and high nibbles
        // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
        int16x8_t aQ5sbMins[2]; // int16 as its needed for bias_acc later
        int16x8_t aQ5sbScales[2];
        for (int vI = 0; vI < 2; vI++) {
          int8_t vAuxQ5sb[8];
          const int cOffset = vSb * 24 + vI * 12;
          fDecodeQKx86bitScales(&vQ5Ptr[vB].scales[cOffset], &aQ5sbMins[vI], vAuxQ5sb);
          aQ5sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ5sb));
        }

        const uint8_t *vQsBase = vQ5Ptr[vB].qs + vSb * QK_K;

        // Load the 64 quants from q8K duplicated to use vecdots with the interleaved columns
        const int8_t *vQ8Base = vQ8Ptr[vB].qs + vSb * 64;
        int8x16_t vQ8Qs[8];
        for (int vI = 0; vI < 8; vI++) {
          vQ8Qs[vI] = (int8x16_t)vld1q_dup_s64((const int64_t *)(vQ8Base + vI * 8));
        }

        // Q5s column pair loop unrolled
        {
          // Cols 01
          uint8x16_t vQs0 = vld1q_u8(vQsBase);
          uint8x16_t vQs1 = vld1q_u8(vQsBase + 64);
          uint8x16_t vQs2 = vld1q_u8(vQsBase + 128);
          uint8x16_t vQs3 = vld1q_u8(vQsBase + 192);

          uint8x16_t vHbitLo0 = vandq_u8(vQh[0][0], cMone);
          uint8x16_t vHbitLo1 = vandq_u8(vQh[0][1], cMone);
          uint8x16_t vHbitLo2 = vandq_u8(vQh[0][2], cMone);
          uint8x16_t vHbitLo3 = vandq_u8(vQh[0][3], cMone);
          uint8x16_t vHbitHi0 = vshlq_n_u8(vandq_u8(vQh[0][0], cMtwo), 3);
          uint8x16_t vHbitHi1 = vshlq_n_u8(vandq_u8(vQh[0][1], cMtwo), 3);
          uint8x16_t vHbitHi2 = vshlq_n_u8(vandq_u8(vQh[0][2], cMtwo), 3);
          uint8x16_t vHbitHi3 = vshlq_n_u8(vandq_u8(vQh[0][3], cMtwo), 3);

          vQh[0][0] = vshrq_n_u8(vQh[0][0], 2);
          vQh[0][1] = vshrq_n_u8(vQh[0][1], 2);
          vQh[0][2] = vshrq_n_u8(vQh[0][2], 2);
          vQh[0][3] = vshrq_n_u8(vQh[0][3], 2);

          vAccLo[0] =
              ggml_vdotq_s32(vAccLo[0], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs0, cM4b), vHbitLo0, 4)), vQ8Qs[0]);
          vAccLo[0] =
              ggml_vdotq_s32(vAccLo[0], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs1, cM4b), vHbitLo1, 4)), vQ8Qs[1]);
          vAccLo[0] =
              ggml_vdotq_s32(vAccLo[0], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs2, cM4b), vHbitLo2, 4)), vQ8Qs[2]);
          vAccLo[0] =
              ggml_vdotq_s32(vAccLo[0], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs3, cM4b), vHbitLo3, 4)), vQ8Qs[3]);
          vAccHi[0] = ggml_vdotq_s32(vAccHi[0], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs0, 4), vHbitHi0)), vQ8Qs[4]);
          vAccHi[0] = ggml_vdotq_s32(vAccHi[0], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs1, 4), vHbitHi1)), vQ8Qs[5]);
          vAccHi[0] = ggml_vdotq_s32(vAccHi[0], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs2, 4), vHbitHi2)), vQ8Qs[6]);
          vAccHi[0] = ggml_vdotq_s32(vAccHi[0], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs3, 4), vHbitHi3)), vQ8Qs[7]);

          // Cols 23
          vQs0 = vld1q_u8(vQsBase + 16);
          vQs1 = vld1q_u8(vQsBase + 80);
          vQs2 = vld1q_u8(vQsBase + 144);
          vQs3 = vld1q_u8(vQsBase + 208);

          vHbitLo0 = vandq_u8(vQh[1][0], cMone);
          vHbitLo1 = vandq_u8(vQh[1][1], cMone);
          vHbitLo2 = vandq_u8(vQh[1][2], cMone);
          vHbitLo3 = vandq_u8(vQh[1][3], cMone);
          vHbitHi0 = vshlq_n_u8(vandq_u8(vQh[1][0], cMtwo), 3);
          vHbitHi1 = vshlq_n_u8(vandq_u8(vQh[1][1], cMtwo), 3);
          vHbitHi2 = vshlq_n_u8(vandq_u8(vQh[1][2], cMtwo), 3);
          vHbitHi3 = vshlq_n_u8(vandq_u8(vQh[1][3], cMtwo), 3);

          vQh[1][0] = vshrq_n_u8(vQh[1][0], 2);
          vQh[1][1] = vshrq_n_u8(vQh[1][1], 2);
          vQh[1][2] = vshrq_n_u8(vQh[1][2], 2);
          vQh[1][3] = vshrq_n_u8(vQh[1][3], 2);

          vAccLo[1] =
              ggml_vdotq_s32(vAccLo[1], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs0, cM4b), vHbitLo0, 4)), vQ8Qs[0]);
          vAccLo[1] =
              ggml_vdotq_s32(vAccLo[1], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs1, cM4b), vHbitLo1, 4)), vQ8Qs[1]);
          vAccLo[1] =
              ggml_vdotq_s32(vAccLo[1], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs2, cM4b), vHbitLo2, 4)), vQ8Qs[2]);
          vAccLo[1] =
              ggml_vdotq_s32(vAccLo[1], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs3, cM4b), vHbitLo3, 4)), vQ8Qs[3]);
          vAccHi[1] = ggml_vdotq_s32(vAccHi[1], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs0, 4), vHbitHi0)), vQ8Qs[4]);
          vAccHi[1] = ggml_vdotq_s32(vAccHi[1], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs1, 4), vHbitHi1)), vQ8Qs[5]);
          vAccHi[1] = ggml_vdotq_s32(vAccHi[1], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs2, 4), vHbitHi2)), vQ8Qs[6]);
          vAccHi[1] = ggml_vdotq_s32(vAccHi[1], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs3, 4), vHbitHi3)), vQ8Qs[7]);

          // Cols 45
          vQs0 = vld1q_u8(vQsBase + 32);
          vQs1 = vld1q_u8(vQsBase + 96);
          vQs2 = vld1q_u8(vQsBase + 160);
          vQs3 = vld1q_u8(vQsBase + 224);

          vHbitLo0 = vandq_u8(vQh[2][0], cMone);
          vHbitLo1 = vandq_u8(vQh[2][1], cMone);
          vHbitLo2 = vandq_u8(vQh[2][2], cMone);
          vHbitLo3 = vandq_u8(vQh[2][3], cMone);
          vHbitHi0 = vshlq_n_u8(vandq_u8(vQh[2][0], cMtwo), 3);
          vHbitHi1 = vshlq_n_u8(vandq_u8(vQh[2][1], cMtwo), 3);
          vHbitHi2 = vshlq_n_u8(vandq_u8(vQh[2][2], cMtwo), 3);
          vHbitHi3 = vshlq_n_u8(vandq_u8(vQh[2][3], cMtwo), 3);

          vQh[2][0] = vshrq_n_u8(vQh[2][0], 2);
          vQh[2][1] = vshrq_n_u8(vQh[2][1], 2);
          vQh[2][2] = vshrq_n_u8(vQh[2][2], 2);
          vQh[2][3] = vshrq_n_u8(vQh[2][3], 2);

          vAccLo[2] =
              ggml_vdotq_s32(vAccLo[2], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs0, cM4b), vHbitLo0, 4)), vQ8Qs[0]);
          vAccLo[2] =
              ggml_vdotq_s32(vAccLo[2], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs1, cM4b), vHbitLo1, 4)), vQ8Qs[1]);
          vAccLo[2] =
              ggml_vdotq_s32(vAccLo[2], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs2, cM4b), vHbitLo2, 4)), vQ8Qs[2]);
          vAccLo[2] =
              ggml_vdotq_s32(vAccLo[2], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs3, cM4b), vHbitLo3, 4)), vQ8Qs[3]);
          vAccHi[2] = ggml_vdotq_s32(vAccHi[2], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs0, 4), vHbitHi0)), vQ8Qs[4]);
          vAccHi[2] = ggml_vdotq_s32(vAccHi[2], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs1, 4), vHbitHi1)), vQ8Qs[5]);
          vAccHi[2] = ggml_vdotq_s32(vAccHi[2], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs2, 4), vHbitHi2)), vQ8Qs[6]);
          vAccHi[2] = ggml_vdotq_s32(vAccHi[2], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs3, 4), vHbitHi3)), vQ8Qs[7]);

          // Cols 45
          vQs0 = vld1q_u8(vQsBase + 48);
          vQs1 = vld1q_u8(vQsBase + 112);
          vQs2 = vld1q_u8(vQsBase + 176);
          vQs3 = vld1q_u8(vQsBase + 240);

          vHbitLo0 = vandq_u8(vQh[3][0], cMone);
          vHbitLo1 = vandq_u8(vQh[3][1], cMone);
          vHbitLo2 = vandq_u8(vQh[3][2], cMone);
          vHbitLo3 = vandq_u8(vQh[3][3], cMone);
          vHbitHi0 = vshlq_n_u8(vandq_u8(vQh[3][0], cMtwo), 3);
          vHbitHi1 = vshlq_n_u8(vandq_u8(vQh[3][1], cMtwo), 3);
          vHbitHi2 = vshlq_n_u8(vandq_u8(vQh[3][2], cMtwo), 3);
          vHbitHi3 = vshlq_n_u8(vandq_u8(vQh[3][3], cMtwo), 3);

          vQh[3][0] = vshrq_n_u8(vQh[3][0], 2);
          vQh[3][1] = vshrq_n_u8(vQh[3][1], 2);
          vQh[3][2] = vshrq_n_u8(vQh[3][2], 2);
          vQh[3][3] = vshrq_n_u8(vQh[3][3], 2);

          vAccLo[3] =
              ggml_vdotq_s32(vAccLo[3], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs0, cM4b), vHbitLo0, 4)), vQ8Qs[0]);
          vAccLo[3] =
              ggml_vdotq_s32(vAccLo[3], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs1, cM4b), vHbitLo1, 4)), vQ8Qs[1]);
          vAccLo[3] =
              ggml_vdotq_s32(vAccLo[3], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs2, cM4b), vHbitLo2, 4)), vQ8Qs[2]);
          vAccLo[3] =
              ggml_vdotq_s32(vAccLo[3], vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQs3, cM4b), vHbitLo3, 4)), vQ8Qs[3]);
          vAccHi[3] = ggml_vdotq_s32(vAccHi[3], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs0, 4), vHbitHi0)), vQ8Qs[4]);
          vAccHi[3] = ggml_vdotq_s32(vAccHi[3], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs1, 4), vHbitHi1)), vQ8Qs[5]);
          vAccHi[3] = ggml_vdotq_s32(vAccHi[3], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs2, 4), vHbitHi2)), vQ8Qs[6]);
          vAccHi[3] = ggml_vdotq_s32(vAccHi[3], vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQs3, 4), vHbitHi3)), vQ8Qs[7]);
        }

        // Prepare bsum vectors for bias computation
        // Each pair of subblocks share the same bsums
        int16x4_t vBsumsVecLo = vdup_n_s16(vBsumsArr[2 * vSb + 0]);
        int16x4_t vBsumsVecHi = vdup_n_s16(vBsumsArr[2 * vSb + 1]);

        // Iterates over a pair of column pairs (4 columns) to use a single 128 register
        // p = 0 -> 0123  p2 -> 4567
        for (int vI = 0, vP = 0; vP < cColPairs; vI++, vP += 2) {
          int16x4_t vGroupScalesLo = vP == 0 ? vget_low_s16(aQ5sbScales[0]) : vget_high_s16(aQ5sbScales[0]);
          int16x4_t vGroupScalesHi = vP == 0 ? vget_low_s16(aQ5sbScales[1]) : vget_high_s16(aQ5sbScales[1]);
          int16x4_t vGroupMinsLo = vP == 0 ? vget_low_s16(aQ5sbMins[0]) : vget_high_s16(aQ5sbMins[0]);
          int16x4_t vGroupMinsHi = vP == 0 ? vget_low_s16(aQ5sbMins[1]) : vget_high_s16(aQ5sbMins[1]);
          float32x4_t vSbScale = vP == 0 ? vSbScale0 : vSbScale1;
          float32x4_t vSbMin = vP == 0 ? vSbMin0 : vSbMin1;

          // 0123 or 4567
          float32x4_t vSumf0 =
              vcvtq_f32_s32(vmulq_s32(vmovl_s16(vGroupScalesLo), vpaddq_s32(vAccLo[vP], vAccLo[vP + 1])));
          vAccF32[vI] = vfmaq_f32(vAccF32[vI], vSbScale, vSumf0);

          float32x4_t vSumf1 =
              vcvtq_f32_s32(vmulq_s32(vmovl_s16(vGroupScalesHi), vpaddq_s32(vAccHi[vP], vAccHi[vP + 1])));
          vAccF32[vI] = vfmaq_f32(vAccF32[vI], vSbScale, vSumf1);

          // FUSED BIAS: Compute and subtract bias immediately
          // bias = (bsums_lo * mins_lo + bsums_hi * mins_hi) * sb_min
          int32x4_t vBias = vmull_s16(vBsumsVecLo, vGroupMinsLo);
          vBias = vmlal_s16(vBias, vBsumsVecHi, vGroupMinsHi);
          float32x4_t vBiasF32 = vcvtq_f32_s32(vBias);
          vAccF32[vI] = vmlsq_f32(vAccF32[vI], vSbMin, vBiasF32);
        }
      } // for sb
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q5_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q6_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColGroups = cNcolsInterleaved / 4;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMaskLo = vdupq_n_u8(0x03);
  const uint8x16_t cMaskHi = vdupq_n_u8(0x30);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[2];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q6_Kx8 *GGML_RESTRICT vQ6Ptr = (const block_q6_Kx8 *)pVx + (vX * cNb);

    for (int vI = 0; vI < cColGroups; vI++) {
      vAccF32[vI] = vdupq_n_f32(0);
    }

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ6D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ6D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0 = vmulq_f32(vQ6D0, vQ8D);
      float32x4_t vSbScale1 = vmulq_f32(vQ6D1, vQ8D);

      int32x4_t vAcc[cColGroups];
      for (int vI = 0; vI < cColGroups; vI++) {
        vAcc[vI] = vdupq_n_s32(0);
      }

      // Load all 16 scales once and widen to int16 (Q6_K has 16 scales per block)
      // Reused for bias and dequantization later
      int16_t vQ6Scales[16 * 8];
      for (int vI = 0; vI < 16; vI++) {
        int16x8_t vScales = vmovl_s8(vld1_s8(vQ6Ptr[vB].scales + vI * 8));
        vst1q_s16(vQ6Scales + vI * 8, vScales);
      }

      // Compute bias per column using q8 bsums and preloaded scales to skip the -32 shift
      int32x4_t vBiasLo = vdupq_n_s32(0);
      int32x4_t vBiasHi = vdupq_n_s32(0);

      // Load bsums in chunks of 4 to process with vectorized operations
      for (int vI = 0; vI < 16; vI += 4) {
        int16x4_t vBsumsVec = vld1_s16(vQ8Ptr[vB].bsums + vI);
        int16x4_t vScalesLo0 = vld1_s16(vQ6Scales + (vI + 0) * 8);
        int16x4_t vScalesHi0 = vld1_s16(vQ6Scales + (vI + 0) * 8 + 4);
        int16x4_t vScalesLo1 = vld1_s16(vQ6Scales + (vI + 1) * 8);
        int16x4_t vScalesHi1 = vld1_s16(vQ6Scales + (vI + 1) * 8 + 4);
        int16x4_t vScalesLo2 = vld1_s16(vQ6Scales + (vI + 2) * 8);
        int16x4_t vScalesHi2 = vld1_s16(vQ6Scales + (vI + 2) * 8 + 4);
        int16x4_t vScalesLo3 = vld1_s16(vQ6Scales + (vI + 3) * 8);
        int16x4_t vScalesHi3 = vld1_s16(vQ6Scales + (vI + 3) * 8 + 4);

        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo0, vBsumsVec, 0);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi0, vBsumsVec, 0);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo1, vBsumsVec, 1);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi1, vBsumsVec, 1);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo2, vBsumsVec, 2);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi2, vBsumsVec, 2);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo3, vBsumsVec, 3);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi3, vBsumsVec, 3);
      }
      vBiasLo = vshlq_n_s32(vBiasLo, 5);
      vBiasHi = vshlq_n_s32(vBiasHi, 5);

      // Process two 128-value halves per superblock
      for (int vHalf = 0; vHalf < 2; vHalf++) {
        const uint8_t *vQlBase = vQ6Ptr[vB].ql + vHalf * 512;
        const uint8_t *vQhBase = vQ6Ptr[vB].qh + vHalf * 256;

        // A subblock (sb) is a set of weights that share the scale
        // Since q6_K scales are per 16 elements
        // num sbs -> 256 elements / (16 elements/scale * 2 elements/byte * 2 halves)
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          const int8_t *vQ8BaseL = vQ8Ptr[vB].qs + vHalf * 128 + vSb * 16;
          const int8_t *vQ8BaseH = vQ8BaseL + 64;

          // Load and duplicate q8 values (each register covers four interleaved columns of q6)
          int8x16_t vQ8L[4];
          int8x16_t vQ8H[4];
          for (int vI = 0; vI < 4; vI++) {
            vQ8L[vI] = (int8x16_t)vld1q_dup_s32((const int32_t *)(vQ8BaseL + vI * 4));
            vQ8H[vI] = (int8x16_t)vld1q_dup_s32((const int32_t *)(vQ8BaseH + vI * 4));
          }

          const int cQlOffBase = vSb * QK_K / 2;
          const int cQhOffBase = cQlOffBase & 255; // wraps after 256 bytes

          // Load 4 vectors at once (64 bytes each for ql_0, ql_1, qh_0, qh_1)
          uint8x16x4_t vQ6Ql0 = vld1q_u8_x4(vQlBase + cQlOffBase);
          uint8x16x4_t vQ6Ql1 = vld1q_u8_x4(vQlBase + cQlOffBase + 64);
          uint8x16x4_t vQ6Qh0 = vld1q_u8_x4(vQhBase + cQhOffBase);
          uint8x16x4_t vQ6Qh1 = vld1q_u8_x4(vQhBase + cQhOffBase + 64);

          // Adjust qh for subblocks 2 and 3 (shift right by 2)
          if (vSb > 1) {
            vQ6Qh0.val[0] = vshrq_n_u8(vQ6Qh0.val[0], 2);
            vQ6Qh0.val[1] = vshrq_n_u8(vQ6Qh0.val[1], 2);
            vQ6Qh0.val[2] = vshrq_n_u8(vQ6Qh0.val[2], 2);
            vQ6Qh0.val[3] = vshrq_n_u8(vQ6Qh0.val[3], 2);
            vQ6Qh1.val[0] = vshrq_n_u8(vQ6Qh1.val[0], 2);
            vQ6Qh1.val[1] = vshrq_n_u8(vQ6Qh1.val[1], 2);
            vQ6Qh1.val[2] = vshrq_n_u8(vQ6Qh1.val[2], 2);
            vQ6Qh1.val[3] = vshrq_n_u8(vQ6Qh1.val[3], 2);
          }

          const uint8x16_t aQ6Ql[8] = {vQ6Ql0.val[0], vQ6Ql0.val[1], vQ6Ql0.val[2], vQ6Ql0.val[3],
                                       vQ6Ql1.val[0], vQ6Ql1.val[1], vQ6Ql1.val[2], vQ6Ql1.val[3]};
          const uint8x16_t aQ6Qh[8] = {vQ6Qh0.val[0], vQ6Qh0.val[1], vQ6Qh0.val[2], vQ6Qh0.val[3],
                                       vQ6Qh1.val[0], vQ6Qh1.val[1], vQ6Qh1.val[2], vQ6Qh1.val[3]};

          // Process column groups (0-3, 4-7)
          for (int vG = 0; vG < cColGroups; vG++) {
            int32x4_t vSbAccL = vdupq_n_s32(0);
            int32x4_t vSbAccH = vdupq_n_s32(0);

            for (int vChunk = 0; vChunk < 4; vChunk++) {
              const int cIdx = vChunk * 2 + vG;

              const uint8x16_t cQ6QsL = aQ6Ql[cIdx];
              const uint8x16_t cQ6QsH = aQ6Qh[cIdx];

              // Extract high 2 bits for upper nibble reconstruction
              const uint8x16_t cQ6QsHh = vandq_u8(cQ6QsH, cMaskHi);

              // q6 = (low4 | high2<<4), without -32 bias (handled via bsums)
              const int8x16_t cQ6L =
                  vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ6QsL, cM4b), vandq_u8(cQ6QsH, cMaskLo), 4));
              const int8x16_t cQ6H = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ6QsL, 4), cQ6QsHh));

              vSbAccL = vdotq_s32(vSbAccL, cQ6L, vQ8L[vChunk]);
              vSbAccH = vdotq_s32(vSbAccH, cQ6H, vQ8H[vChunk]);
            }

            const int cScaleIdxL = vHalf * 8 + vSb;
            const int cScaleIdxH = vHalf * 8 + vSb + 4;

            const int32x4_t cScaleVecL = vmovl_s16(vld1_s16(vQ6Scales + cScaleIdxL * 8 + vG * 4));
            const int32x4_t cScaleVecH = vmovl_s16(vld1_s16(vQ6Scales + cScaleIdxH * 8 + vG * 4));

            vAcc[vG] = vmlaq_s32(vAcc[vG], vSbAccL, cScaleVecL);
            vAcc[vG] = vmlaq_s32(vAcc[vG], vSbAccH, cScaleVecH);
          }
        }
      } // for half

      // Bias correction
      vAcc[0] = vsubq_s32(vAcc[0], vBiasLo);
      vAcc[1] = vsubq_s32(vAcc[1], vBiasHi);

      // Apply superblock scale (no mins for q6_K)
      // acc[g] has [c0, c1, c2, c3]
      float32x4_t vW0123 = vmulq_f32(vcvtq_f32_s32(vAcc[0]), vSbScale0);
      float32x4_t vW4567 = vmulq_f32(vcvtq_f32_s32(vAcc[1]), vSbScale1);

      vAccF32[0] = vaddq_f32(vAccF32[0], vW0123);
      vAccF32[1] = vaddq_f32(vAccF32[1], vW4567);
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q6_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q6_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cColPairs = cNcolsInterleaved / 2;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMaskLo = vdupq_n_u8(0x03);
  const uint8x16_t cMaskHi = vdupq_n_u8(0x30);

  // 1x8 tile = 2 x 4
  float32x4_t vAccF32[2];

  const block_q8_K *GGML_RESTRICT vQ8Ptr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q6_Kx8 *GGML_RESTRICT vQ6Ptr = (const block_q6_Kx8 *)pVx + (vX * cNb);

    vAccF32[0] = vdupq_n_f32(0);
    vAccF32[1] = vdupq_n_f32(0);

    for (int vB = 0; vB < cNb; vB++) {
      float32x4_t vQ6D0 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d));     // d0 d1 d2 d3
      float32x4_t vQ6D1 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d + 4)); // d4 d5 d6 d7
      float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d);
      float32x4_t vSbScale0 = vmulq_f32(vQ6D0, vQ8D);
      float32x4_t vSbScale1 = vmulq_f32(vQ6D1, vQ8D);

      int32x2_t vAcc[cColPairs];
      for (int vI = 0; vI < cColPairs; vI++) {
        vAcc[vI] = vdup_n_s32(0);
      }

      // Load all 16 scales once and widen to int16 (Q6_K has 16 scales per block)
      // Reused for bias and dequantization later
      int16_t vQ6Scales[16 * 8];
      for (int vI = 0; vI < 16; vI++) {
        int16x8_t vScales = vmovl_s8(vld1_s8(vQ6Ptr[vB].scales + vI * 8));
        vst1q_s16(vQ6Scales + vI * 8, vScales);
      }

      // Compute bias per column using q8 bsums and preloaded scales to skip the -32 shift
      int32x4_t vBiasLo = vdupq_n_s32(0);
      int32x4_t vBiasHi = vdupq_n_s32(0);

      // Load bsums in chunks of 4 to process with vectorized operations
      for (int vI = 0; vI < 16; vI += 4) {
        int16x4_t vBsumsVec = vld1_s16(vQ8Ptr[vB].bsums + vI);
        int16x4_t vScalesLo0 = vld1_s16(vQ6Scales + (vI + 0) * 8);
        int16x4_t vScalesHi0 = vld1_s16(vQ6Scales + (vI + 0) * 8 + 4);
        int16x4_t vScalesLo1 = vld1_s16(vQ6Scales + (vI + 1) * 8);
        int16x4_t vScalesHi1 = vld1_s16(vQ6Scales + (vI + 1) * 8 + 4);
        int16x4_t vScalesLo2 = vld1_s16(vQ6Scales + (vI + 2) * 8);
        int16x4_t vScalesHi2 = vld1_s16(vQ6Scales + (vI + 2) * 8 + 4);
        int16x4_t vScalesLo3 = vld1_s16(vQ6Scales + (vI + 3) * 8);
        int16x4_t vScalesHi3 = vld1_s16(vQ6Scales + (vI + 3) * 8 + 4);

        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo0, vBsumsVec, 0);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi0, vBsumsVec, 0);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo1, vBsumsVec, 1);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi1, vBsumsVec, 1);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo2, vBsumsVec, 2);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi2, vBsumsVec, 2);
        vBiasLo = vmlal_lane_s16(vBiasLo, vScalesLo3, vBsumsVec, 3);
        vBiasHi = vmlal_lane_s16(vBiasHi, vScalesHi3, vBsumsVec, 3);
      }
      vBiasLo = vshlq_n_s32(vBiasLo, 5);
      vBiasHi = vshlq_n_s32(vBiasHi, 5);

      // Process two 128-value halves per superblock
      for (int vHalf = 0; vHalf < 2; vHalf++) {
        const uint8_t *vQlBase = vQ6Ptr[vB].ql + vHalf * 512;
        const uint8_t *vQhBase = vQ6Ptr[vB].qh + vHalf * 256;

        // A subblock (sb) is a set of weights that share the scale
        // Since q6_K scales are per 16 elements
        // num sbs -> 256 elements / (16 elements/scale * 2 elements/byte * 2 halves)
        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          const int8_t *vQ8BaseL = vQ8Ptr[vB].qs + vHalf * 128 + vSb * 16;
          const int8_t *vQ8BaseH = vQ8BaseL + 64;

          // Load and duplicate q8 values (each register covers two interleaved columns of q6)
          int8x16_t vQ8L[2];
          int8x16_t vQ8H[2];
          for (int vI = 0; vI < 2; vI++) {
            vQ8L[vI] = (int8x16_t)vld1q_dup_s64((const int64_t *)(vQ8BaseL + vI * 8));
            vQ8H[vI] = (int8x16_t)vld1q_dup_s64((const int64_t *)(vQ8BaseH + vI * 8));
          }

          const int cQlOffBase = vSb * QK_K / 2;
          const int cQhOffBase = cQlOffBase & 255; // wraps after 256 bytes

          // Load 4 vectors at once (64 bytes each for ql_0, ql_1, qh_0, qh_1)
          uint8x16x4_t vQ6Ql0 = vld1q_u8_x4(vQlBase + cQlOffBase);
          uint8x16x4_t vQ6Ql1 = vld1q_u8_x4(vQlBase + cQlOffBase + 64);
          uint8x16x4_t vQ6Qh0 = vld1q_u8_x4(vQhBase + cQhOffBase);
          uint8x16x4_t vQ6Qh1 = vld1q_u8_x4(vQhBase + cQhOffBase + 64);

          // Adjust qh for subblocks 2 and 3 (shift right by 2)
          if (vSb > 1) {
            vQ6Qh0.val[0] = vshrq_n_u8(vQ6Qh0.val[0], 2);
            vQ6Qh0.val[1] = vshrq_n_u8(vQ6Qh0.val[1], 2);
            vQ6Qh0.val[2] = vshrq_n_u8(vQ6Qh0.val[2], 2);
            vQ6Qh0.val[3] = vshrq_n_u8(vQ6Qh0.val[3], 2);
            vQ6Qh1.val[0] = vshrq_n_u8(vQ6Qh1.val[0], 2);
            vQ6Qh1.val[1] = vshrq_n_u8(vQ6Qh1.val[1], 2);
            vQ6Qh1.val[2] = vshrq_n_u8(vQ6Qh1.val[2], 2);
            vQ6Qh1.val[3] = vshrq_n_u8(vQ6Qh1.val[3], 2);
          }

          // Process column pairs (0-1, 2-3, 4-5, 6-7)
          for (int vCp = 0; vCp < cColPairs; vCp++) {
            const uint8x16_t cQ6QsCp0L = vQ6Ql0.val[vCp];
            const uint8x16_t cQ6QsCp1L = vQ6Ql1.val[vCp];
            const uint8x16_t cQ6QsCp0H = vQ6Qh0.val[vCp];
            const uint8x16_t cQ6QsCp1H = vQ6Qh1.val[vCp];

            // Extract high 2 bits for upper nibble reconstruction
            const uint8x16_t cQ6QsCp0Hh = vandq_u8(cQ6QsCp0H, cMaskHi);
            const uint8x16_t cQ6QsCp1Hh = vandq_u8(cQ6QsCp1H, cMaskHi);

            // q6 = (low4 | high2<<4), without -32 bias (handled via bsums)
            const int8x16_t cQ6L0 =
                vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ6QsCp0L, cM4b), vandq_u8(cQ6QsCp0H, cMaskLo), 4));
            const int8x16_t cQ6L1 =
                vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ6QsCp1L, cM4b), vandq_u8(cQ6QsCp1H, cMaskLo), 4));
            const int8x16_t cQ6H0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ6QsCp0L, 4), cQ6QsCp0Hh));
            const int8x16_t cQ6H1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ6QsCp1L, 4), cQ6QsCp1Hh));

            int32x4_t vSbAccL = vdupq_n_s32(0);
            vSbAccL = vdotq_s32(vSbAccL, cQ6L0, vQ8L[0]);
            vSbAccL = vdotq_s32(vSbAccL, cQ6L1, vQ8L[1]);

            int32x4_t vSbAccH = vdupq_n_s32(0);
            vSbAccH = vdotq_s32(vSbAccH, cQ6H0, vQ8H[0]);
            vSbAccH = vdotq_s32(vSbAccH, cQ6H1, vQ8H[1]);

            // Pairwise add to get per-column sums: [col0, col1]
            int32x2_t vSumL = vpadd_s32(vget_low_s32(vSbAccL), vget_high_s32(vSbAccL));
            int32x2_t vSumH = vpadd_s32(vget_low_s32(vSbAccH), vget_high_s32(vSbAccH));

            const int cScaleIdxL = vHalf * 8 + vSb;
            const int cScaleIdxH = vHalf * 8 + vSb + 4;

            // Access scales using array indexing (scales are interleaved by column)
            const int32x2_t cScaleVecL = {(int32_t)vQ6Scales[cScaleIdxL * 8 + vCp * 2],
                                          (int32_t)vQ6Scales[cScaleIdxL * 8 + vCp * 2 + 1]};
            const int32x2_t cScaleVecH = {(int32_t)vQ6Scales[cScaleIdxH * 8 + vCp * 2],
                                          (int32_t)vQ6Scales[cScaleIdxH * 8 + vCp * 2 + 1]};

            // Accumulate scaled results
            vAcc[vCp] = vmla_s32(vAcc[vCp], vSumL, cScaleVecL);
            vAcc[vCp] = vmla_s32(vAcc[vCp], vSumH, cScaleVecH);
          }
        }
      } // for half

      // Bias correction
      vAcc[0] = vsub_s32(vAcc[0], vget_low_s32(vBiasLo));
      vAcc[1] = vsub_s32(vAcc[1], vget_high_s32(vBiasLo));
      vAcc[2] = vsub_s32(vAcc[2], vget_low_s32(vBiasHi));
      vAcc[3] = vsub_s32(vAcc[3], vget_high_s32(vBiasHi));

      // Apply superblock scale (no mins for q6_K)
      // acc[cp] has [c0, c1]
      float32x2_t vW01 = vmul_f32(vcvt_f32_s32(vAcc[0]), vget_low_f32(vSbScale0));
      float32x2_t vW23 = vmul_f32(vcvt_f32_s32(vAcc[1]), vget_high_f32(vSbScale0));
      float32x2_t vW45 = vmul_f32(vcvt_f32_s32(vAcc[2]), vget_low_f32(vSbScale1));
      float32x2_t vW67 = vmul_f32(vcvt_f32_s32(vAcc[3]), vget_high_f32(vSbScale1));

      vAccF32[0] = vaddq_f32(vAccF32[0], vcombine_f32(vW01, vW23));
      vAccF32[1] = vaddq_f32(vAccF32[1], vcombine_f32(vW45, vW67));
    } // for b

    int vBase = vX * cNcolsInterleaved;
    vst1q_f32(pS + vBase, vAccF32[0]);
    vst1q_f32(pS + vBase + 4, vAccF32[1]);
  } // for x
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q6_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q8_0_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx;

  for (int vC = 0; vC < pNc; vC += cNcolsInterleaved) {
    const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
    float32x4_t vAcc = vdupq_n_f32(0);
    for (int vB = 0; vB < cNb; vB++) {
      int8x16x4_t vBLow = vld1q_s8_x4((const int8_t *)vBPtr->qs);
      int8x16x4_t vBHigh = vld1q_s8_x4((const int8_t *)vBPtr->qs + 64);
      float16x4_t vBd = vld1_f16((const __fp16 *)vBPtr->d);

      int8x16x2_t vA = vld1q_s8_x2(vAPtr->qs);
      float16x4_t vAd = vld1_dup_f16((const __fp16 *)&vAPtr->d);

      int32x4_t vRet = vdupq_n_s32(0);

      vRet = vdotq_laneq_s32(vRet, vBLow.val[0], vA.val[0], 0);
      vRet = vdotq_laneq_s32(vRet, vBLow.val[1], vA.val[0], 1);
      vRet = vdotq_laneq_s32(vRet, vBLow.val[2], vA.val[0], 2);
      vRet = vdotq_laneq_s32(vRet, vBLow.val[3], vA.val[0], 3);

      vRet = vdotq_laneq_s32(vRet, vBHigh.val[0], vA.val[1], 0);
      vRet = vdotq_laneq_s32(vRet, vBHigh.val[1], vA.val[1], 1);
      vRet = vdotq_laneq_s32(vRet, vBHigh.val[2], vA.val[1], 2);
      vRet = vdotq_laneq_s32(vRet, vBHigh.val[3], vA.val[1], 3);

      vAcc = vfmaq_f32(vAcc, vcvtq_f32_s32(vRet), vmulq_f32(vcvt_f32_f16(vAd), vcvt_f32_f16(vBd)));
      vAPtr++;
      vBPtr++;
    }
    vst1q_f32(pS, vAcc);
    pS += cNcolsInterleaved;
  }
  return;

#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q8_0_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemv_q8_0_4x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx;

  for (int vC = 0; vC < pNc; vC += cNcolsInterleaved) {
    const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
    float32x4_t vAcc = vdupq_n_f32(0);

    for (int vB = 0; vB < cNb; vB++) {
      int8x16x4_t vBLow = vld1q_s8_x4((const int8_t *)vBPtr->qs);
      int8x16x4_t vBHigh = vld1q_s8_x4((const int8_t *)vBPtr->qs + 64);
      float16x4_t vBd = vld1_f16((const __fp16 *)vBPtr->d);

      int8x8x4_t vAChunks = vld1_s8_x4(vAPtr->qs);
      int8x16_t vA0 = vcombine_s8(vAChunks.val[0], vAChunks.val[0]);
      int8x16_t vA1 = vcombine_s8(vAChunks.val[1], vAChunks.val[1]);
      int8x16_t vA2 = vcombine_s8(vAChunks.val[2], vAChunks.val[2]);
      int8x16_t vA3 = vcombine_s8(vAChunks.val[3], vAChunks.val[3]);
      float16x4_t vAd = vld1_dup_f16((const __fp16 *)&vAPtr->d);

      int32x4_t vRet0 = vdupq_n_s32(0);
      int32x4_t vRet1 = vdupq_n_s32(0);

      // 0..7
      vRet0 = vdotq_s32(vRet0, vBLow.val[0], vA0);
      vRet1 = vdotq_s32(vRet1, vBLow.val[1], vA0);
      // 8..15
      vRet0 = vdotq_s32(vRet0, vBLow.val[2], vA1);
      vRet1 = vdotq_s32(vRet1, vBLow.val[3], vA1);
      // 16..23
      vRet0 = vdotq_s32(vRet0, vBHigh.val[0], vA2);
      vRet1 = vdotq_s32(vRet1, vBHigh.val[1], vA2);
      // 24..31
      vRet0 = vdotq_s32(vRet0, vBHigh.val[2], vA3);
      vRet1 = vdotq_s32(vRet1, vBHigh.val[3], vA3);

      int32x4_t vRet = vpaddq_s32(vRet0, vRet1);

      vAcc = vfmaq_f32(vAcc, vcvtq_f32_s32(vRet), vmulq_f32(vcvt_f32_f16(vAd), vcvt_f32_f16(vBd)));
      vAPtr++;
      vBPtr++;
    }
    vst1q_f32(pS, vAcc);
    pS += cNcolsInterleaved;
  }
  return;

#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemv_q8_0_4x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q4_0_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int nr, int nc) {
  const int cQk = QK8_0;
  const int nb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(nr % 4 == 0);
  assert(nc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const void *b_ptr = pVx;
  const void *a_ptr = pVy;
  float *res_ptr = pS;
  size_t res_stride = pBs * sizeof(float);

  __asm__ __volatile__("mov x10, %x[nr]\n"
                       "mov x9, #0x88\n"
                       "cmp x10, #0x10\n"
                       "mul x9, %x[nb], x9\n"
                       "blt 4f\n"
                       "1:" // Row loop
                       "add x28, %x[b_ptr], #0x8\n"
                       "mov x27, %x[nc]\n"
                       "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
                       "2:" // Column loop
                       "add x25, %x[a_ptr], #0x8\n"
                       "movi v15.16b, #0x0\n"
                       "movi v19.16b, #0x0\n"
                       "mov x24, %x[nb]\n"
                       "add x23, x25, x9\n"
                       "movi v18.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "add x22, x23, x9\n"
                       "movi v11.16b, #0x0\n"
                       "movi v13.16b, #0x0\n"
                       "add x21, x22, x9\n"
                       "movi v23.16b, #0x0\n"
                       "movi v16.16b, #0x0\n"
                       "movi v25.16b, #0x0\n"
                       "movi v7.16b, #0x0\n"
                       "movi v0.16b, #0x0\n"
                       "movi v4.16b, #0x0\n"
                       "movi v5.16b, #0x0\n"
                       "movi v21.16b, #0x0\n"
                       "movi v8.16b, #0x0\n"
                       "movi v1.16b, #0x0\n"
                       "3:" // Block loop
                       "ldr q3, [x28, #0x0]\n"
                       "ldr q31, [x25, #0x0]\n"
                       "movi v28.16b, #0x4\n"
                       "movi v10.4s, #0x0\n"
                       "ldr q22, [x28, #0x10]\n"
                       "ldr q6, [x25, #0x10]\n"
                       "movi v29.4s, #0x0\n"
                       "movi v9.4s, #0x0\n"
                       "ldr q27, [x28, #0x20]\n"
                       "ldr q30, [x28, #0x30]\n"
                       "movi v20.4s, #0x0\n"
                       "movi v24.16b, #0xf0\n"
                       "ldr d2, [x25, #-0x8]\n"
                       "ldr d26, [x23, #-0x8]\n"
                       "sshl v12.16b, v3.16b, v28.16b\n"
                       "sub x20, x28, #0x8\n"
                       "ldr d17, [x20, #0x0]\n"
                       "and v3.16b, v3.16b, v24.16b\n"
                       "subs x24, x24, #0x1\n"
                       "add x28, x28, #0x48\n"
                       ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
                       ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
                       ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
                       ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
                       "sshl v31.16b, v22.16b, v28.16b\n"
                       "and v22.16b, v22.16b, v24.16b\n"
                       "fcvtl v17.4s, v17.4h\n"
                       "fcvtl v2.4s, v2.4h\n"
                       "fcvtl v26.4s, v26.4h\n"
                       ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
                       ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
                       ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
                       ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
                       "sshl v6.16b, v27.16b, v28.16b\n"
                       "sshl v28.16b, v30.16b, v28.16b\n"
                       "and v27.16b, v27.16b, v24.16b\n"
                       "and v30.16b, v30.16b, v24.16b\n"
                       "ldr q24, [x25, #0x20]\n"
                       ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
                       ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
                       "ldr q24, [x25, #0x30]\n"
                       ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
                       ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
                       ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
                       "ldr q24, [x25, #0x40]\n"
                       ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
                       ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
                       "ldr q24, [x25, #0x50]\n"
                       ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
                       ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
                       ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
                       "ldr q24, [x25, #0x60]\n"
                       ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
                       ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
                       ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
                       "ldr q24, [x25, #0x70]\n"
                       "add x25, x25, #0x88\n"
                       ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
                       ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
                       ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
                       "fmul v24.4s, v17.4s, v2.s[0]\n"
                       "scvtf v10.4s, v10.4s, #0x4\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v15.4s, v10.4s, v24.4s\n"
                       "ldr q24, [x23, #0x0]\n"
                       "fmul v10.4s, v17.4s, v2.s[1]\n"
                       "fmla v19.4s, v29.4s, v10.4s\n"
                       "ldr q10, [x23, #0x10]\n"
                       "fmul v29.4s, v17.4s, v2.s[2]\n"
                       "fmul v2.4s, v17.4s, v2.s[3]\n"
                       "fmla v18.4s, v9.4s, v29.4s\n"
                       "movi v9.4s, #0x0\n"
                       "movi v29.4s, #0x0\n"
                       ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
                       "fmla v14.4s, v20.4s, v2.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v2.4s, #0x0\n"
                       ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
                       "ldr q24, [x23, #0x20]\n"
                       ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
                       ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
                       ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
                       ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
                       "ldr q10, [x23, #0x30]\n"
                       ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
                       ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
                       "ldr q24, [x23, #0x40]\n"
                       ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
                       ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
                       ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
                       ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
                       "ldr q10, [x23, #0x50]\n"
                       ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
                       ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
                       "ldr q24, [x23, #0x60]\n"
                       ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
                       ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
                       ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
                       ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
                       "ldr q10, [x23, #0x70]\n"
                       "add x23, x23, #0x88\n"
                       ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
                       ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
                       ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
                       "ldr q24, [x22, #0x0]\n"
                       ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
                       ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
                       ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
                       ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
                       "fmul v10.4s, v17.4s, v26.s[0]\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v2.4s, v2.4s, #0x4\n"
                       "fmla v11.4s, v9.4s, v10.4s\n"
                       "ldr q9, [x22, #0x10]\n"
                       "fmul v10.4s, v17.4s, v26.s[1]\n"
                       "fmla v13.4s, v29.4s, v10.4s\n"
                       "ldr d29, [x22, #-0x8]\n"
                       "fmul v10.4s, v17.4s, v26.s[2]\n"
                       "fmul v26.4s, v17.4s, v26.s[3]\n"
                       "fcvtl v29.4s, v29.4h\n"
                       "fmla v23.4s, v20.4s, v10.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v10.4s, #0x0\n"
                       "fmla v16.4s, v2.4s, v26.4s\n"
                       "movi v26.4s, #0x0\n"
                       "movi v2.4s, #0x0\n"
                       ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
                       ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
                       "ldr q24, [x22, #0x20]\n"
                       ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
                       ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
                       ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
                       "ldr q9, [x22, #0x30]\n"
                       ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
                       ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
                       "ldr q24, [x22, #0x40]\n"
                       ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
                       ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
                       ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
                       "ldr q9, [x22, #0x50]\n"
                       ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
                       ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
                       "ldr q24, [x22, #0x60]\n"
                       ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
                       ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
                       ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
                       "ldr q9, [x22, #0x70]\n"
                       "add x22, x22, #0x88\n"
                       ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
                       ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
                       ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
                       "ldr q24, [x21, #0x0]\n"
                       ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
                       ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
                       ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
                       "fmul v9.4s, v17.4s, v29.s[0]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v10.4s, v10.4s, #0x4\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "scvtf v2.4s, v2.4s, #0x4\n"
                       "fmla v25.4s, v20.4s, v9.4s\n"
                       "ldr q9, [x21, #0x10]\n"
                       "fmul v20.4s, v17.4s, v29.s[1]\n"
                       "fmla v7.4s, v10.4s, v20.4s\n"
                       "ldr d20, [x21, #-0x8]\n"
                       "fmul v10.4s, v17.4s, v29.s[2]\n"
                       "fmul v29.4s, v17.4s, v29.s[3]\n"
                       "fcvtl v20.4s, v20.4h\n"
                       "fmla v0.4s, v26.4s, v10.4s\n"
                       "movi v26.4s, #0x0\n"
                       "movi v10.4s, #0x0\n"
                       "fmla v4.4s, v2.4s, v29.4s\n"
                       "movi v2.4s, #0x0\n"
                       "movi v29.4s, #0x0\n"
                       ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
                       ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
                       "ldr q12, [x21, #0x20]\n"
                       "fmul v24.4s, v17.4s, v20.s[0]\n"
                       ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
                       ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
                       ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
                       "ldr q9, [x21, #0x30]\n"
                       "fmul v31.4s, v17.4s, v20.s[1]\n"
                       ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
                       ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
                       ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
                       ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
                       "ldr q12, [x21, #0x40]\n"
                       "fmul v6.4s, v17.4s, v20.s[2]\n"
                       "fmul v20.4s, v17.4s, v20.s[3]\n"
                       ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
                       ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
                       ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
                       "ldr q9, [x21, #0x50]\n"
                       ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
                       ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
                       ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
                       ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
                       "ldr q12, [x21, #0x60]\n"
                       ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
                       ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
                       ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
                       ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
                       "ldr q17, [x21, #0x70]\n"
                       "add x21, x21, #0x88\n"
                       ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
                       ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
                       ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
                       ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
                       ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
                       ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
                       ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
                       ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "scvtf v10.4s, v10.4s, #0x4\n"
                       "fmla v5.4s, v26.4s, v24.4s\n"
                       "scvtf v2.4s, v2.4s, #0x4\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "fmla v21.4s, v10.4s, v31.4s\n"
                       "fmla v8.4s, v2.4s, v6.4s\n"
                       "fmla v1.4s, v29.4s, v20.4s\n"
                       "bgt 3b\n"
                       "mov x20, %x[res_ptr]\n"
                       "subs x27, x27, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "str q15, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q19, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q18, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q14, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q11, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q13, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q23, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q16, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q25, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q7, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q0, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q4, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q5, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q21, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q8, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q1, [x20, #0x0]\n"
                       "bne 2b\n"
                       "mov x20, #0x4\n"
                       "sub x10, x10, #0x10\n"
                       "cmp x10, #0x10\n"
                       "mov %x[res_ptr], x26\n"
                       "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
                       "bge 1b\n"
                       "4:" // Row loop skip
                       "cbz x10, 9f\n"
                       "5:" // Row tail: Row loop
                       "add x24, %x[b_ptr], #0x8\n"
                       "mov x23, %x[nc]\n"
                       "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
                       "6:" // Row tail: Column loop
                       "movi v15.16b, #0x0\n"
                       "movi v19.16b, #0x0\n"
                       "add x25, %x[a_ptr], #0x8\n"
                       "mov x21, %x[nb]\n"
                       "movi v18.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "7:" // Row tail: Block loop
                       "ldr q7, [x24, #0x0]\n"
                       "ldr q5, [x25, #0x0]\n"
                       "movi v9.16b, #0x4\n"
                       "movi v4.4s, #0x0\n"
                       "ldr q3, [x24, #0x10]\n"
                       "ldr q2, [x25, #0x10]\n"
                       "movi v1.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       "ldr q13, [x24, #0x20]\n"
                       "ldr q31, [x25, #0x20]\n"
                       "movi v30.4s, #0x0\n"
                       "movi v29.16b, #0xf0\n"
                       "ldr q28, [x24, #0x30]\n"
                       "ldr q27, [x25, #0x30]\n"
                       "sshl v20.16b, v7.16b, v9.16b\n"
                       "sub x20, x24, #0x8\n"
                       "ldr q26, [x25, #0x40]\n"
                       "ldr q25, [x25, #0x50]\n"
                       "sshl v17.16b, v3.16b, v9.16b\n"
                       "and v7.16b, v7.16b, v29.16b\n"
                       "ldr q24, [x25, #0x60]\n"
                       "ldr q16, [x25, #0x70]\n"
                       "sshl v22.16b, v13.16b, v9.16b\n"
                       "and v3.16b, v3.16b, v29.16b\n"
                       "ldr d21, [x20, #0x0]\n"
                       "ldr d12, [x25, #-0x8]\n"
                       ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
                       ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
                       ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
                       ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
                       "sshl v9.16b, v28.16b, v9.16b\n"
                       "subs x21, x21, #0x1\n"
                       "and v13.16b, v13.16b, v29.16b\n"
                       "and v28.16b, v28.16b, v29.16b\n"
                       "add x25, x25, #0x88\n"
                       "add x24, x24, #0x48\n"
                       "fcvtl v21.4s, v21.4h\n"
                       "fcvtl v12.4s, v12.4h\n"
                       ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
                       ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
                       ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
                       ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
                       "fmul v11.4s, v21.4s, v12.s[0]\n"
                       "fmul v23.4s, v21.4s, v12.s[1]\n"
                       "fmul v17.4s, v21.4s, v12.s[2]\n"
                       ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
                       "fmul v6.4s, v21.4s, v12.s[3]\n"
                       ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
                       ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
                       ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
                       ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
                       ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
                       ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
                       ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
                       ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
                       ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
                       ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
                       ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
                       ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
                       ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
                       ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
                       ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
                       ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
                       ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
                       ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
                       ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
                       ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
                       ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
                       ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
                       ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
                       "scvtf v4.4s, v4.4s, #0x4\n"
                       "scvtf v1.4s, v1.4s, #0x4\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "fmla v15.4s, v4.4s, v11.4s\n"
                       "scvtf v30.4s, v30.4s, #0x4\n"
                       "fmla v19.4s, v1.4s, v23.4s\n"
                       "fmla v18.4s, v0.4s, v17.4s\n"
                       "fmla v14.4s, v30.4s, v6.4s\n"
                       "bgt 7b\n"
                       "mov x20, %x[res_ptr]\n"
                       "cmp x10, #0x1\n"
                       "str q15, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x2\n"
                       "str q19, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x3\n"
                       "str q18, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "str q14, [x20, #0x0]\n"
                       "8:" // Row tail: Accumulator store skip
                       "subs x23, x23, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "bne 6b\n"
                       "subs x10, x10, #0x4\n"
                       "add %x[a_ptr], %x[a_ptr], x9\n"
                       "mov %x[res_ptr], x22\n"
                       "bgt 5b\n"
                       "9:" // Row tail: Row loop skip
                       : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                       : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
                       : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                         "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                         "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
                         "x24", "x25", "x26", "x27", "x28");
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
  ggml_gemm_q4_0_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, nr, nc);
}

void ggml_gemm_q4_0_4x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int nr, int nc) {
  const int cQk = QK8_0;
  const int nb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(nr % 4 == 0);
  assert(nc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_MATMUL_INT8)
  const void *b_ptr = pVx;
  const void *a_ptr = pVy;
  float *res_ptr = pS;
  size_t res_stride = pBs * sizeof(float);

  __asm__ __volatile__("mov x10, %x[nr]\n"
                       "mov x9, #0x88\n"
                       "cmp x10, #0x10\n"
                       "mul x9, %x[nb], x9\n"
                       "blt 4f\n"
                       "1:" // Row loop
                       "add x28, %x[b_ptr], #0x8\n"
                       "mov x27, %x[nc]\n"
                       "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
                       "2:" // Column loop
                       "add x25, %x[a_ptr], #0x8\n"
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "mov x24, %x[nb]\n"
                       "add x23, x25, x9\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "add x22, x23, x9\n"
                       "movi v11.16b, #0x0\n"
                       "movi v13.16b, #0x0\n"
                       "add x21, x22, x9\n"
                       "movi v22.16b, #0x0\n"
                       "movi v23.16b, #0x0\n"
                       "movi v25.16b, #0x0\n"
                       "movi v5.16b, #0x0\n"
                       "movi v7.16b, #0x0\n"
                       "movi v4.16b, #0x0\n"
                       "movi v6.16b, #0x0\n"
                       "movi v30.16b, #0x0\n"
                       "movi v24.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "3:" // Block loop
                       "ldr q21, [x28, #0x0]\n"
                       "ldr q16, [x28, #0x10]\n"
                       "movi v1.16b, #0x4\n"
                       "movi v19.4s, #0x0\n"
                       "ldr q27, [x25, #0x0]\n"
                       "ldr q15, [x25, #0x10]\n"
                       "movi v26.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       "ldr q29, [x28, #0x20]\n"
                       "ldr q3, [x28, #0x30]\n"
                       "movi v17.4s, #0x0\n"
                       "movi v0.16b, #0xf0\n"
                       "ldr d20, [x25, #-0x8]\n"
                       "ldr d9, [x23, #-0x8]\n"
                       "sshl v8.16b, v21.16b, v1.16b\n"
                       "sshl v31.16b, v16.16b, v1.16b\n"
                       "and v21.16b, v21.16b, v0.16b\n"
                       "and v16.16b, v16.16b, v0.16b\n"
                       "sub x20, x28, #0x8\n"
                       "subs x24, x24, #0x1\n"
                       "add x28, x28, #0x48\n"
                       ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
                       ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
                       "ldr q27, [x25, #0x20]\n"
                       ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
                       ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
                       "sshl v15.16b, v29.16b, v1.16b\n"
                       "sshl v1.16b, v3.16b, v1.16b\n"
                       "and v29.16b, v29.16b, v0.16b\n"
                       "and v3.16b, v3.16b, v0.16b\n"
                       "ldr q0, [x25, #0x30]\n"
                       "fcvtl v20.4s, v20.4h\n"
                       ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
                       "fcvtl v9.4s, v9.4h\n"
                       ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
                       "ldr q27, [x25, #0x40]\n"
                       ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
                       ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
                       "ldr q0, [x25, #0x50]\n"
                       ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
                       ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
                       "ldr q27, [x25, #0x60]\n"
                       ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
                       ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
                       "ldr q0, [x25, #0x70]\n"
                       "add x25, x25, #0x88\n"
                       ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
                       ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
                       "ldr d27, [x20, #0x0]\n"
                       ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
                       ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
                       "fcvtl v27.4s, v27.4h\n"
                       "uzp1 v0.2d, v19.2d, v26.2d\n"
                       "uzp2 v26.2d, v19.2d, v26.2d\n"
                       "fmul v19.4s, v27.4s, v20.s[0]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "fmla v2.4s, v0.4s, v19.4s\n"
                       "ldr q19, [x23, #0x0]\n"
                       "uzp1 v0.2d, v18.2d, v17.2d\n"
                       "uzp2 v18.2d, v18.2d, v17.2d\n"
                       "fmul v17.4s, v27.4s, v20.s[1]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v10.4s, v26.4s, v17.4s\n"
                       "ldr q17, [x23, #0x10]\n"
                       "fmul v26.4s, v27.4s, v20.s[2]\n"
                       "fmul v20.4s, v27.4s, v20.s[3]\n"
                       "fmla v12.4s, v0.4s, v26.4s\n"
                       "ldr d0, [x22, #-0x8]\n"
                       "ldr d26, [x21, #-0x8]\n"
                       "fcvtl v0.4s, v0.4h\n"
                       "fmla v28.4s, v18.4s, v20.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x23, #0x20]\n"
                       "fcvtl v26.4s, v26.4h\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x23, #0x40]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q19, [x23, #0x60]\n"
                       ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
                       ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
                       "uzp1 v19.2d, v20.2d, v18.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp2 v20.2d, v20.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v9.s[0]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v11.4s, v19.4s, v18.4s\n"
                       "ldr q18, [x22, #0x0]\n"
                       "fmul v19.4s, v27.4s, v9.s[1]\n"
                       "fmla v13.4s, v20.4s, v19.4s\n"
                       "movi v19.4s, #0x0\n"
                       "movi v20.4s, #0x0\n"
                       ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x23, #0x30]\n"
                       ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x23, #0x50]\n"
                       ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x23, #0x70]\n"
                       "add x23, x23, #0x88\n"
                       ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v9.s[2]\n"
                       "fmul v9.4s, v27.4s, v9.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v22.4s, v17.4s, v19.4s\n"
                       "ldr q17, [x22, #0x10]\n"
                       "movi v19.4s, #0x0\n"
                       ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
                       "fmla v23.4s, v20.4s, v9.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v9.4s, #0x0\n"
                       ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
                       "ldr q18, [x22, #0x20]\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
                       ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
                       "ldr q18, [x22, #0x40]\n"
                       ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
                       ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
                       "ldr q18, [x22, #0x60]\n"
                       ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
                       ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x22, #0x30]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x22, #0x50]\n"
                       ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x22, #0x70]\n"
                       "add x22, x22, #0x88\n"
                       ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v0.s[0]\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v25.4s, v17.4s, v19.4s\n"
                       "ldr q19, [x21, #0x0]\n"
                       "fmul v17.4s, v27.4s, v0.s[1]\n"
                       "fmla v5.4s, v20.4s, v17.4s\n"
                       "ldr q17, [x21, #0x10]\n"
                       "uzp1 v20.2d, v9.2d, v18.2d\n"
                       "uzp2 v9.2d, v9.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v0.s[2]\n"
                       "fmul v0.4s, v27.4s, v0.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "fmla v7.4s, v20.4s, v18.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x21, #0x20]\n"
                       "fmla v4.4s, v9.4s, v0.4s\n"
                       "movi v9.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       "fmul v8.4s, v27.4s, v26.s[0]\n"
                       ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x21, #0x30]\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       "fmul v31.4s, v27.4s, v26.s[1]\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x21, #0x40]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       "fmul v15.4s, v27.4s, v26.s[2]\n"
                       "fmul v27.4s, v27.4s, v26.s[3]\n"
                       ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
                       "ldr q1, [x21, #0x50]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q26, [x21, #0x60]\n"
                       ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
                       ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
                       "ldr q21, [x21, #0x70]\n"
                       "add x21, x21, #0x88\n"
                       ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
                       ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
                       ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
                       ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
                       "uzp1 v29.2d, v20.2d, v18.2d\n"
                       "uzp2 v21.2d, v20.2d, v18.2d\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "uzp1 v18.2d, v9.2d, v0.2d\n"
                       "uzp2 v16.2d, v9.2d, v0.2d\n"
                       "scvtf v21.4s, v21.4s, #0x4\n"
                       "fmla v6.4s, v29.4s, v8.4s\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v30.4s, v21.4s, v31.4s\n"
                       "fmla v24.4s, v18.4s, v15.4s\n"
                       "fmla v14.4s, v16.4s, v27.4s\n"
                       "bgt 3b\n"
                       "mov x20, %x[res_ptr]\n"
                       "subs x27, x27, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q28, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q11, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q13, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q22, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q23, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q25, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q5, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q7, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q4, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q6, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q30, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q24, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q14, [x20, #0x0]\n"
                       "bne 2b\n"
                       "mov x20, #0x4\n"
                       "sub x10, x10, #0x10\n"
                       "cmp x10, #0x10\n"
                       "mov %x[res_ptr], x26\n"
                       "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
                       "bge 1b\n"
                       "4:" // Row loop skip
                       "cbz x10, 9f\n"
                       "5:" // Row tail: Row loop
                       "add x24, %x[b_ptr], #0x8\n"
                       "mov x23, %x[nc]\n"
                       "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
                       "6:" // Row tail: Column loop
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "add x25, %x[a_ptr], #0x8\n"
                       "mov x21, %x[nb]\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "7:" // Row tail: Block loop
                       "ldr q6, [x24, #0x0]\n"
                       "ldr q5, [x24, #0x10]\n"
                       "movi v17.16b, #0x4\n"
                       "movi v8.4s, #0x0\n"
                       "ldr q4, [x25, #0x0]\n"
                       "ldr q13, [x25, #0x10]\n"
                       "movi v27.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       "ldr q31, [x24, #0x20]\n"
                       "ldr q14, [x24, #0x30]\n"
                       "movi v29.4s, #0x0\n"
                       "movi v22.16b, #0xf0\n"
                       "ldr q11, [x25, #0x20]\n"
                       "ldr q23, [x25, #0x30]\n"
                       "sshl v21.16b, v6.16b, v17.16b\n"
                       "sshl v16.16b, v5.16b, v17.16b\n"
                       "ldr q20, [x25, #0x40]\n"
                       "ldr q26, [x25, #0x50]\n"
                       "and v6.16b, v6.16b, v22.16b\n"
                       "and v5.16b, v5.16b, v22.16b\n"
                       "ldr q25, [x25, #0x60]\n"
                       "ldr q3, [x25, #0x70]\n"
                       "sshl v19.16b, v31.16b, v17.16b\n"
                       "sshl v18.16b, v14.16b, v17.16b\n"
                       "ldr d17, [x25, #-0x8]\n"
                       ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
                       ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
                       "and v31.16b, v31.16b, v22.16b\n"
                       ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
                       ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
                       "and v14.16b, v14.16b, v22.16b\n"
                       "sub x20, x24, #0x8\n"
                       "ldr d16, [x20, #0x0]\n"
                       "subs x21, x21, #0x1\n"
                       "add x25, x25, #0x88\n"
                       "fcvtl v17.4s, v17.4h\n"
                       "add x24, x24, #0x48\n"
                       ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
                       ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
                       ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
                       ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
                       "fcvtl v16.4s, v16.4h\n"
                       ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
                       ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
                       "fmul v23.4s, v16.4s, v17.s[0]\n"
                       "fmul v21.4s, v16.4s, v17.s[1]\n"
                       "fmul v1.4s, v16.4s, v17.s[2]\n"
                       "fmul v20.4s, v16.4s, v17.s[3]\n"
                       ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
                       ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
                       ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
                       ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
                       ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
                       ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
                       "uzp1 v19.2d, v8.2d, v27.2d\n"
                       "uzp2 v18.2d, v8.2d, v27.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp1 v17.2d, v0.2d, v29.2d\n"
                       "uzp2 v16.2d, v0.2d, v29.2d\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v2.4s, v19.4s, v23.4s\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v10.4s, v18.4s, v21.4s\n"
                       "fmla v12.4s, v17.4s, v1.4s\n"
                       "fmla v28.4s, v16.4s, v20.4s\n"
                       "bgt 7b\n"
                       "mov x20, %x[res_ptr]\n"
                       "cmp x10, #0x1\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x2\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x3\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "str q28, [x20, #0x0]\n"
                       "8:" // Row tail: Accumulator store skip
                       "subs x23, x23, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "bne 6b\n"
                       "subs x10, x10, #0x4\n"
                       "add %x[a_ptr], %x[a_ptr], x9\n"
                       "mov %x[res_ptr], x22\n"
                       "bgt 5b\n"
                       "9:" // Row tail: Row loop skip
                       : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                       : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
                       : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                         "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                         "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
                         "x24", "x25", "x26", "x27", "x28");
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&
       // defined(__ARM_FEATURE_MATMUL_INT8)
  ggml_gemm_q4_0_4x8_q8_0_generic(pN, pS, pBs, pVx, pVy, nr, nc);
}

void ggml_gemm_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int nr, int nc) {
  const int cQk = QK8_0;
  const int nb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(nr % 4 == 0);
  assert(nc % cNcolsInterleaved == 0);

  UNUSED(pS);
  UNUSED(pBs);
  UNUSED(pVx);
  UNUSED(pVy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (ggml_cpu_get_sve_cnt() == QK8_0) {
    const void *b_ptr = pVx;
    const void *a_ptr = pVy;
    float *res_ptr = pS;
    size_t res_stride = pBs * sizeof(float);

    __asm__ __volatile__("mov x20, #0x4\n"
                         "mov x13, %x[nr]\n"
                         "mov z28.s, #-0x4\n"
                         "mov x12, #0x88\n"
                         "ptrue p1.b\n"
                         "whilelt p0.s, XZR, x20\n"
                         "cmp x13, #0x10\n"
                         "mul x12, %x[nb], x12\n"
                         "blt 4f\n"
                         "1:" // Row loop
                         "add x11, %x[b_ptr], #0x10\n"
                         "mov x10, %x[nc]\n"
                         "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
                         "2:" // Column loop
                         "add x28, %x[a_ptr], #0x8\n"
                         "mov z24.b, #0x0\n"
                         "mov z15.b, #0x0\n"
                         "mov x27, %x[nb]\n"
                         "add x26, x28, x12\n"
                         "mov z12.b, #0x0\n"
                         "mov z0.b, #0x0\n"
                         "add x25, x26, x12\n"
                         "mov z13.b, #0x0\n"
                         "mov z1.b, #0x0\n"
                         "add x24, x25, x12\n"
                         "mov z20.b, #0x0\n"
                         "mov z25.b, #0x0\n"
                         "mov z11.b, #0x0\n"
                         "mov z16.b, #0x0\n"
                         "mov z19.b, #0x0\n"
                         "mov z26.b, #0x0\n"
                         "mov z8.b, #0x0\n"
                         "mov z29.b, #0x0\n"
                         "mov z27.b, #0x0\n"
                         "mov z10.b, #0x0\n"
                         "3:" // Block loop
                         "ld1b { z30.b }, p1/Z, [x11]\n"
                         "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
                         "mov z18.s, #0x0\n"
                         "mov z7.s, #0x0\n"
                         "ld1rqb { z3.b }, p1/Z, [x28]\n"
                         "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
                         "mov z9.s, #0x0\n"
                         "mov z22.s, #0x0\n"
                         "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
                         "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
                         "sub x20, x11, #0x10\n"
                         "sub x23, x28, #0x8\n"
                         "lsl z31.b, z30.b, #0x4\n"
                         "lsl z6.b, z21.b, #0x4\n"
                         "ld1h { z23.s }, p1/Z, [x20]\n"
                         "sub x22, x26, #0x8\n"
                         "and z30.b, z30.b, #0xf0\n"
                         "and z21.b, z21.b, #0xf0\n"
                         "sub x21, x25, #0x8\n"
                         "sub x20, x24, #0x8\n"
                         "lsl z14.b, z4.b, #0x4\n"
                         "lsl z2.b, z17.b, #0x4\n"
                         "subs x27, x27, #0x1\n"
                         "add x11, x11, #0x90\n"
                         ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
                         ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
                         "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
                         "and z4.b, z4.b, #0xf0\n"
                         ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                         ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
                         "and z17.b, z17.b, #0xf0\n"
                         "fcvt z23.s, p1/m, z23.h\n"
                         ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
                         ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
                         "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
                         ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                         ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
                         "fscale z23.s, p1/m, z23.s, z28.s\n"
                         ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
                         ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
                         "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
                         ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                         ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
                         "add x28, x28, #0x88\n"
                         ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
                         ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
                         "ld1h { z3.s }, p0/Z, [x23]\n"
                         ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                         ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                         "fcvt z3.s, p1/m, z3.h\n"
                         "uzp1 z5.d, z18.d, z7.d\n"
                         "uzp2 z18.d, z18.d, z7.d\n"
                         "mov z3.q, z3.q[0]\n"
                         "uzp1 z7.d, z9.d, z22.d\n"
                         "uzp2 z22.d, z9.d, z22.d\n"
                         "fmul z9.s, z23.s, z3.s[0]\n"
                         "scvtf z5.s, p1/m, z5.s\n"
                         "scvtf z18.s, p1/m, z18.s\n"
                         "scvtf z7.s, p1/m, z7.s\n"
                         "scvtf z22.s, p1/m, z22.s\n"
                         "fmla z24.s, p1/M, z5.s, z9.s\n"
                         "ld1rqb { z5.b }, p1/Z, [x26]\n"
                         "fmul z9.s, z23.s, z3.s[1]\n"
                         "fmla z15.s, p1/M, z18.s, z9.s\n"
                         "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
                         "fmul z9.s, z23.s, z3.s[2]\n"
                         "fmul z3.s, z23.s, z3.s[3]\n"
                         "fmla z12.s, p1/M, z7.s, z9.s\n"
                         "mov z9.s, #0x0\n"
                         "ld1h { z7.s }, p0/Z, [x22]\n"
                         ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                         "fmla z0.s, p1/M, z22.s, z3.s\n"
                         "mov z22.s, #0x0\n"
                         "ld1h { z3.s }, p0/Z, [x21]\n"
                         ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
                         "fcvt z7.s, p1/m, z7.h\n"
                         "fcvt z3.s, p1/m, z3.h\n"
                         ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                         ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
                         "mov z7.q, z7.q[0]\n"
                         "mov z3.q, z3.q[0]\n"
                         ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                         ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
                         ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                         ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                         "uzp1 z5.d, z9.d, z22.d\n"
                         "scvtf z5.s, p1/m, z5.s\n"
                         "uzp2 z22.d, z9.d, z22.d\n"
                         "fmul z9.s, z23.s, z7.s[0]\n"
                         "scvtf z22.s, p1/m, z22.s\n"
                         "fmla z13.s, p1/M, z5.s, z9.s\n"
                         "ld1rqb { z9.b }, p1/Z, [x25]\n"
                         "fmul z5.s, z23.s, z7.s[1]\n"
                         "fmla z1.s, p1/M, z22.s, z5.s\n"
                         "mov z5.s, #0x0\n"
                         "mov z22.s, #0x0\n"
                         ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
                         ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
                         ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
                         ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
                         ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
                         ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
                         "add x26, x26, #0x88\n"
                         ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
                         ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
                         "uzp1 z18.d, z5.d, z22.d\n"
                         "scvtf z18.s, p1/m, z18.s\n"
                         "uzp2 z22.d, z5.d, z22.d\n"
                         "fmul z5.s, z23.s, z7.s[2]\n"
                         "fmul z7.s, z23.s, z7.s[3]\n"
                         "scvtf z22.s, p1/m, z22.s\n"
                         "fmla z20.s, p1/M, z18.s, z5.s\n"
                         "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
                         "ld1h { z5.s }, p0/Z, [x20]\n"
                         "fcvt z5.s, p1/m, z5.h\n"
                         "fmla z25.s, p1/M, z22.s, z7.s\n"
                         "mov z22.s, #0x0\n"
                         "mov z7.s, #0x0\n"
                         ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
                         ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                         "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
                         "mov z5.q, z5.q[0]\n"
                         ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
                         ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                         "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
                         ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
                         ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
                         "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
                         ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
                         ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
                         "uzp1 z9.d, z22.d, z7.d\n"
                         "scvtf z9.s, p1/m, z9.s\n"
                         "uzp2 z22.d, z22.d, z7.d\n"
                         "fmul z7.s, z23.s, z3.s[0]\n"
                         "scvtf z22.s, p1/m, z22.s\n"
                         "fmla z11.s, p1/M, z9.s, z7.s\n"
                         "ld1rqb { z9.b }, p1/Z, [x24]\n"
                         "fmul z7.s, z23.s, z3.s[1]\n"
                         "fmla z16.s, p1/M, z22.s, z7.s\n"
                         "mov z22.s, #0x0\n"
                         "mov z7.s, #0x0\n"
                         ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
                         ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
                         ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
                         ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
                         ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
                         ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
                         "add x25, x25, #0x88\n"
                         ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
                         ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
                         "uzp1 z18.d, z22.d, z7.d\n"
                         "scvtf z18.s, p1/m, z18.s\n"
                         "uzp2 z7.d, z22.d, z7.d\n"
                         "fmul z22.s, z23.s, z3.s[2]\n"
                         "fmul z3.s, z23.s, z3.s[3]\n"
                         "scvtf z7.s, p1/m, z7.s\n"
                         "fmla z19.s, p1/M, z18.s, z22.s\n"
                         "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
                         "fmul z22.s, z23.s, z5.s[0]\n"
                         "fmla z26.s, p1/M, z7.s, z3.s\n"
                         "mov z3.s, #0x0\n"
                         "mov z7.s, #0x0\n"
                         ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
                         ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                         "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
                         ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
                         ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                         "mov z9.s, #0x0\n"
                         ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
                         "mov z31.s, #0x0\n"
                         ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
                         "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
                         "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
                         ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
                         "fmul z14.s, z23.s, z5.s[1]\n"
                         ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
                         "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
                         "fmul z2.s, z23.s, z5.s[2]\n"
                         "fmul z23.s, z23.s, z5.s[3]\n"
                         ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
                         ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                         "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
                         ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
                         ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
                         "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
                         "add x24, x24, #0x88\n"
                         ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
                         ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
                         ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
                         ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
                         "uzp1 z18.d, z3.d, z7.d\n"
                         "uzp2 z5.d, z3.d, z7.d\n"
                         "scvtf z18.s, p1/m, z18.s\n"
                         "uzp1 z6.d, z9.d, z31.d\n"
                         "uzp2 z9.d, z9.d, z31.d\n"
                         "scvtf z5.s, p1/m, z5.s\n"
                         "fmla z8.s, p1/M, z18.s, z22.s\n"
                         "scvtf z6.s, p1/m, z6.s\n"
                         "scvtf z9.s, p1/m, z9.s\n"
                         "fmla z29.s, p1/M, z5.s, z14.s\n"
                         "fmla z27.s, p1/M, z6.s, z2.s\n"
                         "fmla z10.s, p1/M, z9.s, z23.s\n"
                         "bgt 3b\n"
                         "mov x20, %x[res_ptr]\n"
                         "subs x10, x10, #0x8\n"
                         "add %x[res_ptr], %x[res_ptr], #0x20\n"
                         "st1w { z24.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z15.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z12.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z0.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z13.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z1.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z20.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z25.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z11.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z16.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z19.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z26.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z8.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z29.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z27.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "st1w { z10.s }, p1, [x20]\n"
                         "bne 2b\n"
                         "mov x20, #0x4\n"
                         "sub x13, x13, #0x10\n"
                         "cmp x13, #0x10\n"
                         "mov %x[res_ptr], x9\n"
                         "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
                         "bge 1b\n"
                         "4:" // Row loop skip
                         "cbz x13, 9f\n"
                         "5:" // Row tail: Row loop
                         "add x25, %x[b_ptr], #0x10\n"
                         "mov x24, %x[nc]\n"
                         "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
                         "6:" // Row tail: Column loop
                         "mov z24.b, #0x0\n"
                         "mov z15.b, #0x0\n"
                         "add x28, %x[a_ptr], #0x8\n"
                         "mov x22, %x[nb]\n"
                         "mov z12.b, #0x0\n"
                         "mov z0.b, #0x0\n"
                         "7:" // Row tail: Block loop
                         "ld1b { z3.b }, p1/Z, [x25]\n"
                         "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
                         "mov z2.s, #0x0\n"
                         "mov z25.s, #0x0\n"
                         "ld1rqb { z26.b }, p1/Z, [x28]\n"
                         "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
                         "mov z27.s, #0x0\n"
                         "mov z19.s, #0x0\n"
                         "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
                         "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
                         "sub x21, x25, #0x10\n"
                         "sub x20, x28, #0x8\n"
                         "lsl z20.b, z3.b, #0x4\n"
                         "lsl z4.b, z6.b, #0x4\n"
                         "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
                         "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
                         "and z3.b, z3.b, #0xf0\n"
                         "and z6.b, z6.b, #0xf0\n"
                         "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
                         "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
                         "lsl z8.b, z29.b, #0x4\n"
                         "lsl z14.b, z16.b, #0x4\n"
                         "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
                         "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
                         ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
                         ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
                         "and z29.b, z29.b, #0xf0\n"
                         "ld1h { z17.s }, p1/Z, [x21]\n"
                         ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
                         ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
                         "and z16.b, z16.b, #0xf0\n"
                         "ld1h { z4.s }, p0/Z, [x20]\n"
                         "subs x22, x22, #0x1\n"
                         "add x28, x28, #0x88\n"
                         "fcvt z17.s, p1/m, z17.h\n"
                         "add x25, x25, #0x90\n"
                         ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
                         ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
                         "fcvt z4.s, p1/m, z4.h\n"
                         ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
                         ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
                         "fscale z17.s, p1/m, z17.s, z28.s\n"
                         "mov z4.q, z4.q[0]\n"
                         ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
                         ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
                         "fmul z23.s, z17.s, z4.s[0]\n"
                         "fmul z9.s, z17.s, z4.s[1]\n"
                         "fmul z21.s, z17.s, z4.s[2]\n"
                         "fmul z4.s, z17.s, z4.s[3]\n"
                         ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
                         ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
                         ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
                         ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
                         ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
                         ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
                         "uzp1 z31.d, z2.d, z25.d\n"
                         "uzp2 z13.d, z2.d, z25.d\n"
                         "scvtf z31.s, p1/m, z31.s\n"
                         "uzp1 z17.d, z27.d, z19.d\n"
                         "uzp2 z18.d, z27.d, z19.d\n"
                         "scvtf z13.s, p1/m, z13.s\n"
                         "fmla z24.s, p1/M, z31.s, z23.s\n"
                         "scvtf z17.s, p1/m, z17.s\n"
                         "scvtf z18.s, p1/m, z18.s\n"
                         "fmla z15.s, p1/M, z13.s, z9.s\n"
                         "fmla z12.s, p1/M, z17.s, z21.s\n"
                         "fmla z0.s, p1/M, z18.s, z4.s\n"
                         "bgt 7b\n"
                         "mov x20, %x[res_ptr]\n"
                         "cmp x13, #0x1\n"
                         "st1w { z24.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "ble 8f\n"
                         "cmp x13, #0x2\n"
                         "st1w { z15.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "ble 8f\n"
                         "cmp x13, #0x3\n"
                         "st1w { z12.s }, p1, [x20]\n"
                         "add x20, x20, %x[res_stride]\n"
                         "ble 8f\n"
                         "st1w { z0.s }, p1, [x20]\n"
                         "8:" // Row tail: Accumulator store skip
                         "subs x24, x24, #0x8\n"
                         "add %x[res_ptr], %x[res_ptr], #0x20\n"
                         "bne 6b\n"
                         "subs x13, x13, #0x4\n"
                         "add %x[a_ptr], %x[a_ptr], x12\n"
                         "mov %x[res_ptr], x23\n"
                         "bgt 5b\n"
                         "9:" // Row tail: Row loop skip
                         : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                         : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
                         : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23",
                           "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
                           "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21",
                           "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
    return;
  }
#endif // #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
  ggml_gemm_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const int8x16_t cKvalues = vld1q_s8(kvalues_iq4nl);

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_iq4_nlx4 *vBPtr = (const block_iq4_nlx4 *)pVx + (vX * cNb);

      float32x4_t vSumf[4];
      for (int vM = 0; vM < 4; vM++) {
        vSumf[vM] = vdupq_n_f32(0);
      }

      for (int vL = 0; vL < cNb; vL++) {
        float32x4_t vAD = vcvt_f32_f16(vld1_f16((const float16_t *)vAPtr[vL].d));
        float32x4_t vBD = vcvt_f32_f16(vld1_f16((const float16_t *)vBPtr[vL].d));

        int32x4_t vSumi0 = vdupq_n_s32(0);
        int32x4_t vSumi1 = vdupq_n_s32(0);
        int32x4_t vSumi2 = vdupq_n_s32(0);
        int32x4_t vSumi3 = vdupq_n_s32(0);

        for (int vK = 0; vK < 4; vK++) {
          int8x16_t vA0 = vld1q_s8(vAPtr[vL].qs + 16 * vK + 0);
          int8x16_t vA1 = vld1q_s8(vAPtr[vL].qs + 16 * vK + 64);

          uint8x16_t vB = vld1q_u8(vBPtr[vL].qs + 16 * vK);
          int8x16_t vBHi = vqtbl1q_s8(cKvalues, vB >> 4);
          int8x16_t vBLo = vqtbl1q_s8(cKvalues, vB & 0xF);

          vSumi0 = vdotq_laneq_s32(vSumi0, vBLo, vA0, 0);
          vSumi1 = vdotq_laneq_s32(vSumi1, vBLo, vA0, 1);
          vSumi2 = vdotq_laneq_s32(vSumi2, vBLo, vA0, 2);
          vSumi3 = vdotq_laneq_s32(vSumi3, vBLo, vA0, 3);
          vSumi0 = vdotq_laneq_s32(vSumi0, vBHi, vA1, 0);
          vSumi1 = vdotq_laneq_s32(vSumi1, vBHi, vA1, 1);
          vSumi2 = vdotq_laneq_s32(vSumi2, vBHi, vA1, 2);
          vSumi3 = vdotq_laneq_s32(vSumi3, vBHi, vA1, 3);
        }

        vSumf[0] = vmlaq_f32(vSumf[0], vmulq_laneq_f32(vBD, vAD, 0), vcvtq_f32_s32(vSumi0));
        vSumf[1] = vmlaq_f32(vSumf[1], vmulq_laneq_f32(vBD, vAD, 1), vcvtq_f32_s32(vSumi1));
        vSumf[2] = vmlaq_f32(vSumf[2], vmulq_laneq_f32(vBD, vAD, 2), vcvtq_f32_s32(vSumi2));
        vSumf[3] = vmlaq_f32(vSumf[3], vmulq_laneq_f32(vBD, vAD, 3), vcvtq_f32_s32(vSumi3));
      }

      for (int vM = 0; vM < 4; vM++) {
        vst1q_f32(pS + (vY * 4 + vM) * pBs + vX * 4, vSumf[vM]);
      }
    }
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
  ggml_gemm_iq4_nl_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_mxfp4_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) &&                    \
    defined(__ARM_FEATURE_DOTPROD)
  const int8x16_t cKvalues = vld1q_s8(kvalues_mxfp4);

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_mxfp4x4 *vBPtr = (const block_mxfp4x4 *)pVx + (vX * cNb);

      float32x4_t vSumf[4];
      for (int vM = 0; vM < 4; vM++) {
        vSumf[vM] = vdupq_n_f32(0);
      }

      for (int vL = 0; vL < cNb; vL++) {
        float32x4_t vAD = vcvt_f32_f16(vld1_f16((const float16_t *)vAPtr[vL].d));
        float32x4_t vBD = {
            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[0]),
            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[1]),
            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[2]),
            GGML_CPU_E8M0_TO_FP32_HALF(vBPtr[vL].e[3]),
        };

        int32x4_t vSumi0 = vdupq_n_s32(0);
        int32x4_t vSumi1 = vdupq_n_s32(0);
        int32x4_t vSumi2 = vdupq_n_s32(0);
        int32x4_t vSumi3 = vdupq_n_s32(0);

        for (int vK = 0; vK < 4; vK++) {
          int8x16_t vA0 = vld1q_s8(vAPtr[vL].qs + 16 * vK + 0);
          int8x16_t vA1 = vld1q_s8(vAPtr[vL].qs + 16 * vK + 64);

          uint8x16_t vB = vld1q_u8(vBPtr[vL].qs + 16 * vK);
          int8x16_t vBHi = vqtbl1q_s8(cKvalues, vB >> 4);
          int8x16_t vBLo = vqtbl1q_s8(cKvalues, vB & 0xF);

          vSumi0 = vdotq_laneq_s32(vSumi0, vBLo, vA0, 0);
          vSumi1 = vdotq_laneq_s32(vSumi1, vBLo, vA0, 1);
          vSumi2 = vdotq_laneq_s32(vSumi2, vBLo, vA0, 2);
          vSumi3 = vdotq_laneq_s32(vSumi3, vBLo, vA0, 3);
          vSumi0 = vdotq_laneq_s32(vSumi0, vBHi, vA1, 0);
          vSumi1 = vdotq_laneq_s32(vSumi1, vBHi, vA1, 1);
          vSumi2 = vdotq_laneq_s32(vSumi2, vBHi, vA1, 2);
          vSumi3 = vdotq_laneq_s32(vSumi3, vBHi, vA1, 3);
        }

        vSumf[0] = vmlaq_f32(vSumf[0], vmulq_laneq_f32(vBD, vAD, 0), vcvtq_f32_s32(vSumi0));
        vSumf[1] = vmlaq_f32(vSumf[1], vmulq_laneq_f32(vBD, vAD, 1), vcvtq_f32_s32(vSumi1));
        vSumf[2] = vmlaq_f32(vSumf[2], vmulq_laneq_f32(vBD, vAD, 2), vcvtq_f32_s32(vSumi2));
        vSumf[3] = vmlaq_f32(vSumf[3], vmulq_laneq_f32(vBD, vAD, 3), vcvtq_f32_s32(vSumi3));
      }

      for (int vM = 0; vM < 4; vM++) {
        vst1q_f32(pS + (vY * 4 + vM) * pBs + vX * 4, vSumf[vM]);
      }
    }
  }
  return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
  ggml_gemm_mxfp4_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q4_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cQ8KBlocklen = 4;
  constexpr int cAccSize = 2 * 4; // 2 row pairs × 4 col pairs
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);

  // 8 accumulators: 2 row pairs × 4 col pairs
  float32x4_t vAccF32[cAccSize];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_Kx8 *GGML_RESTRICT vQ4Ptr = (const block_q4_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cAccSize; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        // d4 0 1 2 3, 4 5 6 7
        float32x4_t vQ4D0123 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d));
        float32x4_t vQ4D4567 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].d + 4));
        // d8 0 1 2 3
        float32x4_t vQ8D0123 = vld1q_f32(vQ8Ptr[vB].d);
        // mins
        float32x4_t vQ4Dmin0123 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin));
        float32x4_t vQ4Dmin4567 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ4Ptr[vB].dmin + 4));

        // Precomputation of scales and mins
        float32x4_t aSbdScale0123[cQ8KBlocklen];
        float32x4_t aSbdScale4567[cQ8KBlocklen];
        float32x4_t aSbdMin0123[cQ8KBlocklen];
        float32x4_t aSbdMin4567[cQ8KBlocklen];

        aSbdScale0123[0] = vmulq_laneq_f32(vQ4D0123, vQ8D0123, 0);
        aSbdScale4567[0] = vmulq_laneq_f32(vQ4D4567, vQ8D0123, 0);
        aSbdMin0123[0] = vmulq_laneq_f32(vQ4Dmin0123, vQ8D0123, 0);
        aSbdMin4567[0] = vmulq_laneq_f32(vQ4Dmin4567, vQ8D0123, 0);

        aSbdScale0123[1] = vmulq_laneq_f32(vQ4D0123, vQ8D0123, 1);
        aSbdScale4567[1] = vmulq_laneq_f32(vQ4D4567, vQ8D0123, 1);
        aSbdMin0123[1] = vmulq_laneq_f32(vQ4Dmin0123, vQ8D0123, 1);
        aSbdMin4567[1] = vmulq_laneq_f32(vQ4Dmin4567, vQ8D0123, 1);

        aSbdScale0123[2] = vmulq_laneq_f32(vQ4D0123, vQ8D0123, 2);
        aSbdScale4567[2] = vmulq_laneq_f32(vQ4D4567, vQ8D0123, 2);
        aSbdMin0123[2] = vmulq_laneq_f32(vQ4Dmin0123, vQ8D0123, 2);
        aSbdMin4567[2] = vmulq_laneq_f32(vQ4Dmin4567, vQ8D0123, 2);

        aSbdScale0123[3] = vmulq_laneq_f32(vQ4D0123, vQ8D0123, 3);
        aSbdScale4567[3] = vmulq_laneq_f32(vQ4D4567, vQ8D0123, 3);
        aSbdMin0123[3] = vmulq_laneq_f32(vQ4Dmin0123, vQ8D0123, 3);
        aSbdMin4567[3] = vmulq_laneq_f32(vQ4Dmin4567, vQ8D0123, 3);

        // Precomputation of bsums, each vpaddq calcs all the bsums for each row
        const int16x8_t cBsums[cQ8KBlocklen] = {
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3 + 8)),
        };
        int16_t vBsumsArr[QK_K / 64][8];
        for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
          vst1q_s16(vBsumsArr[vQ8Row], cBsums[vQ8Row]);
        }

        // interleaved bias_acc: [0]->r0 0123, [1]->r1 0123, .., [4]->r0 4567, [5]->r1 4567 ..
        int32x4_t aBiasAcc[cAccSize];
        for (int vI = 0; vI < cAccSize; vI++) {
          aBiasAcc[vI] = vdupq_n_s32(0);
        }

        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          // Int accumulators for qs vecdot (4 row x 2 col quartets)
          int32x4_t vAccLo[cAccSize];
          int32x4_t vAccHi[cAccSize];
          for (int vI = 0; vI < cAccSize; vI++) {
            vAccLo[vI] = vdupq_n_s32(0);
            vAccHi[vI] = vdupq_n_s32(0);
          }
          // Need scales for the low and high nibbles
          // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
          int16x8_t aQ4sbScales[2];
          int16x8_t aQ4sbMins[2];
          for (int vI = 0; vI < 2; vI++) {
            int8_t vAuxQ4sb[8];
            const int cOffset = vSb * 24 + vI * 12;
            fDecodeQKx86bitScales(&vQ4Ptr[vB].scales[cOffset], &aQ4sbMins[vI], vAuxQ4sb);
            aQ4sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ4sb));
          }

          constexpr int cReadsPerSb = 8; // 8 * 16 bytes each => 32 qs * 4 rows
          for (int vK = 0; vK < cReadsPerSb; vK++) {
            const int8x16_t cQ8Blk0 = vld1q_s8(vQ8Ptr[vB].qs + vSb * 256 + 16 * vK);
            const int8x16_t cQ8Blk1 = vld1q_s8(vQ8Ptr[vB].qs + vSb * 256 + 16 * vK + 128);

            // 0..3 & 32..35
            const uint8x16_t cQ40123 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 32 * vK);
            const uint8x16_t cQ44567 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 32 * vK + 16);

            const int8x16_t cQ40123Lo = vreinterpretq_s8_u8(vandq_u8(cQ40123, cM4b));
            const int8x16_t cQ40123Hi = vreinterpretq_s8_u8(vshrq_n_u8(cQ40123, 4));

            vAccLo[0] = vdotq_laneq_s32(vAccLo[0], cQ40123Lo, cQ8Blk0, 0); //  0..3  r0 c0123
            vAccLo[1] = vdotq_laneq_s32(vAccLo[1], cQ40123Lo, cQ8Blk0, 1); //  0..3  r1 c0123
            vAccLo[2] = vdotq_laneq_s32(vAccLo[2], cQ40123Lo, cQ8Blk0, 2); //  0..3  r2 c0123
            vAccLo[3] = vdotq_laneq_s32(vAccLo[3], cQ40123Lo, cQ8Blk0, 3); //  0..3  r3 c0123

            vAccHi[0] = vdotq_laneq_s32(vAccHi[0], cQ40123Hi, cQ8Blk1, 0); // 32..35 r0 c0123
            vAccHi[1] = vdotq_laneq_s32(vAccHi[1], cQ40123Hi, cQ8Blk1, 1); // 32..35 r1 c0123
            vAccHi[2] = vdotq_laneq_s32(vAccHi[2], cQ40123Hi, cQ8Blk1, 2); // 32..35 r2 c0123
            vAccHi[3] = vdotq_laneq_s32(vAccHi[3], cQ40123Hi, cQ8Blk1, 3); // 32..35 r3 c0123

            const int8x16_t cQ44567Lo = vreinterpretq_s8_u8(vandq_u8(cQ44567, cM4b));
            const int8x16_t cQ44567Hi = vreinterpretq_s8_u8(vshrq_n_u8(cQ44567, 4));

            vAccLo[4] = vdotq_laneq_s32(vAccLo[4], cQ44567Lo, cQ8Blk0, 0); //  0..3  r0 c4567
            vAccLo[5] = vdotq_laneq_s32(vAccLo[5], cQ44567Lo, cQ8Blk0, 1); //  0..3  r1 c4567
            vAccLo[6] = vdotq_laneq_s32(vAccLo[6], cQ44567Lo, cQ8Blk0, 2); //  0..3  r2 c4567
            vAccLo[7] = vdotq_laneq_s32(vAccLo[7], cQ44567Lo, cQ8Blk0, 3); //  0..3  r3 c4567

            vAccHi[4] = vdotq_laneq_s32(vAccHi[4], cQ44567Hi, cQ8Blk1, 0); // 32..35 r0 c4567
            vAccHi[5] = vdotq_laneq_s32(vAccHi[5], cQ44567Hi, cQ8Blk1, 1); // 32..35 r1 c4567
            vAccHi[6] = vdotq_laneq_s32(vAccHi[6], cQ44567Hi, cQ8Blk1, 2); // 32..35 r2 c4567
            vAccHi[7] = vdotq_laneq_s32(vAccHi[7], cQ44567Hi, cQ8Blk1, 3); // 32..35 r3 c4567
          }

          // Scale and bias application
          // acc is stored interleaved to match output layout
          const int16x4_t cSc0123Lo = vget_low_s16(aQ4sbScales[0]);
          const int16x4_t cSc4567Lo = vget_high_s16(aQ4sbScales[0]);
          const int16x4_t cSc0123Hi = vget_low_s16(aQ4sbScales[1]);
          const int16x4_t cSc4567Hi = vget_high_s16(aQ4sbScales[1]);
          for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
            // Bias correction
            // row c0123 blk0 and blk1
            const float32x4_t cSumf0123 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(cSc0123Lo), vAccLo[vRow]),
                                                                  vmulq_s32(vmovl_s16(cSc0123Hi), vAccHi[vRow])));
            vAccF32[2 * vRow] = vfmaq_f32(vAccF32[2 * vRow], aSbdScale0123[vRow], cSumf0123);

            // row c4567 blk0 and blk1
            const float32x4_t cSumf4567 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(cSc4567Lo), vAccLo[vRow + 4]),
                                                                  vmulq_s32(vmovl_s16(cSc4567Hi), vAccHi[vRow + 4])));
            vAccF32[2 * vRow + 1] = vfmaq_f32(vAccF32[2 * vRow + 1], aSbdScale4567[vRow], cSumf4567);

            // Bias
            const int16x4_t cBsumsVecLo = vdup_n_s16(vBsumsArr[vSb][vRow * 2]);
            const int16x4_t cBsumsVecHi = vdup_n_s16(vBsumsArr[vSb][vRow * 2 + 1]);

            // row c0123 blk0 and blk1
            aBiasAcc[2 * vRow] = vmlal_s16(aBiasAcc[2 * vRow], cBsumsVecLo, vget_low_s16(aQ4sbMins[0]));
            aBiasAcc[2 * vRow] = vmlal_s16(aBiasAcc[2 * vRow], cBsumsVecHi, vget_low_s16(aQ4sbMins[1]));

            // row c4567 blk0 and blk1
            aBiasAcc[2 * vRow + 1] = vmlal_s16(aBiasAcc[2 * vRow + 1], cBsumsVecLo, vget_high_s16(aQ4sbMins[0]));
            aBiasAcc[2 * vRow + 1] = vmlal_s16(aBiasAcc[2 * vRow + 1], cBsumsVecHi, vget_high_s16(aQ4sbMins[1]));
          }
        } // for sb

        for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
          vAccF32[2 * vRow] = vmlsq_f32(vAccF32[2 * vRow], vcvtq_f32_s32(aBiasAcc[2 * vRow]), aSbdMin0123[vRow]);
          vAccF32[2 * vRow + 1] =
              vmlsq_f32(vAccF32[2 * vRow + 1], vcvtq_f32_s32(aBiasAcc[2 * vRow + 1]), aSbdMin4567[vRow]);
        }
      } // for b

      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemm_q4_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q5_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cQ8KBlocklen = 4;
  constexpr int cAccSize = 2 * 4; // 2 row pairs, 4 col pairs
  constexpr int cColGroups = cNcolsInterleaved / 4;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMone = vdupq_n_u8(1);
  const uint8x16_t cMtwo = vdupq_n_u8(2);

  // 8 accumulators: 2 row pairs, 4 col pairs
  float32x4_t vAccF32[cAccSize];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q5_Kx8 *GGML_RESTRICT vQ5Ptr = (const block_q5_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cAccSize; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        // d5 0 1 2 3, 4 5 6 7
        float32x4_t vQ5D0123 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d));
        float32x4_t vQ5D4567 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].d + 4));
        // d8 0 1 2 3
        float32x4_t vQ8D0123 = vld1q_f32(vQ8Ptr[vB].d);
        // mins
        float32x4_t vQ5Dmin0123 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin));
        float32x4_t vQ5Dmin4567 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ5Ptr[vB].dmin + 4));

        // Precomputation of scales and mins
        float32x4_t aSbdScale0123[cQ8KBlocklen];
        float32x4_t aSbdScale4567[cQ8KBlocklen];
        float32x4_t aSbdMin0123[cQ8KBlocklen];
        float32x4_t aSbdMin4567[cQ8KBlocklen];

        aSbdScale0123[0] = vmulq_laneq_f32(vQ5D0123, vQ8D0123, 0);
        aSbdScale4567[0] = vmulq_laneq_f32(vQ5D4567, vQ8D0123, 0);
        aSbdMin0123[0] = vmulq_laneq_f32(vQ5Dmin0123, vQ8D0123, 0);
        aSbdMin4567[0] = vmulq_laneq_f32(vQ5Dmin4567, vQ8D0123, 0);

        aSbdScale0123[1] = vmulq_laneq_f32(vQ5D0123, vQ8D0123, 1);
        aSbdScale4567[1] = vmulq_laneq_f32(vQ5D4567, vQ8D0123, 1);
        aSbdMin0123[1] = vmulq_laneq_f32(vQ5Dmin0123, vQ8D0123, 1);
        aSbdMin4567[1] = vmulq_laneq_f32(vQ5Dmin4567, vQ8D0123, 1);

        aSbdScale0123[2] = vmulq_laneq_f32(vQ5D0123, vQ8D0123, 2);
        aSbdScale4567[2] = vmulq_laneq_f32(vQ5D4567, vQ8D0123, 2);
        aSbdMin0123[2] = vmulq_laneq_f32(vQ5Dmin0123, vQ8D0123, 2);
        aSbdMin4567[2] = vmulq_laneq_f32(vQ5Dmin4567, vQ8D0123, 2);

        aSbdScale0123[3] = vmulq_laneq_f32(vQ5D0123, vQ8D0123, 3);
        aSbdScale4567[3] = vmulq_laneq_f32(vQ5D4567, vQ8D0123, 3);
        aSbdMin0123[3] = vmulq_laneq_f32(vQ5Dmin0123, vQ8D0123, 3);
        aSbdMin4567[3] = vmulq_laneq_f32(vQ5Dmin4567, vQ8D0123, 3);

        // Precomputation of bsums, each vpaddq calcs all the bsums for each row
        const int16x8_t cBsums[cQ8KBlocklen] = {
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3 + 8)),
        };
        int16_t vBsumsArr[QK_K / 64][8];
        for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
          vst1q_s16(vBsumsArr[vQ8Row], cBsums[vQ8Row]);
        }

        // interleaved bias_acc: [0]->r0 0123, [1]->r1 0123, .., [4]->r0 4567, [5]->r1 4567 ..
        int32x4_t aBiasAcc[cAccSize];
        for (int vI = 0; vI < cAccSize; vI++) {
          aBiasAcc[vI] = vdupq_n_s32(0);
        }

        uint8x16_t vQh[cColGroups][8];
        for (int vC = 0; vC < cColGroups; vC++) {
          for (int vI = 0; vI < 8; vI++) {
            vQh[vC][vI] = vld1q_u8(vQ5Ptr[vB].qh + vI * 32 + 16 * vC);
          }
        }

        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          // Int accumulators for qs vecdot (4 row * 2 col quartets)
          int32x4_t vAccLo[cAccSize];
          int32x4_t vAccHi[cAccSize];
          for (int vI = 0; vI < cAccSize; vI++) {
            vAccLo[vI] = vdupq_n_s32(0);
            vAccHi[vI] = vdupq_n_s32(0);
          }
          // Need scales for the low and high nibbles
          // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
          int16x8_t aQ5sbScales[2];
          int16x8_t aQ5sbMins[2];
          for (int vI = 0; vI < 2; vI++) {
            int8_t vAuxQ5sb[8];
            const int cOffset = vSb * 24 + vI * 12;
            fDecodeQKx86bitScales(&vQ5Ptr[vB].scales[cOffset], &aQ5sbMins[vI], vAuxQ5sb);
            aQ5sbScales[vI] = vmovl_s8(vld1_s8(vAuxQ5sb));
          }

          constexpr int cReadsPerSb = 8; // 8 * 16 bytes each => 32 qs * 4 rows
          for (int vK = 0; vK < cReadsPerSb; vK++) {
            const int8x16_t cQ8Blk0 = vld1q_s8(vQ8Ptr[vB].qs + vSb * 256 + 16 * vK);
            const int8x16_t cQ8Blk1 = vld1q_s8(vQ8Ptr[vB].qs + vSb * 256 + 16 * vK + 128);

            // 0..3 & 32..35
            const uint8x16_t cQ50123 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 32 * vK);
            const uint8x16_t cQ54567 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 32 * vK + 16);

            // NOTE: This is the only difference with q4_K
            const uint8x16_t cHbitLo0123 = vandq_u8(vQh[0][vK], cMone);
            const uint8x16_t cHbitHi0123 = vshlq_n_u8(vandq_u8(vQh[0][vK], cMtwo), 3);
            vQh[0][vK] = vshrq_n_u8(vQh[0][vK], 2);
            const uint8x16_t cHbitLo4567 = vandq_u8(vQh[1][vK], cMone);
            const uint8x16_t cHbitHi4567 = vshlq_n_u8(vandq_u8(vQh[1][vK], cMtwo), 3);
            vQh[1][vK] = vshrq_n_u8(vQh[1][vK], 2);
            // From here, same as q4_K

            const int8x16_t cQ50123Lo = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ50123, cM4b), cHbitLo0123, 4));
            const int8x16_t cQ50123Hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ50123, 4), cHbitHi0123));

            vAccLo[0] = vdotq_laneq_s32(vAccLo[0], cQ50123Lo, cQ8Blk0, 0); //  0..3  r0 c0123
            vAccLo[1] = vdotq_laneq_s32(vAccLo[1], cQ50123Lo, cQ8Blk0, 1); //  0..3  r1 c0123
            vAccLo[2] = vdotq_laneq_s32(vAccLo[2], cQ50123Lo, cQ8Blk0, 2); //  0..3  r2 c0123
            vAccLo[3] = vdotq_laneq_s32(vAccLo[3], cQ50123Lo, cQ8Blk0, 3); //  0..3  r3 c0123

            vAccHi[0] = vdotq_laneq_s32(vAccHi[0], cQ50123Hi, cQ8Blk1, 0); // 32..35 r0 c0123
            vAccHi[1] = vdotq_laneq_s32(vAccHi[1], cQ50123Hi, cQ8Blk1, 1); // 32..35 r1 c0123
            vAccHi[2] = vdotq_laneq_s32(vAccHi[2], cQ50123Hi, cQ8Blk1, 2); // 32..35 r2 c0123
            vAccHi[3] = vdotq_laneq_s32(vAccHi[3], cQ50123Hi, cQ8Blk1, 3); // 32..35 r3 c0123

            const int8x16_t cQ54567Lo = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ54567, cM4b), cHbitLo4567, 4));
            const int8x16_t cQ54567Hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ54567, 4), cHbitHi4567));

            vAccLo[4] = vdotq_laneq_s32(vAccLo[4], cQ54567Lo, cQ8Blk0, 0); //  0..3  r0 c4567
            vAccLo[5] = vdotq_laneq_s32(vAccLo[5], cQ54567Lo, cQ8Blk0, 1); //  0..3  r1 c4567
            vAccLo[6] = vdotq_laneq_s32(vAccLo[6], cQ54567Lo, cQ8Blk0, 2); //  0..3  r2 c4567
            vAccLo[7] = vdotq_laneq_s32(vAccLo[7], cQ54567Lo, cQ8Blk0, 3); //  0..3  r3 c4567

            vAccHi[4] = vdotq_laneq_s32(vAccHi[4], cQ54567Hi, cQ8Blk1, 0); // 32..35 r0 c4567
            vAccHi[5] = vdotq_laneq_s32(vAccHi[5], cQ54567Hi, cQ8Blk1, 1); // 32..35 r1 c4567
            vAccHi[6] = vdotq_laneq_s32(vAccHi[6], cQ54567Hi, cQ8Blk1, 2); // 32..35 r2 c4567
            vAccHi[7] = vdotq_laneq_s32(vAccHi[7], cQ54567Hi, cQ8Blk1, 3); // 32..35 r3 c4567
          }

          // Scale and bias application
          // acc is stored interleaved to match output layout
          const int16x4_t cSc0123Lo = vget_low_s16(aQ5sbScales[0]);
          const int16x4_t cSc4567Lo = vget_high_s16(aQ5sbScales[0]);
          const int16x4_t cSc0123Hi = vget_low_s16(aQ5sbScales[1]);
          const int16x4_t cSc4567Hi = vget_high_s16(aQ5sbScales[1]);
          for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
            // Bias correction
            // row c0123 blk0 and blk1
            const float32x4_t cSumf0123 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(cSc0123Lo), vAccLo[vRow]),
                                                                  vmulq_s32(vmovl_s16(cSc0123Hi), vAccHi[vRow])));
            vAccF32[2 * vRow] = vfmaq_f32(vAccF32[2 * vRow], aSbdScale0123[vRow], cSumf0123);

            // row c4567 blk0 and blk1
            const float32x4_t cSumf4567 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(cSc4567Lo), vAccLo[vRow + 4]),
                                                                  vmulq_s32(vmovl_s16(cSc4567Hi), vAccHi[vRow + 4])));
            vAccF32[2 * vRow + 1] = vfmaq_f32(vAccF32[2 * vRow + 1], aSbdScale4567[vRow], cSumf4567);

            // Bias
            const int16x4_t cBsumsVecLo = vdup_n_s16(vBsumsArr[vSb][vRow * 2]);
            const int16x4_t cBsumsVecHi = vdup_n_s16(vBsumsArr[vSb][vRow * 2 + 1]);

            // row c0123 blk0 and blk1
            aBiasAcc[2 * vRow] = vmlal_s16(aBiasAcc[2 * vRow], cBsumsVecLo, vget_low_s16(aQ5sbMins[0]));
            aBiasAcc[2 * vRow] = vmlal_s16(aBiasAcc[2 * vRow], cBsumsVecHi, vget_low_s16(aQ5sbMins[1]));

            // row c4567 blk0 and blk1
            aBiasAcc[2 * vRow + 1] = vmlal_s16(aBiasAcc[2 * vRow + 1], cBsumsVecLo, vget_high_s16(aQ5sbMins[0]));
            aBiasAcc[2 * vRow + 1] = vmlal_s16(aBiasAcc[2 * vRow + 1], cBsumsVecHi, vget_high_s16(aQ5sbMins[1]));
          }
        } // for sb

        for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
          vAccF32[2 * vRow] = vmlsq_f32(vAccF32[2 * vRow], vcvtq_f32_s32(aBiasAcc[2 * vRow]), aSbdMin0123[vRow]);
          vAccF32[2 * vRow + 1] =
              vmlsq_f32(vAccF32[2 * vRow + 1], vcvtq_f32_s32(aBiasAcc[2 * vRow + 1]), aSbdMin4567[vRow]);
        }
      } // for b

      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemm_q5_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q4_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (svcntb() * 8 == 256) {
    constexpr int cQ8KBlocklen = 4;
    const svuint8_t cM4b1 = svdup_n_u8(0x0f);
    // 8 accumulators: 2 row pairs × 4 col pairs
    svfloat32_t vAccF3201, vAccF3223, vAccF3245, vAccF3267;
    uint32_t aIdxArr[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    svbool_t vPg = svptrue_pat_b32(SV_VL8);
    svuint32_t vIdx = svld1(vPg, aIdxArr);

    static const uint32_t aIdxData[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    svuint32_t vIdx1 = svld1_u32(svptrue_b32(), aIdxData);

    for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
      const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

      for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
        const block_q4_Kx8 *GGML_RESTRICT vQ4Ptr = (const block_q4_Kx8 *)pVx + (vX * cNb);

        vAccF3201 = svdup_n_f32(0);
        vAccF3223 = svdup_n_f32(0);
        vAccF3245 = svdup_n_f32(0);
        vAccF3267 = svdup_n_f32(0);

        for (int vB = 0; vB < cNb; vB++) {
          // bsums pairs belongs to the same q8_k subblock
          // 64 elements loaded and made sum of 0-7 and 8-15 sum || 16-23 and 24 - 31 sum
          const int16x8_t cBsums[4]{
              vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0 + 8)),
              vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1 + 8)),
              vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2 + 8)),
              vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3 + 8)),
          };

          int32_t vBsumsArr32[4][8];

          for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
            int16x8_t v16 = cBsums[vQ8Row];

            // low 4
            int32x4_t vV32Lo = vmovl_s16(vget_low_s16(v16));
            vst1q_s32(&vBsumsArr32[vQ8Row][0], vV32Lo);

            // high 4
            int32x4_t vV32Hi = vmovl_s16(vget_high_s16(v16));
            vst1q_s32(&vBsumsArr32[vQ8Row][4], vV32Hi);
          }

          svint32_t vSbAcc0 = svdup_n_s32(0);
          svint32_t vSbAcc2 = svdup_n_s32(0);

          svint32_t vAcc00 = svdup_n_s32(0);
          svint32_t vAcc11 = svdup_n_s32(0);
          svint32_t vAcc22 = svdup_n_s32(0);
          svint32_t vAcc33 = svdup_n_s32(0);
          svint32_t vAcc44 = svdup_n_s32(0);
          svint32_t vAcc55 = svdup_n_s32(0);
          svint32_t vAcc66 = svdup_n_s32(0);
          svint32_t vAcc77 = svdup_n_s32(0);

          svint32_t vBiasAcc00 = svdup_n_s32(0);
          svint32_t vBiasAcc22 = svdup_n_s32(0);
          svint32_t vBiasAcc44 = svdup_n_s32(0);
          svint32_t vBiasAcc66 = svdup_n_s32(0);

          for (int vSb = 0; vSb < QK_K / 64; vSb++) {
            // Need scales for the low and high nibbles
            // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
            svint32_t vBlockScale0, vBlockScale1, vBlockScale2, vBlockScale3;
            svint32_t vQ4sbMins0, vQ4sbMins1;
            {
              // 2-superblock I am working on
              const int cOffset = vSb * 24 + 0 * 12;
              const uint8_t *vScalesIn = &vQ4Ptr[vB].scales[cOffset];

              const int cOffset1 = vSb * 24 + 12;
              const uint8_t *vScalesIn1 = &vQ4Ptr[vB].scales[cOffset1];

              constexpr uint32_t cKmask1 = 0x3f3f3f3f;
              constexpr uint32_t cKmask2 = 0x0f0f0f0f;
              constexpr uint32_t cKmask3 = 0x03030303;
              constexpr uint8_t cScalesSize = 12;

              uint32_t aSm[3];
              memcpy(aSm, vScalesIn, cScalesSize);

              uint32_t aSm1[3];
              memcpy(aSm1, vScalesIn1, cScalesSize);

              const uint32_t cMins03 = aSm[1] & cKmask1;
              const uint32_t cMins47 = ((aSm[2] >> 4) & cKmask2) | (((aSm[1] >> 6) & cKmask3) << 4);

              const uint32_t cMins031 = aSm1[1] & cKmask1;
              const uint32_t cMins471 = ((aSm1[2] >> 4) & cKmask2) | (((aSm1[1] >> 6) & cKmask3) << 4);

              svuint32_t vMinsU32Temp = svzip1_u32(svdup_n_u32(cMins03), svdup_n_u32(cMins47));
              svuint32_t vMinsU32Temp1 = svzip1_u32(svdup_n_u32(cMins031), svdup_n_u32(cMins471));

              /* reinterpret u32 → u8 */
              svuint8_t vMinsU8 = svreinterpret_u8_u32(vMinsU32Temp);
              svuint8_t vMinsU81 = svreinterpret_u8_u32(vMinsU32Temp1);

              /* widen u8 → u16->u32 (lower half only) */
              svuint32_t vMinsU16 = svunpklo_u32(svunpklo_u16(vMinsU8));
              svuint32_t vMinsU161 = svunpklo_u32(svunpklo_u16(vMinsU81));

              vQ4sbMins0 = svreinterpret_s32_u32(vMinsU16);
              vQ4sbMins1 = svreinterpret_s32_u32(vMinsU161);

              uint32_t vScalesU320 = aSm[0] & cKmask1;
              uint32_t vScalesU321 = (aSm[2] & cKmask2) | (((aSm[0] >> 6) & cKmask3) << 4);
              uint32_t vScalesU322 = aSm1[0] & cKmask1;
              uint32_t vScalesU323 = (aSm1[2] & cKmask2) | (((aSm1[0] >> 6) & cKmask3) << 4);

              svuint32_t vS01 = svdup_n_u32(vScalesU320);
              svuint32_t vS23 = svdup_n_u32(vScalesU321);
              svuint32_t vR01 = svdup_n_u32(vScalesU322);
              svuint32_t vR23 = svdup_n_u32(vScalesU323);

              svint8_t vS01B = svreinterpret_s8_u32(vS01);
              svint8_t vS23B = svreinterpret_s8_u32(vS23);
              svint8_t vR01B = svreinterpret_s8_u32(vR01);
              svint8_t vR23B = svreinterpret_s8_u32(vR23);

              svint32_t vS01D = svunpklo_s32(svunpklo_s16(svzip1_s8(vS01B, vS01B)));
              svint32_t vR01D = svunpklo_s32(svunpklo_s16(svzip1_s8(vR01B, vR01B)));
              svint32_t vS23D = svunpklo_s32(svunpklo_s16(svzip1_s8(vS23B, vS23B)));
              svint32_t vR23D = svunpklo_s32(svunpklo_s16(svzip1_s8(vR23B, vR23B)));

              vBlockScale0 = svtbl_s32(svzip1_s32(vS01D, vR01D), vIdx);
              vBlockScale1 = svtbl_s32(svzip2_s32(vS01D, vR01D), vIdx);
              vBlockScale2 = svtbl_s32(svzip1_s32(vS23D, vR23D), vIdx);
              vBlockScale3 = svtbl_s32(svzip2_s32(vS23D, vR23D), vIdx);
            }

            const int8_t *vQ8Base1 = vQ8Ptr[vB].qs + vSb * 256;

            // Load 32-byte per row pair, 1 subblock each time
            // predicate for activating higher lanes for 16 int8 elements
            const svbool_t cPh16 = svptrue_pat_b8(SV_VL16);
            // predicate for activating lower lanes for  16 int8 elements
            const svbool_t cPl16 = svnot_b_z(svptrue_b8(), cPh16);

            svint8_t vQ8Qs0 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 0), svld1_s8(cPl16, vQ8Base1 + 112));
            svint8_t vQ8Qs2 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 32), svld1_s8(cPl16, vQ8Base1 + 144));
            svint8_t vQ8Qs4 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 64), svld1_s8(cPl16, vQ8Base1 + 176));
            svint8_t vQ8Qs6 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 96), svld1_s8(cPl16, vQ8Base1 + 208));

            svint8_t vQ8Qs1 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 16), svld1_s8(cPl16, vQ8Base1 + 128));
            svint8_t vQ8Qs3 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 48), svld1_s8(cPl16, vQ8Base1 + 160));
            svint8_t vQ8Qs5 = svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 80), svld1_s8(cPl16, vQ8Base1 + 192));
            svint8_t vQ8Qs7 =
                svadd_s8_x(svptrue_b8(), svld1_s8(cPh16, vQ8Base1 + 112), svld1_s8(cPl16, vQ8Base1 + 224));

            // Q4s columns iterated in pairs (01, 23, 45, 67)
            for (int vCp = 0; vCp < cNcolsInterleaved / 2; vCp++) {

              vSbAcc0 = svdup_n_s32(0);
              vSbAcc2 = svdup_n_s32(0);

              svuint8_t vQ4QsCp00 = svld1rq_u8(svptrue_b8(), vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 0);
              svuint8_t vQ4QsCp01 = svld1rq_u8(svptrue_b8(), vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 64);
              svuint8_t vQ4QsCp02 = svld1rq_u8(svptrue_b8(), vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 128);
              svuint8_t vQ4QsCp03 = svld1rq_u8(svptrue_b8(), vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 192);

              svint8_t vQ4Nibbles00 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_u8_m(cPh16, vQ4QsCp00, cM4b1), 4));
              svint8_t vQ4Nibbles01 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_u8_m(cPh16, vQ4QsCp01, cM4b1), 4));
              svint8_t vQ4Nibbles02 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_u8_m(cPh16, vQ4QsCp02, cM4b1), 4));
              svint8_t vQ4Nibbles03 = svreinterpret_s8_u8(svlsr_n_u8_m(cPl16, svand_u8_m(cPh16, vQ4QsCp03, cM4b1), 4));

              vSbAcc0 = svmmla_s32(vSbAcc0, vQ4Nibbles00, vQ8Qs0);
              vSbAcc0 = svmmla_s32(vSbAcc0, vQ4Nibbles01, vQ8Qs2);

              vSbAcc0 = svmmla_s32(vSbAcc0, vQ4Nibbles02, vQ8Qs4);
              vSbAcc0 = svmmla_s32(vSbAcc0, vQ4Nibbles03, vQ8Qs6);

              vSbAcc2 = svmmla_s32(vSbAcc2, vQ4Nibbles00, vQ8Qs1);
              vSbAcc2 = svmmla_s32(vSbAcc2, vQ4Nibbles01, vQ8Qs3);

              vSbAcc2 = svmmla_s32(vSbAcc2, vQ4Nibbles02, vQ8Qs5);
              vSbAcc2 = svmmla_s32(vSbAcc2, vQ4Nibbles03, vQ8Qs7);

              if (vCp == 0) {
                vAcc00 = svmla_s32_m(svptrue_b32(), vAcc00, vSbAcc0, vBlockScale0);
                vAcc44 = svmla_s32_m(svptrue_b32(), vAcc44, vSbAcc2, vBlockScale0);
              }
              if (vCp == 1) {
                vAcc11 = svmla_s32_m(svptrue_b32(), vAcc11, vSbAcc0, vBlockScale1);
                vAcc55 = svmla_s32_m(svptrue_b32(), vAcc55, vSbAcc2, vBlockScale1);
              }
              if (vCp == 2) {
                vAcc22 = svmla_s32_m(svptrue_b32(), vAcc22, vSbAcc0, vBlockScale2);
                vAcc66 = svmla_s32_m(svptrue_b32(), vAcc66, vSbAcc2, vBlockScale2);
              }
              if (vCp == 3) {
                vAcc33 = svmla_s32_m(svptrue_b32(), vAcc33, vSbAcc0, vBlockScale3);
                vAcc77 = svmla_s32_m(svptrue_b32(), vAcc77, vSbAcc2, vBlockScale3);
              }
            }

            vBiasAcc00 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc00, svdup_n_s32(vBsumsArr32[vSb][0]), vQ4sbMins0);
            vBiasAcc00 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc00, svdup_n_s32(vBsumsArr32[vSb][1]), vQ4sbMins1);

            vBiasAcc22 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc22, svdup_n_s32(vBsumsArr32[vSb][2]), vQ4sbMins0);
            vBiasAcc22 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc22, svdup_n_s32(vBsumsArr32[vSb][3]), vQ4sbMins1);

            vBiasAcc44 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc44, svdup_n_s32(vBsumsArr32[vSb][4]), vQ4sbMins0);
            vBiasAcc44 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc44, svdup_n_s32(vBsumsArr32[vSb][5]), vQ4sbMins1);

            vBiasAcc66 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc66, svdup_n_s32(vBsumsArr32[vSb][6]), vQ4sbMins0);
            vBiasAcc66 = svmla_s32_m(svptrue_pat_b32(SV_VL8), vBiasAcc66, svdup_n_s32(vBsumsArr32[vSb][7]), vQ4sbMins1);
          } // for sb

          vAcc00 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc00, svext_s32(vAcc00, vAcc00, 4));
          vAcc11 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc11, svext_s32(vAcc11, vAcc11, 4));
          vAcc22 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc22, svext_s32(vAcc22, vAcc22, 4));
          vAcc33 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc33, svext_s32(vAcc33, vAcc33, 4));
          vAcc44 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc44, svext_s32(vAcc44, vAcc44, 4));
          vAcc55 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc55, svext_s32(vAcc55, vAcc55, 4));
          vAcc66 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc66, svext_s32(vAcc66, vAcc66, 4));
          vAcc77 = svadd_s32_z(svptrue_pat_b32(SV_VL4), vAcc77, svext_s32(vAcc77, vAcc77, 4));

          svint32_t vReorderAcc01 =
              svtbl_s32(svzip1_s32(svtrn1_s32(vAcc00, vAcc11), svtrn1_s32(vAcc22, vAcc33)), vIdx1);
          svint32_t vReorderAcc23 =
              svtbl_s32(svzip1_s32(svtrn2_s32(vAcc00, vAcc11), svtrn2_s32(vAcc22, vAcc33)), vIdx1);

          svint32_t vReorderAcc45 =
              svtbl_s32(svzip1_s32(svtrn1_s32(vAcc44, vAcc55), svtrn1_s32(vAcc66, vAcc77)), vIdx1);
          svint32_t vReorderAcc67 =
              svtbl_s32(svzip1_s32(svtrn2_s32(vAcc44, vAcc55), svtrn2_s32(vAcc66, vAcc77)), vIdx1);

          // Broadcast q8 scalar
          svfloat32_t vQ8D = svdup_f32(vQ8Ptr[vB].d[0]);

          svfloat32_t vQ4DminTemp = svcvt_f32_f16_x(
              svptrue_b32(),
              svzip1_f16(svld1_f16(svptrue_pat_b16(SV_VL8), (const __fp16 *)vQ4Ptr[vB].dmin), svdup_f16(0)));

          svfloat32_t vQ4DTemp = svcvt_f32_f16_x(
              svptrue_b32(),
              svzip1_f16(svld1_f16(svptrue_pat_b16(SV_VL8), (const __fp16 *)vQ4Ptr[vB].d), svdup_f16(0)));

          svfloat32_t vScale1 = svmul_f32_x(svptrue_b32(), vQ4DTemp, vQ8D);
          svfloat32_t vDmins1 = svmul_f32_x(svptrue_b32(), vQ4DminTemp, vQ8D);

          vAccF3201 = svmls_f32_m(svptrue_b32(), vAccF3201, svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vBiasAcc00),
                                  vDmins1);
          vAccF3201 = svmla_f32_m(svptrue_b32(), vAccF3201,
                                  svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vReorderAcc01), vScale1);

          vQ8D = svdup_f32(vQ8Ptr[vB].d[1]);

          vScale1 = svmul_f32_x(svptrue_b32(), vQ4DTemp, vQ8D);
          vDmins1 = svmul_f32_x(svptrue_b32(), vQ4DminTemp, vQ8D);

          vAccF3223 = svmls_f32_m(svptrue_b32(), vAccF3223, svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vBiasAcc22),
                                  vDmins1);
          vAccF3223 = svmla_f32_m(svptrue_b32(), vAccF3223,
                                  svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vReorderAcc23), vScale1);

          vQ8D = svdup_f32(vQ8Ptr[vB].d[2]);

          vScale1 = svmul_f32_x(svptrue_b32(), vQ4DTemp, vQ8D);
          vDmins1 = svmul_f32_x(svptrue_b32(), vQ4DminTemp, vQ8D);

          vAccF3245 = svmls_f32_m(svptrue_b32(), vAccF3245, svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vBiasAcc44),
                                  vDmins1);
          vAccF3245 = svmla_f32_m(svptrue_b32(), vAccF3245,
                                  svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vReorderAcc45), vScale1);

          vQ8D = svdup_f32(vQ8Ptr[vB].d[3]);

          vScale1 = svmul_f32_x(svptrue_b32(), vQ4DTemp, vQ8D);
          vDmins1 = svmul_f32_x(svptrue_b32(), vQ4DminTemp, vQ8D);

          vAccF3267 = svmls_f32_m(svptrue_b32(), vAccF3267, svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vBiasAcc66),
                                  vDmins1);
          vAccF3267 = svmla_f32_m(svptrue_b32(), vAccF3267,
                                  svcvt_f32_s32_m(svdup_n_f32(0), svptrue_b32(), vReorderAcc67), vScale1);

        } // for b

        // With the previous reorder, the tile is already in the correct memory layout.
        // Predicate for exactly 4 lanes
        svbool_t vPg4 = svptrue_pat_b32(SV_VL4);
        for (int vI = 0; vI < cQ8KBlocklen; vI++) {
          int vRow = vY * cQ8KBlocklen + vI;
          for (int vJ = 0; vJ < 2; vJ++) {
            int vCol = vX * cNcolsInterleaved + vJ * 4;
            int vOffset = vRow * pBs + vCol;

            if (vI == 0 && vJ == 0) {
              // acc_f32_0 → lower half of acc_f32_01
              svst1_f32(vPg4, pS + vOffset, vAccF3201);
            } else if (vI == 0 && vJ == 1) {
              // acc_f32_1 → upper half of acc_f32_01
              svst1_f32(vPg4, pS + vOffset, svext_f32(vAccF3201, vAccF3201, 4));
            } else if (vI == 1 && vJ == 0) {
              // acc_f32_2
              svst1_f32(vPg4, pS + vOffset, vAccF3223);
            } else if (vI == 1 && vJ == 1) {
              // acc_f32_3
              svst1_f32(vPg4, pS + vOffset, svext_f32(vAccF3223, vAccF3223, 4));
            } else if (vI == 2 && vJ == 0) {
              // acc_f32_4
              svst1_f32(vPg4, pS + vOffset, vAccF3245);
            } else if (vI == 2 && vJ == 1) {
              // acc_f32_5
              svst1_f32(vPg4, pS + vOffset, svext_f32(vAccF3245, vAccF3245, 4));
            } else if (vI == 3 && vJ == 0) {
              // acc_f32_6
              svst1_f32(vPg4, pS + vOffset, vAccF3267);
            } else if (vI == 3 && vJ == 1) {
              // acc_f32_7
              svst1_f32(vPg4, pS + vOffset, svext_f32(vAccF3267, vAccF3267, 4));
            }
          }
        }
      } // for x
    } // for y
    return;
  }
#endif // SVE compile-time end

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  constexpr int cQ8KBlocklen = 4;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);

  // 8 accumulators: 2 row pairs × 4 col pairs
  float32x4_t vAccF32[cBlocklen];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_Kx8 *GGML_RESTRICT vQ4Ptr = (const block_q4_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cBlocklen; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        // bsums pairs belongs to the same q8_k subblock
        const int16x8_t cBsums[4]{
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3 + 8)),
        };
        int16_t vBsumsArr[4][8];
        for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
          vst1q_s16(vBsumsArr[vQ8Row], cBsums[vQ8Row]);
        }

        int32x4_t aSbAcc[4];   // Aux accumulators to store subblock (partial) results
        int32x4_t vAcc[8];     // rows 01 stored in [0][1][2][3] rows 23 stored in [4][5][6][7]
        int32x4_t aBiasAcc[8]; // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567, [2]->r1 0123 ...
        for (int vI = 0; vI < 8; vI++) {
          vAcc[vI] = vdupq_n_s32(0);
          aBiasAcc[vI] = vdupq_n_s32(0);
        }

        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          // Need scales for the low and high nibbles
          // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
          int8_t aQ4sbScales[2][8];
          int16x8_t aQ4sbMins[2]; // int16 as its needed for bias_acc later
          for (int vI = 0; vI < 2; vI++) {
            const int cOffset = vSb * 24 + vI * 12;
            fDecodeQKx86bitScales(&vQ4Ptr[vB].scales[cOffset], &aQ4sbMins[vI], aQ4sbScales[vI]);
          }

          // q8_ptr[b].qs has interleaved Q8 rows (01, 23)
          const int8_t *vQ8Base = vQ8Ptr[vB].qs + vSb * 256;

          int8x16_t aQ8Qs01[8];
          int8x16_t aQ8Qs23[8];

          // Load 32-byte per row pair, 1 subblock each time
          for (int vI = 0; vI < 8; vI++) {
            const int cOffset = vI * 32; // 16 for row 01, 16 for row 23
            aQ8Qs01[vI] = vld1q_s8(vQ8Base + cOffset);
            aQ8Qs23[vI] = vld1q_s8(vQ8Base + cOffset + 16);
          }

          const int8x16_t aQ8s[2][8] = {
              {aQ8Qs01[0], aQ8Qs01[1], aQ8Qs01[2], aQ8Qs01[3], aQ8Qs01[4], aQ8Qs01[5], aQ8Qs01[6], aQ8Qs01[7]},
              {aQ8Qs23[0], aQ8Qs23[1], aQ8Qs23[2], aQ8Qs23[3], aQ8Qs23[4], aQ8Qs23[5], aQ8Qs23[6], aQ8Qs23[7]},
          };

          // Q4s columns iterated in pairs (01, 23, 45, 67)
          for (int vCp = 0; vCp < cNcolsInterleaved / 2; vCp++) {
            for (int vI = 0; vI < 4; vI++) {
              aSbAcc[vI] = vdupq_n_s32(0);
            }

            uint8x16_t vQ4QsCp0 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 0);   // 0 .. 7 & 32..39
            uint8x16_t vQ4QsCp1 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 64);  // 8 ..15 & 40..47
            uint8x16_t vQ4QsCp2 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 128); // 16..23 & 48..55
            uint8x16_t vQ4QsCp3 = vld1q_u8(vQ4Ptr[vB].qs + vSb * QK_K + 16 * vCp + 192); // 24..31 & 56..63
            const int8x16_t aQ4Nibbles[2][4] = {{
                                                    vreinterpretq_s8_u8(vandq_u8(vQ4QsCp0, cM4b)),
                                                    vreinterpretq_s8_u8(vandq_u8(vQ4QsCp1, cM4b)),
                                                    vreinterpretq_s8_u8(vandq_u8(vQ4QsCp2, cM4b)),
                                                    vreinterpretq_s8_u8(vandq_u8(vQ4QsCp3, cM4b)),
                                                },
                                                {
                                                    vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp0, 4)),
                                                    vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp1, 4)),
                                                    vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp2, 4)),
                                                    vreinterpretq_s8_u8(vshrq_n_u8(vQ4QsCp3, 4)),
                                                }};

            // Calculates the Qs muladd of every row pair (rp) rows 01 and 23 of q8
            // for each of the internal 32 qs subblock (blk)
            for (int vRp = 0; vRp < 2; vRp++) {
              for (int vBlk = 0; vBlk < 2; vBlk++) {
                const int8x16_t *vQ8 = &aQ8s[vRp][4 * vBlk];
                const int8x16_t *vQ4 = aQ4Nibbles[vBlk];
                int32x4_t vAcc = aSbAcc[2 * vRp + vBlk];
                // mul add for each qs in the same subblock
                for (int vQsOffset = 0; vQsOffset < 4; vQsOffset++) {
                  vAcc = vmmlaq_s32(vAcc, vQ4[vQsOffset], vQ8[vQsOffset]);
                }
                aSbAcc[2 * vRp + vBlk] = vAcc;
              }
            }

            // Scales[i] corresponds to column i
            const int cScaleOffset = vCp * 2;
            const int32_t cScale00 = aQ4sbScales[0][cScaleOffset];
            const int32_t cScale01 = aQ4sbScales[0][cScaleOffset + 1];
            const int32_t cScale10 = aQ4sbScales[1][cScaleOffset];
            const int32_t cScale11 = aQ4sbScales[1][cScaleOffset + 1];
            const int32x4_t cBlockScale0 = vcombine_s32(vdup_n_s32(cScale00), vdup_n_s32(cScale01));
            const int32x4_t cBlockScale1 = vcombine_s32(vdup_n_s32(cScale10), vdup_n_s32(cScale11));

            vAcc[vCp] = vmlaq_s32(vAcc[vCp], aSbAcc[0], cBlockScale0);
            vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], aSbAcc[2], cBlockScale0);
            vAcc[vCp] = vmlaq_s32(vAcc[vCp], aSbAcc[1], cBlockScale1);
            vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], aSbAcc[3], cBlockScale1);
          }

          // Multiply Acc bsum + mins
          for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
            // Each pair of subblocks share the same bsums
            // Load scalar bsum → broadcast to a vector (vdupq_n_s16(s)).
            int16x4_t vBsumsVecLo = vdup_n_s16(vBsumsArr[vSb][vQ8Row * 2]);
            int16x4_t vBsumsVecHi = vdup_n_s16(vBsumsArr[vSb][vQ8Row * 2 + 1]);

            aBiasAcc[2 * vQ8Row] = vmlal_s16(aBiasAcc[2 * vQ8Row], vBsumsVecLo, vget_low_s16(aQ4sbMins[0]));
            aBiasAcc[2 * vQ8Row] = vmlal_s16(aBiasAcc[2 * vQ8Row], vBsumsVecHi, vget_low_s16(aQ4sbMins[1]));
            aBiasAcc[2 * vQ8Row + 1] = vmlal_s16(aBiasAcc[2 * vQ8Row + 1], vBsumsVecLo, vget_high_s16(aQ4sbMins[0]));
            aBiasAcc[2 * vQ8Row + 1] = vmlal_s16(aBiasAcc[2 * vQ8Row + 1], vBsumsVecHi, vget_high_s16(aQ4sbMins[1]));
          }
        } // for sb

        // Reorder of i8mm output with bias and output layout
        for (int vI = 0; vI < 8; vI++) {
          int32x2x2_t vAux = vzip_s32(vget_low_s32(vAcc[vI]), vget_high_s32(vAcc[vI]));
          vAcc[vI] = vcombine_s32(vAux.val[0], vAux.val[1]);
        }
        int32x4_t aReorderAcc[8] = {
            vcombine_s32(vget_low_s32(vAcc[0]), vget_low_s32(vAcc[1])),
            vcombine_s32(vget_low_s32(vAcc[2]), vget_low_s32(vAcc[3])),
            vcombine_s32(vget_high_s32(vAcc[0]), vget_high_s32(vAcc[1])),
            vcombine_s32(vget_high_s32(vAcc[2]), vget_high_s32(vAcc[3])),
            vcombine_s32(vget_low_s32(vAcc[4]), vget_low_s32(vAcc[5])),
            vcombine_s32(vget_low_s32(vAcc[6]), vget_low_s32(vAcc[7])),
            vcombine_s32(vget_high_s32(vAcc[4]), vget_high_s32(vAcc[5])),
            vcombine_s32(vget_high_s32(vAcc[6]), vget_high_s32(vAcc[7])),
        };

        for (int vI = 0; vI < cQ8KBlocklen; vI++) {
          for (int vJ = 0; vJ < 2; vJ++) {
            float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d[vI]);
            float32x4_t vQ4Dmin = vcvt_f32_f16(vld1_f16((const __fp16 *)(vQ4Ptr[vB].dmin + vJ * 4)));
            const float32x4_t cDmins = vmulq_f32(vQ4Dmin, vQ8D);

            float32x4_t vQ4D = vcvt_f32_f16(vld1_f16((const __fp16 *)(vQ4Ptr[vB].d + vJ * 4)));
            const float32x4_t cScale = vmulq_f32(vQ4D, vQ8D);

            vAccF32[2 * vI + vJ] = vmlsq_f32(vAccF32[2 * vI + vJ], vcvtq_f32_s32(aBiasAcc[2 * vI + vJ]), cDmins);
            vAccF32[2 * vI + vJ] = vmlaq_f32(vAccF32[2 * vI + vJ], vcvtq_f32_s32(aReorderAcc[2 * vI + vJ]), cScale);
          }
        }
      } // for b

      // With the previous reorder, the tile is already in the correct memory layout.
      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  ggml_gemm_q4_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q5_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  constexpr int cQ8KBlocklen = 4;
  constexpr int cColPairs = cNcolsInterleaved / 2;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMone = vdupq_n_u8(1);
  const uint8x16_t cMtwo = vdupq_n_u8(2);

  // 8 accumulators: 2 row pairs × 4 col pairs
  float32x4_t vAccF32[cBlocklen];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q5_Kx8 *GGML_RESTRICT vQ5Ptr = (const block_q5_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cBlocklen; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        // bsums pairs belongs to the same q8_k subblock
        const int16x8_t cBsums[4]{
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 0 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 1 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 2 + 8)),
            vpaddq_s16(vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3), vld1q_s16(vQ8Ptr[vB].bsums + 16 * 3 + 8)),
        };
        int16_t vBsumsArr[4][8];
        for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
          vst1q_s16(vBsumsArr[vQ8Row], cBsums[vQ8Row]);
        }

        int32x4_t aSbAcc[4];   // Aux accumulators to store subblock (partial) results
        int32x4_t vAcc[8];     // rows 01 stored in [0][1][2][3] rows 23 stored in [4][5][6][7]
        int32x4_t aBiasAcc[8]; // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567, [2]->r1 0123 ...
        for (int vI = 0; vI < 8; vI++) {
          vAcc[vI] = vdupq_n_s32(0);
          aBiasAcc[vI] = vdupq_n_s32(0);
        }

        // Load qh once per block and shift after each subblock
        const uint8_t *vQhBase = vQ5Ptr[vB].qh;
        uint8x16_t vQh[cColPairs][4];
        for (int vCp = 0; vCp < cColPairs; vCp++) {
          vQh[vCp][0] = vld1q_u8(vQhBase + 16 * vCp);
          vQh[vCp][1] = vld1q_u8(vQhBase + 16 * vCp + 64);
          vQh[vCp][2] = vld1q_u8(vQhBase + 16 * vCp + 128);
          vQh[vCp][3] = vld1q_u8(vQhBase + 16 * vCp + 192);
        }

        for (int vSb = 0; vSb < QK_K / 64; vSb++) {
          // Need scales for the low and high nibbles
          // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
          int8_t aQ5sbScales[2][8];
          int16x8_t aQ5sbMins[2]; // int16 as its needed for bias_acc later
          for (int vI = 0; vI < 2; vI++) {
            const int cOffset = vSb * 24 + vI * 12;
            fDecodeQKx86bitScales(&vQ5Ptr[vB].scales[cOffset], &aQ5sbMins[vI], aQ5sbScales[vI]);
          }

          // q8_ptr[b].qs has interleaved Q8 rows (01, 23)
          const int8_t *vQ8Base = vQ8Ptr[vB].qs + vSb * 256;

          int8x16_t aQ8Qs01[8];
          int8x16_t aQ8Qs23[8];

          // Load 32-byte per row pair, 1 subblock each time
          for (int vI = 0; vI < 8; vI++) {
            const int cOffset = vI * 32; // 16 for row 01, 16 for row 23
            aQ8Qs01[vI] = vld1q_s8(vQ8Base + cOffset);
            aQ8Qs23[vI] = vld1q_s8(vQ8Base + cOffset + 16);
          }

          const int8x16_t aQ8s[2][8] = {
              {aQ8Qs01[0], aQ8Qs01[1], aQ8Qs01[2], aQ8Qs01[3], aQ8Qs01[4], aQ8Qs01[5], aQ8Qs01[6], aQ8Qs01[7]},
              {aQ8Qs23[0], aQ8Qs23[1], aQ8Qs23[2], aQ8Qs23[3], aQ8Qs23[4], aQ8Qs23[5], aQ8Qs23[6], aQ8Qs23[7]},
          };

          // Q5s columns iterated in pairs (01, 23, 45, 67)
          for (int vCp = 0; vCp < cColPairs; vCp++) {
            for (int vI = 0; vI < 4; vI++) {
              aSbAcc[vI] = vdupq_n_s32(0);
            }

            uint8x16_t vQsCp0 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 16 * vCp + 0);   // 0 .. 7 & 32..39
            uint8x16_t vQsCp1 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 16 * vCp + 64);  // 8 ..15 & 40..47
            uint8x16_t vQsCp2 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 16 * vCp + 128); // 16..23 & 48..55
            uint8x16_t vQsCp3 = vld1q_u8(vQ5Ptr[vB].qs + vSb * QK_K + 16 * vCp + 192); // 24..31 & 56..63

            // This is the only part of the algorithm that differs with Q4_K
            // Extract High bits and pack into 5 bit weights
            uint8x16_t vHbitLo0 = vandq_u8(vQh[vCp][0], cMone);
            uint8x16_t vHbitHi0 = vshlq_n_u8(vandq_u8(vQh[vCp][0], cMtwo), 3);
            vQh[vCp][0] = vshrq_n_u8(vQh[vCp][0], 2);
            // Same as Q4_K, i8mm to dequantize the weights.
            const int8x16_t cQsLo0 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQsCp0, cM4b), vHbitLo0, 4));
            int32x4_t vAcc0 = aSbAcc[0];
            vAcc0 = vmmlaq_s32(vAcc0, cQsLo0, aQ8s[0][0]);
            int32x4_t vAcc2 = aSbAcc[2];
            vAcc2 = vmmlaq_s32(vAcc2, cQsLo0, aQ8s[1][0]);
            const int8x16_t cQsHi0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQsCp0, 4), vHbitHi0));
            int32x4_t vAcc1 = aSbAcc[1];
            vAcc1 = vmmlaq_s32(vAcc1, cQsHi0, aQ8s[0][4]);
            int32x4_t vAcc3 = aSbAcc[3];
            vAcc3 = vmmlaq_s32(vAcc3, cQsHi0, aQ8s[1][4]);

            // Repeat for the other 3 columns (8..15, 16..23, 24..31)
            uint8x16_t vHbitHi1 = vshlq_n_u8(vandq_u8(vQh[vCp][1], cMtwo), 3);
            uint8x16_t vHbitLo1 = vandq_u8(vQh[vCp][1], cMone);
            vQh[vCp][1] = vshrq_n_u8(vQh[vCp][1], 2);
            const int8x16_t cQsLo1 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQsCp1, cM4b), vHbitLo1, 4));
            vAcc0 = vmmlaq_s32(vAcc0, cQsLo1, aQ8s[0][1]);
            vAcc2 = vmmlaq_s32(vAcc2, cQsLo1, aQ8s[1][1]);
            const int8x16_t cQsHi1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQsCp1, 4), vHbitHi1));
            vAcc1 = vmmlaq_s32(vAcc1, cQsHi1, aQ8s[0][5]);
            vAcc3 = vmmlaq_s32(vAcc3, cQsHi1, aQ8s[1][5]);

            uint8x16_t vHbitHi2 = vshlq_n_u8(vandq_u8(vQh[vCp][2], cMtwo), 3);
            uint8x16_t vHbitLo2 = vandq_u8(vQh[vCp][2], cMone);
            vQh[vCp][2] = vshrq_n_u8(vQh[vCp][2], 2);
            const int8x16_t cQsLo2 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQsCp2, cM4b), vHbitLo2, 4));
            vAcc0 = vmmlaq_s32(vAcc0, cQsLo2, aQ8s[0][2]);
            vAcc2 = vmmlaq_s32(vAcc2, cQsLo2, aQ8s[1][2]);
            const int8x16_t cQsHi2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQsCp2, 4), vHbitHi2));
            vAcc1 = vmmlaq_s32(vAcc1, cQsHi2, aQ8s[0][6]);
            vAcc3 = vmmlaq_s32(vAcc3, cQsHi2, aQ8s[1][6]);

            uint8x16_t vHbitLo3 = vandq_u8(vQh[vCp][3], cMone);
            uint8x16_t vHbitHi3 = vshlq_n_u8(vandq_u8(vQh[vCp][3], cMtwo), 3);
            vQh[vCp][3] = vshrq_n_u8(vQh[vCp][3], 2);
            const int8x16_t cQsLo3 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQsCp3, cM4b), vHbitLo3, 4));
            vAcc0 = vmmlaq_s32(vAcc0, cQsLo3, aQ8s[0][3]);
            aSbAcc[0] = vAcc0;
            vAcc2 = vmmlaq_s32(vAcc2, cQsLo3, aQ8s[1][3]);
            aSbAcc[2] = vAcc2;

            // Scales[i] corresponds to column i
            const int cScaleOffset = vCp * 2;
            const int32_t cS0 = aQ5sbScales[0][cScaleOffset];
            const int32_t cS1 = aQ5sbScales[0][cScaleOffset + 1];
            const int32x4_t cBlockScale = vcombine_s32(vdup_n_s32(cS0), vdup_n_s32(cS1));
            vAcc[vCp] = vmlaq_s32(vAcc[vCp], aSbAcc[0], cBlockScale);
            vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], aSbAcc[2], cBlockScale);

            const int8x16_t cQsHi3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQsCp3, 4), vHbitHi3));
            vAcc1 = vmmlaq_s32(vAcc1, cQsHi3, aQ8s[0][7]);
            aSbAcc[1] = vAcc1;
            vAcc3 = vmmlaq_s32(vAcc3, cQsHi3, aQ8s[1][7]);
            aSbAcc[3] = vAcc3;

            const int32_t cS2 = aQ5sbScales[1][cScaleOffset];
            const int32_t cS3 = aQ5sbScales[1][cScaleOffset + 1];
            const int32x4_t cBlockScale2 = vcombine_s32(vdup_n_s32(cS2), vdup_n_s32(cS3));
            vAcc[vCp] = vmlaq_s32(vAcc[vCp], aSbAcc[1], cBlockScale2);
            vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], aSbAcc[3], cBlockScale2);
          }

          // Multiply Acc bsum + mins
          for (int vQ8Row = 0; vQ8Row < 4; vQ8Row++) {
            // Each pair of subblocks share the same bsums
            // Load scalar bsum → broadcast to a vector (vdupq_n_s16(s)).
            int16x4_t vBsumsVecLo = vdup_n_s16(vBsumsArr[vSb][vQ8Row * 2]);
            int16x4_t vBsumsVecHi = vdup_n_s16(vBsumsArr[vSb][vQ8Row * 2 + 1]);

            aBiasAcc[2 * vQ8Row] = vmlal_s16(aBiasAcc[2 * vQ8Row], vBsumsVecLo, vget_low_s16(aQ5sbMins[0]));
            aBiasAcc[2 * vQ8Row] = vmlal_s16(aBiasAcc[2 * vQ8Row], vBsumsVecHi, vget_low_s16(aQ5sbMins[1]));
            aBiasAcc[2 * vQ8Row + 1] = vmlal_s16(aBiasAcc[2 * vQ8Row + 1], vBsumsVecLo, vget_high_s16(aQ5sbMins[0]));
            aBiasAcc[2 * vQ8Row + 1] = vmlal_s16(aBiasAcc[2 * vQ8Row + 1], vBsumsVecHi, vget_high_s16(aQ5sbMins[1]));
          }
        } // for sb

        // Reorder of i8mm output with bias and output layout
        for (int vI = 0; vI < 8; vI++) {
          int32x2x2_t vAux = vzip_s32(vget_low_s32(vAcc[vI]), vget_high_s32(vAcc[vI]));
          vAcc[vI] = vcombine_s32(vAux.val[0], vAux.val[1]);
        }
        int32x4_t aReorderAcc[8] = {
            vcombine_s32(vget_low_s32(vAcc[0]), vget_low_s32(vAcc[1])),
            vcombine_s32(vget_low_s32(vAcc[2]), vget_low_s32(vAcc[3])),
            vcombine_s32(vget_high_s32(vAcc[0]), vget_high_s32(vAcc[1])),
            vcombine_s32(vget_high_s32(vAcc[2]), vget_high_s32(vAcc[3])),
            vcombine_s32(vget_low_s32(vAcc[4]), vget_low_s32(vAcc[5])),
            vcombine_s32(vget_low_s32(vAcc[6]), vget_low_s32(vAcc[7])),
            vcombine_s32(vget_high_s32(vAcc[4]), vget_high_s32(vAcc[5])),
            vcombine_s32(vget_high_s32(vAcc[6]), vget_high_s32(vAcc[7])),
        };

        for (int vI = 0; vI < cQ8KBlocklen; vI++) {
          for (int vJ = 0; vJ < 2; vJ++) {
            float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d[vI]);
            float32x4_t vQ5Dmin = vcvt_f32_f16(vld1_f16((const __fp16 *)(vQ5Ptr[vB].dmin + vJ * 4)));
            const float32x4_t cDmins = vmulq_f32(vQ5Dmin, vQ8D);

            float32x4_t vQ5D = vcvt_f32_f16(vld1_f16((const __fp16 *)(vQ5Ptr[vB].d + vJ * 4)));
            const float32x4_t cScale = vmulq_f32(vQ5D, vQ8D);

            vAccF32[2 * vI + vJ] = vmlsq_f32(vAccF32[2 * vI + vJ], vcvtq_f32_s32(aBiasAcc[2 * vI + vJ]), cDmins);
            vAccF32[2 * vI + vJ] = vmlaq_f32(vAccF32[2 * vI + vJ], vcvtq_f32_s32(aReorderAcc[2 * vI + vJ]), cScale);
          }
        }
      } // for b

      // With the previous reorder, the tile is already in the correct memory layout.
      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  ggml_gemm_q5_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q6_K_8x4_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  constexpr int cQ8KBlocklen = 4;
  constexpr int cColGroups = cNcolsInterleaved / 4;
  constexpr int cAccSize = cQ8KBlocklen * cColGroups; // 4 rows, 2 column groups
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMaskLo = vdupq_n_u8(0x03);
  const uint8x16_t cMaskHi = vdupq_n_u8(0x30);
  const int8x16_t cM32s = vdupq_n_s8(32);

  float32x4_t vAccF32[cAccSize];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q6_Kx8 *GGML_RESTRICT vQ6Ptr = (const block_q6_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cAccSize; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        float32x4_t vQ6D0123 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d));
        float32x4_t vQ6D4567 = vcvt_f32_f16(vld1_f16((const __fp16 *)vQ6Ptr[vB].d + 4));
        float32x4_t vQ8D0123 = vld1q_f32(vQ8Ptr[vB].d);

        float32x4_t aSbdScale0123[cQ8KBlocklen];
        float32x4_t aSbdScale4567[cQ8KBlocklen];

        aSbdScale0123[0] = vmulq_laneq_f32(vQ6D0123, vQ8D0123, 0);
        aSbdScale4567[0] = vmulq_laneq_f32(vQ6D4567, vQ8D0123, 0);
        aSbdScale0123[1] = vmulq_laneq_f32(vQ6D0123, vQ8D0123, 1);
        aSbdScale4567[1] = vmulq_laneq_f32(vQ6D4567, vQ8D0123, 1);
        aSbdScale0123[2] = vmulq_laneq_f32(vQ6D0123, vQ8D0123, 2);
        aSbdScale4567[2] = vmulq_laneq_f32(vQ6D4567, vQ8D0123, 2);
        aSbdScale0123[3] = vmulq_laneq_f32(vQ6D0123, vQ8D0123, 3);
        aSbdScale4567[3] = vmulq_laneq_f32(vQ6D4567, vQ8D0123, 3);

        int32x4_t aAccS32[cAccSize];
        for (int vI = 0; vI < cAccSize; vI++) {
          aAccS32[vI] = vdupq_n_s32(0);
        }

        int16_t vQ6Scales[8 * 16];
        for (int vI = 0; vI < 16; vI++) {
          int16x8_t vScales = vmovl_s8(vld1_s8(vQ6Ptr[vB].scales + vI * 8));
          vst1q_s16(vQ6Scales + vI * 8, vScales);
        }

        for (int vHalf = 0; vHalf < 2; vHalf++) {
          const uint8_t *vQlBase = vQ6Ptr[vB].ql + vHalf * 512;
          const uint8_t *vQhBase = vQ6Ptr[vB].qh + vHalf * 256;

          for (int vSb = 0; vSb < QK_K / 64; vSb++) {
            int32x4_t vAccLo[cAccSize];
            int32x4_t vAccHi[cAccSize];
            for (int vI = 0; vI < cAccSize; vI++) {
              vAccLo[vI] = vdupq_n_s32(0);
              vAccHi[vI] = vdupq_n_s32(0);
            }

            const int8_t *vQ8BaseL = vQ8Ptr[vB].qs + vHalf * 512 + vSb * 64;
            const int8_t *vQ8BaseH = vQ8Ptr[vB].qs + vHalf * 512 + 256 + vSb * 64;

            // 4 rows * 16 elements per scale
            // 4 reads of 16 bytes each
            constexpr int cReadsPerSb = 4;
            int8x16_t vQ8L[cReadsPerSb];
            int8x16_t vQ8H[cReadsPerSb];
            for (int vK = 0; vK < cReadsPerSb; vK++) {
              vQ8L[vK] = vld1q_s8(vQ8BaseL + 16 * vK);
              vQ8H[vK] = vld1q_s8(vQ8BaseH + 16 * vK);
            }

            const int cQlOffBase = vSb * QK_K / 2;
            const int cQhOffBase = cQlOffBase & 255;

            uint8x16_t vQ6Ql0123[cReadsPerSb];
            uint8x16_t vQ6Ql4567[cReadsPerSb];
            uint8x16_t vQ6Qh0123[cReadsPerSb];
            uint8x16_t vQ6Qh4567[cReadsPerSb];

            for (int vK = 0; vK < cReadsPerSb; vK++) {
              vQ6Ql0123[vK] = vld1q_u8(vQlBase + cQlOffBase + vK * 32);
              vQ6Ql4567[vK] = vld1q_u8(vQlBase + cQlOffBase + vK * 32 + 16);
              vQ6Qh0123[vK] = vld1q_u8(vQhBase + cQhOffBase + vK * 32);
              vQ6Qh4567[vK] = vld1q_u8(vQhBase + cQhOffBase + vK * 32 + 16);
            }

            if (vSb > 1) {
              for (int vK = 0; vK < cReadsPerSb; vK++) {
                vQ6Qh0123[vK] = vshrq_n_u8(vQ6Qh0123[vK], 2);
                vQ6Qh4567[vK] = vshrq_n_u8(vQ6Qh4567[vK], 2);
              }
            }

            for (int vK = 0; vK < cReadsPerSb; vK++) {
              // q = (ql | qh) - 32
              const uint8x16_t cHbitLo0123 = vandq_u8(vQ6Qh0123[vK], cMaskLo);
              const uint8x16_t cHbitHi0123 = vandq_u8(vQ6Qh0123[vK], cMaskHi);
              const uint8x16_t cHbitLo4567 = vandq_u8(vQ6Qh4567[vK], cMaskLo);
              const uint8x16_t cHbitHi4567 = vandq_u8(vQ6Qh4567[vK], cMaskHi);

              const int8x16_t cQ60123Lo =
                  vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQ6Ql0123[vK], cM4b), cHbitLo0123, 4)), cM32s);
              const int8x16_t cQ60123Hi =
                  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQ6Ql0123[vK], 4), cHbitHi0123)), cM32s);

              vAccLo[0] = vdotq_laneq_s32(vAccLo[0], cQ60123Lo, vQ8L[vK], 0); //  0..3  r0 c0123
              vAccLo[1] = vdotq_laneq_s32(vAccLo[1], cQ60123Lo, vQ8L[vK], 1); //  0..3  r1 c0123
              vAccLo[2] = vdotq_laneq_s32(vAccLo[2], cQ60123Lo, vQ8L[vK], 2); //  0..3  r2 c0123
              vAccLo[3] = vdotq_laneq_s32(vAccLo[3], cQ60123Lo, vQ8L[vK], 3); //  0..3  r3 c0123

              vAccHi[0] = vdotq_laneq_s32(vAccHi[0], cQ60123Hi, vQ8H[vK], 0); // 64..67 r0 c0123
              vAccHi[1] = vdotq_laneq_s32(vAccHi[1], cQ60123Hi, vQ8H[vK], 1); // 64..67 r1 c0123
              vAccHi[2] = vdotq_laneq_s32(vAccHi[2], cQ60123Hi, vQ8H[vK], 2); // 64..67 r2 c0123
              vAccHi[3] = vdotq_laneq_s32(vAccHi[3], cQ60123Hi, vQ8H[vK], 3); // 64..67 r3 c0123

              const int8x16_t cQ64567Lo =
                  vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(vQ6Ql4567[vK], cM4b), cHbitLo4567, 4)), cM32s);
              const int8x16_t cQ64567Hi =
                  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(vQ6Ql4567[vK], 4), cHbitHi4567)), cM32s);

              vAccLo[4] = vdotq_laneq_s32(vAccLo[4], cQ64567Lo, vQ8L[vK], 0); //  0..3  r0 c4567
              vAccLo[5] = vdotq_laneq_s32(vAccLo[5], cQ64567Lo, vQ8L[vK], 1); //  0..3  r1 c4567
              vAccLo[6] = vdotq_laneq_s32(vAccLo[6], cQ64567Lo, vQ8L[vK], 2); //  0..3  r2 c4567
              vAccLo[7] = vdotq_laneq_s32(vAccLo[7], cQ64567Lo, vQ8L[vK], 3); //  0..3  r3 c4567

              vAccHi[4] = vdotq_laneq_s32(vAccHi[4], cQ64567Hi, vQ8H[vK], 0); // 64..67 r0 c4567
              vAccHi[5] = vdotq_laneq_s32(vAccHi[5], cQ64567Hi, vQ8H[vK], 1); // 64..67 r1 c4567
              vAccHi[6] = vdotq_laneq_s32(vAccHi[6], cQ64567Hi, vQ8H[vK], 2); // 64..67 r2 c4567
              vAccHi[7] = vdotq_laneq_s32(vAccHi[7], cQ64567Hi, vQ8H[vK], 3); // 64..67 r3 c4567
            }

            // Scale and bias
            const int cScaleIdxL = vHalf * 8 + vSb;
            const int cScaleIdxH = vHalf * 8 + vSb + 4;

            for (int vG = 0; vG < cColGroups; vG++) {
              const int16x4_t cScalesL16 = vld1_s16(vQ6Scales + cScaleIdxL * 8 + vG * 4);
              const int16x4_t cScalesH16 = vld1_s16(vQ6Scales + cScaleIdxH * 8 + vG * 4);
              const int32x4_t cScaleVecL = vmovl_s16(cScalesL16);
              const int32x4_t cScaleVecH = vmovl_s16(cScalesH16);
              const int cAccOffset = vG * cQ8KBlocklen;

              for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
                const int cIdx = vRow * 2 + vG;
                aAccS32[cIdx] = vmlaq_s32(aAccS32[cIdx], vAccLo[cAccOffset + vRow], cScaleVecL);
                aAccS32[cIdx] = vmlaq_s32(aAccS32[cIdx], vAccHi[cAccOffset + vRow], cScaleVecH);
              }
            }
          }
        }

        // Finally we apply the superblock scales
        for (int vRow = 0; vRow < cQ8KBlocklen; vRow++) {
          const int cIdx0 = 2 * vRow;
          const int cIdx1 = 2 * vRow + 1;
          const int32x4_t cAcc0123 = aAccS32[cIdx0];
          const int32x4_t cAcc4567 = aAccS32[cIdx1];

          vAccF32[cIdx0] = vmlaq_f32(vAccF32[cIdx0], vcvtq_f32_s32(cAcc0123), aSbdScale0123[vRow]);
          vAccF32[cIdx1] = vmlaq_f32(vAccF32[cIdx1], vcvtq_f32_s32(cAcc4567), aSbdScale4567[vRow]);
        }
      } // for b

      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemm_q6_K_8x4_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q6_K_8x8_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  constexpr int cQk = QK_K;
  const int cNb = pN / cQk;

  constexpr int cNcolsInterleaved = 8;
  constexpr int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  constexpr int cQ8KBlocklen = 4;
  const uint8x16_t cM4b = vdupq_n_u8(0x0f);
  const uint8x16_t cMaskLo = vdupq_n_u8(0x03);
  const uint8x16_t cMaskHi = vdupq_n_u8(0x30);
  const int8x16_t cM32s = vdupq_n_s8(32);

  // 8 accumulators: 4 q8 rows × 2 col groups (0-3, 4-7)
  float32x4_t vAccF32[cBlocklen];

  for (int vY = 0; vY < pNr / cQ8KBlocklen; vY++) {
    const block_q8_Kx4 *GGML_RESTRICT vQ8Ptr = (const block_q8_Kx4 *)pVy + (vY * cNb);

    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q6_Kx8 *GGML_RESTRICT vQ6Ptr = (const block_q6_Kx8 *)pVx + (vX * cNb);

      for (int vI = 0; vI < cBlocklen; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        int32x4_t vAcc[8]; // rows 01 stored in [0][1][2][3], rows 23 stored in [4][5][6][7]
        for (int vI = 0; vI < 8; vI++) {
          vAcc[vI] = vdupq_n_s32(0);
        }

        // Q6_K has simple 8-bit scales, 16 per block (one per 16 values)
        // Reused for bias and dequantization later
        int16_t vQ6Scales[16 * 8];
        for (int vI = 0; vI < 16; ++vI) {
          int16x8_t vS16 = vmovl_s8(vld1_s8(vQ6Ptr[vB].scales + vI * 8));
          vst1q_s16(vQ6Scales + vI * 8, vS16);
        }

        // Process two 128-value halves per superblock
        for (int vHalf = 0; vHalf < 2; vHalf++) {

          const uint8_t *vQlBase = vQ6Ptr[vB].ql + vHalf * 512;
          const uint8_t *vQhBase = vQ6Ptr[vB].qh + vHalf * 256;

          // A subblock (sb) is a set of weights that share the scale
          // Since q6_K scales are per 16 elements
          // num sbs -> 256 elements / (16 elements/scale * 2 elements/byte * 2 halves)
          for (int vSb = 0; vSb < QK_K / 64; vSb++) {
            // Q6_K weight index increasing by 64 instead of 32 requires
            // loading various q8 memory regions
            const int8_t *vQ8BaseL = vQ8Ptr[vB].qs + vHalf * 512 + vSb * 64;
            const int8_t *vQ8BaseH = vQ8Ptr[vB].qs + vHalf * 512 + 256 + vSb * 64;

            int8x16_t aQ8L01[2];
            int8x16_t aQ8L23[2];
            for (int vI = 0; vI < 2; vI++) {
              const int cOffset = vI * 32;
              aQ8L01[vI] = vld1q_s8(vQ8BaseL + cOffset);      // 0..7 & 8..15 (r01)
              aQ8L23[vI] = vld1q_s8(vQ8BaseL + cOffset + 16); // 0..7 & 8..15 (r23)
            }

            int8x16_t aQ8H01[2];
            int8x16_t aQ8H23[2];
            for (int vI = 0; vI < 2; vI++) {
              const int cOffset = vI * 32;
              aQ8H01[vI] = vld1q_s8(vQ8BaseH + cOffset);
              aQ8H23[vI] = vld1q_s8(vQ8BaseH + cOffset + 16);
            }

            const int cQlOffBase = vSb * QK_K / 2;

            uint8x16_t aQ6Ql0[4];
            uint8x16_t aQ6Ql1[4];
            for (int vK = 0; vK < 4; vK++) {
              aQ6Ql0[vK] = vld1q_u8(vQlBase + cQlOffBase + 16 * vK);
              aQ6Ql1[vK] = vld1q_u8(vQlBase + cQlOffBase + 64 + 16 * vK);
            }

            const int cQhOffBase = (vSb * QK_K / 2) & 255; // wrap after 256 bytes
            uint8x16_t vQ6Qh0[4];
            uint8x16_t vQ6Qh1[4];
            for (int vK = 0; vK < 4; vK++) {
              vQ6Qh0[vK] = vld1q_u8(vQhBase + cQhOffBase + 16 * vK);
              vQ6Qh1[vK] = vld1q_u8(vQhBase + cQhOffBase + 64 + 16 * vK);
            }

            // Adjust for the proper high bits (Sb 2 and 3)
            if (vSb > 1) {
              for (int vK = 0; vK < 4; vK++) {
                vQ6Qh0[vK] = vshrq_n_u8(vQ6Qh0[vK], 2);
                vQ6Qh1[vK] = vshrq_n_u8(vQ6Qh1[vK], 2);
              }
            }

            // Process column pairs (0-1, 2-3, 4-5, 6-7)
            for (int vCp = 0; vCp < cNcolsInterleaved / 2; vCp++) {
              const uint8x16_t cQ6QsCp0L = aQ6Ql0[vCp];
              const uint8x16_t cQ6QsCp1L = aQ6Ql1[vCp];
              const uint8x16_t cQ6QsCp0H = vQ6Qh0[vCp];
              const uint8x16_t cQ6QsCp1H = vQ6Qh1[vCp];

              // Extract high 2 bits for upper nibble reconstruction
              const uint8x16_t cQ6QsCp0Hh = vandq_u8(cQ6QsCp0H, cMaskHi);
              const uint8x16_t cQ6QsCp1Hh = vandq_u8(cQ6QsCp1H, cMaskHi);

              // q6 = (low4 | high2<<4) - 32
              // Use vsliq_n_u8 to combine shift-left-insert in one instruction (like Q5_K)
              const int8x16_t cQ6L0 = vsubq_s8(
                  vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ6QsCp0L, cM4b), vandq_u8(cQ6QsCp0H, cMaskLo), 4)), cM32s);
              const int8x16_t cQ6L1 = vsubq_s8(
                  vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(cQ6QsCp1L, cM4b), vandq_u8(cQ6QsCp1H, cMaskLo), 4)), cM32s);
              const int8x16_t cQ6H0 =
                  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ6QsCp0L, 4), cQ6QsCp0Hh)), cM32s);
              const int8x16_t cQ6H1 =
                  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(cQ6QsCp1L, 4), cQ6QsCp1Hh)), cM32s);

              // row pair 0, base_l
              int32x4_t vSbAcc0l = vmmlaq_s32(vdupq_n_s32(0), cQ6L0, aQ8L01[0]);
              vSbAcc0l = vmmlaq_s32(vSbAcc0l, cQ6L1, aQ8L01[1]);
              // row pair 0, base_h
              int32x4_t vSbAcc0h = vmmlaq_s32(vdupq_n_s32(0), cQ6H0, aQ8H01[0]);
              vSbAcc0h = vmmlaq_s32(vSbAcc0h, cQ6H1, aQ8H01[1]);
              // row pair 1, base_l
              int32x4_t vSbAcc1l = vmmlaq_s32(vdupq_n_s32(0), cQ6L0, aQ8L23[0]);
              vSbAcc1l = vmmlaq_s32(vSbAcc1l, cQ6L1, aQ8L23[1]);
              // row pair 1, base_h
              int32x4_t vSbAcc1h = vmmlaq_s32(vdupq_n_s32(0), cQ6H0, aQ8H23[0]);
              vSbAcc1h = vmmlaq_s32(vSbAcc1h, cQ6H1, aQ8H23[1]);

              const int cScaleIdxL = vHalf * 8 + vSb;
              const int cScaleIdxH = vHalf * 8 + vSb + 4;

              const int32x4_t cScaleVecL = {
                  vQ6Scales[cScaleIdxL * 8 + vCp * 2 + 0],
                  vQ6Scales[cScaleIdxL * 8 + vCp * 2 + 0],
                  vQ6Scales[cScaleIdxL * 8 + vCp * 2 + 1],
                  vQ6Scales[cScaleIdxL * 8 + vCp * 2 + 1],
              };
              const int32x4_t cScaleVecH = {
                  vQ6Scales[cScaleIdxH * 8 + vCp * 2 + 0],
                  vQ6Scales[cScaleIdxH * 8 + vCp * 2 + 0],
                  vQ6Scales[cScaleIdxH * 8 + vCp * 2 + 1],
                  vQ6Scales[cScaleIdxH * 8 + vCp * 2 + 1],
              };

              vAcc[vCp] = vmlaq_s32(vAcc[vCp], vSbAcc0l, cScaleVecL);
              vAcc[vCp] = vmlaq_s32(vAcc[vCp], vSbAcc0h, cScaleVecH);
              vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], vSbAcc1l, cScaleVecL);
              vAcc[vCp + 4] = vmlaq_s32(vAcc[vCp + 4], vSbAcc1h, cScaleVecH);
            }
          }
        } // for half

        // Reorder i8mm output to match memory layout
        for (int vI = 0; vI < 8; vI++) {
          int32x2x2_t vAux = vzip_s32(vget_low_s32(vAcc[vI]), vget_high_s32(vAcc[vI]));
          vAcc[vI] = vcombine_s32(vAux.val[0], vAux.val[1]);
        }
        int32x4_t aReorderAcc[8] = {
            vcombine_s32(vget_low_s32(vAcc[0]), vget_low_s32(vAcc[1])),
            vcombine_s32(vget_low_s32(vAcc[2]), vget_low_s32(vAcc[3])),
            vcombine_s32(vget_high_s32(vAcc[0]), vget_high_s32(vAcc[1])),
            vcombine_s32(vget_high_s32(vAcc[2]), vget_high_s32(vAcc[3])),
            vcombine_s32(vget_low_s32(vAcc[4]), vget_low_s32(vAcc[5])),
            vcombine_s32(vget_low_s32(vAcc[6]), vget_low_s32(vAcc[7])),
            vcombine_s32(vget_high_s32(vAcc[4]), vget_high_s32(vAcc[5])),
            vcombine_s32(vget_high_s32(vAcc[6]), vget_high_s32(vAcc[7])),
        };

        // Apply superblock scale (no mins for q6_K)
        for (int vI = 0; vI < cQ8KBlocklen; vI++) {
          for (int vJ = 0; vJ < 2; vJ++) {
            float32x4_t vQ8D = vdupq_n_f32(vQ8Ptr[vB].d[vI]);
            float32x4_t vQ6D = vcvt_f32_f16(vld1_f16((const __fp16 *)(vQ6Ptr[vB].d + vJ * 4)));
            const float32x4_t cScale = vmulq_f32(vQ6D, vQ8D);

            vAccF32[2 * vI + vJ] = vmlaq_f32(vAccF32[2 * vI + vJ], vcvtq_f32_s32(aReorderAcc[2 * vI + vJ]), cScale);
          }
        }
      } // for b

      // Store results
      for (int vI = 0; vI < cQ8KBlocklen; vI++) {
        int vRow = vY * cQ8KBlocklen + vI;
        for (int vJ = 0; vJ < 2; vJ++) {
          int vCol = vX * cNcolsInterleaved + vJ * 4;
          int vOffset = vRow * pBs + vCol;
          vst1q_f32(pS + vOffset, vAccF32[2 * vI + vJ]);
        }
      }
    } // for x
  } // for y
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  ggml_gemm_q6_K_8x8_q8_K_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q8_0_4x4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 4;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q8_0x4 *vBPtr = (const block_q8_0x4 *)pVx + (vX * cNb);

      float32x4_t vSumf[4];
      for (int vM = 0; vM < 4; vM++) {
        vSumf[vM] = vdupq_n_f32(0);
      }

      for (int vL = 0; vL < cNb; vL++) {
        float32x4_t vAD = vcvt_f32_f16(vld1_f16((const float16_t *)vAPtr[vL].d));
        float32x4_t vBD = vcvt_f32_f16(vld1_f16((const float16_t *)vBPtr[vL].d));

        int32x4_t vSumi0 = vdupq_n_s32(0);
        int32x4_t vSumi1 = vdupq_n_s32(0);
        int32x4_t vSumi2 = vdupq_n_s32(0);
        int32x4_t vSumi3 = vdupq_n_s32(0);

        for (int vKGroup = 0; vKGroup < 8; vKGroup += 4) {
          int8x16x4_t vA = vld1q_s8_x4(vAPtr[vL].qs + 16 * vKGroup);
          int8x16x4_t vB = vld1q_s8_x4(vBPtr[vL].qs + 16 * vKGroup);

          for (int vK = 0; vK < 4; vK++) {
            vSumi0 = vdotq_laneq_s32(vSumi0, vB.val[vK], vA.val[vK], 0);
            vSumi1 = vdotq_laneq_s32(vSumi1, vB.val[vK], vA.val[vK], 1);
            vSumi2 = vdotq_laneq_s32(vSumi2, vB.val[vK], vA.val[vK], 2);
            vSumi3 = vdotq_laneq_s32(vSumi3, vB.val[vK], vA.val[vK], 3);
          }
        }

        vSumf[0] = vmlaq_f32(vSumf[0], vmulq_laneq_f32(vBD, vAD, 0), vcvtq_f32_s32(vSumi0));
        vSumf[1] = vmlaq_f32(vSumf[1], vmulq_laneq_f32(vBD, vAD, 1), vcvtq_f32_s32(vSumi1));
        vSumf[2] = vmlaq_f32(vSumf[2], vmulq_laneq_f32(vBD, vAD, 2), vcvtq_f32_s32(vSumi2));
        vSumf[3] = vmlaq_f32(vSumf[3], vmulq_laneq_f32(vBD, vAD, 3), vcvtq_f32_s32(vSumi3));
      }

      for (int vM = 0; vM < 4; vM++) {
        vst1q_f32(pS + (vY * 4 + vM) * pBs + vX * 4, vSumf[vM]);
      }
    }
  }
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  ggml_gemm_q8_0_4x4_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

void ggml_gemm_q8_0_4x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 4;
  const int cBlocklen = 8;

  assert(pN % cQk == 0);
  assert(pNr % 4 == 0);
  assert(pNc % cNcolsInterleaved == 0);

  UNUSED(cNb);
  UNUSED(cNcolsInterleaved);
  UNUSED(cBlocklen);

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (svcntb() * 8 == 256) {
    const block_q8_0x4 *vBPtrBase = (const block_q8_0x4 *)pVx;

    static const uint32_t aIdxArr[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    svuint32_t vIdx = svld1(svptrue_b32(), aIdxArr);
    static const uint32_t aIdxArr1[8] = {0, 1, 2, 3, 1, 2, 3, 0};
    svuint32_t vIdxSc1 = svld1(svptrue_b32(), aIdxArr1);
    static const uint32_t aIdxArr2[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    svuint32_t vIdxSc2 = svld1(svptrue_b32(), aIdxArr2);

    for (int vY = 0; vY < pNr; vY += 4) {
      const block_q8_0x4 *vAPtrBase = (const block_q8_0x4 *)pVy + (vY / 4) * cNb;

      for (int vX = 0; vX < pNc; vX += cNcolsInterleaved) {
        const block_q8_0x4 *vBPtr = vBPtrBase + (vX / 4) * cNb;
        const block_q8_0x4 *vAPtr = vAPtrBase;

        svfloat32_t vAccF3201 = svdup_f32(0);
        svfloat32_t vAccF3223 = svdup_f32(0);

        for (int vB = 0; vB < cNb; vB++) {

          svint32_t vAcc01 = svdup_s32(0);
          svint32_t vAcc23 = svdup_s32(0);

          // Process 4 chunks of 8 positions each
          for (int vChunk = 0; vChunk < 4; vChunk++) {
            svint8_t vSA01 = svld1rq_s8(svptrue_b8(), vAPtr->qs + vChunk * 32);
            svint8_t vSA23 = svld1rq_s8(svptrue_b8(), vAPtr->qs + vChunk * 32 + 16);
            svint8_t vSB0123 = svld1_s8(svptrue_b8(), vBPtr->qs + vChunk * 32);

            vAcc01 = svmmla_s32(vAcc01, vSA01, vSB0123);
            vAcc23 = svmmla_s32(vAcc23, vSA23, vSB0123);
          }

          // Reorder outputs from 2×2 tiles to row-major
          // acc[01] = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
          // acc[23] = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]

          svint32_t vRow01 = svtbl_s32(vAcc01, vIdx);
          svint32_t vRow23 = svtbl_s32(vAcc23, vIdx);

          svfloat16_t vTemp1 = svld1_f16(svptrue_pat_b16(SV_VL4), (const __fp16 *)vAPtr->d);
          svfloat16_t vTemp2 = svld1_f16(svptrue_pat_b16(SV_VL4), (const __fp16 *)vBPtr->d);
          svfloat32_t vSvAD = svtbl_f32(svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(vTemp1, vTemp1)), vIdxSc1);
          svfloat32_t vSvBD = svtbl_f32(svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(vTemp2, vTemp2)), vIdxSc2);

          vAccF3201 = svmla_f32_x(svptrue_b32(), vAccF3201, svcvt_f32_s32_x(svptrue_b32(), vRow01),
                                  svmul_lane_f32(vSvBD, vSvAD, 0));
          vAccF3223 = svmla_f32_x(svptrue_b32(), vAccF3223, svcvt_f32_s32_x(svptrue_b32(), vRow23),
                                  svmul_lane_f32(vSvBD, vSvAD, 2));
          vAPtr++;
          vBPtr++;
        }

        svbool_t vPg4 = svptrue_pat_b32(SV_VL4);
        svst1_f32(vPg4, pS + (vY + 0) * pBs + vX, vAccF3201);
        svst1_f32(vPg4, pS + (vY + 1) * pBs + vX, svext_f32(vAccF3201, vAccF3201, 4));
        svst1_f32(vPg4, pS + (vY + 2) * pBs + vX, vAccF3223);
        svst1_f32(vPg4, pS + (vY + 3) * pBs + vX, svext_f32(vAccF3223, vAccF3223, 4));
      }
    }
    return;
  }
#endif // SVE compile-time end

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const block_q8_0x4 *vBPtrBase = (const block_q8_0x4 *)pVx;

  for (int vY = 0; vY < pNr; vY += 4) {
    const block_q8_0x4 *vAPtrBase = (const block_q8_0x4 *)pVy + (vY / 4) * cNb;

    for (int vX = 0; vX < pNc; vX += cNcolsInterleaved) {
      const block_q8_0x4 *vBPtr = vBPtrBase + (vX / 4) * cNb;
      const block_q8_0x4 *vAPtr = vAPtrBase;

      float32x4_t vAccF32[4];
      for (int vI = 0; vI < 4; vI++) {
        vAccF32[vI] = vdupq_n_f32(0);
      }

      for (int vB = 0; vB < cNb; vB++) {
        int32x4_t vAcc[4];
        for (int vI = 0; vI < 4; vI++) {
          vAcc[vI] = vdupq_n_s32(0);
        }

        // Process 4 chunks of 8 positions each
        for (int vChunk = 0; vChunk < 4; vChunk++) {
          int8x16_t vA01 = vld1q_s8(vAPtr->qs + vChunk * 32);
          int8x16_t vA23 = vld1q_s8(vAPtr->qs + vChunk * 32 + 16);
          int8x16_t vB01 = vld1q_s8(vBPtr->qs + vChunk * 32);
          int8x16_t vB23 = vld1q_s8(vBPtr->qs + vChunk * 32 + 16);

          vAcc[0] = vmmlaq_s32(vAcc[0], vA01, vB01);
          vAcc[1] = vmmlaq_s32(vAcc[1], vA01, vB23);
          vAcc[2] = vmmlaq_s32(vAcc[2], vA23, vB01);
          vAcc[3] = vmmlaq_s32(vAcc[3], vA23, vB23);
        }

        // Reorder outputs from 2×2 tiles to row-major
        // acc[0] = [r0c0, r0c1, r1c0, r1c1]
        // acc[1] = [r0c2, r0c3, r1c2, r1c3]
        // acc[2] = [r2c0, r2c1, r3c0, r3c1]
        // acc[3] = [r2c2, r2c3, r3c2, r3c3]
        int32x4_t vRow0 = vcombine_s32(vget_low_s32(vAcc[0]), vget_low_s32(vAcc[1]));
        int32x4_t vRow1 = vcombine_s32(vget_high_s32(vAcc[0]), vget_high_s32(vAcc[1]));
        int32x4_t vRow2 = vcombine_s32(vget_low_s32(vAcc[2]), vget_low_s32(vAcc[3]));
        int32x4_t vRow3 = vcombine_s32(vget_high_s32(vAcc[2]), vget_high_s32(vAcc[3]));

        // Scales
        float32x4_t vAD = vcvt_f32_f16(vld1_f16((const __fp16 *)vAPtr->d));
        float32x4_t vBD = vcvt_f32_f16(vld1_f16((const __fp16 *)vBPtr->d));

        vAccF32[0] = vfmaq_f32(vAccF32[0], vcvtq_f32_s32(vRow0), vmulq_laneq_f32(vBD, vAD, 0));
        vAccF32[1] = vfmaq_f32(vAccF32[1], vcvtq_f32_s32(vRow1), vmulq_laneq_f32(vBD, vAD, 1));
        vAccF32[2] = vfmaq_f32(vAccF32[2], vcvtq_f32_s32(vRow2), vmulq_laneq_f32(vBD, vAD, 2));
        vAccF32[3] = vfmaq_f32(vAccF32[3], vcvtq_f32_s32(vRow3), vmulq_laneq_f32(vBD, vAD, 3));

        vAPtr++;
        vBPtr++;
      }

      for (int vRow = 0; vRow < 4; vRow++) {
        vst1q_f32(pS + (vY + vRow) * pBs + vX, vAccF32[vRow]);
      }
    }
  }
  return;
#endif // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  ggml_gemm_q8_0_4x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}
