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

#if defined(__wasm_simd128__)
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

#if defined __wasm_simd128__
  for (int vI = 0; vI < cNb; vI++) {
    v128_t aSrcv[8];
    v128_t aAsrcv[8];
    v128_t vAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = wasm_v128_load(pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = wasm_f32x4_abs(aSrcv[vJ]);

    for (int vJ = 0; vJ < 4; vJ++)
      vAmaxv[2 * vJ] = wasm_f32x4_max(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      vAmaxv[4 * vJ] = wasm_f32x4_max(vAmaxv[4 * vJ], vAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      vAmaxv[8 * vJ] = wasm_f32x4_max(vAmaxv[8 * vJ], vAmaxv[8 * vJ + 4]);

    const float cAmax = MAX(MAX(wasm_f32x4_extract_lane(vAmaxv[0], 0), wasm_f32x4_extract_lane(vAmaxv[0], 1)),
                            MAX(wasm_f32x4_extract_lane(vAmaxv[0], 2), wasm_f32x4_extract_lane(vAmaxv[0], 3)));

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < 8; vJ++) {
      const v128_t cV = wasm_f32x4_mul(aSrcv[vJ], wasm_f32x4_splat(cId));
      const v128_t cVi = wasm_i32x4_trunc_sat_f32x4(cV);

      vY[vI].qs[4 * vJ + 0] = wasm_i32x4_extract_lane(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = wasm_i32x4_extract_lane(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = wasm_i32x4_extract_lane(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = wasm_i32x4_extract_lane(cVi, 3);
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
#if defined __wasm_simd128__
  for (int vI = 0; vI < cNb; vI++) {
    v128_t aSrcv[8];
    v128_t aAsrcv[8];
    v128_t vAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = wasm_v128_load(pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = wasm_f32x4_abs(aSrcv[vJ]);

    for (int vJ = 0; vJ < 4; vJ++)
      vAmaxv[2 * vJ] = wasm_f32x4_max(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      vAmaxv[4 * vJ] = wasm_f32x4_max(vAmaxv[4 * vJ], vAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      vAmaxv[8 * vJ] = wasm_f32x4_max(vAmaxv[8 * vJ], vAmaxv[8 * vJ + 4]);

    const float cAmax = MAX(MAX(wasm_f32x4_extract_lane(vAmaxv[0], 0), wasm_f32x4_extract_lane(vAmaxv[0], 1)),
                            MAX(wasm_f32x4_extract_lane(vAmaxv[0], 2), wasm_f32x4_extract_lane(vAmaxv[0], 3)));

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    v128_t vAccv = wasm_i32x4_splat(0);

    for (int vJ = 0; vJ < 8; vJ++) {
      const v128_t cV = wasm_f32x4_mul(aSrcv[vJ], wasm_f32x4_splat(cId));
      const v128_t cVi = wasm_i32x4_trunc_sat_f32x4(cV);

      vY[vI].qs[4 * vJ + 0] = wasm_i32x4_extract_lane(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = wasm_i32x4_extract_lane(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = wasm_i32x4_extract_lane(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = wasm_i32x4_extract_lane(cVi, 3);

      vAccv = wasm_i32x4_add(vAccv, cVi);
    }

    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * (wasm_i32x4_extract_lane(vAccv, 0) + wasm_i32x4_extract_lane(vAccv, 1) +
                                           wasm_i32x4_extract_lane(vAccv, 2) + wasm_i32x4_extract_lane(vAccv, 3)));
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_1_ref(pX, vY, pK);
#endif
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
#ifdef __wasm_simd128__
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;
  block_q8_K *GGML_RESTRICT vYc = pY; // Cast to proper type

  for (int vI = 0; vI < cNb; vI++) {
    const float *vXBlock = pX + vI * QK_K;

    v128_t vMinVec = wasm_v128_load(vXBlock);
    v128_t vMaxVec = vMinVec;

    for (int vJ = 4; vJ < QK_K; vJ += 4) {
      v128_t vXVec = wasm_v128_load(vXBlock + vJ);
      vMaxVec = wasm_f32x4_pmax(vMaxVec, vXVec);
      vMinVec = wasm_f32x4_pmin(vMinVec, vXVec);
    }
    vMaxVec = wasm_f32x4_pmax(vMaxVec, wasm_i32x4_shuffle(vMaxVec, vMaxVec, 2, 3, 0, 1));
    vMaxVec = wasm_f32x4_pmax(vMaxVec, wasm_i32x4_shuffle(vMaxVec, vMaxVec, 1, 0, 3, 2));
    vMinVec = wasm_f32x4_pmin(vMinVec, wasm_i32x4_shuffle(vMinVec, vMinVec, 2, 3, 0, 1));
    vMinVec = wasm_f32x4_pmin(vMinVec, wasm_i32x4_shuffle(vMinVec, vMinVec, 1, 0, 3, 2));
    float vMax = wasm_f32x4_extract_lane(vMaxVec, 0);
    float vMin = wasm_f32x4_extract_lane(vMinVec, 0);
    float vAmax = -vMin > vMax ? vMin : vMax;

    if (vAmax == 0.0f) {
      vYc[vI].d = 0.0f;
      const v128_t cZero = wasm_i8x16_splat(0);
      for (int vJ = 0; vJ < QK_K; vJ += 16) {
        wasm_v128_store(vYc[vI].qs + vJ, cZero);
      }
      continue;
    }

    const float cIscale = -127.0f / vAmax;
    const v128_t cScaleVec = wasm_f32x4_splat(cIscale);

    // Process 16 elements per iteration
    for (int vJ = 0, vJb = 0; vJ < QK_K; vJ += 16, vJb++) {
      // Load and quantize 16 floats
      v128_t vX0 = wasm_v128_load(vXBlock + vJ);
      v128_t vX1 = wasm_v128_load(vXBlock + vJ + 4);
      v128_t vX2 = wasm_v128_load(vXBlock + vJ + 8);
      v128_t vX3 = wasm_v128_load(vXBlock + vJ + 12);

      v128_t vQ0 = wasm_f32x4_nearest(wasm_f32x4_mul(vX0, cScaleVec));
      v128_t vQ1 = wasm_f32x4_nearest(wasm_f32x4_mul(vX1, cScaleVec));
      v128_t vQ2 = wasm_f32x4_nearest(wasm_f32x4_mul(vX2, cScaleVec));
      v128_t vQ3 = wasm_f32x4_nearest(wasm_f32x4_mul(vX3, cScaleVec));

      // Convert to i32 with saturation
      v128_t vI0 = wasm_i32x4_trunc_sat_f32x4(vQ0);
      v128_t vI1 = wasm_i32x4_trunc_sat_f32x4(vQ1);
      v128_t vI2 = wasm_i32x4_trunc_sat_f32x4(vQ2);
      v128_t vI3 = wasm_i32x4_trunc_sat_f32x4(vQ3);

      // Pack into 16 i8 values
      v128_t vI8 = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(vI0, vI1), wasm_i16x8_narrow_i32x4(vI2, vI3));
      wasm_v128_store(vYc[vI].qs + vJ, vI8);

      // Calculate bsums using SIMD
      v128_t vSum16 = wasm_i16x8_add(wasm_i16x8_extend_low_i8x16(vI8), wasm_i16x8_extend_high_i8x16(vI8));
      v128_t vSum32 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vSum16), wasm_i32x4_extend_high_i16x8(vSum16));
      vSum32 = wasm_i32x4_add(vSum32, wasm_i32x4_shuffle(vSum32, vSum32, 2, 3, 0, 1));
      vSum32 = wasm_i32x4_add(vSum32, wasm_i32x4_shuffle(vSum32, vSum32, 1, 0, 3, 2));
      vYc[vI].bsums[vJb] = wasm_i32x4_extract_lane(vSum32, 0);
    }

    vYc[vI].d = 1.0f / cIscale;
  }
#else
  quantize_row_q8_K_ref(pX, pY, pK);
#endif
}

//===================================== Dot products =================================

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

#if defined __wasm_simd128__
  v128_t vSumv = wasm_f32x4_splat(0.0f);

  const v128_t cM4b = wasm_i8x16_splat(0x0F);
  const v128_t cS8b = wasm_i8x16_splat(0x8);

  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q4_0 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q4_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    // Load and process x0
    v128_t vV00 = wasm_v128_load(vX0->qs);
    v128_t vV00l = wasm_v128_and(vV00, cM4b);
    v128_t vV00h = wasm_u8x16_shr(vV00, 4);
    v128_t vV00ls = wasm_i8x16_sub(vV00l, cS8b);
    v128_t vV00hs = wasm_i8x16_sub(vV00h, cS8b);

    // Load y0 vectors
    v128_t vY0L = wasm_v128_load(vY0->qs);
    v128_t vY0H = wasm_v128_load(vY0->qs + 16);

    // Extend to i16x8 and compute dot products
    v128_t vDx0l = wasm_i16x8_extend_low_i8x16(vV00ls);
    v128_t vDx0h = wasm_i16x8_extend_high_i8x16(vV00ls);
    v128_t vDx0hl = wasm_i16x8_extend_low_i8x16(vV00hs);
    v128_t vDx0hh = wasm_i16x8_extend_high_i8x16(vV00hs);

    v128_t vDy0ll = wasm_i16x8_extend_low_i8x16(vY0L);
    v128_t vDy0lh = wasm_i16x8_extend_high_i8x16(vY0L);
    v128_t vDy0hl = wasm_i16x8_extend_low_i8x16(vY0H);
    v128_t vDy0hh = wasm_i16x8_extend_high_i8x16(vY0H);

    v128_t vDp0 =
        wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_dot_i16x8(vDx0l, vDy0ll), wasm_i32x4_dot_i16x8(vDx0h, vDy0lh)),
                       wasm_i32x4_add(wasm_i32x4_dot_i16x8(vDx0hl, vDy0hl), wasm_i32x4_dot_i16x8(vDx0hh, vDy0hh)));

    // Load and process x1
    v128_t vV01 = wasm_v128_load(vX1->qs);
    v128_t vV01l = wasm_v128_and(vV01, cM4b);
    v128_t vV01h = wasm_u8x16_shr(vV01, 4);
    v128_t vV01ls = wasm_i8x16_sub(vV01l, cS8b);
    v128_t vV01hs = wasm_i8x16_sub(vV01h, cS8b);

    // Load y1 vectors
    v128_t vY1L = wasm_v128_load(vY1->qs);
    v128_t vY1H = wasm_v128_load(vY1->qs + 16);

    // Extend to i16x8 and compute dot products
    v128_t vDx1l = wasm_i16x8_extend_low_i8x16(vV01ls);
    v128_t vDx1h = wasm_i16x8_extend_high_i8x16(vV01ls);
    v128_t vDx1hl = wasm_i16x8_extend_low_i8x16(vV01hs);
    v128_t vDx1hh = wasm_i16x8_extend_high_i8x16(vV01hs);

    v128_t vDy1ll = wasm_i16x8_extend_low_i8x16(vY1L);
    v128_t vDy1lh = wasm_i16x8_extend_high_i8x16(vY1L);
    v128_t vDy1hl = wasm_i16x8_extend_low_i8x16(vY1H);
    v128_t vDy1hh = wasm_i16x8_extend_high_i8x16(vY1H);

    v128_t vDp1 =
        wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_dot_i16x8(vDx1l, vDy1ll), wasm_i32x4_dot_i16x8(vDx1h, vDy1lh)),
                       wasm_i32x4_add(wasm_i32x4_dot_i16x8(vDx1hl, vDy1hl), wasm_i32x4_dot_i16x8(vDx1hh, vDy1hh)));

    // Accumulate results with scaling
    float vScale0 = GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d);
    float vScale1 = GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d);

    vSumv = wasm_f32x4_add(vSumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vDp0), wasm_f32x4_splat(vScale0)));
    vSumv = wasm_f32x4_add(vSumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vDp1), wasm_f32x4_splat(vScale1)));
  }

  vSumf = wasm_f32x4_extract_lane(vSumv, 0) + wasm_f32x4_extract_lane(vSumv, 1) + wasm_f32x4_extract_lane(vSumv, 2) +
          wasm_f32x4_extract_lane(vSumv, 3);

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

  float vSumf = 0;

#if defined __wasm_simd128__
  v128_t vSumv = wasm_f32x4_splat(0.0f);
  float vSumms = 0.0f;

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const block_q4_1 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb];

    vSumms += GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s);

    const v128_t cRaw = wasm_v128_load(vX0->qs);
    const v128_t cV0s = wasm_v128_and(cRaw, wasm_i8x16_splat(0x0F));
    const v128_t cV1s = wasm_u8x16_shr(cRaw, 4);

    const v128_t cYsLo = wasm_v128_load(vY0->qs);
    const v128_t cYsHi = wasm_v128_load(vY0->qs + 16);

    const v128_t cV0sL = wasm_u16x8_extend_low_u8x16(cV0s);
    const v128_t cV0sH = wasm_u16x8_extend_high_u8x16(cV0s);
    const v128_t cYloL = wasm_i16x8_extend_low_i8x16(cYsLo);
    const v128_t cYloH = wasm_i16x8_extend_high_i8x16(cYsLo);
    const v128_t cV1sL = wasm_u16x8_extend_low_u8x16(cV1s);
    const v128_t cV1sH = wasm_u16x8_extend_high_u8x16(cV1s);
    const v128_t cYhiL = wasm_i16x8_extend_low_i8x16(cYsHi);
    const v128_t cYhiH = wasm_i16x8_extend_high_i8x16(cYsHi);

    const v128_t cAcc =
        wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV0sL, cYloL), wasm_i32x4_dot_i16x8(cV0sH, cYloH)),
                       wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV1sL, cYhiL), wasm_i32x4_dot_i16x8(cV1sH, cYhiH)));

    vSumv = wasm_f32x4_add(
        vSumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(cAcc),
                              wasm_f32x4_splat(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d))));
  }

  vSumf = wasm_f32x4_extract_lane(vSumv, 0) + wasm_f32x4_extract_lane(vSumv, 1) + wasm_f32x4_extract_lane(vSumv, 2) +
          wasm_f32x4_extract_lane(vSumv, 3) + vSumms;

  *pS = vSumf;

#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vSumf);

  ggml_vec_dot_q4_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
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

#if defined __wasm_simd128__
  v128_t vSumv = wasm_f32x4_splat(0.0f);

  uint32_t vQh;
  uint64_t aTmp[4];

  // TODO: check if unrolling this is better
  for (; vIb < cNb; ++vIb) {
    const block_q5_0 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];

    const v128_t cM4b = wasm_i8x16_splat(0x0F);

    // extract the 5th bit
    memcpy(&vQh, vX0->qh, sizeof(vQh));

    aTmp[0] = table_b2b_1[(vQh >> 0) & 0xFF];
    aTmp[1] = table_b2b_1[(vQh >> 8) & 0xFF];
    aTmp[2] = table_b2b_1[(vQh >> 16) & 0xFF];
    aTmp[3] = table_b2b_1[(vQh >> 24)];

    const v128_t cQhl = wasm_v128_load(aTmp + 0);
    const v128_t cQhh = wasm_v128_load(aTmp + 2);

    const v128_t cV0 = wasm_v128_load(vX0->qs);

    // 4-bit -> 8-bit
    const v128_t cV0l = wasm_v128_and(cV0, cM4b);
    const v128_t cV0h = wasm_u8x16_shr(cV0, 4);

    // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
    const v128_t cV0lf = wasm_i8x16_sub(cV0l, cQhl);
    const v128_t cV0hf = wasm_i8x16_sub(cV0h, cQhh);

    // load y
    const v128_t cV1l = wasm_v128_load(vY0->qs);
    const v128_t cV1h = wasm_v128_load(vY0->qs + 16);

    // int8x16 -> int16x8
    const v128_t cV0lfl = wasm_i16x8_extend_low_i8x16(cV0lf);
    const v128_t cV0lfh = wasm_i16x8_extend_high_i8x16(cV0lf);
    const v128_t cV0hfl = wasm_i16x8_extend_low_i8x16(cV0hf);
    const v128_t cV0hfh = wasm_i16x8_extend_high_i8x16(cV0hf);

    const v128_t cV1ll = wasm_i16x8_extend_low_i8x16(cV1l);
    const v128_t cV1lh = wasm_i16x8_extend_high_i8x16(cV1l);
    const v128_t cV1hl = wasm_i16x8_extend_low_i8x16(cV1h);
    const v128_t cV1hh = wasm_i16x8_extend_high_i8x16(cV1h);

    // dot product
    vSumv = wasm_f32x4_add(
        vSumv,
        wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_add(
                           wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV0lfl, cV1ll), wasm_i32x4_dot_i16x8(cV0lfh, cV1lh)),
                           wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV0hfl, cV1hl), wasm_i32x4_dot_i16x8(cV0hfh, cV1hh)))),
                       wasm_f32x4_splat(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d))));
  }

  vSumf = wasm_f32x4_extract_lane(vSumv, 0) + wasm_f32x4_extract_lane(vSumv, 1) + wasm_f32x4_extract_lane(vSumv, 2) +
          wasm_f32x4_extract_lane(vSumv, 3);

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
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

#if defined __wasm_simd128__
  v128_t vSumv = wasm_f32x4_splat(0.0f);

  float vSumms = 0.0f;

  uint32_t vQh;
  uint64_t aTmp[4];

  // TODO: check if unrolling this is better
  for (; vIb < cNb; ++vIb) {
    const block_q5_1 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb];

    vSumms += GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s);

    const v128_t cM4b = wasm_i8x16_splat(0x0F);

    // extract the 5th bit
    memcpy(&vQh, vX0->qh, sizeof(vQh));

    aTmp[0] = table_b2b_0[(vQh >> 0) & 0xFF];
    aTmp[1] = table_b2b_0[(vQh >> 8) & 0xFF];
    aTmp[2] = table_b2b_0[(vQh >> 16) & 0xFF];
    aTmp[3] = table_b2b_0[(vQh >> 24)];

    const v128_t cQhl = wasm_v128_load(aTmp + 0);
    const v128_t cQhh = wasm_v128_load(aTmp + 2);

    const v128_t cV0 = wasm_v128_load(vX0->qs);

    // 4-bit -> 8-bit
    const v128_t cV0l = wasm_v128_and(cV0, cM4b);
    const v128_t cV0h = wasm_u8x16_shr(cV0, 4);

    // add high bit
    const v128_t cV0lf = wasm_v128_or(cV0l, cQhl);
    const v128_t cV0hf = wasm_v128_or(cV0h, cQhh);

    // load y
    const v128_t cV1l = wasm_v128_load(vY0->qs);
    const v128_t cV1h = wasm_v128_load(vY0->qs + 16);

    // int8x16 -> int16x8
    const v128_t cV0lfl = wasm_i16x8_extend_low_i8x16(cV0lf);
    const v128_t cV0lfh = wasm_i16x8_extend_high_i8x16(cV0lf);
    const v128_t cV0hfl = wasm_i16x8_extend_low_i8x16(cV0hf);
    const v128_t cV0hfh = wasm_i16x8_extend_high_i8x16(cV0hf);

    const v128_t cV1ll = wasm_i16x8_extend_low_i8x16(cV1l);
    const v128_t cV1lh = wasm_i16x8_extend_high_i8x16(cV1l);
    const v128_t cV1hl = wasm_i16x8_extend_low_i8x16(cV1h);
    const v128_t cV1hh = wasm_i16x8_extend_high_i8x16(cV1h);

    // dot product
    vSumv = wasm_f32x4_add(
        vSumv,
        wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_add(
                           wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV0lfl, cV1ll), wasm_i32x4_dot_i16x8(cV0lfh, cV1lh)),
                           wasm_i32x4_add(wasm_i32x4_dot_i16x8(cV0hfl, cV1hl), wasm_i32x4_dot_i16x8(cV0hfh, cV1hh)))),
                       wasm_f32x4_splat(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d))));
  }

  vSumf = wasm_f32x4_extract_lane(vSumv, 0) + wasm_f32x4_extract_lane(vSumv, 1) + wasm_f32x4_extract_lane(vSumv, 2) +
          wasm_f32x4_extract_lane(vSumv, 3) + vSumms;

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
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

#if defined __wasm_simd128__
  v128_t vSumv = wasm_f32x4_splat(0.0f);

  for (; vIb < cNb; ++vIb) {
    const block_q8_0 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];

    const v128_t cX00 = wasm_v128_load(vX0->qs);
    const v128_t cX01 = wasm_v128_load(vX0->qs + 16);
    const v128_t cY00 = wasm_v128_load(vY0->qs);
    const v128_t cY01 = wasm_v128_load(vY0->qs + 16);

    // Extend 8-bit to 16-bit
    const v128_t cX00l = wasm_i16x8_extend_low_i8x16(cX00);
    const v128_t cX00h = wasm_i16x8_extend_high_i8x16(cX00);
    const v128_t cX01l = wasm_i16x8_extend_low_i8x16(cX01);
    const v128_t cX01h = wasm_i16x8_extend_high_i8x16(cX01);

    const v128_t cY00l = wasm_i16x8_extend_low_i8x16(cY00);
    const v128_t cY00h = wasm_i16x8_extend_high_i8x16(cY00);
    const v128_t cY01l = wasm_i16x8_extend_low_i8x16(cY01);
    const v128_t cY01h = wasm_i16x8_extend_high_i8x16(cY01);

    // Compute dot products
    const v128_t cDx00 = wasm_i32x4_dot_i16x8(cX00l, cY00l);
    const v128_t cDx01 = wasm_i32x4_dot_i16x8(cX00h, cY00h);
    const v128_t cDx10 = wasm_i32x4_dot_i16x8(cX01l, cY01l);
    const v128_t cDx11 = wasm_i32x4_dot_i16x8(cX01h, cY01h);

    // Sum all dot products
    const v128_t cSumDots = wasm_i32x4_add(wasm_i32x4_add(cDx00, cDx01), wasm_i32x4_add(cDx10, cDx11));

    // Convert to float and accumulate
    const float cScale = GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d);
    vSumv = wasm_f32x4_add(vSumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(cSumDots), wasm_f32x4_splat(cScale)));
  }

  vSumf = wasm_f32x4_extract_lane(vSumv, 0) + wasm_f32x4_extract_lane(vSumv, 1) + wasm_f32x4_extract_lane(vSumv, 2) +
          wasm_f32x4_extract_lane(vSumv, 3);

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_q8_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
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

#if defined __wasm_simd128__
  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *vQ2 = vX[vI].qs;
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vSc = vX[vI].scales;

    // Vectorized summs calculation
    v128_t vSummsVec = wasm_i32x4_splat(0);
    {
      v128_t vScVec = wasm_v128_load(vSc);
      v128_t vScUpper = wasm_u8x16_shr(vScVec, 4);

      v128_t vScLow = wasm_u16x8_extend_low_u8x16(vScUpper);
      v128_t vScHigh = wasm_u16x8_extend_high_u8x16(vScUpper);

      v128_t vBsums1 = wasm_v128_load(&vY[vI].bsums[0]);
      v128_t vBsums2 = wasm_v128_load(&vY[vI].bsums[8]);

      vSummsVec = wasm_i32x4_add(
          wasm_i32x4_add(wasm_i32x4_dot_i16x8(vScLow, vBsums1), wasm_i32x4_dot_i16x8(vScHigh, vBsums2)), vSummsVec);

      vSummsVec = wasm_i32x4_add(vSummsVec, wasm_i32x4_shuffle(vSummsVec, vSummsVec, 2, 3, 0, 1));
      vSummsVec = wasm_i32x4_add(vSummsVec, wasm_i32x4_shuffle(vSummsVec, vSummsVec, 1, 0, 3, 2));
    }
    int32_t vSumms = wasm_i32x4_extract_lane(vSummsVec, 0);

    // Vectorized isum calculation
    int32_t vIsum = 0;
    const uint8_t *vScPtr = vSc;
    const int cKIters = QK_K / 128;

    for (int vK = 0; vK < cKIters; ++vK) {
      v128_t vIsumVec = wasm_i32x4_splat(0);
      int vShift = 0;

      for (int vJ = 0; vJ < 4; ++vJ) {
        const int cD0 = (vScPtr[0] & 0xF);
        const int cD1 = (vScPtr[1] & 0xF);
        vScPtr += 2;

        // Process first 16 elements
        v128_t vQ20 = wasm_v128_load(vQ2);
        v128_t vQ80 = wasm_v128_load(vQ8);
        v128_t vQ2Shift0 = wasm_u8x16_shr(vQ20, vShift);
        v128_t vQ2Bits0 = wasm_v128_and(vQ2Shift0, wasm_i8x16_splat(0x03));

        // Process next 16 elements
        v128_t vQ21 = wasm_v128_load(vQ2 + 16);
        v128_t vQ81 = wasm_v128_load(vQ8 + 16);
        v128_t vQ2Shift1 = wasm_u8x16_shr(vQ21, vShift);
        v128_t vQ2Bits1 = wasm_v128_and(vQ2Shift1, wasm_i8x16_splat(0x03));

        // Calculate dot products
        v128_t vP0 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ80), wasm_i16x8_extend_low_i8x16(vQ2Bits0));
        v128_t vP1 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ80), wasm_i16x8_extend_high_i8x16(vQ2Bits0));
        v128_t vP2 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ81), wasm_i16x8_extend_low_i8x16(vQ2Bits1));
        v128_t vP3 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ81), wasm_i16x8_extend_high_i8x16(vQ2Bits1));

        // Accumulate scaled results
        v128_t vScaled = wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(vP0, vP1), wasm_i32x4_splat(cD0)),
                                        wasm_i32x4_mul(wasm_i32x4_add(vP2, vP3), wasm_i32x4_splat(cD1)));

        vIsumVec = wasm_i32x4_add(vIsumVec, vScaled);
        vQ8 += 32;
        vShift += 2;
      }
      vQ2 += 32;

      // Horizontal sum of isum_vec
      vIsumVec = wasm_i32x4_add(vIsumVec, wasm_i32x4_shuffle(vIsumVec, vIsumVec, 2, 3, 0, 1));
      vIsumVec = wasm_i32x4_add(vIsumVec, wasm_i32x4_shuffle(vIsumVec, vIsumVec, 1, 0, 3, 2));
      vIsum += wasm_i32x4_extract_lane(vIsumVec, 0);
    }

    const float cDall = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const float cDmin = GGML_CPU_FP16_TO_FP32(vX[vI].dmin) * vY[vI].d;
    vSumf += cDall * vIsum - cDmin * vSumms;
  }

  *pS = vSumf;

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

#if defined __wasm_simd128__
  int8_t aAux8[QK_K];
  float aSums[8] = {0};
  uint32_t aAuxs[4];

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vHm = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // Process blocks with SIMD
    int8_t *vA = aAux8;
    uint8_t vM = 1;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int vShift = 0; vShift <= 6; vShift += 2) {
        v128_t vVM = wasm_i8x16_splat(vM);
        for (int vL = 0; vL < 32; vL += 16) {
          v128_t vVQ3 = wasm_v128_load(vQ3 + vL);
          v128_t vVShift = wasm_i8x16_shr(vVQ3, vShift);
          v128_t vVLow2 = wasm_v128_and(vVShift, wasm_i8x16_splat(0x03));

          v128_t vVHm = wasm_v128_load(vHm + vL);
          v128_t vVMask = wasm_v128_and(vVHm, vVM);
          vVMask = wasm_i8x16_ne(vVMask, wasm_i8x16_splat(0));

          vVLow2 = wasm_i8x16_sub(vVLow2, wasm_v128_and(wasm_i8x16_splat(4), wasm_v128_not(vVMask)));
          wasm_v128_store(vA + vL, vVLow2);
        }
        vA += 32;
        vM <<= 1;
      }
      vQ3 += 32;
    }

    // Extract scales
    memcpy(aAuxs, vX[vI].scales, 12);
    uint32_t vTmp = aAuxs[2];
    aAuxs[2] = ((aAuxs[0] >> 4) & cKmask2) | (((vTmp >> 4) & cKmask1) << 4);
    aAuxs[3] = ((aAuxs[1] >> 4) & cKmask2) | (((vTmp >> 6) & cKmask1) << 4);
    aAuxs[0] = (aAuxs[0] & cKmask2) | (((vTmp >> 0) & cKmask1) << 4);
    aAuxs[1] = (aAuxs[1] & cKmask2) | (((vTmp >> 2) & cKmask1) << 4);
    const int8_t *vScales = (const int8_t *)aAuxs;

    // SIMD dot product with register accumulators
    v128_t vVAcc0 = wasm_i32x4_splat(0);
    v128_t vVAcc1 = wasm_i32x4_splat(0);
    vA = aAux8;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      const v128_t cVScale = wasm_i16x8_splat(vScales[vJ] - 32);

      // Process 16 elements per iteration
      for (int vK = 0; vK < 2; ++vK) {
        const v128_t cVQ8 = wasm_i16x8_load8x8(vQ8);
        const v128_t cVA = wasm_i16x8_load8x8(vA);

        v128_t vVProd = wasm_i16x8_mul(cVQ8, cVA);
        vVProd = wasm_i16x8_mul(vVProd, cVScale);

        vVAcc0 = wasm_i32x4_add(vVAcc0, wasm_i32x4_extend_low_i16x8(vVProd));
        vVAcc1 = wasm_i32x4_add(vVAcc1, wasm_i32x4_extend_high_i16x8(vVProd));

        vQ8 += 8;
        vA += 8;
      }
    }

    // Accumulate results
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const v128_t cVD = wasm_f32x4_splat(cD);
    v128_t vVSum = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vVAcc0), cVD),
                                  wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vVAcc1), cVD));

    // Accumulate into sums vector
    wasm_v128_store(aSums, wasm_f32x4_add(wasm_v128_load(aSums), vVSum));
  }

  // Horizontal sum
  v128_t vVSum = wasm_f32x4_add(wasm_v128_load(aSums), wasm_v128_load(aSums + 4));
  vSumf = wasm_f32x4_extract_lane(vVSum, 0) + wasm_f32x4_extract_lane(vVSum, 1) + wasm_f32x4_extract_lane(vVSum, 2) +
          wasm_f32x4_extract_lane(vVSum, 3);

  *pS = vSumf;

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

#if defined __wasm_simd128__
  const uint8_t *vScales = (const uint8_t *)&vUtmp[0];
  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin); // Corrected sign

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // Process scales and mins
    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    // Sum mins * q8sums
    int32_t vSumi = 0;
    const int16_t *GGML_RESTRICT vQ8sums = vY[vI].bsums;
    const uint8_t *vM = (const uint8_t *)&vUtmp[2];
    for (int vJ = 0; vJ < 16; vJ += 2) {
      vSumi += (vQ8sums[vJ] + vQ8sums[vJ + 1]) * vM[vJ / 2];
    }
    vSumf -= cDmin * vSumi;

    int32_t vSumi1 = 0;
    int32_t vSumi2 = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      // Load 64 4-bit weights (32 bytes)
      const v128_t cQ4x0 = wasm_v128_load(vQ4);
      const v128_t cQ4x1 = wasm_v128_load(vQ4 + 16);
      vQ4 += 32;

      // Split into low/high nibbles
      const v128_t cQ4l0 = wasm_v128_and(cQ4x0, wasm_i8x16_splat(0x0F));
      const v128_t cQ4h0 = wasm_u8x16_shr(cQ4x0, 4);
      const v128_t cQ4l1 = wasm_v128_and(cQ4x1, wasm_i8x16_splat(0x0F));
      const v128_t cQ4h1 = wasm_u8x16_shr(cQ4x1, 4);

      // Load 64 8-bit values (64 bytes)
      const v128_t cQ8x0 = wasm_v128_load(vQ8);
      const v128_t cQ8x1 = wasm_v128_load(vQ8 + 16);
      const v128_t cQ8x2 = wasm_v128_load(vQ8 + 32);
      const v128_t cQ8x3 = wasm_v128_load(vQ8 + 48);
      vQ8 += 64;

      // Low nibble products
      v128_t vVacc1 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(cQ4l0), wasm_i16x8_extend_low_i8x16(cQ8x0));
      vVacc1 = wasm_i32x4_add(
          vVacc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(cQ4l0), wasm_i16x8_extend_high_i8x16(cQ8x0)));
      vVacc1 = wasm_i32x4_add(
          vVacc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(cQ4l1), wasm_i16x8_extend_low_i8x16(cQ8x1)));
      vVacc1 = wasm_i32x4_add(
          vVacc1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(cQ4l1), wasm_i16x8_extend_high_i8x16(cQ8x1)));

      // High nibble products
      v128_t vVacc2 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(cQ4h0), wasm_i16x8_extend_low_i8x16(cQ8x2));
      vVacc2 = wasm_i32x4_add(
          vVacc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(cQ4h0), wasm_i16x8_extend_high_i8x16(cQ8x2)));
      vVacc2 = wasm_i32x4_add(
          vVacc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(cQ4h1), wasm_i16x8_extend_low_i8x16(cQ8x3)));
      vVacc2 = wasm_i32x4_add(
          vVacc2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(cQ4h1), wasm_i16x8_extend_high_i8x16(cQ8x3)));

      // Accumulate scaled results
      int32_t vVacc1Sum = wasm_i32x4_extract_lane(vVacc1, 0) + wasm_i32x4_extract_lane(vVacc1, 1) +
                          wasm_i32x4_extract_lane(vVacc1, 2) + wasm_i32x4_extract_lane(vVacc1, 3);
      vSumi1 += vVacc1Sum * vScales[2 * vJ];

      int32_t vVacc2Sum = wasm_i32x4_extract_lane(vVacc2, 0) + wasm_i32x4_extract_lane(vVacc2, 1) +
                          wasm_i32x4_extract_lane(vVacc2, 2) + wasm_i32x4_extract_lane(vVacc2, 3);
      vSumi2 += vVacc2Sum * vScales[2 * vJ + 1];
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

#if defined __wasm_simd128__
  // const uint8_t * scales = (const uint8_t*)&utmp[0];
  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin); // Fixed sign

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // Process scales and mins
    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    // Sum mins * q8sums
    int32_t vSumiMins = 0;
    const int16_t *GGML_RESTRICT vQ8sums = vY[vI].bsums;
    const uint8_t *vM = (const uint8_t *)&vUtmp[2];
    for (int vJ = 0; vJ < 16; vJ += 2) {
      vSumiMins += (vQ8sums[vJ] + vQ8sums[vJ + 1]) * vM[vJ / 2];
    }
    vSumf -= cDmin * vSumiMins; // Correct subtraction

    v128_t vQh0 = wasm_v128_load(vQh);
    v128_t vQh1 = wasm_v128_load(vQh + 16);
    const uint8_t *vSc = (const uint8_t *)vUtmp;

    int32_t vSumi = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      const int cShift = vJ * 2;
      v128_t vQhShift0 = wasm_u8x16_shr(vQh0, cShift);
      v128_t vQhShift1 = wasm_u8x16_shr(vQh1, cShift);

      v128_t vQhLow0 = wasm_i8x16_shl(wasm_v128_and(vQhShift0, wasm_i8x16_splat(0x01)), 4);
      v128_t vQhHigh0 = wasm_i8x16_shl(wasm_v128_and(vQhShift0, wasm_i8x16_splat(0x02)), 3);
      v128_t vQhLow1 = wasm_i8x16_shl(wasm_v128_and(vQhShift1, wasm_i8x16_splat(0x01)), 4);
      v128_t vQhHigh1 = wasm_i8x16_shl(wasm_v128_and(vQhShift1, wasm_i8x16_splat(0x02)), 3);

      v128_t vQ50 = wasm_v128_load(vQ5);
      v128_t vQ51 = wasm_v128_load(vQ5 + 16);
      vQ5 += 32;

      v128_t vQ5l0 = wasm_v128_or(wasm_v128_and(vQ50, wasm_i8x16_splat(0x0F)), vQhLow0);
      v128_t vQ5h0 = wasm_v128_or(wasm_u8x16_shr(vQ50, 4), vQhHigh0);
      v128_t vQ5l1 = wasm_v128_or(wasm_v128_and(vQ51, wasm_i8x16_splat(0x0F)), vQhLow1);
      v128_t vQ5h1 = wasm_v128_or(wasm_u8x16_shr(vQ51, 4), vQhHigh1);

      v128_t vQ80 = wasm_v128_load(vQ8);
      v128_t vQ81 = wasm_v128_load(vQ8 + 16);
      v128_t vQ82 = wasm_v128_load(vQ8 + 32);
      v128_t vQ83 = wasm_v128_load(vQ8 + 48);
      vQ8 += 64;

      // Process low quants
      v128_t vPl0 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ5l0), wasm_i16x8_extend_low_i8x16(vQ80));
      vPl0 = wasm_i32x4_add(
          vPl0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ5l0), wasm_i16x8_extend_high_i8x16(vQ80)));
      v128_t vPl1 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ5l1), wasm_i16x8_extend_low_i8x16(vQ81));
      vPl1 = wasm_i32x4_add(
          vPl1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ5l1), wasm_i16x8_extend_high_i8x16(vQ81)));
      v128_t vSumLow = wasm_i32x4_add(vPl0, vPl1);

      // Process high quants
      v128_t vPh0 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ5h0), wasm_i16x8_extend_low_i8x16(vQ82));
      vPh0 = wasm_i32x4_add(
          vPh0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ5h0), wasm_i16x8_extend_high_i8x16(vQ82)));
      v128_t vPh1 = wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(vQ5h1), wasm_i16x8_extend_low_i8x16(vQ83));
      vPh1 = wasm_i32x4_add(
          vPh1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(vQ5h1), wasm_i16x8_extend_high_i8x16(vQ83)));
      v128_t vSumHigh = wasm_i32x4_add(vPh0, vPh1);

      // Accumulate with scale factors
      int32_t vSl = wasm_i32x4_extract_lane(vSumLow, 0) + wasm_i32x4_extract_lane(vSumLow, 1) +
                    wasm_i32x4_extract_lane(vSumLow, 2) + wasm_i32x4_extract_lane(vSumLow, 3);
      int32_t vSh = wasm_i32x4_extract_lane(vSumHigh, 0) + wasm_i32x4_extract_lane(vSumHigh, 1) +
                    wasm_i32x4_extract_lane(vSumHigh, 2) + wasm_i32x4_extract_lane(vSumHigh, 3);

      vSumi += vSl * vSc[2 * vJ] + vSh * vSc[2 * vJ + 1];
    }

    vSumf += cD * vSumi;
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
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __wasm_simd128__
  int8_t aAux8[QK_K] __attribute__((aligned(16)));
  int32_t aAux32[8] __attribute__((aligned(16))) = {0};
  float aSums[8] __attribute__((aligned(16))) = {0};

  for (int vI = 0; vI < cNb; ++vI) {
    // Unpack 6-bit quantized data into aux8 (unchanged)
    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    int8_t *vA = aAux8;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int vL = 0; vL < 32; ++vL) {
        vA[vL + 0] = (int8_t)((vQ4[vL + 0] & 0xF) | (((vQh[vL] >> 0) & 3) << 4)) - 32;
        vA[vL + 32] = (int8_t)((vQ4[vL + 32] & 0xF) | (((vQh[vL] >> 2) & 3) << 4)) - 32;
        vA[vL + 64] = (int8_t)((vQ4[vL + 0] >> 4) | (((vQh[vL] >> 4) & 3) << 4)) - 32;
        vA[vL + 96] = (int8_t)((vQ4[vL + 32] >> 4) | (((vQh[vL] >> 6) & 3) << 4)) - 32;
      }
      vA += 128;
      vQ4 += 64;
      vQh += 32;
    }

    const int8_t *GGML_RESTRICT vAPtr = aAux8;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    v128_t vAcc0 = wasm_i32x4_splat(0);
    v128_t vAcc1 = wasm_i32x4_splat(0);

    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      const int cScale = vX[vI].scales[vJ];
      const v128_t cVscale = wasm_i32x4_splat(cScale);

      // Load 16 elements from a and q8
      const v128_t cAVec = wasm_v128_load(vAPtr);
      const v128_t cQ8Vec = wasm_v128_load(vQ8);

      // Process low 8 elements
      v128_t vALow = wasm_i16x8_extend_low_i8x16(cAVec);
      v128_t vQ8Low = wasm_i16x8_extend_low_i8x16(cQ8Vec);
      v128_t vProdLow = wasm_i16x8_mul(vALow, vQ8Low);
      v128_t vProdLoLo = wasm_i32x4_extend_low_i16x8(vProdLow);
      v128_t vProdLoHi = wasm_i32x4_extend_high_i16x8(vProdLow);

      // Process high 8 elements
      v128_t vAHigh = wasm_i16x8_extend_high_i8x16(cAVec);
      v128_t vQ8High = wasm_i16x8_extend_high_i8x16(cQ8Vec);
      v128_t vProdHigh = wasm_i16x8_mul(vAHigh, vQ8High);
      v128_t vProdHiLo = wasm_i32x4_extend_low_i16x8(vProdHigh);
      v128_t vProdHiHi = wasm_i32x4_extend_high_i16x8(vProdHigh);

      // Scale and accumulate
      vProdLoLo = wasm_i32x4_mul(vProdLoLo, cVscale);
      vProdLoHi = wasm_i32x4_mul(vProdLoHi, cVscale);
      vProdHiLo = wasm_i32x4_mul(vProdHiLo, cVscale);
      vProdHiHi = wasm_i32x4_mul(vProdHiHi, cVscale);

      vAcc0 = wasm_i32x4_add(vAcc0, wasm_i32x4_add(vProdLoLo, vProdHiLo));
      vAcc1 = wasm_i32x4_add(vAcc1, wasm_i32x4_add(vProdLoHi, vProdHiHi));

      vAPtr += 16;
      vQ8 += 16;
    }

    // Store accumulated results
    wasm_v128_store(&aAux32[0], vAcc0);
    wasm_v128_store(&aAux32[4], vAcc1);

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    for (int vL = 0; vL < 8; ++vL) {
      aSums[vL] += cD * aAux32[vL];
    }
  }

  // Sum final results
  float vSumf = 0;
  for (int vL = 0; vL < 8; ++vL) {
    vSumf += aSums[vL];
  }
  *pS = vSumf;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
