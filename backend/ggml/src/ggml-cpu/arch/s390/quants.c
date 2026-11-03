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

#if defined(__VXE__) || defined(__VXE2__)
#define B1(c, s, n) 0x##n##c, 0x##n##s
#define B2(c, s, n) B1(c, s, n##c), B1(c, s, n##s)
#define B3(c, s, n) B2(c, s, n##c), B2(c, s, n##s)
#define B4(c, s, n) B3(c, s, n##c), B3(c, s, n##s)
#define B5(c, s, n) B4(c, s, n##c), B4(c, s, n##s)
#define B6(c, s, n) B5(c, s, n##c), B5(c, s, n##s)
#define B7(c, s, n) B6(c, s, n##c), B6(c, s, n##s)
#define B8(c, s) B7(c, s, c), B7(c, s, s)

// precomputed tables for expanding 8bits to 8 bytes:
static const __attribute__((aligned(16))) uint64_t table_b2b_0[1 << 8] = {B8(00, 10)}; // ( b ) << 4
static const __attribute__((aligned(16))) uint64_t table_b2b_1[1 << 8] = {B8(10, 00)}; // (!b) << 4

// permute mask for byteswapping
static const uint8x16_t v_kperm = (const uint8x16_t){7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};
#endif

void quantize_row_q8_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__VXE__) || defined(__VXE2__)
  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t vSrcv[8];
    float32x4_t aAsrcv[8];
    float32x4_t vAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      vSrcv[vJ] = vec_xl(0, pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vec_abs(vSrcv[vJ]);
    for (int vJ = 0; vJ < 4; vJ++)
      vAmaxv[2 * vJ] = vec_max(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      vAmaxv[4 * vJ] = vec_max(vAmaxv[4 * vJ], vAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      vAmaxv[8 * vJ] = vec_max(vAmaxv[8 * vJ], vAmaxv[8 * vJ + 4]);

    const float cAmax = MAX(MAX(vec_extract(vAmaxv[0], 0), vec_extract(vAmaxv[0], 1)),
                            MAX(vec_extract(vAmaxv[0], 2), vec_extract(vAmaxv[0], 3)));

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < 8; vJ++) {
      const float32x4_t cV = vec_mul(vSrcv[vJ], vec_splats(cId));
      /* Uses non-default rounding for vec_signed or vec_round */
      const int32x4_t cVi = vec_signed(__builtin_s390_vfisb(cV, 4, 1));

      vY[vI].qs[4 * vJ + 0] = vec_extract(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = vec_extract(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = vec_extract(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = vec_extract(cVi, 3);
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

#if defined(__VXE__) || defined(__VXE2__)
  for (int vI = 0; vI < cNb; vI++) {
    float32x4_t vSrcv[8];
    float32x4_t aAsrcv[8];
    float32x4_t vAmaxv[8];

    for (int vJ = 0; vJ < 8; vJ++)
      vSrcv[vJ] = vec_xl(0, pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vec_abs(vSrcv[vJ]);
    for (int vJ = 0; vJ < 4; vJ++)
      vAmaxv[2 * vJ] = vec_max(aAsrcv[2 * vJ], aAsrcv[2 * vJ + 1]);
    for (int vJ = 0; vJ < 2; vJ++)
      vAmaxv[4 * vJ] = vec_max(vAmaxv[4 * vJ], vAmaxv[4 * vJ + 2]);
    for (int vJ = 0; vJ < 1; vJ++)
      vAmaxv[8 * vJ] = vec_max(vAmaxv[8 * vJ], vAmaxv[8 * vJ + 4]);

    const float cAmax = MAX(MAX(vec_extract(vAmaxv[0], 0), vec_extract(vAmaxv[0], 1)),
                            MAX(vec_extract(vAmaxv[0], 2), vec_extract(vAmaxv[0], 3)));

    const float cD = cAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    int32x4_t vAcc = vec_splats(0);

    for (int vJ = 0; vJ < 8; vJ++) {
      const float32x4_t cV = vec_mul(vSrcv[vJ], vec_splats(cId));
      /* Uses non-default rounding for vec_signed or vec_round */
      const int32x4_t cVi = vec_signed(__builtin_s390_vfisb(cV, 4, 1));

      vY[vI].qs[4 * vJ + 0] = vec_extract(cVi, 0);
      vY[vI].qs[4 * vJ + 1] = vec_extract(cVi, 1);
      vY[vI].qs[4 * vJ + 2] = vec_extract(cVi, 2);
      vY[vI].qs[4 * vJ + 3] = vec_extract(cVi, 3);

      vAcc = vec_add(vAcc, cVi);
    }

    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * (vAcc[0] + vAcc[1] + vAcc[2] + vAcc[3]));
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_1_ref(pX, vY, pK);
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

#if defined(__VXE__) || defined(__VXE2__)
  float32x4_t vAcc = vec_splats(0.0f);

  const uint8x16_t cVM = vec_splats((const uint8_t)0x0F);
  const int8x16_t cVS = vec_splats((const int8_t)0x08);

  for (; vIb < cNb; ++vIb) {
    const uint8x16_t cVX = vec_xl(0, vX[vIb].qs);
    const int8x16_t cVXl = (const int8x16_t)(cVX & cVM);
    const int8x16_t cVXh = (const int8x16_t)(cVX >> 4);

    const int8x16_t cVXls = vec_sub(cVXl, cVS);
    const int8x16_t cVXhs = vec_sub(cVXh, cVS);

    const int8x16_t cVYl = vec_xl(0, vY[vIb].qs);
    const int8x16_t cVYh = vec_xl(QK8_0 / 2, vY[vIb].qs);

    const int16x8_t cVXylso = vec_mulo(cVXls, cVYl);
    const int16x8_t cVXyl = vec_meadd(cVXls, cVYl, cVXylso);
    const int16x8_t cVXyhso = vec_mulo(cVXhs, cVYh);
    const int16x8_t cVXyh = vec_meadd(cVXhs, cVYh, cVXyhso);

    int16x8_t vVXy = cVXyl + cVXyh;
    vVXy += vec_reve(vVXy);

    const float32x4_t cVXy = vec_float(vec_unpackh(vVXy));
    const float32x4_t cVD = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    vAcc = vec_madd(cVXy, cVD, vAcc);
  }

  vSumf = vec_hsum_f32x4(vAcc);
  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_q4_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
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
  float vSumf = 0;

#if defined(__VXE__) || defined(__VXE2__)
  float vSumms = 0;
  float32x4_t vAcc = vec_splats(0.0f);

  const uint8x16_t cVM = vec_splat_u8(0x0F);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    const uint8x16_t cVX = vec_xl(0, vX[vIb].qs);
    const int8x16_t cVXl = (const int8x16_t)(cVX & cVM);
    const int8x16_t cVXh = (const int8x16_t)(cVX >> 4);

    const int8x16_t cVYl = vec_xl(0, vY[vIb].qs);
    const int8x16_t cVYh = vec_xl(QK8_1 / 2, vY[vIb].qs);

    const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVXl, cVYl), cVXh, cVYh);
    const float32x4_t v_xy = vec_float(v_xy_);

    const float32x4_t cVD = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    vAcc = vec_madd(v_xy, cVD, vAcc);
  }

  vSumf = vec_hsum_f32x4(vAcc) + vSumms;
  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
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

  const int cQk = QK_MXFP4;
  const int cNb = pN / cQk;

  const block_mxfp4 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0.0f;

#if defined(__VXE__) || defined(__VXE2__)
  const int8x16_t cVK = vec_xl(0, kvalues_mxfp4);
  const uint8x16_t cVM = vec_splats((const uint8_t)0x0F);

  float32x4_t vVAcc = vec_splats(0.0f);

#pragma GCC unroll 8
  for (; vIb + 1 < cNb; vIb += 2) {
    const block_mxfp4 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_mxfp4 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    const uint8x16_t cVX0 = vec_xl(0, vX0->qs);
    const uint8x16_t cVX1 = vec_xl(0, vX1->qs);

    int8x16_t vVX0l = (int8x16_t)vec_and(cVX0, cVM);
    int8x16_t vVX0h = (int8x16_t)vec_sr(cVX0, 4);
    int8x16_t vVX1l = (int8x16_t)vec_and(cVX1, cVM);
    int8x16_t vVX1h = (int8x16_t)vec_sr(cVX1, 4);

    vVX0l = vec_perm(cVK, cVK, (uchar8x16_t)vVX0l);
    vVX0h = vec_perm(cVK, cVK, (uchar8x16_t)vVX0h);
    vVX1l = vec_perm(cVK, cVK, (uchar8x16_t)vVX1l);
    vVX1h = vec_perm(cVK, cVK, (uchar8x16_t)vVX1h);

    const int8x16_t cVY0l = vec_xl(0, vY0->qs);
    const int8x16_t cVY0h = vec_xl(QK8_0 / 2, vY0->qs);
    const int8x16_t cVY1l = vec_xl(0, vY1->qs);
    const int8x16_t cVY1h = vec_xl(QK8_0 / 2, vY1->qs);

    const int32x4_t cVXy0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVX0l, cVY0l), vVX0h, cVY0h);
    const int32x4_t cVXy1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVX1l, cVY1l), vVX1h, cVY1h);

    const float32x4_t cVXy0f = vec_float(cVXy0);
    const float32x4_t cVXy1f = vec_float(cVXy1);

    const float32x4_t cVD0 = vec_splats(GGML_E8M0_TO_FP32_HALF(vX0->e) * GGML_CPU_FP16_TO_FP32(vY0->d));
    const float32x4_t cVD1 = vec_splats(GGML_E8M0_TO_FP32_HALF(vX1->e) * GGML_CPU_FP16_TO_FP32(vY1->d));

    vVAcc = vec_madd(cVXy0f, cVD0, vVAcc);
    vVAcc = vec_madd(cVXy1f, cVD1, vVAcc);
  }

  for (; vIb < cNb; ++vIb) {
    const block_mxfp4 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];

    const uint8x16_t cVX = vec_xl(0, vX0->qs);

    int8x16_t vVXl = (int8x16_t)vec_and(cVX, cVM);
    int8x16_t vVXh = (int8x16_t)vec_sr(cVX, 4);

    vVXl = vec_perm(cVK, cVK, (uchar8x16_t)vVXl);
    vVXh = vec_perm(cVK, cVK, (uchar8x16_t)vVXh);

    const int8x16_t cVYl = vec_xl(0, vY0->qs);
    const int8x16_t cVYh = vec_xl(QK8_0 / 2, vY0->qs);

    const int32x4_t cVXy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVXl, cVYl), vVXh, cVYh);
    const float32x4_t cVXyf = vec_float(cVXy);

    const float32x4_t cVD = vec_splats(GGML_E8M0_TO_FP32_HALF(vX0->e) * GGML_CPU_FP16_TO_FP32(vY0->d));
    vVAcc = vec_madd(cVXyf, cVD, vVAcc);
  }

  vSumf = vec_hsum_f32x4(vVAcc);
  *pS = vSumf;
#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_mxfp4_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(cQk == QK5_0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0.0f;

#if defined(__VXE__) || defined(__VXE2__)
  float32x4_t vVSum0 = vec_splats(0.0f);
  float32x4_t vVSum1 = vec_splats(0.0f);

  uint32_t vQh0, vQh1;
  uint64_t aTmp0[4], aTmp1[4];

  const uint8x16_t cVM = vec_splats((uint8_t)0x0F);

#pragma GCC unroll 4
  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q5_0 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q5_0 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_0 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    memcpy(&vQh0, vX0->qh, sizeof(vQh0));
    memcpy(&vQh1, vX1->qh, sizeof(vQh1));

    aTmp0[0] = table_b2b_1[(vQh0 >> 0) & 0xFF];
    aTmp0[1] = table_b2b_1[(vQh0 >> 8) & 0xFF];
    aTmp0[2] = table_b2b_1[(vQh0 >> 16) & 0xFF];
    aTmp0[3] = table_b2b_1[(vQh0 >> 24)];

    aTmp1[0] = table_b2b_1[(vQh1 >> 0) & 0xFF];
    aTmp1[1] = table_b2b_1[(vQh1 >> 8) & 0xFF];
    aTmp1[2] = table_b2b_1[(vQh1 >> 16) & 0xFF];
    aTmp1[3] = table_b2b_1[(vQh1 >> 24)];

    int8x16_t vVQh0l = vec_xl(0, (const int8_t *)(aTmp0 + 0));
    int8x16_t vVQh0h = vec_xl(0, (const int8_t *)(aTmp0 + 2));
    int8x16_t vVQh1l = vec_xl(0, (const int8_t *)(aTmp1 + 0));
    int8x16_t vVQh1h = vec_xl(0, (const int8_t *)(aTmp1 + 2));

    // required for fixing the byteorder
    vVQh0l = vec_perm(vVQh0l, vVQh0l, v_kperm);
    vVQh0h = vec_perm(vVQh0h, vVQh0h, v_kperm);
    vVQh1l = vec_perm(vVQh1l, vVQh1l, v_kperm);
    vVQh1h = vec_perm(vVQh1h, vVQh1h, v_kperm);

    const uint8x16_t cVX0 = vec_xl(0, (const uint8_t *)vX0->qs);
    const uint8x16_t cVX1 = vec_xl(0, (const uint8_t *)vX1->qs);

    int8x16_t vVX0l = (int8x16_t)vec_and(cVX0, cVM);
    int8x16_t vVX0h = (int8x16_t)vec_sr(cVX0, 4);
    int8x16_t vVX1l = (int8x16_t)vec_and(cVX1, cVM);
    int8x16_t vVX1h = (int8x16_t)vec_sr(cVX1, 4);

    const int8x16_t cVX0lf = vec_sub(vVX0l, vVQh0l);
    const int8x16_t cVX0hf = vec_sub(vVX0h, vVQh0h);
    const int8x16_t cVX1lf = vec_sub(vVX1l, vVQh1l);
    const int8x16_t cVX1hf = vec_sub(vVX1h, vVQh1h);

    const int8x16_t cVY0l = vec_xl(0, (const int8_t *)vY0->qs);
    const int8x16_t cVY0h = vec_xl(QK8_0 / 2, (const int8_t *)vY0->qs);
    const int8x16_t cVY1l = vec_xl(0, (const int8_t *)vY1->qs);
    const int8x16_t cVY1h = vec_xl(QK8_0 / 2, (const int8_t *)vY1->qs);

    const int32x4_t cVXy0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVX0lf, cVY0l), cVX0hf, cVY0h);
    const int32x4_t cVXy1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVX1lf, cVY1l), cVX1hf, cVY1h);

    const float32x4_t cVXy0f = vec_float(cVXy0);
    const float32x4_t cVXy1f = vec_float(cVXy1);

    const float32x4_t cVD0 = vec_splats(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    const float32x4_t cVD1 = vec_splats(GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));

    vVSum0 = vec_madd(cVXy0f, cVD0, vVSum0);
    vVSum1 = vec_madd(cVXy1f, cVD1, vVSum1);
  }

  vSumf += vec_hsum_f32x4(vVSum0) + vec_hsum_f32x4(vVSum1);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    const block_q5_0 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];

    uint32_t vQh;
    memcpy(&vQh, vX0->qh, sizeof(vQh));

    uint64_t aTmp[4];
    aTmp[0] = table_b2b_1[(vQh >> 0) & 0xFF];
    aTmp[1] = table_b2b_1[(vQh >> 8) & 0xFF];
    aTmp[2] = table_b2b_1[(vQh >> 16) & 0xFF];
    aTmp[3] = table_b2b_1[(vQh >> 24)];

    int8x16_t vVQhl = vec_xl(0, (const int8_t *)(aTmp + 0));
    int8x16_t vVQhh = vec_xl(0, (const int8_t *)(aTmp + 2));

    // required for fixing the byteorder
    vVQhl = vec_perm(vVQhl, vVQhl, v_kperm);
    vVQhh = vec_perm(vVQhh, vVQhh, v_kperm);

    const uint8x16_t cVX = vec_xl(0, (const uint8_t *)vX0->qs);
    int8x16_t vVXl = (int8x16_t)vec_and(cVX, cVM);
    int8x16_t vVXh = (int8x16_t)vec_sr(cVX, 4);

    const int8x16_t cVXlf = vec_sub(vVXl, vVQhl);
    const int8x16_t cVXhf = vec_sub(vVXh, vVQhh);

    const int8x16_t cVYl = vec_xl(0, (const int8_t *)vY0->qs);
    const int8x16_t cVYh = vec_xl(QK8_0 / 2, (const int8_t *)vY0->qs);

    const int32x4_t cVXy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVXlf, cVYl), cVXhf, cVYh);
    const float32x4_t cVXyf = vec_float(cVXy);

    const float32x4_t cVD = vec_splats(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    const float32x4_t cVAcc = vec_madd(cVXyf, cVD, vec_splats(0.0f));

    vSumf += vec_hsum_f32x4(cVAcc);
  }

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_q5_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(cQk == QK5_1);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0.0f;

#if defined(__VXE__) || defined(__VXE2__)
  float32x4_t vVSum0 = vec_splats(0.0f);
  float32x4_t vVSum1 = vec_splats(0.0f);

  float vSumms0 = 0.0f;
  float vSumms1 = 0.0f;

  uint32_t vQh0;
  uint32_t vQh1;

  uint64_t aTmp0[4];
  uint64_t aTmp1[4];

  const uint8x16_t cVM = vec_splats((uint8_t)0x0F);

#pragma GCC unroll 4
  for (; vIb + 1 < cNb; vIb += 2) {
    const block_q5_1 *GGML_RESTRICT vX0 = &vX[vIb + 0];
    const block_q5_1 *GGML_RESTRICT vX1 = &vX[vIb + 1];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb + 0];
    const block_q8_1 *GGML_RESTRICT vY1 = &vY[vIb + 1];

    vSumms0 += GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s);
    vSumms1 += GGML_CPU_FP16_TO_FP32(vX1->m) * GGML_CPU_FP16_TO_FP32(vY1->s);

    memcpy(&vQh0, vX0->qh, sizeof(vQh0));
    memcpy(&vQh1, vX1->qh, sizeof(vQh1));

    aTmp0[0] = table_b2b_0[(vQh0 >> 0) & 0xFF];
    aTmp0[1] = table_b2b_0[(vQh0 >> 8) & 0xFF];
    aTmp0[2] = table_b2b_0[(vQh0 >> 16) & 0xFF];
    aTmp0[3] = table_b2b_0[(vQh0 >> 24)];

    aTmp1[0] = table_b2b_0[(vQh1 >> 0) & 0xFF];
    aTmp1[1] = table_b2b_0[(vQh1 >> 8) & 0xFF];
    aTmp1[2] = table_b2b_0[(vQh1 >> 16) & 0xFF];
    aTmp1[3] = table_b2b_0[(vQh1 >> 24)];

    int8x16_t vVQh0l = vec_xl(0, (const int8_t *)(aTmp0 + 0));
    int8x16_t vVQh0h = vec_xl(0, (const int8_t *)(aTmp0 + 2));
    int8x16_t vVQh1l = vec_xl(0, (const int8_t *)(aTmp1 + 0));
    int8x16_t vVQh1h = vec_xl(0, (const int8_t *)(aTmp1 + 2));

    // required for fixing the byteorder
    vVQh0l = vec_perm(vVQh0l, vVQh0l, v_kperm);
    vVQh0h = vec_perm(vVQh0h, vVQh0h, v_kperm);
    vVQh1l = vec_perm(vVQh1l, vVQh1l, v_kperm);
    vVQh1h = vec_perm(vVQh1h, vVQh1h, v_kperm);

    const uint8x16_t cVX0 = vec_xl(0, vX0->qs);
    const uint8x16_t cVX1 = vec_xl(0, vX1->qs);

    const int8x16_t cVX0l = (int8x16_t)vec_and(cVX0, cVM);
    const int8x16_t cVX0h = (int8x16_t)vec_sr(cVX0, 4);
    const int8x16_t cVX1l = (int8x16_t)vec_and(cVX1, cVM);
    const int8x16_t cVX1h = (int8x16_t)vec_sr(cVX1, 4);

    const int8x16_t cVX0lf = vec_or(cVX0l, vVQh0l);
    const int8x16_t cVX0hf = vec_or(cVX0h, vVQh0h);
    const int8x16_t cVX1lf = vec_or(cVX1l, vVQh1l);
    const int8x16_t cVX1hf = vec_or(cVX1h, vVQh1h);

    const int8x16_t cVY0l = vec_xl(0, vY0->qs);
    const int8x16_t cVY0h = vec_xl(QK8_1 / 2, vY0->qs);
    const int8x16_t cVY1l = vec_xl(0, vY1->qs);
    const int8x16_t cVY1h = vec_xl(QK8_1 / 2, vY1->qs);

    const int32x4_t cVXy0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVX0lf, cVY0l), cVX0hf, cVY0h);
    const int32x4_t cVXy1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVX1lf, cVY1l), cVX1hf, cVY1h);

    const float32x4_t cVXy0f = vec_float(cVXy0);
    const float32x4_t cVXy1f = vec_float(cVXy1);

    const float32x4_t cVD0 = vec_splats(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    const float32x4_t cVD1 = vec_splats(GGML_CPU_FP16_TO_FP32(vX1->d) * GGML_CPU_FP16_TO_FP32(vY1->d));

    vVSum0 = vec_madd(cVXy0f, cVD0, vVSum0);
    vVSum1 = vec_madd(cVXy1f, cVD1, vVSum1);
  }

  vSumf += vec_hsum_f32x4(vVSum0) + vec_hsum_f32x4(vVSum1) + vSumms0 + vSumms1;

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    const block_q5_1 *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_1 *GGML_RESTRICT vY0 = &vY[vIb];

    float vSumms = GGML_CPU_FP16_TO_FP32(vX0->m) * GGML_CPU_FP16_TO_FP32(vY0->s);

    uint32_t vQh;
    memcpy(&vQh, vX0->qh, sizeof(vQh));

    uint64_t aTmp[4];
    aTmp[0] = table_b2b_0[(vQh >> 0) & 0xFF];
    aTmp[1] = table_b2b_0[(vQh >> 8) & 0xFF];
    aTmp[2] = table_b2b_0[(vQh >> 16) & 0xFF];
    aTmp[3] = table_b2b_0[(vQh >> 24)];

    int8x16_t vVQhl = vec_xl(0, (const int8_t *)(aTmp + 0));
    int8x16_t vVQhh = vec_xl(0, (const int8_t *)(aTmp + 2));

    // required for fixing the byteorder
    vVQhl = vec_perm(vVQhl, vVQhl, v_kperm);
    vVQhh = vec_perm(vVQhh, vVQhh, v_kperm);

    const uint8x16_t cVX = vec_xl(0, vX0->qs);
    const int8x16_t cVXl = (int8x16_t)vec_and(cVX, cVM);
    const int8x16_t cVXh = (int8x16_t)vec_sr(cVX, 4);

    const int8x16_t cVXlf = vec_or(cVXl, vVQhl);
    const int8x16_t cVXhf = vec_or(cVXh, vVQhh);

    const int8x16_t cVYl = vec_xl(0, vY0->qs);
    const int8x16_t cVYh = vec_xl(QK8_1 / 2, vY0->qs);

    const int32x4_t cVXy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVXlf, cVYl), cVXhf, cVYh);
    const float32x4_t cVXyf = vec_float(cVXy);

    const float32x4_t cVD = vec_splats(GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d));
    const float32x4_t cVAcc = vec_madd(cVXyf, cVD, cVAcc);

    vSumf += vec_hsum_f32x4(cVAcc) + vSumms;
  }

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
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

#if defined(__VXE__) || defined(__VXE2__)
  float32x4_t vAcc = vec_splats(0.0f);

#pragma GCC unroll 8
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    const int8x16_t cVXl = vec_xl(0, vX[vIb].qs);
    const int8x16_t cVXh = vec_xl(QK8_0 / 2, vX[vIb].qs);
    const int8x16_t cVYl = vec_xl(0, vY[vIb].qs);
    const int8x16_t cVYh = vec_xl(QK8_0 / 2, vY[vIb].qs);

    const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), cVXl, cVYl), cVXh, cVYh);
    const float32x4_t v_xy = vec_float(v_xy_);
    const float32x4_t cVD = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    vAcc = vec_madd(v_xy, cVD, vAcc);
  }

  vSumf = vec_hsum_f32x4(vAcc);

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

#if defined(__VXE__) || defined(__VXE2__)
  uint32_t aAux[3];
  uint32_t vUtmp[4];

  const int32x4_t cVZ = vec_splat_s32(0);
  const uint8x16_t cV3m = vec_splat_u8(0x03);

  const uint8x16_t cV0c = vec_splat_u8(1);
  const uint8x16_t cV1c = vec_sl(cV0c, 1);
  const uint8x16_t cV2c = vec_sl(cV0c, 2);
  const uint8x16_t cV3c = vec_sl(cV0c, 3);

  uint8x16_t vQ3h[4];
  uint8x16_t vQ3b[2];
  int8x16_t aQ3bytes[4];
  int8x16_t aQ8bytes[8];
  uint8x16_t vQhbits[2];

  float vSum = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *restrict vX0l = vX[vI].qs;
    const uint8_t *restrict vX0h = vX[vI].hmask;
    const int8_t *restrict vY0 = vY[vI].qs;

    vQhbits[0] = vec_xl(0, vX0h);
    vQhbits[1] = vec_xl(16, vX0h);

    int32_t vIsum = 0;

    memcpy(aAux, vX[vI].scales, 12);
    vUtmp[3] = ((aAux[1] >> 4) & cKmask2) | (((aAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((aAux[0] >> 4) & cKmask2) | (((aAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (aAux[1] & cKmask2) | (((aAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (aAux[0] & cKmask2) | (((aAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;
    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= 32;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      int32x4_t vIsum0, vIsum1, vIsum2, vIsum3;

      vQ3b[0] = vec_xl(0, vX0l);
      vQ3b[1] = vec_xl(16, vX0l);
      vX0l += 32;

      aQ8bytes[0] = vec_xl(0, vY0);
      aQ8bytes[1] = vec_xl(16, vY0);
      aQ8bytes[2] = vec_xl(32, vY0);
      aQ8bytes[3] = vec_xl(48, vY0);
      aQ8bytes[4] = vec_xl(64, vY0);
      aQ8bytes[5] = vec_xl(80, vY0);
      aQ8bytes[6] = vec_xl(96, vY0);
      aQ8bytes[7] = vec_xl(112, vY0);
      vY0 += 128;

      vQ3h[0] = vec_sl(vec_andc(cV0c, vQhbits[0]), 2);
      vQ3h[1] = vec_sl(vec_andc(cV0c, vQhbits[1]), 2);
      vQ3h[2] = vec_sl(vec_andc(cV1c, vQhbits[0]), 1);
      vQ3h[3] = vec_sl(vec_andc(cV1c, vQhbits[1]), 1);

      aQ3bytes[0] = vec_sub((int8x16_t)vec_and(vQ3b[0], cV3m), (int8x16_t)vQ3h[0]);
      aQ3bytes[1] = vec_sub((int8x16_t)vec_and(vQ3b[1], cV3m), (int8x16_t)vQ3h[1]);
      aQ3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[0], 2), cV3m), (int8x16_t)vQ3h[2]);
      aQ3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[1], 2), cV3m), (int8x16_t)vQ3h[3]);

      vIsum0 = ggml_vec_dot(cVZ, aQ3bytes[0], aQ8bytes[0]);
      vIsum1 = ggml_vec_dot(cVZ, aQ3bytes[1], aQ8bytes[1]);
      vIsum2 = ggml_vec_dot(cVZ, aQ3bytes[2], aQ8bytes[2]);
      vIsum3 = ggml_vec_dot(cVZ, aQ3bytes[3], aQ8bytes[3]);

      vIsum += (vIsum0[0] + vIsum0[1] + vIsum0[2] + vIsum0[3]) * vScale[0];
      vIsum += (vIsum1[0] + vIsum1[1] + vIsum1[2] + vIsum1[3]) * vScale[1];
      vIsum += (vIsum2[0] + vIsum2[1] + vIsum2[2] + vIsum2[3]) * vScale[2];
      vIsum += (vIsum3[0] + vIsum3[1] + vIsum3[2] + vIsum3[3]) * vScale[3];

      vScale += 4;

      vQ3h[0] = vec_andc(cV2c, vQhbits[0]);
      vQ3h[1] = vec_andc(cV2c, vQhbits[1]);
      vQ3h[2] = vec_sr(vec_andc(cV3c, vQhbits[0]), 1);
      vQ3h[3] = vec_sr(vec_andc(cV3c, vQhbits[1]), 1);

      aQ3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[0], 4), cV3m), (int8x16_t)vQ3h[0]);
      aQ3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[1], 4), cV3m), (int8x16_t)vQ3h[1]);
      aQ3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[0], 6), cV3m), (int8x16_t)vQ3h[2]);
      aQ3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(vQ3b[1], 6), cV3m), (int8x16_t)vQ3h[3]);

      vIsum0 = ggml_vec_dot(cVZ, aQ3bytes[0], aQ8bytes[4]);
      vIsum1 = ggml_vec_dot(cVZ, aQ3bytes[1], aQ8bytes[5]);
      vIsum2 = ggml_vec_dot(cVZ, aQ3bytes[2], aQ8bytes[6]);
      vIsum3 = ggml_vec_dot(cVZ, aQ3bytes[3], aQ8bytes[7]);

      vIsum += vec_hsum_i32x4(vIsum0) * vScale[0];
      vIsum += vec_hsum_i32x4(vIsum1) * vScale[1];
      vIsum += vec_hsum_i32x4(vIsum2) * vScale[2];
      vIsum += vec_hsum_i32x4(vIsum3) * vScale[3];

      vScale += 4;

      if (vJ == 0) {
        vQhbits[0] = vec_sr(vQhbits[0], 4);
        vQhbits[1] = vec_sr(vQhbits[1], 4);
      }
    }

    vSum += cD * vIsum;
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

#if defined(__VXE__) || defined(__VXE2__)
  const uint8x16_t cVLm = vec_splat_u8(0x0F);
  const int32x4_t cVZ = vec_splat_s32(0);

  uint8x16_t vVX[2];
  int8x16_t vVXl[2];
  int8x16_t aVY[2];

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const int16x8_t cVYsumsl = vec_xl(0, vY[vI].bsums);
    const int16x8_t cVYsumsh = vec_xl(16, vY[vI].bsums);
    const int16x8_t cVYsums = vec_padd_s16(cVYsumsl, cVYsumsh);

    memcpy(vUtmp, vX[vI].scales, 12);

    uint32x4_t vVMins8 = {0};
    vVMins8 = vec_insert(vUtmp[1] & cKmask1, vVMins8, 0);
    vVMins8 = vec_insert(((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4), vVMins8, 1);

    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[0] &= cKmask1;

    const int16x8_t cVMinsh = (int16x8_t)vec_unpackh((uint8x16_t)vVMins8);

    const int32x4_t cVMinso = vec_mulo(cVYsums, cVMinsh);
    const int32x4_t cVMins = vec_meadd(cVYsums, cVMinsh, cVMinso);
    vSumf -= cDmin * (cVMins[0] + cVMins[1] + cVMins[2] + cVMins[3]);

    const uint8_t *vScales = (const uint8_t *)vUtmp;
    const uint8_t *GGML_RESTRICT vX0 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vY0 = vY[vI].qs;

    int32_t vSumi1 = 0;
    int32_t vSumi2 = 0;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      vVX[0] = vec_xl(0, vX0);
      vVX[1] = vec_xl(16, vX0);
      vX0 += 32;

      aVY[0] = vec_xl(0, vY0);
      aVY[1] = vec_xl(16, vY0);
      vY0 += 32;

      vVXl[0] = (int8x16_t)vec_and(vVX[0], cVLm);
      vVXl[1] = (int8x16_t)vec_and(vVX[1], cVLm);

      const int32x4_t cP1 = ggml_vec_dot(ggml_vec_dot(cVZ, vVXl[0], aVY[0]), vVXl[1], aVY[1]);
      vSumi1 += vec_hsum_i32x4(cP1) * vScales[2 * vJ + 0];

      aVY[0] = vec_xl(0, vY0);
      aVY[1] = vec_xl(16, vY0);
      vY0 += 32;

      vVXl[0] = (int8x16_t)vec_sr(vVX[0], 4);
      vVXl[1] = (int8x16_t)vec_sr(vVX[1], 4);

      const int32x4_t cP2 = ggml_vec_dot(ggml_vec_dot(cVZ, vVXl[0], aVY[0]), vVXl[1], aVY[1]);
      vSumi2 += vec_hsum_i32x4(cP2) * vScales[2 * vJ + 1];
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

#if defined(__VXE__) || defined(__VXE2__)
  const uint8x16_t cVLm = vec_splat_u8(0x0F);
  const uint8x16_t cV1m = vec_splat_u8(0x01);
  const uint8x16_t cV2m = vec_splat_u8(0x02);

  const int32x4_t cVZ = vec_splat_s32(0);

  const uchar8x16_t cVMinsm = {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                               0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

  int8x16_t aQ5b[4];
  uint8x16_t vQ5h[4];

  uint8x16_t vVXl[2];
  uint8x16_t vVXh[2];
  int8x16_t aVY[4];

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const int16x8_t cVYsumsl = vec_xl(0, vY[vI].bsums);
    const int16x8_t cVYsumsh = vec_xl(16, vY[vI].bsums);
    const int16x8_t cVYsums = vec_padd_s16(cVYsumsl, cVYsumsh);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const uint8x16_t cVMins16 = vec_xl(0, (const uint8_t *)vUtmp);
    const uint8x16_t cVMins8 = vec_perm(cVMins16, cVMins16, cVMinsm);
    const int16x8_t cVMinsh = (int16x8_t)vec_unpackh(cVMins8);

    const int32x4_t cVMinsho = vec_mulo(cVYsums, cVMinsh);
    const int32x4_t cVMins = vec_meadd(cVYsums, cVMinsh, cVMinsho);
    const int32_t cMins = vec_hsum_i32x4(cVMins);

    const uint8_t *vScales = (const uint8_t *)vUtmp;
    const uint8_t *GGML_RESTRICT vX0l = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vX0h = vX[vI].qh;
    const int8_t *GGML_RESTRICT vY0 = vY[vI].qs;

    vVXh[0] = vec_xl(0, vX0h);
    vVXh[1] = vec_xl(16, vX0h);

    int32_t vSumi = 0;
    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      vVXl[0] = vec_xl(0, vX0l);
      vVXl[1] = vec_xl(16, vX0l);
      vX0l += 32;

      aVY[0] = vec_xl(0, vY0);
      aVY[1] = vec_xl(16, vY0);
      aVY[2] = vec_xl(32, vY0);
      aVY[3] = vec_xl(48, vY0);
      vY0 += 64;

      vQ5h[0] = vec_sl(vec_and(cV1m, vVXh[0]), 4);
      vQ5h[1] = vec_sl(vec_and(cV1m, vVXh[1]), 4);
      vQ5h[2] = vec_sl(vec_and(cV2m, vVXh[0]), 3);
      vQ5h[3] = vec_sl(vec_and(cV2m, vVXh[1]), 3);
      vVXh[0] = vec_sr(vVXh[0], 2);
      vVXh[1] = vec_sr(vVXh[1], 2);

      aQ5b[0] = (int8x16_t)vec_or(vec_and(vVXl[0], cVLm), vQ5h[0]);
      aQ5b[1] = (int8x16_t)vec_or(vec_and(vVXl[1], cVLm), vQ5h[1]);
      aQ5b[2] = (int8x16_t)vec_or(vec_sr(vVXl[0], 4), vQ5h[2]);
      aQ5b[3] = (int8x16_t)vec_or(vec_sr(vVXl[1], 4), vQ5h[3]);

      int32x4_t vSumi0 = ggml_vec_dot(ggml_vec_dot(cVZ, aQ5b[0], aVY[0]), aQ5b[1], aVY[1]);
      int32x4_t vSumi1 = ggml_vec_dot(ggml_vec_dot(cVZ, aQ5b[2], aVY[2]), aQ5b[3], aVY[3]);

      vSumi += vec_hsum_i32x4(vSumi0) * *vScales++;
      vSumi += vec_hsum_i32x4(vSumi1) * *vScales++;
    }

    vSumf += cD * vSumi - cDmin * cMins;
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

#if defined(__VXE__) || defined(__VXE2__)
  float vSum = 0;

  // Lower 4-bit and upper 2-bit masks
  const uint8x16_t cVLm = vec_splat_u8(0x0F);
  const uint8x16_t cVUm = vec_splat_u8(0x03);

  const int32x4_t cVZ = vec_splat_s32(0);

  int8x16_t aQ6b[4];
  uint8x16_t vQ6h[4];

  uint8x16_t vVXl[4];
  uint8x16_t vVXh[2];
  int8x16_t aVY[4];

  for (int vI = 0; vI < cNb; ++vI) {
    const float cDAll = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vX0l = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vX0h = vX[vI].qh;
    const int8_t *GGML_RESTRICT vY0 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    const int16x8_t cVYsumsl = vec_xl(0, vY[vI].bsums);
    const int16x8_t cVYsumsh = vec_xl(16, vY[vI].bsums);

    const int8x16_t cVScale = vec_xl(0, vScale);
    const int16x8_t cVScalel = vec_unpackh(cVScale);
    const int16x8_t cVScaleh = vec_unpackl(cVScale);

    const int32x4_t cVMinslo = vec_mulo(cVYsumsl, cVScalel);
    const int32x4_t cVMinsl = vec_meadd(cVYsumsl, cVScalel, cVMinslo);
    const int32x4_t cVMinsho = vec_mulo(cVYsumsh, cVScaleh);
    const int32x4_t cVMinsh = vec_meadd(cVYsumsh, cVScaleh, cVMinsho);
    const int32x4_t cVMins = vec_add(cVMinsl, cVMinsh);

    const int32_t cMins = vec_hsum_i32x4(cVMins);

    int32_t vIsum = 0;
    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // Load model upper 2 bits
      vVXh[0] = vec_xl(0, vX0h);
      vVXh[1] = vec_xl(16, vX0h);
      vX0h += 32;

      // Load model lower 4 bits
      vVXl[0] = vec_xl(0, vX0l);
      vVXl[1] = vec_xl(16, vX0l);
      vVXl[2] = vec_xl(32, vX0l);
      vVXl[3] = vec_xl(48, vX0l);
      vX0l += 64;

      // Load activation quants
      aVY[0] = vec_xl(0, vY0);
      aVY[1] = vec_xl(16, vY0);
      aVY[2] = vec_xl(32, vY0);
      aVY[3] = vec_xl(48, vY0);
      vY0 += 64;

      vQ6h[0] = vec_sl(vec_and(cVUm, vVXh[0]), 4);
      vQ6h[1] = vec_sl(vec_and(cVUm, vVXh[1]), 4);
      uint8x16_t vShifted = vec_sr(vVXh[0], 2);
      vQ6h[2] = vec_sl(vec_and(cVUm, vShifted), 4);
      vShifted = vec_sr(vVXh[1], 2);
      vQ6h[3] = vec_sl(vec_and(cVUm, vShifted), 4);

      aQ6b[0] = (int8x16_t)(vec_or(vec_and(vVXl[0], cVLm), vQ6h[0]));
      aQ6b[1] = (int8x16_t)(vec_or(vec_and(vVXl[1], cVLm), vQ6h[1]));
      aQ6b[2] = (int8x16_t)(vec_or(vec_and(vVXl[2], cVLm), vQ6h[2]));
      aQ6b[3] = (int8x16_t)(vec_or(vec_and(vVXl[3], cVLm), vQ6h[3]));

      int32x4_t vSumms0 = ggml_vec_dot(cVZ, aQ6b[0], aVY[0]);
      int32x4_t vSumms1 = ggml_vec_dot(cVZ, aQ6b[1], aVY[1]);
      int32x4_t vSumms2 = ggml_vec_dot(cVZ, aQ6b[2], aVY[2]);
      int32x4_t vSumms3 = ggml_vec_dot(cVZ, aQ6b[3], aVY[3]);

      vIsum += vec_hsum_i32x4(vSumms0) * vScale[0] + vec_hsum_i32x4(vSumms1) * vScale[1] +
               vec_hsum_i32x4(vSumms2) * vScale[2] + vec_hsum_i32x4(vSumms3) * vScale[3];

      vScale += 4;

      // Load activation quants
      aVY[0] = vec_xl(0, vY0);
      aVY[1] = vec_xl(16, vY0);
      aVY[2] = vec_xl(32, vY0);
      aVY[3] = vec_xl(48, vY0);
      vY0 += 64;

      vShifted = vec_sr(vVXh[0], 4);
      vQ6h[0] = vec_sl(vec_and(cVUm, vShifted), 4);
      vShifted = vec_sr(vVXh[1], 4);
      vQ6h[1] = vec_sl(vec_and(cVUm, vShifted), 4);
      vShifted = vec_sr(vVXh[0], 6);
      vQ6h[2] = vec_sl(vec_and(cVUm, vShifted), 4);
      vShifted = vec_sr(vVXh[1], 6);
      vQ6h[3] = vec_sl(vec_and(cVUm, vShifted), 4);

      aQ6b[0] = (int8x16_t)(vec_or(vec_sr(vVXl[0], 4), vQ6h[0]));
      aQ6b[1] = (int8x16_t)(vec_or(vec_sr(vVXl[1], 4), vQ6h[1]));
      aQ6b[2] = (int8x16_t)(vec_or(vec_sr(vVXl[2], 4), vQ6h[2]));
      aQ6b[3] = (int8x16_t)(vec_or(vec_sr(vVXl[3], 4), vQ6h[3]));

      vSumms0 = ggml_vec_dot(cVZ, aQ6b[0], aVY[0]);
      vSumms1 = ggml_vec_dot(cVZ, aQ6b[1], aVY[1]);
      vSumms2 = ggml_vec_dot(cVZ, aQ6b[2], aVY[2]);
      vSumms3 = ggml_vec_dot(cVZ, aQ6b[3], aVY[3]);

      vIsum += vec_hsum_i32x4(vSumms0) * vScale[0] + vec_hsum_i32x4(vSumms1) * vScale[1] +
               vec_hsum_i32x4(vSumms2) * vScale[2] + vec_hsum_i32x4(vSumms3) * vScale[3];

      vScale += 4;
    }

    vSum += cDAll * vY[vI].d * (vIsum - 32 * cMins);
  }

  *pS = vSum;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

// #if defined(__VXE__) || defined(__VXE2__)
// static const int8_t keven_signs_q2xs[1024] = {
//      1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,
//      1,  1,  1,  1, 1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,
//      -1, -1, -1,  1,  1,  1,  1, -1, 1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,
//      1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1, 1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1, 1,
//      -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1, 1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,
//      1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1, 1,  1, -1,  1, -1,  1,  1,  1, -1, 1,
//      -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1, 1,  1,  1, -1, -1,  1,
//      1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1, 1,  1,
//      -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,
//      1, -1, 1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1, 1,
//      1,  1, -1,  1, -1, 1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1,
//      -1, -1, -1, -1,  1,  1, -1,  1,  1, 1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1,
//      -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, 1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,
//      1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1, 1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, 1,
//      -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1, 1,  1, -1,  1, -1, -1,  1, -1,
//      -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1, 1,  1,  1, -1,
//      -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
//      1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
//      -1, -1,  1,  1, 1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,
//      -1, -1,  1,  1,  1,  1, -1, -1, 1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,
//      1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1, 1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1, 1,
//      -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1, 1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,
//      1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1, 1,  1,  1,  1, -1,  1, -1,  1, -1, 1,
//      1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1, 1,  1, -1,  1, -1,  1,
//      -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1, 1,  1,
//      1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1,
//      -1, -1, 1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1,
//      -1, -1, -1,  1, -1,  1, 1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1,
//      -1, -1, -1, -1,  1,  1,  1, -1, -1,  1, 1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1,
//      -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1, 1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,
//      -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1, 1,  1, -1, -1,  1, -1, -1,  1, -1,  1,
//      -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1, 1,  1,  1,  1, -1, -1,
//      -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1, 1,  1,
//      -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,
//      -1,  1, 1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
//      1, -1, -1, -1, -1,  1, 1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1,
//      -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
// };
// #endif

// void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
// const void * GGML_RESTRICT vy, size_t by, int nrc) {
//     assert(n % QK_K == 0);
//     assert(nrc == 1);
//     UNUSED(nrc);
//     UNUSED(bx);
//     UNUSED(by);
//     UNUSED(bs);

//     const block_iq2_xxs * GGML_RESTRICT x = vx;
//     const block_q8_K    * GGML_RESTRICT y = vy;

//     const int nb = n / QK_K;

// #if defined(__VXE__) || defined(__VXE2__)
//    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

//    uint32_t aux32[4];
//    const uint8_t * aux8 = (const uint8_t *)aux32;

//    float sumf = 0;

//    for (int i = 0; i < nb; ++i) {
//        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

//        float sumf1 = 0, sumf2 = 0;

//        for (int ib32 = 0; ib32 < QK_K/32; ib += 2) {
//            int8x16_t q8b0 = vec_xl( 0, q8);
//            int8x16_t qb81 = vec_xl(16, q8);
//            int8x16_t q8b2 = vec_xl(32, q8);
//            int8x16_t q8b3 = vec_xl(48, q8);
//            q8 += 64;

//            memcpy(aux32, q2, 4 * sizeof(uint32_t));
//            q2 += 8;

//            int8x16_t q2u0 = { *(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1])
//            }; int8x16_t q2u1 = { *(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[
//            3]) }; int8x16_t q2u2 = { *(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid +
//            aux8[ 9]) }; int8x16_t q2u3 = { *(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid
//            + aux8[11]) };

//            int8x16_t q2s0 = { *(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 +
//            ((aux32[1] >>  7) & 127)) }; int8x16_t q2s1 = { *(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)),
//            *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127)) }; int8x16_t q2s2 = { *(const int64_t *)(signs64 +
//            ((aux32[3] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127)) }; int8x16_t q2s3 = {
//            *(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) &
//            127)) };

//            q2u0 = vec_mul(q2u0, q2s0);
//            q2u1 = vec_mul(q2u1, q2s1);
//            q2u2 = vec_mul(q2u2, q2s2);
//            q2u3 = vec_mul(q2u3, q2s3);

//            const int32x4_t p1 = ggml_vec_dot(ggml_vec_dot(vec_splat_s32(0), q2u0, q8b0), q2u1, q8b1);
//            const int32x4_t p2 = ggml_vec_dot(ggml_vec_dot(vec_splat_s32(0), q2u2, q8b2), q2u3, q8b3);

//            sumf1 += (p1[0] + p1[1] + p1[2] + p1[3]) * (0.5f + (aux32[1] >> 28));
//            sumf2 += (p2[0] + p2[1] + p2[2] + p2[3]) * (0.5f + (aux32[3] >> 28));
//        }

//        sumf += d * (sumf1 + sumf2);
//    }

//    *s = 0.25f * sumf;

// #else

//     uint32_t aux32[2];
//     const uint8_t * aux8 = (const uint8_t *)aux32;

//     float sumf = 0.f;
//     for (int i = 0; i < nb; ++i) {
//         const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//         const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//         const int8_t   * GGML_RESTRICT q8 = y[i].qs;
//         int32_t bsum = 0;
//         for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
//             memcpy(aux32, q2, 2*sizeof(uint32_t));
//             q2 += 4;
//             const uint32_t ls = 2*(aux32[1] >> 28) + 1;
//             int32_t sumi = 0;
//             for (int l = 0; l < 4; ++l) {
//                 const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
//                 const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
//                 for (int j = 0; j < 8; ++j) {
//                     sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
//                 }
//                 q8 += 8;
//             }
//             bsum += sumi * ls;
//         }
//         sumf += d * bsum;
//     }
//     *s = 0.125f * sumf;
// #endif
// }

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

#if defined(__VXE__) || defined(__VXE2__)
  const int8x16_t cVK = vec_xl(0, kvalues_iq4nl);
  const uint8x16_t cVM = vec_splat_u8(0x0F);

  for (; vIb < cNb; ++vIb) {
    const block_iq4_nl *GGML_RESTRICT vX0 = &vX[vIb];
    const block_q8_0 *GGML_RESTRICT vY0 = &vY[vIb];

    const uint8x16_t cVX = vec_xl(0, vX0->qs);
    int8x16_t vVXl = (int8x16_t)vec_and(cVX, cVM);
    int8x16_t vVXh = (int8x16_t)vec_sr(cVX, 4);

    vVXl = vec_perm(cVK, cVK, (uchar8x16_t)vVXl);
    vVXh = vec_perm(cVK, cVK, (uchar8x16_t)vVXh);

    const int8x16_t cVYl = vec_xl(0, vY0->qs);
    const int8x16_t cVYh = vec_xl(QK8_0 / 2, vY0->qs);
    const int32x4_t cVXy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVXl, cVYl), vVXh, cVYh);

    vSumf += GGML_CPU_FP16_TO_FP32(vX0->d) * GGML_CPU_FP16_TO_FP32(vY0->d) * vec_hsum_i32x4(cVXy);
  }

  *pS = vSumf;
#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_iq4_nl_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
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

#if defined(__VXE__) || defined(__VXE2__)
  const int8x16_t cVK = vec_xl(0, kvalues_iq4nl);
  const uint8x16_t cVM = vec_splat_u8(0x0F);

  float vSumf = 0;

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const uint8_t *GGML_RESTRICT vQ4 = vX[vIbl].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vIbl].qs;

    uint16_t vH = vX[vIbl].scales_h;

    int vSumi1 = 0, vSumi2 = 0;
    for (int vIb = 0; vIb < QK_K / 64; ++vIb) {
      const uint8x16_t cVX0 = vec_xl(0, vQ4);
      const uint8x16_t cVX1 = vec_xl(QK4_NL / 2, vQ4);
      vQ4 += 32;

      int8x16_t vVX0l = (int8x16_t)vec_and(cVX0, cVM);
      int8x16_t vVX0h = (int8x16_t)vec_sr(cVX0, 4);
      int8x16_t vVX1l = (int8x16_t)vec_and(cVX1, cVM);
      int8x16_t vVX1h = (int8x16_t)vec_sr(cVX1, 4);

      vVX0l = vec_perm(cVK, cVK, (uchar8x16_t)vVX0l);
      vVX0h = vec_perm(cVK, cVK, (uchar8x16_t)vVX0h);
      vVX1l = vec_perm(cVK, cVK, (uchar8x16_t)vVX1l);
      vVX1h = vec_perm(cVK, cVK, (uchar8x16_t)vVX1h);

      const int8x16_t cVY0 = vec_xl(0, vQ8);
      const int8x16_t cVY1 = vec_xl(16, vQ8);
      const int8x16_t cVY2 = vec_xl(32, vQ8);
      const int8x16_t cVY3 = vec_xl(48, vQ8);
      vQ8 += 64;

      int32x4_t vVsumi0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVX0l, cVY0), vVX0h, cVY1);
      int32x4_t vVsumi1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), vVX1l, cVY2), vVX1h, cVY3);

      int vLs1 = ((vX[vIbl].scales_l[vIb] & 0xF) | ((vH << 4) & 0x30)) - 32;
      int vLs2 = ((vX[vIbl].scales_l[vIb] >> 4) | ((vH << 2) & 0x30)) - 32;

      vH >>= 4;

      vSumi1 += vec_hsum_i32x4(vVsumi0) * vLs1;
      vSumi2 += vec_hsum_i32x4(vVsumi1) * vLs2;
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
