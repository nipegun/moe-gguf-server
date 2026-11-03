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

#if defined(__POWER9_VECTOR__)
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

#if defined(__POWER9_VECTOR__)
  for (int vI = 0; vI < cNb; vI++) {
    vector float aSrcv[8];
    vector float aAsrcv[8];
    vector float vAmaxv[8];
    vector signed int aVi[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = vec_xl(0, pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vec_abs(aSrcv[vJ]);

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
    const vector float cVid = vec_splats(cId);

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < 8; vJ++) {
      const vector float cV = vec_round(vec_mul(aSrcv[vJ], cVid));
      aVi[vJ] = vec_cts(cV, 0);
    }
    vec_xst(vec_pack(vec_pack(aVi[0], aVi[1]), vec_pack(aVi[2], aVi[3])), 0, &vY[vI].qs[0]);
    vec_xst(vec_pack(vec_pack(aVi[4], aVi[5]), vec_pack(aVi[6], aVi[7])), 16, &vY[vI].qs[0]);
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

#if defined(__POWER9_VECTOR__)
  for (int vI = 0; vI < cNb; vI++) {
    vector float aSrcv[8];
    vector float aAsrcv[8];
    vector float vAmaxv[8];
    vector signed int aVi[8];

    for (int vJ = 0; vJ < 8; vJ++)
      aSrcv[vJ] = vec_xl(0, pX + vI * 32 + 4 * vJ);
    for (int vJ = 0; vJ < 8; vJ++)
      aAsrcv[vJ] = vec_abs(aSrcv[vJ]);

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
    const vector float cVid = vec_splats(cId);

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    vector int vAccv = vec_splats(0);

    for (int vJ = 0; vJ < 8; vJ++) {
      const vector float cV = vec_round(vec_mul(aSrcv[vJ], cVid));
      aVi[vJ] = vec_cts(cV, 0);

      vAccv = vec_add(vAccv, aVi[vJ]);
    }
    vec_xst(vec_pack(vec_pack(aVi[0], aVi[1]), vec_pack(aVi[2], aVi[3])), 0, &vY[vI].qs[0]);
    vec_xst(vec_pack(vec_pack(aVi[4], aVi[5]), vec_pack(aVi[6], aVi[7])), 16, &vY[vI].qs[0]);

    vAccv = vec_add(vAccv, vec_sld(vAccv, vAccv, 4));
    vAccv = vec_add(vAccv, vec_sld(vAccv, vAccv, 8));
    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * vec_extract(vAccv, 0));
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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);
  const vector signed char cV8 = vec_splats((signed char)0x8);

  vector float vVsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);
    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed char vQ4x0 = vec_and(vQxs, cLowMask);
    vector signed char vQ4x1 = vec_sr(vQxs, cV4);

    vQ4x0 = vec_sub(vQ4x0, cV8);
    vQ4x1 = vec_sub(vQ4x1, cV8);

    vector signed short vQv0 = vec_add(vec_mule(vQ4x0, vQ8y0), vec_mulo(vQ4x0, vQ8y0));
    vector signed short vQv1 = vec_add(vec_mule(vQ4x1, vQ8y1), vec_mulo(vQ4x1, vQ8y1));

    vector signed int vVsumi0 = cV0;

    vVsumi0 = vec_sum4s(vQv0, vVsumi0);
    vVsumi0 = vec_sum4s(vQv1, vVsumi0);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

  *pS = vSumf;
#else
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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector float vVxmin = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].m));
    vector float vVys = {GGML_CPU_FP16_TO_FP32(vY[vIb].s), 0.0f, 0.0f, 0.0f};
    vVsumf0 = vec_madd(vVxmin, vVys, vVsumf0);

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);
    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector unsigned char vQ4x0 = (vector unsigned char)vec_and(vQxs, cLowMask);
    vector unsigned char vQ4x1 = (vector unsigned char)vec_sr(vQxs, cV4);

    vector signed int vVsumi0 = cV0;

    vVsumi0 = vec_msum(vQ8y0, vQ4x0, vVsumi0);
    vVsumi0 = vec_msum(vQ8y1, vQ4x1, vVsumi0);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

  *pS = vSumf;
#else
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

  const block_mxfp4 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_MXFP4;

  int vIb = 0;
  float vSumf = 0;

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector unsigned char cVshift4 = vec_splats((unsigned char)4);
  vector float vVsumf0 = vec_splats(0.0f);

  vector signed char vKv = vec_xl(0, (const signed char *)kvalues_mxfp4);

#pragma GCC unroll 8
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_E8M0_TO_FP32_HALF(vX[vIb].e));

    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);

    vector unsigned char vLoNibbles = (vector unsigned char)vec_and(vQxs, cLowMask);
    vector unsigned char vHiNibbles = (vector unsigned char)vec_sr(vQxs, cVshift4);

    vector signed char vQ4x0 = vec_perm(vKv, vKv, vLoNibbles);
    vector signed char vQ4x1 = vec_perm(vKv, vKv, vHiNibbles);

    vector signed short vQv0 = vec_add(vec_mule(vQ4x0, vQ8y0), vec_mulo(vQ4x0, vQ8y0));
    vector signed short vQv1 = vec_add(vec_mule(vQ4x1, vQ8y1), vec_mulo(vQ4x1, vQ8y1));

    vector signed int vVsumi0 = vec_splats((int32_t)0);
    vVsumi0 = vec_sum4s(vQv0, vVsumi0);
    vVsumi0 = vec_sum4s(vQv1, vVsumi0);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVyd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));
  vSumf = vec_extract(vVsumf0, 0);
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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector unsigned char cV4 = vec_splats((unsigned char)4);

  vector float vVsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed long long vAux64x20 = {(uint64_t)(table_b2b_1[vX[vIb].qh[0]]),
                                         (uint64_t)(table_b2b_1[vX[vIb].qh[1]])};
    vector signed long long vAux64x21 = {(uint64_t)(table_b2b_1[vX[vIb].qh[2]]),
                                         (uint64_t)(table_b2b_1[vX[vIb].qh[3]])};

    vector signed char vQh0 = (vector signed char)vAux64x20;
    vector signed char vQh1 = (vector signed char)vAux64x21;

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);

    vector signed char vQ5x0 = vec_sub(vec_and(vQxs, cLowMask), vQh0);
    vector signed char vQ5x1 = vec_sub(vec_sr(vQxs, cV4), vQh1);

    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed short vQv0 = vec_add(vec_mule(vQ5x0, vQ8y0), vec_mulo(vQ5x0, vQ8y0));
    vector signed short vQv1 = vec_add(vec_mule(vQ5x1, vQ8y1), vec_mulo(vQ5x1, vQ8y1));

    vQv0 = vec_add(vQv0, vQv1);

    vector signed int vVsumi0 = vec_add(vec_unpackh(vQv0), vec_unpackl(vQv0));

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

  *pS = vSumf;
#else
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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector float vVxmin = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].m));
    vector float vVys = {GGML_CPU_FP16_TO_FP32(vY[vIb].s), 0.f, 0.f, 0.f};
    vVsumf0 = vec_madd(vVxmin, vVys, vVsumf0);

    vector unsigned long long vAux64x20 = {(uint64_t)(table_b2b_0[vX[vIb].qh[0]]),
                                           (uint64_t)(table_b2b_0[vX[vIb].qh[1]])};
    vector unsigned long long vAux64x21 = {(uint64_t)(table_b2b_0[vX[vIb].qh[2]]),
                                           (uint64_t)(table_b2b_0[vX[vIb].qh[3]])};

    vector signed char vQh0 = (vector signed char)vAux64x20;
    vector signed char vQh1 = (vector signed char)vAux64x21;

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);

    vector unsigned char vQ5x0 = (vector unsigned char)vec_or(vec_and(vQxs, cLowMask), vQh0);
    vector unsigned char vQ5x1 = (vector unsigned char)vec_or(vec_sr(vQxs, cV4), vQh1);

    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed int vVsumi0 = cV0;

    vVsumi0 = vec_msum(vQ8y0, vQ5x0, vVsumi0);
    vVsumi0 = vec_msum(vQ8y1, vQ5x1, vVsumi0);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed int cV0 = vec_splats((int32_t)0);
  vector float vVsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed char vQ8x0 = vec_xl(0, vX[vIb].qs);
    vector signed char vQ8x1 = vec_xl(16, vX[vIb].qs);
    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed short vQv0 = vec_mule(vQ8x0, vQ8y0);
    vector signed short vQv1 = vec_mulo(vQ8x0, vQ8y0);
    vector signed short vQv2 = vec_mule(vQ8x1, vQ8y1);
    vector signed short vQv3 = vec_mulo(vQ8x1, vQ8y1);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;

    vVsumi0 = vec_sum4s(vQv0, vVsumi0);
    vVsumi1 = vec_sum4s(vQv1, vVsumi1);
    vVsumi0 = vec_sum4s(vQv2, vVsumi0);
    vVsumi1 = vec_sum4s(vQv3, vVsumi1);

    vVsumi0 = vec_add(vVsumi0, vVsumi1);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0x3);
  const vector signed char cLowScaleMask = vec_splats((signed char)0xF);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV2 = vec_splats((unsigned char)0x2);
  const vector unsigned char cV6 = vec_splats((unsigned char)0x6);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector float vVxmin = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].dmin));
    vector float vVdmin = vec_mul(vVxmin, vVyd);

    vector signed short vQ8ysums0 = vec_xl(0, vY[vI].bsums);
    vector signed short vQ8ysums1 = vec_xl(16, vY[vI].bsums);

    vector signed char vQ2xmins = (vector signed char)vec_xl(0, vX[vI].scales);
    vector signed char vVscales = vec_and(vQ2xmins, cLowScaleMask);

    vQ2xmins = vec_sr(vQ2xmins, cV4);
    vector signed short vQ2xmins0 = vec_unpackh(vQ2xmins);
    vector signed short vQ2xmins1 = vec_unpackl(vQ2xmins);

    vector signed int vProd0 = vec_mule(vQ2xmins0, vQ8ysums0);
    vector signed int vProd1 = vec_mulo(vQ2xmins0, vQ8ysums0);
    vector signed int vProd2 = vec_mule(vQ2xmins1, vQ8ysums1);
    vector signed int vProd3 = vec_mulo(vQ2xmins1, vQ8ysums1);

    vVsumf0 = vec_nmsub(vec_ctf(vProd0, 0), vVdmin, vVsumf0);
    vVsumf1 = vec_nmsub(vec_ctf(vProd1, 0), vVdmin, vVsumf1);
    vVsumf2 = vec_nmsub(vec_ctf(vProd2, 0), vVdmin, vVsumf2);
    vVsumf3 = vec_nmsub(vec_ctf(vProd3, 0), vVdmin, vVsumf3);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;
    vector signed int vVsumi4 = cV0;
    vector signed int vVsumi5 = cV0;
    vector signed int vVsumi6 = cV0;
    vector signed int vVsumi7 = cV0;

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      __builtin_prefetch(vQ2, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ2);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ2);
      vQ2 += 32;

      vector unsigned char vQ2x00 = (vector unsigned char)vec_and(vQxs0, cLowMask);
      vector unsigned char vQ2x01 = (vector unsigned char)vec_and(vec_sr(vQxs0, cV2), cLowMask);
      vector unsigned char vQ2x02 = (vector unsigned char)vec_and(vec_sr(vQxs0, cV4), cLowMask);
      vector unsigned char vQ2x03 = (vector unsigned char)vec_and(vec_sr(vQxs0, cV6), cLowMask);
      vector unsigned char vQ2x10 = (vector unsigned char)vec_and(vQxs1, cLowMask);
      vector unsigned char vQ2x11 = (vector unsigned char)vec_and(vec_sr(vQxs1, cV2), cLowMask);
      vector unsigned char vQ2x12 = (vector unsigned char)vec_and(vec_sr(vQxs1, cV4), cLowMask);
      vector unsigned char vQ2x13 = (vector unsigned char)vec_and(vec_sr(vQxs1, cV6), cLowMask);

      vector signed char vQ8y00 = vec_xl(0, vQ8);
      vector signed char vQ8y10 = vec_xl(16, vQ8);
      vector signed char vQ8y01 = vec_xl(32, vQ8);
      vector signed char vQ8y11 = vec_xl(48, vQ8);
      vector signed char vQ8y02 = vec_xl(64, vQ8);
      vector signed char vQ8y12 = vec_xl(80, vQ8);
      vector signed char vQ8y03 = vec_xl(96, vQ8);
      vector signed char vQ8y13 = vec_xl(112, vQ8);
      vQ8 += 128;

      vector signed int vQv0 = vec_msum(vQ8y00, vQ2x00, cV0);
      vector signed int vQv1 = vec_msum(vQ8y01, vQ2x01, cV0);
      vector signed int vQv2 = vec_msum(vQ8y02, vQ2x02, cV0);
      vector signed int vQv3 = vec_msum(vQ8y03, vQ2x03, cV0);
      vector signed int vQv4 = vec_msum(vQ8y10, vQ2x10, cV0);
      vector signed int vQv5 = vec_msum(vQ8y11, vQ2x11, cV0);
      vector signed int vQv6 = vec_msum(vQ8y12, vQ2x12, cV0);
      vector signed int vQv7 = vec_msum(vQ8y13, vQ2x13, cV0);

      vector signed short vVscales07 = vec_unpackh(vVscales);
      vector signed int vVscales03 = vec_unpackh(vVscales07);
      vector signed int vVscales47 = vec_unpackl(vVscales07);
      vector signed int vVs0 = vec_splat(vVscales03, 0);
      vector signed int vVs1 = vec_splat(vVscales03, 1);
      vector signed int vVs2 = vec_splat(vVscales03, 2);
      vector signed int vVs3 = vec_splat(vVscales03, 3);
      vector signed int vVs4 = vec_splat(vVscales47, 0);
      vector signed int vVs5 = vec_splat(vVscales47, 1);
      vector signed int vVs6 = vec_splat(vVscales47, 2);
      vector signed int vVs7 = vec_splat(vVscales47, 3);
      vVscales = vec_sld(vVscales, vVscales, 8);

      vVsumi0 = vec_add(vec_mul(vQv0, vVs0), vVsumi0);
      vVsumi1 = vec_add(vec_mul(vQv1, vVs2), vVsumi1);
      vVsumi2 = vec_add(vec_mul(vQv2, vVs4), vVsumi2);
      vVsumi3 = vec_add(vec_mul(vQv3, vVs6), vVsumi3);
      vVsumi4 = vec_add(vec_mul(vQv4, vVs1), vVsumi4);
      vVsumi5 = vec_add(vec_mul(vQv5, vVs3), vVsumi5);
      vVsumi6 = vec_add(vec_mul(vQv6, vVs5), vVsumi6);
      vVsumi7 = vec_add(vec_mul(vQv7, vVs7), vVsumi7);
    }

    vVsumi0 = vec_add(vVsumi0, vVsumi4);
    vVsumi1 = vec_add(vVsumi1, vVsumi5);
    vVsumi2 = vec_add(vVsumi2, vVsumi6);
    vVsumi3 = vec_add(vVsumi3, vVsumi7);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0x3);
  const vector signed char cLowMask1 = vec_splats((int8_t)0xf);
  const vector signed char cLowMask2 = vec_splats((int8_t)0x30);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector signed char cV1 = vec_splats((signed char)0x1);
  const vector unsigned char cV2 = vec_splats((unsigned char)0x2);
  const vector unsigned char cV3 = vec_splats((unsigned char)0x3);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);
  const vector unsigned char cV6 = vec_splats((unsigned char)0x6);
  const vector signed char cOff = vec_splats((signed char)0x20);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    UNUSED(cKmask1);
    UNUSED(cKmask2);

    vector signed char vU0 = (vector signed char)vec_xl_len(vX[vI].scales, 8);
    vector signed char vU1 = vec_and(vU0, cLowMask1);
    vector signed char vU2 = (vector signed char)vec_xl_len(vX[vI].scales + 8, 4);
    vector signed char vU3 =
        (vector signed char)vec_mergeh((vector signed int)vU2, (vector signed int)vec_sr(vU2, cV2));
    vector signed char vU30 = vec_sl(vec_and(vU3, cLowMask), cV4);
    vector signed char vU31 = vec_and(vU3, cLowMask2);

    vU1 = vec_or(vU1, vU30);
    vU2 = vec_or(vec_sr(vU0, cV4), vU31);

    vector signed char vVscales =
        (vector signed char)vec_mergeh((vector signed long long)vU1, (vector signed long long)vU2);
    vector signed char vQxhs0 = (vector signed char)vec_xl(0, vX[vI].hmask);
    vector signed char vQxhs1 = (vector signed char)vec_xl(16, vX[vI].hmask);

    vVscales = vec_sub(vVscales, cOff);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;
    vector signed int vVsumi4 = cV0;
    vector signed int vVsumi5 = cV0;
    vector signed int vVsumi6 = cV0;
    vector signed int vVsumi7 = cV0;

    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      __builtin_prefetch(vQ3, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ3);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ3);
      vQ3 += 32;

      // the low 2 bits
      vector signed char vQxs00 = vec_and(vQxs0, cLowMask);
      vector signed char vQxs01 = vec_and(vec_sr(vQxs0, cV2), cLowMask);
      vector signed char vQxs02 = vec_and(vec_sr(vQxs0, cV4), cLowMask);
      vector signed char vQxs03 = vec_and(vec_sr(vQxs0, cV6), cLowMask);
      vector signed char vQxs10 = vec_and(vQxs1, cLowMask);
      vector signed char vQxs11 = vec_and(vec_sr(vQxs1, cV2), cLowMask);
      vector signed char vQxs12 = vec_and(vec_sr(vQxs1, cV4), cLowMask);
      vector signed char vQxs13 = vec_and(vec_sr(vQxs1, cV6), cLowMask);

      // the 3rd bit
      vector signed char vQxh00 = vec_sl(vec_andc(cV1, vQxhs0), cV2);
      vector signed char vQxh01 = vec_sl(vec_andc(cV1, vec_sr(vQxhs0, (vector unsigned char)cV1)), cV2);
      vector signed char vQxh02 = vec_sl(vec_andc(cV1, vec_sr(vQxhs0, cV2)), cV2);
      vector signed char vQxh03 = vec_sl(vec_andc(cV1, vec_sr(vQxhs0, cV3)), cV2);
      vector signed char vQxh10 = vec_sl(vec_andc(cV1, vQxhs1), cV2);
      vector signed char vQxh11 = vec_sl(vec_andc(cV1, vec_sr(vQxhs1, (vector unsigned char)cV1)), cV2);
      vector signed char vQxh12 = vec_sl(vec_andc(cV1, vec_sr(vQxhs1, cV2)), cV2);
      vector signed char vQxh13 = vec_sl(vec_andc(cV1, vec_sr(vQxhs1, cV3)), cV2);
      vQxhs0 = vec_sr(vQxhs0, cV4);
      vQxhs1 = vec_sr(vQxhs1, cV4);

      vector signed char vQ3x00 = vec_sub(vQxs00, vQxh00);
      vector signed char vQ3x01 = vec_sub(vQxs01, vQxh01);
      vector signed char vQ3x02 = vec_sub(vQxs02, vQxh02);
      vector signed char vQ3x03 = vec_sub(vQxs03, vQxh03);
      vector signed char vQ3x10 = vec_sub(vQxs10, vQxh10);
      vector signed char vQ3x11 = vec_sub(vQxs11, vQxh11);
      vector signed char vQ3x12 = vec_sub(vQxs12, vQxh12);
      vector signed char vQ3x13 = vec_sub(vQxs13, vQxh13);

      vector signed char vQ8y00 = vec_xl(0, vQ8);
      vector signed char vQ8y10 = vec_xl(16, vQ8);
      vector signed char vQ8y01 = vec_xl(32, vQ8);
      vector signed char vQ8y11 = vec_xl(48, vQ8);
      vector signed char vQ8y02 = vec_xl(64, vQ8);
      vector signed char vQ8y12 = vec_xl(80, vQ8);
      vector signed char vQ8y03 = vec_xl(96, vQ8);
      vector signed char vQ8y13 = vec_xl(112, vQ8);
      vQ8 += 128;

      vector signed short vVscalesH = vec_unpackh(vVscales);
      vector signed short vVs0 = vec_splat(vVscalesH, 0);
      vector signed short vVs1 = vec_splat(vVscalesH, 1);
      vector signed short vVs2 = vec_splat(vVscalesH, 2);
      vector signed short vVs3 = vec_splat(vVscalesH, 3);
      vector signed short vVs4 = vec_splat(vVscalesH, 4);
      vector signed short vVs5 = vec_splat(vVscalesH, 5);
      vector signed short vVs6 = vec_splat(vVscalesH, 6);
      vector signed short vVs7 = vec_splat(vVscalesH, 7);
      vVscales = vec_sld(vVscales, vVscales, 8);

      vector signed short vQv00 = vec_add(vec_mule(vQ3x00, vQ8y00), vec_mulo(vQ3x00, vQ8y00));
      vector signed short vQv01 = vec_add(vec_mule(vQ3x01, vQ8y01), vec_mulo(vQ3x01, vQ8y01));
      vector signed short vQv02 = vec_add(vec_mule(vQ3x02, vQ8y02), vec_mulo(vQ3x02, vQ8y02));
      vector signed short vQv03 = vec_add(vec_mule(vQ3x03, vQ8y03), vec_mulo(vQ3x03, vQ8y03));
      vector signed short vQv10 = vec_add(vec_mule(vQ3x10, vQ8y10), vec_mulo(vQ3x10, vQ8y10));
      vector signed short vQv11 = vec_add(vec_mule(vQ3x11, vQ8y11), vec_mulo(vQ3x11, vQ8y11));
      vector signed short vQv12 = vec_add(vec_mule(vQ3x12, vQ8y12), vec_mulo(vQ3x12, vQ8y12));
      vector signed short vQv13 = vec_add(vec_mule(vQ3x13, vQ8y13), vec_mulo(vQ3x13, vQ8y13));

      vVsumi0 = vec_msum(vQv00, vVs0, vVsumi0);
      vVsumi1 = vec_msum(vQv01, vVs2, vVsumi1);
      vVsumi2 = vec_msum(vQv02, vVs4, vVsumi2);
      vVsumi3 = vec_msum(vQv03, vVs6, vVsumi3);
      vVsumi4 = vec_msum(vQv10, vVs1, vVsumi4);
      vVsumi5 = vec_msum(vQv11, vVs3, vVsumi5);
      vVsumi6 = vec_msum(vQv12, vVs5, vVsumi6);
      vVsumi7 = vec_msum(vQv13, vVs7, vVsumi7);
    }

    vVsumi0 = vec_add(vVsumi0, vVsumi4);
    vVsumi1 = vec_add(vVsumi1, vVsumi5);
    vVsumi2 = vec_add(vVsumi2, vVsumi6);
    vVsumi3 = vec_add(vVsumi3, vVsumi7);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed char cLowMask1 = vec_splats((int8_t)0x3f);
  const vector signed char cLowMask2 = vec_splats((int8_t)0x30);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV2 = vec_splats((uint8_t)2);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector float vVxmin = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].dmin));
    vector float vVdmin = vec_mul(vVxmin, vVyd);

    vector signed short vQ8ysums0 = vec_xl(0, vY[vI].bsums);
    vector signed short vQ8ysums1 = vec_xl(16, vY[vI].bsums);

    UNUSED(cKmask1);
    UNUSED(cKmask2);
    UNUSED(cKmask3);
    UNUSED(vUtmp);

    vector signed char vU0 = (vector signed char)vec_xl_len(vX[vI].scales, 8);
    vector signed char vU1 = vec_and(vec_sr(vU0, cV2), cLowMask2);
    vector signed char vU2 = (vector signed char)vec_xl_len(vX[vI].scales + 8, 4);
    vector signed char vU3 = vec_sr(vU2, cV4);

    vector signed char vU30 = vU1;
    vector signed char vU31 =
        (vector signed char)vec_mergeh((vector signed int)vec_and(vU2, cLowMask), (vector signed int)vU3);

    vU1 = vec_and(vU0, cLowMask1);
    vU2 = vec_or(vU30, vU31);

    vector signed char vUtmps = (vector signed char)vec_mergeh((vector signed int)vU1, (vector signed int)vU2);

    vector signed short vVscales = vec_unpackh(vUtmps);
    vector signed short vQ4xmins = vec_unpackl(vUtmps);
    vector signed short vQ4xmins0 = vec_mergeh(vQ4xmins, vQ4xmins);
    vector signed short vQ4xmins1 = vec_mergel(vQ4xmins, vQ4xmins);

    vector signed int vProd0 = vec_mule(vQ4xmins0, vQ8ysums0);
    vector signed int vProd1 = vec_mule(vQ4xmins1, vQ8ysums1);
    vector signed int vProd2 = vec_mulo(vQ4xmins0, vQ8ysums0);
    vector signed int vProd3 = vec_mulo(vQ4xmins1, vQ8ysums1);

    vVsumf0 = vec_nmsub(vec_ctf(vProd0, 0), vVdmin, vVsumf0);
    vVsumf1 = vec_nmsub(vec_ctf(vProd1, 0), vVdmin, vVsumf1);
    vVsumf2 = vec_nmsub(vec_ctf(vProd2, 0), vVdmin, vVsumf2);
    vVsumf3 = vec_nmsub(vec_ctf(vProd3, 0), vVdmin, vVsumf3);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 64; vJ += 2) {
      __builtin_prefetch(vQ4, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ4);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ4);
      vector signed char vQxs2 = (vector signed char)vec_xl(32, vQ4);
      vector signed char vQxs3 = (vector signed char)vec_xl(48, vQ4);
      vQ4 += 64;

      vector unsigned char vQ4x00 = (vector unsigned char)vec_and(vQxs0, cLowMask);
      vector unsigned char vQ4x01 = (vector unsigned char)vec_sr(vQxs0, cV4);
      vector unsigned char vQ4x10 = (vector unsigned char)vec_and(vQxs1, cLowMask);
      vector unsigned char vQ4x11 = (vector unsigned char)vec_sr(vQxs1, cV4);
      vector unsigned char vQ4x20 = (vector unsigned char)vec_and(vQxs2, cLowMask);
      vector unsigned char vQ4x21 = (vector unsigned char)vec_sr(vQxs2, cV4);
      vector unsigned char vQ4x30 = (vector unsigned char)vec_and(vQxs3, cLowMask);
      vector unsigned char vQ4x31 = (vector unsigned char)vec_sr(vQxs3, cV4);

      vector signed char vQ8y00 = vec_xl(0, vQ8);
      vector signed char vQ8y10 = vec_xl(16, vQ8);
      vector signed char vQ8y01 = vec_xl(32, vQ8);
      vector signed char vQ8y11 = vec_xl(48, vQ8);
      vector signed char vQ8y20 = vec_xl(64, vQ8);
      vector signed char vQ8y30 = vec_xl(80, vQ8);
      vector signed char vQ8y21 = vec_xl(96, vQ8);
      vector signed char vQ8y31 = vec_xl(112, vQ8);
      vQ8 += 128;

      vector signed int vQv00 = vec_msum(vQ8y00, vQ4x00, cV0);
      vector signed int vQv01 = vec_msum(vQ8y01, vQ4x01, cV0);
      vector signed int vQv10 = vec_msum(vQ8y10, vQ4x10, cV0);
      vector signed int vQv11 = vec_msum(vQ8y11, vQ4x11, cV0);
      vector signed int vQv20 = vec_msum(vQ8y20, vQ4x20, cV0);
      vector signed int vQv21 = vec_msum(vQ8y21, vQ4x21, cV0);
      vector signed int vQv30 = vec_msum(vQ8y30, vQ4x30, cV0);
      vector signed int vQv31 = vec_msum(vQ8y31, vQ4x31, cV0);

      vector signed int vVscalesH = vec_unpackh(vVscales);
      vector signed int vVs0 = vec_splat(vVscalesH, 0);
      vector signed int vVs1 = vec_splat(vVscalesH, 1);
      vector signed int vVs2 = vec_splat(vVscalesH, 2);
      vector signed int vVs3 = vec_splat(vVscalesH, 3);
      vVscales = vec_sld(vVscales, vVscales, 8);

      vVsumi0 = vec_add(vec_mul(vQv00, vVs0), vVsumi0);
      vVsumi1 = vec_add(vec_mul(vQv01, vVs1), vVsumi1);
      vVsumi2 = vec_add(vec_mul(vQv20, vVs2), vVsumi2);
      vVsumi3 = vec_add(vec_mul(vQv21, vVs3), vVsumi3);

      vVsumi0 = vec_add(vec_mul(vQv10, vVs0), vVsumi0);
      vVsumi1 = vec_add(vec_mul(vQv11, vVs1), vVsumi1);
      vVsumi2 = vec_add(vec_mul(vQv30, vVs2), vVsumi2);
      vVsumi3 = vec_add(vec_mul(vQv31, vVs3), vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed char cLowMask1 = vec_splats((int8_t)0x3f);
  const vector signed char cLowMask2 = vec_splats((int8_t)0x30);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV1 = vec_splats((unsigned char)0x1);
  const vector unsigned char cV2 = vec_splats((unsigned char)0x2);
  const vector unsigned char cV3 = vec_splats((unsigned char)0x3);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector float vVxmin = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].dmin));
    vector float vVdmin = vec_mul(vVxmin, vVyd);

    UNUSED(cKmask1);
    UNUSED(cKmask2);
    UNUSED(cKmask3);
    UNUSED(vUtmp);

    vector signed char vU0 = (vector signed char)vec_xl_len(vX[vI].scales, 8);
    vector signed char vU1 = vec_and(vec_sr(vU0, cV2), cLowMask2);
    vector signed char vU2 = (vector signed char)vec_xl_len(vX[vI].scales + 8, 4);
    vector signed char vU3 = vec_sr(vU2, cV4);

    vector signed char vU30 = vU1;
    vector signed char vU31 =
        (vector signed char)vec_mergeh((vector signed int)vec_and(vU2, cLowMask), (vector signed int)vU3);

    vU1 = vec_and(vU0, cLowMask1);
    vU2 = vec_or(vU30, vU31);

    vector signed char vUtmps = (vector signed char)vec_mergeh((vector signed int)vU1, (vector signed int)vU2);

    vector signed short vQ8ysums0 = vec_xl(0, vY[vI].bsums);
    vector signed short vQ8ysums1 = vec_xl(16, vY[vI].bsums);

    vector signed short vVscales = vec_unpackh(vUtmps);

    vector signed short vQ5xmins = vec_unpackl(vUtmps);
    vector signed short vQ5xmins0 = vec_mergeh(vQ5xmins, vQ5xmins);
    vector signed short vQ5xmins1 = vec_mergel(vQ5xmins, vQ5xmins);

    vector signed int vProd0 = vec_mule(vQ5xmins0, vQ8ysums0);
    vector signed int vProd1 = vec_mule(vQ5xmins1, vQ8ysums1);
    vector signed int vProd2 = vec_mulo(vQ5xmins0, vQ8ysums0);
    vector signed int vProd3 = vec_mulo(vQ5xmins1, vQ8ysums1);

    vVsumf0 = vec_nmsub(vec_ctf(vProd0, 0), vVdmin, vVsumf0);
    vVsumf1 = vec_nmsub(vec_ctf(vProd1, 0), vVdmin, vVsumf1);
    vVsumf2 = vec_nmsub(vec_ctf(vProd2, 0), vVdmin, vVsumf2);
    vVsumf3 = vec_nmsub(vec_ctf(vProd3, 0), vVdmin, vVsumf3);

    vector signed char vQxhs0 = (vector signed char)vec_xl(0, vX[vI].qh);
    vector signed char vQxhs1 = (vector signed char)vec_xl(16, vX[vI].qh);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      __builtin_prefetch(vQ5, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ5);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ5);
      vQ5 += 32;

      vector signed char vQxs00 = vec_and(vQxs0, cLowMask);
      vector signed char vQxs01 = vec_sr(vQxs0, cV4);
      vector signed char vQxs10 = vec_and(vQxs1, cLowMask);
      vector signed char vQxs11 = vec_sr(vQxs1, cV4);

      vector signed char vQ5h00 = vec_sl(vec_and((vector signed char)cV1, vQxhs0), cV4);
      vector signed char vQ5h01 = vec_sl(vec_and((vector signed char)cV2, vQxhs0), cV3);
      vector signed char vQ5h10 = vec_sl(vec_and((vector signed char)cV1, vQxhs1), cV4);
      vector signed char vQ5h11 = vec_sl(vec_and((vector signed char)cV2, vQxhs1), cV3);
      vQxhs0 = vec_sr(vQxhs0, cV2);
      vQxhs1 = vec_sr(vQxhs1, cV2);

      vector unsigned char vQ5x00 = (vector unsigned char)vec_or(vQ5h00, vQxs00);
      vector unsigned char vQ5x01 = (vector unsigned char)vec_or(vQ5h01, vQxs01);
      vector unsigned char vQ5x10 = (vector unsigned char)vec_or(vQ5h10, vQxs10);
      vector unsigned char vQ5x11 = (vector unsigned char)vec_or(vQ5h11, vQxs11);

      vector signed char vQ8y00 = vec_xl(0, vQ8);
      vector signed char vQ8y10 = vec_xl(16, vQ8);
      vector signed char vQ8y01 = vec_xl(32, vQ8);
      vector signed char vQ8y11 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed int vQv00 = vec_msum(vQ8y00, vQ5x00, cV0);
      vector signed int vQv01 = vec_msum(vQ8y01, vQ5x01, cV0);
      vector signed int vQv10 = vec_msum(vQ8y10, vQ5x10, cV0);
      vector signed int vQv11 = vec_msum(vQ8y11, vQ5x11, cV0);

      vector signed int vVscalesH = vec_unpackh(vVscales);
      vector signed int vVs0 = vec_splat(vVscalesH, 0);
      vector signed int vVs1 = vec_splat(vVscalesH, 1);
      vVscales = vec_sld(vVscales, vVscales, 12);

      vVsumi0 = vec_add(vec_mul(vQv00, vVs0), vVsumi0);
      vVsumi1 = vec_add(vec_mul(vQv10, vVs0), vVsumi1);
      vVsumi2 = vec_add(vec_mul(vQv01, vVs1), vVsumi2);
      vVsumi3 = vec_add(vec_mul(vQv11, vVs1), vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV2 = vec_splats((unsigned char)0x2);
  const vector unsigned char cV3 = vec_splats((unsigned char)0x3);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);
  const vector unsigned char cV6 = vec_splats((unsigned char)0x6);
  const vector signed char cOff = vec_splats((signed char)0x20);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;
    vector signed int vVsumi4 = cV0;
    vector signed int vVsumi5 = cV0;
    vector signed int vVsumi6 = cV0;
    vector signed int vVsumi7 = cV0;

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQs = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      __builtin_prefetch(vQ6, 0, 0);
      __builtin_prefetch(vQh, 0, 0);
      __builtin_prefetch(vQ8, 0, 0);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ6);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ6);
      vector signed char vQxs2 = (vector signed char)vec_xl(32, vQ6);
      vector signed char vQxs3 = (vector signed char)vec_xl(48, vQ6);
      vQ6 += 64;

      vector signed char vQxs00 = vec_and(vQxs0, cLowMask);
      vector signed char vQxs01 = vec_sr(vQxs0, cV4);
      vector signed char vQxs10 = vec_and(vQxs1, cLowMask);
      vector signed char vQxs11 = vec_sr(vQxs1, cV4);
      vector signed char vQxs20 = vec_and(vQxs2, cLowMask);
      vector signed char vQxs21 = vec_sr(vQxs2, cV4);
      vector signed char vQxs30 = vec_and(vQxs3, cLowMask);
      vector signed char vQxs31 = vec_sr(vQxs3, cV4);

      vector signed char vQxhs0 = (vector signed char)vec_xl(0, vQh);
      vector signed char vQxhs1 = (vector signed char)vec_xl(16, vQh);
      vQh += 32;

      vector signed char vQxh00 = vec_sl(vec_and((vector signed char)cV3, vQxhs0), cV4);
      vector signed char vQxh01 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs0, cV4)), cV4);
      vector signed char vQxh10 = vec_sl(vec_and((vector signed char)cV3, vQxhs1), cV4);
      vector signed char vQxh11 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs1, cV4)), cV4);
      vector signed char vQxh20 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs0, cV2)), cV4);
      vector signed char vQxh21 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs0, cV6)), cV4);
      vector signed char vQxh30 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs1, cV2)), cV4);
      vector signed char vQxh31 = vec_sl(vec_and((vector signed char)cV3, vec_sr(vQxhs1, cV6)), cV4);

      vector signed char vQ6x00 = vec_sub(vec_or(vQxh00, vQxs00), cOff);
      vector signed char vQ6x01 = vec_sub(vec_or(vQxh01, vQxs01), cOff);
      vector signed char vQ6x10 = vec_sub(vec_or(vQxh10, vQxs10), cOff);
      vector signed char vQ6x11 = vec_sub(vec_or(vQxh11, vQxs11), cOff);
      vector signed char vQ6x20 = vec_sub(vec_or(vQxh20, vQxs20), cOff);
      vector signed char vQ6x21 = vec_sub(vec_or(vQxh21, vQxs21), cOff);
      vector signed char vQ6x30 = vec_sub(vec_or(vQxh30, vQxs30), cOff);
      vector signed char vQ6x31 = vec_sub(vec_or(vQxh31, vQxs31), cOff);

      vector signed char vQ8y00 = vec_xl(0, vQ8);
      vector signed char vQ8y10 = vec_xl(16, vQ8);
      vector signed char vQ8y20 = vec_xl(32, vQ8);
      vector signed char vQ8y30 = vec_xl(48, vQ8);
      vector signed char vQ8y01 = vec_xl(64, vQ8);
      vector signed char vQ8y11 = vec_xl(80, vQ8);
      vector signed char vQ8y21 = vec_xl(96, vQ8);
      vector signed char vQ8y31 = vec_xl(112, vQ8);
      vQ8 += 128;

      vector signed short vQv00 = vec_add(vec_mule(vQ6x00, vQ8y00), vec_mulo(vQ6x00, vQ8y00));
      vector signed short vQv10 = vec_add(vec_mule(vQ6x10, vQ8y10), vec_mulo(vQ6x10, vQ8y10));
      vector signed short vQv20 = vec_add(vec_mule(vQ6x20, vQ8y20), vec_mulo(vQ6x20, vQ8y20));
      vector signed short vQv30 = vec_add(vec_mule(vQ6x30, vQ8y30), vec_mulo(vQ6x30, vQ8y30));
      vector signed short vQv01 = vec_add(vec_mule(vQ6x01, vQ8y01), vec_mulo(vQ6x01, vQ8y01));
      vector signed short vQv11 = vec_add(vec_mule(vQ6x11, vQ8y11), vec_mulo(vQ6x11, vQ8y11));
      vector signed short vQv21 = vec_add(vec_mule(vQ6x21, vQ8y21), vec_mulo(vQ6x21, vQ8y21));
      vector signed short vQv31 = vec_add(vec_mule(vQ6x31, vQ8y31), vec_mulo(vQ6x31, vQ8y31));

      vector signed short vVscales = vec_unpackh(vec_xl_len(vQs, 8));
      vQs += 8;

      vector signed short vVs0 = vec_splat(vVscales, 0);
      vector signed short vVs1 = vec_splat(vVscales, 1);
      vector signed short vVs2 = vec_splat(vVscales, 2);
      vector signed short vVs3 = vec_splat(vVscales, 3);
      vector signed short vVs4 = vec_splat(vVscales, 4);
      vector signed short vVs5 = vec_splat(vVscales, 5);
      vector signed short vVs6 = vec_splat(vVscales, 6);
      vector signed short vVs7 = vec_splat(vVscales, 7);

      vVsumi0 = vec_msum(vQv00, vVs0, vVsumi0);
      vVsumi1 = vec_msum(vQv01, vVs4, vVsumi1);
      vVsumi2 = vec_msum(vQv10, vVs1, vVsumi2);
      vVsumi3 = vec_msum(vQv11, vVs5, vVsumi3);
      vVsumi4 = vec_msum(vQv20, vVs2, vVsumi4);
      vVsumi5 = vec_msum(vQv21, vVs6, vVsumi5);
      vVsumi6 = vec_msum(vQv30, vVs3, vVsumi6);
      vVsumi7 = vec_msum(vQv31, vVs7, vVsumi7);
    }

    vVsumi0 = vec_add(vVsumi0, vVsumi4);
    vVsumi1 = vec_add(vVsumi1, vVsumi5);
    vVsumi2 = vec_add(vVsumi2, vVsumi6);
    vVsumi3 = vec_add(vVsumi3, vVsumi7);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined(__POWER9_VECTOR__)
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

#if defined(__POWER9_VECTOR__)
  const vector int cV0 = vec_splats((int32_t)0);
  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 32; vJ += 2) {
      __builtin_prefetch(vQ2, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      uint32_t aAux32[4];
      const uint8_t *vAux8 = (const uint8_t *)aAux32;

      memcpy(aAux32, vQ2, 4 * sizeof(uint32_t));
      vQ2 += 8;

      vector signed long long vAux64x20 = {*(const int64_t *)(iq2xxs_grid + vAux8[0]),
                                           *(const int64_t *)(iq2xxs_grid + vAux8[1])};
      vector signed long long vAux64x21 = {*(const int64_t *)(iq2xxs_grid + vAux8[2]),
                                           *(const int64_t *)(iq2xxs_grid + vAux8[3])};
      vector signed long long vAux64x22 = {*(const int64_t *)(iq2xxs_grid + vAux8[8]),
                                           *(const int64_t *)(iq2xxs_grid + vAux8[9])};
      vector signed long long vAux64x23 = {*(const int64_t *)(iq2xxs_grid + vAux8[10]),
                                           *(const int64_t *)(iq2xxs_grid + vAux8[11])};

      vector signed long long vVsigns0 = {*(const int64_t *)(vSigns64 + ((aAux32[1] >> 0) & 127)),
                                          *(const int64_t *)(vSigns64 + ((aAux32[1] >> 7) & 127))};
      vector signed long long vVsigns1 = {*(const int64_t *)(vSigns64 + ((aAux32[1] >> 14) & 127)),
                                          *(const int64_t *)(vSigns64 + ((aAux32[1] >> 21) & 127))};
      vector signed long long vVsigns2 = {*(const int64_t *)(vSigns64 + ((aAux32[3] >> 0) & 127)),
                                          *(const int64_t *)(vSigns64 + ((aAux32[3] >> 7) & 127))};
      vector signed long long vVsigns3 = {*(const int64_t *)(vSigns64 + ((aAux32[3] >> 14) & 127)),
                                          *(const int64_t *)(vSigns64 + ((aAux32[3] >> 21) & 127))};

      vector signed char vQ2x0 =
          (vector signed char)vec_mul((vector signed char)vVsigns0, (vector signed char)vAux64x20);
      vector signed char vQ2x1 =
          (vector signed char)vec_mul((vector signed char)vVsigns1, (vector signed char)vAux64x21);
      vector signed char vQ2x2 =
          (vector signed char)vec_mul((vector signed char)vVsigns2, (vector signed char)vAux64x22);
      vector signed char vQ2x3 =
          (vector signed char)vec_mul((vector signed char)vVsigns3, (vector signed char)vAux64x23);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ2x0, vQ8y0), vec_mulo(vQ2x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ2x1, vQ8y1), vec_mulo(vQ2x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ2x2, vQ8y2), vec_mulo(vQ2x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ2x3, vQ8y3), vec_mulo(vQ2x3, vQ8y3));

      const uint16_t cLs0 = aAux32[1] >> 28;
      const uint16_t cLs1 = aAux32[3] >> 28;

      vector signed short vVscales01 = vec_splats((int16_t)(2 * cLs0 + 1));
      vector signed short vVscales23 = vec_splats((int16_t)(2 * cLs1 + 1));

      vVsumi0 = vec_msum(vQv0, vVscales01, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales01, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales23, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales23, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = 0.125f * vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector int cV0 = vec_splats((int32_t)0);
  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      __builtin_prefetch(vQ2, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed long long vAux64x20 = {*(const int64_t *)(iq2xs_grid + (vQ2[0] & 511)),
                                           *(const int64_t *)(iq2xs_grid + (vQ2[1] & 511))};
      vector signed long long vAux64x21 = {*(const int64_t *)(iq2xs_grid + (vQ2[2] & 511)),
                                           *(const int64_t *)(iq2xs_grid + (vQ2[3] & 511))};
      vector signed long long vAux64x22 = {*(const int64_t *)(iq2xs_grid + (vQ2[4] & 511)),
                                           *(const int64_t *)(iq2xs_grid + (vQ2[5] & 511))};
      vector signed long long vAux64x23 = {*(const int64_t *)(iq2xs_grid + (vQ2[6] & 511)),
                                           *(const int64_t *)(iq2xs_grid + (vQ2[7] & 511))};

      vector signed long long vVsigns0 = {*(const int64_t *)(vSigns64 + ((vQ2[0] >> 9))),
                                          *(const int64_t *)(vSigns64 + ((vQ2[1] >> 9)))};
      vector signed long long vVsigns1 = {*(const int64_t *)(vSigns64 + ((vQ2[2] >> 9))),
                                          *(const int64_t *)(vSigns64 + ((vQ2[3] >> 9)))};
      vector signed long long vVsigns2 = {*(const int64_t *)(vSigns64 + ((vQ2[4] >> 9))),
                                          *(const int64_t *)(vSigns64 + ((vQ2[5] >> 9)))};
      vector signed long long vVsigns3 = {*(const int64_t *)(vSigns64 + ((vQ2[6] >> 9))),
                                          *(const int64_t *)(vSigns64 + ((vQ2[7] >> 9)))};
      vQ2 += 8;

      vector signed char vQ2x0 =
          (vector signed char)vec_mul((vector signed char)vVsigns0, (vector signed char)vAux64x20);
      vector signed char vQ2x1 =
          (vector signed char)vec_mul((vector signed char)vVsigns1, (vector signed char)vAux64x21);
      vector signed char vQ2x2 =
          (vector signed char)vec_mul((vector signed char)vVsigns2, (vector signed char)vAux64x22);
      vector signed char vQ2x3 =
          (vector signed char)vec_mul((vector signed char)vVsigns3, (vector signed char)vAux64x23);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ2x0, vQ8y0), vec_mulo(vQ2x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ2x1, vQ8y1), vec_mulo(vQ2x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ2x2, vQ8y2), vec_mulo(vQ2x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ2x3, vQ8y3), vec_mulo(vQ2x3, vQ8y3));

      const uint16_t cLs0 = (uint16_t)(vSc[0] & 0xf);
      const uint16_t cLs1 = (uint16_t)(vSc[0] >> 4);
      const uint16_t cLs2 = (uint16_t)(vSc[1] & 0xf);
      const uint16_t cLs3 = (uint16_t)(vSc[1] >> 4);
      vSc += 2;

      vector signed short vVscales0 = vec_splats((int16_t)(2 * cLs0 + 1));
      vector signed short vVscales1 = vec_splats((int16_t)(2 * cLs1 + 1));
      vector signed short vVscales2 = vec_splats((int16_t)(2 * cLs2 + 1));
      vector signed short vVscales3 = vec_splats((int16_t)(2 * cLs3 + 1));

      vVsumi0 = vec_msum(vQv0, vVscales0, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales1, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales2, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales3, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = 0.125f * vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  static const uint8_t aKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t aKMask2[16] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const vector int cV0 = vec_splats((int32_t)0);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  const vector unsigned char cMask0 = vec_xl(0, aKMask1);
  const vector unsigned char cMask1 = vec_xl(16, aKMask1);
  const vector signed char cMask2 = (vector signed char)vec_xl(0, aKMask2);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].qs + QK_K / 8);
    const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    for (int vJ = 0; vJ < QK_K / 32; vJ += 2) {
      __builtin_prefetch(vQ2, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed long long vAux64x20 = {*(const int64_t *)(iq2s_grid + (vQ2[0] | ((vQh[0] << 8) & 0x300))),
                                           *(const int64_t *)(iq2s_grid + (vQ2[1] | ((vQh[0] << 6) & 0x300)))};
      vector signed long long vAux64x21 = {*(const int64_t *)(iq2s_grid + (vQ2[2] | ((vQh[0] << 4) & 0x300))),
                                           *(const int64_t *)(iq2s_grid + (vQ2[3] | ((vQh[0] << 2) & 0x300)))};
      vector signed long long vAux64x22 = {*(const int64_t *)(iq2s_grid + (vQ2[4] | ((vQh[1] << 8) & 0x300))),
                                           *(const int64_t *)(iq2s_grid + (vQ2[5] | ((vQh[1] << 6) & 0x300)))};
      vector signed long long vAux64x23 = {*(const int64_t *)(iq2s_grid + (vQ2[6] | ((vQh[1] << 4) & 0x300))),
                                           *(const int64_t *)(iq2s_grid + (vQ2[7] | ((vQh[1] << 2) & 0x300)))};
      vQ2 += 8;
      vQh += 2;

      vector signed char vVsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&vSigns[0]);
      vector signed char vVsigns23 = (vector signed char)vec_splats(*(const uint32_t *)&vSigns[2]);
      vSigns += 4;

      vector signed char vVsigns0 = vec_perm(vVsigns01, vVsigns01, cMask0);
      vector signed char vVsigns1 = vec_perm(vVsigns01, vVsigns01, cMask1);
      vector signed char vVsigns2 = vec_perm(vVsigns23, vVsigns23, cMask0);
      vector signed char vVsigns3 = vec_perm(vVsigns23, vVsigns23, cMask1);

      vVsigns0 = (vector signed char)vec_cmpeq(vec_and(vVsigns0, cMask2), cMask2);
      vVsigns1 = (vector signed char)vec_cmpeq(vec_and(vVsigns1, cMask2), cMask2);
      vVsigns2 = (vector signed char)vec_cmpeq(vec_and(vVsigns2, cMask2), cMask2);
      vVsigns3 = (vector signed char)vec_cmpeq(vec_and(vVsigns3, cMask2), cMask2);

      vector signed char vQ2x0 = vec_sub(vec_xor(vVsigns0, (vector signed char)vAux64x20), vVsigns0);
      vector signed char vQ2x1 = vec_sub(vec_xor(vVsigns1, (vector signed char)vAux64x21), vVsigns1);
      vector signed char vQ2x2 = vec_sub(vec_xor(vVsigns2, (vector signed char)vAux64x22), vVsigns2);
      vector signed char vQ2x3 = vec_sub(vec_xor(vVsigns3, (vector signed char)vAux64x23), vVsigns3);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ2x0, vQ8y0), vec_mulo(vQ2x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ2x1, vQ8y1), vec_mulo(vQ2x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ2x2, vQ8y2), vec_mulo(vQ2x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ2x3, vQ8y3), vec_mulo(vQ2x3, vQ8y3));

      const uint16_t cLs0 = (uint16_t)(vSc[0] & 0xf);
      const uint16_t cLs1 = (uint16_t)(vSc[0] >> 4);
      const uint16_t cLs2 = (uint16_t)(vSc[1] & 0xf);
      const uint16_t cLs3 = (uint16_t)(vSc[1] >> 4);
      vSc += 2;

      vector signed short vVscales0 = vec_splats((int16_t)(2 * cLs0 + 1));
      vector signed short vVscales1 = vec_splats((int16_t)(2 * cLs1 + 1));
      vector signed short vVscales2 = vec_splats((int16_t)(2 * cLs2 + 1));
      vector signed short vVscales3 = vec_splats((int16_t)(2 * cLs3 + 1));

      vVsumi0 = vec_msum(vQv0, vVscales0, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales1, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales2, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales3, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = 0.125f * vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  const vector int cV0 = vec_splats((int32_t)0);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint32_t *GGML_RESTRICT vSigns = (const uint32_t *)(vX[vI].qs + QK_K / 4);
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

#pragma GCC unroll 1
    for (int vJ = 0; vJ < QK_K / 32; vJ += 2) {
      __builtin_prefetch(vQ3, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector unsigned int vAux32x40 = {iq3xxs_grid[vQ3[0]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[2]],
                                       iq3xxs_grid[vQ3[3]]};
      vector unsigned int vAux32x41 = {iq3xxs_grid[vQ3[4]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[6]],
                                       iq3xxs_grid[vQ3[7]]};
      vector unsigned int vAux32x42 = {iq3xxs_grid[vQ3[8]], iq3xxs_grid[vQ3[9]], iq3xxs_grid[vQ3[10]],
                                       iq3xxs_grid[vQ3[11]]};
      vector unsigned int vAux32x43 = {iq3xxs_grid[vQ3[12]], iq3xxs_grid[vQ3[13]], iq3xxs_grid[vQ3[14]],
                                       iq3xxs_grid[vQ3[15]]};
      vQ3 += 16;

      vector unsigned long long vAux64x20 = {(uint64_t)(vSigns64[(vSigns[0] >> 0) & 127]),
                                             (uint64_t)(vSigns64[(vSigns[0] >> 7) & 127])};
      vector unsigned long long vAux64x21 = {(uint64_t)(vSigns64[(vSigns[0] >> 14) & 127]),
                                             (uint64_t)(vSigns64[(vSigns[0] >> 21) & 127])};
      vector unsigned long long vAux64x22 = {(uint64_t)(vSigns64[(vSigns[1] >> 0) & 127]),
                                             (uint64_t)(vSigns64[(vSigns[1] >> 7) & 127])};
      vector unsigned long long vAux64x23 = {(uint64_t)(vSigns64[(vSigns[1] >> 14) & 127]),
                                             (uint64_t)(vSigns64[(vSigns[1] >> 21) & 127])};

      vector signed char vQ3x0 = vec_mul((vector signed char)vAux64x20, (vector signed char)vAux32x40);
      vector signed char vQ3x1 = vec_mul((vector signed char)vAux64x21, (vector signed char)vAux32x41);
      vector signed char vQ3x2 = vec_mul((vector signed char)vAux64x22, (vector signed char)vAux32x42);
      vector signed char vQ3x3 = vec_mul((vector signed char)vAux64x23, (vector signed char)vAux32x43);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ3x0, vQ8y0), vec_mulo(vQ3x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ3x1, vQ8y1), vec_mulo(vQ3x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ3x2, vQ8y2), vec_mulo(vQ3x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ3x3, vQ8y3), vec_mulo(vQ3x3, vQ8y3));

      const uint16_t cLs0 = (uint16_t)(vSigns[0] >> 28);
      const uint16_t cLs1 = (uint16_t)(vSigns[1] >> 28);
      vSigns += 2;

      vector signed short vVscales01 = (vector signed short)vec_splats((uint16_t)(2 * cLs0 + 1));
      vector signed short vVscales23 = (vector signed short)vec_splats((uint16_t)(2 * cLs1 + 1));

      vVsumi0 = vec_msum(vQv0, vVscales01, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales01, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales23, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales23, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = 0.25f * vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  static const uint8_t aKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t aKMask2[16] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const vector int cV0 = vec_splats((int32_t)0);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  const vector unsigned char cMask0 = vec_xl(0, aKMask1);
  const vector unsigned char cMask1 = vec_xl(16, aKMask1);
  const vector signed char cMask2 = (vector signed char)vec_xl(0, aKMask2);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].signs);
    const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    for (int vJ = 0; vJ < QK_K / 32; vJ += 2) {
      __builtin_prefetch(vQ3, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector unsigned int vAux32x40 = {
          iq3s_grid[vQ3[0] | ((vQh[0] << 8) & 256)], iq3s_grid[vQ3[1] | ((vQh[0] << 7) & 256)],
          iq3s_grid[vQ3[2] | ((vQh[0] << 6) & 256)], iq3s_grid[vQ3[3] | ((vQh[0] << 5) & 256)]};
      vector unsigned int vAux32x41 = {
          iq3s_grid[vQ3[4] | ((vQh[0] << 4) & 256)], iq3s_grid[vQ3[5] | ((vQh[0] << 3) & 256)],
          iq3s_grid[vQ3[6] | ((vQh[0] << 2) & 256)], iq3s_grid[vQ3[7] | ((vQh[0] << 1) & 256)]};
      vector unsigned int vAux32x42 = {
          iq3s_grid[vQ3[8] | ((vQh[1] << 8) & 256)], iq3s_grid[vQ3[9] | ((vQh[1] << 7) & 256)],
          iq3s_grid[vQ3[10] | ((vQh[1] << 6) & 256)], iq3s_grid[vQ3[11] | ((vQh[1] << 5) & 256)]};
      vector unsigned int vAux32x43 = {
          iq3s_grid[vQ3[12] | ((vQh[1] << 4) & 256)], iq3s_grid[vQ3[13] | ((vQh[1] << 3) & 256)],
          iq3s_grid[vQ3[14] | ((vQh[1] << 2) & 256)], iq3s_grid[vQ3[15] | ((vQh[1] << 1) & 256)]};
      vQ3 += 16;
      vQh += 2;

      vector signed char vVsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&vSigns[0]);
      vector signed char vVsigns02 = (vector signed char)vec_splats(*(const uint32_t *)&vSigns[2]);
      vSigns += 4;

      vector signed char vVsigns0 = vec_perm(vVsigns01, vVsigns01, cMask0);
      vector signed char vVsigns1 = vec_perm(vVsigns01, vVsigns01, cMask1);
      vector signed char vVsigns2 = vec_perm(vVsigns02, vVsigns02, cMask0);
      vector signed char vVsigns3 = vec_perm(vVsigns02, vVsigns02, cMask1);

      vVsigns0 = (vector signed char)vec_cmpeq(vec_and(vVsigns0, cMask2), cMask2);
      vVsigns1 = (vector signed char)vec_cmpeq(vec_and(vVsigns1, cMask2), cMask2);
      vVsigns2 = (vector signed char)vec_cmpeq(vec_and(vVsigns2, cMask2), cMask2);
      vVsigns3 = (vector signed char)vec_cmpeq(vec_and(vVsigns3, cMask2), cMask2);

      vector signed char vQ3x0 = vec_sub(vec_xor(vVsigns0, (vector signed char)vAux32x40), vVsigns0);
      vector signed char vQ3x1 = vec_sub(vec_xor(vVsigns1, (vector signed char)vAux32x41), vVsigns1);
      vector signed char vQ3x2 = vec_sub(vec_xor(vVsigns2, (vector signed char)vAux32x42), vVsigns2);
      vector signed char vQ3x3 = vec_sub(vec_xor(vVsigns3, (vector signed char)vAux32x43), vVsigns3);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ3x0, vQ8y0), vec_mulo(vQ3x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ3x1, vQ8y1), vec_mulo(vQ3x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ3x2, vQ8y2), vec_mulo(vQ3x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ3x3, vQ8y3), vec_mulo(vQ3x3, vQ8y3));

      const uint16_t cLs0 = (uint16_t)(vSc[0] & 0xf);
      const uint16_t cLs1 = (uint16_t)(vSc[0] >> 4);
      vSc++;

      vector signed short vVscales01 = (vector signed short)vec_splats((uint16_t)(2 * cLs0 + 1));
      vector signed short vVscales23 = (vector signed short)vec_splats((uint16_t)(2 * cLs1 + 1));

      vVsumi0 = vec_msum(vQv0, vVscales01, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales01, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales23, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales23, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector unsigned char cV0 = vec_splats((unsigned char)0x0);
  const vector unsigned short cVsign = vec_splats((unsigned short)0x8000);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  for (int vI = 0; vI < cNb; ++vI) {
    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vI].d));
    vector float vVyd = vec_splats(vY[vI].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = vec_splats((int32_t)0);
    vector signed int vVsumi1 = vec_splats((int32_t)0);
    vector signed int vVsumi2 = vec_splats((int32_t)0);
    vector signed int vVsumi3 = vec_splats((int32_t)0);
    vector signed int vVsumi8 = vec_splats((int32_t)0);

    const uint8_t *GGML_RESTRICT vQ1 = vX[vI].qs;
    const uint16_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    const int16_t *GGML_RESTRICT vQs = vY[vI].bsums;

    for (int vJ = 0; vJ < QK_K / 32; vJ += 2) {
      __builtin_prefetch(vQ1, 0, 1);
      __builtin_prefetch(vQh, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed long long vAux64x20 = {*(const int64_t *)(iq1s_grid + (vQ1[0] | ((vQh[0] << 8) & 0x700))),
                                           *(const int64_t *)(iq1s_grid + (vQ1[1] | ((vQh[0] << 5) & 0x700)))};
      vector signed long long vAux64x21 = {*(const int64_t *)(iq1s_grid + (vQ1[2] | ((vQh[0] << 2) & 0x700))),
                                           *(const int64_t *)(iq1s_grid + (vQ1[3] | ((vQh[0] >> 1) & 0x700)))};
      vector signed long long vAux64x22 = {*(const int64_t *)(iq1s_grid + (vQ1[4] | ((vQh[1] << 8) & 0x700))),
                                           *(const int64_t *)(iq1s_grid + (vQ1[5] | ((vQh[1] << 5) & 0x700)))};
      vector signed long long vAux64x23 = {*(const int64_t *)(iq1s_grid + (vQ1[6] | ((vQh[1] << 2) & 0x700))),
                                           *(const int64_t *)(iq1s_grid + (vQ1[7] | ((vQh[1] >> 1) & 0x700)))};
      vQ1 += 8;

      vector signed char vQ1x0 = (vector signed char)vAux64x20;
      vector signed char vQ1x1 = (vector signed char)vAux64x21;
      vector signed char vQ1x2 = (vector signed char)vAux64x22;
      vector signed char vQ1x3 = (vector signed char)vAux64x23;

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ1x0, vQ8y0), vec_mulo(vQ1x0, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ1x1, vQ8y1), vec_mulo(vQ1x1, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ1x2, vQ8y2), vec_mulo(vQ1x2, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ1x3, vQ8y3), vec_mulo(vQ1x3, vQ8y3));

      const uint16_t cLs0 = (uint16_t)((vQh[0] >> 12) & 7);
      const uint16_t cLs1 = (uint16_t)((vQh[1] >> 12) & 7);

      vector signed short vVscales01 = (vector signed short)vec_splats((uint16_t)(2 * cLs0 + 1));
      vector signed short vVscales23 = (vector signed short)vec_splats((uint16_t)(2 * cLs1 + 1));
      vector signed short vVscales = vec_sld(vVscales23, vVscales01, 8);

      vVsumi0 = vec_msum(vQv0, vVscales01, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales01, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales23, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales23, vVsumi3);

      vector signed short vQ8ysums = vec_xl_len(vQs, 8);
      vQs += 4;
      vQ8ysums = vec_mergeh(vQ8ysums, (vector signed short)cV0);

      vector signed short vQxh = (vector signed short)vec_sld(vec_splats(vQh[1]), vec_splats(vQh[0]), 8);
      vQh += 2;
      vector __bool short vVsel = vec_cmpge(vQxh, (vector signed short)cV0);

      vector signed short vQ8ysum =
          vec_sel((vector signed short)vec_xor((vector unsigned short)vQ8ysums, cVsign), vQ8ysums, vVsel);

      vVsumi8 = vec_add(vec_mule(vQ8ysum, vVscales), vVsumi8);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);

    vVsumf0 = vec_madd(vec_ctf(vVsumi8, 0), vec_mul(vVd, vec_splats(IQ1S_DELTA)), vVsumf0);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq1_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector signed int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);

  const vector signed char cValues = vec_xl(0, kvalues_iq4nl);

#pragma GCC unroll 4
  for (; vIb < cNb; ++vIb) {
    __builtin_prefetch(vX[vIb].qs, 0, 1);
    __builtin_prefetch(vY[vIb].qs, 0, 1);

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIb].d));
    vector float vVyd = vec_splats(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed char vQxs = (vector signed char)vec_xl(0, vX[vIb].qs);
    vector signed char vQ4x0 = vec_and(vQxs, cLowMask);
    vector signed char vQ4x1 = vec_sr(vQxs, cV4);

    vQ4x0 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x0);
    vQ4x1 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x1);

    vector signed char vQ8y0 = vec_xl(0, vY[vIb].qs);
    vector signed char vQ8y1 = vec_xl(16, vY[vIb].qs);

    vector signed short vQv0 = vec_add(vec_mule(vQ4x0, vQ8y0), vec_mulo(vQ4x0, vQ8y0));
    vector signed short vQv1 = vec_add(vec_mule(vQ4x1, vQ8y1), vec_mulo(vQ4x1, vQ8y1));

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;

    vVsumi0 = vec_sum4s(vQv0, vVsumi0);
    vVsumi1 = vec_sum4s(vQv1, vVsumi1);

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  vSumf = vec_extract(vVsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
  const vector signed char cLowMask = vec_splats((signed char)0xF);
  const vector int cV0 = vec_splats((int32_t)0);
  const vector unsigned char cV4 = vec_splats((unsigned char)0x4);

  vector float vVsumf0 = vec_splats(0.0f);
  vector float vVsumf1 = vec_splats(0.0f);
  vector float vVsumf2 = vec_splats(0.0f);
  vector float vVsumf3 = vec_splats(0.0f);

  const vector signed char cValues = vec_xl(0, kvalues_iq4nl);

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {

    vector float vVxd = vec_splats(GGML_CPU_FP16_TO_FP32(vX[vIbl].d));
    vector float vVyd = vec_splats(vY[vIbl].d);
    vector float vVd = vec_mul(vVxd, vVyd);

    vector signed int vVsumi0 = cV0;
    vector signed int vVsumi1 = cV0;
    vector signed int vVsumi2 = cV0;
    vector signed int vVsumi3 = cV0;

    uint16_t vH = vX[vIbl].scales_h;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vIbl].qs;
    const uint8_t *GGML_RESTRICT vSc = vX[vIbl].scales_l;
    const int8_t *GGML_RESTRICT vQ8 = vY[vIbl].qs;

    for (int vIb = 0; vIb < QK_K / 64; vIb++) {
      __builtin_prefetch(vQ4, 0, 1);
      __builtin_prefetch(vQ8, 0, 1);

      vector signed char vQxs0 = (vector signed char)vec_xl(0, vQ4);
      vector signed char vQxs1 = (vector signed char)vec_xl(16, vQ4);
      vQ4 += 32;

      vector signed char vQ4x00 = (vector signed char)vec_and(vQxs0, cLowMask);
      vector signed char vQ4x01 = (vector signed char)vec_sr(vQxs0, cV4);
      vector signed char vQ4x10 = (vector signed char)vec_and(vQxs1, cLowMask);
      vector signed char vQ4x11 = (vector signed char)vec_sr(vQxs1, cV4);

      vQ4x00 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x00);
      vQ4x01 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x01);
      vQ4x10 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x10);
      vQ4x11 = vec_perm(cValues, cValues, (vector unsigned char)vQ4x11);

      vector signed char vQ8y0 = vec_xl(0, vQ8);
      vector signed char vQ8y1 = vec_xl(16, vQ8);
      vector signed char vQ8y2 = vec_xl(32, vQ8);
      vector signed char vQ8y3 = vec_xl(48, vQ8);
      vQ8 += 64;

      vector signed short vQv0 = vec_add(vec_mule(vQ4x00, vQ8y0), vec_mulo(vQ4x00, vQ8y0));
      vector signed short vQv1 = vec_add(vec_mule(vQ4x01, vQ8y1), vec_mulo(vQ4x01, vQ8y1));
      vector signed short vQv2 = vec_add(vec_mule(vQ4x10, vQ8y2), vec_mulo(vQ4x10, vQ8y2));
      vector signed short vQv3 = vec_add(vec_mule(vQ4x11, vQ8y3), vec_mulo(vQ4x11, vQ8y3));

      const uint16_t cLs0 = (uint16_t)(((vSc[0] & 0xf) | ((vH << 4) & 0x30)) - 32);
      const uint16_t cLs1 = (uint16_t)(((vSc[0] >> 4) | ((vH << 2) & 0x30)) - 32);
      vH >>= 4;
      vSc++;

      vector signed short vVscales01 = vec_splats((int16_t)cLs0);
      vector signed short vVscales23 = vec_splats((int16_t)cLs1);

      vVsumi0 = vec_msum(vQv0, vVscales01, vVsumi0);
      vVsumi1 = vec_msum(vQv1, vVscales01, vVsumi1);
      vVsumi2 = vec_msum(vQv2, vVscales23, vVsumi2);
      vVsumi3 = vec_msum(vQv3, vVscales23, vVsumi3);
    }

    vVsumf0 = vec_madd(vec_ctf(vVsumi0, 0), vVd, vVsumf0);
    vVsumf1 = vec_madd(vec_ctf(vVsumi1, 0), vVd, vVsumf1);
    vVsumf2 = vec_madd(vec_ctf(vVsumi2, 0), vVd, vVsumf2);
    vVsumf3 = vec_madd(vec_ctf(vVsumi3, 0), vVd, vVsumf3);
  }

  vVsumf0 = vec_add(vVsumf0, vVsumf2);
  vVsumf1 = vec_add(vVsumf1, vVsumf3);

  vVsumf0 = vec_add(vVsumf0, vVsumf1);

  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 4));
  vVsumf0 = vec_add(vVsumf0, vec_sld(vVsumf0, vVsumf0, 8));

  *pS = vec_extract(vVsumf0, 0);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
