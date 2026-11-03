// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double ggml_float;

#if defined(__ARM_FEATURE_SVE)
inline static void ggml_sve_f16_fma_widened(svfloat32_t *acc_lo, svfloat32_t *acc_hi, svfloat16_t x, svfloat16_t y) {
#if defined(__ARM_FEATURE_SVE2)
  *acc_lo = svmlalb_f32(*acc_lo, x, y);
  *acc_hi = svmlalt_f32(*acc_hi, x, y);
#else
  // Plain SVE fallback path if SVE2 instructions not available
  svfloat16_t x_even = svtrn1_f16(x, x);
  svfloat16_t x_odd = svtrn2_f16(x, x);

  svfloat16_t y_even = svtrn1_f16(y, y);
  svfloat16_t y_odd = svtrn2_f16(y, y);

  svbool_t pg = svptrue_b32();

  *acc_lo = svmla_f32_x(pg, *acc_lo, svcvt_f32_f16_x(pg, x_even), svcvt_f32_f16_x(pg, y_even));
  *acc_hi = svmla_f32_x(pg, *acc_hi, svcvt_f32_f16_x(pg, x_odd), svcvt_f32_f16_x(pg, y_odd));
#endif
}

inline static ggml_float ggml_sve_sum_f32x2(svfloat32_t sum_lo, svfloat32_t sum_hi) {
  return (ggml_float)(svaddv_f32(svptrue_b32(), sum_lo) + svaddv_f32(svptrue_b32(), sum_hi));
}
#endif

#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL 2
#define GGML_VEC_MAD_UNROLL 32

#ifdef __cplusplus
extern "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void ggml_vec_dot_f32(int pN, float *GGML_RESTRICT pS, size_t pBs, const float *GGML_RESTRICT pX, size_t pBx,
                      const float *GGML_RESTRICT pY, size_t pBy, int pNrc);
void ggml_vec_dot_bf16(int pN, float *GGML_RESTRICT pS, size_t pBs, ggml_bf16_t *GGML_RESTRICT pX, size_t pBx,
                       ggml_bf16_t *GGML_RESTRICT pY, size_t pBy, int pNrc);
void ggml_vec_dot_f16(int pN, float *GGML_RESTRICT pS, size_t pBs, ggml_fp16_t *GGML_RESTRICT pX, size_t pBx,
                      ggml_fp16_t *GGML_RESTRICT pY, size_t pBy, int pNrc);

void ggml_vec_silu_f32(const int pN, float *pY, const float *pX);
ggml_float ggml_vec_cvar_f32(const int pN, float *pY, const float *pX,
                             const float pMean); // it will also center y ( y = y - mean )
ggml_float ggml_vec_soft_max_f32(const int pN, float *pY, const float *pX, float pMax);
ggml_float ggml_vec_log_soft_max_f32(const int pN, float *pY, const float *pX, float pMax);

inline static void ggml_vec_set_i8(const int pN, int8_t *pX, const int8_t pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}
inline static void ggml_vec_set_i16(const int pN, int16_t *pX, const int16_t pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}

inline static void ggml_vec_set_i32(const int pN, int32_t *pX, const int32_t pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}
inline static void ggml_vec_cpy_i32(const int pN, int32_t *pY, const int32_t *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = pX[vI];
}

inline static void ggml_vec_set_f16(const int pN, ggml_fp16_t *pX, const ggml_fp16_t pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}
inline static void ggml_vec_set_bf16(const int pN, ggml_bf16_t *pX, const ggml_bf16_t pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}

inline static void ggml_vec_add_f32(const int pN, float *pZ, const float *pX, const float *pY) {
  int vI = 0;
#if defined(__AVX2__)
  for (; vI + 7 < pN; vI += 8) {
    __m256 vVx = _mm256_loadu_ps(pX + vI);
    __m256 vVy = _mm256_loadu_ps(pY + vI);
    __m256 vVz = _mm256_add_ps(vVx, vVy);
    _mm256_storeu_ps(pZ + vI, vVz);
  }
#endif
  for (; vI < pN; ++vI) {
    pZ[vI] = pX[vI] + pY[vI];
  }
}

inline static void ggml_vec_add_f16(const int pN, ggml_fp16_t *pZ, const ggml_fp16_t *pX, const ggml_fp16_t *pY) {
  for (int vI = 0; vI < pN; ++vI) {
    pZ[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pX[vI]) + GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
}
inline static void ggml_vec_add1_f32(const int pN, float *pZ, const float *pX, const float pV) {
  for (int vI = 0; vI < pN; ++vI)
    pZ[vI] = pX[vI] + pV;
}
inline static void ggml_vec_acc_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] += pX[vI];
}
inline static void ggml_vec_acc1_f32(const int pN, float *pY, const float pV) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] += pV;
}
inline static void ggml_vec_sub_f32(const int pN, float *pZ, const float *pX, const float *pY) {
  for (int vI = 0; vI < pN; ++vI)
    pZ[vI] = pX[vI] - pY[vI];
}
inline static void ggml_vec_sub_f16(const int pN, ggml_fp16_t *pZ, const ggml_fp16_t *pX, const ggml_fp16_t *pY) {
  for (int vI = 0; vI < pN; ++vI) {
    pZ[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pX[vI]) - GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
}
inline static void ggml_vec_set_f32(const int pN, float *pX, const float pV) {
  for (int vI = 0; vI < pN; ++vI)
    pX[vI] = pV;
}
inline static void ggml_vec_cpy_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = pX[vI];
}
inline static void ggml_vec_neg_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = -pX[vI];
}
inline static void ggml_vec_neg_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(pX[vI]));
  }
}

inline static void ggml_vec_mul_f32(const int pN, float *pZ, const float *pX, const float *pY) {
  for (int vI = 0; vI < pN; ++vI)
    pZ[vI] = pX[vI] * pY[vI];
}
inline static void ggml_vec_mul_f16(const int pN, ggml_fp16_t *pZ, const ggml_fp16_t *pX, const ggml_fp16_t *pY) {
  for (int vI = 0; vI < pN; ++vI) {
    pZ[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pX[vI]) * GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
}
inline static void ggml_vec_div_f32(const int pN, float *pZ, const float *pX, const float *pY) {
  for (int vI = 0; vI < pN; ++vI)
    pZ[vI] = pX[vI] / pY[vI];
}
inline static void ggml_vec_div_f16(const int pN, ggml_fp16_t *pZ, const ggml_fp16_t *pX, const ggml_fp16_t *pY) {
  for (int vI = 0; vI < pN; ++vI) {
    pZ[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pX[vI]) / GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int pN, const int pXs, float *GGML_RESTRICT pS,
                                           void *GGML_RESTRICT pXv, ggml_fp16_t *GGML_RESTRICT pY) {
  ggml_float vSumf[GGML_VEC_DOT_UNROLL] = {0.0};

  ggml_fp16_t *GGML_RESTRICT vX[GGML_VEC_DOT_UNROLL];

  for (int vI = 0; vI < GGML_VEC_DOT_UNROLL; ++vI) {
    vX[vI] = (ggml_fp16_t *)((char *)pXv + vI * pXs);
  }

#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)

  const int ggml_f16_epr = svcnth();
  const int ggml_f16_step = 2 * ggml_f16_epr;
  int cNp = pN - (pN % ggml_f16_step);
  int np2 = pN - (pN % ggml_f16_epr);

  svfloat32_t sum_0_0_lo = svdup_n_f32(0.0f);
  svfloat32_t sum_0_0_hi = svdup_n_f32(0.0f);
  svfloat32_t sum_0_1_lo = svdup_n_f32(0.0f);
  svfloat32_t sum_0_1_hi = svdup_n_f32(0.0f);
  svfloat32_t sum_1_0_lo = svdup_n_f32(0.0f);
  svfloat32_t sum_1_0_hi = svdup_n_f32(0.0f);
  svfloat32_t sum_1_1_lo = svdup_n_f32(0.0f);
  svfloat32_t sum_1_1_hi = svdup_n_f32(0.0f);

  for (int vI = 0; vI < cNp; vI += ggml_f16_step) {
    const svfloat16_t ay0 = GGML_F16x_VEC_LOAD(pY + vI, 0);
    const svfloat16_t ax00 = GGML_F16x_VEC_LOAD(vX[0] + vI, 0);
    const svfloat16_t ax01 = GGML_F16x_VEC_LOAD(vX[1] + vI, 0);

    ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, ax00, ay0);
    ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, ax01, ay0);

    const svfloat16_t ay1 = GGML_F16x_VEC_LOAD(pY + vI + 1 * ggml_f16_epr, 0);
    const svfloat16_t ax10 = GGML_F16x_VEC_LOAD(vX[0] + vI + 1 * ggml_f16_epr, 0);
    const svfloat16_t ax11 = GGML_F16x_VEC_LOAD(vX[1] + vI + 1 * ggml_f16_epr, 0);

    ggml_sve_f16_fma_widened(&sum_0_1_lo, &sum_0_1_hi, ax10, ay1);
    ggml_sve_f16_fma_widened(&sum_1_1_lo, &sum_1_1_hi, ax11, ay1);
  }

  for (int vI = cNp; vI < np2; vI += ggml_f16_epr) {
    const svfloat16_t ry = GGML_F16x_VEC_LOAD(pY + vI, 0);
    const svfloat16_t rx0 = GGML_F16x_VEC_LOAD(vX[0] + vI, 0);
    const svfloat16_t rx1 = GGML_F16x_VEC_LOAD(vX[1] + vI, 0);

    ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, rx0, ry);
    ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, rx1, ry);
  }

  if (np2 < pN) {
    const svbool_t pg = svwhilelt_b16(np2, pN);
    const svfloat16_t vAy = svld1_f16(pg, (const __fp16 *)(pY + np2));
    const svfloat16_t ax0 = svld1_f16(pg, (const __fp16 *)(vX[0] + np2));
    const svfloat16_t ax1 = svld1_f16(pg, (const __fp16 *)(vX[1] + np2));

    ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, ax0, vAy);
    ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, ax1, vAy);
  }

  svfloat32_t sum_0_lo = svadd_f32_x(DEFAULT_PG32, sum_0_0_lo, sum_0_1_lo);
  svfloat32_t sum_0_hi = svadd_f32_x(DEFAULT_PG32, sum_0_0_hi, sum_0_1_hi);
  svfloat32_t sum_1_lo = svadd_f32_x(DEFAULT_PG32, sum_1_0_lo, sum_1_1_lo);
  svfloat32_t sum_1_hi = svadd_f32_x(DEFAULT_PG32, sum_1_0_hi, sum_1_1_hi);
  vSumf[0] = ggml_sve_sum_f32x2(sum_0_lo, sum_0_hi);
  vSumf[1] = ggml_sve_sum_f32x2(sum_1_lo, sum_1_hi);
  cNp = pN;
#elif defined(__riscv_v_intrinsic)
#if defined(__riscv_zvfh)
  size_t vl = __riscv_vsetvlmax_e32m4();

  // initialize accumulators to all zeroes
  vfloat32m4_t vsum0_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t vsum0_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t vsum1_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t vsum1_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

  // calculate step size
  const size_t epr = __riscv_vsetvlmax_e16m2();
  const size_t step = epr * 2;
  int cNp = (pN & ~(step - 1));

  // unroll by 2 along the row dimension
  for (int vI = 0; vI < cNp; vI += step) {
    vfloat16m2_t ay0 = __riscv_vle16_v_f16m2((const _Float16 *)(pY + vI), epr);
    vfloat16m2_t ax0_0 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[0] + vI), epr);
    vfloat16m2_t ax1_0 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[1] + vI), epr);
    vsum0_0 = __riscv_vfwmacc_vv_f32m4(vsum0_0, ax0_0, ay0, epr);
    vsum1_0 = __riscv_vfwmacc_vv_f32m4(vsum1_0, ax1_0, ay0, epr);

    vfloat16m2_t ay1 = __riscv_vle16_v_f16m2((const _Float16 *)(pY + vI + epr), epr);
    vfloat16m2_t ax0_1 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[0] + vI + epr), epr);
    vfloat16m2_t ax1_1 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[1] + vI + epr), epr);
    vsum0_1 = __riscv_vfwmacc_vv_f32m4(vsum0_1, ax0_1, ay1, epr);
    vsum1_1 = __riscv_vfwmacc_vv_f32m4(vsum1_1, ax1_1, ay1, epr);
  }

  vfloat32m4_t vsum0 = __riscv_vfadd_vv_f32m4(vsum0_0, vsum0_1, vl);
  vfloat32m4_t vsum1 = __riscv_vfadd_vv_f32m4(vsum1_0, vsum1_1, vl);

  // leftovers
  for (int vI = cNp; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m2(pN - vI);
    vfloat16m2_t vAy = __riscv_vle16_v_f16m2((const _Float16 *)(pY + vI), vl);
    vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[0] + vI), vl);
    vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16 *)(vX[1] + vI), vl);

    vsum0 = __riscv_vfwmacc_vv_f32m4(vsum0, ax0, vAy, vl);
    vsum1 = __riscv_vfwmacc_vv_f32m4(vsum1, ax1, vAy, vl);
  }

  // reduce
  vl = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t acc0_0 =
      __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum0, 0), __riscv_vget_v_f32m4_f32m2(vsum0, 1), vl);
  vl = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t acc0_1 =
      __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc0_0, 0), __riscv_vget_v_f32m2_f32m1(acc0_0, 1), vl);
  vfloat32m1_t redsum0 = __riscv_vfredusum_vs_f32m1_f32m1(acc0_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);

  vl = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t acc1_0 =
      __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum1, 0), __riscv_vget_v_f32m4_f32m2(vsum1, 1), vl);
  vl = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t acc1_1 =
      __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc1_0, 0), __riscv_vget_v_f32m2_f32m1(acc1_0, 1), vl);
  vfloat32m1_t redsum1 = __riscv_vfredusum_vs_f32m1_f32m1(acc1_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
  vSumf[0] = __riscv_vfmv_f_s_f32m1_f32(redsum0);
  vSumf[1] = __riscv_vfmv_f_s_f32m1_f32(redsum1);
  cNp = pN;
#else
  const int cNp = 0;
#endif
#else
  const int cNp = (pN & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vSum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = {{GGML_F16_VEC_ZERO}};

  GGML_F16_VEC vAx[GGML_F16_ARR];
  GGML_F16_VEC vAy[GGML_F16_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F16_STEP) {
    for (int vJ = 0; vJ < GGML_F16_ARR; vJ++) {
      vAy[vJ] = GGML_F16_VEC_LOAD(pY + vI + vJ * GGML_F16_EPR, vJ);

      for (int vK = 0; vK < GGML_VEC_DOT_UNROLL; ++vK) {
        vAx[vJ] = GGML_F16_VEC_LOAD(vX[vK] + vI + vJ * GGML_F16_EPR, vJ);

        vSum[vK][vJ] = GGML_F16_VEC_FMA(vSum[vK][vJ], vAx[vJ], vAy[vJ]);
      }
    }
  }

  // reduce sum0..sum3 to sum0
  for (int vK = 0; vK < GGML_VEC_DOT_UNROLL; ++vK) {
    GGML_F16_VEC_REDUCE(vSumf[vK], vSum[vK]);
  }
#endif
#else
  // scalar path
  const int cNp = 0;
#endif
  // scalar and leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    for (int vJ = 0; vJ < GGML_VEC_DOT_UNROLL; ++vJ) {
      vSumf[vJ] += (ggml_float)(GGML_CPU_FP16_TO_FP32(vX[vJ][vI]) * GGML_CPU_FP16_TO_FP32(pY[vI]));
    }
  }

  for (int vI = 0; vI < GGML_VEC_DOT_UNROLL; ++vI) {
    pS[vI] = (float)vSumf[vI];
  }
}

inline static void ggml_vec_mad_f32(const int pN, float *GGML_RESTRICT pY, const float *GGML_RESTRICT pX,
                                    const float pV) {
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)

  const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
  const int ggml_f32_epr = sve_register_length / 32; // 8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
  const int ggml_f32_step = 8 * ggml_f32_epr;        // choose 8 SVE registers
  GGML_F32_VEC vVx = GGML_F32_VEC_SET1(pV);

  const int cNp = (pN & ~(ggml_f32_step - 1));
  svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
  svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
  for (int vI = 0; vI < cNp; vI += ggml_f32_step) {

    ax1 = GGML_F32_VEC_LOAD(pX + vI);
    ay1 = GGML_F32_VEC_LOAD(pY + vI);
    ay1 = GGML_F32_VEC_FMA(ay1, ax1, vVx);

    GGML_F32_VEC_STORE(pY + vI, ay1);

    ax2 = GGML_F32_VEC_LOAD(pX + vI + 1 * ggml_f32_epr);
    ay2 = GGML_F32_VEC_LOAD(pY + vI + 1 * ggml_f32_epr);
    ay2 = GGML_F32_VEC_FMA(ay2, ax2, vVx);

    GGML_F32_VEC_STORE(pY + vI + 1 * ggml_f32_epr, ay2);

    ax3 = GGML_F32_VEC_LOAD(pX + vI + 2 * ggml_f32_epr);
    ay3 = GGML_F32_VEC_LOAD(pY + vI + 2 * ggml_f32_epr);
    ay3 = GGML_F32_VEC_FMA(ay3, ax3, vVx);

    GGML_F32_VEC_STORE(pY + vI + 2 * ggml_f32_epr, ay3);

    ax4 = GGML_F32_VEC_LOAD(pX + vI + 3 * ggml_f32_epr);
    ay4 = GGML_F32_VEC_LOAD(pY + vI + 3 * ggml_f32_epr);
    ay4 = GGML_F32_VEC_FMA(ay4, ax4, vVx);

    GGML_F32_VEC_STORE(pY + vI + 3 * ggml_f32_epr, ay4);

    ax5 = GGML_F32_VEC_LOAD(pX + vI + 4 * ggml_f32_epr);
    ay5 = GGML_F32_VEC_LOAD(pY + vI + 4 * ggml_f32_epr);
    ay5 = GGML_F32_VEC_FMA(ay5, ax5, vVx);

    GGML_F32_VEC_STORE(pY + vI + 4 * ggml_f32_epr, ay5);

    ax6 = GGML_F32_VEC_LOAD(pX + vI + 5 * ggml_f32_epr);
    ay6 = GGML_F32_VEC_LOAD(pY + vI + 5 * ggml_f32_epr);
    ay6 = GGML_F32_VEC_FMA(ay6, ax6, vVx);

    GGML_F32_VEC_STORE(pY + vI + 5 * ggml_f32_epr, ay6);

    ax7 = GGML_F32_VEC_LOAD(pX + vI + 6 * ggml_f32_epr);
    ay7 = GGML_F32_VEC_LOAD(pY + vI + 6 * ggml_f32_epr);
    ay7 = GGML_F32_VEC_FMA(ay7, ax7, vVx);

    GGML_F32_VEC_STORE(pY + vI + 6 * ggml_f32_epr, ay7);

    ax8 = GGML_F32_VEC_LOAD(pX + vI + 7 * ggml_f32_epr);
    ay8 = GGML_F32_VEC_LOAD(pY + vI + 7 * ggml_f32_epr);
    ay8 = GGML_F32_VEC_FMA(ay8, ax8, vVx);

    GGML_F32_VEC_STORE(pY + vI + 7 * ggml_f32_epr, ay8);
  }
  // leftovers
  // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
  const int np2 = (pN & ~(ggml_f32_epr - 1));
  for (int vI = cNp; vI < np2; vI += ggml_f32_epr) {
    ax1 = GGML_F32_VEC_LOAD(pX + vI);
    ay1 = GGML_F32_VEC_LOAD(pY + vI);
    ay1 = GGML_F32_VEC_FMA(ay1, ax1, vVx);

    GGML_F32_VEC_STORE(pY + vI, ay1);
  }
  // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements
  // only
  if (np2 < pN) {
    svbool_t pg = svwhilelt_b32(np2, pN);
    ax1 = svld1_f32(pg, pX + np2);
    ay1 = svld1_f32(pg, pY + np2);
    ay1 = svmad_f32_m(pg, ax1, vVx, ay1);

    svst1_f32(pg, pY + np2, ay1);
  }
#elif defined(__riscv_v_intrinsic)
  for (int vI = 0, avl; vI < pN; vI += avl) {
    avl = __riscv_vsetvl_e32m8(pN - vI);
    vfloat32m8_t vAx = __riscv_vle32_v_f32m8(&pX[vI], avl);
    vfloat32m8_t vAy = __riscv_vle32_v_f32m8(&pY[vI], avl);
    vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(vAx, pV, vAy, avl);
    __riscv_vse32_v_f32m8(&pY[vI], ny, avl);
  }
#else
  const int cNp = (pN & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vVx = GGML_F32_VEC_SET1(pV);

  GGML_F32_VEC vAx[GGML_F32_ARR];
  GGML_F32_VEC vAy[GGML_F32_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F32_STEP) {
    for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
      vAx[vJ] = GGML_F32_VEC_LOAD(pX + vI + vJ * GGML_F32_EPR);
      vAy[vJ] = GGML_F32_VEC_LOAD(pY + vI + vJ * GGML_F32_EPR);
      vAy[vJ] = GGML_F32_VEC_FMA(vAy[vJ], vAx[vJ], vVx);

      GGML_F32_VEC_STORE(pY + vI + vJ * GGML_F32_EPR, vAy[vJ]);
    }
  }

  // leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    pY[vI] += pX[vI] * pV;
  }
#endif
#else
  // scalar
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] += pX[vI] * pV;
  }
#endif
}

inline static void ggml_vec_mad_f16(const int pN, ggml_fp16_t *GGML_RESTRICT pY, const ggml_fp16_t *GGML_RESTRICT pX,
                                    const float pV) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
  const int sve_register_length = svcntb() * 8;
  const int ggml_f16_epr = sve_register_length / 16;
  const int ggml_f16_step = 8 * ggml_f16_epr;

  GGML_F16x_VEC vVx = GGML_F16x_VEC_SET1(pV);

  int cNp = (pN & ~(ggml_f16_step - 1));

  svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
  svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
  for (int vI = 0; vI < cNp; vI += ggml_f16_step) {
    ax1 = GGML_F16x_VEC_LOAD(pX + vI + 0 * ggml_f16_epr, 0);
    ay1 = GGML_F16x_VEC_LOAD(pY + vI + 0 * ggml_f16_epr, 0);
    ay1 = GGML_F16x_VEC_FMA(ay1, ax1, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 0 * ggml_f16_epr, ay1, 0);

    ax2 = GGML_F16x_VEC_LOAD(pX + vI + 1 * ggml_f16_epr, 1);
    ay2 = GGML_F16x_VEC_LOAD(pY + vI + 1 * ggml_f16_epr, 1);
    ay2 = GGML_F16x_VEC_FMA(ay2, ax2, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 1 * ggml_f16_epr, ay2, 1);

    ax3 = GGML_F16x_VEC_LOAD(pX + vI + 2 * ggml_f16_epr, 2);
    ay3 = GGML_F16x_VEC_LOAD(pY + vI + 2 * ggml_f16_epr, 2);
    ay3 = GGML_F16x_VEC_FMA(ay3, ax3, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 2 * ggml_f16_epr, ay3, 2);

    ax4 = GGML_F16x_VEC_LOAD(pX + vI + 3 * ggml_f16_epr, 3);
    ay4 = GGML_F16x_VEC_LOAD(pY + vI + 3 * ggml_f16_epr, 3);
    ay4 = GGML_F16x_VEC_FMA(ay4, ax4, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 3 * ggml_f16_epr, ay4, 3);

    ax5 = GGML_F16x_VEC_LOAD(pX + vI + 4 * ggml_f16_epr, 4);
    ay5 = GGML_F16x_VEC_LOAD(pY + vI + 4 * ggml_f16_epr, 4);
    ay5 = GGML_F16x_VEC_FMA(ay5, ax5, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 4 * ggml_f16_epr, ay5, 4);

    ax6 = GGML_F16x_VEC_LOAD(pX + vI + 5 * ggml_f16_epr, 5);
    ay6 = GGML_F16x_VEC_LOAD(pY + vI + 5 * ggml_f16_epr, 5);
    ay6 = GGML_F16x_VEC_FMA(ay6, ax6, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 5 * ggml_f16_epr, ay6, 5);

    ax7 = GGML_F16x_VEC_LOAD(pX + vI + 6 * ggml_f16_epr, 6);
    ay7 = GGML_F16x_VEC_LOAD(pY + vI + 6 * ggml_f16_epr, 6);
    ay7 = GGML_F16x_VEC_FMA(ay7, ax7, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 6 * ggml_f16_epr, ay7, 6);

    ax8 = GGML_F16x_VEC_LOAD(pX + vI + 7 * ggml_f16_epr, 7);
    ay8 = GGML_F16x_VEC_LOAD(pY + vI + 7 * ggml_f16_epr, 7);
    ay8 = GGML_F16x_VEC_FMA(ay8, ax8, vVx);

    GGML_F16x_VEC_STORE(pY + vI + 7 * ggml_f16_epr, ay8, 7);
  }
  const int np2 = (pN & ~(ggml_f16_epr - 1));
  for (int k = cNp; k < np2; k += ggml_f16_epr) {
    svfloat16_t rx = GGML_F16x_VEC_LOAD(pX + k, 0);
    svfloat16_t ry = GGML_F16x_VEC_LOAD(pY + k, 0);
    ry = GGML_F16x_VEC_FMA(ry, rx, vVx);

    GGML_F16x_VEC_STORE(pY + k, ry, 0);
  }

  if (np2 < pN) {
    svbool_t pg = svwhilelt_b16(np2, pN);
    svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(pX + np2));
    svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(pY + np2));
    hy = svmad_f16_x(pg, hx, vVx, hy);
    svst1_f16(pg, (__fp16 *)(pY + np2), hy);
  }
  cNp = pN;
#elif defined(__riscv_v_intrinsic) // implies __riscv_v_intrinsic
#if defined(__riscv_zvfh)
  const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(pV);
  const _Float16 scale = *(const _Float16 *)(&s);

  // calculate step size
  const int epr = __riscv_vsetvlmax_e16m4();
  const int step = epr * 2;
  int cNp = (pN & ~(step - 1));

  // unroll by 2
  for (int vI = 0; vI < cNp; vI += step) {
    vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16 *)pX + vI, epr);
    vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI, epr);
    ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, epr);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI, ay0, epr);
    __asm__ __volatile__("" ::: "memory");

    vfloat16m4_t ax1 = __riscv_vle16_v_f16m4((const _Float16 *)pX + vI + epr, epr);
    vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI + epr, epr);
    ay1 = __riscv_vfmacc_vf_f16m4(ay1, scale, ax1, epr);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI + epr, ay1, epr);
    __asm__ __volatile__("" ::: "memory");
  }

  // leftovers
  int vl;
  for (int vI = cNp; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m4(pN - vI);
    vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16 *)pX + vI, vl);
    vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI, vl);
    ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, vl);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI, ay0, vl);
  }
  cNp = pN;
#else
  // fall to scalar path
  const int cNp = 0;
#endif
#elif defined(GGML_SIMD)
  const int cNp = (pN & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vVx = GGML_F16_VEC_SET1(pV);

  GGML_F16_VEC vAx[GGML_F16_ARR];
  GGML_F16_VEC vAy[GGML_F16_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F16_STEP) {
    for (int vJ = 0; vJ < GGML_F16_ARR; vJ++) {
      vAx[vJ] = GGML_F16_VEC_LOAD(pX + vI + vJ * GGML_F16_EPR, vJ);
      vAy[vJ] = GGML_F16_VEC_LOAD(pY + vI + vJ * GGML_F16_EPR, vJ);
      vAy[vJ] = GGML_F16_VEC_FMA(vAy[vJ], vAx[vJ], vVx);

      GGML_F16_VEC_STORE(pY + vI + vJ * GGML_F16_EPR, vAy, vJ);
    }
  }
#else
  // scalar path
  const int cNp = 0;
#endif

  // scalar and leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pY[vI]) + GGML_CPU_FP16_TO_FP32(pX[vI]) * pV);
  }
}

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int pN, const int pXs, const int pVs, float *GGML_RESTRICT pY,
                                           const float *GGML_RESTRICT pXv, const float *GGML_RESTRICT pVv) {

  const float *GGML_RESTRICT vX[GGML_VEC_MAD_UNROLL];
  const float *GGML_RESTRICT v_[GGML_VEC_MAD_UNROLL];

  for (int vI = 0; vI < GGML_VEC_MAD_UNROLL; ++vI) {
    vX[vI] = (const float *)((const char *)pXv + vI * pXs);
    v_[vI] = (const float *)((const char *)pVv + vI * pVs);
  }

#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
  // scalar Route to scalar implementation       //TODO: Write SVE code
  for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; ++vK) {
    for (int vI = 0; vI < pN; ++vI) {
      pY[vI] += vX[vK][vI] * v_[vK][0];
    }
  }
#elif defined(__riscv_v_intrinsic)
  for (int vI = 0, avl; vI < pN; vI += avl) {
    avl = __riscv_vsetvl_e32m8(pN - vI);
    vfloat32m8_t vAy = __riscv_vle32_v_f32m8(&pY[vI], avl);
    for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; vK++) {
      vfloat32m8_t vAx = __riscv_vle32_v_f32m8(&vX[vK][vI], avl);
      vAy = __riscv_vfmadd_vf_f32m8(vAx, v_[vK][0], vAy, avl);
    }
    __riscv_vse32_v_f32m8(&pY[vI], vAy, avl);
  }
#else
  const int cNp = (pN & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vVx[GGML_VEC_MAD_UNROLL];

  for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; ++vK) {
    vVx[vK] = GGML_F32_VEC_SET1(v_[vK][0]);
  }

  GGML_F32_VEC vAx[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
  GGML_F32_VEC vAy[GGML_F32_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F32_STEP) {
    for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
      vAy[vJ] = GGML_F32_VEC_LOAD(pY + vI + vJ * GGML_F32_EPR);

      for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; ++vK) {
        vAx[vK][vJ] = GGML_F32_VEC_LOAD(vX[vK] + vI + vJ * GGML_F32_EPR);
        vAy[vJ] = GGML_F32_VEC_FMA(vAy[vJ], vAx[vK][vJ], vVx[vK]);
      }

      GGML_F32_VEC_STORE(pY + vI + vJ * GGML_F32_EPR, vAy[vJ]);
    }
  }

  // leftovers
  for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; ++vK) {
    for (int vI = cNp; vI < pN; ++vI) {
      pY[vI] += vX[vK][vI] * v_[vK][0];
    }
  }
#endif
#else
  // scalar
  for (int vK = 0; vK < GGML_VEC_MAD_UNROLL; ++vK) {
    for (int vI = 0; vI < pN; ++vI) {
      pY[vI] += vX[vK][vI] * v_[vK][0];
    }
  }
#endif
}

inline static void ggml_vec_mad1_f32(const int pN, float *pY, const float *pX, const float pS, const float pB) {
#if defined(GGML_USE_ACCELERATE)
  vDSP_vsmsa(pX, 1, &pS, &pB, pY, 1, pN);
#elif defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
  // scalar ; TODO: Write SVE code
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = pX[vI] * pS + pB;
  }
#elif defined(__riscv_v_intrinsic)
  for (int vI = 0, avl; vI < pN; vI += avl) {
    avl = __riscv_vsetvl_e32m8(pN - vI);
    vfloat32m8_t ax = __riscv_vle32_v_f32m8(&pX[vI], avl);
    vfloat32m8_t vVb = __riscv_vfmv_v_f_f32m8(pB, avl);
    vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, pS, vVb, avl);
    __riscv_vse32_v_f32m8(&pY[vI], ny, avl);
  }
#else
  const int cNp = (pN & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vVs = GGML_F32_VEC_SET1(pS);
  GGML_F32_VEC vVb = GGML_F32_VEC_SET1(pB);

  GGML_F32_VEC vAy[GGML_F32_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F32_STEP) {
    for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
      vAy[vJ] = GGML_F32_VEC_LOAD(pX + vI + vJ * GGML_F32_EPR);
      vAy[vJ] = GGML_F32_VEC_FMA(vVb, vAy[vJ], vVs);

      GGML_F32_VEC_STORE(pY + vI + vJ * GGML_F32_EPR, vAy[vJ]);
    }
  }

  // leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    pY[vI] = pX[vI] * pS + pB;
  }
#endif
#else
  // scalar
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = pX[vI] * pS + pB;
  }
#endif
}

// inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *=
// v;          }
inline static void ggml_vec_scale_f32(const int pN, float *pY, const float pV) {
#if defined(GGML_USE_ACCELERATE)
  vDSP_vsmul(pY, 1, &pV, pY, 1, pN);
#elif defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
  const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
  const int ggml_f32_epr = sve_register_length / 32; // 8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
  const int ggml_f32_step = 2 * ggml_f32_epr;

  GGML_F32_VEC vVx = GGML_F32_VEC_SET1(pV);
  const int cNp = (pN & ~(ggml_f32_step - 1));
  svfloat32_t ay1;
  svfloat32_t ay2;
  for (int vI = 0; vI < cNp; vI += ggml_f32_step) {
    ay1 = GGML_F32_VEC_LOAD(pY + vI);
    ay1 = GGML_F32_VEC_MUL(ay1, vVx);
    GGML_F32_VEC_STORE(pY + vI, ay1);

    ay2 = GGML_F32_VEC_LOAD(pY + vI + 1 * ggml_f32_epr);
    ay2 = GGML_F32_VEC_MUL(ay2, vVx);
    GGML_F32_VEC_STORE(pY + vI + 1 * ggml_f32_epr, ay2);
  }
  // leftovers
  // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements
  // only
  for (int vI = cNp; vI < pN; vI += ggml_f32_epr) {
    svbool_t pg = svwhilelt_b32(vI, pN);
    ay1 = svld1_f32(pg, pY + vI);
    ay1 = svmul_f32_m(pg, ay1, vVx);
    svst1_f32(pg, pY + vI, ay1);
  }
#elif defined(__riscv_v_intrinsic)
  for (int vI = 0, avl; vI < pN; vI += avl) {
    avl = __riscv_vsetvl_e32m8(pN - vI);
    vfloat32m8_t vAy = __riscv_vle32_v_f32m8(&pY[vI], avl);
    vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(vAy, pV, avl);
    __riscv_vse32_v_f32m8(&pY[vI], ny, avl);
  }
#else
  const int cNp = (pN & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vVx = GGML_F32_VEC_SET1(pV);

  GGML_F32_VEC vAy[GGML_F32_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F32_STEP) {
    for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
      vAy[vJ] = GGML_F32_VEC_LOAD(pY + vI + vJ * GGML_F32_EPR);
      vAy[vJ] = GGML_F32_VEC_MUL(vAy[vJ], vVx);

      GGML_F32_VEC_STORE(pY + vI + vJ * GGML_F32_EPR, vAy[vJ]);
    }
  }

  // leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    pY[vI] *= pV;
  }
#endif
#else
  // scalar
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] *= pV;
  }
#endif
}

inline static void ggml_vec_scale_f16(const int pN, ggml_fp16_t *pY, const float pV) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
  const int sve_register_length = svcntb() * 8;
  const int ggml_f16_epr = sve_register_length / 16;
  const int ggml_f16_step = 2 * ggml_f16_epr;

  GGML_F16x_VEC vVx = GGML_F16x_VEC_SET1(pV);
  int cNp = (pN & ~(ggml_f16_step - 1));
  svfloat16_t ay1, ay2;

  for (int vI = 0; vI < cNp; vI += ggml_f16_step) {
    ay1 = GGML_F16x_VEC_LOAD(pY + vI + 0 * ggml_f16_epr, 0);
    ay1 = GGML_F16x_VEC_MUL(ay1, vVx);
    GGML_F16x_VEC_STORE(pY + vI + 0 * ggml_f16_epr, ay1, 0);

    ay2 = GGML_F16x_VEC_LOAD(pY + vI + 1 * ggml_f16_epr, 1);
    ay2 = GGML_F16x_VEC_MUL(ay2, vVx);
    GGML_F16x_VEC_STORE(pY + vI + 1 * ggml_f16_epr, ay2, 1);
  }
  // leftovers
  // maximum number of leftover elements will be less that ggmlF_16x_epr. Apply predicated svmad on available elements
  // only
  if (cNp < pN) {
    svbool_t pg = svwhilelt_b16(cNp, pN);
    svfloat16_t hy = svld1_f16(pg, (__fp16 *)(pY + cNp));
    svfloat16_t out = svmul_f16_m(pg, hy, vVx);
    svst1_f16(pg, (__fp16 *)(pY + cNp), out);
  }
  cNp = pN;
#elif defined(__riscv_v_intrinsic)
#if defined(__riscv_zvfh)
  const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(pV);
  const _Float16 scale = *(const _Float16 *)(&s);

  // calculate step size
  const int epr = __riscv_vsetvlmax_e16m4();
  const int step = epr * 2;
  int cNp = (pN & ~(step - 1));

  // unroll by 2
  for (int vI = 0; vI < cNp; vI += step) {
    vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI, epr);
    ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, epr);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI, ay0, epr);
    __asm__ __volatile__("" ::: "memory");

    vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI + epr, epr);
    ay1 = __riscv_vfmul_vf_f16m4(ay1, scale, epr);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI + epr, ay1, epr);
    __asm__ __volatile__("" ::: "memory");
  }

  // leftovers
  int vl;
  for (int vI = cNp; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m4(pN - vI);
    vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16 *)pY + vI, vl);
    ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, vl);
    __riscv_vse16_v_f16m4((_Float16 *)pY + vI, ay0, vl);
  }
  cNp = pN;
#else
  // fall to scalar path
  const int cNp = 0;
#endif
#elif defined(GGML_SIMD)
  const int cNp = (pN & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vVx = GGML_F16_VEC_SET1(pV);

  GGML_F16_VEC vAy[GGML_F16_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F16_STEP) {
    for (int vJ = 0; vJ < GGML_F16_ARR; vJ++) {
      vAy[vJ] = GGML_F16_VEC_LOAD(pY + vI + vJ * GGML_F16_EPR, vJ);
      vAy[vJ] = GGML_F16_VEC_MUL(vAy[vJ], vVx);

      GGML_F16_VEC_STORE(pY + vI + vJ * GGML_F16_EPR, vAy, vJ);
    }
  }
#else
  // scalar path
  const int cNp = 0;
#endif
  // scalar and leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pY[vI]) * pV);
  }
}

inline static void ggml_vec_norm_f32(const int pN, float *pS, const float *pX) {
  ggml_vec_dot_f32(pN, pS, 0, pX, 0, pX, 0, 1);
  *pS = sqrtf(*pS);
}
inline static void ggml_vec_sqr_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = pX[vI] * pX[vI];
}
inline static void ggml_vec_sqr_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(v_ * v_);
  }
}
inline static void ggml_vec_sqrt_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = sqrtf(pX[vI]);
}
inline static void ggml_vec_sqrt_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_log_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = logf(pX[vI]);
}
inline static void ggml_vec_log_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_sin_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = sinf(pX[vI]);
}
inline static void ggml_vec_sin_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_cos_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = cosf(pX[vI]);
}
inline static void ggml_vec_cos_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_abs_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = fabsf(pX[vI]);
}
inline static void ggml_vec_abs_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(fabsf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_sgn_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = (pX[vI] > 0.f) ? 1.f : ((pX[vI] < 0.f) ? -1.f : 0.f);
}
inline static void ggml_vec_sgn_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16((v_ > 0.f) ? 1.f : ((v_ < 0.f) ? -1.f : 0.f));
  }
}
inline static void ggml_vec_step_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = (pX[vI] > 0.f) ? 1.f : 0.f;
}
inline static void ggml_vec_step_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16((GGML_CPU_FP16_TO_FP32(pX[vI]) > 0.f) ? 1.f : 0.f);
  }
}
inline static void ggml_vec_tanh_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = tanhf(pX[vI]);
}
inline static void ggml_vec_tanh_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(tanhf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}
inline static void ggml_vec_elu_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = (pX[vI] > 0.f) ? pX[vI] : expm1f(pX[vI]);
}
inline static void ggml_vec_elu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    const float cV = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16((cV > 0.f) ? cV : expm1f(cV));
  }
}
inline static void ggml_vec_relu_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = (pX[vI] > 0.f) ? pX[vI] : 0.f;
}
inline static void ggml_vec_relu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16((v_ > 0.f) ? v_ : 0.f);
  }
}
inline static void ggml_vec_leaky_relu_f32(const int pN, float *pY, const float *pX, const float pNs) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = ((pX[vI] > 0.f) ? pX[vI] : 0.f) + pNs * ((pX[vI] < 0.0f) ? pX[vI] : 0.f);
}
inline static void ggml_vec_leaky_relu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX, const float pNs) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(((v_ > 0.f) ? v_ : 0.f) + pNs * ((v_ < 0.0f) ? v_ : 0.f));
  }
}
inline static void ggml_vec_sigmoid_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = 1.f / (1.f + expf(-pX[vI]));
}
inline static void ggml_vec_sigmoid_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(1.f / (1.f + expf(-GGML_CPU_FP16_TO_FP32(pX[vI]))));
  }
}
// TODO: optimize performance
inline static void ggml_vec_hardswish_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = pX[vI] * fminf(1.0f, fmaxf(0.0f, (pX[vI] + 3.0f) / 6.0f));
}
inline static void ggml_vec_hardswish_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(v_ * fminf(1.0f, fmaxf(0.0f, (v_ + 3.0f) / 6.0f)));
  }
}
inline static void ggml_vec_hardsigmoid_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = fminf(1.0f, fmaxf(0.0f, (pX[vI] + 3.0f) / 6.0f));
}
inline static void ggml_vec_hardsigmoid_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (GGML_CPU_FP16_TO_FP32(pX[vI]) + 3.0f) / 6.0f)));
  }
}
inline static void ggml_vec_exp_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI)
    pY[vI] = expf(pX[vI]);
}
inline static void ggml_vec_exp_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(pX[vI])));
  }
}

static const float GELU_COEF_A = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
static const float SQRT_2_INV = 0.70710678118654752440084436210484f;

inline static float ggml_gelu_f32(float pX) {
  return 0.5f * pX * (1.0f + tanhf(SQRT_2_OVER_PI * pX * (1.0f + GELU_COEF_A * pX * pX)));
}

inline static void ggml_vec_gelu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  const uint16_t *vI16 = (const uint16_t *)pX;
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = ggml_table_gelu_f16[vI16[vI]];
  }
}

inline static void ggml_vec_gelu_erf_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float vXi = GGML_CPU_FP16_TO_FP32(pX[vI]);
    float vRes = 0.5f * vXi * (1.0f + erff(vXi * SQRT_2_INV));
    pY[vI] = GGML_CPU_FP32_TO_FP16(vRes);
  }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int pN, float *pY, const float *pX) {
  uint16_t vT;
  for (int vI = 0; vI < pN; ++vI) {
    if (pX[vI] <= -10.0f) {
      pY[vI] = 0.0f;
    } else if (pX[vI] >= 10.0f) {
      pY[vI] = pX[vI];
    } else {
      ggml_fp16_t vFp16 = GGML_CPU_FP32_TO_FP16(pX[vI]);
      memcpy(&vT, &vFp16, sizeof(uint16_t));
      pY[vI] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[vT]);
    }
  }
}
#else
inline static void ggml_vec_gelu_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = ggml_gelu_f32(x[i]);
  }
}
#endif

inline static void ggml_vec_gelu_erf_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    float vXi = pX[vI];
    pY[vI] = 0.5f * vXi * (1.0f + erff(vXi * SQRT_2_INV));
  }
}

inline static float ggml_gelu_quick_f32(float pX) { return pX * (1.0f / (1.0f + expf(GELU_QUICK_COEF * pX))); }

inline static void ggml_vec_gelu_quick_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  const uint16_t *vI16 = (const uint16_t *)pX;
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = ggml_table_gelu_quick_f16[vI16[vI]];
  }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_gelu_quick_f32(const int pN, float *pY, const float *pX) {
  uint16_t vT;
  for (int vI = 0; vI < pN; ++vI) {
    ggml_fp16_t vFp16 = GGML_CPU_FP32_TO_FP16(pX[vI]);
    memcpy(&vT, &vFp16, sizeof(uint16_t));
    pY[vI] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[vT]);
  }
}
#else
inline static void ggml_vec_gelu_quick_f32(const int n, float *y, const float *x) {
  for (int i = 0; i < n; ++i) {
    y[i] = ggml_gelu_quick_f32(x[i]);
  }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float pX) { return pX / (1.0f + expf(-pX)); }
inline static ggml_fp16_t ggml_silu_f16(ggml_fp16_t pX) {
  float v_ = GGML_CPU_FP16_TO_FP32(pX);
  return GGML_CPU_FP32_TO_FP16(v_ / (1.0f + expf(-v_)));
}

#if __FINITE_MATH_ONLY__
#error                                                                                                                 \
    "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#endif

/* Below function was borrowed from the GitHub repository:
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.hpp
*/
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
inline static svfloat32_t exp_ps_sve(svbool_t pg, svfloat32_t src) {
  // Constants
  const svfloat32_t log2_e = svdup_n_f32(1.4426950409f);
  const svfloat32_t ln2 = svdup_n_f32(0.6931473921f);
  const svfloat32_t half_ln2_sq = svdup_n_f32(0.2413862043f);
  const svuint32_t not_mask17 = svdup_n_u32(~((1u << 17) - 1));
  const svfloat32_t one = svdup_n_f32(1.0f);
  const svfloat32_t inactive1 = svdup_n_f32(0.0f);
  const svint32_t inactive2 = svdup_n_s32(0);

  // Algorithm starts here
  svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);     // y = x * log2(e)
  svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0); // rount to int (float)
  svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1); // n

  t1 = svsub_f32_m(pg, t0, t1);  // a = y - floor(y)
  t1 = svadd_f32_m(pg, t1, one); // b = a + 1

  svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17); // v = b >> 17 (u32)
  svfloat32_t t4 = svexpa_f32(t3);                                  // c = fexpa(v)
  t4 = svscale_f32_m(pg, t4, t2);                                   // fexpa(v) * 2^(n)

  // and_(t2.d, t1.d, not_mask17.d)
  svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
  t5 = svsub_f32_m(pg, t1, t5);               // z
  t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq); // ln2 + half_ln2_sq * z
  t0 = svmla_f32_m(pg, one, t5, t0);          // 1 + (ln2 * z) + (half_ln2_sq * z * z)
  t0 = svmul_f32_m(pg, t0, t4);               // Final result

  return t0;
}
#endif

#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)

inline static svfloat32_t ggml_v_expf(svbool_t pg, svfloat32_t x) {
  const svfloat32_t r = svdup_n_f32_x(pg, 0x1.8p23f);
  const svfloat32_t z = svmla_n_f32_x(pg, r, x, 0x1.715476p+0f);
  const svfloat32_t n = svsub_f32_x(pg, z, r);
  const svfloat32_t b = svmls_n_f32_x(pg, svmls_n_f32_x(pg, x, n, 0x1.62e4p-1f), n, 0x1.7f7d1cp-20f);
  const svuint32_t e = svlsl_n_u32_x(pg, svreinterpret_u32_f32(z), 23);
  const svfloat32_t k = svreinterpret_f32_u32(svadd_u32_x(pg, e, svreinterpret_u32_f32(svdup_n_f32_x(pg, 1))));
  const svbool_t c = svacgt_n_f32(pg, n, 126);
  const svfloat32_t u = svmul_f32_x(pg, b, b);
  const svfloat32_t j = svmla_f32_x(
      pg, svmul_n_f32_x(pg, b, 0x1.ffffecp-1f),
      svmla_f32_x(pg, svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.fffdb6p-2f), svdup_n_f32_x(pg, 0x1.555e66p-3f), b),
                  svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.573e2ep-5f), svdup_n_f32_x(pg, 0x1.0e4020p-7f), b), u),
      u);
  const svuint32_t d = svdup_n_u32_z(svcmple_n_f32(pg, n, 0.0), 0x82000000);
  const svfloat32_t s1 = svreinterpret_f32_u32(svadd_n_u32_x(pg, d, 0x7f000000));
  const svfloat32_t s2 = svreinterpret_f32_u32(svsub_u32_x(pg, e, d));
  return svsel_f32(svacgt_f32(pg, n, svdup_n_f32_x(pg, 192)), svmul_f32_x(pg, s1, s1),
                   svsel_f32(c, svmul_f32_x(pg, svmla_f32_x(pg, s2, s2, j), s1), svmla_f32_x(pg, k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static svfloat32_t ggml_v_silu(svbool_t pg, svfloat32_t x) {
  const svfloat32_t one = svdup_n_f32_x(pg, 1.0f);
  const svfloat32_t zero = svdup_n_f32_x(pg, 0.0f);
  const svfloat32_t neg_x = svsub_f32_x(pg, zero, x);
  const svfloat32_t exp_neg_x = ggml_v_expf(pg, neg_x);
  const svfloat32_t one_plus_exp_neg_x = svadd_f32_x(pg, one, exp_neg_x);
  return svdiv_f32_x(pg, x, one_plus_exp_neg_x);
}

#elif defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t ggml_v_expf(float32x4_t x) {
  const float32x4_t r = vdupq_n_f32(0x1.8p23f);
  const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
  const float32x4_t n = vsubq_f32(z, r);
  const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
  const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
  const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
  const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
  const float32x4_t u = vmulq_f32(b, b);
  const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                                  vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                                            vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
                                  u);
  if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
    return vfmaq_f32(k, j, k);
  const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
  const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
  const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
  return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                   vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t ggml_v_silu(float32x4_t x) {
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t neg_x = vsubq_f32(zero, x);
  const float32x4_t exp_neg_x = ggml_v_expf(neg_x);
  const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
  return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 ggml_v_expf(__m512 pX) {
  const __m512 cR = _mm512_set1_ps(0x1.8p23f);
  const __m512 cZ = _mm512_fmadd_ps(pX, _mm512_set1_ps(0x1.715476p+0f), cR);
  const __m512 cN = _mm512_sub_ps(cZ, cR);
  const __m512 cB =
      _mm512_fnmadd_ps(cN, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(cN, _mm512_set1_ps(0x1.62e4p-1f), pX));
  const __mmask16 cD = _mm512_cmp_ps_mask(_mm512_abs_ps(cN), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 cU = _mm512_mul_ps(cB, cB);
  const __m512 cJ = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), cB, _mm512_set1_ps(0x1.573e2ep-5f)), cU,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), cB, _mm512_set1_ps(0x1.fffdb6p-2f))),
      cU, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), cB, _mm512_set1_ps(1.0F)));
  const __m512 cRes = _mm512_scalef_ps(cJ, cN);
  if (_mm512_kortestz(cD, cD))
    return cRes;
  const __m512 cZero = _mm512_setzero_ps();
  const __m512 cAlt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(cN, cZero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), cZero);
  return _mm512_mask_blend_ps(cD, cRes, cAlt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 ggml_v_silu(__m512 pX) {
  const __m512 cOne = _mm512_set1_ps(1);
  const __m512 cZero = _mm512_setzero_ps();
  const __m512 cNegX = _mm512_sub_ps(cZero, pX);
  const __m512 cExpNegX = ggml_v_expf(cNegX);
  const __m512 cOnePlusExpNegX = _mm512_add_ps(cOne, cExpNegX);
  return _mm512_div_ps(pX, cOnePlusExpNegX);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b =
      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f), _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c =
      _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(
      _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                      _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
      u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                     _mm256_set1_epi32(0x82000000u));
  const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d =
      _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                       _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 ggml_v_silu(__m256 x) {
  const __m256 one = _mm256_set1_ps(1);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 neg_x = _mm256_sub_ps(zero, x);
  const __m256 exp_neg_x = ggml_v_expf(neg_x);
  const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
  return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 ggml_v_expf(__m128 x) {
  const __m128 r = _mm_set1_ps(0x1.8p23f);
  const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
  const __m128 n = _mm_sub_ps(z, r);
  const __m128 b = NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
  const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
  const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
  const __m128i c = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
  const __m128 u = _mm_mul_ps(b, b);
  const __m128 j = MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                                   MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                           u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm_movemask_epi8(c))
    return MADD128(j, k, k);
  const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())), _mm_set1_epi32(0x82000000u));
  const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
  const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
  const __m128i d = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
  return _mm_or_ps(
      _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
      _mm_andnot_ps(_mm_castsi128_ps(d), _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                                   _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 ggml_v_silu(__m128 x) {
  const __m128 one = _mm_set1_ps(1);
  const __m128 zero = _mm_setzero_ps();
  const __m128 neg_x = _mm_sub_ps(zero, x);
  const __m128 exp_neg_x = ggml_v_expf(neg_x);
  const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
  return _mm_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__riscv_v_intrinsic)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static vfloat32m2_t ggml_v_expf_m2(vfloat32m2_t x, int vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
#ifdef __riscv_xtheadvector
  // workaround for compiler bug (gcc 14.3.0: Error: unrecognized opcode `th.vmv1r.v v2,v4')
  vfloat32m2_t z = __riscv_vfadd_vf_f32m2(r, 0.0f, vl);
  z = __riscv_vfmacc_vf_f32m2(z, 0x1.715476p+0f, x, vl);
#else
  const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
#endif
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b =
      __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl)); // 1.0f
  const vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
      __riscv_vfmacc_vv_f32m2(
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
      u, vl);
  if (!__riscv_vcpop_m_b16(c, vl))
    return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
  const vbool16_t dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
  const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t r1 =
      __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                               __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
  return __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                  __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static vfloat32m2_t ggml_v_silu_m2(vfloat32m2_t x, int vl) {
  const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
  const vfloat32m2_t exp_neg_x = ggml_v_expf_m2(neg_x, vl);
  const vfloat32m2_t one_plus_exp_neg_x = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
  return __riscv_vfdiv_vv_f32m2(x, one_plus_exp_neg_x, vl);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__ / __riscv_v_intrinsic

inline static void ggml_vec_silu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = ggml_silu_f16(pX[vI]);
  }
}

inline static float ggml_silu_backward_f32(float pX, float pDy) {
  const float cS = 1.0f / (1.0f + expf(-pX));
  return pDy * cS * (1.0f + pX * (1.0f - cS));
}

inline static ggml_fp16_t ggml_silu_backward_f16(ggml_fp16_t pX, ggml_fp16_t pDy) {
  const float cV = GGML_CPU_FP16_TO_FP32(pX);
  const float cS = 1.0f / (1.0f + expf(-cV));
  return GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(pDy) * cS * (1.0f + cV * (1.0f - cS)));
}

inline static void ggml_vec_silu_backward_f32(const int pN, float *pDx, const float *pX, const float *pDy) {
  for (int vI = 0; vI < pN; ++vI) {
    pDx[vI] = ggml_silu_backward_f32(pX[vI], pDy[vI]);
  }
}

inline static void ggml_vec_silu_backward_f16(const int pN, ggml_fp16_t *pDx, const ggml_fp16_t *pX,
                                              const ggml_fp16_t *pDy) {
  for (int vI = 0; vI < pN; ++vI) {
    pDx[vI] = ggml_silu_backward_f16(pX[vI], pDy[vI]);
  }
}

inline static void ggml_vec_reglu_f32(const int pN, float *pY, const float *pX, const float *pG) {
  for (int vI = 0; vI < pN; ++vI) {
    pY[vI] = (pX[vI] > 0.f) ? pX[vI] * pG[vI] : 0.f;
  }
}

inline static void ggml_vec_reglu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX, const ggml_fp16_t *pG) {
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pX[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16((v_ > 0.f) ? v_ * GGML_CPU_FP16_TO_FP32(pG[vI]) : 0.f);
  }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_geglu_f32(const int pN, float *pY, const float *pX, const float *pG) {
  uint16_t vT;
  for (int vI = 0; vI < pN; ++vI) {
    if (pX[vI] <= -10.0f) {
      pY[vI] = 0.0f;
    } else if (pX[vI] >= 10.0f) {
      pY[vI] = pX[vI] * pG[vI];
    } else {
      ggml_fp16_t vFp16 = GGML_CPU_FP32_TO_FP16(pX[vI]);
      memcpy(&vT, &vFp16, sizeof(uint16_t));
      pY[vI] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[vT]) * pG[vI];
    }
  }
}
#else
inline static void ggml_vec_geglu_f32(const int n, float *y, const float *x, const float *g) {
  for (int i = 0; i < n; ++i) {
    y[i] = ggml_gelu_f32(x[i]) * g[i];
  }
}
#endif

inline static void ggml_vec_geglu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX, const ggml_fp16_t *pG) {
  const uint16_t *vI16 = (const uint16_t *)pX;
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pG[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[vI16[vI]]) * v_);
  }
}

void ggml_vec_swiglu_f32(const int pN, float *pY, const float *pX, const float *pG);

inline static void ggml_vec_swiglu_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX, const ggml_fp16_t *pG) {
  for (int vI = 0; vI < pN; ++vI) {
    float vXi = GGML_CPU_FP16_TO_FP32(pX[vI]);
    float vGi = GGML_CPU_FP16_TO_FP32(pG[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16((vXi / (1.0f + expf(-vXi))) * vGi);
  }
}

inline static void ggml_vec_geglu_erf_f32(const int pN, float *pY, const float *pX, const float *pG) {
  for (int vI = 0; vI < pN; ++vI) {
    float vXi = pX[vI];
    pY[vI] = 0.5f * vXi * (1.0f + erff(vXi * SQRT_2_INV)) * pG[vI];
  }
}

inline static void ggml_vec_geglu_erf_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX, const ggml_fp16_t *pG) {
  for (int vI = 0; vI < pN; ++vI) {
    float vXi = GGML_CPU_FP16_TO_FP32(pX[vI]);
    float vGi = GGML_CPU_FP16_TO_FP32(pG[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(0.5f * vXi * (1.0f + erff(vXi * SQRT_2_INV)) * vGi);
  }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_geglu_quick_f32(const int pN, float *pY, const float *pX, const float *pG) {
  uint16_t vT;
  for (int vI = 0; vI < pN; ++vI) {
    ggml_fp16_t vFp16 = GGML_CPU_FP32_TO_FP16(pX[vI]);
    memcpy(&vT, &vFp16, sizeof(uint16_t));
    pY[vI] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[vT]) * pG[vI];
  }
}
#else
inline static void ggml_vec_geglu_quick_f32(const int n, float *y, const float *x, const float *g) {
  for (int i = 0; i < n; ++i) {
    y[i] = ggml_gelu_quick_f32(x[i]) * g[i];
  }
}
#endif

inline static void ggml_vec_geglu_quick_f16(const int pN, ggml_fp16_t *pY, const ggml_fp16_t *pX,
                                            const ggml_fp16_t *pG) {
  const uint16_t *vI16 = (const uint16_t *)pX;
  for (int vI = 0; vI < pN; ++vI) {
    float v_ = GGML_CPU_FP16_TO_FP32(pG[vI]);
    pY[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[vI16[vI]]) * v_);
  }
}

inline static void ggml_vec_sum_f32(const int pN, float *pS, const float *pX) {
#ifndef GGML_USE_ACCELERATE
  ggml_float vSum = 0.0;
  for (int vI = 0; vI < pN; ++vI) {
    vSum += (ggml_float)pX[vI];
  }
  *pS = (float)vSum;
#else
  vDSP_sve(pX, 1, pS, pN);
#endif
}

inline static void ggml_vec_cumsum_f32(const int pN, float *pY, const float *pX) {
  for (int vI = 0; vI < pN; ++vI) {
    if (vI == 0) {
      pY[vI] = pX[vI];
    } else {
      pY[vI] = pY[vI - 1] + pX[vI];
    }
  }
}

inline static void ggml_vec_sum_f32_ggf(const int pN, ggml_float *pS, const float *pX) {
  ggml_float vSum = 0.0;
  for (int vI = 0; vI < pN; ++vI) {
    vSum += (ggml_float)pX[vI];
  }
  *pS = vSum;
}

inline static void ggml_vec_sum_f16_ggf(const int pN, float *pS, const ggml_fp16_t *pX) {
  float vSum = 0.0f;
  for (int vI = 0; vI < pN; ++vI) {
    vSum += GGML_CPU_FP16_TO_FP32(pX[vI]);
  }
  *pS = vSum;
}

inline static void ggml_vec_sum_bf16_ggf(const int pN, float *pS, const ggml_bf16_t *pX) {
  float vSum = 0.0f;
  for (int vI = 0; vI < pN; ++vI) {
    vSum += GGML_BF16_TO_FP32(pX[vI]);
  }
  *pS = vSum;
}

inline static void ggml_vec_max_f32(const int pN, float *pS, const float *pX) {
#ifndef GGML_USE_ACCELERATE
  float vMax = -INFINITY;
  for (int vI = 0; vI < pN; ++vI) {
    vMax = MAX(vMax, pX[vI]);
  }
  *pS = vMax;
#else
  vDSP_maxv(pX, 1, pS, pN);
#endif
}

inline static void ggml_vec_norm_inv_f32(const int pN, float *pS, const float *pX) {
  ggml_vec_norm_f32(pN, pS, pX);
  *pS = 1.f / (*pS);
}

inline static void ggml_vec_argmax_f32(const int pN, int *pS, const float *pX) {
  float vMax = -INFINITY;
  int vIdx = 0;
  for (int vI = 0; vI < pN; ++vI) {
    vMax = MAX(vMax, pX[vI]);
    if (vMax == pX[vI]) {
      vIdx = vI;
    }
  }
  *pS = vIdx;
}

#ifdef __cplusplus
}
#endif
