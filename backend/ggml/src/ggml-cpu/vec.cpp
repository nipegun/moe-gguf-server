#include "vec.h"

#include <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int pN, float *GGML_RESTRICT pS, size_t pBs, const float *GGML_RESTRICT pX, size_t pBx,
                      const float *GGML_RESTRICT pY, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  GGML_UNUSED(pNrc);
  GGML_UNUSED(pBx);
  GGML_UNUSED(pBy);
  GGML_UNUSED(pBs);

#if defined(GGML_SIMD)
  float vSumf = 0.0f;

#if defined(__ARM_FEATURE_SVE)
  const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
  const int ggml_f32_epr = sve_register_length / 32; // 8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
  const int ggml_f32_step = 8 * ggml_f32_epr;        // choose 8 SVE registers

  const int cNp = (pN & ~(ggml_f32_step - 1));
  svfloat32_t sum1 = svdup_n_f32(0.0f);
  svfloat32_t sum2 = svdup_n_f32(0.0f);
  svfloat32_t sum3 = svdup_n_f32(0.0f);
  svfloat32_t sum4 = svdup_n_f32(0.0f);
  svfloat32_t sum5 = svdup_n_f32(0.0f);
  svfloat32_t sum6 = svdup_n_f32(0.0f);
  svfloat32_t sum7 = svdup_n_f32(0.0f);
  svfloat32_t sum8 = svdup_n_f32(0.0f);
  svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
  svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
  for (int vI = 0; vI < cNp; vI += ggml_f32_step) {
    ax1 = GGML_F32_VEC_LOAD(pX + vI);
    ay1 = GGML_F32_VEC_LOAD(pY + vI);
    sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

    ax2 = GGML_F32_VEC_LOAD(pX + vI + 1 * ggml_f32_epr);
    ay2 = GGML_F32_VEC_LOAD(pY + vI + 1 * ggml_f32_epr);
    sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

    ax3 = GGML_F32_VEC_LOAD(pX + vI + 2 * ggml_f32_epr);
    ay3 = GGML_F32_VEC_LOAD(pY + vI + 2 * ggml_f32_epr);
    sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

    ax4 = GGML_F32_VEC_LOAD(pX + vI + 3 * ggml_f32_epr);
    ay4 = GGML_F32_VEC_LOAD(pY + vI + 3 * ggml_f32_epr);
    sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

    ax5 = GGML_F32_VEC_LOAD(pX + vI + 4 * ggml_f32_epr);
    ay5 = GGML_F32_VEC_LOAD(pY + vI + 4 * ggml_f32_epr);
    sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

    ax6 = GGML_F32_VEC_LOAD(pX + vI + 5 * ggml_f32_epr);
    ay6 = GGML_F32_VEC_LOAD(pY + vI + 5 * ggml_f32_epr);
    sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

    ax7 = GGML_F32_VEC_LOAD(pX + vI + 6 * ggml_f32_epr);
    ay7 = GGML_F32_VEC_LOAD(pY + vI + 6 * ggml_f32_epr);
    sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

    ax8 = GGML_F32_VEC_LOAD(pX + vI + 7 * ggml_f32_epr);
    ay8 = GGML_F32_VEC_LOAD(pY + vI + 7 * ggml_f32_epr);
    sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
  }
  // leftovers
  // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
  const int np2 = (pN & ~(ggml_f32_epr - 1));
  for (int vI = cNp; vI < np2; vI += ggml_f32_epr) {
    ax1 = GGML_F32_VEC_LOAD(pX + vI);
    ay1 = GGML_F32_VEC_LOAD(pY + vI);
    sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
  }
  // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmla on available elements
  // only
  if (np2 < pN) {
    svbool_t pg = svwhilelt_b32(np2, pN);
    ax1 = svld1_f32(pg, pX + np2);
    ay1 = svld1_f32(pg, pY + np2);
    sum1 = svmla_f32_m(pg, sum1, ax1, ay1);
  }
  // reduce sum1,sum2 to sum1
  GGML_F32_VEC_REDUCE(vSumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
#elif defined(__riscv_v_intrinsic)
  int vl = __riscv_vsetvlmax_e32m8();
  vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m8_t vsum;
  vfloat32m8_t vAx;
  vfloat32m8_t vAy;
  vsum = __riscv_vfmv_v_f_f32m8_tu(vsum, 0.0f, vl);
  for (int vI = 0; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e32m8(pN - vI);
    vAx = __riscv_vle32_v_f32m8_tu(vAx, &pX[vI], vl);
    vAy = __riscv_vle32_v_f32m8_tu(vAy, &pY[vI], vl);
    vsum = __riscv_vfmacc_vv_f32m8_tu(vsum, vAx, vAy, vl);
  }
  vl = __riscv_vsetvlmax_e32m8();
  vs = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vs, vl);
  vSumf += __riscv_vfmv_f_s_f32m1_f32(vs);
#else
  const int cNp = (pN & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vSum[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};

  GGML_F32_VEC vAx[GGML_F32_ARR];
  GGML_F32_VEC vAy[GGML_F32_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F32_STEP) {
    for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
      vAx[vJ] = GGML_F32_VEC_LOAD(pX + vI + vJ * GGML_F32_EPR);
      vAy[vJ] = GGML_F32_VEC_LOAD(pY + vI + vJ * GGML_F32_EPR);

      vSum[vJ] = GGML_F32_VEC_FMA(vSum[vJ], vAx[vJ], vAy[vJ]);
    }
  }

  // reduce sum0..sum3 to sum0
  GGML_F32_VEC_REDUCE(vSumf, vSum);

  // leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    vSumf += pX[vI] * pY[vI];
  }
#endif
#else
  // scalar
  ggml_float vSumf = 0.0;
  for (int vI = 0; vI < pN; ++vI) {
    vSumf += (ggml_float)(pX[vI] * pY[vI]);
  }
#endif

  *pS = vSumf;
}

void ggml_vec_dot_bf16(int pN, float *GGML_RESTRICT pS, size_t pBs, ggml_bf16_t *GGML_RESTRICT pX, size_t pBx,
                       ggml_bf16_t *GGML_RESTRICT pY, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  GGML_UNUSED(pNrc);
  GGML_UNUSED(pBx);
  GGML_UNUSED(pBy);
  GGML_UNUSED(pBs);
  int vI = 0;
  ggml_float vSumf = 0;

#if defined(__AVX512BF16__)
  __m512 vC1 = _mm512_setzero_ps();
  __m512 vC2 = _mm512_setzero_ps();
  for (; vI + 64 <= pN; vI += 64) {
    vC1 = _mm512_dpbf16_ps(vC1, m512bh(_mm512_loadu_si512((pX + vI))), m512bh(_mm512_loadu_si512((pY + vI))));
    vC2 = _mm512_dpbf16_ps(vC2, m512bh(_mm512_loadu_si512((pX + vI + 32))), m512bh(_mm512_loadu_si512((pY + vI + 32))));
  }
  vSumf += (ggml_float)_mm512_reduce_add_ps(vC1);
  vSumf += (ggml_float)_mm512_reduce_add_ps(vC2);

#elif defined(__AVX512F__)
#define LOAD(p)                                                                                                        \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
  __m512 vC1 = _mm512_setzero_ps();
  __m512 vC2 = _mm512_setzero_ps();
  for (; vI + 32 <= pN; vI += 32) {
    vC1 = _mm512_add_ps(_mm512_mul_ps(LOAD(pX + vI), LOAD(pY + vI)), vC1);
    vC2 = _mm512_add_ps(_mm512_mul_ps(LOAD(pX + vI + 16), LOAD(pY + vI + 16)), vC2);
  }
  vSumf += (ggml_float)_mm512_reduce_add_ps(vC1);
  vSumf += (ggml_float)_mm512_reduce_add_ps(vC2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p)                                                                                                        \
  _mm256_castsi256_ps(_mm256_insertf128_si256(                                                                         \
      _mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)),           \
      (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
  __m256 vC1 = _mm256_setzero_ps();
  __m256 vC2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps();
  for (; vI + 32 <= pN; vI += 32) {
    vC1 = _mm256_add_ps(_mm256_mul_ps(LOAD(pX + vI), LOAD(pY + vI)), vC1);
    vC2 = _mm256_add_ps(_mm256_mul_ps(LOAD(pX + vI + 8), LOAD(pY + vI + 8)), vC2);
    c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(pX + vI + 16), LOAD(pY + vI + 16)), c3);
    c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(pX + vI + 24), LOAD(pY + vI + 24)), c4);
  }
  __m128 g;
  vC1 = _mm256_add_ps(_mm256_add_ps(vC1, c3), _mm256_add_ps(vC2, c4));
  g = _mm_add_ps(_mm256_extractf128_ps(vC1, 1), _mm256_castps256_ps128(vC1));
  g = _mm_add_ps(g, _mm_movehl_ps(g, g));
  g = _mm_add_ss(g, _mm_movehdup_ps(g));
  vSumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
  size_t vl = __riscv_vsetvlmax_e32m4();

  // initialize accumulators to all zeroes
  vfloat32m4_t vsum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t vsum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

  // calculate step size
  const size_t epr = __riscv_vsetvlmax_e16m2();
  const size_t step = epr * 2;
  const int np = (pN & ~(step - 1));

  // unroll by 2
  for (; vI < np; vI += step) {
    vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)&pX[vI], epr);
    vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((const __bf16 *)&pY[vI], epr);
    vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, epr);
    __asm__ __volatile__("" ::: "memory");

    vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16 *)&pX[vI + epr], epr);
    vbfloat16m2_t ay1 = __riscv_vle16_v_bf16m2((const __bf16 *)&pY[vI + epr], epr);
    vsum1 = __riscv_vfwmaccbf16_vv_f32m4(vsum1, ax1, ay1, epr);
    __asm__ __volatile__("" ::: "memory");
  }

  // accumulate in 1 register
  vsum0 = __riscv_vfadd_vv_f32m4(vsum0, vsum1, vl);

  // leftovers
  for (vI = np; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m2(pN - vI);
    vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16 *)&pX[vI], vl);
    vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((const __bf16 *)&pY[vI], vl);
    vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, vl);
  }

  // reduce
  vl = __riscv_vsetvlmax_e32m4();
  vfloat32m1_t redsum = __riscv_vfredusum_vs_f32m4_f32m1(vsum0, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
  vSumf += __riscv_vfmv_f_s_f32m1_f32(redsum);

#elif defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
  const int np = (pN & ~(GGML_BF16_STEP - 1));
  if (np > 0) {
    GGML_F32_VEC sum[4] = {GGML_F32_VEC_ZERO};
    for (; vI < np; vI += GGML_BF16_STEP) {
      GGML_BF16_VEC vx0 = GGML_BF16_VEC_LOAD(pX + vI);
      GGML_BF16_VEC vx1 = GGML_BF16_VEC_LOAD(pX + vI + 8);
      GGML_BF16_VEC vy0 = GGML_BF16_VEC_LOAD(pY + vI);
      GGML_BF16_VEC vy1 = GGML_BF16_VEC_LOAD(pY + vI + 8);
      GGML_BF16_FMA_LO(sum[0], vx0, vy0);
      GGML_BF16_FMA_HI(sum[1], vx0, vy0);
      GGML_BF16_FMA_LO(sum[2], vx1, vy1);
      GGML_BF16_FMA_HI(sum[3], vx1, vy1);
    }
    GGML_F32x4_REDUCE_4(vSumf, sum[0], sum[1], sum[2], sum[3]);
  }
#endif

  for (; vI < pN; ++vI) {
    vSumf += (ggml_float)(GGML_BF16_TO_FP32(pX[vI]) * GGML_BF16_TO_FP32(pY[vI]));
  }
  *pS = vSumf;
}

void ggml_vec_dot_f16(int pN, float *GGML_RESTRICT pS, size_t pBs, ggml_fp16_t *GGML_RESTRICT pX, size_t pBx,
                      ggml_fp16_t *GGML_RESTRICT pY, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  GGML_UNUSED(pNrc);
  GGML_UNUSED(pBx);
  GGML_UNUSED(pBy);
  GGML_UNUSED(pBs);

  ggml_float vSumf = 0.0;

#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
  const int ggml_f16_epr = svcnth();
  const int ggml_f16_step = 8 * ggml_f16_epr;
  const int cNp = pN - (pN % ggml_f16_step);
  const int np2 = pN - (pN % ggml_f16_epr);

  svfloat32_t sum1_lo = svdup_n_f32(0.0f);
  svfloat32_t sum1_hi = svdup_n_f32(0.0f);
  svfloat32_t sum2_lo = svdup_n_f32(0.0f);
  svfloat32_t sum2_hi = svdup_n_f32(0.0f);
  svfloat32_t sum3_lo = svdup_n_f32(0.0f);
  svfloat32_t sum3_hi = svdup_n_f32(0.0f);
  svfloat32_t sum4_lo = svdup_n_f32(0.0f);
  svfloat32_t sum4_hi = svdup_n_f32(0.0f);

  for (int vI = 0; vI < cNp; vI += ggml_f16_step) {
    ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(pX + vI + 0 * ggml_f16_epr, 0),
                             GGML_F16x_VEC_LOAD(pY + vI + 0 * ggml_f16_epr, 0));
    ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_LOAD(pX + vI + 1 * ggml_f16_epr, 1),
                             GGML_F16x_VEC_LOAD(pY + vI + 1 * ggml_f16_epr, 1));
    ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_LOAD(pX + vI + 2 * ggml_f16_epr, 2),
                             GGML_F16x_VEC_LOAD(pY + vI + 2 * ggml_f16_epr, 2));
    ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_LOAD(pX + vI + 3 * ggml_f16_epr, 3),
                             GGML_F16x_VEC_LOAD(pY + vI + 3 * ggml_f16_epr, 3));
    ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(pX + vI + 4 * ggml_f16_epr, 4),
                             GGML_F16x_VEC_LOAD(pY + vI + 4 * ggml_f16_epr, 4));
    ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_LOAD(pX + vI + 5 * ggml_f16_epr, 5),
                             GGML_F16x_VEC_LOAD(pY + vI + 5 * ggml_f16_epr, 5));
    ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_LOAD(pX + vI + 6 * ggml_f16_epr, 6),
                             GGML_F16x_VEC_LOAD(pY + vI + 6 * ggml_f16_epr, 6));
    ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_LOAD(pX + vI + 7 * ggml_f16_epr, 7),
                             GGML_F16x_VEC_LOAD(pY + vI + 7 * ggml_f16_epr, 7));
  }

  for (int vI = cNp; vI < np2; vI += ggml_f16_epr) {
    ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_LOAD(pX + vI, 0), GGML_F16x_VEC_LOAD(pY + vI, 0));
  }

  if (np2 < pN) {
    const svbool_t pg = svwhilelt_b16(np2, pN);
    const svfloat16_t rx = svld1_f16(pg, (const __fp16 *)(pX + np2));
    const svfloat16_t ry = svld1_f16(pg, (const __fp16 *)(pY + np2));

    ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, rx, ry);
  }

  sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum2_lo);
  sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum2_hi);
  sum3_lo = svadd_f32_m(DEFAULT_PG32, sum3_lo, sum4_lo);
  sum3_hi = svadd_f32_m(DEFAULT_PG32, sum3_hi, sum4_hi);
  sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum3_lo);
  sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum3_hi);

  vSumf = ggml_sve_sum_f32x2(sum1_lo, sum1_hi);
#elif defined(__riscv_v_intrinsic)
#if defined(__riscv_zvfh)
  int vl = __riscv_vsetvlmax_e32m2();
  vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m2_t vsum;
  vfloat16m1_t vAx;
  vfloat16m1_t vAy;
  vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
  for (int vI = 0; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e16m1(pN - vI);
    vAx = __riscv_vle16_v_f16m1_tu(vAx, (const _Float16 *)&pX[vI], vl);
    vAy = __riscv_vle16_v_f16m1_tu(vAy, (const _Float16 *)&pY[vI], vl);
    vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, vAx, vAy, vl);
  }
  vl = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t ac0 =
      __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
  vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
  vSumf += __riscv_vfmv_f_s_f32m1_f32(vs);
#else
  for (int vI = 0; vI < pN; ++vI) {
    vSumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(pX[vI]) * GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
#endif // __riscv_zvfh
#else
  const int cNp = (pN & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vSum[GGML_F16_ARR] = {GGML_F16_VEC_ZERO};

  GGML_F16_VEC vAx[GGML_F16_ARR];
  GGML_F16_VEC vAy[GGML_F16_ARR];

  for (int vI = 0; vI < cNp; vI += GGML_F16_STEP) {
    for (int vJ = 0; vJ < GGML_F16_ARR; vJ++) {
      vAx[vJ] = GGML_F16_VEC_LOAD(pX + vI + vJ * GGML_F16_EPR, vJ);
      vAy[vJ] = GGML_F16_VEC_LOAD(pY + vI + vJ * GGML_F16_EPR, vJ);

      vSum[vJ] = GGML_F16_VEC_FMA(vSum[vJ], vAx[vJ], vAy[vJ]);
    }
  }

  // reduce sum0..sum3 to sum0
  GGML_F16_VEC_REDUCE(vSumf, vSum);

  // leftovers
  for (int vI = cNp; vI < pN; ++vI) {
    vSumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(pX[vI]) * GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
  // if you hit this, you are likely running outside the FP range
  assert(!isnan(vSumf) && !isinf(vSumf));
#endif
#else
  for (int vI = 0; vI < pN; ++vI) {
    vSumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(pX[vI]) * GGML_CPU_FP16_TO_FP32(pY[vI]));
  }
#endif // GGML_SIMD

  *pS = vSumf;
}

void ggml_vec_silu_f32(const int pN, float *pY, const float *pX) {
  int vI = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  for (; vI + 15 < pN; vI += 16) {
    _mm512_storeu_ps(pY + vI, ggml_v_silu(_mm512_loadu_ps(pX + vI)));
  }
#elif defined(__AVX2__) && defined(__FMA__)
  for (; vI + 7 < pN; vI += 8) {
    _mm256_storeu_ps(pY + vI, ggml_v_silu(_mm256_loadu_ps(pX + vI)));
  }
#elif defined(__SSE2__)
  for (; vI + 3 < pN; vI += 4) {
    _mm_storeu_ps(pY + vI, ggml_v_silu(_mm_loadu_ps(pX + vI)));
  }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
  const int vlen = svcntw();
  for (; vI < pN; vI += vlen) {
    const svbool_t pg = svwhilelt_b32_s32(vI, pN);
    svst1_f32(pg, pY + vI, ggml_v_silu(pg, svld1_f32(pg, pX + vI)));
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  for (; vI + 3 < pN; vI += 4) {
    vst1q_f32(pY + vI, ggml_v_silu(vld1q_f32(pX + vI)));
  }
#elif defined(__riscv_v_intrinsic)
  for (int vl; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e32m2(pN - vI);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(&pX[vI], vl);
    vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);
    __riscv_vse32_v_f32m2(&pY[vI], vy, vl);
  }
#endif
  for (; vI < pN; ++vI) {
    pY[vI] = ggml_silu_f32(pX[vI]);
  }
}

void ggml_vec_swiglu_f32(const int pN, float *pY, const float *pX, const float *pG) {
  int vI = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  for (; vI + 15 < pN; vI += 16) {
    _mm512_storeu_ps(pY + vI, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(pX + vI)), _mm512_loadu_ps(pG + vI)));
  }
#elif defined(__AVX2__) && defined(__FMA__)
  for (; vI + 7 < pN; vI += 8) {
    _mm256_storeu_ps(pY + vI, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(pX + vI)), _mm256_loadu_ps(pG + vI)));
  }
#elif defined(__SSE2__)
  for (; vI + 3 < pN; vI += 4) {
    _mm_storeu_ps(pY + vI, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(pX + vI)), _mm_loadu_ps(pG + vI)));
  }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
  const int vlen = svcntw();
  for (; vI < pN; vI += vlen) {
    const svbool_t pg = svwhilelt_b32_s32(vI, pN);
    svst1_f32(pg, pY + vI, svmul_f32_x(pg, ggml_v_silu(pg, svld1_f32(pg, pX + vI)), svld1_f32(pg, pG + vI)));
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  for (; vI + 3 < pN; vI += 4) {
    vst1q_f32(pY + vI, vmulq_f32(ggml_v_silu(vld1q_f32(pX + vI)), vld1q_f32(pG + vI)));
  }
#elif defined(__riscv_v_intrinsic)
  for (int vl; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e32m2(pN - vI);
    vfloat32m2_t vx = __riscv_vle32_v_f32m2(&pX[vI], vl);
    vfloat32m2_t vg = __riscv_vle32_v_f32m2(&pG[vI], vl);
    vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_silu_m2(vx, vl), vg, vl);
    __riscv_vse32_v_f32m2(&pY[vI], vy, vl);
  }
#endif
  for (; vI < pN; ++vI) {
    pY[vI] = ggml_silu_f32(pX[vI]) * pG[vI];
  }
}

ggml_float ggml_vec_cvar_f32(const int pN, float *pY, const float *pX, const float pMean) {
  int vI = 0;
  ggml_float vSum = 0;
// TODO: optimize to process the remaining elements in groups using the smaller vector sizes from AVX2 and SSE
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  for (; vI + 15 < pN; vI += 16) {
    __m512 vVal = _mm512_sub_ps(_mm512_loadu_ps(pX + vI), _mm512_set1_ps(pMean));
    _mm512_storeu_ps(pY + vI, vVal);
    vSum += (ggml_float)_mm512_reduce_add_ps(_mm512_mul_ps(vVal, vVal));
  }
#elif defined(__AVX2__) && defined(__FMA__)
  for (; vI + 7 < pN; vI += 8) {
    __m256 vVal = _mm256_sub_ps(_mm256_loadu_ps(pX + vI), _mm256_set1_ps(pMean));
    _mm256_storeu_ps(pY + vI, vVal);
    vVal = _mm256_mul_ps(vVal, vVal);
    __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(vVal, 1), _mm256_castps256_ps128(vVal));
    val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
    val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
    vSum += (ggml_float)_mm_cvtss_f32(val2);
  }
#elif defined(__SSE2__)
  for (; vI + 3 < pN; vI += 4) {
    __m128 vVal = _mm_sub_ps(_mm_loadu_ps(pX + vI), _mm_set1_ps(pMean));
    _mm_storeu_ps(pY + vI, vVal);
    vVal = _mm_mul_ps(vVal, vVal);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    vVal = _mm_add_ps(vVal, _mm_movehl_ps(vVal, vVal));
    vVal = _mm_add_ss(vVal, _mm_movehdup_ps(vVal));
#else
    __m128 tmp = _mm_shuffle_ps(vVal, vVal, _MM_SHUFFLE(2, 3, 0, 1));
    vVal = _mm_add_ps(vVal, tmp);
    tmp = _mm_movehl_ps(tmp, vVal);
    vVal = _mm_add_ss(vVal, tmp);
#endif // __AVX__ || __AVX2__ || __AVX512F__
    vSum += (ggml_float)_mm_cvtss_f32(vVal);
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  for (; vI + 3 < pN; vI += 4) {
    float32x4_t vVal = vsubq_f32(vld1q_f32(pX + vI), vdupq_n_f32(pMean));
    vst1q_f32(pY + vI, vVal);
    vVal = vmulq_f32(vVal, vVal);
    vSum += (ggml_float)vaddvq_f32(vVal);
  }
#elif defined(__VXE__) || defined(__VXE2__)
  for (; vI + 3 < pN; vI += 4) {
    float32x4_t vVal = vec_sub(vec_xl(0, pX + vI), vec_splats(pMean));
    vec_xst(vVal, 0, pY + vI);
    vVal = vec_mul(vVal, vVal);
    vSum += (ggml_float)vec_hsum_f32x4(vVal);
  }
#elif defined(__riscv_v_intrinsic)
  vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
  for (int vl; vI < pN; vI += vl) {
    vl = __riscv_vsetvl_e32m2(pN - vI);
    vfloat32m2_t vVal = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&pX[vI], vl), pMean, vl);
    __riscv_vse32_v_f32m2(&pY[vI], vVal, vl);
    vVal = __riscv_vfmul_vv_f32m2(vVal, vVal, vl);
    vsum = __riscv_vfwredusum_vs_f32m2_f64m1(vVal, vsum, vl);
  }
  vSum = (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#endif
  for (; vI < pN; ++vI) {
    float vVal = pX[vI] - pMean;
    pY[vI] = vVal;
    vVal *= vVal;
    vSum += (ggml_float)vVal;
  }
  return vSum / pN;
}

ggml_float ggml_vec_soft_max_f32(const int pN, float *pY, const float *pX, float pMax) {
  int vI = 0;
  ggml_float vSum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  for (; vI + 15 < pN; vI += 16) {
    __m512 vVal = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(pX + vI), _mm512_set1_ps(pMax)));
    _mm512_storeu_ps(pY + vI, vVal);
    vSum += (ggml_float)_mm512_reduce_add_ps(vVal);
  }
#elif defined(__AVX2__) && defined(__FMA__)
  for (; vI + 7 < pN; vI += 8) {
    __m256 vVal = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(pX + vI), _mm256_set1_ps(pMax)));
    _mm256_storeu_ps(pY + vI, vVal);
    __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(vVal, 1), _mm256_castps256_ps128(vVal));
    val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
    val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
    vSum += (ggml_float)_mm_cvtss_f32(val2);
  }
#elif defined(__SSE2__)
  for (; vI + 3 < pN; vI += 4) {
    __m128 vVal = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(pX + vI), _mm_set1_ps(pMax)));
    _mm_storeu_ps(pY + vI, vVal);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    vVal = _mm_add_ps(vVal, _mm_movehl_ps(vVal, vVal));
    vVal = _mm_add_ss(vVal, _mm_movehdup_ps(vVal));
#else
    __m128 tmp = _mm_shuffle_ps(vVal, vVal, _MM_SHUFFLE(2, 3, 0, 1));
    vVal = _mm_add_ps(vVal, tmp);
    tmp = _mm_movehl_ps(tmp, vVal);
    vVal = _mm_add_ss(vVal, tmp);
#endif
    vSum += (ggml_float)_mm_cvtss_f32(vVal);
  }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
  const int vlen = svcntw();
  for (; vI < pN; vI += vlen) {
    const svbool_t pg = svwhilelt_b32_s32(vI, pN);
    svfloat32_t vVal = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, pX + vI), svdup_n_f32_x(pg, pMax)));
    svst1_f32(pg, pY + vI, vVal);
    vSum += (ggml_float)svaddv_f32(pg, vVal);
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  for (; vI + 3 < pN; vI += 4) {
    float32x4_t vVal = ggml_v_expf(vsubq_f32(vld1q_f32(pX + vI), vdupq_n_f32(pMax)));
    vst1q_f32(pY + vI, vVal);
    vSum += (ggml_float)vaddvq_f32(vVal);
  }
#elif defined(__riscv_v_intrinsic)
  vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
  for (int avl; vI < pN; vI += avl) {
    avl = __riscv_vsetvl_e32m2(pN - vI);
    vfloat32m2_t vVal = ggml_v_expf_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&pX[vI], avl), pMax, avl), avl);
    __riscv_vse32_v_f32m2(&pY[vI], vVal, avl);
    vsum = __riscv_vfwredusum_vs_f32m2_f64m1(vVal, vsum, avl);
  }
  return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#endif
  for (; vI < pN; ++vI) {
    float vVal = expf(pX[vI] - pMax);
    vSum += (ggml_float)vVal;
    pY[vI] = vVal;
  }
  return vSum;
}

ggml_float ggml_vec_log_soft_max_f32(const int pN, float *pY, const float *pX, float pMax) {
  // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) -
  // log(soft_max_i)

  int vI = 0;
  ggml_float vSum = 0;
  for (; vI < pN; ++vI) {
    float vVal = pX[vI] - pMax;
    pY[vI] = vVal;
    vSum += (ggml_float)expf(vVal);
  }
  return vSum = (ggml_float)logf(vSum);
}
