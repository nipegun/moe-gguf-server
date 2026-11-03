#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined(__ARM_NEON__)
static constexpr int GEMM_RM = 4;
static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
static constexpr int GEMM_RM = 6;
static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
static constexpr int GEMM_RM = 2;
static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(float *GGML_RESTRICT pC, const float *GGML_RESTRICT pA,
                                     const float *GGML_RESTRICT pB, int pK, int pN) {
  static constexpr int cKn = GGML_F32_EPR;

  GGML_F32_VEC vAcc[RM][RN];
  for (int64_t vI = 0; vI < RM; vI++) {
    for (int vR = 0; vR < RN; vR++) {
      vAcc[vI][vR] = GGML_F32_VEC_LOAD(pC + vI * pN + vR * cKn);
    }
  }

  for (int64_t vKk = 0; vKk < pK; vKk++) {
    GGML_F32_VEC vBv[RN];
    for (int vR = 0; vR < RN; vR++) {
      vBv[vR] = GGML_F32_VEC_LOAD(pB + vKk * pN + vR * cKn);
    }
    for (int64_t vI = 0; vI < RM; vI++) {
      GGML_F32_VEC vP = GGML_F32_VEC_SET1(pA[vI * pK + vKk]);
      for (int vR = 0; vR < RN; vR++) {
        vAcc[vI][vR] = GGML_F32_VEC_FMA(vAcc[vI][vR], vBv[vR], vP);
      }
    }
  }

  for (int64_t vI = 0; vI < RM; vI++) {
    for (int vR = 0; vR < RN; vR++) {
      GGML_F32_VEC_STORE(pC + vI * pN + vR * cKn, vAcc[vI][vR]);
    }
  }
}

template <int RM>
static inline void simd_gemm_ukernel_tail(float *GGML_RESTRICT pC, const float *GGML_RESTRICT pA,
                                          const float *GGML_RESTRICT pB, int pK, int pN, int pCols) {
#if defined(__AVX512F__)
  const __mmask16 cMask = (1u << pCols) - 1;
  __m512 vAcc[RM];
  for (int64_t vI = 0; vI < RM; vI++) {
    vAcc[vI] = _mm512_maskz_loadu_ps(cMask, pC + vI * pN);
  }
  for (int64_t vKk = 0; vKk < pK; vKk++) {
    const __m512 cB = _mm512_maskz_loadu_ps(cMask, pB + vKk * pN);
    for (int64_t vI = 0; vI < RM; vI++) {
      vAcc[vI] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(pA[vI * pK + vKk]), cB, vAcc[vI], cMask);
    }
  }
  for (int64_t vI = 0; vI < RM; vI++) {
    _mm512_mask_storeu_ps(pC + vI * pN, cMask, vAcc[vI]);
  }
#elif defined(__AVX2__)
  const __m256i cMask = _mm256_cmpgt_epi32(_mm256_set1_epi32(pCols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
  __m256 vAcc[RM];
  for (int64_t vI = 0; vI < RM; vI++) {
    vAcc[vI] = _mm256_maskload_ps(pC + vI * pN, cMask);
  }
  for (int64_t vKk = 0; vKk < pK; vKk++) {
    const __m256 cB = _mm256_maskload_ps(pB + vKk * pN, cMask);
    for (int64_t vI = 0; vI < RM; vI++) {
      vAcc[vI] = GGML_F32_VEC_FMA(vAcc[vI], cB, _mm256_set1_ps(pA[vI * pK + vKk]));
    }
  }
  for (int64_t vI = 0; vI < RM; vI++) {
    _mm256_maskstore_ps(pC + vI * pN, cMask, vAcc[vI]);
  }
#else
  for (int64_t j = 0; j < pCols; j++) {
    for (int64_t vI = 0; vI < RM; vI++) {
      float a = pC[vI * pN + j];
      for (int64_t vKk = 0; vKk < pK; vKk++) {
        a += pA[vI * pK + vKk] * pB[vKk * pN + j];
      }
      pC[vI * pN + j] = a;
    }
  }
#endif
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(float *GGML_RESTRICT pC, const float *GGML_RESTRICT pA, const float *GGML_RESTRICT pB, int pM,
                      int pK, int pN) {
  static constexpr int cKn = GGML_F32_EPR;

  int64_t vIi = 0;
  for (; vIi + GEMM_RM <= pM; vIi += GEMM_RM) {
    int64_t vJj = 0;
    for (; vJj + GEMM_RN * cKn <= pN; vJj += GEMM_RN * cKn) {
      simd_gemm_ukernel<GEMM_RM, GEMM_RN>(pC + vJj, pA, pB + vJj, pK, pN);
    }
    for (; vJj + cKn <= pN; vJj += cKn) {
      simd_gemm_ukernel<GEMM_RM, 1>(pC + vJj, pA, pB + vJj, pK, pN);
    }
    if (vJj < pN) {
      simd_gemm_ukernel_tail<GEMM_RM>(pC + vJj, pA, pB + vJj, pK, pN, pN - vJj);
    }

    pA += GEMM_RM * pK;
    pC += GEMM_RM * pN;
  }

  // Tail rows: one at a time
  for (; vIi < pM; vIi++) {
    int64_t vJj = 0;
    for (; vJj + GEMM_RN * cKn <= pN; vJj += GEMM_RN * cKn) {
      simd_gemm_ukernel<1, GEMM_RN>(pC + vJj, pA, pB + vJj, pK, pN);
    }
    for (; vJj + cKn <= pN; vJj += cKn) {
      simd_gemm_ukernel<1, 1>(pC + vJj, pA, pB + vJj, pK, pN);
    }
    if (vJj < pN) {
      simd_gemm_ukernel_tail<1>(pC + vJj, pA, pB + vJj, pK, pN, pN - vJj);
    }

    pA += pK;
    pC += pN;
  }
}
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <int RM>
static inline void rvv_simd_gemm_ukernel(float *GGML_RESTRICT C, const float *GGML_RESTRICT A,
                                         const float *GGML_RESTRICT B, int K, int N, size_t vl) {
  static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

  vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
  vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
  if constexpr (RM > 1)
    acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
  if constexpr (RM > 2)
    acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
  if constexpr (RM > 3)
    acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
  if constexpr (RM > 4)
    acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
  if constexpr (RM > 5)
    acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
  if constexpr (RM > 6)
    acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

  for (int kk = 0; kk < K; kk++) {
    vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

    acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
    if constexpr (RM > 1)
      acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
    if constexpr (RM > 2)
      acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
    if constexpr (RM > 3)
      acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
    if constexpr (RM > 4)
      acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
    if constexpr (RM > 5)
      acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
    if constexpr (RM > 6)
      acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
  }

  __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
  if constexpr (RM > 1)
    __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
  if constexpr (RM > 2)
    __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
  if constexpr (RM > 3)
    __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
  if constexpr (RM > 4)
    __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
  if constexpr (RM > 5)
    __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
  if constexpr (RM > 6)
    __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <int RM>
static inline void rvv_simd_gemm_dispatch_tail(float *GGML_RESTRICT C, const float *GGML_RESTRICT A,
                                               const float *GGML_RESTRICT B, int K, int N, int KN, int remaining_rows) {
  if constexpr (RM > 0) {
    if (remaining_rows == RM) {
      int64_t jj = 0;
      for (; jj + KN <= N; jj += KN) {
        rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
      }
      if (jj < N) {
        rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
      }
    } else {
      rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
    }
  }
}

static constexpr int GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(float *GGML_RESTRICT C, const float *GGML_RESTRICT A, const float *GGML_RESTRICT B, int M, int K,
                      int N) {
  const int KN = (int)__riscv_vlenb();
  int64_t ii = 0;
  for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
    int64_t jj = 0;
    for (; jj + KN <= N; jj += KN) {
      rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
    }
    if (jj < N) {
      rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
    }
    A += GEMM_RM * K;
    C += GEMM_RM * N;
  }

  int remaining_rows = M - ii;
  rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else // scalar path

static void simd_gemm(float *GGML_RESTRICT C, const float *GGML_RESTRICT A, const float *GGML_RESTRICT B, int M, int K,
                      int N) {
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      float sum = C[i * N + j];
      for (int64_t kk = 0; kk < K; kk++) {
        sum += A[i * K + kk] * B[kk * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

#endif // GGML_SIMD
