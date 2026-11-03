#include "rvv_kernels.h"

#include "common.h"
#include "ggml.h"
#include "ops.h"
#include "string.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#error "riscv zfh extension not enabled"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace spacemit_kernels::rvv {

namespace {

auto fAlignUp(size_t pValue, size_t pAlignment) { return (pValue + pAlignment - 1) / pAlignment * pAlignment; }

static inline bool fFlashAttnExtSupportedDVlen1024Vf16(int64_t pD) { return pD > 0 && pD <= 128; }

static inline bool fFlashAttnExtSupportedShapeVlen1024Vf16(int64_t pDk, int64_t pDv) {
  return fFlashAttnExtSupportedDVlen1024Vf16(pDk) && fFlashAttnExtSupportedDVlen1024Vf16(pDv);
}

static inline float fReduceSumF32m4Vlen1024(vfloat32m4_t pV, size_t pVl) {
  vfloat32m1_t vSV = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vSV = __riscv_vfredusum_vs_f32m4_f32m1(pV, vSV, pVl);
  return __riscv_vfmv_f_s_f32m1_f32(vSV);
}

static inline float fReduceSumF32m2Vlen1024(vfloat32m2_t pV, size_t pVl) {
  vfloat32m1_t vSV = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vSV = __riscv_vfredusum_vs_f32m2_f32m1(pV, vSV, pVl);
  return __riscv_vfmv_f_s_f32m1_f32(vSV);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
static inline vfloat32m2_t fRvvExpfApproxF32m2(vfloat32m2_t pX, size_t pVl) {
  const vfloat32m2_t cR = __riscv_vfmv_v_f_f32m2(0x1.8p23f, pVl);
  const vfloat32m2_t cZ = __riscv_vfmacc_vf_f32m2(cR, 0x1.715476p+0f, pX, pVl);
  const vfloat32m2_t cN = __riscv_vfsub_vv_f32m2(cZ, cR, pVl);
  const vfloat32m2_t cB =
      __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(pX, 0x1.62e4p-1f, cN, pVl), 0x1.7f7d1cp-20f, cN, pVl);
  const vuint32m2_t cE = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(cZ), 23, pVl);
  const vfloat32m2_t cK = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(cE, 0x3f800000, pVl));
  const vbool16_t c_ = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(cN, pVl), 126.0f, pVl);
  const vfloat32m2_t cU = __riscv_vfmul_vv_f32m2(cB, cB, pVl);
  const vfloat32m2_t cJ = __riscv_vfmacc_vv_f32m2(
      __riscv_vfmul_vf_f32m2(cB, 0x1.ffffecp-1f, pVl),
      __riscv_vfmacc_vv_f32m2(
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, pVl), 0x1.555e66p-3f, cB, pVl),
          __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, pVl), 0x1.0e4020p-7f, cB, pVl), cU, pVl),
      cU, pVl);

  if (!__riscv_vcpop_m_b16(c_, pVl)) {
    return __riscv_vfmacc_vv_f32m2(cK, cJ, cK, pVl);
  }

  const vbool16_t cDm = __riscv_vmfle_vf_f32m2_b16(cN, 0.0f, pVl);
  const vuint32m2_t cD = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, pVl), 0x82000000, cDm, pVl);
  const vfloat32m2_t cS1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(cD, 0x7f000000, pVl));
  const vfloat32m2_t cS2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(cE, cD, pVl));
  const vfloat32m2_t cR1 =
      __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(cK, cK, cJ, pVl),
                               __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(cS2, cS2, cJ, pVl), cS1, pVl), c_, pVl);
  return __riscv_vmerge_vvm_f32m2(cR1, __riscv_vfmul_vv_f32m2(cS1, cS1, pVl),
                                  __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(cN, pVl), 192.0f, pVl), pVl);
}

static inline vfloat32m2_t fRvvTanhApproxF32m2(vfloat32m2_t pX, size_t pVl) {
  const vfloat32m2_t cAbsX = __riscv_vfabs_v_f32m2(pX, pVl);
  const vfloat32m2_t cNeg2Abs = __riscv_vfmul_vf_f32m2(cAbsX, -2.0f, pVl);
  const vfloat32m2_t cExpTerm = fRvvExpfApproxF32m2(cNeg2Abs, pVl);
  const vfloat32m2_t cNumerator = __riscv_vfsub_vf_f32m2(cExpTerm, 1.0f, pVl);
  const vfloat32m2_t cDenominator = __riscv_vfadd_vf_f32m2(cExpTerm, 1.0f, pVl);
  const vfloat32m2_t cTanhAbs = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(cNumerator, cDenominator, pVl), pVl);
  const vbool16_t cNegMask = __riscv_vmflt_vf_f32m2_b16(pX, 0.0f, pVl);
  const vfloat32m2_t cTanhNeg = __riscv_vfneg_v_f32m2(cTanhAbs, pVl);
  return __riscv_vmerge_vvm_f32m2(cTanhAbs, cTanhNeg, cNegMask, pVl);
}

static void fRvvSoftcapTanhInplaceF32(float *pDst, int64_t pDstStride, int64_t pTileRows, int64_t pN, float pSoftcap) {
  for (int vTq = 0; vTq < pTileRows; ++vTq, pDst += pDstStride) {
    float *vDstRow = pDst;
    int64_t vRemaining = pN;
    while (vRemaining > 0) {
      const size_t cVl = __riscv_vsetvl_e32m2(vRemaining);
      vfloat32m2_t v_ = __riscv_vle32_v_f32m2(vDstRow, cVl);
      v_ = fRvvTanhApproxF32m2(v_, cVl);
      v_ = __riscv_vfmul_vf_f32m2(v_, pSoftcap, cVl);
      __riscv_vse32_v_f32m2(vDstRow, v_, cVl);
      vDstRow += cVl;
      vRemaining -= cVl;
    }
  }
}

static inline float fRvvSoftmaxExpInplaceF32(float *pDst, int64_t pN, float pMaxValue) {
  float vRowSum = 0.0f;
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m2(pN);
    vfloat32m2_t v_ = __riscv_vle32_v_f32m2(pDst, cVl);
    v_ = __riscv_vfsub_vf_f32m2(v_, pMaxValue, cVl);
    v_ = fRvvExpfApproxF32m2(v_, cVl);
    __riscv_vse32_v_f32m2(pDst, v_, cVl);
    vRowSum += fReduceSumF32m2Vlen1024(v_, cVl);
    pDst += cVl;
    pN -= cVl;
  }
  return vRowSum;
}

static inline float fRvvAddMaxInplaceF32(float *pDst, const float *pSrc, int64_t pN) {
  float vMaxVal = -INFINITY;
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m4(pN);
    vfloat32m4_t vVdst = __riscv_vle32_v_f32m4(pDst, cVl);
    vfloat32m4_t vVsrc = __riscv_vle32_v_f32m4(pSrc, cVl);
    vVdst = __riscv_vfadd_vv_f32m4(vVdst, vVsrc, cVl);
    __riscv_vse32_v_f32m4(pDst, vVdst, cVl);

    vfloat32m1_t vSeed = __riscv_vfmv_v_f_f32m1(vMaxVal, 1);
    vSeed = __riscv_vfredmax_vs_f32m4_f32m1(vVdst, vSeed, cVl);
    vMaxVal = __riscv_vfmv_f_s_f32m1_f32(vSeed);

    pDst += cVl;
    pSrc += cVl;
    pN -= cVl;
  }
  return vMaxVal;
}

static inline float fRvvSoftcapAddMaxInplaceF32(float *pDst, const float *pSrc, int64_t pN, float pSoftcap) {
  if (pSoftcap == 0.0f) {
    return fRvvAddMaxInplaceF32(pDst, pSrc, pN);
  }

  float vMaxVal = -INFINITY;
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m2(pN);
    vfloat32m2_t vVdst = __riscv_vle32_v_f32m2(pDst, cVl);
    vfloat32m2_t vVsrc = __riscv_vle32_v_f32m2(pSrc, cVl);
    vVdst = fRvvTanhApproxF32m2(vVdst, cVl);
    vVdst = __riscv_vfmul_vf_f32m2(vVdst, pSoftcap, cVl);
    vVdst = __riscv_vfadd_vv_f32m2(vVdst, vVsrc, cVl);
    __riscv_vse32_v_f32m2(pDst, vVdst, cVl);

    vfloat32m1_t vSeed = __riscv_vfmv_v_f_f32m1(vMaxVal, 1);
    vSeed = __riscv_vfredmax_vs_f32m2_f32m1(vVdst, vSeed, cVl);
    vMaxVal = __riscv_vfmv_f_s_f32m1_f32(vSeed);

    pDst += cVl;
    pSrc += cVl;
    pN -= cVl;
  }
  return vMaxVal;
}

static inline void fRvvZeroF32(float *pDst, int64_t pN) {
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m4(pN);
    const vfloat32m4_t cZ = __riscv_vfmv_v_f_f32m4(0.0f, cVl);
    __riscv_vse32_v_f32m4(pDst, cZ, cVl);
    pDst += cVl;
    pN -= cVl;
  }
}

static inline void fRvvScaleF32(float *pDst, float pScale, int64_t pN) {
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m4(pN);
    vfloat32m4_t v_ = __riscv_vle32_v_f32m4(pDst, cVl);
    v_ = __riscv_vfmul_vf_f32m4(v_, pScale, cVl);
    __riscv_vse32_v_f32m4(pDst, v_, cVl);
    pDst += cVl;
    pN -= cVl;
  }
}

static inline void fRvvAddInplaceF32(float *pDst, int64_t pDstStride, const float *pSrc, int64_t pSrcStride,
                                     int64_t pTileRows, int64_t pN) {
  for (int vTq = 0; vTq < pTileRows; ++vTq, pDst += pDstStride, pSrc += pSrcStride) {
    int64_t vRemaining = pN;
    float *vDstRow = pDst;
    const float *vSrcRow = pSrc;
    while (vRemaining > 0) {
      const size_t cVl = __riscv_vsetvl_e32m4(vRemaining);
      vfloat32m4_t vVdst = __riscv_vle32_v_f32m4(vDstRow, cVl);
      vfloat32m4_t vVsrc = __riscv_vle32_v_f32m4(vSrcRow, cVl);
      vVdst = __riscv_vfadd_vv_f32m4(vVdst, vVsrc, cVl);
      __riscv_vse32_v_f32m4(vDstRow, vVdst, cVl);
      vDstRow += cVl;
      vSrcRow += cVl;
      vRemaining -= cVl;
    }
  }
}

static inline float fRvvMaxF32(const float *pSrc, int64_t pN) {
  float vMaxVal = -INFINITY;
  while (pN > 0) {
    const size_t cVl = __riscv_vsetvl_e32m4(pN);
    const vfloat32m4_t cV = __riscv_vle32_v_f32m4(pSrc, cVl);
    vfloat32m1_t vSeed = __riscv_vfmv_v_f_f32m1(vMaxVal, 1);
    vSeed = __riscv_vfredmax_vs_f32m4_f32m1(cV, vSeed, cVl);
    vMaxVal = __riscv_vfmv_f_s_f32m1_f32(vSeed);
    pSrc += cVl;
    pN -= cVl;
  }
  return vMaxVal;
}

static void fRvvPackF32AsScaledF16(void *pDst, int64_t pDstRowStride, const void *pSrc, int64_t pSrcRowStride,
                                   int64_t pTileRows, int64_t pN, float pScale) {
  for (int vTq = 0; vTq < pTileRows; ++vTq) {
    const float *vRowPtr = (const float *)((const char *)pSrc + vTq * pSrcRowStride);
    _Float16 *vDstRowPtr = (_Float16 *)((char *)pDst + vTq * pDstRowStride);
    int64_t vRemaining = pN;
    while (vRemaining > 0) {
      const size_t cVl = __riscv_vsetvl_e32m4(vRemaining);
      vfloat32m4_t v32 = __riscv_vle32_v_f32m4(vRowPtr, cVl);
      v32 = __riscv_vfmul_vf_f32m4(v32, pScale, cVl);
      const vfloat16m2_t cV16 = __riscv_vfncvt_f_f_w_f16m2(v32, cVl);
      __riscv_vse16_v_f16m2(vDstRowPtr, cV16, cVl);
      vDstRowPtr += cVl;
      vRowPtr += cVl;
      vRemaining -= cVl;
    }
  }
}

static void fRvvPackScaledF16AsF32(void *pDst, int64_t pDstRowStride, const void *pSrc, int64_t pSrcRowStride,
                                   int64_t pTileRows, int64_t pN, float pScale) {
  for (int vTq = 0; vTq < pTileRows; ++vTq) {
    const _Float16 *vRowPtr = (const _Float16 *)((const char *)pSrc + vTq * pSrcRowStride);
    float *vDstRowPtr = (float *)((char *)pDst + vTq * pDstRowStride);
    int64_t vRemaining = pN;
    while (vRemaining > 0) {
      const size_t cVl = __riscv_vsetvl_e16m2(vRemaining);
      const vfloat16m2_t cV16 = __riscv_vle16_v_f16m2(vRowPtr, cVl);
      vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(cV16, cVl);
      v32 = __riscv_vfmul_vf_f32m4(v32, pScale, cVl);
      __riscv_vse32_v_f32m4(vDstRowPtr, v32, cVl);
      vDstRowPtr += cVl;
      vRowPtr += cVl;
      vRemaining -= cVl;
    }
  }
}

static void fRvvPackScaledF32AsF32(void *pDst, int64_t pDstRowStride, const void *pSrc, int64_t pSrcRowStride,
                                   int64_t pTileRows, int64_t pN, float *pScale) {
  for (int vTq = 0; vTq < pTileRows; ++vTq) {
    const float *vRowPtr = (const float *)((const char *)pSrc + vTq * pSrcRowStride);
    float *vDstRowPtr = (float *)((char *)pDst + vTq * pDstRowStride);
    int64_t vRemaining = pN;
    while (vRemaining > 0) {
      const size_t cVl = __riscv_vsetvl_e32m4(vRemaining);
      vfloat32m4_t v32 = __riscv_vle32_v_f32m4(vRowPtr, cVl);
      v32 = __riscv_vfmul_vf_f32m4(v32, pScale[vTq], cVl);
      __riscv_vse32_v_f32m4(vDstRowPtr, v32, cVl);
      vDstRowPtr += cVl;
      vRowPtr += cVl;
      vRemaining -= cVl;
    }
  }
}

static inline void fRvvTransposedS32MnToNm(int8_t *dst, int64_t pNDstStride, int8_t *src, int64_t pMSrcStride,
                                           int64_t pM, int64_t pN) {
  int8_t *vIn = src;
  int8_t *vOut = dst;

  __asm__ volatile("vsetvli                t0, zero, e32, m1, tu, mu     \n\t"
                   "mul                    t3, t0, %[os0]                \n\t"
                   "srli                   t2, %[isz0], 3                \n\t"
                   "blez                   t2, M1%=                      \n\t"

                   "LOOP_M8%=:                                           \n\t"
                   "addi                   a1, %[dst], 0                 \n\t"
                   "addi                   s1, %[src], 0                 \n\t"
                   "add                    s2, %[src], %[is0]            \n\t"
                   "add                    s3, s2, %[is0]                \n\t"
                   "add                    s4, s3, %[is0]                \n\t"
                   "add                    s5, s4, %[is0]                \n\t"
                   "add                    s6, s5, %[is0]                \n\t"
                   "add                    s7, s6, %[is0]                \n\t"
                   "add                    s8, s7, %[is0]                \n\t"
                   "addi                   t1, %[isz1], 0                \n\t"

                   "LOOP_M8N%=:                                          \n\t"
                   "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
                   "sub                    t1, t1, t0                    \n\t"
                   "vle32.v                v0, (s1)                      \n\t"
                   "sh2add                 s1, t0, s1                    \n\t"
                   "vle32.v                v1, (s2)                      \n\t"
                   "sh2add                 s2, t0, s2                    \n\t"
                   "vle32.v                v2, (s3)                      \n\t"
                   "sh2add                 s3, t0, s3                    \n\t"
                   "vle32.v                v3, (s4)                      \n\t"
                   "sh2add                 s4, t0, s4                    \n\t"
                   "vle32.v                v4, (s5)                      \n\t"
                   "sh2add                 s5, t0, s5                    \n\t"
                   "vle32.v                v5, (s6)                      \n\t"
                   "sh2add                 s6, t0, s6                    \n\t"
                   "vle32.v                v6, (s7)                      \n\t"
                   "sh2add                 s7, t0, s7                    \n\t"
                   "vle32.v                v7, (s8)                      \n\t"
                   "sh2add                 s8, t0, s8                    \n\t"
                   "vssseg8e32.v           v0, (a1), %[os0]              \n\t"
                   "add                    a1, a1, t3                    \n\t"
                   "bnez                   t1, LOOP_M8N%=                \n\t"
                   "sh3add                 %[src], %[is0], %[src]        \n\t"
                   "addi                   %[dst], %[dst], 32            \n\t"
                   "addi                   t2, t2, -1                    \n\t"
                   "bnez                   t2, LOOP_M8%=                 \n\t"

                   "M1%=:                                                \n\t"
                   "andi                   t2, %[isz0], 7                \n\t"
                   "blez                   t2, END%=                     \n\t"

                   "LOOP_M1%=:                                           \n\t"
                   "addi                   a1, %[dst], 0                 \n\t"
                   "addi                   s1, %[src], 0                 \n\t"
                   "addi                   t1, %[isz1], 0                \n\t"

                   "LOOP_M1N%=:                                          \n\t"
                   "vsetvli                t0, t1, e32, m1, tu, mu       \n\t"
                   "sub                    t1, t1, t0                    \n\t"
                   "vle32.v                v0, (s1)                      \n\t"
                   "sh2add                 s1, t0, s1                    \n\t"
                   "vsse32.v               v0, (a1), %[os0]              \n\t"
                   "add                    a1, a1, t3                    \n\t"
                   "bnez                   t1, LOOP_M1N%=                \n\t"
                   "add                    %[src], %[is0], %[src]        \n\t"
                   "addi                   %[dst], %[dst], 4             \n\t"
                   "addi                   t2, t2, -1                    \n\t"
                   "bnez                   t2, LOOP_M1%=                 \n\t"
                   "END%=:                                               \n\t"

                   : [src] "+r"(vIn), [dst] "+r"(vOut), [isz0] "+r"(pM)
                   : [isz1] "r"(pN), [is0] "r"(pMSrcStride), [os0] "r"(pNDstStride)
                   : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

static inline void fRvvTransposedS16MnToNm(int8_t *dst, int64_t pNDstStride, int8_t *src, int64_t pMSrcStride,
                                           int64_t pM, int64_t pN) {
  int8_t *vIn = src;
  int8_t *vOut = dst;

  __asm__ volatile("vsetvli                t0, zero, e16, m1, tu, mu     \n\t"
                   "mul                    t3, t0, %[os0]                \n\t"
                   "srli                   t2, %[isz0], 3                \n\t"
                   "blez                   t2, M1%=                      \n\t"

                   "LOOP_M8%=:                                           \n\t"
                   "addi                   a1, %[dst], 0                 \n\t"
                   "addi                   s1, %[src], 0                 \n\t"
                   "add                    s2, %[src], %[is0]            \n\t"
                   "add                    s3, s2, %[is0]                \n\t"
                   "add                    s4, s3, %[is0]                \n\t"
                   "add                    s5, s4, %[is0]                \n\t"
                   "add                    s6, s5, %[is0]                \n\t"
                   "add                    s7, s6, %[is0]                \n\t"
                   "add                    s8, s7, %[is0]                \n\t"
                   "addi                   t1, %[isz1], 0                \n\t"

                   "LOOP_M8N%=:                                          \n\t"
                   "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
                   "sub                    t1, t1, t0                    \n\t"
                   "vle16.v                v0, (s1)                      \n\t"
                   "sh1add                 s1, t0, s1                    \n\t"
                   "vle16.v                v1, (s2)                      \n\t"
                   "sh1add                 s2, t0, s2                    \n\t"
                   "vle16.v                v2, (s3)                      \n\t"
                   "sh1add                 s3, t0, s3                    \n\t"
                   "vle16.v                v3, (s4)                      \n\t"
                   "sh1add                 s4, t0, s4                    \n\t"
                   "vle16.v                v4, (s5)                      \n\t"
                   "sh1add                 s5, t0, s5                    \n\t"
                   "vle16.v                v5, (s6)                      \n\t"
                   "sh1add                 s6, t0, s6                    \n\t"
                   "vle16.v                v6, (s7)                      \n\t"
                   "sh1add                 s7, t0, s7                    \n\t"
                   "vle16.v                v7, (s8)                      \n\t"
                   "sh1add                 s8, t0, s8                    \n\t"
                   "vssseg8e16.v           v0, (a1), %[os0]              \n\t"
                   "add                    a1, a1, t3                    \n\t"
                   "bnez                   t1, LOOP_M8N%=                \n\t"
                   "sh3add                 %[src], %[is0], %[src]        \n\t"
                   "addi                   %[dst], %[dst], 16            \n\t"
                   "addi                   t2, t2, -1                    \n\t"
                   "bnez                   t2, LOOP_M8%=                 \n\t"

                   "M1%=:                                                \n\t"
                   "andi                   t2, %[isz0], 7                \n\t"
                   "blez                   t2, END%=                     \n\t"

                   "LOOP_M1%=:                                           \n\t"
                   "addi                   a1, %[dst], 0                 \n\t"
                   "addi                   s1, %[src], 0                 \n\t"
                   "addi                   t1, %[isz1], 0                \n\t"

                   "LOOP_M1N%=:                                          \n\t"
                   "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
                   "sub                    t1, t1, t0                    \n\t"
                   "vle16.v                v0, (s1)                      \n\t"
                   "sh1add                 s1, t0, s1                    \n\t"
                   "vsse16.v               v0, (a1), %[os0]              \n\t"
                   "add                    a1, a1, t3                    \n\t"
                   "bnez                   t1, LOOP_M1N%=                \n\t"
                   "add                    %[src], %[is0], %[src]        \n\t"
                   "addi                   %[dst], %[dst], 2             \n\t"
                   "addi                   t2, t2, -1                    \n\t"
                   "bnez                   t2, LOOP_M1%=                 \n\t"
                   "END%=:                                               \n\t"

                   : [src] "+r"(vIn), [dst] "+r"(vOut), [isz0] "+r"(pM)
                   : [isz1] "r"(pN), [is0] "r"(pMSrcStride), [os0] "r"(pNDstStride)
                   : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1");
}

static inline void fRvvQkDotTileF16X1(float *pDst, const _Float16 *pQRow, const _Float16 *pKPack, int64_t pDk,
                                      int64_t pKvTile) {
  const size_t cVl = __riscv_vsetvl_e16m1(pKvTile);
  vfloat32m2_t vAcc = __riscv_vfmv_v_f_f32m2(0.0f, cVl);

  for (int64_t vD = 0; vD < pDk; ++vD) {
    const vfloat16m1_t cKVec = __riscv_vle16_v_f16m1(pKPack + vD * ggml_fa_tile_config::KV, cVl);
    vAcc = __riscv_vfwmacc_vf_f32m2(vAcc, pQRow[vD], cKVec, cVl);
  }

  __riscv_vse32_v_f32m2(pDst, vAcc, cVl);
}

static inline void fRvvQkDotTileF16X4(float *pDst0, float *pDst1, float *pDst2, float *pDst3, const _Float16 *pQ0,
                                      const _Float16 *pQ1, const _Float16 *pQ2, const _Float16 *pQ3,
                                      const _Float16 *pKPack, int64_t pDk, int64_t pKvTile) {
  const size_t cVl = __riscv_vsetvl_e16m1(pKvTile);
  vfloat32m2_t vAcc0 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
  vfloat32m2_t vAcc1 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
  vfloat32m2_t vAcc2 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
  vfloat32m2_t vAcc3 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);

  for (int64_t vD = 0; vD < pDk; ++vD) {
    const vfloat16m1_t cKVec = __riscv_vle16_v_f16m1(pKPack + vD * ggml_fa_tile_config::KV, cVl);
    vAcc0 = __riscv_vfwmacc_vf_f32m2(vAcc0, pQ0[vD], cKVec, cVl);
    vAcc1 = __riscv_vfwmacc_vf_f32m2(vAcc1, pQ1[vD], cKVec, cVl);
    vAcc2 = __riscv_vfwmacc_vf_f32m2(vAcc2, pQ2[vD], cKVec, cVl);
    vAcc3 = __riscv_vfwmacc_vf_f32m2(vAcc3, pQ3[vD], cKVec, cVl);
  }

  __riscv_vse32_v_f32m2(pDst0, vAcc0, cVl);
  __riscv_vse32_v_f32m2(pDst1, vAcc1, cVl);
  __riscv_vse32_v_f32m2(pDst2, vAcc2, cVl);
  __riscv_vse32_v_f32m2(pDst3, vAcc3, cVl);
}

static inline void fRvvPvAccumulateF16X1(float *pDst, const float *pProb, const _Float16 *pVPack, int64_t pKvTile,
                                         int64_t pDv) {
  int64_t vDLeft = pDv;
  int64_t vDOff = 0;

  while (vDLeft > 0) {
    const size_t cVl = __riscv_vsetvl_e16m2(vDLeft);
    vfloat32m4_t vAcc = __riscv_vle32_v_f32m4(pDst + vDOff, cVl);

    for (int64_t vTk = 0; vTk < pKvTile; ++vTk) {
      const vfloat16m2_t cV16 = __riscv_vle16_v_f16m2(pVPack + vTk * pDv + vDOff, cVl);
      const vfloat32m4_t cV32 = __riscv_vfwcvt_f_f_v_f32m4(cV16, cVl);
      vAcc = __riscv_vfmacc_vf_f32m4(vAcc, pProb[vTk], cV32, cVl);
    }

    __riscv_vse32_v_f32m4(pDst + vDOff, vAcc, cVl);
    vDLeft -= cVl;
    vDOff += cVl;
  }
}

static inline void fRvvPvAccumulateF16X4(float *pDst0, float *pDst1, float *pDst2, float *pDst3, const float *pProb0,
                                         const float *pProb1, const float *pProb2, const float *pProb3,
                                         const _Float16 *pVPack, int64_t pKvTile, int64_t pDv) {
  int64_t vDLeft = pDv;
  int64_t vDOff = 0;

  while (vDLeft > 0) {
    const size_t cVl = __riscv_vsetvl_e16m2(vDLeft);
    vfloat32m4_t vAcc0 = __riscv_vle32_v_f32m4(pDst0 + vDOff, cVl);
    vfloat32m4_t vAcc1 = __riscv_vle32_v_f32m4(pDst1 + vDOff, cVl);
    vfloat32m4_t vAcc2 = __riscv_vle32_v_f32m4(pDst2 + vDOff, cVl);
    vfloat32m4_t vAcc3 = __riscv_vle32_v_f32m4(pDst3 + vDOff, cVl);

    for (int64_t vTk = 0; vTk < pKvTile; ++vTk) {
      const vfloat16m2_t cV16 = __riscv_vle16_v_f16m2(pVPack + vTk * pDv + vDOff, cVl);
      const vfloat32m4_t cV32 = __riscv_vfwcvt_f_f_v_f32m4(cV16, cVl);
      vAcc0 = __riscv_vfmacc_vf_f32m4(vAcc0, pProb0[vTk], cV32, cVl);
      vAcc1 = __riscv_vfmacc_vf_f32m4(vAcc1, pProb1[vTk], cV32, cVl);
      vAcc2 = __riscv_vfmacc_vf_f32m4(vAcc2, pProb2[vTk], cV32, cVl);
      vAcc3 = __riscv_vfmacc_vf_f32m4(vAcc3, pProb3[vTk], cV32, cVl);
    }

    __riscv_vse32_v_f32m4(pDst0 + vDOff, vAcc0, cVl);
    __riscv_vse32_v_f32m4(pDst1 + vDOff, vAcc1, cVl);
    __riscv_vse32_v_f32m4(pDst2 + vDOff, vAcc2, cVl);
    __riscv_vse32_v_f32m4(pDst3 + vDOff, vAcc3, cVl);
    vDLeft -= cVl;
    vDOff += cVl;
  }
}

static inline void fRvvQkDotTile(float *pDst, const float *pQRow, const float *pKPack, int64_t pDk, int64_t pKvTile,
                                 float pScale) {
  const size_t cVl = __riscv_vsetvl_e32m4(pKvTile);
  vfloat32m4_t vAcc = __riscv_vfmv_v_f_f32m4(0.0f, cVl);

  for (int64_t vD = 0; vD < pDk; ++vD) {
    const vfloat32m4_t cKVec = __riscv_vle32_v_f32m4(pKPack + vD * pKvTile, cVl);
    vAcc = __riscv_vfmacc_vf_f32m4(vAcc, pQRow[vD] * pScale, cKVec, cVl);
  }

  __riscv_vse32_v_f32m4(pDst, vAcc, cVl);
}

static inline void fRvvPvAccumulate(float *pDst, const float *pProb, const float *pVPack, int64_t pKvTile,
                                    int64_t pDv) {
  int64_t vDLeft = pDv;
  int64_t vDOff = 0;

  while (vDLeft > 0) {
    const size_t cVl = __riscv_vsetvl_e32m4(vDLeft);
    vfloat32m4_t vAcc = __riscv_vle32_v_f32m4(pDst + vDOff, cVl);

    for (int64_t vTk = 0; vTk < pKvTile; ++vTk) {
      const vfloat32m4_t cVVec = __riscv_vle32_v_f32m4(pVPack + vTk * pDv + vDOff, cVl);
      vAcc = __riscv_vfmacc_vf_f32m4(vAcc, pProb[vTk], cVVec, cVl);
    }

    __riscv_vse32_v_f32m4(pDst + vDOff, vAcc, cVl);
    vDLeft -= cVl;
    vDOff += cVl;
  }
}

static void fPermuteTransposeImpl(const ggml_tensor *pSrc0, ggml_tensor *pDst, int64_t pBatch, int64_t pM, int64_t pN,
                                  int64_t pBatchStride, int64_t pMSrcStride, int64_t pNSrcStride, int64_t pNDstStride,
                                  int pIth, int pNth) {
  GGML_ASSERT(pNSrcStride == sizeof(int32_t) || pNSrcStride == sizeof(int16_t));

  if (pNSrcStride == sizeof(int32_t)) {
    for (int64_t vBi = pIth; vBi < pBatch; vBi += pNth) {
      fRvvTransposedS32MnToNm((int8_t *)((char *)pDst->data + vBi * pBatchStride), pNDstStride,
                              (int8_t *)((char *)pSrc0->data + vBi * pBatchStride), pMSrcStride, pM, pN);
    }
  } else if (pNSrcStride == sizeof(int16_t)) {
    for (int64_t vBi = pIth; vBi < pBatch; vBi += pNth) {
      fRvvTransposedS32MnToNm((int8_t *)((char *)pDst->data + vBi * pBatchStride), pNDstStride,
                              (int8_t *)((char *)pSrc0->data + vBi * pBatchStride), pMSrcStride, pM, pN);
    }
  } else {
    GGML_ABORT("not implemented");
  }
}

template <size_t QLEN>
static void fFlashAttnExtF16OneChunkInnerVlen1024Vf16Mrow(float **pPq, const char *pKDataRow, const char *pVDataRow,
                                                          const ggml_fp16_t *pMp, float **pSinks, float **pDst,
                                                          float pScale, float pLogitSoftcap, float pSlope,
                                                          int64_t pNek1, int64_t pNbk1, int64_t pNbv1, int64_t pDv,
                                                          int64_t pDk, void *pTcmBuffer, size_t pTcmBufferSize) {
  GGML_ASSERT(fFlashAttnExtSupportedShapeVlen1024Vf16(pDk, pDv));
  float S[QLEN] = {0.0f};       // sum
  float aM[QLEN] = {-INFINITY}; // maximum KQ value

  _Float16 *vKq16Buffer = (_Float16 *)pTcmBuffer;
  _Float16 *vQvBuffer = vKq16Buffer + QLEN * pDv;
  const size_t cQkvTempBufferSize = (QLEN * pDv + QLEN * pDk) * sizeof(_Float16);
  char *vKvTileBuffer = (char *)(vQvBuffer + QLEN * pDk);

  {
    vfloat16m2_t vVkQ16V = __riscv_vfmv_v_f_f16m2(0.0f, pDv);
    for (int64_t vI = 0; vI < QLEN; ++vI) {
      __riscv_vse16_v_f16m2(vKq16Buffer + vI * pDv, vVkQ16V, pDv);
      vfloat16m2_t vQQV = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pPq[vI], pDk), pDk);
      __riscv_vse16_v_f16m2(vQvBuffer + vI * pDk, vQQV, pDk);
    }
  }

  const uintptr_t cScratchAddr = reinterpret_cast<uintptr_t>(vKvTileBuffer);
  const size_t cScratchSize = pTcmBufferSize > cQkvTempBufferSize ? pTcmBufferSize - cQkvTempBufferSize : 0;
  const uintptr_t cKqTileAddr = fAlignUp(cScratchAddr, alignof(float));
  const size_t cScratchPrefix = cKqTileAddr - cScratchAddr;
  const size_t cPackedTileSize = QLEN * sizeof(float) + pDk * sizeof(_Float16) + pDv * sizeof(_Float16) + sizeof(float);
  const int64_t cMaxIcTileStep = ((int64_t)__riscv_vsetvlmax_e16m1()) & ~((int64_t)7);
  const int64_t cMaxFitByTcm =
      cScratchSize > cScratchPrefix ? (int64_t)((cScratchSize - cScratchPrefix) / cPackedTileSize) : 0;
  const int64_t cIcTileStep = std::min(cMaxIcTileStep, cMaxFitByTcm) & ~((int64_t)7);

  const uintptr_t cKTileAddr = cKqTileAddr + QLEN * cIcTileStep * sizeof(float);
  const uintptr_t cVTileAddr = cKTileAddr + pDk * cIcTileStep * sizeof(_Float16);
  const uintptr_t cMvTileAddr = cVTileAddr + cIcTileStep * pDv * sizeof(_Float16);

  if (cIcTileStep >= 8) {
    float *vKqTileBuffer = reinterpret_cast<float *>(cKqTileAddr);
    _Float16 *vKTilePack = reinterpret_cast<_Float16 *>(cKTileAddr);
    _Float16 *vVTilePack = reinterpret_cast<_Float16 *>(cVTileAddr);
    float *vMvTilePack = reinterpret_cast<float *>(cMvTileAddr);

    const int64_t cKTileByteStride = cIcTileStep * (int64_t)sizeof(_Float16);

    int64_t vIcStep = 0;
    for (int64_t vIc = 0; vIc < pNek1; ++vIc) {
      const float cMv = pMp ? pSlope * ((_Float16 *)pMp)[vIc] : 0.0f;

      if (cMv != -INFINITY) {
        const _Float16 *vKData = (const _Float16 *)(pKDataRow + vIc * pNbk1);
        const _Float16 *vVData = (const _Float16 *)(pVDataRow + vIc * pNbv1);

        const vfloat16m2_t cKDataV = __riscv_vle16_v_f16m2(vKData, pDk);
        const vfloat16m2_t cVDataV = __riscv_vle16_v_f16m2(vVData, pDv);
        __riscv_vsse16_v_f16m2(vKTilePack + vIcStep, cKTileByteStride, cKDataV, pDk);
        __riscv_vse16_v_f16m2(vVTilePack + vIcStep * pDv, cVDataV, pDv);
        vMvTilePack[vIcStep] = cMv;
        vIcStep++;
      }

      if (vIcStep > 0 && (vIcStep == cIcTileStep || vIc == (pNek1 - 1))) {
        if constexpr (QLEN == 4) {
          const size_t qk_vl = __riscv_vsetvl_e16m1(vIcStep);
          vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
          vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
          vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
          vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

          for (int64_t d = 0; d < pDk; ++d) {
            const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(vKTilePack + d * cIcTileStep, qk_vl);
            qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, vQvBuffer[0 * pDk + d], k_vec, qk_vl);
            qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, vQvBuffer[1 * pDk + d], k_vec, qk_vl);
            qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, vQvBuffer[2 * pDk + d], k_vec, qk_vl);
            qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, vQvBuffer[3 * pDk + d], k_vec, qk_vl);
          }

          qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, pScale, qk_vl);
          qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, pScale, qk_vl);
          qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, pScale, qk_vl);
          qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, pScale, qk_vl);

          __riscv_vse32_v_f32m2(vKqTileBuffer + 0 * cIcTileStep, qk_acc0, qk_vl);
          __riscv_vse32_v_f32m2(vKqTileBuffer + 1 * cIcTileStep, qk_acc1, qk_vl);
          __riscv_vse32_v_f32m2(vKqTileBuffer + 2 * cIcTileStep, qk_acc2, qk_vl);
          __riscv_vse32_v_f32m2(vKqTileBuffer + 3 * cIcTileStep, qk_acc3, qk_vl);
        } else {
          static_assert(QLEN == 2, "unsupported QLEN");

          const size_t cQkVl = __riscv_vsetvl_e16m1(vIcStep);
          vfloat32m2_t vQkAcc0 = __riscv_vfmv_v_f_f32m2(0.0f, cQkVl);
          vfloat32m2_t vQkAcc1 = __riscv_vfmv_v_f_f32m2(0.0f, cQkVl);

          for (int64_t vD = 0; vD < pDk; ++vD) {
            const vfloat16m1_t cKVec = __riscv_vle16_v_f16m1(vKTilePack + vD * cIcTileStep, cQkVl);
            vQkAcc0 = __riscv_vfwmacc_vf_f32m2(vQkAcc0, vQvBuffer[0 * pDk + vD], cKVec, cQkVl);
            vQkAcc1 = __riscv_vfwmacc_vf_f32m2(vQkAcc1, vQvBuffer[1 * pDk + vD], cKVec, cQkVl);
          }

          vQkAcc0 = __riscv_vfmul_vf_f32m2(vQkAcc0, pScale, cQkVl);
          vQkAcc1 = __riscv_vfmul_vf_f32m2(vQkAcc1, pScale, cQkVl);

          __riscv_vse32_v_f32m2(vKqTileBuffer + 0 * cIcTileStep, vQkAcc0, cQkVl);
          __riscv_vse32_v_f32m2(vKqTileBuffer + 1 * cIcTileStep, vQkAcc1, cQkVl);
        }

        for (int vI = 0; vI < QLEN; ++vI) {
          float *vRowPtr = vKqTileBuffer + vI * cIcTileStep;
          const float cTileMax = fRvvSoftcapAddMaxInplaceF32(vRowPtr, vMvTilePack, vIcStep, pLogitSoftcap);

          const float cMold = aM[vI];

          if (cTileMax > cMold) {
            const float cMs = expf(cMold - cTileMax);
            aM[vI] = cTileMax;
            S[vI] *= cMs;

            vfloat16m2_t vVkQ16V = __riscv_vle16_v_f16m2(vKq16Buffer + vI * pDv, pDv);
            vVkQ16V = __riscv_vfmul_vf_f16m2(vVkQ16V, (_Float16)cMs, pDv);
            __riscv_vse16_v_f16m2(vKq16Buffer + vI * pDv, vVkQ16V, pDv);
          }

          S[vI] += fRvvSoftmaxExpInplaceF32(vRowPtr, vIcStep, aM[vI]);
        }

        if constexpr (QLEN == 4) {
          vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(vKq16Buffer + 0 * pDv, pDv);
          vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(vKq16Buffer + 1 * pDv, pDv);
          vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(vKq16Buffer + 2 * pDv, pDv);
          vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(vKq16Buffer + 3 * pDv, pDv);

          for (int64_t tk = 0; tk < vIcStep; ++tk) {
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(vVTilePack + tk * pDv, pDv);
            pv_acc0 = __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16)vKqTileBuffer[0 * cIcTileStep + tk], v16, pDv);
            pv_acc1 = __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16)vKqTileBuffer[1 * cIcTileStep + tk], v16, pDv);
            pv_acc2 = __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16)vKqTileBuffer[2 * cIcTileStep + tk], v16, pDv);
            pv_acc3 = __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16)vKqTileBuffer[3 * cIcTileStep + tk], v16, pDv);
          }

          __riscv_vse16_v_f16m2(vKq16Buffer + 0 * pDv, pv_acc0, pDv);
          __riscv_vse16_v_f16m2(vKq16Buffer + 1 * pDv, pv_acc1, pDv);
          __riscv_vse16_v_f16m2(vKq16Buffer + 2 * pDv, pv_acc2, pDv);
          __riscv_vse16_v_f16m2(vKq16Buffer + 3 * pDv, pv_acc3, pDv);
        } else {
          static_assert(QLEN == 2, "unsupported QLEN");
          vfloat16m2_t vPvAcc0 = __riscv_vle16_v_f16m2(vKq16Buffer + 0 * pDv, pDv);
          vfloat16m2_t vPvAcc1 = __riscv_vle16_v_f16m2(vKq16Buffer + 1 * pDv, pDv);

          for (int64_t vTk = 0; vTk < vIcStep; ++vTk) {
            const vfloat16m2_t cV16 = __riscv_vle16_v_f16m2(vVTilePack + vTk * pDv, pDv);
            vPvAcc0 = __riscv_vfmacc_vf_f16m2(vPvAcc0, (_Float16)vKqTileBuffer[0 * cIcTileStep + vTk], cV16, pDv);
            vPvAcc1 = __riscv_vfmacc_vf_f16m2(vPvAcc1, (_Float16)vKqTileBuffer[1 * cIcTileStep + vTk], cV16, pDv);
          }

          __riscv_vse16_v_f16m2(vKq16Buffer + 0 * pDv, vPvAcc0, pDv);
          __riscv_vse16_v_f16m2(vKq16Buffer + 1 * pDv, vPvAcc1, pDv);
        }

        vIcStep = 0;
      }
    }
  } else {
    for (int64_t vIc = 0; vIc < pNek1; ++vIc) {
      const float cMv = pMp ? pSlope * ((_Float16 *)pMp)[vIc] : 0.0f;

      const char *vKData = pKDataRow + vIc * pNbk1;
      const char *vVData = pVDataRow + vIc * pNbv1;

      vfloat16m2_t vKDataV;
      vfloat16m2_t vVDataV;

      if (cMv != -INFINITY) {
        vKDataV = __riscv_vle16_v_f16m2((_Float16 *)vKData, pDk);
        vVDataV = __riscv_vle16_v_f16m2((_Float16 *)vVData, pDv);
      } else {
        continue;
      }

      for (int vI = 0; vI < QLEN; ++vI) {
        vfloat16m2_t vQQV = __riscv_vle16_v_f16m2(vQvBuffer + vI * pDk, pDk);
        vfloat32m4_t vQkAccV = __riscv_vfwmul_vv_f32m4(vKDataV, vQQV, pDk);
        float s = fReduceSumF32m4Vlen1024(vQkAccV, pDk);
        s = s * pScale;
        if (pLogitSoftcap != 0.0f) {
          s = pLogitSoftcap * tanhf(s);
        }
        s += cMv;

        const float cMold = aM[vI];

        float vMs = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
        float vVs = 1.0f; // post-softmax KQ value, expf(s - M)

        vfloat16m2_t vVkQ16V = __riscv_vle16_v_f16m2(vKq16Buffer + vI * pDv, pDv);
        if (s > aM[vI]) {
          // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
          aM[vI] = s;
          vMs = expf(cMold - aM[vI]);

          // V = V*expf(Mold - M)
          vVkQ16V = __riscv_vfmul_vf_f16m2(vVkQ16V, vMs, pDv);
        } else {
          // no new maximum, ms == 1.0f, vs != 1.0f
          vVs = expf(s - aM[vI]);
        }
        vVkQ16V = __riscv_vfmacc_vf_f16m2(vVkQ16V, vVs, vVDataV, pDv);
        __riscv_vse16_v_f16m2(vKq16Buffer + vI * pDv, vVkQ16V, pDv);
        S[vI] = S[vI] * vMs + vVs; // scale and increment sum with partial sum
      }
    }
  }

  for (int vI = 0; vI < QLEN; ++vI) {
    vfloat16m2_t vVkQ16V = __riscv_vle16_v_f16m2(vKq16Buffer + vI * pDv, pDv);
    vfloat32m4_t vVkQ32V = __riscv_vfwcvt_f_f_v_f32m4(vVkQ16V, pDv);

    // sinks
    if (pSinks[vI]) {
      const float cS = *(pSinks[vI]);

      float vMs = 1.0f;
      float vVs = 1.0f;

      if (cS > aM[vI]) {
        vMs = expf(aM[vI] - cS);
        aM[vI] = cS;
        vVkQ32V = __riscv_vfmul_vf_f32m4(vVkQ32V, vMs, pDv);
      } else {
        vVs = expf(cS - aM[vI]);
      }

      S[vI] = S[vI] * vMs + vVs;
    }

    // V /= S
    const float cSInv = S[vI] == 0.0f ? 0.0f : 1.0f / S[vI];

    vVkQ32V = __riscv_vfmul_vf_f32m4(vVkQ32V, cSInv, pDv);

    __riscv_vse32_v_f32m4(pDst[vI], vVkQ32V, pDv);
  }
}

static void fFlashAttnExtF16OneChunkInnerVlen1024Vf16M1(const float *pPq, const char *pKDataRow, const char *pVDataRow,
                                                        const ggml_fp16_t *pMp, const float *pSinks, float *pDst,
                                                        float pScale, float pLogitSoftcap, float pSlope, int64_t pNek1,
                                                        int64_t pNbk1, int64_t pNbv1, int64_t pDv, int64_t pDk) {
  GGML_ASSERT(fFlashAttnExtSupportedShapeVlen1024Vf16(pDk, pDv));

  float S = 0.0f;       // sum
  float vM = -INFINITY; // maximum KQ value

  vfloat16m2_t vVkQ16V = __riscv_vfmv_v_f_f16m2(0.0f, pDv);

  vfloat16m2_t vQQV = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pPq, pDk), pDk);

  for (int64_t vIc = 0; vIc < pNek1; ++vIc) {
    const float cMv = pMp ? pSlope * ((_Float16 *)pMp)[vIc] : 0.0f;
    if (cMv == -INFINITY) {
      continue;
    }

    const char *vKData = pKDataRow + vIc * pNbk1;

    vfloat16m2_t vKDataV = __riscv_vle16_v_f16m2((_Float16 *)vKData, pDk);

    vfloat32m4_t vQkAccV = __riscv_vfwmul_vv_f32m4(vKDataV, vQQV, pDk);
    float s = fReduceSumF32m4Vlen1024(vQkAccV, pDk);

    s = s * pScale; // scale KQ value

    if (pLogitSoftcap != 0.0f) {
      s = pLogitSoftcap * tanhf(s);
    }

    s += cMv; // apply mask

    const float cMold = vM;

    float vMs = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
    float vVs = 1.0f; // post-softmax KQ value, expf(s - M)

    const char *vVData = pVDataRow + vIc * pNbv1;

    vfloat16m2_t vVDataV = __riscv_vle16_v_f16m2((_Float16 *)vVData, pDv);

    if (s > vM) {
      // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
      vM = s;
      vMs = expf(cMold - vM);

      // V = V*expf(Mold - M)
      vVkQ16V = __riscv_vfmul_vf_f16m2(vVkQ16V, vMs, pDv);
    } else {
      // no new maximum, ms == 1.0f, vs != 1.0f
      vVs = expf(s - vM);
    }

    vVkQ16V = __riscv_vfmacc_vf_f16m2(vVkQ16V, vVs, vVDataV, pDv);

    S = S * vMs + vVs; // scale and increment sum with partial sum
  }

  vfloat32m4_t vVkQ32V = __riscv_vfwcvt_f_f_v_f32m4(vVkQ16V, pDv);

  // sinks
  if (pSinks) {
    const float cS = *pSinks;

    float vMs = 1.0f;
    float vVs = 1.0f;

    if (cS > vM) {
      vMs = expf(vM - cS);
      vM = cS;
      vVkQ32V = __riscv_vfmul_vf_f32m4(vVkQ32V, vMs, pDv);
    } else {
      vVs = expf(cS - vM);
    }

    S = S * vMs + vVs;
  }

  // V /= S
  const float cSInv = S == 0.0f ? 0.0f : 1.0f / S;

  vVkQ32V = __riscv_vfmul_vf_f32m4(vVkQ32V, cSInv, pDv);

  __riscv_vse32_v_f32m4(pDst, vVkQ32V, pDv);
}

} // namespace

void memcpy1d(void *pDst, const void *pSrc, int64_t size) {
  size_t vByteSizeAll = size;
  size_t vVlen = __riscv_vlenb() * 8;
  if (vVlen == 256) {
    // 1024 bytes
    __asm__ volatile(
        //
        "srli           t0, %[size], 10             \n\t"
        "blez           t0, memcpy_tail%=           \n\t"
        "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
        "memcpy_main_loop%=:                        \n\t"
        "addi           t0, t0, -1                  \n\t"
        "vle8.v         v0, (%[s])                  \n\t"
        "addi           %[s], %[s], 256             \n\t"
        "vle8.v         v8, (%[s])                  \n\t"
        "addi           %[s], %[s], 256             \n\t"
        "vle8.v         v16, (%[s])                 \n\t"
        "addi           %[s], %[s], 256             \n\t"
        "vle8.v         v24, (%[s])                 \n\t"
        "addi           %[s], %[s], 256             \n\t"
        //
        "vse8.v         v0, (%[d])                  \n\t"
        "addi           %[d], %[d], 256             \n\t"
        "vse8.v         v8, (%[d])                  \n\t"
        "addi           %[d], %[d], 256             \n\t"
        "vse8.v         v16, (%[d])                 \n\t"
        "addi           %[d], %[d], 256             \n\t"
        "vse8.v         v24, (%[d])                 \n\t"
        "addi           %[d], %[d], 256             \n\t"
        //
        "bnez           t0, memcpy_main_loop%=      \n\t"
        "memcpy_tail%=:                             \n\t"
        "andi           t1, %[size], 1023           \n\t"
        "blez           t1, out%=                   \n\t"
        "memcpy_tail_loop%=:                        \n\t"
        "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
        "sub            t1, t1, t0                  \n\t"
        "vle8.v         v0, (%[s])                  \n\t"
        "add            %[s], %[s], t0              \n\t"
        "vse8.v         v0, (%[d])                  \n\t"
        "add            %[d], %[d], t0              \n\t"
        "bnez           t1, memcpy_tail_loop%=      \n\t"
        "out%=:                                     \n\t"
        : [s] "+r"(pSrc), [d] "+r"(pDst)
        : [size] "r"(vByteSizeAll)
        : "cc", "t0", "t1");
  } else if (vVlen == 1024) {
    // 2048 bytes
    __asm__ volatile(
        //
        "srli           t0, %[size], 11             \n\t"
        "blez           t0, memcpy_tail%=           \n\t"
        "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
        "addi           t2, %[s], 1024              \n\t"
        "addi           t3, %[d], 1024              \n\t"
        "li             t5, 2048                    \n\t"
        "memcpy_main_loop%=:                        \n\t"
        "addi           t0, t0, -1                  \n\t"
        "vle8.v         v0, (%[s])                  \n\t"
        "add            %[s], %[s], t5              \n\t"
        "vle8.v         v8, (t2)                    \n\t"
        "add            t2, t2, t5                  \n\t"
        //
        "vse8.v         v0, (%[d])                  \n\t"
        "add            %[d], %[d], t5              \n\t"
        "vse8.v         v8, (t3)                    \n\t"
        "add            t3, t3, t5                  \n\t"
        //
        "bnez           t0, memcpy_main_loop%=      \n\t"
        "memcpy_tail%=:                             \n\t"
        "andi           t1, %[size], 2047           \n\t"
        "blez           t1, out%=                   \n\t"
        "memcpy_tail_loop%=:                        \n\t"
        "vsetvli        t0, t1, e8, m2, tu, mu      \n\t"
        "sub            t1, t1, t0                  \n\t"
        "vle8.v         v0, (%[s])                  \n\t"
        "add            %[s], %[s], t0              \n\t"
        "vse8.v         v0, (%[d])                  \n\t"
        "add            %[d], %[d], t0              \n\t"
        "bnez           t1, memcpy_tail_loop%=      \n\t"
        "out%=:                                     \n\t"
        : [s] "+r"(pSrc), [d] "+r"(pDst)
        : [size] "r"(vByteSizeAll)
        : "cc", "t0", "t1", "t2", "t3", "t5");
  } else {
    __asm__ volatile(
        //
        "add            t1, %[size], zero           \n\t"
        "memcpy_tail_loop%=:                        \n\t"
        "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
        "sub            t1, t1, t0                  \n\t"
        "vle8.v         v0, (%[s])                  \n\t"
        "add            %[s], %[s], t0              \n\t"
        "vse8.v         v0, (%[d])                  \n\t"
        "add            %[d], %[d], t0              \n\t"
        "bnez           t1, memcpy_tail_loop%=      \n\t"
        : [s] "+r"(pSrc), [d] "+r"(pDst)
        : [size] "r"(vByteSizeAll)
        : "cc", "t0", "t1", "t2", "t4", "t3");
  }
}

void memcpy2d(void *pDst, int64_t pDstStride, const void *pSrc, int64_t pSrcStride, int64_t pTileRows, int64_t pSize) {
  for (int64_t vI = 0; vI < pTileRows; ++vI) {
    memcpy1d((char *)pDst + vI * pDstStride, (const char *)pSrc + vI * pSrcStride, pSize);
  }
}

void forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                        int pIr1, void *pTcmBuffer, size_t pTcmBufferSize) {
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];
  const ggml_tensor *vSinks = pDst->src[4];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int64_t cDk = nek0;
  const int64_t cDv = nev0;
  const int64_t cN = neq1;

  GGML_ASSERT(fFlashAttnExtSupportedShapeVlen1024Vf16(cDk, cDv));

  // broadcast factors
  const int64_t cRk2 = neq2 / nek2;
  const int64_t cRk3 = neq3 / nek3;

  const int64_t cRv2 = neq2 / nev2;
  const int64_t cRv3 = neq3 / nev3;

  // parallelize by q rows using ggml_vec_dot_f32

  float vScale = *((float *)pDst->op_params + 0);
  float vMaxBias = *((float *)pDst->op_params + 1);
  float vLogitSoftcap = *((float *)pDst->op_params + 2);

  if (vLogitSoftcap != 0) {
    vScale /= vLogitSoftcap;
  }

  const uint32_t cNHead = neq2;
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floor(log2(cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  const int cKvRowSize = cDk * sizeof(_Float16) + cDv * sizeof(_Float16);

  int vIth = pParams->ith;
  int vIrStep = 1;
  for (int vIr = pIr0; vIr < pIr1; vIr += vIrStep) {
    // q indices
    const int cIq3 = vIr / (neq2 * neq1);
    const int cIq2 = (vIr - cIq3 * neq2 * neq1) / neq1;
    const int cIq1 = (vIr - cIq3 * neq2 * neq1 - cIq2 * neq1);

    const int cIq31 = (vIr + 1) / (neq2 * neq1);
    const int cIq21 = (vIr + 1 - cIq31 * neq2 * neq1) / neq1;
    const int cIq11 = (vIr + 1 - cIq31 * neq2 * neq1 - cIq21 * neq1);

    const int cIq32 = (vIr + 2) / (neq2 * neq1);
    const int cIq22 = (vIr + 2 - cIq32 * neq2 * neq1) / neq1;
    const int cIq12 = (vIr + 2 - cIq32 * neq2 * neq1 - cIq22 * neq1);

    const int cIq33 = (vIr + 3) / (neq2 * neq1);
    const int cIq23 = (vIr + 3 - cIq33 * neq2 * neq1) / neq1;
    const int cIq13 = (vIr + 3 - cIq33 * neq2 * neq1 - cIq23 * neq1);

    const uint32_t cH = cIq2; // head index
    const float cSlope =
        (vMaxBias > 0.0f) ? cH < cNHeadLog2 ? powf(cM0, cH + 1) : powf(cM1, 2 * (cH - cNHeadLog2) + 1) : 1.0f;

    const ggml_fp16_t *vMp =
        vMask ? (ggml_fp16_t *)((char *)vMask->data + cIq1 * vMask->nb[1] + (cIq2 % vMask->ne[2]) * vMask->nb[2] +
                                (cIq3 % vMask->ne[3]) * vMask->nb[3])
              : NULL;

    const bool cMpEqual2 = cIq11 == cIq1 && (cIq2 % vMask->ne[2]) == (cIq21 % vMask->ne[2]) &&
                           (cIq3 % vMask->ne[3]) == (cIq31 % vMask->ne[3]);

    const bool cMpEqual4 = cMpEqual2 && cIq12 == cIq1 && (cIq2 % vMask->ne[2]) == (cIq22 % vMask->ne[2]) &&
                           (cIq3 % vMask->ne[3]) == (cIq32 % vMask->ne[3]) && cIq13 == cIq1 &&
                           (cIq2 % vMask->ne[2]) == (cIq23 % vMask->ne[2]) &&
                           (cIq3 % vMask->ne[3]) == (cIq33 % vMask->ne[3]);

    // k indices
    const int cIk3 = cIq3 / cRk3;
    const int cIk2 = cIq2 / cRk2;

    const int cIk31 = cIq31 / cRk3;
    const int cIk21 = cIq21 / cRk2;

    const int cIk32 = cIq32 / cRk3;
    const int cIk22 = cIq22 / cRk2;

    const int cIk33 = cIq33 / cRk3;
    const int cIk23 = cIq23 / cRk2;

    // v indices
    const int cIv3 = cIq3 / cRv3;
    const int cIv2 = cIq2 / cRv2;

    const int cIv31 = cIq31 / cRv3;
    const int cIv21 = cIq21 / cRv2;

    const int cIv32 = cIq32 / cRv3;
    const int cIv22 = cIq22 / cRv2;

    const int cIv33 = cIq33 / cRv3;
    const int cIv23 = cIq23 / cRv2;

    const float *vPq = (const float *)((char *)vQ->data + (cIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));

    std::array<float *, 4> aPqBuffer;
    std::array<float *, 4> aSinksBuffer;
    std::array<float *, 4> aDstBuffer;

    if (pTcmBuffer != nullptr && 4 * cKvRowSize < pTcmBufferSize && vIr < (pIr1 - 3) && cMpEqual4 && cIk33 == cIk3 &&
        cIk23 == cIk2 && cIv33 == cIv3 && cIv23 == cIv2 && cIk32 == cIk3 && cIk22 == cIk2 && cIv32 == cIv3 &&
        cIv22 == cIv2 && cIk31 == cIk3 && cIk21 == cIk2 && cIv31 == cIv3 && cIv21 == cIv2) {
      vIrStep = 4;

      aPqBuffer[0] = (float *)((char *)vQ->data + (cIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));
      aPqBuffer[1] = (float *)((char *)vQ->data + (cIq11 * nbq1 + cIq21 * nbq2 + cIq31 * nbq3));
      aPqBuffer[2] = (float *)((char *)vQ->data + (cIq12 * nbq1 + cIq22 * nbq2 + cIq32 * nbq3));
      aPqBuffer[3] = (float *)((char *)vQ->data + (cIq13 * nbq1 + cIq23 * nbq2 + cIq33 * nbq3));

      aSinksBuffer[0] = vSinks ? ((float *)((char *)vSinks->data)) + cIq2 : nullptr;
      aSinksBuffer[1] = vSinks ? ((float *)((char *)vSinks->data)) + cIq21 : nullptr;
      aSinksBuffer[2] = vSinks ? ((float *)((char *)vSinks->data)) + cIq22 : nullptr;
      aSinksBuffer[3] = vSinks ? ((float *)((char *)vSinks->data)) + cIq23 : nullptr;

      aDstBuffer[0] = (float *)((char *)pDst->data + (cIq3 * ne2 * ne1 + cIq2 + cIq1 * ne1) * nb1);
      aDstBuffer[1] = (float *)((char *)pDst->data + (cIq31 * ne2 * ne1 + cIq21 + cIq11 * ne1) * nb1);
      aDstBuffer[2] = (float *)((char *)pDst->data + (cIq32 * ne2 * ne1 + cIq22 + cIq12 * ne1) * nb1);
      aDstBuffer[3] = (float *)((char *)pDst->data + (cIq33 * ne2 * ne1 + cIq23 + cIq13 * ne1) * nb1);

      fFlashAttnExtF16OneChunkInnerVlen1024Vf16Mrow<4>(         //
          aPqBuffer.data(),                                     //
          (const char *)vK->data + (cIk2 * nbk2 + cIk3 * nbk3), //
          (const char *)v_->data + (cIv2 * nbv2 + cIv3 * nbv3), //
          vMp,                                                  //
          aSinksBuffer.data(),                                  //
          aDstBuffer.data(),                                    //
          vScale, vLogitSoftcap, cSlope, nek1, nbk1, nbv1, cDv, cDk, pTcmBuffer, pTcmBufferSize);
    } else if (pTcmBuffer != nullptr && 2 * cKvRowSize < pTcmBufferSize && vIr < (pIr1 - 1) && cMpEqual2 &&
               cIk31 == cIk3 && cIk21 == cIk2 && cIv31 == cIv3 && cIv21 == cIv2) {
      vIrStep = 2;

      aPqBuffer[0] = (float *)((char *)vQ->data + (cIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));
      aPqBuffer[1] = (float *)((char *)vQ->data + (cIq11 * nbq1 + cIq21 * nbq2 + cIq31 * nbq3));

      aSinksBuffer[0] = vSinks ? ((float *)((char *)vSinks->data)) + cIq2 : nullptr;
      aSinksBuffer[1] = vSinks ? ((float *)((char *)vSinks->data)) + cIq21 : nullptr;

      aDstBuffer[0] = (float *)((char *)pDst->data + (cIq3 * ne2 * ne1 + cIq2 + cIq1 * ne1) * nb1);
      aDstBuffer[1] = (float *)((char *)pDst->data + (cIq31 * ne2 * ne1 + cIq21 + cIq11 * ne1) * nb1);

      fFlashAttnExtF16OneChunkInnerVlen1024Vf16Mrow<2>(         //
          aPqBuffer.data(),                                     //
          (const char *)vK->data + (cIk2 * nbk2 + cIk3 * nbk3), //
          (const char *)v_->data + (cIv2 * nbv2 + cIv3 * nbv3), //
          vMp,                                                  //
          aSinksBuffer.data(),                                  //
          aDstBuffer.data(),                                    //
          vScale, vLogitSoftcap, cSlope, nek1, nbk1, nbv1, cDv, cDk, pTcmBuffer, pTcmBufferSize);
    } else {
      vIrStep = 1;
      fFlashAttnExtF16OneChunkInnerVlen1024Vf16M1(                                      //
          vPq,                                                                          //
          (const char *)vK->data + (cIk2 * nbk2 + cIk3 * nbk3),                         //
          (const char *)v_->data + (cIv2 * nbv2 + cIv3 * nbv3),                         //
          vMp,                                                                          //
          vSinks ? ((float *)((char *)vSinks->data)) + cH : nullptr,                    //
          (float *)((char *)pDst->data + (cIq3 * ne2 * ne1 + cIq2 + cIq1 * ne1) * nb1), //
          vScale, vLogitSoftcap, cSlope, nek1, nbk1, nbv1, cDv, cDk);
    }
  }
}

void forward_flash_attn_ext_f16_tiled_vlen1024_vf16(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                    int pIr1, void *pTcmBuffer, size_t pTcmBufferSize) {
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];
  const ggml_tensor *vSinks = pDst->src[4];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int64_t cDk = nek0;
  const int64_t cDv = nev0;
  const int64_t cN = neq1;

  GGML_ASSERT(fFlashAttnExtSupportedShapeVlen1024Vf16(cDk, cDv));

  GGML_ASSERT(ne0 == cDv);
  GGML_ASSERT(ne2 == cN);

  // input tensor rows must be contiguous
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbv0 == ggml_type_size(v_->type));

  GGML_ASSERT(neq0 == cDk);
  GGML_ASSERT(nek0 == cDk);
  GGML_ASSERT(nev0 == cDv);

  GGML_ASSERT(neq1 == cN);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(vK->type == v_->type);
  const ggml_type cKvType = vK->type;

  // broadcast factors
  const int64_t cRk2 = neq2 / nek2;
  const int64_t cRk3 = neq3 / nek3;

  const int64_t cRv2 = neq2 / nev2;
  const int64_t cRv3 = neq3 / nev3;

  float *vParamList = (float *)pDst->op_params;
  float vScale = vParamList[0];
  float vMaxBias = vParamList[1];
  float vLogitSoftcap = vParamList[2];

  if (vLogitSoftcap != 0) {
    vScale /= vLogitSoftcap;
  }

  const uint32_t cNHead = neq2;
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floor(log2(cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  int vIth = pParams->ith;

  static constexpr int cQTileSz = ggml_fa_tile_config::Q;
  static constexpr int cKvTileSz = ggml_fa_tile_config::KV;

  // Per-thread scratch layout:
  // Q_f32:   Q_TILE_SZ * DK
  // KQ:      Q_TILE_SZ * KV_TILE_SZ
  // mask32:  Q_TILE_SZ * KV_TILE_SZ
  // VKQ32:   Q_TILE_SZ * DV
  // V32:     KV_TILE_SZ * DV
  // K_f32:   DK * KV_TILE_SZ (transposed K tile)
  float *vBase = (float *)pParams->wdata + vIth * (cQTileSz * cDk + 2 * cQTileSz * cKvTileSz + cQTileSz * cDv +
                                                   cKvTileSz * cDv + cKvTileSz * cDk + CACHE_LINE_SIZE_F32);
  const size_t cBaseSize =
      (cQTileSz * cDk + 2 * cQTileSz * cKvTileSz + cQTileSz * cDv + cKvTileSz * cDv + cKvTileSz * cDk) * sizeof(float) +
      CACHE_LINE_SIZE_F32;

  if (cBaseSize <= pTcmBufferSize && pTcmBuffer != nullptr) {
    vBase = (float *)pTcmBuffer;
  }

  float aSMBuf[cQTileSz * 2]; // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
  float *vS = aSMBuf;
  float *vM = aSMBuf + cQTileSz;

  int vIr = pIr0;
  while (vIr < pIr1) {
    // q indices for the start of this tile
    const int cIq3 = vIr / (neq2 * neq1);
    const int cIq2 = (vIr - cIq3 * neq2 * neq1) / neq1;
    const int cIq1 = (vIr - cIq3 * neq2 * neq1 - cIq2 * neq1);

    // Number of valid rows in this tile:
    // - limited by tile size (Q_TILE_SZ)
    // - limited by chunk boundary (ir1 - ir)
    // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
    const int cTileRows = MIN(cQTileSz, MIN((int)(pIr1 - vIr), (int)(neq1 - cIq1)));
    GGML_ASSERT(cTileRows > 0);

    const uint32_t cH = cIq2; // head index
    const float cSlope =
        (vMaxBias > 0.0f) ? cH < cNHeadLog2 ? powf(cM0, cH + 1) : powf(cM1, 2 * (cH - cNHeadLog2) + 1) : 1.0f;

    for (int vI = 0; vI < cQTileSz; ++vI) {
      vS[vI] = 0.;
      vM[vI] = -INFINITY;
    }

    float *vQF32 = vBase;
    float *vKq = (float *)((char *)vBase + cQTileSz * cDk * sizeof(float));
    float *vMask32 = vKq + cQTileSz * cKvTileSz;
    float *vVkQ32 = vMask32 + cQTileSz * cKvTileSz;
    float *vV32 = vVkQ32 + cQTileSz * cDv;
    float *vKF32 = vV32 + cKvTileSz * cDv;
    _Float16 *vQF16 = (_Float16 *)vQF32;
    _Float16 *vVF16 = (_Float16 *)vV32;
    _Float16 *vKF16 = (_Float16 *)vKF32;

    fRvvZeroF32(vVkQ32, cQTileSz * cDv);

    // k indices
    const int cIk3 = cIq3 / cRk3;
    const int cIk2 = cIq2 / cRk2;

    // v indices
    const int cIv3 = cIq3 / cRv3;
    const int cIv2 = cIq2 / cRv2;

    const float *vPq = (const float *)((char *)vQ->data + (cIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));
    if (cKvType == GGML_TYPE_F16) {
      fRvvPackF32AsScaledF16((uint8_t *)vQF16, cDk * sizeof(_Float16), (uint8_t *)vPq, nbq1, cTileRows, cDk, vScale);
    } else {
      memcpy2d(vQF32, cDk * sizeof(float), vPq, nbq1, cTileRows, cDk * sizeof(float));
    }

    for (int64_t vIc = 0; vIc < nek1; vIc += cKvTileSz) {
      const int cKvTile = (int)std::min((int64_t)cKvTileSz, nek1 - vIc);

      fRvvZeroF32(vKF32, cDk * cKvTileSz);
      fRvvZeroF32(vV32, cKvTileSz * cDv);

      // skip the tile entirely if all the masks are -inf
      if (vMask) {
        bool vCanSkip = true;
        const ggml_fp16_t *vMpRow =
            (const ggml_fp16_t *)((const char *)vMask->data + cIq1 * vMask->nb[1] +
                                  (cIq2 % vMask->ne[2]) * vMask->nb[2] + (cIq3 % vMask->ne[3]) * vMask->nb[3]);
        fRvvPackScaledF16AsF32(vMask32, cKvTileSz * sizeof(float), vMpRow + vIc, vMask->nb[1], cTileRows, cKvTile,
                               cSlope);

        for (int vTq = 0; vTq < cTileRows; vTq++) {
          for (int vTk = 0; vTk < cKvTile; vTk++) {
            if (vMask32[vTq * cKvTileSz + vTk] != -INFINITY) {
              vCanSkip = false;
            }
          }
          // Pad remaining mask entries with -inf
          for (int vTk = cKvTile; vTk < cKvTileSz; vTk++) {
            vMask32[vTq * cKvTileSz + vTk] = -INFINITY;
          }
        }

        if (vCanSkip) {
          continue;
        }
      }

      if (cKvType == GGML_TYPE_F16) {
        fRvvTransposedS16MnToNm((int8_t *)vKF16, cKvTileSz * sizeof(_Float16),
                                (int8_t *)vK->data + vIc * nbk1 + cIk2 * nbk2 + cIk3 * nbk3, nbk1, cKvTile, cDk);

        int vTq = 0;
        for (; vTq + 3 < cTileRows; vTq += 4) {
          fRvvQkDotTileF16X4(vKq + (vTq + 0) * cKvTileSz, vKq + (vTq + 1) * cKvTileSz, vKq + (vTq + 2) * cKvTileSz,
                             vKq + (vTq + 3) * cKvTileSz, vQF16 + (vTq + 0) * cDk, vQF16 + (vTq + 1) * cDk,
                             vQF16 + (vTq + 2) * cDk, vQF16 + (vTq + 3) * cDk, vKF16, cDk, cKvTile);
        }
        for (; vTq < cTileRows; ++vTq) {
          fRvvQkDotTileF16X1(vKq + vTq * cKvTileSz, vQF16 + vTq * cDk, vKF16, cDk, cKvTile);
        }
      } else {
        for (int vTk = 0; vTk < cKvTile; vTk++) {
          const char *vKData = (const char *)vK->data + (vIc + vTk) * nbk1 + cIk2 * nbk2 + cIk3 * nbk3;
          float *vKCol = vKF32 + vTk;
          const float *vKSrc = (const float *)vKData;
          for (int64_t vDk = 0; vDk < cDk; ++vDk) {
            vKCol[vDk * cKvTileSz] = vKSrc[vDk];
          }
        }

        for (int vTq = 0; vTq < cTileRows; ++vTq) {
          fRvvQkDotTile(vKq + vTq * cKvTileSz, vQF32 + vTq * cDk, vKF32, cDk, cKvTileSz, vScale);
        }
      }

      // Set padded KQ entries to -inf so softmax gives them zero weight
      if (cKvTile < cKvTileSz) {
        for (int vTq = 0; vTq < cTileRows; vTq++) {
          for (int vTk = cKvTile; vTk < cKvTileSz; vTk++) {
            vKq[vTq * cKvTileSz + vTk] = -INFINITY;
          }
        }
      }

      if (vLogitSoftcap != 0.0f) {
        fRvvSoftcapTanhInplaceF32(vKq, cKvTileSz, cTileRows, cKvTileSz, vLogitSoftcap);
      }

      if (vMask) {
        fRvvAddInplaceF32(vKq, cKvTileSz, vMask32, cKvTileSz, cTileRows, cKvTileSz);
      }

      bool aSkip[cQTileSz] = {};

      for (int vTq = 0; vTq < cTileRows; vTq++) {
        float *vKqRow = vKq + vTq * cKvTileSz;

        const float cTileMax = fRvvMaxF32(vKqRow, cKvTileSz);

        if (cTileMax == -INFINITY) {
          aSkip[vTq] = true;
          continue;
        }

        const float cMold = vM[vTq];
        const float cMnew = fmaxf(cMold, cTileMax);

        if (cMnew > cMold) {
          const float cMs = expf(cMold - cMnew);
          fRvvScaleF32(vVkQ32 + vTq * cDv, cMs, cDv);
          vS[vTq] *= cMs;
        }
        vM[vTq] = cMnew;

        vS[vTq] += fRvvSoftmaxExpInplaceF32(vKqRow, cKvTileSz, cMnew);
      }

      // Pack V as contiguous [KV_TILE_SZ][DV].
      if (cKvType == GGML_TYPE_F16) {
        const char *vVData = (const char *)v_->data + vIc * nbv1 + cIv2 * nbv2 + cIv3 * nbv3;
        memcpy2d(vVF16, cDv * sizeof(_Float16), vVData, nbv1, cKvTile, cDv * sizeof(_Float16));

        int vTq = 0;
        for (; vTq + 3 < cTileRows; vTq += 4) {
          if (aSkip[vTq + 0] || aSkip[vTq + 1] || aSkip[vTq + 2] || aSkip[vTq + 3]) {
            for (int vI = 0; vI < 4; ++vI) {
              if (!aSkip[vTq + vI]) {
                fRvvPvAccumulateF16X1(vVkQ32 + (vTq + vI) * cDv, vKq + (vTq + vI) * cKvTileSz, vVF16, cKvTileSz, cDv);
              }
            }
            continue;
          }

          fRvvPvAccumulateF16X4(vVkQ32 + (vTq + 0) * cDv, vVkQ32 + (vTq + 1) * cDv, vVkQ32 + (vTq + 2) * cDv,
                                vVkQ32 + (vTq + 3) * cDv, vKq + (vTq + 0) * cKvTileSz, vKq + (vTq + 1) * cKvTileSz,
                                vKq + (vTq + 2) * cKvTileSz, vKq + (vTq + 3) * cKvTileSz, vVF16, cKvTileSz, cDv);
        }
        for (; vTq < cTileRows; ++vTq) {
          if (!aSkip[vTq]) {
            fRvvPvAccumulateF16X1(vVkQ32 + vTq * cDv, vKq + vTq * cKvTileSz, vVF16, cKvTileSz, cDv);
          }
        }
      } else {
        const char *vVData = (const char *)v_->data + vIc * nbv1 + cIv2 * nbv2 + cIv3 * nbv3;
        memcpy2d(vV32, cDv * sizeof(float), vVData, nbv1, cKvTile, cDv * sizeof(float));

        for (int vTq = 0; vTq < cTileRows; ++vTq) {
          if (!aSkip[vTq]) {
            fRvvPvAccumulate(vVkQ32 + vTq * cDv, vKq + vTq * cKvTileSz, vV32, cKvTileSz, cDv);
          }
        }
      }
    }

    // sinks (apply only to valid rows in the tile)
    if (vSinks) {
      const float cS = ((float *)((char *)vSinks->data))[cH];

      for (int vTq = 0; vTq < cTileRows; vTq++) {
        float vMs = 1.0f;
        float vVs = 1.0f;

        if (cS > vM[vTq]) {
          vMs = expf(vM[vTq] - cS);
          fRvvScaleF32(vVkQ32 + vTq * cDv, vMs, cDv);
        } else {
          vVs = expf(cS - vM[vTq]);
        }

        float vSTemp = vS[vTq] * vMs + vVs;
        vS[vTq] = vSTemp == 0.0f ? 0.0f : 1.0f / vSTemp;
      }
    } else {
      for (int vTq = 0; vTq < cTileRows; vTq++) {
        const float cSInv = vS[vTq] == 0.0f ? 0.0f : 1.0f / vS[vTq];
        vS[vTq] = cSInv;
      }
    }

    float *vDstPtr = (float *)((char *)pDst->data + (cIq3 * ne2 * ne1 + cIq2 + (cIq1)*ne1) * nb1);
    fRvvPackScaledF32AsF32(vDstPtr, nb1 * ne1, vVkQ32, cDv * sizeof(float), cTileRows, cDv, vS);

    vIr += cTileRows;
  }
}

void forward_rms_norm_f32(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  ggml_tensor *dst = pOp;
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(src0->nb[0] == sizeof(float));

  int vIth = pParams->ith;
  int vNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float vEpsilon = *((float *)dst->op_params);

  GGML_ASSERT(vEpsilon > 0.0f);

  auto *vInput = (char *)src0->data;
  auto *vOutput = (char *)dst->data;

  const auto cHiddenSize = ne00;
  const auto cTaskCount = ne01 * ne02 * ne03;
  const auto cTaskPerThread = (cTaskCount + vNth - 1) / vNth;

  const auto cTaskBegin = vIth * cTaskPerThread;
  const auto cTaskEnd = std::min((vIth + 1) * cTaskPerThread, cTaskCount);

  for (auto vTaskIdx = cTaskBegin; vTaskIdx < cTaskEnd; vTaskIdx++) {
    int64_t vI03 = vTaskIdx / (ne02 * ne01);
    int64_t vI02 = (vTaskIdx - vI03 * ne02 * ne01) / ne01;
    int64_t vI01 = (vTaskIdx - vI03 * ne02 * ne01 - vI02 * ne01);

    auto *vPInput = (float *)(vInput + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
    auto *vPOutput = (float *)(vOutput + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);
    auto *vPTempOutput = vPOutput;

    size_t vGvl = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vSumSq = __riscv_vfmv_v_f_f32m4(0.f, vGvl);
    int64_t vLength = cHiddenSize;
    while (vLength > 0) {
      vGvl = __riscv_vsetvl_e32m4(vLength);
      vfloat32m4_t vSrcData = __riscv_vle32_v_f32m4(vPInput, vGvl);
      vSumSq = __riscv_vfmacc_vv_f32m4(vSumSq, vSrcData, vSrcData, vGvl);
      __riscv_vse32_v_f32m4(vPTempOutput, vSrcData, vGvl);

      vPInput += vGvl;
      vPTempOutput += vGvl;
      vLength -= vGvl;
    }

    vGvl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vZeroV = __riscv_vfmv_v_f_f32m1(0.f, vGvl);
    vfloat32m1_t vMeanSquareV =
        __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vSumSq, 0), __riscv_vget_v_f32m4_f32m1(vSumSq, 1), vGvl);

    vMeanSquareV = __riscv_vfadd_vv_f32m1(vMeanSquareV, __riscv_vget_v_f32m4_f32m1(vSumSq, 2), vGvl);
    vMeanSquareV = __riscv_vfadd_vv_f32m1(vMeanSquareV, __riscv_vget_v_f32m4_f32m1(vSumSq, 3), vGvl);
    vMeanSquareV = __riscv_vfredusum_vs_f32m1_f32m1(vMeanSquareV, vZeroV, vGvl);

    float vMeanSquare = __riscv_vfmv_f_s_f32m1_f32(vMeanSquareV);
    vMeanSquare /= cHiddenSize;

    vMeanSquare = sqrt(vMeanSquare + vEpsilon);

    vMeanSquare = 1.0f / vMeanSquare;
    vLength = cHiddenSize;
    vPTempOutput = vPOutput;

    while (vLength > 0) {
      vGvl = __riscv_vsetvl_e32m4(vLength);
      vfloat32m4_t vSrcData = __riscv_vle32_v_f32m4(vPTempOutput, vGvl);
      vSrcData = __riscv_vfmul_vf_f32m4(vSrcData, vMeanSquare, vGvl);
      __riscv_vse32_v_f32m4(vPOutput, vSrcData, vGvl);
      vPTempOutput += vGvl;
      vPOutput += vGvl;
      vLength -= vGvl;
    }
  }
}

template <size_t MB_ROWS>
void fQuantizeANrowI8Ref(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  int64_t vABlkStride = q8_blk_size(pBlkLen, true);
  int64_t vANrowBlockStride = vABlkStride * MB_ROWS;
  for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
    float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
    int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * MB_ROWS);
    int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * MB_ROWS + sizeof(int16_t) * MB_ROWS);

    for (size_t vRow = 0; vRow < MB_ROWS; vRow++) {
      float vMaxAbsA = 0.0f;
      for (size_t vBk = 0; vBk < pBlkLen; vBk++) {
        vMaxAbsA = std::max(vMaxAbsA, std::abs(pAPtr[vRow * pCountK + vK + vBk]));
      }

      float vRepScaleA = ((1 << 7) - 1) / vMaxAbsA;
      vScaleAPtr[vRow] = 1 / vRepScaleA;

      int16_t vASum = 0;
      for (size_t vBk = 0; vBk < pBlkLen; vBk++) {
        const int8_t cQuantized = static_cast<int8_t>(
            std::clamp(std::nearbyintf(pAPtr[vRow * pCountK + vK + vBk] * vRepScaleA), -128.0f, 127.0f));
        vQuantABlk[vRow * pBlkLen + vBk] = cQuantized;
        vASum += cQuantized;
      }
      vASumPtr[vRow] = -vASum;
    }
  }
}

template <size_t MB_ROWS>
void fQuantizeANrowI8HpRef(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  constexpr size_t cKSubblkLen = 32;
  const size_t cSubblkCount = pBlkLen / cKSubblkLen;

  GGML_ASSERT(pBlkLen == 256);

  float aScaleTemp[8] = {0.0f};
  int64_t vABlkStride = q8_hp_blk_size(pBlkLen, true, true);
  int64_t vANrowBlockStride = vABlkStride * MB_ROWS;
  int64_t vASubblkStride = q8_hp_blk_size(cKSubblkLen, false, false) * MB_ROWS;

  for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
    _Float16 *vASumPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vASubblkStride * cSubblkCount);

    float vScaleAvg = 0.0f;
    for (size_t vKk = 0; vKk < cSubblkCount; vKk++) {
      float vMaxAbsA = 0.0f;
      for (size_t vRow = 0; vRow < MB_ROWS; vRow++) {
        for (size_t vBk = 0; vBk < cKSubblkLen; vBk++) {
          vMaxAbsA = std::max(vMaxAbsA, std::abs(pAPtr[vRow * pCountK + vK + vBk + vKk * cKSubblkLen]));
        }
      }
      aScaleTemp[vKk] = vMaxAbsA / ((1 << 7) - 1);
      vScaleAvg += aScaleTemp[vKk];
    }

    vScaleAvg /= cSubblkCount;
    float vScaleFactor = 1.0f / vScaleAvg;

    _Float16 *vScaleAvgPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vANrowBlockStride - sizeof(_Float16) * MB_ROWS);
    vScaleAvgPtr[0] = vScaleAvg;

    for (size_t vKk = 0; vKk < cSubblkCount; vKk++) {
      uint8_t *vASubblkBase = pQuantAPtr + vKk * vASubblkStride;
      _Float16 *vScaleAPtr = reinterpret_cast<_Float16 *>(vASubblkBase);
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(vASubblkBase + sizeof(_Float16) * MB_ROWS);

      vScaleAPtr[0] = static_cast<_Float16>(aScaleTemp[vKk] * vScaleFactor);

      const float cRepScaleA = 1.0f / aScaleTemp[vKk];

      for (size_t vRow = 0; vRow < MB_ROWS; vRow++) {
        int16_t vASum = 0;
        for (size_t vBk = 0; vBk < cKSubblkLen; vBk++) {
          const int8_t cQuantized = static_cast<int8_t>(std::clamp(
              std::nearbyintf(pAPtr[vRow * pCountK + vK + vBk + vKk * cKSubblkLen] * cRepScaleA), -128.0f, 127.0f));
          vQuantABlk[vRow * cKSubblkLen + vBk] = cQuantized;
          vASum += cQuantized;
        }
        vASumPtr[vRow * cSubblkCount + vKk] = static_cast<_Float16>(-vASum) * static_cast<_Float16>(8.0f);
      }
    }
  }
}

template <size_t MB_ROWS>
void fQuantizeANrowI8kRef(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  int64_t vABlkStride = q8k_blk_size(256);
  int64_t vANrowBlockStride = vABlkStride * MB_ROWS;
  int64_t vASumSize = 256 / 16;

  for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
    float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
    int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * MB_ROWS);
    int8_t *vQuantABlk =
        reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * MB_ROWS + sizeof(int16_t) * vASumSize * MB_ROWS);

    for (size_t vRow = 0; vRow < MB_ROWS; vRow++) {
      float vMaxA = 0.0f;
      float vMaxAbsA = 0.0f;
      for (size_t vBk = 0; vBk < pBlkLen; vBk++) {
        float vAx = std::abs(pAPtr[vRow * pCountK + vK + vBk]);
        if (vAx > vMaxAbsA) {
          vMaxAbsA = vAx;
          vMaxA = pAPtr[vRow * pCountK + vK + vBk];
        }
      }

      if (!vMaxAbsA) {
        vScaleAPtr[vRow] = 0;
        for (size_t vBki = 0; vBki < vASumSize; vBki++) {
          for (size_t vBk = vBki * 16; vBk < (vBki + 1) * 16; vBk++) {
            vQuantABlk[vRow * pBlkLen + vBk] = 0;
          }
          vASumPtr[vRow * vASumSize + vBki] = 0;
        }
        continue;
      }

      float vRepScaleA = ((1 << 7) - 1) / vMaxAbsA;
      vScaleAPtr[vRow] = 1 / vRepScaleA;

      for (size_t vBki = 0; vBki < vASumSize; vBki++) {
        int16_t vASum = 0;
        for (size_t vBk = vBki * 16; vBk < (vBki + 1) * 16; vBk++) {
          const int8_t cQuantized = static_cast<int8_t>(
              std::clamp(std::nearbyintf(pAPtr[vRow * pCountK + vK + vBk] * vRepScaleA), -128.0f, 127.0f));
          vQuantABlk[vRow * pBlkLen + vBk] = cQuantized;
          vASum += cQuantized;
        }
        vASumPtr[vRow * vASumSize + vBki] = -vASum;
      }
    }
  }
}

void quantize_a_row_i8(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  GGML_ASSERT(pBlkLen == 32);
  int64_t vABlkStride = q8_blk_size(pBlkLen, true);
  size_t vVlenb = __riscv_vlenb();

  if (vVlenb == 128) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vABlkStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float));
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) + sizeof(int16_t));

      size_t vVl = __riscv_vsetvl_e32m1(pBlkLen);
      vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vK, vVl);
      vfloat32m1_t vVAAbs = __riscv_vfabs_v_f32m1(vVA, vVl);

      vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
      vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m1_f32m1(vVAAbs, vTmp, vVl);
      float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

      float vScaleA = vMaxAbsA / ((1 << 7) - 1);
      float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
      vScaleAPtr[0] = vScaleA;

      vfloat32m1_t vVAScale = __riscv_vfmul_vf_f32m1(vVA, vRepScaleA, vVl);
      vint16mf2_t vVAQuant = __riscv_vfncvt_x_f_w_i16mf2(vVAScale, vVl);
      vint8mf4_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVAQuant, vVl);

      vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
      vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf4_i16m1(vVAQuantI8, vTmpSum, vVl);
      int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
      vASumPtr[0] = -vASum;

      __riscv_vse8_v_i8mf4(vQuantABlk, vVAQuantI8, vVl);
    }
  } else if (vVlenb == 32) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vABlkStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float));
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) + sizeof(int16_t));

      size_t vVl = __riscv_vsetvl_e32m4(pBlkLen);
      vfloat32m4_t vVA = __riscv_vle32_v_f32m4(pAPtr + vK, vVl);
      vfloat32m4_t vVAAbs = __riscv_vfabs_v_f32m4(vVA, vVl);

      vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
      vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m4_f32m1(vVAAbs, vTmp, vVl);
      float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

      float vScaleA = vMaxAbsA / ((1 << 7) - 1);
      float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
      vScaleAPtr[0] = vScaleA;

      vfloat32m4_t vVAScale = __riscv_vfmul_vf_f32m4(vVA, vRepScaleA, vVl);
      vint16m2_t vVAQuant = __riscv_vfncvt_x_f_w_i16m2(vVAScale, vVl);
      vint8m1_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8m1(vVAQuant, vVl);

      vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
      vint16m1_t vVASum = __riscv_vwredsum_vs_i8m1_i16m1(vVAQuantI8, vTmpSum, vVl);
      int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
      vASumPtr[0] = -vASum;

      __riscv_vse8_v_i8m1(vQuantABlk, vVAQuantI8, vVl);
    }
  } else {
    fQuantizeANrowI8Ref<1>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void quantize_a_4row_i8(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  GGML_ASSERT(pBlkLen == 32);
  int64_t vABlkStride = q8_blk_size(pBlkLen, true);
  int64_t vANrowBlockStride = vABlkStride * 4;
  size_t vVlenb = __riscv_vlenb();

  if (vVlenb == 128) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * 4);
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * 4 + sizeof(int16_t) * 4);

      for (size_t vMi = 0; vMi < 4; vMi++) {
        size_t vVl = __riscv_vsetvl_e32m1(pBlkLen);
        vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vMi * pCountK + vK, vVl);
        vfloat32m1_t vVAAbs = __riscv_vfabs_v_f32m1(vVA, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m1_f32m1(vVAAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        float vScaleA = vMaxAbsA / ((1 << 7) - 1);
        float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
        vScaleAPtr[vMi] = vScaleA;

        vfloat32m1_t vVAScale = __riscv_vfmul_vf_f32m1(vVA, vRepScaleA, vVl);
        vint16mf2_t vVAQuant = __riscv_vfncvt_x_f_w_i16mf2(vVAScale, vVl);
        vint8mf4_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf4_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vMi] = -vASum;

        __riscv_vse8_v_i8mf4(vQuantABlk + vMi * pBlkLen, vVAQuantI8, vVl);
      }
    }
  } else if (vVlenb == 32) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * 4);
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * 4 + sizeof(int16_t) * 4);

      for (size_t vMi = 0; vMi < 4; vMi++) {
        size_t vVl = __riscv_vsetvl_e32m4(pBlkLen);
        vfloat32m4_t vVA = __riscv_vle32_v_f32m4(pAPtr + vMi * pCountK + vK, vVl);
        vfloat32m4_t vVAAbs = __riscv_vfabs_v_f32m4(vVA, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m4_f32m1(vVAAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        float vScaleA = vMaxAbsA / ((1 << 7) - 1);
        float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
        vScaleAPtr[vMi] = vScaleA;

        vfloat32m4_t vVAScale = __riscv_vfmul_vf_f32m4(vVA, vRepScaleA, vVl);
        vint16m2_t vVAQuant = __riscv_vfncvt_x_f_w_i16m2(vVAScale, vVl);
        vint8m1_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8m1(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8m1_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vMi] = -vASum;

        __riscv_vse8_v_i8m1(vQuantABlk + vMi * pBlkLen, vVAQuantI8, vVl);
      }
    }
  } else {
    fQuantizeANrowI8Ref<4>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void quantize_a_row_i8_hp(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  constexpr size_t cKSubblkLen = 32;
  GGML_ASSERT(pBlkLen == 256);

  constexpr size_t cSubblkCount = 256 / cKSubblkLen;
  int64_t vABlkStride = q8_hp_blk_size(pBlkLen, true, true);
  int64_t vASubblkStride = q8_hp_blk_size(cKSubblkLen, false, false);
  size_t vVlenb = __riscv_vlenb();
  float aScaleTemp[cSubblkCount] = {0.0f};

  if (vVlenb == 128) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vABlkStride) {
      _Float16 *vASumPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vASubblkStride * cSubblkCount);
      _Float16 *vScaleAvgPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vABlkStride - sizeof(_Float16));
      float vScaleAvg = 0.0f;

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        const float *vASrcPtr = pAPtr + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m1(cKSubblkLen);
        vfloat32m1_t vVA = __riscv_vle32_v_f32m1(vASrcPtr, vVl);
        vfloat32m1_t vVAAbs = __riscv_vfabs_v_f32m1(vVA, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m1_f32m1(vVAAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        aScaleTemp[vKk] = vMaxAbsA / ((1 << 7) - 1);
        vScaleAvg += aScaleTemp[vKk];
      }

      vScaleAvg /= cSubblkCount;
      const float cScaleFactor = vScaleAvg ? 1.0f / vScaleAvg : 0.0f;
      vScaleAvgPtr[0] = static_cast<_Float16>(vScaleAvg);

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        uint8_t *vASubblkBase = pQuantAPtr + vKk * vASubblkStride;
        _Float16 *vScaleAPtr = reinterpret_cast<_Float16 *>(vASubblkBase);
        int8_t *vQuantABlk = reinterpret_cast<int8_t *>(vASubblkBase + sizeof(_Float16));
        const float *vASrcPtr = pAPtr + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m1(cKSubblkLen);
        vfloat32m1_t vVA = __riscv_vle32_v_f32m1(vASrcPtr, vVl);
        float vRepScaleA = aScaleTemp[vKk] ? 1.0f / aScaleTemp[vKk] : 0.0f;
        vScaleAPtr[0] = static_cast<_Float16>(aScaleTemp[vKk] * cScaleFactor);

        vfloat32m1_t vVAScale = __riscv_vfmul_vf_f32m1(vVA, vRepScaleA, vVl);
        vint16mf2_t vVAQuant = __riscv_vfncvt_x_f_w_i16mf2(vVAScale, vVl);
        vint8mf4_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf4_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vKk] = static_cast<_Float16>(-vASum) * static_cast<_Float16>(8.0f);

        __riscv_vse8_v_i8mf4(vQuantABlk, vVAQuantI8, vVl);
      }
    }
  } else if (vVlenb == 32) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vABlkStride) {
      _Float16 *vASumPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vASubblkStride * cSubblkCount);
      _Float16 *vScaleAvgPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vABlkStride - sizeof(_Float16));
      float vScaleAvg = 0.0f;

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        const float *vASrcPtr = pAPtr + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m4(cKSubblkLen);
        vfloat32m4_t vVA = __riscv_vle32_v_f32m4(vASrcPtr, vVl);
        vfloat32m4_t vVAAbs = __riscv_vfabs_v_f32m4(vVA, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m4_f32m1(vVAAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        aScaleTemp[vKk] = vMaxAbsA / ((1 << 7) - 1);
        vScaleAvg += aScaleTemp[vKk];
      }

      vScaleAvg /= cSubblkCount;
      const float cScaleFactor = vScaleAvg ? 1.0f / vScaleAvg : 0.0f;
      vScaleAvgPtr[0] = static_cast<_Float16>(vScaleAvg);

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        uint8_t *vASubblkBase = pQuantAPtr + vKk * vASubblkStride;
        _Float16 *vScaleAPtr = reinterpret_cast<_Float16 *>(vASubblkBase);
        int8_t *vQuantABlk = reinterpret_cast<int8_t *>(vASubblkBase + sizeof(_Float16));
        const float *vASrcPtr = pAPtr + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m4(cKSubblkLen);
        vfloat32m4_t vVA = __riscv_vle32_v_f32m4(vASrcPtr, vVl);
        float vRepScaleA = aScaleTemp[vKk] ? 1.0f / aScaleTemp[vKk] : 0.0f;
        vScaleAPtr[0] = static_cast<_Float16>(aScaleTemp[vKk] * cScaleFactor);

        vfloat32m4_t vVAScale = __riscv_vfmul_vf_f32m4(vVA, vRepScaleA, vVl);
        vint16m2_t vVAQuant = __riscv_vfncvt_x_f_w_i16m2(vVAScale, vVl);
        vint8m1_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8m1(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8m1_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vKk] = static_cast<_Float16>(-vASum) * static_cast<_Float16>(8.0f);

        __riscv_vse8_v_i8m1(vQuantABlk, vVAQuantI8, vVl);
      }
    }
  } else {
    fQuantizeANrowI8HpRef<1>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void quantize_a_4row_i8_hp(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  constexpr size_t cKSubblkLen = 32;
  GGML_ASSERT(pBlkLen == 256);

  constexpr size_t cSubblkCount = 256 / cKSubblkLen;
  int64_t vABlkStride = q8_hp_blk_size(pBlkLen, true, true);
  int64_t vANrowBlockStride = vABlkStride * 4;
  int64_t vASubblkStride = q8_hp_blk_size(cKSubblkLen, false, false) * 4;
  size_t vVlenb = __riscv_vlenb();
  float aScaleTemp[cSubblkCount] = {0.0f};

  if (vVlenb == 128) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
      _Float16 *vASumPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vASubblkStride * cSubblkCount);
      _Float16 *vScaleAvgPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vANrowBlockStride - sizeof(_Float16) * 4);
      float vScaleAvg = 0.0f;

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        const float *vASrcPtr0 = pAPtr + 0 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr1 = pAPtr + 1 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr2 = pAPtr + 2 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr3 = pAPtr + 3 * pCountK + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m1(cKSubblkLen);
        vfloat32m1_t vVA0 = __riscv_vle32_v_f32m1(vASrcPtr0, vVl);
        vfloat32m1_t vVA1 = __riscv_vle32_v_f32m1(vASrcPtr1, vVl);
        vfloat32m1_t vVA2 = __riscv_vle32_v_f32m1(vASrcPtr2, vVl);
        vfloat32m1_t vVA3 = __riscv_vle32_v_f32m1(vASrcPtr3, vVl);
        vfloat32m1_t vVA0Abs = __riscv_vfabs_v_f32m1(vVA0, vVl);
        vfloat32m1_t vVA1Abs = __riscv_vfabs_v_f32m1(vVA1, vVl);
        vfloat32m1_t vVA2Abs = __riscv_vfabs_v_f32m1(vVA2, vVl);
        vfloat32m1_t vVA3Abs = __riscv_vfabs_v_f32m1(vVA3, vVl);

        vfloat32m1_t vVMaxAbs = __riscv_vfmax_vv_f32m1(vVA0Abs, vVA1Abs, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m1(vVMaxAbs, vVA2Abs, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m1(vVMaxAbs, vVA3Abs, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m1_f32m1(vVMaxAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        aScaleTemp[vKk] = vMaxAbsA / ((1 << 7) - 1);
        vScaleAvg += aScaleTemp[vKk];
      }

      vScaleAvg /= cSubblkCount;
      const float cScaleFactor = vScaleAvg ? 1.0f / vScaleAvg : 0.0f;
      vScaleAvgPtr[0] = static_cast<_Float16>(vScaleAvg);

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        uint8_t *vASubblkBase = pQuantAPtr + vKk * vASubblkStride;
        _Float16 *vScaleAPtr = reinterpret_cast<_Float16 *>(vASubblkBase);
        int8_t *vQuantABlk = reinterpret_cast<int8_t *>(vASubblkBase + sizeof(_Float16) * 4);
        const float *vASrcPtr0 = pAPtr + 0 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr1 = pAPtr + 1 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr2 = pAPtr + 2 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr3 = pAPtr + 3 * pCountK + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m1(cKSubblkLen);
        vfloat32m1_t vVA0 = __riscv_vle32_v_f32m1(vASrcPtr0, vVl);
        vfloat32m1_t vVA1 = __riscv_vle32_v_f32m1(vASrcPtr1, vVl);
        vfloat32m1_t vVA2 = __riscv_vle32_v_f32m1(vASrcPtr2, vVl);
        vfloat32m1_t vVA3 = __riscv_vle32_v_f32m1(vASrcPtr3, vVl);

        float vRepScaleA = aScaleTemp[vKk] ? 1.0f / aScaleTemp[vKk] : 0.0f;
        vScaleAPtr[0] = static_cast<_Float16>(aScaleTemp[vKk] * cScaleFactor);

        vfloat32m1_t vVA0Scale = __riscv_vfmul_vf_f32m1(vVA0, vRepScaleA, vVl);
        vfloat32m1_t vVA1Scale = __riscv_vfmul_vf_f32m1(vVA1, vRepScaleA, vVl);
        vfloat32m1_t vVA2Scale = __riscv_vfmul_vf_f32m1(vVA2, vRepScaleA, vVl);
        vfloat32m1_t vVA3Scale = __riscv_vfmul_vf_f32m1(vVA3, vRepScaleA, vVl);
        vint16mf2_t vVA0Quant = __riscv_vfncvt_x_f_w_i16mf2(vVA0Scale, vVl);
        vint16mf2_t vVA1Quant = __riscv_vfncvt_x_f_w_i16mf2(vVA1Scale, vVl);
        vint16mf2_t vVA2Quant = __riscv_vfncvt_x_f_w_i16mf2(vVA2Scale, vVl);
        vint16mf2_t vVA3Quant = __riscv_vfncvt_x_f_w_i16mf2(vVA3Scale, vVl);
        vint8mf4_t vVA0QuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVA0Quant, vVl);
        vint8mf4_t vVA1QuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVA1Quant, vVl);
        vint8mf4_t vVA2QuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVA2Quant, vVl);
        vint8mf4_t vVA3QuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVA3Quant, vVl);

        vint16m1_t vTmpSum0 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum1 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum2 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum3 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVA0Sum = __riscv_vwredsum_vs_i8mf4_i16m1(vVA0QuantI8, vTmpSum0, vVl);
        vint16m1_t vVA1Sum = __riscv_vwredsum_vs_i8mf4_i16m1(vVA1QuantI8, vTmpSum1, vVl);
        vint16m1_t vVA2Sum = __riscv_vwredsum_vs_i8mf4_i16m1(vVA2QuantI8, vTmpSum2, vVl);
        vint16m1_t vVA3Sum = __riscv_vwredsum_vs_i8mf4_i16m1(vVA3QuantI8, vTmpSum3, vVl);

        vASumPtr[0 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA0Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[1 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA1Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[2 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA2Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[3 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA3Sum)) * static_cast<_Float16>(8.0f);

        __riscv_vse8_v_i8mf4(vQuantABlk + 0 * cKSubblkLen, vVA0QuantI8, vVl);
        __riscv_vse8_v_i8mf4(vQuantABlk + 1 * cKSubblkLen, vVA1QuantI8, vVl);
        __riscv_vse8_v_i8mf4(vQuantABlk + 2 * cKSubblkLen, vVA2QuantI8, vVl);
        __riscv_vse8_v_i8mf4(vQuantABlk + 3 * cKSubblkLen, vVA3QuantI8, vVl);
      }
    }
  } else if (vVlenb == 32) {
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += vANrowBlockStride) {
      _Float16 *vASumPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vASubblkStride * cSubblkCount);
      _Float16 *vScaleAvgPtr = reinterpret_cast<_Float16 *>(pQuantAPtr + vANrowBlockStride - sizeof(_Float16) * 4);
      float vScaleAvg = 0.0f;

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        const float *vASrcPtr0 = pAPtr + 0 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr1 = pAPtr + 1 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr2 = pAPtr + 2 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr3 = pAPtr + 3 * pCountK + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m4(cKSubblkLen);
        vfloat32m4_t vVA0 = __riscv_vle32_v_f32m4(vASrcPtr0, vVl);
        vfloat32m4_t vVA1 = __riscv_vle32_v_f32m4(vASrcPtr1, vVl);
        vfloat32m4_t vVA2 = __riscv_vle32_v_f32m4(vASrcPtr2, vVl);
        vfloat32m4_t vVA3 = __riscv_vle32_v_f32m4(vASrcPtr3, vVl);

        vfloat32m4_t vVA0Abs = __riscv_vfabs_v_f32m4(vVA0, vVl);
        vfloat32m4_t vVA1Abs = __riscv_vfabs_v_f32m4(vVA1, vVl);
        vfloat32m4_t vVA2Abs = __riscv_vfabs_v_f32m4(vVA2, vVl);
        vfloat32m4_t vVA3Abs = __riscv_vfabs_v_f32m4(vVA3, vVl);

        vfloat32m4_t vVMaxAbs = __riscv_vfmax_vv_f32m4(vVA0Abs, vVA1Abs, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m4(vVMaxAbs, vVA2Abs, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m4(vVMaxAbs, vVA3Abs, vVl);

        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVAMax = __riscv_vfredmax_vs_f32m4_f32m1(vVMaxAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVAMax);

        aScaleTemp[vKk] = vMaxAbsA / ((1 << 7) - 1);
        vScaleAvg += aScaleTemp[vKk];
      }

      vScaleAvg /= cSubblkCount;
      const float cScaleFactor = vScaleAvg ? 1.0f / vScaleAvg : 0.0f;
      vScaleAvgPtr[0] = static_cast<_Float16>(vScaleAvg);

      for (size_t vKk = 0; vKk < cSubblkCount; ++vKk) {
        uint8_t *vASubblkBase = pQuantAPtr + vKk * vASubblkStride;
        _Float16 *vScaleAPtr = reinterpret_cast<_Float16 *>(vASubblkBase);
        int8_t *vQuantABlk = reinterpret_cast<int8_t *>(vASubblkBase + sizeof(_Float16) * 4);
        const float *vASrcPtr0 = pAPtr + 0 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr1 = pAPtr + 1 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr2 = pAPtr + 2 * pCountK + vK + vKk * cKSubblkLen;
        const float *vASrcPtr3 = pAPtr + 3 * pCountK + vK + vKk * cKSubblkLen;

        size_t vVl = __riscv_vsetvl_e32m4(cKSubblkLen);
        vfloat32m4_t vVA0 = __riscv_vle32_v_f32m4(vASrcPtr0, vVl);
        vfloat32m4_t vVA1 = __riscv_vle32_v_f32m4(vASrcPtr1, vVl);
        vfloat32m4_t vVA2 = __riscv_vle32_v_f32m4(vASrcPtr2, vVl);
        vfloat32m4_t vVA3 = __riscv_vle32_v_f32m4(vASrcPtr3, vVl);

        float vRepScaleA = aScaleTemp[vKk] ? 1.0f / aScaleTemp[vKk] : 0.0f;
        vScaleAPtr[0] = static_cast<_Float16>(aScaleTemp[vKk] * cScaleFactor);

        vfloat32m4_t vVA0Scale = __riscv_vfmul_vf_f32m4(vVA0, vRepScaleA, vVl);
        vfloat32m4_t vVA1Scale = __riscv_vfmul_vf_f32m4(vVA1, vRepScaleA, vVl);
        vfloat32m4_t vVA2Scale = __riscv_vfmul_vf_f32m4(vVA2, vRepScaleA, vVl);
        vfloat32m4_t vVA3Scale = __riscv_vfmul_vf_f32m4(vVA3, vRepScaleA, vVl);
        vint16m2_t vVA0Quant = __riscv_vfncvt_x_f_w_i16m2(vVA0Scale, vVl);
        vint16m2_t vVA1Quant = __riscv_vfncvt_x_f_w_i16m2(vVA1Scale, vVl);
        vint16m2_t vVA2Quant = __riscv_vfncvt_x_f_w_i16m2(vVA2Scale, vVl);
        vint16m2_t vVA3Quant = __riscv_vfncvt_x_f_w_i16m2(vVA3Scale, vVl);
        vint8m1_t vVA0QuantI8 = __riscv_vncvt_x_x_w_i8m1(vVA0Quant, vVl);
        vint8m1_t vVA1QuantI8 = __riscv_vncvt_x_x_w_i8m1(vVA1Quant, vVl);
        vint8m1_t vVA2QuantI8 = __riscv_vncvt_x_x_w_i8m1(vVA2Quant, vVl);
        vint8m1_t vVA3QuantI8 = __riscv_vncvt_x_x_w_i8m1(vVA3Quant, vVl);

        vint16m1_t vTmpSum0 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum1 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum2 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vTmpSum3 = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVA0Sum = __riscv_vwredsum_vs_i8m1_i16m1(vVA0QuantI8, vTmpSum0, vVl);
        vint16m1_t vVA1Sum = __riscv_vwredsum_vs_i8m1_i16m1(vVA1QuantI8, vTmpSum1, vVl);
        vint16m1_t vVA2Sum = __riscv_vwredsum_vs_i8m1_i16m1(vVA2QuantI8, vTmpSum2, vVl);
        vint16m1_t vVA3Sum = __riscv_vwredsum_vs_i8m1_i16m1(vVA3QuantI8, vTmpSum3, vVl);

        vASumPtr[0 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA0Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[1 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA1Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[2 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA2Sum)) * static_cast<_Float16>(8.0f);
        vASumPtr[3 * cSubblkCount + vKk] =
            static_cast<_Float16>(-__riscv_vmv_x_s_i16m1_i16(vVA3Sum)) * static_cast<_Float16>(8.0f);

        __riscv_vse8_v_i8m1(vQuantABlk + 0 * cKSubblkLen, vVA0QuantI8, vVl);
        __riscv_vse8_v_i8m1(vQuantABlk + 1 * cKSubblkLen, vVA1QuantI8, vVl);
        __riscv_vse8_v_i8m1(vQuantABlk + 2 * cKSubblkLen, vVA2QuantI8, vVl);
        __riscv_vse8_v_i8m1(vQuantABlk + 3 * cKSubblkLen, vVA3QuantI8, vVl);
      }
    }
  } else {
    fQuantizeANrowI8HpRef<4>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void quantize_a_row_i8k(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  GGML_ASSERT(pBlkLen == 256);
  constexpr int64_t cABlkStride = q8k_blk_size(256);
  constexpr int64_t cASumSize = 256 / 16;
  size_t vVlenb = __riscv_vlenb();

  if (vVlenb == 128) {
    // vlen = 1024 bits, can process 32 float32 elements with m1
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += cABlkStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float));
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) + sizeof(int16_t) * cASumSize);

      // Find max absolute value across all 256 elements
      size_t vVl = __riscv_vsetvl_e32m1(16);
      vfloat32m1_t vVMaxAbs = __riscv_vfmv_v_f_f32m1(0.0f, vVl);

      for (size_t vBki = 0; vBki < cASumSize; vBki++) {
        vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vK + vBki * 16, vVl);
        vfloat32m1_t vVAAbs = __riscv_vfabs_v_f32m1(vVA, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m1(vVAAbs, vVMaxAbs, vVl);
      }
      vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
      vfloat32m1_t vVLocalMax = __riscv_vfredmax_vs_f32m1_f32m1(vVMaxAbs, vTmp, vVl);
      float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVLocalMax);

      float vScaleA = vMaxAbsA / ((1 << 7) - 1);
      float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
      vScaleAPtr[0] = vScaleA;

      // Quantize and compute sums for each 16-element group
      for (size_t vBki = 0; vBki < cASumSize; vBki++) {
        vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vK + vBki * 16, vVl);
        vfloat32m1_t vVAScale = __riscv_vfmul_vf_f32m1(vVA, vRepScaleA, vVl);
        vint16mf2_t vVAQuant = __riscv_vfncvt_x_f_w_i16mf2(vVAScale, vVl);
        vint8mf4_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf4_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vBki] = -vASum;

        __riscv_vse8_v_i8mf4(vQuantABlk + vBki * 16, vVAQuantI8, vVl);
      }
    }
  } else if (vVlenb == 32) {
    // vlen = 256 bits, can process 8 float32 elements with m1
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += cABlkStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float));
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) + sizeof(int16_t) * cASumSize);

      // Find max absolute value across all 256 elements
      size_t vVl = __riscv_vsetvl_e32m2(16);
      vfloat32m2_t vVMaxAbs = __riscv_vfmv_v_f_f32m2(0.0f, vVl);

      for (size_t vBki = 0; vBki < cASumSize; vBki++) {
        vfloat32m2_t vVA = __riscv_vle32_v_f32m2(pAPtr + vK + vBki * 16, vVl);
        vfloat32m2_t vVAAbs = __riscv_vfabs_v_f32m2(vVA, vVl);
        vVMaxAbs = __riscv_vfmax_vv_f32m2(vVAAbs, vVMaxAbs, vVl);
      }
      vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
      vfloat32m1_t vVLocalMax = __riscv_vfredmax_vs_f32m2_f32m1(vVMaxAbs, vTmp, vVl);
      float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVLocalMax);

      float vScaleA = vMaxAbsA / ((1 << 7) - 1);
      float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
      vScaleAPtr[0] = vScaleA;

      // Quantize and compute sums for each 16-element group
      for (size_t vBki = 0; vBki < cASumSize; vBki++) {
        vfloat32m2_t vVA = __riscv_vle32_v_f32m2(pAPtr + vK + vBki * 16, vVl);
        vfloat32m2_t vVAScale = __riscv_vfmul_vf_f32m2(vVA, vRepScaleA, vVl);
        vint16m1_t vVAQuant = __riscv_vfncvt_x_f_w_i16m1(vVAScale, vVl);
        vint8mf2_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf2(vVAQuant, vVl);

        vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
        vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf2_i16m1(vVAQuantI8, vTmpSum, vVl);
        int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
        vASumPtr[vBki] = -vASum;

        __riscv_vse8_v_i8mf2(vQuantABlk + vBki * 16, vVAQuantI8, vVl);
      }
    }
  } else {
    fQuantizeANrowI8kRef<1>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void quantize_a_4row_i8k(size_t pBlkLen, const float *pAPtr, size_t pCountK, uint8_t *pQuantAPtr) {
  GGML_ASSERT(pBlkLen == 256);
  constexpr int64_t cABlkStride = q8k_blk_size(256);
  constexpr int64_t cANrowBlockStride = cABlkStride * 4;
  constexpr int64_t cASumSize = 256 / 16;
  size_t vVlenb = __riscv_vlenb();

  if (vVlenb == 128) {
    // vlen = 1024 bits
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += cANrowBlockStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * 4);
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * 4 + sizeof(int16_t) * cASumSize * 4);

      for (size_t vMi = 0; vMi < 4; vMi++) {
        // Find max absolute value across all 256 elements for this row
        size_t vVl = __riscv_vsetvl_e32m1(16);
        vfloat32m1_t vVMaxAbs = __riscv_vfmv_v_f_f32m1(0.0f, vVl);

        for (size_t vBki = 0; vBki < cASumSize; vBki++) {
          vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vMi * pCountK + vK + vBki * 16, vVl);
          vfloat32m1_t vVAAbs = __riscv_vfabs_v_f32m1(vVA, vVl);
          vVMaxAbs = __riscv_vfmax_vv_f32m1(vVAAbs, vVMaxAbs, vVl);
        }
        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVLocalMax = __riscv_vfredmax_vs_f32m1_f32m1(vVMaxAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVLocalMax);

        float vScaleA = vMaxAbsA / ((1 << 7) - 1);
        float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
        vScaleAPtr[vMi] = vScaleA;

        // Quantize and compute sums for each 16-element group
        for (size_t vBki = 0; vBki < cASumSize; vBki++) {
          vfloat32m1_t vVA = __riscv_vle32_v_f32m1(pAPtr + vMi * pCountK + vK + vBki * 16, vVl);
          vfloat32m1_t vVAScale = __riscv_vfmul_vf_f32m1(vVA, vRepScaleA, vVl);
          vint16mf2_t vVAQuant = __riscv_vfncvt_x_f_w_i16mf2(vVAScale, vVl);
          vint8mf4_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf4(vVAQuant, vVl);

          vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
          vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf4_i16m1(vVAQuantI8, vTmpSum, vVl);
          int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
          vASumPtr[vMi * cASumSize + vBki] = -vASum;

          __riscv_vse8_v_i8mf4(vQuantABlk + vMi * pBlkLen + vBki * 16, vVAQuantI8, vVl);
        }
      }
    }
  } else if (vVlenb == 32) {
    // vlen = 256 bits
    for (size_t vK = 0; vK < pCountK; vK += pBlkLen, pQuantAPtr += cANrowBlockStride) {
      float *vScaleAPtr = reinterpret_cast<float *>(pQuantAPtr);
      int16_t *vASumPtr = reinterpret_cast<int16_t *>(pQuantAPtr + sizeof(float) * 4);
      int8_t *vQuantABlk = reinterpret_cast<int8_t *>(pQuantAPtr + sizeof(float) * 4 + sizeof(int16_t) * cASumSize * 4);

      for (size_t vMi = 0; vMi < 4; vMi++) {
        // Find max absolute value across all 256 elements for this row
        size_t vVl = __riscv_vsetvl_e32m2(16);
        vfloat32m2_t vVMaxAbs = __riscv_vfmv_v_f_f32m2(0.0f, vVl);

        for (size_t vBki = 0; vBki < cASumSize; vBki++) {
          vfloat32m2_t vVA = __riscv_vle32_v_f32m2(pAPtr + vMi * pCountK + vK + vBki * 16, vVl);
          vfloat32m2_t vVAAbs = __riscv_vfabs_v_f32m2(vVA, vVl);
          vVMaxAbs = __riscv_vfmax_vv_f32m2(vVAAbs, vVMaxAbs, vVl);
        }
        vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
        vfloat32m1_t vVLocalMax = __riscv_vfredmax_vs_f32m2_f32m1(vVMaxAbs, vTmp, vVl);
        float vMaxAbsA = __riscv_vfmv_f_s_f32m1_f32(vVLocalMax);

        float vScaleA = vMaxAbsA / ((1 << 7) - 1);
        float vRepScaleA = vScaleA ? 1.0f / vScaleA : 0.0f;
        vScaleAPtr[vMi] = vScaleA;

        // Quantize and compute sums for each 16-element group
        for (size_t vBki = 0; vBki < cASumSize; vBki++) {
          vfloat32m2_t vVA = __riscv_vle32_v_f32m2(pAPtr + vMi * pCountK + vK + vBki * 16, vVl);
          vfloat32m2_t vVAScale = __riscv_vfmul_vf_f32m2(vVA, vRepScaleA, vVl);
          vint16m1_t vVAQuant = __riscv_vfncvt_x_f_w_i16m1(vVAScale, vVl);
          vint8mf2_t vVAQuantI8 = __riscv_vncvt_x_x_w_i8mf2(vVAQuant, vVl);

          vint16m1_t vTmpSum = __riscv_vmv_v_x_i16m1(0, vVl);
          vint16m1_t vVASum = __riscv_vwredsum_vs_i8mf2_i16m1(vVAQuantI8, vTmpSum, vVl);
          int16_t vASum = __riscv_vmv_x_s_i16m1_i16(vVASum);
          vASumPtr[vMi * cASumSize + vBki] = -vASum;

          __riscv_vse8_v_i8mf2(vQuantABlk + vMi * pBlkLen + vBki * 16, vVAQuantI8, vVl);
        }
      }
    }
  } else {
    fQuantizeANrowI8kRef<4>(pBlkLen, pAPtr, pCountK, pQuantAPtr);
  }
}

void forward_cpy_with_permute(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *vSrc0 = pOp->src[0];
  ggml_tensor *vDst = pOp;
  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // [batch, m, n] -> [batch, n, m]
  int64_t vBatch = vSrc0->ne[2] * vSrc0->ne[3];
  int64_t vM = vSrc0->ne[1];
  int64_t vN = vSrc0->ne[0];

  int64_t vBatchStride = vSrc0->nb[2];
  int64_t vMSrcStride = vSrc0->nb[0];
  int64_t vNSrcStride = vSrc0->nb[1];
  int64_t vNDstStride = vNSrcStride * vM;

  fPermuteTransposeImpl(vSrc0, vDst, vBatch, vM, vN, vBatchStride, vMSrcStride, vNSrcStride, vNDstStride, cIth, cNth);
}

void forward_cont_with_permute(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *vSrc0 = pOp->src[0];
  ggml_tensor *vDst = pOp;
  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // [batch, m, n] -> [batch, n, m]
  int64_t vBatch = vDst->ne[2] * vDst->ne[3];
  int64_t vN = vDst->ne[1];
  int64_t vM = vDst->ne[0];

  int64_t vBatchStride = vDst->nb[2];
  int64_t vMSrcStride = vSrc0->nb[0];
  int64_t vNSrcStride = vSrc0->nb[1];
  int64_t vNDstStride = vDst->nb[1];

  fPermuteTransposeImpl(vSrc0, vDst, vBatch, vM, vN, vBatchStride, vMSrcStride, vNSrcStride, vNDstStride, cIth, cNth);
}

void forward_norm_f32(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  ggml_tensor *dst = pOp;
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(src0->nb[0] == sizeof(float));

  int vIth = pParams->ith;
  int vNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float vEpsilon = *((float *)dst->op_params);

  GGML_ASSERT(vEpsilon > 0.0f);

  auto *vInput = (char *)src0->data;
  auto *vOutput = (char *)dst->data;

  const auto cHiddenSize = ne00;
  const auto cTaskCount = ne01 * ne02 * ne03;
  const auto cTaskPerThread = (cTaskCount + vNth - 1) / vNth;

  const auto cTaskBegin = vIth * cTaskPerThread;
  const auto cTaskEnd = std::min((vIth + 1) * cTaskPerThread, cTaskCount);

  for (auto vTaskIdx = cTaskBegin; vTaskIdx < cTaskEnd; vTaskIdx++) {
    int64_t vI03 = vTaskIdx / (ne02 * ne01);
    int64_t vI02 = (vTaskIdx - vI03 * ne02 * ne01) / ne01;
    int64_t vI01 = (vTaskIdx - vI03 * ne02 * ne01 - vI02 * ne01);

    auto *vPInput = (float *)(vInput + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
    auto *vPOutput = (float *)(vOutput + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);
    auto *vPTempOutput = vPOutput;

    size_t vGvl = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vSum = __riscv_vfmv_v_f_f32m4(0.f, vGvl);
    vfloat32m4_t vSumSq = __riscv_vfmv_v_f_f32m4(0.f, vGvl);
    int64_t vLength = cHiddenSize;
    while (vLength > 0) {
      vGvl = __riscv_vsetvl_e32m4(vLength);
      // load data
      vfloat32m4_t vSrcData = __riscv_vle32_v_f32m4(vPInput, vGvl);

      vSum = __riscv_vfadd_vv_f32m4(vSum, vSrcData, vGvl);
      vSumSq = __riscv_vfmacc_vv_f32m4(vSumSq, vSrcData, vSrcData, vGvl);

      __riscv_vse32_v_f32m4(vPTempOutput, vSrcData, vGvl);

      vPInput += vGvl;
      vPTempOutput += vGvl;
      vLength -= vGvl;
    }

    vGvl = __riscv_vsetvlmax_e32m1();

    float vMean = 0.f;
    vfloat32m1_t vZeroV = __riscv_vfmv_v_f_f32m1(0.f, vGvl);
    vfloat32m1_t vMeanV =
        __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vSum, 0), __riscv_vget_v_f32m4_f32m1(vSum, 1), vGvl);
    vMeanV = __riscv_vfadd_vv_f32m1(vMeanV, __riscv_vget_v_f32m4_f32m1(vSum, 2), vGvl);
    vMeanV = __riscv_vfadd_vv_f32m1(vMeanV, __riscv_vget_v_f32m4_f32m1(vSum, 3), vGvl);
    vMeanV = __riscv_vfredusum_vs_f32m1_f32m1(vMeanV, vZeroV, vGvl);
    vMean = __riscv_vfmv_f_s_f32m1_f32(vMeanV);
    vMean /= cHiddenSize;

    vfloat32m1_t vMeanSquareV =
        __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vSumSq, 0), __riscv_vget_v_f32m4_f32m1(vSumSq, 1), vGvl);
    vMeanSquareV = __riscv_vfadd_vv_f32m1(vMeanSquareV, __riscv_vget_v_f32m4_f32m1(vSumSq, 2), vGvl);
    vMeanSquareV = __riscv_vfadd_vv_f32m1(vMeanSquareV, __riscv_vget_v_f32m4_f32m1(vSumSq, 3), vGvl);
    vMeanSquareV = __riscv_vfredusum_vs_f32m1_f32m1(vMeanSquareV, vZeroV, vGvl);

    float vMeanSquare = __riscv_vfmv_f_s_f32m1_f32(vMeanSquareV);
    vMeanSquare /= cHiddenSize;
    vMeanSquare = sqrt(vMeanSquare - vMean * vMean + vEpsilon);

    vMeanSquare = 1.0f / vMeanSquare;
    vLength = cHiddenSize;
    vPTempOutput = vPOutput;

    while (vLength > 0) {
      vGvl = __riscv_vsetvl_e32m4(vLength);
      vfloat32m4_t vSrcData = __riscv_vle32_v_f32m4(vPTempOutput, vGvl);
      vSrcData = __riscv_vfsub_vf_f32m4(vSrcData, vMean, vGvl);
      vSrcData = __riscv_vfmul_vf_f32m4(vSrcData, vMeanSquare, vGvl);
      __riscv_vse32_v_f32m4(vPOutput, vSrcData, vGvl);
      vPTempOutput += vGvl;
      vPOutput += vGvl;
      vLength -= vGvl;
    }
  }
}

template <ggml_op op_type, typename T> void forward_binary(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  const ggml_tensor *src1 = pOp->src[1];
  ggml_tensor *dst = pOp;
  GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

  auto vSrc0Rows = ggml_nrows(src0);
  auto vSrc1Rows = ggml_nrows(src1);

  int vIth = pParams->ith;
  int vNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(T));
  GGML_ASSERT(nb00 == sizeof(T));

  const auto [ir0, ir1] = get_thread_range(pParams, src0);

  auto vComputeFuncVv = [&](int64_t pBlkLen, int64_t pR, T *pSrc0Ptr, T *pSrc1Ptr, T *pDstPtr) {
    int64_t vIdx = 0;
    if constexpr (op_type == GGML_OP_ADD) {
      if constexpr (std::is_same_v<T, float>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e32m4(pBlkLen);
          vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(pSrc0Ptr + vIdx + pR, vVl);
          vfloat32m4_t vRhs = __riscv_vle32_v_f32m4(pSrc1Ptr + vIdx, vVl);
          vfloat32m4_t vRes = __riscv_vfadd_vv_f32m4(vLhs, vRhs, vVl);
          __riscv_vse32_v_f32m4(pDstPtr + vIdx + pR, vRes, vVl);
        }
      } else if constexpr (std::is_same_v<T, _Float16>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e16m4(pBlkLen);
          vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((pSrc0Ptr + vIdx + pR), vVl);
          vfloat16m4_t vRhs = __riscv_vle16_v_f16m4((pSrc1Ptr + vIdx), vVl);
          vfloat16m4_t vRes = __riscv_vfadd_vv_f16m4(vLhs, vRhs, vVl);
          __riscv_vse16_v_f16m4((pDstPtr + vIdx + pR), vRes, vVl);
        }
      } else {
        GGML_ABORT("fatal error");
      }
    } else if constexpr (op_type == GGML_OP_SUB) {
      if constexpr (std::is_same_v<T, float>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e32m4(pBlkLen);
          vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(pSrc0Ptr + vIdx + pR, vVl);
          vfloat32m4_t vRhs = __riscv_vle32_v_f32m4(pSrc1Ptr + vIdx, vVl);
          vfloat32m4_t vRes = __riscv_vfsub_vv_f32m4(vLhs, vRhs, vVl);
          __riscv_vse32_v_f32m4(pDstPtr + vIdx + pR, vRes, vVl);
        }
      } else if constexpr (std::is_same_v<T, _Float16>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e16m4(pBlkLen);
          vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((pSrc0Ptr + vIdx + pR), vVl);
          vfloat16m4_t vRhs = __riscv_vle16_v_f16m4((pSrc1Ptr + vIdx), vVl);
          vfloat16m4_t vRes = __riscv_vfsub_vv_f16m4(vLhs, vRhs, vVl);
          __riscv_vse16_v_f16m4((pDstPtr + vIdx + pR), vRes, vVl);
        }
      } else {
        GGML_ABORT("fatal error");
      }
    } else if constexpr (op_type == GGML_OP_MUL) {
      if constexpr (std::is_same_v<T, float>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e32m4(pBlkLen);
          vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(pSrc0Ptr + vIdx + pR, vVl);
          vfloat32m4_t vRhs = __riscv_vle32_v_f32m4(pSrc1Ptr + vIdx, vVl);
          vfloat32m4_t vRes = __riscv_vfmul_vv_f32m4(vLhs, vRhs, vVl);
          __riscv_vse32_v_f32m4(pDstPtr + vIdx + pR, vRes, vVl);
        }
      } else if constexpr (std::is_same_v<T, _Float16>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e16m4(pBlkLen);
          vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((pSrc0Ptr + vIdx + pR), vVl);
          vfloat16m4_t vRhs = __riscv_vle16_v_f16m4((pSrc1Ptr + vIdx), vVl);
          vfloat16m4_t vRes = __riscv_vfmul_vv_f16m4(vLhs, vRhs, vVl);
          __riscv_vse16_v_f16m4((pDstPtr + vIdx + pR), vRes, vVl);
        }
      } else {
        GGML_ABORT("fatal error");
      }
    } else if constexpr (op_type == GGML_OP_DIV) {
      if constexpr (std::is_same_v<T, float>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e32m4(pBlkLen);
          vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(pSrc0Ptr + vIdx + pR, vVl);
          vfloat32m4_t vRhs = __riscv_vle32_v_f32m4(pSrc1Ptr + vIdx, vVl);
          vfloat32m4_t vRes = __riscv_vfdiv_vv_f32m4(vLhs, vRhs, vVl);
          __riscv_vse32_v_f32m4(pDstPtr + vIdx + pR, vRes, vVl);
        }
      } else if constexpr (std::is_same_v<T, _Float16>) {
        for (size_t vVl; pBlkLen > 0; pBlkLen -= vVl, vIdx += vVl) {
          vVl = __riscv_vsetvl_e16m4(pBlkLen);
          vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((pSrc0Ptr + vIdx + pR), vVl);
          vfloat16m4_t vRhs = __riscv_vle16_v_f16m4((pSrc1Ptr + vIdx), vVl);
          vfloat16m4_t vRes = __riscv_vfdiv_vv_f16m4(vLhs, vRhs, vVl);
          __riscv_vse16_v_f16m4((pDstPtr + vIdx + pR), vRes, vVl);
        }
      } else {
        GGML_ABORT("fatal error");
      }
    } else {
      GGML_ABORT("fatal error");
    }
  };

  if (vSrc0Rows == vSrc1Rows && vSrc0Rows == 1 && ne00 == ne10) {
    int64_t vTaskPerThread = (ne00 + vNth - 1) / vNth;
    int64_t vTaskBegin = vIth * vTaskPerThread;
    int64_t vTaskEnd = std::min((vIth + 1) * vTaskPerThread, ne00);

    T *vDstPtr = ((T *)dst->data) + vTaskBegin;
    T *vSrc0Ptr = ((T *)src0->data) + vTaskBegin;
    T *vSrc1Ptr = ((T *)src1->data) + vTaskBegin;

    vComputeFuncVv(vTaskEnd - vTaskBegin, 0, vSrc0Ptr, vSrc1Ptr, vDstPtr);
  } else if (ne10 > 1) {
    for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
      const int64_t cI03 = vIr / (ne02 * ne01);
      const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
      const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

      const int64_t cI13 = cI03 % ne13;
      const int64_t cI12 = cI02 % ne12;
      const int64_t cI11 = cI01 % ne11;

      T *vDstPtr = (T *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);
      T *vSrc0Ptr = (T *)((char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);
      T *vSrc1Ptr = (T *)((char *)src1->data + cI13 * nb13 + cI12 * nb12 + cI11 * nb11);

      // src1 is broadcastable across src0 and dst in i1, i2, i3
      for (int64_t vR = 0; vR < ne00; vR += ne10) {
        vComputeFuncVv(ne10, vR, vSrc0Ptr, vSrc1Ptr, vDstPtr);
      }
    }
  } else {
    for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
      const int64_t cI03 = vIr / (ne02 * ne01);
      const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
      const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

      const int64_t cI13 = cI03 % ne13;
      const int64_t cI12 = cI02 % ne12;
      const int64_t cI11 = cI01 % ne11;

      T *vDstPtr = (T *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);
      T *vSrc0Ptr = (T *)((char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);
      T *vSrc1Ptr = (T *)((char *)src1->data + cI13 * nb13 + cI12 * nb12 + cI11 * nb11);

      T vRhsScalar = vSrc1Ptr[0];
      int64_t vBlkLen = ne00;
      int64_t vR = 0;

      for (size_t vVl; vBlkLen > 0; vBlkLen -= vVl, vR += vVl) {
        if constexpr (op_type == GGML_OP_ADD) {
          if constexpr (std::is_same_v<T, float>) {
            vVl = __riscv_vsetvl_e32m4(vBlkLen);
            vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(vSrc0Ptr + vR, vVl);
            vfloat32m4_t vRes = __riscv_vfadd_vf_f32m4(vLhs, vRhsScalar, vVl);
            __riscv_vse32_v_f32m4(vDstPtr + vR, vRes, vVl);
          } else if constexpr (std::is_same_v<T, _Float16>) {
            vVl = __riscv_vsetvl_e16m4(vBlkLen);
            vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((vSrc0Ptr + vR), vVl);
            vfloat16m4_t vRes = __riscv_vfadd_vf_f16m4(vLhs, vRhsScalar, vVl);
            __riscv_vse16_v_f16m4((vDstPtr + vR), vRes, vVl);
          } else {
            GGML_ABORT("fatal error");
          }
        } else if constexpr (op_type == GGML_OP_SUB) {
          if constexpr (std::is_same_v<T, float>) {
            vVl = __riscv_vsetvl_e32m4(vBlkLen);
            vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(vSrc0Ptr + vR, vVl);
            vfloat32m4_t vRes = __riscv_vfsub_vf_f32m4(vLhs, vRhsScalar, vVl);
            __riscv_vse32_v_f32m4(vDstPtr + vR, vRes, vVl);
          } else if constexpr (std::is_same_v<T, _Float16>) {
            vVl = __riscv_vsetvl_e16m4(vBlkLen);
            vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((vSrc0Ptr + vR), vVl);
            vfloat16m4_t vRes = __riscv_vfsub_vf_f16m4(vLhs, vRhsScalar, vVl);
            __riscv_vse16_v_f16m4((vDstPtr + vR), vRes, vVl);
          } else {
            GGML_ABORT("fatal error");
          }
        } else if constexpr (op_type == GGML_OP_MUL) {
          if constexpr (std::is_same_v<T, float>) {
            vVl = __riscv_vsetvl_e32m4(vBlkLen);
            vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(vSrc0Ptr + vR, vVl);
            vfloat32m4_t vRes = __riscv_vfmul_vf_f32m4(vLhs, vRhsScalar, vVl);
            __riscv_vse32_v_f32m4(vDstPtr + vR, vRes, vVl);
          } else if constexpr (std::is_same_v<T, _Float16>) {
            vVl = __riscv_vsetvl_e16m4(vBlkLen);
            vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((vSrc0Ptr + vR), vVl);
            vfloat16m4_t vRes = __riscv_vfmul_vf_f16m4(vLhs, vRhsScalar, vVl);
            __riscv_vse16_v_f16m4((vDstPtr + vR), vRes, vVl);
          } else {
            GGML_ABORT("fatal error");
          }
        } else if constexpr (op_type == GGML_OP_DIV) {
          if constexpr (std::is_same_v<T, float>) {
            vVl = __riscv_vsetvl_e32m4(vBlkLen);
            vfloat32m4_t vLhs = __riscv_vle32_v_f32m4(vSrc0Ptr + vR, vVl);
            vfloat32m4_t vRes = __riscv_vfdiv_vf_f32m4(vLhs, vRhsScalar, vVl);
            __riscv_vse32_v_f32m4(vDstPtr + vR, vRes, vVl);
          } else if constexpr (std::is_same_v<T, _Float16>) {
            vVl = __riscv_vsetvl_e16m4(vBlkLen);
            vfloat16m4_t vLhs = __riscv_vle16_v_f16m4((vSrc0Ptr + vR), vVl);
            vfloat16m4_t vRes = __riscv_vfdiv_vf_f16m4(vLhs, vRhsScalar, vVl);
            __riscv_vse16_v_f16m4((vDstPtr + vR), vRes, vVl);
          } else {
            GGML_ABORT("fatal error");
          }
        } else {
          GGML_ABORT("fatal error");
        }
      }
    }
  }
}

template <typename T> void forward_sum_rows(const ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  ggml_tensor *dst = pOp;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne0 == 1);
  GGML_ASSERT(ne1 == ne01);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  int64_t vNTask = ne01 * ne02 * ne03;
  int64_t vTaskPerThread = (vNTask + cNth - 1) / cNth;
  int64_t vIrStart = cIth * vTaskPerThread;
  int64_t vIrEnd = std::min(vIrStart + vTaskPerThread, vNTask);

  for (int64_t vIr = vIrStart; vIr < vIrEnd; vIr++) {
    const int64_t cI3 = vIr / (ne02 * ne01);
    const int64_t cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int64_t cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    T *vSrcRow = (T *)((char *)src0->data + cI1 * nb01 + cI2 * nb02 + cI3 * nb03);
    T *vDstRow = (T *)((char *)pOp->data + cI1 * nb1 + cI2 * nb2 + cI3 * nb3);

    float vRowSum = 0;

    if constexpr (std::is_same_v<T, float>) {
      size_t vGvl = __riscv_vsetvlmax_e32m4();
      vfloat32m4_t vAccVec = __riscv_vfmv_v_f_f32m4(0.0f, vGvl);
      int64_t vLength = ne00;
      const float *vPData = vSrcRow;

      while (vLength > 0) {
        size_t vVl = __riscv_vsetvl_e32m4(vLength);
        vfloat32m4_t vVec = __riscv_vle32_v_f32m4(vPData, vVl);
        vAccVec = __riscv_vfadd_vv_f32m4(vAccVec, vVec, vVl);
        vPData += vVl;
        vLength -= vVl;
      }

      vGvl = __riscv_vsetvlmax_e32m1();
      vfloat32m1_t vZeroV = __riscv_vfmv_v_f_f32m1(0.0f, vGvl);
      vfloat32m1_t vSumV =
          __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vAccVec, 0), __riscv_vget_v_f32m4_f32m1(vAccVec, 1), vGvl);
      vSumV = __riscv_vfadd_vv_f32m1(vSumV, __riscv_vget_v_f32m4_f32m1(vAccVec, 2), vGvl);
      vSumV = __riscv_vfadd_vv_f32m1(vSumV, __riscv_vget_v_f32m4_f32m1(vAccVec, 3), vGvl);
      vSumV = __riscv_vfredusum_vs_f32m1_f32m1(vSumV, vZeroV, vGvl);
      vRowSum = __riscv_vfmv_f_s_f32m1_f32(vSumV);
    } else if constexpr (std::is_same_v<T, _Float16>) {
      size_t vGvl = __riscv_vsetvlmax_e16m2();
      vfloat32m4_t vAccVec = __riscv_vfmv_v_f_f32m4(0.0f, vGvl);
      int64_t vLength = ne00;
      const _Float16 *vPData = vSrcRow;

      while (vLength > 0) {
        size_t vVl = __riscv_vsetvl_e16m2(vLength);
        vfloat16m2_t vVecF16 = __riscv_vle16_v_f16m2(vPData, vVl);
        vfloat32m4_t vVecF32 = __riscv_vfwcvt_f_f_v_f32m4(vVecF16, vVl);
        vAccVec = __riscv_vfadd_vv_f32m4(vAccVec, vVecF32, vVl);
        vPData += vVl;
        vLength -= vVl;
      }

      vGvl = __riscv_vsetvlmax_e32m1();
      vfloat32m1_t vZeroV = __riscv_vfmv_v_f_f32m1(0.0f, vGvl);
      vfloat32m1_t vSumV =
          __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(vAccVec, 0), __riscv_vget_v_f32m4_f32m1(vAccVec, 1), vGvl);
      vSumV = __riscv_vfadd_vv_f32m1(vSumV, __riscv_vget_v_f32m4_f32m1(vAccVec, 2), vGvl);
      vSumV = __riscv_vfadd_vv_f32m1(vSumV, __riscv_vget_v_f32m4_f32m1(vAccVec, 3), vGvl);
      vSumV = __riscv_vfredusum_vs_f32m1_f32m1(vSumV, vZeroV, vGvl);
      vRowSum = __riscv_vfmv_f_s_f32m1_f32(vSumV);
    } else {
      GGML_ABORT("fatal error");
    }

    vDstRow[0] = vRowSum;
  }
}

template <typename T> void forward_repeat_nrows(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *vSrc0 = pOp->src[0];
  ggml_tensor *vDst = pOp;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  int64_t vNrows = ggml_nrows(vSrc0);
  int64_t vNrowsPerThread = (vNrows + cNth - 1) / cNth;
  int64_t vIrStart = cIth * vNrowsPerThread;
  int64_t vIrEnd = std::min(vIrStart + vNrowsPerThread, vNrows);

  if (vSrc0->ne[0] == 1) {
    for (int64_t vIr = vIrStart; vIr < vIrEnd; vIr++) {
      T *vSrcRow = (T *)((char *)vSrc0->data + vIr * vSrc0->nb[1]);
      T *vDstRow = (T *)((char *)vDst->data + vIr * vDst->nb[1]);

      T vSrcScalar = vSrcRow[0];

      int64_t vLength = vDst->ne[0];
      int64_t vIdx = 0;
      size_t vVl = 0;

      while (vLength > 0) {
        if constexpr (std::is_same_v<T, int32_t>) {
          vVl = __riscv_vsetvl_e32m4(vLength);
          vint32m4_t vVec = __riscv_vmv_v_x_i32m4(vSrcScalar, vVl);
          __riscv_vse32_v_i32m4(vDstRow + vIdx, vVec, vVl);
        } else if constexpr (std::is_same_v<T, int16_t>) {
          vVl = __riscv_vsetvl_e16m4(vLength);
          vint16m4_t vVec = __riscv_vmv_v_x_i16m4(vSrcScalar, vVl);
          __riscv_vse16_v_i16m4((vDstRow + vIdx), vVec, vVl);
        } else {
          GGML_ABORT("fatal error");
        }
        vIdx += vVl;
        vLength -= vVl;
      }
    }
  } else if (vSrc0->ne[0] == vDst->ne[0]) {
    for (int64_t vIr = vIrStart; vIr < vIrEnd; vIr++) {
      T *vSrcRow = (T *)((char *)vSrc0->data + vIr * vSrc0->nb[1]);
      T *vDstRow = (T *)((char *)vDst->data + vIr * vDst->nb[1]);

      int64_t vLength = vDst->ne[0];
      int64_t vIdx = 0;
      size_t vVl = 0;

      while (vLength > 0) {
        if constexpr (std::is_same_v<T, int32_t>) {
          vVl = __riscv_vsetvl_e32m4(vLength);
          vint32m4_t vVec = __riscv_vle32_v_i32m4(vSrcRow + vIdx, vVl);
          __riscv_vse32_v_i32m4(vDstRow + vIdx, vVec, vVl);
        } else if constexpr (std::is_same_v<T, int16_t>) {
          vVl = __riscv_vsetvl_e16m4(vLength);
          vint16m4_t vVec = __riscv_vle16_v_i16m4((vSrcRow + vIdx), vVl);
          __riscv_vse16_v_i16m4((vDstRow + vIdx), vVec, vVl);
        } else {
          GGML_ABORT("fatal error");
        }
        vIdx += vVl;
        vLength -= vVl;
      }
    }
  } else {
    GGML_ABORT("fatal error");
  }
}

template <typename T> void forward_repeat_dim1(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *vSrc0 = pOp->src[0];
  ggml_tensor *vDst = pOp;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNe0 = vDst->ne[0];
  const int64_t cNe1 = vDst->ne[1];
  const int64_t cNe2 = vDst->ne[2];
  const int64_t cNe3 = vDst->ne[3];

  const int64_t cTotalBatches = cNe2 * cNe3;
  const int64_t cBatchesPerThread = (cTotalBatches + cNth - 1) / cNth;
  const int64_t cBatchStart = cIth * cBatchesPerThread;
  const int64_t cBatchEnd = std::min(cBatchStart + cBatchesPerThread, cTotalBatches);

  for (int64_t vB = cBatchStart; vB < cBatchEnd; vB++) {
    const int64_t cI3 = vB / cNe2;
    const int64_t cI2 = vB % cNe2;

    T *vSrcBase = (T *)((char *)vSrc0->data + cI2 * vSrc0->nb[2] + cI3 * vSrc0->nb[3]);
    T *vDstBatch = (T *)((char *)vDst->data + cI2 * vDst->nb[2] + cI3 * vDst->nb[3]);

    for (int64_t vI1 = 0; vI1 < cNe1; vI1++) {
      T *vDstPtr = (T *)((char *)vDstBatch + vI1 * vDst->nb[1]);
      int64_t vLength = cNe0;
      int64_t vIdx = 0;

      while (vLength > 0) {
        if constexpr (std::is_same_v<T, int32_t>) {
          size_t vVl = __riscv_vsetvl_e32m4(vLength);
          vint32m4_t vVec = __riscv_vle32_v_i32m4(vSrcBase + vIdx, vVl);
          __riscv_vse32_v_i32m4(vDstPtr + vIdx, vVec, vVl);
          vIdx += vVl;
          vLength -= vVl;
        } else if constexpr (std::is_same_v<T, int16_t>) {
          size_t vVl = __riscv_vsetvl_e16m4(vLength);
          vint16m4_t vVec = __riscv_vle16_v_i16m4((vSrcBase + vIdx), vVl);
          __riscv_vse16_v_i16m4((vDstPtr + vIdx), vVec, vVl);
          vIdx += vVl;
          vLength -= vVl;
        } else {
          GGML_ABORT("fatal error");
        }
      }
    }
  }
}

template <typename T> void forward_get_rows(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  const ggml_tensor *src1 = pOp->src[1];
  ggml_tensor *dst = pOp;

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ggml_nelements(src1);

  assert(ne0 == cNc);
  assert(ne02 == ne11);
  assert(nb00 == sizeof(float));
  assert(ggml_nrows(pOp) == cNr);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  int vRowsNth = cNth;
  int vColsNth = 1;

  if (cNr == 1) {
    vRowsNth = 1;
    vColsNth = cNth;
  }

  // rows per thread
  const int cDr = (cNr + vRowsNth - 1) / vRowsNth;
  const int cDc = (cNc + vColsNth - 1) / vColsNth;

  int vRowsIth = cIth % vRowsNth;
  int vColsIth = cIth % vColsNth;

  // row range for this thread
  const int cIr0 = cDr * vRowsIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  const int cCr0 = cDc * vColsIth;
  const int cCr1 = MIN(cCr0 + cDc, cNc);

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI12 = vI / (ne11 * ne10);
    const int64_t cI11 = (vI - cI12 * ne11 * ne10) / ne10;
    const int64_t cI10 = (vI - cI12 * ne11 * ne10 - cI11 * ne10);
    const int64_t cI01 = *(int32_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

    GGML_ASSERT(cI01 >= 0 && cI01 < ne01);

    memcpy1d(((char *)dst->data + cI10 * nb1 + cI11 * nb2 + cI12 * nb3) + cCr0 * sizeof(T),
             ((char *)src0->data + cI01 * nb01 + cI11 * nb02 + cI12 * nb03) + cCr0 * sizeof(T),
             (cCr1 - cCr0) * sizeof(T));
  }
}

template <typename T> void forward_concat(ggml_compute_params *pParams, ggml_tensor *pOp) {
  const ggml_tensor *src0 = pOp->src[0];
  const ggml_tensor *src1 = pOp->src[1];
  ggml_tensor *dst = pOp;

  GGML_ASSERT(ggml_type_size(src0->type) == sizeof(float));

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cDim = ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(cDim == 0 && nb0 == sizeof(float) && nb1 == sizeof(float) * (ne00 + ne10));

  const int64_t cNr = ggml_nrows(dst);
  const int64_t cNc = ne0;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  int vRowsNth = cNth;
  int vColsNth = 1;

  if (cNr == 1) {
    vRowsNth = 1;
    vColsNth = cNth;
  }

  const int cDr = (cNr + vRowsNth - 1) / vRowsNth;
  const int cDc = (cNc + vColsNth - 1) / vColsNth;

  int vRowsIth = cIth % vRowsNth;
  int vColsIth = cIth % vColsNth;

  // row range for this thread
  const int cIr0 = cDr * vRowsIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  const int cCr0 = cDc * vColsIth;
  const int cCr1 = MIN(cCr0 + cDc, cNc);

  int64_t aO[4] = {0, 0, 0, 0};
  aO[cDim] = src0->ne[cDim];
  const float *vX;

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI3 = vI / (ne02 * ne01);
    const int64_t cI2 = (vI - cI3 * ne02 * ne01) / ne01;
    const int64_t cI1 = (vI - cI3 * ne02 * ne01 - cI2 * ne01);

    for (int vI0 = cCr0; vI0 < cCr1; vI0++) {
      if (vI0 < ne00 && cI1 < ne01 && cI2 < ne02 && cI3 < ne03) {
        vX = (const float *)((const char *)src0->data + (vI0)*nb00 + (cI1)*nb01 + (cI2)*nb02 + (cI3)*nb03);
      } else {
        vX = (const float *)((const char *)src1->data + (vI0 - aO[0]) * nb10 + (cI1 - aO[1]) * nb11 +
                             (cI2 - aO[2]) * nb12 + (cI3 - aO[3]) * nb13);
      }

      float *vY = (float *)((char *)dst->data + vI0 * nb0 + cI1 * nb1 + cI2 * nb2 + cI3 * nb3);

      *vY = *vX;
    }
  }
}

template void forward_binary<GGML_OP_ADD, float>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_SUB, float>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_MUL, float>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_DIV, float>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_ADD, _Float16>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_SUB, _Float16>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_MUL, _Float16>(ggml_compute_params *params, ggml_tensor *op);
template void forward_binary<GGML_OP_DIV, _Float16>(ggml_compute_params *params, ggml_tensor *op);
template void forward_sum_rows<float>(const ggml_compute_params *params, ggml_tensor *op);
template void forward_sum_rows<_Float16>(const ggml_compute_params *params, ggml_tensor *op);
template void forward_repeat_nrows<int32_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_repeat_nrows<int16_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_repeat_dim1<int32_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_repeat_dim1<int16_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_get_rows<int32_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_get_rows<int16_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_concat<int32_t>(ggml_compute_params *params, ggml_tensor *op);
template void forward_concat<int16_t>(ggml_compute_params *params, ggml_tensor *op);

} // namespace spacemit_kernels::rvv
