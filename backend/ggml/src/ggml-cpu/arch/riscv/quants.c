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

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

void quantize_row_q8_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__riscv_v)

  size_t vVl = QK8_0;

  for (int vI = 0; vI < cNb; vI++) {
    // load elements
    vfloat32m8_t vVX = __riscv_vle32_v_f32m8(pX + vI * QK8_0, vVl);

    vfloat32m8_t vVfabs = __riscv_vfabs_v_f32m8(vVX, vVl);
    vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0f, vVl);
    vfloat32m1_t vVmax = __riscv_vfredmax_vs_f32m8_f32m1(vVfabs, vTmp, vVl);
    float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVmax);

    const float cD = vAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    vfloat32m8_t vX0 = __riscv_vfmul_vf_f32m8(vVX, cId, vVl);

    // convert to integer
    vint16m4_t vVi = __riscv_vfncvt_x_f_w_i16m4(vX0, vVl);
    vint8m2_t vVs = __riscv_vncvt_x_x_w_i8m2(vVi, vVl);

    // store result
    __riscv_vse8_v_i8m2(vY[vI].qs, vVs, vVl);
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

#if defined(__riscv_v)

  size_t vVl = QK8_1;

  for (int vI = 0; vI < cNb; vI++) {
    // load elements
    vfloat32m8_t vVX = __riscv_vle32_v_f32m8(pX + vI * QK8_1, vVl);

    vfloat32m8_t vVfabs = __riscv_vfabs_v_f32m8(vVX, vVl);
    vfloat32m1_t vTmp = __riscv_vfmv_v_f_f32m1(0.0, vVl);
    vfloat32m1_t vVmax = __riscv_vfredmax_vs_f32m8_f32m1(vVfabs, vTmp, vVl);
    float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVmax);

    const float cD = vAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);

    vfloat32m8_t vX0 = __riscv_vfmul_vf_f32m8(vVX, cId, vVl);

    // convert to integer
    vint16m4_t vVi = __riscv_vfncvt_x_f_w_i16m4(vX0, vVl);
    vint8m2_t vVs = __riscv_vncvt_x_x_w_i8m2(vVi, vVl);

    // store result
    __riscv_vse8_v_i8m2(vY[vI].qs, vVs, vVl);

    // compute sum for y[i].s
    vint16m1_t vTmp2 = __riscv_vmv_v_x_i16m1(0, vVl);
    vint16m1_t vVwrs = __riscv_vwredsum_vs_i8m2_i16m1(vVs, vTmp2, vVl);

    // set y[i].s
    int vSum = __riscv_vmv_x_s_i16m1_i16(vVwrs);
    vY[vI].s = GGML_CPU_FP32_TO_FP16(vSum * cD);
  }

#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_1_ref(pX, vY, pK);
#endif
}

void quantize_row_q8_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t k) {
  assert(k % QK_K == 0);
  size_t vNb = k / QK_K;

#if defined __riscv_v
  block_q8_K *vYBlocks = (block_q8_K *)pY;
  const size_t cVlmaxF32m8 = __riscv_vsetvlmax_e32m8();

  for (size_t vI = 0; vI < vNb; vI++) {
    const float *vXBlock = pX + vI * QK_K;
    block_q8_K *vYBlock = &vYBlocks[vI];

    // 1. Calculate Min/Max
    vfloat32m8_t vMaxV = __riscv_vfmv_v_f_f32m8(-__builtin_inff(), cVlmaxF32m8);
    vfloat32m8_t vMinV = __riscv_vfmv_v_f_f32m8(__builtin_inff(), cVlmaxF32m8);

    size_t vRem = QK_K;
    size_t vOffset = 0;
    while (vRem > 0) {
      size_t vVl = __riscv_vsetvl_e32m8(vRem);
      vfloat32m8_t vVCurr = __riscv_vle32_v_f32m8(vXBlock + vOffset, vVl);
      vMaxV = __riscv_vfmax_vv_f32m8(vMaxV, vVCurr, vVl);
      vMinV = __riscv_vfmin_vv_f32m8(vMinV, vVCurr, vVl);
      vRem -= vVl;
      vOffset += vVl;
    }

    vfloat32m1_t vVInitMax = __riscv_vfmv_s_f_f32m1(-__builtin_inff(), 1);
    vfloat32m1_t vVInitMin = __riscv_vfmv_s_f_f32m1(__builtin_inff(), 1);

    vfloat32m1_t vVScalarMax = __riscv_vfredmax_vs_f32m8_f32m1(vMaxV, vVInitMax, cVlmaxF32m8);
    vfloat32m1_t vVScalarMin = __riscv_vfredmin_vs_f32m8_f32m1(vMinV, vVInitMin, cVlmaxF32m8);

    float vMaxVal = __riscv_vfmv_f_s_f32m1_f32(vVScalarMax);
    float vMinVal = __riscv_vfmv_f_s_f32m1_f32(vVScalarMin);

    float vAmax = fabsf(vMaxVal) > fabsf(vMinVal) ? fabsf(vMaxVal) : fabsf(vMinVal);

    if (vAmax == 0.0f) {
      vYBlock->d = 0.0f;
      memset(vYBlock->qs, 0, QK_K);
      memset(vYBlock->bsums, 0, sizeof(vYBlock->bsums));
      continue;
    }

    const float cIscale = -127.f / (fabsf(vMaxVal) > fabsf(vMinVal) ? vMaxVal : vMinVal);
    vYBlock->d = 1.0f / cIscale;

    // 2. Quantize and Calculate Sums
    vOffset = 0;
    vRem = QK_K;
    vint16m1_t vVZeroSum = __riscv_vmv_v_x_i16m1(0, 1);

    while (vRem > 0) {
      size_t vVl = __riscv_vsetvl_e32m8(vRem);
      vfloat32m8_t vVF = __riscv_vle32_v_f32m8(vXBlock + vOffset, vVl);

      vVF = __riscv_vfmul_vf_f32m8(vVF, cIscale, vVl);

      vint32m8_t vVI32 = __riscv_vfcvt_x_f_v_i32m8_rm(vVF, __RISCV_FRM_RNE, vVl);
      vint16m4_t vVI16 = __riscv_vnclip_wx_i16m4(vVI32, 0, __RISCV_VXRM_RNE, vVl);
      vint8m2_t vVQ = __riscv_vnclip_wx_i8m2(vVI16, 0, __RISCV_VXRM_RNE, vVl);

      __riscv_vse8_v_i8m2(vYBlock->qs + vOffset, vVQ, vVl);

      // first iteration clear

      int vSumIdx;
      vint8m1_t vChunkM1;
      vint16m1_t vVSum;
      vSumIdx = vOffset / 16;
      vChunkM1 = __riscv_vget_v_i8m2_i8m1(vVQ, 0);
      vVSum = __riscv_vwredsum_vs_i8m1_i16m1(vChunkM1, vVZeroSum, 16);
      vYBlock->bsums[vSumIdx] = (int16_t)__riscv_vmv_x_s_i16m1_i16(vVSum);

      // remaining iterations
      vint8m2_t vSlidQ = vVQ;
      for (size_t k = 16; k < vVl; k += 16) {
        vSlidQ = __riscv_vslidedown_vx_i8m2(vSlidQ, 16, vVl);

        vSumIdx = (vOffset + k) / 16;
        vChunkM1 = __riscv_vget_v_i8m2_i8m1(vSlidQ, 0);

        vVSum = __riscv_vwredsum_vs_i8m1_i16m1(vChunkM1, vVZeroSum, 16);
        vYBlock->bsums[vSumIdx] = (int16_t)__riscv_vmv_x_s_i16m1_i16(vVSum);
      }

      vRem -= vVl;
      vOffset += vVl;
    }
  }
#else
  GGML_UNUSED(vNb);
  // scalar
  quantize_row_q8_K_ref(pX, pY, k);
#endif
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined(__riscv_v)
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

  size_t vVl = cQk / 2;

  for (; vIb < cNb; ++vIb) {
    // load elements
    vuint8m1_t vTx = __riscv_vle8_v_u8m1(vX[vIb].qs, vVl);

    vint8m1_t vY0 = __riscv_vle8_v_i8m1(vY[vIb].qs, vVl);
    vint8m1_t vY1 = __riscv_vle8_v_i8m1(vY[vIb].qs + 16, vVl);

    // mask and store lower part of x, and then upper part
    vuint8m1_t vXA = __riscv_vand_vx_u8m1(vTx, 0x0F, vVl);
    vuint8m1_t vXL = __riscv_vsrl_vx_u8m1(vTx, 0x04, vVl);

    vint8m1_t vXAi = __riscv_vreinterpret_v_u8m1_i8m1(vXA);
    vint8m1_t vXLi = __riscv_vreinterpret_v_u8m1_i8m1(vXL);

    // subtract offset
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(vXAi, 8, vVl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(vXLi, 8, vVl);

    vint16m2_t vVecMul1 = __riscv_vwmul_vv_i16m2(v0, vY0, vVl);
    vint16m2_t vVecMul2 = __riscv_vwmacc_vv_i16m2(vVecMul1, v1, vY1, vVl);

    vint32m1_t vVecZero = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVs2 = __riscv_vwredsum_vs_i16m2_i32m1(vVecMul2, vVecZero, vVl);

    int vSumi = __riscv_vmv_x_s_i32m1_i32(vVs2);

    vSumf += vSumi * GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d);
  }

  *pS = vSumf;
#else
  ggml_vec_dot_q4_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q4_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined(__riscv_v)
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

  size_t vVl = cQk / 2;

  for (; vIb < cNb; ++vIb) {
    // load elements
    vuint8m1_t vTx = __riscv_vle8_v_u8m1(vX[vIb].qs, vVl);

    vint8m1_t vY0 = __riscv_vle8_v_i8m1(vY[vIb].qs, vVl);
    vint8m1_t vY1 = __riscv_vle8_v_i8m1(vY[vIb].qs + 16, vVl);

    // mask and store lower part of x, and then upper part
    vuint8m1_t vXA = __riscv_vand_vx_u8m1(vTx, 0x0F, vVl);
    vuint8m1_t vXL = __riscv_vsrl_vx_u8m1(vTx, 0x04, vVl);

    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(vXA);
    vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(vXL);

    vint16m2_t vVecMul1 = __riscv_vwmul_vv_i16m2(v0, vY0, vVl);
    vint16m2_t vVecMul2 = __riscv_vwmacc_vv_i16m2(vVecMul1, v1, vY1, vVl);

    vint32m1_t vVecZero = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVs2 = __riscv_vwredsum_vs_i16m2_i32m1(vVecMul2, vVecZero, vVl);

    int vSumi = __riscv_vmv_x_s_i32m1_i32(vVs2);

    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
#else
  ggml_vec_dot_q4_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined(__riscv_v)
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

  size_t vVl;
  size_t vVlenb = __riscv_vlenb();

  for (; vIb < cNb; ++vIb) {
    vVl = cQk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(vX[vIb].qs, vVl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vVl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vVl));
    vint8m2_t v0c;
    if (vVlenb == 16) {
      v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
    } else {
      v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
      v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
    }

    vVl = cQk;
    vbool4_t vQh = __riscv_vlm_v_b4(vX[vIb].qh, vVl);
    vQh = __riscv_vmnand_mm_b4(vQh, vQh, vVl);
    vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(vQh, v0c, v0c, 0x10, vVl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(vY[vIb].qs, vVl);
    vint16m4_t vMul = __riscv_vwmul_vv_i16m4(v0f, v1, vVl);
    vint32m1_t vZero = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vSum = __riscv_vwredsum_vs_i16m4_i32m1(vMul, vZero, vVl);
    int32_t vSumi = __riscv_vmv_x_s_i32m1_i32(vSum);

    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi;
  }

  *pS = vSumf;
#else
  ggml_vec_dot_q5_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined(__riscv_v)
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

  size_t vVl;
  size_t vVlenb = __riscv_vlenb();

  for (; vIb < cNb; ++vIb) {
    vVl = cQk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(vX[vIb].qs, vVl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vVl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vVl));
    vint8m2_t v0c;
    if (vVlenb == 16) {
      v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
    } else {
      v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
      v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
    }

    vVl = cQk;
    vbool4_t vQh = __riscv_vlm_v_b4(vX[vIb].qh, vVl);
    vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(vQh, v0c, v0c, 0x10, vVl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(vY[vIb].qs, vVl);
    vint16m4_t vMul = __riscv_vwmul_vv_i16m4(v0f, v1, vVl);
    vint32m1_t vZero = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vSum = __riscv_vwredsum_vs_i16m4_i32m1(vMul, vZero, vVl);
    int32_t vSumi = __riscv_vmv_x_s_i32m1_i32(vSum);

    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
#else
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

#if defined(__riscv_v)
  size_t vVl = cQk;

  for (; vIb < cNb; ++vIb) {
    // load elements
    vint8m2_t vBx0 = __riscv_vle8_v_i8m2(vX[vIb].qs, vVl);
    vint8m2_t vBy0 = __riscv_vle8_v_i8m2(vY[vIb].qs, vVl);

    vint16m4_t vVwMul = __riscv_vwmul_vv_i16m4(vBx0, vBy0, vVl);

    vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVSum = __riscv_vwredsum_vs_i16m4_i32m1(vVwMul, vVZero, vVl);

    int vSumi = __riscv_vmv_x_s_i32m1_i32(vVSum);

    vSumf += vSumi * (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));
  }

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

#if defined(__riscv_v)
static NOINLINE void fGgmlVecDotQ10Q80Vl256(const int pN, float *GGML_RESTRICT pS, const void *GGML_RESTRICT pVx,
                                            const void *GGML_RESTRICT pVy) {
  const int cQk = QK1_0;
  const int cNb = pN / cQk;
  assert(pN % cQk == 0);

  const block_q1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  // LMUL = 1, VLMAX = 32
  const size_t cVl32 = __riscv_vsetvl_e8m1(32);
  assert(cVl32 == 32);

  const vint16m1_t cZero = __riscv_vmv_v_x_i16m1(0, 1);

  float vSumf = 0;

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);

    float vAcc = 0;

    for (int vK = 0; vK < 4; ++vK) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vIb * 4 + vK];
      const vbool8_t cIsNotZero = __riscv_vlm_v_b8(vX[vIb].qs + 4 * vK, cVl32);

      const vint8m1_t cQy = __riscv_vle8_v_i8m1(vYb->qs, cVl32);
      const vint8m1_t cNegQy = __riscv_vneg_v_i8m1(cQy, cVl32);
      const vint8m1_t cSy = __riscv_vmerge_vvm_i8m1(cNegQy, cQy, cIsNotZero, cVl32);

      const vint16m1_t cRed = __riscv_vwredsum_vs_i8m1_i16m1(cSy, cZero, cVl32);
      vAcc += GGML_CPU_FP16_TO_FP32(vYb->d) * (float)__riscv_vmv_x_s_i16m1_i16(cRed);
    }

    vSumf += cD0 * vAcc;
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotQ10Q80Vl128(const int pN, float *GGML_RESTRICT pS, const void *GGML_RESTRICT pVx,
                                            const void *GGML_RESTRICT pVy) {
  const int cQk = QK1_0;
  const int cNb = pN / cQk;
  assert(pN % cQk == 0);

  const block_q1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  // LMUL = 2, VLMAX = 32
  const size_t cVl32 = __riscv_vsetvl_e8m2(32);
  assert(cVl32 == 32);

  const vint16m1_t cZero = __riscv_vmv_v_x_i16m1(0, 1);

  float vSumf = 0;

  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);

    float vAcc = 0;

    for (int vK = 0; vK < 4; ++vK) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vIb * 4 + vK];
      const vbool4_t cIsNotZero = __riscv_vlm_v_b4(vX[vIb].qs + 4 * vK, cVl32);

      const vint8m2_t cQy = __riscv_vle8_v_i8m2(vYb->qs, cVl32);
      const vint8m2_t cNegQy = __riscv_vneg_v_i8m2(cQy, cVl32);
      const vint8m2_t cSy = __riscv_vmerge_vvm_i8m2(cNegQy, cQy, cIsNotZero, cVl32);

      const vint16m1_t cRed = __riscv_vwredsum_vs_i8m2_i16m1(cSy, cZero, cVl32);
      vAcc += GGML_CPU_FP16_TO_FP32(vYb->d) * (float)__riscv_vmv_x_s_i16m1_i16(cRed);
    }

    vSumf += cD0 * vAcc;
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_q1_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined(__riscv_v)
  assert(pNrc == 1);

  const size_t cVlenBits = __riscv_vlenb() * 8;

  if (cVlenBits >= 256) {
    fGgmlVecDotQ10Q80Vl256(pN, pS, pVx, pVy);
  } else if (cVlenBits >= 128) {
    fGgmlVecDotQ10Q80Vl128(pN, pS, pVx, pVy);
  } else {
    ggml_vec_dot_q1_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
  }
#else
  ggml_vec_dot_q1_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_xtheadvector
void ggml_vec_dot_q2_K_q8_K_xtheadvector(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                         size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
  assert(nrc == 1);
  UNUSED(nrc);
  UNUSED(bx);
  UNUSED(by);
  UNUSED(bs);

  const block_q2_K *GGML_RESTRICT x = vx;
  const block_q8_K *GGML_RESTRICT y = vy;

  const int nb = n / QK_K;

  float sumf = 0;
  uint8_t atmp[16];

  for (int i = 0; i < nb; ++i) {
    const uint8_t *q2 = x[i].qs;
    const int8_t *q8 = y[i].qs;
    const uint8_t *sc = x[i].scales;
    const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
    uint8_t *patmp = atmp;
    int vsums;
    int tmp;
    __asm__ __volatile__("th.vsetvli zero, %[vl16], e8, m1\n\t"
                         "th.vmv.v.x v8, zero\n\t"
                         "th.vlb.v v1, (%[sc])\n\t"
                         "th.vand.vi v0, v1, 0xF\n\t"
                         "th.vsrl.vi v1, v1, 4\n\t"
                         "th.vsb.v v0, (%[scale])\n\t"
                         "th.vwaddu.vx v16, v1, zero\n\t"
                         "th.vsetvli zero, %[vl16], e16, m2\n\t"
                         "th.vlh.v v2, (%[bsums])\n\t"
                         "th.vwmul.vv v4, v16, v2\n\t"
                         "th.vsetvli zero, %[vl16], e32, m4\n\t"
                         "th.vredsum.vs v8, v4, v8\n\t"
                         "th.vmv.x.s %[vsums], v8"
                         : [tmp] "=&r"(tmp), [vsums] "=&r"(vsums)
                         : [sc] "r"(sc), [scale] "r"(atmp), [bsums] "r"(y[i].bsums), [vl16] "r"(16)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");
    sumf += dmin * vsums;
    int isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      __asm__ __volatile__("th.vsetvli zero, %[vl32], e8, m2\n\t"
                           "th.vlb.v v0, (%[q2])\n\t"
                           "th.vsrl.vi v2, v0, 2\n\t"
                           "th.vsrl.vi v4, v0, 4\n\t"
                           "th.vsrl.vi v6, v0, 6\n\t"
                           "th.vand.vi v0, v0, 0x3\n\t"
                           "th.vand.vi v2, v2, 0x3\n\t"
                           "th.vand.vi v4, v4, 0x3\n\t"
                           "th.vsetvli zero, %[vl128], e8, m8\n\t"
                           "th.vlb.v v8, (%[q8])\n\t"
                           "th.vsetvli zero, %[vl64], e8, m4\n\t"
                           "th.vwmul.vv v16, v0, v8\n\t"
                           "th.vwmul.vv v24, v4, v12\n\t"
                           "th.vsetvli zero, %[vl16], e16, m2\n\t"
                           "th.vmv.v.x v0, zero\n\t"
                           "th.vwredsum.vs v10, v16, v0\n\t"
                           "th.vwredsum.vs v9, v18, v0\n\t"
                           "th.vwredsum.vs v8, v20, v0\n\t"
                           "th.vwredsum.vs v7, v22, v0\n\t"
                           "th.vwredsum.vs v11, v24, v0\n\t"
                           "th.vwredsum.vs v12, v26, v0\n\t"
                           "th.vwredsum.vs v13, v28, v0\n\t"
                           "th.vwredsum.vs v14, v30, v0\n\t"
                           "li %[tmp], 4\n\t"
                           "th.vsetvli zero, %[tmp], e32, m1\n\t"
                           "th.vslideup.vi v10, v9, 1\n\t"
                           "th.vslideup.vi v8, v7, 1\n\t"
                           "th.vslideup.vi v11, v12, 1\n\t"
                           "th.vslideup.vi v13, v14, 1\n\t"
                           "th.vslideup.vi v10, v8, 2\n\t"
                           "th.vslideup.vi v11, v13, 2\n\t"
                           "li %[tmp], 8\n\t"
                           "th.vsetvli zero, %[tmp], e32, m2\n\t"
                           "th.vlbu.v v12, (%[scale])\n\t"
                           "th.vmul.vv v10, v10, v12\n\t"
                           "th.vredsum.vs v0, v10, v0\n\t"
                           "th.vmv.x.s %[tmp], v0\n\t"
                           "add %[isum], %[isum], %[tmp]"
                           : [tmp] "=&r"(tmp), [isum] "+&r"(isum)
                           : [q2] "r"(q2), [scale] "r"(patmp), [q8] "r"(q8), [vl16] "r"(16), [vl32] "r"(32),
                             [vl64] "r"(64), [vl128] "r"(128)
                           : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                             "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                             "v26", "v27", "v28", "v29", "v30", "v31");
      q2 += 32;
      q8 += 128;
      patmp += 8;
    }

    sumf += dall * isum;
  }

  *s = sumf;
}
#endif

#if defined __riscv_v
void fGgmlVecDotQ2KQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  uint8_t aAtmp[16];

  uint8_t aTemp01[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *q2 = vX[vI].qs;
    const int8_t *q8 = vY[vI].qs;
    const uint8_t *sc = vX[vI].scales;
    const float cDall = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);
    uint8_t *vPatmp = aAtmp;
    int vsums;
    int tmp, t1, t2, t3, t4, t5, t6, t7;
    __asm__ __volatile__("vsetivli zero, 16, e8, m1\n\t"
                         "vmv.v.x v8, zero\n\t"
                         "lb zero, 15(%[sc])\n\t"
                         "vle8.v v1, (%[sc])\n\t"
                         "vle8.v v2, (%[bsums])\n\t"
                         "addi %[tmp], %[bsums], 16\n\t"
                         "vand.vi v0, v1, 0xF\n\t"
                         "vsrl.vi v1, v1, 4\n\t"
                         "vle8.v v3, (%[tmp])\n\t"
                         "vse8.v v0, (%[scale])\n\t"
                         "vsetivli zero, 16, e16, m2\n\t"
                         "vzext.vf2 v0, v1\n\t"
                         "vwmul.vv v4, v0, v2\n\t"
                         "vsetivli zero, 16, e32, m4\n\t"
                         "vredsum.vs v8, v4, v8\n\t"
                         "vmv.x.s %[vsums], v8"
                         : [tmp] "=&r"(tmp), [vsums] "=&r"(vsums)
                         : [sc] "r"(sc), [scale] "r"(aAtmp), [bsums] "r"(vY[vI].bsums)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");
    vSumf += cDmin * vsums;
    int isum = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      __asm__ __volatile__("lb zero, 31(%[q2])\n\t"
                           "addi %[tmp], %[q2], 16\n\t"
                           "addi %[t1], %[q8], 16\n\t"
                           "vsetivli zero, 16, e8, m1\n\t"
                           "vle8.v v0, (%[q2])\n\t"
                           "vle8.v v1, (%[tmp])\n\t"
                           "vsrl.vi v2, v0, 2\n\t"
                           "vsrl.vi v3, v1, 2\n\t"
                           "vsrl.vi v4, v0, 4\n\t"
                           "addi %[tmp], %[q8], 32\n\t"
                           "vle8.v v8, (%[q8])\n\t"
                           "vle8.v v9, (%[t1])\n\t"
                           "addi %[t1], %[t1], 32\n\t"
                           "vsrl.vi v5, v1, 4\n\t"
                           "vsrl.vi v6, v0, 6\n\t"
                           "vsrl.vi v7, v1, 6\n\t"
                           "vle8.v v10, (%[tmp])\n\t"
                           "vle8.v v11, (%[t1])\n\t"
                           "addi %[tmp], %[tmp], 32\n\t"
                           "addi %[t1], %[t1], 32\n\t"
                           "vand.vi v0, v0, 0x3\n\t"
                           "vand.vi v1, v1, 0x3\n\t"
                           "vand.vi v2, v2, 0x3\n\t"
                           "vle8.v v12, (%[tmp])\n\t"
                           "vle8.v v13, (%[t1])\n\t"
                           "addi %[tmp], %[tmp], 32\n\t"
                           "addi %[t1], %[t1], 32\n\t"
                           "vand.vi v3, v3, 0x3\n\t"
                           "vand.vi v4, v4, 0x3\n\t"
                           "vand.vi v5, v5, 0x3\n\t"
                           "vle8.v v14, (%[tmp])\n\t"
                           "vle8.v v15, (%[t1])\n\t"
                           "vwmul.vv v16, v0, v8\n\t"
                           "vwmul.vv v18, v1, v9\n\t"
                           "vwmul.vv v20, v2, v10\n\t"
                           "vwmul.vv v22, v3, v11\n\t"
                           "vwmul.vv v24, v4, v12\n\t"
                           "vwmul.vv v26, v5, v13\n\t"
                           "vwmul.vv v28, v6, v14\n\t"
                           "vwmul.vv v30, v7, v15\n\t"
                           "vsetivli zero, 8, e16, m1\n\t"
                           "vmv.v.x v0, zero\n\t"
                           "lbu %[tmp], 0(%[scale])\n\t"
                           "vwredsum.vs v8, v16, v0\n\t"
                           "vwredsum.vs v9, v18, v0\n\t"
                           "lbu %[t1], 1(%[scale])\n\t"
                           "vwredsum.vs v10, v20, v0\n\t"
                           "vwredsum.vs v11, v22, v0\n\t"
                           "lbu %[t2], 2(%[scale])\n\t"
                           "vwredsum.vs v12, v24, v0\n\t"
                           "vwredsum.vs v13, v26, v0\n\t"
                           "lbu %[t3], 3(%[scale])\n\t"
                           "vwredsum.vs v14, v28, v0\n\t"
                           "vwredsum.vs v15, v30, v0\n\t"
                           "lbu %[t4], 4(%[scale])\n\t"
                           "vwredsum.vs v8, v17, v8\n\t"
                           "vwredsum.vs v9, v19, v9\n\t"
                           "lbu %[t5], 5(%[scale])\n\t"
                           "vwredsum.vs v10, v21, v10\n\t"
                           "vwredsum.vs v11, v23, v11\n\t"
                           "lbu %[t6], 6(%[scale])\n\t"
                           "vwredsum.vs v12, v25, v12\n\t"
                           "vwredsum.vs v13, v27, v13\n\t"
                           "lbu %[t7], 7(%[scale])\n\t"
                           "vwredsum.vs v14, v29, v14\n\t"
                           "vwredsum.vs v15, v31, v15\n\t"
                           "vsetivli zero, 4, e32, m1\n\t"
                           "vmul.vx v0, v8, %[tmp]\n\t"
                           "vmul.vx v1, v9, %[t1]\n\t"
                           "vmacc.vx v0, %[t2], v10\n\t"
                           "vmacc.vx v1, %[t3], v11\n\t"
                           "vmacc.vx v0, %[t4], v12\n\t"
                           "vmacc.vx v1, %[t5], v13\n\t"
                           "vmacc.vx v0, %[t6], v14\n\t"
                           "vmacc.vx v1, %[t7], v15\n\t"
                           "vmv.x.s %[tmp], v0\n\t"
                           "vmv.x.s %[t1], v1\n\t"
                           "add %[isum], %[isum], %[tmp]\n\t"
                           "add %[isum], %[isum], %[t1]"
                           : [tmp] "=&r"(tmp), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3), [t4] "=&r"(t4),
                             [t5] "=&r"(t5), [t6] "=&r"(t6), [t7] "=&r"(t7), [isum] "+&r"(isum)
                           : [q2] "r"(q2), [scale] "r"(vPatmp), [q8] "r"(q8)
                           : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                             "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                             "v26", "v27", "v28", "v29", "v30", "v31");
      q2 += 32;
      q8 += 128;
      vPatmp += 8;
    }

    vSumf += cDall * isum;
  }

  *pS = vSumf;
}

void fGgmlVecDotQ2KQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  uint8_t aAtmp[16];

  uint8_t aTemp01[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *vQ2 = vX[vI].qs;
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vSc = vX[vI].scales;

    const float cDall = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    size_t vVl = 16;

    vuint8m1_t vScales = __riscv_vle8_v_u8m1(vSc, vVl);
    vuint8m1_t vAux = __riscv_vand_vx_u8m1(vScales, 0x0F, vVl);

    vint16m1_t vQ8sums = __riscv_vle16_v_i16m1(vY[vI].bsums, vVl);

    vuint8mf2_t vScales2 = __riscv_vle8_v_u8mf2(vSc, vVl);
    vuint8mf2_t vMins8 = __riscv_vsrl_vx_u8mf2(vScales2, 0x4, vVl);
    vint16m1_t vMins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vMins8, vVl));
    vint32m2_t vProd = __riscv_vwmul_vv_i32m2(vQ8sums, vMins, vVl);
    vint32m1_t vVsums = __riscv_vredsum_vs_i32m2_i32m1(vProd, __riscv_vmv_v_x_i32m1(0, 1), vVl);

    vSumf += cDmin * __riscv_vmv_x_s_i32m1_i32(vVsums);

    vVl = 32;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8m1_t vVB = __riscv_vle8_v_u8m1(aTemp01, vVl);

    uint8_t vIs = 0;
    int vIsum = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load Q2
      vuint8m1_t vQ2X = __riscv_vle8_v_u8m1(vQ2, vVl);

      vuint8m1_t vQ20 = __riscv_vand_vx_u8m1(vQ2X, 0x03, vVl);
      vuint8m1_t vQ21 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ2X, 0x2, vVl), 0x03, vVl);
      vuint8m1_t vQ22 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ2X, 0x4, vVl), 0x03, vVl);
      vuint8m1_t vQ23 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ2X, 0x6, vVl), 0x03, vVl);

      // duplicate scale elements for product
      vuint8m1_t vSc0 = __riscv_vrgather_vv_u8m1(vAux, __riscv_vadd_vx_u8m1(vVB, 0 + vIs, vVl), vVl);
      vuint8m1_t vSc1 = __riscv_vrgather_vv_u8m1(vAux, __riscv_vadd_vx_u8m1(vVB, 2 + vIs, vVl), vVl);
      vuint8m1_t vSc2 = __riscv_vrgather_vv_u8m1(vAux, __riscv_vadd_vx_u8m1(vVB, 4 + vIs, vVl), vVl);
      vuint8m1_t vSc3 = __riscv_vrgather_vv_u8m1(vAux, __riscv_vadd_vx_u8m1(vVB, 6 + vIs, vVl), vVl);

      vint16m2_t vP0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(vQ20, vSc0, vVl));
      vint16m2_t vP1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(vQ21, vSc1, vVl));
      vint16m2_t vP2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(vQ22, vSc2, vVl));
      vint16m2_t vP3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(vQ23, vSc3, vVl));

      // load Q8
      vint8m1_t vQ80 = __riscv_vle8_v_i8m1(vQ8, vVl);
      vint8m1_t vQ81 = __riscv_vle8_v_i8m1(vQ8 + 32, vVl);
      vint8m1_t vQ82 = __riscv_vle8_v_i8m1(vQ8 + 64, vVl);
      vint8m1_t vQ83 = __riscv_vle8_v_i8m1(vQ8 + 96, vVl);

      vint32m4_t vS0 = __riscv_vwmul_vv_i32m4(vP0, __riscv_vwcvt_x_x_v_i16m2(vQ80, vVl), vVl);
      vint32m4_t vS1 = __riscv_vwmul_vv_i32m4(vP1, __riscv_vwcvt_x_x_v_i16m2(vQ81, vVl), vVl);
      vint32m4_t vS2 = __riscv_vwmul_vv_i32m4(vP2, __riscv_vwcvt_x_x_v_i16m2(vQ82, vVl), vVl);
      vint32m4_t vS3 = __riscv_vwmul_vv_i32m4(vP3, __riscv_vwcvt_x_x_v_i16m2(vQ83, vVl), vVl);

      vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(vS0, vS1, vVl), vVzero, vVl);
      vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(vS2, vS3, vVl), vIsum0, vVl);

      vIsum += __riscv_vmv_x_s_i32m1_i32(vIsum1);

      vQ2 += 32;
      vQ8 += 128;
      vIs = 8;
    }

    vSumf += cDall * vIsum;
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_q2_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_xtheadvector
  ggml_vec_dot_q2_K_q8_K_xtheadvector(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#elif defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotQ2KQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    fGgmlVecDotQ2KQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_q2_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_xtheadvector
void ggml_vec_dot_q3_K_q8_K_xtheadvector(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                         size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);
  UNUSED(nrc);
  UNUSED(bx);
  UNUSED(by);
  UNUSED(bs);

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const block_q3_K *GGML_RESTRICT x = vx;
  const block_q8_K *GGML_RESTRICT y = vy;

  const int nb = n / QK_K;

  uint32_t utmp[4];
  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const uint8_t *restrict q3 = x[i].qs;
    const uint8_t *restrict qh = x[i].hmask;
    const int8_t *restrict q8 = y[i].qs;

    int8_t *scale = (int8_t *)utmp;
    int tmp;
    __asm__ __volatile__("li %[tmp], 12\n\t"
                         "th.vsetvli zero, %[tmp], e8, m1\n\t"
                         "th.vlb.v v0, (%[s6b])\n\t"
                         "th.vmv.v.v v2, v0\n\t"
                         "li %[tmp], 2\n\t"
                         "th.vsetvli zero, %[tmp], e64, m1\n\t"
                         "th.vmv.v.x v9, %[sh]\n\t"
                         "th.vslidedown.vi v1, v0, 1\n\t"
                         "th.vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
                         "th.vslideup.vi v0, v2, 1\n\t"     // {aux[0], aux[1], aux[0], aux[1]}
                         "li %[tmp], 4\n\t"
                         "th.vsetvli zero, %[tmp], e32, m1\n\t"
                         "th.vid.v v9\n\t"
                         "th.vmv.x.s %[tmp], v1\n\t"
                         "th.vsll.vi v9, v9, 1\n\t"  // {0, 2, 4, 6}
                         "th.vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
                         "th.vsrl.vv v4, v1, v9\n\t"
                         "th.vsrl.vv v2, v0, v8\n\t"
                         "th.vand.vx v5, v4, %[kmask1]\n\t"
                         "th.vand.vx v3, v2, %[kmask2]\n\t"
                         "th.vsll.vi v6, v5, 4\n\t"
                         "th.vor.vv v7, v6, v3\n\t"
                         "li %[tmp], 16\n\t"
                         "th.vsetvli zero, %[tmp], e8, m1\n\t"
                         "th.vsub.vx v0, v7, %[c]\n\t"
                         "th.vsb.v v0, (%[scale])"
                         : [tmp] "=&r"(tmp)
                         : [sh] "r"(0x0000000400000004), [s6b] "r"(x[i].scales), [c] "r"(32), [scale] "r"(scale),
                           [kmask1] "r"(kmask1), [kmask2] "r"(kmask2)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");

    uint8_t m = 1;
    int isum = 0;
    for (int j = 0; j < QK_K; j += 128) {
      __asm__ __volatile__(
          // fixme: use v0p7 mask layout directly
          "th.vsetvli zero, %[vl32], e8, m2\n\t"
          "th.vlb.v v8, (%[q3])\n\t"
          "th.vsrl.vi v10, v8, 2\n\t"
          "th.vsrl.vi v12, v8, 4\n\t"
          "th.vsrl.vi v14, v8, 6\n\t"
          "th.vand.vi v8, v8, 3\n\t"
          "th.vand.vi v10, v10, 3\n\t"
          "th.vand.vi v12, v12, 3\n\t"
          "th.vlb.v v2, (%[qh])\n\t"
          "th.vand.vx v4, v2, %[m]\n\t"
          "slli %[m], %[m], 1\n\t"
          "th.vmseq.vx v0, v4, zero\n\t"
          "th.vadd.vi v8, v8, -4, v0.t\n\t"
          "th.vand.vx v4, v2, %[m]\n\t"
          "slli %[m], %[m], 1\n\t"
          "th.vmseq.vx v0, v4, zero\n\t"
          "th.vadd.vi v10, v10, -4, v0.t\n\t"
          "th.vand.vx v4, v2, %[m]\n\t"
          "slli %[m], %[m], 1\n\t"
          "th.vmseq.vx v0, v4, zero\n\t"
          "th.vadd.vi v12, v12, -4, v0.t\n\t"
          "th.vand.vx v4, v2, %[m]\n\t"
          "slli %[m], %[m], 1\n\t"
          "th.vmseq.vx v0, v4, zero\n\t"
          "th.vadd.vi v14, v14, -4, v0.t\n\t"
          "th.vsetvli zero, %[vl128], e8, m8\n\t"
          "th.vlb.v v0, (%[q8])\n\t"
          "th.vsetvli zero, %[vl64], e8, m4\n\t"
          "th.vwmul.vv v16, v0, v8\n\t"
          "th.vwmul.vv v24, v4, v12\n\t"
          "li %[tmp], 16\n\t"
          "th.vsetvli zero, %[tmp], e16, m2\n\t"
          "th.vmv.v.x v0, zero\n\t"
          "th.vwredsum.vs v10, v16, v0\n\t"
          "th.vwredsum.vs v9, v18, v0\n\t"
          "th.vwredsum.vs v8, v20, v0\n\t"
          "th.vwredsum.vs v7, v22, v0\n\t"
          "th.vwredsum.vs v11, v24, v0\n\t"
          "th.vwredsum.vs v12, v26, v0\n\t"
          "th.vwredsum.vs v13, v28, v0\n\t"
          "th.vwredsum.vs v14, v30, v0\n\t"
          "li %[tmp], 4\n\t"
          "th.vsetvli zero, %[tmp], e32, m1\n\t"
          "th.vslideup.vi v10, v9, 1\n\t"
          "th.vslideup.vi v8, v7, 1\n\t"
          "th.vslideup.vi v11, v12, 1\n\t"
          "th.vslideup.vi v13, v14, 1\n\t"
          "th.vslideup.vi v10, v8, 2\n\t"
          "th.vslideup.vi v11, v13, 2\n\t"
          "li %[tmp], 8\n\t"
          "th.vsetvli zero, %[tmp], e32, m2\n\t"
          "th.vlb.v v12, (%[scale])\n\t"
          "th.vmul.vv v10, v10, v12\n\t"
          "th.vredsum.vs v0, v10, v0\n\t"
          "th.vmv.x.s %[tmp], v0\n\t"
          "add %[isum], %[isum], %[tmp]"
          : [tmp] "=&r"(tmp), [m] "+&r"(m), [isum] "+&r"(isum)
          :
          [vl128] "r"(128), [vl64] "r"(64), [vl32] "r"(32), [q3] "r"(q3), [qh] "r"(qh), [scale] "r"(scale), [q8] "r"(q8)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
            "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
            "v30", "v31");
      q3 += 32;
      q8 += 128;
      scale += 8;
    }

    const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
    sumf += d * isum;
  }

  *s = sumf;
}
#endif

#if defined __riscv_v
void fGgmlVecDotQ3KQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  const block_q3_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  uint32_t vUtmp[4];
  float vSumf = 0;
  uint32_t aAux[3];

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *restrict q3 = vX[vI].qs;
    const uint8_t *restrict qh = vX[vI].hmask;
    const int8_t *restrict q8 = vY[vI].qs;

    int8_t *scale = (int8_t *)vUtmp;
    int tmp, t1, t2, t3, t4, t5, t6, t7;
    __asm__ __volatile__("vsetivli zero, 12, e8, m1\n\t"
                         "vle8.v v0, (%[s6b])\n\t"
                         "vmv1r.v v2, v0\n\t"
                         "vsetivli zero, 2, e64, m1\n\t"
                         "vmv.v.x v9, %[sh]\n\t"
                         "vslidedown.vi v1, v0, 1\n\t"
                         "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
                         "vslideup.vi v0, v2, 1\n\t"     // {aux[0], aux[1], aux[0], aux[1]}
                         "vsetivli zero, 4, e32, m1\n\t"
                         "vid.v v9\n\t"
                         "vmv.x.s %[tmp], v1\n\t"
                         "vsll.vi v9, v9, 1\n\t"  // {0, 2, 4, 6}
                         "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
                         "vsrl.vv v4, v1, v9\n\t"
                         "vsrl.vv v2, v0, v8\n\t"
                         "vand.vx v5, v4, %[kmask1]\n\t"
                         "vand.vx v3, v2, %[kmask2]\n\t"
                         "vsll.vi v6, v5, 4\n\t"
                         "vor.vv v7, v6, v3\n\t"
                         "vsetivli zero, 16, e8, m1\n\t"
                         "vsub.vx v0, v7, %[c]\n\t"
                         "vse8.v v0, (%[scale])"
                         : [tmp] "=&r"(tmp)
                         : [sh] "r"(0x0000000400000004), [s6b] "r"(vX[vI].scales), [c] "r"(32), [scale] "r"(scale),
                           [kmask1] "r"(kmask1), [kmask2] "r"(kmask2)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");

    uint8_t m = 1;
    int isum = 0;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      __asm__ __volatile__("lb zero, 31(%[q3])\n\t"
                           "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                           "vle8.v v8, (%[q3])\n\t"
                           "vsrl.vi v10, v8, 2\n\t"
                           "vsrl.vi v12, v8, 4\n\t"
                           "vsrl.vi v14, v8, 6\n\t"
                           "lb zero, 64(%[q8])\n\t"
                           "vand.vi v8, v8, 3\n\t"
                           "vand.vi v10, v10, 3\n\t"
                           "vand.vi v12, v12, 3\n\t"
                           "vle8.v v2, (%[qh])\n\t"
                           "lb zero, 127(%[q8])\n\t"
                           "vand.vx v4, v2, %[m]\n\t"
                           "slli %[m], %[m], 1\n\t"
                           "vmseq.vx v0, v4, zero\n\t"
                           "vadd.vi v8, v8, -4, v0.t\n\t"
                           "lb zero, 0(%[q8])\n\t"
                           "vand.vx v4, v2, %[m]\n\t"
                           "slli %[m], %[m], 1\n\t"
                           "vmseq.vx v0, v4, zero\n\t"
                           "vadd.vi v10, v10, -4, v0.t\n\t"
                           "vand.vx v4, v2, %[m]\n\t"
                           "slli %[m], %[m], 1\n\t"
                           "vmseq.vx v0, v4, zero\n\t"
                           "vadd.vi v12, v12, -4, v0.t\n\t"
                           "vand.vx v4, v2, %[m]\n\t"
                           "slli %[m], %[m], 1\n\t"
                           "vmseq.vx v0, v4, zero\n\t"
                           "vadd.vi v14, v14, -4, v0.t\n\t"
                           "vsetvli zero, %[vl128], e8, m8\n\t"
                           "vle8.v v0, (%[q8])\n\t"
                           "lb %[tmp], 0(%[scale])\n\t"
                           "lb %[t1], 1(%[scale])\n\t"
                           "lb %[t2], 2(%[scale])\n\t"
                           "lb %[t3], 3(%[scale])\n\t"
                           "vsetvli zero, %[vl64], e8, m4\n\t"
                           "vwmul.vv v16, v0, v8\n\t"
                           "vwmul.vv v24, v4, v12\n\t"
                           "vsetivli zero, 16, e16, m2\n\t"
                           "vmv.v.x v0, zero\n\t"
                           "vwredsum.vs v8, v16, v0\n\t"
                           "lb %[t4], 4(%[scale])\n\t"
                           "lb %[t5], 5(%[scale])\n\t"
                           "vwredsum.vs v9, v18, v0\n\t"
                           "vwredsum.vs v10, v20, v0\n\t"
                           "vwredsum.vs v11, v22, v0\n\t"
                           "vwredsum.vs v12, v24, v0\n\t"
                           "lb %[t6], 6(%[scale])\n\t"
                           "lb %[t7], 7(%[scale])\n\t"
                           "vwredsum.vs v13, v26, v0\n\t"
                           "vwredsum.vs v14, v28, v0\n\t"
                           "vwredsum.vs v15, v30, v0\n\t"
                           "vsetivli zero, 4, e32, m1\n\t"
                           "vmul.vx v0, v8, %[tmp]\n\t"
                           "vmul.vx v1, v9, %[t1]\n\t"
                           "vmacc.vx v0, %[t2], v10\n\t"
                           "vmacc.vx v1, %[t3], v11\n\t"
                           "vmacc.vx v0, %[t4], v12\n\t"
                           "vmacc.vx v1, %[t5], v13\n\t"
                           "vmacc.vx v0, %[t6], v14\n\t"
                           "vmacc.vx v1, %[t7], v15\n\t"
                           "vmv.x.s %[tmp], v0\n\t"
                           "vmv.x.s %[t1], v1\n\t"
                           "add %[isum], %[isum], %[tmp]\n\t"
                           "add %[isum], %[isum], %[t1]"
                           : [tmp] "=&r"(tmp), [t1] "=&r"(t1), [t2] "=&r"(t2), [t3] "=&r"(t3), [t4] "=&r"(t4),
                             [t5] "=&r"(t5), [t6] "=&r"(t6), [t7] "=&r"(t7), [m] "+&r"(m), [isum] "+&r"(isum)
                           : [vl128] "r"(128), [vl64] "r"(64), [vl32] "r"(32), [q3] "r"(q3), [qh] "r"(qh),
                             [scale] "r"(scale), [q8] "r"(q8)
                           : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                             "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                             "v26", "v27", "v28", "v29", "v30", "v31");
      q3 += 32;
      q8 += 128;
      scale += 8;
    }

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    vSumf += cD * isum;
  }

  *pS = vSumf;
}

void fGgmlVecDotQ3KQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
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
  uint32_t vUtmp[4];
  float vSumf = 0;
  uint32_t aAux[3];

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(aAux, vX[vI].scales, 12);
    vUtmp[3] = ((aAux[1] >> 4) & cKmask2) | (((aAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((aAux[0] >> 4) & cKmask2) | (((aAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (aAux[1] & cKmask2) | (((aAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (aAux[0] & cKmask2) | (((aAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;
    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= 32;

    size_t vVl = 32;
    uint8_t vM = 1;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8m1_t vVqh = __riscv_vle8_v_u8m1(vQh, vVl);

    int vSumT = 0;

    for (int vJ = 0; vJ < QK_K; vJ += 128) {

      vVl = 32;

      // load Q3
      vuint8m1_t vQ3X = __riscv_vle8_v_u8m1(vQ3, vVl);

      vint8m1_t vQ30 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(vQ3X, 0x03, vVl));
      vint8m1_t vQ31 =
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ3X, 0x2, vVl), 0x03, vVl));
      vint8m1_t vQ32 =
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ3X, 0x4, vVl), 0x03, vVl));
      vint8m1_t vQ33 =
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQ3X, 0x6, vVl), 0x03, vVl));

      // compute mask for subtraction
      vuint8m1_t vQhM0 = __riscv_vand_vx_u8m1(vVqh, vM, vVl);
      vbool8_t vVmask0 = __riscv_vmseq_vx_u8m1_b8(vQhM0, 0, vVl);
      vint8m1_t vQ3M0 = __riscv_vsub_vx_i8m1_mu(vVmask0, vQ30, vQ30, 0x4, vVl);
      vM <<= 1;

      vuint8m1_t vQhM1 = __riscv_vand_vx_u8m1(vVqh, vM, vVl);
      vbool8_t vVmask1 = __riscv_vmseq_vx_u8m1_b8(vQhM1, 0, vVl);
      vint8m1_t vQ3M1 = __riscv_vsub_vx_i8m1_mu(vVmask1, vQ31, vQ31, 0x4, vVl);
      vM <<= 1;

      vuint8m1_t vQhM2 = __riscv_vand_vx_u8m1(vVqh, vM, vVl);
      vbool8_t vVmask2 = __riscv_vmseq_vx_u8m1_b8(vQhM2, 0, vVl);
      vint8m1_t vQ3M2 = __riscv_vsub_vx_i8m1_mu(vVmask2, vQ32, vQ32, 0x4, vVl);
      vM <<= 1;

      vuint8m1_t vQhM3 = __riscv_vand_vx_u8m1(vVqh, vM, vVl);
      vbool8_t vVmask3 = __riscv_vmseq_vx_u8m1_b8(vQhM3, 0, vVl);
      vint8m1_t vQ3M3 = __riscv_vsub_vx_i8m1_mu(vVmask3, vQ33, vQ33, 0x4, vVl);
      vM <<= 1;

      // load Q8 and take product with Q3
      vint16m2_t vA0 = __riscv_vwmul_vv_i16m2(vQ3M0, __riscv_vle8_v_i8m1(vQ8, vVl), vVl);
      vint16m2_t vA1 = __riscv_vwmul_vv_i16m2(vQ3M1, __riscv_vle8_v_i8m1(vQ8 + 32, vVl), vVl);
      vint16m2_t vA2 = __riscv_vwmul_vv_i16m2(vQ3M2, __riscv_vle8_v_i8m1(vQ8 + 64, vVl), vVl);
      vint16m2_t vA3 = __riscv_vwmul_vv_i16m2(vQ3M3, __riscv_vle8_v_i8m1(vQ8 + 96, vVl), vVl);

      vVl = 16;

      // retrieve lane to multiply with scale
      vint32m2_t vAux00 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA0, 0), (vScale[0]), vVl);
      vint32m2_t vAux01 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA0, 1), (vScale[1]), vVl);
      vint32m2_t vAux10 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA1, 0), (vScale[2]), vVl);
      vint32m2_t vAux11 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA1, 1), (vScale[3]), vVl);
      vint32m2_t vAux20 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA2, 0), (vScale[4]), vVl);
      vint32m2_t vAux21 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA2, 1), (vScale[5]), vVl);
      vint32m2_t vAux30 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA3, 0), (vScale[6]), vVl);
      vint32m2_t vAux31 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vA3, 1), (vScale[7]), vVl);

      vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vAux00, vAux01, vVl), vVzero, vVl);
      vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vAux10, vAux11, vVl), vIsum0, vVl);
      vint32m1_t vIsum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vAux20, vAux21, vVl), vIsum1, vVl);
      vint32m1_t vIsum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vAux30, vAux31, vVl), vIsum2, vVl);

      vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum3);

      vQ3 += 32;
      vQ8 += 128;
      vScale += 8;
    }

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}

void fGgmlVecDotQ3KQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
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

  // mask for processing 16 elements per prod register
  const vuint16m1_t cVaIndex = __riscv_vid_v_u16m1(32);
  const vbool16_t cVaMask = __riscv_vmsgtu_vx_u16m1_b16(cVaIndex, 15, 32);

  uint32_t vUtmp[4];
  float vSumf = 0;
  uint32_t aAux[3];

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(aAux, vX[vI].scales, 12);
    vUtmp[3] = ((aAux[1] >> 4) & cKmask2) | (((aAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((aAux[0] >> 4) & cKmask2) | (((aAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (aAux[1] & cKmask2) | (((aAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (aAux[0] & cKmask2) | (((aAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;
    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= 32;

    size_t vVl = 32;
    uint8_t vM = 1;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8mf2_t vVqh = __riscv_vle8_v_u8mf2(vQh, vVl);

    int vSumT = 0;

    vint32m2_t vVaux0 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux1 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux2 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux3 = __riscv_vmv_v_x_i32m2(0, vVl);

    for (int vJ = 0; vJ < QK_K; vJ += 128) {

      vVl = 32;

      // load Q3
      vuint8mf2_t vQ3X = __riscv_vle8_v_u8mf2(vQ3, vVl);

      vint8mf2_t vQ30 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(vQ3X, 0x03, vVl));
      vint8mf2_t vQ31 =
          __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQ3X, 0x2, vVl), 0x03, vVl));
      vint8mf2_t vQ32 =
          __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQ3X, 0x4, vVl), 0x03, vVl));
      vint8mf2_t vQ33 =
          __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQ3X, 0x6, vVl), 0x03, vVl));

      // compute mask for subtraction
      vuint8mf2_t vQhM0 = __riscv_vand_vx_u8mf2(vVqh, vM, vVl);
      vbool16_t vVmask0 = __riscv_vmseq_vx_u8mf2_b16(vQhM0, 0, vVl);
      vint8mf2_t vQ3M0 = __riscv_vsub_vx_i8mf2_mu(vVmask0, vQ30, vQ30, 0x4, vVl);
      vM <<= 1;

      vuint8mf2_t vQhM1 = __riscv_vand_vx_u8mf2(vVqh, vM, vVl);
      vbool16_t vVmask1 = __riscv_vmseq_vx_u8mf2_b16(vQhM1, 0, vVl);
      vint8mf2_t vQ3M1 = __riscv_vsub_vx_i8mf2_mu(vVmask1, vQ31, vQ31, 0x4, vVl);
      vM <<= 1;

      vuint8mf2_t vQhM2 = __riscv_vand_vx_u8mf2(vVqh, vM, vVl);
      vbool16_t vVmask2 = __riscv_vmseq_vx_u8mf2_b16(vQhM2, 0, vVl);
      vint8mf2_t vQ3M2 = __riscv_vsub_vx_i8mf2_mu(vVmask2, vQ32, vQ32, 0x4, vVl);
      vM <<= 1;

      vuint8mf2_t vQhM3 = __riscv_vand_vx_u8mf2(vVqh, vM, vVl);
      vbool16_t vVmask3 = __riscv_vmseq_vx_u8mf2_b16(vQhM3, 0, vVl);
      vint8mf2_t vQ3M3 = __riscv_vsub_vx_i8mf2_mu(vVmask3, vQ33, vQ33, 0x4, vVl);
      vM <<= 1;

      // load Q8 and take product
      vint16m1_t vVaQ0 = __riscv_vwmul_vv_i16m1(vQ3M0, __riscv_vle8_v_i8mf2(vQ8, vVl), vVl);
      vint16m1_t vVaQ1 = __riscv_vwmul_vv_i16m1(vQ3M1, __riscv_vle8_v_i8mf2(vQ8 + 32, vVl), vVl);
      vint16m1_t vVaQ2 = __riscv_vwmul_vv_i16m1(vQ3M2, __riscv_vle8_v_i8mf2(vQ8 + 64, vVl), vVl);
      vint16m1_t vVaQ3 = __riscv_vwmul_vv_i16m1(vQ3M3, __riscv_vle8_v_i8mf2(vQ8 + 96, vVl), vVl);

      // accumulate
      vVaux0 = __riscv_vwmacc_vx_i32m2(vVaux0, vScale[0], vVaQ0, 16);
      vVaux1 = __riscv_vwmacc_vx_i32m2(vVaux1, vScale[2], vVaQ1, 16);
      vVaux2 = __riscv_vwmacc_vx_i32m2(vVaux2, vScale[4], vVaQ2, 16);
      vVaux3 = __riscv_vwmacc_vx_i32m2(vVaux3, vScale[6], vVaQ3, 16);
      //
      vVaux0 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux0, vScale[1], vVaQ0, vVl);
      vVaux1 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux1, vScale[3], vVaQ1, vVl);
      vVaux2 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux2, vScale[5], vVaQ2, vVl);
      vVaux3 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux3, vScale[7], vVaQ3, vVl);

      vQ3 += 32;
      vQ8 += 128;
      vScale += 8;
    }

    vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux0, vVaux1, vVl), vVzero, vVl);
    vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux2, vVaux3, vVl), vIsum0, vVl);

    vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum1);

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}

void fGgmlVecDotQ3KQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
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

  // mask for processing 16 elements per prod register
  const vuint16mf2_t cVaIndex = __riscv_vid_v_u16mf2(32);
  const vbool32_t cVaMask = __riscv_vmsgtu_vx_u16mf2_b32(cVaIndex, 15, 32);

  uint32_t vUtmp[4];
  float vSumf = 0;
  uint32_t aAux[3];

  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(aAux, vX[vI].scales, 12);
    vUtmp[3] = ((aAux[1] >> 4) & cKmask2) | (((aAux[2] >> 6) & cKmask1) << 4);
    vUtmp[2] = ((aAux[0] >> 4) & cKmask2) | (((aAux[2] >> 4) & cKmask1) << 4);
    vUtmp[1] = (aAux[1] & cKmask2) | (((aAux[2] >> 2) & cKmask1) << 4);
    vUtmp[0] = (aAux[0] & cKmask2) | (((aAux[2] >> 0) & cKmask1) << 4);

    int8_t *vScale = (int8_t *)vUtmp;
    for (int vJ = 0; vJ < 16; ++vJ)
      vScale[vJ] -= 32;

    size_t vVl = 32;
    uint8_t vM = 1;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8mf4_t vVqh = __riscv_vle8_v_u8mf4(vQh, vVl);

    int vSumT = 0;

    vint32m1_t vVaux0 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux1 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux2 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux3 = __riscv_vmv_v_x_i32m1(0, vVl);

    for (int vJ = 0; vJ < QK_K; vJ += 128) {

      vVl = 32;

      // load Q3
      vuint8mf4_t vQ3X = __riscv_vle8_v_u8mf4(vQ3, vVl);

      vint8mf4_t vQ30 = __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(vQ3X, 0x03, vVl));
      vint8mf4_t vQ31 =
          __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQ3X, 0x2, vVl), 0x03, vVl));
      vint8mf4_t vQ32 =
          __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQ3X, 0x4, vVl), 0x03, vVl));
      vint8mf4_t vQ33 =
          __riscv_vreinterpret_v_u8mf4_i8mf4(__riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQ3X, 0x6, vVl), 0x03, vVl));

      // compute mask for subtraction
      vuint8mf4_t vQhM0 = __riscv_vand_vx_u8mf4(vVqh, vM, vVl);
      vbool32_t vVmask0 = __riscv_vmseq_vx_u8mf4_b32(vQhM0, 0, vVl);
      vint8mf4_t vQ3M0 = __riscv_vsub_vx_i8mf4_mu(vVmask0, vQ30, vQ30, 0x4, vVl);
      vM <<= 1;

      vuint8mf4_t vQhM1 = __riscv_vand_vx_u8mf4(vVqh, vM, vVl);
      vbool32_t vVmask1 = __riscv_vmseq_vx_u8mf4_b32(vQhM1, 0, vVl);
      vint8mf4_t vQ3M1 = __riscv_vsub_vx_i8mf4_mu(vVmask1, vQ31, vQ31, 0x4, vVl);
      vM <<= 1;

      vuint8mf4_t vQhM2 = __riscv_vand_vx_u8mf4(vVqh, vM, vVl);
      vbool32_t vVmask2 = __riscv_vmseq_vx_u8mf4_b32(vQhM2, 0, vVl);
      vint8mf4_t vQ3M2 = __riscv_vsub_vx_i8mf4_mu(vVmask2, vQ32, vQ32, 0x4, vVl);
      vM <<= 1;

      vuint8mf4_t vQhM3 = __riscv_vand_vx_u8mf4(vVqh, vM, vVl);
      vbool32_t vVmask3 = __riscv_vmseq_vx_u8mf4_b32(vQhM3, 0, vVl);
      vint8mf4_t vQ3M3 = __riscv_vsub_vx_i8mf4_mu(vVmask3, vQ33, vQ33, 0x4, vVl);
      vM <<= 1;

      // load Q8 and take product
      vint16mf2_t vVaQ0 = __riscv_vwmul_vv_i16mf2(vQ3M0, __riscv_vle8_v_i8mf4(vQ8, vVl), vVl);
      vint16mf2_t vVaQ1 = __riscv_vwmul_vv_i16mf2(vQ3M1, __riscv_vle8_v_i8mf4(vQ8 + 32, vVl), vVl);
      vint16mf2_t vVaQ2 = __riscv_vwmul_vv_i16mf2(vQ3M2, __riscv_vle8_v_i8mf4(vQ8 + 64, vVl), vVl);
      vint16mf2_t vVaQ3 = __riscv_vwmul_vv_i16mf2(vQ3M3, __riscv_vle8_v_i8mf4(vQ8 + 96, vVl), vVl);

      // accumulate
      vVaux0 = __riscv_vwmacc_vx_i32m1(vVaux0, vScale[0], vVaQ0, 16);
      vVaux1 = __riscv_vwmacc_vx_i32m1(vVaux1, vScale[2], vVaQ1, 16);
      vVaux2 = __riscv_vwmacc_vx_i32m1(vVaux2, vScale[4], vVaQ2, 16);
      vVaux3 = __riscv_vwmacc_vx_i32m1(vVaux3, vScale[6], vVaQ3, 16);
      //
      vVaux0 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux0, vScale[1], vVaQ0, vVl);
      vVaux1 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux1, vScale[3], vVaQ1, vVl);
      vVaux2 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux2, vScale[5], vVaQ2, vVl);
      vVaux3 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux3, vScale[7], vVaQ3, vVl);

      vQ3 += 32;
      vQ8 += 128;
      vScale += 8;
    }

    vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vVaux0, vVaux1, vVl), vVzero, vVl);
    vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vVaux2, vVaux3, vVl), vIsum0, vVl);

    vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum1);

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_q3_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_xtheadvector
  ggml_vec_dot_q3_K_q8_K_xtheadvector(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#elif defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotQ3KQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotQ3KQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotQ3KQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 1024:
    fGgmlVecDotQ3KQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    ggml_vec_dot_q3_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_q3_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_xtheadvector
static NOINLINE void ggml_vec_dot_q4_K_q8_K_xtheadvector(int n, float *GGML_RESTRICT s, size_t bs,
                                                         const void *GGML_RESTRICT vx, size_t bx,
                                                         const void *GGML_RESTRICT vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);
  UNUSED(nrc);
  UNUSED(bx);
  UNUSED(by);
  UNUSED(bs);

  const block_q4_K *GGML_RESTRICT x = vx;
  const block_q8_K *GGML_RESTRICT y = vy;

  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t *scales = (const uint8_t *)&utmp[0];
  const uint8_t *mins = (const uint8_t *)&utmp[2];

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

    int tmp, tmp2, sumi;
    __asm__ __volatile__("li %[t1], 12\n\t"
                         "th.vsetvli zero, %[t1], e8, m1\n\t"
                         "th.vlb.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
                         "li %[t1], 4\n\t"
                         "th.vsetvli zero, %[t1], e32, m1\n\t"
                         "th.vslidedown.vi v2, v1, 2\n\t"
                         "th.vmv.v.v v3, v2\n\t"
                         "th.vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                         "li %[t1], 2\n\t"
                         "th.vsetvli zero, %[t1], e32, m1\n\t"
                         "th.vmv.v.i v4, 4\n\t"
                         "th.vand.vx v8, v1, %[kmask1]\n\t"
                         "th.vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                         "th.vsrl.vi v6, v1, 6\n\t"
                         "th.vsrl.vv v7, v2, v5\n\t"
                         "th.vand.vx v0, v6, %[kmask3]\n\t"
                         "th.vand.vx v2, v7, %[kmask2]\n\t"
                         "th.vsll.vi v6, v0, 4\n\t"
                         "li %[t2], 8\n\t"
                         "addi %[t1], %[utmp], 4\n\t"
                         "th.vor.vv v1, v6, v2\n\t"
                         "th.vssw.v v8, (%[utmp]), %[t2]\n\t"
                         "th.vssw.v v1, (%[t1]), %[t2]\n\t"
                         "th.vsetvli zero, zero, e32, m2\n\t" // vl == 8
                         "th.vlw.v v2, (%[bsums])\n\t"
                         "th.vsetvli zero, %[t2], e16, m1\n\t"
                         "th.vnsrl.vi v0, v2, 0\n\t"
                         "th.vnsrl.vi v1, v2, 16\n\t"
                         "th.vadd.vv v2, v0, v1\n\t"
                         "th.vlbu.v v4, (%[mins])\n\t"
                         "th.vwmul.vv v6, v4, v2\n\t"
                         "th.vmv.v.x v0, zero\n\t"
                         "th.vsetvli zero, %[t2], e32, m2\n\t"
                         "th.vredsum.vs v0, v6, v0\n\t"
                         "th.vmv.x.s %[sumi], v0"
                         : [t1] "=&r"(tmp), [t2] "=&r"(tmp2), [sumi] "=&r"(sumi)
                         : [bsums] "r"(y[i].bsums), [mins] "r"(mins), [utmp] "r"(utmp), [s6b] "r"(x[i].scales),
                           [kmask1] "r"(kmask1), [kmask2] "r"(kmask2), [kmask3] "r"(kmask3)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");
    sumf -= dmin * sumi;

    const uint8_t *restrict q4 = x[i].qs;
    const int8_t *restrict q8 = y[i].qs;

    sumi = 0;
    const uint8_t *scale = scales;

    for (int j = 0; j < QK_K / 128; ++j) {
      int vl128 = 128, vl64 = 64, vl32 = 32;
      __asm__ __volatile__(
          "th.vsetvli zero, %[vl128], e8, m8\n\t"
          "th.vlb.v v8, (%[q8])\n\t"
          "th.vsetvli zero, %[vl64], e8, m4\n\t"
          "th.vlb.v v0, (%[q4])\n\t"
          "th.vsrl.vi v4, v0, 4\n\t"
          "th.vand.vi v0, v0, 0xF\n\t"
          "th.vsetvli zero, %[vl32], e8, m2\n\t"
          "th.vwmul.vv v28, v6, v14\n\t"
          "th.vwmul.vv v20, v4, v10\n\t"
          "th.vwmul.vv v24, v2, v12\n\t"
          "th.vwmul.vv v16, v0, v8\n\t"
          "li %[tmp], 4\n\t"
          "th.vsetvli zero, %[tmp], e32, m1\n\t"
          "th.vlbu.v v1, (%[scale])\n\t"
          "th.vmv.v.x v0, zero\n\t"
          "th.vsetvli zero, %[vl32], e16, m4\n\t"
          "th.vwredsum.vs v6, v24, v0\n\t"
          "th.vwredsum.vs v7, v28, v0\n\t"
          "th.vwredsum.vs v4, v16, v0\n\t"
          "th.vwredsum.vs v5, v20, v0\n\t"
          "th.vsetvli zero, %[tmp], e32, m1\n\t"
          "th.vslideup.vi v6, v7, 1\n\t"
          "th.vslideup.vi v4, v5, 1\n\t"
          "th.vslideup.vi v4, v6, 2\n\t"
          "th.vmul.vv v8, v4, v1\n\t"
          "th.vredsum.vs v0, v8, v0\n\t"
          "th.vmv.x.s %[tmp], v0\n\t"
          "add %[sumi], %[sumi], %[tmp]"
          : [tmp] "=&r"(tmp), [sumi] "+&r"(sumi)
          : [vl128] "r"(vl128), [vl64] "r"(vl64), [vl32] "r"(vl32), [q4] "r"(q4), [q8] "r"(q8), [scale] "r"(scale)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
            "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
            "v30", "v31");

      q4 += 64;
      q8 += 128;
      scale += 4;
    }

    sumf += d * sumi;
  }

  *s = sumf;
}
#endif

#if defined __riscv_v
static NOINLINE void fGgmlVecDotQ4KQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                            size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t *vScales = (const uint8_t *)&utmp[0];
  const uint8_t *mins = (const uint8_t *)&utmp[2];

  float sumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float d = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float dmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    float ftmp, ft2;
    const uint8_t *restrict q40;
    const uint8_t *restrict q41;
    const uint8_t *restrict q42;
    const uint8_t *restrict q43;
    const int8_t *restrict q80;
    const int8_t *restrict q81;
    const int8_t *restrict q82;
    const int8_t *restrict q83;
    int s0, s1, s2, s3;

    __asm__ __volatile__("li %[s1], 8\n\t"
                         "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                         "vle32.v v1, (%[s6b])\n\t"
                         "vslide1down.vx v1, v1, zero\n\t"
                         "vmv.v.x v16, zero\n\t"
                         "vslidedown.vi v2, v1, 2\n\t"
                         "vmv1r.v v3, v2\n\t"
                         "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                         "vsetivli zero, 2, e32, m1, ta, ma\n\t"
                         "vmv.v.i v4, 4\n\t"
                         "vand.vx v8, v1, %[kmask1]\n\t"
                         "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                         "vsrl.vi v6, v1, 6\n\t"
                         "vsrl.vv v7, v2, v5\n\t"
                         "vsse32.v v8, (%[utmp]), %[s1]\n\t"
                         "vand.vx v0, v6, %[kmask3]\n\t"
                         "vand.vx v2, v7, %[kmask2]\n\t"
                         "vsll.vi v6, v0, 4\n\t"
                         "addi %[s0], %[utmp], 4\n\t"
                         "vor.vv v1, v6, v2\n\t"
                         "vsse32.v v1, (%[s0]), %[s1]\n\t"
                         "vsetivli zero, 8, e16, m1, ta, ma\n\t"
                         "vle32.v v2, (%[bsums])\n\t"
                         "vnsrl.wi v0, v2, 0\n\t"
                         "vnsrl.wi v1, v2, 16\n\t"
                         "vadd.vv v2, v0, v1\n\t"
                         "vle8.v v3, (%[mins])\n\t"
                         "vzext.vf2 v4, v3\n\t"
                         "vwmul.vv v6, v4, v2\n\t"
                         "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                         "vredsum.vs v0, v6, v16\n\t"
                         "vredsum.vs v0, v7, v0\n\t"
                         "vfcvt.f.x.v v0, v0\n\t"
                         "vfmv.f.s %[ftmp], v0\n\t"
                         "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                         "vle8.v v0, (%[xs])\n\t"
                         "fnmsub.s %[sumf], %[dmin], %[ftmp], %[sumf]\n\t"
                         "addi %[q40], %[xs], 64\n\t"
                         "addi %[q41], %[xs], 16\n\t"
                         "addi %[q42], %[xs], 32\n\t"
                         "addi %[q43], %[xs], 48\n\t"
                         "addi %[q80], %[ys], 64\n\t"
                         "vle8.v v1, (%[q41])\n\t"
                         "vle8.v v2, (%[q42])\n\t"
                         "addi %[q81], %[ys], 16\n\t"
                         "addi %[q41], %[q41], 64\n\t"
                         "addi %[q82], %[ys], 32\n\t"
                         "vle8.v v3, (%[q43])\n\t"
                         "vle8.v v8, (%[ys])\n\t"
                         "addi %[q42], %[q42], 64\n\t"
                         "addi %[q83], %[ys], 48\n\t"
                         "addi %[q43], %[q43], 64\n\t"
                         "vsrl.vi v4, v0, 4\n\t"
                         "vle8.v v9, (%[q81])\n\t"
                         "vle8.v v10, (%[q82])\n\t"
                         "vand.vi v0, v0, 0xF\n\t"
                         "addi %[q81], %[q81], 64\n\t"
                         "vsrl.vi v5, v1, 4\n\t"
                         "addi %[q82], %[q82], 64\n\t"
                         "vle8.v v11, (%[q83])\n\t"
                         "vle8.v v12, (%[q80])\n\t"
                         "vand.vi v1, v1, 0xF\n\t"
                         "addi %[q83], %[q83], 64\n\t"
                         "vsrl.vi v6, v2, 4\n\t"
                         "addi %[q80], %[q80], 64\n\t"
                         "vle8.v v13, (%[q81])\n\t"
                         "vle8.v v14, (%[q82])\n\t"
                         "vand.vi v2, v2, 0xF\n\t"
                         "addi %[q81], %[q81], 64\n\t"
                         "vsrl.vi v7, v3, 4\n\t"
                         "addi %[q82], %[q82], 64\n\t"
                         "vwmul.vv v16, v0, v8\n\t"
                         "vle8.v v15, (%[q83])\n\t"
                         "vle8.v v0, (%[q40])\n\t"
                         "vand.vi v3, v3, 0xF\n\t"
                         "addi %[q83], %[q83], 64\n\t"
                         "vwmul.vv v24, v2, v12\n\t"
                         "vwmul.vv v20, v4, v10\n\t"
                         "vwmul.vv v28, v6, v14\n\t"
                         "vwmacc.vv v16, v1, v9\n\t"
                         "vle8.v v1, (%[q41])\n\t"
                         "vle8.v v2, (%[q42])\n\t"
                         "vwmacc.vv v24, v3, v13\n\t"
                         "vwmacc.vv v20, v5, v11\n\t"
                         "vwmacc.vv v28, v7, v15\n\t"
                         "addi %[q40], %[q80], 64\n\t"
                         "addi %[q41], %[q81], 64\n\t"
                         "vle8.v v3, (%[q43])\n\t"
                         "vle8.v v8, (%[q80])\n\t"
                         "addi %[q42], %[q82], 64\n\t"
                         "addi %[q43], %[q83], 64\n\t"
                         "vsrl.vi v4, v0, 4\n\t"
                         "vle8.v v9, (%[q81])\n\t"
                         "vle8.v v10, (%[q82])\n\t"
                         "vand.vi v0, v0, 0xF\n\t"
                         "vsrl.vi v5, v1, 4\n\t"
                         "vsrl.vi v7, v3, 4\n\t"
                         "vand.vi v3, v3, 0xF\n\t"
                         "vle8.v v11, (%[q83])\n\t"
                         "vle8.v v12, (%[q40])\n\t"
                         "vand.vi v1, v1, 0xF\n\t"
                         "vsrl.vi v6, v2, 4\n\t"
                         "vand.vi v2, v2, 0xF\n\t"
                         "vwmul.vv v18, v0, v8\n\t"
                         "vle8.v v13, (%[q41])\n\t"
                         "vle8.v v14, (%[q42])\n\t"
                         "vwmul.vv v26, v2, v12\n\t"
                         "vwmul.vv v22, v4, v10\n\t"
                         "vwmul.vv v30, v6, v14\n\t"
                         "vwmacc.vv v18, v1, v9\n\t"
                         "vle8.v v15, (%[q43])\n\t"
                         "vwmacc.vv v26, v3, v13\n\t"
                         "vwmacc.vv v22, v5, v11\n\t"
                         "vwmacc.vv v30, v7, v15\n\t"
                         "vmv.v.x v0, zero\n\t"
                         "vsetivli zero, 16, e16, m2, ta, ma\n\t"
                         "vwredsum.vs v4, v16, v0\n\t"
                         "lbu %[s0], 0(%[scale])\n\t"
                         "vwredsum.vs v5, v20, v0\n\t"
                         "lbu %[s1], 1(%[scale])\n\t"
                         "vwredsum.vs v6, v24, v0\n\t"
                         "lbu %[s2], 2(%[scale])\n\t"
                         "vwredsum.vs v7, v28, v0\n\t"
                         "lbu %[s3], 3(%[scale])\n\t"
                         "vwredsum.vs v8, v18, v0\n\t"
                         "lbu %[q40], 4(%[scale])\n\t"
                         "vwredsum.vs v9, v22, v0\n\t"
                         "lbu %[q41], 5(%[scale])\n\t"
                         "vwredsum.vs v10, v26, v0\n\t"
                         "lbu %[q42], 6(%[scale])\n\t"
                         "vwredsum.vs v11, v30, v0\n\t"
                         "lbu %[q43], 7(%[scale])\n\t"
                         "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                         "vmul.vx v0, v4, %[s0]\n\t"
                         "vmul.vx v1, v8, %[q40]\n\t"
                         "vmacc.vx v0, %[s1], v5\n\t"
                         "vmacc.vx v1, %[q41], v9\n\t"
                         "vmacc.vx v0, %[s2], v6\n\t"
                         "vmacc.vx v1, %[q42], v10\n\t"
                         "vmacc.vx v0, %[s3], v7\n\t"
                         "vmacc.vx v1, %[q43], v11\n\t"
                         "vfcvt.f.x.v v0, v0\n\t"
                         "vfcvt.f.x.v v1, v1\n\t"
                         "vfmv.f.s %[ft2], v0\n\t"
                         "vfmv.f.s %[ftmp], v1\n\t"
                         "fadd.s %[ft2], %[ft2], %[ftmp]\n\t"
                         "fmadd.s %[sumf], %[d], %[ft2], %[sumf]"
                         : [ftmp] "=&f"(ftmp), [sumf] "+&f"(sumf), [ft2] "=&f"(ft2), [s0] "=&r"(s0), [s1] "=&r"(s1),
                           [s2] "=&r"(s2), [s3] "=&r"(s3), [q40] "=&r"(q40), [q41] "=&r"(q41), [q42] "=&r"(q42),
                           [q43] "=&r"(q43), [q80] "=&r"(q80), [q81] "=&r"(q81), [q82] "=&r"(q82), [q83] "=&r"(q83)
                         : [d] "f"(d), [ys] "r"(vY[vI].qs), [xs] "r"(vX[vI].qs), [scale] "r"(vScales),
                           [bsums] "r"(vY[vI].bsums), [mins] "r"(mins), [utmp] "r"(utmp), [s6b] "r"(&vX[vI]),
                           [kmask1] "r"(kmask1), [dmin] "f"(dmin), [kmask2] "r"(kmask2), [kmask3] "r"(kmask3)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                           "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                           "v26", "v27", "v28", "v29", "v30", "v31");
  }

  *pS = sumf;
}

static NOINLINE void fGgmlVecDotQ4KQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                            size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  const uint8_t *vScales = (const uint8_t *)&vUtmp[0];
  const uint8_t *vMins = (const uint8_t *)&vUtmp[2];

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    size_t vVl = 8;

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    vint16mf2_t vQ8sums0 = __riscv_vlse16_v_i16mf2(vY[vI].bsums, 4, vVl);
    vint16mf2_t vQ8sums1 = __riscv_vlse16_v_i16mf2(vY[vI].bsums + 1, 4, vVl);
    vint16mf2_t vQ8sums = __riscv_vadd_vv_i16mf2(vQ8sums0, vQ8sums1, vVl);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    vuint8mf4_t vMins8 = __riscv_vle8_v_u8mf4(vMins, vVl);
    vint16mf2_t vVMins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(vMins8, vVl));
    vint32m1_t vProd = __riscv_vwmul_vv_i32m1(vQ8sums, vVMins, vVl);

    vint32m1_t vSumi = __riscv_vredsum_vs_i32m1_i32m1(vProd, __riscv_vmv_v_x_i32m1(0, 1), vVl);
    vSumf -= cDmin * __riscv_vmv_x_s_i32m1_i32(vSumi);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    vVl = 32;

    int32_t vSum1 = 0;
    int32_t vSum2 = 0;

    vint16m1_t vVzero = __riscv_vmv_v_x_i16m1(0, 1);

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      // load Q4
      vuint8m1_t vQ4X = __riscv_vle8_v_u8m1(vQ4, vVl);

      // load Q8 and multiply it with lower Q4 nibble
      vint8m1_t vQ80 = __riscv_vle8_v_i8m1(vQ8, vVl);
      vint8m1_t vQ40 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(vQ4X, 0x0F, vVl));
      vint16m2_t vQv0 = __riscv_vwmul_vv_i16m2(vQ40, vQ80, vVl);
      vint16m1_t vVs0 = __riscv_vredsum_vs_i16m2_i16m1(vQv0, vVzero, vVl);

      vSum1 += __riscv_vmv_x_s_i16m1_i16(vVs0) * vScales[2 * vJ + 0];

      // load Q8 and multiply it with upper Q4 nibble
      vint8m1_t vQ81 = __riscv_vle8_v_i8m1(vQ8 + 32, vVl);
      vint8m1_t vQ41 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(vQ4X, 0x04, vVl));
      vint16m2_t vQv1 = __riscv_vwmul_vv_i16m2(vQ41, vQ81, vVl);
      vint16m1_t vVs1 = __riscv_vredsum_vs_i16m2_i16m1(vQv1, vVzero, vVl);

      vSum2 += __riscv_vmv_x_s_i16m1_i16(vVs1) * vScales[2 * vJ + 1];

      vQ4 += 32;
      vQ8 += 64;
    }

    vSumf += cD * (vSum1 + vSum2);
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_q4_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_xtheadvector
  ggml_vec_dot_q4_K_q8_K_xtheadvector(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#elif defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotQ4KQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 256 and above
    fGgmlVecDotQ4KQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
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

#if defined __riscv_v

  const uint8_t *vScales = (const uint8_t *)&vUtmp[0];
  const uint8_t *vMins = (const uint8_t *)&vUtmp[2];

  float vSumf = 0;
  float vSums = 0.0;

  size_t vVl;

  for (int vI = 0; vI < cNb; ++vI) {

    vVl = 8;

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vHm = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const float cDmin = GGML_CPU_FP16_TO_FP32(vX[vI].dmin) * vY[vI].d;

    vint16m1_t vQ8sums0 = __riscv_vlse16_v_i16m1(vY[vI].bsums, 4, vVl);
    vint16m1_t vQ8sums1 = __riscv_vlse16_v_i16m1(vY[vI].bsums + 1, 4, vVl);
    vint16m1_t vQ8sums = __riscv_vadd_vv_i16m1(vQ8sums0, vQ8sums1, vVl);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    vuint8mf2_t vMins8 = __riscv_vle8_v_u8mf2(vMins, vVl);
    vint16m1_t vVMins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vMins8, vVl));
    vint32m2_t vProd = __riscv_vwmul_vv_i32m2(vQ8sums, vVMins, vVl);

    vint32m1_t vSumi = __riscv_vredsum_vs_i32m2_i32m1(vProd, __riscv_vmv_v_x_i32m1(0, 1), vVl);
    vSumf -= cDmin * __riscv_vmv_x_s_i32m1_i32(vSumi);

    vVl = 32;
    int32_t vAux32 = 0;
    int vIs = 0;

    uint8_t vM = 1;
    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);
    vuint8m2_t vVqh = __riscv_vle8_v_u8m2(vHm, vVl);

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      // load Q5 and Q8
      vuint8m2_t vQ5X = __riscv_vle8_v_u8m2(vQ5, vVl);
      vint8m2_t vQ8Y1 = __riscv_vle8_v_i8m2(vQ8, vVl);
      vint8m2_t vQ8Y2 = __riscv_vle8_v_i8m2(vQ8 + 32, vVl);

      // compute mask for addition
      vint8m2_t vQ5A = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(vQ5X, 0x0F, vVl));
      vuint8m2_t vQhM1 = __riscv_vand_vx_u8m2(vVqh, vM, vVl);
      vbool4_t vVmask1 = __riscv_vmsne_vx_u8m2_b4(vQhM1, 0, vVl);
      vint8m2_t vQ5M1 = __riscv_vadd_vx_i8m2_mu(vVmask1, vQ5A, vQ5A, 16, vVl);
      vM <<= 1;

      vint8m2_t vQ5L = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(vQ5X, 0x04, vVl));
      vuint8m2_t vQhM2 = __riscv_vand_vx_u8m2(vVqh, vM, vVl);
      vbool4_t vVmask2 = __riscv_vmsne_vx_u8m2_b4(vQhM2, 0, vVl);
      vint8m2_t vQ5M2 = __riscv_vadd_vx_i8m2_mu(vVmask2, vQ5L, vQ5L, 16, vVl);
      vM <<= 1;

      vint16m4_t v0 = __riscv_vwmul_vv_i16m4(vQ5M1, vQ8Y1, vVl);
      vint16m4_t v1 = __riscv_vwmul_vv_i16m4(vQ5M2, vQ8Y2, vVl);

      vint32m8_t vVs1 = __riscv_vwmul_vx_i32m8(v0, vScales[vIs++], vVl);
      vint32m8_t vVs2 = __riscv_vwmul_vx_i32m8(v1, vScales[vIs++], vVl);

      vint32m1_t vVacc1 = __riscv_vredsum_vs_i32m8_i32m1(vVs1, vVzero, vVl);
      vint32m1_t vVacc2 = __riscv_vredsum_vs_i32m8_i32m1(vVs2, vVacc1, vVl);

      vAux32 += __riscv_vmv_x_s_i32m1_i32(vVacc2);
      vQ5 += 32;
      vQ8 += 64;
    }

    vSums += vAux32 * cD;
  }

  *pS = vSumf + vSums;

#else

  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  UNUSED(cNb);
  UNUSED(vUtmp);

  ggml_vec_dot_q5_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_xtheadvector
static NOINLINE void ggml_vec_dot_q6_K_q8_K_xtheadvector(int n, float *GGML_RESTRICT s, size_t bs,
                                                         const void *GGML_RESTRICT vx, size_t bx,
                                                         const void *GGML_RESTRICT vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);
  UNUSED(nrc);
  UNUSED(bx);
  UNUSED(by);
  UNUSED(bs);

  const block_q6_K *GGML_RESTRICT x = vx;
  const block_q8_K *GGML_RESTRICT y = vy;

  const int nb = n / QK_K;

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {

    const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

    const uint8_t *restrict q6 = x[i].ql;
    const uint8_t *restrict qh = x[i].qh;
    const int8_t *restrict q8 = y[i].qs;

    const int8_t *restrict scale = x[i].scales;

    int sum_t = 0;
    int t0;

    for (int j = 0; j < QK_K / 128; ++j) {
      __asm__ __volatile__("th.vsetvli zero, %[vl32], e8, m2\n\t" // vl == 32
                           "th.vlb.v v4, (%[qh])\n\t"
                           "th.vsll.vi v0, v4, 4\n\t"
                           "th.vsll.vi v2, v4, 2\n\t"
                           "th.vsrl.vi v6, v4, 2\n\t"
                           "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                           "th.vlb.v v8, (%[q6])\n\t"
                           "th.vsrl.vi v12, v8, 4\n\t"
                           "th.vand.vi v8, v8, 0xF\n\t"
                           "th.vsetvli zero, %[vl128], e8, m8\n\t" // vl == 128
                           "th.vand.vx v0, v0, %[mask]\n\t"
                           "th.vor.vv v8, v8, v0\n\t"
                           "th.vlb.v v0, (%[q8])\n\t"
                           "th.vsub.vx v8, v8, %[vl32]\n\t"
                           "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                           "th.vwmul.vv v16, v0, v8\n\t"
                           "th.vwmul.vv v24, v4, v12\n\t"
                           "li %[t0], 16\n\t"
                           "th.vsetvli zero, %[t0], e16, m2\n\t" // vl == 16
                           "th.vmv.v.x v0, zero\n\t"
                           "th.vwredsum.vs v10, v16, v0\n\t"
                           "th.vwredsum.vs v9, v18, v0\n\t"
                           "th.vwredsum.vs v8, v20, v0\n\t"
                           "th.vwredsum.vs v7, v22, v0\n\t"
                           "th.vwredsum.vs v11, v24, v0\n\t"
                           "th.vwredsum.vs v12, v26, v0\n\t"
                           "th.vwredsum.vs v13, v28, v0\n\t"
                           "th.vwredsum.vs v14, v30, v0\n\t"
                           "li %[t0], 4\n\t"
                           "th.vsetvli zero, %[t0], e32, m1\n\t" // vl == 4
                           "th.vslideup.vi v10, v9, 1\n\t"
                           "th.vslideup.vi v8, v7, 1\n\t"
                           "th.vslideup.vi v11, v12, 1\n\t"
                           "th.vslideup.vi v13, v14, 1\n\t"
                           "th.vslideup.vi v10, v8, 2\n\t"
                           "th.vslideup.vi v11, v13, 2\n\t"
                           "li %[t0], 8\n\t"
                           "th.vsetvli zero, %[t0], e32, m2\n\t" // vl == 8
                           "th.vlb.v v4, (%[scale])\n\t"
                           "th.vmul.vv v2, v4, v10\n\t"
                           "th.vredsum.vs v0, v2, v0\n\t"
                           "th.vmv.x.s %[t0], v0\n\t"
                           "add %[sumi], %[sumi], %[t0]"
                           : [sumi] "+&r"(sum_t), [t0] "=&r"(t0)
                           : [qh] "r"(qh), [q6] "r"(q6), [q8] "r"(q8), [scale] "r"(scale), [vl32] "r"(32),
                             [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30)
                           : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                             "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                             "v26", "v27", "v28", "v29", "v30", "v31");
      q6 += 64;
      qh += 32;
      q8 += 128;
      scale += 8;
    }

    sumf += d * sum_t;
  }

  *s = sumf;
}
#endif

#if defined __riscv_v
static NOINLINE void fGgmlVecDotQ6KQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                            size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float sumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    __builtin_prefetch(&vX[vI + 1].d, 0, 1);

    const float d = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *restrict q6 = vX[vI].ql;
    const uint8_t *restrict qh = vX[vI].qh;
    const int8_t *restrict q8 = vY[vI].qs;

    const int8_t *restrict scale = vX[vI].scales;

    int q6h;
    float ftmp;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      __asm__ __volatile__(
          "addi %[q6h], %[q6], 32\n\t"
          "ld t0, 0(%[scale])\n\t"
          "addi %[scale], %[scale], 8\n\t"
          "slli t6, t0, 1 * 8\n\t"
          "lb zero, 0(%[q6])\n\t"
          "slli t5, t0, 2 * 8\n\t"
          "slli t4, t0, 3 * 8\n\t"
          "lb zero, 0(%[q6h])\n\t"
          "slli t3, t0, 4 * 8\n\t"
          "slli t2, t0, 5 * 8\n\t"
          "lb zero, 0(%[qh])\n\t"
          "lb zero, 31(%[q6h])\n\t"
          "slli t1, t0, 6 * 8\n\t"
          "srai a7, t0, 56\n\t"
          "vsetvli zero, %[vl32], e8, m2\n\t"
          "vle8.v v8, (%[q6])\n\t"
          "srai t6, t6, 56\n\t"
          "srai t5, t5, 56\n\t"
          "srai t4, t4, 56\n\t"
          "srai t3, t3, 56\n\t"
          "vle8.v v10, (%[q6h])\n\t"
          "addi %[q6], %[q6], 64\n\t"
          "slli t0, t0, 7 * 8\n\t"
          "srai t2, t2, 56\n\t"
          "srai t1, t1, 56\n\t"
          "srai t0, t0, 56\n\t"
          "vle8.v v4, (%[qh])\n\t"
          "vsrl.vi v12, v8, 4\n\t"
          "vsrl.vi v14, v10, 4\n\t"
          "lb zero, 0(%[q8])\n\t"
          "vand.vi v8, v8, 0xF\n\t"
          "vand.vi v10, v10, 0xF\n\t"
          "lb zero, 32(%[q8])\n\t"
          "vsll.vi v0, v4, 4\n\t"
          "vsll.vi v2, v4, 2\n\t"
          "lb zero, 64(%[q8])\n\t"
          "vsrl.vi v6, v4, 2\n\t"
          "vand.vx v0, v0, %[mask]\n\t"
          "lb zero, 96(%[q8])\n\t"
          "vand.vx v2, v2, %[mask]\n\t"
          "vand.vx v4, v4, %[mask]\n\t"
          "vand.vx v6, v6, %[mask]\n\t"
          "vor.vv v8, v8, v0\n\t"
          "lb zero, 127(%[q8])\n\t"
          "vor.vv v10, v10, v2\n\t"
          "vor.vv v12, v12, v4\n\t"
          "vor.vv v14, v14, v6\n\t"
          "vsetvli zero, %[vl128], e8, m8\n\t"
          "vle8.v v0, (%[q8])\n\t"
          "vsub.vx v8, v8, %[vl32]\n\t"
          "vsetvli zero, %[vl64], e8, m4\n\t"
          "vwmul.vv v16, v0, v8\n\t"
          "vwmul.vv v24, v4, v12\n\t"
          "vsetivli zero, 16, e16, m2\n\t"
          "vmv.v.x v0, zero\n\t"
          "vwredsum.vs v10, v16, v0\n\t"
          "vwredsum.vs v9, v18, v0\n\t"
          "vwredsum.vs v8, v20, v0\n\t"
          "vwredsum.vs v7, v22, v0\n\t"
          "vwredsum.vs v11, v24, v0\n\t"
          "vwredsum.vs v12, v26, v0\n\t"
          "vwredsum.vs v13, v28, v0\n\t"
          "vwredsum.vs v14, v30, v0\n\t"
          "vsetivli zero, 4, e32, m1\n\t"
          "vmul.vx v0, v10, t0\n\t"
          "vmul.vx v1, v9, t1\n\t"
          "vmacc.vx v0, t2, v8\n\t"
          "vmacc.vx v1, t3, v7\n\t"
          "vmacc.vx v0, t4, v11\n\t"
          "vmacc.vx v1, t5, v12\n\t"
          "vmacc.vx v0, t6, v13\n\t"
          "vmacc.vx v1, a7, v14\n\t"
          "vadd.vv v0, v0, v1\n\t"
          "vfcvt.f.x.v v0, v0\n\t"
          "vfmv.f.s %[ftmp], v0\n\t"
          "fmadd.s %[sumf], %[d], %[ftmp], %[sumf]"
          : [q6] "+&r"(q6), [q6h] "=&r"(q6h), [scale] "+&r"(scale), [sumf] "+&f"(sumf), [ftmp] "=&f"(ftmp)
          : [qh] "r"(qh), [q8] "r"(q8), [vl32] "r"(32), [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30), [d] "f"(d)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
            "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
            "v30", "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7", "a6", "a5", "a4", "a3");
      qh += 32;
      q8 += 128;
    }
  }

  *pS = sumf;
}

static NOINLINE void fGgmlVecDotQ6KQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                            size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    size_t vVl;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);

    int vSumT = 0;
    int vIs = 0;

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      vVl = 32;

      // load qh
      vuint8m1_t vQhX = __riscv_vle8_v_u8m1(vQh, vVl);

      // load Q6
      vuint8m1_t vQ60 = __riscv_vle8_v_u8m1(vQ6, vVl);
      vuint8m1_t vQ61 = __riscv_vle8_v_u8m1(vQ6 + 32, vVl);

      vuint8m1_t vQ6a0 = __riscv_vand_vx_u8m1(vQ60, 0x0F, vVl);
      vuint8m1_t vQ6a1 = __riscv_vand_vx_u8m1(vQ61, 0x0F, vVl);
      vuint8m1_t vQ6s0 = __riscv_vsrl_vx_u8m1(vQ60, 0x04, vVl);
      vuint8m1_t vQ6s1 = __riscv_vsrl_vx_u8m1(vQ61, 0x04, vVl);

      vuint8m1_t vQh0 = __riscv_vand_vx_u8m1(vQhX, 0x03, vVl);
      vuint8m1_t vQh1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQhX, 0x2, vVl), 0x03, vVl);
      vuint8m1_t vQh2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQhX, 0x4, vVl), 0x03, vVl);
      vuint8m1_t vQh3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vQhX, 0x6, vVl), 0x03, vVl);

      vuint8m1_t vQhi0 = __riscv_vor_vv_u8m1(vQ6a0, __riscv_vsll_vx_u8m1(vQh0, 0x04, vVl), vVl);
      vuint8m1_t vQhi1 = __riscv_vor_vv_u8m1(vQ6a1, __riscv_vsll_vx_u8m1(vQh1, 0x04, vVl), vVl);
      vuint8m1_t vQhi2 = __riscv_vor_vv_u8m1(vQ6s0, __riscv_vsll_vx_u8m1(vQh2, 0x04, vVl), vVl);
      vuint8m1_t vQhi3 = __riscv_vor_vv_u8m1(vQ6s1, __riscv_vsll_vx_u8m1(vQh3, 0x04, vVl), vVl);

      vint8m1_t vA0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vQhi0), 32, vVl);
      vint8m1_t vA1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vQhi1), 32, vVl);
      vint8m1_t vA2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vQhi2), 32, vVl);
      vint8m1_t vA3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vQhi3), 32, vVl);

      // load Q8 and take product
      vint16m2_t vVaQ0 = __riscv_vwmul_vv_i16m2(vA0, __riscv_vle8_v_i8m1(vQ8, vVl), vVl);
      vint16m2_t vVaQ1 = __riscv_vwmul_vv_i16m2(vA1, __riscv_vle8_v_i8m1(vQ8 + 32, vVl), vVl);
      vint16m2_t vVaQ2 = __riscv_vwmul_vv_i16m2(vA2, __riscv_vle8_v_i8m1(vQ8 + 64, vVl), vVl);
      vint16m2_t vVaQ3 = __riscv_vwmul_vv_i16m2(vA3, __riscv_vle8_v_i8m1(vQ8 + 96, vVl), vVl);

      vVl = 16;

      vint32m2_t vVaux0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ0, 0), vScale[vIs + 0], vVl);
      vint32m2_t vVaux1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ0, 1), vScale[vIs + 1], vVl);
      vint32m2_t vVaux2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ1, 0), vScale[vIs + 2], vVl);
      vint32m2_t vVaux3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ1, 1), vScale[vIs + 3], vVl);
      vint32m2_t vVaux4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ2, 0), vScale[vIs + 4], vVl);
      vint32m2_t vVaux5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ2, 1), vScale[vIs + 5], vVl);
      vint32m2_t vVaux6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ3, 0), vScale[vIs + 6], vVl);
      vint32m2_t vVaux7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(vVaQ3, 1), vScale[vIs + 7], vVl);

      vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux0, vVaux1, vVl), vVzero, vVl);
      vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux2, vVaux3, vVl), vIsum0, vVl);
      vint32m1_t vIsum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux4, vVaux5, vVl), vIsum1, vVl);
      vint32m1_t vIsum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux6, vVaux7, vVl), vIsum2, vVl);

      vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum3);

      vQ6 += 64;
      vQh += 32;
      vQ8 += 128;
      vIs = 8;
    }

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotQ6KQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                            size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  // mask for processing 16 elements per prod register
  const vuint16m1_t cVaIndex = __riscv_vid_v_u16m1(32);
  const vbool16_t cVaMask = __riscv_vmsgtu_vx_u16m1_b16(cVaIndex, 15, 32);

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    size_t vVl = 32;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);

    int vSumT = 0;
    int vIs = 0;

    vint32m2_t vVaux0 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux1 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux2 = __riscv_vmv_v_x_i32m2(0, vVl);
    vint32m2_t vVaux3 = __riscv_vmv_v_x_i32m2(0, vVl);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load qh
      vuint8mf2_t vQhX = __riscv_vle8_v_u8mf2(vQh, vVl);

      // load Q6
      vuint8mf2_t vQ60 = __riscv_vle8_v_u8mf2(vQ6, vVl);
      vuint8mf2_t vQ61 = __riscv_vle8_v_u8mf2(vQ6 + 32, vVl);

      vuint8mf2_t vQ6a0 = __riscv_vand_vx_u8mf2(vQ60, 0x0F, vVl);
      vuint8mf2_t vQ6a1 = __riscv_vand_vx_u8mf2(vQ61, 0x0F, vVl);
      vuint8mf2_t vQ6s0 = __riscv_vsrl_vx_u8mf2(vQ60, 0x04, vVl);
      vuint8mf2_t vQ6s1 = __riscv_vsrl_vx_u8mf2(vQ61, 0x04, vVl);

      vuint8mf2_t vQh0 = __riscv_vand_vx_u8mf2(vQhX, 0x03, vVl);
      vuint8mf2_t vQh1 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQhX, 0x2, vVl), 0x03, vVl);
      vuint8mf2_t vQh2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQhX, 0x4, vVl), 0x03, vVl);
      vuint8mf2_t vQh3 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vQhX, 0x6, vVl), 0x03, vVl);

      vuint8mf2_t vQhi0 = __riscv_vor_vv_u8mf2(vQ6a0, __riscv_vsll_vx_u8mf2(vQh0, 0x04, vVl), vVl);
      vuint8mf2_t vQhi1 = __riscv_vor_vv_u8mf2(vQ6a1, __riscv_vsll_vx_u8mf2(vQh1, 0x04, vVl), vVl);
      vuint8mf2_t vQhi2 = __riscv_vor_vv_u8mf2(vQ6s0, __riscv_vsll_vx_u8mf2(vQh2, 0x04, vVl), vVl);
      vuint8mf2_t vQhi3 = __riscv_vor_vv_u8mf2(vQ6s1, __riscv_vsll_vx_u8mf2(vQh3, 0x04, vVl), vVl);

      vint8mf2_t vA0 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(vQhi0), 32, vVl);
      vint8mf2_t vA1 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(vQhi1), 32, vVl);
      vint8mf2_t vA2 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(vQhi2), 32, vVl);
      vint8mf2_t vA3 = __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(vQhi3), 32, vVl);

      // load Q8 and take product
      vint16m1_t vVaQ0 = __riscv_vwmul_vv_i16m1(vA0, __riscv_vle8_v_i8mf2(vQ8, vVl), vVl);
      vint16m1_t vVaQ1 = __riscv_vwmul_vv_i16m1(vA1, __riscv_vle8_v_i8mf2(vQ8 + 32, vVl), vVl);
      vint16m1_t vVaQ2 = __riscv_vwmul_vv_i16m1(vA2, __riscv_vle8_v_i8mf2(vQ8 + 64, vVl), vVl);
      vint16m1_t vVaQ3 = __riscv_vwmul_vv_i16m1(vA3, __riscv_vle8_v_i8mf2(vQ8 + 96, vVl), vVl);

      // accumulate
      vVaux0 = __riscv_vwmacc_vx_i32m2(vVaux0, vScale[vIs + 0], vVaQ0, 16);
      vVaux1 = __riscv_vwmacc_vx_i32m2(vVaux1, vScale[vIs + 2], vVaQ1, 16);
      vVaux2 = __riscv_vwmacc_vx_i32m2(vVaux2, vScale[vIs + 4], vVaQ2, 16);
      vVaux3 = __riscv_vwmacc_vx_i32m2(vVaux3, vScale[vIs + 6], vVaQ3, 16);
      //
      vVaux0 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux0, vScale[vIs + 1], vVaQ0, vVl);
      vVaux1 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux1, vScale[vIs + 3], vVaQ1, vVl);
      vVaux2 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux2, vScale[vIs + 5], vVaQ2, vVl);
      vVaux3 = __riscv_vwmacc_vx_i32m2_m(cVaMask, vVaux3, vScale[vIs + 7], vVaQ3, vVl);

      vQ6 += 64;
      vQh += 32;
      vQ8 += 128;
      vIs = 8;
    }

    vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux0, vVaux1, vVl), vVzero, vVl);
    vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vVaux2, vVaux3, vVl), vIsum0, vVl);

    vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum1);

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotQ6KQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  // mask for processing 16 elements per prod register
  const vuint16mf2_t cVaIndex = __riscv_vid_v_u16mf2(32);
  const vbool32_t cVaMask = __riscv_vmsgtu_vx_u16mf2_b32(cVaIndex, 15, 32);

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ6 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const int8_t *GGML_RESTRICT vScale = vX[vI].scales;

    size_t vVl = 32;

    vint32m1_t vVzero = __riscv_vmv_v_x_i32m1(0, 1);

    int vSumT = 0;
    int vIs = 0;

    vint32m1_t vVaux0 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux1 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux2 = __riscv_vmv_v_x_i32m1(0, vVl);
    vint32m1_t vVaux3 = __riscv_vmv_v_x_i32m1(0, vVl);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load qh
      vuint8mf4_t vQhX = __riscv_vle8_v_u8mf4(vQh, vVl);

      // load Q6
      vuint8mf4_t vQ60 = __riscv_vle8_v_u8mf4(vQ6, vVl);
      vuint8mf4_t vQ61 = __riscv_vle8_v_u8mf4(vQ6 + 32, vVl);

      vuint8mf4_t vQ6a0 = __riscv_vand_vx_u8mf4(vQ60, 0x0F, vVl);
      vuint8mf4_t vQ6a1 = __riscv_vand_vx_u8mf4(vQ61, 0x0F, vVl);
      vuint8mf4_t vQ6s0 = __riscv_vsrl_vx_u8mf4(vQ60, 0x04, vVl);
      vuint8mf4_t vQ6s1 = __riscv_vsrl_vx_u8mf4(vQ61, 0x04, vVl);

      vuint8mf4_t vQh0 = __riscv_vand_vx_u8mf4(vQhX, 0x03, vVl);
      vuint8mf4_t vQh1 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQhX, 0x2, vVl), 0x03, vVl);
      vuint8mf4_t vQh2 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQhX, 0x4, vVl), 0x03, vVl);
      vuint8mf4_t vQh3 = __riscv_vand_vx_u8mf4(__riscv_vsrl_vx_u8mf4(vQhX, 0x6, vVl), 0x03, vVl);

      vuint8mf4_t vQhi0 = __riscv_vor_vv_u8mf4(vQ6a0, __riscv_vsll_vx_u8mf4(vQh0, 0x04, vVl), vVl);
      vuint8mf4_t vQhi1 = __riscv_vor_vv_u8mf4(vQ6a1, __riscv_vsll_vx_u8mf4(vQh1, 0x04, vVl), vVl);
      vuint8mf4_t vQhi2 = __riscv_vor_vv_u8mf4(vQ6s0, __riscv_vsll_vx_u8mf4(vQh2, 0x04, vVl), vVl);
      vuint8mf4_t vQhi3 = __riscv_vor_vv_u8mf4(vQ6s1, __riscv_vsll_vx_u8mf4(vQh3, 0x04, vVl), vVl);

      vint8mf4_t vA0 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(vQhi0), 32, vVl);
      vint8mf4_t vA1 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(vQhi1), 32, vVl);
      vint8mf4_t vA2 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(vQhi2), 32, vVl);
      vint8mf4_t vA3 = __riscv_vsub_vx_i8mf4(__riscv_vreinterpret_v_u8mf4_i8mf4(vQhi3), 32, vVl);

      // load Q8 and take product
      vint16mf2_t vVaQ0 = __riscv_vwmul_vv_i16mf2(vA0, __riscv_vle8_v_i8mf4(vQ8, vVl), vVl);
      vint16mf2_t vVaQ1 = __riscv_vwmul_vv_i16mf2(vA1, __riscv_vle8_v_i8mf4(vQ8 + 32, vVl), vVl);
      vint16mf2_t vVaQ2 = __riscv_vwmul_vv_i16mf2(vA2, __riscv_vle8_v_i8mf4(vQ8 + 64, vVl), vVl);
      vint16mf2_t vVaQ3 = __riscv_vwmul_vv_i16mf2(vA3, __riscv_vle8_v_i8mf4(vQ8 + 96, vVl), vVl);

      // accumulate
      vVaux0 = __riscv_vwmacc_vx_i32m1(vVaux0, vScale[vIs + 0], vVaQ0, 16);
      vVaux1 = __riscv_vwmacc_vx_i32m1(vVaux1, vScale[vIs + 2], vVaQ1, 16);
      vVaux2 = __riscv_vwmacc_vx_i32m1(vVaux2, vScale[vIs + 4], vVaQ2, 16);
      vVaux3 = __riscv_vwmacc_vx_i32m1(vVaux3, vScale[vIs + 6], vVaQ3, 16);
      //
      vVaux0 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux0, vScale[vIs + 1], vVaQ0, vVl);
      vVaux1 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux1, vScale[vIs + 3], vVaQ1, vVl);
      vVaux2 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux2, vScale[vIs + 5], vVaQ2, vVl);
      vVaux3 = __riscv_vwmacc_vx_i32m1_m(cVaMask, vVaux3, vScale[vIs + 7], vVaQ3, vVl);

      vQ6 += 64;
      vQh += 32;
      vQ8 += 128;
      vIs = 8;
    }

    vint32m1_t vIsum0 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vVaux0, vVaux1, vVl), vVzero, vVl);
    vint32m1_t vIsum1 = __riscv_vredsum_vs_i32m1_i32m1(__riscv_vadd_vv_i32m1(vVaux2, vVaux3, vVl), vIsum0, vVl);

    vSumT += __riscv_vmv_x_s_i32m1_i32(vIsum1);

    vSumf += cD * vSumT;
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_q6_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_xtheadvector
  ggml_vec_dot_q6_K_q8_K_xtheadvector(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#elif defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotQ6KQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotQ6KQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotQ6KQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 1024:
    fGgmlVecDotQ6KQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq1SQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    // Load qh once for the entire superblock.
    vuint16m1_t vQh = __riscv_vle16_v_u16m1(vX[vI].qh, 8);

    // Calculate ls.
    vuint16m1_t vTemp = __riscv_vsrl_vx_u16m1(vQh, 12, 8);
    vTemp = __riscv_vand_vx_u16m1(vTemp, 7, 8);
    vint32m2_t vLs = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vwmulu_vx_u32m2(vTemp, 2, 8));
    vLs = __riscv_vadd_vx_i32m2(vLs, 1, 8);

    // Calculate delta.
    vbool16_t vMask = __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(vQh, 0x8000, 8), 0, 8);
    vint32m2_t vDeltaNeg = __riscv_vmv_v_x_i32m2(-1, 8);
    vint32m2_t vDeltaPos = __riscv_vmv_v_x_i32m2(1, 8);
    vint32m2_t vDelta = __riscv_vmerge_vvm_i32m2(vDeltaNeg, vDeltaPos, vMask, 8);

    // Load qs.
    vuint8m2_t vQs = __riscv_vle8_v_u8m2(vX[vI].qs, 32);

    // Prepare the indices.
    const uint64_t cShift = 0x0009000600030000;
    vuint16m4_t vQhShift = __riscv_vreinterpret_v_u64m4_u16m4(__riscv_vmv_v_x_u64m4(cShift, 8));
    vuint16m4_t vQhGatherIndex = __riscv_vreinterpret_v_i16m4_u16m4(
        __riscv_vdiv_vx_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vid_v_u16m4(32)), 4, 32));
    vuint16m4_t vQhExt = __riscv_vlmul_ext_v_u16m2_u16m4(__riscv_vlmul_ext_v_u16m1_u16m2(vQh));
    vuint16m4_t vQhIndex = __riscv_vrgather_vv_u16m4(vQhExt, vQhGatherIndex, 32);
    vQhIndex = __riscv_vsrl_vv_u16m4(vQhIndex, vQhShift, 32);
    vQhIndex = __riscv_vand_vx_u16m4(vQhIndex, 7, 32);
    vQhIndex = __riscv_vsll_vx_u16m4(vQhIndex, 8, 32);
    vQhIndex = __riscv_vor_vv_u16m4(vQhIndex, __riscv_vzext_vf2_u16m4(vQs, 32), 32);
    vuint16m4_t vIndex = __riscv_vsll_vx_u16m4(vQhIndex, 3, 32);

    // Final lsums.
    int32_t aLsumsS[8];
    vint32m1_t vOneScalar = __riscv_vmv_v_x_i32m1(0, 1);

    // Sub-blocks 1-2
    {
      vuint16m1_t vGridIndex0 = __riscv_vget_v_u16m4_u16m1(vIndex, 0);
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex0, 8));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(&vY[vI].qs[0], 64);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 128);
      aLsumsS[0] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 0), vOneScalar, 32));
      aLsumsS[1] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 1), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    // Sub-blocks 3-4
    {
      vuint16m1_t vGridIndex0 = __riscv_vget_v_u16m4_u16m1(vIndex, 1);
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex0, 8));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(&vY[vI].qs[64], 64);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 128);
      aLsumsS[2] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 0), vOneScalar, 32));
      aLsumsS[3] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 1), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    // Sub-blocks 5-6
    {
      vuint16m1_t vGridIndex0 = __riscv_vget_v_u16m4_u16m1(vIndex, 2);
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex0, 8));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(&vY[vI].qs[128], 64);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 128);
      aLsumsS[4] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 0), vOneScalar, 32));
      aLsumsS[5] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 1), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    // Sub-blocks 7-8
    {
      vuint16m1_t vGridIndex0 = __riscv_vget_v_u16m4_u16m1(vIndex, 3);
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex0, 8));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(&vY[vI].qs[192], 64);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 128);
      aLsumsS[6] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 0), vOneScalar, 32));
      aLsumsS[7] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(vLsum0, 1), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    vint32m2_t vLsums = __riscv_vle32_v_i32m2(&aLsumsS[0], 8);

    // Calculate the bsums.
    vint16m2_t vBsums0 = __riscv_vle16_v_i16m2(vY[vI].bsums, 16);
    const vuint32m2_t cBsumsI32 = __riscv_vreinterpret_v_u16m2_u32m2(__riscv_vreinterpret_v_i16m2_u16m2(vBsums0));
    const vint16m1_t cBsumsI320 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(cBsumsI32, 0, 8));
    const vint16m1_t cBsumsI321 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vnsrl_wx_u16m1(cBsumsI32, 16, 8));
    const vint32m2_t cBsums = __riscv_vwadd_vv_i32m2(cBsumsI320, cBsumsI321, 8);

    // Accumulation.
    vint32m2_t vSumiV = __riscv_vmul_vv_i32m2(vLs, vLsums, 8);
    vint32m2_t vSumi1V = __riscv_vmul_vv_i32m2(__riscv_vmul_vv_i32m2(vLs, vDelta, 8), cBsums, 8);

    // Update sumf.
    int vSumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vSumiV, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vSumi1V, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    vSumf += GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d * (vSumi + IQ1S_DELTA * vSumi1);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1SQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    // Load qh once for the entire superblock.
    vuint16mf2_t vQh = __riscv_vle16_v_u16mf2(vX[vI].qh, 8);

    // Calculate ls.
    vuint16mf2_t vTemp = __riscv_vsrl_vx_u16mf2(vQh, 12, 8);
    vTemp = __riscv_vand_vx_u16mf2(vTemp, 7, 8);
    vint32m1_t vLs = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vwmulu_vx_u32m1(vTemp, 2, 8));
    vLs = __riscv_vadd_vx_i32m1(vLs, 1, 8);

    // Calculate delta.
    vbool32_t vMask = __riscv_vmseq_vx_u16mf2_b32(__riscv_vand_vx_u16mf2(vQh, 0x8000, 8), 0, 8);
    vint32m1_t vDeltaNeg = __riscv_vmv_v_x_i32m1(-1, 8);
    vint32m1_t vDeltaPos = __riscv_vmv_v_x_i32m1(1, 8);
    vint32m1_t vDelta = __riscv_vmerge_vvm_i32m1(vDeltaNeg, vDeltaPos, vMask, 8);

    // Load qs.
    vuint8m1_t vQs = __riscv_vle8_v_u8m1(vX[vI].qs, 32);

    // Prepare the indices.
    const uint64_t cShift = 0x0009000600030000;
    vuint16m2_t vQhShift = __riscv_vreinterpret_v_u64m2_u16m2(__riscv_vmv_v_x_u64m2(cShift, 8));
    vuint16m2_t vQhGatherIndex = __riscv_vreinterpret_v_i16m2_u16m2(
        __riscv_vdiv_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(32)), 4, 32));
    vuint16m2_t vQhExt = __riscv_vlmul_ext_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16mf2_u16m1(vQh));
    vuint16m2_t vQhIndex = __riscv_vrgather_vv_u16m2(vQhExt, vQhGatherIndex, 32);
    vQhIndex = __riscv_vsrl_vv_u16m2(vQhIndex, vQhShift, 32);
    vQhIndex = __riscv_vand_vx_u16m2(vQhIndex, 7, 32);
    vQhIndex = __riscv_vsll_vx_u16m2(vQhIndex, 8, 32);
    vQhIndex = __riscv_vor_vv_u16m2(vQhIndex, __riscv_vzext_vf2_u16m2(vQs, 32), 32);
    vuint16m2_t vIndex = __riscv_vsll_vx_u16m2(vQhIndex, 3, 32);

    // Final lsums.
    int32_t aLsumsS[8];
    vint32m1_t vOneScalar = __riscv_vmv_v_x_i32m1(0, 1);

    // Sub-blocks 1-4
    {
      vuint16m1_t vGridIndex0 = __riscv_vget_v_u16m2_u16m1(vIndex, 0);
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex0, 16));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(vY[vI].qs, 128);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 128);
      aLsumsS[0] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum0, 0), vOneScalar, 32));
      aLsumsS[1] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum0, 1), vOneScalar, 32));
      aLsumsS[2] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum0, 2), vOneScalar, 32));
      aLsumsS[3] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum0, 3), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    // Sub-blocks 5-8
    {
      vuint16m1_t vGridIndex1 = __riscv_vget_v_u16m2_u16m1(vIndex, 1);
      vint8m4_t vGrid1 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vGridIndex1, 16));
      vint8m4_t vQ81 = __riscv_vle8_v_i8m4(&vY[vI].qs[128], 128);
      vint16m8_t vLsum1 = __riscv_vwmul_vv_i16m8(vGrid1, vQ81, 128);
      aLsumsS[4] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum1, 0), vOneScalar, 32));
      aLsumsS[5] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum1, 1), vOneScalar, 32));
      aLsumsS[6] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum1, 2), vOneScalar, 32));
      aLsumsS[7] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vLsum1, 3), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    vint32m1_t vLsums = __riscv_vle32_v_i32m1(&aLsumsS[0], 8);

    // Calculate the bsums.
    vint16m1_t vBsums0 = __riscv_vle16_v_i16m1(vY[vI].bsums, 16);
    const vuint32m1_t cBsumsI32 = __riscv_vreinterpret_v_u16m1_u32m1(__riscv_vreinterpret_v_i16m1_u16m1(vBsums0));
    const vint16mf2_t cBsumsI320 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cBsumsI32, 0, 8));
    const vint16mf2_t cBsumsI321 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cBsumsI32, 16, 8));
    const vint32m1_t cBsums = __riscv_vwadd_vv_i32m1(cBsumsI320, cBsumsI321, 8);

    // Accumulation.
    vint32m1_t vSumiV = __riscv_vmul_vv_i32m1(vLs, vLsums, 8);
    vint32m1_t vSumi1V = __riscv_vmul_vv_i32m1(__riscv_vmul_vv_i32m1(vLs, vDelta, 8), cBsums, 8);

    // Update sumf.
    int vSumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(vSumiV, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(vSumi1V, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    vSumf += GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d * (vSumi + IQ1S_DELTA * vSumi1);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1SQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    // Load qh once for the entire superblock.
    vuint16mf4_t vQh = __riscv_vle16_v_u16mf4(vX[vI].qh, 8);

    // Calculate ls.
    vuint16mf4_t vTemp = __riscv_vsrl_vx_u16mf4(vQh, 12, 8);
    vTemp = __riscv_vand_vx_u16mf4(vTemp, 7, 8);
    vint32mf2_t vLs = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vwmulu_vx_u32mf2(vTemp, 2, 8));
    vLs = __riscv_vadd_vx_i32mf2(vLs, 1, 8);

    // Calculate delta.
    vbool64_t vMask = __riscv_vmseq_vx_u16mf4_b64(__riscv_vand_vx_u16mf4(vQh, 0x8000, 8), 0, 8);
    vint32mf2_t vDeltaNeg = __riscv_vmv_v_x_i32mf2(-1, 8);
    vint32mf2_t vDeltaPos = __riscv_vmv_v_x_i32mf2(1, 8);
    vint32mf2_t vDelta = __riscv_vmerge_vvm_i32mf2(vDeltaNeg, vDeltaPos, vMask, 8);

    // Load qs.
    vuint8mf2_t vQs = __riscv_vle8_v_u8mf2(vX[vI].qs, 32);

    // Prepare the indices.
    const uint64_t cShift = 0x0009000600030000;
    vuint16m1_t vQhShift = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vmv_v_x_u64m1(cShift, 8));
    vuint16m1_t vQhGatherIndex = __riscv_vreinterpret_v_i16m1_u16m1(
        __riscv_vdiv_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(32)), 4, 32));
    vuint16m1_t vQhExt = __riscv_vlmul_ext_v_u16mf2_u16m1(__riscv_vlmul_ext_v_u16mf4_u16mf2(vQh));
    vuint16m1_t vQhIndex = __riscv_vrgather_vv_u16m1(vQhExt, vQhGatherIndex, 32);
    vQhIndex = __riscv_vsrl_vv_u16m1(vQhIndex, vQhShift, 32);
    vQhIndex = __riscv_vand_vx_u16m1(vQhIndex, 7, 32);
    vQhIndex = __riscv_vsll_vx_u16m1(vQhIndex, 8, 32);
    vQhIndex = __riscv_vor_vv_u16m1(vQhIndex, __riscv_vzext_vf2_u16m1(vQs, 32), 32);
    vuint16m1_t vIndex = __riscv_vsll_vx_u16m1(vQhIndex, 3, 32);

    // Final lsums.
    int32_t aLsumsS[8];
    vint32m1_t vOneScalar = __riscv_vmv_v_x_i32m1(0, 1);

    // Sub-blocks 1-8
    {
      vint8m4_t vGrid0 =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t *)iq1s_grid, vIndex, 32));
      vint8m4_t vQ80 = __riscv_vle8_v_i8m4(vY[vI].qs, 256);
      vint16m8_t vLsum0 = __riscv_vwmul_vv_i16m8(vGrid0, vQ80, 256);
      aLsumsS[0] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 0), vOneScalar, 32));
      aLsumsS[1] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 1), vOneScalar, 32));
      aLsumsS[2] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 2), vOneScalar, 32));
      aLsumsS[3] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 3), vOneScalar, 32));
      aLsumsS[4] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 4), vOneScalar, 32));
      aLsumsS[5] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 5), vOneScalar, 32));
      aLsumsS[6] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 6), vOneScalar, 32));
      aLsumsS[7] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vLsum0, 7), vOneScalar, 32));
    }
    __asm__ __volatile__("" ::: "memory");
    vint32mf2_t vLsums = __riscv_vle32_v_i32mf2(&aLsumsS[0], 8);

    // Calculate the bsums.
    vint16mf2_t vBsums0 = __riscv_vle16_v_i16mf2(vY[vI].bsums, 16);
    const vuint32mf2_t cBsumsI32 = __riscv_vreinterpret_v_u16mf2_u32mf2(__riscv_vreinterpret_v_i16mf2_u16mf2(vBsums0));
    const vint16mf4_t cBsumsI320 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(cBsumsI32, 0, 8));
    const vint16mf4_t cBsumsI321 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(cBsumsI32, 16, 8));
    const vint32mf2_t cBsums = __riscv_vwadd_vv_i32mf2(cBsumsI320, cBsumsI321, 8);

    // Accumulation.
    vint32mf2_t vSumiV = __riscv_vmul_vv_i32mf2(vLs, vLsums, 8);
    vint32mf2_t vSumi1V = __riscv_vmul_vv_i32mf2(__riscv_vmul_vv_i32mf2(vLs, vDelta, 8), cBsums, 8);

    // Update sumf.
    int vSumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(vSumiV, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(vSumi1V, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    vSumf += GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d * (vSumi + IQ1S_DELTA * vSumi1);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1SQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  // Mask for processing 32 elements per lsum register.
  vuint16m1_t vLIndex = __riscv_vid_v_u16m1(64);
  vbool16_t vLMask = __riscv_vmsgtu_vx_u16m1_b16(vLIndex, 31, 64);

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    // Load qh once for the entire superblock.
    vuint16mf4_t vQh = __riscv_vle16_v_u16mf4(vX[vI].qh, 8);

    // Calculate ls.
    vuint16mf4_t vTemp = __riscv_vsrl_vx_u16mf4(vQh, 12, 8);
    vTemp = __riscv_vand_vx_u16mf4(vTemp, 7, 8);
    vint32mf2_t vLs = __riscv_vreinterpret_v_u32mf2_i32mf2(__riscv_vwmulu_vx_u32mf2(vTemp, 2, 8));
    vLs = __riscv_vadd_vx_i32mf2(vLs, 1, 8);

    // Calculate delta.
    vbool64_t vMask = __riscv_vmseq_vx_u16mf4_b64(__riscv_vand_vx_u16mf4(vQh, 0x8000, 8), 0, 8);
    vint32mf2_t vDeltaNeg = __riscv_vmv_v_x_i32mf2(-1, 8);
    vint32mf2_t vDeltaPos = __riscv_vmv_v_x_i32mf2(1, 8);
    vint32mf2_t vDelta = __riscv_vmerge_vvm_i32mf2(vDeltaNeg, vDeltaPos, vMask, 8);

    // Load qs.
    vuint8mf2_t vQs = __riscv_vle8_v_u8mf2(vX[vI].qs, 32);

    // Prepare the indices.
    const uint64_t cShift = 0x0009000600030000;
    vuint16m1_t vQhShift = __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vmv_v_x_u64m1(cShift, 8));
    vuint16m1_t vQhGatherIndex = __riscv_vreinterpret_v_i16m1_u16m1(
        __riscv_vdiv_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vid_v_u16m1(32)), 4, 32));
    vuint16m1_t vQhExt = __riscv_vlmul_ext_v_u16mf2_u16m1(__riscv_vlmul_ext_v_u16mf4_u16mf2(vQh));
    vuint16m1_t vQhIndex = __riscv_vrgather_vv_u16m1(vQhExt, vQhGatherIndex, 32);
    vQhIndex = __riscv_vsrl_vv_u16m1(vQhIndex, vQhShift, 32);
    vQhIndex = __riscv_vand_vx_u16m1(vQhIndex, 7, 32);
    vQhIndex = __riscv_vsll_vx_u16m1(vQhIndex, 8, 32);
    vQhIndex = __riscv_vor_vv_u16m1(vQhIndex, __riscv_vzext_vf2_u16m1(vQs, 32), 32);
    vuint16mf2_t vIndex = __riscv_vlmul_trunc_v_u16m1_u16mf2(__riscv_vsll_vx_u16m1(vQhIndex, 3, 32));

    // Final lsums.
    int32_t aLsumsS[8];
    vint32m1_t vOneScalar = __riscv_vmv_v_x_i32m1(0, 1);

    // Sub-blocks 1-8
    {
      vint8m2_t vGrid0 =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vluxei16_v_i64m2((const int64_t *)iq1s_grid, vIndex, 32));
      vint8m2_t vQ80 = __riscv_vle8_v_i8m2(vY[vI].qs, 256);
      vint16m4_t vLsum0 = __riscv_vwmul_vv_i16m4(vGrid0, vQ80, 256);

      // Reduce.
      aLsumsS[0] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vLsum0, 0), vOneScalar, 32));
      aLsumsS[1] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1_m(vLMask, __riscv_vget_v_i16m4_i16m1(vLsum0, 0), vOneScalar, 64));
      aLsumsS[2] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vLsum0, 1), vOneScalar, 32));
      aLsumsS[3] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1_m(vLMask, __riscv_vget_v_i16m4_i16m1(vLsum0, 1), vOneScalar, 64));
      aLsumsS[4] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vLsum0, 2), vOneScalar, 32));
      aLsumsS[5] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1_m(vLMask, __riscv_vget_v_i16m4_i16m1(vLsum0, 2), vOneScalar, 64));
      aLsumsS[6] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vLsum0, 3), vOneScalar, 32));
      aLsumsS[7] = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1_m(vLMask, __riscv_vget_v_i16m4_i16m1(vLsum0, 3), vOneScalar, 64));
    }
    __asm__ __volatile__("" ::: "memory");
    vint32mf2_t vLsums = __riscv_vle32_v_i32mf2(&aLsumsS[0], 8);

    // Calculate the bsums.
    vint16mf2_t vBsums0 = __riscv_vle16_v_i16mf2(vY[vI].bsums, 16);
    const vuint32mf2_t cBsumsI32 = __riscv_vreinterpret_v_u16mf2_u32mf2(__riscv_vreinterpret_v_i16mf2_u16mf2(vBsums0));
    const vint16mf4_t cBsumsI320 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(cBsumsI32, 0, 8));
    const vint16mf4_t cBsumsI321 = __riscv_vreinterpret_v_u16mf4_i16mf4(__riscv_vnsrl_wx_u16mf4(cBsumsI32, 16, 8));
    const vint32mf2_t cBsums = __riscv_vwadd_vv_i32mf2(cBsumsI320, cBsumsI321, 8);

    // Accumulation.
    vint32mf2_t vSumiV = __riscv_vmul_vv_i32mf2(vLs, vLsums, 8);
    vint32mf2_t vSumi1V = __riscv_vmul_vv_i32mf2(__riscv_vmul_vv_i32mf2(vLs, vDelta, 8), cBsums, 8);

    // Update sumf.
    int vSumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(vSumiV, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32mf2_i32m1(vSumi1V, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
    vSumf += GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d * (vSumi + IQ1S_DELTA * vSumi1);
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_iq1_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq1SQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq1SQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotIq1SQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 1024:
    fGgmlVecDotIq1SQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    ggml_vec_dot_iq1_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq1_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq1MQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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
  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    // Accumulators.
    vint32m4_t vAcc1 = __riscv_vmv_v_x_i32m4(0, 16);
    vint32m4_t vAcc2 = __riscv_vmv_v_x_i32m4(0, 16);

// We process 8 16-element sub-blocks together.
#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 128; vIb++) {
      // Load qh for 8 sub-blocks.
      const vuint8mf2_t cQh8 = __riscv_vle8_v_u8mf2(vQh, 8);
      const vuint16m1_t cQh16Lo = __riscv_vzext_vf2_u16m1(cQh8, 8);
      const vuint16m1_t cQh16Hi = __riscv_vsll_vx_u16m1(cQh16Lo, 8, 8);
      const vuint16m2_t cQhb =
          __riscv_vzext_vf2_u16m2(__riscv_vreinterpret_v_u16m1_u8m1(__riscv_vor_vv_u16m1(cQh16Lo, cQh16Hi, 8)), 16);
      vQh += 8;

      // Prepare grid indices.
      const vuint16m2_t cQsb = __riscv_vzext_vf2_u16m2(__riscv_vle8_v_u8m1(&vQs[0], 16), 16);
      const vuint16m2_t cShift = __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00040008, 8));
      vuint16m2_t vIndex =
          __riscv_vor_vv_u16m2(cQsb, __riscv_vand_vx_u16m2(__riscv_vsll_vv_u16m2(cQhb, cShift, 16), 0x700, 16), 16);
      vIndex = __riscv_vsll_vx_u16m2(vIndex, 3, 16);
      vQs += 16;

      // Prepare the deltas.
      const vbool8_t cMask = __riscv_vmsgtu_vx_u16m2_b8(
          __riscv_vand_vv_u16m2(cQhb, __riscv_vreinterpret_v_u32m2_u16m2(__riscv_vmv_v_x_u32m2(0x00800008, 8)), 16), 0,
          16);
      const vint64m8_t cDeltaPos = __riscv_vmv_v_x_i64m8(0x0101010101010101, 16);
      const vint8m8_t cDelta =
          __riscv_vreinterpret_v_i64m8_i8m8(__riscv_vmerge_vxm_i64m8(cDeltaPos, 0xffffffffffffffff, cMask, 16));

      // Sub-blocks 0-3
      {
        // Load the grid.
        const vint8m4_t cIq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
            __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(vIndex, 0), 8)));

        // Calculate the lsums.
        //
        // Sub-block 0, 1
        {
          // Load q8 for each sub-block.
          const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 32);
          vQ8 += 32;

          // Calculate the lsums.
          const vint16m4_t cLsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(cIq1b, 0), cQ8b, 32);
          const vint16m4_t cLsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(cDelta, 0), cQ8b, 32);

          // Prepare the scales.
          const int16_t cLs0 = 2 * ((vSc[0] >> 0) & 0x7) + 1;
          const int16_t cLs1 = 2 * ((vSc[0] >> 3) & 0x7) + 1;

          // Accumulate in acc0 and acc1 for each sub-block.
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum1, 0), 16);
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum1, 1), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum2, 0), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum2, 1), 16);
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-block 2, 3
        {
          // Load q8 for each sub-block.
          const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 32);
          vQ8 += 32;

          // Calculate the lsums.
          const vint16m4_t cLsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(cIq1b, 1), cQ8b, 32);
          const vint16m4_t cLsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(cDelta, 1), cQ8b, 32);

          // Prepare the scales.
          const int16_t cLs0 = 2 * ((vSc[0] >> 6) & 0x7) + 1;
          const int16_t cLs1 = 2 * ((vSc[0] >> 9) & 0x7) + 1;

          // Accumulate in acc0 and acc1 for each sub-block.
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum1, 0), 16);
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum1, 1), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum2, 0), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum2, 1), 16);
        }
        vSc += 1;
      }
      __asm__ __volatile__("" ::: "memory");
      // Sub-blocks 4-7
      {
        // Load the grid.
        const vint8m4_t cIq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
            __riscv_vluxei16_v_u64m4(iq1s_grid, __riscv_vget_v_u16m2_u16m1(vIndex, 1), 8)));

        // Calculate the lsums.
        //
        // Sub-block 4, 5
        {
          // Load q8 for each sub-block.
          const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 32);
          vQ8 += 32;

          // Calculate the lsums.
          const vint16m4_t cLsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(cIq1b, 0), cQ8b, 32);
          const vint16m4_t cLsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(cDelta, 2), cQ8b, 32);

          // Prepare the scales.
          const int16_t cLs0 = 2 * ((vSc[0] >> 0) & 0x7) + 1;
          const int16_t cLs1 = 2 * ((vSc[0] >> 3) & 0x7) + 1;

          // Accumulate in acc0 and acc1 for each sub-block.
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum1, 0), 16);
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum1, 1), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum2, 0), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum2, 1), 16);
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-block 6, 7
        {
          // Load q8 for each sub-block.
          const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 32);
          vQ8 += 32;

          // Calculate the lsums.
          const vint16m4_t cLsum1 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m4_i8m2(cIq1b, 1), cQ8b, 32);
          const vint16m4_t cLsum2 = __riscv_vwmul_vv_i16m4(__riscv_vget_v_i8m8_i8m2(cDelta, 3), cQ8b, 32);

          // Prepare the scales.
          const int16_t cLs0 = 2 * ((vSc[0] >> 6) & 0x7) + 1;
          const int16_t cLs1 = 2 * ((vSc[0] >> 9) & 0x7) + 1;

          // Accumulate in acc0 and acc1 for each sub-block.
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum1, 0), 16);
          vAcc1 = __riscv_vwmacc_vx_i32m4(vAcc1, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum1, 1), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs0, __riscv_vget_v_i16m4_i16m2(cLsum2, 0), 16);
          vAcc2 = __riscv_vwmacc_vx_i32m4(vAcc2, cLs1, __riscv_vget_v_i16m4_i16m2(cLsum2, 1), 16);
        }
        vSc += 1;
      }
    }

    // Reduce and accumulate in `sumf`.
    vint32m1_t vOne = __riscv_vmv_v_x_i32m1(0, 1);
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(vAcc1, vOne, 16));
    int vSumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(vAcc2, vOne, 16));
    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16) * (vSumi1 + IQ1M_DELTA * vSumi2);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1MQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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
  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    // Accumulators.
    vint32m2_t vAcc1 = __riscv_vmv_v_x_i32m2(0, 16);
    vint32m2_t vAcc2 = __riscv_vmv_v_x_i32m2(0, 16);

// We process 8 16-element sub-blocks together.
#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 128; vIb++) {
      // Load qh for 8 sub-blocks.
      const vuint8mf4_t cQh8 = __riscv_vle8_v_u8mf4(vQh, 8);
      const vuint16mf2_t cQh16Lo = __riscv_vzext_vf2_u16mf2(cQh8, 8);
      const vuint16mf2_t cQh16Hi = __riscv_vsll_vx_u16mf2(cQh16Lo, 8, 8);
      const vuint16m1_t cQhb =
          __riscv_vzext_vf2_u16m1(__riscv_vreinterpret_v_u16mf2_u8mf2(__riscv_vor_vv_u16mf2(cQh16Lo, cQh16Hi, 8)), 16);
      vQh += 8;

      __asm__ __volatile__("" ::: "memory");

      // Prepare grid indices.
      const vuint16m1_t cQsb = __riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(&vQs[0], 16), 16);
      const vuint16m1_t cShift = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00040008, 8));
      vuint16m1_t vIndex =
          __riscv_vor_vv_u16m1(cQsb, __riscv_vand_vx_u16m1(__riscv_vsll_vv_u16m1(cQhb, cShift, 16), 0x700, 16), 16);
      vIndex = __riscv_vsll_vx_u16m1(vIndex, 3, 16);
      vQs += 16;

      __asm__ __volatile__("" ::: "memory");

      // Load the grid.
      const vint8m4_t cIq1b = __riscv_vreinterpret_v_i64m4_i8m4(
          __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vluxei16_v_u64m4(iq1s_grid, vIndex, 16)));

      // Prepare the deltas.
      const vbool16_t cMask = __riscv_vmsgtu_vx_u16m1_b16(
          __riscv_vand_vv_u16m1(cQhb, __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00800008, 8)), 16), 0,
          16);
      const vint64m4_t cDeltaPos = __riscv_vmv_v_x_i64m4(0x0101010101010101, 16);
      const vint8m4_t cDelta =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vmerge_vxm_i64m4(cDeltaPos, 0xffffffffffffffff, cMask, 16));

      // Load q8 for sub-blocks.
      const vint8m4_t cQ8b = __riscv_vle8_v_i8m4(vQ8, 128);
      vQ8 += 128;

      // Calculate the lsums.
      const vint16m8_t cLsum1 = __riscv_vwmul_vv_i16m8(cIq1b, cQ8b, 128);
      const vint16m8_t cLsum2 = __riscv_vwmul_vv_i16m8(cDelta, cQ8b, 128);

      // Prepare the scales.
      const int16_t cLs00 = 2 * ((vSc[0] >> 0) & 0x7) + 1;
      const int16_t cLs01 = 2 * ((vSc[0] >> 3) & 0x7) + 1;
      const int16_t cLs10 = 2 * ((vSc[0] >> 6) & 0x7) + 1;
      const int16_t cLs11 = 2 * ((vSc[0] >> 9) & 0x7) + 1;
      const int16_t cLs20 = 2 * ((vSc[1] >> 0) & 0x7) + 1;
      const int16_t cLs21 = 2 * ((vSc[1] >> 3) & 0x7) + 1;
      const int16_t cLs30 = 2 * ((vSc[1] >> 6) & 0x7) + 1;
      const int16_t cLs31 = 2 * ((vSc[1] >> 9) & 0x7) + 1;
      vSc += 2;

      // Accumulate in acc0 and acc1 for each sub-block.
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs00, __riscv_vget_v_i16m8_i16m1(cLsum1, 0), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs01, __riscv_vget_v_i16m8_i16m1(cLsum1, 1), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs00, __riscv_vget_v_i16m8_i16m1(cLsum2, 0), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs01, __riscv_vget_v_i16m8_i16m1(cLsum2, 1), 16);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs10, __riscv_vget_v_i16m8_i16m1(cLsum1, 2), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs11, __riscv_vget_v_i16m8_i16m1(cLsum1, 3), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs10, __riscv_vget_v_i16m8_i16m1(cLsum2, 2), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs11, __riscv_vget_v_i16m8_i16m1(cLsum2, 3), 16);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs20, __riscv_vget_v_i16m8_i16m1(cLsum1, 4), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs21, __riscv_vget_v_i16m8_i16m1(cLsum1, 5), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs20, __riscv_vget_v_i16m8_i16m1(cLsum2, 4), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs21, __riscv_vget_v_i16m8_i16m1(cLsum2, 5), 16);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs30, __riscv_vget_v_i16m8_i16m1(cLsum1, 6), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs31, __riscv_vget_v_i16m8_i16m1(cLsum1, 7), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs30, __riscv_vget_v_i16m8_i16m1(cLsum2, 6), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs31, __riscv_vget_v_i16m8_i16m1(cLsum2, 7), 16);

      __asm__ __volatile__("" ::: "memory");
    }

    // Reduce and accumulate in `sumf`.
    vint32m1_t vOne = __riscv_vmv_v_x_i32m1(0, 1);
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc1, vOne, 16));
    int vSumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc2, vOne, 16));
    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16) * (vSumi1 + IQ1M_DELTA * vSumi2);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1MQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  // Mask for processing 16 elements per lsum register.
  const vuint16m1_t cLIndex = __riscv_vid_v_u16m1(32);
  const vbool16_t cLMask = __riscv_vmsgtu_vx_u16m1_b16(cLIndex, 15, 32);

  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    // Accumulators.
    vint32m2_t vAcc1 = __riscv_vmv_v_x_i32m2(0, 32);
    vint32m2_t vAcc2 = __riscv_vmv_v_x_i32m2(0, 32);

// We process all the sub-blocks together.
#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 256; vIb++) {
      // Load qh for all 16 sub-blocks.
      const vuint8mf4_t cQh8 = __riscv_vle8_v_u8mf4(vQh, 16);
      const vuint16mf2_t cQh16Lo = __riscv_vzext_vf2_u16mf2(cQh8, 16);
      const vuint16mf2_t cQh16Hi = __riscv_vsll_vx_u16mf2(cQh16Lo, 8, 16);
      const vuint16m1_t cQhb =
          __riscv_vzext_vf2_u16m1(__riscv_vreinterpret_v_u16mf2_u8mf2(__riscv_vor_vv_u16mf2(cQh16Lo, cQh16Hi, 16)), 32);
      __asm__ __volatile__("" ::: "memory");

      // Prepare grid indices.
      const vuint16m1_t cQsb = __riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(&vQs[0], 32), 32);
      const vuint16m1_t cShift = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00040008, 16));
      vuint16m1_t vIndex =
          __riscv_vor_vv_u16m1(cQsb, __riscv_vand_vx_u16m1(__riscv_vsll_vv_u16m1(cQhb, cShift, 32), 0x700, 32), 32);
      vIndex = __riscv_vsll_vx_u16m1(vIndex, 3, 32);
      __asm__ __volatile__("" ::: "memory");

      // Load the grid.
      const vint8m4_t cIq1b = __riscv_vreinterpret_v_i64m4_i8m4(
          __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vluxei16_v_u64m4(iq1s_grid, vIndex, 32)));

      // Prepare the deltas.
      const vbool16_t cMask = __riscv_vmsgtu_vx_u16m1_b16(
          __riscv_vand_vv_u16m1(cQhb, __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00800008, 16)), 32), 0,
          32);
      const vint64m4_t cDeltaPos = __riscv_vmv_v_x_i64m4(0x0101010101010101, 32);
      const vint8m4_t cDelta =
          __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vmerge_vxm_i64m4(cDeltaPos, 0xffffffffffffffff, cMask, 32));

      // Load q8 for sub-blocks.
      const vint8m4_t cQ8b = __riscv_vle8_v_i8m4(vQ8, 256);

      // Calculate the lsums.
      const vint16m8_t cLsum1 = __riscv_vwmul_vv_i16m8(cIq1b, cQ8b, 256);
      const vint16m8_t cLsum2 = __riscv_vwmul_vv_i16m8(cDelta, cQ8b, 256);

      // Prepare the scales.
      const int16_t cLs0 = 2 * ((vSc[0] >> 0) & 0x7) + 1;
      const int16_t cLs1 = 2 * ((vSc[0] >> 3) & 0x7) + 1;
      const int16_t cLs2 = 2 * ((vSc[0] >> 6) & 0x7) + 1;
      const int16_t cLs3 = 2 * ((vSc[0] >> 9) & 0x7) + 1;
      const int16_t cLs4 = 2 * ((vSc[1] >> 0) & 0x7) + 1;
      const int16_t cLs5 = 2 * ((vSc[1] >> 3) & 0x7) + 1;
      const int16_t cLs6 = 2 * ((vSc[1] >> 6) & 0x7) + 1;
      const int16_t cLs7 = 2 * ((vSc[1] >> 9) & 0x7) + 1;
      const int16_t cLs8 = 2 * ((vSc[2] >> 0) & 0x7) + 1;
      const int16_t cLs9 = 2 * ((vSc[2] >> 3) & 0x7) + 1;
      const int16_t cLs10 = 2 * ((vSc[2] >> 6) & 0x7) + 1;
      const int16_t cLs11 = 2 * ((vSc[2] >> 9) & 0x7) + 1;
      const int16_t cLs12 = 2 * ((vSc[3] >> 0) & 0x7) + 1;
      const int16_t cLs13 = 2 * ((vSc[3] >> 3) & 0x7) + 1;
      const int16_t cLs14 = 2 * ((vSc[3] >> 6) & 0x7) + 1;
      const int16_t cLs15 = 2 * ((vSc[3] >> 9) & 0x7) + 1;

      // Accumulate in acc0 and acc1 for each sub-block.
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs0, __riscv_vget_v_i16m8_i16m1(cLsum1, 0), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs1, __riscv_vget_v_i16m8_i16m1(cLsum1, 0), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs0, __riscv_vget_v_i16m8_i16m1(cLsum2, 0), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs1, __riscv_vget_v_i16m8_i16m1(cLsum2, 0), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs2, __riscv_vget_v_i16m8_i16m1(cLsum1, 1), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs3, __riscv_vget_v_i16m8_i16m1(cLsum1, 1), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs2, __riscv_vget_v_i16m8_i16m1(cLsum2, 1), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs3, __riscv_vget_v_i16m8_i16m1(cLsum2, 1), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs4, __riscv_vget_v_i16m8_i16m1(cLsum1, 2), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs5, __riscv_vget_v_i16m8_i16m1(cLsum1, 2), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs4, __riscv_vget_v_i16m8_i16m1(cLsum2, 2), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs5, __riscv_vget_v_i16m8_i16m1(cLsum2, 2), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs6, __riscv_vget_v_i16m8_i16m1(cLsum1, 3), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs7, __riscv_vget_v_i16m8_i16m1(cLsum1, 3), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs6, __riscv_vget_v_i16m8_i16m1(cLsum2, 3), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs7, __riscv_vget_v_i16m8_i16m1(cLsum2, 3), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs8, __riscv_vget_v_i16m8_i16m1(cLsum1, 4), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs9, __riscv_vget_v_i16m8_i16m1(cLsum1, 4), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs8, __riscv_vget_v_i16m8_i16m1(cLsum2, 4), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs9, __riscv_vget_v_i16m8_i16m1(cLsum2, 4), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs10, __riscv_vget_v_i16m8_i16m1(cLsum1, 5), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs11, __riscv_vget_v_i16m8_i16m1(cLsum1, 5), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs10, __riscv_vget_v_i16m8_i16m1(cLsum2, 5), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs11, __riscv_vget_v_i16m8_i16m1(cLsum2, 5), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs12, __riscv_vget_v_i16m8_i16m1(cLsum1, 6), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs13, __riscv_vget_v_i16m8_i16m1(cLsum1, 6), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs12, __riscv_vget_v_i16m8_i16m1(cLsum2, 6), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs13, __riscv_vget_v_i16m8_i16m1(cLsum2, 6), 32);
      //
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs14, __riscv_vget_v_i16m8_i16m1(cLsum1, 7), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc1, cLs15, __riscv_vget_v_i16m8_i16m1(cLsum1, 7), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs14, __riscv_vget_v_i16m8_i16m1(cLsum2, 7), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask, vAcc2, cLs15, __riscv_vget_v_i16m8_i16m1(cLsum2, 7), 32);

      __asm__ __volatile__("" ::: "memory");
    }

    // Reduce and accumulate in `sumf`.
    vint32m1_t vOne = __riscv_vmv_v_x_i32m1(0, 1);
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc1, vOne, 32));
    int vSumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc2, vOne, 32));
    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16) * (vSumi1 + IQ1M_DELTA * vSumi2);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq1MQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
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
  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    // Accumulators.
    vint32m2_t vAcc1 = __riscv_vmv_v_x_i32m2(0, 64);
    vint32m2_t vAcc2 = __riscv_vmv_v_x_i32m2(0, 64);

// We process all the sub-blocks together.
#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 256; vIb++) {
      // Load qh for all 16 sub-blocks.
      const vuint8mf8_t cQh8 = __riscv_vle8_v_u8mf8(vQh, 16);
      const vuint16mf4_t cQh16Lo = __riscv_vzext_vf2_u16mf4(cQh8, 16);
      const vuint16mf4_t cQh16Hi = __riscv_vsll_vx_u16mf4(cQh16Lo, 8, 16);
      const vuint16mf2_t cQhb = __riscv_vzext_vf2_u16mf2(
          __riscv_vreinterpret_v_u16mf4_u8mf4(__riscv_vor_vv_u16mf4(cQh16Lo, cQh16Hi, 16)), 32);
      __asm__ __volatile__("" ::: "memory");

      // Prepare grid indices.
      const vuint16mf2_t cQsb = __riscv_vzext_vf2_u16mf2(__riscv_vle8_v_u8mf4(&vQs[0], 32), 32);
      const vuint16mf2_t cShift = __riscv_vreinterpret_v_u32mf2_u16mf2(__riscv_vmv_v_x_u32mf2(0x00040008, 16));
      vuint16mf2_t vIndex =
          __riscv_vor_vv_u16mf2(cQsb, __riscv_vand_vx_u16mf2(__riscv_vsll_vv_u16mf2(cQhb, cShift, 32), 0x700, 32), 32);
      vIndex = __riscv_vsll_vx_u16mf2(vIndex, 3, 32);
      __asm__ __volatile__("" ::: "memory");

      // Load the grid.
      const vint8m2_t cIq1b = __riscv_vreinterpret_v_i64m2_i8m2(
          __riscv_vreinterpret_v_u64m2_i64m2(__riscv_vluxei16_v_u64m2(iq1s_grid, vIndex, 32)));

      // Prepare the deltas.
      const vbool32_t cMask = __riscv_vmsgtu_vx_u16mf2_b32(
          __riscv_vand_vv_u16mf2(cQhb, __riscv_vreinterpret_v_u32mf2_u16mf2(__riscv_vmv_v_x_u32mf2(0x00800008, 16)),
                                 32),
          0, 32);
      const vint64m2_t cDeltaPos = __riscv_vmv_v_x_i64m2(0x0101010101010101, 32);
      const vint8m2_t cDelta =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmerge_vxm_i64m2(cDeltaPos, 0xffffffffffffffff, cMask, 32));

      // Load q8 for sub-blocks.
      const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 256);

      // Calculate the lsums.
      const vint16m4_t cLsum1 = __riscv_vwmul_vv_i16m4(cIq1b, cQ8b, 256);
      const vint16m4_t cLsum2 = __riscv_vwmul_vv_i16m4(cDelta, cQ8b, 256);

      // Prepare the scales.
      const int16_t cLs0 = 2 * ((vSc[0] >> 0) & 0x7) + 1;
      const int16_t cLs1 = 2 * ((vSc[0] >> 3) & 0x7) + 1;
      const int16_t cLs2 = 2 * ((vSc[0] >> 6) & 0x7) + 1;
      const int16_t cLs3 = 2 * ((vSc[0] >> 9) & 0x7) + 1;
      const int16_t cLs4 = 2 * ((vSc[1] >> 0) & 0x7) + 1;
      const int16_t cLs5 = 2 * ((vSc[1] >> 3) & 0x7) + 1;
      const int16_t cLs6 = 2 * ((vSc[1] >> 6) & 0x7) + 1;
      const int16_t cLs7 = 2 * ((vSc[1] >> 9) & 0x7) + 1;
      const int16_t cLs8 = 2 * ((vSc[2] >> 0) & 0x7) + 1;
      const int16_t cLs9 = 2 * ((vSc[2] >> 3) & 0x7) + 1;
      const int16_t cLs10 = 2 * ((vSc[2] >> 6) & 0x7) + 1;
      const int16_t cLs11 = 2 * ((vSc[2] >> 9) & 0x7) + 1;
      const int16_t cLs12 = 2 * ((vSc[3] >> 0) & 0x7) + 1;
      const int16_t cLs13 = 2 * ((vSc[3] >> 3) & 0x7) + 1;
      const int16_t cLs14 = 2 * ((vSc[3] >> 6) & 0x7) + 1;
      const int16_t cLs15 = 2 * ((vSc[3] >> 9) & 0x7) + 1;

      // Mask for processing 16 elements per lsum register.
      const vuint16m1_t cLIndex = __riscv_vid_v_u16m1(64);

      // Accumulate in acc1 and acc2 for each sub-block.
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs0, __riscv_vget_v_i16m4_i16m1(cLsum1, 0), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs0, __riscv_vget_v_i16m4_i16m1(cLsum2, 0), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs4, __riscv_vget_v_i16m4_i16m1(cLsum1, 1), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs4, __riscv_vget_v_i16m4_i16m1(cLsum2, 1), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs8, __riscv_vget_v_i16m4_i16m1(cLsum1, 2), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs8, __riscv_vget_v_i16m4_i16m1(cLsum2, 2), 16);
      vAcc1 = __riscv_vwmacc_vx_i32m2(vAcc1, cLs12, __riscv_vget_v_i16m4_i16m1(cLsum1, 3), 16);
      vAcc2 = __riscv_vwmacc_vx_i32m2(vAcc2, cLs12, __riscv_vget_v_i16m4_i16m1(cLsum2, 3), 16);
      //
      const vbool16_t cLMask1632 = __riscv_vmsgtu_vx_u16m1_b16(cLIndex, 15, 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc1, cLs1, __riscv_vget_v_i16m4_i16m1(cLsum1, 0), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc2, cLs1, __riscv_vget_v_i16m4_i16m1(cLsum2, 0), 32);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc1, cLs5, __riscv_vget_v_i16m4_i16m1(cLsum1, 1), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc2, cLs5, __riscv_vget_v_i16m4_i16m1(cLsum2, 1), 32);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc1, cLs9, __riscv_vget_v_i16m4_i16m1(cLsum1, 2), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc2, cLs9, __riscv_vget_v_i16m4_i16m1(cLsum2, 2), 32);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc1, cLs13, __riscv_vget_v_i16m4_i16m1(cLsum1, 3), 32);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask1632, vAcc2, cLs13, __riscv_vget_v_i16m4_i16m1(cLsum2, 3), 32);
      //
      const vbool16_t cLMask3248 = __riscv_vmsgtu_vx_u16m1_b16(cLIndex, 31, 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc1, cLs2, __riscv_vget_v_i16m4_i16m1(cLsum1, 0), 48);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc2, cLs2, __riscv_vget_v_i16m4_i16m1(cLsum2, 0), 48);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc1, cLs6, __riscv_vget_v_i16m4_i16m1(cLsum1, 1), 48);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc2, cLs6, __riscv_vget_v_i16m4_i16m1(cLsum2, 1), 48);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc1, cLs10, __riscv_vget_v_i16m4_i16m1(cLsum1, 2), 48);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc2, cLs10, __riscv_vget_v_i16m4_i16m1(cLsum2, 2), 48);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc1, cLs14, __riscv_vget_v_i16m4_i16m1(cLsum1, 3), 48);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask3248, vAcc2, cLs14, __riscv_vget_v_i16m4_i16m1(cLsum2, 3), 48);
      //
      const vbool16_t cLMask4864 = __riscv_vmsgtu_vx_u16m1_b16(cLIndex, 47, 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc1, cLs3, __riscv_vget_v_i16m4_i16m1(cLsum1, 0), 64);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc2, cLs3, __riscv_vget_v_i16m4_i16m1(cLsum2, 0), 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc1, cLs7, __riscv_vget_v_i16m4_i16m1(cLsum1, 1), 64);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc2, cLs7, __riscv_vget_v_i16m4_i16m1(cLsum2, 1), 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc1, cLs11, __riscv_vget_v_i16m4_i16m1(cLsum1, 2), 64);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc2, cLs11, __riscv_vget_v_i16m4_i16m1(cLsum2, 2), 64);
      vAcc1 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc1, cLs15, __riscv_vget_v_i16m4_i16m1(cLsum1, 3), 64);
      vAcc2 = __riscv_vwmacc_vx_i32m2_m(cLMask4864, vAcc2, cLs15, __riscv_vget_v_i16m4_i16m1(cLsum2, 3), 64);

      __asm__ __volatile__("" ::: "memory");
    }

    // Reduce and accumulate in `sumf`.
    vint32m1_t vOne = __riscv_vmv_v_x_i32m1(0, 1);
    int vSumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc1, vOne, 64));
    int vSumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(vAcc2, vOne, 64));
    vSumf += vY[vI].d * GGML_CPU_FP16_TO_FP32(vScale.f16) * (vSumi1 + IQ1M_DELTA * vSumi2);
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_iq1_m_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq1MQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq1MQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotIq1MQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 1024:
    fGgmlVecDotIq1MQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    ggml_vec_dot_iq1_m_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq1_m_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static const uint8_t sign_gather_indices_arr[64] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                    2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                                                    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};

static const uint8_t sign_bit_masks_arr[64] = {
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

static NOINLINE void fGgmlVecDotIq2SQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vGrid64 = (const uint64_t *)iq2s_grid;

  // Pre-load Constants
  vuint8m2_t vVIds = __riscv_vid_v_u8m2(32);
  vuint8m2_t vVSignGatherIndices = __riscv_vsrl_vx_u8m2(vVIds, 3, 32);
  vuint8m2_t vVOnes = __riscv_vmv_v_x_u8m2(1, 32);
  vuint8m2_t vVShiftAmts = __riscv_vand_vx_u8m2(vVIds, 7, 32);
  vuint8m2_t vVSignMasks = __riscv_vsll_vv_u8m2(vVOnes, vVShiftAmts, 32);
  uint16_t aShiftQhArr[4] = {11, 9, 7, 5};
  vuint16mf2_t vVShiftQh = __riscv_vle16_v_u16mf2(aShiftQhArr, 4);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const uint8_t *vSignsPtr = vQs + 32;
    float vSumBlock = 0.0f;

    for (int vIb = 0; vIb < 8; ++vIb) {

      // Load Low Bits [4 bytes]
      vuint8mf4_t vVQsU8 = __riscv_vle8_v_u8mf4(vQs, 4);
      vQs += 4;

      // Load 1 byte. It contains bits for 4 mini-blocks.
      uint8_t vQhVal = *vQh++;

      // Combine Low + High bits of 10bit indices
      vuint8mf4_t vVQhRaw = __riscv_vmv_v_x_u8mf4(vQhVal, 4);
      vuint16mf2_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16mf2(vVQhRaw, 4);
      vuint16mf2_t vVQhMf2 = __riscv_vsll_vv_u16mf2(vVQhU16, vVShiftQh, 4);
      vVQhMf2 = __riscv_vand_vx_u16mf2(vVQhMf2, 0x1800, 4);
      vuint16mf2_t vVQsU16Mf2 = __riscv_vwcvtu_x_x_v_u16mf2(vVQsU8, 4);
      vuint16mf2_t vVQsU16 = __riscv_vsll_vx_u16mf2(vVQsU16Mf2, 3, 4);
      vuint16mf2_t vVGridOffsets = __riscv_vor_vv_u16mf2(vVQsU16, vVQhMf2, 4);

      // Lookup Grid
      vint8m2_t vVGridI8 = __riscv_vreinterpret_v_u8m2_i8m2(
          __riscv_vreinterpret_v_u64m2_u8m2(__riscv_vluxei16_v_u64m2(vGrid64, vVGridOffsets, 4)));

      vuint8mf4_t vVSignsRaw = __riscv_vle8_v_u8mf4(vSignsPtr, 4);
      vSignsPtr += 4;
      vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf4_u8m2(vVSignsRaw);
      vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 32);

      // generating sign mask
      vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 32);
      vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 32);

      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 32);
      vQ8 += 32;

      // apply signs
      vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 32);
      vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVGridI8, vVQ8Signed, 32);

      // Reduction
      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);

      // Reduce 0-15 (First Half)
      int32_t vS0 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m4_i16m2(vVDot, 0), vVZero, 16));

      // Reduce 16-31 (Second Half)
      int32_t vS1 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m4_i16m2(vVDot, 1), vVZero, 16));

      // Apply sub Scales
      uint8_t vSc = *vScales++;

      vSumBlock += vS0 * (2 * (vSc & 0xF) + 1);
      vSumBlock += vS1 * (2 * (vSc >> 4) + 1);
    }
    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2SQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vGrid64 = (const uint64_t *)iq2s_grid;

  // --- Pre-load Constants ---
  uint16_t aGatherQhArr[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  vuint16mf2_t vVGatherQh = __riscv_vle16_v_u16mf2(aGatherQhArr, 8);
  uint16_t aShiftQhArr[8] = {11, 9, 7, 5, 11, 9, 7, 5};
  vuint16mf2_t vVShiftQh = __riscv_vle16_v_u16mf2(aShiftQhArr, 8);

  // Constants for sign extraction
  vuint8m2_t vVSignGatherIndices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
  vuint8m2_t vVSignMasks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const uint8_t *vSignsPtr = vQs + 32;

    float vSumBlock = 0.0f;

    for (int vIb = 0; vIb < 4; ++vIb) {
      // Combine low + high bits
      vuint8mf4_t vVQsU8 = __riscv_vle8_v_u8mf4(vQs, 8);
      vQs += 8;
      uint16_t vQhVal;
      memcpy(&vQhVal, vQh, 2);
      vQh += 2;
      vuint8mf8_t vVQhRaw = __riscv_vle8_v_u8mf8((const uint8_t *)&vQhVal, 2);
      vuint16mf4_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16mf4(vVQhRaw, 2);
      vuint16mf2_t vVQhU16Ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(vVQhU16);
      vuint16mf2_t vVQhExpanded = __riscv_vrgather_vv_u16mf2(vVQhU16Ext, vVGatherQh, 8);
      vVQhExpanded = __riscv_vsll_vv_u16mf2(vVQhExpanded, vVShiftQh, 8);

      // Mask: We want bits 11-12. 0x1800 = 0001 1000 0000 0000
      vVQhExpanded = __riscv_vand_vx_u16mf2(vVQhExpanded, 0x1800, 8);
      vuint16mf2_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16mf2(vVQsU8, 8);

      // Multiply by 8 to get byte offset, instead of element offset
      vVQsU16 = __riscv_vsll_vx_u16mf2(vVQsU16, 3, 8);
      vuint16mf2_t vVGridOffsets = __riscv_vor_vv_u16mf2(vVQsU16, vVQhExpanded, 8);

      // Lookup Grid using Byte Offsets
      vuint64m2_t vVGridVals = __riscv_vluxei16_v_u64m2(vGrid64, vVGridOffsets, 8);

      vuint8m2_t vVGridU8 = __riscv_vreinterpret_v_u64m2_u8m2(vVGridVals);
      vint8m2_t vVGridI8 = __riscv_vreinterpret_v_u8m2_i8m2(vVGridU8);

      // Load signs and generate sign mask
      vuint8mf4_t vVSignsRaw = __riscv_vle8_v_u8mf4(vSignsPtr, 8);
      vSignsPtr += 8;

      vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf4_u8m2(vVSignsRaw);
      vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 64);

      vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 64);
      vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 64);

      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 64);
      vQ8 += 64;

      vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 64);
      vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVGridI8, vVQ8Signed, 64);

      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);

      int32_t vS0 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vVDot, 0), vVZero, 16));
      int32_t vS1 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vVDot, 1), vVZero, 16));
      int32_t vS2 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vVDot, 2), vVZero, 16));
      int32_t vS3 =
          __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vVDot, 3), vVZero, 16));

      uint8_t vSc0 = vScales[0];
      uint8_t vSc1 = vScales[1];
      vScales += 2;

      vSumBlock += vS0 * (2 * (vSc0 & 0xF) + 1);
      vSumBlock += vS1 * (2 * (vSc0 >> 4) + 1);
      vSumBlock += vS2 * (2 * (vSc1 & 0xF) + 1);
      vSumBlock += vS3 * (2 * (vSc1 >> 4) + 1);
    }
    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2SQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vGrid64 = (const uint64_t *)iq2s_grid;

  vuint8m2_t vVIds = __riscv_vid_v_u8m2(128);
  vuint8m2_t vVSignGatherIndices = __riscv_vsrl_vx_u8m2(vVIds, 3, 128);

  vuint8m2_t vVOnes = __riscv_vmv_v_x_u8m2(1, 128);
  vuint8m2_t vVShiftAmts = __riscv_vand_vx_u8m2(vVIds, 7, 128);
  vuint8m2_t vVSignMasks = __riscv_vsll_vv_u8m2(vVOnes, vVShiftAmts, 128);

  uint16_t aGatherQhArr[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  vuint16mf2_t vVGatherQh = __riscv_vle16_v_u16mf2(aGatherQhArr, 16);

  uint16_t aShiftQhArr[16] = {11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5};
  vuint16mf2_t vVShiftQh = __riscv_vle16_v_u16mf2(aShiftQhArr, 16);

  // Masks for selecting lower/upper 16 lanes within a 32-lane i16m1 register
  vuint16m1_t vVIds16 = __riscv_vid_v_u16m1(32);
  vbool16_t vMHi16 = __riscv_vmsgeu_vx_u16m1_b16(vVIds16, 16, 32);
  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const uint8_t *vSignsPtr = vQs + 32;

    float vSumBlock = 0.0f;

    for (int vIb = 0; vIb < 2; ++vIb) {
      vuint8mf4_t vVQsU8 = __riscv_vle8_v_u8mf4(vQs, 16);
      vQs += 16;

      vuint8mf8_t vVQhRaw = __riscv_vle8_v_u8mf8(vQh, 4);
      vQh += 4;

      vuint16mf4_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16mf4(vVQhRaw, 4);
      vuint16mf2_t vVQhU16Ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(vVQhU16);
      vuint16mf2_t vVQhExpanded = __riscv_vrgather_vv_u16mf2(vVQhU16Ext, vVGatherQh, 16);
      vVQhExpanded = __riscv_vsll_vv_u16mf2(vVQhExpanded, vVShiftQh, 16);
      vVQhExpanded = __riscv_vand_vx_u16mf2(vVQhExpanded, 0x1800, 16);

      vuint16mf2_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16mf2(vVQsU8, 16);
      vVQsU16 = __riscv_vsll_vx_u16mf2(vVQsU16, 3, 16);

      vuint16mf2_t vVGridOffsets = __riscv_vor_vv_u16mf2(vVQsU16, vVQhExpanded, 16);
      vuint64m2_t vVGridVals = __riscv_vluxei16_v_u64m2(vGrid64, vVGridOffsets, 16);
      vuint8m2_t vVGridU8 = __riscv_vreinterpret_v_u64m2_u8m2(vVGridVals);
      vint8m2_t vVGridI8 = __riscv_vreinterpret_v_u8m2_i8m2(vVGridU8);

      vuint8mf4_t vVSignsRaw = __riscv_vle8_v_u8mf4(vSignsPtr, 16);
      vSignsPtr += 16;

      vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf4_u8m2(vVSignsRaw);
      vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 128);
      vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 128);
      vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 128);
      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 128);
      vQ8 += 128;

      vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 128);
      vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVGridI8, vVQ8Signed, 128);

      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);
      vint16m1_t v0 = __riscv_vget_v_i16m4_i16m1(vVDot, 0);
      vint16m1_t v1 = __riscv_vget_v_i16m4_i16m1(vVDot, 1);
      vint16m1_t v2 = __riscv_vget_v_i16m4_i16m1(vVDot, 2);
      vint16m1_t v3 = __riscv_vget_v_i16m4_i16m1(vVDot, 3);

      int32_t vS0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v0, vVZero, 16));
      int32_t vS1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(vMHi16, v0, vVZero, 32));
      int32_t vS2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v1, vVZero, 16));
      int32_t vS3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(vMHi16, v1, vVZero, 32));
      int32_t vS4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v2, vVZero, 16));
      int32_t vS5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(vMHi16, v2, vVZero, 32));
      int32_t vS6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(v3, vVZero, 16));
      int32_t vS7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(vMHi16, v3, vVZero, 32));

      uint8_t vSc0 = vScales[0];
      uint8_t vSc1 = vScales[1];
      uint8_t vSc2 = vScales[2];
      uint8_t vSc3 = vScales[3];
      vScales += 4;

      vSumBlock += vS0 * (2 * (vSc0 & 0xF) + 1);
      vSumBlock += vS1 * (2 * (vSc0 >> 4) + 1);
      vSumBlock += vS2 * (2 * (vSc1 & 0xF) + 1);
      vSumBlock += vS3 * (2 * (vSc1 >> 4) + 1);
      vSumBlock += vS4 * (2 * (vSc2 & 0xF) + 1);
      vSumBlock += vS5 * (2 * (vSc2 >> 4) + 1);
      vSumBlock += vS6 * (2 * (vSc3 & 0xF) + 1);
      vSumBlock += vS7 * (2 * (vSc3 >> 4) + 1);
    }

    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2SQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vGrid64 = (const uint64_t *)iq2s_grid;
  vuint8m2_t vVIds = __riscv_vid_v_u8m2(256);
  vuint8m2_t vVSignGatherIndices = __riscv_vsrl_vx_u8m2(vVIds, 3, 256);

  vuint8m2_t vVOnes = __riscv_vmv_v_x_u8m2(1, 256);
  vuint8m2_t vVShiftAmts = __riscv_vand_vx_u8m2(vVIds, 7, 256);
  vuint8m2_t vVSignMasks = __riscv_vsll_vv_u8m2(vVOnes, vVShiftAmts, 256);

  uint16_t aGatherQhArr[32] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                               4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
  vuint16mf2_t vVGatherQh = __riscv_vle16_v_u16mf2(aGatherQhArr, 32);

  uint16_t aShiftQhArr[32] = {11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5,
                              11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5, 11, 9, 7, 5};
  vuint16mf2_t vVShiftQh = __riscv_vle16_v_u16mf2(aShiftQhArr, 32);

  // Masks for 4 groups of 16 lanes within a 64-lane i16m4 chunk
  vuint16m4_t vVIds64 = __riscv_vid_v_u16m4(64);
  vbool4_t vMG0 = __riscv_vmsltu_vx_u16m4_b4(vVIds64, 16, 64);
  vbool4_t vMG1 =
      __riscv_vmand_mm_b4(__riscv_vmsgeu_vx_u16m4_b4(vVIds64, 16, 64), __riscv_vmsltu_vx_u16m4_b4(vVIds64, 32, 64), 64);
  vbool4_t vMG2 =
      __riscv_vmand_mm_b4(__riscv_vmsgeu_vx_u16m4_b4(vVIds64, 32, 64), __riscv_vmsltu_vx_u16m4_b4(vVIds64, 48, 64), 64);
  vbool4_t vMG3 = __riscv_vmsgeu_vx_u16m4_b4(vVIds64, 48, 64);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const uint8_t *vSignsPtr = vQs + 32;

    float vSumBlock = 0.0f;

    vuint8mf4_t vVQsU8 = __riscv_vle8_v_u8mf4(vQs, 32);
    vQs += 32;

    vuint8mf8_t vVQhRaw = __riscv_vle8_v_u8mf8(vQh, 8);
    vQh += 8;

    vuint16mf4_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16mf4(vVQhRaw, 8);
    vuint16mf2_t vVQhU16Ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(vVQhU16);
    vuint16mf2_t vVQhExpanded = __riscv_vrgather_vv_u16mf2(vVQhU16Ext, vVGatherQh, 32);
    vVQhExpanded = __riscv_vsll_vv_u16mf2(vVQhExpanded, vVShiftQh, 32);
    vVQhExpanded = __riscv_vand_vx_u16mf2(vVQhExpanded, 0x1800, 32);

    vuint16mf2_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16mf2(vVQsU8, 32);
    vVQsU16 = __riscv_vsll_vx_u16mf2(vVQsU16, 3, 32);

    vuint16mf2_t vVGridOffsets = __riscv_vor_vv_u16mf2(vVQsU16, vVQhExpanded, 32);
    vuint64m2_t vVGridVals = __riscv_vluxei16_v_u64m2(vGrid64, vVGridOffsets, 32);
    vuint8m2_t vVGridU8 = __riscv_vreinterpret_v_u64m2_u8m2(vVGridVals);
    vint8m2_t vVGridI8 = __riscv_vreinterpret_v_u8m2_i8m2(vVGridU8);

    // loading signs
    vuint8mf2_t vVSignsRaw = __riscv_vle8_v_u8mf2(vSignsPtr, 32);
    vSignsPtr += 32;

    vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf2_u8m2(vVSignsRaw);
    vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 256);
    vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 256);
    vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 256);

    vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 256);
    vQ8 += 256;

    vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 256);
    vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVGridI8, vVQ8Signed, 256);

    vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);

    vint16m4_t vC = vVDot;

    int32_t vS0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG0, vC, vVZero, 64));
    int32_t vS1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG1, vC, vVZero, 64));
    int32_t vS2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG2, vC, vVZero, 64));
    int32_t vS3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG3, vC, vVZero, 64));

    vC = __riscv_vslidedown_vx_i16m4(vC, 64, 256);
    int32_t vS4 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG0, vC, vVZero, 64));
    int32_t vS5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG1, vC, vVZero, 64));
    int32_t vS6 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG2, vC, vVZero, 64));
    int32_t vS7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG3, vC, vVZero, 64));

    vC = __riscv_vslidedown_vx_i16m4(vC, 64, 256);
    int32_t vS8 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG0, vC, vVZero, 64));
    int32_t vS9 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG1, vC, vVZero, 64));
    int32_t vS10 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG2, vC, vVZero, 64));
    int32_t vS11 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG3, vC, vVZero, 64));

    vC = __riscv_vslidedown_vx_i16m4(vC, 64, 256);
    int32_t vS12 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG0, vC, vVZero, 64));
    int32_t vS13 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG1, vC, vVZero, 64));
    int32_t vS14 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG2, vC, vVZero, 64));
    int32_t vS15 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1_m(vMG3, vC, vVZero, 64));

    int32_t aSumsArr[16] = {vS0, vS1, vS2, vS3, vS4, vS5, vS6, vS7, vS8, vS9, vS10, vS11, vS12, vS13, vS14, vS15};

    // Load 8 scale bytes and split into 16 nibbles
    vuint8mf2_t vVSc8 = __riscv_vle8_v_u8mf2(vScales, 8);
    vScales += 8;

    vuint8mf2_t vVLo8 = __riscv_vand_vx_u8mf2(vVSc8, 0x0F, 8);
    vuint8mf2_t vVHi8 = __riscv_vsrl_vx_u8mf2(vVSc8, 4, 8);

    vuint8m1_t vVIdx16 = __riscv_vid_v_u8m1(16);
    vuint8m1_t vVHalf = __riscv_vsrl_vx_u8m1(vVIdx16, 1, 16);
    vbool8_t vMEven = __riscv_vmseq_vx_u8m1_b8(__riscv_vand_vx_u8m1(vVIdx16, 1, 16), 0, 16);

    vuint8m1_t vVLoExt = __riscv_vlmul_ext_v_u8mf2_u8m1(vVLo8);
    vuint8m1_t vVHiExt = __riscv_vlmul_ext_v_u8mf2_u8m1(vVHi8);
    vuint8m1_t vVLoG = __riscv_vrgather_vv_u8m1(vVLoExt, vVHalf, 16);
    vuint8m1_t vVHiG = __riscv_vrgather_vv_u8m1(vVHiExt, vVHalf, 16);
    vuint8m1_t vVNib = __riscv_vmerge_vvm_u8m1(vVLoG, vVHiG, vMEven, 16);

    static const uint8_t aIq2sScaleLut16Local[16] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    vuint8m1_t vVLut = __riscv_vle8_v_u8m1(aIq2sScaleLut16Local, 16);
    vuint8m1_t vVSc8v = __riscv_vrgather_vv_u8m1(vVLut, vVNib, 16);

    vint32m4_t vVSums = __riscv_vle32_v_i32m4(aSumsArr, 16);
    vuint16m2_t vVSc16 = __riscv_vwcvtu_x_x_v_u16m2(vVSc8v, 16);
    vuint32m4_t vVSc32u = __riscv_vwcvtu_x_x_v_u32m4(vVSc16, 16);
    vint32m4_t vVSc32 = __riscv_vreinterpret_v_u32m4_i32m4(vVSc32u);
    vint32m4_t vVProd = __riscv_vmul_vv_i32m4(vVSums, vVSc32, 16);

    vint32m1_t vVZero32 = __riscv_vmv_v_x_i32m1(0, 1);
    int32_t vSumPart = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(vVProd, vVZero32, 16));
    vSumBlock += vSumPart;

    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}
#endif

void ggml_vec_dot_iq2_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq2SQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq2SQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotIq2SQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    fGgmlVecDotIq2SQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq2_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
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

static NOINLINE void fGgmlVecDotIq2XsQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xs_grid;

  float vSumf = 0.0f;
#pragma GCC unroll 1
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQs = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;

    int32_t vSumInt = 0;

    // Loop over 4 subblocks of 64 elements
    for (int vIb64 = 0; vIb64 < QK_K / 64; ++vIb64) {

      // Load indices.
      vuint16m1_t vVQs = __riscv_vle16_v_u16m1(vQs, 8);
      vQs += 8;

      // Prepare offsets
      vuint16m1_t vVidxGrid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(vVQs, 511, 8), 3, 8);
      vuint16m1_t vVidxSign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(vVQs, 9, 8), 3, 8);

      // load values and signs from the lookup tables
      vuint64m4_t vVq264 = __riscv_vluxei16_v_u64m4(vGrid64, vVidxGrid, 8);
      vuint64m4_t vVs264 = __riscv_vluxei16_v_u64m4(vSigns64, vVidxSign, 8);
      vint8m4_t vQ2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVq264));
      vint8m4_t vQ2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVs264));
      vint8m4_t vQ2Final = __riscv_vmul_vv_i8m4(vQ2u, vQ2s, 64);
      asm volatile("" ::: "memory");
      vint8m4_t vQ8v = __riscv_vle8_v_i8m4(vQ8, 64);
      vQ8 += 64;

      vint16m8_t vProd = __riscv_vwmul_vv_i16m8(vQ2Final, vQ8v, 64);
      asm volatile("" ::: "memory");
      vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);

      int32_t vSum0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 0), vZeroVec, 16));

      int32_t vSum1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 1), vZeroVec, 16));

      int32_t vSum2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 2), vZeroVec, 16));

      int32_t vSum3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 3), vZeroVec, 16));

      const uint8_t cScaleByte1 = vScales[0];
      const uint8_t cScaleByte2 = vScales[1];
      vScales += 2;

      vSumInt += vSum0 * ((cScaleByte1 & 0x0F) * 2 + 1);
      vSumInt += vSum1 * ((cScaleByte1 >> 4) * 2 + 1);
      vSumInt += vSum2 * ((cScaleByte2 & 0x0F) * 2 + 1);
      vSumInt += vSum3 * ((cScaleByte2 >> 4) * 2 + 1);
    }

    vSumf += cD * vSumInt;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2XsQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xs_grid;

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQs = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;

    int32_t vSumInt = 0;

    for (int vIb128 = 0; vIb128 < 2; ++vIb128) {

      vuint16m1_t vVQs = __riscv_vle16_v_u16m1(vQs, 16);
      vQs += 16;

      // Prepare offsets for grid and signs
      vuint16m1_t vVidxGrid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(vVQs, 511, 16), 3, 16);
      vuint16m1_t vVidxSign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(vVQs, 9, 16), 3, 16);

      // Indexed load 128 weights (16 x 8-byte chunks)
      vuint64m4_t vVq264 = __riscv_vluxei16_v_u64m4(vGrid64, vVidxGrid, 16);
      vuint64m4_t vVs264 = __riscv_vluxei16_v_u64m4(vSigns64, vVidxSign, 16);

      vint8m4_t vQ2u = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVq264));
      vint8m4_t vQ2s = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVs264));

      // Apply signs to get dequantized IQ2 values
      vint8m4_t vQ2Final = __riscv_vmul_vv_i8m4(vQ2u, vQ2s, 128);
      asm volatile("" ::: "memory");

      // Load corresponding Q8 weights
      vint8m4_t vQ8v = __riscv_vle8_v_i8m4(vQ8, 128);
      vQ8 += 128;

      vint16m8_t vProd = __riscv_vwmul_vv_i16m8(vQ2Final, vQ8v, 128);
      asm volatile("" ::: "memory");

      uint8_t vSc0 = vScales[0];
      uint8_t vSc1 = vScales[1];
      uint8_t vSc2 = vScales[2];
      uint8_t vSc3 = vScales[3];
      vScales += 4;

      vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);

      // 9. Reduce each 16-element chunk and apply corresponding nibble scale

      int32_t vS0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 0), vZeroVec, 16));
      vSumInt += vS0 * ((vSc0 & 0x0F) * 2 + 1);

      int32_t vS1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 1), vZeroVec, 16));
      vSumInt += vS1 * ((vSc0 >> 4) * 2 + 1);

      int32_t vS2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 2), vZeroVec, 16));
      vSumInt += vS2 * ((vSc1 & 0x0F) * 2 + 1);

      int32_t vS3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 3), vZeroVec, 16));
      vSumInt += vS3 * ((vSc1 >> 4) * 2 + 1);

      int32_t vS4 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 4), vZeroVec, 16));
      vSumInt += vS4 * ((vSc2 & 0x0F) * 2 + 1);

      int32_t vS5 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 5), vZeroVec, 16));
      vSumInt += vS5 * ((vSc2 >> 4) * 2 + 1);

      int32_t vS6 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 6), vZeroVec, 16));
      vSumInt += vS6 * ((vSc3 & 0x0F) * 2 + 1);

      int32_t vS7 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(vProd, 7), vZeroVec, 16));
      vSumInt += vS7 * ((vSc3 >> 4) * 2 + 1);
    }

    vSumf += cD * (float)vSumInt;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2XsQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xs_grid;

  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint16_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    vint8m4_t vQ8All = __riscv_vle8_v_i8m4(vQ8, 256);

    // Load indices ---
    vuint16m1_t vVQs = __riscv_vle16_v_u16m1(vQs, 32);

    // Extract low 9 bits and multiply by 8 (shift left 3) for byte offset into uint64 table
    vuint16m1_t vVidxGrid = __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(vVQs, 511, 32), 3, 32);

    // Extract high 7 bits (shift right 9) and multiply by 8 (shift left 3) for byte offset
    vuint16m1_t vVidxSign = __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(vVQs, 9, 32), 3, 32);

    vuint64m4_t vVq264 = __riscv_vluxei16_v_u64m4(vGrid64, vVidxGrid, 32);
    vuint64m4_t vVs264 = __riscv_vluxei16_v_u64m4(vSigns64, vVidxSign, 32);

    vint8m4_t vQ2All = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVq264));
    vint8m4_t vS2All = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVs264));

    vint8m4_t vQ2Signed = __riscv_vmul_vv_i8m4(vQ2All, vS2All, 256);
    vint16m8_t vDotAll = __riscv_vwmul_vv_i16m8(vQ2Signed, vQ8All, 256);
    float vSum = 0.0f;
    vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);

#pragma GCC unroll 1
    for (int vJ = 0; vJ < 8; ++vJ) {
      uint8_t vSc = vScales[vJ];
      int16_t vScLo = 2 * (vSc & 0x0F) + 1;
      int16_t vScHi = 2 * (vSc >> 4) + 1;

      vint32m1_t vSumV0 =
          __riscv_vwredsum_vs_i16m8_i32m1(__riscv_vslidedown_vx_i16m8(vDotAll, vJ * 32, 16), vZeroVec, 16);
      int32_t vIsum0 = __riscv_vmv_x_s_i32m1_i32(vSumV0);

      vint32m1_t vSumV1 =
          __riscv_vwredsum_vs_i16m8_i32m1(__riscv_vslidedown_vx_i16m8(vDotAll, vJ * 32 + 16, 16), vZeroVec, 16);
      int32_t vIsum1 = __riscv_vmv_x_s_i32m1_i32(vSumV1);

      vSum += (float)vIsum0 * vScLo + (float)vIsum1 * vScHi;
    }

    vSumf += vSum * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}
#endif

void ggml_vec_dot_iq2_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq2XsQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq2XsQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 512 and above
    fGgmlVecDotIq2XsQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq2_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq2XxsQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xxs_grid;

  uint32_t aShiftConstants[4] = {0, 7, 14, 21};
  vuint32m1_t vVShifts = __riscv_vle32_v_u32m1(aShiftConstants, 4);

  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ2Ptr = (const uint8_t *)vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vSum = 0.0f;

#pragma GCC unroll 1
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vint8m2_t vQ81 = __riscv_vle8_v_i8m2(vQ8, 32);
      vQ8 += 32;
      vint8m2_t vQ82 = __riscv_vle8_v_i8m2(vQ8, 32);
      vQ8 += 32;

      vuint8mf4_t vVRawQ21 = __riscv_vle8_v_u8mf4(vQ2Ptr, 4);
      vuint8mf4_t vVRawQ22 = __riscv_vle8_v_u8mf4(vQ2Ptr + 8, 4);

      vuint16mf2_t vVidxQ21 = __riscv_vwcvtu_x_x_v_u16mf2(vVRawQ21, 4);
      vuint16mf2_t vVidxQ22 = __riscv_vwcvtu_x_x_v_u16mf2(vVRawQ22, 4);

      vVidxQ21 = __riscv_vsll_vx_u16mf2(vVidxQ21, 3, 4);
      vVidxQ22 = __riscv_vsll_vx_u16mf2(vVidxQ22, 3, 4);

      uint32_t vSPacked1, vSPacked2;
      memcpy(&vSPacked1, vQ2Ptr + 4, 4);
      memcpy(&vSPacked2, vQ2Ptr + 12, 4);

      vuint32m1_t vVS1 = __riscv_vmv_v_x_u32m1(vSPacked1, 4);
      vuint32m1_t vVS2 = __riscv_vmv_v_x_u32m1(vSPacked2, 4);
      vVS1 = __riscv_vsrl_vv_u32m1(vVS1, vVShifts, 4);
      vVS2 = __riscv_vsrl_vv_u32m1(vVS2, vVShifts, 4);

      vVS1 = __riscv_vand_vx_u32m1(vVS1, 127, 4);
      vVS2 = __riscv_vand_vx_u32m1(vVS2, 127, 4);

      vuint16mf2_t vVidxS21 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(vVS1, 4), 3, 4);
      vuint16mf2_t vVidxS22 = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(vVS2, 4), 3, 4);

      vuint64m2_t vVq2641 = __riscv_vluxei16_v_u64m2(vGrid64, vVidxQ21, 4);
      vuint64m2_t vVq2642 = __riscv_vluxei16_v_u64m2(vGrid64, vVidxQ22, 4);

      vint8m2_t vQ21 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVq2641));
      vint8m2_t vQ22 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVq2642));

      vuint64m2_t vVs2641 = __riscv_vluxei16_v_u64m2(vSigns64, vVidxS21, 4);
      vuint64m2_t vVs2642 = __riscv_vluxei16_v_u64m2(vSigns64, vVidxS22, 4);
      vint8m2_t vS21 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVs2641));
      vint8m2_t vS22 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVs2642));

      vint8m2_t vQ8s1 = __riscv_vmul_vv_i8m2(vQ81, vS21, 32);
      vint8m2_t vQ8s2 = __riscv_vmul_vv_i8m2(vQ82, vS22, 32);

      vint16m4_t vDot1 = __riscv_vwmul_vv_i16m4(vQ8s1, vQ21, 32);
      vint16m4_t vDot2 = __riscv_vwmul_vv_i16m4(vQ8s2, vQ22, 32);

      vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vSumv1 = __riscv_vwredsum_vs_i16m4_i32m1(vDot1, vZeroVec, 32);
      vint32m1_t vSumv2 = __riscv_vwredsum_vs_i16m4_i32m1(vDot2, vZeroVec, 32);

      int32_t vScalarSum1 = __riscv_vmv_x_s_i32m1_i32(vSumv1);
      int32_t vScalarSum2 = __riscv_vmv_x_s_i32m1_i32(vSumv2);

      int16_t vScale1 = 2 * ((vSPacked1 >> 28) & 0xF) + 1;
      int16_t vScale2 = 2 * ((vSPacked2 >> 28) & 0xF) + 1;

      vSum += vScalarSum1 * vScale1 + vScalarSum2 * vScale2;
      vQ2Ptr += 16;
    }
    vSumf += vSum * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2XxsQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xxs_grid;

  uint32_t aShiftConstants[4] = {0, 7, 14, 21};
  vuint32mf2_t vVShifts = __riscv_vle32_v_u32mf2(aShiftConstants, 4);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ2Ptr = (const uint8_t *)vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vSum = 0.0f;

    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      vint8m1_t vQ81 = __riscv_vle8_v_i8m1(vQ8, 32);
      vQ8 += 32;
      vint8m1_t vQ82 = __riscv_vle8_v_i8m1(vQ8, 32);
      vQ8 += 32;

      vuint8mf8_t vVRawQ21 = __riscv_vle8_v_u8mf8(vQ2Ptr, 4);
      vuint8mf8_t vVRawQ22 = __riscv_vle8_v_u8mf8(vQ2Ptr + 8, 4);

      vuint16mf4_t vVidxQ21 = __riscv_vwcvtu_x_x_v_u16mf4(vVRawQ21, 4);
      vuint16mf4_t vVidxQ22 = __riscv_vwcvtu_x_x_v_u16mf4(vVRawQ22, 4);

      vVidxQ21 = __riscv_vsll_vx_u16mf4(vVidxQ21, 3, 4);
      vVidxQ22 = __riscv_vsll_vx_u16mf4(vVidxQ22, 3, 4);

      uint32_t vSPacked1, vSPacked2;
      memcpy(&vSPacked1, vQ2Ptr + 4, 4);
      memcpy(&vSPacked2, vQ2Ptr + 12, 4);

      vuint32mf2_t vVS1 = __riscv_vmv_v_x_u32mf2(vSPacked1, 4);
      vuint32mf2_t vVS2 = __riscv_vmv_v_x_u32mf2(vSPacked2, 4);

      vVS1 = __riscv_vsrl_vv_u32mf2(vVS1, vVShifts, 4);
      vVS2 = __riscv_vsrl_vv_u32mf2(vVS2, vVShifts, 4);

      vVS1 = __riscv_vand_vx_u32mf2(vVS1, 127, 4);
      vVS2 = __riscv_vand_vx_u32mf2(vVS2, 127, 4);

      // Narrow u32 -> u16 (vncvt) and Scale by 8 to get byte offsets
      vuint16mf4_t vVidxS21 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(vVS1, 4), 3, 4);
      vuint16mf4_t vVidxS22 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(vVS2, 4), 3, 4);

      // Load q2 values from lookup grid
      vuint64m1_t vVq2641 = __riscv_vluxei16_v_u64m1(vGrid64, vVidxQ21, 4);
      vuint64m1_t vVq2642 = __riscv_vluxei16_v_u64m1(vGrid64, vVidxQ22, 4);
      vint8m1_t vQ21 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vVq2641));
      vint8m1_t vQ22 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vVq2642));

      // Load sign values
      vuint64m1_t vVs2641 = __riscv_vluxei16_v_u64m1(vSigns64, vVidxS21, 4);
      vuint64m1_t vVs2642 = __riscv_vluxei16_v_u64m1(vSigns64, vVidxS22, 4);
      vint8m1_t vS21 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vVs2641));
      vint8m1_t vS22 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vVs2642));

      // Apply signs to q8
      vint8m1_t vQ8s1 = __riscv_vmul_vv_i8m1(vQ81, vS21, 32);
      vint8m1_t vQ8s2 = __riscv_vmul_vv_i8m1(vQ82, vS22, 32);

      // multiplying q2 with q8
      vint16m2_t vDot1 = __riscv_vwmul_vv_i16m2(vQ8s1, vQ21, 32);
      vint16m2_t vDot2 = __riscv_vwmul_vv_i16m2(vQ8s2, vQ22, 32);

      vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vSumv1 = __riscv_vwredsum_vs_i16m2_i32m1(vDot1, vZeroVec, 32);
      vint32m1_t vSumv2 = __riscv_vwredsum_vs_i16m2_i32m1(vDot2, vZeroVec, 32);
      int32_t vScalarSum1 = __riscv_vmv_x_s_i32m1_i32(vSumv1);
      int32_t vScalarSum2 = __riscv_vmv_x_s_i32m1_i32(vSumv2);
      int16_t vScale1 = 2 * ((vSPacked1 >> 28) & 0xF) + 1;
      int16_t vScale2 = 2 * ((vSPacked2 >> 28) & 0xF) + 1;

      vSum += vScalarSum1 * vScale1 + vScalarSum2 * vScale2;
      vQ2Ptr += 16;
    }
    vSumf += vSum * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}

static NOINLINE void fGgmlVecDotIq2XxsQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint64_t *vGrid64 = (const uint64_t *)iq2xxs_grid;
  // Shift pattern {0,7,14,21} repeated 8 times for all 8 sub-blocks
  uint8_t aShiftArr[32] = {0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21,
                           0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21, 0, 7, 14, 21};
  vuint8mf2_t vVShifts = __riscv_vle8_v_u8mf2(aShiftArr, 32);

  // Gather pattern to broadcast the 8 sub-block scales across the 32 lookup slots
  uint8_t aGatherArr[32] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
  vuint8mf2_t vVSignGatherIdx = __riscv_vle8_v_u8mf2(aGatherArr, 32);

  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ2Ptr = (const uint8_t *)vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    vint8m4_t vQ8All = __riscv_vle8_v_i8m4(vQ8, 256);

    // De-interleave all 8 Index/Scale pairs for the 8x32-element sub-blocks
    vuint32mf2x2_t vTuple = __riscv_vlseg2e32_v_u32mf2x2((const uint32_t *)vQ2Ptr, 8);
    vuint32mf2_t vVInd32 = __riscv_vget_v_u32mf2x2_u32mf2(vTuple, 0);
    vuint32mf2_t vVSc32 = __riscv_vget_v_u32mf2x2_u32mf2(vTuple, 1);

    vuint8mf2_t vVRawQ2 = __riscv_vreinterpret_v_u32mf2_u8mf2(vVInd32);
    vuint16m1_t vVidxQ2 = __riscv_vwcvtu_x_x_v_u16m1(vVRawQ2, 32);
    vVidxQ2 = __riscv_vsll_vx_u16m1(vVidxQ2, 3, 32);

    vuint32m2_t vVS = __riscv_vrgatherei16_vv_u32m2(__riscv_vlmul_ext_v_u32mf2_u32m2(vVSc32),
                                                    __riscv_vwcvtu_x_x_v_u16m1(vVSignGatherIdx, 32), 32);
    vVS = __riscv_vsrl_vv_u32m2(vVS, __riscv_vwcvtu_x_x_v_u32m2(__riscv_vwcvtu_x_x_v_u16m1(vVShifts, 32), 32), 32);
    vVS = __riscv_vand_vx_u32m2(vVS, 127, 32);
    vuint16m1_t vVidxS2 = __riscv_vsll_vx_u16m1(__riscv_vncvt_x_x_w_u16m1(vVS, 32), 3, 32);

    vuint64m4_t vVq264 = __riscv_vluxei16_v_u64m4(vGrid64, vVidxQ2, 32);
    vuint64m4_t vVs264 = __riscv_vluxei16_v_u64m4(vSigns64, vVidxS2, 32);
    vint8m4_t vQ2All = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVq264));
    vint8m4_t vS2All = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVs264));

    vint8m4_t vQ8sAll = __riscv_vmul_vv_i8m4(vQ8All, vS2All, 256);
    vint16m8_t vDotAll = __riscv_vwmul_vv_i16m8(vQ8sAll, vQ2All, 256);

    float vSum = 0.0f;
    vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);

    for (int vJ = 0; vJ < 8; ++vJ) {
      uint32_t vSP = __riscv_vmv_x_s_u32mf2_u32(__riscv_vslidedown_vx_u32mf2(vVSc32, vJ, 8));
      int16_t vSc = 2 * ((vSP >> 28) & 0xF) + 1;
      vDotAll = __riscv_vslidedown_vx_i16m8(vDotAll, vJ * 32, 32);
      vint32m1_t vSumV = __riscv_vwredsum_vs_i16m8_i32m1(vDotAll, vZeroVec, 32);
      int32_t vIsum = __riscv_vmv_x_s_i32m1_i32(vSumV);
      vSum += (float)vIsum * vSc;
    }

    vSumf += vSum * cCombinedScale;
  }
  *pS = 0.125f * vSumf;
}
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq2XxsQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq2XxsQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 512 and above
    fGgmlVecDotIq2XxsQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq2_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq3SQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint32_t *vGrid32 = (const uint32_t *)iq3s_grid;

  vuint8mf2_t vVId8 = __riscv_vid_v_u8mf2(8);
  vuint8m2_t vVId32 = __riscv_vid_v_u8m2(32);

  // Keeping these in a tight scope to hint they're only needed for the mask computation.
  vuint8m2_t vVSignGatherIndices, vVSignMasks;
  {
    vuint8m2_t vVShifts = __riscv_vand_vx_u8m2(vVId32, 7, 32);
    vuint8m2_t vVOne32 = __riscv_vmv_v_x_u8m2(1, 32);
    vVSignGatherIndices = __riscv_vsrl_vx_u8m2(vVId32, 3, 32);
    vVSignMasks = __riscv_vsll_vv_u8m2(vVOne32, vVShifts, 32);
  }

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cCombinedScale = cD * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const uint8_t *GGML_RESTRICT vSigns = vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vSumBlock = 0.0f;

    for (int vIb = 0; vIb < 8; ++vIb) {

      // Grid lookup
      vuint8m2_t vVGridU8;
      {
        vuint8mf2_t vVQsU8 = __riscv_vle8_v_u8mf2(vQs, 8);
        vQs += 8;

        uint8_t vQhVal = *vQh++;
        vuint8mf2_t vVQhVal = __riscv_vmv_v_x_u8mf2(vQhVal, 8);
        vVQhVal = __riscv_vsrl_vv_u8mf2(vVQhVal, vVId8, 8);
        vVQhVal = __riscv_vand_vx_u8mf2(vVQhVal, 1, 8);

        vuint16m1_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16m1(vVQsU8, 8);
        vVQsU16 = __riscv_vsll_vx_u16m1(vVQsU16, 2, 8);

        vuint16m1_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16m1(vVQhVal, 8);
        vVQhU16 = __riscv_vsll_vx_u16m1(vVQhU16, 10, 8);

        vuint16m1_t vVGridOffsets = __riscv_vor_vv_u16m1(vVQsU16, vVQhU16, 8);

        vuint32m2_t vVGridPacked = __riscv_vluxei16_v_u32m2(vGrid32, vVGridOffsets, 8);
        vVGridU8 = __riscv_vreinterpret_v_u32m2_u8m2(vVGridPacked);
      }
      __asm__ volatile("" ::: "memory");

      // Sign application and dot product
      int32_t vSVal;
      {
        vuint8mf4_t vVSignsRaw = __riscv_vle8_v_u8mf4(vSigns, 4);
        vSigns += 4;

        vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf4_u8m2(vVSignsRaw);
        vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 32);
        vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 32);
        vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 32);

        vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 32);
        vQ8 += 32;

        vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 32);
        vint16m4_t vVDot = __riscv_vwmulsu_vv_i16m4(vVQ8Signed, vVGridU8, 32);

        vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);
        vSVal = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(vVDot, vVZero, 32));
      }
      __asm__ volatile("" ::: "memory");
      {
        uint8_t vScByte = vScales[vIb >> 1];
        int vScVal = (vIb & 1) ? (vScByte >> 4) : (vScByte & 0xF);
        vScVal = vScVal * 2 + 1;
        vSumBlock += (float)(vSVal * vScVal);
      }
    }
    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq3SQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const uint64_t *vGrid64 = (const uint64_t *)iq3s_grid;

  // --- Pre-load Constants ---
  const uint16_t aQhBitShiftsArr[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vuint8m2_t vVSignGatherIndices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
  vuint8m2_t vVSignMasks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);
  vuint16m1_t vVQhShifts = __riscv_vle16_v_u16m1(aQhBitShiftsArr, 16);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cCombinedScale = cD * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const uint8_t *GGML_RESTRICT vSigns = vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vSumBlock = 0.0f;

    // Loop: Process 64 weights (16 mini-blocks of 4) per iteration
    for (int vIb = 0; vIb < 4; ++vIb) {

      vuint8mf2_t vVQsU8 = __riscv_vle8_v_u8mf2(vQs, 16);
      vQs += 16;

      uint16_t vQhVal;
      memcpy(&vQhVal, vQh, 2);
      vQh += 2;

      vuint16m1_t vVQhVal = __riscv_vmv_v_x_u16m1(vQhVal, 16);
      // Extract bits: (qh >> i) & 1
      vVQhVal = __riscv_vsrl_vv_u16m1(vVQhVal, vVQhShifts, 16);
      vVQhVal = __riscv_vand_vx_u16m1(vVQhVal, 1, 16);

      vuint16m1_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16m1(vVQsU8, 16);
      vVQsU16 = __riscv_vsll_vx_u16m1(vVQsU16, 2, 16);
      vVQhVal = __riscv_vsll_vx_u16m1(vVQhVal, 10, 16);
      vuint16m1_t vVGridOffsets = __riscv_vor_vv_u16m1(vVQsU16, vVQhVal, 16);

      // Grid value is 4xuint8
      vuint32m2_t vVGridPacked = __riscv_vluxei16_v_u32m2((const uint32_t *)vGrid64, vVGridOffsets, 16);
      vuint8m2_t vVGridU8 = __riscv_vreinterpret_v_u32m2_u8m2(vVGridPacked);
      vuint8mf4_t vVSignsRaw = __riscv_vle8_v_u8mf4(vSigns, 8);
      vSigns += 8;

      // Generate sign mask
      vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf4_u8m2(vVSignsRaw);
      vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGatherIndices, 64);
      vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 64);
      vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 64);

      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 64);
      vQ8 += 64;

      // Apply Signs
      vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 64);
      vint16m4_t vVDot = __riscv_vwmulsu_vv_i16m4(vVQ8Signed, vVGridU8, 64);

      // Reduction
      vint16m2_t vVDotLo = __riscv_vget_v_i16m4_i16m2(vVDot, 0);
      vint16m2_t vVDotHi = __riscv_vget_v_i16m4_i16m2(vVDot, 1);
      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);

      int32_t vSLo = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(vVDotLo, vVZero, 32));
      int32_t vSHi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(vVDotHi, vVZero, 32));

      // Apply sub-scales
      uint8_t vScByte = *vScales++;
      int vScLo = (vScByte & 0xF) * 2 + 1;
      int vScHi = (vScByte >> 4) * 2 + 1;

      vSumBlock += vSLo * vScLo + vSHi * vScHi;
    }
    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq3SQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  const uint32_t *vGrid32 = (const uint32_t *)iq3s_grid;

  // Generate Constants
  vuint8mf2_t vVId32 = __riscv_vid_v_u8mf2(32);
  vuint8mf2_t vVQhGather = __riscv_vsrl_vx_u8mf2(vVId32, 3, 32);
  vuint8mf2_t vVQhShifts = __riscv_vand_vx_u8mf2(vVId32, 7, 32);
  vuint8m2_t vVId128 = __riscv_vid_v_u8m2(128);
  vuint8m2_t vVSignGather = __riscv_vsrl_vx_u8m2(vVId128, 3, 128);    // byte index
  vuint8m2_t vVSignShiftAmts = __riscv_vand_vx_u8m2(vVId128, 7, 128); // bit shift
  vuint8m2_t vVOne128 = __riscv_vmv_v_x_u8m2(1, 128);
  vuint8m2_t vVSignMasks = __riscv_vsll_vv_u8m2(vVOne128, vVSignShiftAmts, 128);
  vuint8m2_t vVScaleIndices = __riscv_vsrl_vx_u8m2(vVId128, 5, 128);

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cCombinedScale = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vScales = vX[vI].scales;
    const uint8_t *GGML_RESTRICT vSigns = vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vSumBlock = 0.0f;
    for (int vIb = 0; vIb < 2; ++vIb) {
      vuint8mf2_t vVQsU8 = __riscv_vle8_v_u8mf2(vQs, 32);
      vQs += 32;
      vuint8mf2_t vVQhLoaded = __riscv_vle8_v_u8mf2(vQh, 4);
      vQh += 4;
      vuint8mf2_t vVQhExpanded = __riscv_vrgather_vv_u8mf2(vVQhLoaded, vVQhGather, 32);
      vVQhExpanded = __riscv_vsrl_vv_u8mf2(vVQhExpanded, vVQhShifts, 32);
      vVQhExpanded = __riscv_vand_vx_u8mf2(vVQhExpanded, 1, 32);
      vuint16m1_t vVQsU16 = __riscv_vwcvtu_x_x_v_u16m1(vVQsU8, 32);
      vVQsU16 = __riscv_vsll_vx_u16m1(vVQsU16, 2, 32); // * 4

      vuint16m1_t vVQhU16 = __riscv_vwcvtu_x_x_v_u16m1(vVQhExpanded, 32);
      vVQhU16 = __riscv_vsll_vx_u16m1(vVQhU16, 10, 32); // * 256 * 4

      vuint16m1_t vVGridOffsets = __riscv_vor_vv_u16m1(vVQsU16, vVQhU16, 32);
      vuint32m2_t vVGridPacked = __riscv_vluxei16_v_u32m2(vGrid32, vVGridOffsets, 32);
      vuint8m2_t vVGridU8 = __riscv_vreinterpret_v_u32m2_u8m2(vVGridPacked);
      vuint8mf2_t vVSignsRaw = __riscv_vle8_v_u8mf2(vSigns, 16);
      vSigns += 16;

      vuint8m2_t vVSignsSource = __riscv_vlmul_ext_v_u8mf2_u8m2(vVSignsRaw);
      vuint8m2_t vVSignsBcast = __riscv_vrgather_vv_u8m2(vVSignsSource, vVSignGather, 128);
      vuint8m2_t vVSignBits = __riscv_vand_vv_u8m2(vVSignsBcast, vVSignMasks, 128);
      vbool4_t vMNegative = __riscv_vmsne_vx_u8m2_b4(vVSignBits, 0, 128);

      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 128);
      vQ8 += 128;

      vint8m2_t vVQ8Signed = __riscv_vrsub_vx_i8m2_mu(vMNegative, vVQ8, vVQ8, 0, 128);
      vint16m4_t vVDot = __riscv_vwmulsu_vv_i16m4(vVQ8Signed, vVGridU8, 128);
      uint16_t vScRaw;
      memcpy(&vScRaw, vScales, 2);
      vScales += 2; // Advance 2 bytes

      uint8_t aScUnpacked[4];
      aScUnpacked[0] = (vScRaw & 0xF);
      aScUnpacked[1] = (vScRaw >> 4) & 0xF;
      aScUnpacked[2] = (vScRaw >> 8) & 0xF;
      aScUnpacked[3] = (vScRaw >> 12) & 0xF;

      vuint8mf2_t vVSc4 = __riscv_vle8_v_u8mf2(aScUnpacked, 4);
      vVSc4 = __riscv_vmul_vx_u8mf2(vVSc4, 2, 4);
      vVSc4 = __riscv_vadd_vx_u8mf2(vVSc4, 1, 4);
      vuint8m2_t vVSc4Expanded = __riscv_vlmul_ext_v_u8mf2_u8m2(vVSc4);
      vuint8m2_t vVScalesBcast = __riscv_vrgather_vv_u8m2(vVSc4Expanded, vVScaleIndices, 128);
      vint16m4_t vVScalesI16 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vwcvtu_x_x_v_u16m4(vVScalesBcast, 128));
      vint32m8_t vVWeightedSum = __riscv_vwmul_vv_i32m8(vVDot, vVScalesI16, 128);
      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);
      int32_t vSVal = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(vVWeightedSum, vVZero, 128));

      vSumBlock += vSVal;
    }
    vSumf += vSumBlock * cCombinedScale;
  }
  *pS = vSumf;
}
#endif

void ggml_vec_dot_iq3_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq3SQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq3SQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 512 and above
    fGgmlVecDotIq3SQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq3_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq3XxsQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;
  const int cNb = pN / QK_K;

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint32_t *vGrid32 = (const uint32_t *)iq3xxs_grid;

  // constants for unpacking logic
  const uint32_t aShiftsVal[8] = {0, 7, 14, 21, 0, 7, 14, 21};
  vuint32m2_t vVShifts = __riscv_vle32_v_u32m2(aShiftsVal, 8);

  const uint32_t aGatherIdxVal[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  vuint32m2_t vVGatherIdx = __riscv_vle32_v_u32m2(aGatherIdxVal, 8);

  uint32_t aAux32[2];
  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ3Indices = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vMetadata = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vBlockSum = 0.0f;

    // Process 64 weights per loop
    for (int vIb = 0; vIb < QK_K / 64; ++vIb) {

      // load of metadata via memcpy
      memcpy(aAux32, vMetadata, 2 * sizeof(uint32_t));
      vMetadata += 2 * sizeof(uint32_t);

      vuint8m1_t vVQ3IdxU8 = __riscv_vle8_v_u8m1(vQ3Indices, 16);
      vQ3Indices += 16;

      vuint16m2_t vVQ3IdxU16 = __riscv_vwmulu_vx_u16m2(vVQ3IdxU8, 4, 16);

      vuint32m4_t vVQ3MagnitudesU32 = __riscv_vluxei16_v_u32m4(vGrid32, vVQ3IdxU16, 16);

      vint8m4_t vVQ3Magnitudes = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u32m4_u8m4(vVQ3MagnitudesU32));

      vuint32m2_t vVAux = __riscv_vle32_v_u32m2(aAux32, 2);

      vuint32m2_t vVAuxExpanded = __riscv_vrgather_vv_u32m2(vVAux, vVGatherIdx, 8);

      vuint32m2_t vVSValsRaw = __riscv_vand_vx_u32m2(__riscv_vsrl_vv_u32m2(vVAuxExpanded, vVShifts, 8), 127, 8);

      vuint16m1_t vSignIndicesByteOffset = __riscv_vsll_vx_u16m1(__riscv_vncvt_x_x_w_u16m1(vVSValsRaw, 8), 3, 8);

      vuint64m4_t vVSValsU64 = __riscv_vluxei16_v_u64m4(vSigns64, vSignIndicesByteOffset, 8);

      vint8m4_t vVSVals = __riscv_vreinterpret_v_u8m4_i8m4(__riscv_vreinterpret_v_u64m4_u8m4(vVSValsU64));

      vint8m4_t vVQ3Signed = __riscv_vmul_vv_i8m4(vVQ3Magnitudes, vVSVals, 64);
      asm volatile("" ::: "memory");
      vint8m4_t vVQ8 = __riscv_vle8_v_i8m4(vQ8, 64);
      vQ8 += 64;

      vint16m8_t vVDot = __riscv_vwmul_vv_i16m8(vVQ8, vVQ3Signed, 64);

      asm volatile("" ::: "memory");

      vint16m4_t vVDot1 = __riscv_vget_v_i16m8_i16m4(vVDot, 0);
      vint16m4_t vVDot2 = __riscv_vget_v_i16m8_i16m4(vVDot, 1);

      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);

      vint32m1_t vVSum1 = __riscv_vwredsum_vs_i16m4_i32m1(vVDot1, vVZero, 32);
      vint32m1_t vVSum2 = __riscv_vwredsum_vs_i16m4_i32m1(vVDot2, vVZero, 32);

      int32_t vSum1I = __riscv_vmv_x_s_i32m1_i32(vVSum1);
      int32_t vSum2I = __riscv_vmv_x_s_i32m1_i32(vVSum2);

      const float cScale1F = (float)(2 * (aAux32[0] >> 28) + 1);
      const float cScale2F = (float)(2 * (aAux32[1] >> 28) + 1);

      vBlockSum += vSum1I * cScale1F + vSum2I * cScale2F;
    }

    vSumf += cD * vBlockSum;
  }
  *pS = 0.25f * vSumf;
}

static NOINLINE void fGgmlVecDotIq3XxsQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;
  const int cNb = pN / QK_K;

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint32_t *vGrid32 = (const uint32_t *)iq3xxs_grid;

  // constants for unpacking logic
  const uint32_t aShiftsVal[8] = {0, 7, 14, 21, 0, 7, 14, 21};
  vuint32m1_t vVShifts = __riscv_vle32_v_u32m1(aShiftsVal, 8);

  const uint32_t aGatherIdxVal[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  vuint32m1_t vVGatherIdx = __riscv_vle32_v_u32m1(aGatherIdxVal, 8);

  uint32_t aAux32[2];
  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ3Indices = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vMetadata = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vBlockSum = 0.0f;

    for (int vIb = 0; vIb < QK_K / 64; ++vIb) {
      // Load q8 (64 bytes)
      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 64);
      vQ8 += 64;

      // load of metadata via memcpy
      memcpy(aAux32, vMetadata, 2 * sizeof(uint32_t));
      vMetadata += 2 * sizeof(uint32_t);

      // Load q3 indices and gather magnitudes
      vuint8mf2_t vVQ3IdxU8 = __riscv_vle8_v_u8mf2(vQ3Indices, 16);
      vQ3Indices += 16;

      vuint16m1_t vVQ3IdxU16 = __riscv_vwmulu_vx_u16m1(vVQ3IdxU8, 4, 16);
      vuint32m2_t vVQ3MagnitudesU32 = __riscv_vluxei16_v_u32m2(vGrid32, vVQ3IdxU16, 16);
      vint8m2_t vVQ3Magnitudes = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u32m2_u8m2(vVQ3MagnitudesU32));

      // --- Unpacking of Sign Indices ---

      // 1. Load the 2 auxiliary 32-bit integers into a vector
      vuint32m1_t vVAux = __riscv_vle32_v_u32m1(aAux32, 2);

      // 2. Broadcast/Gather: replicate aux[0] to first 4 lanes, aux[1] to next 4 lanes
      vuint32m1_t vVAuxExpanded = __riscv_vrgather_vv_u32m1(vVAux, vVGatherIdx, 8);

      // 3. Apply Shifts and Mask: ((val >> shift) & 127)
      vuint32m1_t vVSValsRaw = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(vVAuxExpanded, vVShifts, 8), 127, 8);

      // 4. Narrow to u16 (required for vluxei index) and multiply by 8 (byte offset for u64 table)
      vuint16mf2_t vSignIndicesByteOffset = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(vVSValsRaw, 8), 3, 8);

      // 5. Gather Signs
      vuint64m2_t vVSValsU64 = __riscv_vluxei16_v_u64m2(vSigns64, vSignIndicesByteOffset, 8);
      vint8m2_t vVSVals = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVSValsU64));

      vint8m2_t vVQ3Signed = __riscv_vmul_vv_i8m2(vVQ3Magnitudes, vVSVals, 64);
      vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVQ8, vVQ3Signed, 64);

      vint16m2_t vVDot1 = __riscv_vget_v_i16m4_i16m2(vVDot, 0);
      vint16m2_t vVDot2 = __riscv_vget_v_i16m4_i16m2(vVDot, 1);

      vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vVSum1 = __riscv_vwredsum_vs_i16m2_i32m1(vVDot1, vVZero, 32);
      vint32m1_t vVSum2 = __riscv_vwredsum_vs_i16m2_i32m1(vVDot2, vVZero, 32);

      int32_t vSum1I = __riscv_vmv_x_s_i32m1_i32(vVSum1);
      int32_t vSum2I = __riscv_vmv_x_s_i32m1_i32(vVSum2);

      const float cScale1F = (float)(2 * (aAux32[0] >> 28) + 1);
      const float cScale2F = (float)(2 * (aAux32[1] >> 28) + 1);

      vBlockSum += vSum1I * cScale1F + vSum2I * cScale2F;
    }

    vSumf += cD * vBlockSum;
  }
  *pS = 0.25f * vSumf;
}

static NOINLINE void fGgmlVecDotIq3XxsQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;
  const int cNb = pN / QK_K;

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint32_t *vGrid32 = (const uint32_t *)iq3xxs_grid;

  // generate constants for unpacking metadata words into sign indices
  vuint32m1_t vVShifts;
  {
    vuint32m1_t vVBase = __riscv_vid_v_u32m1(16);
    vuint32m1_t vVMod4 = __riscv_vand_vx_u32m1(vVBase, 3, 16);
    vVShifts = __riscv_vmul_vx_u32m1(vVMod4, 7, 16);
  }

  vuint16mf2_t vVGatherIdx;
  {
    vuint16mf2_t vVIdx = __riscv_vid_v_u16mf2(16);
    vVGatherIdx = __riscv_vsrl_vx_u16mf2(vVIdx, 2, 16);
  }

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ3Indices = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vMetadata = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    float vBlockSum = 0.0f;
    for (int vIb128 = 0; vIb128 < 2; ++vIb128) {

      vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 128);
      vQ8 += 128;
      vuint8mf2_t vVQ3IdxU8 = __riscv_vle8_v_u8mf2(vQ3Indices, 32);
      vQ3Indices += 32;

      vuint16m1_t vVQ3IdxU16 = __riscv_vwmulu_vx_u16m1(vVQ3IdxU8, 4, 32);
      vuint32m2_t vVQ3MagU32 = __riscv_vluxei16_v_u32m2(vGrid32, vVQ3IdxU16, 32);
      vint8m2_t vVQ3Magnitudes = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u32m2_u8m2(vVQ3MagU32));
      vuint32m1_t vVAux = __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vle8_v_u8m1(vMetadata, 16));
      vMetadata += 4 * sizeof(uint32_t);

      vuint32m1_t vVAuxExpanded = __riscv_vrgatherei16_vv_u32m1(vVAux, vVGatherIdx, 16);

      vuint32m1_t vVSRaw = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(vVAuxExpanded, vVShifts, 16), 127, 16);
      vuint16mf2_t vSignByteOffset = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(vVSRaw, 16), 3, 16);
      vuint64m2_t vVSU64 = __riscv_vluxei16_v_u64m2(vSigns64, vSignByteOffset, 16);
      vint8m2_t vVSigns = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVSU64));
      vint8m2_t vVQ3Signed = __riscv_vmul_vv_i8m2(vVQ3Magnitudes, vVSigns, 128);
      vint16m4_t vProd = __riscv_vwmul_vv_i16m4(vVQ3Signed, vVQ8, 128);

      vint32m1_t vZeroVec = __riscv_vmv_v_x_i32m1(0, 1);
      int32_t vGroup0Sum = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vProd, 0), vZeroVec, 32));
      int32_t vGroup1Sum = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vProd, 1), vZeroVec, 32));
      int32_t vGroup2Sum = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vProd, 2), vZeroVec, 32));
      int32_t vGroup3Sum = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(vProd, 3), vZeroVec, 32));

      vuint32m1_t vVScalesRaw = __riscv_vsrl_vx_u32m1(vVAux, 28, 4);
      vuint32m1_t vVScales = __riscv_vadd_vx_u32m1(__riscv_vsll_vx_u32m1(vVScalesRaw, 1, 4), 1, 4);
      int32_t vScale0 = (int32_t)__riscv_vmv_x_s_u32m1_u32(vVScales);
      int32_t vScale1 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(vVScales, 1, 4));
      int32_t vScale2 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(vVScales, 2, 4));
      int32_t vScale3 = (int32_t)__riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(vVScales, 3, 4));

      vBlockSum += (float)(vGroup0Sum * vScale0 + vGroup1Sum * vScale1 + vGroup2Sum * vScale2 + vGroup3Sum * vScale3);
    }

    vSumf += cD * vBlockSum;
  }
  *pS = 0.25f * vSumf;
}

static NOINLINE void fGgmlVecDotIq3XxsQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                                const void *GGML_RESTRICT pVx, size_t pBx,
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

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;
  const uint32_t *vGrid32 = (const uint32_t *)iq3xxs_grid;

  vuint32m1_t vVShifts;
  {
    vuint32m1_t vVId = __riscv_vid_v_u32m1(32);
    vuint32m1_t vVMod4 = __riscv_vand_vx_u32m1(vVId, 3, 32);
    vVShifts = __riscv_vmul_vx_u32m1(vVMod4, 7, 32);
  }
  vuint16mf2_t vVGatherIdx;
  {
    vuint16mf2_t vVId16 = __riscv_vid_v_u16mf2(32);
    vVGatherIdx = __riscv_vsrl_vx_u16mf2(vVId16, 2, 32);
  }

  float vSumf = 0.0f;
  uint32_t aAux32[8]; // Buffer for block metadata

  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;

    const uint8_t *GGML_RESTRICT vQ3Indices = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vMetadata = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    vint8m2_t vVQ8 = __riscv_vle8_v_i8m2(vQ8, 256);
    vuint8mf2_t vVQ3IdxRaw = __riscv_vle8_v_u8mf2(vQ3Indices, 64);
    vuint16m1_t vVQ3IdxU16 = __riscv_vwmulu_vx_u16m1(vVQ3IdxRaw, 4, 64);

    vuint32m2_t vVQ3GridVals = __riscv_vluxei16_v_u32m2(vGrid32, vVQ3IdxU16, 64);

    vint8m2_t vVQ3Mags = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u32m2_u8m2(vVQ3GridVals));

    memcpy(aAux32, vMetadata, 8 * sizeof(uint32_t));
    vuint32m1_t vVAux8 = __riscv_vle32_v_u32m1(aAux32, 8);

    vuint32m1_t vVAux32 = __riscv_vrgatherei16_vv_u32m1(vVAux8, vVGatherIdx, 32);

    vuint32m1_t vVSignIdxRaw = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(vVAux32, vVShifts, 32), 127, 32);

    vuint16mf2_t vVSignOffsets = __riscv_vsll_vx_u16mf2(__riscv_vncvt_x_x_w_u16mf2(vVSignIdxRaw, 32), 3, 32);

    vuint64m2_t vVSignsU64 = __riscv_vluxei16_v_u64m2(vSigns64, vVSignOffsets, 32);

    vint8m2_t vVSigns = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(vVSignsU64));

    vint8m2_t vVQ3Final = __riscv_vmul_vv_i8m2(vVQ3Mags, vVSigns, 256);

    vint16m4_t vVDot = __riscv_vwmul_vv_i16m4(vVQ8, vVQ3Final, 256);
    float vBlockSum = 0.0f;
    vint32m1_t vVZero = __riscv_vmv_v_x_i32m1(0, 1);
    vint16m4_t vVAccum = vVDot;

    for (int vJ = 0; vJ < 8; ++vJ) {
      float vScale = (float)(2 * (aAux32[vJ] >> 28) + 1);

      vint32m1_t vVPartialSum = __riscv_vwredsum_vs_i16m4_i32m1(vVAccum, vVZero, 32);

      int32_t vPartialSumI = __riscv_vmv_x_s_i32m1_i32(vVPartialSum);
      vBlockSum += vPartialSumI * vScale;
      vVAccum = __riscv_vslidedown_vx_i16m4(vVAccum, 32, 32);
    }

    vSumf += cD * vBlockSum;
  }
  *pS = 0.25f * vSumf;
}
#endif

void ggml_vec_dot_iq3_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq3XxsQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq3XxsQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotIq3XxsQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 1024 and above
    fGgmlVecDotIq3XxsQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq3_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq4NlQ80Vl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
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

  // Load the lookup table once.
  const vint8m2_t cValues = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
  int vAcc1, vAcc2;

  // We process 2 blocks at once.
  for (; vIb + 1 < cNb; vIb += 2) {
    // Weights and activations.
    vuint8m1_t vIq4Packed1 = __riscv_vle8_v_u8m1(vX[vIb + 0].qs, 16);
    vint8m2_t vQ8b1 = __riscv_vle8_v_i8m2(vY[vIb + 0].qs, 32);
    vuint8m1_t vIq4Packed2 = __riscv_vle8_v_u8m1(vX[vIb + 1].qs, 16);
    vint8m2_t vQ8b2 = __riscv_vle8_v_i8m2(vY[vIb + 1].qs, 32);

    // Unpack the weight blocks.
    vuint8m2_t vIq4bits1 = __riscv_vcreate_v_u8m1_u8m2(__riscv_vand_vx_u8m1(vIq4Packed1, 0xf, 16),
                                                       __riscv_vsrl_vx_u8m1(vIq4Packed1, 4, 16));
    vuint8m2_t vIq4bits2 = __riscv_vcreate_v_u8m1_u8m2(__riscv_vand_vx_u8m1(vIq4Packed2, 0xf, 16),
                                                       __riscv_vsrl_vx_u8m1(vIq4Packed2, 4, 16));

    // Gather values from the lookup table.
    vint8m2_t vIq4b1 = __riscv_vrgather_vv_i8m2(cValues, vIq4bits1, 32);
    vint8m2_t vIq4b2 = __riscv_vrgather_vv_i8m2(cValues, vIq4bits2, 32);

    // Accumulation.
    vint16m4_t vSum1 = __riscv_vwmul_vv_i16m4(vQ8b1, vIq4b1, 32);
    vint16m4_t vSum2 = __riscv_vwmul_vv_i16m4(vQ8b2, vIq4b2, 32);
    __riscv_vse32_v_i32m1(&vAcc1, __riscv_vwredsum_vs_i16m4_i32m1(vSum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    __riscv_vse32_v_i32m1(&vAcc2, __riscv_vwredsum_vs_i16m4_i32m1(vSum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    vSumf += ((GGML_CPU_FP16_TO_FP32(vX[vIb + 0].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vAcc1));
    vSumf += ((GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vAcc2));
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq4NlQ80Vl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
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

  // Load the lookup table once.
  const vint8mf2_t cValues = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
  int vAcc1, vAcc2;

  // We process 2 blocks at once.
  for (; vIb + 1 < cNb; vIb += 2) {
    // Weights and activations.
    vuint8mf2_t vIq4Packed1 = __riscv_vle8_v_u8mf2(vX[vIb + 0].qs, 16);
    vint8mf2_t vQ8bLo1 = __riscv_vle8_v_i8mf2(vY[vIb + 0].qs, 16);
    vint8mf2_t vQ8bHi1 = __riscv_vle8_v_i8mf2(vY[vIb + 0].qs + 16, 16);
    vuint8mf2_t vIq4Packed2 = __riscv_vle8_v_u8mf2(vX[vIb + 1].qs, 16);
    vint8mf2_t vQ8bLo2 = __riscv_vle8_v_i8mf2(vY[vIb + 1].qs, 16);
    vint8mf2_t vQ8bHi2 = __riscv_vle8_v_i8mf2(vY[vIb + 1].qs + 16, 16);

    // Unpack the weight blocks.
    vuint8mf2_t vIq4bitsLo1 = __riscv_vand_vx_u8mf2(vIq4Packed1, 0xf, 16);
    vuint8mf2_t vIq4bitsHi1 = __riscv_vsrl_vx_u8mf2(vIq4Packed1, 4, 16);
    vuint8mf2_t vIq4bitsLo2 = __riscv_vand_vx_u8mf2(vIq4Packed2, 0xf, 16);
    vuint8mf2_t vIq4bitsHi2 = __riscv_vsrl_vx_u8mf2(vIq4Packed2, 4, 16);

    // Gather values from the lookup table.
    vint8mf2_t vIq4bLo1 = __riscv_vrgather_vv_i8mf2(cValues, vIq4bitsLo1, 16);
    vint8mf2_t vIq4bHi1 = __riscv_vrgather_vv_i8mf2(cValues, vIq4bitsHi1, 16);
    vint8mf2_t vIq4bLo2 = __riscv_vrgather_vv_i8mf2(cValues, vIq4bitsLo2, 16);
    vint8mf2_t vIq4bHi2 = __riscv_vrgather_vv_i8mf2(cValues, vIq4bitsHi2, 16);

    // Accumulation.
    vint16m1_t vSum1 = __riscv_vwmul_vv_i16m1(vQ8bLo1, vIq4bLo1, 16);
    vSum1 = __riscv_vwmacc_vv_i16m1(vSum1, vQ8bHi1, vIq4bHi1, 16);
    vint16m1_t vSum2 = __riscv_vwmul_vv_i16m1(vQ8bLo2, vIq4bLo2, 16);
    vSum2 = __riscv_vwmacc_vv_i16m1(vSum2, vQ8bHi2, vIq4bHi2, 16);
    __riscv_vse32_v_i32m1(&vAcc1, __riscv_vwredsum_vs_i16m1_i32m1(vSum1, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    __riscv_vse32_v_i32m1(&vAcc2, __riscv_vwredsum_vs_i16m1_i32m1(vSum2, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    vSumf += ((GGML_CPU_FP16_TO_FP32(vX[vIb + 0].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vAcc1));
    vSumf += ((GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vAcc2));
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_iq4_nl_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq4NlQ80Vl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 256 and above
    fGgmlVecDotIq4NlQ80Vl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq4_nl_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotIq4XsQ8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const vint8m4_t cValues = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
  float vSumf = 0;

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const int8_t *vQ8 = vY[vIbl].qs;
    const uint8_t *vIq4 = vX[vIbl].qs;
    uint16_t vH = vX[vIbl].scales_h;

    // We process 2 sub-blocks together.
    int vSumi1 = 0, vSumi2 = 0;
#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 64; ++vIb) {
      // Load the packed weights.
      const vuint8m2_t cIq4Packed = __riscv_vle8_v_u8m2(vIq4, 32);
      vIq4 += 32;

      // Unpack the weight blocks.
      const vuint8m2_t cIq4bitsLo = __riscv_vand_vx_u8m2(cIq4Packed, 0xf, 32);
      const vuint8m2_t cIq4bitsHi = __riscv_vsrl_vx_u8m2(cIq4Packed, 4, 32);
      const vuint8m4_t cIq4bits = __riscv_vcreate_v_u8m2_u8m4(cIq4bitsLo, cIq4bitsHi);
      const vuint8m4_t cIq4bitsReorder =
          __riscv_vcreate_v_u8m1_u8m4(__riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(cIq4bits, 0), 16),
                                      __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(cIq4bits, 2), 16),
                                      __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(cIq4bits, 1), 16),
                                      __riscv_vmv_v_v_u8m1(__riscv_vget_v_u8m4_u8m1(cIq4bits, 3), 16));
      const vint8m4_t cIq4b = __riscv_vrgather_vv_i8m4(cValues, cIq4bitsReorder, 64);

      // Multiply with activations.
      const vint8m4_t cQ8b = __riscv_vle8_v_i8m4(vQ8, 64);
      vQ8 += 64;
      const vint16m8_t cProd = __riscv_vwmul_vv_i16m8(cIq4b, cQ8b, 64);

      // Reduce separately.
      const int cAcc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(cProd, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m4_i32m1(__riscv_vget_v_i16m8_i16m4(cProd, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));

      const int cLs1 = ((vX[vIbl].scales_l[vIb] & 0xf) | ((vH << 4) & 0x30)) - 32;
      const int cLs2 = ((vX[vIbl].scales_l[vIb] >> 4) | ((vH << 2) & 0x30)) - 32;
      vH >>= 4;

      vSumi1 += cAcc0 * cLs1;
      vSumi2 += cAcc1 * cLs2;

      __asm__ __volatile__("" ::: "memory");
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d * (vSumi1 + vSumi2);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq4XsQ8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const vint8m4_t cValues = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
  float vSumf = 0;

  // Indices for re-ordering IQ4 data.
  uint16_t aIndex[16] = {
      0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
  };
  vuint16m1_t vIVec = __riscv_vle16_v_u16m1(aIndex, 16);

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const int8_t *vQ8 = vY[vIbl].qs;
    const uint8_t *vIq4 = vX[vIbl].qs;
    uint16_t vH = vX[vIbl].scales_h;

    int vSumi1 = 0, vSumi2 = 0, vSumi3 = 0, vSumi4 = 0;

#pragma GCC unroll 1
    for (int vIb = 0; vIb < QK_K / 128; ++vIb) {
      // Weights and activations.
      vuint8m2_t vIq4Packed = __riscv_vle8_v_u8m2(vIq4, 64);
      vIq4 += 64;

      // Unpack the weight blocks.
      vuint8m2_t vIq4bitsLo = __riscv_vand_vx_u8m2(vIq4Packed, 0xf, 64);
      vuint8m2_t vIq4bitsHi = __riscv_vsrl_vx_u8m2(vIq4Packed, 4, 64);
      vuint8m4_t vIq4bits = __riscv_vcreate_v_u8m2_u8m4(vIq4bitsLo, vIq4bitsHi);
      vuint8m4_t vIq4bitsReorder = __riscv_vreinterpret_v_u64m4_u8m4(
          __riscv_vrgatherei16_vv_u64m4(__riscv_vreinterpret_v_u8m4_u64m4(vIq4bits), vIVec, 16));
      vint8m4_t vIq4b = __riscv_vrgather_vv_i8m4(cValues, vIq4bitsReorder, 128);

      __asm__ __volatile__("" ::: "memory");

      // Multiply with activations.
      vint8m4_t vQ8b = __riscv_vle8_v_i8m4(vQ8, 128);
      vint16m8_t vProd = __riscv_vwmul_vv_i16m8(vIq4b, vQ8b, 128);
      vQ8 += 128;

      __asm__ __volatile__("" ::: "memory");

      // Reduce separately.
      int vAcc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int vAcc1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int vAcc2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
      int vAcc3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(vProd, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));

      int vLs1 = ((vX[vIbl].scales_l[vIb * 2 + 0] & 0xf) | ((vH << 4) & 0x30)) - 32;
      int vLs2 = ((vX[vIbl].scales_l[vIb * 2 + 0] >> 4) | ((vH << 2) & 0x30)) - 32;
      int vLs3 = ((vX[vIbl].scales_l[vIb * 2 + 1] & 0xf) | ((vH << 0) & 0x30)) - 32;
      int vLs4 = ((vX[vIbl].scales_l[vIb * 2 + 1] >> 4) | ((vH >> 2) & 0x30)) - 32;
      vH >>= 8;

      vSumi1 += vAcc0 * vLs1;
      vSumi2 += vAcc1 * vLs2;
      vSumi3 += vAcc2 * vLs3;
      vSumi4 += vAcc3 * vLs4;

      __asm__ __volatile__("" ::: "memory");
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d * (vSumi1 + vSumi2 + vSumi3 + vSumi4);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq4XsQ8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const vint8m4_t cValues = __riscv_vle8_v_i8m4(kvalues_iq4nl, 16);
  float vSumf = 0;

  // Indices for re-ordering IQ4 data.
  const uint16_t aIndex[32] = {
      0, 1, 16, 17, 2,  3,  18, 19, 4,  5,  20, 21, 6,  7,  22, 23,
      8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31,
  };
  const vuint16m1_t cIVec = __riscv_vle16_v_u16m1(aIndex, 32);

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const int8_t *vQ8 = vY[vIbl].qs;
    const uint8_t *vIq4 = vX[vIbl].qs;
    uint16_t vH = vX[vIbl].scales_h;

    int vSumi = 0;

#pragma GCC unroll 1
    // Process the entire super-block together.
    for (int vIb = 0; vIb < QK_K / 256; ++vIb) {
      // Weights and activations.
      const vuint8m2_t cIq4Packed = __riscv_vle8_v_u8m2(vIq4, 128);
      vIq4 += 128;

      // Unpack the weight blocks.
      const vuint8m2_t cIq4bitsLo = __riscv_vand_vx_u8m2(cIq4Packed, 0xf, 128);
      const vuint8m2_t cIq4bitsHi = __riscv_vsrl_vx_u8m2(cIq4Packed, 4, 128);
      const vuint8m4_t cIq4bits = __riscv_vcreate_v_u8m2_u8m4(cIq4bitsLo, cIq4bitsHi);
      const vuint8m4_t cIq4bitsReorder = __riscv_vreinterpret_v_u64m4_u8m4(
          __riscv_vrgatherei16_vv_u64m4(__riscv_vreinterpret_v_u8m4_u64m4(cIq4bits), cIVec, 32));
      const vint8m4_t cIq4b = __riscv_vrgather_vv_i8m4(cValues, cIq4bitsReorder, 256);

      __asm__ __volatile__("" ::: "memory");

      // Multiply with activations.
      const vint8m4_t cQ8b = __riscv_vle8_v_i8m4(vQ8, 256);
      const vint16m8_t cProd = __riscv_vwmul_vv_i16m8(cIq4b, cQ8b, 256);
      vQ8 += 256;

      // Reduce separately.
      const int cAcc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc4 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 4), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc5 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 5), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc6 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 6), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc7 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m8_i16m1(cProd, 7), __riscv_vmv_v_x_i32m1(0, 1), 32));

      const int cLs0 = ((vX[vIbl].scales_l[0] & 0xf) | ((vH << 4) & 0x30)) - 32;
      const int cLs1 = ((vX[vIbl].scales_l[0] >> 4) | ((vH << 2) & 0x30)) - 32;
      const int cLs2 = ((vX[vIbl].scales_l[1] & 0xf) | ((vH << 0) & 0x30)) - 32;
      const int cLs3 = ((vX[vIbl].scales_l[1] >> 4) | ((vH >> 2) & 0x30)) - 32;
      vH >>= 8;
      const int cLs4 = ((vX[vIbl].scales_l[2] & 0xf) | ((vH << 4) & 0x30)) - 32;
      const int cLs5 = ((vX[vIbl].scales_l[2] >> 4) | ((vH << 2) & 0x30)) - 32;
      const int cLs6 = ((vX[vIbl].scales_l[3] & 0xf) | ((vH << 0) & 0x30)) - 32;
      const int cLs7 = ((vX[vIbl].scales_l[3] >> 4) | ((vH >> 2) & 0x30)) - 32;

      vSumi += cAcc0 * cLs0;
      vSumi += cAcc1 * cLs1;
      vSumi += cAcc2 * cLs2;
      vSumi += cAcc3 * cLs3;
      vSumi += cAcc4 * cLs4;
      vSumi += cAcc5 * cLs5;
      vSumi += cAcc6 * cLs6;
      vSumi += cAcc7 * cLs7;

      __asm__ __volatile__("" ::: "memory");
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d * (vSumi);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotIq4XsQ8KVl1024(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                               const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                               size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const vint8m2_t cValues = __riscv_vle8_v_i8m2(kvalues_iq4nl, 16);
  float vSumf = 0;

  // Indices for re-ordering IQ4 data.
  const uint16_t aIndex[32] = {
      0, 1, 16, 17, 2,  3,  18, 19, 4,  5,  20, 21, 6,  7,  22, 23,
      8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31,
  };
  const vuint16mf2_t cIVec = __riscv_vle16_v_u16mf2(aIndex, 32);

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const int8_t *vQ8 = vY[vIbl].qs;
    const uint8_t *vIq4 = vX[vIbl].qs;
    uint16_t vH = vX[vIbl].scales_h;

    int vSumi = 0;

#pragma GCC unroll 1
    // Process the entire super-block together.
    for (int vIb = 0; vIb < QK_K / 256; ++vIb) {
      // Weights and activations.
      const vuint8m1_t cIq4Packed = __riscv_vle8_v_u8m1(vIq4, 128);
      vIq4 += 128;

      // Unpack the weight blocks.
      const vuint8m1_t cIq4bitsLo = __riscv_vand_vx_u8m1(cIq4Packed, 0xf, 128);
      const vuint8m1_t cIq4bitsHi = __riscv_vsrl_vx_u8m1(cIq4Packed, 4, 128);
      const vuint8m2_t cIq4bits = __riscv_vcreate_v_u8m1_u8m2(cIq4bitsLo, cIq4bitsHi);
      const vuint8m2_t cIq4bitsReorder = __riscv_vreinterpret_v_u64m2_u8m2(
          __riscv_vrgatherei16_vv_u64m2(__riscv_vreinterpret_v_u8m2_u64m2(cIq4bits), cIVec, 32));
      const vint8m2_t cIq4b = __riscv_vrgather_vv_i8m2(cValues, cIq4bitsReorder, 256);

      __asm__ __volatile__("" ::: "memory");

      // Multiply with activations.
      const vint8m2_t cQ8b = __riscv_vle8_v_i8m2(vQ8, 256);
      const vint16m4_t cProd = __riscv_vwmul_vv_i16m4(cIq4b, cQ8b, 256);
      vQ8 += 256;

      // Mask for processing 32 elements per prod register.
      const vuint16m1_t cPIndex = __riscv_vid_v_u16m1(64);
      const vbool16_t cPMask = __riscv_vmsgtu_vx_u16m1_b16(cPIndex, 31, 64);

      // Reduce separately.
      const int cAcc0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(cProd, 0), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(
          cPMask, __riscv_vget_v_i16m4_i16m1(cProd, 0), __riscv_vmv_v_x_i32m1(0, 1), 64));
      const int cAcc2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(cProd, 1), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(
          cPMask, __riscv_vget_v_i16m4_i16m1(cProd, 1), __riscv_vmv_v_x_i32m1(0, 1), 64));
      const int cAcc4 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(cProd, 2), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc5 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(
          cPMask, __riscv_vget_v_i16m4_i16m1(cProd, 2), __riscv_vmv_v_x_i32m1(0, 1), 64));
      const int cAcc6 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(__riscv_vget_v_i16m4_i16m1(cProd, 3), __riscv_vmv_v_x_i32m1(0, 1), 32));
      const int cAcc7 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1_m(
          cPMask, __riscv_vget_v_i16m4_i16m1(cProd, 3), __riscv_vmv_v_x_i32m1(0, 1), 64));

      const int cLs0 = ((vX[vIbl].scales_l[0] & 0xf) | ((vH << 4) & 0x30)) - 32;
      const int cLs1 = ((vX[vIbl].scales_l[0] >> 4) | ((vH << 2) & 0x30)) - 32;
      const int cLs2 = ((vX[vIbl].scales_l[1] & 0xf) | ((vH << 0) & 0x30)) - 32;
      const int cLs3 = ((vX[vIbl].scales_l[1] >> 4) | ((vH >> 2) & 0x30)) - 32;
      vH >>= 8;
      const int cLs4 = ((vX[vIbl].scales_l[2] & 0xf) | ((vH << 4) & 0x30)) - 32;
      const int cLs5 = ((vX[vIbl].scales_l[2] >> 4) | ((vH << 2) & 0x30)) - 32;
      const int cLs6 = ((vX[vIbl].scales_l[3] & 0xf) | ((vH << 0) & 0x30)) - 32;
      const int cLs7 = ((vX[vIbl].scales_l[3] >> 4) | ((vH >> 2) & 0x30)) - 32;

      vSumi += cAcc0 * cLs0;
      vSumi += cAcc1 * cLs1;
      vSumi += cAcc2 * cLs2;
      vSumi += cAcc3 * cLs3;
      vSumi += cAcc4 * cLs4;
      vSumi += cAcc5 * cLs5;
      vSumi += cAcc6 * cLs6;
      vSumi += cAcc7 * cLs7;

      __asm__ __volatile__("" ::: "memory");
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d * (vSumi);
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_iq4_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotIq4XsQ8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotIq4XsQ8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 512:
    fGgmlVecDotIq4XsQ8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 1024:
    fGgmlVecDotIq4XsQ8KVl1024(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default:
    ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotTq10Q8KVl128(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.0f;
  uint8_t aPow[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  for (int vI = 0; vI < cNb; vI++) {
    const uint8_t *GGML_RESTRICT vTq = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    // First loop.
    vint16m4_t vSuml1;
    {
      const int cVl = 32;
      const vuint8m2_t cTqb = __riscv_vle8_v_u8m2(vTq, cVl);
      vTq += 32;

      {
        const vuint16m4_t cTq0 = __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(cTqb, 3, cVl), 8, cVl);
        const vint16m4_t cQ80 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(vQ8, cVl), cVl);
        vSuml1 =
            __riscv_vmul_vv_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(cTq0, 1, cVl)), cQ80, cVl);
        vQ8 += 32;
      }

      uint8_t vPow3 = 3;
#pragma GCC unroll 1
      for (int vT = 0; vT < 4; vT++) {
        const vuint16m4_t cTqn =
            __riscv_vsrl_vx_u16m4(__riscv_vwmulu_vx_u16m4(__riscv_vmul_vx_u8m2(cTqb, vPow3, cVl), 3, cVl), 8, cVl);
        const vint16m4_t cQ8n = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(vQ8, cVl), cVl);
        vSuml1 = __riscv_vmacc_vv_i16m4(vSuml1, __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsub_vx_u16m4(cTqn, 1, cVl)),
                                        cQ8n, cVl);
        vPow3 *= 3;
        vQ8 += 32;
      }
    }

    // Second loop.
    vint16m2_t vSuml2;
    {
      const int cVl = 16;
      const vuint8m1_t cTqb = __riscv_vle8_v_u8m1(vTq, cVl);

      {
        const vuint16m2_t cTq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(cTqb, 3, cVl), 8, cVl);
        const vint16m2_t cQ80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vQ8, cVl), cVl);
        vSuml2 =
            __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(cTq0, 1, cVl)), cQ80, cVl);
        vQ8 += 16;
      }

      uint8_t vPow3 = 3;
#pragma GCC unroll 1
      for (int vT = 0; vT < 4; vT++) {
        const vuint16m2_t cTqn =
            __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(cTqb, vPow3, cVl), 3, cVl), 8, cVl);
        const vint16m2_t cQ8n = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vQ8, cVl), cVl);
        vSuml2 = __riscv_vmacc_vv_i16m2(vSuml2, __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(cTqn, 1, cVl)),
                                        cQ8n, cVl);
        vPow3 *= 3;
        vQ8 += 16;
      }
    }

    // Third loop.
    vint16m2_t vSuml3;
    {
      const int cVl = 16;

      uint32_t vQh;
      memcpy(&vQh, &vX[vI].qh[0], 4);
      // Prevent fusion with vmv.
      __asm__ __volatile__("" : "+r"(vQh));
      const vuint8m1_t cTqb = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vmv_v_x_u32m1(vQh, cVl / 4));

      const vuint8m1_t cP = __riscv_vle8_v_u8m1(aPow, cVl);

      const vuint16m2_t cTq0 =
          __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vv_u8m1(cTqb, cP, cVl), 3, cVl), 8, cVl);

      const vint16m2_t cQ80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vQ8, cVl), cVl);

      vSuml3 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(cTq0, 1, cVl)), cQ80, cVl);
    }

    vint16m2_t vSumb =
        __riscv_vadd_vv_i16m2(__riscv_vget_v_i16m4_i16m2(vSuml1, 0), __riscv_vget_v_i16m4_i16m2(vSuml1, 1), 16);
    vSumb = __riscv_vadd_vv_i16m2(vSumb, vSuml2, 16);
    vSumb = __riscv_vadd_vv_i16m2(vSumb, vSuml3, 16);

    vint32m1_t vSum = __riscv_vwredsum_vs_i16m2_i32m1(vSumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
    vSumf += __riscv_vmv_x_s_i32m1_i32(vSum) * vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotTq10Q8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.0f;
  uint8_t aPow[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  for (int vI = 0; vI < cNb; vI++) {
    // First loop.
    vint16m2_t vSuml1;
    {
      const int cVl = 32;
      vuint8m1_t vTq = __riscv_vle8_v_u8m1(vX[vI].qs, cVl);

      vuint16m2_t vTq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(vTq, 3, cVl), 8, cVl);
      vuint16m2_t vTq1 =
          __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(vTq, 3, cVl), 3, cVl), 8, cVl);
      vuint16m2_t vTq2 =
          __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(vTq, 9, cVl), 3, cVl), 8, cVl);
      vuint16m2_t vTq3 =
          __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(vTq, 27, cVl), 3, cVl), 8, cVl);
      vuint16m2_t vTq4 =
          __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(vTq, 81, cVl), 3, cVl), 8, cVl);

      vint16m2_t vQ80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vY[vI].qs + 0, cVl), cVl);
      vint16m2_t vQ81 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vY[vI].qs + 32, cVl), cVl);
      vint16m2_t vQ82 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vY[vI].qs + 64, cVl), cVl);
      vint16m2_t vQ83 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vY[vI].qs + 96, cVl), cVl);
      vint16m2_t vQ84 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(vY[vI].qs + 128, cVl), cVl);

      vint16m2_t vSum0 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(vTq0, 1, cVl)), vQ80, cVl);
      vint16m2_t vSum1 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(vTq1, 1, cVl)), vQ81, cVl);
      vint16m2_t vSum2 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(vTq2, 1, cVl)), vQ82, cVl);
      vint16m2_t vSum3 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(vTq3, 1, cVl)), vQ83, cVl);
      vint16m2_t vSum4 =
          __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(vTq4, 1, cVl)), vQ84, cVl);

      vint16m2_t vSumi0 = __riscv_vadd_vv_i16m2(vSum0, vSum1, cVl);
      vint16m2_t vSumi1 = __riscv_vadd_vv_i16m2(vSum2, vSum3, cVl);
      vSuml1 = __riscv_vadd_vv_i16m2(vSum4, __riscv_vadd_vv_i16m2(vSumi0, vSumi1, cVl), cVl);
    }

    // Second loop.
    vint16m1_t vSuml2;
    {
      const int cVl = 16;
      vuint8mf2_t vTq = __riscv_vle8_v_u8mf2(vX[vI].qs + 32, cVl);

      vuint16m1_t vTq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(vTq, 3 * 1, cVl), 8, cVl);
      vuint16m1_t vTq1 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 3, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq2 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 9, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq3 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 27, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq4 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 81, cVl), 3, cVl), 8, cVl);

      vint16m1_t vQ80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 160, cVl), cVl);
      vint16m1_t vQ81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 176, cVl), cVl);
      vint16m1_t vQ82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 192, cVl), cVl);
      vint16m1_t vQ83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 208, cVl), cVl);
      vint16m1_t vQ84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 224, cVl), cVl);

      vint16m1_t vSum0 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq0, 1, cVl)), vQ80, cVl);
      vint16m1_t vSum1 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq1, 1, cVl)), vQ81, cVl);
      vint16m1_t vSum2 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq2, 1, cVl)), vQ82, cVl);
      vint16m1_t vSum3 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq3, 1, cVl)), vQ83, cVl);
      vint16m1_t vSum4 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq4, 1, cVl)), vQ84, cVl);

      vint16m1_t vSumi0 = __riscv_vadd_vv_i16m1(vSum0, vSum1, cVl);
      vint16m1_t vSumi1 = __riscv_vadd_vv_i16m1(vSum2, vSum3, cVl);
      vSuml2 = __riscv_vadd_vv_i16m1(vSum4, __riscv_vadd_vv_i16m1(vSumi0, vSumi1, cVl), cVl);
    }

    // Third loop.
    vint16m1_t vSuml3;
    {
      const int cVl = 16;

      uint32_t vQh;
      memcpy(&vQh, &vX[vI].qh[0], 4);
      // Prevent fusion with vmv.
      __asm__ __volatile__("" : "+r"(vQh));
      vuint8mf2_t vTq = __riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(vQh, cVl / 4));

      vuint8mf2_t vP = __riscv_vle8_v_u8mf2(aPow, cVl);

      vuint16m1_t vTq0 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vv_u8mf2(vTq, vP, cVl), 3, cVl), 8, cVl);

      vint16m1_t vQ80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 240, cVl), cVl);

      vSuml3 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq0, 1, cVl)), vQ80, cVl);
    }

    vint16m1_t vSumb =
        __riscv_vadd_vv_i16m1(__riscv_vget_v_i16m2_i16m1(vSuml1, 0), __riscv_vget_v_i16m2_i16m1(vSuml1, 1), 16);
    vSumb = __riscv_vadd_vv_i16m1(vSumb, __riscv_vadd_vv_i16m1(vSuml2, vSuml3, 16), 16);

    vint32m1_t vSum = __riscv_vwredsum_vs_i16m1_i32m1(vSumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
    vSumf += __riscv_vmv_x_s_i32m1_i32(vSum) * vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotTq10Q8KVl512(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.0f;
  uint8_t aPow[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  for (int vI = 0; vI < cNb; vI++) {
    // First loop.
    vint16m1_t vSuml1;
    {
      const int cVl = 32;
      vuint8mf2_t vTq = __riscv_vle8_v_u8mf2(vX[vI].qs, cVl);

      vuint16m1_t vTq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(vTq, 3, cVl), 8, cVl);
      vuint16m1_t vTq1 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 3, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq2 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 9, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq3 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 27, cVl), 3, cVl), 8, cVl);
      vuint16m1_t vTq4 =
          __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(vTq, 81, cVl), 3, cVl), 8, cVl);

      vint16m1_t vQ80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 0, cVl), cVl);
      vint16m1_t vQ81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 32, cVl), cVl);
      vint16m1_t vQ82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 64, cVl), cVl);
      vint16m1_t vQ83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 96, cVl), cVl);
      vint16m1_t vQ84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(vY[vI].qs + 128, cVl), cVl);

      vint16m1_t vSum0 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq0, 1, cVl)), vQ80, cVl);
      vint16m1_t vSum1 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq1, 1, cVl)), vQ81, cVl);
      vint16m1_t vSum2 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq2, 1, cVl)), vQ82, cVl);
      vint16m1_t vSum3 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq3, 1, cVl)), vQ83, cVl);
      vint16m1_t vSum4 =
          __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(vTq4, 1, cVl)), vQ84, cVl);

      vint16m1_t vSumi0 = __riscv_vadd_vv_i16m1(vSum0, vSum1, cVl);
      vint16m1_t vSumi1 = __riscv_vadd_vv_i16m1(vSum2, vSum3, cVl);
      vSuml1 = __riscv_vadd_vv_i16m1(vSum4, __riscv_vadd_vv_i16m1(vSumi0, vSumi1, cVl), cVl);
    }

    // Second loop.
    vint16mf2_t vSuml2;
    {
      const int cVl = 16;
      vuint8mf4_t vTq = __riscv_vle8_v_u8mf4(vX[vI].qs + 32, cVl);

      vuint16mf2_t vTq0 = __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(vTq, 3 * 1, cVl), 8, cVl);
      vuint16mf2_t vTq1 =
          __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(vTq, 3, cVl), 3, cVl), 8, cVl);
      vuint16mf2_t vTq2 =
          __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(vTq, 9, cVl), 3, cVl), 8, cVl);
      vuint16mf2_t vTq3 =
          __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(vTq, 27, cVl), 3, cVl), 8, cVl);
      vuint16mf2_t vTq4 =
          __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vx_u8mf4(vTq, 81, cVl), 3, cVl), 8, cVl);

      vint16mf2_t vQ80 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 160, cVl), cVl);
      vint16mf2_t vQ81 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 176, cVl), cVl);
      vint16mf2_t vQ82 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 192, cVl), cVl);
      vint16mf2_t vQ83 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 208, cVl), cVl);
      vint16mf2_t vQ84 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 224, cVl), cVl);

      vint16mf2_t vSum0 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq0, 1, cVl)), vQ80, cVl);
      vint16mf2_t vSum1 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq1, 1, cVl)), vQ81, cVl);
      vint16mf2_t vSum2 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq2, 1, cVl)), vQ82, cVl);
      vint16mf2_t vSum3 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq3, 1, cVl)), vQ83, cVl);
      vint16mf2_t vSum4 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq4, 1, cVl)), vQ84, cVl);

      vint16mf2_t vSumi0 = __riscv_vadd_vv_i16mf2(vSum0, vSum1, cVl);
      vint16mf2_t vSumi1 = __riscv_vadd_vv_i16mf2(vSum2, vSum3, cVl);
      vSuml2 = __riscv_vadd_vv_i16mf2(vSum4, __riscv_vadd_vv_i16mf2(vSumi0, vSumi1, cVl), cVl);
    }

    // Third loop.
    vint16mf2_t vSuml3;
    {
      const int cVl = 16;

      uint32_t vQh;
      memcpy(&vQh, &vX[vI].qh[0], 4);
      // Prevent fusion with vmv.
      __asm__ __volatile__("" : "+r"(vQh));
      vuint8mf4_t vTq =
          __riscv_vlmul_trunc_v_u8mf2_u8mf4(__riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(vQh, cVl / 4)));

      vuint8mf4_t vP = __riscv_vle8_v_u8mf4(aPow, cVl);

      vuint16mf2_t vTq0 =
          __riscv_vsrl_vx_u16mf2(__riscv_vwmulu_vx_u16mf2(__riscv_vmul_vv_u8mf4(vTq, vP, cVl), 3, cVl), 8, cVl);

      vint16mf2_t vQ80 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(vY[vI].qs + 240, cVl), cVl);

      vSuml3 =
          __riscv_vmul_vv_i16mf2(__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vsub_vx_u16mf2(vTq0, 1, cVl)), vQ80, cVl);
    }

    vint32m1_t vSum = __riscv_vwredsum_vs_i16m1_i32m1(vSuml1, __riscv_vmv_v_x_i32m1(0, 1), 32);
    vSum = __riscv_vwredsum_vs_i16mf2_i32m1(__riscv_vadd_vv_i16mf2(vSuml2, vSuml3, 16), vSum, 16);
    vSumf += __riscv_vmv_x_s_i32m1_i32(vSum) * vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_tq1_0_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotTq10Q8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  case 256:
    fGgmlVecDotTq10Q8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 512 and above
    fGgmlVecDotTq10Q8KVl512(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_tq1_0_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotTq20Q8KVl128(const int pN, float *GGML_RESTRICT pS, size_t pBs,
                                             const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                             size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq2_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;
  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    int32_t vSumi = 0;

    for (size_t vJ = 0; vJ < sizeof(vX[0].qs); vJ += 32) {
      const int8_t *vPy0 = &vY[vI].qs[vJ * 4 + 0 * 32];
      const int8_t *vPy1 = &vY[vI].qs[vJ * 4 + 1 * 32];
      const int8_t *vPy2 = &vY[vI].qs[vJ * 4 + 2 * 32];
      const int8_t *vPy3 = &vY[vI].qs[vJ * 4 + 3 * 32];
      const uint8_t *vPx = &vX[vI].qs[vJ];

      size_t vVl = __riscv_vsetvl_e16m4(32);
      vint16m4_t vVacc16 = __riscv_vmv_v_x_i16m4(0, vVl);

      // Load Raw Packed elements
      vVl = __riscv_vsetvl_e8m2(32);
      vuint8m2_t vVxU8 = __riscv_vle8_v_u8m2(vPx, vVl);

      // Process bits 1:0
      {
        // Unpack
        vuint8m2_t vT0 = __riscv_vand_vx_u8m2(vVxU8, 0x03, vVl);
        vint8m2_t vVq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(vT0), 1, vVl);
        vint8m2_t vVy = __riscv_vle8_v_i8m2(vPy0, vVl);
        // Accumulate
        vVacc16 = __riscv_vwmacc_vv_i16m4(vVacc16, vVq, vVy, vVl);
      }
      __asm__ volatile("" ::: "memory");
      // Process bits 3:2
      {
        vuint8m2_t vT1 = __riscv_vsrl_vx_u8m2(vVxU8, 2, vVl);
        vT1 = __riscv_vand_vx_u8m2(vT1, 0x03, vVl);
        vint8m2_t vVq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(vT1), 1, vVl);

        vint8m2_t vVy = __riscv_vle8_v_i8m2(vPy1, vVl);
        vVacc16 = __riscv_vwmacc_vv_i16m4(vVacc16, vVq, vVy, vVl);
      }
      __asm__ volatile("" ::: "memory");
      // Process bits 5:4
      {
        vuint8m2_t vT2 = __riscv_vsrl_vx_u8m2(vVxU8, 4, vVl);
        vT2 = __riscv_vand_vx_u8m2(vT2, 0x03, vVl);
        vint8m2_t vVq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(vT2), 1, vVl);

        vint8m2_t vVy = __riscv_vle8_v_i8m2(vPy2, vVl);
        vVacc16 = __riscv_vwmacc_vv_i16m4(vVacc16, vVq, vVy, vVl);
      }
      __asm__ volatile("" ::: "memory");
      // Process bits 7:6
      {
        vuint8m2_t vT3 = __riscv_vsrl_vx_u8m2(vVxU8, 6, vVl);
        vint8m2_t vVq = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(vT3), 1, vVl);

        vint8m2_t vVy = __riscv_vle8_v_i8m2(vPy3, vVl);
        vVacc16 = __riscv_vwmacc_vv_i16m4(vVacc16, vVq, vVy, vVl);
      }
      __asm__ volatile("" ::: "memory");
      vVl = __riscv_vsetvl_e16m4(32);
      vint32m1_t vVzero32 = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vVred32 = __riscv_vwredsum_vs_i16m4_i32m1(vVacc16, vVzero32, vVl);
      vSumi += __riscv_vmv_x_s_i32m1_i32(vVred32);
    }

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    vSumf += (float)vSumi * cD;
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotTq20Q8KVl256(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                             size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq2_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.0f;
  for (int vI = 0; vI < cNb; ++vI) {
    int32_t vSumi = 0;

    for (size_t vJ = 0; vJ < sizeof(vX[0].qs); vJ += 32) {
      const int8_t *vPy0 = &vY[vI].qs[vJ * 4 + 0 * 32];
      const int8_t *vPy1 = &vY[vI].qs[vJ * 4 + 1 * 32];
      const int8_t *vPy2 = &vY[vI].qs[vJ * 4 + 2 * 32];
      const int8_t *vPy3 = &vY[vI].qs[vJ * 4 + 3 * 32];
      const uint8_t *vPx = &vX[vI].qs[vJ];

      size_t vVlmax16m2 = __riscv_vsetvl_e16m2(32);
      vint16m2_t vVacc16 = __riscv_vmv_v_x_i16m2(0, vVlmax16m2);

      size_t vVl = __riscv_vsetvl_e8m1(32);

      vuint8m1_t vVxU8 = __riscv_vle8_v_u8m1(vPx, vVl);

      vint8m1_t vVy0 = __riscv_vle8_v_i8m1(vPy0, vVl);
      vint8m1_t vVy1 = __riscv_vle8_v_i8m1(vPy1, vVl);
      vint8m1_t vVy2 = __riscv_vle8_v_i8m1(vPy2, vVl);
      vint8m1_t vVy3 = __riscv_vle8_v_i8m1(vPy3, vVl);

      // l=0 (bits 1:0)
      vuint8m1_t vT0 = __riscv_vand_vx_u8m1(vVxU8, 0x03, vVl);
      vint8m1_t vVq0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vT0), 1, vVl);

      // l=1 (bits 3:2)
      vuint8m1_t vT1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVxU8, 2, vVl), 0x03, vVl);
      vint8m1_t vVq1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vT1), 1, vVl);

      // l=2 (bits 5:4)
      vuint8m1_t vT2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVxU8, 4, vVl), 0x03, vVl);
      vint8m1_t vVq2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vT2), 1, vVl);

      // l=3 (bits 7:6)
      vuint8m1_t vT3 = __riscv_vsrl_vx_u8m1(vVxU8, 6, vVl); // No final AND needed as vsrl shifts in zeros
      vint8m1_t vVq3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vT3), 1, vVl);

      // 4. Multiply and accumulate
      vVacc16 = __riscv_vwmacc_vv_i16m2(vVacc16, vVq0, vVy0, vVl);
      vVacc16 = __riscv_vwmacc_vv_i16m2(vVacc16, vVq1, vVy1, vVl);
      vVacc16 = __riscv_vwmacc_vv_i16m2(vVacc16, vVq2, vVy2, vVl);
      vVacc16 = __riscv_vwmacc_vv_i16m2(vVacc16, vVq3, vVy3, vVl);

      vVlmax16m2 = __riscv_vsetvl_e16m2(32);
      vint32m1_t vVzero32 = __riscv_vmv_v_x_i32m1(0, 1);
      vint32m1_t vVred32 = __riscv_vwredsum_vs_i16m2_i32m1(vVacc16, vVzero32, vVlmax16m2);

      vSumi += __riscv_vmv_x_s_i32m1_i32(vVred32);
    }
    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    vSumf += (float)vSumi * cD;
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_tq2_0_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotTq20Q8KVl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 256 and above
    fGgmlVecDotTq20Q8KVl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_tq2_0_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined __riscv_v
static NOINLINE void fGgmlVecDotMxfp4Q80Vl128(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
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

  // Load the lookup table once.
  const vint8m2_t cValues = __riscv_vle8_v_i8m2(kvalues_mxfp4, 16);
  int vAcc1, vAcc2;

  // We process 2 blocks at once.
  for (; vIb + 1 < cNb; vIb += 2) {
    // Weights and activations.
    vuint8m1_t vMxPacked1 = __riscv_vle8_v_u8m1(vX[vIb + 0].qs, 16);
    vint8m2_t vQ8b1 = __riscv_vle8_v_i8m2(vY[vIb + 0].qs, 32);
    vuint8m1_t vMxPacked2 = __riscv_vle8_v_u8m1(vX[vIb + 1].qs, 16);
    vint8m2_t vQ8b2 = __riscv_vle8_v_i8m2(vY[vIb + 1].qs, 32);

    // Unpack the weight blocks.
    vuint8m2_t vMxbits1 =
        __riscv_vcreate_v_u8m1_u8m2(__riscv_vand_vx_u8m1(vMxPacked1, 0xf, 16), __riscv_vsrl_vx_u8m1(vMxPacked1, 4, 16));
    vuint8m2_t vMxbits2 =
        __riscv_vcreate_v_u8m1_u8m2(__riscv_vand_vx_u8m1(vMxPacked2, 0xf, 16), __riscv_vsrl_vx_u8m1(vMxPacked2, 4, 16));

    // Gather values from the lookup table.
    vint8m2_t vMxb1 = __riscv_vrgather_vv_i8m2(cValues, vMxbits1, 32);
    vint8m2_t vMxb2 = __riscv_vrgather_vv_i8m2(cValues, vMxbits2, 32);

    // Accumulation.
    vint16m4_t vSum1 = __riscv_vwmul_vv_i16m4(vQ8b1, vMxb1, 32);
    vint16m4_t vSum2 = __riscv_vwmul_vv_i16m4(vQ8b2, vMxb2, 32);
    __riscv_vse32_v_i32m1(&vAcc1, __riscv_vwredsum_vs_i16m4_i32m1(vSum1, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    __riscv_vse32_v_i32m1(&vAcc2, __riscv_vwredsum_vs_i16m4_i32m1(vSum2, __riscv_vmv_v_x_i32m1(0, 1), 32), 1);
    vSumf += ((GGML_E8M0_TO_FP32_HALF(vX[vIb + 0].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vAcc1));
    vSumf += ((GGML_E8M0_TO_FP32_HALF(vX[vIb + 1].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vAcc2));
  }

  *pS = vSumf;
}

static NOINLINE void fGgmlVecDotMxfp4Q80Vl256(int pN, float *GGML_RESTRICT pS, size_t pBs,
                                              const void *GGML_RESTRICT pVx, size_t pBx, const void *GGML_RESTRICT pVy,
                                              size_t pBy, int pNrc) {
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

  // Load the lookup table once.
  const vint8mf2_t cValues = __riscv_vle8_v_i8mf2(kvalues_mxfp4, 16);
  int vAcc1, vAcc2;

  // We process 2 blocks at once.
  for (; vIb + 1 < cNb; vIb += 2) {
    // Weights and activations.
    vuint8mf2_t vMxPacked1 = __riscv_vle8_v_u8mf2(vX[vIb + 0].qs, 16);
    vint8mf2_t vQ8bLo1 = __riscv_vle8_v_i8mf2(vY[vIb + 0].qs, 16);
    vint8mf2_t vQ8bHi1 = __riscv_vle8_v_i8mf2(vY[vIb + 0].qs + 16, 16);
    vuint8mf2_t vMxPacked2 = __riscv_vle8_v_u8mf2(vX[vIb + 1].qs, 16);
    vint8mf2_t vQ8bLo2 = __riscv_vle8_v_i8mf2(vY[vIb + 1].qs, 16);
    vint8mf2_t vQ8bHi2 = __riscv_vle8_v_i8mf2(vY[vIb + 1].qs + 16, 16);

    // Unpack the weight blocks.
    vuint8mf2_t vMxbitsLo1 = __riscv_vand_vx_u8mf2(vMxPacked1, 0xf, 16);
    vuint8mf2_t vMxbitsHi1 = __riscv_vsrl_vx_u8mf2(vMxPacked1, 4, 16);
    vuint8mf2_t vMxbitsLo2 = __riscv_vand_vx_u8mf2(vMxPacked2, 0xf, 16);
    vuint8mf2_t vMxbitsHi2 = __riscv_vsrl_vx_u8mf2(vMxPacked2, 4, 16);

    // Gather values from the lookup table.
    vint8mf2_t vMxbLo1 = __riscv_vrgather_vv_i8mf2(cValues, vMxbitsLo1, 16);
    vint8mf2_t vMxbHi1 = __riscv_vrgather_vv_i8mf2(cValues, vMxbitsHi1, 16);
    vint8mf2_t vMxbLo2 = __riscv_vrgather_vv_i8mf2(cValues, vMxbitsLo2, 16);
    vint8mf2_t vMxbHi2 = __riscv_vrgather_vv_i8mf2(cValues, vMxbitsHi2, 16);

    // Accumulation.
    vint16m1_t vSum1 = __riscv_vwmul_vv_i16m1(vQ8bLo1, vMxbLo1, 16);
    vSum1 = __riscv_vwmacc_vv_i16m1(vSum1, vQ8bHi1, vMxbHi1, 16);
    vint16m1_t vSum2 = __riscv_vwmul_vv_i16m1(vQ8bLo2, vMxbLo2, 16);
    vSum2 = __riscv_vwmacc_vv_i16m1(vSum2, vQ8bHi2, vMxbHi2, 16);
    __riscv_vse32_v_i32m1(&vAcc1, __riscv_vwredsum_vs_i16m1_i32m1(vSum1, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    __riscv_vse32_v_i32m1(&vAcc2, __riscv_vwredsum_vs_i16m1_i32m1(vSum2, __riscv_vmv_v_x_i32m1(0, 1), 16), 1);
    vSumf += ((GGML_E8M0_TO_FP32_HALF(vX[vIb + 0].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * vAcc1));
    vSumf += ((GGML_E8M0_TO_FP32_HALF(vX[vIb + 1].e) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * vAcc2));
  }

  *pS = vSumf;
}
#endif

void ggml_vec_dot_mxfp4_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
#if defined __riscv_v
  switch (__riscv_vlenb() * 8) {
  case 128:
    fGgmlVecDotMxfp4Q80Vl128(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  default: // 256 and above
    fGgmlVecDotMxfp4Q80Vl256(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
    break;
  }
#else
  ggml_vec_dot_mxfp4_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
