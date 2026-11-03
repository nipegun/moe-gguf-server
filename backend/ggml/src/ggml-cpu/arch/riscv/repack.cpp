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

void ggml_quantize_mat_q8_0_4x8(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

#if defined(__riscv_v_intrinsic)
  block_q8_0x4 *GGML_RESTRICT vY = (block_q8_0x4 *)pVy;
  const size_t cVlCalc = __riscv_vsetvl_e32m8(QK8_0);
  const size_t cVlSave = __riscv_vsetvl_e64m2(4);
  vfloat32m1_t vVScalarZero = __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));

  for (int vI = 0; vI < cNb; vI++) {
    const float *vXBlockBase = pX + vI * QK8_0;
    vint8m2_t vQR0, vQR1, vQR2, vQR3;
    {
      vfloat32m8_t vVSrc = __riscv_vle32_v_f32m8(vXBlockBase + 0 * pK, cVlCalc);
      vfloat32m8_t vVAbs = __riscv_vfabs_v_f32m8(vVSrc, cVlCalc);
      vfloat32m1_t vVMax = __riscv_vfredmax_vs_f32m8_f32m1(vVAbs, vVScalarZero, cVlCalc);
      float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVMax);

      float vD = vAmax / 127.0f;
      vY[vI].d[0] = GGML_CPU_FP32_TO_FP16(vD);

      float vId = vD ? 1.0f / vD : 0.0f;
      vfloat32m8_t vVScaled = __riscv_vfmul_vf_f32m8(vVSrc, vId, cVlCalc);
      vint16m4_t vVI16 = __riscv_vfncvt_x_f_w_i16m4_rm(vVScaled, 4, cVlCalc);
      vQR0 = __riscv_vncvt_x_x_w_i8m2(vVI16, cVlCalc);
    }
    asm volatile("" ::: "memory");

    {
      vfloat32m8_t vVSrc = __riscv_vle32_v_f32m8(vXBlockBase + 1 * pK, cVlCalc);
      vfloat32m8_t vVAbs = __riscv_vfabs_v_f32m8(vVSrc, cVlCalc);
      vfloat32m1_t vVMax = __riscv_vfredmax_vs_f32m8_f32m1(vVAbs, vVScalarZero, cVlCalc);
      float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVMax);

      float vD = vAmax / 127.0f;
      vY[vI].d[1] = GGML_CPU_FP32_TO_FP16(vD);
      float vId = vD ? 1.0f / vD : 0.0f;

      vfloat32m8_t vVScaled = __riscv_vfmul_vf_f32m8(vVSrc, vId, cVlCalc);
      vint16m4_t vVI16 = __riscv_vfncvt_x_f_w_i16m4_rm(vVScaled, 4, cVlCalc);
      vQR1 = __riscv_vncvt_x_x_w_i8m2(vVI16, cVlCalc);
    }
    asm volatile("" ::: "memory");
    {
      vfloat32m8_t vVSrc = __riscv_vle32_v_f32m8(vXBlockBase + 2 * pK, cVlCalc);
      vfloat32m8_t vVAbs = __riscv_vfabs_v_f32m8(vVSrc, cVlCalc);
      vfloat32m1_t vVMax = __riscv_vfredmax_vs_f32m8_f32m1(vVAbs, vVScalarZero, cVlCalc);
      float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVMax);

      float vD = vAmax / 127.0f;
      vY[vI].d[2] = GGML_CPU_FP32_TO_FP16(vD);
      float vId = vD ? 1.0f / vD : 0.0f;

      vfloat32m8_t vVScaled = __riscv_vfmul_vf_f32m8(vVSrc, vId, cVlCalc);
      vint16m4_t vVI16 = __riscv_vfncvt_x_f_w_i16m4_rm(vVScaled, 4, cVlCalc);
      vQR2 = __riscv_vncvt_x_x_w_i8m2(vVI16, cVlCalc);
    }
    asm volatile("" ::: "memory");
    {
      vfloat32m8_t vVSrc = __riscv_vle32_v_f32m8(vXBlockBase + 3 * pK, cVlCalc);
      vfloat32m8_t vVAbs = __riscv_vfabs_v_f32m8(vVSrc, cVlCalc);
      vfloat32m1_t vVMax = __riscv_vfredmax_vs_f32m8_f32m1(vVAbs, vVScalarZero, cVlCalc);
      float vAmax = __riscv_vfmv_f_s_f32m1_f32(vVMax);

      float vD = vAmax / 127.0f;
      vY[vI].d[3] = GGML_CPU_FP32_TO_FP16(vD);
      float vId = vD ? 1.0f / vD : 0.0f;

      vfloat32m8_t vVScaled = __riscv_vfmul_vf_f32m8(vVSrc, vId, cVlCalc);
      vint16m4_t vVI16 = __riscv_vfncvt_x_f_w_i16m4_rm(vVScaled, 4, cVlCalc);
      vQR3 = __riscv_vncvt_x_x_w_i8m2(vVI16, cVlCalc);
    }
    vint64m2_t vVQ64R0 = __riscv_vreinterpret_v_i8m2_i64m2(vQR0);
    vint64m2_t vVQ64R1 = __riscv_vreinterpret_v_i8m2_i64m2(vQR1);
    vint64m2_t vVQ64R2 = __riscv_vreinterpret_v_i8m2_i64m2(vQR2);
    vint64m2_t vVQ64R3 = __riscv_vreinterpret_v_i8m2_i64m2(vQR3);
    vint64m2x4_t vVQuantTuple = __riscv_vcreate_v_i64m2x4(vVQ64R0, vVQ64R1, vVQ64R2, vVQ64R3);
    __riscv_vsseg4e64_v_i64m2x4((int64_t *)vY[vI].qs, vVQuantTuple, cVlSave);
  }
#else
  UNUSED(cNb);
  ggml_quantize_mat_q8_0_4x8_generic(pX, pVy, pK);
#endif
}

void ggml_gemv_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
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

#if defined __riscv_v
  if (__riscv_vlenb() >= QK4_0) {
    const size_t cVl = QK4_0;

    const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_0x8 *vBPtr = (const block_q4_0x8 *)pVx + (vX * cNb);

      vfloat32m1_t vSumf = __riscv_vfmv_v_f_f32m1(0.0, cVl / 4);
      for (int vL = 0; vL < cNb; vL++) {
        const int64_t cA0 = *(const int64_t *)&vAPtr[vL].qs[0];
        const int64_t cA1 = *(const int64_t *)&vAPtr[vL].qs[8];
        const int64_t cA2 = *(const int64_t *)&vAPtr[vL].qs[16];
        const int64_t cA3 = *(const int64_t *)&vAPtr[vL].qs[24];
        __asm__ __volatile__("" ::
                                 : "memory"); // prevent gcc from emitting fused vlse64, violating alignment constraints
        const vint8m2_t cLhs08 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA0, cVl / 4));
        const vint8m2_t cLhs18 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA1, cVl / 4));
        const vint8m2_t cLhs28 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA2, cVl / 4));
        const vint8m2_t cLhs38 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA3, cVl / 4));

        const vint8m4_t cRhsRawVec = __riscv_vle8_v_i8m4((const int8_t *)vBPtr[vL].qs, cVl * 4);
        const vint8m4_t cRhsVecLo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(cRhsRawVec, 4, cVl * 4), 4, cVl * 4);
        const vint8m4_t cRhsVecHi = __riscv_vsra_vx_i8m4(cRhsRawVec, 4, cVl * 4);
        const vint8m2_t cRhsVecLo0 = __riscv_vget_v_i8m4_i8m2(cRhsVecLo, 0);
        const vint8m2_t cRhsVecLo1 = __riscv_vget_v_i8m4_i8m2(cRhsVecLo, 1);
        const vint8m2_t cRhsVecHi0 = __riscv_vget_v_i8m4_i8m2(cRhsVecHi, 0);
        const vint8m2_t cRhsVecHi1 = __riscv_vget_v_i8m4_i8m2(cRhsVecHi, 1);

        const vint16m4_t cSumiLo0 = __riscv_vwmul_vv_i16m4(cRhsVecLo0, cLhs08, cVl * 2);
        const vint16m4_t cSumiLo1 = __riscv_vwmacc_vv_i16m4(cSumiLo0, cRhsVecLo1, cLhs18, cVl * 2);
        const vint16m4_t cSumiHi0 = __riscv_vwmacc_vv_i16m4(cSumiLo1, cRhsVecHi0, cLhs28, cVl * 2);
        const vint16m4_t cSumiHiM = __riscv_vwmacc_vv_i16m4(cSumiHi0, cRhsVecHi1, cLhs38, cVl * 2);

        const vuint32m4_t cSumiI32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(cSumiHiM));
        const vuint16m2_t cSumiH20 = __riscv_vnsrl_wx_u16m2(cSumiI32, 0, cVl);
        const vuint16m2_t cSumiH21 = __riscv_vnsrl_wx_u16m2(cSumiI32, 16, cVl);
        const vuint16m2_t cSumiH2 = __riscv_vadd_vv_u16m2(cSumiH20, cSumiH21, cVl);
        const vuint32m2_t cSumiH2I32 = __riscv_vreinterpret_v_u16m2_u32m2(cSumiH2);
        const vuint16m1_t cSumiH40 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 0, cVl / 2);
        const vuint16m1_t cSumiH41 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 16, cVl / 2);
        const vuint16m1_t cSumiH4 = __riscv_vadd_vv_u16m1(cSumiH40, cSumiH41, cVl / 2);
        const vuint32m1_t cSumiH4I32 = __riscv_vreinterpret_v_u16m1_u32m1(cSumiH4);
        const vint16mf2_t cSumiH80 =
            __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 0, cVl / 4));
        const vint16mf2_t cSumiH81 =
            __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 16, cVl / 4));
        const vint32m1_t cSumiH8 = __riscv_vwadd_vv_i32m1(cSumiH80, cSumiH81, cVl / 4);
        const vfloat32m1_t cFacc = __riscv_vfcvt_f_x_v_f32m1(cSumiH8, cVl / 4);

        // vector version needs Zvfhmin extension
        const float cAScale = GGML_CPU_FP16_TO_FP32(vAPtr[vL].d);
        const float aBScales[8] = {GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[0]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[1]),
                                   GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[2]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[3]),
                                   GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[4]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[5]),
                                   GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[6]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[7])};
        const vfloat32m1_t cBScalesVec = __riscv_vle32_v_f32m1(aBScales, cVl / 4);
        const vfloat32m1_t cTmp1 = __riscv_vfmul_vf_f32m1(cFacc, cAScale, cVl / 4);
        vSumf = __riscv_vfmacc_vv_f32m1(vSumf, cTmp1, cBScalesVec, cVl / 4);
      }
      __riscv_vse32_v_f32m1(pS + vX * cNcolsInterleaved, vSumf, cVl / 4);
    }
    return;
  }

#endif
  ggml_gemv_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_0x16 *vBPtr = (const block_q4_0x16 *)pVx + (vX * cNb);

    // 1x16 Accumulator
    vfloat32m2_t vSumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

    for (int vL = 0; vL < cNb; vL++) {
      // 1x16 Integer Accumulator
      vint16m1_t vSumi0Lo16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
      vint16m1_t vSumi0Hi16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

      // Accumulation loop.
      for (int vI = 0; vI < QK4_0 / 2; vI++) {
        // Load `b_ptr`.
        const vint8mf2_t cB0Packed = __riscv_vle8_v_i8mf2((const int8_t *)&vBPtr[vL].qs[vI * 16], 16);
        const vint8mf2_t cB0Lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(cB0Packed, 4, 16), 4, 16);
        const vint8mf2_t cB0Hi = __riscv_vsra_vx_i8mf2(cB0Packed, 4, 16);

        vSumi0Lo16 = __riscv_vwmacc_vx_i16m1(vSumi0Lo16, vAPtr[vL].qs[vI], cB0Lo, 16);
        vSumi0Hi16 = __riscv_vwmacc_vx_i16m1(vSumi0Hi16, vAPtr[vL].qs[16 + vI], cB0Hi, 16);
      }

      const vint32m2_t cSumi = __riscv_vwadd_vv_i32m2(vSumi0Lo16, vSumi0Hi16, 16);

      const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
      const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d, 16);

      vSumf = __riscv_vfmacc_vv_f32m2(vSumf, __riscv_vfcvt_f_x_v_f32m2(cSumi, 16), cD0, 16);
    }

    __riscv_vse32_v_f32m2(pS + vX * 16, vSumf, 16);
  }
}

void ggml_gemv_q4_K_16x1_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  const block_q8_K *vAPtr = (const block_q8_K *)pVy;

  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q4_Kx16 *vBPtr = (const block_q4_Kx16 *)pVx + (vX * cNb);

    // 1x16 Accumulator
    vfloat32m2_t vSumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

    for (int vL = 0; vL < cNb; vL++) {
      vint32m2_t vSumi = __riscv_vmv_v_x_i32m2(0, 16);

      // Load `dmin`.
      const vfloat32m2_t cDminsD = __riscv_vfmul_vf_f32m2(
          __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].dmin, 16), 16), vAPtr[vL].d, 16);

      // We process 4 sub-blocks at once.
      for (int vJ = 0; vJ < QK_K / 128; vJ++) {
        // Extract the scales and the mins.
        //
        // Low bits.
        vuint8m2_t vScalesMinsLo = __riscv_vle8_v_u8m2(&vBPtr[vL].scales[vJ * 64], 64);
        vuint8m2_t vScalesLo = __riscv_vand_vx_u8m2(vScalesMinsLo, 0x0F, 64);
        vuint8m2_t vMinsLo = __riscv_vsrl_vx_u8m2(vScalesMinsLo, 4, 64);

        // High bits.
        vuint8m2_t vScalesMinsHi = __riscv_vle8_v_u8m2(&vBPtr[vL].scales[128], 64);
        vuint8m2_t vScalesHi;
        vuint8m2_t vMinsHi;
        if (!vJ) {
          vScalesHi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0x03, 64), 4, 64);
          vMinsHi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0x0C, 64), 2, 64);
        } else {
          vScalesHi = __riscv_vand_vx_u8m2(vScalesMinsHi, 0x30, 64);
          vMinsHi = __riscv_vsrl_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0xC0, 64), 2, 64);
        }
        vuint16m4_t vScales = __riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(vScalesHi, vScalesLo, 64), 64);
        vint16m4_t vMins =
            __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(vMinsHi, vMinsLo, 64), 64));

        // Reduce the mins and multiply with `dmin`.
        //
        // Correct in `sumf`.
        vint32m2_t vBsums = __riscv_vmv_v_x_i32m2(0, 16);
        vBsums = __riscv_vwmacc_vx_i32m2(vBsums, vAPtr[vL].bsums[vJ * 8] + vAPtr[vL].bsums[vJ * 8 + 1],
                                         __riscv_vget_v_i16m4_i16m1(vMins, 0), 16);
        vBsums = __riscv_vwmacc_vx_i32m2(vBsums, vAPtr[vL].bsums[vJ * 8 + 2] + vAPtr[vL].bsums[vJ * 8 + 3],
                                         __riscv_vget_v_i16m4_i16m1(vMins, 1), 16);
        vBsums = __riscv_vwmacc_vx_i32m2(vBsums, vAPtr[vL].bsums[vJ * 8 + 4] + vAPtr[vL].bsums[vJ * 8 + 5],
                                         __riscv_vget_v_i16m4_i16m1(vMins, 2), 16);
        vBsums = __riscv_vwmacc_vx_i32m2(vBsums, vAPtr[vL].bsums[vJ * 8 + 6] + vAPtr[vL].bsums[vJ * 8 + 7],
                                         __riscv_vget_v_i16m4_i16m1(vMins, 3), 16);

        vSumf = __riscv_vfsub_vv_f32m2(vSumf,
                                       __riscv_vfmul_vv_f32m2(cDminsD, __riscv_vfcvt_f_x_v_f32m2(vBsums, 16), 16), 16);

        // Accumulation for 2 sub-blocks.
        //
        // This might overflow, so we accumulate in two steps.
        //
        // Recheck.
        for (int vK = 0; vK < 2; vK++) {
          vint16m1_t vSumiS016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
          vint16m1_t vSumiS116 = __riscv_vmv_v_x_i16m1(0.0f, 16);

          for (int vI = vK * 16; vI < vK * 16 + QK4_0 / 2; vI++) {
            // Load `b_ptr`.
            const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2(&vBPtr[vL].qs[vJ * 1024 + vI * 16], 16);
            const vint8mf2_t cBS0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(cB0Packed, 0xF, 16));
            const vint8mf2_t cBS1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16));

            vSumiS016 = __riscv_vwmacc_vx_i16m1(vSumiS016, vAPtr[vL].qs[vJ * 128 + vI], cBS0, 16);
            vSumiS116 = __riscv_vwmacc_vx_i16m1(vSumiS116, vAPtr[vL].qs[vJ * 128 + 32 + vI], cBS1, 16);
          }

          vSumi = __riscv_vwmacc_vv_i32m2(
              vSumi, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 0)), vSumiS016, 16);
          vSumi = __riscv_vwmacc_vv_i32m2(
              vSumi, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 1)), vSumiS116, 16);
        }
        // Accumulation for 2 sub-blocks.
        //
        // This might overflow, so we accumulate in two steps.
        //
        // Recheck.
        for (int vK = 0; vK < 2; vK++) {
          vint16m1_t vSumiS016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
          vint16m1_t vSumiS116 = __riscv_vmv_v_x_i16m1(0.0f, 16);

          for (int vI = vK * 16; vI < vK * 16 + QK4_0 / 2; vI++) {
            // Load `b_ptr`.
            const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2(&vBPtr[vL].qs[vJ * 1024 + 512 + vI * 16], 16);
            const vint8mf2_t cBS0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(cB0Packed, 0xF, 16));
            const vint8mf2_t cBS1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16));

            vSumiS016 = __riscv_vwmacc_vx_i16m1(vSumiS016, vAPtr[vL].qs[vJ * 128 + 64 + vI], cBS0, 16);
            vSumiS116 = __riscv_vwmacc_vx_i16m1(vSumiS116, vAPtr[vL].qs[vJ * 128 + 96 + vI], cBS1, 16);
          }

          vSumi = __riscv_vwmacc_vv_i32m2(
              vSumi, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 2)), vSumiS016, 16);
          vSumi = __riscv_vwmacc_vv_i32m2(
              vSumi, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 3)), vSumiS116, 16);
        }
      }

      const vfloat32m2_t cBD =
          __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16 *)&vBPtr[vL].d[0], 16), 16);
      const vfloat32m2_t cD0 = __riscv_vfmul_vf_f32m2(cBD, vAPtr[vL].d, 16);

      vSumf = __riscv_vfmacc_vv_f32m2(vSumf, __riscv_vfcvt_f_x_v_f32m2(vSumi, 16), cD0, 16);
    }

    __riscv_vse32_v_f32m2(pS + vX * 16, vSumf, 16);
  }
}

void ggml_gemv_iq4_nl_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  const vint8mf2_t cValues = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_iq4_nlx16 *vBPtr = (const block_iq4_nlx16 *)pVx + (vX * cNb);

    // 1x16 Accumulator1
    vfloat32m2_t vSumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

    for (int vL = 0; vL < cNb; vL++) {
      // 1x16 integer accumulator
      vint32m2_t vSumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

      // Accumulation loop.
      for (int vI = 0; vI < QK4_NL / 2; vI++) {
        // Load `b_ptr`.
        const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2((const uint8_t *)&vBPtr[vL].qs[vI * 16], 16);
        const vint8mf2_t cB0Lo = __riscv_vrgather_vv_i8mf2(cValues, __riscv_vand_vx_u8mf2(cB0Packed, 0xf, 16), 16);
        const vint8mf2_t cB0Hi = __riscv_vrgather_vv_i8mf2(cValues, __riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16), 16);
        // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
        // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

        const vint16m1_t cSumiLo = __riscv_vwmul_vx_i16m1(cB0Lo, vAPtr[vL].qs[vI], 16);
        const vint16m1_t cSumiHi = __riscv_vwmul_vx_i16m1(cB0Hi, vAPtr[vL].qs[16 + vI], 16);
        vSumi = __riscv_vadd_vv_i32m2(vSumi, __riscv_vwadd_vv_i32m2(cSumiLo, cSumiHi, 16), 16);
      }

      const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
      const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d, 16);

      vSumf = __riscv_vfmacc_vv_f32m2(vSumf, __riscv_vfcvt_f_x_v_f32m2(vSumi, 16), cD0, 16);
    }

    __riscv_vse32_v_f32m2(pS + vX * 16, vSumf, 16);
  }
}

void ggml_gemv_q8_0_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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
  UNUSED(pBs);

  const block_q8_0 *vAPtr = (const block_q8_0 *)pVy;
  for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
    const block_q8_0x16 *vBPtr = (const block_q8_0x16 *)pVx + (vX * cNb);

    // 1x16 Accumulator
    vfloat32m2_t vSumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

    for (int vL = 0; vL < cNb; vL++) {
      // 1x16 Integer Accumulator
      vint32m2_t vSumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

      // Accumulation loop.
      for (int vI = 0; vI < QK8_0; vI++) {
        // Load `b_ptr`.
        const vint8mf2_t cB0 = __riscv_vle8_v_i8mf2((const int8_t *)&vBPtr[vL].qs[vI * 16], 16);
        // const vint16m1_t b_0_16 = __riscv_vwcvt_x_x_v_i16m1(b_0, 16);

        vSumi = __riscv_vwadd_wv_i32m2(vSumi, __riscv_vwmul_vx_i16m1(cB0, vAPtr[vL].qs[vI], 16), 16);
      }

      const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
      const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d, 16);

      vSumf = __riscv_vfmacc_vv_f32m2(vSumf, __riscv_vfcvt_f_x_v_f32m2(vSumi, 16), cD0, 16);
    }

    __riscv_vse32_v_f32m2(pS + vX * 16, vSumf, 16);
  }
}

void ggml_gemv_q2_K_16x1_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  assert(pN % QK_K == 0);
  assert(pNr == 1);
  assert(pNc % 16 == 0);

  UNUSED(pBs);

  const int cNColsTile = 16;
  const int cNumKBlocks = pN / QK_K;

  const size_t cVl = __riscv_vsetvl_e32m2(cNColsTile);
  for (int vColTile = 0; vColTile < pNc; vColTile += cNColsTile) {

    const block_q8_K *vLhsBasePtr = (const block_q8_K *)pVy;
    const block_q2_Kx16 *vRhsBasePtr = (const block_q2_Kx16 *)pVx + (vColTile / cNColsTile) * cNumKBlocks;

    vfloat32m2_t vVSumf = __riscv_vfmv_v_f_f32m2(0.0f, cVl);

    for (int vKBlock = 0; vKBlock < cNumKBlocks; ++vKBlock) {
      const block_q8_K *vLhsCurrent = &vLhsBasePtr[vKBlock];
      const block_q2_Kx16 *vRhsCurrent = &vRhsBasePtr[vKBlock];

      // 1. Prepare Global Min Scales
      vfloat16m1_t vVGMinF16 = __riscv_vle16_v_f16m1((const _Float16 *)vRhsCurrent->dmin, cVl);
      vfloat32m2_t vVGMinBase = __riscv_vfwcvt_f_f_v_f32m2(vVGMinF16, cVl);

      vfloat32m2_t vVGMinFinal = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d, cVl);

      vint32m2_t vVIsum = __riscv_vmv_v_x_i32m2(0, cVl);

      const uint8_t *vRhsQsPtr = vRhsCurrent->qs;
      const uint8_t *vRhsScPtr = vRhsCurrent->scales;
      const int8_t *vLhsQsPtr = vLhsCurrent->qs;

      // --- Phase Loop (4 phases x 64 elements) ---
      for (int vPhase = 0; vPhase < 4; ++vPhase) {

        // A. Load Scales/Mins
        vuint16m1_t vVDSb0, vVDSb1, vVDSb2, vVDSb3;
        vuint16m1_t vVMSb0, vVMSb1, vVMSb2, vVMSb3;

        {
          vuint8mf2_t vVRaw;
          // Sub-block 0
          vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 0, cVl);
          vVDSb0 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
          vVMSb0 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

          // Sub-block 1
          vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 16, cVl);
          vVDSb1 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
          vVMSb1 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

          // Sub-block 2
          vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 32, cVl);
          vVDSb2 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
          vVMSb2 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

          // Sub-block 3
          vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 48, cVl);
          vVDSb3 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
          vVMSb3 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

          vRhsScPtr += 64;
        }

        int vBaseKPhase = (vPhase < 2) ? (vPhase * 16) : (128 + (vPhase - 2) * 16);
        int aKOffsets[4] = {0, 32, 64, 96};

        // B. Inner Dot Product Loop
        for (int vL = 0; vL < 16; ++vL) {
          vuint8mf2_t vVRhsData = __riscv_vle8_v_u8mf2(vRhsQsPtr, cVl);
          vRhsQsPtr += 16;

          // Sub-block 0
          {
            vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(vVRhsData, 3, cVl);
            vint16m1_t vVW =
                __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                      __riscv_vreinterpret_v_u16m1_i16m1(vVDSb0), cVl);

            int8_t vQ8 = vLhsQsPtr[vBaseKPhase + aKOffsets[0] + vL];
            vVIsum = __riscv_vwmacc_vx_i32m2(vVIsum, (int16_t)vQ8, vVW, cVl);
          }
          // Sub-block 1
          {
            vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 2, cVl), 3, cVl);
            vint16m1_t vVW =
                __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                      __riscv_vreinterpret_v_u16m1_i16m1(vVDSb1), cVl);

            int8_t vQ8 = vLhsQsPtr[vBaseKPhase + aKOffsets[1] + vL];
            vVIsum = __riscv_vwmacc_vx_i32m2(vVIsum, (int16_t)vQ8, vVW, cVl);
          }
          // Sub-block 2
          {
            vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 4, cVl), 3, cVl);
            vint16m1_t vVW =
                __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                      __riscv_vreinterpret_v_u16m1_i16m1(vVDSb2), cVl);

            int8_t vQ8 = vLhsQsPtr[vBaseKPhase + aKOffsets[2] + vL];
            vVIsum = __riscv_vwmacc_vx_i32m2(vVIsum, (int16_t)vQ8, vVW, cVl);
          }
          // Sub-block 3
          {
            vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 6, cVl), 3, cVl);
            vint16m1_t vVW =
                __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                      __riscv_vreinterpret_v_u16m1_i16m1(vVDSb3), cVl);

            int8_t vQ8 = vLhsQsPtr[vBaseKPhase + aKOffsets[3] + vL];
            vVIsum = __riscv_vwmacc_vx_i32m2(vVIsum, (int16_t)vQ8, vVW, cVl);
          }
        }

        // correction
        int vSbBaseAbs = vBaseKPhase / 16;

        // Sub-block 0
        {
          int vSbIdx = vSbBaseAbs + (aKOffsets[0] / 16);
          int16_t vBsum = vLhsCurrent->bsums[vSbIdx];
          vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb0);
          vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vBsum, cVl);
          vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMinFinal, cVl);
          vVSumf = __riscv_vfsub_vv_f32m2(vVSumf, vVfC, cVl);
        }
        // Sub-block 1
        {
          int vSbIdx = vSbBaseAbs + (aKOffsets[1] / 16);
          int16_t vBsum = vLhsCurrent->bsums[vSbIdx];
          vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb1);
          vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vBsum, cVl);
          vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMinFinal, cVl);
          vVSumf = __riscv_vfsub_vv_f32m2(vVSumf, vVfC, cVl);
        }
        // Sub-block 2
        {
          int vSbIdx = vSbBaseAbs + (aKOffsets[2] / 16);
          int16_t vBsum = vLhsCurrent->bsums[vSbIdx];
          vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb2);
          vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vBsum, cVl);
          vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMinFinal, cVl);
          vVSumf = __riscv_vfsub_vv_f32m2(vVSumf, vVfC, cVl);
        }
        // Sub-block 3
        {
          int vSbIdx = vSbBaseAbs + (aKOffsets[3] / 16);
          int16_t vBsum = vLhsCurrent->bsums[vSbIdx];
          vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb3);
          vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vBsum, cVl);
          vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMinFinal, cVl);
          vVSumf = __riscv_vfsub_vv_f32m2(vVSumf, vVfC, cVl);
        }

      } // End Phase Loop

      // Apply global Scales
      vfloat16m1_t vVGAllF16 = __riscv_vle16_v_f16m1((const _Float16 *)vRhsCurrent->d, cVl);
      vfloat32m2_t vVGAllBase = __riscv_vfwcvt_f_f_v_f32m2(vVGAllF16, cVl);

      vfloat32m2_t vVGAllFinal = __riscv_vfmul_vf_f32m2(vVGAllBase, vLhsCurrent->d, cVl);
      vfloat32m2_t vVSum = __riscv_vfcvt_f_x_v_f32m2(vVIsum, cVl);
      vVSum = __riscv_vfmul_vv_f32m2(vVSum, vVGAllFinal, cVl);
      vVSumf = __riscv_vfadd_vv_f32m2(vVSumf, vVSum, cVl);

    } // End K-Block
    __riscv_vse32_v_f32m2(pS + vColTile, vVSumf, cVl);
  }
}
#endif

void ggml_gemm_q4_0_8x8_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                             const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 8;
  const int cBlocklen = 8;

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

#if defined __riscv_v
  if (__riscv_vlenb() >= QK4_0) {
    const size_t cVl = QK4_0;

    for (int vY = 0; vY < pNr / 4; vY++) {
      const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
      for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
        const block_q4_0x8 *vBPtr = (const block_q4_0x8 *)pVx + (vX * cNb);
        vfloat32m1_t vSumf0 = __riscv_vfmv_v_f_f32m1(0.0, cVl / 4);
        vfloat32m1_t vSumf1 = __riscv_vfmv_v_f_f32m1(0.0, cVl / 4);
        vfloat32m1_t vSumf2 = __riscv_vfmv_v_f_f32m1(0.0, cVl / 4);
        vfloat32m1_t vSumf3 = __riscv_vfmv_v_f_f32m1(0.0, cVl / 4);
        for (int vL = 0; vL < cNb; vL++) {
          const vint8m4_t cRhsRawVec = __riscv_vle8_v_i8m4((const int8_t *)vBPtr[vL].qs, cVl * 4);
          const vint8m4_t cRhsVecLo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(cRhsRawVec, 4, cVl * 4), 4, cVl * 4);
          const vint8m4_t cRhsVecHi = __riscv_vsra_vx_i8m4(cRhsRawVec, 4, cVl * 4);
          const vint8m2_t cRhsVecLo0 = __riscv_vget_v_i8m4_i8m2(cRhsVecLo, 0);
          const vint8m2_t cRhsVecLo1 = __riscv_vget_v_i8m4_i8m2(cRhsVecLo, 1);
          const vint8m2_t cRhsVecHi0 = __riscv_vget_v_i8m4_i8m2(cRhsVecHi, 0);
          const vint8m2_t cRhsVecHi1 = __riscv_vget_v_i8m4_i8m2(cRhsVecHi, 1);

          // vector version needs Zvfhmin extension
          const float aAScales[4] = {GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[0]), GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[1]),
                                     GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[2]), GGML_CPU_FP16_TO_FP32(vAPtr[vL].d[3])};
          const float aBScales[8] = {GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[0]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[1]),
                                     GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[2]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[3]),
                                     GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[4]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[5]),
                                     GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[6]), GGML_CPU_FP16_TO_FP32(vBPtr[vL].d[7])};
          const vfloat32m1_t cBScalesVec = __riscv_vle32_v_f32m1(aBScales, cVl / 4);

          const int64_t cA0 = *(const int64_t *)&vAPtr[vL].qs[0];
          const int64_t cA4 = *(const int64_t *)&vAPtr[vL].qs[32];
          const int64_t cA8 = *(const int64_t *)&vAPtr[vL].qs[64];
          const int64_t cAc = *(const int64_t *)&vAPtr[vL].qs[96];
          __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
          vint16m4_t vSumiL0;
          {
            const vint8m2_t cLhs08 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA0, cVl / 4));
            const vint8m2_t cLhs18 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA4, cVl / 4));
            const vint8m2_t cLhs28 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA8, cVl / 4));
            const vint8m2_t cLhs38 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAc, cVl / 4));
            const vint16m4_t cSumiLo0 = __riscv_vwmul_vv_i16m4(cRhsVecLo0, cLhs08, cVl * 2);
            const vint16m4_t cSumiLo1 = __riscv_vwmacc_vv_i16m4(cSumiLo0, cRhsVecLo1, cLhs18, cVl * 2);
            const vint16m4_t cSumiHi0 = __riscv_vwmacc_vv_i16m4(cSumiLo1, cRhsVecHi0, cLhs28, cVl * 2);
            const vint16m4_t cSumiHiM = __riscv_vwmacc_vv_i16m4(cSumiHi0, cRhsVecHi1, cLhs38, cVl * 2);

            vSumiL0 = cSumiHiM;
          }

          {
            const vuint32m4_t cSumiI32 =
                __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(vSumiL0));
            const vuint16m2_t cSumiH20 = __riscv_vnsrl_wx_u16m2(cSumiI32, 0, cVl);
            const vuint16m2_t cSumiH21 = __riscv_vnsrl_wx_u16m2(cSumiI32, 16, cVl);
            const vuint16m2_t cSumiH2 = __riscv_vadd_vv_u16m2(cSumiH20, cSumiH21, cVl);
            const vuint32m2_t cSumiH2I32 = __riscv_vreinterpret_v_u16m2_u32m2(cSumiH2);
            const vuint16m1_t cSumiH40 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 0, cVl / 2);
            const vuint16m1_t cSumiH41 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 16, cVl / 2);
            const vuint16m1_t cSumiH4 = __riscv_vadd_vv_u16m1(cSumiH40, cSumiH41, cVl / 2);
            const vuint32m1_t cSumiH4I32 = __riscv_vreinterpret_v_u16m1_u32m1(cSumiH4);
            const vint16mf2_t cSumiH80 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 0, cVl / 4));
            const vint16mf2_t cSumiH81 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 16, cVl / 4));
            const vint32m1_t cSumiH8 = __riscv_vwadd_vv_i32m1(cSumiH80, cSumiH81, cVl / 4);
            const vfloat32m1_t cFacc = __riscv_vfcvt_f_x_v_f32m1(cSumiH8, cVl / 4);

            const vfloat32m1_t cTmp1 = __riscv_vfmul_vf_f32m1(cFacc, aAScales[0], cVl / 4);
            vSumf0 = __riscv_vfmacc_vv_f32m1(vSumf0, cTmp1, cBScalesVec, cVl / 4);
          }

          const int64_t cA1 = *(const int64_t *)&vAPtr[vL].qs[8];
          const int64_t cA5 = *(const int64_t *)&vAPtr[vL].qs[40];
          const int64_t cA9 = *(const int64_t *)&vAPtr[vL].qs[72];
          const int64_t cAd = *(const int64_t *)&vAPtr[vL].qs[104];
          __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
          vint16m4_t vSumiL1;
          {
            const vint8m2_t cLhs08 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA1, cVl / 4));
            const vint8m2_t cLhs18 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA5, cVl / 4));
            const vint8m2_t cLhs28 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA9, cVl / 4));
            const vint8m2_t cLhs38 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAd, cVl / 4));
            const vint16m4_t cSumiLo0 = __riscv_vwmul_vv_i16m4(cRhsVecLo0, cLhs08, cVl * 2);
            const vint16m4_t cSumiLo1 = __riscv_vwmacc_vv_i16m4(cSumiLo0, cRhsVecLo1, cLhs18, cVl * 2);
            const vint16m4_t cSumiHi0 = __riscv_vwmacc_vv_i16m4(cSumiLo1, cRhsVecHi0, cLhs28, cVl * 2);
            const vint16m4_t cSumiHiM = __riscv_vwmacc_vv_i16m4(cSumiHi0, cRhsVecHi1, cLhs38, cVl * 2);

            vSumiL1 = cSumiHiM;
          }

          {
            const vuint32m4_t cSumiI32 =
                __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(vSumiL1));
            const vuint16m2_t cSumiH20 = __riscv_vnsrl_wx_u16m2(cSumiI32, 0, cVl);
            const vuint16m2_t cSumiH21 = __riscv_vnsrl_wx_u16m2(cSumiI32, 16, cVl);
            const vuint16m2_t cSumiH2 = __riscv_vadd_vv_u16m2(cSumiH20, cSumiH21, cVl);
            const vuint32m2_t cSumiH2I32 = __riscv_vreinterpret_v_u16m2_u32m2(cSumiH2);
            const vuint16m1_t cSumiH40 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 0, cVl / 2);
            const vuint16m1_t cSumiH41 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 16, cVl / 2);
            const vuint16m1_t cSumiH4 = __riscv_vadd_vv_u16m1(cSumiH40, cSumiH41, cVl / 2);
            const vuint32m1_t cSumiH4I32 = __riscv_vreinterpret_v_u16m1_u32m1(cSumiH4);
            const vint16mf2_t cSumiH80 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 0, cVl / 4));
            const vint16mf2_t cSumiH81 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 16, cVl / 4));
            const vint32m1_t cSumiH8 = __riscv_vwadd_vv_i32m1(cSumiH80, cSumiH81, cVl / 4);
            const vfloat32m1_t cFacc = __riscv_vfcvt_f_x_v_f32m1(cSumiH8, cVl / 4);

            const vfloat32m1_t cTmp1 = __riscv_vfmul_vf_f32m1(cFacc, aAScales[1], cVl / 4);
            vSumf1 = __riscv_vfmacc_vv_f32m1(vSumf1, cTmp1, cBScalesVec, cVl / 4);
          }

          const int64_t cA2 = *(const int64_t *)&vAPtr[vL].qs[16];
          const int64_t cA6 = *(const int64_t *)&vAPtr[vL].qs[48];
          const int64_t cAa = *(const int64_t *)&vAPtr[vL].qs[80];
          const int64_t cAe = *(const int64_t *)&vAPtr[vL].qs[112];
          __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
          vint16m4_t vSumiL2;
          {
            const vint8m2_t cLhs08 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA2, cVl / 4));
            const vint8m2_t cLhs18 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA6, cVl / 4));
            const vint8m2_t cLhs28 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAa, cVl / 4));
            const vint8m2_t cLhs38 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAe, cVl / 4));
            const vint16m4_t cSumiLo0 = __riscv_vwmul_vv_i16m4(cRhsVecLo0, cLhs08, cVl * 2);
            const vint16m4_t cSumiLo1 = __riscv_vwmacc_vv_i16m4(cSumiLo0, cRhsVecLo1, cLhs18, cVl * 2);
            const vint16m4_t cSumiHi0 = __riscv_vwmacc_vv_i16m4(cSumiLo1, cRhsVecHi0, cLhs28, cVl * 2);
            const vint16m4_t cSumiHiM = __riscv_vwmacc_vv_i16m4(cSumiHi0, cRhsVecHi1, cLhs38, cVl * 2);

            vSumiL2 = cSumiHiM;
          }

          {
            const vuint32m4_t cSumiI32 =
                __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(vSumiL2));
            const vuint16m2_t cSumiH20 = __riscv_vnsrl_wx_u16m2(cSumiI32, 0, cVl);
            const vuint16m2_t cSumiH21 = __riscv_vnsrl_wx_u16m2(cSumiI32, 16, cVl);
            const vuint16m2_t cSumiH2 = __riscv_vadd_vv_u16m2(cSumiH20, cSumiH21, cVl);
            const vuint32m2_t cSumiH2I32 = __riscv_vreinterpret_v_u16m2_u32m2(cSumiH2);
            const vuint16m1_t cSumiH40 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 0, cVl / 2);
            const vuint16m1_t cSumiH41 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 16, cVl / 2);
            const vuint16m1_t cSumiH4 = __riscv_vadd_vv_u16m1(cSumiH40, cSumiH41, cVl / 2);
            const vuint32m1_t cSumiH4I32 = __riscv_vreinterpret_v_u16m1_u32m1(cSumiH4);
            const vint16mf2_t cSumiH80 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 0, cVl / 4));
            const vint16mf2_t cSumiH81 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 16, cVl / 4));
            const vint32m1_t cSumiH8 = __riscv_vwadd_vv_i32m1(cSumiH80, cSumiH81, cVl / 4);
            const vfloat32m1_t cFacc = __riscv_vfcvt_f_x_v_f32m1(cSumiH8, cVl / 4);

            const vfloat32m1_t cTmp1 = __riscv_vfmul_vf_f32m1(cFacc, aAScales[2], cVl / 4);
            vSumf2 = __riscv_vfmacc_vv_f32m1(vSumf2, cTmp1, cBScalesVec, cVl / 4);
          }

          const int64_t cA3 = *(const int64_t *)&vAPtr[vL].qs[24];
          const int64_t cA7 = *(const int64_t *)&vAPtr[vL].qs[56];
          const int64_t cAb = *(const int64_t *)&vAPtr[vL].qs[88];
          const int64_t cAf = *(const int64_t *)&vAPtr[vL].qs[120];
          __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
          vint16m4_t vSumiL3;
          {
            const vint8m2_t cLhs08 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA3, cVl / 4));
            const vint8m2_t cLhs18 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cA7, cVl / 4));
            const vint8m2_t cLhs28 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAb, cVl / 4));
            const vint8m2_t cLhs38 = __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(cAf, cVl / 4));
            const vint16m4_t cSumiLo0 = __riscv_vwmul_vv_i16m4(cRhsVecLo0, cLhs08, cVl * 2);
            const vint16m4_t cSumiLo1 = __riscv_vwmacc_vv_i16m4(cSumiLo0, cRhsVecLo1, cLhs18, cVl * 2);
            const vint16m4_t cSumiHi0 = __riscv_vwmacc_vv_i16m4(cSumiLo1, cRhsVecHi0, cLhs28, cVl * 2);
            const vint16m4_t cSumiHiM = __riscv_vwmacc_vv_i16m4(cSumiHi0, cRhsVecHi1, cLhs38, cVl * 2);

            vSumiL3 = cSumiHiM;
          }

          {
            const vuint32m4_t cSumiI32 =
                __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(vSumiL3));
            const vuint16m2_t cSumiH20 = __riscv_vnsrl_wx_u16m2(cSumiI32, 0, cVl);
            const vuint16m2_t cSumiH21 = __riscv_vnsrl_wx_u16m2(cSumiI32, 16, cVl);
            const vuint16m2_t cSumiH2 = __riscv_vadd_vv_u16m2(cSumiH20, cSumiH21, cVl);
            const vuint32m2_t cSumiH2I32 = __riscv_vreinterpret_v_u16m2_u32m2(cSumiH2);
            const vuint16m1_t cSumiH40 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 0, cVl / 2);
            const vuint16m1_t cSumiH41 = __riscv_vnsrl_wx_u16m1(cSumiH2I32, 16, cVl / 2);
            const vuint16m1_t cSumiH4 = __riscv_vadd_vv_u16m1(cSumiH40, cSumiH41, cVl / 2);
            const vuint32m1_t cSumiH4I32 = __riscv_vreinterpret_v_u16m1_u32m1(cSumiH4);
            const vint16mf2_t cSumiH80 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 0, cVl / 4));
            const vint16mf2_t cSumiH81 =
                __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(cSumiH4I32, 16, cVl / 4));
            const vint32m1_t cSumiH8 = __riscv_vwadd_vv_i32m1(cSumiH80, cSumiH81, cVl / 4);
            const vfloat32m1_t cFacc = __riscv_vfcvt_f_x_v_f32m1(cSumiH8, cVl / 4);

            const vfloat32m1_t cTmp1 = __riscv_vfmul_vf_f32m1(cFacc, aAScales[3], cVl / 4);
            vSumf3 = __riscv_vfmacc_vv_f32m1(vSumf3, cTmp1, cBScalesVec, cVl / 4);
          }
        }
        __riscv_vse32_v_f32m1(&pS[(vY * 4 + 0) * pBs + vX * cNcolsInterleaved], vSumf0, cVl / 4);
        __riscv_vse32_v_f32m1(&pS[(vY * 4 + 1) * pBs + vX * cNcolsInterleaved], vSumf1, cVl / 4);
        __riscv_vse32_v_f32m1(&pS[(vY * 4 + 2) * pBs + vX * cNcolsInterleaved], vSumf2, cVl / 4);
        __riscv_vse32_v_f32m1(&pS[(vY * 4 + 3) * pBs + vX * cNcolsInterleaved], vSumf3, cVl / 4);
      }
    }

    return;
  }

#endif
  ggml_gemm_q4_0_8x8_q8_0_generic(pN, pS, pBs, pVx, pVy, pNr, pNc);
}

#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_0x16 *vBPtr = (const block_q4_0x16 *)pVx + (vX * cNb);

      // 4x16 Accumulators
      vfloat32m2_t vSumf0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

      for (int vL = 0; vL < cNb; vL++) {
        // 4x16 integer accumulators
        vint16m1_t vSumi0Lo16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi1Lo16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi2Lo16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi3Lo16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi0Hi16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi1Hi16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi2Hi16 = __riscv_vmv_v_x_i16m1(0.0f, 16);
        vint16m1_t vSumi3Hi16 = __riscv_vmv_v_x_i16m1(0.0f, 16);

        // Accumulation loop.
        for (int vI = 0; vI < QK4_0 / 2; vI++) {
          // Load `b_ptr`.
          const vint8mf2_t cB0Packed = __riscv_vle8_v_i8mf2((const int8_t *)&vBPtr[vL].qs[vI * 16], 16);
          const vint8mf2_t cB0Lo = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(cB0Packed, 4, 16), 4, 16);
          const vint8mf2_t cB0Hi = __riscv_vsra_vx_i8mf2(cB0Packed, 4, 16);

          vSumi0Lo16 = __riscv_vwmacc_vx_i16m1(vSumi0Lo16, vAPtr[vL].qs[vI * 4], cB0Lo, 16);
          vSumi1Lo16 = __riscv_vwmacc_vx_i16m1(vSumi1Lo16, vAPtr[vL].qs[vI * 4 + 1], cB0Lo, 16);
          vSumi2Lo16 = __riscv_vwmacc_vx_i16m1(vSumi2Lo16, vAPtr[vL].qs[vI * 4 + 2], cB0Lo, 16);
          vSumi3Lo16 = __riscv_vwmacc_vx_i16m1(vSumi3Lo16, vAPtr[vL].qs[vI * 4 + 3], cB0Lo, 16);

          vSumi0Hi16 = __riscv_vwmacc_vx_i16m1(vSumi0Hi16, vAPtr[vL].qs[64 + vI * 4], cB0Hi, 16);
          vSumi1Hi16 = __riscv_vwmacc_vx_i16m1(vSumi1Hi16, vAPtr[vL].qs[64 + vI * 4 + 1], cB0Hi, 16);
          vSumi2Hi16 = __riscv_vwmacc_vx_i16m1(vSumi2Hi16, vAPtr[vL].qs[64 + vI * 4 + 2], cB0Hi, 16);
          vSumi3Hi16 = __riscv_vwmacc_vx_i16m1(vSumi3Hi16, vAPtr[vL].qs[64 + vI * 4 + 3], cB0Hi, 16);
        }

        // Do the final accumulation in i32 to prevent overflow.
        const vint32m2_t cSumi0 = __riscv_vwadd_vv_i32m2(vSumi0Lo16, vSumi0Hi16, 16);
        const vint32m2_t cSumi1 = __riscv_vwadd_vv_i32m2(vSumi1Lo16, vSumi1Hi16, 16);
        const vint32m2_t cSumi2 = __riscv_vwadd_vv_i32m2(vSumi2Lo16, vSumi2Hi16, 16);
        const vint32m2_t cSumi3 = __riscv_vwadd_vv_i32m2(vSumi3Lo16, vSumi3Hi16, 16);

        const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
        const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[0], 16);
        const vfloat32m2_t cD1 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[1], 16);
        const vfloat32m2_t cD2 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[2], 16);
        const vfloat32m2_t cD3 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[3], 16);

        vSumf0 = __riscv_vfmacc_vv_f32m2(vSumf0, __riscv_vfcvt_f_x_v_f32m2(cSumi0, 16), cD0, 16);
        vSumf1 = __riscv_vfmacc_vv_f32m2(vSumf1, __riscv_vfcvt_f_x_v_f32m2(cSumi1, 16), cD1, 16);
        vSumf2 = __riscv_vfmacc_vv_f32m2(vSumf2, __riscv_vfcvt_f_x_v_f32m2(cSumi2, 16), cD2, 16);
        vSumf3 = __riscv_vfmacc_vv_f32m2(vSumf3, __riscv_vfcvt_f_x_v_f32m2(cSumi3, 16), cD3, 16);
      }

      __riscv_vse32_v_f32m2(pS + (vY * 4 + 0) * pBs + vX * 16, vSumf0, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 1) * pBs + vX * 16, vSumf1, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 2) * pBs + vX * 16, vSumf2, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 3) * pBs + vX * 16, vSumf3, 16);
    }
  }
}

void ggml_gemm_q4_K_16x1_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK_K;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_Kx4 *vAPtr = (const block_q8_Kx4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q4_Kx16 *vBPtr = (const block_q4_Kx16 *)pVx + (vX * cNb);

      // 4x16 Accumulators
      vfloat32m2_t vSumf0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

      for (int vL = 0; vL < cNb; vL++) {
        vint32m2_t vSumi0 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t vSumi1 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t vSumi2 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t vSumi3 = __riscv_vmv_v_x_i32m2(0, 16);

        // Load `dmin`.
        const vfloat32m2_t cDmins =
            __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].dmin, 16), 16);

        // We process 4 sub-blocks at once.
        for (int vJ = 0; vJ < QK_K / 128; vJ++) {
          // Extract the scales and the mins.
          //
          // Low bits.
          vuint8m2_t vScalesMinsLo = __riscv_vle8_v_u8m2(&vBPtr[vL].scales[vJ * 64], 64);
          vuint8m2_t vScalesLo = __riscv_vand_vx_u8m2(vScalesMinsLo, 0x0F, 64);
          vuint8m2_t vMinsLo = __riscv_vsrl_vx_u8m2(vScalesMinsLo, 4, 64);

          // High bits.
          vuint8m2_t vScalesMinsHi = __riscv_vle8_v_u8m2(&vBPtr[vL].scales[128], 64);
          vuint8m2_t vScalesHi;
          vuint8m2_t vMinsHi;
          if (!vJ) {
            vScalesHi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0x03, 64), 4, 64);
            vMinsHi = __riscv_vsll_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0x0C, 64), 2, 64);
          } else {
            vScalesHi = __riscv_vand_vx_u8m2(vScalesMinsHi, 0x30, 64);
            vMinsHi = __riscv_vsrl_vx_u8m2(__riscv_vand_vx_u8m2(vScalesMinsHi, 0xC0, 64), 2, 64);
          }
          vuint16m4_t vScales = __riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(vScalesHi, vScalesLo, 64), 64);
          vint16m4_t vMins = __riscv_vreinterpret_v_u16m4_i16m4(
              __riscv_vzext_vf2_u16m4(__riscv_vor_vv_u8m2(vMinsHi, vMinsLo, 64), 64));

          // Reduce the mins and multiply with `dmin`.
          //
          // Correct in `sumf`.
          vint32m2_t vBsums0 = __riscv_vmv_v_x_i32m2(0, 16);
          vint32m2_t vBsums1 = __riscv_vmv_v_x_i32m2(0, 16);
          vint32m2_t vBsums2 = __riscv_vmv_v_x_i32m2(0, 16);
          vint32m2_t vBsums3 = __riscv_vmv_v_x_i32m2(0, 16);

          vBsums0 = __riscv_vwmacc_vx_i32m2(vBsums0, vAPtr[vL].bsums[vJ * 32] + vAPtr[vL].bsums[vJ * 32 + 4],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 0), 16);
          vBsums1 = __riscv_vwmacc_vx_i32m2(vBsums1, vAPtr[vL].bsums[vJ * 32 + 1] + vAPtr[vL].bsums[vJ * 32 + 5],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 0), 16);
          vBsums2 = __riscv_vwmacc_vx_i32m2(vBsums2, vAPtr[vL].bsums[vJ * 32 + 2] + vAPtr[vL].bsums[vJ * 32 + 6],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 0), 16);
          vBsums3 = __riscv_vwmacc_vx_i32m2(vBsums3, vAPtr[vL].bsums[vJ * 32 + 3] + vAPtr[vL].bsums[vJ * 32 + 7],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 0), 16);
          vBsums0 = __riscv_vwmacc_vx_i32m2(vBsums0, vAPtr[vL].bsums[vJ * 32 + 8] + vAPtr[vL].bsums[vJ * 32 + 8 + 4],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 1), 16);
          vBsums1 =
              __riscv_vwmacc_vx_i32m2(vBsums1, vAPtr[vL].bsums[vJ * 32 + 8 + 1] + vAPtr[vL].bsums[vJ * 32 + 8 + 5],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 1), 16);
          vBsums2 =
              __riscv_vwmacc_vx_i32m2(vBsums2, vAPtr[vL].bsums[vJ * 32 + 8 + 2] + vAPtr[vL].bsums[vJ * 32 + 8 + 6],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 1), 16);
          vBsums3 =
              __riscv_vwmacc_vx_i32m2(vBsums3, vAPtr[vL].bsums[vJ * 32 + 8 + 3] + vAPtr[vL].bsums[vJ * 32 + 8 + 7],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 1), 16);
          vBsums0 = __riscv_vwmacc_vx_i32m2(vBsums0, vAPtr[vL].bsums[vJ * 32 + 16] + vAPtr[vL].bsums[vJ * 32 + 16 + 4],
                                            __riscv_vget_v_i16m4_i16m1(vMins, 2), 16);
          vBsums1 =
              __riscv_vwmacc_vx_i32m2(vBsums1, vAPtr[vL].bsums[vJ * 32 + 16 + 1] + vAPtr[vL].bsums[vJ * 32 + 16 + 5],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 2), 16);
          vBsums2 =
              __riscv_vwmacc_vx_i32m2(vBsums2, vAPtr[vL].bsums[vJ * 32 + 16 + 2] + vAPtr[vL].bsums[vJ * 32 + 16 + 6],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 2), 16);
          vBsums3 =
              __riscv_vwmacc_vx_i32m2(vBsums3, vAPtr[vL].bsums[vJ * 32 + 16 + 3] + vAPtr[vL].bsums[vJ * 32 + 16 + 7],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 2), 16);
          vBsums0 =
              __riscv_vwmacc_vx_i32m2(vBsums0, vAPtr[vL].bsums[vJ * 32 + 24 + 0] + vAPtr[vL].bsums[vJ * 32 + 24 + 4],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 3), 16);
          vBsums1 =
              __riscv_vwmacc_vx_i32m2(vBsums1, vAPtr[vL].bsums[vJ * 32 + 24 + 1] + vAPtr[vL].bsums[vJ * 32 + 24 + 5],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 3), 16);
          vBsums2 =
              __riscv_vwmacc_vx_i32m2(vBsums2, vAPtr[vL].bsums[vJ * 32 + 24 + 2] + vAPtr[vL].bsums[vJ * 32 + 24 + 6],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 3), 16);
          vBsums3 =
              __riscv_vwmacc_vx_i32m2(vBsums3, vAPtr[vL].bsums[vJ * 32 + 24 + 3] + vAPtr[vL].bsums[vJ * 32 + 24 + 7],
                                      __riscv_vget_v_i16m4_i16m1(vMins, 3), 16);

          const vfloat32m2_t cDminsD0 = __riscv_vfmul_vf_f32m2(cDmins, vAPtr[vL].d[0], 16);
          const vfloat32m2_t cDminsD1 = __riscv_vfmul_vf_f32m2(cDmins, vAPtr[vL].d[1], 16);
          const vfloat32m2_t cDminsD2 = __riscv_vfmul_vf_f32m2(cDmins, vAPtr[vL].d[2], 16);
          const vfloat32m2_t cDminsD3 = __riscv_vfmul_vf_f32m2(cDmins, vAPtr[vL].d[3], 16);

          vSumf0 = __riscv_vfsub_vv_f32m2(
              vSumf0, __riscv_vfmul_vv_f32m2(cDminsD0, __riscv_vfcvt_f_x_v_f32m2(vBsums0, 16), 16), 16);
          vSumf1 = __riscv_vfsub_vv_f32m2(
              vSumf1, __riscv_vfmul_vv_f32m2(cDminsD1, __riscv_vfcvt_f_x_v_f32m2(vBsums1, 16), 16), 16);
          vSumf2 = __riscv_vfsub_vv_f32m2(
              vSumf2, __riscv_vfmul_vv_f32m2(cDminsD2, __riscv_vfcvt_f_x_v_f32m2(vBsums2, 16), 16), 16);
          vSumf3 = __riscv_vfsub_vv_f32m2(
              vSumf3, __riscv_vfmul_vv_f32m2(cDminsD3, __riscv_vfcvt_f_x_v_f32m2(vBsums3, 16), 16), 16);

          // Accumulation for 2 sub-blocks.
          //
          // This might overflow, so we accumulate in two steps.
          //
          // Recheck.
          for (int vK = 0; vK < 2; vK++) {
            // 4x16 integer accumulators
            vint16m1_t vSumi0S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi1S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi2S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi3S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi0S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi1S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi2S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi3S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);

            for (int vI = vK * 16; vI < vK * 16 + QK4_0 / 2; vI++) {
              // Load `b_ptr`.
              const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2(&vBPtr[vL].qs[vJ * 1024 + vI * 16], 16);
              const vint8mf2_t cBS0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(cB0Packed, 0xF, 16));
              const vint8mf2_t cBS1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16));

              vSumi0S016 = __riscv_vwmacc_vx_i16m1(vSumi0S016, vAPtr[vL].qs[vJ * 512 + vI * 4], cBS0, 16);
              vSumi1S016 = __riscv_vwmacc_vx_i16m1(vSumi1S016, vAPtr[vL].qs[vJ * 512 + vI * 4 + 1], cBS0, 16);
              vSumi2S016 = __riscv_vwmacc_vx_i16m1(vSumi2S016, vAPtr[vL].qs[vJ * 512 + vI * 4 + 2], cBS0, 16);
              vSumi3S016 = __riscv_vwmacc_vx_i16m1(vSumi3S016, vAPtr[vL].qs[vJ * 512 + vI * 4 + 3], cBS0, 16);

              vSumi0S116 = __riscv_vwmacc_vx_i16m1(vSumi0S116, vAPtr[vL].qs[vJ * 512 + 128 + vI * 4], cBS1, 16);
              vSumi1S116 = __riscv_vwmacc_vx_i16m1(vSumi1S116, vAPtr[vL].qs[vJ * 512 + 128 + vI * 4 + 1], cBS1, 16);
              vSumi2S116 = __riscv_vwmacc_vx_i16m1(vSumi2S116, vAPtr[vL].qs[vJ * 512 + 128 + vI * 4 + 2], cBS1, 16);
              vSumi3S116 = __riscv_vwmacc_vx_i16m1(vSumi3S116, vAPtr[vL].qs[vJ * 512 + 128 + vI * 4 + 3], cBS1, 16);
            }

            vSumi0 = __riscv_vwmacc_vv_i32m2(
                vSumi0, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 0)), vSumi0S016, 16);
            vSumi0 = __riscv_vwmacc_vv_i32m2(
                vSumi0, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 1)), vSumi0S116, 16);
            vSumi1 = __riscv_vwmacc_vv_i32m2(
                vSumi1, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 0)), vSumi1S016, 16);
            vSumi1 = __riscv_vwmacc_vv_i32m2(
                vSumi1, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 1)), vSumi1S116, 16);
            vSumi2 = __riscv_vwmacc_vv_i32m2(
                vSumi2, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 0)), vSumi2S016, 16);
            vSumi2 = __riscv_vwmacc_vv_i32m2(
                vSumi2, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 1)), vSumi2S116, 16);
            vSumi3 = __riscv_vwmacc_vv_i32m2(
                vSumi3, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 0)), vSumi3S016, 16);
            vSumi3 = __riscv_vwmacc_vv_i32m2(
                vSumi3, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 1)), vSumi3S116, 16);
          }
          // Accumulation for 2 sub-blocks.
          //
          // This might overflow, so we accumulate in two steps.
          //
          // Recheck.
          for (int vK = 0; vK < 2; vK++) {
            // 4x16 integer accumulators
            vint16m1_t vSumi0S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi1S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi2S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi3S016 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi0S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi1S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi2S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);
            vint16m1_t vSumi3S116 = __riscv_vmv_v_x_i16m1(0.0f, 16);

            for (int vI = vK * 16; vI < vK * 16 + QK4_0 / 2; vI++) {
              // Load `b_ptr`.
              const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2(&vBPtr[vL].qs[vJ * 1024 + 512 + vI * 16], 16);
              const vint8mf2_t cBS0 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vand_vx_u8mf2(cB0Packed, 0xF, 16));
              const vint8mf2_t cBS1 = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16));

              vSumi0S016 = __riscv_vwmacc_vx_i16m1(vSumi0S016, vAPtr[vL].qs[vJ * 512 + 256 + vI * 4], cBS0, 16);
              vSumi1S016 = __riscv_vwmacc_vx_i16m1(vSumi1S016, vAPtr[vL].qs[vJ * 512 + 256 + vI * 4 + 1], cBS0, 16);
              vSumi2S016 = __riscv_vwmacc_vx_i16m1(vSumi2S016, vAPtr[vL].qs[vJ * 512 + 256 + vI * 4 + 2], cBS0, 16);
              vSumi3S016 = __riscv_vwmacc_vx_i16m1(vSumi3S016, vAPtr[vL].qs[vJ * 512 + 256 + vI * 4 + 3], cBS0, 16);

              vSumi0S116 = __riscv_vwmacc_vx_i16m1(vSumi0S116, vAPtr[vL].qs[vJ * 512 + 384 + vI * 4], cBS1, 16);
              vSumi1S116 = __riscv_vwmacc_vx_i16m1(vSumi1S116, vAPtr[vL].qs[vJ * 512 + 384 + vI * 4 + 1], cBS1, 16);
              vSumi2S116 = __riscv_vwmacc_vx_i16m1(vSumi2S116, vAPtr[vL].qs[vJ * 512 + 384 + vI * 4 + 2], cBS1, 16);
              vSumi3S116 = __riscv_vwmacc_vx_i16m1(vSumi3S116, vAPtr[vL].qs[vJ * 512 + 384 + vI * 4 + 3], cBS1, 16);
            }

            vSumi0 = __riscv_vwmacc_vv_i32m2(
                vSumi0, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 2)), vSumi0S016, 16);
            vSumi0 = __riscv_vwmacc_vv_i32m2(
                vSumi0, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 3)), vSumi0S116, 16);
            vSumi1 = __riscv_vwmacc_vv_i32m2(
                vSumi1, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 2)), vSumi1S016, 16);
            vSumi1 = __riscv_vwmacc_vv_i32m2(
                vSumi1, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 3)), vSumi1S116, 16);
            vSumi2 = __riscv_vwmacc_vv_i32m2(
                vSumi2, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 2)), vSumi2S016, 16);
            vSumi2 = __riscv_vwmacc_vv_i32m2(
                vSumi2, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 3)), vSumi2S116, 16);
            vSumi3 = __riscv_vwmacc_vv_i32m2(
                vSumi3, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 2)), vSumi3S016, 16);
            vSumi3 = __riscv_vwmacc_vv_i32m2(
                vSumi3, __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vget_v_u16m4_u16m1(vScales, 3)), vSumi3S116, 16);
          }
        }

        const vfloat32m2_t cBD =
            __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16), 16);
        const vfloat32m2_t cD0 = __riscv_vfmul_vf_f32m2(cBD, vAPtr[vL].d[0], 16);
        const vfloat32m2_t cD1 = __riscv_vfmul_vf_f32m2(cBD, vAPtr[vL].d[1], 16);
        const vfloat32m2_t cD2 = __riscv_vfmul_vf_f32m2(cBD, vAPtr[vL].d[2], 16);
        const vfloat32m2_t cD3 = __riscv_vfmul_vf_f32m2(cBD, vAPtr[vL].d[3], 16);

        vSumf0 = __riscv_vfmacc_vv_f32m2(vSumf0, __riscv_vfcvt_f_x_v_f32m2(vSumi0, 16), cD0, 16);
        vSumf1 = __riscv_vfmacc_vv_f32m2(vSumf1, __riscv_vfcvt_f_x_v_f32m2(vSumi1, 16), cD1, 16);
        vSumf2 = __riscv_vfmacc_vv_f32m2(vSumf2, __riscv_vfcvt_f_x_v_f32m2(vSumi2, 16), cD2, 16);
        vSumf3 = __riscv_vfmacc_vv_f32m2(vSumf3, __riscv_vfcvt_f_x_v_f32m2(vSumi3, 16), cD3, 16);
      }

      __riscv_vse32_v_f32m2(pS + (vY * 4 + 0) * pBs + vX * 16, vSumf0, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 1) * pBs + vX * 16, vSumf1, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 2) * pBs + vX * 16, vSumf2, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 3) * pBs + vX * 16, vSumf3, 16);
    }
  }
}

void ggml_gemm_iq4_nl_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  const vint8mf2_t cValues = __riscv_vle8_v_i8mf2(kvalues_iq4nl, 16);
  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_iq4_nlx16 *vBPtr = (const block_iq4_nlx16 *)pVx + (vX * cNb);

      // 4x16 Accumulators
      vfloat32m2_t vSumf0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

      for (int vL = 0; vL < cNb; vL++) {
        // 4x16 integer accumulators
        vint32m2_t vSumi0 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi1 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi2 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi3 = __riscv_vmv_v_x_i32m2(0.0f, 16);

        // Accumulation loop.
        for (int vI = 0; vI < QK4_NL / 2; vI++) {
          // Load `b_ptr`.
          const vuint8mf2_t cB0Packed = __riscv_vle8_v_u8mf2((const uint8_t *)&vBPtr[vL].qs[vI * 16], 16);
          const vint8mf2_t cB0Lo = __riscv_vrgather_vv_i8mf2(cValues, __riscv_vand_vx_u8mf2(cB0Packed, 0xf, 16), 16);
          const vint8mf2_t cB0Hi = __riscv_vrgather_vv_i8mf2(cValues, __riscv_vsrl_vx_u8mf2(cB0Packed, 4, 16), 16);
          // const vint16m1_t b_0_lo_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_lo, 16);
          // const vint16m1_t b_0_hi_16 = __riscv_vwcvt_x_x_v_i16m1(b_0_hi, 16);

          const vint16m1_t cSumi0Lo = __riscv_vwmul_vx_i16m1(cB0Lo, vAPtr[vL].qs[vI * 4], 16);
          const vint16m1_t cSumi1Lo = __riscv_vwmul_vx_i16m1(cB0Lo, vAPtr[vL].qs[vI * 4 + 1], 16);
          const vint16m1_t cSumi2Lo = __riscv_vwmul_vx_i16m1(cB0Lo, vAPtr[vL].qs[vI * 4 + 2], 16);
          const vint16m1_t cSumi3Lo = __riscv_vwmul_vx_i16m1(cB0Lo, vAPtr[vL].qs[vI * 4 + 3], 16);

          const vint16m1_t cSumi0Hi = __riscv_vwmul_vx_i16m1(cB0Hi, vAPtr[vL].qs[64 + vI * 4], 16);
          const vint16m1_t cSumi1Hi = __riscv_vwmul_vx_i16m1(cB0Hi, vAPtr[vL].qs[64 + vI * 4 + 1], 16);
          const vint16m1_t cSumi2Hi = __riscv_vwmul_vx_i16m1(cB0Hi, vAPtr[vL].qs[64 + vI * 4 + 2], 16);
          const vint16m1_t cSumi3Hi = __riscv_vwmul_vx_i16m1(cB0Hi, vAPtr[vL].qs[64 + vI * 4 + 3], 16);

          vSumi0 = __riscv_vadd_vv_i32m2(vSumi0, __riscv_vwadd_vv_i32m2(cSumi0Lo, cSumi0Hi, 16), 16);
          vSumi1 = __riscv_vadd_vv_i32m2(vSumi1, __riscv_vwadd_vv_i32m2(cSumi1Lo, cSumi1Hi, 16), 16);
          vSumi2 = __riscv_vadd_vv_i32m2(vSumi2, __riscv_vwadd_vv_i32m2(cSumi2Lo, cSumi2Hi, 16), 16);
          vSumi3 = __riscv_vadd_vv_i32m2(vSumi3, __riscv_vwadd_vv_i32m2(cSumi3Lo, cSumi3Hi, 16), 16);
        }

        const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
        const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[0], 16);
        const vfloat32m2_t cD1 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[1], 16);
        const vfloat32m2_t cD2 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[2], 16);
        const vfloat32m2_t cD3 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[3], 16);

        vSumf0 = __riscv_vfmacc_vv_f32m2(vSumf0, __riscv_vfcvt_f_x_v_f32m2(vSumi0, 16), cD0, 16);
        vSumf1 = __riscv_vfmacc_vv_f32m2(vSumf1, __riscv_vfcvt_f_x_v_f32m2(vSumi1, 16), cD1, 16);
        vSumf2 = __riscv_vfmacc_vv_f32m2(vSumf2, __riscv_vfcvt_f_x_v_f32m2(vSumi2, 16), cD2, 16);
        vSumf3 = __riscv_vfmacc_vv_f32m2(vSumf3, __riscv_vfcvt_f_x_v_f32m2(vSumi3, 16), cD3, 16);
      }

      __riscv_vse32_v_f32m2(pS + (vY * 4 + 0) * pBs + vX * 16, vSumf0, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 1) * pBs + vX * 16, vSumf1, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 2) * pBs + vX * 16, vSumf2, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 3) * pBs + vX * 16, vSumf3, 16);
    }
  }
}

void ggml_gemm_q8_0_16x1_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;
  const int cNcolsInterleaved = 16;
  const int cBlocklen = 1;

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

  for (int vY = 0; vY < pNr / 4; vY++) {
    const block_q8_0x4 *vAPtr = (const block_q8_0x4 *)pVy + (vY * cNb);
    for (int vX = 0; vX < pNc / cNcolsInterleaved; vX++) {
      const block_q8_0x16 *vBPtr = (const block_q8_0x16 *)pVx + (vX * cNb);

      // 4x16 Accumulators
      vfloat32m2_t vSumf0 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf1 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf2 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
      vfloat32m2_t vSumf3 = __riscv_vfmv_v_f_f32m2(0.0f, 16);

      for (int vL = 0; vL < cNb; vL++) {
        // 4x16 Integer Accumulators
        vint32m2_t vSumi0 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi1 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi2 = __riscv_vmv_v_x_i32m2(0.0f, 16);
        vint32m2_t vSumi3 = __riscv_vmv_v_x_i32m2(0.0f, 16);

        // Accumulation loop.
        for (int vI = 0; vI < QK8_0; vI++) {
          // Load `b_ptr`.
          const vint8mf2_t cB0 = __riscv_vle8_v_i8mf2((const int8_t *)&vBPtr[vL].qs[vI * 16], 16);
          // const vint16m1_t b_0_16 = __riscv_vwcvt_x_x_v_i16m1(b_0, 16);

          vSumi0 = __riscv_vwadd_wv_i32m2(vSumi0, __riscv_vwmul_vx_i16m1(cB0, vAPtr[vL].qs[vI * 4 + 0], 16), 16);
          vSumi1 = __riscv_vwadd_wv_i32m2(vSumi1, __riscv_vwmul_vx_i16m1(cB0, vAPtr[vL].qs[vI * 4 + 1], 16), 16);
          vSumi2 = __riscv_vwadd_wv_i32m2(vSumi2, __riscv_vwmul_vx_i16m1(cB0, vAPtr[vL].qs[vI * 4 + 2], 16), 16);
          vSumi3 = __riscv_vwadd_wv_i32m2(vSumi3, __riscv_vwmul_vx_i16m1(cB0, vAPtr[vL].qs[vI * 4 + 3], 16), 16);
        }

        const vfloat16m1_t cBD = __riscv_vle16_v_f16m1((const _Float16 *)vBPtr[vL].d, 16);
        const vfloat32m2_t cD0 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[0], 16);
        const vfloat32m2_t cD1 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[1], 16);
        const vfloat32m2_t cD2 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[2], 16);
        const vfloat32m2_t cD3 = __riscv_vfwmul_vf_f32m2(cBD, *(const _Float16 *)&vAPtr[vL].d[3], 16);

        vSumf0 = __riscv_vfmacc_vv_f32m2(vSumf0, __riscv_vfcvt_f_x_v_f32m2(vSumi0, 16), cD0, 16);
        vSumf1 = __riscv_vfmacc_vv_f32m2(vSumf1, __riscv_vfcvt_f_x_v_f32m2(vSumi1, 16), cD1, 16);
        vSumf2 = __riscv_vfmacc_vv_f32m2(vSumf2, __riscv_vfcvt_f_x_v_f32m2(vSumi2, 16), cD2, 16);
        vSumf3 = __riscv_vfmacc_vv_f32m2(vSumf3, __riscv_vfcvt_f_x_v_f32m2(vSumi3, 16), cD3, 16);
      }

      __riscv_vse32_v_f32m2(pS + (vY * 4 + 0) * pBs + vX * 16, vSumf0, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 1) * pBs + vX * 16, vSumf1, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 2) * pBs + vX * 16, vSumf2, 16);
      __riscv_vse32_v_f32m2(pS + (vY * 4 + 3) * pBs + vX * 16, vSumf3, 16);
    }
  }
}

void ggml_gemm_q2_K_16x1_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                              const void *GGML_RESTRICT pVy, int pNr, int pNc) {
  assert(pN % QK_K == 0);
  const int cNumKBlocks = pN / QK_K;
  const int cNRowsTile = 4;
  const int cNColsTile = 16;
  assert(pNr % cNRowsTile == 0);
  assert(pNc % cNColsTile == 0);

  const size_t cVl = __riscv_vsetvl_e32m2(cNColsTile);
  // --- Tiling Loops ---
#pragma GCC unroll 1
  for (int vRowTile = 0; vRowTile < pNr; vRowTile += cNRowsTile) {
#pragma GCC unroll 1
    for (int vColTile = 0; vColTile < pNc; vColTile += cNColsTile) {
      // Base Pointers
      const block_q8_Kx4 *vLhsBasePtr = (const block_q8_Kx4 *)pVy + (vRowTile / cNRowsTile) * cNumKBlocks;
      const block_q2_Kx16 *vRhsBasePtr = (const block_q2_Kx16 *)pVx + (vColTile / cNColsTile) * cNumKBlocks;

      // Persistent Float Accumulators
      vfloat32m2_t vVSumf0 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
      vfloat32m2_t vVSumf1 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
      vfloat32m2_t vVSumf2 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);
      vfloat32m2_t vVSumf3 = __riscv_vfmv_v_f_f32m2(0.0f, cVl);

      // --- Super-Block Loop (K=0..255) ---
#pragma GCC unroll 1
      for (int vKBlock = 0; vKBlock < cNumKBlocks; ++vKBlock) {
        const block_q8_Kx4 *vLhsCurrent = &vLhsBasePtr[vKBlock];
        const block_q2_Kx16 *vRhsCurrent = &vRhsBasePtr[vKBlock];

        // 1. Load Global Min Scales (Keep as F16/LMUL=1 to save registers)
        vfloat16m1_t vVGMinF16 = __riscv_vle16_v_f16m1((const _Float16 *)vRhsCurrent->dmin, cVl);
        vfloat32m2_t vVGMinBase = __riscv_vfwcvt_f_f_v_f32m2(vVGMinF16, cVl);

        // 2. Initialize Integer Accumulators
        vint32m2_t vVIsum0 = __riscv_vmv_v_x_i32m2(0, cVl);
        vint32m2_t vVIsum1 = __riscv_vmv_v_x_i32m2(0, cVl);
        vint32m2_t vVIsum2 = __riscv_vmv_v_x_i32m2(0, cVl);
        vint32m2_t vVIsum3 = __riscv_vmv_v_x_i32m2(0, cVl);

        const uint8_t *vRhsQsPtr = vRhsCurrent->qs;
        const uint8_t *vRhsScPtr = vRhsCurrent->scales;
        const int8_t *vLhsQsPtr = vLhsCurrent->qs;

        // --- Phase Loop (4 phases x 64 elements) ---
#pragma GCC unroll 1
        for (int vPhase = 0; vPhase < 4; ++vPhase) {

          // A. Load Scales/Mins for the 4 interleaved sub-blocks
          vuint16m1_t vVDSb0, vVDSb1, vVDSb2, vVDSb3;
          vuint16m1_t vVMSb0, vVMSb1, vVMSb2, vVMSb3;

          // Unrolled Load Logic
          {
            vuint8mf2_t vVRaw;
            // Sub-block 0
            vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 0, cVl);
            vVDSb0 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
            vVMSb0 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

            // Sub-block 1
            vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 16, cVl);
            vVDSb1 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
            vVMSb1 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

            // Sub-block 2
            vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 32, cVl);
            vVDSb2 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
            vVMSb2 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

            // Sub-block 3
            vVRaw = __riscv_vle8_v_u8mf2(vRhsScPtr + 48, cVl);
            vVDSb3 = __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(vVRaw, 0xF, cVl), cVl);
            vVMSb3 = __riscv_vzext_vf2_u16m1(__riscv_vsrl_vx_u8mf2(vVRaw, 4, cVl), cVl);

            vRhsScPtr += 64;
          }

          int vBaseKPhase = (vPhase < 2) ? (vPhase * 16) : (128 + (vPhase - 2) * 16);
          int aKOffsets[4] = {0, 32, 64, 96};

          // B. Inner Dot Product Loop
#pragma GCC unroll 1
          for (int vL = 0; vL < 16; ++vL) {
            vuint8mf2_t vVRhsData = __riscv_vle8_v_u8mf2(vRhsQsPtr, cVl);
            vRhsQsPtr += 16;

            // Unroll over 4 sub-blocks (0, 1, 2, 3 relative to phase)

            // --- Sub-block 0 ---
            {
              vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(vVRhsData, 3, cVl);
              vint16m1_t vVW =
                  __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                        __riscv_vreinterpret_v_u16m1_i16m1(vVDSb0), cVl);

              const int8_t *vQ8 = &vLhsQsPtr[(vBaseKPhase + aKOffsets[0] + vL) * 4];
              vVIsum0 = __riscv_vwmacc_vx_i32m2(vVIsum0, (int16_t)vQ8[0], vVW, cVl);
              vVIsum1 = __riscv_vwmacc_vx_i32m2(vVIsum1, (int16_t)vQ8[1], vVW, cVl);
              vVIsum2 = __riscv_vwmacc_vx_i32m2(vVIsum2, (int16_t)vQ8[2], vVW, cVl);
              vVIsum3 = __riscv_vwmacc_vx_i32m2(vVIsum3, (int16_t)vQ8[3], vVW, cVl);
            }
            // --- Sub-block 1 ---
            {
              vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 2, cVl), 3, cVl);
              vint16m1_t vVW =
                  __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                        __riscv_vreinterpret_v_u16m1_i16m1(vVDSb1), cVl);

              const int8_t *vQ8 = &vLhsQsPtr[(vBaseKPhase + aKOffsets[1] + vL) * 4];
              vVIsum0 = __riscv_vwmacc_vx_i32m2(vVIsum0, (int16_t)vQ8[0], vVW, cVl);
              vVIsum1 = __riscv_vwmacc_vx_i32m2(vVIsum1, (int16_t)vQ8[1], vVW, cVl);
              vVIsum2 = __riscv_vwmacc_vx_i32m2(vVIsum2, (int16_t)vQ8[2], vVW, cVl);
              vVIsum3 = __riscv_vwmacc_vx_i32m2(vVIsum3, (int16_t)vQ8[3], vVW, cVl);
            }
            // --- Sub-block 2 ---
            {
              vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 4, cVl), 3, cVl);
              vint16m1_t vVW =
                  __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                        __riscv_vreinterpret_v_u16m1_i16m1(vVDSb2), cVl);

              const int8_t *vQ8 = &vLhsQsPtr[(vBaseKPhase + aKOffsets[2] + vL) * 4];
              vVIsum0 = __riscv_vwmacc_vx_i32m2(vVIsum0, (int16_t)vQ8[0], vVW, cVl);
              vVIsum1 = __riscv_vwmacc_vx_i32m2(vVIsum1, (int16_t)vQ8[1], vVW, cVl);
              vVIsum2 = __riscv_vwmacc_vx_i32m2(vVIsum2, (int16_t)vQ8[2], vVW, cVl);
              vVIsum3 = __riscv_vwmacc_vx_i32m2(vVIsum3, (int16_t)vQ8[3], vVW, cVl);
            }
            // --- Sub-block 3 ---
            {
              vuint8mf2_t vVQ2 = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vVRhsData, 6, cVl), 3, cVl);
              vint16m1_t vVW =
                  __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(vVQ2, cVl)),
                                        __riscv_vreinterpret_v_u16m1_i16m1(vVDSb3), cVl);

              const int8_t *vQ8 = &vLhsQsPtr[(vBaseKPhase + aKOffsets[3] + vL) * 4];
              vVIsum0 = __riscv_vwmacc_vx_i32m2(vVIsum0, (int16_t)vQ8[0], vVW, cVl);
              vVIsum1 = __riscv_vwmacc_vx_i32m2(vVIsum1, (int16_t)vQ8[1], vVW, cVl);
              vVIsum2 = __riscv_vwmacc_vx_i32m2(vVIsum2, (int16_t)vQ8[2], vVW, cVl);
              vVIsum3 = __riscv_vwmacc_vx_i32m2(vVIsum3, (int16_t)vQ8[3], vVW, cVl);
            }
          }

          // C CORRECTION
          int vSbBaseAbs = vBaseKPhase / 16;

          // --- Correction Sub-block 0 ---
          {
            int vSbAbs = vSbBaseAbs + (aKOffsets[0] / 16);
            vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb0);

            // Row 0
            vfloat32m2_t vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[0], cVl);
            vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 0], cVl);
            vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf0 = __riscv_vfsub_vv_f32m2(vVSumf0, vVfC, cVl);

            // Row 1
            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[1], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 1], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf1 = __riscv_vfsub_vv_f32m2(vVSumf1, vVfC, cVl);

            // Row 2
            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[2], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 2], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf2 = __riscv_vfsub_vv_f32m2(vVSumf2, vVfC, cVl);

            // Row 3
            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[3], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 3], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf3 = __riscv_vfsub_vv_f32m2(vVSumf3, vVfC, cVl);
          }

          // --- Correction Sub-block 1 ---
          {
            int vSbAbs = vSbBaseAbs + (aKOffsets[1] / 16);
            vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb1);

            vfloat32m2_t vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[0], cVl);
            vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 0], cVl);
            vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf0 = __riscv_vfsub_vv_f32m2(vVSumf0, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[1], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 1], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf1 = __riscv_vfsub_vv_f32m2(vVSumf1, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[2], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 2], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf2 = __riscv_vfsub_vv_f32m2(vVSumf2, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[3], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 3], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf3 = __riscv_vfsub_vv_f32m2(vVSumf3, vVfC, cVl);
          }

          // --- Correction Sub-block 2 ---
          {
            int vSbAbs = vSbBaseAbs + (aKOffsets[2] / 16);
            vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb2);

            vfloat32m2_t vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[0], cVl);
            vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 0], cVl);
            vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf0 = __riscv_vfsub_vv_f32m2(vVSumf0, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[1], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 1], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf1 = __riscv_vfsub_vv_f32m2(vVSumf1, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[2], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 2], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf2 = __riscv_vfsub_vv_f32m2(vVSumf2, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[3], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 3], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf3 = __riscv_vfsub_vv_f32m2(vVSumf3, vVfC, cVl);
          }

          // --- Correction Sub-block 3 ---
          {
            int vSbAbs = vSbBaseAbs + (aKOffsets[3] / 16);
            vint16m1_t vVMin = __riscv_vreinterpret_v_u16m1_i16m1(vVMSb3);

            vfloat32m2_t vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[0], cVl);
            vint32m2_t vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 0], cVl);
            vfloat32m2_t vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf0 = __riscv_vfsub_vv_f32m2(vVSumf0, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[1], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 1], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf1 = __riscv_vfsub_vv_f32m2(vVSumf1, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[2], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 2], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf2 = __riscv_vfsub_vv_f32m2(vVSumf2, vVfC, cVl);

            vVGMin = __riscv_vfmul_vf_f32m2(vVGMinBase, vLhsCurrent->d[3], cVl);
            vVC = __riscv_vwmul_vx_i32m2(vVMin, vLhsCurrent->bsums[vSbAbs * 4 + 3], cVl);
            vVfC = __riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(vVC, cVl), vVGMin, cVl);
            vVSumf3 = __riscv_vfsub_vv_f32m2(vVSumf3, vVfC, cVl);
          }

        } // End Phase Loop

        // --- Apply Main Scales ---
        vfloat16m1_t vVGAllF16 = __riscv_vle16_v_f16m1((const _Float16 *)vRhsCurrent->d, cVl);
        vfloat32m2_t vVGAllBase = __riscv_vfwcvt_f_f_v_f32m2(vVGAllF16, cVl);

        {
          vfloat32m2_t vVGAll = __riscv_vfmul_vf_f32m2(vVGAllBase, vLhsCurrent->d[0], cVl);
          vfloat32m2_t vVSum = __riscv_vfcvt_f_x_v_f32m2(vVIsum0, cVl);
          vVSum = __riscv_vfmul_vv_f32m2(vVSum, vVGAll, cVl);
          vVSumf0 = __riscv_vfadd_vv_f32m2(vVSumf0, vVSum, cVl);
        }
        // Row 1
        {
          vfloat32m2_t vVGAll = __riscv_vfmul_vf_f32m2(vVGAllBase, vLhsCurrent->d[1], cVl);
          vfloat32m2_t vVSum = __riscv_vfcvt_f_x_v_f32m2(vVIsum1, cVl);
          vVSum = __riscv_vfmul_vv_f32m2(vVSum, vVGAll, cVl);
          vVSumf1 = __riscv_vfadd_vv_f32m2(vVSumf1, vVSum, cVl);
        }
        // Row 2
        {
          vfloat32m2_t vVGAll = __riscv_vfmul_vf_f32m2(vVGAllBase, vLhsCurrent->d[2], cVl);
          vfloat32m2_t vVSum = __riscv_vfcvt_f_x_v_f32m2(vVIsum2, cVl);
          vVSum = __riscv_vfmul_vv_f32m2(vVSum, vVGAll, cVl);
          vVSumf2 = __riscv_vfadd_vv_f32m2(vVSumf2, vVSum, cVl);
        }
        // Row 3
        {
          vfloat32m2_t vVGAll = __riscv_vfmul_vf_f32m2(vVGAllBase, vLhsCurrent->d[3], cVl);
          vfloat32m2_t vVSum = __riscv_vfcvt_f_x_v_f32m2(vVIsum3, cVl);
          vVSum = __riscv_vfmul_vv_f32m2(vVSum, vVGAll, cVl);
          vVSumf3 = __riscv_vfadd_vv_f32m2(vVSumf3, vVSum, cVl);
        }

      } // End K-Block

      __riscv_vse32_v_f32m2(pS + (vRowTile + 0) * pBs + vColTile, vVSumf0, cVl);
      __riscv_vse32_v_f32m2(pS + (vRowTile + 1) * pBs + vColTile, vVSumf1, cVl);
      __riscv_vse32_v_f32m2(pS + (vRowTile + 2) * pBs + vColTile, vVSumf2, cVl);
      __riscv_vse32_v_f32m2(pS + (vRowTile + 3) * pBs + vColTile, vVSumf3, cVl);
    }
  }
}
#endif
