#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"

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

void quantize_row_q1_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q1_0_ref(pX, pY, pK);
}

void quantize_row_q2_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q2_0_ref(pX, pY, pK);
}

void quantize_row_q4_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q4_0_ref(pX, pY, pK);
}

void quantize_row_q4_1(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q4_1_ref(pX, pY, pK);
}

void quantize_row_q5_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q5_0_ref(pX, pY, pK);
}

void quantize_row_q5_1(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q5_1_ref(pX, pY, pK);
}

void quantize_row_q8_0_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q8_0_ref(pX, pY, pK);
}

void quantize_row_q8_1_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q8_1_ref(pX, pY, pK);
}

void quantize_row_mxfp4(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_mxfp4_ref(pX, pY, pK);
}

void quantize_row_nvfp4(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_nvfp4_ref(pX, pY, pK);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  quantize_row_q2_K_ref(pX, pVy, pK);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  quantize_row_q3_K_ref(pX, pVy, pK);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  block_q4_K *GGML_RESTRICT vY = pVy;
  quantize_row_q4_K_ref(pX, vY, pK);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  block_q5_K *GGML_RESTRICT vY = pVy;
  quantize_row_q5_K_ref(pX, vY, pK);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  block_q6_K *GGML_RESTRICT vY = pVy;
  quantize_row_q6_K_ref(pX, vY, pK);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  block_tq1_0 *GGML_RESTRICT vY = pVy;
  quantize_row_tq1_0_ref(pX, vY, pK);
}

void quantize_row_tq2_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  block_tq2_0 *GGML_RESTRICT vY = pVy;
  quantize_row_tq2_0_ref(pX, vY, pK);
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  quantize_row_q8_K_ref(pX, pY, pK);
}

//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK1_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  float vSumf = 0.0;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    float vSumi = 0.0f;

    for (int vK = 0; vK < 4; vK++) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vI * 4 + vK];
      const float cD1 = GGML_CPU_FP16_TO_FP32(vYb->d);
      int vSumiBlock = 0;

      const uint8_t *GGML_RESTRICT vBits = &vX[vI].qs[vK * 4];
      const int8_t *GGML_RESTRICT vQy = vYb->qs;

      for (int vB = 0; vB < 4; ++vB, vQy += 8) {
        const unsigned cMask = vBits[vB];
        vSumiBlock += ((cMask & 0x01) ? vQy[0] : -vQy[0]) + ((cMask & 0x02) ? vQy[1] : -vQy[1]) +
                      ((cMask & 0x04) ? vQy[2] : -vQy[2]) + ((cMask & 0x08) ? vQy[3] : -vQy[3]) +
                      ((cMask & 0x10) ? vQy[4] : -vQy[4]) + ((cMask & 0x20) ? vQy[5] : -vQy[5]) +
                      ((cMask & 0x40) ? vQy[6] : -vQy[6]) + ((cMask & 0x80) ? vQy[7] : -vQy[7]);
      }

      vSumi += cD1 * vSumiBlock;
    }

    vSumf += cD0 * vSumi;
  }

  *pS = vSumf;
}

void ggml_vec_dot_q2_0_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK2_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vI].d);

    float vSumi = 0.0f;

    // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
    for (int vK = 0; vK < 2; vK++) {
      const block_q8_0 *GGML_RESTRICT vYb = &vY[vI * 2 + vK];
      const float cD1 = GGML_CPU_FP16_TO_FP32(vYb->d);
      int vSumiBlock = 0;

      const uint8_t *GGML_RESTRICT vQs = &vX[vI].qs[vK * 8];
      const int8_t *GGML_RESTRICT vQy = vYb->qs;

      for (int vB = 0; vB < 8; ++vB) {
        const uint8_t cByte = vQs[vB];
        // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
        vSumiBlock += ((int)((cByte >> 0) & 3) - 1) * vQy[vB * 4 + 0];
        vSumiBlock += ((int)((cByte >> 2) & 3) - 1) * vQy[vB * 4 + 1];
        vSumiBlock += ((int)((cByte >> 4) & 3) - 1) * vQy[vB * 4 + 2];
        vSumiBlock += ((int)((cByte >> 6) & 3) - 1) * vQy[vB * 4 + 3];
      }

      vSumi += cD1 * vSumiBlock;
    }

    vSumf += cD0 * vSumi;
  }

  *pS = vSumf;
}

void ggml_vec_dot_q4_0_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cV0 = (vX[vIb].qs[vJ] & 0x0F);
      const int cV1 = (vX[vIb].qs[vJ] >> 4);

      vSumi0 += (cV0 * vY[vIb].qs[vJ]);
      vSumi1 += (cV1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_E8M0_TO_FP32_HALF(vX[vIb].e);

    int vSumi1 = 0;
    int vSumi2 = 0;
    for (int vJ = 0; vJ < QK_MXFP4 / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_mxfp4[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK_MXFP4 / 2] * kvalues_mxfp4[vX[vIb].qs[vJ] >> 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
void ggml_vec_dot_nvfp4_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_NVFP4 == 0);

  const block_nvfp4 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_NVFP4;

  float vSumf = 0;

  for (int vIb = 0; vIb < cNb; ++vIb) {
    for (int vSIdx = 0; vSIdx < 4; ++vSIdx) {
      const float cD = ggml_ue4m3_to_fp32(vX[vIb].d[vSIdx]);
      const int cQ8Block = vSIdx / 2;
      const int cQ8Off = (vSIdx % 2) * QK_NVFP4_SUB;
      const float cDy = GGML_CPU_FP16_TO_FP32(vY[2 * vIb + cQ8Block].d);

      int vSumiLo = 0, vSumiHi = 0;
      for (int vJ = 0; vJ < QK_NVFP4_SUB / 2; ++vJ) {
        const uint8_t cQv = vX[vIb].qs[vSIdx * (QK_NVFP4_SUB / 2) + vJ];
        vSumiLo += vY[2 * vIb + cQ8Block].qs[cQ8Off + vJ + 0] * kvalues_mxfp4[cQv & 0xf];
        vSumiHi += vY[2 * vIb + cQ8Block].qs[cQ8Off + vJ + QK_NVFP4_SUB / 2] * kvalues_mxfp4[cQv >> 4];
      }

      vSumf += cDy * cD * (vSumiLo + vSumiHi);
    }
  }
  *pS = vSumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    uint32_t vQh;
    memcpy(&vQh, vX[vIb].qh, sizeof(vQh));

    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh & (1u << (vJ + 0))) >> (vJ + 0)) << 4;
      const uint8_t cXh1 = ((vQh & (1u << (vJ + 16))) >> (vJ + 12));

      const int32_t cX0 = (int8_t)(((vX[vIb].qs[vJ] & 0x0F) | cXh0) - 16);
      const int32_t cX1 = (int8_t)(((vX[vIb].qs[vJ] >> 4) | cXh1) - 16);

      vSumi0 += (cX0 * vY[vIb].qs[vJ]);
      vSumi1 += (cX1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi;
  }

  *pS = vSumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    uint32_t vQh;
    memcpy(&vQh, vX[vIb].qh, sizeof(vQh));

    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh >> (vJ + 0)) << 4) & 0x10;
      const uint8_t cXh1 = ((vQh >> (vJ + 12))) & 0x10;

      const int32_t cX0 = (vX[vIb].qs[vJ] & 0xF) | cXh0;
      const int32_t cX1 = (vX[vIb].qs[vJ] >> 4) | cXh1;

      vSumi0 += (cX0 * vY[vIb].qs[vJ]);
      vSumi1 += (cX1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)) * vSumi +
             GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);
  }

  *pS = vSumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    int vSumi = 0;

    for (int vJ = 0; vJ < cQk; vJ++) {
      vSumi += vX[vIb].qs[vJ] * vY[vIb].qs[vJ];
    }

    vSumf += vSumi * (GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));
  }

  *pS = vSumf;
}

void ggml_vec_dot_tq1_0_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_tq1_0 *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  const uint8_t aPow3[6] = {1, 3, 9, 27, 81, 243};

  float vSumf = 0.0f;

  for (int vI = 0; vI < cNb; ++vI) {
    int vSum = 0;

    for (size_t vJ = 0; vJ < sizeof(vX->qs) - sizeof(vX->qs) % 32; vJ += 32) {
      for (size_t vL = 0; vL < 5; ++vL) {
        for (size_t vM = 0; vM < 32; ++vM) {
          uint8_t vQ = vX[vI].qs[vJ + vM] * aPow3[vL];
          uint16_t vXi = ((uint16_t)vQ * 3) >> 8;
          vSum += (vXi - 1) * vY[vI].qs[vJ * 5 + vL * 32 + vM];
        }
      }
    }
    for (size_t vJ = sizeof(vX->qs) - sizeof(vX->qs) % 32; vJ < sizeof(vX->qs); vJ += 16) {
      for (size_t vL = 0; vL < 5; ++vL) {
        for (size_t vM = 0; vM < 16; ++vM) {
          uint8_t vQ = vX[vI].qs[vJ + vM] * aPow3[vL];
          uint16_t vXi = ((uint16_t)vQ * 3) >> 8;
          vSum += (vXi - 1) * vY[vI].qs[vJ * 5 + vL * 16 + vM];
        }
      }
    }

    for (size_t vL = 0; vL < 4; ++vL) {
      for (size_t vJ = 0; vJ < sizeof(vX->qh); ++vJ) {
        uint8_t vQ = vX[vI].qh[vJ] * aPow3[vL];
        uint16_t vXi = ((uint16_t)vQ * 3) >> 8;
        vSum += (vXi - 1) * vY[vI].qs[sizeof(vX->qs) * 5 + vL * sizeof(vX->qh) + vJ];
      }
    }

    vSumf += (float)vSum * (GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d);
  }

  *pS = vSumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

    for (size_t vJ = 0; vJ < sizeof(vX->qs); vJ += 32) {
      for (size_t vL = 0; vL < 4; ++vL) {
        for (size_t vK = 0; vK < 32; ++vK) {
          vSumi += vY[vI].qs[vJ * 4 + vL * 32 + vK] * (((vX[vI].qs[vJ + vK] >> (vL * 2)) & 3) - 1);
        }
      }
    }

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    vSumf += (float)vSumi * cD;
  }

  *pS = vSumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;

  for (int vI = 0; vI < cNb; ++vI) {

    const uint8_t *vQ2 = vX[vI].qs;
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vSc = vX[vI].scales;

    int vSumms = 0;
    for (int vJ = 0; vJ < 16; ++vJ) {
      vSumms += vY[vI].bsums[vJ] * (vSc[vJ] >> 4);
    }

    const float cDall = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    int vIsum = 0;
    int vIs = 0;
    int vD;
    for (int vK = 0; vK < QK_K / 128; ++vK) {
      int vShift = 0;
      for (int vJ = 0; vJ < 4; ++vJ) {
        vD = vSc[vIs++] & 0xF;
        int vIsuml = 0;
        for (int vL = 0; vL < 16; ++vL)
          vIsuml += vQ8[vL] * ((vQ2[vL] >> vShift) & 3);
        vIsum += vD * vIsuml;
        vD = vSc[vIs++] & 0xF;
        vIsuml = 0;
        for (int vL = 16; vL < 32; ++vL)
          vIsuml += vQ8[vL] * ((vQ2[vL] >> vShift) & 3);
        vIsum += vD * vIsuml;
        vShift += 2;
        vQ8 += 32;
      }
      vQ2 += 32;
    }
    vSumf += cDall * vIsum - cDmin * vSumms;
  }
  *pS = vSumf;
}

void ggml_vec_dot_q3_K_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  // scalar version
  // This function is written like this so the compiler can manage to vectorize most of it
  // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
  // manually vectorized version above. Every other version I tried would run at least 4 times slower.
  // The ideal situation would be if we could just write the code once, and the compiler would
  // automatically produce the best possible set of machine instructions, instead of us having to manually
  // write vectorized versions for AVX, ARM_NEON, etc.

  int8_t aAux8[QK_K];
  int16_t aAux16[8];
  float aSums[8];
  int32_t aAux32[8];
  memset(aSums, 0, 8 * sizeof(float));

  uint32_t aAuxs[4];
  const int8_t *vScales = (const int8_t *)aAuxs;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vHm = vX[vI].hmask;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    memset(aAux32, 0, 8 * sizeof(int32_t));
    int8_t *GGML_RESTRICT vA = aAux8;
    uint8_t vM = 1;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = vQ3[vL] & 3;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] -= (vHm[vL] & vM ? 0 : 4);
      vA += 32;
      vM <<= 1;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (vQ3[vL] >> 2) & 3;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] -= (vHm[vL] & vM ? 0 : 4);
      vA += 32;
      vM <<= 1;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (vQ3[vL] >> 4) & 3;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] -= (vHm[vL] & vM ? 0 : 4);
      vA += 32;
      vM <<= 1;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (vQ3[vL] >> 6) & 3;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] -= (vHm[vL] & vM ? 0 : 4);
      vA += 32;
      vM <<= 1;
      vQ3 += 32;
    }
    vA = aAux8;

    memcpy(aAuxs, vX[vI].scales, 12);
    uint32_t vTmp = aAuxs[2];
    aAuxs[2] = ((aAuxs[0] >> 4) & cKmask2) | (((vTmp >> 4) & cKmask1) << 4);
    aAuxs[3] = ((aAuxs[1] >> 4) & cKmask2) | (((vTmp >> 6) & cKmask1) << 4);
    aAuxs[0] = (aAuxs[0] & cKmask2) | (((vTmp >> 0) & cKmask1) << 4);
    aAuxs[1] = (aAuxs[1] & cKmask2) | (((vTmp >> 2) & cKmask1) << 4);
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += (vScales[vJ] - 32) * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += (vScales[vJ] - 32) * aAux16[vL];
      vQ8 += 8;
      vA += 8;
    }
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    for (int vL = 0; vL < 8; ++vL)
      aSums[vL] += cD * aAux32[vL];
  }
  for (int vL = 0; vL < 8; ++vL)
    vSumf += aSums[vL];
  *pS = vSumf;
}

void ggml_vec_dot_q4_K_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

  uint32_t aUtmp[4];

  const uint8_t *vScales = (const uint8_t *)&aUtmp[0];
  const uint8_t *vMins = (const uint8_t *)&aUtmp[2];

  int8_t aAux8[QK_K];
  int16_t aAux16[8];
  float aSums[8];
  int32_t aAux32[8];
  memset(aSums, 0, 8 * sizeof(float));

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    memset(aAux32, 0, 8 * sizeof(int32_t));
    int8_t *GGML_RESTRICT vA = aAux8;
    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (int8_t)(vQ4[vL] & 0xF);
      vA += 32;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (int8_t)(vQ4[vL] >> 4);
      vA += 32;
      vQ4 += 32;
    }
    memcpy(aUtmp, vX[vI].scales, 12);
    aUtmp[3] = ((aUtmp[2] >> 4) & cKmask2) | (((aUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = aUtmp[1] & cKmask1;
    aUtmp[1] = (aUtmp[2] & cKmask2) | (((aUtmp[0] >> 6) & cKmask3) << 4);
    aUtmp[2] = cUaux;
    aUtmp[0] &= cKmask1;

    int vSumi = 0;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ)
      vSumi += vY[vI].bsums[vJ] * vMins[vJ / 2];
    vA = aAux8;
    int vIs = 0;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      int32_t vScale = vScales[vIs++];
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
    }
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    for (int vL = 0; vL < 8; ++vL)
      aSums[vL] += cD * aAux32[vL];
    const float cDmin = GGML_CPU_FP16_TO_FP32(vX[vI].dmin) * vY[vI].d;
    vSumf -= cDmin * vSumi;
  }
  for (int vL = 0; vL < 8; ++vL)
    vSumf += aSums[vL];
  *pS = vSumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                    size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  uint32_t aUtmp[4];

  const uint8_t *vScales = (const uint8_t *)&aUtmp[0];
  const uint8_t *vMins = (const uint8_t *)&aUtmp[2];

  int8_t aAux8[QK_K];
  int16_t aAux16[8];
  float aSums[8];
  int32_t aAux32[8];
  memset(aSums, 0, 8 * sizeof(float));

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vHm = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    memset(aAux32, 0, 8 * sizeof(int32_t));
    int8_t *GGML_RESTRICT vA = aAux8;
    uint8_t vM = 1;
    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (int8_t)(vQ4[vL] & 0xF);
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] += (vHm[vL] & vM ? 16 : 0);
      vA += 32;
      vM <<= 1;
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] = (int8_t)(vQ4[vL] >> 4);
      for (int vL = 0; vL < 32; ++vL)
        vA[vL] += (vHm[vL] & vM ? 16 : 0);
      vA += 32;
      vM <<= 1;
      vQ4 += 32;
    }
    memcpy(aUtmp, vX[vI].scales, 12);
    aUtmp[3] = ((aUtmp[2] >> 4) & cKmask2) | (((aUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = aUtmp[1] & cKmask1;
    aUtmp[1] = (aUtmp[2] & cKmask2) | (((aUtmp[0] >> 6) & cKmask3) << 4);
    aUtmp[2] = cUaux;
    aUtmp[0] &= cKmask1;

    int vSumi = 0;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ)
      vSumi += vY[vI].bsums[vJ] * vMins[vJ / 2];
    vA = aAux8;
    int vIs = 0;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      int32_t vScale = vScales[vIs++];
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
    }
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    for (int vL = 0; vL < 8; ++vL)
      aSums[vL] += cD * aAux32[vL];
    const float cDmin = GGML_CPU_FP16_TO_FP32(vX[vI].dmin) * vY[vI].d;
    vSumf -= cDmin * vSumi;
  }
  for (int vL = 0; vL < 8; ++vL)
    vSumf += aSums[vL];
  *pS = vSumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

  int8_t aAux8[QK_K];
  int16_t aAux16[8];
  float aSums[8];
  int32_t aAux32[8];
  memset(aSums, 0, 8 * sizeof(float));

  float vSumf = 0;
  for (int vI = 0; vI < cNb; ++vI) {
    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    memset(aAux32, 0, 8 * sizeof(int32_t));
    int8_t *GGML_RESTRICT vA = aAux8;
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
    vA = aAux8;
    int vIs = 0;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      int vScale = vX[vI].scales[vIs++];
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
      for (int vL = 0; vL < 8; ++vL)
        aAux16[vL] = vQ8[vL] * vA[vL];
      for (int vL = 0; vL < 8; ++vL)
        aAux32[vL] += vScale * aAux16[vL];
      vQ8 += 8;
      vA += 8;
    }
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    for (int vL = 0; vL < 8; ++vL)
      aSums[vL] += cD * aAux32[vL];
  }
  for (int vL = 0; vL < 8; ++vL)
    vSumf += aSums[vL];
  *pS = vSumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  uint32_t aAux32[2];
  const uint8_t *vAux8 = (const uint8_t *)aAux32;

  float vSumf = 0.f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    int32_t vBsum = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      memcpy(aAux32, vQ2, 2 * sizeof(uint32_t));
      vQ2 += 4;
      const uint32_t cLs = 2 * (aAux32[1] >> 28) + 1;
      int32_t vSumi = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2xxs_grid + vAux8[vL]);
        const uint8_t cSigns = ksigns_iq2xs[(aAux32[1] >> 7 * vL) & 127];
        for (int vJ = 0; vJ < 8; ++vJ) {
          vSumi += vGrid[vJ] * vQ8[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vBsum += vSumi * cLs;
    }
    vSumf += cD * vBsum;
  }
  *pS = 0.125f * vSumf;
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vSc = vX[vI].scales;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    int32_t vBsum = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      const uint16_t cLs1 = 2 * (vSc[vIb32] & 0xf) + 1;
      const uint16_t cLs2 = 2 * (vSc[vIb32] >> 4) + 1;
      int32_t vSumi = 0;
      for (int vL = 0; vL < 2; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2xs_grid + (vQ2[vL] & 511));
        const uint8_t cSigns = ksigns_iq2xs[vQ2[vL] >> 9];
        for (int vJ = 0; vJ < 8; ++vJ) {
          vSumi += vGrid[vJ] * vQ8[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vBsum += vSumi * cLs1;
      vSumi = 0;
      for (int vL = 2; vL < 4; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2xs_grid + (vQ2[vL] & 511));
        const uint8_t cSigns = ksigns_iq2xs[vQ2[vL] >> 9];
        for (int vJ = 0; vJ < 8; ++vJ) {
          vSumi += vGrid[vJ] * vQ8[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vBsum += vSumi * cLs2;
      vQ2 += 4;
    }
    vSumf += cD * vBsum;
  }
  *pS = 0.125f * vSumf;
}

void ggml_vec_dot_iq2_s_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint8_t *vSigns = vQs + QK_K / 8;

    int vBsum = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      int vLs1 = 1 + 2 * (vX[vI].scales[vIb32] & 0xf);
      int vLs2 = 1 + 2 * (vX[vI].scales[vIb32] >> 4);
      int vSumi1 = 0, vSumi2 = 0;
      for (int vL = 0; vL < 2; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2s_grid + (vQs[vL] | (vQh[vIb32] << (8 - 2 * vL) & 0x300)));
        for (int vJ = 0; vJ < 8; ++vJ) {
          vSumi1 += vQ8[vJ] * vGrid[vJ] * (vSigns[vL] & kmask_iq2xs[vJ] ? -1 : 1);
        }
        vQ8 += 8;
      }
      for (int vL = 2; vL < 4; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2s_grid + (vQs[vL] | (vQh[vIb32] << (8 - 2 * vL) & 0x300)));
        for (int vJ = 0; vJ < 8; ++vJ) {
          vSumi2 += vQ8[vJ] * vGrid[vJ] * (vSigns[vL] & kmask_iq2xs[vJ] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vBsum += vLs1 * vSumi1 + vLs2 * vSumi2;
      vQs += 4;
      vSigns += 4;
    }

    vSumf += cD * vBsum;
  }

  *pS = 0.125f * vSumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                       size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  uint32_t vAux32;

  float vSumf = 0.f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vGas = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    int32_t vBsum = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      memcpy(&vAux32, vGas, sizeof(uint32_t));
      vGas += sizeof(uint32_t);
      const uint32_t cLs = 2 * (vAux32 >> 28) + 1;
      int32_t vSumi = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid1 = (const uint8_t *)(iq3xxs_grid + vQ3[2 * vL + 0]);
        const uint8_t *vGrid2 = (const uint8_t *)(iq3xxs_grid + vQ3[2 * vL + 1]);
        const uint8_t cSigns = ksigns_iq2xs[(vAux32 >> 7 * vL) & 127];
        for (int vJ = 0; vJ < 4; ++vJ) {
          vSumi += vGrid1[vJ] * vQ8[vJ + 0] * (cSigns & kmask_iq2xs[vJ + 0] ? -1 : 1);
          vSumi += vGrid2[vJ] * vQ8[vJ + 4] * (cSigns & kmask_iq2xs[vJ + 4] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vQ3 += 8;
      vBsum += vSumi * cLs;
    }
    vSumf += cD * vBsum;
  }
  *pS = 0.25f * vSumf;
}

void ggml_vec_dot_iq3_s_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                     size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0.f;
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint8_t *GGML_RESTRICT vSigns = vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    int32_t vBsum = 0;
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const uint32_t cLs1 = 2 * (vX[vI].scales[vIb32 / 2] & 0xf) + 1;
      const uint32_t cLs2 = 2 * (vX[vI].scales[vIb32 / 2] >> 4) + 1;
      int32_t vSumi = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid1 =
            (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 0] | ((vQh[vIb32 + 0] << (8 - 2 * vL)) & 256)));
        const uint8_t *vGrid2 =
            (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 1] | ((vQh[vIb32 + 0] << (7 - 2 * vL)) & 256)));
        for (int vJ = 0; vJ < 4; ++vJ) {
          vSumi += vGrid1[vJ] * vQ8[vJ + 0] * (vSigns[vL] & kmask_iq2xs[vJ + 0] ? -1 : 1);
          vSumi += vGrid2[vJ] * vQ8[vJ + 4] * (vSigns[vL] & kmask_iq2xs[vJ + 4] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vQs += 8;
      vSigns += 4;
      vBsum += vSumi * cLs1;
      vSumi = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid1 =
            (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 0] | ((vQh[vIb32 + 1] << (8 - 2 * vL)) & 256)));
        const uint8_t *vGrid2 =
            (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 1] | ((vQh[vIb32 + 1] << (7 - 2 * vL)) & 256)));
        for (int vJ = 0; vJ < 4; ++vJ) {
          vSumi += vGrid1[vJ] * vQ8[vJ + 0] * (vSigns[vL] & kmask_iq2xs[vJ + 0] ? -1 : 1);
          vSumi += vGrid2[vJ] * vQ8[vJ + 4] * (vSigns[vL] & kmask_iq2xs[vJ + 4] ? -1 : 1);
        }
        vQ8 += 8;
      }
      vQs += 8;
      vSigns += 4;
      vBsum += vSumi * cLs2;
    }
    vSumf += cD * vBsum;
  }
  *pS = vSumf;
}

void ggml_vec_dot_iq1_s_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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
  for (int vI = 0; vI < cNb; vI++) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint16_t *vQh = vX[vI].qh;

    int vSumi = 0, vSumi1 = 0;
    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const int cLs = 2 * ((vQh[vIb] >> 12) & 7) + 1;
      const int cDelta = vQh[vIb] & 0x8000 ? -1 : 1;
      int vLsum = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const int8_t *vGrid = (const int8_t *)(iq1s_grid + (vQs[vL] | (((vQh[vIb] >> 3 * vL) & 7) << 8)));
        for (int vJ = 0; vJ < 8; ++vJ) {
          vLsum += vQ8[vJ] * vGrid[vJ];
        }
        vQ8 += 8;
      }
      vSumi += cLs * vLsum;
      vSumi1 += cLs * cDelta * (vY[vI].bsums[2 * vIb + 0] + vY[vI].bsums[2 * vIb + 1]);
      vQs += 4;
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d * (vSumi + IQ1S_DELTA * vSumi1);
  }

  *pS = vSumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
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

  int aSum1[2], aSum2[2], aDelta[4];

  float vSumf = 0;
  for (int vI = 0; vI < cNb; vI++) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint8_t *vQh = vX[vI].qh;
    const uint16_t *vSc = (const uint16_t *)vX[vI].scales;

    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);

    int vSumi1 = 0, vSumi2 = 0;
    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      aDelta[0] = vQh[0] & 0x08 ? -1 : 1;
      aDelta[1] = vQh[0] & 0x80 ? -1 : 1;
      aDelta[2] = vQh[1] & 0x08 ? -1 : 1;
      aDelta[3] = vQh[1] & 0x80 ? -1 : 1;
      aSum1[0] = aSum1[1] = aSum2[0] = aSum2[1] = 0;
      for (int vL = 0; vL < 4; ++vL) {
        const int8_t *vGrid =
            (const int8_t *)(iq1s_grid + (vQs[vL] | (((uint16_t)vQh[vL / 2] << (8 - 4 * (vL % 2))) & 0x700)));
        int vLsum1 = 0, vLsum2 = 0;
        for (int vJ = 0; vJ < 8; ++vJ) {
          vLsum1 += vQ8[vJ] * vGrid[vJ];
          vLsum2 += vQ8[vJ];
        }
        vQ8 += 8;
        aSum1[vL / 2] += vLsum1;
        aSum2[vL / 2] += vLsum2 * aDelta[vL];
      }

      const int cLs1 = 2 * ((vSc[vIb / 2] >> (6 * (vIb % 2) + 0)) & 0x7) + 1;
      const int cLs2 = 2 * ((vSc[vIb / 2] >> (6 * (vIb % 2) + 3)) & 0x7) + 1;

      vSumi1 += aSum1[0] * cLs1 + aSum1[1] * cLs2;
      vSumi2 += aSum2[0] * cLs1 + aSum2[1] * cLs2;
      vQs += 4;
      vQh += 2;
    }

    vSumf += GGML_CPU_FP16_TO_FP32(vScale.f16) * vY[vI].d * (vSumi1 + IQ1M_DELTA * vSumi2);
  }

  *pS = vSumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
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

  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    int vSumi1 = 0, vSumi2 = 0;
    for (int vJ = 0; vJ < QK4_NL / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_iq4nl[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK4_NL / 2] * kvalues_iq4nl[vX[vIb].qs[vJ] >> 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx,
                                      size_t pBx, const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  float vSumf = 0;
  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const float cD4d8 = GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d;
    uint16_t vH = vX[vIbl].scales_h;
    const uint8_t *vQs = vX[vIbl].qs;
    const int8_t *vQ8 = vY[vIbl].qs;
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const uint8_t cLs1 = (vX[vIbl].scales_l[vIb / 2] & 0xf) | ((vH << 4) & 0x30);
      const uint8_t cLs2 = (vX[vIbl].scales_l[vIb / 2] >> 4) | ((vH << 2) & 0x30);
      vH >>= 4;
      const float cD1 = cD4d8 * (cLs1 - 32);
      const float cD2 = cD4d8 * (cLs2 - 32);
      int vSumi1 = 0, vSumi2 = 0;
      for (int vJ = 0; vJ < 16; ++vJ) {
        vSumi1 += vQ8[vJ + 0] * kvalues_iq4nl[vQs[vJ] & 0xf];
        vSumi2 += vQ8[vJ + 16] * kvalues_iq4nl[vQs[vJ] >> 4];
      }
      vSumf += cD1 * (vSumi1 + vSumi2);
      vQs += 16;
      vQ8 += 32;
      vSumi1 = vSumi2 = 0;
      for (int vJ = 0; vJ < 16; ++vJ) {
        vSumi1 += vQ8[vJ + 0] * kvalues_iq4nl[vQs[vJ] & 0xf];
        vSumi2 += vQ8[vJ + 16] * kvalues_iq4nl[vQs[vJ] >> 4];
      }
      vSumf += cD2 * (vSumi1 + vSumi2);
      vQs += 16;
      vQ8 += 32;
    }
  }
  *pS = vSumf;
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK4_NL == 0);
  quantize_row_iq4_nl_ref(pX, pY, pK);
}

void quantize_row_iq4_xs(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  quantize_iq4_xs(pX, pY, 1, pK, NULL);
}
