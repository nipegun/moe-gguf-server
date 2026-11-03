#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

static inline int fBestIndexInt8(int pN, const int8_t *pVal, float pX) {
  if (pX <= pVal[0])
    return 0;
  if (pX >= pVal[pN - 1])
    return pN - 1;
  int vMl = 0, vMu = pN - 1;
  while (vMu - vMl > 1) {
    int vMav = (vMl + vMu) / 2;
    if (pX < pVal[vMav])
      vMu = vMav;
    else
      vMl = vMav;
  }
  return pX - pVal[vMu - 1] < pVal[vMu] - pX ? vMu - 1 : vMu;
}

// reference implementation for deterministic creation of model files
void quantize_row_q1_0_ref(const float *GGML_RESTRICT pX, block_q1_0 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK1_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vSumAbs = 0.0f;
    for (int vJ = 0; vJ < cQk; vJ++) {
      vSumAbs += fabsf(pX[vI * cQk + vJ]);
    }
    const float cD = vSumAbs / cQk;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    // Clear all bits first
    for (int vJ = 0; vJ < cQk / 8; ++vJ) {
      pY[vI].qs[vJ] = 0;
    }

    // Just store sign of each weight directly (no normalization)
    for (int vJ = 0; vJ < cQk; ++vJ) {
      const int cBitIndex = vJ;
      const int cByteIndex = cBitIndex / 8;
      const int cBitOffset = cBitIndex % 8;

      if (pX[vI * cQk + vJ] >= 0.0f) {
        pY[vI].qs[cByteIndex] |= (1 << cBitOffset);
      }
    }
  }
}

void quantize_row_q2_0_ref(const float *GGML_RESTRICT pX, block_q2_0 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK2_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    // Compute scale as max absolute value in the block
    float vAmax = 0.0f;
    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cA = fabsf(pX[vI * cQk + vJ]);
      if (cA > vAmax)
        vAmax = cA;
    }
    const float cD = vAmax;
    const float cId = cD > 0.0f ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    // Clear quant bytes
    for (int vJ = 0; vJ < cQk / 4; ++vJ) {
      pY[vI].qs[vJ] = 0;
    }

    // Encode 2-bit values: round(w/d) clamped to [-1, 2], then add 1
    // 00 (-1) = -scale, 01 (0) = 0, 10 (+1) = +scale, 11 (+2) = 2*scale
    for (int vJ = 0; vJ < cQk; ++vJ) {
      const float cW = pX[vI * cQk + vJ];
      int vQ = (int)roundf(cW * cId) + 1;
      if (vQ < 0)
        vQ = 0;
      if (vQ > 3)
        vQ = 3;
      const int cByteIndex = vJ / 4;
      const int cBitOffset = (vJ % 4) * 2;
      pY[vI].qs[cByteIndex] |= ((uint8_t)vQ << cBitOffset);
    }
  }
}

// reference implementation for deterministic creation of model files
void quantize_row_q4_0_ref(const float *GGML_RESTRICT pX, block_q4_0 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK4_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max
    float vMax = 0.0f;

    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cV = pX[vI * cQk + vJ];
      if (vAmax < fabsf(cV)) {
        vAmax = fabsf(cV);
        vMax = cV;
      }
    }

    const float cD = vMax / -8;
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const float cX0 = pX[vI * cQk + 0 + vJ] * cId;
      const float cX1 = pX[vI * cQk + cQk / 2 + vJ] * cId;

      const uint8_t cXi0 = MIN(15, (int8_t)(cX0 + 8.5f));
      const uint8_t cXi1 = MIN(15, (int8_t)(cX1 + 8.5f));

      pY[vI].qs[vJ] = cXi0;
      pY[vI].qs[vJ] |= cXi1 << 4;
    }
  }
}

void quantize_row_q4_1_ref(const float *GGML_RESTRICT pX, block_q4_1 *GGML_RESTRICT pY, int64_t pK) {
  const int cQk = QK4_1;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vMin = FLT_MAX;
    float vMax = -FLT_MAX;

    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cV = pX[vI * cQk + vJ];

      if (cV < vMin)
        vMin = cV;
      if (cV > vMax)
        vMax = cV;
    }

    const float cD = (vMax - vMin) / ((1 << 4) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);
    pY[vI].m = GGML_FP32_TO_FP16(vMin);

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const float cX0 = (pX[vI * cQk + 0 + vJ] - vMin) * cId;
      const float cX1 = (pX[vI * cQk + cQk / 2 + vJ] - vMin) * cId;

      const uint8_t cXi0 = MIN(15, (int8_t)(cX0 + 0.5f));
      const uint8_t cXi1 = MIN(15, (int8_t)(cX1 + 0.5f));

      pY[vI].qs[vJ] = cXi0;
      pY[vI].qs[vJ] |= cXi1 << 4;
    }
  }
}

void quantize_row_q5_0_ref(const float *GGML_RESTRICT pX, block_q5_0 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK5_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max
    float vMax = 0.0f;

    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cV = pX[vI * cQk + vJ];
      if (vAmax < fabsf(cV)) {
        vAmax = fabsf(cV);
        vMax = cV;
      }
    }

    const float cD = vMax / -16;
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    uint32_t vQh = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const float cX0 = pX[vI * cQk + 0 + vJ] * cId;
      const float cX1 = pX[vI * cQk + cQk / 2 + vJ] * cId;

      const uint8_t cXi0 = MIN(31, (int8_t)(cX0 + 16.5f));
      const uint8_t cXi1 = MIN(31, (int8_t)(cX1 + 16.5f));

      pY[vI].qs[vJ] = (cXi0 & 0x0F) | ((cXi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
      vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + cQk / 2);
    }

    memcpy(&pY[vI].qh, &vQh, sizeof(vQh));
  }
}

void quantize_row_q5_1_ref(const float *GGML_RESTRICT pX, block_q5_1 *GGML_RESTRICT pY, int64_t pK) {
  const int cQk = QK5_1;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vMin = FLT_MAX;
    float vMax = -FLT_MAX;

    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cV = pX[vI * cQk + vJ];

      if (cV < vMin)
        vMin = cV;
      if (cV > vMax)
        vMax = cV;
    }

    const float cD = (vMax - vMin) / ((1 << 5) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);
    pY[vI].m = GGML_FP32_TO_FP16(vMin);

    uint32_t vQh = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const float cX0 = (pX[vI * cQk + 0 + vJ] - vMin) * cId;
      const float cX1 = (pX[vI * cQk + cQk / 2 + vJ] - vMin) * cId;

      const uint8_t cXi0 = (uint8_t)(cX0 + 0.5f);
      const uint8_t cXi1 = (uint8_t)(cX1 + 0.5f);

      pY[vI].qs[vJ] = (cXi0 & 0x0F) | ((cXi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
      vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + cQk / 2);
    }

    memcpy(&pY[vI].qh, &vQh, sizeof(pY[vI].qh));
  }
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_0_ref(const float *GGML_RESTRICT pX, block_q8_0 *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  for (int vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max

    for (int vJ = 0; vJ < QK8_0; vJ++) {
      const float cV = pX[vI * QK8_0 + vJ];
      vAmax = MAX(vAmax, fabsf(cV));
    }

    const float cD = vAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    for (int vJ = 0; vJ < QK8_0; ++vJ) {
      const float cX0 = pX[vI * QK8_0 + vJ] * cId;

      pY[vI].qs[vJ] = roundf(cX0);
    }
  }
}

// reference implementation for deterministic creation of model files
void quantize_row_q8_1_ref(const float *GGML_RESTRICT pX, block_q8_1 *GGML_RESTRICT pY, int64_t pK) {
  assert(QK8_1 == 32);
  assert(pK % QK8_1 == 0);
  const int cNb = pK / QK8_1;

  for (int vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max

    for (int vJ = 0; vJ < QK8_1; vJ++) {
      const float cV = pX[vI * QK8_1 + vJ];
      vAmax = MAX(vAmax, fabsf(cV));
    }

    const float cD = vAmax / ((1 << 7) - 1);
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    int vSum = 0;

    for (int vJ = 0; vJ < QK8_1 / 2; ++vJ) {
      const float cV0 = pX[vI * QK8_1 + vJ] * cId;
      const float cV1 = pX[vI * QK8_1 + QK8_1 / 2 + vJ] * cId;

      pY[vI].qs[vJ] = roundf(cV0);
      pY[vI].qs[QK8_1 / 2 + vJ] = roundf(cV1);

      vSum += pY[vI].qs[vJ];
      vSum += pY[vI].qs[QK8_1 / 2 + vJ];
    }

    pY[vI].s = GGML_FP32_TO_FP16(vSum * cD);
  }
}

static inline int fBestIndexMxfp4(float pX, float pE) {
  int vBestIndex = 0;
  float vBestErr = fabsf(kvalues_mxfp4[0] * pE - pX);
  for (int vI = 1; vI < 16; vI++) {
    float vErr = fabsf(kvalues_mxfp4[vI] * pE - pX);
    if (vErr < vBestErr) {
      vBestIndex = vI;
      vBestErr = vErr;
    }
  }
  return vBestIndex;
}

void quantize_row_mxfp4_ref(const float *GGML_RESTRICT pX, block_mxfp4 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK_MXFP4;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max

    for (int vJ = 0; vJ < cQk; vJ++) {
      const float cV = pX[vI * cQk + vJ];

      if (vAmax < fabsf(cV)) {
        vAmax = fabsf(cV);
      }
    }

    const uint8_t cE = vAmax > 0.0f ? (uint8_t)(floorf(log2f(vAmax)) - 2 + 127) : 0;

    const float cD = GGML_E8M0_TO_FP32_HALF(cE);

    pY[vI].e = cE;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cX0 = fBestIndexMxfp4(pX[vI * cQk + 0 + vJ], cD);
      const uint8_t cX1 = fBestIndexMxfp4(pX[vI * cQk + cQk / 2 + vJ], cD);

      pY[vI].qs[vJ] = cX0;
      pY[vI].qs[vJ] |= cX1 << 4;
    }
  }
}

void quantize_row_nvfp4_ref(const float *GGML_RESTRICT pX, block_nvfp4 *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK_NVFP4;
  static const int cQkSub = QK_NVFP4_SUB;
  static const int cNSub = QK_NVFP4 / QK_NVFP4_SUB;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    for (int vS = 0; vS < cNSub; vS++) {
      const float *vXb = pX + vI * cQk + vS * cQkSub;

      float vAmax = 0.0f;
      for (int vJ = 0; vJ < cQkSub; vJ++) {
        if (vAmax < fabsf(vXb[vJ])) {
          vAmax = fabsf(vXb[vJ]);
        }
      }

      // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
      const uint8_t cUe = ggml_fp32_to_ue4m3(vAmax / 6.0f);
      pY[vI].d[vS] = cUe;
      const float cD = ggml_ue4m3_to_fp32(cUe);

      for (int vJ = 0; vJ < cQkSub / 2; ++vJ) {
        const uint8_t cX0 = fBestIndexMxfp4(vXb[0 + vJ], cD);
        const uint8_t cX1 = fBestIndexMxfp4(vXb[cQkSub / 2 + vJ], cD);

        pY[vI].qs[vS * (cQkSub / 2) + vJ] = cX0 | (cX1 << 4);
      }
    }
  }
}

void dequantize_row_q1_0(const block_q1_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK1_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cNegD = -cD;

    for (int vJ = 0; vJ < cQk; ++vJ) {
      const int cByteIndex = vJ / 8;
      const int cBitOffset = vJ % 8;
      const uint8_t cBit = (pX[vI].qs[cByteIndex] >> cBitOffset) & 1;
      pY[vI * cQk + vJ] = cBit ? cD : cNegD;
    }
  }
}

void dequantize_row_q2_0(const block_q2_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK2_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vJ = 0; vJ < cQk; ++vJ) {
      const int cByteIndex = vJ / 4;
      const int cBitOffset = (vJ % 4) * 2;
      const uint8_t cQ = (pX[vI].qs[cByteIndex] >> cBitOffset) & 0x03;
      // 00=-1, 01=0, 10=+1, 11=+2
      pY[vI * cQk + vJ] = ((int)cQ - 1) * cD;
    }
  }
}

void dequantize_row_q4_0(const block_q4_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK4_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cX0 = (pX[vI].qs[vJ] & 0x0F) - 8;
      const int cX1 = (pX[vI].qs[vJ] >> 4) - 8;

      pY[vI * cQk + vJ + 0] = cX0 * cD;
      pY[vI * cQk + vJ + cQk / 2] = cX1 * cD;
    }
  }
}

void dequantize_row_q4_1(const block_q4_1 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK4_1;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cM = GGML_FP16_TO_FP32(pX[vI].m);

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cX0 = (pX[vI].qs[vJ] & 0x0F);
      const int cX1 = (pX[vI].qs[vJ] >> 4);

      pY[vI * cQk + vJ + 0] = cX0 * cD + cM;
      pY[vI * cQk + vJ + cQk / 2] = cX1 * cD + cM;
    }
  }
}

void dequantize_row_q5_0(const block_q5_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK5_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    uint32_t vQh;
    memcpy(&vQh, pX[vI].qh, sizeof(vQh));

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh >> (vJ + 0)) << 4) & 0x10;
      const uint8_t cXh1 = ((vQh >> (vJ + 12))) & 0x10;

      const int32_t cX0 = ((pX[vI].qs[vJ] & 0x0F) | cXh0) - 16;
      const int32_t cX1 = ((pX[vI].qs[vJ] >> 4) | cXh1) - 16;

      pY[vI * cQk + vJ + 0] = cX0 * cD;
      pY[vI * cQk + vJ + cQk / 2] = cX1 * cD;
    }
  }
}

void dequantize_row_q5_1(const block_q5_1 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK5_1;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cM = GGML_FP16_TO_FP32(pX[vI].m);

    uint32_t vQh;
    memcpy(&vQh, pX[vI].qh, sizeof(vQh));

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const uint8_t cXh0 = ((vQh >> (vJ + 0)) << 4) & 0x10;
      const uint8_t cXh1 = ((vQh >> (vJ + 12))) & 0x10;

      const int cX0 = (pX[vI].qs[vJ] & 0x0F) | cXh0;
      const int cX1 = (pX[vI].qs[vJ] >> 4) | cXh1;

      pY[vI * cQk + vJ + 0] = cX0 * cD + cM;
      pY[vI * cQk + vJ + cQk / 2] = cX1 * cD + cM;
    }
  }
}

void dequantize_row_q8_0(const block_q8_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK8_0;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vJ = 0; vJ < cQk; ++vJ) {
      pY[vI * cQk + vJ] = pX[vI].qs[vJ] * cD;
    }
  }
}

void dequantize_row_mxfp4(const block_mxfp4 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK_MXFP4;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_E8M0_TO_FP32_HALF(pX[vI].e);

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int8_t cX0 = kvalues_mxfp4[pX[vI].qs[vJ] & 0x0F];
      const int8_t cX1 = kvalues_mxfp4[pX[vI].qs[vJ] >> 4];

      pY[vI * cQk + vJ + 0] = cX0 * cD;
      pY[vI * cQk + vJ + cQk / 2] = cX1 * cD;
    }
  }
}

void dequantize_row_nvfp4(const block_nvfp4 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  static const int cQk = QK_NVFP4;
  static const int cQkSub = QK_NVFP4_SUB;
  static const int cNSub = QK_NVFP4 / QK_NVFP4_SUB;

  assert(pK % cQk == 0);

  const int cNb = pK / cQk;

  for (int vI = 0; vI < cNb; vI++) {
    for (int vS = 0; vS < cNSub; vS++) {
      const float cD = ggml_ue4m3_to_fp32(pX[vI].d[vS]);
      float *vYb = pY + vI * cQk + vS * cQkSub;

      for (int vJ = 0; vJ < cQkSub / 2; ++vJ) {
        const int8_t cV0 = kvalues_mxfp4[pX[vI].qs[vS * (cQkSub / 2) + vJ] & 0x0F];
        const int8_t cV1 = kvalues_mxfp4[pX[vI].qs[vS * (cQkSub / 2) + vJ] >> 4];

        vYb[vJ + 0] = cV0 * cD;
        vYb[vJ + cQkSub / 2] = cV1 * cD;
      }
    }
  }
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
static inline int fNearestInt(float pFval) {
  assert(fabsf(pFval) <= 4194303.f);
  float vVal = pFval + 12582912.f;
  int vI;
  memcpy(&vI, &vVal, sizeof(int));
  return (vI & 0x007fffff) - 0x00400000;
}

static float fMakeQxQuants(int pN, int pNmax, const float *GGML_RESTRICT pX, int8_t *GGML_RESTRICT pL, int pRmseType,
                           const float *GGML_RESTRICT pQw) {
  float vMax = 0;
  float vAmax = 0;
  for (int vI = 0; vI < pN; ++vI) {
    float vAx = fabsf(pX[vI]);
    if (vAx > vAmax) {
      vAmax = vAx;
      vMax = pX[vI];
    }
  }
  if (vAmax < GROUP_MAX_EPS) { // all zero
    for (int vI = 0; vI < pN; ++vI) {
      pL[vI] = 0;
    }
    return 0.f;
  }
  float vIscale = -pNmax / vMax;
  if (pRmseType == 0) {
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * pX[vI]);
      pL[vI] = pNmax + MAX(-pNmax, MIN(pNmax - 1, vL));
    }
    return 1 / vIscale;
  }
  bool vReturnEarly = false;
  if (pRmseType < 0) {
    pRmseType = -pRmseType;
    vReturnEarly = true;
  }
  float vSumlx = 0;
  float vSuml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int vI = 0; vI < pN; ++vI) {
#else
  for (int vI = 0; vI < pN; ++vI) {
#endif
    int vL = fNearestInt(vIscale * pX[vI]);
    vL = MAX(-pNmax, MIN(pNmax - 1, vL));
    pL[vI] = vL + pNmax;
    float vW = pQw              ? pQw[vI]
               : pRmseType == 1 ? pX[vI] * pX[vI]
               : pRmseType == 2 ? 1
               : pRmseType == 3 ? fabsf(pX[vI])
                                : sqrtf(fabsf(pX[vI]));
    vSumlx += vW * pX[vI] * vL;
    vSuml2 += vW * vL * vL;
  }
  float vScale = vSuml2 ? vSumlx / vSuml2 : 0.0f;
  if (vReturnEarly)
    return vSuml2 > 0 ? 0.5f * (vScale + 1 / vIscale) : 1 / vIscale;
  float vBest = vScale * vSumlx;
  for (int vIs = -9; vIs <= 9; ++vIs) {
    if (vIs == 0) {
      continue;
    }
    vIscale = -(pNmax + 0.1f * vIs) / vMax;
    vSumlx = vSuml2 = 0;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * pX[vI]);
      vL = MAX(-pNmax, MIN(pNmax - 1, vL));
      float vW = pQw              ? pQw[vI]
                 : pRmseType == 1 ? pX[vI] * pX[vI]
                 : pRmseType == 2 ? 1
                 : pRmseType == 3 ? fabsf(pX[vI])
                                  : sqrtf(fabsf(pX[vI]));
      vSumlx += vW * pX[vI] * vL;
      vSuml2 += vW * vL * vL;
    }
    if (vSuml2 > 0 && vSumlx * vSumlx > vBest * vSuml2) {
      for (int vI = 0; vI < pN; ++vI) {
        int vL = fNearestInt(vIscale * pX[vI]);
        pL[vI] = pNmax + MAX(-pNmax, MIN(pNmax - 1, vL));
      }
      vScale = vSumlx / vSuml2;
      vBest = vScale * vSumlx;
    }
  }
  return vScale;
}

static float fMakeQ3Quants(int pN, int pNmax, const float *GGML_RESTRICT pX, int8_t *GGML_RESTRICT pL, bool pDoRmse) {
  float vMax = 0;
  float vAmax = 0;
  for (int vI = 0; vI < pN; ++vI) {
    float vAx = fabsf(pX[vI]);
    if (vAx > vAmax) {
      vAmax = vAx;
      vMax = pX[vI];
    }
  }
  if (vAmax < GROUP_MAX_EPS) { // all zero
    for (int vI = 0; vI < pN; ++vI) {
      pL[vI] = 0;
    }
    return 0.f;
  }
  float vIscale = -pNmax / vMax;
  if (pDoRmse) {
    float vSumlx = 0;
    float vSuml2 = 0;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * pX[vI]);
      vL = MAX(-pNmax, MIN(pNmax - 1, vL));
      pL[vI] = vL;
      float vW = pX[vI] * pX[vI];
      vSumlx += vW * pX[vI] * vL;
      vSuml2 += vW * vL * vL;
    }
    for (int vItry = 0; vItry < 5; ++vItry) {
      int vNChanged = 0;
      for (int vI = 0; vI < pN; ++vI) {
        float vW = pX[vI] * pX[vI];
        float vSlx = vSumlx - vW * pX[vI] * pL[vI];
        if (vSlx > 0) {
          float vSl2 = vSuml2 - vW * pL[vI] * pL[vI];
          int vNewL = fNearestInt(pX[vI] * vSl2 / vSlx);
          vNewL = MAX(-pNmax, MIN(pNmax - 1, vNewL));
          if (vNewL != pL[vI]) {
            vSlx += vW * pX[vI] * vNewL;
            vSl2 += vW * vNewL * vNewL;
            if (vSl2 > 0 && vSlx * vSlx * vSuml2 > vSumlx * vSumlx * vSl2) {
              pL[vI] = vNewL;
              vSumlx = vSlx;
              vSuml2 = vSl2;
              ++vNChanged;
            }
          }
        }
      }
      if (!vNChanged) {
        break;
      }
    }
    for (int vI = 0; vI < pN; ++vI) {
      pL[vI] += pNmax;
    }
    return vSuml2 > 0.0f ? vSumlx / vSuml2 : 0.0f;
  }
  for (int vI = 0; vI < pN; ++vI) {
    int vL = fNearestInt(vIscale * pX[vI]);
    vL = MAX(-pNmax, MIN(pNmax - 1, vL));
    pL[vI] = vL + pNmax;
  }
  return 1 / vIscale;
}

static float fMakeQkx1Quants(int pN, int pNmax, const float *GGML_RESTRICT pX, uint8_t *GGML_RESTRICT pL,
                             float *GGML_RESTRICT pTheMin, int pNtry, float pAlpha) {
  float vMin = pX[0];
  float vMax = pX[0];
  for (int vI = 1; vI < pN; ++vI) {
    if (pX[vI] < vMin)
      vMin = pX[vI];
    if (pX[vI] > vMax)
      vMax = pX[vI];
  }
  if (vMax == vMin) {
    for (int vI = 0; vI < pN; ++vI)
      pL[vI] = 0;
    *pTheMin = 0;
    return 0.f;
  }
  if (vMin > 0)
    vMin = 0;
  float vIscale = pNmax / (vMax - vMin);
  float vScale = 1 / vIscale;
  for (int vItry = 0; vItry < pNtry; ++vItry) {
    float vSumlx = 0;
    int vSuml2 = 0;
    bool vDidChange = false;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * (pX[vI] - vMin));
      vL = MAX(0, MIN(pNmax, vL));
      if (vL != pL[vI]) {
        pL[vI] = vL;
        vDidChange = true;
      }
      vSumlx += (pX[vI] - vMin) * vL;
      vSuml2 += vL * vL;
    }
    vScale = vSumlx / vSuml2;
    float vSum = 0;
    for (int vI = 0; vI < pN; ++vI) {
      vSum += pX[vI] - vScale * pL[vI];
    }
    vMin = pAlpha * vMin + (1 - pAlpha) * vSum / pN;
    if (vMin > 0)
      vMin = 0;
    vIscale = 1 / vScale;
    if (!vDidChange)
      break;
  }
  *pTheMin = -vMin;
  return vScale;
}

static float fMakeQkx2Quants(int pN, int pNmax, const float *GGML_RESTRICT pX, const float *GGML_RESTRICT pWeights,
                             uint8_t *GGML_RESTRICT pL, float *GGML_RESTRICT pTheMin, uint8_t *GGML_RESTRICT pLaux,
                             float pRmin, float pRdelta, int pNstep, bool pUseMad) {
  float vMin = pX[0];
  float vMax = pX[0];
  float vSumW = pWeights[0];
  float vSumX = vSumW * pX[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int vI = 1; vI < pN; ++vI) {
#else
  for (int vI = 1; vI < pN; ++vI) {
#endif
    if (pX[vI] < vMin)
      vMin = pX[vI];
    if (pX[vI] > vMax)
      vMax = pX[vI];
    float vW = pWeights[vI];
    vSumW += vW;
    vSumX += vW * pX[vI];
  }
  if (vMin > 0)
    vMin = 0;
  if (vMax == vMin) {
    for (int vI = 0; vI < pN; ++vI)
      pL[vI] = 0;
    *pTheMin = -vMin;
    return 0.f;
  }
  float vIscale = pNmax / (vMax - vMin);
  float vScale = 1 / vIscale;
  float vBestError = 0;
  for (int vI = 0; vI < pN; ++vI) {
    int vL = fNearestInt(vIscale * (pX[vI] - vMin));
    pL[vI] = MAX(0, MIN(pNmax, vL));
    float vDiff = vScale * pL[vI] + vMin - pX[vI];
    vDiff = pUseMad ? fabsf(vDiff) : vDiff * vDiff;
    float vW = pWeights[vI];
    vBestError += vW * vDiff;
  }
  if (pNstep < 1) {
    *pTheMin = -vMin;
    return vScale;
  }
  for (int vIs = 0; vIs <= pNstep; ++vIs) {
    vIscale = (pRmin + pRdelta * vIs + pNmax) / (vMax - vMin);
    float vSumL = 0, vSumL2 = 0, vSumXl = 0;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * (pX[vI] - vMin));
      vL = MAX(0, MIN(pNmax, vL));
      pLaux[vI] = vL;
      float vW = pWeights[vI];
      vSumL += vW * vL;
      vSumL2 += vW * vL * vL;
      vSumXl += vW * vL * pX[vI];
    }
    float vD = vSumW * vSumL2 - vSumL * vSumL;
    if (vD > 0) {
      float vThisScale = (vSumW * vSumXl - vSumX * vSumL) / vD;
      float vThisMin = (vSumL2 * vSumX - vSumL * vSumXl) / vD;
      if (vThisMin > 0) {
        vThisMin = 0;
        vThisScale = vSumXl / vSumL2;
      }
      float vCurError = 0;
      for (int vI = 0; vI < pN; ++vI) {
        float vDiff = vThisScale * pLaux[vI] + vThisMin - pX[vI];
        vDiff = pUseMad ? fabsf(vDiff) : vDiff * vDiff;
        float vW = pWeights[vI];
        vCurError += vW * vDiff;
      }
      if (vCurError < vBestError) {
        for (int vI = 0; vI < pN; ++vI) {
          pL[vI] = pLaux[vI];
        }
        vBestError = vCurError;
        vScale = vThisScale;
        vMin = vThisMin;
      }
    }
  }
  *pTheMin = -vMin;
  return vScale;
}

static inline void fGetScaleMinK4(int pJ, const uint8_t *GGML_RESTRICT pQ, uint8_t *GGML_RESTRICT pD,
                                  uint8_t *GGML_RESTRICT pM) {
  if (pJ < 4) {
    *pD = pQ[pJ] & 63;
    *pM = pQ[pJ + 4] & 63;
  } else {
    *pD = (pQ[pJ + 4] & 0xF) | ((pQ[pJ - 4] >> 6) << 4);
    *pM = (pQ[pJ + 4] >> 4) | ((pQ[pJ - 0] >> 6) << 4);
  }
}

//========================- 2-bit (de)-quantization

void quantize_row_q2_K_ref(const float *GGML_RESTRICT pX, block_q2_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  uint8_t L[QK_K];
  uint8_t aLaux[16];
  float aWeights[16];
  float aMins[QK_K / 16];
  float vScales[QK_K / 16];

  const float cQ4scale = 15.f;

  for (int vI = 0; vI < cNb; vI++) {
    float vMaxScale = 0; // as we are deducting the min, scales are always positive
    float vMaxMin = 0;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      for (int l = 0; l < 16; ++l)
        aWeights[l] = fabsf(pX[16 * vJ + l]);
      vScales[vJ] =
          fMakeQkx2Quants(16, 3, pX + 16 * vJ, aWeights, L + 16 * vJ, &aMins[vJ], aLaux, -0.5f, 0.1f, 15, true);
      float vScale = vScales[vJ];
      if (vScale > vMaxScale) {
        vMaxScale = vScale;
      }
      float vMin = aMins[vJ];
      if (vMin > vMaxMin) {
        vMaxMin = vMin;
      }
    }

    if (vMaxScale > 0) {
      float vIscale = cQ4scale / vMaxScale;
      for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
        int l = fNearestInt(vIscale * vScales[vJ]);
        pY[vI].scales[vJ] = l;
      }
      pY[vI].d = GGML_FP32_TO_FP16(vMaxScale / cQ4scale);
    } else {
      for (int vJ = 0; vJ < QK_K / 16; ++vJ)
        pY[vI].scales[vJ] = 0;
      pY[vI].d = GGML_FP32_TO_FP16(0.f);
    }
    if (vMaxMin > 0) {
      float vIscale = cQ4scale / vMaxMin;
      for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
        int l = fNearestInt(vIscale * aMins[vJ]);
        pY[vI].scales[vJ] |= (l << 4);
      }
      pY[vI].dmin = GGML_FP32_TO_FP16(vMaxMin / cQ4scale);
    } else {
      pY[vI].dmin = GGML_FP32_TO_FP16(0.f);
    }
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      const float cD = GGML_FP16_TO_FP32(pY[vI].d) * (pY[vI].scales[vJ] & 0xF);
      if (!cD)
        continue;
      const float cDm = GGML_FP16_TO_FP32(pY[vI].dmin) * (pY[vI].scales[vJ] >> 4);
      for (int vIi = 0; vIi < 16; ++vIi) {
        int l = fNearestInt((pX[16 * vJ + vIi] + cDm) / cD);
        l = MAX(0, MIN(3, l));
        L[16 * vJ + vIi] = l;
      }
    }

    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        pY[vI].qs[vJ / 4 + l] = L[vJ + l] | (L[vJ + l + 32] << 2) | (L[vJ + l + 64] << 4) | (L[vJ + l + 96] << 6);
      }
    }

    pX += QK_K;
  }
}

void dequantize_row_q2_K(const block_q2_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cMin = GGML_FP16_TO_FP32(pX[vI].dmin);

    const uint8_t *vQ = pX[vI].qs;

    int vIs = 0;
    float vDl, vMl;
    for (int vN = 0; vN < QK_K; vN += 128) {
      int vShift = 0;
      for (int vJ = 0; vJ < 4; ++vJ) {

        uint8_t vSc = pX[vI].scales[vIs++];
        vDl = cD * (vSc & 0xF);
        vMl = cMin * (vSc >> 4);
        for (int vL = 0; vL < 16; ++vL)
          *pY++ = vDl * ((int8_t)((vQ[vL] >> vShift) & 3)) - vMl;

        vSc = pX[vI].scales[vIs++];
        vDl = cD * (vSc & 0xF);
        vMl = cMin * (vSc >> 4);
        for (int vL = 0; vL < 16; ++vL)
          *pY++ = vDl * ((int8_t)((vQ[vL + 16] >> vShift) & 3)) - vMl;

        vShift += 2;
      }
      vQ += 32;
    }
  }
}

static float fMakeQkx3Quants(int pN, int pNmax, const float *GGML_RESTRICT pX, const float *GGML_RESTRICT pWeights,
                             uint8_t *GGML_RESTRICT pL, float *GGML_RESTRICT pTheMin, uint8_t *GGML_RESTRICT pLaux,
                             float pRmin, float pRdelta, int pNstep, bool pUseMad) {
  float vMin = pX[0];
  float vMax = pX[0];
  float vSumW = pWeights ? pWeights[0] : pX[0] * pX[0];
  float vSumX = vSumW * pX[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int vI = 1; vI < pN; ++vI) {
#else
  for (int vI = 1; vI < pN; ++vI) {
#endif
    if (pX[vI] < vMin)
      vMin = pX[vI];
    if (pX[vI] > vMax)
      vMax = pX[vI];
    float vW = pWeights ? pWeights[vI] : pX[vI] * pX[vI];
    vSumW += vW;
    vSumX += vW * pX[vI];
  }
  if (vMin > 0) {
    vMin = 0;
  }
  if (vMax <= vMin) {
    memset(pL, 0, pN);
    *pTheMin = -vMin;
    return 0.f;
  }
  float vIscale = pNmax / (vMax - vMin);
  float vScale = 1 / vIscale;
  float vBestMad = 0;
  for (int vI = 0; vI < pN; ++vI) {
    int vL = fNearestInt(vIscale * (pX[vI] - vMin));
    pL[vI] = MAX(0, MIN(pNmax, vL));
    float vDiff = vScale * pL[vI] + vMin - pX[vI];
    vDiff = pUseMad ? fabsf(vDiff) : vDiff * vDiff;
    float vW = pWeights ? pWeights[vI] : pX[vI] * pX[vI];
    vBestMad += vW * vDiff;
  }
  if (pNstep < 1) {
    *pTheMin = -vMin;
    return vScale;
  }
  for (int vIs = 0; vIs <= pNstep; ++vIs) {
    vIscale = (pRmin + pRdelta * vIs + pNmax) / (vMax - vMin);
    float vSumL = 0, vSumL2 = 0, vSumXl = 0;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscale * (pX[vI] - vMin));
      vL = MAX(0, MIN(pNmax, vL));
      pLaux[vI] = vL;
      float vW = pWeights ? pWeights[vI] : pX[vI] * pX[vI];
      vSumL += vW * vL;
      vSumL2 += vW * vL * vL;
      vSumXl += vW * vL * pX[vI];
    }
    float vD = vSumW * vSumL2 - vSumL * vSumL;
    if (vD > 0) {
      float vThisScale = (vSumW * vSumXl - vSumX * vSumL) / vD;
      float vThisMin = (vSumL2 * vSumX - vSumL * vSumXl) / vD;
      if (vThisMin > 0) {
        vThisMin = 0;
        vThisScale = vSumXl / vSumL2;
      }
      float vMad = 0;
      for (int vI = 0; vI < pN; ++vI) {
        float vDiff = vThisScale * pLaux[vI] + vThisMin - pX[vI];
        vDiff = pUseMad ? fabsf(vDiff) : vDiff * vDiff;
        float vW = pWeights ? pWeights[vI] : pX[vI] * pX[vI];
        vMad += vW * vDiff;
      }
      if (vMad < vBestMad) {
        for (int vI = 0; vI < pN; ++vI) {
          pL[vI] = pLaux[vI];
        }
        vBestMad = vMad;
        vScale = vThisScale;
        vMin = vThisMin;
      }
    }
  }
  *pTheMin = -vMin;
  return vScale;
}

static float fMakeQpQuants(int pN, int pNmax, const float *GGML_RESTRICT pX, uint8_t *GGML_RESTRICT pL,
                           const float *pQuantWeights) {
  float vMax = 0;
  for (int vI = 0; vI < pN; ++vI) {
    vMax = MAX(vMax, pX[vI]);
  }
  if (vMax < GROUP_MAX_EPS) { // all zero
    for (int vI = 0; vI < pN; ++vI) {
      pL[vI] = 0;
    }
    return 0.f;
  }
  float vIscale = pNmax / vMax;
  for (int vI = 0; vI < pN; ++vI) {
    pL[vI] = fNearestInt(vIscale * pX[vI]);
  }
  float vScale = 1 / vIscale;
  float vBestMse = 0;
  for (int vI = 0; vI < pN; ++vI) {
    float vDiff = pX[vI] - vScale * pL[vI];
    float vW = pQuantWeights[vI];
    vBestMse += vW * vDiff * vDiff;
  }
  for (int vIs = -4; vIs <= 4; ++vIs) {
    if (vIs == 0)
      continue;
    float vIscaleIs = (0.1f * vIs + pNmax) / vMax;
    float vScaleIs = 1 / vIscaleIs;
    float vMse = 0;
    for (int vI = 0; vI < pN; ++vI) {
      int vL = fNearestInt(vIscaleIs * pX[vI]);
      vL = MIN(pNmax, vL);
      float vDiff = pX[vI] - vScaleIs * vL;
      float vW = pQuantWeights[vI];
      vMse += vW * vDiff * vDiff;
    }
    if (vMse < vBestMse) {
      vBestMse = vMse;
      vIscale = vIscaleIs;
    }
  }
  float vSumlx = 0;
  float vSuml2 = 0;
  for (int vI = 0; vI < pN; ++vI) {
    int vL = fNearestInt(vIscale * pX[vI]);
    vL = MIN(pNmax, vL);
    pL[vI] = vL;
    float vW = pQuantWeights[vI];
    vSumlx += vW * pX[vI] * vL;
    vSuml2 += vW * vL * vL;
  }
  for (int vItry = 0; vItry < 5; ++vItry) {
    int vNChanged = 0;
    for (int vI = 0; vI < pN; ++vI) {
      float vW = pQuantWeights[vI];
      float vSlx = vSumlx - vW * pX[vI] * pL[vI];
      float vSl2 = vSuml2 - vW * pL[vI] * pL[vI];
      if (vSlx > 0 && vSl2 > 0) {
        int vNewL = fNearestInt(pX[vI] * vSl2 / vSlx);
        vNewL = MIN(pNmax, vNewL);
        if (vNewL != pL[vI]) {
          vSlx += vW * pX[vI] * vNewL;
          vSl2 += vW * vNewL * vNewL;
          if (vSlx * vSlx * vSuml2 > vSumlx * vSumlx * vSl2) {
            pL[vI] = vNewL;
            vSumlx = vSlx;
            vSuml2 = vSl2;
            ++vNChanged;
          }
        }
      }
    }
    if (!vNChanged) {
      break;
    }
  }
  return vSuml2 > 0.0f ? vSumlx / vSuml2 : 0.0f;
}

static void fQuantizeRowQ2KImpl(const float *GGML_RESTRICT pX, block_q2_K *GGML_RESTRICT pY, int pK,
                                const float *GGML_RESTRICT pQuantWeights) {
  GGML_ASSERT(pQuantWeights);
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;
  const bool cRequantize = true;

  uint8_t L[QK_K];
  uint8_t aLaux[16];
  float aMins[QK_K / 16];
  float vScales[QK_K / 16];
  float aSw[QK_K / 16];
  float aWeight[16];
  uint8_t vLs[QK_K / 16], vLm[QK_K / 16];

  for (int vI = 0; vI < cNb; vI++) {
    memset(aSw, 0, QK_K / 16 * sizeof(float));
    float vSumx2 = 0;
    for (int vJ = 0; vJ < QK_K; ++vJ)
      vSumx2 += pX[vJ] * pX[vJ];
    float vSigma2 = vSumx2 / QK_K;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      const float *GGML_RESTRICT vQw = pQuantWeights + QK_K * vI + 16 * vJ;
      for (int l = 0; l < 16; ++l)
        aWeight[l] = vQw[l] * sqrtf(vSigma2 + pX[16 * vJ + l] * pX[16 * vJ + l]);
      for (int l = 0; l < QK_K / 16; ++l)
        aSw[vJ] += aWeight[l];
      vScales[vJ] =
          fMakeQkx3Quants(16, 3, pX + 16 * vJ, aWeight, L + 16 * vJ, &aMins[vJ], aLaux, -0.9f, 0.05f, 36, false);
    }

    float vDm, vMm;
    vDm = fMakeQpQuants(QK_K / 16, 15, vScales, vLs, aSw);
    vMm = fMakeQpQuants(QK_K / 16, 15, aMins, vLm, aSw);

    pY[vI].d = GGML_FP32_TO_FP16(vDm);
    pY[vI].dmin = GGML_FP32_TO_FP16(vMm);
    vDm = GGML_FP16_TO_FP32(pY[vI].d);
    vMm = GGML_FP16_TO_FP32(pY[vI].dmin);

    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      pY[vI].scales[vJ] = vLs[vJ] | (vLm[vJ] << 4);
    }

    if (cRequantize) {
      for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
        const float cD = vDm * (pY[vI].scales[vJ] & 0xF);
        if (!cD)
          continue;
        const float cM = vMm * (pY[vI].scales[vJ] >> 4);
        for (int vIi = 0; vIi < 16; ++vIi) {
          int l = fNearestInt((pX[16 * vJ + vIi] + cM) / cD);
          l = MAX(0, MIN(3, l));
          L[16 * vJ + vIi] = l;
        }
      }
    }

    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        pY[vI].qs[vJ / 4 + l] = L[vJ + l] | (L[vJ + l + 32] << 2) | (L[vJ + l + 64] << 4) | (L[vJ + l + 96] << 6);
      }
    }

    pX += QK_K;
  }
}

size_t quantize_q2_K(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q2_K, pNPerRow);
  if (!pQuantWeights) {
    quantize_row_q2_K_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  } else {
    char *vQrow = (char *)pDst;
    for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
      fQuantizeRowQ2KImpl(pSrc, (block_q2_K *)vQrow, pNPerRow, pQuantWeights);
      pSrc += pNPerRow;
      vQrow += vRowSize;
    }
  }
  return pNrow * vRowSize;
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K_ref(const float *GGML_RESTRICT pX, block_q3_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  int8_t L[QK_K];
  float vScales[QK_K / 16];

  for (int vI = 0; vI < cNb; vI++) {

    float vMaxScale = 0;
    float vAmax = 0;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      vScales[vJ] = fMakeQ3Quants(16, 4, pX + 16 * vJ, L + 16 * vJ, true);
      float vScale = fabsf(vScales[vJ]);
      if (vScale > vAmax) {
        vAmax = vScale;
        vMaxScale = vScales[vJ];
      }
    }

    memset(pY[vI].scales, 0, 12);
    if (vMaxScale) {
      float vIscale = -32.f / vMaxScale;
      for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
        int8_t l = fNearestInt(vIscale * vScales[vJ]);
        l = MAX(-32, MIN(31, l)) + 32;
        if (vJ < 8) {
          pY[vI].scales[vJ] = l & 0xF;
        } else {
          pY[vI].scales[vJ - 8] |= ((l & 0xF) << 4);
        }
        l >>= 4;
        pY[vI].scales[vJ % 4 + 8] |= (l << (2 * (vJ / 4)));
      }
      pY[vI].d = GGML_FP32_TO_FP16(1 / vIscale);
    } else {
      pY[vI].d = GGML_FP32_TO_FP16(0.f);
    }

    int8_t vSc;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      vSc = vJ < 8 ? pY[vI].scales[vJ] & 0xF : pY[vI].scales[vJ - 8] >> 4;
      vSc = (vSc | (((pY[vI].scales[8 + vJ % 4] >> (2 * (vJ / 4))) & 3) << 4)) - 32;
      float vD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!vD) {
        continue;
      }
      for (int vIi = 0; vIi < 16; ++vIi) {
        int l = fNearestInt(pX[16 * vJ + vIi] / vD);
        l = MAX(-4, MIN(3, l));
        L[16 * vJ + vIi] = l + 4;
      }
    }

    memset(pY[vI].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int vM = 0;
    uint8_t vHm = 1;
    for (int vJ = 0; vJ < QK_K; ++vJ) {
      if (L[vJ] > 3) {
        pY[vI].hmask[vM] |= vHm;
        L[vJ] -= 4;
      }
      if (++vM == QK_K / 8) {
        vM = 0;
        vHm <<= 1;
      }
    }
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        pY[vI].qs[vJ / 4 + l] = L[vJ + l] | (L[vJ + l + 32] << 2) | (L[vJ + l + 64] << 4) | (L[vJ + l + 96] << 6);
      }
    }

    pX += QK_K;
  }
}

void dequantize_row_q3_K(const block_q3_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  const uint32_t cKmask1 = 0x03030303;
  const uint32_t cKmask2 = 0x0f0f0f0f;

  uint32_t aAux[4];
  const int8_t *vScales = (const int8_t *)aAux;

  for (int vI = 0; vI < cNb; vI++) {

    const float cDAll = GGML_FP16_TO_FP32(pX[vI].d);

    const uint8_t *GGML_RESTRICT vQ = pX[vI].qs;
    const uint8_t *GGML_RESTRICT vHm = pX[vI].hmask;
    uint8_t vM = 1;

    memcpy(aAux, pX[vI].scales, 12);
    uint32_t vTmp = aAux[2];
    aAux[2] = ((aAux[0] >> 4) & cKmask2) | (((vTmp >> 4) & cKmask1) << 4);
    aAux[3] = ((aAux[1] >> 4) & cKmask2) | (((vTmp >> 6) & cKmask1) << 4);
    aAux[0] = (aAux[0] & cKmask2) | (((vTmp >> 0) & cKmask1) << 4);
    aAux[1] = (aAux[1] & cKmask2) | (((vTmp >> 2) & cKmask1) << 4);

    int vIs = 0;
    float vDl;
    for (int vN = 0; vN < QK_K; vN += 128) {
      int vShift = 0;
      for (int vJ = 0; vJ < 4; ++vJ) {

        vDl = cDAll * (vScales[vIs++] - 32);
        for (int vL = 0; vL < 16; ++vL) {
          *pY++ = vDl * ((int8_t)((vQ[vL + 0] >> vShift) & 3) - ((vHm[vL + 0] & vM) ? 0 : 4));
        }

        vDl = cDAll * (vScales[vIs++] - 32);
        for (int vL = 0; vL < 16; ++vL) {
          *pY++ = vDl * ((int8_t)((vQ[vL + 16] >> vShift) & 3) - ((vHm[vL + 16] & vM) ? 0 : 4));
        }

        vShift += 2;
        vM <<= 1;
      }
      vQ += 32;
    }
  }
}

static void fQuantizeRowQ3KImpl(const float *GGML_RESTRICT pX, block_q3_K *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *GGML_RESTRICT pQuantWeights) {
  assert(pNPerRow % QK_K == 0);
  const int cNb = pNPerRow / QK_K;

  int8_t L[QK_K];
  float vScales[QK_K / 16];
  float aWeight[16];
  float aSw[QK_K / 16];
  int8_t vLs[QK_K / 16];

  for (int vI = 0; vI < cNb; vI++) {

    float vSumx2 = 0;
    for (int vJ = 0; vJ < QK_K; ++vJ)
      vSumx2 += pX[vJ] * pX[vJ];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vI + 16 * vJ;
        for (int l = 0; l < 16; ++l)
          aWeight[l] = vQw[l] * sqrtf(vSigma2 + pX[16 * vJ + l] * pX[16 * vJ + l]);
      } else {
        for (int l = 0; l < 16; ++l)
          aWeight[l] = pX[16 * vJ + l] * pX[16 * vJ + l];
      }
      float vSumw = 0;
      for (int l = 0; l < 16; ++l)
        vSumw += aWeight[l];
      aSw[vJ] = vSumw;

      vScales[vJ] = fMakeQxQuants(16, 4, pX + 16 * vJ, L + 16 * vJ, 1, aWeight);
    }

    memset(pY[vI].scales, 0, 12);

    float vDBlock = fMakeQxQuants(QK_K / 16, 32, vScales, vLs, 1, aSw);
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      int l = vLs[vJ];
      if (vJ < 8) {
        pY[vI].scales[vJ] = l & 0xF;
      } else {
        pY[vI].scales[vJ - 8] |= ((l & 0xF) << 4);
      }
      l >>= 4;
      pY[vI].scales[vJ % 4 + 8] |= (l << (2 * (vJ / 4)));
    }
    pY[vI].d = GGML_FP32_TO_FP16(vDBlock);

    int8_t vSc;
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      vSc = vJ < 8 ? pY[vI].scales[vJ] & 0xF : pY[vI].scales[vJ - 8] >> 4;
      vSc = (vSc | (((pY[vI].scales[8 + vJ % 4] >> (2 * (vJ / 4))) & 3) << 4)) - 32;
      float vD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!vD) {
        continue;
      }
      for (int vIi = 0; vIi < 16; ++vIi) {
        int l = fNearestInt(pX[16 * vJ + vIi] / vD);
        l = MAX(-4, MIN(3, l));
        L[16 * vJ + vIi] = l + 4;
      }
    }

    memset(pY[vI].hmask, 0, QK_K / 8);
    // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
    int vM = 0;
    uint8_t vHm = 1;
    for (int vJ = 0; vJ < QK_K; ++vJ) {
      if (L[vJ] > 3) {
        pY[vI].hmask[vM] |= vHm;
        L[vJ] -= 4;
      }
      if (++vM == QK_K / 8) {
        vM = 0;
        vHm <<= 1;
      }
    }
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        pY[vI].qs[vJ / 4 + l] = L[vJ + l] | (L[vJ + l + 32] << 2) | (L[vJ + l + 64] << 4) | (L[vJ + l + 96] << 6);
      }
    }

    pX += QK_K;
  }
}

size_t quantize_q3_K(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q3_K, pNPerRow);
  if (!pQuantWeights) {
    quantize_row_q3_K_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  } else {
    char *vQrow = (char *)pDst;
    for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
      fQuantizeRowQ3KImpl(pSrc, (block_q3_K *)vQrow, pNPerRow, pQuantWeights);
      pSrc += pNPerRow;
      vQrow += vRowSize;
    }
  }
  return pNrow * vRowSize;
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K_ref(const float *GGML_RESTRICT pX, block_q4_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  uint8_t L[QK_K];
  uint8_t aLaux[32];
  float aWeights[32];
  float aMins[QK_K / 32];
  float vScales[QK_K / 32];

  for (int vI = 0; vI < cNb; vI++) {
    float vMaxScale = 0; // as we are deducting the min, scales are always positive
    float vMaxMin = 0;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
      float vSumX2 = 0;
      for (int l = 0; l < 32; ++l)
        vSumX2 += pX[32 * vJ + l] * pX[32 * vJ + l];
      float vAvX = sqrtf(vSumX2 / 32);
      for (int l = 0; l < 32; ++l)
        aWeights[l] = vAvX + fabsf(pX[32 * vJ + l]);
      vScales[vJ] =
          fMakeQkx2Quants(32, 15, pX + 32 * vJ, aWeights, L + 32 * vJ, &aMins[vJ], aLaux, -1.f, 0.1f, 20, false);
      float vScale = vScales[vJ];
      if (vScale > vMaxScale) {
        vMaxScale = vScale;
      }
      float vMin = aMins[vJ];
      if (vMin > vMaxMin) {
        vMaxMin = vMin;
      }
    }

    float vInvScale = vMaxScale > 0 ? 63.f / vMaxScale : 0.f;
    float vInvMin = vMaxMin > 0 ? 63.f / vMaxMin : 0.f;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      uint8_t vLs = fNearestInt(vInvScale * vScales[vJ]);
      uint8_t vLm = fNearestInt(vInvMin * aMins[vJ]);
      vLs = MIN(63, vLs);
      vLm = MIN(63, vLm);
      if (vJ < 4) {
        pY[vI].scales[vJ] = vLs;
        pY[vI].scales[vJ + 4] = vLm;
      } else {
        pY[vI].scales[vJ + 4] = (vLs & 0xF) | ((vLm & 0xF) << 4);
        pY[vI].scales[vJ - 4] |= ((vLs >> 4) << 6);
        pY[vI].scales[vJ - 0] |= ((vLm >> 4) << 6);
      }
    }
    pY[vI].d = GGML_FP32_TO_FP16(vMaxScale / 63.f);
    pY[vI].dmin = GGML_FP32_TO_FP16(vMaxMin / 63.f);

    uint8_t vSc, vM;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      fGetScaleMinK4(vJ, pY[vI].scales, &vSc, &vM);
      const float cD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!cD)
        continue;
      const float cDm = GGML_FP16_TO_FP32(pY[vI].dmin) * vM;
      for (int vIi = 0; vIi < 32; ++vIi) {
        int l = fNearestInt((pX[32 * vJ + vIi] + cDm) / cD);
        l = MAX(0, MIN(15, l));
        L[32 * vJ + vIi] = l;
      }
    }

    uint8_t *vQ = pY[vI].qs;
    for (int vJ = 0; vJ < QK_K; vJ += 64) {
      for (int l = 0; l < 32; ++l)
        vQ[l] = L[vJ + l] | (L[vJ + l + 32] << 4);
      vQ += 32;
    }

    pX += QK_K;
  }
}

void dequantize_row_q4_K(const block_q4_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {
    const uint8_t *vQ = pX[vI].qs;

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cMin = GGML_FP16_TO_FP32(pX[vI].dmin);

    int vIs = 0;
    uint8_t vSc, vM;
    for (int vJ = 0; vJ < QK_K; vJ += 64) {
      fGetScaleMinK4(vIs + 0, pX[vI].scales, &vSc, &vM);
      const float cD1 = cD * vSc;
      const float cM1 = cMin * vM;
      fGetScaleMinK4(vIs + 1, pX[vI].scales, &vSc, &vM);
      const float cD2 = cD * vSc;
      const float cM2 = cMin * vM;
      for (int vL = 0; vL < 32; ++vL)
        *pY++ = cD1 * (vQ[vL] & 0xF) - cM1;
      for (int vL = 0; vL < 32; ++vL)
        *pY++ = cD2 * (vQ[vL] >> 4) - cM2;
      vQ += 32;
      vIs += 2;
    }
  }
}

static void fQuantizeRowQ4KImpl(const float *GGML_RESTRICT pX, block_q4_K *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  assert(pNPerRow % QK_K == 0);
  const int64_t cNb = pNPerRow / QK_K;

  uint8_t L[QK_K];
  uint8_t aLaux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float aWeights[32];
  float aSw[QK_K / 32];
  float aMins[QK_K / 32];
  float vScales[QK_K / 32];

  for (int vI = 0; vI < cNb; vI++) {

    float vSumX2 = 0;
    for (int l = 0; l < QK_K; ++l)
      vSumX2 += pX[l] * pX[l];
    float vSigma2 = 2 * vSumX2 / QK_K;
    float vAvX = sqrtf(vSigma2);

    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vI + 32 * vJ;
        for (int l = 0; l < 32; ++l)
          aWeights[l] = vQw[l] * sqrtf(vSigma2 + pX[32 * vJ + l] * pX[32 * vJ + l]);
      } else {
        for (int l = 0; l < 32; ++l)
          aWeights[l] = vAvX + fabsf(pX[32 * vJ + l]);
      }
      float vSumw = 0;
      for (int l = 0; l < 32; ++l)
        vSumw += aWeights[l];
      aSw[vJ] = vSumw;
      vScales[vJ] =
          fMakeQkx3Quants(32, 15, pX + 32 * vJ, aWeights, L + 32 * vJ, &aMins[vJ], aLaux, -0.9f, 0.05f, 36, false);
    }

    float vDBlock = fMakeQpQuants(QK_K / 32, 63, vScales, Ls, aSw);
    float vMBlock = fMakeQpQuants(QK_K / 32, 63, aMins, Lm, aSw);
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      uint8_t ls = Ls[vJ];
      uint8_t lm = Lm[vJ];
      if (vJ < 4) {
        pY[vI].scales[vJ] = ls;
        pY[vI].scales[vJ + 4] = lm;
      } else {
        pY[vI].scales[vJ + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        pY[vI].scales[vJ - 4] |= ((ls >> 4) << 6);
        pY[vI].scales[vJ - 0] |= ((lm >> 4) << 6);
      }
    }
    pY[vI].d = GGML_FP32_TO_FP16(vDBlock);
    pY[vI].dmin = GGML_FP32_TO_FP16(vMBlock);

    uint8_t vSc, vM;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      fGetScaleMinK4(vJ, pY[vI].scales, &vSc, &vM);
      const float cD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!cD)
        continue;
      const float cDm = GGML_FP16_TO_FP32(pY[vI].dmin) * vM;
      for (int vIi = 0; vIi < 32; ++vIi) {
        int l = fNearestInt((pX[32 * vJ + vIi] + cDm) / cD);
        l = MAX(0, MIN(15, l));
        L[32 * vJ + vIi] = l;
      }
    }
    uint8_t *vQ = pY[vI].qs;
    for (int vJ = 0; vJ < QK_K; vJ += 64) {
      for (int l = 0; l < 32; ++l)
        vQ[l] = L[vJ + l] | (L[vJ + l + 32] << 4);
      vQ += 32;
    }

    pX += QK_K;
  }
}

size_t quantize_q4_K(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q4_K, pNPerRow);
  if (!pQuantWeights) {
    quantize_row_q4_K_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  } else {
    char *vQrow = (char *)pDst;
    for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
      fQuantizeRowQ4KImpl(pSrc, (block_q4_K *)vQrow, pNPerRow, pQuantWeights);
      pSrc += pNPerRow;
      vQrow += vRowSize;
    }
  }
  return pNrow * vRowSize;
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K_ref(const float *GGML_RESTRICT pX, block_q5_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  uint8_t L[QK_K];
  float aMins[QK_K / 32];
  float vScales[QK_K / 32];
  float aWeights[32];
  uint8_t aLaux[32];

  for (int vI = 0; vI < cNb; vI++) {
    float vMaxScale = 0; // as we are deducting the min, scales are always positive
    float vMaxMin = 0;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      // scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
      float vSumX2 = 0;
      for (int l = 0; l < 32; ++l)
        vSumX2 += pX[32 * vJ + l] * pX[32 * vJ + l];
      float vAvX = sqrtf(vSumX2 / 32);
      for (int l = 0; l < 32; ++l)
        aWeights[l] = vAvX + fabsf(pX[32 * vJ + l]);
      vScales[vJ] =
          fMakeQkx2Quants(32, 31, pX + 32 * vJ, aWeights, L + 32 * vJ, &aMins[vJ], aLaux, -0.5f, 0.1f, 15, false);
      float vScale = vScales[vJ];
      if (vScale > vMaxScale) {
        vMaxScale = vScale;
      }
      float vMin = aMins[vJ];
      if (vMin > vMaxMin) {
        vMaxMin = vMin;
      }
    }

    float vInvScale = vMaxScale > 0 ? 63.f / vMaxScale : 0.f;
    float vInvMin = vMaxMin > 0 ? 63.f / vMaxMin : 0.f;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      uint8_t vLs = fNearestInt(vInvScale * vScales[vJ]);
      uint8_t vLm = fNearestInt(vInvMin * aMins[vJ]);
      vLs = MIN(63, vLs);
      vLm = MIN(63, vLm);
      if (vJ < 4) {
        pY[vI].scales[vJ] = vLs;
        pY[vI].scales[vJ + 4] = vLm;
      } else {
        pY[vI].scales[vJ + 4] = (vLs & 0xF) | ((vLm & 0xF) << 4);
        pY[vI].scales[vJ - 4] |= ((vLs >> 4) << 6);
        pY[vI].scales[vJ - 0] |= ((vLm >> 4) << 6);
      }
    }
    pY[vI].d = GGML_FP32_TO_FP16(vMaxScale / 63.f);
    pY[vI].dmin = GGML_FP32_TO_FP16(vMaxMin / 63.f);

    uint8_t vSc, vM;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      fGetScaleMinK4(vJ, pY[vI].scales, &vSc, &vM);
      const float cD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!cD)
        continue;
      const float cDm = GGML_FP16_TO_FP32(pY[vI].dmin) * vM;
      for (int vIi = 0; vIi < 32; ++vIi) {
        int l = fNearestInt((pX[32 * vJ + vIi] + cDm) / cD);
        l = MAX(0, MIN(31, l));
        L[32 * vJ + vIi] = l;
      }
    }

    uint8_t *GGML_RESTRICT vQh = pY[vI].qh;
    uint8_t *GGML_RESTRICT vQl = pY[vI].qs;
    memset(vQh, 0, QK_K / 8);

    uint8_t vM1 = 1, vM2 = 2;
    for (int vN = 0; vN < QK_K; vN += 64) {
      for (int vJ = 0; vJ < 32; ++vJ) {
        int vL1 = L[vN + vJ];
        if (vL1 > 15) {
          vL1 -= 16;
          vQh[vJ] |= vM1;
        }
        int vL2 = L[vN + vJ + 32];
        if (vL2 > 15) {
          vL2 -= 16;
          vQh[vJ] |= vM2;
        }
        vQl[vJ] = vL1 | (vL2 << 4);
      }
      vM1 <<= 2;
      vM2 <<= 2;
      vQl += 32;
    }

    pX += QK_K;
  }
}

void dequantize_row_q5_K(const block_q5_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {
    const uint8_t *vQl = pX[vI].qs;
    const uint8_t *vQh = pX[vI].qh;

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const float cMin = GGML_FP16_TO_FP32(pX[vI].dmin);

    int vIs = 0;
    uint8_t vSc, vM;
    uint8_t vU1 = 1, vU2 = 2;
    for (int vJ = 0; vJ < QK_K; vJ += 64) {
      fGetScaleMinK4(vIs + 0, pX[vI].scales, &vSc, &vM);
      const float cD1 = cD * vSc;
      const float cM1 = cMin * vM;
      fGetScaleMinK4(vIs + 1, pX[vI].scales, &vSc, &vM);
      const float cD2 = cD * vSc;
      const float cM2 = cMin * vM;
      for (int vL = 0; vL < 32; ++vL)
        *pY++ = cD1 * ((vQl[vL] & 0xF) + (vQh[vL] & vU1 ? 16 : 0)) - cM1;
      for (int vL = 0; vL < 32; ++vL)
        *pY++ = cD2 * ((vQl[vL] >> 4) + (vQh[vL] & vU2 ? 16 : 0)) - cM2;
      vQl += 32;
      vIs += 2;
      vU1 <<= 2;
      vU2 <<= 2;
    }
  }
}

static void fQuantizeRowQ5KImpl(const float *GGML_RESTRICT pX, block_q5_K *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  assert(pNPerRow % QK_K == 0);
  const int64_t cNb = pNPerRow / QK_K;

  uint8_t L[QK_K];
  uint8_t aLaux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float aMins[QK_K / 32];
  float vScales[QK_K / 32];
  float aSw[QK_K / 32];
  float aWeights[32];

  for (int vI = 0; vI < cNb; vI++) {

    float vSumX2 = 0;
    for (int l = 0; l < QK_K; ++l)
      vSumX2 += pX[l] * pX[l];
    float vSigma2 = 2 * vSumX2 / QK_K;
    float vAvX = sqrtf(vSigma2);

    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vI + 32 * vJ;
        for (int l = 0; l < 32; ++l)
          aWeights[l] = vQw[l] * sqrtf(vSigma2 + pX[32 * vJ + l] * pX[32 * vJ + l]);
      } else {
        for (int l = 0; l < 32; ++l)
          aWeights[l] = vAvX + fabsf(pX[32 * vJ + l]);
      }
      float vSumw = 0;
      for (int l = 0; l < 32; ++l)
        vSumw += aWeights[l];
      aSw[vJ] = vSumw;

      vScales[vJ] =
          fMakeQkx3Quants(32, 31, pX + 32 * vJ, aWeights, L + 32 * vJ, &aMins[vJ], aLaux, -0.9f, 0.05f, 36, false);
    }

    float vDBlock = fMakeQpQuants(QK_K / 32, 63, vScales, Ls, aSw);
    float vMBlock = fMakeQpQuants(QK_K / 32, 63, aMins, Lm, aSw);

    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      uint8_t ls = Ls[vJ];
      uint8_t lm = Lm[vJ];
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (vJ < 4) {
        pY[vI].scales[vJ] = ls;
        pY[vI].scales[vJ + 4] = lm;
      } else {
        pY[vI].scales[vJ + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        pY[vI].scales[vJ - 4] |= ((ls >> 4) << 6);
        pY[vI].scales[vJ - 0] |= ((lm >> 4) << 6);
      }
    }
    pY[vI].d = GGML_FP32_TO_FP16(vDBlock);
    pY[vI].dmin = GGML_FP32_TO_FP16(vMBlock);

    uint8_t vSc, vM;
    for (int vJ = 0; vJ < QK_K / 32; ++vJ) {
      fGetScaleMinK4(vJ, pY[vI].scales, &vSc, &vM);
      const float cD = GGML_FP16_TO_FP32(pY[vI].d) * vSc;
      if (!cD)
        continue;
      const float cDm = GGML_FP16_TO_FP32(pY[vI].dmin) * vM;
      for (int vIi = 0; vIi < 32; ++vIi) {
        int l = fNearestInt((pX[32 * vJ + vIi] + cDm) / cD);
        l = MAX(0, MIN(31, l));
        L[32 * vJ + vIi] = l;
      }
    }

    uint8_t *GGML_RESTRICT vQh = pY[vI].qh;
    uint8_t *GGML_RESTRICT vQl = pY[vI].qs;
    memset(vQh, 0, QK_K / 8);

    uint8_t vM1 = 1, vM2 = 2;
    for (int vN = 0; vN < QK_K; vN += 64) {
      for (int vJ = 0; vJ < 32; ++vJ) {
        int vL1 = L[vN + vJ];
        if (vL1 > 15) {
          vL1 -= 16;
          vQh[vJ] |= vM1;
        }
        int vL2 = L[vN + vJ + 32];
        if (vL2 > 15) {
          vL2 -= 16;
          vQh[vJ] |= vM2;
        }
        vQl[vJ] = vL1 | (vL2 << 4);
      }
      vM1 <<= 2;
      vM2 <<= 2;
      vQl += 32;
    }

    pX += QK_K;
  }
}

size_t quantize_q5_K(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q5_K, pNPerRow);
  if (!pQuantWeights) {
    quantize_row_q5_K_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  } else {
    char *vQrow = (char *)pDst;
    for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
      fQuantizeRowQ5KImpl(pSrc, (block_q5_K *)vQrow, pNPerRow, pQuantWeights);
      pSrc += pNPerRow;
      vQrow += vRowSize;
    }
  }
  return pNrow * vRowSize;
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_ref(const float *GGML_RESTRICT pX, block_q6_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  int8_t L[QK_K];
  float vScales[QK_K / 16];

  for (int vI = 0; vI < cNb; vI++) {

    float vMaxScale = 0;
    float vMaxAbsScale = 0;

    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {

      const float cScale = fMakeQxQuants(16, 32, pX + 16 * vIb, L + 16 * vIb, 1, NULL);
      vScales[vIb] = cScale;

      const float cAbsScale = fabsf(cScale);
      if (cAbsScale > vMaxAbsScale) {
        vMaxAbsScale = cAbsScale;
        vMaxScale = cScale;
      }
    }

    if (vMaxAbsScale < GROUP_MAX_EPS) {
      memset(&pY[vI], 0, sizeof(block_q6_K));
      pY[vI].d = GGML_FP32_TO_FP16(0.f);
      pX += QK_K;
      continue;
    }

    float vIscale = -128.f / vMaxScale;
    pY[vI].d = GGML_FP32_TO_FP16(1 / vIscale);
    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      pY[vI].scales[vIb] = MIN(127, fNearestInt(vIscale * vScales[vIb]));
    }

    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      float vD = GGML_FP16_TO_FP32(pY[vI].d) * pY[vI].scales[vJ];
      if (!vD) {
        continue;
      }
      for (int vIi = 0; vIi < 16; ++vIi) {
        int l = fNearestInt(pX[16 * vJ + vIi] / vD);
        l = MAX(-32, MIN(31, l));
        L[16 * vJ + vIi] = l + 32;
      }
    }

    uint8_t *GGML_RESTRICT vQl = pY[vI].ql;
    uint8_t *GGML_RESTRICT vQh = pY[vI].qh;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t cQ1 = L[vJ + l + 0] & 0xF;
        const uint8_t cQ2 = L[vJ + l + 32] & 0xF;
        const uint8_t cQ3 = L[vJ + l + 64] & 0xF;
        const uint8_t cQ4 = L[vJ + l + 96] & 0xF;
        vQl[l + 0] = cQ1 | (cQ3 << 4);
        vQl[l + 32] = cQ2 | (cQ4 << 4);
        vQh[l] = (L[vJ + l] >> 4) | ((L[vJ + l + 32] >> 4) << 2) | ((L[vJ + l + 64] >> 4) << 4) |
                 ((L[vJ + l + 96] >> 4) << 6);
      }
      vQl += 64;
      vQh += 32;
    }

    pX += QK_K;
  }
}

void dequantize_row_q6_K(const block_q6_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {
    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    const uint8_t *GGML_RESTRICT vQl = pX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = pX[vI].qh;
    const int8_t *GGML_RESTRICT vSc = pX[vI].scales;

    for (int vN = 0; vN < QK_K; vN += 128) {
      for (int vL = 0; vL < 32; ++vL) {
        int vIs = vL / 16;
        const int8_t cQ1 = (int8_t)((vQl[vL + 0] & 0xF) | (((vQh[vL] >> 0) & 3) << 4)) - 32;
        const int8_t cQ2 = (int8_t)((vQl[vL + 32] & 0xF) | (((vQh[vL] >> 2) & 3) << 4)) - 32;
        const int8_t cQ3 = (int8_t)((vQl[vL + 0] >> 4) | (((vQh[vL] >> 4) & 3) << 4)) - 32;
        const int8_t cQ4 = (int8_t)((vQl[vL + 32] >> 4) | (((vQh[vL] >> 6) & 3) << 4)) - 32;
        pY[vL + 0] = cD * vSc[vIs + 0] * cQ1;
        pY[vL + 32] = cD * vSc[vIs + 2] * cQ2;
        pY[vL + 64] = cD * vSc[vIs + 4] * cQ3;
        pY[vL + 96] = cD * vSc[vIs + 6] * cQ4;
      }
      pY += 128;
      vQl += 64;
      vQh += 32;
      vSc += 8;
    }
  }
}

static void fQuantizeRowQ6KImpl(const float *GGML_RESTRICT pX, block_q6_K *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  assert(pNPerRow % QK_K == 0);
  const int64_t cNb = pNPerRow / QK_K;

  int8_t L[QK_K];
  float vScales[QK_K / 16];
  // float   weights[16];

  for (int vI = 0; vI < cNb; vI++) {

    // float sum_x2 = 0;
    // for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
    // float sigma2 = sum_x2/QK_K;

    float vMaxScale = 0;
    float vMaxAbsScale = 0;

    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {

      float vScale;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vI + 16 * vIb;
        // for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
        // scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
        vScale = fMakeQxQuants(16, 32, pX + 16 * vIb, L + 16 * vIb, 1, vQw);
      } else {
        vScale = fMakeQxQuants(16, 32, pX + 16 * vIb, L + 16 * vIb, 1, NULL);
      }
      vScales[vIb] = vScale;

      const float cAbsScale = fabsf(vScale);
      if (cAbsScale > vMaxAbsScale) {
        vMaxAbsScale = cAbsScale;
        vMaxScale = vScale;
      }
    }

    if (vMaxAbsScale < GROUP_MAX_EPS) {
      memset(&pY[vI], 0, sizeof(block_q6_K));
      pY[vI].d = GGML_FP32_TO_FP16(0.f);
      pX += QK_K;
      continue;
    }

    float vIscale = -128.f / vMaxScale;
    pY[vI].d = GGML_FP32_TO_FP16(1 / vIscale);
    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      pY[vI].scales[vIb] = MIN(127, fNearestInt(vIscale * vScales[vIb]));
    }

    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      float vD = GGML_FP16_TO_FP32(pY[vI].d) * pY[vI].scales[vJ];
      if (!vD) {
        continue;
      }
      for (int vIi = 0; vIi < 16; ++vIi) {
        int l = fNearestInt(pX[16 * vJ + vIi] / vD);
        l = MAX(-32, MIN(31, l));
        L[16 * vJ + vIi] = l + 32;
      }
    }

    uint8_t *GGML_RESTRICT vQl = pY[vI].ql;
    uint8_t *GGML_RESTRICT vQh = pY[vI].qh;
    for (int vJ = 0; vJ < QK_K; vJ += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t cQ1 = L[vJ + l + 0] & 0xF;
        const uint8_t cQ2 = L[vJ + l + 32] & 0xF;
        const uint8_t cQ3 = L[vJ + l + 64] & 0xF;
        const uint8_t cQ4 = L[vJ + l + 96] & 0xF;
        vQl[l + 0] = cQ1 | (cQ3 << 4);
        vQl[l + 32] = cQ2 | (cQ4 << 4);
        vQh[l] = (L[vJ + l] >> 4) | ((L[vJ + l + 32] >> 4) << 2) | ((L[vJ + l + 64] >> 4) << 4) |
                 ((L[vJ + l + 96] >> 4) << 6);
      }
      vQl += 64;
      vQh += 32;
    }

    pX += QK_K;
  }
}

size_t quantize_q6_K(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q6_K, pNPerRow);
  if (!pQuantWeights) {
    quantize_row_q6_K_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  } else {
    char *vQrow = (char *)pDst;
    for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
      fQuantizeRowQ6KImpl(pSrc, (block_q6_K *)vQrow, pNPerRow, pQuantWeights);
      pSrc += pNPerRow;
      vQrow += vRowSize;
    }
  }
  return pNrow * vRowSize;
}

static void fQuantizeRowQ40Impl(const float *GGML_RESTRICT pX, block_q4_0 *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  if (!pQuantWeights) {
    quantize_row_q4_0_ref(pX, pY, pNPerRow);
    return;
  }

  float aWeight[QK4_0];
  int8_t vL[QK4_0];

  float vSumX2 = 0;
  for (int vJ = 0; vJ < pNPerRow; ++vJ)
    vSumX2 += pX[vJ] * pX[vJ];
  float vSigma2 = vSumX2 / pNPerRow;

  const int64_t cNb = pNPerRow / QK4_0;
  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float *vXb = pX + QK4_0 * vIb;
    const float *vQw = pQuantWeights + QK4_0 * vIb;
    for (int vJ = 0; vJ < QK4_0; ++vJ)
      aWeight[vJ] = vQw[vJ] * sqrtf(vSigma2 + vXb[vJ] * vXb[vJ]);
    float vD = fMakeQxQuants(QK4_0, 8, vXb, vL, 1, aWeight);
    pY[vIb].d = GGML_FP32_TO_FP16(vD);
    for (int vJ = 0; vJ < 16; ++vJ) {
      pY[vIb].qs[vJ] = vL[vJ] | (vL[vJ + 16] << 4);
    }
  }
}

size_t quantize_q1_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q1_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q1_0, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q1_0, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    quantize_row_q1_0_ref(pSrc, (block_q1_0 *)vQrow, pNPerRow);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

size_t quantize_q2_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q2_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q2_0, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q2_0, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    quantize_row_q2_0_ref(pSrc, (block_q2_0 *)vQrow, pNPerRow);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

size_t quantize_q4_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q4_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q4_0, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q4_0, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowQ40Impl(pSrc, (block_q4_0 *)vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

static void fQuantizeRowQ41Impl(const float *GGML_RESTRICT pX, block_q4_1 *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  static_assert(QK4_1 == 32, "QK4_1 must be 32");

  if (!pQuantWeights) {
    quantize_row_q4_1_ref(pX, pY, pNPerRow);
    return;
  }

  float aWeight[QK4_1];
  uint8_t vL[QK4_1], aLaux[QK4_1];

  float vSumX2 = 0;
  for (int vJ = 0; vJ < pNPerRow; ++vJ)
    vSumX2 += pX[vJ] * pX[vJ];
  float vSigma2 = vSumX2 / pNPerRow;

  const int64_t cNb = pNPerRow / QK4_1;
  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float *vXb = pX + QK4_1 * vIb;
    const float *vQw = pQuantWeights + QK4_1 * vIb;
    for (int vJ = 0; vJ < QK4_1; ++vJ)
      aWeight[vJ] = vQw[vJ] * sqrtf(vSigma2 + vXb[vJ] * vXb[vJ]);
    float vMin;
    float vD = fMakeQkx3Quants(QK4_1, 15, vXb, aWeight, vL, &vMin, aLaux, -0.9f, 0.05f, 36, false);
    pY[vIb].d = GGML_FP32_TO_FP16(vD);
    pY[vIb].m = GGML_FP32_TO_FP16(-vMin);
    for (int vJ = 0; vJ < 16; ++vJ) {
      pY[vIb].qs[vJ] = vL[vJ] | (vL[vJ + 16] << 4);
    }
  }
}

size_t quantize_q4_1(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q4_1_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q4_1, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q4_1, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowQ41Impl(pSrc, (block_q4_1 *)vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

static void fQuantizeRowQ50Impl(const float *GGML_RESTRICT pX, block_q5_0 *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  if (!pQuantWeights) {
    quantize_row_q5_0_ref(pX, pY, pNPerRow);
    return;
  }

  float aWeight[QK5_0];
  int8_t vL[QK5_0];

  float vSumX2 = 0;
  for (int vJ = 0; vJ < pNPerRow; ++vJ)
    vSumX2 += pX[vJ] * pX[vJ];
  float vSigma2 = vSumX2 / pNPerRow;

  const int64_t cNb = pNPerRow / QK5_0;
  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float *vXb = pX + QK5_0 * vIb;
    const float *vQw = pQuantWeights + QK5_0 * vIb;
    for (int vJ = 0; vJ < QK5_0; ++vJ)
      aWeight[vJ] = vQw[vJ] * sqrtf(vSigma2 + vXb[vJ] * vXb[vJ]);
    float vD = fMakeQxQuants(QK5_0, 16, vXb, vL, 1, aWeight);
    pY[vIb].d = GGML_FP32_TO_FP16(vD);

    uint32_t vQh = 0;

    for (int vJ = 0; vJ < 16; ++vJ) {
      const uint8_t cXi0 = vL[vJ];
      const uint8_t cXi1 = vL[vJ + 16];
      pY[vIb].qs[vJ] = (cXi0 & 0x0F) | ((cXi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
      vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + QK5_0 / 2);
    }

    memcpy(&pY[vIb].qh, &vQh, sizeof(vQh));
  }
}

size_t quantize_q5_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q5_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q5_0, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q5_0, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowQ50Impl(pSrc, (block_q5_0 *)vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

static void fQuantizeRowQ51Impl(const float *GGML_RESTRICT pX, block_q5_1 *GGML_RESTRICT pY, int64_t pNPerRow,
                                const float *pQuantWeights) {
  static_assert(QK5_1 == 32, "QK5_1 must be 32");

  if (!pQuantWeights) {
    quantize_row_q5_1_ref(pX, pY, pNPerRow);
    return;
  }

  float aWeight[QK5_1];
  uint8_t vL[QK5_1], aLaux[QK5_1];

  float vSumX2 = 0;
  for (int vJ = 0; vJ < pNPerRow; ++vJ)
    vSumX2 += pX[vJ] * pX[vJ];
  float vSigma2 = vSumX2 / pNPerRow;

  const int64_t cNb = pNPerRow / QK5_1;
  for (int vIb = 0; vIb < cNb; ++vIb) {
    const float *vXb = pX + QK5_1 * vIb;
    const float *vQw = pQuantWeights + QK5_1 * vIb;
    for (int vJ = 0; vJ < QK5_1; ++vJ)
      aWeight[vJ] = vQw[vJ] * sqrtf(vSigma2 + vXb[vJ] * vXb[vJ]);
    float vMin;
    float vD = fMakeQkx3Quants(QK5_1, 31, vXb, aWeight, vL, &vMin, aLaux, -0.9f, 0.05f, 36, false);
    pY[vIb].d = GGML_FP32_TO_FP16(vD);
    pY[vIb].m = GGML_FP32_TO_FP16(-vMin);

    uint32_t vQh = 0;
    for (int vJ = 0; vJ < 16; ++vJ) {
      const uint8_t cXi0 = vL[vJ];
      const uint8_t cXi1 = vL[vJ + 16];
      pY[vIb].qs[vJ] = (cXi0 & 0x0F) | ((cXi1 & 0x0F) << 4);
      // get the 5-th bit and store it in qh at the right position
      vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
      vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + QK5_0 / 2);
    }
    memcpy(&pY[vIb].qh, &vQh, sizeof(vQh));
  }
}

size_t quantize_q5_1(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  if (!pQuantWeights) {
    quantize_row_q5_1_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
    return pNrow * ggml_row_size(GGML_TYPE_Q5_1, pNPerRow);
  }
  size_t vRowSize = ggml_row_size(GGML_TYPE_Q5_1, pNPerRow);
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowQ51Impl(pSrc, (block_q5_1 *)vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vRowSize;
  }
  return pNrow * vRowSize;
}

size_t quantize_q8_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                     const float *pQuantWeights) {
  (void)pQuantWeights; // not used
  const size_t cRowSize = ggml_row_size(GGML_TYPE_Q8_0, pNPerRow);
  quantize_row_q8_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  return pNrow * cRowSize;
}

size_t quantize_mxfp4(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_UNUSED(pQuantWeights);
  quantize_row_mxfp4_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  return pNrow * ggml_row_size(GGML_TYPE_MXFP4, pNPerRow);
}

size_t quantize_nvfp4(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_UNUSED(pQuantWeights);
  quantize_row_nvfp4_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  return pNrow * ggml_row_size(GGML_TYPE_NVFP4, pNPerRow);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0_ref(const float *GGML_RESTRICT pX, block_tq1_0 *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int64_t vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max

    for (int vJ = 0; vJ < QK_K; vJ++) {
      const float cV = pX[vJ];
      vAmax = MAX(vAmax, fabsf(cV));
    }

    const float cD = vAmax;
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    // 5 elements per byte, along 32 bytes
    for (size_t vJ = 0; vJ < sizeof(pY->qs) - sizeof(pY->qs) % 32; vJ += 32) {
      for (size_t vM = 0; vM < 32; ++vM) {
        uint8_t vQ = 0;
        for (size_t vN = 0; vN < 5; ++vN) {
          int vXi = lroundf(pX[vM + vN * 32] * cId) + 1; // -1, 0, 1 -> 0, 1, 2
          vQ *= 3;
          vQ += vXi;
        }
        // ceiling division (243 == pow(3, 5))
        vQ = ((uint16_t)vQ * 256 + (243 - 1)) / 243;
        pY[vI].qs[vJ + vM] = vQ;
      }
      pX += 5 * 32;
    }
    // along 16 bytes
    for (size_t vJ = sizeof(pY->qs) - sizeof(pY->qs) % 32; vJ < sizeof(pY->qs); vJ += 16) {
      for (size_t vM = 0; vM < 16; ++vM) {
        uint8_t vQ = 0;
        for (size_t vN = 0; vN < 5; ++vN) {
          int vXi = lroundf(pX[vM + vN * 16] * cId) + 1; // -1, 0, 1 -> 0, 1, 2
          vQ *= 3;
          vQ += vXi;
        }
        // ceiling division (243 == pow(3, 5))
        vQ = ((uint16_t)vQ * 256 + (243 - 1)) / 243;
        pY[vI].qs[vJ + vM] = vQ;
      }
      pX += 5 * 16;
    }
    // 4 elements per byte
    for (size_t vJ = 0; vJ < sizeof(pY->qh); ++vJ) {
      uint8_t vQ = 0;
      for (size_t vM = 0; vM < 4; ++vM) {
        // -1, 0, 1 -> 0, 1, 2
        int vXi = lroundf(pX[vJ + vM * sizeof(pY->qh)] * cId) + 1;
        vQ *= 3;
        vQ += vXi;
      }
      // shift the first value to the most significant trit
      vQ *= 3;
      // ceiling division (243 == pow(3, 5))
      vQ = ((uint16_t)vQ * 256 + (243 - 1)) / 243;
      pY[vI].qh[vJ] = vQ;
    }
    pX += 4 * sizeof(pY->qh);
  }
}

void quantize_row_tq2_0_ref(const float *GGML_RESTRICT pX, block_tq2_0 *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int64_t vI = 0; vI < cNb; vI++) {
    float vAmax = 0.0f; // absolute max

    for (int vJ = 0; vJ < QK_K; vJ++) {
      const float cV = pX[vJ];
      vAmax = MAX(vAmax, fabsf(cV));
    }

    const float cD = vAmax;
    const float cId = cD ? 1.0f / cD : 0.0f;

    pY[vI].d = GGML_FP32_TO_FP16(cD);

    for (size_t vJ = 0; vJ < sizeof(pY->qs); vJ += 32) {
      for (size_t vM = 0; vM < 32; ++vM) {
        uint8_t vQ = 0;
        for (size_t vN = 0; vN < 4; ++vN) {
          // -1, 0, 1 -> 0, 1, 2
          int vXi = lroundf(pX[vM + vN * 32] * cId) + 1;
          vQ += (vXi & 3) << (2 * vN);
        }
        pY[vI].qs[vJ + vM] = vQ;
      }
      pX += 4 * 32;
    }
  }
}

size_t quantize_tq1_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  (void)pQuantWeights; // not used
  const size_t cRowSize = ggml_row_size(GGML_TYPE_TQ1_0, pNPerRow);
  quantize_row_tq1_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  return pNrow * cRowSize;
}

size_t quantize_tq2_0(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  (void)pQuantWeights; // not used
  const size_t cRowSize = ggml_row_size(GGML_TYPE_TQ2_0, pNPerRow);
  quantize_row_tq2_0_ref(pSrc, pDst, (int64_t)pNrow * pNPerRow);
  return pNrow * cRowSize;
}

void dequantize_row_tq1_0(const block_tq1_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  const uint8_t aPow3[6] = {1, 3, 9, 27, 81, 243};

  for (int64_t vI = 0; vI < cNb; ++vI) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (size_t vJ = 0; vJ < sizeof(pX->qs) - sizeof(pX->qs) % 32; vJ += 32) {
      for (size_t vN = 0; vN < 5; ++vN) {
        for (size_t vM = 0; vM < 32; ++vM) {
          uint8_t vQ = pX[vI].qs[vJ + vM] * aPow3[vN];
          int16_t vXi = ((uint16_t)vQ * 3) >> 8;
          *pY++ = (float)(vXi - 1) * cD;
        }
      }
    }
    for (size_t vJ = sizeof(pX->qs) - sizeof(pX->qs) % 32; vJ < sizeof(pX->qs); vJ += 16) {
      for (size_t vN = 0; vN < 5; ++vN) {
        for (size_t vM = 0; vM < 16; ++vM) {
          uint8_t vQ = pX[vI].qs[vJ + vM] * aPow3[vN];
          int16_t vXi = ((uint16_t)vQ * 3) >> 8;
          *pY++ = (float)(vXi - 1) * cD;
        }
      }
    }

    for (size_t vN = 0; vN < 4; ++vN) {
      for (size_t vJ = 0; vJ < sizeof(pX->qh); ++vJ) {
        uint8_t vQ = pX[vI].qh[vJ] * aPow3[vN];
        int16_t vXi = ((uint16_t)vQ * 3) >> 8;
        *pY++ = (float)(vXi - 1) * cD;
      }
    }
  }
}

void dequantize_row_tq2_0(const block_tq2_0 *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int64_t vI = 0; vI < cNb; ++vI) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (size_t vJ = 0; vJ < sizeof(pX->qs); vJ += 32) {
      for (size_t vL = 0; vL < 4; ++vL) {
        for (size_t vM = 0; vM < 32; ++vM) {
          int8_t vQ = (pX[vI].qs[vJ + vM] >> (vL * 2)) & 3;
          *pY++ = (float)(vQ - 1) * cD;
        }
      }
    }
  }
}

// ====================== "True" 2-bit (de)-quantization

void dequantize_row_iq2_xxs(const block_iq2_xxs *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  uint32_t aAux32[2];
  const uint8_t *vAux8 = (const uint8_t *)aAux32;

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      memcpy(aAux32, pX[vI].qs + 4 * vIb32, 2 * sizeof(uint32_t));
      const float cDb = cD * (0.5f + (aAux32[1] >> 28)) * 0.25f;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2xxs_grid + vAux8[vL]);
        const uint8_t cSigns = ksigns_iq2xs[(aAux32[1] >> 7 * vL) & 127];
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = cDb * vGrid[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1.f : 1.f);
        }
        pY += 8;
      }
    }
  }
}

// ====================== 2.3125 bpw (de)-quantization

void dequantize_row_iq2_xs(const block_iq2_xs *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  float aDb[2];

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      aDb[0] = cD * (0.5f + (pX[vI].scales[vIb32] & 0xf)) * 0.25f;
      aDb[1] = cD * (0.5f + (pX[vI].scales[vIb32] >> 4)) * 0.25f;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid = (const uint8_t *)(iq2xs_grid + (pX[vI].qs[4 * vIb32 + vL] & 511));
        const uint8_t cSigns = ksigns_iq2xs[pX[vI].qs[4 * vIb32 + vL] >> 9];
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = aDb[vL / 2] * vGrid[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1.f : 1.f);
        }
        pY += 8;
      }
    }
  }
}

// ====================== 2.5625 bpw (de)-quantization

void dequantize_row_iq2_s(const block_iq2_s *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  float aDb[2];

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const uint8_t *vQs = pX[vI].qs;
    const uint8_t *vQh = pX[vI].qh;
    const uint8_t *vSigns = vQs + QK_K / 8;

    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      aDb[0] = cD * (0.5f + (pX[vI].scales[vIb32] & 0xf)) * 0.25f;
      aDb[1] = cD * (0.5f + (pX[vI].scales[vIb32] >> 4)) * 0.25f;
      for (int vL = 0; vL < 4; ++vL) {
        const float cDl = aDb[vL / 2];
        const uint8_t *vGrid = (const uint8_t *)(iq2s_grid + (vQs[vL] | (vQh[vIb32] << (8 - 2 * vL) & 0x300)));
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = cDl * vGrid[vJ] * (vSigns[vL] & kmask_iq2xs[vJ] ? -1.f : 1.f);
        }
        pY += 8;
      }
      vQs += 4;
      vSigns += 4;
    }
  }
}

// ====================== 3.0625 bpw (de)-quantization

void dequantize_row_iq3_xxs(const block_iq3_xxs *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  uint32_t vAux32;

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const uint8_t *vQs = pX[vI].qs;
    const uint8_t *vScalesAndSigns = vQs + QK_K / 4;

    for (int vIb32 = 0; vIb32 < QK_K / 32; ++vIb32) {
      memcpy(&vAux32, vScalesAndSigns + 4 * vIb32, sizeof(uint32_t));
      const float cDb = cD * (0.5f + (vAux32 >> 28)) * 0.5f;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t cSigns = ksigns_iq2xs[(vAux32 >> 7 * vL) & 127];
        const uint8_t *vGrid1 = (const uint8_t *)(iq3xxs_grid + vQs[2 * vL + 0]);
        const uint8_t *vGrid2 = (const uint8_t *)(iq3xxs_grid + vQs[2 * vL + 1]);
        for (int vJ = 0; vJ < 4; ++vJ) {
          pY[vJ + 0] = cDb * vGrid1[vJ] * (cSigns & kmask_iq2xs[vJ + 0] ? -1.f : 1.f);
          pY[vJ + 4] = cDb * vGrid2[vJ] * (cSigns & kmask_iq2xs[vJ + 4] ? -1.f : 1.f);
        }
        pY += 8;
      }
      vQs += 8;
    }
  }
}

// ====================== 3.3125 bpw (de)-quantization

void dequantize_row_iq3_s(const block_iq3_s *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const uint8_t *vQs = pX[vI].qs;
    const uint8_t *vQh = pX[vI].qh;
    const uint8_t *vSigns = pX[vI].signs;

    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const float cDb1 = cD * (1 + 2 * (pX[vI].scales[vIb32 / 2] & 0xf));
      const float cDb2 = cD * (1 + 2 * (pX[vI].scales[vIb32 / 2] >> 4));
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid1 = (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 0] | ((vQh[0] << (8 - 2 * vL)) & 256)));
        const uint8_t *vGrid2 = (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 1] | ((vQh[0] << (7 - 2 * vL)) & 256)));
        for (int vJ = 0; vJ < 4; ++vJ) {
          pY[vJ + 0] = cDb1 * vGrid1[vJ] * (vSigns[vL] & kmask_iq2xs[vJ + 0] ? -1.f : 1.f);
          pY[vJ + 4] = cDb1 * vGrid2[vJ] * (vSigns[vL] & kmask_iq2xs[vJ + 4] ? -1.f : 1.f);
        }
        pY += 8;
      }
      vQs += 8;
      vSigns += 4;
      for (int vL = 0; vL < 4; ++vL) {
        const uint8_t *vGrid1 = (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 0] | ((vQh[1] << (8 - 2 * vL)) & 256)));
        const uint8_t *vGrid2 = (const uint8_t *)(iq3s_grid + (vQs[2 * vL + 1] | ((vQh[1] << (7 - 2 * vL)) & 256)));
        for (int vJ = 0; vJ < 4; ++vJ) {
          pY[vJ + 0] = cDb2 * vGrid1[vJ] * (vSigns[vL] & kmask_iq2xs[vJ + 0] ? -1.f : 1.f);
          pY[vJ + 4] = cDb2 * vGrid2[vJ] * (vSigns[vL] & kmask_iq2xs[vJ + 4] ? -1.f : 1.f);
        }
        pY += 8;
      }
      vQh += 2;
      vQs += 8;
      vSigns += 4;
    }
  }
}

// ====================== 1.5625 bpw (de)-quantization

void dequantize_row_iq1_s(const block_iq1_s *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    const uint8_t *vQs = pX[vI].qs;
    const uint16_t *vQh = pX[vI].qh;

    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const float cDl = cD * (2 * ((vQh[vIb] >> 12) & 7) + 1);
      const float cDelta = vQh[vIb] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
      for (int vL = 0; vL < 4; ++vL) {
        const int8_t *vGrid = (const int8_t *)(iq1s_grid + (vQs[vL] | (((vQh[vIb] >> 3 * vL) & 7) << 8)));
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = cDl * (vGrid[vJ] + cDelta);
        }
        pY += 8;
      }
      vQs += 4;
    }
  }
}

void dequantize_row_iq1_m(const block_iq1_m *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  float aDelta[4];
  uint16_t aIdx[4];

  iq1m_scale_t vScale;

  for (int vI = 0; vI < cNb; vI++) {

    const uint16_t *vSc = (const uint16_t *)pX[vI].scales;
    vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);
    const float cD = GGML_FP16_TO_FP32(vScale.f16);

    const uint8_t *vQs = pX[vI].qs;
    const uint8_t *vQh = pX[vI].qh;

    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const float cDl1 = cD * (2 * ((vSc[vIb / 2] >> (6 * (vIb % 2) + 0)) & 0x7) + 1);
      const float cDl2 = cD * (2 * ((vSc[vIb / 2] >> (6 * (vIb % 2) + 3)) & 0x7) + 1);

      aIdx[0] = vQs[0] | ((vQh[0] << 8) & 0x700);
      aIdx[1] = vQs[1] | ((vQh[0] << 4) & 0x700);
      aIdx[2] = vQs[2] | ((vQh[1] << 8) & 0x700);
      aIdx[3] = vQs[3] | ((vQh[1] << 4) & 0x700);
      aDelta[0] = vQh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
      aDelta[1] = vQh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
      aDelta[2] = vQh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
      aDelta[3] = vQh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
      for (int vL = 0; vL < 2; ++vL) {
        const int8_t *vGrid = (const int8_t *)(iq1s_grid + aIdx[vL]);
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = cDl1 * (vGrid[vJ] + aDelta[vL]);
        }
        pY += 8;
      }
      for (int vL = 2; vL < 4; ++vL) {
        const int8_t *vGrid = (const int8_t *)(iq1s_grid + aIdx[vL]);
        for (int vJ = 0; vJ < 8; ++vJ) {
          pY[vJ] = cDl2 * (vGrid[vJ] + aDelta[vL]);
        }
        pY += 8;
      }
      vQs += 4;
      vQh += 2;
    }
  }
}

void dequantize_row_iq4_nl(const block_iq4_nl *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK4_NL == 0);
  const int64_t cNb = pK / QK4_NL;

  for (int vI = 0; vI < cNb; vI++) {

    const uint8_t *vQs = pX[vI].qs;

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);
    for (int vJ = 0; vJ < QK4_NL / 2; ++vJ) {
      pY[vJ + 0] = cD * kvalues_iq4nl[vQs[vJ] & 0xf];
      pY[vJ + QK4_NL / 2] = cD * kvalues_iq4nl[vQs[vJ] >> 4];
    }
    pY += QK4_NL;
    vQs += QK4_NL / 2;
  }
}

void dequantize_row_iq4_xs(const block_iq4_xs *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {

    const uint8_t *vQs = pX[vI].qs;

    const float cD = GGML_FP16_TO_FP32(pX[vI].d);

    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const int cLs = ((pX[vI].scales_l[vIb / 2] >> 4 * (vIb % 2)) & 0xf) | (((pX[vI].scales_h >> 2 * vIb) & 3) << 4);
      const float cDl = cD * (cLs - 32);
      for (int vJ = 0; vJ < 16; ++vJ) {
        pY[vJ + 0] = cDl * kvalues_iq4nl[vQs[vJ] & 0xf];
        pY[vJ + 16] = cDl * kvalues_iq4nl[vQs[vJ] >> 4];
      }
      pY += 32;
      vQs += 16;
    }
  }
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_ref(const float *GGML_RESTRICT pX, block_q8_K *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {

    float vMax = 0;
    float vAmax = 0;
    for (int vJ = 0; vJ < QK_K; ++vJ) {
      float vAx = fabsf(pX[vJ]);
      if (vAx > vAmax) {
        vAmax = vAx;
        vMax = pX[vJ];
      }
    }
    if (!vAmax) {
      pY[vI].d = 0;
      memset(pY[vI].qs, 0, QK_K);
      pX += QK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    //  We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
    const float cIscale = -127.f / vMax;
    for (int vJ = 0; vJ < QK_K; ++vJ) {
      int v_ = fNearestInt(cIscale * pX[vJ]);
      pY[vI].qs[vJ] = MIN(127, v_);
    }
    for (int vJ = 0; vJ < QK_K / 16; ++vJ) {
      int vSum = 0;
      for (int vIi = 0; vIi < 16; ++vIi) {
        vSum += pY[vI].qs[vJ * 16 + vIi];
      }
      pY[vI].bsums[vJ] = vSum;
    }
    pY[vI].d = 1 / cIscale;
    pX += QK_K;
  }
}

void dequantize_row_q8_K(const block_q8_K *GGML_RESTRICT pX, float *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  const int64_t cNb = pK / QK_K;

  for (int vI = 0; vI < cNb; vI++) {
    for (int vJ = 0; vJ < QK_K; ++vJ) {
      *pY++ = pX[vI].d * pX[vI].qs[vJ];
    }
  }
}

// ================================ IQ2 quantization =============================================

typedef struct {
  uint64_t *vGrid;
  int *vMap;
  uint16_t *vNeighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int fIq2DataIndex(enum ggml_type pType) {
  GGML_ASSERT(pType == GGML_TYPE_IQ2_XXS || pType == GGML_TYPE_IQ2_XS || pType == GGML_TYPE_IQ1_S ||
              pType == GGML_TYPE_IQ1_M || pType == GGML_TYPE_IQ2_S);
  return pType == GGML_TYPE_IQ2_XXS                             ? 0
         : pType == GGML_TYPE_IQ2_XS                            ? 1
         : pType == GGML_TYPE_IQ1_S || pType == GGML_TYPE_IQ1_M ? 2
                                                                : 3;
}

static inline int fIq2GridSize(enum ggml_type pType) {
  GGML_ASSERT(pType == GGML_TYPE_IQ2_XXS || pType == GGML_TYPE_IQ2_XS || pType == GGML_TYPE_IQ1_S ||
              pType == GGML_TYPE_IQ1_M || pType == GGML_TYPE_IQ2_S);
  return pType == GGML_TYPE_IQ2_XXS                             ? 256
         : pType == GGML_TYPE_IQ2_XS                            ? 512
         : pType == GGML_TYPE_IQ1_S || pType == GGML_TYPE_IQ1_M ? NGRID_IQ1S
                                                                : 1024;
}

static int fIq2CompareFunc(const void *pLeft, const void *pRight) {
  const int *vL = (const int *)pLeft;
  const int *vR = (const int *)pRight;
  return vL[0] < vR[0] ? -1 : vL[0] > vR[0] ? 1 : vL[1] < vR[1] ? -1 : vL[1] > vR[1] ? 1 : 0;
}

void iq2xs_init_impl(enum ggml_type pType) {
  const int cGindex = fIq2DataIndex(pType);
  const int cGridSize = fIq2GridSize(pType);
  if (iq2_data[cGindex].vGrid) {
    return;
  }
  static const uint16_t aKgrid2bit256[256] = {
      0,     2,     5,     8,     10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
      1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
      1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
      2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
      4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
      5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
      8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
      10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
      16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
      17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
      20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
      22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
      25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
      33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
      37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
  };
  static const uint16_t aKgrid2bit512[512] = {
      0,     2,     5,     8,     10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
      73,    80,    82,    85,    88,    97,    100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
      260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
      352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
      640,   650,   1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
      1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
      1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
      2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
      2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
      4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
      4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
      5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
      5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
      8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
      8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
      10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
      16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
      16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
      16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
      17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
      18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
      20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
      21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
      22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
      24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
      32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
      33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
      33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
      35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
      37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
      40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
      42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
  };
  static const uint16_t aKgrid1bit2048[NGRID_IQ1S] = {
      0,     2,     5,     8,     10,    17,    21,    32,    34,    40,    42,    69,    81,    84,    86,    101,
      128,   130,   136,   138,   149,   160,   162,   168,   170,   260,   261,   273,   276,   278,   281,   282,
      293,   321,   326,   329,   338,   341,   346,   353,   356,   358,   360,   389,   401,   404,   406,   421,
      512,   514,   520,   522,   533,   544,   546,   552,   554,   581,   593,   601,   612,   617,   640,   642,
      648,   650,   657,   661,   665,   672,   674,   680,   682,   1041,  1044,  1046,  1061,  1089,  1097,  1109,
      1114,  1124,  1125,  1169,  1177,  1189,  1281,  1284,  1285,  1286,  1301,  1304,  1306,  1321,  1344,  1349,
      1354,  1360,  1361,  1364,  1365,  1366,  1369,  1376,  1378,  1381,  1384,  1386,  1409,  1425,  1429,  1432,
      1434,  1441,  1444,  1445,  1446,  1449,  1556,  1561,  1601,  1604,  1616,  1618,  1621,  1624,  1632,  1633,
      1638,  1641,  1669,  1681,  1684,  1689,  2048,  2050,  2056,  2058,  2069,  2080,  2082,  2088,  2090,  2117,
      2129,  2134,  2149,  2176,  2178,  2184,  2186,  2197,  2208,  2210,  2216,  2218,  2309,  2321,  2324,  2329,
      2340,  2341,  2369,  2384,  2385,  2389,  2401,  2404,  2409,  2449,  2452,  2454,  2457,  2469,  2560,  2562,
      2568,  2570,  2581,  2592,  2594,  2600,  2602,  2629,  2641,  2649,  2657,  2661,  2688,  2690,  2693,  2696,
      2698,  2709,  2720,  2722,  2728,  2730,  4112,  4113,  4116,  4121,  4132,  4133,  4161,  4164,  4176,  4181,
      4184,  4193,  4196,  4197,  4201,  4241,  4244,  4246,  4257,  4261,  4353,  4356,  4358,  4361,  4368,  4370,
      4373,  4376,  4385,  4388,  4393,  4421,  4426,  4432,  4433,  4434,  4436,  4437,  4438,  4441,  4448,  4453,
      4484,  4498,  4501,  4513,  4516,  4625,  4628,  4630,  4645,  4672,  4678,  4681,  4690,  4693,  4696,  4698,
      4708,  4710,  4741,  4753,  4756,  4758,  4773,  5121,  5126,  5129,  5140,  5141,  5144,  5145,  5153,  5158,
      5185,  5189,  5190,  5192,  5194,  5201,  5204,  5205,  5206,  5209,  5218,  5221,  5224,  5252,  5257,  5264,
      5268,  5269,  5272,  5273,  5274,  5281,  5284,  5285,  5289,  5378,  5381,  5386,  5393,  5396,  5397,  5398,
      5401,  5408,  5410,  5413,  5416,  5418,  5441,  5444,  5445,  5446,  5457,  5458,  5460,  5461,  5462,  5465,
      5466,  5473,  5476,  5477,  5478,  5481,  5504,  5506,  5508,  5509,  5512,  5514,  5520,  5521,  5524,  5525,
      5526,  5529,  5530,  5536,  5538,  5541,  5633,  5636,  5637,  5638,  5653,  5654,  5656,  5658,  5665,  5670,
      5696,  5698,  5700,  5701,  5704,  5706,  5713,  5717,  5718,  5720,  5721,  5729,  5732,  5733,  5736,  5737,
      5738,  5766,  5770,  5778,  5781,  5796,  5801,  6161,  6166,  6181,  6209,  6212,  6214,  6217,  6224,  6229,
      6232,  6234,  6240,  6241,  6244,  6246,  6249,  6277,  6289,  6292,  6309,  6416,  6418,  6421,  6426,  6433,
      6437,  6466,  6468,  6469,  6472,  6481,  6484,  6485,  6486,  6489,  6490,  6496,  6501,  6506,  6537,  6545,
      6546,  6549,  6552,  6561,  6566,  6569,  6665,  6678,  6692,  6694,  6724,  6726,  6729,  6736,  6738,  6741,
      6744,  6753,  6758,  6761,  6789,  6801,  6806,  6810,  8192,  8194,  8200,  8202,  8213,  8224,  8226,  8229,
      8232,  8234,  8261,  8273,  8281,  8289,  8293,  8320,  8322,  8328,  8330,  8341,  8352,  8354,  8357,  8360,
      8362,  8453,  8465,  8468,  8473,  8485,  8514,  8516,  8521,  8533,  8536,  8538,  8545,  8548,  8549,  8550,
      8581,  8592,  8598,  8601,  8613,  8705,  8712,  8714,  8721,  8725,  8736,  8738,  8744,  8746,  8773,  8785,
      8790,  8793,  8805,  8833,  8840,  8842,  8849,  8853,  8864,  8866,  8872,  8874,  9221,  9236,  9238,  9241,
      9253,  9284,  9285,  9286,  9289,  9298,  9301,  9304,  9306,  9318,  9349,  9361,  9364,  9369,  9377,  9381,
      9481,  9493,  9505,  9513,  9536,  9541,  9544,  9553,  9556,  9557,  9561,  9570,  9573,  9576,  9609,  9616,
      9620,  9621,  9624,  9626,  9633,  9636,  9638,  9641,  9733,  9744,  9746,  9753,  9765,  9793,  9801,  9813,
      9824,  9825,  9833,  9860,  9862,  9872,  9882,  10240, 10242, 10248, 10250, 10261, 10272, 10274, 10280, 10282,
      10309, 10321, 10324, 10341, 10368, 10370, 10376, 10378, 10400, 10402, 10408, 10410, 10505, 10513, 10516, 10521,
      10533, 10566, 10569, 10578, 10581, 10593, 10596, 10598, 10601, 10629, 10640, 10646, 10649, 10660, 10661, 10752,
      10754, 10760, 10762, 10784, 10786, 10792, 10794, 10821, 10833, 10838, 10841, 10853, 10880, 10882, 10888, 10890,
      10901, 10912, 10914, 10920, 10922, 16389, 16401, 16406, 16421, 16457, 16466, 16469, 16472, 16474, 16481, 16484,
      16486, 16532, 16537, 16545, 16550, 16640, 16641, 16644, 16646, 16649, 16658, 16661, 16662, 16664, 16666, 16673,
      16678, 16681, 16709, 16712, 16714, 16721, 16724, 16725, 16726, 16729, 16730, 16741, 16744, 16746, 16769, 16772,
      16774, 16784, 16786, 16789, 16800, 16801, 16802, 16901, 16913, 16916, 16918, 16933, 16961, 16978, 16981, 16986,
      16996, 17001, 17033, 17044, 17061, 17409, 17429, 17433, 17449, 17477, 17480, 17482, 17489, 17492, 17493, 17494,
      17505, 17506, 17509, 17512, 17514, 17537, 17542, 17545, 17552, 17554, 17557, 17568, 17569, 17577, 17665, 17666,
      17669, 17674, 17681, 17684, 17685, 17686, 17689, 17696, 17701, 17706, 17729, 17732, 17733, 17734, 17737, 17744,
      17745, 17748, 17749, 17750, 17752, 17753, 17761, 17764, 17765, 17766, 17769, 17794, 17796, 17797, 17800, 17809,
      17812, 17813, 17814, 17817, 17818, 17829, 17832, 17834, 17921, 17925, 17929, 17940, 17941, 17944, 17946, 17953,
      17956, 17961, 17984, 17986, 17989, 17992, 18000, 18001, 18002, 18005, 18006, 18009, 18018, 18021, 18024, 18049,
      18053, 18058, 18068, 18069, 18081, 18084, 18086, 18437, 18449, 18453, 18458, 18469, 18498, 18505, 18512, 18517,
      18520, 18529, 18532, 18534, 18537, 18565, 18577, 18580, 18582, 18585, 18597, 18689, 18693, 18694, 18698, 18704,
      18708, 18709, 18712, 18721, 18724, 18726, 18752, 18757, 18762, 18769, 18770, 18772, 18773, 18774, 18777, 18784,
      18786, 18789, 18790, 18794, 18822, 18825, 18834, 18837, 18838, 18840, 18849, 18852, 18854, 18857, 18966, 19012,
      19014, 19017, 19029, 19032, 19034, 19044, 19049, 19092, 19109, 20481, 20484, 20485, 20486, 20489, 20498, 20501,
      20506, 20513, 20516, 20521, 20544, 20549, 20552, 20561, 20564, 20565, 20566, 20569, 20581, 20584, 20614, 20617,
      20629, 20632, 20640, 20641, 20646, 20649, 20741, 20744, 20745, 20746, 20753, 20756, 20757, 20758, 20760, 20761,
      20768, 20773, 20774, 20776, 20778, 20801, 20804, 20805, 20806, 20809, 20816, 20817, 20818, 20820, 20821, 20822,
      20824, 20825, 20826, 20833, 20836, 20837, 20838, 20841, 20866, 20869, 20881, 20884, 20885, 20886, 20889, 20896,
      20901, 20906, 20993, 20998, 21010, 21013, 21018, 21025, 21028, 21058, 21061, 21066, 21073, 21076, 21077, 21078,
      21081, 21090, 21093, 21125, 21136, 21138, 21141, 21145, 21146, 21156, 21508, 21509, 21521, 21524, 21525, 21526,
      21528, 21529, 21537, 21541, 21544, 21546, 21569, 21572, 21573, 21574, 21577, 21578, 21584, 21585, 21588, 21589,
      21590, 21592, 21593, 21594, 21601, 21602, 21604, 21605, 21606, 21609, 21632, 21640, 21642, 21649, 21652, 21653,
      21654, 21657, 21665, 21668, 21669, 21674, 21761, 21762, 21764, 21765, 21766, 21769, 21776, 21777, 21778, 21780,
      21781, 21782, 21785, 21786, 21793, 21796, 21797, 21798, 21801, 21824, 21825, 21826, 21828, 21829, 21830, 21832,
      21833, 21840, 21841, 21842, 21844, 21845, 21846, 21848, 21849, 21850, 21856, 21857, 21860, 21861, 21862, 21864,
      21865, 21866, 21889, 21892, 21893, 21897, 21898, 21904, 21905, 21908, 21909, 21910, 21912, 21913, 21921, 21924,
      21925, 21926, 21929, 22016, 22017, 22018, 22020, 22022, 22024, 22025, 22033, 22036, 22037, 22040, 22041, 22048,
      22049, 22050, 22052, 22053, 22054, 22056, 22057, 22081, 22085, 22086, 22088, 22089, 22090, 22096, 22097, 22098,
      22100, 22101, 22102, 22104, 22105, 22106, 22113, 22116, 22117, 22121, 22146, 22149, 22150, 22152, 22153, 22154,
      22161, 22165, 22170, 22178, 22181, 22182, 22184, 22185, 22532, 22533, 22534, 22537, 22544, 22549, 22552, 22561,
      22570, 22597, 22600, 22602, 22609, 22612, 22613, 22614, 22616, 22617, 22624, 22626, 22628, 22629, 22658, 22665,
      22672, 22674, 22677, 22680, 22689, 22697, 22785, 22786, 22789, 22794, 22801, 22804, 22805, 22806, 22809, 22821,
      22849, 22852, 22853, 22854, 22857, 22864, 22865, 22866, 22868, 22869, 22870, 22872, 22873, 22874, 22881, 22884,
      22885, 22886, 22889, 22913, 22917, 22921, 22929, 22932, 22933, 22934, 22936, 22937, 22949, 23044, 23048, 23061,
      23066, 23072, 23077, 23078, 23081, 23109, 23112, 23113, 23121, 23125, 23126, 23128, 23129, 23138, 23141, 23144,
      23146, 23169, 23178, 23186, 23189, 23190, 23192, 23194, 23201, 24581, 24596, 24598, 24601, 24613, 24644, 24656,
      24661, 24662, 24664, 24666, 24673, 24676, 24678, 24681, 24705, 24726, 24741, 24833, 24836, 24838, 24841, 24850,
      24853, 24865, 24866, 24870, 24873, 24901, 24905, 24913, 24917, 24918, 24921, 24933, 24934, 24938, 24964, 24970,
      24978, 24981, 24993, 24998, 25001, 25105, 25110, 25113, 25152, 25153, 25158, 25173, 25174, 25176, 25184, 25221,
      25233, 25238, 25253, 25617, 25618, 25621, 25622, 25626, 25633, 25638, 25641, 25664, 25666, 25669, 25672, 25674,
      25681, 25684, 25685, 25686, 25689, 25690, 25696, 25698, 25701, 25732, 25733, 25737, 25744, 25746, 25748, 25749,
      25750, 25752, 25754, 25761, 25764, 25769, 25861, 25864, 25866, 25873, 25877, 25878, 25881, 25924, 25925, 25926,
      25929, 25936, 25937, 25940, 25941, 25942, 25945, 25953, 25956, 25957, 25958, 25961, 25990, 25993, 25994, 26001,
      26005, 26006, 26009, 26010, 26018, 26021, 26022, 26024, 26114, 26121, 26133, 26144, 26150, 26152, 26153, 26176,
      26181, 26184, 26186, 26193, 26196, 26197, 26198, 26200, 26202, 26208, 26213, 26216, 26240, 26242, 26245, 26250,
      26260, 26262, 26264, 26265, 26272, 26276, 26278, 26282, 26646, 26649, 26661, 26689, 26706, 26709, 26714, 26721,
      26729, 26757, 26769, 26776, 26790, 26881, 26884, 26896, 26901, 26913, 26916, 26918, 26921, 26944, 26945, 26949,
      26950, 26952, 26961, 26964, 26965, 26966, 26969, 26976, 26981, 26986, 27010, 27012, 27018, 27029, 27041, 27044,
      27045, 27049, 27153, 27158, 27160, 27201, 27204, 27209, 27216, 27221, 27224, 27226, 27236, 27237, 27241, 27270,
      27284, 27288, 27290, 27302, 32768, 32770, 32776, 32778, 32800, 32802, 32808, 32810, 32837, 32848, 32849, 32852,
      32854, 32857, 32869, 32896, 32898, 32904, 32906, 32917, 32928, 32930, 32936, 32938, 33029, 33041, 33044, 33046,
      33049, 33061, 33089, 33092, 33097, 33104, 33106, 33109, 33110, 33112, 33113, 33124, 33126, 33129, 33157, 33161,
      33172, 33174, 33177, 33189, 33280, 33282, 33288, 33290, 33301, 33312, 33314, 33320, 33322, 33361, 33364, 33369,
      33381, 33408, 33410, 33416, 33418, 33429, 33440, 33442, 33448, 33450, 33812, 33817, 33857, 33860, 33873, 33877,
      33882, 33889, 33892, 33897, 33940, 33945, 34049, 34057, 34066, 34069, 34074, 34086, 34089, 34112, 34113, 34117,
      34120, 34129, 34132, 34133, 34134, 34137, 34138, 34149, 34150, 34152, 34154, 34177, 34180, 34182, 34185, 34192,
      34194, 34197, 34200, 34214, 34321, 34326, 34329, 34341, 34369, 34372, 34377, 34378, 34384, 34389, 34393, 34394,
      34401, 34406, 34410, 34437, 34449, 34458, 34468, 34816, 34818, 34824, 34826, 34837, 34848, 34850, 34856, 34858,
      34881, 34885, 34897, 34900, 34905, 34917, 34921, 34944, 34946, 34952, 34954, 34965, 34976, 34978, 34984, 34986,
      35077, 35078, 35089, 35092, 35094, 35109, 35137, 35140, 35142, 35145, 35152, 35154, 35157, 35162, 35169, 35172,
      35205, 35222, 35225, 35237, 35328, 35330, 35336, 35338, 35349, 35360, 35362, 35368, 35370, 35397, 35409, 35412,
      35414, 35456, 35458, 35464, 35466, 35477, 35488, 35490, 35496, 35498, 36869, 36881, 36886, 36888, 36889, 36901,
      36929, 36934, 36937, 36949, 36952, 36954, 36969, 36970, 36997, 37009, 37012, 37014, 37017, 37029, 37121, 37124,
      37126, 37129, 37136, 37141, 37144, 37146, 37153, 37156, 37158, 37161, 37184, 37189, 37200, 37201, 37204, 37205,
      37206, 37209, 37218, 37221, 37252, 37254, 37266, 37269, 37272, 37281, 37284, 37286, 37289, 37381, 37393, 37396,
      37401, 37413, 37444, 37446, 37449, 37456, 37458, 37461, 37464, 37478, 37481, 37509, 37524, 37526, 37545, 37889,
      37892, 37894, 37904, 37909, 37912, 37926, 37952, 37962, 37969, 37972, 37973, 37974, 37976, 37977, 37984, 37985,
      37986, 37989, 38020, 38022, 38034, 38036, 38037, 38040, 38049, 38057, 38144, 38149, 38152, 38154, 38160, 38161,
      38164, 38165, 38166, 38169, 38177, 38181, 38185, 38186, 38209, 38212, 38213, 38214, 38217, 38224, 38225, 38226,
      38228, 38229, 38230, 38232, 38233, 38234, 38241, 38244, 38245, 38246, 38249, 38273, 38277, 38280, 38289, 38290,
      38292, 38293, 38294, 38297, 38298, 38304, 38306, 38309, 38312, 38314, 38401, 38404, 38416, 38421, 38425, 38432,
      38438, 38441, 38469, 38472, 38473, 38481, 38482, 38485, 38486, 38489, 38501, 38504, 38530, 38532, 38537, 38538,
      38546, 38548, 38549, 38564, 38566, 38569, 38917, 38934, 38937, 38949, 38977, 38982, 38992, 38994, 38997, 38998,
      39002, 39012, 39013, 39045, 39057, 39062, 39065, 39077, 39172, 39174, 39177, 39184, 39186, 39189, 39192, 39194,
      39200, 39201, 39204, 39206, 39232, 39234, 39237, 39240, 39242, 39249, 39252, 39253, 39254, 39257, 39266, 39269,
      39270, 39274, 39297, 39300, 39312, 39314, 39317, 39322, 39329, 39334, 39429, 39445, 39461, 39492, 39494, 39497,
      39504, 39509, 39512, 39521, 39557, 39569, 39572, 39573, 39574, 40960, 40962, 40968, 40970, 40981, 40992, 40994,
      41000, 41002, 41029, 41041, 41044, 41046, 41049, 41088, 41090, 41096, 41098, 41109, 41120, 41122, 41128, 41130,
      41221, 41225, 41233, 41236, 41238, 41241, 41242, 41286, 41289, 41297, 41301, 41304, 41306, 41313, 41316, 41349,
      41360, 41362, 41366, 41369, 41474, 41480, 41482, 41488, 41497, 41506, 41512, 41514, 41541, 41553, 41558, 41561,
      41573, 41600, 41602, 41608, 41610, 41621, 41632, 41634, 41640, 41642, 42009, 42021, 42049, 42052, 42064, 42068,
      42069, 42072, 42074, 42081, 42085, 42086, 42088, 42089, 42117, 42246, 42249, 42256, 42258, 42261, 42264, 42278,
      42281, 42306, 42309, 42321, 42324, 42325, 42326, 42329, 42341, 42346, 42369, 42372, 42373, 42374, 42377, 42386,
      42389, 42392, 42501, 42513, 42518, 42522, 42529, 42533, 42564, 42566, 42570, 42578, 42581, 42582, 42584, 42592,
      42594, 42630, 42640, 42645, 42646, 42649, 42657, 42660, 42662, 43008, 43010, 43016, 43018, 43040, 43042, 43048,
      43050, 43089, 43092, 43094, 43097, 43136, 43138, 43144, 43146, 43157, 43168, 43170, 43176, 43178, 43269, 43284,
      43289, 43297, 43301, 43329, 43344, 43349, 43354, 43361, 43366, 43369, 43408, 43414, 43520, 43522, 43528, 43530,
      43552, 43554, 43560, 43562, 43601, 43604, 43606, 43648, 43650, 43656, 43658, 43669, 43680, 43682, 43688, 43690,
  };
  static const uint16_t aKgrid2bit1024[1024] = {
      0,     2,     5,     8,     10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
      73,    80,    82,    85,    88,    97,    100,   102,   105,   128,   130,   133,   136,   145,   148,   160,
      165,   170,   257,   260,   262,   265,   272,   274,   277,   280,   289,   292,   320,   322,   325,   328,
      337,   340,   342,   345,   352,   357,   360,   385,   388,   400,   402,   405,   417,   420,   512,   514,
      517,   520,   529,   532,   544,   554,   577,   580,   582,   585,   592,   597,   640,   645,   650,   660,
      674,   1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1062,  1065,  1088,  1090,  1093,
      1096,  1098,  1105,  1108,  1110,  1113,  1120,  1122,  1125,  1153,  1156,  1158,  1161,  1168,  1173,  1176,
      1185,  1188,  1280,  1282,  1285,  1288,  1290,  1297,  1300,  1302,  1305,  1312,  1317,  1320,  1345,  1348,
      1350,  1353,  1360,  1362,  1365,  1368,  1377,  1380,  1408,  1410,  1413,  1416,  1425,  1428,  1440,  1537,
      1540,  1542,  1545,  1552,  1557,  1600,  1605,  1608,  1617,  1620,  1632,  1665,  1668,  1680,  2048,  2050,
      2053,  2056,  2065,  2068,  2070,  2073,  2080,  2085,  2090,  2113,  2116,  2118,  2121,  2128,  2130,  2133,
      2136,  2145,  2148,  2176,  2181,  2196,  2218,  2305,  2308,  2320,  2322,  2325,  2328,  2337,  2368,  2373,
      2376,  2385,  2388,  2400,  2433,  2448,  2560,  2577,  2580,  2594,  2600,  2602,  2640,  2713,  4097,  4100,
      4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4134,  4160,  4162,  4165,  4168,  4177,  4180,  4182,
      4185,  4192,  4194,  4197,  4200,  4225,  4228,  4230,  4240,  4245,  4248,  4257,  4260,  4352,  4354,  4357,
      4360,  4362,  4369,  4372,  4374,  4377,  4384,  4386,  4389,  4392,  4417,  4420,  4422,  4425,  4432,  4434,
      4437,  4440,  4449,  4452,  4480,  4482,  4485,  4488,  4497,  4500,  4609,  4612,  4617,  4624,  4629,  4641,
      4644,  4672,  4677,  4689,  4692,  4737,  4740,  4752,  5120,  5122,  5125,  5128,  5137,  5140,  5142,  5145,
      5152,  5157,  5160,  5185,  5188,  5190,  5193,  5200,  5202,  5205,  5208,  5217,  5220,  5248,  5250,  5253,
      5256,  5265,  5268,  5280,  5377,  5380,  5382,  5385,  5392,  5394,  5397,  5400,  5409,  5412,  5440,  5442,
      5445,  5448,  5457,  5460,  5472,  5505,  5508,  5520,  5632,  5637,  5640,  5649,  5652,  5664,  5697,  5700,
      5712,  5760,  5802,  6145,  6148,  6150,  6153,  6160,  6165,  6168,  6177,  6208,  6210,  6213,  6216,  6225,
      6228,  6240,  6273,  6276,  6400,  6402,  6405,  6408,  6417,  6420,  6432,  6465,  6468,  6480,  6505,  6562,
      6660,  6672,  6720,  6742,  8192,  8194,  8197,  8200,  8209,  8212,  8214,  8217,  8224,  8229,  8234,  8257,
      8260,  8272,  8274,  8277,  8292,  8320,  8330,  8340,  8362,  8449,  8452,  8464,  8466,  8469,  8481,  8512,
      8514,  8517,  8529,  8532,  8544,  8577,  8580,  8592,  8704,  8714,  8738,  8744,  8746,  8772,  8784,  8840,
      8842,  8872,  9217,  9220,  9222,  9225,  9232,  9237,  9240,  9249,  9252,  9280,  9282,  9285,  9288,  9297,
      9300,  9312,  9345,  9348,  9360,  9472,  9477,  9480,  9489,  9492,  9504,  9537,  9540,  9552,  9574,  9600,
      9729,  9732,  9744,  9792,  9817,  10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
      10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
      16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
      16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
      16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
      16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
      17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
      17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
      17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
      17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
      18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
      18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
      18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
      20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
      20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
      20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
      21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
      21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
      22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
      22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
      24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
      24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
      25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
      26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
      32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
      33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
      33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
      33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
      34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
      35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
      36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
      37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
      38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
      39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
      41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
      42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
  };

  const int cKmapSize = 43692;
  // const int nwant = type == GGML_TYPE_IQ1_S ? 3 : 2;
  const int cNwant = pType == GGML_TYPE_IQ1_S || pType == GGML_TYPE_IQ1_M ? 3 : pType == GGML_TYPE_IQ2_S ? 1 : 2;
  const uint16_t *vKgrid = pType == GGML_TYPE_IQ2_XXS                             ? aKgrid2bit256
                           : pType == GGML_TYPE_IQ2_XS                            ? aKgrid2bit512
                           : pType == GGML_TYPE_IQ1_S || pType == GGML_TYPE_IQ1_M ? aKgrid1bit2048
                                                                                  : aKgrid2bit1024;
  uint64_t *vKgridQ2xs;
  int *vKmapQ2xs;
  uint16_t *vKneighborsQ2xs;

  // printf("================================================================= %s(grid_size = %d)\n", __func__,
  // grid_size);
  uint64_t *vTheGrid = (uint64_t *)malloc(cGridSize * sizeof(uint64_t));
  for (int vK = 0; vK < cGridSize; ++vK) {
    int8_t *vPos = (int8_t *)(vTheGrid + vK);
    for (int vI = 0; vI < 8; ++vI) {
      int vL = (vKgrid[vK] >> 2 * vI) & 0x3;
      vPos[vI] = 2 * vL + 1;
    }
  }
  vKgridQ2xs = vTheGrid;
  iq2_data[cGindex].vGrid = vTheGrid;
  vKmapQ2xs = (int *)malloc(cKmapSize * sizeof(int));
  iq2_data[cGindex].vMap = vKmapQ2xs;
  for (int vI = 0; vI < cKmapSize; ++vI)
    vKmapQ2xs[vI] = -1;
  uint64_t vAux64;
  uint8_t *vAux8 = (uint8_t *)&vAux64;
  for (int vI = 0; vI < cGridSize; ++vI) {
    vAux64 = vKgridQ2xs[vI];
    uint16_t vIndex = 0;
    for (int vK = 0; vK < 8; ++vK) {
      uint16_t vQ = (vAux8[vK] - 1) / 2;
      vIndex |= (vQ << 2 * vK);
    }
    vKmapQ2xs[vIndex] = vI;
  }
  // The neighbour search runs in three passes:
  //   1. Parallel: for each i, qsort and count its neighbours into n_per_i,
  //      and reduce the totals (num_neighbors, num_not_in_map).
  //   2. Serial: prefix-sum n_per_i into offsets[], so each i has a
  //      pre-assigned slice of kneighbors_q2xs to write into.
  //   3. Parallel: redo the qsort and write each i's neighbour list at
  //      offsets[i].
  int *vNPerI = (int *)malloc(cKmapSize * sizeof(int));
  GGML_ASSERT(vNPerI);
  int vNumNeighbors = 0, vNumNotInMap = 0;
#ifdef GGML_USE_OPENMP
#pragma omp parallel reduction(+ : vNumNeighbors, vNumNotInMap)
#endif
  {
    int *dist2 = (int *)malloc(2 * cGridSize * sizeof(int));
    GGML_ASSERT(dist2);
    int8_t pos[8];
    int i;
#ifdef GGML_USE_OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (i = 0; i < cKmapSize; ++i) {
      if (vKmapQ2xs[i] >= 0) {
        vNPerI[i] = 0;
        continue;
      }
      ++vNumNotInMap;
      for (int k = 0; k < 8; ++k) {
        int l = (i >> 2 * k) & 0x3;
        pos[k] = 2 * l + 1;
      }
      for (int j = 0; j < cGridSize; ++j) {
        const int8_t *pg = (const int8_t *)(vKgridQ2xs + j);
        int d2 = 0;
        for (int k = 0; k < 8; ++k)
          d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
        dist2[2 * j + 0] = d2;
        dist2[2 * j + 1] = j;
      }
      qsort(dist2, cGridSize, 2 * sizeof(int), fIq2CompareFunc);
      int n = 0;
      int d2 = dist2[0];
      int nhave = 1;
      for (int j = 0; j < cGridSize; ++j) {
        if (dist2[2 * j] > d2) {
          if (nhave == cNwant)
            break;
          d2 = dist2[2 * j];
          ++nhave;
        }
        ++n;
      }
      vNPerI[i] = n;
      vNumNeighbors += n;
    }
    free(dist2);
  }
  // printf("%s: %d neighbours in total\n", __func__, num_neighbors);
  vKneighborsQ2xs = (uint16_t *)malloc((vNumNeighbors + vNumNotInMap) * sizeof(uint16_t));
  iq2_data[cGindex].vNeighbours = vKneighborsQ2xs;

  int *vOffsets = (int *)malloc(cKmapSize * sizeof(int));
  GGML_ASSERT(vOffsets);
  int vCounter = 0;
  for (int vI = 0; vI < cKmapSize; ++vI) {
    if (vKmapQ2xs[vI] >= 0) {
      vOffsets[vI] = -1;
      continue;
    }
    vOffsets[vI] = vCounter;
    vCounter += 1 + vNPerI[vI];
  }

#ifdef GGML_USE_OPENMP
#pragma omp parallel
#endif
  {
    int *dist2 = (int *)malloc(2 * cGridSize * sizeof(int));
    GGML_ASSERT(dist2);
    int8_t pos[8];
    int i;
#ifdef GGML_USE_OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (i = 0; i < cKmapSize; ++i) {
      if (vKmapQ2xs[i] >= 0)
        continue;
      for (int k = 0; k < 8; ++k) {
        int l = (i >> 2 * k) & 0x3;
        pos[k] = 2 * l + 1;
      }
      for (int j = 0; j < cGridSize; ++j) {
        const int8_t *pg = (const int8_t *)(vKgridQ2xs + j);
        int d2 = 0;
        for (int k = 0; k < 8; ++k)
          d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
        dist2[2 * j + 0] = d2;
        dist2[2 * j + 1] = j;
      }
      qsort(dist2, cGridSize, 2 * sizeof(int), fIq2CompareFunc);
      int local_counter = vOffsets[i];
      vKmapQ2xs[i] = -(local_counter + 1);
      int d2 = dist2[0];
      uint16_t *start = &vKneighborsQ2xs[local_counter++];
      int n = 0, nhave = 1;
      for (int j = 0; j < cGridSize; ++j) {
        if (dist2[2 * j] > d2) {
          if (nhave == cNwant)
            break;
          d2 = dist2[2 * j];
          ++nhave;
        }
        vKneighborsQ2xs[local_counter++] = dist2[2 * j + 1];
        ++n;
      }
      *start = n;
    }
    free(dist2);
  }
  free(vOffsets);
  free(vNPerI);
}

void iq2xs_free_impl(enum ggml_type pType) {
  GGML_ASSERT(pType == GGML_TYPE_IQ2_XXS || pType == GGML_TYPE_IQ2_XS || pType == GGML_TYPE_IQ1_S ||
              pType == GGML_TYPE_IQ1_M || pType == GGML_TYPE_IQ2_S);
  const int cGindex = fIq2DataIndex(pType);
  if (iq2_data[cGindex].vGrid) {
    free(iq2_data[cGindex].vGrid);
    iq2_data[cGindex].vGrid = NULL;
    free(iq2_data[cGindex].vMap);
    iq2_data[cGindex].vMap = NULL;
    free(iq2_data[cGindex].vNeighbours);
    iq2_data[cGindex].vNeighbours = NULL;
  }
}

static int fIq2FindBestNeighbour(const uint16_t *GGML_RESTRICT pNeighbours, const uint64_t *GGML_RESTRICT pGrid,
                                 const float *GGML_RESTRICT pXval, const float *GGML_RESTRICT pWeight, float pScale,
                                 int8_t *GGML_RESTRICT pL) {
  int vNumNeighbors = pNeighbours[0];
  GGML_ASSERT(vNumNeighbors > 0);
  float vBestD2 = FLT_MAX;
  int vGridIndex = -1;
  for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
    const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
    float vD2 = 0;
    for (int vI = 0; vI < 8; ++vI) {
      float vQ = vPg[vI];
      float vDiff = pScale * vQ - pXval[vI];
      vD2 += pWeight[vI] * vDiff * vDiff;
    }
    if (vD2 < vBestD2) {
      vBestD2 = vD2;
      vGridIndex = pNeighbours[vJ];
    }
  }
  GGML_ASSERT(vGridIndex >= 0);
  const int8_t *vPg = (const int8_t *)(pGrid + vGridIndex);
  for (int vI = 0; vI < 8; ++vI)
    pL[vI] = (vPg[vI] - 1) / 2;
  return vGridIndex;
}

static void fQuantizeRowIq2XxsImpl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                   const float *GGML_RESTRICT pQuantWeights) {

  const int cGindex = fIq2DataIndex(GGML_TYPE_IQ2_XXS);

  const uint64_t *vKgridQ2xs = iq2_data[cGindex].vGrid;
  const int *vKmapQ2xs = iq2_data[cGindex].vMap;
  const uint16_t *vKneighborsQ2xs = iq2_data[cGindex].vNeighbours;

  GGML_ASSERT(pQuantWeights && "missing quantization weights");
  GGML_ASSERT(vKgridQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKmapQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  const int cKMaxQ = 3;

  const int64_t cNbl = pN / QK_K;

  block_iq2_xxs *vY = pVy;

  float vScales[QK_K / 32];
  float aWeight[32];
  float vXval[32];
  int8_t L[32];
  int8_t aLaux[32];
  float aWaux[32];
  uint8_t aBlockSigns[4];
  uint32_t aQ2[2 * (QK_K / 32)];

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    vY[vIbl].d = GGML_FP32_TO_FP16(0.f);
    memset(aQ2, 0, QK_K / 4);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const float *vXb = vXbl + 32 * vIb;
      const float *vQw = pQuantWeights + QK_K * vIbl + 32 * vIb;
      for (int vI = 0; vI < 32; ++vI)
        aWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      for (int vI = 0; vI < 32; ++vI)
        aWaux[vI] = sqrtf(aWeight[vI]);
      for (int vK = 0; vK < 4; ++vK) {
        int vNflip = 0;
        uint8_t vS = 0;
        for (int vI = 0; vI < 8; ++vI) {
          if (vXb[8 * vK + vI] >= 0)
            vXval[8 * vK + vI] = vXb[8 * vK + vI];
          else {
            vXval[8 * vK + vI] = -vXb[8 * vK + vI];
            ++vNflip;
            vS |= (1 << vI);
          }
        }
        if (vNflip % 2) {
          int vImin = 0;
          float vMin = aWeight[8 * vK + vImin] * vXb[8 * vK + vImin] * vXb[8 * vK + vImin];
          for (int vI = 1; vI < 8; ++vI) {
            float vAx = aWeight[8 * vK + vI] * vXb[8 * vK + vI] * vXb[8 * vK + vI];
            if (vAx < vMin) {
              vMin = vAx;
              vImin = vI;
            }
          }
          vXval[8 * vK + vImin] = -vXval[8 * vK + vImin];
          vS ^= (1 << vImin);
        }
        aBlockSigns[vK] = vS & 127;
      }
      float vMax = vXval[0];
      for (int vI = 1; vI < 32; ++vI)
        vMax = MAX(vMax, vXval[vI]);
      if (vMax < GROUP_MAX_EPS) {
        vScales[vIb] = 0;
        memset(L, 0, 32);
        continue;
      }
      float vScale = fMakeQpQuants(32, cKMaxQ + 1, vXval, (uint8_t *)L, aWeight);
      float vEffMax = vScale * cKMaxQ;
      if (vEffMax <= 0) {
        vScales[vIb] = 0;
        memset(L, 0, 32);
        continue;
      }
      float vBest = 0;
      for (int vIs = -6; vIs <= 6; ++vIs) {
        float vId = (2 * cKMaxQ - 1 + vIs * 0.1f) / vEffMax;
        float vThisScale = 1 / vId;
        for (int vK = 0; vK < 4; ++vK) {
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            aLaux[8 * vK + vI] = MAX(0, MIN(cKMaxQ - 1, l));
          }
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI)
            vU |= (aLaux[8 * vK + vI] << 2 * vI);
          int vGridIndex = vKmapQ2xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex = fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vThisScale,
                                               aLaux + 8 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 32; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * aLaux[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
          vScale = vSumqx / vSumq2;
          vBest = vScale * vSumqx;
          memcpy(L, aLaux, 32);
        }
      }
      if (vScale > 0) {
        float vId = 1 / vScale;
        for (int vK = 0; vK < 4; ++vK) {
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            l = MAX(0, MIN(cKMaxQ - 1, l));
            vU |= (l << 2 * vI);
          }
          int vGridIndex = vKmapQ2xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex =
                fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vScale, L + 8 * vK);
          }
          const int8_t *vPg = (const int8_t *)(vKgridQ2xs + vGridIndex);
          for (int vI = 0; vI < 8; ++vI)
            L[8 * vK + vI] = (vPg[vI] - 1) / 2;
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 32; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * L[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      if (vScale < 0) {
        // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the
        // scale) and correspondingly flip quant signs.
        vScale = -vScale;
        for (int vK = 0; vK < 4; ++vK)
          aBlockSigns[vK] = (~aBlockSigns[vK]) & 127;
      }
      for (int vK = 0; vK < 4; ++vK) {
        uint16_t vU = 0;
        for (int vI = 0; vI < 8; ++vI)
          vU |= (L[8 * vK + vI] << 2 * vI);
        int vGridIndex = vKmapQ2xs[vU];
        if (vGridIndex < 0) {
          printf("Oops: found point %u not on grid:", vU);
          for (int vI = 0; vI < 8; ++vI)
            printf(" %d", L[8 * vK + vI]);
          printf("\n");
          GGML_ABORT("fatal error");
        }
        aQ2[2 * vIb + 0] |= ((uint32_t)vGridIndex << 8 * vK);
        aQ2[2 * vIb + 1] |= (aBlockSigns[vK] << 7 * vK);
      }
      GGML_ASSERT(vScale >= 0);
      vScales[vIb] = vScale;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      memset(vY[vIbl].qs, 0, QK_K / 4);
      continue;
    }

    float vD = vMaxScale / 31;
    vY[vIbl].d = GGML_FP32_TO_FP16(vD);
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      int l = fNearestInt(0.5f * (vId * vScales[vIb] - 1));
      l = MAX(0, MIN(15, l));
      aQ2[2 * vIb + 1] |= ((uint32_t)l << 28);
    }
    memcpy(vY[vIbl].qs, aQ2, QK_K / 4);
  }
}

static void fQuantizeRowIq2XsImpl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                  const float *GGML_RESTRICT pQuantWeights) {

  const int cGindex = fIq2DataIndex(GGML_TYPE_IQ2_XS);

  const uint64_t *vKgridQ2xs = iq2_data[cGindex].vGrid;
  const int *vKmapQ2xs = iq2_data[cGindex].vMap;
  const uint16_t *vKneighborsQ2xs = iq2_data[cGindex].vNeighbours;

  GGML_ASSERT(pQuantWeights && "missing quantization weights");
  GGML_ASSERT(vKmapQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKgridQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  const int cKMaxQ = 3;

  const int64_t cNbl = pN / QK_K;

  block_iq2_xs *vY = pVy;

  float vScales[QK_K / 16];
  float aWeight[16];
  float vXval[16];
  int8_t L[16];
  int8_t aLaux[16];
  float aWaux[16];
  bool vIsOnGrid[2];
  bool vIsOnGridAux[2];
  uint8_t aBlockSigns[2];
  uint16_t aQ2[2 * (QK_K / 16)];

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    vY[vIbl].d = GGML_FP32_TO_FP16(0.f);
    memset(aQ2, 0, QK_K / 4);
    memset(vY[vIbl].scales, 0, QK_K / 32);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      const float *vXb = vXbl + 16 * vIb;
      const float *vQw = pQuantWeights + QK_K * vIbl + 16 * vIb;
      for (int vI = 0; vI < 16; ++vI)
        aWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      for (int vI = 0; vI < 16; ++vI)
        aWaux[vI] = sqrtf(aWeight[vI]);
      for (int vK = 0; vK < 2; ++vK) {
        int vNflip = 0;
        uint8_t vS = 0;
        for (int vI = 0; vI < 8; ++vI) {
          if (vXb[8 * vK + vI] >= 0)
            vXval[8 * vK + vI] = vXb[8 * vK + vI];
          else {
            vXval[8 * vK + vI] = -vXb[8 * vK + vI];
            ++vNflip;
            vS |= (1 << vI);
          }
        }
        if (vNflip % 2) {
          int vImin = 0;
          float vMin = aWeight[8 * vK + vImin] * vXb[8 * vK + vImin] * vXb[8 * vK + vImin];
          for (int vI = 1; vI < 8; ++vI) {
            float vAx = aWeight[8 * vK + vI] * vXb[8 * vK + vI] * vXb[8 * vK + vI];
            if (vAx < vMin) {
              vMin = vAx;
              vImin = vI;
            }
          }
          vXval[8 * vK + vImin] = -vXval[8 * vK + vImin];
          vS ^= (1 << vImin);
        }
        aBlockSigns[vK] = vS & 127;
      }
      float vMax = vXval[0];
      for (int vI = 1; vI < 16; ++vI)
        vMax = MAX(vMax, vXval[vI]);
      memset(L, 0, 16);
      if (vMax < GROUP_MAX_EPS) {
        vScales[vIb] = 0;
        continue;
      }
      float vBest = 0;
      float vScale = vMax / (2 * cKMaxQ - 1);
      vIsOnGrid[0] = vIsOnGrid[1] = true;
      for (int vIs = -9; vIs <= 9; ++vIs) {
        float vId = (2 * cKMaxQ - 1 + vIs * 0.1f) / vMax;
        float vThisScale = 1 / vId;
        for (int vK = 0; vK < 2; ++vK) {
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            aLaux[8 * vK + vI] = MAX(0, MIN(cKMaxQ - 1, l));
          }
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI)
            vU |= (aLaux[8 * vK + vI] << 2 * vI);
          int vGridIndex = vKmapQ2xs[vU];
          vIsOnGridAux[vK] = true;
          if (vGridIndex < 0) {
            vIsOnGridAux[vK] = false;
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex = fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vThisScale,
                                               aLaux + 8 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 16; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * aLaux[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
          vScale = vSumqx / vSumq2;
          vBest = vScale * vSumqx;
          for (int vI = 0; vI < 16; ++vI)
            L[vI] = aLaux[vI];
          for (int vK = 0; vK < 2; ++vK)
            vIsOnGrid[vK] = vIsOnGridAux[vK];
        }
      }
      int vNNotOngrid = 0;
      for (int vK = 0; vK < 2; ++vK)
        if (!vIsOnGrid[vK])
          ++vNNotOngrid;
      if (vNNotOngrid > 0 && vScale > 0) {
        float vId = 1 / vScale;
        for (int vK = 0; vK < 2; ++vK) {
          if (vIsOnGrid[vK])
            continue;
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            l = MAX(0, MIN(cKMaxQ - 1, l));
            vU |= (l << 2 * vI);
            L[8 * vK + vI] = l;
          }
          int vGridIndex = vKmapQ2xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex =
                fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vScale, L + 8 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 16; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * L[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      if (vScale < 0) {
        vScale = -vScale;
        for (int vK = 0; vK < 2; ++vK)
          aBlockSigns[vK] = (~aBlockSigns[vK]) & 127;
      }
      for (int vK = 0; vK < 2; ++vK) {
        uint16_t vU = 0;
        for (int vI = 0; vI < 8; ++vI)
          vU |= (L[8 * vK + vI] << 2 * vI);
        int vGridIndex = vKmapQ2xs[vU];
        if (vGridIndex < 0) {
          printf("Oops: found point %u not on grid:", vU);
          for (int vI = 0; vI < 8; ++vI)
            printf(" %d", L[8 * vK + vI]);
          printf("\n");
          GGML_ABORT("fatal error");
        }
        aQ2[2 * vIb + vK] = vGridIndex | (aBlockSigns[vK] << 9);
      }
      GGML_ASSERT(vScale >= 0);
      vScales[vIb] = vScale;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      memset(vY[vIbl].qs, 0, QK_K / 4);
      continue;
    }

    float vD = vMaxScale / 31;
    vY[vIbl].d = GGML_FP32_TO_FP16(vD);
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      int l = fNearestInt(0.5f * (vId * vScales[vIb] - 1));
      l = MAX(0, MIN(15, l));
      if (vIb % 2 == 0)
        vY[vIbl].scales[vIb / 2] = l;
      else
        vY[vIbl].scales[vIb / 2] |= (l << 4);
    }
    memcpy(vY[vIbl].qs, aQ2, QK_K / 4);
  }
}

size_t quantize_iq2_xxs(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                        const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq2XxsImpl(pSrc, vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq2_xxs);
  }
  return pNrow * vNblock * sizeof(block_iq2_xxs);
}

size_t quantize_iq2_xs(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                       const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq2XsImpl(pSrc, vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq2_xs);
  }
  return pNrow * vNblock * sizeof(block_iq2_xs);
}

//
// ============================================= 3-bit using D4 lattice
//

typedef struct {
  uint32_t *vGrid;
  int *vMap;
  uint16_t *vNeighbours;
} iq3_entry_t;

static iq3_entry_t iq3_data[2] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int fIq3DataIndex(int pGridSize) {
  (void)pGridSize;
  GGML_ASSERT(pGridSize == 256 || pGridSize == 512);
  return pGridSize == 256 ? 0 : 1;
}

static int fIq3CompareFunc(const void *pLeft, const void *pRight) {
  const int *vL = (const int *)pLeft;
  const int *vR = (const int *)pRight;
  return vL[0] < vR[0] ? -1 : vL[0] > vR[0] ? 1 : vL[1] < vR[1] ? -1 : vL[1] > vR[1] ? 1 : 0;
}

void iq3xs_init_impl(int pGridSize) {
  const int cGindex = fIq3DataIndex(pGridSize);
  if (iq3_data[cGindex].vGrid) {
    return;
  }
  static const uint16_t aKgrid256[256] = {
      0,    2,    4,    9,    11,   15,   16,   18,   25,   34,   59,   61,   65,   67,   72,   74,   81,   85,   88,
      90,   97,   108,  120,  128,  130,  132,  137,  144,  146,  153,  155,  159,  169,  175,  189,  193,  199,  200,
      202,  213,  248,  267,  287,  292,  303,  315,  317,  321,  327,  346,  362,  413,  436,  456,  460,  462,  483,
      497,  513,  515,  520,  522,  529,  531,  536,  538,  540,  551,  552,  576,  578,  585,  592,  594,  641,  643,
      648,  650,  657,  664,  698,  704,  706,  720,  729,  742,  758,  769,  773,  808,  848,  852,  870,  889,  901,
      978,  992,  1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105, 1112, 1139,
      1143, 1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228, 1272, 1276, 1309, 1323, 1347,
      1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553, 1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625,
      1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722, 1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950,
      1971, 2011, 2032, 2052, 2063, 2077, 2079, 2091, 2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345,
      2356, 2389, 2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746, 2754,
      2795, 2824, 2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153, 3185, 3215, 3252, 3288,
      3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610, 3626, 3670, 3680, 3722, 3749, 3754, 3776,
      3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
  };
  static const uint16_t aKgrid512[512] = {
      0,    1,    2,    5,    7,    8,    9,    10,   12,   14,   16,   17,   21,   27,   32,   34,   37,   39,   41,
      43,   48,   50,   57,   60,   63,   64,   65,   66,   68,   72,   73,   77,   80,   83,   87,   89,   93,   100,
      113,  117,  122,  128,  129,  133,  135,  136,  139,  142,  145,  149,  152,  156,  162,  165,  167,  169,  171,
      184,  187,  195,  201,  205,  208,  210,  217,  219,  222,  228,  232,  234,  247,  249,  253,  256,  267,  271,
      273,  276,  282,  288,  291,  297,  312,  322,  324,  336,  338,  342,  347,  353,  357,  359,  374,  379,  390,
      393,  395,  409,  426,  441,  448,  450,  452,  464,  466,  470,  475,  488,  492,  512,  513,  514,  516,  520,
      521,  523,  525,  527,  528,  530,  537,  540,  542,  556,  558,  561,  570,  576,  577,  579,  582,  584,  588,
      593,  600,  603,  609,  616,  618,  632,  638,  640,  650,  653,  655,  656,  660,  666,  672,  675,  685,  688,
      698,  705,  708,  711,  712,  715,  721,  727,  728,  732,  737,  754,  760,  771,  773,  778,  780,  793,  795,
      802,  806,  808,  812,  833,  840,  843,  849,  856,  858,  873,  912,  916,  919,  932,  934,  961,  963,  968,
      970,  977,  989,  993,  1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047, 1048,
      1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103, 1106, 1109, 1113, 1116,
      1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199, 1209, 1212, 1216, 1218, 1221, 1225, 1234,
      1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296, 1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375,
      1376, 1387, 1400, 1408, 1410, 1415, 1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549,
      1551, 1554, 1561, 1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
      1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761, 1763, 1767, 1773,
      1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877, 1882, 1892, 1902, 1915, 1934, 1953,
      1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068, 2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123,
      2130, 2136, 2141, 2147, 2153, 2157, 2177, 2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251,
      2256, 2258, 2265, 2269, 2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509,
      2520, 2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634, 2641, 2650,
      2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805, 2811, 2817, 2820, 2832, 2842,
      2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083, 3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188,
      3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276, 3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488,
      3529, 3531, 3584, 3588, 3591, 3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720,
      3726, 3729, 3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
  };

  const int cKmapSize = 4096;
  const int cNwant = pGridSize == 256 ? 2 : 3;
  const uint16_t *vKgrid = pGridSize == 256 ? aKgrid256 : aKgrid512;
  uint32_t *vKgridQ3xs;
  int *vKmapQ3xs;
  uint16_t *vKneighborsQ3xs;

  // printf("================================================================= %s(grid_size = %d)\n", __func__,
  // grid_size);
  uint32_t *vTheGrid = (uint32_t *)malloc(pGridSize * sizeof(uint32_t));
  for (int vK = 0; vK < pGridSize; ++vK) {
    int8_t *vPos = (int8_t *)(vTheGrid + vK);
    for (int vI = 0; vI < 4; ++vI) {
      int vL = (vKgrid[vK] >> 3 * vI) & 0x7;
      vPos[vI] = 2 * vL + 1;
    }
  }
  vKgridQ3xs = vTheGrid;
  iq3_data[cGindex].vGrid = vTheGrid;
  vKmapQ3xs = (int *)malloc(cKmapSize * sizeof(int));
  iq3_data[cGindex].vMap = vKmapQ3xs;
  for (int vI = 0; vI < cKmapSize; ++vI)
    vKmapQ3xs[vI] = -1;
  uint32_t vAux32;
  uint8_t *vAux8 = (uint8_t *)&vAux32;
  for (int vI = 0; vI < pGridSize; ++vI) {
    vAux32 = vKgridQ3xs[vI];
    uint16_t vIndex = 0;
    for (int vK = 0; vK < 4; ++vK) {
      uint16_t vQ = (vAux8[vK] - 1) / 2;
      vIndex |= (vQ << 3 * vK);
    }
    vKmapQ3xs[vIndex] = vI;
  }
  // See explanation of parallelism in iq2xs_init_impl
  int *vNPerI = (int *)malloc(cKmapSize * sizeof(int));
  GGML_ASSERT(vNPerI);
  int vNumNeighbors = 0, vNumNotInMap = 0;
#ifdef GGML_USE_OPENMP
#pragma omp parallel reduction(+ : vNumNeighbors, vNumNotInMap)
#endif
  {
    int *dist2 = (int *)malloc(2 * pGridSize * sizeof(int));
    GGML_ASSERT(dist2);
    int8_t pos[4];
    int i;
#ifdef GGML_USE_OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (i = 0; i < cKmapSize; ++i) {
      if (vKmapQ3xs[i] >= 0) {
        vNPerI[i] = 0;
        continue;
      }
      ++vNumNotInMap;
      for (int k = 0; k < 4; ++k) {
        int l = (i >> 3 * k) & 0x7;
        pos[k] = 2 * l + 1;
      }
      for (int j = 0; j < pGridSize; ++j) {
        const int8_t *pg = (const int8_t *)(vKgridQ3xs + j);
        int d2 = 0;
        for (int k = 0; k < 4; ++k)
          d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
        dist2[2 * j + 0] = d2;
        dist2[2 * j + 1] = j;
      }
      qsort(dist2, pGridSize, 2 * sizeof(int), fIq3CompareFunc);
      int n = 0;
      int d2 = dist2[0];
      int nhave = 1;
      for (int j = 0; j < pGridSize; ++j) {
        if (dist2[2 * j] > d2) {
          if (nhave == cNwant)
            break;
          d2 = dist2[2 * j];
          ++nhave;
        }
        ++n;
      }
      vNPerI[i] = n;
      vNumNeighbors += n;
    }
    free(dist2);
  }
  // printf("%s: %d neighbours in total\n", __func__, num_neighbors);
  vKneighborsQ3xs = (uint16_t *)malloc((vNumNeighbors + vNumNotInMap) * sizeof(uint16_t));
  iq3_data[cGindex].vNeighbours = vKneighborsQ3xs;

  int *vOffsets = (int *)malloc(cKmapSize * sizeof(int));
  GGML_ASSERT(vOffsets);
  int vCounter = 0;
  for (int vI = 0; vI < cKmapSize; ++vI) {
    if (vKmapQ3xs[vI] >= 0) {
      vOffsets[vI] = -1;
      continue;
    }
    vOffsets[vI] = vCounter;
    vCounter += 1 + vNPerI[vI];
  }

#ifdef GGML_USE_OPENMP
#pragma omp parallel
#endif
  {
    int *dist2 = (int *)malloc(2 * pGridSize * sizeof(int));
    GGML_ASSERT(dist2);
    int8_t pos[4];
    int i;
#ifdef GGML_USE_OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (i = 0; i < cKmapSize; ++i) {
      if (vKmapQ3xs[i] >= 0)
        continue;
      for (int k = 0; k < 4; ++k) {
        int l = (i >> 3 * k) & 0x7;
        pos[k] = 2 * l + 1;
      }
      for (int j = 0; j < pGridSize; ++j) {
        const int8_t *pg = (const int8_t *)(vKgridQ3xs + j);
        int d2 = 0;
        for (int k = 0; k < 4; ++k)
          d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
        dist2[2 * j + 0] = d2;
        dist2[2 * j + 1] = j;
      }
      qsort(dist2, pGridSize, 2 * sizeof(int), fIq3CompareFunc);
      int local_counter = vOffsets[i];
      vKmapQ3xs[i] = -(local_counter + 1);
      int d2 = dist2[0];
      uint16_t *start = &vKneighborsQ3xs[local_counter++];
      int n = 0, nhave = 1;
      for (int j = 0; j < pGridSize; ++j) {
        if (dist2[2 * j] > d2) {
          if (nhave == cNwant)
            break;
          d2 = dist2[2 * j];
          ++nhave;
        }
        vKneighborsQ3xs[local_counter++] = dist2[2 * j + 1];
        ++n;
      }
      *start = n;
    }
    free(dist2);
  }
  free(vOffsets);
  free(vNPerI);
}

void iq3xs_free_impl(int pGridSize) {
  GGML_ASSERT(pGridSize == 256 || pGridSize == 512);
  const int cGindex = fIq3DataIndex(pGridSize);
  if (iq3_data[cGindex].vGrid) {
    free(iq3_data[cGindex].vGrid);
    iq3_data[cGindex].vGrid = NULL;
    free(iq3_data[cGindex].vMap);
    iq3_data[cGindex].vMap = NULL;
    free(iq3_data[cGindex].vNeighbours);
    iq3_data[cGindex].vNeighbours = NULL;
  }
}

static int fIq3FindBestNeighbour(const uint16_t *GGML_RESTRICT pNeighbours, const uint32_t *GGML_RESTRICT pGrid,
                                 const float *GGML_RESTRICT pXval, const float *GGML_RESTRICT pWeight, float pScale,
                                 int8_t *GGML_RESTRICT pL) {
  int vNumNeighbors = pNeighbours[0];
  GGML_ASSERT(vNumNeighbors > 0);
  float vBestD2 = FLT_MAX;
  int vGridIndex = -1;
  for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
    const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
    float vD2 = 0;
    for (int vI = 0; vI < 4; ++vI) {
      float vQ = vPg[vI];
      float vDiff = pScale * vQ - pXval[vI];
      vD2 += pWeight[vI] * vDiff * vDiff;
    }
    if (vD2 < vBestD2) {
      vBestD2 = vD2;
      vGridIndex = pNeighbours[vJ];
    }
  }
  GGML_ASSERT(vGridIndex >= 0);
  const int8_t *vPg = (const int8_t *)(pGrid + vGridIndex);
  for (int vI = 0; vI < 4; ++vI)
    pL[vI] = (vPg[vI] - 1) / 2;
  return vGridIndex;
}

static void fQuantizeRowIq3XxsImpl(int pGridSize, const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                   const float *GGML_RESTRICT pQuantWeights) {

  const int cGindex = fIq3DataIndex(pGridSize);

  const uint32_t *vKgridQ3xs = iq3_data[cGindex].vGrid;
  const int *vKmapQ3xs = iq3_data[cGindex].vMap;
  const uint16_t *vKneighborsQ3xs = iq3_data[cGindex].vNeighbours;

  // GGML_ASSERT(quant_weights   && "missing quantization weights");
  GGML_ASSERT(vKgridQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKmapQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  const int cKMaxQ = 8;

  const int64_t cNbl = pN / QK_K;

  ggml_fp16_t *vDh;
  uint8_t *vQs;
  int vBlockSize;
  if (pGridSize == 256) {
    block_iq3_xxs *vY = pVy;
    vDh = &vY->d;
    vQs = vY->qs;
    vBlockSize = sizeof(block_iq3_xxs);
  } else {
    block_iq3_s *vY = pVy;
    vDh = &vY->d;
    vQs = vY->qs;
    vBlockSize = sizeof(block_iq3_s);
  }
  int vQuantSize = vBlockSize - sizeof(ggml_fp16_t);

  float vScales[QK_K / 32];
  float aWeight[32];
  float vXval[32];
  int8_t L[32];
  int8_t aLaux[32];
  float aWaux[32];
  bool vIsOnGrid[8];
  bool vIsOnGridAux[8];
  uint8_t aBlockSigns[8];
  uint8_t aQ3[3 * (QK_K / 8) + QK_K / 32];
  uint32_t *vScalesAndSigns = (uint32_t *)(aQ3 + QK_K / 4);
  uint8_t *vQh = aQ3 + 3 * (QK_K / 8);

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    vDh[0] = GGML_FP32_TO_FP16(0.f);
    memset(aQ3, 0, 3 * QK_K / 8 + QK_K / 32);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const float *vXb = vXbl + 32 * vIb;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vIbl + 32 * vIb;
        for (int vI = 0; vI < 32; ++vI)
          aWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      } else {
        for (int vI = 0; vI < 32; ++vI)
          aWeight[vI] = vXb[vI] * vXb[vI];
      }
      for (int vI = 0; vI < 32; ++vI)
        aWaux[vI] = sqrtf(aWeight[vI]);
      for (int vK = 0; vK < 4; ++vK) {
        int vNflip = 0;
        uint8_t vS = 0;
        for (int vI = 0; vI < 8; ++vI) {
          if (vXb[8 * vK + vI] >= 0)
            vXval[8 * vK + vI] = vXb[8 * vK + vI];
          else {
            vXval[8 * vK + vI] = -vXb[8 * vK + vI];
            ++vNflip;
            vS |= (1 << vI);
          }
        }
        if (vNflip % 2) {
          int vImin = 0;
          float vMin = aWeight[8 * vK + vImin] * vXb[8 * vK + vImin] * vXb[8 * vK + vImin];
          for (int vI = 1; vI < 8; ++vI) {
            float vAx = aWeight[8 * vK + vI] * vXb[8 * vK + vI] * vXb[8 * vK + vI];
            if (vAx < vMin) {
              vMin = vAx;
              vImin = vI;
            }
          }
          vXval[8 * vK + vImin] = -vXval[8 * vK + vImin];
          vS ^= (1 << vImin);
        }
        aBlockSigns[vK] = vS & 127;
      }
      float vMax = vXval[0];
      for (int vI = 1; vI < 32; ++vI)
        vMax = MAX(vMax, vXval[vI]);
      memset(L, 0, 32);
      if (vMax < GROUP_MAX_EPS_IQ3_XXS) {
        vScales[vIb] = 0;
        continue;
      }
      float vBest = 0;
      float vScale = vMax / (2 * cKMaxQ - 1);
      for (int vK = 0; vK < 8; ++vK)
        vIsOnGrid[vK] = true;
      for (int vIs = -15; vIs <= 15; ++vIs) {
        float vId = (2 * cKMaxQ - 1 + vIs * 0.2f) / vMax;
        float vThisScale = 1 / vId;
        for (int vK = 0; vK < 8; ++vK) {
          for (int vI = 0; vI < 4; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[4 * vK + vI] - 1));
            aLaux[4 * vK + vI] = MAX(0, MIN(cKMaxQ - 1, l));
          }
          uint16_t vU = 0;
          for (int vI = 0; vI < 4; ++vI)
            vU |= (aLaux[4 * vK + vI] << 3 * vI);
          int vGridIndex = vKmapQ3xs[vU];
          vIsOnGridAux[vK] = true;
          if (vGridIndex < 0) {
            vIsOnGridAux[vK] = false;
            const uint16_t *vNeighbours = vKneighborsQ3xs - vKmapQ3xs[vU] - 1;
            vGridIndex = fIq3FindBestNeighbour(vNeighbours, vKgridQ3xs, vXval + 4 * vK, aWaux + 4 * vK, vThisScale,
                                               aLaux + 4 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 32; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * aLaux[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
          vScale = vSumqx / vSumq2;
          vBest = vScale * vSumqx;
          for (int vI = 0; vI < 32; ++vI)
            L[vI] = aLaux[vI];
          for (int vK = 0; vK < 8; ++vK)
            vIsOnGrid[vK] = vIsOnGridAux[vK];
        }
      }
      int vNNotOngrid = 0;
      for (int vK = 0; vK < 8; ++vK)
        if (!vIsOnGrid[vK])
          ++vNNotOngrid;
      if (vNNotOngrid > 0 && vScale > 0) {
        float vId = 1 / vScale;
        for (int vK = 0; vK < 8; ++vK) {
          if (vIsOnGrid[vK])
            continue;
          uint16_t vU = 0;
          for (int vI = 0; vI < 4; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[4 * vK + vI] - 1));
            l = MAX(0, MIN(cKMaxQ - 1, l));
            vU |= (l << 3 * vI);
          }
          int vGridIndex = vKmapQ3xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ3xs - vKmapQ3xs[vU] - 1;
            vGridIndex =
                fIq3FindBestNeighbour(vNeighbours, vKgridQ3xs, vXval + 4 * vK, aWaux + 4 * vK, vScale, L + 4 * vK);
          }
          const int8_t *vPg = (const int8_t *)(vKgridQ3xs + vGridIndex);
          for (int vI = 0; vI < 4; ++vI)
            L[4 * vK + vI] = (vPg[vI] - 1) / 2;
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 32; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * L[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      if (vScale < 0) {
        // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the
        // scale) and correspondingly flip quant signs.
        vScale = -vScale;
        for (int vK = 0; vK < 4; ++vK)
          aBlockSigns[vK] = (~aBlockSigns[vK]) & 127;
      }
      for (int vK = 0; vK < 8; ++vK) {
        uint16_t vU = 0;
        for (int vI = 0; vI < 4; ++vI)
          vU |= (L[4 * vK + vI] << 3 * vI);
        int vGridIndex = vKmapQ3xs[vU];
        if (vGridIndex < 0) {
          printf("Oops: found point %u not on grid:", vU);
          for (int vI = 0; vI < 4; ++vI)
            printf(" %d", L[4 * vK + vI]);
          printf("\n");
          GGML_ABORT("fatal error");
        }
        if (pGridSize == 256) {
          aQ3[8 * vIb + vK] = vGridIndex;
        } else {
          aQ3[8 * vIb + vK] = vGridIndex & 255;
          vQh[vIb] |= ((vGridIndex >> 8) << vK);
        }
      }
      vScalesAndSigns[vIb] = aBlockSigns[0] | (aBlockSigns[1] << 7) | (aBlockSigns[2] << 14) | (aBlockSigns[3] << 21);
      GGML_ASSERT(vScale >= 0);
      vScales[vIb] = vScale;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      memset(vQs, 0, vQuantSize);
      vDh += vBlockSize / sizeof(ggml_fp16_t);
      vQs += vBlockSize;
      continue;
    }

    float vD = vMaxScale / 31;
    vDh[0] = GGML_FP32_TO_FP16(vD * 1.0125f); // small improvement via this fudge factor
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      int l = fNearestInt(0.5f * (vId * vScales[vIb] - 1));
      l = MAX(0, MIN(15, l));
      vScalesAndSigns[vIb] |= ((uint32_t)l << 28);
    }
    memcpy(vQs, aQ3, vQuantSize);

    vDh += vBlockSize / sizeof(ggml_fp16_t);
    vQs += vBlockSize;
  }
}

size_t quantize_iq3_xxs(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                        const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq3XxsImpl(256, pSrc, vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq3_xxs);
  }
  return pNrow * vNblock * sizeof(block_iq3_xxs);
}

void quantize_row_iq3_xxs_ref(const float *GGML_RESTRICT pX, block_iq3_xxs *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  fQuantizeRowIq3XxsImpl(256, pX, pY, pK, NULL);
}

static void fQuantizeRowIq3SImpl(int pBlockSize, const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int pN,
                                 const float *GGML_RESTRICT pQuantWeights, float *pScales, float *pWeight, float *pXval,
                                 int8_t *pL, int8_t *pLaux, float *pWaux, bool *pIsOnGrid, bool *pIsOnGridAux,
                                 uint8_t *pBlockSigns) {

  const int cGindex = fIq3DataIndex(512);

  const uint32_t *vKgridQ3xs = iq3_data[cGindex].vGrid;
  const int *vKmapQ3xs = iq3_data[cGindex].vMap;
  const uint16_t *vKneighborsQ3xs = iq3_data[cGindex].vNeighbours;

  // GGML_ASSERT(quant_weights   && "missing quantization weights");
  GGML_ASSERT(vKgridQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKmapQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ3xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  const int cKMaxQ = 8;

  const int64_t cNbl = pN / QK_K;

  block_iq3_s *vY = pVy;

  const int cBs4 = pBlockSize / 4;
  const int cBs8 = pBlockSize / 8;

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    memset(&vY[vIbl], 0, sizeof(block_iq3_s));
    vY[vIbl].d = GGML_FP32_TO_FP16(0.f);

    uint8_t *vQs = vY[vIbl].qs;
    uint8_t *vQh = vY[vIbl].qh;
    uint8_t *vSigns = vY[vIbl].signs;

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / pBlockSize; ++vIb) {
      const float *vXb = vXbl + pBlockSize * vIb;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vIbl + pBlockSize * vIb;
        for (int vI = 0; vI < pBlockSize; ++vI)
          pWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      } else {
        for (int vI = 0; vI < pBlockSize; ++vI)
          pWeight[vI] = vXb[vI] * vXb[vI];
      }
      for (int vI = 0; vI < pBlockSize; ++vI)
        pWaux[vI] = sqrtf(pWeight[vI]);
      for (int vK = 0; vK < cBs8; ++vK) {
        uint8_t vS = 0;
        for (int vI = 0; vI < 8; ++vI) {
          if (vXb[8 * vK + vI] >= 0)
            pXval[8 * vK + vI] = vXb[8 * vK + vI];
          else {
            pXval[8 * vK + vI] = -vXb[8 * vK + vI];
            vS |= (1 << vI);
          }
        }
        pBlockSigns[vK] = vS;
      }
      float vMax = pXval[0];
      for (int vI = 1; vI < pBlockSize; ++vI)
        vMax = MAX(vMax, pXval[vI]);
      memset(pL, 0, pBlockSize);
      if (!vMax) {
        pScales[vIb] = 0;
        continue;
      }
      float vBest = 0;
      float vScale = vMax / (2 * cKMaxQ - 1);
      for (int vK = 0; vK < cBs4; ++vK)
        pIsOnGrid[vK] = false;
      for (int vIs = -9; vIs <= 9; ++vIs) {
        float vId = (2 * cKMaxQ - 1 + vIs * 0.2f) / vMax;
        float vThisScale = 1 / vId;
        for (int vK = 0; vK < cBs4; ++vK) {
          for (int vI = 0; vI < 4; ++vI) {
            int vL = fNearestInt(0.5f * (vId * pXval[4 * vK + vI] - 1));
            pLaux[4 * vK + vI] = MAX(0, MIN(cKMaxQ - 1, vL));
          }
          uint16_t vU = 0;
          for (int vI = 0; vI < 4; ++vI)
            vU |= (pLaux[4 * vK + vI] << 3 * vI);
          int vGridIndex = vKmapQ3xs[vU];
          pIsOnGridAux[vK] = true;
          if (vGridIndex < 0) {
            pIsOnGridAux[vK] = false;
            const uint16_t *vNeighbours = vKneighborsQ3xs - vKmapQ3xs[vU] - 1;
            vGridIndex = fIq3FindBestNeighbour(vNeighbours, vKgridQ3xs, pXval + 4 * vK, pWaux + 4 * vK, vThisScale,
                                               pLaux + 4 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < pBlockSize; ++vI) {
          float vW = pWeight[vI];
          float vQ = 2 * pLaux[vI] + 1;
          vSumqx += vW * pXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
          vScale = vSumqx / vSumq2;
          vBest = vScale * vSumqx;
          for (int vI = 0; vI < pBlockSize; ++vI)
            pL[vI] = pLaux[vI];
          for (int vK = 0; vK < cBs4; ++vK)
            pIsOnGrid[vK] = pIsOnGridAux[vK];
        }
      }
      int vNNotOngrid = 0;
      for (int vK = 0; vK < cBs4; ++vK)
        if (!pIsOnGrid[vK])
          ++vNNotOngrid;
      if (vNNotOngrid > 0 && vScale > 0) {
        float vId = 1 / vScale;
        for (int vK = 0; vK < cBs4; ++vK) {
          // if (is_on_grid[k]) continue;
          uint16_t vU = 0;
          for (int vI = 0; vI < 4; ++vI) {
            int vL = fNearestInt(0.5f * (vId * pXval[4 * vK + vI] - 1));
            vL = MAX(0, MIN(cKMaxQ - 1, vL));
            vU |= (vL << 3 * vI);
          }
          int vGridIndex = vKmapQ3xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ3xs - vKmapQ3xs[vU] - 1;
            vGridIndex =
                fIq3FindBestNeighbour(vNeighbours, vKgridQ3xs, pXval + 4 * vK, pWaux + 4 * vK, vScale, pL + 4 * vK);
          }
          const int8_t *vPg = (const int8_t *)(vKgridQ3xs + vGridIndex);
          for (int vI = 0; vI < 4; ++vI)
            pL[4 * vK + vI] = (vPg[vI] - 1) / 2;
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < pBlockSize; ++vI) {
          float vW = pWeight[vI];
          float vQ = 2 * pL[vI] + 1;
          vSumqx += vW * pXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      if (vScale < 0) {
        // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the
        // scale) and correspondingly flip quant signs.
        vScale = -vScale;
        for (int vK = 0; vK < cBs8; ++vK)
          pBlockSigns[vK] = ~pBlockSigns[vK];
      }
      for (int vK = 0; vK < cBs4; ++vK) {
        uint16_t vU = 0;
        for (int vI = 0; vI < 4; ++vI)
          vU |= (pL[4 * vK + vI] << 3 * vI);
        int vGridIndex = vKmapQ3xs[vU];
        if (vGridIndex < 0) {
          printf("Oops: found point %u not on grid:", vU);
          for (int vI = 0; vI < 4; ++vI)
            printf(" %d", pL[4 * vK + vI]);
          printf("\n");
          GGML_ABORT("fatal error");
        }
        vQs[vK] = vGridIndex & 255;
        vQh[(vIb * cBs4 + vK) / 8] |= ((vGridIndex >> 8) << ((vIb * cBs4 + vK) % 8));
      }
      vQs += cBs4;
      for (int vK = 0; vK < cBs8; ++vK)
        vSigns[vK] = pBlockSigns[vK];
      vSigns += cBs8;
      GGML_ASSERT(vScale >= 0);
      pScales[vIb] = vScale;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      continue;
    }

    float vD = vMaxScale / 31;
    vY[vIbl].d = GGML_FP32_TO_FP16(vD * 1.033f);
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / pBlockSize; vIb += 2) {
      int vL1 = fNearestInt(0.5f * (vId * pScales[vIb + 0] - 1));
      vL1 = MAX(0, MIN(15, vL1));
      int vL2 = fNearestInt(0.5f * (vId * pScales[vIb + 1] - 1));
      vL2 = MAX(0, MIN(15, vL2));
      vY[vIbl].scales[vIb / 2] = vL1 | (vL2 << 4);
    }
  }
}

#define IQ3S_BLOCK_SIZE 32
size_t quantize_iq3_s(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  float vScales[QK_K / IQ3S_BLOCK_SIZE];
  float aWeight[IQ3S_BLOCK_SIZE];
  float vXval[IQ3S_BLOCK_SIZE];
  int8_t vL[IQ3S_BLOCK_SIZE];
  int8_t aLaux[IQ3S_BLOCK_SIZE];
  float aWaux[IQ3S_BLOCK_SIZE];
  bool vIsOnGrid[IQ3S_BLOCK_SIZE / 4];
  bool vIsOnGridAux[IQ3S_BLOCK_SIZE / 4];
  uint8_t aBlockSigns[IQ3S_BLOCK_SIZE / 8];
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq3SImpl(IQ3S_BLOCK_SIZE, pSrc, vQrow, pNPerRow, pQuantWeights, vScales, aWeight, vXval, vL, aLaux,
                         aWaux, vIsOnGrid, vIsOnGridAux, aBlockSigns);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq3_s);
  }
  return pNrow * vNblock * sizeof(block_iq3_s);
}

void quantize_row_iq3_s_ref(const float *GGML_RESTRICT pX, block_iq3_s *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  quantize_iq3_s(pX, pY, 1, pK, NULL);
}

// =================================== 1.5 bpw ===================================================

static int fIq1FindBestNeighbour(const uint16_t *GGML_RESTRICT pNeighbours, const uint64_t *GGML_RESTRICT pGrid,
                                 const float *GGML_RESTRICT pXval, const float *GGML_RESTRICT pWeight, float *pScale,
                                 int8_t *GGML_RESTRICT pL, int pNgrid) {
  int vNumNeighbors = pNeighbours[0];
  GGML_ASSERT(vNumNeighbors > 0);
  float vBestScore = -FLT_MAX;
  int vGridIndex = -1;
  for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
    const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
    float vSumqx = 0, vSumq2 = 0;
    for (int vI = 0; vI < 8; ++vI) {
      float vQ = (vPg[vI] - 3) / 2;
      float vW = pWeight[vI];
      vSumqx += vW * vQ * pXval[vI];
      vSumq2 += vW * vQ * vQ;
    }
    if (vSumqx > 0 && vSumq2 > 0 && vSumqx * vSumqx > vBestScore * vSumq2) {
      *pScale = vSumqx / vSumq2;
      vBestScore = *pScale * vSumqx;
      vGridIndex = pNeighbours[vJ];
    }
  }
  if (vGridIndex < 0) {
    for (int vI = 0; vI < pNgrid; ++vI) {
      const int8_t *vGridI = (const int8_t *)(pGrid + vI);
      float vSumqx = 0, vSumq2 = 0;
      for (int vJ = 0; vJ < 8; ++vJ) {
        float vW = pWeight[vJ];
        float vQ = (vGridI[vJ] - 3) / 2;
        vSumqx += vW * vQ * pXval[vJ];
        vSumq2 += vW * vQ * vQ;
      }
      if (vSumqx > 0 && vSumq2 > 0 && vSumqx * vSumqx > vBestScore * vSumq2) {
        *pScale = vSumqx / vSumq2;
        vBestScore = *pScale * vSumqx;
        vGridIndex = vI;
      }
    }
  }
  if (vGridIndex < 0) {
    printf("Oops, did not find grid point\n");
    printf("Have %d neighbours\n", vNumNeighbors);
    for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
      const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
      float vSumqx = 0, vSumq2 = 0;
      for (int vI = 0; vI < 8; ++vI) {
        float vQ = (vPg[vI] - 3) / 2;
        float vW = pWeight[vI];
        vSumqx += vW * vQ * pXval[vI];
        vSumq2 += vW * vQ * vQ;
      }
      printf("    neighbour %d: sumqx = %g sumq2 = %g\n", vJ, (double)vSumqx, (double)vSumq2);
    }
  }
  GGML_ASSERT(vGridIndex >= 0);
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  *pScale *= 1.05f; // This is a fudge factor. Don't ask me why it improves the result.
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const int8_t *vPg = (const int8_t *)(pGrid + vGridIndex);
  for (int vI = 0; vI < 8; ++vI)
    pL[vI] = (vPg[vI] - 1) / 2;
  return vGridIndex;
}

static int fIq1FindBestNeighbour2(const uint16_t *GGML_RESTRICT pNeighbours, const uint64_t *GGML_RESTRICT pGrid,
                                  const float *GGML_RESTRICT pXval, const float *GGML_RESTRICT pWeight, float pScale,
                                  const float *GGML_RESTRICT pXg, int8_t *GGML_RESTRICT pL, int pNgrid) {
  int vNumNeighbors = pNeighbours[0];
  GGML_ASSERT(vNumNeighbors > 0);
  float vBestScore = FLT_MAX;
  int vGridIndex = -1;
  for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
    const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
    float vD2 = 0;
    for (int vI = 0; vI < 8; ++vI) {
      float vQ = pXg[(vPg[vI] - 1) / 2];
      float vW = pWeight[vI];
      float vDiff = pScale * vQ - pXval[vI];
      vD2 += vW * vDiff * vDiff;
    }
    if (vD2 < vBestScore) {
      vBestScore = vD2;
      vGridIndex = pNeighbours[vJ];
    }
  }
  if (vGridIndex < 0) {
    for (int vI = 0; vI < pNgrid; ++vI) {
      const int8_t *vGridI = (const int8_t *)(pGrid + vI);
      float vD2 = 0;
      for (int vJ = 0; vJ < 8; ++vJ) {
        float vW = pWeight[vJ];
        float vQ = pXg[(vGridI[vJ] - 1) / 2];
        float vDiff = pScale * vQ - pXval[vI];
        vD2 += vW * vDiff * vDiff;
      }
      if (vD2 < vBestScore) {
        vBestScore = vD2;
        vGridIndex = vI;
      }
    }
  }
  if (vGridIndex < 0) {
    printf("Oops, did not find grid point\n");
    printf("Have %d neighbours\n", vNumNeighbors);
    for (int vJ = 1; vJ <= vNumNeighbors; ++vJ) {
      const int8_t *vPg = (const int8_t *)(pGrid + pNeighbours[vJ]);
      float vSumqx = 0, vSumq2 = 0;
      for (int vI = 0; vI < 8; ++vI) {
        float vQ = pXg[(vPg[vI] - 1) / 2];
        float vW = pWeight[vI];
        vSumqx += vW * vQ * pXval[vI];
        vSumq2 += vW * vQ * vQ;
      }
      printf("    neighbour %d: sumqx = %g sumq2 = %g\n", vJ, (double)vSumqx, (double)vSumq2);
    }
  }
  GGML_ASSERT(vGridIndex >= 0);
  const int8_t *vPg = (const int8_t *)(pGrid + vGridIndex);
  for (int vI = 0; vI < 8; ++vI)
    pL[vI] = (vPg[vI] - 1) / 2;
  return vGridIndex;
}

static int fIq1SortHelper(const void *pLeft, const void *pRight) {
  const float *vL = pLeft;
  const float *vR = pRight;
  return *vL < *vR ? -1 : *vL > *vR ? 1 : 0;
}

#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
static void fQuantizeRowIq1SImpl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                 const float *GGML_RESTRICT pQuantWeights, float *pScales, float *pWeight, float *pSumx,
                                 float *pSumw, float *pPairs, int8_t *pL, uint16_t *pIndex, int8_t *pShifts) {

  const int cGindex = fIq2DataIndex(GGML_TYPE_IQ1_S);

  const uint64_t *vKgridQ2xs = iq2_data[cGindex].vGrid;
  const int *vKmapQ2xs = iq2_data[cGindex].vMap;
  const uint16_t *vKneighborsQ2xs = iq2_data[cGindex].vNeighbours;

  GGML_ASSERT(pQuantWeights && "missing quantization weights");
  GGML_ASSERT(vKgridQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKmapQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  block_iq1_s *vY = pVy;

  const int64_t cNbl = pN / QK_K;

  const int cBlockSize = IQ1S_BLOCK_SIZE;

  const float aXP[3] = {-1 + IQ1S_DELTA, IQ1S_DELTA, 1 + IQ1S_DELTA};
  const float aXM[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};

  int *vIdx = (int *)(pPairs + 1);

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    vY[vIbl].d = GGML_FP32_TO_FP16(0.f);
    memset(vY[vIbl].qs, 0, QK_K / 8);
    memset(vY[vIbl].qh, 0, QK_K / 16);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / cBlockSize; ++vIb) {
      const float *vXb = vXbl + cBlockSize * vIb;
      const float *vQw = pQuantWeights + QK_K * vIbl + cBlockSize * vIb;
      for (int vI = 0; vI < cBlockSize; ++vI)
        pWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      float vMax = fabsf(vXb[0]);
      for (int vI = 1; vI < cBlockSize; ++vI)
        vMax = MAX(vMax, fabsf(vXb[vI]));
      if (vMax < GROUP_MAX_EPS_IQ1_S) {
        pScales[vIb] = 0;
        pShifts[vIb] = 1;
        memset(pL, 1, cBlockSize);
        continue;
      }
      // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
      // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
      // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
      // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
      // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
      // for each possible and score for each split.
      for (int vJ = 0; vJ < cBlockSize; ++vJ) {
        pPairs[2 * vJ] = vXb[vJ];
        vIdx[2 * vJ] = vJ;
      }
      qsort(pPairs, cBlockSize, 2 * sizeof(float), fIq1SortHelper);
      {
        pSumx[0] = pSumw[0] = 0;
        for (int vJ = 0; vJ < cBlockSize; ++vJ) {
          int vI = vIdx[2 * vJ];
          pSumx[vJ + 1] = pSumx[vJ] + pWeight[vI] * vXb[vI];
          pSumw[vJ + 1] = pSumw[vJ] + pWeight[vI];
        }
      }
      float vBestScore = -FLT_MAX, vScale = vMax;
      int vBesti1 = -1, vBesti2 = -1, vBestShift = 0;
      for (int vI1 = 0; vI1 <= cBlockSize; ++vI1) {
        for (int vI2 = vI1; vI2 <= cBlockSize; ++vI2) {
          float vSumqx = (pSumx[vI1] - pSumx[0]) * aXP[0] + (pSumx[vI2] - pSumx[vI1]) * aXP[1] +
                         (pSumx[cBlockSize] - pSumx[vI2]) * aXP[2];
          float vSumq2 = (pSumw[vI1] - pSumw[0]) * aXP[0] * aXP[0] + (pSumw[vI2] - pSumw[vI1]) * aXP[1] * aXP[1] +
                         (pSumw[cBlockSize] - pSumw[vI2]) * aXP[2] * aXP[2];
          if (vSumq2 > 0 && vSumqx * vSumqx > vBestScore * vSumq2) {
            vScale = vSumqx / vSumq2;
            vBestScore = vScale * vSumqx;
            vBesti1 = vI1;
            vBesti2 = vI2;
            vBestShift = 1;
          }
          vSumqx = (pSumx[vI1] - pSumx[0]) * aXM[0] + (pSumx[vI2] - pSumx[vI1]) * aXM[1] +
                   (pSumx[cBlockSize] - pSumx[vI2]) * aXM[2];
          vSumq2 = (pSumw[vI1] - pSumw[0]) * aXM[0] * aXM[0] + (pSumw[vI2] - pSumw[vI1]) * aXM[1] * aXM[1] +
                   (pSumw[cBlockSize] - pSumw[vI2]) * aXM[2] * aXM[2];
          if (vSumq2 > 0 && vSumqx * vSumqx > vBestScore * vSumq2) {
            vScale = vSumqx / vSumq2;
            vBestScore = vScale * vSumqx;
            vBesti1 = vI1;
            vBesti2 = vI2;
            vBestShift = -1;
          }
        }
      }
      if (vBesti1 < 0 || vBesti2 < 0 || vBestShift == 0) {
        pScales[vIb] = 0;
        pShifts[vIb] = 1;
        memset(pL, 1, cBlockSize);
        continue;
      }
      for (int vJ = 0; vJ < vBesti1; ++vJ)
        pL[vIdx[2 * vJ]] = 0;
      for (int vJ = vBesti1; vJ < vBesti2; ++vJ)
        pL[vIdx[2 * vJ]] = 1;
      for (int vJ = vBesti2; vJ < cBlockSize; ++vJ)
        pL[vIdx[2 * vJ]] = 2;
      if (vScale < 0) {
        for (int vJ = 0; vJ < cBlockSize; ++vJ)
          pL[vJ] = 2 - pL[vJ];
        vScale = -vScale;
        vBestShift = -vBestShift;
      }
      bool vAllOnGrid = true;
      const float *vXx = vBestShift == 1 ? aXP : aXM;
      for (int vK = 0; vK < cBlockSize / 8; ++vK) {
        uint16_t vU = 0;
        for (int vJ = 0; vJ < 8; ++vJ)
          vU |= (pL[8 * vK + vJ] << 2 * vJ);
        int vGridIndex = vKmapQ2xs[vU];
        if (vGridIndex < 0) {
          vAllOnGrid = false;
          const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
          vGridIndex = fIq1FindBestNeighbour2(vNeighbours, vKgridQ2xs, vXb + 8 * vK, pWeight + 8 * vK, vScale, vXx,
                                              pL + 8 * vK, NGRID_IQ1S);
          GGML_ASSERT(vGridIndex >= 0);
        }
        pIndex[vK] = vGridIndex;
      }
      if (!vAllOnGrid) {
        float vSumqx = 0, vSumq2 = 0;
        for (int vK = 0; vK < cBlockSize / 8; ++vK) {
          const int8_t *vPg = (const int8_t *)(vKgridQ2xs + pIndex[vK]);
          for (int vJ = 0; vJ < 8; ++vJ) {
            float vW = pWeight[8 * vK + vJ];
            float vQ = vXx[(vPg[vJ] - 1) / 2];
            vSumqx += vW * vQ * vXb[8 * vK + vJ];
            vSumq2 += vW * vQ * vQ;
          }
        }
        if (vSumqx > 0 && vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      uint16_t vH = 0;
      for (int vK = 0; vK < cBlockSize / 8; ++vK) {
        vY[vIbl].qs[(cBlockSize / 8) * vIb + vK] = pIndex[vK] & 255;
        vH |= (pIndex[vK] >> 8) << 3 * vK;
      }
      vY[vIbl].qh[vIb] = vH;
      GGML_ASSERT(vScale >= 0);
      pScales[vIb] = vScale;
      pShifts[vIb] = vBestShift;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      continue;
    }

    float vD = vMaxScale / 15;
    vY[vIbl].d = GGML_FP32_TO_FP16(vD * 1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / cBlockSize; ++vIb) {
      int vL = fNearestInt(0.5f * (vId * pScales[vIb] - 1));
      vL = MAX(0, MIN(7, vL));
      if (pShifts[vIb] == -1)
        vL |= 8;
      vY[vIbl].qh[vIb] |= (vL << 12);
    }
  }
}

size_t quantize_iq1_s(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  float vScales[QK_K / IQ1S_BLOCK_SIZE];
  float aWeight[IQ1S_BLOCK_SIZE];
  int8_t vL[IQ1S_BLOCK_SIZE];
  float aSumx[IQ1S_BLOCK_SIZE + 1];
  float aSumw[IQ1S_BLOCK_SIZE + 1];
  float aPairs[2 * IQ1S_BLOCK_SIZE];
  uint16_t aIndex[IQ1S_BLOCK_SIZE / 8];
  int8_t aShifts[QK_K / IQ1S_BLOCK_SIZE];
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq1SImpl(pSrc, vQrow, pNPerRow, pQuantWeights, vScales, aWeight, aSumx, aSumw, aPairs, vL, aIndex,
                         aShifts);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq1_s);
  }
  return pNrow * vNblock * sizeof(block_iq1_s);
}

static void fQuantizeRowIq1MImpl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                 const float *GGML_RESTRICT pQuantWeights, float *pScales, float *pWeight,
                                 float *pPairs, int8_t *pL, uint16_t *pIndex, int8_t *pShifts) {

  const int cGindex = fIq2DataIndex(GGML_TYPE_IQ1_M);

  const uint64_t *vKgridQ2xs = iq2_data[cGindex].vGrid;
  const int *vKmapQ2xs = iq2_data[cGindex].vMap;
  const uint16_t *vKneighborsQ2xs = iq2_data[cGindex].vNeighbours;

  // GGML_ASSERT(quant_weights   && "missing quantization weights");
  GGML_ASSERT(vKgridQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKmapQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  block_iq1_m *vY = pVy;

  const int64_t cNbl = pN / QK_K;

  const int cBlockSize = IQ1M_BLOCK_SIZE;

  const float aXP[3] = {-1 + IQ1M_DELTA, IQ1M_DELTA, 1 + IQ1M_DELTA};
  const float aXM[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
  const uint8_t aMasks[4] = {0x00, 0x80, 0x08, 0x88};

  int *vIdx = (int *)(pPairs + 1);

  float aSumqx[4], aSumq2[4];

  iq1m_scale_t vS;
  const float *vXx;

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {
    memset(vY[vIbl].qs, 0, QK_K / 8);
    memset(vY[vIbl].qh, 0, QK_K / 16);
    memset(vY[vIbl].scales, 0, QK_K / 32);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / cBlockSize; ++vIb) {
      const float *vXb = vXbl + cBlockSize * vIb;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vIbl + cBlockSize * vIb;
        for (int vI = 0; vI < cBlockSize; ++vI)
          pWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      } else {
        for (int vI = 0; vI < cBlockSize; ++vI)
          pWeight[vI] = vXb[vI] * vXb[vI];
      }
      float vMax = fabsf(vXb[0]);
      for (int vI = 1; vI < cBlockSize; ++vI)
        vMax = MAX(vMax, fabsf(vXb[vI]));
      if (vMax < GROUP_MAX_EPS_IQ1_M) {
        pScales[vIb] = 0;
        pShifts[vIb] = 0;
        memset(pL, 1, cBlockSize);
        continue;
      }
      // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
      // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
      // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
      // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
      // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
      // for each possible and score for each split.
      for (int vJ = 0; vJ < cBlockSize; ++vJ) {
        pPairs[2 * vJ] = vXb[vJ];
        vIdx[2 * vJ] = vJ;
      }
      qsort(pPairs, cBlockSize, 2 * sizeof(float), fIq1SortHelper);
      float vBestScore = -FLT_MAX, vScale = vMax;
      int vBesti1 = -1, vBesti2 = -1, vBestK = -1;
      // 0: +, +
      // 1: +, -
      // 2: -, +
      // 3: -, -
      for (int vI1 = 0; vI1 <= cBlockSize; ++vI1) {
        for (int vI2 = vI1; vI2 <= cBlockSize; ++vI2) {
          memset(aSumqx, 0, 4 * sizeof(float));
          memset(aSumq2, 0, 4 * sizeof(float));
          for (int vJ = 0; vJ < vI1; ++vJ) {
            int vI = vIdx[2 * vJ];
            if (vI < cBlockSize / 2) {
              aSumqx[0] += pWeight[vI] * aXP[0] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXP[0] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXM[0] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[0] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[0] * aXP[0];
              aSumq2[1] += pWeight[vI] * aXP[0] * aXP[0];
              aSumq2[2] += pWeight[vI] * aXM[0] * aXM[0];
              aSumq2[3] += pWeight[vI] * aXM[0] * aXM[0];
            } else {
              aSumqx[0] += pWeight[vI] * aXP[0] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXP[0] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXM[0] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[0] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[0] * aXP[0];
              aSumq2[2] += pWeight[vI] * aXP[0] * aXP[0];
              aSumq2[1] += pWeight[vI] * aXM[0] * aXM[0];
              aSumq2[3] += pWeight[vI] * aXM[0] * aXM[0];
            }
          }
          for (int vJ = vI1; vJ < vI2; ++vJ) {
            int vI = vIdx[2 * vJ];
            if (vI < cBlockSize / 2) {
              aSumqx[0] += pWeight[vI] * aXP[1] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXP[1] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXM[1] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[1] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[1] * aXP[1];
              aSumq2[1] += pWeight[vI] * aXP[1] * aXP[1];
              aSumq2[2] += pWeight[vI] * aXM[1] * aXM[1];
              aSumq2[3] += pWeight[vI] * aXM[1] * aXM[1];
            } else {
              aSumqx[0] += pWeight[vI] * aXP[1] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXP[1] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXM[1] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[1] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[1] * aXP[1];
              aSumq2[2] += pWeight[vI] * aXP[1] * aXP[1];
              aSumq2[1] += pWeight[vI] * aXM[1] * aXM[1];
              aSumq2[3] += pWeight[vI] * aXM[1] * aXM[1];
            }
          }
          for (int vJ = vI2; vJ < cBlockSize; ++vJ) {
            int vI = vIdx[2 * vJ];
            if (vI < cBlockSize / 2) {
              aSumqx[0] += pWeight[vI] * aXP[2] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXP[2] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXM[2] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[2] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[2] * aXP[2];
              aSumq2[1] += pWeight[vI] * aXP[2] * aXP[2];
              aSumq2[2] += pWeight[vI] * aXM[2] * aXM[2];
              aSumq2[3] += pWeight[vI] * aXM[2] * aXM[2];
            } else {
              aSumqx[0] += pWeight[vI] * aXP[2] * vXb[vI];
              aSumqx[2] += pWeight[vI] * aXP[2] * vXb[vI];
              aSumqx[1] += pWeight[vI] * aXM[2] * vXb[vI];
              aSumqx[3] += pWeight[vI] * aXM[2] * vXb[vI];
              aSumq2[0] += pWeight[vI] * aXP[2] * aXP[2];
              aSumq2[2] += pWeight[vI] * aXP[2] * aXP[2];
              aSumq2[1] += pWeight[vI] * aXM[2] * aXM[2];
              aSumq2[3] += pWeight[vI] * aXM[2] * aXM[2];
            }
          }
          for (int vK = 0; vK < 4; ++vK) {
            if (aSumq2[vK] > 0 && aSumqx[vK] * aSumqx[vK] > vBestScore * aSumq2[vK]) {
              vScale = aSumqx[vK] / aSumq2[vK];
              vBestScore = vScale * aSumqx[vK];
              vBesti1 = vI1;
              vBesti2 = vI2;
              vBestK = vK;
            }
          }
        }
      }
      if (vBesti1 < 0 || vBesti2 < 0 || vBestK < 0) {
        pScales[vIb] = 0;
        pShifts[vIb] = 0;
        memset(pL, 1, cBlockSize);
        continue;
      }
      for (int vJ = 0; vJ < vBesti1; ++vJ)
        pL[vIdx[2 * vJ]] = 0;
      for (int vJ = vBesti1; vJ < vBesti2; ++vJ)
        pL[vIdx[2 * vJ]] = 1;
      for (int vJ = vBesti2; vJ < cBlockSize; ++vJ)
        pL[vIdx[2 * vJ]] = 2;
      if (vScale < 0) {
        for (int vJ = 0; vJ < cBlockSize; ++vJ)
          pL[vJ] = 2 - pL[vJ];
        vScale = -vScale;
        vBestK = vBestK == 0 ? 3 : vBestK == 1 ? 2 : vBestK == 2 ? 1 : 0;
      }
      bool vAllOnGrid = true;
      for (int vK = 0; vK < cBlockSize / 8; ++vK) {
        if (vK == 0)
          vXx = vBestK < 2 ? aXP : aXM;
        else
          vXx = vBestK % 2 == 0 ? aXP : aXM;
        uint16_t vU = 0;
        for (int vJ = 0; vJ < 8; ++vJ)
          vU |= (pL[8 * vK + vJ] << 2 * vJ);
        int vGridIndex = vKmapQ2xs[vU];
        if (vGridIndex < 0) {
          vAllOnGrid = false;
          const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
          vGridIndex = fIq1FindBestNeighbour2(vNeighbours, vKgridQ2xs, vXb + 8 * vK, pWeight + 8 * vK, vScale, vXx,
                                              pL + 8 * vK, NGRID_IQ1S);
          GGML_ASSERT(vGridIndex >= 0);
        }
        pIndex[vK] = vGridIndex;
      }
      if (!vAllOnGrid) {
        float vSumqxF = 0, vSumq2F = 0;
        for (int vK = 0; vK < cBlockSize / 8; ++vK) {
          if (vK == 0)
            vXx = vBestK < 2 ? aXP : aXM;
          else
            vXx = vBestK % 2 == 0 ? aXP : aXM;
          const int8_t *vPg = (const int8_t *)(vKgridQ2xs + pIndex[vK]);
          for (int vJ = 0; vJ < 8; ++vJ) {
            float vW = pWeight[8 * vK + vJ];
            float vQ = vXx[(vPg[vJ] - 1) / 2];
            vSumqxF += vW * vQ * vXb[8 * vK + vJ];
            vSumq2F += vW * vQ * vQ;
          }
        }
        if (vSumqxF > 0 && vSumq2F > 0)
          vScale = vSumqxF / vSumq2F;
      }
      vY[vIbl].qs[2 * vIb + 0] = pIndex[0] & 255;
      vY[vIbl].qs[2 * vIb + 1] = pIndex[1] & 255;
      vY[vIbl].qh[vIb] = (pIndex[0] >> 8) | ((pIndex[1] >> 8) << 4);
      GGML_ASSERT(vScale >= 0);
      pScales[vIb] = vScale;
      pShifts[vIb] = vBestK;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      continue;
    }

    uint16_t *vSc = (uint16_t *)vY[vIbl].scales;
    float vD = vMaxScale / 15;
    float vId = 1 / vD;
    float vSumqxF = 0, vSumq2F = 0;
    for (int vIb = 0; vIb < QK_K / cBlockSize; ++vIb) {
      int vL = fNearestInt(0.5f * (vId * pScales[vIb + 0] - 1));
      vL = MAX(0, MIN(7, vL));
      vSc[vIb / 4] |= (vL << 3 * (vIb % 4));
      vY[vIbl].qh[vIb] |= aMasks[pShifts[vIb]];
      const float *vXb = vXbl + cBlockSize * vIb;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vIbl + cBlockSize * vIb;
        for (int vI = 0; vI < cBlockSize; ++vI)
          pWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      } else {
        for (int vI = 0; vI < cBlockSize; ++vI)
          pWeight[vI] = vXb[vI] * vXb[vI];
      }
      for (int vK = 0; vK < cBlockSize / 8; ++vK) {
        if (vK == 0)
          vXx = pShifts[vIb] < 2 ? aXP : aXM;
        else
          vXx = pShifts[vIb] % 2 == 0 ? aXP : aXM;
        const int8_t *vPg =
            (const int8_t *)(vKgridQ2xs + vY[vIbl].qs[2 * vIb + vK] + ((vY[vIbl].qh[vIb] << (8 - 4 * vK)) & 0x700));
        for (int vJ = 0; vJ < 8; ++vJ) {
          float vW = pWeight[8 * vK + vJ];
          float vQ = vXx[(vPg[vJ] - 1) / 2] * (2 * vL + 1);
          vSumqxF += vW * vQ * vXb[8 * vK + vJ];
          vSumq2F += vW * vQ * vQ;
        }
      }
    }
    if (vSumq2F > 0)
      vD = vSumqxF / vSumq2F;
    vS.f16 = GGML_FP32_TO_FP16(vD * 1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
    vSc[0] |= ((vS.u16 & 0x000f) << 12);
    vSc[1] |= ((vS.u16 & 0x00f0) << 8);
    vSc[2] |= ((vS.u16 & 0x0f00) << 4);
    vSc[3] |= ((vS.u16 & 0xf000) << 0);
  }
}

size_t quantize_iq1_m(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  float vScales[QK_K / IQ1M_BLOCK_SIZE];
  float aWeight[IQ1M_BLOCK_SIZE];
  int8_t vL[IQ1M_BLOCK_SIZE];
  float aPairs[2 * IQ1M_BLOCK_SIZE];
  uint16_t aIndex[IQ1M_BLOCK_SIZE / 8];
  int8_t aShifts[QK_K / IQ1M_BLOCK_SIZE];
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq1MImpl(pSrc, vQrow, pNPerRow, pQuantWeights, vScales, aWeight, aPairs, vL, aIndex, aShifts);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq1_m);
  }
  return pNrow * vNblock * sizeof(block_iq1_m);
}

// ============================ 4-bit non-linear quants

static void fQuantizeRowIq4NlImpl(const int pSuperBlockSize, const int pBlockSize, const float *GGML_RESTRICT pX,
                                  ggml_fp16_t *pDh, uint8_t *pQ4, uint16_t *pScalesH, uint8_t *pScalesL, float *pScales,
                                  float *pWeight, uint8_t *pL, const int8_t *pValues, const float *pQuantWeights,
                                  const int pNtry) {

  float vSigma2 = 0;
  for (int vJ = 0; vJ < pSuperBlockSize; ++vJ)
    vSigma2 += pX[vJ] * pX[vJ];
  vSigma2 *= 2.f / pSuperBlockSize;

  memset(pQ4, 0, pSuperBlockSize / 2);
  pDh[0] = GGML_FP32_TO_FP16(0.f);

  float vMaxScale = 0, vAmaxScale = 0;
  for (int vIb = 0; vIb < pSuperBlockSize / pBlockSize; ++vIb) {
    const float *vXb = pX + vIb * pBlockSize;
    uint8_t *vLb = pL + vIb * pBlockSize;
    if (pQuantWeights) {
      const float *vQw = pQuantWeights + vIb * pBlockSize;
      for (int vJ = 0; vJ < pBlockSize; ++vJ)
        pWeight[vJ] = vQw[vJ] * sqrtf(vSigma2 + vXb[vJ] * vXb[vJ]);
    } else {
      for (int vJ = 0; vJ < pBlockSize; ++vJ)
        pWeight[vJ] = vXb[vJ] * vXb[vJ];
    }
    float vAmax = 0, vMax = 0;
    for (int vJ = 0; vJ < pBlockSize; ++vJ) {
      float vAx = fabsf(vXb[vJ]);
      if (vAx > vAmax) {
        vAmax = vAx;
        vMax = vXb[vJ];
      }
    }
    if (vAmax < GROUP_MAX_EPS) {
      pScales[vIb] = 0;
      continue;
    }
    float vD = pNtry > 0 ? -vMax / pValues[0] : vMax / pValues[0];
    float vId = 1 / vD;
    float vSumqx = 0, vSumq2 = 0;
    for (int vJ = 0; vJ < pBlockSize; ++vJ) {
      float vAl = vId * vXb[vJ];
      int vL = fBestIndexInt8(16, pValues, vAl);
      vLb[vJ] = vL;
      float vQ = pValues[vL];
      float vW = pWeight[vJ];
      vSumqx += vW * vQ * vXb[vJ];
      vSumq2 += vW * vQ * vQ;
    }
    vD = vSumq2 > 0 ? vSumqx / vSumq2 : 0.f;
    float vBest = vD * vSumqx;
    for (int vItry = -pNtry; vItry <= pNtry; ++vItry) {
      vId = (vItry + pValues[0]) / vMax;
      vSumqx = vSumq2 = 0;
      for (int vJ = 0; vJ < pBlockSize; ++vJ) {
        float vAl = vId * vXb[vJ];
        int vL = fBestIndexInt8(16, pValues, vAl);
        float vQ = pValues[vL];
        float vW = pWeight[vJ];
        vSumqx += vW * vQ * vXb[vJ];
        vSumq2 += vW * vQ * vQ;
      }
      if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
        vD = vSumqx / vSumq2;
        vBest = vD * vSumqx;
      }
    }
    pScales[vIb] = vD;
    float vAbsD = fabsf(vD);
    if (vAbsD > vAmaxScale) {
      vAmaxScale = vAbsD;
      vMaxScale = vD;
    }
  }

  if (pSuperBlockSize / pBlockSize > 1) {
    int vNb = pSuperBlockSize / pBlockSize;
    memset(pScalesH, 0, ((vNb + 7) / 8) * sizeof(uint16_t));
    float vD = -vMaxScale / 32;
    pDh[0] = GGML_FP32_TO_FP16(vD);
    float vId = vD ? 1 / vD : 0.f;
    for (int vIb = 0; vIb < pSuperBlockSize / pBlockSize; ++vIb) {
      int vL = fNearestInt(vId * pScales[vIb]);
      vL = MAX(-32, MIN(31, vL));
      float vDl = vD * vL;
      float vIdl = vDl ? 1 / vDl : 0.f;
      uint8_t *vLb = pL + vIb * pBlockSize;
      const float *vXb = pX + vIb * pBlockSize;
      for (int vJ = 0; vJ < pBlockSize; ++vJ) {
        vLb[vJ] = fBestIndexInt8(16, pValues, vIdl * vXb[vJ]);
      }
      vL += 32;
      uint8_t vLL = vL & 0xf;
      uint8_t vLH = vL >> 4;
      if (vIb % 2 == 0)
        pScalesL[vIb / 2] = vLL;
      else
        pScalesL[vIb / 2] |= (vLL << 4);
      pScalesH[vIb / 8] |= (vLH << 2 * (vIb % 8));
    }
  } else {
    pDh[0] = GGML_FP32_TO_FP16(pScales[0]);
    if (pNtry > 0) {
      float vId = pScales[0] ? 1 / pScales[0] : 0;
      for (int vJ = 0; vJ < pSuperBlockSize; ++vJ) {
        pL[vJ] = fBestIndexInt8(16, pValues, vId * pX[vJ]);
      }
    }
  }

  for (int vI = 0; vI < pSuperBlockSize / 32; ++vI) {
    for (int vJ = 0; vJ < 16; ++vJ) {
      pQ4[16 * vI + vJ] = pL[32 * vI + vJ] | (pL[32 * vI + 16 + vJ] << 4);
    }
  }
}

size_t quantize_iq4_nl(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                       const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK4_NL == 0);
  int64_t vNblock = pNPerRow / QK4_NL;
  char *vQrow = (char *)pDst;
  uint8_t vL[QK4_NL];
  float aWeight[QK4_NL];
  uint16_t vUnusedH;
  uint8_t *vUnusedL = NULL;
  float vScale;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    block_iq4_nl *vIq4 = (block_iq4_nl *)vQrow;
    for (int vIbl = 0; vIbl < vNblock; ++vIbl) {
      const float *vQw = pQuantWeights ? pQuantWeights + QK4_NL * vIbl : NULL;
      fQuantizeRowIq4NlImpl(QK4_NL, 32, pSrc + QK4_NL * vIbl, &vIq4[vIbl].d, vIq4[vIbl].qs, &vUnusedH, vUnusedL,
                            &vScale, aWeight, vL, kvalues_iq4nl, vQw, 7);
    }
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq4_nl);
  }
  return pNrow * vNblock * sizeof(block_iq4_nl);
}

// void quantize_row_iq4_nl_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
void quantize_row_iq4_nl_ref(const float *GGML_RESTRICT pX, block_iq4_nl *GGML_RESTRICT pY, int64_t pK) {
  GGML_ASSERT(pK % QK4_NL == 0);
  int64_t vNblock = pK / QK4_NL;
  uint8_t vL[QK4_NL];
  float aWeight[QK4_NL];
  uint16_t vUnusedH;
  uint8_t *vUnusedL = NULL;
  float vScale;
  block_iq4_nl *vIq4 = pY;
  for (int vIbl = 0; vIbl < vNblock; ++vIbl) {
    fQuantizeRowIq4NlImpl(QK4_NL, 32, pX + QK4_NL * vIbl, &vIq4[vIbl].d, vIq4[vIbl].qs, &vUnusedH, vUnusedL, &vScale,
                          aWeight, vL, kvalues_iq4nl, NULL, -1);
  }
}

size_t quantize_iq4_xs(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                       const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  uint8_t vL[QK_K];
  float aWeight[32];
  float vScales[QK_K / 32];
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    block_iq4_xs *vIq4 = (block_iq4_xs *)vQrow;
    for (int vIbl = 0; vIbl < vNblock; ++vIbl) {
      const float *vQw = pQuantWeights ? pQuantWeights + QK_K * vIbl : NULL;
      fQuantizeRowIq4NlImpl(QK_K, 32, pSrc + QK_K * vIbl, &vIq4[vIbl].d, vIq4[vIbl].qs, &vIq4[vIbl].scales_h,
                            vIq4[vIbl].scales_l, vScales, aWeight, vL, kvalues_iq4nl, vQw, 7);
    }
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq4_xs);
  }
  return pNrow * vNblock * sizeof(block_iq4_xs);
}

void quantize_row_iq4_xs_ref(const float *GGML_RESTRICT pX, block_iq4_xs *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  quantize_iq4_xs(pX, pY, 1, pK, NULL);
}

// =============================== 2.5625 bpw

static void fQuantizeRowIq2SImpl(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pN,
                                 const float *GGML_RESTRICT pQuantWeights) {

  const int cGindex = fIq2DataIndex(GGML_TYPE_IQ2_S);

  const uint64_t *vKgridQ2xs = iq2_data[cGindex].vGrid;
  const int *vKmapQ2xs = iq2_data[cGindex].vMap;
  const uint16_t *vKneighborsQ2xs = iq2_data[cGindex].vNeighbours;

  GGML_ASSERT(vKmapQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKgridQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(vKneighborsQ2xs && "forgot to call ggml_quantize_init()?");
  GGML_ASSERT(pN % QK_K == 0);

  const int cKMaxQ = 3;

  const int64_t cNbl = pN / QK_K;

  block_iq2_s *vY = pVy;

  float vScales[QK_K / 16];
  float aWeight[16];
  float vXval[16];
  int8_t L[16];
  int8_t aLaux[16];
  float aWaux[16];
  bool vIsOnGrid[2];
  bool vIsOnGridAux[2];
  uint8_t aBlockSigns[2];

  for (int vIbl = 0; vIbl < cNbl; ++vIbl) {

    memset(&vY[vIbl], 0, sizeof(block_iq2_s));
    vY[vIbl].d = GGML_FP32_TO_FP16(0.f);

    float vMaxScale = 0;

    const float *vXbl = pX + QK_K * vIbl;
    float vSumx2 = 0;
    for (int vI = 0; vI < QK_K; ++vI)
      vSumx2 += vXbl[vI] * vXbl[vI];
    float vSigma2 = 2 * vSumx2 / QK_K;

    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      const float *vXb = vXbl + 16 * vIb;
      if (pQuantWeights) {
        const float *vQw = pQuantWeights + QK_K * vIbl + 16 * vIb;
        for (int vI = 0; vI < 16; ++vI)
          aWeight[vI] = vQw[vI] * sqrtf(vSigma2 + vXb[vI] * vXb[vI]);
      } else {
        for (int vI = 0; vI < 16; ++vI)
          aWeight[vI] = 0.25f * vSigma2 + vXb[vI] * vXb[vI];
      }
      for (int vI = 0; vI < 16; ++vI)
        aWaux[vI] = sqrtf(aWeight[vI]);
      for (int vK = 0; vK < 2; ++vK) {
        uint8_t vS = 0;
        for (int vI = 0; vI < 8; ++vI) {
          if (vXb[8 * vK + vI] >= 0)
            vXval[8 * vK + vI] = vXb[8 * vK + vI];
          else {
            vXval[8 * vK + vI] = -vXb[8 * vK + vI];
            vS |= (1 << vI);
          }
        }
        aBlockSigns[vK] = vS;
      }
      float vMax = vXval[0];
      for (int vI = 1; vI < 16; ++vI)
        vMax = MAX(vMax, vXval[vI]);
      memset(L, 0, 16);
      if (vMax < GROUP_MAX_EPS_IQ2_S) {
        vScales[vIb] = 0;
        continue;
      }
      float vBest = 0;
      float vScale = vMax / (2 * cKMaxQ - 1);
      vIsOnGrid[0] = vIsOnGrid[1] = true;
      for (int vIs = -9; vIs <= 9; ++vIs) {
        float vId = (2 * cKMaxQ - 1 + vIs * 0.1f) / vMax;
        float vThisScale = 1 / vId;
        for (int vK = 0; vK < 2; ++vK) {
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            aLaux[8 * vK + vI] = MAX(0, MIN(cKMaxQ - 1, l));
          }
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI)
            vU |= (aLaux[8 * vK + vI] << 2 * vI);
          int vGridIndex = vKmapQ2xs[vU];
          vIsOnGridAux[vK] = true;
          if (vGridIndex < 0) {
            vIsOnGridAux[vK] = false;
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex = fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vThisScale,
                                               aLaux + 8 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 16; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * aLaux[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0 && vSumqx * vSumqx > vBest * vSumq2) {
          vScale = vSumqx / vSumq2;
          vBest = vScale * vSumqx;
          for (int vI = 0; vI < 16; ++vI)
            L[vI] = aLaux[vI];
          for (int vK = 0; vK < 2; ++vK)
            vIsOnGrid[vK] = vIsOnGridAux[vK];
        }
      }
      int vNNotOngrid = 0;
      for (int vK = 0; vK < 2; ++vK)
        if (!vIsOnGrid[vK])
          ++vNNotOngrid;
      if (vNNotOngrid > 0 && vScale > 0) {
        float vId = 1 / vScale;
        for (int vK = 0; vK < 2; ++vK) {
          if (vIsOnGrid[vK])
            continue;
          uint16_t vU = 0;
          for (int vI = 0; vI < 8; ++vI) {
            int l = fNearestInt(0.5f * (vId * vXval[8 * vK + vI] - 1));
            l = MAX(0, MIN(cKMaxQ - 1, l));
            vU |= (l << 2 * vI);
            L[8 * vK + vI] = l;
          }
          int vGridIndex = vKmapQ2xs[vU];
          if (vGridIndex < 0) {
            const uint16_t *vNeighbours = vKneighborsQ2xs - vKmapQ2xs[vU] - 1;
            vGridIndex =
                fIq2FindBestNeighbour(vNeighbours, vKgridQ2xs, vXval + 8 * vK, aWaux + 8 * vK, vScale, L + 8 * vK);
          }
        }
        float vSumqx = 0, vSumq2 = 0;
        for (int vI = 0; vI < 16; ++vI) {
          float vW = aWeight[vI];
          float vQ = 2 * L[vI] + 1;
          vSumqx += vW * vXval[vI] * vQ;
          vSumq2 += vW * vQ * vQ;
        }
        if (vSumq2 > 0)
          vScale = vSumqx / vSumq2;
      }
      if (vScale < 0) {
        vScale = -vScale;
        for (int vK = 0; vK < 2; ++vK)
          aBlockSigns[vK] = ~aBlockSigns[vK];
      }
      for (int vK = 0; vK < 2; ++vK) {
        uint16_t vU = 0;
        for (int vI = 0; vI < 8; ++vI)
          vU |= (L[8 * vK + vI] << 2 * vI);
        int vGridIndex = vKmapQ2xs[vU];
        if (vGridIndex < 0) {
          printf("Oops: found point %u not on grid:", vU);
          for (int vI = 0; vI < 8; ++vI)
            printf(" %d", L[8 * vK + vI]);
          printf("\n");
          GGML_ABORT("fatal error");
        }
        const int cI8 = 2 * vIb + vK;
        vY[vIbl].qs[cI8] = vGridIndex & 255;
        vY[vIbl].qh[cI8 / 4] |= ((vGridIndex >> 8) << 2 * (cI8 % 4));
        vY[vIbl].qs[QK_K / 8 + cI8] = aBlockSigns[vK];
      }
      GGML_ASSERT(vScale >= 0);
      vScales[vIb] = vScale;
      vMaxScale = MAX(vMaxScale, vScale);
    }

    if (!vMaxScale) {
      continue;
    }

    float vD = vMaxScale / 31;
    vY[vIbl].d = GGML_FP32_TO_FP16(vD * 0.9875f);
    float vId = 1 / vD;
    for (int vIb = 0; vIb < QK_K / 16; ++vIb) {
      int l = fNearestInt(0.5f * (vId * vScales[vIb] - 1));
      l = MAX(0, MIN(15, l));
      if (vIb % 2 == 0)
        vY[vIbl].scales[vIb / 2] = l;
      else
        vY[vIbl].scales[vIb / 2] |= (l << 4);
    }
  }
}

size_t quantize_iq2_s(const float *GGML_RESTRICT pSrc, void *GGML_RESTRICT pDst, int64_t pNrow, int64_t pNPerRow,
                      const float *pQuantWeights) {
  GGML_ASSERT(pNPerRow % QK_K == 0);
  int64_t vNblock = pNPerRow / QK_K;
  char *vQrow = (char *)pDst;
  for (int64_t vRow = 0; vRow < pNrow; ++vRow) {
    fQuantizeRowIq2SImpl(pSrc, vQrow, pNPerRow, pQuantWeights);
    pSrc += pNPerRow;
    vQrow += vNblock * sizeof(block_iq2_s);
  }
  return pNrow * vNblock * sizeof(block_iq2_s);
}

void quantize_row_iq2_s_ref(const float *GGML_RESTRICT pX, block_iq2_s *GGML_RESTRICT pY, int64_t pK) {
  assert(pK % QK_K == 0);
  quantize_iq2_s(pX, pY, 1, pK, NULL);
}

// =============================== data validation

static bool fValidateFloat(float pF, size_t pI) {
  if (isinf(pF)) {
    fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", pI);
    return false;
  }

  if (isnan(pF)) {
    fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", pI);
    return false;
  }

  return true;
}

static bool fIsinfFp16(ggml_fp16_t pF) { return (pF & 0x7c00) == 0x7c00 && (pF & 0x03ff) == 0; }

static bool fIsnanFp16(ggml_fp16_t pF) { return (pF & 0x7c00) == 0x7c00 && (pF & 0x03ff) != 0; }

static bool validate_fp16(ggml_fp16_t pF, size_t pI) {
  if (fIsinfFp16(pF)) {
    fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", pI);
    return false;
  }

  if (fIsnanFp16(pF)) {
    fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", pI);
    return false;
  }

  return true;
}

static bool validate_e_e8m0(uint8_t pE, size_t pI) {
  if (pE == 0xff) {
    fprintf(stderr, "ggml_validate_row_data: found invalid e value %d at block %zu\n", pE, pI);
    return false;
  }

  return true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb)                                                                   \
  const type *q = (const type *)(data);                                                                                \
  for (size_t i = 0; i < (nb); ++i) {                                                                                  \
    if (!validate_fp16(q[i].d, i)) {                                                                                   \
      return false;                                                                                                    \
    }                                                                                                                  \
  }

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m)                                                            \
  const type *q = (const type *)(data);                                                                                \
  for (size_t i = 0; i < (nb); ++i) {                                                                                  \
    if (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) {                                                      \
      return false;                                                                                                    \
    }                                                                                                                  \
  }

#define VALIDATE_ROW_DATA_E_E8M0_IMPL(type, data, nb)                                                                  \
  const type *q = (const type *)(data);                                                                                \
  for (size_t i = 0; i < (nb); ++i) {                                                                                  \
    if (!validate_e_e8m0(q[i].e, i)) {                                                                                 \
      return false;                                                                                                    \
    }                                                                                                                  \
  }

#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr)                                                            \
  const type *q = (const type *)(data);                                                                                \
  for (size_t i = 0; i < (nb); ++i) {                                                                                  \
    for (size_t j = 0; j < (nr); ++j) {                                                                                \
      if (!validate_fp16(q[i].d[j], i)) {                                                                              \
        return false;                                                                                                  \
      }                                                                                                                \
    }                                                                                                                  \
  }

bool ggml_validate_row_data(enum ggml_type pType, const void *pData, size_t pNbytes) {
  if (pType < 0 || pType >= GGML_TYPE_COUNT) {
    fprintf(stderr, "%s: invalid type %d\n", __func__, pType);
    return false;
  }

  if (pNbytes % ggml_type_size(pType) != 0) {
    fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, pNbytes, ggml_type_name(pType),
            ggml_type_size(pType));
    return false;
  }

  const size_t cNb = pNbytes / ggml_type_size(pType);

  switch (pType) {
  case GGML_TYPE_BF16: {
    int vNans = 0;
    int vInfs = 0;
    const unsigned short *vF = (const unsigned short *)pData;
    for (size_t vI = 0; vI < cNb; ++vI) {
      vNans += (vF[vI] & 0x7fff) > 0x7f80;
      vInfs += (vF[vI] & 0x7fff) == 0x7f80;
    }
    if (vNans) {
      fprintf(stderr, "%s: found %d NaNs in row of %zu BF16 values\n", __func__, vNans, cNb);
      return false;
    }
    if (vInfs) {
      fprintf(stderr, "%s: found %d infinities in row of %zu BF16 values\n", __func__, vInfs, cNb);
      return false;
    }
  } break;
  case GGML_TYPE_F16: {
    const ggml_fp16_t *vF = (const ggml_fp16_t *)pData;
    size_t vI = 0;
#if defined(__AVX2__)
    for (; vI + 15 < cNb; vI += 16) {
      __m256i v = _mm256_loadu_si256((const __m256i *)(vF + vI));
      __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
      __m256i cmp = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
      int mask = _mm256_movemask_epi8(cmp);
      if (mask) {
        for (size_t j = 0; j < 16; ++j) {
          if (!validate_fp16(vF[vI + j], vI + j)) {
            return false;
          }
        }
        GGML_UNREACHABLE();
      }
    }
#elif defined(__ARM_NEON)
    for (; vI + 7 < cNb; vI += 8) {
      uint16x8_t v = vld1q_u16(vF + vI);
      uint16x8_t vexp = vandq_u16(v, vdupq_n_u16(0x7c00));
      uint16x8_t cmp = vceqq_u16(vexp, vdupq_n_u16(0x7c00));
      uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0);
      if (mask) {
        for (size_t j = 0; j < 8; ++j) {
          if (!validate_fp16(vF[vI + j], vI + j)) {
            return false;
          }
        }
        GGML_UNREACHABLE();
      }
    }
#endif
    for (; vI < cNb; ++vI) {
      if (!validate_fp16(vF[vI], vI)) {
        return false;
      }
    }
  } break;
  case GGML_TYPE_F32: {
    const float *vF = (const float *)pData;
    size_t vI = 0;
#if defined(__AVX2__)
    for (; vI + 7 < cNb; vI += 8) {
      __m256i v = _mm256_loadu_si256((const __m256i *)(vF + vI));
      __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi32(0x7f800000));
      __m256i cmp = _mm256_cmpeq_epi32(vexp, _mm256_set1_epi32(0x7f800000));
      int mask = _mm256_movemask_epi8(cmp);
      if (mask) {
        for (size_t j = 0; j < 8; ++j) {
          if (!fValidateFloat(vF[vI + j], vI + j)) {
            return false;
          }
        }
        GGML_UNREACHABLE();
      }
    }
#elif defined(__ARM_NEON)
    for (; vI + 3 < cNb; vI += 4) {
      uint32x4_t v = vld1q_u32((const uint32_t *)vF + vI);
      uint32x4_t vexp = vandq_u32(v, vdupq_n_u32(0x7f800000));
      uint32x4_t cmp = vceqq_u32(vexp, vdupq_n_u32(0x7f800000));
      uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(cmp, 8)), 0);
      if (mask) {
        for (size_t j = 0; j < 4; ++j) {
          if (!fValidateFloat(vF[vI + j], vI + j)) {
            return false;
          }
        }
        GGML_UNREACHABLE();
      }
    }
#endif
    for (; vI < cNb; ++vI) {
      if (!fValidateFloat(vF[vI], vI)) {
        return false;
      }
    }
  } break;
  case GGML_TYPE_F64: {
    const double *vF = (const double *)pData;
    for (size_t vI = 0; vI < cNb; ++vI) {
      if (!fValidateFloat(vF[vI], vI)) {
        return false;
      }
    }
  } break;
  case GGML_TYPE_Q1_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q1_0, pData, cNb);
  } break;
  case GGML_TYPE_Q2_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q2_0, pData, cNb);
  } break;
  case GGML_TYPE_Q4_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q4_0, pData, cNb);
  } break;
  case GGML_TYPE_Q4_1: {
    VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_1, pData, cNb, d, m);
  } break;
  case GGML_TYPE_Q5_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q5_0, pData, cNb);
  } break;
  case GGML_TYPE_Q5_1: {
    VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_1, pData, cNb, d, m);
  } break;
  case GGML_TYPE_Q8_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q8_0, pData, cNb);
  } break;
  case GGML_TYPE_MXFP4: {
    VALIDATE_ROW_DATA_E_E8M0_IMPL(block_mxfp4, pData, cNb);
  } break;
  case GGML_TYPE_NVFP4: {
    // UE4M3 scales are uint8_t — all byte values are valid
    GGML_UNUSED(pData);
    GGML_UNUSED(cNb);
  } break;
  case GGML_TYPE_Q2_K: {
    VALIDATE_ROW_DATA_DM_F16_IMPL(block_q2_K, pData, cNb, d, dmin);
  } break;
  case GGML_TYPE_Q3_K: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q3_K, pData, cNb);
  } break;
  case GGML_TYPE_Q4_K: {
    VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_K, pData, cNb, d, dmin);
  } break;
  case GGML_TYPE_Q5_K: {
    VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_K, pData, cNb, d, dmin);
  } break;
  case GGML_TYPE_Q6_K: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_K, pData, cNb);
  } break;
  case GGML_TYPE_Q8_K: {
    const block_q8_K *vQ = (const block_q8_K *)pData;
    for (size_t vI = 0; vI < cNb; ++vI) {
      if (!fValidateFloat(vQ[vI].d, vI)) {
        return false;
      }
    }
  } break;
  case GGML_TYPE_TQ1_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_tq1_0, pData, cNb);
  } break;
  case GGML_TYPE_TQ2_0: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, pData, cNb);
  } break;
  case GGML_TYPE_IQ1_S: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq1_s, pData, cNb);
  } break;
  case GGML_TYPE_IQ1_M: {
    const block_iq1_m *vQ = (const block_iq1_m *)pData;
    for (size_t vI = 0; vI < cNb; ++vI) {
      iq1m_scale_t vScale;
      const uint16_t *vSc = (const uint16_t *)vQ[vI].scales;
      vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);
      if (!validate_fp16(vScale.f16, vI)) {
        return false;
      }
    }
  } break;
  case GGML_TYPE_IQ2_XXS: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xxs, pData, cNb);
  } break;
  case GGML_TYPE_IQ2_XS: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xs, pData, cNb);
  } break;
  case GGML_TYPE_IQ2_S: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_s, pData, cNb);
  } break;
  case GGML_TYPE_IQ3_XXS: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_xxs, pData, cNb);
  } break;

  case GGML_TYPE_IQ3_S: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_s, pData, cNb);
  } break;
  case GGML_TYPE_IQ4_XS: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_xs, pData, cNb);
  } break;
  case GGML_TYPE_IQ4_NL: {
    VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_nl, pData, cNb);
  } break;

  case GGML_TYPE_I8:
  case GGML_TYPE_I16:
  case GGML_TYPE_I32:
  case GGML_TYPE_I64:
    // nothing to validate
    break;
  default: {
    fprintf(stderr, "%s: invalid type %d\n", __func__, pType);
    return false;
  }
  }

  return true;
}
