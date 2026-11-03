#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int pN, const int8_t *pVal, float pX) {
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

static __device__ void quantize_f32_q4_0_block(const float *__restrict__ pX, block_q4_0 *__restrict__ pY) {
  float vAmax = 0.0f;
  float vVmax = 0.0f;

  for (int vJ = 0; vJ < QK4_0; ++vJ) {
    const float cV = pX[vJ];
    if (vAmax < fabsf(cV)) {
      vAmax = fabsf(cV);
      vVmax = cV;
    }
  }

  const float cD = vVmax / -8;
  const float cId = cD ? 1.0f / cD : 0.0f;

  pY->d = cD;

  for (int vJ = 0; vJ < QK4_0 / 2; ++vJ) {
    const float cX0 = pX[0 + vJ] * cId;
    const float cX1 = pX[QK4_0 / 2 + vJ] * cId;

    const uint8_t cXi0 = min(15, (int8_t)(cX0 + 8.5f));
    const uint8_t cXi1 = min(15, (int8_t)(cX1 + 8.5f));

    pY->qs[vJ] = cXi0;
    pY->qs[vJ] |= cXi1 << 4;
  }
}

static __device__ void quantize_f32_q4_1_block(const float *__restrict__ pX, block_q4_1 *__restrict__ pY) {
  float vVmin = FLT_MAX;
  float vVmax = -FLT_MAX;

  for (int vJ = 0; vJ < QK4_1; ++vJ) {
    const float cV = pX[vJ];
    if (cV < vVmin)
      vVmin = cV;
    if (cV > vVmax)
      vVmax = cV;
  }

  const float cD = (vVmax - vVmin) / ((1 << 4) - 1);
  const float cId = cD ? 1.0f / cD : 0.0f;

  pY->dm.x = cD;
  pY->dm.y = vVmin;

  for (int vJ = 0; vJ < QK4_1 / 2; ++vJ) {
    const float cX0 = (pX[0 + vJ] - vVmin) * cId;
    const float cX1 = (pX[QK4_1 / 2 + vJ] - vVmin) * cId;

    const uint8_t cXi0 = min(15, (int8_t)(cX0 + 0.5f));
    const uint8_t cXi1 = min(15, (int8_t)(cX1 + 0.5f));

    pY->qs[vJ] = cXi0;
    pY->qs[vJ] |= cXi1 << 4;
  }
}

static __device__ void quantize_f32_q5_0_block(const float *__restrict__ pX, block_q5_0 *__restrict__ pY) {
  float vAmax = 0.0f;
  float vVmax = 0.0f;

  for (int vJ = 0; vJ < QK5_0; ++vJ) {
    const float cV = pX[vJ];
    if (vAmax < fabsf(cV)) {
      vAmax = fabsf(cV);
      vVmax = cV;
    }
  }

  const float cD = vVmax / -16;
  const float cId = cD ? 1.0f / cD : 0.0f;

  pY->d = cD;

  uint32_t vQh = 0;
  for (int vJ = 0; vJ < QK5_0 / 2; ++vJ) {
    const float cX0 = pX[0 + vJ] * cId;
    const float cX1 = pX[QK5_0 / 2 + vJ] * cId;

    const uint8_t cXi0 = min(31, (int8_t)(cX0 + 16.5f));
    const uint8_t cXi1 = min(31, (int8_t)(cX1 + 16.5f));

    pY->qs[vJ] = (cXi0 & 0xf) | ((cXi1 & 0xf) << 4);
    vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
    vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + QK5_0 / 2);
  }
  memcpy(pY->qh, &vQh, sizeof(vQh));
}

static __device__ void quantize_f32_q5_1_block(const float *__restrict__ pX, block_q5_1 *__restrict__ pY) {
  float vMin = pX[0];
  float vMax = pX[0];

  for (int vJ = 1; vJ < QK5_1; ++vJ) {
    const float cV = pX[vJ];
    vMin = cV < vMin ? cV : vMin;
    vMax = cV > vMax ? cV : vMax;
  }

  const float cD = (vMax - vMin) / 31;
  const float cId = cD ? 1.0f / cD : 0.0f;

  pY->dm.x = cD;
  pY->dm.y = vMin;

  uint32_t vQh = 0;
  for (int vJ = 0; vJ < QK5_1 / 2; ++vJ) {
    const float cX0 = (pX[0 + vJ] - vMin) * cId;
    const float cX1 = (pX[QK5_1 / 2 + vJ] - vMin) * cId;

    const uint8_t cXi0 = (uint8_t)(cX0 + 0.5f);
    const uint8_t cXi1 = (uint8_t)(cX1 + 0.5f);

    pY->qs[vJ] = (cXi0 & 0xf) | ((cXi1 & 0xf) << 4);
    vQh |= ((cXi0 & 0x10u) >> 4) << (vJ + 0);
    vQh |= ((cXi1 & 0x10u) >> 4) << (vJ + QK5_1 / 2);
  }
  memcpy(pY->qh, &vQh, sizeof(vQh));
}

static __device__ void quantize_f32_q8_0_block(const float *__restrict__ pX, block_q8_0 *__restrict__ pY) {
  float vAmax = 0.0f; // absolute max

  for (int vJ = 0; vJ < QK8_0; vJ++) {
    const float cV = pX[vJ];
    vAmax = fmaxf(vAmax, fabsf(cV));
  }

  const float cD = vAmax / ((1 << 7) - 1);
  const float cId = cD ? 1.0f / cD : 0.0f;

  pY->d = cD;

  for (int vJ = 0; vJ < QK8_0; ++vJ) {
    const float cX0 = pX[vJ] * cId;
    pY->qs[vJ] = roundf(cX0);
  }
}

static __device__ void quantize_f32_iq4_nl_block(const float *__restrict__ pX, block_iq4_nl *__restrict__ pY) {
  float vAmax = 0.0f;
  float vVmax = 0.0f;

  for (int vJ = 0; vJ < QK4_NL; ++vJ) {
    const float cV = pX[vJ];
    if (vAmax < fabsf(cV)) {
      vAmax = fabsf(cV);
      vVmax = cV;
    }
  }

  float vD = vVmax / kvalues_iq4nl[0];
  const float cId = vD ? 1.0f / vD : 0.0f;

  float vSumqx = 0, vSumq2 = 0;
  for (int vJ = 0; vJ < QK4_NL / 2; ++vJ) {
    const float cX0 = pX[0 + vJ] * cId;
    const float cX1 = pX[QK4_NL / 2 + vJ] * cId;
    const uint8_t cXi0 = best_index_int8(16, kvalues_iq4nl, cX0);
    const uint8_t cXi1 = best_index_int8(16, kvalues_iq4nl, cX1);
    pY->qs[vJ] = cXi0 | (cXi1 << 4);
    const float cV0 = kvalues_iq4nl[cXi0];
    const float cV1 = kvalues_iq4nl[cXi1];
    const float cW0 = pX[0 + vJ] * pX[0 + vJ];
    const float cW1 = pX[QK4_NL / 2 + vJ] * pX[QK4_NL / 2 + vJ];
    vSumqx += cW0 * cV0 * pX[vJ] + cW1 * cV1 * pX[QK4_NL / 2 + vJ];
    vSumq2 += cW0 * cV0 * cV0 + cW1 * cV1 * cV1;
  }

  pY->d = vSumq2 > 0 ? vSumqx / vSumq2 : vD;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char *pCxi, char *pCdsti) {
  quantize_f32_q4_0_block((const float *)pCxi, (block_q4_0 *)pCdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char *pCxi, char *pCdsti) {
  quantize_f32_q4_1_block((const float *)pCxi, (block_q4_1 *)pCdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char *pCxi, char *pCdsti) {
  quantize_f32_q5_0_block((const float *)pCxi, (block_q5_0 *)pCdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char *pCxi, char *pCdsti) {
  quantize_f32_q5_1_block((const float *)pCxi, (block_q5_1 *)pCdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char *pCxi, char *pCdsti) {
  quantize_f32_q8_0_block((const float *)pCxi, (block_q8_0 *)pCdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char *pCxi, char *pCdsti) {
  quantize_f32_iq4_nl_block((const float *)pCxi, (block_iq4_nl *)pCdsti);
}

template <typename src_t, typename dst_t> static __device__ void cpy_1_scalar(const char *pCxi, char *pCdsti) {
  *(dst_t *)pCdsti = ggml_cuda_cast<dst_t>(*(const src_t *)pCxi);
}
