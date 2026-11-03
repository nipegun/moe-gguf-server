#include "common.cuh"
#include "convert.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q1_0 *vX = (const block_q1_0 *)pVx;

  const float cD = vX[pIb].d;

  const int cBitIndex0 = pIqs;
  const int cBitIndex1 = pIqs + 1;

  const int cByteIndex0 = cBitIndex0 / 8;
  const int cBitOffset0 = cBitIndex0 % 8;

  const int cByteIndex1 = cBitIndex1 / 8;
  const int cBitOffset1 = cBitIndex1 % 8;

  // Extract bits: 1 = +d, 0 = -d (branchless)
  const int cBit0 = (vX[pIb].qs[cByteIndex0] >> cBitOffset0) & 1;
  const int cBit1 = (vX[pIb].qs[cByteIndex1] >> cBitOffset1) & 1;

  pV.x = (2 * cBit0 - 1) * cD;
  pV.y = (2 * cBit1 - 1) * cD;
}

static __device__ __forceinline__ void dequantize_q2_0(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q2_0 *vX = (const block_q2_0 *)pVx;

  const float cD = vX[pIb].d;

  // Q2_0: 2 bits per element, 4 elements per byte.
  // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
  const int cByteIndex0 = pIqs / 4;
  const int cBitOffset0 = (pIqs % 4) * 2;

  const int cByteIndex1 = (pIqs + 1) / 4;
  const int cBitOffset1 = ((pIqs + 1) % 4) * 2;

  const int c0 = (vX[pIb].qs[cByteIndex0] >> cBitOffset0) & 0x3;
  const int c1 = (vX[pIb].qs[cByteIndex1] >> cBitOffset1) & 0x3;

  pV.x = (c0 - 1) * cD;
  pV.y = (c1 - 1) * cD;
}

static __device__ __forceinline__ void dequantize_q4_0(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q4_0 *vX = (const block_q4_0 *)pVx;

  const float cD = vX[pIb].d;

  const int cVui = vX[pIb].qs[pIqs];

  pV.x = cVui & 0xF;
  pV.y = cVui >> 4;

  pV.x = (pV.x - 8.0f) * cD;
  pV.y = (pV.y - 8.0f) * cD;
}

static __device__ __forceinline__ void dequantize_q4_1(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q4_1 *vX = (const block_q4_1 *)pVx;

  const float2 cDm = __half22float2(vX[pIb].dm);

  const int cVui = vX[pIb].qs[pIqs];

  pV.x = cVui & 0xF;
  pV.y = cVui >> 4;

  pV.x = (pV.x * cDm.x) + cDm.y;
  pV.y = (pV.y * cDm.x) + cDm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q5_0 *vX = (const block_q5_0 *)pVx;

  const float cD = vX[pIb].d;

  uint32_t vQh;
  memcpy(&vQh, vX[pIb].qh, sizeof(vQh));

  const int cXh0 = ((vQh >> (pIqs + 0)) << 4) & 0x10;
  const int cXh1 = ((vQh >> (pIqs + 12))) & 0x10;

  pV.x = ((vX[pIb].qs[pIqs] & 0xf) | cXh0);
  pV.y = ((vX[pIb].qs[pIqs] >> 4) | cXh1);

  pV.x = (pV.x - 16.0f) * cD;
  pV.y = (pV.y - 16.0f) * cD;
}

static __device__ __forceinline__ void dequantize_q5_1(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q5_1 *vX = (const block_q5_1 *)pVx;

  const float2 cDm = __half22float2(vX[pIb].dm);

  uint32_t vQh;
  memcpy(&vQh, vX[pIb].qh, sizeof(vQh));

  const int cXh0 = ((vQh >> (pIqs + 0)) << 4) & 0x10;
  const int cXh1 = ((vQh >> (pIqs + 12))) & 0x10;

  pV.x = ((vX[pIb].qs[pIqs] & 0xf) | cXh0);
  pV.y = ((vX[pIb].qs[pIqs] >> 4) | cXh1);

  pV.x = (pV.x * cDm.x) + cDm.y;
  pV.y = (pV.y * cDm.x) + cDm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void *pVx, const int64_t pIb, const int pIqs, float2 &pV) {
  const block_q8_0 *vX = (const block_q8_0 *)pVx;

  const float cD = vX[pIb].d;

  pV.x = vX[pIb].qs[pIqs + 0];
  pV.y = vX[pIb].qs[pIqs + 1];

  pV.x *= cD;
  pV.y *= cD;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template <typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void *pVx, const int64_t pIb, dst_t *pYy, const int pTid) {
  const block_q2_K *vX = (const block_q2_K *)pVx;

  const int64_t cN = pTid / 32;
  const int64_t cL = pTid - 32 * cN;
  const int64_t cIs = 8 * cN + cL / 16;

  const uint8_t cQ = vX[pIb].qs[32 * cN + cL];
  dst_t *vY = pYy + 128 * cN;

  float vDall = __low2half(vX[pIb].dm);
  float vDmin = __high2half(vX[pIb].dm);
  vY[cL + 0] = ggml_cuda_cast<dst_t>(vDall * (vX[pIb].scales[cIs + 0] & 0xF) * ((cQ >> 0) & 3) -
                                     vDmin * (vX[pIb].scales[cIs + 0] >> 4));
  vY[cL + 32] = ggml_cuda_cast<dst_t>(vDall * (vX[pIb].scales[cIs + 2] & 0xF) * ((cQ >> 2) & 3) -
                                      vDmin * (vX[pIb].scales[cIs + 2] >> 4));
  vY[cL + 64] = ggml_cuda_cast<dst_t>(vDall * (vX[pIb].scales[cIs + 4] & 0xF) * ((cQ >> 4) & 3) -
                                      vDmin * (vX[pIb].scales[cIs + 4] >> 4));
  vY[cL + 96] = ggml_cuda_cast<dst_t>(vDall * (vX[pIb].scales[cIs + 6] & 0xF) * ((cQ >> 6) & 3) -
                                      vDmin * (vX[pIb].scales[cIs + 6] >> 4));
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void *pVx, const int64_t pIb, dst_t *pYy, const int pTid) {
  const block_q3_K *vX = (const block_q3_K *)pVx;

  const int64_t cR = pTid / 4;
  const int64_t cT = cR / 2;
  const int64_t cIs0 = cR % 2;
  const int64_t cL0 = 16 * cIs0 + 4 * (pTid % 4);
  const int64_t cN = cT / 4;
  const int64_t cJ = cT - 4 * cN;

  uint8_t vM = 1 << (4 * cN + cJ);
  int64_t vIs = 8 * cN + 2 * cJ + cIs0;
  int vShift = 2 * cJ;

  int8_t vUs = vIs < 4    ? (vX[pIb].scales[vIs - 0] & 0xF) | (((vX[pIb].scales[vIs + 8] >> 0) & 3) << 4)
               : vIs < 8  ? (vX[pIb].scales[vIs - 0] & 0xF) | (((vX[pIb].scales[vIs + 4] >> 2) & 3) << 4)
               : vIs < 12 ? (vX[pIb].scales[vIs - 8] >> 4) | (((vX[pIb].scales[vIs + 0] >> 4) & 3) << 4)
                          : (vX[pIb].scales[vIs - 8] >> 4) | (((vX[pIb].scales[vIs - 4] >> 6) & 3) << 4);
  float vDAll = vX[pIb].d;
  float vDl = vDAll * (vUs - 32);

  dst_t *vY = pYy + 128 * cN + 32 * cJ;
  const uint8_t *vQ = vX[pIb].qs + 32 * cN;
  const uint8_t *vHm = vX[pIb].hmask;

  for (int vL = cL0; vL < cL0 + 4; ++vL) {
    vY[vL] = ggml_cuda_cast<dst_t>(vDl * ((int8_t)((vQ[vL] >> vShift) & 3) - ((vHm[vL] & vM) ? 0 : 4)));
  }
}

static inline __device__ void get_scale_min_k4(int pJ, const uint8_t *pQ, uint8_t &pD, uint8_t &pM) {
  if (pJ < 4) {
    pD = pQ[pJ] & 63;
    pM = pQ[pJ + 4] & 63;
  } else {
    pD = (pQ[pJ + 4] & 0xF) | ((pQ[pJ - 4] >> 6) << 4);
    pM = (pQ[pJ + 4] >> 4) | ((pQ[pJ - 0] >> 6) << 4);
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void *pVx, const int64_t pIb, dst_t *pYy, const int pTid) {
  const block_q4_K *vX = (const block_q4_K *)pVx;

  // assume 32 threads
  const int64_t cIl = pTid / 8;
  const int64_t cIr = pTid % 8;
  const int64_t cIs = 2 * cIl;
  const int64_t cN = 4;

  dst_t *vY = pYy + 64 * cIl + cN * cIr;

  const float cDall = __low2half(vX[pIb].dm);
  const float cDmin = __high2half(vX[pIb].dm);

  const uint8_t *vQ = vX[pIb].qs + 32 * cIl + cN * cIr;

  uint8_t vSc, vM;
  get_scale_min_k4(cIs + 0, vX[pIb].scales, vSc, vM);
  const float cD1 = cDall * vSc;
  const float cM1 = cDmin * vM;
  get_scale_min_k4(cIs + 1, vX[pIb].scales, vSc, vM);
  const float cD2 = cDall * vSc;
  const float cM2 = cDmin * vM;
  for (int vL = 0; vL < cN; ++vL) {
    vY[vL + 0] = ggml_cuda_cast<dst_t>(cD1 * (vQ[vL] & 0xF) - cM1);
    vY[vL + 32] = ggml_cuda_cast<dst_t>(cD2 * (vQ[vL] >> 4) - cM2);
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void *pVx, const int64_t pIb, dst_t *pYy, const int pTid) {
  const block_q5_K *vX = (const block_q5_K *)pVx;

  // assume 64 threads - this is very slightly better than the one below
  const int64_t cIl = pTid / 16; // il is in 0...3
  const int64_t cIr = pTid % 16; // ir is in 0...15
  const int64_t cIs = 2 * cIl;   // is is in 0...6

  dst_t *vY = pYy + 64 * cIl + 2 * cIr;

  const float cDall = __low2half(vX[pIb].dm);
  const float cDmin = __high2half(vX[pIb].dm);

  const uint8_t *vQl = vX[pIb].qs + 32 * cIl + 2 * cIr;
  const uint8_t *vQh = vX[pIb].qh + 2 * cIr;

  uint8_t vSc, vM;
  get_scale_min_k4(cIs + 0, vX[pIb].scales, vSc, vM);
  const float cD1 = cDall * vSc;
  const float cM1 = cDmin * vM;
  get_scale_min_k4(cIs + 1, vX[pIb].scales, vSc, vM);
  const float cD2 = cDall * vSc;
  const float cM2 = cDmin * vM;

  uint8_t vHm = 1 << (2 * cIl);
  vY[0] = ggml_cuda_cast<dst_t>(cD1 * ((vQl[0] & 0xF) + (vQh[0] & vHm ? 16 : 0)) - cM1);
  vY[1] = ggml_cuda_cast<dst_t>(cD1 * ((vQl[1] & 0xF) + (vQh[1] & vHm ? 16 : 0)) - cM1);
  vHm <<= 1;
  vY[32] = ggml_cuda_cast<dst_t>(cD2 * ((vQl[0] >> 4) + (vQh[0] & vHm ? 16 : 0)) - cM2);
  vY[33] = ggml_cuda_cast<dst_t>(cD2 * ((vQl[1] >> 4) + (vQh[1] & vHm ? 16 : 0)) - cM2);
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void *pVx, const int64_t pIb, dst_t *pYy, const int pTid) {
  const block_q6_K *vX = (const block_q6_K *)pVx;

  // assume 64 threads - this is very slightly better than the one below
  const int64_t cIp = pTid / 32;       // ip is 0 or 1
  const int64_t cIl = pTid - 32 * cIp; // 0...32
  const int64_t cIs = 8 * cIp + cIl / 16;

  dst_t *vY = pYy + 128 * cIp + cIl;

  const float cD = vX[pIb].d;

  const uint8_t *vQl = vX[pIb].ql + 64 * cIp + cIl;
  const uint8_t cQh = vX[pIb].qh[32 * cIp + cIl];
  const int8_t *vSc = vX[pIb].scales + cIs;

  vY[0] = ggml_cuda_cast<dst_t>(cD * vSc[0] * ((int8_t)((vQl[0] & 0xF) | (((cQh >> 0) & 3) << 4)) - 32));
  vY[32] = ggml_cuda_cast<dst_t>(cD * vSc[2] * ((int8_t)((vQl[32] & 0xF) | (((cQh >> 2) & 3) << 4)) - 32));
  vY[64] = ggml_cuda_cast<dst_t>(cD * vSc[4] * ((int8_t)((vQl[0] >> 4) | (((cQh >> 4) & 3) << 4)) - 32));
  vY[96] = ggml_cuda_cast<dst_t>(cD * vSc[6] * ((int8_t)((vQl[32] >> 4) | (((cQh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                          const int pTid) {

  const block_iq2_xxs *vX = (const block_iq2_xxs *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint16_t *vQ2 = vX[pIbs].qs + 4 * cIb;
  const uint8_t *vAux8 = (const uint8_t *)vQ2;
  const uint8_t *vGrid = (const uint8_t *)(iq2xxs_grid + vAux8[cIl]);
  const uint32_t cAux32 = vQ2[2] | (vQ2[3] << 16);
  const float cD = (float)vX[pIbs].d * (0.5f + (cAux32 >> 28)) * 0.25f;
  const uint8_t cSigns = ksigns_iq2xs[(cAux32 >> 7 * cIl) & 127];
  for (int vJ = 0; vJ < 8; ++vJ) {
    vY[vJ] = ggml_cuda_cast<dst_t>(cD * vGrid[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1.f : 1.f));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                         const int pTid) {

  const block_iq2_xs *vX = (const block_iq2_xs *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint16_t *vQ2 = vX[pIbs].qs + 4 * cIb;
  const uint8_t *vGrid = (const uint8_t *)(iq2xs_grid + (vQ2[cIl] & 511));
  const float cD = (float)vX[pIbs].d * (0.5f + ((vX[pIbs].scales[cIb] >> 4 * (cIl / 2)) & 0xf)) * 0.25f;
  const uint8_t cSigns = ksigns_iq2xs[vQ2[cIl] >> 9];
  for (int vJ = 0; vJ < 8; ++vJ) {
    vY[vJ] = ggml_cuda_cast<dst_t>(cD * vGrid[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1.f : 1.f));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                        const int pTid) {

  const block_iq2_s *vX = (const block_iq2_s *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint8_t *vGrid =
      (const uint8_t *)(iq2s_grid + (vX[pIbs].qs[4 * cIb + cIl] | ((vX[pIbs].qh[cIb] << (8 - 2 * cIl)) & 0x300)));
  const float cD = (float)vX[pIbs].d * (0.5f + ((vX[pIbs].scales[cIb] >> 4 * (cIl / 2)) & 0xf)) * 0.25f;
  const uint8_t cSigns = vX[pIbs].qs[QK_K / 8 + 4 * cIb + cIl];
  for (int vJ = 0; vJ < 8; ++vJ) {
    vY[vJ] = ggml_cuda_cast<dst_t>(cD * vGrid[vJ] * (cSigns & kmask_iq2xs[vJ] ? -1.f : 1.f));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                          const int pTid) {

  const block_iq3_xxs *vX = (const block_iq3_xxs *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint8_t *vQ3 = vX[pIbs].qs + 8 * cIb;
  const uint16_t *vGas = (const uint16_t *)(vX[pIbs].qs + QK_K / 4) + 2 * cIb;
  const uint8_t *vGrid1 = (const uint8_t *)(iq3xxs_grid + vQ3[2 * cIl + 0]);
  const uint8_t *vGrid2 = (const uint8_t *)(iq3xxs_grid + vQ3[2 * cIl + 1]);
  const uint32_t cAux32 = vGas[0] | (vGas[1] << 16);
  const float cD = (float)vX[pIbs].d * (0.5f + (cAux32 >> 28)) * 0.5f;
  const uint8_t cSigns = ksigns_iq2xs[(cAux32 >> 7 * cIl) & 127];
  for (int vJ = 0; vJ < 4; ++vJ) {
    vY[vJ + 0] = ggml_cuda_cast<dst_t>(cD * vGrid1[vJ] * (cSigns & kmask_iq2xs[vJ + 0] ? -1.f : 1.f));
    vY[vJ + 4] = ggml_cuda_cast<dst_t>(cD * vGrid2[vJ] * (cSigns & kmask_iq2xs[vJ + 4] ? -1.f : 1.f));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                        const int pTid) {

  const block_iq3_s *vX = (const block_iq3_s *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint8_t *vQs = vX[pIbs].qs + 8 * cIb;
  const uint8_t *vGrid1 =
      (const uint8_t *)(iq3s_grid + (vQs[2 * cIl + 0] | ((vX[pIbs].qh[cIb] << (8 - 2 * cIl)) & 256)));
  const uint8_t *vGrid2 =
      (const uint8_t *)(iq3s_grid + (vQs[2 * cIl + 1] | ((vX[pIbs].qh[cIb] << (7 - 2 * cIl)) & 256)));
  const float cD = (float)vX[pIbs].d * (1 + 2 * ((vX[pIbs].scales[cIb / 2] >> 4 * (cIb % 2)) & 0xf));
  const uint8_t cSigns = vX[pIbs].signs[4 * cIb + cIl];
  for (int vJ = 0; vJ < 4; ++vJ) {
    vY[vJ + 0] = ggml_cuda_cast<dst_t>(cD * vGrid1[vJ] * (cSigns & kmask_iq2xs[vJ + 0] ? -1.f : 1.f));
    vY[vJ + 4] = ggml_cuda_cast<dst_t>(cD * vGrid2[vJ] * (cSigns & kmask_iq2xs[vJ + 4] ? -1.f : 1.f));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                        const int pTid) {

  const block_iq1_s *vX = (const block_iq1_s *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const float cDelta = vX[pIbs].qh[cIb] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
  const float cD = (float)vX[pIbs].d * (2 * ((vX[pIbs].qh[cIb] >> 12) & 7) + 1);
  uint32_t vGrid32[2];
  const int8_t *vQ = (const int8_t *)vGrid32;
  vGrid32[0] = iq1s_grid_gpu[vX[pIbs].qs[4 * cIb + cIl] | (((vX[pIbs].qh[cIb] >> 3 * cIl) & 7) << 8)];
  vGrid32[1] = (vGrid32[0] >> 4) & 0x0f0f0f0f;
  vGrid32[0] &= 0x0f0f0f0f;
  for (int vJ = 0; vJ < 8; ++vJ) {
    vY[vJ] = ggml_cuda_cast<dst_t>(cD * (vQ[vJ] + cDelta));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                        const int pTid) {

  const block_iq1_m *vX = (const block_iq1_m *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 8 * cIl;
  const uint16_t *vSc = (const uint16_t *)vX[pIbs].scales;
  iq1m_scale_t vScale;
  vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00f0) | ((vSc[2] >> 4) & 0x0f00) | (vSc[3] & 0xf000);
  const int64_t cIb16 = 2 * cIb + cIl / 2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
  const float cD = (float)vScale.f16 * (2 * ((vSc[cIb16 / 4] >> 3 * (cIb16 % 4)) & 0x7) + 1);
  const float cDelta = vX[pIbs].qh[2 * cIb + cIl / 2] & (0x08 << 4 * (cIl % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
  uint32_t vGrid32[2];
  const int8_t *vQ = (const int8_t *)vGrid32;
  vGrid32[0] =
      iq1s_grid_gpu[vX[pIbs].qs[4 * cIb + cIl] | (((vX[pIbs].qh[2 * cIb + cIl / 2] >> 4 * (cIl % 2)) & 7) << 8)];
  vGrid32[1] = (vGrid32[0] >> 4) & 0x0f0f0f0f;
  vGrid32[0] &= 0x0f0f0f0f;
  for (int vJ = 0; vJ < 8; ++vJ) {
    vY[vJ] = ggml_cuda_cast<dst_t>(cD * (vQ[vJ] + cDelta));
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                         const int pTid) {

  const block_iq4_nl *vX = (const block_iq4_nl *)pVx + pIbs * (QK_K / QK4_NL);

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 4 * cIl;
  const uint8_t *vQ4 = vX[cIb].qs + 4 * cIl;
  const float cD = (float)vX[cIb].d;
  for (int vJ = 0; vJ < 4; ++vJ) {
    vY[vJ + 0] = ggml_cuda_cast<dst_t>(cD * kvalues_iq4nl[vQ4[vJ] & 0xf]);
    vY[vJ + 16] = ggml_cuda_cast<dst_t>(cD * kvalues_iq4nl[vQ4[vJ] >> 4]);
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                         const int pTid) {
  const block_iq4_xs *vX = (const block_iq4_xs *)pVx;

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 4 * cIl;
  const uint8_t *vQ4 = vX[pIbs].qs + 16 * cIb + 4 * cIl;
  const float cD =
      (float)vX[pIbs].d *
      ((((vX[pIbs].scales_l[cIb / 2] >> 4 * (cIb % 2)) & 0xf) | (((vX[pIbs].scales_h >> 2 * cIb) & 3) << 4)) - 32);
  for (int vJ = 0; vJ < 4; ++vJ) {
    vY[vJ + 0] = ggml_cuda_cast<dst_t>(cD * kvalues_iq4nl[vQ4[vJ] & 0xf]);
    vY[vJ + 16] = ggml_cuda_cast<dst_t>(cD * kvalues_iq4nl[vQ4[vJ] >> 4]);
  }
}

template <typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void *pVx, const int64_t pIbs, dst_t *pYy,
                                                        const int pTid) {

  const block_mxfp4 *vX = (const block_mxfp4 *)pVx + pIbs * (QK_K / QK_MXFP4);

  const int64_t cIl = pTid / 8; // 0...3
  const int64_t cIb = pTid % 8; // 0...7
  dst_t *vY = pYy + 32 * cIb + 4 * cIl;
  const uint8_t *vQ4 = vX[cIb].qs + 4 * cIl;
  const float cD = ggml_cuda_e8m0_to_fp32(vX[cIb].e);
  for (int vJ = 0; vJ < 4; ++vJ) {
    vY[vJ + 0] = ggml_cuda_cast<dst_t>(cD * kvalues_mxfp4[vQ4[vJ] & 0xf] * 0.5f);
    vY[vJ + 16] = ggml_cuda_cast<dst_t>(cD * kvalues_mxfp4[vQ4[vJ] >> 4] * 0.5f);
  }
}
