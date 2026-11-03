#pragma once

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ int get_int_b1(const void *pX, const int &pI32) {
  const uint8_t *vX8 = (const uint8_t *)pX;

  int vX32 = vX8[4 * pI32 + 0] << 0;
  vX32 |= vX8[4 * pI32 + 1] << 8;
  vX32 |= vX8[4 * pI32 + 2] << 16;
  vX32 |= vX8[4 * pI32 + 3] << 24;

  return vX32;
}

static __device__ __forceinline__ int get_int_b2(const void *pX, const int &pI32) {
  const uint16_t *vX16 = (const uint16_t *)pX; // assume at least 2 byte alignment

  int vX32 = vX16[2 * pI32 + 0] << 0;
  vX32 |= vX16[2 * pI32 + 1] << 16;

  return vX32;
}

static __device__ __forceinline__ int get_int_b4(const void *pX, const int &pI32) {
  return ((const int *)pX)[pI32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int &pQ4, const int8_t *pTable) {
#if defined(GGML_USE_HIP)
  // Load the 16-byte table into four 32-bit unsigned integers.
  const uint32_t *values = (const uint32_t *)pTable;

  const uint32_t q_even = pQ4;
  const uint32_t q_odd = (pQ4 >> 4);

  // Perform lookups in the lower half of the table (indices 0-7).
  uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
  uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);

  // Perform lookups in the upper half of the table (indices 8-15).
  uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
  uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

  // Select between the low and high results based on the MSB of each index nibble.
  uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
  uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
  uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
  uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

  return make_int2(res_x, res_y);
#else
  // CUDA does not have an instruction for selecting bytes with 4 bit indices.
  // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
  const uint32_t *vTable32 = (const uint32_t *)pTable;

  // __byte_perm selects bytes based on the lower 16 bits in its third argument.
  // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
  // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3
  // bits. Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
  uint32_t vTmp[2];
  const uint32_t cLowHighSelectionIndices = (0x32103210 | ((pQ4 & 0x88888888) >> 1));
#pragma unroll
  for (uint32_t vI = 0; vI < 2; ++vI) {
    const uint32_t cShift = 16 * vI;

    const uint32_t cLow = __byte_perm(vTable32[0], vTable32[1], pQ4 >> cShift);
    const uint32_t cHigh = __byte_perm(vTable32[2], vTable32[3], pQ4 >> cShift);
    vTmp[vI] = __byte_perm(cLow, cHigh, cLowHighSelectionIndices >> cShift);
  }

  // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
  // However, for the result we need ints with all even/odd 4 bit indices in q4.
  // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
  return make_int2(__byte_perm(vTmp[0], vTmp[1], 0x6420), __byte_perm(vTmp[0], vTmp[1], 0x7531));
#endif
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t pV) {
  // v is a 7 bit int, with the 8th sign being encodable as popcnt
  // with xor we can "correct" the bit instead of having to mask
  const uint32_t cP = __popc(pV) & 1;
  const uint32_t cS = pV ^ cP << 7;
  // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
  return cS * 0x01010101;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q1_0_Q8_1_MMVQ 1 // Process one 32-element chunk at a time for parallelism
#define VDR_Q1_0_Q8_1_MMQ 4  // Q1_0 has 128 bits (4 ints) per block

#define VDR_Q2_0_Q8_1_MMVQ 1 // Process one 32-element chunk at a time for parallelism
#define VDR_Q2_0_Q8_1_MMQ 2  // Q2_0 group 64: 128 bits (4 ints) per block, 2 32-element chunks

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ 4

template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(const int *pV, const int *pU, const float &pD4,
                                                               const half2 &pDs8) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    const int cVi0 = (pV[vI] >> 0) & 0x0F0F0F0F;
    const int cVi1 = (pV[vI] >> 4) & 0x0F0F0F0F;

    // SIMD dot product of quantized values
    vSumi = ggml_cuda_dp4a(cVi0, pU[2 * vI + 0], vSumi);
    vSumi = ggml_cuda_dp4a(cVi1, pU[2 * vI + 1], vSumi);
  }

  const float2 cDs8f = __half22float2(pDs8);

  // second part effectively subtracts 8 from each quant value
  return pD4 * (vSumi * cDs8f.x - (8 * vdr / QI4_0) * cDs8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ 4

template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(const int *pV, const int *pU, const half2 &pDm4,
                                                               const half2 &pDs8) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    const int cVi0 = (pV[vI] >> 0) & 0x0F0F0F0F;
    const int cVi1 = (pV[vI] >> 4) & 0x0F0F0F0F;

    // SIMD dot product of quantized values
    vSumi = ggml_cuda_dp4a(cVi0, pU[2 * vI + 0], vSumi);
    vSumi = ggml_cuda_dp4a(cVi1, pU[2 * vI + 1], vSumi);
  }

#ifdef FAST_FP16_AVAILABLE
  const float2 cTmp = __half22float2(__hmul2(pDm4, pDs8));
  const float cD4d8 = cTmp.x;
  const float cM4s8 = cTmp.y;
#else
  const float2 cDm4f = __half22float2(pDm4);
  const float2 cDs8f = __half22float2(pDs8);
  const float cD4d8 = cDm4f.x * cDs8f.x;
  const float cM4s8 = cDm4f.y * cDs8f.y;
#endif // FAST_FP16_AVAILABLE

  // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
  return vSumi * cD4d8 + cM4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ 4

template <int vdr>
static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(const int *pVl, const int *pVh, const int *pU,
                                                               const float &pD5, const half2 &pDs8) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    int vVi0 = (pVl[vI] >> 0) & 0x0F0F0F0F;              // lower 4 qs bits, still need qh as 5th bits
    vVi0 |= (pVh[vI] << 4) & 0x00000010;                 // 0 ->  4
    vVi0 |= (pVh[vI] << 11) & 0x00001000;                // 1 -> 12
    vVi0 |= (pVh[vI] << 18) & 0x00100000;                // 2 -> 20
    vVi0 |= (pVh[vI] << 25) & 0x10000000;                // 3 -> 28
    vSumi = ggml_cuda_dp4a(vVi0, pU[2 * vI + 0], vSumi); // SIMD dot product of quantized values

    int vVi1 = (pVl[vI] >> 4) & 0x0F0F0F0F;              // upper 4 qs bits, still need qh as 5th bits
    vVi1 |= (pVh[vI] >> 12) & 0x00000010;                // 16 ->  4
    vVi1 |= (pVh[vI] >> 5) & 0x00001000;                 // 17 -> 12
    vVi1 |= (pVh[vI] << 2) & 0x00100000;                 // 18 -> 20
    vVi1 |= (pVh[vI] << 9) & 0x10000000;                 // 19 -> 28
    vSumi = ggml_cuda_dp4a(vVi1, pU[2 * vI + 1], vSumi); // SIMD dot product of quantized values
  }

  const float2 cDs8f = __half22float2(pDs8);

  // second part effectively subtracts 16 from each quant value
  return pD5 * (vSumi * cDs8f.x - (16 * vdr / QI5_0) * cDs8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ 4

template <int vdr>
static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(const int *pVl, const int *pVh, const int *pU,
                                                               const half2 &pDm5, const half2 &pDs8) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    int vVi0 = (pVl[vI] >> 0) & 0x0F0F0F0F;              // lower 4 qs bits, still need qh as 5th bits
    vVi0 |= (pVh[vI] << 4) & 0x00000010;                 // 0 ->  4
    vVi0 |= (pVh[vI] << 11) & 0x00001000;                // 1 -> 12
    vVi0 |= (pVh[vI] << 18) & 0x00100000;                // 2 -> 20
    vVi0 |= (pVh[vI] << 25) & 0x10000000;                // 3 -> 28
    vSumi = ggml_cuda_dp4a(vVi0, pU[2 * vI + 0], vSumi); // SIMD dot product of quantized values

    int vVi1 = (pVl[vI] >> 4) & 0x0F0F0F0F;              // upper 4 qs bits, still need qh as 5th bits
    vVi1 |= (pVh[vI] >> 12) & 0x00000010;                // 16 ->  4
    vVi1 |= (pVh[vI] >> 5) & 0x00001000;                 // 17 -> 12
    vVi1 |= (pVh[vI] << 2) & 0x00100000;                 // 18 -> 20
    vVi1 |= (pVh[vI] << 9) & 0x10000000;                 // 19 -> 28
    vSumi = ggml_cuda_dp4a(vVi1, pU[2 * vI + 1], vSumi); // SIMD dot product of quantized values
  }

#ifdef FAST_FP16_AVAILABLE
  const float2 cTmp = __half22float2(__hmul2(pDm5, pDs8));
  const float cD5d8 = cTmp.x;
  const float cM5s8 = cTmp.y;
#else
  const float2 cDm5f = __half22float2(pDm5);
  const float2 cDs8f = __half22float2(pDs8);
  const float cD5d8 = cDm5f.x * cDs8f.x;
  const float cM5s8 = cDm5f.y * cDs8f.y;
#endif // FAST_FP16_AVAILABLE

  // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
  return vSumi * cD5d8 + cM5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <typename T, int vdr>
static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(const int *pV, const int *pU, const T &pD80, const T &pD81) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    // SIMD dot product of quantized values
    vSumi = ggml_cuda_dp4a(pV[vI], pU[vI], vSumi);
  }

  return pD80 * pD81 * ((T)vSumi);
}

template <int vdr>
static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(const int *pV, const int *pU, const half2 &pDm8,
                                                               const half2 &pDs8) {

  int vSumi = 0;

#pragma unroll
  for (int vI = 0; vI < vdr; ++vI) {
    // SIMD dot product of quantized values
    vSumi = ggml_cuda_dp4a(pV[vI], pU[vI], vSumi);
  }

#ifdef FAST_FP16_AVAILABLE
  const float2 cTmp = __half22float2(__hmul2(pDm8, pDs8));
  const float cD8d8 = cTmp.x;
  const float cM8s8 = cTmp.y;
#else
  const float2 cDm8f = __half22float2(pDm8);
  const float2 cDs8f = __half22float2(pDs8);
  const float cD8d8 = cDm8f.x * cDs8f.x;
  const float cM8s8 = cDm8f.y * cDs8f.y;
#endif // FAST_FP16_AVAILABLE

  // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
  return vSumi * cD8d8 + cM8s8 / (QI8_1 / vdr);
}

template <int vdr>
static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_impl(const int *pV, const int *pU, const float *pD80,
                                                                  const float &pD81) {

  float vSumf = 0.0f;

#pragma unroll
  for (int vI0 = 0; vI0 < vdr; vI0 += QI8_0 / 2) {
    int vSumi = 0;

#pragma unroll
    for (int vI = vI0; vI < vI0 + QI8_0 / 2; ++vI) {
      // SIMD dot product of quantized values
      vSumi = ggml_cuda_dp4a(pV[vI], pU[vI], vSumi);
    }

    vSumf += pD80[vI0 / (QI8_0 / 2)] * vSumi;
  }

  return pD81 * vSumf;
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ 4

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                           const int &pIqs) {

  const block_mxfp4 *vBq4 = (const block_mxfp4 *)pVbq + pKbx;

  const int *vQ8 = (const int *)pBq81->qs + pIqs;

  int vSumi = 0;
#pragma unroll
  for (int vL = 0; vL < VDR_MXFP4_Q8_1_MMVQ; ++vL) {
    const int cAuxQ4 = get_int_b1(vBq4->qs, pIqs + vL);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_mxfp4);

    vSumi = ggml_cuda_dp4a(cV.x, vQ8[vL + 0], vSumi);
    vSumi = ggml_cuda_dp4a(cV.y, vQ8[vL + 4], vSumi);
  }

  const float cD = ggml_cuda_e8m0_to_fp32(vBq4->e) * 0.5f * __low2float(pBq81->ds);
  return cD * vSumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ 8

static __device__ __forceinline__ float vec_dot_nvfp4_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int32_t &pKbx,
                                                           const int32_t &pIqs) {

  const block_nvfp4 *vBq4 = (const block_nvfp4 *)pVbq + pKbx;
  float vSum = 0.0f;
#pragma unroll
  for (int vI = 0; vI < VDR_NVFP4_Q8_1_MMVQ / 2; vI++) {
    const int32_t cIqs0 = pIqs + 2 * vI;
    const int32_t cIqs1 = cIqs0 + 1;
    const int32_t cIs = cIqs0 >> 1;
    const int2 cV0 = get_int_from_table_16(get_int_b4(vBq4->qs, cIqs0), kvalues_mxfp4);
    const int2 cV1 = get_int_from_table_16(get_int_b4(vBq4->qs, cIqs1), kvalues_mxfp4);
    const block_q8_1 *vBq8 = pBq81 + (cIs >> 1);
    const int32_t cI8 = ((cIs & 1) << 2);

    int vSumi = ggml_cuda_dp4a(cV0.x, get_int_b4(vBq8->qs, cI8 + 0), 0);
    vSumi = ggml_cuda_dp4a(cV0.y, get_int_b4(vBq8->qs, cI8 + 2), vSumi);
    vSumi = ggml_cuda_dp4a(cV1.x, get_int_b4(vBq8->qs, cI8 + 1), vSumi);
    vSumi = ggml_cuda_dp4a(cV1.y, get_int_b4(vBq8->qs, cI8 + 3), vSumi);

    const float cD = ggml_cuda_ue4m3_to_fp32(vBq4->d[cIs]) * __low2float(vBq8->ds);
    vSum += cD * float(vSumi);
  }

  return vSum;
}
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ 4

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(const int &pV, const int *__restrict__ pU,
                                                                    const uint8_t *__restrict__ pScales,
                                                                    const half2 &pDm2, const float *__restrict__ pD8) {

  float vSumfD = 0.0f;
  float vSumfM = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR2_K; ++vI) {
    const int cSc = pScales[2 * vI];

    const int cVi = (pV >> (2 * vI)) & 0x03030303;

    vSumfD += pD8[vI] * (ggml_cuda_dp4a(cVi, pU[vI], 0) * (cSc & 0xF)); // SIMD dot product

    // fill int with 4x m
    int vM = cSc >> 4;
    vM |= vM << 8;
    vM |= vM << 16;
    vSumfM += pD8[vI] * ggml_cuda_dp4a(vM, pU[vI], 0); // multiply constant q2_K part with sum of q8_1 values
  }

  const float2 cDm2f = __half22float2(pDm2);

  return cDm2f.x * vSumfD - cDm2f.y * vSumfM;
}

// contiguous v/x + u/y values
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(const int *__restrict__ pV,
                                                                   const int *__restrict__ pU, const half2 *pDm2,
                                                                   const float &pD8, const half2 *pS8) {

  float vSumf = 0.0f;
  float vSumfD8 = 0.0f;

#pragma unroll
  for (int vI0 = 0; vI0 < QR2_K * VDR_Q2_K_Q8_1_MMQ; vI0 += QI8_1) {
    const float2 cDm2f0 = __half22float2(pDm2[vI0 / (QI8_1 / 2) + 0]);
    int vSumiD0 = 0;

    const float2 cDm2f1 = __half22float2(pDm2[vI0 / (QI8_1 / 2) + 1]);
    int vSumiD1 = 0;

#pragma unroll
    for (int vI = vI0; vI < vI0 + QI8_1 / 2; ++vI) {
      vSumiD0 = ggml_cuda_dp4a(pV[vI], pU[vI], vSumiD0);
    }
    vSumfD8 += cDm2f0.x * vSumiD0;

#pragma unroll
    for (int vI = vI0 + QI8_1 / 2; vI < vI0 + QI8_1; ++vI) {
      vSumiD1 = ggml_cuda_dp4a(pV[vI], pU[vI], vSumiD1);
    }
    vSumfD8 += cDm2f1.x * vSumiD1;

    if (vI0 / QI8_1 < ns8) {
      const float2 cS8f = __half22float2(pS8[vI0 / QI8_1]);
      vSumf -= cDm2f0.y * cS8f.x;
      vSumf -= cDm2f1.y * cS8f.y;
    } else {
      int vSumiM0 = 0;
#pragma unroll
      for (int vI = vI0; vI < vI0 + QI8_1 / 2; ++vI) {
        vSumiM0 = ggml_cuda_dp4a(0x01010101, pU[vI], vSumiM0);
      }
      vSumfD8 -= cDm2f0.y * vSumiM0;

      int vSumiM1 = 0;
#pragma unroll
      for (int vI = vI0 + QI8_1 / 2; vI < vI0 + QI8_1; ++vI) {
        vSumiM1 = ggml_cuda_dp4a(0x01010101, pU[vI], vSumiM1);
      }
      vSumfD8 -= cDm2f1.y * vSumiM1;
    }
  }

  return vSumf + pD8 * vSumfD8;
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ 2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(const int &pVl, const int &pVh,
                                                                    const int *__restrict__ pU,
                                                                    const uint8_t *__restrict__ pScales,
                                                                    const int &pScaleOffset, const float &pD3,
                                                                    const float *__restrict__ pD8) {

  float vSumf = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR3_K; ++vI) {
    const int cIsc = pScaleOffset + 2 * vI;

    const int cIscLow = cIsc % (QK_K / 32);
    const int cScShiftLow = 4 * (cIsc / (QK_K / 32));
    const int cScLow = (pScales[cIscLow] >> cScShiftLow) & 0xF;

    const int cIscHigh = cIsc % (QK_K / 64);
    const int cScShiftHigh = 2 * (cIsc / (QK_K / 64));
    const int cScHigh = ((pScales[(QK_K / 32) + cIscHigh] >> cScShiftHigh) & 3) << 4;

    const int cSc = (cScLow | cScHigh) - 32;

    const int cVil = (pVl >> (2 * vI)) & 0x03030303;

    const int cVih = ((pVh >> vI) << 2) & 0x04040404;

    const int cVi = __vsubss4(cVil, cVih);

    vSumf += pD8[vI] * (ggml_cuda_dp4a(cVi, pU[vI], 0) * cSc); // SIMD dot product
  }

  return pD3 * vSumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(const int *__restrict__ pV,
                                                                   const int *__restrict__ pU,
                                                                   const int8_t *__restrict__ pScales, const float &pD3,
                                                                   const float &pD8) {

  int vSumi = 0;

#pragma unroll
  for (int vI0 = 0; vI0 < QR3_K * VDR_Q3_K_Q8_1_MMQ; vI0 += QI8_1 / 2) {
    int vSumiSc = 0;

#pragma unroll
    for (int vI = vI0; vI < vI0 + QI8_1 / 2; ++vI) {
      vSumiSc = ggml_cuda_dp4a(pV[vI], pU[vI], vSumiSc); // SIMD dot product
    }

    vSumi += vSumiSc * pScales[vI0 / (QI8_1 / 2)];
  }

  return pD3 * pD8 * vSumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ 8

// contiguous v/x values
static __device__ __forceinline__ float
vec_dot_q4_K_q8_1_impl_vmmq(const int *__restrict__ pV, const int *__restrict__ pU, const uint8_t *__restrict__ pSc,
                            const uint8_t *__restrict__ pM, const half2 &pDm4, const float *__restrict__ pD8) {

  float vSumfD = 0.0f;
  float vSumfM = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR4_K; ++vI) {
    const int cV0i = (pV[0] >> (4 * vI)) & 0x0F0F0F0F;
    const int cV1i = (pV[1] >> (4 * vI)) & 0x0F0F0F0F;

    const int cDot1 = ggml_cuda_dp4a(cV1i, pU[2 * vI + 1], ggml_cuda_dp4a(cV0i, pU[2 * vI + 0], 0)); // SIMD dot product
    const int cDot2 =
        ggml_cuda_dp4a(0x01010101, pU[2 * vI + 1], ggml_cuda_dp4a(0x01010101, pU[2 * vI + 0], 0)); // sum of u

    vSumfD += pD8[vI] * (cDot1 * pSc[vI]);
    vSumfM += pD8[vI] * (cDot2 * pM[vI]); // multiply constant part of q4_K with sum of q8_1 values
  }

  const float2 cDm4f = __half22float2(pDm4);

  return cDm4f.x * vSumfD - cDm4f.y * vSumfM;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float
vec_dot_q4_K_q8_1_impl_mmq(const int *__restrict__ pV, const int *__restrict__ pU, const uint8_t *__restrict__ pSc,
                           const uint8_t *__restrict__ pM, const half2 &pDm4, const half2 *__restrict__ pDs8) {

  float vSumfD = 0.0f;
  float vSumfM = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; ++vI) {
    int vSumiD = 0;

#pragma unroll
    for (int vJ = 0; vJ < QI8_1; ++vJ) {
      vSumiD = ggml_cuda_dp4a((pV[vJ] >> (4 * vI)) & 0x0F0F0F0F, pU[vI * QI8_1 + vJ], vSumiD); // SIMD dot product
    }

    const float2 cDs8f = __half22float2(pDs8[vI]);

    vSumfD += cDs8f.x * (pSc[vI] * vSumiD);
    vSumfM += cDs8f.y * pM[vI]; // sum of q8_1 block * q4_K min val
  }

  const float2 cDm4f = __half22float2(pDm4);

  return cDm4f.x * vSumfD - cDm4f.y * vSumfM;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ 8

// contiguous v/x values
static __device__ __forceinline__ float
vec_dot_q5_K_q8_1_impl_vmmq(const int *__restrict__ pVl, const int *__restrict__ pVh, const int *__restrict__ pU,
                            const uint8_t *__restrict__ pSc, const uint8_t *__restrict__ pM, const half2 &pDm5,
                            const float *__restrict__ pD8) {

  float vSumfD = 0.0f;
  float vSumfM = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR5_K; ++vI) {
    const int cVl0i = (pVl[0] >> (4 * vI)) & 0x0F0F0F0F;
    const int cVl1i = (pVl[1] >> (4 * vI)) & 0x0F0F0F0F;

    const int cVh0i = ((pVh[0] >> vI) << 4) & 0x10101010;
    const int cVh1i = ((pVh[1] >> vI) << 4) & 0x10101010;

    const int cV0i = cVl0i | cVh0i;
    const int cV1i = cVl1i | cVh1i;

    const int cDot1 = ggml_cuda_dp4a(cV0i, pU[2 * vI + 0], ggml_cuda_dp4a(cV1i, pU[2 * vI + 1], 0)); // SIMD dot product
    const int cDot2 =
        ggml_cuda_dp4a(0x01010101, pU[2 * vI + 0], ggml_cuda_dp4a(0x01010101, pU[2 * vI + 1], 0)); // sum of u

    vSumfD += pD8[vI] * (cDot1 * pSc[vI]);
    vSumfM += pD8[vI] * (cDot2 * pM[vI]);
  }

  const float2 cDm5f = __half22float2(pDm5);

  return cDm5f.x * vSumfD - cDm5f.y * vSumfM;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float
vec_dot_q5_K_q8_1_impl_mmq(const int *__restrict__ pV, const int *__restrict__ pU, const uint8_t *__restrict__ pSc,
                           const uint8_t *__restrict__ pM, const half2 &pDm4, const half2 *__restrict__ pDs8) {

  float vSumfD = 0.0f;
  float vSumfM = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR5_K * VDR_Q5_K_Q8_1_MMQ / QI8_1; ++vI) {
    int vSumiD = 0;

#pragma unroll
    for (int vJ = 0; vJ < QI8_1; ++vJ) {
      vSumiD = ggml_cuda_dp4a(pV[vI * QI8_1 + vJ], pU[vI * QI8_1 + vJ], vSumiD); // SIMD dot product
    }

    const float2 cDs8f = __half22float2(pDs8[vI]);

    vSumfD += cDs8f.x * (pSc[vI] * vSumiD);
    vSumfM += cDs8f.y * pM[vI]; // sum of q8_1 block * q4_K min val
  }

  const float2 cDm4f = __half22float2(pDm4);

  return cDm4f.x * vSumfD - cDm4f.y * vSumfM;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ 8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(const int &pVl, const int &pVh,
                                                                    const int *__restrict__ pU,
                                                                    const int8_t *__restrict__ pScales, const float &pD,
                                                                    const float *__restrict__ pD8) {

  float vSumf = 0.0f;

#pragma unroll
  for (int vI = 0; vI < QR6_K; ++vI) {
    const int cSc = pScales[4 * vI];

    const int cVil = (pVl >> (4 * vI)) & 0x0F0F0F0F;

    const int cVih = ((pVh >> (4 * vI)) << 4) & 0x30303030;

    const int cVi = __vsubss4((cVil | cVih), 0x20202020); // vi = (vil | vih) - 32

    vSumf += pD8[vI] * (ggml_cuda_dp4a(cVi, pU[vI], 0) * cSc); // SIMD dot product
  }

  return pD * vSumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(const int *__restrict__ pV,
                                                                   const int *__restrict__ pU,
                                                                   const int8_t *__restrict__ pSc, const float &pD6,
                                                                   const float *__restrict__ pD8) {

  float vSumfD = 0.0f;

  const int cScPacked = get_int_b4(pSc, 0);
  const int8_t *vScReg = (const int8_t *)&cScPacked;

#pragma unroll
  for (int vI0 = 0; vI0 < VDR_Q6_K_Q8_1_MMQ; vI0 += 4) {
    int2 vSumiD = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
    for (int vI = vI0; vI < vI0 + 2; ++vI) {
      vSumiD.x = ggml_cuda_dp4a(pV[2 * vI + 0], pU[2 * vI + 0], vSumiD.x); // SIMD dot product
      vSumiD.x = ggml_cuda_dp4a(pV[2 * vI + 1], pU[2 * vI + 1], vSumiD.x); // SIMD dot product

      vSumiD.y = ggml_cuda_dp4a(pV[2 * vI + 4], pU[2 * vI + 4], vSumiD.y); // SIMD dot product
      vSumiD.y = ggml_cuda_dp4a(pV[2 * vI + 5], pU[2 * vI + 5], vSumiD.y); // SIMD dot product
    }

    vSumfD += pD8[vI0 / 4] * (vScReg[vI0 / 2 + 0] * vSumiD.x + vScReg[vI0 / 2 + 1] * vSumiD.y);
  }

  return pD6 * vSumfD;
}

static __device__ __forceinline__ float vec_dot_q1_0_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q1_0 *vBq10 = (const block_q1_0 *)pVbq + pKbx;

  // Q1_0: 128 elements with ONE scale
  // Q8_1: 32 elements per block with individual scales
  // iqs selects which of the 4 chunks of 32 elements to process (0-3)

  const float cD1 = vBq10->d;
  const int16_t *vQs = (const int16_t *)vBq10->qs + pIqs * 2;

  // Process only the chunk specified by iqs
  const block_q8_1 *vBq81Chunk = pBq81 + pIqs;

  int vSumi = 0;
#pragma unroll
  for (int vJ = 0; vJ < 2; ++vJ) {
    const int cQ = vQs[vJ];

    const int cU0 = get_int_b4(vBq81Chunk->qs, vJ * 4 + 0);
    const int cU1 = get_int_b4(vBq81Chunk->qs, vJ * 4 + 1);
    const int cU2 = get_int_b4(vBq81Chunk->qs, vJ * 4 + 2);
    const int cU3 = get_int_b4(vBq81Chunk->qs, vJ * 4 + 3);

    // unpack crumbs into nibble indices
    const int cN0 = __byte_perm(0x11100100, 0x11100100, cQ >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
    const int cN1 = __byte_perm(0x11100100, 0x11100100, cQ >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
    // unpack nibbles into byte values
    const int cS0 = __byte_perm(0x01FF, 0x01FF, cN0 >> 0);
    const int cS1 = __byte_perm(0x01FF, 0x01FF, cN1 >> 0);
    const int cS2 = __byte_perm(0x01FF, 0x01FF, cN0 >> 16);
    const int cS3 = __byte_perm(0x01FF, 0x01FF, cN1 >> 16);
    // unshuffle values
    const int cV0 = __byte_perm(cS0, cS1, 0x5410);
    const int cV1 = __byte_perm(cS0, cS1, 0x7632);
    const int cV2 = __byte_perm(cS2, cS3, 0x5410);
    const int cV3 = __byte_perm(cS2, cS3, 0x7632);

    vSumi = ggml_cuda_dp4a(cV0, cU0, vSumi);
    vSumi = ggml_cuda_dp4a(cV1, cU1, vSumi);
    vSumi = ggml_cuda_dp4a(cV2, cU2, vSumi);
    vSumi = ggml_cuda_dp4a(cV3, cU3, vSumi);
  }

  // Apply Q1_0's single scale and this chunk's Q8_1 scale
  const float cD8 = __low2float(vBq81Chunk->ds);
  return cD1 * cD8 * vSumi;
}

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q2_0 *vBq20 = (const block_q2_0 *)pVbq + pKbx;

  // Q2_0 (group 64): 64 elements with ONE scale, 2 bits per element (4 elements per byte)
  // Q8_1: 32 elements per block with individual scales
  // iqs selects which of the 2 chunks of 32 elements to process (0-1)

  const float cD2 = vBq20->d;
  const int16_t *vQs = (const int16_t *)vBq20->qs + pIqs * 4;

  // Process only the chunk specified by iqs
  const block_q8_1 *vBq81Chunk = pBq81 + pIqs;

  int vSumi = 0;
#pragma unroll
  for (int vJ = 0; vJ < 4; ++vJ) {
    const int cQ = vQs[vJ];
    const int cU = get_int_b4(vBq81Chunk->qs, vJ * 2 + 0);
    const int cV = get_int_b4(vBq81Chunk->qs, vJ * 2 + 1);

    // unpack even and odd crumbs into byte values
    const int cQe = __byte_perm(0x020100FF, 0x020100FF, cQ >> 0);
    const int cQo = __byte_perm(0x020100FF, 0x020100FF, cQ >> 2);
    // unshuffle values
    const int cQx = __byte_perm(cQe, cQo, 0x5140);
    const int cQy = __byte_perm(cQe, cQo, 0x7362);

    vSumi = ggml_cuda_dp4a(cU, cQx, vSumi);
    vSumi = ggml_cuda_dp4a(cV, cQy, vSumi);
  }

  // Apply Q2_0's single scale and this chunk's Q8_1 scale
  const float cD8 = __low2float(vBq81Chunk->ds);
  return cD2 * cD8 * vSumi;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q4_0 *vBq40 = (const block_q4_0 *)pVbq + pKbx;

  int v_[VDR_Q4_0_Q8_1_MMVQ];
  int vU[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
  for (int vI = 0; vI < VDR_Q4_0_Q8_1_MMVQ; ++vI) {
    v_[vI] = get_int_b2(vBq40->qs, pIqs + vI);
    vU[2 * vI + 0] = get_int_b4(pBq81->qs, pIqs + vI);
    vU[2 * vI + 1] = get_int_b4(pBq81->qs, pIqs + vI + QI4_0);
  }

  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v_, vU, vBq40->d, pBq81->ds);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q4_1 *vBq41 = (const block_q4_1 *)pVbq + pKbx;

  int v_[VDR_Q4_1_Q8_1_MMVQ];
  int vU[2 * VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
  for (int vI = 0; vI < VDR_Q4_1_Q8_1_MMVQ; ++vI) {
    v_[vI] = get_int_b4(vBq41->qs, pIqs + vI);
    vU[2 * vI + 0] = get_int_b4(pBq81->qs, pIqs + vI);
    vU[2 * vI + 1] = get_int_b4(pBq81->qs, pIqs + vI + QI4_1);
  }

  return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v_, vU, vBq41->dm, pBq81->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q5_0 *vBq50 = (const block_q5_0 *)pVbq + pKbx;

  int vVl[VDR_Q5_0_Q8_1_MMVQ];
  int vVh[VDR_Q5_0_Q8_1_MMVQ];
  int vU[2 * VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
  for (int vI = 0; vI < VDR_Q5_0_Q8_1_MMVQ; ++vI) {
    vVl[vI] = get_int_b2(vBq50->qs, pIqs + vI);
    vVh[vI] = get_int_b2(vBq50->qh, 0) >> (4 * (pIqs + vI));
    vU[2 * vI + 0] = get_int_b4(pBq81->qs, pIqs + vI);
    vU[2 * vI + 1] = get_int_b4(pBq81->qs, pIqs + vI + QI5_0);
  }

  return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vVl, vVh, vU, vBq50->d, pBq81->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q5_1 *vBq51 = (const block_q5_1 *)pVbq + pKbx;

  int vVl[VDR_Q5_1_Q8_1_MMVQ];
  int vVh[VDR_Q5_1_Q8_1_MMVQ];
  int vU[2 * VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
  for (int vI = 0; vI < VDR_Q5_1_Q8_1_MMVQ; ++vI) {
    vVl[vI] = get_int_b4(vBq51->qs, pIqs + vI);
    vVh[vI] = get_int_b4(vBq51->qh, 0) >> (4 * (pIqs + vI));
    vU[2 * vI + 0] = get_int_b4(pBq81->qs, pIqs + vI);
    vU[2 * vI + 1] = get_int_b4(pBq81->qs, pIqs + vI + QI5_1);
  }

  return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vVl, vVh, vU, vBq51->dm, pBq81->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q8_0 *vBq80 = (const block_q8_0 *)pVbq + pKbx;

  int v_[VDR_Q8_0_Q8_1_MMVQ];
  int vU[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int vI = 0; vI < VDR_Q8_0_Q8_1_MMVQ; ++vI) {
    v_[vI] = get_int_b2(vBq80->qs, pIqs + vI);
    vU[vI] = get_int_b4(pBq81->qs, pIqs + vI);
  }

  return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v_, vU, vBq80->d, __low2half(pBq81->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q2_K *vBq2K = (const block_q2_K *)pVbq + pKbx;

  const int cBq8Offset = QR2_K * (pIqs / QI8_1);
  const int cScaleOffset = pIqs - pIqs % QI8_1 + (pIqs % QI8_1) / (QI8_1 / 2);

  const uint8_t *vScales = vBq2K->scales + cScaleOffset;

  const int cV = get_int_b4(vBq2K->qs, pIqs);
  int vU[QR2_K];
  float vD8[QR2_K];

#pragma unroll
  for (int vI = 0; vI < QR2_K; ++vI) {
    vU[vI] = get_int_b4(pBq81[cBq8Offset + vI].qs, pIqs % QI8_1);
    vD8[vI] = __low2float(pBq81[cBq8Offset + vI].ds);
  }

  return vec_dot_q2_K_q8_1_impl_mmvq(cV, vU, vScales, vBq2K->dm, vD8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q3_K *vBq3K = (const block_q3_K *)pVbq + pKbx;

  const int cBq8Offset = QR3_K * (pIqs / (QI3_K / 2));
  const int cScaleOffset = pIqs - pIqs % QI8_1 + (pIqs % QI8_1) / (QI8_1 / 2);

  const float cD = vBq3K->d;

  const int cVl = get_int_b2(vBq3K->qs, pIqs);

  // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
  const int cVh = ~get_int_b2(vBq3K->hmask, pIqs % (QI3_K / 2)) >> cBq8Offset;

  int vU[QR3_K];
  float vD8[QR3_K];

#pragma unroll
  for (int vI = 0; vI < QR3_K; ++vI) {
    vU[vI] = get_int_b4(pBq81[cBq8Offset + vI].qs, pIqs % QI8_1);
    vD8[vI] = __low2float(pBq81[cBq8Offset + vI].ds);
  }

  return vec_dot_q3_K_q8_1_impl_mmvq(cVl, cVh, vU, vBq3K->scales, cScaleOffset, cD, vD8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q4_K *vBq4K = (const block_q4_K *)pVbq + pKbx;

  int v_[2];
  int vU[2 * QR4_K];
  float vD8[QR4_K];

  // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
  const int cBq8Offset = QR4_K * ((pIqs / 2) / (QI8_1 / 2));

  // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
  // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
  // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
  // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

  const int *vQ4 = (const int *)(vBq4K->qs + 16 * cBq8Offset + 4 * ((pIqs / 2) % 4));
  v_[0] = vQ4[0];
  v_[1] = vQ4[4];

  // branchless so nvcc can hoist this out of the ncols_dst loop
  const uint16_t *vScales = (const uint16_t *)vBq4K->scales;
  const int cJ = cBq8Offset / 2;
  const int cJm = cJ & 1;

  const uint32_t cS0 = vScales[cJm + 0];
  const uint32_t cS2 = vScales[cJm + 2];
  const uint32_t cS4 = vScales[cJm + 4];

  const uint32_t cHi = (uint32_t)-(int32_t)(cJ >= 2);

  uint16_t vAux[2];
  vAux[0] = (uint16_t)(((cS0 & 0x3f3f) & ~cHi) | ((((cS4 >> 0) & 0x0f0f) | ((cS0 & 0xc0c0) >> 2)) & cHi));
  vAux[1] = (uint16_t)(((cS2 & 0x3f3f) & ~cHi) | ((((cS4 >> 4) & 0x0f0f) | ((cS2 & 0xc0c0) >> 2)) & cHi));
  const uint8_t *vSc = (const uint8_t *)vAux;
  const uint8_t *vM = vSc + 2;

  for (int vI = 0; vI < QR4_K; ++vI) {
    const block_q8_1 *vBq8i = pBq81 + cBq8Offset + vI;
    vD8[vI] = __low2float(vBq8i->ds);

    const int *vQ8 = (const int *)vBq8i->qs + ((pIqs / 2) % 4);
    vU[2 * vI + 0] = vQ8[0];
    vU[2 * vI + 1] = vQ8[4];
  }

  return vec_dot_q4_K_q8_1_impl_vmmq(v_, vU, vSc, vM, vBq4K->dm, vD8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q5_K *vBq5K = (const block_q5_K *)pVbq + pKbx;

  int vVl[2];
  int vVh[2];
  int vU[2 * QR5_K];
  float vD8[QR5_K];

  const int cBq8Offset = QR5_K * ((pIqs / 2) / (QI8_1 / 2));
  const int *vQl = (const int *)(vBq5K->qs + 16 * cBq8Offset + 4 * ((pIqs / 2) % 4));
  const int *vQh = (const int *)(vBq5K->qh + 4 * ((pIqs / 2) % 4));

  vVl[0] = vQl[0];
  vVl[1] = vQl[4];

  vVh[0] = vQh[0] >> cBq8Offset;
  vVh[1] = vQh[4] >> cBq8Offset;

  // same as q4_K
  const uint16_t *vScales = (const uint16_t *)vBq5K->scales;
  const int cJ = cBq8Offset / 2;
  const int cJm = cJ & 1;

  const uint32_t cS0 = vScales[cJm + 0];
  const uint32_t cS2 = vScales[cJm + 2];
  const uint32_t cS4 = vScales[cJm + 4];

  const uint32_t cHi = (uint32_t)-(int32_t)(cJ >= 2);

  uint16_t vAux[2];
  vAux[0] = (uint16_t)(((cS0 & 0x3f3f) & ~cHi) | ((((cS4 >> 0) & 0x0f0f) | ((cS0 & 0xc0c0) >> 2)) & cHi));
  vAux[1] = (uint16_t)(((cS2 & 0x3f3f) & ~cHi) | ((((cS4 >> 4) & 0x0f0f) | ((cS2 & 0xc0c0) >> 2)) & cHi));

  const uint8_t *vSc = (const uint8_t *)vAux;
  const uint8_t *vM = vSc + 2;

#pragma unroll
  for (int vI = 0; vI < QR5_K; ++vI) {
    const block_q8_1 *vBq8i = pBq81 + cBq8Offset + vI;
    vD8[vI] = __low2float(vBq8i->ds);

    const int *vQ8 = (const int *)vBq8i->qs + ((pIqs / 2) % 4);
    vU[2 * vI + 0] = vQ8[0];
    vU[2 * vI + 1] = vQ8[4];
  }

  return vec_dot_q5_K_q8_1_impl_vmmq(vVl, vVh, vU, vSc, vM, vBq5K->dm, vD8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(const void *__restrict__ pVbq,
                                                          const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                          const int &pIqs) {

  const block_q6_K *vBq6K = (const block_q6_K *)pVbq + pKbx;

  const int cBq8Offset = 2 * QR6_K * (pIqs / (QI6_K / 2)) + (pIqs % (QI6_K / 2)) / (QI6_K / 4);
  const int cScaleOffset = (QI6_K / 4) * (pIqs / (QI6_K / 2)) + (pIqs % (QI6_K / 2)) / (QI6_K / 8);
  const int cVhShift = 2 * ((pIqs % (QI6_K / 2)) / (QI6_K / 4));

  const int cVl = get_int_b2(vBq6K->ql, pIqs);
  const int cVh = get_int_b2(vBq6K->qh, (QI6_K / 4) * (pIqs / (QI6_K / 2)) + pIqs % (QI6_K / 4)) >> cVhShift;

  const int8_t *vScales = vBq6K->scales + cScaleOffset;

  int vU[QR6_K];
  float vD8[QR6_K];

#pragma unroll
  for (int vI = 0; vI < QR6_K; ++vI) {
    vU[vI] = get_int_b4(pBq81[cBq8Offset + 2 * vI].qs, pIqs % QI8_1);
    vD8[vI] = __low2float(pBq81[cBq8Offset + 2 * vI].ds);
  }

  return vec_dot_q6_K_q8_1_impl_mmvq(cVl, cVh, vU, vScales, vBq6K->d, vD8);
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ 2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void *__restrict__ pVbq,
                                                             const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                             const int &pIqs) {

  const block_iq2_xxs *vBq2 = (const block_iq2_xxs *)pVbq + pKbx;

  const int cQ2 = get_int_b2(vBq2->qs, pIqs);
  const uint8_t *vAux8 = (const uint8_t *)&cQ2;
  const uint32_t cAux32 = get_int_b2(vBq2->qs, pIqs + 1);

  int vSumi = 0;
#pragma unroll
  for (int vK0 = 0; vK0 < 8; vK0 += 2) {
    const uint2 cGridPos = ((const uint2 *)iq2xxs_grid)[vAux8[vK0 / 2]];
    const uint32_t cSigns = unpack_ksigns(cAux32 >> (7 * vK0 / 2));

    const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
    const int cGrid0 = __vsub4(cGridPos.x ^ cSigns0, cSigns0);
    const int cU0 = get_int_b4(pBq81[pIqs / 2].qs, vK0 + 0);
    vSumi = ggml_cuda_dp4a(cGrid0, cU0, vSumi);

    const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
    const int cGrid1 = __vsub4(cGridPos.y ^ cSigns1, cSigns1);
    const int cU1 = get_int_b4(pBq81[pIqs / 2].qs, vK0 + 1);
    vSumi = ggml_cuda_dp4a(cGrid1, cU1, vSumi);
  }

  const int cLs = cAux32 >> 27 | 1; // (scale * 2 + 1)
  vSumi = vSumi * cLs / 8;          // (sumi * scale + sumi / 2) / 4
  const float cD = __half2float(vBq2->d) * __low2float(pBq81[pIqs / 2].ds);
  return cD * vSumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2
#define VDR_IQ2_XS_Q8_1_MMQ 2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ pVbq,
                                                            const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                            const int &pIqs) {

  const block_iq2_xs *vBq2 = (const block_iq2_xs *)pVbq + pKbx;

  const int2 cQ2Packed = make_int2(get_int_b2(vBq2->qs, pIqs + 0), get_int_b2(vBq2->qs, pIqs + 1));
  const uint16_t *vQ2 = (const uint16_t *)&cQ2Packed;
  const int cLs0 = vBq2->scales[pIqs / 2] & 0x0F;
  const int cLs1 = vBq2->scales[pIqs / 2] >> 4;

  int vSumi0 = 0;
  int vSumi1 = 0;
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const uint2 cGridPos = ((const uint2 *)iq2xs_grid)[vQ2[vL0 / 2] & 0x1FF];
    const uint32_t cSigns = unpack_ksigns(vQ2[vL0 / 2] >> 9);

    const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
    const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);
    const int cU0 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 0);

    const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
    const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);
    const int cU1 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 1);

    if (vL0 < 4) {
      vSumi0 = ggml_cuda_dp4a(cGridL, cU0, vSumi0);
      vSumi0 = ggml_cuda_dp4a(cGridH, cU1, vSumi0);
    } else {
      vSumi1 = ggml_cuda_dp4a(cGridL, cU0, vSumi1);
      vSumi1 = ggml_cuda_dp4a(cGridH, cU1, vSumi1);
    }
  }
  const int cSumi = (vSumi0 * cLs0 + vSumi1 * cLs1 + (vSumi0 + vSumi1) / 2) / 4;
  const float cD = __half2float(vBq2->d) * __low2float(pBq81[pIqs / 2].ds);
  return cD * cSumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2
#define VDR_IQ2_S_Q8_1_MMQ 2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                           const int &pIqs) {

  const block_iq2_s *vBq2 = (const block_iq2_s *)pVbq + pKbx;

  const int cQsPacked = get_int_b2(vBq2->qs, pIqs / 2);
  const uint8_t *vQs = (const uint8_t *)&cQsPacked;

  const int cQh = vBq2->qh[pIqs / 2];

  const int cSignsPacked32 = get_int_b2(vBq2->qs, QK_K / 32 + pIqs / 2);
  const uint8_t *vSignsPacked8 = (const uint8_t *)&cSignsPacked32;

  const int cLs0 = vBq2->scales[pIqs / 2] & 0x0F;
  const int cLs1 = vBq2->scales[pIqs / 2] >> 4;

  int vSumi0 = 0;
  int vSumi1 = 0;
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const int *vGridPos = (const int *)(iq2s_grid + (vQs[vL0 / 2] | ((cQh << (8 - vL0)) & 0x300)));

    const int cSigns0 =
        __vcmpne4(((vSignsPacked8[vL0 / 2] & 0x03) << 7) | ((vSignsPacked8[vL0 / 2] & 0x0C) << 21), 0x00000000);
    const int cSigns1 =
        __vcmpne4(((vSignsPacked8[vL0 / 2] & 0x30) << 3) | ((vSignsPacked8[vL0 / 2] & 0xC0) << 17), 0x00000000);

    const int cGridL = __vsub4(vGridPos[0] ^ cSigns0, cSigns0);
    const int cGridH = __vsub4(vGridPos[1] ^ cSigns1, cSigns1);

    const int cU0 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 0);
    const int cU1 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 1);

    if (vL0 < 4) {
      vSumi0 = ggml_cuda_dp4a(cGridL, cU0, vSumi0);
      vSumi0 = ggml_cuda_dp4a(cGridH, cU1, vSumi0);
    } else {
      vSumi1 = ggml_cuda_dp4a(cGridL, cU0, vSumi1);
      vSumi1 = ggml_cuda_dp4a(cGridH, cU1, vSumi1);
    }
  }
  const int cSumi = (vSumi0 * cLs0 + vSumi1 * cLs1 + (vSumi0 + vSumi1) / 2) / 4;

  const float cD = __half2float(vBq2->d) * __low2float(pBq81[pIqs / 2].ds);
  return cD * cSumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define VDR_IQ3_XXS_Q8_1_MMQ 2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void *__restrict__ pVbq,
                                                             const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                             const int &pIqs) {

  const block_iq3_xxs *vBq3 = (const block_iq3_xxs *)pVbq + pKbx;

  const int2 cQ3Packed = make_int2(get_int_b2(vBq3->qs, pIqs), get_int_b2(vBq3->qs, pIqs + 1));
  const uint8_t *vQ3 = (const uint8_t *)&cQ3Packed;
  const uint32_t cAux32 = get_int_b2(vBq3->qs, QK_K / 16 + pIqs / 2);

  int vSumi = 0;
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const int2 cGridPos = make_int2(iq3xxs_grid[vQ3[vL0 + 0]], iq3xxs_grid[vQ3[vL0 + 1]]);
    const uint32_t cSigns = unpack_ksigns(cAux32 >> (7 * vL0 / 2));

    const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
    const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);

    const int cU0 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 0);

    const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
    const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

    const int cU1 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 1);

    vSumi = ggml_cuda_dp4a(cGridL, cU0, vSumi);
    vSumi = ggml_cuda_dp4a(cGridH, cU1, vSumi);
  }

  const int cLs = cAux32 >> 28;
  vSumi = (cLs * vSumi + vSumi / 2) / 2;
  const float cD = __half2float(vBq3->d) * __low2float(pBq81[pIqs / 2].ds);
  return cD * vSumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ 2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                           const int &pIqs) {

  const block_iq3_s *vBq3 = (const block_iq3_s *)pVbq + pKbx;

  const int2 cQsPacked = make_int2(get_int_b2(vBq3->qs, pIqs + 0), get_int_b2(vBq3->qs, pIqs + 1));
  const uint8_t *vQs = (const uint8_t *)&cQsPacked;

  const int cQh = vBq3->qh[pIqs / 2];

  const int cSignsPacked32 = get_int_b2(vBq3->signs, pIqs / 2);
  const uint8_t *vSignsPacked8 = (const uint8_t *)&cSignsPacked32;

  int vSumi = 0;
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const int2 cGridPos = make_int2(iq3s_grid[vQs[vL0 + 0] | ((cQh << (8 - vL0)) & 0x100)],
                                    iq3s_grid[vQs[vL0 + 1] | ((cQh << (7 - vL0)) & 0x100)]);

    const int cSigns0 =
        __vcmpne4(((vSignsPacked8[vL0 / 2] & 0x03) << 7) | ((vSignsPacked8[vL0 / 2] & 0x0C) << 21), 0x00000000);
    const int cSigns1 =
        __vcmpne4(((vSignsPacked8[vL0 / 2] & 0x30) << 3) | ((vSignsPacked8[vL0 / 2] & 0xC0) << 17), 0x00000000);

    const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);
    const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

    const int cU0 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 0);
    const int cU1 = get_int_b4(pBq81[pIqs / 2].qs, vL0 + 1);

    vSumi = ggml_cuda_dp4a(cGridL, cU0, vSumi);
    vSumi = ggml_cuda_dp4a(cGridH, cU1, vSumi);
  }

  vSumi *= 1 + 2 * ((vBq3->scales[pIqs / 4] >> ((pIqs << 1) & 0x04)) & 0x0F);

  const float cD = __half2float(vBq3->d) * __low2float(pBq81[pIqs / 2].ds);
  return cD * vSumi;
}

#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMQ 1

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                           const int &pIqs) {
  const block_iq1_s *vBq1 = (const block_iq1_s *)pVbq + pKbx;

  const int cQsPacked = get_int_b2(vBq1->qs, pIqs);
  const uint8_t *vQs = (const uint8_t *)&cQsPacked;

  const int cQh = vBq1->qh[pIqs];

  int vSumi = 0;
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const int cGrid = iq1s_grid_gpu[vQs[vL0 / 2] | (((cQh >> 3 * (vL0 / 2)) & 0x07) << 8)];

    const int cGrid0 = (cGrid >> 0) & 0x0F0F0F0F;
    const int cGrid1 = (cGrid >> 4) & 0x0F0F0F0F;

    const int cU0 = get_int_b4(pBq81[pIqs].qs, vL0 + 0);
    const int cU1 = get_int_b4(pBq81[pIqs].qs, vL0 + 1);

    vSumi = ggml_cuda_dp4a(cGrid0, cU0, vSumi);
    vSumi = ggml_cuda_dp4a(cGrid1, cU1, vSumi);
  }

  const float cD1q = __half2float(vBq1->d) * (((cQh >> 11) & 0x0E) + 1);
  const float cDelta = -1.0f + IQ1S_DELTA - (cQh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
  const float2 cDs = __half22float2(pBq81[pIqs].ds);
  return cD1q * (cDs.x * vSumi + cDs.y * cDelta);
}

#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMQ 1

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void *__restrict__ pVbq,
                                                           const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                           const int &pIqs) {

  const block_iq1_m *vBq1 = (const block_iq1_m *)pVbq + pKbx;

  const int cQsPacked = get_int_b4(vBq1->qs, pIqs);
  const uint8_t *vQs = (const uint8_t *)&cQsPacked;

  int vSumi[2] = {0};
  float vSumf[2] = {0.0f};
#pragma unroll
  for (int vL0 = 0; vL0 < 8; vL0 += 2) {
    const int cQhl = vBq1->qh[2 * pIqs + vL0 / 4] >> (4 * ((vL0 / 2) % 2));

    const int cGrid = iq1s_grid_gpu[vQs[vL0 / 2] | ((cQhl & 0x07) << 8)];

    const int cGrid0 = (cGrid >> 0) & 0x0F0F0F0F;
    const int cGrid1 = (cGrid >> 4) & 0x0F0F0F0F;

    const int cU0 = get_int_b4(pBq81[pIqs].qs, vL0 + 0);
    const int cU1 = get_int_b4(pBq81[pIqs].qs, vL0 + 1);

    vSumi[vL0 / 4] = ggml_cuda_dp4a(cGrid0, cU0, vSumi[vL0 / 4]);
    vSumi[vL0 / 4] = ggml_cuda_dp4a(cGrid1, cU1, vSumi[vL0 / 4]);

    const float cDelta = -1.0f + IQ1M_DELTA - (cQhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
    int vSumy = 0;
    vSumy = ggml_cuda_dp4a(cU0, 0x01010101, vSumy);
    vSumy = ggml_cuda_dp4a(cU1, 0x01010101, vSumy);
    vSumf[vL0 / 4] += cDelta * vSumy;
  }

  const uint16_t *vSc = (const uint16_t *)vBq1->scales;

  iq1m_scale_t vScale;
  vScale.u16 = (vSc[0] >> 12) | ((vSc[1] >> 8) & 0x00F0) | ((vSc[2] >> 4) & 0x0F00) | (vSc[3] & 0xF000);
  const float cD = __half2float(vScale.f16) * __low2float(pBq81[pIqs].ds);

  const int cTmp = vSc[pIqs / 2] >> (6 * (pIqs % 2));
  const int cSc0 = 2 * ((cTmp >> 0) & 0x07) + 1;
  const int cSc1 = 2 * ((cTmp >> 3) & 0x07) + 1;
  return cD * ((vSumi[0] + vSumf[0]) * cSc0 + (vSumi[1] + vSumf[1]) * cSc1);
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2
#define VDR_IQ4_NL_Q8_1_MMQ 4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void *__restrict__ pVbq,
                                                            const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                            const int &pIqs) {

  const block_iq4_nl *vBq4 = (const block_iq4_nl *)pVbq + pKbx;

  const int *vQ8 = (const int *)pBq81->qs + pIqs;

  int vSumi = 0;
#pragma unroll
  for (int vL = 0; vL < VDR_Q4_0_Q8_1_MMVQ; ++vL) {
    const int cAuxQ4 = get_int_b2(vBq4->qs, pIqs + vL);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_iq4nl);

    vSumi = ggml_cuda_dp4a(cV.x, vQ8[vL + 0], vSumi);
    vSumi = ggml_cuda_dp4a(cV.y, vQ8[vL + 4], vSumi);
  }

  const float cD = __half2float(vBq4->d) * __low2float(pBq81->ds);
  return cD * vSumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_IQ4_XS_Q8_1_MMQ 4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void *__restrict__ pVbq,
                                                            const block_q8_1 *__restrict__ pBq81, const int &pKbx,
                                                            const int &pIqs) {

  const block_iq4_xs *vBq4 = (const block_iq4_xs *)pVbq + pKbx;

  int vSumi = 0;
#pragma unroll
  for (int vJ = 0; vJ < 4; ++vJ) {
    const int cAuxQ4 = get_int_b4(vBq4->qs, pIqs + vJ);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_iq4nl);

    const int cU0 = get_int_b4(pBq81[pIqs / 4].qs, vJ + 0);
    const int cU1 = get_int_b4(pBq81[pIqs / 4].qs, vJ + 4);

    vSumi = ggml_cuda_dp4a(cV.x, cU0, vSumi);
    vSumi = ggml_cuda_dp4a(cV.y, cU1, vSumi);
  }

  const int cLs = ((vBq4->scales_l[pIqs / 8] >> (pIqs & 0x04)) & 0x0F) | (((vBq4->scales_h >> (pIqs / 2)) & 0x03) << 4);
  vSumi *= cLs - 32;

  const float cD = __half2float(vBq4->d) * __low2float(pBq81[pIqs / 4].ds);
  return cD * vSumi;
}
