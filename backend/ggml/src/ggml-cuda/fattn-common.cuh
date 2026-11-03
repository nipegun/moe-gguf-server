#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE 256
#define HALF_MAX_HALF                                                                                                  \
  __float2half(65504.0f /                                                                                              \
               2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should
// be negligible. Still, the value range should be shifted as much as necessary but as little as possible. The macro on
// the following line shifts it by a factor of 2**3=8.
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

typedef void (*fattn_kernel_t)(const char *__restrict__ pQ, const char *__restrict__ pK, const char *__restrict__ pV,
                               const char *__restrict__ pMask, const char *__restrict__ pSinks,
                               const int *__restrict__ pKvMax, float *__restrict__ pDst, float2 *__restrict__ pDstMeta,
                               const float pScale, const float pMaxBias, const float pM0, const float pM1,
                               const uint32_t pNHeadLog2, const float pLogitSoftcap, const int32_t pNe00,
                               const uint3 pNe01, const int32_t pNe02, const int32_t pNe03, const int32_t pNb01,
                               const int32_t pNb02, const int32_t pNb03, const int32_t pNe10, const int32_t pNe11,
                               const int32_t pNe12, const int32_t pNe13, const int32_t pNb11, const int32_t pNb12,
                               const int64_t pNb13, const int32_t pNb21, const int32_t pNb22, const int64_t pNb23,
                               const int32_t pNe31, const int32_t pNe32, const int32_t pNe33, const int32_t pNb31,
                               const int32_t pNb32, const int64_t pNb33);

typedef float (*vec_dot_KQ_t)(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                              const void *__restrict__ pQDs);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
  uintptr_t K;
  uintptr_t V;
  uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data
ggml_cuda_flash_attn_ext_get_f16_extra_data(const ggml_tensor *pDst, const bool pNeedF16K, const bool pNeedF16V) {
  GGML_ASSERT(pDst->op == GGML_OP_FLASH_ATTN_EXT);

  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];

  GGML_ASSERT(vK != nullptr);
  GGML_ASSERT(vV != nullptr);

  const bool cVIsKView =
      vV->view_src && (vV->view_src == vK || (vV->view_src == vK->view_src && vV->view_offs == vK->view_offs));

  ggml_cuda_flash_attn_ext_f16_extra_data vData = {};
  vData.end = (uintptr_t)pDst->data + ggml_nbytes(pDst);

  if (pNeedF16K && vK->type != GGML_TYPE_F16) {
    vData.end = GGML_PAD(vData.end, 128);
    vData.K = vData.end;
    vData.end += ggml_nelements(vK) * ggml_type_size(GGML_TYPE_F16);
  }

  if (pNeedF16V && vV->type != GGML_TYPE_F16) {
    if (cVIsKView) {
      vData.V = vData.K;
    } else {
      vData.end = GGML_PAD(vData.end, 128);
      vData.V = vData.end;
      vData.end += ggml_nelements(vV) * ggml_type_size(GGML_TYPE_F16);
    }
  }

  return vData;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_f16(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                         const void *__restrict__ pQDsV) {

  const half2 *vKH2 = (const half2 *)pKC;
  GGML_UNUSED(pQQ8);
  GGML_UNUSED(pQDsV);

  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < D / 2; vKKq0 += nthreads * cCpyNe) {
    __align__(16) half2 vTmp[cCpyNe];
    ggml_cuda_memcpy_1<sizeof(vTmp)>(vTmp, vKH2 + vKKq0 + (threadIdx.x % nthreads) * cCpyNe);
#pragma unroll
    for (int vKKq1 = 0; vKKq1 < cCpyNe; ++vKKq1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
      ggml_cuda_mad(vSum, vTmp[vKKq1], ((const half2 *)pQV)[vKKq0 / nthreads + vKKq1]);
#else
      ggml_cuda_mad(vSum, __half22float2(vTmp[vKKq1]), ((const float2 *)pQV)[vKKq0 / nthreads + vKKq1]);
#endif // V_DOT2_F32_F16_AVAILABLE
    }
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_bf16(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const nv_bfloat162 *vKBf16 = (const nv_bfloat162 *)pKC;
  GGML_UNUSED(pQQ8);
  GGML_UNUSED(pQDsV);

  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < D / 2; vKKq0 += nthreads * cCpyNe) {
    __align__(16) nv_bfloat162 vTmp[cCpyNe];
    ggml_cuda_memcpy_1<sizeof(vTmp)>(vTmp, vKBf16 + vKKq0 + (threadIdx.x % nthreads) * cCpyNe);
#pragma unroll
    for (int vKKq1 = 0; vKKq1 < cCpyNe; ++vKKq1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
      // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
      ggml_cuda_mad(vSum, ggml_cuda_cast<float2>(vTmp[vKKq1]),
                    __half22float2(((const half2 *)pQV)[vKKq0 / nthreads + vKKq1]));
#else
      ggml_cuda_mad(vSum, ggml_cuda_cast<float2>(vTmp[vKKq1]), ((const float2 *)pQV)[vKKq0 / nthreads + vKKq1]);
#endif // V_DOT2_F32_F16_AVAILABLE
    }
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_q4_0(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const block_q4_0 *vKQ40 = (const block_q4_0 *)pKC;
  GGML_UNUSED(pQV);

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < int(D / sizeof(int)); vKKq0 += nthreads) {
    const int cKKq = vKKq0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

    const int cIb = cKKq / QI8_1;
    const int cIqs4 = cKKq % QI4_0;
    const int cShift = cKKq & (QI8_1 / 2);

    int v_;
    ggml_cuda_memcpy_1<sizeof(int), 2>(&v_, vKQ40[cIb].qs + sizeof(int) * cIqs4);
    v_ = (v_ >> cShift) & 0x0F0F0F0F;
    const int cU = pQQ8[vKKq0 / nthreads];

    const int cSumi = ggml_cuda_dp4a(v_, cU, 0);

    const float2 cQDs = ((const float2 *)pQDsV)[vKKq0 / nthreads];
    vSum += __half2float(vKQ40[cIb].d) * (cSumi * cQDs.x - (8 / QI8_1) * cQDs.y);
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_q4_1(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const block_q4_1 *vKQ41 = (const block_q4_1 *)pKC;
  GGML_UNUSED(pQV);

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < int(D / sizeof(int)); vKKq0 += nthreads) {
    const int cKKq = vKKq0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

    const int cIb = cKKq / QI8_1;
    const int cIqs4 = cKKq % QI4_1;
    const int cShift = cKKq & (QI8_1 / 2);

    int v_;
    ggml_cuda_memcpy_1<sizeof(int)>(&v_, vKQ41[cIb].qs + sizeof(int) * cIqs4);
    v_ = (v_ >> cShift) & 0x0F0F0F0F;
    const int cU = pQQ8[vKKq0 / nthreads];

    const int cSumi = ggml_cuda_dp4a(v_, cU, 0);

    const float2 cKDm = __half22float2(vKQ41[cIb].dm);
    const float2 cQDs = ((const float2 *)pQDsV)[vKKq0 / nthreads];

    vSum += cKDm.x * cQDs.x * cSumi + cKDm.y * cQDs.y / QI8_1;
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_q5_0(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const block_q5_0 *vKQ50 = (const block_q5_0 *)pKC;
  GGML_UNUSED(pQV);

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < int(D / sizeof(int)); vKKq0 += nthreads) {
    const int cKKq = vKKq0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

    const int cIb = cKKq / QI8_1;
    const int cIqs4 = cKKq % QI5_0;
    const int cIqs8 = cKKq % QI8_1;
    const int cShift = cKKq & (QI8_1 / 2);

    int v_;
    ggml_cuda_memcpy_1<sizeof(int), 2>(&v_, vKQ50[cIb].qs + sizeof(int) * cIqs4);
    v_ = (v_ >> cShift) & 0x0F0F0F0F;

    {
      int vVh;
      ggml_cuda_memcpy_1<sizeof(int), 2>(&vVh, vKQ50[cIb].qh);
      vVh >>= cIqs8 * QI5_0;

      v_ |= (vVh << 4) & 0x00000010;  // 0 ->  4
      v_ |= (vVh << 11) & 0x00001000; // 1 -> 12
      v_ |= (vVh << 18) & 0x00100000; // 2 -> 20
      v_ |= (vVh << 25) & 0x10000000; // 3 -> 28
    }

    const int cU = pQQ8[vKKq0 / nthreads];

    const int cSumi = ggml_cuda_dp4a(v_, cU, 0);

    const float2 cQDs = ((const float2 *)pQDsV)[vKKq0 / nthreads];

    vSum += __half2float(vKQ50[cIb].d) * (cSumi * cQDs.x - (16 / QI8_1) * cQDs.y);
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_q5_1(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const block_q5_1 *vKQ51 = (const block_q5_1 *)pKC;
  GGML_UNUSED(pQV);

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < int(D / sizeof(int)); vKKq0 += nthreads) {
    const int cKKq = vKKq0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

    const int cIb = cKKq / QI8_1;
    const int cIqs4 = cKKq % QI5_1;
    const int cIqs8 = cKKq % QI8_1;
    const int cShift = cKKq & (QI8_1 / 2);

    int v_;
    ggml_cuda_memcpy_1<sizeof(int)>(&v_, vKQ51[cIb].qs + sizeof(int) * cIqs4);
    v_ = (v_ >> cShift) & 0x0F0F0F0F;

    {
      int vVh;
      ggml_cuda_memcpy_1<sizeof(int)>(&vVh, vKQ51[cIb].qh);
      vVh >>= cIqs8 * QI5_0;

      v_ |= (vVh << 4) & 0x00000010;  // 0 ->  4
      v_ |= (vVh << 11) & 0x00001000; // 1 -> 12
      v_ |= (vVh << 18) & 0x00100000; // 2 -> 20
      v_ |= (vVh << 25) & 0x10000000; // 3 -> 28
    }

    const int cU = pQQ8[vKKq0 / nthreads];

    const int cSumi = ggml_cuda_dp4a(v_, cU, 0);

    const float2 cKDm = __half22float2(vKQ51[cIb].dm);
    const float2 cQDs = ((const float2 *)pQDsV)[vKKq0 / nthreads];

    vSum += cKDm.x * cQDs.x * cSumi + cKDm.y * cQDs.y / QI8_1;
  }

  return vSum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float
vec_dot_fattn_vec_KQ_q8_0(const char *__restrict__ pKC, const void *__restrict__ pQV, const int *__restrict__ pQQ8,
                          const void *__restrict__ pQDsV) {

  const block_q8_0 *vKQ80 = (const block_q8_0 *)pKC;
  GGML_UNUSED(pQV);

  float vSum = 0.0f;

#pragma unroll
  for (int vKKq0 = 0; vKKq0 < int(D / sizeof(int)); vKKq0 += nthreads) {
    const int cKKq = vKKq0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

    const int cIb = cKKq / QI8_0;
    const int cIqs = cKKq % QI8_0;

    int v_;
    ggml_cuda_memcpy_1<sizeof(v_), 2>(&v_, vKQ80[cIb].qs + 4 * cIqs);

    const float2 *vQDs = (const float2 *)pQDsV;
    const float cQD = vQDs[vKKq0 / nthreads].x;

    vSum += vec_dot_q8_0_q8_1_impl<float, 1>(&v_, &pQQ8[vKKq0 / nthreads], vKQ80[cIb].d, cQD);
  }

  return vSum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(const float *__restrict__ pX, const float pScale,
                                                               int *__restrict__ pYq32, void *__restrict__ pYds) {

  float vVals[sizeof(int)] = {0.0f};
#pragma unroll
  for (int vL = 0; vL < int(sizeof(int)); ++vL) {
    vVals[vL] = (ni == WARP_SIZE || threadIdx.x < ni) ? pScale * pX[4 * threadIdx.x + vL] : 0.0f;
  }

  float vAmax = fabsf(vVals[0]);
  float vSum = vVals[0];
#pragma unroll
  for (int vL = 1; vL < int(sizeof(int)); ++vL) {
    vAmax = fmaxf(vAmax, fabsf(vVals[vL]));
    vSum += vVals[vL];
  }
#pragma unroll
  for (int vMask = QI8_1 / 2; vMask > 0; vMask >>= 1) {
    vAmax = fmaxf(vAmax, __shfl_xor_sync(0xFFFFFFFF, vAmax, vMask, 32));
    vSum += __shfl_xor_sync(0xFFFFFFFF, vSum, vMask, 32);
  }

  const float cD = vAmax / 127;
  int vQ32 = 0;
  int8_t *vQ8 = (int8_t *)&vQ32;

  if (cD != 0.0f) {
#pragma unroll
    for (int vL = 0; vL < int(sizeof(int)); ++vL) {
      vQ8[vL] = roundf(vVals[vL] / cD);
    }
  }

  pYq32[threadIdx.x] = vQ32;
  if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
    if (std::is_same<Tds, half2>::value) {
      ((half2 *)pYds)[threadIdx.x / QI8_1] = make_half2(cD, vSum);
    } else {
      ((float2 *)pYds)[threadIdx.x / QI8_1] = make_float2(cD, vSum);
    }
  }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                        const int64_t pI0) {
  if constexpr (std::is_same_v<T, half>) {
    ggml_cuda_memcpy_1<ne * sizeof(half)>(pDst, (const half *)pVx + pI0);
  } else if constexpr (std::is_same_v<T, float>) {
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) half2 vTmp[ne / 2];
    ggml_cuda_memcpy_1<ne * sizeof(half)>(vTmp, (const half *)pVx + pI0);
    float2 *vDstF2 = (float2 *)pDst;
#pragma unroll
    for (int vL = 0; vL < ne / 2; ++vL) {
      vDstF2[vL] = __half22float2(vTmp[vL]);
    }
  } else {
    static_assert(std::is_same_v<T, void>, "unsupported type");
  }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
  static_assert(ne % 2 == 0, "bad ne");
  __align__(16) nv_bfloat162 vTmp[ne / 2];
  ggml_cuda_memcpy_1<ne * sizeof(nv_bfloat16)>(vTmp, (const nv_bfloat16 *)pVx + pI0);
  float2 *vDstF2 = (float2 *)pDst;
#pragma unroll
  for (int vL = 0; vL < ne / 2; ++vL) {
    vDstF2[vL] = ggml_cuda_cast<float2>(vTmp[vL]);
  }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  const block_q4_0 *vX = (const block_q4_0 *)pVx;

  const int64_t cIb = pI0 / QK4_0;
  const int cIqs = pI0 % (QK4_0 / 2);
  const int cShift = (pI0 % QK4_0) / (QK4_0 / 2);

  int vQ;
  static_assert(ne == 2 || ne == 4, "bad ne");
  ggml_cuda_memcpy_1<ne, 2>(&vQ, vX[cIb].qs + cIqs);
  vQ >>= 4 * cShift;
  vQ &= 0x0F0F0F0F;
  vQ = __vsubss4(vQ, 0x08080808);

  const int8_t *vQ8 = (const int8_t *)&vQ;

#ifdef FP16_AVAILABLE
  if constexpr (std::is_same_v<T, half>) {
    const half2 cD = __half2half2(vX[cIb].d);

#pragma unroll
    for (int vL0 = 0; vL0 < ne; vL0 += 2) {
      ((half2 *)pDst)[vL0 / 2] = cD * make_half2(vQ8[vL0 + 0], vQ8[vL0 + 1]);
    }
  } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
      const float cD = vX[cIb].d;

#pragma unroll
      for (int vL = 0; vL < ne; ++vL) {
        ((float *)pDst)[vL] = cD * vQ8[vL];
      }
    } else {
      static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  const block_q4_1 *vX = (const block_q4_1 *)pVx;

  const int64_t cIb = pI0 / QK4_1;
  const int cIqs = pI0 % (QK4_1 / 2);
  const int cShift = (pI0 % QK4_1) / (QK4_1 / 2);

  int vQ;
  static_assert(ne == 2 || ne == 4, "bad ne");
  ggml_cuda_memcpy_1<ne>(&vQ, vX[cIb].qs + cIqs);
  vQ >>= 4 * cShift;
  vQ &= 0x0F0F0F0F;

  const int8_t *vQ8 = (const int8_t *)&vQ;

#ifdef FP16_AVAILABLE
  if constexpr (std::is_same_v<T, half>) {
    const half2 cDm = vX[cIb].dm;
    const half2 cD = __half2half2(__low2half(cDm));
    const half2 cM = __half2half2(__high2half(cDm));

#pragma unroll
    for (int vL0 = 0; vL0 < ne; vL0 += 2) {
      ((half2 *)pDst)[vL0 / 2] = cD * make_half2(vQ8[vL0 + 0], vQ8[vL0 + 1]) + cM;
    }
  } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
      const float2 cDm = __half22float2(vX[cIb].dm);

#pragma unroll
      for (int vL = 0; vL < ne; ++vL) {
        ((float *)pDst)[vL] = cDm.x * vQ8[vL] + cDm.y;
      }
    } else {
      static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  const block_q5_0 *vX = (const block_q5_0 *)pVx;

  const int64_t cIb = pI0 / QK5_0;
  const int cIdq = pI0 % QK5_0;
  const int cIqs = pI0 % (QK5_0 / 2);
  const int cShift = (pI0 % QK5_0) / (QK5_0 / 2);

  int vQ;
  static_assert(ne == 2 || ne == 4, "bad ne");
  ggml_cuda_memcpy_1<ne, 2>(&vQ, vX[cIb].qs + cIqs);
  vQ >>= 4 * cShift;
  vQ &= 0x0F0F0F0F;

  {
    int vQh;
    ggml_cuda_memcpy_1<ne, 2>(&vQh, vX[cIb].qh);
#pragma unroll
    for (int vL = 0; vL < ne; ++vL) {
      vQ |= ((vQh >> (cIdq + vL)) & 0x00000001) << (8 * vL + 4);
    }
  }

  vQ = __vsubss4(vQ, 0x10101010);

  const int8_t *vQ8 = (const int8_t *)&vQ;

#ifdef FP16_AVAILABLE
  if constexpr (std::is_same_v<T, half>) {
    const half2 cD = __half2half2(vX[cIb].d);

#pragma unroll
    for (int vL0 = 0; vL0 < ne; vL0 += 2) {
      ((half2 *)pDst)[vL0 / 2] = cD * make_half2(vQ8[vL0 + 0], vQ8[vL0 + 1]);
    }
  } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
      const float cD = vX[cIb].d;

#pragma unroll
      for (int vL = 0; vL < ne; ++vL) {
        ((float *)pDst)[vL] = cD * vQ8[vL];
      }
    } else {
      static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  const block_q5_1 *vX = (const block_q5_1 *)pVx;

  const int64_t cIb = pI0 / QK5_1;
  const int cIdq = pI0 % QK5_1;
  const int cIqs = pI0 % (QK5_1 / 2);
  const int cShift = (pI0 % QK5_1) / (QK5_1 / 2);

  int vQ;
  static_assert(ne == 2 || ne == 4, "bad ne");
  ggml_cuda_memcpy_1<ne>(&vQ, vX[cIb].qs + cIqs);
  vQ >>= 4 * cShift;
  vQ &= 0x0F0F0F0F;

  {
    int vQh;
    ggml_cuda_memcpy_1<ne>(&vQh, vX[cIb].qh);
#pragma unroll
    for (int vL = 0; vL < ne; ++vL) {
      vQ |= ((vQh >> (cIdq + vL)) & 0x00000001) << (8 * vL + 4);
    }
  }

  const int8_t *vQ8 = (const int8_t *)&vQ;

#ifdef FP16_AVAILABLE
  if constexpr (std::is_same_v<T, half>) {
    const half2 cDm = vX[cIb].dm;
    const half2 cD = __half2half2(__low2half(cDm));
    const half2 cM = __half2half2(__high2half(cDm));

#pragma unroll
    for (int vL0 = 0; vL0 < ne; vL0 += 2) {
      ((half2 *)pDst)[vL0 / 2] = cD * make_half2(vQ8[vL0 + 0], vQ8[vL0 + 1]) + cM;
    }
  } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
      const float2 cDm = __half22float2(vX[cIb].dm);

#pragma unroll
      for (int vL = 0; vL < ne; ++vL) {
        ((float *)pDst)[vL] = cDm.x * vQ8[vL] + cDm.y;
      }
    } else {
      static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void *__restrict__ pVx, void *__restrict__ pDst,
                                                         const int64_t pI0) {
  const block_q8_0 *vX = (const block_q8_0 *)pVx;

  const int64_t cIb = pI0 / QK8_0;
  const int cIqs = pI0 % QK8_0;

  static_assert(ne % 2 == 0, "bad ne");
  int8_t vQs[ne];
  ggml_cuda_memcpy_1<ne, 2>(vQs, vX[cIb].qs + cIqs);

#ifdef FP16_AVAILABLE
  if constexpr (std::is_same<T, half>::value) {
    const half2 cD = __half2half2(vX[cIb].d);

#pragma unroll
    for (int vL0 = 0; vL0 < ne; vL0 += 2) {
      ((half2 *)pDst)[vL0 / 2] = cD * make_half2(vQs[vL0 + 0], vQs[vL0 + 1]);
    }
  } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
      const float cD = vX[cIb].d;

#pragma unroll
      for (int vL = 0; vL < ne; ++vL) {
        ((float *)pDst)[vL] = cD * vQs[vL];
      }
    } else {
      static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads> constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
  if constexpr (type_K == GGML_TYPE_F16) {
    return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_Q4_0) {
    return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_Q4_1) {
    return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_Q5_0) {
    return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_Q5_1) {
    return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_Q8_0) {
    return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
  } else if constexpr (type_K == GGML_TYPE_BF16) {
    return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
  } else {
    static_assert(type_K == -1, "bad type");
    return nullptr;
  }
}

template <ggml_type type_V, typename T, int ne> constexpr __device__ dequantize_V_t get_dequantize_V() {
  if constexpr (type_V == GGML_TYPE_F16) {
    return dequantize_V_f16<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_Q4_0) {
    return dequantize_V_q4_0<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_Q4_1) {
    return dequantize_V_q4_1<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_Q5_0) {
    return dequantize_V_q5_0<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_Q5_1) {
    return dequantize_V_q5_1<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_Q8_0) {
    return dequantize_V_q8_0<T, ne>;
  } else if constexpr (type_V == GGML_TYPE_BF16) {
    return dequantize_V_bf16<float, ne>;
  } else {
    static_assert(type_V == -1, "bad type");
    return nullptr;
  }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE / 2, 1) static __global__
    void flash_attn_mask_to_KV_max(const half2 *pMaskPtr, int *pKvMaxPtr, const int pNe30, const int64_t pS31,
                                   const int64_t pS33) {
  const half2 *GGML_CUDA_RESTRICT vMask = pMaskPtr;
  int *GGML_CUDA_RESTRICT vKvMax = pKvMaxPtr;

  const int cNe31 = gridDim.x;
  const int cTid = threadIdx.x;
  const int cSequence = blockIdx.y;
  const int cJt = blockIdx.x;

  vMask += cSequence * pS33 + cJt * ncols1 * pS31;

  __shared__ int vBufIw[WARP_SIZE];
  if (cTid < WARP_SIZE) {
    vBufIw[cTid] = 1;
  }
  ggml_cuda_pdl_sync();
  __syncthreads();

  int vKvMaxSj = (pNe30 - 1) * FATTN_KQ_STRIDE;
  for (; vKvMaxSj >= 0; vKvMaxSj -= FATTN_KQ_STRIDE) {
    int vAllInf = 1;

#pragma unroll
    for (int vJ = 0; vJ < ncols1; ++vJ) {
      const float2 cTmp = __half22float2(vMask[vJ * pS31 + vKvMaxSj / 2 + cTid]);
      vAllInf = vAllInf && int(isinf(cTmp.x)) && int(isinf(cTmp.y));
    }

    vAllInf = warp_reduce_all(vAllInf);
    if (cTid % WARP_SIZE == 0) {
      vBufIw[cTid / WARP_SIZE] = vAllInf;
    }
    __syncthreads();
    vAllInf = vBufIw[cTid % WARP_SIZE];
    __syncthreads();
    vAllInf = warp_reduce_all(vAllInf);

    if (!vAllInf) {
      break;
    }
  }

  // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
  // If the break was triggered it's the lower edge of the tile with the first non-masked values.
  // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
  vKvMaxSj += FATTN_KQ_STRIDE;

  if (threadIdx.x != 0) {
    return;
  }

  vKvMax[cSequence * cNe31 + cJt] = vKvMaxSj;
}

template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1) static __global__
    void flash_attn_stream_k_fixup_uniform(float *pDstPtr, const float2 *pDstFixupPtr, const int pNe01, const int pNe02,
                                           const int pNe12, const int pNblocksStreamK, const int pGqaRatio,
                                           const int pBlocksPerTile, const uint3 pFdIterJZNe12, const uint3 pFdIterJZ,
                                           const uint3 pFdIterJ) {
  constexpr int cNcols = ncols1 * ncols2;
  ggml_cuda_pdl_lc();
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const float2 *GGML_CUDA_RESTRICT vDstFixup = pDstFixupPtr;

  const int cTileIdx = blockIdx.x; // One block per output tile.
  const int cJ = blockIdx.y;
  const int c_ = blockIdx.z;
  const int cJc = cJ * ncols2 + c_;
  const int cTid = threadIdx.x;

  // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
  const int cBFirst = cTileIdx * pBlocksPerTile;
  const int cBLast = cBFirst + pBlocksPerTile - 1;

  const float *vDstFixupData = ((const float *)vDstFixup) + pNblocksStreamK * (2 * 2 * cNcols);

  // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
  const uint2 cDm0 = fast_div_modulo(cTileIdx, pFdIterJZNe12);
  const uint2 cDm1 = fast_div_modulo(cDm0.y, pFdIterJZ);
  const uint2 cDm2 = fast_div_modulo(cDm1.y, pFdIterJ);

  const int cSequence = cDm0.x;
  const int cZKv = cDm1.x;
  const int cZtGqa = cDm2.x;
  const int cJt = cDm2.y;

  const int cZtQ = cZKv * pGqaRatio + cZtGqa * ncols2; // Global Q head start index.

  if (cJt * ncols1 + cJ >= pNe01 || cZtGqa * ncols2 + c_ >= pGqaRatio) {
    return;
  }

  vDst += cSequence * pNe02 * pNe01 * D + cJt * pNe02 * (ncols1 * D) + cZtQ * D + (cJ * pNe02 + c_) * D + cTid;

  ggml_cuda_pdl_sync();
  // Load the partial result that needs a fixup
  float vDstVal = *vDst;
  float vMaxVal;
  float vRowsum;
  {
    const float2 cTmp = vDstFixup[cBLast * cNcols + cJc];
    vMaxVal = cTmp.x;
    vRowsum = cTmp.y;
  }

  // Combine with all previous blocks in this tile.
  for (int vBidx = cBLast - 1; vBidx >= cBFirst; --vBidx) {
    const float cDstAdd = vDstFixupData[vBidx * cNcols * D + cJc * D + cTid];

    const float2 cTmp = vDstFixup[(pNblocksStreamK + vBidx) * cNcols + cJc];

    const float cMaxValNew = fmaxf(vMaxVal, cTmp.x);

    const float cDiffVal = vMaxVal - cMaxValNew;
    const float cDiffAdd = cTmp.x - cMaxValNew;

    const float cScaleVal = cDiffVal >= SOFTMAX_FTZ_THRESHOLD ? expf(cDiffVal) : 0.0f;
    const float cScaleAdd = cDiffAdd >= SOFTMAX_FTZ_THRESHOLD ? expf(cDiffAdd) : 0.0f;

    vDstVal = cScaleVal * vDstVal + cScaleAdd * cDstAdd;
    vRowsum = cScaleVal * vRowsum + cScaleAdd * cTmp.y;

    vMaxVal = cMaxValNew;
  }

  // Write back final result:
  *vDst = vDstVal / vRowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1) static __global__
    void flash_attn_stream_k_fixup_general(float *pDstPtr, const float2 *pDstFixupPtr, const int pNe01, const int pNe02,
                                           const int pGqaRatio, const int pTotalWork, const uint3 pFdIterKJZNe12,
                                           const uint3 pFdIterKJZ, const uint3 pFdIterKJ, const uint3 pFdIterK) {
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const float2 *GGML_CUDA_RESTRICT vDstFixup = pDstFixupPtr;
  constexpr int cNcols = ncols1 * ncols2;

  const int cBidx0 = blockIdx.x;
  const int cJ = blockIdx.y;
  const int c_ = blockIdx.z;
  const int cJc = cJ * ncols2 + c_;
  const int cTid = threadIdx.x;

  const float *vDstFixupData = ((const float *)vDstFixup) + gridDim.x * (2 * 2 * cNcols);

  const int cKbc0 = int64_t(cBidx0 + 0) * pTotalWork / gridDim.x;
  const int cKbc0Stop = int64_t(cBidx0 + 1) * pTotalWork / gridDim.x;

  const bool cDidNotHaveAnyData = cKbc0 == cKbc0Stop;
  const bool cWroteBeginningOfTile = fastmodulo(cKbc0, pFdIterK) == 0;
  const bool cDidNotWriteLast =
      fastdiv(cKbc0, pFdIterK) == fastdiv(cKbc0Stop, pFdIterK) && fastmodulo(cKbc0Stop, pFdIterK) != 0;
  if (cDidNotHaveAnyData || cWroteBeginningOfTile || cDidNotWriteLast) {
    return;
  }

  // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
  const uint2 cDm0 = fast_div_modulo(cKbc0, pFdIterKJZNe12);
  const uint2 cDm1 = fast_div_modulo(cDm0.y, pFdIterKJZ);
  const uint2 cDm2 = fast_div_modulo(cDm1.y, pFdIterKJ);
  const uint2 cDm3 = fast_div_modulo(cDm2.y, pFdIterK);

  const int cSequence = cDm0.x;
  const int cZKv = cDm1.x;
  const int cZtGqa = cDm2.x;
  const int cJt = cDm3.x;

  const int cZtQ = cZKv * pGqaRatio + cZtGqa * ncols2; // Global Q head start index.

  if (cJt * ncols1 + cJ >= pNe01 || cZtGqa * ncols2 + c_ >= pGqaRatio) {
    return;
  }

  vDst += cSequence * pNe02 * pNe01 * D + cJt * pNe02 * (ncols1 * D) + cZtQ * D + (cJ * pNe02 + c_) * D + cTid;

  // Load the partial result that needs a fixup:
  float vDstVal = 0.0f;
  float vMaxVal = 0.0f;
  float vRowsum = 0.0f;
  ggml_cuda_pdl_sync();
  {
    vDstVal = *vDst;

    const float2 cTmp = vDstFixup[cBidx0 * cNcols + cJc];
    vMaxVal = cTmp.x;
    vRowsum = cTmp.y;
  }

  // Iterate over previous blocks and compute the combined results.
  // All CUDA blocks that get here must have a previous block that needs a fixup.
  const int cTileKbc0 = fastdiv(cKbc0, pFdIterK);
  int vBidx = cBidx0 - 1;
  int vKbcStop = cKbc0;
  while (true) {
    const int cKbc = int64_t(vBidx) * pTotalWork / gridDim.x;
    if (cKbc == vKbcStop) { // Did not have any data.
      vBidx--;
      vKbcStop = cKbc;
      continue;
    }

    const float cDstAdd = vDstFixupData[vBidx * cNcols * D + cJc * D + cTid];

    const float2 cTmp = vDstFixup[(gridDim.x + vBidx) * cNcols + cJc];

    // Scale the current and new value accumulators depending on the max. values.
    const float cMaxValNew = fmaxf(vMaxVal, cTmp.x);

    const float cDiffVal = vMaxVal - cMaxValNew;
    const float cDiffAdd = cTmp.x - cMaxValNew;

    const float cScaleVal = cDiffVal >= SOFTMAX_FTZ_THRESHOLD ? expf(cDiffVal) : 0.0f;
    const float cScaleAdd = cDiffAdd >= SOFTMAX_FTZ_THRESHOLD ? expf(cDiffAdd) : 0.0f;

    vDstVal = cScaleVal * vDstVal + cScaleAdd * cDstAdd;
    vRowsum = cScaleVal * vRowsum + cScaleAdd * cTmp.y;

    vMaxVal = cMaxValNew;

    // If this block started in a previous tile we are done and don't need to combine additional partial results.
    if (fastmodulo(cKbc, pFdIterK) == 0 || fastdiv(cKbc, pFdIterK) < cTileKbc0) {
      break;
    }
    vBidx--;
    vKbcStop = cKbc;
  }

  // Write back final result:
  *vDst = vDstVal / vRowsum;
}

template <int D> // D == head size
__launch_bounds__(D, 1) static __global__
    void flash_attn_combine_results(const float *pVkqPartsPtr, const float2 *pVkqMetaPtr, float *pDstPtr,
                                    const int pParallelBlocks) {
  ggml_cuda_pdl_lc();
  const float *GGML_CUDA_RESTRICT vVkqParts = pVkqPartsPtr;
  const float2 *GGML_CUDA_RESTRICT vVkqMeta = pVkqMetaPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  // Dimension 0: threadIdx.x
  // Dimension 1: blockIdx.x
  // Dimension 2: blockIdx.y
  // Dimension 3: blockIdx.z
  // Memory layout is permuted with [0, 2, 1, 3]

  const int cNe01 = gridDim.x;
  const int cNe02 = gridDim.y;

  const int cCol = blockIdx.x;
  const int cHead = blockIdx.y;
  const int cSequence = blockIdx.z;

  const int cJDstUnrolled = (cSequence * cNe01 + cCol) * cNe02 + cHead;

  vVkqParts += cJDstUnrolled * pParallelBlocks * D;
  vVkqMeta += cJDstUnrolled * pParallelBlocks;
  vDst += cJDstUnrolled * D;

  const int cTid = threadIdx.x;
  __builtin_assume(cTid < D);

  extern __shared__ float2 vMeta[];
  ggml_cuda_pdl_sync();
  for (int vI = cTid; vI < 2 * pParallelBlocks; vI += D) {
    ((float *)vMeta)[vI] = ((const float *)vVkqMeta)[vI];
  }

  __syncthreads();

  float vKqmax = vMeta[0].x;
  for (int vL = 1; vL < pParallelBlocks; ++vL) {
    vKqmax = max(vKqmax, vMeta[vL].x);
  }

  float vVkqNumerator = 0.0f;
  float vVkqDenominator = 0.0f;
  for (int vL = 0; vL < pParallelBlocks; ++vL) {
    const float cKqMaxScale = expf(vMeta[vL].x - vKqmax);

    vVkqNumerator += cKqMaxScale * vVkqParts[vL * D + cTid];
    vVkqDenominator += cKqMaxScale * vMeta[vL].y;
  }

  vDst[cTid] = vVkqNumerator / vVkqDenominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, fattn_kernel_t pFattnKernel, const int pNwarps,
                  const size_t pNbytesShared, const int pNbatchFa, const bool pNeedF16K, const bool pNeedF16V,
                  const bool pStreamK, const int pWarpSize = WARP_SIZE) {
  constexpr int cNcols = ncols1 * ncols2;

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];

  const bool cVIsKView =
      vV->view_src && (vV->view_src == vK || (vV->view_src == vK->view_src && vV->view_offs == vK->view_offs));

  const ggml_tensor *vMask = pDst->src[3];
  const ggml_tensor *vSinks = pDst->src[4];

  ggml_tensor *vKqv = pDst;

  GGML_ASSERT(vQ->type == GGML_TYPE_F32);
  GGML_ASSERT(vKqv->type == GGML_TYPE_F32);

  GGML_ASSERT(vQ->nb[0] == ggml_element_size(vQ));
  GGML_ASSERT(vK->nb[0] == ggml_element_size(vK));
  GGML_ASSERT(vV->nb[0] == ggml_element_size(vV));

  GGML_ASSERT(!vMask || vMask->type == GGML_TYPE_F16);

  ggml_cuda_pool &vPool = pCtx.pool();
  cudaStream_t vMainStream = pCtx.stream();
  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;
  const int cNsm = ggml_cuda_info().devices[cId].nsm;

  const ggml_cuda_flash_attn_ext_f16_extra_data cF16Extra =
      ggml_cuda_flash_attn_ext_get_f16_extra_data(vKqv, pNeedF16K, pNeedF16V);

  ggml_cuda_pool_alloc<int> vKvMax(vPool);
  ggml_cuda_pool_alloc<float> vDstTmp(vPool);
  ggml_cuda_pool_alloc<float2> vDstTmpMeta(vPool);

  const char *vKData = (const char *)vK->data;
  size_t vNb11 = vK->nb[1];
  size_t vNb12 = vK->nb[2];
  size_t vNb13 = vK->nb[3];

  const char *vVData = (const char *)vV->data;
  size_t vNb21 = vV->nb[1];
  size_t vNb22 = vV->nb[2];
  size_t vNb23 = vV->nb[3];

  if (pNeedF16K && vK->type != GGML_TYPE_F16) {
    const size_t cBs = ggml_blck_size(vK->type);
    const size_t cTs = ggml_type_size(vK->type);

    GGML_ASSERT(cF16Extra.K != 0);
    half *vKF16 = (half *)cF16Extra.K;
    if (ggml_is_contiguously_allocated(vK)) {
      to_fp16_cuda_t vToFp16 = ggml_get_to_fp16_cuda(vK->type);
      vToFp16(vKData, vKF16, ggml_nelements(vK), vMainStream);

      vNb11 = vNb11 * cBs * sizeof(half) / cTs;
      vNb12 = vNb12 * cBs * sizeof(half) / cTs;
      vNb13 = vNb13 * cBs * sizeof(half) / cTs;
    } else {
      GGML_ASSERT(vK->nb[0] == cTs);
      to_fp16_nc_cuda_t vToFp16 = ggml_get_to_fp16_nc_cuda(vK->type);
      const int64_t cS01 = vNb11 / cTs;
      const int64_t cS02 = vNb12 / cTs;
      const int64_t cS03 = vNb13 / cTs;
      vToFp16(vKData, vKF16, vK->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], cS01, cS02, cS03, vMainStream);

      vNb11 = vK->ne[0] * sizeof(half);
      vNb12 = vK->ne[1] * vNb11;
      vNb13 = vK->ne[2] * vNb12;
    }
    vKData = (char *)vKF16;
  }

  if (pNeedF16V && vV->type != GGML_TYPE_F16) {
    if (cVIsKView) {
      vVData = vKData;
      vNb21 = vNb11;
      vNb22 = vNb12;
      vNb23 = vNb13;
    } else {
      const size_t cBs = ggml_blck_size(vV->type);
      const size_t cTs = ggml_type_size(vV->type);

      GGML_ASSERT(cF16Extra.V != 0);
      half *vVF16 = (half *)cF16Extra.V;
      if (ggml_is_contiguously_allocated(vV)) {
        to_fp16_cuda_t vToFp16 = ggml_get_to_fp16_cuda(vV->type);
        vToFp16(vVData, vVF16, ggml_nelements(vV), vMainStream);
        vVData = (char *)vVF16;

        vNb21 = vNb21 * cBs * sizeof(half) / cTs;
        vNb22 = vNb22 * cBs * sizeof(half) / cTs;
        vNb23 = vNb23 * cBs * sizeof(half) / cTs;
      } else {
        GGML_ASSERT(vV->nb[0] == cTs);
        to_fp16_nc_cuda_t vToFp16 = ggml_get_to_fp16_nc_cuda(vV->type);
        const int64_t cS01 = vNb21 / cTs;
        const int64_t cS02 = vNb22 / cTs;
        const int64_t cS03 = vNb23 / cTs;
        vToFp16(vVData, vVF16, vV->ne[0], vV->ne[1], vV->ne[2], vV->ne[3], cS01, cS02, cS03, vMainStream);

        vNb21 = vV->ne[0] * sizeof(half);
        vNb22 = vV->ne[1] * vNb21;
        vNb23 = vV->ne[2] * vNb22;
      }
      vVData = (char *)vVF16;
    }
  }

  const int cNtilesX = ((vQ->ne[1] + ncols1 - 1) / ncols1);
  const int cGqaRatio = vQ->ne[2] / vK->ne[2];
  const int cNtilesZGqa = ((cGqaRatio + ncols2 - 1) / ncols2);
  const int cNtilesDst = cNtilesX * cNtilesZGqa * vK->ne[2] * vQ->ne[3];

  // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
  // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
  //     multiple sequences of possibly different lengths.
  if (vMask && vK->ne[1] % FATTN_KQ_STRIDE == 0 && (vQ->ne[1] >= 1024 || vQ->ne[3] > 1)) {
    const int64_t cS31 = vMask->nb[1] / sizeof(half2);
    const int64_t cS33 = vMask->nb[3] / sizeof(half2);

    const dim3 cBlocksNumKvMax(cNtilesX, vQ->ne[3], 1);
    const dim3 cBlockDimKvMax(FATTN_KQ_STRIDE / 2, 1, 1);

    const int cNeKvMax = cBlocksNumKvMax.x * cBlocksNumKvMax.y;
    const int cIterK = vK->ne[1] / FATTN_KQ_STRIDE;

    vKvMax.alloc(cNeKvMax);
    ggml_cuda_kernel_launch_params vLaunchParams =
        ggml_cuda_kernel_launch_params(cBlocksNumKvMax, cBlockDimKvMax, 0, vMainStream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, vLaunchParams, (const half2 *)vMask->data, vKvMax.ptr,
                            cIterK, cS31, cS33);
    CUDA_CHECK(cudaGetLastError());
  }

  const dim3 cBlockDim(pWarpSize, pNwarps, 1);
  int vMaxBlocksPerSm = 1; // Max. number of active blocks limited by occupancy.
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&vMaxBlocksPerSm, pFattnKernel,
                                                           cBlockDim.x * cBlockDim.y * cBlockDim.z, pNbytesShared));
  GGML_ASSERT(vMaxBlocksPerSm > 0);
  int vParallelBlocks = vMaxBlocksPerSm;

  const int cNtilesKv =
      (vK->ne[1] + pNbatchFa - 1) / pNbatchFa; // Max. number of parallel blocks limited by KV cache length.

  dim3 vBlocksNum;
  if (pStreamK) {
    // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
    const int cMaxBlocks = vMaxBlocksPerSm * cNsm;
    const int cTilesNwaves = (cNtilesDst + cMaxBlocks - 1) / cMaxBlocks;
    const int cTilesEfficiencyPercent = 100 * cNtilesDst / (cMaxBlocks * cTilesNwaves);

    const bool cUseStreamK =
        cCc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cCc) || cTilesEfficiencyPercent < 75;

    vBlocksNum.x = cNtilesDst;
    vBlocksNum.y = 1;
    vBlocksNum.z = 1;

    if (cUseStreamK) {
      const int cNblocksStreamKRaw = std::min(cMaxBlocks, cNtilesKv * cNtilesDst);
      // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
      // Only do this if the occupancy loss from rounding is acceptable.
      const int cNblocksStreamKRounded = (cNblocksStreamKRaw / cNtilesDst) * cNtilesDst;
      const int cMaxEfficiencyLossPercent = 5;
      const int cEfficiencyLossPercent =
          cNblocksStreamKRounded > 0 ? 100 * (cNblocksStreamKRaw - cNblocksStreamKRounded) / cNblocksStreamKRaw : 100;
      const int cNblocksStreamK =
          cEfficiencyLossPercent <= cMaxEfficiencyLossPercent ? cNblocksStreamKRounded : cNblocksStreamKRaw;

      vBlocksNum.x = cNblocksStreamK;
    }

    if (cNtilesDst % vBlocksNum.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
      vDstTmpMeta.alloc((size_t(vBlocksNum.x) * cNcols * (2 + DV / 2)));
    }
  } else {
    // parallel_blocks must not be larger than what the tensor size allows:
    vParallelBlocks = std::min(vParallelBlocks, cNtilesKv);

    // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
    // Test whether parallel_blocks can be set to a higher value for better efficiency.
    const int cBlocksPerWave = cNsm * vMaxBlocksPerSm;
    int vNwavesBest = 0;
    int vEfficiencyPercentBest = 0;
    for (int vParallelBlocksTest = vParallelBlocks; vParallelBlocksTest <= cNtilesKv; ++vParallelBlocksTest) {
      const int cNblocksTotal = cNtilesDst * vParallelBlocksTest;
      const int cNwaves = (cNblocksTotal + cBlocksPerWave - 1) / cBlocksPerWave;
      const int cEfficiencyPercent = 100 * cNblocksTotal / (cNwaves * cBlocksPerWave);

      // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
      if (vEfficiencyPercentBest >= 95 && cNwaves > vNwavesBest) {
        break;
      }

      if (cEfficiencyPercent > vEfficiencyPercentBest) {
        vNwavesBest = cNwaves;
        vEfficiencyPercentBest = cEfficiencyPercent;
        vParallelBlocks = vParallelBlocksTest;
      }
    }

    vBlocksNum.x = cNtilesX;
    vBlocksNum.y = vParallelBlocks;
    vBlocksNum.z = cNtilesZGqa * vK->ne[2] * vQ->ne[3];

    if (vParallelBlocks > 1) {
      vDstTmp.alloc(vParallelBlocks * ggml_nelements(vKqv));
      vDstTmpMeta.alloc(vParallelBlocks * ggml_nrows(vKqv));
    }
  }

  float vScale = 1.0f;
  float vMaxBias = 0.0f;
  float vLogitSoftcap = 0.0f;

  memcpy(&vScale, (const float *)vKqv->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));
  memcpy(&vLogitSoftcap, (const float *)vKqv->op_params + 2, sizeof(float));

  if (vLogitSoftcap != 0.0f) {
    vScale /= vLogitSoftcap;
  }

  const uint32_t cNHead = vQ->ne[2];
  const uint32_t cNHeadLog2 = 1u << uint32_t(floorf(log2f(float(cNHead))));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  // TODO other tensor dimensions after removal of WMMA kernel:
  const uint3 cNe01 = init_fastdiv_values(vQ->ne[1]);

  GGML_ASSERT(cBlockDim.x % pWarpSize == 0);

  ggml_cuda_kernel_launch_params vLaunchParams =
      ggml_cuda_kernel_launch_params(vBlocksNum, cBlockDim, pNbytesShared, vMainStream);
  ggml_cuda_kernel_launch(
      pFattnKernel, vLaunchParams, (const char *)vQ->data, vKData, vVData,
      vMask ? ((const char *)vMask->data) : nullptr, vSinks ? ((const char *)vSinks->data) : nullptr, vKvMax.ptr,
      !pStreamK && vParallelBlocks > 1 ? vDstTmp.ptr : (float *)vKqv->data, vDstTmpMeta.ptr, vScale, vMaxBias, cM0, cM1,
      cNHeadLog2, vLogitSoftcap, vQ->ne[0], cNe01, vQ->ne[2], vQ->ne[3], vQ->nb[1], vQ->nb[2], vQ->nb[3], vK->ne[0],
      vK->ne[1], vK->ne[2], vK->ne[3], vNb11, vNb12, vNb13, vNb21, vNb22, vNb23, vMask ? vMask->ne[1] : 0,
      vMask ? vMask->ne[2] : 0, vMask ? vMask->ne[3] : 0, vMask ? vMask->nb[1] : 0, vMask ? vMask->nb[2] : 0,
      vMask ? vMask->nb[3] : 0);
  CUDA_CHECK(cudaGetLastError());

  if (pStreamK) {
    if ((int)vBlocksNum.x % cNtilesDst == 0 && (int)vBlocksNum.x > cNtilesDst) {
      // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
      const int cNblocksSk = (int)vBlocksNum.x;
      const int cBpt = cNblocksSk / cNtilesDst;

      const uint3 cFd0 = init_fastdiv_values(cNtilesX * cNtilesZGqa * vK->ne[2]);
      const uint3 cFd1 = init_fastdiv_values(cNtilesX * cNtilesZGqa);
      const uint3 cFd2 = init_fastdiv_values(cNtilesX);

      const dim3 cBlockDimCombine(DV, 1, 1);
      const dim3 cBlocksNumCombine = {(unsigned)cNtilesDst, ncols1, ncols2};

      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocksNumCombine, cBlockDimCombine, 0, vMainStream);
      ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, cLaunchParams, (float *)vKqv->data,
                              vDstTmpMeta.ptr, vQ->ne[1], vQ->ne[2], vK->ne[2], cNblocksSk, cGqaRatio, cBpt, cFd0, cFd1,
                              cFd2);
    } else if (cNtilesDst % vBlocksNum.x != 0) {
      // General fixup for the cases where nblocks_stream_k < ntiles_dst.
      const int cTotalWork = cNtilesKv * cNtilesDst;

      const uint3 cFdKJZNe12 = init_fastdiv_values(cNtilesKv * cNtilesX * cNtilesZGqa * vK->ne[2]);
      const uint3 cFdKJZ = init_fastdiv_values(cNtilesKv * cNtilesX * cNtilesZGqa);
      const uint3 cFdKJ = init_fastdiv_values(cNtilesKv * cNtilesX);
      const uint3 cFdK = init_fastdiv_values(cNtilesKv);

      const dim3 cBlockDimCombine(DV, 1, 1);
      const dim3 cBlocksNumCombine = {vBlocksNum.x, ncols1, ncols2};

      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(cBlocksNumCombine, cBlockDimCombine, 0, vMainStream);
      ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, cLaunchParams, (float *)vKqv->data,
                              vDstTmpMeta.ptr, vQ->ne[1], vQ->ne[2], cGqaRatio, cTotalWork, cFdKJZNe12, cFdKJZ, cFdKJ,
                              cFdK);
    }
  } else if (vParallelBlocks > 1) {
    const dim3 cBlockDimCombine(DV, 1, 1);
    const dim3 cBlocksNumCombine(vQ->ne[1], vQ->ne[2], vQ->ne[3]);
    const size_t cNbytesSharedCombine = vParallelBlocks * sizeof(float2);

    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlocksNumCombine, cBlockDimCombine, cNbytesSharedCombine, vMainStream);
    ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, cLaunchParams, vDstTmp.ptr, vDstTmpMeta.ptr,
                            (float *)vKqv->data, vParallelBlocks);
  }
  CUDA_CHECK(cudaGetLastError());
}
