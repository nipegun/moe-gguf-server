#include "quantize.cuh"
#include <cstdint>

#if defined(BLACKWELL_MMA_AVAILABLE)
// this maps to 256-bit loads in PTX on supported devices,
// and otherwise falls back to 2 128-bit loads
struct __builtin_align__(32) float8 {
  float x;
  float y;
  float z;
  float w;
  float p;
  float q;
  float r;
  float s;
};
#endif

#if CUDART_VERSION >= 12080
static __device__ __forceinline__ float fNvfp4NativeScaleError(const float pVals[QK_NVFP4_SUB],
                                                               const float pInvColScale, const float pInvScale,
                                                               const float pScale) {
  const float cScaleDequant = 2.0f * pScale;
  float vErr = 0.0f;

#pragma unroll
  for (int vK = 0; vK < QK_NVFP4_SUB; vK += 4) {
    const float cV0 = pVals[vK + 0] * pInvColScale;
    const float cV1 = pVals[vK + 1] * pInvColScale;
    const float cV2 = pVals[vK + 2] * pInvColScale;
    const float cV3 = pVals[vK + 3] * pInvColScale;

    const __nv_fp4x4_e2m1 cQ(make_float4(cV0 * pInvScale, cV1 * pInvScale, cV2 * pInvScale, cV3 * pInvScale));
    const __nv_fp4x4_storage_t cQStorage = cQ.__x;
    const __nv_fp4x2_storage_t cQLo = static_cast<__nv_fp4x2_storage_t>(cQStorage);
    const __nv_fp4x2_storage_t cQHi = static_cast<__nv_fp4x2_storage_t>(cQStorage >> 8U);

    const __half2_raw cHraw2Lo = __nv_cvt_fp4x2_to_halfraw2(cQLo, __NV_E2M1);
    const __half2_raw cHraw2Hi = __nv_cvt_fp4x2_to_halfraw2(cQHi, __NV_E2M1);
    const __half2 cH2Lo = static_cast<__half2>(cHraw2Lo);
    const __half2 cH2Hi = static_cast<__half2>(cHraw2Hi);
    const float2 cDqLo = __half22float2(cH2Lo);
    const float2 cDqHi = __half22float2(cH2Hi);

    const float cErr0 = fabsf(cV0) - fabsf(cDqLo.x) * cScaleDequant;
    const float cErr1 = fabsf(cV1) - fabsf(cDqLo.y) * cScaleDequant;
    const float cErr2 = fabsf(cV2) - fabsf(cDqHi.x) * cScaleDequant;
    const float cErr3 = fabsf(cV3) - fabsf(cDqHi.y) * cScaleDequant;

    vErr = fmaf(cErr0, cErr0, vErr);
    vErr = fmaf(cErr1, cErr1, vErr);
    vErr = fmaf(cErr2, cErr2, vErr);
    vErr = fmaf(cErr3, cErr3, vErr);
  }

  return vErr;
}
#endif // CUDART_VERSION >= 12080

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1) static __global__
    void fQuantizeQ81(const float *pXPtr, void *pVyPtr, const int64_t pNe00, const int64_t pS01, const int64_t pS02,
                      const int64_t pS03, const int64_t pNe0, const uint32_t pNe1, const uint3 pNe2) {
  ggml_cuda_pdl_lc();
  const float *GGML_CUDA_RESTRICT vX = pXPtr;
  void *GGML_CUDA_RESTRICT vVy = pVyPtr;
  const int64_t cI0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (cI0 >= pNe0) {
    return;
  }

  const int64_t cI3 = fastdiv(blockIdx.z, pNe2);
  const int64_t cI2 = blockIdx.z - cI3 * pNe2.z;
  const int64_t cI1 = blockIdx.y;

  const int64_t &vI00 = cI0;
  const int64_t &vI01 = cI1;
  const int64_t &vI02 = cI2;
  const int64_t &vI03 = cI3;

  const int64_t cICont = ((cI3 * pNe2.z + cI2) * pNe1 + cI1) * pNe0 + cI0;

  block_q8_1 *vY = (block_q8_1 *)vVy;

  const int64_t cIb = cICont / QK8_1;  // block index
  const int64_t cIqs = cICont % QK8_1; // quant index

  ggml_cuda_pdl_sync();
  const float cXi = cI0 < pNe00 ? vX[vI03 * pS03 + vI02 * pS02 + vI01 * pS01 + vI00] : 0.0f;
  float vAmax = fabsf(cXi);
  float vSum = cXi;

  vAmax = warp_reduce_max<QK8_1>(vAmax);
  vSum = warp_reduce_sum<QK8_1>(vSum);

  const float cD = vAmax / 127.0f;
  const int8_t cQ = vAmax == 0.0f ? 0 : roundf(cXi / cD);

  vY[cIb].qs[cIqs] = cQ;

  if (cIqs > 0) {
    return;
  }

  vY[cIb].ds = make_half2(cD, vSum);
}

__device__ __forceinline__ uint8_t fComputeE8m0Scale(float pAmax) {
  if (!(pAmax > 0.0f)) {
    return 0;
  }

  // FP4 E2M1: max exponent (unbiased) is 2.
  constexpr int cFP4E2M1Emax = 2;

  const float cE = log2f(pAmax);

  // "even" -> round-to-nearest integer, ties-to-even
  const int cEInt = __float2int_rn(cE);

  const int cSharedExp = cEInt - cFP4E2M1Emax;

  int vBiased = cSharedExp + 127;

  vBiased = max(vBiased, 0);
  vBiased = min(vBiased, 254);

  return static_cast<uint8_t>(vBiased);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter, bool use_aligned_float8>
static __global__ void fQuantizeMmqNvfp4(const float *__restrict__ pX, const int32_t *__restrict__ pIds,
                                         void *__restrict__ pVy, float *__restrict__ pScale, const int64_t pNe00,
                                         const int64_t pS01, const int64_t pS02, const int64_t pS03, const int64_t pNe0,
                                         const int64_t pNe1, const int64_t pNe2, const int pNExpertUsed) {
#if defined(BLACKWELL_MMA_AVAILABLE)

  const int64_t blocks_per_col = (pNe0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;

  int64_t base_idx;
  if constexpr (scatter) {
    base_idx = (int64_t)blockIdx.x * pS02; // one physical row per token
  } else {
    const int64_t i2 = blockIdx.y % pNe2;
    const int64_t i3 = blockIdx.y / pNe2;
    const int64_t i01 = pIds ? pIds[blockIdx.x] : blockIdx.x;
    base_idx = i3 * pS03 + i2 * pS02 + i01 * pS01;
  }
  const float *__restrict__ x_row = pX + base_idx;

  float amax = 0.0f;
  if constexpr (use_aligned_float8) {
    for (int64_t i0 = 8 * threadIdx.x; i0 < pNe00; i0 += 8 * blockDim.x) {
      const float *x_base = x_row + i0;
      const float8 v = reinterpret_cast<const float8 *>(x_base)[0];
      amax = fmaxf(amax, fabsf(v.x));
      amax = fmaxf(amax, fabsf(v.y));
      amax = fmaxf(amax, fabsf(v.z));
      amax = fmaxf(amax, fabsf(v.w));
      amax = fmaxf(amax, fabsf(v.p));
      amax = fmaxf(amax, fabsf(v.q));
      amax = fmaxf(amax, fabsf(v.r));
      amax = fmaxf(amax, fabsf(v.s));
    }
  } else {
    for (int64_t i0 = threadIdx.x; i0 < pNe00; i0 += blockDim.x) {
      amax = fmaxf(amax, fabsf(x_row[i0]));
    }
  }

  amax = warp_reduce_max<WARP_SIZE>(amax);

  __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;

  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();

  if (warp == 0) {
    amax = threadIdx.x < int(CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE) ? warp_amax[lane] : 0.0f;
    amax = warp_reduce_max<WARP_SIZE>(amax);
    if (lane == 0) {
      warp_amax[0] = amax / (6.0f * 448.0f);
      if constexpr (scatter) {
#pragma unroll
        for (int slot = 0; slot < pNExpertUsed; ++slot) {
          const int64_t i = pIds[(int64_t)blockIdx.x * pNExpertUsed + slot];
          pScale[i] = warp_amax[0];
        }
      } else {
        pScale[blockIdx.y * pNe1 + blockIdx.x] = warp_amax[0];
      }
    }
  }
  __syncthreads();

  block_fp4_mmq *y = (block_fp4_mmq *)pVy;
  const int64_t n_subblocks = (pNe0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;

  for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
    const int64_t i0_base = isb * QK_NVFP4_SUB;
    const int64_t k_block = i0_base / QK_FP4_MMQ;
    const int sub = (i0_base % QK_FP4_MMQ) / QK_NVFP4_SUB;

    const float row_scale = warp_amax[0];
    const float inv_col_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;

    float vals[QK_NVFP4_SUB];
    if constexpr (use_aligned_float8) {
      const float *x_base = x_row + i0_base;
      const float8 v0 = i0_base + 7 < pNe00 ? reinterpret_cast<const float8 *>(x_base)[0]
                                            : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      const float8 v1 = i0_base + 15 < pNe00 ? reinterpret_cast<const float8 *>(x_base + 8)[0]
                                             : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      vals[0] = v0.x;
      vals[1] = v0.y;
      vals[2] = v0.z;
      vals[3] = v0.w;
      vals[4] = v0.p;
      vals[5] = v0.q;
      vals[6] = v0.r;
      vals[7] = v0.s;
      vals[8] = v1.x;
      vals[9] = v1.y;
      vals[10] = v1.z;
      vals[11] = v1.w;
      vals[12] = v1.p;
      vals[13] = v1.q;
      vals[14] = v1.r;
      vals[15] = v1.s;
    } else {
#pragma unroll
      for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        const int64_t i00 = i0_base + k;
        vals[k] = i00 < pNe00 ? x_row[i00] : 0.0f;
      }
    }

    uint32_t q0 = 0;
    uint32_t q1 = 0;

    float amax_sub = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
      amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
    }

    static constexpr int test_offsets[5] = {0, -1, 1, -2, 2};
    const int first_fp8_code = (int)ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

    uint8_t fp8_code = (uint8_t)first_fp8_code;
    float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
    float inv_scale_err = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#if CUDART_VERSION >= 12080
    float best_err = fNvfp4NativeScaleError(vals, inv_col_scale, inv_scale_err, subblock_scale);
#else
    float best_err = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
      const float v = vals[k] * inv_col_scale;
      const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, inv_scale_err);
      const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * subblock_scale;
      best_err = fmaf(err_diff, err_diff, best_err);
    }
#endif // CUDART_VERSION >= 12080

#pragma unroll
    for (int i = 1; i < 5; ++i) {
      const int test_code = first_fp8_code + test_offsets[i];
      if (test_code < 0 || test_code > 0x7e) {
        continue;
      }

      const float test_scale = ggml_cuda_ue4m3_to_fp32((uint8_t)test_code);
      const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
#if CUDART_VERSION >= 12080
      const float cur_err = fNvfp4NativeScaleError(vals, inv_col_scale, test_inv_scale, test_scale);
#else
      float cur_err = 0.0f;
#pragma unroll
      for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        const float v = vals[k] * inv_col_scale;
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
        const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * test_scale;
        cur_err = fmaf(err_diff, err_diff, cur_err);
      }
#endif // CUDART_VERSION >= 12080

      if (cur_err < best_err) {
        best_err = cur_err;
        fp8_code = (uint8_t)test_code;
        subblock_scale = test_scale;
      }
    }
#if CUDART_VERSION >= 12080
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    const float s = inv_col_scale * inv_scale;

    __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8] * s, vals[1] * s, vals[9] * s));
    __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
    __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
    __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

    const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
    const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
    const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
    const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

    q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) << 8) | (uint32_t(uint8_t(q0_hi_c.x)) << 16) |
         (uint32_t(uint8_t(q0_hi_c.y)) << 24);
    q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) << 8) | (uint32_t(uint8_t(q1_hi_c.x)) << 16) |
         (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
      q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 0] * inv_col_scale, inv_scale)) << (8 * k);
      q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 8] * inv_col_scale, inv_scale)) << (8 * k + 4);
      q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 4] * inv_col_scale, inv_scale)) << (8 * k);
      q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale)) << (8 * k + 4);
    }
#endif // CUDART_VERSION >= 12080

    if constexpr (scatter) {
#pragma unroll
      for (int slot = 0; slot < pNExpertUsed; ++slot) {
        const int64_t i = pIds[(int64_t)blockIdx.x * pNExpertUsed + slot];
        block_fp4_mmq *yb = y + (k_block * pNe1 + i);
        uint32_t *yqs = reinterpret_cast<uint32_t *>(yb->qs);
        yqs[2 * sub + 0] = q0;
        yqs[2 * sub + 1] = q1;
        reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
      }
    } else {
      block_fp4_mmq *yb = y + (blockIdx.y * ((int64_t)blocks_per_col * pNe1) + k_block * pNe1 + blockIdx.x);
      uint32_t *yqs = reinterpret_cast<uint32_t *>(yb->qs);
      yqs[2 * sub + 0] = q0;
      yqs[2 * sub + 1] = q1;
      reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
    }
  }
#else
  GGML_UNUSED_VARS(pX, pIds, pVy, pScale, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, pNExpertUsed);
  NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter>
static __global__ void fQuantizeMmqMxfp4(const float *__restrict__ pX, const int32_t *__restrict__ pIds,
                                         void *__restrict__ pVy, const int64_t pNe00, const int64_t pS01,
                                         const int64_t pS02, const int64_t pS03, const int64_t pNe0, const int pNe1,
                                         const int pNe2, const int pNExpertUsed) {
  constexpr int cValsPerScale = 32;
  constexpr int cValsPerWarp = 2 * cValsPerScale; // Each warp processes 2 blocks of 32 = 64 values

  const int cWarpId = threadIdx.y;
  const int cLaneId32 = threadIdx.x;

  const int cNwarps = blockDim.y;

  const int64_t cWarpStartOffset = (blockIdx.y * cNwarps + cWarpId) * cValsPerWarp;

  if (cWarpStartOffset >= pNe0) {
    return;
  }

  const int64_t cBlockFp4MmqSize = QK_FP4_MMQ;
  const int64_t cKBlock = cWarpStartOffset / cBlockFp4MmqSize;
  const int64_t cQuadIdxInBlock = (cWarpStartOffset % cBlockFp4MmqSize) / cValsPerWarp;

  const int cGroupId = cLaneId32 / 4;
  const int cLaneInGroup = cLaneId32 % 4;
  const int cBase = cGroupId * 2;

  ggml_cuda_pdl_sync();
  int64_t vBasePos;
  if constexpr (scatter) {
    vBasePos = (int64_t)blockIdx.x * pS02; // one physical row per token
  } else {
    const int64_t cI2 = blockIdx.z % pNe2;
    const int64_t cI3 = blockIdx.z / pNe2;
    const int64_t cI01 = pIds ? pIds[blockIdx.x] : blockIdx.x;
    vBasePos = cI3 * pS03 + cI2 * pS02 + cI01 * pS01;
  }

  uint8_t aScales[2];
  char2 vPacked[2];

#pragma unroll
  for (int vB = 0; vB < 2; ++vB) {
    const int64_t cI0 = cWarpStartOffset + vB * cValsPerScale + cLaneId32;
    const float cXi = (cI0 < pNe00) ? pX[vBasePos + cI0] : 0.0f;

    float vAmax = fabsf(cXi);
#pragma unroll
    for (int vMask = 16; vMask > 0; vMask >>= 1) {
      vAmax = fmaxf(vAmax, __shfl_xor_sync(0xFFFFFFFF, vAmax, vMask, WARP_SIZE));
    }

    const uint8_t cE = fComputeE8m0Scale(vAmax);
    aScales[vB] = cE;
    const float cInvS = (vAmax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(cE));

#if CUDART_VERSION >= 12080
    const float cScaledVal = cXi * cInvS;

    const float cVal0 = __shfl_sync(0xFFFFFFFF, cScaledVal, cBase, WARP_SIZE);
    const float cVal1 = __shfl_sync(0xFFFFFFFF, cScaledVal, cBase + 16, WARP_SIZE);
    const float cVal2 = __shfl_sync(0xFFFFFFFF, cScaledVal, cBase + 1, WARP_SIZE);
    const float cVal3 = __shfl_sync(0xFFFFFFFF, cScaledVal, cBase + 17, WARP_SIZE);

    __nv_fp4x4_e2m1 vFp4Packed(make_float4(cVal0, cVal1, cVal2, cVal3));
    vPacked[vB] = *(char2 *)&vFp4Packed;
#else
    // Fallback: manual FP4 conversion using LUT
    const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(cXi, cInvS);

    const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, cBase, WARP_SIZE);
    const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, cBase + 1, WARP_SIZE);
    const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, cBase + 16, WARP_SIZE);
    const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, cBase + 17, WARP_SIZE);

    char2 q;
    q.x = (q_hi_0 << 4) | q_lo_0;
    q.y = (q_hi_1 << 4) | q_lo_1;
    vPacked[vB] = q;
#endif // CUDART_VERSION >= 12080
  }

  block_fp4_mmq *vY = (block_fp4_mmq *)pVy;
  if constexpr (scatter) {
#pragma unroll
    for (int slot = 0; slot < pNExpertUsed; ++slot) {
      const int64_t i = pIds[(int64_t)blockIdx.x * pNExpertUsed + slot];
      block_fp4_mmq *yb = vY + (cKBlock * pNe1 + i);
      char2 *yqs2 = (char2 *)yb->qs;
      if (cLaneInGroup == 0) {
        yqs2[cQuadIdxInBlock * 16 + 0 * 8 + cGroupId] = vPacked[0];
        yqs2[cQuadIdxInBlock * 16 + 1 * 8 + cGroupId] = vPacked[1];
      }
      if (cLaneId32 == 0) {
        yb->d4[cQuadIdxInBlock] = (aScales[1] << 8) | aScales[0];
      }
    }
  } else {
    const int64_t cIb0 = blockIdx.z * ((int64_t)pNe1 * (pNe0 / cBlockFp4MmqSize));
    block_fp4_mmq *vYb = vY + (cIb0 + cKBlock * pNe1 + blockIdx.x);
    char2 *vYqs2 = (char2 *)vYb->qs;
    if (cLaneInGroup == 0) {
      vYqs2[cQuadIdxInBlock * 16 + 0 * 8 + cGroupId] = vPacked[0];
      vYqs2[cQuadIdxInBlock * 16 + 1 * 8 + cGroupId] = vPacked[1];
    }
    if (cLaneId32 == 0) {
      vYb->d4[cQuadIdxInBlock] = (aScales[1] << 8) | aScales[0];
    }
  }
  GGML_UNUSED(pNExpertUsed);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void fQuantizeMmqQ81(const float *__restrict__ pX, const int32_t *__restrict__ pIds,
                                       void *__restrict__ pVy, const int64_t pNe00, const int64_t pS01,
                                       const int64_t pS02, const int64_t pS03, const int64_t pNe0, const int pNe1,
                                       const int pNe2, const int pNExpertUsed) {

  constexpr int cValsPerScale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int cValsPerSum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

  const int64_t cI0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (cI0 >= pNe0) {
    return;
  }

  const int64_t cI00 = cI0;
  ggml_cuda_pdl_sync();

  int64_t vBaseIdx;
  if constexpr (scatter) {
    vBaseIdx = (int64_t)blockIdx.x * pS02; // one physical row per token
  } else {
    const int64_t i2 = blockIdx.z % pNe2;
    const int64_t i3 = blockIdx.z / pNe2;
    const int64_t i01 = pIds ? pIds[blockIdx.x] : blockIdx.x;
    vBaseIdx = i3 * pS03 + i2 * pS02 + i01 * pS01;
  }

  const float4 *vX4 = (const float4 *)pX;
  block_q8_1_mmq *vY = (block_q8_1_mmq *)pVy;

  const int64_t cKBlock = cI0 / QK8_1_MMQ; // column block in the channel
  const int64_t cIqs = cI0 % QK8_1_MMQ;    // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 cXi = cI0 < pNe00 ? vX4[(vBaseIdx + cI00) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float vAmax = fabsf(cXi.x);
  vAmax = fmaxf(vAmax, fabsf(cXi.y));
  vAmax = fmaxf(vAmax, fabsf(cXi.z));
  vAmax = fmaxf(vAmax, fabsf(cXi.w));

  // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
  for (int vOffset = cValsPerScale / 8; vOffset > 0; vOffset >>= 1) {
    vAmax = fmaxf(vAmax, __shfl_xor_sync(0xFFFFFFFF, vAmax, vOffset, WARP_SIZE));
  }

  float vSum;
  if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    vSum = cXi.x + cXi.y + cXi.z + cXi.w;

    // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
    for (int vOffset = cValsPerSum / 8; vOffset > 0; vOffset >>= 1) {
      vSum += __shfl_xor_sync(0xFFFFFFFF, vSum, vOffset, WARP_SIZE);
    }
  }

  const float cDInv = 127.0f / vAmax;
  char4 vQ;
  vQ.x = roundf(cXi.x * cDInv);
  vQ.y = roundf(cXi.y * cDInv);
  vQ.z = roundf(cXi.z * cDInv);
  vQ.w = roundf(cXi.w * cDInv);
  const float cD = 1.0f / cDInv;

  // write the block once (normal) or to each of the token's compact rows (scatter)
  const int cNwrite = scatter ? pNExpertUsed : 1;
#pragma unroll
  for (int vSlot = 0; vSlot < cNwrite; ++vSlot) {
    int64_t vIb;
    if constexpr (scatter) {
      const int64_t cI = pIds[(int64_t)blockIdx.x * pNExpertUsed + vSlot];
      vIb = cKBlock * pNe1 + cI;
    } else {
      const int64_t ib0 = blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x / QK8_1); // first block of channel
      vIb = ib0 + cKBlock * pNe1 + blockIdx.x;
    }

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 *vYqs4 = (char4 *)vY[vIb].qs;
    vYqs4[cIqs / 4] = vQ;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
      if (cIqs % 16 == 0 && cIqs < 96) {
        vY[vIb].d2s6[2 + cIqs / 16] = vSum;
        if (cIqs % 64 == 0) {
          vY[vIb].d2s6[cIqs / 64] = cD;
        }
      }
    } else if (cIqs % 32 == 0) {
      if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        vY[vIb].ds4[cIqs / 32] = make_half2(cD, vSum);
      } else {
        vY[vIb].d4[cIqs / 32] = cD;
      }
    }
  }
  GGML_UNUSED(pNExpertUsed);
}

void quantize_row_q8_1_cuda(const float *pX, const int32_t *pIds, void *pVy, const ggml_type pTypeSrc0,
                            const int64_t pNe00, const int64_t pS01, const int64_t pS02, const int64_t pS03,
                            const int64_t pNe0, const int64_t pNe1, const int64_t pNe2, const int64_t pNe3,
                            cudaStream_t pStream) {
  GGML_ASSERT(!pIds);
  GGML_ASSERT(pNe0 % QK8_1 == 0);

  const uint3 cNe2Fastdiv = init_fastdiv_values(pNe2);

  const int64_t cBlockNumX = (pNe0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  const dim3 cNumBlocks(cBlockNumX, pNe1, pNe2 * pNe3);
  const dim3 cBlockSize(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cNumBlocks, cBlockSize, 0, pStream);
  ggml_cuda_kernel_launch(fQuantizeQ81, cLaunchParams, pX, pVy, pNe00, pS01, pS02, pS03, pNe0, pNe1, cNe2Fastdiv);
  GGML_UNUSED(pTypeSrc0);
}

void quantize_mmq_q8_1_cuda(const float *pX, const int32_t *pIds, void *pVy, const ggml_type pTypeSrc0,
                            const int64_t pNe00, const int64_t pS01, const int64_t pS02, const int64_t pS03,
                            const int64_t pNe0, const int64_t pNe1, const int64_t pNe2, const int64_t pNe3,
                            cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 4 == 0);
  GGML_ASSERT(pNe0 % QK8_1_MMQ == 0);

  // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
  const int64_t cBlockNumY = (pNe0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 cNumBlocks(pNe1, cBlockNumY, pNe2 * pNe3);
  const dim3 cBlockSize(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  switch (mmq_get_q8_1_ds_layout(pTypeSrc0)) {
  case MMQ_Q8_1_DS_LAYOUT_D4:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_D4, false><<<cNumBlocks, cBlockSize, 0, pStream>>>(
        pX, pIds, pVy, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, /*n_expert_used=*/0);
    break;
  case MMQ_Q8_1_DS_LAYOUT_DS4:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_DS4, false><<<cNumBlocks, cBlockSize, 0, pStream>>>(
        pX, pIds, pVy, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, /*n_expert_used=*/0);
    break;
  case MMQ_Q8_1_DS_LAYOUT_D2S6:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_D2S6, false><<<cNumBlocks, cBlockSize, 0, pStream>>>(
        pX, pIds, pVy, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, /*n_expert_used=*/0);
    break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

// scatter=true reuses the quant kernel: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_q8_1_cuda(const float *pX, const int32_t *pIdsSrc1Inv, void *pVy, const ggml_type pTypeSrc0,
                                    const int64_t pNe00, const int64_t pStrideToken, const int64_t pNe0,
                                    const int64_t pNTokens, const int64_t pNrowsDst, const int pNExpertUsed,
                                    cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 4 == 0);
  GGML_ASSERT(pNe0 % QK8_1_MMQ == 0);

  const int64_t cBlockNumY = (pNe0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 cNumBlocks(pNTokens, cBlockNumY, 1);
  const dim3 cBlockSize(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  switch (mmq_get_q8_1_ds_layout(pTypeSrc0)) {
  case MMQ_Q8_1_DS_LAYOUT_D4:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_D4, true>
        <<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pNe00, /*s01=*/0, /*s02=*/pStrideToken,
                                                 /*s03=*/0, pNe0, /*ne1=*/(int)pNrowsDst, /*ne2=*/1, pNExpertUsed);
    break;
  case MMQ_Q8_1_DS_LAYOUT_DS4:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_DS4, true>
        <<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pNe00, /*s01=*/0, /*s02=*/pStrideToken,
                                                 /*s03=*/0, pNe0, /*ne1=*/(int)pNrowsDst, /*ne2=*/1, pNExpertUsed);
    break;
  case MMQ_Q8_1_DS_LAYOUT_D2S6:
    fQuantizeMmqQ81<MMQ_Q8_1_DS_LAYOUT_D2S6, true>
        <<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pNe00, /*s01=*/0, /*s02=*/pStrideToken,
                                                 /*s03=*/0, pNe0, /*ne1=*/(int)pNrowsDst, /*ne2=*/1, pNExpertUsed);
    break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

// scatter=true reuses the quant kernels: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_fp4_cuda(const float *pX, const int32_t *pIdsSrc1Inv, void *pVy, float *pScale,
                                   const ggml_type pTypeSrc0, const bool pUseAlignedFloat8, const int64_t pNe00,
                                   const int64_t pStrideToken, const int64_t pNe0, const int64_t pNTokens,
                                   const int64_t pNrowsDst, const int pNExpertUsed, cudaStream_t pStream) {
  GGML_ASSERT(pNe0 > 0);
  if (pTypeSrc0 == GGML_TYPE_NVFP4) {
    GGML_ASSERT(pScale);
    GGML_ASSERT(pNe00 % QK_NVFP4 == 0);
    const dim3 cBlockSize(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const dim3 cNumBlocks(pNTokens, 1, 1);
    if (pUseAlignedFloat8) {
      fQuantizeMmqNvfp4<true, true>
          <<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pScale, pNe00, /*s01=*/0, /*s02=*/pStrideToken,
                                                   /*s03=*/0, pNe0, /*ne1=*/pNrowsDst, /*ne2=*/1, pNExpertUsed);
    } else {
      fQuantizeMmqNvfp4<true, false>
          <<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pScale, pNe00, /*s01=*/0, /*s02=*/pStrideToken,
                                                   /*s03=*/0, pNe0, /*ne1=*/pNrowsDst, /*ne2=*/1, pNExpertUsed);
    }
  } else {
    GGML_ASSERT(pTypeSrc0 == GGML_TYPE_MXFP4);
    constexpr int cNwarps = 8;
    constexpr int cValsPerBlock = cNwarps * 2 * QK_MXFP4;
    const int64_t cBlockNumY = (pNe0 + cValsPerBlock - 1) / cValsPerBlock;
    const dim3 cBlockSize(WARP_SIZE, cNwarps, 1);
    const dim3 cNumBlocks(pNTokens, cBlockNumY, 1);
    fQuantizeMmqMxfp4<true><<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIdsSrc1Inv, pVy, pNe00, /*s01=*/0,
                                                                    /*s02=*/pStrideToken, /*s03=*/0, pNe0,
                                                                    /*ne1=*/(int)pNrowsDst, /*ne2=*/1, pNExpertUsed);
  }
}

void quantize_mmq_fp4_cuda(const float *pX, const int32_t *pIds, void *pVy, float *pScale, const ggml_type pTypeSrc0,
                           const bool pUseAlignedFloat8, const int64_t pNe00, const int64_t pS01, const int64_t pS02,
                           const int64_t pS03, const int64_t pNe0, const int64_t pNe1, const int64_t pNe2,
                           const int64_t pNe3, cudaStream_t pStream) {
  GGML_ASSERT(pTypeSrc0 == GGML_TYPE_MXFP4 || pTypeSrc0 == GGML_TYPE_NVFP4);
  GGML_ASSERT(pNe0 > 0);

  if (pTypeSrc0 == GGML_TYPE_NVFP4) {
    GGML_ASSERT(pScale);
    GGML_ASSERT(pNe00 % QK_NVFP4 == 0);
    const dim3 cBlockSize(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const dim3 cNumBlocks(pNe1, pNe2 * pNe3, 1);
    if (pUseAlignedFloat8) {
      fQuantizeMmqNvfp4<false, true><<<cNumBlocks, cBlockSize, 0, pStream>>>(
          pX, pIds, pVy, pScale, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, /*n_expert_used=*/0);
    } else {
      fQuantizeMmqNvfp4<false, false><<<cNumBlocks, cBlockSize, 0, pStream>>>(
          pX, pIds, pVy, pScale, pNe00, pS01, pS02, pS03, pNe0, pNe1, pNe2, /*n_expert_used=*/0);
    }
  } else {
    GGML_ASSERT(pNe0 % (2 * QK_MXFP4) == 0);

    constexpr int cNwarps = 8;
    constexpr int cValsPerWarp = 2 * QK_MXFP4;
    constexpr int cValsPerBlock = cNwarps * cValsPerWarp;

    const int64_t cBlockNumY = (pNe0 + cValsPerBlock - 1) / cValsPerBlock;
    const dim3 cNumBlocks(pNe1, cBlockNumY, pNe2 * pNe3);
    const dim3 cBlockSize(WARP_SIZE, cNwarps, 1);

    fQuantizeMmqMxfp4<false><<<cNumBlocks, cBlockSize, 0, pStream>>>(pX, pIds, pVy, pNe00, pS01, pS02, pS03, pNe0, pNe1,
                                                                     pNe2, /*n_expert_used=*/0);
  }
}
