#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int pCc) {
  return 128;
  GGML_UNUSED(pCc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() { return 128; }

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif                                                                                  // __clang__
template <int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1) static __global__
    void flash_attn_ext_vec(const char *pQPtr, const char *pKPtr, const char *pVPtr, const char *pMaskPtr,
                            const char *pSinksPtr, const int *pKvMaxPtr, float *pDstPtr, float2 *pDstMetaPtr,
                            const float pScale, const float pMaxBias, const float pM0, const float pM1,
                            const uint32_t pNHeadLog2, const float pLogitSoftcap, const int32_t pNe00,
                            const uint3 pNe01, const int32_t pNe02, const int32_t pNe03, const int32_t pNb01,
                            const int32_t pNb02, const int32_t pNb03, const int32_t pNe10, const int32_t pNe11,
                            const int32_t pNe12, const int32_t pNe13, const int32_t pNb11, const int32_t pNb12,
                            const int64_t pNb13, const int32_t pNb21, const int32_t pNb22, const int64_t pNb23,
                            const int32_t pNe31, const int32_t pNe32, const int32_t pNe33, const int32_t pNb31,
                            const int32_t pNb32, const int64_t pNb33) {
  ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
  const char *GGML_CUDA_RESTRICT vQ = pQPtr;
  const char *GGML_CUDA_RESTRICT K = pKPtr;
  const char *GGML_CUDA_RESTRICT vV = pVPtr;
  const char *GGML_CUDA_RESTRICT vMask = pMaskPtr;
  const char *GGML_CUDA_RESTRICT vSinks = pSinksPtr;
  const int *GGML_CUDA_RESTRICT vKvMax = pKvMaxPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  float2 *GGML_CUDA_RESTRICT vDstMeta = pDstMetaPtr;

  // Skip unused kernel variants for faster compilation:
  if (use_logit_softcap && !(D == 128 || D == 256)) {
    GGML_UNUSED_VARS(vQ, K, vV, vMask, vSinks, vKvMax, vDst, vDstMeta, pScale, pMaxBias, pM0, pM1, pNHeadLog2,
                     pLogitSoftcap, pNe00, pNe01, pNe02, pNe03, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNe13, pNb11,
                     pNb12, pNb13, pNb21, pNb22, pNb23, pNe31, pNe32, pNe33, pNb31, pNb32, pNb33);
    NO_DEVICE_CODE;
    return;
  }

  // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
  constexpr int cNthreadsKqQ = 2;
#else
  constexpr int cNthreadsKqQ = 4;
#endif // RDNA
  constexpr int cNthreadsVQ = (D / 4 < 32 ? D / 4 : 32);
#else
  constexpr int cNthreadsKqQ = (D / 4 < 32 ? D / 4 : 32);
  constexpr int cNthreadsVQ = (D / 4 < 32 ? D / 4 : 32);
#endif // GGML_USE_HIP

  constexpr int cNthreads = ggml_cuda_fattn_vec_get_nthreads_device();
  constexpr int cNthreadsKq = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cCpyNb : cNthreadsKqQ;
  constexpr int cNthreadsV = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cCpyNb : cNthreadsVQ;

  static_assert(WARP_SIZE % cNthreadsKq == 0, "bad nthreads_K");
  static_assert(WARP_SIZE % cNthreadsV == 0, "bad nthreads_V");

  constexpr int cVRowsPerThread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2 * cCpyNe : 4;
  constexpr int cVColsPerIter = WARP_SIZE / cNthreadsV;

  constexpr vec_dot_KQ_t cVecDotKq = get_vec_dot_KQ<type_K, D, cNthreadsKq>();
  constexpr bool cQQ81 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
  constexpr dequantize_V_t cDequantizeV = get_dequantize_V<type_V, half, cVRowsPerThread>();
#else
  constexpr dequantize_V_t cDequantizeV = get_dequantize_V<type_V, float, cVRowsPerThread>();
#endif // V_DOT2_F32_F16_AVAILABLE

  const int cIc0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

  const int cSequence = blockIdx.z / pNe02;
  const int cHead = blockIdx.z - cSequence * pNe02;
  const int cGqaRatio = pNe02 / pNe12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
  vQ += pNb03 * cSequence + pNb02 * cHead + pNb01 * cIc0;
  K += pNb13 * cSequence + pNb12 * (cHead / cGqaRatio);
  vV += pNb23 * cSequence + pNb22 * (cHead / cGqaRatio);

  const half *vMaskh = (const half *)(vMask + pNb33 * (cSequence % pNe33) + pNb31 * cIc0);

  const float cSlope = get_alibi_slope(pMaxBias, cHead, pNHeadLog2, pM0, pM1);

  static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
  constexpr int cNwarps = cNthreads / WARP_SIZE;
  const int cTid = WARP_SIZE * threadIdx.y + threadIdx.x;
  __builtin_assume(cTid < cNthreads);

  constexpr int cNeKq = ncols * D;
  constexpr int cNeCombine = cNwarps * cVColsPerIter * D;
#ifdef V_DOT2_F32_F16_AVAILABLE
  half2 vVkq[ncols][(D / 2) / cNthreadsV] = {{{0.0f, 0.0f}}};
  __shared__ half vKq[cNeKq > cNeCombine ? cNeKq : cNeCombine];
#else
  float2 vVkq[ncols][(D / 2) / cNthreadsV] = {{{0.0f, 0.0f}}};
  __shared__ float vKq[cNeKq > cNeCombine ? cNeKq : cNeCombine];
#endif // V_DOT2_F32_F16_AVAILABLE

  float vKqMax[ncols];
  float vKqSum[ncols];
#pragma unroll
  for (int j = 0; j < ncols; ++j) {
    vKqMax[j] = -FLT_MAX / 2.0f;
    vKqSum[j] = 0.0f;
  }

  // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
  half2 vQReg[ncols][(D / 2) / cNthreadsKq]; // Will be initialized completely.
#else
  __align__(16) float2 vQReg[ncols][(D / 2) / cNthreadsKq] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
  int vQI32[ncols][1 > D / (sizeof(int) * cNthreadsKq) ? 1 : D / (sizeof(int) * cNthreadsKq)];
  float2 vQDs[ncols][1 > D / (sizeof(int) * cNthreadsKq) ? 1 : D / (sizeof(int) * cNthreadsKq)];

  ggml_cuda_pdl_sync();
  if constexpr (cQQ81) {
#pragma unroll
    for (int vJ0 = 0; vJ0 < ncols; vJ0 += cNwarps) {
      const int j = vJ0 + threadIdx.y;

      if (vJ0 + cNwarps > ncols && j >= ncols) {
        break;
      }

      // Reuse KQ as temporary storage for converting Q to q8_1:
      int *vTmpQI32 = (int *)&vKq[j * D];
      float2 *vTmpQDs = (float2 *)(vTmpQI32 + D / sizeof(int));

      // Set memory to zero if out of bounds:
      if (ncols > 1 && cIc0 + j >= int(pNe01.z)) {
#pragma unroll
        for (int vI0 = 0; vI0 < int(D / sizeof(int)); vI0 += WARP_SIZE) {
          const int cI = vI0 + threadIdx.x;

          if (vI0 + WARP_SIZE <= int(D / sizeof(int)) || cI < int(D / sizeof(int))) {
            vTmpQI32[cI] = 0;
          }
        }
        if (threadIdx.x < D / QK8_1) {
          vTmpQDs[threadIdx.x] = make_float2(0.0f, 0.0f);
        }
      } else {
        const float *vQF = (const float *)(vQ + j * pNb01);
        constexpr int cNthreadsQuantize = D / sizeof(int) < WARP_SIZE ? D / sizeof(int) : WARP_SIZE;
#pragma unroll
        for (int vI0 = 0; vI0 < int(D / sizeof(int)); vI0 += cNthreadsQuantize) {
          quantize_q8_1_to_shared<float2, cNthreadsQuantize>(vQF + vI0 * sizeof(int), pScale, vTmpQI32 + vI0,
                                                             vTmpQDs + vI0 / QI8_1);
        }
      }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
      int *vTmpQI32 = (int *)&vKq[j * D];
      float2 *vTmpQDs = (float2 *)(vTmpQI32 + D / sizeof(int));

#pragma unroll
      for (int vI0 = 0; vI0 < int(D / sizeof(int)); vI0 += cNthreadsKq) {
        const int cI = vI0 + (cNthreadsKq == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsKq);

        vQI32[j][vI0 / cNthreadsKq] = vTmpQI32[cI];
        vQDs[j][vI0 / cNthreadsKq] = vTmpQDs[cI / QI8_1];
      }
    }

    __syncthreads();
  } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
    const half2 scale_h2 = make_half2(pScale, pScale);
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
      const float2 *vQJ = (const float2 *)(vQ + j * pNb01);
#pragma unroll
      for (int vI0 = 0; vI0 < D / 2; vI0 += cNthreadsKq * cCpyNe) {
        const int cI = vI0 + (cNthreadsKq == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsKq) * cCpyNe;

        __align__(16) float2 vTmp[cCpyNe] = {{0.0f, 0.0f}};
        if (ncols == 1 || cIc0 + j < int(pNe01.z)) {
          ggml_cuda_memcpy_1<cCpyNb>(vTmp, &vQJ[cI]);
          ggml_cuda_memcpy_1<cCpyNb>(vTmp + cCpyNe / 2, &vQJ[cI + cCpyNe / 2]);
        }
#pragma unroll
        for (int i1 = 0; i1 < cCpyNe; ++i1) {
          vQReg[j][vI0 / cNthreadsKq + i1] = make_half2(vTmp[i1].x, vTmp[i1].y);
        }
      }
#pragma unroll
      for (int k = 0; k < (D / 2) / cNthreadsKq; ++k) {
        vQReg[j][k] *= scale_h2;
      }
    }
#else
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
      const float2 *vQJ = (const float2 *)(vQ + j * pNb01);
#pragma unroll
      for (int vI0 = 0; vI0 < D / 2; vI0 += cNthreadsKq * cCpyNe) {
        const int cI = vI0 + (cNthreadsKq == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsKq) * cCpyNe;
        if (ncols == 1 || cIc0 + j < int(pNe01.z)) {
          ggml_cuda_memcpy_1<cCpyNb>(&vQReg[j][vI0 / cNthreadsKq], &vQJ[cI]);
          ggml_cuda_memcpy_1<cCpyNb>(&vQReg[j][vI0 / cNthreadsKq + cCpyNe / 2], &vQJ[cI + cCpyNe / 2]);
        }
      }
#pragma unroll
      for (int k = 0; k < (D / 2) / cNthreadsKq; ++k) {
        vQReg[j][k].x *= pScale;
        vQReg[j][k].y *= pScale;
      }
    }
#endif // V_DOT2_F32_F16_AVAILABLE
  }

  const int cKVkqMax = vKvMax ? vKvMax[cSequence * gridDim.x + blockIdx.x] : pNe11;
  K += blockIdx.y * cNthreads * pNb11;
  vV += blockIdx.y * cNthreads * pNb21;
  vMaskh += blockIdx.y * cNthreads;
  for (int vKVkq0 = blockIdx.y * cNthreads; vKVkq0 < cKVkqMax; vKVkq0 += gridDim.y * cNthreads,
           // Increment pointers after each loop:
                                                               K += gridDim.y * cNthreads * pNb11,
           vV += gridDim.y * cNthreads * pNb21, vMaskh += gridDim.y * cNthreads) {

    // Calculate KQ tile and keep track of new maximum KQ values:
    float vKqReg[ncols]; // KQ in registers.

    float vKqMaxNew[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
      vKqMaxNew[j] = vKqMax[j];
    }

#pragma unroll
    for (int vIKq0 = 0; vIKq0 < cNthreadsKq; ++vIKq0) {
      const int cIKq =
          threadIdx.y * WARP_SIZE + (cNthreadsKq == WARP_SIZE ? 0 : (threadIdx.x & ~(cNthreadsKq - 1))) + vIKq0;

#pragma unroll
      for (int j = 0; j < ncols; ++j) {
        float vSum = cVecDotKq(K + cIKq * pNb11, vQReg[j], vQI32[j], vQDs[j]);
        vSum = warp_reduce_sum<cNthreadsKq>(vSum);

        if (use_logit_softcap) {
          vSum = pLogitSoftcap * tanhf(vSum);
        }

        if (vMask && (ncols == 1 || cIc0 + j < int(pNe01.z))) {
          vSum += cSlope * __half2float(vMaskh[j * pNe11 + cIKq]);
        }

        vKqMaxNew[j] = fmaxf(vKqMaxNew[j], vSum + FATTN_KQ_MAX_OFFSET);

        if ((cNthreadsKq == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsKq) == uint32_t(vIKq0)) {
          vKqReg[j] = vSum;
        }
      }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
      for (int vOffset = cNthreadsKq; vOffset < WARP_SIZE; vOffset <<= 1) {
        vKqMaxNew[j] = fmaxf(vKqMaxNew[j], __shfl_xor_sync(0xFFFFFFFF, vKqMaxNew[j], vOffset, WARP_SIZE));
      }
      const float cKqMaxScale = expf(vKqMax[j] - vKqMaxNew[j]);
      vKqMax[j] = vKqMaxNew[j];

      vKqReg[j] = expf(vKqReg[j] - vKqMax[j]);
      vKqSum[j] = vKqSum[j] * cKqMaxScale + vKqReg[j];
      vKq[j * cNthreads + cTid] = vKqReg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
      const half2 KQ_max_scale_h2 = make_half2(cKqMaxScale, cKqMaxScale);
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
        vVkq[j][vIVkq0 / cNthreadsV] *= KQ_max_scale_h2;
      }
#else
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
        vVkq[j][vIVkq0 / cNthreadsV].x *= cKqMaxScale;
        vVkq[j][vIVkq0 / cNthreadsV].y *= cKqMaxScale;
      }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    ggml_cuda_syncwarp();

#pragma unroll
    for (int vK0 = 0; vK0 < WARP_SIZE; vK0 += cVColsPerIter) {
      const int k = threadIdx.y * WARP_SIZE + vK0 + (cNthreadsV == WARP_SIZE ? 0 : threadIdx.x / cNthreadsV);

#ifdef V_DOT2_F32_F16_AVAILABLE
      half2 vKqK[ncols];
#pragma unroll
      for (int j = 0; j < ncols; ++j) {
        vKqK[j] = __half2half2(vKq[j * cNthreads + k]);
      }
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV * cVRowsPerThread / 2) {
        half2 vTmp[cVRowsPerThread / 2];
        if constexpr (type_V == GGML_TYPE_BF16) {
          float2 tmp_f[cVRowsPerThread / 2];
          cDequantizeV(vV + k * pNb21, tmp_f,
                       2 * vIVkq0 +
                           (cNthreadsV == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsV) * cVRowsPerThread);
#pragma unroll
          for (int vIVkq1 = 0; vIVkq1 < cVRowsPerThread / 2; ++vIVkq1) {
            vTmp[vIVkq1] = __float22half2_rn(tmp_f[vIVkq1]);
          }
        } else {
          cDequantizeV(vV + k * pNb21, vTmp,
                       2 * vIVkq0 +
                           (cNthreadsV == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsV) * cVRowsPerThread);
        }
#pragma unroll
        for (int vIVkq1 = 0; vIVkq1 < cVRowsPerThread / 2; ++vIVkq1) {
#pragma unroll
          for (int j = 0; j < ncols; ++j) {
            vVkq[j][vIVkq0 / cNthreadsV + vIVkq1] += vTmp[vIVkq1] * vKqK[j];
          }
        }
      }
#else
      float vKqK[ncols];
#pragma unroll
      for (int j = 0; j < ncols; ++j) {
        vKqK[j] = vKq[j * cNthreads + k];
      }
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV * cVRowsPerThread / 2) {
        float2 vTmp[cVRowsPerThread / 2];
        cDequantizeV(vV + k * pNb21, vTmp,
                     2 * vIVkq0 + (cNthreadsV == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsV) * cVRowsPerThread);
#pragma unroll
        for (int vIVkq1 = 0; vIVkq1 < cVRowsPerThread / 2; ++vIVkq1) {
#pragma unroll
          for (int j = 0; j < ncols; ++j) {
            vVkq[j][vIVkq0 / cNthreadsV + vIVkq1].x += vTmp[vIVkq1].x * vKqK[j];
            vVkq[j][vIVkq0 / cNthreadsV + vIVkq1].y += vTmp[vIVkq1].y * vKqK[j];
          }
        }
      }
#endif // V_DOT2_F32_F16_AVAILABLE
    }
  }

  if (vSinks && blockIdx.y == 0) {
    const float cSink = ((const float *)vSinks)[cHead];

#pragma unroll
    for (int vJ0 = 0; vJ0 < ncols; vJ0 += cNwarps) {
      const int j = vJ0 + threadIdx.y;

      if (vJ0 + cNwarps > ncols && j >= ncols) {
        break;
      }

      const float cKqmaxNewJ = fmaxf(cSink, vKqMax[j]);
      const float cKqMaxScale = expf(vKqMax[j] - cKqmaxNewJ);
      vKqMax[j] = cKqmaxNewJ;

      vKqSum[j] = vKqSum[j] * cKqMaxScale + (threadIdx.x == 0 ? expf(cSink - vKqMax[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
      const half2 KQ_max_scale_h2 = make_half2(cKqMaxScale, cKqMaxScale);
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
        vVkq[j][vIVkq0 / cNthreadsV] *= KQ_max_scale_h2;
      }
#else
#pragma unroll
      for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
        vVkq[j][vIVkq0 / cNthreadsV].x *= cKqMaxScale;
        vVkq[j][vIVkq0 / cNthreadsV].y *= cKqMaxScale;
      }
#endif // V_DOT2_F32_F16_AVAILABLE
    }
  }

  __shared__ float vKqMaxShared[ncols][WARP_SIZE];
  __shared__ float vKqSumShared[ncols][WARP_SIZE];
#pragma unroll
  for (int j = 0; j < ncols; ++j) {
    if (threadIdx.y == 0) {
      vKqMaxShared[j][threadIdx.x] = -FLT_MAX / 2.0f;
      vKqSumShared[j][threadIdx.x] = 0.0f;
    }
  }

  __syncthreads();

#pragma unroll
  for (int j = 0; j < ncols; ++j) {
    if (threadIdx.x == 0) {
      vKqMaxShared[j][threadIdx.y] = vKqMax[j];
    }
  }
  __syncthreads();

#pragma unroll
  for (int vJVkq = 0; vJVkq < ncols; ++vJVkq) {
    if (ncols > 1 && cIc0 + vJVkq >= int(pNe01.z)) {
      break;
    }

    float vKqmaxNew = vKqMaxShared[vJVkq][threadIdx.x];
    vKqmaxNew = warp_reduce_max(vKqmaxNew);
    const float cKqmaxScale = expf(vKqMax[vJVkq] - vKqmaxNew);
    vKqMax[vJVkq] = vKqmaxNew;

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2 *vVkqTmp = (half2 *)vKq + threadIdx.y * (cVColsPerIter * D / 2) +
                     (cNthreadsV == WARP_SIZE ? 0 : threadIdx.x / cNthreadsV) * (D / 2);

    const half2 kqmax_scale_h2 = make_half2(cKqmaxScale, cKqmaxScale);
#pragma unroll
    for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
      vVkq[vJVkq][vIVkq0 / cNthreadsV] *= kqmax_scale_h2;
    }
#pragma unroll
    for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV * cVRowsPerThread / 2) {
      const int cIVkq =
          vIVkq0 + (cNthreadsV == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsV) * (cVRowsPerThread / 2);

      ggml_cuda_memcpy_1<cVRowsPerThread * sizeof(half)>(vVkqTmp + cIVkq, &vVkq[vJVkq][vIVkq0 / cNthreadsV]);
    }
#else
    float2 *vVkqTmp = (float2 *)vKq + threadIdx.y * (cVColsPerIter * D / 2) +
                      (cNthreadsV == WARP_SIZE ? 0 : threadIdx.x / cNthreadsV) * (D / 2);

#pragma unroll
    for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV) {
      vVkq[vJVkq][vIVkq0 / cNthreadsV].x *= cKqmaxScale;
      vVkq[vJVkq][vIVkq0 / cNthreadsV].y *= cKqmaxScale;
    }
#pragma unroll
    for (int vIVkq0 = 0; vIVkq0 < D / 2; vIVkq0 += cNthreadsV * cVRowsPerThread / 2) {
      const int cIVkq =
          vIVkq0 + (cNthreadsV == WARP_SIZE ? threadIdx.x : threadIdx.x % cNthreadsV) * (cVRowsPerThread / 2);

      ggml_cuda_memcpy_1<cVRowsPerThread / 2 * sizeof(float)>(vVkqTmp + cIVkq, &vVkq[vJVkq][vIVkq0 / cNthreadsV]);
      ggml_cuda_memcpy_1<cVRowsPerThread / 2 * sizeof(float)>(vVkqTmp + cIVkq + cVRowsPerThread / 4,
                                                              &vVkq[vJVkq][vIVkq0 / cNthreadsV + cVRowsPerThread / 4]);
    }
#endif // V_DOT2_F32_F16_AVAILABLE

    vKqSum[vJVkq] *= cKqmaxScale;
    vKqSum[vJVkq] = warp_reduce_sum(vKqSum[vJVkq]);
    if (threadIdx.x == 0) {
      vKqSumShared[vJVkq][threadIdx.y] = vKqSum[vJVkq];
    }

    __syncthreads();

    if (cNthreads <= D || cTid < D) {
      vKqSum[vJVkq] = vKqSumShared[vJVkq][threadIdx.x];
      vKqSum[vJVkq] = warp_reduce_sum(vKqSum[vJVkq]);

#pragma unroll
      for (int vI0 = 0; vI0 < D; vI0 += cNthreads) {
        float vDstVal = 0;
#pragma unroll
        for (int vW = 0; vW < cNwarps; ++vW) {
#pragma unroll
          for (int v_ = 0; v_ < cVColsPerIter; ++v_) {
            vDstVal += float(vKq[vW * cVColsPerIter * D + v_ * D + vI0 + cTid]);
          }
        }
        if (gridDim.y == 1) {
          vDstVal /= vKqSum[vJVkq];
        }
        vDst[(((cSequence * int(pNe01.z) + cIc0 + vJVkq) * pNe02 + cHead) * gridDim.y + blockIdx.y) * D + vI0 + cTid] =
            vDstVal;
      }
    }

    if (vJVkq < ncols - 1) {
      __syncthreads();
    }
  }

  if (gridDim.y != 1 && cTid < ncols && (ncols == 1 || cIc0 + cTid < int(pNe01.z))) {
    vDstMeta[((cSequence * int(pNe01.z) + cIc0 + cTid) * pNe02 + cHead) * gridDim.y + blockIdx.y] =
        make_float2(vKqMax[cTid], vKqSum[cTid]);
  }
#else
  GGML_UNUSED_VARS(pQPtr, pKPtr, pVPtr, pMaskPtr, pSinksPtr, pKvMaxPtr, pDstPtr, pDstMetaPtr, pScale, pMaxBias, pM0,
                   pM1, pNHeadLog2, pLogitSoftcap, pNe00, pNe01, pNe02, pNe03, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                   pNe13, pNb11, pNb12, pNb13, pNb21, pNb22, pNb23, pNe31, pNe32, pNe33, pNb31, pNb32, pNb33);
  NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

  const int cNthreads = ggml_cuda_fattn_vec_get_nthreads_host(cCc);
  const int cNwarps = cNthreads / WARP_SIZE;
  fattn_kernel_t vFattnKernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
  const bool cNeedF16K = type_K == GGML_TYPE_F16;
  const bool cNeedF16V = type_V == GGML_TYPE_F16;
  constexpr size_t cNbytesShared = 0;
  launch_fattn<D, cols_per_block, 1>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, D, cNeedF16K, cNeedF16V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKqv = pDst;
  const ggml_tensor *vQ = pDst->src[0];

  float vLogitSoftcap;
  memcpy(&vLogitSoftcap, (const float *)vKqv->op_params + 2, sizeof(float));

  if (vQ->ne[1] == 1) {
    constexpr int cColsPerBlock = 1;
    if (vLogitSoftcap == 0.0f) {
      constexpr bool cUseLogitSoftcap = false;
      ggml_cuda_flash_attn_ext_vec_case_impl<D, cColsPerBlock, type_K, type_V, cUseLogitSoftcap>(pCtx, pDst);
    } else {
      constexpr bool cUseLogitSoftcap = true;
      ggml_cuda_flash_attn_ext_vec_case_impl<D, cColsPerBlock, type_K, type_V, cUseLogitSoftcap>(pCtx, pDst);
    }
    return;
  }

  constexpr int cColsPerBlock = 2;
  if (vLogitSoftcap == 0.0f) {
    constexpr bool cUseLogitSoftcap = false;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cColsPerBlock, type_K, type_V, cUseLogitSoftcap>(pCtx, pDst);
  } else {
    constexpr bool cUseLogitSoftcap = true;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cColsPerBlock, type_K, type_V, cUseLogitSoftcap>(pCtx, pDst);
  }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                                                                         \
  template void ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)                                                                         \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);                                                                \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0);                                                               \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1);                                                               \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0);                                                               \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1);                                                               \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0);                                                               \
  extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16);

EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)
