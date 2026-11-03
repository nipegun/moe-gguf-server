#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void fDequantizeBlock(const void *__restrict__ pVx, dst_t *__restrict__ pY, const int64_t pNe00,
                                        const int64_t pNe01, const int64_t pNe0203, const uint3 pNe02,
                                        const int64_t pS01, const int64_t pS02, const int64_t pS03) {
  const int64_t cI00 = 2 * (int64_t(blockDim.x) * blockIdx.x + threadIdx.x);

  if (cI00 >= pNe00) {
    return;
  }

  for (int64_t vI01 = blockIdx.y; vI01 < pNe01; vI01 += gridDim.y) {
    for (int64_t vI0203 = blockIdx.z; vI0203 < pNe0203; vI0203 += gridDim.z) {
      const uint2 cDm = fast_div_modulo((uint32_t)vI0203, pNe02);
      const int64_t cI02 = cDm.y;
      const int64_t cI03 = cDm.x;

      const int64_t cIbx0 = cI03 * pS03 + cI02 * pS02 + vI01 * pS01;

      const int64_t cIb = cIbx0 + cI00 / qk;  // block index
      const int64_t cIqs = (cI00 % qk) / qr;  // quant index
      const int64_t cIybs = cI00 - cI00 % qk; // y block start index
      const int64_t cYOffset = qr == 1 ? 1 : qk / 2;

      // dequantize
      float2 v_;
      dequantize_kernel(pVx, cIb, cIqs, v_);

      const int64_t cIy0 = (vI0203 * pNe01 + vI01) * pNe00 + cIybs + cIqs;
      pY[cIy0 + 0] = ggml_cuda_cast<dst_t>(v_.x);
      pY[cIy0 + cYOffset] = ggml_cuda_cast<dst_t>(v_.y);
    }
  }
}

template <bool need_check>
static __global__ void fDequantizeBlockQ80F16(const void *__restrict__ pVx, half *__restrict__ pY, const int64_t pK) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
  constexpr int cNint = CUDA_Q8_0_NE_ALIGN / sizeof(int) + WARP_SIZE;

  const int64_t cI0 = CUDA_Q8_0_NE_ALIGN * blockIdx.x;
  const int *vX0 = ((int *)pVx) + blockIdx.x * cNint;
  half2 *vY2 = (half2 *)(pY + cI0);

  __shared__ int vVals[cNint];

#pragma unroll
  for (int vIx0 = 0; vIx0 < cNint; vIx0 += WARP_SIZE) {
    if (need_check &&
        cI0 * sizeof(block_q8_0) / QK8_0 + sizeof(int) * (vIx0 + threadIdx.x) >= pK * sizeof(block_q8_0) / QK8_0) {
      break;
    }

    const int cIx = vIx0 + threadIdx.x;
    vVals[cIx] = vX0[cIx];
  }

  __syncthreads();

#pragma unroll
  for (int vIy = 0; vIy < CUDA_Q8_0_NE_ALIGN; vIy += 2 * WARP_SIZE) {
    if (need_check && cI0 + vIy + 2 * threadIdx.x >= pK) {
      return;
    }

    const half *vB0 = ((const half *)vVals) + (sizeof(block_q8_0) / sizeof(half)) * ((vIy + 2 * threadIdx.x) / QK8_0);
    const half cD = *vB0;
    const char2 cQs = ((const char2 *)(vB0 + 1))[threadIdx.x % (QK8_0 / 2)];

    vY2[vIy / 2 + threadIdx.x] = __hmul2(make_half2(cQs.x, cQs.y), __half2half2(cD));
  }
#else
  GGML_UNUSED_VARS(pVx, pY, pK);
  NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ40(const void *__restrict__ pVx, dst_t *__restrict__ pYy, int pNb32) {

  const int64_t cI = blockIdx.x;

  // assume 32 threads
  const int64_t cTid = threadIdx.x;
  const int64_t cIl = cTid / 8;
  const int64_t cIr = cTid % 8;
  const int64_t cIb = 8 * cI + cIr;
  if (cIb >= pNb32) {
    return;
  }

  dst_t *vY = pYy + 256 * cI + 32 * cIr + 4 * cIl;

  const block_q4_0 *vX = (const block_q4_0 *)pVx + cIb;
  const float cD = __half2float(vX->d);
  const float cDm = -8 * cD;

  const uint8_t *vQ = vX->qs + 4 * cIl;

  for (int vL = 0; vL < 4; ++vL) {
    vY[vL + 0] = ggml_cuda_cast<dst_t>(cD * (vQ[vL] & 0xF) + cDm);
    vY[vL + 16] = ggml_cuda_cast<dst_t>(cD * (vQ[vL] >> 4) + cDm);
  }
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ41(const void *__restrict__ pVx, dst_t *__restrict__ pYy, int pNb32) {

  const int64_t cI = blockIdx.x;

  // assume 32 threads
  const int64_t cTid = threadIdx.x;
  const int64_t cIl = cTid / 8;
  const int64_t cIr = cTid % 8;
  const int64_t cIb = 8 * cI + cIr;
  if (cIb >= pNb32) {
    return;
  }

  dst_t *vY = pYy + 256 * cI + 32 * cIr + 4 * cIl;

  const block_q4_1 *vX = (const block_q4_1 *)pVx + cIb;
  const float2 cD = __half22float2(vX->dm);

  const uint8_t *vQ = vX->qs + 4 * cIl;

  for (int vL = 0; vL < 4; ++vL) {
    vY[vL + 0] = ggml_cuda_cast<dst_t>(cD.x * (vQ[vL] & 0xF) + cD.y);
    vY[vL + 16] = ggml_cuda_cast<dst_t>(cD.x * (vQ[vL] >> 4) + cD.y);
  }
}

//================================== k-quants

template <typename dst_t>
static __global__ void fDequantizeBlockQ2K(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_q2_K(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ3K(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_q3_K(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ4K(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_q4_K(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ5K(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_q5_K(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockQ6K(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_q6_K(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq2Xxs(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq2_xxs(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq2Xs(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq2_xs(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq2S(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq2_s(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq3Xxs(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq3_xxs(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq3S(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq3_s(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq1S(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq1_s(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq1M(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq1_m(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq4Nl(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq4_nl(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockIq4Xs(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_iq4_xs(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <typename dst_t>
static __global__ void fDequantizeBlockMxfp4(const void *__restrict__ pVx, dst_t *__restrict__ pYy) {
  const int64_t cI = blockIdx.x;

  dequantize_mxfp4(pVx, cI, pYy + cI * QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void fDequantizeBlockCuda(const void *pVx, dst_t *pY, const int64_t pNe00, const int64_t pNe01,
                                 const int64_t pNe02, const int64_t pNe03, const int64_t pS01, const int64_t pS02,
                                 const int64_t pS03, cudaStream_t pStream) {
  const int64_t cNe0203 = pNe02 * pNe03;
  const uint3 cNe02Fdv = init_fastdiv_values(pNe02);
  const dim3 cNumBlocks((pNe00 + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
                        (int)std::min(pNe01, (int64_t)65535), (int)std::min(cNe0203, (int64_t)65535));
  fDequantizeBlock<qk, qr, dequantize_kernel><<<cNumBlocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, pStream>>>(
      pVx, pY, pNe00, pNe01, cNe0203, cNe02Fdv, pS01, pS02, pS03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void fDequantizeBlockContCuda(const void *__restrict__ pVx, dst_t *__restrict__ pY, const int64_t pK,
                                     cudaStream_t pStream) {
  fDequantizeBlockCuda<qk, qr, dequantize_kernel, dst_t>(pVx, pY, pK, 1, 1, 1, pK / qk, pK / qk, pK / qk, pStream);
}

static void fDequantizeBlockQ80F16Cuda(const void *__restrict__ pVx, half *__restrict__ pY, const int64_t pK,
                                       cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
  if (pK % CUDA_Q8_0_NE_ALIGN == 0) {
    const bool cNeedCheck = false;
    fDequantizeBlockQ80F16<cNeedCheck><<<cNumBlocks, WARP_SIZE, 0, pStream>>>(pVx, pY, pK);
  } else {
    const bool cNeedCheck = true;
    fDequantizeBlockQ80F16<cNeedCheck><<<cNumBlocks, WARP_SIZE, 0, pStream>>>(pVx, pY, pK);
  }
}

template <typename dst_t>
static void fDequantizeRowQ2KCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockQ2K<<<cNb, 64, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowQ3KCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockQ3K<<<cNb, 64, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowQ40Cuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb32 = pK / 32;
  const int cNb = (pK + 255) / 256;
  fDequantizeBlockQ40<<<cNb, 32, 0, pStream>>>(pVx, pY, cNb32);
}

template <typename dst_t>
static void fDequantizeRowQ41Cuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb32 = pK / 32;
  const int cNb = (pK + 255) / 256;
  fDequantizeBlockQ41<<<cNb, 32, 0, pStream>>>(pVx, pY, cNb32);
}

template <typename dst_t>
static void fDequantizeRowQ4KCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockQ4K<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowQ5KCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockQ5K<<<cNb, 64, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowQ6KCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockQ6K<<<cNb, 64, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq2XxsCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq2Xxs<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq2XsCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq2Xs<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq2SCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq2S<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq3XxsCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq3Xxs<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq3SCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq3S<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq1SCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq1S<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq4NlCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = (pK + QK_K - 1) / QK_K;
  fDequantizeBlockIq4Nl<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq1MCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = pK / QK_K;
  fDequantizeBlockIq1M<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowIq4XsCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = (pK + QK_K - 1) / QK_K;
  fDequantizeBlockIq4Xs<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static void fDequantizeRowMxfp4Cuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  const int cNb = (pK + QK_K - 1) / QK_K;
  fDequantizeBlockMxfp4<<<cNb, 32, 0, pStream>>>(pVx, pY);
}

template <typename dst_t>
static __global__ void fDequantizeBlockNvfp4(const void *__restrict__ pVx, dst_t *__restrict__ pYy, const int64_t pNe) {
  const int64_t cI = blockIdx.x;
  const int cTid = threadIdx.x;

  const int64_t cBase = cI * QK_NVFP4;
  if (cBase >= pNe) {
    return;
  }

  const block_nvfp4 *vX = (const block_nvfp4 *)pVx;
  const block_nvfp4 &vXb = vX[cI];

  const int cSub = cTid / (QK_NVFP4_SUB / 2);
  const int cJ = cTid % (QK_NVFP4_SUB / 2);

  const float cD = ggml_cuda_ue4m3_to_fp32(vXb.d[cSub]);
  const uint8_t cQ = vXb.qs[cSub * (QK_NVFP4_SUB / 2) + cJ];

  const int64_t cY0 = cBase + cSub * QK_NVFP4_SUB + cJ;
  const int64_t cY1 = cY0 + QK_NVFP4_SUB / 2;

  pYy[cY0] = ggml_cuda_cast<dst_t>(cD * kvalues_mxfp4[cQ & 0x0F]);
  pYy[cY1] = ggml_cuda_cast<dst_t>(cD * kvalues_mxfp4[cQ >> 4]);
}

template <typename dst_t>
static void fDequantizeRowNvfp4Cuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  GGML_ASSERT(pK % QK_NVFP4 == 0);
  const int cNb = pK / QK_NVFP4;
  fDequantizeBlockNvfp4<<<cNb, 32, 0, pStream>>>(pVx, pY, pK);
}
template <typename src_t, typename dst_t>
static __global__ void fConvertUnary(const void *__restrict__ pVx, dst_t *__restrict__ pY, const int64_t pNe00,
                                     const int64_t pNe01, const int64_t pNe0203, const uint3 pNe02, const int64_t pS01,
                                     const int64_t pS02, const int64_t pS03) {
  const int64_t cI00 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (cI00 >= pNe00) {
    return;
  }

  const src_t *vX = (const src_t *)pVx;

  for (int64_t vI01 = blockIdx.y; vI01 < pNe01; vI01 += gridDim.y) {
    for (int64_t vI0203 = blockIdx.z; vI0203 < pNe0203; vI0203 += gridDim.z) {
      const uint2 cDm = fast_div_modulo((uint32_t)vI0203, pNe02);
      const int64_t cI02 = cDm.y;
      const int64_t cI03 = cDm.x;

      const int64_t cIx = cI03 * pS03 + cI02 * pS02 + vI01 * pS01 + cI00;
      const int64_t cIy = (vI0203 * pNe01 + vI01) * pNe00 + cI00;
      pY[cIy] = ggml_cuda_cast<dst_t>(vX[cIx]);
    }
  }
}

template <typename src_t, typename dst_t>
static void fConvertUnaryCuda(const void *pVx, dst_t *pY, const int64_t pNe00, const int64_t pNe01, const int64_t pNe02,
                              const int64_t pNe03, const int64_t pS01, const int64_t pS02, const int64_t pS03,
                              cudaStream_t pStream) {
  const int64_t cNe0203 = pNe02 * pNe03;
  const uint3 cNe02Fdv = init_fastdiv_values(pNe02);
  const dim3 cNumBlocks((pNe00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE,
                        (int)std::min(pNe01, (int64_t)65535), (int)std::min(cNe0203, (int64_t)65535));
  fConvertUnary<src_t><<<cNumBlocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, pStream>>>(pVx, pY, pNe00, pNe01, cNe0203, cNe02Fdv,
                                                                               pS01, pS02, pS03);
}

template <typename src_t, typename dst_t>
static void fConvertUnaryContCuda(const void *pVx, dst_t *pY, const int64_t pK, cudaStream_t pStream) {
  fConvertUnaryCuda<src_t>(pVx, pY, pK, 1, 1, 1, pK, pK, pK, pStream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockContCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockContCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeRowQ40Cuda;
  case GGML_TYPE_Q4_1:
    return fDequantizeRowQ41Cuda;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockContCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockContCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    return fDequantizeBlockContCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_Q2_K:
    return fDequantizeRowQ2KCuda;
  case GGML_TYPE_Q3_K:
    return fDequantizeRowQ3KCuda;
  case GGML_TYPE_Q4_K:
    return fDequantizeRowQ4KCuda;
  case GGML_TYPE_Q5_K:
    return fDequantizeRowQ5KCuda;
  case GGML_TYPE_Q6_K:
    return fDequantizeRowQ6KCuda;
  case GGML_TYPE_IQ2_XXS:
    return fDequantizeRowIq2XxsCuda;
  case GGML_TYPE_IQ2_XS:
    return fDequantizeRowIq2XsCuda;
  case GGML_TYPE_IQ2_S:
    return fDequantizeRowIq2SCuda;
  case GGML_TYPE_IQ3_XXS:
    return fDequantizeRowIq3XxsCuda;
  case GGML_TYPE_IQ1_S:
    return fDequantizeRowIq1SCuda;
  case GGML_TYPE_IQ1_M:
    return fDequantizeRowIq1MCuda;
  case GGML_TYPE_IQ4_NL:
    return fDequantizeRowIq4NlCuda;
  case GGML_TYPE_IQ4_XS:
    return fDequantizeRowIq4XsCuda;
  case GGML_TYPE_IQ3_S:
    return fDequantizeRowIq3SCuda;
  case GGML_TYPE_MXFP4:
    return fDequantizeRowMxfp4Cuda;
  case GGML_TYPE_NVFP4:
    return fDequantizeRowNvfp4Cuda;
  case GGML_TYPE_F32:
    return fConvertUnaryContCuda<float>;
  case GGML_TYPE_F16:
    return fConvertUnaryContCuda<half>;
  default:
    return nullptr;
  }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockContCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockContCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeRowQ40Cuda;
  case GGML_TYPE_Q4_1:
    return fDequantizeRowQ41Cuda;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockContCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockContCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
      return fDequantizeBlockQ80F16Cuda;
    }
    return fDequantizeBlockContCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_Q2_K:
    return fDequantizeRowQ2KCuda;
  case GGML_TYPE_Q3_K:
    return fDequantizeRowQ3KCuda;
  case GGML_TYPE_Q4_K:
    return fDequantizeRowQ4KCuda;
  case GGML_TYPE_Q5_K:
    return fDequantizeRowQ5KCuda;
  case GGML_TYPE_Q6_K:
    return fDequantizeRowQ6KCuda;
  case GGML_TYPE_IQ2_XXS:
    return fDequantizeRowIq2XxsCuda;
  case GGML_TYPE_IQ2_XS:
    return fDequantizeRowIq2XsCuda;
  case GGML_TYPE_IQ2_S:
    return fDequantizeRowIq2SCuda;
  case GGML_TYPE_IQ3_XXS:
    return fDequantizeRowIq3XxsCuda;
  case GGML_TYPE_IQ1_S:
    return fDequantizeRowIq1SCuda;
  case GGML_TYPE_IQ1_M:
    return fDequantizeRowIq1MCuda;
  case GGML_TYPE_IQ4_NL:
    return fDequantizeRowIq4NlCuda;
  case GGML_TYPE_IQ4_XS:
    return fDequantizeRowIq4XsCuda;
  case GGML_TYPE_IQ3_S:
    return fDequantizeRowIq3SCuda;
  case GGML_TYPE_MXFP4:
    return fDequantizeRowMxfp4Cuda;
  case GGML_TYPE_NVFP4:
    return fDequantizeRowNvfp4Cuda;
  case GGML_TYPE_F32:
    return fConvertUnaryContCuda<float>;
  case GGML_TYPE_BF16:
    return fConvertUnaryContCuda<nv_bfloat16>;
  default:
    return nullptr;
  }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockContCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockContCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeRowQ40Cuda;
  case GGML_TYPE_Q4_1:
    return fDequantizeRowQ41Cuda;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockContCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockContCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    return fDequantizeBlockContCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_Q2_K:
    return fDequantizeRowQ2KCuda;
  case GGML_TYPE_Q3_K:
    return fDequantizeRowQ3KCuda;
  case GGML_TYPE_Q4_K:
    return fDequantizeRowQ4KCuda;
  case GGML_TYPE_Q5_K:
    return fDequantizeRowQ5KCuda;
  case GGML_TYPE_Q6_K:
    return fDequantizeRowQ6KCuda;
  case GGML_TYPE_IQ2_XXS:
    return fDequantizeRowIq2XxsCuda;
  case GGML_TYPE_IQ2_XS:
    return fDequantizeRowIq2XsCuda;
  case GGML_TYPE_IQ2_S:
    return fDequantizeRowIq2SCuda;
  case GGML_TYPE_IQ3_XXS:
    return fDequantizeRowIq3XxsCuda;
  case GGML_TYPE_IQ1_S:
    return fDequantizeRowIq1SCuda;
  case GGML_TYPE_IQ1_M:
    return fDequantizeRowIq1MCuda;
  case GGML_TYPE_IQ4_NL:
    return fDequantizeRowIq4NlCuda;
  case GGML_TYPE_IQ4_XS:
    return fDequantizeRowIq4XsCuda;
  case GGML_TYPE_IQ3_S:
    return fDequantizeRowIq3SCuda;
  case GGML_TYPE_MXFP4:
    return fDequantizeRowMxfp4Cuda;
  case GGML_TYPE_NVFP4:
    return fDequantizeRowNvfp4Cuda;
  case GGML_TYPE_F16:
    return fConvertUnaryContCuda<half>;
  case GGML_TYPE_BF16:
    return fConvertUnaryContCuda<nv_bfloat16>;
  default:
    return nullptr;
  }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_F32:
    return fConvertUnaryCuda<float>;
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeBlockCuda<QK4_0, QR4_0, dequantize_q4_0>;
  case GGML_TYPE_Q4_1:
    return fDequantizeBlockCuda<QK4_1, QR4_1, dequantize_q4_1>;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    return fDequantizeBlockCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_BF16:
    return fConvertUnaryCuda<nv_bfloat16>;
  default:
    return nullptr;
  }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_F32:
    return fConvertUnaryCuda<float, nv_bfloat16>;
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeBlockCuda<QK4_0, QR4_0, dequantize_q4_0>;
  case GGML_TYPE_Q4_1:
    return fDequantizeBlockCuda<QK4_1, QR4_1, dequantize_q4_1>;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    return fDequantizeBlockCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_F16:
    return fConvertUnaryCuda<half, nv_bfloat16>;
  default:
    return nullptr;
  }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type pType) {
  switch (pType) {
  case GGML_TYPE_F16:
    return fConvertUnaryCuda<half, float>;
  case GGML_TYPE_Q1_0:
    return fDequantizeBlockCuda<QK1_0, QR1_0, dequantize_q1_0>;
  case GGML_TYPE_Q2_0:
    return fDequantizeBlockCuda<QK2_0, QR2_0, dequantize_q2_0>;
  case GGML_TYPE_Q4_0:
    return fDequantizeBlockCuda<QK4_0, QR4_0, dequantize_q4_0>;
  case GGML_TYPE_Q4_1:
    return fDequantizeBlockCuda<QK4_1, QR4_1, dequantize_q4_1>;
  case GGML_TYPE_Q5_0:
    return fDequantizeBlockCuda<QK5_0, QR5_0, dequantize_q5_0>;
  case GGML_TYPE_Q5_1:
    return fDequantizeBlockCuda<QK5_1, QR5_1, dequantize_q5_1>;
  case GGML_TYPE_Q8_0:
    return fDequantizeBlockCuda<QK8_0, QR8_0, dequantize_q8_0>;
  case GGML_TYPE_BF16:
    return fConvertUnaryCuda<nv_bfloat16, float>;
  default:
    return nullptr;
  }
}
