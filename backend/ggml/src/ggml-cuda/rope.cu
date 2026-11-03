#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
  float v_[2];
};

struct mrope_sections {
  int v_[4];
};

static __device__ float fRopeYarnRamp(const float pLow, const float pHigh, const int pI0) {
  const float cY = (pI0 / 2 - pLow) / max(0.001f, pHigh - pLow);
  return 1.0f - min(1.0f, max(0.0f, cY));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template <bool forward>
static __device__ void fRopeYarn(const float pThetaExtrap, const float pFreqScale, const rope_corr_dims pCorrDims,
                                 const int64_t pI0, const float pExtFactor, float pMscale, float &pCosTheta,
                                 float &pSinTheta) {
  // Get n-d rotational scaling corrected for extrapolation
  float vThetaInterp = pFreqScale * pThetaExtrap;
  float vTheta = vThetaInterp;
  if (pExtFactor != 0.0f) {
    float vRampMix = fRopeYarnRamp(pCorrDims.v_[0], pCorrDims.v_[1], pI0) * pExtFactor;
    vTheta = vThetaInterp * (1 - vRampMix) + pThetaExtrap * vRampMix;

    // Get n-d magnitude scaling corrected for interpolation
    pMscale *= 1.0f + 0.1f * logf(1.0f / pFreqScale);
  }
  pCosTheta = cosf(vTheta) * pMscale;
  pSinTheta = sinf(vTheta) * pMscale;
  if (!forward) {
    pSinTheta *= -1.0f;
  }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void
fRopeNorm(const T *pX, D *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01, const int pS02,
          const int pS03, const int pS1, const int pS2, const int pS3, const int pNDims, const int32_t *pPos,
          const float pFreqScale, const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
          const float pThetaScale, const float *pFreqFactors, const int64_t *pRowIndices, const int pSetRowsStride) {
  const int cI0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (cI0 >= pNe00) {
    return;
  }

  const int cRowDst = blockDim.x * blockIdx.x + threadIdx.x;

  const uint32_t cI3 = cRowDst / (pNe01 * pNe02);
  const uint32_t cI2 = (cRowDst - cI3 * pNe01 * pNe02) / pNe01;
  const uint32_t cI1 = cRowDst - cI3 * pNe01 * pNe02 - cI2 * pNe01;

  int vIdst = cI0 + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;
  const int cIx = cI0 + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;
  // Fusion optimization: ROPE + VIEW + SET_ROWS.
  // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
  if (pSetRowsStride != 0) {
    vIdst = cI1 * pS1 + cI0;
    vIdst += pRowIndices[cI2] * pSetRowsStride;
  }

  const auto &vStoreCoaelsced = [&](float pX0, float pX1) {
    if constexpr (std::is_same_v<float, D>) {
      float2 v = make_float2(pX0, pX1);
      ggml_cuda_memcpy_1<8>(pDst + vIdst, &v);
    } else if constexpr (std::is_same_v<half, D>) {
      half2 v_ = make_half2(pX0, pX1);
      ggml_cuda_memcpy_1<4>(pDst + vIdst, &v_);
    }
  };
  if (cI0 >= pNDims) {
    vStoreCoaelsced(pX[cIx + 0], pX[cIx + 1]);
    return;
  }

  const float cThetaBase = pPos[cI2] * powf(pThetaScale, cI0 / 2.0f);

  const float cFreqFactor = has_ff ? pFreqFactors[cI0 / 2] : 1.0f;

  float vCosTheta;
  float vSinTheta;

  fRopeYarn<forward>(cThetaBase / cFreqFactor, pFreqScale, pCorrDims, cI0, pExtFactor, pAttnFactor, vCosTheta,
                     vSinTheta);

  const float cX0 = pX[cIx + 0];
  const float cX1 = pX[cIx + 1];

  vStoreCoaelsced(cX0 * vCosTheta - cX1 * vSinTheta, cX0 * vSinTheta + cX1 * vCosTheta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void
fRopeNeox(const T *pX, D *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01, const int pS02,
          const int pS03, const int pS1, const int pS2, const int pS3, const int pNDims, const int32_t *pPos,
          const float pFreqScale, const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
          const float pThetaScale, const float *pFreqFactors, const int64_t *pRowIndices, const int pSetRowsStride) {
  ggml_cuda_pdl_lc();
  const int cI0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (cI0 >= pNe00) {
    return;
  }

  const int cRowDst = blockDim.x * blockIdx.x + threadIdx.x;

  const uint32_t cI3 = cRowDst / (pNe01 * pNe02);
  const uint32_t cI2 = (cRowDst - cI3 * pNe01 * pNe02) / pNe01;
  const uint32_t cI1 = cRowDst - cI3 * pNe01 * pNe02 - cI2 * pNe01;

  int vIdst = cI0 / 2 + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;
  const int cIx = cI0 / 2 + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;
  ggml_cuda_pdl_sync();

  // Fusion optimization: ROPE + VIEW + SET_ROWS.
  // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
  if (pSetRowsStride != 0) {
    vIdst = cI1 * pS1 + cI0 / 2;
    vIdst += pRowIndices[cI2] * pSetRowsStride;
  }

  if (cI0 >= pNDims) {
    pDst[vIdst + cI0 / 2 + 0] = ggml_cuda_cast<D>(pX[cIx + cI0 / 2 + 0]);
    pDst[vIdst + cI0 / 2 + 1] = ggml_cuda_cast<D>(pX[cIx + cI0 / 2 + 1]);

    return;
  }

  const float cThetaBase = pPos[cI2] * powf(pThetaScale, cI0 / 2.0f);

  const float cFreqFactor = has_ff ? pFreqFactors[cI0 / 2] : 1.0f;

  float vCosTheta;
  float vSinTheta;

  fRopeYarn<forward>(cThetaBase / cFreqFactor, pFreqScale, pCorrDims, cI0, pExtFactor, pAttnFactor, vCosTheta,
                     vSinTheta);

  const float cX0 = pX[cIx + 0];
  const float cX1 = pX[cIx + pNDims / 2];

  pDst[vIdst + 0] = ggml_cuda_cast<D>(cX0 * vCosTheta - cX1 * vSinTheta);
  pDst[vIdst + pNDims / 2] = ggml_cuda_cast<D>(cX0 * vSinTheta + cX1 * vCosTheta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void
fRopeMulti(const T *pX, T *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01, const int pS02,
           const int pS03, const int pS1, const int pS2, const int pS3, const int pNDims, const int32_t *pPos,
           const float pFreqScale, const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
           const float pThetaScale, const float *pFreqFactors, const mrope_sections pSections, const bool pIsImrope) {
  const int cI0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (cI0 >= pNe00) {
    return;
  }

  const int cRowDst = blockDim.x * blockIdx.x + threadIdx.x;

  const uint32_t cI3 = cRowDst / (pNe01 * pNe02);
  const uint32_t cI2 = (cRowDst - cI3 * pNe01 * pNe02) / pNe01;
  const uint32_t cI1 = cRowDst - cI3 * pNe01 * pNe02 - cI2 * pNe01;

  int vIdst = cI0 / 2 + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;
  const int cIx = cI0 / 2 + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;

  ggml_cuda_pdl_sync();
  if (cI0 >= pNDims) {
    pDst[vIdst + cI0 / 2 + 0] = pX[cIx + cI0 / 2 + 0];
    pDst[vIdst + cI0 / 2 + 1] = pX[cIx + cI0 / 2 + 1];

    return;
  }

  const int cSectDims = pSections.v_[0] + pSections.v_[1] + pSections.v_[2] + pSections.v_[3];
  const int cSecW = pSections.v_[1] + pSections.v_[0];
  const int cSector = (cI0 / 2) % cSectDims;

  float vThetaBase = 0.0;
  if (pIsImrope) {
    if (cSector % 3 == 1 && cSector < 3 * pSections.v_[1]) { // h
      vThetaBase = pPos[cI2 + pNe02 * 1] * powf(pThetaScale, cI0 / 2.0f);
    } else if (cSector % 3 == 2 && cSector < 3 * pSections.v_[2]) { // w
      vThetaBase = pPos[cI2 + pNe02 * 2] * powf(pThetaScale, cI0 / 2.0f);
    } else if (cSector % 3 == 0 && cSector < 3 * pSections.v_[0]) { // t
      vThetaBase = pPos[cI2] * powf(pThetaScale, cI0 / 2.0f);
    } else {
      vThetaBase = pPos[cI2 + pNe02 * 3] * powf(pThetaScale, cI0 / 2.0f);
    }
  } else {
    if (cSector < pSections.v_[0]) {
      vThetaBase = pPos[cI2] * powf(pThetaScale, cI0 / 2.0f);
    } else if (cSector >= pSections.v_[0] && cSector < cSecW) {
      vThetaBase = pPos[cI2 + pNe02 * 1] * powf(pThetaScale, cI0 / 2.0f);
    } else if (cSector >= cSecW && cSector < cSecW + pSections.v_[2]) {
      vThetaBase = pPos[cI2 + pNe02 * 2] * powf(pThetaScale, cI0 / 2.0f);
    } else if (cSector >= cSecW + pSections.v_[2]) {
      vThetaBase = pPos[cI2 + pNe02 * 3] * powf(pThetaScale, cI0 / 2.0f);
    }
  }

  const float cFreqFactor = has_ff ? pFreqFactors[cI0 / 2] : 1.0f;

  float vCosTheta;
  float vSinTheta;

  fRopeYarn<forward>(vThetaBase / cFreqFactor, pFreqScale, pCorrDims, cI0, pExtFactor, pAttnFactor, vCosTheta,
                     vSinTheta);

  const float cX0 = pX[cIx + 0];
  const float cX1 = pX[cIx + pNDims / 2];

  pDst[vIdst + 0] = cX0 * vCosTheta - cX1 * vSinTheta;
  pDst[vIdst + pNDims / 2] = cX0 * vSinTheta + cX1 * vCosTheta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void fRopeVision(const T *pX, T *pDst, const int pNe00, const int pNe01, const int pNe02,
                                   const int pS01, const int pS02, const int pS03, const int pS1, const int pS2,
                                   const int pS3, const int pNDims, const int32_t *pPos, const float pFreqScale,
                                   const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
                                   const float pThetaScale, const float *pFreqFactors, const mrope_sections pSections) {
  const int cI0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (cI0 >= pNe00) {
    return;
  }

  const int cRowDst = blockDim.x * blockIdx.x + threadIdx.x;

  const uint32_t cI3 = cRowDst / (pNe01 * pNe02);
  const uint32_t cI2 = (cRowDst - cI3 * pNe01 * pNe02) / pNe01;
  const uint32_t cI1 = cRowDst - cI3 * pNe01 * pNe02 - cI2 * pNe01;

  int vIdst = cI0 / 2 + cI1 * pS1 + cI2 * pS2 + cI3 * pS3;
  const int cIx = cI0 / 2 + cI1 * pS01 + cI2 * pS02 + cI3 * pS03;

  ggml_cuda_pdl_sync();
  const int cSectDims = pSections.v_[0] + pSections.v_[1];
  const int cSecW = pSections.v_[1] + pSections.v_[0];
  const int cSector = (cI0 / 2) % cSectDims;

  float vThetaBase = 0.0;
  if (cSector < pSections.v_[0]) {
    const int cP = cSector;
    vThetaBase = pPos[cI2] * powf(pThetaScale, cP);
  } else if (cSector >= pSections.v_[0] && cSector < cSecW) {
    const int cP = cSector - pSections.v_[0];
    vThetaBase = pPos[cI2 + pNe02] * powf(pThetaScale, cP);
  }

  const float cFreqFactor = has_ff ? pFreqFactors[cI0 / 2] : 1.0f;

  float vCosTheta;
  float vSinTheta;

  fRopeYarn<forward>(vThetaBase / cFreqFactor, pFreqScale, pCorrDims, cI0, pExtFactor, pAttnFactor, vCosTheta,
                     vSinTheta);

  const float cX0 = pX[cIx + 0];
  const float cX1 = pX[cIx + pNDims];

  pDst[vIdst + 0] = cX0 * vCosTheta - cX1 * vSinTheta;
  pDst[vIdst + pNDims] = cX0 * vSinTheta + cX1 * vCosTheta;
}

template <bool forward, typename T, typename D>
static void fRopeNormCuda(const T *pX, D *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01,
                          const int pS02, const int pS03, const int pS1, const int pS2, const int pS3, const int pNDims,
                          const int pNr, const int32_t *pPos, const float pFreqScale, const float pFreqBase,
                          const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
                          const float *pFreqFactors, const int64_t *pRowIndices, const int pSetRowsStride,
                          cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 2 == 0);
  const dim3 cBlockDims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int cNBlocksX = (pNe00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 cBlockNums(pNr, cNBlocksX, 1);

  const float cThetaScale = powf(pFreqBase, -2.0f / pNDims);

  if (pFreqFactors == nullptr) {
    fRopeNorm<forward, false><<<cBlockNums, cBlockDims, 0, pStream>>>(
        pX, pDst, pNe00, pNe01, pNe02, pS01, pS02, pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor,
        pAttnFactor, pCorrDims, cThetaScale, pFreqFactors, pRowIndices, pSetRowsStride);
  } else {
    fRopeNorm<forward, true><<<cBlockNums, cBlockDims, 0, pStream>>>(
        pX, pDst, pNe00, pNe01, pNe02, pS01, pS02, pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor,
        pAttnFactor, pCorrDims, cThetaScale, pFreqFactors, pRowIndices, pSetRowsStride);
  }
}

template <bool forward, typename T, typename D>
static void fRopeNeoxCuda(const T *pX, D *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01,
                          const int pS02, const int pS03, const int pS1, const int pS2, const int pS3, const int pNDims,
                          const int pNr, const int32_t *pPos, const float pFreqScale, const float pFreqBase,
                          const float pExtFactor, const float pAttnFactor, const rope_corr_dims pCorrDims,
                          const float *pFreqFactors, const int64_t *pRowIndices, const int pSetRowsStride,
                          cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 2 == 0);
  const dim3 cBlockDims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int cNBlocksX = (pNe00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 cBlockNums(pNr, cNBlocksX, 1);

  const float cThetaScale = powf(pFreqBase, -2.0f / pNDims);
  const ggml_cuda_kernel_launch_params cLaunchParams = {cBlockNums, cBlockDims, 0, pStream};

  if (pFreqFactors == nullptr) {
    ggml_cuda_kernel_launch(fRopeNeox<forward, false, T, D>, cLaunchParams, pX, pDst, pNe00, pNe01, pNe02, pS01, pS02,
                            pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor, pAttnFactor, pCorrDims,
                            cThetaScale, pFreqFactors, pRowIndices, pSetRowsStride);
  } else {
    ggml_cuda_kernel_launch(fRopeNeox<forward, true, T, D>, cLaunchParams, pX, pDst, pNe00, pNe01, pNe02, pS01, pS02,
                            pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor, pAttnFactor, pCorrDims,
                            cThetaScale, pFreqFactors, pRowIndices, pSetRowsStride);
  }
}

template <bool forward, typename T>
static void fRopeMultiCuda(const T *pX, T *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01,
                           const int pS02, const int pS03, const int pS1, const int pS2, const int pS3,
                           const int pNDims, const int pNr, const int32_t *pPos, const float pFreqScale,
                           const float pFreqBase, const float pExtFactor, const float pAttnFactor,
                           const rope_corr_dims pCorrDims, const float *pFreqFactors, const mrope_sections pSections,
                           const bool pIsImrope, cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 2 == 0);
  const dim3 cBlockDims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int cNBlocksX = (pNe00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 cBlockNums(pNr, cNBlocksX, 1);

  const float cThetaScale = powf(pFreqBase, -2.0f / pNDims);

  if (pFreqFactors == nullptr) {
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, pStream);
    ggml_cuda_kernel_launch(fRopeMulti<forward, false, T>, cLaunchParams, pX, pDst, pNe00, pNe01, pNe02, pS01, pS02,
                            pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor, pAttnFactor, pCorrDims,
                            cThetaScale, pFreqFactors, pSections, pIsImrope);
  } else {
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cBlockNums, cBlockDims, 0, pStream);
    ggml_cuda_kernel_launch(fRopeMulti<forward, true, T>, cLaunchParams, pX, pDst, pNe00, pNe01, pNe02, pS01, pS02,
                            pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor, pAttnFactor, pCorrDims,
                            cThetaScale, pFreqFactors, pSections, pIsImrope);
  }
}

template <bool forward, typename T>
static void fRopeVisionCuda(const T *pX, T *pDst, const int pNe00, const int pNe01, const int pNe02, const int pS01,
                            const int pS02, const int pS03, const int pS1, const int pS2, const int pS3,
                            const int pNDims, const int pNr, const int32_t *pPos, const float pFreqScale,
                            const float pFreqBase, const float pExtFactor, const float pAttnFactor,
                            const rope_corr_dims pCorrDims, const float *pFreqFactors, const mrope_sections pSections,
                            cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % 2 == 0);
  const dim3 cBlockDims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int cNBlocksX = (pNe00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 cBlockNums(pNr, cNBlocksX, 1);
  // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
  // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

  const float cThetaScale = powf(pFreqBase, -2.0f / pNDims);

  if (pFreqFactors == nullptr) {
    fRopeVision<forward, false, T><<<cBlockNums, cBlockDims, 0, pStream>>>(
        pX, pDst, pNe00, pNe01, pNe02, pS01, pS02, pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor,
        pAttnFactor, pCorrDims, cThetaScale, pFreqFactors, pSections);
  } else {
    fRopeVision<forward, true, T><<<cBlockNums, cBlockDims, 0, pStream>>>(
        pX, pDst, pNe00, pNe01, pNe02, pS01, pS02, pS03, pS1, pS2, pS3, pNDims, pPos, pFreqScale, pExtFactor,
        pAttnFactor, pCorrDims, cThetaScale, pFreqFactors, pSections);
  }
}

template <bool forward>
void fGgmlCudaOpRopeImpl(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, const ggml_tensor *pSetRows = nullptr) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  const ggml_tensor *vSrc2 = pDst->src[2];

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;

  void *vDstD = pDst->data;
  const int64_t *vRowIndices = nullptr;
  ggml_type vDstType = pDst->type;
  int vSetRowsStride = 0;

  if (pSetRows != nullptr) {
    GGML_ASSERT(forward);
    vDstD = pSetRows->data;
    vRowIndices = (const int64_t *)pSetRows->src[1]->data;
    vDstType = pSetRows->type;
    vSetRowsStride = pSetRows->nb[1] / ggml_type_size(pSetRows->type);
  }
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  // When not fused, src0 and dst types must match
  // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
  GGML_ASSERT(vSrc0->type == pDst->type || (vSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F16));

  const int64_t cNe00 = vSrc0->ne[0]; // head dims
  const int64_t cNe01 = vSrc0->ne[1]; // num heads
  const int64_t cNe02 = vSrc0->ne[2]; // num heads
  const int64_t cNr = ggml_nrows(vSrc0);

  const size_t cS01 = vSrc0->nb[1] / ggml_type_size(vSrc0->type);
  const size_t cS02 = vSrc0->nb[2] / ggml_type_size(vSrc0->type);
  const size_t cS03 = vSrc0->nb[3] / ggml_type_size(vSrc0->type);

  const size_t cS1 = pDst->nb[1] / ggml_type_size(pDst->type);
  const size_t cS2 = pDst->nb[2] / ggml_type_size(pDst->type);
  const size_t cS3 = pDst->nb[3] / ggml_type_size(pDst->type);

  // const int n_past     = ((int32_t *) dst->op_params)[0];
  const int cNDims = ((int32_t *)pDst->op_params)[1];
  const int cMode = ((int32_t *)pDst->op_params)[2];
  // const int n_ctx      = ((int32_t *) dst->op_params)[3];
  const int cNCtxOrig = ((int32_t *)pDst->op_params)[4];
  mrope_sections vSections;

  // RoPE alteration for extended context
  float vFreqBase;
  float vFreqScale;
  float vExtFactor;
  float vAttnFactor;
  float vBetaFast;
  float vBetaSlow;

  memcpy(&vFreqBase, (int32_t *)pDst->op_params + 5, sizeof(float));
  memcpy(&vFreqScale, (int32_t *)pDst->op_params + 6, sizeof(float));
  memcpy(&vExtFactor, (int32_t *)pDst->op_params + 7, sizeof(float));
  memcpy(&vAttnFactor, (int32_t *)pDst->op_params + 8, sizeof(float));
  memcpy(&vBetaFast, (int32_t *)pDst->op_params + 9, sizeof(float));
  memcpy(&vBetaSlow, (int32_t *)pDst->op_params + 10, sizeof(float));
  memcpy(&vSections.v_, (int32_t *)pDst->op_params + 11, sizeof(int) * 4);

  const bool cIsNeox = cMode & GGML_ROPE_TYPE_NEOX;
  const bool cIsMrope = cMode & GGML_ROPE_TYPE_MROPE;
  const bool cIsImrope = cMode == GGML_ROPE_TYPE_IMROPE;
  const bool cIsVision = cMode == GGML_ROPE_TYPE_VISION;

  if (cIsMrope) {
    GGML_ASSERT(vSections.v_[0] > 0 || vSections.v_[1] > 0 || vSections.v_[2] > 0);
  }

  if (cIsVision) {
    GGML_ASSERT(cNDims == cNe00 / 2);
  }

  const int32_t *vPos = (const int32_t *)vSrc1D;

  const float *vFreqFactors = nullptr;
  if (vSrc2 != nullptr) {
    vFreqFactors = (const float *)vSrc2->data;
  }

  rope_corr_dims vCorrDims;
  ggml_rope_yarn_corr_dims(cNDims, cNCtxOrig, vFreqBase, vBetaFast, vBetaSlow, vCorrDims.v_);

  // compute
  if (cIsNeox) {
    if (vSrc0->type == GGML_TYPE_F32 && vDstType == GGML_TYPE_F32) {
      fRopeNeoxCuda<forward, float, float>((const float *)vSrc0D, (float *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                           cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                           vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else if (vSrc0->type == GGML_TYPE_F32 && vDstType == GGML_TYPE_F16) {
      fRopeNeoxCuda<forward, float, half>((const float *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                          cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                          vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else if (vSrc0->type == GGML_TYPE_F16 && vDstType == GGML_TYPE_F16) {
      fRopeNeoxCuda<forward, half, half>((const half *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                         cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                         vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else {
      GGML_ABORT("fatal error");
    }
  } else if (cIsMrope && !cIsVision) {
    if (vSrc0->type == GGML_TYPE_F32) {
      fRopeMultiCuda<forward>((const float *)vSrc0D, (float *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03, cS1, cS2,
                              cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor, vAttnFactor, vCorrDims,
                              vFreqFactors, vSections, cIsImrope, vStream);
    } else if (vSrc0->type == GGML_TYPE_F16) {
      fRopeMultiCuda<forward>((const half *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03, cS1, cS2, cS3,
                              cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor, vAttnFactor, vCorrDims,
                              vFreqFactors, vSections, cIsImrope, vStream);
    } else {
      GGML_ABORT("fatal error");
    }
  } else if (cIsVision) {
    if (vSrc0->type == GGML_TYPE_F32) {
      fRopeVisionCuda<forward>((const float *)vSrc0D, (float *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03, cS1, cS2,
                               cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor, vAttnFactor, vCorrDims,
                               vFreqFactors, vSections, vStream);
    } else if (vSrc0->type == GGML_TYPE_F16) {
      fRopeVisionCuda<forward>((const half *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03, cS1, cS2,
                               cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor, vAttnFactor, vCorrDims,
                               vFreqFactors, vSections, vStream);
    } else {
      GGML_ABORT("fatal error");
    }
  } else {
    if (vSrc0->type == GGML_TYPE_F32 && vDstType == GGML_TYPE_F32) {
      fRopeNormCuda<forward, float, float>((const float *)vSrc0D, (float *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                           cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                           vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else if (vSrc0->type == GGML_TYPE_F32 && vDstType == GGML_TYPE_F16) {
      fRopeNormCuda<forward, float, half>((const float *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                          cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                          vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else if (vSrc0->type == GGML_TYPE_F16 && vDstType == GGML_TYPE_F16) {
      fRopeNormCuda<forward, half, half>((const half *)vSrc0D, (half *)vDstD, cNe00, cNe01, cNe02, cS01, cS02, cS03,
                                         cS1, cS2, cS3, cNDims, cNr, vPos, vFreqScale, vFreqBase, vExtFactor,
                                         vAttnFactor, vCorrDims, vFreqFactors, vRowIndices, vSetRowsStride, vStream);
    } else {
      GGML_ABORT("fatal error");
    }
  }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpRopeImpl<true>(pCtx, pDst); }

void ggml_cuda_op_rope_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpRopeImpl<false>(pCtx, pDst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context &pCtx, ggml_tensor *pRope, ggml_tensor *pSetRows) {
  fGgmlCudaOpRopeImpl<true>(pCtx, pRope, pSetRows);
}
