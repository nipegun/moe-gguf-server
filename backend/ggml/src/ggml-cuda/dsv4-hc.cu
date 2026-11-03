#include "common.cuh"
#include "dsv4-hc.cuh"

static constexpr int DSV4_HC = 4;

static __device__ void fDsv4HcCombNormCols(float *pComb, float pEps) {
  for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
    float vSum = pEps;
    for (int vIsrc = 0; vIsrc < DSV4_HC; ++vIsrc) {
      vSum += pComb[vIdst + DSV4_HC * vIsrc];
    }

    const float cInvSum = 1.0f / vSum;
    for (int vIsrc = 0; vIsrc < DSV4_HC; ++vIsrc) {
      pComb[vIdst + DSV4_HC * vIsrc] *= cInvSum;
    }
  }
}

static __device__ void fDsv4HcCombNormRows(float *pComb, float pEps) {
  for (int vIsrc = 0; vIsrc < DSV4_HC; ++vIsrc) {
    float vSum = pEps;
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      vSum += pComb[vIdst + DSV4_HC * vIsrc];
    }

    const float cInvSum = 1.0f / vSum;
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      pComb[vIdst + DSV4_HC * vIsrc] *= cInvSum;
    }
  }
}

static __global__ void fDsv4HcCombF32(const float *pMixes, const float *pScale, const float *pBase, float *pDst,
                                      int64_t pNTokens, int64_t pSm0, int64_t pSm1, int64_t pSs0, int64_t pSb0,
                                      int64_t pSd0, int64_t pSd1, int64_t pSd2, float pEps, int32_t pNIter) {
  constexpr int cCombOffset = 2 * DSV4_HC;

  ggml_cuda_pdl_lc();
  const int64_t cIt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (cIt >= pNTokens) {
    return;
  }

  ggml_cuda_pdl_sync();

  const float cScaleComb = pScale[2 * pSs0];
  float vComb[DSV4_HC * DSV4_HC];

  for (int vIsrc = 0; vIsrc < DSV4_HC; ++vIsrc) {
    float vMax = -INFINITY;
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      const int cIdx = vIdst + DSV4_HC * vIsrc;
      const float cV =
          pMixes[(cCombOffset + cIdx) * pSm0 + cIt * pSm1] * cScaleComb + pBase[(cCombOffset + cIdx) * pSb0];
      vComb[cIdx] = cV;
      vMax = fmaxf(vMax, cV);
    }

    float vSum = 0.0f;
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      const int cIdx = vIdst + DSV4_HC * vIsrc;
      const float cV = expf(vComb[cIdx] - vMax);
      vComb[cIdx] = cV;
      vSum += cV;
    }

    const float cInvSum = 1.0f / vSum;
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      const int cIdx = vIdst + DSV4_HC * vIsrc;
      vComb[cIdx] = vComb[cIdx] * cInvSum + pEps;
    }
  }

  fDsv4HcCombNormCols(vComb, pEps);
  for (int32_t vI = 1; vI < pNIter; ++vI) {
    fDsv4HcCombNormRows(vComb, pEps);
    fDsv4HcCombNormCols(vComb, pEps);
  }

  for (int vIsrc = 0; vIsrc < DSV4_HC; ++vIsrc) {
    for (int vIdst = 0; vIdst < DSV4_HC; ++vIdst) {
      const int cIdx = vIdst + DSV4_HC * vIsrc;
      pDst[vIdst * pSd0 + vIsrc * pSd1 + cIt * pSd2] = vComb[cIdx];
    }
  }
}

template <bool gated>
static __global__ void fDsv4HcPreF32(const float *pX, const float *pWeights, float *pDst, int64_t pNEmbd, int64_t pHc,
                                     int64_t pNTokens, int64_t pSx0, int64_t pSx1, int64_t pSx2, int64_t pSw0,
                                     int64_t pSw1, int64_t pSw2, int64_t pSd0, int64_t pSd1, float pScale) {
  ggml_cuda_pdl_lc();
  const int64_t cIr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t cNr = pNEmbd * pNTokens;

  if (cIr >= cNr) {
    return;
  }

  ggml_cuda_pdl_sync();

  const int64_t cI0 = cIr % pNEmbd;
  const int64_t cIt = cIr / pNEmbd;

  float vSum = 0.0f;
  for (int64_t vIh = 0; vIh < pHc; ++vIh) {
    const float cXv = pX[cI0 * pSx0 + vIh * pSx1 + cIt * pSx2];
    float vWv;
    if constexpr (gated) {
      vWv = 1.0f / (1.0f + expf(-pWeights[cI0 * pSw0 + vIh * pSw1 + cIt * pSw2]));
    } else {
      vWv = pWeights[vIh * pSw0 + cIt * pSw1];
    }
    vSum += cXv * vWv;
  }

  pDst[cI0 * pSd0 + cIt * pSd1] = pScale * vSum;
}

template <bool has_comb>
static __global__ void fDsv4HcPostF32(const float *pX, const float *pResidual, const float *pPost, const float *pComb,
                                      float *pDst, int64_t pNEmbd, int64_t pHc, int64_t pNTokens, int64_t pSx0,
                                      int64_t pSx1, int64_t pSr0, int64_t pSr1, int64_t pSr2, int64_t pSp0,
                                      int64_t pSp1, int64_t pSc0, int64_t pSc1, int64_t pSc2, int64_t pSd0,
                                      int64_t pSd1, int64_t pSd2) {
  ggml_cuda_pdl_lc();
  const int64_t cIr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t cNr = pNEmbd * pHc * pNTokens;

  if (cIr >= cNr) {
    return;
  }

  ggml_cuda_pdl_sync();

  const int64_t cI0 = cIr % pNEmbd;
  const int64_t cIdst = (cIr / pNEmbd) % pHc;
  const int64_t cIt = cIr / (pNEmbd * pHc);

  float vSum = pX[cI0 * pSx0 + cIt * pSx1] * pPost[cIdst * pSp0 + cIt * pSp1];
  if constexpr (has_comb) {
    for (int64_t isrc = 0; isrc < pHc; ++isrc) {
      vSum += pResidual[cI0 * pSr0 + isrc * pSr1 + cIt * pSr2] * pComb[cIdst * pSc0 + isrc * pSc1 + cIt * pSc2];
    }
  } else {
    vSum += pResidual[cI0 * pSr0 + cIdst * pSr1 + cIt * pSr2];
  }

  pDst[cI0 * pSd0 + cIdst * pSd1 + cIt * pSd2] = vSum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vMixes = pDst->src[0];
  const ggml_tensor *vScale = pDst->src[1];
  const ggml_tensor *vBase = pDst->src[2];

  GGML_ASSERT(vMixes->type == GGML_TYPE_F32);
  GGML_ASSERT(vScale->type == GGML_TYPE_F32);
  GGML_ASSERT(vBase->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  constexpr int64_t cHcMixDim = (2 + DSV4_HC) * DSV4_HC;

  GGML_ASSERT(vMixes->ne[0] == cHcMixDim);
  GGML_ASSERT(pDst->ne[0] == DSV4_HC);
  GGML_ASSERT(pDst->ne[1] == DSV4_HC);
  GGML_ASSERT(pDst->ne[2] == vMixes->ne[1]);
  GGML_ASSERT(vScale->ne[0] >= 3);
  GGML_ASSERT(vBase->ne[0] == cHcMixDim);

  GGML_TENSOR_LOCALS(size_t, nbm, vMixes, nb);
  GGML_TENSOR_LOCALS(size_t, nbs, vScale, nb);
  GGML_TENSOR_LOCALS(size_t, nbb, vBase, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const int64_t cNTokens = vMixes->ne[1];
  const float cEps = ggml_get_op_params_f32(pDst, 0);
  const int32_t cNIter = ggml_get_op_params_i32(pDst, 1);

  const int cBlockSize = 256;
  const dim3 cBlockDims(cBlockSize, 1, 1);
  const dim3 cGridDims((cNTokens + cBlockSize - 1) / cBlockSize, 1, 1);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cGridDims, cBlockDims, 0, pCtx.stream());

  ggml_cuda_kernel_launch(fDsv4HcCombF32, cLaunchParams, (const float *)vMixes->data, (const float *)vScale->data,
                          (const float *)vBase->data, (float *)pDst->data, cNTokens, nbm0 / sizeof(float),
                          nbm1 / sizeof(float), nbs0 / sizeof(float), nbb0 / sizeof(float), nbd0 / sizeof(float),
                          nbd1 / sizeof(float), nbd2 / sizeof(float), cEps, cNIter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vX = pDst->src[0];
  const ggml_tensor *vWeights = pDst->src[1];

  GGML_ASSERT(vX->type == GGML_TYPE_F32);
  GGML_ASSERT(vWeights->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_TENSOR_LOCALS(size_t, nbx, vX, nb);
  GGML_TENSOR_LOCALS(size_t, nbw, vWeights, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const int64_t cNEmbd = vX->ne[0];
  const int64_t cHc = vX->ne[1];
  const int64_t cNTokens = vX->ne[2];

  const float cScale = ggml_get_op_params_f32(pDst, 0);
  const bool cGated = ggml_get_op_params_i32(pDst, 1) != 0;

  const int cBlockSize = 256;
  const int64_t cNr = cNEmbd * cNTokens;
  const dim3 cBlockDims(cBlockSize, 1, 1);
  const dim3 cGridDims((cNr + cBlockSize - 1) / cBlockSize, 1, 1);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cGridDims, cBlockDims, 0, pCtx.stream());

  auto vKernel = cGated ? fDsv4HcPreF32<true> : fDsv4HcPreF32<false>;
  ggml_cuda_kernel_launch(vKernel, cLaunchParams, (const float *)vX->data, (const float *)vWeights->data,
                          (float *)pDst->data, cNEmbd, cHc, cNTokens, nbx0 / sizeof(float), nbx1 / sizeof(float),
                          nbx2 / sizeof(float), nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                          nbd0 / sizeof(float), nbd1 / sizeof(float), cScale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vX = pDst->src[0];
  const ggml_tensor *vResidual = pDst->src[1];
  const ggml_tensor *vPost = pDst->src[2];
  const ggml_tensor *vComb = pDst->src[3];

  GGML_ASSERT(vX->type == GGML_TYPE_F32);
  GGML_ASSERT(vResidual->type == GGML_TYPE_F32);
  GGML_ASSERT(vPost->type == GGML_TYPE_F32);
  GGML_ASSERT(vComb == nullptr || vComb->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  GGML_TENSOR_LOCALS(size_t, nbx, vX, nb);
  GGML_TENSOR_LOCALS(size_t, nbr, vResidual, nb);
  GGML_TENSOR_LOCALS(size_t, nbp, vPost, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const size_t cNbc0 = vComb ? vComb->nb[0] : 0;
  const size_t cNbc1 = vComb ? vComb->nb[1] : 0;
  const size_t cNbc2 = vComb ? vComb->nb[2] : 0;

  const int64_t cNEmbd = vX->ne[0];
  const int64_t cNTokens = vX->ne[1];
  const int64_t cHc = vResidual->ne[1];

  const int cBlockSize = 256;
  const int64_t cNr = cNEmbd * cHc * cNTokens;
  const dim3 cBlockDims(cBlockSize, 1, 1);
  const dim3 cGridDims((cNr + cBlockSize - 1) / cBlockSize, 1, 1);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params(cGridDims, cBlockDims, 0, pCtx.stream());

  auto vKernel = vComb ? fDsv4HcPostF32<true> : fDsv4HcPostF32<false>;
  ggml_cuda_kernel_launch(vKernel, cLaunchParams, (const float *)vX->data, (const float *)vResidual->data,
                          (const float *)vPost->data, vComb ? (const float *)vComb->data : nullptr, (float *)pDst->data,
                          cNEmbd, cHc, cNTokens, nbx0 / sizeof(float), nbx1 / sizeof(float), nbr0 / sizeof(float),
                          nbr1 / sizeof(float), nbr2 / sizeof(float), nbp0 / sizeof(float), nbp1 / sizeof(float),
                          cNbc0 / sizeof(float), cNbc1 / sizeof(float), cNbc2 / sizeof(float), nbd0 / sizeof(float),
                          nbd1 / sizeof(float), nbd2 / sizeof(float));
}
