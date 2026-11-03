#include "upscale.cuh"

static __global__ void fUpscaleF32(const float *pX, float *pDst, const int pNb00, const int pNb01, const int pNb02,
                                   const int pNb03, const int pNe10, const int pNe11, const int pNe12, const int pNe13,
                                   const float pSf0, const float pSf1, const float pSf2, const float pSf3) {
  int vIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIndex >= pNe10 * pNe11 * pNe12 * pNe13) {
    return;
  }

  int vI10 = vIndex % pNe10;
  int vI11 = (vIndex / pNe10) % pNe11;
  int vI12 = (vIndex / (pNe10 * pNe11)) % pNe12;
  int vI13 = (vIndex / (pNe10 * pNe11 * pNe12)) % pNe13;

  int vI00 = vI10 / pSf0;
  int vI01 = vI11 / pSf1;
  int vI02 = vI12 / pSf2;
  int vI03 = vI13 / pSf3;

  pDst[vIndex] = *((const float *)((const char *)pX + vI03 * pNb03 + vI02 * pNb02 + vI01 * pNb01 + vI00 * pNb00));
}

static __global__ void fUpscaleF32Bilinear(const float *pX, float *pDst, const int pNb00, const int pNb01,
                                           const int pNb02, const int pNb03, const int pNe00Src, const int pNe01Src,
                                           const int pNe10Dst, const int pNe11Dst, const int pNe12Dst,
                                           const int pNe13Dst, const float pSf0, const float pSf1, const float pSf2,
                                           const float pSf3, const float pPixelOffset) {
  const int64_t cIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t cDstTotalElements = pNe10Dst * pNe11Dst * pNe12Dst * pNe13Dst;

  if (cIndex >= cDstTotalElements) {
    return;
  }

  const int cI10Dst = cIndex % pNe10Dst;
  const int cI11Dst = (cIndex / pNe10Dst) % pNe11Dst;
  const int cI12Dst = (cIndex / (pNe10Dst * pNe11Dst)) % pNe12Dst;
  const int cI13Dst = cIndex / (pNe10Dst * pNe11Dst * pNe12Dst);

  const int cI02Src = (int)(cI12Dst / pSf2);
  const int cI03Src = (int)(cI13Dst / pSf3);

  const float cYSrcF = ((float)cI11Dst + pPixelOffset) / pSf1 - pPixelOffset;
  int vY0Src = (int)floorf(cYSrcF);
  int vY1Src = vY0Src + 1;

  vY0Src = max(0, min(vY0Src, pNe01Src - 1));
  vY1Src = max(0, min(vY1Src, pNe01Src - 1));

  float vDy = cYSrcF - (float)vY0Src;
  vDy = max(0.0f, min(vDy, 1.0f));

  float vXSrcF = ((float)cI10Dst + pPixelOffset) / pSf0 - pPixelOffset;
  int vX0Src = (int)floorf(vXSrcF);
  int vX1Src = vX0Src + 1;

  vX0Src = max(0, min(vX0Src, pNe00Src - 1));
  vX1Src = max(0, min(vX1Src, pNe00Src - 1));

  float vDx = vXSrcF - (float)vX0Src;
  vDx = max(0.0f, min(vDx, 1.0f));

  const float *vPA = (const float *)((const char *)pX + (int64_t)vX0Src * pNb00 + (int64_t)vY0Src * pNb01 +
                                     (int64_t)cI02Src * pNb02 + (int64_t)cI03Src * pNb03);
  const float *vPB = (const float *)((const char *)pX + (int64_t)vX1Src * pNb00 + (int64_t)vY0Src * pNb01 +
                                     (int64_t)cI02Src * pNb02 + (int64_t)cI03Src * pNb03);
  const float *vPC = (const float *)((const char *)pX + (int64_t)vX0Src * pNb00 + (int64_t)vY1Src * pNb01 +
                                     (int64_t)cI02Src * pNb02 + (int64_t)cI03Src * pNb03);
  const float *vPD = (const float *)((const char *)pX + (int64_t)vX1Src * pNb00 + (int64_t)vY1Src * pNb01 +
                                     (int64_t)cI02Src * pNb02 + (int64_t)cI03Src * pNb03);

  const float cValA = *vPA;
  const float cValB = *vPB;
  const float cValC = *vPC;
  const float cValD = *vPD;

  float vResult =
      cValA * (1.0f - vDx) * (1.0f - vDy) + cValB * vDx * (1.0f - vDy) + cValC * (1.0f - vDx) * vDy + cValD * vDx * vDy;

  pDst[cIndex] = vResult;
}

// Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
static __global__ void fUpscaleF32BilinearAntialias(const float *pSrc0, float *pDst, const int pNb00, const int pNb01,
                                                    const int pNb02, const int pNb03, const int pNe00Src,
                                                    const int pNe01Src, const int pNe10Dst, const int pNe11Dst,
                                                    const int pNe12Dst, const int pNe13Dst, const float pSf0,
                                                    const float pSf1, const float pSf2, const float pSf3,
                                                    const float pPixelOffset) {
  const int64_t cIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t cDstTotalElements = pNe10Dst * pNe11Dst * pNe12Dst * pNe13Dst;

  if (cIndex >= cDstTotalElements) {
    return;
  }

  const int cI10Dst = cIndex % pNe10Dst;
  const int cI11Dst = (cIndex / pNe10Dst) % pNe11Dst;
  const int cI12Dst = (cIndex / (pNe10Dst * pNe11Dst)) % pNe12Dst;
  const int cI13Dst = cIndex / (pNe10Dst * pNe11Dst * pNe12Dst);

  const int cI02Src = (int)(cI12Dst / pSf2);
  const int cI03Src = (int)(cI13Dst / pSf3);

  const float cY = ((float)cI11Dst + pPixelOffset) / pSf1;
  const float cX = ((float)cI10Dst + pPixelOffset) / pSf0;

  // support and invscale, minimum 1 pixel for bilinear
  const float cSupport1 = max(1.0f / pSf1, 1.0f);
  const float cInvscale1 = 1.0f / cSupport1;
  const float cSupport0 = max(1.0f / pSf0, 1.0f);
  const float cInvscale0 = 1.0f / cSupport0;

  // the range of source pixels that contribute
  const int64_t cXMin = max(int64_t(0), int64_t(cX - cSupport0 + pPixelOffset));
  const int64_t cXMax = min(int64_t(pNe00Src), int64_t(cX + cSupport0 + pPixelOffset));
  const int64_t cYMin = max(int64_t(0), int64_t(cY - cSupport1 + pPixelOffset));
  const int64_t cYMax = min(int64_t(pNe01Src), int64_t(cY + cSupport1 + pPixelOffset));

  // bilinear filter with antialiasing
  float vVal = 0.0f;
  float vTotalWeight = 0.0f;

  auto fTriangleFilter = [](float pX) -> float { return max(1.0f - fabsf(pX), 0.0f); };

  for (int64_t vSy = cYMin; vSy < cYMax; vSy++) {
    const float cWeightY = fTriangleFilter((vSy - cY + pPixelOffset) * cInvscale1);

    for (int64_t vSx = cXMin; vSx < cXMax; vSx++) {
      const float cWeightX = fTriangleFilter((vSx - cX + pPixelOffset) * cInvscale0);
      const float cWeight = cWeightX * cWeightY;

      if (cWeight <= 0.0f) {
        continue;
      }

      const float cPixel =
          *(const float *)((const char *)pSrc0 + vSx * pNb00 + vSy * pNb01 + cI02Src * pNb02 + cI03Src * pNb03);
      vVal += cPixel * cWeight;
      vTotalWeight += cWeight;
    }
  }

  if (vTotalWeight > 0.0f) {
    vVal /= vTotalWeight;
  }

  pDst[cIndex] = vVal;
}

namespace bicubic_interpolation {
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
__device__ const float a = -0.75f; // use alpha = -0.75 (same as PyTorch)

static __device__ float fWeight1(float pX) { return ((a + 2) * pX - (a + 3)) * pX * pX + 1; };
static __device__ float fWeight2(float pX) { return ((a * pX - 5 * a) * pX + 8 * a) * pX - 4 * a; };

static __device__ float fBicubic(float p0, float p1, float p2, float p3, float pX) {
  const float cW0 = fWeight2(pX + 1);
  const float cW1 = fWeight1(pX + 0);
  const float cW2 = fWeight1(1 - pX);
  const float cW3 = fWeight2(2 - pX);
  return p0 * cW0 + p1 * cW1 + p2 * cW2 + p3 * cW3;
};
} // namespace bicubic_interpolation

static __global__ void fUpscaleF32Bicubic(const float *pX, float *pDst, const int pNb00, const int pNb01,
                                          const int pNb02, const int pNb03, const int pNe00Src, const int pNe01Src,
                                          const int pNe10Dst, const int pNe11Dst, const int pNe12Dst,
                                          const int pNe13Dst, const float pSf0, const float pSf1, const float pSf2,
                                          const float pSf3, const float pPixelOffset) {
  using bicubic_interpolation::fBicubic;

  const int64_t cIndex = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t cDstTotalElements = pNe10Dst * pNe11Dst * pNe12Dst * pNe13Dst;

  if (cIndex >= cDstTotalElements) {
    return;
  }

  const int cI10Dst = cIndex % pNe10Dst;
  const int cI11Dst = (cIndex / pNe10Dst) % pNe11Dst;
  const int cI12Dst = (cIndex / (pNe10Dst * pNe11Dst)) % pNe12Dst;
  const int cI13Dst = cIndex / (pNe10Dst * pNe11Dst * pNe12Dst);

  const int cI02Src = (int)(cI12Dst / pSf2);
  const int cI03Src = (int)(cI13Dst / pSf3);

  const float cYSrcF = ((float)cI11Dst + pPixelOffset) / pSf1 - pPixelOffset;
  const int cY0Src = (int)floorf(cYSrcF);
  const float cDy = cYSrcF - (float)cY0Src;

  const float cXSrcF = ((float)cI10Dst + pPixelOffset) / pSf0 - pPixelOffset;
  const int cX0Src = (int)floorf(cXSrcF);
  const float cDx = cXSrcF - (float)cX0Src;

  const char *vXBase = (const char *)pX + (int64_t)cI02Src * pNb02 + (int64_t)cI03Src * pNb03;

  auto fLoad = [=](int pXOff, int pYOff) -> float {
    int vI00Src = max(0, min(cX0Src + pXOff, pNe00Src - 1));
    int vI01Src = max(0, min(cY0Src + pYOff, pNe01Src - 1));
    return *(const float *)(vXBase + (int64_t)vI00Src * pNb00 + (int64_t)vI01Src * pNb01);
  };

  const float cResult = fBicubic(fBicubic(fLoad(-1, -1), fLoad(0, -1), fLoad(1, -1), fLoad(2, -1), cDx),
                                 fBicubic(fLoad(-1, 0), fLoad(0, 0), fLoad(1, 0), fLoad(2, 0), cDx),
                                 fBicubic(fLoad(-1, 1), fLoad(0, 1), fLoad(1, 1), fLoad(2, 1), cDx),
                                 fBicubic(fLoad(-1, 2), fLoad(0, 2), fLoad(1, 2), fLoad(2, 2), cDx), cDy);

  pDst[cIndex] = cResult;
}

static void fUpscaleF32Cuda(const float *pX, float *pDst, const int pNb00, const int pNb01, const int pNb02,
                            const int pNb03, const int pNe10, const int pNe11, const int pNe12, const int pNe13,
                            const float pSf0, const float pSf1, const float pSf2, const float pSf3,
                            cudaStream_t pStream) {
  const int64_t cDstSize = pNe10 * pNe11 * pNe12 * pNe13;
  const int64_t cNumBlocks = (cDstSize + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

  fUpscaleF32<<<cNumBlocks, CUDA_UPSCALE_BLOCK_SIZE, 0, pStream>>>(pX, pDst, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                                   pNe12, pNe13, pSf0, pSf1, pSf2, pSf3);
}

static void fUpscaleF32BilinearCuda(const float *pX, float *pDst, const int pNb00, const int pNb01, const int pNb02,
                                    const int pNb03, const int pNe00Src, const int pNe01Src, const int pNe10Dst,
                                    const int pNe11Dst, const int pNe12Dst, const int pNe13Dst, const float pSf0,
                                    const float pSf1, const float pSf2, const float pSf3, const float pPixelOffset,
                                    bool pAntialias, cudaStream_t pStream) {
  const int64_t cDstSize = pNe10Dst * pNe11Dst * pNe12Dst * pNe13Dst;
  const int64_t cNumBlocks = (cDstSize + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

  if (pAntialias) {
    fUpscaleF32BilinearAntialias<<<cNumBlocks, CUDA_UPSCALE_BLOCK_SIZE, 0, pStream>>>(
        pX, pDst, pNb00, pNb01, pNb02, pNb03, pNe00Src, pNe01Src, pNe10Dst, pNe11Dst, pNe12Dst, pNe13Dst, pSf0, pSf1,
        pSf2, pSf3, pPixelOffset);
  } else {
    fUpscaleF32Bilinear<<<cNumBlocks, CUDA_UPSCALE_BLOCK_SIZE, 0, pStream>>>(
        pX, pDst, pNb00, pNb01, pNb02, pNb03, pNe00Src, pNe01Src, pNe10Dst, pNe11Dst, pNe12Dst, pNe13Dst, pSf0, pSf1,
        pSf2, pSf3, pPixelOffset);
  }
}

static void fUpscaleF32BicubicCuda(const float *pX, float *pDst, const int pNb00, const int pNb01, const int pNb02,
                                   const int pNb03, const int pNe00Src, const int pNe01Src, const int pNe10Dst,
                                   const int pNe11Dst, const int pNe12Dst, const int pNe13Dst, const float pSf0,
                                   const float pSf1, const float pSf2, const float pSf3, const float pPixelOffset,
                                   cudaStream_t pStream) {
  const int64_t cDstSize = pNe10Dst * pNe11Dst * pNe12Dst * pNe13Dst;
  const int64_t cNumBlocks = (cDstSize + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

  fUpscaleF32Bicubic<<<cNumBlocks, CUDA_UPSCALE_BLOCK_SIZE, 0, pStream>>>(
      pX, pDst, pNb00, pNb01, pNb02, pNb03, pNe00Src, pNe01Src, pNe10Dst, pNe11Dst, pNe12Dst, pNe13Dst, pSf0, pSf1,
      pSf2, pSf3, pPixelOffset);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int cModeFlags = pDst->op_params[0];
  const ggml_scale_mode cMode = (ggml_scale_mode)(cModeFlags & 0xFF);

  float vSf0 = (float)pDst->ne[0] / vSrc0->ne[0];
  float vSf1 = (float)pDst->ne[1] / vSrc0->ne[1];
  float vSf2 = (float)pDst->ne[2] / vSrc0->ne[2];
  const float cSf3 = (float)pDst->ne[3] / vSrc0->ne[3];

  float vPixelOffset = 0.5f;
  if (cModeFlags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
    vSf0 = pDst->ne[0] > 1 && vSrc0->ne[0] > 1 ? (float)(pDst->ne[0] - 1) / (vSrc0->ne[0] - 1) : vSf0;
    vSf1 = pDst->ne[1] > 1 && vSrc0->ne[1] > 1 ? (float)(pDst->ne[1] - 1) / (vSrc0->ne[1] - 1) : vSf1;
    vPixelOffset = 0.0f;
  }

  if (cMode == GGML_SCALE_MODE_NEAREST) {
    fUpscaleF32Cuda(vSrc0D, vDstD, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], pDst->ne[0], pDst->ne[1],
                    pDst->ne[2], pDst->ne[3], vSf0, vSf1, vSf2, cSf3, vStream);
  } else if (cMode == GGML_SCALE_MODE_BILINEAR) {
    const bool cAntialias = (cModeFlags & GGML_SCALE_FLAG_ANTIALIAS);
    fUpscaleF32BilinearCuda(vSrc0D, vDstD, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], vSrc0->ne[0],
                            vSrc0->ne[1], pDst->ne[0], pDst->ne[1], pDst->ne[2], pDst->ne[3], vSf0, vSf1, vSf2, cSf3,
                            vPixelOffset, cAntialias, vStream);
  } else if (cMode == GGML_SCALE_MODE_BICUBIC) {
    fUpscaleF32BicubicCuda(vSrc0D, vDstD, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], vSrc0->ne[0],
                           vSrc0->ne[1], pDst->ne[0], pDst->ne[1], pDst->ne[2], pDst->ne[3], vSf0, vSf1, vSf2, cSf3,
                           vPixelOffset, vStream);
  }
}
