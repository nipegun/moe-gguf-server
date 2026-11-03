#include "mtmd-image.h"

#include <algorithm>
#include <cmath>
#include <vector>

void mtmd_image_preproc_out::mAppend(const clip_hparams &pHparams, const clip_image_u8 &pImg, bool pNormalized) {
  clip_image_f32 vDst;
  vDst.mFromU8(pImg);
  if (pNormalized) {
    vDst.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
  lEntries.push_back(std::move(vDst));
}

void mtmd_image_preproc_out::mAppend(const clip_hparams &pHparams, const std::vector<clip_image_u8> &pImgs,
                                     bool pNormalized) {
  for (const auto &vImg : pImgs) {
    mAppend(pHparams, vImg, pNormalized);
  }
}

void mtmd_image_preproc_out::mAppend(const clip_hparams &pHparams, clip_image_f32 &pImg, bool pNormalized) {
  if (pNormalized) {
    pImg.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
  lEntries.push_back(std::move(pImg));
}

void mtmd_image_preproc_out::mAppendOverview(const clip_hparams &pHparams, const clip_image_u8 &pImg,
                                             bool pNormalized) {
  vOverview.mFromU8(pImg);
  if (pNormalized) {
    vOverview.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
}

// set of tools to manipulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
  static void mResize(const clip_image_u8 &pSrc, clip_image_u8 &pDst, const clip_image_size &pTargetResolution,
                      resize_algo pAlgo, pad_style pPadding = cPadCeil, std::array<uint8_t, 3> pPadColor = {0, 0, 0}) {
    pDst.mSetSize(pTargetResolution, pSrc.mIsPlaceholder());

    if (pSrc.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    if (pDst.mGetSize() == pSrc.mGetSize()) {
      // no resize needed, simple copy
      pDst.mCpyBuf(pSrc.mGetRoBuf());
      return;
    }

    if (pPadding == cPadNone) {
      // direct resize
      switch (pAlgo) {
      case cResizeAlgoBilinear:
        mResizeBilinear(pSrc, pDst, pTargetResolution.vWidth, pTargetResolution.vHeight);
        break;
      case cResizeAlgoBicubic:
        mResizeBicubic(pSrc, pDst, pTargetResolution.vWidth, pTargetResolution.vHeight);
        break;
      case cResizeAlgoBicubicPillow:
        mResizeBicubicPillow(pSrc, pDst, pTargetResolution.vWidth, pTargetResolution.vHeight);
        break;
      case cResizeAlgoLanczos:
        mResizeLanczosPillow(pSrc, pDst, pTargetResolution.vWidth, pTargetResolution.vHeight);
        break;
      default:
        throw std::runtime_error("Unsupported resize algorithm");
      }
    } else {
      // resize with padding
      clip_image_u8 vResizedImage;
      float vScaleW = static_cast<float>(pTargetResolution.vWidth) / pSrc.mGetSize().vWidth;
      float vScaleH = static_cast<float>(pTargetResolution.vHeight) / pSrc.mGetSize().vHeight;
      float vScale = std::min(vScaleW, vScaleH);

      int vNewWidth, vNewHeight;
      if (pPadding == cPadNearest) {
        vNewWidth = std::min(static_cast<int>(std::round(pSrc.mGetSize().vWidth * vScale)), pTargetResolution.vWidth);
        vNewHeight =
            std::min(static_cast<int>(std::round(pSrc.mGetSize().vHeight * vScale)), pTargetResolution.vHeight);
      } else {
        vNewWidth = std::min(static_cast<int>(std::ceil(pSrc.mGetSize().vWidth * vScale)), pTargetResolution.vWidth);
        vNewHeight = std::min(static_cast<int>(std::ceil(pSrc.mGetSize().vHeight * vScale)), pTargetResolution.vHeight);
      }

      switch (pAlgo) {
      case cResizeAlgoBilinear:
        mResizeBilinear(pSrc, vResizedImage, vNewWidth, vNewHeight);
        break;
      case cResizeAlgoBicubic:
        mResizeBicubic(pSrc, vResizedImage, vNewWidth, vNewHeight);
        break;
      case cResizeAlgoBicubicPillow:
        mResizeBicubicPillow(pSrc, vResizedImage, vNewWidth, vNewHeight);
        break;
      case cResizeAlgoLanczos:
        mResizeLanczosPillow(pSrc, vResizedImage, vNewWidth, vNewHeight);
        break;
      default:
        throw std::runtime_error("Unsupported resize algorithm");
      }

      // fill dst with pad_color
      mFill(pDst, pPadColor);

      int vOffsetX, vOffsetY;
      if (pPadding == cPadNearest) {
        vOffsetX = static_cast<int>(std::round((pTargetResolution.vWidth - vNewWidth) / 2.0f));
        vOffsetY = static_cast<int>(std::round((pTargetResolution.vHeight - vNewHeight) / 2.0f));
      } else {
        vOffsetX = (pTargetResolution.vWidth - vNewWidth) / 2;
        vOffsetY = (pTargetResolution.vHeight - vNewHeight) / 2;
      }
      mComposite(pDst, vResizedImage, vOffsetX, vOffsetY);
    }
  }

  static void mCrop(const clip_image_u8 &pImage, clip_image_u8 &pDst, int pX, int pY, int pW, int pH) {
    GGML_ASSERT(pX >= 0 && pY >= 0 && pW > 0 && pH > 0);
    GGML_ASSERT(pX + pW <= pImage.mGetSize().vWidth && pY + pH <= pImage.mGetSize().vHeight);
    pDst.mSetSize({pW, pH}, pImage.mIsPlaceholder());

    if (pImage.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    for (int vI = 0; vI < pH; ++vI) {
      for (int vJ = 0; vJ < pW; ++vJ) {
        pDst.mSetPixel(vJ, vI, pImage.mGetPixel(pX + vJ, pY + vI));
      }
    }
  }

  // calculate the size of the **resized** image, while preserving the aspect ratio
  // the calculated size will be aligned to the nearest multiple of align_size
  // if H or W size is larger than longest_edge, it will be resized to longest_edge
  static clip_image_size mCalcSizePreservedRatio(const clip_image_size &pInpSize, const int pAlignSize,
                                                 const int pLongestEdge) {
    GGML_ASSERT(pAlignSize > 0);
    if (pInpSize.vWidth <= 0 || pInpSize.vHeight <= 0 || pLongestEdge <= 0) {
      return {0, 0};
    }

    float vScale = std::min(static_cast<float>(pLongestEdge) / pInpSize.vWidth,
                            static_cast<float>(pLongestEdge) / pInpSize.vHeight);

    float vTargetWidthF = static_cast<float>(pInpSize.vWidth) * vScale;
    float vTargetHeightF = static_cast<float>(pInpSize.vHeight) * vScale;

    auto vCeilByFactor = [vF = pAlignSize](float pX) {
      return static_cast<int>(std::ceil(pX / static_cast<float>(vF))) * vF;
    };
    int vAlignedWidth = vCeilByFactor(vTargetWidthF);
    int vAlignedHeight = vCeilByFactor(vTargetHeightF);

    return {vAlignedWidth, vAlignedHeight};
  }

  // calculate the size of the **resized** image, while preserving the aspect ratio
  // the calculated size will have min_pixels <= W*H <= max_pixels
  // this is referred as "smart_resize" in transformers code
  static clip_image_size mCalcSizePreservedRatio(const clip_image_size &pInpSize, const int pAlignSize,
                                                 const int pMinPixels, const int pMaxPixels) {
    GGML_ASSERT(pAlignSize > 0);
    const int cWidth = pInpSize.vWidth;
    const int cHeight = pInpSize.vHeight;

    auto vRoundByFactor = [vF = pAlignSize](float pX) {
      return static_cast<int>(std::round(pX / static_cast<float>(vF))) * vF;
    };
    auto vCeilByFactor = [vF = pAlignSize](float pX) {
      return static_cast<int>(std::ceil(pX / static_cast<float>(vF))) * vF;
    };
    auto vFloorByFactor = [vF = pAlignSize](float pX) {
      return static_cast<int>(std::floor(pX / static_cast<float>(vF))) * vF;
    };

    // always align up first
    int vHBar = std::max(pAlignSize, vRoundByFactor(cHeight));
    int vWBar = std::max(pAlignSize, vRoundByFactor(cWidth));

    if (vHBar * vWBar > pMaxPixels) {
      const auto cBeta = std::sqrt(static_cast<float>(cHeight * cWidth) / pMaxPixels);
      vHBar = std::max(pAlignSize, vFloorByFactor(cHeight / cBeta));
      vWBar = std::max(pAlignSize, vFloorByFactor(cWidth / cBeta));
    } else if (vHBar * vWBar < pMinPixels) {
      const auto cBeta = std::sqrt(static_cast<float>(pMinPixels) / (cHeight * cWidth));
      vHBar = vCeilByFactor(cHeight * cBeta);
      vWBar = vCeilByFactor(cWidth * cBeta);
    }

    return {vWBar, vHBar};
  }

  // draw src image into dst image at offset (offset_x, offset_y)
  static void mComposite(clip_image_u8 &pDst, const clip_image_u8 &pSrc, int pOffsetX, int pOffsetY) {
    if (pSrc.mIsPlaceholder()) {
      // no-op for placeholder image
      return;
    }

    const auto cSrcSize = pSrc.mGetSize();
    const auto cDstSize = pDst.mGetSize();
    for (int vY = 0; vY < cSrcSize.vHeight; ++vY) {
      for (int vX = 0; vX < cSrcSize.vWidth; ++vX) {
        int vDx = vX + pOffsetX;
        int vDy = vY + pOffsetY;
        // skip pixels that would be out of bounds in the destination
        if (vDx < 0 || vDy < 0 || vDx >= cDstSize.vWidth || vDy >= cDstSize.vHeight) {
          continue;
        }
        pDst.mSetPixel(vDx, vDy, pSrc.mGetPixel(vX, vY));
      }
    }
  }

  // fill the image with a solid color
  static void mFill(clip_image_u8 &pImg, const std::array<uint8_t, 3> &pColor) {
    if (pImg.mIsPlaceholder()) {
      // no-op for placeholder image
      return;
    }

    const auto cSize = pImg.mGetSize();
    for (int vY = 0; vY < cSize.vHeight; ++vY) {
      for (int vX = 0; vX < cSize.vWidth; ++vX) {
        pImg.mSetPixel(vX, vY, pColor);
      }
    }
  }

private:
  // Bilinear resize function
  static void mResizeBilinear(const clip_image_u8 &pSrc, clip_image_u8 &pDst, int pTargetWidth, int pTargetHeight) {
    const auto cSrcSize = pSrc.mGetSize();
    if (cSrcSize.vWidth == 0 || cSrcSize.vHeight == 0) {
      pDst.mSetSize({0, 0}, false);
      return;
    }
    if (pTargetWidth <= 0)
      pTargetWidth = 1;
    if (pTargetHeight <= 0)
      pTargetHeight = 1;

    pDst.mSetSize({pTargetWidth, pTargetHeight}, false);

    if (pSrc.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    float vXRatio = pTargetWidth > 1 ? static_cast<float>(cSrcSize.vWidth - 1) / (pTargetWidth - 1) : 0.0f;
    float vYRatio = pTargetHeight > 1 ? static_cast<float>(cSrcSize.vHeight - 1) / (pTargetHeight - 1) : 0.0f;

    for (int vY = 0; vY < pTargetHeight; ++vY) {
      for (int vX = 0; vX < pTargetWidth; ++vX) {
        float vPx = vX * vXRatio;
        float vPy = vY * vYRatio;

        int vX0 = std::min(static_cast<int>(vPx), cSrcSize.vWidth - 1);
        int vY0 = std::min(static_cast<int>(vPy), cSrcSize.vHeight - 1);
        int vX1 = std::min(vX0 + 1, cSrcSize.vWidth - 1);
        int vY1 = std::min(vY0 + 1, cSrcSize.vHeight - 1);

        float vXf = vPx - vX0;
        float vYf = vPy - vY0;

        const auto cP00 = pSrc.mGetPixel(vX0, vY0);
        const auto cP10 = pSrc.mGetPixel(vX1, vY0);
        const auto cP01 = pSrc.mGetPixel(vX0, vY1);
        const auto cP11 = pSrc.mGetPixel(vX1, vY1);

        std::array<uint8_t, 3> aPixel;
        for (int vC = 0; vC < 3; ++vC) {
          float vTop = mLerp(static_cast<float>(cP00[vC]), static_cast<float>(cP10[vC]), vXf);
          float vBottom = mLerp(static_cast<float>(cP01[vC]), static_cast<float>(cP11[vC]), vXf);
          aPixel[vC] = static_cast<uint8_t>(mLerp(vTop, vBottom, vYf));
        }
        pDst.mSetPixel(vX, vY, aPixel);
      }
    }
  }

  // Bicubic resize function
  // part of image will be cropped if the aspect ratio is different
  static void mResizeBicubic(const clip_image_u8 &pImg, clip_image_u8 &pDst, int pTargetWidth, int pTargetHeight) {
    const auto cImgSize = pImg.mGetSize();
    const int cNx = cImgSize.vWidth;
    const int cNy = cImgSize.vHeight;

    pDst.mSetSize({pTargetWidth, pTargetHeight}, false);

    if (pImg.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    float vCc;
    float aC[5] = {};
    float vD0, vD2, vD3, vA0, vA1, vA2, vA3;
    int vI, vJ, vK, vJj;
    int vX, vY;
    float vDx, vDy;
    float vTx, vTy;

    vTx = (float)cNx / (float)pTargetWidth;
    vTy = (float)cNy / (float)pTargetHeight;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (vI = 0; vI < pTargetHeight; vI++) {
      for (vJ = 0; vJ < pTargetWidth; vJ++) {
        vX = (int)(vTx * vJ);
        vY = (int)(vTy * vI);

        vDx = vTx * vJ - vX;
        vDy = vTy * vI - vY;

        std::array<uint8_t, 3> aPixel;
        for (vK = 0; vK < 3; vK++) {
          for (vJj = 0; vJj <= 3; vJj++) {
            vD0 = pImg.mGetPixel(mClip(vX - 1, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK] -
                  pImg.mGetPixel(mClip(vX, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK];
            vD2 = pImg.mGetPixel(mClip(vX + 1, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK] -
                  pImg.mGetPixel(mClip(vX, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK];
            vD3 = pImg.mGetPixel(mClip(vX + 2, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK] -
                  pImg.mGetPixel(mClip(vX, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK];
            vA0 = pImg.mGetPixel(mClip(vX, 0, cNx - 1), mClip(vY - 1 + vJj, 0, cNy - 1))[vK];

            vA1 = -1.0 / 3 * vD0 + vD2 - 1.0 / 6 * vD3;
            vA2 = 1.0 / 2 * vD0 + 1.0 / 2 * vD2;
            vA3 = -1.0 / 6 * vD0 - 1.0 / 2 * vD2 + 1.0 / 6 * vD3;

            aC[vJj] = vA0 + vA1 * vDx + vA2 * vDx * vDx + vA3 * vDx * vDx * vDx;

            vD0 = aC[0] - aC[1];
            vD2 = aC[2] - aC[1];
            vD3 = aC[3] - aC[1];
            vA0 = aC[1];
            vA1 = -1.0 / 3 * vD0 + vD2 - 1.0 / 6 * vD3;
            vA2 = 1.0 / 2 * vD0 + 1.0 / 2 * vD2;
            vA3 = -1.0 / 6 * vD0 - 1.0 / 2 * vD2 + 1.0 / 6 * vD3;
            vCc = vA0 + vA1 * vDy + vA2 * vDy * vDy + vA3 * vDy * vDy * vDy;

            const uint8_t cCc2 = std::min(std::max(std::round(vCc), 0.0f), 255.0f);
            aPixel[vK] = cCc2;
          }
        }
        pDst.mSetPixel(vJ, vI, aPixel);
      }
    }
  }

  // Pillow-compatible separable resampling (Bicubic and Lanczos)
  // Adapted from https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
  //
  // Key properties:
  // 1. Separable filtering: horizontal pass followed by vertical pass
  // 2. Pre-computes normalized filter coefficients for each output pixel
  // 3. Fixed-point integer arithmetic (22 fractional bits) for speed and determinism
  static bool mResizeBicubicPillow(const clip_image_u8 &pImg, clip_image_u8 &pDst, int pTargetWidth,
                                   int pTargetHeight) {
    return mResizePillow(pImg, pDst, pTargetWidth, pTargetHeight, /*use_lanczos=*/false);
  }

  // Lanczos-3 (support radius 3), matches Pillow's Image.LANCZOS
  static bool mResizeLanczosPillow(const clip_image_u8 &pImg, clip_image_u8 &pDst, int pTargetWidth,
                                   int pTargetHeight) {
    return mResizePillow(pImg, pDst, pTargetWidth, pTargetHeight, /*use_lanczos=*/true);
  }

  static bool mResizePillow(const clip_image_u8 &pImg, clip_image_u8 &pDst, int pTargetWidth, int pTargetHeight,
                            bool pUseLanczos) {
    // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
    // This allows encoding fractional weights as integers: weight * 2^22
    const int cPrecisionBits = 32 - 8 - 2;

    // Resample filter: Lanczos-3 (support [-3, 3]) or bicubic with a = -0.5 (support [-2, 2])
    // Note: GGML/PyTorch bicubic uses a = -0.75, Pillow uses a = -0.5
    // Returns filter weight for distance x from pixel center
    auto vResampleFilter = [pUseLanczos](double pX) -> double {
      if (pUseLanczos) {
        if (-3.0 <= pX && pX < 3.0) {
          auto vSinc = [](double pV) {
            if (pV == 0.0) {
              return 1.0;
            }
            const double cPiV = pV * 3.141592653589793238462643383279502884;
            return std::sin(cPiV) / cPiV;
          };
          return vSinc(pX) * vSinc(pX / 3.0);
        }
        return 0.0;
      }

      constexpr double cA = -0.5;
      if (pX < 0.0) {
        pX = -pX;
      }
      if (pX < 1.0) {
        return ((cA + 2.0) * pX - (cA + 3.0)) * pX * pX + 1;
      }
      if (pX < 2.0) {
        return (((pX - 5) * pX + 8) * pX - 4) * cA;
      }
      return 0.0; // Zero outside [-2, 2]
    };

    // Filter support radius: 2 for bicubic, 3 for lanczos
    const double cFilterSupport = pUseLanczos ? 3.0 : 2.0;

    // Clipping function for 8-bit values
    auto vClip8 = [](int pVal) -> uint8_t {
      if (pVal < 0)
        return 0;
      if (pVal > 255)
        return 255;
      return static_cast<uint8_t>(pVal);
    };

    // Precompute filter coefficients for ONE dimension (horizontal or vertical)
    //
    // Parameters:
    //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
    //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
    //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
    //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
    //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
    //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
    //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
    //
    // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
    auto vPrecomputeWeights = [&](int pInSize, int pOutSize, std::vector<int> &pBounds,
                                  std::vector<int32_t> &pWeights) -> int {
      GGML_ASSERT(pInSize > 0 && pOutSize > 0);
      double vSupport, vScale, vFilterscale;
      double vCenter, vWw, vSs;
      int vXx, vX, vKsize, vXmin, vXmax;

      // Calculate scaling factor: ratio of input range to output size
      vFilterscale = vScale = static_cast<double>(pInSize) / pOutSize;
      // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
      // For downsampling (scale > 1), widen filter to prevent aliasing
      if (vFilterscale < 1.0) {
        vFilterscale = 1.0;
      }

      // Determine filter support radius and kernel size
      vSupport = cFilterSupport * vFilterscale;               // Widen filter when downsampling
      vKsize = static_cast<int>(std::ceil(vSupport)) * 2 + 1; // Total pixels in kernel

      std::vector<double> lPreWeights(pOutSize * vKsize); // Temporary weights
      pBounds.resize(pOutSize * 2);

      // For each output pixel, compute its filter coefficients
      for (vXx = 0; vXx < pOutSize; vXx++) {
        // Calculate the center position in input space (pixel-center convention: +0.5)
        vCenter = (vXx + 0.5) * vScale;
        vWw = 0.0;                // Sum of weights for normalization
        vSs = 1.0 / vFilterscale; // Scale factor for filter function

        // Determine the range of input pixels that contribute to this output pixel
        vXmin = static_cast<int>(vCenter - vSupport + 0.5);
        if (vXmin < 0) {
          vXmin = 0;
        }

        vXmax = static_cast<int>(vCenter + vSupport + 0.5);
        if (vXmax > pInSize) {
          vXmax = pInSize;
        }

        vXmax -= vXmin;

        // Compute filter weights for each contributing input pixel
        for (vX = 0; vX < vXmax; vX++) {
          // Distance from input pixel center to output pixel center in input space
          double vW = vResampleFilter((vX + vXmin - vCenter + 0.5) * vSs);
          lPreWeights[vXx * vKsize + vX] = vW;
          vWw += vW; // Accumulate for normalization
        }

        // Normalize weights to sum to 1.0 (preserves brightness)
        for (vX = 0; vX < vXmax; vX++) {
          if (vWw != 0.0) {
            lPreWeights[vXx * vKsize + vX] /= vWw;
          }
        }

        // Zero-pad remaining kernel positions
        for (; vX < vKsize; vX++) {
          lPreWeights[vXx * vKsize + vX] = 0;
        }

        // Store input pixel range for this output pixel
        pBounds[vXx * 2 + 0] = vXmin;
        pBounds[vXx * 2 + 1] = vXmax;
      }

      // Convert floating-point coefficients to fixed-point integers
      // Formula: int32 = round(float * 2^PRECISION_BITS)
      pWeights.resize(pOutSize * vKsize);

      const double cFxpScale = std::ldexp(1.0, cPrecisionBits); // 1.0 * 2^PRECISION_BITS

      for (int vI = 0; vI < pOutSize * vKsize; vI++) {
        if (pUseLanczos) {
          // Pillow adds +/- 0.5 then truncates toward zero; std::round would round twice
          const double cRounded = lPreWeights[vI] * cFxpScale + (lPreWeights[vI] < 0 ? -0.5 : 0.5);
          pWeights[vI] = static_cast<int32_t>(cRounded);
          continue;
        }
        double vTmpVal = lPreWeights[vI] * cFxpScale;
        if (lPreWeights[vI] < 0) {
          vTmpVal -= 0.5;
        } else {
          vTmpVal += 0.5;
        }
        vTmpVal = std::round(vTmpVal);
        vTmpVal = std::clamp(vTmpVal, static_cast<double>(std::numeric_limits<int32_t>::min()),
                             static_cast<double>(std::numeric_limits<int32_t>::max()));
        pWeights[vI] = static_cast<int32_t>(vTmpVal);
      }

      return vKsize;
    };

    // Horizontal resampling pass
    // Resizes width from imIn to out_nx, preserving height
    auto vResampleHorizontal = [&](const clip_image_u8 &pImIn, clip_image_u8 &pImOut, int pOutNx, int pKsize,
                                   const std::vector<int> &pBounds, const std::vector<int32_t> &pWeights) {
      const int cInNy = pImIn.mGetSize().vHeight;
      pImOut.mSetSize({pOutNx, cInNy}, false);

      // Process each row independently
      for (int vYy = 0; vYy < cInNy; vYy++) {
        // For each output pixel in this row
        for (int vXx = 0; vXx < pOutNx; vXx++) {
          // Get the range of input pixels and filter coefficients
          int vXmin = pBounds[vXx * 2 + 0]; // First input pixel index
          int vXcnt = pBounds[vXx * 2 + 1]; // Number of input pixels

          // Initialize accumulators for RGB channels with rounding bias (0.5 in fixed-point)
          int32_t vSs0 = 1 << (cPrecisionBits - 1);
          int32_t vSs1 = 1 << (cPrecisionBits - 1);
          int32_t vSs2 = 1 << (cPrecisionBits - 1);

          // Convolve: sum weighted input pixels
          for (int vX = 0; vX < vXcnt; vX++) {
            const auto cSrcPx = pImIn.mGetPixel(vX + vXmin, vYy);
            vSs0 += cSrcPx[0] * pWeights[vXx * pKsize + vX]; // R channel
            vSs1 += cSrcPx[1] * pWeights[vXx * pKsize + vX]; // G channel
            vSs2 += cSrcPx[2] * pWeights[vXx * pKsize + vX]; // B channel
          }

          // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
          pImOut.mSetPixel(
              vXx, vYy,
              {vClip8(vSs0 >> cPrecisionBits), vClip8(vSs1 >> cPrecisionBits), vClip8(vSs2 >> cPrecisionBits)});
        }
      }
    };

    // Vertical resampling pass
    // Resizes height from imIn to out_ny, preserving width
    auto vResampleVertical = [&](const clip_image_u8 &pImIn, clip_image_u8 &pImOut, int pOutNy, int pKsize,
                                 const std::vector<int> &pBounds, const std::vector<int32_t> &pWeight) {
      const int cInNx = pImIn.mGetSize().vWidth;
      pImOut.mSetSize({cInNx, pOutNy}, false);

      // For each output row
      for (int vYy = 0; vYy < pOutNy; vYy++) {
        // Get the range of input rows and filter coefficients
        int vYmin = pBounds[vYy * 2 + 0]; // First input row index
        int vYcnt = pBounds[vYy * 2 + 1]; // Number of input rows

        // Process each column in this output row
        for (int vXx = 0; vXx < cInNx; vXx++) {
          // Initialize accumulators for RGB channels with rounding bias
          int32_t vSs0 = 1 << (cPrecisionBits - 1);
          int32_t vSs1 = 1 << (cPrecisionBits - 1);
          int32_t vSs2 = 1 << (cPrecisionBits - 1);

          // Convolve: sum weighted input pixels vertically
          for (int vY = 0; vY < vYcnt; vY++) {
            const auto cSrcPx = pImIn.mGetPixel(vXx, vY + vYmin);
            vSs0 += cSrcPx[0] * pWeight[vYy * pKsize + vY]; // R channel
            vSs1 += cSrcPx[1] * pWeight[vYy * pKsize + vY]; // G channel
            vSs2 += cSrcPx[2] * pWeight[vYy * pKsize + vY]; // B channel
          }

          // Convert back from fixed-point and clamp to [0,255]
          pImOut.mSetPixel(
              vXx, vYy,
              {vClip8(vSs0 >> cPrecisionBits), vClip8(vSs1 >> cPrecisionBits), vClip8(vSs2 >> cPrecisionBits)});
        }
      }
    };

    // Main resampling logic using separable two-pass approach
    const int cSrcWidth = pImg.mGetSize().vWidth;
    const int cSrcHeight = pImg.mGetSize().vHeight;

    bool vNeedHorizontal = (pTargetWidth != cSrcWidth);
    bool vNeedVertical = (pTargetHeight != cSrcHeight);

    // Precompute filter coefficients for both dimensions
    std::vector<int> lBoundsHoriz, lBoundsVert;
    std::vector<int32_t> lWeightsHoriz, lWeightsVert;
    int vKsizeHoriz = 0, vKsizeVert = 0;

    if (vNeedHorizontal) {
      vKsizeHoriz = vPrecomputeWeights(cSrcWidth, pTargetWidth, lBoundsHoriz, lWeightsHoriz);
    }

    if (vNeedVertical) {
      vKsizeVert = vPrecomputeWeights(cSrcHeight, pTargetHeight, lBoundsVert, lWeightsVert);
    }

    // Perform two-pass resampling
    if (vNeedHorizontal && vNeedVertical) {
      // Both horizontal and vertical
      clip_image_u8 vTemp;
      vResampleHorizontal(pImg, vTemp, pTargetWidth, vKsizeHoriz, lBoundsHoriz, lWeightsHoriz);
      vResampleVertical(vTemp, pDst, pTargetHeight, vKsizeVert, lBoundsVert, lWeightsVert);
    } else if (vNeedHorizontal) {
      // Only horizontal
      vResampleHorizontal(pImg, pDst, pTargetWidth, vKsizeHoriz, lBoundsHoriz, lWeightsHoriz);
    } else if (vNeedVertical) {
      // Only vertical
      vResampleVertical(pImg, pDst, pTargetHeight, vKsizeVert, lBoundsVert, lWeightsVert);
    } else {
      // No resizing needed - direct copy
      pDst.mSetSize(pImg.mGetSize(), pImg.mIsPlaceholder());
      if (!pImg.mIsPlaceholder()) {
        pDst.mCpyBuf(pImg.mGetRoBuf());
      }
    }

    return true;
  }

  static inline int mClip(int pX, int pLower, int pUpper) { return std::max(pLower, std::min(pX, pUpper)); }

  // Linear interpolation between two points
  static inline float mLerp(float pS, float pE, float pT) { return pS + (pE - pS) * pT; }
};

//
// mtmd_image_preprocessor_llava_uhd
//

mtmd_image_preproc_out mtmd_image_preprocessor_llava_uhd::mPreprocess(const clip_image_u8 &pImg) {
  const clip_image_size cOriginalSize = pImg.mGetSize();
  auto const cInst = mGetSliceInstructions(cOriginalSize);
  auto vSliced = mSliceImage(pImg, cInst);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.vGridX = cInst.vGridSize.vWidth;
  vOutput.vGridY = cInst.vGridSize.vHeight;

  return vOutput;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions
mtmd_image_preprocessor_llava_uhd::mGetSliceInstructions(const clip_image_size &pOriginalSize) {
  mtmd_image_preprocessor_llava_uhd::slice_instructions vRes;
  // align slices by patch_size * n_merge so an integer number of merger output tokens fits per slice
  const int cNMerge = vHparams.vNMerge;
  const int cPatchSize = vHparams.vPatchSize * cNMerge;
  const int cSliceSize = vHparams.vImageSize;
  const int cOriginalWidth = pOriginalSize.vWidth;
  const int cOriginalHeight = pOriginalSize.vHeight;

  const bool cHasSlices = pOriginalSize.vWidth > cSliceSize || pOriginalSize.vHeight > cSliceSize;
  const bool cHasPinpoints = !vHparams.lImageResCandidates.empty();

  if (!cHasSlices) {
    // skip slicing logic
    vRes.vOverviewSize = clip_image_size{cSliceSize, cSliceSize};
    vRes.vRefinedSize = clip_image_size{0, 0};
    vRes.vGridSize = clip_image_size{0, 0};

    return vRes;
  }

  if (cHasPinpoints) {
    // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
    auto vRefineSize = mSelectBestResolution(pOriginalSize, vHparams.lImageResCandidates);
    vRes.vOverviewSize = clip_image_size{cSliceSize, cSliceSize};
    vRes.vRefinedSize = vRefineSize;
    vRes.vGridSize = clip_image_size{0, 0};

    LOG_DBG("%s: using pinpoints for slicing\n", __func__);
    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n", __func__, cOriginalWidth,
            cOriginalHeight, vRes.vOverviewSize.vWidth, vRes.vOverviewSize.vHeight, vRes.vRefinedSize.vWidth,
            vRes.vRefinedSize.vHeight);

    for (int vY = 0; vY < vRefineSize.vHeight; vY += cSliceSize) {
      for (int vX = 0; vX < vRefineSize.vWidth; vX += cSliceSize) {
        slice_coordinates vSlice;
        vSlice.vX = vX;
        vSlice.vY = vY;
        vSlice.vSize.vWidth = std::min(cSliceSize, vRefineSize.vWidth - vX);
        vSlice.vSize.vHeight = std::min(cSliceSize, vRefineSize.vHeight - vY);
        vRes.lSlices.push_back(vSlice);
        LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n", __func__, (int)vRes.lSlices.size() - 1, vSlice.vX, vSlice.vY,
                vSlice.vSize.vWidth, vSlice.vSize.vHeight);
      }
    }

    vRes.vGridSize.vHeight = vRefineSize.vHeight / cSliceSize;
    vRes.vGridSize.vWidth = vRefineSize.vWidth / cSliceSize;
    LOG_DBG("%s: grid size: %d x %d\n", __func__, vRes.vGridSize.vWidth, vRes.vGridSize.vHeight);

    return vRes;
  }

  // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

  auto vBestSize = mGetBestResize(pOriginalSize, cSliceSize, cPatchSize, !cHasSlices);
  vRes.vOverviewSize = vBestSize;

  {
    const int cMaxSliceNums = 9; // TODO: this is only used by minicpmv, maybe remove it
    const float cLogRatio = log((float)cOriginalWidth / cOriginalHeight);
    const float cRatio = (float)cOriginalWidth * cOriginalHeight / (cSliceSize * cSliceSize);
    const int cMultiple = fmin(ceil(cRatio), cMaxSliceNums);

    auto vBestGrid = mGetBestGrid(cMaxSliceNums, cMultiple, cLogRatio);
    auto vRefineSize = mGetRefineSize(pOriginalSize, vBestGrid, cSliceSize, cPatchSize, true);
    vRes.vGridSize = vBestGrid;
    vRes.vRefinedSize = vRefineSize;

    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", __func__,
            cOriginalWidth, cOriginalHeight, vRes.vOverviewSize.vWidth, vRes.vOverviewSize.vHeight,
            vRes.vRefinedSize.vWidth, vRes.vRefinedSize.vHeight, vRes.vGridSize.vWidth, vRes.vGridSize.vHeight);

    int vWidth = vRefineSize.vWidth;
    int vHeight = vRefineSize.vHeight;
    int vGridX = int(vWidth / vBestGrid.vWidth);
    int vGridY = int(vHeight / vBestGrid.vHeight);
    for (int vPatchesY = 0, vIc = 0; vPatchesY < vRefineSize.vHeight && vIc < vBestGrid.vHeight;
         vPatchesY += vGridY, vIc += 1) {
      for (int vPatchesX = 0, vJc = 0; vPatchesX < vRefineSize.vWidth && vJc < vBestGrid.vWidth;
           vPatchesX += vGridX, vJc += 1) {
        slice_coordinates vSlice;
        vSlice.vX = vPatchesX;
        vSlice.vY = vPatchesY;
        vSlice.vSize.vWidth = vGridX;
        vSlice.vSize.vHeight = vGridY;
        vRes.lSlices.push_back(vSlice);
        LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n", __func__, (int)vRes.lSlices.size() - 1, vSlice.vX, vSlice.vY,
                vSlice.vSize.vWidth, vSlice.vSize.vHeight);
      }
    }
  }

  return vRes;
}

mtmd_image_preprocessor_llava_uhd::slice_output
mtmd_image_preprocessor_llava_uhd::mSliceImage(const clip_image_u8 &pImg,
                                               const mtmd_image_preprocessor_llava_uhd::slice_instructions &pInst) {
  slice_output vOutput;

  // resize to overview size
  img_tool::mResize(pImg, vOutput.vOverview, pInst.vOverviewSize, vHparams.vImageResizeAlgoOv, vHparams.vImagePadOv,
                    vHparams.aImagePadColorOv);

  if (pInst.lSlices.empty()) {
    // no slices, just return the overview image
    return vOutput;
  }

  // resize to refined size
  clip_image_u8 vRefinedImg;
  img_tool::mResize(pImg, vRefinedImg, pInst.vRefinedSize, vHparams.vImageResizeAlgoRf, vHparams.vImagePadRf,
                    vHparams.aImagePadColorRf);

  // create slices
  for (const auto &vSlice : pInst.lSlices) {
    int vX = vSlice.vX;
    int vY = vSlice.vY;
    int vW = vSlice.vSize.vWidth;
    int vH = vSlice.vSize.vHeight;

    clip_image_u8 vImgSlice;
    img_tool::mCrop(vRefinedImg, vImgSlice, vX, vY, vW, vH);
    vOutput.lSlices.push_back(std::move(vImgSlice));
  }

  return vOutput;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetBestResize(const clip_image_size &pOriginalSize,
                                                                  int pScaleResolution, int pPatchSize,
                                                                  bool pAllowUpscale) {
  int vWidth = pOriginalSize.vWidth;
  int vHeight = pOriginalSize.vHeight;
  if ((vWidth * vHeight > pScaleResolution * pScaleResolution) || pAllowUpscale) {
    float vR = static_cast<float>(vWidth) / vHeight;
    vHeight = static_cast<int>(pScaleResolution / std::sqrt(vR));
    vWidth = static_cast<int>(vHeight * vR);
  }
  clip_image_size vRes;
  vRes.vWidth = mEnsureDivide(vWidth, pPatchSize);
  vRes.vHeight = mEnsureDivide(vHeight, pPatchSize);
  return vRes;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mResizeMaintainAspectRatio(const clip_image_size &pOrig,
                                                                              const clip_image_size &pTargetMax) {
  float vScaleWidth = static_cast<float>(pTargetMax.vWidth) / pOrig.vWidth;
  float vScaleHeight = static_cast<float>(pTargetMax.vHeight) / pOrig.vHeight;
  float vScale = std::min(vScaleWidth, vScaleHeight);
  return clip_image_size{
      static_cast<int>(pOrig.vWidth * vScale),
      static_cast<int>(pOrig.vHeight * vScale),
  };
}

clip_image_size
mtmd_image_preprocessor_llava_uhd::mSelectBestResolution(const clip_image_size &pOriginalSize,
                                                         const std::vector<clip_image_size> &pPossibleResolutions) {
  clip_image_size vBestFit;
  int vMinWastedArea = std::numeric_limits<int>::max();
  int vMaxEffectiveResolution = 0;

  for (const clip_image_size &vCandidate : pPossibleResolutions) {
    auto vTargetSize = mResizeMaintainAspectRatio(pOriginalSize, vCandidate);
    int vEffectiveResolution =
        std::min(vTargetSize.vWidth * vTargetSize.vHeight, pOriginalSize.vWidth * pOriginalSize.vHeight);
    int vWastedArea = (vCandidate.vWidth * vCandidate.vHeight) - vEffectiveResolution;

    if (vEffectiveResolution > vMaxEffectiveResolution ||
        (vEffectiveResolution == vMaxEffectiveResolution && vWastedArea < vMinWastedArea)) {
      vMaxEffectiveResolution = vEffectiveResolution;
      vMinWastedArea = vWastedArea;
      vBestFit = vCandidate;
    }

    LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, vCandidate.vWidth,
            vCandidate.vHeight, vTargetSize.vWidth, vTargetSize.vHeight, vWastedArea, vEffectiveResolution);
  }

  return vBestFit;
}

int mtmd_image_preprocessor_llava_uhd::mEnsureDivide(int pLength, int pPatchSize) {
  return std::max(static_cast<int>(std::round(static_cast<float>(pLength) / pPatchSize) * pPatchSize), pPatchSize);
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetRefineSize(const clip_image_size &pOriginalSize,
                                                                  const clip_image_size &pGrid, int pScaleResolution,
                                                                  int pPatchSize, bool pAllowUpscale) {
  int vWidth = pOriginalSize.vWidth;
  int vHeight = pOriginalSize.vHeight;
  int vGridX = pGrid.vWidth;
  int vGridY = pGrid.vHeight;

  int vRefineWidth = mEnsureDivide(vWidth, vGridX);
  int vRefineHeight = mEnsureDivide(vHeight, vGridY);

  clip_image_size vGridSize;
  vGridSize.vWidth = vRefineWidth / vGridX;
  vGridSize.vHeight = vRefineHeight / vGridY;

  auto vBestGridSize = mGetBestResize(vGridSize, pScaleResolution, pPatchSize, pAllowUpscale);
  int vBestGridWidth = vBestGridSize.vWidth;
  int vBestGridHeight = vBestGridSize.vHeight;

  clip_image_size vRefineSize;
  vRefineSize.vWidth = vBestGridWidth * vGridX;
  vRefineSize.vHeight = vBestGridHeight * vGridY;
  return vRefineSize;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetBestGrid(const int pMaxSliceNums, const int pMultiple,
                                                                const float pLogRatio) {
  std::vector<int> lCandidateSplitGridsNums;
  for (int vI : {pMultiple - 1, pMultiple, pMultiple + 1}) {
    if (vI == 1 || vI > pMaxSliceNums) {
      continue;
    }
    lCandidateSplitGridsNums.push_back(vI);
  }

  std::vector<clip_image_size> lCandidateGrids;
  for (int vSplitGridsNums : lCandidateSplitGridsNums) {
    int vM = 1;
    while (vM <= vSplitGridsNums) {
      if (vSplitGridsNums % vM == 0) {
        lCandidateGrids.push_back(clip_image_size{vM, vSplitGridsNums / vM});
      }
      ++vM;
    }
  }

  clip_image_size vBestGrid{1, 1};
  float vMinError = std::numeric_limits<float>::infinity();
  for (const auto &vGrid : lCandidateGrids) {
    float vError = std::abs(pLogRatio - std::log(1.0 * vGrid.vWidth / vGrid.vHeight));
    if (vError < vMinError) {
      vBestGrid = vGrid;
      vMinError = vError;
    }
  }
  return vBestGrid;
}

//
// mtmd_image_preprocessor_fixed_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_fixed_size::mPreprocess(const clip_image_u8 &pImg) {
  clip_image_u8 vResizedImage;
  int vSz = vHparams.vImageSize;
  img_tool::mResize(pImg, vResizedImage, {vSz, vSz}, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                    vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_dyn_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_dyn_size::mPreprocess(const clip_image_u8 &pImg) {
  GGML_ASSERT(vHparams.vImageMinPixels > 0 && vHparams.vImageMaxPixels > 0);
  clip_image_u8 vResizedImage;
  const clip_image_size cOriginalSize = pImg.mGetSize();
  // the original pixtral model doesn't have n_merge
  const int cCurMerge = vHparams.vNMerge;
  const clip_image_size cTargetSize = img_tool::mCalcSizePreservedRatio(
      cOriginalSize, vHparams.vPatchSize * cCurMerge, vHparams.vImageMinPixels, vHparams.vImageMaxPixels);
  img_tool::mResize(pImg, vResizedImage, cTargetSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                    vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_glm5v
//

// The canvas is ceil-aligned to patch_size*n_merge and fitted to the token budget.
// The content is only rescaled to meet the budget and sits top-left, with black padding on the right and bottom
mtmd_image_preproc_out mtmd_image_preprocessor_glm5v::mPreprocess(const clip_image_u8 &pImg) {
  GGML_ASSERT(vHparams.vImageMinPixels > 0 && vHparams.vImageMaxPixels > 0);

  const int64_t cFactor = vHparams.vPatchSize * vHparams.vNMerge;
  const int64_t cMinPx = vHparams.vImageMinPixels; // single-frame pixel counts
  const int64_t cMaxPx = vHparams.vImageMaxPixels;
  const int64_t cHeight = pImg.mGetSize().vHeight;
  const int64_t cWidth = pImg.mGetSize().vWidth;

  auto fAlign = [cFactor](int64_t pV) { return (pV + cFactor - 1) / cFactor * cFactor; };

  // aligned canvas within the budget
  int64_t vCanvasH = fAlign(cHeight);
  int64_t vCanvasW = fAlign(cWidth);

  if (vCanvasH * vCanvasW < cMinPx) {
    const double cScale = std::sqrt((double)cMinPx / (double)(cHeight * cWidth));
    vCanvasH = fAlign(std::max<int64_t>(1, (int64_t)std::ceil(cHeight * cScale)));
    vCanvasW = fAlign(std::max<int64_t>(1, (int64_t)std::ceil(cWidth * cScale)));
  }

  if (vCanvasH * vCanvasW > cMaxPx) {
    // largest content height whose aligned canvas fits the budget
    int64_t vLo = 1;
    int64_t vHi = cHeight;
    int64_t vBestH = cFactor;
    int64_t vBestW = cFactor;
    while (vLo <= vHi) {
      const int64_t cCh = (vLo + vHi) / 2;
      const int64_t cCw = std::max<int64_t>(1, cWidth * cCh / cHeight);
      const int64_t cAh = fAlign(cCh);
      const int64_t cAw = fAlign(cCw);
      if (cAh * cAw <= cMaxPx) {
        vBestH = cAh;
        vBestW = cAw;
        vLo = cCh + 1;
      } else {
        vHi = cCh - 1;
      }
    }
    vCanvasH = vBestH;
    vCanvasW = vBestW;
  }

  // scaled to fit the canvas, and never upscaled, unless below the min budget
  double vScale = std::min((double)vCanvasH / cHeight, (double)vCanvasW / cWidth);
  if (cHeight * cWidth >= cMinPx) {
    vScale = std::min(1.0, vScale);
  }
  const int cContentH = (int)std::max<int64_t>(1, std::min<int64_t>(vCanvasH, (int64_t)std::floor(cHeight * vScale)));
  const int cContentW = (int)std::max<int64_t>(1, std::min<int64_t>(vCanvasW, (int64_t)std::floor(cWidth * vScale)));

  clip_image_u8 vContent;
  img_tool::mResize(pImg, vContent, clip_image_size{cContentW, cContentH}, vHparams.vImageResizeAlgo, cPadNone);

  clip_image_u8 vCanvas;
  vCanvas.mSetSize(clip_image_size{(int)vCanvasW, (int)vCanvasH}, pImg.mIsPlaceholder());
  img_tool::mFill(vCanvas, {0, 0, 0});
  img_tool::mComposite(vCanvas, vContent, 0, 0);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vCanvas, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_longest_edge
//

mtmd_image_preproc_out mtmd_image_preprocessor_longest_edge::mPreprocess(const clip_image_u8 &pImg) {
  GGML_ASSERT(vHparams.vImageLongestEdge > 0);
  clip_image_u8 vResizedImage;
  const clip_image_size cOriginalSize = pImg.mGetSize();
  // the original pixtral model doesn't have n_merge
  const int cCurMerge = vHparams.vNMerge == 0 ? 1 : vHparams.vNMerge;
  const clip_image_size cTargetSize =
      img_tool::mCalcSizePreservedRatio(cOriginalSize, vHparams.vPatchSize * cCurMerge, vHparams.vImageLongestEdge);
  img_tool::mResize(pImg, vResizedImage, cTargetSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                    vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_minicpmv
//

mtmd_image_preprocessor_llava_uhd::slice_instructions
mtmd_image_preprocessor_minicpmv::mGetSliceInstructions(const clip_image_size &pOriginalSize) {
  if (vHparams.vNMerge == 2) {
    const int cSliceSize = vHparams.vImageSize;
    const float cRatio = (float)pOriginalSize.vWidth * pOriginalSize.vHeight / (cSliceSize * cSliceSize);
    if (cRatio <= 1.0f) {
      mtmd_image_preprocessor_llava_uhd::slice_instructions vInst;
      const int cPatchSize = vHparams.vPatchSize * vHparams.vNMerge;
      vInst.vOverviewSize = mGetBestResize(pOriginalSize, cSliceSize, cPatchSize, true);
      vInst.vRefinedSize = clip_image_size{0, 0};
      vInst.vGridSize = clip_image_size{0, 0};
      return vInst;
    }
  }
  return mtmd_image_preprocessor_llava_uhd::mGetSliceInstructions(pOriginalSize);
}

//
// mtmd_image_preprocessor_lfm2
//

mtmd_image_preprocessor_llava_uhd::slice_instructions
mtmd_image_preprocessor_lfm2::mGetSliceInstructions(const clip_image_size &pOriginalSize) {
  mtmd_image_preprocessor_llava_uhd::slice_instructions vInst;
  const int cAlignSize = vHparams.vPatchSize * vHparams.vNMerge;
  vInst.vOverviewSize =
      img_tool::mCalcSizePreservedRatio(pOriginalSize, cAlignSize, vHparams.vImageMinPixels, vHparams.vImageMaxPixels);
  // tile if either dimension exceeds tile_size with tolerance
  const bool cNeedsTiling =
      pOriginalSize.vWidth > cTileSize * cMaxPixelsTolerance || pOriginalSize.vHeight > cTileSize * cMaxPixelsTolerance;

  if (!cNeedsTiling) {
    vInst.vRefinedSize = clip_image_size{0, 0};
    vInst.vGridSize = clip_image_size{0, 0};
    return vInst;
  }

  const clip_image_size cGrid = mGetGridLayout(pOriginalSize.vHeight, pOriginalSize.vWidth);

  vInst.vGridSize = cGrid;
  vInst.vRefinedSize = clip_image_size{cTileSize * cGrid.vWidth, cTileSize * cGrid.vHeight};

  LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", __func__,
          pOriginalSize.vWidth, pOriginalSize.vHeight, vInst.vOverviewSize.vWidth, vInst.vOverviewSize.vHeight,
          vInst.vRefinedSize.vWidth, vInst.vRefinedSize.vHeight, cGrid.vWidth, cGrid.vHeight);

  for (int vRow = 0; vRow < cGrid.vHeight; vRow++) {
    for (int vCol = 0; vCol < cGrid.vWidth; vCol++) {
      mtmd_image_preprocessor_llava_uhd::slice_coordinates vSlice;
      vSlice.vX = vCol * cTileSize;
      vSlice.vY = vRow * cTileSize;
      vSlice.vSize = clip_image_size{cTileSize, cTileSize};
      vInst.lSlices.push_back(vSlice);
      LOG_DBG("%s: slice %d: x=%d, y=%d, size=%d x %d\n", __func__, (int)vInst.lSlices.size() - 1, vSlice.vX, vSlice.vY,
              vSlice.vSize.vWidth, vSlice.vSize.vHeight);
    }
  }

  return vInst;
}

clip_image_size mtmd_image_preprocessor_lfm2::mFindClosestAspectRatio(float pAspectRatio,
                                                                      const std::vector<clip_image_size> &pTargetRatios,
                                                                      int pWidth, int pHeight) {
  float vBestRatioDiff = std::numeric_limits<float>::max();
  clip_image_size vBestRatio = {1, 1};
  const float cArea = static_cast<float>(pWidth * pHeight);

  for (const auto &vRatio : pTargetRatios) {
    const float cTargetAspectRatio = static_cast<float>(vRatio.vWidth) / vRatio.vHeight;
    const float cRatioDiff = std::abs(pAspectRatio - cTargetAspectRatio);
    if (cRatioDiff < vBestRatioDiff) {
      vBestRatioDiff = cRatioDiff;
      vBestRatio = vRatio;
    } else if (cRatioDiff == vBestRatioDiff) {
      const float cTargetArea = static_cast<float>(cTileSize * cTileSize * vRatio.vWidth * vRatio.vHeight);
      if (cArea > 0.5f * cTargetArea) {
        vBestRatio = vRatio;
      }
    }
  }
  return vBestRatio;
}

std::vector<clip_image_size> mtmd_image_preprocessor_lfm2::mGetTargetRatios() {
  std::vector<clip_image_size> lRatios;
  for (int vN = cMinTiles; vN <= cMaxTiles; vN++) {
    for (int vW = 1; vW <= vN; vW++) {
      for (int vH = 1; vH <= vN; vH++) {
        if (vW * vH >= cMinTiles && vW * vH <= cMaxTiles) {
          bool vFound = false;
          for (const auto &vR : lRatios) {
            if (vR.vWidth == vW && vR.vHeight == vH) {
              vFound = true;
              break;
            }
          }
          if (!vFound) {
            lRatios.push_back({vW, vH});
          }
        }
      }
    }
  }
  std::sort(lRatios.begin(), lRatios.end(), [](const clip_image_size &pA, const clip_image_size &pB) {
    return pA.vWidth * pA.vHeight < pB.vWidth * pB.vHeight;
  });
  return lRatios;
}

clip_image_size mtmd_image_preprocessor_lfm2::mGetGridLayout(int pHeight, int pWidth) {
  const float cAspectRatio = static_cast<float>(pWidth) / pHeight;
  const auto cRatios = mGetTargetRatios();
  return mFindClosestAspectRatio(cAspectRatio, cRatios, pWidth, pHeight);
}

//
// mtmd_image_preprocessor_idefics3
//

mtmd_image_preproc_out mtmd_image_preprocessor_idefics3::mPreprocess(const clip_image_u8 &pImg) {
  // The refined size has two steps:
  // 1. Resize w/ aspect-ratio preserving such that the longer side is
  //      the preprocessor longest size
  // 2. Resize w/out preserving aspect ratio such that both sides are
  //      multiples of image_size (always rounding up)
  //
  // CITE:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
  const clip_image_size cOriginalSize = pImg.mGetSize();
  const clip_image_size cRefinedSize =
      img_tool::mCalcSizePreservedRatio(cOriginalSize, vHparams.vImageSize, vHparams.vImageLongestEdge);
  // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
  //         __func__, cOriginalSize.vWidth, cOriginalSize.vHeight,
  //         cRefinedSize.vWidth, cRefinedSize.vHeight);

  mtmd_image_preprocessor_llava_uhd::slice_instructions vInstructions;
  vInstructions.vOverviewSize = clip_image_size{vHparams.vImageSize, vHparams.vImageSize};
  vInstructions.vRefinedSize = cRefinedSize;
  vInstructions.vGridSize = clip_image_size{
      static_cast<int>(std::ceil(static_cast<float>(cRefinedSize.vWidth) / vHparams.vImageSize)),
      static_cast<int>(std::ceil(static_cast<float>(cRefinedSize.vHeight) / vHparams.vImageSize)),
  };
  for (int vY = 0; vY < cRefinedSize.vHeight; vY += vHparams.vImageSize) {
    for (int vX = 0; vX < cRefinedSize.vWidth; vX += vHparams.vImageSize) {
      // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, vX, vY);
      vInstructions.lSlices.push_back(mtmd_image_preprocessor_llava_uhd::slice_coordinates{
          /* x    */ vX,
          /* y    */ vY,
          /* size */
          clip_image_size{std::min(vHparams.vImageSize, cRefinedSize.vWidth - vX),
                          std::min(vHparams.vImageSize, cRefinedSize.vHeight - vY)}});
    }
  }
  auto vSliced = mSliceImage(pImg, vInstructions);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.vGridX = vInstructions.vGridSize.vWidth;
  vOutput.vGridY = vInstructions.vGridSize.vHeight;
  return vOutput;
}

//
// mtmd_image_preprocessor_internvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_internvl::mPreprocess(const clip_image_u8 &pImg) {
  GGML_ASSERT(!vHparams.lImageResCandidates.empty());
  const clip_image_size cOriginalSize = pImg.mGetSize();
  auto const cInst = mGetSliceInstructions(cOriginalSize);
  auto vSliced = mSliceImage(pImg, cInst);

  mtmd_image_preproc_out vOutput;
  // InternVL: slices first, then overview
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.vGridX = cInst.vGridSize.vWidth;
  vOutput.vGridY = cInst.vGridSize.vHeight;
  return vOutput;
}

//
// mtmd_image_preprocessor_step3vl
//

void mtmd_image_preprocessor_step3vl::mImgU8ResizeBilinearToF32(const clip_image_u8 &pSrc, clip_image_f32 &pDst,
                                                                int pTargetWidth, int pTargetHeight,
                                                                const float pMean[3], const float pStd[3]) {
  const auto cSrcSize = pSrc.mGetSize();
  if (cSrcSize.vWidth == pTargetWidth && cSrcSize.vHeight == pTargetHeight) {
    pDst.mFromU8(pSrc);
    pDst.mNormalize(pMean, pStd);
    return;
  }

  pDst.mSetSize({pTargetWidth, pTargetHeight}, pSrc.mIsPlaceholder(), false);

  if (pSrc.mIsPlaceholder()) {
    // no-op for placeholder image, just set the size and return
    return;
  }

  const float cScaleX = static_cast<float>(cSrcSize.vWidth) / pTargetWidth;
  const float cScaleY = static_cast<float>(cSrcSize.vHeight) / pTargetHeight;

  std::vector<float> lLocalBuf(3 * pTargetWidth * pTargetHeight);

  for (int vY = 0; vY < pTargetHeight; ++vY) {
    const float cSrcY = (static_cast<float>(vY) + 0.5f) * cScaleY - 0.5f;
    const int cY0Floor = static_cast<int>(std::floor(cSrcY));
    const int cY0 = std::max(0, std::min(cY0Floor, cSrcSize.vHeight - 1));
    const int cY1 = std::max(0, std::min(cY0Floor + 1, cSrcSize.vHeight - 1));
    const float cLy = cSrcY - cY0Floor;

    for (int vX = 0; vX < pTargetWidth; ++vX) {
      const float cSrcX = (static_cast<float>(vX) + 0.5f) * cScaleX - 0.5f;
      const int cX0Floor = static_cast<int>(std::floor(cSrcX));
      const int cX0 = std::max(0, std::min(cX0Floor, cSrcSize.vWidth - 1));
      const int cX1 = std::max(0, std::min(cX0Floor + 1, cSrcSize.vWidth - 1));
      const float cLx = cSrcX - cX0Floor;

      const auto cP00 = pSrc.mGetPixel(cX0, cY0);
      const auto cP01 = pSrc.mGetPixel(cX1, cY0);
      const auto cP10 = pSrc.mGetPixel(cX0, cY1);
      const auto cP11 = pSrc.mGetPixel(cX1, cY1);

      const size_t cIdxDst = 3 * (vY * pTargetWidth + vX);
      for (int vC = 0; vC < 3; ++vC) {
        const float cV00 = (static_cast<float>(cP00[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV01 = (static_cast<float>(cP01[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV10 = (static_cast<float>(cP10[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV11 = (static_cast<float>(cP11[vC]) / 255.0f - pMean[vC]) / pStd[vC];

        const float cTop = cV00 + (cV01 - cV00) * cLx;
        const float cBot = cV10 + (cV11 - cV10) * cLx;
        lLocalBuf[cIdxDst + vC] = cTop + (cBot - cTop) * cLy;
      }
    }
  }
  pDst.mCpyBuf(lLocalBuf);
}

int mtmd_image_preprocessor_step3vl::mDetermineWindowSize(const clip_hparams &pParams, int pLonger, int pShorter) {
  const int cImageSize = pParams.vImageSize;
  const int cCropSize = cDefaultImageCropSize;
  const float cAspectRatio = static_cast<float>(pLonger) / pShorter;

  if (pLonger <= cImageSize) {
    return cAspectRatio > cSmallAspectRatioLimit ? pShorter : 0;
  }

  return cAspectRatio > cWideAspectRatioLimit ? std::min(pShorter, cCropSize) : cCropSize;
}

int mtmd_image_preprocessor_step3vl::mCalcCropExtent(int pLength, int pWindowSize) {
  const float cRatio = static_cast<float>(pLength) / pWindowSize;
  if (cRatio < 1.0f) {
    return pLength;
  }

  const float cDecimal = cRatio - std::floor(cRatio);
  const int cRounded = cDecimal > cCropRoundingThreshold ? static_cast<int>(std::floor(cRatio)) + 1
                                                         : static_cast<int>(std::floor(cRatio));
  return pWindowSize * cRounded;
}

std::vector<int> mtmd_image_preprocessor_step3vl::mCalcGrid(int pLength, int pWindowSize) {
  const int cN = pLength <= pWindowSize
                     ? 1
                     : static_cast<int>(std::ceil(static_cast<float>(pLength - pWindowSize) / pWindowSize + 1.0f));
  std::vector<int> lStarts(cN);

  for (int vI = 0; vI < cN; ++vI) {
    lStarts[vI] = pWindowSize * vI;
  }

  if (cN > 1 && lStarts.back() + pWindowSize > pLength) {
    lStarts.back() = pLength - pWindowSize;
  }

  return lStarts;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::mPrepareImage(const clip_image_u8 &pImg, const clip_hparams &pParams) {
  clip_image_u8 vResized = pImg;
  const auto cImgSize = pImg.mGetSize();
  const float cAspectRatio = cImgSize.vHeight > 0 ? static_cast<float>(cImgSize.vWidth) / cImgSize.vHeight : 1.0f;
  if (std::min(cImgSize.vWidth, cImgSize.vHeight) < 32 &&
      (cAspectRatio > cWideAspectRatioLimit || cAspectRatio < 1.0f / cWideAspectRatioLimit)) {
    const int cSquareSize = std::max(cImgSize.vWidth, cImgSize.vHeight);
    clip_image_u8 vPadded;
    vPadded.mSetSize({cSquareSize, cSquareSize}, pImg.mIsPlaceholder());
    img_tool::mFill(vPadded, {0, 0, 0});
    img_tool::mComposite(vPadded, pImg, 0, 0);
    vResized = std::move(vPadded);
  }

  const int cMaxImageSize = pParams.vImageLongestEdge;
  const auto cResizedSize = vResized.mGetSize();
  if (std::max(cResizedSize.vWidth, cResizedSize.vHeight) > cMaxImageSize) {
    const float cScale = static_cast<float>(cMaxImageSize) / std::max(cResizedSize.vWidth, cResizedSize.vHeight);
    const clip_image_size cNewSize = {
        std::max(1, static_cast<int>(std::floor(cResizedSize.vWidth * cScale))),
        std::max(1, static_cast<int>(std::floor(cResizedSize.vHeight * cScale))),
    };
    clip_image_u8 vScaled;
    img_tool::mResize(vResized, vScaled, cNewSize, cResizeAlgoBilinear, cPadNone);
    vResized = std::move(vScaled);
  }

  return vResized;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::mCropWithBlackPadding(const clip_image_u8 &pImage, int pX, int pY,
                                                                     int pW, int pH) {
  clip_image_u8 vDst;
  vDst.mSetSize({pW, pH}, pImage.mIsPlaceholder());

  if (pImage.mIsPlaceholder()) {
    // a placeholder image (used for counting tokens) only carries its size
    return vDst;
  }

  img_tool::mFill(vDst, {0, 0, 0});

  const auto cImgSize = pImage.mGetSize();
  const int cSrcX0 = std::max(0, pX);
  const int cSrcY0 = std::max(0, pY);
  const int cSrcX1 = std::min(cImgSize.vWidth, pX + pW);
  const int cSrcY1 = std::min(cImgSize.vHeight, pY + pH);

  if (cSrcX0 >= cSrcX1 || cSrcY0 >= cSrcY1) {
    return vDst;
  }

  const int cDstX0 = cSrcX0 - pX;
  const int cDstY0 = cSrcY0 - pY;

  for (int vYy = 0; vYy < cSrcY1 - cSrcY0; ++vYy) {
    for (int vXx = 0; vXx < cSrcX1 - cSrcX0; ++vXx) {
      vDst.mSetPixel(cDstX0 + vXx, cDstY0 + vYy, pImage.mGetPixel(cSrcX0 + vXx, cSrcY0 + vYy));
    }
  }

  return vDst;
}

mtmd_image_preprocessor_step3vl::slice_instructions
mtmd_image_preprocessor_step3vl::mBuildSliceInstructions(const clip_hparams &pParams,
                                                         const clip_image_size &pPreparedSize) {
  slice_instructions vInstructions;
  vInstructions.vOverviewSize = pPreparedSize;

  const int cWindowSize = mDetermineWindowSize(pParams, std::max(pPreparedSize.vWidth, pPreparedSize.vHeight),
                                               std::min(pPreparedSize.vWidth, pPreparedSize.vHeight));
  if (cWindowSize <= 0) {
    vInstructions.vRefinedSize = clip_image_size{0, 0};
    vInstructions.vGridSize = clip_image_size{0, 0};
    return vInstructions;
  }

  const int cCropWidth = mCalcCropExtent(pPreparedSize.vWidth, cWindowSize);
  const int cCropHeight = mCalcCropExtent(pPreparedSize.vHeight, cWindowSize);
  vInstructions.vRefinedSize = clip_image_size{cCropWidth, cCropHeight};

  const auto cXs = mCalcGrid(cCropWidth, cWindowSize);
  const auto cYs = mCalcGrid(cCropHeight, cWindowSize);
  vInstructions.vGridSize = clip_image_size{
      static_cast<int>(cXs.size()),
      static_cast<int>(cYs.size()),
  };

  for (int vY : cYs) {
    for (int vX : cXs) {
      vInstructions.lSlices.push_back(slice_coordinates{
          /* x    */ vX,
          /* y    */ vY,
          /* size */ clip_image_size{cWindowSize, cWindowSize},
      });
    }
  }

  return vInstructions;
}

mtmd_image_preproc_out mtmd_image_preprocessor_step3vl::mPreprocess(const clip_image_u8 &pImg) {
  clip_image_u8 vPrepared = mPrepareImage(pImg, vHparams);
  const auto cInstructions = mBuildSliceInstructions(vHparams, vPrepared.mGetSize());

  mtmd_image_preproc_out vOutput;
  // overview (normalized f32, already includes mean/std)
  mImgU8ResizeBilinearToF32(vPrepared, vOutput.vOverview, vHparams.vImageSize, vHparams.vImageSize, vHparams.aImageMean,
                            vHparams.aImageStd);

  if (cInstructions.lSlices.empty()) {
    vOutput.vGridX = 0;
    vOutput.vGridY = 0;
    return vOutput;
  }

  clip_image_u8 vImgForCrop = vPrepared;
  const auto cPreparedSize = vPrepared.mGetSize();
  if (cInstructions.vRefinedSize.vWidth != cPreparedSize.vWidth ||
      cInstructions.vRefinedSize.vHeight != cPreparedSize.vHeight) {
    clip_image_u8 vRefined;
    img_tool::mResize(vPrepared, vRefined, cInstructions.vRefinedSize, cResizeAlgoBilinear, cPadNone);
    vImgForCrop = std::move(vRefined);
  }

  const int cCropSize = cDefaultImageCropSize;
  for (const auto &vSlice : cInstructions.lSlices) {
    // If the requested patch extends past the source image, pad the out-of-bounds area with black.
    clip_image_u8 vPatch =
        mCropWithBlackPadding(vImgForCrop, vSlice.vX, vSlice.vY, vSlice.vSize.vWidth, vSlice.vSize.vHeight);

    clip_image_f32 vPatchF32;
    mImgU8ResizeBilinearToF32(vPatch, vPatchF32, cCropSize, cCropSize, vHparams.aImageMean, vHparams.aImageStd);
    vOutput.mAppend(vHparams, vPatchF32, false);
  }

  vOutput.vGridX = cInstructions.vGridSize.vWidth;
  vOutput.vGridY = cInstructions.vGridSize.vHeight;

  return vOutput;
}

//
// mtmd_image_preprocessor_youtuvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_youtuvl::mPreprocess(const clip_image_u8 &pImg) {
  const int cPatchSize = vHparams.vPatchSize;     // typically 16
  const int cMergeSize = vHparams.vNMerge;        // typically 2
  const int cAlignSize = cPatchSize * cMergeSize; // 32

  const int cMaxNumPatches = vHparams.vImageMaxPixels > 0 ? vHparams.vImageMaxPixels / (cPatchSize * cPatchSize) : 256;

  // Linear search for optimal scale to fit within max_num_patches
  const auto cImgSize = pImg.mGetSize();
  float vScale = 1.0f;
  int vTargetHeight = cImgSize.vHeight;
  int vTargetWidth = cImgSize.vWidth;

  auto vGetScaledImageSize = [cAlignSize](float pScale, int pSize) -> int {
    float vScaledSize = pSize * pScale;
    // Round up to nearest multiple of align_size
    int vAligned = static_cast<int>(std::ceil(vScaledSize / cAlignSize)) * cAlignSize;
    // Ensure at least one patch
    return std::max(cAlignSize, vAligned);
  };

  // Linear search with 0.02 step size
  while (vScale > 0.0f) {
    vTargetHeight = vGetScaledImageSize(vScale, cImgSize.vHeight);
    vTargetWidth = vGetScaledImageSize(vScale, cImgSize.vWidth);

    int vNumPatchesH = vTargetHeight / cPatchSize;
    int vNumPatchesW = vTargetWidth / cPatchSize;
    int vNumPatches = vNumPatchesH * vNumPatchesW;

    if (vNumPatches > cMaxNumPatches) {
      vScale -= 0.02f;
    } else {
      break;
    }
  }

  clip_image_size vNewSize = {vTargetWidth, vTargetHeight};

  // Resize the image
  clip_image_u8 vResized;
  img_tool::mResize(pImg, vResized, vNewSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResized, true);
  return vOutput;
}

mtmd_image_preproc_out mtmd_image_preprocessor_granite::mPreprocess(const clip_image_u8 &pImg) {
  auto vOutput = mtmd_image_preprocessor_llava_uhd::mPreprocess(pImg);
  if (vOutput.lEntries.size() == 0) {
    // Single-tile (overview only): append one newline row.
    vOutput.vOverview.vAddNewline = true;
  } else {
    // Multi-tile: overview gets no newline, grid tiles get one.
    vOutput.vOverview.vAddNewline = false;
    for (size_t vI = 0; vI < vOutput.lEntries.size(); ++vI) {
      vOutput.lEntries[vI].vAddNewline = true;
    }
  }
  return vOutput;
}
