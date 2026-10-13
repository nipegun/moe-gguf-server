#include "mtmd-audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>

// some of the code here is copied from whisper.cpp

constexpr bool cDebug = false;

void mtmd_audio_cache::mFillSinCosTable(uint32_t pN) {
  lSinVals.resize(pN);
  lCosVals.resize(pN);
  for (uint32_t vI = 0; vI < pN; vI++) {
    double vTheta = (2 * M_PI * vI) / pN;
    lSinVals[vI] = sinf(vTheta);
    lCosVals[vI] = cosf(vTheta);
  }
}

void mtmd_audio_cache::mFillHannWindow(uint32_t pLength, bool pPeriodic) {
  lHannWindow.resize(pLength);
  int vOffset = pPeriodic ? 0 : -1;
  for (uint32_t vI = 0; vI < pLength; vI++) {
    lHannWindow[vI] = 0.5 * (1.0 - cosf((2.0 * M_PI * vI) / (pLength + vOffset)));
  }
}

void mtmd_audio_cache::mFillMelFilterbankMatrix(int64_t pNMel, int64_t pNFft, int pSampleRate, float pFmin, float pFmax,
                                                bool pSlaneyAreaNorm, float pScale, bool pUseHtk) {
  GGML_ASSERT(pNMel > 0 && pNFft > 1);
  if (pFmax <= 0.0f) {
    pFmax = 0.5f * pSampleRate;
  }

  std::function<double(double)> vHzToMel;
  std::function<double(double)> vMelToHz;

  if (pUseHtk) {
    vHzToMel = [](const double pFHz) -> double { return 2595.0 * log10(1.0 + pFHz / 700.0); };
    vMelToHz = [](const double pM) -> double { return 700.0 * (pow(10.0, pM / 2595.0) - 1.0); };
  } else {
    // Slaney scale (matches librosa default)
    const double cMinLogHz = 1000.0;
    const double cLinSlope = 3 / 200.;
    const double cMinLogMel = cMinLogHz * cLinSlope;
    const double cLogStep = log(6.4) / 27.0;
    vHzToMel = [cMinLogHz, cLinSlope, cLogStep, cMinLogMel](const double pFHz) -> double {
      return (pFHz < cMinLogHz) ? pFHz * cLinSlope : cMinLogMel + log(pFHz / cMinLogHz) / cLogStep;
    };
    vMelToHz = [cMinLogHz, cLinSlope, cLogStep, cMinLogMel](const double pM) -> double {
      return (pM < cMinLogMel) ? pM / cLinSlope : cMinLogHz * exp((pM - cMinLogMel) * cLogStep);
    };
  }

  // infer N_fft from n_fft_bins
  const double cBinHzStep = double(pSampleRate) / double(pNFft);

  // mel grid: n_mel + 2 edges
  const double cMLo = vHzToMel(pFmin);
  const double cMHi = vHzToMel(pFmax);
  std::vector<double> lMelPts(pNMel + 2);
  for (int vI = 0; vI < pNMel + 2; ++vI) {
    lMelPts[vI] = cMLo + (cMHi - cMLo) * (double(vI) / (pNMel + 1));
  }

  // convert to Hz
  std::vector<double> lHzPts(pNMel + 2);
  for (int vI = 0; vI < pNMel + 2; ++vI) {
    lHzPts[vI] = vMelToHz(lMelPts[vI]);
  }

  const int64_t cNFftBins = pNFft / 2 + 1;

  // Validate allocation size
  if ((size_t)pNMel * (size_t)cNFftBins > SIZE_MAX) {
    GGML_ASSERT(false && "mel filterbank allocation too large");
  }

  // filterbank
  std::vector<float> lOut((size_t)pNMel * (size_t)cNFftBins, 0);
  for (int64_t vM = 0; vM < pNMel; ++vM) {
    const double cFLeft = lHzPts[vM];
    const double cFCenter = lHzPts[vM + 1];
    const double cFRight = lHzPts[vM + 2];

    const double cDenomL = std::max(1e-30, cFCenter - cFLeft);
    const double cDenomR = std::max(1e-30, cFRight - cFCenter);
    const double cEnorm = pSlaneyAreaNorm ? (2.0 / std::max(1e-30, cFRight - cFLeft)) : 1.0;

    for (int vK = 0; vK < cNFftBins; ++vK) {
      const double cF = vK * cBinHzStep;
      double vW = 0.0;
      if (cF >= cFLeft && cF <= cFCenter) {
        vW = (cF - cFLeft) / cDenomL;
      } else if (cF > cFCenter && cF <= cFRight) {
        vW = (cFRight - cF) / cDenomR;
      }
      lOut[size_t(vM) * size_t(cNFftBins) + size_t(vK)] = float(vW * cEnorm * pScale);
    }
  }

  vFilters.vNMel = pNMel;
  vFilters.vNFft = pNFft;
  vFilters.lData = std::move(lOut);

  if (cDebug) { // debug
    for (size_t vI = 0; vI < vFilters.lData.size(); ++vI) {
      if (vFilters.lData[vI] != 0.0f) {
        printf("filters[%zu] = %f\n", vI, vFilters.lData[vI] * 1000.0f);
      }
    }
  }
}

// Unified DFT implementation for both forward and inverse transforms
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued (stride 1), avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fDftImpl(const mtmd_audio_cache &pCache, const float *pIn, int pN, float *pOut) {
  const int cNSinCosVals = pCache.lSinVals.size();
  const int cSinCosStep = cNSinCosVals / pN;

  constexpr float cSign = Inverse ? 1.0f : -1.0f;
  const float cScale = Inverse ? (1.0f / pN) : 1.0f;

  for (int vK = 0; vK < pN; vK++) {
    float vRe = 0;
    float vIm = 0;

    for (int vN = 0; vN < pN; vN++) {
      int vIdx = (vK * vN * cSinCosStep) % cNSinCosVals;
      float vCosVal = pCache.lCosVals[vIdx];
      float vSinVal = pCache.lSinVals[vIdx];

      if constexpr (RealInput) {
        // Real input: in_im = 0, simplifies to:
        // re += in_re * cos_val
        // im += sign * in_re * sin_val
        float vInRe = pIn[vN];
        vRe += vInRe * vCosVal;
        vIm += cSign * vInRe * vSinVal;
      } else {
        float vInRe = pIn[vN * 2 + 0];
        float vInIm = pIn[vN * 2 + 1];
        // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
        vRe += vInRe * vCosVal - cSign * vInIm * vSinVal;
        vIm += cSign * vInRe * vSinVal + vInIm * vCosVal;
      }
    }

    pOut[vK * 2 + 0] = vRe * cScale;
    pOut[vK * 2 + 1] = vIm * cScale;
  }
}

// Cooley-Tukey FFT/IFFT unified implementation
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each level
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fFftImpl(const mtmd_audio_cache &pCache, float *pIn, int pN, float *pOut) {
  GGML_ASSERT(pN > 0);
  const int cNSinCosVals = pCache.lSinVals.size();

  if (pN == 1) {
    pOut[0] = pIn[0];
    if constexpr (RealInput) {
      pOut[1] = 0.0f;
    } else {
      pOut[1] = pIn[1];
    }
    return;
  }

  const int cHalfN = pN / 2;
  if (pN - cHalfN * 2 == 1) {
    // Odd N: fall back to DFT
    fDftImpl<Inverse, RealInput>(pCache, pIn, pN, pOut);
    return;
  }

  // Split into even and odd
  if constexpr (RealInput) {
    // Real input: stride is 1, copy only real values
    float *vEven = pIn + pN;
    for (int vI = 0; vI < cHalfN; ++vI) {
      vEven[vI] = pIn[2 * vI];
    }
    float *vEvenFft = pOut + 2 * pN;
    fFftImpl<Inverse, true>(pCache, vEven, cHalfN, vEvenFft);

    float *vOdd = vEven;
    for (int vI = 0; vI < cHalfN; ++vI) {
      vOdd[vI] = pIn[2 * vI + 1];
    }
    float *vOddFft = vEvenFft + pN;
    fFftImpl<Inverse, true>(pCache, vOdd, cHalfN, vOddFft);
  } else {
    // Complex input: stride is 2, copy complex pairs
    float *vEven = pIn + pN * 2;
    for (int vI = 0; vI < cHalfN; ++vI) {
      vEven[vI * 2 + 0] = pIn[2 * vI * 2 + 0];
      vEven[vI * 2 + 1] = pIn[2 * vI * 2 + 1];
    }
    float *vEvenFft = pOut + 2 * pN;
    fFftImpl<Inverse, false>(pCache, vEven, cHalfN, vEvenFft);

    float *vOdd = vEven;
    for (int vI = 0; vI < cHalfN; ++vI) {
      vOdd[vI * 2 + 0] = pIn[(2 * vI + 1) * 2 + 0];
      vOdd[vI * 2 + 1] = pIn[(2 * vI + 1) * 2 + 1];
    }
    float *vOddFft = vEvenFft + pN;
    fFftImpl<Inverse, false>(pCache, vOdd, cHalfN, vOddFft);
  }

  float *vEvenFft = pOut + 2 * pN;
  float *vOddFft = vEvenFft + pN;

  const int cSinCosStep = cNSinCosVals / pN;

  constexpr float cSign = Inverse ? 1.0f : -1.0f;
  constexpr float cScale = Inverse ? 0.5f : 1.0f;

  for (int vK = 0; vK < cHalfN; vK++) {
    int vIdx = vK * cSinCosStep; // t = 2*M_PI*k/N
    float vRe = pCache.lCosVals[vIdx];
    float vIm = cSign * pCache.lSinVals[vIdx];

    float vReOdd = vOddFft[2 * vK + 0];
    float vImOdd = vOddFft[2 * vK + 1];

    pOut[2 * vK + 0] = cScale * (vEvenFft[2 * vK + 0] + vRe * vReOdd - vIm * vImOdd);
    pOut[2 * vK + 1] = cScale * (vEvenFft[2 * vK + 1] + vRe * vImOdd + vIm * vReOdd);

    pOut[2 * (vK + cHalfN) + 0] = cScale * (vEvenFft[2 * vK + 0] - vRe * vReOdd + vIm * vImOdd);
    pOut[2 * (vK + cHalfN) + 1] = cScale * (vEvenFft[2 * vK + 1] - vRe * vImOdd - vIm * vReOdd);
  }
}

// Forward FFT for real input (used by mel spectrogram)
static void fFft(const mtmd_audio_cache &pCache, float *pIn, int pN, float *pOut) {
  fFftImpl<false, true>(pCache, pIn, pN, pOut);
}

// Inverse FFT for complex input
static void fIfft(const mtmd_audio_cache &pCache, float *pIn, int pN, float *pOut) {
  fFftImpl<true, false>(pCache, pIn, pN, pOut);
}

struct filter_params {
  int64_t vNMel;
  int64_t vNFftBins;
  int32_t vHannWindowSize;
  int32_t vHopLength;
  int32_t vSampleRate;
  bool vNoPadding = false;
  bool vCenterPadding = false;
  float vPreemph = 0.f;
  bool vUseNaturalLog = false;
  bool vNormPerFeature = false;
  bool vUseMagnitude = false; // |X| instead of |X|^2
  float vMelFloor = 5.960464477539063e-08f;
};

static void fLogMelSpectrogramWorkerThread(int pIth, const float *pHann, const std::vector<float> &pSamples,
                                           int pNSamples, int pFrameSize, int pFrameStep, int pNThreads,
                                           const filter_params &pParams, const mtmd_audio_cache &pCache,
                                           mtmd_audio_mel &pOut) {
  std::vector<float> lFftIn(pFrameSize * 2, 0.0);
  std::vector<float> lFftOut(pFrameSize * 2 * 2 * 2);

  int64_t vNFftBins = pParams.vNFftBins;
  int64_t vI = pIth;

  const auto &vFilters = pCache.vFilters;

  // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
  GGML_ASSERT(vNFftBins == 1 + (pFrameSize / 2));
  GGML_ASSERT(pCache.lSinVals.size() == pCache.lCosVals.size());
  // calculate FFT only when fft_in are not all zero
  for (; vI < std::min((int64_t)(pNSamples / pFrameStep + 1), pOut.vNLen); vI += pNThreads) {
    const int64_t cOffset = vI * pFrameStep;

    // apply Hann window (~10% faster)
    const int cValidLen = std::min(pFrameSize, std::max(0, pNSamples - (int)cOffset));
    for (int vJ = 0; vJ < cValidLen; vJ++) {
      lFftIn[vJ] = pHann[vJ] * pSamples[cOffset + vJ];
    }

    // fill the rest with zeros
    if (cValidLen < pFrameSize) {
      std::fill(lFftIn.begin() + cValidLen, lFftIn.end(), 0.0);
    }

    // FFT
    fFft(pCache, lFftIn.data(), pFrameSize, lFftOut.data());

    // Calculate modulus^2 (power) or modulus (magnitude)
    for (int vJ = 0; vJ < vNFftBins; vJ++) {
      float vPower = (lFftOut[2 * vJ + 0] * lFftOut[2 * vJ + 0] + lFftOut[2 * vJ + 1] * lFftOut[2 * vJ + 1]);
      lFftOut[vJ] = pParams.vUseMagnitude ? sqrtf(vPower) : vPower;
    }

    // mel spectrogram
    for (int64_t vJ = 0; vJ < pOut.vNMel; vJ++) {
      double vSum = 0.0;
      // unroll loop (suggested by GH user @lunixbochs)
      int vK = 0;
      for (vK = 0; vK < vNFftBins - 3; vK += 4) {
        size_t vIdx = size_t(vJ) * size_t(vNFftBins) + size_t(vK);
        vSum += lFftOut[vK + 0] * vFilters.lData[vIdx + 0] + lFftOut[vK + 1] * vFilters.lData[vIdx + 1] +
                lFftOut[vK + 2] * vFilters.lData[vIdx + 2] + lFftOut[vK + 3] * vFilters.lData[vIdx + 3];
      }
      // handle n_fft remainder
      for (; vK < vNFftBins; vK++) {
        vSum += lFftOut[vK] * vFilters.lData[(size_t)vJ * vNFftBins + vK];
      }
      vSum = std::max(vSum, (double)pParams.vMelFloor);
      vSum = pParams.vUseNaturalLog ? log(vSum) : log10(vSum);
      pOut.lData[(size_t)vJ * pOut.vNLen + vI] = vSum;
    }
  }

  // Otherwise fft_out are all zero
  double vSum = pParams.vUseNaturalLog ? log(1e-10) : log10(1e-10);
  for (; vI < pOut.vNLen; vI += pNThreads) {
    for (int64_t vJ = 0; vJ < pOut.vNMel; vJ++) {
      pOut.lData[(size_t)vJ * pOut.vNLen + vI] = vSum;
    }
  }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool fLogMelSpectrogram(const float *pSamples, const int pNSamplesIn, const int pNThreads,
                               const filter_params &pParams, const mtmd_audio_cache &pCache, mtmd_audio_mel &pOut) {
  // const int64_t cTStartUs = ggml_time_us();

  pOut.vNLenOrg = pNSamplesIn;
  int vNSamples = pNSamplesIn;

  // Hann window
  const float *vHann = pCache.lHannWindow.data();
  const int cFrameSize = (pParams.vNFftBins - 1) * 2;
  const int cFrameStep = pParams.vHopLength;

  // Padding
  std::vector<float> lSamplesPadded;
  if (pParams.vNoPadding) {
    // no padding, use samples as-is
    lSamplesPadded = std::vector<float>(pSamples, pSamples + vNSamples);
    pSamples = lSamplesPadded.data();
    vNSamples = lSamplesPadded.size();
  } else if (pParams.vCenterPadding) {
    const auto cPadAmount = cFrameSize / 2;
    lSamplesPadded = std::vector<float>(vNSamples + 2 * cPadAmount, 0);
    std::copy(pSamples, pSamples + vNSamples, lSamplesPadded.data() + cPadAmount);
    pSamples = lSamplesPadded.data();
    vNSamples = lSamplesPadded.size();
  } else {
    // existing padding logic
    int64_t vStage1Pad = pParams.vSampleRate * 30;
    int64_t vStage2Pad = cFrameSize / 2;
    lSamplesPadded.resize(vNSamples + vStage1Pad + vStage2Pad * 2);
    std::copy(pSamples, pSamples + vNSamples, lSamplesPadded.begin() + vStage2Pad);
    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(lSamplesPadded.begin() + vNSamples + vStage2Pad,
              lSamplesPadded.begin() + vNSamples + vStage1Pad + 2 * vStage2Pad, 0);
    // reflective pad 200 samples at the beginning of audio
    if (vNSamples < vStage2Pad + 1) {
      // TODO: Handle short audio differently or return error
      return false;
    }
    std::reverse_copy(pSamples + 1, pSamples + 1 + vStage2Pad, lSamplesPadded.begin());

    // expose the padded buffer to downstream FFT and to pOut.vNLen computation
    // mirrors the no_padding and center_padding branches above
    pSamples = lSamplesPadded.data();
    vNSamples = lSamplesPadded.size();
  }

  // preemphasis
  if (pParams.vPreemph) {
    const int cPadAmount = cFrameSize / 2;
    const float cPreemph = 0.97f;
    float vPrev = lSamplesPadded[cPadAmount];
    for (int vI = cPadAmount + 1; vI + cPadAmount < vNSamples; ++vI) {
      float vCur = lSamplesPadded[vI];
      lSamplesPadded[vI] = vCur - cPreemph * vPrev;
      vPrev = vCur;
    }
  }

  // pad hann window if it's smaller than frame_size
  // TODO: probably unnecessary here? (or better doing it in g_cache?)
  std::vector<float> lHannWindowPadded;
  if (pParams.vHannWindowSize < cFrameSize) {
    lHannWindowPadded.resize(cFrameSize);
    const int cPadding = (cFrameSize - pParams.vHannWindowSize) / 2;
    std::copy(vHann, vHann + pParams.vHannWindowSize, &lHannWindowPadded[cPadding]);
    vHann = lHannWindowPadded.data();
  }

  GGML_ASSERT(pParams.vNFftBins > 0);
  GGML_ASSERT(pParams.vHopLength > 0);
  pOut.vNMel = pParams.vNMel;
  pOut.vNLen = (vNSamples - cFrameSize) / cFrameStep + 1;
  // Validate dimensions before allocation to prevent integer overflow
  if (pOut.vNMel <= 0 || pOut.vNLen <= 0) {
    LOG_ERR("%s: invalid mel dimensions n_mel=%lld n_len=%lld\n", __func__, (long long)pOut.vNMel,
            (long long)pOut.vNLen);
    return false;
  }
  const size_t cTotalSize = (size_t)pOut.vNMel * (size_t)pOut.vNLen;
  if (cTotalSize > SIZE_MAX / sizeof(float)) {
    LOG_ERR("%s: size overflow: n_mel=%lld n_len=%lld\n", __func__, (long long)pOut.vNMel, (long long)pOut.vNLen);
    return false;
  }
  if (vNSamples < cFrameSize) {
    LOG_ERR("%s: not enough samples after padding\n", __func__);
    return false;
  }
  pOut.lData.resize(cTotalSize);

  {
    std::vector<std::thread> lWorkers(pNThreads - 1);
    for (int vIw = 0; vIw < pNThreads - 1; ++vIw) {
      lWorkers[vIw] =
          std::thread(fLogMelSpectrogramWorkerThread, vIw + 1, vHann, std::cref(lSamplesPadded), vNSamples, cFrameSize,
                      cFrameStep, pNThreads, std::cref(pParams), std::cref(pCache), std::ref(pOut));
    }

    // main thread
    fLogMelSpectrogramWorkerThread(0, vHann, lSamplesPadded, vNSamples, cFrameSize, cFrameStep, pNThreads, pParams,
                                   pCache, pOut);
    for (int vIw = 0; vIw < pNThreads - 1; ++vIw) {
      lWorkers[vIw].join();
    }
  }

  const int64_t cEffectiveNLen = pNSamplesIn / cFrameStep;
  if (pParams.vNormPerFeature) {
    GGML_ASSERT(cEffectiveNLen > 1);
    for (int64_t vI = 0; vI < pOut.vNMel; vI++) {
      double vMean = 0;
      for (int64_t vJ = 0; vJ < cEffectiveNLen; ++vJ) {
        vMean += pOut.lData[(size_t)vI * pOut.vNLen + vJ];
      }
      vMean /= cEffectiveNLen;

      double vVar = 0.0;
      for (int64_t vJ = 0; vJ < cEffectiveNLen; ++vJ) {
        const double cValue = pOut.lData[(size_t)vI * pOut.vNLen + vJ] - vMean;
        vVar += cValue * cValue;
      }
      vVar /= cEffectiveNLen - 1; // unbiased
      const double cMstd = std::sqrt(vVar + 1e-5);

      for (int64_t vJ = 0; vJ < cEffectiveNLen; ++vJ) {
        auto &vValue = pOut.lData[(size_t)vI * pOut.vNLen + vJ];
        vValue = (vValue - vMean) / cMstd;
      }

      // pad the rest with zeros
      for (int64_t vJ = cEffectiveNLen; vJ < pOut.vNLen; ++vJ) {
        pOut.lData[(size_t)vI * pOut.vNLen + vJ] = 0.0;
      }
    }
  } else if (!pParams.vNoPadding) {
    // Whisper-style clamping and normalization
    double vMmax = -1e20;
    const size_t cMelSize = (size_t)pOut.vNMel * (size_t)pOut.vNLen;
    for (size_t vI = 0; vI < cMelSize; vI++) {
      if (pOut.lData[vI] > vMmax) {
        vMmax = pOut.lData[vI];
      }
    }

    vMmax -= 8.0;

    for (size_t vI = 0; vI < cMelSize; vI++) {
      if (pOut.lData[vI] < vMmax) {
        pOut.lData[vI] = vMmax;
      }
      pOut.lData[vI] = (pOut.lData[vI] + 4.0) / 4.0;
    }
  }

  // Dump log_mel_spectrogram
  if (cDebug) {
    std::ofstream vOutFile("log_mel_spectrogram.json");
    vOutFile << "[";
    for (uint64_t vI = 0; vI < pOut.lData.size() - 1; vI++) {
      vOutFile << pOut.lData[vI] << ", ";
    }
    vOutFile << pOut.lData[pOut.lData.size() - 1] << "]";
    vOutFile.close();
  }

  return true;
}

//
// mtmd_audio_preprocessor_whisper
//

void mtmd_audio_preprocessor_whisper::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);
  vCache.mFillHannWindow(vHparams.vAudioWindowLen, true);
  vCache.mFillMelFilterbankMatrix(vHparams.vNMelBins, vHparams.vAudioNFft, vHparams.vAudioSampleRate);
}

bool mtmd_audio_preprocessor_whisper::mPreprocess(const float *pSamples, size_t pNSamples,
                                                  std::vector<mtmd_audio_mel> &pOutput) {
  if (pNSamples == 0) {
    // empty audio
    return false;
  }

  std::vector<float> lSmpl;
  // if input is too short, pad with zeros
  // this is to avoid potential issues with stage1/2 padding in fLogMelSpectrogram
  // TODO: maybe handle this better
  size_t vMinSamples = (size_t)vHparams.vAudioSampleRate * (vHparams.vAudioChunkLen + 1); // +1 second margin
  if (pNSamples < vMinSamples) {
    lSmpl.resize(vMinSamples, 0.0f);
    std::memcpy(lSmpl.data(), pSamples, pNSamples * sizeof(float));
    pSamples = lSmpl.data();
    pNSamples = lSmpl.size();
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins;
  vParams.vNFftBins = 1 + (vHparams.vAudioNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;
  vParams.vCenterPadding = false;
  vParams.vPreemph = 0.0f; // disabled
  vParams.vUseNaturalLog = false;
  vParams.vNormPerFeature = false;

  // make sure the cache is initialized
  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  mtmd_audio_mel vOutFull;
  bool vOk = fLogMelSpectrogram(pSamples, pNSamples,
                                4, // n_threads
                                vParams, vCache, vOutFull);
  if (!vOk) {
    return false;
  }

  // because the cgraph in clip.cpp only accepts 3000 frames each, we need to split the mel
  // we always expect the mel to have 3000 silent frames at the end
  if (cDebug) {
    printf("output: n_mel = %d, n_len = %d\n", (int)vOutFull.vNMel, (int)vOutFull.vNLen);
  }
  const size_t cFramesPerChunk = 3000;
  GGML_ASSERT((size_t)vOutFull.vNLen > cFramesPerChunk);
  for (size_t vOff = 0; vOff < (size_t)vOutFull.vNLen; vOff += cFramesPerChunk) {
    int64_t vNLen = std::min((int64_t)cFramesPerChunk, vOutFull.vNLen - (int64_t)vOff);
    if (vNLen < (int64_t)cFramesPerChunk) {
      break; // last incomplete chunk will always be a padded chunk, safe to ignore
    }

    mtmd_audio_mel vOutChunk;
    vOutChunk.vNLen = vNLen;
    vOutChunk.vNMel = vOutFull.vNMel;
    vOutChunk.vNLenOrg = vOutFull.vNMel; // unused
    vOutChunk.lData.reserve((size_t)vOutChunk.vNMel * (size_t)vOutChunk.vNLen);

    for (int64_t vI = 0; vI < vOutFull.vNMel; vI++) {
      auto vSrc = vOutFull.lData.begin() + (size_t)vI * vOutFull.vNLen + vOff;
      vOutChunk.lData.insert(vOutChunk.lData.end(), vSrc, vSrc + cFramesPerChunk);
    }

    pOutput.push_back(std::move(vOutChunk));
  }

  return true;
}

//
// mtmd_audio_preprocessor_qwen3a
//
// Matches the Python WhisperFeatureExtractor called with truncation=False:
//   - reflection padding of n_fft/2 samples at each end (center=True)
//   - Whisper-style log10 + (max-8)/4 normalization applied to full audio
//   - output split into ≤30s (3000 mel frames) windows, each padded to a
//     multiple of 200 frames (n_window * 2) for the cgraph batch view
//

void mtmd_audio_preprocessor_qwen3a::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);
  vCache.mFillHannWindow(vHparams.vAudioWindowLen, true);
  vCache.mFillMelFilterbankMatrix(vHparams.vNMelBins, vHparams.vAudioNFft, vHparams.vAudioSampleRate);
}

bool mtmd_audio_preprocessor_qwen3a::mPreprocess(const float *pSamples, size_t pNSamples,
                                                 std::vector<mtmd_audio_mel> &pOutput) {
  if (pNSamples == 0) {
    return false;
  }

  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  // Reflection-pad n_fft/2 samples at each end, matching WhisperFeatureExtractor center=True
  const int cPad = vHparams.vAudioNFft / 2; // = 200

  std::vector<float> lPadded(pNSamples + 2 * cPad, 0.0f);
  // Reflect start: padded[0..pad-1] = samples[pad..1] (reversed)
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = cPad - vI; // samples[pad], samples[pad-1], ..., samples[1]
    lPadded[vI] = (vSrc < (int)pNSamples) ? pSamples[vSrc] : 0.0f;
  }
  std::copy(pSamples, pSamples + pNSamples, lPadded.begin() + cPad);
  // Reflect end: padded[n+pad..n+2*pad-1] = samples[n-2..n-pad-1] (reversed)
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = (int)pNSamples - 2 - vI; // samples[n-2], samples[n-3], ...
    lPadded[pNSamples + cPad + vI] = (vSrc >= 0) ? pSamples[vSrc] : 0.0f;
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins;
  vParams.vNFftBins = 1 + (vHparams.vAudioNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;
  vParams.vNoPadding = true;      // reflection padding already applied above
  vParams.vUseNaturalLog = false; // log10

  mtmd_audio_mel vMelFull;
  bool vOk = fLogMelSpectrogram(lPadded.data(), (int)lPadded.size(), 4, vParams, vCache, vMelFull);
  if (!vOk) {
    return false;
  }

  // Whisper-style normalization: clamp to (max - 8), scale to [-1, 1]
  {
    double vMmax = -1e20;
    for (float vV : vMelFull.lData) {
      if (vV > vMmax)
        vMmax = vV;
    }
    vMmax -= 8.0;
    for (float &vV : vMelFull.lData) {
      vV = (std::max((double)vV, vMmax) + 4.0) / 4.0;
    }
  }

  // The effective frame count: center-padded STFT gives ~n_samples/hop_length frames.
  // We take min(vMelFull.vNLen, n_samples/hop + 1) to avoid including excess frames.
  const int64_t cNEff = std::min(vMelFull.vNLen, (int64_t)(pNSamples / vHparams.vAudioHopLen) + 1);

  // Split into inference windows matching n_window_infer=800 from model config.
  // Each window is padded to the next multiple of chunk_size for the cgraph.
  // The mtmd caller loops over output entries, so long audio is handled automatically.
  const int cChunkSize = 100;  // conv sub-chunk size (n_window * 2, n_window=50)
  const int cWindowSize = 800; // mel frames per forward pass (n_window_infer=800)

  for (int64_t vOff = 0; vOff < cNEff; vOff += cWindowSize) {
    const int64_t cWinEff = std::min((int64_t)cWindowSize, cNEff - vOff);
    const int64_t cNChunks = (cWinEff + cChunkSize - 1) / cChunkSize;
    const int64_t cNPadded = cNChunks * cChunkSize;

    mtmd_audio_mel vOut;
    vOut.vNMel = vMelFull.vNMel;
    vOut.vNLen = cNPadded;
    vOut.vNLenOrg = cWinEff;
    vOut.lData.assign((size_t)vOut.vNMel * (size_t)vOut.vNLen, 0.0f);
    for (int64_t vM = 0; vM < vOut.vNMel; vM++) {
      const int64_t cCopyLen = std::min((int64_t)cWinEff, vMelFull.vNLen - vOff);
      if (cCopyLen > 0) {
        std::copy(vMelFull.lData.begin() + (size_t)vM * vMelFull.vNLen + vOff,
                  vMelFull.lData.begin() + (size_t)vM * vMelFull.vNLen + vOff + cCopyLen,
                  vOut.lData.begin() + (size_t)vM * vOut.vNLen);
      }
    }
    pOutput.push_back(std::move(vOut));
  }
  return true;
}

//
// mtmd_audio_preprocessor_mimo_audio
//
// Matches torchaudio.transforms.MelSpectrogram(power=1.0, center=True) followed by
// log(clip(spec, min=1e-7)): HTK mel scale, no Slaney area norm, magnitude (not power)
// spectrogram, natural log, reflect-padded by n_fft/2 on each side.
//

void mtmd_audio_preprocessor_mimo_audio::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);
  vCache.mFillHannWindow(vHparams.vAudioWindowLen, true);
  vCache.mFillMelFilterbankMatrix(vHparams.vNMelBins, vHparams.vAudioNFft, vHparams.vAudioSampleRate, 0.0f,
                                  vHparams.vAudioSampleRate / 2.0f,
                                  /*slaney_area_norm=*/false,
                                  /*scale=*/1.0f,
                                  /*use_htk=*/true);
}

bool mtmd_audio_preprocessor_mimo_audio::mPreprocess(const float *pSamples, size_t pNSamples,
                                                     std::vector<mtmd_audio_mel> &pOutput) {
  if (pNSamples == 0) {
    return false;
  }

  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  const int cPad = vHparams.vAudioNFft / 2;

  std::vector<float> lPadded(pNSamples + 2 * cPad, 0.0f);
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = cPad - vI;
    lPadded[vI] = (vSrc < (int)pNSamples) ? pSamples[vSrc] : 0.0f;
  }
  std::copy(pSamples, pSamples + pNSamples, lPadded.begin() + cPad);
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = (int)pNSamples - 2 - vI;
    lPadded[pNSamples + cPad + vI] = (vSrc >= 0) ? pSamples[vSrc] : 0.0f;
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins;
  vParams.vNFftBins = 1 + (vHparams.vAudioNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;
  vParams.vNoPadding = true; // reflect padding already applied above
  vParams.vUseNaturalLog = true;
  vParams.vUseMagnitude = true;
  vParams.vMelFloor = 1e-7f;
  vParams.vNormPerFeature = false;

  mtmd_audio_mel vOut;
  bool vOk = fLogMelSpectrogram(lPadded.data(), (int)lPadded.size(), 4, vParams, vCache, vOut);
  if (!vOk) {
    return false;
  }

  pOutput.push_back(std::move(vOut));
  return true;
}

//
// mtmd_audio_preprocessor_conformer
//

void mtmd_audio_preprocessor_conformer::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);
  vCache.mFillHannWindow(vHparams.vAudioWindowLen, true);
  vCache.mFillMelFilterbankMatrix(vHparams.vNMelBins, vHparams.vAudioNFft, vHparams.vAudioSampleRate);
}

bool mtmd_audio_preprocessor_conformer::mPreprocess(const float *pSamples, size_t pNSamples,
                                                    std::vector<mtmd_audio_mel> &pOutput) {
  // empty audio
  if (pNSamples == 0) {
    return false;
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins;
  vParams.vNFftBins = 1 + (vHparams.vAudioNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;
  vParams.vCenterPadding = true;
  vParams.vPreemph = 0.97f;
  vParams.vUseNaturalLog = true;
  vParams.vNormPerFeature = true;

  // make sure the cache is initialized
  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  mtmd_audio_mel vOutFull;
  bool vOk = fLogMelSpectrogram(pSamples, pNSamples,
                                4, // n_threads
                                vParams, vCache, vOutFull);
  if (!vOk) {
    return false;
  }

  pOutput.push_back(std::move(vOutFull));
  return true;
}

//
// mtmd_audio_preprocessor_granite_speech
//

void mtmd_audio_preprocessor_granite_speech::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);
  vCache.mFillHannWindow(vHparams.vAudioWindowLen, true);
  vCache.mFillMelFilterbankMatrix(vHparams.vNMelBins / 2, vHparams.vAudioNFft, vHparams.vAudioSampleRate, 0.0f, -1.0f,
                                  false, 1.0f, true);
}

bool mtmd_audio_preprocessor_granite_speech::mPreprocess(const float *pSamples, size_t pNSamples,
                                                         std::vector<mtmd_audio_mel> &pOutput) {
  if (pNSamples == 0) {
    return false;
  }

  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  const int cNFft = vHparams.vAudioNFft;
  const int cPad = cNFft / 2;

  // reflect padding
  const int cNPadded = (int)pNSamples + 2 * cPad;
  std::vector<float> lPadded(cNPadded, 0.0f);
  std::copy(pSamples, pSamples + pNSamples, lPadded.data() + cPad);
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = vI + 1;
    if (vSrc >= (int)pNSamples) {
      vSrc = (int)pNSamples - 1;
    }
    lPadded[cPad - 1 - vI] = pSamples[vSrc];
  }
  for (int vI = 0; vI < cPad; vI++) {
    int vSrc = (int)pNSamples - 2 - vI;
    if (vSrc < 0) {
      vSrc = 0;
    }
    lPadded[cPad + (int)pNSamples + vI] = pSamples[vSrc];
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins / 2;
  vParams.vNFftBins = 1 + (cNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;
  vParams.vNoPadding = true;
  vParams.vCenterPadding = false;
  vParams.vPreemph = 0.0f;
  vParams.vUseNaturalLog = false;
  vParams.vNormPerFeature = false;
  vParams.vMelFloor = 1e-10f;

  mtmd_audio_mel vMel;
  if (!fLogMelSpectrogram(lPadded.data(), cNPadded, 4, vParams, vCache, vMel)) {
    return false;
  }

  double vMmax = -1e20;
  const size_t cMelSize = (size_t)vMel.vNMel * (size_t)vMel.vNLen;
  for (size_t vI = 0; vI < cMelSize; vI++) {
    if (vMel.lData[vI] > vMmax) {
      vMmax = vMel.lData[vI];
    }
  }
  vMmax -= 8.0;

  for (size_t vI = 0; vI < cMelSize; vI++) {
    if (vMel.lData[vI] < vMmax) {
      vMel.lData[vI] = vMmax;
    }
    vMel.lData[vI] = (vMel.lData[vI] + 4.0) / 4.0;
  }

  int64_t vNFrames = vMel.vNLen;
  if (vNFrames % 2 == 1) {
    vNFrames--;
  }
  const int64_t cNMel = vMel.vNMel;
  const int64_t cNStacked = vNFrames / 2;

  mtmd_audio_mel vStacked;
  vStacked.vNMel = 2 * cNMel;
  vStacked.vNLen = cNStacked;
  vStacked.vNLenOrg = (int64_t)pNSamples;
  vStacked.lData.resize((size_t)2 * (size_t)cNMel * (size_t)cNStacked);

  for (int64_t vT = 0; vT < cNStacked; vT++) {
    for (int64_t vM = 0; vM < cNMel; vM++) {
      vStacked.lData[(size_t)vM * cNStacked + vT] = vMel.lData[(size_t)vM * vMel.vNLen + 2 * vT];
      vStacked.lData[(size_t)(vM + cNMel) * cNStacked + vT] = vMel.lData[(size_t)vM * vMel.vNLen + 2 * vT + 1];
    }
  }

  pOutput.push_back(std::move(vStacked));
  return true;
}

//
// mtmd_audio_preprocessor_parakeet implementation
//

void mtmd_audio_preprocessor_parakeet::mWorkerThread(int pIth, const float *pWindowFunc, int pWindowSize,
                                                     const std::vector<float> &pSamples, int pNSamples, int pFrameSize,
                                                     int pFrameStep, int pNThreads, int pNFftBins,
                                                     const mtmd_audio_cache &pCache, mtmd_audio_mel &pMel) {
  std::vector<float> lFftIn(pFrameSize * 2, 0.0);
  std::vector<float> lFftOut(pFrameSize * 2 * 2 * 2);

  int vNFb = pNFftBins;
  int vI = pIth;

  GGML_ASSERT(vNFb == 1 + (pFrameSize / 2));

  const double cEps = 5.960464477539063e-08;

  for (; vI < std::min(pNSamples / pFrameStep + 1, (int)pMel.vNLen); vI += pNThreads) {
    const int cOffset = vI * pFrameStep;
    const int cWindowPadLeft = (pFrameSize - pWindowSize) / 2;

    // Zero-pad left.
    std::fill(lFftIn.begin(), lFftIn.begin() + cWindowPadLeft, 0.0f);

    // Apply windowed samples in the center.
    const int cNToProcess = std::min({pWindowSize, pNSamples - cOffset});
    for (int vJ = 0; vJ < cNToProcess; vJ++) {
      lFftIn[cWindowPadLeft + vJ] = pWindowFunc[vJ] * pSamples[cOffset + cWindowPadLeft + vJ];
    }

    // Zero-pad right.
    std::fill(lFftIn.begin() + cWindowPadLeft + cNToProcess, lFftIn.begin() + pFrameSize, 0.0f);

    // FFT.
    fFft(pCache, lFftIn.data(), pFrameSize, lFftOut.data());

    // Calculate modulus^2 of complex numbers.
    for (int vJ = 0; vJ < vNFb; vJ++) {
      lFftOut[vJ] = (lFftOut[2 * vJ + 0] * lFftOut[2 * vJ + 0] + lFftOut[2 * vJ + 1] * lFftOut[2 * vJ + 1]);
    }

    // mel spectrogram.
    for (int vJ = 0; vJ < pMel.vNMel; vJ++) {
      double vSum = 0.0;
      int vK = 0;
      for (vK = 0; vK < vNFb - 3; vK += 4) {
        vSum += lFftOut[vK + 0] * pCache.vFilters.lData[vJ * vNFb + vK + 0] +
                lFftOut[vK + 1] * pCache.vFilters.lData[vJ * vNFb + vK + 1] +
                lFftOut[vK + 2] * pCache.vFilters.lData[vJ * vNFb + vK + 2] +
                lFftOut[vK + 3] * pCache.vFilters.lData[vJ * vNFb + vK + 3];
      }
      for (; vK < vNFb; vK++) {
        vSum += lFftOut[vK] * pCache.vFilters.lData[vJ * vNFb + vK];
      }
      pMel.lData[vJ * pMel.vNLen + vI] = std::log(vSum + cEps);
    }
  }

  // Otherwise fft_out are all zero.
  const double cEmptySum = std::log(cEps);
  for (; vI < pMel.vNLen; vI += pNThreads) {
    for (int vJ = 0; vJ < pMel.vNMel; vJ++) {
      pMel.lData[vJ * pMel.vNLen + vI] = cEmptySum;
    }
  }
}

void mtmd_audio_preprocessor_parakeet::mInitialize() {
  vCache.mFillSinCosTable(vHparams.vAudioNFft);

  const size_t cNFft = vHparams.vAudioNFft / 2 + 1;
  GGML_ASSERT(vHparams.lMelFilters.size() == (size_t)vHparams.vNMelBins * cNFft);
  vCache.vFilters.vNMel = vHparams.vNMelBins;
  vCache.vFilters.vNFft = cNFft;
  vCache.vFilters.lData = vHparams.lMelFilters;

  GGML_ASSERT(vHparams.lWindow.size() == (size_t)vHparams.vAudioWindowLen);
  GGML_ASSERT(vHparams.lWindow.size() <= (size_t)vHparams.vAudioNFft);
  vCache.lHannWindow = vHparams.lWindow;
}

bool mtmd_audio_preprocessor_parakeet::mPreprocess(const float *pSamples, size_t pNSamplesIn,
                                                   std::vector<mtmd_audio_mel> &pOutput) {
  if (pNSamplesIn == 0) {
    return false;
  }

  filter_params vParams;
  vParams.vNMel = vHparams.vNMelBins;
  vParams.vNFftBins = 1 + (vHparams.vAudioNFft / 2);
  vParams.vHannWindowSize = vHparams.vAudioWindowLen;
  vParams.vHopLength = vHparams.vAudioHopLen;
  vParams.vSampleRate = vHparams.vAudioSampleRate;

  GGML_ASSERT(!vCache.lSinVals.empty());
  GGML_ASSERT(!vCache.lCosVals.empty());
  GGML_ASSERT(!vCache.vFilters.lData.empty());

  const float *vWindowFunc = vCache.lHannWindow.data();
  const int cWindowSize = vParams.vHannWindowSize;
  const int cFrameSize = (vParams.vNFftBins - 1) * 2;
  const int cFrameStep = vParams.vHopLength;

  // Apply preemphasis filter (high-pass): x[i] = x[i] - 0.97 * x[i-1]
  std::vector<float> lSamplesPreprocessed(pSamples, pSamples + pNSamplesIn);
  {
    const float cPreemph = 0.97f;
    for (int vI = pNSamplesIn - 1; vI > 0; vI--) {
      lSamplesPreprocessed[vI] = lSamplesPreprocessed[vI] - cPreemph * lSamplesPreprocessed[vI - 1];
    }
  }

  // Parakeet uses centered constant padding
  const size_t cPad = (size_t)(cFrameSize / 2);
  std::vector<float> lSamplesPadded(pNSamplesIn + 2 * cPad, 0.0f);
  std::copy(lSamplesPreprocessed.begin(), lSamplesPreprocessed.end(), lSamplesPadded.begin() + cPad);

  mtmd_audio_mel vOutFull;
  vOutFull.vNMel = vParams.vNMel;
  vOutFull.vNLen = (lSamplesPadded.size() - cFrameSize) / cFrameStep + 1;
  vOutFull.vNLenOrg = vOutFull.vNLen;
  vOutFull.lData.resize(vOutFull.vNMel * vOutFull.vNLen);

  const int cNThreads = 4;
  std::vector<std::thread> lWorkers(cNThreads - 1);
  for (int vIw = 0; vIw < cNThreads - 1; ++vIw) {
    lWorkers[vIw] =
        std::thread(mWorkerThread, vIw + 1, vWindowFunc, cWindowSize, std::cref(lSamplesPadded), lSamplesPadded.size(),
                    cFrameSize, cFrameStep, cNThreads, vParams.vNFftBins, std::cref(vCache), std::ref(vOutFull));
  }

  mWorkerThread(0, vWindowFunc, cWindowSize, lSamplesPadded, lSamplesPadded.size(), cFrameSize, cFrameStep, cNThreads,
                vParams.vNFftBins, vCache, vOutFull);

  for (int vIw = 0; vIw < cNThreads - 1; ++vIw) {
    lWorkers[vIw].join();
  }

  // Per-feature normalization (only on valid frames)
  {
    const double cEps = 1e-5;
    int vValidFrames = pNSamplesIn / cFrameStep;

    for (int vJ = 0; vJ < vOutFull.vNMel; vJ++) {
      double vSum = 0.0;
      double vSqDiffSum = 0.0;

      // Calculate Mean ONLY on valid audio frames
      for (int vI = 0; vI < vValidFrames; vI++) {
        vSum += (double)vOutFull.lData[vJ * vOutFull.vNLen + vI];
      }
      double vMean = vSum / vValidFrames;

      // Calculate Variance ONLY on valid audio frames
      for (int vI = 0; vI < vValidFrames; vI++) {
        double vDiff = (double)vOutFull.lData[vJ * vOutFull.vNLen + vI] - vMean;
        vSqDiffSum += vDiff * vDiff;
      }

      double vStdDev = std::sqrt(vSqDiffSum / (vValidFrames - 1.0));
      double vDenominator = vStdDev + cEps;

      // Apply to ALL frames (including the padded ones)
      for (int vI = 0; vI < vOutFull.vNLen; vI++) {
        vOutFull.lData[vJ * vOutFull.vNLen + vI] =
            (float)((vOutFull.lData[vJ * vOutFull.vNLen + vI] - vMean) / vDenominator);
      }
    }
  }

  pOutput.push_back(std::move(vOutFull));
  return true;
}

//
// mtmd_audio_streaming_istft implementation
//

mtmd_audio_streaming_istft::mtmd_audio_streaming_istft(int pNFft, int pHopLength)
    : vNFft(pNFft), vHopLength(pHopLength), vNFftBins(pNFft / 2 + 1), lOverlapBuffer(pNFft, 0.0f),
      lWindowSumBuffer(pNFft, 0.0f), vPaddingToRemove((pNFft - pHopLength) / 2),
      lIfftIn(pNFft * 2 * 4, 0.0f), // extra space for recursive IFFT
      lIfftOut(pNFft * 2 * 4, 0.0f) {
  GGML_ASSERT(pNFft > 0 && pHopLength > 0 && pHopLength <= pNFft);
  vCache.mFillSinCosTable(pNFft);
  vCache.mFillHannWindow(pNFft, true);
}

void mtmd_audio_streaming_istft::mReset() {
  std::fill(lOverlapBuffer.begin(), lOverlapBuffer.end(), 0.0f);
  std::fill(lWindowSumBuffer.begin(), lWindowSumBuffer.end(), 0.0f);
  vPaddingToRemove = (vNFft - vHopLength) / 2;
}

std::vector<float> mtmd_audio_streaming_istft::mProcessFrame(const float *pFrameSpectrum) {
  std::vector<float> lOutput(vHopLength);

  // copy frequencies
  for (int vJ = 0; vJ < vNFftBins; vJ++) {
    lIfftIn[vJ * 2 + 0] = pFrameSpectrum[vJ * 2 + 0];
    lIfftIn[vJ * 2 + 1] = pFrameSpectrum[vJ * 2 + 1];
  }

  // mirror negative frequencies
  for (int vJ = 1; vJ < vNFftBins - 1; vJ++) {
    int vMirrorIdx = vNFft - vJ;
    lIfftIn[vMirrorIdx * 2 + 0] = lIfftIn[vJ * 2 + 0];
    lIfftIn[vMirrorIdx * 2 + 1] = -lIfftIn[vJ * 2 + 1]; // conjugate
  }

  fIfft(vCache, lIfftIn.data(), vNFft, lIfftOut.data());

  // update window sum and overlap buffer
  for (int vJ = 0; vJ < vNFft; vJ++) {
    lWindowSumBuffer[vJ] += vCache.lHannWindow[vJ] * vCache.lHannWindow[vJ];
    lOverlapBuffer[vJ] += lIfftOut[vJ * 2] * vCache.lHannWindow[vJ];
  }

  // extract hop_length samples with normalization
  for (int vI = 0; vI < vHopLength; vI++) {
    if (lWindowSumBuffer[vI] > 1e-8f) {
      lOutput[vI] = lOverlapBuffer[vI] / lWindowSumBuffer[vI];
    } else {
      lOutput[vI] = lOverlapBuffer[vI];
    }
  }

  // shift buffers left by hop_length
  std::copy(lOverlapBuffer.begin() + vHopLength, lOverlapBuffer.end(), lOverlapBuffer.begin());
  std::fill(lOverlapBuffer.end() - vHopLength, lOverlapBuffer.end(), 0.0f);

  std::copy(lWindowSumBuffer.begin() + vHopLength, lWindowSumBuffer.end(), lWindowSumBuffer.begin());
  std::fill(lWindowSumBuffer.end() - vHopLength, lWindowSumBuffer.end(), 0.0f);

  // Remove padding if needed
  int vToRemove = std::min(vPaddingToRemove, (int)lOutput.size());
  vPaddingToRemove -= vToRemove;
  lOutput.erase(lOutput.begin(), lOutput.begin() + vToRemove);

  return lOutput;
}

std::vector<float> mtmd_audio_streaming_istft::mFlush() {
  std::vector<float> lOutput;

  // Extract remaining samples from overlap buffer
  // Continue until we've extracted all meaningful samples
  int vRemaining = vNFft - vHopLength;
  while (vRemaining > 0) {
    int vChunkSize = std::min(vRemaining, vHopLength);

    for (int vI = 0; vI < vChunkSize; vI++) {
      float vSample;
      if (lWindowSumBuffer[vI] > 1e-8f) {
        vSample = lOverlapBuffer[vI] / lWindowSumBuffer[vI];
      } else {
        vSample = lOverlapBuffer[vI];
      }
      lOutput.push_back(vSample);
    }

    // Shift buffers
    std::copy(lOverlapBuffer.begin() + vChunkSize, lOverlapBuffer.end(), lOverlapBuffer.begin());
    std::fill(lOverlapBuffer.end() - vChunkSize, lOverlapBuffer.end(), 0.0f);

    std::copy(lWindowSumBuffer.begin() + vChunkSize, lWindowSumBuffer.end(), lWindowSumBuffer.begin());
    std::fill(lWindowSumBuffer.end() - vChunkSize, lWindowSumBuffer.end(), 0.0f);

    vRemaining -= vChunkSize;
  }

  return lOutput;
}
