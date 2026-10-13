#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <thread>
#include <algorithm>

void fCommonNgramCacheUpdate(common_ngram_cache &pNgramCache, int pNgramMin, int pNgramMax,
                             std::vector<mgs_token> &pInp, int pNnew, bool pPrintProgress) {
  const int64_t cTStartMs = ggml_time_ms();
  const int64_t cInpSize = pInp.size();

  const int64_t cNTodo = cInpSize * (pNgramMax - pNgramMin + 1);
  int64_t vNDone = 0;

  for (int64_t vNgramSize = pNgramMin; vNgramSize <= pNgramMax; ++vNgramSize) {
    const int64_t cIStart = std::max(cInpSize - pNnew, vNgramSize);
    for (int64_t vI = cIStart; vI < cInpSize; ++vI) {
      const int64_t cNgramStart = vI - vNgramSize;
      common_ngram vNgram(&pInp[cNgramStart], vNgramSize);
      const mgs_token cToken = pInp[vI];

      common_ngram_cache::iterator vPartIt = pNgramCache.find(vNgram);
      if (vPartIt == pNgramCache.end()) {
        common_ngram_cache_part vPart;
        vPart.emplace(cToken, 1);
        pNgramCache.emplace(vNgram, vPart);
      } else {
        common_ngram_cache_part::iterator vTokenCountIt = vPartIt->second.find(cToken);
        if (vTokenCountIt == vPartIt->second.end()) {
          vPartIt->second.emplace(cToken, 1);
        } else {
          vTokenCountIt->second++;
        }
      }
      ++vNDone;

      if (pPrintProgress && vNDone % 10000000 == 0) {
        const int64_t cTNowMs = ggml_time_ms();
        const int64_t cEtaMs = (cInpSize * (pNgramMax - pNgramMin + 1) - vNDone) * (cTNowMs - cTStartMs) / vNDone;
        const int64_t cEtaMin = cEtaMs / (60 * 1000);
        const int64_t cEtaS = (cEtaMs - 60 * 1000 * cEtaMin) / 1000;

        fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, vNDone,
                cNTodo, cEtaMin, cEtaS);
      }
    }
  }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static mgs_token fGetToken(const std::vector<mgs_token> &pInp, const std::vector<mgs_token> &pDraft, const size_t pI) {
  return pI < pInp.size() ? pInp[pI] : pDraft[1 + pI - pInp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int aDraftMinSampleSizeLax[MGS_NGRAM_MAX] = {2, 2, 1, 1};
constexpr int aDraftMinPercentLax[MGS_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int aDraftMinSampleSizeStrict[MGS_NGRAM_MAX] = {4, 3, 2, 2};
constexpr int aDraftMinPercentStrict[MGS_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static mgs_token fTryDraft(common_ngram_cache &pNcStatic, const common_ngram pNgramStatic) {
  common_ngram_cache::iterator vPartStaticIt = pNcStatic.find(pNgramStatic);
  if (vPartStaticIt == pNcStatic.end()) {
    return MGS_TOKEN_NULL;
  }
  const common_ngram_cache_part cPartStatic = vPartStaticIt->second;

  int vMaxCountStatic = 0;
  int vSumCountStatic = 0;
  mgs_token vMaxToken = MGS_TOKEN_NULL;

  for (std::pair<mgs_token, int> vTokenCountStatic : cPartStatic) {
    const mgs_token cToken = vTokenCountStatic.first;
    const int32_t cCountStatic = vTokenCountStatic.second;

    if (cCountStatic > vMaxCountStatic) {
      vMaxToken = cToken;
      vMaxCountStatic = cCountStatic;
    }
    vSumCountStatic += cCountStatic;
  }

  if (vSumCountStatic < aDraftMinSampleSizeLax[MGS_NGRAM_STATIC - 1]) {
    return MGS_TOKEN_NULL;
  }
  if (100 * vMaxCountStatic < aDraftMinPercentLax[MGS_NGRAM_STATIC - 1] * vSumCountStatic) {
    return MGS_TOKEN_NULL;
  }
  return vMaxToken;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static mgs_token fTryDraft(common_ngram_cache &pNcPrimary, const std::vector<common_ngram> &pNgramsPrimary,
                           common_ngram_cache_part &pPartStatic, const int *pMinSampleSize, const int *pMinPercent) {

  mgs_token vDraftedToken = MGS_TOKEN_NULL;

  for (int vI = pNgramsPrimary.size() - 1; vI >= 0 && vDraftedToken == MGS_TOKEN_NULL; --vI) {
    const common_ngram cNgramPrimary = pNgramsPrimary[vI];

    common_ngram_cache::iterator vPartPrimaryIt = pNcPrimary.find(cNgramPrimary);
    if (vPartPrimaryIt == pNcPrimary.end()) {
      continue;
    }
    const common_ngram_cache_part cPartPrimary = vPartPrimaryIt->second;

    int vMaxCountPrimary = 0;
    int vMaxCountStatic = 0;
    int vSumCountPrimary = 0;
    mgs_token vMaxToken = MGS_TOKEN_NULL;

    for (std::pair<mgs_token, int> vTokenCountPrimary : cPartPrimary) {
      const mgs_token cToken = vTokenCountPrimary.first;

      common_ngram_cache_part::iterator vTokenCountStaticIt = pPartStatic.find(cToken);

      const int32_t cCountPrimary = vTokenCountPrimary.second;
      const int32_t cCountStatic = vTokenCountStaticIt != pPartStatic.end() ? 100 * vTokenCountStaticIt->second : 1;

      if (cCountPrimary * cCountStatic > vMaxCountPrimary * vMaxCountStatic) {
        vMaxToken = cToken;
        vMaxCountPrimary = cCountPrimary;
        vMaxCountStatic = cCountStatic;
      }
      vSumCountPrimary += cCountPrimary;
    }

    if (vSumCountPrimary < pMinSampleSize[vI]) {
      continue;
    }
    if (100 * vMaxCountPrimary < pMinPercent[vI] * vSumCountPrimary) {
      continue;
      ;
    }
    vDraftedToken = vMaxToken;
  }

  return vDraftedToken;
}

void fCommonNgramCacheDraft(std::vector<mgs_token> &pInp, std::vector<mgs_token> &pDraft, int pNDraft, int pNgramMin,
                            int pNgramMax, common_ngram_cache &pNcContext, common_ngram_cache &pNcDynamic,
                            common_ngram_cache &pNcStatic) {
  GGML_ASSERT(pDraft.size() == 1);
  const int cInpSize = pInp.size();

  if (cInpSize < MGS_NGRAM_STATIC) {
    return;
  }

  while ((int)pDraft.size() - 1 < pNDraft) {
    mgs_token vDraftedToken = MGS_TOKEN_NULL;

    const int cNgramStartStatic = cInpSize - MGS_NGRAM_STATIC + pDraft.size() - 1;
    common_ngram vNgramStatic;
    for (int vJ = cNgramStartStatic; vJ < cNgramStartStatic + MGS_NGRAM_STATIC; ++vJ) {
      vNgramStatic.aTokens[vJ - cNgramStartStatic] = fGetToken(pInp, pDraft, vJ);
    }
    common_ngram_cache::iterator vPartStaticIt = pNcStatic.find(vNgramStatic);
    common_ngram_cache_part vPartStatic;
    if (vPartStaticIt != pNcStatic.end()) {
      vPartStatic = vPartStaticIt->second;
    }

    // cd = context + dynamic
    std::vector<common_ngram> lNgramsCd;
    for (int vNgramSizeCd = pNgramMin; vNgramSizeCd <= pNgramMax; ++vNgramSizeCd) {
      const int cNgramStartCd = cInpSize - vNgramSizeCd + pDraft.size() - 1;
      common_ngram vNgramCd;
      for (int vJ = cNgramStartCd; vJ < cNgramStartCd + vNgramSizeCd; ++vJ) {
        vNgramCd.aTokens[vJ - cNgramStartCd] = fGetToken(pInp, pDraft, vJ);
      }
      lNgramsCd.push_back(vNgramCd);
    }
    if (vDraftedToken == MGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcContext, lNgramsCd, vPartStatic, aDraftMinSampleSizeLax, aDraftMinPercentLax);
    }
    if (vDraftedToken == MGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcDynamic, lNgramsCd, vPartStatic, aDraftMinSampleSizeStrict, aDraftMinPercentStrict);
    }
    if (vDraftedToken == MGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcStatic, vNgramStatic);
    }

    if (vDraftedToken == MGS_TOKEN_NULL) {
      break;
    }

    LOG_DBG(" - draft candidate: token=%d\n", vDraftedToken);
    pDraft.push_back(vDraftedToken);
  }
}

void fCommonNgramCacheSave(common_ngram_cache &pNgramCache, const std::string &pFilename) {
  std::ofstream vFileOut(pFilename, std::ios::binary);
  for (std::pair<common_ngram, common_ngram_cache_part> vItem : pNgramCache) {
    const common_ngram cNgram = vItem.first;
    common_ngram_cache_part vTokenCounts = vItem.second;
    GGML_ASSERT(!vTokenCounts.empty());
    const int32_t cNtokens = vTokenCounts.size();
    GGML_ASSERT(cNtokens > 0);

    vFileOut.write(reinterpret_cast<const char *>(&cNgram), sizeof(common_ngram));
    vFileOut.write(reinterpret_cast<const char *>(&cNtokens), sizeof(int32_t));
    for (std::pair<mgs_token, int32_t> vItem2 : vTokenCounts) {
      const mgs_token cToken = vItem2.first;
      const int32_t cCount = vItem2.second;
      GGML_ASSERT(cCount > 0);

      vFileOut.write(reinterpret_cast<const char *>(&cToken), sizeof(mgs_token));
      vFileOut.write(reinterpret_cast<const char *>(&cCount), sizeof(int32_t));
    }
  }
}

common_ngram_cache fCommonNgramCacheLoad(const std::string &pFilename) {
  std::ifstream vHashmapFile(pFilename, std::ios::binary);
  if (!vHashmapFile) {
    throw std::ifstream::failure("Unable to open file " + pFilename);
  }
  common_ngram_cache vNgramCache;

  common_ngram vNgram;
  int32_t vNtokens;
  mgs_token vToken;
  int32_t vCount;

  char *vNgramc = reinterpret_cast<char *>(&vNgram);
  char *vNtokensc = reinterpret_cast<char *>(&vNtokens);
  char *vTokenc = reinterpret_cast<char *>(&vToken);
  char *vCountc = reinterpret_cast<char *>(&vCount);
  while (vHashmapFile.read(vNgramc, sizeof(common_ngram))) {
    GGML_ASSERT(!vHashmapFile.eof());
    GGML_ASSERT(vHashmapFile.read(vNtokensc, sizeof(int32_t)));
    GGML_ASSERT(vNtokens > 0);
    common_ngram_cache_part vTokenCounts;

    for (int vI = 0; vI < vNtokens; ++vI) {
      GGML_ASSERT(!vHashmapFile.eof());
      GGML_ASSERT(vHashmapFile.read(vTokenc, sizeof(mgs_token)));
      GGML_ASSERT(!vHashmapFile.eof());
      GGML_ASSERT(vHashmapFile.read(vCountc, sizeof(int32_t)));
      GGML_ASSERT(vCount > 0);
      vTokenCounts.emplace(vToken, vCount);
    }

    vNgramCache.emplace(vNgram, vTokenCounts);
  }
  GGML_ASSERT(vHashmapFile.eof());

  return vNgramCache;
}

void fCommonNgramCacheMerge(common_ngram_cache &pNgramCacheTarget, common_ngram_cache &pNgramCacheAdd) {
  for (std::pair<common_ngram, common_ngram_cache_part> vNgramPart : pNgramCacheAdd) {
    const common_ngram cNgram = vNgramPart.first;
    common_ngram_cache_part vPart = vNgramPart.second;

    common_ngram_cache::iterator vPartMergedIt = pNgramCacheTarget.find(cNgram);
    if (vPartMergedIt == pNgramCacheTarget.end()) {
      pNgramCacheTarget.emplace(cNgram, vPart);
      continue;
    }

    for (std::pair<mgs_token, int32_t> vTokenCount : vPart) {
      const mgs_token cToken = vTokenCount.first;
      const int32_t cCount = vTokenCount.second;
      GGML_ASSERT(cCount > 0);

      common_ngram_cache_part::iterator vTokenCountMergedIt = vPartMergedIt->second.find(cToken);
      if (vTokenCountMergedIt == vPartMergedIt->second.end()) {
        vPartMergedIt->second.emplace(cToken, cCount);
        continue;
      }

      vTokenCountMergedIt->second += cCount;
    }
  }
}
