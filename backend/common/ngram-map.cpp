#include "common.h"
#include "log.h"
#include "ngram-map.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <sstream>

// prime number used for LCG hash function (32 bit), it is near (sqrt(5) - 1)/2 * 2^32.
#define LCG_FACTOR 2654435761UL

// Compute the LCG hash of a n-gram of size len at offset start.
static uint32_t fCommonNgramMapHash(const mgs_tokens &pTokens, size_t pStart, size_t pLen) {
  uint32_t vHash = 0;
  for (size_t vI = 0; vI < pLen; ++vI) {
    vHash = vHash * LCG_FACTOR + pTokens[pStart + vI];
  }
  return vHash;
}

// Print the values of a sublist of `mgs_tokens & inp` to a string in the form [v0, v1, v2, ...].
static std::string fCommonTokensToStr(const mgs_tokens &pInp, size_t pStart, size_t pLength) {
  std::ostringstream vOss;
  vOss << '[';
  for (size_t vI = 0; vI < pLength; ++vI) {
    if (vI > 0) {
      vOss << ", ";
    }
    vOss << pInp[pStart + vI];
  }
  vOss << ']';
  return vOss.str();
}

// n-gram simple
//

/**
 * Perform speculative generation using the model's own token history.
 * Searches for a matching pattern in the token history and returns draft tokens.
 *
 * @param state     Current state of this implementation
 * @param tokens    Token history to search in
 * @param sampled   Last sampled token
 * @return Vector of draft tokens, empty if no matching pattern is found
 */
mgs_tokens fCommonNgramSimpleDraft(const common_ngram_simple_config &pConfig, const mgs_tokens &pTokens,
                                   mgs_token pSampled) {

  // Simple implementation of self-speculative decoding without a draft model.
  //
  const size_t cCurLen = pTokens.size();

  const size_t cNDraftMin = pConfig.vSizeNgram; // size of n-gram to lookup in token history
  const size_t cNDraftMax = pConfig.vSizeMgram; // the m-gram following the found n-gram is used for draft

  // vector for tokens we want to verify.
  // return empty vector if there is no match.
  mgs_tokens vDraftTokens;

  // We need at least n_draft_min + n_draft_max + 1 tokens.
  if (cCurLen <= static_cast<size_t>(cNDraftMin + cNDraftMax + 1)) {
    return vDraftTokens;
  }

  // pattern search
  mgs_tokens vPattern;
  vPattern.reserve(cNDraftMin);
  for (size_t vJ = cCurLen - cNDraftMin + 1; vJ < cCurLen; ++vJ) {
    vPattern.push_back(pTokens[vJ]);
  }
  vPattern.push_back(pSampled); // add the last token to the pattern

  size_t vMatchPos = 0; // we ignore position 0, position 0 == no match
                        // search backwards, but skip the current match (we are currently there)
  for (size_t vJ = cCurLen - cNDraftMin - 1; vJ > 0; --vJ) {
    bool vMatch = true;
    for (size_t vK = 0; vK < vPattern.size(); ++vK) {
      if (pTokens[vJ + vK] != vPattern[vK]) {
        vMatch = false;
        break;
      }
    }
    if (vMatch) {
      vMatchPos = vJ;
      break;
    }
  }
  if (vMatchPos == 0) {
    return vDraftTokens;
  }

  const size_t cCopyMax = std::min(cNDraftMax, cCurLen - (vMatchPos + cNDraftMin));
  if (cCopyMax < cNDraftMin) {
    return vDraftTokens;
  }
  LOG_DBG("%s: #tokens = %zu: found matching pattern at pos %zu, length %zu, draft length %zu\n", __func__, cCurLen,
          vMatchPos, vPattern.size(), cCopyMax);

  vDraftTokens.reserve(cCopyMax);
  for (size_t vJ = 0; vJ < cCopyMax; ++vJ) {
    vDraftTokens.push_back(pTokens[vMatchPos + cNDraftMin + vJ]);
  }
  return vDraftTokens;
}

// n-gram map
//

// maximum number of counted values of a ngram map value.
#define COMMON_NGRAM_MAX_VALUE_COUNT 16380

void fCommonNgramMapBegin(common_ngram_map &pMap, const mgs_tokens &pTokens) {
  size_t vSizeBegin = pTokens.size();

  LOG_DBG("%s: begin, idx_last_draft=%zu, new begin=%zu, #keys=%zu\n", __func__, pMap.vIdxLastCheck, vSizeBegin,
          pMap.lKeys.size());

  size_t vIdxBeginCleanup = pMap.vSizeLastBegin;
  if (vIdxBeginCleanup > vSizeBegin) {
    if (vSizeBegin > (size_t)pMap.vSizeKey + pMap.vSizeValue) {
      vIdxBeginCleanup = vSizeBegin - pMap.vSizeKey - pMap.vSizeValue;
    } else {
      vIdxBeginCleanup = 0;
    }
    LOG_INF("%s: shrink cleanup begin: %zu -> %zu\n", __func__, pMap.vSizeLastBegin, vIdxBeginCleanup);
  }

  size_t vCountMapEntriesUpd = 0;
  if (!pMap.lKeyMap.empty() && vSizeBegin < pMap.vIdxLastCheck) {
    if (pMap.vShowKeyMapStats) {
      // Print statistics of hash map map_key.
      size_t vCountNonzero = 0;
      uint32_t vMinIdx = UINT32_MAX;
      uint32_t vMaxIdx = 0;
      for (size_t vI = 0; vI < pMap.lKeyMap.size(); ++vI) {
        uint32_t vKeyIdx = pMap.lKeyMap[vI];
        if (vKeyIdx != 0) {
          ++vCountNonzero;
          if (vKeyIdx < vMinIdx)
            vMinIdx = vKeyIdx;
          if (vKeyIdx > vMaxIdx)
            vMaxIdx = vKeyIdx;
        }
      }
      if (vCountNonzero == 0) {
        vMinIdx = 0;
      }
      LOG_INF("%s: key_map stats: entries=%zu, min_idx=%u, max_idx=%u, key_map_last_idx=%u\n", __func__, vCountNonzero,
              vMinIdx, vMaxIdx, pMap.vKeyMapLastIdx);
    }

    // Update the map from hash to key index (clear outdated entries).
    for (size_t vI = 0; vI < pMap.lKeyMap.size(); ++vI) {
      uint32_t vKeyIdx = pMap.lKeyMap[vI];
      if (vKeyIdx != 0 && vKeyIdx >= vIdxBeginCleanup) {
        pMap.lKeyMap[vI] = 0;
        vCountMapEntriesUpd++;
      }
    }
    pMap.vKeyMapLastIdx = (vIdxBeginCleanup > 0) ? (uint32_t)(vIdxBeginCleanup - 1) : 0;
  }

  if (vSizeBegin < pMap.vIdxLastCheck && !pMap.lKeys.empty()) {
    size_t vCountKeys = pMap.lKeys.size();
    size_t vCountKeysDel = 0;
    size_t vCountValuesDel = 0;
    for (int32_t vI = pMap.lKeys.size() - 1; vI >= 0; --vI) {
      common_ngram_map_key &vKey = pMap.lKeys[vI];
      if (vKey.vKeyIdx >= vIdxBeginCleanup) {
        // Delete the key.
        LOG_DBG("%s: delete key %d at index %zu (>= idx_begin_cleanup=%zu)\n", __func__, vI, vKey.vKeyIdx,
                vIdxBeginCleanup);
        pMap.lKeys.erase(pMap.lKeys.begin() + vI);
        vCountKeysDel++;
        continue;
      }
      if (pMap.vKeyOnly) {
        continue;
      }

      // Check the indices of the values.
      for (int16_t vJ = COMMON_NGRAM_MAX_VALUES - 1; vJ >= 0; --vJ) {
        common_ngram_map_value &vValue = vKey.aValues[vJ];
        if (vValue.vValueIdx != 0 && vValue.vValueIdx >= vIdxBeginCleanup) {
          // Delete the value.
          vCountValuesDel++;

          // Move all values after this value to the left.
          for (uint16_t vK = vJ; vK < COMMON_NGRAM_MAX_VALUES - 1; ++vK) {
            vKey.aValues[vK] = vKey.aValues[vK + 1];
          }
          // Clear the last value.
          vKey.aValues[COMMON_NGRAM_MAX_VALUES - 1].vValueIdx = 0;
          vKey.aValues[COMMON_NGRAM_MAX_VALUES - 1].vValueNum = 0;
        }
      }
      if (vKey.aValues[0].vValueIdx == 0) {
        // No values left, delete the key.
        LOG_DBG("%s: delete key %d at index %zu (no values left)\n", __func__, vI, vKey.vKeyIdx);
        pMap.lKeys.erase(pMap.lKeys.begin() + vI);
        vCountKeysDel++;
      }
    }

    LOG_INF("%s: refresh map: idx_last_draft=%zu, new begin=%zu, #keys_checked=%zu, #keys_del=%zu, #values_del=%zu, "
            "#hashes_upd=%zu\n",
            __func__, pMap.vIdxLastCheck, vSizeBegin, vCountKeys, vCountKeysDel, vCountValuesDel, vCountMapEntriesUpd);
  }

  pMap.vIdxLastCheck = vSizeBegin;
  pMap.vSizeLastBegin = vSizeBegin;
}

void fCommonNgramMapDraft(common_ngram_map &pMap, const mgs_tokens &pInp, mgs_token pSampled, mgs_tokens &pDraft) {
  // reset last key and value.
  pMap.vLastDraftCreated = false;
  pMap.vLastDraftKeyIdx = 0;
  pMap.vLastDraftValueIdx = 0;

  const size_t cCurLen = pInp.size();
  const uint16_t cN = pMap.vSizeKey;
  const uint16_t cM = pMap.vSizeValue;
  if (cCurLen < static_cast<size_t>(2 * cN + cM)) {
    return;
  }
  if (cCurLen >= static_cast<size_t>(UINT32_MAX)) {
    // key_map uses uint32_t instead of size_t.
    GGML_ABORT("%s: cur_len exceeds UINT32_MAX: %zu", __func__, cCurLen);
  }

  if (pMap.vIdxLastCheck > cCurLen) {
    // Should not happen because of fCommonNgramMapBegin().
    GGML_ABORT("%s: map.idx_last_check > cur_len: %zu > %zu", __func__, pMap.vIdxLastCheck, cCurLen);
  }
  pMap.vIdxLastCheck = cCurLen;

  // search pattern, the key n-gram
  std::vector<mgs_token> lKeyTokens;
  lKeyTokens.reserve(cN);
  for (size_t vJ = cCurLen - cN + 1; vJ < cCurLen; ++vJ) {
    lKeyTokens.push_back(pInp[vJ]);
  }
  lKeyTokens.push_back(pSampled);

  // search for the key in the map
  size_t vMatchPos = 0;
  if (pMap.vSizeLastBegin > cCurLen) {
    GGML_ABORT("%s: map.size_last_begin > cur_len: %zu > %zu", __func__, pMap.vSizeLastBegin, cCurLen);
  }
  if (!pMap.lKeyMap.empty()) {
    // Search for the key in the map key_map from hash of ngrams to index of ngram.
    uint32_t vIdxHash = (fCommonNgramMapHash(lKeyTokens, 0, cN) % pMap.lKeyMap.size());
    uint32_t vIdxKey = pMap.lKeyMap[vIdxHash];
    if (vIdxKey != 0 && vIdxKey < cCurLen - cN - cM - 1) {
      // Check if the key matches the key at idx_key (because of possible collisions).
      bool vMatch = true;
      for (size_t vK = 0; vK < cN; ++vK) {
        if (pInp[vIdxKey + vK] != lKeyTokens[vK]) {
          vMatch = false;
          break;
        }
      }
      LOG_DBG("%s: key hash %x -> idx_key %d: match %d\n", __func__, vIdxHash, vIdxKey, vMatch ? 1 : 0);
      if (vMatch) {
        vMatchPos = vIdxKey;
      }
    }
  }
  if (vMatchPos == 0 && pMap.vSizeLastBegin > (size_t)(cN + cM + 1)) {
    // Search for the key in [1, pMap.vSizeLastBegin - cN - cM -1], descending.
    for (size_t vJ = pMap.vSizeLastBegin - cN - cM - 1; vJ > pMap.vKeyMapLastIdx; --vJ) {
      // Check if the key matches the key.
      bool vMatch = true;
      for (size_t vK = 0; vK < cN; ++vK) {
        if (pInp[vJ + vK] != lKeyTokens[vK]) {
          vMatch = false;
          break;
        }
      }
      if (vMatch) {
        vMatchPos = vJ;
        break;
      }
    }
  }
  if (vMatchPos == 0) {
    // In case of a reasoning chat, the part after size_last_begin may be deleted/reordered later.
    //
    // Search in [size_last_begin, cur_len - n - m - 1], descending.
    for (size_t vJ = cCurLen - cN - cM - 1; vJ > pMap.vSizeLastBegin && vJ > pMap.vKeyMapLastIdx; --vJ) {
      bool vMatch = true;
      for (size_t vK = 0; vK < cN; ++vK) {
        if (pInp[vJ + vK] != lKeyTokens[vK]) {
          vMatch = false;
          break;
        }
      }
      if (vMatch) {
        vMatchPos = vJ;
        break;
      }
    }
  }
  if (vMatchPos > 0) {
    LOG_DBG("%s: cur_len = %zu, n = %d, m = %d, sz_tkns = %zu, sampled = %d, match_pos = %zu\n", __func__, cCurLen, cN,
            cM, lKeyTokens.size(), pSampled, vMatchPos);
  }

  if (!pMap.lKeyMap.empty()) {
    // Add hashes of new ngrams in key_map.
    //
    // Use the same order as above.
    if (pMap.vSizeLastBegin > (size_t)(cN + cM + 1)) {
      for (size_t vJ = pMap.vSizeLastBegin - cN - cM - 1; vJ > pMap.vKeyMapLastIdx; --vJ) {
        // compute hash and store index of ngram at idx j in the map.
        uint32_t vIdxHash = (fCommonNgramMapHash(pInp, vJ, cN) % pMap.lKeyMap.size());
        if (pMap.lKeyMap[vIdxHash] == 0) {
          pMap.lKeyMap[vIdxHash] = vJ; // collisions may occur
        }
      }
    }

    for (size_t vJ = cCurLen - cN - cM - 1; vJ > pMap.vSizeLastBegin && vJ > pMap.vKeyMapLastIdx; --vJ) {
      // compute hash and store index of ngram at idx j in the map.
      uint32_t vIdxHash = (fCommonNgramMapHash(pInp, vJ, cN) % pMap.lKeyMap.size());
      if (pMap.lKeyMap[vIdxHash] == 0) {
        pMap.lKeyMap[vIdxHash] = vJ;
      }
    }
    pMap.vKeyMapLastIdx = std::max(static_cast<uint32_t>(cCurLen - cN - cM - 1), pMap.vKeyMapLastIdx);
  }

  if (vMatchPos == 0) {
    return;
  }

  // We have a match, now we look for the statistics of the key.
  size_t vKeyOffset = pMap.lKeys.size(); // offset in the map
  // We iterate through the std::vector<common_ngram_map_key> pMap.lKeys.
  for (size_t vI = 0; vI < pMap.lKeys.size(); ++vI) {
    bool vMatch = true;
    for (size_t vJ = 0; vJ < cN; ++vJ) {
      if (pInp[pMap.lKeys[vI].vKeyIdx + vJ] != lKeyTokens[vJ]) {
        vMatch = false;
        break;
      }
    }
    if (vMatch) {
      vKeyOffset = vI;
      break;
    }
  }
  if (vKeyOffset == pMap.lKeys.size()) {
    // We create a new key-entry, it will get offset key_offset.
    common_ngram_map_key vNewKey;
    vNewKey.vKeyIdx = vMatchPos;
    vNewKey.vStatIdx = 0;
    vNewKey.vKeyNum = 0;
    for (int vI = 0; vI < COMMON_NGRAM_MAX_VALUES; ++vI) {
      vNewKey.aValues[vI].vValueNum = 0;
      vNewKey.aValues[vI].vNAccepted = cM;
    }
    pMap.lKeys.push_back(vNewKey);
  }

  // our key n-gram:
  common_ngram_map_key &vCurrKey = pMap.lKeys[vKeyOffset];

  // update number of key hits
  vCurrKey.vKeyNum = (uint16_t)std::min((int)pMap.lKeys[vKeyOffset].vKeyNum + 1, (int)COMMON_NGRAM_MAX_VALUE_COUNT);

  if (pMap.vKeyOnly) {
    // simple mode:
    // Fill in the draft with the m tokens following the key.
    // We work with value values[0] only.
    int vNDraftTokens = std::min((int)cM, (int)vCurrKey.aValues[0].vNAccepted);

    for (int vI = 0; vI < vNDraftTokens; ++vI) {
      pDraft.push_back(pInp[vMatchPos + cN + vI]);
    }

    LOG_DBG("%s: key_idx = %zu, key_offset = %zu, key_num = %d, draft.size = %zu\n", __func__, vCurrKey.vKeyIdx,
            vKeyOffset, vCurrKey.vKeyNum, pDraft.size());

    pMap.vLastDraftCreated = true;
    pMap.vLastDraftKeyIdx = vKeyOffset;
    pMap.vLastDraftValueIdx = 0; // value 0 is used for simple mode
    return;
  }

  if (vCurrKey.vKeyNum < pMap.vMinHits) {
    // not enough hits to consider this a good draft
    LOG_DBG("%s: key_offset = %zu, key_num = %d, min_hits = %d, no draft\n", __func__, vKeyOffset, vCurrKey.vKeyNum,
            pMap.vMinHits);
    return;
  }

  // complex mode: examine the different m-grams after this key n-gram.
  //

  // determine all (max COMMON_NGRAM_MAX_VALUES) m-grams after the key n-gram.
  for (size_t vI = vCurrKey.vStatIdx; vI <= vMatchPos; ++vI) {
    // begins the key n-gram at index i?
    bool vMatchKey = true;
    for (size_t vK = 0; vK < cN; ++vK) {
      if (pInp[vI + vK] != lKeyTokens[vK]) {
        vMatchKey = false;
        break;
      }
    }
    if (!vMatchKey) {
      continue;
    }

    // Do we haven a existing value m-gram or a new one after the key at index i?
    size_t vIdxBeginValueKey = vI + cN;
    int vIdxValue = -1;
    for (int vV = 0; vV < COMMON_NGRAM_MAX_VALUES; ++vV) {
      size_t vIdxBeginValueV = vCurrKey.aValues[vV].vValueIdx;
      if (vIdxBeginValueV == 0) {
        // We found an empty value slot => we found a new value m-gram after the key n-gram.
        vCurrKey.aValues[vV].vValueIdx = vIdxBeginValueKey;
        vCurrKey.aValues[vV].vValueNum = 0;
        vCurrKey.aValues[vV].vNAccepted = cM;
        vIdxValue = vV;
        break;
      }
      bool vMatch = true;
      for (size_t vJ = 0; vJ < cM; ++vJ) {
        if (pInp[vIdxBeginValueKey + vJ] != pInp[vIdxBeginValueV + vJ]) {
          vMatch = false;
          break;
        }
      }
      if (vMatch) {
        // We found an existing value m-gram after the key n-gram.
        vIdxValue = vV;
        break;
      }
    }
    if (vIdxValue >= 0) {
      // We found a value m-gram of the key n-gram.
      vCurrKey.aValues[vIdxValue].vValueNum =
          (uint16_t)std::min((int)vCurrKey.aValues[vIdxValue].vValueNum + 1, (int)COMMON_NGRAM_MAX_VALUE_COUNT);
    }
  }
  // the statistics are updated up to match_pos.
  vCurrKey.vStatIdx = vMatchPos;

  // Do we have a value we could use for the draft?
  uint16_t vMaxOccur = 0;
  int vSlotMax = 0;
  for (int vV = 0; vV < COMMON_NGRAM_MAX_VALUES; ++vV) {
    uint16_t vCurrOccur = vCurrKey.aValues[vV].vValueNum;
    if (vCurrOccur > vMaxOccur) {
      vMaxOccur = vCurrOccur;
      vSlotMax = vV;
    }
  }
  // What is sum of the other occurrences?
  uint32_t vSumOccur = 0;
  for (int vV = 0; vV < COMMON_NGRAM_MAX_VALUES; ++vV) {
    if (vV == vSlotMax) {
      continue;
    }
    uint16_t vCurrOccur = vCurrKey.aValues[vV].vValueNum;
    vSumOccur += vCurrOccur;
  }

  LOG_DBG("%s: key_offset = %zu, max_occur = %d, sum_occur = %d, slot_max = %d [%zu/%d, %zu/%d, %zu/%d, %zu/%d]\n",
          __func__, vKeyOffset, vMaxOccur, vSumOccur, vSlotMax, vCurrKey.aValues[0].vValueIdx,
          vCurrKey.aValues[0].vValueNum, vCurrKey.aValues[1].vValueIdx, vCurrKey.aValues[1].vValueNum,
          vCurrKey.aValues[2].vValueIdx, vCurrKey.aValues[2].vValueNum, vCurrKey.aValues[3].vValueIdx,
          vCurrKey.aValues[3].vValueNum);
  // Print the tokens of the four values (if idx != 0), use LOG_INF
  for (int vV = 0; vV < COMMON_NGRAM_MAX_VALUES; ++vV) {
    if (vCurrKey.aValues[vV].vValueIdx != 0) {
      LOG_DBG("%s: value[%d] = %s\n", __func__, vV,
              fCommonTokensToStr(pInp, vCurrKey.aValues[vV].vValueIdx, cM).c_str());
    }
  }

  if (vSumOccur > 0 && vMaxOccur < 2 * vSumOccur) {
    // The most frequent value is not much more frequent than the other values.
    // We do not use the draft.
    return;
  }

  // We use the most frequent value values[slot_max] for the draft.
  // Fill in the draft with the m tokens following the key.
  int vNDraftTokens = std::min((int)cM, (int)vCurrKey.aValues[vSlotMax].vNAccepted);

  for (int vI = 0; vI < vNDraftTokens; ++vI) {
    pDraft.push_back(pInp[vMatchPos + cN + vI]);
  }

  LOG_DBG("%s: key_offset = %zu, slot_max = %d, key_num = %d, draft.size = %zu\n", __func__, vKeyOffset, vSlotMax,
          vCurrKey.vKeyNum, pDraft.size());

  pMap.vLastDraftCreated = true;
  pMap.vLastDraftKeyIdx = vKeyOffset;
  pMap.vLastDraftValueIdx = vSlotMax; // value used for draft generation.
}

void fCommonNgramMapAccept(common_ngram_map &pMap, uint16_t pNAccepted) {
  if (!pMap.vLastDraftCreated) {
    return;
  }

  // find the key and its chosen value.
  const size_t cKeyIdx = pMap.vLastDraftKeyIdx;
  const size_t cValIdx = pMap.vLastDraftValueIdx;

  // find key corresponding to key_idx.
  common_ngram_map_key &vCurrKey = pMap.lKeys[cKeyIdx];
  // find value corresponding to val_idx.
  struct common_ngram_map_value &vCurrValue = vCurrKey.aValues[cValIdx]; // value used for draft generation.

  // update the value statistics
  LOG_DBG("common_ngram_map_send_accepted: n_accepted = %d, prev value_num = %d\n", pNAccepted, vCurrValue.vNAccepted);
  vCurrValue.vNAccepted = pNAccepted;
}
