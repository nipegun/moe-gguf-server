#pragma once

#include "mgs.h"

#include <unordered_map>
#include <string>
#include <vector>

#define MGS_NGRAM_MIN 1
#define MGS_NGRAM_MAX 4
#define MGS_NGRAM_STATIC 2

// Data structures to map n-grams to empirical token probabilities:

struct common_ngram {
  mgs_token aTokens[MGS_NGRAM_MAX];

  common_ngram() {
    for (int vI = 0; vI < MGS_NGRAM_MAX; ++vI) {
      aTokens[vI] = MGS_TOKEN_NULL;
    }
  }

  common_ngram(const mgs_token *pInput, const int pNgramSize) {
    for (int vI = 0; vI < MGS_NGRAM_MAX; ++vI) {
      aTokens[vI] = vI < pNgramSize ? pInput[vI] : MGS_TOKEN_NULL;
    }
  }

  bool operator==(const common_ngram &pOther) const {
    for (int vI = 0; vI < MGS_NGRAM_MAX; ++vI) {
      if (aTokens[vI] != pOther.aTokens[vI]) {
        return false;
      }
    }
    return true;
  }
};

struct common_token_hash_function {
  size_t operator()(const mgs_token pToken) const {
    // see
    // https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
    return pToken * 11400714819323198485llu;
  }
};

struct common_ngram_hash_function {
  size_t operator()(const common_ngram &pNgram) const {
    size_t vHash = common_token_hash_function{}(pNgram.aTokens[0]);
    for (int vI = 1; vI < MGS_NGRAM_MAX; ++vI) {
      vHash ^= common_token_hash_function{}(pNgram.aTokens[vI]);
    }
    return vHash;
  }
};

// token -> number of times token has been seen
typedef std::unordered_map<mgs_token, int32_t> common_ngram_cache_part;

// n-gram -> empirical distribution of following tokens
typedef std::unordered_map<common_ngram, common_ngram_cache_part, common_ngram_hash_function> common_ngram_cache;

// Update an ngram cache with tokens.
// ngram_cache:         the cache to modify.
// ngram_min/ngram_max: the min/max size of the ngrams to extract from inp_data.
// inp_data:            the token sequence with which to update ngram_cache.
// nnew:                how many new tokens have been appended to inp_data since the last call to this function.
// print_progress:      whether to print progress to stderr.
//
// In order to get correct results inp_data can ONLY BE APPENDED TO.
// Changes in the middle need a complete rebuild.
void fCommonNgramCacheUpdate(common_ngram_cache &pNgramCache, int pNgramMin, int pNgramMax,
                             std::vector<mgs_token> &pInpData, int pNnew, bool pPrintProgress);

// Try to draft tokens from ngram caches.
// inp:                the tokens generated so far.
// draft:              the token sequence to draft. Expected to initially contain the previously sampled token.
// n_draft:            maximum number of tokens to add to draft.
// ngram_min/gram_max: the min/max size of the ngrams in nc_context and nc_dynamic.
// nc_context:         ngram cache based on current context.
// nc_dynamic:         ngram cache based on previous user generations.
// nc_static:          ngram cache generated from a large text corpus, used for validation.
void fCommonNgramCacheDraft(std::vector<mgs_token> &pInp, std::vector<mgs_token> &pDraft, int pNDraft, int pNgramMin,
                            int pNgramMax, common_ngram_cache &pNcContext, common_ngram_cache &pNcDynamic,
                            common_ngram_cache &pNcStatic);

// Save an ngram cache to a file.
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void fCommonNgramCacheSave(common_ngram_cache &pNgramCache, const std::string &pFilename);

// Load an ngram cache saved with fCommonNgramCacheSave.
// filename: the path from which to load the ngram cache.
// returns:  an ngram cache containing the information saved to filename.
common_ngram_cache fCommonNgramCacheLoad(const std::string &pFilename);

// Merge two ngram caches.
// ngram_cache_target: the ngram cache to which to add the information from ngram_cache_add.
// ngram_cache_add:    the ngram cache to add to ngram_cache_target.
void fCommonNgramCacheMerge(common_ngram_cache &pNgramCacheTarget, common_ngram_cache &pNgramCacheAdd);
