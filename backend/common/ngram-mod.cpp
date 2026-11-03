#include "ngram-mod.h"

#include <algorithm>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t pN, size_t pSize) : vN(pN), vUsed(0) {
  lEntries.resize(pSize);

  mReset();
}

size_t common_ngram_mod::mIdx(const entry_t *pTokens) const {
  size_t vRes = 0;

  for (size_t vI = 0; vI < vN; ++vI) {
    vRes = vRes * 6364136223846793005ULL + pTokens[vI];
  }

  vRes = vRes % lEntries.size();

  return vRes;
}

void common_ngram_mod::mAdd(const entry_t *pTokens) {
  const size_t cI = mIdx(pTokens);

  if (lEntries[cI] == cEmpty) {
    vUsed++;
  }

  lEntries[cI] = pTokens[vN];
}

common_ngram_mod::entry_t common_ngram_mod::mGet(const entry_t *pTokens) const {
  const size_t cI = mIdx(pTokens);

  return lEntries[cI];
}

void common_ngram_mod::mReset() {
  std::fill(lEntries.begin(), lEntries.end(), cEmpty);
  vUsed = 0;
}

size_t common_ngram_mod::mGetN() const { return vN; }

size_t common_ngram_mod::mGetUsed() const { return vUsed; }

size_t common_ngram_mod::size() const { return lEntries.size(); }

size_t common_ngram_mod::mSizeBytes() const { return lEntries.size() * sizeof(lEntries[0]); }
