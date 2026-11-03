#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
//

// basic n-gram hasher
struct common_ngram_mod {
  using entry_t = int32_t;

  static constexpr entry_t cEmpty = -1;

  common_ngram_mod(uint16_t pN, size_t pSize);

  size_t mIdx(const entry_t *pTokens) const;
  void mAdd(const entry_t *pTokens);
  entry_t mGet(const entry_t *pTokens) const; // return -1 if not found

  void mReset();

  size_t mGetN() const;
  size_t mGetUsed() const;

  size_t size() const;
  size_t mSizeBytes() const;

private:
  size_t vN; // ngram size to hash

  size_t vUsed;

  std::vector<entry_t> lEntries;
};
