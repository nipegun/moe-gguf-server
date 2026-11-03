#pragma once

#include <cstdint>
#include <string>
#include <vector>

// TODO: reimplement this structure in endian-independent way
struct unicode_cpt_flags {
  enum {
    cUndefined = 0x0001,
    cNumber = 0x0002,      // regex: \p{N}
    cLetter = 0x0004,      // regex: \p{L}
    cSeparator = 0x0008,   // regex: \p{Z}
    cAccentMark = 0x0010,  // regex: \p{M}
    cPunctuation = 0x0020, // regex: \p{P}
    cSymbol = 0x0040,      // regex: \p{S}
    cControl = 0x0080,     // regex: \p{C}
    cMaskCategories = 0x00FF,
    cWhitespace = 0x0100,
    cLowercase = 0x0200,
    cUppercase = 0x0400,
    cNfd = 0x0800,
  };

  // codepoint type
  uint16_t vIsUndefined : 1;
  uint16_t vIsNumber : 1;      // regex: \p{N}
  uint16_t vIsLetter : 1;      // regex: \p{L}
  uint16_t vIsSeparator : 1;   // regex: \p{Z}
  uint16_t vIsAccentMark : 1;  // regex: \p{M}
  uint16_t vIsPunctuation : 1; // regex: \p{P}
  uint16_t vIsSymbol : 1;      // regex: \p{S}
  uint16_t vIsControl : 1;     // regex: \p{C}
  // helper flags
  uint16_t vIsWhitespace : 1; // regex: \s
  uint16_t vIsLowercase : 1;
  uint16_t vIsUppercase : 1;
  uint16_t vIsNfd : 1;

  // decode from uint16
  inline unicode_cpt_flags(const uint16_t pFlags = 0) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    *reinterpret_cast<uint16_t *>(this) = pFlags;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    vIsUndefined = (pFlags & cUndefined) ? 1 : 0;
    vIsNumber = (pFlags & cNumber) ? 1 : 0;
    vIsLetter = (pFlags & cLetter) ? 1 : 0;
    vIsSeparator = (pFlags & cSeparator) ? 1 : 0;
    vIsAccentMark = (pFlags & cAccentMark) ? 1 : 0;
    vIsPunctuation = (pFlags & cPunctuation) ? 1 : 0;
    vIsSymbol = (pFlags & cSymbol) ? 1 : 0;
    vIsControl = (pFlags & cControl) ? 1 : 0;
    vIsWhitespace = (pFlags & cWhitespace) ? 1 : 0;
    vIsLowercase = (pFlags & cLowercase) ? 1 : 0;
    vIsUppercase = (pFlags & cUppercase) ? 1 : 0;
    vIsNfd = (pFlags & cNfd) ? 1 : 0;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
  }

  inline uint16_t mAsUint() const {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return *reinterpret_cast<const uint16_t *>(this);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint16_t vResult = vIsUndefined * cUndefined + vIsNumber * cNumber + vIsLetter * cLetter +
                       vIsSeparator * cSeparator + vIsAccentMark * cAccentMark + vIsPunctuation * cPunctuation +
                       vIsSymbol * cSymbol + vIsControl * cControl + vIsWhitespace * cWhitespace +
                       vIsLowercase * cLowercase + vIsUppercase * cUppercase + vIsNfd * cNfd;

    return vResult;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
  }

  inline uint16_t mCategoryFlag() const { return this->mAsUint() & cMaskCategories; }
};

size_t fUnicodeLenUtf8(char pSrc);

std::string fUnicodeCptToUtf8(uint32_t pCpt);
uint32_t fUnicodeCptFromUtf8(const std::string &pUtf8, size_t &pOffset);

std::vector<uint32_t> fUnicodeCptsFromUtf8(const std::string &pUtf8);

std::vector<uint32_t> fUnicodeCptsNormalizeNfd(const std::vector<uint32_t> &pCpts);

unicode_cpt_flags fUnicodeCptFlagsFromCpt(uint32_t pCpt);
unicode_cpt_flags fUnicodeCptFlagsFromUtf8(const std::string &pUtf8);

std::string fUnicodeByteToUtf8(uint8_t pByte);
uint8_t fUnicodeUtf8ToByte(const std::string &pUtf8);

uint32_t fUnicodeTolower(uint32_t pCpt);

bool fUnicodeCptIsHan(uint32_t pCpt);

std::vector<std::string> fUnicodeRegexSplit(const std::string &pText, const std::vector<std::string> &pRegexExprs,
                                            bool pByteEncode = true);
