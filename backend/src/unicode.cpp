#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t fUnicodeLenUtf8(char pSrc) {
  const size_t aLookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t vHighbits = static_cast<uint8_t>(pSrc) >> 4;
  return aLookup[vHighbits];
}

static std::string fUnicodeCptsToUtf8(const std::vector<uint32_t> &pCps) {
  std::string vResult;
  for (size_t vI = 0; vI < pCps.size(); ++vI) {
    vResult.append(fUnicodeCptToUtf8(pCps[vI]));
  }
  return vResult;
}

uint32_t fUnicodeCptFromUtf8(const std::string &pUtf8, size_t &pOffset) {
  assert(pOffset < pUtf8.size());
  if (!(pUtf8[pOffset + 0] & 0x80)) {
    auto vResult = pUtf8[pOffset + 0];
    pOffset += 1;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x40)) {
    throw std::invalid_argument("invalid character");
  }
  if (!(pUtf8[pOffset + 0] & 0x20)) {
    if (pOffset + 1 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult = ((pUtf8[pOffset + 0] & 0x1f) << 6) | (pUtf8[pOffset + 1] & 0x3f);
    pOffset += 2;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x10)) {
    if (pOffset + 2 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80) ||
        !((pUtf8[pOffset + 2] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult =
        ((pUtf8[pOffset + 0] & 0x0f) << 12) | ((pUtf8[pOffset + 1] & 0x3f) << 6) | (pUtf8[pOffset + 2] & 0x3f);
    pOffset += 3;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x08)) {
    if (pOffset + 3 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80) ||
        !((pUtf8[pOffset + 2] & 0xc0) == 0x80) || !((pUtf8[pOffset + 3] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult = ((pUtf8[pOffset + 0] & 0x07) << 18) | ((pUtf8[pOffset + 1] & 0x3f) << 12) |
                   ((pUtf8[pOffset + 2] & 0x3f) << 6) | (pUtf8[pOffset + 3] & 0x3f);
    pOffset += 4;
    return vResult;
  }
  throw std::invalid_argument("failed to convert utf8 to codepoint");
}

// static std::vector<uint16_t> fUnicodeCptToUtf16(uint32_t pCpt) {
//     std::vector<uint16_t> lResult;
//     if (/* 0x0000 <= pCpt && */ pCpt <= 0xffff) {
//         lResult.emplace_back(pCpt);
//         return lResult;
//     }
//     if (0x10000 <= pCpt && pCpt <= 0x10ffff) {
//         lResult.emplace_back(0xd800 | ((pCpt - 0x10000) >> 10));
//         lResult.emplace_back(0xdc00 | ((pCpt - 0x10000) & 0x03ff));
//         return lResult;
//     }
//     throw std::invalid_argument("failed to convert codepoint to utf16");
// }

// static std::vector<uint16_t> fUnicodeCptsToUtf16(const std::vector<uint32_t> & pCps) {
//     std::vector<uint16_t> lResult;
//     for (size_t vI = 0; vI < pCps.size(); ++vI) {
//         auto lTemp = fUnicodeCptToUtf16(pCps[vI]);
//         lResult.insert(lResult.end(), lTemp.begin(), lTemp.end());
//     }
//     return lResult;
// }

// static uint32_t fUnicodeCptFromUtf16(const std::vector<uint16_t> & pUtf16, size_t & pOffset) {
//     assert(pOffset < pUtf16.size());
//     if (((pUtf16[0] >> 10) << 10) != 0xd800) {
//         auto vResult = pUtf16[pOffset + 0];
//         pOffset += 1;
//         return vResult;
//     }
//
//     if (pOffset + 1 >= pUtf16.size() || !((pUtf16[1] & 0xdc00) == 0xdc00)) {
//         throw std::invalid_argument("invalid character");
//     }
//
//     auto vResult = 0x10000 + (((pUtf16[0] & 0x03ff) << 10) | (pUtf16[1] & 0x03ff));
//     pOffset += 2;
//     return vResult;
// }

// static std::vector<uint32_t> fUnicodeCptsFromUtf16(const std::vector<uint16_t> & pUtf16) {
//     std::vector<uint32_t> lResult;
//     size_t vOffset = 0;
//     while (vOffset < pUtf16.size()) {
//         lResult.push_back(fUnicodeCptFromUtf16(pUtf16, vOffset));
//     }
//     return lResult;
// }

static std::vector<unicode_cpt_flags> fUnicodeCptFlagsArray() {
  std::vector<unicode_cpt_flags> lCptFlags(cMaxCodepoints, unicode_cpt_flags::cUndefined);

  assert(lUnicodeRangesFlags.begin()[0].first == 0);
  assert(lUnicodeRangesFlags.begin()[lUnicodeRangesFlags.size() - 1].first == cMaxCodepoints);
  for (size_t vI = 1; vI < lUnicodeRangesFlags.size(); ++vI) {
    const auto cRangeIni = lUnicodeRangesFlags.begin()[vI - 1]; // codepoint_ini, flags
    const auto cRangeEnd = lUnicodeRangesFlags.begin()[vI];     // codepoint_end, flags
    for (uint32_t vCpt = cRangeIni.first; vCpt < cRangeEnd.first; ++vCpt) {
      lCptFlags[vCpt] = cRangeIni.second;
    }
  }

  for (auto vCpt : sUnicodeSetWhitespace) {
    lCptFlags[vCpt].vIsWhitespace = true;
  }

  for (auto vP : lUnicodeMapLowercase) {
    lCptFlags[vP.second].vIsLowercase = true;
  }

  for (auto vP : lUnicodeMapUppercase) {
    lCptFlags[vP.second].vIsUppercase = true;
  }

  for (auto &vRange : lUnicodeRangesNfd) { // start, last, nfd
    lCptFlags[vRange.vNfd].vIsNfd = true;
  }

  return lCptFlags;
}

static std::unordered_map<uint8_t, std::string> fUnicodeByteToUtf8Map() {
  std::unordered_map<uint8_t, std::string> dMap;
  for (int vCh = 0x21; vCh <= 0x7E; ++vCh) { // u'!' to u'~'
    assert(0 <= vCh && vCh < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  for (int vCh = 0xA1; vCh <= 0xAC; ++vCh) { // u'¡' to u'¬'
    assert(0 <= vCh && vCh < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  for (int vCh = 0xAE; vCh <= 0xFF; ++vCh) { // u'®' to u'ÿ'
    assert(0 <= vCh && vCh < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  auto vN = 0;
  for (int vCh = 0; vCh < 256; ++vCh) {
    if (dMap.find(vCh) == dMap.end()) {
      dMap[vCh] = fUnicodeCptToUtf8(256 + vN);
      ++vN;
    }
  }
  return dMap;
}

static std::unordered_map<std::string, uint8_t> fUnicodeUtf8ToByteMap() {
  std::unordered_map<std::string, uint8_t> dMap;
  for (int vCh = 0x21; vCh <= 0x7E; ++vCh) { // u'!' to u'~'
    assert(0 <= vCh && vCh < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  for (int vCh = 0xA1; vCh <= 0xAC; ++vCh) { // u'¡' to u'¬'
    assert(0 <= vCh && vCh < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  for (int vCh = 0xAE; vCh <= 0xFF; ++vCh) { // u'®' to u'ÿ'
    assert(0 <= vCh && vCh < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  auto vN = 0;
  for (int vCh = 0; vCh < 256; ++vCh) {
    if (dMap.find(fUnicodeCptToUtf8(vCh)) == dMap.end()) {
      dMap[fUnicodeCptToUtf8(256 + vN)] = vCh;
      ++vN;
    }
  }
  return dMap;
}

static std::vector<std::string> fUnicodeByteEncodingProcess(const std::vector<std::string> &pBpeWords) {
  std::vector<std::string> lBpeEncodedWords;
  for (const auto &vWord : pBpeWords) {
    std::string vTextUtf;
    auto vUtfWord = fUnicodeCptsFromUtf8(vWord);
    for (size_t vI = 0; vI < vUtfWord.size(); ++vI) {
      vTextUtf += fUnicodeCptToUtf8(vUtfWord[vI]);
    }

    std::string vEncodedToken;
    for (char &vC : vTextUtf) {
      vEncodedToken += fUnicodeByteToUtf8(vC);
    }
    lBpeEncodedWords.emplace_back(vEncodedToken);
  }
  return lBpeEncodedWords;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> fUnicodeRegexSplitCustomGpt2(const std::string &pText, const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;      // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size()); // Reserve memory for the approximate size

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto vGetCpt = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? cCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      // if (vLen > 0) {
      //     std::string vS = "";
      //     for(size_t vP = pEnd-vLen; vP < pEnd; vP++)
      //         vS += fUnicodeCptToUtf8(cCpts[vP]);
      //     printf(">>> '%s'\n", vS.c_str());
      // }
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt = vGetCpt(vPos);
      const auto cFlags = vGetFlags(vPos);

      // regex: 's|'t|'re|'ve|'m|'ll|'d
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = vGetCpt(vPos + 1);
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = vGetCpt(vPos + 2);
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      // regex: <space>?\p{L}+
      if (vFlags2.vIsLetter) {
        vPos += (cCpt == ' ');
        while (vFlags2.vIsLetter) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }
      // regex: <space>?\p{N}+
      if (vFlags2.vIsNumber) {
        vPos += (cCpt == ' ');
        while (vFlags2.vIsNumber) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }
      // regex: <space>?[^\s\p{L}\p{N}]+
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && vFlags2.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        vNumWhitespaces++;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> fUnicodeRegexSplitCustomLlama3(const std::string &pText,
                                                          const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;      // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size()); // Reserve memory for the approximate size

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto vGetCpt = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? cCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      // if (vLen > 0) {
      //     std::string vS = "";
      //     for(size_t vP = pEnd-vLen; vP < pEnd; vP++)
      //         vS += fUnicodeCptToUtf8(cCpts[vP]);
      //     printf(">>> '%s'\n", vS.c_str());
      // }
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt = vGetCpt(vPos);
      const auto cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsNumber)) {
        if (cFlags.vIsLetter || vGetFlags(vPos + 1).vIsLetter) { // one or more letters
          vPos++;
          while (vGetFlags(vPos).vIsLetter) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}{1,3}
      if (cFlags.vIsNumber) {
        size_t vIni = vPos;
        while (vGetFlags(vPos).vIsNumber) {
          if (++vPos - vIni >= 3) {
            vAddToken(vPos);
            vIni = vPos;
          }
        }
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|
// ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
static std::vector<size_t> fUnicodeRegexSplitCustomQwen2(const std::string &pText,
                                                         const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;      // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size()); // Reserve memory for the approximate size

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto vGetCpt = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? cCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      // if (vLen > 0) {
      //     std::string vS = "";
      //     for(size_t vP = pEnd-vLen; vP < pEnd; vP++)
      //         vS += fUnicodeCptToUtf8(cCpts[vP]);
      //     printf(">>> '%s'\n", vS.c_str());
      // }
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt = vGetCpt(vPos);
      const auto cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsNumber)) {
        if (cFlags.vIsLetter || vGetFlags(vPos + 1).vIsLetter) { // one or more letters
          vPos++;
          while (vGetFlags(vPos).vIsLetter) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}
      if (cFlags.vIsNumber) {
        vPos++;
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsNumber) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// Qwen3.5 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}|
// ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" Compared to Qwen2, letter-runs also consume
// Unicode combining marks (\p{M}): [\p{L}\p{M}]+ instead of \p{L}+
static std::vector<size_t> fUnicodeRegexSplitCustomQwen35(const std::string &pText,
                                                          const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;      // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size()); // Reserve memory for the approximate size

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto vGetCpt = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? cCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt = vGetCpt(vPos);
      const auto cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsNumber)) {
        if (cFlags.vIsLetter || cFlags.vIsAccentMark || vGetFlags(vPos + 1).vIsAccentMark ||
            vGetFlags(vPos + 1).vIsLetter) {
          vPos++;
          while (vGetFlags(vPos).vIsLetter || vGetFlags(vPos).vIsAccentMark) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}
      if (cFlags.vIsNumber) {
        vPos++;
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsAccentMark | vFlags2.vIsNumber) &&
          cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsAccentMark | vFlags2.vIsNumber) &&
               vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

template <typename CharT>
static std::vector<size_t> fUnicodeRegexSplitStl(const std::basic_string<CharT> &pText,
                                                 const std::basic_string<CharT> &pRegex,
                                                 const std::vector<size_t> &pOffsets) {
  using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
  // Bypass bug in MSVC
  constexpr auto cRegexFlags = std::regex_constants::ECMAScript;
#else
  constexpr auto cRegexFlags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
  std::basic_regex<CharT> vExpr(pRegex, cRegexFlags);
  std::vector<size_t> lBpeOffsets;      // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size()); // Reserve memory for the approximate size
  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    std::regex_iterator<BidirIt> vIt(pText.begin() + vStart, pText.begin() + vStart + vOffset, vExpr);
    std::regex_iterator<BidirIt> vEnd;

    int64_t vStartIdx = 0;
    while (vIt != vEnd) {
      std::match_results<BidirIt> vMatch = *vIt;
      if (vMatch.position() > vStartIdx) {
        lBpeOffsets.emplace_back(vMatch.position() - vStartIdx);
      }
      lBpeOffsets.emplace_back(vMatch.length());
      vStartIdx = vMatch.position() + vMatch.length();
      ++vIt;
    }

    if (vStartIdx < (int64_t)vOffset) {
      lBpeOffsets.emplace_back(vOffset - vStartIdx);
    }
    vStart += vOffset;
  }

  return lBpeOffsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}|
// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> fUnicodeRegexSplitCustomKimiK2(const std::string &pText,
                                                          const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto vGetCpt = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? cCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt = vGetCpt(vPos);
      const auto cFlags = vGetFlags(vPos);

      // Pattern 1: [\p{Han}]+ (Chinese characters)
      if (fUnicodeCptIsHan(cCpt)) {
        while (fUnicodeCptIsHan(vGetCpt(vPos))) {
          vPos++;
        }
        vAddToken(vPos);
        continue;
      }

      // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
      // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
      // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
      // Check if current char is a letter OR if current char could be a leading char and next char is a letter
      bool vIsLetterPattern = (cFlags.vIsLetter && !fUnicodeCptIsHan(cCpt)) ||
                              (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsLetter || cFlags.vIsNumber) &&
                               vGetFlags(vPos + 1).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos + 1)));

      if (vIsLetterPattern) {
        // Handle optional leading non-letter/non-number character
        bool vHasLeadingChar = false;
        if (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsLetter || cFlags.vIsNumber)) {
          vHasLeadingChar = true;
          vPos++;
        }

        // Match letter sequence (excluding Han characters)
        bool vHasLetters = false;
        while (vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos))) {
          vHasLetters = true;
          vPos++;
        }

        // Only proceed if we found letters (after potentially skipping leading char)
        if (vHasLetters || (!vHasLeadingChar && vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos)))) {
          if (!vHasLetters)
            vPos++; // consume the first letter if we didn't already

          // Continue consuming letters
          while (vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos))) {
            vPos++;
          }

          // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
          if (vGetCpt(vPos) == '\'' && vPos + 1 < cOffsetEnd) {
            uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
            if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
              vPos += 2;
            } else if (vPos + 2 < cOffsetEnd) {
              uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
              if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
                  (vCptNext == 'l' && vCptNextNext == 'l')) {
                vPos += 3;
              }
            }
          }

          vAddToken(vPos);
          continue;
        } else if (vHasLeadingChar) {
          // We consumed a leading char but found no letters, backtrack
          vPos--;
        }
      }

      // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
      if (cFlags.vIsNumber) {
        size_t vIni = vPos;
        while (vGetFlags(vPos).vIsNumber) {
          if (++vPos - vIni >= 3) {
            vAddToken(vPos);
            vIni = vPos;
          }
        }
        vAddToken(vPos);
        continue;
      }

      // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace || vFlags2.vIsLetter || vFlags2.vIsNumber) && vFlags2.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace || vFlags2.vIsLetter || vFlags2.vIsNumber) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        // Match optional [\r\n]*
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      // Count whitespace characters
      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // Pattern 7: \s+(?!\S) (trailing whitespace)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // Pattern 8: \s+ (general whitespace)
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // No matches - consume single character
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// AFMOE digit handling: splits digits with leading 1-2 based on total length modulo 3
static std::vector<size_t> fUnicodeRegexSplitCustomAfmoe(const std::string &pText,
                                                         const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(cCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd = cOffsetIni;
    auto vAddToken = [&](const size_t pEnd) -> size_t {
      assert(vPrevEnd <= pEnd && pEnd <= cOffsetEnd);
      size_t vLen = pEnd - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = pEnd;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd;) {
      const auto cFlags = vGetFlags(vPos);

      // Handle digit sequences with special splitting logic
      if (cFlags.vIsNumber) {
        size_t vDigitStart = vPos;
        size_t vDigitCount = 0;

        // Count consecutive digits
        while (vGetFlags(vPos).vIsNumber && vPos < cOffsetEnd) {
          vDigitCount++;
          vPos++;
        }

        // Split based on total length modulo 3
        size_t vRemainder = vDigitCount % 3;
        size_t vCurrent = vDigitStart;

        // Emit leading 1-2 digits if needed
        if (vRemainder > 0) {
          vAddToken(vCurrent + vRemainder);
          vCurrent += vRemainder;
        }

        // Emit groups of 3
        while (vCurrent < vDigitStart + vDigitCount) {
          vAddToken(vCurrent + 3);
          vCurrent += 3;
        }
        continue;
      }

      // For non-digits, just move forward
      vPos++;
    }

    // Add any remaining content
    if (vPrevEnd < cOffsetEnd) {
      vAddToken(cOffsetEnd);
    }
  }

  return lBpeOffsets;
}

// regex: [^\n]+|[\n]+
// splits text into runs of non-newline characters and runs of newline characters
static std::vector<size_t> fUnicodeRegexSplitCustomNewlines(const std::string &pText,
                                                            const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(cOffsetEnd <= cCpts.size());
    vStart = cOffsetEnd;

    size_t vPos = cOffsetIni;
    while (vPos < cOffsetEnd) {
      const bool cIsNewline = (cCpts[vPos] == '\n');
      const size_t cRunStart = vPos;
      while (vPos < cOffsetEnd && (cCpts[vPos] == '\n') == cIsNewline) {
        vPos++;
      }
      lBpeOffsets.push_back(vPos - cRunStart);
    }
  }

  return lBpeOffsets;
}

static std::vector<size_t> fUnicodeRegexSplitCustom(const std::string &pText, const std::string &pRegexExpr,
                                                    const std::vector<size_t> &pOffsets) {
  std::vector<size_t> lBpeOffsets;

  if (pRegexExpr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
    lBpeOffsets = fUnicodeRegexSplitCustomGpt2(pText, pOffsets);
  } else if (pRegexExpr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
                           "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
             pRegexExpr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
                           "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomLlama3(pText, pOffsets);
  } else if (pRegexExpr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
                           "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomQwen2(pText, pOffsets);
  } else if (pRegexExpr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{"
                           "M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomQwen35(pText, pOffsets);
  } else if (pRegexExpr == "\\p{Han}+") {
    // K2's first pattern - handle all K2 patterns together
    lBpeOffsets = fUnicodeRegexSplitCustomKimiK2(pText, pOffsets);
  } else if (pRegexExpr == "\\p{AFMoE_digits}") {
    // AFMOE digit pattern - use custom implementation for proper splitting
    lBpeOffsets = fUnicodeRegexSplitCustomAfmoe(pText, pOffsets);
  } else if (pRegexExpr == "[^\\n]+|[\\n]+") {
    lBpeOffsets = fUnicodeRegexSplitCustomNewlines(pText, pOffsets);
  } else if (pRegexExpr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
    // tiny_aya digit grouping pattern from tokenizer.json:
    //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
    // Splits digits into groups of 3 from the right (e.g., 1234567 -> 1, 234, 567)
    // TODO: Revisit this regex, in case there are any subtle tokenization differences with the original regex.
    lBpeOffsets = fUnicodeRegexSplitCustomAfmoe(pText, pOffsets);
  }

  return lBpeOffsets;
}

//
// interface
//

std::string fUnicodeCptToUtf8(uint32_t pCpt) {
  std::string vResult;

  if (/* 0x00 <= cpt && */ pCpt <= 0x7f) {
    vResult.push_back(pCpt);
    return vResult;
  }
  if (0x80 <= pCpt && pCpt <= 0x7ff) {
    vResult.push_back(0xc0 | ((pCpt >> 6) & 0x1f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }
  if (0x800 <= pCpt && pCpt <= 0xffff) {
    vResult.push_back(0xe0 | ((pCpt >> 12) & 0x0f));
    vResult.push_back(0x80 | ((pCpt >> 6) & 0x3f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }
  if (0x10000 <= pCpt && pCpt <= 0x10ffff) {
    vResult.push_back(0xf0 | ((pCpt >> 18) & 0x07));
    vResult.push_back(0x80 | ((pCpt >> 12) & 0x3f));
    vResult.push_back(0x80 | ((pCpt >> 6) & 0x3f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }

  throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> fUnicodeCptsNormalizeNfd(const std::vector<uint32_t> &pCpts) {
  auto vComp = [](const uint32_t pCpt, const range_nfd &pRange) { return pCpt < pRange.vFirst; };
  std::vector<uint32_t> lResult(pCpts.size());
  for (size_t vI = 0; vI < pCpts.size(); ++vI) {
    const uint32_t cCpt = pCpts[vI];
    auto vIt = std::upper_bound(lUnicodeRangesNfd.begin(), lUnicodeRangesNfd.end(), cCpt, vComp) - 1;
    lResult[vI] = (vIt->vFirst <= cCpt && cCpt <= vIt->vLast) ? vIt->vNfd : cCpt;
  }
  return lResult;
}

std::vector<uint32_t> fUnicodeCptsFromUtf8(const std::string &pUtf8) {
  std::vector<uint32_t> lResult;
  lResult.reserve(pUtf8.size());
  size_t vOffset = 0;
  while (vOffset < pUtf8.size()) {
    try {
      lResult.push_back(fUnicodeCptFromUtf8(pUtf8, vOffset));
    } catch (const std::invalid_argument & /*ex*/) {
      // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond mgs_tokenize
      ++vOffset;
      lResult.emplace_back(0xFFFD); // replacement character
    }
  }
  return lResult;
}

unicode_cpt_flags fUnicodeCptFlagsFromCpt(const uint32_t pCpt) {
  static const unicode_cpt_flags cUndef(unicode_cpt_flags::cUndefined);
  static const auto cCptFlags = fUnicodeCptFlagsArray();
  return pCpt < cCptFlags.size() ? cCptFlags[pCpt] : cUndef;
}

unicode_cpt_flags fUnicodeCptFlagsFromUtf8(const std::string &pUtf8) {
  static const unicode_cpt_flags cUndef(unicode_cpt_flags::cUndefined);
  if (pUtf8.empty()) {
    return cUndef; // undefined
  }
  size_t vOffset = 0;
  return fUnicodeCptFlagsFromCpt(fUnicodeCptFromUtf8(pUtf8, vOffset));
}

std::string fUnicodeByteToUtf8(uint8_t pByte) {
  static std::unordered_map<uint8_t, std::string> dMap = fUnicodeByteToUtf8Map();
  return dMap.at(pByte);
}

uint8_t fUnicodeUtf8ToByte(const std::string &pUtf8) {
  static std::unordered_map<std::string, uint8_t> dMap = fUnicodeUtf8ToByteMap();
  return dMap.at(pUtf8);
}

uint32_t fUnicodeTolower(uint32_t pCpt) {
  // binary search
  auto vIt = std::lower_bound(
      lUnicodeMapLowercase.begin(), lUnicodeMapLowercase.end(), pCpt,
      [](const std::pair<uint32_t, uint32_t> &pPair, uint32_t pValue) { return pPair.first < pValue; });
  if (vIt != lUnicodeMapLowercase.end() && vIt->first == pCpt) {
    return vIt->second;
  }
  return pCpt; // Return the original code point if no lowercase mapping is found
}

bool fUnicodeCptIsHan(uint32_t pCpt) {
  // Han character ranges (Chinese/CJK characters)
  // CJK Unified Ideographs (most common)
  if (pCpt >= 0x4E00 && pCpt <= 0x9FFF)
    return true;

  // CJK Extension A
  if (pCpt >= 0x3400 && pCpt <= 0x4DBF)
    return true;

  // CJK Extension B
  if (pCpt >= 0x20000 && pCpt <= 0x2A6DF)
    return true;

  // CJK Extension C
  if (pCpt >= 0x2A700 && pCpt <= 0x2B73F)
    return true;

  // CJK Extension D
  if (pCpt >= 0x2B740 && pCpt <= 0x2B81F)
    return true;

  // CJK Extension E
  if (pCpt >= 0x2B820 && pCpt <= 0x2CEAF)
    return true;

  // CJK Extension F
  if (pCpt >= 0x2CEB0 && pCpt <= 0x2EBEF)
    return true;

  // CJK Compatibility Ideographs
  if (pCpt >= 0xF900 && pCpt <= 0xFAFF)
    return true;

  // CJK Compatibility Ideographs Supplement
  if (pCpt >= 0x2F800 && pCpt <= 0x2FA1F)
    return true;

  return false;
}

std::vector<std::string> fUnicodeRegexSplit(const std::string &pText, const std::vector<std::string> &pRegexExprs,
                                            bool pByteEncode) {
  // unicode categories
  static const std::map<std::string, int> dKUcatEnum = {
      {"\\p{N}", unicode_cpt_flags::cNumber},      {"\\p{L}", unicode_cpt_flags::cLetter},
      {"\\p{P}", unicode_cpt_flags::cPunctuation}, {"\\p{M}", unicode_cpt_flags::cAccentMark},
      {"\\p{S}", unicode_cpt_flags::cSymbol},      {"\\p{Lu}", unicode_cpt_flags::cLetter}, // Uppercase letter
      {"\\p{Ll}", unicode_cpt_flags::cLetter},                                              // Lowercase letter
      {"\\p{Lt}", unicode_cpt_flags::cLetter},                                              // Titlecase letter
      {"\\p{Lm}", unicode_cpt_flags::cLetter},                                              // Modifier letter
      {"\\p{Lo}", unicode_cpt_flags::cLetter},                                              // Other letter
  };

  static const std::map<int, int> dKUcatCpt = {
      {unicode_cpt_flags::cNumber, 0xD1},      {unicode_cpt_flags::cLetter, 0xD2},
      {unicode_cpt_flags::cPunctuation, 0xD3}, {unicode_cpt_flags::cAccentMark, 0xD4},
      {unicode_cpt_flags::cSymbol, 0xD5},
  };

  static const std::map<int, std::string> dKUcatMap = {
      {unicode_cpt_flags::cNumber, "\x30-\x39"},          // 0-9
      {unicode_cpt_flags::cLetter, "\x41-\x5A\x61-\x7A"}, // A-Za-z
      {unicode_cpt_flags::cPunctuation,
       "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"}, // !-#%-*,-/:-;?-@\[-\]_\{\}
      {unicode_cpt_flags::cAccentMark, ""},                                           // no sub-128 codepoints
      {unicode_cpt_flags::cSymbol, "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C"},            // $+<=>^`|
  };

  // compute collapsed codepoints only if needed by at least one regex
  bool vNeedCollapse = false;
  for (const auto &vRegexExpr : pRegexExprs) {
    // search for unicode categories
    for (const auto &vUcat : dKUcatEnum) {
      if (std::string::npos != vRegexExpr.find(vUcat.first)) {
        vNeedCollapse = true;
        break;
      }
    }
  }

  const auto cCpts = fUnicodeCptsFromUtf8(pText);

  // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
  std::string vTextCollapsed;
  if (vNeedCollapse) {
    // collapse all unicode categories
    vTextCollapsed.resize(cCpts.size());

    for (size_t vI = 0; vI < cCpts.size(); ++vI) {
      // keep single-byte codepoints as is
      if (cCpts[vI] < 128) {
        vTextCollapsed[vI] = cCpts[vI];
        continue;
      }

      const auto cFlags = fUnicodeCptFlagsFromCpt(cCpts[vI]);

      if (cFlags.vIsWhitespace) {
        // NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
        // text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
        vTextCollapsed[vI] = (char)0x0B; // <vertical tab> as whitespace fallback
      } else if (dKUcatCpt.find(cFlags.mCategoryFlag()) != dKUcatCpt.end()) {
        vTextCollapsed[vI] = dKUcatCpt.at(cFlags.mCategoryFlag());
      } else {
        vTextCollapsed[vI] = (char)0xD0; // fallback
      }
    }
  }

  std::vector<size_t> lBpeOffsets = {cCpts.size()};

  for (const auto &vRegexExpr : pRegexExprs) {
    // first, see if we have an efficient custom regex implementation
    auto vTmp = fUnicodeRegexSplitCustom(pText, vRegexExpr, lBpeOffsets);

    if (!vTmp.empty()) {
      lBpeOffsets = std::move(vTmp);
      continue;
    }

    // fallback to general-purpose std::regex / std::wregex
    try {
      // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
      // with the corresponding collapsed representation
      bool vUseCollapsed = false;
      for (const auto &vUcat : dKUcatEnum) {
        if (std::string::npos != vRegexExpr.find(vUcat.first)) {
          vUseCollapsed = true;
          break;
        }
      }
      const auto cCptsRegex = fUnicodeCptsFromUtf8(vRegexExpr);

      if (vUseCollapsed) {
        // sanity-check that the original regex does not contain any non-ASCII characters
        for (size_t vI = 0; vI < cCptsRegex.size(); ++vI) {
          if (cCptsRegex[vI] >= 128) {
            throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
          }
        }

        // generate a collapsed representation of the regex
        std::string vRegexExprCollapsed;

        // track if we are inside [], because nested [] are not allowed
        bool vInside = false;
        for (size_t vI = 0; vI < vRegexExpr.size(); ++vI) {
          if (vRegexExpr[vI] == '[' && (vI == 0 || vRegexExpr[vI - 1] != '\\')) {
            vRegexExprCollapsed += '[';
            vInside = true;
            continue;
          }

          if (vInside && vRegexExpr[vI] == ']' && vRegexExpr[vI - 1] != '\\') {
            vRegexExprCollapsed += ']';
            vInside = false;
            continue;
          }

          // Match \p{...} Unicode properties of varying lengths
          if (vRegexExpr[vI + 0] == '\\' && vI + 3 < vRegexExpr.size() && vRegexExpr[vI + 1] == 'p' &&
              vRegexExpr[vI + 2] == '{') {
            // Find the closing brace
            size_t vClosingBrace = vRegexExpr.find('}', vI + 3);
            if (vClosingBrace != std::string::npos && vClosingBrace <= vI + 10) { // reasonable limit
              const std::string cPat = vRegexExpr.substr(vI, vClosingBrace - vI + 1);
              if (dKUcatEnum.find(cPat) != dKUcatEnum.end()) {
                if (!vInside) {
                  vRegexExprCollapsed += '[';
                }
                vRegexExprCollapsed += dKUcatCpt.at(dKUcatEnum.at(cPat));
                vRegexExprCollapsed += dKUcatMap.at(dKUcatEnum.at(cPat));
                if (!vInside) {
                  vRegexExprCollapsed += ']';
                }
                vI = vClosingBrace;
                continue;
              }
            }
          }

          vRegexExprCollapsed += vRegexExpr[vI];
        }

        // printf("text_collapsed: %s\n", vTextCollapsed.c_str());
        // printf("regex_expr_collapsed: %s\n", vRegexExprCollapsed.c_str());
        lBpeOffsets = fUnicodeRegexSplitStl(vTextCollapsed, vRegexExprCollapsed, lBpeOffsets);
      } else {
        // no unicode category used, we can use std::wregex directly
        std::wstring vWregexExpr(cCptsRegex.begin(), cCptsRegex.end());

        // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
        std::wstring vWtext(cCpts.begin(), cCpts.end());
        for (size_t vI = 0; vI < vWtext.size(); ++vI) {
          if (vWtext[vI] > 0x7F && fUnicodeCptFlagsFromCpt(vWtext[vI]).vIsWhitespace) {
            vWtext[vI] = 0x0B;
          }
        }

        // printf("text: %s\n", pText.c_str());
        // printf("regex_expr: %s\n", vRegexExpr.c_str());
        lBpeOffsets = fUnicodeRegexSplitStl(vWtext, vWregexExpr, lBpeOffsets);
      }
    } catch (std::regex_error &vE) {
      fprintf(stderr, "Failed to process regex: '%s'\n", vRegexExpr.c_str());
      fprintf(stderr, "Regex error: %s\n", vE.what());
      throw std::runtime_error("Failed to process regex");
    }
  }

  std::vector<std::string> lBpeWords;
  lBpeWords.reserve(lBpeOffsets.size()); // reserve memory for the approximate size

  size_t vStart = 0;
  for (size_t &vOffset : lBpeOffsets) {
    lBpeWords.emplace_back();
    for (size_t vI = vStart; vI < vStart + vOffset; ++vI) {
      lBpeWords.back() += fUnicodeCptToUtf8(cCpts[vI]);
    }
    vStart += vOffset;
  }

  if (pByteEncode) {
    return fUnicodeByteEncodingProcess(lBpeWords);
  }

  return lBpeWords;
}
