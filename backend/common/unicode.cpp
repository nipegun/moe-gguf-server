#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

// implementation adopted from src/unicode.cpp

size_t fCommonUtf8SequenceLength(unsigned char pFirstByte) {
  const size_t aLookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t vHighbits = static_cast<uint8_t>(pFirstByte) >> 4;
  return aLookup[vHighbits];
}

utf8_parse_result fCommonParseUtf8Codepoint(std::string_view pInput, size_t pOffset) {
  if (pOffset >= pInput.size()) {
    return utf8_parse_result(utf8_parse_result::cIncomplete);
  }

  // ASCII fast path
  if (!(pInput[pOffset] & 0x80)) {
    return utf8_parse_result(utf8_parse_result::cSuccess, pInput[pOffset], 1);
  }

  // Invalid: continuation byte as first byte
  if (!(pInput[pOffset] & 0x40)) {
    return utf8_parse_result(utf8_parse_result::cInvalid);
  }

  // 2-byte sequence
  if (!(pInput[pOffset] & 0x20)) {
    if (pOffset + 1 >= pInput.size()) {
      return utf8_parse_result(utf8_parse_result::cIncomplete);
    }
    if ((pInput[pOffset + 1] & 0xc0) != 0x80) {
      return utf8_parse_result(utf8_parse_result::cInvalid);
    }
    auto vResult = ((pInput[pOffset] & 0x1f) << 6) | (pInput[pOffset + 1] & 0x3f);
    return utf8_parse_result(utf8_parse_result::cSuccess, vResult, 2);
  }

  // 3-byte sequence
  if (!(pInput[pOffset] & 0x10)) {
    if (pOffset + 2 >= pInput.size()) {
      return utf8_parse_result(utf8_parse_result::cIncomplete);
    }
    if ((pInput[pOffset + 1] & 0xc0) != 0x80 || (pInput[pOffset + 2] & 0xc0) != 0x80) {
      return utf8_parse_result(utf8_parse_result::cInvalid);
    }
    auto vResult =
        ((pInput[pOffset] & 0x0f) << 12) | ((pInput[pOffset + 1] & 0x3f) << 6) | (pInput[pOffset + 2] & 0x3f);
    return utf8_parse_result(utf8_parse_result::cSuccess, vResult, 3);
  }

  // 4-byte sequence
  if (!(pInput[pOffset] & 0x08)) {
    if (pOffset + 3 >= pInput.size()) {
      return utf8_parse_result(utf8_parse_result::cIncomplete);
    }
    if ((pInput[pOffset + 1] & 0xc0) != 0x80 || (pInput[pOffset + 2] & 0xc0) != 0x80 ||
        (pInput[pOffset + 3] & 0xc0) != 0x80) {
      return utf8_parse_result(utf8_parse_result::cInvalid);
    }
    auto vResult = ((pInput[pOffset] & 0x07) << 18) | ((pInput[pOffset + 1] & 0x3f) << 12) |
                   ((pInput[pOffset + 2] & 0x3f) << 6) | (pInput[pOffset + 3] & 0x3f);
    return utf8_parse_result(utf8_parse_result::cSuccess, vResult, 4);
  }

  // Invalid first byte
  return utf8_parse_result(utf8_parse_result::cInvalid);
}

bool fCommonUtf8IsComplete(const std::string &pS) {
  if (pS.empty()) {
    return true;
  }
  for (int vI = 1; vI <= std::min(4, (int)pS.size()); vI++) {
    unsigned char vC = pS[pS.size() - vI];
    if ((vC & 0xC0) != 0x80) {
      int vExpected = (vC >= 0xF0) ? 4 : (vC >= 0xE0) ? 3 : (vC >= 0xC0) ? 2 : 1;
      return vI >= vExpected;
    }
  }
  return false;
}

std::string fCommonUnicodeCptsToUtf8(const std::vector<uint32_t> &pCps) {
  std::string vResult;
  for (size_t vI = 0; vI < pCps.size(); ++vI) {
    vResult.append(fCommonUnicodeCptToUtf8(pCps[vI]));
  }
  return vResult;
}

std::string fCommonUnicodeCptToUtf8(uint32_t pCpt) {
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
