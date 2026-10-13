#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct range_nfd {
  uint32_t vFirst;
  uint32_t vLast;
  uint32_t vNfd;
};

static const uint32_t cMaxCodepoints = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> lUnicodeRangesFlags;
extern const std::unordered_set<uint32_t> sUnicodeSetWhitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> lUnicodeMapLowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> lUnicodeMapUppercase;
extern const std::initializer_list<range_nfd> lUnicodeRangesNfd;
