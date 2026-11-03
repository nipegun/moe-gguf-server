#pragma once

#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace jinja {

static void fStringReplaceAll(std::string &pS, const std::string &pSearch, const std::string &pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

// for displaying source code around error position
static std::string fPeakSource(const std::string &pSource, size_t pPos, size_t pMaxPeakChars = 40) {
  if (pSource.empty()) {
    return "(no source available)";
  }
  std::string vOutput;
  size_t vStart = (pPos >= pMaxPeakChars) ? (pPos - pMaxPeakChars) : 0;
  size_t vEnd = std::min(pPos + pMaxPeakChars, pSource.length());
  std::string vSubstr = pSource.substr(vStart, vEnd - vStart);
  fStringReplaceAll(vSubstr, "\n", "↵");
  vOutput += "..." + vSubstr + "...\n";
  std::string vSpaces(pPos - vStart + 3, ' ');
  vOutput += vSpaces + "^";
  return vOutput;
}

static std::string fFmtErrorWithSource(const std::string &pTag, const std::string &pMsg, const std::string &pSource,
                                       size_t pPos) {
  std::ostringstream vOss;
  vOss << pTag << ": " << pMsg << "\n";
  vOss << fPeakSource(pSource, pPos);
  return vOss.str();
}

// Note: this is a simple hasher, not cryptographically secure, just for hash table usage
struct hasher {
  static constexpr auto cSizeTDigits = sizeof(size_t) * 8;
  static constexpr size_t cPrime = cSizeTDigits == 64 ? 0x100000001b3 : 0x01000193;
  static constexpr size_t cSeed = cSizeTDigits == 64 ? 0xcbf29ce484222325 : 0x811c9dc5;
  static constexpr auto cBlockSize = sizeof(size_t); // in bytes; allowing the compiler to vectorize the computation

  static_assert(cSizeTDigits == 64 || cSizeTDigits == 32);
  static_assert(cBlockSize == 8 || cBlockSize == 4);

  uint8_t aBuffer[cBlockSize];
  size_t vIdx = 0; // current index in buffer
  size_t vState = cSeed;

  hasher() = default;
  hasher(const std::type_info &pTypeInf) noexcept {
    const auto cTypeHash = pTypeInf.hash_code();
    mUpdate(&cTypeHash, sizeof(cTypeHash));
  }

  // Properties:
  //   - update is not associative: update(a).update(b) != update(b).update(a)
  //   - update(a ~ b) == update(a).update(b) with ~ as concatenation operator --> useful for streaming
  //   - update("", 0) --> state unchanged with empty input
  hasher &mUpdate(void const *pBytes, size_t pLen) noexcept {
    const uint8_t *vC = static_cast<uint8_t const *>(pBytes);
    if (pLen == 0) {
      return *this;
    }
    size_t vProcessed = 0;

    // first, fill the existing buffer if it's partial
    if (vIdx > 0) {
      size_t vToFill = cBlockSize - vIdx;
      if (vToFill > pLen) {
        vToFill = pLen;
      }
      std::memcpy(aBuffer + vIdx, vC, vToFill);
      vIdx += vToFill;
      vProcessed += vToFill;
      if (vIdx == cBlockSize) {
        mUpdateBlock(aBuffer);
        vIdx = 0;
      }
    }

    // process full blocks from the remaining input
    for (; vProcessed + cBlockSize <= pLen; vProcessed += cBlockSize) {
      mUpdateBlock(vC + vProcessed);
    }

    // buffer any remaining bytes
    size_t vRemaining = pLen - vProcessed;
    if (vRemaining > 0) {
      std::memcpy(aBuffer, vC + vProcessed, vRemaining);
      vIdx = vRemaining;
    }
    return *this;
  }

  // convenience function for testing only
  hasher &mUpdate(const std::string &pS) noexcept { return mUpdate(pS.data(), pS.size()); }

  // finalize and get the hash value
  // note: after calling digest, the hasher state is modified, do not call update() again
  size_t mDigest() noexcept {
    // if there are remaining bytes in buffer, fill the rest with zeros and process
    if (vIdx > 0) {
      for (size_t vI = vIdx; vI < cBlockSize; ++vI) {
        aBuffer[vI] = 0;
      }
      mUpdateBlock(aBuffer);
      vIdx = 0;
    }

    return vState;
  }

private:
  // IMPORTANT: block must have at least block_size bytes
  void mUpdateBlock(const uint8_t *pBlock) noexcept {
    size_t vBlk = static_cast<uint32_t>(pBlock[0]) | (static_cast<uint32_t>(pBlock[1]) << 8) |
                  (static_cast<uint32_t>(pBlock[2]) << 16) | (static_cast<uint32_t>(pBlock[3]) << 24);
    if constexpr (cBlockSize == 8) {
      vBlk = vBlk | (static_cast<uint64_t>(pBlock[4]) << 32) | (static_cast<uint64_t>(pBlock[5]) << 40) |
             (static_cast<uint64_t>(pBlock[6]) << 48) | (static_cast<uint64_t>(pBlock[7]) << 56);
    }
    vState ^= vBlk;
    vState *= cPrime;
  }
};

} // namespace jinja
