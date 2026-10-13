#include "jinja/string.h"
#include "jinja/value.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

//
// string_part
//

bool string_part::mIsUppercase() const {
  for (char vC : vVal) {
    if (std::islower(static_cast<unsigned char>(vC))) {
      return false;
    }
  }
  return true;
}

bool string_part::mIsLowercase() const {
  for (char vC : vVal) {
    if (std::isupper(static_cast<unsigned char>(vC))) {
      return false;
    }
  }
  return true;
}

//
// string
//

void string::mMarkInput() {
  for (auto &vPart : lParts) {
    vPart.vIsInput = true;
  }
}

std::string string::mStr() const {
  if (lParts.size() == 1) {
    return lParts[0].vVal;
  }
  std::ostringstream vOss;
  for (const auto &vPart : lParts) {
    vOss << vPart.vVal;
  }
  return vOss.str();
}

size_t string::mLength() const {
  size_t vLen = 0;
  for (const auto &vPart : lParts) {
    vLen += vPart.vVal.length();
  }
  return vLen;
}

void string::mHashUpdate(hasher &pHash) const noexcept {
  for (const auto &vPart : lParts) {
    pHash.mUpdate(vPart.vVal.data(), vPart.vVal.length());
  }
}

bool string::mAllPartsAreInput() const {
  for (const auto &vPart : lParts) {
    if (!vPart.vIsInput) {
      return false;
    }
  }
  return true;
}

bool string::mIsUppercase() const {
  for (const auto &vPart : lParts) {
    if (!vPart.mIsUppercase()) {
      return false;
    }
  }
  return true;
}

bool string::mIsLowercase() const {
  for (const auto &vPart : lParts) {
    if (!vPart.mIsLowercase()) {
      return false;
    }
  }
  return true;
}

// mark this string as input if other has ALL parts as input
void string::mMarkInputBasedOn(const string &pOther) {
  if (pOther.mAllPartsAreInput()) {
    for (auto &vPart : lParts) {
      vPart.vIsInput = true;
    }
  }
}

string string::mAppend(const string &pOther) {
  for (const auto &vPart : pOther.lParts) {
    lParts.push_back(vPart);
  }
  return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string &)>;
static string fApplyTransform(string &pSelf, const transform_fn &pFn) {
  for (auto &vPart : pSelf.lParts) {
    vPart.vVal = pFn(vPart.vVal);
  }
  return pSelf;
}

string string::mUppercase() {
  return fApplyTransform(*this, [](const std::string &pS) {
    std::string vRes = pS;
    std::transform(vRes.begin(), vRes.end(), vRes.begin(), ::toupper);
    return vRes;
  });
}
string string::mLowercase() {
  return fApplyTransform(*this, [](const std::string &pS) {
    std::string vRes = pS;
    std::transform(vRes.begin(), vRes.end(), vRes.begin(), ::tolower);
    return vRes;
  });
}
string string::mCapitalize() {
  return fApplyTransform(*this, [](const std::string &pS) {
    if (pS.empty())
      return pS;
    std::string vRes = pS;
    vRes[0] = ::toupper(static_cast<unsigned char>(vRes[0]));
    std::transform(vRes.begin() + 1, vRes.end(), vRes.begin() + 1, ::tolower);
    return vRes;
  });
}
string string::mTitlecase() {
  return fApplyTransform(*this, [](const std::string &pS) {
    std::string vRes = pS;
    bool vCapitalizeNext = true;
    for (char &vC : vRes) {
      if (isspace(static_cast<unsigned char>(vC))) {
        vCapitalizeNext = true;
      } else if (vCapitalizeNext) {
        vC = ::toupper(static_cast<unsigned char>(vC));
        vCapitalizeNext = false;
      } else {
        vC = ::tolower(static_cast<unsigned char>(vC));
      }
    }
    return vRes;
  });
}
string string::mStrip(bool pLeft, bool pRight, std::optional<const std::string_view> pChars) {
  static auto vStripPart = [](const std::string &pS, bool pLeft, bool pRight,
                              std::optional<const std::string_view> pChars) -> std::string {
    size_t vStart = 0;
    size_t vEnd = pS.length();
    auto vMatchChar = [&pChars](unsigned char pC) -> bool {
      return pChars ? (*pChars).find(pC) != std::string::npos : isspace(pC);
    };
    if (pLeft) {
      while (vStart < vEnd && vMatchChar(static_cast<unsigned char>(pS[vStart]))) {
        ++vStart;
      }
    }
    if (pRight) {
      while (vEnd > vStart && vMatchChar(static_cast<unsigned char>(pS[vEnd - 1]))) {
        --vEnd;
      }
    }
    return pS.substr(vStart, vEnd - vStart);
  };
  if (lParts.empty()) {
    return *this;
  }
  if (pLeft) {
    for (size_t vI = 0; vI < lParts.size(); ++vI) {
      lParts[vI].vVal = vStripPart(lParts[vI].vVal, true, false, pChars);
      if (lParts[vI].vVal.empty()) {
        // remove empty part
        lParts.erase(lParts.begin() + vI);
        --vI;
        continue;
      } else {
        break;
      }
    }
  }
  if (pRight) {
    for (size_t vI = lParts.size(); vI-- > 0;) {
      lParts[vI].vVal = vStripPart(lParts[vI].vVal, false, true, pChars);
      if (lParts[vI].vVal.empty()) {
        // remove empty part
        lParts.erase(lParts.begin() + vI);
        continue;
      } else {
        break;
      }
    }
  }
  return *this;
}

} // namespace jinja
