#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "peg-parser.h"

#include <cctype>
#include <numeric>

using json = nlohmann::ordered_json;

std::string fTrimWhitespace(const std::string &pStr) {
  size_t vStart = 0;
  while (vStart < pStr.length() && std::isspace(static_cast<unsigned char>(pStr[vStart]))) {
    vStart++;
  }

  if (vStart == pStr.length()) {
    return "";
  }

  size_t vEnd = pStr.length() - 1;
  while (vEnd > vStart && std::isspace(static_cast<unsigned char>(pStr[vEnd]))) {
    vEnd--;
  }

  return pStr.substr(vStart, vEnd - vStart + 1);
}

std::string fTrimLeadingWhitespace(const std::string &pStr) {
  size_t vStart = 0;
  while (vStart < pStr.length() && std::isspace(static_cast<unsigned char>(pStr[vStart]))) {
    vStart++;
  }

  return pStr.substr(vStart);
}

std::string fTrimTrailingWhitespace(const std::string &pStr) {
  if (pStr.empty()) {
    return "";
  }

  size_t vEnd = pStr.length() - 1;
  while (vEnd > 0 && std::isspace(static_cast<unsigned char>(pStr[vEnd]))) {
    vEnd--;
  }

  // If first char is also whitespace, return empty string
  if (vEnd == 0 && std::isspace(static_cast<unsigned char>(pStr[0]))) {
    return "";
  }

  return pStr.substr(0, vEnd + 1);
}

std::string fTrimTrailingNewlines(const std::string &pStr) {
  size_t vEnd = pStr.length();
  while (vEnd > 0 && pStr[vEnd - 1] == '\n') {
    vEnd--;
  }

  return pStr.substr(0, vEnd);
}

static size_t fCommonPrefixLen(const std::string &pLeft, const std::string &pRight) {
  size_t vPrefixLen = 0;
  size_t vMinLen = std::min(pLeft.length(), pRight.length());
  while (vPrefixLen < vMinLen && pLeft[vPrefixLen] == pRight[vPrefixLen]) {
    vPrefixLen++;
  }
  return vPrefixLen;
}

static size_t fCommonSuffixLen(const std::string &pLeft, const std::string &pRight) {
  size_t vSuffixLen = 0;
  size_t vMinLen = std::min(pLeft.length(), pRight.length());
  while (vSuffixLen < vMinLen && pLeft[pLeft.length() - 1 - vSuffixLen] == pRight[pRight.length() - 1 - vSuffixLen]) {
    vSuffixLen++;
  }
  return vSuffixLen;
}

diff_split fCalculateDiffSplit(const std::string &pLeft, const std::string &pRight) {
  diff_split vResult;

  auto vLeftSeg = fSegmentizeMarkers(pLeft);
  auto vRightSeg = fSegmentizeMarkers(pRight);

  if (vLeftSeg.empty()) {
    vResult.vRight = pRight;
    return vResult;
  }
  if (vRightSeg.empty()) {
    vResult.vLeft = pLeft;
    return vResult;
  }

  auto vLeftStart = vLeftSeg.begin();
  auto vLeftEnd = --vLeftSeg.end();
  auto vRightStart = vRightSeg.begin();
  auto vRightEnd = --vRightSeg.end();

  auto vTest = [&]() { return vLeftStart != vLeftEnd && vRightStart != vRightEnd; };

  bool vLeftFullyConsumed = false;
  bool vRightFullyConsumed = false;

  while (vTest()) {
    bool vAdvanced = false;
    if (*vLeftStart == *vRightStart) {
      vResult.vPrefix.append(vLeftStart->vValue);
      vLeftStart++;
      vRightStart++;
      vAdvanced = true;
    }
    if (*vLeftEnd == *vRightEnd) {
      vResult.vSuffix = vLeftEnd->vValue + vResult.vSuffix;
      if (vLeftStart != vLeftEnd) {
        vLeftEnd--;
      } else {
        vLeftFullyConsumed = true;
      }
      if (vRightStart != vRightEnd) {
        vRightEnd--;
      } else {
        vRightFullyConsumed = true;
      }
      vAdvanced = true;
    }
    if (!vAdvanced) {
      break;
    }
  }

  if (vLeftStart == vLeftEnd && vRightStart != vRightEnd) {
    if (*vLeftStart == *vRightEnd) {
      vResult.vSuffix = vRightEnd->vValue + vResult.vSuffix;
      vRightEnd--;
      vLeftFullyConsumed = true;
    } else if (*vLeftStart == *vRightStart) {
      vResult.vPrefix.append(vRightStart->vValue);
      vRightStart++;
      vLeftFullyConsumed = true;
    }
  } else if (vRightStart == vRightEnd && vLeftStart != vLeftEnd) {
    if (*vLeftEnd == *vRightStart) {
      vResult.vSuffix = vLeftEnd->vValue + vResult.vSuffix;
      vLeftEnd--;
      vRightFullyConsumed = true;
    } else if (*vLeftStart == *vRightStart) {
      vResult.vPrefix.append(vLeftStart->vValue);
      vLeftStart++;
      vRightFullyConsumed = true;
    }
  } else if (vLeftStart == vLeftEnd && vRightStart == vRightEnd && *vLeftStart == *vRightStart &&
             vLeftStart->vType == segment_type::cMarker) {
    vResult.vPrefix.append(vRightStart->vValue);
    vLeftFullyConsumed = true;
    vRightFullyConsumed = true;
  }

  auto vEatSegment = [](std::string pStr, const segment &pSeg) -> std::string { return std::move(pStr) + pSeg.vValue; };

  bool vCanHaveTextSuffix = vLeftEnd->vType == segment_type::cText && vRightEnd->vType == segment_type::cText;
  bool vCanHaveTextPrefix = vRightStart->vType == segment_type::cText && vLeftStart->vType == segment_type::cText;

  std::string vRemainderLeft =
      std::accumulate(vLeftStart, vLeftFullyConsumed ? vLeftEnd : ++vLeftEnd, std::string(), vEatSegment);
  std::string vRemainderRight =
      std::accumulate(vRightStart, vRightFullyConsumed ? vRightEnd : ++vRightEnd, std::string(), vEatSegment);

  size_t vSuffixLen = vCanHaveTextSuffix ? fCommonSuffixLen(vRemainderLeft, vRemainderRight) : 0;
  // avoid overlaps between prefix and suffix
  size_t vPrefixLen = vCanHaveTextPrefix
                          ? fCommonPrefixLen(vRemainderLeft.substr(0, vRemainderLeft.size() - vSuffixLen),
                                             vRemainderRight.substr(0, vRemainderRight.size() - vSuffixLen))
                          : 0;

  vResult.vPrefix.append(vRemainderLeft.substr(0, vPrefixLen));
  vResult.vSuffix = vRemainderLeft.substr(vRemainderLeft.length() - vSuffixLen, vSuffixLen) + vResult.vSuffix;
  vResult.vLeft = vRemainderLeft.substr(vPrefixLen, vRemainderLeft.length() - vPrefixLen - vSuffixLen);
  vResult.vRight = vRemainderRight.substr(vPrefixLen, vRemainderRight.length() - vPrefixLen - vSuffixLen);

  if (vResult.vLeft == "" && vResult.vRight == "") {
    // degenerate case, no diff
    vResult.vPrefix = pLeft;
    vResult.vSuffix = "";
    // pick prefix = all as representation
  }

  // When left has no unique content (vResult.vLeft is empty), left is entirely
  // shared with right. The simultaneous prefix/suffix segment matching can
  // incorrectly consume trailing segments of left as suffix when those same
  // segments also appear at the end of right (e.g. "\n" at the end of both
  // the shared content and the generation prompt). This rotates the diff.
  // Fix: if left is a prefix of right, enforce that directly.
  if (vResult.vLeft.empty() && !vResult.vRight.empty() && pLeft.size() <= pRight.size() &&
      pRight.substr(0, pLeft.size()) == pLeft) {
    vResult.vPrefix = pLeft;
    vResult.vSuffix = "";
    vResult.vRight = pRight.substr(pLeft.size());
  }

  return vResult;
}

// Returns the prefix of `pFull` up until the first occurrence of the common prefix of `pLeft` and `pRight`
std::string fUntilCommonPrefix(const std::string &pFull, const std::string &pLeft, const std::string &pRight) {
  // Find the common prefix of left and right
  size_t vCommonPrefixLen = 0;
  size_t vMinLen = std::min(pLeft.length(), pRight.length());
  while (vCommonPrefixLen < vMinLen && pLeft[vCommonPrefixLen] == pRight[vCommonPrefixLen]) {
    vCommonPrefixLen++;
  }

  // If there's no common prefix, return empty string
  if (vCommonPrefixLen == 0) {
    return "";
  }

  // Find the common prefix in the full string
  std::string vCommonPrefix = pLeft.substr(0, vCommonPrefixLen);
  size_t vPos = pFull.find(vCommonPrefix);

  // If not found, return empty string
  if (vPos == std::string::npos) {
    return "";
  }

  // Return everything before the common prefix
  return pFull.substr(0, vPos);
}

// Returns the suffix of `pFull` after the last occurrence of the common suffix of `pLeft` and `pRight`
std::string fAfterCommonSuffix(const std::string &pFull, const std::string &pLeft, const std::string &pRight) {
  // Find the common suffix of left and right (compare from the end)
  size_t vCommonSuffixLen = 0;
  size_t vMinLen = std::min(pLeft.length(), pRight.length());
  while (vCommonSuffixLen < vMinLen &&
         pLeft[pLeft.length() - 1 - vCommonSuffixLen] == pRight[pRight.length() - 1 - vCommonSuffixLen]) {
    vCommonSuffixLen++;
  }

  // If there's no common suffix, return empty string
  if (vCommonSuffixLen == 0) {
    return "";
  }

  // Extract the common suffix
  std::string vCommonSuffix = pLeft.substr(pLeft.length() - vCommonSuffixLen);

  // Find the last occurrence of the common suffix in the full string
  size_t vPos = pFull.rfind(vCommonSuffix);

  // If not found, return empty string
  if (vPos == std::string::npos) {
    return "";
  }

  // Return everything after the common suffix
  return pFull.substr(vPos + vCommonSuffixLen);
}

// TODO: segmentize will treat a JSON array inside tags as a tag: <calls>[{ "fun": { ... } }]</calls> will be three
// markers not too worried about that because it hasn't turned out as a problem anywhere, but noting here in case it
// will Might have to put some restrictions on tag contents as well (like "no { }")
std::vector<segment> fSegmentizeMarkers(const std::string &pText) {
  std::vector<segment> lRetval;
  bool vInMarker = false;
  char vMarkerOpener = '\0';

  auto vIsMarkerOpener = [](char pC) -> bool { return pC == '<' || pC == '['; };
  auto vIsMarkerCloser = [](char pOp, char pC) -> bool {
    return (pOp == '<' && pC == '>') || (pOp == '[' && pC == ']');
  };

  size_t vLastBorder = 0;

  for (size_t vCurPos = 0; vCurPos < pText.length(); vCurPos++) {
    if (!vInMarker && vIsMarkerOpener(pText[vCurPos])) {
      if (vLastBorder < vCurPos) {
        lRetval.push_back(segment(segment_type::cText, pText.substr(vLastBorder, vCurPos - vLastBorder)));
      }
      vLastBorder = vCurPos;
      vInMarker = true;
      vMarkerOpener = pText[vCurPos];
    } else if (vInMarker && vIsMarkerCloser(vMarkerOpener, pText[vCurPos])) {
      // no need to check because last_border will always be smaller
      lRetval.push_back(segment(segment_type::cMarker, pText.substr(vLastBorder, vCurPos - vLastBorder + 1)));
      vLastBorder = vCurPos + 1;
      vInMarker = false;
      vMarkerOpener = '\0';
    }
  }
  if (vLastBorder < pText.length()) {
    lRetval.push_back(segment(segment_type::cText, pText.substr(vLastBorder)));
  }
  return lRetval;
}

std::vector<segment> fPruneWhitespaceSegments(const std::vector<segment> &pSegments) {
  std::vector<segment> lResult;
  for (const auto &vSeg : pSegments) {
    if (!fTrimWhitespace(vSeg.vValue).empty()) {
      lResult.push_back(vSeg);
    }
  }
  return lResult;
}

namespace autoparser {

static const std::string cErrTmpl = "#**ERROR**#";

std::string fApplyTemplate(const common_chat_template &pTmpl, const template_params &pParams) {
  vGenerationParams vTmplParams;
  vTmplParams.vMessages = pParams.vMessages;
  vTmplParams.vTools = pParams.vTools;
  vTmplParams.vAddGenerationPrompt = pParams.vAddGenerationPrompt;
  vTmplParams.vEnableThinking = pParams.vEnableThinking;

  if (pParams.vExtraContext) {
    vTmplParams.vExtraContext = *pParams.vExtraContext;
  }
  vTmplParams.vExtraContext["enable_thinking"] = pParams.vEnableThinking;

  try {
    return fCommonChatTemplateDirectApply(pTmpl, vTmplParams);
  } catch (const std::exception &vE) {
    LOG_DBG("Template application failed: %s\n", vE.what());
    return cErrTmpl;
  }
}

std::optional<compare_variants_result> fCompareVariants(const common_chat_template &pTmpl,
                                                        const template_params &pParamsA,
                                                        const std::function<void(template_params &)> &pParamsModifier) {
  // Create variant B by copying A
  template_params vParamsB = pParamsA;

  // Apply modifier to create variant B
  if (pParamsModifier) {
    pParamsModifier(vParamsB);
  }

  // Apply template to both variants
  std::string vOutputA = fApplyTemplate(pTmpl, pParamsA);
  std::string vOutputB = fApplyTemplate(pTmpl, vParamsB);

  // Check for template application failures
  if (vOutputA == cErrTmpl || vOutputB == cErrTmpl) {
    return std::nullopt;
  }

  // Calculate diff and return result with both outputs
  compare_variants_result vResult;
  vResult.vDiff = fCalculateDiffSplit(vOutputA, vOutputB);
  vResult.vOutputA = vOutputA;
  vResult.vOutputB = vOutputB;

  return vResult;
}

} // namespace autoparser
