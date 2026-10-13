#include "trie.h"

#include "unicode.h"

#include <deque>

common_trie::match_result common_trie::mCheckAt(std::string_view pSv, size_t pStartPos) const {
  size_t vCurrent = 0; // Start at root
  size_t vPos = pStartPos;

  // LOG_DBG("%s: checking at pos %zu, sv='%s'\n", __func__, pStartPos, std::string(pSv).c_str());

  while (vPos < pSv.size()) {
    auto vResult = fCommonParseUtf8Codepoint(pSv, vPos);
    if (vResult.vStatus != utf8_parse_result::cSuccess) {
      break;
    }

    auto vIt = lNodes[vCurrent].dChildren.find(vResult.vCodepoint);
    if (vIt == lNodes[vCurrent].dChildren.end()) {
      // Can't continue matching
      return match_result{match_result::cNoMatch};
    }

    vCurrent = vIt->second;
    vPos += vResult.vBytesConsumed;

    // Check if we've matched a complete word
    if (lNodes[vCurrent].vPattern >= 0) {
      return match_result{match_result::cCompleteMatch};
    }
  }

  // Reached end of input while still in the trie (not at root)
  if (vCurrent != 0) {
    // We're in the middle of a potential match
    return match_result{match_result::cPartialMatch};
  }

  // Reached end at root (no match)
  return match_result{match_result::cNoMatch};
}

int32_t common_trie::mInsert(const std::string &pWord) {
  std::vector<uint32_t> lSymbols;
  size_t vPos = 0;
  while (vPos < pWord.length()) {
    auto vResult = fCommonParseUtf8Codepoint(pWord, vPos);
    if (vResult.vStatus != utf8_parse_result::cSuccess) {
      break;
    }

    lSymbols.push_back(vResult.vCodepoint);
    vPos += vResult.vBytesConsumed;
  }
  return mInsert(lSymbols);
}

int32_t common_trie::mInsert(const std::vector<uint32_t> &pSymbols) {
  size_t vCurrent = 0;
  for (uint32_t vCh : pSymbols) {
    auto vIt = lNodes[vCurrent].dChildren.find(vCh);
    if (vIt == lNodes[vCurrent].dChildren.end()) {
      size_t vChild = mCreateNode();
      lNodes[vCurrent].dChildren[vCh] = vChild;
      vCurrent = vChild;
    } else {
      vCurrent = vIt->second;
    }
  }
  if (lNodes[vCurrent].vPattern < 0) {
    lNodes[vCurrent].vPattern = vNPatterns++;
  }
  return lNodes[vCurrent].vPattern;
}

common_aho_corasick::common_aho_corasick(common_trie pTrie) : vT(std::move(pTrie)) {
  const auto &vNodes = vT.lNodes;
  const size_t cN = vNodes.size();

  lFail.assign(cN, 0);
  lOrder.reserve(cN);

  std::deque<size_t> lQueue{0};
  while (!lQueue.empty()) {
    size_t vU = lQueue.front();
    lQueue.pop_front();
    lOrder.push_back(vU);
    for (const auto &[vCh, vV] : vNodes[vU].dChildren) {
      if (vU != 0) {
        size_t vF = lFail[vU];
        while (vF && vNodes[vF].dChildren.find(vCh) == vNodes[vF].dChildren.end()) {
          vF = lFail[vF];
        }
        auto vIt = vNodes[vF].dChildren.find(vCh);
        lFail[vV] = (vIt != vNodes[vF].dChildren.end() && vIt->second != vV) ? vIt->second : 0;
      }
      lQueue.push_back(vV);
    }
  }

  // fail[u] points to a strictly shorter suffix, so the first pattern found on
  // the fail chain (including u itself) is the longest pattern ending at u
  lMatch.assign(cN, -1);
  for (size_t vU : lOrder) {
    lMatch[vU] = vNodes[vU].vPattern >= 0 ? vNodes[vU].vPattern : (vU != 0 ? lMatch[lFail[vU]] : -1);
  }

  for (const auto &vNode : vNodes) {
    for (const auto &[vCh, vV] : vNode.dChildren) {
      sAlphabet.insert(vCh);
    }
  }
}

size_t common_aho_corasick::mNext(size_t pState, uint32_t pCh) const {
  const auto &vNodes = vT.lNodes;
  while (pState && vNodes[pState].dChildren.find(pCh) == vNodes[pState].dChildren.end()) {
    pState = lFail[pState];
  }
  auto vIt = vNodes[pState].dChildren.find(pCh);
  return vIt != vNodes[pState].dChildren.end() ? vIt->second : 0;
}
