#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct common_trie {
  struct node {
    std::map<uint32_t, size_t> dChildren; // Use uint32_t to store Unicode codepoints
    int32_t vPattern = -1;                // index of the pattern ending at this node, -1 if none
  };

  std::vector<node> lNodes;

  common_trie() {
    mCreateNode(); // root node
  }

  common_trie(const std::vector<std::string> &pWords) : common_trie() {
    for (const auto &vW : pWords) {
      mInsert(vW);
    }
  }

  enum match_result { cNoMatch, cPartialMatch, cCompleteMatch };

  // Check if a delimiter starts at the given position
  match_result mCheckAt(std::string_view pSv, size_t pStartPos) const;

  // Insert a word as a sequence of Unicode codepoints, returns its pattern index
  int32_t mInsert(const std::string &pWord);

  // Insert a raw symbol sequence, returns its pattern index (insertion order,
  // duplicates keep the first index)
  int32_t mInsert(const std::vector<uint32_t> &pSymbols);

private:
  int32_t vNPatterns = 0;

  size_t mCreateNode() {
    size_t vIndex = lNodes.size();
    lNodes.emplace_back();
    return vIndex;
  }
};

// Aho-Corasick automaton
struct common_aho_corasick {
  common_trie vT;
  std::vector<size_t> lFail;    // failure links
  std::vector<size_t> lOrder;   // states in BFS order
  std::vector<int32_t> lMatch;  // longest pattern ending at each state (directly or via a suffix link), -1 if none
  std::set<uint32_t> sAlphabet; // every character with a transition

  common_aho_corasick(common_trie pTrie);

  common_aho_corasick(const std::vector<std::string> &pStrings) : common_aho_corasick(common_trie(pStrings)) {}

  size_t mNumStates() const { return vT.lNodes.size(); }
  bool mIsTerminal(size_t pS) const { return lMatch[pS] >= 0; }

  // index of the longest pattern ending at this state, -1 if none
  int32_t mMatchPattern(size_t pS) const { return lMatch[pS]; }

  // follow failure links until a transition on `ch` exists.
  size_t mNext(size_t pState, uint32_t pCh) const;
};
