#pragma once

#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace jinja {

// allow differentiate between user input strings and template strings
// transformations should handle this information as follows:
// - one-to-one (e.g., uppercase, lowercase): preserve is_input flag
// - one-to-many (e.g., strip): if input string is marked as is_input, all resulting parts should be marked as is_input
// - many-to-one (e.g., concat): if ALL input parts are marked as is_input, resulting part should be marked as is_input
struct string_part {
  bool vIsInput = false; // may skip parsing special tokens if true
  std::string vVal;

  bool mIsUppercase() const;
  bool mIsLowercase() const;
};

struct string {
  std::vector<string_part> lParts;
  string() = default;
  string(const std::string &pV, bool pUserInput = false) { lParts.push_back({pUserInput, pV}); }
  string(int pV) { lParts.push_back({false, std::to_string(pV)}); }
  string(double pV) { lParts.push_back({false, std::to_string(pV)}); }

  // mark all parts as user input
  void mMarkInput();

  std::string mStr() const;
  size_t mLength() const;
  void mHashUpdate(hasher &pHash) const noexcept;
  bool mAllPartsAreInput() const;
  bool mIsUppercase() const;
  bool mIsLowercase() const;

  // mark this string as input if other has ALL parts as input
  void mMarkInputBasedOn(const string &pOther);

  string mAppend(const string &pOther);

  // in-place transformations

  string mUppercase();
  string mLowercase();
  string mCapitalize();
  string mTitlecase();
  string mStrip(bool pLeft, bool pRight, std::optional<const std::string_view> pChars = std::nullopt);
};

} // namespace jinja
