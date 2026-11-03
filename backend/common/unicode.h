#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>

// UTF-8 parsing utilities for streaming-aware unicode support

struct utf8_parse_result {
  uint32_t vCodepoint;   // Decoded codepoint (only valid if status == SUCCESS)
  size_t vBytesConsumed; // How many bytes this codepoint uses (1-4)
  enum status { cSuccess, cIncomplete, cInvalid } vStatus;

  utf8_parse_result(enum status pS, uint32_t pCp = 0, size_t pBytes = 0)
      : vCodepoint(pCp), vBytesConsumed(pBytes), vStatus(pS) {}
};

// Determine the expected length of a UTF-8 sequence from its first byte
// Returns 0 for invalid first bytes
size_t fCommonUtf8SequenceLength(unsigned char pFirstByte);

// Check if a string ends with a complete UTF-8 sequence.
bool fCommonUtf8IsComplete(const std::string &pS);

// Parse a single UTF-8 codepoint from input
utf8_parse_result fCommonParseUtf8Codepoint(std::string_view pInput, size_t pOffset);

std::string fCommonUnicodeCptsToUtf8(const std::vector<uint32_t> &pCps);
std::string fCommonUnicodeCptToUtf8(uint32_t pCpt);
