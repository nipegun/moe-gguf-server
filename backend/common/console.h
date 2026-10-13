// Console functions

#pragma once

#include "common.h"

#include <functional>
#include <string>
#include <vector>

enum display_type {
  cDisplayTypeReset = 0,
  cDisplayTypeInfo,
  cDisplayTypePrompt,
  cDisplayTypeReasoning,
  cDisplayTypeUserInput,
  cDisplayTypeError
};

namespace console {
void fInit(bool pUseSimpleIo, bool pUseAdvancedDisplay);
void fCleanup();
void fSetDisplay(display_type pDisplay);
bool fReadline(std::string &pLine, bool pMultilineInput);

using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
void fSetCompletionCallback(completion_callback pCb);

namespace spinner {
void fStart();
void fStop();
} // namespace spinner

// note: the logging API below output directly to stdout
// it can negatively impact performance if used on inference thread
// only use in in a dedicated CLI thread
// for logging in inference thread, use log.h instead

MGS_COMMON_ATTRIBUTE_FORMAT(1, 2)
void fLog(const char *pFmt, ...);

MGS_COMMON_ATTRIBUTE_FORMAT(1, 2)
void fError(const char *pFmt, ...);

void fFlush();
} // namespace console
