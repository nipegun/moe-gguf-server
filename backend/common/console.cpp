#include "console.h"
#include "log.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <cctype>
#include <cwctype>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdarg.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <climits>
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <termios.h>
#endif

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_GRAY "\x1b[90m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_BOLD "\x1b[1m"

namespace console {

#if defined(_WIN32)
namespace {
// Use private-use unicode values to represent special keys that are not reported
// as characters (e.g. arrows on Windows). These values should never clash with
// real input and let the rest of the code handle navigation uniformly.
static constexpr char32_t cKeyArrowLeft = 0xE000;
static constexpr char32_t cKeyArrowRight = 0xE001;
static constexpr char32_t cKeyArrowUp = 0xE002;
static constexpr char32_t cKeyArrowDown = 0xE003;
static constexpr char32_t cKeyHome = 0xE004;
static constexpr char32_t cKeyEnd = 0xE005;
static constexpr char32_t cKeyCtrlArrowLeft = 0xE006;
static constexpr char32_t cKeyCtrlArrowRight = 0xE007;
static constexpr char32_t cKeyDelete = 0xE008;
} // namespace

//
// Console state
//
#endif

static bool vAdvancedDisplay = false;
static bool vSimpleIo = true;
static display_type vCurrentDisplay = cDisplayTypeReset;

static FILE *vOut = stdout;

#if defined(_WIN32)
static void *vHConsole;
#else
static FILE *vTty = nullptr;
static termios vInitialState;
#endif

static completion_callback vCompletionCb = nullptr;

//
// Init and cleanup
//

void fInit(bool pUseSimpleIo, bool pUseAdvancedDisplay) {
  vAdvancedDisplay = pUseAdvancedDisplay;
  vSimpleIo = pUseSimpleIo;
#if defined(_WIN32)
  // Windows-specific console initialization
  DWORD vDwMode = 0;
  vHConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if (vHConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(vHConsole, &vDwMode)) {
    vHConsole = GetStdHandle(STD_ERROR_HANDLE);
    if (vHConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(vHConsole, &vDwMode))) {
      vHConsole = nullptr;
      vSimpleIo = true;
    }
  }
  if (vHConsole) {
    // Check conditions combined to reduce nesting
    if (vAdvancedDisplay && !(vDwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
        !SetConsoleMode(vHConsole, vDwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
      vAdvancedDisplay = false;
    }
    // Set console output codepage to UTF8
    SetConsoleOutputCP(CP_UTF8);
  }
  HANDLE vHConIn = GetStdHandle(STD_INPUT_HANDLE);
  if (vHConIn != INVALID_HANDLE_VALUE && GetConsoleMode(vHConIn, &vDwMode)) {
    // Set console input codepage to UTF16
    _setmode(_fileno(stdin), _O_WTEXT);

    // Set ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
    if (vSimpleIo) {
      vDwMode |= ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT;
    } else {
      vDwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
    }
    if (!SetConsoleMode(vHConIn, vDwMode)) {
      vSimpleIo = true;
    }
  }
  if (vSimpleIo) {
    _setmode(_fileno(stdin), _O_U8TEXT);
  }
#else
  // POSIX-specific console initialization
  if (!vSimpleIo) {
    struct termios vNewTermios;
    tcgetattr(STDIN_FILENO, &vInitialState);
    vNewTermios = vInitialState;
    vNewTermios.c_lflag &= ~(ICANON | ECHO);
    vNewTermios.c_cc[VMIN] = 1;
    vNewTermios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &vNewTermios);

    vTty = fopen("/dev/tty", "w+");
    if (vTty != nullptr) {
      vOut = vTty;
    }
  }

  setlocale(LC_ALL, "");
#endif
}

void fCleanup() {
  // Reset console display
  fSetDisplay(cDisplayTypeReset);

#if !defined(_WIN32)
  // Restore settings on POSIX systems
  if (!vSimpleIo) {
    if (vTty != nullptr) {
      vOut = stdout;
      fclose(vTty);
      vTty = nullptr;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &vInitialState);
  }
#endif
}

//
// Display and IO
//

// Keep track of current display and only emit ANSI code if it changes
void fSetDisplay(display_type pDisplay) {
  if (vAdvancedDisplay && vCurrentDisplay != pDisplay) {
    fCommonLogFlush(fCommonLogMain());
    switch (pDisplay) {
    case cDisplayTypeReset:
      fprintf(vOut, ANSI_COLOR_RESET);
      break;
    case cDisplayTypeInfo:
      fprintf(vOut, ANSI_COLOR_MAGENTA);
      break;
    case cDisplayTypePrompt:
      fprintf(vOut, ANSI_COLOR_YELLOW);
      break;
    case cDisplayTypeReasoning:
      fprintf(vOut, ANSI_COLOR_GRAY);
      break;
    case cDisplayTypeUserInput:
      fprintf(vOut, ANSI_BOLD ANSI_COLOR_GREEN);
      break;
    case cDisplayTypeError:
      fprintf(vOut, ANSI_BOLD ANSI_COLOR_RED);
    }
    vCurrentDisplay = pDisplay;
    fflush(vOut);
  }
}

static char32_t fGetchar32() {
#if defined(_WIN32)
  HANDLE vHConIn = GetStdHandle(STD_INPUT_HANDLE);
  wchar_t vHighSurrogate = 0;

  while (true) {
    INPUT_RECORD vRecord;
    DWORD vCount;
    if (!ReadConsoleInputW(vHConIn, &vRecord, 1, &vCount) || vCount == 0) {
      return WEOF;
    }

    if (vRecord.EventType == KEY_EVENT && vRecord.Event.KeyEvent.bKeyDown) {
      wchar_t vWc = vRecord.Event.KeyEvent.uChar.UnicodeChar;
      if (vWc == 0) {
        const DWORD cCtrlMask = LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED;
        const bool cCtrlPressed = (vRecord.Event.KeyEvent.dwControlKeyState & cCtrlMask) != 0;
        switch (vRecord.Event.KeyEvent.wVirtualKeyCode) {
        case VK_LEFT:
          return cCtrlPressed ? cKeyCtrlArrowLeft : cKeyArrowLeft;
        case VK_RIGHT:
          return cCtrlPressed ? cKeyCtrlArrowRight : cKeyArrowRight;
        case VK_UP:
          return cKeyArrowUp;
        case VK_DOWN:
          return cKeyArrowDown;
        case VK_HOME:
          return cKeyHome;
        case VK_END:
          return cKeyEnd;
        case VK_DELETE:
          return cKeyDelete;
        default:
          continue;
        }
      }

      if ((vWc >= 0xD800) && (vWc <= 0xDBFF)) { // Check if wc is a high surrogate
        vHighSurrogate = vWc;
        continue;
      }
      if ((vWc >= 0xDC00) && (vWc <= 0xDFFF)) { // Check if wc is a low surrogate
        if (vHighSurrogate != 0) {              // Check if we have a high surrogate
          return ((vHighSurrogate - 0xD800) << 10) + (vWc - 0xDC00) + 0x10000;
        }
      }

      vHighSurrogate = 0; // Reset the high surrogate
      return static_cast<char32_t>(vWc);
    }
  }
#else
  wchar_t vWc = getwchar();
  if (static_cast<wint_t>(vWc) == WEOF) {
    return WEOF;
  }

#if WCHAR_MAX == 0xFFFF
  if ((vWc >= 0xD800) && (vWc <= 0xDBFF)) { // Check if wc is a high surrogate
    wchar_t vLowSurrogate = getwchar();
    if ((vLowSurrogate >= 0xDC00) && (vLowSurrogate <= 0xDFFF)) { // Check if the next wchar is a low surrogate
      return (static_cast<char32_t>(vWc & 0x03FF) << 10) + (vLowSurrogate & 0x03FF) + 0x10000;
    }
  }
  if ((vWc >= 0xD800) && (vWc <= 0xDFFF)) { // Invalid surrogate pair
    return 0xFFFD;                        // Return the replacement character U+FFFD
  }
#endif

  return static_cast<char32_t>(vWc);
#endif
}

static void fPopCursor() {
#if defined(_WIN32)
  if (vHConsole != NULL) {
    CONSOLE_SCREEN_BUFFER_INFO vBufferInfo;
    GetConsoleScreenBufferInfo(vHConsole, &vBufferInfo);

    COORD vNewCursorPosition = vBufferInfo.dwCursorPosition;
    if (vNewCursorPosition.X == 0) {
      vNewCursorPosition.X = vBufferInfo.dwSize.X - 1;
      vNewCursorPosition.Y -= 1;
    } else {
      vNewCursorPosition.X -= 1;
    }

    SetConsoleCursorPosition(vHConsole, vNewCursorPosition);
    return;
  }
#endif
  putc('\b', vOut);
}

static int fEstimateWidth(char32_t pCodepoint) {
#if defined(_WIN32)
  (void)pCodepoint;
  return 1;
#else
  return wcwidth(pCodepoint);
#endif
}

static int fPutCodepoint(const char *pUtf8Codepoint, size_t pLength, int pExpectedWidth) {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO vBufferInfo;
  if (!GetConsoleScreenBufferInfo(vHConsole, &vBufferInfo)) {
    // go with the default
    return pExpectedWidth;
  }
  COORD vInitialPosition = vBufferInfo.dwCursorPosition;
  DWORD vNNumberOfChars = pLength;
  WriteConsole(vHConsole, pUtf8Codepoint, vNNumberOfChars, &vNNumberOfChars, NULL);

  CONSOLE_SCREEN_BUFFER_INFO vNewBufferInfo;
  GetConsoleScreenBufferInfo(vHConsole, &vNewBufferInfo);

  // Figure out our real position if we're in the last column
  if (pUtf8Codepoint[0] != 0x09 && vInitialPosition.X == vNewBufferInfo.dwSize.X - 1) {
    DWORD vNNumberOfChars;
    WriteConsole(vHConsole, &" \b", 2, &vNNumberOfChars, NULL);
    GetConsoleScreenBufferInfo(vHConsole, &vNewBufferInfo);
  }

  int vWidth = vNewBufferInfo.dwCursorPosition.X - vInitialPosition.X;
  if (vWidth < 0) {
    vWidth += vNewBufferInfo.dwSize.X;
  }
  return vWidth;
#else
  // We can trust expectedWidth if we've got one
  if (pExpectedWidth >= 0 || vTty == nullptr) {
    fwrite(pUtf8Codepoint, pLength, 1, vOut);
    return pExpectedWidth;
  }

  fputs("\033[6n", vTty); // Query cursor position
  int vX1;
  int vY1;
  int vX2;
  int vY2;
  int vResults = 0;
  vResults = fscanf(vTty, "\033[%d;%dR", &vY1, &vX1);

  fwrite(pUtf8Codepoint, pLength, 1, vTty);

  fputs("\033[6n", vTty); // Query cursor position
  vResults += fscanf(vTty, "\033[%d;%dR", &vY2, &vX2);

  if (vResults != 4) {
    return pExpectedWidth;
  }

  int vWidth = vX2 - vX1;
  if (vWidth < 0) {
    // Calculate the width considering text wrapping
    struct winsize vW;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &vW);
    vWidth += vW.ws_col;
  }
  return vWidth;
#endif
}

static void fReplaceLast(char pCh) {
#if defined(_WIN32)
  fPopCursor();
  fPutCodepoint(&pCh, 1, 1);
#else
  fprintf(vOut, "\b%c", pCh);
#endif
}

static char32_t fDecodeUtf8(const std::string &pInput, size_t pPos, size_t &pAdvance) {
  unsigned char vC = static_cast<unsigned char>(pInput[pPos]);
  if ((vC & 0x80u) == 0u) {
    pAdvance = 1;
    return vC;
  }
  if ((vC & 0xE0u) == 0xC0u && pPos + 1 < pInput.size()) {
    unsigned char vC1 = static_cast<unsigned char>(pInput[pPos + 1]);
    if ((vC1 & 0xC0u) != 0x80u) {
      pAdvance = 1;
      return 0xFFFD;
    }
    pAdvance = 2;
    return ((vC & 0x1Fu) << 6) | (static_cast<unsigned char>(pInput[pPos + 1]) & 0x3Fu);
  }
  if ((vC & 0xF0u) == 0xE0u && pPos + 2 < pInput.size()) {
    unsigned char vC1 = static_cast<unsigned char>(pInput[pPos + 1]);
    unsigned char vC2 = static_cast<unsigned char>(pInput[pPos + 2]);
    if ((vC1 & 0xC0u) != 0x80u || (vC2 & 0xC0u) != 0x80u) {
      pAdvance = 1;
      return 0xFFFD;
    }
    pAdvance = 3;
    return ((vC & 0x0Fu) << 12) | ((static_cast<unsigned char>(pInput[pPos + 1]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(pInput[pPos + 2]) & 0x3Fu);
  }
  if ((vC & 0xF8u) == 0xF0u && pPos + 3 < pInput.size()) {
    unsigned char vC1 = static_cast<unsigned char>(pInput[pPos + 1]);
    unsigned char vC2 = static_cast<unsigned char>(pInput[pPos + 2]);
    unsigned char vC3 = static_cast<unsigned char>(pInput[pPos + 3]);
    if ((vC1 & 0xC0u) != 0x80u || (vC2 & 0xC0u) != 0x80u || (vC3 & 0xC0u) != 0x80u) {
      pAdvance = 1;
      return 0xFFFD;
    }
    pAdvance = 4;
    return ((vC & 0x07u) << 18) | ((static_cast<unsigned char>(pInput[pPos + 1]) & 0x3Fu) << 12) |
           ((static_cast<unsigned char>(pInput[pPos + 2]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(pInput[pPos + 3]) & 0x3Fu);
  }

  pAdvance = 1;
  return 0xFFFD; // replacement character for invalid input
}

static void fAppendUtf8(char32_t pCh, std::string &pOut) {
  if (pCh <= 0x7F) {
    pOut.push_back(static_cast<unsigned char>(pCh));
  } else if (pCh <= 0x7FF) {
    pOut.push_back(static_cast<unsigned char>(0xC0 | ((pCh >> 6) & 0x1F)));
    pOut.push_back(static_cast<unsigned char>(0x80 | (pCh & 0x3F)));
  } else if (pCh <= 0xFFFF) {
    pOut.push_back(static_cast<unsigned char>(0xE0 | ((pCh >> 12) & 0x0F)));
    pOut.push_back(static_cast<unsigned char>(0x80 | ((pCh >> 6) & 0x3F)));
    pOut.push_back(static_cast<unsigned char>(0x80 | (pCh & 0x3F)));
  } else if (pCh <= 0x10FFFF) {
    pOut.push_back(static_cast<unsigned char>(0xF0 | ((pCh >> 18) & 0x07)));
    pOut.push_back(static_cast<unsigned char>(0x80 | ((pCh >> 12) & 0x3F)));
    pOut.push_back(static_cast<unsigned char>(0x80 | ((pCh >> 6) & 0x3F)));
    pOut.push_back(static_cast<unsigned char>(0x80 | (pCh & 0x3F)));
  } else {
    // Invalid Unicode code point
  }
}

// Helper function to remove the last UTF-8 character from a string
static size_t fPrevUtf8CharPos(const std::string &pLine, size_t pPos) {
  if (pPos == 0)
    return 0;
  pPos--;
  while (pPos > 0 && (pLine[pPos] & 0xC0) == 0x80) {
    pPos--;
  }
  return pPos;
}

static size_t fNextUtf8CharPos(const std::string &pLine, size_t pPos) {
  if (pPos >= pLine.length())
    return pLine.length();
  pPos++;
  while (pPos < pLine.length() && (pLine[pPos] & 0xC0) == 0x80) {
    pPos++;
  }
  return pPos;
}

static void fMoveCursor(int pDelta);
static void fMoveWordLeft(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                          const std::string &pLine);
static void fMoveWordRight(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                           const std::string &pLine);
static void fMoveToLineStart(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths);
static void fMoveToLineEnd(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                           const std::string &pLine);

static void fDeleteAtCursor(std::string &pLine, std::vector<int> &pWidths, size_t &pCharPos, size_t &pBytePos) {
  if (pCharPos >= pWidths.size()) {
    return;
  }

  size_t vNextPos = fNextUtf8CharPos(pLine, pBytePos);
  int vW = pWidths[pCharPos];
  size_t vCharLen = vNextPos - pBytePos;

  pLine.erase(pBytePos, vCharLen);
  pWidths.erase(pWidths.begin() + pCharPos);

  size_t vP = pBytePos;
  int vTailWidth = 0;
  for (size_t vI = pCharPos; vI < pWidths.size(); ++vI) {
    size_t vFollowing = fNextUtf8CharPos(pLine, vP);
    fPutCodepoint(pLine.c_str() + vP, vFollowing - vP, pWidths[vI]);
    vTailWidth += pWidths[vI];
    vP = vFollowing;
  }

  for (int vI = 0; vI < vW; ++vI) {
    fputc(' ', vOut);
  }

  fMoveCursor(-(vTailWidth + vW));
}

static void fClearCurrentLine(const std::vector<int> &pWidths) {
  int vTotalWidth = 0;
  for (int vW : pWidths) {
    vTotalWidth += (vW > 0 ? vW : 1);
  }

  if (vTotalWidth > 0) {
    std::string vSpaces(vTotalWidth, ' ');
    fwrite(vSpaces.c_str(), 1, vTotalWidth, vOut);
    fMoveCursor(-vTotalWidth);
  }
}

static void fSetLineContents(std::string pNewLine, std::string &pLine, std::vector<int> &pWidths, size_t &pCharPos,
                             size_t &pBytePos, int pCursorBytePos = -1) {
  fMoveToLineStart(pCharPos, pBytePos, pWidths);
  fClearCurrentLine(pWidths);

  pLine = std::move(pNewLine);
  pWidths.clear();
  pBytePos = 0;
  pCharPos = 0;

  size_t vIdx = 0;
  int vBackWidth = 0;
  while (vIdx < pLine.size()) {
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vIdx, vAdvance);
    int vExpectedWidth = fEstimateWidth(vCp);
    int vRealWidth = fPutCodepoint(pLine.c_str() + vIdx, vAdvance, vExpectedWidth);
    if (vRealWidth < 0)
      vRealWidth = 0;
    pWidths.push_back(vRealWidth);
    vIdx += vAdvance;
    if (pCursorBytePos >= 0 && static_cast<size_t>(pCursorBytePos) < vIdx) {
      vBackWidth += vRealWidth;
    } else {
      ++pCharPos;
      pBytePos = vIdx;
    }
  }
  if (pCursorBytePos >= 0) {
    fMoveCursor(-vBackWidth);
  }
}

static void fMoveToLineStart(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths) {
  int vBackWidth = 0;
  for (size_t vI = 0; vI < pCharPos; ++vI) {
    vBackWidth += pWidths[vI];
  }
  fMoveCursor(-vBackWidth);
  pCharPos = 0;
  pBytePos = 0;
}

static void fMoveToLineEnd(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                           const std::string &pLine) {
  int vForwardWidth = 0;
  for (size_t vI = pCharPos; vI < pWidths.size(); ++vI) {
    vForwardWidth += pWidths[vI];
  }
  fMoveCursor(vForwardWidth);
  pCharPos = pWidths.size();
  pBytePos = pLine.length();
}

static bool fHasCtrlModifier(const std::string &pParams) {
  size_t vStart = 0;
  while (vStart < pParams.size()) {
    size_t vEnd = pParams.find(';', vStart);
    size_t vLen = (vEnd == std::string::npos) ? pParams.size() - vStart : vEnd - vStart;
    if (vLen > 0) {
      int vValue = 0;
      for (size_t vI = 0; vI < vLen; ++vI) {
        char vCh = pParams[vStart + vI];
        if (!std::isdigit(static_cast<unsigned char>(vCh))) {
          vValue = -1;
          break;
        }
        vValue = vValue * 10 + (vCh - '0');
      }
      if (vValue == 5) {
        return true;
      }
    }

    if (vEnd == std::string::npos) {
      break;
    }
    vStart = vEnd + 1;
  }
  return false;
}

static bool fIsSpaceCodepoint(char32_t pCp) { return std::iswspace(static_cast<wint_t>(pCp)) != 0; }

static void fMoveWordLeft(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                          const std::string &pLine) {
  if (pCharPos == 0) {
    return;
  }

  size_t vNewCharPos = pCharPos;
  size_t vNewBytePos = pBytePos;
  int vMoveWidth = 0;

  while (vNewCharPos > 0) {
    size_t vPrevByte = fPrevUtf8CharPos(pLine, vNewBytePos);
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vPrevByte, vAdvance);
    if (!fIsSpaceCodepoint(vCp)) {
      break;
    }
    vMoveWidth += pWidths[vNewCharPos - 1];
    vNewCharPos--;
    vNewBytePos = vPrevByte;
  }

  while (vNewCharPos > 0) {
    size_t vPrevByte = fPrevUtf8CharPos(pLine, vNewBytePos);
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vPrevByte, vAdvance);
    if (fIsSpaceCodepoint(vCp)) {
      break;
    }
    vMoveWidth += pWidths[vNewCharPos - 1];
    vNewCharPos--;
    vNewBytePos = vPrevByte;
  }

  fMoveCursor(-vMoveWidth);
  pCharPos = vNewCharPos;
  pBytePos = vNewBytePos;
}

static void fMoveWordRight(size_t &pCharPos, size_t &pBytePos, const std::vector<int> &pWidths,
                           const std::string &pLine) {
  if (pCharPos >= pWidths.size()) {
    return;
  }

  size_t vNewCharPos = pCharPos;
  size_t vNewBytePos = pBytePos;
  int vMoveWidth = 0;

  while (vNewCharPos < pWidths.size()) {
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vNewBytePos, vAdvance);
    if (!fIsSpaceCodepoint(vCp)) {
      break;
    }
    vMoveWidth += pWidths[vNewCharPos];
    vNewCharPos++;
    vNewBytePos += vAdvance;
  }

  while (vNewCharPos < pWidths.size()) {
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vNewBytePos, vAdvance);
    if (fIsSpaceCodepoint(vCp)) {
      break;
    }
    vMoveWidth += pWidths[vNewCharPos];
    vNewCharPos++;
    vNewBytePos += vAdvance;
  }

  while (vNewCharPos < pWidths.size()) {
    size_t vAdvance = 0;
    char32_t vCp = fDecodeUtf8(pLine, vNewBytePos, vAdvance);
    if (!fIsSpaceCodepoint(vCp)) {
      break;
    }
    vMoveWidth += pWidths[vNewCharPos];
    vNewCharPos++;
    vNewBytePos += vAdvance;
  }

  fMoveCursor(vMoveWidth);
  pCharPos = vNewCharPos;
  pBytePos = vNewBytePos;
}

static void fMoveCursor(int pDelta) {
  if (pDelta == 0)
    return;
#if defined(_WIN32)
  if (vHConsole != NULL) {
    CONSOLE_SCREEN_BUFFER_INFO vBufferInfo;
    GetConsoleScreenBufferInfo(vHConsole, &vBufferInfo);
    COORD vNewCursorPosition = vBufferInfo.dwCursorPosition;
    int vWidth = vBufferInfo.dwSize.X;
    int vNewX = vNewCursorPosition.X + pDelta;
    int vNewY = vNewCursorPosition.Y;

    while (vNewX >= vWidth) {
      vNewX -= vWidth;
      vNewY++;
    }
    while (vNewX < 0) {
      vNewX += vWidth;
      vNewY--;
    }

    vNewCursorPosition.X = vNewX;
    vNewCursorPosition.Y = vNewY;
    SetConsoleCursorPosition(vHConsole, vNewCursorPosition);
  }
#else
  if (pDelta < 0) {
    for (int vI = 0; vI < -pDelta; vI++)
      fprintf(vOut, "\b");
  } else {
    for (int vI = 0; vI < pDelta; vI++)
      fprintf(vOut, "\033[C");
  }
#endif
}

struct history_t {
  std::vector<std::string> lEntries;
  size_t vViewingIdx = SIZE_MAX;
  std::string vBackupLine; // current line before viewing history
  void mAdd(std::string_view pLine) {
    if (pLine.empty()) {
      return;
    }
    // avoid duplicates with the last entry
    if (lEntries.empty() || lEntries.back() != pLine) {
      lEntries.emplace_back(pLine);
    }
    // also clear viewing state
    mEndViewing();
  }
  bool mPrev(std::string &pCurLine) {
    if (lEntries.empty()) {
      return false;
    }
    if (vViewingIdx == SIZE_MAX) {
      return false;
    }
    if (vViewingIdx > 0) {
      vViewingIdx--;
    }
    pCurLine = lEntries[vViewingIdx];
    return true;
  }
  bool mNext(std::string &pCurLine) {
    if (lEntries.empty() || vViewingIdx == SIZE_MAX) {
      return false;
    }
    vViewingIdx++;
    if (vViewingIdx >= lEntries.size()) {
      pCurLine = vBackupLine;
      mEndViewing();
    } else {
      pCurLine = lEntries[vViewingIdx];
    }
    return true;
  }
  void mBeginViewing(const std::string &pLine) {
    vBackupLine = pLine;
    vViewingIdx = lEntries.size();
  }
  void mEndViewing() {
    vViewingIdx = SIZE_MAX;
    vBackupLine.clear();
  }
  bool mIsViewing() const { return vViewingIdx != SIZE_MAX; }
} vHistory;

static bool fReadlineAdvanced(std::string &pLine, bool pMultilineInput) {
  if (vOut != stdout) {
    fflush(stdout);
  }

  pLine.clear();
  std::vector<int> lWidths;
  bool vIsSpecialChar = false;
  bool vEndOfStream = false;

  size_t vBytePos = 0; // current byte index
  size_t vCharPos = 0; // current character index (one char can be multiple bytes)

  char32_t vInputChar;
  while (true) {
    assert(vCharPos <= vBytePos);
    assert(vCharPos <= lWidths.size());
    auto vHistoryPrev = [&]() {
      if (!vHistory.mIsViewing()) {
        vHistory.mBeginViewing(pLine);
      }
      std::string vNewLine;
      if (!vHistory.mPrev(vNewLine)) {
        return;
      }
      fSetLineContents(vNewLine, pLine, lWidths, vCharPos, vBytePos);
    };
    auto vHistoryNext = [&]() {
      if (vHistory.mIsViewing()) {
        std::string vNewLine;
        if (!vHistory.mNext(vNewLine)) {
          return;
        }
        fSetLineContents(vNewLine, pLine, lWidths, vCharPos, vBytePos);
      }
    };

    fflush(vOut); // Ensure all output is displayed before waiting for input
    vInputChar = fGetchar32();

    if (vInputChar == '\r' || vInputChar == '\n') {
      break;
    }

    if (vCompletionCb && vInputChar == '\t') {
      auto vCandidates = vCompletionCb(pLine, vBytePos);

      if (!vCandidates.empty()) {
        if (vCandidates.size() > 1 || vCandidates[0].first != pLine) {
          // TODO?: Display all candidates
          fSetLineContents(vCandidates[0].first, pLine, lWidths, vCharPos, vBytePos, vCandidates[0].second);
        } else {
          // TODO: Move cursor to new byte_pos
        }
        continue;
      }
    }

    if (vInputChar == (char32_t)WEOF || vInputChar == 0x04 /* Ctrl+D */) {
      vEndOfStream = true;
      break;
    }

    if (vIsSpecialChar) {
      fReplaceLast(pLine.back());
      vIsSpecialChar = false;
    }

    if (vInputChar == '\033') { // Escape sequence
      char32_t vCode = fGetchar32();
      if (vCode == '[') {
        std::string vParams;
        while (true) {
          vCode = fGetchar32();
          if ((vCode >= 'A' && vCode <= 'Z') || (vCode >= 'a' && vCode <= 'z') || vCode == '~' ||
              vCode == (char32_t)WEOF) {
            break;
          }
          vParams.push_back(static_cast<char>(vCode));
        }

        const bool cCtrlModifier = fHasCtrlModifier(vParams);

        if (vCode == 'D') { // left
          if (cCtrlModifier) {
            fMoveWordLeft(vCharPos, vBytePos, lWidths, pLine);
          } else if (vCharPos > 0) {
            int vW = lWidths[vCharPos - 1];
            fMoveCursor(-vW);
            vCharPos--;
            vBytePos = fPrevUtf8CharPos(pLine, vBytePos);
          }
        } else if (vCode == 'C') { // right
          if (cCtrlModifier) {
            fMoveWordRight(vCharPos, vBytePos, lWidths, pLine);
          } else if (vCharPos < lWidths.size()) {
            int vW = lWidths[vCharPos];
            fMoveCursor(vW);
            vCharPos++;
            vBytePos = fNextUtf8CharPos(pLine, vBytePos);
          }
        } else if (vCode == 'H') { // home
          fMoveToLineStart(vCharPos, vBytePos, lWidths);
        } else if (vCode == 'F') { // end
          fMoveToLineEnd(vCharPos, vBytePos, lWidths, pLine);
        } else if (vCode == 'A' || vCode == 'B') {
          // up/down
          if (vCode == 'A') {
            vHistoryPrev();
            vIsSpecialChar = false;
          } else if (vCode == 'B') {
            vHistoryNext();
            vIsSpecialChar = false;
          }
        } else if ((vCode == '~' || (vCode >= 'A' && vCode <= 'Z') || (vCode >= 'a' && vCode <= 'z')) &&
                   !vParams.empty()) {
          std::string vDigits;
          for (char vCh : vParams) {
            if (vCh == ';') {
              break;
            }
            if (std::isdigit(static_cast<unsigned char>(vCh))) {
              vDigits.push_back(vCh);
            }
          }

          if (vCode == '~') {
            if (vDigits == "1" || vDigits == "7") { // home
              fMoveToLineStart(vCharPos, vBytePos, lWidths);
            } else if (vDigits == "4" || vDigits == "8") { // end
              fMoveToLineEnd(vCharPos, vBytePos, lWidths, pLine);
            } else if (vDigits == "3") { // delete
              fDeleteAtCursor(pLine, lWidths, vCharPos, vBytePos);
            }
          }
        }
      } else if (vCode == 0x1B) {
        // Discard the rest of the escape sequence
        while ((vCode = fGetchar32()) != (char32_t)WEOF) {
          if ((vCode >= 'A' && vCode <= 'Z') || (vCode >= 'a' && vCode <= 'z') || vCode == '~') {
            break;
          }
        }
      }
#if defined(_WIN32)
    } else if (vInputChar == cKeyArrowLeft) {
      if (vCharPos > 0) {
        int vW = lWidths[vCharPos - 1];
        fMoveCursor(-vW);
        vCharPos--;
        vBytePos = fPrevUtf8CharPos(pLine, vBytePos);
      }
    } else if (vInputChar == cKeyArrowRight) {
      if (vCharPos < lWidths.size()) {
        int vW = lWidths[vCharPos];
        fMoveCursor(vW);
        vCharPos++;
        vBytePos = fNextUtf8CharPos(pLine, vBytePos);
      }
    } else if (vInputChar == cKeyCtrlArrowLeft) {
      fMoveWordLeft(vCharPos, vBytePos, lWidths, pLine);
    } else if (vInputChar == cKeyCtrlArrowRight) {
      fMoveWordRight(vCharPos, vBytePos, lWidths, pLine);
    } else if (vInputChar == cKeyHome) {
      fMoveToLineStart(vCharPos, vBytePos, lWidths);
    } else if (vInputChar == cKeyEnd) {
      fMoveToLineEnd(vCharPos, vBytePos, lWidths, pLine);
    } else if (vInputChar == cKeyDelete) {
      fDeleteAtCursor(pLine, lWidths, vCharPos, vBytePos);
    } else if (vInputChar == cKeyArrowUp || vInputChar == cKeyArrowDown) {
      if (vInputChar == cKeyArrowUp) {
        vHistoryPrev();
        vIsSpecialChar = false;
      } else if (vInputChar == cKeyArrowDown) {
        vHistoryNext();
        vIsSpecialChar = false;
      }
#endif
    } else if (vInputChar == 0x08 || vInputChar == 0x7F) { // Backspace
      if (vCharPos > 0) {
        int vW = lWidths[vCharPos - 1];
        fMoveCursor(-vW);
        vCharPos--;
        size_t vPrevPos = fPrevUtf8CharPos(pLine, vBytePos);
        size_t vCharLen = vBytePos - vPrevPos;
        vBytePos = vPrevPos;

        // remove the character
        pLine.erase(vBytePos, vCharLen);
        lWidths.erase(lWidths.begin() + vCharPos);

        // redraw tail
        size_t vP = vBytePos;
        int vTailWidth = 0;
        for (size_t vI = vCharPos; vI < lWidths.size(); ++vI) {
          size_t vNextP = fNextUtf8CharPos(pLine, vP);
          fPutCodepoint(pLine.c_str() + vP, vNextP - vP, lWidths[vI]);
          vTailWidth += lWidths[vI];
          vP = vNextP;
        }

        // clear display
        for (int vI = 0; vI < vW; ++vI) {
          fputc(' ', vOut);
        }
        fMoveCursor(-(vTailWidth + vW));
      }
    } else {
      // insert character
      std::string vNewCharStr;
      fAppendUtf8(vInputChar, vNewCharStr);
      int vW = fEstimateWidth(vInputChar);

      if (vCharPos == lWidths.size()) {
        // insert at the end
        pLine += vNewCharStr;
        int vRealW = fPutCodepoint(vNewCharStr.c_str(), vNewCharStr.length(), vW);
        if (vRealW < 0)
          vRealW = 0;
        lWidths.push_back(vRealW);
        vBytePos += vNewCharStr.length();
        vCharPos++;
      } else {
        // insert in middle
        pLine.insert(vBytePos, vNewCharStr);

        int vRealW = fPutCodepoint(vNewCharStr.c_str(), vNewCharStr.length(), vW);
        if (vRealW < 0)
          vRealW = 0;

        lWidths.insert(lWidths.begin() + vCharPos, vRealW);

        // print the tail
        size_t vP = vBytePos + vNewCharStr.length();
        int vTailWidth = 0;
        for (size_t vI = vCharPos + 1; vI < lWidths.size(); ++vI) {
          size_t vNextP = fNextUtf8CharPos(pLine, vP);
          fPutCodepoint(pLine.c_str() + vP, vNextP - vP, lWidths[vI]);
          vTailWidth += lWidths[vI];
          vP = vNextP;
        }

        fMoveCursor(-vTailWidth);

        vBytePos += vNewCharStr.length();
        vCharPos++;
      }
    }

    if (!pLine.empty() && (pLine.back() == '\\' || pLine.back() == '/')) {
      fReplaceLast(pLine.back());
      vIsSpecialChar = true;
    }
  }

  bool vHasMore = pMultilineInput;
  if (vIsSpecialChar) {
    fReplaceLast(' ');
    fPopCursor();

    char vLast = pLine.back();
    pLine.pop_back();
    if (vLast == '\\') {
      pLine += '\n';
      fputc('\n', vOut);
      vHasMore = !vHasMore;
    } else {
      // the tokenizer will just eat the single space, it won't act as a space
      if (pLine.length() == 1 && pLine.back() == ' ') {
        pLine.clear();
        fPopCursor();
      }
      vHasMore = false;
    }
  } else {
    if (vEndOfStream) {
      vHasMore = false;
    } else {
      pLine += '\n';
      fputc('\n', vOut);
    }
  }

  if (!vEndOfStream && !pLine.empty()) {
    // remove the trailing newline for history storage
    std::string_view vHline = pLine;
    if (!pLine.empty() && pLine.back() == '\n') {
      vHline.remove_suffix(1);
    }
    // TODO: maybe support multiline history entries?
    vHistory.mAdd(vHline);
  }

  fflush(vOut);
  return vHasMore;
}

static bool fReadlineSimple(std::string &pLine, bool pMultilineInput) {
#if defined(_WIN32)
  std::wstring vWline;
  if (!std::getline(std::wcin, vWline)) {
    // Input stream is bad or EOF received
    pLine.clear();
    GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
    return false;
  }

  int vSizeNeeded = WideCharToMultiByte(CP_UTF8, 0, &vWline[0], (int)vWline.size(), NULL, 0, NULL, NULL);
  pLine.resize(vSizeNeeded);
  WideCharToMultiByte(CP_UTF8, 0, &vWline[0], (int)vWline.size(), &pLine[0], vSizeNeeded, NULL, NULL);
#else
  if (!std::getline(std::cin, pLine)) {
    // Input stream is bad or EOF received
    pLine.clear();
    return false;
  }
#endif
  if (!pLine.empty()) {
    char vLast = pLine.back();
    if (vLast == '/') { // Always return control on '/' symbol
      pLine.pop_back();
      return false;
    }
    if (vLast == '\\') { // '\\' changes the default action
      pLine.pop_back();
      pMultilineInput = !pMultilineInput;
    }
  }
  pLine += '\n';

  // By default, continue input if multiline_input is set
  return pMultilineInput;
}

bool fReadline(std::string &pLine, bool pMultilineInput) {
  if (vSimpleIo) {
    return fReadlineSimple(pLine, pMultilineInput);
  }
  return fReadlineAdvanced(pLine, pMultilineInput);
}

void fSetCompletionCallback(completion_callback pCb) { vCompletionCb = pCb; }

namespace spinner {
static const char aLoadingChars[] = {'|', '/', '-', '\\'};
static std::condition_variable vCvStop;
static std::thread vTh;
static size_t vFrame = 0; // only modified by one thread
static bool vRunning = false;
static std::mutex vMtx;
static auto vWaitTime = std::chrono::milliseconds(100);
static void fDrawNextFrame() {
  // don't need lock because only one thread modifies running
  vFrame = (vFrame + 1) % sizeof(aLoadingChars);
  fReplaceLast(aLoadingChars[vFrame]);
  fflush(vOut);
}
void fStart() {
  std::unique_lock<std::mutex> vLock(vMtx);
  if (vSimpleIo || vRunning) {
    return;
  }
  fCommonLogFlush(fCommonLogMain());
  fprintf(vOut, "%c", aLoadingChars[0]);
  fflush(vOut);
  vFrame = 1;
  vRunning = true;
  vTh = std::thread([]() {
    std::unique_lock<std::mutex> vLock(vMtx);
    while (true) {
      if (vCvStop.wait_for(vLock, vWaitTime, [] { return !vRunning; })) {
        break;
      }
      fDrawNextFrame();
    }
  });
}
void fStop() {
  {
    std::unique_lock<std::mutex> vLock(vMtx);
    if (vSimpleIo || !vRunning) {
      return;
    }
    vRunning = false;
    vCvStop.notify_all();
  }
  if (vTh.joinable()) {
    vTh.join();
  }
  fReplaceLast(' ');
  fPopCursor();
  fflush(vOut);
}
} // namespace spinner

void fLog(const char *pFmt, ...) {
  va_list vArgs;
  va_start(vArgs, pFmt);
  vfprintf(vOut, pFmt, vArgs);
  va_end(vArgs);
}

void fError(const char *pFmt, ...) {
  va_list vArgs;
  va_start(vArgs, pFmt);
  display_type vCur = vCurrentDisplay;
  fSetDisplay(cDisplayTypeError);
  vfprintf(vOut, pFmt, vArgs);
  fSetDisplay(vCur); // restore previous color
  va_end(vArgs);
}

void fFlush() { fflush(vOut); }
} // namespace console
