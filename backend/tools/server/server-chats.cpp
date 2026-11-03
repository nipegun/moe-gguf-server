#include "server-chats.h"
#include "server-ui-settings.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static const char *cChatsFolder = ".mgs/chats"; // under the home of the user that runs mgs
static constexpr int cMaxChatNumber = 99;
static constexpr int cDefaultSearchLimit = 20;
static constexpr int cMaxSearchLimit = 100;

// A chat of the folder: its file and its conversation in the form of the API (dates in milliseconds).
struct server_chat_entry {
  fs::path vPath;
  json dConversation;
};

// One request at a time reads or writes the folder; the index is read from it the first time it is needed.
static std::mutex vChatsMutex;
static bool vChatsLoaded = false;
static std::map<std::string, server_chat_entry> dChats;
// Changes with every write; it starts at the time the server started, so a restart is noticed as a change as well.
static std::atomic<uint64_t> vChatsVersion{(uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count()};

static int64_t fNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string fChatsNativeDate(int64_t pMs) {
  if (pMs < 0) {
    pMs = 0;
  }
  const time_t cSeconds = (time_t)(pMs / 1000);
  std::tm vTm{};
#if defined(_WIN32)
  gmtime_s(&vTm, &cSeconds);
#else
  gmtime_r(&cSeconds, &vTm);
#endif
  char aText[32];
  std::snprintf(aText, sizeof(aText), "%04d-%02d-%02d@%02d-%02d-%02d.%03d", vTm.tm_year + 1900, vTm.tm_mon + 1,
                vTm.tm_mday, vTm.tm_hour, vTm.tm_min, vTm.tm_sec, (int)(pMs % 1000));
  return aText;
}

int64_t fChatsDateToMs(const std::string &pDate) {
  int vYear = 0, vMonth = 0, vDay = 0, vHour = 0, vMinute = 0, vSecond = 0, vMillisecond = 0;
  char aRest[2] = {0};
  if (std::sscanf(pDate.c_str(), "%4d-%2d-%2d@%2d-%2d-%2d.%3d%1s", &vYear, &vMonth, &vDay, &vHour, &vMinute,
                  &vSecond, &vMillisecond, aRest) != 7 ||
      pDate.size() != 23) {
    return -1;
  }
  std::tm vTm{};
  vTm.tm_year = vYear - 1900;
  vTm.tm_mon = vMonth - 1;
  vTm.tm_mday = vDay;
  vTm.tm_hour = vHour;
  vTm.tm_min = vMinute;
  vTm.tm_sec = vSecond;
#if defined(_WIN32)
  const time_t cSeconds = _mkgmtime(&vTm);
#else
  const time_t cSeconds = timegm(&vTm);
#endif
  if (cSeconds < 0) {
    return -1;
  }
  return (int64_t)cSeconds * 1000 + vMillisecond;
}

//
// Dates: native in the files, milliseconds in the API
//

static const char *aConversationDates[] = {"createdAt", "lastModified"};
static const char *aMessageDates[] = {"timestamp"};

template <size_t N> static json fDatesToFile(json pObject, const char *const (&pFields)[N]) {
  for (const char *vField : pFields) {
    if (pObject.contains(vField) && pObject.at(vField).is_number()) {
      pObject[vField] = fChatsNativeDate(pObject.at(vField).get<int64_t>());
    }
  }
  return pObject;
}

template <size_t N> static json fDatesToApi(json pObject, const char *const (&pFields)[N]) {
  for (const char *vField : pFields) {
    if (pObject.contains(vField) && pObject.at(vField).is_string()) {
      const int64_t cMs = fChatsDateToMs(pObject.at(vField).get<std::string>());
      if (cMs >= 0) {
        pObject[vField] = cMs;
      } else {
        pObject.erase(vField);
      }
    }
  }
  return pObject;
}

//
// Files
//

static fs::path fChatsFolderPath() {
  const std::string cHome = fConfigHomeDirectory();
  return cHome.empty() ? fs::path() : fs::path(cHome) / cChatsFolder;
}

static bool fEnsureChatsFolder(std::string &pError) {
  const fs::path cFolder = fChatsFolderPath();
  if (cFolder.empty()) {
    pError = "the home folder of the user that runs mgs is unknown";
    return false;
  }
#if defined(_WIN32)
  std::error_code vCode;
  fs::create_directories(cFolder, vCode);
  if (vCode) {
    pError = "cannot create " + cFolder.string() + ": " + vCode.message();
    return false;
  }
#else
  for (const fs::path &vFolder : {cFolder.parent_path(), cFolder}) {
    if (mkdir(vFolder.c_str(), 0700) != 0 && errno != EEXIST) {
      pError = "cannot create " + vFolder.string() + ": " + std::strerror(errno);
      return false;
    }
  }
#endif
  return true;
}

// Reads a chat file: its conversation and, with pMessages, its messages, both in the form of the API. False when the
// file cannot be read or its first line is not a conversation.
static bool fReadChatFile(const fs::path &pPath, json &pConversation, json *pMessages) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return false;
  }
  std::string vLine;
  if (!std::getline(vFile, vLine)) {
    return false;
  }
  const json cFirst = json::parse(vLine, nullptr, false);
  if (!cFirst.is_object() || !cFirst.contains("conversation") || !cFirst.at("conversation").is_object() ||
      !cFirst.at("conversation").contains("id") || !cFirst.at("conversation").at("id").is_string()) {
    return false;
  }
  pConversation = fDatesToApi(cFirst.at("conversation"), aConversationDates);
  if (pMessages == nullptr) {
    return true;
  }
  *pMessages = json::array();
  while (std::getline(vFile, vLine)) {
    if (vLine.empty()) {
      continue;
    }
    const json cItem = json::parse(vLine, nullptr, false);
    if (cItem.is_object() && cItem.contains("message") && cItem.at("message").is_object()) {
      pMessages->push_back(fDatesToApi(cItem.at("message"), aMessageDates));
    }
  }
  return true;
}

// Writes a chat file through a temporary file that then replaces the old one, readable only by this user.
static bool fWriteChatFile(const fs::path &pPath, const json &pConversation, const json &pMessages,
                           std::string &pError) {
  std::string vText = json{{"conversation", fDatesToFile(pConversation, aConversationDates)}}.dump() + "\n";
  for (const json &cMessage : pMessages) {
    vText += json{{"message", fDatesToFile(cMessage, aMessageDates)}}.dump() + "\n";
  }
#if defined(_WIN32)
  const fs::path cTemporary = pPath.string() + ".tmp";
  {
    std::ofstream vFile(cTemporary, std::ios::binary | std::ios::trunc);
    vFile << vText;
    if (!vFile) {
      pError = "cannot write " + cTemporary.string();
      return false;
    }
  }
  std::error_code vCode;
  fs::rename(cTemporary, pPath, vCode);
  if (vCode) {
    pError = "cannot replace " + pPath.string() + ": " + vCode.message();
    return false;
  }
  return true;
#else
  const fs::path cTemporary = pPath.string() + ".tmp-" + std::to_string(getpid());
  const int cFd = open(cTemporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (cFd < 0) {
    pError = "cannot create " + cTemporary.string() + ": " + std::strerror(errno);
    return false;
  }
  size_t vWritten = 0;
  while (vWritten < vText.size()) {
    const ssize_t cChunk = write(cFd, vText.data() + vWritten, vText.size() - vWritten);
    if (cChunk < 0) {
      if (errno == EINTR) {
        continue;
      }
      pError = "cannot write " + cTemporary.string() + ": " + std::strerror(errno);
      close(cFd);
      unlink(cTemporary.c_str());
      return false;
    }
    vWritten += static_cast<size_t>(cChunk);
  }
  if (fsync(cFd) != 0 || close(cFd) != 0) {
    pError = "cannot write " + cTemporary.string() + ": " + std::strerror(errno);
    unlink(cTemporary.c_str());
    return false;
  }
  if (rename(cTemporary.c_str(), pPath.c_str()) != 0) {
    pError = "cannot replace " + pPath.string() + ": " + std::strerror(errno);
    unlink(cTemporary.c_str());
    return false;
  }
  return true;
#endif
}

// Reads the conversations of the folder the first time; vChatsMutex must be held.
static void fLoadChatsIndex() {
  if (vChatsLoaded) {
    return;
  }
  vChatsLoaded = true;
  const fs::path cFolder = fChatsFolderPath();
  std::error_code vCode;
  if (cFolder.empty() || !fs::is_directory(cFolder, vCode)) {
    return;
  }
  for (const fs::directory_entry &vEntry : fs::directory_iterator(cFolder, vCode)) {
    if (!vEntry.is_regular_file(vCode) || vEntry.path().extension() != ".jsonl") {
      continue;
    }
    json vConversation;
    if (!fReadChatFile(vEntry.path(), vConversation, nullptr)) {
      SRV_WRN("chats: %s is not a chat file, it is left out\n", vEntry.path().string().c_str());
      continue;
    }
    const std::string cId = vConversation.at("id").get<std::string>();
    if (dChats.count(cId) > 0) {
      SRV_WRN("chats: %s repeats the chat %s, it is left out\n", vEntry.path().string().c_str(), cId.c_str());
      continue;
    }
    dChats[cId] = server_chat_entry{vEntry.path(), vConversation};
  }
  SRV_INF("chats: %zu in %s\n", dChats.size(), cFolder.string().c_str());
}

// The file for a new chat: its creation date, with -2, -3... when that name is taken.
static fs::path fNewChatPath(int64_t pCreatedAt) {
  const fs::path cFolder = fChatsFolderPath();
  const std::string cBase = fChatsNativeDate(pCreatedAt);
  std::error_code vCode;
  for (int vSuffix = 1;; vSuffix++) {
    const fs::path cPath = cFolder / (cBase + (vSuffix > 1 ? "-" + std::to_string(vSuffix) : "") + ".jsonl");
    if (!fs::exists(cPath, vCode)) {
      return cPath;
    }
  }
}

// The lowest number from 1 to 99 that no chat other than pId holds, or 0 when all of them are taken.
static int fFreeChatNumber(const std::string &pId) {
  std::vector<bool> aTaken(cMaxChatNumber + 1, false);
  for (const auto &[vId, vEntry] : dChats) {
    if (vId == pId || !vEntry.dConversation.contains("number") || !vEntry.dConversation.at("number").is_number()) {
      continue;
    }
    const int cNumber = vEntry.dConversation.at("number").get<int>();
    if (cNumber >= 1 && cNumber <= cMaxChatNumber) {
      aTaken[cNumber] = true;
    }
  }
  for (int vNumber = 1; vNumber <= cMaxChatNumber; vNumber++) {
    if (!aTaken[vNumber]) {
      return vNumber;
    }
  }
  return 0;
}

static bool fChatNumberTaken(int pNumber, const std::string &pId) {
  for (const auto &[vId, vEntry] : dChats) {
    if (vId != pId && vEntry.dConversation.contains("number") && vEntry.dConversation.at("number").is_number() &&
        vEntry.dConversation.at("number").get<int>() == pNumber) {
      return true;
    }
  }
  return false;
}

// Ids of chats and messages are UUIDs or similar: letters, digits, '-' and '_'.
static bool fValidChatId(const std::string &pId) {
  if (pId.empty() || pId.size() > 128) {
    return false;
  }
  return std::all_of(pId.begin(), pId.end(),
                     [](unsigned char pC) { return std::isalnum(pC) || pC == '-' || pC == '_'; });
}

//
// Search: text without case or accents
//

// Latin-1 Supplement and Latin Extended-A (U+00C0 to U+017F) without accents, in lower case.
static const char *const aLatinFold[] = {
    "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i",
    "d", "n", "o", "o", "o", "o", "o", "\xC3\x97", "o", "u", "u", "u", "u", "y", "th", "ss",
    "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i",
    "d", "n", "o", "o", "o", "o", "o", "\xC3\xB7", "o", "u", "u", "u", "u", "y", "th", "y",
    "a", "a", "a", "a", "a", "a", "c", "c", "c", "c", "c", "c", "c", "c", "d", "d",
    "d", "d", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "g", "g", "g", "g",
    "g", "g", "g", "g", "h", "h", "h", "h", "i", "i", "i", "i", "i", "i", "i", "i",
    "i", "i", "ij", "ij", "j", "j", "k", "k", "k", "l", "l", "l", "l", "l", "l", "l",
    "l", "l", "l", "n", "n", "n", "n", "n", "n", "n", "n", "n", "o", "o", "o", "o",
    "o", "o", "oe", "oe", "r", "r", "r", "r", "r", "r", "s", "s", "s", "s", "s", "s",
    "s", "s", "t", "t", "t", "t", "t", "t", "u", "u", "u", "u", "u", "u", "u", "u",
    "u", "u", "u", "u", "w", "w", "y", "y", "y", "z", "z", "z", "z", "z", "z", "s",
};

static void fAppendUtf8(std::string &pOut, uint32_t pCode) {
  if (pCode < 0x80) {
    pOut += (char)pCode;
  } else if (pCode < 0x800) {
    pOut += (char)(0xC0 | (pCode >> 6));
    pOut += (char)(0x80 | (pCode & 0x3F));
  } else if (pCode < 0x10000) {
    pOut += (char)(0xE0 | (pCode >> 12));
    pOut += (char)(0x80 | ((pCode >> 6) & 0x3F));
    pOut += (char)(0x80 | (pCode & 0x3F));
  } else {
    pOut += (char)(0xF0 | (pCode >> 18));
    pOut += (char)(0x80 | ((pCode >> 12) & 0x3F));
    pOut += (char)(0x80 | ((pCode >> 6) & 0x3F));
    pOut += (char)(0x80 | (pCode & 0x3F));
  }
}

// Lower case without the accents of Latin letters (and without combining marks), Greek and Cyrillic in lower case.
static std::string fChatsFoldText(const std::string &pText) {
  std::string vOut;
  vOut.reserve(pText.size());
  for (size_t vI = 0; vI < pText.size();) {
    const unsigned char cByte = (unsigned char)pText[vI];
    uint32_t vCode = cByte;
    size_t vLength = 1;
    if (cByte >= 0xF0 && vI + 3 < pText.size()) {
      vCode = ((cByte & 0x07) << 18) | ((pText[vI + 1] & 0x3F) << 12) | ((pText[vI + 2] & 0x3F) << 6) |
              (pText[vI + 3] & 0x3F);
      vLength = 4;
    } else if (cByte >= 0xE0 && vI + 2 < pText.size()) {
      vCode = ((cByte & 0x0F) << 12) | ((pText[vI + 1] & 0x3F) << 6) | (pText[vI + 2] & 0x3F);
      vLength = 3;
    } else if (cByte >= 0xC0 && vI + 1 < pText.size()) {
      vCode = ((cByte & 0x1F) << 6) | (pText[vI + 1] & 0x3F);
      vLength = 2;
    }
    vI += vLength;

    if (vCode >= 'A' && vCode <= 'Z') {
      vOut += (char)(vCode + 32);
    } else if (vCode >= 0xC0 && vCode <= 0x17F) {
      vOut += aLatinFold[vCode - 0xC0];
    } else if (vCode >= 0x300 && vCode <= 0x36F) {
      // combining marks: the base letter already went out
    } else if (vCode >= 0x391 && vCode <= 0x3A9 && vCode != 0x3A2) {
      fAppendUtf8(vOut, vCode + 0x20);
    } else if (vCode >= 0x410 && vCode <= 0x42F) {
      fAppendUtf8(vOut, vCode + 0x20);
    } else if (vCode >= 0x400 && vCode <= 0x40F) {
      fAppendUtf8(vOut, vCode + 0x50);
    } else if (vLength == 1) {
      vOut += (char)vCode;
    } else {
      vOut.append(pText, vI - vLength, vLength);
    }
  }
  return vOut;
}

static std::vector<std::string> fSearchWords(const std::string &pQuery) {
  std::vector<std::string> lWords;
  std::istringstream vStream(fChatsFoldText(pQuery));
  std::string vWord;
  while (vStream >> vWord) {
    lWords.push_back(vWord);
  }
  return lWords;
}

//
// Responses
//

static server_http_res_ptr fChatsJson(const json &pData) {
  auto vRes = std::make_unique<server_http_res>();
  vRes->vData = fSafeJsonToStr(pData);
  return vRes;
}

static server_http_res_ptr fChatsError(const std::string &pMessage, enum error_type pType) {
  auto vRes = std::make_unique<server_http_res>();
  const json cError = fFormatErrorResponse(pMessage, pType);
  vRes->vStatus = cError.at("code").get<int>();
  vRes->vData = fSafeJsonToStr({{"error", cError}});
  return vRes;
}

//
// Handlers
//

server_http_context::handler_t fChatsMakeListHandler() {
  return [](const server_http_req &) {
    json aConversations = json::array();
    {
      std::lock_guard<std::mutex> vLock(vChatsMutex);
      fLoadChatsIndex();
      for (const auto &[vId, vEntry] : dChats) {
        aConversations.push_back(vEntry.dConversation);
      }
    }
    std::sort(aConversations.begin(), aConversations.end(), [](const json &pA, const json &pB) {
      return pA.value("lastModified", (int64_t)0) > pB.value("lastModified", (int64_t)0);
    });
    return fChatsJson({{"version", vChatsVersion.load()}, {"conversations", aConversations}});
  };
}

server_http_context::handler_t fChatsMakeVersionHandler() {
  return [](const server_http_req &) { return fChatsJson({{"version", vChatsVersion.load()}}); };
}

server_http_context::handler_t fChatsMakeSearchHandler() {
  return [](const server_http_req &pReq) {
    const std::vector<std::string> lWords = fSearchWords(pReq.mGetParam("q"));
    int vLimit = cDefaultSearchLimit;
    try {
      vLimit = std::clamp(std::stoi(pReq.mGetParam("limit", std::to_string(cDefaultSearchLimit))), 1, cMaxSearchLimit);
    } catch (const std::exception &) {
      return fChatsError("limit must be a number", cErrorTypeInvalidRequest);
    }
    json aFound = json::array();
    if (lWords.empty()) {
      return fChatsJson({{"messages", aFound}});
    }
    std::lock_guard<std::mutex> vLock(vChatsMutex);
    fLoadChatsIndex();
    for (const auto &[vId, vEntry] : dChats) {
      json vConversation;
      json aMessages;
      if (!fReadChatFile(vEntry.vPath, vConversation, &aMessages)) {
        continue;
      }
      // the newest matching message of the chat
      const json *pBest = nullptr;
      for (const json &cMessage : aMessages) {
        const std::string cRole = cMessage.value("role", std::string());
        if ((cRole != "user" && cRole != "assistant") || !cMessage.contains("content") ||
            !cMessage.at("content").is_string()) {
          continue;
        }
        const std::string cContent = fChatsFoldText(cMessage.at("content").get<std::string>());
        if (!std::all_of(lWords.begin(), lWords.end(),
                         [&](const std::string &pWord) { return cContent.find(pWord) != std::string::npos; })) {
          continue;
        }
        if (pBest == nullptr || cMessage.value("timestamp", (int64_t)0) > pBest->value("timestamp", (int64_t)0)) {
          pBest = &cMessage;
        }
      }
      if (pBest != nullptr) {
        json vHit = *pBest;
        vHit.erase("extra"); // attachments are not needed to show where the words are
        aFound.push_back(vHit);
      }
    }
    std::sort(aFound.begin(), aFound.end(), [](const json &pA, const json &pB) {
      return pA.value("timestamp", (int64_t)0) > pB.value("timestamp", (int64_t)0);
    });
    if ((int)aFound.size() > vLimit) {
      aFound.erase(aFound.begin() + vLimit, aFound.end());
    }
    return fChatsJson({{"messages", aFound}});
  };
}

server_http_context::handler_t fChatsMakeGetHandler() {
  return [](const server_http_req &pReq) {
    const std::string cId = pReq.mGetParam("id_chat");
    std::lock_guard<std::mutex> vLock(vChatsMutex);
    fLoadChatsIndex();
    const auto cIt = dChats.find(cId);
    if (cIt == dChats.end()) {
      return fChatsError("no chat " + cId, cErrorTypeNotFound);
    }
    json vConversation;
    json aMessages;
    if (!fReadChatFile(cIt->second.vPath, vConversation, &aMessages)) {
      return fChatsError("cannot read " + cIt->second.vPath.string(), cErrorTypeServer);
    }
    return fChatsJson({{"conversation", vConversation}, {"messages", aMessages}});
  };
}

server_http_context::handler_t fChatsMakePostHandler() {
  return [](const server_http_req &pReq) {
    const std::string cId = pReq.mGetParam("id_chat");
    if (!fValidChatId(cId)) {
      return fChatsError("invalid chat id", cErrorTypeInvalidRequest);
    }
    const json cBody = json::parse(pReq.vBody.empty() ? "{}" : pReq.vBody, nullptr, false);
    if (!cBody.is_object()) {
      return fChatsError("expected a JSON object", cErrorTypeInvalidRequest);
    }
    const bool cReplace = cBody.value("replace", false);
    if (cBody.contains("conversation") &&
        (!cBody.at("conversation").is_object() || cBody.at("conversation").value("id", std::string()) != cId)) {
      return fChatsError("conversation must be an object whose id is the one of the path", cErrorTypeInvalidRequest);
    }
    for (const char *vField : {"messages", "deleted"}) {
      if (cBody.contains(vField) && !cBody.at(vField).is_array()) {
        return fChatsError(std::string(vField) + " must be an array", cErrorTypeInvalidRequest);
      }
    }

    std::lock_guard<std::mutex> vLock(vChatsMutex);
    fLoadChatsIndex();
    std::string vError;
    if (!fEnsureChatsFolder(vError)) {
      return fChatsError(vError, cErrorTypeServer);
    }

    const auto cIt = dChats.find(cId);
    json vConversation;
    json aMessages = json::array();
    if (cIt != dChats.end()) {
      if (!fReadChatFile(cIt->second.vPath, vConversation, &aMessages)) {
        return fChatsError("cannot read " + cIt->second.vPath.string(), cErrorTypeServer);
      }
    } else if (!cBody.contains("conversation")) {
      return fChatsError("no chat " + cId, cErrorTypeNotFound);
    }

    if (cBody.contains("conversation")) {
      const json cOld = vConversation;
      vConversation = cBody.at("conversation");
      // what the server gave the chat stays unless the client sends a valid value
      if (!vConversation.contains("createdAt") || !vConversation.at("createdAt").is_number()) {
        vConversation["createdAt"] = cOld.is_object() && cOld.contains("createdAt") ? cOld.at("createdAt")
                                                                                     : json(fNowMs());
      }
    }
    int vNumber = vConversation.contains("number") && vConversation.at("number").is_number()
                      ? vConversation.at("number").get<int>()
                      : 0;
    if (vNumber < 1 || vNumber > cMaxChatNumber || fChatNumberTaken(vNumber, cId)) {
      vNumber = fFreeChatNumber(cId);
    }
    vConversation["number"] = vNumber;

    if (cReplace) {
      aMessages = json::array();
    }
    std::map<std::string, size_t> dPosition;
    for (size_t vI = 0; vI < aMessages.size(); vI++) {
      dPosition[aMessages[vI].value("id", std::string())] = vI;
    }
    for (const json &cMessage : cBody.value("messages", json::array())) {
      if (!cMessage.is_object() || !fValidChatId(cMessage.value("id", std::string()))) {
        return fChatsError("every message needs an id", cErrorTypeInvalidRequest);
      }
      json vMessage = cMessage;
      vMessage["convId"] = cId;
      const std::string cMessageId = vMessage.at("id").get<std::string>();
      const auto cPos = dPosition.find(cMessageId);
      if (cPos != dPosition.end()) {
        aMessages[cPos->second] = vMessage;
      } else {
        dPosition[cMessageId] = aMessages.size();
        aMessages.push_back(vMessage);
      }
    }
    if (cBody.contains("deleted")) {
      std::vector<std::string> lDeleted;
      for (const json &cDeleted : cBody.at("deleted")) {
        if (cDeleted.is_string()) {
          lDeleted.push_back(cDeleted.get<std::string>());
        }
      }
      json aKept = json::array();
      for (const json &cMessage : aMessages) {
        if (std::find(lDeleted.begin(), lDeleted.end(), cMessage.value("id", std::string())) == lDeleted.end()) {
          aKept.push_back(cMessage);
        }
      }
      aMessages = aKept;
    }

    const fs::path cPath =
        cIt != dChats.end() ? cIt->second.vPath : fNewChatPath(vConversation.at("createdAt").get<int64_t>());
    if (!fWriteChatFile(cPath, vConversation, aMessages, vError)) {
      return fChatsError(vError, cErrorTypeServer);
    }
    dChats[cId] = server_chat_entry{cPath, vConversation};
    const uint64_t cVersion = ++vChatsVersion;
    return fChatsJson({{"version", cVersion}, {"conversation", vConversation}});
  };
}

server_http_context::handler_t fChatsMakeDeleteHandler() {
  return [](const server_http_req &pReq) {
    const std::string cId = pReq.mGetParam("id_chat");
    std::lock_guard<std::mutex> vLock(vChatsMutex);
    fLoadChatsIndex();
    const auto cIt = dChats.find(cId);
    if (cIt == dChats.end()) {
      return fChatsJson({{"version", vChatsVersion.load()}});
    }
    std::error_code vCode;
    fs::remove(cIt->second.vPath, vCode);
    if (vCode) {
      return fChatsError("cannot delete " + cIt->second.vPath.string() + ": " + vCode.message(), cErrorTypeServer);
    }
    dChats.erase(cIt);
    const uint64_t cVersion = ++vChatsVersion;
    return fChatsJson({{"version", cVersion}});
  };
}
