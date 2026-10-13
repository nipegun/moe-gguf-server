#include "server-ui-settings.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#if !defined(_WIN32)
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static const char *cSettingsFolder = ".mgs/config"; // under the home of the user that runs mgs
static const char *cSettingsFile = "ui-settings.json";
static constexpr size_t cMaxSettingsBytes = 16 * 1024 * 1024;

// One writer at a time; a reader sees either the old or the new file, because the new one replaces it by rename.
static std::mutex vSettingsMutex;

static std::string fHomeDirectory() {
  const char *vHome = std::getenv("HOME");
  if (vHome != nullptr && vHome[0] != '\0') {
    return vHome;
  }
#if defined(_WIN32)
  const char *vProfile = std::getenv("USERPROFILE");
  return vProfile != nullptr ? vProfile : "";
#else
  const passwd *vAccount = getpwuid(geteuid());
  return vAccount != nullptr && vAccount->pw_dir != nullptr ? vAccount->pw_dir : "";
#endif
}

static fs::path fSettingsPath() {
  const std::string cHome = fHomeDirectory();
  return cHome.empty() ? fs::path() : fs::path(cHome) / cSettingsFolder / cSettingsFile;
}

// Saved document, or null when there is none or it is not a JSON object.
static json fReadSettings() {
  const fs::path cPath = fSettingsPath();
  if (cPath.empty()) {
    return nullptr;
  }
  std::ifstream vFile(cPath, std::ios::binary);
  if (!vFile) {
    return nullptr;
  }
  std::stringstream vText;
  vText << vFile.rdbuf();
  try {
    json vDocument = json::parse(vText.str());
    if (vDocument.is_object()) {
      return vDocument;
    }
    SRV_WRN("%s does not hold a JSON object; the Web UI starts with its defaults\n", cPath.string().c_str());
  } catch (const std::exception &vE) {
    SRV_WRN("cannot read %s: %s; the Web UI starts with its defaults\n", cPath.string().c_str(), vE.what());
  }
  return nullptr;
}

// Writes the document to a temporary file that then replaces the old one, readable only by this user, because the
// settings may hold the headers of MCP servers.
static bool fWriteSettings(const json &pDocument, std::string &pError) {
  const fs::path cPath = fSettingsPath();
  if (cPath.empty()) {
    pError = "the home folder of the user that runs mgs is unknown";
    return false;
  }
  const std::string cText = pDocument.dump(2) + "\n";
#if defined(_WIN32)
  std::error_code vCode;
  fs::create_directories(cPath.parent_path(), vCode);
  if (vCode) {
    pError = "cannot create " + cPath.parent_path().string() + ": " + vCode.message();
    return false;
  }
  const fs::path cTemporary = cPath.string() + ".tmp";
  {
    std::ofstream vFile(cTemporary, std::ios::binary | std::ios::trunc);
    vFile << cText;
    if (!vFile) {
      pError = "cannot write " + cTemporary.string();
      return false;
    }
  }
  fs::rename(cTemporary, cPath, vCode);
  if (vCode) {
    pError = "cannot replace " + cPath.string() + ": " + vCode.message();
    return false;
  }
  return true;
#else
  for (const fs::path &vFolder : {cPath.parent_path().parent_path(), cPath.parent_path()}) {
    if (mkdir(vFolder.c_str(), 0700) != 0 && errno != EEXIST) {
      pError = "cannot create " + vFolder.string() + ": " + std::strerror(errno);
      return false;
    }
  }
  const fs::path cTemporary = cPath.string() + ".tmp-" + std::to_string(getpid());
  const int cFd = open(cTemporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (cFd < 0) {
    pError = "cannot create " + cTemporary.string() + ": " + std::strerror(errno);
    return false;
  }
  size_t vWritten = 0;
  while (vWritten < cText.size()) {
    const ssize_t cChunk = write(cFd, cText.data() + vWritten, cText.size() - vWritten);
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
  if (rename(cTemporary.c_str(), cPath.c_str()) != 0) {
    pError = "cannot replace " + cPath.string() + ": " + std::strerror(errno);
    unlink(cTemporary.c_str());
    return false;
  }
  return true;
#endif
}

std::string fUiSettingsLanguage() {
  json vDocument;
  {
    std::lock_guard<std::mutex> vLock(vSettingsMutex);
    vDocument = fReadSettings();
  }
  if (!vDocument.is_object() || !vDocument.contains("language") || !vDocument.at("language").is_string()) {
    return "";
  }
  // only a language tag such as "es-ES" is published to the page
  const std::string cLanguage = vDocument.at("language").get<std::string>();
  const bool cValid = !cLanguage.empty() && cLanguage.size() <= 16 &&
                      cLanguage.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-") ==
                          std::string::npos;
  return cValid ? cLanguage : "";
}

static server_http_res_ptr fUiSettingsError(const std::string &pMessage, const enum error_type pType) {
  auto vRes = std::make_unique<server_http_res>();
  const json cError = fFormatErrorResponse(pMessage, pType);
  vRes->vStatus = cError.at("code").get<int>();
  vRes->vData = fSafeJsonToStr({{"error", cError}});
  return vRes;
}

server_http_context::handler_t fUiSettingsMakeGetHandler() {
  return [](const server_http_req &) {
    json vDocument;
    {
      std::lock_guard<std::mutex> vLock(vSettingsMutex);
      vDocument = fReadSettings();
    }
    auto vRes = std::make_unique<server_http_res>();
    vRes->dHeaders["Cache-Control"] = "no-store";
    vRes->vData = fSafeJsonToStr({{"settings", vDocument}});
    return vRes;
  };
}

server_http_context::handler_t fUiSettingsMakePostHandler() {
  return [](const server_http_req &pRequest) {
    if (pRequest.vBody.size() > cMaxSettingsBytes) {
      return fUiSettingsError("the settings document is too large", cErrorTypeInvalidRequest);
    }
    json vBody;
    try {
      vBody = json::parse(pRequest.vBody);
    } catch (const std::exception &vE) {
      return fUiSettingsError(std::string("invalid JSON: ") + vE.what(), cErrorTypeInvalidRequest);
    }
    if (!vBody.is_object() || !vBody.contains("settings") || !vBody.at("settings").is_object()) {
      return fUiSettingsError("the body must be {\"settings\": {...}}", cErrorTypeInvalidRequest);
    }
    std::string vError;
    bool vSaved = false;
    {
      std::lock_guard<std::mutex> vLock(vSettingsMutex);
      vSaved = fWriteSettings(vBody.at("settings"), vError);
    }
    if (!vSaved) {
      SRV_ERR("cannot save the Web UI settings: %s\n", vError.c_str());
      return fUiSettingsError("cannot save the settings: " + vError, cErrorTypeServer);
    }
    auto vRes = std::make_unique<server_http_res>();
    vRes->vData = fSafeJsonToStr({{"success", true}});
    return vRes;
  };
}
