#include "spec-tuning.h"

#include "build-info.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static const char *cTuningFolder = ".mgs/tuning"; // under the home of the user that runs mgs
static const char *cTuningFile = "mtp.json";

// One writer at a time in this process; a reader sees either the old or the new file, because the new one replaces
// it by rename.
static std::mutex vTuningMutex;

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

std::string fSpecTuningPath() {
  const std::string cHome = fHomeDirectory();
  return cHome.empty() ? std::string() : (fs::path(cHome) / cTuningFolder / cTuningFile).string();
}

std::string fSpecTuningKey(const common_params &pParams) {
  std::error_code vCode;
  const fs::path cModel = fs::absolute(pParams.vModel.vPath, vCode);
  std::string vKey = std::string("mgs ") + mgs_version() + " | " + (vCode ? pParams.vModel.vPath : cModel.string());
  const uintmax_t cSize = fs::file_size(pParams.vModel.vPath, vCode);
  vKey += " (" + std::to_string(vCode ? 0 : cSize) + " bytes)";

  // The GPUs the model may use (--device, or every one) with their memory, and the system memory (the cgroup limit
  // when there is one), which decide how many experts stay on the GPUs and in RAM.
  std::vector<ggml_backend_dev_t> lDevices = pParams.lDevices;
  if (lDevices.empty()) {
    for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
      lDevices.push_back(ggml_backend_dev_get(vI));
    }
  }
  for (ggml_backend_dev_t vDevice : lDevices) {
    if (vDevice == nullptr) {
      continue;
    }
    const enum ggml_backend_dev_type cType = ggml_backend_dev_type(vDevice);
    if (cType != GGML_BACKEND_DEVICE_TYPE_GPU && cType != GGML_BACKEND_DEVICE_TYPE_IGPU) {
      continue;
    }
    size_t vFree = 0;
    size_t vTotal = 0;
    ggml_backend_dev_memory(vDevice, &vFree, &vTotal);
    vKey += std::string(" | ") + ggml_backend_dev_name(vDevice) + " " + ggml_backend_dev_description(vDevice) + " " +
            std::to_string(vTotal / (1024 * 1024)) + " MiB";
  }
  const int64_t cHostTotal = mgs_host_memory_total();
  vKey += " | RAM " + std::to_string(cHostTotal > 0 ? cHostTotal / (1024 * 1024) : 0) + " MiB";
  return vKey;
}

// Saved document, or an empty object when there is none or it cannot be read.
static json fReadTuning(const std::string &pPath) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return json::object();
  }
  std::stringstream vText;
  vText << vFile.rdbuf();
  try {
    json vDocument = json::parse(vText.str());
    if (vDocument.is_object()) {
      return vDocument;
    }
    LOG_WRN("%s does not hold a JSON object; the number of MTP draft tokens is measured again\n", pPath.c_str());
  } catch (const std::exception &vE) {
    LOG_WRN("cannot read %s: %s; the number of MTP draft tokens is measured again\n", pPath.c_str(), vE.what());
  }
  return json::object();
}

int32_t fSpecTuningLoad(const std::string &pKey, std::string &pSummary) {
  const std::string cPath = fSpecTuningPath();
  if (cPath.empty()) {
    return -1;
  }
  std::lock_guard<std::mutex> vLock(vTuningMutex);
  const json cDocument = fReadTuning(cPath);
  if (!cDocument.contains("results") || !cDocument["results"].is_object() || !cDocument["results"].contains(pKey)) {
    return -1;
  }
  const json &cEntry = cDocument["results"][pKey];
  if (!cEntry.is_object() || !cEntry.contains("draft") || !cEntry["draft"].is_number_integer()) {
    return -1;
  }
  const int32_t cDraft = cEntry["draft"].get<int32_t>();
  if (cDraft < 0 || cDraft > cSpecTuningMaxDraft) {
    return -1;
  }
  pSummary.clear();
  if (cEntry.contains("tokens_per_second") && cEntry["tokens_per_second"].is_object()) {
    for (const auto &[vDraft, vRate] : cEntry["tokens_per_second"].items()) {
      if (vRate.is_number()) {
        char aRate[64];
        snprintf(aRate, sizeof(aRate), "%s%s: %.2f", pSummary.empty() ? "" : ", ", vDraft.c_str(),
                 vRate.get<double>());
        pSummary += aRate;
      }
    }
    pSummary = "tokens/s by number of draft tokens " + pSummary;
  }
  if (cEntry.contains("measured") && cEntry["measured"].is_string()) {
    pSummary += (pSummary.empty() ? "" : ", ") + std::string("measured ") + cEntry["measured"].get<std::string>();
  }
  return cDraft;
}

// Writes the document to a temporary file that then replaces the old one, readable only by this user.
static bool fWriteTuning(const std::string &pPath, const json &pDocument, std::string &pError) {
  const fs::path cPath(pPath);
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

bool fSpecTuningSave(const std::string &pKey, int32_t pDraft, const std::vector<double> &pRates, std::string &pError) {
  const std::string cPath = fSpecTuningPath();
  if (cPath.empty()) {
    pError = "the home folder of the user that runs mgs is unknown";
    return false;
  }
  std::lock_guard<std::mutex> vLock(vTuningMutex);
  json vDocument = fReadTuning(cPath);
  if (!vDocument.contains("results") || !vDocument["results"].is_object()) {
    vDocument["results"] = json::object();
  }
  json vRates = json::object();
  for (size_t vDraft = 0; vDraft < pRates.size(); ++vDraft) {
    if (pRates[vDraft] > 0.0) {
      vRates[std::to_string(vDraft)] = std::round(pRates[vDraft] * 1000.0) / 1000.0;
    }
  }
  char aMeasured[32] = "";
  const std::time_t cNow = std::time(nullptr);
  std::tm vLocal{};
#if defined(_WIN32)
  localtime_s(&vLocal, &cNow);
#else
  localtime_r(&cNow, &vLocal);
#endif
  std::strftime(aMeasured, sizeof(aMeasured), "%Y-%m-%d %H:%M", &vLocal);
  vDocument["results"][pKey] = {{"draft", pDraft}, {"tokens_per_second", vRates}, {"measured", aMeasured}};
  return fWriteTuning(cPath, vDocument, pError);
}

common_spec_tuner::common_spec_tuner(int32_t pMaxDraft) {
  const int32_t cMax = std::max<int32_t>(pMaxDraft, 1);
  // The former default (2 draft tokens) first, then none, then the others.
  std::vector<int32_t> lRound = {std::min<int32_t>(2, cMax), 0};
  for (int32_t vDraft = 1; vDraft <= cMax; ++vDraft) {
    if (vDraft != lRound[0]) {
      lRound.push_back(vDraft);
    }
  }
  // One round in this order and the next one reversed, repeated until every candidate has generated cMinTokens.
  lOrder = lRound;
  lOrder.insert(lOrder.end(), lRound.rbegin(), lRound.rend());
  lTokens.assign(cMax + 1, 0);
  lTimeUs.assign(cMax + 1, 0);
}

int32_t common_spec_tuner::mStep(int32_t pSlotId, int32_t pTaskId, int64_t pNDecoded, int64_t pTimeUs, bool pClean,
                                 bool pReplay) {
  if (vBest >= 0) {
    return vBest;
  }
  if (pClean && pSlotId == vLastSlot && pTaskId == vLastTask && pNDecoded >= vLastDecoded) {
    const int64_t cTokens = pNDecoded - vLastDecoded;
    if (vWarmupTokens < cWarmupTokens) {
      vWarmupTokens += cTokens;
    } else {
      const int32_t cDraft = lOrder[vOrderIndex];
      lTokens[cDraft] += cTokens;
      lTimeUs[cDraft] += pTimeUs - vLastTimeUs;
      if (!pReplay) {
        vOrderIndex = (vOrderIndex + 1) % lOrder.size();
      }
      if (vOrderIndex == 0 && *std::min_element(lTokens.begin(), lTokens.end()) >= cMinTokens) {
        // The fastest number of draft tokens, kept only when it beats no draft by cKeepMargin.
        const std::vector<double> cRates = mRates();
        int32_t vFastest = 1;
        for (int32_t vDraft = 2; vDraft < (int32_t)cRates.size(); ++vDraft) {
          if (cRates[vDraft] > cRates[vFastest]) {
            vFastest = vDraft;
          }
        }
        vBest = cRates[vFastest] > cRates[0] * (1.0 + cKeepMargin) ? vFastest : 0;
        return vBest;
      }
    }
  }
  vLastSlot = pClean ? pSlotId : -1;
  vLastTask = pClean ? pTaskId : -1;
  vLastDecoded = pNDecoded;
  vLastTimeUs = pTimeUs;
  return lOrder[vOrderIndex];
}

std::vector<double> common_spec_tuner::mRates() const {
  std::vector<double> lRates(lTokens.size(), 0.0);
  for (size_t vDraft = 0; vDraft < lTokens.size(); ++vDraft) {
    if (lTimeUs[vDraft] > 0) {
      lRates[vDraft] = 1e6 * (double)lTokens[vDraft] / (double)lTimeUs[vDraft];
    }
  }
  return lRates;
}
