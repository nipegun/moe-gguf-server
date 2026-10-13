#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-monitor.h"
#include "server-stream.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "http.h"
#include "subproc.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// address for child process, this is needed because router may run on 0.0.0.0
#define CHILD_ADDR "127.0.0.1"

struct server_subproc {
  common_subproc vSproc;             // not yet spawned while in DOWNLOADING state
  std::atomic<bool> vStopped{false}; // set to cancel a download or signal child process exit

  bool mIsAlive() { return vSproc.mAlive(); }

  void mRequestExit() {
    FILE *vStdinFile = vSproc.mStdinFile();
    if (vStdinFile) {
      fprintf(vStdinFile, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
      fflush(vStdinFile);
    }
    vStopped.store(true, std::memory_order_relaxed);
  }

  void mTerminate() { vSproc.mTerminate(); }
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.vTimeoutRead/vTimeoutWrite which only applies to the generation proxy
static constexpr int cStreamLookupTimeoutMs = 250;

// children answer /monitor/memory from a published snapshot, even while they decode
static constexpr int cMonitorChildTimeoutMs = 1000;

static std::filesystem::path fGetServerExecPath() {
  // The relocatable local bundle runs the server through its own dynamic loader, so the process image is
  // the loader; its launcher publishes the path that child servers must execute instead.
  const char *vLauncherPath = std::getenv("MGS_EXECUTABLE");
  if (vLauncherPath != nullptr && vLauncherPath[0] != '\0') {
    return std::filesystem::path(vLauncherPath);
  }
#if defined(_WIN32)
  wchar_t aBuf[32768] = {0}; // Large buffer to handle long paths
  DWORD vLen = GetModuleFileNameW(nullptr, aBuf, _countof(aBuf));
  if (vLen == 0 || vLen >= _countof(aBuf)) {
    throw std::runtime_error("GetModuleFileNameW failed or path too long");
  }
  return std::filesystem::path(aBuf);
#elif defined(__APPLE__) && defined(__MACH__)
  char aSmallPath[PATH_MAX];
  uint32_t vSize = sizeof(aSmallPath);

  if (_NSGetExecutablePath(aSmallPath, &vSize) == 0) {
    // resolve any symlinks to get absolute path
    try {
      return std::filesystem::canonical(std::filesystem::path(aSmallPath));
    } catch (...) {
      return std::filesystem::path(aSmallPath);
    }
  } else {
    // buffer was too small, allocate required size and call again
    std::vector<char> lBuf(vSize);
    if (_NSGetExecutablePath(lBuf.data(), &vSize) == 0) {
      try {
        return std::filesystem::canonical(std::filesystem::path(lBuf.data()));
      } catch (...) {
        return std::filesystem::path(lBuf.data());
      }
    }
    throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
  }
#else
  char aPath[FILENAME_MAX];
  ssize_t vCount = readlink("/proc/self/exe", aPath, FILENAME_MAX);
  if (vCount <= 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  return std::filesystem::path(std::string(aPath, vCount));
#endif
}

static void fUnsetReservedArgs(common_preset &pPreset, bool pUnsetModelArgs) {
  pPreset.mUnsetOption("MGS_ARG_SSL_KEY_FILE");
  pPreset.mUnsetOption("MGS_ARG_SSL_CERT_FILE");
  pPreset.mUnsetOption("MGS_ARG_HTTPS");
  pPreset.mUnsetOption("MGS_ARG_HTTP_PORT");
  pPreset.mUnsetOption("MGS_API_KEY");
  pPreset.mUnsetOption("MGS_ARG_API_KEY_FILE");
  pPreset.mUnsetOption("MGS_ARG_MODELS_DIR");
  pPreset.mUnsetOption("MGS_ARG_MODELS_MAX");
  pPreset.mUnsetOption("MGS_ARG_MODELS_PRESET");
  pPreset.mUnsetOption("MGS_ARG_MODELS_AUTOLOAD");
  // the router serves the shell MCP server; private model processes never run commands
  pPreset.mUnsetOption("MGS_ARG_LOCAL_SHELL_MCP_SERVER");
  if (pUnsetModelArgs) {
    pPreset.mUnsetOption("MGS_ARG_MODEL");
    pPreset.mUnsetOption("MGS_ARG_MMPROJ");
    pPreset.mUnsetOption("MGS_ARG_ALIAS");
    pPreset.mUnsetOption("MGS_ARG_HF_REPO");
  }
}

// Options whose value is a credential, or the file or JSON that holds one: child processes still receive them, but
// model lists and logs only show them redacted
static const std::set<std::string> sSecretOptionEnvs = {
    "HF_TOKEN",
    "MGS_API_KEY",
    "MGS_ARG_API_KEY_FILE",
    "MGS_ARG_SSL_KEY_FILE",
    "MGS_ARG_MCP_SERVERS_JSON",
    "MGS_ARG_MCP_SERVERS_CONFIG",
};
static const char *cRedactedValue = "<redacted>";

static bool fIsSecretOption(const common_arg &pOpt) {
  return pOpt.vEnv != nullptr && sSecretOptionEnvs.count(pOpt.vEnv) > 0;
}

// copy of a preset with the values of its secret options redacted
static common_preset fRedactPreset(const common_preset &pPreset) {
  common_preset vCopy = pPreset;
  for (auto &[vOpt, vValue] : vCopy.dOptions) {
    if (fIsSecretOption(vOpt)) {
      vValue = cRedactedValue;
    }
  }
  return vCopy;
}

// copy of a rendered command line with the value of every secret option redacted, whichever spelling of the option
// it uses ("--hf-token VALUE" or "--hf-token=VALUE")
static std::vector<std::string> fRedactArgs(const std::vector<std::string> &pArgs,
                                            const common_preset_context &pCtxPreset) {
  std::set<std::string> sSecretFlags;
  for (const std::string &vEnv : sSecretOptionEnvs) {
    const auto cIt = pCtxPreset.dKeyToOpt.find(vEnv);
    if (cIt == pCtxPreset.dKeyToOpt.end()) {
      continue;
    }
    for (const char *vFlag : cIt->second.lArgs) {
      sSecretFlags.insert(vFlag);
    }
  }
  std::vector<std::string> lRedacted = pArgs;
  for (size_t vI = 0; vI < lRedacted.size(); ++vI) {
    const std::string cArg = lRedacted[vI];
    const size_t cEquals = cArg.find('=');
    if (cEquals != std::string::npos && sSecretFlags.count(cArg.substr(0, cEquals)) > 0) {
      lRedacted[vI] = cArg.substr(0, cEquals + 1) + cRedactedValue;
    } else if (sSecretFlags.count(cArg) > 0 && vI + 1 < lRedacted.size()) {
      lRedacted[++vI] = cRedactedValue;
    }
  }
  return lRedacted;
}

#ifdef _WIN32
static std::string fWideToUtf8(const wchar_t *pWs) {
  if (!pWs || !*pWs) {
    return {};
  }

  const int cLen = static_cast<int>(std::wcslen(pWs));
  const int cBytes = WideCharToMultiByte(CP_UTF8, 0, pWs, cLen, nullptr, 0, nullptr, nullptr);
  if (cBytes == 0) {
    return {};
  }

  std::string vUtf8(cBytes, '\0');
  WideCharToMultiByte(CP_UTF8, 0, pWs, cLen, vUtf8.data(), cBytes, nullptr, nullptr);

  return vUtf8;
}
#endif

static std::vector<std::string> fGetEnvironment() {
  std::vector<std::string> lEnv;

#ifdef _WIN32
  LPWCH vEnvBlock = GetEnvironmentStringsW();
  if (!vEnvBlock) {
    return lEnv;
  }
  for (LPWCH vE = vEnvBlock; *vE; vE += wcslen(vE) + 1) {
    lEnv.emplace_back(fWideToUtf8(vE));
  }
  FreeEnvironmentStringsW(vEnvBlock);
#else
  if (environ == nullptr) {
    return lEnv;
  }
  for (char **vE = environ; *vE != nullptr; vE++) {
    lEnv.emplace_back(*vE);
  }
#endif

  return lEnv;
}

void server_model_meta::mUpdateArgs(common_preset_context &pCtxPreset, std::string pBinPath) {
  // update params
  fUnsetReservedArgs(vPreset, false);
  vPreset.mSetOption(pCtxPreset, "MGS_ARG_HOST", CHILD_ADDR);
  vPreset.mSetOption(pCtxPreset, "MGS_ARG_PORT", std::to_string(vPort));
  vPreset.mSetOption(pCtxPreset, "MGS_ARG_HTTPS", "false");
  vPreset.mSetOption(pCtxPreset, "MGS_ARG_ALIAS", vName);
  vPreset.mSetOption(pCtxPreset, "MGS_ARG_API_PREFIX", "/api");
  // TODO: maybe validate preset before rendering ?
  // render args
  lArgs = vPreset.mToArgs(pBinPath);

  // unified binary dispatches by subcommand, re-inject it right after the
  // binary path so the child starts as 'mgs serve ...' not 'mgs ...'
  const char *vAppCmd = std::getenv("MGS_APP_CMD");
  if (vAppCmd != nullptr && vAppCmd[0] != '\0' && !pBinPath.empty()) {
    lArgs.insert(lArgs.begin() + 1, vAppCmd);
  }
}

void server_model_meta::mUpdateCaps() {
  try {
    common_params vParams;
    vPreset.mApplyToParams(vParams, {
                                        "MGS_ARG_MODEL",
                                        "MGS_ARG_MODEL_URL",
                                        "MGS_ARG_MMPROJ",
                                        "MGS_ARG_MMPROJ_URL",
                                        "MGS_ARG_MMPROJ_AUTO",
                                        "MGS_ARG_HF_REPO",
                                        "MGS_ARG_HF_REPO_FILE",
                                    });
    vParams.vOffline = true;
    common_models_handler vHandler = fCommonModelsHandlerInit(vParams, cMgsExampleServer);
    fCommonModelsHandlerApply(vHandler, vParams); // note: this won't download the model because offline=true
    if (vParams.vNoMmproj || vParams.vMmproj.vPath.empty()) {
      vMultimodal = {false, false};
    } else {
      vMultimodal = mtmd_get_cap_from_file(vParams.vMmproj.vPath.c_str());
    }
  } catch (const std::exception &vE) {
    LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", vE.what());
    vMultimodal = {false, false};
  }
}

//
// server_models
//

server_models::server_models(const common_params &pParams, int pArgc, char **pArgv)
    : vCtxPreset(cMgsExampleServer), vBaseParams(pParams), lBaseEnv(fGetEnvironment()),
      vBasePreset(vCtxPreset.mLoadFromArgs(pArgc, pArgv)) {
  // clean up base preset
  fUnsetReservedArgs(vBasePreset, true);
  // Public listener settings must not leak into private model processes.
  lBaseEnv.erase(std::remove_if(lBaseEnv.begin(), lBaseEnv.end(),
                                [](const std::string &pEntry) {
                                  const std::string vName = pEntry.substr(0, pEntry.find('='));
                                  return vName == "MGS_ARG_SSL_KEY_FILE" || vName == "MGS_ARG_SSL_CERT_FILE" ||
                                         vName == "MGS_ARG_HTTPS" || vName == "MGS_ARG_HTTP_PORT" ||
                                         vName == "MGS_ARG_LOCAL_SHELL_MCP_SERVER" || vName == "AIP_MODE" ||
                                         vName == "AIP_HTTP_PORT";
                                }),
                 lBaseEnv.end());
  // set binary path
  try {
    vBinPath = fGetServerExecPath().string();
  } catch (const std::exception &vE) {
    vBinPath = pArgv[0];
    LOG_WRN("failed to get server executable path: %s\n", vE.what());
    LOG_WRN("using original argv[0] as fallback: %s\n", pArgv[0]);
  }
  mLoadModels();
}

void server_models::mAddModel(server_model_meta &&pMeta) {
  if (dMapping.find(pMeta.vName) != dMapping.end()) {
    throw std::runtime_error(fStringFormat("model '%s' appears multiple times", pMeta.vName.c_str()));
  }

  // check model name does not conflict with existing aliases
  for (const auto &[vKey, vInst] : dMapping) {
    if (vInst.vMeta.sAliases.count(pMeta.vName)) {
      throw std::runtime_error(
          fStringFormat("model name '%s' conflicts with alias of model '%s'", pMeta.vName.c_str(), vKey.c_str()));
    }
  }

  // parse aliases from preset's --alias option (comma-separated)
  std::string vAliasStr;
  if (pMeta.vPreset.mGetOption("MGS_ARG_ALIAS", vAliasStr) && !vAliasStr.empty()) {
    for (auto &vAlias : fStringSplit<std::string>(vAliasStr, ',')) {
      vAlias = fStringStrip(vAlias);
      if (!vAlias.empty()) {
        pMeta.sAliases.insert(vAlias);
      }
    }
  }

  // parse tags from preset's --tags option (comma-separated)
  std::string vTagsStr;
  if (pMeta.vPreset.mGetOption("MGS_ARG_TAGS", vTagsStr) && !vTagsStr.empty()) {
    for (auto &vTag : fStringSplit<std::string>(vTagsStr, ',')) {
      vTag = fStringStrip(vTag);
      if (!vTag.empty()) {
        pMeta.sTags.insert(vTag);
      }
    }
  }

  // validate aliases do not conflict with existing names or aliases
  for (const auto &vAlias : pMeta.sAliases) {
    if (dMapping.find(vAlias) != dMapping.end()) {
      throw std::runtime_error(fStringFormat("alias '%s' for model '%s' conflicts with existing model name",
                                             vAlias.c_str(), pMeta.vName.c_str()));
    }
    for (const auto &[vKey, vInst] : dMapping) {
      if (vInst.vMeta.sAliases.count(vAlias)) {
        throw std::runtime_error(fStringFormat("alias '%s' for model '%s' conflicts with alias of model '%s'",
                                               vAlias.c_str(), pMeta.vName.c_str(), vKey.c_str()));
      }
    }
  }

  pMeta.mUpdateArgs(vCtxPreset, vBinPath); // render args
  pMeta.mUpdateCaps();
  std::string vName = pMeta.vName;
  dMapping[vName] = instance_t{/* subproc */ std::make_shared<server_subproc>(),
                               /* th      */ std::thread(),
                               /* meta    */ std::move(pMeta)};
}

void server_models::mNotifySse(const std::string &pEvent, const std::string &pModelId, const json &pData) {
  std::unique_ptr<server_task_result_router> vResult = std::make_unique<server_task_result_router>();
  vResult->vData = {
      {"model", pModelId},
      {"event", pEvent},
  };
  if (!pData.is_null()) {
    vResult->vData["data"] = pData;
  }
  SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", pEvent.c_str(), pModelId.c_str(),
          fSafeJsonToStr(vResult->vData).c_str());
  vSse.mBroadcast(std::move(vResult));
}

void server_models::mLoadModels() {
  // Phase 1: load presets from all sources - pure I/O, no lock needed
  // 1. cached models
  common_presets vCachedModels = vCtxPreset.mLoadFromCache();
  SRV_INF("Loaded %zu cached model presets\n", vCachedModels.size());
  // 2. local models from --models-dir
  common_presets vLocalModels;
  if (!vBaseParams.vModelsDir.empty()) {
    vLocalModels = vCtxPreset.mLoadFromModelsDir(vBaseParams.vModelsDir);
    SRV_INF("Loaded %zu local model presets from %s\n", vLocalModels.size(), vBaseParams.vModelsDir.c_str());
  }
  // 3. custom-path models from presets
  common_preset vGlobal = {};
  common_presets vCustomPresets = {};
  if (!vBaseParams.vModelsPreset.empty()) {
    vCustomPresets = vCtxPreset.mLoadFromIni(vBaseParams.vModelsPreset, vGlobal);
    SRV_INF("Loaded %zu custom model presets from %s\n", vCustomPresets.size(), vBaseParams.vModelsPreset.c_str());
  }

  // cascade, apply global preset first
  vCachedModels = vCtxPreset.mCascade(vGlobal, vCachedModels);
  vLocalModels = vCtxPreset.mCascade(vGlobal, vLocalModels);
  vCustomPresets = vCtxPreset.mCascade(vGlobal, vCustomPresets);

  // note: if a model exists in both cached and local, local takes precedence
  common_presets vFinalPresets;
  std::unordered_map<std::string, server_model_source> dSourceMap;
  for (const auto &[vName, vPreset] : vCachedModels) {
    vFinalPresets[vName] = vPreset;
    dSourceMap[vName] = cServerModelSourceCache;
  }
  for (const auto &[vName, vPreset] : vLocalModels) {
    vFinalPresets[vName] = vPreset;
    dSourceMap[vName] = cServerModelSourceModelsDir;
  }
  for (const auto &[vName, vCustom] : vCustomPresets) {
    if (vFinalPresets.find(vName) != vFinalPresets.end()) {
      vFinalPresets[vName].mMerge(vCustom);
    } else {
      vFinalPresets[vName] = vCustom;
    }
    dSourceMap[vName] = cServerModelSourcePreset;
  }

  // overlay router's own CLI args on top of every model preset so that
  // e.g. `mgs --temp 0` is honoured by all child processes
  for (auto &[vName, vPreset] : vFinalPresets) {
    vPreset.mMerge(vBasePreset);
  }

  auto vGetSource = [&](const std::string &pName) {
    return dSourceMap.count(pName) ? dSourceMap.at(pName) : cServerModelSourcePreset;
  };

  // Helpers that read `dMapping` - must be called while holding the lock.
  std::unordered_set<std::string> sCustomNames;
  for (const auto &[vName, vPreset] : vCustomPresets)
    sCustomNames.insert(vName);
  auto vJoinSet = [](const std::set<std::string> &pS) {
    std::string vResult;
    for (const auto &vV : pS) {
      if (!vResult.empty())
        vResult += ", ";
      vResult += vV;
    }
    return vResult;
  };
  auto vLogAvailableModels = [&]() {
    SRV_INF("Available models (%zu) (*: custom preset)\n", dMapping.size());
    for (const auto &[vName, vInst] : dMapping) {
      bool vHasCustom = sCustomNames.find(vName) != sCustomNames.end();
      std::string vInfo;
      if (!vInst.vMeta.sAliases.empty())
        vInfo += " (aliases: " + vJoinSet(vInst.vMeta.sAliases) + ")";
      if (!vInst.vMeta.sTags.empty())
        vInfo += " [tags: " + vJoinSet(vInst.vMeta.sTags) + "]";
      SRV_INF("  %c %s%s\n", vHasCustom ? '*' : ' ', vName.c_str(), vInfo.c_str());
    }
  };
  auto vApplyStopTimeout = [&]() {
    for (auto &[vName, vInst] : dMapping) {
      std::string vVal;
      if (vInst.vMeta.vPreset.mGetOption(COMMON_ARG_PRESET_STOP_TIMEOUT, vVal)) {
        try {
          vInst.vMeta.vStopTimeout = std::stoi(vVal);
        } catch (...) {
          SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n", vVal.c_str(),
                  vName.c_str(), DEFAULT_STOP_TIMEOUT);
          vInst.vMeta.vStopTimeout = DEFAULT_STOP_TIMEOUT;
        }
      }
    }
  };
  // mUpdateArgs() injects HOST/PORT/ALIAS, so strip them before comparing presets
  auto vPresetOptionsForCompare = [](common_preset pP) {
    pP.mUnsetOption("MGS_ARG_HOST");
    pP.mUnsetOption("MGS_ARG_PORT");
    pP.mUnsetOption("MGS_ARG_HTTPS");
    pP.mUnsetOption("MGS_ARG_ALIAS");
    return pP.dOptions;
  };

  // Phase 2: acquire the lock once for all mapping mutations.
  // We temporarily release it only when calling functions that acquire it internally
  // (unload, load) or when joining threads (the monitoring thread calls update_status
  // which locks the mutex, so joining while holding it would deadlock).
  std::unique_lock<std::mutex> vLk(vMutex);

  vNeedReload = false;
  bool vIsFirstLoad = dMapping.empty();

  if (vIsFirstLoad) {
    // FIRST LOAD: add all models, then unlock for autoloading
    for (const auto &[vName, vPreset] : vFinalPresets) {
      server_model_meta vMeta{
          /* source        */ vGetSource(vName),
          /* preset        */ vPreset,
          /* name          */ vName,
          /* aliases       */ {},
          /* tags          */ {},
          /* port          */ 0,
          /* status        */ cServerModelStatusUnloaded,
          /* last_used     */ 0,
          /* args          */ std::vector<std::string>(),
          /* loaded_info   */ {},
          /* progress      */ {},
          /* exit_code     */ 0,
          /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
          /* multimodal    */ mtmd_caps{false, false},
          // /* need_download */ false,
      };
      mAddModel(std::move(vMeta));
    }
    vApplyStopTimeout();
    vLogAvailableModels();

    std::vector<std::string> lModelsToLoad;
    for (const auto &[vName, vInst] : dMapping) {
      std::string vVal;
      if (vInst.vMeta.vPreset.mGetOption(COMMON_ARG_PRESET_LOAD_ON_STARTUP, vVal) && common_arg_utils::fIsTruthy(vVal)) {
        lModelsToLoad.push_back(vName);
      }
    }
    if ((int)lModelsToLoad.size() > vBaseParams.vModelsMax) {
      throw std::runtime_error(fStringFormat("number of models to load on startup (%zu) exceeds models_max (%d)",
                                             lModelsToLoad.size(), vBaseParams.vModelsMax));
    }

    vLk.unlock();
    for (const auto &vName : lModelsToLoad) {
      SRV_INF("(startup) loading model %s\n", vName.c_str());
      mLoad(vName);
    }
  } else {
    // RELOAD: diff the new preset list against the current mapping and reconcile
    vIsReloading = true;

    // find running models whose source was removed or whose preset changed
    std::vector<std::string> lToUnload;
    for (const auto &[vName, vInst] : dMapping) {
      if (!vInst.vMeta.mIsRunning())
        continue;
      auto vIt = vFinalPresets.find(vName);
      if (vIt == vFinalPresets.end()) {
        lToUnload.push_back(vName); // removed from source
      } else if (vPresetOptionsForCompare(vInst.vMeta.vPreset) != vPresetOptionsForCompare(vIt->second)) {
        lToUnload.push_back(vName); // preset changed
      }
    }

    // mUnload() acquires the lock internally, so release before each call
    for (const auto &vName : lToUnload) {
      SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", vName.c_str());
      vLk.unlock();
      mUnload(vName);
      vLk.lock();
    }

    // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
    vCv.wait(vLk, [&]() {
      for (const auto &vName : lToUnload) {
        auto vIt = dMapping.find(vName);
        if (vIt != dMapping.end() && vIt->second.vMeta.mIsRunning())
          return false;
      }
      return true;
    });

    // collect all threads to join in one pass while the lock is held:
    // - monitoring threads from just-unloaded models (to_unload)
    // - threads of finished downloads (DOWNLOADED), they acquire the mutex on exit
    // - threads of already-UNLOADED models that are being removed from source
    std::vector<std::thread> lThreadsToJoin;
    for (const auto &vName : lToUnload) {
      auto vIt = dMapping.find(vName);
      if (vIt != dMapping.end() && vIt->second.vTh.joinable()) {
        lThreadsToJoin.push_back(std::move(vIt->second.vTh));
      }
    }
    for (auto &[vName, vInst] : dMapping) {
      if (vInst.vMeta.vStatus == cServerModelStatusDownloading) {
        continue; // downloading models are not from config sources, leave them alone
      }
      if (vInst.vMeta.vStatus == cServerModelStatusDownloaded) {
        // joining this thread under the lock deadlocks: it locks the mutex on its way out
        if (vInst.vTh.joinable()) {
          lThreadsToJoin.push_back(std::move(vInst.vTh));
        }
        continue;
      }
      if (vFinalPresets.find(vName) == vFinalPresets.end() && !vInst.vMeta.mIsRunning() && vInst.vTh.joinable()) {
        lThreadsToJoin.push_back(std::move(vInst.vTh));
      }
    }

    // join outside the lock - monitoring thread calls update_status (needs lock)
    vLk.unlock();
    for (auto &vTh : lThreadsToJoin)
      vTh.join();
    vLk.lock();

    // erase models no longer in any source
    for (auto vIt = dMapping.begin(); vIt != dMapping.end();) {
      if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
        ++vIt; // download thread is still busy, skip
      } else if (vIt->second.vMeta.vStatus == cServerModelStatusDownloaded) {
        // download finished, thread is joined above, safe to erase
        GGML_ASSERT(!vIt->second.vTh.joinable());
        vIt = dMapping.erase(vIt);
      } else if (vFinalPresets.find(vIt->first) == vFinalPresets.end()) {
        SRV_INF("(reload) removing model name=%s (no longer in source)\n", vIt->first.c_str());
        GGML_ASSERT(!vIt->second.vTh.joinable()); // must have been joined above
        vIt = dMapping.erase(vIt);
      } else {
        ++vIt;
      }
    }

    // update presets for non-running models still in source
    for (auto &[vName, vInst] : dMapping) {
      if (vInst.vMeta.mIsRunning())
        continue;
      auto vIt = vFinalPresets.find(vName);
      if (vIt == vFinalPresets.end())
        continue; // erased above

      vInst.vMeta.vPreset = vIt->second;

      // re-parse aliases, then validate against other models
      std::set<std::string> sNewAliases;
      std::string vAliasStr;
      if (vInst.vMeta.vPreset.mGetOption("MGS_ARG_ALIAS", vAliasStr) && !vAliasStr.empty()) {
        for (auto &vAlias : fStringSplit<std::string>(vAliasStr, ',')) {
          vAlias = fStringStrip(vAlias);
          if (!vAlias.empty())
            sNewAliases.insert(vAlias);
        }
      }
      vInst.vMeta.sAliases.clear();
      for (const auto &vAlias : sNewAliases) {
        bool vConflict = false;
        for (const auto &[vOtherName, vOtherInst] : dMapping) {
          if (vOtherName == vName)
            continue;
          if (vOtherName == vAlias || vOtherInst.vMeta.sAliases.count(vAlias)) {
            SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n", vAlias.c_str(),
                    vName.c_str(), vOtherName.c_str());
            vConflict = true;
            break;
          }
        }
        if (!vConflict)
          vInst.vMeta.sAliases.insert(vAlias);
      }

      // re-parse tags
      vInst.vMeta.sTags.clear();
      std::string vTagsStr;
      if (vInst.vMeta.vPreset.mGetOption("MGS_ARG_TAGS", vTagsStr) && !vTagsStr.empty()) {
        for (auto &vTag : fStringSplit<std::string>(vTagsStr, ',')) {
          vTag = fStringStrip(vTag);
          if (!vTag.empty())
            vInst.vMeta.sTags.insert(vTag);
        }
      }

      vInst.vMeta.vExitCode = 0; // clear failed state so the model can be reloaded
      vInst.vMeta.mUpdateArgs(vCtxPreset, vBinPath);
      vInst.vMeta.mUpdateCaps();
    }

    // add models that are new in this reload
    std::vector<std::string> lNewlyAdded;
    for (const auto &[vName, vPreset] : vFinalPresets) {
      if (dMapping.find(vName) == dMapping.end()) {
        server_model_meta vMeta{
            /* source        */ vGetSource(vName),
            /* preset        */ vPreset,
            /* name          */ vName,
            /* aliases       */ {},
            /* tags          */ {},
            /* port          */ 0,
            /* status        */ cServerModelStatusUnloaded,
            /* last_used     */ 0,
            /* args          */ std::vector<std::string>(),
            /* loaded_info   */ {},
            /* progress      */ {},
            /* exit_code     */ 0,
            /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
            /* multimodal    */ mtmd_caps{false, false},
            // /* need_download */ false,
        };
        mAddModel(std::move(vMeta));
        lNewlyAdded.push_back(vName);
      }
    }

    vApplyStopTimeout();

    // clear reload flag before unlocking for autoload - load() blocks on !is_reloading,
    // so clearing it here (while still locked) prevents a deadlock in the autoload calls below
    vIsReloading = false;
    vCv.notify_all();

    vLogAvailableModels();

    // collect autoload candidates while still under the lock
    std::vector<std::string> lToAutoload;
    for (const auto &vName : lNewlyAdded) {
      auto vIt = dMapping.find(vName);
      if (vIt != dMapping.end()) {
        std::string vVal;
        if (vIt->second.vMeta.vPreset.mGetOption(COMMON_ARG_PRESET_LOAD_ON_STARTUP, vVal) &&
            common_arg_utils::fIsTruthy(vVal)) {
          lToAutoload.push_back(vName);
        }
      }
    }

    vLk.unlock();
    for (const auto &vName : lToAutoload) {
      SRV_INF("(reload) loading new model %s\n", vName.c_str());
      mLoad(vName);
    }

    mNotifySse("models_reload", "*");
  }
}

void server_models::mUpdateMeta(const std::string &pName, const server_model_meta &pMeta) {
  std::lock_guard<std::mutex> vLk(vMutex);
  auto vIt = dMapping.find(pName);
  if (vIt != dMapping.end()) {
    vIt->second.vMeta = pMeta;
  }
  vCv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::mHasModel(const std::string &pName) {
  std::lock_guard<std::mutex> vLk(vMutex);
  if (dMapping.find(pName) != dMapping.end()) {
    return true;
  }
  for (const auto &[vKey, vInst] : dMapping) {
    if (vInst.vMeta.sAliases.count(pName)) {
      return true;
    }
  }
  return false;
}

std::optional<server_model_meta> server_models::mGetMeta(const std::string &pName) {
  std::unique_lock<std::mutex> vLk(vMutex);
  if (vNeedReload) {
    vLk.unlock();
    mLoadModels();
    vLk.lock();
  }

  auto vIt = dMapping.find(pName);
  if (vIt != dMapping.end()) {
    return vIt->second.vMeta;
  }
  for (const auto &[vKey, vInst] : dMapping) {
    if (vInst.vMeta.sAliases.count(pName)) {
      return vInst.vMeta;
    }
  }
  return std::nullopt;
}

std::vector<server_model_meta> server_models::mGetAllMeta() {
  std::unique_lock<std::mutex> vLk(vMutex);
  if (vNeedReload) {
    vLk.unlock();
    mLoadModels();
    vLk.lock();
  }

  std::vector<server_model_meta> lResult;
  lResult.reserve(dMapping.size());
  for (const auto &[vName, vInst] : dMapping) {
    lResult.push_back(vInst.vMeta);
  }
  return lResult;
}

void server_models::mUnloadLru() {
  if (vBaseParams.vModelsMax <= 0) {
    return; // no limit
  }
  // remove one of the servers if we passed the models_max (least recently used - LRU)
  std::string vLruModelName = "";
  int64_t vLruLastUsed = ggml_time_ms();
  size_t vCountActive = 0;
  {
    std::unique_lock<std::mutex> vLk(vMutex);
    for (const auto &vM : dMapping) {
      if (vM.second.vMeta.mIsRunning()) {
        vCountActive++;
        if (vM.second.vMeta.vLastUsed < vLruLastUsed) {
          vLruModelName = vM.first;
          vLruLastUsed = vM.second.vMeta.vLastUsed;
        }
      }
    }
  }
  if (!vLruModelName.empty() && vCountActive >= (size_t)vBaseParams.vModelsMax) {
    SRV_INF("models_max limit reached, removing LRU name=%s\n", vLruModelName.c_str());
    mUnload(vLruModelName);
    // wait for unload to complete
    {
      std::unique_lock<std::mutex> vLk(vMutex);
      vCv.wait(vLk, [this, &vLruModelName]() {
        return dMapping[vLruModelName].vMeta.vStatus == cServerModelStatusUnloaded;
      });
    }
  }
}

void server_models::mLoad(const std::string &pName) { mLoad(pName, load_options{}); }

void server_models::mLoad(const std::string &pName, const load_options &pOpts) {
  if (!pOpts.vCustomMeta.has_value()) {
    if (!mHasModel(pName)) {
      throw std::runtime_error("model name=" + pName + " is not found");
    }
    mUnloadLru();
  }

  std::unique_lock<std::mutex> vLk(vMutex);
  // edge case: block until any in-progress reload has finished so we always load
  // against the freshest preset and a consistent mapping state
  vCv.wait(vLk, [this]() { return !vIsReloading; });

  auto vMeta = pOpts.vCustomMeta.has_value() ? *pOpts.vCustomMeta : dMapping[pName].vMeta;
  if (vMeta.vStatus != cServerModelStatusUnloaded) {
    SRV_INF("model %s is not ready\n", pName.c_str());
    return;
  }

  // Re-check capacity under the lock to prevent concurrent loads from
  // exceeding models_max. Without this, the window between mUnloadLru()
  // releasing its lock and this lock_guard acquiring allows multiple
  // threads to each observe capacity and all proceed to load.
  if (vBaseParams.vModelsMax > 0) {
    size_t vCountActive = 0;
    for (const auto &vM : dMapping) {
      if (vM.second.vMeta.mIsRunning()) {
        vCountActive++;
      }
    }
    if (vCountActive >= (size_t)vBaseParams.vModelsMax) {
      throw std::runtime_error("model limit reached, try again later");
    }
  }

  // prepare new instance info
  instance_t vInst;
  vInst.vMeta = vMeta;
  vInst.vMeta.vPort = fCommonHttpGetFreePort();
  vInst.vMeta.vStatus = cServerModelStatusLoading;
  vInst.vMeta.vLoadedInfo = json{};
  vInst.vMeta.vLastUsed = ggml_time_ms();

  if (vInst.vMeta.vPort <= 0) {
    throw std::runtime_error("failed to get a port number");
  }

  vInst.vSubproc = std::make_shared<server_subproc>();
  {
    SRV_INF("spawning server instance with name=%s on port %d\n", vInst.vMeta.vName.c_str(), vInst.vMeta.vPort);

    vInst.vMeta.mUpdateArgs(vCtxPreset, vBinPath); // render args

    std::vector<std::string> lChildArgs = vInst.vMeta.lArgs; // copy
    std::vector<std::string> lChildEnv = lBaseEnv;           // copy
    lChildEnv.push_back("MGS_SERVER_ROUTER_PORT=" + std::to_string(vBaseParams.vPort));

    if (pOpts.vMode == cServerChildModeDownload) {
      vInst.vMeta.vStatus = cServerModelStatusDownloading;
      lChildEnv.push_back("MGS_SERVER_CHILD_MODE=download");
      lChildEnv.push_back("MGS_ARG_HF_REPO=" + pName);
    }

    SRV_INF("%s", "spawning server instance with args:\n");
    for (const auto &vArg : fRedactArgs(lChildArgs, vCtxPreset)) {
      SRV_INF("  %s\n", vArg.c_str());
    }
    vInst.vMeta.lArgs = lChildArgs; // save for debugging

    // TODO @ngxson : maybe separate stdout and stderr in the future
    //                so that we can use stdout for commands and stderr for logging
    int vOptions = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    if (!vInst.vSubproc->vSproc.mCreate(lChildArgs, vOptions, lChildEnv)) {
      throw std::runtime_error("failed to spawn server instance");
    }
  }

  // start a thread to manage the child process
  // captured variables are guaranteed to be destroyed only after the thread is joined
  vInst.vTh = std::thread([this, pName, vChildProc = vInst.vSubproc, vPort = vInst.vMeta.vPort,
                           vStopTimeout = vInst.vMeta.vStopTimeout, vChildMode = pOpts.vMode]() {
    FILE *vStdinFile = vChildProc->vSproc.mStdinFile();
    FILE *vStdoutFile = vChildProc->vSproc.mStdoutFile(); // combined stdout/stderr

    std::thread vLogThread([&]() {
      // read stdout/stderr and forward to main server log
      // also handle status report from child process
      std::vector<char> lVecBuf(128 * 1024); // large buffer for storing info
      char *vBuffer = lVecBuf.data();
      if (vStdoutFile) {
        while (fgets(vBuffer, lVecBuf.size(), vStdoutFile) != nullptr) {
          LOG("[%5d] %s", vPort, vBuffer);
          std::string vStr(vBuffer);
          if (fStringStartsWith(vBuffer, CMD_CHILD_TO_ROUTER_STATE)) {
            this->mHandleChildState(pName, vStr);
          }
        }
      } else {
        SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", pName.c_str());
      }
    });

    std::thread vStoppingThread([&]() {
      // thread to monitor explicit stop requests; child crash is signalled via vChildProc->vStopped
      auto vIsStopping = [this, &pName]() { return this->sStoppingModels.find(pName) != this->sStoppingModels.end(); };
      {
        std::unique_lock<std::mutex> vLk(this->vMutex);
        this->vCvStop.wait(vLk,
                           [&]() { return vIsStopping() || vChildProc->vStopped.load(std::memory_order_acquire); });
      }
      // child crashed or finished on its own, skip graceful shutdown sequence
      if (vChildProc->vStopped.load(std::memory_order_acquire)) {
        return;
      }
      SRV_INF("stopping model instance name=%s\n", pName.c_str());
      fprintf(vStdinFile, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
      fflush(vStdinFile);
      int64_t vStartTime = ggml_time_ms();
      while (true) {
        std::unique_lock<std::mutex> vLk(this->vMutex);
        if (!vIsStopping() || vChildProc->vStopped.load(std::memory_order_acquire)) {
          return;
        }
        int64_t vElapsed = ggml_time_ms() - vStartTime;
        if (vElapsed >= vStopTimeout * 1000) {
          vLk.unlock();
          SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", pName.c_str(), vStopTimeout);
          vChildProc->mTerminate();
          return;
        }
        this->vCvStop.wait_for(vLk, std::chrono::seconds(1), [&]() {
          return !vIsStopping() || vChildProc->vStopped.load(std::memory_order_acquire);
        });
      }
    });

    // we reach here when the child process exits (stdout EOF)
    // note: we cannot join() prior to this point because it will close stdin_file
    if (vLogThread.joinable()) {
      vLogThread.join();
    }

    vChildProc->vStopped.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> vLk(this->vMutex);
      sStoppingModels.erase(pName);
      vCvStop.notify_all();
    }
    if (vStoppingThread.joinable()) {
      vStoppingThread.join();
    }

    // get the exit code
    int vExitCode = vChildProc->vSproc.mJoin();

    // update status and exit code
    if (vChildMode == cServerChildModeDownload) {
      // instance will be cleaned up on next mLoadModels() call
    } else {
      this->mUpdateStatus(pName, {cServerModelStatusUnloaded, vExitCode});
    }
    SRV_INF("instance name=%s exited with status %d\n", pName.c_str(), vExitCode);
  });

  // clean up old process/thread if exists
  {
    auto &vOldInstance = dMapping[pName];
    // old process should have exited already, but just in case, we clean it up here
    if (vOldInstance.vSubproc && vOldInstance.vSubproc->mIsAlive()) {
      SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", pName.c_str());
      vOldInstance.vSubproc->mTerminate(); // force kill
    }
    if (vOldInstance.vTh.joinable()) {
      vOldInstance.vTh.join();
    }
  }

  mNotifySse("model_status", pName,
             {
                 {"status", fServerModelStatusToString(vInst.vMeta.vStatus)},
             });

  dMapping[pName] = std::move(vInst);
  vCv.notify_all();
}

void server_models::mUnload(const std::string &pName) {
  std::unique_lock<std::mutex> vLk(vMutex);
  auto vIt = dMapping.find(pName);
  if (vIt != dMapping.end()) {
    if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
      SRV_INF("cancelling download for model name=%s\n", pName.c_str());
      vIt->second.vSubproc->mRequestExit();
      // for convenience, we wait the status change here
      mWait(vLk, pName,
            [](const server_model_meta &pNewMeta) { return pNewMeta.vStatus != cServerModelStatusDownloading; });
    } else if (vIt->second.vMeta.mIsRunning()) {
      SRV_INF("stopping model instance name=%s\n", pName.c_str());
      sStoppingModels.insert(pName);
      if (vIt->second.vMeta.vStatus == cServerModelStatusLoading) {
        // special case: if model is in loading state, unloading means force-killing it
        SRV_WRN("model name=%s is still loading, force-killing\n", pName.c_str());
        vIt->second.vSubproc->mTerminate();
      }
      vCvStop.notify_all();
      // status change will be handled by the managing thread
    } else {
      SRV_WRN("model instance name=%s is not running\n", pName.c_str());
    }
  }
}

void server_models::mUnloadAll() {
  std::vector<std::thread> lToJoin;
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    for (auto &[vName, vInst] : dMapping) {
      if (vInst.vMeta.vStatus == cServerModelStatusDownloading) {
        SRV_INF("cancelling download for model name=%s\n", vName.c_str());
        vInst.vSubproc->vStopped.store(true, std::memory_order_relaxed);
      } else if (vInst.vMeta.mIsRunning()) {
        SRV_INF("stopping model instance name=%s\n", vName.c_str());
        sStoppingModels.insert(vName);
        vCvStop.notify_all();
        // status change will be handled by the managing thread
      }
      // moving the thread to join list to avoid deadlock
      lToJoin.push_back(std::move(vInst.vTh));
    }
  }
  for (auto &vTh : lToJoin) {
    if (vTh.joinable()) {
      vTh.join();
    }
  }
}

void server_models::mUpdateStatus(const std::string &pName, const update_status_args &pArgs) {
  std::unique_lock<std::mutex> vLk(vMutex);
  auto vIt = dMapping.find(pName);
  if (vIt != dMapping.end()) {
    auto &vMeta = vIt->second.vMeta;
    vMeta.vStatus = pArgs.vStatus;
    vMeta.vExitCode = pArgs.vExitCode;
    if (!pArgs.vLoadedInfo.is_null()) {
      vMeta.vLoadedInfo = pArgs.vLoadedInfo;
    }
    if (!pArgs.vProgress.is_null()) {
      vMeta.vProgress = pArgs.vProgress;
    }
  }
  // broadcast status change to SSE
  {
    json vData = {
        {"status", fServerModelStatusToString(pArgs.vStatus)},
    };
    if (pArgs.vStatus == cServerModelStatusUnloaded) {
      vData["exit_code"] = pArgs.vExitCode;
    }
    if (!pArgs.vLoadedInfo.is_null()) {
      vData["info"] = pArgs.vLoadedInfo;
    }
    if (!pArgs.vProgress.is_null()) {
      vData["progress"] = pArgs.vProgress;
    }
    // note: notify_sse doesn't acquire the lock, so no deadlock here
    mNotifySse("status_change", pName, vData);
  }
  vCv.notify_all();
}

void server_models::mUpdateDownloadProgress(const std::string &pName, const common_download_progress &pProgress,
                                            bool pDone, bool pOk) {
  json vCurr;
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    auto vIt = dMapping.find(pName);
    if (vIt != dMapping.end()) {
      if (pDone) {
        // mark the instance to be erased on next mLoadModels() call
        vIt->second.vMeta.vStatus = cServerModelStatusDownloaded;
        vNeedReload = true;
      } else {
        json &vInfo = vIt->second.vMeta.vLoadedInfo;
        if (!vInfo.contains("progress")) {
          vInfo["progress"] = json{};
        }
        vInfo["progress"][pProgress.vUrl] = {
            {"done", pProgress.vDownloaded},
            {"total", pProgress.vTotal},
        };
        vCurr = vIt->second.vMeta.vLoadedInfo; // copy
      }
    }
  }
  if (pDone) {
    vCv.notify_all(); // notify in case mUnload() is waiting for download to be cancelled
    mNotifySse(pOk ? "download_finished" : "download_failed", pName, {});
  } else {
    mNotifySse("download_progress", pName, vCurr);
  }
}

bool server_models::mRemove(const std::string &pName) {
  // do everything under one lock acquisition; avoid mGetMeta() /
  // mUnload() because they can trigger mLoadModels() which erases
  // transient DOWNLOADING / DOWNLOADED entries as a side-effect
  std::unique_lock<std::mutex> vLk(vMutex);

  auto vIt = dMapping.find(pName);
  if (vIt == dMapping.end()) {
    throw std::runtime_error("model name=" + pName + " is not found");
  }
  if (vIt->second.vMeta.vSource != cServerModelSourceCache) {
    throw std::runtime_error("model name=" + pName + " is not removable (not from cache)");
  }

  if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
    // cancel in-flight download
    SRV_INF("cancelling download for model name=%s\n", pName.c_str());
    vIt->second.vSubproc->mRequestExit();
  } else if (vIt->second.vMeta.mIsRunning()) {
    // stop running instance
    SRV_INF("stopping model instance name=%s\n", pName.c_str());
    sStoppingModels.insert(pName);
    if (vIt->second.vMeta.vStatus == cServerModelStatusLoading) {
      vIt->second.vSubproc->mTerminate();
    }
    vCvStop.notify_all();
  }

  // wait until the monitoring thread finishes
  mWait(vLk, pName, [](const server_model_meta &pMeta) {
    return pMeta.vStatus == cServerModelStatusUnloaded || pMeta.vStatus == cServerModelStatusDownloaded;
  });

  // re-find after wait - mLoadModels() may have erased the entry during the wait
  vIt = dMapping.find(pName);
  if (vIt == dMapping.end()) {
    // load_models() already joined the thread and erased the entry;
    // we just need to clean up the cached files on disk
    vLk.unlock();
    bool vOk = fCommonDownloadRemove(pName);
    SRV_INF("removing model name=%s from cache (%s)\n", pName.c_str(), vOk ? "succeeded" : "partial");
    mNotifySse("model_remove", pName, {});
    return true;
  }

  // join before erasing - thread no longer acquires this mutex
  if (vIt->second.vTh.joinable()) {
    vIt->second.vTh.join();
  }

  // remove from disk (best-effort: cancelled downloads may have no cached files)
  bool vOk = fCommonDownloadRemove(pName);
  dMapping.erase(pName);
  if (!vOk) {
    SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", pName.c_str());
  }
  SRV_INF("removing model name=%s from cache (%s)\n", pName.c_str(), vOk ? "succeeded" : "partial");
  mNotifySse("model_remove", pName, {});
  return true;
}

void server_models::mWait(const std::string &pName, std::function<bool(const server_model_meta &)> pPredicate) {
  std::unique_lock<std::mutex> vLk(vMutex);
  mWait(vLk, pName, pPredicate);
}

void server_models::mWait(std::unique_lock<std::mutex> &pLk, const std::string &pName,
                          std::function<bool(const server_model_meta &)> pPredicate) {
  vCv.wait(pLk, [this, &pName, &pPredicate]() {
    auto vIt = dMapping.find(pName);
    if (vIt != dMapping.end()) {
      return pPredicate(vIt->second.vMeta);
    }
    // model was removed from mapping by another code path (e.g. mLoadModels()).
    // nothing left to wait for - tell the caller to proceed.
    return true;
  });
}

bool server_models::mEnsureModelReady(const std::string &pName) {
  auto vMeta = mGetMeta(pName);
  if (!vMeta.has_value()) {
    throw std::runtime_error("model name=" + pName + " is not found");
  }
  if (vMeta->mIsReady()) {
    return false; // ready for taking requests
  }
  if (vMeta->vStatus == cServerModelStatusSleeping) {
    return false; // child is sleeping but still running; new request will wake it up
  }
  if (vMeta->vStatus == cServerModelStatusUnloaded) {
    SRV_INF("model name=%s is not loaded, loading...\n", pName.c_str());
    mLoad(pName);
  }

  // wait for loading to complete
  SRV_INF("waiting until model name=%s is fully loaded...\n", pName.c_str());
  mWait(pName, [&vMeta](const server_model_meta &pNewMeta) {
    if (pNewMeta.vStatus != cServerModelStatusLoading) {
      vMeta = pNewMeta; // update meta for final check after wait
      return true;
    }
    return false;
  });

  // check final status
  if (!vMeta.has_value() || vMeta->mIsFailed()) {
    throw std::runtime_error("model name=" + pName + " failed to load");
  }

  return true;
}

server_http_res_ptr server_models::mProxyRequest(const server_http_req &pReq, const std::string &pMethod,
                                                 const std::string &pName, bool pUpdateLastUsed, bool pDetached) {
  auto vMeta = mGetMeta(pName);
  if (!vMeta.has_value()) {
    throw std::runtime_error("model name=" + pName + " is not found");
  }
  if (!vMeta->mIsRunning()) {
    throw std::invalid_argument("model name=" + pName + " is not running");
  }
  if (pUpdateLastUsed) {
    std::unique_lock<std::mutex> vLk(vMutex);
    dMapping[pName].vMeta.vLastUsed = ggml_time_ms();
  }
  SRV_INF("proxying request to model %s on port %d\n", pName.c_str(), vMeta->vPort);
  std::string vProxyPath = "/api" + pReq.vPath;
  if (!pReq.vQueryString.empty()) {
    vProxyPath += '?' + pReq.vQueryString;
  }
  // the router has already checked the browser's Origin; a child only sees the router (its Host is the loopback
  // address), so a child running with tools or MCP servers would refuse the forwarded POST as cross-site
  std::map<std::string, std::string> dChildHeaders = pReq.dHeaders;
  for (auto vIt = dChildHeaders.begin(); vIt != dChildHeaders.end();) {
    const std::string &cKey = vIt->first;
    const bool cIsOrigin =
        cKey.size() == 6 && std::equal(cKey.begin(), cKey.end(), "origin", [](char pA, char pB) {
          return std::tolower(static_cast<unsigned char>(pA)) == pB;
        });
    vIt = cIsOrigin ? dChildHeaders.erase(vIt) : std::next(vIt);
  }
  auto vProxy = std::make_unique<server_http_proxy>(
      pMethod, "http", CHILD_ADDR, vMeta->vPort, vProxyPath, dChildHeaders, pReq.vBody, pReq.dFiles,
      // a detached request belongs to a replay session that outlives the client socket:
      // it reaches the child even when the downstream died during the load wait, the
      // session buffer is the recipient and DELETE remains the stop
      pDetached ? std::function<bool()>([]() { return false; }) : pReq.vShouldStop, vBaseParams.vTimeoutRead,
      vBaseParams.vTimeoutWrite);
  return vProxy;
}

void server_models::mHandleChildState(const std::string &pName, const std::string &pRawInput) {
  server_state vState;
  json vPayload;

  try {
    json vData = json::parse(pRawInput.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
    vState = fServerStateFromStr(fJsonValue(vData, "state", std::string()));
    vPayload = fJsonValue(vData, "payload", json{});
  } catch (const std::exception &vE) {
    SRV_ERR("failed to parse child state update for name=%s: %s\n", pName.c_str(), vE.what());
    return;
  }

  switch (vState) {
  case cServerStateDownloading: {
    std::string vResult = fJsonValue(vPayload, "result", std::string());
    std::string vUrl = fJsonValue(vPayload, "url", std::string());
    auto vRequestExit = [&]() {
      std::lock_guard<std::mutex> vLk(vMutex);
      auto vIt = dMapping.find(pName);
      if (vIt != dMapping.end()) {
        return vIt->second.vSubproc->mRequestExit();
      }
    };
    if (vResult == "download_finished") {
      mUpdateDownloadProgress(pName, {}, true, true);
      vRequestExit();
    } else if (vResult == "download_failed") {
      mUpdateDownloadProgress(pName, {}, true, false);
      vRequestExit();
    } else if (!vUrl.empty()) {
      common_download_progress vP;
      vP.vUrl = vUrl;
      vP.vDownloaded = fJsonValue(vPayload, "downloaded", (size_t)0);
      vP.vTotal = fJsonValue(vPayload, "total", (size_t)0);
      mUpdateDownloadProgress(pName, vP, false);
    }
  } break;
  case cServerStateLoading: {
    mUpdateStatus(pName, {
                             cServerModelStatusLoading,
                             0,
                             nullptr, // no loaded_info yet
                             vPayload,
                         });
  } break;
  case cServerStateReady: {
    mUpdateStatus(pName, {
                             cServerModelStatusLoaded,
                             0,
                             // note: payload can be empty if this is a wakeup from sleep
                             vPayload.size() > 0 ? vPayload : nullptr,
                             {}, // reset progress info
                         });
  } break;
  case cServerStateSleeping: {
    mUpdateStatus(pName, {cServerModelStatusSleeping});
  } break;
  default:
    // should never happen, but just in case
    GGML_ASSERT(false && "unexpected state from child server");
  }
}

//
// server_child
//

bool server_child::mIsChild() {
  const char *vRouterPort = std::getenv("MGS_SERVER_ROUTER_PORT");
  return vRouterPort != nullptr;
}

server_child_mode server_child::mGetMode() {
  const char *vMode = std::getenv("MGS_SERVER_CHILD_MODE");
  std::string vModeStr(vMode ? vMode : "");
  if (vModeStr == "download") {
    return cServerChildModeDownload;
  } else {
    return cServerChildModeNormal;
  }
}

struct server_download_state : public common_download_callback {
  server_child *vSelf;
  std::function<bool()> vShouldStop;
  std::atomic<int64_t> vLastProgressTime{0}; // multiple files downloading in different threads
  bool vIsOk = false;

  server_download_state(server_child *pS) : vSelf(pS) {}

  bool mRun(common_params &pParams) {
    try {
      common_models_handler vHandler = fCommonModelsHandlerInit(pParams, cMgsExampleServer);
      fCommonModelsHandlerApply(vHandler, pParams, this);
      vIsOk = true;
    } catch (const std::exception &vE) {
      auto vModelName = pParams.vModel.mGetName();
      SRV_ERR("download failed for model name=%s: %s\n", vModelName.c_str(), vE.what());
      vIsOk = false;
    }
    return vIsOk;
  }
  void mOnProgress(const common_download_progress &pP) {
    json vData = {
        {"url", pP.vUrl},
        {"downloaded", pP.vDownloaded},
        {"total", pP.vTotal},
    };
    vSelf->mNotifyToRouter(fServerStateToStr(cServerStateDownloading), vData);
  }
  void mOnStart(const common_download_progress &pP) override { mOnProgress(pP); }
  void mOnUpdate(const common_download_progress &pP) override {
    int64_t vNow = ggml_time_ms();
    // throttle progress updates to avoid flooding logs
    if (vNow - vLastProgressTime.load(std::memory_order_relaxed) >= 100) {
      mOnProgress(pP);
      vLastProgressTime.store(vNow, std::memory_order_relaxed);
    }
  }
  void mOnDone(const common_download_progress &pP, bool) override { mOnProgress(pP); }
  bool mIsCancelled() const override { return vShouldStop ? vShouldStop() : false; }
};

int server_child::mRunDownload(common_params &pParams) {
  auto vCancelled = std::make_shared<std::atomic<bool>>(false);

  // monitor stdin for cancellation command from the router
  std::thread vSignalThread = mSetup([vCancelled](int) { vCancelled->store(true, std::memory_order_relaxed); });

  server_download_state vDl(this);
  vDl.vShouldStop = [vCancelled]() { return vCancelled->load(std::memory_order_relaxed); };

  bool vOk = vDl.mRun(pParams);

  mNotifyToRouter(fServerStateToStr(cServerStateDownloading),
                  {
                      {"result", vOk ? "download_finished" : "download_failed"},
                  });

  // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
  if (vSignalThread.joinable()) {
    vSignalThread.join();
  }

  SRV_INF("download completed %s\n", vOk ? "successfully" : "with errors");
  return 0;
}

std::thread server_child::mSetup(const std::function<void(int)> &pShutdownHandler) {
  // setup thread for monitoring stdin
  return std::thread([pShutdownHandler]() {
    // wait for EOF on stdin
    SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
    bool vEof = false;
    while (true) {
      std::string vLine;
      if (!std::getline(std::cin, vLine)) {
        // EOF detected, that means the router server is unexpectedly exit or killed
        vEof = true;
        break;
      }
      if (vLine.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
        SRV_INF("%s", "exit command received, exiting...\n");
        pShutdownHandler(0);
        break;
      }
    }
    if (vEof) {
      SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
      exit(1);
    }
  });
}

void server_child::mNotifyToRouter(const std::string &pState, const json &pPayload) {
  json vData = {
      {"state", pState},
      {"payload", pPayload},
  };
  std::lock_guard<std::mutex> vLk(vMtxStdout);
  fCommonLogPause(fCommonLogMain());
  fflush(stdout);
  fprintf(stdout, "%s%s\n", CMD_CHILD_TO_ROUTER_STATE, fSafeJsonToStr(vData).c_str());
  fflush(stdout);
  fCommonLogResume(fCommonLogMain());
}

//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> vSseClientIdCounter = 0;
struct server_models_sse_client {
  server_response &vQueueResults;
  int vClientId;
  server_models_sse_client(server_response &pQ)
      : vQueueResults(pQ), vClientId(vSseClientIdCounter.fetch_add(1, std::memory_order_relaxed)) {
    SRV_DBG("new SSE client connected, assigned client_id=%d\n", vClientId);
    vQueueResults.mAddWaitingTaskId(vClientId);
  }
  ~server_models_sse_client() {
    SRV_DBG("SSE client disconnected, removing client_id=%d\n", vClientId);
    vQueueResults.mRemoveWaitingTaskId(vClientId);
  }

  // return nullptr if pShouldStop() is true before receiving a result
  // note: if one error is received, it will stop further processing and return error result
  server_task_result_ptr mNext(const std::function<bool()> &pShouldStop) {
    while (true) {
      static const int cHttpPollingSeconds = 1; // check should_stop every 1 second
      server_task_result_ptr vResult = vQueueResults.mRecvWithTimeout({vClientId}, cHttpPollingSeconds);
      if (vResult == nullptr) {
        // timeout, check stop condition
        if (pShouldStop()) {
          return nullptr;
        }
        // continue waiting otherwise
      } else {
        SRV_DBG("recv result for client_id=%d: %s\n", vClientId, fSafeJsonToStr(vResult->to_json()).c_str());
        return vResult;
      }
    }
    // should not reach here
  }
};

static void fResOk(std::unique_ptr<server_http_res> &pRes, const json &pResponseData) {
  pRes->vStatus = 200;
  pRes->vData = fSafeJsonToStr(pResponseData);
}

static void fResErr(std::unique_ptr<server_http_res> &pRes, const json &pErrorData) {
  pRes->vStatus = fJsonValue(pErrorData, "code", 500);
  pRes->vData = fSafeJsonToStr({{"error", pErrorData}});
}

static bool fRouterValidateModel(std::string &pName, server_models &pModels, bool pModelsAutoload,
                                 std::unique_ptr<server_http_res> &pRes) {
  if (pName.empty()) {
    fResErr(pRes, fFormatErrorResponse("model name is missing from the request", cErrorTypeInvalidRequest));
    return false;
  }
  auto vMeta = pModels.mGetMeta(pName);
  if (!vMeta.has_value()) {
    fResErr(pRes, fFormatErrorResponse(fStringFormat("model '%s' not found", pName.c_str()), cErrorTypeInvalidRequest));
    return false;
  }
  // resolve alias to canonical model name
  pName = vMeta->vName;
  if (!pModelsAutoload && !vMeta->mIsRunning()) {
    fResErr(pRes, fFormatErrorResponse("model is not loaded", cErrorTypeInvalidRequest));
    return false;
  }
  return true;
}

static bool fIsAutoload(const common_params &pParams, const server_http_req &pReq) {
  std::string vAutoload = pReq.mGetParam("autoload");
  if (vAutoload.empty()) {
    return pParams.vModelsAutoload;
  } else {
    return vAutoload == "true" || vAutoload == "1";
  }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string fEncodeQs(const std::string &pIn) {
  std::string vOut;
  vOut.reserve(pIn.size() * 3);
  for (unsigned char vC : pIn) {
    bool vSafe = (vC >= 'A' && vC <= 'Z') || (vC >= 'a' && vC <= 'z') || (vC >= '0' && vC <= '9') || vC == '-' ||
                 vC == '_' || vC == '.' || vC == '~';
    if (vSafe) {
      vOut.push_back(char(vC));
    } else {
      char aBuf[4];
      std::snprintf(aBuf, sizeof(aBuf), "%%%02X", vC);
      vOut.append(aBuf, 3);
    }
  }
  return vOut;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> fResolveChildForConv(server_models &pModels,
                                                             const std::string &pConversationId) {
  if (pConversationId.empty()) {
    return std::nullopt;
  }
  auto vTracked = pModels.vConvModels.mLookup(pConversationId);
  if (!vTracked.has_value()) {
    return std::nullopt;
  }
  auto vMeta = pModels.mGetMeta(*vTracked);
  if (vMeta.has_value() && vMeta->mIsReady()) {
    return vMeta;
  }
  return std::nullopt;
}

// Memory snapshots of the running children. Loading and sleeping children only report their state,
// so the monitor never loads or wakes a model.
static std::vector<json> fCollectMonitorSnapshots(server_models &pModels) {
  std::vector<json> lLSnapshots;
  for (const server_model_meta &vMeta : pModels.mGetAllMeta()) {
    if (!vMeta.mIsRunning()) {
      continue;
    }
    if (!vMeta.mIsReady()) {
      lLSnapshots.push_back({{"state", fServerModelStatusToString(vMeta.vStatus)}, {"model", vMeta.vName}});
      continue;
    }
    try {
      httplib::Client vClient(CHILD_ADDR, vMeta.vPort);
      vClient.set_connection_timeout(0, cMonitorChildTimeoutMs * 1000);
      vClient.set_read_timeout(0, cMonitorChildTimeoutMs * 1000);
      vClient.set_write_timeout(0, cMonitorChildTimeoutMs * 1000);
      auto vResponse = vClient.Get("/api/monitor/memory");
      if (!vResponse || vResponse->status != 200) {
        lLSnapshots.push_back({{"state", "unavailable"}, {"model", vMeta.vName}});
        continue;
      }
      json dBody = json::parse(vResponse->body);
      for (json dSnapshot : dBody.value("models", json::array())) {
        dSnapshot["model"] = vMeta.vName;
        lLSnapshots.push_back(std::move(dSnapshot));
      }
    } catch (const std::exception &) {
      lLSnapshots.push_back({{"state", "unavailable"}, {"model", vMeta.vName}});
    }
  }
  return lLSnapshots;
}

void server_models_routes::mInitRoutes() {
  if (!common_subproc::mIsSupported()) {
    throw std::runtime_error("subprocess is not enabled on this build");
  }

  this->vGetRouterMonitor = [this](const server_http_req &) {
    auto vRes = std::make_unique<server_http_res>();
    fResOk(vRes, fMonitorReport(fCollectMonitorSnapshots(vModels)));
    return vRes;
  };

  this->vGetRouterMonitorMemory = [this](const server_http_req &) {
    auto vRes = std::make_unique<server_http_res>();
    json aModels = json::array();
    for (json &dSnapshot : fCollectMonitorSnapshots(vModels)) {
      aModels.push_back(std::move(dSnapshot));
    }
    fResOk(vRes, json{{"models", aModels}});
    return vRes;
  };

  this->vGetRouterProps = [this](const server_http_req &pReq) {
    std::string vName = pReq.mGetParam("model");
    if (vName.empty()) {
      // main instance
      auto vRes = std::make_unique<server_http_res>();
      fResOk(vRes, {
                       // TODO: add support for this on web UI
                       {"role", "router"},
                       {"max_instances", vParams.vModelsMax},
                       {"models_autoload", vParams.vModelsAutoload},
                       // this is a dummy response to make sure the UI doesn't break
                       {"model_alias", "mgs"},
                       {"model_path", "none"},
                       {"default_generation_settings",
                        {
                            {"params", json{}},
                            {"n_ctx", 0},
                        }},
                       // New key
                       {"ui_settings", vUiSettings},
                       {"build_info", std::string(mgs_version())},
                       {"cors_proxy_enabled", vParams.vUiMcpProxy},
                       {"shell_mcp_server", vParams.vShellMcpServer},
                   });
      return vRes;
    }
    return vProxyGet(pReq);
  };

  this->vProxyGet = [this](const server_http_req &pReq) {
    std::string vMethod = "GET";
    std::string vName = pReq.mGetParam("model");
    bool vAutoload = fIsAutoload(vParams, pReq);
    auto vErrorRes = std::make_unique<server_http_res>();
    if (!fRouterValidateModel(vName, vModels, vAutoload, vErrorRes)) {
      return vErrorRes;
    }
    if (vAutoload) {
      vModels.mEnsureModelReady(vName);
    }
    return vModels.mProxyRequest(pReq, vMethod, vName, false);
  };

  this->vProxyPost = [this](const server_http_req &pReq) {
    std::string vMethod = "POST";
    json vBody = json::parse(pReq.vBody);
    std::string vName = fJsonValue(vBody, "model", std::string());
    bool vAutoload = fIsAutoload(vParams, pReq);
    auto vErrorRes = std::make_unique<server_http_res>();
    if (!fRouterValidateModel(vName, vModels, vAutoload, vErrorRes)) {
      return vErrorRes;
    }
    // remember which child serves this conversation so the stream routes can route straight
    // to it without polling, keyed on the exact conv id from the header. registered before
    // the load wait so a stop issued while the model loads can erase the entry and cancel
    // this request instead of leaving an orphan generation
    std::string vConvId = fServerStreamConvIdFromHeaders(pReq.dHeaders);
    uint64_t vTicket = vModels.vConvModels.mRemember(vConvId, vName);
    bool vWaited = vAutoload && vModels.mEnsureModelReady(vName);
    if (vTicket != 0 && !vModels.vConvModels.mAlive(vConvId, vTicket)) {
      SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n", vConvId.c_str(), vName.c_str());
      fResErr(vErrorRes, fFormatErrorResponse("request cancelled by a stop while the model was loading",
                                              cErrorTypeInvalidRequest));
      return vErrorRes;
    }
    // a session request that waited for a load detaches from the client socket: the
    // client may have dropped during the wait (page reload) and the session buffer must
    // still receive the generation for a later resume
    return vModels.mProxyRequest(pReq, vMethod, vName, true,
                                 vWaited && vTicket != 0); // update last usage for POST request only
  };

  this->vPostRouterModelsLoad = [this](const server_http_req &pReq) {
    auto vRes = std::make_unique<server_http_res>();
    json vBody = json::parse(pReq.vBody);
    std::string vName = fJsonValue(vBody, "model", std::string());
    auto vMeta = vModels.mGetMeta(vName);
    if (!vMeta.has_value()) {
      fResErr(vRes, fFormatErrorResponse("model is not found", cErrorTypeNotFound));
      return vRes;
    }
    if (vMeta->mIsRunning()) {
      fResErr(vRes, fFormatErrorResponse("model is already running", cErrorTypeInvalidRequest));
      return vRes;
    }
    vModels.mLoad(vMeta->vName);
    fResOk(vRes, {{"success", true}});
    return vRes;
  };

  this->vGetRouterModels = [this](const server_http_req &pReq) {
    bool vReload = !pReq.mGetParam("reload", "").empty();
    if (vReload) {
      vModels.mLoadModels();
    }
    auto vRes = std::make_unique<server_http_res>();
    json vModelsJson = json::array();
    auto vAllModels = vModels.mGetAllMeta();
    std::time_t vT = std::time(0);
    for (const auto &vMeta : vAllModels) {
      json vStatus{
          {"value", fServerModelStatusToString(vMeta.vStatus)},
          {"args", fRedactArgs(vMeta.lArgs, vModels.vCtxPreset)},
      };
      if (!vMeta.vPreset.vName.empty()) {
        common_preset vPresetCopy = fRedactPreset(vMeta.vPreset);
        fUnsetReservedArgs(vPresetCopy, false);
        vPresetCopy.mUnsetOption("MGS_ARG_HOST");
        vPresetCopy.mUnsetOption("MGS_ARG_PORT");
        vPresetCopy.mUnsetOption("MGS_ARG_ALIAS");
        vPresetCopy.mUnsetOption("MGS_ARG_TAGS");
        vStatus["preset"] = vPresetCopy.mToIni();
      }
      if (vMeta.mIsFailed()) {
        vStatus["exit_code"] = vMeta.vExitCode;
        vStatus["failed"] = true;
      }

      // pi coding agent multimodal compatibility
      json vInputModalities = json::array({"text"});
      if (vMeta.vMultimodal.inp_vision) {
        vInputModalities.push_back("image");
      }
      if (vMeta.vMultimodal.inp_audio) {
        vInputModalities.push_back("audio");
      }
      json vArchitecture{
          {"input_modalities", vInputModalities},
          {"output_modalities", json::array({"text"})},
      };

      json vModelInfo = json{
          {"id", vMeta.vName},
          {"aliases", vMeta.sAliases},
          {"tags", vMeta.sTags},
          {"object", "model"}, // for OAI-compat
          {"owned_by", "mgs"}, // for OAI-compat
          {"created", vT},     // for OAI-compat
          {"status", vStatus},
          {"architecture", vArchitecture},
          {"source", fServerModelSourceToString(vMeta.vSource)},
          {"can_remove", vMeta.vSource == cServerModelSourceCache},
          // {"need_download", meta.need_download},
          // TODO: add other fields, may require reading GGUF metadata
      };

      // merge with loaded_info from the child process if available
      if (vMeta.mIsRunning()) {
        for (auto vIt = vMeta.vLoadedInfo.begin(); vIt != vMeta.vLoadedInfo.end(); ++vIt) {
          if (!vModelInfo.contains(vIt.key())) {
            vModelInfo[vIt.key()] = vIt.value();
          }
        }
      }
      vModelsJson.push_back(vModelInfo);
    }
    fResOk(vRes, {
                     {"data", vModelsJson},
                     {"object", "list"},
                 });
    return vRes;
  };

  this->vPostRouterModelsUnload = [this](const server_http_req &pReq) {
    auto vRes = std::make_unique<server_http_res>();
    json vBody = json::parse(pReq.vBody);
    std::string vName = fJsonValue(vBody, "model", std::string());
    auto vModel = vModels.mGetMeta(vName);
    if (!vModel.has_value()) {
      fResErr(vRes, fFormatErrorResponse("model is not found", cErrorTypeInvalidRequest));
      return vRes;
    }
    if (!vModel->mIsRunning() && vModel->vStatus != cServerModelStatusDownloading) {
      fResErr(vRes, fFormatErrorResponse("model is not running", cErrorTypeInvalidRequest));
      return vRes;
    }
    vModels.mUnload(vModel->vName);
    fResOk(vRes, {{"success", true}});
    return vRes;
  };

  this->vGetRouterModelsSse = [this](const server_http_req &pReq) {
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 200;
    vRes->vContentType = "text/event-stream";
    auto vSseClient = std::make_shared<server_models_sse_client>(vModels.vSse);
    vRes->vNext = [this, vSseClient, &pReq](std::string &pOutput) -> bool {
      auto vResult =
          vSseClient->mNext([&]() { return vStopping.load(std::memory_order_relaxed) || pReq.vShouldStop(); });
      if (vResult == nullptr) {
        return false; // client disconnected or should_stop
      }
      pOutput = "data: " + fSafeJsonToStr(vResult->to_json()) + "\n\n";
      return true; // listen for the next event
    };
    return vRes;
  };

  this->vPostRouterModels = [this](const server_http_req &pReq) {
    auto vRes = std::make_unique<server_http_res>();

    json vBody = json::parse(pReq.vBody);
    std::string vName = fJsonValue(vBody, "model", std::string());
    if (vName.empty()) {
      throw std::invalid_argument("model must be a non-empty string");
    }

    common_params vP;
    vP.vModel.vHfRepo = vName;
    vP.hf_token = vParams.hf_token;

    // validate by fetching metadata
    bool vOk = false;
    try {
      fCommonModelsHandlerInit(vP, cMgsExampleServer);
      vOk = true;
    } catch (...) {
      SRV_ERR("unknown error while validating model '%s'\n", vName.c_str());
      // other exceptions will be handled by the outer ex_wrapper()
      throw;
    }

    if (!vOk) {
      throw std::invalid_argument("model validation failed, unable to download");
    }

    // reject if model already exists
    if (vModels.mHasModel(vName)) {
      throw std::invalid_argument("model '" + vName + "' already exists");
    }

    // then, proceed with the actual download
    SRV_INF("starting download for model '%s'\n", vName.c_str());
    {
      server_models::load_options vLoadOpts;
      vLoadOpts.vMode = cServerChildModeDownload;
      vLoadOpts.vCustomMeta = server_model_meta{};
      vLoadOpts.vCustomMeta->vSource = cServerModelSourceCache;
      vLoadOpts.vCustomMeta->vName = vName;
      vModels.mLoad(vName, vLoadOpts);
    }

    fResOk(vRes, {{"success", true}});
    return vRes;
  };

  this->vDelRouterModels = [this](const server_http_req &pReq) {
    auto vRes = std::make_unique<server_http_res>();

    std::string vName = pReq.mGetParam("model");
    if (vName.empty()) {
      throw std::invalid_argument("model must be a non-empty string");
    }

    vModels.mRemove(vName); // throws on error

    fResOk(vRes, {{"success", true}});
    return vRes;
  };

  this->vRouterStreamGet = [this](const server_http_req &pReq) {
    // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
    // map, 404 when nothing maps
    auto vRes = std::make_unique<server_http_res>();
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      fResErr(vRes, fFormatErrorResponse("Missing conversation id in path", cErrorTypeInvalidRequest));
      return vRes;
    }
    std::optional<server_model_meta> vOwner = fResolveChildForConv(vModels, vConvId);
    if (!vOwner.has_value()) {
      // a registered conv whose model is still loading earns a retry: the session appears
      // once the load ends and the pending request reaches the child
      auto vTracked = vModels.vConvModels.mLookup(vConvId);
      auto vMeta = vTracked.has_value() ? vModels.mGetMeta(*vTracked) : std::nullopt;
      bool vTransient = vMeta.has_value() && (vMeta->vStatus == cServerModelStatusLoading ||
                                              vMeta->vStatus == cServerModelStatusDownloading ||
                                              vMeta->vStatus == cServerModelStatusDownloaded);
      if (vTransient) {
        fResErr(vRes, fFormatErrorResponse("Stream owner model is loading, retry later", cErrorTypeUnavailable));
      } else {
        fResErr(vRes, fFormatErrorResponse("Stream not found or expired", cErrorTypeNotFound));
      }
      return vRes;
    }
    std::string vFrom = pReq.mGetParam("from");
    std::string vChildPath = "/api/v1/stream?conv_id=" + fEncodeQs(vConvId);
    if (!vFrom.empty()) {
      vChildPath += "&from=" + vFrom;
    }
    SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n", vOwner->vName.c_str(), vOwner->vPort,
            vChildPath.c_str());
    auto vProxy = std::make_unique<server_http_proxy>("GET", "http", CHILD_ADDR, vOwner->vPort, vChildPath,
                                                      pReq.dHeaders, pReq.vBody, pReq.dFiles, pReq.vShouldStop,
                                                      vParams.vTimeoutRead, vParams.vTimeoutWrite);
    return std::unique_ptr<server_http_res>(std::move(vProxy));
  };

  this->vRouterStreamsLookup = [this](const server_http_req &pReq) {
    // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
    // map, group the ids per child, and query only the children that actually own some of
    // them instead of fanning out to every ready child. a child only answers for the ids
    // it owns, never lists anything else
    auto vRes = std::make_unique<server_http_res>();
    std::vector<std::string> lRequested;
    try {
      json vBody = json::parse(pReq.vBody);
      if (vBody.contains("conversation_ids") && vBody["conversation_ids"].is_array()) {
        for (const auto &vV : vBody["conversation_ids"]) {
          if (vV.is_string() && !vV.get<std::string>().empty()) {
            lRequested.push_back(vV.get<std::string>());
          }
        }
      }
    } catch (const std::exception &) {
      fResOk(vRes, json::array());
      return vRes;
    }

    // group requested ids by the child port that owns them, drop ids that map to nothing
    std::unordered_map<int, json> dPerChild;
    for (const auto &vCid : lRequested) {
      auto vOwner = fResolveChildForConv(vModels, vCid);
      if (!vOwner.has_value()) {
        continue;
      }
      dPerChild[vOwner->vPort].push_back(vCid);
    }

    json vAggregated = json::array();
    for (auto &[vPort, vIds] : dPerChild) {
      json vChildBody = {{"conversation_ids", vIds}};
      httplib::Client vCli(CHILD_ADDR, vPort);
      vCli.set_connection_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_read_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_write_timeout(0, cStreamLookupTimeoutMs * 1000);
      auto vResp = vCli.Post("/api/v1/streams/lookup", vChildBody.dump(), "application/json");
      if (!vResp || vResp->status != 200) {
        continue;
      }
      try {
        json vChildArr = json::parse(vResp->body);
        if (!vChildArr.is_array()) {
          continue;
        }
        for (auto &vEntry : vChildArr) {
          if (vEntry.is_object()) {
            vAggregated.push_back(vEntry);
          }
        }
      } catch (const std::exception &) {
        continue;
      }
    }
    fResOk(vRes, vAggregated);
    return vRes;
  };

  this->vRouterStreamDelete = [this](const server_http_req &pReq) {
    // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
    // it, evict_and_cancel is idempotent on the child
    auto vRes = std::make_unique<server_http_res>();
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      fResErr(vRes, fFormatErrorResponse("Missing conversation id in path", cErrorTypeInvalidRequest));
      return vRes;
    }
    std::string vChildPath = "/api/v1/stream?conv_id=" + fEncodeQs(vConvId);
    auto vOwner = fResolveChildForConv(vModels, vConvId);
    if (vOwner.has_value()) {
      httplib::Client vCli(CHILD_ADDR, vOwner->vPort);
      vCli.set_connection_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_read_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_write_timeout(0, cStreamLookupTimeoutMs * 1000);
      auto vResp = vCli.Delete(vChildPath.c_str());
      (void)vResp; // the child logs its own miss when the session is unknown there
    } else if (auto vTracked = vModels.vConvModels.mLookup(vConvId); vTracked.has_value()) {
      // the entry exists but its model is still loading: the forget below erases it,
      // which cancels the request parked in proxy_post before the generation starts
      SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
              vConvId.c_str(), vTracked->c_str());
    } else {
      SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n", vConvId.c_str());
    }
    // drop the tracking entry, the session is being torn down
    vModels.vConvModels.mForget(vConvId);
    vRes->vStatus = 204;
    vRes->vContentType = "application/json";
    return vRes;
  };
}

//
// server_http_proxy
//

static std::string fToLowerCopy(const std::string &pValue) {
  std::string vLowered(pValue.size(), '\0');
  std::transform(pValue.begin(), pValue.end(), vLowered.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return vLowered;
}

static bool fShouldStripProxyHeader(const std::string &pHeaderName) {
  // Headers that get duplicated when router forwards child responses
  if (pHeaderName == "server" || pHeaderName == "transfer-encoding" || pHeaderName == "content-length" ||
      pHeaderName == "keep-alive") {
    return true;
  }

  // Router injects CORS, child also sends them: duplicate
  if (pHeaderName.rfind("access-control-", 0) == 0) {
    return true;
  }

  return false;
}

static std::string fGenerateMultipartBoundary() {
  thread_local std::mt19937 vGen(std::random_device{}());
  static const char aChars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
  std::uniform_int_distribution<> vDis(0, sizeof(aChars) - 2);
  std::string vBoundary = "----mgs-proxy-";
  for (int vI = 0; vI < 16; vI++) {
    vBoundary += aChars[vDis(vGen)];
  }
  return vBoundary;
}

static std::string fBuildMultipartBody(const json &pFormFields, const std::map<std::string, uploaded_file> &pFiles,
                                       const std::string &pBoundary) {
  static auto vSanitizeField = [](const std::string &pText) {
    std::string vResult;
    vResult.reserve(pText.size());
    for (char vC : pText) {
      if (vC != '\n' && vC != '\r' && vC != '"') {
        vResult += vC;
      }
    }
    return vResult;
  };

  std::ostringstream vBody;

  for (const auto &[vKey, vValue] : pFormFields.items()) {
    if (vValue.is_array()) {
      for (const auto &vItem : vValue) {
        vBody << "--" << pBoundary << "\r\n";
        vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(vKey) << "\"\r\n";
        vBody << "\r\n";
        if (!vItem.is_string()) {
          throw std::invalid_argument("expected string");
        }
        vBody << vItem.get<std::string>() << "\r\n";
      }
    } else {
      vBody << "--" << pBoundary << "\r\n";
      vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(vKey) << "\"\r\n";
      vBody << "\r\n";
      if (!vValue.is_string()) {
        throw std::invalid_argument("expected string");
      }
      vBody << vValue.get<std::string>() << "\r\n";
    }
  }

  for (const auto &[vKey, vFile] : pFiles) {
    vBody << "--" << pBoundary << "\r\n";
    vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(vKey) << "\"";
    if (!vFile.vFilename.empty()) {
      vBody << "; filename=\"" << vSanitizeField(vFile.vFilename) << "\"";
    }
    vBody << "\r\n";
    if (!vFile.vContentType.empty()) {
      vBody << "Content-Type: " << vSanitizeField(vFile.vContentType) << "\r\n";
    } else {
      vBody << "Content-Type: application/octet-stream\r\n";
    }
    vBody << "\r\n";
    vBody.write(reinterpret_cast<const char *>(vFile.vData.data()), vFile.vData.size());
    vBody << "\r\n";
  }

  vBody << "--" << pBoundary << "--\r\n";
  return vBody.str();
}

server_http_proxy::server_http_proxy(const std::string &pMethod, const std::string &pScheme, const std::string &pHost,
                                     int pPort, const std::string &pPath,
                                     const std::map<std::string, std::string> &pHeaders, const std::string &pBody,
                                     const std::map<std::string, uploaded_file> &pFiles,
                                     const std::function<bool()> pShouldStop, int32_t pTimeoutRead,
                                     int32_t pTimeoutWrite) {
  // shared between reader and writer threads
  auto vCli = std::make_shared<httplib::ClientImpl>(pHost, pPort);
  auto vPipe = std::make_shared<server_pipe<msg_t>>();

  if (pScheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    vCli.reset(new httplib::SSLClient(pHost, pPort));
#else
    throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
  }

  // setup Client
  vCli->set_follow_location(true);
  vCli->set_connection_timeout(pTimeoutRead, 0); // use --timeout value instead of hardcoded 5 s
  vCli->set_write_timeout(pTimeoutRead, 0);      // reversed for cli (client) vs srv (server)
  vCli->set_read_timeout(pTimeoutWrite, 0);
  this->vStatus = 500; // to be overwritten upon response
  this->vCleanup = [vPipe]() {
    vPipe->mCloseRead();
    vPipe->mCloseWrite();
  };

  // wire up the receive end of the pipe
  this->vNext = [vPipe, pShouldStop](std::string &pOut) -> bool {
    msg_t vMsg;
    bool vHasNext = vPipe->mRead(vMsg, pShouldStop);
    if (!vMsg.vData.empty()) {
      pOut = std::move(vMsg.vData);
    }
    return vHasNext; // false if EOF or pipe broken
  };

  // wire up the HTTP client
  // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
  httplib::ResponseHandler vResponseHandler = [vPipe, vCli](const httplib::Response &pResponse) {
    msg_t vMsg;
    vMsg.vStatus = pResponse.status;
    for (const auto &[vKey, vValue] : pResponse.headers) {
      const auto cLowered = fToLowerCopy(vKey);
      if (fShouldStripProxyHeader(cLowered)) {
        continue;
      }
      if (cLowered == "content-type") {
        vMsg.vContentType = vValue;
        continue;
      }
      vMsg.dHeaders[vKey] = vValue;
    }
    return vPipe->mWrite(std::move(vMsg)); // send headers first
  };
  httplib::ContentReceiverWithProgress vContentReceiver = [vPipe](const char *pData, size_t pDataLength, size_t,
                                                                  size_t) {
    // send data chunks
    // returns false if pipe is closed / broken (signal to stop receiving)
    return vPipe->mWrite({{}, 0, std::string(pData, pDataLength), ""});
  };

  // when files are present, the body was converted from multipart form data to JSON
  // we need to reconstruct the multipart body for the downstream server
  std::string vEffectiveBody = pBody;
  std::string vOverrideContentType;
  bool vHasFiles = !pFiles.empty();

  if (vHasFiles) {
    json vFormFields = json::parse(pBody, nullptr, false);
    if (!vFormFields.is_discarded()) {
      auto vBoundary = fGenerateMultipartBoundary();
      vEffectiveBody = fBuildMultipartBody(vFormFields, pFiles, vBoundary);
      vOverrideContentType = "multipart/form-data; boundary=" + vBoundary;
    } else {
      throw std::runtime_error("failed to parse multipart form fields JSON");
    }
  }

  // prepare the request to destination server
  httplib::Request vReq;
  {
    vReq.method = pMethod;
    vReq.path = pPath;
    for (const auto &[vKey, vValue] : pHeaders) {
      const auto cLowered = fToLowerCopy(vKey);
      if (cLowered == "accept-encoding") {
        // disable Accept-Encoding to avoid compressed responses
        continue;
      }
      if (cLowered == "transfer-encoding") {
        // the body is already decoded
        continue;
      }
      if (cLowered == "content-length") {
        // let httplib calculate Content-Length from the actual body
        continue;
      }
      if (cLowered == "content-type") {
        if (vHasFiles) {
          // we set our own Content-Type with the new boundary
          continue;
        }
        // when no files but the original request was multipart,
        // the body is now JSON, so correct the Content-Type
        if (vValue.find("multipart/form-data") != std::string::npos) {
          vOverrideContentType = "application/json; charset=utf-8";
          continue;
        }
      }
      if (cLowered == "host") {
        bool vIsDefaultPort = (pScheme == "https" && pPort == 443) || (pScheme == "http" && pPort == 80);
        const std::string cUrlHost = fCommonHttpFormatHost(pHost);
        vReq.set_header(vKey, vIsDefaultPort ? cUrlHost : cUrlHost + ":" + std::to_string(pPort));
      } else {
        vReq.set_header(vKey, vValue);
      }
    }
    vReq.body = vEffectiveBody;
    if (!vOverrideContentType.empty()) {
      vReq.set_header("Content-Type", vOverrideContentType);
    }
    vReq.response_handler = vResponseHandler;
    vReq.content_receiver = vContentReceiver;
  }

  // start the proxy thread
  SRV_DBG("start proxy thread %s %s\n", vReq.method.c_str(), vReq.path.c_str());
  this->vThread = std::thread([vCli, vPipe, vReq]() {
    auto vResult = vCli->send(std::move(vReq));
    if (vResult.error() != httplib::Error::Success) {
      auto vErrStr = httplib::to_string(vResult.error());
      SRV_ERR("http client error: %s\n", vErrStr.c_str());
      vPipe->mWrite({{}, 500, "", ""});                      // header
      vPipe->mWrite({{}, 0, "proxy error: " + vErrStr, ""}); // body
    }
    vPipe->mCloseWrite(); // signal EOF to reader
    SRV_DBG("%s", "client request thread ended\n");
  });
  this->vThread.detach();

  // wait for the first chunk (headers)
  {
    msg_t vHeader;
    if (vPipe->mRead(vHeader, pShouldStop)) {
      SRV_DBG("%s", "received response headers\n");
      this->vStatus = vHeader.vStatus;
      this->dHeaders = std::move(vHeader.dHeaders);
      if (!vHeader.vContentType.empty()) {
        this->vContentType = std::move(vHeader.vContentType);
      }
    } else {
      SRV_DBG("%s", "no response headers received (request cancelled?)\n");
    }
  }
}
