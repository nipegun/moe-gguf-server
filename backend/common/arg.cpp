#include "arg.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "download.h"
#include "json-schema-to-grammar.h"
#include "mgs.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "preset.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <list>
#include <regex>
#include <set>
#include <string>
#include <thread> // for hardware_concurrency
#include <vector>

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#if !defined(PATH_MAX)
#define PATH_MAX MAX_PATH
#endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

#define MGS_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

using json = nlohmann::ordered_json;
using namespace common_arg_utils;

static std::string fReadFile(const std::string &pFileName) {
  std::ifstream vFile(pFileName);
  if (!vFile) {
    throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pFileName.c_str()));
  }
  std::string vContent((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());
  vFile.close();
  return vContent;
}

static const std::vector<common_arg> &fGetCommonArgDefs() {
  static const std::vector<common_arg> lOptions = [] {
    common_params vParams;
    auto vCtx = fCommonParamsParserInit(vParams, nullptr);
    return vCtx.lOptions;
  }();
  return lOptions;
}

common_arg &common_arg::mSetEnv(const char *pEnv) {
  vHelp = vHelp + "\n(env: " + pEnv + ")";
  this->vEnv = pEnv;
  return *this;
}

common_arg &common_arg::mSetSampling() {
  vIsSampling = true;
  return *this;
}

common_arg &common_arg::mSetSpec() {
  vIsSpec = true;
  return *this;
}

common_arg &common_arg::mSetPresetOnly() {
  vIsPresetOnly = true;
  return *this;
}

bool common_arg::mGetValueFromEnv(std::string &pOutput) const {
  if (vEnv == nullptr)
    return false;
  char *vValue = std::getenv(vEnv);
  if (vValue) {
    pOutput = vValue;
    return true;
  }
  return false;
}

bool common_arg::mHasValueFromEnv() const { return vEnv != nullptr && std::getenv(vEnv); }

static std::vector<std::string> fBreakStrIntoLines(std::string pInput, size_t pMaxCharPerLine) {
  std::vector<std::string> lResult;
  std::istringstream vIss(pInput);
  std::string vLine;
  auto fAddLine = [&](const std::string &pL) {
    if (pL.length() <= pMaxCharPerLine) {
      lResult.push_back(pL);
    } else {
      std::istringstream vLineStream(pL);
      std::string vWord, vCurrentLine;
      while (vLineStream >> vWord) {
        if (vCurrentLine.length() + !vCurrentLine.empty() + vWord.length() > pMaxCharPerLine) {
          if (!vCurrentLine.empty())
            lResult.push_back(vCurrentLine);
          vCurrentLine = vWord;
        } else {
          vCurrentLine += (!vCurrentLine.empty() ? " " : "") + vWord;
        }
      }
      if (!vCurrentLine.empty())
        lResult.push_back(vCurrentLine);
    }
  };
  while (std::getline(vIss, vLine)) {
    fAddLine(vLine);
  }
  return lResult;
}

std::string common_arg::mToString() const {
  // params for printing to console
  const static int cNLeadingSpaces = 40;
  const static int cNCharPerLineHelp = 70; // TODO: detect this based on current console
  std::string vLeadingSpaces(cNLeadingSpaces, ' ');

  std::ostringstream vSs;
  auto vAllArgs = mGetArgs(); // also contains args_neg
  for (const auto &vArg : vAllArgs) {
    if (vArg == vAllArgs.front()) {
      if (vAllArgs.size() == 1) {
        vSs << vArg;
      } else {
        // first arg is usually abbreviation, we need padding to make it more beautiful
        auto vTmp = std::string(vArg) + ", ";
        auto vSpaces = std::string(std::max(0, 7 - (int)vTmp.size()), ' ');
        vSs << vTmp << vSpaces;
      }
    } else {
      vSs << vArg << (vArg != vAllArgs.back() ? ", " : "");
    }
  }
  if (vValueHint)
    vSs << " " << vValueHint;
  if (vValueHint2)
    vSs << " " << vValueHint2;
  if (vSs.tellp() > cNLeadingSpaces - 3) {
    // current line is too long, add new line
    vSs << "\n" << vLeadingSpaces;
  } else {
    // padding between arg and help, same line
    vSs << std::string(vLeadingSpaces.size() - vSs.tellp(), ' ');
  }
  const auto cHelpLines = fBreakStrIntoLines(vHelp, cNCharPerLineHelp);
  for (const auto &vLine : cHelpLines) {
    vSs << (&vLine == &cHelpLines.front() ? "" : vLeadingSpaces) << vLine << "\n";
  }
  return vSs.str();
}

std::vector<std::string> common_arg::mGetArgs() const {
  std::vector<std::string> lResult;
  for (const auto &vArg : lArgs) {
    lResult.push_back(std::string(vArg));
  }
  for (const auto &vArg : lArgsNeg) {
    lResult.push_back(std::string(vArg));
  }
  return lResult;
}

std::vector<std::string> common_arg::mGetEnv() const {
  std::vector<std::string> lResult;
  if (vEnv) {
    lResult.push_back(std::string(vEnv));
  }
  return lResult;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void fParseTensorBufferOverrides(const std::string &pValue,
                                        std::vector<mgs_model_tensor_buft_override> &pOverrides) {
  ggml_backend_load_all();

  std::map<std::string, ggml_backend_buffer_type_t> dBuftList;
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    auto *vDev = ggml_backend_dev_get(vI);
    auto *vBuft = ggml_backend_dev_buffer_type(vDev);
    if (vBuft) {
      dBuftList[ggml_backend_buft_name(vBuft)] = vBuft;
    }
  }

  for (const auto &vOverride : fStringSplit<std::string>(pValue, ',')) {
    std::string::size_type vPos = vOverride.find('=');
    if (vPos == std::string::npos) {
      throw std::invalid_argument("invalid value");
    }
    std::string vTensorName = vOverride.substr(0, vPos);
    std::string vBufferType = vOverride.substr(vPos + 1);

    if (dBuftList.find(vBufferType) == dBuftList.end()) {
      printf("Available buffer types:\n");
      for (const auto &vIt : dBuftList) {
        printf("  %s\n", ggml_backend_buft_name(vIt.second));
      }
      throw std::invalid_argument("unknown buffer type");
    }
    // keep strings alive and avoid leaking memory by storing them in a static vector
    static std::list<std::string> lBuftOverrides;
    lBuftOverrides.push_back(vTensorName);
    pOverrides.push_back({lBuftOverrides.back().c_str(), dBuftList.at(vBufferType)});
  }
}

static std::string fCleanFileName(const std::string &pFname) {
  std::string vCleanFname = pFname;
  fStringReplaceAll(vCleanFname, "\\", "_");
  fStringReplaceAll(vCleanFname, "/", "_");
  return vCleanFname;
}

struct handle_model_result {
  bool vFoundMmproj = false;
  common_params_model vMmproj;

  bool vFoundMtp = false;
  common_params_model vMtp;

  bool vFoundPreset = false;
  std::string vPresetPath;
};

const std::vector<ggml_type> lKvCacheTypes = {
    GGML_TYPE_F32,  GGML_TYPE_F16,    GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
};

static ggml_type fKvCacheTypeFromStr(const std::string &pS) {
  for (const auto &vType : lKvCacheTypes) {
    if (ggml_type_name(vType) == pS) {
      return vType;
    }
  }
  throw std::runtime_error("Unsupported cache type: " + pS);
}

static std::string fGetAllKvCacheTypes() {
  std::ostringstream vMsg;
  for (const auto &vType : lKvCacheTypes) {
    vMsg << ggml_type_name(vType) << (&vType == &lKvCacheTypes.back() ? "" : ", ");
  }
  return vMsg.str();
}

static bool fParseBoolValue(const std::string &pValue) {
  if (fIsTruthy(pValue)) {
    return true;
  } else if (fIsFalsey(pValue)) {
    return false;
  } else {
    throw std::invalid_argument("invalid boolean value");
  }
}

//
// common_models_handler
//

static std::string fGetDefaultLocalPath(const std::string &pUrl) {
  auto vF = fStringSplit<std::string>(pUrl, '#').front();
  vF = fStringSplit<std::string>(vF, '?').front();
  return fFsGetCacheFile(fStringSplit<std::string>(vF, '/').back());
}

static bool fSpecTypesIsDefault(const common_params &pParams) {
  return pParams.vSpeculative.lTypes == std::vector<enum common_speculative_type>{cCommonSpeculativeTypeNone};
}

common_models_handler fCommonModelsHandlerInit(const common_params &pParams) {
  common_download_hf_plan vPlan;
  common_download_hf_plan vPlanSpec;
  common_download_opts vOpts;

  const bool cSpecTypeDraftMtp = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                           cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

  vOpts.vBearerToken = pParams.vHfToken;
  vOpts.vOffline = pParams.vOffline;
  vOpts.vDownloadMtp = cSpecTypeDraftMtp;
  vOpts.vDownloadMmproj = !pParams.vNoMmproj && pParams.vMmproj.vPath.empty() && pParams.vMmproj.vUrl.empty();

  if (!pParams.vModel.vHfRepo.empty()) {
    vPlan = fCommonDownloadGetHfPlan(pParams.vModel, vOpts);
  }

  if (!pParams.vSpeculative.vDraft.vMparams.vHfRepo.empty()) {
    // without a requested type, discover the MTP sidecar the draft repo may ship to infer the type later
    auto vOptsSpec = vOpts;
    if (fSpecTypesIsDefault(pParams)) {
      vOptsSpec.vDownloadMtp = true;
    }
    vPlanSpec = fCommonDownloadGetHfPlan(pParams.vSpeculative.vDraft.vMparams, vOptsSpec);
  }

  return common_models_handler{vPlan, vPlanSpec, vOpts};
}

bool fCommonModelsHandlerIsPresetRepo(const common_models_handler &pHandler) {
  return !pHandler.vPlan.vPreset.vUrl.empty();
}

// the Hugging Face token only authenticates against the Hugging Face endpoint (MGS_MODEL_ENDPOINT): a plain URL on
// any other origin is downloaded without it, so a model host can not collect it. The endpoint ends with '/', so the
// prefix covers its whole authority (a longer host name or a user name in the URL does not match)
static common_download_opts fUrlDownloadOpts(const std::string &pUrl, common_download_opts pOpts) {
  if (pUrl.rfind(fCommonGetModelEndpoint(), 0) != 0) {
    pOpts.vBearerToken.clear();
  }
  return pOpts;
}

static std::vector<common_download_task> fBuildUrlTasks(const common_params_model &pModel, common_download_opts pOpts) {
  auto vParts = fCommonDownloadGetAllParts(pModel.vUrl);
  std::vector<common_download_task> lTasks;

  // single-part: download straight to model.path if the user gave one (-m), else the cache default
  if (vParts.size() == 1) {
    common_download_task vTask;
    vTask.vUrl = vParts[0];
    vTask.vLocalPath = pModel.vPath.empty() ? fGetDefaultLocalPath(vParts[0]) : pModel.vPath;
    vTask.vOpts = fUrlDownloadOpts(vTask.vUrl, pOpts);
    lTasks.push_back(std::move(vTask));
    return lTasks;
  }

  // multi-part: place each part under the user's -m directory (if given), else the cache default
  std::string vBaseDir;
  if (!pModel.vPath.empty()) {
    auto vPos = pModel.vPath.rfind('/');
    vBaseDir = vPos == std::string::npos ? std::string(".") : pModel.vPath.substr(0, vPos);
  }

  for (const auto &vPart : vParts) {
    common_download_task vTask;
    vTask.vUrl = vPart;
    vTask.vOpts = fUrlDownloadOpts(vTask.vUrl, pOpts);

    std::string vLocal = fGetDefaultLocalPath(vPart);
    if (!vBaseDir.empty()) {
      auto vPos = vLocal.rfind('/');
      std::string vName = vPos == std::string::npos ? vLocal : vLocal.substr(vPos + 1);
      vLocal = vBaseDir + "/" + vName;
    }
    vTask.vLocalPath = vLocal;
    lTasks.push_back(std::move(vTask));
  }
  return lTasks;
}

void fCommonModelsHandlerApply(common_models_handler &pHandler, common_params &pParams,
                               common_download_callback *pCallback) {
  std::vector<common_download_task> lTasks;

  auto &vPlan = pHandler.vPlan;
  auto &vPlanSpec = pHandler.vPlanSpec;

  auto vOpts = pHandler.vOpts; // copy
  vOpts.vCallback = pCallback;

  // handle plain "url" if needed
  auto fHandleUrl = [&](common_params_model &pModel) {
    if (!pModel.vUrl.empty()) {
      if (pModel.vPath.empty()) {
        pModel.vPath = fGetDefaultLocalPath(pModel.vUrl);
      }
    }
  };
  fHandleUrl(pParams.vModel);
  fHandleUrl(pParams.vMmproj);
  fHandleUrl(pParams.vSpeculative.vDraft.vMparams);

  // optionally, if docker repo is set, resolve it
  if (!pParams.vModel.vDockerRepo.empty()) {
    pParams.vModel.vUrl = fCommonDockerResolveModel(pParams.vModel.vDockerRepo);
    pParams.vModel.vPath = fGetDefaultLocalPath(pParams.vModel.vUrl);
  }

  // handle plain "url" tasks (non-hf)
  if (!pParams.vModel.vUrl.empty()) {
    auto vUrlTasks = fBuildUrlTasks(pParams.vModel, vOpts);
    // the first part is what gets loaded, so point params.model.path at it
    if (!vUrlTasks.empty()) {
      std::string vFirstPath = vUrlTasks.front().vLocalPath;
      vUrlTasks.front().vOnDone = [&, vFirstPath]() { pParams.vModel.vPath = vFirstPath; };
    }
    for (auto &vTask : vUrlTasks) {
      lTasks.push_back(std::move(vTask));
    }
  }
  if (!pParams.vMmproj.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl = pParams.vMmproj.vUrl;
    vTask.vLocalPath = pParams.vMmproj.vPath;
    vTask.vOpts = fUrlDownloadOpts(vTask.vUrl, vOpts);
    lTasks.push_back(vTask);
  }

  bool vHadSpecUrl = false;
  if (!pParams.vSpeculative.vDraft.vMparams.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl = pParams.vSpeculative.vDraft.vMparams.vUrl;
    vTask.vLocalPath = pParams.vSpeculative.vDraft.vMparams.vPath;
    vTask.vOpts = fUrlDownloadOpts(vTask.vUrl, vOpts);
    lTasks.push_back(vTask);
    vHadSpecUrl = true;
  }

  // handle hf_plan tasks
  auto fAddTasks = [&vOpts, &lTasks](const hf_cache::hf_files &pModelFiles, const hf_cache::vHfFile &pPrimary,
                                     common_params_model &pModel) {
    for (size_t vI = 0; vI < pModelFiles.size(); ++vI) {
      auto &vModelFile = pModelFiles[vI];
      bool vIsPrimary = (vModelFile.vPath == pPrimary.vPath);
      lTasks.emplace_back(vModelFile, vOpts, [&, vIsPrimary]() {
        if (vIsPrimary) {
          // the primary file is the first split (00001-of), use it as model path
          pModel.vPath = hf_cache::fFinalizeFile(vModelFile);
        } else {
          hf_cache::fFinalizeFile(vModelFile);
        }
      });
    }
  };

  // an explicit draft file selection (e.g. --spec-draft-model with --spec-draft-hf) disables the sidecar resolution
  // of the draft repo
  if (!pParams.vSpeculative.vDraft.vMparams.vHfFile.empty()) {
    vPlanSpec.vMtp = {};
  }

  // infer the speculative type from the sidecar shipped by the draft repo when none is requested
  if (fSpecTypesIsDefault(pParams) && !vPlanSpec.vMtp.vLocalPath.empty()) {
    pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftMtp};
  }

  // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
  const bool cSpecSidecarFound = !vPlanSpec.vMtp.vLocalPath.empty();
  if (!vPlanSpec.vMtp.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vMtp, vOpts, [&]() {
      // only use the discovered MTP head when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vMtp);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vMtp);
      }
    });
  }

  // a wired draft sidecar counts as an explicit draft for the main plan fallback below
  if (cSpecSidecarFound) {
    vHadSpecUrl = true;
  }

  // handle plan_spec (e.g. --spec-draft-hf)
  if (!vPlanSpec.vModelFiles.empty() && !vHadSpecUrl && !cSpecSidecarFound) {
    fAddTasks(vPlanSpec.vModelFiles, vPlanSpec.vPrimary, pParams.vSpeculative.vDraft.vMparams);
    vHadSpecUrl = true;
  }

  if (!vPlan.vModelFiles.empty()) {
    fAddTasks(vPlan.vModelFiles, vPlan.vPrimary, pParams.vModel);
  }
  if (!vPlan.vMmproj.vLocalPath.empty()) {
    lTasks.emplace_back(vPlan.vMmproj, vOpts,
                        [&]() { pParams.vMmproj.vPath = hf_cache::fFinalizeFile(vPlan.vMmproj); });
  }
  if (!vPlan.vMtp.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vMtp, vOpts, [&]() {
      // only fall back to the discovered MTP head when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vMtp);
      } else {
        hf_cache::fFinalizeFile(vPlan.vMtp);
      }
    });
  }
  if (!vPlan.vPreset.vLocalPath.empty()) {
    lTasks.emplace_back(vPlan.vPreset, vOpts, [&]() {
      // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
      pParams.vModelsPresetHf = pParams.vModel.vHfRepo; // only for showing a warning
      pParams.vModelsPreset = hf_cache::fFinalizeFile(vPlan.vPreset);
      pParams.vModel = common_params_model{}; // make sure to clear model, so server starts in router mode
    });
  }

  // run all tasks in parallel
  if (!pParams.vOffline) {
    // if duplicated files are found, only download once (but still call on_done for each task)
    std::unordered_map<std::string, common_download_task *> dUniqueTasks;
    for (auto &vTask : lTasks) {
      auto vIt = dUniqueTasks.find(vTask.vLocalPath);
      if (vIt == dUniqueTasks.end()) {
        dUniqueTasks[vTask.vLocalPath] = &vTask;
      }
    }
    std::vector<common_download_task> lUniqueTasksVec;
    for (auto &vPair : dUniqueTasks) {
      LOG_DBG("download task: %s -> %s\n", vPair.second->vUrl.c_str(), vPair.second->vLocalPath.c_str());
      lUniqueTasksVec.push_back(*vPair.second);
    }
    fCommonDownloadRunTasks(lUniqueTasksVec);
  }

  // download successful, update params with the downloaded paths
  for (const auto &vTask : lTasks) {
    if (vTask.vOnDone) {
      vTask.vOnDone();
    }
  }
}

//
// CLI argument parsing functions
//

static bool fCommonParamsParseEx(int pArgc, char **pArgv, common_params_context &pCtxArg) {
  common_params &vParams = pCtxArg.vParams;

  // setup log directly from vParams.vVerbosity
  fCommonLogSetVerbosityThold(vParams.vVerbosity);

  std::unordered_map<std::string, std::pair<common_arg *, bool>> dArgToOptions;
  for (auto &vOpt : pCtxArg.lOptions) {
    for (const auto &vArg : vOpt.lArgs) {
      dArgToOptions[vArg] = {&vOpt, /* is_positive */ true};
    }
    for (const auto &vArg : vOpt.lArgsNeg) {
      dArgToOptions[vArg] = {&vOpt, /* is_positive */ false};
    }
  }

  // handle environment variables
  for (auto &vOpt : pCtxArg.lOptions) {
    std::string vValue;
    if (vOpt.mGetValueFromEnv(vValue)) {
      try {
        if (vOpt.vHandlerVoid && fIsTruthy(vValue)) {
          vOpt.vHandlerVoid(vParams);
        }
        if (vOpt.vHandlerInt) {
          vOpt.vHandlerInt(vParams, std::stoi(vValue));
        }
        if (vOpt.vHandlerBool) {
          vOpt.vHandlerBool(vParams, fParseBoolValue(vValue));
        }
        if (vOpt.vHandlerString) {
          vOpt.vHandlerString(vParams, vValue);
          continue;
        }
      } catch (std::exception &vE) {
        throw std::invalid_argument(
            fStringFormat("error while handling environment variable \"%s\": %s\n\n", vOpt.vEnv, vE.what()));
      }
    }
  }

  // handle command line arguments
  auto fCheckArg = [&](int pI) {
    if (pI + 1 >= pArgc) {
      throw std::invalid_argument("expected value for argument");
    }
  };

  auto fParseCliArgs = [&]() {
    std::set<std::string> sSeenArgs;

    for (int vI = 1; vI < pArgc; vI++) {
      const std::string cArg = pArgv[vI];
      if (dArgToOptions.find(cArg) == dArgToOptions.end()) {
        throw std::invalid_argument(fStringFormat("error: invalid argument: %s", cArg.c_str()));
      }
      if (!sSeenArgs.insert(cArg).second) {
        const bool cSkip = (cArg == "--spec-type");

        if (!cSkip) {
          throw std::invalid_argument(fStringFormat(
              "argument '%s' may only be specified once; use comma-separated values for lists", cArg.c_str()));
        }
      }
      auto &vTmp = dArgToOptions[cArg];
      auto vOpt = *vTmp.first;
      bool vIsPositive = vTmp.second;
      if (vOpt.mHasValueFromEnv()) {
        fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n",
                vOpt.vEnv, cArg.c_str());
      }
      try {
        if (vOpt.vHandlerVoid) {
          vOpt.vHandlerVoid(vParams);
          continue;
        }
        if (vOpt.vHandlerBool) {
          vOpt.vHandlerBool(vParams, vIsPositive);
          continue;
        }

        // arg with single value
        fCheckArg(vI);
        std::string vVal = pArgv[++vI];
        if (vOpt.vHandlerInt) {
          vOpt.vHandlerInt(vParams, std::stoi(vVal));
          continue;
        }
        if (vOpt.vHandlerString) {
          vOpt.vHandlerString(vParams, vVal);
          continue;
        }

        // arg with 2 values
        fCheckArg(vI);
        std::string vVal2 = pArgv[++vI];
        if (vOpt.vHandlerStrStr) {
          vOpt.vHandlerStrStr(vParams, vVal, vVal2);
          continue;
        }
      } catch (std::exception &vE) {
        throw std::invalid_argument(fStringFormat("error while handling argument \"%s\": %s\n\n"
                                                  "usage:\n%s\n\nto show complete usage, run with -h",
                                                  cArg.c_str(), vE.what(), vOpt.mToString().c_str()));
      }
    }
  };

  // parse all CLI args now, so that -hf is available below for remote preset resolution
  fParseCliArgs();

  fPostprocessCpuParams(vParams.vCpuparams, nullptr);
  fPostprocessCpuParams(vParams.vCpuparamsBatch, &vParams.vCpuparams);

  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparams, &vParams.vCpuparams);
  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparamsBatch, &vParams.vCpuparamsBatch);

  // the server resolves and downloads the models itself (fCommonModelsHandlerInit) after parsing

  if (vParams.vEscape) {
    for (auto &vSeqBreaker : vParams.vSampling.lDrySequenceBreakers) {
      fStringProcessEscapes(vSeqBreaker);
    }
  }

  if (!vParams.lKvOverrides.empty()) {
    vParams.lKvOverrides.emplace_back();
    vParams.lKvOverrides.back().key[0] = 0;
  }

  // pad tensor_buft_overrides for mgs_params_fit:
  const size_t cNtbo = mgs_max_tensor_buft_overrides();
  while (vParams.lTensorBuftOverrides.size() < cNtbo) {
    vParams.lTensorBuftOverrides.push_back({nullptr, nullptr});
  }

  if (!vParams.vSpeculative.vDraft.lTensorBuftOverrides.empty()) {
    vParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back({nullptr, nullptr});
  }

  if (!vParams.vChatTemplate.empty() && !fCommonChatVerifyTemplate(vParams.vChatTemplate)) {
    throw std::runtime_error(
        fStringFormat("error: the supplied chat template is not supported: %s\n", vParams.vChatTemplate.c_str()));
  }

  return true;
}

static void fCommonParamsPrintUsage(common_params_context &pCtxArg) {
  auto fPrintOptions = [](std::vector<common_arg *> &pOptions) {
    for (common_arg *vOpt : pOptions) {
      printf("%s", vOpt->mToString().c_str());
    }
  };

  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;
  for (auto &vOpt : pCtxArg.lOptions) {
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else {
      lCommonOptions.push_back(&vOpt);
    }
  }
  bool vFirst = true;
  auto fPrintSection = [&](const char *pHeader, std::vector<common_arg *> &pOptions) {
    if (pOptions.empty()) {
      return;
    }
    printf("%s----- %s -----\n\n", vFirst ? "" : "\n\n", pHeader);
    vFirst = false;
    fPrintOptions(pOptions);
  };
  fPrintSection("general params", lCommonOptions);
  fPrintSection("sampling params", lSamplingOptions);
  fPrintSection("speculative params", lSpecOptions);
}

static void fCommonParamsPrintCompletion(common_params_context &pCtxArg) {
  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;

  for (auto &vOpt : pCtxArg.lOptions) {
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else {
      lCommonOptions.push_back(&vOpt);
    }
  }

  printf("_mgs_completions() {\n");
  printf("    local cur prev opts\n");
  printf("    COMPREPLY=()\n");
  printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
  printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

  printf("    opts=\"");
  auto fPrintOptions = [](const std::vector<common_arg *> &pOptions) {
    for (const common_arg *vOpt : pOptions) {
      for (const char *vArg : vOpt->lArgs) {
        printf("%s ", vArg);
      }
    }
  };

  fPrintOptions(lCommonOptions);
  fPrintOptions(lSamplingOptions);
  fPrintOptions(lSpecOptions);
  printf("\"\n\n");

  printf("    case \"$prev\" in\n");
  printf("        --model|-m)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        --grammar-file)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        --chat-template-file)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        *)\n");
  printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("    esac\n");
  printf("}\n\n");

  std::set<std::string> sExecutables = {"mgs"};

  for (const auto &vExe : sExecutables) {
    printf("complete -F _mgs_completions %s\n", vExe.c_str());
  }
}

static std::vector<ggml_backend_dev_t> fParseDeviceList(const std::string &pValue) {
  std::vector<ggml_backend_dev_t> lDevices;
  auto vDevNames = fStringSplit<std::string>(pValue, ',');
  if (vDevNames.empty()) {
    throw std::invalid_argument("no devices specified");
  }
  if (vDevNames.size() == 1 && vDevNames[0] == "none") {
    lDevices.push_back(nullptr);
  } else {
    ggml_backend_load_all();
    for (const auto &vDevice : vDevNames) {
      auto *vDev = ggml_backend_dev_by_name(vDevice.c_str());
      if (!vDev || ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        throw std::invalid_argument(fStringFormat("invalid device: %s", vDevice.c_str()));
      }
      lDevices.push_back(vDev);
    }
    lDevices.push_back(nullptr);
  }
  return lDevices;
}

void fCommonPrintAvailableDevices() {
  constexpr size_t cMiB = 1024 * 1024;
  std::vector<ggml_backend_dev_t> lDevices;

  ggml_backend_load_all();

  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    auto *vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
      lDevices.push_back(vDev);
    }
  }
  printf("Available devices:\n");

  if (lDevices.empty()) {
    printf("  (none)\n");
    return;
  }
  for (auto *vDev : lDevices) {
    size_t vFree, vTotal;
    ggml_backend_dev_memory(vDev, &vFree, &vTotal);
    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
           vTotal / cMiB, vFree / cMiB);
  }
}

bool fCommonParamsToMap(int pArgc, char **pArgv, std::map<common_arg, std::string> &pOutMap) {
  common_params vDummyParams;
  common_params_context vCtxArg = fCommonParamsParserInit(vDummyParams, nullptr);

  std::unordered_map<std::string, common_arg *> dArgToOptions;
  for (auto &vOpt : vCtxArg.lOptions) {
    for (const auto &vArg : vOpt.lArgs) {
      dArgToOptions[vArg] = &vOpt;
    }
    for (const auto &vArg : vOpt.lArgsNeg) {
      dArgToOptions[vArg] = &vOpt;
    }
  }

  // TODO @ngxson : find a way to deduplicate this code

  // handle command line arguments
  auto fCheckArg = [&](int pI) {
    if (pI + 1 >= pArgc) {
      throw std::invalid_argument("expected value for argument");
    }
  };

  std::set<std::string> sSeenArgs;

  for (int vI = 1; vI < pArgc; vI++) {
    const std::string cArg = pArgv[vI];
    if (dArgToOptions.find(cArg) == dArgToOptions.end()) {
      throw std::invalid_argument(fStringFormat("error: invalid argument: %s", cArg.c_str()));
    }
    if (!sSeenArgs.insert(cArg).second) {
      const bool cSkip = (cArg == "--spec-type");

      if (!cSkip) {
        throw std::invalid_argument(fStringFormat(
            "argument '%s' may only be specified once; use comma-separated values for lists", cArg.c_str()));
      }
    }
    auto vOpt = *dArgToOptions[cArg];
    std::string vVal;
    if (vOpt.vValueHint == nullptr && vOpt.vValueHint2 == nullptr) {
      // bool arg (need to reverse the meaning for negative args)
      bool vIsNeg = std::find(vOpt.lArgsNeg.begin(), vOpt.lArgsNeg.end(), cArg) != vOpt.lArgsNeg.end();
      vVal = vIsNeg ? "0" : "1";
    }
    if (vOpt.vValueHint != nullptr) {
      // arg with single value
      fCheckArg(vI);
      vVal = pArgv[++vI];
    }
    if (vOpt.vValueHint2 != nullptr) {
      // TODO: support arg with 2 values
      throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
    }
    pOutMap[vOpt] = vVal;
  }

  return true;
}

#ifdef _WIN32
struct utf8_argv {
  std::vector<std::string> lBuf;
  std::vector<char *> lPtrs;
};

static utf8_argv fMakeUtf8Argv() {
  utf8_argv vOut;
  int vWargc = 0;
  LPWSTR *vWargv = CommandLineToArgvW(GetCommandLineW(), &vWargc);
  if (!vWargv)
    return vOut;

  vOut.lBuf.reserve(vWargc);
  for (int vI = 0; vI < vWargc; ++vI) {
    int vN = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, vWargv[vI], -1, nullptr, 0, nullptr, nullptr);
    if (vN <= 0) {
      vOut.lBuf.emplace_back();
      continue;
    }
    auto &vS = vOut.lBuf.emplace_back();
    vS.resize(static_cast<size_t>(vN - 1));
    (void)WideCharToMultiByte(CP_UTF8, 0, vWargv[vI], -1, vS.data(), vN, nullptr, nullptr);
  }
  LocalFree(vWargv);

  vOut.lPtrs.reserve(vOut.lBuf.size() + 1);
  for (auto &vS : vOut.lBuf)
    vOut.lPtrs.push_back(vS.data());
  vOut.lPtrs.push_back(nullptr);
  return vOut;
}
#endif

bool fCommonParamsParse(int pArgc, char **pArgv, common_params &pParams, void (*pPrintUsage)(int, char **)) {
#ifdef _WIN32
  auto vUtf8 = fMakeUtf8Argv();
  // repair argv only when it matches the process command line
  if (static_cast<int>(vUtf8.lBuf.size()) == pArgc) {
    pArgv = vUtf8.lPtrs.data();
  }
#endif

  auto vCtxArg = fCommonParamsParserInit(pParams, pPrintUsage);
  const common_params cParamsOrg = vCtxArg.vParams;

  try {
    if (!fCommonParamsParseEx(pArgc, pArgv, vCtxArg)) {
      vCtxArg.vParams = cParamsOrg;
      return false;
    }
    if (vCtxArg.vParams.vUsage) {
      fCommonParamsPrintUsage(vCtxArg);
      if (vCtxArg.vPrintUsage) {
        vCtxArg.vPrintUsage(pArgc, pArgv);
      }
      fCommonLogFlush(fCommonLogMain());
      exit(0);
    }
    if (vCtxArg.vParams.vCompletion) {
      fCommonParamsPrintCompletion(vCtxArg);
      exit(0);
    }
  } catch (const std::invalid_argument &vEx) {
    fprintf(stderr, "%s\n", vEx.what());
    vCtxArg.vParams = cParamsOrg;
    return false;
  } catch (std::exception &vEx) {
    fprintf(stderr, "%s\n", vEx.what());
    exit(1); // for other exceptions, we exit with status code 1
  }

  return true;
}

bool common_arg_utils::fIsTruthy(const std::string &pValue) {
  return pValue == "on" || pValue == "enabled" || pValue == "true" || pValue == "1";
}

bool common_arg_utils::fIsFalsey(const std::string &pValue) {
  return pValue == "off" || pValue == "disabled" || pValue == "false" || pValue == "0";
}

bool common_arg_utils::fIsAutoy(const std::string &pValue) { return pValue == "auto" || pValue == "-1"; }

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
static std::vector<std::string> fParseCsvRow(const std::string &pInput) {
  std::vector<std::string> lFields;
  std::string vField;
  bool vInQuotes = false;

  for (size_t vI = 0; vI < pInput.length(); ++vI) {
    char vCh = pInput[vI];

    if (vCh == '"') {
      if (!vInQuotes) {
        // start of quoted field (only valid if at beginning of field)
        if (!vField.empty()) {
          // quote appeared in middle of unquoted field, treat as literal
          vField += '"';
        } else {
          vInQuotes = true; // start
        }
      } else {
        if (vI + 1 < pInput.length() && pInput[vI + 1] == '"') {
          // escaped quote: ""
          vField += '"';
          ++vI; // skip the next quote
        } else {
          vInQuotes = false; // end
        }
      }
    } else if (vCh == ',') {
      if (vInQuotes) {
        vField += ',';
      } else {
        lFields.push_back(std::move(vField));
        vField.clear();
      }
    } else {
      vField += vCh;
    }
  }

  // Add the last field
  lFields.push_back(std::move(vField));

  return lFields;
}

common_params_context fCommonParamsParserInit(common_params &pParams, void (*pPrintUsage)(int, char **)) {
  common_params_context vCtxArg(pParams);
  vCtxArg.vPrintUsage = pPrintUsage;

  std::string vSamplerTypeChars;
  std::string vSamplerTypeNames;
  for (const auto &vSampler : pParams.vSampling.lSamplers) {
    vSamplerTypeChars += fCommonSamplerTypeToChr(vSampler);
    vSamplerTypeNames += fCommonSamplerTypeToStr(vSampler) + ";";
  }
  if (!vSamplerTypeNames.empty()) {
    vSamplerTypeNames.pop_back(); // remove last semicolon
  }

  auto fAddOpt = [&](common_arg pArg) { vCtxArg.lOptions.push_back(std::move(pArg)); };

  fAddOpt(common_arg({"-h", "--help"}, "print usage and exit", [](common_params &pParams) {
            pParams.vUsage = true;
          }));
  fAddOpt(common_arg({"--version"}, "show version and build info", [](common_params &) {
    fprintf(stderr, "version: %s\n", mgs_version());
    fprintf(stderr, "built with %s for %s\n", mgs_compiler(), mgs_build_target());
    exit(0);
  }));
  fAddOpt(common_arg({"-cl", "--cache-list"}, "show list of models in cache", [](common_params &) {
    auto vModels = fCommonListCachedModels();
    printf("number of models in cache: %zu\n", vModels.size());
    for (size_t vI = 0; vI < vModels.size(); vI++) {
      printf("%4zu. %s\n", vI + 1, vModels[vI].mToString().c_str());
    }
    exit(0);
  }));
  fAddOpt(common_arg({"--completion-bash"}, "print source-able bash completion script for mgs",
                     [](common_params &pParams) { pParams.vCompletion = true; }));
  fAddOpt(
      common_arg({"-t", "--threads"}, "N",
                 "number of CPU threads to use during generation (default: the cores useful for math, at most half of "
                 "the logical CPUs)",
                 [](common_params &pParams, int pValue) {
                   pParams.vCpuparams.vNThreads = pValue;
                   if (pParams.vCpuparams.vNThreads <= 0) {
                     pParams.vCpuparams.vNThreads = std::thread::hardware_concurrency();
                   }
                 })
          .mSetEnv("MGS_ARG_THREADS"));
  fAddOpt(common_arg({"-tb", "--threads-batch"}, "N",
                     "number of threads to use during batch and prompt processing (default: same as --threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vCpuparamsBatch.vNThreads = pValue;
                       if (pParams.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     }));
  fAddOpt(common_arg({"-C", "--cpu-mask"}, "M",
                     "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     }));
  fAddOpt(common_arg({"-Cr", "--cpu-range"}, "lo-hi", "range of CPUs for affinity. Complements --cpu-mask",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  fAddOpt(common_arg(
      {"--cpu-strict"}, "<0|1>",
      fStringFormat("use strict CPU placement (default: %u)\n", (unsigned)pParams.vCpuparams.vStrictCpu),
      [](common_params &pParams, const std::string &pValue) { pParams.vCpuparams.vStrictCpu = std::stoul(pValue); }));
  fAddOpt(common_arg(
      {"--prio"}, "N",
      fStringFormat("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n",
                    pParams.vCpuparams.vPriority),
      [](common_params &pParams, int pPrio) {
        if (pPrio < GGML_SCHED_PRIO_LOW || pPrio > GGML_SCHED_PRIO_REALTIME) {
          throw std::invalid_argument("invalid value");
        }
        pParams.vCpuparams.vPriority = (enum ggml_sched_priority)pPrio;
      }));
  fAddOpt(common_arg(
      {"--poll"}, "<0...100>",
      fStringFormat("use polling level to wait for work (0 - no polling, default: %u)\n",
                    (unsigned)pParams.vCpuparams.vPoll),
      [](common_params &pParams, const std::string &pValue) { pParams.vCpuparams.vPoll = std::stoul(pValue); }));
  fAddOpt(
      common_arg({"-Cb", "--cpu-mask-batch"}, "M",
                 "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
                 [](common_params &pParams, const std::string &pMask) {
                   pParams.vCpuparamsBatch.vMaskValid = true;
                   if (!fParseCpuMask(pMask, pParams.vCpuparamsBatch.aCpumask)) {
                     throw std::invalid_argument("invalid cpumask");
                   }
                 }));
  fAddOpt(common_arg({"-Crb", "--cpu-range-batch"}, "lo-hi",
                     "ranges of CPUs for affinity. Complements --cpu-mask-batch",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  fAddOpt(common_arg({"--cpu-strict-batch"}, "<0|1>", "use strict CPU placement (default: same as --cpu-strict)",
                     [](common_params &pParams, int pValue) { pParams.vCpuparamsBatch.vStrictCpu = pValue; }));
  fAddOpt(
      common_arg({"--prio-batch"}, "N",
                 fStringFormat("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                               pParams.vCpuparamsBatch.vPriority),
                 [](common_params &pParams, int pPrio) {
                   if (pPrio < 0 || pPrio > 3) {
                     throw std::invalid_argument("invalid value");
                   }
                   pParams.vCpuparamsBatch.vPriority = (enum ggml_sched_priority)pPrio;
                 }));
  fAddOpt(common_arg({"--poll-batch"}, "<0|1>", "use polling to wait for work (default: same as --poll)",
                     [](common_params &pParams, int pValue) { pParams.vCpuparamsBatch.vPoll = pValue; }));
  fAddOpt(common_arg({"-lcs", "--lookup-cache-static"}, "FNAME",
                     "path to static lookup cache to use for lookup decoding (not updated by generation)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheStatic = pValue;
                     }));
  fAddOpt(common_arg({"-lcd", "--lookup-cache-dynamic"}, "FNAME",
                     "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheDynamic = pValue;
                     }));
  fAddOpt(common_arg({"-c", "--ctx-size"}, "N",
                     fStringFormat("size of the prompt context (default: %d, 0 = loaded from model)", pParams.vNCtx),
                     [](common_params &pParams, int pValue) {
                       pParams.vNCtx = pValue;
                       if (pValue == 0) {
                         // disable context reduction in mgs_params_fit if the user explicitly requests the full
                         // context size:
                         pParams.vFitParamsMinCtx = UINT32_MAX;
                       }
                     })
              .mSetEnv("MGS_ARG_CTX_SIZE"));
  fAddOpt(common_arg(
              {"-n", "--predict"}, "N",
              fStringFormat("number of tokens to predict (default: %d, -1 = infinity)", pParams.vNPredict),
              [](common_params &pParams, int pValue) { pParams.vNPredict = pValue; })
              .mSetEnv("MGS_ARG_PREDICT"));
  fAddOpt(common_arg({"-b", "--batch-size"}, "N",
                     fStringFormat("logical maximum batch size (default: %d; --fit raises it with the physical batch "
                                   "size when the MoE experts stay in system memory)",
                                   pParams.vNBatch),
                     [](common_params &pParams, int pValue) {
                       pParams.vNBatch = pValue;
                       pParams.vBatchExplicit = true;
                     })
              .mSetEnv("MGS_ARG_BATCH_SIZE"));
  fAddOpt(common_arg({"-ub", "--ubatch-size"}, "N",
                     fStringFormat("physical maximum batch size (default: %d; --fit raises it up to 4096 when the MoE "
                                   "experts stay in system memory)",
                                   pParams.vNUbatch),
                     [](common_params &pParams, int pValue) {
                       pParams.vNUbatch = pValue;
                       pParams.vBatchExplicit = true;
                     })
              .mSetEnv("MGS_ARG_UBATCH_SIZE"));
  fAddOpt(common_arg(
      {"--keep"}, "N",
      fStringFormat("number of tokens to keep from the initial prompt (default: %d, -1 = all)", pParams.vNKeep),
      [](common_params &pParams, int pValue) { pParams.vNKeep = pValue; }));
  fAddOpt(common_arg({"--swa-full"},
                     fStringFormat("use full-size SWA cache (default: %s)", pParams.vSwaFull ? "true" : "false"),
                     [](common_params &pParams) { pParams.vSwaFull = true; })
              .mSetEnv("MGS_ARG_SWA_FULL"));
  fAddOpt(common_arg({"-ctxcp", "--ctx-checkpoints"}, "N",
                     fStringFormat("max number of context checkpoints to create per slot (default: %d)",
                                   pParams.vNCtxCheckpoints),
                     [](common_params &pParams, int pValue) { pParams.vNCtxCheckpoints = pValue; })
              .mSetEnv("MGS_ARG_CTX_CHECKPOINTS"));
  fAddOpt(
      common_arg({"-cms", "--checkpoint-min-step"}, "N",
                 fStringFormat("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)",
                               pParams.vCheckpointMinStep),
                 [](common_params &pParams, int pValue) {
                   if (pValue < 0) {
                     throw std::invalid_argument("checkpoint-min-step must be non-negative");
                   }
                   pParams.vCheckpointMinStep = pValue;
                 })
          .mSetEnv("MGS_ARG_CHECKPOINT_MIN_STEP"));
  fAddOpt(common_arg({"-cram", "--cache-ram"}, "N",
                     fStringFormat("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)",
                                   pParams.vCacheRamMib),
                     [](common_params &pParams, int pValue) {
                       pParams.vCacheRamMib = pValue;
                       pParams.vCacheRamExplicit = true;
                     })
              .mSetEnv("MGS_ARG_CACHE_RAM"));
  fAddOpt(common_arg(
              {"-kvu", "--kv-unified"}, {"-no-kvu", "--no-kv-unified"},
              "use single unified KV buffer shared across all sequences (default: disabled)",
              [](common_params &pParams, bool pValue) { pParams.vKvUnified = pValue; })
              .mSetEnv("MGS_ARG_KV_UNIFIED"));
  fAddOpt(common_arg({"--cache-idle-slots"}, {"--no-cache-idle-slots"},
                     "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: "
                     "enabled, requires cache-ram)",
                     [](common_params &pParams, bool pValue) { pParams.vCacheIdleSlots = pValue; })
              .mSetEnv("MGS_ARG_CACHE_IDLE_SLOTS"));
  fAddOpt(common_arg({"--context-shift"}, {"--no-context-shift"},
                     fStringFormat("whether to use context shift on infinite text generation (default: %s)",
                                   pParams.vCtxShift ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCtxShift = pValue; })
              .mSetEnv("MGS_ARG_CONTEXT_SHIFT"));
  fAddOpt(common_arg({"-fa", "--flash-attn"}, "[on|off|auto]",
                     fStringFormat("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                   mgs_flash_attn_type_name(pParams.vFlashAttnType)),
                     [](common_params &pParams, const std::string &pValue) {
                       if (fIsTruthy(pValue)) {
                         pParams.vFlashAttnType = MGS_FLASH_ATTN_TYPE_ENABLED;
                       } else if (fIsFalsey(pValue)) {
                         pParams.vFlashAttnType = MGS_FLASH_ATTN_TYPE_DISABLED;
                       } else if (fIsAutoy(pValue)) {
                         pParams.vFlashAttnType = MGS_FLASH_ATTN_TYPE_AUTO;
                       } else {
                         throw std::runtime_error(
                             fStringFormat("error: unknown value for --flash-attn: '%s'\n", pValue.c_str()));
                       }
                     })
              .mSetEnv("MGS_ARG_FLASH_ATTN"));
  fAddOpt(common_arg({"--perf"}, {"--no-perf"},
                     fStringFormat("whether to enable internal mgs performance timings (default: %s)",
                                   pParams.vNoPerf ? "true" : "false"),
                     [](common_params &pParams, bool pValue) {
                       pParams.vNoPerf = !pValue;
                       pParams.vSampling.vNoPerf = !pValue;
                     })
              .mSetEnv("MGS_ARG_PERF"));
  fAddOpt(
      common_arg({"-e", "--escape"}, {"--no-escape"},
                 fStringFormat("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)",
                               pParams.vEscape ? "true" : "false"),
                 [](common_params &pParams, bool pValue) { pParams.vEscape = pValue; }));
  fAddOpt(common_arg(
              {"-r", "--reverse-prompt"}, "PROMPT", "halt generation at PROMPT, return control in interactive mode\n",
              [](common_params &pParams, const std::string &pValue) { pParams.lAntiprompt.emplace_back(pValue); }));
  fAddOpt(common_arg({"-sp", "--special"},
                     fStringFormat("special tokens output enabled (default: %s)", pParams.vSpecial ? "true" : "false"),
                     [](common_params &pParams) { pParams.vSpecial = true; }));
  fAddOpt(common_arg({"--warmup"}, {"--no-warmup"},
                     fStringFormat("whether to perform warmup with an empty run (default: %s)",
                                   pParams.vWarmup ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vWarmup = pValue; }));
  fAddOpt(common_arg({"--spm-infill"},
                     fStringFormat("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as "
                                   "some models prefer this. (default: %s)",
                                   pParams.vSpmInfill ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vSpmInfill = true; }));
  fAddOpt(common_arg(
              {"--samplers"}, "SAMPLERS",
              fStringFormat("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)",
                            vSamplerTypeNames.c_str()),
              [](common_params &pParams, const std::string &pValue) {
                const auto cSamplerNames = fStringSplit<std::string>(pValue, ';');
                pParams.vSampling.lSamplers = fCommonSamplerTypesFromNames(cSamplerNames);
                pParams.vSampling.vUserSamplingConfig |=
                    common_params_sampling_config::cCommonParamsSamplingConfigSamplers;
              })
              .mSetSampling());
  fAddOpt(
      common_arg(
          {"-s", "--seed"}, "SEED",
          fStringFormat("RNG seed (default: %d, use random seed for %d)", pParams.vSampling.vSeed, MGS_DEFAULT_SEED),
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vSeed = std::stoul(pValue); })
          .mSetSampling());
  fAddOpt(common_arg({"--sampler-seq"}, "SEQUENCE",
                     fStringFormat("simplified sequence for samplers that will be used (default: %s)",
                                   vSamplerTypeChars.c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.lSamplers = fCommonSamplerTypesFromChars(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--ignore-eos"},
                     "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
                     [](common_params &pParams) { pParams.vSampling.vIgnoreEos = true; })
              .mSetSampling());
  fAddOpt(common_arg({"--temp"}, "N",
                     fStringFormat("temperature (default: %.2f)", (double)pParams.vSampling.vTemp),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vTemp = std::stof(pValue);
                       pParams.vSampling.vTemp = std::max(pParams.vSampling.vTemp, 0.0f);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigTemp;
                     })
              .mSetSampling());
  fAddOpt(common_arg(
              {"--top-k"}, "N", fStringFormat("top-k sampling (default: %d, 0 = disabled)", pParams.vSampling.vTopK),
              [](common_params &pParams, int pValue) {
                pParams.vSampling.vTopK = pValue;
                pParams.vSampling.vUserSamplingConfig |= common_params_sampling_config::cCommonParamsSamplingConfigTopK;
              })
              .mSetSampling()
              .mSetEnv("MGS_ARG_TOP_K"));
  fAddOpt(common_arg({"--top-p"}, "N",
                     fStringFormat("top-p sampling (default: %.2f, 1.0 = disabled)", (double)pParams.vSampling.vTopP),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vTopP = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigTopP;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--min-p"}, "N",
                     fStringFormat("min-p sampling (default: %.2f, 0.0 = disabled)", (double)pParams.vSampling.vMinP),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMinP = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMinP;
                     })
              .mSetSampling());
  fAddOpt(
      common_arg(
          {"--top-nsigma"}, "N",
          fStringFormat("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", pParams.vSampling.vTopNSigma),
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vTopNSigma = std::stof(pValue); })
          .mSetSampling());
  fAddOpt(common_arg({"--xtc-probability"}, "N",
                     fStringFormat("xtc probability (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vXtcProbability),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vXtcProbability = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigXtcProbability;
                     })
              .mSetSampling());
  fAddOpt(common_arg(
              {"--xtc-threshold"}, "N",
              fStringFormat("xtc threshold (default: %.2f, 1.0 = disabled)", (double)pParams.vSampling.vXtcThreshold),
              [](common_params &pParams, const std::string &pValue) {
                pParams.vSampling.vXtcThreshold = std::stof(pValue);
                pParams.vSampling.vUserSamplingConfig |=
                    common_params_sampling_config::cCommonParamsSamplingConfigXtcThreshold;
              })
              .mSetSampling());
  fAddOpt(
      common_arg({"--typical"}, "N",
                 fStringFormat("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)",
                               (double)pParams.vSampling.vTypP),
                 [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vTypP = std::stof(pValue); })
          .mSetSampling());
  fAddOpt(common_arg({"--repeat-last-n"}, "N",
                     fStringFormat("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)",
                                   pParams.vSampling.vPenaltyLastN),
                     [](common_params &pParams, int pValue) {
                       if (pValue < -1) {
                         throw std::runtime_error(fStringFormat("error: invalid repeat-last-n = %d\n", pValue));
                       }
                       pParams.vSampling.vPenaltyLastN = pValue;
                       pParams.vSampling.vNPrev = std::max(pParams.vSampling.vNPrev, pParams.vSampling.vPenaltyLastN);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigPenaltyLastN;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--repeat-penalty"}, "N",
                     fStringFormat("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)",
                                   (double)pParams.vSampling.vPenaltyRepeat),
                     [](common_params &pParams, const std::string &pValue) {
                       const float cPenaltyRepeat = std::stof(pValue);
                       if (!std::isfinite(cPenaltyRepeat) || cPenaltyRepeat <= 0.0f ||
                           !std::isfinite(1.0f / cPenaltyRepeat)) {
                         throw std::runtime_error("error: repeat-penalty must be finite and greater than 0\n");
                       }
                       pParams.vSampling.vPenaltyRepeat = cPenaltyRepeat;
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigPenaltyRepeat;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--presence-penalty"}, "N",
                     fStringFormat("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vPenaltyPresent),
                     [](common_params &pParams, const std::string &pValue) {
                       const float cPenaltyPresent = std::stof(pValue);
                       if (!std::isfinite(cPenaltyPresent)) {
                         throw std::runtime_error("error: presence-penalty must be finite\n");
                       }
                       pParams.vSampling.vPenaltyPresent = cPenaltyPresent;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--frequency-penalty"}, "N",
                     fStringFormat("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vPenaltyFreq),
                     [](common_params &pParams, const std::string &pValue) {
                       const float cPenaltyFreq = std::stof(pValue);
                       if (!std::isfinite(cPenaltyFreq)) {
                         throw std::runtime_error("error: frequency-penalty must be finite\n");
                       }
                       pParams.vSampling.vPenaltyFreq = cPenaltyFreq;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--dry-multiplier"}, "N",
                     fStringFormat("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vDryMultiplier),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDryMultiplier = std::stof(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--dry-base"}, "N",
                     fStringFormat("set DRY sampling base value (default: %.2f)", (double)pParams.vSampling.vDryBase),
                     [](common_params &pParams, const std::string &pValue) {
                       float vPotentialBase = std::stof(pValue);
                       if (vPotentialBase >= 1.0f) {
                         pParams.vSampling.vDryBase = vPotentialBase;
                       }
                     })
              .mSetSampling());
  fAddOpt(common_arg(
              {"--dry-allowed-length"}, "N",
              fStringFormat("set allowed length for DRY sampling (default: %d)", pParams.vSampling.vDryAllowedLength),
              [](common_params &pParams, int pValue) { pParams.vSampling.vDryAllowedLength = pValue; })
              .mSetSampling());
  fAddOpt(
      common_arg({"--dry-penalty-last-n"}, "N",
                 fStringFormat("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)",
                               pParams.vSampling.vDryPenaltyLastN),
                 [](common_params &pParams, int pValue) {
                   if (pValue < -1) {
                     throw std::runtime_error(fStringFormat("error: invalid dry-penalty-last-n = %d\n", pValue));
                   }
                   pParams.vSampling.vDryPenaltyLastN = pValue;
                 })
          .mSetSampling());
  fAddOpt(common_arg({"--dry-sequence-breaker"}, "STRING",
                     fStringFormat("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the "
                                   "process; use \"none\" to not use any sequence breakers\n",
                                   pParams.vSampling.lDrySequenceBreakers.empty()
                                       ? "none"
                                       : std::accumulate(std::next(pParams.vSampling.lDrySequenceBreakers.begin()),
                                                         pParams.vSampling.lDrySequenceBreakers.end(),
                                                         std::string("'") +
                                                             (pParams.vSampling.lDrySequenceBreakers[0] == "\n"
                                                                  ? "\\n"
                                                                  : pParams.vSampling.lDrySequenceBreakers[0]) +
                                                             "'",
                                                         [](const std::string &pA, const std::string &pB) {
                                                           std::string vFormattedB = (pB == "\n") ? "\\n" : pB;
                                                           return pA + ", '" + vFormattedB + "'";
                                                         })
                                             .c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       static bool vDefaultsCleared = false;

                       if (!vDefaultsCleared) {
                         pParams.vSampling.lDrySequenceBreakers.clear();
                         vDefaultsCleared = true;
                       }

                       if (pValue == "none") {
                         pParams.vSampling.lDrySequenceBreakers.clear();
                       } else {
                         pParams.vSampling.lDrySequenceBreakers.emplace_back(pValue);
                       }
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--adaptive-target"}, "N",
                     fStringFormat("adaptive-p: select tokens near this probability (valid range 0.0 "
                                   "to 1.0; negative = disabled) (default: %.2f)",
                                   (double)pParams.vSampling.vAdaptiveTarget),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vAdaptiveTarget = std::stof(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--adaptive-decay"}, "N",
                     fStringFormat("adaptive-p: decay rate for target adaptation over time. lower values "
                                   "are more reactive, higher values are more stable.\n"
                                   "(valid range 0.0 to 0.99) (default: %.2f)",
                                   (double)pParams.vSampling.vAdaptiveDecay),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vAdaptiveDecay = std::stof(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--dynatemp-range"}, "N",
                     fStringFormat("dynamic temperature range (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vDynatempRange),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDynatempRange = std::stof(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--dynatemp-exp"}, "N",
                     fStringFormat("dynamic temperature exponent (default: %.2f)",
                                   (double)pParams.vSampling.vDynatempExponent),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDynatempExponent = std::stof(pValue);
                     })
              .mSetSampling());
  fAddOpt(common_arg(
              {"--mirostat"}, "N",
              fStringFormat("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
                            "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
                            pParams.vSampling.vMirostat),
              [](common_params &pParams, int pValue) {
                pParams.vSampling.vMirostat = pValue;
                pParams.vSampling.vUserSamplingConfig |=
                    common_params_sampling_config::cCommonParamsSamplingConfigMirostat;
              })
              .mSetSampling());
  fAddOpt(common_arg({"--mirostat-lr"}, "N",
                     fStringFormat("Mirostat learning rate, parameter eta (default: %.2f)",
                                   (double)pParams.vSampling.vMirostatEta),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMirostatEta = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMirostatEta;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--mirostat-ent"}, "N",
                     fStringFormat("Mirostat target entropy, parameter tau (default: %.2f)",
                                   (double)pParams.vSampling.vMirostatTau),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMirostatTau = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMirostatTau;
                     })
              .mSetSampling());
  fAddOpt(common_arg({"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
                     "modifies the likelihood of token appearing in the completion,\n"
                     "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
                     "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
                     [](common_params &pParams, const std::string &pValue) {
                       std::stringstream vSs(pValue);
                       mgs_token vKey;
                       char vSign;
                       std::string vValueStr;
                       try {
                         if (vSs >> vKey && vSs >> vSign && std::getline(vSs, vValueStr) &&
                             (vSign == '+' || vSign == '-')) {
                           const float cBias = std::stof(vValueStr) * ((vSign == '-') ? -1.0f : 1.0f);
                           pParams.vSampling.lLogitBias.push_back({vKey, cBias});
                         } else {
                           throw std::invalid_argument("invalid input format");
                         }
                       } catch (const std::exception &) {
                         throw std::invalid_argument("invalid input format");
                       }
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--grammar"}, "GRAMMAR",
                     "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeUser, pValue};
                     })
              .mSetSampling());
  fAddOpt(common_arg({"--grammar-file"}, "FNAME", "file to read grammar from",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeUser, fReadFile(pValue)};
                     })
              .mSetSampling());
  fAddOpt(common_arg({"-j", "--json-schema"}, "SCHEMA",
                     "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON "
                     "object\nFor schemas w/ external $refs, convert them to a GBNF grammar and pass it with "
                     "--grammar or --grammar-file instead",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeOutputFormat,
                                                     fJsonSchemaToGrammar(json::parse(pValue))};
                     })
              .mSetSampling());
  fAddOpt(common_arg(
              {"-jf", "--json-schema-file"}, "FILE",
              "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any "
              "JSON object\nFor schemas w/ external $refs, convert them to a GBNF grammar and pass it with "
              "--grammar or --grammar-file instead",
              [](common_params &pParams, const std::string &pValue) {
                std::ifstream vFile(pValue);
                if (!vFile) {
                  throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pValue.c_str()));
                }
                std::string vSchema;
                std::copy(std::istreambuf_iterator<char>(vFile), std::istreambuf_iterator<char>(),
                          std::back_inserter(vSchema));
                pParams.vSampling.vGrammar = {cCommonGrammarTypeOutputFormat,
                                              fJsonSchemaToGrammar(json::parse(vSchema))};
              })
              .mSetSampling());
  fAddOpt(common_arg({"-bs", "--backend-sampling"}, "enable backend sampling (experimental) (default: disabled)",
                     [](common_params &pParams) { pParams.vSampling.vBackendSampling = true; })
              .mSetSampling()
              .mSetEnv("MGS_ARG_BACKEND_SAMPLING"));
  fAddOpt(common_arg({"--pooling"}, "{none,mean,cls,last,rank}",
                     "pooling type for embeddings, use model default if unspecified",
                     [](common_params &pParams, const std::string &pValue) {
                       /**/ if (pValue == "none") {
                         pParams.vPoolingType = MGS_POOLING_TYPE_NONE;
                       } else if (pValue == "mean") {
                         pParams.vPoolingType = MGS_POOLING_TYPE_MEAN;
                       } else if (pValue == "cls") {
                         pParams.vPoolingType = MGS_POOLING_TYPE_CLS;
                       } else if (pValue == "last") {
                         pParams.vPoolingType = MGS_POOLING_TYPE_LAST;
                       } else if (pValue == "rank") {
                         pParams.vPoolingType = MGS_POOLING_TYPE_RANK;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
              .mSetEnv("MGS_ARG_POOLING"));
  fAddOpt(common_arg({"--rope-scaling"}, "{none,linear,yarn}",
                     "RoPE frequency scaling method, defaults to linear unless specified by the model",
                     [](common_params &pParams, const std::string &pValue) {
                       /**/ if (pValue == "none") {
                         pParams.vRopeScalingType = MGS_ROPE_SCALING_TYPE_NONE;
                       } else if (pValue == "linear") {
                         pParams.vRopeScalingType = MGS_ROPE_SCALING_TYPE_LINEAR;
                       } else if (pValue == "yarn") {
                         pParams.vRopeScalingType = MGS_ROPE_SCALING_TYPE_YARN;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
              .mSetEnv("MGS_ARG_ROPE_SCALING"));
  fAddOpt(common_arg({"--rope-scale"}, "N", "RoPE context scaling factor, expands context by a factor of N",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vRopeFreqScale = 1.0f / std::stof(pValue);
                     })
              .mSetEnv("MGS_ARG_ROPE_SCALE"));
  fAddOpt(common_arg(
              {"--rope-freq-base"}, "N", "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
              [](common_params &pParams, const std::string &pValue) { pParams.vRopeFreqBase = std::stof(pValue); })
              .mSetEnv("MGS_ARG_ROPE_FREQ_BASE"));
  fAddOpt(
      common_arg({"--rope-freq-scale"}, "N", "RoPE frequency scaling factor, expands context by a factor of 1/N",
                 [](common_params &pParams, const std::string &pValue) { pParams.vRopeFreqScale = std::stof(pValue); })
          .mSetEnv("MGS_ARG_ROPE_FREQ_SCALE"));
  fAddOpt(common_arg({"--yarn-orig-ctx"}, "N",
                     fStringFormat("YaRN: original context size of model (default: %d = model training context size)",
                                   pParams.vYarnOrigCtx),
                     [](common_params &pParams, int pValue) { pParams.vYarnOrigCtx = pValue; })
              .mSetEnv("MGS_ARG_YARN_ORIG_CTX"));
  fAddOpt(
      common_arg({"--yarn-ext-factor"}, "N",
                 fStringFormat("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)",
                               (double)pParams.vYarnExtFactor),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnExtFactor = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_EXT_FACTOR"));
  fAddOpt(
      common_arg(
          {"--yarn-attn-factor"}, "N",
          fStringFormat("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)pParams.vYarnAttnFactor),
          [](common_params &pParams, const std::string &pValue) { pParams.vYarnAttnFactor = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_ATTN_FACTOR"));
  fAddOpt(
      common_arg({"--yarn-beta-slow"}, "N",
                 fStringFormat("YaRN: high correction dim or alpha (default: %.2f)", (double)pParams.vYarnBetaSlow),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnBetaSlow = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_BETA_SLOW"));
  fAddOpt(
      common_arg({"--yarn-beta-fast"}, "N",
                 fStringFormat("YaRN: low correction dim or beta (default: %.2f)", (double)pParams.vYarnBetaFast),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnBetaFast = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_BETA_FAST"));
  fAddOpt(common_arg({"-kvo", "--kv-offload"}, {"-nkvo", "--no-kv-offload"},
                     fStringFormat("whether to enable KV cache offloading (default: %s)",
                                   pParams.vNoKvOffload ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoKvOffload = !pValue; })
              .mSetEnv("MGS_ARG_KV_OFFLOAD"));
  fAddOpt(common_arg({"--repack"}, {"-nr", "--no-repack"},
                     fStringFormat("whether to enable weight repacking (default: %s)",
                                   pParams.vNoExtraBufts ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoExtraBufts = !pValue; })
              .mSetEnv("MGS_ARG_REPACK"));
  fAddOpt(common_arg({"--no-host"}, "bypass host buffer allowing extra buffers to be used", [](common_params &pParams) {
            pParams.vNoHost = true;
          }).mSetEnv("MGS_ARG_NO_HOST"));
  fAddOpt(common_arg({"-ctk", "--cache-type-k"}, "TYPE",
                     fStringFormat("KV cache data type for K\n"
                                   "allowed values: %s\n"
                                   "(default: chosen from the weights of the model: f16 for unquantized weights, q8_0 "
                                   "for more than 5 bits per weight, q4_0 for the rest, on every backend; f16 when Flash "
                                   "Attention cannot read that type)",
                                   fGetAllKvCacheTypes().c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vCacheTypeK = fKvCacheTypeFromStr(pValue);
                       pParams.vCacheTypeExplicit = true;
                     })
              .mSetEnv("MGS_ARG_CACHE_TYPE_K"));
  fAddOpt(common_arg({"-ctv", "--cache-type-v"}, "TYPE",
                     fStringFormat("KV cache data type for V\n"
                                   "allowed values: %s\n"
                                   "(default: chosen from the weights of the model: f16 for unquantized weights, q8_0 "
                                   "for more than 5 bits per weight, q4_0 for the rest, on every backend; f16 when Flash "
                                   "Attention cannot read that type)",
                                   fGetAllKvCacheTypes().c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vCacheTypeV = fKvCacheTypeFromStr(pValue);
                       pParams.vCacheTypeExplicit = true;
                     })
              .mSetEnv("MGS_ARG_CACHE_TYPE_V"));
  fAddOpt(common_arg({"-np", "--parallel"}, "N",
                     fStringFormat("number of server slots (default: %d: one request at a time with the whole "
                                   "context, the others wait in arrival order; more slots run requests in parallel "
                                   "and split the context among them)",
                                   pParams.vNParallel),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("error: invalid value for n_parallel\n");
                       }
                       pParams.vNParallel = pValue;
                     })
              .mSetEnv("MGS_ARG_PARALLEL"));
  fAddOpt(common_arg({"-cb", "--cont-batching"}, {"-nocb", "--no-cont-batching"},
                     fStringFormat("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)",
                                   pParams.vContBatching ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vContBatching = pValue; })
              .mSetEnv("MGS_ARG_CONT_BATCHING"));
  fAddOpt(common_arg({"-mm", "--mmproj"}, "FILE",
                     "path to a multimodal projector file. see tools/mtmd/README.md\n"
                     "note: if -hf is used, this argument can be omitted",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMmproj.vPath = pValue; })
              .mSetEnv("MGS_ARG_MMPROJ"));
  fAddOpt(common_arg({"-mmu", "--mmproj-url"}, "URL", "URL to a multimodal projector file. see tools/mtmd/README.md",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMmproj.vUrl = pValue; })
              .mSetEnv("MGS_ARG_MMPROJ_URL"));
  fAddOpt(common_arg({"--mmproj-auto"}, {"--no-mmproj"},
                     fStringFormat(
                         "whether to use multimodal projector file (if available), useful when using -hf (default: %s)",
                         pParams.vNoMmproj ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoMmproj = !pValue; })
              .mSetEnv("MGS_ARG_MMPROJ_AUTO"));
  fAddOpt(common_arg({"--mmproj-offload"}, {"--no-mmproj-offload"},
                     fStringFormat("whether to enable GPU offloading for multimodal projector (default: %s)",
                                   pParams.vMmprojUseGpu ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vMmprojUseGpu = pValue; })
              .mSetEnv("MGS_ARG_MMPROJ_OFFLOAD"));
  fAddOpt(common_arg({"--image-min-tokens"}, "N",
                     "minimum number of tokens each image can take, only used by vision models with dynamic resolution "
                     "(default: read from model)",
                     [](common_params &pParams, int pValue) { pParams.vImageMinTokens = pValue; })
              .mSetEnv("MGS_ARG_IMAGE_MIN_TOKENS"));
  fAddOpt(common_arg({"--image-max-tokens"}, "N",
                     "maximum number of tokens each image can take, only used by vision models with dynamic resolution "
                     "(default: read from model)",
                     [](common_params &pParams, int pValue) { pParams.vImageMaxTokens = pValue; })
              .mSetEnv("MGS_ARG_IMAGE_MAX_TOKENS"));
  fAddOpt(common_arg({"--mtmd-batch-max-tokens"}, "N",
                     fStringFormat("maximum number of image tokens per batch when encoding images (default: %d)",
                                   pParams.vMtmdBatchMaxTokens),
                     [](common_params &pParams, int pValue) { pParams.vMtmdBatchMaxTokens = pValue; })
              .mSetEnv("MGS_ARG_MTMD_BATCH_MAX_TOKENS"));
  fAddOpt(
      common_arg({"-lm", "--load-mode"}, "MODE",
                 "model loading mode (default: mmap)\n"
                 "- none: no special loading mode\n"
                 "- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n"
                 "- mlock: force system to keep model in RAM rather than swapping or compressing\n"
                 "- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n"
                 "- dio: use DirectIO if available\n",
                 [](common_params &pParams, const std::string &pValue) {
                   /**/ if (pValue == "none") {
                     pParams.vLoadMode = MGS_LOAD_MODE_NONE;
                   } else if (pValue == "mmap") {
                     pParams.vLoadMode = MGS_LOAD_MODE_MMAP;
                   } else if (pValue == "mlock") {
                     pParams.vLoadMode = MGS_LOAD_MODE_MLOCK;
                   } else if (pValue == "mmap+mlock") {
                     pParams.vLoadMode = MGS_LOAD_MODE_MMAP_MLOCK;
                   } else if (pValue == "dio") {
                     pParams.vLoadMode = MGS_LOAD_MODE_DIRECT_IO;
                   } else {
                     throw std::invalid_argument("invalid value");
                   }
                 })
          .mSetEnv("MGS_ARG_LOAD_MODE"));
  fAddOpt(
      common_arg({"--numa"}, "TYPE",
                 "attempt optimizations that help on some NUMA systems\n"
                 "- distribute: spread execution evenly over all nodes\n"
                 "- isolate: only spawn threads on CPUs on the node that execution started on\n"
                 "- numactl: use the CPU map provided by numactl\n"
                 "if run without this previously, it is recommended to drop the system page cache before using this",
                 [](common_params &pParams, const std::string &pValue) {
                   /**/ if (pValue == "distribute" || pValue == "") {
                     pParams.vNuma = GGML_NUMA_STRATEGY_DISTRIBUTE;
                   } else if (pValue == "isolate") {
                     pParams.vNuma = GGML_NUMA_STRATEGY_ISOLATE;
                   } else if (pValue == "numactl") {
                     pParams.vNuma = GGML_NUMA_STRATEGY_NUMACTL;
                   } else {
                     throw std::invalid_argument("invalid value");
                   }
                 })
          .mSetEnv("MGS_ARG_NUMA"));
  fAddOpt(
      common_arg({"-dev", "--device"}, "<dev1,dev2,..>",
                 "comma-separated list of devices to use for offloading (none = don't offload)\n"
                 "use --list-devices to see a list of available devices",
                 [](common_params &pParams, const std::string &pValue) { pParams.lDevices = fParseDeviceList(pValue); })
          .mSetEnv("MGS_ARG_DEVICE"));
  fAddOpt(common_arg({"--list-devices"}, "print list of available devices and exit", [](common_params &) {
    fCommonPrintAvailableDevices();
    exit(0);
  }));
  fAddOpt(common_arg({"-ot", "--override-tensor"}, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type",
                     [](common_params &pParams, const std::string &pValue) {
                       fParseTensorBufferOverrides(pValue, pParams.lTensorBuftOverrides);
                     })
              .mSetEnv("MGS_ARG_OVERRIDE_TENSOR"));
  fAddOpt(common_arg({"-cmoe", "--cpu-moe"}, "keep all Mixture of Experts (MoE) weights in the CPU",
                     [](common_params &pParams) { pParams.lTensorBuftOverrides.push_back(fLlmFfnExpsCpuOverride()); })
              .mSetEnv("MGS_ARG_CPU_MOE"));
  fAddOpt(common_arg({"--moe-cache"}, {"--no-moe-cache"},
                     "keep the most recently used MoE experts stored in system memory in a device cache, and split "
                     "every decoding step between that cache and the CPU (default: enabled)",
                     [](common_params &pParams, bool pValue) { pParams.vMoeCache = pValue; })
              .mSetEnv("MGS_ARG_MOE_CACHE"));
  fAddOpt(common_arg({"--moe-cache-size"}, "MiB",
                     "device memory for the MoE expert cache in MiB; with several GPUs the total is split among them "
                     "by the experts each one caches (default: -1, the memory left free on every GPU after the "
                     "model, the context and --moe-cache-reserve)",
                     [](common_params &pParams, int pValue) { pParams.vMoeCacheMib = pValue < 0 ? -1 : pValue; })
              .mSetEnv("MGS_ARG_MOE_CACHE_SIZE"));
  fAddOpt(common_arg({"--moe-cache-reserve"}, "MiB",
                     "device memory left free on every GPU besides the MoE expert cache and the expert prefetch "
                     "slots, in MiB (default: 1024)",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vMoeCacheReserveMib = pValue;
                     })
              .mSetEnv("MGS_ARG_MOE_CACHE_RESERVE"));
  fAddOpt(common_arg({"-ncmoe", "--n-cpu-moe"}, "N",
                     "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       for (int vI = 0; vI < pValue; ++vI) {
                         // keep strings alive and avoid leaking memory by storing them in a static vector
                         static std::list<std::string> lBuftOverrides;
                         lBuftOverrides.push_back(fLlmFfnExpsBlockRegex(vI));
                         pParams.lTensorBuftOverrides.push_back(
                             {lBuftOverrides.back().c_str(), ggml_backend_cpu_buffer_type()});
                       }
                     })
              .mSetEnv("MGS_ARG_N_CPU_MOE"));
  GGML_ASSERT(pParams.vNGpuLayers < 0); // string_format would need to be extended for a default >= 0
  fAddOpt(
      common_arg({"-ngl", "--gpu-layers"}, "N",
                 fStringFormat(
                     "max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
                     pParams.vNGpuLayers == -1 ? "auto" : "all"),
                 [](common_params &pParams, const std::string &pValue) {
                   if (pValue == "auto") {
                     pParams.vNGpuLayers = -1;
                   } else if (pValue == "all") {
                     pParams.vNGpuLayers = -2;
                   } else {
                     pParams.vNGpuLayers = std::stoi(pValue);
                   }
                   if (!mgs_supports_gpu_offload()) {
                     fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                     fprintf(stderr, "warning: one possible reason is that mgs was compiled without GPU support\n");
                     fprintf(stderr, "warning: see doc/MANUAL.md for the build instructions\n");
                   }
                 })
          .mSetEnv("MGS_ARG_GPU_LAYERS"));
  fAddOpt(common_arg({"-sm", "--split-mode"}, "{none,layer,row,tensor}",
                     "how to split the model across multiple GPUs, one of:\n"
                     "- none: use one GPU only\n"
                     "- layer (default): split layers and KV across GPUs (pipelined)\n"
                     "- row: split weight across GPUs by rows (parallelized)\n"
                     "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
                     [](common_params &pParams, const std::string &pValue) {
                       if (pValue == "none") {
                         pParams.vSplitMode = MGS_SPLIT_MODE_NONE;
                       } else if (pValue == "layer") {
                         pParams.vSplitMode = MGS_SPLIT_MODE_LAYER;
                       } else if (pValue == "row") {
                         pParams.vSplitMode = MGS_SPLIT_MODE_ROW;
                       } else if (pValue == "tensor") {
                         pParams.vSplitMode = MGS_SPLIT_MODE_TENSOR;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                       if (!mgs_supports_gpu_offload()) {
                         fprintf(stderr, "warning: mgs was compiled without support for GPU offload. Setting the "
                                         "split mode has no effect.\n");
                       }
                     })
              .mSetEnv("MGS_ARG_SPLIT_MODE"));
  fAddOpt(common_arg({"-ts", "--tensor-split"}, "N0,N1,N2,...",
                     "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
                     [](common_params &pParams, const std::string &pValue) {
                       std::string vArgNext = pValue;

                       // split string by , and /
                       const std::regex cRegex{R"([,/]+)"};
                       std::sregex_token_iterator vIt{vArgNext.begin(), vArgNext.end(), cRegex, -1};
                       std::vector<std::string> lSplitArg{vIt, {}};
                       if (lSplitArg.size() >= mgs_max_devices()) {
                         throw std::invalid_argument(
                             fStringFormat("got %zu input configs, but system only has %zu devices", lSplitArg.size(),
                                           mgs_max_devices()));
                       }
                       for (size_t vI = 0; vI < mgs_max_devices(); ++vI) {
                         if (vI < lSplitArg.size()) {
                           pParams.aTensorSplit[vI] = std::stof(lSplitArg[vI]);
                         } else {
                           pParams.aTensorSplit[vI] = 0.0f;
                         }
                       }
                       if (!mgs_supports_gpu_offload()) {
                         fprintf(stderr, "warning: mgs was compiled without support for GPU offload. Setting a "
                                         "tensor split has no effect.\n");
                       }
                     })
              .mSetEnv("MGS_ARG_TENSOR_SPLIT"));
  fAddOpt(common_arg({"-mg", "--main-gpu"}, "INDEX",
                     fStringFormat("the GPU to use for the model (with split-mode = none), or for intermediate results "
                                   "and KV (with split-mode = row) (default: %d)",
                                   pParams.vMainGpu),
                     [](common_params &pParams, int pValue) {
                       pParams.vMainGpu = pValue;
                       if (!mgs_supports_gpu_offload()) {
                         fprintf(stderr, "warning: mgs was compiled without support for GPU offload. Setting the "
                                         "main GPU has no effect.\n");
                       }
                     })
              .mSetEnv("MGS_ARG_MAIN_GPU"));
  fAddOpt(common_arg(
              {"-fit", "--fit"}, "[on|off]",
              fStringFormat("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')",
                            pParams.vFitParams ? "on" : "off"),
              [](common_params &pParams, const std::string &pValue) {
                if (fIsTruthy(pValue)) {
                  pParams.vFitParams = true;
                } else if (fIsFalsey(pValue)) {
                  pParams.vFitParams = false;
                } else {
                  throw std::runtime_error(fStringFormat("error: unknown value for --fit: '%s'\n", pValue.c_str()));
                }
              })
              .mSetEnv("MGS_ARG_FIT"));
  fAddOpt(common_arg({"-fitt", "--fit-target"}, "MiB0,MiB1,MiB2,...",
                     fStringFormat("target margin per device for --fit, comma-separated list of values, "
                                   "single value is broadcast across all devices, default: %zu",
                                   pParams.lFitParamsTarget[0] / (1024 * 1024)),
                     [](common_params &pParams, const std::string &pValue) {
                       std::string vArgNext = pValue;

                       // split string by , and /
                       const std::regex cRegex{R"([,/]+)"};
                       std::sregex_token_iterator vIt{vArgNext.begin(), vArgNext.end(), cRegex, -1};
                       std::vector<std::string> lSplitArg{vIt, {}};
                       if (lSplitArg.size() >= mgs_max_devices()) {
                         throw std::invalid_argument(
                             fStringFormat("got %zu input configs, but system only has %zu devices", lSplitArg.size(),
                                           mgs_max_devices()));
                       }
                       if (lSplitArg.size() == 1) {
                         std::fill(pParams.lFitParamsTarget.begin(), pParams.lFitParamsTarget.end(),
                                   std::stoull(lSplitArg[0]) * 1024 * 1024);
                         return;
                       }
                       for (size_t vI = 0; vI < lSplitArg.size(); vI++) {
                         pParams.lFitParamsTarget[vI] = std::stoull(lSplitArg[vI]) * 1024 * 1024;
                       }
                     })
              .mSetEnv("MGS_ARG_FIT_TARGET"));
  fAddOpt(common_arg({"-fitc", "--fit-ctx"}, "N",
                     fStringFormat("minimum ctx size that can be set by --fit option, default: %" PRIu32,
                                   pParams.vFitParamsMinCtx),
                     [](common_params &pParams, int pValue) { pParams.vFitParamsMinCtx = pValue; })
              .mSetEnv("MGS_ARG_FIT_CTX"));
  fAddOpt(common_arg({"--check-tensors"},
                     fStringFormat("check model tensor data for invalid values (default: %s)",
                                   pParams.vCheckTensors ? "true" : "false"),
                     [](common_params &pParams) { pParams.vCheckTensors = true; }));
  fAddOpt(common_arg({"--override-kv"}, "KEY=TYPE:VALUE,...",
                     "advanced option to override model metadata by key. to specify multiple overrides, either use "
                     "comma-separated values.\n"
                     "types: int, float, bool, str. example: --override-kv "
                     "tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         if (!fStringParseKvOverride(vItem.c_str(), pParams.lKvOverrides)) {
                           throw std::runtime_error(
                               fStringFormat("error: Invalid type for KV override: %s\n", vItem.c_str()));
                         }
                       }
                     }));
  fAddOpt(common_arg({"--op-offload"}, {"--no-op-offload"},
                     fStringFormat("whether to offload host tensor operations to device (default: %s)",
                                   pParams.vNoOpOffload ? "false" : "true"),
                     [](common_params &pParams, bool pValue) { pParams.vNoOpOffload = !pValue; }));
  fAddOpt(common_arg({"--lora"}, "FNAME", "path to LoRA adapter (use comma-separated values to load multiple adapters)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         pParams.lLoraAdapters.push_back({vItem, 1.0, "", "", nullptr});
                       }
                     }
                     // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora,
                     // it will be categorized as "example-specific" arg
                     ));
  fAddOpt(common_arg({"--lora-scaled"}, "FNAME:SCALE,...",
                     "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\n"
                     "note: use comma-separated values",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         auto vParts = fStringSplit<std::string>(vItem, ':');
                         if (vParts.size() != 2) {
                           throw std::invalid_argument("lora-scaled format: FNAME:SCALE");
                         }
                         pParams.lLoraAdapters.push_back({vParts[0], std::stof(vParts[1]), "", "", nullptr});
                       }
                     }
                     // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora,
                     // it will be categorized as "example-specific" arg
                     ));
  fAddOpt(common_arg({"--control-vector"}, "FNAME",
                     "add a control vector\nnote: use comma-separated values to add multiple control vectors",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         pParams.lControlVectors.push_back({
                             1.0f,
                             vItem,
                         });
                       }
                     }));
  fAddOpt(common_arg({"--control-vector-scaled"}, "FNAME:SCALE,...",
                     "add a control vector with user defined scaling SCALE\n"
                     "note: use comma-separated values (format: FNAME:SCALE,...)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         auto vParts = fStringSplit<std::string>(vItem, ':');
                         if (vParts.size() != 2) {
                           throw std::invalid_argument("control-vector-scaled format: FNAME:SCALE");
                         }
                         pParams.lControlVectors.push_back({std::stof(vParts[1]), vParts[0]});
                       }
                     }));
  fAddOpt(common_arg({"--control-vector-layer-range"}, "START", "END",
                     "layer range to apply the control vector(s) to, start and end inclusive",
                     [](common_params &pParams, const std::string &pStart, const std::string &pEnd) {
                       pParams.vControlVectorLayerStart = std::stoi(pStart);
                       pParams.vControlVectorLayerEnd = std::stoi(pEnd);
                     }));
  fAddOpt(common_arg({"-a", "--alias"}, "STRING", "set model name aliases, comma-separated (to be used by API)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (auto &vAlias : fStringSplit<std::string>(pValue, ',')) {
                         vAlias = fStringStrip(vAlias);
                         if (!vAlias.empty()) {
                           pParams.sModelAlias.insert(vAlias);
                         }
                       }
                     })
              .mSetEnv("MGS_ARG_ALIAS"));
  fAddOpt(common_arg({"--tags"}, "STRING", "set model tags, comma-separated (informational, not used for routing)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (auto &vTag : fStringSplit<std::string>(pValue, ',')) {
                         vTag = fStringStrip(vTag);
                         if (!vTag.empty()) {
                           pParams.sModelTags.insert(vTag);
                         }
                       }
                     })
              .mSetEnv("MGS_ARG_TAGS"));
  fAddOpt(common_arg({"-m", "--model"}, "FNAME", "model path to load",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vPath = pValue; })
              .mSetEnv("MGS_ARG_MODEL"));
  fAddOpt(common_arg({"-mu", "--model-url"}, "MODEL_URL", "model download url (default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vUrl = pValue; })
              .mSetEnv("MGS_ARG_MODEL_URL"));
  fAddOpt(common_arg(
              {"-dr", "--docker-repo"}, "[<repo>/]<model>[:quant]",
              "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
              "example: gemma3\n"
              "(default: unused)",
              [](common_params &pParams, const std::string &pValue) { pParams.vModel.vDockerRepo = pValue; })
              .mSetEnv("MGS_ARG_DOCKER_REPO"));
  fAddOpt(common_arg({"-hf", "--hf-repo"}, "<user>/<model>[:quant]",
                     "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls "
                     "back to the first file in the repo if Q4_K_M doesn't exist.\n"
                     "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
                     "example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n"
                     "(default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vHfRepo = pValue; })
              .mSetEnv("MGS_ARG_HF_REPO"));
  fAddOpt(common_arg({"-hff", "--hf-file"}, "FILE",
                     "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vHfFile = pValue; })
              .mSetEnv("MGS_ARG_HF_FILE"));
  fAddOpt(common_arg({"-hft", "--hf-token"}, "TOKEN",
                     "Hugging Face access token (default: value from HF_TOKEN environment variable)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vHfToken = pValue; })
              .mSetEnv("HF_TOKEN"));
  fAddOpt(common_arg({"--embd-normalize"}, "N",
                     fStringFormat("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, "
                                   "1=taxicab, 2=euclidean, >2=p-norm)",
                                   pParams.vEmbdNormalize),
                     [](common_params &pParams, int pValue) { pParams.vEmbdNormalize = pValue; }));
  fAddOpt(common_arg({"--host"}, "HOST",
                     fStringFormat(
                         "ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: %s)",
                         pParams.vHostname.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vHostname = pValue; })
              .mSetEnv("MGS_ARG_HOST"));
  fAddOpt(common_arg({"--port"}, "PORT",
                     "fixed listening port; 0 selects an OS-assigned port (default: first free from 11443 for HTTPS, "
                     "11080 for --http)",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 0 || pValue > 65535) {
                         throw std::invalid_argument("--port must be between 0 and 65535");
                       }
                       pParams.vPort = pValue;
                       pParams.vPortExplicit = true;
                     })
              .mSetEnv("MGS_ARG_PORT"));
  fAddOpt(common_arg({"--https"}, {"--http"},
                     "serve HTTPS with HTTP redirection (default: enabled); --http explicitly serves plain HTTP",
                     [](common_params &pParams, bool pValue) { pParams.vUseHttps = pValue; })
              .mSetEnv("MGS_ARG_HTTPS"));
  fAddOpt(common_arg({"--http-port"}, "PORT", "first HTTP redirect port to try (default: 11080)",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 65535) {
                         throw std::invalid_argument("--http-port must be between 1 and 65535");
                       }
                       pParams.vHttpPort = pValue;
                     })
              .mSetEnv("MGS_ARG_HTTP_PORT"));
  fAddOpt(common_arg({"--reuse-port"},
                     fStringFormat("allow multiple sockets to bind to the same port (default: %s)",
                                   pParams.vReusePort ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vReusePort = true; })
              .mSetEnv("MGS_ARG_REUSE_PORT"));
  fAddOpt(common_arg({"--path"}, "PATH",
                     fStringFormat("path to serve static files from (default: %s)", pParams.vPublicPath.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vPublicPath = pValue; })
              .mSetEnv("MGS_ARG_PATH"));
  fAddOpt(
      common_arg({"--cors-origins"}, "ORIGINS",
                 fStringFormat("comma-separated list of allowed origins for CORS (default: %s)\n"
                               "the special value 'localhost' only lets pages of localhost and of this same server "
                               "call the API: requests other than GET and HEAD from any other site are refused",
                               pParams.vCorsOrigins.c_str()),
                 [](common_params &pParams, const std::string &pValue) { pParams.vCorsOrigins = pValue; })
          .mSetEnv("MGS_ARG_CORS_ORIGINS"));
  fAddOpt(common_arg({"--cors-methods"}, "METHODS",
                     fStringFormat("comma-separated list of allowed methods for CORS (default: %s)",
                                   pParams.vCorsMethods.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCorsMethods = pValue; })
              .mSetEnv("MGS_ARG_CORS_METHODS"));
  fAddOpt(common_arg({"--cors-headers"}, "HEADERS",
                     fStringFormat("comma-separated list of allowed headers for CORS (default: %s)",
                                   pParams.vCorsHeaders.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCorsHeaders = pValue; })
              .mSetEnv("MGS_ARG_CORS_HEADERS"));
  fAddOpt(common_arg({"--cors-credentials"}, {"--no-cors-credentials"},
                     fStringFormat("whether to allow credentials for CORS (default: %s)\n"
                                   "note: if this is enabled and --cors-origins is set to *, the Origin header "
                                   "will be echoed back, and credentials will always be allowed",
                                   pParams.vCorsCredentials ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCorsCredentials = pValue; })
              .mSetEnv("MGS_ARG_CORS_CREDENTIALS"));
  fAddOpt(common_arg({"--api-prefix"}, "PREFIX",
                     fStringFormat("prefix path the server serves from, without the trailing slash (default: %s)",
                                   pParams.vApiPrefix.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vApiPrefix = pValue; })
              .mSetEnv("MGS_ARG_API_PREFIX"));
  fAddOpt(common_arg({"--ui-config"}, "JSON", "JSON that provides default UI settings (overrides UI defaults)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vUiConfigJson = pValue; })
              .mSetEnv("MGS_ARG_UI_CONFIG"));
  fAddOpt(
      common_arg({"--ui-config-file"}, "PATH", "JSON file that provides default UI settings (overrides UI defaults)",
                 [](common_params &pParams, const std::string &pValue) { pParams.vUiConfigJson = fReadFile(pValue); })
          .mSetEnv("MGS_ARG_UI_CONFIG_FILE"));
  fAddOpt(common_arg({"--ui-mcp-proxy"}, {"--no-ui-mcp-proxy"},
                     "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments "
                     "(default: disabled)",
                     [](common_params &pParams, bool pValue) { pParams.vUiMcpProxy = pValue; })
              .mSetEnv("MGS_ARG_UI_MCP_PROXY"));
  fAddOpt(
      common_arg({"--tools"}, "TOOL1,TOOL2,...",
                 "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted "
                 "environments (default: no tools)\n"
                 "specify \"all\" to enable all tools\n"
                 "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, "
                 "edit_file, get_datetime, get_info",
                 [](common_params &pParams, const std::string &pValue) { pParams.lServerTools = fParseCsvRow(pValue); })
          .mSetEnv("MGS_ARG_TOOLS"));
  fAddOpt(common_arg({"--mcp-servers-config"}, "PATH",
                     "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not "
                     "enable in untrusted environments (default: none)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMcpServersConfig = pValue; })
              .mSetEnv("MGS_ARG_MCP_SERVERS_CONFIG"));
  fAddOpt(common_arg({"--mcp-servers-json"}, "JSON",
                     "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable "
                     "in untrusted environments (default: none)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMcpServersJson = pValue; })
              .mSetEnv("MGS_ARG_MCP_SERVERS_JSON"));
  fAddOpt(common_arg({"--local-shell-mcp-server"},
                     "experimental: serve the local shell MCP server at <api prefix>/mcp/shell, which runs commands in "
                     "the shells installed on this machine (bash.run, zsh.run, sh.run, psh.run; only the shells found "
                     "are exposed); the Web UI adds it and enables it by default - do not enable in untrusted "
                     "environments (default: disabled)",
                     [](common_params &pParams) { pParams.vShellMcpServer = true; })
              .mSetEnv("MGS_ARG_LOCAL_SHELL_MCP_SERVER"));
  fAddOpt(common_arg({"-ag", "--agent"}, {"-no-ag", "--no-agent"},
                     "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments "
                     "(default: disabled)",
                     [](common_params &pParams, bool pValue) {
                       if (pValue) {
                         pParams.lServerTools = {"all"};
                         pParams.vUiMcpProxy = true;
                       } else {
                         pParams.lServerTools.clear();
                         pParams.vUiMcpProxy = false;
                       }
                     })
              .mSetEnv("MGS_ARG_AGENT"));
  fAddOpt(common_arg({"--ui"}, {"--no-ui"},
                     fStringFormat("whether to enable the Web UI (default: %s)", pParams.vUi ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vUi = pValue; })
              .mSetEnv("MGS_ARG_UI"));
  fAddOpt(common_arg(
              {"--embedding"},
              fStringFormat(
                  "restrict to only support embedding use case; use only with dedicated embedding models (default: %s)",
                  pParams.vEmbedding ? "enabled" : "disabled"),
              [](common_params &pParams) { pParams.vEmbedding = true; })
              .mSetEnv("MGS_ARG_EMBEDDING"));
  fAddOpt(common_arg({"--rerank"},
                     fStringFormat("enable reranking endpoint on server (default: %s)", "disabled"),
                     [](common_params &pParams) {
                       pParams.vEmbedding = true;
                       pParams.vPoolingType = MGS_POOLING_TYPE_RANK;
                     })
              .mSetEnv("MGS_ARG_RERANK"));
  fAddOpt(
      common_arg(
          {"--api-key"}, "KEY",
          "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
          [](common_params &pParams, const std::string &pValue) {
            for (const auto &vKey : fParseCsvRow(pValue)) {
              if (!vKey.empty()) {
                pParams.lApiKeys.push_back(vKey);
              }
            }
          })
          .mSetEnv("MGS_ARG_API_KEY"));
  fAddOpt(common_arg({"--api-key-file"}, "FNAME",
                     "path to file containing API keys, one per line; lines starting with a hash are treated as "
                     "comments (default: none)",
                     [](common_params &pParams, const std::string &pValue) {
                       std::ifstream vKeyFile(pValue);
                       if (!vKeyFile) {
                         throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pValue.c_str()));
                       }
                       std::string vKey;
                       while (std::getline(vKeyFile, vKey)) {
                         if (!vKey.empty() && vKey[0] != '#') {
                           pParams.lApiKeys.push_back(vKey);
                         }
                       }
                       vKeyFile.close();
                     })
              .mSetEnv("MGS_ARG_API_KEY_FILE"));
  fAddOpt(common_arg({"--ssl-key-file"}, "FNAME",
                     "PEM private key, used with --ssl-cert-file (default: the self-signed key created once in "
                     "~/.mgs/certificates/privkey.pem)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vSslFileKey = pValue; })
              .mSetEnv("MGS_ARG_SSL_KEY_FILE"));
  fAddOpt(common_arg({"--ssl-cert-file"}, "FNAME",
                     "PEM certificate, used with --ssl-key-file (default: self-signed, created once in "
                     "~/.mgs/certificates/fullchain.pem and reused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vSslFileCert = pValue; })
              .mSetEnv("MGS_ARG_SSL_CERT_FILE"));
  fAddOpt(common_arg({"--chat-template-kwargs"}, "STRING",
                     "sets additional params for the json template parser, must be a valid json object string, e.g. "
                     "'{\"key1\":\"value1\",\"key2\":\"value2\"}'",
                     [](common_params &pParams, const std::string &pValue) {
                       auto vParsed = json::parse(pValue);
                       for (const auto &vItem : vParsed.items()) {
                         if (vItem.key() == "enable_thinking") {
                           throw std::invalid_argument("Use --reasoning on / --reasoning off to control thinking");
                         }
                         pParams.dDefaultTemplateKwargs[vItem.key()] = vItem.value().dump();
                       }
                     })
              .mSetEnv("MGS_ARG_CHAT_TEMPLATE_KWARGS"));
  fAddOpt(common_arg({"-to", "--timeout"}, "N",
                     fStringFormat("server write timeout and router-to-model proxy timeout in seconds; reading a "
                                   "request waits at most min(N, 30) seconds per read (default: %d)",
                                   pParams.vTimeoutRead),
                     [](common_params &pParams, int pValue) {
                       pParams.vTimeoutRead = pValue;
                       pParams.vTimeoutWrite = pValue;
                     })
              .mSetEnv("MGS_ARG_TIMEOUT"));
  fAddOpt(common_arg({"--sse-ping-interval"}, "N",
                     fStringFormat("server SSE ping interval in seconds (-1 = disabled, default: %d)",
                                   pParams.vSsePingInterval),
                     [](common_params &pParams, int pValue) { pParams.vSsePingInterval = pValue; })
              .mSetEnv("MGS_ARG_SSE_PING_INTERVAL"));
  fAddOpt(
      common_arg({"--threads-http"}, "N",
                 fStringFormat("number of threads used to process HTTP requests (default: %d)", pParams.vNThreadsHttp),
                 [](common_params &pParams, int pValue) { pParams.vNThreadsHttp = pValue; })
          .mSetEnv("MGS_ARG_THREADS_HTTP"));
  fAddOpt(common_arg({"--cache-prompt"}, {"--no-cache-prompt"},
                     fStringFormat("whether to enable prompt caching (default: %s)",
                                   pParams.vCachePrompt ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCachePrompt = pValue; })
              .mSetEnv("MGS_ARG_CACHE_PROMPT"));
  fAddOpt(common_arg({"--cache-reuse"}, "N",
                     fStringFormat("min chunk size to attempt reusing from the cache via KV shifting, requires prompt "
                                   "caching to be enabled (default: %d)\n"
                                   "[(card)](https://ggml.ai/f0.png)",
                                   pParams.vNCacheReuse),
                     [](common_params &pParams, int pValue) { pParams.vNCacheReuse = pValue; })
              .mSetEnv("MGS_ARG_CACHE_REUSE"));
  fAddOpt(common_arg({"--metrics"},
                     fStringFormat("enable prometheus compatible metrics endpoint (default: %s)",
                                   pParams.vEndpointMetrics ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vEndpointMetrics = true; })
              .mSetEnv("MGS_ARG_METRICS"));
  fAddOpt(common_arg({"--props"},
                     fStringFormat("enable changing global properties via POST /props (default: %s)",
                                   pParams.vEndpointProps ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vEndpointProps = true; })
              .mSetEnv("MGS_ARG_PROPS"));
  fAddOpt(common_arg({"--slots"}, {"--no-slots"},
                     fStringFormat("expose slots monitoring endpoint (default: %s)",
                                   pParams.vEndpointSlots ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vEndpointSlots = pValue; })
              .mSetEnv("MGS_ARG_SLOTS"));
  fAddOpt(common_arg({"--slot-save-path"}, "PATH", "path to save slot kv cache (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSlotSavePath = pValue;
                       if (!fFsIsDirectory(pParams.vSlotSavePath)) {
                         throw std::invalid_argument("not a directory: " + pValue);
                       }
                       // if doesn't end with DIRECTORY_SEPARATOR, add it
                       if (!pParams.vSlotSavePath.empty() &&
                           pParams.vSlotSavePath[pParams.vSlotSavePath.size() - 1] != DIRECTORY_SEPARATOR) {
                         pParams.vSlotSavePath += DIRECTORY_SEPARATOR;
                       }
                     }));
  fAddOpt(common_arg({"--media-path"}, "PATH",
                     "directory for loading local media files; files can be accessed via file:// URLs using relative "
                     "paths (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vMediaPath = pValue;
                       if (!fFsIsDirectory(pParams.vMediaPath)) {
                         throw std::invalid_argument("not a directory: " + pValue);
                       }
                       // if doesn't end with DIRECTORY_SEPARATOR, add it
                       if (!pParams.vMediaPath.empty() &&
                           pParams.vMediaPath[pParams.vMediaPath.size() - 1] != DIRECTORY_SEPARATOR) {
                         pParams.vMediaPath += DIRECTORY_SEPARATOR;
                       }
                     }));
  fAddOpt(common_arg({"--media-allow-private-urls"},
                     "let remote media URLs in requests (image_url, input_audio, input_video) reach loopback, private, "
                     "link-local and other non-public addresses; by default only public addresses are fetched - do "
                     "not enable in untrusted environments (default: disabled)",
                     [](common_params &pParams) { pParams.vMediaAllowPrivateUrls = true; })
              .mSetEnv("MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS"));
  fAddOpt(common_arg({"--models-dir"}, "PATH", "directory containing models for the router server (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModelsDir = pValue; })
              .mSetEnv("MGS_ARG_MODELS_DIR"));
  fAddOpt(common_arg({"--models-preset"}, "PATH",
                     "path to INI file containing model presets for the router server (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModelsPreset = pValue; })
              .mSetEnv("MGS_ARG_MODELS_PRESET"));
  fAddOpt(
      common_arg({"--models-max"}, "N",
                 fStringFormat(
                     "for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)",
                     pParams.vModelsMax),
                 [](common_params &pParams, int pValue) { pParams.vModelsMax = pValue; })
          .mSetEnv("MGS_ARG_MODELS_MAX"));
  fAddOpt(common_arg({"--models-autoload"}, {"--no-models-autoload"},
                     fStringFormat("for router server, whether to automatically load models (default: %s)",
                                   pParams.vModelsAutoload ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vModelsAutoload = pValue; })
              .mSetEnv("MGS_ARG_MODELS_AUTOLOAD"));
  fAddOpt(common_arg({"--reasoning-format"}, "FORMAT",
                     "controls whether thought tags are allowed and/or extracted from the response, and in which "
                     "format they're returned; one of:\n"
                     "- none: leaves thoughts unparsed in `message.content`\n"
                     "- deepseek: puts thoughts in `message.reasoning_content`\n"
                     "(default: auto)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vReasoningFormat = fCommonReasoningFormatFromName(pValue);
                     })
              .mSetEnv("MGS_ARG_REASONING_FORMAT"));
  fAddOpt(
      common_arg({"-rea", "--reasoning"}, "[on|off|auto]",
                 "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
                 [](common_params &pParams, const std::string &pValue) {
                   if (fIsTruthy(pValue)) {
                     pParams.vEnableReasoning = 1;
                     pParams.dDefaultTemplateKwargs["enable_thinking"] = "true";
                   } else if (fIsFalsey(pValue)) {
                     pParams.vEnableReasoning = 0;
                     pParams.dDefaultTemplateKwargs["enable_thinking"] = "false";
                   } else if (fIsAutoy(pValue)) {
                     pParams.vEnableReasoning = -1;
                   } else {
                     throw std::invalid_argument(
                         fStringFormat("error: unknown value for --reasoning: '%s'\n", pValue.c_str()));
                   }
                 })
          .mSetEnv("MGS_ARG_REASONING"));
  fAddOpt(common_arg(
              {"--reasoning-budget"}, "N",
              "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
              [](common_params &pParams, int pValue) {
                if (pValue < -1) {
                  throw std::invalid_argument("invalid value");
                }
                pParams.vSampling.vReasoningBudgetTokens = pValue;
              })
              .mSetEnv("MGS_ARG_REASONING_BUDGET"));
  fAddOpt(
      common_arg(
          {"--reasoning-budget-message"}, "MESSAGE",
          "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vReasoningBudgetMessage = pValue; })
          .mSetEnv("MGS_ARG_REASONING_BUDGET_MESSAGE"));
  fAddOpt(common_arg({"--reasoning-preserve"}, {"--no-reasoning-preserve"},
                     "preserve reasoning trace in the full history, not just the last assistant message (default: "
                     "template default)\n"
                     "compatible with certain templates having 'supports_preserve_reasoning' capability\n"
                     "example: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking",
                     [](common_params &pParams, bool pValue) {
                       if (pValue) {
                         pParams.dDefaultTemplateKwargs["preserve_reasoning"] = "true";
                       } else {
                         pParams.dDefaultTemplateKwargs["preserve_reasoning"] = "false";
                       }
                     })
              .mSetEnv("MGS_ARG_REASONING_PRESERVE"));
  fAddOpt(
      common_arg({"--chat-template"}, "JINJA_TEMPLATE",
                 "set custom jinja chat template (default: template taken from model's metadata; chatml when it has "
                 "none)\n"
                 "if suffix/prefix are specified, template will be disabled\n"
                 "the name chatml selects the built-in ChatML template",
                 [](common_params &pParams, const std::string &pValue) { pParams.vChatTemplate = pValue; })
          .mSetEnv("MGS_ARG_CHAT_TEMPLATE"));
  fAddOpt(
      common_arg({"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
                 "set custom jinja chat template file (default: template taken from model's metadata; chatml when it "
                 "has none)\n"
                 "if suffix/prefix are specified, template will be disabled",
                 [](common_params &pParams, const std::string &pValue) { pParams.vChatTemplate = fReadFile(pValue); })
          .mSetEnv("MGS_ARG_CHAT_TEMPLATE_FILE"));
  fAddOpt(
      common_arg({"--skip-chat-parsing"}, {"--no-skip-chat-parsing"},
                 fStringFormat(
                     "force a pure content parser, even if a Jinja template is specified; model will output everything "
                     "in the content section, including any reasoning and/or tool calls (default: disabled)"),
                 [](common_params &pParams, bool pValue) { pParams.vForcePureContentParser = pValue; })
          .mSetEnv("MGS_ARG_SKIP_CHAT_PARSING"));
  fAddOpt(common_arg({"--prefill-assistant"}, {"--no-prefill-assistant"},
                     fStringFormat("whether to prefill the assistant's response if the last message is an assistant "
                                   "message (default: prefill enabled)\n"
                                   "when this flag is set, if the last message is an assistant message then it will be "
                                   "treated as a full message and not prefilled\n"),
                     [](common_params &pParams, bool pValue) { pParams.vPrefillAssistant = pValue; })
              .mSetEnv("MGS_ARG_PREFILL_ASSISTANT"));
  fAddOpt(
      common_arg(
          {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
          fStringFormat("how much the prompt of a request must match the prompt of a slot in order to use that "
                        "slot (default: %.2f, 0.0 = disabled)\n",
                        pParams.vSlotPromptSimilarity),
          [](common_params &pParams, const std::string &pValue) { pParams.vSlotPromptSimilarity = std::stof(pValue); }));
  fAddOpt(common_arg({"--lora-init-without-apply"},
                     fStringFormat(
                         "load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)",
                         pParams.vLoraInitWithoutApply ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vLoraInitWithoutApply = true; }));
  fAddOpt(common_arg({"--sleep-idle-seconds"}, "SECONDS",
                     fStringFormat(
                         "number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)",
                         pParams.vSleepIdleSeconds),
                     [](common_params &pParams, int pValue) {
                       if (pValue == 0 || pValue < -1) {
                         throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
                       }
                       pParams.vSleepIdleSeconds = pValue;
                     }));
  fAddOpt(common_arg({"--log-disable"}, "Log disable", [](common_params &) { fCommonLogPause(fCommonLogMain()); }));
  fAddOpt(common_arg({"--log-file"}, "FNAME", "Log to file", [](common_params &, const std::string &pValue) {
            fCommonLogSetFile(fCommonLogMain(), pValue.c_str());
          }).mSetEnv("MGS_ARG_LOG_FILE"));
  fAddOpt(
      common_arg({"--log-prompts-dir"}, "PATH",
                 "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)",
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vPathPromptsLogDir = pValue;
                   std::error_code vEc;
                   std::filesystem::create_directories(pValue, vEc);
                   if (vEc) {
                     fprintf(stderr, "warning: failed to create prompts-log-dir '%s': %s\n", pValue.c_str(),
                             vEc.message().c_str());
                   }
                 }));
  fAddOpt(common_arg({"--log-colors"}, "[on|off|auto]",
                     "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
                     "'auto' enables colors when output is to a terminal",
                     [](common_params &, const std::string &pValue) {
                       if (fIsTruthy(pValue)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsEnabled);
                       } else if (fIsFalsey(pValue)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsDisabled);
                       } else if (fIsAutoy(pValue)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsAuto);
                       } else {
                         throw std::invalid_argument(
                             fStringFormat("error: unknown value for --log-colors: '%s'\n", pValue.c_str()));
                       }
                     })
              .mSetEnv("MGS_ARG_LOG_COLORS"));
  fAddOpt(common_arg({"-v", "--verbose"},
                     "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
                     [](common_params &pParams) {
                       pParams.vVerbosity = INT_MAX;
                       fCommonLogSetVerbosityThold(INT_MAX);
                     }));
  fAddOpt(common_arg({"--offline"}, "Offline mode: forces use of cache, prevents network access",
                     [](common_params &pParams) { pParams.vOffline = true; })
              .mSetEnv("MGS_ARG_OFFLINE"));
  fAddOpt(common_arg(
              {"-lv", "--verbosity"}, "N",
              fStringFormat("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
                            " - 0: generic output\n"
                            " - 1: error\n"
                            " - 2: warning\n"
                            " - 3: info\n"
                            " - 4: trace (more info)\n"
                            " - 5: debug\n"
                            "(default: %d)\n",
                            pParams.vVerbosity),
              [](common_params &pParams, int pValue) {
                pParams.vVerbosity = pValue;
                fCommonLogSetVerbosityThold(pValue);
              })
              .mSetEnv("MGS_ARG_VERBOSITY"));
  fAddOpt(common_arg({"--log-prefix"}, {"--no-log-prefix"}, "Enable prefix in log messages",
                     [](common_params &, bool pValue) { fCommonLogSetPrefix(fCommonLogMain(), pValue); })
              .mSetEnv("MGS_ARG_LOG_PREFIX"));
  fAddOpt(common_arg({"--log-timestamps"}, {"--no-log-timestamps"}, "Enable timestamps in log messages",
                     [](common_params &, bool pValue) { fCommonLogSetTimestamps(fCommonLogMain(), pValue); })
              .mSetEnv("MGS_ARG_LOG_TIMESTAMPS"));

  //
  // speculative parameters
  //

  fAddOpt(common_arg({"--spec-draft-hf"}, "<user>/<model>[:quant]",
                     "Same as --hf-repo, but for the draft model (default: unused)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vMparams.vHfRepo = pValue;
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_HF"));
  fAddOpt(common_arg({"--spec-draft-threads"}, "N",
                     "number of threads to use during generation (default: same as --threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparams.vNThreads = pValue;
                       if (pParams.vSpeculative.vDraft.vCpuparams.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparams.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-threads-batch"}, "N",
                     "number of threads to use during batch and prompt processing (default: same as "
                     "--spec-draft-threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = pValue;
                       if (pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-cpu-mask"}, "M",
                     "Draft model CPU affinity mask. Complements --spec-draft-cpu-range (default: same as --cpu-mask)",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-cpu-range"}, "lo-hi",
                     "Ranges of CPUs for affinity. Complements --spec-draft-cpu-mask",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     })
              .mSetSpec());
  fAddOpt(
      common_arg({"--spec-draft-cpu-strict"}, "<0|1>",
                 "Use strict CPU placement for draft model (default: same as --cpu-strict)",
                 [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparams.vStrictCpu = pValue; })
          .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-prio"}, "N",
                     fStringFormat(
                         "set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                         pParams.vSpeculative.vDraft.vCpuparams.vPriority),
                     [](common_params &pParams, int pPrio) {
                       if (pPrio < 0 || pPrio > 3) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vDraft.vCpuparams.vPriority = (enum ggml_sched_priority)pPrio;
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-poll"}, "<0|1>",
                     "Use polling to wait for draft model work (default: same as --poll)",
                     [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparams.vPoll = pValue; })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-cpu-mask-batch"}, "M",
                     "Draft model CPU affinity mask. Complements --spec-draft-cpu-range-batch (default: same as "
                     "--cpu-mask)",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-cpu-strict-batch"}, "<0|1>",
                     "Use strict CPU placement for draft model (default: --spec-draft-cpu-strict)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vStrictCpu = pValue;
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-prio-batch"}, "N",
                     fStringFormat(
                         "set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                         pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority),
                     [](common_params &pParams, int pPrio) {
                       if (pPrio < 0 || pPrio > 3) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority = (enum ggml_sched_priority)pPrio;
                     })
              .mSetSpec());
  fAddOpt(
      common_arg({"--spec-draft-poll-batch"}, "<0|1>",
                 "Use polling to wait for draft model work (default: --spec-draft-poll)",
                 [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparamsBatch.vPoll = pValue; })
          .mSetSpec());
  fAddOpt(
      common_arg({"--spec-draft-type-k"}, "TYPE",
                 fStringFormat("KV cache data type for K for the draft model\n"
                               "allowed values: %s\n"
                               "(default: chosen from the weights of the draft model: f16 for unquantized weights, "
                               "q8_0 for more than 5 bits per weight, q4_0 for the rest, on every backend; f16 when "
                               "Flash Attention cannot read that type)",
                               fGetAllKvCacheTypes().c_str()),
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vSpeculative.vDraft.vCacheTypeK = fKvCacheTypeFromStr(pValue);
                   pParams.vSpeculative.vDraft.vCacheTypeExplicit = true;
                 })
          .mSetEnv("MGS_ARG_SPEC_DRAFT_TYPE_K"));
  fAddOpt(
      common_arg({"--spec-draft-type-v"}, "TYPE",
                 fStringFormat("KV cache data type for V for the draft model\n"
                               "allowed values: %s\n"
                               "(default: chosen from the weights of the draft model: f16 for unquantized weights, "
                               "q8_0 for more than 5 bits per weight, q4_0 for the rest, on every backend; f16 when "
                               "Flash Attention cannot read that type)",
                               fGetAllKvCacheTypes().c_str()),
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vSpeculative.vDraft.vCacheTypeV = fKvCacheTypeFromStr(pValue);
                   pParams.vSpeculative.vDraft.vCacheTypeExplicit = true;
                 })
          .mSetEnv("MGS_ARG_SPEC_DRAFT_TYPE_V"));
  fAddOpt(common_arg({"--spec-draft-override-tensor"}, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type for draft model",
                     [](common_params &pParams, const std::string &pValue) {
                       fParseTensorBufferOverrides(pValue, pParams.vSpeculative.vDraft.lTensorBuftOverrides);
                     })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-draft-cpu-moe"},
                     "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
                     [](common_params &pParams) {
                       pParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back(fLlmFfnExpsCpuOverride());
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_CPU_MOE"));
  fAddOpt(common_arg({"--spec-draft-n-cpu-moe"}, "N",
                     "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       for (int vI = 0; vI < pValue; ++vI) {
                         static std::list<std::string> lBuftOverridesDraft;
                         lBuftOverridesDraft.push_back(fLlmFfnExpsBlockRegex(vI));
                         pParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back(
                             {lBuftOverridesDraft.back().c_str(), ggml_backend_cpu_buffer_type()});
                       }
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_CPU_MOE"));

  fAddOpt(common_arg({"--spec-draft-n-max"}, "N",
                     fStringFormat("number of tokens to draft for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMax),
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vNMax = pValue;
                       pParams.vSpeculative.vDraft.vNMaxExplicit = true;
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_MAX"));
  fAddOpt(common_arg({"--spec-draft-n-min"}, "N",
                     fStringFormat("minimum number of draft tokens to use for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMin),
                     [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vNMin = pValue; })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_MIN"));

  fAddOpt(common_arg({"--spec-draft-p-split"}, "P",
                     fStringFormat("speculative decoding split probability (default: %.2f)",
                                   (double)pParams.vSpeculative.vDraft.vPSplit),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vPSplit = std::stof(pValue);
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_P_SPLIT"));
  fAddOpt(common_arg({"--spec-draft-p-min"}, "P",
                     fStringFormat("minimum speculative decoding probability (greedy) (default: %.2f)",
                                   (double)pParams.vSpeculative.vDraft.vPMin),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vPMin = std::stof(pValue);
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_P_MIN"));
  fAddOpt(common_arg({"--spec-draft-backend-sampling"}, {"--no-spec-draft-backend-sampling"},
                     fStringFormat("offload draft sampling to the backend (default: %s)",
                                   pParams.vSpeculative.vDraft.vBackendSampling ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vSpeculative.vDraft.vBackendSampling = pValue; })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
  fAddOpt(common_arg({"--spec-draft-device"}, "<dev1,dev2,..>",
                     "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
                     "use --list-devices to see a list of available devices",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.lDevices = fParseDeviceList(pValue);
                     })
              .mSetSpec());
  GGML_ASSERT(pParams.vSpeculative.vDraft.vNGpuLayers <
              0); // string_format would need to be extended for a default >= 0
  fAddOpt(common_arg({"--spec-draft-ngl"}, "N",
                     fStringFormat("max. number of draft model layers to store in VRAM, either an exact number, "
                                   "'auto', or 'all' (default: %s)",
                                   pParams.vSpeculative.vDraft.vNGpuLayers == -1 ? "auto" : "all"),
                     [](common_params &pParams, const std::string &pValue) {
                       if (pValue == "auto") {
                         pParams.vSpeculative.vDraft.vNGpuLayers = -1;
                       } else if (pValue == "all") {
                         pParams.vSpeculative.vDraft.vNGpuLayers = -2;
                       } else {
                         pParams.vSpeculative.vDraft.vNGpuLayers = std::stoi(pValue);
                       }
                       if (!mgs_supports_gpu_offload()) {
                         fprintf(stderr, "warning: no usable GPU found, --spec-draft-ngl option will be ignored\n");
                         fprintf(stderr, "warning: one possible reason is that mgs was compiled without GPU support\n");
                         fprintf(stderr, "warning: see doc/MANUAL.md for the build instructions\n");
                       }
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_NGL"));
  fAddOpt(common_arg({"--spec-draft-model"}, "FNAME", "draft model for speculative decoding (default: unused)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vMparams.vPath = pValue;
                       pParams.vSpeculative.vDraft.vMparams.vHfFile = pValue; // will be used if --spec-draft-hf is set
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_DRAFT_MODEL"));
  fAddOpt(common_arg({"--spec-type"}, fCommonSpeculativeAllTypesStr(),
                     fStringFormat("comma-separated list of types of speculative decoding to use (default: %s; "
                                   "overrides automatic MTP detection)\n",
                                   fCommonSpeculativeTypeNameStr(pParams.vSpeculative.lTypes).c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       const auto cTypesStr = fStringSplit<std::string>(pValue, ',');
                       auto vTypes = fCommonSpeculativeTypesFromNames(cTypesStr);
                       pParams.vSpeculative.lTypes.insert(pParams.vSpeculative.lTypes.end(), vTypes.begin(),
                                                          vTypes.end());
                       pParams.vSpeculative.vAutoMtp = false;
                     })
              .mSetSpec()
              .mSetEnv("MGS_ARG_SPEC_TYPE"));
  fAddOpt(common_arg(
              {"--spec-ngram-mod-n-min"}, "N",
              fStringFormat("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                            pParams.vSpeculative.vNgramMod.vNMin),
              [](common_params &pParams, int pValue) {
                if (pValue < 0 || pValue > 1024) {
                  throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMod.vNMin = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg(
              {"--spec-ngram-mod-n-max"}, "N",
              fStringFormat("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                            pParams.vSpeculative.vNgramMod.vNMax),
              [](common_params &pParams, int pValue) {
                if (pValue < 0 || pValue > 1024) {
                  throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMod.vNMax = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-ngram-mod-n-match"}, "N",
                     fStringFormat("ngram-mod lookup length (default: %d)", pParams.vSpeculative.vNgramMod.vNMatch),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 1024) {
                         throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                       }
                       pParams.vSpeculative.vNgramMod.vNMatch = pValue;
                     })
              .mSetSpec());

  fAddOpt(common_arg(
              {"--spec-ngram-simple-size-n"}, "N",
              fStringFormat("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)",
                            pParams.vSpeculative.vNgramSimple.vSizeN),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramSimple.vSizeN = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg(
              {"--spec-ngram-simple-size-m"}, "N",
              fStringFormat("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramSimple.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramSimple.vSizeM = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-ngram-simple-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-simple speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramSimple.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramSimple.vMinHits = pValue;
                     })
              .mSetSpec());

  fAddOpt(common_arg(
              {"--spec-ngram-map-k-size-n"}, "N",
              fStringFormat("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK.vSizeN),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK.vSizeN = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg(
              {"--spec-ngram-map-k-size-m"}, "N",
              fStringFormat("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK.vSizeM = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-ngram-map-k-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-map-k speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK.vMinHits = pValue;
                     })
              .mSetSpec());

  fAddOpt(common_arg({"--spec-ngram-map-k4v-size-n"}, "N",
                     fStringFormat(
                         "ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)",
                         pParams.vSpeculative.vNgramMapK4v.vSizeN),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 1024) {
                         throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                       }
                       pParams.vSpeculative.vNgramMapK4v.vSizeN = pValue;
                     })
              .mSetSpec());
  fAddOpt(common_arg(
              {"--spec-ngram-map-k4v-size-m"}, "N",
              fStringFormat("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK4v.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK4v.vSizeM = pValue;
              })
              .mSetSpec());
  fAddOpt(common_arg({"--spec-ngram-map-k4v-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-map-k4v speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK4v.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK4v.vMinHits = pValue;
                     })
              .mSetSpec());

  return vCtxArg;
}

void fCommonParamsAddPresetOptions(std::vector<common_arg> &pArgs) {
  // arguments below won't be treated as CLI args, only preset options
  pArgs.push_back(common_arg({"load-on-startup"}, "NAME", "in server router mode, autoload this model on startup",
                             [](common_params &, const std::string &) { /* unused */ })
                      .mSetEnv(COMMON_ARG_PRESET_LOAD_ON_STARTUP)
                      .mSetPresetOnly());

  pArgs.push_back(
      common_arg({"stop-timeout"}, "SECONDS",
                 "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
                 [](common_params &, int) { /* unused */ })
          .mSetEnv(COMMON_ARG_PRESET_STOP_TIMEOUT)
          .mSetPresetOnly());

  // pArgs.push_back(common_arg(
  //     {"pin"},
  //     "in server router mode, do not unload this model if models_max is exceeded",
  //     [](common_params &) { /* unused */ }
  // ).mSetPresetOnly());
}
