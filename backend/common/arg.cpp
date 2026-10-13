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

static std::initializer_list<enum mgs_example> lMmprojExamples = {
    cMgsExampleMtmd,
    cMgsExampleServer,
    cMgsExampleCli,
};

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
    auto vCtx = fCommonParamsParserInit(vParams, cMgsExampleServer, nullptr);
    return vCtx.lOptions;
  }();
  return lOptions;
}

common_arg &common_arg::mSetExamples(std::initializer_list<enum mgs_example> pExamples) {
  this->sExamples = pExamples;
  return *this;
}

common_arg &common_arg::mSetExcludes(std::initializer_list<enum mgs_example> pExcludes) {
  this->sExcludes = pExcludes;
  return *this;
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

bool common_arg::mInExample(enum mgs_example pEx) { return sExamples.find(pEx) != sExamples.end(); }

bool common_arg::mIsExclude(enum mgs_example pEx) { return sExcludes.find(pEx) != sExcludes.end(); }

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
  auto vAddLine = [&](const std::string &pL) {
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
    vAddLine(vLine);
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

common_models_handler fCommonModelsHandlerInit(const common_params &pParams, mgs_example pCurrEx) {
  common_download_hf_plan vPlan;
  common_download_hf_plan vPlanSpec;
  common_download_hf_plan vPlanVoc;
  common_download_opts vOpts;

  const bool cSpecTypeDraftMtp = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                           cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftDflash = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftDflash) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftEagle3 = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftEagle3) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftDspark = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftDspark) != pParams.vSpeculative.lTypes.end();

  // only download mmproj if the current example is using it
  bool vUseMmproj = false;
  for (const auto &vEx : lMmprojExamples) {
    if (pCurrEx == vEx) {
      vUseMmproj = true;
      break;
    }
  }

  vOpts.vBearerToken = pParams.hf_token;
  vOpts.vOffline = pParams.vOffline;
  vOpts.vDownloadMtp = cSpecTypeDraftMtp;
  vOpts.vDownloadEagle3 = cSpecTypeDraftEagle3;
  vOpts.vDownloadDflash = cSpecTypeDraftDflash;
  vOpts.vDownloadDspark = cSpecTypeDraftDspark;
  vOpts.vDownloadMmproj =
      vUseMmproj && !pParams.vNoMmproj && pParams.vMmproj.vPath.empty() && pParams.vMmproj.vUrl.empty();

  if (!pParams.vModel.vHfRepo.empty()) {
    vPlan = fCommonDownloadGetHfPlan(pParams.vModel, vOpts);
  }

  if (!pParams.vSpeculative.vDraft.vMparams.vHfRepo.empty()) {
    // without a requested type, discover every sidecar the draft repo ships to infer the type later
    auto vOptsSpec = vOpts;
    if (fSpecTypesIsDefault(pParams)) {
      vOptsSpec.vDownloadMtp = true;
      vOptsSpec.vDownloadDflash = true;
      vOptsSpec.vDownloadEagle3 = true;
      vOptsSpec.vDownloadDspark = true;
    }
    vPlanSpec = fCommonDownloadGetHfPlan(pParams.vSpeculative.vDraft.vMparams, vOptsSpec);
  }

  if (!pParams.vVocoder.vModel.vHfRepo.empty()) {
    vPlanVoc = fCommonDownloadGetHfPlan(pParams.vVocoder.vModel, vOpts);
  }

  return common_models_handler{vPlan, vPlanSpec, vPlanVoc, vOpts};
}

bool fCommonModelsHandlerIsPresetRepo(const common_models_handler &pHandler) {
  return !pHandler.vPlan.vPreset.vUrl.empty();
}

static std::vector<common_download_task> fBuildUrlTasks(const common_params_model &pModel, common_download_opts pOpts) {
  auto vParts = fCommonDownloadGetAllParts(pModel.vUrl);
  std::vector<common_download_task> lTasks;

  // single-part: download straight to model.path if the user gave one (-m), else the cache default
  if (vParts.size() == 1) {
    common_download_task vTask;
    vTask.vUrl = vParts[0];
    vTask.vLocalPath = pModel.vPath.empty() ? fGetDefaultLocalPath(vParts[0]) : pModel.vPath;
    vTask.vOpts = pOpts;
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
    vTask.vOpts = pOpts;

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
  auto &vPlanVoc = pHandler.vPlanVoc;

  auto vOpts = pHandler.vOpts; // copy
  vOpts.vCallback = pCallback;

  // handle plain "url" if needed
  auto vHandleUrl = [&](common_params_model &pModel) {
    if (!pModel.vUrl.empty()) {
      if (pModel.vPath.empty()) {
        pModel.vPath = fGetDefaultLocalPath(pModel.vUrl);
      }
    }
  };
  vHandleUrl(pParams.vModel);
  vHandleUrl(pParams.vMmproj);
  vHandleUrl(pParams.vVocoder.vModel);
  vHandleUrl(pParams.vSpeculative.vDraft.vMparams);

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
    vTask.vOpts = vOpts;
    lTasks.push_back(vTask);
  }
  if (!pParams.vVocoder.vModel.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl = pParams.vVocoder.vModel.vUrl;
    vTask.vLocalPath = pParams.vVocoder.vModel.vPath;
    vTask.vOpts = vOpts;
    lTasks.push_back(vTask);
  }

  bool vHadSpecUrl = false;
  if (!pParams.vSpeculative.vDraft.vMparams.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl = pParams.vSpeculative.vDraft.vMparams.vUrl;
    vTask.vLocalPath = pParams.vSpeculative.vDraft.vMparams.vPath;
    vTask.vOpts = vOpts;
    lTasks.push_back(vTask);
    vHadSpecUrl = true;
  }

  // handle hf_plan tasks
  auto vAddTasks = [&vOpts, &lTasks](const hf_cache::hf_files &pModelFiles, const hf_cache::vHfFile &pPrimary,
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
    vPlanSpec.vDflash = {};
    vPlanSpec.vEagle3 = {};
    vPlanSpec.vDspark = {};
  }

  // infer the speculative type from the sidecar shipped by the draft repo when none is requested
  if (fSpecTypesIsDefault(pParams)) {
    if (!vPlanSpec.vMtp.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftMtp};
      vPlanSpec.vDspark = {};
      vPlanSpec.vDflash = {};
      vPlanSpec.vEagle3 = {};
    } else if (!vPlanSpec.vDspark.vLocalPath.empty()) {
      // dspark outranks dflash, its sidecar carries the extra Markov head
      pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftDspark};
      vPlanSpec.vDflash = {};
      vPlanSpec.vEagle3 = {};
    } else if (!vPlanSpec.vDflash.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftDflash};
      vPlanSpec.vEagle3 = {};
    } else if (!vPlanSpec.vEagle3.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftEagle3};
    }
  }

  // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
  const bool cSpecSidecarFound = !vPlanSpec.vMtp.vLocalPath.empty() || !vPlanSpec.vDflash.vLocalPath.empty() ||
                                 !vPlanSpec.vEagle3.vLocalPath.empty() || !vPlanSpec.vDspark.vLocalPath.empty();
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
  if (!vPlanSpec.vDflash.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vDflash, vOpts, [&]() {
      // only use the discovered DFlash sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vDflash);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vDflash);
      }
    });
  }
  if (!vPlanSpec.vEagle3.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vEagle3, vOpts, [&]() {
      // only use the discovered Eagle3 sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vEagle3);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vEagle3);
      }
    });
  }
  if (!vPlanSpec.vDspark.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vDspark, vOpts, [&]() {
      // only use the discovered DSpark sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vDspark);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vDspark);
      }
    });
  }

  // a wired draft sidecar counts as an explicit draft for the main plan fallback below
  if (cSpecSidecarFound) {
    vHadSpecUrl = true;
  }

  // handle plan_spec (e.g. --spec-draft-hf)
  if (!vPlanSpec.vModelFiles.empty() && !vHadSpecUrl && !cSpecSidecarFound) {
    vAddTasks(vPlanSpec.vModelFiles, vPlanSpec.vPrimary, pParams.vSpeculative.vDraft.vMparams);
    vHadSpecUrl = true;
  }

  // handle vocoder plan (e.g. --hf-repo-v)
  if (!vPlanVoc.vModelFiles.empty()) {
    vAddTasks(vPlanVoc.vModelFiles, vPlanVoc.vPrimary, pParams.vVocoder.vModel);
  }

  if (!vPlan.vModelFiles.empty()) {
    vAddTasks(vPlan.vModelFiles, vPlan.vPrimary, pParams.vModel);
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
  if (!vPlan.vDflash.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vDflash, vOpts, [&]() {
      // only fall back to the discovered DFlash sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vDflash);
      } else {
        hf_cache::fFinalizeFile(vPlan.vDflash);
      }
    });
  }
  if (!vPlan.vEagle3.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vEagle3, vOpts, [&]() {
      // only fall back to the discovered Eagle3 sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vEagle3);
      } else {
        hf_cache::fFinalizeFile(vPlan.vEagle3);
      }
    });
  }
  if (!vPlan.vDspark.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vDspark, vOpts, [&]() {
      // only fall back to the discovered DSpark sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vDspark);
      } else {
        hf_cache::fFinalizeFile(vPlan.vDspark);
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
  auto vCheckArg = [&](int pI) {
    if (pI + 1 >= pArgc) {
      throw std::invalid_argument("expected value for argument");
    }
  };

  auto vParseCliArgs = [&]() {
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
        vCheckArg(vI);
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
        vCheckArg(vI);
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
  vParseCliArgs();

  fPostprocessCpuParams(vParams.vCpuparams, nullptr);
  fPostprocessCpuParams(vParams.vCpuparamsBatch, &vParams.vCpuparams);

  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparams, &vParams.vCpuparams);
  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparamsBatch, &vParams.vCpuparamsBatch);

  if (vParams.vPromptCacheAll && (vParams.vInteractive || vParams.vInteractiveFirst)) {
    throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
  }

  const bool cSkipModelDownload =
      // server will call common_params_handle_models() later, so we skip it here
      pCtxArg.vEx == cMgsExampleServer ||
      // download calls common_params_handle_models() itself and prints the paths
      pCtxArg.vEx == cMgsExampleDownload ||
      // export_graph_ops loads only metadata
      pCtxArg.vEx == cMgsExampleExportGraphOps;

  if (!cSkipModelDownload) {
    // handle model and download
    common_models_handler vHandler = fCommonModelsHandlerInit(vParams, pCtxArg.vEx);
    fCommonModelsHandlerApply(vHandler, vParams);

    // model is required (except for server)
    // TODO @ngxson : maybe show a list of available models in CLI in this case
    bool vCanSkipModel = vParams.vUsage || vParams.vCompletion || !vParams.vServerBase.empty();
    if (!vCanSkipModel && vParams.vModel.vPath.empty()) {
      throw std::invalid_argument("error: --model is required\n");
    }
  }

  if (vParams.vEscape) {
    fStringProcessEscapes(vParams.prompt);
    fStringProcessEscapes(vParams.pInputPrefix);
    fStringProcessEscapes(vParams.pInputSuffix);
    for (auto &vAntiprompt : vParams.lAntiprompt) {
      fStringProcessEscapes(vAntiprompt);
    }
    for (auto &vSeqBreaker : vParams.vSampling.lDrySequenceBreakers) {
      fStringProcessEscapes(vSeqBreaker);
    }
  }

  if (!vParams.lKvOverrides.empty()) {
    vParams.lKvOverrides.emplace_back();
    vParams.lKvOverrides.back().key[0] = 0;
  }

  const bool cMcpEnabled =
      !vParams.vMcpServersConfig.empty() || !vParams.vMcpServersJson.empty() || vParams.vShellMcpServer;
  if ((!vParams.lServerTools.empty() || cMcpEnabled) && !vParams.vCorsOriginsExplicit) {
    LOG_WRN("server tools or MCP servers are enabled, using localhost as default CORS origin (change via "
            "--cors-origins)\n");
    vParams.vCorsOrigins = "localhost";
  }

  // pad tensor_buft_overrides for mgs_params_fit:
  const size_t cNtbo = mgs_max_tensor_buft_overrides();
  while (vParams.lTensorBuftOverrides.size() < cNtbo) {
    vParams.lTensorBuftOverrides.push_back({nullptr, nullptr});
  }

  if (!vParams.vSpeculative.vDraft.lTensorBuftOverrides.empty()) {
    vParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back({nullptr, nullptr});
  }

  if (!vParams.chat_template.empty() && !fCommonChatVerifyTemplate(vParams.chat_template)) {
    throw std::runtime_error(
        fStringFormat("error: the supplied chat template is not supported: %s\n", vParams.chat_template.c_str()));
  }

  return true;
}

static void fCommonParamsPrintUsage(common_params_context &pCtxArg) {
  auto vPrintOptions = [](std::vector<common_arg *> &pOptions) {
    for (common_arg *vOpt : pOptions) {
      printf("%s", vOpt->mToString().c_str());
    }
  };

  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;
  std::vector<common_arg *> lSpecificOptions;
  for (auto &vOpt : pCtxArg.lOptions) {
    // in case multiple MGS_EXAMPLE_* are set, we prioritize the MGS_EXAMPLE_* matching current example
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else if (vOpt.mInExample(pCtxArg.vEx)) {
      lSpecificOptions.push_back(&vOpt);
    } else {
      lCommonOptions.push_back(&vOpt);
    }
  }
  bool vFirst = true;
  auto vPrintSection = [&](const char *pHeader, std::vector<common_arg *> &pOptions) {
    if (pOptions.empty()) {
      return;
    }
    printf("%s----- %s -----\n\n", vFirst ? "" : "\n\n", pHeader);
    vFirst = false;
    vPrintOptions(pOptions);
  };
  vPrintSection("common params", lCommonOptions);
  vPrintSection("sampling params", lSamplingOptions);
  vPrintSection("speculative params", lSpecOptions);
  vPrintSection("example-specific params", lSpecificOptions);
}

static void fCommonParamsPrintCompletion(common_params_context &pCtxArg) {
  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;
  std::vector<common_arg *> lSpecificOptions;

  for (auto &vOpt : pCtxArg.lOptions) {
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else if (vOpt.mInExample(pCtxArg.vEx)) {
      lSpecificOptions.push_back(&vOpt);
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
  auto vPrintOptions = [](const std::vector<common_arg *> &pOptions) {
    for (const common_arg *vOpt : pOptions) {
      for (const char *vArg : vOpt->lArgs) {
        printf("%s ", vArg);
      }
    }
  };

  vPrintOptions(lCommonOptions);
  vPrintOptions(lSamplingOptions);
  vPrintOptions(lSpecOptions);
  vPrintOptions(lSpecificOptions);
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

static void fAddRpcDevices(const std::string &pServers) {
  auto vRpcServers = fStringSplit<std::string>(pServers, ',');
  if (vRpcServers.empty()) {
    throw std::invalid_argument("no RPC servers specified");
  }
  ggml_backend_load_all();
  ggml_backend_reg_t vRpcReg = ggml_backend_reg_by_name("RPC");
  if (!vRpcReg) {
    throw std::invalid_argument("failed to find RPC backend");
  }
  typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char *pEndpoint);
  ggml_backend_rpc_add_server_t vGgmlBackendRpcAddServerFn =
      (ggml_backend_rpc_add_server_t)ggml_backend_reg_get_proc_address(vRpcReg, "ggml_backend_rpc_add_server");
  if (!vGgmlBackendRpcAddServerFn) {
    throw std::invalid_argument("failed to find RPC add server function");
  }
  for (const auto &vServer : vRpcServers) {
    auto vReg = vGgmlBackendRpcAddServerFn(vServer.c_str());
    ggml_backend_register(vReg);
  }
}

bool fCommonParamsToMap(int pArgc, char **pArgv, mgs_example pEx, std::map<common_arg, std::string> &pOutMap) {
  common_params vDummyParams;
  common_params_context vCtxArg = fCommonParamsParserInit(vDummyParams, pEx, nullptr);

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
  auto vCheckArg = [&](int pI) {
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
      vCheckArg(vI);
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

bool fCommonParamsParse(int pArgc, char **pArgv, common_params &pParams, mgs_example pEx,
                        void (*pPrintUsage)(int, char **)) {
#ifdef _WIN32
  auto vUtf8 = fMakeUtf8Argv();
  // repair argv only when it matches the process command line
  if (static_cast<int>(vUtf8.lBuf.size()) == pArgc) {
    pArgv = vUtf8.lPtrs.data();
  }
#endif

  auto vCtxArg = fCommonParamsParserInit(pParams, pEx, pPrintUsage);
  const common_params cParamsOrg = vCtxArg.vParams; // the example can modify the default params

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
    pParams.vLr.mInit();
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

common_params_context fCommonParamsParserInit(common_params &pParams, mgs_example pEx,
                                              void (*pPrintUsage)(int, char **)) {
  // per-example default params
  // we define here to make sure it's included in the generated docs
  if (pEx == cMgsExampleMtmd) {
    pParams.vSampling.vTemp = 0.2; // lower temp by default for better quality
  } else if (pEx == cMgsExampleTokenize) {
    pParams.vParseSpecial = true; // parse special tokens by default, like the old tokenize tool
  }

  pParams.vUseColor = fTtyCanUseColors();

  common_params_context vCtxArg(pParams);
  vCtxArg.vPrintUsage = pPrintUsage;
  vCtxArg.vEx = pEx;

  std::string vSamplerTypeChars;
  std::string vSamplerTypeNames;
  for (const auto &vSampler : pParams.vSampling.lSamplers) {
    vSamplerTypeChars += fCommonSamplerTypeToChr(vSampler);
    vSamplerTypeNames += fCommonSamplerTypeToStr(vSampler) + ";";
  }
  if (!vSamplerTypeNames.empty()) {
    vSamplerTypeNames.pop_back(); // remove last semicolon
  }

  /**
   * filter options by example
   * rules:
   * - all examples inherit options from MGS_EXAMPLE_COMMON
   * - if MGS_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
   * - if both {MGS_EXAMPLE_COMMON, MGS_EXAMPLE_*,} are set, we will prioritize the MGS_EXAMPLE_* matching current
   * example
   */
  auto vAddOpt = [&](common_arg pArg) {
    // download only exposes the handful of args explicitly tagged for it
    const bool cInheritCommon = pEx != cMgsExampleDownload;
    if ((pArg.mInExample(pEx) || (cInheritCommon && pArg.mInExample(cMgsExampleCommon))) && !pArg.mIsExclude(pEx)) {
      vCtxArg.lOptions.push_back(std::move(pArg));
    }
  };

  vAddOpt(common_arg({"-h", "--help"}, "print usage and exit", [](common_params &pParams) {
            pParams.vUsage = true;
          }).mSetExamples({cMgsExampleCommon, cMgsExampleDownload}));
  vAddOpt(common_arg({"--version"}, "show version and build info", [](common_params &) {
    fprintf(stderr, "version: %s\n", mgs_version());
    fprintf(stderr, "built with %s for %s\n", mgs_compiler(), mgs_build_target());
    exit(0);
  }));
  vAddOpt(common_arg({"-cl", "--cache-list"}, "show list of models in cache", [](common_params &) {
    auto vModels = fCommonListCachedModels();
    printf("number of models in cache: %zu\n", vModels.size());
    for (size_t vI = 0; vI < vModels.size(); vI++) {
      printf("%4zu. %s\n", vI + 1, vModels[vI].mToString().c_str());
    }
    exit(0);
  }));
  vAddOpt(common_arg({"--completion-bash"}, "print source-able bash completion script for mgs",
                     [](common_params &pParams) { pParams.vCompletion = true; }));
  vAddOpt(common_arg({"--server-base"}, "URL",
                     fStringFormat("connect to this server instead of starting a new one, example: "
                                   "'https://localhost:11443' (default: none)"),
                     [](common_params &pParams, const std::string &pValue) { pParams.vServerBase = pValue; })
              .mSetExamples({cMgsExampleCli}));
  vAddOpt(common_arg({"--verbose-prompt"},
                     fStringFormat("print a verbose prompt before generation (default: %s)",
                                   pParams.vVerbosePrompt ? "true" : "false"),
                     [](common_params &pParams) { pParams.vVerbosePrompt = true; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleEmbedding, cMgsExampleRetrieval}));
  vAddOpt(common_arg({"--display-prompt"}, {"--no-display-prompt"},
                     fStringFormat("whether to print prompt at generation (default: %s)",
                                   pParams.vDisplayPrompt ? "true" : "false"),
                     [](common_params &pParams, bool pValue) { pParams.vDisplayPrompt = pValue; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli}));
  vAddOpt(common_arg({"-co", "--color"}, "[on|off|auto]",
                     "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', "
                     "default: 'auto')\n"
                     "'auto' enables colors when output is to a terminal",
                     [](common_params &pParams, const std::string &pValue) {
                       if (fIsTruthy(pValue)) {
                         pParams.vUseColor = true;
                       } else if (fIsFalsey(pValue)) {
                         pParams.vUseColor = false;
                       } else if (fIsAutoy(pValue)) {
                         pParams.vUseColor = fTtyCanUseColors();
                       } else {
                         throw std::invalid_argument(
                             fStringFormat("error: unknown value for --color: '%s'\n", pValue.c_str()));
                       }
                     })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleSpeculative, cMgsExampleLookup}));
  vAddOpt(
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
  vAddOpt(common_arg({"-tb", "--threads-batch"}, "N",
                     "number of threads to use during batch and prompt processing (default: same as --threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vCpuparamsBatch.vNThreads = pValue;
                       if (pParams.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     }));
  vAddOpt(common_arg({"-C", "--cpu-mask"}, "M",
                     "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     }));
  vAddOpt(common_arg({"-Cr", "--cpu-range"}, "lo-hi", "range of CPUs for affinity. Complements --cpu-mask",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  vAddOpt(common_arg(
      {"--cpu-strict"}, "<0|1>",
      fStringFormat("use strict CPU placement (default: %u)\n", (unsigned)pParams.vCpuparams.vStrictCpu),
      [](common_params &pParams, const std::string &pValue) { pParams.vCpuparams.vStrictCpu = std::stoul(pValue); }));
  vAddOpt(common_arg(
      {"--prio"}, "N",
      fStringFormat("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n",
                    pParams.vCpuparams.vPriority),
      [](common_params &pParams, int pPrio) {
        if (pPrio < GGML_SCHED_PRIO_LOW || pPrio > GGML_SCHED_PRIO_REALTIME) {
          throw std::invalid_argument("invalid value");
        }
        pParams.vCpuparams.vPriority = (enum ggml_sched_priority)pPrio;
      }));
  vAddOpt(common_arg(
      {"--poll"}, "<0...100>",
      fStringFormat("use polling level to wait for work (0 - no polling, default: %u)\n",
                    (unsigned)pParams.vCpuparams.vPoll),
      [](common_params &pParams, const std::string &pValue) { pParams.vCpuparams.vPoll = std::stoul(pValue); }));
  vAddOpt(
      common_arg({"-Cb", "--cpu-mask-batch"}, "M",
                 "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
                 [](common_params &pParams, const std::string &pMask) {
                   pParams.vCpuparamsBatch.vMaskValid = true;
                   if (!fParseCpuMask(pMask, pParams.vCpuparamsBatch.aCpumask)) {
                     throw std::invalid_argument("invalid cpumask");
                   }
                 }));
  vAddOpt(common_arg({"-Crb", "--cpu-range-batch"}, "lo-hi",
                     "ranges of CPUs for affinity. Complements --cpu-mask-batch",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  vAddOpt(common_arg({"--cpu-strict-batch"}, "<0|1>", "use strict CPU placement (default: same as --cpu-strict)",
                     [](common_params &pParams, int pValue) { pParams.vCpuparamsBatch.vStrictCpu = pValue; }));
  vAddOpt(
      common_arg({"--prio-batch"}, "N",
                 fStringFormat("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                               pParams.vCpuparamsBatch.vPriority),
                 [](common_params &pParams, int pPrio) {
                   if (pPrio < 0 || pPrio > 3) {
                     throw std::invalid_argument("invalid value");
                   }
                   pParams.vCpuparamsBatch.vPriority = (enum ggml_sched_priority)pPrio;
                 }));
  vAddOpt(common_arg({"--poll-batch"}, "<0|1>", "use polling to wait for work (default: same as --poll)",
                     [](common_params &pParams, int pValue) { pParams.vCpuparamsBatch.vPoll = pValue; }));
  vAddOpt(common_arg({"-lcs", "--lookup-cache-static"}, "FNAME",
                     "path to static lookup cache to use for lookup decoding (not updated by generation)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheStatic = pValue;
                     })
              .mSetExamples({cMgsExampleLookup, cMgsExampleServer}));
  vAddOpt(common_arg({"-lcd", "--lookup-cache-dynamic"}, "FNAME",
                     "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheDynamic = pValue;
                     })
              .mSetExamples({cMgsExampleLookup, cMgsExampleServer}));
  vAddOpt(common_arg({"-c", "--ctx-size"}, "N",
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
  vAddOpt(common_arg(
              {"-n", "--predict"}, "N",
              fStringFormat(pEx == cMgsExampleCompletion
                                ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
                                : "number of tokens to predict (default: %d, -1 = infinity)",
                            pParams.vNPredict),
              [](common_params &pParams, int pValue) { pParams.vNPredict = pValue; })
              .mSetEnv("MGS_ARG_N_PREDICT"));
  vAddOpt(common_arg({"-b", "--batch-size"}, "N",
                     fStringFormat("logical maximum batch size (default: %d; --fit raises it with the physical batch "
                                   "size when the MoE experts stay in system memory)",
                                   pParams.vNBatch),
                     [](common_params &pParams, int pValue) {
                       pParams.vNBatch = pValue;
                       pParams.vBatchExplicit = true;
                     })
              .mSetEnv("MGS_ARG_BATCH"));
  vAddOpt(common_arg({"-ub", "--ubatch-size"}, "N",
                     fStringFormat("physical maximum batch size (default: %d; --fit raises it up to 4096 when the MoE "
                                   "experts stay in system memory)",
                                   pParams.vNUbatch),
                     [](common_params &pParams, int pValue) {
                       pParams.vNUbatch = pValue;
                       pParams.vBatchExplicit = true;
                     })
              .mSetEnv("MGS_ARG_UBATCH"));
  vAddOpt(common_arg(
      {"--keep"}, "N",
      fStringFormat("number of tokens to keep from the initial prompt (default: %d, -1 = all)", pParams.vNKeep),
      [](common_params &pParams, int pValue) { pParams.vNKeep = pValue; }));
  vAddOpt(common_arg({"--swa-full"},
                     fStringFormat("use full-size SWA cache (default: %s)", pParams.vSwaFull ? "true" : "false"),
                     [](common_params &pParams) { pParams.vSwaFull = true; })
              .mSetEnv("MGS_ARG_SWA_FULL"));
  vAddOpt(common_arg({"-ctxcp", "--ctx-checkpoints"}, "N",
                     fStringFormat("max number of context checkpoints to create per slot (default: %d)",
                                   pParams.vNCtxCheckpoints),
                     [](common_params &pParams, int pValue) { pParams.vNCtxCheckpoints = pValue; })
              .mSetEnv("MGS_ARG_CTX_CHECKPOINTS")
              .mSetExamples({cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(
      common_arg({"-cms", "--checkpoint-min-step"}, "N",
                 fStringFormat("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)",
                               pParams.vCheckpointMinStep),
                 [](common_params &pParams, int pValue) {
                   if (pValue < 0) {
                     throw std::invalid_argument("checkpoint-min-step must be non-negative");
                   }
                   pParams.vCheckpointMinStep = pValue;
                 })
          .mSetEnv("MGS_ARG_CHECKPOINT_MIN_STEP")
          .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"-cram", "--cache-ram"}, "N",
                     fStringFormat("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)",
                                   pParams.vCacheRamMib),
                     [](common_params &pParams, int pValue) { pParams.vCacheRamMib = pValue; })
              .mSetEnv("MGS_ARG_CACHE_RAM")
              .mSetExamples({cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg(
              {"-kvu", "--kv-unified"}, {"-no-kvu", "--no-kv-unified"},
              "use single unified KV buffer shared across all sequences (default: disabled)",
              [](common_params &pParams, bool pValue) { pParams.vKvUnified = pValue; })
              .mSetEnv("MGS_ARG_KV_UNIFIED")
              .mSetExamples({cMgsExampleServer, cMgsExamplePerplexity, cMgsExampleBatched, cMgsExampleBench,
                             cMgsExampleParallel}));
  vAddOpt(common_arg({"--cache-idle-slots"}, {"--no-cache-idle-slots"},
                     "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: "
                     "enabled, requires cache-ram)",
                     [](common_params &pParams, bool pValue) { pParams.vCacheIdleSlots = pValue; })
              .mSetEnv("MGS_ARG_CACHE_IDLE_SLOTS")
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--context-shift"}, {"--no-context-shift"},
                     fStringFormat("whether to use context shift on infinite text generation (default: %s)",
                                   pParams.vCtxShift ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCtxShift = pValue; })
              .mSetExamples(
                  {cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer, cMgsExampleImatrix, cMgsExamplePerplexity})
              .mSetEnv("MGS_ARG_CONTEXT_SHIFT"));
  vAddOpt(common_arg({"--chunks"}, "N",
                     fStringFormat("max number of chunks to process (default: %d, -1 = all)", pParams.vNChunks),
                     [](common_params &pParams, int pValue) { pParams.vNChunks = pValue; })
              .mSetExamples({cMgsExampleImatrix, cMgsExamplePerplexity, cMgsExampleRetrieval}));
  vAddOpt(common_arg({"-fa", "--flash-attn"}, "[on|off|auto]",
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
  vAddOpt(common_arg({"-p", "--prompt"}, "PROMPT", "prompt to start generation with; for system message, use -sys",
                     [](common_params &pParams, const std::string &pValue) { pParams.prompt = pValue; })
              .mSetExcludes({cMgsExampleServer}));
  vAddOpt(common_arg({"-sys", "--system-prompt"}, "PROMPT",
                     "system prompt to use with model (if applicable, depending on chat template)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vSystemPrompt = pValue; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleDiffusion, cMgsExampleMtmd}));
  vAddOpt(common_arg({"--perf"}, {"--no-perf"},
                     fStringFormat("whether to enable internal mgs performance timings (default: %s)",
                                   pParams.vNoPerf ? "true" : "false"),
                     [](common_params &pParams, bool pValue) {
                       pParams.vNoPerf = !pValue;
                       pParams.vSampling.vNoPerf = !pValue;
                     })
              .mSetEnv("MGS_ARG_PERF"));
  vAddOpt(common_arg({"--show-timings"}, {"--no-show-timings"},
                     fStringFormat("whether to show timing information after each response (default: %s)",
                                   pParams.vShowTimings ? "true" : "false"),
                     [](common_params &pParams, bool pValue) { pParams.vShowTimings = pValue; })
              .mSetExamples({cMgsExampleCli})
              .mSetEnv("MGS_ARG_SHOW_TIMINGS"));
  vAddOpt(common_arg({"-f", "--file"}, "FNAME", "a file containing the prompt (default: none)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.prompt = fReadFile(pValue);
                       // store the external file name in params
                       pParams.prompt_file = pValue;
                       if (!pParams.prompt.empty() && pParams.prompt.back() == '\n') {
                         pParams.prompt.pop_back();
                       }
                     })
              .mSetExcludes({cMgsExampleServer}));
  vAddOpt(common_arg({"-sysf", "--system-prompt-file"}, "FNAME", "a file containing the system prompt (default: none)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSystemPrompt = fReadFile(pValue);
                       if (!pParams.vSystemPrompt.empty() && pParams.vSystemPrompt.back() == '\n') {
                         pParams.vSystemPrompt.pop_back();
                       }
                     })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleDiffusion}));
  vAddOpt(common_arg({"--in-file"}, "FNAME", "an input file (use comma-separated values to specify multiple files)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         std::ifstream vFile(vItem);
                         if (!vFile) {
                           throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", vItem.c_str()));
                         }
                         pParams.lInFiles.push_back(vItem);
                       }
                     })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"-bf", "--binary-file"}, "FNAME", "binary file containing the prompt (default: none)",
                     [](common_params &pParams, const std::string &pValue) {
                       std::ifstream vFile(pValue, std::ios::binary);
                       if (!vFile) {
                         throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pValue.c_str()));
                       }
                       // store the external file name in params
                       pParams.prompt_file = pValue;
                       std::ostringstream vSs;
                       vSs << vFile.rdbuf();
                       pParams.prompt = vSs.str();
                       fprintf(stderr, "Read %zu bytes from binary file %s\n", pParams.prompt.size(), pValue.c_str());
                     })
              .mSetExcludes({cMgsExampleServer}));
  vAddOpt(
      common_arg({"-e", "--escape"}, {"--no-escape"},
                 fStringFormat("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)",
                               pParams.vEscape ? "true" : "false"),
                 [](common_params &pParams, bool pValue) { pParams.vEscape = pValue; }));
  vAddOpt(common_arg({"-ptc", "--print-token-count"}, "N",
                     fStringFormat("print token count every N tokens (default: %d)", pParams.vNPrint),
                     [](common_params &pParams, int pValue) { pParams.vNPrint = pValue; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--prompt-cache"}, "FNAME", "file to cache prompt state for faster startup (default: none)",
                     [](common_params &pParams, const std::string &pValue) { pParams.path_prompt_cache = pValue; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--prompt-cache-all"}, "if specified, saves user input and generations to cache as well\n",
                     [](common_params &pParams) { pParams.vPromptCacheAll = true; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--prompt-cache-ro"}, "if specified, uses the prompt cache but does not update it",
                     [](common_params &pParams) { pParams.vPromptCacheRo = true; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg(
              {"-r", "--reverse-prompt"}, "PROMPT", "halt generation at PROMPT, return control in interactive mode\n",
              [](common_params &pParams, const std::string &pValue) { pParams.lAntiprompt.emplace_back(pValue); })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer}));
  vAddOpt(common_arg({"-sp", "--special"},
                     fStringFormat("special tokens output enabled (default: %s)", pParams.vSpecial ? "true" : "false"),
                     [](common_params &pParams) { pParams.vSpecial = true; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer}));
  vAddOpt(common_arg({"-cnv", "--conversation"}, {"-no-cnv", "--no-conversation"},
                     "whether to run in conversation mode:\n"
                     "- does not print special tokens and suffix/prefix\n"
                     "- interactive mode is also enabled\n"
                     "(default: auto enabled if chat template is available)",
                     [](common_params &pParams, bool pValue) {
                       pParams.vConversationMode =
                           pValue ? cCommonConversationModeEnabled : cCommonConversationModeDisabled;
                     })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli}));
  vAddOpt(common_arg({"-st", "--single-turn"},
                     "run conversation for a single turn only, then exit when done\n"
                     "will not be interactive if first turn is predefined with --prompt\n"
                     "(default: false)",
                     [](common_params &pParams) { pParams.vSingleTurn = true; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli}));
  vAddOpt(common_arg({"-i", "--interactive"},
                     fStringFormat("run in interactive mode (default: %s)", pParams.vInteractive ? "true" : "false"),
                     [](common_params &pParams) { pParams.vInteractive = true; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"-if", "--interactive-first"},
                     fStringFormat("run in interactive mode and wait for input right away (default: %s)",
                                   pParams.vInteractiveFirst ? "true" : "false"),
                     [](common_params &pParams) { pParams.vInteractiveFirst = true; })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"-mli", "--multiline-input"},
                     "allows you to write or paste multiple lines without ending each in '\\'",
                     [](common_params &pParams) { pParams.vMultilineInput = true; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli}));
  vAddOpt(common_arg({"--in-prefix-bos"}, "prefix BOS to user inputs, preceding the `--in-prefix` string",
                     [](common_params &pParams) {
                       pParams.vInputPrefixBos = true;
                       pParams.vEnableChatTemplate = false;
                     })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--in-prefix"}, "STRING", "string to prefix user inputs with (default: empty)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.pInputPrefix = pValue;
                       pParams.vEnableChatTemplate = false;
                     })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--in-suffix"}, "STRING", "string to suffix after user inputs with (default: empty)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.pInputSuffix = pValue;
                       pParams.vEnableChatTemplate = false;
                     })
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"--warmup"}, {"--no-warmup"},
                     fStringFormat("whether to perform warmup with an empty run (default: %s)",
                                   pParams.vWarmup ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vWarmup = pValue; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer, cMgsExampleMtmd,
                             cMgsExampleEmbedding, cMgsExampleRetrieval, cMgsExamplePerplexity, cMgsExampleDebug}));
  vAddOpt(common_arg({"--spm-infill"},
                     fStringFormat("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as "
                                   "some models prefer this. (default: %s)",
                                   pParams.vSpmInfill ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vSpmInfill = true; })
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg(
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
  vAddOpt(
      common_arg(
          {"-s", "--seed"}, "SEED",
          fStringFormat("RNG seed (default: %d, use random seed for %d)", pParams.vSampling.vSeed, MGS_DEFAULT_SEED),
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vSeed = std::stoul(pValue); })
          .mSetSampling());
  vAddOpt(common_arg({"--sampler-seq"}, "SEQUENCE",
                     fStringFormat("simplified sequence for samplers that will be used (default: %s)",
                                   vSamplerTypeChars.c_str()),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.lSamplers = fCommonSamplerTypesFromChars(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--ignore-eos"},
                     "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
                     [](common_params &pParams) { pParams.vSampling.vIgnoreEos = true; })
              .mSetSampling());
  vAddOpt(common_arg({"--temp"}, "N",
                     fStringFormat("temperature (default: %.2f)", (double)pParams.vSampling.vTemp),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vTemp = std::stof(pValue);
                       pParams.vSampling.vTemp = std::max(pParams.vSampling.vTemp, 0.0f);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigTemp;
                     })
              .mSetSampling());
  vAddOpt(common_arg(
              {"--top-k"}, "N", fStringFormat("top-k sampling (default: %d, 0 = disabled)", pParams.vSampling.vTopK),
              [](common_params &pParams, int pValue) {
                pParams.vSampling.vTopK = pValue;
                pParams.vSampling.vUserSamplingConfig |= common_params_sampling_config::cCommonParamsSamplingConfigTopK;
              })
              .mSetSampling()
              .mSetEnv("MGS_ARG_TOP_K"));
  vAddOpt(common_arg({"--top-p"}, "N",
                     fStringFormat("top-p sampling (default: %.2f, 1.0 = disabled)", (double)pParams.vSampling.vTopP),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vTopP = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigTopP;
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--min-p"}, "N",
                     fStringFormat("min-p sampling (default: %.2f, 0.0 = disabled)", (double)pParams.vSampling.vMinP),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMinP = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMinP;
                     })
              .mSetSampling());
  vAddOpt(
      common_arg(
          {"--top-nsigma"}, "N",
          fStringFormat("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", pParams.vSampling.vTopNSigma),
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vTopNSigma = std::stof(pValue); })
          .mSetSampling());
  vAddOpt(common_arg({"--xtc-probability"}, "N",
                     fStringFormat("xtc probability (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vXtcProbability),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vXtcProbability = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigXtcProbability;
                     })
              .mSetSampling());
  vAddOpt(common_arg(
              {"--xtc-threshold"}, "N",
              fStringFormat("xtc threshold (default: %.2f, 1.0 = disabled)", (double)pParams.vSampling.vXtcThreshold),
              [](common_params &pParams, const std::string &pValue) {
                pParams.vSampling.vXtcThreshold = std::stof(pValue);
                pParams.vSampling.vUserSamplingConfig |=
                    common_params_sampling_config::cCommonParamsSamplingConfigXtcThreshold;
              })
              .mSetSampling());
  vAddOpt(
      common_arg({"--typical"}, "N",
                 fStringFormat("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)",
                               (double)pParams.vSampling.vTypP),
                 [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vTypP = std::stof(pValue); })
          .mSetSampling());
  vAddOpt(common_arg({"--repeat-last-n"}, "N",
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
  vAddOpt(common_arg({"--repeat-penalty"}, "N",
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
  vAddOpt(common_arg({"--presence-penalty"}, "N",
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
  vAddOpt(common_arg({"--frequency-penalty"}, "N",
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
  vAddOpt(common_arg({"--dry-multiplier"}, "N",
                     fStringFormat("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vDryMultiplier),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDryMultiplier = std::stof(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--dry-base"}, "N",
                     fStringFormat("set DRY sampling base value (default: %.2f)", (double)pParams.vSampling.vDryBase),
                     [](common_params &pParams, const std::string &pValue) {
                       float vPotentialBase = std::stof(pValue);
                       if (vPotentialBase >= 1.0f) {
                         pParams.vSampling.vDryBase = vPotentialBase;
                       }
                     })
              .mSetSampling());
  vAddOpt(common_arg(
              {"--dry-allowed-length"}, "N",
              fStringFormat("set allowed length for DRY sampling (default: %d)", pParams.vSampling.vDryAllowedLength),
              [](common_params &pParams, int pValue) { pParams.vSampling.vDryAllowedLength = pValue; })
              .mSetSampling());
  vAddOpt(
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
  vAddOpt(common_arg({"--dry-sequence-breaker"}, "STRING",
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
  vAddOpt(common_arg({"--adaptive-target"}, "N",
                     fStringFormat("adaptive-p: select tokens near this probability (valid range 0.0 "
                                   "to 1.0; negative = disabled) (default: %.2f)",
                                   (double)pParams.vSampling.vAdaptiveTarget),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vAdaptiveTarget = std::stof(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--adaptive-decay"}, "N",
                     fStringFormat("adaptive-p: decay rate for target adaptation over time. lower values "
                                   "are more reactive, higher values are more stable.\n"
                                   "(valid range 0.0 to 0.99) (default: %.2f)",
                                   (double)pParams.vSampling.vAdaptiveDecay),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vAdaptiveDecay = std::stof(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--dynatemp-range"}, "N",
                     fStringFormat("dynamic temperature range (default: %.2f, 0.0 = disabled)",
                                   (double)pParams.vSampling.vDynatempRange),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDynatempRange = std::stof(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--dynatemp-exp"}, "N",
                     fStringFormat("dynamic temperature exponent (default: %.2f)",
                                   (double)pParams.vSampling.vDynatempExponent),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vDynatempExponent = std::stof(pValue);
                     })
              .mSetSampling());
  vAddOpt(common_arg(
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
  vAddOpt(common_arg({"--mirostat-lr"}, "N",
                     fStringFormat("Mirostat learning rate, parameter eta (default: %.2f)",
                                   (double)pParams.vSampling.vMirostatEta),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMirostatEta = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMirostatEta;
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--mirostat-ent"}, "N",
                     fStringFormat("Mirostat target entropy, parameter tau (default: %.2f)",
                                   (double)pParams.vSampling.vMirostatTau),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vMirostatTau = std::stof(pValue);
                       pParams.vSampling.vUserSamplingConfig |=
                           common_params_sampling_config::cCommonParamsSamplingConfigMirostatTau;
                     })
              .mSetSampling());
  vAddOpt(common_arg({"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
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
  vAddOpt(common_arg({"--grammar"}, "GRAMMAR",
                     "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeUser, pValue};
                     })
              .mSetSampling());
  vAddOpt(common_arg({"--grammar-file"}, "FNAME", "file to read grammar from",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeUser, fReadFile(pValue)};
                     })
              .mSetSampling());
  vAddOpt(common_arg({"-j", "--json-schema"}, "SCHEMA",
                     "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON "
                     "object\nFor schemas w/ external $refs, convert them to a GBNF grammar and pass it with "
                     "--grammar or --grammar-file instead",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSampling.vGrammar = {cCommonGrammarTypeOutputFormat,
                                                     fJsonSchemaToGrammar(json::parse(pValue))};
                     })
              .mSetSampling());
  vAddOpt(common_arg(
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
  vAddOpt(common_arg({"-bs", "--backend-sampling"}, "enable backend sampling (experimental) (default: disabled)",
                     [](common_params &pParams) { pParams.vSampling.vBackendSampling = true; })
              .mSetSampling()
              .mSetEnv("MGS_ARG_BACKEND_SAMPLING"));
  vAddOpt(common_arg({"--pooling"}, "{none,mean,cls,last,rank}",
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
              .mSetExamples({cMgsExampleEmbedding, cMgsExampleRetrieval, cMgsExampleServer, cMgsExampleDebug})
              .mSetEnv("MGS_ARG_POOLING"));
  vAddOpt(common_arg({"--attention"}, "{causal,non-causal}",
                     "attention type for embeddings, use model default if unspecified",
                     [](common_params &pParams, const std::string &pValue) {
                       /**/ if (pValue == "causal") {
                         pParams.vAttentionType = MGS_ATTENTION_TYPE_CAUSAL;
                       } else if (pValue == "non-causal") {
                         pParams.vAttentionType = MGS_ATTENTION_TYPE_NON_CAUSAL;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
              .mSetExamples({cMgsExampleEmbedding}));
  vAddOpt(common_arg({"--rope-scaling"}, "{none,linear,yarn}",
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
              .mSetEnv("MGS_ARG_ROPE_SCALING_TYPE"));
  vAddOpt(common_arg({"--rope-scale"}, "N", "RoPE context scaling factor, expands context by a factor of N",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vRopeFreqScale = 1.0f / std::stof(pValue);
                     })
              .mSetEnv("MGS_ARG_ROPE_SCALE"));
  vAddOpt(common_arg(
              {"--rope-freq-base"}, "N", "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
              [](common_params &pParams, const std::string &pValue) { pParams.vRopeFreqBase = std::stof(pValue); })
              .mSetEnv("MGS_ARG_ROPE_FREQ_BASE"));
  vAddOpt(
      common_arg({"--rope-freq-scale"}, "N", "RoPE frequency scaling factor, expands context by a factor of 1/N",
                 [](common_params &pParams, const std::string &pValue) { pParams.vRopeFreqScale = std::stof(pValue); })
          .mSetEnv("MGS_ARG_ROPE_FREQ_SCALE"));
  vAddOpt(common_arg({"--yarn-orig-ctx"}, "N",
                     fStringFormat("YaRN: original context size of model (default: %d = model training context size)",
                                   pParams.vYarnOrigCtx),
                     [](common_params &pParams, int pValue) { pParams.vYarnOrigCtx = pValue; })
              .mSetEnv("MGS_ARG_YARN_ORIG_CTX"));
  vAddOpt(
      common_arg({"--yarn-ext-factor"}, "N",
                 fStringFormat("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)",
                               (double)pParams.vYarnExtFactor),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnExtFactor = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_EXT_FACTOR"));
  vAddOpt(
      common_arg(
          {"--yarn-attn-factor"}, "N",
          fStringFormat("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)pParams.vYarnAttnFactor),
          [](common_params &pParams, const std::string &pValue) { pParams.vYarnAttnFactor = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_ATTN_FACTOR"));
  vAddOpt(
      common_arg({"--yarn-beta-slow"}, "N",
                 fStringFormat("YaRN: high correction dim or alpha (default: %.2f)", (double)pParams.vYarnBetaSlow),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnBetaSlow = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_BETA_SLOW"));
  vAddOpt(
      common_arg({"--yarn-beta-fast"}, "N",
                 fStringFormat("YaRN: low correction dim or beta (default: %.2f)", (double)pParams.vYarnBetaFast),
                 [](common_params &pParams, const std::string &pValue) { pParams.vYarnBetaFast = std::stof(pValue); })
          .mSetEnv("MGS_ARG_YARN_BETA_FAST"));
  vAddOpt(common_arg({"-gan", "--grp-attn-n"}, "N",
                     fStringFormat("group-attention factor (default: %d)", pParams.vGrpAttnN),
                     [](common_params &pParams, int pValue) { pParams.vGrpAttnN = pValue; })
              .mSetEnv("MGS_ARG_GRP_ATTN_N")
              .mSetExamples({cMgsExampleCompletion, cMgsExamplePasskey}));
  vAddOpt(common_arg({"-gaw", "--grp-attn-w"}, "N",
                     fStringFormat("group-attention width (default: %d)", pParams.vGrpAttnW),
                     [](common_params &pParams, int pValue) { pParams.vGrpAttnW = pValue; })
              .mSetEnv("MGS_ARG_GRP_ATTN_W")
              .mSetExamples({cMgsExampleCompletion}));
  vAddOpt(common_arg({"-kvo", "--kv-offload"}, {"-nkvo", "--no-kv-offload"},
                     fStringFormat("whether to enable KV cache offloading (default: %s)",
                                   pParams.vNoKvOffload ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoKvOffload = !pValue; })
              .mSetEnv("MGS_ARG_KV_OFFLOAD"));
  vAddOpt(common_arg({"--repack"}, {"-nr", "--no-repack"},
                     fStringFormat("whether to enable weight repacking (default: %s)",
                                   pParams.vNoExtraBufts ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoExtraBufts = !pValue; })
              .mSetEnv("MGS_ARG_REPACK"));
  vAddOpt(common_arg({"--no-host"}, "bypass host buffer allowing extra buffers to be used", [](common_params &pParams) {
            pParams.vNoHost = true;
          }).mSetEnv("MGS_ARG_NO_HOST"));
  vAddOpt(common_arg({"-ctk", "--cache-type-k"}, "TYPE",
                     fStringFormat("KV cache data type for K\n"
                                   "allowed values: %s\n"
                                   "(default: %s)",
                                   fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vCacheTypeK)),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vCacheTypeK = fKvCacheTypeFromStr(pValue);
                     })
              .mSetEnv("MGS_ARG_CACHE_TYPE_K"));
  vAddOpt(common_arg({"-ctv", "--cache-type-v"}, "TYPE",
                     fStringFormat("KV cache data type for V\n"
                                   "allowed values: %s\n"
                                   "(default: %s)",
                                   fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vCacheTypeV)),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vCacheTypeV = fKvCacheTypeFromStr(pValue);
                     })
              .mSetEnv("MGS_ARG_CACHE_TYPE_V"));
  vAddOpt(common_arg({"--hellaswag"}, "compute HellaSwag score over random tasks from datafile supplied with -f",
                     [](common_params &pParams) { pParams.vHellaswag = true; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--hellaswag-tasks"}, "N",
                     fStringFormat("number of tasks to use when computing the HellaSwag score (default: %zu)",
                                   pParams.vHellaswagTasks),
                     [](common_params &pParams, int pValue) { pParams.vHellaswagTasks = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--winogrande"}, "compute Winogrande score over random tasks from datafile supplied with -f",
                     [](common_params &pParams) { pParams.vWinogrande = true; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--winogrande-tasks"}, "N",
                     fStringFormat("number of tasks to use when computing the Winogrande score (default: %zu)",
                                   pParams.vWinograndeTasks),
                     [](common_params &pParams, int pValue) { pParams.vWinograndeTasks = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--multiple-choice"},
                     "compute multiple choice score over random tasks from datafile supplied with -f",
                     [](common_params &pParams) { pParams.vMultipleChoice = true; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--multiple-choice-tasks"}, "N",
                     fStringFormat("number of tasks to use when computing the multiple choice score (default: %zu)",
                                   pParams.vMultipleChoiceTasks),
                     [](common_params &pParams, int pValue) { pParams.vMultipleChoiceTasks = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--kl-divergence"}, "computes KL-divergence to logits provided via --save-all-logits",
                     [](common_params &pParams) { pParams.vKlDivergence = true; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--save-all-logits"}, "FNAME", "set logits file",
                     [](common_params &pParams, const std::string &pValue) { pParams.logits_file = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--ppl-stride"}, "N",
                     fStringFormat("stride for perplexity calculation (default: %d)", pParams.vPplStride),
                     [](common_params &pParams, int pValue) { pParams.vPplStride = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  vAddOpt(common_arg({"--ppl-output-type"}, "<0|1>",
                     fStringFormat("output type for perplexity calculation (default: %d)", pParams.vPplOutputType),
                     [](common_params &pParams, int pValue) { pParams.vPplOutputType = pValue; })
              .mSetExamples({cMgsExamplePerplexity}));
  if (pEx == cMgsExampleServer) {
    // this is to make sure this option appears in the server-specific section of the help message
    vAddOpt(common_arg({"-np", "--parallel"}, "N",
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
                .mSetEnv("MGS_ARG_N_PARALLEL")
                .mSetExamples({cMgsExampleServer}));
  } else {
    vAddOpt(common_arg({"-np", "--parallel"}, "N",
                       fStringFormat("number of parallel sequences to decode (default: %d)", pParams.vNParallel),
                       [](common_params &pParams, int pValue) { pParams.vNParallel = pValue; })
                .mSetEnv("MGS_ARG_N_PARALLEL"));
  }
  vAddOpt(common_arg({"-ns", "--sequences"}, "N",
                     fStringFormat("number of sequences to decode (default: %d)", pParams.vNSequences),
                     [](common_params &pParams, int pValue) { pParams.vNSequences = pValue; })
              .mSetExamples({cMgsExampleParallel}));
  vAddOpt(common_arg({"-cb", "--cont-batching"}, {"-nocb", "--no-cont-batching"},
                     fStringFormat("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)",
                                   pParams.vContBatching ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vContBatching = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CONT_BATCHING"));
  vAddOpt(common_arg({"-mm", "--mmproj"}, "FILE",
                     "path to a multimodal projector file. see tools/mtmd/README.md\n"
                     "note: if -hf is used, this argument can be omitted",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMmproj.vPath = pValue; })
              .mSetExamples(lMmprojExamples)
              .mSetEnv("MGS_ARG_MMPROJ"));
  vAddOpt(common_arg({"-mmu", "--mmproj-url"}, "URL", "URL to a multimodal projector file. see tools/mtmd/README.md",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMmproj.vUrl = pValue; })
              .mSetExamples(lMmprojExamples)
              .mSetEnv("MGS_ARG_MMPROJ_URL"));
  vAddOpt(common_arg({"--mmproj-auto"}, {"--no-mmproj"},
                     fStringFormat(
                         "whether to use multimodal projector file (if available), useful when using -hf (default: %s)",
                         pParams.vNoMmproj ? "disabled" : "enabled"),
                     [](common_params &pParams, bool pValue) { pParams.vNoMmproj = !pValue; })
              .mSetExamples({cMgsExampleMtmd, cMgsExampleServer, cMgsExampleCli, cMgsExampleDownload})
              .mSetEnv("MGS_ARG_MMPROJ_AUTO"));
  vAddOpt(common_arg({"--mmproj-offload"}, {"--no-mmproj-offload"},
                     fStringFormat("whether to enable GPU offloading for multimodal projector (default: %s)",
                                   pParams.vMmprojUseGpu ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vMmprojUseGpu = pValue; })
              .mSetExamples(lMmprojExamples)
              .mSetEnv("MGS_ARG_MMPROJ_OFFLOAD"));
  vAddOpt(common_arg({"--image", "--audio", "--video"}, "FILE",
                     "path to an image, audio, or video file. use with multimodal models, use comma-separated values "
                     "for multiple files\n",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         pParams.lImage.emplace_back(vItem);
                       }
                     })
              .mSetExamples({cMgsExampleMtmd, cMgsExampleCli}));
  vAddOpt(common_arg({"--image-min-tokens"}, "N",
                     "minimum number of tokens each image can take, only used by vision models with dynamic resolution "
                     "(default: read from model)",
                     [](common_params &pParams, int pValue) { pParams.vImageMinTokens = pValue; })
              .mSetExamples(lMmprojExamples)
              .mSetEnv("MGS_ARG_IMAGE_MIN_TOKENS"));
  vAddOpt(common_arg({"--image-max-tokens"}, "N",
                     "maximum number of tokens each image can take, only used by vision models with dynamic resolution "
                     "(default: read from model)",
                     [](common_params &pParams, int pValue) { pParams.vImageMaxTokens = pValue; })
              .mSetExamples(lMmprojExamples)
              .mSetEnv("MGS_ARG_IMAGE_MAX_TOKENS"));
  vAddOpt(common_arg({"--mtmd-batch-max-tokens"}, "N",
                     fStringFormat("maximum number of image tokens per batch when encoding images (default: %d)",
                                   pParams.vMtmdBatchMaxTokens),
                     [](common_params &pParams, int pValue) { pParams.vMtmdBatchMaxTokens = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MTMD_BATCH_MAX_TOKENS"));
  if (pParams.vIsGenDocs || mgs_supports_rpc()) {
    vAddOpt(common_arg({"--rpc"}, "SERVERS", "comma-separated list of RPC servers (host:port)",
                       [](common_params &pParams, const std::string &pValue) {
                         fAddRpcDevices(pValue);
                         GGML_UNUSED(pParams);
                       })
                .mSetEnv("MGS_ARG_RPC"));
  }
  vAddOpt(
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
  vAddOpt(
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
  vAddOpt(
      common_arg({"-dev", "--device"}, "<dev1,dev2,..>",
                 "comma-separated list of devices to use for offloading (none = don't offload)\n"
                 "use --list-devices to see a list of available devices",
                 [](common_params &pParams, const std::string &pValue) { pParams.lDevices = fParseDeviceList(pValue); })
          .mSetEnv("MGS_ARG_DEVICE"));
  vAddOpt(common_arg({"--list-devices"}, "print list of available devices and exit", [](common_params &) {
    fCommonPrintAvailableDevices();
    exit(0);
  }));
  vAddOpt(common_arg({"-ot", "--override-tensor"}, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type",
                     [](common_params &pParams, const std::string &pValue) {
                       fParseTensorBufferOverrides(pValue, pParams.lTensorBuftOverrides);
                     })
              .mSetEnv("MGS_ARG_OVERRIDE_TENSOR"));
  vAddOpt(common_arg({"-cmoe", "--cpu-moe"}, "keep all Mixture of Experts (MoE) weights in the CPU",
                     [](common_params &pParams) { pParams.lTensorBuftOverrides.push_back(fLlmFfnExpsCpuOverride()); })
              .mSetEnv("MGS_ARG_CPU_MOE"));
  // The engine reads these settings from the environment, which keeps the public API unchanged.
  vAddOpt(common_arg({"--moe-cache"}, {"--no-moe-cache"},
                     "keep the most recently used MoE experts stored in system memory in a device cache, and split "
                     "every decoding step between that cache and the CPU (default: enabled)",
                     [](common_params &pParams, bool pValue) {
                       (void)pParams;
                       fCommonSetEnv("MGS_MOE_CACHE", pValue ? "1" : "0");
                     })
              .mSetEnv("MGS_ARG_MOE_CACHE"));
  vAddOpt(common_arg({"--moe-cache-size"}, "MiB",
                     "device memory for the MoE expert cache in MiB; with several GPUs the total is split among them "
                     "by the experts each one caches (default: -1, the memory left free on every GPU after the "
                     "model, the context and --moe-cache-reserve)",
                     [](common_params &pParams, int pValue) {
                       (void)pParams;
                       fCommonSetEnv("MGS_MOE_CACHE_MIB", pValue < 0 ? "" : std::to_string(pValue));
                     })
              .mSetEnv("MGS_ARG_MOE_CACHE_SIZE"));
  vAddOpt(common_arg({"--moe-cache-reserve"}, "MiB",
                     "device memory left free on every GPU besides the MoE expert cache and the expert prefetch "
                     "slots, in MiB (default: 1024)",
                     [](common_params &pParams, int pValue) {
                       (void)pParams;
                       if (pValue < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       fCommonSetEnv("MGS_MOE_CACHE_RESERVE_MIB", std::to_string(pValue));
                     })
              .mSetEnv("MGS_ARG_MOE_CACHE_RESERVE"));
  vAddOpt(common_arg({"-ncmoe", "--n-cpu-moe"}, "N",
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
  vAddOpt(
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
          .mSetEnv("MGS_ARG_N_GPU_LAYERS"));
  vAddOpt(common_arg({"-sm", "--split-mode"}, "{none,layer,row,tensor}",
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
  vAddOpt(common_arg({"-ts", "--tensor-split"}, "N0,N1,N2,...",
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
  vAddOpt(common_arg({"-mg", "--main-gpu"}, "INDEX",
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
  vAddOpt(common_arg(
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
  vAddOpt(common_arg({"-fitp", "--fit-print"}, "[on|off]",
                     fStringFormat("print the estimated required memory ('on' or 'off', default: '%s')",
                                   pParams.vFitParamsPrint ? "on" : "off"),
                     [](common_params &pParams, const std::string &pValue) {
                       if (fIsTruthy(pValue)) {
                         pParams.vFitParamsPrint = true;
                       } else if (fIsFalsey(pValue)) {
                         pParams.vFitParamsPrint = false;
                       } else {
                         throw std::runtime_error(
                             fStringFormat("error: unknown value for --fit-print: '%s'\n", pValue.c_str()));
                       }
                     })
              .mSetExamples({cMgsExampleFitParams})
              .mSetEnv("MGS_ARG_FIT_PRINT"));
  vAddOpt(common_arg({"-fitt", "--fit-target"}, "MiB0,MiB1,MiB2,...",
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
  vAddOpt(common_arg({"-fitc", "--fit-ctx"}, "N",
                     fStringFormat("minimum ctx size that can be set by --fit option, default: %" PRIu32,
                                   pParams.vFitParamsMinCtx),
                     [](common_params &pParams, int pValue) { pParams.vFitParamsMinCtx = pValue; })
              .mSetEnv("MGS_ARG_FIT_CTX"));
  vAddOpt(common_arg({"--check-tensors"},
                     fStringFormat("check model tensor data for invalid values (default: %s)",
                                   pParams.vCheckTensors ? "true" : "false"),
                     [](common_params &pParams) { pParams.vCheckTensors = true; }));
  vAddOpt(common_arg({"--override-kv"}, "KEY=TYPE:VALUE,...",
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
  vAddOpt(common_arg({"--op-offload"}, {"--no-op-offload"},
                     fStringFormat("whether to offload host tensor operations to device (default: %s)",
                                   pParams.vNoOpOffload ? "false" : "true"),
                     [](common_params &pParams, bool pValue) { pParams.vNoOpOffload = !pValue; }));
  vAddOpt(common_arg({"--lora"}, "FNAME", "path to LoRA adapter (use comma-separated values to load multiple adapters)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         pParams.lLoraAdapters.push_back({vItem, 1.0, "", "", nullptr});
                       }
                     }
                     // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora,
                     // it will be categorized as "example-specific" arg
                     )
              .mSetExamples({cMgsExampleCommon, cMgsExampleExportLora}));
  vAddOpt(common_arg({"--lora-scaled"}, "FNAME:SCALE,...",
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
                     )
              .mSetExamples({cMgsExampleCommon, cMgsExampleExportLora}));
  vAddOpt(common_arg({"--control-vector"}, "FNAME",
                     "add a control vector\nnote: use comma-separated values to add multiple control vectors",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         pParams.lControlVectors.push_back({
                             1.0f,
                             vItem,
                         });
                       }
                     }));
  vAddOpt(common_arg({"--control-vector-scaled"}, "FNAME:SCALE,...",
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
  vAddOpt(common_arg({"--control-vector-layer-range"}, "START", "END",
                     "layer range to apply the control vector(s) to, start and end inclusive",
                     [](common_params &pParams, const std::string &pStart, const std::string &pEnd) {
                       pParams.vControlVectorLayerStart = std::stoi(pStart);
                       pParams.vControlVectorLayerEnd = std::stoi(pEnd);
                     }));
  vAddOpt(common_arg({"-a", "--alias"}, "STRING", "set model name aliases, comma-separated (to be used by API)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (auto &vAlias : fStringSplit<std::string>(pValue, ',')) {
                         vAlias = fStringStrip(vAlias);
                         if (!vAlias.empty()) {
                           pParams.model_alias.insert(vAlias);
                         }
                       }
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_ALIAS"));
  vAddOpt(common_arg({"--tags"}, "STRING", "set model tags, comma-separated (informational, not used for routing)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (auto &vTag : fStringSplit<std::string>(pValue, ',')) {
                         vTag = fStringStrip(vTag);
                         if (!vTag.empty()) {
                           pParams.sModelTags.insert(vTag);
                         }
                       }
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_TAGS"));
  vAddOpt(common_arg({"-m", "--model"}, "FNAME",
                     pEx == cMgsExampleExportLora ? "model path from which to load base model" : "model path to load",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vPath = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleExportLora, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_MODEL"));
  vAddOpt(common_arg({"-mu", "--model-url"}, "MODEL_URL", "model download url (default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vUrl = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_MODEL_URL"));
  vAddOpt(common_arg(
              {"-dr", "--docker-repo"}, "[<repo>/]<model>[:quant]",
              "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
              "example: gemma3\n"
              "(default: unused)",
              [](common_params &pParams, const std::string &pValue) { pParams.vModel.vDockerRepo = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_DOCKER_REPO"));
  vAddOpt(common_arg({"-hf", "--hf-repo"}, "<user>/<model>[:quant]",
                     "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls "
                     "back to the first file in the repo if Q4_K_M doesn't exist.\n"
                     "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
                     "example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n"
                     "(default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vHfRepo = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_HF_REPO"));
  vAddOpt(common_arg({"-hff", "--hf-file"}, "FILE",
                     "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModel.vHfFile = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_HF_FILE"));
  vAddOpt(
      common_arg({"-hfv", "--hf-repo-v"}, "<user>/<model>[:quant]",
                 "Hugging Face model repository for the vocoder model (default: unused)",
                 [](common_params &pParams, const std::string &pValue) { pParams.vVocoder.vModel.vHfRepo = pValue; })
          .mSetEnv("MGS_ARG_HF_REPO_V"));
  vAddOpt(
      common_arg({"-hffv", "--hf-file-v"}, "FILE", "Hugging Face model file for the vocoder model (default: unused)",
                 [](common_params &pParams, const std::string &pValue) { pParams.vVocoder.vModel.vHfFile = pValue; })
          .mSetEnv("MGS_ARG_HF_FILE_V"));
  vAddOpt(common_arg({"-hft", "--hf-token"}, "TOKEN",
                     "Hugging Face access token (default: value from HF_TOKEN environment variable)",
                     [](common_params &pParams, const std::string &pValue) { pParams.hf_token = pValue; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("HF_TOKEN"));
  vAddOpt(
      common_arg({"--mtp"}, "also download the multi-token prediction (MTP) head, if available (default: unused)",
                 [](common_params &pParams) { pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeDraftMtp); })
          .mSetExamples({cMgsExampleDownload}));
  vAddOpt(common_arg(
              {"--dflash"}, "also download the DFlash sidecar, if available (default: unused)",
              [](common_params &pParams) { pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeDraftDflash); })
              .mSetExamples({cMgsExampleDownload}));
  vAddOpt(common_arg(
              {"--eagle3"}, "also download the Eagle3 sidecar, if available (default: unused)",
              [](common_params &pParams) { pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeDraftEagle3); })
              .mSetExamples({cMgsExampleDownload}));
  vAddOpt(common_arg({"--context-file"}, "FNAME",
                     "file to load context from (use comma-separated values to specify multiple files)",
                     [](common_params &pParams, const std::string &pValue) {
                       for (const auto &vItem : fParseCsvRow(pValue)) {
                         std::ifstream vFile(vItem, std::ios::binary);
                         if (!vFile) {
                           throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", vItem.c_str()));
                         }
                         pParams.lContextFiles.push_back(vItem);
                       }
                     })
              .mSetExamples({cMgsExampleRetrieval}));
  vAddOpt(common_arg({"--chunk-size"}, "N",
                     fStringFormat("minimum length of embedded text chunks (default: %d)", pParams.vChunkSize),
                     [](common_params &pParams, int pValue) { pParams.vChunkSize = pValue; })
              .mSetExamples({cMgsExampleRetrieval}));
  vAddOpt(common_arg({"--chunk-separator"}, "STRING",
                     fStringFormat("separator between chunks (default: '%s')", pParams.vChunkSeparator.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vChunkSeparator = pValue; })
              .mSetExamples({cMgsExampleRetrieval}));
  vAddOpt(common_arg({"--junk"}, "N",
                     fStringFormat("number of times to repeat the junk text (default: %d)", pParams.vNJunk),
                     [](common_params &pParams, int pValue) { pParams.vNJunk = pValue; })
              .mSetExamples({cMgsExamplePasskey, cMgsExampleParallel}));
  vAddOpt(common_arg({"--pos"}, "N",
                     fStringFormat("position of the passkey in the junk text (default: %d)", pParams.vIPos),
                     [](common_params &pParams, int pValue) { pParams.vIPos = pValue; })
              .mSetExamples({cMgsExamplePasskey}));
  vAddOpt(common_arg({"-o", "--output"}, "FNAME",
                     fStringFormat("output file (default: '%s')", pParams.vOutFile.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vOutFile = pValue; })
              .mSetExamples({cMgsExampleImatrix, cMgsExampleCvectorGenerator, cMgsExampleExportLora, cMgsExampleTts,
                             cMgsExampleFinetune, cMgsExampleResults, cMgsExampleExportGraphOps, cMgsExampleCli}));
  vAddOpt(common_arg({"-ofreq", "--output-frequency"}, "N",
                     fStringFormat("output the imatrix every N iterations (default: %d)", pParams.vNOutFreq),
                     [](common_params &pParams, int pValue) { pParams.vNOutFreq = pValue; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"--save-frequency"}, "N",
                     fStringFormat("save an imatrix copy every N iterations (default: %d)", pParams.vNSaveFreq),
                     [](common_params &pParams, int pValue) { pParams.vNSaveFreq = pValue; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"--process-output"},
                     fStringFormat("collect data for the output tensor (default: %s)",
                                   pParams.vProcessOutput ? "true" : "false"),
                     [](common_params &pParams) { pParams.vProcessOutput = true; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(
      common_arg({"--ppl"}, {"--no-ppl"},
                 fStringFormat("whether to compute perplexity (default: %s)", pParams.vComputePpl ? "true" : "false"),
                 [](common_params &pParams, bool pValue) { pParams.vComputePpl = pValue; })
          .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"--chunk"}, "N",
                     fStringFormat("start processing the input from chunk N (default: %d)", pParams.vIChunk),
                     [](common_params &pParams, int pValue) { pParams.vIChunk = pValue; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"--show-statistics"},
                     fStringFormat("show imatrix statistics and then exit (default: %s)",
                                   pParams.vShowStatistics ? "true" : "false"),
                     [](common_params &pParams) { pParams.vShowStatistics = true; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(common_arg({"--parse-special"},
                     fStringFormat("parse special tokens (chat, tool, etc) (default: %s)",
                                   pParams.vParseSpecial ? "true" : "false"),
                     [](common_params &pParams) { pParams.vParseSpecial = true; })
              .mSetExamples({cMgsExampleImatrix}));
  vAddOpt(
      common_arg({"--ids"},
                 fStringFormat("only print the token IDs, in a Python-parseable list form like [1, 2, 3] (default: %s)",
                               pParams.vTokenizeIds ? "true" : "false"),
                 [](common_params &pParams) { pParams.vTokenizeIds = true; })
          .mSetExamples({cMgsExampleTokenize}));
  vAddOpt(common_arg({"--stdin"},
                     fStringFormat(
                         "read the prompt from stdin (takes precedence over -f/--file and -p/--prompt) (default: %s)",
                         pParams.vTokenizeStdin ? "true" : "false"),
                     [](common_params &pParams) { pParams.vTokenizeStdin = true; })
              .mSetExamples({cMgsExampleTokenize}));
  vAddOpt(common_arg(
              {"--no-bos"},
              fStringFormat("do not add a BOS token to the prompt, even if the model normally uses one (default: %s)",
                            pParams.vTokenizeNoBos ? "true" : "false"),
              [](common_params &pParams) { pParams.vTokenizeNoBos = true; })
              .mSetExamples({cMgsExampleTokenize}));
  vAddOpt(common_arg({"--no-parse-special"},
                     fStringFormat("do not parse special tokens (chat, tool, etc) (default: %s)",
                                   !pParams.vParseSpecial ? "true" : "false"),
                     [](common_params &pParams) { pParams.vParseSpecial = false; })
              .mSetExamples({cMgsExampleTokenize}));
  vAddOpt(common_arg({"--show-count"},
                     fStringFormat("print the total number of tokens (default: %s)",
                                   pParams.vTokenizeShowCount ? "true" : "false"),
                     [](common_params &pParams) { pParams.vTokenizeShowCount = true; })
              .mSetExamples({cMgsExampleTokenize}));
  vAddOpt(common_arg({"-pps"},
                     fStringFormat("is the prompt shared across parallel sequences (default: %s)",
                                   pParams.vIsPpShared ? "true" : "false"),
                     [](common_params &pParams) { pParams.vIsPpShared = true; })
              .mSetExamples({cMgsExampleBench, cMgsExampleParallel}));
  vAddOpt(common_arg({"-tgs"},
                     fStringFormat("is the text generation separated across the different sequences (default: %s)",
                                   pParams.vIsTgSeparate ? "true" : "false"),
                     [](common_params &pParams) { pParams.vIsTgSeparate = true; })
              .mSetExamples({cMgsExampleBench, cMgsExampleParallel}));
  vAddOpt(common_arg({"-npp"}, "n0,n1,...", "number of prompt tokens",
                     [](common_params &pParams, const std::string &pValue) {
                       auto vP = fStringSplit<int>(pValue, ',');
                       pParams.lNPp.insert(pParams.lNPp.end(), vP.begin(), vP.end());
                     })
              .mSetExamples({cMgsExampleBench}));
  vAddOpt(common_arg({"-ntg"}, "n0,n1,...", "number of text generation tokens",
                     [](common_params &pParams, const std::string &pValue) {
                       auto vP = fStringSplit<int>(pValue, ',');
                       pParams.lNTg.insert(pParams.lNTg.end(), vP.begin(), vP.end());
                     })
              .mSetExamples({cMgsExampleBench}));
  vAddOpt(common_arg({"-npl"}, "n0,n1,...", "number of parallel prompts",
                     [](common_params &pParams, const std::string &pValue) {
                       auto vP = fStringSplit<int>(pValue, ',');
                       pParams.lNPl.insert(pParams.lNPl.end(), vP.begin(), vP.end());
                     })
              .mSetExamples({cMgsExampleBench}));
  vAddOpt(common_arg({"--embd-normalize"}, "N",
                     fStringFormat("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, "
                                   "1=taxicab, 2=euclidean, >2=p-norm)",
                                   pParams.vEmbdNormalize),
                     [](common_params &pParams, int pValue) { pParams.vEmbdNormalize = pValue; })
              .mSetExamples({cMgsExampleEmbedding, cMgsExampleServer, cMgsExampleDebug}));
  vAddOpt(common_arg({"--embd-output-format"}, "FORMAT",
                     "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + "
                     "cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vEmbdOut = pValue; })
              .mSetExamples({cMgsExampleEmbedding}));
  vAddOpt(common_arg({"--embd-separator"}, "STRING", "separator of embeddings (default \\n) for example \"<#sep#>\"",
                     [](common_params &pParams, const std::string &pValue) { pParams.vEmbdSep = pValue; })
              .mSetExamples({cMgsExampleEmbedding}));
  vAddOpt(common_arg({"--cls-separator"}, "STRING",
                     "separator of classification sequences (default \\t) for example \"<#seq#>\"",
                     [](common_params &pParams, const std::string &pValue) { pParams.vClsSep = pValue; })
              .mSetExamples({cMgsExampleEmbedding}));
  vAddOpt(common_arg({"--host"}, "HOST",
                     fStringFormat(
                         "ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: %s)",
                         pParams.vHostname.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vHostname = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_HOST"));
  vAddOpt(common_arg({"--port"}, "PORT",
                     "fixed listening port; 0 selects an OS-assigned port (default: first free from 11443 for HTTPS, "
                     "11080 for --http)",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 0 || pValue > 65535) {
                         throw std::invalid_argument("--port must be between 0 and 65535");
                       }
                       pParams.vPort = pValue;
                       pParams.vPortExplicit = true;
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_PORT"));
  vAddOpt(common_arg({"--https"}, {"--http"},
                     "serve HTTPS with HTTP redirection (default: enabled); --http explicitly serves plain HTTP",
                     [](common_params &pParams, bool pValue) { pParams.vUseHttps = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_HTTPS"));
  vAddOpt(common_arg({"--http-port"}, "PORT", "first HTTP redirect port to try (default: 11080)",
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 65535) {
                         throw std::invalid_argument("--http-port must be between 1 and 65535");
                       }
                       pParams.vHttpPort = pValue;
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_HTTP_PORT"));
  vAddOpt(common_arg({"--reuse-port"},
                     fStringFormat("allow multiple sockets to bind to the same port (default: %s)",
                                   pParams.vReusePort ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vReusePort = true; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_REUSE_PORT"));
  vAddOpt(common_arg({"--path"}, "PATH",
                     fStringFormat("path to serve static files from (default: %s)", pParams.public_path.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.public_path = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_STATIC_PATH"));
  vAddOpt(
      common_arg({"--cors-origins"}, "ORIGINS",
                 fStringFormat("comma-separated list of allowed origins for CORS (default: %s)\n"
                               "if set to special value 'localhost', reflect the Origin header only if it is localhost",
                               pParams.vCorsOrigins.c_str()),
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vCorsOrigins = pValue;
                   pParams.vCorsOriginsExplicit = true;
                 })
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_ARG_CORS_ORIGINS"));
  vAddOpt(common_arg({"--cors-methods"}, "METHODS",
                     fStringFormat("comma-separated list of allowed methods for CORS (default: %s)",
                                   pParams.vCorsMethods.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCorsMethods = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CORS_METHODS"));
  vAddOpt(common_arg({"--cors-headers"}, "HEADERS",
                     fStringFormat("comma-separated list of allowed headers for CORS (default: %s)",
                                   pParams.vCorsHeaders.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCorsHeaders = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CORS_HEADERS"));
  vAddOpt(common_arg({"--cors-credentials"}, {"--no-cors-credentials"},
                     fStringFormat("whether to allow credentials for CORS (default: %s)\n"
                                   "note: if this is enabled and --cors-origins is set to * (default), the Origin "
                                   "header will be echoed back, and credentials will always be allowed",
                                   pParams.vCorsCredentials ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCorsCredentials = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CORS_CREDENTIALS"));
  vAddOpt(common_arg({"--api-prefix"}, "PREFIX",
                     fStringFormat("prefix path the server serves from, without the trailing slash (default: %s)",
                                   pParams.api_prefix.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.api_prefix = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_API_PREFIX"));
  vAddOpt(common_arg({"--ui-config"}, "JSON", "JSON that provides default UI settings (overrides UI defaults)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vUiConfigJson = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_UI_CONFIG"));
  vAddOpt(
      common_arg({"--ui-config-file"}, "PATH", "JSON file that provides default UI settings (overrides UI defaults)",
                 [](common_params &pParams, const std::string &pValue) { pParams.vUiConfigJson = fReadFile(pValue); })
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_ARG_UI_CONFIG_FILE"));
  vAddOpt(common_arg({"--ui-mcp-proxy"}, {"--no-ui-mcp-proxy"},
                     "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments "
                     "(default: disabled)",
                     [](common_params &pParams, bool pValue) { pParams.vUiMcpProxy = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_UI_MCP_PROXY"));
  vAddOpt(
      common_arg({"--tools"}, "TOOL1,TOOL2,...",
                 "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted "
                 "environments (default: no tools)\n"
                 "specify \"all\" to enable all tools\n"
                 "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, "
                 "edit_file, get_datetime, get_info\n"
                 "note: for security reasons, this will limit --cors-origins to localhost by default",
                 [](common_params &pParams, const std::string &pValue) { pParams.lServerTools = fParseCsvRow(pValue); })
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_ARG_TOOLS"));
  vAddOpt(common_arg({"--mcp-servers-config"}, "PATH",
                     "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not "
                     "enable in untrusted environments (default: none)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMcpServersConfig = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MCP_SERVERS_CONFIG"));
  vAddOpt(common_arg({"--mcp-servers-json"}, "JSON",
                     "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable "
                     "in untrusted environments (default: none)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params &pParams, const std::string &pValue) { pParams.vMcpServersJson = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MCP_SERVERS_JSON"));
  vAddOpt(common_arg({"--local-shell-mcp-server"},
                     "experimental: serve the local shell MCP server at <api prefix>/mcp/shell, which runs commands in "
                     "the shells installed on this machine (bash.run, zsh.run, sh.run, psh.run; only the shells found "
                     "are exposed); the Web UI adds it and enables it by default - do not enable in untrusted "
                     "environments (default: disabled)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params &pParams) { pParams.vShellMcpServer = true; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_LOCAL_SHELL_MCP_SERVER"));
  vAddOpt(common_arg({"-ag", "--agent"}, {"-no-ag", "--no-agent"},
                     "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments "
                     "(default: disabled)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params &pParams, bool pValue) {
                       if (pValue) {
                         pParams.lServerTools = {"all"};
                         pParams.vUiMcpProxy = true;
                       } else {
                         pParams.lServerTools.clear();
                         pParams.vUiMcpProxy = false;
                       }
                       // note: do not modify cors_origins here, as the options are not evaluated in order (user may
                       // explicitly set --cors-origins before --agent)
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_AGENT"));
  vAddOpt(common_arg({"--ui"}, {"--no-ui"},
                     fStringFormat("whether to enable the Web UI (default: %s)", pParams.vUi ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vUi = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_UI"));
  vAddOpt(common_arg(
              {"--embedding"},
              fStringFormat(
                  "restrict to only support embedding use case; use only with dedicated embedding models (default: %s)",
                  pParams.vEmbedding ? "enabled" : "disabled"),
              [](common_params &pParams) { pParams.vEmbedding = true; })
              .mSetExamples({cMgsExampleServer, cMgsExampleDebug})
              .mSetEnv("MGS_ARG_EMBEDDINGS"));
  vAddOpt(common_arg({"--rerank"},
                     fStringFormat("enable reranking endpoint on server (default: %s)", "disabled"),
                     [](common_params &pParams) {
                       pParams.vEmbedding = true;
                       pParams.vPoolingType = MGS_POOLING_TYPE_RANK;
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_RERANKING"));
  vAddOpt(
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
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_API_KEY"));
  vAddOpt(common_arg({"--api-key-file"}, "FNAME",
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
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_API_KEY_FILE"));
  vAddOpt(common_arg({"--ssl-key-file"}, "FNAME",
                     "PEM private key, used with --ssl-cert-file (default: the self-signed key created once in "
                     "~/.mgs/certificates/privkey.pem)",
                     [](common_params &pParams, const std::string &pValue) { pParams.ssl_file_key = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_SSL_KEY_FILE"));
  vAddOpt(common_arg({"--ssl-cert-file"}, "FNAME",
                     "PEM certificate, used with --ssl-key-file (default: self-signed, created once in "
                     "~/.mgs/certificates/fullchain.pem and reused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.ssl_file_cert = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_SSL_CERT_FILE"));
  vAddOpt(common_arg({"--chat-template-kwargs"}, "STRING",
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
              .mSetExamples({cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_CHAT_TEMPLATE_KWARGS"));
  vAddOpt(common_arg({"-to", "--timeout"}, "N",
                     fStringFormat("server write timeout and router-to-model proxy timeout in seconds; reading a "
                                   "request waits at most min(N, 30) seconds per read (default: %d)",
                                   pParams.vTimeoutRead),
                     [](common_params &pParams, int pValue) {
                       pParams.vTimeoutRead = pValue;
                       pParams.vTimeoutWrite = pValue;
                     })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_TIMEOUT"));
  vAddOpt(common_arg({"--sse-ping-interval"}, "N",
                     fStringFormat("server SSE ping interval in seconds (-1 = disabled, default: %d)",
                                   pParams.vSsePingInterval),
                     [](common_params &pParams, int pValue) { pParams.vSsePingInterval = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_SSE_PING_INTERVAL"));
  vAddOpt(
      common_arg({"--threads-http"}, "N",
                 fStringFormat("number of threads used to process HTTP requests (default: %d)", pParams.vNThreadsHttp),
                 [](common_params &pParams, int pValue) { pParams.vNThreadsHttp = pValue; })
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_ARG_THREADS_HTTP"));
  vAddOpt(common_arg({"--cache-prompt"}, {"--no-cache-prompt"},
                     fStringFormat("whether to enable prompt caching (default: %s)",
                                   pParams.vCachePrompt ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vCachePrompt = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CACHE_PROMPT"));
  vAddOpt(common_arg({"--cache-reuse"}, "N",
                     fStringFormat("min chunk size to attempt reusing from the cache via KV shifting, requires prompt "
                                   "caching to be enabled (default: %d)\n"
                                   "[(card)](https://ggml.ai/f0.png)",
                                   pParams.vNCacheReuse),
                     [](common_params &pParams, int pValue) { pParams.vNCacheReuse = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_CACHE_REUSE"));
  vAddOpt(common_arg({"--metrics"},
                     fStringFormat("enable prometheus compatible metrics endpoint (default: %s)",
                                   pParams.vEndpointMetrics ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vEndpointMetrics = true; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_ENDPOINT_METRICS"));
  vAddOpt(common_arg({"--props"},
                     fStringFormat("enable changing global properties via POST /props (default: %s)",
                                   pParams.vEndpointProps ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vEndpointProps = true; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_ENDPOINT_PROPS"));
  vAddOpt(common_arg({"--slots"}, {"--no-slots"},
                     fStringFormat("expose slots monitoring endpoint (default: %s)",
                                   pParams.vEndpointSlots ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vEndpointSlots = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_ENDPOINT_SLOTS"));
  vAddOpt(common_arg({"--slot-save-path"}, "PATH", "path to save slot kv cache (default: disabled)",
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
                     })
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--media-path"}, "PATH",
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
                     })
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--media-allow-private-urls"},
                     "let remote media URLs in requests (image_url, input_audio, input_video) reach loopback, private, "
                     "link-local and other non-public addresses; by default only public addresses are fetched - do "
                     "not enable in untrusted environments (default: disabled)",
                     [](common_params &pParams) { pParams.vMediaAllowPrivateUrls = true; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MEDIA_ALLOW_PRIVATE_URLS"));
  vAddOpt(common_arg({"--models-dir"}, "PATH", "directory containing models for the router server (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModelsDir = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MODELS_DIR"));
  vAddOpt(common_arg({"--models-preset"}, "PATH",
                     "path to INI file containing model presets for the router server (default: disabled)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vModelsPreset = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MODELS_PRESET"));
  vAddOpt(
      common_arg({"--models-max"}, "N",
                 fStringFormat(
                     "for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)",
                     pParams.vModelsMax),
                 [](common_params &pParams, int pValue) { pParams.vModelsMax = pValue; })
          .mSetExamples({cMgsExampleServer})
          .mSetEnv("MGS_ARG_MODELS_MAX"));
  vAddOpt(common_arg({"--models-autoload"}, {"--no-models-autoload"},
                     fStringFormat("for router server, whether to automatically load models (default: %s)",
                                   pParams.vModelsAutoload ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vModelsAutoload = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_MODELS_AUTOLOAD"));
  vAddOpt(common_arg({"--reasoning-format"}, "FORMAT",
                     "controls whether thought tags are allowed and/or extracted from the response, and in which "
                     "format they're returned; one of:\n"
                     "- none: leaves thoughts unparsed in `message.content`\n"
                     "- deepseek: puts thoughts in `message.reasoning_content`\n"
                     "(default: auto)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vReasoningFormat = fCommonReasoningFormatFromName(pValue);
                     })
              .mSetExamples({cMgsExampleServer, cMgsExampleCompletion, cMgsExampleCli})
              .mSetEnv("MGS_ARG_REASONING_FORMAT"));
  vAddOpt(
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
          .mSetExamples({cMgsExampleServer, cMgsExampleCompletion, cMgsExampleCli})
          .mSetEnv("MGS_ARG_REASONING"));
  vAddOpt(common_arg(
              {"--reasoning-budget"}, "N",
              "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
              [](common_params &pParams, int pValue) {
                if (pValue < -1) {
                  throw std::invalid_argument("invalid value");
                }
                pParams.vSampling.vReasoningBudgetTokens = pValue;
              })
              .mSetExamples({cMgsExampleServer, cMgsExampleCompletion, cMgsExampleCli})
              .mSetEnv("MGS_ARG_REASONING_BUDGET"));
  vAddOpt(
      common_arg(
          {"--reasoning-budget-message"}, "MESSAGE",
          "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
          [](common_params &pParams, const std::string &pValue) { pParams.vSampling.vReasoningBudgetMessage = pValue; })
          .mSetExamples({cMgsExampleServer, cMgsExampleCompletion, cMgsExampleCli})
          .mSetEnv("MGS_ARG_REASONING_BUDGET_MESSAGE"));
  vAddOpt(common_arg({"--reasoning-preserve"}, {"--no-reasoning-preserve"},
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
              .mSetExamples({cMgsExampleServer, cMgsExampleCompletion, cMgsExampleCli})
              .mSetEnv("MGS_ARG_REASONING_PRESERVE"));
  vAddOpt(
      common_arg({"--chat-template"}, "JINJA_TEMPLATE",
                 "set custom jinja chat template (default: template taken from model's metadata; chatml when it has "
                 "none)\n"
                 "if suffix/prefix are specified, template will be disabled\n"
                 "the name chatml selects the built-in ChatML template",
                 [](common_params &pParams, const std::string &pValue) { pParams.chat_template = pValue; })
          .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer, cMgsExampleMtmd})
          .mSetEnv("MGS_ARG_CHAT_TEMPLATE"));
  vAddOpt(
      common_arg({"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
                 "set custom jinja chat template file (default: template taken from model's metadata; chatml when it "
                 "has none)\n"
                 "if suffix/prefix are specified, template will be disabled",
                 [](common_params &pParams, const std::string &pValue) { pParams.chat_template = fReadFile(pValue); })
          .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer})
          .mSetEnv("MGS_ARG_CHAT_TEMPLATE_FILE"));
  vAddOpt(
      common_arg({"--skip-chat-parsing"}, {"--no-skip-chat-parsing"},
                 fStringFormat(
                     "force a pure content parser, even if a Jinja template is specified; model will output everything "
                     "in the content section, including any reasoning and/or tool calls (default: disabled)"),
                 [](common_params &pParams, bool pValue) { pParams.vForcePureContentParser = pValue; })
          .mSetExamples({cMgsExampleCompletion, cMgsExampleCli, cMgsExampleServer})
          .mSetEnv("MGS_ARG_SKIP_CHAT_PARSING"));
  vAddOpt(common_arg({"--prefill-assistant"}, {"--no-prefill-assistant"},
                     fStringFormat("whether to prefill the assistant's response if the last message is an assistant "
                                   "message (default: prefill enabled)\n"
                                   "when this flag is set, if the last message is an assistant message then it will be "
                                   "treated as a full message and not prefilled\n"),
                     [](common_params &pParams, bool pValue) { pParams.vPrefillAssistant = pValue; })
              .mSetExamples({cMgsExampleServer})
              .mSetEnv("MGS_ARG_PREFILL_ASSISTANT"));
  vAddOpt(
      common_arg(
          {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
          fStringFormat("how much the prompt of a request must match the prompt of a slot in order to use that "
                        "slot (default: %.2f, 0.0 = disabled)\n",
                        pParams.vSlotPromptSimilarity),
          [](common_params &pParams, const std::string &pValue) { pParams.vSlotPromptSimilarity = std::stof(pValue); })
          .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--lora-init-without-apply"},
                     fStringFormat(
                         "load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)",
                         pParams.vLoraInitWithoutApply ? "enabled" : "disabled"),
                     [](common_params &pParams) { pParams.vLoraInitWithoutApply = true; })
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--sleep-idle-seconds"}, "SECONDS",
                     fStringFormat(
                         "number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)",
                         pParams.vSleepIdleSeconds),
                     [](common_params &pParams, int pValue) {
                       if (pValue == 0 || pValue < -1) {
                         throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
                       }
                       pParams.vSleepIdleSeconds = pValue;
                     })
              .mSetExamples({cMgsExampleServer}));
  vAddOpt(common_arg({"--simple-io"}, "use basic IO for better compatibility in subprocesses and limited consoles",
                     [](common_params &pParams) { pParams.vSimpleIo = true; })
              .mSetExamples({cMgsExampleCompletion, cMgsExampleCli}));
  vAddOpt(common_arg({"--positive-file"}, "FNAME",
                     fStringFormat("positive prompts file, one prompt per line (default: '%s')",
                                   pParams.vCvectorPositiveFile.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCvectorPositiveFile = pValue; })
              .mSetExamples({cMgsExampleCvectorGenerator}));
  vAddOpt(common_arg({"--negative-file"}, "FNAME",
                     fStringFormat("negative prompts file, one prompt per line (default: '%s')",
                                   pParams.vCvectorNegativeFile.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.vCvectorNegativeFile = pValue; })
              .mSetExamples({cMgsExampleCvectorGenerator}));
  vAddOpt(
      common_arg({"--pca-batch"}, "N",
                 fStringFormat("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)",
                               pParams.vNPcaBatch),
                 [](common_params &pParams, int pValue) { pParams.vNPcaBatch = pValue; })
          .mSetExamples({cMgsExampleCvectorGenerator}));
  vAddOpt(common_arg({"--pca-iter"}, "N",
                     fStringFormat("number of iterations used for PCA (default: %d)", pParams.vNPcaIterations),
                     [](common_params &pParams, int pValue) { pParams.vNPcaIterations = pValue; })
              .mSetExamples({cMgsExampleCvectorGenerator}));
  vAddOpt(common_arg({"--method"}, "{pca, mean}", "dimensionality reduction method to be used (default: pca)",
                     [](common_params &pParams, const std::string &pValue) {
                       /**/ if (pValue == "pca") {
                         pParams.vCvectorDimreMethod = cDimreMethodPca;
                       } else if (pValue == "mean") {
                         pParams.vCvectorDimreMethod = cDimreMethodMean;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
              .mSetExamples({cMgsExampleCvectorGenerator}));
  vAddOpt(common_arg({"--output-format"}, "{md,jsonl}", "output format for batched-bench results (default: md)",
                     [](common_params &pParams, const std::string &pValue) {
                       /**/ if (pValue == "jsonl") {
                         pParams.vBatchedBenchOutputJsonl = true;
                       } else if (pValue == "md") {
                         pParams.vBatchedBenchOutputJsonl = false;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
              .mSetExamples({cMgsExampleBench}));
  vAddOpt(common_arg({"--log-disable"}, "Log disable", [](common_params &) { fCommonLogPause(fCommonLogMain()); }));
  vAddOpt(common_arg({"--log-file"}, "FNAME", "Log to file", [](common_params &, const std::string &pValue) {
            fCommonLogSetFile(fCommonLogMain(), pValue.c_str());
          }).mSetEnv("MGS_ARG_LOG_FILE"));
  vAddOpt(
      common_arg({"--log-prompts-dir"}, "PATH",
                 "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)",
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.path_prompts_log_dir = pValue;
                   std::error_code vEc;
                   std::filesystem::create_directories(pValue, vEc);
                   if (vEc) {
                     fprintf(stderr, "warning: failed to create prompts-log-dir '%s': %s\n", pValue.c_str(),
                             vEc.message().c_str());
                   }
                 })
          .mSetExamples({cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--log-colors"}, "[on|off|auto]",
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
  vAddOpt(common_arg({"-v", "--verbose"},
                     "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
                     [](common_params &pParams) {
                       pParams.vVerbosity = INT_MAX;
                       fCommonLogSetVerbosityThold(INT_MAX);
                     }));
  vAddOpt(common_arg({"--offline"}, "Offline mode: forces use of cache, prevents network access",
                     [](common_params &pParams) { pParams.vOffline = true; })
              .mSetExamples({cMgsExampleCommon, cMgsExampleDownload, cMgsExampleTokenize})
              .mSetEnv("MGS_ARG_OFFLINE"));
  vAddOpt(common_arg(
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
              .mSetEnv("MGS_ARG_LOG_VERBOSITY"));
  vAddOpt(common_arg({"--log-prefix"}, {"--no-log-prefix"}, "Enable prefix in log messages",
                     [](common_params &, bool pValue) { fCommonLogSetPrefix(fCommonLogMain(), pValue); })
              .mSetEnv("MGS_ARG_LOG_PREFIX"));
  vAddOpt(common_arg({"--log-timestamps"}, {"--no-log-timestamps"}, "Enable timestamps in log messages",
                     [](common_params &, bool pValue) { fCommonLogSetTimestamps(fCommonLogMain(), pValue); })
              .mSetEnv("MGS_ARG_LOG_TIMESTAMPS"));

  //
  // speculative parameters
  //

  vAddOpt(common_arg({"--spec-draft-hf"}, "<user>/<model>[:quant]",
                     "Same as --hf-repo, but for the draft model (default: unused)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vMparams.vHfRepo = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_HF_REPO"));
  vAddOpt(common_arg({"--spec-draft-threads"}, "N",
                     "number of threads to use during generation (default: same as --threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparams.vNThreads = pValue;
                       if (pParams.vSpeculative.vDraft.vCpuparams.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparams.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-threads-batch"}, "N",
                     "number of threads to use during batch and prompt processing (default: same as "
                     "--spec-draft-threads)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = pValue;
                       if (pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-cpu-mask"}, "M",
                     "Draft model CPU affinity mask. Complements --spec-draft-cpu-range (default: same as --cpu-mask)",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-cpu-range"}, "lo-hi",
                     "Ranges of CPUs for affinity. Complements --spec-draft-cpu-mask",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(
      common_arg({"--spec-draft-cpu-strict"}, "<0|1>",
                 "Use strict CPU placement for draft model (default: same as --cpu-strict)",
                 [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparams.vStrictCpu = pValue; })
          .mSetSpec()
          .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-prio"}, "N",
                     fStringFormat(
                         "set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                         pParams.vSpeculative.vDraft.vCpuparams.vPriority),
                     [](common_params &pParams, int pPrio) {
                       if (pPrio < 0 || pPrio > 3) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vDraft.vCpuparams.vPriority = (enum ggml_sched_priority)pPrio;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-poll"}, "<0|1>",
                     "Use polling to wait for draft model work (default: same as --poll)",
                     [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparams.vPoll = pValue; })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-cpu-mask-batch"}, "M",
                     "Draft model CPU affinity mask. Complements --spec-draft-cpu-range-batch (default: same as "
                     "--cpu-mask)",
                     [](common_params &pParams, const std::string &pMask) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-cpu-range-batch"}, "lo-hi",
                     "Ranges of CPUs for affinity. Complements --spec-draft-cpu-mask-batch",
                     [](common_params &pParams, const std::string &pRange) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vSpeculative.vDraft.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative}));
  vAddOpt(common_arg({"--spec-draft-cpu-strict-batch"}, "<0|1>",
                     "Use strict CPU placement for draft model (default: --spec-draft-cpu-strict)",
                     [](common_params &pParams, int pValue) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vStrictCpu = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-prio-batch"}, "N",
                     fStringFormat(
                         "set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                         pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority),
                     [](common_params &pParams, int pPrio) {
                       if (pPrio < 0 || pPrio > 3) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority = (enum ggml_sched_priority)pPrio;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(
      common_arg({"--spec-draft-poll-batch"}, "<0|1>",
                 "Use polling to wait for draft model work (default: --spec-draft-poll)",
                 [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vCpuparamsBatch.vPoll = pValue; })
          .mSetSpec()
          .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(
      common_arg({"--spec-draft-type-k"}, "TYPE",
                 fStringFormat("KV cache data type for K for the draft model\n"
                               "allowed values: %s\n"
                               "(default: %s)",
                               fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vSpeculative.vDraft.vCacheTypeK)),
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vSpeculative.vDraft.vCacheTypeK = fKvCacheTypeFromStr(pValue);
                 })
          .mSetEnv("MGS_ARG_SPEC_DRAFT_TYPE_K"));
  vAddOpt(
      common_arg({"--spec-draft-type-v"}, "TYPE",
                 fStringFormat("KV cache data type for V for the draft model\n"
                               "allowed values: %s\n"
                               "(default: %s)",
                               fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vSpeculative.vDraft.vCacheTypeV)),
                 [](common_params &pParams, const std::string &pValue) {
                   pParams.vSpeculative.vDraft.vCacheTypeV = fKvCacheTypeFromStr(pValue);
                 })
          .mSetEnv("MGS_ARG_SPEC_DRAFT_TYPE_V"));
  vAddOpt(common_arg({"--spec-draft-override-tensor"}, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type for draft model",
                     [](common_params &pParams, const std::string &pValue) {
                       fParseTensorBufferOverrides(pValue, pParams.vSpeculative.vDraft.lTensorBuftOverrides);
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-draft-cpu-moe"},
                     "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
                     [](common_params &pParams) {
                       pParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back(fLlmFfnExpsCpuOverride());
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_CPU_MOE"));
  vAddOpt(common_arg({"--spec-draft-n-cpu-moe"}, "N",
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
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_CPU_MOE"));

  vAddOpt(common_arg({"--spec-draft-n-max"}, "N",
                     fStringFormat("number of tokens to draft for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMax),
                     [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vNMax = pValue; })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleLookup, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_MAX"));
  vAddOpt(common_arg({"--spec-draft-n-min"}, "N",
                     fStringFormat("minimum number of draft tokens to use for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMin),
                     [](common_params &pParams, int pValue) { pParams.vSpeculative.vDraft.vNMin = pValue; })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleLookup, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_N_MIN"));

  vAddOpt(common_arg({"--spec-draft-p-split"}, "P",
                     fStringFormat("speculative decoding split probability (default: %.2f)",
                                   (double)pParams.vSpeculative.vDraft.vPSplit),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vPSplit = std::stof(pValue);
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_P_SPLIT"));
  vAddOpt(common_arg({"--spec-draft-p-min"}, "P",
                     fStringFormat("minimum speculative decoding probability (greedy) (default: %.2f)",
                                   (double)pParams.vSpeculative.vDraft.vPMin),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vPMin = std::stof(pValue);
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_P_MIN"));
  vAddOpt(common_arg({"--spec-draft-backend-sampling"}, {"--no-spec-draft-backend-sampling"},
                     fStringFormat("offload draft sampling to the backend (default: %s)",
                                   pParams.vSpeculative.vDraft.vBackendSampling ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vSpeculative.vDraft.vBackendSampling = pValue; })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
  vAddOpt(common_arg({"--spec-draft-device"}, "<dev1,dev2,..>",
                     "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
                     "use --list-devices to see a list of available devices",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.lDevices = fParseDeviceList(pValue);
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  GGML_ASSERT(pParams.vSpeculative.vDraft.vNGpuLayers <
              0); // string_format would need to be extended for a default >= 0
  vAddOpt(common_arg({"--spec-draft-ngl"}, "N",
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
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_NGL"));
  vAddOpt(common_arg({"--spec-draft-model"}, "FNAME", "draft model for speculative decoding (default: unused)",
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vSpeculative.vDraft.vMparams.vPath = pValue;
                       pParams.vSpeculative.vDraft.vMparams.vHfFile = pValue; // will be used if --spec-draft-hf is set
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_DRAFT_MODEL"));
  vAddOpt(common_arg({"--spec-mtp-auto"}, {"--no-spec-mtp-auto"},
                     fStringFormat("automatically enable speculative decoding when the GGUF contains a compatible "
                                   "embedded MTP head (default: %s)",
                                   pParams.vSpeculative.vAutoMtp ? "enabled" : "disabled"),
                     [](common_params &pParams, bool pValue) { pParams.vSpeculative.vAutoMtp = pValue; })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_MTP_AUTO"));
  vAddOpt(common_arg({"--spec-type"}, fCommonSpeculativeAllTypesStr(),
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
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli})
              .mSetEnv("MGS_ARG_SPEC_TYPE"));
  vAddOpt(common_arg(
              {"--spec-ngram-mod-n-min"}, "N",
              fStringFormat("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                            pParams.vSpeculative.vNgramMod.vNMin),
              [](common_params &pParams, int pValue) {
                if (pValue < 0 || pValue > 1024) {
                  throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMod.vNMin = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg(
              {"--spec-ngram-mod-n-max"}, "N",
              fStringFormat("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                            pParams.vSpeculative.vNgramMod.vNMax),
              [](common_params &pParams, int pValue) {
                if (pValue < 0 || pValue > 1024) {
                  throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMod.vNMax = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-ngram-mod-n-match"}, "N",
                     fStringFormat("ngram-mod lookup length (default: %d)", pParams.vSpeculative.vNgramMod.vNMatch),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 1024) {
                         throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                       }
                       pParams.vSpeculative.vNgramMod.vNMatch = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));

  vAddOpt(common_arg(
              {"--spec-ngram-simple-size-n"}, "N",
              fStringFormat("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)",
                            pParams.vSpeculative.vNgramSimple.vSizeN),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramSimple.vSizeN = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg(
              {"--spec-ngram-simple-size-m"}, "N",
              fStringFormat("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramSimple.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramSimple.vSizeM = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-ngram-simple-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-simple speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramSimple.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramSimple.vMinHits = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));

  vAddOpt(common_arg(
              {"--spec-ngram-map-k-size-n"}, "N",
              fStringFormat("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK.vSizeN),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK.vSizeN = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg(
              {"--spec-ngram-map-k-size-m"}, "N",
              fStringFormat("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK.vSizeM = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-ngram-map-k-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-map-k speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK.vMinHits = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));

  vAddOpt(common_arg({"--spec-ngram-map-k4v-size-n"}, "N",
                     fStringFormat(
                         "ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)",
                         pParams.vSpeculative.vNgramMapK4v.vSizeN),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1 || pValue > 1024) {
                         throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                       }
                       pParams.vSpeculative.vNgramMapK4v.vSizeN = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg(
              {"--spec-ngram-map-k4v-size-m"}, "N",
              fStringFormat("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)",
                            pParams.vSpeculative.vNgramMapK4v.vSizeM),
              [](common_params &pParams, int pValue) {
                if (pValue < 1 || pValue > 1024) {
                  throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                }
                pParams.vSpeculative.vNgramMapK4v.vSizeM = pValue;
              })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));
  vAddOpt(common_arg({"--spec-ngram-map-k4v-min-hits"}, "N",
                     fStringFormat("minimum hits for ngram-map-k4v speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK4v.vMinHits),
                     [](common_params &pParams, int pValue) {
                       if (pValue < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK4v.vMinHits = pValue;
                     })
              .mSetSpec()
              .mSetExamples({cMgsExampleSpeculative, cMgsExampleServer, cMgsExampleCli}));

  //
  // TTS params
  //

  vAddOpt(common_arg({"-mv", "--model-vocoder"}, "FNAME", "vocoder model for audio generation (default: unused)",
                     [](common_params &pParams, const std::string &pValue) { pParams.vVocoder.vModel.vPath = pValue; })
              .mSetExamples({cMgsExampleTts, cMgsExampleServer}));
  vAddOpt(
      common_arg({"--tts-use-guide-tokens"}, "Use guide tokens to improve TTS word recall", [](common_params &pParams) {
        pParams.vVocoder.vUseGuideTokens = true;
      }).mSetExamples({cMgsExampleTts, cMgsExampleServer}));
  vAddOpt(common_arg({"--tts-speaker-file"}, "FNAME", "speaker file path for audio generation",
                     [](common_params &pParams, const std::string &pValue) { pParams.vVocoder.vSpeakerFile = pValue; })
              .mSetExamples({cMgsExampleTts}));

  //
  // diffusion params
  //

  vAddOpt(common_arg({"--diffusion-steps"}, "N",
                     fStringFormat("number of diffusion steps (default: %d)", pParams.vDiffusion.vSteps),
                     [](common_params &pParams, int pValue) { pParams.vDiffusion.vSteps = pValue; })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg({"--diffusion-visual"},
                     fStringFormat("enable visual diffusion mode (show progressive generation) (default: %s)",
                                   pParams.vDiffusion.vVisualMode ? "true" : "false"),
                     [](common_params &pParams) { pParams.vDiffusion.vVisualMode = true; })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(
      common_arg({"--diffusion-eps"}, "F",
                 fStringFormat("epsilon for timesteps (default: %.6f)", (double)pParams.vDiffusion.vEps),
                 [](common_params &pParams, const std::string &pValue) { pParams.vDiffusion.vEps = std::stof(pValue); })
          .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg(
              {"--diffusion-algorithm"}, "N",
              fStringFormat("diffusion algorithm: 0=DIFFUSION_ALGORITHM_ORIGIN, 1=DIFFUSION_ALGORITHM_ENTROPY_BASED, "
                            "2=DIFFUSION_ALGORITHM_MARGIN_BASED, 3=DIFFUSION_ALGORITHM_RANDOM, "
                            "4=DIFFUSION_ALGORITHM_CONFIDENCE_BASED (default: %d)",
                            pParams.vDiffusion.vAlgorithm),
              [](common_params &pParams, int pValue) { pParams.vDiffusion.vAlgorithm = pValue; })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg({"--diffusion-alg-temp"}, "F",
                     fStringFormat("dream algorithm temperature (default: %.3f)", (double)pParams.vDiffusion.vAlgTemp),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vDiffusion.vAlgTemp = std::stof(pValue);
                     })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg({"--diffusion-block-length"}, "N",
                     fStringFormat("llada block length for generation (default: %d)", pParams.vDiffusion.vBlockLength),
                     [](common_params &pParams, int pValue) { pParams.vDiffusion.vBlockLength = pValue; })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(
      common_arg(
          {"--diffusion-cfg-scale"}, "F",
          fStringFormat("llada classifier-free guidance scale (default: %.3f)", (double)pParams.vDiffusion.vCfgScale),
          [](common_params &pParams, const std::string &pValue) { pParams.vDiffusion.vCfgScale = std::stof(pValue); })
          .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg({"--diffusion-add-gumbel-noise"}, "F",
                     fStringFormat("add gumbel noise to the logits if temp > 0.0 (default: %s)",
                                   pParams.vDiffusion.vAddGumbelNoise ? "true" : "false"),
                     [](common_params &pParams, const std::string &pValue) {
                       pParams.vDiffusion.vAddGumbelNoise = std::stof(pValue);
                     })
              .mSetExamples({cMgsExampleDiffusion}));
  vAddOpt(common_arg({"-lr", "--learning-rate"}, "ALPHA",
                     fStringFormat(
                         "adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)",
                         (double)pParams.vLr.vLr0),
                     [](common_params &pParams, const std::string &pValue) { pParams.vLr.vLr0 = std::stof(pValue); })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(common_arg({"-lr-min", "--learning-rate-min"}, "ALPHA",
                     fStringFormat("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
                                   (double)pParams.vLr.vLrMin),
                     [](common_params &pParams, const std::string &pValue) { pParams.vLr.vLrMin = std::stof(pValue); })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(common_arg(
              {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
              fStringFormat(
                  "(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)",
                  (double)pParams.vLr.vDecayEpochs),
              [](common_params &pParams, const std::string &pValue) { pParams.vLr.vDecayEpochs = std::stof(pValue); })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(
      common_arg({"-wd", "--weight-decay"}, "WD",
                 fStringFormat(
                     "adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).",
                     (double)pParams.vLr.vWd),
                 [](common_params &pParams, const std::string &pValue) { pParams.vLr.vWd = std::stof(pValue); })
          .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(common_arg({"-val-split", "--val-split"}, "FRACTION",
                     fStringFormat("fraction of data to use as validation set for training (default: %.2g).",
                                   (double)pParams.vValSplit),
                     [](common_params &pParams, const std::string &pValue) { pParams.vValSplit = std::stof(pValue); })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(common_arg({"-epochs", "--epochs"}, "N",
                     fStringFormat("optimizer max # of epochs (default: %d)", pParams.vLr.vEpochs),
                     [](common_params &pParams, int pEpochs) { pParams.vLr.vEpochs = pEpochs; })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(common_arg({"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
                     [](common_params &pParams, const std::string &pName) {
                       pParams.vOptimizer = fCommonOptGetOptimizer(pName.c_str());
                       if (pParams.vOptimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                         throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
                       }
                     })
              .mSetExamples({cMgsExampleFinetune}));
  vAddOpt(
      common_arg({"--check"},
                 fStringFormat("check rather than generate results (default: %s)", pParams.vCheck ? "true" : "false"),
                 [](common_params &pParams) { pParams.vCheck = true; })
          .mSetExamples({cMgsExampleResults}));
  vAddOpt(common_arg({"--save-logits"},
                     fStringFormat("save final logits to files for verification (default: %s)",
                                   pParams.save_logits ? "true" : "false"),
                     [](common_params &pParams) { pParams.save_logits = true; })
              .mSetExamples({cMgsExampleDebug}));
  vAddOpt(common_arg({"--logits-output-dir"}, "PATH",
                     fStringFormat("directory for saving logits output files (default: %s)",
                                   pParams.logits_output_dir.c_str()),
                     [](common_params &pParams, const std::string &pValue) { pParams.logits_output_dir = pValue; })
              .mSetExamples({cMgsExampleDebug}));
  vAddOpt(common_arg({"--tensor-filter"}, "REGEX",
                     "filter tensor names for debug output (regex pattern, can be specified multiple times)",
                     [](common_params &pParams, const std::string &pValue) { pParams.tensor_filter.push_back(pValue); })
              .mSetExamples({cMgsExampleDebug}));

  // presets
  vAddOpt(common_arg({"--tts-oute-default"},
                     fStringFormat("use default OuteTTS models (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "OuteAI/OuteTTS-0.2-500M-GGUF";
                       pParams.vModel.vHfFile = "OuteTTS-0.2-500M-Q8_0.gguf";
                       pParams.vVocoder.vModel.vHfRepo = "ggml-org/WavTokenizer";
                       pParams.vVocoder.vModel.vHfFile = "WavTokenizer-Large-75-F16.gguf";
                     })
              .mSetExamples({cMgsExampleTts}));

  vAddOpt(common_arg({"--embd-gemma-default"},
                     fStringFormat("use default EmbeddingGemma model (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
                       pParams.vModel.vHfFile = "embeddinggemma-300M-qat-Q4_0.gguf";
                       pParams.vPort = 8011;
                       pParams.vNUbatch = 2048;
                       pParams.vNBatch = 2048;
                       pParams.vNParallel = 32;
                       pParams.vNCtx = 2048 * pParams.vNParallel;
                       pParams.vVerbosePrompt = true;
                       pParams.vEmbedding = true;
                     })
              .mSetExamples({cMgsExampleEmbedding, cMgsExampleServer}));

  vAddOpt(common_arg({"--fim-qwen-1.5b-default"},
                     fStringFormat("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
                       pParams.vModel.vHfFile = "qwen2.5-coder-1.5b-q8_0.gguf";
                       pParams.vPort = 8012;
                       pParams.vNUbatch = 1024;
                       pParams.vNBatch = 1024;
                       pParams.vNCtx = 0;
                       pParams.vNCacheReuse = 256;
                     })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg({"--fim-qwen-3b-default"},
                     fStringFormat("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
                       pParams.vModel.vHfFile = "qwen2.5-coder-3b-q8_0.gguf";
                       pParams.vPort = 8012;
                       pParams.vNUbatch = 1024;
                       pParams.vNBatch = 1024;
                       pParams.vNCtx = 0;
                       pParams.vNCacheReuse = 256;
                     })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg({"--fim-qwen-7b-default"},
                     fStringFormat("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
                       pParams.vModel.vHfFile = "qwen2.5-coder-7b-q8_0.gguf";
                       pParams.vPort = 8012;
                       pParams.vNUbatch = 1024;
                       pParams.vNBatch = 1024;
                       pParams.vNCtx = 0;
                       pParams.vNCacheReuse = 256;
                     })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg({"--fim-qwen-7b-spec"},
                     fStringFormat("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download "
                                   "weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
                       pParams.vModel.vHfFile = "qwen2.5-coder-7b-q8_0.gguf";
                       pParams.vSpeculative.vDraft.vMparams.vHfRepo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
                       pParams.vSpeculative.vDraft.vMparams.vHfFile = "qwen2.5-coder-0.5b-q8_0.gguf";
                       pParams.vPort = 8012;
                       pParams.vNUbatch = 1024;
                       pParams.vNBatch = 1024;
                       pParams.vNCtx = 0;
                       pParams.vNCacheReuse = 256;
                     })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg({"--fim-qwen-14b-spec"},
                     fStringFormat("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download "
                                   "weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
                       pParams.vModel.vHfFile = "qwen2.5-coder-14b-q8_0.gguf";
                       pParams.vSpeculative.vDraft.vMparams.vHfRepo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
                       pParams.vSpeculative.vDraft.vMparams.vHfFile = "qwen2.5-coder-0.5b-q8_0.gguf";
                       pParams.vPort = 8012;
                       pParams.vNUbatch = 1024;
                       pParams.vNBatch = 1024;
                       pParams.vNCtx = 0;
                       pParams.vNCacheReuse = 256;
                     })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg(
              {"--fim-qwen-30b-default"},
              fStringFormat("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
              [](common_params &pParams) {
                pParams.vModel.vHfRepo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
                pParams.vModel.vHfFile = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
                pParams.vPort = 8012;
                pParams.vNUbatch = 1024;
                pParams.vNBatch = 1024;
                pParams.vNCtx = 0;
                pParams.vNCacheReuse = 256;
              })
              .mSetExamples({cMgsExampleServer}));

  vAddOpt(common_arg({"--gpt-oss-20b-default"},
                     fStringFormat("use gpt-oss-20b (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/gpt-oss-20b-GGUF";
                       pParams.vModel.vHfFile = "gpt-oss-20b-mxfp4.gguf";
                       pParams.vPort = 8013;
                       pParams.vNUbatch = 2048;
                       pParams.vNBatch = 32768;
                       pParams.vNParallel = 2;
                       pParams.vNCtx = 131072 * pParams.vNParallel;
                       pParams.vSampling.vTemp = 1.0f;
                       pParams.vSampling.vTopP = 1.0f;
                       pParams.vSampling.vTopK = 0;
                       pParams.vSampling.vMinP = 0.01f;
                     })
              .mSetExamples({cMgsExampleServer, cMgsExampleCli}));

  vAddOpt(common_arg({"--gpt-oss-120b-default"},
                     fStringFormat("use gpt-oss-120b (note: can download weights from the internet)"),
                     [](common_params &pParams) {
                       pParams.vModel.vHfRepo = "ggml-org/gpt-oss-120b-GGUF";
                       pParams.vPort = 8013;
                       pParams.vNUbatch = 2048;
                       pParams.vNBatch = 32768;
                       pParams.vNParallel = 2;
                       pParams.vNCtx = 131072 * pParams.vNParallel;
                       pParams.vSampling.vTemp = 1.0f;
                       pParams.vSampling.vTopP = 1.0f;
                       pParams.vSampling.vTopK = 0;
                       pParams.vSampling.vMinP = 0.01f;
                     })
              .mSetExamples({cMgsExampleServer, cMgsExampleCli}));

  vAddOpt(common_arg({"--spec-default"}, fStringFormat("enable default speculative decoding config"),
                     [](common_params &pParams) {
                       pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeNgramMod);
                       pParams.vSpeculative.vNgramMod.vNMatch = 24;
                       pParams.vSpeculative.vNgramMod.vNMin = 48;
                       pParams.vSpeculative.vNgramMod.vNMax = 64;

                       // TODO: not sure if this is a good config - explore more settings and potentially enable it
                       // pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeNgramMapK4v);
                       // pParams.vSpeculative.vNgramMapK4v.vSizeN = 8;
                       // pParams.vSpeculative.vNgramMapK4v.vSizeM = 24;
                       // pParams.vSpeculative.vNgramMapK4v.vMinHits = 2;
                     })
              .mSetExamples({cMgsExampleServer, cMgsExampleCli}));

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
