#pragma once

#include "common.h"
#include "download.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// pseudo-env variable to identify preset-only arguments
#define COMMON_ARG_PRESET_LOAD_ON_STARTUP "__PRESET_LOAD_ON_STARTUP"
#define COMMON_ARG_PRESET_STOP_TIMEOUT "__PRESET_STOP_TIMEOUT"

//
// CLI argument parsing
//

struct common_arg {
  std::set<enum mgs_example> sExamples = {cMgsExampleCommon};
  std::set<enum mgs_example> sExcludes = {};
  std::vector<const char *> lArgs;
  std::vector<const char *> lArgsNeg; // for negated args like --no-xxx
  const char *vValueHint = nullptr;   // help text or example for arg value
  const char *vValueHint2 = nullptr;  // for second arg value
  const char *vEnv = nullptr;
  std::string vHelp;
  bool vIsSampling = false;   // is current arg a sampling param?
  bool vIsSpec = false;       // is current arg a speculative decoding param?
  bool vIsPresetOnly = false; // is current arg preset-only (not treated as CLI arg)
  void (*vHandlerVoid)(common_params &pParams) = nullptr;
  void (*vHandlerString)(common_params &pParams, const std::string &) = nullptr;
  void (*vHandlerStrStr)(common_params &pParams, const std::string &, const std::string &) = nullptr;
  void (*vHandlerInt)(common_params &pParams, int) = nullptr;
  void (*vHandlerBool)(common_params &pParams, bool) = nullptr;

  common_arg() = default;

  common_arg(const std::initializer_list<const char *> &pArgs, const char *pValueHint, const std::string &pHelp,
             void (*pHandler)(common_params &pParams, const std::string &))
      : lArgs(pArgs), vValueHint(pValueHint), vHelp(pHelp), vHandlerString(pHandler) {}

  common_arg(const std::initializer_list<const char *> &pArgs, const char *pValueHint, const std::string &pHelp,
             void (*pHandler)(common_params &pParams, int))
      : lArgs(pArgs), vValueHint(pValueHint), vHelp(pHelp), vHandlerInt(pHandler) {}

  common_arg(const std::initializer_list<const char *> &pArgs, const std::string &pHelp,
             void (*pHandler)(common_params &pParams))
      : lArgs(pArgs), vHelp(pHelp), vHandlerVoid(pHandler) {}

  common_arg(const std::initializer_list<const char *> &pArgs, const std::initializer_list<const char *> &pArgsNeg,
             const std::string &pHelp, void (*pHandler)(common_params &pParams, bool))
      : lArgs(pArgs), lArgsNeg(pArgsNeg), vHelp(pHelp), vHandlerBool(pHandler) {}

  // support 2 values for arg
  common_arg(const std::initializer_list<const char *> &pArgs, const char *pValueHint, const char *pValueHint2,
             const std::string &pHelp,
             void (*pHandler)(common_params &pParams, const std::string &, const std::string &))
      : lArgs(pArgs), vValueHint(pValueHint), vValueHint2(pValueHint2), vHelp(pHelp), vHandlerStrStr(pHandler) {}

  common_arg &mSetExamples(std::initializer_list<enum mgs_example> pExamples);
  common_arg &mSetExcludes(std::initializer_list<enum mgs_example> pExcludes);
  common_arg &mSetEnv(const char *pEnv);
  common_arg &mSetSampling();
  common_arg &mSetSpec();
  common_arg &mSetPresetOnly();
  bool mInExample(enum mgs_example pEx);
  bool mIsExclude(enum mgs_example pEx);
  bool mGetValueFromEnv(std::string &pOutput) const;
  bool mHasValueFromEnv() const;
  std::string mToString() const;

  // for using as key in std::map
  bool operator<(const common_arg &pOther) const {
    if (lArgs.empty() || pOther.lArgs.empty()) {
      return false;
    }
    return strcmp(lArgs[0], pOther.lArgs[0]) < 0;
  }
  bool operator==(const common_arg &pOther) const {
    if (lArgs.empty() || pOther.lArgs.empty()) {
      return false;
    }
    return strcmp(lArgs[0], pOther.lArgs[0]) == 0;
  }

  // get all args and env vars (including negated args/env)
  std::vector<std::string> mGetArgs() const;
  std::vector<std::string> mGetEnv() const;
};

namespace common_arg_utils {
bool fIsTruthy(const std::string &pValue);
bool fIsFalsey(const std::string &pValue);
bool fIsAutoy(const std::string &pValue);
} // namespace common_arg_utils

struct common_params_context {
  enum mgs_example vEx = cMgsExampleCommon;
  common_params &vParams;
  std::vector<common_arg> lOptions;
  void (*vPrintUsage)(int, char **) = nullptr;
  common_params_context(common_params &pParams) : vParams(pParams) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full
// usage message)
bool fCommonParamsParse(int pArgc, char **pArgv, common_params &pParams, mgs_example pEx,
                        void (*pPrintUsage)(int, char **) = nullptr);

// load all backends and print the list of available (non-CPU) devices to stdout
void fCommonPrintAvailableDevices();

// parse input arguments from CLI into a map
bool fCommonParamsToMap(int pArgc, char **pArgv, mgs_example pEx, std::map<common_arg, std::string> &pOutMap);

// populate preset-only arguments
// these arguments are not treated as command line arguments
void fCommonParamsAddPresetOptions(std::vector<common_arg> &pArgs);

struct common_models_handler {
  common_download_hf_plan vPlan;
  common_download_hf_plan vPlanSpec;
  common_download_hf_plan vPlanVoc;
  common_download_opts vOpts;
};

// initialize downloading opts and hf_plan if needed, but does not download anything yet
common_models_handler fCommonModelsHandlerInit(const common_params &pParams, mgs_example pCurrEx);

// check if the model is a preset repo (i.e. has a preset file)
bool fCommonModelsHandlerIsPresetRepo(const common_models_handler &pHandler);

// download and update params with the downloaded model path
void fCommonModelsHandlerApply(common_models_handler &pHandler, common_params &pParams,
                               common_download_callback *pCallback = nullptr);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context fCommonParamsParserInit(common_params &pParams, mgs_example pEx,
                                              void (*pPrintUsage)(int, char **) = nullptr);
