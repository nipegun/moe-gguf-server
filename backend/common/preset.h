#pragma once

#include "common.h"
#include "arg.h"

#include <string>
#include <vector>
#include <map>
#include <set>

//
// INI preset parser and writer
//

constexpr const char *cCommonPresetDefaultName = "default";

struct common_preset_context;

struct common_preset {
  std::string vName;

  // options are stored as common_arg to string mapping, representing CLI arg and its value
  std::map<common_arg, std::string> dOptions;

  // convert preset to CLI argument list
  std::vector<std::string> mToArgs(const std::string &pBinPath = "") const;

  // convert preset to INI format string
  std::string mToIni() const;

  // TODO: maybe implement to_env() if needed

  // modify preset options where argument is identified by its env variable
  void mSetOption(const common_preset_context &pCtx, const std::string &pEnv, const std::string &pValue);

  // unset option by its env variable
  void mUnsetOption(const std::string &pEnv);

  // get option value by its env variable, return false if not found
  bool mGetOption(const std::string &pEnv, std::string &pValue) const;

  // merge another preset into this one, overwriting existing options
  void mMerge(const common_preset &pOther);

  // apply preset options to common_params
  // optionally specify handled_keys to only apply a subset of options (identified by their env), if empty, apply all
  // options
  void mApplyToParams(common_params &pParams,
                      const std::set<std::string> &pHandledKeys = std::set<std::string>()) const;
};

// interface for multiple presets in one file
using common_presets = std::map<std::string, common_preset>;

// context for loading and editing presets
struct common_preset_context {
  common_params vDefaultParams; // unused for now
  common_params_context vCtxParams;
  std::map<std::string, common_arg> dKeyToOpt;

  bool vFilterAllowedKeys = false;
  std::set<std::string> sAllowedKeys;

  // if only_remote_allowed is true, only accept whitelisted keys
  common_preset_context(mgs_example pEx);

  // load presets from INI file
  common_presets mLoadFromIni(const std::string &pPath, common_preset &pGlobal) const;

  // generate presets from cached models
  common_presets mLoadFromCache() const;

  // generate presets from local models directory
  // for the directory structure, see "Using multiple models" in server/README.md
  common_presets mLoadFromModelsDir(const std::string &pModelsDir) const;

  // generate one preset from CLI arguments
  common_preset mLoadFromArgs(int pArgc, char **pArgv) const;

  // cascade multiple presets if exist on both: base < added
  // if preset does not exist in base, it will be added without modification
  common_presets mCascade(const common_presets &pBase, const common_presets &pAdded) const;

  // apply presets over a base preset (same idea as CSS cascading)
  common_presets mCascade(const common_preset &pBase, const common_presets &pPresets) const;
};
