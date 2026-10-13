#include "arg.h"
#include "preset.h"
#include "peg-parser.h"
#include "log.h"
#include "download.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>

static std::string fRmLeadingDashes(const std::string &pStr) {
  size_t vPos = 0;
  while (vPos < pStr.size() && pStr[vPos] == '-') {
    ++vPos;
  }
  return pStr.substr(vPos);
}

static std::string fCanonicalTag(const std::string &pTag) {
  static const std::regex cReTag("[-.]([A-Z0-9_]+)$", std::regex::icase);
  std::smatch vM;
  if (std::regex_search(pTag, vM, cReTag)) {
    std::string vCanon = vM[1].str();
    for (char &vC : vCanon) {
      vC = (char)std::toupper((unsigned char)vC);
    }
    return vCanon;
  }
  std::string vUpper = pTag;
  for (char &vC : vUpper) {
    vC = (char)std::toupper((unsigned char)vC);
  }
  return vUpper;
}

std::vector<std::string> common_preset::mToArgs(const std::string &pBinPath) const {
  std::vector<std::string> lArgs;

  if (!pBinPath.empty()) {
    lArgs.push_back(pBinPath);
  }

  for (const auto &[vOpt, vValue] : dOptions) {
    if (vOpt.vIsPresetOnly) {
      continue; // skip preset-only options (they are not CLI args)
    }

    // use the last arg as the main arg (i.e. --long-form)
    lArgs.push_back(vOpt.lArgs.back());

    // handle value(s)
    if (vOpt.vValueHint == nullptr && vOpt.vValueHint2 == nullptr) {
      // flag option, no value
      if (common_arg_utils::fIsFalsey(vValue)) {
        // use negative arg if available
        if (!vOpt.lArgsNeg.empty()) {
          lArgs.back() = vOpt.lArgsNeg.back();
        } else {
          // otherwise, skip the flag
          // TODO: maybe throw an error instead?
          lArgs.pop_back();
        }
      }
    }
    if (vOpt.vValueHint != nullptr) {
      // single value
      lArgs.push_back(vValue);
    }
    if (vOpt.vValueHint != nullptr && vOpt.vValueHint2 != nullptr) {
      throw std::runtime_error(fStringFormat(
          "common_preset::mToArgs(): option '%s' has two values, which is not supported yet", vOpt.lArgs.back()));
    }
  }

  return lArgs;
}

std::string common_preset::mToIni() const {
  std::ostringstream vSs;

  vSs << "[" << vName << "]\n";
  for (const auto &[vOpt, vValue] : dOptions) {
    auto vEspacedValue = vValue;
    fStringReplaceAll(vEspacedValue, "\n", "\\\n");
    vSs << fRmLeadingDashes(vOpt.lArgs.back()) << " = ";
    vSs << vEspacedValue << "\n";
  }
  vSs << "\n";

  return vSs.str();
}

void common_preset::mSetOption(const common_preset_context &pCtx, const std::string &pEnv, const std::string &pValue) {
  // try if option exists, update it
  for (auto &[vOpt, vVal] : dOptions) {
    if (vOpt.vEnv && pEnv == vOpt.vEnv) {
      vVal = pValue;
      return;
    }
  }
  // if option does not exist, we need to add it
  if (pCtx.dKeyToOpt.find(pEnv) == pCtx.dKeyToOpt.end()) {
    throw std::runtime_error(fStringFormat("%s: option with env '%s' not found in ctx_params", __func__, pEnv.c_str()));
  }
  dOptions[pCtx.dKeyToOpt.at(pEnv)] = pValue;
}

void common_preset::mUnsetOption(const std::string &pEnv) {
  for (auto vIt = dOptions.begin(); vIt != dOptions.end();) {
    const common_arg &vOpt = vIt->first;
    if (vOpt.vEnv && pEnv == vOpt.vEnv) {
      vIt = dOptions.erase(vIt);
      return;
    } else {
      ++vIt;
    }
  }
}

bool common_preset::mGetOption(const std::string &pEnv, std::string &pValue) const {
  for (const auto &[vOpt, vVal] : dOptions) {
    if (vOpt.vEnv && pEnv == vOpt.vEnv) {
      pValue = vVal;
      return true;
    }
  }
  return false;
}

void common_preset::mMerge(const common_preset &pOther) {
  for (const auto &[vOpt, vVal] : pOther.dOptions) {
    dOptions[vOpt] = vVal; // overwrite existing options
  }
}

void common_preset::mApplyToParams(common_params &pParams, const std::set<std::string> &pHandledKeys) const {
  for (const auto &[vOpt, vVal] : dOptions) {
    if (!pHandledKeys.empty()) {
      if (!vOpt.vEnv || pHandledKeys.find(vOpt.vEnv) == pHandledKeys.end()) {
        continue;
      }
    }
    // apply each option to params
    if (vOpt.vHandlerString) {
      vOpt.vHandlerString(pParams, vVal);
    } else if (vOpt.vHandlerInt) {
      vOpt.vHandlerInt(pParams, std::stoi(vVal));
    } else if (vOpt.vHandlerBool) {
      vOpt.vHandlerBool(pParams, common_arg_utils::fIsTruthy(vVal));
    } else if (vOpt.vHandlerStrStr) {
      // not supported yet
      throw std::runtime_error(fStringFormat("%s: option with two values is not supported yet", __func__));
    } else if (vOpt.vHandlerVoid) {
      vOpt.vHandlerVoid(pParams);
    } else {
      GGML_ABORT("unknown handler type");
    }
  }
}

static std::map<std::string, std::map<std::string, std::string>> fParseIniFromFile(const std::string &pPath) {
  std::map<std::string, std::map<std::string, std::string>> dParsed;

  if (!std::filesystem::exists(pPath)) {
    throw std::runtime_error("preset file does not exist: " + pPath);
  }

  std::ifstream vFile(pPath);
  if (!vFile.good()) {
    throw std::runtime_error("failed to open server preset file: " + pPath);
  }

  std::string vContents((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());

  static const auto cParser = fBuildPegParser([](auto &pP) {
    // newline ::= "\r\n" / "\n" / "\r"
    auto vNewline = pP.mRule("newline", pP.mLiteral("\r\n") | pP.mLiteral("\n") | pP.mLiteral("\r"));

    // ws ::= [ \t]*
    auto vWs = pP.mRule("ws", pP.mChars("[ \t]", 0, -1));

    // comment ::= [;#] (!newline .)*
    auto vComment = pP.mRule("comment", pP.mChars("[;#]", 1, 1) + pP.mZeroOrMore(pP.mNegate(vNewline) + pP.mAny()));

    // eol ::= ws comment? (newline / EOF)
    auto vEol = pP.mRule("eol", vWs + pP.mOptional(vComment) + (vNewline | pP.end()));

    // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
    auto vIdent = pP.mRule("ident", pP.mChars("[a-zA-Z_]", 1, 1) + pP.mChars("[a-zA-Z0-9_.-]", 0, -1));

    // value ::= (!eol-start .)*
    auto vEolStart = pP.mRule("eol-start", vWs + (pP.mChars("[;#]", 1, 1) | vNewline | pP.end()));
    auto vValue = pP.mRule("value", pP.mZeroOrMore(pP.mNegate(vEolStart) + pP.mAny()));

    // header-line ::= "[" ws ident ws "]" eol
    auto vHeaderLine =
        pP.mRule("header-line", "[" + vWs + pP.mTag("section-name", pP.mChars("[^]]")) + vWs + "]" + vEol);

    // kv-line ::= ident ws "=" ws value eol
    auto vKvLine = pP.mRule("kv-line", pP.mTag("key", vIdent) + vWs + "=" + vWs + pP.mTag("value", vValue) + vEol);

    // comment-line ::= ws comment (newline / EOF)
    auto vCommentLine = pP.mRule("comment-line", vWs + vComment + (vNewline | pP.end()));

    // blank-line ::= ws (newline / EOF)
    auto vBlankLine = pP.mRule("blank-line", vWs + (vNewline | pP.end()));

    // line ::= header-line / kv-line / comment-line / blank-line
    auto vLine = pP.mRule("line", vHeaderLine | vKvLine | vCommentLine | vBlankLine);

    // ini ::= line* EOF
    auto vIni = pP.mRule("ini", pP.mZeroOrMore(vLine) + pP.end());

    return vIni;
  });

  common_peg_parse_context vCtx(vContents);
  const auto cResult = cParser.mParse(vCtx);
  if (!cResult.mSuccess()) {
    throw std::runtime_error("failed to parse server config file: " + pPath);
  }

  std::string vCurrentSection = cCommonPresetDefaultName;
  std::string vCurrentKey;

  vCtx.vAst.mVisit(cResult, [&](const auto &pNode) {
    if (pNode.vTag == "section-name") {
      const std::string cSection = std::string(pNode.vText);
      vCurrentSection = cSection;
      dParsed[vCurrentSection] = {};
    } else if (pNode.vTag == "key") {
      const std::string cKey = std::string(pNode.vText);
      vCurrentKey = cKey;
    } else if (pNode.vTag == "value" && !vCurrentKey.empty() && !vCurrentSection.empty()) {
      dParsed[vCurrentSection][vCurrentKey] = std::string(pNode.vText);
      vCurrentKey.clear();
    }
  });

  return dParsed;
}

static std::map<std::string, common_arg> fGetMapKeyOpt(common_params_context &pCtxParams) {
  std::map<std::string, common_arg> dMapping;
  for (const auto &vOpt : pCtxParams.lOptions) {
    for (const auto &vEnv : vOpt.mGetEnv()) {
      dMapping[vEnv] = vOpt;
    }
    for (const auto &vArg : vOpt.mGetArgs()) {
      dMapping[fRmLeadingDashes(vArg)] = vOpt;
    }
  }
  return dMapping;
}

static bool fIsBoolArg(const common_arg &pArg) { return !pArg.lArgsNeg.empty(); }

static std::string fParseBoolArg(const common_arg &pArg, const std::string &pKey, const std::string &pValue) {
  // if this is a negated arg, we need to reverse the value
  for (const auto &vNegArg : pArg.lArgsNeg) {
    if (fRmLeadingDashes(vNegArg) == pKey) {
      return common_arg_utils::fIsTruthy(pValue) ? "false" : "true";
    }
  }
  // otherwise, not negated
  return pValue;
}

common_preset_context::common_preset_context(mgs_example pEx)
    : vCtxParams(fCommonParamsParserInit(vDefaultParams, pEx)) {
  fCommonParamsAddPresetOptions(vCtxParams.lOptions);
  dKeyToOpt = fGetMapKeyOpt(vCtxParams);
}

common_presets common_preset_context::mLoadFromIni(const std::string &pPath, common_preset &pGlobal) const {
  common_presets vOut;
  auto vIniData = fParseIniFromFile(pPath);

  for (auto vSection : vIniData) {
    common_preset vPreset;
    std::string vSectionName = vSection.first.empty() ? std::string(cCommonPresetDefaultName) : vSection.first;
    if (vSectionName != "*" && vSectionName != cCommonPresetDefaultName) {
      auto vColonIdx = vSectionName.rfind(':');
      if (vColonIdx != std::string::npos) {
        std::string vTag = vSectionName.substr(vColonIdx + 1);
        std::string vCanonTag = fCanonicalTag(vTag);
        if (vCanonTag != vTag) {
          vSectionName = vSectionName.substr(0, vColonIdx + 1) + vCanonTag;
        }
      }
    }
    vPreset.vName = vSectionName;
    LOG_DBG("loading preset: %s\n", vPreset.vName.c_str());
    for (const auto &[vKey, vValue] : vSection.second) {
      if (vKey == "version") {
        // skip version key (reserved for future use)
        continue;
      }

      LOG_DBG("option: %s = %s\n", vKey.c_str(), vValue.c_str());
      if (vFilterAllowedKeys && sAllowedKeys.find(vKey) == sAllowedKeys.end()) {
        throw std::runtime_error(fStringFormat("option '%s' is not allowed in remote presets", vKey.c_str()));
      }
      if (dKeyToOpt.find(vKey) != dKeyToOpt.end()) {
        const auto &vOpt = dKeyToOpt.at(vKey);
        if (fIsBoolArg(vOpt)) {
          vPreset.dOptions[vOpt] = fParseBoolArg(vOpt, vKey, vValue);
        } else {
          vPreset.dOptions[vOpt] = vValue;
        }
        LOG_DBG("accepted option: %s = %s\n", vKey.c_str(), vPreset.dOptions[vOpt].c_str());
      } else {
        throw std::runtime_error(
            fStringFormat("option '%s' not recognized in preset '%s'", vKey.c_str(), vPreset.vName.c_str()));
      }
    }

    if (vPreset.vName == cCommonPresetDefaultName && vPreset.dOptions.empty()) {
      continue;
    }

    if (vPreset.vName == "*") {
      // handle global preset
      pGlobal = vPreset;
    } else {
      vOut[vPreset.vName] = vPreset;
    }
  }

  return vOut;
}

common_presets common_preset_context::mLoadFromCache() const {
  common_presets vOut;

  auto vCachedModels = fCommonListCachedModels();
  for (const auto &vModel : vCachedModels) {
    common_preset vPreset;
    vPreset.vName = vModel.mToString();
    vPreset.mSetOption(*this, "MGS_ARG_HF_REPO", vModel.mToString());
    vOut[vPreset.vName] = vPreset;
  }

  return vOut;
}

struct local_model {
  std::string vName;
  std::string vPath;
  std::string vPathMmproj;
};

common_presets common_preset_context::mLoadFromModelsDir(const std::string &pModelsDir) const {
  if (!std::filesystem::exists(pModelsDir) || !std::filesystem::is_directory(pModelsDir)) {
    throw std::runtime_error(fStringFormat("error: '%s' does not exist or is not a directory\n", pModelsDir.c_str()));
  }

  std::vector<local_model> lModels;
  auto vScanSubdir = [&lModels](const std::string &pSubdirPath, const std::string &pName) {
    auto vFiles = fFsList(pSubdirPath, false);
    common_file_info vModelFile;
    common_file_info vFirstShardFile;
    common_file_info vMmprojFile;
    for (const auto &vFile : vFiles) {
      if (fStringEndsWith(vFile.vName, ".gguf")) {
        if (vFile.vName.find("mmproj") != std::string::npos) {
          vMmprojFile = vFile;
        } else if (vFile.vName.find("-00001-of-") != std::string::npos) {
          vFirstShardFile = vFile;
        } else {
          vModelFile = vFile;
        }
      }
    }
    // single file model
    local_model vModel{
        /* name        */ pName,
        /* path        */ vFirstShardFile.vPath.empty() ? vModelFile.vPath : vFirstShardFile.vPath,
        /* path_mmproj */ vMmprojFile.vPath // can be empty
    };
    if (!vModel.vPath.empty()) {
      lModels.push_back(vModel);
    }
  };

  auto vFiles = fFsList(pModelsDir, true);
  for (const auto &vFile : vFiles) {
    if (vFile.vIsDir) {
      vScanSubdir(vFile.vPath, vFile.vName);
    } else if (fStringEndsWith(vFile.vName, ".gguf")) {
      // single file model
      std::string vName = vFile.vName;
      fStringReplaceAll(vName, ".gguf", "");
      local_model vModel{/* name        */ vName,
                         /* path        */ vFile.vPath,
                         /* path_mmproj */ ""};
      lModels.push_back(vModel);
    }
  }

  // convert local models to presets
  common_presets vOut;
  for (const auto &vModel : lModels) {
    common_preset vPreset;
    vPreset.vName = vModel.vName;
    vPreset.mSetOption(*this, "MGS_ARG_MODEL", vModel.vPath);
    if (!vModel.vPathMmproj.empty()) {
      vPreset.mSetOption(*this, "MGS_ARG_MMPROJ", vModel.vPathMmproj);
    }
    vOut[vPreset.vName] = vPreset;
  }

  return vOut;
}

common_preset common_preset_context::mLoadFromArgs(int pArgc, char **pArgv) const {
  common_preset vPreset;
  vPreset.vName = cCommonPresetDefaultName;

  bool vOk = fCommonParamsToMap(pArgc, pArgv, vCtxParams.vEx, vPreset.dOptions);
  if (!vOk) {
    throw std::runtime_error("failed to parse CLI arguments into preset");
  }

  return vPreset;
}

common_presets common_preset_context::mCascade(const common_presets &pBase, const common_presets &pAdded) const {
  common_presets vOut = pBase; // copy
  for (const auto &[vName, vPresetAdded] : pAdded) {
    if (vOut.find(vName) != vOut.end()) {
      // if exists, merge
      common_preset &vTarget = vOut[vName];
      vTarget.mMerge(vPresetAdded);
    } else {
      // otherwise, add directly
      vOut[vName] = vPresetAdded;
    }
  }
  return vOut;
}

common_presets common_preset_context::mCascade(const common_preset &pBase, const common_presets &pPresets) const {
  common_presets vOut;
  for (const auto &[vName, vPreset] : pPresets) {
    common_preset vTmp = pBase; // copy
    vTmp.vName = vName;
    vTmp.mMerge(vPreset);
    vOut[vName] = std::move(vTmp);
  }
  return vOut;
}
