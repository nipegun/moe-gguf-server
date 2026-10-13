#include "mgs-model.h"

#include "mgs-arch.h"
#include "mgs-ext.h"
#include "mgs-hparams.h"
#include "mgs-impl.h"
#include "mgs-mmap.h"
#include "mgs-cparams.h"
#include "mgs-model-loader.h"

#include "mgs-kv-cache.h"
#include "mgs-kv-cache-iswa.h"
#include "mgs-kv-cache-dsa.h"
#include "mgs-kv-cache-msa.h"
#include "mgs-kv-cache-dsv41.h"
#include "mgs-memory-hybrid.h"
#include "mgs-memory-hybrid-iswa.h"
#include "mgs-memory-hybrid-idx.h"
#include "mgs-memory-recurrent.h"

#include "mgs.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static mgs_model *fMgsModelMapping(llm_arch pArch, const mgs_model_params &pParams) {
  switch (pArch) {
  case cLlmArchGrok:
    return new mgs_model_grok(pParams);
  case cLlmArchNomicBertMoe:
    return new mgs_model_nomic_bert_moe(pParams);
  case cLlmArchMellum:
    return new mgs_model_mellum(pParams);
  case cLlmArchLladaMoe:
    return new mgs_model_llada_moe(pParams);
  case cLlmArchRnd1:
    return new mgs_model_rnd1(pParams);
  case cLlmArchQwen2moe:
    return new mgs_model_qwen2moe(pParams);
  case cLlmArchQwen3moe:
    return new mgs_model_qwen3moe(pParams);
  case cLlmArchQwen3vlmoe:
    return new mgs_model_qwen3vlmoe(pParams);
  case cLlmArchPhimoe:
    return new mgs_model_phimoe(pParams);
  case cLlmArchJamba:
    return new mgs_model_jamba(pParams);
  case cLlmArchCohere2moe:
    return new mgs_model_cohere2moe(pParams);
  case cLlmArchDbrx:
    return new mgs_model_dbrx(pParams);
  case cLlmArchOlmoe:
    return new mgs_model_olmoe(pParams);
  case cLlmArchArctic:
    return new mgs_model_arctic(pParams);
  case cLlmArchDeepseek41:
    return new mgs_model_deepseek41(pParams);
  case cLlmArchGlmDsa:
    return new mgs_model_glm_dsa(pParams);
  case cLlmArchGlm5Next:
    return new mgs_model_glm5_next(pParams);
  case cLlmArchGlm4Moe:
    return new mgs_model_glm4_moe(pParams);
  case cLlmArchNemotronHMoe:
    return new mgs_model_nemotron_h_moe(pParams);
  case cLlmArchExaoneMoe:
    return new mgs_model_exaone_moe(pParams);
  case cLlmArchGraniteMoe:
    return new mgs_model_granite_moe(pParams);
  case cLlmArchGraniteHybrid:
    return new mgs_model_granite_hybrid(pParams);
  case cLlmArchBailingmoe:
    return new mgs_model_bailingmoe(pParams);
  case cLlmArchBailingmoe2:
    return new mgs_model_bailingmoe2(pParams);
  case cLlmArchDots1:
    return new mgs_model_dots1(pParams);
  case cLlmArchAfmoe:
    return new mgs_model_afmoe(pParams);
  case cLlmArchLaguna:
    return new mgs_model_laguna(pParams);
  case cLlmArchERNIE4Point5Moe:
    return new mgs_model_ernie4_5_moe(pParams);
  case cLlmArchHunyuanMoe:
    return new mgs_model_hunyuan_moe(pParams);
  case cLlmArchHyV3:
    return new mgs_model_hy_v3(pParams);
  case cLlmArchOpenaiMoe:
    return new mgs_model_openai_moe(pParams);
  case cLlmArchLfm2moe:
    return new mgs_model_lfm2moe(pParams);
  case cLlmArchSmallthinker:
    return new mgs_model_smallthinker(pParams);
  case cLlmArchGrovemoe:
    return new mgs_model_grovemoe(pParams);
  case cLlmArchMinimaxM2:
    return new mgs_model_minimax_m2(pParams);
  case cLlmArchMinimaxM3:
    return new mgs_model_minimax_m3(pParams);
  case cLlmArchQwen3next:
    return new mgs_model_qwen3next(pParams);
  case cLlmArchQwen35:
    return new mgs_model_qwen35(pParams);
  case cLlmArchQwen35moe:
    return new mgs_model_qwen35moe(pParams);
  case cLlmArchQwen4exp:
    return new mgs_model_qwen4exp(pParams);
  case cLlmArchMistral3:
    return new mgs_model_mistral3(pParams);
  case cLlmArchDflash:
    return new mgs_model_dflash(pParams);
  case cLlmArchMimo2:
    return new mgs_model_mimo2(pParams);
  case cLlmArchKimiLinear:
    return new mgs_model_kimi_linear(pParams);
  case cLlmArchKimiK3:
    return new mgs_model_kimi_k3(pParams);
  case cLlmArchStep35:
    return new mgs_model_step35(pParams);
  case cLlmArchMinicpm:
    return new mgs_model_minicpm(pParams);
  default:
    throw std::runtime_error(std::string("unsupported model architecture: '") + fLlmArchName(pArch) + "'");
  }
}

mgs_model *mgs_model_create(llm_arch pArch, const mgs_model_params &pParams) {
  mgs_model *vModel = fMgsModelMapping(pArch, pParams);

  if (vModel != nullptr) {
    vModel->vArch = pArch;
    if (pParams.split_mode == MGS_SPLIT_MODE_TENSOR && !fLlmArchSupportsSmTensor(pArch)) {
      throw std::runtime_error(std::string("MGS_SPLIT_MODE_TENSOR not implemented for architecture '") +
                               fLlmArchName(pArch) + "'");
    }
  }

  return vModel;
}

mgs_model *mgs_model_create(mgs_model_loader &pMl, const mgs_model_params &pParams) {
  llm_arch vArch = pMl.mGetArch();
  if (vArch == cLlmArchUnknown) {
    throw std::runtime_error("unknown model architecture: '" + pMl.mGetArchName() + "'");
  }

  uint32_t vExpertCount = 0;
  uint32_t vLayerCount = 0;
  uint32_t vMtpLayerCount = 0;
  pMl.mGetKey(cLlmKvExpertCount, vExpertCount, false);
  pMl.mGetKey(cLlmKvBlockCount, vLayerCount, false);
  pMl.mGetKey(cLlmKvNextnPredictLayers, vMtpLayerCount, false);
  // A separate MTP head can be dense while its target is MoE. It must really
  // be a head-only file and be loaded explicitly as MTP, never as a main model.
  const bool cMtpHeadOnly = pParams.load_mtp && vMtpLayerCount > 0 &&
                            (vLayerCount == vMtpLayerCount || pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  if (vExpertCount == 0 && !cMtpHeadOnly) {
    throw std::runtime_error("mgs only supports MoE models; dense models cannot be loaded");
  }

  return mgs_model_create(vArch, pParams);
}

struct ggml_backend_meta_split_state mgs_meta_device_get_split_state(const struct ggml_tensor *pTensor,
                                                                     void *pUserdata) {
  const mgs_meta_device_get_split_state_userdata *vUd = (const mgs_meta_device_get_split_state_userdata *)pUserdata;
  const mgs_hparams &vHparams = vUd->vModel->vHparams;
  const std::string cTensorName = pTensor->name;

  static const std::regex cPatternQWeight("blk\\.\\d*\\.attn_q.weight");
  static const std::regex cPatternKvWeight("blk\\.\\d*\\.attn_(k|v).weight");
  static const std::regex cPatternQkvWeight("blk\\.\\d*\\.attn_qkv.weight");
  static const std::regex cPatternQBias("blk\\.\\d*\\.attn_q\\.bias");
  static const std::regex cPatternKvBias("blk\\.\\d*\\.attn_(k|v)\\.bias");
  static const std::regex cPatternQkvBias("blk\\.\\d*\\.attn_qkv.bias");
  static const std::regex cPatternQkNorm("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
  static const std::regex cPatternKvCache("cache_(k|v)_l\\d*");
  static const std::regex cPatternIdxCache("cache_idx_(k|v)_l\\d*");
  static const std::regex cPatternAttnSinks("blk\\.\\d*\\.attn_sinks.weight");
  static const std::regex cPatternAttnOutWeight("blk\\.\\d*\\.attn_output.weight");
  static const std::regex cPatternAttnOutBias("blk\\.\\d*\\.attn_output.bias");
  static const std::regex cPatternAttnGateWeight("blk\\.\\d*\\.attn_gate.weight");

  static const std::regex cPatternSsmDt("blk\\.\\d*\\.ssm_dt.bias");
  static const std::regex cPatternSsmA("blk\\.\\d*\\.ssm_a");
  static const std::regex cPatternSsmAlpha("blk\\.\\d*\\.ssm_alpha.weight");
  static const std::regex cPatternSsmBeta("blk\\.\\d*\\.ssm_beta.weight");
  static const std::regex cPatternSsmBetaAlpha("blk\\.\\d*\\.ssm_ba.weight");
  static const std::regex cPatternRCache("cache_r_l\\d*");
  static const std::regex cPatternPleRCache("cache_ple_r_l\\d*");
  static const std::regex cPatternSCache("cache_s_l\\d*");
  static const std::regex cPatternSsmConv1d("blk\\.\\d*\\.ssm_conv1d.weight");
  static const std::regex cPatternSsmOutWeight("blk\\.\\d*\\.ssm_out.weight");

  static const std::regex cPatternFfnUpWeight("blk\\.\\d*\\.ffn_up(_exps)?.weight");
  static const std::regex cPatternFfnUpBias("blk\\.\\d*\\.ffn_up(_exps)?.bias");
  static const std::regex cPatternFfnGateWeight("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
  static const std::regex cPatternFfnGateBias("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
  static const std::regex cPatternFfnGateUpWeight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
  static const std::regex cPatternFfnDownWeight("blk\\.\\d*\\.ffn_down(_exps)?.weight");
  static const std::regex cPatternFfnDownBias("blk\\.\\d*\\.ffn_down.bias");
  static const std::regex cPatternFfnDownExpsBias("blk\\.\\d*\\.ffn_down_exps.bias");

  static const std::regex cPatternOutputWeight("output\\.weight");
  static const std::regex cPatternOutputBias("output\\.bias");

  struct tensor_config {
    ggml_backend_meta_split_axis vAxis;

    const ggml_tensor *vTensorAxis0;

    uint32_t vIl;
    size_t vRotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
  };

  auto vGetTensorConfigImpl = [&](const ggml_backend_meta_split_axis pAxis, const std::string &pSuffix = "",
                                  const std::string &pSuffixFallback = "") -> tensor_config {
    // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be
    // aliasing effects,
    //     count only the same type of previous layers to avoid this
    auto vGetIlEff = [&](const size_t pIl) {
      size_t vRet = 0;
      const bool cIlIsRecr = vHparams.mIsRecr(pIl);
      const bool cIlIsSwa = vHparams.mIsSwa(pIl);
      for (size_t vIlPrev = 0; vIlPrev < pIl; vIlPrev++) {
        vRet += vHparams.mIsRecr(vIlPrev) == cIlIsRecr && vHparams.mIsSwa(vIlPrev) == cIlIsSwa;
      }
      return vRet;
    };

    uint32_t vIl;
    std::string vPrefix;
    size_t vRotation;
    if (cTensorName.substr(0, 4) == "blk.") {
      const size_t cLengthPrefix = cTensorName.find('.', 4);
      GGML_ASSERT(cLengthPrefix != std::string::npos);
      vPrefix = cTensorName.substr(0, cLengthPrefix + 1);
      vIl = std::stoull(cTensorName.substr(4, cLengthPrefix));
      vRotation = vGetIlEff(vIl) % vUd->vNDevices;
    } else if (cTensorName.substr(0, 6) == "cache_") {
      const size_t cLayerIndexStart = cTensorName.find("_l", 6);
      GGML_ASSERT(cLayerIndexStart != std::string::npos);
      vIl = std::stoull(cTensorName.substr(cLayerIndexStart + 2));
      vPrefix = "blk." + std::to_string(vIl) + ".";
      vRotation = vGetIlEff(vIl) % vUd->vNDevices;
    } else {
      vIl = 0;
      vRotation = vHparams.mNLayer() % vUd->vNDevices;
    }
    const ggml_tensor *vTensorAxis0 = pSuffix.empty() ? pTensor : vUd->vModel->mGetTensor((vPrefix + pSuffix).c_str());
    if (vTensorAxis0 == nullptr) {
      GGML_ASSERT(!pSuffixFallback.empty());
      vTensorAxis0 = vUd->vModel->mGetTensor((vPrefix + pSuffixFallback).c_str());
    }
    GGML_ASSERT(vTensorAxis0 != nullptr);
    return {pAxis, vTensorAxis0, vIl, vRotation};
  };

  auto vGetTensorConfig = [&]() -> tensor_config {
    // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
    if (std::regex_match(cTensorName, cPatternIdxCache)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole
    // history
    if (std::regex_match(cTensorName, cPatternPleRCache)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    // standard attention
    if (std::regex_match(cTensorName, cPatternQWeight) || std::regex_match(cTensorName, cPatternKvWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQBias) || std::regex_match(cTensorName, cPatternKvBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkvWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkvBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkNorm)) {
      return vGetTensorConfigImpl(pTensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1,
                                  "attn_output.weight");
    }
    if (std::regex_match(cTensorName, cPatternKvCache) || std::regex_match(cTensorName, cPatternAttnSinks)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
    }
    if (std::regex_match(cTensorName, cPatternAttnOutWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }
    if (std::regex_match(cTensorName, cPatternAttnOutBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    if (std::regex_match(cTensorName, cPatternAttnGateWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta) ||
        std::regex_match(cTensorName, cPatternSsmBetaAlpha)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternRCache) || std::regex_match(cTensorName, cPatternSCache)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmConv1d)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmOutWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }

    // FFN
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnGateWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnUpBias) || std::regex_match(cTensorName, cPatternFfnGateBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnDownWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnDownBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }
    if (std::regex_match(cTensorName, cPatternFfnDownExpsBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
    }

    // output
    if (std::regex_match(cTensorName, cPatternOutputWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1);
    }
    if (std::regex_match(cTensorName, cPatternOutputBias)) {
      const ggml_tensor *vOutputWeight = vUd->vModel->mGetTensor("output.weight");
      GGML_ASSERT(vOutputWeight != nullptr);
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }

    // everything else
    return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
  };

  auto vGetSplitSegments = [&](int pAxis, uint32_t pIl) -> std::vector<std::pair<int64_t, uint32_t>> {
    if (vUd->vModel->vArch == cLlmArchQwen3next || vUd->vModel->vArch == cLlmArchQwen35 ||
        vUd->vModel->vArch == cLlmArchQwen35moe || vUd->vModel->vArch == cLlmArchQwen4exp) {
      const int64_t cHeadKDim = vHparams.vSsmDState;
      const int64_t cHeadVDim = vHparams.vSsmDState;
      const int64_t cNKHeads = vHparams.vSsmNGroup;
      const int64_t cNVHeads = vHparams.vSsmDtRank;
      const int64_t cKeyDim = cHeadKDim * cNKHeads;
      const int64_t cValueDim = cHeadVDim * cNVHeads;

      // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
      //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
      //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct
      //   pattern)
      if (vUd->vModel->vArch == cLlmArchQwen3next) {
        if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternSsmConv1d)) {
          GGML_ASSERT(pTensor->ne[pAxis] == 2 * cKeyDim + cValueDim);
          return {{cKeyDim, 2}, {cValueDim, 1}};
        }
      } else {
        const int64_t cHeadRatio = cNVHeads / cNKHeads;
        if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternSsmConv1d)) {
          GGML_ASSERT(pTensor->ne[pAxis] == 2 * cKeyDim + cValueDim);
          return {{cKeyDim, 2 + cHeadRatio}};
        }
        if (std::regex_match(cTensorName, cPatternAttnGateWeight) ||
            std::regex_match(cTensorName, cPatternSsmOutWeight)) {
          return {{cKeyDim, cHeadRatio}};
        }
        if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA) ||
            std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta)) {
          return {{cNKHeads, cHeadRatio}};
        }
        if (std::regex_match(cTensorName, cPatternRCache)) {
          return {{cKeyDim * (vHparams.vSsmDConv - 1), 2 + cHeadRatio}};
        }
        if (std::regex_match(cTensorName, cPatternSCache)) {
          return {{cNKHeads * cHeadVDim * cHeadVDim, cHeadRatio}};
        }
      }

      // the FFN is the same for Qwen 3 Next and Qwen 3.5:
      if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
        const int64_t cNFfExp = vHparams.vNFfExp;
        GGML_ASSERT(pTensor->ne[pAxis] == 2 * cNFfExp);
        return {{cNFfExp, 2}};
      }
      return {{pTensor->ne[pAxis], 1}};
    }

    if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternQkvBias)) {
      const int64_t cNEmbd = vHparams.vNEmbd;
      const int64_t cNEmbdGqa = vHparams.mNEmbdVGqa(pIl);
      GGML_ASSERT(vHparams.mNEmbdKGqa() == cNEmbdGqa);
      GGML_ASSERT(pTensor->ne[pAxis] == cNEmbd + 2 * cNEmbdGqa);
      return {{cNEmbd, 1}, {cNEmbdGqa, 2}};
    }
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnUpBias)) {
      const int64_t cNFf = vHparams.mNFf(pIl);
      // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
      if (pTensor->ne[pAxis] == 2 * cNFf) {
        return {{cNFf, 2}};
      }
      return {{pTensor->ne[pAxis], 1}};
    }
    if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
      const int64_t cNFfExp = vHparams.vNFfExp;
      GGML_ASSERT(pTensor->ne[pAxis] == 2 * cNFfExp);
      return {{cNFfExp, 2}};
    }
    return {{pTensor->ne[pAxis], 1}};
  };

  auto vGetSplitGranularity = [&](int64_t pBlckSize, uint32_t pIl,
                                  const std::vector<std::pair<int64_t, uint32_t>> &pSegments) -> std::vector<int64_t> {
    // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient
    // kernels can be used
    if (vHparams.mIsRecr(pIl)) {
      // linear attention
      const int64_t cHeadDim = vHparams.vSsmDState;
      const int64_t cBlckSizePerf = std::lcm(pBlckSize, 128);
      const int64_t cGranularityQkv = std::lcm(cBlckSizePerf, cHeadDim);
      if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternAttnGateWeight) ||
          std::regex_match(cTensorName, cPatternSsmConv1d) || std::regex_match(cTensorName, cPatternSsmOutWeight)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv);
      }
      if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA) ||
          std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv / cHeadDim);
      }
      if (std::regex_match(cTensorName, cPatternSsmBetaAlpha)) {
        return std::vector<int64_t>(pSegments.size(), 2 * (cGranularityQkv / cHeadDim));
      }
      if (std::regex_match(cTensorName, cPatternRCache)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv * (vHparams.vSsmDConv - 1));
      }
      if (std::regex_match(cTensorName, cPatternSCache)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv * cHeadDim);
      }
    } else {
      // regular attention
      const uint32_t cNGqa = vHparams.mNGqa(pIl);
      const uint32_t cNEmbdQ = cNGqa * vHparams.mNEmbdHeadK(pIl);

      // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
      int64_t vBlckSizePerf = pBlckSize;
      while (vBlckSizePerf < 128 && vBlckSizePerf * vUd->vNDevices < cNEmbdQ) {
        vBlckSizePerf *= 2;
      }

      if (std::regex_match(cTensorName, cPatternAttnSinks)) {
        GGML_ASSERT(pSegments.size() == 1);
        return {std::lcm(cNEmbdQ, vBlckSizePerf) / cNEmbdQ * cNGqa};
      }

      const int64_t cGranularityQ = std::lcm(cNEmbdQ, vBlckSizePerf);
      if (std::regex_match(cTensorName, cPatternQWeight) || std::regex_match(cTensorName, cPatternQBias)) {
        GGML_ASSERT(pSegments.size() == 1);
        // some models have Q gate tensors, for those cases the granularity needs to be doubled:
        if (vUd->vModel->vArch == cLlmArchQwen3next || vUd->vModel->vArch == cLlmArchQwen35 ||
            vUd->vModel->vArch == cLlmArchQwen35moe || vUd->vModel->vArch == cLlmArchQwen4exp) {
          return {std::lcm(2 * cNEmbdQ, vBlckSizePerf)};
        }
        return {cGranularityQ};
      }
      if (std::regex_match(cTensorName, cPatternAttnOutWeight)) {
        GGML_ASSERT(pSegments.size() == 1);
        return {cGranularityQ};
      }

      const int64_t cGranularityKv = cGranularityQ / cNGqa;
      if (std::regex_match(cTensorName, cPatternKvWeight) || std::regex_match(cTensorName, cPatternKvBias) ||
          std::regex_match(cTensorName, cPatternKvCache)) {
        GGML_ASSERT(pSegments.size() == 1);
        return {cGranularityKv};
      }
      if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternQkvBias)) {
        GGML_ASSERT(pSegments.size() == 2);
        return {cGranularityQ, cGranularityKv};
      }
    }

    // FFN
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnUpBias) ||
        std::regex_match(cTensorName, cPatternFfnGateWeight) || std::regex_match(cTensorName, cPatternFfnGateBias) ||
        std::regex_match(cTensorName, cPatternFfnGateUpWeight) ||
        std::regex_match(cTensorName, cPatternFfnDownWeight)) {
      const int64_t cBlckSizePerf = std::lcm(pBlckSize, 128);
      GGML_ASSERT(pSegments.size() == 1);
      return {cBlckSizePerf};
    }

    // everything else
    GGML_ASSERT(pSegments.size() == 1);
    return {1};
  };

  ggml_backend_meta_split_state vSplitState;
  memset(&vSplitState, 0, sizeof(vSplitState));
  tensor_config vTc = vGetTensorConfig();
  vSplitState.axis = vTc.vAxis;
  if (vSplitState.axis >= 0 && vSplitState.axis < GGML_MAX_DIMS) {
    const int64_t cBlckSize = ggml_blck_size(vTc.vTensorAxis0->type);
    const float *vTensorSplit = vUd->vModel->mTensorSplit();
    std::vector<float> lTensorSplitScan;
    lTensorSplitScan.reserve(vUd->vNDevices);
    for (size_t vJ = 0; vJ < vUd->vNDevices; vJ++) {
      lTensorSplitScan.push_back(vTensorSplit == nullptr ? 0.0f : vTensorSplit[(vJ + vTc.vRotation) % vUd->vNDevices]);
      if (vJ > 0) {
        lTensorSplitScan[vJ] += lTensorSplitScan[vJ - 1];
      }
    }
    const std::vector<std::pair<int64_t, uint32_t>> lSegments = vGetSplitSegments(vSplitState.axis, vTc.vIl);
    const std::vector<int64_t> lGranularity = vGetSplitGranularity(cBlckSize, vTc.vIl, lSegments);
    for (size_t vIs = 0; vIs < lSegments.size(); vIs++) {
      const int64_t cNeS = lSegments[vIs].first;
      const uint32_t cNrS = lSegments[vIs].second;
      const int64_t cGS = lGranularity[vIs];
      int64_t vLow = 0;
      size_t vJ = 0;
      for (; vJ < vUd->vNDevices - 1; vJ++) {
        int64_t vHigh = lTensorSplitScan.back() == 0.0f ? cNeS * (vJ + 1) / vUd->vNDevices
                                                        : cNeS * lTensorSplitScan[vJ] / lTensorSplitScan.back();
        if (vHigh % cGS != 0) {
          vHigh -= vHigh % cGS;
        }
        vSplitState.ne[vIs * vUd->vNDevices + (vJ + vTc.vRotation) % vUd->vNDevices] = vHigh - vLow;
        vLow = vHigh;
      }
      vSplitState.ne[vIs * vUd->vNDevices + (vJ + vTc.vRotation) % vUd->vNDevices] = cNeS - vLow;
      vSplitState.nr[vIs] = cNrS;
    }
    vSplitState.n_segments = lSegments.size();
  } else {
    memset(vSplitState.ne, 0, sizeof(vSplitState.ne));
    vSplitState.nr[0] = 1;
    vSplitState.n_segments = 1;
  }
  return vSplitState;
  GGML_UNUSED(pUserdata);
}

const char *fLlmTypeName(llm_type pType) {
  switch (pType) {
  case cLlmType14m:
    return "14M";
  case cLlmType17m:
    return "17M";
  case cLlmType22m:
    return "22M";
  case cLlmType33m:
    return "33M";
  case cLlmType47m:
    return "47M";
  case cLlmType60m:
    return "60M";
  case cLlmType70m:
    return "70M";
  case cLlmType80m:
    return "80M";
  case cLlmType109m:
    return "109M";
  case cLlmType137m:
    return "137M";
  case cLlmType140m:
    return "140M";
  case cLlmType149m:
    return "149M";
  case cLlmType160m:
    return "160M";
  case cLlmType190m:
    return "190M";
  case cLlmType220m:
    return "220M";
  case cLlmType230m:
    return "230M";
  case cLlmType250m:
    return "250M";
  case cLlmType256m:
    return "256M";
  case cLlmType270m:
    return "270M";
  case cLlmType335m:
    return "335M";
  case cLlmType350m:
    return "350M";
  case cLlmType360m:
    return "360M";
  case cLlmType395m:
    return "395M";
  case cLlmType410m:
    return "410M";
  case cLlmType450m:
    return "450M";
  case cLlmType475m:
    return "475M";
  case cLlmType558m:
    return "558M";
  case cLlmType700m:
    return "700M";
  case cLlmType770m:
    return "770M";
  case cLlmType780m:
    return "780M";
  case cLlmType950m:
    return "950M";
  case cLlmType0Point3B:
    return "0.3B";
  case cLlmType0Point5B:
    return "0.5B";
  case cLlmType0Point6B:
    return "0.6B";
  case cLlmType0Point8B:
    return "0.8B";
  case cLlmType1b:
    return "1B";
  case cLlmType1Point2B:
    return "1.2B";
  case cLlmType1Point3B:
    return "1.3B";
  case cLlmType1Point4B:
    return "1.4B";
  case cLlmType1Point5B:
    return "1.5B";
  case cLlmType1Point6B:
    return "1.6B";
  case cLlmType1Point7B:
    return "1.7B";
  case cLlmType1Point8B:
    return "1.8B";
  case cLlmType2b:
    return "2B";
  case cLlmType2Point6B:
    return "2.6B";
  case cLlmType2Point8B:
    return "2.8B";
  case cLlmType2Point9B:
    return "2.9B";
  case cLlmType3b:
    return "3B";
  case cLlmType4b:
    return "4B";
  case cLlmType6b:
    return "6B";
  case cLlmType6Point9B:
    return "6.9B";
  case cLlmType7b:
    return "7B";
  case cLlmType8b:
    return "8B";
  case cLlmType9b:
    return "9B";
  case cLlmType11b:
    return "11B";
  case cLlmType12b:
    return "12B";
  case cLlmType13b:
    return "13B";
  case cLlmType14b:
    return "14B";
  case cLlmType15b:
    return "15B";
  case cLlmType16b:
    return "16B";
  case cLlmType20b:
    return "20B";
  case cLlmType26b:
    return "26B";
  case cLlmType27b:
    return "27B";
  case cLlmType30b:
    return "30B";
  case cLlmType32b:
    return "32B";
  case cLlmType34b:
    return "34B";
  case cLlmType35b:
    return "35B";
  case cLlmType36b:
    return "36B";
  case cLlmType40b:
    return "40B";
  case cLlmType65b:
    return "65B";
  case cLlmType70b:
    return "70B";
  case cLlmType120b:
    return "120B";
  case cLlmType142b:
    return "142B";
  case cLlmType290b:
    return "290B";
  case cLlmType314b:
    return "314B";
  case cLlmType405b:
    return "405B";
  case cLlmTypeSmall:
    return "0.1B";
  case cLlmTypeMedium:
    return "0.4B";
  case cLlmTypeLarge:
    return "0.8B";
  case cLlmTypeXl:
    return "1.5B";
  case cLlmTypeA1Point7B:
    return "A1.7B";
  case cLlmTypeA2Point7B:
    return "A2.7B";
  case cLlmType8x7B:
    return "8x7B";
  case cLlmType8x22B:
    return "8x22B";
  case cLlmType16x12B:
    return "16x12B";
  case cLlmType16x3Point8B:
    return "16x3.8B";
  case cLlmType10b128x3Point66B:
    return "10B+128x3.66B";
  case cLlmType57bA14b:
    return "57B.A14B";
  case cLlmTypeA13b:
    return "A13B";
  case cLlmType7bA1b:
    return "7B.A1B";
  case cLlmType8bA1b:
    return "8B.A1B";
  case cLlmType12bA2Point5B:
    return "12B.A2.5B";
  case cLlmType16bA1b:
    return "16B.A1B";
  case cLlmType21bA3b:
    return "21B.A3B";
  case cLlmType24bA2b:
    return "24B.A2B";
  case cLlmType30bA3b:
    return "30B.A3B";
  case cLlmType31bA3Point5B:
    return "31B.A3.5B";
  case cLlmType35bA3b:
    return "35B.A3B";
  case cLlmType48bA3b:
    return "48B.A3B";
  case cLlmType80bA3b:
    return "80B.A3B";
  case cLlmTypeA3b:
    return "A3B";
  case cLlmType100bA6b:
    return "100B.A6B";
  case cLlmType102bA12b:
    return "102B.A12B";
  case cLlmType106bA12b:
    return "106B.A12B";
  case cLlmType118bA8b:
    return "118B.A8B";
  case cLlmType120bA12b:
    return "120B.A12B";
  case cLlmType122bA10b:
    return "122B.A10B";
  case cLlmType196bA11b:
    return "196B.A11B";
  case cLlmType230bA10b:
    return "230B.A10B";
  case cLlmType428bA23b:
    return "428B.A23B";
  case cLlmType235bA22b:
    return "235B.A22B";
  case cLlmType300bA47b:
    return "300B.A47B";
  case cLlmType310bA15b:
    return "310B.A15B";
  case cLlmType320bA18b:
    return "320B.A18B";
  case cLlmType355bA32b:
    return "355B.A32B";
  case cLlmType397bA17b:
    return "397B.A17B";
  case cLlmType744bA40b:
    return "744B.A40B";
  case cLlmType2Point8tA50b:
    return "2.8T.A50B";
  default:
    return "?B";
  }
}

static const char *fMgsExpertGatingFuncName(mgs_expert_gating_func_type pType) {
  switch (pType) {
  case cMgsExpertGatingFuncTypeSoftmax:
    return "softmax";
  case cMgsExpertGatingFuncTypeSigmoid:
    return "sigmoid";
  case cMgsExpertGatingFuncTypeSqrtSoftplus:
    return "sqrtsoftplus";
  default:
    return "unknown";
  }
}

static const std::map<mgs_rope_scaling_type, const char *> dMgsRopeScalingTypes = {
    {MGS_ROPE_SCALING_TYPE_NONE, "none"},
    {MGS_ROPE_SCALING_TYPE_LINEAR, "linear"},
    {MGS_ROPE_SCALING_TYPE_YARN, "yarn"},
    {MGS_ROPE_SCALING_TYPE_LONGROPE, "longrope"},
};

std::string mgs_rope_scaling_type_name(mgs_rope_scaling_type pRopeScalingType) {
  return dMgsRopeScalingTypes.at(pRopeScalingType);
}

static mgs_rope_scaling_type fMgsRopeScalingTypeFromString(const std::string &pName) {
  for (const auto &vKv : dMgsRopeScalingTypes) {
    if (vKv.second == pName) {
      return (mgs_rope_scaling_type)vKv.first;
    }
  }

  return MGS_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// Maps the GGUF `<arch>.hidden_activation` string to the FFN op type used by the
// graph builders. Only gated activations that map cleanly to llm_ffn_op_type are
// listed; unrecognized values fall back to GeGLU, which matches the historical
// default for ModernBert-style architectures.
static const std::map<std::string, llm_ffn_op_type> dLlmFfnOpTypesFromString = {
    {"gelu", cLlmFfnGeglu},    {"geglu", cLlmFfnGeglu}, {"silu", cLlmFfnSwiglu}, {"swish", cLlmFfnSwiglu},
    {"swiglu", cLlmFfnSwiglu}, {"relu", cLlmFfnRelu},   {"reglu", cLlmFfnReglu},
};

llm_ffn_op_type fLlmFfnOpTypeFromString(const std::string &pName, llm_ffn_op_type pFallback) {
  const auto cIt = dLlmFfnOpTypesFromString.find(pName);
  if (cIt != dLlmFfnOpTypesFromString.end()) {
    return cIt->second;
  }
  return pFallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t fMakeCpuBuftList(const std::vector<mgs_device> &pDevices, bool pUseExtraBufts, bool pNoHost) {
  buft_list_t vBuftList;

  // add ACCEL buffer types
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
      auto *vBuft = ggml_backend_dev_buffer_type(vDev);
      // skip
      if (vBuft != ggml_backend_cpu_buffer_type()) {
        vBuftList.emplace_back(vDev, vBuft);
      }
    }
  }

  // add a host buffer type
  // storing the tensors in a host buffer is useful when the processing of large batches
  // is offloaded to a GPU device, since it reduces the time spent on data transfers
  // generally, this will be done using the first device in the list
  // a better approach would be to handle this on a weight-by-weight basis using the offload_op
  // function of the device to determine if it would benefit from being stored in a host buffer
  if (!pNoHost) {
    for (const auto &vDev : pDevices) {
      ggml_backend_buffer_type_t vBuft = ggml_backend_dev_host_buffer_type(vDev.vDev);
      if (vBuft) {
        vBuftList.emplace_back(vDev.vDev, vBuft);
        break;
      }
    }
  }

  // add extra buffer types
  if (pUseExtraBufts) {
    auto *vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (vCpuDev == nullptr) {
      throw std::runtime_error(fFormat("%s: no CPU backend found", __func__));
    }

    auto *vCpuReg = ggml_backend_dev_backend_reg(vCpuDev);
    auto vGgmlBackendDevGetExtraBuftsFn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        vCpuReg, "ggml_backend_dev_get_extra_bufts");
    if (vGgmlBackendDevGetExtraBuftsFn) {
      ggml_backend_buffer_type_t *vExtraBufts = vGgmlBackendDevGetExtraBuftsFn(vCpuDev);
      while (vExtraBufts && *vExtraBufts) {
        vBuftList.emplace_back(vCpuDev, *vExtraBufts);
        ++vExtraBufts;
      }
    }
  }

  // add the CPU buffer type
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
      vBuftList.emplace_back(vDev, ggml_backend_dev_buffer_type(vDev));
    }
  }

  return vBuftList;
}

// GPU: split if MGS_SPLIT_MODE_ROW -> GPU
static buft_list_t fMakeGpuBuftList(ggml_backend_dev_t pDev, mgs_split_mode pSplitMode, const float *pTensorSplit) {
  buft_list_t vBuftList;

  // add the device split buffer type if requested and available
  if (pSplitMode == MGS_SPLIT_MODE_ROW) {
    ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(pDev);
    auto vGgmlBackendSplitBufferTypeFn =
        (ggml_backend_split_buffer_type_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_split_buffer_type");
    if (vGgmlBackendSplitBufferTypeFn) {
      size_t vDevIndex = [&]() {
        auto *vReg = ggml_backend_dev_backend_reg(pDev);
        for (size_t vI = 0; vI < ggml_backend_reg_dev_count(vReg); ++vI) {
          if (ggml_backend_reg_dev_get(vReg, vI) == pDev) {
            return vI;
          }
        }
        throw std::runtime_error(fFormat("device %s not found in its backend reg", ggml_backend_dev_name(pDev)));
      }();
      auto *vBuft = vGgmlBackendSplitBufferTypeFn(vDevIndex, pTensorSplit);
      if (vBuft != nullptr) {
        vBuftList.emplace_back(pDev, vBuft);
      }
    } else {
      throw std::runtime_error(fFormat("device %s does not support split buffers", ggml_backend_dev_name(pDev)));
    }
  }

  // add the device default buffer type
  vBuftList.emplace_back(pDev, ggml_backend_dev_buffer_type(pDev));

  // add the device extra buffer type (if any)
  ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(pDev);
  if (vReg) {
    auto vGgmlBackendDevGetExtraBuftsFn =
        (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_dev_get_extra_bufts");

    if (vGgmlBackendDevGetExtraBuftsFn) {
      ggml_backend_buffer_type_t *vExtraBufts = vGgmlBackendDevGetExtraBuftsFn(pDev);
      while (vExtraBufts && *vExtraBufts) {
        vBuftList.emplace_back(pDev, *vExtraBufts);
        ++vExtraBufts;
      }
    }
  }

  return vBuftList;
}

struct mgs_model::impl {
  impl() = default;
  ~impl() = default;

  uint64_t vNElements = 0;

  size_t vNBytes = 0;

  std::string vDescStr;

  mgs_ftype vFtype = MGS_FTYPE_ALL_F32;

  // model memory mapped files
  mgs_mmaps vMappings;

  // objects representing data potentially being locked in memory
  mgs_mlocks vMlockBufs;
  mgs_mlocks vMlockMmaps;

  // contexts where the model tensors metadata is stored as well as the corresponding buffers:
  std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> lCtxsBufs;

  buft_list_t vCpuBuftList;
  std::map<ggml_backend_dev_t, buft_list_t> dGpuBuftList;
  std::map<ggml_backend_dev_t, buft_list_t> dMtpBufferLists;

  struct layer_dev {
    ggml_backend_dev_t vDev;
    buft_list_t *vBuftList;
  };

  layer_dev vDevInput = {};
  layer_dev vDevOutput = {};
  std::vector<layer_dev> lDevLayer;

  bool vHasTensorOverrides;

  std::vector<float> lTensorSplitOwned;
};

mgs_model::mgs_model(const mgs_model_params &pParams) : vParams(pParams), vPimpl(std::make_unique<impl>()) {
  if (pParams.tensor_split != nullptr) {
    // mgs_model_params stores tensor_split as a borrowed pointer, but the model
    // may need it later for tensor-parallel KV-cache split metadata.
    vPimpl->lTensorSplitOwned.assign(pParams.tensor_split, pParams.tensor_split + mgs_max_devices());
    this->vParams.tensor_split = vPimpl->lTensorSplitOwned.data();
  }
  vPimpl->vHasTensorOverrides = pParams.tensor_buft_overrides && pParams.tensor_buft_overrides[0].pattern;
}

mgs_model::~mgs_model() {
  for (auto *vLora : sLoras) {
    delete vLora;
  }
}

void mgs_model_base::mLoadStats(mgs_model_loader &pMl) {
  vPimpl->vNElements = pMl.vNElements;
  vPimpl->vNBytes = pMl.vNBytes;
}

void mgs_model_base::mLoadHparams(mgs_model_loader &pMl) {
  const gguf_context *vCtx = pMl.vMetadata;

  // get metadata as string
  for (int vI = 0; vI < gguf_get_n_kv(vCtx); vI++) {
    gguf_type vType = gguf_get_kv_type(vCtx, vI);
    if (vType == GGUF_TYPE_ARRAY) {
      continue;
    }
    const char *vName = gguf_get_key(vCtx, vI);
    const std::string cValue = fGgufKvToStr(vCtx, vI);
    dGgufKv.emplace(vName, cValue);
  }

  // get general kv
  pMl.mGetKey(cLlmKvGeneralName, vName, false);

  // everything past this point is not vocab-related
  // for CLIP models, we only need to load tensors, no hparams
  if (vHparams.vVocabOnly || pMl.mGetArch() == cLlmArchClip) {
    return;
  }

  pMl.mGetKey(cLlmKvContextLength, vHparams.vNCtxTrain);
  pMl.mGetKey(cLlmKvEmbeddingLength, vHparams.vNEmbd);
  pMl.mGetKey(cLlmKvEmbeddingLengthOut, vHparams.vNEmbdOutImpl, false);
  pMl.mGetKey(cLlmKvAttentionCausal, vHparams.vCausalAttn, false);
  pMl.mGetKey(cLlmKvPoolingType, vHparams.vPoolingType, false);
  pMl.mGetKey(cLlmKvBlockCount, vHparams.vNLayerAll);
  GGML_ASSERT(vHparams.vNLayerAll > 0 && vHparams.vNLayerAll <= MGS_MAX_LAYERS);
  pMl.mGetKey(cLlmKvExpertCount, vHparams.vNExpert, false);
  pMl.mGetKey(cLlmKvExpertUsedCount, vHparams.vNExpertUsed, false);
  pMl.mGetKey(cLlmKvExpertGroupCount, vHparams.vNExpertGroups, false);
  pMl.mGetKey(cLlmKvExpertGroupUsedCount, vHparams.vNGroupUsed, false);

  if (vArch == cLlmArchHunyuanVl || vArch == cLlmArchHunyuanDense) {
    if (vHparams.vNExpert <= 1) {
      vHparams.vNExpert = 0;
      vHparams.vNExpertUsed = 0;
    }
  }

  if (vArch == cLlmArchWavtokenizerDec) {
    pMl.mGetKey(cLlmKvFeaturesLength, vHparams.vNEmbd);
    pMl.mGetKey(cLlmKvEmbeddingLength, vHparams.vNEmbdOutImpl);

    pMl.mGetKey(cLlmKvPosnetEmbeddingLength, vHparams.vPosnet.vNEmbd);
    pMl.mGetKey(cLlmKvPosnetBlockCount, vHparams.vPosnet.vNLayer);

    pMl.mGetKey(cLlmKvConvnextEmbeddingLength, vHparams.vConvnext.vNEmbd);
    pMl.mGetKey(cLlmKvConvnextBlockCount, vHparams.vConvnext.vNLayer);
  }

  GGML_ASSERT(vHparams.vNExpert <= MGS_MAX_EXPERTS);
  GGML_ASSERT(vHparams.vNExpertUsed <= vHparams.vNExpert);
  if (vHparams.vNExpert > 0) {
    GGML_ASSERT(vHparams.vNExpertUsed > 0);
    GGML_ASSERT(vHparams.vNExpertGroups < vHparams.vNExpert);
    if (vHparams.vNExpertGroups > 1) {
      GGML_ASSERT(vHparams.vNExpert % vHparams.vNExpertGroups == 0);
      GGML_ASSERT(vHparams.vNGroupUsed > 0);
      GGML_ASSERT(vHparams.vNGroupUsed < vHparams.vNExpertGroups);
    }
  } else {
    GGML_ASSERT(vHparams.vNExpertUsed == 0);
    GGML_ASSERT(vHparams.vNExpertGroups == 0);
  }

  std::fill(vHparams.aNHeadArr.begin(), vHparams.aNHeadArr.end(), 0);
  std::fill(vHparams.aNHeadKvArr.begin(), vHparams.aNHeadKvArr.end(), 0);
  std::fill(vHparams.aNFfArr.begin(), vHparams.aNFfArr.end(), 0);

  std::fill(vHparams.aRopeSections.begin(), vHparams.aRopeSections.end(), 0);
  std::fill(vHparams.aIsSwaImpl.begin(), vHparams.aIsSwaImpl.end(), 0);
  std::fill(vHparams.aIsRecrImpl.begin(), vHparams.aIsRecrImpl.end(), fLlmArchIsRecurrent(pMl.mGetArch()) ? 1 : 0);
  std::fill(vHparams.aIsIndexerFullImpl.begin(), vHparams.aIsIndexerFullImpl.end(), 0);

  std::fill(vHparams.aXieluAlphaN.begin(), vHparams.aXieluAlphaN.end(), 0.0f);
  std::fill(vHparams.aXieluAlphaP.begin(), vHparams.aXieluAlphaP.end(), 0.0f);
  std::fill(vHparams.aXieluBeta.begin(), vHparams.aXieluBeta.end(), 0.0f);
  std::fill(vHparams.aXieluEps.begin(), vHparams.aXieluEps.end(), 0.0f);

  std::fill(vHparams.aSwigluClampExp.begin(), vHparams.aSwigluClampExp.end(), 0.0f);
  std::fill(vHparams.aSwigluClampShexp.begin(), vHparams.aSwigluClampShexp.end(), 0.0f);

  pMl.mGetKeyOrArr(cLlmKvFeedForwardLength, vHparams.aNFfArr, vHparams.mNLayer(), false);
  pMl.mGetKeyOrArr(cLlmKvAttentionHeadCount, vHparams.aNHeadArr, vHparams.mNLayer(), false);

  // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
  std::fill(vHparams.aDeepstackMappingArr.begin(), vHparams.aDeepstackMappingArr.end(), -1);

  // n_head_kv is optional, default to n_head
  vHparams.aNHeadKvArr = vHparams.aNHeadArr;

  pMl.mGetKeyOrArr(cLlmKvAttentionHeadCountKv, vHparams.aNHeadKvArr, vHparams.mNLayer(), false);

  bool vRopeFinetuned = false;
  pMl.mGetKey(cLlmKvRopeScalingFinetuned, vRopeFinetuned, false);
  vHparams.vRopeFinetuned = vRopeFinetuned;

  vHparams.vNCtxOrigYarn = vHparams.vNCtxTrain;
  pMl.mGetKey(cLlmKvRopeScalingOrigCtxLen, vHparams.vNCtxOrigYarn, false);

  // rope_freq_base (optional)
  vHparams.vRopeFreqBaseTrain = 10000.0f;
  pMl.mGetKey(cLlmKvRopeFreqBase, vHparams.vRopeFreqBaseTrain, false);

  std::string vRopeScaling("linear");
  pMl.mGetKey(cLlmKvRopeScalingType, vRopeScaling, false);
  vHparams.vRopeScalingTypeTrain = fMgsRopeScalingTypeFromString(vRopeScaling);
  GGML_ASSERT(vHparams.vRopeScalingTypeTrain != MGS_ROPE_SCALING_TYPE_UNSPECIFIED);

  // TODO: Handle SWA metadata similarly when models start implementing it
  // rope_freq_scale (inverse of the kv) is optional
  float vRopescale = 0.0f;
  pMl.mGetKey(cLlmKvRopeScalingFactor, vRopescale, false);
  vHparams.vRopeFreqScaleTrain = vRopescale == 0.0f ? 1.0f : 1.0f / vRopescale;

  pMl.mGetKey(cLlmKvRopeScalingAttnFactor, vHparams.vRopeAttnFactor, false);
  pMl.mGetKey(cLlmKvRopeScalingAlpha, vHparams.vRopeScalingAlpha, false);

  // non-transformer models do not have attention heads
  if (vHparams.mNHead() > 0) {
    // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
    // gpt-j n_rot = rotary_dim

    vHparams.vNEmbdHeadKFull = vHparams.vNEmbd / vHparams.mNHead();
    pMl.mGetKey(cLlmKvAttentionKeyLength, vHparams.vNEmbdHeadKFull, false);

    vHparams.vNEmbdHeadVFull = vHparams.vNEmbd / vHparams.mNHead();
    pMl.mGetKey(cLlmKvAttentionValueLength, vHparams.vNEmbdHeadVFull, false);

    // sanity check for n_rot (optional)
    vHparams.vNRotFull = vHparams.vNEmbdHeadKFull;

    pMl.mGetKey(cLlmKvRopeDimensionCount, vHparams.vNRotFull, false);

    if (vArch == cLlmArchDeci || vArch == cLlmArchFalcon) {
      if (vHparams.vNRotFull != vHparams.vNEmbdHeadKFull) {
        throw std::runtime_error(
            fFormat("invalid n_rot: %u, expected %u", vHparams.vNRotFull, vHparams.vNEmbdHeadKFull));
      }
    }
  } else {
    vHparams.vNRotFull = 0;
    vHparams.vNEmbdHeadKFull = 0;
    vHparams.vNEmbdHeadVFull = 0;
  }

  // head size and n_rot for SWA layers
  {
    vHparams.vNEmbdHeadKSwa = vHparams.vNEmbdHeadKFull;
    vHparams.vNEmbdHeadVSwa = vHparams.vNEmbdHeadVFull;
    pMl.mGetKey(cLlmKvAttentionKeyLengthSwa, vHparams.vNEmbdHeadKSwa, false);
    pMl.mGetKey(cLlmKvAttentionValueLengthSwa, vHparams.vNEmbdHeadVSwa, false);

    vHparams.vNRotSwa = vHparams.vNRotFull;
    pMl.mGetKey(cLlmKvRopeDimensionCountSwa, vHparams.vNRotSwa, false);
  }

  // for classifier models
  pMl.mGetArr(cLlmKvClassifierOutputLabels, lClassifierLabels, false);
  if (!lClassifierLabels.empty()) {
    vHparams.vNClsOut = lClassifierLabels.size();
  }

  // per-arch hparams
  mLoadArchHparams(pMl);

  vPimpl->vNBytes = pMl.vNBytes;

  vPimpl->vDescStr = mArchName() + " " + mTypeName() + " " + pMl.mFtypeName();

  vPimpl->vFtype = pMl.vFtype;

  if (vHparams.vFMaxAlibiBias > 0.0f) {
    vHparams.vUseAlibi = true;
  }

  vHparams.vRopeType = mgs_model_rope_type(this);
}

void mgs_model_base::mLoadVocab(mgs_model_loader &pMl) {
  const auto cKv = LLM_KV(vArch);

  vVocab.mLoad(pMl, cKv);
}

bool mgs_model_base::mLoadTensors(mgs_model_loader &pMl) {
  const auto &vSplitMode = vParams.split_mode;
  const bool cUseMlock = vParams.load_mode == MGS_LOAD_MODE_MLOCK || vParams.load_mode == MGS_LOAD_MODE_MMAP_MLOCK;
  const auto &vTensorSplit = vParams.tensor_split;

  const int cNLayerAll = vHparams.vNLayerAll;
  const int cNGpuLayers = this->mNGpuLayers();

  const bool cUseMmapBuffer = true;

  this->vMl = &pMl; // to be used by mCreateTensor() and mLoadArchTensors()

  MGS_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n", __func__,
               mgs_load_mode_name(vParams.load_mode));

  // build a list of buffer types for the CPU and GPU devices
  vPimpl->vCpuBuftList = fMakeCpuBuftList(lDevices, vParams.use_extra_bufts, vParams.no_host);
  for (const auto &vDev : lDevices) {
    buft_list_t vBuftList = fMakeGpuBuftList(vDev.vDev, vSplitMode, vTensorSplit);
    // add CPU buffer types as a fallback
    vBuftList.insert(vBuftList.end(), vPimpl->vCpuBuftList.begin(), vPimpl->vCpuBuftList.end());
    vPimpl->dGpuBuftList.emplace(vDev.vDev, std::move(vBuftList));
  }

  ggml_backend_dev_t vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (vCpuDev == nullptr) {
    throw std::runtime_error(fFormat("%s: no CPU backend found", __func__));
  }

  // calculate the split points
  bool vAllZero = vTensorSplit == nullptr ||
                  std::all_of(vTensorSplit, vTensorSplit + mNDevices(), [](float pX) { return pX == 0.0f; });
  std::vector<float> lSplits(mNDevices());
  if (vAllZero) {
    // default split, by free memory
    for (size_t vI = 0; vI < mNDevices(); ++vI) {
      ggml_backend_dev_t vDev = lDevices[vI].vDev;
      size_t vTotal;
      size_t vFree;
      ggml_backend_dev_memory(vDev, &vFree, &vTotal);

      // devices can return 0 bytes for free and total memory if they do not
      // have any to report. in this case, we will use the host memory as a fallback
      if (vFree == 0 && vTotal == 0) {
        ggml_backend_dev_memory(vCpuDev, &vFree, &vTotal);
      }
      lSplits[vI] = vFree;
    }
  } else {
    std::copy(vTensorSplit, vTensorSplit + mNDevices(), lSplits.begin());
  }

  // sum and normalize the splits to get the split points
  float vSplitSum = 0.0f;
  for (size_t vI = 0; vI < mNDevices(); ++vI) {
    vSplitSum += lSplits[vI];
    lSplits[vI] = vSplitSum;
  }
  for (size_t vI = 0; vI < mNDevices(); ++vI) {
    lSplits[vI] /= vSplitSum;
  }

  const int cIGpuStart = std::max(cNLayerAll + 1 - cNGpuLayers, 0);
  const int cActGpuLayers = lDevices.empty() ? 0 : std::min(cNGpuLayers, cNLayerAll + 1);
  auto vGetLayerBuftList = [&](int pIl) -> mgs_model::impl::layer_dev {
    const bool cIsSwa = pIl < cNLayerAll && vHparams.mIsSwa(pIl);
    if (pIl < cIGpuStart || (pIl - cIGpuStart) >= cActGpuLayers) {
      MGS_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", pIl, ggml_backend_dev_name(vCpuDev),
                    cIsSwa);
      return {vCpuDev, &vPimpl->vCpuBuftList};
    }
    const int cLayerGpu =
        std::upper_bound(lSplits.begin(), lSplits.begin() + mNDevices(), float(pIl - cIGpuStart) / cActGpuLayers) -
        lSplits.begin();
    auto *vDev = lDevices.at(cLayerGpu).vDev;
    MGS_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", pIl, ggml_backend_dev_name(vDev),
                  cIsSwa);
    return {vDev, &vPimpl->dGpuBuftList.at(vDev)};
  };

  // assign the input layer
  // there is very little benefit to offloading the input layer, so always keep it on the CPU
  vPimpl->vDevInput = {vCpuDev, &vPimpl->vCpuBuftList};

  // assign the repeating layers to the devices according to the splits
  vPimpl->lDevLayer.resize(cNLayerAll);
  for (int vIl = 0; vIl < cNLayerAll; ++vIl) {
    vPimpl->lDevLayer[vIl] = vGetLayerBuftList(vIl);
  }

  // assign the output layer
  vPimpl->vDevOutput = vGetLayerBuftList(cNLayerAll);

  // Reserve the placement of the MTP layers and their output head before
  // fitting any other weights. Inspect every selected GPU and account for
  // backend allocation sizes, alignment and tied output embeddings.
  if (pMl.vLoadMtp && vHparams.vNLayerNextn > 0 && !lDevices.empty()) {
    const int cMtpFirstLayer = (int)vHparams.mNLayer();
    const size_t cOutputGroup = vHparams.vNLayerNextn;
    // hyper-connection trunks (qwen4exp) have the mixer norm in place of the attention norm
    pMl.vMtpHeadOnly = cMtpFirstLayer == 0 || (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr &&
                                               pMl.mGetWeight("blk.0.hc_attn_norm.weight") == nullptr);
    std::vector<std::vector<const ggml_tensor *>> ldMtpWeights(cOutputGroup + 1);
    const bool cTiedOutput = pMl.mGetWeight("output.weight") == nullptr;
    for (const auto &dWeight : pMl.dWeightsMap) {
      int vLayer = -1;
      if (sscanf(dWeight.first.c_str(), "blk.%d.", &vLayer) == 1 && vLayer >= cMtpFirstLayer && vLayer < cNLayerAll) {
        ldMtpWeights[vLayer - cMtpFirstLayer].push_back(dWeight.second.vTensor);
      } else if (pMl.vMtpHeadOnly || dWeight.first.rfind("output.", 0) == 0 ||
                 dWeight.first.rfind("output_norm.", 0) == 0 ||
                 (cTiedOutput && dWeight.first.rfind("token_embd.", 0) == 0)) {
        ldMtpWeights[cOutputGroup].push_back(dWeight.second.vTensor);
      }
    }

    std::vector<size_t> aFreeBytes(lDevices.size());
    std::vector<std::vector<size_t>> ldGroupBytes(lDevices.size(), std::vector<size_t>(ldMtpWeights.size()));
    for (size_t vDevice = 0; vDevice < lDevices.size(); ++vDevice) {
      auto vGpu = lDevices[vDevice].vDev;
      size_t vTotal = 0;
      ggml_backend_dev_memory(vGpu, &aFreeBytes[vDevice], &vTotal);
      auto vBufferType = ggml_backend_dev_buffer_type(vGpu);
      const size_t cAlignment = ggml_backend_buft_get_alignment(vBufferType);
      // MTP uses real device memory, even when the trunk uses row splitting.
      vPimpl->dMtpBufferLists.emplace(vGpu, buft_list_t{{vGpu, vBufferType}});
      for (size_t vGroup = 0; vGroup < ldMtpWeights.size(); ++vGroup) {
        for (const auto *pTensor : ldMtpWeights[vGroup]) {
          const size_t cBytes = ggml_backend_buft_get_alloc_size(vBufferType, pTensor);
          ldGroupBytes[vDevice][vGroup] += GGML_PAD(cBytes, cAlignment);
        }
      }
    }

    auto fAssignMtpGroup = [&](size_t pGroup, size_t pDevice) {
      auto vGpu = lDevices[pDevice].vDev;
      const mgs_model::impl::layer_dev dPlacement = {vGpu, &vPimpl->dMtpBufferLists.at(vGpu)};
      if (pGroup == cOutputGroup) {
        vPimpl->vDevOutput = dPlacement;
      } else {
        vPimpl->lDevLayer[cMtpFirstLayer + pGroup] = dPlacement;
      }
      aFreeBytes[pDevice] -= ldGroupBytes[pDevice][pGroup];
      MGS_LOG_INFO("%s: MTP %s %zu assigned to %s (%.2f MiB)\n", __func__, pGroup == cOutputGroup ? "output" : "layer",
                   pGroup, ggml_backend_dev_name(vGpu), ldGroupBytes[pDevice][pGroup] / 1024.0 / 1024.0);
    };

    // Keep the heads beside their existing placement when they fit there;
    // otherwise inspect every GPU before falling back to system memory.
    auto vPreferredDevice = vPimpl->lDevLayer[cMtpFirstLayer].vDev;
    if (vPreferredDevice == vCpuDev) {
      vPreferredDevice = vPimpl->vDevOutput.vDev;
    }
    if (vPreferredDevice == vCpuDev) {
      vPreferredDevice = lDevices.back().vDev;
    }
    size_t vWholeDevice = lDevices.size();
    size_t vBestRemainder = 0;
    for (size_t vDevice = 0; vDevice < lDevices.size(); ++vDevice) {
      const size_t cBytes = std::accumulate(ldGroupBytes[vDevice].begin(), ldGroupBytes[vDevice].end(), size_t(0));
      if (cBytes <= aFreeBytes[vDevice] && lDevices[vDevice].vDev == vPreferredDevice) {
        vWholeDevice = vDevice;
        break;
      }
      if (cBytes <= aFreeBytes[vDevice] &&
          (vWholeDevice == lDevices.size() || aFreeBytes[vDevice] - cBytes > vBestRemainder)) {
        vWholeDevice = vDevice;
        vBestRemainder = aFreeBytes[vDevice] - cBytes;
      }
    }
    if (vWholeDevice < lDevices.size()) {
      for (size_t vGroup = 0; vGroup < ldMtpWeights.size(); ++vGroup) {
        fAssignMtpGroup(vGroup, vWholeDevice);
      }
    } else {
      // Multiple heads may fit across GPUs even when they do not fit together.
      // Place larger groups first to avoid stranding space needed by the head.
      std::vector<size_t> aGroups(ldMtpWeights.size());
      std::iota(aGroups.begin(), aGroups.end(), 0);
      std::stable_sort(aGroups.begin(), aGroups.end(), [&](size_t pLeft, size_t pRight) {
        return ldGroupBytes.front()[pLeft] > ldGroupBytes.front()[pRight];
      });
      for (size_t vGroup : aGroups) {
        size_t vBestDevice = lDevices.size();
        size_t vRemaining = 0;
        for (size_t vDevice = 0; vDevice < lDevices.size(); ++vDevice) {
          const size_t cBytes = ldGroupBytes[vDevice][vGroup];
          if (cBytes <= aFreeBytes[vDevice] &&
              (vBestDevice == lDevices.size() || aFreeBytes[vDevice] - cBytes < vRemaining)) {
            vBestDevice = vDevice;
            vRemaining = aFreeBytes[vDevice] - cBytes;
          }
        }
        if (vBestDevice < lDevices.size()) {
          fAssignMtpGroup(vGroup, vBestDevice);
        } else {
          const mgs_model::impl::layer_dev dPlacement = {vCpuDev, &vPimpl->vCpuBuftList};
          if (vGroup == cOutputGroup) {
            vPimpl->vDevOutput = dPlacement;
          } else {
            vPimpl->lDevLayer[cMtpFirstLayer + vGroup] = dPlacement;
          }
          MGS_LOG_WARN("%s: no selected GPU has enough free memory for MTP %s %zu; using system memory\n", __func__,
                       vGroup == cOutputGroup ? "output" : "layer", vGroup);
        }
      }
    }
    pMl.vKeepMtpOnDevice = true;
    pMl.vMtpFirstLayer = cMtpFirstLayer;
    if (pMl.vMtpHeadOnly) {
      vPimpl->vDevInput = vPimpl->vDevOutput;
    }
  }

  const auto cTensorNotRequired = mgs_model_loader::cTensorNotRequired;

  // create tensors for the weights
  {
    // TODO: move to a separate function
    const auto cTn = LLM_TN(vArch);

    const int64_t cNExpert = vHparams.vNExpert;
    const int64_t cNExpertUsed = vHparams.vNExpertUsed;

    if (cNExpert > 0 && cNExpertUsed == 0) {
      throw std::runtime_error("model has expert layers but no expert layers are used");
    }

    lLayers.resize(cNLayerAll);

    // call the per-model loading function
    mLoadArchTensors(pMl);

    // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
    // this avoids having to add scale loading to every architecture
    for (int vI = 0; vI < cNLayerAll; ++vI) {
      auto &vLayer = lLayers[vI];

      // attention weight scales (per-tensor, shape {1})
      if (!vLayer.vWqS && vLayer.vWq) {
        vLayer.vWqS = mCreateTensor(cTn(cLlmTensorAttnQ, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWkS && vLayer.vWk) {
        vLayer.vWkS = mCreateTensor(cTn(cLlmTensorAttnK, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWvS && vLayer.vWv) {
        vLayer.vWvS = mCreateTensor(cTn(cLlmTensorAttnV, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWoS && vLayer.vWo) {
        vLayer.vWoS = mCreateTensor(cTn(cLlmTensorAttnOut, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWqkvS && vLayer.vWqkv) {
        vLayer.vWqkvS = mCreateTensor(cTn(cLlmTensorAttnQkv, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWqkvGateS && vLayer.vWqkvGate) {
        vLayer.vWqkvGateS = mCreateTensor(cTn(cLlmTensorAttnGate, "scale", vI), {1}, cTensorNotRequired);
      }

      // dense FFN weight scales (per-tensor, shape {1})
      if (!vLayer.vFfnGateS && vLayer.vFfnGate) {
        vLayer.vFfnGateS = mCreateTensor(cTn(cLlmTensorFfnGate, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownS && vLayer.vFfnDown) {
        vLayer.vFfnDownS = mCreateTensor(cTn(cLlmTensorFfnDown, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpS && vLayer.vFfnUp) {
        vLayer.vFfnUpS = mCreateTensor(cTn(cLlmTensorFfnUp, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnGateShexpS && vLayer.vFfnGateShexp) {
        vLayer.vFfnGateShexpS = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownShexpS && vLayer.vFfnDownShexp) {
        vLayer.vFfnDownShexpS = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpShexpS && vLayer.vFfnUpShexp) {
        vLayer.vFfnUpShexpS = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "scale", vI), {1}, cTensorNotRequired);
      }

      // MoE expert weight scales (per-expert, shape {n_expert})
      if (!vLayer.vFfnGateExpsS && vLayer.vFfnGateExps) {
        vLayer.vFfnGateExpsS = mCreateTensor(cTn(cLlmTensorFfnGateExps, "scale", vI), {cNExpert}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownExpsS && vLayer.vFfnDownExps) {
        vLayer.vFfnDownExpsS = mCreateTensor(cTn(cLlmTensorFfnDownExps, "scale", vI), {cNExpert}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpExpsS && vLayer.vFfnUpExps) {
        vLayer.vFfnUpExpsS = mCreateTensor(cTn(cLlmTensorFfnUpExps, "scale", vI), {cNExpert}, cTensorNotRequired);
      }

      // recurrent / linear-attention weight scales (per-tensor, shape {1})
      if (!vLayer.vSsmInS && vLayer.vSsmIn) {
        vLayer.vSsmInS = mCreateTensor(cTn(cLlmTensorSsmIn, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmOutS && vLayer.vSsmOut) {
        vLayer.vSsmOutS = mCreateTensor(cTn(cLlmTensorSsmOut, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmAlphaS && vLayer.vSsmAlpha) {
        vLayer.vSsmAlphaS = mCreateTensor(cTn(cLlmTensorSsmAlpha, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmBetaS && vLayer.vSsmBeta) {
        vLayer.vSsmBetaS = mCreateTensor(cTn(cLlmTensorSsmBeta, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vNextn.vEhProjS && vLayer.vNextn.vEhProj) {
        vLayer.vNextn.vEhProjS = mCreateTensor(cTn(cLlmTensorNextnEhProj, "scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vNextn.vSharedHeadHeadS && vLayer.vNextn.vSharedHeadHead) {
        vLayer.vNextn.vSharedHeadHeadS =
            mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "scale", vI), {1}, cTensorNotRequired);
      }

      // input scales
      if (!vLayer.vWqInS && vLayer.vWq) {
        vLayer.vWqInS = mCreateTensor(cTn(cLlmTensorAttnQ, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWkInS && vLayer.vWk) {
        vLayer.vWkInS = mCreateTensor(cTn(cLlmTensorAttnK, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWvInS && vLayer.vWv) {
        vLayer.vWvInS = mCreateTensor(cTn(cLlmTensorAttnV, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWoInS && vLayer.vWo) {
        vLayer.vWoInS = mCreateTensor(cTn(cLlmTensorAttnOut, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWqkvInS && vLayer.vWqkv) {
        vLayer.vWqkvInS = mCreateTensor(cTn(cLlmTensorAttnQkv, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vWqkvGateInS && vLayer.vWqkvGate) {
        vLayer.vWqkvGateInS = mCreateTensor(cTn(cLlmTensorAttnGate, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnGateInS && vLayer.vFfnGate) {
        vLayer.vFfnGateInS = mCreateTensor(cTn(cLlmTensorFfnGate, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownInS && vLayer.vFfnDown) {
        vLayer.vFfnDownInS = mCreateTensor(cTn(cLlmTensorFfnDown, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpInS && vLayer.vFfnUp) {
        vLayer.vFfnUpInS = mCreateTensor(cTn(cLlmTensorFfnUp, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnGateExpsInS && vLayer.vFfnGateExps) {
        vLayer.vFfnGateExpsInS =
            mCreateTensor(cTn(cLlmTensorFfnGateExps, "input_scale", vI), {cNExpert}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownExpsInS && vLayer.vFfnDownExps) {
        vLayer.vFfnDownExpsInS =
            mCreateTensor(cTn(cLlmTensorFfnDownExps, "input_scale", vI), {cNExpert}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpExpsInS && vLayer.vFfnUpExps) {
        vLayer.vFfnUpExpsInS =
            mCreateTensor(cTn(cLlmTensorFfnUpExps, "input_scale", vI), {cNExpert}, cTensorNotRequired);
      }
      if (!vLayer.vFfnGateShexpInS && vLayer.vFfnGateShexp) {
        vLayer.vFfnGateShexpInS =
            mCreateTensor(cTn(cLlmTensorFfnGateShexp, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnDownShexpInS && vLayer.vFfnDownShexp) {
        vLayer.vFfnDownShexpInS =
            mCreateTensor(cTn(cLlmTensorFfnDownShexp, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vFfnUpShexpInS && vLayer.vFfnUpShexp) {
        vLayer.vFfnUpShexpInS = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmInInS && vLayer.vSsmIn) {
        vLayer.vSsmInInS = mCreateTensor(cTn(cLlmTensorSsmIn, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmOutInS && vLayer.vSsmOut) {
        vLayer.vSsmOutInS = mCreateTensor(cTn(cLlmTensorSsmOut, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmAlphaInS && vLayer.vSsmAlpha) {
        vLayer.vSsmAlphaInS = mCreateTensor(cTn(cLlmTensorSsmAlpha, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vSsmBetaInS && vLayer.vSsmBeta) {
        vLayer.vSsmBetaInS = mCreateTensor(cTn(cLlmTensorSsmBeta, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vNextn.vEhProjInS && vLayer.vNextn.vEhProj) {
        vLayer.vNextn.vEhProjInS =
            mCreateTensor(cTn(cLlmTensorNextnEhProj, "input_scale", vI), {1}, cTensorNotRequired);
      }
      if (!vLayer.vNextn.vSharedHeadHeadInS && vLayer.vNextn.vSharedHeadHead) {
        vLayer.vNextn.vSharedHeadHeadInS =
            mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "input_scale", vI), {1}, cTensorNotRequired);
      }
    }
    // output scales
    if (vOutput && vOutput->type == GGML_TYPE_NVFP4) {
      // weight scale
      if (!vOutputS) {
        vOutputS = mCreateTensor(cTn(cLlmTensorOutput, "scale"), {1}, cTensorNotRequired);
      }
      // input scale
      if (!vOutputInS) {
        vOutputInS = mCreateTensor(cTn(cLlmTensorOutput, "input_scale"), {1}, cTensorNotRequired);
      }
    }
  }
  pMl.mDoneGettingTensors();

  // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
  // If sidecar scales exist, the output weight must be an actual output tensor.
  GGML_ASSERT(!(vOutput && vTokEmbd && strcmp(vOutput->name, vTokEmbd->name) == 0 && vOutput->type == GGML_TYPE_NVFP4 &&
                (vOutputS || vOutputInS)));
  // MTP contexts precede the trunk both for allocation and for data upload.
  std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr *>> ldWeightContexts;
  ldWeightContexts.reserve(pMl.dMtpContexts.size() + pMl.dCtxMap.size());
  for (auto *pContexts : {&pMl.dMtpContexts, &pMl.dCtxMap}) {
    for (auto &[vBufferType, vContext] : *pContexts) {
      ldWeightContexts.emplace_back(vBufferType, &vContext);
    }
  }
  // populate tensors_by_name
  for (auto &[vBufferType, pContext] : ldWeightContexts) {
    const auto &vCtxPtr = *pContext;
    for (auto *vCur = ggml_get_first_tensor(vCtxPtr.get()); vCur != NULL;
         vCur = ggml_get_next_tensor(vCtxPtr.get(), vCur)) {
      lTensorsByName.emplace_back(ggml_get_name(vCur), vCur);
    }
  }

  pMl.mInitMappings(true, cUseMlock ? &vPimpl->vMlockMmaps : nullptr);
  vPimpl->vMappings.reserve(pMl.vMappings.size());

  // create the backend buffers
  std::vector<std::pair<ggml_context *, mgs_buf_map>> lCtxBufMaps;
  lCtxBufMaps.reserve(ldWeightContexts.size());

  // Ensure we have enough capacity for the maximum backend buffer we will potentially create
  const size_t cNMaxBackendBuffer = ldWeightContexts.size() * pMl.vFiles.size();
  vPimpl->lCtxsBufs.reserve(cNMaxBackendBuffer);

  for (auto &[vBuft, pContext] : ldWeightContexts) {
    auto &vCtxPtr = *pContext;
    ggml_context *vCtx = vCtxPtr.get();

    // skip contexts without tensors
    if (ggml_get_first_tensor(vCtx) == nullptr) {
      continue;
    }

    mgs_buf_map vBufMap;
    vBufMap.reserve(cNMaxBackendBuffer);

    // check if it is possible to use buffer_from_host_ptr with this buffer type
    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft);
    if (!vDev) {
      // FIXME: workaround for CPU backend buft having a NULL device
      vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      if (!vDev) {
        throw std::runtime_error(fFormat("%s: no CPU backend found", __func__));
      }
    }
    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDev, &vProps);
    bool vBufferFromHostPtrSupported = vProps.caps.buffer_from_host_ptr;
    bool vIsDefaultBuft = vBuft == ggml_backend_dev_buffer_type(vDev);

    std::vector<ggml_backend_buffer_ptr> lBufs;
    if (pMl.vUseMmap && cUseMmapBuffer && vBufferFromHostPtrSupported && vIsDefaultBuft) {
      GGML_ASSERT(!pMl.vNoAlloc);
      for (uint32_t vIdx = 0; vIdx < pMl.vFiles.size(); vIdx++) {
        // only the mmap region containing the tensors in the model is mapped to the backend buffer
        // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
        //     then we could just use metal for all layers
        // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
        void *vAddr = nullptr;
        size_t first, last; // NOLINT
        pMl.mGetMappingRange(&first, &last, &vAddr, vIdx, vCtx);
        if (first >= last) {
          continue;
        }
        const size_t cMaxSize = ggml_get_max_tensor_size(vCtx);
        ggml_backend_buffer_t vBuf =
            ggml_backend_dev_buffer_from_host_ptr(vDev, (char *)vAddr + first, last - first, cMaxSize);
        if (vBuf == nullptr) {
          throw std::runtime_error(fFormat("unable to allocate %s buffer", ggml_backend_buft_name(vBuft)));
        }
        lBufs.emplace_back(vBuf);
        vBufMap.emplace(vIdx, vBuf);
      }
    } else {
      ggml_backend_buffer_t vBuf;
      if (pMl.vNoAlloc) {
        vBuf = ggml_backend_buft_alloc_buffer(vBuft, /*size =*/0); // dummy buffer
        for (ggml_tensor *vT = ggml_get_first_tensor(vCtx); vT != nullptr; vT = ggml_get_next_tensor(vCtx, vT)) {
          vT->buffer = vBuf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
        }
      } else {
        vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx, vBuft); // real buffer
      }
      if (vBuf == nullptr) {
        throw std::runtime_error(fFormat("unable to allocate %s buffer", ggml_backend_buft_name(vBuft)));
      }
      if (cUseMlock && ggml_backend_buffer_is_host(vBuf)) {
        vPimpl->vMlockBufs.emplace_back(new mgs_mlock);
        auto &vMlockBuf = vPimpl->vMlockBufs.back();
        vMlockBuf->mInit(ggml_backend_buffer_get_base(vBuf));
        vMlockBuf->mGrowTo(ggml_backend_buffer_get_size(vBuf));
      }
      lBufs.emplace_back(vBuf);
      for (uint32_t vIdx = 0; vIdx < pMl.vFiles.size(); vIdx++) {
        vBufMap.emplace(vIdx, vBuf);
      }
    }

    for (auto &vBuf : lBufs) {
      // indicate that this buffer contains weights
      // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to
      // the backend that contains the weight
      ggml_backend_buffer_set_usage(vBuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    vPimpl->lCtxsBufs.emplace_back(std::move(vCtxPtr), std::move(lBufs));

    lCtxBufMaps.emplace_back(vCtx, vBufMap);
  }

  if (mgs_supports_gpu_offload()) {
    const int cNGpu = std::min(cNGpuLayers, cNLayerAll);

    int vNRepeating = cNGpu;
    if (vNRepeating > 0) {
      MGS_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
      vNRepeating--;
    }
    MGS_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, vNRepeating);

    const int cMaxBackendSupportedLayers = cNLayerAll + 1;
    const int cMaxOffloadableLayers = cNLayerAll + 1;

    MGS_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(cNGpuLayers, cMaxOffloadableLayers),
                 cMaxBackendSupportedLayers);
  }

  // print memory requirements per buffer type
  for (auto &[_, vBufs] : vPimpl->lCtxsBufs) {
    for (auto &vBuf : vBufs) {
      MGS_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf.get()),
                   ggml_backend_buffer_get_size(vBuf.get()) / 1024.0 / 1024.0);
    }
  }

  if (pMl.vNoAlloc) {
    return true;
  }

  // load tensor data
  for (auto &[vCtx, vBufMap] : lCtxBufMaps) {
    if (!pMl.mLoadAllData(vCtx, vBufMap, cUseMlock ? &vPimpl->vMlockMmaps : NULL, vParams.progress_callback,
                          vParams.progress_callback_user_data)) {
      return false;
    }
  }

  if (cUseMmapBuffer) {
    for (auto &vMapping : pMl.vMappings) {
      vPimpl->vMappings.emplace_back(std::move(vMapping));
    }
  }

  return true;
}

ggml_tensor *mgs_model_base::mCreateTensor(mgs_model_loader &pMl, const LLM_TN_IMPL &pTn,
                                           const std::initializer_list<int64_t> &pNe, int pFlags) {
  const buft_list_t *vBuftListLayer = pTn.cBid == -1 ? nullptr : vPimpl->lDevLayer.at(pTn.cBid).vBuftList;
  return pMl.mCreateTensor(vHparams, &vPimpl->vCpuBuftList, vPimpl->vDevInput.vBuftList, vPimpl->vDevOutput.vBuftList,
                           vBuftListLayer, pTn, pNe, pFlags);
}

std::string mgs_model::mArchName() const { return fLlmArchName(vArch); }

std::string mgs_model::mTypeName() const { return fLlmTypeName(vType); }

std::string mgs_model::mDesc() const { return vPimpl->vDescStr; }

mgs_ftype mgs_model::mFtype() const { return vPimpl->vFtype; }

size_t mgs_model::size() const { return vPimpl->vNBytes; }

size_t mgs_model::mNTensors() const { return lTensorsByName.size(); }

size_t mgs_model::mNDevices() const { return lDevices.size(); }

const float *mgs_model::mTensorSplit() const { return vParams.tensor_split; }

uint32_t mgs_model::mNGpuLayers() const {
  // note: plus 1 for the "output" layer
  return vParams.n_gpu_layers >= 0 ? vParams.n_gpu_layers : vHparams.vNLayerAll + 1;
}

mgs_split_mode mgs_model::mSplitMode() const { return vParams.split_mode; }

std::map<ggml_backend_buffer_type_t, size_t> mgs_model::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto &[vCtx, vBufs] : vPimpl->lCtxsBufs) {
    if (vHparams.vNoAlloc) {
      GGML_ASSERT(vBufs.size() == 1);
      ggml_backend_buffer_t vBuf = vBufs[0].get();
      GGML_ASSERT(ggml_backend_buffer_get_base(vBuf) == nullptr);
      ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(vBuf);
      dRet[vBuft] += ggml_backend_alloc_ctx_tensors_from_buft_size(vCtx.get(), vBuft);
    } else {
      for (const auto &vBuf : vBufs) {
        // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
        dRet[ggml_backend_buffer_get_type(vBuf.get())] += ggml_backend_buffer_get_size(vBuf.get());
      }
    }
  }
  return dRet;
}

uint64_t mgs_model::mNElements() const { return vPimpl->vNElements; }

void mgs_model::mPrintInfo() const {
  const std::string cRopeScalingType = mgs_rope_scaling_type_name(vHparams.vRopeScalingTypeTrain);

  auto vPrintF = [](const std::function<int32_t(uint32_t)> &pF, uint32_t pN) {
    bool vIsVar = false;

    std::vector<int32_t> lV;
    for (uint32_t vI = 0; vI < pN; ++vI) {
      lV.push_back(pF(vI));
      if (lV[vI] != lV[0]) {
        vIsVar = true;
      }
    }

    std::stringstream vSs;

    if (vIsVar) {
      vSs << "[";
      for (uint32_t vI = 0; vI < pN; ++vI) {
        vSs << lV[vI];
        if (vI < pN - 1) {
          vSs << ", ";
        }
      }
      vSs << "]";
    } else {
      vSs << lV[0];
    }

    return vSs.str();
  };

  // hparams
  MGS_LOG_INFO("%s: arch                  = %s\n", __func__, mArchName().c_str());
  MGS_LOG_INFO("%s: vocab_only            = %d\n", __func__, vHparams.vVocabOnly);
  MGS_LOG_INFO("%s: no_alloc              = %d\n", __func__, vHparams.vNoAlloc);

  if (!vHparams.vVocabOnly) {
    MGS_LOG_INFO("%s: n_ctx_train           = %u\n", __func__, vHparams.vNCtxTrain);
    MGS_LOG_INFO("%s: n_embd_inp            = %u\n", __func__, vHparams.mNEmbdInp());
    MGS_LOG_INFO("%s: n_embd                = %u\n", __func__, vHparams.vNEmbd);
    MGS_LOG_INFO("%s: n_embd_out            = %u\n", __func__, vHparams.mNEmbdOut());
    MGS_LOG_INFO("%s: n_layer               = %u\n", __func__, vHparams.mNLayer());
    MGS_LOG_INFO("%s: n_layer_all           = %u\n", __func__, vHparams.vNLayerAll);
    MGS_LOG_INFO("%s: n_head                = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNHead(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: n_head_kv             = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNHeadKv(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: n_rot                 = %u\n", __func__, vHparams.vNRotFull);
    MGS_LOG_INFO("%s: n_swa                 = %u\n", __func__, vHparams.vNSwa);
    MGS_LOG_INFO("%s: is_swa_any            = %u\n", __func__, vHparams.mIsSwaAny());
    MGS_LOG_INFO("%s: n_embd_head_k         = %u\n", __func__, vHparams.vNEmbdHeadKFull);
    MGS_LOG_INFO("%s: n_embd_head_v         = %u\n", __func__, vHparams.vNEmbdHeadVFull);
    MGS_LOG_INFO("%s: n_gqa                 = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNGqa(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: n_embd_k_gqa          = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNEmbdKGqa(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: n_embd_v_gqa          = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNEmbdVGqa(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: f_norm_eps            = %.1e\n", __func__, vHparams.vFNormEps);
    MGS_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n", __func__, vHparams.vFNormRmsEps);
    MGS_LOG_INFO("%s: f_clamp_kqv           = %.1e\n", __func__, vHparams.vFClampKqv);
    MGS_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n", __func__, vHparams.vFMaxAlibiBias);
    MGS_LOG_INFO("%s: f_logit_scale         = %.1e\n", __func__, vHparams.vFLogitScale);
    MGS_LOG_INFO("%s: f_attn_scale          = %.1e\n", __func__, vHparams.vFAttentionScale);
    MGS_LOG_INFO("%s: f_attn_value_scale    = %.4f\n", __func__, vHparams.vFAttnValueScale);
    MGS_LOG_INFO("%s: n_ff                  = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return vHparams.mNFf(pIl); }, vHparams.vNLayerAll).c_str());
    MGS_LOG_INFO("%s: n_expert              = %u\n", __func__, vHparams.vNExpert);
    MGS_LOG_INFO("%s: n_expert_used         = %u\n", __func__, vHparams.vNExpertUsed);
    MGS_LOG_INFO("%s: n_expert_groups       = %d\n", __func__, vHparams.vNExpertGroups);
    MGS_LOG_INFO("%s: n_group_used          = %d\n", __func__, vHparams.vNGroupUsed);
    MGS_LOG_INFO("%s: causal attn           = %d\n", __func__, vHparams.vCausalAttn);
    MGS_LOG_INFO("%s: pooling type          = %d\n", __func__, vHparams.vPoolingType);
    MGS_LOG_INFO("%s: rope type             = %d\n", __func__, vHparams.vRopeType);
    MGS_LOG_INFO("%s: rope scaling          = %s\n", __func__, cRopeScalingType.c_str());
    MGS_LOG_INFO("%s: freq_base_train       = %.1f\n", __func__, vHparams.vRopeFreqBaseTrain);
    MGS_LOG_INFO("%s: freq_scale_train      = %g\n", __func__, vHparams.vRopeFreqScaleTrain);
    if (vHparams.vSwaType != cMgsSwaTypeNone) {
      MGS_LOG_INFO("%s: freq_base_swa         = %.1f\n", __func__, vHparams.vRopeFreqBaseTrainSwa);
      MGS_LOG_INFO("%s: freq_scale_swa        = %g\n", __func__, vHparams.vRopeFreqScaleTrainSwa);
      MGS_LOG_INFO("%s: n_embd_head_k_swa     = %u\n", __func__, vHparams.vNEmbdHeadKSwa);
      MGS_LOG_INFO("%s: n_embd_head_v_swa     = %u\n", __func__, vHparams.vNEmbdHeadVSwa);
      MGS_LOG_INFO("%s: n_rot_swa             = %u\n", __func__, vHparams.vNRotSwa);
    }
    MGS_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n", __func__, vHparams.vNCtxOrigYarn);
    MGS_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n", __func__, vHparams.vRopeYarnLogMul);
    MGS_LOG_INFO("%s: rope_finetuned        = %s\n", __func__, vHparams.vRopeFinetuned ? "yes" : "unknown");
    if (vArch == cLlmArchGranite &&
        std::any_of(vHparams.aDeepstackMappingArr.begin(), vHparams.aDeepstackMappingArr.end(),
                    [](const auto &pEntry) { return pEntry >= 0; })) {
      MGS_LOG_INFO(
          "%s: deepstack_mapping_arr = %s\n", __func__,
          vPrintF([&](uint32_t pIl) { return vHparams.aDeepstackMappingArr[pIl]; }, vHparams.vNLayerAll).c_str());
    }
    // MRoPE (Multi-axis Rotary Position Embedding) sections
    if (const auto &vS = vHparams.aRopeSections; vS[0] || vS[1] || vS[2] || vS[3]) {
      MGS_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, vS[0], vS[1], vS[2], vS[3]);
    }
    if (!lClassifierLabels.empty()) {
      MGS_LOG_INFO("%s: n_cls_out             = %u\n", __func__, vHparams.vNClsOut);

      size_t vI = 0;
      for (const auto &vLabel : lClassifierLabels) {
        MGS_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, vI++, vLabel.c_str());
      }
    }

    if (vArch == cLlmArchMamba || vArch == cLlmArchMamba2 || vArch == cLlmArchJamba || vArch == cLlmArchFalconH1 ||
        vArch == cLlmArchPlamo2 || vArch == cLlmArchGraniteHybrid || vArch == cLlmArchQwen3next ||
        vArch == cLlmArchQwen35 || vArch == cLlmArchQwen35moe || vArch == cLlmArchQwen4exp ||
        vArch == cLlmArchNemotronH || vArch == cLlmArchNemotronHMoe) {
      MGS_LOG_INFO("%s: ssm_d_conv            = %u\n", __func__, vHparams.vSsmDConv);
      MGS_LOG_INFO("%s: ssm_d_inner           = %u\n", __func__, vHparams.vSsmDInner);
      MGS_LOG_INFO("%s: ssm_d_state           = %u\n", __func__, vHparams.vSsmDState);
      MGS_LOG_INFO("%s: ssm_dt_rank           = %u\n", __func__, vHparams.vSsmDtRank);
      MGS_LOG_INFO("%s: ssm_n_group           = %u\n", __func__, vHparams.vSsmNGroup);
      MGS_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n", __func__, vHparams.vSsmDtBCRms);
    }

    MGS_LOG_INFO("%s: model type            = %s\n", __func__, mTypeName().c_str());
    if (vPimpl->vNElements >= 1e12) {
      MGS_LOG_INFO("%s: model params          = %.2f T\n", __func__, vPimpl->vNElements * 1e-12);
    } else if (vPimpl->vNElements >= 1e9) {
      MGS_LOG_INFO("%s: model params          = %.2f B\n", __func__, vPimpl->vNElements * 1e-9);
    } else if (vPimpl->vNElements >= 1e6) {
      MGS_LOG_INFO("%s: model params          = %.2f M\n", __func__, vPimpl->vNElements * 1e-6);
    } else {
      MGS_LOG_INFO("%s: model params          = %.2f K\n", __func__, vPimpl->vNElements * 1e-3);
    }

    // general kv
    MGS_LOG_INFO("%s: general.name          = %s\n", __func__, vName.c_str());

    if (vArch == cLlmArchGlmDsa) {
      MGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, vHparams.vNLayerDenseLead);
      MGS_LOG_INFO("%s: n_lora_q              = %d\n", __func__, vHparams.vNLoraQ);
      MGS_LOG_INFO("%s: n_lora_kv             = %d\n", __func__, vHparams.vNLoraKv);
      MGS_LOG_INFO("%s: n_embd_head_k_mla     = %d\n", __func__, vHparams.mNEmbdHeadKMla());
      MGS_LOG_INFO("%s: n_embd_head_v_mla     = %d\n", __func__, vHparams.mNEmbdHeadVMla());
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, vHparams.vNExpertShared);
      MGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, vHparams.vExpertWeightsScale);
      MGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, vHparams.vExpertWeightsNorm);
      MGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fMgsExpertGatingFuncName((mgs_expert_gating_func_type)vHparams.vExpertGatingFunc));
    }

    if (vArch == cLlmArchQwen2moe) {
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, vHparams.vNFfShexp);
    }

    if (vArch == cLlmArchMellum || vArch == cLlmArchCohere2moe || vArch == cLlmArchQwen3moe ||
        vArch == cLlmArchOpenaiMoe || vArch == cLlmArchQwen3vlmoe || vArch == cLlmArchRnd1) {
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
    }

    if (vArch == cLlmArchMinicpm || vArch == cLlmArchGranite || vArch == cLlmArchGraniteMoe ||
        vArch == cLlmArchGraniteHybrid || vArch == cLlmArchNemotronHMoe) {
      MGS_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, vHparams.vFEmbeddingScale);
      MGS_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, vHparams.vFResidualScale);
      MGS_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, vHparams.vFAttentionScale);
      MGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, vHparams.vNFfShexp);
    }

    if (vArch == cLlmArchBailingmoe) {
      MGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, vHparams.vNLayerDenseLead);
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, vHparams.vNExpertShared);
      MGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, vHparams.vExpertWeightsScale);
      MGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, vHparams.vExpertWeightsNorm);
    }

    if (vArch == cLlmArchBailingmoe2) {
      MGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, vHparams.vNLayerDenseLead);
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, vHparams.vNFfShexp);
      MGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, vHparams.vNExpertShared);
      MGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, vHparams.vExpertWeightsScale);
      MGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, vHparams.vExpertWeightsNorm);
      MGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fMgsExpertGatingFuncName((mgs_expert_gating_func_type)vHparams.vExpertGatingFunc));
      MGS_LOG_INFO("%s: n_layer_nextn         = %d\n", __func__, vHparams.vNLayerNextn);
    }

    if (vArch == cLlmArchSmallthinker || vArch == cLlmArchLfm2moe) {
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fMgsExpertGatingFuncName((mgs_expert_gating_func_type)vHparams.vExpertGatingFunc));
    }

    if (vArch == cLlmArchGrovemoe) {
      MGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, vHparams.vNFfExp);
      MGS_LOG_INFO("%s: n_ff_chexp            = %d\n", __func__, vHparams.vNFfChexp);
      MGS_LOG_INFO("%s: n_group_experts       = %d\n", __func__, vHparams.vNGroupExperts);
      MGS_LOG_INFO("%s: expert_group_scale    = %.2f\n", __func__, vHparams.vExpertGroupScale);
    }
  }

  vVocab.mPrintInfo();
}

ggml_backend_dev_t mgs_model::mDevLayer(int pIl) const { return vPimpl->lDevLayer.at(pIl).vDev; }

ggml_backend_dev_t mgs_model::mDevOutput() const { return vPimpl->vDevOutput.vDev; }

template <typename F> static bool fBuftSupported(ggml_backend_buffer_type_t pBuft, ggml_backend_dev_t pDev, F &pFn) {
  ggml_init_params vParams = {
      /*.mem_size   =*/ggml_tensor_overhead() * 8,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,
  };

  ggml_context_ptr vCtx{ggml_init(vParams)};
  if (!vCtx) {
    throw std::runtime_error(fFormat("failed to create ggml context"));
  }

  ggml_backend_buffer_ptr vBuf{ggml_backend_buft_alloc_buffer(pBuft, 0)};
  ggml_tensor *vOpTensor = pFn(vCtx.get());
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    if (vOpTensor->src[vI] != nullptr) {
      assert(vOpTensor->src[vI]->buffer == nullptr);
      vOpTensor->src[vI]->buffer = vBuf.get();
    }
  }

  bool vOpSupported = ggml_backend_dev_supports_op(pDev, vOpTensor);

  return vOpSupported;
}

template <typename F> static ggml_backend_buffer_type_t fSelectBuft(const buft_list_t &pBuftList, const F &pFn) {
  for (const auto &vCur : pBuftList) {
    ggml_backend_dev_t vCurDev = vCur.first;
    ggml_backend_buffer_type_t vCurBuft = vCur.second;
    if (fBuftSupported(vCurBuft, vCurDev, pFn)) {
      return vCurBuft;
    }
  }

  throw std::runtime_error(fFormat("no suitable buffer type found"));
}

ggml_backend_buffer_type_t mgs_model::mSelectBuft(int pIl) const {
  return ::fSelectBuft(*vPimpl->lDevLayer.at(pIl).vBuftList, [&](ggml_context *pCtx) {
    ggml_tensor *vCur = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, vHparams.vNEmbd);
    ggml_tensor *vLayerDir = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, vHparams.vNEmbd);
    return ggml_add(pCtx, vCur, vLayerDir);
  });
}

bool mgs_model::mHasTensorOverrides() const { return vPimpl->vHasTensorOverrides; }

const ggml_tensor *mgs_model::mGetTensor(const char *pName) const {
  auto vIt = std::find_if(lTensorsByName.begin(), lTensorsByName.end(),
                          [pName](const std::pair<std::string, ggml_tensor *> &pIt) { return pIt.first == pName; });
  if (vIt == lTensorsByName.end()) {
    return nullptr;
  }

  return vIt->second;
}

float mgs_model::mGetRopeFreqBase(const mgs_cparams &pCparams, int pIl) const {
  return vHparams.mIsSwa(pIl) ? vHparams.vRopeFreqBaseTrainSwa : pCparams.vRopeFreqBase;
}

float mgs_model::mGetRopeFreqScale(const mgs_cparams &pCparams, int pIl) const {
  return vHparams.mIsSwa(pIl) ? vHparams.vRopeFreqScaleTrainSwa : pCparams.vRopeFreqScale;
}

ggml_tensor *mgs_model::mGetRopeFactors(const mgs_cparams &pCparams, int pIl) const {
  const uint32_t cNCtxSeq = pCparams.vNCtxSeq;

  // choose long/short freq factors based on the context size
  if (lLayers[pIl].vRopeFreqs != nullptr) {
    return lLayers[pIl].vRopeFreqs;
  }

  if (cNCtxSeq > vHparams.vNCtxOrigYarn) {
    return lLayers[pIl].vRopeLong;
  }

  return lLayers[pIl].vRopeShort;
}

mgs_memory_i *mgs_model::mCreateMemory(const mgs_memory_params &pParams, const mgs_cparams &pCparams) const {
  mgs_memory_i *vRes;

  switch (vArch) {
  // Models that need specific instantiation should be handled in the
  // switch statement
  case cLlmArchBert:
  case cLlmArchJinaBertV2:
  case cLlmArchJinaBertV3:
  case cLlmArchNomicBert:
  case cLlmArchNomicBertMoe:
  case cLlmArchNeoBert:
  case cLlmArchEurobert:
  case cLlmArchWavtokenizerDec:
  case cLlmArchModernBert:
  case cLlmArchGemmaEmbedding:
  case cLlmArchDream:
  case cLlmArchLlada:
  case cLlmArchLladaMoe:
  case cLlmArchRnd1: {
    vRes = nullptr;
  } break;
  case cLlmArchMinimaxM3: {
    // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
    mgs_kv_cache::layer_filter_cb vFilterIdx = [&](int32_t pIl) { return (uint32_t)pIl >= vHparams.vNLayerDenseLead; };

    vRes = new mgs_kv_cache_msa(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax, 1, vHparams.vNSwa,
                                vHparams.vSwaType, nullptr, vFilterIdx, nullptr);
  } break;
  case cLlmArchGlmDsa: {
    if (pParams.vCtxType == MGS_CONTEXT_TYPE_MTP && vHparams.vNLayerNextn > 0) {
      // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
      // MTP context uses a plain attention KV cache holding only the
      // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
      mgs_kv_cache::layer_filter_cb vFilter = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };

      vRes = new mgs_kv_cache(*this, vHparams, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn,
                              pCparams.vOffloadKqv, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax, 1,
                              vHparams.vNSwa, vHparams.vSwaType, nullptr, vFilter, nullptr, nullptr);
    } else {
      // Main context: DSA cache for the trunk layers only - the nextn
      // layer(s) are never attended by the trunk graph.
      mgs_kv_cache::layer_filter_cb vFilterMla = nullptr;
      if (vHparams.vNLayerNextn > 0) {
        vFilterMla = [&](uint32_t pIl) { return pIl < vHparams.mNLayer(); };
      }
      mgs_kv_cache::layer_filter_cb vFilterLid = [&](uint32_t pIl) {
        return pIl < vHparams.mNLayer() && vHparams.mIsIndexerFull(pIl);
      };

      vRes = new mgs_kv_cache_dsa(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                  pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax, 1, vHparams.vNSwa,
                                  vHparams.vSwaType, vFilterMla, vFilterLid, nullptr);
    }
  } break;
  case cLlmArchGlm5Next: {
    // KDA layers are recurrent, the DSA layers use a K-only MLA cache plus an indexer cache.
    // The NextN block is never attended by the trunk graph
    mgs_memory_hybrid_idx::layer_filter_cb vFilterAttn = [&](uint32_t pIl) {
      return pIl < vHparams.mNLayer() && !vHparams.mIsRecr(pIl);
    };
    mgs_memory_hybrid_idx::layer_filter_cb vFilterIdx = [&](uint32_t pIl) {
      return pIl < vHparams.mNLayer() && !vHparams.mIsRecr(pIl) && vHparams.mIsIndexerFull(pIl);
    };
    mgs_memory_hybrid_idx::layer_filter_cb vFilterRecr = [&](uint32_t pIl) {
      return pIl < vHparams.mNLayer() && vHparams.mIsRecr(pIl);
    };

    // the draft head is a single DSA layer
    if (pParams.vCtxType == MGS_CONTEXT_TYPE_MTP) {
      if (vHparams.vNLayerNextn == 0) {
        throw std::runtime_error("GLM5-Next MTP requires the NextN block in the GGUF");
      }
      vFilterAttn = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
      vFilterIdx = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
      vFilterRecr = [&](uint32_t) { return false; };
    }

    vRes = new mgs_memory_hybrid_idx(
        /* model             */ *this,
        /* attn_type_k       */ pParams.vTypeK,
        /* attn_type_v       */ pParams.vTypeV,
        /* attn_v_trans      */ !pCparams.vFlashAttn,
        /* attn_kv_size      */ pCparams.vNCtxSeq,
        /* attn_n_pad        */ 1,
        /* attn_n_swa        */ vHparams.vNSwa,
        /* attn_swa_type     */ vHparams.vSwaType,
        /* recurrent_type_r  */ GGML_TYPE_F32,
        /* recurrent_type_s  */ GGML_TYPE_F32,
        /* recurrent_rs_size */ std::max((uint32_t)1, pCparams.vNSeqMax),
        /* n_seq_max         */ pCparams.vNSeqMax,
        /* n_rs_seq          */ pCparams.vNRsSeq,
        /* offload           */ pCparams.vOffloadKqv,
        /* unified           */ pCparams.vKvUnified,
        /* filter_attn       */ std::move(vFilterAttn),
        /* filter_recr       */ std::move(vFilterRecr),
        /* filter_idx        */ std::move(vFilterIdx));
  } break;
  case cLlmArchDeepseek41: {
    GGML_ASSERT(vHparams.vSwaType != cMgsSwaTypeNone);

    vRes = new mgs_kv_cache_dsv41(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                  pParams.vSwaFull, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax,
                                  pCparams.vNUbatch, 1, pCparams.vNRsSeq, nullptr);
  } break;
  // Models that need standard caching should rely on recurrent/hybrid
  // checks
  default: {
    // The MTP head is dense-attention only on hybrid Qwen3-Next/3.5/3.6, so use a plain
    // attention KV cache for the MTP context instead of the hybrid wrapper.
    const bool cMtpOnHybridQwen = pParams.vCtxType == MGS_CONTEXT_TYPE_MTP &&
                                  (vArch == cLlmArchQwen3next || vArch == cLlmArchQwen35 || vArch == cLlmArchQwen35moe);

    if (fLlmArchIsRecurrent(vArch)) {
      vRes = new mgs_memory_recurrent(*this, GGML_TYPE_F32, GGML_TYPE_F32, pCparams.vOffloadKqv,
                                      std::max((uint32_t)1, pCparams.vNSeqMax), pCparams.vNSeqMax, pCparams.vNRsSeq,
                                      nullptr);
    } else if (fLlmArchIsHybrid(vArch) && !cMtpOnHybridQwen) {
      // The main difference between hybrid architectures is the
      // layer filters, so pick the right one here
      mgs_memory_hybrid::layer_filter_cb vFilterAttn = nullptr;
      mgs_memory_hybrid::layer_filter_cb vFilterRecr = nullptr;
      // qwen4exp keeps its QSA indexer keys in a third cache: a null vFilterIdx means the GGUF has no indexer
      mgs_memory_hybrid::layer_filter_cb vFilterIdx = nullptr;
      const bool cNeedsMemIdx = (vArch == cLlmArchQwen4exp);
      if (vArch == cLlmArchFalconH1) {
        vFilterAttn = [&](uint32_t) { return true; };
        vFilterRecr = [&](uint32_t) { return true; };
      } else if (vArch == cLlmArchNemotronH || vArch == cLlmArchNemotronHMoe) {
        vFilterAttn = [&](uint32_t pIl) { return !vHparams.mIsRecr(pIl) && vHparams.mNFf(pIl) == 0; };
        vFilterRecr = [&](uint32_t pIl) { return vHparams.mIsRecr(pIl) && vHparams.mNFf(pIl) == 0; };
      } else if (vArch == cLlmArchQwen3next || vArch == cLlmArchQwen35 || vArch == cLlmArchQwen35moe) {
        vFilterAttn = [&](uint32_t pIl) { return pIl < vHparams.mNLayer() && !vHparams.mIsRecr(pIl); };
        vFilterRecr = [&](uint32_t pIl) { return pIl < vHparams.mNLayer() && vHparams.mIsRecr(pIl); };
      } else if (vArch == cLlmArchQwen4exp) {
        vFilterAttn = [&](uint32_t pIl) { return pIl < vHparams.mNLayer() && !vHparams.mIsRecr(pIl); };
        vFilterRecr = [&](uint32_t pIl) { return pIl < vHparams.mNLayer() && vHparams.mIsRecr(pIl); };

        if (vHparams.vIndexerHeadSize > 0) {
          // QSA runs on the dense-attention layers only
          vFilterIdx = [&](uint32_t pIl) { return pIl < vHparams.mNLayer() && !vHparams.mIsRecr(pIl); };
        }

        // the MTP draft context holds the MTP block alone: its attention and indexer, no recurrent layer
        if (pParams.vCtxType == MGS_CONTEXT_TYPE_MTP) {
          vFilterAttn = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
          vFilterRecr = [&](uint32_t) { return false; };
          if (vFilterIdx) {
            vFilterIdx = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
          }
        }
      }

      if (cNeedsMemIdx) {
        // sparse attention over a per-token indexer cache, in its own memory type
        vRes = new mgs_memory_hybrid_idx(
            /* model             */ *this,
            /* attn_type_k       */ pParams.vTypeK,
            /* attn_type_v       */ pParams.vTypeV,
            /* attn_v_trans      */ !pCparams.vFlashAttn,
            /* attn_kv_size      */ pCparams.vNCtxSeq,
            /* attn_n_pad        */ 1,
            /* attn_n_swa        */ vHparams.vNSwa,
            /* attn_swa_type     */ vHparams.vSwaType,
            /* recurrent_type_k  */ GGML_TYPE_F32,
            /* recurrent_type_v  */ GGML_TYPE_F32,
            /* recurrent_kv_size */ std::max((uint32_t)1, pCparams.vNSeqMax),
            /* n_seq_max         */ pCparams.vNSeqMax,
            /* n_rs_seq          */ pCparams.vNRsSeq,
            /* offload           */ pCparams.vOffloadKqv,
            /* unified           */ pCparams.vKvUnified,
            /* filter_attn       */ std::move(vFilterAttn),
            /* filter_recr       */ std::move(vFilterRecr),
            /* filter_idx        */ std::move(vFilterIdx));
      } else if (vHparams.vSwaType != cMgsSwaTypeNone) {
        // Use hybrid-iswa for hybrid models with SWA
        vRes = new mgs_memory_hybrid_iswa(
            /* model             */ *this,
            /* attn_type_k       */ pParams.vTypeK,
            /* attn_type_v       */ pParams.vTypeV,
            /* attn_v_trans      */ !pCparams.vFlashAttn,
            /* attn_swa_full     */ pParams.vSwaFull,
            /* attn_kv_size      */ pCparams.vNCtxSeq,
            /* attn_n_ubatch     */ pCparams.vNUbatch,
            /* attn_n_pad        */ 1,
            /* recurrent_type_r  */ GGML_TYPE_F32,
            /* recurrent_type_s  */ GGML_TYPE_F32,
            /* recurrent_rs_size */ std::max((uint32_t)1, pCparams.vNSeqMax),
            /* n_seq_max         */ pCparams.vNSeqMax,
            /* n_rs_seq          */ pCparams.vNRsSeq,
            /* offload           */ pCparams.vOffloadKqv,
            /* unified           */ pCparams.vKvUnified,
            /* filter_attn       */ std::move(vFilterAttn),
            /* filter_recr       */ std::move(vFilterRecr));
      } else {
        vRes = new mgs_memory_hybrid(
            /* model             */ *this,
            /* attn_type_k       */ pParams.vTypeK,
            /* attn_type_v       */ pParams.vTypeV,
            /* attn_v_trans      */ !pCparams.vFlashAttn,
            /* attn_kv_size      */ pCparams.vNCtxSeq,
            /* attn_n_pad        */ 1,
            /* attn_n_swa        */ vHparams.vNSwa,
            /* attn_swa_type     */ vHparams.vSwaType,
            /* recurrent_type_k  */ GGML_TYPE_F32,
            /* recurrent_type_v  */ GGML_TYPE_F32,
            /* recurrent_kv_size */ std::max((uint32_t)1, pCparams.vNSeqMax),
            /* n_seq_max         */ pCparams.vNSeqMax,
            /* n_rs_seq          */ pCparams.vNRsSeq,
            /* offload           */ pCparams.vOffloadKqv,
            /* unified           */ pCparams.vKvUnified,
            /* filter_attn       */ std::move(vFilterAttn),
            /* filter_recr       */ std::move(vFilterRecr));
      }
    } else {
      mgs_kv_cache::layer_filter_cb vFilter = nullptr;

      if (cMtpOnHybridQwen) {
        vFilter = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
      }

      // don't filter when a model is entirely n_layer_nextn layers and has no trunk
      if (vHparams.vNLayerNextn > 0 && vHparams.mNLayer() > 0) {
        if (pParams.vCtxType == MGS_CONTEXT_TYPE_MTP) {
          vFilter = [&](uint32_t pIl) { return pIl >= vHparams.mNLayer(); };
        } else {
          vFilter = [&](uint32_t pIl) { return pIl < vHparams.mNLayer(); };
        }
      }

      if (vHparams.vSwaType != cMgsSwaTypeNone) {
        GGML_ASSERT(vHparams.mIsSwaAny());

        vRes = new mgs_kv_cache_iswa(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                     pParams.vSwaFull, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax,
                                     pCparams.vNUbatch, 1, nullptr, vFilter, nullptr, nullptr);
      } else {
        GGML_ASSERT(!vHparams.mIsSwaAny());

        vRes = new mgs_kv_cache(*this, vHparams, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn,
                                pCparams.vOffloadKqv, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax, 1,
                                vHparams.vNSwa, vHparams.vSwaType, nullptr, vFilter, nullptr, nullptr);
      }
    }
  }
  }

  return vRes;
}

ggml_cgraph *mgs_model::mBuildGraph(const llm_graph_params &pParams) const {
  std::unique_ptr<llm_graph_context> vLlm = mBuildArchGraph(pParams);

  // add on pooling layer
  vLlm->mBuildPooling(vCls, vClsB, vClsOut, vClsOutB, vClsNorm);

  // add backend sampling layers (if any)
  vLlm->mBuildSampling();

  // if the gguf model was converted with --sentence-transformers-dense-modules
  // there will be two additional dense projection layers
  // dense linear projections are applied after pooling
  // TODO: move reranking logic here and generalize
  vLlm->mBuildDenseOut(vDense2OutLayers, vDense2OutLayersB, vDense3OutLayers);

  vLlm->vRes->mSetOutputs(pParams);

  return vLlm->vRes->mGetGf();
}

//
// interface implementation
//

mgs_model_params mgs_model_default_params() {
  mgs_model_params vResult = {
      /*.devices                     =*/nullptr,
      /*.tensor_buft_overrides       =*/nullptr,
      /*.n_gpu_layers                =*/-1,
      /*.split_mode                  =*/MGS_SPLIT_MODE_LAYER,
      /*.load_mode                   =*/MGS_LOAD_MODE_MMAP,
      /*.main_gpu                    =*/0,
      /*.tensor_split                =*/nullptr,
      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr,
      /*.kv_overrides                =*/nullptr,
      /*.vocab_only                  =*/false,
      /*.check_tensors               =*/false,
      /*.use_extra_bufts             =*/true,
      /*.no_host                     =*/false,
      /*.no_alloc                    =*/false,
      /*.load_mtp                    =*/false,
  };

  return vResult;
}

const mgs_vocab *mgs_model_get_vocab(const mgs_model *pModel) { return &pModel->vVocab; }

void mgs_model_free(mgs_model *pModel) { delete pModel; }

int32_t mgs_model_n_ctx_train(const mgs_model *pModel) { return pModel->vHparams.vNCtxTrain; }

int32_t mgs_model_n_embd(const mgs_model *pModel) { return pModel->vHparams.vNEmbd; }

int32_t mgs_model_n_embd_inp(const mgs_model *pModel) { return pModel->vHparams.mNEmbdInp(); }

int32_t mgs_model_n_embd_out(const mgs_model *pModel) { return pModel->vHparams.mNEmbdOut(); }

int32_t mgs_model_n_layer(const mgs_model *pModel) { return pModel->vHparams.mNLayer(); }

int32_t mgs_model_n_layer_nextn(const mgs_model *pModel) { return pModel->vHparams.vNLayerNextn; }

int32_t mgs_model_n_head(const mgs_model *pModel) { return pModel->vHparams.mNHead(); }

int32_t mgs_model_n_head_kv(const mgs_model *pModel) { return pModel->vHparams.mNHeadKv(); }

int32_t mgs_model_n_swa(const mgs_model *pModel) {
  // the DeepSeek-V4.1 cache has SWA, but its compressed rows cannot be rolled back
  // within the window, so we return 0 here
  if (pModel->vArch == cLlmArchDeepseek41) {
    return 0;
  }
  return pModel->vHparams.vNSwa;
}

uint32_t mgs_model_n_cls_out(const struct mgs_model *pModel) { return pModel->vHparams.vNClsOut; }

const char *mgs_model_cls_label(const struct mgs_model *pModel, uint32_t pI) {
  if (pI < pModel->lClassifierLabels.size()) {
    return pModel->lClassifierLabels[pI].c_str();
  }

  return nullptr;
}

mgs_rope_type mgs_model_rope_type(const mgs_model *pModel) {
  switch (pModel->vArch) {
  // these models do not use RoPE
  case cLlmArchClip:
  case cLlmArchGpt2:
  case cLlmArchGptj:
  case cLlmArchMpt:
  case cLlmArchRefact:
  case cLlmArchBloom:
  case cLlmArchMamba:
  case cLlmArchMamba2:
  case cLlmArchJamba:
  case cLlmArchJinaBertV2:
  case cLlmArchT5:
  case cLlmArchT5encoder:
  case cLlmArchJais:
  case cLlmArchRwkv6:
  case cLlmArchRwkv6qwen2:
  case cLlmArchRwkv7:
  case cLlmArchArwkv7:
  case cLlmArchWavtokenizerDec:
  case cLlmArchNemotronH:
  case cLlmArchNemotronHMoe:
  case cLlmArchKimiLinear:
  case cLlmArchKimiK3:
  case cLlmArchGlm5Next:
    return MGS_ROPE_TYPE_NONE;

  // use what we call a normal RoPE, operating on pairs of consecutive head values
  case cLlmArchLlada:
  case cLlmArchDeci:
  case cLlmArchBaichuan:
  case cLlmArchStarcoder:
  case cLlmArchInternlm2:
  case cLlmArchMinicpm:
  case cLlmArchXverse:
  case cLlmArchCommandR:
  case cLlmArchCohere2:
  case cLlmArchCohere2moe:
  case cLlmArchOlmo:
  case cLlmArchArctic:
  case cLlmArchDeepseek41:
  case cLlmArchPlm:
  case cLlmArchChatglm:
  case cLlmArchGranite:
  case cLlmArchGraniteMoe:
  case cLlmArchGraniteHybrid:
  case cLlmArchChameleon:
  case cLlmArchBailingmoe:
  case cLlmArchNeoBert:
  case cLlmArchSmollm3:
  case cLlmArchArcee:
  case cLlmArchERNIE4Point5:
  case cLlmArchERNIE4Point5Moe:
  case cLlmArchMistral3:
  case cLlmArchEagle3:
  case cLlmArchMaincoder:
  case cLlmArchGlmDsa:
  case cLlmArchNanbeige:
    return MGS_ROPE_TYPE_NORM;

  // the pairs of head values are offset by n_rot/2
  case cLlmArchFalcon:
  case cLlmArchFalconH1:
  case cLlmArchGrok:
  case cLlmArchDbrx:
  case cLlmArchBert:
  case cLlmArchJinaBertV3:
  case cLlmArchModernBert:
  case cLlmArchNomicBert:
  case cLlmArchNomicBertMoe:
  case cLlmArchEurobert:
  case cLlmArchStablelm:
  case cLlmArchBitnet:
  case cLlmArchQwen:
  case cLlmArchQwen2:
  case cLlmArchDream:
  case cLlmArchQwen2moe:
  case cLlmArchQwen3:
  case cLlmArchQwen3moe:
  case cLlmArchLladaMoe:
  case cLlmArchRnd1:
  case cLlmArchOlmo2:
  case cLlmArchOlmoe:
  case cLlmArchPhi2:
  case cLlmArchPhi3:
  case cLlmArchPhimoe:
  case cLlmArchPlamo:
  case cLlmArchPlamo2:
  case cLlmArchPlamo3:
  case cLlmArchGemma:
  case cLlmArchGemma2:
  case cLlmArchGemma3:
  case cLlmArchGemma3n:
  case cLlmArchGemmaEmbedding:
  case cLlmArchStarcoder2:
  case cLlmArchOpenelm:
  case cLlmArchGptneox:
  case cLlmArchCodeshell:
  case cLlmArchOrion:
  case cLlmArchNemotron:
  case cLlmArchExaone:
  case cLlmArchExaone4:
  case cLlmArchExaoneMoe:
  case cLlmArchMinicpm3:
  case cLlmArchBailingmoe2:
  case cLlmArchDots1:
  case cLlmArchHunyuanMoe:
  case cLlmArchJais2:
  case cLlmArchOpenaiMoe:
  case cLlmArchHunyuanDense:
  case cLlmArchHyV3:
  case cLlmArchLfm2:
  case cLlmArchLfm2moe:
  case cLlmArchSmallthinker:
  case cLlmArchSeedOss:
  case cLlmArchGrovemoe:
  case cLlmArchApertus:
  case cLlmArchMinimaxM2:
  case cLlmArchMinimaxM3:
  case cLlmArchCogvlm:
  case cLlmArchPanguEmbed:
  case cLlmArchAfmoe:
  case cLlmArchLaguna:
  case cLlmArchQwen3next:
  case cLlmArchMimo2:
  case cLlmArchStep35:
  case cLlmArchTalkie:
  case cLlmArchMellum:
    return MGS_ROPE_TYPE_NEOX;

  case cLlmArchDflash:
    // the DFlash drafters use Qwen3-style backbones
    return MGS_ROPE_TYPE_NEOX;

  case cLlmArchQwen2vl:
  case cLlmArchPaddleocr:
    return MGS_ROPE_TYPE_MROPE;
  case cLlmArchQwen3vl:
  case cLlmArchQwen3vlmoe:
  case cLlmArchQwen35:
  case cLlmArchQwen35moe:
  case cLlmArchQwen4exp:
    return MGS_ROPE_TYPE_IMROPE;

  case cLlmArchGlm4:
    return pModel->vHparams.mUseMrope() ? MGS_ROPE_TYPE_MROPE : MGS_ROPE_TYPE_NORM;
  case cLlmArchGlm4Moe:
    return pModel->vHparams.mUseMrope() ? MGS_ROPE_TYPE_MROPE : MGS_ROPE_TYPE_NEOX;

  case cLlmArchHunyuanVl:
    return pModel->vHparams.mUseMrope() ? MGS_ROPE_TYPE_MROPE : MGS_ROPE_TYPE_NEOX;

  // all model arches should be listed explicitly here
  case cLlmArchUnknown:
    GGML_ABORT("unknown architecture");
  }

  return MGS_ROPE_TYPE_NONE;
}

float mgs_model_rope_freq_scale_train(const mgs_model *pModel) { return pModel->vHparams.vRopeFreqScaleTrain; }

int32_t mgs_model_meta_val_str(const mgs_model *pModel, const char *pKey, char *pBuf, size_t pBufSize) {
  const auto &vIt = pModel->dGgufKv.find(pKey);
  if (vIt == pModel->dGgufKv.end()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

int32_t mgs_model_meta_count(const mgs_model *pModel) { return (int)pModel->dGgufKv.size(); }

const char *mgs_model_meta_key_str(mgs_model_meta_key pKey) {
  switch (pKey) {
  case MGS_MODEL_META_KEY_SAMPLING_SEQUENCE:
    return "general.sampling.sequence";
  case MGS_MODEL_META_KEY_SAMPLING_TOP_K:
    return "general.sampling.top_k";
  case MGS_MODEL_META_KEY_SAMPLING_TOP_P:
    return "general.sampling.top_p";
  case MGS_MODEL_META_KEY_SAMPLING_MIN_P:
    return "general.sampling.min_p";
  case MGS_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY:
    return "general.sampling.xtc_probability";
  case MGS_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:
    return "general.sampling.xtc_threshold";
  case MGS_MODEL_META_KEY_SAMPLING_TEMP:
    return "general.sampling.temp";
  case MGS_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:
    return "general.sampling.penalty_last_n";
  case MGS_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:
    return "general.sampling.penalty_repeat";
  case MGS_MODEL_META_KEY_SAMPLING_MIROSTAT:
    return "general.sampling.mirostat";
  case MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:
    return "general.sampling.mirostat_tau";
  case MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:
    return "general.sampling.mirostat_eta";
  default:
    return nullptr;
  }
}

int32_t mgs_model_meta_key_by_index(const mgs_model *pModel, int pI, char *pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int)pModel->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pModel->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->first.c_str());
}

int32_t mgs_model_meta_val_str_by_index(const mgs_model *pModel, int32_t pI, char *pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int)pModel->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pModel->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

int32_t mgs_model_desc(const mgs_model *pModel, char *pBuf, size_t pBufSize) {
  return snprintf(pBuf, pBufSize, "%s", pModel->mDesc().c_str());
}

mgs_ftype mgs_model_ftype(const mgs_model *pModel) { return pModel->mFtype(); }

uint64_t mgs_model_size(const mgs_model *pModel) { return pModel->size(); }

const char *mgs_model_chat_template(const mgs_model *pModel, const char *pName) {
  const auto cKey = pName ? LLM_KV(pModel->vArch, pName)(cLlmKvTokenizerChatTemplate)
                          : LLM_KV(pModel->vArch)(cLlmKvTokenizerChatTemplate);
  const auto &vIt = pModel->dGgufKv.find(cKey);
  if (vIt == pModel->dGgufKv.end()) {
    return nullptr;
  }

  return vIt->second.c_str();
}

uint64_t mgs_model_n_params(const mgs_model *pModel) { return pModel->mNElements(); }

bool mgs_model_has_encoder(const mgs_model *pModel) {
  switch (pModel->vArch) {
  case cLlmArchT5:
  case cLlmArchT5encoder:
  case cLlmArchEagle3:
  case cLlmArchDflash:
    return true;
  default:
    return false;
  }
}

bool mgs_model_has_decoder(const mgs_model *pModel) {
  switch (pModel->vArch) {
  case cLlmArchT5encoder:
    return false;
  default:
    return true;
  }
}

mgs_token mgs_model_decoder_start_token(const mgs_model *pModel) { return pModel->vHparams.vDecStartTokenId; }

bool mgs_model_is_recurrent(const mgs_model *pModel) { return fLlmArchIsRecurrent(pModel->vArch); }

bool mgs_model_is_hybrid(const mgs_model *pModel) { return fLlmArchIsHybrid(pModel->vArch); }

bool mgs_model_is_diffusion(const mgs_model *pModel) { return fLlmArchIsDiffusion(pModel->vArch); }

const std::vector<std::pair<std::string, ggml_tensor *>> &mgs_internal_get_tensor_map(const mgs_model *pModel) {
  return pModel->lTensorsByName;
}

int32_t mgs_model_n_expert(const struct mgs_model *pModel) { return pModel->vHparams.vNExpert; }

int32_t mgs_model_n_devices(const struct mgs_model *pModel) { return (int32_t)pModel->lDevices.size(); }

ggml_backend_dev_t mgs_model_get_device(const struct mgs_model *pModel, int pI) {
  if (pI < 0 || pI >= (int)pModel->lDevices.size()) {
    return nullptr;
  }
  return pModel->lDevices[pI].vDev;
}

mgs_model_weight_breakdown mgs_model_get_weight_breakdown(const struct mgs_model *pModel) {
  mgs_model_weight_breakdown dBreakdown;
  // the nextn layers follow the trunk; without them no layer reaches n_layer()
  const int vMtpFirstLayer = (int)pModel->vHparams.mNLayer();
  for (const auto &[vName, vTensor] : pModel->lTensorsByName) {
    if (vTensor == nullptr || vTensor->buffer == nullptr) {
      continue; // not loaded, e.g. MTP layers replaced by a separate MTP GGUF
    }
    int vLayer = -1;
    const bool vMtp = sscanf(vName.c_str(), "blk.%d.", &vLayer) == 1 && vLayer >= vMtpFirstLayer;
    const bool vExperts = !vMtp && vName.find("exps.") != std::string::npos;
    if (!vMtp && !vExperts) {
      continue;
    }
    const size_t vBytes = ggml_backend_buffer_get_alloc_size(vTensor->buffer, vTensor);
    mgs_model_weight_split &vSplit = dBreakdown[ggml_backend_buffer_get_type(vTensor->buffer)];
    if (vMtp) {
      vSplit.mtp += vBytes;
    } else {
      vSplit.experts += vBytes;
    }
  }
  return dBreakdown;
}

//
// mgs_model_base
//

mgs_model_base::mgs_model_base(const struct mgs_model_params &pParams)
    : mgs_model(pParams), vModel(this), cTn(vModel->vArch), cTensorDuplicated(mgs_model_loader::cTensorDuplicated),
      cTensorNotRequired(mgs_model_loader::cTensorNotRequired), cTensorSkip(mgs_model_loader::cTensorSkip),
      cTensorSkipIfVirtual(mgs_model_loader::cTensorSkipIfVirtual),
      cTensorAllowReshape(mgs_model_loader::cTensorAllowReshape) {}

ggml_tensor *mgs_model_base::mCreateTensor(const LLM_TN_IMPL &pTn, const std::initializer_list<int64_t> &pNe,
                                           int pFlags) {
  GGML_ASSERT(vMl != nullptr);
  return mCreateTensor(*vMl, pTn, pNe, pFlags);
}

void mgs_model_base::mCreateTensorGateUpExps(mgs_layer &pLayer, int pBid, int64_t pNEmbd, int64_t pNFf,
                                             int64_t pNExpert, int pFlags) {
  if (pFlags & cTensorSkip) {
    const int cSkip = cTensorNotRequired | cTensorSkip;

    mCreateTensor(cTn(cLlmTensorFfnGateUpExps, "weight", pBid), {pNEmbd, pNFf * 2, pNExpert},
                  cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pBid), {pNEmbd, pNFf, pNExpert}, cSkip);
    mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pBid), {pNEmbd, pNFf, pNExpert}, cSkip);
    return;
  }

  pLayer.vFfnGateUpExps =
      mCreateTensor(cTn(cLlmTensorFfnGateUpExps, "weight", pBid), {pNEmbd, pNFf * 2, pNExpert}, cTensorNotRequired);
  if (pLayer.vFfnGateUpExps == nullptr) {
    pLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pBid), {pNEmbd, pNFf, pNExpert}, pFlags);
    pLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pBid), {pNEmbd, pNFf, pNExpert}, pFlags);
  }
}

void mgs_model_base::mCreateTensorQkv(mgs_layer &pLayer, int pBid, int64_t pNEmbd, int64_t pNEmbdQ, int64_t pNEmbdK,
                                      int64_t pNEmbdV, int pFlags) {
  const int64_t cNEmbdQkv = pNEmbdQ + pNEmbdK + pNEmbdV;

  if (pFlags & cTensorSkip) {
    const int cSkip = cTensorNotRequired | cTensorSkip;

    mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pBid), {pNEmbd, cNEmbdQkv}, cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorAttnQkv, "bias", pBid), {cNEmbdQkv}, cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorAttnQ, "weight", pBid), {pNEmbd, pNEmbdQ}, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnK, "weight", pBid), {pNEmbd, pNEmbdK}, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnV, "weight", pBid), {pNEmbd, pNEmbdV}, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnQ, "bias", pBid), {pNEmbdQ}, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnK, "bias", pBid), {pNEmbdK}, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnV, "bias", pBid), {pNEmbdV}, cSkip);
    return;
  }

  pLayer.vWqkv = mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pBid), {pNEmbd, cNEmbdQkv},
                               cTensorNotRequired | cTensorSkipIfVirtual);
  if (pLayer.vWqkv) {
    pLayer.vWqkvB =
        mCreateTensor(cTn(cLlmTensorAttnQkv, "bias", pBid), {cNEmbdQkv}, cTensorNotRequired | cTensorSkipIfVirtual);
  } else {
    pLayer.vWq = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", pBid), {pNEmbd, pNEmbdQ}, pFlags);
    pLayer.vWk = mCreateTensor(cTn(cLlmTensorAttnK, "weight", pBid), {pNEmbd, pNEmbdK}, pFlags);
    pLayer.vWv = mCreateTensor(cTn(cLlmTensorAttnV, "weight", pBid), {pNEmbd, pNEmbdV}, pFlags);
    pLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQ, "bias", pBid), {pNEmbdQ}, cTensorNotRequired);
    pLayer.vWkB = mCreateTensor(cTn(cLlmTensorAttnK, "bias", pBid), {pNEmbdK}, cTensorNotRequired);
    pLayer.vWvB = mCreateTensor(cTn(cLlmTensorAttnV, "bias", pBid), {pNEmbdV}, cTensorNotRequired);
  }
}

const int32_t *mgs_model_target_layer_ids(const struct mgs_model *pModel) {
  const auto &vV = pModel->lTargetLayerIds;
  return vV.empty() ? nullptr : vV.data();
}

uint32_t mgs_model_target_layer_ids_n(const struct mgs_model *pModel) {
  return (uint32_t)pModel->lTargetLayerIds.size();
}
