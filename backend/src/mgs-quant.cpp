#include "mgs-impl.h"
#include "mgs-model.h"
#include "mgs-model-loader.h"
#include "mgs-ext.h"
#include "mgs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>

// result of parsing --tensor-type option
struct tensor_type_option {
  std::string vName;
  ggml_type vType = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
  cTokenEmbd,
  cAttentionQ,
  cAttentionV,
  cAttentionK,
  cAttentionQkv,
  cAttentionOutput,
  cFfnUp,
  cFfnGate,
  cFfnDown,
  cOutput,
  cOther
};

static void fZeros(std::ofstream &pFile, size_t pN) {
  char vZero = 0;
  for (size_t vI = 0; vI < pN; ++vI) {
    pFile.write(&vZero, 1);
  }
}

static std::string fRemapLayer(const std::string &pOrigName, const std::vector<int> &pPrune,
                               std::map<int, std::string> &pMapped, int &pNextId) {
  if (pPrune.empty()) {
    return pOrigName;
  }

  static const std::regex cPattern(R"(blk\.(\d+)\.)");
  if (std::smatch vMatch; std::regex_search(pOrigName, vMatch, cPattern)) {
    const int cBlk = std::stoi(vMatch[1]);
    std::string vNewName = pOrigName;

    if (pMapped.count(cBlk)) {
      // Already mapped, do nothing
    } else if (std::find(pPrune.begin(), pPrune.end(), cBlk) != pPrune.end()) {
      pMapped[cBlk] = "";
    } else if (cBlk < pPrune.front()) {
      pMapped[cBlk] = std::to_string(cBlk);
      pNextId = cBlk + 1;
    } else {
      pMapped[cBlk] = std::to_string(pNextId);
      ++pNextId;
    }

    return pMapped[cBlk].empty() ? pMapped[cBlk]
                                 : vNewName.replace(vMatch.position(1), vMatch.length(1), pMapped[cBlk]);
  }

  return pOrigName;
}

static std::string fRemapImatrix(const std::string &pOrigName, const std::map<int, std::string> &pMapped) {
  if (pMapped.empty()) {
    return pOrigName;
  }

  static const std::regex cPattern(R"(blk\.(\d+)\.)");
  if (std::smatch vMatch; std::regex_search(pOrigName, vMatch, cPattern)) {
    const std::string cBlk(vMatch[1]);
    std::string vNewName = pOrigName;

    for (const auto &vP : pMapped) {
      if (vP.second == cBlk) {
        return vNewName.replace(vMatch.position(1), vMatch.length(1), std::to_string(vP.first));
      }
    }
    GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, pOrigName.c_str());
  }

  return pOrigName;
}

//
// helper functions for tensor name matching
//

static bool fTensorNameMatchTokenEmbd(const char *pTensorName) {
  return std::strcmp(pTensorName, "token_embd.weight") == 0 ||
         std::strcmp(pTensorName, "per_layer_token_embd.weight") == 0;
}

static bool fTensorNameMatchOutputWeight(const char *pTensorName) {
  return std::strcmp(pTensorName, "output.weight") == 0;
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category fTensorGetCategory(const std::string &pTensorName) {
  if (fTensorNameMatchOutputWeight(pTensorName.c_str())) {
    return tensor_category::cOutput;
  }
  if (fTensorNameMatchTokenEmbd(pTensorName.c_str())) {
    return tensor_category::cTokenEmbd;
  }
  if (pTensorName.find("attn_qkv.weight") != std::string::npos) {
    return tensor_category::cAttentionQkv;
  }
  if (pTensorName.find("attn_v.weight") != std::string::npos) {
    return tensor_category::cAttentionV;
  }
  if (pTensorName.find("attn_k.weight") != std::string::npos) {
    return tensor_category::cAttentionK;
  }
  if (pTensorName.find("attn_q.weight") != std::string::npos) {
    return tensor_category::cAttentionQ;
  }
  if (pTensorName.find("attn_output.weight") != std::string::npos) {
    return tensor_category::cAttentionOutput;
  }
  if (pTensorName.find("ffn_up") != std::string::npos) {
    return tensor_category::cFfnUp;
  }
  if (pTensorName.find("ffn_gate") != std::string::npos) {
    return tensor_category::cFfnGate;
  }
  if (pTensorName.find("ffn_down") != std::string::npos) {
    return tensor_category::cFfnDown;
  }
  return tensor_category::cOther;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool fCategoryIsAttnV(tensor_category pCat) {
  return pCat == tensor_category::cAttentionV || pCat == tensor_category::cAttentionQkv;
}

//
// quantization state
//

struct quantize_state_impl {
  const mgs_model &vModel;
  const mgs_model_quantize_params *vParams;

  int vNAttentionWv = 0;
  int vNFfnDown = 0;
  int vNFfnGate = 0;
  int vNFfnUp = 0;
  int vIAttentionWv = 0;
  int vIFfnDown = 0;
  int vIFfnGate = 0;
  int vIFfnUp = 0;

  int vNFallback = 0;

  bool vHasImatrix = false;

  // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
  bool vHasTiedEmbeddings = true; // assume tied until we see output.weight

  // tensor type override patterns (compiled once, used twice)
  std::vector<std::pair<std::regex, ggml_type>> lTensorTypePatterns;

  quantize_state_impl(const mgs_model &pModel, const mgs_model_quantize_params *pParams)
      : vModel(pModel), vParams(pParams) {
    // compile regex patterns once - they are expensive
    if (pParams->tt_overrides) {
      for (const auto *vP = pParams->tt_overrides; vP->pattern != nullptr; vP++) {
        lTensorTypePatterns.emplace_back(std::regex(vP->pattern), vP->type);
      }
    }
  }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
  std::string vName;
  ggml_type vTargetType;
  tensor_category vCategory;
  std::string vRemappedImatrixName;
  bool vAllowsQuantization;
  bool vRequiresImatrix;
};

//
// dequantization
//

static void fMgsTensorDequantizeImpl(ggml_tensor *pTensor, std::vector<no_init<float>> &pOutput,
                                     std::vector<std::thread> &pWorkers, const size_t pNelements, const int pNthread) {
  if (pOutput.size() < pNelements) {
    pOutput.resize(pNelements);
  }
  float *vF32Output = (float *)pOutput.data();

  const ggml_type_traits *vQtype = ggml_get_type_traits(pTensor->type);
  if (ggml_is_quantized(pTensor->type)) {
    if (vQtype->to_float == NULL) {
      throw std::runtime_error(fFormat("type %s unsupported for integer quantization: no dequantization available",
                                       ggml_type_name(pTensor->type)));
    }
  } else if (pTensor->type != GGML_TYPE_F16 && pTensor->type != GGML_TYPE_BF16) {
    throw std::runtime_error(fFormat("cannot dequantize/convert tensor type %s", ggml_type_name(pTensor->type)));
  }

  if (pNthread < 2) {
    if (pTensor->type == GGML_TYPE_F16) {
      ggml_fp16_to_fp32_row((ggml_fp16_t *)pTensor->data, vF32Output, pNelements);
    } else if (pTensor->type == GGML_TYPE_BF16) {
      ggml_bf16_to_fp32_row((ggml_bf16_t *)pTensor->data, vF32Output, pNelements);
    } else if (ggml_is_quantized(pTensor->type)) {
      vQtype->to_float(pTensor->data, vF32Output, pNelements);
    } else {
      GGML_ABORT("fatal error"); // unreachable
    }
    return;
  }

  size_t vBlockSize;
  if (pTensor->type == GGML_TYPE_F16 || pTensor->type == GGML_TYPE_BF16) {
    vBlockSize = 1;
  } else {
    vBlockSize = (size_t)ggml_blck_size(pTensor->type);
  }

  size_t vBlockSizeBytes = ggml_type_size(pTensor->type);

  GGML_ASSERT(pNelements % vBlockSize == 0);
  size_t vNblocks = pNelements / vBlockSize;
  size_t vBlocksPerThread = vNblocks / pNthread;
  size_t vSpareBlocks = vNblocks - (vBlocksPerThread * pNthread); // if blocks aren't divisible by thread count

  size_t vInBuffOffs = 0;
  size_t vOutBuffOffs = 0;

  for (int vTnum = 0; vTnum < pNthread; vTnum++) {
    size_t vThrBlocks = vBlocksPerThread + (vTnum == pNthread - 1 ? vSpareBlocks : 0); // num blocks for this thread
    size_t vThrElems = vThrBlocks * vBlockSize;           // number of elements for this thread
    size_t vThrBlockBytes = vThrBlocks * vBlockSizeBytes; // number of input bytes for this thread

    auto vCompute = [vQtype](ggml_type pTyp, uint8_t *pInbuf, float *pOutbuf, int pNels) {
      if (pTyp == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((ggml_fp16_t *)pInbuf, pOutbuf, pNels);
      } else if (pTyp == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((ggml_bf16_t *)pInbuf, pOutbuf, pNels);
      } else {
        vQtype->to_float(pInbuf, pOutbuf, pNels);
      }
    };
    pWorkers.emplace_back(vCompute, pTensor->type, (uint8_t *)pTensor->data + vInBuffOffs, vF32Output + vOutBuffOffs,
                          vThrElems);
    vInBuffOffs += vThrBlockBytes;
    vOutBuffOffs += vThrElems;
  }
  for (auto &vW : pWorkers) {
    vW.join();
  }
  pWorkers.clear();
}

//
// do we allow this tensor to be quantized?
//

static bool fTensorAllowsQuantization(const mgs_model_quantize_params *pParams, llm_arch pArch,
                                      const ggml_tensor *pTensor) {
  // trivial checks first -- no string ops needed
  if (pParams->only_copy)
    return false;

  // quantize only 2D and 3D tensors (experts)
  if (ggml_n_dims(pTensor) < 2)
    return false;

  const std::string cName = ggml_get_name(pTensor);

  // This used to be a regex, but <regex> has an extreme cost to compile times.
  bool vQuantize = cName.rfind("weight") == cName.size() - 6; // ends with 'weight'?

  // do not quantize norm tensors
  vQuantize &= cName.find("_norm.weight") == std::string::npos;

  vQuantize &= pParams->quantize_output_tensor || cName != "output.weight";

  // do not quantize expert gating tensors
  // NOTE: can't use LLM_TN here because the layer number is not known
  vQuantize &= cName.find("ffn_gate_inp.weight") == std::string::npos;

  // DeepSeek-V4.1: the engram tables are read from the file as they are and the engram gate weights are tiny
  vQuantize &= cName.find("engram_embd.weight") == std::string::npos;
  vQuantize &= cName.find("engram_q.weight") == std::string::npos;
  vQuantize &= cName.find("engram_k.weight") == std::string::npos;
  vQuantize &= cName.find("engram_q_norm.weight") == std::string::npos;
  vQuantize &= cName.find("engram_k_norm.weight") == std::string::npos;

  // these are very small (e.g. 4x4)
  vQuantize &= cName.find("altup") == std::string::npos;
  vQuantize &= cName.find("laurel") == std::string::npos;

  // do not quantize positional embeddings and token types (BERT)
  vQuantize &= cName != LLM_TN(pArch)(cLlmTensorPosEmbd, "weight");
  vQuantize &= cName != LLM_TN(pArch)(cLlmTensorTokenTypes, "weight");

  // do not quantize Mamba/Kimi's small conv1d weights
  // NOTE: can't use LLM_TN here because the layer number is not known
  vQuantize &= cName.find("ssm_conv1d") == std::string::npos;
  vQuantize &= cName.find("shortconv.conv.weight") == std::string::npos;

  // do not quantize MiniMax's indexer projection weights, they are tiny
  vQuantize &= cName.find("indexer.k_proj.weight") == std::string::npos;
  vQuantize &= cName.find("indexer.q_proj.weight") == std::string::npos;

  // glm5-next
  if (pArch == cLlmArchGlm5Next) {
    vQuantize &= cName.find("hc_") == std::string::npos;
    vQuantize &= cName.find("indexer.attn_q_b") == std::string::npos;
    vQuantize &= cName.find("indexer.attn_k") == std::string::npos;
    vQuantize &= cName.find("indexer.proj") == std::string::npos;
    vQuantize &= cName.find("indexer_compressor_gate") == std::string::npos;
    vQuantize &= cName.find("indexer_compressor_ape") == std::string::npos;
    vQuantize &= cName.find("ssm_f_a.weight") == std::string::npos;
    vQuantize &= cName.find("ssm_f_b.weight") == std::string::npos;
    vQuantize &= cName.find("ssm_g_a.weight") == std::string::npos;
    vQuantize &= cName.find("ssm_g_b.weight") == std::string::npos;
    vQuantize &= cName.find("ssm_beta.weight") == std::string::npos;
    vQuantize &= cName.find("attn_kv_a_mqa.weight") == std::string::npos;
    vQuantize &= cName.find("attn_k_b.weight") == std::string::npos;
    vQuantize &= cName.find("attn_v_b.weight") == std::string::npos;
  }

  // do not quantize RWKV's small yet 2D weights
  vQuantize &= cName.find("time_mix_first.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_w0.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_w1.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_w2.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_v0.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_v1.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_v2.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_a0.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_a1.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_a2.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_g1.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_g2.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_decay_w1.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_decay_w2.weight") == std::string::npos;
  vQuantize &= cName.find("time_mix_lerp_fused.weight") == std::string::npos;

  // do not quantize relative position bias (T5)
  vQuantize &= cName.find("attn_rel_b.weight") == std::string::npos;

  // do not quantize specific multimodal tensors
  vQuantize &= cName.find(".position_embd") == std::string::npos;
  vQuantize &= cName.find("sam.pos_embd") == std::string::npos;
  vQuantize &= cName.find("sam.neck.") == std::string::npos;
  vQuantize &= cName.find("sam.net_") == std::string::npos;
  vQuantize &= cName.find(".rel_pos") == std::string::npos;
  vQuantize &= cName.find(".patch_embd") == std::string::npos;
  vQuantize &= cName.find(".patch_merger") == std::string::npos;

  // audio codebook
  vQuantize &= cName.find("a.rvq.codebook") == std::string::npos;
  vQuantize &= cName.find("mm.a.code_embd") == std::string::npos;

  return vQuantize;
}

//
// tensor type selection
//

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type fTensorTypeFallback(quantize_state_impl &pQs, const ggml_tensor *pT, const ggml_type pTargetType) {
  ggml_type vReturnType = pTargetType;

  const int64_t cNcols = pT->ne[0];
  const int64_t cQkK = ggml_blck_size(pTargetType);

  if (cNcols % cQkK != 0) { // this tensor's shape is incompatible with this quant
    MGS_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ", pT->name,
                 cNcols, cQkK, ggml_type_name(pTargetType));
    ++pQs.vNFallback;

    switch (pTargetType) {
    // types on the left: block size 256
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S: // types on the right: block size 32
    case GGML_TYPE_IQ4_XS:
      vReturnType = GGML_TYPE_IQ4_NL;
      break;
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_TQ1_0:
    case GGML_TYPE_TQ2_0:
      vReturnType = GGML_TYPE_Q4_0;
      break;
    case GGML_TYPE_Q4_K:
      vReturnType = GGML_TYPE_Q5_0;
      break;
    case GGML_TYPE_Q5_K:
      vReturnType = GGML_TYPE_Q5_1;
      break;
    case GGML_TYPE_Q6_K:
      vReturnType = GGML_TYPE_Q8_0;
      break;
    default:
      throw std::runtime_error(fFormat("no tensor type fallback is defined for type %s", ggml_type_name(pTargetType)));
    }
    if (cNcols % ggml_blck_size(vReturnType) != 0) {
      //
      // the fallback return type is still not compatible for this tensor!
      //
      // most likely, this tensor's first dimension is not divisible by 32.
      // this is very rare. we can either abort the quantization, or
      // fallback to F16 / F32.
      //
      MGS_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
      vReturnType = GGML_TYPE_F16;
    }
    MGS_LOG_WARN("-> falling back to %7s\n", ggml_type_name(vReturnType));
  }
  return vReturnType;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type fMgsTensorGetTypeImpl(quantize_state_impl &pQs, ggml_type pNewType, const ggml_tensor *pTensor,
                                       mgs_ftype pFtype, tensor_category pCategory) {
  const std::string cName = ggml_get_name(pTensor);

  // TODO: avoid hardcoded tensor names - use the TN_* constants
  const llm_arch cArch = pQs.vModel.vArch;

  auto vUseMoreBits = [](int pILayer, int pNLayers) -> bool {
    return pILayer < pNLayers / 8 || pILayer >= 7 * pNLayers / 8 || (pILayer - pNLayers / 8) % 3 == 2;
  };
  const int cNExpert = std::max(1, (int)pQs.vModel.vHparams.vNExpert);
  auto vLayerInfo = [cNExpert](int pILayer, int pNLayer, const char *pName) {
    if (cNExpert > 1) {
      // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
      // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
      // for getting the current layer as I initially thought, and we need to resort to parsing the
      // tensor name.
      if (sscanf(pName, "blk.%d.", &pILayer) != 1) {
        throw std::runtime_error(fFormat("Failed to determine layer for tensor %s", pName));
      }
      if (pILayer < 0 || pILayer >= pNLayer) {
        throw std::runtime_error(fFormat("Bad layer %d for tensor %s. Must be in [0, %d)", pILayer, pName, pNLayer));
      }
    }
    return std::make_pair(pILayer, pNLayer);
  };

  // by default, for glm5-next, don't let these tensors be quantized below Q8_0
  if (cArch == cLlmArchGlm5Next && (cName.find("attn_q_a") != std::string::npos ||
                                    cName.find("attn_q_b") != std::string::npos ||
                                    cName.find("nextn.eh_proj") != std::string::npos)) {
    switch (pNewType) {
    case GGML_TYPE_F32:
    case GGML_TYPE_BF16:
    case GGML_TYPE_F16:
      break;
    default:
      return GGML_TYPE_Q8_0;
    }
  }

  // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
  // with the quantization of the output tensor
  if (pCategory == tensor_category::cOutput || (pQs.vHasTiedEmbeddings && pCategory == tensor_category::cTokenEmbd)) {
    if (pQs.vParams->output_tensor_type < GGML_TYPE_COUNT) {
      pNewType = pQs.vParams->output_tensor_type;
    } else {
      const int64_t cNx = pTensor->ne[0];
      const int64_t cQkK = ggml_blck_size(pNewType);

      if (pFtype == MGS_FTYPE_MOSTLY_MXFP4_MOE) {
        pNewType = GGML_TYPE_Q8_0;
      } else if (cArch == cLlmArchFalcon || cNx % cQkK != 0) {
        pNewType = GGML_TYPE_Q8_0;
      } else if (pFtype == MGS_FTYPE_MOSTLY_IQ2_XXS || pFtype == MGS_FTYPE_MOSTLY_IQ2_XS ||
                 pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS || pFtype == MGS_FTYPE_MOSTLY_IQ1_S ||
                 pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M ||
                 pFtype == MGS_FTYPE_MOSTLY_IQ1_M) {
        pNewType = GGML_TYPE_Q5_K;
      } else if (pNewType != GGML_TYPE_Q8_0) {
        pNewType = GGML_TYPE_Q6_K;
      }
    }
  } else if (pFtype == MGS_FTYPE_MOSTLY_MXFP4_MOE) {
    // MoE   tensors -> MXFP4
    // other tensors -> Q8_0
    if (pTensor->ne[2] > 1) {
      pNewType = GGML_TYPE_MXFP4;
    } else {
      pNewType = GGML_TYPE_Q8_0;
    }
  } else if (pCategory == tensor_category::cTokenEmbd) {
    if (pQs.vParams->token_embedding_type < GGML_TYPE_COUNT) {
      pNewType = pQs.vParams->token_embedding_type;
    } else {
      if (pFtype == MGS_FTYPE_MOSTLY_IQ2_XXS || pFtype == MGS_FTYPE_MOSTLY_IQ2_XS || pFtype == MGS_FTYPE_MOSTLY_IQ1_S ||
          pFtype == MGS_FTYPE_MOSTLY_IQ1_M) {
        pNewType = GGML_TYPE_Q2_K;
      } else if (pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M) {
        pNewType = GGML_TYPE_IQ3_S;
      } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS) {
        pNewType = GGML_TYPE_IQ3_S;
      } else if (pFtype == MGS_FTYPE_MOSTLY_TQ1_0 || pFtype == MGS_FTYPE_MOSTLY_TQ2_0 ||
                 pFtype == MGS_FTYPE_MOSTLY_Q2_0) {
        pNewType = GGML_TYPE_Q4_K;
      }
    }
  } else if (pFtype == MGS_FTYPE_MOSTLY_IQ2_XXS || pFtype == MGS_FTYPE_MOSTLY_IQ2_XS ||
             pFtype == MGS_FTYPE_MOSTLY_IQ1_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M ||
             pFtype == MGS_FTYPE_MOSTLY_IQ1_M) {
    if (fCategoryIsAttnV(pCategory)) {
      if (pQs.vModel.vHparams.mNGqa() >= 4 || pQs.vModel.vHparams.vNExpert >= 4)
        pNewType = GGML_TYPE_Q4_K;
      else
        pNewType =
            pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
      ++pQs.vIAttentionWv;
    } else if (pQs.vModel.vHparams.vNExpert == 8 && pCategory == tensor_category::cAttentionK) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pCategory == tensor_category::cFfnDown) {
      if (pQs.vIFfnDown < pQs.vNFfnDown / 8) {
        pNewType =
            pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
      }
      ++pQs.vIFfnDown;
    } else if (pCategory == tensor_category::cAttentionOutput) {
      if (pQs.vModel.vHparams.vNExpert == 8) {
        pNewType = GGML_TYPE_Q5_K;
      } else {
        if (pFtype == MGS_FTYPE_MOSTLY_IQ1_S || pFtype == MGS_FTYPE_MOSTLY_IQ1_M)
          pNewType = GGML_TYPE_IQ2_XXS;
        else if (pFtype == MGS_FTYPE_MOSTLY_IQ2_S || pFtype == MGS_FTYPE_MOSTLY_IQ2_M)
          pNewType = GGML_TYPE_IQ3_S;
      }
    }
  } else if (fCategoryIsAttnV(pCategory)) {
    if (pFtype == MGS_FTYPE_MOSTLY_Q2_K) {
      pNewType = pQs.vModel.vHparams.mNGqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q2_K_S && pQs.vModel.vHparams.mNGqa() >= 4) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS) {
      pNewType = pQs.vModel.vHparams.mNGqa() >= 4 ? GGML_TYPE_Q4_K
                 : !pQs.vHasImatrix               ? GGML_TYPE_IQ3_S
                                                  : GGML_TYPE_IQ3_XXS;
    } else if ((pFtype == MGS_FTYPE_MOSTLY_IQ3_XS || pFtype == MGS_FTYPE_MOSTLY_IQ3_S) &&
               pQs.vModel.vHparams.mNGqa() >= 4) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_M) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_M) {
      pNewType = pQs.vIAttentionWv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_L)
      pNewType = GGML_TYPE_Q5_K;
    else if ((pFtype == MGS_FTYPE_MOSTLY_IQ4_NL || pFtype == MGS_FTYPE_MOSTLY_IQ4_XS) &&
             pQs.vModel.vHparams.mNGqa() >= 4) {
      pNewType = GGML_TYPE_Q5_K;
    } else if ((pFtype == MGS_FTYPE_MOSTLY_Q4_K_M || pFtype == MGS_FTYPE_MOSTLY_Q5_K_M) &&
               vUseMoreBits(pQs.vIAttentionWv, pQs.vNAttentionWv))
      pNewType = GGML_TYPE_Q6_K;
    else if (pFtype == MGS_FTYPE_MOSTLY_Q4_K_S && pQs.vIAttentionWv < 4)
      pNewType = GGML_TYPE_Q5_K;
    if (pQs.vModel.vType == cLlmType70b) {
      // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
      // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
      // nearly negligible increase in model size by quantizing this tensor with more bits:
      if (pNewType == GGML_TYPE_Q3_K || pNewType == GGML_TYPE_Q4_K)
        pNewType = GGML_TYPE_Q5_K;
    }
    if (pQs.vModel.vHparams.vNExpert == 8) {
      // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
      // TODO: explore better strategies
      pNewType = GGML_TYPE_Q8_0;
    }
    ++pQs.vIAttentionWv;
  } else if (pCategory == tensor_category::cAttentionK) {
    if (pQs.vModel.vHparams.vNExpert == 8) {
      // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
      // TODO: explore better strategies
      pNewType = GGML_TYPE_Q8_0;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XS) {
      pNewType = GGML_TYPE_IQ3_XXS;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS) {
      pNewType = GGML_TYPE_IQ2_S;
    }
  } else if (pCategory == tensor_category::cAttentionQ) {
    if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XS) {
      pNewType = GGML_TYPE_IQ3_XXS;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS) {
      pNewType = GGML_TYPE_IQ2_S;
    }
  } else if (pCategory == tensor_category::cFfnDown) {
    auto vInfo = vLayerInfo(pQs.vIFfnDown, pQs.vNFfnDown, cName.c_str());
    int vILayer = vInfo.first, vNLayer = vInfo.second;
    if (pFtype == MGS_FTYPE_MOSTLY_Q2_K)
      pNewType = GGML_TYPE_Q3_K;
    else if (pFtype == MGS_FTYPE_MOSTLY_Q2_K_S) {
      if (vILayer < vNLayer / 8)
        pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS && !pQs.vHasImatrix) {
      pNewType = vILayer < vNLayer / 8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_M) {
      pNewType = vILayer < vNLayer / 16                                      ? GGML_TYPE_Q5_K
                 : cArch != cLlmArchFalcon || vUseMoreBits(vILayer, vNLayer) ? GGML_TYPE_Q4_K
                                                                             : GGML_TYPE_Q3_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_M &&
               (vILayer < vNLayer / 8 || (pQs.vModel.vHparams.vNExpert == 8 && vUseMoreBits(vILayer, vNLayer)))) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_L) {
      pNewType = cArch == cLlmArchFalcon ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q4_K_M) {
      if (cArch == cLlmArchFalcon) {
        pNewType = vILayer < vNLayer / 16           ? GGML_TYPE_Q6_K
                   : vUseMoreBits(vILayer, vNLayer) ? GGML_TYPE_Q5_K
                                                    : GGML_TYPE_Q4_K;
      } else {
        if (vUseMoreBits(vILayer, vNLayer))
          pNewType = GGML_TYPE_Q6_K;
      }
    } else if (vILayer < vNLayer / 8 && (pFtype == MGS_FTYPE_MOSTLY_IQ4_NL || pFtype == MGS_FTYPE_MOSTLY_IQ4_XS) &&
               !pQs.vHasImatrix) {
      pNewType = GGML_TYPE_Q5_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q5_K_M && vUseMoreBits(vILayer, vNLayer))
      pNewType = GGML_TYPE_Q6_K;
    else if (pFtype == MGS_FTYPE_MOSTLY_Q4_K_S && cArch != cLlmArchFalcon && vILayer < vNLayer / 8) {
      pNewType = GGML_TYPE_Q5_K;
    } else if ((pFtype == MGS_FTYPE_MOSTLY_Q4_0 || pFtype == MGS_FTYPE_MOSTLY_Q5_0) && pQs.vHasImatrix &&
               vILayer < vNLayer / 8) {
      // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
      // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
      // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
      pNewType = pFtype == MGS_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
    }
    ++pQs.vIFfnDown;
  } else if (pCategory == tensor_category::cAttentionOutput) {
    if (cArch != cLlmArchFalcon) {
      if (pQs.vModel.vHparams.vNExpert == 8) {
        if (pFtype == MGS_FTYPE_MOSTLY_Q2_K || pFtype == MGS_FTYPE_MOSTLY_IQ3_XS ||
            pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS || pFtype == MGS_FTYPE_MOSTLY_Q3_K_S ||
            pFtype == MGS_FTYPE_MOSTLY_Q3_K_M || pFtype == MGS_FTYPE_MOSTLY_IQ4_NL ||
            pFtype == MGS_FTYPE_MOSTLY_Q4_K_S || pFtype == MGS_FTYPE_MOSTLY_Q4_K_M ||
            pFtype == MGS_FTYPE_MOSTLY_IQ3_S || pFtype == MGS_FTYPE_MOSTLY_IQ3_M || pFtype == MGS_FTYPE_MOSTLY_IQ4_XS) {
          pNewType = GGML_TYPE_Q5_K;
        }
      } else {
        if (pFtype == MGS_FTYPE_MOSTLY_Q2_K)
          pNewType = GGML_TYPE_Q3_K;
        else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XXS)
          pNewType = GGML_TYPE_IQ3_S;
        else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_M)
          pNewType = GGML_TYPE_Q4_K;
        else if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_L)
          pNewType = GGML_TYPE_Q5_K;
        else if (pFtype == MGS_FTYPE_MOSTLY_IQ3_M)
          pNewType = GGML_TYPE_Q4_K;
      }
    } else {
      if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_L)
        pNewType = GGML_TYPE_Q4_K;
    }
  } else if (pCategory == tensor_category::cAttentionQkv) {
    if (pFtype == MGS_FTYPE_MOSTLY_Q3_K_M || pFtype == MGS_FTYPE_MOSTLY_Q3_K_L || pFtype == MGS_FTYPE_MOSTLY_IQ3_M) {
      pNewType = GGML_TYPE_Q4_K;
    } else if (pFtype == MGS_FTYPE_MOSTLY_Q4_K_M)
      pNewType = GGML_TYPE_Q5_K;
    else if (pFtype == MGS_FTYPE_MOSTLY_Q5_K_M)
      pNewType = GGML_TYPE_Q6_K;
  } else if (pCategory == tensor_category::cFfnGate) {
    auto vInfo = vLayerInfo(pQs.vIFfnGate, pQs.vNFfnGate, cName.c_str());
    int vILayer = vInfo.first, vNLayer = vInfo.second;
    if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XS && (vILayer >= vNLayer / 8 && vILayer < 7 * vNLayer / 8)) {
      pNewType = GGML_TYPE_IQ3_XXS;
    }
    ++pQs.vIFfnGate;
  } else if (pCategory == tensor_category::cFfnUp) {
    auto vInfo = vLayerInfo(pQs.vIFfnUp, pQs.vNFfnUp, cName.c_str());
    int vILayer = vInfo.first, vNLayer = vInfo.second;
    if (pFtype == MGS_FTYPE_MOSTLY_IQ3_XS && (vILayer >= vNLayer / 8 && vILayer < 7 * vNLayer / 8)) {
      pNewType = GGML_TYPE_IQ3_XXS;
    }
    ++pQs.vIFfnUp;
  }

  return pNewType;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type fMgsTensorGetType(quantize_state_impl &pQs, const mgs_model_quantize_params *pParams,
                                   const ggml_tensor *pTensor, ggml_type pDefaultType, const tensor_metadata &pTm) {
  if (!fTensorAllowsQuantization(pParams, pQs.vModel.vArch, pTensor)) {
    return pTensor->type;
  }
  if (pParams->token_embedding_type < GGML_TYPE_COUNT && pTm.vCategory == tensor_category::cTokenEmbd) {
    return pParams->token_embedding_type;
  }
  if (pParams->output_tensor_type < GGML_TYPE_COUNT && pTm.vCategory == tensor_category::cOutput) {
    return pParams->output_tensor_type;
  }

  ggml_type vNewType = pDefaultType;

  // get more optimal quantization type based on the tensor shape, layer, etc.
  if (ggml_is_quantized(pDefaultType)) {
    // if the user provided tensor types - use those
    bool vManual = false;
    if (!pQs.lTensorTypePatterns.empty()) {
      const std::string cTensorName(pTensor->name);
      for (const auto &[vPattern, vQtype] : pQs.lTensorTypePatterns) {
        if (std::regex_search(cTensorName, vPattern)) {
          if (vQtype != vNewType) {
            MGS_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n", __func__, cTensorName.c_str(),
                         ggml_type_name(vNewType), ggml_type_name(vQtype));
            vNewType = vQtype;
          }
          vManual = true;
          break;
        }
      }
    }

    // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
    if (!vManual && !pParams->pure) {
      vNewType = fMgsTensorGetTypeImpl(pQs, vNewType, pTensor, pParams->ftype, pTm.vCategory);
    }

    // incompatible tensor shapes are handled here - fallback to a compatible type
    vNewType = fTensorTypeFallback(pQs, pTensor, vNewType);
  }

  return vNewType;
}

//
// quantization implementation
//

static size_t fMgsTensorQuantizeImpl(enum ggml_type pNewType, const float *pF32Data, void *pNewData,
                                     const int64_t pChunkSize, int64_t pNrows, int64_t pNPerRow, const float *pImatrix,
                                     std::vector<std::thread> &pWorkers, const int pNthread) {
  if (pNthread < 2) {
    // single-thread
    size_t vNewSize = ggml_quantize_chunk(pNewType, pF32Data, pNewData, 0, pNrows, pNPerRow, pImatrix);
    if (!ggml_validate_row_data(pNewType, pNewData, vNewSize)) {
      throw std::runtime_error("quantized data validation failed");
    }
    return vNewSize;
  }

  std::mutex vMutex;
  int64_t vCounter = 0;
  size_t vNewSize = 0;
  bool vValid = true;
  auto vCompute = [&vMutex, &vCounter, &vNewSize, &vValid, pNewType, pF32Data, pNewData, pChunkSize, pNrows, pNPerRow,
                   pImatrix]() {
    const int64_t cNrowsPerChunk = pChunkSize / pNPerRow;
    size_t vLocalSize = 0;
    while (true) {
      std::unique_lock<std::mutex> vLock(vMutex);
      int64_t vFirstRow = vCounter;
      vCounter += cNrowsPerChunk;
      if (vFirstRow >= pNrows) {
        if (vLocalSize > 0) {
          vNewSize += vLocalSize;
        }
        break;
      }
      vLock.unlock();
      const int64_t cThisNrow = std::min(pNrows - vFirstRow, cNrowsPerChunk);
      size_t vThisSize =
          ggml_quantize_chunk(pNewType, pF32Data, pNewData, vFirstRow * pNPerRow, cThisNrow, pNPerRow, pImatrix);
      vLocalSize += vThisSize;

      // validate the quantized data
      const size_t cRowSize = ggml_row_size(pNewType, pNPerRow);
      void *vThisData = (char *)pNewData + vFirstRow * cRowSize;
      if (!ggml_validate_row_data(pNewType, vThisData, vThisSize)) {
        std::unique_lock<std::mutex> vLock(vMutex);
        vValid = false;
        break;
      }
    }
  };
  for (int vIt = 0; vIt < pNthread - 1; ++vIt) {
    pWorkers.emplace_back(vCompute);
  }
  vCompute();
  for (auto &vW : pWorkers) {
    vW.join();
  }
  pWorkers.clear();
  if (!vValid) {
    throw std::runtime_error("quantized data validation failed");
  }
  return vNewSize;
}

//
// imatrix requirement check
//

static bool fTensorRequiresImatrix(const char *pTensorName, const ggml_type pDstType, const mgs_ftype pFtype) {
  if (fTensorNameMatchTokenEmbd(pTensorName) || fTensorNameMatchOutputWeight(pTensorName)) {
    return false;
  }
  switch (pDstType) {
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ1_S:
    return true;
  case GGML_TYPE_Q2_K:
    // as a general rule, the k-type quantizations don't require imatrix data.
    // the only exception is Q2_K tensors that are part of a Q2_K_S file.
    return pFtype == MGS_FTYPE_MOSTLY_Q2_K_S;
  default:
    return false;
  }
}

//
// given a file type, get the default tensor type
//

ggml_type mgs_ftype_get_default_type(mgs_ftype pFtype) {
  switch (pFtype) {
  case MGS_FTYPE_MOSTLY_Q4_0:
    return GGML_TYPE_Q4_0;
  case MGS_FTYPE_MOSTLY_Q4_1:
    return GGML_TYPE_Q4_1;
  case MGS_FTYPE_MOSTLY_Q5_0:
    return GGML_TYPE_Q5_0;
  case MGS_FTYPE_MOSTLY_Q5_1:
    return GGML_TYPE_Q5_1;
  case MGS_FTYPE_MOSTLY_Q8_0:
    return GGML_TYPE_Q8_0;
  case MGS_FTYPE_MOSTLY_F16:
    return GGML_TYPE_F16;
  case MGS_FTYPE_MOSTLY_BF16:
    return GGML_TYPE_BF16;
  case MGS_FTYPE_ALL_F32:
    return GGML_TYPE_F32;
  case MGS_FTYPE_MOSTLY_Q1_0:
    return GGML_TYPE_Q1_0;
  case MGS_FTYPE_MOSTLY_Q2_0:
    return GGML_TYPE_Q2_0;

  case MGS_FTYPE_MOSTLY_MXFP4_MOE:
    return GGML_TYPE_MXFP4;

  // K-quants
  case MGS_FTYPE_MOSTLY_Q2_K_S:
  case MGS_FTYPE_MOSTLY_Q2_K:
    return GGML_TYPE_Q2_K;
  case MGS_FTYPE_MOSTLY_IQ3_XS:
    return GGML_TYPE_IQ3_S;
  case MGS_FTYPE_MOSTLY_Q3_K_S:
  case MGS_FTYPE_MOSTLY_Q3_K_M:
  case MGS_FTYPE_MOSTLY_Q3_K_L:
    return GGML_TYPE_Q3_K;
  case MGS_FTYPE_MOSTLY_Q4_K_S:
  case MGS_FTYPE_MOSTLY_Q4_K_M:
    return GGML_TYPE_Q4_K;
  case MGS_FTYPE_MOSTLY_Q5_K_S:
  case MGS_FTYPE_MOSTLY_Q5_K_M:
    return GGML_TYPE_Q5_K;
  case MGS_FTYPE_MOSTLY_Q6_K:
    return GGML_TYPE_Q6_K;
  case MGS_FTYPE_MOSTLY_TQ1_0:
    return GGML_TYPE_TQ1_0;
  case MGS_FTYPE_MOSTLY_TQ2_0:
    return GGML_TYPE_TQ2_0;
  case MGS_FTYPE_MOSTLY_IQ2_XXS:
    return GGML_TYPE_IQ2_XXS;
  case MGS_FTYPE_MOSTLY_IQ2_XS:
    return GGML_TYPE_IQ2_XS;
  case MGS_FTYPE_MOSTLY_IQ2_S:
    return GGML_TYPE_IQ2_XS;
  case MGS_FTYPE_MOSTLY_IQ2_M:
    return GGML_TYPE_IQ2_S;
  case MGS_FTYPE_MOSTLY_IQ3_XXS:
    return GGML_TYPE_IQ3_XXS;
  case MGS_FTYPE_MOSTLY_IQ1_S:
    return GGML_TYPE_IQ1_S;
  case MGS_FTYPE_MOSTLY_IQ1_M:
    return GGML_TYPE_IQ1_M;
  case MGS_FTYPE_MOSTLY_IQ4_NL:
    return GGML_TYPE_IQ4_NL;
  case MGS_FTYPE_MOSTLY_IQ4_XS:
    return GGML_TYPE_IQ4_XS;
  case MGS_FTYPE_MOSTLY_IQ3_S:
  case MGS_FTYPE_MOSTLY_IQ3_M:
    return GGML_TYPE_IQ3_S;

  default:
    return GGML_TYPE_COUNT;
  }
}

static void fInitQuantizeStateCounters(quantize_state_impl &pQs, std::vector<tensor_metadata> &pMetadata) {
  for (auto &vTm : pMetadata) {
    tensor_category vCat = fTensorGetCategory(vTm.vName);
    vTm.vCategory = vCat;

    if (fCategoryIsAttnV(vCat)) {
      ++pQs.vNAttentionWv;
    }

    if (vCat == tensor_category::cOutput) {
      pQs.vHasTiedEmbeddings = false;
    }
  }
  pQs.vNFfnDown = pQs.vNFfnGate = pQs.vNFfnUp = (int)pQs.vModel.vHparams.vNLayerAll;
}

//
// main quantization driver
//

static void fMgsModelQuantizeImpl(const std::string &pFnameInp, const std::string &pFnameOut,
                                  const mgs_model_quantize_params *pParams) {
  mgs_ftype vFtype = pParams->ftype;

  int vNthread = pParams->nthread;

  if (vNthread <= 0) {
    vNthread = std::thread::hardware_concurrency();
  }

  ggml_type vDefaultType = mgs_ftype_get_default_type(vFtype);
  if (vDefaultType == GGML_TYPE_COUNT) {
    throw std::runtime_error(fFormat("invalid output file type %d\n", vFtype));
  }

  // mmap consistently increases speed on Linux, and also increases speed on Windows with
  // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
  constexpr mgs_load_mode cLoadMode = MGS_LOAD_MODE_MMAP;
#else
  constexpr mgs_load_mode cLoadMode = MGS_LOAD_MODE_NONE;
#endif

  const mgs_model_kv_override *vKvOverrides = pParams->kv_overrides;
  std::vector<std::string> lSplits = {};
  mgs_model_loader vMl(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr, pFnameInp,
                       lSplits, /*file*/ nullptr, /*load_mode*/ cLoadMode, /*check_tensors*/ true, /*no_alloc*/ false,
                       /*load_mtp*/ true, vKvOverrides, nullptr);
  vMl.mInitMappings(false); // no prefetching

  auto vMparams = mgs_model_default_params();
  std::unique_ptr<mgs_model> vModelPtr(mgs_model_create(vMl, vMparams));

  auto *vModel = dynamic_cast<mgs_model_base *>(vModelPtr.get());
  if (vModel == nullptr) {
    GGML_ABORT("fatal error: model does not implement mgs_model_base");
  }

  vModel->mLoadHparams(vMl);
  vModel->mLoadStats(vMl);

  quantize_state_impl vQs(*vModel, pParams);

  if (pParams->only_copy) {
    vFtype = vMl.vFtype;
  }
  std::unordered_map<std::string, std::vector<float>> dIData;
  const std::unordered_map<std::string, std::vector<float>> *dImatrixData = nullptr;
  if (pParams->imatrix) {
    for (const mgs_model_imatrix_data *vP = pParams->imatrix; vP->name != nullptr; vP++) {
      dIData.emplace(vP->name, std::vector<float>(vP->data, vP->data + vP->size));
    }
    dImatrixData = &dIData;
    if (dImatrixData) {
      MGS_LOG_INFO("\n%s: have importance matrix data with %d entries\n", __func__, (int)dImatrixData->size());
      vQs.vHasImatrix = true;
      // check imatrix for nans or infs
      for (const auto &vKv : *dImatrixData) {
        for (float vF : vKv.second) {
          if (!std::isfinite(vF)) {
            throw std::runtime_error(fFormat("imatrix contains non-finite value %f\n", vF));
          }
        }
      }
    }
  }

  const size_t cAlign = GGUF_DEFAULT_ALIGNMENT;
  gguf_context_ptr vCtxOut{gguf_init_empty()};

  std::vector<int> lPruneList = {};
  if (pParams->prune_layers) {
    for (const int32_t *vP = pParams->prune_layers; *vP != -1; vP++) {
      lPruneList.push_back(*vP);
    }
  }

  // copy the KV pairs from the input file
  gguf_set_kv(vCtxOut.get(), vMl.vMetadata);
  gguf_set_val_u32(vCtxOut.get(), vMl.vLlmKv(cLlmKvGeneralQuantizationVersion).c_str(), GGML_QNT_VERSION);
  gguf_set_val_u32(vCtxOut.get(), vMl.vLlmKv(cLlmKvGeneralFileType).c_str(), vFtype);

  // Remove split metadata
  gguf_remove_key(vCtxOut.get(), vMl.vLlmKv(cLlmKvSplitNo).c_str());
  gguf_remove_key(vCtxOut.get(), vMl.vLlmKv(cLlmKvSplitCount).c_str());
  gguf_remove_key(vCtxOut.get(), vMl.vLlmKv(cLlmKvSplitTensorsCount).c_str());

  if (pParams->kv_overrides) {
    for (const mgs_model_kv_override *vO = pParams->kv_overrides; vO->key[0] != 0; ++vO) {
      if (vO->tag == MGS_KV_OVERRIDE_TYPE_FLOAT) {
        gguf_set_val_f32(vCtxOut.get(), vO->key, vO->val_f64);
      } else if (vO->tag == MGS_KV_OVERRIDE_TYPE_INT) {
        // Setting type to UINT32.
        gguf_set_val_u32(vCtxOut.get(), vO->key, (uint32_t)std::abs(vO->val_i64));
      } else if (vO->tag == MGS_KV_OVERRIDE_TYPE_BOOL) {
        gguf_set_val_bool(vCtxOut.get(), vO->key, vO->val_bool);
      } else if (vO->tag == MGS_KV_OVERRIDE_TYPE_STR) {
        gguf_set_val_str(vCtxOut.get(), vO->key, vO->val_str);
      } else {
        MGS_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, vO->key);
      }
    }
  }

  std::map<int, std::string> dMapped;
  int vBlkId = 0;

  // make a list of weights
  std::vector<const mgs_model_loader::mgs_tensor_weight *> lTensors;
  lTensors.reserve(vMl.dWeightsMap.size());
  for (const auto &vIt : vMl.dWeightsMap) {
    const std::string cRemappedName(fRemapLayer(vIt.first, lPruneList, dMapped, vBlkId));
    if (cRemappedName.empty()) {
      MGS_LOG_DEBUG("%s: pruning tensor %s\n", __func__, vIt.first.c_str());
      continue;
    }

    if (cRemappedName != vIt.first) {
      ggml_set_name(vIt.second.vTensor, cRemappedName.c_str());
      MGS_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, vIt.first.c_str(), ggml_get_name(vIt.second.vTensor));
    }
    lTensors.push_back(&vIt.second);
  }
  if (!lPruneList.empty()) {
    gguf_set_val_u32(vCtxOut.get(), vMl.vLlmKv(cLlmKvBlockCount).c_str(), vBlkId);
  }

  // keep_split requires that the weights are sorted by split index
  if (pParams->keep_split) {
    std::sort(lTensors.begin(), lTensors.end(),
              [](const mgs_model_loader::mgs_tensor_weight *pA, const mgs_model_loader::mgs_tensor_weight *pB) {
                if (pA->vIdx == pB->vIdx) {
                  return pA->vOffs < pB->vOffs;
                }
                return pA->vIdx < pB->vIdx;
              });
  }

  // compute tensor metadata once and cache it
  std::vector<tensor_metadata> lMetadata(lTensors.size());
  for (size_t vI = 0; vI < lTensors.size(); ++vI) {
    lMetadata[vI].vName = ggml_get_name(lTensors[vI]->vTensor);
  }

  // initialize quantization state counters and metadata categories
  fInitQuantizeStateCounters(vQs, lMetadata);

  int vIdx = 0;
  uint16_t vNSplit = 1;

  // Assume split index is continuous
  if (pParams->keep_split) {
    for (const auto *vIt : lTensors) {
      vNSplit = std::max(uint16_t(vIt->vIdx + 1), vNSplit);
    }
  }
  std::vector<gguf_context_ptr> lCtxOuts(vNSplit);
  lCtxOuts[0] = std::move(vCtxOut);

  // flag for --dry-run
  bool vWillRequireImatrix = false;

  //
  // preliminary iteration over all weights
  //

  for (size_t vI = 0; vI < lTensors.size(); ++vI) {
    const auto *vIt = lTensors[vI];
    const struct ggml_tensor *vTensor = vIt->vTensor;

    uint16_t vISplit = pParams->keep_split ? vIt->vIdx : 0;
    if (!lCtxOuts[vISplit]) {
      lCtxOuts[vISplit].reset(gguf_init_empty());
    }
    gguf_add_tensor(lCtxOuts[vISplit].get(), vTensor);

    lMetadata[vI].vAllowsQuantization = fTensorAllowsQuantization(pParams, vModel->vArch, vTensor);

    if (lMetadata[vI].vAllowsQuantization) {
      lMetadata[vI].vTargetType = fMgsTensorGetType(vQs, pParams, vTensor, vDefaultType, lMetadata[vI]);
    } else {
      lMetadata[vI].vTargetType = vTensor->type;
    }

    lMetadata[vI].vRequiresImatrix = fTensorRequiresImatrix(vTensor->name, lMetadata[vI].vTargetType, vFtype);

    if (pParams->imatrix) {
      lMetadata[vI].vRemappedImatrixName = fRemapImatrix(vTensor->name, dMapped);
    } else if (lMetadata[vI].vAllowsQuantization && lMetadata[vI].vRequiresImatrix) {
      if (pParams->dry_run) {
        vWillRequireImatrix = true;
      } else {
        MGS_LOG_ERROR("\n============================================================================\n"
                      " ERROR: this quantization requires an importance matrix!\n"
                      "        - offending tensor: %s\n"
                      "        - target type: %s\n"
                      "============================================================================\n\n",
                      lMetadata[vI].vName.c_str(), ggml_type_name(lMetadata[vI].vTargetType));
        throw std::runtime_error("this quantization requires an imatrix!");
      }
    }
  }

  // Set split info if needed
  if (vNSplit > 1) {
    for (size_t vI = 0; vI < lCtxOuts.size(); ++vI) {
      gguf_set_val_u16(lCtxOuts[vI].get(), vMl.vLlmKv(cLlmKvSplitNo).c_str(), vI);
      gguf_set_val_u16(lCtxOuts[vI].get(), vMl.vLlmKv(cLlmKvSplitCount).c_str(), vNSplit);
      gguf_set_val_i32(lCtxOuts[vI].get(), vMl.vLlmKv(cLlmKvSplitTensorsCount).c_str(), (int32_t)lTensors.size());
    }
  }

  size_t vTotalSizeOrg = 0;
  size_t vTotalSizeNew = 0;

  std::vector<std::thread> lWorkers;
  lWorkers.reserve(vNthread);

  std::vector<no_init<uint8_t>> lReadData;
  std::vector<no_init<uint8_t>> lWork;
  std::vector<no_init<float>> lF32ConvBuf;

  int vCurSplit = -1;
  std::ofstream vFout;
  auto vCloseOfstream = [&]() {
    // Write metadata and close file handler
    if (vFout.is_open()) {
      vFout.seekp(0);
      std::vector<uint8_t> lData(gguf_get_meta_size(lCtxOuts[vCurSplit].get()));
      gguf_get_meta_data(lCtxOuts[vCurSplit].get(), lData.data());
      vFout.write((const char *)lData.data(), lData.size());
      vFout.close();
    }
  };
  auto vNewOfstream = [&](int pIndex) {
    vCurSplit = pIndex;
    GGML_ASSERT(lCtxOuts[vCurSplit] && "Find uninitialized gguf_context");
    std::string vFname = pFnameOut;
    if (pParams->keep_split) {
      std::vector<char> lSplitPath(mgs_path_max(), 0);
      mgs_split_path(lSplitPath.data(), lSplitPath.size(), pFnameOut.c_str(), vCurSplit, vNSplit);
      vFname = std::string(lSplitPath.data());
    }

    vFout = std::ofstream(vFname, std::ios::binary);
    vFout.exceptions(std::ofstream::failbit); // fail fast on write errors
    const size_t cMetaSize = gguf_get_meta_size(lCtxOuts[vCurSplit].get());
    // placeholder for the meta data
    ::fZeros(vFout, cMetaSize);
  };

  // no output file for --dry-run
  if (!pParams->dry_run) {
    vNewOfstream(0);
  }

  //
  // main loop: iterate over all weights
  //

  for (size_t vI = 0; vI < lTensors.size(); ++vI) {
    const auto &vWeight = *lTensors[vI];
    const auto &vTm = lMetadata[vI];
    ggml_tensor *vTensor = vWeight.vTensor;

    if (!pParams->dry_run && (vWeight.vIdx != vCurSplit && pParams->keep_split)) {
      vCloseOfstream();
      vNewOfstream(vWeight.vIdx);
    }

    const size_t cTensorSize = ggml_nbytes(vTensor);

    if (!pParams->dry_run) {
      if (!vMl.vUseMmap) {
        if (lReadData.size() < cTensorSize) {
          lReadData.resize(cTensorSize);
        }
        vTensor->data = lReadData.data();
      }
      vMl.mLoadDataFor(vTensor);
    }

    MGS_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ", ++vIdx, vMl.vNTensors, ggml_get_name(vTensor),
                 mgs_format_tensor_shape(vTensor).c_str(), ggml_type_name(vTensor->type));

    const ggml_type cCurType = vTensor->type;
    const ggml_type cNewType = vTm.vTargetType;

    // If we've decided to quantize to the same type the tensor is already
    // in then there's nothing to do.
    bool vQuantize = cCurType != cNewType;

    void *vNewData;
    size_t vNewSize;

    if (pParams->dry_run) {
      // the --dry-run option calculates the final quantization size without quantizing
      if (vQuantize) {
        vNewSize = ggml_nrows(vTensor) * ggml_row_size(cNewType, vTensor->ne[0]);
        MGS_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n", cTensorSize / 1024.0 / 1024.0, vNewSize / 1024.0 / 1024.0,
                     ggml_type_name(cNewType));
        if (!vWillRequireImatrix && vTm.vRequiresImatrix) {
          vWillRequireImatrix = true;
        }
      } else {
        vNewSize = cTensorSize;
        MGS_LOG_INFO("size = %8.3f MiB\n", vNewSize / 1024.0 / 1024.0);
      }
      vTotalSizeOrg += cTensorSize;
      vTotalSizeNew += vNewSize;
      continue;
    } else {
      // no --dry-run, perform quantization
      if (!vQuantize) {
        vNewData = vTensor->data;
        vNewSize = cTensorSize;
        MGS_LOG_INFO("size = %8.3f MiB\n", cTensorSize / 1024.0 / 1024.0);
      } else {
        const int64_t cNelements = ggml_nelements(vTensor);

        const float *vImatrix = nullptr;
        if (dImatrixData) {
          auto vIt = dImatrixData->find(vTm.vRemappedImatrixName);
          if (vIt == dImatrixData->end()) {
            MGS_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, vTensor->name);
          } else {
            if (vIt->second.size() == (size_t)vTensor->ne[0] * vTensor->ne[2]) {
              vImatrix = vIt->second.data();
            } else {
              MGS_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                           int(vIt->second.size()), int(vTensor->ne[0] * vTensor->ne[2]), vTensor->name);

              // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
              // this is a significant error and it may be good idea to abort the process if this happens,
              // since many people will miss the error and not realize that most of the model is being quantized without
              // an imatrix tok_embd should be ignored in this case, since it always causes this warning
              if (!fTensorNameMatchTokenEmbd(vTensor->name)) {
                throw std::runtime_error(fFormat("imatrix size %d is different from tensor size %d for %s",
                                                 int(vIt->second.size()), int(vTensor->ne[0] * vTensor->ne[2]),
                                                 vTensor->name));
              }
            }
          }
        }
        if (!vImatrix && vTm.vRequiresImatrix) {
          MGS_LOG_ERROR("\n\n============================================================\n");
          MGS_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", vTensor->name);
          MGS_LOG_ERROR("The result will be garbage, so bailing out\n");
          MGS_LOG_ERROR("============================================================\n\n");
          throw std::runtime_error(
              fFormat("Missing importance matrix for tensor %s in a very low-bit quantization", vTensor->name));
        }

        float *vF32Data;

        if (vTensor->type == GGML_TYPE_F32) {
          vF32Data = (float *)vTensor->data;
        } else if (ggml_is_quantized(vTensor->type) && !pParams->allow_requantize) {
          throw std::runtime_error(fFormat("requantizing from type %s is disabled", ggml_type_name(vTensor->type)));
        } else {
          fMgsTensorDequantizeImpl(vTensor, lF32ConvBuf, lWorkers, cNelements, vNthread);
          vF32Data = (float *)lF32ConvBuf.data();
        }

        MGS_LOG_INFO("converting to %s .. ", ggml_type_name(cNewType));
        fflush(stdout);

        if (lWork.size() < (size_t)cNelements * 4) {
          lWork.resize(cNelements * 4); // upper bound on size
        }
        vNewData = lWork.data();

        const int64_t cNPerRow = vTensor->ne[0];
        const int64_t cNrows = vTensor->ne[1];

        static const int64_t cMinChunkSize = 32 * 512;
        const int64_t cChunkSize =
            (cNPerRow >= cMinChunkSize ? cNPerRow : cNPerRow * ((cMinChunkSize + cNPerRow - 1) / cNPerRow));

        const int64_t cNelementsMatrix = vTensor->ne[0] * vTensor->ne[1];
        const int64_t cNchunk = (cNelementsMatrix + cChunkSize - 1) / cChunkSize;
        const int64_t cNthreadUse = vNthread > 1 ? std::max((int64_t)1, std::min((int64_t)vNthread, cNchunk)) : 1;

        // quantize each expert separately since they have different importance matrices
        vNewSize = 0;
        for (int64_t vI03 = 0; vI03 < vTensor->ne[2]; ++vI03) {
          const float *vF32Data03 = vF32Data + vI03 * cNelementsMatrix;
          void *vNewData03 = (char *)vNewData + ggml_row_size(cNewType, cNPerRow) * vI03 * cNrows;
          const float *vImatrix03 = vImatrix ? vImatrix + vI03 * cNPerRow : nullptr;

          vNewSize += fMgsTensorQuantizeImpl(cNewType, vF32Data03, vNewData03, cChunkSize, cNrows, cNPerRow, vImatrix03,
                                             lWorkers, cNthreadUse);
        }
        MGS_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", cTensorSize / 1024.0 / 1024.0, vNewSize / 1024.0 / 1024.0);
      }
      vTotalSizeOrg += cTensorSize;
      vTotalSizeNew += vNewSize;

      // update the gguf meta data as we go
      gguf_set_tensor_type(lCtxOuts[vCurSplit].get(), lMetadata[vI].vName.c_str(), cNewType);
      GGML_ASSERT(gguf_get_tensor_size(lCtxOuts[vCurSplit].get(),
                                       gguf_find_tensor(lCtxOuts[vCurSplit].get(), lMetadata[vI].vName.c_str())) ==
                  vNewSize);
      gguf_set_tensor_data(lCtxOuts[vCurSplit].get(), lMetadata[vI].vName.c_str(), vNewData);

      // write tensor data + padding
      vFout.write((const char *)vNewData, vNewSize);
      fZeros(vFout, GGML_PAD(vNewSize, cAlign) - vNewSize);
    } // no --dry-run
  } // main loop

  if (!pParams->dry_run) {
    vCloseOfstream();
  }

  MGS_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, vTotalSizeOrg / 1024.0 / 1024.0,
               vTotalSizeOrg * 8.0 / vMl.vNElements);
  MGS_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, vTotalSizeNew / 1024.0 / 1024.0,
               vTotalSizeNew * 8.0 / vMl.vNElements);

  if (!pParams->imatrix && pParams->dry_run && vWillRequireImatrix) {
    MGS_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will "
                 "require an imatrix!\n",
                 __func__);
  }

  if (vQs.vNFallback > 0) {
    MGS_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n", __func__, vQs.vNFallback,
                 vMl.vNTensors);
  }
}

//
// interface implementation
//

mgs_model_quantize_params mgs_model_quantize_default_params() {
  mgs_model_quantize_params vResult = {/*.nthread                     =*/0,
                                       /*.ftype                       =*/MGS_FTYPE_MOSTLY_Q8_0,
                                       /*.output_tensor_type          =*/GGML_TYPE_COUNT,
                                       /*.token_embedding_type        =*/GGML_TYPE_COUNT,
                                       /*.allow_requantize            =*/false,
                                       /*.quantize_output_tensor      =*/true,
                                       /*.only_copy                   =*/false,
                                       /*.pure                        =*/false,
                                       /*.keep_split                  =*/false,
                                       /*.dry_run                     =*/false,
                                       /*.imatrix                     =*/nullptr,
                                       /*.kv_overrides                =*/nullptr,
                                       /*.tt_overrides                =*/nullptr,
                                       /*.prune_layers                =*/nullptr};

  return vResult;
}

uint32_t mgs_model_quantize(const char *pFnameInp, const char *pFnameOut, const mgs_model_quantize_params *pParams) {
  try {
    fMgsModelQuantizeImpl(pFnameInp, pFnameOut, pParams);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: failed to quantize: %s\n", __func__, vErr.what());
    return 1;
  }

  return 0;
}

//
// Helper functions for external tools exposed in mgs-ext.h
//

quantize_state_impl *mgs_quant_init(const mgs_model *pModel, const mgs_model_quantize_params *pParams) {
  return new quantize_state_impl(*pModel, pParams);
}

void mgs_quant_free(quantize_state_impl *pQs) { delete pQs; }

mgs_model *mgs_quant_model_from_metadata(const mgs_quant_model_desc *pDesc) {
  struct mgs_model_params vMparams = mgs_model_default_params();
  auto vArch = fLlmArchFromString(pDesc->architecture);
  auto *vModel = mgs_model_create(vArch, vMparams);
  vModel->vArch = vArch;

  vModel->vHparams.vNEmbd = pDesc->n_embd;
  vModel->vHparams.vNEmbdHeadKFull = pDesc->n_embd_head_k;
  vModel->vHparams.vNEmbdHeadVFull = pDesc->n_embd_head_v;
  vModel->vHparams.vNLayerAll = pDesc->n_layer;
  GGML_ASSERT(pDesc->n_layer > 0 && pDesc->n_layer <= MGS_MAX_LAYERS);
  vModel->vHparams.vNExpert = pDesc->n_expert;

  for (uint32_t vI = 0; vI < pDesc->n_layer; vI++) {
    vModel->vHparams.aNHeadArr[vI] = pDesc->n_head;
    vModel->vHparams.aNHeadKvArr[vI] = pDesc->n_head_kv;
    vModel->vHparams.aNFfArr[vI] = pDesc->n_ff;
  }

  return vModel;
}

bool mgs_quant_tensor_allows_quantization(const quantize_state_impl *pQs, const ggml_tensor *pTensor) {
  return fTensorAllowsQuantization(pQs->vParams, pQs->vModel.vArch, pTensor);
}

void mgs_quant_compute_types(quantize_state_impl *pQs, mgs_ftype pFtype, ggml_tensor **pTensors,
                             ggml_type *pResultTypes, size_t pNTensors) {
  // reset per-computation state
  pQs->vNAttentionWv = 0;
  pQs->vNFfnDown = 0;
  pQs->vNFfnGate = 0;
  pQs->vNFfnUp = 0;
  pQs->vIAttentionWv = 0;
  pQs->vIFfnDown = 0;
  pQs->vIFfnGate = 0;
  pQs->vIFfnUp = 0;
  pQs->vNFallback = 0;
  pQs->vHasImatrix = false;
  pQs->vHasTiedEmbeddings = true;

  // build metadata from tensor names
  std::vector<tensor_metadata> lMetadata(pNTensors);
  for (size_t vI = 0; vI < pNTensors; vI++) {
    lMetadata[vI].vName = ggml_get_name(pTensors[vI]);
  }

  // initialize counters and categories
  fInitQuantizeStateCounters(*pQs, lMetadata);

  // use a local copy of params with the requested ftype
  mgs_model_quantize_params vLocalParams = *pQs->vParams;
  vLocalParams.ftype = pFtype;

  ggml_type vDefaultType = mgs_ftype_get_default_type(pFtype);

  // compute types
  for (size_t vI = 0; vI < pNTensors; vI++) {
    pResultTypes[vI] = fMgsTensorGetType(*pQs, &vLocalParams, pTensors[vI], vDefaultType, lMetadata[vI]);
  }
}
